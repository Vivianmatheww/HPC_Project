#include "rasp/selector.hpp"

#include <algorithm>
#include <cstring>

#include "tree_generated.hpp"

namespace rasp {

const char* to_string(SelectorKind k) {
  switch (k) {
    case SelectorKind::AlwaysCusparse: return "always_cusparse";
    case SelectorKind::Rules: return "rules";
    case SelectorKind::Tree: return "tree";
  }
  return "?";
}

bool needs_conversion(const KernelConfig& c) {
  return c.format != Format::Cusparse && c.format != Format::CsrScalar && c.format != Format::CsrVector;
}

// v1: hand-written rules, read off the Phase 2 oracle table (results/spmv_oracle.csv). The thresholds are
// deliberately round numbers; the decision tree (v2) is the tuned version of the same idea.
Choice select_rules(const MatrixFeatures& f, bool fp64, bool allow_conversion) {
  auto pick = [](const char* label, double s, const char* why) { return Choice{KernelConfig::parse(label), s, why}; };
  if (f.nnz >= 1'000'000) return pick("cusparse_alg1", 1.0, "large (>= 1M nnz): cuSPARSE is near-optimal");
  if (f.long_nnz_frac > 0 || f.max_len > 16 * std::max(f.mean_len, 1.0))
    return pick("cusparse_alg2", 1.0, "very long rows: cuSPARSE ALG2 balances them");
  if (f.mean_len <= 3) return pick("csr_vector_v2", 1.1, "short rows: 2 lanes per row");
  if (f.mean_len <= 8) return pick("csr_vector_v4", 1.1, "medium rows: 4 lanes per row");
  if (allow_conversion && f.cv_len > 0.5) return pick("csr_binned_t4", 1.05, "varied row lengths: per-row lanes");
  (void)fp64;
  return pick(f.mean_len <= 24 ? "csr_vector_v8" : "csr_vector_v16", 1.05, "long regular rows: 8-16 lanes per row");
}

Choice select_tree(const MatrixFeatures& f, bool fp64, bool allow_conversion) {
  if (generated::kLargeGate && double(f.nnz) >= generated::kLargeNnz)
    return Choice{KernelConfig::parse("cusparse_alg1"), 1.0, "large matrix: cuSPARSE ALG1 (never-lose gate)"};
  const auto v = feature_vector(f, fp64);
  const auto r = allow_conversion ? generated::tree_full(v.data()) : generated::tree_noconv(v.data());
  if (r.speedup < generated::kMinGain && std::strcmp(r.config, "cusparse_alg1") != 0)
    return Choice{KernelConfig::parse("cusparse_alg1"), 1.0,
                  std::string("decision tree: ") + r.config + " expected only " + std::to_string(r.speedup) +
                      "x, below the minimum gain; keep cuSPARSE"};
  return Choice{KernelConfig::parse(r.config), r.speedup,
                allow_conversion ? "decision tree" : "decision tree (no-conversion candidates)"};
}

double predict_base_ms(int64_t rows, int64_t cols, int64_t nnz, bool fp64) {
  const double vb = fp64 ? 8 : 4;
  const double bytes = double(nnz) * (vb + 4) + double(rows + 1) * 4 + double(cols) * vb + double(rows) * vb;
  return generated::kBaseT0Ms + bytes / generated::kBaseBytesPerMs;
}

double predict_convert_ms(const KernelConfig& c, int64_t rows, int64_t nnz, bool fp64) {
  if (!needs_conversion(c)) return 0;
  const std::string fam = c.family();
  for (const auto& m : generated::kConvModels)
    if (fam == m.family && m.fp64 == int(fp64)) return m.c0 + m.c_nnz * double(nnz) + m.c_rows * double(rows);
  return 1e30;  // unknown family: never worth converting
}

double predict_feature_ms(int64_t rows) { return generated::kFeatureT0Ms + generated::kFeaturePerRowMs * double(rows); }

double max_gain() { return generated::kMaxGain; }

double expected_gain() { return generated::kExpectedGain; }

std::string check_generated_features() {
  const auto names = feature_names();
  if (int(names.size()) != generated::kNumFeatures)
    return "generated tree expects " + std::to_string(generated::kNumFeatures) + " features, library has " +
           std::to_string(names.size());
  for (int i = 0; i < generated::kNumFeatures; ++i)
    if (names[size_t(i)] != generated::kFeatureNames[i])
      return "feature " + std::to_string(i) + ": generated '" + generated::kFeatureNames[i] + "' vs library '" +
             names[size_t(i)] + "'";
  return "";
}

}  // namespace rasp
