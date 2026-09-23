// Phase 3 data collection: GPU features and GPU conversion costs for every matrix.
//
//   bench_features --matrices <dir-or-.mtx> [--out results/phase3/features.csv]
//                  [--convert-out results/phase3/gpu_convert.csv] [--precisions fp32,fp64] [--no-convert]
//
// features.csv: one row per (matrix, precision): every MatrixFeatures field, feature_ms (median of 10 GPU
//   extractions, wall time incl. sync), and the max relative difference to the CPU reference features.
// gpu_convert.csv: one row per (matrix, precision, config): GPU conversion time (median of 3) and a
//   correctness check of the GPU-converted kernel against the CPU SpMV reference.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "rasp/csv.hpp"
#include "rasp/features.hpp"
#include "rasp/formats.hpp"
#include "rasp/io.hpp"
#include "rasp/reference.hpp"
#include "rasp/spmv.hpp"
#include "rasp/timer.hpp"

namespace fs = std::filesystem;
using namespace rasp;

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

double rel_diff(double a, double b) { return std::abs(a - b) / std::max(1e-12, std::max(std::abs(a), std::abs(b))); }

double max_feature_diff(const MatrixFeatures& g, const MatrixFeatures& c) {
  const auto a = feature_vector(g, false), b = feature_vector(c, false);
  double d = 0;
  for (size_t i = 0; i < a.size(); ++i) d = std::max(d, rel_diff(a[i], b[i]));
  return d;
}

template <class T>
void run(const fs::path& path, const std::string& name, const HostCsr<double>& A64, bool convert, FeatureExtractor& fx,
         CsvWriter& fcsv, CsvWriter& ccsv, int& failures) {
  const bool fp64 = sizeof(T) == 8;
  const char* prec = fp64 ? "fp64" : "fp32";
  const HostCsr<T> h = convert_values<T>(A64);
  const auto d = DeviceCsr<T>::upload(h);

  // Features: 2 warm-ups, median of 10.
  MatrixFeatures f;
  std::vector<double> t;
  for (int i = 0; i < 12; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    f = fx.extract(d);
    if (i >= 2) t.push_back(ms_since(t0));
  }
  const double diff = max_feature_diff(f, cpu_features(h));
  Fields row = {{"matrix", name}, {"precision", prec}, {"feature_ms", fmt(percentile(t, 50))},
                {"cpu_diff", fmt(diff)}, {"rows", std::to_string(f.rows)}, {"cols", std::to_string(f.cols)},
                {"nnz", std::to_string(f.nnz)}, {"max_len", std::to_string(f.max_len)}};
  const auto names = feature_names();
  const auto vec = feature_vector(f, fp64);
  for (size_t i = 0; i < names.size(); ++i) row.emplace_back(names[i], fmt(vec[i]));
  for (auto& kv : environment_stamp()) row.push_back(kv);
  fcsv.row(row);
  if (diff > 1e-9) {
    ++failures;
    std::printf("    %s features differ from CPU reference: %.3g\n", prec, diff);
  }
  if (!convert) return;

  const auto x = random_vector<T>(size_t(h.cols));
  DeviceBuffer<T> dx, dy(size_t(h.rows));
  dx.upload(x);
  const auto ref = cpu_spmv(h, x);
  const double tol = fp64 ? 1e-12 : 1e-5;
  for (const auto& cfg : all_configs()) {
    Fields r = {{"matrix", name}, {"precision", prec}, {"kernel", cfg.family()}, {"config", cfg.label()},
                {"rows", std::to_string(h.rows)}, {"nnz", std::to_string(h.nnz)}};
    try {
      std::vector<double> ct;
      std::unique_ptr<SpmvKernel<T>> k;
      for (int i = 0; i < 3; ++i) {
        k.reset();
        k = make_spmv_kernel_gpu(cfg, d, dx.data(), dy.data());
        ct.push_back(k->convert_ms());
      }
      RASP_CUDA(cudaMemset(dy.data(), 0xff, dy.bytes()));
      k->run(nullptr);
      RASP_CUDA(cudaDeviceSynchronize());
      const double err = rel_l2_error(dy.download(), ref);
      r.insert(r.end(), {{"convert_gpu_ms", fmt(percentile(ct, 50))}, {"stored_slots", std::to_string(k->stored_slots())},
                         {"rel_err", fmt(err)}, {"check", err <= tol ? "pass" : "FAIL"}, {"status", "ok"}});
      if (err > tol) {
        ++failures;
        std::printf("    %s %s GPU-converted kernel FAILED: err %.3g\n", prec, cfg.label().c_str(), err);
      }
    } catch (const FormatInfeasible&) {
      r.insert(r.end(), {{"convert_gpu_ms", ""}, {"stored_slots", ""}, {"rel_err", ""}, {"check", ""}, {"status", "infeasible"}});
    }
    ccsv.row(r);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string matrices, out = "results/phase3/features.csv", cout_path = "results/phase3/gpu_convert.csv";
  std::vector<std::string> precisions{"fp32", "fp64"};
  bool convert = true;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&] { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (k == "--matrices") matrices = val();
    else if (k == "--out") out = val();
    else if (k == "--convert-out") cout_path = val();
    else if (k == "--no-convert") convert = false;
    else if (k == "--precisions") {
      precisions.clear();
      std::string v = val();
      for (size_t p = 0; p <= v.size();) {
        const size_t c = v.find(',', p);
        precisions.push_back(v.substr(p, c - p));
        if (c == std::string::npos) break;
        p = c + 1;
      }
    } else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      return 2;
    }
  }
  if (matrices.empty()) {
    std::fprintf(stderr, "usage: bench_features --matrices <dir|file.mtx> [--out f.csv] [--convert-out f.csv] [--no-convert]\n");
    return 2;
  }
  std::vector<fs::path> files;
  if (fs::is_regular_file(matrices)) files.emplace_back(matrices);
  else
    for (auto& e : fs::recursive_directory_iterator(matrices))
      if (e.is_regular_file() && e.path().extension() == ".mtx") files.push_back(e.path());
  std::sort(files.begin(), files.end());

  CsvWriter fcsv(out), ccsv(cout_path);
  FeatureExtractor fx;
  int failures = 0;
  for (size_t i = 0; i < files.size(); ++i) {
    const std::string name = files[i].parent_path().filename().string() + "/" + files[i].stem().string();
    std::printf("[%zu/%zu] %s\n", i + 1, files.size(), name.c_str());
    const auto A = read_matrix_market(files[i].string());
    for (auto& p : precisions) {
      if (p == "fp64") run<double>(files[i], name, A, convert, fx, fcsv, ccsv, failures);
      else run<float>(files[i], name, A, convert, fx, fcsv, ccsv, failures);
    }
  }
  std::printf("\n%zu matrices, %d failures -> %s, %s\n", files.size(), failures, out.c_str(), cout_path.c_str());
  return failures ? 1 : 0;
}
