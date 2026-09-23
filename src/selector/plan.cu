#include <chrono>
#include <cstdio>

#include "rasp/formats.hpp"
#include "rasp/plan.hpp"

namespace rasp {
namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

FeatureExtractor& extractor() {
  static FeatureExtractor fx;  // scratch buffers reused across plans
  return fx;
}

}  // namespace

std::string PlanReport::to_string() const {
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "config %s (steady state %s) | predicted %.2fx vs cuSPARSE | features %s %.3f ms | convert %.3f ms | "
                "plan %.3f ms | %s",
                config.label().c_str(), steady_state.label().c_str(), predicted_speedup, used_features ? "yes" : "no",
                feature_ms, convert_ms, plan_ms, reason.c_str());
  return buf;
}

template <class T>
SpmvPlan<T>::SpmvPlan(const DeviceCsr<T>& A, const T* x, T* y, const PlanOptions& opt, cudaStream_t stream) {
  const auto t0 = std::chrono::steady_clock::now();
  const bool fp64 = sizeof(T) == 8;
  const double n = double(std::max<int64_t>(opt.expected_calls, 1));
  PlanReport& r = report_;
  r.predicted_base_ms = predict_base_ms(A.rows(), A.cols(), A.nnz(), fp64);

  Choice choice{KernelConfig::parse("cusparse_alg1"), 1.0, "baseline"};
  if (opt.selector != SelectorKind::AlwaysCusparse) {
    // 1. Can feature extraction pay for itself? On average the selector saves expected_gain() of the SpMV time
    //    (cross-validated), so extract features only if that saving over n calls exceeds their predicted cost.
    if (opt.break_even && n * r.predicted_base_ms * expected_gain() < predict_feature_ms(A.rows())) {
      choice.reason = "too few calls to repay feature extraction";
    } else {
      // 2. Steady-state choice.
      const auto tf = std::chrono::steady_clock::now();
      r.features = extractor().extract(A, stream);
      r.feature_ms = ms_since(tf);
      r.used_features = true;
      choice = opt.selector == SelectorKind::Rules ? select_rules(r.features, fp64) : select_tree(r.features, fp64);
      r.steady_state = choice.config;
      // 3. Is its conversion repaid within the expected calls?
      if (opt.break_even && needs_conversion(choice.config)) {
        const double saving = n * r.predicted_base_ms * (1.0 - 1.0 / std::max(choice.predicted_speedup, 1e-9));
        const double cost = predict_convert_ms(choice.config, A.rows(), A.nnz(), fp64);
        if (saving <= cost) {
          choice = opt.selector == SelectorKind::Rules ? select_rules(r.features, fp64, false)
                                                       : select_tree(r.features, fp64, false);
          choice.reason += "; conversion to " + r.steady_state.label() + " not repaid in " +
                           std::to_string(opt.expected_calls) + " calls";
        }
      }
    }
  }
  if (!r.used_features) r.steady_state = choice.config;

  // 4. Build it (GPU conversion if needed). A padded format can still be refused for this matrix (too much
  //    padding); then fall back to the best no-conversion choice.
  try {
    kernel_ = make_spmv_kernel_gpu(choice.config, A, x, y, stream);
  } catch (const FormatInfeasible&) {
    const std::string refused = choice.config.label();
    choice = r.used_features && opt.selector == SelectorKind::Rules ? select_rules(r.features, fp64, false)
             : r.used_features                                        ? select_tree(r.features, fp64, false)
                                                                      : Choice{KernelConfig::parse("cusparse_alg1"), 1.0, ""};
    choice.reason += "; " + refused + " infeasible for this matrix";
    kernel_ = make_spmv_kernel_gpu(choice.config, A, x, y, stream);
  }
  r.config = choice.config;
  r.predicted_speedup = choice.predicted_speedup;
  r.reason = choice.reason;
  r.convert_ms = kernel_->convert_ms();
  RASP_CUDA(cudaStreamSynchronize(stream));
  r.plan_ms = ms_since(t0);
}

template class SpmvPlan<float>;
template class SpmvPlan<double>;

}  // namespace rasp
