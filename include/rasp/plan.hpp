#pragma once
// SpmvPlan: the RASP runtime entry point for SpMV.
//
//   auto plan = rasp::SpmvPlan<double>(A, x, y, {.expected_calls = 500});
//   for (...) plan.run(stream);            // y = A x with the selected kernel
//   std::puts(plan.report().to_string().c_str());
//
// Construction decides, in order:
//   1. Is the call budget large enough to pay for feature extraction? (predicted cuSPARSE time x calls x
//      the selector's cross-validated expected gain vs predicted feature cost). If not: cuSPARSE ALG1.
//   2. Extract features on the GPU and ask the selector for the steady-state best configuration.
//   3. If that configuration needs a format conversion, is the conversion repaid within expected_calls?
//      (predicted conversion time vs calls x predicted per-call saving). If not: the best no-conversion choice.
//   4. Build the kernel (GPU conversion if needed).
#include <memory>
#include <string>

#include "rasp/selector.hpp"

namespace rasp {

struct PlanOptions {
  int64_t expected_calls = 100;
  SelectorKind selector = SelectorKind::Tree;
  bool break_even = true;  // false: always use the steady-state choice (ablation)
};

struct PlanReport {
  KernelConfig config;        // what run() uses
  KernelConfig steady_state;  // what the selector would pick for unlimited calls
  bool used_features = false;
  MatrixFeatures features;
  double predicted_speedup = 1.0, predicted_base_ms = 0;
  double feature_ms = 0, convert_ms = 0, plan_ms = 0;  // measured; plan_ms = whole constructor
  std::string reason;
  std::string to_string() const;
};

template <class T>
class SpmvPlan {
 public:
  SpmvPlan(const DeviceCsr<T>& A, const T* x, T* y, const PlanOptions& opt = {}, cudaStream_t stream = nullptr);
  void run(cudaStream_t stream = nullptr) { kernel_->run(stream); }
  const PlanReport& report() const { return report_; }

 private:
  std::unique_ptr<SpmvKernel<T>> kernel_;
  PlanReport report_;
};

}  // namespace rasp
