// The live demo: what RASP decides, and why, for a circuit matrix and a FEM matrix (SpMV), and for an out-of-core
// multigrid product (SpGEMM).
//
//   rasp_demo [--spmv data/suitesparse/Rajat/rajat23.mtx,data/suitesparse/Schenk_AFE/af_shell1.mtx]
//             [--calls 1,100,10000] [--spgemm-n 160] [--budget-mb 256] [--precision fp64]
//
// SpMV: the features RASP extracts, then the plan for each call budget (break-even at work) and what it costs:
// plan + N calls with the chosen kernel vs N cuSPARSE calls. SpGEMM: Laplace3D A*P under a budget with Alg. 4 (paper 1)
// and RASP, their plans, times and the first tiles of RASP's pipeline.
#include <cstdio>
#include <string>
#include <vector>

#include "rasp/cusparse_ref.hpp"
#include "rasp/gallery.hpp"
#include "rasp/io.hpp"
#include "rasp/plan.hpp"
#include "rasp/reference.hpp"
#include "rasp/spgemm.hpp"
#include "rasp/timer.hpp"

using namespace rasp;

namespace {

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  for (size_t p = 0; p <= s.size();) {
    const size_t c = s.find(',', p);
    out.push_back(s.substr(p, c - p));
    if (c == std::string::npos) break;
    p = c + 1;
  }
  return out;
}

template <class T>
void spmv_demo(const std::string& path, const std::vector<int64_t>& calls) {
  const auto h = convert_values<T>(read_matrix_market(path));
  const auto A = DeviceCsr<T>::upload(h);
  DeviceBuffer<T> x, y(size_t(h.rows));
  x.upload(random_vector<T>(size_t(h.cols)));
  std::printf("\n=== SpMV: %s  (%d x %d, nnz %lld, %s)\n", path.c_str(), h.rows, h.cols, (long long)h.nnz,
              sizeof(T) == 8 ? "fp64" : "fp32");
  FeatureExtractor fx;
  const auto f = fx.extract(A);
  std::printf("features: mean row %.1f, max row %d, CV %.2f, empty %.1f%%, ELL fill %.1fx, SELL-32 fill %.2fx, "
              "nnz in rows > 4096: %.1f%%\n",
              f.mean_len, f.max_len, f.cv_len, 100 * f.empty_frac, f.ell_fill, f.sell32_fill, 100 * f.long_nnz_frac);

  BenchOptions bo;
  CusparseSpmv<T> base(A, SpmvAlg::CsrAlg1, x.data(), y.data());
  const double base_ms = time_gpu([&](cudaStream_t s) { base.run(s); }, nullptr, bo).median_ms;
  std::printf("cuSPARSE ALG1: %.4f ms per call\n", base_ms);
  for (int64_t n : calls) {
    PlanOptions o;
    o.expected_calls = n;
    SpmvPlan<T> plan(A, x.data(), y.data(), o);
    const double run_ms = time_gpu([&](cudaStream_t s) { plan.run(s); }, nullptr, bo).median_ms;
    const auto& r = plan.report();
    const double rasp_total = r.plan_ms + double(n) * run_ms, cus_total = double(n) * base_ms;
    std::printf("\n  expected_calls = %lld\n    %s\n    per call %.4f ms; plan + %lld calls = %.3f ms vs cuSPARSE %.3f ms "
                "-> %.2fx\n",
                (long long)n, r.to_string().c_str(), run_ms, (long long)n, rasp_total, cus_total, cus_total / rasp_total);
  }
}

void spgemm_demo(int n, double budget_mb) {
  const auto A = PinnedCsr<double>::from(laplace3d(n));
  const auto P = PinnedCsr<double>::from(aggregation(n));
  std::printf("\n=== SpGEMM: Laplace3D A*P, n = %d (A: %d rows, nnz %lld), fast-memory budget %.0f MiB\n", n, A.rows,
              (long long)A.nnz, budget_mb);
  SpgemmContext ctx(size_t(budget_mb * double(1 << 20)));
  std::printf("PCIe measured at start-up: H2D %.1f GB/s, D2H %.1f GB/s\n", ctx.calibration().h2d_gbs,
              ctx.calibration().d2h_gbs);
  PinnedCsr<double> C;
  SpgemmReport rep;
  SpgemmOptions alg4;
  alg4.planner = SpgemmPlanner::Alg4;
  double times[2] = {0, 0};
  for (int which = 0; which < 2; ++which) {
    const SpgemmOptions o = which ? SpgemmOptions{} : alg4;
    ctx.multiply(A, P, C, o, &rep);  // warm-up
    ctx.multiply(A, P, C, o, &rep);
    times[which] = rep.total_ms;
    std::printf("\n  %s\n", rep.to_string().c_str());
    if (which) {
      std::printf("  first tiles (ms since start: upload | compute | download):\n");
      int shown = 0;
      for (const auto& c : rep.chunks) {
        if (c.p < 0) continue;
        if (shown++ == 6) break;
        std::printf("    rows %8d-%-8d  %7.2f-%-7.2f | %7.2f-%-7.2f | %7.2f-%-7.2f%s\n", c.a0, c.a1, c.h2d_start,
                    c.h2d_end, c.comp_start, c.comp_end, c.d2h_start, c.d2h_end, c.replan ? "  (re-planned after this)" : "");
      }
      std::printf("  measured: %.2f G multiply-adds/s, nnz(C)/upper bound %.2f, workspace %.1f B/flop\n",
                  rep.measured.flops_per_ms / 1e6, rep.measured.gamma, rep.measured.ws_bytes_per_flop);
    }
  }
  std::printf("\n  RASP vs Alg. 4: %.2fx\n", times[0] / times[1]);
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> spmv{"data/suitesparse/Rajat/rajat23.mtx", "data/suitesparse/Schenk_AFE/af_shell1.mtx"};
  std::vector<int64_t> calls{1, 100, 10000};
  int n = 160;
  double budget_mb = 256;
  std::string precision = "fp64";
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", k.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (k == "--spmv") spmv = split(val());
    else if (k == "--calls") { calls.clear(); for (auto& s : split(val())) calls.push_back(std::stoll(s)); }
    else if (k == "--spgemm-n") n = std::stoi(val());
    else if (k == "--budget-mb") budget_mb = std::stod(val());
    else if (k == "--precision") precision = val();
    else {
      std::fprintf(stderr, "usage: rasp_demo [--spmv a.mtx,b.mtx] [--calls 1,100,10000] [--spgemm-n N] "
                           "[--budget-mb MB] [--precision fp64|fp32]\n");
      return 2;
    }
  }
  for (const auto& path : spmv) {
    if (path.empty()) continue;
    if (precision == "fp32") spmv_demo<float>(path, calls);
    else spmv_demo<double>(path, calls);
  }
  if (n > 0) spgemm_demo(n, budget_mb);
  return 0;
}
