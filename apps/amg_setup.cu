// Phase 5 end-to-end SpGEMM application: the setup phase of an aggregation AMG hierarchy for Laplace3D, i.e. the
// Galerkin triple products A_{l+1} = R_l (A_l P_l) on every level, through SpgemmContext under a fast-memory budget.
//
//   amg_setup --n 192 --budget-mb 256 [--methods alg4,rasp,zerocopy,incore] [--reps 3] [--out results/phase5/amg.csv]
//
// Levels: n, n/2, ... while the grid is even and >= 8; P_l is 2x2x2 aggregation and R_l = P_l^T (gallery.hpp). Every
// product is timed by its report (estimate + plan + execution: pinned inputs in, pinned output out). Building P_l and
// R_l and allocating the pinned outputs are setup, not timed (reported as setup_ms). Checks on every level: all
// methods give the same A_{l+1} (rel. diff <= 1e-12), and the Galerkin row sums hold, A_{l+1} 1 = R_l (A_l 1)
// (exact for Laplace in fp64: every entry is a small integer).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "rasp/csv.hpp"
#include "rasp/gallery.hpp"
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

struct Args {
  int n = 128, reps = 3;
  double budget_mb = 256;
  std::vector<std::string> methods{"alg4", "rasp", "zerocopy", "incore"};
  std::string out = "results/phase5/amg.csv";
};

[[noreturn]] void usage(const std::string& msg) {
  std::fprintf(stderr, "error: %s\nusage: amg_setup --n N --budget-mb MB [--methods alg4,rasp,zerocopy,incore] "
                       "[--reps N] [--out f.csv]\n", msg.c_str());
  std::exit(2);
}

SpgemmOptions method_options(const std::string& m) {
  SpgemmOptions o;
  if (m == "alg4") o.planner = SpgemmPlanner::Alg4;
  else if (m == "alg4_overlap") { o.planner = SpgemmPlanner::Alg4; o.depth = 2; }
  else if (m == "rasp") {}
  else if (m == "rasp_static") o.adaptive = false;
  else if (m == "zerocopy") o.mode = SpgemmMode::ZeroCopy;
  else if (m == "incore") o.mode = SpgemmMode::InCore;
  else usage("unknown method " + m);
  return o;
}

std::vector<double> row_sums(const HostCsr<double>& A) {
  std::vector<double> s(size_t(A.rows), 0);
  for (int32_t r = 0; r < A.rows; ++r)
    for (int32_t k = A.row_ptr[size_t(r)]; k < A.row_ptr[size_t(r) + 1]; ++k) s[size_t(r)] += A.vals[size_t(k)];
  return s;
}

// max |A_next 1 - R (A 1)| / max |R (A 1)|
double galerkin_rowsum_error(const HostCsr<double>& A, const HostCsr<double>& R, const HostCsr<double>& Anext) {
  const auto s = row_sums(A);
  std::vector<double> t(size_t(R.rows), 0);
  for (int32_t r = 0; r < R.rows; ++r)
    for (int32_t k = R.row_ptr[size_t(r)]; k < R.row_ptr[size_t(r) + 1]; ++k)
      t[size_t(r)] += R.vals[size_t(k)] * s[size_t(R.col_idx[size_t(k)])];
  const auto u = row_sums(Anext);
  double scale = 0, diff = 0;
  for (size_t i = 0; i < t.size(); ++i) {
    scale = std::max(scale, std::abs(t[i]));
    diff = std::max(diff, std::abs(u[i] - t[i]));
  }
  return scale > 0 ? diff / scale : diff;
}

const std::vector<std::string> kCols = {
    "n", "budget_bytes", "method", "level", "grid", "product", "rows", "nnz_a", "nnz_b", "nnz_c", "flops", "mode",
    "parts_ac", "parts_b", "depth", "replans", "splits", "reps", "median_ms", "estimate_ms", "plan_ms", "exec_ms",
    "setup_ms", "gflops", "bytes_h2d", "bytes_d2h", "peak_device_bytes", "rel_diff", "rowsum_err", "check", "status",
    "error"};

}  // namespace

int main(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) usage("missing value for " + k);
      return argv[++i];
    };
    if (k == "--n") args.n = std::stoi(val());
    else if (k == "--budget-mb") args.budget_mb = std::stod(val());
    else if (k == "--methods") args.methods = split(val());
    else if (k == "--reps") args.reps = std::stoi(val());
    else if (k == "--out") args.out = val();
    else usage("unknown option " + k);
  }
  if (args.n < 8 || args.n % 2) usage("--n must be even and >= 8");
  for (auto& m : args.methods) method_options(m);  // validate early

  // The hierarchy's fixed operators (setup).
  struct Level {
    int grid;
    PinnedCsr<double> P, R;
    HostCsr<double> Rh;
  };
  std::vector<Level> levels;
  for (int g = args.n; g >= 8 && g % 2 == 0; g /= 2) {
    auto P = aggregation(g);
    auto R = transpose(P);
    levels.push_back({g, PinnedCsr<double>::from(P), PinnedCsr<double>::from(R), std::move(R)});
  }
  const auto A0h = laplace3d(args.n);
  const auto A0 = PinnedCsr<double>::from(A0h);
  std::printf("AMG setup: Laplace3D n=%d (%d rows, nnz %lld), %zu levels, budget %.0f MiB\n", args.n, A0.rows,
              (long long)A0.nnz, levels.size(), args.budget_mb);

  CsvWriter csv(args.out);
  const size_t budget = size_t(args.budget_mb * double(1 << 20));
  int failures = 0;
  std::vector<HostCsr<double>> reference;  // coarse operators of the first method

  auto emit = [&](std::map<std::string, std::string> v) {
    v["n"] = std::to_string(args.n);
    v["reps"] = std::to_string(args.reps);
    Fields f;
    for (auto& k : kCols) f.emplace_back(k, v.count(k) ? v[k] : "");
    for (auto& kv : environment_stamp()) f.push_back(kv);
    csv.row(f);
  };

  for (const auto& method : args.methods) {
    const bool incore = method == "incore";
    size_t ctx_bytes = budget;
    if (incore) {
      size_t free_b = 0, total_b = 0;
      RASP_CUDA(cudaMemGetInfo(&free_b, &total_b));
      ctx_bytes = size_t(double(free_b) * 0.85);
    }
    std::printf("== %s (context %.0f MiB)\n", method.c_str(), double(ctx_bytes) / double(1 << 20));
    try {
      SpgemmContext ctx(ctx_bytes);
      const auto opt = method_options(method);
      const size_t L = levels.size();
      std::vector<PinnedCsr<double>> AP(L), Anext(L);
      std::vector<std::vector<double>> prod_ms(2 * L), setup_ms(2 * L);
      std::vector<SpgemmReport> rep(2 * L);
      std::vector<double> total_ms;
      for (int r = 0; r < args.reps; ++r) {
        double total = 0;
        const PinnedCsr<double>* Al = &A0;
        for (size_t l = 0; l < L; ++l) {
          ctx.multiply(*Al, levels[l].P, AP[l], opt, &rep[2 * l]);
          ctx.multiply(levels[l].R, AP[l], Anext[l], opt, &rep[2 * l + 1]);
          for (size_t j : {2 * l, 2 * l + 1}) {
            prod_ms[j].push_back(rep[j].total_ms);
            setup_ms[j].push_back(rep[j].setup_ms);
            total += rep[j].total_ms;
          }
          Al = &Anext[l];
        }
        total_ms.push_back(total);
      }
      // Checks on the last repetition's operators.
      bool all_ok = true;
      HostCsr<double> prev = A0h;
      for (size_t l = 0; l < L; ++l) {
        const auto Ah = Anext[l].to_host();
        const double rowsum = galerkin_rowsum_error(prev, levels[l].Rh, Ah);
        double diff = 0;
        if (reference.size() < L) reference.push_back(Ah);
        else diff = csr_max_rel_diff(Ah, reference[l]);
        const bool ok = rowsum <= 1e-12 && diff <= 1e-12;
        all_ok = all_ok && ok;
        if (!ok) ++failures;
        for (int which = 0; which < 2; ++which) {
          const SpgemmReport& rp = rep[2 * l + size_t(which)];
          const auto& times = prod_ms[2 * l + size_t(which)];
          const double med = percentile(times, 50);
          emit({{"budget_bytes", std::to_string(ctx_bytes)}, {"method", method}, {"level", std::to_string(l)},
                {"grid", std::to_string(levels[l].grid)}, {"product", which ? "R*(AP)" : "A*P"},
                {"rows", std::to_string(which ? levels[l].R.rows : prev.rows)},
                {"nnz_a", std::to_string(which ? levels[l].R.nnz : prev.nnz)},
                {"nnz_b", std::to_string(which ? AP[l].nnz : levels[l].P.nnz)}, {"nnz_c", std::to_string(rp.nnz_c)},
                {"flops", std::to_string(rp.flops)}, {"mode", to_string(rp.plan.mode)},
                {"parts_ac", std::to_string(rp.plan.parts_ac())}, {"parts_b", std::to_string(rp.plan.parts_b())},
                {"depth", std::to_string(rp.plan.depth)}, {"replans", std::to_string(rp.replans)},
                {"splits", std::to_string(rp.splits)}, {"median_ms", fmt(med)}, {"estimate_ms", fmt(rp.estimate_ms)},
                {"plan_ms", fmt(rp.plan_ms)}, {"exec_ms", fmt(rp.exec_ms)},
                {"setup_ms", fmt(setup_ms[2 * l + size_t(which)].front())},
                {"gflops", fmt(med > 0 ? 2.0 * double(rp.flops) / (med * 1e6) : 0)},
                {"bytes_h2d", std::to_string(rp.bytes_h2d)}, {"bytes_d2h", std::to_string(rp.bytes_d2h)},
                {"peak_device_bytes", std::to_string(rp.peak_device_bytes)}, {"rel_diff", fmt(diff)},
                {"rowsum_err", fmt(rowsum)}, {"check", ok ? "pass" : "FAIL"}, {"status", "ok"}});
          std::printf("  level %zu (grid %3d) %-7s %8.2f ms  %s\n", l, levels[l].grid, which ? "R*(AP)" : "A*P", med,
                      rp.plan.to_string().c_str());
        }
        prev = Ah;
      }
      emit({{"budget_bytes", std::to_string(ctx_bytes)}, {"method", method}, {"level", "-1"}, {"product", "setup"},
            {"median_ms", fmt(percentile(total_ms, 50))}, {"check", all_ok ? "pass" : "FAIL"}, {"status", "ok"}});
      std::printf("  setup total %.2f ms (median of %d)%s\n", percentile(total_ms, 50), args.reps,
                  all_ok ? "" : "  <-- CHECK FAILED");
    } catch (const BudgetTooSmall& e) {
      emit({{"budget_bytes", std::to_string(ctx_bytes)}, {"method", method}, {"level", "-1"}, {"product", "setup"},
            {"status", "infeasible"}, {"error", e.what()}});
      std::printf("  infeasible: %s\n", e.what());
    } catch (const std::exception& e) {
      cudaGetLastError();
      ++failures;
      emit({{"budget_bytes", std::to_string(ctx_bytes)}, {"method", method}, {"level", "-1"}, {"product", "setup"},
            {"status", "error"}, {"error", e.what()}});
      std::printf("  ERROR %s\n", e.what());
    }
  }
  std::printf("\n%d failures -> %s\n", failures, args.out.c_str());
  return failures == 0 ? 0 : 1;
}
