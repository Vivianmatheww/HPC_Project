// Phase 5 end-to-end SpMV application: Jacobi-preconditioned conjugate gradients, every SpMV through SpmvPlan.
//
//   cg_solver --matrix data/spd/HB/bcsstk17.mtx [--precision fp64] [--tol 1e-8] [--max-iter 5000]
//             [--selector tree] [-v]
//   cg_solver --list results/phase5/spd_list.txt [--laplace 64,128] [--precisions fp64] [--reps 5]
//             [--out results/phase5/cg.csv]
//
// A solver knows an upper bound on how many SpMVs it will make: its iteration cap. That is the expected_calls hint
// SpmvPlan's break-even logic needs (G4), so the selectors compared are
//   always_cusparse   cuSPARSE ALG1 (the baseline)
//   tree              decision tree + break-even with expected_calls = max_iter (what a solver would pass)
//   tree_exact        the same with expected_calls = the real iteration count (the best possible hint)
//   tree_nobe         the steady-state choice regardless of the call count (ablation)
// b = A x_true (x_true random), x0 = 0. Two host synchronisations per iteration (p.q, then r.r and r.z, fused into
// the vector update). The final relative residual is recomputed on the CPU from the downloaded x, never taken from
// the iteration. Matrices whose iteration breaks down (p.Ap <= 0: not SPD in floating point) are reported, not timed.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "rasp/csv.hpp"
#include "rasp/gallery.hpp"
#include "rasp/io.hpp"
#include "rasp/plan.hpp"
#include "rasp/reference.hpp"
#include "rasp/timer.hpp"

using namespace rasp;

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

constexpr int kThreads = 256;
unsigned grid_for(int64_t n) { return unsigned(std::min<int64_t>((n + kThreads - 1) / kThreads, 1024)); }

// Sums v over the block and adds it to *out (one atomic per block).
__device__ void block_add(double v, double* out) {
  __shared__ double warp_sums[kThreads / 32];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x / 32] = v;
  __syncthreads();
  if (threadIdx.x < 32) {
    v = threadIdx.x < kThreads / 32 ? warp_sums[threadIdx.x] : 0.0;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (threadIdx.x == 0) atomicAdd(out, v);
  }
}

template <class T>
__global__ void inv_diag_kernel(int rows, const int* __restrict__ rp, const int* __restrict__ ci,
                                const T* __restrict__ v, T* __restrict__ dinv) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  T d = 0;
  for (int k = rp[r]; k < rp[r + 1]; ++k)
    if (ci[k] == r) d = v[k];
  dinv[r] = d != T(0) ? T(1) / d : T(1);
}

// x = 0, r = b, z = D^-1 r, p = z; acc += (r.r, r.z).
template <class T>
__global__ void init_kernel(int n, const T* __restrict__ b, const T* __restrict__ dinv, T* __restrict__ x,
                            T* __restrict__ r, T* __restrict__ z, T* __restrict__ p, double* __restrict__ acc) {
  double rr = 0, rz = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const T zi = dinv[i] * b[i];
    x[i] = 0;
    r[i] = b[i];
    z[i] = zi;
    p[i] = zi;
    rr += double(b[i]) * double(b[i]);
    rz += double(b[i]) * double(zi);
  }
  block_add(rr, acc);
  block_add(rz, acc + 1);
}

template <class T>
__global__ void dot_kernel(int n, const T* __restrict__ a, const T* __restrict__ b, double* __restrict__ acc) {
  double s = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) s += double(a[i]) * double(b[i]);
  block_add(s, acc);
}

// x += alpha p, r -= alpha q, z = D^-1 r; acc += (r.r, r.z).
template <class T>
__global__ void update_kernel(int n, T alpha, const T* __restrict__ p, const T* __restrict__ q,
                              const T* __restrict__ dinv, T* __restrict__ x, T* __restrict__ r, T* __restrict__ z,
                              double* __restrict__ acc) {
  double rr = 0, rz = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    x[i] += alpha * p[i];
    const T ri = r[i] - alpha * q[i];
    const T zi = dinv[i] * ri;
    r[i] = ri;
    z[i] = zi;
    rr += double(ri) * double(ri);
    rz += double(ri) * double(zi);
  }
  block_add(rr, acc);
  block_add(rz, acc + 1);
}

template <class T>
__global__ void direction_kernel(int n, T beta, const T* __restrict__ z, T* __restrict__ p) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = z[i] + beta * p[i];
}

// Device vectors and scalars of one solve (reused across selectors and repetitions).
template <class T>
struct Workspace {
  DeviceBuffer<T> b, dinv, x, r, z, p, q;
  DeviceBuffer<double> acc;
  double* host = nullptr;  // pinned (2 doubles)
  cudaStream_t s = nullptr;
  explicit Workspace(size_t n) : b(n), dinv(n), x(n), r(n), z(n), p(n), q(n), acc(2) {
    RASP_CUDA(cudaMallocHost(&host, 2 * sizeof(double)));
    RASP_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  }
  ~Workspace() {
    cudaFreeHost(host);
    cudaStreamDestroy(s);
  }
  // Reads the two accumulators (one synchronisation) and clears them for the next kernel.
  void read(double& a0, double& a1) {
    RASP_CUDA(cudaMemcpyAsync(host, acc.data(), 2 * sizeof(double), cudaMemcpyDeviceToHost, s));
    RASP_CUDA(cudaMemsetAsync(acc.data(), 0, 2 * sizeof(double), s));
    RASP_CUDA(cudaStreamSynchronize(s));
    a0 = host[0];
    a1 = host[1];
  }
};

struct Solve {
  int iters = 0;
  bool converged = false, breakdown = false;
  double rel_res = 0;           // from the iteration
  double plan_ms = 0, solve_ms = 0, total_ms = 0;
  PlanReport report;
};

template <class T>
Solve cg(const DeviceCsr<T>& A, Workspace<T>& w, double bnorm, const PlanOptions& opt, double tol, int max_iter,
         bool verbose) {
  Solve out;
  const int n = A.rows();
  const auto t0 = Clock::now();
  SpmvPlan<T> plan(A, w.p.data(), w.q.data(), opt, w.s);
  out.report = plan.report();
  out.plan_ms = ms_since(t0);

  const auto t1 = Clock::now();
  RASP_CUDA(cudaMemsetAsync(w.acc.data(), 0, 2 * sizeof(double), w.s));
  init_kernel<T><<<grid_for(n), kThreads, 0, w.s>>>(n, w.b.data(), w.dinv.data(), w.x.data(), w.r.data(), w.z.data(),
                                                    w.p.data(), w.acc.data());
  double rr = 0, rz = 0, pq = 0, unused = 0;
  w.read(rr, rz);
  out.rel_res = std::sqrt(rr) / bnorm;
  out.converged = out.rel_res <= tol;
  for (int it = 1; it <= max_iter && !out.converged; ++it) {
    plan.run(w.s);  // q = A p
    dot_kernel<T><<<grid_for(n), kThreads, 0, w.s>>>(n, w.p.data(), w.q.data(), w.acc.data());
    w.read(pq, unused);
    if (!(pq > 0)) {
      out.breakdown = true;
      out.iters = it;
      break;
    }
    const double alpha = rz / pq;
    update_kernel<T><<<grid_for(n), kThreads, 0, w.s>>>(n, T(alpha), w.p.data(), w.q.data(), w.dinv.data(), w.x.data(),
                                                        w.r.data(), w.z.data(), w.acc.data());
    double rz_new = 0;
    w.read(rr, rz_new);
    out.iters = it;
    out.rel_res = std::sqrt(rr) / bnorm;
    if (verbose && (it % 100 == 0 || it < 5)) std::printf("    iter %5d  |r|/|b| %.3e\n", it, out.rel_res);
    if (out.rel_res <= tol) {
      out.converged = true;
      break;
    }
    direction_kernel<T><<<grid_for(n), kThreads, 0, w.s>>>(n, T(rz_new / rz), w.z.data(), w.p.data());
    rz = rz_new;
  }
  RASP_CUDA(cudaStreamSynchronize(w.s));
  RASP_CUDA(cudaGetLastError());
  out.solve_ms = ms_since(t1);
  out.total_ms = out.plan_ms + out.solve_ms;
  return out;
}

// ||b - A x|| / ||b|| on the CPU, from the downloaded solution.
template <class T>
double true_residual(const HostCsr<T>& h, const std::vector<T>& x, const std::vector<double>& b) {
  const auto ax = cpu_spmv(h, x);
  double num = 0, den = 0;
  for (size_t i = 0; i < b.size(); ++i) {
    num += (b[i] - ax[i]) * (b[i] - ax[i]);
    den += b[i] * b[i];
  }
  return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

PlanOptions selector_options(const std::string& sel, int64_t calls) {
  PlanOptions o;
  o.expected_calls = calls;
  if (sel == "always_cusparse") o.selector = SelectorKind::AlwaysCusparse;
  else if (sel == "tree" || sel == "tree_exact") o.selector = SelectorKind::Tree;
  else if (sel == "tree_nobe") {
    o.selector = SelectorKind::Tree;
    o.break_even = false;
  } else if (sel == "rules") o.selector = SelectorKind::Rules;
  else throw std::invalid_argument("unknown selector " + sel);
  return o;
}

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
  std::string matrix, list, out = "results/phase5/cg.csv", selector = "tree";
  std::vector<int> laplace;
  std::vector<std::string> precisions{"fp64"};
  std::vector<std::string> selectors{"always_cusparse", "tree", "tree_exact", "tree_nobe"};
  double tol = 0;  // 0: 1e-8 for fp64, 1e-5 for fp32
  int max_iter = 5000, reps = 5;
  bool verbose = false;
};

[[noreturn]] void usage(const std::string& msg) {
  std::fprintf(stderr, "error: %s\nusage: cg_solver --matrix f.mtx [--selector tree] [--precision fp64] [--tol t] "
                       "[--max-iter N] [-v]\n       cg_solver --list files.txt [--laplace 64,128] [--precisions fp64] "
                       "[--selectors a,b] [--reps N] [--out f.csv]\n", msg.c_str());
  std::exit(2);
}

const std::vector<std::string> kCols = {
    "matrix", "path", "rows", "nnz", "precision", "selector", "expected_calls", "config", "steady_state",
    "used_features", "reason", "plan_ms", "feature_ms", "convert_ms", "iters", "converged", "rel_res", "rel_res_true",
    "solve_ms", "total_ms", "total_p25_ms", "total_p75_ms", "spmv_ms", "spmv_share", "tol", "max_iter", "reps",
    "check", "status", "error"};

template <class T>
void bench_matrix(const std::string& name, const std::string& path, const HostCsr<double>& A64, const Args& args,
                  CsvWriter& csv, int& failures) {
  const char* prec = sizeof(T) == 8 ? "fp64" : "fp32";
  const double tol = args.tol > 0 ? args.tol : (sizeof(T) == 8 ? 1e-8 : 1e-5);
  const auto h = convert_values<T>(A64);
  const auto A = DeviceCsr<T>::upload(h);
  const auto xt = random_vector<T>(size_t(h.cols), 7);
  const auto b64 = cpu_spmv(h, xt);
  std::vector<T> b(b64.begin(), b64.end());
  double bnorm = 0;
  for (double v : b64) bnorm += v * v;
  bnorm = std::sqrt(bnorm);
  Workspace<T> w(size_t(h.rows));
  w.b.upload(b);
  inv_diag_kernel<T><<<unsigned((h.rows + kThreads - 1) / kThreads), kThreads>>>(h.rows, A.row_ptr(), A.col_idx(),
                                                                                  A.vals(), w.dinv.data());
  RASP_CUDA(cudaDeviceSynchronize());

  std::map<std::string, std::string> base{{"matrix", name}, {"path", path}, {"rows", std::to_string(h.rows)},
                                          {"nnz", std::to_string(h.nnz)}, {"precision", prec}, {"tol", fmt(tol)},
                                          {"max_iter", std::to_string(args.max_iter)},
                                          {"reps", std::to_string(args.reps)}};
  auto emit = [&](std::map<std::string, std::string> v) {
    v.insert(base.begin(), base.end());
    Fields f;
    for (auto& k : kCols) f.emplace_back(k, v.count(k) ? v[k] : "");
    for (auto& kv : environment_stamp()) f.push_back(kv);
    csv.row(f);
  };

  // A first cuSPARSE solve: warm-up, the iteration count for tree_exact, and a symmetric-positive-definite check.
  const Solve first = cg(A, w, bnorm, selector_options("always_cusparse", args.max_iter), tol, args.max_iter, false);
  if (first.breakdown) {
    emit({{"status", "not_spd"}, {"error", "p.Ap <= 0 at iteration " + std::to_string(first.iters)}});
    std::printf("  %s not SPD in floating point (breakdown at iteration %d), skipped\n", prec, first.iters);
    return;
  }
  const int64_t exact = std::max(first.iters, 1);

  std::vector<std::vector<double>> totals(args.selectors.size());
  std::vector<Solve> last(args.selectors.size());
  std::vector<double> rel_true(args.selectors.size(), 0);
  std::vector<size_t> order(args.selectors.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::mt19937 rng(4242);
  for (int rep = 0; rep < args.reps; ++rep) {
    std::shuffle(order.begin(), order.end(), rng);  // WSL timing has position effects (Phase 3)
    for (size_t si : order) {
      const auto& sel = args.selectors[si];
      const int64_t calls = sel == "tree_exact" ? exact : args.max_iter;
      last[si] = cg(A, w, bnorm, selector_options(sel, calls), tol, args.max_iter, false);
      totals[si].push_back(last[si].total_ms);
      if (rep == args.reps - 1) rel_true[si] = true_residual(h, w.x.download(), b64);  // this solve's x
    }
  }
  for (size_t si = 0; si < args.selectors.size(); ++si) {
    const auto& sel = args.selectors[si];
    const Solve& s = last[si];
    // SpMV cost of the chosen kernel inside the loop (no L2 flush: the solver re-reads A every iteration).
    SpmvPlan<T> plan(A, w.p.data(), w.q.data(), selector_options(sel, sel == "tree_exact" ? exact : args.max_iter), w.s);
    BenchOptions bo;
    bo.flush_l2 = false;
    const double spmv = time_gpu([&](cudaStream_t st) { plan.run(st); }, w.s, bo).median_ms;
    const bool ok = std::isfinite(rel_true[si]) && (!s.converged || rel_true[si] <= 10 * tol);
    if (!ok) ++failures;
    emit({{"selector", sel}, {"expected_calls", std::to_string(sel == "tree_exact" ? exact : args.max_iter)},
          {"config", s.report.config.label()}, {"steady_state", s.report.steady_state.label()},
          {"used_features", s.report.used_features ? "1" : "0"}, {"reason", s.report.reason},
          {"plan_ms", fmt(s.plan_ms)}, {"feature_ms", fmt(s.report.feature_ms)}, {"convert_ms", fmt(s.report.convert_ms)},
          {"iters", std::to_string(s.iters)}, {"converged", s.converged ? "1" : "0"}, {"rel_res", fmt(s.rel_res)},
          {"rel_res_true", fmt(rel_true[si])}, {"solve_ms", fmt(s.solve_ms)}, {"total_ms", fmt(percentile(totals[si], 50))},
          {"total_p25_ms", fmt(percentile(totals[si], 25))}, {"total_p75_ms", fmt(percentile(totals[si], 75))},
          {"spmv_ms", fmt(spmv)}, {"spmv_share", fmt(s.solve_ms > 0 ? s.iters * spmv / s.solve_ms : 0)},
          {"check", ok ? (s.converged ? "pass" : "not_converged") : "FAIL"}, {"status", "ok"}});
    std::printf("  %s %-16s %-22s %5d it %s  total %8.3f ms (plan %.3f)  true res %.1e%s\n", prec, sel.c_str(),
                s.report.config.label().c_str(), s.iters, s.converged ? "conv" : "CAP ", percentile(totals[si], 50),
                s.plan_ms, rel_true[si], ok ? "" : "  <-- FAIL");
  }
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) usage("missing value for " + k);
      return argv[++i];
    };
    if (k == "--matrix") args.matrix = val();
    else if (k == "--list") args.list = val();
    else if (k == "--laplace") for (auto& s : split(val())) args.laplace.push_back(std::stoi(s));
    else if (k == "--precision" || k == "--precisions") args.precisions = split(val());
    else if (k == "--selector") args.selector = val();
    else if (k == "--selectors") args.selectors = split(val());
    else if (k == "--tol") args.tol = std::stod(val());
    else if (k == "--max-iter") args.max_iter = std::stoi(val());
    else if (k == "--reps") args.reps = std::stoi(val());
    else if (k == "--out") args.out = val();
    else if (k == "-v" || k == "--verbose") args.verbose = true;
    else usage("unknown option " + k);
  }

  if (!args.matrix.empty()) {  // one solve, explained
    const auto A64 = read_matrix_market(args.matrix);
    for (auto& p : args.precisions) {
      auto run = [&](auto tag) {
        using T = decltype(tag);
        const double tol = args.tol > 0 ? args.tol : (sizeof(T) == 8 ? 1e-8 : 1e-5);
        const auto h = convert_values<T>(A64);
        const auto A = DeviceCsr<T>::upload(h);
        const auto b64 = cpu_spmv(h, random_vector<T>(size_t(h.cols), 7));
        std::vector<T> b(b64.begin(), b64.end());
        double bnorm = 0;
        for (double v : b64) bnorm += v * v;
        Workspace<T> w(size_t(h.rows));
        w.b.upload(b);
        inv_diag_kernel<T><<<unsigned((h.rows + kThreads - 1) / kThreads), kThreads>>>(h.rows, A.row_ptr(),
                                                                                        A.col_idx(), A.vals(), w.dinv.data());
        const Solve s = cg(A, w, std::sqrt(bnorm), selector_options(args.selector, args.max_iter), tol, args.max_iter,
                           args.verbose);
        std::printf("%s %s: %d x %d, nnz %lld\n  plan: %s\n  %d iterations, %s, |r|/|b| %.3e (true %.3e), total %.3f ms "
                    "(plan %.3f, solve %.3f)\n",
                    args.matrix.c_str(), p.c_str(), h.rows, h.cols, (long long)h.nnz, s.report.to_string().c_str(),
                    s.iters, s.breakdown ? "BREAKDOWN (not SPD)" : s.converged ? "converged" : "hit the cap",
                    s.rel_res, true_residual(h, w.x.download(), b64), s.total_ms, s.plan_ms, s.solve_ms);
      };
      if (p == "fp32") run(float{});
      else run(double{});
    }
    return 0;
  }

  if (args.list.empty() && args.laplace.empty()) usage("--matrix, --list or --laplace is required");
  std::vector<std::string> files;
  if (!args.list.empty()) {
    std::ifstream in(args.list);
    if (!in) usage("cannot read " + args.list);
    for (std::string line; std::getline(in, line);)
      if (!line.empty() && line[0] != '#') files.push_back(line);
  }
  CsvWriter csv(args.out);
  int failures = 0;
  const size_t total = files.size() + args.laplace.size();
  size_t idx = 0;
  auto run_one = [&](const std::string& name, const std::string& path, const HostCsr<double>& A) {
    std::printf("[%zu/%zu] %s  %d x %d  nnz %lld\n", ++idx, total, name.c_str(), A.rows, A.cols, (long long)A.nnz);
    for (auto& p : args.precisions) {
      try {
        if (p == "fp32") bench_matrix<float>(name, path, A, args, csv, failures);
        else bench_matrix<double>(name, path, A, args, csv, failures);
      } catch (const std::exception& e) {
        cudaGetLastError();
        ++failures;
        std::printf("  %s ERROR %s\n", p.c_str(), e.what());
      }
    }
  };
  for (const auto& f : files) {
    try {
      const auto A = read_matrix_market(f);
      const auto slash = f.find_last_of('/');
      const auto dir = f.substr(0, slash);
      const std::string name = dir.substr(dir.find_last_of('/') + 1) + "/" + f.substr(slash + 1, f.size() - slash - 5);
      run_one(name, f, A);
    } catch (const std::exception& e) {
      ++failures;
      std::printf("  %s: ERROR %s\n", f.c_str(), e.what());
    }
  }
  for (int n : args.laplace) run_one("gallery/laplace3d_n" + std::to_string(n), "", laplace3d(n));
  std::printf("\n%d failures -> %s\n", failures, args.out.c_str());
  return failures == 0 ? 0 : 1;
}
