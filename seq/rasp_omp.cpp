// RASP, OpenMP implementation: rasp_seq.cpp with the hotspots parallelised over the CPU cores (no CUDA).
// Same algorithms, command line, output lines, CSV format and phase names as rasp_seq, so the analysis scripts read
// both. Extra options: --threads N, --schedule static|dynamic[,chunk]|guided[,chunk] (SpMV CSR and SELL), --verify.
//
//   rasp_omp [--matrices a.mtx,b.mtx] [--spmv-reps 50] [--cg laplace:64 | --cg file.mtx] [--cg-max-iter 1000]
//            [--amg 96] [--csv results/omp/run.csv] [--label name] [--threads N] [--schedule s] [--verify]
//
// --verify also runs the sequential kernels (the *_seq functions, copies of rasp_seq's) on the same inputs and
// compares: SpMV (rel. error <= 1e-12), CG (converged, iterations within 2), AMG (identical structure, values 1e-12).
//
// Workloads (each optional):
//   SpMV  for every matrix: read the Matrix Market file, build CSR, extract structural features, convert to ELL and
//         SELL-32-256, then time --spmv-reps SpMVs in each format (choosing the best format by trying them all,
//         which is what RASP replaces with a runtime selector).
//   CG    Jacobi-preconditioned conjugate gradients on an SPD matrix (a SuiteSparse file or Laplace3D of size n^3):
//         one SpMV, two dot products and three vector updates per iteration.
//   AMG   set-up of an aggregation multigrid hierarchy on Laplace3D: the Galerkin triple products
//         A_{l+1} = R_l (A_l P_l) on every level, each SpGEMM as flop estimate, symbolic pass and numeric pass
//         (Gustavson's row-by-row algorithm with a dense accumulator).
//
// Each step is a separate function marked noinline and noclone, so gprof attributes time to it even at -O2 (gprof
// ignores GCC's ".isra"/".constprop" clones and would charge their samples to whichever function precedes them in
// memory). The program also times every phase itself and prints/writes the breakdown.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>
#include <sys/resource.h>

#define NOINLINE __attribute__((noinline, noclone))

namespace {

// ---- phase timers ----------------------------------------------------------------------------
// Wall time, plus user and system CPU time (getrusage): a sampling profiler such as gprof only sees user time, and
// system time here is mostly the kernel zeroing freshly allocated pages (page faults).
struct Phase {
  std::string workload, name;
  double seconds = 0, user = 0, sys = 0;
  long calls = 0;
};

void cpu_times(double& user, double& sys) {
  rusage u{};
  getrusage(RUSAGE_SELF, &u);
  user = double(u.ru_utime.tv_sec) + 1e-6 * double(u.ru_utime.tv_usec);
  sys = double(u.ru_stime.tv_sec) + 1e-6 * double(u.ru_stime.tv_usec);
}
std::vector<Phase> g_phases;
bool g_timing = true;   // off while --verify runs the sequential reference, so it does not show up in the phases
bool g_verify = false, g_failed = false;

void check(bool ok, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
void check(bool ok, const char* fmt, ...) {  // one "verify:" line per check; any failure fails the run
  va_list ap;
  va_start(ap, fmt);
  std::printf("verify: %s ", ok ? "ok    " : "FAILED");
  std::vprintf(fmt, ap);
  std::printf("\n");
  va_end(ap);
  if (!ok) g_failed = true;
}

Phase& phase(const std::string& workload, const std::string& name) {
  for (auto& p : g_phases)
    if (p.workload == workload && p.name == name) return p;
  g_phases.push_back({workload, name, 0, 0, 0, 0});
  return g_phases.back();
}

class Timer {
 public:
  Timer(const std::string& workload, const std::string& name)
      : p_(g_timing ? &phase(workload, name) : nullptr), t0_(std::chrono::steady_clock::now()) {
    cpu_times(u0_, s0_);
  }
  ~Timer() {
    if (!p_) return;
    double u, s;
    cpu_times(u, s);
    p_->seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
    p_->user += u - u0_;
    p_->sys += s - s0_;
    ++p_->calls;
  }

 private:
  Phase* p_;
  std::chrono::steady_clock::time_point t0_;
  double u0_ = 0, s0_ = 0;
};

// ---- matrices ----------------------------------------------------------------------------------
struct Csr {
  int rows = 0, cols = 0;
  std::vector<int> row_ptr, col;
  std::vector<double> val;
  long nnz() const { return long(col.size()); }
};

struct Coo {
  int rows = 0, cols = 0;
  std::vector<int> r, c;
  std::vector<double> v;
};

// Matrix Market: coordinate real/integer/pattern, general/symmetric/skew-symmetric. Symmetric storage is expanded.
NOINLINE Coo read_matrix_market(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  std::fseek(f, 0, SEEK_END);
  const long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::string buf(size_t(size), '\0');
  if (std::fread(&buf[0], 1, size_t(size), f) != size_t(size)) throw std::runtime_error("short read " + path);
  std::fclose(f);

  const char* p = buf.c_str();
  const std::string header(p, std::strchr(p, '\n'));
  const bool pattern = header.find("pattern") != std::string::npos;
  const bool symmetric = header.find("symmetric") != std::string::npos && header.find("skew") == std::string::npos;
  const bool skew = header.find("skew-symmetric") != std::string::npos;
  if (header.find("coordinate") == std::string::npos || header.find("complex") != std::string::npos)
    throw std::runtime_error(path + ": only real/integer/pattern coordinate matrices are supported");
  while (*p == '%') p = std::strchr(p, '\n') + 1;  // header and comments

  char* end = nullptr;
  Coo A;
  A.rows = int(std::strtol(p, &end, 10));
  A.cols = int(std::strtol(end, &end, 10));
  const long entries = std::strtol(end, &end, 10);
  const size_t cap = size_t(entries) * (symmetric || skew ? 2 : 1);
  A.r.reserve(cap);
  A.c.reserve(cap);
  A.v.reserve(cap);
  p = end;
  for (long k = 0; k < entries; ++k) {
    const int i = int(std::strtol(p, &end, 10)) - 1;
    const int j = int(std::strtol(end, &end, 10)) - 1;
    const double x = pattern ? 1.0 : std::strtod(end, &end);
    p = end;
    A.r.push_back(i);
    A.c.push_back(j);
    A.v.push_back(x);
    if ((symmetric || skew) && i != j) {
      A.r.push_back(j);
      A.c.push_back(i);
      A.v.push_back(skew ? -x : x);
    }
  }
  return A;
}

// Counting sort by row, then every row sorted by column with duplicates summed.
NOINLINE Csr coo_to_csr(const Coo& A) {
  Csr C;
  C.rows = A.rows;
  C.cols = A.cols;
  C.row_ptr.assign(size_t(A.rows) + 1, 0);
  for (int i : A.r) ++C.row_ptr[size_t(i) + 1];
  for (int i = 0; i < A.rows; ++i) C.row_ptr[size_t(i) + 1] += C.row_ptr[size_t(i)];
  std::vector<int> next(C.row_ptr.begin(), C.row_ptr.end() - 1);
  std::vector<std::pair<int, double>> e(A.r.size());
  for (size_t k = 0; k < A.r.size(); ++k) e[size_t(next[size_t(A.r[k])]++)] = {A.c[k], A.v[k]};
  C.col.reserve(e.size());
  C.val.reserve(e.size());
  std::vector<int> out_ptr(size_t(A.rows) + 1, 0);
  for (int i = 0; i < A.rows; ++i) {
    auto b = e.begin() + C.row_ptr[size_t(i)], en = e.begin() + C.row_ptr[size_t(i) + 1];
    std::sort(b, en, [](const auto& x, const auto& y) { return x.first < y.first; });
    for (auto it = b; it != en; ++it) {
      if (!C.col.empty() && long(C.col.size()) > out_ptr[size_t(i)] && C.col.back() == it->first) C.val.back() += it->second;
      else {
        C.col.push_back(it->first);
        C.val.push_back(it->second);
      }
    }
    out_ptr[size_t(i) + 1] = int(C.col.size());
  }
  C.row_ptr = std::move(out_ptr);
  return C;
}

// ---- structural features (what RASP's selector looks at) --------------------------------------------
struct Features {
  double mean = 0, cv = 0, empty = 0, ell_fill = 1, sell_fill = 1;
  int max_len = 0;
};

NOINLINE Features compute_features(const Csr& A) {
  Features f;
  double sum = 0, sq = 0;
  long empty = 0, sell_slots = 0;
  for (int s = 0; s < A.rows; s += 32) {
    int slice_max = 0;
    for (int i = s; i < std::min(A.rows, s + 32); ++i) {
      const int len = A.row_ptr[size_t(i) + 1] - A.row_ptr[size_t(i)];
      sum += len;
      sq += double(len) * len;
      empty += len == 0;
      f.max_len = std::max(f.max_len, len);
      slice_max = std::max(slice_max, len);
    }
    sell_slots += 32L * slice_max;
  }
  const double n = std::max(A.rows, 1);
  f.mean = sum / n;
  f.cv = f.mean > 0 ? std::sqrt(std::max(sq / n - f.mean * f.mean, 0.0)) / f.mean : 0;
  f.empty = double(empty) / n;
  f.ell_fill = A.nnz() ? double(A.rows) * f.max_len / double(A.nnz()) : 1;
  f.sell_fill = A.nnz() ? double(sell_slots) / double(A.nnz()) : 1;
  return f;
}

// ---- formats -------------------------------------------------------------------------------------
struct Ell {
  int rows = 0, width = 0;
  std::vector<int> col;     // column-major: slot j of row i at j * rows + i; -1 = padding
  std::vector<double> val;
};

NOINLINE Ell csr_to_ell(const Csr& A, int max_len) {
  Ell E;
  E.rows = A.rows;
  E.width = max_len;
  E.col.assign(size_t(A.rows) * size_t(max_len), -1);
  E.val.assign(E.col.size(), 0.0);
  for (int i = 0; i < A.rows; ++i)
    for (int k = A.row_ptr[size_t(i)], j = 0; k < A.row_ptr[size_t(i) + 1]; ++k, ++j) {
      E.col[size_t(j) * size_t(A.rows) + size_t(i)] = A.col[size_t(k)];
      E.val[size_t(j) * size_t(A.rows) + size_t(i)] = A.val[size_t(k)];
    }
  return E;
}

struct Sell {
  int rows = 0, C = 32;
  std::vector<int> perm, slice_ptr, slice_width, col;
  std::vector<double> val;
};

// SELL-C-sigma (Kreutzer et al. 2014): rows sorted by length inside windows of sigma rows, slices of C padded.
NOINLINE Sell csr_to_sell(const Csr& A, int C, int sigma) {
  Sell S;
  S.rows = A.rows;
  S.C = C;
  S.perm.resize(size_t(A.rows));
  std::iota(S.perm.begin(), S.perm.end(), 0);
  auto len = [&](int i) { return A.row_ptr[size_t(i) + 1] - A.row_ptr[size_t(i)]; };
  for (int w = 0; w < A.rows; w += sigma)
    std::stable_sort(S.perm.begin() + w, S.perm.begin() + std::min(A.rows, w + sigma),
                     [&](int a, int b) { return len(a) > len(b); });
  const int slices = (A.rows + C - 1) / C;
  S.slice_ptr.assign(size_t(slices) + 1, 0);
  S.slice_width.assign(size_t(slices), 0);
  for (int s = 0; s < slices; ++s) {
    int w = 0;
    for (int i = s * C; i < std::min(A.rows, (s + 1) * C); ++i) w = std::max(w, len(S.perm[size_t(i)]));
    S.slice_width[size_t(s)] = w;
    S.slice_ptr[size_t(s) + 1] = S.slice_ptr[size_t(s)] + w * C;
  }
  S.col.assign(size_t(S.slice_ptr.back()), -1);
  S.val.assign(S.col.size(), 0.0);
  for (int i = 0; i < A.rows; ++i) {
    const int r = S.perm[size_t(i)];
    const size_t base = size_t(S.slice_ptr[size_t(i / C)]) + size_t(i % C);
    for (int k = A.row_ptr[size_t(r)], j = 0; k < A.row_ptr[size_t(r) + 1]; ++k, ++j) {
      S.col[base + size_t(j) * size_t(C)] = A.col[size_t(k)];
      S.val[base + size_t(j) * size_t(C)] = A.val[size_t(k)];
    }
  }
  return S;
}

// ---- sequential reference kernels (verbatim from rasp_seq.cpp; only --verify calls them) ----------------------
void spmv_csr_seq(const Csr& A, const double* x, double* y) {
  for (int i = 0; i < A.rows; ++i) {
    double s = 0;
    for (int k = A.row_ptr[size_t(i)]; k < A.row_ptr[size_t(i) + 1]; ++k) s += A.val[size_t(k)] * x[A.col[size_t(k)]];
    y[i] = s;
  }
}

double dot_seq(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0;
  for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}
void axpy_seq(double alpha, const std::vector<double>& x, std::vector<double>& y) {
  for (size_t i = 0; i < x.size(); ++i) y[i] += alpha * x[i];
}
void jacobi_seq(const std::vector<double>& dinv, const std::vector<double>& r, std::vector<double>& z) {
  for (size_t i = 0; i < r.size(); ++i) z[i] = dinv[i] * r[i];
}
void xpby_seq(const std::vector<double>& z, double beta, std::vector<double>& p) {
  for (size_t i = 0; i < z.size(); ++i) p[i] = z[i] + beta * p[i];
}

Csr spgemm_seq(const Csr& A, const Csr& B, long* flops) {
  *flops = 0;
  for (long k = 0; k < A.nnz(); ++k) *flops += B.row_ptr[size_t(A.col[size_t(k)]) + 1] - B.row_ptr[size_t(A.col[size_t(k)])];
  std::vector<int> rp(size_t(A.rows) + 1, 0), marker(size_t(B.cols), -1);
  for (int i = 0; i < A.rows; ++i) {
    int count = 0;
    for (int ka = A.row_ptr[size_t(i)]; ka < A.row_ptr[size_t(i) + 1]; ++ka) {
      const int k = A.col[size_t(ka)];
      for (int kb = B.row_ptr[size_t(k)]; kb < B.row_ptr[size_t(k) + 1]; ++kb)
        if (marker[size_t(B.col[size_t(kb)])] != i) {
          marker[size_t(B.col[size_t(kb)])] = i;
          ++count;
        }
    }
    rp[size_t(i) + 1] = rp[size_t(i)] + count;
  }
  Csr C;
  C.rows = A.rows;
  C.cols = B.cols;
  C.col.resize(size_t(rp.back()));
  C.val.resize(size_t(rp.back()));
  std::vector<double> acc(size_t(B.cols), 0.0);
  std::fill(marker.begin(), marker.end(), -1);
  std::vector<int> touched;
  for (int i = 0; i < A.rows; ++i) {
    touched.clear();
    for (int ka = A.row_ptr[size_t(i)]; ka < A.row_ptr[size_t(i) + 1]; ++ka) {
      const int k = A.col[size_t(ka)];
      const double a = A.val[size_t(ka)];
      for (int kb = B.row_ptr[size_t(k)]; kb < B.row_ptr[size_t(k) + 1]; ++kb) {
        const int j = B.col[size_t(kb)];
        if (marker[size_t(j)] != i) {
          marker[size_t(j)] = i;
          acc[size_t(j)] = 0;
          touched.push_back(j);
        }
        acc[size_t(j)] += a * B.val[size_t(kb)];
      }
    }
    std::sort(touched.begin(), touched.end());
    int o = rp[size_t(i)];
    for (int j : touched) {
      C.col[size_t(o)] = j;
      C.val[size_t(o++)] = acc[size_t(j)];
    }
  }
  C.row_ptr = std::move(rp);
  return C;
}

// ---- SpMV (OpenMP) --------------------------------------------------------------------------------------
// Rows have very different lengths, so the loop schedule is a run-time choice (--schedule); default static.
NOINLINE void spmv_csr(const Csr& A, const double* x, double* y) {
#pragma omp parallel for schedule(runtime)
  for (int i = 0; i < A.rows; ++i) {
    double s = 0;
    for (int k = A.row_ptr[size_t(i)]; k < A.row_ptr[size_t(i) + 1]; ++k) s += A.val[size_t(k)] * x[A.col[size_t(k)]];
    y[i] = s;
  }
}

// One contiguous block of rows per thread, each thread runs the original j-then-i loops over its block (a parallel
// inner loop would fork and join once per ELL column).
NOINLINE void spmv_ell(const Ell& E, const double* x, double* y) {
#pragma omp parallel
  {
    const long nt = omp_get_num_threads(), tid = omp_get_thread_num();
    const int lo = int(E.rows * tid / nt), hi = int(E.rows * (tid + 1) / nt);
    for (int i = lo; i < hi; ++i) y[i] = 0;
    for (int j = 0; j < E.width; ++j) {
      const int* c = E.col.data() + size_t(j) * size_t(E.rows);
      const double* v = E.val.data() + size_t(j) * size_t(E.rows);
      for (int i = lo; i < hi; ++i)
        if (c[i] >= 0) y[i] += v[i] * x[c[i]];
    }
  }
}

// Slices write disjoint y[perm[...]] entries (a permutation), so there are no conflicts.
NOINLINE void spmv_sell(const Sell& S, const double* x, double* y) {
  const int slices = int(S.slice_width.size());
#pragma omp parallel for schedule(runtime)
  for (int s = 0; s < slices; ++s) {
    const int r0 = s * S.C, n = std::min(S.C, S.rows - r0);
    double acc[64] = {};
    for (int j = 0; j < S.slice_width[size_t(s)]; ++j) {
      const size_t base = size_t(S.slice_ptr[size_t(s)]) + size_t(j) * size_t(S.C);
      for (int t = 0; t < n; ++t) {
        const int c = S.col[base + size_t(t)];
        if (c >= 0) acc[t] += S.val[base + size_t(t)] * x[c];
      }
    }
    for (int t = 0; t < n; ++t) y[S.perm[size_t(r0 + t)]] = acc[t];
  }
}

// ---- vector kernels (CG) -------------------------------------------------------------------------------
// The reduction changes the summation order, so CG's iteration count may differ from the sequential one by 1-2.
NOINLINE double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0;
  const long n = long(a.size());
#pragma omp parallel for schedule(static) reduction(+ : s)
  for (long i = 0; i < n; ++i) s += a[size_t(i)] * b[size_t(i)];
  return s;
}
NOINLINE void axpy(double alpha, const std::vector<double>& x, std::vector<double>& y) {
  const long n = long(x.size());
#pragma omp parallel for schedule(static)
  for (long i = 0; i < n; ++i) y[size_t(i)] += alpha * x[size_t(i)];
}
NOINLINE void jacobi(const std::vector<double>& dinv, const std::vector<double>& r, std::vector<double>& z) {
  const long n = long(r.size());
#pragma omp parallel for schedule(static)
  for (long i = 0; i < n; ++i) z[size_t(i)] = dinv[size_t(i)] * r[size_t(i)];
}
NOINLINE void xpby(const std::vector<double>& z, double beta, std::vector<double>& p) {
  const long n = long(z.size());
#pragma omp parallel for schedule(static)
  for (long i = 0; i < n; ++i) p[size_t(i)] = z[size_t(i)] + beta * p[size_t(i)];
}

// ---- generated matrices (paper 1's multigrid workload) -----------------------------------------------------
NOINLINE Csr laplace3d(int n) {
  Csr A;
  A.rows = A.cols = n * n * n;
  A.row_ptr.reserve(size_t(A.rows) + 1);
  A.row_ptr.push_back(0);
  for (int x = 0; x < n; ++x)
    for (int y = 0; y < n; ++y)
      for (int z = 0; z < n; ++z) {
        const int i = (x * n + y) * n + z;
        auto put = [&](int c, double v) {
          A.col.push_back(c);
          A.val.push_back(v);
        };
        if (x > 0) put(i - n * n, -1);
        if (y > 0) put(i - n, -1);
        if (z > 0) put(i - 1, -1);
        put(i, 6);
        if (z < n - 1) put(i + 1, -1);
        if (y < n - 1) put(i + n, -1);
        if (x < n - 1) put(i + n * n, -1);
        A.row_ptr.push_back(int(A.col.size()));
      }
  return A;
}

NOINLINE Csr aggregation(int n) {  // n^3 x (n/2)^3, one 1 per row: the 2x2x2 aggregate of each cell
  const int m = n / 2;
  Csr P;
  P.rows = n * n * n;
  P.cols = m * m * m;
  P.row_ptr.resize(size_t(P.rows) + 1);
  P.col.resize(size_t(P.rows));
  P.val.assign(size_t(P.rows), 1.0);
  for (int x = 0; x < n; ++x)
    for (int y = 0; y < n; ++y)
      for (int z = 0; z < n; ++z) {
        const int i = (x * n + y) * n + z;
        P.row_ptr[size_t(i)] = i;
        P.col[size_t(i)] = ((x / 2) * m + y / 2) * m + z / 2;
      }
  P.row_ptr[size_t(P.rows)] = P.rows;
  return P;
}

NOINLINE Csr transpose(const Csr& A) {
  Csr T;
  T.rows = A.cols;
  T.cols = A.rows;
  T.row_ptr.assign(size_t(A.cols) + 1, 0);
  T.col.resize(A.col.size());
  T.val.resize(A.val.size());
  for (int c : A.col) ++T.row_ptr[size_t(c) + 1];
  for (int i = 0; i < A.cols; ++i) T.row_ptr[size_t(i) + 1] += T.row_ptr[size_t(i)];
  std::vector<int> next(T.row_ptr.begin(), T.row_ptr.end() - 1);
  for (int i = 0; i < A.rows; ++i)
    for (int k = A.row_ptr[size_t(i)]; k < A.row_ptr[size_t(i) + 1]; ++k) {
      const int pos = next[size_t(A.col[size_t(k)])]++;
      T.col[size_t(pos)] = i;
      T.val[size_t(pos)] = A.val[size_t(k)];
    }
  return T;
}

// ---- SpGEMM (Gustavson, row by row) ------------------------------------------------------------------------
// Multiply-adds of C = A B: sum over the nonzeros A(i,k) of nnz(B(k,:)). This is also an upper bound on nnz(C).
NOINLINE long spgemm_flops(const Csr& A, const Csr& B) {
  long f = 0;
  const long nnz = A.nnz();
#pragma omp parallel for schedule(static) reduction(+ : f)
  for (long k = 0; k < nnz; ++k) f += B.row_ptr[size_t(A.col[size_t(k)]) + 1] - B.row_ptr[size_t(A.col[size_t(k)])];
  return f;
}

// Symbolic pass: the exact number of nonzeros of every row of C (a marker array over B's columns, one per thread).
// Each row stores only its count in rp[i+1]; the prefix sum follows the parallel loop.
NOINLINE std::vector<int> spgemm_symbolic(const Csr& A, const Csr& B) {
  std::vector<int> rp(size_t(A.rows) + 1, 0);
#pragma omp parallel
  {
    std::vector<int> marker(size_t(B.cols), -1);
#pragma omp for schedule(dynamic, 256)
    for (int i = 0; i < A.rows; ++i) {
      int count = 0;
      for (int ka = A.row_ptr[size_t(i)]; ka < A.row_ptr[size_t(i) + 1]; ++ka) {
        const int k = A.col[size_t(ka)];
        for (int kb = B.row_ptr[size_t(k)]; kb < B.row_ptr[size_t(k) + 1]; ++kb)
          if (marker[size_t(B.col[size_t(kb)])] != i) {
            marker[size_t(B.col[size_t(kb)])] = i;
            ++count;
          }
      }
      rp[size_t(i) + 1] = count;
    }
  }
  for (int i = 0; i < A.rows; ++i) rp[size_t(i) + 1] += rp[size_t(i)];  // O(rows): sequential is fine
  return rp;
}

// Numeric pass: per-thread dense accumulator over B's columns, the touched columns sorted per row. Row i writes
// only its own slice [rp[i], rp[i+1]) of C, so there are no write conflicts.
NOINLINE Csr spgemm_numeric(const Csr& A, const Csr& B, std::vector<int> rp) {
  Csr C;
  C.rows = A.rows;
  C.cols = B.cols;
  C.col.resize(size_t(rp.back()));
  C.val.resize(size_t(rp.back()));
#pragma omp parallel
  {
    std::vector<double> acc(size_t(B.cols), 0.0);
    std::vector<int> marker(size_t(B.cols), -1), touched;
#pragma omp for schedule(dynamic, 256)
    for (int i = 0; i < A.rows; ++i) {
      touched.clear();
      for (int ka = A.row_ptr[size_t(i)]; ka < A.row_ptr[size_t(i) + 1]; ++ka) {
        const int k = A.col[size_t(ka)];
        const double a = A.val[size_t(ka)];
        for (int kb = B.row_ptr[size_t(k)]; kb < B.row_ptr[size_t(k) + 1]; ++kb) {
          const int j = B.col[size_t(kb)];
          if (marker[size_t(j)] != i) {
            marker[size_t(j)] = i;
            acc[size_t(j)] = 0;
            touched.push_back(j);
          }
          acc[size_t(j)] += a * B.val[size_t(kb)];
        }
      }
      std::sort(touched.begin(), touched.end());
      int o = rp[size_t(i)];
      for (int j : touched) {
        C.col[size_t(o)] = j;
        C.val[size_t(o++)] = acc[size_t(j)];
      }
    }
  }
  C.row_ptr = std::move(rp);
  return C;
}

Csr spgemm(const Csr& A, const Csr& B, const std::string& wl, long* flops) {
  {
    Timer t(wl, "spgemm: flop estimate");
    *flops += spgemm_flops(A, B);
  }
  std::vector<int> rp;
  {
    Timer t(wl, "spgemm: symbolic");
    rp = spgemm_symbolic(A, B);
  }
  Timer t(wl, "spgemm: numeric");
  return spgemm_numeric(A, B, std::move(rp));
}

// ---- workloads -------------------------------------------------------------------------------------------
std::vector<double> random_vector(size_t n, unsigned seed) {
  std::vector<double> v(n);
  uint64_t s = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  for (auto& x : v) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    x = double((s >> 11) & ((1ULL << 52) - 1)) / double(1ULL << 52) * 2 - 1;
  }
  return v;
}

std::string base_name(const std::string& path) {
  const auto slash = path.find_last_of('/');
  std::string s = slash == std::string::npos ? path : path.substr(slash + 1);
  return s.substr(0, s.find('.'));
}

struct SpmvResult {
  std::string matrix, format;
  long nnz = 0;
  double ms = 0, gflops = 0;
};

void workload_spmv(const std::string& path, int reps, std::vector<SpmvResult>& out) {
  const std::string wl = "spmv";
  Coo coo;
  {
    Timer t(wl, "read Matrix Market (parse)");
    coo = read_matrix_market(path);
  }
  Csr A;
  {
    Timer t(wl, "build CSR (sort, merge)");
    A = coo_to_csr(coo);
  }
  coo = Coo{};
  Features f;
  {
    Timer t(wl, "features");
    f = compute_features(A);
  }
  std::printf("%-18s %8d rows %10ld nnz | mean %.1f max %d cv %.2f | ELL fill %.1fx, SELL-32 fill %.2fx\n",
              base_name(path).c_str(), A.rows, A.nnz(), f.mean, f.max_len, f.cv, f.ell_fill, f.sell_fill);
  const auto x = random_vector(size_t(A.cols), 1);
  std::vector<double> y(size_t(A.rows)), ref(size_t(A.rows));
  auto run = [&](const std::string& fmt, auto&& kernel) {
    const auto t0 = std::chrono::steady_clock::now();
    {
      Timer t(wl, "SpMV " + fmt);
      for (int r = 0; r < reps; ++r) kernel();
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    double err = 0, scale = 0;
    for (size_t i = 0; i < y.size(); ++i) {
      err = std::max(err, std::abs(y[i] - ref[i]));
      scale = std::max(scale, std::abs(ref[i]));
    }
    if (err > 1e-10 * std::max(scale, 1.0)) std::printf("  WARNING: %s differs from CSR by %.3g\n", fmt.c_str(), err);
    if (g_verify)
      check(err <= 1e-12 * scale, "spmv %s %s: max|y_omp - y_seq| / max|y_seq| = %.2e", base_name(path).c_str(),
            fmt.c_str(), scale > 0 ? err / scale : err);
    out.push_back({base_name(path), fmt, A.nnz(), ms, 2.0 * double(A.nnz()) / (ms * 1e6)});
    std::printf("  SpMV %-12s %9.3f ms  %6.2f GFLOP/s\n", fmt.c_str(), ms, out.back().gflops);
  };
  spmv_csr_seq(A, x.data(), ref.data());
  run("CSR", [&] { spmv_csr(A, x.data(), y.data()); });
  if (f.ell_fill <= 3.0) {  // the same padding cap as the GPU version
    Ell E;
    {
      Timer t(wl, "convert to ELL");
      E = csr_to_ell(A, f.max_len);
    }
    run("ELL", [&] { spmv_ell(E, x.data(), y.data()); });
  } else {
    std::printf("  ELL skipped (padding %.1fx > 3x)\n", f.ell_fill);
  }
  Sell S;
  {
    Timer t(wl, "convert to SELL-32-256");
    S = csr_to_sell(A, 32, 256);
  }
  run("SELL-32-256", [&] { spmv_sell(S, x.data(), y.data()); });
}

struct CgResult {
  int iters;
  double res;
};

CgResult workload_cg(const std::string& what, int max_iter, double tol, bool ref = false) {
  const std::string wl = "cg";
  const auto spmv = ref ? spmv_csr_seq : spmv_csr;
  const auto dot = ref ? dot_seq : ::dot;
  const auto axpy = ref ? axpy_seq : ::axpy;
  const auto jacobi = ref ? jacobi_seq : ::jacobi;
  const auto xpby = ref ? xpby_seq : ::xpby;
  g_timing = !ref;
  Csr A;
  {
    Timer t(wl, "load / generate matrix");
    if (what.rfind("laplace:", 0) == 0) A = laplace3d(std::atoi(what.c_str() + 8));
    else A = coo_to_csr(read_matrix_market(what));
  }
  const size_t n = size_t(A.rows);
  std::vector<double> b(n), x(n, 0.0), r(n), z(n), p(n), q(n), dinv(n, 1.0);
  {
    Timer t(wl, "set-up (b = A x_true, Jacobi)");
    const auto xt = random_vector(n, 3);
    spmv(A, xt.data(), b.data());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < A.rows; ++i)
      for (int k = A.row_ptr[size_t(i)]; k < A.row_ptr[size_t(i) + 1]; ++k)
        if (A.col[size_t(k)] == i && A.val[size_t(k)] != 0) dinv[size_t(i)] = 1.0 / A.val[size_t(k)];
  }
  const double bnorm = std::sqrt(dot(b, b));
  r = b;
  jacobi(dinv, r, z);
  p = z;
  double rz = dot(r, z), res = 1;
  int it = 0;
  for (it = 1; it <= max_iter; ++it) {
    { Timer t(wl, "SpMV (q = A p)"); spmv(A, p.data(), q.data()); }
    double pq;
    { Timer t(wl, "dot products"); pq = dot(p, q); }
    const double alpha = rz / pq;
    { Timer t(wl, "vector updates (axpy)"); axpy(alpha, p, x); axpy(-alpha, q, r); }
    double rr;
    { Timer t(wl, "dot products"); rr = dot(r, r); }
    res = std::sqrt(rr) / bnorm;
    if (res <= tol) break;
    { Timer t(wl, "preconditioner (Jacobi)"); jacobi(dinv, r, z); }
    double rz_new;
    { Timer t(wl, "dot products"); rz_new = dot(r, z); }
    { Timer t(wl, "vector updates (axpy)"); xpby(z, rz_new / rz, p); }
    rz = rz_new;
  }
  g_timing = true;
  if (!ref)
    std::printf("CG on %s: %d rows, nnz %ld, %d iterations, |r|/|b| = %.2e\n", what.c_str(), A.rows, A.nnz(),
                std::min(it, max_iter), res);
  return {std::min(it, max_iter), res};
}

// The parallel product must have exactly the sequential row_ptr and col; values only differ by rounding (none here:
// every row is still accumulated by one thread in the same order).
void verify_spgemm(const Csr& A, const Csr& B, const Csr& C, int grid, const char* what) {
  long flops = 0;
  const Csr S = spgemm_seq(A, B, &flops);
  const bool same = C.row_ptr == S.row_ptr && C.col == S.col;
  double err = 0, scale = 0;
  if (same)
    for (size_t k = 0; k < S.val.size(); ++k) {
      err = std::max(err, std::abs(C.val[k] - S.val[k]));
      scale = std::max(scale, std::abs(S.val[k]));
    }
  check(same && err <= 1e-12 * scale, "amg grid %d %s: structure %s, max rel diff of values %.2e", grid, what,
        same ? "identical" : "DIFFERENT", scale > 0 ? err / scale : err);
}

void workload_amg(int n, int reps) {
  const std::string wl = "amg";
  Csr A0;
  {
    Timer t(wl, "generate Laplace3D");
    A0 = laplace3d(n);
  }
  long flops = 0;
  for (int rep = 0; rep < reps; ++rep) {  // repetitions of the whole set-up only add profiler samples
  Csr A = A0;
  for (int g = n; g >= 8 && g % 2 == 0; g /= 2) {
    Csr P, R;
    {
      Timer t(wl, "generate P, R = P^T");
      P = aggregation(g);
      R = transpose(P);
    }
    const Csr AP = spgemm(A, P, wl, &flops);
    Csr next = spgemm(R, AP, wl, &flops);
    if (g_verify && rep == 0) {
      verify_spgemm(A, P, AP, g, "A P");
      verify_spgemm(R, AP, next, g, "R (A P)");
    }
    if (rep == 0)
      std::printf("AMG level grid %3d: A %8d rows nnz %9ld -> coarse %7d rows nnz %8ld\n", g, A.rows, A.nnz(),
                  next.rows, next.nnz());
    A = std::move(next);
  }
  }
  std::printf("AMG set-up (x%d): %.3g multiply-adds per set-up\n", reps, double(flops) / reps);
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  for (size_t p = 0; p <= s.size();) {
    const size_t c = s.find(',', p);
    if (c != p) out.push_back(s.substr(p, c - p));
    if (c == std::string::npos) break;
    p = c + 1;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> matrices;
  std::string cg, csv, label = "run";
  int reps = 50, cg_iter = 1000, amg = 0, amg_reps = 1, threads = 0;
  std::string schedule = "static";
  double tol = 1e-8;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", k.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (k == "--matrices") matrices = split(val());
    else if (k == "--spmv-reps") reps = std::atoi(val().c_str());
    else if (k == "--cg") cg = val();
    else if (k == "--cg-max-iter") cg_iter = std::atoi(val().c_str());
    else if (k == "--cg-tol") tol = std::atof(val().c_str());
    else if (k == "--amg") amg = std::atoi(val().c_str());
    else if (k == "--amg-reps") amg_reps = std::atoi(val().c_str());
    else if (k == "--csv") csv = val();
    else if (k == "--label") label = val();
    else if (k == "--threads") threads = std::atoi(val().c_str());
    else if (k == "--schedule") schedule = val();
    else if (k == "--verify") g_verify = true;
    else {
      std::fprintf(stderr, "usage: rasp_omp [--matrices a.mtx,b.mtx] [--spmv-reps N] [--cg laplace:N|file.mtx] "
                           "[--cg-max-iter N] [--cg-tol t] [--amg N] [--amg-reps N] [--csv out.csv] [--label name] "
                           "[--threads N] [--schedule static|dynamic[,chunk]|guided[,chunk]] [--verify]\n");
      return 2;
    }
  }

  if (threads > 0) omp_set_num_threads(threads);
  {
    const auto comma = schedule.find(',');
    const std::string kind = schedule.substr(0, comma);
    const int chunk = comma == std::string::npos ? 0 : std::atoi(schedule.c_str() + comma + 1);
    if (kind == "static") omp_set_schedule(omp_sched_static, chunk);
    else if (kind == "dynamic") omp_set_schedule(omp_sched_dynamic, chunk);
    else if (kind == "guided") omp_set_schedule(omp_sched_guided, chunk);
    else {
      std::fprintf(stderr, "unknown --schedule %s\n", schedule.c_str());
      return 2;
    }
  }
  std::printf("OpenMP: %d threads, SpMV schedule %s\n", omp_get_max_threads(), schedule.c_str());

  const auto t0 = std::chrono::steady_clock::now();
  std::vector<SpmvResult> spmv;
  for (const auto& m : matrices) workload_spmv(m, reps, spmv);
  if (!cg.empty()) {
    const CgResult par = workload_cg(cg, cg_iter, tol);
    if (g_verify) {
      const CgResult seq = workload_cg(cg, cg_iter, tol, true);
      check(par.res <= tol && std::abs(par.iters - seq.iters) <= 2,
            "cg %s: omp %d iterations (|r|/|b| %.2e), seq %d (%.2e)", cg.c_str(), par.iters, par.res, seq.iters, seq.res);
    }
  }
  if (amg > 0) workload_amg(amg, amg_reps);
  const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  std::printf("\n%-6s %-34s %9s %7s %8s %8s %9s\n", "", "phase", "wall s", "share", "user s", "sys s", "calls");
  for (const auto& p : g_phases)
    std::printf("%-6s %-34s %9.3f %6.1f%% %8.3f %8.3f %9ld\n", p.workload.c_str(), p.name.c_str(), p.seconds,
                100 * p.seconds / total, p.user, p.sys, p.calls);
  std::printf("%-6s %-34s %10.3f\n", "", "total", total);

  if (g_verify) std::printf(g_failed ? "verify: FAILED (see above)\n" : "verify: ok\n");

  if (!csv.empty()) {
    FILE* f = std::fopen(csv.c_str(), "w");
    if (!f) {
      std::fprintf(stderr, "cannot write %s\n", csv.c_str());
      return 1;
    }
    std::fprintf(f, "label,kind,workload,name,seconds,share,user,sys,calls,nnz,ms_per_call,gflops\n");
    for (const auto& p : g_phases)
      std::fprintf(f, "%s,phase,%s,\"%s\",%.6f,%.6f,%.6f,%.6f,%ld,,,\n", label.c_str(), p.workload.c_str(),
                   p.name.c_str(), p.seconds, p.seconds / total, p.user, p.sys, p.calls);
    for (const auto& s : spmv)
      std::fprintf(f, "%s,spmv,%s,%s,,,,,%d,%ld,%.6f,%.6f\n", label.c_str(), s.matrix.c_str(), s.format.c_str(),
                   reps, s.nnz, s.ms, s.gflops);
    std::fprintf(f, "%s,total,,,%.6f,1,,,1,,,\n", label.c_str(), total);
    std::fclose(f);
  }
  return g_failed ? 1 : 0;
}
