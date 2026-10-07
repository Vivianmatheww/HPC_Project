// Flop count and C size bounds (paper 1's estimator): one pass over A's column indices with random reads of B's row
// pointers, then prefix sums. Host code; with OpenMP every pass is parallel (a two-pass scan for the prefix sums).
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "rasp/spgemm_plan.hpp"

namespace rasp {

template <class T>
void estimate(const CsrView<T>& A, const CsrView<T>& B, ProblemSizes& s) {
  if (A.cols != B.rows) throw std::invalid_argument("estimate: A.cols != B.rows");
  s.a_rows = A.rows;
  s.a_cols = A.cols;
  s.b_cols = B.cols;
  s.value_bytes = sizeof(T);
  s.a_ptr = A.row_ptr;
  s.b_ptr = B.row_ptr;
  const int64_t n = A.rows;
  s.flops_ptr.resize(size_t(n) + 1);
  s.cub_ptr.resize(size_t(n) + 1);
  s.flops_ptr[0] = s.cub_ptr[0] = 0;

  const int32_t* ap = A.row_ptr;
  const int32_t* ac = A.col_idx;
  const int32_t* bp = B.row_ptr;
  const int64_t b_cols = B.cols;
  int64_t* fl = s.flops_ptr.data() + 1;
  int64_t* cu = s.cub_ptr.data() + 1;
  std::vector<int64_t> part_f, part_c;  // per-thread totals for the scan

#pragma omp parallel
  {
#ifdef _OPENMP
    const int t = omp_get_thread_num(), nt = omp_get_num_threads();
#else
    const int t = 0, nt = 1;
#endif
#pragma omp single
    {
      part_f.assign(size_t(nt) + 1, 0);
      part_c.assign(size_t(nt) + 1, 0);
    }
    // Pass 1: local inclusive prefix sums over a contiguous row range per thread.
    const int64_t lo = n * t / nt, hi = n * (t + 1) / nt;
    int64_t sf = 0, sc = 0;
    for (int64_t i = lo; i < hi; ++i) {
      int64_t f = 0;
      for (int32_t k = ap[i]; k < ap[i + 1]; ++k) f += bp[ac[k] + 1] - bp[ac[k]];
      sf += f;
      sc += std::min(f, b_cols);
      fl[i] = sf;
      cu[i] = sc;
    }
    part_f[size_t(t) + 1] = sf;
    part_c[size_t(t) + 1] = sc;
#pragma omp barrier
#pragma omp single
    for (int j = 1; j <= nt; ++j) {
      part_f[size_t(j)] += part_f[size_t(j) - 1];
      part_c[size_t(j)] += part_c[size_t(j) - 1];
    }
    // Pass 2: add the totals of the threads before this one.
    const int64_t of = part_f[size_t(t)], oc = part_c[size_t(t)];
    if (of || oc)
      for (int64_t i = lo; i < hi; ++i) {
        fl[i] += of;
        cu[i] += oc;
      }
  }
}

template <class T>
ProblemSizes estimate(const CsrView<T>& A, const CsrView<T>& B) {
  ProblemSizes s;
  estimate(A, B, s);
  return s;
}

template ProblemSizes estimate(const CsrView<float>&, const CsrView<float>&);
template ProblemSizes estimate(const CsrView<double>&, const CsrView<double>&);
template void estimate(const CsrView<float>&, const CsrView<float>&, ProblemSizes&);
template void estimate(const CsrView<double>&, const CsrView<double>&, ProblemSizes&);

}  // namespace rasp
