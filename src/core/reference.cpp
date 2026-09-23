#include "rasp/reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace rasp {

template <class T>
std::vector<double> cpu_spmv(const HostCsr<T>& A, const std::vector<T>& x) {
  std::vector<double> y(size_t(A.rows), 0.0);
  for (int32_t r = 0; r < A.rows; ++r) {
    double s = 0;
    for (int32_t k = A.row_ptr[size_t(r)]; k < A.row_ptr[size_t(r) + 1]; ++k)
      s += double(A.vals[size_t(k)]) * double(x[size_t(A.col_idx[size_t(k)])]);
    y[size_t(r)] = s;
  }
  return y;
}

template <class T>
double rel_l2_error(const std::vector<T>& y, const std::vector<double>& ref) {
  if (y.size() != ref.size()) return std::numeric_limits<double>::infinity();
  double num = 0, den = 0;
  for (size_t i = 0; i < y.size(); ++i) {
    const double d = double(y[i]) - ref[i];
    num += d * d;
    den += ref[i] * ref[i];
  }
  return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

template <class T>
HostCsr<T> cpu_spgemm(const HostCsr<T>& A, const HostCsr<T>& B) {
  HostCsr<T> C;
  C.rows = A.rows;
  C.cols = B.cols;
  C.row_ptr.assign(size_t(A.rows) + 1, 0);
  std::vector<double> acc(size_t(B.cols), 0.0);
  std::vector<int32_t> marker(size_t(B.cols), -1), cols_in_row;
  for (int32_t i = 0; i < A.rows; ++i) {
    cols_in_row.clear();
    for (int32_t ka = A.row_ptr[size_t(i)]; ka < A.row_ptr[size_t(i) + 1]; ++ka) {
      const int32_t k = A.col_idx[size_t(ka)];
      const double a = double(A.vals[size_t(ka)]);
      for (int32_t kb = B.row_ptr[size_t(k)]; kb < B.row_ptr[size_t(k) + 1]; ++kb) {
        const int32_t j = B.col_idx[size_t(kb)];
        if (marker[size_t(j)] != i) {  // structural nonzero, even if the values later cancel
          marker[size_t(j)] = i;
          acc[size_t(j)] = 0;
          cols_in_row.push_back(j);
        }
        acc[size_t(j)] += a * double(B.vals[size_t(kb)]);
      }
    }
    std::sort(cols_in_row.begin(), cols_in_row.end());
    for (int32_t j : cols_in_row) {
      C.col_idx.push_back(j);
      C.vals.push_back(T(acc[size_t(j)]));
    }
    C.row_ptr[size_t(i) + 1] = int32_t(C.col_idx.size());
  }
  C.nnz = int64_t(C.col_idx.size());
  return C;
}

template <class T>
double csr_max_rel_diff(const HostCsr<T>& A, const HostCsr<T>& B) {
  if (A.rows != B.rows || A.cols != B.cols || A.row_ptr != B.row_ptr || A.col_idx != B.col_idx)
    return std::numeric_limits<double>::infinity();
  double scale = 0, diff = 0;
  for (size_t k = 0; k < B.vals.size(); ++k) {
    scale = std::max(scale, std::abs(double(B.vals[k])));
    diff = std::max(diff, std::abs(double(A.vals[k]) - double(B.vals[k])));
  }
  return scale > 0 ? diff / scale : diff;
}

template <class T>
std::vector<T> random_vector(size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<T> v(n);
  for (auto& x : v) x = T(dist(rng));
  return v;
}

#define RASP_INSTANTIATE(T)                                                       \
  template std::vector<double> cpu_spmv(const HostCsr<T>&, const std::vector<T>&); \
  template double rel_l2_error(const std::vector<T>&, const std::vector<double>&); \
  template HostCsr<T> cpu_spgemm(const HostCsr<T>&, const HostCsr<T>&);            \
  template double csr_max_rel_diff(const HostCsr<T>&, const HostCsr<T>&);          \
  template std::vector<T> random_vector<T>(size_t, unsigned);
RASP_INSTANTIATE(float)
RASP_INSTANTIATE(double)
#undef RASP_INSTANTIATE

}  // namespace rasp
