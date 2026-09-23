#include <algorithm>
#include <limits>

#include "rasp/formats.hpp"

namespace rasp {

template <class T>
EllHost<T> build_ell(const HostCsr<T>& A, double max_fill) {
  EllHost<T> E;
  E.rows = A.rows;
  E.cols = A.cols;
  for (int32_t r = 0; r < A.rows; ++r)
    E.width = std::max(E.width, A.row_ptr[size_t(r) + 1] - A.row_ptr[size_t(r)]);

  const double stored = double(A.rows) * double(E.width);
  if (stored > max_fill * double(std::max<int64_t>(A.nnz, 1)))
    throw FormatInfeasible("ELL would store " + std::to_string(int64_t(stored)) + " slots for " +
                           std::to_string(A.nnz) + " nonzeros");
  if (stored >= double(std::numeric_limits<int32_t>::max()) || stored * double(sizeof(T) + 4) > kMaxFormatBytes)
    throw FormatInfeasible("ELL exceeds 32-bit offsets or the format size cap");

  E.col.assign(size_t(stored), -1);
  E.val.assign(size_t(stored), T(0));
  for (int32_t r = 0; r < A.rows; ++r)
    for (int32_t k = A.row_ptr[size_t(r)], j = 0; k < A.row_ptr[size_t(r) + 1]; ++k, ++j) {
      const size_t slot = size_t(j) * size_t(A.rows) + size_t(r);
      E.col[slot] = A.col_idx[size_t(k)];
      E.val[slot] = A.vals[size_t(k)];
    }
  return E;
}

template EllHost<float> build_ell(const HostCsr<float>&, double);
template EllHost<double> build_ell(const HostCsr<double>&, double);

}  // namespace rasp
