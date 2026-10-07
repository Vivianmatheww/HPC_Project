#include "rasp/gallery.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace rasp {

HostCsr<double> laplace3d(int n) {
  if (n < 1) throw std::invalid_argument("laplace3d: n must be >= 1");
  const int64_t rows = int64_t(n) * n * n;
  const int64_t nnz = 7 * rows - 6 * int64_t(n) * n;
  if (rows > std::numeric_limits<int32_t>::max() || nnz >= std::numeric_limits<int32_t>::max())
    throw std::invalid_argument("laplace3d: n = " + std::to_string(n) + " exceeds 32-bit indices");

  HostCsr<double> A;
  A.rows = A.cols = int32_t(rows);
  A.nnz = nnz;
  A.row_ptr.resize(size_t(rows) + 1);
  A.col_idx.resize(size_t(nnz));
  A.vals.resize(size_t(nnz));
  const int32_t nn = n * n;
  int64_t k = 0;
  for (int x = 0; x < n; ++x)
    for (int y = 0; y < n; ++y)
      for (int z = 0; z < n; ++z) {
        const int32_t i = x * nn + y * n + z;
        A.row_ptr[size_t(i)] = int32_t(k);
        auto put = [&](int32_t c, double v) {
          A.col_idx[size_t(k)] = c;
          A.vals[size_t(k)] = v;
          ++k;
        };
        // Columns in increasing order: x-1, y-1, z-1, self, z+1, y+1, x+1.
        if (x > 0) put(i - nn, -1);
        if (y > 0) put(i - n, -1);
        if (z > 0) put(i - 1, -1);
        put(i, 6);
        if (z < n - 1) put(i + 1, -1);
        if (y < n - 1) put(i + n, -1);
        if (x < n - 1) put(i + nn, -1);
      }
  A.row_ptr[size_t(rows)] = int32_t(k);
  return A;
}

HostCsr<double> aggregation(int n) {
  if (n < 2 || n % 2) throw std::invalid_argument("aggregation: n must be even and >= 2");
  const int64_t rows = int64_t(n) * n * n;
  if (rows > std::numeric_limits<int32_t>::max()) throw std::invalid_argument("aggregation: n too large");
  const int m = n / 2;
  HostCsr<double> P;
  P.rows = int32_t(rows);
  P.cols = m * m * m;
  P.nnz = rows;
  P.row_ptr.resize(size_t(rows) + 1);
  P.col_idx.resize(size_t(rows));
  P.vals.assign(size_t(rows), 1.0);
  for (int x = 0; x < n; ++x)
    for (int y = 0; y < n; ++y)
      for (int z = 0; z < n; ++z) {
        const size_t i = size_t(x) * n * n + size_t(y) * n + size_t(z);
        P.row_ptr[i] = int32_t(i);
        P.col_idx[i] = (x / 2) * m * m + (y / 2) * m + z / 2;
      }
  P.row_ptr[size_t(rows)] = int32_t(rows);
  return P;
}

template <class T>
HostCsr<T> transpose(const HostCsr<T>& A) {
  HostCsr<T> B;
  B.rows = A.cols;
  B.cols = A.rows;
  B.nnz = A.nnz;
  B.row_ptr.assign(size_t(A.cols) + 1, 0);
  B.col_idx.resize(size_t(A.nnz));
  B.vals.resize(size_t(A.nnz));
  for (int64_t k = 0; k < A.nnz; ++k) ++B.row_ptr[size_t(A.col_idx[size_t(k)]) + 1];
  for (int32_t c = 0; c < A.cols; ++c) B.row_ptr[size_t(c) + 1] += B.row_ptr[size_t(c)];
  std::vector<int32_t> next(B.row_ptr.begin(), B.row_ptr.end() - 1);
  for (int32_t r = 0; r < A.rows; ++r)  // rows in increasing order, so every transposed row comes out sorted
    for (int32_t k = A.row_ptr[size_t(r)]; k < A.row_ptr[size_t(r) + 1]; ++k) {
      const int32_t pos = next[size_t(A.col_idx[size_t(k)])]++;
      B.col_idx[size_t(pos)] = r;
      B.vals[size_t(pos)] = A.vals[size_t(k)];
    }
  return B;
}

template HostCsr<float> transpose(const HostCsr<float>&);
template HostCsr<double> transpose(const HostCsr<double>&);

}  // namespace rasp
