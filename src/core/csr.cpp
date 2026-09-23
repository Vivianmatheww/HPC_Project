#include "rasp/csr.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace rasp {

template <class T>
void HostCsr<T>::validate() const {
  auto bad = [](const std::string& m) { throw std::runtime_error("invalid CSR: " + m); };
  if (rows < 0 || cols < 0 || nnz < 0) bad("negative size");
  if (nnz >= std::numeric_limits<int32_t>::max()) bad("nnz does not fit 32-bit indices");
  if (row_ptr.size() != size_t(rows) + 1) bad("row_ptr has wrong length");
  if (col_idx.size() != size_t(nnz) || vals.size() != size_t(nnz)) bad("col_idx/vals length != nnz");
  if (row_ptr[0] != 0 || row_ptr[size_t(rows)] != nnz) bad("row_ptr must start at 0 and end at nnz");
  for (int32_t r = 0; r < rows; ++r) {
    const int32_t s = row_ptr[size_t(r)], e = row_ptr[size_t(r) + 1];
    if (e < s) bad("row_ptr decreases at row " + std::to_string(r));
    for (int32_t k = s; k < e; ++k) {
      const int32_t c = col_idx[size_t(k)];
      if (c < 0 || c >= cols) bad("column out of range in row " + std::to_string(r));
      if (k > s && c <= col_idx[size_t(k) - 1]) bad("columns not strictly increasing in row " + std::to_string(r));
    }
  }
}

template struct HostCsr<float>;
template struct HostCsr<double>;

}  // namespace rasp
