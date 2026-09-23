#pragma once
// Matrix Market I/O with a binary cache.
//
// read_matrix_market() accepts `matrix coordinate {real|integer|pattern} {general|symmetric|skew-symmetric}`.
// It expands symmetric storage, sums duplicates and sorts columns, and returns 0-based CSR.
// With CacheMode::Use it loads `<file>.csr.bin` if that cache is newer than the .mtx, and writes the
// cache after parsing otherwise. Text parsing of large matrices takes seconds to minutes; the cache
// loads at disk speed.
#include <string>

#include "rasp/csr.hpp"

namespace rasp {

enum class CacheMode { Use, Off };

HostCsr<double> read_matrix_market(const std::string& path, CacheMode cache = CacheMode::Use);
void write_matrix_market(const std::string& path, const HostCsr<double>& A);

// Raw binary CSR (the cache format). Exposed for tests and tools.
void save_csr_binary(const std::string& path, const HostCsr<double>& A);
HostCsr<double> load_csr_binary(const std::string& path);
std::string cache_path_for(const std::string& mtx_path);

template <class T>
HostCsr<T> convert_values(const HostCsr<double>& A) {
  HostCsr<T> B;
  B.rows = A.rows;
  B.cols = A.cols;
  B.nnz = A.nnz;
  B.row_ptr = A.row_ptr;
  B.col_idx = A.col_idx;
  B.vals.assign(A.vals.begin(), A.vals.end());
  return B;
}

}  // namespace rasp
