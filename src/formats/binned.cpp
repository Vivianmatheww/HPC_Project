#include <algorithm>

#include "rasp/formats.hpp"

namespace rasp {

int binned_bin(int32_t len, int elems_per_lane) {
  if (len > kLongRow) return kBins - 1;
  const int lanes = (len + elems_per_lane - 1) / elems_per_lane;
  int bin = 0;
  while (bin < kBins - 2 && (1 << bin) < lanes) ++bin;  // smallest power of two >= lanes, capped at 32
  return bin;
}

template <class T>
BinnedHost build_binned(const HostCsr<T>& A, int elems_per_lane) {
  if (elems_per_lane < 1) throw std::invalid_argument("build_binned: elems_per_lane must be >= 1");
  BinnedHost H;
  H.rows = A.rows;
  H.elems_per_lane = elems_per_lane;
  std::vector<int> bin(size_t(A.rows));
  H.bin_row_start.assign(kBins + 1, 0);
  for (int32_t r = 0; r < A.rows; ++r) {
    bin[size_t(r)] = binned_bin(A.row_ptr[size_t(r) + 1] - A.row_ptr[size_t(r)], elems_per_lane);
    ++H.bin_row_start[size_t(bin[size_t(r)]) + 1];
  }
  for (int k = 0; k < kBins; ++k) H.bin_row_start[size_t(k) + 1] += H.bin_row_start[size_t(k)];

  H.order.resize(size_t(A.rows));
  std::vector<int32_t> next(H.bin_row_start.begin(), H.bin_row_start.end() - 1);
  for (int32_t r = 0; r < A.rows; ++r) H.order[size_t(next[size_t(bin[size_t(r)])]++)] = r;  // stable counting sort

  H.bin_block_start.assign(kBins + 1, 0);
  for (int k = 0; k < kBins; ++k) {
    const int32_t n = H.bin_row_start[size_t(k) + 1] - H.bin_row_start[size_t(k)];
    const int32_t rows_per_block = k == kBins - 1 ? 1 : kBinnedBlock >> k;  // 256 / v
    H.bin_block_start[size_t(k) + 1] = H.bin_block_start[size_t(k)] + (n + rows_per_block - 1) / rows_per_block;
  }
  H.blocks = H.bin_block_start[kBins];
  return H;
}

template BinnedHost build_binned(const HostCsr<float>&, int);
template BinnedHost build_binned(const HostCsr<double>&, int);

}  // namespace rasp
