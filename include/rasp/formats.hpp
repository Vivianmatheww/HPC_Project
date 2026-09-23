#pragma once
// Host-side storage formats for SpMV and their CSR converters (Phase 2: conversion runs on the CPU;
// its cost is recorded by the benchmark as convert_ms).
//
// Padding convention in every padded format: column index -1, value 0. Padding always sits at the end of
// a row's (or chunk's) slots, so kernels stop at the first -1.
#include <stdexcept>
#include <string>
#include <vector>

#include "rasp/csr.hpp"

#if defined(__CUDACC__)
#define RASP_HD __host__ __device__
#else
#define RASP_HD
#endif

namespace rasp {

// Thrown when a format would need too much padding / memory for this matrix. Benchmarks record it
// as "infeasible", not as an error.
struct FormatInfeasible : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Every converter also refuses formats larger than this (index + value bytes), whatever the padding ratio.
constexpr double kMaxFormatBytes = 2e9;

// ELLPACK: every row padded to the longest row, stored column-major (slot j of row r at j*rows + r).
template <class T>
struct EllHost {
  int32_t rows = 0, cols = 0, width = 0;
  std::vector<int32_t> col;
  std::vector<T> val;
};
// Refuses if rows * width > max_fill * nnz.
template <class T>
EllHost<T> build_ell(const HostCsr<T>& A, double max_fill = 3.0);

// SELL-C-sigma (Kreutzer et al., SISC 2014): rows sorted by decreasing length inside windows of sigma rows
// (sigma = 1: no sorting, sigma = 0: one window over all rows); every slice of C consecutive (sorted) rows is
// padded to its longest row and stored column-major within the slice.
template <class T>
struct SellHost {
  int32_t rows = 0, cols = 0, C = 0, sigma = 0, slices = 0;
  std::vector<int32_t> slice_ptr;    // slices + 1 entries: first stored element of each slice
  std::vector<int32_t> slice_width;  // slices entries
  std::vector<int32_t> perm;         // sorted position -> original row (rows entries)
  std::vector<int32_t> col;
  std::vector<T> val;
};
template <class T>
SellHost<T> build_sell(const HostCsr<T>& A, int C, int sigma, double max_fill = 64.0);

// Adaptive Row-grouped CSR (Heller & Oberhuber 2012, arXiv:1203.5737).
// Rows are packed greedily into groups of at most `block` rows and ~desired_chunk*block nonzeros. Inside a
// group each row starts with one thread; spare threads go one at a time to the row with the largest chunk
// (ceil(len/threads)) until that no longer shrinks it. Every chunk in a group is padded to the group's
// chunk size and chunks are stored interleaved (element i of thread t at offset + i*block + t).
struct ArgGroup {
  int32_t first_row, size, offset, chunk;  // same layout as the paper's argcsrGroupInfo
};
template <class T>
struct ArgCsrHost {
  int32_t rows = 0, cols = 0, block = 0, desired_chunk = 0;
  std::vector<ArgGroup> groups;
  std::vector<int32_t> threads_end;  // per row: inclusive prefix sum of threads within its group
  std::vector<int32_t> col;
  std::vector<T> val;
};
template <class T>
ArgCsrHost<T> build_argcsr(const HostCsr<T>& A, int block, int desired_chunk, double max_fill = 64.0);

// The part of ArgCSR conversion that needs only row lengths (shared by the host and the GPU converter):
// groups, threads per row, and for every row its group's offset, its first thread and its chunk filling.
struct ArgCsrLayout {
  std::vector<ArgGroup> groups;
  std::vector<int32_t> threads_end, row_offset, row_tbegin, row_fill;
  int64_t stored = 0;
};
ArgCsrLayout argcsr_layout(const std::vector<int32_t>& row_ptr, int block, int desired_chunk, size_t value_bytes,
                           double max_fill = 64.0);

// ArgCSR-local (RASP extension): desiredChunkSize chosen per window of `window` rows instead of globally.
// Phase 2 data showed the best global D keeps ~20-75 rows per group (median 41) and grows with the mean row
// length, so each window gets D = pow2_round(target_rows * mean_len / block), clamped to [1, 32], halved when
// the window is very irregular (row-length CV > 1). The kernel is unchanged: groups already carry their own chunk.
int argcsr_local_chunk(double mean_len, double cv, int block, int target_rows);
template <class T>
ArgCsrHost<T> build_argcsr_local(const HostCsr<T>& A, int block, int target_rows, int window = 4096,
                                 double max_fill = 64.0);
ArgCsrLayout argcsr_local_layout(const std::vector<int32_t>& row_ptr, int block, int target_rows, int window,
                                 size_t value_bytes, double max_fill = 64.0);

// CSR-binned (RASP extension): CSR-vector with the number of lanes chosen per row instead of per matrix.
// Row r gets v = pow2ceil(ceil(len / elems_per_lane)) lanes (1..32); rows longer than kLongRow get a whole
// 256-thread block. Rows are grouped by bin (stable, so row order is kept inside a bin) and one kernel launch
// covers every bin. No padding: only the row order is stored.
constexpr int kBinnedBlock = 256;
constexpr int kLongRow = 4096;
constexpr int kBins = 7;  // v = 1, 2, 4, 8, 16, 32, block-per-row
struct BinnedHost {
  int32_t rows = 0, blocks = 0, elems_per_lane = 0;
  std::vector<int32_t> order;             // rows grouped by bin
  std::vector<int32_t> bin_row_start;     // kBins + 1 entries into `order`
  std::vector<int32_t> bin_block_start;   // kBins + 1 entries: first thread block of each bin
};
RASP_HD inline int binned_bin(int32_t len, int elems_per_lane) {
  if (len > kLongRow) return kBins - 1;
  const int lanes = (len + elems_per_lane - 1) / elems_per_lane;
  int bin = 0;
  while (bin < kBins - 2 && (1 << bin) < lanes) ++bin;  // smallest power of two >= lanes, capped at 32
  return bin;
}
template <class T>
BinnedHost build_binned(const HostCsr<T>& A, int elems_per_lane);

}  // namespace rasp
