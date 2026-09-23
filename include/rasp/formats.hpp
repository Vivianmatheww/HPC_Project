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

}  // namespace rasp
