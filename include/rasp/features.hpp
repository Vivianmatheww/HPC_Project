#pragma once
// Structural features of a CSR matrix for the runtime selector. Computed on the GPU by one kernel that
// reads row_ptr plus two column indices per row (first and last): much less traffic than one SpMV.
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "rasp/csr.hpp"

namespace rasp {

constexpr int kLenBuckets = 8;  // row lengths <=1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, >64

struct MatrixFeatures {
  int32_t rows = 0, cols = 0;
  int64_t nnz = 0;
  int32_t max_len = 0;
  double mean_len = 0;       // nnz / rows
  double cv_len = 0;         // std / mean of row lengths
  double empty_frac = 0;     // fraction of rows with no nonzeros
  double ell_fill = 1;       // rows * max_len / nnz (ELL padding factor)
  double sell32_fill = 1;    // sum over 32-row slices of 32 * slice max / nnz (SELL-32 padding, unsorted)
  double long_nnz_frac = 0;  // fraction of nonzeros in rows longer than kLongRow (4096)
  double col_span = 0;       // mean over non-empty rows of (last col - first col + 1) / cols
  double len_hist[kLenBuckets] = {};  // fraction of rows per length bucket
};

// Owns the device scratch and pinned result buffer, so repeated extraction allocates nothing.
class FeatureExtractor {
 public:
  FeatureExtractor();
  ~FeatureExtractor();
  FeatureExtractor(const FeatureExtractor&) = delete;
  FeatureExtractor& operator=(const FeatureExtractor&) = delete;

  // Synchronises `stream` (the selector needs the values on the host).
  template <class T>
  MatrixFeatures extract(const DeviceCsr<T>& A, cudaStream_t stream = nullptr);

 private:
  void* acc_ = nullptr;   // device accumulators
  void* host_ = nullptr;  // pinned copy
};

// Reference implementation on the host (tests, and documentation of the exact definitions).
template <class T>
MatrixFeatures cpu_features(const HostCsr<T>& A);

// The selector's input vector, in the order of feature_names(). fp64 is 1 for double precision.
std::vector<std::string> feature_names();
std::vector<double> feature_vector(const MatrixFeatures& f, bool fp64);

}  // namespace rasp
