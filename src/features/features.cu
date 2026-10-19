#include <algorithm>
#include <cmath>

#include "rasp/features.hpp"
#include "rasp/formats.hpp"  // kLongRow

namespace rasp {
namespace {

struct Acc {
  double sum_sq;
  double span_sum;
  unsigned long long empty, nonempty, sell_slots, long_nnz;
  unsigned long long hist[kLenBuckets];
  int max_len;
};

__host__ __device__ inline int len_bucket(int len) {
  if (len <= 1) return 0;
#ifdef __CUDA_ARCH__
  const int b = 32 - __clz(len - 1);
#else
  const int b = 32 - __builtin_clz(unsigned(len - 1));
#endif
  return b < kLenBuckets - 1 ? b : kLenBuckets - 1;
}

template <class V>
__device__ inline V warp_sum(V v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}

// Warp-wide max, returned to every lane. __reduce_max_sync needs sm_80; older GPUs (the T4 on Kaggle and Colab is
// sm_75) use a butterfly of shuffles.
__device__ inline int warp_max(int v) {
#if __CUDA_ARCH__ >= 800
  return __reduce_max_sync(0xffffffffu, v);
#else
  for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
#endif
}

// Warp lanes always cover 32 consecutive, 32-aligned rows (the stride is a multiple of 32), so a warp-wide
// max is exactly one SELL-32 slice.
__global__ void features_kernel(int rows, int cols, const int* __restrict__ rp, const int* __restrict__ ci,
                                Acc* __restrict__ acc) {
  const int lane = threadIdx.x & 31;
  const long long stride = (long long)gridDim.x * blockDim.x;
  double sq = 0, span = 0;
  unsigned long long empty = 0, nonempty = 0, slots = 0, long_nnz = 0, hist[kLenBuckets] = {};
  int mx = 0;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i - lane < rows; i += stride) {
    int len = 0;
    if (i < rows) {
      const int b = rp[i], e = rp[i + 1];
      len = e - b;
      sq += double(len) * len;
      mx = max(mx, len);
      ++hist[len_bucket(len)];
      if (len == 0) {
        ++empty;
      } else {
        ++nonempty;
        span += double(ci[e - 1] - ci[b] + 1) / cols;
      }
      if (len > kLongRow) long_nnz += len;
    }
    const int slice_max = warp_max(len);
    if (lane == 0) slots += 32ull * slice_max;
  }
  sq = warp_sum(sq);
  span = warp_sum(span);
  empty = warp_sum(empty);
  nonempty = warp_sum(nonempty);
  slots = warp_sum(slots);
  long_nnz = warp_sum(long_nnz);
  for (int k = 0; k < kLenBuckets; ++k) hist[k] = warp_sum(hist[k]);
  mx = warp_max(mx);
  if (lane == 0) {
    atomicAdd(&acc->sum_sq, sq);
    atomicAdd(&acc->span_sum, span);
    atomicAdd(&acc->empty, empty);
    atomicAdd(&acc->nonempty, nonempty);
    atomicAdd(&acc->sell_slots, slots);
    atomicAdd(&acc->long_nnz, long_nnz);
    for (int k = 0; k < kLenBuckets; ++k) atomicAdd(&acc->hist[k], hist[k]);
    atomicMax(&acc->max_len, mx);
  }
}

MatrixFeatures finish(int32_t rows, int32_t cols, int64_t nnz, const Acc& a) {
  MatrixFeatures f;
  f.rows = rows;
  f.cols = cols;
  f.nnz = nnz;
  f.max_len = a.max_len;
  if (rows > 0) {
    f.mean_len = double(nnz) / rows;
    const double var = std::max(0.0, a.sum_sq / rows - f.mean_len * f.mean_len);
    f.cv_len = f.mean_len > 0 ? std::sqrt(var) / f.mean_len : 0.0;
    f.empty_frac = double(a.empty) / rows;
    for (int k = 0; k < kLenBuckets; ++k) f.len_hist[k] = double(a.hist[k]) / rows;
  }
  if (nnz > 0) {
    f.ell_fill = double(rows) * a.max_len / double(nnz);
    f.sell32_fill = double(a.sell_slots) / double(nnz);
    f.long_nnz_frac = double(a.long_nnz) / double(nnz);
  }
  f.col_span = a.nonempty ? a.span_sum / double(a.nonempty) : 0.0;
  return f;
}

}  // namespace

FeatureExtractor::FeatureExtractor() {
  RASP_CUDA(cudaMalloc(&acc_, sizeof(Acc)));
  RASP_CUDA(cudaMallocHost(&host_, sizeof(Acc)));
}

FeatureExtractor::~FeatureExtractor() {
  cudaFree(acc_);
  cudaFreeHost(host_);
}

template <class T>
MatrixFeatures FeatureExtractor::extract(const DeviceCsr<T>& A, cudaStream_t s) {
  Acc* h = static_cast<Acc*>(host_);
  RASP_CUDA(cudaMemsetAsync(acc_, 0, sizeof(Acc), s));
  if (A.rows() > 0) {
    static int grid_cap = 0;
    if (!grid_cap) {
      int dev = 0, sms = 0;
      RASP_CUDA(cudaGetDevice(&dev));
      RASP_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
      grid_cap = sms * 8;
    }
    const int grid = int(std::min<long long>((A.rows() + 255) / 256, grid_cap));
    features_kernel<<<grid, 256, 0, s>>>(A.rows(), std::max(A.cols(), 1), A.row_ptr(), A.col_idx(),
                                         static_cast<Acc*>(acc_));
  }
  RASP_CUDA(cudaMemcpyAsync(h, acc_, sizeof(Acc), cudaMemcpyDeviceToHost, s));
  RASP_CUDA(cudaStreamSynchronize(s));
  RASP_CUDA(cudaGetLastError());
  return finish(A.rows(), A.cols(), A.nnz(), *h);
}

template <class T>
MatrixFeatures cpu_features(const HostCsr<T>& A) {
  Acc a{};
  const int cols = std::max(A.cols, 1);
  for (int32_t s = 0; s * 32 < std::max(A.rows, 0); ++s) {
    int slice_max = 0;
    for (int32_t r = s * 32; r < std::min(A.rows, s * 32 + 32); ++r) {
      const int b = A.row_ptr[size_t(r)], e = A.row_ptr[size_t(r) + 1], len = e - b;
      a.sum_sq += double(len) * len;
      a.max_len = std::max(a.max_len, len);
      ++a.hist[len_bucket(len)];
      if (len == 0) {
        ++a.empty;
      } else {
        ++a.nonempty;
        a.span_sum += double(A.col_idx[size_t(e) - 1] - A.col_idx[size_t(b)] + 1) / cols;
      }
      if (len > kLongRow) a.long_nnz += unsigned(len);
      slice_max = std::max(slice_max, len);
    }
    a.sell_slots += 32ull * unsigned(slice_max);
  }
  return finish(A.rows, A.cols, A.nnz, a);
}

std::vector<std::string> feature_names() {
  std::vector<std::string> n = {"log2_rows", "log2_nnz", "mean_len", "cv_len", "log2_max_len", "max_over_mean",
                                "empty_frac", "log2_ell_fill", "sell32_fill", "long_nnz_frac", "col_span"};
  for (int k = 0; k < kLenBuckets; ++k) n.push_back("len_hist_" + std::to_string(k));
  n.push_back("fp64");
  return n;
}

std::vector<double> feature_vector(const MatrixFeatures& f, bool fp64) {
  std::vector<double> v = {std::log2(std::max(f.rows, 1)),
                           std::log2(double(std::max<int64_t>(f.nnz, 1))),
                           f.mean_len,
                           f.cv_len,
                           std::log2(std::max(f.max_len, 1)),
                           f.mean_len > 0 ? f.max_len / f.mean_len : 0.0,
                           f.empty_frac,
                           std::log2(std::max(f.ell_fill, 1e-9)),
                           f.sell32_fill,
                           f.long_nnz_frac,
                           f.col_span};
  for (int k = 0; k < kLenBuckets; ++k) v.push_back(f.len_hist[k]);
  v.push_back(fp64 ? 1.0 : 0.0);
  return v;
}

template MatrixFeatures FeatureExtractor::extract(const DeviceCsr<float>&, cudaStream_t);
template MatrixFeatures FeatureExtractor::extract(const DeviceCsr<double>&, cudaStream_t);
template MatrixFeatures cpu_features(const HostCsr<float>&);
template MatrixFeatures cpu_features(const HostCsr<double>&);

}  // namespace rasp
