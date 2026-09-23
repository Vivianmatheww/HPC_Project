// Phase 2 SpMV kernels: CSR-scalar, CSR-vector, ELL, SELL-C-sigma, Adaptive Row-grouped CSR.
// All accumulate in T (like cuSPARSE) and read x through the read-only path (const __restrict__).
#include <algorithm>
#include <chrono>
#include <tuple>

#include "rasp/cusparse_ref.hpp"
#include "rasp/formats.hpp"
#include "rasp/spmv.hpp"

namespace rasp {
namespace {

constexpr int kBlock = 256;

unsigned grid_for(int64_t threads, int block = kBlock) { return unsigned((threads + block - 1) / block); }

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// ---------------------------------------------------------------- kernels
template <class T>
__global__ void csr_scalar_kernel(int rows, const int* __restrict__ rp, const int* __restrict__ ci,
                                  const T* __restrict__ v, const T* __restrict__ x, T* __restrict__ y) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  T s = 0;
  for (int k = rp[r]; k < rp[r + 1]; ++k) s += v[k] * x[ci[k]];
  y[r] = s;
}

// V consecutive lanes per row; lanes of one row always belong to the same warp (V divides 32).
template <class T, int V>
__global__ void csr_vector_kernel(int rows, const int* __restrict__ rp, const int* __restrict__ ci,
                                  const T* __restrict__ v, const T* __restrict__ x, T* __restrict__ y) {
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int row = int(tid / V);
  const int lane = threadIdx.x % V;
  if (row >= rows) return;  // whole sub-warp exits together
  const unsigned mask = V == 32 ? 0xffffffffu : (((1u << V) - 1) << ((threadIdx.x % 32) / V * V));
  T s = 0;
  for (int k = rp[row] + lane; k < rp[row + 1]; k += V) s += v[k] * x[ci[k]];
  for (int o = V / 2; o > 0; o >>= 1) s += __shfl_down_sync(mask, s, o, V);
  if (lane == 0) y[row] = s;
}

template <class T>
__global__ void ell_kernel(int rows, int width, const int* __restrict__ col, const T* __restrict__ val,
                           const T* __restrict__ x, T* __restrict__ y) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  T s = 0;
  for (int j = 0; j < width; ++j) {
    const size_t p = size_t(j) * rows + r;
    const int c = col[p];
    if (c < 0) break;  // padding is trailing
    s += val[p] * x[c];
  }
  y[r] = s;
}

template <class T, int C>
__global__ void sell_kernel(int rows, const int* __restrict__ slice_ptr, const int* __restrict__ slice_width,
                            const int* __restrict__ perm, const int* __restrict__ col, const T* __restrict__ val,
                            const T* __restrict__ x, T* __restrict__ y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows) return;
  const int s = i / C;
  int p = slice_ptr[s] + i % C;
  const int w = slice_width[s];
  T sum = 0;
  for (int j = 0; j < w; ++j, p += C) {
    const int c = col[p];
    if (c < 0) break;
    sum += val[p] * x[c];
  }
  y[perm[i]] = sum;
}

// Listing 2 of Heller & Oberhuber: one block per group, one thread per chunk. Partial sums go to shared
// memory, then the first `size` threads each reduce their row's chunks.
template <class T>
__global__ void argcsr_kernel(const ArgGroup* __restrict__ groups, const int* __restrict__ threads_end,
                              const int* __restrict__ col, const T* __restrict__ val, const T* __restrict__ x,
                              T* __restrict__ y) {
  extern __shared__ __align__(16) unsigned char smem[];
  ArgGroup* g = reinterpret_cast<ArgGroup*>(smem);
  int* map = reinterpret_cast<int*>(smem + sizeof(ArgGroup));
  T* partial = reinterpret_cast<T*>(smem + sizeof(ArgGroup) + blockDim.x * sizeof(int));

  // The paper fetches the 16-byte group record with four threads; one 16-byte load is equivalent.
  if (threadIdx.x == 0) *g = groups[blockIdx.x];
  __syncthreads();
  const int size = g->size, first = g->first_row, chunk = g->chunk;
  if (threadIdx.x < size) map[threadIdx.x] = threads_end[first + threadIdx.x];

  T sum = 0;
  int p = g->offset + threadIdx.x;
  for (int i = 0; i < chunk; ++i, p += blockDim.x) {
    const int c = col[p];
    if (c < 0) break;
    sum += val[p] * x[c];
  }
  partial[threadIdx.x] = sum;
  __syncthreads();

  if (threadIdx.x < size) {
    const int begin = threadIdx.x ? map[threadIdx.x - 1] : 0, end = map[threadIdx.x];
    T s = 0;
    for (int i = begin; i < end; ++i) s += partial[i];
    y[first + threadIdx.x] = s;
  }
}

// ---------------------------------------------------------------- kernel objects
template <class T>
class CusparseKernel final : public SpmvKernel<T> {
 public:
  CusparseKernel(int alg, const DeviceCsr<T>& A, const T* x, T* y)
      : op_(A, alg == 1 ? SpmvAlg::CsrAlg1 : SpmvAlg::CsrAlg2, x, y) {
    this->convert_ms_ = op_.preprocess_ms();
    this->format_bytes_ = A.bytes() + op_.workspace_bytes();
    this->stored_slots_ = A.nnz();
  }
  void run(cudaStream_t s) override { op_.run(s); }

 private:
  CusparseSpmv<T> op_;
};

template <class T>
class CsrKernel final : public SpmvKernel<T> {
 public:
  CsrKernel(int v, const DeviceCsr<T>& A, const T* x, T* y) : v_(v), A_(A), x_(x), y_(y) {
    this->format_bytes_ = A.bytes();
    this->stored_slots_ = A.nnz();
  }
  void run(cudaStream_t s) override {
    const int rows = A_.rows();
    if (rows == 0) return;
    auto args = std::make_tuple(rows, A_.row_ptr(), A_.col_idx(), A_.vals(), x_, y_);
    const unsigned grid = grid_for(int64_t(rows) * std::max(v_, 1));
    switch (v_) {
      case 0: std::apply([&](auto... a) { csr_scalar_kernel<T><<<grid, kBlock, 0, s>>>(a...); }, args); break;
      case 2: std::apply([&](auto... a) { csr_vector_kernel<T, 2><<<grid, kBlock, 0, s>>>(a...); }, args); break;
      case 4: std::apply([&](auto... a) { csr_vector_kernel<T, 4><<<grid, kBlock, 0, s>>>(a...); }, args); break;
      case 8: std::apply([&](auto... a) { csr_vector_kernel<T, 8><<<grid, kBlock, 0, s>>>(a...); }, args); break;
      case 16: std::apply([&](auto... a) { csr_vector_kernel<T, 16><<<grid, kBlock, 0, s>>>(a...); }, args); break;
      case 32: std::apply([&](auto... a) { csr_vector_kernel<T, 32><<<grid, kBlock, 0, s>>>(a...); }, args); break;
      default: throw std::invalid_argument("csr_vector: v must be 2, 4, 8, 16 or 32");
    }
  }

 private:
  int v_;  // 0 = scalar
  const DeviceCsr<T>& A_;
  const T* x_;
  T* y_;
};

template <class T>
class EllKernel final : public SpmvKernel<T> {
 public:
  EllKernel(const HostCsr<T>& h, const T* x, T* y) : x_(x), y_(y) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto E = build_ell(h);
    rows_ = E.rows;
    width_ = E.width;
    col_.upload(E.col);
    val_.upload(E.val);
    RASP_CUDA(cudaDeviceSynchronize());
    this->convert_ms_ = ms_since(t0);
    this->format_bytes_ = col_.bytes() + val_.bytes();
    this->stored_slots_ = int64_t(E.col.size());
  }
  void run(cudaStream_t s) override {
    if (rows_ == 0) return;
    if (width_ == 0) {
      RASP_CUDA(cudaMemsetAsync(y_, 0, size_t(rows_) * sizeof(T), s));
      return;
    }
    ell_kernel<T><<<grid_for(rows_), kBlock, 0, s>>>(rows_, width_, col_.data(), val_.data(), x_, y_);
  }

 private:
  int rows_ = 0, width_ = 0;
  DeviceBuffer<int32_t> col_;
  DeviceBuffer<T> val_;
  const T* x_;
  T* y_;
};

template <class T>
class SellKernel final : public SpmvKernel<T> {
 public:
  SellKernel(int C, int sigma, const HostCsr<T>& h, const T* x, T* y) : x_(x), y_(y) {
    if (C != 8 && C != 32) throw std::invalid_argument("sell: C must be 8 or 32");
    const auto t0 = std::chrono::steady_clock::now();
    const auto S = build_sell(h, C, sigma);
    rows_ = S.rows;
    C_ = C;
    slice_ptr_.upload(S.slice_ptr);
    slice_width_.upload(S.slice_width);
    perm_.upload(S.perm);
    col_.upload(S.col);
    val_.upload(S.val);
    RASP_CUDA(cudaDeviceSynchronize());
    this->convert_ms_ = ms_since(t0);
    this->format_bytes_ = slice_ptr_.bytes() + slice_width_.bytes() + perm_.bytes() + col_.bytes() + val_.bytes();
    this->stored_slots_ = int64_t(S.col.size());
  }
  void run(cudaStream_t s) override {
    if (rows_ == 0) return;
    if (C_ == 8)
      sell_kernel<T, 8><<<grid_for(rows_), kBlock, 0, s>>>(rows_, slice_ptr_.data(), slice_width_.data(),
                                                             perm_.data(), col_.data(), val_.data(), x_, y_);
    else
      sell_kernel<T, 32><<<grid_for(rows_), kBlock, 0, s>>>(rows_, slice_ptr_.data(), slice_width_.data(),
                                                              perm_.data(), col_.data(), val_.data(), x_, y_);
  }

 private:
  int rows_ = 0, C_ = 32;
  DeviceBuffer<int32_t> slice_ptr_, slice_width_, perm_, col_;
  DeviceBuffer<T> val_;
  const T* x_;
  T* y_;
};

template <class T>
class ArgCsrKernel final : public SpmvKernel<T> {
 public:
  ArgCsrKernel(int block, int chunk, const HostCsr<T>& h, const T* x, T* y) : block_(block), x_(x), y_(y) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto G = build_argcsr(h, block, chunk);
    groups_ = int(G.groups.size());
    groups_buf_.upload(G.groups);
    threads_end_.upload(G.threads_end);
    col_.upload(G.col);
    val_.upload(G.val);
    RASP_CUDA(cudaDeviceSynchronize());
    this->convert_ms_ = ms_since(t0);
    this->format_bytes_ = groups_buf_.bytes() + threads_end_.bytes() + col_.bytes() + val_.bytes();
    this->stored_slots_ = int64_t(G.col.size());
  }
  void run(cudaStream_t s) override {
    if (groups_ == 0) return;
    const size_t smem = sizeof(ArgGroup) + size_t(block_) * (sizeof(int) + sizeof(T));
    argcsr_kernel<T><<<groups_, block_, smem, s>>>(groups_buf_.data(), threads_end_.data(), col_.data(),
                                                    val_.data(), x_, y_);
  }

 private:
  int block_, groups_ = 0;
  DeviceBuffer<ArgGroup> groups_buf_;
  DeviceBuffer<int32_t> threads_end_, col_;
  DeviceBuffer<T> val_;
  const T* x_;
  T* y_;
};

}  // namespace

template <class T>
std::unique_ptr<SpmvKernel<T>> make_spmv_kernel(const KernelConfig& cfg, const HostCsr<T>& host,
                                                const DeviceCsr<T>& dev, const T* x, T* y) {
  switch (cfg.format) {
    case Format::Cusparse: return std::make_unique<CusparseKernel<T>>(cfg.alg, dev, x, y);
    case Format::CsrScalar: return std::make_unique<CsrKernel<T>>(0, dev, x, y);
    case Format::CsrVector: return std::make_unique<CsrKernel<T>>(cfg.v, dev, x, y);
    case Format::Ell: return std::make_unique<EllKernel<T>>(host, x, y);
    case Format::Sell: return std::make_unique<SellKernel<T>>(cfg.C, cfg.sigma, host, x, y);
    case Format::ArgCsr: return std::make_unique<ArgCsrKernel<T>>(cfg.block, cfg.chunk, host, x, y);
  }
  throw std::invalid_argument("make_spmv_kernel: unknown format");
}

template std::unique_ptr<SpmvKernel<float>> make_spmv_kernel(const KernelConfig&, const HostCsr<float>&,
                                                             const DeviceCsr<float>&, const float*, float*);
template std::unique_ptr<SpmvKernel<double>> make_spmv_kernel(const KernelConfig&, const HostCsr<double>&,
                                                              const DeviceCsr<double>&, const double*, double*);

}  // namespace rasp
