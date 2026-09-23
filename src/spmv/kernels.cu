// Phase 2 SpMV kernels: CSR-scalar, CSR-vector, ELL, SELL-C-sigma, Adaptive Row-grouped CSR.
// All accumulate in T (like cuSPARSE) and read x through the read-only path (const __restrict__).
#include <algorithm>
#include <chrono>
#include <tuple>

#include "rasp/cusparse_ref.hpp"
#include "rasp/formats_device.hpp"
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

// CSR-binned: one launch covers every bin; each block finds its bin from the (tiny) block table.
struct BinTable {
  int row_start[kBins + 1];
  int block_start[kBins + 1];
};

template <class T>
__global__ void csr_binned_kernel(BinTable tab, const int* __restrict__ order, const int* __restrict__ rp,
                                  const int* __restrict__ ci, const T* __restrict__ v, const T* __restrict__ x,
                                  T* __restrict__ y) {
  int bin = 0;
  while (bin < kBins - 1 && int(blockIdx.x) >= tab.block_start[bin + 1]) ++bin;
  const int local = blockIdx.x - tab.block_start[bin];

  if (bin == kBins - 1) {  // one long row per block, block-wide reduction
    __shared__ T warp_sums[kBinnedBlock / 32];
    const int row = order[tab.row_start[bin] + local];
    T s = 0;
    for (int k = rp[row] + threadIdx.x; k < rp[row + 1]; k += blockDim.x) s += v[k] * x[ci[k]];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (threadIdx.x % 32 == 0) warp_sums[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
      s = threadIdx.x < blockDim.x / 32 ? warp_sums[threadIdx.x] : T(0);
      for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
      if (threadIdx.x == 0) y[row] = s;
    }
    return;
  }

  const int V = 1 << bin;  // lanes per row
  const int idx = local * (blockDim.x / V) + threadIdx.x / V;
  if (idx >= tab.row_start[bin + 1] - tab.row_start[bin]) return;  // whole V-lane group exits together
  const int row = order[tab.row_start[bin] + idx];
  const int lane = threadIdx.x % V;
  T s = 0;
  for (int k = rp[row] + lane; k < rp[row + 1]; k += V) s += v[k] * x[ci[k]];
  if (V > 1) {
    const unsigned mask = V == 32 ? 0xffffffffu : (((1u << V) - 1) << ((threadIdx.x % 32) / V * V));
    for (int o = V / 2; o > 0; o >>= 1) s += __shfl_down_sync(mask, s, o, V);
  }
  if (lane == 0) y[row] = s;
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

// Padded-format kernels own a device format (built on the host and uploaded, or converted on the GPU).
template <class T>
class EllKernel final : public SpmvKernel<T> {
 public:
  EllKernel(EllDevice<T> E, double convert_ms, const T* x, T* y) : E_(std::move(E)), x_(x), y_(y) {
    this->convert_ms_ = convert_ms;
    this->format_bytes_ = E_.col.bytes() + E_.val.bytes();
    this->stored_slots_ = E_.stored;
  }
  void run(cudaStream_t s) override {
    if (E_.rows == 0) return;
    if (E_.width == 0) {
      RASP_CUDA(cudaMemsetAsync(y_, 0, size_t(E_.rows) * sizeof(T), s));
      return;
    }
    ell_kernel<T><<<grid_for(E_.rows), kBlock, 0, s>>>(E_.rows, E_.width, E_.col.data(), E_.val.data(), x_, y_);
  }

 private:
  EllDevice<T> E_;
  const T* x_;
  T* y_;
};

template <class T>
class SellKernel final : public SpmvKernel<T> {
 public:
  SellKernel(SellDevice<T> S, double convert_ms, const T* x, T* y) : S_(std::move(S)), x_(x), y_(y) {
    if (S_.C != 8 && S_.C != 32) throw std::invalid_argument("sell: C must be 8 or 32");
    this->convert_ms_ = convert_ms;
    this->format_bytes_ = S_.slice_ptr.bytes() + S_.slice_width.bytes() + S_.perm.bytes() + S_.col.bytes() + S_.val.bytes();
    this->stored_slots_ = S_.stored;
  }
  void run(cudaStream_t s) override {
    if (S_.rows == 0) return;
    if (S_.C == 8)
      sell_kernel<T, 8><<<grid_for(S_.rows), kBlock, 0, s>>>(S_.rows, S_.slice_ptr.data(), S_.slice_width.data(),
                                                               S_.perm.data(), S_.col.data(), S_.val.data(), x_, y_);
    else
      sell_kernel<T, 32><<<grid_for(S_.rows), kBlock, 0, s>>>(S_.rows, S_.slice_ptr.data(), S_.slice_width.data(),
                                                                S_.perm.data(), S_.col.data(), S_.val.data(), x_, y_);
  }

 private:
  SellDevice<T> S_;
  const T* x_;
  T* y_;
};

template <class T>
class CsrBinnedKernel final : public SpmvKernel<T> {
 public:
  CsrBinnedKernel(BinnedDevice B, double convert_ms, const DeviceCsr<T>& A, const T* x, T* y)
      : A_(A), B_(std::move(B)), x_(x), y_(y) {
    for (int k = 0; k <= kBins; ++k) {
      tab_.row_start[k] = B_.bin_row_start[size_t(k)];
      tab_.block_start[k] = B_.bin_block_start[size_t(k)];
    }
    this->convert_ms_ = convert_ms;
    this->format_bytes_ = A.bytes() + B_.order.bytes();
    this->stored_slots_ = A.nnz();
  }
  void run(cudaStream_t s) override {
    if (B_.blocks == 0) return;
    csr_binned_kernel<T><<<B_.blocks, kBinnedBlock, 0, s>>>(tab_, B_.order.data(), A_.row_ptr(), A_.col_idx(),
                                                            A_.vals(), x_, y_);
  }

 private:
  const DeviceCsr<T>& A_;
  BinnedDevice B_;
  BinTable tab_{};
  const T* x_;
  T* y_;
};

template <class T>
class ArgCsrKernel final : public SpmvKernel<T> {
 public:
  ArgCsrKernel(ArgCsrDevice<T> G, double convert_ms, const T* x, T* y) : G_(std::move(G)), x_(x), y_(y) {
    this->convert_ms_ = convert_ms;
    this->format_bytes_ = G_.group_info.bytes() + G_.threads_end.bytes() + G_.col.bytes() + G_.val.bytes();
    this->stored_slots_ = G_.stored;
  }
  void run(cudaStream_t s) override {
    if (G_.groups == 0) return;
    const size_t smem = sizeof(ArgGroup) + size_t(G_.block) * (sizeof(int) + sizeof(T));
    argcsr_kernel<T><<<G_.groups, G_.block, smem, s>>>(G_.group_info.data(), G_.threads_end.data(), G_.col.data(),
                                                       G_.val.data(), x_, y_);
  }

 private:
  ArgCsrDevice<T> G_;
  const T* x_;
  T* y_;
};

// Times `build` (conversion + upload/transfer, synchronised) and hands the result to the kernel constructor.
template <class Build>
auto timed(Build&& build) {
  const auto t0 = std::chrono::steady_clock::now();
  auto out = build();
  RASP_CUDA(cudaDeviceSynchronize());
  return std::make_pair(std::move(out), ms_since(t0));
}

}  // namespace

template <class T>
std::unique_ptr<SpmvKernel<T>> make_spmv_kernel(const KernelConfig& cfg, const HostCsr<T>& host,
                                                const DeviceCsr<T>& dev, const T* x, T* y) {
  switch (cfg.format) {
    case Format::Cusparse: return std::make_unique<CusparseKernel<T>>(cfg.alg, dev, x, y);
    case Format::CsrScalar: return std::make_unique<CsrKernel<T>>(0, dev, x, y);
    case Format::CsrVector: return std::make_unique<CsrKernel<T>>(cfg.v, dev, x, y);
    case Format::Ell: {
      auto [E, ms] = timed([&] { return upload_ell(build_ell(host)); });
      return std::make_unique<EllKernel<T>>(std::move(E), ms, x, y);
    }
    case Format::Sell: {
      auto [S, ms] = timed([&] { return upload_sell(build_sell(host, cfg.C, cfg.sigma)); });
      return std::make_unique<SellKernel<T>>(std::move(S), ms, x, y);
    }
    case Format::ArgCsr: {
      auto [G, ms] = timed([&] { return upload_argcsr(build_argcsr(host, cfg.block, cfg.chunk)); });
      return std::make_unique<ArgCsrKernel<T>>(std::move(G), ms, x, y);
    }
    case Format::ArgCsrLocal: {
      auto [G, ms] = timed([&] { return upload_argcsr(build_argcsr_local(host, cfg.block, cfg.rows)); });
      return std::make_unique<ArgCsrKernel<T>>(std::move(G), ms, x, y);
    }
    case Format::CsrBinned: {
      auto [B, ms] = timed([&] { return upload_binned(build_binned(host, cfg.t)); });
      return std::make_unique<CsrBinnedKernel<T>>(std::move(B), ms, dev, x, y);
    }
  }
  throw std::invalid_argument("make_spmv_kernel: unknown format");
}

template <class T>
std::unique_ptr<SpmvKernel<T>> make_spmv_kernel_gpu(const KernelConfig& cfg, const DeviceCsr<T>& dev, const T* x, T* y,
                                                    cudaStream_t stream) {
  switch (cfg.format) {
    case Format::Cusparse: return std::make_unique<CusparseKernel<T>>(cfg.alg, dev, x, y);
    case Format::CsrScalar: return std::make_unique<CsrKernel<T>>(0, dev, x, y);
    case Format::CsrVector: return std::make_unique<CsrKernel<T>>(cfg.v, dev, x, y);
    case Format::Ell: {
      auto [E, ms] = timed([&] { return convert_ell_gpu(dev, 3.0, stream); });
      return std::make_unique<EllKernel<T>>(std::move(E), ms, x, y);
    }
    case Format::Sell: {
      auto [S, ms] = timed([&] { return convert_sell_gpu(dev, cfg.C, cfg.sigma, 64.0, stream); });
      return std::make_unique<SellKernel<T>>(std::move(S), ms, x, y);
    }
    case Format::ArgCsr: {
      auto [G, ms] = timed([&] { return convert_argcsr_gpu(dev, cfg.block, cfg.chunk, 0, 64.0, stream); });
      return std::make_unique<ArgCsrKernel<T>>(std::move(G), ms, x, y);
    }
    case Format::ArgCsrLocal: {
      auto [G, ms] = timed([&] { return convert_argcsr_gpu(dev, cfg.block, 0, cfg.rows, 64.0, stream); });
      return std::make_unique<ArgCsrKernel<T>>(std::move(G), ms, x, y);
    }
    case Format::CsrBinned: {
      auto [B, ms] = timed([&] { return convert_binned_gpu(dev, cfg.t, stream); });
      return std::make_unique<CsrBinnedKernel<T>>(std::move(B), ms, dev, x, y);
    }
  }
  throw std::invalid_argument("make_spmv_kernel_gpu: unknown format");
}

#define RASP_INSTANTIATE(T)                                                                                    \
  template std::unique_ptr<SpmvKernel<T>> make_spmv_kernel(const KernelConfig&, const HostCsr<T>&,             \
                                                           const DeviceCsr<T>&, const T*, T*);                 \
  template std::unique_ptr<SpmvKernel<T>> make_spmv_kernel_gpu(const KernelConfig&, const DeviceCsr<T>&,       \
                                                               const T*, T*, cudaStream_t);
RASP_INSTANTIATE(float)
RASP_INSTANTIATE(double)
#undef RASP_INSTANTIATE

}  // namespace rasp
