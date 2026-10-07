// Small device kernels of the out-of-core SpGEMM executor: row-pointer bookkeeping and the sum of two CSR matrices
// (partial products of a 2-D chunked SpGEMM).
#include <cub/device/device_scan.cuh>

#include <algorithm>

#include "rasp/common.hpp"
#include "rasp/spgemm_kernels.hpp"

namespace rasp {
namespace {

constexpr int kThreads = 256;
unsigned grid_for(int64_t n) { return unsigned(std::min<int64_t>((n + kThreads - 1) / kThreads, 65535)); }

__global__ void add_scalar_kernel(const int32_t* __restrict__ x, int32_t* __restrict__ y, int64_t n, int32_t delta) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    y[i] = x[i] + delta;
}

__global__ void expand_kernel(const int32_t* __restrict__ blk, int32_t k0, int32_t k1, int32_t rows,
                              int32_t* __restrict__ out) {
  const int32_t base = blk[0], nnz = blk[k1 - k0] - base;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i <= rows; i += int64_t(gridDim.x) * blockDim.x)
    out[i] = i <= k0 ? 0 : i >= k1 ? nnz : blk[i - k0] - base;
}

// One thread per row: merge the two sorted column lists. Rows of the partial products in the chunked SpGEMM are
// short (tens of entries for the multigrid products), so a thread per row is enough.
__global__ void add_count_kernel(int32_t rows, const int32_t* __restrict__ xp, const int32_t* __restrict__ xc,
                                 const int32_t* __restrict__ yp, const int32_t* __restrict__ yc,
                                 int32_t* __restrict__ cnt) {
  const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i > rows) return;
  if (i == rows) {
    cnt[rows] = 0;  // the scan turns this into nnz
    return;
  }
  int32_t a = xp[i], b = yp[i], n = 0;
  const int32_t ae = xp[i + 1], be = yp[i + 1];
  while (a < ae && b < be) {
    const int32_t ca = xc[a], cb = yc[b];
    a += ca <= cb;
    b += cb <= ca;
    ++n;
  }
  cnt[i] = n + (ae - a) + (be - b);
}

template <class T>
__global__ void add_fill_kernel(int32_t rows, const int32_t* __restrict__ xp, const int32_t* __restrict__ xc,
                                const T* __restrict__ xv, const int32_t* __restrict__ yp,
                                const int32_t* __restrict__ yc, const T* __restrict__ yv,
                                const int32_t* __restrict__ cp, int32_t* __restrict__ cc, T* __restrict__ cv) {
  const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i >= rows) return;
  int32_t a = xp[i], b = yp[i], o = cp[i];
  const int32_t ae = xp[i + 1], be = yp[i + 1];
  while (a < ae || b < be) {
    const int32_t ca = a < ae ? xc[a] : INT32_MAX, cb = b < be ? yc[b] : INT32_MAX;
    if (ca == cb) {
      cc[o] = ca;
      cv[o] = xv[a++] + yv[b++];
    } else if (ca < cb) {
      cc[o] = ca;
      cv[o] = xv[a++];
    } else {
      cc[o] = cb;
      cv[o] = yv[b++];
    }
    ++o;
  }
}

size_t align_up(size_t n) { return (n + 255) / 256 * 256; }

size_t scan_temp_bytes(int32_t rows) {
  size_t bytes = 0;
  RASP_CUDA(cub::DeviceScan::ExclusiveSum(nullptr, bytes, static_cast<const int32_t*>(nullptr),
                                          static_cast<int32_t*>(nullptr), int64_t(rows) + 1));
  return bytes;
}

}  // namespace

void rebase_row_ptr(const int32_t* x, int32_t* y, int64_t n, int32_t base, cudaStream_t stream) {
  if (n <= 0) return;
  add_scalar_kernel<<<grid_for(n), kThreads, 0, stream>>>(x, y, n, -base);
  RASP_CUDA(cudaGetLastError());
}

void offset_row_ptr(const int32_t* x, int32_t* y, int64_t n, int32_t offset, cudaStream_t stream) {
  if (n <= 0) return;
  add_scalar_kernel<<<grid_for(n), kThreads, 0, stream>>>(x, y, n, offset);
  RASP_CUDA(cudaGetLastError());
}

void expand_row_ptr(const int32_t* blk, int32_t k0, int32_t k1, int32_t rows, int32_t* out, cudaStream_t stream) {
  expand_kernel<<<grid_for(int64_t(rows) + 1), kThreads, 0, stream>>>(blk, k0, k1, rows, out);
  RASP_CUDA(cudaGetLastError());
}

size_t csr_add_temp_bytes(int32_t rows) {
  return align_up((size_t(rows) + 1) * sizeof(int32_t)) + align_up(scan_temp_bytes(rows));
}

int64_t csr_add_symbolic(int32_t rows, const int32_t* xp, const int32_t* xc, const int32_t* yp, const int32_t* yc,
                         int32_t* cp, void* temp, size_t temp_bytes, cudaStream_t stream) {
  if (temp_bytes < csr_add_temp_bytes(rows)) throw std::invalid_argument("csr_add_symbolic: temp buffer too small");
  auto* cnt = static_cast<int32_t*>(temp);
  void* scan_temp = static_cast<char*>(temp) + align_up((size_t(rows) + 1) * sizeof(int32_t));
  size_t scan_bytes = temp_bytes - align_up((size_t(rows) + 1) * sizeof(int32_t));
  add_count_kernel<<<unsigned((int64_t(rows) + kThreads) / kThreads), kThreads, 0, stream>>>(rows, xp, xc, yp, yc, cnt);
  RASP_CUDA(cudaGetLastError());
  RASP_CUDA(cub::DeviceScan::ExclusiveSum(scan_temp, scan_bytes, cnt, cp, int64_t(rows) + 1, stream));
  int32_t nnz = 0;
  RASP_CUDA(cudaMemcpyAsync(&nnz, cp + rows, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
  RASP_CUDA(cudaStreamSynchronize(stream));
  return nnz;
}

template <class T>
void csr_add_numeric(int32_t rows, const int32_t* xp, const int32_t* xc, const T* xv, const int32_t* yp,
                     const int32_t* yc, const T* yv, const int32_t* cp, int32_t* cc, T* cv, cudaStream_t stream) {
  if (rows <= 0) return;
  add_fill_kernel<T><<<unsigned((int64_t(rows) + kThreads - 1) / kThreads), kThreads, 0, stream>>>(rows, xp, xc, xv, yp,
                                                                                                  yc, yv, cp, cc, cv);
  RASP_CUDA(cudaGetLastError());
}

template void csr_add_numeric(int32_t, const int32_t*, const int32_t*, const float*, const int32_t*, const int32_t*,
                              const float*, const int32_t*, int32_t*, float*, cudaStream_t);
template void csr_add_numeric(int32_t, const int32_t*, const int32_t*, const double*, const int32_t*, const int32_t*,
                              const double*, const int32_t*, int32_t*, double*, cudaStream_t);

}  // namespace rasp
