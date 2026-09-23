// GPU converters DeviceCsr -> ELL / SELL-C-sigma / ArgCSR(-local) / CSR-binned, and host-format uploads.
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <limits>

#include "rasp/formats_device.hpp"

namespace rasp {
namespace {

constexpr int kThreads = 256;
unsigned blocks_for(int64_t n) { return unsigned((n + kThreads - 1) / kThreads); }

__global__ void row_lengths_kernel(int rows, const int* __restrict__ rp, int* __restrict__ len, int* __restrict__ max_len) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const int l = rp[r + 1] - rp[r];
  len[r] = l;
  atomicMax(max_len, l);
}

template <class T>
__global__ void ell_fill_kernel(int rows, const int* __restrict__ rp, const int* __restrict__ ci, const T* __restrict__ v,
                                int* __restrict__ col, T* __restrict__ val) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  for (int k = rp[r], j = 0; k < rp[r + 1]; ++k, ++j) {
    const size_t slot = size_t(j) * rows + r;
    col[slot] = ci[k];
    val[slot] = v[k];
  }
}

// Stable descending-length order inside sigma windows: key = (window << 32) | (~len), sorted ascending.
__global__ void sell_keys_kernel(int rows, int sigma, const int* __restrict__ len, unsigned long long* __restrict__ key,
                                 int* __restrict__ idx) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const unsigned long long window = sigma == 0 ? 0ull : (unsigned long long)(r / sigma);
  key[r] = (window << 32) | (unsigned long long)(0xFFFFFFFFu - unsigned(len[r]));
  idx[r] = r;
}

__global__ void iota_kernel(int n, int* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = i;
}

__global__ void sell_width_kernel(int rows, int slices, int C, const int* __restrict__ perm, const int* __restrict__ len,
                                  int* __restrict__ width, long long* __restrict__ slots) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= slices) return;
  int w = 0;
  for (int i = s * C; i < min(rows, (s + 1) * C); ++i) w = max(w, len[perm[i]]);
  width[s] = w;
  slots[s] = (long long)w * C;
}

__global__ void to_int32_kernel(int n, const long long* __restrict__ in, int* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = int(in[i]);
}

template <class T>
__global__ void sell_fill_kernel(int rows, int C, const int* __restrict__ perm, const int* __restrict__ slice_ptr,
                                 const int* __restrict__ rp, const int* __restrict__ ci, const T* __restrict__ v,
                                 int* __restrict__ col, T* __restrict__ val) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows) return;
  const int r = perm[i];
  const size_t base = size_t(slice_ptr[i / C]) + size_t(i % C);
  for (int k = rp[r], j = 0; k < rp[r + 1]; ++k, ++j) {
    col[base + size_t(j) * C] = ci[k];
    val[base + size_t(j) * C] = v[k];
  }
}

template <class T>
__global__ void argcsr_fill_kernel(int rows, int block, const int* __restrict__ rp, const int* __restrict__ ci,
                                   const T* __restrict__ v, const int* __restrict__ row_offset,
                                   const int* __restrict__ row_tbegin, const int* __restrict__ row_fill,
                                   int* __restrict__ col, T* __restrict__ val) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const int f = row_fill[r], off = row_offset[r], tb = row_tbegin[r];
  for (int k = rp[r], e = 0; k < rp[r + 1]; ++k, ++e) {
    const size_t pos = size_t(off) + size_t(e % f) * block + size_t(tb + e / f);
    col[pos] = ci[k];
    val[pos] = v[k];
  }
}

__global__ void bin_kernel(int rows, int t, const int* __restrict__ rp, unsigned* __restrict__ key, int* __restrict__ idx,
                           unsigned* __restrict__ counts) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const int b = binned_bin(rp[r + 1] - rp[r], t);
  key[r] = unsigned(b);
  idx[r] = r;
  atomicAdd(&counts[b], 1u);
}

// Stable radix sort of (key, value) pairs over the low `bits` bits; returns the sorted values.
template <class Key>
DeviceBuffer<int32_t> stable_sort_values(int n, DeviceBuffer<Key>& keys, DeviceBuffer<int32_t>& vals, int bits,
                                         cudaStream_t s) {
  DeviceBuffer<Key> keys_out(static_cast<size_t>(n));
  DeviceBuffer<int32_t> vals_out(static_cast<size_t>(n));
  size_t temp_bytes = 0;
  RASP_CUDA(cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, keys.data(), keys_out.data(), vals.data(),
                                            vals_out.data(), n, 0, bits, s));
  DeviceBuffer<char> temp(std::max<size_t>(temp_bytes, 1));
  RASP_CUDA(cub::DeviceRadixSort::SortPairs(temp.data(), temp_bytes, keys.data(), keys_out.data(), vals.data(),
                                            vals_out.data(), n, 0, bits, s));
  return vals_out;
}

void check_fill(const char* what, int64_t stored, int64_t nnz, double max_fill, size_t value_bytes) {
  if (double(stored) > max_fill * double(std::max<int64_t>(nnz, 1)) ||
      double(stored) * double(value_bytes + 4) > kMaxFormatBytes || stored >= std::numeric_limits<int32_t>::max())
    throw FormatInfeasible(std::string(what) + " would store " + std::to_string(stored) + " slots for " +
                           std::to_string(nnz) + " nonzeros");
}

}  // namespace

// ---- uploads ----------------------------------------------------------------------------------
template <class T>
EllDevice<T> upload_ell(const EllHost<T>& h) {
  EllDevice<T> d;
  d.rows = h.rows;
  d.width = h.width;
  d.stored = int64_t(h.col.size());
  d.col.upload(h.col);
  d.val.upload(h.val);
  return d;
}

template <class T>
SellDevice<T> upload_sell(const SellHost<T>& h) {
  SellDevice<T> d;
  d.rows = h.rows;
  d.C = h.C;
  d.slices = h.slices;
  d.stored = int64_t(h.col.size());
  d.slice_ptr.upload(h.slice_ptr);
  d.slice_width.upload(h.slice_width);
  d.perm.upload(h.perm);
  d.col.upload(h.col);
  d.val.upload(h.val);
  return d;
}

template <class T>
ArgCsrDevice<T> upload_argcsr(const ArgCsrHost<T>& h) {
  ArgCsrDevice<T> d;
  d.groups = int32_t(h.groups.size());
  d.block = h.block;
  d.stored = int64_t(h.col.size());
  d.group_info.upload(h.groups);
  d.threads_end.upload(h.threads_end);
  d.col.upload(h.col);
  d.val.upload(h.val);
  return d;
}

BinnedDevice upload_binned(const BinnedHost& h) {
  BinnedDevice d;
  d.blocks = h.blocks;
  d.bin_row_start = h.bin_row_start;
  d.bin_block_start = h.bin_block_start;
  d.order.upload(h.order);
  return d;
}

// ---- GPU converters ---------------------------------------------------------------------------
template <class T>
EllDevice<T> convert_ell_gpu(const DeviceCsr<T>& A, double max_fill, cudaStream_t s) {
  EllDevice<T> E;
  E.rows = A.rows();
  if (A.rows() == 0) return E;
  DeviceBuffer<int32_t> len(static_cast<size_t>(A.rows())), max_len(1);
  RASP_CUDA(cudaMemsetAsync(max_len.data(), 0, sizeof(int32_t), s));
  row_lengths_kernel<<<blocks_for(A.rows()), kThreads, 0, s>>>(A.rows(), A.row_ptr(), len.data(), max_len.data());
  RASP_CUDA(cudaMemcpyAsync(&E.width, max_len.data(), sizeof(int32_t), cudaMemcpyDeviceToHost, s));
  RASP_CUDA(cudaStreamSynchronize(s));
  E.stored = int64_t(A.rows()) * E.width;
  if (double(E.stored) > max_fill * double(std::max<int64_t>(A.nnz(), 1)))
    throw FormatInfeasible("ELL would store " + std::to_string(E.stored) + " slots for " + std::to_string(A.nnz()) +
                           " nonzeros");
  check_fill("ELL", E.stored, A.nnz(), 1e18, sizeof(T));
  E.col.resize(size_t(E.stored));
  E.val.resize(size_t(E.stored));
  if (E.stored) {
    RASP_CUDA(cudaMemsetAsync(E.col.data(), 0xff, E.col.bytes(), s));  // -1 padding
    RASP_CUDA(cudaMemsetAsync(E.val.data(), 0, E.val.bytes(), s));
    ell_fill_kernel<T><<<blocks_for(A.rows()), kThreads, 0, s>>>(A.rows(), A.row_ptr(), A.col_idx(), A.vals(),
                                                                  E.col.data(), E.val.data());
  }
  RASP_CUDA(cudaStreamSynchronize(s));
  RASP_CUDA(cudaGetLastError());
  return E;
}

template <class T>
SellDevice<T> convert_sell_gpu(const DeviceCsr<T>& A, int C, int sigma, double max_fill, cudaStream_t s) {
  if (C <= 0 || sigma < 0) throw std::invalid_argument("convert_sell_gpu: C must be > 0 and sigma >= 0");
  SellDevice<T> S;
  const int rows = A.rows();
  S.rows = rows;
  S.C = C;
  S.slices = (rows + C - 1) / C;
  S.slice_ptr.resize(size_t(S.slices) + 1);
  S.slice_width.resize(size_t(S.slices));
  if (rows == 0) {
    RASP_CUDA(cudaMemsetAsync(S.slice_ptr.data(), 0, S.slice_ptr.bytes(), s));
    RASP_CUDA(cudaStreamSynchronize(s));
    return S;
  }
  DeviceBuffer<int32_t> len(static_cast<size_t>(rows)), max_len(1);
  RASP_CUDA(cudaMemsetAsync(max_len.data(), 0, sizeof(int32_t), s));
  row_lengths_kernel<<<blocks_for(rows), kThreads, 0, s>>>(rows, A.row_ptr(), len.data(), max_len.data());

  if (sigma == 1) {
    S.perm.resize(size_t(rows));
    iota_kernel<<<blocks_for(rows), kThreads, 0, s>>>(rows, S.perm.data());
  } else {
    DeviceBuffer<unsigned long long> keys(static_cast<size_t>(rows));
    DeviceBuffer<int32_t> idx(static_cast<size_t>(rows));
    sell_keys_kernel<<<blocks_for(rows), kThreads, 0, s>>>(rows, sigma, len.data(), keys.data(), idx.data());
    const long long windows = sigma == 0 ? 1 : (rows + sigma - 1) / sigma;
    int window_bits = 0;
    while ((1ll << window_bits) < windows) ++window_bits;
    S.perm = stable_sort_values(rows, keys, idx, 32 + window_bits, s);
  }

  DeviceBuffer<long long> slots(static_cast<size_t>(S.slices) + 1), offsets(static_cast<size_t>(S.slices) + 1);
  RASP_CUDA(cudaMemsetAsync(slots.data() + S.slices, 0, sizeof(long long), s));
  sell_width_kernel<<<blocks_for(S.slices), kThreads, 0, s>>>(rows, S.slices, C, S.perm.data(), len.data(),
                                                              S.slice_width.data(), slots.data());
  size_t temp_bytes = 0;
  RASP_CUDA(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, slots.data(), offsets.data(), S.slices + 1, s));
  DeviceBuffer<char> temp(std::max<size_t>(temp_bytes, 1));
  RASP_CUDA(cub::DeviceScan::ExclusiveSum(temp.data(), temp_bytes, slots.data(), offsets.data(), S.slices + 1, s));
  long long total = 0;
  RASP_CUDA(cudaMemcpyAsync(&total, offsets.data() + S.slices, sizeof(long long), cudaMemcpyDeviceToHost, s));
  RASP_CUDA(cudaStreamSynchronize(s));
  S.stored = total;
  check_fill("SELL", S.stored, A.nnz(), max_fill, sizeof(T));
  to_int32_kernel<<<blocks_for(S.slices + 1), kThreads, 0, s>>>(S.slices + 1, offsets.data(), S.slice_ptr.data());

  S.col.resize(size_t(S.stored));
  S.val.resize(size_t(S.stored));
  if (S.stored) {
    RASP_CUDA(cudaMemsetAsync(S.col.data(), 0xff, S.col.bytes(), s));
    RASP_CUDA(cudaMemsetAsync(S.val.data(), 0, S.val.bytes(), s));
    sell_fill_kernel<T><<<blocks_for(rows), kThreads, 0, s>>>(rows, C, S.perm.data(), S.slice_ptr.data(), A.row_ptr(),
                                                               A.col_idx(), A.vals(), S.col.data(), S.val.data());
  }
  RASP_CUDA(cudaStreamSynchronize(s));
  RASP_CUDA(cudaGetLastError());
  return S;
}

template <class T>
ArgCsrDevice<T> convert_argcsr_gpu(const DeviceCsr<T>& A, int block, int desired_chunk, int local_rows,
                                   double max_fill, cudaStream_t s) {
  std::vector<int32_t> row_ptr(size_t(A.rows()) + 1, 0);
  if (A.rows())
    RASP_CUDA(cudaMemcpyAsync(row_ptr.data(), A.row_ptr(), row_ptr.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, s));
  RASP_CUDA(cudaStreamSynchronize(s));
  ArgCsrLayout L = local_rows > 0 ? argcsr_local_layout(row_ptr, block, local_rows, 4096, sizeof(T), max_fill)
                                  : argcsr_layout(row_ptr, block, desired_chunk, sizeof(T), max_fill);
  ArgCsrDevice<T> G;
  G.groups = int32_t(L.groups.size());
  G.block = block;
  G.stored = L.stored;
  G.group_info.upload(L.groups);
  G.threads_end.upload(L.threads_end);
  G.col.resize(size_t(L.stored));
  G.val.resize(size_t(L.stored));
  if (L.stored) {
    DeviceBuffer<int32_t> off, tb, fill;
    off.upload(L.row_offset);
    tb.upload(L.row_tbegin);
    fill.upload(L.row_fill);
    RASP_CUDA(cudaMemsetAsync(G.col.data(), 0xff, G.col.bytes(), s));
    RASP_CUDA(cudaMemsetAsync(G.val.data(), 0, G.val.bytes(), s));
    argcsr_fill_kernel<T><<<blocks_for(A.rows()), kThreads, 0, s>>>(A.rows(), block, A.row_ptr(), A.col_idx(), A.vals(),
                                                                     off.data(), tb.data(), fill.data(), G.col.data(),
                                                                     G.val.data());
    RASP_CUDA(cudaStreamSynchronize(s));  // before off/tb/fill go out of scope
  }
  RASP_CUDA(cudaGetLastError());
  return G;
}

template <class T>
BinnedDevice convert_binned_gpu(const DeviceCsr<T>& A, int t, cudaStream_t s) {
  if (t < 1) throw std::invalid_argument("convert_binned_gpu: elems_per_lane must be >= 1");
  const int rows = A.rows();
  BinnedDevice B;
  B.bin_row_start.assign(kBins + 1, 0);
  B.bin_block_start.assign(kBins + 1, 0);
  if (rows == 0) return B;
  DeviceBuffer<unsigned> keys(static_cast<size_t>(rows)), counts(kBins);
  DeviceBuffer<int32_t> idx(static_cast<size_t>(rows));
  RASP_CUDA(cudaMemsetAsync(counts.data(), 0, counts.bytes(), s));
  bin_kernel<<<blocks_for(rows), kThreads, 0, s>>>(rows, t, A.row_ptr(), keys.data(), idx.data(), counts.data());
  B.order = stable_sort_values(rows, keys, idx, 3, s);  // bins 0..6 fit in 3 bits
  unsigned h[kBins];
  RASP_CUDA(cudaMemcpyAsync(h, counts.data(), sizeof(h), cudaMemcpyDeviceToHost, s));
  RASP_CUDA(cudaStreamSynchronize(s));
  RASP_CUDA(cudaGetLastError());
  for (int k = 0; k < kBins; ++k) {
    B.bin_row_start[size_t(k) + 1] = B.bin_row_start[size_t(k)] + int32_t(h[k]);
    const int32_t rows_per_block = k == kBins - 1 ? 1 : kBinnedBlock >> k;
    B.bin_block_start[size_t(k) + 1] = B.bin_block_start[size_t(k)] + (int32_t(h[k]) + rows_per_block - 1) / rows_per_block;
  }
  B.blocks = B.bin_block_start[kBins];
  return B;
}

#define RASP_INSTANTIATE(T)                                                                                   \
  template EllDevice<T> upload_ell(const EllHost<T>&);                                                        \
  template SellDevice<T> upload_sell(const SellHost<T>&);                                                     \
  template ArgCsrDevice<T> upload_argcsr(const ArgCsrHost<T>&);                                               \
  template EllDevice<T> convert_ell_gpu(const DeviceCsr<T>&, double, cudaStream_t);                           \
  template SellDevice<T> convert_sell_gpu(const DeviceCsr<T>&, int, int, double, cudaStream_t);               \
  template ArgCsrDevice<T> convert_argcsr_gpu(const DeviceCsr<T>&, int, int, int, double, cudaStream_t);      \
  template BinnedDevice convert_binned_gpu(const DeviceCsr<T>&, int, cudaStream_t);
RASP_INSTANTIATE(float)
RASP_INSTANTIATE(double)
#undef RASP_INSTANTIATE

}  // namespace rasp
