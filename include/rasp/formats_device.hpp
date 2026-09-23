#pragma once
// Device-resident storage formats, built either by uploading a host format (formats.hpp) or directly on
// the GPU from a DeviceCsr. The GPU converters produce bit-identical arrays to the host converters (tested),
// so the host versions remain the readable reference.
#include <cuda_runtime.h>

#include "rasp/formats.hpp"

namespace rasp {

template <class T>
struct EllDevice {
  int32_t rows = 0, width = 0;
  int64_t stored = 0;
  DeviceBuffer<int32_t> col;
  DeviceBuffer<T> val;
};

template <class T>
struct SellDevice {
  int32_t rows = 0, C = 0, slices = 0;
  int64_t stored = 0;
  DeviceBuffer<int32_t> slice_ptr, slice_width, perm, col;
  DeviceBuffer<T> val;
};

template <class T>
struct ArgCsrDevice {
  int32_t groups = 0, block = 0;
  int64_t stored = 0;
  DeviceBuffer<ArgGroup> group_info;
  DeviceBuffer<int32_t> threads_end, col;
  DeviceBuffer<T> val;
};

struct BinnedDevice {
  int32_t blocks = 0;
  std::vector<int32_t> bin_row_start, bin_block_start;  // kBins + 1 entries each (tiny, kept on the host)
  DeviceBuffer<int32_t> order;
};

// Host format -> device.
template <class T> EllDevice<T> upload_ell(const EllHost<T>& h);
template <class T> SellDevice<T> upload_sell(const SellHost<T>& h);
template <class T> ArgCsrDevice<T> upload_argcsr(const ArgCsrHost<T>& h);
BinnedDevice upload_binned(const BinnedHost& h);

// DeviceCsr -> device format, on the GPU. Same limits (FormatInfeasible) as the host converters. All work is
// enqueued on `stream`, which is synchronised before returning (sizes are needed on the host to allocate).
// ArgCSR downloads row_ptr only: its layout step is sequential and needs row lengths alone, while every
// nonzero is moved on the GPU.
template <class T> EllDevice<T> convert_ell_gpu(const DeviceCsr<T>& A, double max_fill = 3.0, cudaStream_t stream = nullptr);
template <class T> SellDevice<T> convert_sell_gpu(const DeviceCsr<T>& A, int C, int sigma, double max_fill = 64.0,
                                                  cudaStream_t stream = nullptr);
template <class T> ArgCsrDevice<T> convert_argcsr_gpu(const DeviceCsr<T>& A, int block, int desired_chunk,
                                                      int local_rows = 0, double max_fill = 64.0,
                                                      cudaStream_t stream = nullptr);
template <class T> BinnedDevice convert_binned_gpu(const DeviceCsr<T>& A, int elems_per_lane, cudaStream_t stream = nullptr);

}  // namespace rasp
