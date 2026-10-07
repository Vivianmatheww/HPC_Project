#pragma once
// Page-locked ("pinned") host memory: the slow-memory tier of the SpGEMM experiments, as host-pinned DDR was in
// paper 1. Buffers are allocated mapped and portable, so a kernel can also read or write them in place over PCIe
// (zero-copy) through device_ptr().
//
// Allocating pinned memory is slow (every page is locked) and may synchronise the device, so allocate during setup,
// never inside a timed region. WSL2 limits how much memory can be pinned; the Phase 0 probe pinned 4 GiB without
// trouble.
#include <cstring>
#include <utility>

#include "rasp/csr.hpp"

namespace rasp {

// Owning, move-only pinned host buffer (the host-side twin of DeviceBuffer).
template <class T>
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  explicit PinnedBuffer(size_t n) { resize(n); }
  ~PinnedBuffer() { reset(); }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  PinnedBuffer(PinnedBuffer&& o) noexcept
      : ptr_(std::exchange(o.ptr_, nullptr)), dev_(std::exchange(o.dev_, nullptr)), size_(std::exchange(o.size_, 0)) {}
  PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
    if (this != &o) {
      reset();
      ptr_ = std::exchange(o.ptr_, nullptr);
      dev_ = std::exchange(o.dev_, nullptr);
      size_ = std::exchange(o.size_, 0);
    }
    return *this;
  }

  // Contents are not preserved.
  void resize(size_t n) {
    reset();
    if (n) {
      void* p = nullptr;
      RASP_CUDA(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable));
      ptr_ = static_cast<T*>(p);
      void* d = nullptr;
      const cudaError_t e = cudaHostGetDevicePointer(&d, p, 0);
      if (e != cudaSuccess) {
        cudaFreeHost(p);
        ptr_ = nullptr;
        throw_cuda(e, "cudaHostGetDevicePointer", __FILE__, __LINE__);
      }
      dev_ = static_cast<T*>(d);
    }
    size_ = n;
  }
  void reset() noexcept {
    if (ptr_) cudaFreeHost(ptr_);
    ptr_ = dev_ = nullptr;
    size_ = 0;
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  T* device_ptr() const { return dev_; }  // the same memory as seen by kernels (zero-copy)
  size_t size() const { return size_; }
  size_t bytes() const { return size_ * sizeof(T); }

 private:
  T* ptr_ = nullptr;
  T* dev_ = nullptr;
  size_t size_ = 0;
};

// CSR in pinned host memory: the inputs and the output of the out-of-core SpGEMM. col_idx and vals may be larger
// than nnz (capacity), because an SpGEMM output is allocated at its upper bound before its real size is known.
template <class T>
struct PinnedCsr {
  int32_t rows = 0, cols = 0;
  int64_t nnz = 0;
  PinnedBuffer<int32_t> row_ptr;  // rows + 1 entries
  PinnedBuffer<int32_t> col_idx;  // capacity() >= nnz entries
  PinnedBuffer<T> vals;           // same capacity

  static PinnedCsr from(const HostCsr<T>& h) {
    PinnedCsr p;
    p.reserve(h.rows, h.cols, h.nnz);
    p.nnz = h.nnz;
    std::memcpy(p.row_ptr.data(), h.row_ptr.data(), h.row_ptr.size() * sizeof(int32_t));
    if (h.nnz) {
      std::memcpy(p.col_idx.data(), h.col_idx.data(), size_t(h.nnz) * sizeof(int32_t));
      std::memcpy(p.vals.data(), h.vals.data(), size_t(h.nnz) * sizeof(T));
    }
    return p;
  }

  // Allocates for `rows` x `cols` with room for `nnz_capacity` entries and sets nnz = 0 (an empty output). Keeps
  // the existing buffers if they are already large enough, so a benchmark can reuse one output across repetitions.
  void reserve(int32_t r, int32_t c, int64_t nnz_capacity) {
    if (row_ptr.size() != size_t(r) + 1) row_ptr.resize(size_t(r) + 1);
    if (col_idx.size() < size_t(nnz_capacity)) {
      col_idx.resize(size_t(nnz_capacity));
      vals.resize(size_t(nnz_capacity));
    }
    rows = r;
    cols = c;
    nnz = 0;
    row_ptr.data()[0] = 0;
  }

  HostCsr<T> to_host() const {
    HostCsr<T> h;
    h.rows = rows;
    h.cols = cols;
    h.nnz = nnz;
    h.row_ptr.assign(row_ptr.data(), row_ptr.data() + size_t(rows) + 1);
    h.col_idx.assign(col_idx.data(), col_idx.data() + nnz);
    h.vals.assign(vals.data(), vals.data() + nnz);
    return h;
  }

  CsrView<T> view() const { return {rows, cols, nnz, row_ptr.data(), col_idx.data(), vals.data()}; }  // host
  CsrView<T> mapped_view() const {  // device pointers to the same memory (zero-copy)
    return {rows, cols, nnz, row_ptr.device_ptr(), col_idx.device_ptr(), vals.device_ptr()};
  }
  size_t bytes() const { return (size_t(rows) + 1) * sizeof(int32_t) + size_t(nnz) * (sizeof(int32_t) + sizeof(T)); }
  size_t capacity() const { return col_idx.size(); }
};

}  // namespace rasp
