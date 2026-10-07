#pragma once
// Host and device CSR matrices. Conventions used everywhere in RASP:
//   * 0-based indices, 32-bit row pointers and column indices (cuSPARSE CUSPARSE_INDEX_32I),
//   * columns sorted inside each row, no duplicate entries,
//   * nnz < 2^31.
#include <cstdint>
#include <utility>
#include <vector>

#include "rasp/common.hpp"

namespace rasp {

template <class T>
struct HostCsr {
  int32_t rows = 0, cols = 0;
  int64_t nnz = 0;
  std::vector<int32_t> row_ptr;  // rows + 1 entries
  std::vector<int32_t> col_idx;  // nnz entries
  std::vector<T> vals;           // nnz entries

  // Throws std::runtime_error if the conventions above are violated.
  void validate() const;
};

// Non-owning CSR: the same conventions, over memory owned by someone else (device, mapped pinned host memory,
// managed memory or plain host memory; the pointer kind is the caller's business). Used where one routine must
// work on all of them: the cuSPARSE SpGEMM core, the out-of-core executor and the SpGEMM planners.
template <class T>
struct CsrView {
  int32_t rows = 0, cols = 0;
  int64_t nnz = 0;
  const int32_t* row_ptr = nullptr;
  const int32_t* col_idx = nullptr;
  const T* vals = nullptr;
};

template <class T>
CsrView<T> host_view(const HostCsr<T>& h) {
  return {h.rows, h.cols, h.nnz, h.row_ptr.data(), h.col_idx.data(), h.vals.data()};
}

// Owning, move-only GPU buffer.
template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t n) { resize(n); }
  ~DeviceBuffer() { reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(std::exchange(o.ptr_, nullptr)), size_(std::exchange(o.size_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      reset();
      ptr_ = std::exchange(o.ptr_, nullptr);
      size_ = std::exchange(o.size_, 0);
    }
    return *this;
  }

  void resize(size_t n) {
    reset();
    if (n) RASP_CUDA(cudaMalloc(&ptr_, n * sizeof(T)));
    size_ = n;
  }
  void reset() noexcept {
    if (ptr_) cudaFree(ptr_);
    ptr_ = nullptr;
    size_ = 0;
  }
  void upload(const T* src, size_t n) {
    if (n != size_) resize(n);
    if (n) RASP_CUDA(cudaMemcpy(ptr_, src, n * sizeof(T), cudaMemcpyHostToDevice));
  }
  void upload(const std::vector<T>& v) { upload(v.data(), v.size()); }
  std::vector<T> download() const {
    std::vector<T> v(size_);
    if (size_) RASP_CUDA(cudaMemcpy(v.data(), ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  size_t size() const { return size_; }
  size_t bytes() const { return size_ * sizeof(T); }

 private:
  T* ptr_ = nullptr;
  size_t size_ = 0;
};

template <class T>
class DeviceCsr {
 public:
  DeviceCsr() = default;
  DeviceCsr(int32_t rows, int32_t cols, int64_t nnz, DeviceBuffer<int32_t> row_ptr, DeviceBuffer<int32_t> col_idx,
            DeviceBuffer<T> vals)
      : rows_(rows), cols_(cols), nnz_(nnz), row_ptr_(std::move(row_ptr)), col_idx_(std::move(col_idx)),
        vals_(std::move(vals)) {}

  static DeviceCsr upload(const HostCsr<T>& h) {
    DeviceCsr d;
    d.rows_ = h.rows;
    d.cols_ = h.cols;
    d.nnz_ = h.nnz;
    d.row_ptr_.upload(h.row_ptr);
    d.col_idx_.upload(h.col_idx);
    d.vals_.upload(h.vals);
    return d;
  }
  HostCsr<T> download() const {
    HostCsr<T> h;
    h.rows = rows_;
    h.cols = cols_;
    h.nnz = nnz_;
    h.row_ptr = row_ptr_.download();
    h.col_idx = col_idx_.download();
    h.vals = vals_.download();
    return h;
  }

  int32_t rows() const { return rows_; }
  int32_t cols() const { return cols_; }
  int64_t nnz() const { return nnz_; }
  const int32_t* row_ptr() const { return row_ptr_.data(); }
  const int32_t* col_idx() const { return col_idx_.data(); }
  const T* vals() const { return vals_.data(); }
  size_t bytes() const { return row_ptr_.bytes() + col_idx_.bytes() + vals_.bytes(); }
  CsrView<T> view() const { return {rows_, cols_, nnz_, row_ptr(), col_idx(), vals()}; }

 private:
  int32_t rows_ = 0, cols_ = 0;
  int64_t nnz_ = 0;
  DeviceBuffer<int32_t> row_ptr_, col_idx_;
  DeviceBuffer<T> vals_;
};

}  // namespace rasp
