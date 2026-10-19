#include <algorithm>
#include <chrono>
#include <limits>

#include "rasp/cusparse_ref.hpp"

namespace rasp {
namespace {

template <class T>
constexpr cudaDataType_t cuda_type();
template <>
constexpr cudaDataType_t cuda_type<float>() { return CUDA_R_32F; }
template <>
constexpr cudaDataType_t cuda_type<double>() { return CUDA_R_64F; }

template <class T>
cusparseConstSpMatDescr_t make_const_csr(const CsrView<T>& A) {
  cusparseConstSpMatDescr_t d = nullptr;
  RASP_CUSPARSE(cusparseCreateConstCsr(&d, A.rows, A.cols, A.nnz, A.row_ptr, A.col_idx, A.vals, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, cuda_type<T>()));
  return d;
}

}  // namespace

const char* to_string(SpmvAlg a) { return a == SpmvAlg::CsrAlg1 ? "csr_alg1" : "csr_alg2"; }

// ---- SpMV ------------------------------------------------------------------------------------
template <class T>
CusparseSpmv<T>::CusparseSpmv(const DeviceCsr<T>& A, SpmvAlg alg, const T* x, T* y)
    : A_(A), y_(y), alg_(alg == SpmvAlg::CsrAlg1 ? CUSPARSE_SPMV_CSR_ALG1 : CUSPARSE_SPMV_CSR_ALG2) {
  if (A.nnz() == 0) return;  // run() just zeroes y; cuSPARSE is not asked about empty matrices
  try {
    RASP_CUSPARSE(cusparseCreate(&handle_));
    mat_ = make_const_csr(A.view());
    RASP_CUSPARSE(cusparseCreateConstDnVec(&vx_, A.cols(), x, cuda_type<T>()));
    RASP_CUSPARSE(cusparseCreateDnVec(&vy_, A.rows(), y, cuda_type<T>()));

    const T alpha = 1, beta = 0;
    size_t bytes = 0;
    RASP_CUSPARSE(cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_, vx_, &beta, vy_,
                                          cuda_type<T>(), alg_, &bytes));
    buffer_.resize(std::max<size_t>(bytes, 16));

    cudaEvent_t e0, e1;
    RASP_CUDA(cudaEventCreate(&e0));
    RASP_CUDA(cudaEventCreate(&e1));
    RASP_CUDA(cudaEventRecord(e0));
#if CUDART_VERSION >= 12040  // cusparseSpMV_preprocess arrived in CUDA 12.4; before it, nothing to preprocess
    RASP_CUSPARSE(cusparseSpMV_preprocess(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_, vx_, &beta, vy_,
                                          cuda_type<T>(), alg_, buffer_.data()));
#endif
    RASP_CUDA(cudaEventRecord(e1));
    RASP_CUDA(cudaEventSynchronize(e1));
    float ms = 0;
    RASP_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    preprocess_ms_ = ms;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
  } catch (...) {
    release();
    throw;
  }
}

template <class T>
CusparseSpmv<T>::~CusparseSpmv() {
  release();
}

template <class T>
void CusparseSpmv<T>::release() noexcept {
  if (vy_) cusparseDestroyDnVec(vy_);
  if (vx_) cusparseDestroyDnVec(vx_);
  if (mat_) cusparseDestroySpMat(mat_);
  if (handle_) cusparseDestroy(handle_);
  vy_ = nullptr;
  vx_ = nullptr;
  mat_ = nullptr;
  handle_ = nullptr;
}

template <class T>
void CusparseSpmv<T>::run(cudaStream_t stream) {
  if (A_.nnz() == 0) {
    if (A_.rows()) RASP_CUDA(cudaMemsetAsync(y_, 0, size_t(A_.rows()) * sizeof(T), stream));
    return;
  }
  const T alpha = 1, beta = 0;
  RASP_CUSPARSE(cusparseSetStream(handle_, stream));
  RASP_CUSPARSE(cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_, vx_, &beta, vy_, cuda_type<T>(),
                             alg_, buffer_.data()));
}

// ---- SpGEMM ----------------------------------------------------------------------------------
template <class T>
SpgemmCall cusparse_spgemm_view(cusparseHandle_t handle, const CsrView<T>& A, const CsrView<T>& B, int32_t* c_row_ptr,
                                const SpgemmAlloc& alloc, cudaStream_t stream) {
  if (A.cols != B.rows) throw std::invalid_argument("cusparse_spgemm: A.cols != B.rows");
  SpgemmCall call;
  const size_t rp_bytes = (size_t(A.rows) + 1) * sizeof(int32_t);
  if (A.nnz == 0 || B.nnz == 0) {  // empty product: all-zero row pointers
    RASP_CUDA(cudaMemsetAsync(c_row_ptr, 0, rp_bytes, stream));
    call.complete = true;
    return call;
  }

  cusparseConstSpMatDescr_t matA = nullptr, matB = nullptr;
  cusparseSpMatDescr_t matC = nullptr;
  cusparseSpGEMMDescr_t desc = nullptr;
  auto cleanup = [&] {
    if (desc) cusparseSpGEMM_destroyDescr(desc);
    if (matC) cusparseDestroySpMat(matC);
    if (matB) cusparseDestroySpMat(matB);
    if (matA) cusparseDestroySpMat(matA);
  };

  try {
    RASP_CUSPARSE(cusparseSetStream(handle, stream));
    matA = make_const_csr(A);
    matB = make_const_csr(B);
    RASP_CUSPARSE(cusparseCreateCsr(&matC, A.rows, B.cols, 0, c_row_ptr, nullptr, nullptr, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, cuda_type<T>()));
    RASP_CUSPARSE(cusparseSpGEMM_createDescr(&desc));

    const T alpha = 1, beta = 0;
    const auto op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const auto alg = CUSPARSE_SPGEMM_DEFAULT;
    const auto type = cuda_type<T>();

    size_t b1 = 0, b2 = 0;
    RASP_CUSPARSE(cusparseSpGEMM_workEstimation(handle, op, op, &alpha, matA, matB, &beta, matC, type, alg, desc, &b1,
                                                nullptr));
    call.buffer1_bytes = b1;
    void* buf1 = alloc(SpgemmBuffer::Work1, std::max<size_t>(b1, 16));
    if (!buf1) {
      cleanup();
      return call;
    }
    RASP_CUSPARSE(cusparseSpGEMM_workEstimation(handle, op, op, &alpha, matA, matB, &beta, matC, type, alg, desc, &b1,
                                                buf1));
    RASP_CUSPARSE(cusparseSpGEMM_compute(handle, op, op, &alpha, matA, matB, &beta, matC, type, alg, desc, &b2, nullptr));
    call.buffer2_bytes = b2;
    void* buf2 = alloc(SpgemmBuffer::Work2, std::max<size_t>(b2, 16));
    if (!buf2) {
      cleanup();
      return call;
    }
    RASP_CUSPARSE(cusparseSpGEMM_compute(handle, op, op, &alpha, matA, matB, &beta, matC, type, alg, desc, &b2, buf2));

    int64_t rows = 0, cols = 0, nnz_c = 0;
    RASP_CUSPARSE(cusparseSpMatGetSize(matC, &rows, &cols, &nnz_c));
    if (nnz_c >= std::numeric_limits<int32_t>::max()) throw std::runtime_error("cusparse_spgemm: nnz(C) exceeds 32-bit");
    call.nnz_c = nnz_c;
    if (nnz_c == 0) {  // nonzeros in A and B, but none meet (e.g. A's columns only hit empty rows of B)
      RASP_CUDA(cudaMemsetAsync(c_row_ptr, 0, rp_bytes, stream));
    } else {
      call.c_col = static_cast<int32_t*>(alloc(SpgemmBuffer::CCol, size_t(nnz_c) * sizeof(int32_t)));
      if (call.c_col) call.c_val = alloc(SpgemmBuffer::CVal, size_t(nnz_c) * sizeof(T));
      if (!call.c_col || !call.c_val) {
        cleanup();
        return call;
      }
      RASP_CUSPARSE(cusparseCsrSetPointers(matC, c_row_ptr, call.c_col, call.c_val));
      RASP_CUSPARSE(cusparseSpGEMM_copy(handle, op, op, &alpha, matA, matB, &beta, matC, type, alg, desc));
    }
    call.complete = true;
    cleanup();
    return call;
  } catch (...) {
    cleanup();
    throw;
  }
}

template <class T>
DeviceCsr<T> cusparse_spgemm(const DeviceCsr<T>& A, const DeviceCsr<T>& B, SpgemmInfo* info) {
  if (A.cols() != B.rows()) throw std::invalid_argument("cusparse_spgemm: A.cols != B.rows");
  const auto t0 = std::chrono::steady_clock::now();
  SpgemmInfo local;
  SpgemmInfo& inf = info ? *info : local;
  inf = SpgemmInfo{};

  DeviceBuffer<int32_t> c_row_ptr(size_t(A.rows()) + 1), c_col;
  DeviceBuffer<T> c_val;
  DeviceBuffer<char> buf1, buf2;
  auto alloc = [&](SpgemmBuffer what, size_t bytes) -> void* {
    switch (what) {
      case SpgemmBuffer::Work1: buf1.resize(bytes); return buf1.data();
      case SpgemmBuffer::Work2: buf2.resize(bytes); return buf2.data();
      case SpgemmBuffer::CCol: c_col.resize(bytes / sizeof(int32_t)); return c_col.data();
      case SpgemmBuffer::CVal: c_val.resize(bytes / sizeof(T)); return c_val.data();
    }
    return nullptr;
  };

  cusparseHandle_t handle = nullptr;
  RASP_CUSPARSE(cusparseCreate(&handle));
  SpgemmCall call;
  try {
    call = cusparse_spgemm_view(handle, A.view(), B.view(), c_row_ptr.data(), alloc, nullptr);
    RASP_CUDA(cudaDeviceSynchronize());
  } catch (...) {
    cusparseDestroy(handle);
    throw;
  }
  cusparseDestroy(handle);

  inf.buffer1_bytes = call.buffer1_bytes;
  inf.buffer2_bytes = call.buffer2_bytes;
  inf.nnz_c = call.nnz_c;
  inf.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return DeviceCsr<T>(A.rows(), B.cols(), call.nnz_c, std::move(c_row_ptr), std::move(c_col), std::move(c_val));
}

template class CusparseSpmv<float>;
template class CusparseSpmv<double>;
template DeviceCsr<float> cusparse_spgemm(const DeviceCsr<float>&, const DeviceCsr<float>&, SpgemmInfo*);
template DeviceCsr<double> cusparse_spgemm(const DeviceCsr<double>&, const DeviceCsr<double>&, SpgemmInfo*);
template SpgemmCall cusparse_spgemm_view(cusparseHandle_t, const CsrView<float>&, const CsrView<float>&, int32_t*,
                                         const SpgemmAlloc&, cudaStream_t);
template SpgemmCall cusparse_spgemm_view(cusparseHandle_t, const CsrView<double>&, const CsrView<double>&, int32_t*,
                                         const SpgemmAlloc&, cudaStream_t);

}  // namespace rasp
