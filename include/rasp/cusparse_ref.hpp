#pragma once
// cuSPARSE baselines. These are both the performance baseline every RASP kernel must beat and the
// GPU correctness reference.
#include <string>

#include "rasp/csr.hpp"

namespace rasp {

enum class SpmvAlg { CsrAlg1, CsrAlg2 };
const char* to_string(SpmvAlg a);

// y = A x with the cuSPARSE generic API. The constructor does all one-time setup (handle, descriptors,
// workspace, cusparseSpMV_preprocess); run() is the part to time. The vectors are bound at construction
// because the preprocessing step is tied to them.
template <class T>
class CusparseSpmv {
 public:
  CusparseSpmv(const DeviceCsr<T>& A, SpmvAlg alg, const T* x, T* y);
  ~CusparseSpmv();
  CusparseSpmv(const CusparseSpmv&) = delete;
  CusparseSpmv& operator=(const CusparseSpmv&) = delete;

  void run(cudaStream_t stream);
  double preprocess_ms() const { return preprocess_ms_; }  // one-time cost (G4 break-even analysis)
  size_t workspace_bytes() const { return buffer_.size(); }

 private:
  void release() noexcept;

  const DeviceCsr<T>& A_;
  T* y_;
  cusparseHandle_t handle_ = nullptr;
  cusparseConstSpMatDescr_t mat_ = nullptr;
  cusparseConstDnVecDescr_t vx_ = nullptr;
  cusparseDnVecDescr_t vy_ = nullptr;
  cusparseSpMVAlg_t alg_;
  DeviceBuffer<char> buffer_;
  double preprocess_ms_ = 0;
};

struct SpgemmInfo {
  double total_ms = 0;       // wall time of the whole cuSPARSE call sequence
  size_t buffer1_bytes = 0;  // workEstimation workspace
  size_t buffer2_bytes = 0;  // compute workspace
  int64_t nnz_c = 0;
};

// C = A B with cuSPARSE SpGEMM (CUSPARSE_SPGEMM_DEFAULT). Output columns are sorted.
template <class T>
DeviceCsr<T> cusparse_spgemm(const DeviceCsr<T>& A, const DeviceCsr<T>& B, SpgemmInfo* info = nullptr);

}  // namespace rasp
