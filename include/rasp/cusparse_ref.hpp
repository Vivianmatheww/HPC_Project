#pragma once
// cuSPARSE baselines. These are both the performance baseline every RASP kernel must beat and the
// GPU correctness reference.
#include <functional>
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

// ---- Low-level SpGEMM on views: the building block of the out-of-core executor and the placement study ----------

// The buffers the core asks for: cuSPARSE's two workspaces, then C's column indices and values once nnz(C) is known.
enum class SpgemmBuffer { Work1, Work2, CCol, CVal };

// Returns at least `bytes` bytes (device, mapped pinned or managed memory: the caller decides) or nullptr if it
// cannot. The caller owns everything it hands out; the core never frees.
using SpgemmAlloc = std::function<void*(SpgemmBuffer what, size_t bytes)>;

struct SpgemmCall {
  bool complete = false;  // false: an allocation failed (the sizes below say how far the call got)
  int64_t nnz_c = 0;
  size_t buffer1_bytes = 0, buffer2_bytes = 0;
  int32_t* c_col = nullptr;  // as returned by alloc; nullptr when nnz_c == 0
  void* c_val = nullptr;
};

// C = A B (CUSPARSE_SPGEMM_DEFAULT) on views. Writes c_row_ptr (A.rows + 1 entries, caller-provided) and asks
// `alloc` for everything else. All work is enqueued on `stream` through `handle`; cuSPARSE's size queries
// synchronise that stream only. An empty product (no nonzeros in A, B or the result) only zeroes c_row_ptr.
template <class T>
SpgemmCall cusparse_spgemm_view(cusparseHandle_t handle, const CsrView<T>& A, const CsrView<T>& B, int32_t* c_row_ptr,
                                const SpgemmAlloc& alloc, cudaStream_t stream);

}  // namespace rasp
