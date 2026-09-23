#pragma once
// Common interface for every SpMV implementation (cuSPARSE and ours), so benchmarks and the selector can
// treat them uniformly. Like CusparseSpmv, a kernel is bound to its x and y at construction; construction
// does all conversion and upload (timed as convert_ms), and run() is the part to time.
#include <memory>
#include <string>
#include <vector>

#include "rasp/csr.hpp"

namespace rasp {

enum class Format { Cusparse, CsrScalar, CsrVector, Ell, Sell, ArgCsr, CsrBinned, ArgCsrLocal };

struct KernelConfig {
  Format format = Format::CsrScalar;
  int alg = 1;     // cusparse: 1 or 2
  int v = 0;       // csr_vector: threads per row (2..32, power of two)
  int C = 0;       // sell: slice height (8 or 32)
  int sigma = 0;   // sell: sorting window (1 = none, 0 = all rows)
  int block = 0;   // argcsr, argcsr_local: threads (chunks) per group
  int chunk = 0;   // argcsr: desiredChunkSize
  int t = 0;       // csr_binned: target elements per lane
  int rows = 0;    // argcsr_local: target rows per group

  std::string family() const;  // "argcsr"
  std::string label() const;   // "argcsr_B128_D1" (unique, CSV/filename safe)
  static KernelConfig parse(const std::string& label);  // inverse of label(); throws on unknown labels
};

// All Phase 2 configurations: cuSPARSE ALG1/2, CSR-scalar, CSR-vector v=2..32, ELL,
// SELL C={8,32} x sigma={1,256,all}, ArgCSR block={64,128,256} x desiredChunkSize={1,2,4,8,16,32}.
std::vector<KernelConfig> phase2_configs();

// The candidate-set extensions (Phase 3a): CSR-binned t={1,2,4} and ArgCSR-local B={128,256} x R={32,64}.
std::vector<KernelConfig> extension_configs();

// phase2_configs() + extension_configs().
std::vector<KernelConfig> all_configs();

template <class T>
class SpmvKernel {
 public:
  virtual ~SpmvKernel() = default;
  virtual void run(cudaStream_t stream) = 0;  // y = A x
  double convert_ms() const { return convert_ms_; }   // host conversion + upload (+ cuSPARSE preprocess)
  size_t format_bytes() const { return format_bytes_; }  // device memory of the format (+ workspace)
  int64_t stored_slots() const { return stored_slots_; }  // stored entries incl. padding (nnz for CSR)

 protected:
  double convert_ms_ = 0;
  size_t format_bytes_ = 0;
  int64_t stored_slots_ = 0;
};

// Builds the kernel for `cfg`. `host` and `dev` are the same matrix; CSR kernels use `dev` directly (which must
// outlive the kernel), padded formats convert from `host`. Throws FormatInfeasible if the format would need too
// much padding for this matrix.
template <class T>
std::unique_ptr<SpmvKernel<T>> make_spmv_kernel(const KernelConfig& cfg, const HostCsr<T>& host,
                                                const DeviceCsr<T>& dev, const T* x, T* y);

}  // namespace rasp
