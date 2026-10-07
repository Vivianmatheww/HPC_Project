#pragma once
// Out-of-core SpGEMM planning: everything that decides *how* C = A B is computed under a fast-memory budget, as
// plain host code (no CUDA calls) so it is unit-tested on any machine. SpgemmContext (spgemm.hpp) executes the plans.
//
//   estimate()     per-row work and size bounds from the row pointers of A and B and A's column indices
//   split_rows()   paper 1's BinarySearch: longest consecutive row ranges under a size cap
//   simulate()     predicted wall time of the executor's issue order on one copy engine
//   plan_alg4()    paper 1, Alg. 4: 75/25 split of fast memory, variant chosen by copy volume (the baseline)
//   plan_rasp()    ours: mode, partitions and pipeline depth chosen by simulating candidates with measured rates
//   replan_rows()  re-partition the rows not yet issued, after measurements changed the parameters
//
// Units: bytes, milliseconds, "flops" = multiply-adds (GFLOP/s in reports counts 2 per multiply-add).
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rasp/csr.hpp"

namespace rasp {

enum class SpgemmMode {
  InCore,    // A, B and C all inside the budget
  ZeroCopy,  // B inside the budget; A read and C written in place in pinned host memory (row chunks bound the workspace)
  ChunkB,    // paper 1's chunk2: B row block resident, A/C row blocks streamed (one B block: the 1-D case)
  ChunkAC,   // paper 1's chunk1: A/C row block resident, B row blocks streamed, partial C summed on the device
};
const char* to_string(SpgemmMode m);

// Thrown when no partition fits the budget (a single row, or B's row pointers, are larger than the room left).
struct BudgetTooSmall : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// std::vector whose resize() leaves elements uninitialised: the estimator writes every element in parallel, and a
// zero-fill first would be one more sequential pass over hundreds of MB.
template <class T>
struct NoInitAllocator : std::allocator<T> {
  template <class U>
  struct rebind {
    using other = NoInitAllocator<U>;
  };
  NoInitAllocator() = default;
  template <class U>
  NoInitAllocator(const NoInitAllocator<U>&) noexcept {}
  template <class U, class... Args>
  void construct(U* p, Args&&... args) {
    if constexpr (sizeof...(Args) == 0) ::new (static_cast<void*>(p)) U;
    else ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
  }
};
using PrefixVec = std::vector<int64_t, NoInitAllocator<int64_t>>;

// What the planners know about C = A B. Row i of C has at most min(flops_i, b_cols) nonzeros, where
// flops_i = sum over k in A(i,:) of nnz(B(k,:)) (the "flop count" of paper 1).
struct ProblemSizes {
  int32_t a_rows = 0, a_cols = 0, b_cols = 0;  // B is a_cols x b_cols
  size_t value_bytes = 8;
  const int32_t* a_ptr = nullptr;  // A's row pointers (a_rows + 1), not owned: A must outlive the sizes
  const int32_t* b_ptr = nullptr;  // B's row pointers (a_cols + 1), not owned
  PrefixVec flops_ptr;             // prefix sums of flops_i (a_rows + 1)
  PrefixVec cub_ptr;               // prefix sums of min(flops_i, b_cols): upper bound on nnz(C) (a_rows + 1)

  int64_t flops(int32_t r0, int32_t r1) const { return flops_ptr[size_t(r1)] - flops_ptr[size_t(r0)]; }
  int64_t c_upper(int32_t r0, int32_t r1) const { return cub_ptr[size_t(r1)] - cub_ptr[size_t(r0)]; }
  int64_t a_nnz(int32_t r0, int32_t r1) const { return a_ptr[size_t(r1)] - a_ptr[size_t(r0)]; }
  int64_t b_nnz(int32_t k0, int32_t k1) const { return b_ptr[size_t(k1)] - b_ptr[size_t(k0)]; }
  double csr_bytes(int64_t rows, double nnz) const {
    return double(rows + 1) * 4.0 + nnz * double(4 + value_bytes);
  }
  double a_bytes(int32_t r0, int32_t r1) const { return csr_bytes(r1 - r0, double(a_nnz(r0, r1))); }
  double b_bytes(int32_t k0, int32_t k1) const { return csr_bytes(k1 - k0, double(b_nnz(k0, k1))); }
  double c_bytes(int32_t r0, int32_t r1, double gamma) const { return csr_bytes(r1 - r0, gamma * double(c_upper(r0, r1))); }
  // B row block [k0, k1) as a full-height matrix: its own rows plus row pointers for every row of B (the others empty).
  // This is how A's columns outside the block are skipped without splitting A (paper 1, Section 4.2).
  double ext_bytes() const { return double(a_cols + 1) * 4.0; }
};

// Host pointers; A and B must outlive the result. Parallel over rows with OpenMP when available. The second form
// reuses `out`'s arrays (no page faults on repeated calls of the same size).
template <class T>
ProblemSizes estimate(const CsrView<T>& A, const CsrView<T>& B);
template <class T>
void estimate(const CsrView<T>& A, const CsrView<T>& B, ProblemSizes& out);

// Model parameters. Bandwidths come from the context's calibration; the rest start from these priors and are
// replaced by measurements after every chunk (adaptive runs).
struct CostParams {
  double h2d_gbs = 13.4, d2h_gbs = 13.2;  // pinned PCIe copies (measured on the RTX 4050 laptop, progress.md)
  double device_gbs = 150;                // traffic of our small device kernels (partial-C sums)
  double flops_per_ms = 1e6;              // cuSPARSE SpGEMM multiply-adds per ms
  double tile_overhead_ms = 0.3;          // per cuSPARSE call: descriptors and four synchronising size queries
  double gamma = 1.0;                     // nnz(C) / upper bound used to size C (1 = upper bound, always safe)
  double ws_bytes_per_flop = 24;          // cuSPARSE workspace (buffer1 + buffer2) per multiply-add ...
  double ws_base_bytes = 1 << 20;         // ... plus a constant
  double zc_passes_a = 3, zc_passes_c = 1;  // zero-copy: PCIe passes over A and C (from the placement study)
};

// Paper 1's BinarySearch: splits rows [begin, end) into consecutive ranges, each as long as possible with
// size(r0, r1) <= cap. `size` must be non-decreasing in r1. Returns the boundaries begin, ..., end ({begin} if the
// range is empty). Throws BudgetTooSmall if one row alone exceeds cap.
std::vector<int32_t> split_rows(int32_t begin, int32_t end, const std::function<double(int32_t, int32_t)>& size,
                                double cap);

// One pipeline stage as the executor issues it: upload, compute, download.
struct StageCost {
  double in_ms = 0, comp_ms = 0, out_ms = 0;
};
// Wall time of the executor's issue order: with depth d, uploads run d - 1 stages ahead of the compute; all copies
// share one copy engine in issue order (the RTX 4050 has asyncEngineCount = 1); the host issues stage k + d - 1 when
// compute k - 1 has finished (it blocks in cuSPARSE's size queries). Depth 1 runs every stage strictly in sequence.
double simulate(const std::vector<StageCost>& stages, int depth);

struct SpgemmPlan {
  SpgemmMode mode = SpgemmMode::InCore;
  std::vector<int32_t> ac_bounds;  // A/C row partition boundaries: 0, ..., a_rows
  std::vector<int32_t> b_bounds;   // B row partition boundaries: 0, ..., a_cols
  int depth = 1;                   // buffers for the streamed operand (1 = no overlap)
  bool probe = false;              // the first A/C block is a small calibration probe
  CostParams params;               // the parameters the plan was made with
  double predicted_ms = 0;
  double copy_bytes = 0;           // predicted H2D + D2H volume
  double peak_bytes = 0;           // predicted device footprint (the executor enforces the real one)
  std::string reason;

  int parts_ac() const { return ac_bounds.empty() ? 0 : int(ac_bounds.size()) - 1; }
  int parts_b() const { return b_bounds.empty() ? 0 : int(b_bounds.size()) - 1; }
  std::string to_string() const;
};

// Predicted time and footprint of a given partition (used by both planners, and by the executor to report what a
// re-plan expects).
void predict(const ProblemSizes& s, SpgemmPlan& plan);

// Paper 1, Alg. 4 (our baseline): in-core if everything fits; otherwise 75% of the budget for the resident operand
// and 25% for the streamed one (partitions by BinarySearch), and the variant with the smaller copy volume:
//   chunk1 (A/C resident): |A| + |C| + |B| * P_AC        chunk2 (B resident): |B| + |A| * P_B + |C| * (P_B - 1)
// Partial-C buffers are charged to the A/C share and cuSPARSE's workspace to the streamed share. depth = 1.
SpgemmPlan plan_alg4(const ProblemSizes& s, double budget, const CostParams& p);

// RASP: simulates every feasible candidate (in-core; ChunkB 1-D and multi-block; ChunkAC; zero-copy) over a range of
// partition sizes and depths, using all memory the resident operand leaves free, and returns the fastest.
// depth = 0 lets the planner choose (2 or 3); probe adds a small first block to the 1-D ChunkB plan.
struct RaspPlanOptions {
  int depth = 0;
  bool probe = true;
  bool allow_zero_copy = true;
  bool has_forced_mode = false;
  SpgemmMode forced_mode = SpgemmMode::InCore;
};
SpgemmPlan plan_rasp(const ProblemSizes& s, double budget, const CostParams& p, const RaspPlanOptions& o = {});

// Re-partitions A/C rows [r_begin, a_rows) of `plan` (same mode, depth and B partition) with new parameters, choosing
// the block size by simulation. Returns the new boundaries r_begin, ..., a_rows.
std::vector<int32_t> replan_rows(const ProblemSizes& s, double budget, const SpgemmPlan& plan, const CostParams& p,
                                 int32_t r_begin);

// True if a measured rate, compression or workspace ratio differs from what the plan assumed by more than threshold.
bool needs_replan(const CostParams& planned, const CostParams& measured, double threshold);

}  // namespace rasp
