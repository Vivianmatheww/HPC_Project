#pragma once
// Out-of-core SpGEMM (Phase 4): C = A B with A, B and C in pinned host memory, using at most a fixed budget of device
// memory ("fast memory"), as paper 1 emulates smaller GPUs with its Chunk8/Chunk16 budgets.
//
//   rasp::SpgemmContext ctx(512 << 20);    // one arena of 512 MiB, two streams, PCIe bandwidth calibration
//   rasp::PinnedCsr<double> C;              // reuse it across calls: pinned allocation is slow
//   rasp::SpgemmReport rep;
//   ctx.multiply(A, B, C, {}, &rep);                                            // RASP: overlap + re-planning
//   ctx.multiply(A, B, C, {.planner = rasp::SpgemmPlanner::Alg4}, &rep);       // paper 1, Alg. 4 baseline
//   std::puts(rep.to_string().c_str());
//
// Execution follows the plan (spgemm_plan.hpp): a copy stream and a compute stream (the RTX 4050 has one copy engine,
// so uploads and downloads share it); uploads run depth - 1 blocks ahead of cuSPARSE; every device buffer is carved
// from the budget arena, so the budget holds exactly (peak_device_bytes). A block whose cuSPARSE workspace or output
// does not fit is split in half and redone (over A rows for 1-D plans, over B rows when partial products are summed),
// so planning estimates affect speed, never correctness.
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rasp/pinned.hpp"
#include "rasp/spgemm_plan.hpp"

namespace rasp {

enum class SpgemmPlanner { Rasp, Alg4 };
const char* to_string(SpgemmPlanner p);

struct SpgemmOptions {
  SpgemmPlanner planner = SpgemmPlanner::Rasp;
  int depth = 0;                     // 0: the planner's choice (RASP: 2 or 3, Alg. 4: 1); >= 1 forces it (1 = no overlap)
  bool adaptive = true;              // RASP: probe block, then re-plan when measured rates drift > replan_threshold
  double replan_threshold = 0.2;
  std::optional<SpgemmMode> mode;    // RASP: force an execution mode (experiments, tests)
  bool allow_zero_copy = true;       // RASP: consider the zero-copy mode
  std::optional<CostParams> priors;  // model before any measurement; default CostParams{} with the context's bandwidths
};

// One tile of the execution: A/C rows [a0, a1) times B rows [b0, b1). Times are ms since execution started (-1: none).
struct ChunkRecord {
  int p = 0, q = 0;
  int32_t a0 = 0, a1 = 0, b0 = 0, b1 = 0;
  int64_t flops = 0, nnz = 0;  // estimated multiply-adds of the tile; nnz of its product (or of the summed block)
  size_t bytes_in = 0, bytes_out = 0, workspace = 0;
  float h2d_start = -1, h2d_end = -1, comp_start = -1, comp_end = -1, d2h_start = -1, d2h_end = -1;
  int splits = 0;       // halvings because a buffer did not fit
  bool replan = false;  // the remaining blocks were re-partitioned after this tile
};

struct SpgemmReport {
  SpgemmPlanner planner = SpgemmPlanner::Rasp;
  SpgemmPlan plan;  // as first made; replans revise the A/C blocks (see chunks)
  int replans = 0, splits = 0;
  double total_ms = 0;  // estimate + plan + execution: pinned A and B in, pinned C out
  double estimate_ms = 0, plan_ms = 0, exec_ms = 0;
  double setup_ms = 0;  // pinned allocation of C (and partial-C scratch); excluded from total_ms
  int64_t flops = 0, nnz_c = 0;
  size_t bytes_h2d = 0, bytes_d2h = 0, peak_device_bytes = 0, budget = 0;
  CostParams measured;  // the model parameters measured by this run (where tiles measured them)
  std::vector<ChunkRecord> chunks;

  double gflops() const { return total_ms > 0 ? 2.0 * double(flops) / (total_ms * 1e6) : 0; }
  std::string to_string() const;
};

class SpgemmContext {
 public:
  // Allocates the arena (the whole budget, once), creates the streams and the cuSPARSE handle and, unless told not to,
  // measures pinned H2D/D2H bandwidth with 64 MiB copies (the planner's bandwidths).
  explicit SpgemmContext(size_t fast_mem_budget, bool calibrate = true);
  ~SpgemmContext();
  SpgemmContext(const SpgemmContext&) = delete;
  SpgemmContext& operator=(const SpgemmContext&) = delete;

  // C = A B. C is reallocated at the upper bound of nnz(C) only if its capacity is too small. Throws BudgetTooSmall
  // if no plan fits the budget.
  template <class T>
  void multiply(const PinnedCsr<T>& A, const PinnedCsr<T>& B, PinnedCsr<T>& C, const SpgemmOptions& opt = {},
                SpgemmReport* report = nullptr);

  size_t budget() const;
  const CostParams& calibration() const;  // measured h2d_gbs / d2h_gbs (priors if not calibrated)
  double calibration_ms() const;

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

// One-shot convenience: a temporary context (calibration included) and a fresh output.
template <class T>
PinnedCsr<T> spgemm(const PinnedCsr<T>& A, const PinnedCsr<T>& B, size_t fast_mem_budget, const SpgemmOptions& opt = {},
                    SpgemmReport* report = nullptr);

}  // namespace rasp
