// Out-of-core SpGEMM executor: runs a plan from spgemm_plan.hpp with a copy stream, a compute stream, the budget
// arena and cuSPARSE. See spgemm.hpp for the contract.
//
// Every mode is a list of tiles (A/C rows x B rows) run by one pipeline: tile k + depth - 1 is uploaded before tile k
// is computed, because cuSPARSE blocks the host in its size queries; downloads queue on the same copy stream.
// Buffers go back to the arena through events (free_after), so a buffer is never reused while a stream may still
// touch it. Overflow handling: a 1-D tile whose cuSPARSE workspace or output does not fit is split over A rows (each
// half stored on its own); a tile that sums partial products is split over B rows (the halves are summed in turn).
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <vector>

#include "rasp/cusparse_ref.hpp"
#include "rasp/range_allocator.hpp"
#include "rasp/spgemm.hpp"
#include "rasp/spgemm_kernels.hpp"
#include "rasp/timer.hpp"

namespace rasp {

const char* to_string(SpgemmPlanner p) { return p == SpgemmPlanner::Rasp ? "rasp" : "alg4"; }

std::string SpgemmReport::to_string() const {
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "%s: %s | %.2f ms (estimate %.2f, plan %.2f, exec %.2f) = %.3f GFLOP/s | nnz(C) %lld | H2D %.3g GB, "
                "D2H %.3g GB | peak %.1f / %.1f MiB | %zu tiles, %d replans, %d splits",
                rasp::to_string(planner), plan.to_string().c_str(), total_ms, estimate_ms, plan_ms, exec_ms, gflops(),
                (long long)nnz_c, double(bytes_h2d) / 1e9, double(bytes_d2h) / 1e9,
                double(peak_device_bytes) / double(1 << 20), double(budget) / double(1 << 20), chunks.size(), replans,
                splits);
  return buf;
}

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

struct NvtxRange {
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
};

// The fast-memory budget: one cudaMalloc carved by a RangeAllocator. free_after() returns buffers once the work
// enqueued on a stream so far has finished; alloc() waits for such pending frees (oldest first) before giving up.
class Arena {
 public:
  explicit Arena(size_t bytes) : ra_(bytes) {
    if (ra_.capacity()) RASP_CUDA(cudaMalloc(&base_, ra_.capacity()));
  }
  ~Arena() {
    for (auto& p : pending_) cudaEventSynchronize(p.event);
    for (auto& p : pending_) pool_.push_back(p.event);
    for (cudaEvent_t e : pool_) cudaEventDestroy(e);
    if (base_) cudaFree(base_);
  }
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  void* alloc(size_t bytes) {
    reclaim();
    for (;;) {
      if (auto off = ra_.alloc(bytes)) return static_cast<char*>(base_) + *off;
      if (pending_.empty()) return nullptr;
      wait_oldest();
    }
  }
  template <class T>
  T* alloc_n(int64_t n) {
    return static_cast<T*>(alloc(size_t(std::max<int64_t>(n, 0)) * sizeof(T)));
  }
  void free_now(void* p) {
    if (p) ra_.free(size_t(static_cast<char*>(p) - static_cast<char*>(base_)));
  }
  void free_after(std::vector<void*> ptrs, cudaStream_t stream) {
    ptrs.erase(std::remove(ptrs.begin(), ptrs.end(), nullptr), ptrs.end());
    if (ptrs.empty()) return;
    cudaEvent_t e;
    if (pool_.empty()) {
      RASP_CUDA(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    } else {
      e = pool_.back();
      pool_.pop_back();
    }
    RASP_CUDA(cudaEventRecord(e, stream));
    pending_.push_back({e, std::move(ptrs)});
  }
  void drain() {
    while (!pending_.empty()) wait_oldest();
  }
  size_t capacity() const { return ra_.capacity(); }
  size_t peak() const { return ra_.peak(); }
  void reset_peak() { ra_.reset_peak(); }

 private:
  struct Pending {
    cudaEvent_t event;
    std::vector<void*> ptrs;
  };
  void release(Pending& p) {
    for (void* q : p.ptrs) free_now(q);
    pool_.push_back(p.event);
  }
  void reclaim() {
    for (auto it = pending_.begin(); it != pending_.end();) {
      const cudaError_t q = cudaEventQuery(it->event);
      if (q == cudaSuccess) {
        release(*it);
        it = pending_.erase(it);
      } else {
        if (q != cudaErrorNotReady) RASP_CUDA(q);
        ++it;
      }
    }
  }
  void wait_oldest() {
    RASP_CUDA(cudaEventSynchronize(pending_.front().event));
    release(pending_.front());
    pending_.pop_front();
  }

  RangeAllocator ra_;
  void* base_ = nullptr;
  std::deque<Pending> pending_;
  std::vector<cudaEvent_t> pool_;
};

}  // namespace

struct SpgemmContext::Impl {
  Arena arena;
  cudaStream_t copy = nullptr, compute = nullptr;
  cusparseHandle_t handle = nullptr;
  CostParams calib;
  double calib_ms = 0;
  std::vector<cudaEvent_t> events;  // timing events, handed out during a run and recycled by the next one
  size_t events_used = 0;
  ProblemSizes sizes;                // reused across calls (no page faults on repeated estimates)
  PinnedBuffer<int32_t> scratch_rp;  // partial-C row pointers (ChunkB with several B blocks)

  Impl(size_t budget, bool calibrate) : arena(budget) {
    try {
      RASP_CUDA(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
      RASP_CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
      RASP_CUSPARSE(cusparseCreate(&handle));
      if (calibrate) measure_bandwidth();
    } catch (...) {
      release();
      throw;
    }
  }
  ~Impl() { release(); }

  void release() noexcept {
    if (copy) cudaStreamSynchronize(copy);
    if (compute) cudaStreamSynchronize(compute);
    for (cudaEvent_t e : events) cudaEventDestroy(e);
    events.clear();
    if (handle) cusparseDestroy(handle);
    if (copy) cudaStreamDestroy(copy);
    if (compute) cudaStreamDestroy(compute);
    handle = nullptr;
    copy = compute = nullptr;
  }

  cudaEvent_t event() {
    if (events_used == events.size()) {
      cudaEvent_t e;
      RASP_CUDA(cudaEventCreate(&e));
      events.push_back(e);
    }
    return events[events_used++];
  }

  // Pinned 64 MiB copies, median of 5 after a warm-up: the planner's PCIe bandwidths (transient memory, outside the
  // budget).
  void measure_bandwidth() {
    const auto t0 = Clock::now();
    const size_t bytes = size_t(64) << 20;
    PinnedBuffer<char> h(bytes);
    DeviceBuffer<char> d(bytes);
    std::memset(h.data(), 1, bytes);
    cudaEvent_t e0 = event(), e1 = event();
    auto gbs = [&](bool h2d) {
      std::vector<double> t;
      for (int i = 0; i < 6; ++i) {
        RASP_CUDA(cudaEventRecord(e0, copy));
        RASP_CUDA(cudaMemcpyAsync(h2d ? static_cast<void*>(d.data()) : h.data(), h2d ? static_cast<void*>(h.data()) : d.data(),
                                  bytes, h2d ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, copy));
        RASP_CUDA(cudaEventRecord(e1, copy));
        RASP_CUDA(cudaEventSynchronize(e1));
        float ms = 0;
        RASP_CUDA(cudaEventElapsedTime(&ms, e0, e1));
        if (i) t.push_back(ms);
      }
      return double(bytes) / (percentile(t, 50) * 1e6);
    };
    calib.h2d_gbs = gbs(true);
    calib.d2h_gbs = gbs(false);
    events_used = 0;
    calib_ms = ms_since(t0);
  }
};

namespace {

// A CSR block whose pointers are device (or mapped host) memory; `owned` lists the arena buffers behind it.
template <class T>
struct DevBlock {
  int32_t rows = 0, cols = 0;
  int64_t nnz = 0;
  int32_t* rp = nullptr;
  int32_t* col = nullptr;
  T* val = nullptr;
  std::vector<void*> owned;
  bool empty() const { return rp == nullptr; }
  CsrView<T> view() const { return {rows, cols, nnz, rp, col, val}; }
};

struct Tile {
  int p = 0, q = 0;
  int32_t a0 = 0, a1 = 0, b0 = 0, b1 = 0;
};

struct TileEvents {
  cudaEvent_t h0 = nullptr, h1 = nullptr, c0 = nullptr, c1 = nullptr, d0 = nullptr, d1 = nullptr;
};

template <class T>
class Run {
 public:
  Run(SpgemmContext::Impl& c, const PinnedCsr<T>& A, const PinnedCsr<T>& B, PinnedCsr<T>& C, const SpgemmOptions& o,
      SpgemmReport& r, const ProblemSizes& s, double budget)
      : c_(c), A_(A), B_(B), C_(C), o_(o), r_(r), s_(s), budget_(budget), plan_(r.plan) {}

  void execute() {
    c_.events_used = 0;
    start_ = c_.event();
    RASP_CUDA(cudaEventRecord(start_, c_.compute));
    zero_copy_ = plan_.mode == SpgemmMode::ZeroCopy;
    const int depth = std::max(plan_.depth, 1);
    if (plan_.mode == SpgemmMode::ChunkAC) {
      build_tiles_ac(0, plan_.ac_bounds);
      pipeline(0, depth, true);
    } else if (plan_.mode == SpgemmMode::ChunkB && plan_.parts_b() > 1) {
      run_b_outer(depth);
    } else {  // InCore, ZeroCopy, ChunkB with B whole: B resident, A/C row blocks
      const DevBlock<T> b = upload_b(0, B_.rows, true);
      bdev_ = b;
      build_tiles_1d(0, plan_.ac_bounds);
      pipeline(0, plan_.mode == SpgemmMode::ChunkB ? depth : 1, true);
      c_.arena.free_after(b.owned, c_.compute);
    }
    RASP_CUDA(cudaStreamSynchronize(c_.compute));
    RASP_CUDA(cudaStreamSynchronize(c_.copy));
    c_.arena.drain();
    C_.nnz = c_nnz_;
    finish_records();
  }

 private:
  // ---- tiles --------------------------------------------------------------------------------
  void build_tiles_1d(size_t from, const std::vector<int32_t>& bounds) {
    resize_tiles(from);
    for (size_t i = 0; i + 1 < bounds.size(); ++i)
      push_tile({int(from + i), 0, bounds[i], bounds[i + 1], 0, B_.rows});
  }
  void build_tiles_ac(size_t from, const std::vector<int32_t>& bounds) {
    resize_tiles(from);
    const int first_p = from ? tiles_[from - 1].p + 1 : 0;
    for (size_t i = 0; i + 1 < bounds.size(); ++i)
      for (int q = 0; q < plan_.parts_b(); ++q)
        push_tile({first_p + int(i), q, bounds[i], bounds[i + 1], plan_.b_bounds[size_t(q)],
                   plan_.b_bounds[size_t(q) + 1]});
  }
  void resize_tiles(size_t n) {
    tiles_.resize(n);
    ev_.resize(n);
    rec_.resize(n);
    a_.resize(n);
    bq_.resize(n);
    prev_.resize(n);
  }
  void push_tile(const Tile& t) {
    tiles_.push_back(t);
    ev_.emplace_back();
    a_.emplace_back();
    bq_.emplace_back();
    prev_.emplace_back();
    ChunkRecord rc;
    rc.p = t.p;
    rc.q = t.q;
    rc.a0 = t.a0;
    rc.a1 = t.a1;
    rc.b0 = t.b0;
    rc.b1 = t.b1;
    rc.flops = int64_t(double(s_.flops(t.a0, t.a1)) * b_frac(t.b0, t.b1) + 0.5);
    rec_.push_back(rc);
  }
  double b_frac(int32_t k0, int32_t k1) const {
    const double total = double(B_.nnz);
    return total > 0 ? double(brp(k1) - brp(k0)) / total : 0;
  }
  int64_t arp(int32_t r) const { return A_.row_ptr.data()[r]; }
  int64_t brp(int32_t k) const { return B_.row_ptr.data()[k]; }

  // ---- the pipeline -------------------------------------------------------------------------
  // Runs tiles [k_begin, tiles_.size()) with uploads `depth - 1` tiles ahead. Re-planning (adaptive runs) may rebuild
  // the tiles that have not been uploaded yet.
  void pipeline(size_t k_begin, int depth, bool may_replan) {
    const size_t L = size_t(depth - 1);
    // Only adaptive RASP runs wait for a tile's compute to finish before going on; the others are measured at the end,
    // so they pay no extra host synchronisation.
    const bool adapt = may_replan && o_.adaptive && o_.planner == SpgemmPlanner::Rasp;
    size_t issued = k_begin;
    for (size_t k = k_begin; k < tiles_.size(); ++k) {
      while (issued < tiles_.size() && issued <= k + L && issue(issued)) ++issued;
      if (adapt && k > k_begin) measure_and_replan(k - 1, issued);
      if (issued <= k) {  // memory was short when it was its turn: now it waits for every pending free
        if (!issue(k)) throw BudgetTooSmall("spgemm: the inputs of one tile do not fit the budget");
        issued = k + 1;
      }
      compute(k);
      if (depth == 1) RASP_CUDA(cudaStreamSynchronize(c_.copy));  // strictly sequential (the Alg. 4 baseline)
    }
  }

  // Uploads tile j's streamed operands: A rows (unless zero-copy), partial C (ChunkB, later B blocks), B rows (ChunkAC).
  // Allocates everything first, so a failure (memory still busy) leaves nothing behind.
  bool issue(size_t j) {
    const Tile& t = tiles_[j];
    const bool ac = plan_.mode == SpgemmMode::ChunkAC;
    const bool up_a = !zero_copy_ && (!ac || t.q == 0);
    const bool up_prev = plan_.mode == SpgemmMode::ChunkB && t.q > 0;
    const bool up_b = ac;
    const int32_t m = t.a1 - t.a0;
    const int64_t a_nnz = arp(t.a1) - arp(t.a0), p_nnz = up_prev ? partial_nnz_[size_t(t.p)] : 0;
    const int64_t b_nnz = brp(t.b1) - brp(t.b0);
    if (!up_a && !up_prev && !up_b) return true;  // zero-copy: the kernels read A in place

    std::vector<void*> got;
    auto take = [&](int64_t bytes) {
      void* p = c_.arena.alloc(size_t(std::max<int64_t>(bytes, 0)));
      got.push_back(p);
      return p;
    };
    DevBlock<T> a, prev, b;
    if (up_a) {
      a = {m, A_.cols, a_nnz, static_cast<int32_t*>(take((m + 1) * 4)), static_cast<int32_t*>(take(a_nnz * 4)),
           static_cast<T*>(take(a_nnz * int64_t(sizeof(T)))), {}};
    }
    if (up_prev) {
      prev = {m, B_.cols, p_nnz, static_cast<int32_t*>(take((m + 1) * 4)), static_cast<int32_t*>(take(p_nnz * 4)),
              static_cast<T*>(take(p_nnz * int64_t(sizeof(T)))), {}};
    }
    if (up_b) {
      const int32_t kb = t.b1 - t.b0;
      b = {kb, B_.cols, b_nnz, static_cast<int32_t*>(take((kb + 1) * 4)), static_cast<int32_t*>(take(b_nnz * 4)),
           static_cast<T*>(take(b_nnz * int64_t(sizeof(T)))), {}};
    }
    if (std::find(got.begin(), got.end(), nullptr) != got.end()) {
      for (void* p : got) c_.arena.free_now(p);  // nothing was enqueued on them
      return false;
    }

    ChunkRecord& rc = rec_[j];
    TileEvents& ev = ev_[j];
    ev.h0 = c_.event();
    RASP_CUDA(cudaEventRecord(ev.h0, c_.copy));
    auto h2d = [&](void* dst, const void* src, int64_t bytes) {
      if (bytes <= 0) return;
      RASP_CUDA(cudaMemcpyAsync(dst, src, size_t(bytes), cudaMemcpyHostToDevice, c_.copy));
      rc.bytes_in += size_t(bytes);
      r_.bytes_h2d += size_t(bytes);
    };
    if (up_a) {
      h2d(a.rp, A_.row_ptr.data() + t.a0, (m + 1) * 4);
      h2d(a.col, A_.col_idx.data() + arp(t.a0), a_nnz * 4);
      h2d(a.val, A_.vals.data() + arp(t.a0), a_nnz * int64_t(sizeof(T)));
      a.owned = {a.rp, a.col, a.val};
      a_[j] = a;
    }
    if (up_prev) {
      const int64_t off = s_.cub_ptr[size_t(t.a0)];
      h2d(prev.rp, c_.scratch_rp.data() + t.a0 + t.p, (m + 1) * 4);
      h2d(prev.col, C_.col_idx.data() + off, p_nnz * 4);
      h2d(prev.val, C_.vals.data() + off, p_nnz * int64_t(sizeof(T)));
      prev.owned = {prev.rp, prev.col, prev.val};
      prev_[j] = prev;
    }
    if (up_b) {
      h2d(b.rp, B_.row_ptr.data() + t.b0, int64_t(t.b1 - t.b0 + 1) * 4);
      h2d(b.col, B_.col_idx.data() + brp(t.b0), b_nnz * 4);
      h2d(b.val, B_.vals.data() + brp(t.b0), b_nnz * int64_t(sizeof(T)));
      b.owned = {b.rp, b.col, b.val};
      bq_[j] = b;
    }
    ev.h1 = c_.event();
    RASP_CUDA(cudaEventRecord(ev.h1, c_.copy));
    return true;
  }

  // Uploads B rows [k0, k1) (resident for a whole pass) and makes the compute stream wait for it.
  DevBlock<T> upload_b(int32_t k0, int32_t k1, bool record) {
    const int32_t kb = k1 - k0;
    const int64_t nnz = brp(k1) - brp(k0);
    DevBlock<T> b{kb, B_.cols, nnz, c_.arena.alloc_n<int32_t>(kb + 1), c_.arena.alloc_n<int32_t>(nnz),
                  c_.arena.alloc_n<T>(nnz), {}};
    b.owned = {b.rp, b.col, b.val};
    if (!b.rp || !b.col || !b.val) {
      for (void* p : b.owned) c_.arena.free_now(p);
      throw BudgetTooSmall("spgemm: the resident block of B does not fit the budget");
    }
    ChunkRecord rc;
    rc.p = -1;  // a resident upload, not a tile
    rc.b0 = k0;
    rc.b1 = k1;
    TileEvents ev;
    ev.h0 = c_.event();
    RASP_CUDA(cudaEventRecord(ev.h0, c_.copy));
    const size_t bytes = size_t(kb + 1) * 4 + size_t(nnz) * (4 + sizeof(T));
    RASP_CUDA(cudaMemcpyAsync(b.rp, B_.row_ptr.data() + k0, size_t(kb + 1) * 4, cudaMemcpyHostToDevice, c_.copy));
    if (nnz) {
      RASP_CUDA(cudaMemcpyAsync(b.col, B_.col_idx.data() + brp(k0), size_t(nnz) * 4, cudaMemcpyHostToDevice, c_.copy));
      RASP_CUDA(cudaMemcpyAsync(b.val, B_.vals.data() + brp(k0), size_t(nnz) * sizeof(T), cudaMemcpyHostToDevice,
                                c_.copy));
    }
    ev.h1 = c_.event();
    RASP_CUDA(cudaEventRecord(ev.h1, c_.copy));
    RASP_CUDA(cudaStreamWaitEvent(c_.compute, ev.h1));
    rc.bytes_in = bytes;
    r_.bytes_h2d += bytes;
    if (record) {
      extra_rec_.push_back(rc);
      extra_ev_.push_back(ev);
    }
    return b;
  }

  // ---- compute ------------------------------------------------------------------------------
  void compute(size_t k) {
    const Tile t = tiles_[k];
    char name[48];
    std::snprintf(name, sizeof(name), "tile p%d q%d", t.p, t.q);
    NvtxRange range(name);
    TileEvents& ev = ev_[k];
    if (ev.h1) RASP_CUDA(cudaStreamWaitEvent(c_.compute, ev.h1));
    ev.c0 = c_.event();
    RASP_CUDA(cudaEventRecord(ev.c0, c_.compute));
    const int32_t m = t.a1 - t.a0;
    const int32_t a_off = int32_t(arp(t.a0));

    if (plan_.mode == SpgemmMode::ChunkAC) {
      const size_t k_a = k - size_t(t.q);  // tile (p, 0) uploaded A_p
      if (t.q == 0) rebase_row_ptr(a_[k].rp, a_[k].rp, m + 1, a_off, c_.compute);
      acc_ = accumulate(a_[k_a].view(), std::move(acc_), bq_[k], t.b0, t.b0, t.b1, rec_[k]);
      c_.arena.free_after(bq_[k].owned, c_.compute);
      if (t.q == plan_.parts_b() - 1) {
        store_final(acc_, t.a0, t.a1, ev, rec_[k]);
        acc_ = DevBlock<T>{};
        c_.arena.free_after(a_[k_a].owned, c_.compute);
      }
    } else if (plan_.mode == SpgemmMode::ChunkB && plan_.parts_b() > 1) {
      rebase_row_ptr(a_[k].rp, a_[k].rp, m + 1, a_off, c_.compute);
      DevBlock<T> sum = accumulate(a_[k].view(), std::move(prev_[k]), bdev_, bdev_k0_, t.b0, t.b1, rec_[k]);
      if (t.q == plan_.parts_b() - 1) store_final(sum, t.a0, t.a1, ev, rec_[k]);
      else store_partial(sum, t, ev, rec_[k]);
      c_.arena.free_after(a_[k].owned, c_.compute);
    } else {
      DevBlock<T> a;
      if (zero_copy_) {
        a = {m, A_.cols, arp(t.a1) - arp(t.a0), c_.arena.alloc_n<int32_t>(m + 1),
             A_.col_idx.device_ptr() + a_off, A_.vals.device_ptr() + a_off, {}};
        if (!a.rp) throw BudgetTooSmall("spgemm: no room for one block's row pointers");
        a.owned = {a.rp};
        rebase_row_ptr(A_.row_ptr.device_ptr() + t.a0, a.rp, m + 1, a_off, c_.compute);
      } else {
        a = a_[k];
        rebase_row_ptr(a.rp, a.rp, m + 1, a_off, c_.compute);
      }
      rows_1d(a, t.a0, t.a0, t.a1, ev, rec_[k]);
      c_.arena.free_after(a.owned, c_.compute);
    }
    ev.c1 = c_.event();
    RASP_CUDA(cudaEventRecord(ev.c1, c_.compute));
  }

  struct Product {
    SpgemmCall call;
    int32_t* rp = nullptr;
    std::vector<void*> work, out;  // arena buffers: workspaces / C's arrays
  };

  // One cuSPARSE call with every buffer from the arena (C's arrays straight into pinned C when zero-copy).
  Product product(const CsrView<T>& a, const CsrView<T>& b, ChunkRecord& rc) {
    Product pr;
    pr.rp = c_.arena.alloc_n<int32_t>(int64_t(a.rows) + 1);
    if (!pr.rp) return pr;
    pr.out.push_back(pr.rp);
    auto alloc = [&](SpgemmBuffer w, size_t bytes) -> void* {
      const bool c_array = w == SpgemmBuffer::CCol || w == SpgemmBuffer::CVal;
      if (c_array && zero_copy_)
        return w == SpgemmBuffer::CCol ? static_cast<void*>(C_.col_idx.device_ptr() + c_nnz_)
                                       : static_cast<void*>(C_.vals.device_ptr() + c_nnz_);
      void* p = c_.arena.alloc(bytes);
      if (p) (c_array ? pr.out : pr.work).push_back(p);
      return p;
    };
    pr.call = cusparse_spgemm_view(c_.handle, a, b, pr.rp, alloc, c_.compute);
    if (pr.call.complete) rc.workspace = std::max(rc.workspace, pr.call.buffer1_bytes + pr.call.buffer2_bytes);
    return pr;
  }

  // C rows [r0, r1) from the uploaded A block `a` (rows [a0, ...)) times the resident B; stored at C's next offset.
  void rows_1d(const DevBlock<T>& a, int32_t a0, int32_t r0, int32_t r1, TileEvents& ev, ChunkRecord& rc) {
    const int32_t m = r1 - r0;
    CsrView<T> av = a.view();
    std::vector<void*> tmp;
    if (r0 != a0 || m != a.rows) {  // a sub-range after a split: its own rebased row pointers
      const int32_t base = int32_t(arp(r0) - arp(a0));
      auto* rp = c_.arena.alloc_n<int32_t>(m + 1);
      if (!rp) throw BudgetTooSmall("spgemm: no room for the row pointers of a split block");
      rebase_row_ptr(a.rp + (r0 - a0), rp, m + 1, base, c_.compute);
      av = {m, A_.cols, arp(r1) - arp(r0), rp, a.col + base, a.val + base};
      tmp.push_back(rp);
    }
    Product pr = product(av, bdev_.view(), rc);
    if (!pr.rp || !pr.call.complete) {
      std::vector<void*> all = tmp;
      all.insert(all.end(), pr.work.begin(), pr.work.end());
      all.insert(all.end(), pr.out.begin(), pr.out.end());
      c_.arena.free_after(all, c_.compute);
      if (m <= 1) throw BudgetTooSmall("spgemm: one row of C does not fit the budget");
      const int64_t half = (s_.flops_ptr[size_t(r0)] + s_.flops_ptr[size_t(r1)]) / 2;
      int32_t mid = int32_t(std::lower_bound(s_.flops_ptr.begin() + r0 + 1, s_.flops_ptr.begin() + r1, half) -
                            s_.flops_ptr.begin());
      mid = std::clamp(mid, r0 + 1, r1 - 1);
      ++rc.splits;
      ++r_.splits;
      rows_1d(a, a0, r0, mid, ev, rc);
      rows_1d(a, a0, mid, r1, ev, rc);
      return;
    }
    note_workspace(pr, double(s_.flops(r0, r1)));
    tmp.insert(tmp.end(), pr.work.begin(), pr.work.end());
    c_.arena.free_after(tmp, c_.compute);
    DevBlock<T> out{m, B_.cols, pr.call.nnz_c, pr.rp, pr.call.c_col, static_cast<T*>(pr.call.c_val), pr.out};
    rc.nnz += out.nnz;
    store_final(out, r0, r1, ev, rc);
  }

  // Sums A_block x B rows [k0, k1) (a sub-range of the device block `b`, whose first row is kb0) into `acc`.
  DevBlock<T> accumulate(const CsrView<T>& a, DevBlock<T> acc, const DevBlock<T>& b, int32_t kb0, int32_t k0, int32_t k1,
                         ChunkRecord& rc) {
    const int32_t m = a.rows;
    const int64_t boff = brp(k0) - brp(kb0);
    std::vector<void*> tmp;
    CsrView<T> bv;
    if (k0 == 0 && k1 == B_.rows) {
      bv = b.view();  // all of B: its row pointers already start at 0
    } else {
      auto* ext = c_.arena.alloc_n<int32_t>(int64_t(B_.rows) + 1);
      if (!ext) throw BudgetTooSmall("spgemm: no room for the full-height row pointers of a B block");
      expand_row_ptr(b.rp + (k0 - kb0), k0, k1, B_.rows, ext, c_.compute);
      bv = {B_.rows, B_.cols, brp(k1) - brp(k0), ext, b.col + boff, b.val + boff};
      tmp.push_back(ext);
    }
    Product pr = product(a, bv, rc);
    if (!pr.rp || !pr.call.complete) {
      std::vector<void*> all = tmp;
      all.insert(all.end(), pr.work.begin(), pr.work.end());
      all.insert(all.end(), pr.out.begin(), pr.out.end());
      c_.arena.free_after(all, c_.compute);
      if (k1 - k0 <= 1) throw BudgetTooSmall("spgemm: one row of B does not fit beside its A/C block");
      const int64_t half = (brp(k0) + brp(k1)) / 2;
      int32_t mid = int32_t(std::lower_bound(B_.row_ptr.data() + k0 + 1, B_.row_ptr.data() + k1, half) -
                            B_.row_ptr.data());
      mid = std::clamp(mid, k0 + 1, k1 - 1);
      ++rc.splits;
      ++r_.splits;
      acc = accumulate(a, std::move(acc), b, kb0, k0, mid, rc);
      return accumulate(a, std::move(acc), b, kb0, mid, k1, rc);
    }
    note_workspace(pr, double(s_.flops(rc.a0, rc.a1)) * b_frac(k0, k1));
    tmp.insert(tmp.end(), pr.work.begin(), pr.work.end());
    c_.arena.free_after(tmp, c_.compute);
    DevBlock<T> P{m, B_.cols, pr.call.nnz_c, pr.rp, pr.call.c_col, static_cast<T*>(pr.call.c_val), pr.out};
    if (acc.empty()) return P;

    // acc + P, both with sorted rows.
    const size_t temp_bytes = csr_add_temp_bytes(m);
    void* temp = c_.arena.alloc(temp_bytes);
    auto* srp = c_.arena.alloc_n<int32_t>(int64_t(m) + 1);
    if (!temp || !srp) throw BudgetTooSmall("spgemm: no room to sum partial products");
    const int64_t nnz = csr_add_symbolic(m, acc.rp, acc.col, P.rp, P.col, srp, temp, temp_bytes, c_.compute);
    auto* scol = c_.arena.alloc_n<int32_t>(nnz);
    auto* sval = c_.arena.alloc_n<T>(nnz);
    if (!scol || !sval) throw BudgetTooSmall("spgemm: no room for a summed partial product");
    csr_add_numeric(m, acc.rp, acc.col, acc.val, P.rp, P.col, P.val, srp, scol, sval, c_.compute);
    c_.arena.free_after({temp}, c_.compute);
    c_.arena.free_after(acc.owned, c_.compute);
    c_.arena.free_after(P.owned, c_.compute);
    return DevBlock<T>{m, B_.cols, nnz, srp, scol, sval, {srp, scol, sval}};
  }

  void note_workspace(const Product& pr, double flops) {
    if (flops <= 0) return;
    const double ws = double(pr.call.buffer1_bytes + pr.call.buffer2_bytes) - plan_.params.ws_base_bytes;
    ws_ratio_ = std::max(ws_ratio_, std::max(ws, 0.0) / flops);
  }

  // Rows [r0, r1) of C, final: row pointers shifted to C's next offset, then downloaded (or written in place).
  void store_final(const DevBlock<T>& blk, int32_t r0, int32_t r1, TileEvents& ev, ChunkRecord& rc) {
    const int32_t m = r1 - r0;
    const int32_t off = int32_t(c_nnz_);
    if (zero_copy_) {  // C's arrays were written in place by cuSPARSE; only the row pointers remain
      offset_row_ptr(blk.rp + 1, C_.row_ptr.device_ptr() + r0 + 1, m, off, c_.compute);
      c_.arena.free_after(blk.owned, c_.compute);
    } else {
      offset_row_ptr(blk.rp + 1, blk.rp + 1, m, off, c_.compute);
      download(blk, C_.row_ptr.data() + r0 + 1, blk.rp + 1, m, C_.col_idx.data() + off, C_.vals.data() + off, ev, rc);
    }
    if (plan_.mode != SpgemmMode::ChunkB || plan_.parts_b() == 1) {  // compression of final blocks
      gamma_nnz_ += double(blk.nnz);
      gamma_cub_ += double(s_.c_upper(r0, r1));
    }
    c_nnz_ += blk.nnz;
  }

  // A partial sum (ChunkB with several B blocks): parked in pinned memory until the next B block's pass.
  void store_partial(const DevBlock<T>& blk, const Tile& t, TileEvents& ev, ChunkRecord& rc) {
    const int64_t off = s_.cub_ptr[size_t(t.a0)];
    download(blk, c_.scratch_rp.data() + t.a0 + t.p, blk.rp, t.a1 - t.a0 + 1, C_.col_idx.data() + off,
             C_.vals.data() + off, ev, rc);
    partial_nnz_[size_t(t.p)] = blk.nnz;
  }

  void download(const DevBlock<T>& blk, int32_t* h_rp, const int32_t* d_rp, int64_t n_rp, int32_t* h_col, T* h_val,
                TileEvents& ev, ChunkRecord& rc) {
    cudaEvent_t done = c_.event();
    RASP_CUDA(cudaEventRecord(done, c_.compute));
    RASP_CUDA(cudaStreamWaitEvent(c_.copy, done));
    if (!ev.d0) {
      ev.d0 = c_.event();
      RASP_CUDA(cudaEventRecord(ev.d0, c_.copy));
    }
    auto d2h = [&](void* dst, const void* src, int64_t bytes) {
      if (bytes <= 0) return;
      RASP_CUDA(cudaMemcpyAsync(dst, src, size_t(bytes), cudaMemcpyDeviceToHost, c_.copy));
      rc.bytes_out += size_t(bytes);
      r_.bytes_d2h += size_t(bytes);
    };
    d2h(h_rp, d_rp, n_rp * 4);
    d2h(h_col, blk.col, blk.nnz * 4);
    d2h(h_val, blk.val, blk.nnz * int64_t(sizeof(T)));
    if (!ev.d1) ev.d1 = c_.event();
    RASP_CUDA(cudaEventRecord(ev.d1, c_.copy));
    c_.arena.free_after(blk.owned, c_.copy);
  }

  // ---- ChunkB with several B blocks: one pass per B block, A/C blocks streamed in each ------------------------
  void run_b_outer(int depth) {
    partial_nnz_.assign(size_t(plan_.parts_ac()), 0);
    for (int q = 0; q < plan_.parts_b(); ++q) {
      const int32_t k0 = plan_.b_bounds[size_t(q)], k1 = plan_.b_bounds[size_t(q) + 1];
      bdev_ = upload_b(k0, k1, true);
      bdev_k0_ = k0;
      const size_t first = tiles_.size();
      for (size_t i = 0; i + 1 < plan_.ac_bounds.size(); ++i)
        push_tile({int(i), q, plan_.ac_bounds[i], plan_.ac_bounds[i + 1], k0, k1});
      pipeline(first, depth, false);
      // The next pass re-reads the partial sums this one downloads, and B's block is replaced: drain.
      RASP_CUDA(cudaStreamSynchronize(c_.compute));
      RASP_CUDA(cudaStreamSynchronize(c_.copy));
      c_.arena.free_after(bdev_.owned, c_.compute);
      c_.arena.drain();
    }
  }

  // ---- adaptivity -----------------------------------------------------------------------------
  // After tile j: fold its measured compute time into the model; if the model drifted past the threshold, re-partition
  // the A/C rows whose tiles have not been uploaded yet.
  void measure_and_replan(size_t j, size_t issued) {
    RASP_CUDA(cudaEventSynchronize(ev_[j].c1));
    float ms = 0;
    RASP_CUDA(cudaEventElapsedTime(&ms, ev_[j].c0, ev_[j].c1));
    meas_flops_ += double(rec_[j].flops);
    meas_ms_ += ms;
    ++meas_n_;
    const CostParams cand = measured_params();
    if (!needs_replan(plan_.params, cand, o_.replan_threshold)) return;
    // First tile not yet uploaded that starts an A/C block.
    size_t from = issued;
    while (from < tiles_.size() && tiles_[from].q != 0) ++from;
    if (from >= tiles_.size()) return;
    const auto bounds = replan_rows(s_, budget_, plan_, cand, tiles_[from].a0);
    plan_.params = cand;
    if (plan_.mode == SpgemmMode::ChunkAC) build_tiles_ac(from, bounds);
    else build_tiles_1d(from, bounds);
    rec_[j].replan = true;
    ++r_.replans;
  }

  CostParams measured_params() const {
    CostParams m = plan_.params;
    if (meas_ms_ > 0 && meas_flops_ > 0)
      m.flops_per_ms = meas_flops_ / std::max(meas_ms_ - meas_n_ * m.tile_overhead_ms, 0.25 * meas_ms_);
    if (gamma_cub_ > 0) m.gamma = std::clamp(1.1 * gamma_nnz_ / gamma_cub_, 1e-3, 1.0);
    if (ws_ratio_ > 0) m.ws_bytes_per_flop = 1.1 * ws_ratio_;
    return m;
  }

  void finish_records() {
    auto at = [&](cudaEvent_t e) -> float {
      if (!e) return -1;
      float ms = 0;
      RASP_CUDA(cudaEventElapsedTime(&ms, start_, e));
      return ms;
    };
    r_.chunks.clear();
    auto put = [&](ChunkRecord rc, const TileEvents& ev) {
      rc.h2d_start = at(ev.h0);
      rc.h2d_end = at(ev.h1);
      rc.comp_start = at(ev.c0);
      rc.comp_end = at(ev.c1);
      rc.d2h_start = at(ev.d0);
      rc.d2h_end = at(ev.d1);
      r_.chunks.push_back(rc);
    };
    for (size_t i = 0; i < extra_rec_.size(); ++i) put(extra_rec_[i], extra_ev_[i]);
    for (size_t j = 0; j < tiles_.size(); ++j) put(rec_[j], ev_[j]);
    // The model this run measured, from every tile (re-planning only saw the tiles finished before each decision).
    double fl = 0, ms = 0;
    for (const auto& rc : r_.chunks)
      if (rc.p >= 0 && rc.comp_end >= 0) {
        fl += double(rc.flops);
        ms += rc.comp_end - rc.comp_start;
      }
    r_.measured = plan_.params;
    if (ms > 0 && fl > 0)
      r_.measured.flops_per_ms =
          fl / std::max(ms - double(tiles_.size()) * plan_.params.tile_overhead_ms, 0.25 * ms);
    if (gamma_cub_ > 0) r_.measured.gamma = std::clamp(gamma_nnz_ / gamma_cub_, 1e-6, 1.0);
    if (ws_ratio_ > 0) r_.measured.ws_bytes_per_flop = ws_ratio_;
  }

  SpgemmContext::Impl& c_;
  const PinnedCsr<T>& A_;
  const PinnedCsr<T>& B_;
  PinnedCsr<T>& C_;
  const SpgemmOptions& o_;
  SpgemmReport& r_;
  const ProblemSizes& s_;
  double budget_;
  SpgemmPlan plan_;
  bool zero_copy_ = false;
  cudaEvent_t start_ = nullptr;

  std::vector<Tile> tiles_;
  std::vector<TileEvents> ev_;
  std::vector<ChunkRecord> rec_;
  std::vector<DevBlock<T>> a_, bq_, prev_;  // per tile: uploaded A block, B block (ChunkAC), partial C (ChunkB)
  std::vector<ChunkRecord> extra_rec_;      // resident B uploads
  std::vector<TileEvents> extra_ev_;
  DevBlock<T> bdev_;  // the resident B (block)
  int32_t bdev_k0_ = 0;
  DevBlock<T> acc_;   // ChunkAC: the A/C block's running sum
  std::vector<int64_t> partial_nnz_;
  int64_t c_nnz_ = 0;

  double meas_flops_ = 0, meas_ms_ = 0;
  int meas_n_ = 0;
  double gamma_nnz_ = 0, gamma_cub_ = 0, ws_ratio_ = 0;
};

}  // namespace

SpgemmContext::SpgemmContext(size_t fast_mem_budget, bool calibrate)
    : impl_(std::make_unique<Impl>(fast_mem_budget, calibrate)) {}
SpgemmContext::~SpgemmContext() = default;
size_t SpgemmContext::budget() const { return impl_->arena.capacity(); }
const CostParams& SpgemmContext::calibration() const { return impl_->calib; }
double SpgemmContext::calibration_ms() const { return impl_->calib_ms; }

template <class T>
void SpgemmContext::multiply(const PinnedCsr<T>& A, const PinnedCsr<T>& B, PinnedCsr<T>& C, const SpgemmOptions& opt,
                             SpgemmReport* report) {
  if (A.cols != B.rows) throw std::invalid_argument("spgemm: A.cols != B.rows");
  SpgemmReport local;
  SpgemmReport& r = report ? *report : local;
  r = SpgemmReport{};
  r.planner = opt.planner;
  r.budget = impl_->arena.capacity();
  Impl& c = *impl_;
  NvtxRange range("rasp::spgemm");

  // 1. Estimate (timed).
  auto t0 = Clock::now();
  estimate(A.view(), B.view(), c.sizes);
  const ProblemSizes& s = c.sizes;
  r.estimate_ms = ms_since(t0);
  r.flops = s.flops(0, s.a_rows);

  // Output allocation (setup, not timed): the upper bound on nnz(C).
  auto t1 = Clock::now();
  C.reserve(A.rows, B.cols, s.c_upper(0, s.a_rows));
  double setup_ms = ms_since(t1);

  if (A.nnz == 0 || B.nnz == 0 || A.rows == 0) {  // empty product
    std::fill(C.row_ptr.data(), C.row_ptr.data() + size_t(A.rows) + 1, 0);
    r.plan.mode = SpgemmMode::InCore;
    r.total_ms = r.estimate_ms;
    r.setup_ms = setup_ms;
    return;
  }

  // 2. Plan (timed). The arena keeps a little slack for alignment and fragmentation.
  t0 = Clock::now();
  CostParams pri = opt.priors.value_or(CostParams{});
  if (!opt.priors) {
    pri.h2d_gbs = c.calib.h2d_gbs;
    pri.d2h_gbs = c.calib.d2h_gbs;
  }
  const double budget = 0.97 * double(c.arena.capacity());
  if (opt.planner == SpgemmPlanner::Alg4) {
    r.plan = plan_alg4(s, budget, pri, std::max(opt.depth, 1));
  } else {
    RaspPlanOptions po;
    po.depth = opt.depth;
    po.probe = opt.adaptive;
    po.allow_zero_copy = opt.allow_zero_copy;
    if (opt.mode) {
      po.has_forced_mode = true;
      po.forced_mode = *opt.mode;
    }
    r.plan = plan_rasp(s, budget, pri, po);
  }
  r.plan_ms = ms_since(t0);

  // Partial-C row pointers for ChunkB with several B blocks (setup).
  if (r.plan.mode == SpgemmMode::ChunkB && r.plan.parts_b() > 1) {
    t1 = Clock::now();
    const size_t need = size_t(A.rows) + size_t(r.plan.parts_ac()) + 1;
    if (c.scratch_rp.size() < need) c.scratch_rp.resize(need);
    setup_ms += ms_since(t1);
  }
  r.setup_ms = setup_ms;

  // 3. Execute (timed).
  c.arena.reset_peak();
  t0 = Clock::now();
  Run<T>(c, A, B, C, opt, r, s, budget).execute();
  r.exec_ms = ms_since(t0);
  r.total_ms = r.estimate_ms + r.plan_ms + r.exec_ms;
  r.nnz_c = C.nnz;
  r.peak_device_bytes = c.arena.peak();
}

template <class T>
PinnedCsr<T> spgemm(const PinnedCsr<T>& A, const PinnedCsr<T>& B, size_t fast_mem_budget, const SpgemmOptions& opt,
                    SpgemmReport* report) {
  SpgemmContext ctx(fast_mem_budget);
  PinnedCsr<T> C;
  ctx.multiply(A, B, C, opt, report);
  return C;
}

template void SpgemmContext::multiply(const PinnedCsr<float>&, const PinnedCsr<float>&, PinnedCsr<float>&,
                                      const SpgemmOptions&, SpgemmReport*);
template void SpgemmContext::multiply(const PinnedCsr<double>&, const PinnedCsr<double>&, PinnedCsr<double>&,
                                      const SpgemmOptions&, SpgemmReport*);
template PinnedCsr<float> spgemm(const PinnedCsr<float>&, const PinnedCsr<float>&, size_t, const SpgemmOptions&,
                                 SpgemmReport*);
template PinnedCsr<double> spgemm(const PinnedCsr<double>&, const PinnedCsr<double>&, size_t, const SpgemmOptions&,
                                  SpgemmReport*);

}  // namespace rasp
