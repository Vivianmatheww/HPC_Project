// Phase 4 SpGEMM benchmarks.
//
//   bench_spgemm --study placement [--n 64,96,128,160] [--workloads AxP,RxA] [--precisions fp64] [--reps 10]
//                [--out results/phase4/placement.csv]
//       Paper 1, Table 3 on our GPU: cuSPARSE SpGEMM with each of A, B, C in device, pinned (zero-copy) or managed
//       memory (workspaces always on the device). Inputs already in place; only the cuSPARSE sequence is timed.
//
//   bench_spgemm --study ooc [--n 128,160,192,224,256] [--budget-mb 512] [--ratios 1,2,4] [--methods ...]
//                [--reps 10] [--warmup 2] [--out results/phase4/ooc.csv] [--chunks-out results/phase4/chunks.csv]
//       Out-of-core SpGEMM, pinned A and B in, pinned C out, per budget: Alg. 4 (paper 1, no overlap), Alg. 4 with
//       double buffering, RASP static, RASP adaptive, zero-copy; plus in-core (whole device) and device-only (kernel
//       time with everything already resident: the upper bound). Budgets: every --budget-mb value, and for every
//       --ratios value a budget of (|A| + |B| + |C|) / ratio. Method order is shuffled every repetition.
//
//   bench_spgemm --study plans [--n ...] [--budget-mb ...] [--ratios ...]
//       Prints the problem sizes and the plans RASP and Alg. 4 would choose (prior parameters); needs no GPU.
//
// Workloads: AxP and RxA are paper 1's multigrid products on Laplace3D (gallery.hpp); "LxL:<file.mtx>" is the
// triangle-counting product L * L with L the strictly lower triangle of a (symmetrised) graph.
// Correctness: nnz(C) must agree across methods, and 2,000 random rows of C (plus the first and last) are checked
// against a CPU computation for every method's first repetition (fp64 <= 1e-12, fp32 <= 1e-5).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "rasp/csv.hpp"
#include "rasp/cusparse_ref.hpp"
#include "rasp/gallery.hpp"
#include "rasp/io.hpp"
#include "rasp/reference.hpp"
#include "rasp/spgemm.hpp"
#include "rasp/timer.hpp"

using namespace rasp;

namespace {

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  for (size_t p = 0; p <= s.size();) {
    const size_t c = s.find(',', p);
    out.push_back(s.substr(p, c - p));
    if (c == std::string::npos) break;
    p = c + 1;
  }
  return out;
}

struct Args {
  std::string study;
  std::vector<std::string> workloads{"AxP", "RxA"};
  std::vector<int> n;
  std::vector<std::string> precisions{"fp64"};
  std::vector<double> budget_mb, ratios;
  std::vector<std::string> methods{"alg4", "alg4_overlap", "rasp_static", "rasp", "zerocopy", "incore", "device_only"};
  int reps = 10, warmup = 2, check_rows = 2000;
  std::string out, chunks_out;
};

[[noreturn]] void usage(const std::string& msg) {
  std::fprintf(stderr, "error: %s\nusage: bench_spgemm --study placement|ooc|plans [--workloads AxP,RxA,LxL:f.mtx] "
                       "[--n 64,128] [--precisions fp32,fp64] [--budget-mb 512] [--ratios 1,2,4] [--methods a,b] "
                       "[--reps N] [--warmup N] [--check-rows N] [--out f.csv] [--chunks-out f.csv]\n", msg.c_str());
  std::exit(2);
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) usage("missing value for " + k);
      return argv[++i];
    };
    if (k == "--study") a.study = val();
    else if (k == "--workloads") a.workloads = split(val());
    else if (k == "--n") { a.n.clear(); for (auto& s : split(val())) a.n.push_back(std::stoi(s)); }
    else if (k == "--precisions") a.precisions = split(val());
    else if (k == "--budget-mb") { a.budget_mb.clear(); for (auto& s : split(val())) a.budget_mb.push_back(std::stod(s)); }
    else if (k == "--ratios") { a.ratios.clear(); for (auto& s : split(val())) a.ratios.push_back(std::stod(s)); }
    else if (k == "--methods") a.methods = split(val());
    else if (k == "--reps") a.reps = std::stoi(val());
    else if (k == "--warmup") a.warmup = std::stoi(val());
    else if (k == "--check-rows") a.check_rows = std::stoi(val());
    else if (k == "--out") a.out = val();
    else if (k == "--chunks-out") a.chunks_out = val();
    else usage("unknown option " + k);
  }
  if (a.study != "placement" && a.study != "ooc" && a.study != "plans") usage("--study must be placement, ooc or plans");
  if (a.n.empty()) a.n = a.study == "placement" ? std::vector<int>{64, 96, 128, 160} : std::vector<int>{128, 160, 192, 224, 256};
  if (a.study != "placement" && a.budget_mb.empty() && a.ratios.empty()) a.budget_mb = {512};
  if (a.out.empty()) a.out = "results/phase4/" + a.study + ".csv";
  if (a.chunks_out.empty()) a.chunks_out = "results/phase4/chunks.csv";
  for (auto& p : a.precisions)
    if (p != "fp32" && p != "fp64") usage("unknown precision " + p);
  return a;
}

// ---- workloads -----------------------------------------------------------------------------
struct Workload {
  std::string name;  // AxP, RxA, LxL:<stem>
  int n = 0;         // grid size (Laplace), 0 for graphs
  HostCsr<double> A, B;
};

HostCsr<double> strict_lower(const HostCsr<double>& G) {
  HostCsr<double> L;
  L.rows = G.rows;
  L.cols = G.cols;
  L.row_ptr = {0};
  for (int32_t r = 0; r < G.rows; ++r) {
    for (int32_t k = G.row_ptr[size_t(r)]; k < G.row_ptr[size_t(r) + 1]; ++k)
      if (G.col_idx[size_t(k)] < r) {
        L.col_idx.push_back(G.col_idx[size_t(k)]);
        L.vals.push_back(1.0);
      }
    L.row_ptr.push_back(int32_t(L.col_idx.size()));
  }
  L.nnz = int64_t(L.col_idx.size());
  return L;
}

// Laplace workloads are generated per n; a graph workload is loaded once (n is ignored, reported as 0).
std::vector<Workload> make_workloads(const std::string& w, int n) {
  if (w == "AxP" || w == "RxA") {
    auto A = laplace3d(n);
    auto P = aggregation(n);
    if (w == "AxP") return {{w, n, std::move(A), std::move(P)}};
    return {{w, n, transpose(P), std::move(A)}};
  }
  if (w.rfind("LxL:", 0) == 0) {
    const std::string path = w.substr(4);
    auto G = read_matrix_market(path);
    auto L = strict_lower(G);
    std::string stem = path.substr(path.find_last_of('/') + 1);
    stem = stem.substr(0, stem.find('.'));
    return {{"LxL:" + stem, 0, L, L}};
  }
  usage("unknown workload " + w);
}

template <class T>
HostCsr<T> as(const HostCsr<double>& A) {
  return convert_values<T>(A);
}

double csr_bytes(int64_t rows, int64_t nnz, size_t vb) { return double(rows + 1) * 4 + double(nnz) * double(4 + vb); }

// ---- correctness: sampled rows against the CPU ------------------------------------------------
template <class T>
double check_rows(const HostCsr<T>& A, const HostCsr<T>& B, const int32_t* crp, const int32_t* ccol, const T* cval,
                  int samples, unsigned seed) {
  std::mt19937 rng(seed);
  std::vector<int32_t> rows{0, A.rows - 1};
  for (int i = 0; i < samples && A.rows > 0; ++i) rows.push_back(int32_t(rng() % uint32_t(A.rows)));
  double worst = 0;
  std::vector<std::pair<int32_t, double>> acc;
  for (int32_t r : rows) {
    if (r < 0) continue;
    acc.clear();
    for (int32_t k = A.row_ptr[size_t(r)]; k < A.row_ptr[size_t(r) + 1]; ++k) {
      const int32_t a = A.col_idx[size_t(k)];
      for (int32_t j = B.row_ptr[size_t(a)]; j < B.row_ptr[size_t(a) + 1]; ++j)
        acc.emplace_back(B.col_idx[size_t(j)], double(A.vals[size_t(k)]) * double(B.vals[size_t(j)]));
    }
    std::sort(acc.begin(), acc.end(), [](auto& x, auto& y) { return x.first < y.first; });
    std::vector<std::pair<int32_t, double>> row;
    for (auto& [c, v] : acc) {
      if (!row.empty() && row.back().first == c) row.back().second += v;
      else row.emplace_back(c, v);
    }
    const int32_t b = crp[r], e = crp[r + 1];
    if (size_t(e - b) != row.size()) return std::numeric_limits<double>::infinity();
    double scale = 0, diff = 0;
    for (size_t i = 0; i < row.size(); ++i) {
      if (ccol[b + int32_t(i)] != row[i].first) return std::numeric_limits<double>::infinity();
      scale = std::max(scale, std::abs(row[i].second));
      diff = std::max(diff, std::abs(double(cval[b + int32_t(i)]) - row[i].second));
    }
    worst = std::max(worst, scale > 0 ? diff / scale : diff);
  }
  return worst;
}

// ---- placement study --------------------------------------------------------------------------
enum class Loc { Dev, Pin, Man };
const char* loc_name(Loc l) { return l == Loc::Dev ? "device" : l == Loc::Pin ? "pinned" : "managed"; }

// A buffer in one kind of memory, with a pointer kernels can use.
class Mem {
 public:
  Mem() = default;
  Mem(Loc loc, size_t bytes) : loc_(loc), bytes_(std::max<size_t>(bytes, 16)) {
    switch (loc_) {
      case Loc::Dev: RASP_CUDA(cudaMalloc(&p_, bytes_)); break;
      case Loc::Pin: RASP_CUDA(cudaHostAlloc(&p_, bytes_, cudaHostAllocMapped)); break;
      case Loc::Man: RASP_CUDA(cudaMallocManaged(&p_, bytes_)); break;
    }
  }
  ~Mem() {
    if (!p_) return;
    if (loc_ == Loc::Pin) cudaFreeHost(p_);
    else cudaFree(p_);
  }
  Mem(const Mem&) = delete;
  Mem& operator=(const Mem&) = delete;
  Mem(Mem&& o) noexcept : loc_(o.loc_), bytes_(o.bytes_), p_(std::exchange(o.p_, nullptr)) {}
  Mem& operator=(Mem&& o) noexcept {
    std::swap(loc_, o.loc_);
    std::swap(bytes_, o.bytes_);
    std::swap(p_, o.p_);
    return *this;
  }
  void* ptr() const {  // usable by kernels
    if (loc_ != Loc::Pin) return p_;
    void* d = nullptr;
    RASP_CUDA(cudaHostGetDevicePointer(&d, p_, 0));
    return d;
  }
  void fill(const void* src, size_t n) {
    if (loc_ == Loc::Dev) RASP_CUDA(cudaMemcpy(p_, src, n, cudaMemcpyHostToDevice));
    else std::memcpy(p_, src, n);
  }
  void read(void* dst, size_t n) const {
    if (loc_ == Loc::Dev) RASP_CUDA(cudaMemcpy(dst, p_, n, cudaMemcpyDeviceToHost));
    else std::memcpy(dst, p_, n);
  }
  size_t bytes() const { return bytes_; }

 private:
  Loc loc_ = Loc::Dev;
  size_t bytes_ = 0;
  void* p_ = nullptr;
};

template <class T>
struct Placed {
  Mem rp, col, val;
  CsrView<T> view;
  Placed(const HostCsr<T>& h, Loc loc)
      : rp(loc, (size_t(h.rows) + 1) * 4), col(loc, size_t(h.nnz) * 4), val(loc, size_t(h.nnz) * sizeof(T)) {
    rp.fill(h.row_ptr.data(), (size_t(h.rows) + 1) * 4);
    if (h.nnz) {
      col.fill(h.col_idx.data(), size_t(h.nnz) * 4);
      val.fill(h.vals.data(), size_t(h.nnz) * sizeof(T));
    }
    view = {h.rows, h.cols, h.nnz, static_cast<const int32_t*>(rp.ptr()), static_cast<const int32_t*>(col.ptr()),
            static_cast<const T*>(val.ptr())};
  }
};

// cuSPARSE SpGEMM with fixed placements; buffers are sized by a first (untimed) call and reused, so the timed
// region is the cuSPARSE sequence only.
template <class T>
struct PlacedProduct {
  Loc c_loc;
  Mem crp, ccol, cval, w1, w2;
  cusparseHandle_t handle = nullptr;
  SpgemmCall last;
  PlacedProduct(Loc c, int32_t rows) : c_loc(c), crp(c, (size_t(rows) + 1) * 4) { RASP_CUSPARSE(cusparseCreate(&handle)); }
  ~PlacedProduct() { cusparseDestroy(handle); }
  void run(const CsrView<T>& A, const CsrView<T>& B, cudaStream_t s) {
    auto alloc = [&](SpgemmBuffer w, size_t bytes) -> void* {
      Mem* m = w == SpgemmBuffer::Work1 ? &w1 : w == SpgemmBuffer::Work2 ? &w2 : w == SpgemmBuffer::CCol ? &ccol : &cval;
      const Loc loc = w == SpgemmBuffer::Work1 || w == SpgemmBuffer::Work2 ? Loc::Dev : c_loc;
      if (m->bytes() < bytes) *m = Mem(loc, bytes);
      return m->ptr();
    };
    last = cusparse_spgemm_view(handle, A, B, static_cast<int32_t*>(crp.ptr()), alloc, s);
  }
  HostCsr<T> result(int32_t rows, int32_t cols) const {
    HostCsr<T> C;
    C.rows = rows;
    C.cols = cols;
    C.nnz = last.nnz_c;
    C.row_ptr.resize(size_t(rows) + 1);
    C.col_idx.resize(size_t(C.nnz));
    C.vals.resize(size_t(C.nnz));
    crp.read(C.row_ptr.data(), C.row_ptr.size() * 4);
    if (C.nnz) {
      ccol.read(C.col_idx.data(), size_t(C.nnz) * 4);
      cval.read(C.vals.data(), size_t(C.nnz) * sizeof(T));
    }
    return C;
  }
};

const std::vector<std::string> kPlacementCols = {
    "workload", "n", "precision", "a_loc", "b_loc", "c_loc", "placement", "a_rows", "nnz_a", "nnz_b", "nnz_c", "flops",
    "c_upper", "gamma", "warmup", "reps", "median_ms", "p25_ms", "p75_ms", "min_ms", "gflops", "buffer1_bytes",
    "buffer2_bytes", "rel_err", "check", "status", "error"};

Fields row_of(const std::vector<std::string>& cols, std::map<std::string, std::string> v, bool env = true) {
  Fields f;
  for (auto& k : cols) f.emplace_back(k, v.count(k) ? v[k] : "");
  if (env)
    for (auto& kv : environment_stamp()) f.push_back(kv);
  return f;
}

template <class T>
void placement(const Workload& w, const Args& args, CsvWriter& csv, int& failures) {
  const char* prec = sizeof(T) == 8 ? "fp64" : "fp32";
  const double tol = sizeof(T) == 8 ? 1e-12 : 1e-5;
  const auto A = as<T>(w.A), B = as<T>(w.B);
  const auto s = estimate(host_view(A), host_view(B));
  BenchOptions bench;
  bench.reps = args.reps;
  bench.warmup = args.warmup;
  cudaStream_t stream;
  RASP_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  std::vector<std::array<Loc, 3>> configs;
  for (Loc a : {Loc::Dev, Loc::Pin})
    for (Loc b : {Loc::Dev, Loc::Pin})
      for (Loc c : {Loc::Dev, Loc::Pin}) configs.push_back({a, b, c});
  configs.push_back({Loc::Man, Loc::Dev, Loc::Dev});
  configs.push_back({Loc::Dev, Loc::Man, Loc::Dev});
  configs.push_back({Loc::Dev, Loc::Dev, Loc::Man});
  configs.push_back({Loc::Man, Loc::Man, Loc::Man});

  HostCsr<T> ref;
  for (const auto& cfg : configs) {
    std::map<std::string, std::string> v{{"workload", w.name}, {"n", std::to_string(w.n)}, {"precision", prec},
                                         {"a_loc", loc_name(cfg[0])}, {"b_loc", loc_name(cfg[1])},
                                         {"c_loc", loc_name(cfg[2])},
                                         {"placement", std::string("A=") + loc_name(cfg[0]) + " B=" + loc_name(cfg[1]) +
                                                           " C=" + loc_name(cfg[2])},
                                         {"a_rows", std::to_string(A.rows)}, {"nnz_a", std::to_string(A.nnz)},
                                         {"nnz_b", std::to_string(B.nnz)}, {"flops", std::to_string(s.flops(0, A.rows))},
                                         {"c_upper", std::to_string(s.c_upper(0, A.rows))},
                                         {"warmup", std::to_string(args.warmup)}, {"reps", std::to_string(args.reps)}};
    try {
      Placed<T> pa(A, cfg[0]), pb(B, cfg[1]);
      PlacedProduct<T> pc(cfg[2], A.rows);
      pc.run(pa.view, pb.view, stream);  // sizes the buffers
      RASP_CUDA(cudaStreamSynchronize(stream));
      const auto C = pc.result(A.rows, B.cols);
      if (ref.rows == 0 && ref.nnz == 0) ref = C;  // the first configuration is all-device
      const double err = csr_max_rel_diff(C, ref);
      const auto t = time_gpu([&](cudaStream_t st) { pc.run(pa.view, pb.view, st); }, stream, bench);
      v["nnz_c"] = std::to_string(C.nnz);
      v["gamma"] = fmt(double(C.nnz) / double(std::max<int64_t>(s.c_upper(0, A.rows), 1)));
      v["median_ms"] = fmt(t.median_ms);
      v["p25_ms"] = fmt(t.p25_ms);
      v["p75_ms"] = fmt(t.p75_ms);
      v["min_ms"] = fmt(t.min_ms);
      v["gflops"] = fmt(2.0 * double(s.flops(0, A.rows)) / (t.median_ms * 1e6));
      v["buffer1_bytes"] = std::to_string(pc.last.buffer1_bytes);
      v["buffer2_bytes"] = std::to_string(pc.last.buffer2_bytes);
      v["rel_err"] = fmt(err);
      v["check"] = err <= tol ? "pass" : "FAIL";
      v["status"] = "ok";
      if (err > tol) ++failures;
      std::printf("  %s %-40s %10.3f ms  %7.3f GFLOP/s %s\n", prec, v["placement"].c_str(), t.median_ms,
                  2.0 * double(s.flops(0, A.rows)) / (t.median_ms * 1e6), err <= tol ? "" : " <-- FAIL");
    } catch (const std::exception& e) {
      cudaGetLastError();
      v["status"] = "error";
      v["error"] = e.what();
      ++failures;
      std::printf("  %s %-40s ERROR %s\n", prec, v["placement"].c_str(), e.what());
    }
    csv.row(row_of(kPlacementCols, v));
  }
  cudaStreamDestroy(stream);
}

// ---- out-of-core study --------------------------------------------------------------------------
const std::vector<std::string> kOocCols = {
    "workload", "n", "precision", "a_rows", "nnz_a", "nnz_b", "nnz_c", "flops", "problem_bytes", "budget_bytes",
    "ratio", "method", "planner", "mode", "parts_ac", "parts_b", "depth", "probe", "replans", "splits", "tiles",
    "warmup", "reps", "median_ms", "p25_ms", "p75_ms", "min_ms", "exec_ms", "plan_ms", "estimate_ms", "setup_ms",
    "gflops", "bytes_h2d", "bytes_d2h", "peak_device_bytes", "predicted_ms", "h2d_gbs", "d2h_gbs",
    "measured_flops_per_ms", "measured_gamma", "measured_ws_per_flop", "rel_err", "check", "status", "error"};

const std::vector<std::string> kChunkCols = {
    "workload", "n", "precision", "budget_bytes", "method", "index", "p", "q", "a0", "a1", "b0", "b1", "flops", "nnz",
    "bytes_in", "bytes_out", "workspace", "h2d_start", "h2d_end", "comp_start", "comp_end", "d2h_start", "d2h_end",
    "splits", "replan"};

struct Method {
  std::string name;
  SpgemmOptions opt;
};

std::vector<Method> budget_methods(const std::vector<std::string>& names) {
  std::vector<Method> m;
  for (auto& n : names) {
    SpgemmOptions o;
    if (n == "alg4") o.planner = SpgemmPlanner::Alg4;
    else if (n == "alg4_overlap") { o.planner = SpgemmPlanner::Alg4; o.depth = 2; }
    else if (n == "rasp_static") o.adaptive = false;
    else if (n == "rasp") {}
    else if (n == "zerocopy") o.mode = SpgemmMode::ZeroCopy;
    else if (n == "incore" || n == "device_only") continue;  // not budget methods
    else usage("unknown method " + n);
    m.push_back({n, o});
  }
  return m;
}

template <class T>
void ooc(const Workload& w, const Args& args, CsvWriter& csv, CsvWriter& chunks, int& failures) {
  const char* prec = sizeof(T) == 8 ? "fp64" : "fp32";
  const double tol = sizeof(T) == 8 ? 1e-12 : 1e-5;
  const auto Ah = as<T>(w.A), Bh = as<T>(w.B);
  const auto A = PinnedCsr<T>::from(Ah), B = PinnedCsr<T>::from(Bh);
  const int64_t flops = estimate(host_view(Ah), host_view(Bh)).flops(0, A.rows);
  PinnedCsr<T> C;
  auto wants = [&](const char* m) { return std::find(args.methods.begin(), args.methods.end(), m) != args.methods.end(); };

  int64_t nnz_c = -1;
  double problem = 0;
  const std::map<std::string, std::string> base{
      {"workload", w.name}, {"n", std::to_string(w.n)}, {"precision", prec}, {"a_rows", std::to_string(A.rows)},
      {"nnz_a", std::to_string(A.nnz)}, {"nnz_b", std::to_string(B.nnz)}, {"flops", std::to_string(flops)},
      {"warmup", std::to_string(args.warmup)}, {"reps", std::to_string(args.reps)}};
  // status "" = ran (check decides pass/FAIL); "infeasible" = no plan fits (not a failure); "error" = a failure.
  auto emit = [&](std::map<std::string, std::string> v, const SpgemmReport* rep, const std::vector<double>& times,
                  double err, double budget) {
    v.insert(base.begin(), base.end());
    v["nnz_c"] = std::to_string(nnz_c);
    v["problem_bytes"] = fmt(problem);
    v["budget_bytes"] = fmt(budget);
    v["ratio"] = fmt(budget > 0 ? problem / budget : 0);
    if (!times.empty()) {
      const double med = percentile(times, 50);
      v["median_ms"] = fmt(med);
      v["p25_ms"] = fmt(percentile(times, 25));
      v["p75_ms"] = fmt(percentile(times, 75));
      v["min_ms"] = fmt(*std::min_element(times.begin(), times.end()));
      v["gflops"] = fmt(2.0 * double(flops) / (med * 1e6));
    }
    if (rep) {
      v["planner"] = to_string(rep->planner);
      v["mode"] = to_string(rep->plan.mode);
      v["parts_ac"] = std::to_string(rep->plan.parts_ac());
      v["parts_b"] = std::to_string(rep->plan.parts_b());
      v["depth"] = std::to_string(rep->plan.depth);
      v["probe"] = rep->plan.probe ? "1" : "0";
      v["replans"] = std::to_string(rep->replans);
      v["splits"] = std::to_string(rep->splits);
      v["tiles"] = std::to_string(std::count_if(rep->chunks.begin(), rep->chunks.end(), [](auto& c) { return c.p >= 0; }));
      v["exec_ms"] = fmt(rep->exec_ms);
      v["plan_ms"] = fmt(rep->plan_ms);
      v["estimate_ms"] = fmt(rep->estimate_ms);
      v["setup_ms"] = fmt(rep->setup_ms);
      v["bytes_h2d"] = std::to_string(rep->bytes_h2d);
      v["bytes_d2h"] = std::to_string(rep->bytes_d2h);
      v["peak_device_bytes"] = std::to_string(rep->peak_device_bytes);
      v["predicted_ms"] = fmt(rep->plan.predicted_ms);
      v["h2d_gbs"] = fmt(rep->plan.params.h2d_gbs);
      v["d2h_gbs"] = fmt(rep->plan.params.d2h_gbs);
      v["measured_flops_per_ms"] = fmt(rep->measured.flops_per_ms);
      v["measured_gamma"] = fmt(rep->measured.gamma);
      v["measured_ws_per_flop"] = fmt(rep->measured.ws_bytes_per_flop);
    }
    v["rel_err"] = fmt(err);
    if (!v.count("status")) {
      const bool ok = err <= tol;
      v["check"] = ok ? "pass" : "FAIL";
      v["status"] = "ok";
      if (!ok) ++failures;
    } else if (v["status"] == "error") {
      ++failures;
    }
    csv.row(row_of(kOocCols, v));
  };
  auto emit_chunks = [&](const std::string& method, double budget, const SpgemmReport& rep) {
    for (size_t i = 0; i < rep.chunks.size(); ++i) {
      const auto& c = rep.chunks[i];
      chunks.row(row_of(kChunkCols,
                        {{"workload", w.name}, {"n", std::to_string(w.n)}, {"precision", prec},
                         {"budget_bytes", fmt(budget)}, {"method", method}, {"index", std::to_string(i)},
                         {"p", std::to_string(c.p)}, {"q", std::to_string(c.q)}, {"a0", std::to_string(c.a0)},
                         {"a1", std::to_string(c.a1)}, {"b0", std::to_string(c.b0)}, {"b1", std::to_string(c.b1)},
                         {"flops", std::to_string(c.flops)}, {"nnz", std::to_string(c.nnz)},
                         {"bytes_in", std::to_string(c.bytes_in)}, {"bytes_out", std::to_string(c.bytes_out)},
                         {"workspace", std::to_string(c.workspace)}, {"h2d_start", fmt(c.h2d_start)},
                         {"h2d_end", fmt(c.h2d_end)}, {"comp_start", fmt(c.comp_start)}, {"comp_end", fmt(c.comp_end)},
                         {"d2h_start", fmt(c.d2h_start)}, {"d2h_end", fmt(c.d2h_end)},
                         {"splits", std::to_string(c.splits)}, {"replan", c.replan ? "1" : "0"}},
                        false));
    }
  };
  auto check = [&](const PinnedCsr<T>& Cr, unsigned seed) {
    if (Cr.nnz != nnz_c) return std::numeric_limits<double>::infinity();
    return check_rows(Ah, Bh, Cr.row_ptr.data(), Cr.col_idx.data(), Cr.vals.data(), args.check_rows, seed);
  };

  // In-core with most of the device (pinned A, B in; C out; no overlap). Its first run also gives nnz(C), from which
  // the problem size and the ratio budgets follow; if in-core does not fit, a RASP run on the same context does.
  {
    size_t free_b = 0, total_b = 0;
    RASP_CUDA(cudaMemGetInfo(&free_b, &total_b));
    const size_t big = size_t(double(free_b) * 0.85);
    SpgemmContext ctx(big);
    SpgemmOptions o;
    o.mode = SpgemmMode::InCore;
    SpgemmReport rep;
    std::vector<double> times;
    double err = 0;
    std::string infeasible;
    try {
      for (int i = 0; i < args.warmup + args.reps; ++i) {
        ctx.multiply(A, B, C, o, &rep);
        if (i == 0) {
          nnz_c = C.nnz;
          err = check(C, 1);
        }
        if (i >= args.warmup) times.push_back(rep.total_ms);
        if (!wants("incore")) break;
      }
    } catch (const BudgetTooSmall& e) {
      infeasible = e.what();
      SpgemmReport r2;
      ctx.multiply(A, B, C, {}, &r2);
      nnz_c = C.nnz;
    }
    problem = csr_bytes(A.rows, A.nnz, sizeof(T)) + csr_bytes(B.rows, B.nnz, sizeof(T)) +
              csr_bytes(A.rows, nnz_c, sizeof(T));
    if (wants("incore")) {
      if (infeasible.empty()) {
        emit({{"method", "incore"}}, &rep, times, err, double(big));
        emit_chunks("incore", double(big), rep);
        std::printf("  %s incore      %s\n", prec, rep.to_string().c_str());
      } else {
        emit({{"method", "incore"}, {"status", "infeasible"}, {"error", infeasible}}, nullptr, {}, 0, double(big));
        std::printf("  %s incore      %s\n", prec, infeasible.c_str());
      }
    }
  }
  // Device-only: cuSPARSE with A and B already resident and C left on the device: the kernel-throughput upper bound.
  if (wants("device_only")) {
    try {
      auto dA = DeviceCsr<T>::upload(Ah);
      auto dB = DeviceCsr<T>::upload(Bh);
      PlacedProduct<T> pc(Loc::Dev, A.rows);
      cudaStream_t st;
      RASP_CUDA(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
      pc.run(dA.view(), dB.view(), st);
      RASP_CUDA(cudaStreamSynchronize(st));
      BenchOptions bo;
      bo.reps = args.reps;
      bo.warmup = args.warmup;
      bo.flush_l2 = false;
      const auto t = time_gpu([&](cudaStream_t s) { pc.run(dA.view(), dB.view(), s); }, st, bo);
      cudaStreamDestroy(st);
      const double err = pc.last.nnz_c == nnz_c ? 0.0 : std::numeric_limits<double>::infinity();
      emit({{"method", "device_only"}, {"mode", "device_only"}}, nullptr, {t.median_ms}, err, 0);
      std::printf("  %s device_only %.3f ms\n", prec, t.median_ms);
    } catch (const std::exception& e) {
      cudaGetLastError();
      emit({{"method", "device_only"}, {"status", "error"}, {"error", e.what()}}, nullptr, {}, 0, 0);
      std::printf("  %s device_only ERROR %s\n", prec, e.what());
    }
  }

  std::vector<double> budgets;
  for (double mb : args.budget_mb) budgets.push_back(mb * double(1 << 20));
  for (double r : args.ratios) budgets.push_back(problem / r);
  const auto methods = budget_methods(args.methods);
  std::mt19937 rng(12345);

  for (double budget : budgets) {
    std::printf("  %s budget %.0f MiB (problem %.0f MiB, ratio %.2f)\n", prec, budget / double(1 << 20),
                problem / double(1 << 20), problem / budget);
    SpgemmContext ctx{size_t(budget)};
    std::vector<std::vector<double>> times(methods.size());
    std::vector<SpgemmReport> last(methods.size());
    std::vector<double> err(methods.size(), 0);
    std::vector<std::string> error(methods.size()), status(methods.size());
    std::vector<size_t> order(methods.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    for (int rep = 0; rep < args.warmup + args.reps; ++rep) {
      std::shuffle(order.begin(), order.end(), rng);
      for (size_t mi : order) {
        if (!status[mi].empty()) continue;
        try {
          ctx.multiply(A, B, C, methods[mi].opt, &last[mi]);
          if (rep == 0) err[mi] = check(C, unsigned(7 + mi));
          if (rep >= args.warmup) times[mi].push_back(last[mi].total_ms);
        } catch (const BudgetTooSmall& e) {
          status[mi] = "infeasible";
          error[mi] = e.what();
        } catch (const std::exception& e) {
          cudaGetLastError();
          status[mi] = "error";
          error[mi] = e.what();
        }
      }
    }
    for (size_t mi = 0; mi < methods.size(); ++mi) {
      if (!status[mi].empty()) {
        emit({{"method", methods[mi].name}, {"status", status[mi]}, {"error", error[mi]}}, nullptr, {}, 0, budget);
        std::printf("    %-12s %s: %s\n", methods[mi].name.c_str(), status[mi].c_str(), error[mi].c_str());
        continue;
      }
      emit({{"method", methods[mi].name}}, &last[mi], times[mi], err[mi], budget);
      emit_chunks(methods[mi].name, budget, last[mi]);
      std::printf("    %-12s %8.2f ms  %s\n", methods[mi].name.c_str(), percentile(times[mi], 50),
                  last[mi].to_string().c_str());
    }
  }
}

// ---- plans (no GPU) -------------------------------------------------------------------------------
void plans(const Workload& w, const Args& args) {
  const auto s = estimate(host_view(w.A), host_view(w.B));
  const double ub = s.a_bytes(0, s.a_rows) + s.b_bytes(0, s.a_cols) + s.c_bytes(0, s.a_rows, 1.0);
  std::printf("%s n=%d: |A| %.0f MB, |B| %.0f MB, |C| <= %.0f MB, total <= %.0f MB, %.3g multiply-adds\n",
              w.name.c_str(), w.n, s.a_bytes(0, s.a_rows) / 1e6, s.b_bytes(0, s.a_cols) / 1e6,
              s.c_bytes(0, s.a_rows, 1.0) / 1e6, ub / 1e6, double(s.flops(0, s.a_rows)));
  std::vector<double> budgets;
  for (double mb : args.budget_mb) budgets.push_back(mb * double(1 << 20));
  for (double r : args.ratios) budgets.push_back(ub / r);
  const CostParams p;
  for (double b : budgets) {
    std::printf("  budget %6.0f MiB\n", b / double(1 << 20));
    for (int which = 0; which < 2; ++which) {
      try {
        const auto plan = which ? plan_alg4(s, 0.97 * b, p) : plan_rasp(s, 0.97 * b, p);
        std::printf("    %-5s %s\n", which ? "alg4" : "rasp", plan.to_string().c_str());
      } catch (const std::exception& e) {
        std::printf("    %-5s %s\n", which ? "alg4" : "rasp", e.what());
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse(argc, argv);
  int failures = 0;
  std::unique_ptr<CsvWriter> csv, chunks;
  if (args.study != "plans") {
    csv = std::make_unique<CsvWriter>(args.out);
    if (args.study == "ooc") chunks = std::make_unique<CsvWriter>(args.chunks_out);
    const auto& env = environment_stamp();
    std::printf("%s | cuSPARSE %s | git %s | study %s\n", env[0].second.c_str(), env[4].second.c_str(),
                env[5].second.c_str(), args.study.c_str());
  }
  for (const auto& wname : args.workloads) {
    const bool graph = wname.rfind("LxL:", 0) == 0;
    for (int n : graph ? std::vector<int>{0} : args.n) {
      for (const auto& w : make_workloads(wname, n)) {
        std::printf("== %s n=%d: %d x %d (nnz %lld) * %d x %d (nnz %lld)\n", w.name.c_str(), w.n, w.A.rows, w.A.cols,
                    (long long)w.A.nnz, w.B.rows, w.B.cols, (long long)w.B.nnz);
        if (args.study == "plans") {
          plans(w, args);
          continue;
        }
        for (const auto& p : args.precisions) {
          try {
            if (args.study == "placement") {
              if (p == "fp32") placement<float>(w, args, *csv, failures);
              else placement<double>(w, args, *csv, failures);
            } else {
              if (p == "fp32") ooc<float>(w, args, *csv, *chunks, failures);
              else ooc<double>(w, args, *csv, *chunks, failures);
            }
          } catch (const std::exception& e) {
            cudaGetLastError();
            ++failures;
            std::printf("  %s ERROR %s\n", p.c_str(), e.what());
          }
        }
      }
    }
  }
  if (args.study != "plans") std::printf("\n%d failures -> %s\n", failures, args.out.c_str());
  return failures == 0 ? 0 : 1;
}
