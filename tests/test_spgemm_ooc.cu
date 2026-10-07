// Out-of-core SpGEMM on the GPU: the executor's kernels against host code, the cuSPARSE core against the wrapper,
// and SpgemmContext in every mode, planner and depth against the CPU reference (tiny budgets force many tiles,
// wrong priors force overflow splits and re-planning).
#include <algorithm>
#include <cstring>

#include "rasp/cusparse_ref.hpp"
#include "rasp/gallery.hpp"
#include "rasp/reference.hpp"
#include "rasp/spgemm.hpp"
#include "rasp/spgemm_kernels.hpp"
#include "test_util.hpp"

using namespace rasp;

namespace {

template <class T>
HostCsr<T> as(const HostCsr<double>& A) {
  HostCsr<T> B;
  B.rows = A.rows;
  B.cols = A.cols;
  B.nnz = A.nnz;
  B.row_ptr = A.row_ptr;
  B.col_idx = A.col_idx;
  B.vals.assign(A.vals.begin(), A.vals.end());
  return B;
}

template <class T>
void expect_product(const HostCsr<T>& got, const HostCsr<T>& ref, const std::string& what) {
  const double tol = sizeof(T) == 8 ? 1e-12 : 1e-5;
  const double d = csr_max_rel_diff(got, ref);
  if (!(d <= tol)) rasp_test::fail(__FILE__, __LINE__, what + ": rel diff " + std::to_string(d));
}

// Priors scaled to test-sized matrices (a few MB): the 1 MiB workspace prior would dominate their budgets.
CostParams test_priors() {
  CostParams p;
  p.ws_base_bytes = 256 << 10;
  return p;
}

// Room for cuSPARSE's own workspace on top of a fraction of the problem: test budgets must stay feasible while being
// smaller than the problem (so the executor really chunks).
constexpr double kSlack = 2 << 20;

double problem_bytes(const HostCsr<double>& A, const HostCsr<double>& B, size_t vb) {
  auto s = estimate(host_view(A), host_view(B));
  s.value_bytes = vb;
  return s.a_bytes(0, s.a_rows) + s.b_bytes(0, s.a_cols) + s.c_bytes(0, s.a_rows, 1.0);
}

struct Case {
  std::string name;
  HostCsr<double> A, B;
};

// A few MB each, so budgets of a fraction of the problem plus kSlack are still smaller than the problem.
std::vector<Case> cases() {
  std::vector<Case> c;
  auto L = laplace3d(32);
  auto P = aggregation(32);
  c.push_back({"AxP", L, P});
  c.push_back({"RxA", transpose(P), L});
  // Irregular: empty rows, one long row in A, one dense row in B, rectangular.
  auto A = rasp_test::random_csr<double>(20000, 15000,
                                         [](int32_t r) { return r % 17 == 0 ? 0 : r == 5 ? 4000 : 1 + r % 15; }, 21);
  auto B = rasp_test::random_csr<double>(15000, 12000,
                                         [](int32_t r) { return r == 3 ? 12000 : r % 7 == 0 ? 0 : r % 12; }, 22);
  c.push_back({"random", A, B});
  return c;
}

}  // namespace

// ---- kernels -----------------------------------------------------------------------------------
TEST(row_pointer_kernels) {
  std::vector<int32_t> x{5, 7, 7, 12, 20, 21};
  DeviceBuffer<int32_t> dx, dy(x.size()), dout(11);
  dx.upload(x);
  rebase_row_ptr(dx.data(), dy.data(), int64_t(x.size()), 5, nullptr);
  CHECK((dy.download() == std::vector<int32_t>{0, 2, 2, 7, 15, 16}));
  offset_row_ptr(dy.data(), dy.data(), int64_t(x.size()), 100, nullptr);  // in place
  CHECK((dy.download() == std::vector<int32_t>{100, 102, 102, 107, 115, 116}));
  // B rows [3, 8) of a 10-row matrix whose block row pointers are x: rows outside are empty.
  expand_row_ptr(dx.data(), 3, 8, 10, dout.data(), nullptr);
  RASP_CUDA(cudaDeviceSynchronize());
  CHECK((dout.download() == std::vector<int32_t>{0, 0, 0, 0, 2, 2, 7, 15, 16, 16, 16}));
}

template <class T>
void csr_add_case(const HostCsr<T>& X, const HostCsr<T>& Y) {
  auto dX = DeviceCsr<T>::upload(X), dY = DeviceCsr<T>::upload(Y);
  const int32_t m = X.rows;
  DeviceBuffer<char> temp(csr_add_temp_bytes(m));
  DeviceBuffer<int32_t> cp(size_t(m) + 1);
  const int64_t nnz = csr_add_symbolic(m, dX.row_ptr(), dX.col_idx(), dY.row_ptr(), dY.col_idx(), cp.data(),
                                       temp.data(), temp.size(), nullptr);
  DeviceBuffer<int32_t> cc(size_t(std::max<int64_t>(nnz, 1)));
  DeviceBuffer<T> cv(size_t(std::max<int64_t>(nnz, 1)));
  csr_add_numeric(m, dX.row_ptr(), dX.col_idx(), dX.vals(), dY.row_ptr(), dY.col_idx(), dY.vals(), cp.data(),
                  cc.data(), cv.data(), nullptr);
  RASP_CUDA(cudaDeviceSynchronize());
  HostCsr<T> S;
  S.rows = m;
  S.cols = X.cols;
  S.nnz = nnz;
  S.row_ptr = cp.download();
  S.col_idx = cc.download();
  S.vals = cv.download();
  S.col_idx.resize(size_t(nnz));
  S.vals.resize(size_t(nnz));
  S.validate();
  // Reference: X + Y = [X Y] * [I; I].
  HostCsr<T> XY, II;
  XY.rows = m;
  XY.cols = 2 * X.cols;
  XY.row_ptr = {0};
  for (int32_t r = 0; r < m; ++r) {
    for (int32_t k = X.row_ptr[size_t(r)]; k < X.row_ptr[size_t(r) + 1]; ++k) {
      XY.col_idx.push_back(X.col_idx[size_t(k)]);
      XY.vals.push_back(X.vals[size_t(k)]);
    }
    for (int32_t k = Y.row_ptr[size_t(r)]; k < Y.row_ptr[size_t(r) + 1]; ++k) {
      XY.col_idx.push_back(X.cols + Y.col_idx[size_t(k)]);
      XY.vals.push_back(Y.vals[size_t(k)]);
    }
    XY.row_ptr.push_back(int32_t(XY.col_idx.size()));
  }
  XY.nnz = int64_t(XY.col_idx.size());
  II.rows = 2 * X.cols;
  II.cols = X.cols;
  for (int32_t r = 0; r <= II.rows; ++r) II.row_ptr.push_back(r);
  for (int32_t r = 0; r < II.rows; ++r) {
    II.col_idx.push_back(r % X.cols);
    II.vals.push_back(1);
  }
  II.nnz = II.rows;
  expect_product(S, cpu_spgemm(XY, II), "csr_add");
}

TEST(csr_add_matches_reference) {
  auto X = rasp_test::random_csr<double>(300, 120, [](int32_t r) { return r % 5 == 0 ? 0 : r % 30; }, 31);
  auto Y = rasp_test::random_csr<double>(300, 120, [](int32_t r) { return r % 7 == 0 ? 0 : r == 9 ? 120 : r % 11; }, 32);
  csr_add_case(X, Y);
  csr_add_case(as<float>(X), as<float>(Y));
  auto Z = rasp_test::random_csr<double>(300, 120, [](int32_t) { return 0; }, 33);  // an empty operand
  csr_add_case(X, Z);
  csr_add_case(Z, Y);
}

TEST(spgemm_core_matches_wrapper_and_writes_pinned_output) {
  auto A = DeviceCsr<double>::upload(laplace3d(6));
  auto P = DeviceCsr<double>::upload(aggregation(6));
  auto ref = cusparse_spgemm(A, P).download();
  // Output arrays in mapped pinned memory (the zero-copy placement), workspaces on the device.
  PinnedCsr<double> C;
  C.reserve(A.rows(), P.cols(), ref.nnz);
  DeviceBuffer<int32_t> crp(size_t(A.rows()) + 1);
  DeviceBuffer<char> w1, w2;
  auto alloc = [&](SpgemmBuffer w, size_t bytes) -> void* {
    switch (w) {
      case SpgemmBuffer::Work1: w1.resize(bytes); return w1.data();
      case SpgemmBuffer::Work2: w2.resize(bytes); return w2.data();
      case SpgemmBuffer::CCol: return C.col_idx.device_ptr();
      case SpgemmBuffer::CVal: return C.vals.device_ptr();
    }
    return nullptr;
  };
  cusparseHandle_t h;
  RASP_CUSPARSE(cusparseCreate(&h));
  const auto call = cusparse_spgemm_view(h, A.view(), P.view(), crp.data(), alloc, nullptr);
  RASP_CUDA(cudaDeviceSynchronize());
  cusparseDestroy(h);
  CHECK(call.complete && call.nnz_c == ref.nnz);
  C.nnz = call.nnz_c;
  auto rp = crp.download();
  std::memcpy(C.row_ptr.data(), rp.data(), rp.size() * sizeof(int32_t));
  expect_product(C.to_host(), ref, "core with pinned output");
  // An allocator that refuses the second workspace reports an incomplete call with the sizes it reached.
  RASP_CUSPARSE(cusparseCreate(&h));
  auto refuse = [&](SpgemmBuffer w, size_t bytes) -> void* { return w == SpgemmBuffer::Work1 ? alloc(w, bytes) : nullptr; };
  const auto part = cusparse_spgemm_view(h, A.view(), P.view(), crp.data(), refuse, nullptr);
  RASP_CUDA(cudaDeviceSynchronize());
  cusparseDestroy(h);
  CHECK(!part.complete && part.buffer1_bytes > 0 && part.buffer2_bytes > 0);
}

// ---- the executor --------------------------------------------------------------------------------
// Alg. 4 and forced modes may legitimately find no partition for a small budget (BudgetTooSmall); the default RASP
// planner must always succeed on these budgets.
template <class T>
void run_case(const Case& c, SpgemmContext& ctx, const SpgemmOptions& opt, const std::string& label,
              const HostCsr<T>& ref) {
  const auto A = PinnedCsr<T>::from(as<T>(c.A));
  const auto B = PinnedCsr<T>::from(as<T>(c.B));
  PinnedCsr<T> C;
  SpgemmReport rep;
  try {
    ctx.multiply(A, B, C, opt, &rep);
  } catch (const BudgetTooSmall& e) {
    if (opt.planner == SpgemmPlanner::Rasp && !opt.mode) rasp_test::fail(__FILE__, __LINE__, label + ": " + e.what());
    return;
  } catch (const std::exception& e) {
    rasp_test::fail(__FILE__, __LINE__, label + ": " + e.what());
    return;
  }
  const auto got = C.to_host();
  got.validate();
  expect_product(got, ref, label);
  if (!(rep.peak_device_bytes <= ctx.budget())) rasp_test::fail(__FILE__, __LINE__, label + ": budget exceeded");
  if (rep.plan.mode != SpgemmMode::ZeroCopy && !(rep.bytes_h2d >= A.bytes()))
    rasp_test::fail(__FILE__, __LINE__, label + ": uploaded less than A");
  for (const auto& rc : rep.chunks)
    if (rc.p >= 0 && !(rc.comp_end >= rc.comp_start && rc.comp_start >= 0))
      rasp_test::fail(__FILE__, __LINE__, label + ": bad timeline");
  CHECK(rep.nnz_c == got.nnz && rep.total_ms > 0);
}

template <class T>
void every_mode() {
  for (const auto& c : cases()) {
    const double total = problem_bytes(c.A, c.B, sizeof(T));
    const auto ref = cpu_spgemm(as<T>(c.A), as<T>(c.B));
    for (double frac : {2.0, 0.5, 0.25}) {
      SpgemmContext ctx(size_t(total * frac + kSlack), false);
      const std::string tag = c.name + (sizeof(T) == 8 ? " fp64" : " fp32") + " budget x" + std::to_string(frac);
      for (int depth : {1, 2, 3}) {
        SpgemmOptions rasp;
        rasp.priors = test_priors();
        rasp.depth = depth;
        run_case<T>(c, ctx, rasp, tag + " rasp depth " + std::to_string(depth), ref);
        SpgemmOptions stat = rasp;
        stat.adaptive = false;
        run_case<T>(c, ctx, stat, tag + " rasp static depth " + std::to_string(depth), ref);
        SpgemmOptions alg4 = rasp;
        alg4.planner = SpgemmPlanner::Alg4;
        if (depth < 3) run_case<T>(c, ctx, alg4, tag + " alg4 depth " + std::to_string(depth), ref);
      }
      for (SpgemmMode m : {SpgemmMode::InCore, SpgemmMode::ZeroCopy, SpgemmMode::ChunkB, SpgemmMode::ChunkAC}) {
        SpgemmOptions forced;
        forced.priors = test_priors();
        forced.mode = m;
        run_case<T>(c, ctx, forced, tag + " forced " + to_string(m), ref);
      }
    }
  }
}

TEST(every_mode_fp64) { every_mode<double>(); }
TEST(every_mode_fp32) { every_mode<float>(); }

TEST(multi_block_b_and_chunk_ac_really_split_b) {
  const Case c{"RxA", transpose(aggregation(48)), laplace3d(48)};
  const auto s = estimate(host_view(c.A), host_view(c.B));
  SpgemmContext ctx(size_t(s.b_bytes(0, s.a_cols) * 0.6), false);  // B (~9.7 MB) cannot stay resident whole
  for (SpgemmMode m : {SpgemmMode::ChunkB, SpgemmMode::ChunkAC}) {
    SpgemmOptions o;
    o.priors = test_priors();
    o.mode = m;
    const auto A = PinnedCsr<double>::from(c.A), B = PinnedCsr<double>::from(c.B);
    PinnedCsr<double> C;
    SpgemmReport rep;
    ctx.multiply(A, B, C, o, &rep);
    CHECK(rep.plan.parts_b() > 1);
    expect_product(C.to_host(), cpu_spgemm(c.A, c.B), std::string("split B ") + to_string(m));
  }
}

TEST(wrong_priors_force_splits_and_replans) {
  const Case c{"AxP", laplace3d(40), aggregation(40)};
  const auto A = PinnedCsr<double>::from(c.A), B = PinnedCsr<double>::from(c.B);
  SpgemmContext ctx(size_t(problem_bytes(c.A, c.B, 8) * 0.3 + kSlack), false);
  SpgemmOptions o;
  CostParams lie;  // claims cuSPARSE needs no workspace and C is 100x smaller than it is
  lie.ws_base_bytes = 0;
  lie.ws_bytes_per_flop = 0;
  lie.gamma = 0.01;
  lie.flops_per_ms = 1e3;
  o.priors = lie;
  o.mode = SpgemmMode::ChunkB;
  PinnedCsr<double> C;
  SpgemmReport rep;
  ctx.multiply(A, B, C, o, &rep);
  CHECK(rep.splits > 0);   // blocks sized for the lie did not fit and were halved
  CHECK(rep.replans > 0);  // the probe block corrected the model
  CHECK(rep.measured.gamma > 0.3 && rep.measured.ws_bytes_per_flop > 0);
  expect_product(C.to_host(), cpu_spgemm(c.A, c.B), "wrong priors");
}

TEST(budget_too_small_and_empty_products) {
  const auto A = PinnedCsr<double>::from(laplace3d(8)), P = PinnedCsr<double>::from(aggregation(8));
  SpgemmContext tiny(4096, false);
  PinnedCsr<double> C;
  CHECK_THROWS(tiny.multiply(A, P, C));
  auto Z = aggregation(8);
  std::fill(Z.row_ptr.begin(), Z.row_ptr.end(), 0);
  Z.col_idx.clear();
  Z.vals.clear();
  Z.nnz = 0;
  const auto Zp = PinnedCsr<double>::from(Z);
  SpgemmContext ctx(1 << 20, false);
  ctx.multiply(A, Zp, C);
  CHECK(C.nnz == 0 && C.rows == A.rows && C.row_ptr.data()[A.rows] == 0);
  CHECK_THROWS(ctx.multiply(P, P, C));  // dimension mismatch
}

TEST(reused_output_and_context_calibration) {
  SpgemmContext ctx(64 << 20);  // with the PCIe calibration
  CHECK(ctx.calibration().h2d_gbs > 0.1 && ctx.calibration().d2h_gbs > 0.1);
  const auto A = PinnedCsr<double>::from(laplace3d(10)), P = PinnedCsr<double>::from(aggregation(10));
  PinnedCsr<double> C;
  SpgemmReport r1, r2;
  ctx.multiply(A, P, C, {}, &r1);
  const auto first = C.to_host();
  const int32_t* buf = C.col_idx.data();
  ctx.multiply(A, P, C, {}, &r2);
  CHECK(C.col_idx.data() == buf);  // capacity reused, no new pinned allocation
  CHECK(r2.setup_ms <= r1.setup_ms + 1.0);
  expect_product(C.to_host(), first, "repeat");
  auto D = spgemm(A, P, 64 << 20);
  expect_product(D.to_host(), first, "one-shot spgemm()");
}

RASP_TEST_MAIN
