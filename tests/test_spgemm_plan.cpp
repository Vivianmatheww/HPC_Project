// Out-of-core SpGEMM planning, host only (no GPU needed): the Laplace gallery against gen_laplace.py, the flop/size
// estimator against brute force, paper 1's BinarySearch splitter, the pipeline simulator and both planners.
#include <algorithm>

#include "rasp/gallery.hpp"
#include "rasp/io.hpp"
#include "rasp/reference.hpp"
#include "rasp/spgemm_plan.hpp"
#include "test_util.hpp"

using namespace rasp;

namespace {

void check_same(const HostCsr<double>& a, const HostCsr<double>& b, const char* what) {
  const bool same = a.rows == b.rows && a.cols == b.cols && a.nnz == b.nnz && a.row_ptr == b.row_ptr &&
                    a.col_idx == b.col_idx && a.vals == b.vals;
  if (!same) rasp_test::fail(__FILE__, __LINE__, std::string(what) + " differs from gen_laplace.py");
}

CostParams no_workspace() {
  CostParams p;
  p.ws_base_bytes = 0;
  p.ws_bytes_per_flop = 0;
  return p;
}

// The test matrices are a few hundred KB, so the 1 MiB workspace prior would dwarf them; scale it down.
CostParams small_workspace() {
  CostParams p;
  p.ws_base_bytes = 4096;
  return p;
}

// Boundaries are strictly increasing from 0 to n.
bool covers(const std::vector<int32_t>& b, int32_t n) {
  if (b.size() < 2 || b.front() != 0 || b.back() != n) return false;
  for (size_t i = 1; i < b.size(); ++i)
    if (b[i] <= b[i - 1]) return false;
  return true;
}

struct Laplace {
  HostCsr<double> A, P, R;
  explicit Laplace(int n) : A(laplace3d(n)), P(aggregation(n)), R(transpose(P)) {}
};

}  // namespace

// ---- gallery ---------------------------------------------------------------------------------
TEST(gallery_matches_gen_laplace) {
  const std::string dir = rasp_test::data_path("laplace3d_n8/");
  Laplace L(8);
  check_same(L.A, read_matrix_market(dir + "A.mtx", CacheMode::Off), "A");
  check_same(L.P, read_matrix_market(dir + "P.mtx", CacheMode::Off), "P");
  check_same(L.R, read_matrix_market(dir + "R.mtx", CacheMode::Off), "R");
  L.A.validate();
  L.P.validate();
  L.R.validate();
}

TEST(gallery_edge_cases) {
  auto one = laplace3d(1);
  CHECK(one.rows == 1 && one.nnz == 1 && one.vals[0] == 6);
  CHECK_THROWS(aggregation(3));
  CHECK_THROWS(aggregation(0));
  CHECK_THROWS(laplace3d(2000));  // n^3 rows overflow 32-bit indices
  auto P = aggregation(2);
  CHECK(P.rows == 8 && P.cols == 1 && P.nnz == 8);
}

TEST(transpose_roundtrip) {
  auto A = rasp_test::random_csr<double>(70, 45, [](int32_t r) { return r % 7; }, 3);
  auto T = transpose(A);
  T.validate();
  CHECK(T.rows == 45 && T.cols == 70 && T.nnz == A.nnz);
  auto TT = transpose(T);
  CHECK(TT.row_ptr == A.row_ptr && TT.col_idx == A.col_idx && TT.vals == A.vals);
}

// ---- estimator -------------------------------------------------------------------------------
TEST(estimate_matches_brute_force) {
  auto A = rasp_test::random_csr<double>(120, 90, [](int32_t r) { return r % 13 == 0 ? 0 : 1 + r % 6; }, 11);
  auto B = rasp_test::random_csr<double>(90, 40, [](int32_t r) { return r == 5 ? 40 : r % 9; }, 12);
  const auto s = estimate(host_view(A), host_view(B));
  const auto C = cpu_spgemm(A, B);
  CHECK(s.a_rows == 120 && s.a_cols == 90 && s.b_cols == 40 && s.value_bytes == 8);
  for (int32_t i = 0; i < A.rows; ++i) {
    int64_t f = 0;
    for (int32_t k = A.row_ptr[size_t(i)]; k < A.row_ptr[size_t(i) + 1]; ++k) {
      const int32_t c = A.col_idx[size_t(k)];
      f += B.row_ptr[size_t(c) + 1] - B.row_ptr[size_t(c)];
    }
    CHECK(s.flops(i, i + 1) == f);
    CHECK(s.c_upper(i, i + 1) == std::min<int64_t>(f, 40));
    CHECK(C.row_ptr[size_t(i) + 1] - C.row_ptr[size_t(i)] <= s.c_upper(i, i + 1));  // a true upper bound
  }
  CHECK(s.a_bytes(0, 120) == 121.0 * 4 + double(A.nnz) * 12);
  CHECK(s.b_bytes(0, 90) == 91.0 * 4 + double(B.nnz) * 12);
}

TEST(estimate_laplace_compression) {
  // A*P: an interior row of A (7 cells) touches 4 aggregates; R*A: an interior aggregate touches 32 cells.
  Laplace L(8);
  const auto ap = estimate(host_view(L.A), host_view(L.P));
  CHECK(ap.flops(0, L.A.rows) == L.A.nnz);
  const auto C = cpu_spgemm(L.A, L.P);
  const int32_t interior = 3 * 64 + 3 * 8 + 3;  // cell (3, 3, 3)
  CHECK(C.row_ptr[size_t(interior) + 1] - C.row_ptr[size_t(interior)] == 4);
  CHECK(ap.c_upper(interior, interior + 1) == 7);
  const auto ra = estimate(host_view(L.R), host_view(L.A));
  CHECK(ra.flops(0, L.R.rows) == L.A.nnz);  // every A row is hit exactly once (R = P^T is a partition)
}

TEST(estimate_reuse_gives_same_result) {
  Laplace L(8);
  ProblemSizes s;
  estimate(host_view(L.R), host_view(L.A), s);  // a bigger result first, then reuse its arrays
  estimate(host_view(L.A), host_view(L.P), s);
  const auto fresh = estimate(host_view(L.A), host_view(L.P));
  CHECK(s.a_rows == fresh.a_rows && s.b_cols == fresh.b_cols);
  CHECK(std::equal(s.flops_ptr.begin(), s.flops_ptr.end(), fresh.flops_ptr.begin(), fresh.flops_ptr.end()));
  CHECK(std::equal(s.cub_ptr.begin(), s.cub_ptr.end(), fresh.cub_ptr.begin(), fresh.cub_ptr.end()));
}

TEST(estimate_rejects_mismatch) {
  Laplace L(4);
  CHECK_THROWS(estimate(host_view(L.P), host_view(L.P)));
}

// ---- BinarySearch splitter -------------------------------------------------------------------
TEST(split_rows_is_maximal_and_bounded) {
  std::vector<double> w{5, 1, 1, 7, 2, 2, 2, 9, 0, 0, 3};
  std::vector<double> pre{0};
  for (double x : w) pre.push_back(pre.back() + x);
  auto size = [&](int32_t a, int32_t b) { return pre[size_t(b)] - pre[size_t(a)]; };
  const auto b = split_rows(0, int32_t(w.size()), size, 9);
  CHECK(covers(b, int32_t(w.size())));
  for (size_t i = 0; i + 1 < b.size(); ++i) {
    CHECK(size(b[i], b[i + 1]) <= 9);
    if (b[i + 1] < int32_t(w.size())) CHECK(size(b[i], b[i + 1] + 1) > 9);  // could not take one more row
  }
  CHECK((b == std::vector<int32_t>{0, 3, 5, 7, 10, 11}));
  CHECK((split_rows(4, 4, size, 9) == std::vector<int32_t>{4}));
  CHECK((split_rows(2, 6, size, 100) == std::vector<int32_t>{2, 6}));
  CHECK_THROWS(split_rows(0, int32_t(w.size()), size, 8));  // row 7 alone weighs 9
}

// ---- pipeline simulator ----------------------------------------------------------------------
TEST(simulate_known_schedules) {
  std::vector<StageCost> eq(5, StageCost{1, 1, 0});
  CHECK_NEAR(simulate(eq, 1), 10, 1e-12);  // no overlap
  CHECK_NEAR(simulate(eq, 2), 6, 1e-12);   // fill + 5 overlapped stages
  std::vector<StageCost> copy_bound(4, StageCost{1, 0.1, 0});
  CHECK_NEAR(simulate(copy_bound, 2), 4.1, 1e-12);
  std::vector<StageCost> comp_bound(4, StageCost{0.1, 1, 0});
  CHECK_NEAR(simulate(comp_bound, 2), 4.1, 1e-12);
  CHECK_NEAR(simulate({}, 2), 0, 0);
  // With downloads on the same engine: never worse than sequential, never better than the busier resource.
  std::vector<StageCost> mixed{{1, 2, 0.5}, {0.5, 1, 1}, {2, 0.5, 0.2}, {1, 1, 1}};
  for (int d : {2, 3}) {
    const double t = simulate(mixed, d);
    CHECK(t <= simulate(mixed, 1) + 1e-12);
    CHECK(t >= 4.5 - 1e-12);  // total compute
    CHECK(t >= 7.2 - 1e-12);  // total copy (one engine)
  }
}

// ---- Alg. 4 (paper 1) ------------------------------------------------------------------------
namespace {
// 4 x 4 A and B with one nonzero per row (diagonal), C has one per row: every size is easy to compute by hand.
// a_bytes(rows r) = (r + 1) * 4 + 12 r.
ProblemSizes tiny_diagonal() {
  static const int32_t rp[] = {0, 1, 2, 3, 4};
  ProblemSizes s;
  s.a_rows = s.a_cols = 4;
  s.b_cols = 4;
  s.value_bytes = 8;
  s.a_ptr = s.b_ptr = rp;
  s.flops_ptr = s.cub_ptr = {0, 1, 2, 3, 4};
  return s;
}
}  // namespace

TEST(alg4_hand_computed) {
  const auto s = tiny_diagonal();
  const auto p = no_workspace();
  // |A| = |B| = |C| = 68 bytes: in-core needs 204.
  auto incore = plan_alg4(s, 204, p);
  CHECK(incore.mode == SpgemmMode::InCore && incore.parts_ac() == 1);
  // Budget 160: chunk2 keeps B (68 <= 120) and streams one row of A+C at a time (40 <= 40): copy 68 + 68 = 136.
  // chunk1 keeps <= 3 rows of A+C, but B (68) does not fit the 40-byte streamed share, so it needs partial sums
  // (A + 3C: one row per block, P_AC = 4) and B blocks with full-height row pointers: copy 68 + 68 + 4 * 68 = 408.
  auto c = plan_alg4(s, 160, p);
  CHECK(c.mode == SpgemmMode::ChunkB);
  CHECK(c.parts_b() == 1 && c.parts_ac() == 4);
  CHECK(c.depth == 1);
  CHECK(covers(c.ac_bounds, 4) && covers(c.b_bounds, 4));
  CHECK_THROWS(plan_alg4(s, 30, p));  // nothing fits
}

TEST(alg4_respects_its_shares_on_laplace) {
  Laplace L(16);
  const CostParams p;
  for (auto [A, B] : {std::pair{&L.A, &L.P}, std::pair{&L.R, &L.A}}) {
    const auto s = estimate(host_view(*A), host_view(*B));
    const double total = s.a_bytes(0, s.a_rows) + s.b_bytes(0, s.a_cols) + s.c_bytes(0, s.a_rows, 1.0);
    for (double frac : {0.6, 0.4, 0.25}) {
      const double budget = total * frac + p.ws_base_bytes;
      SpgemmPlan plan;
      try {
        plan = plan_alg4(s, budget, p);
      } catch (const BudgetTooSmall&) {
        continue;
      }
      CHECK(covers(plan.ac_bounds, s.a_rows) && covers(plan.b_bounds, s.a_cols));
      const double ext = plan.parts_b() > 1 ? s.ext_bytes() : 0;
      for (size_t q = 0; q + 1 < plan.b_bounds.size(); ++q) {
        const double bq = s.b_bytes(plan.b_bounds[q], plan.b_bounds[q + 1]) + ext;
        if (plan.mode == SpgemmMode::ChunkB) CHECK(bq <= 0.75 * budget);
        if (plan.mode == SpgemmMode::ChunkAC) CHECK(bq <= 0.25 * budget);
      }
      CHECK(plan.predicted_ms > 0);
    }
  }
}

// ---- RASP planner ----------------------------------------------------------------------------
TEST(rasp_plans_fit_and_cover) {
  Laplace L(16);
  const CostParams p;
  for (auto [A, B] : {std::pair{&L.A, &L.P}, std::pair{&L.R, &L.A}}) {
    const auto s = estimate(host_view(*A), host_view(*B));
    const double total = s.a_bytes(0, s.a_rows) + s.b_bytes(0, s.a_cols) + s.c_bytes(0, s.a_rows, 1.0);
    for (double frac : {2.0, 1.0, 0.6, 0.4, 0.25, 0.15}) {
      const double budget = total * frac + p.ws_base_bytes;
      const auto plan = plan_rasp(s, budget, p);
      CHECK(covers(plan.ac_bounds, s.a_rows) && covers(plan.b_bounds, s.a_cols));
      CHECK(plan.peak_bytes <= budget * (1 + 1e-12));
      CHECK(plan.predicted_ms > 0);
      if (plan.mode != SpgemmMode::InCore && plan.mode != SpgemmMode::ZeroCopy) CHECK(plan.depth >= 2);
    }
  }
}

TEST(rasp_never_predicts_slower_than_alg4) {
  Laplace L(16);
  const CostParams p = small_workspace();
  const auto s = estimate(host_view(L.A), host_view(L.P));
  const double total = s.a_bytes(0, s.a_rows) + s.b_bytes(0, s.a_cols) + s.c_bytes(0, s.a_rows, 1.0);
  for (double frac : {1.5, 0.5, 0.3}) {
    const double budget = total * frac + p.ws_base_bytes;
    const auto a4 = plan_alg4(s, budget, p);
    const auto rp = plan_rasp(s, budget, p, {.depth = 0, .probe = false});
    CHECK(rp.predicted_ms <= a4.predicted_ms * (1 + 1e-9));  // the search space includes depth-2 versions of Alg. 4
  }
}

TEST(rasp_forced_mode_and_depth) {
  Laplace L(8);
  const auto s = estimate(host_view(L.A), host_view(L.P));
  const CostParams p;
  const double big = 1e9;
  for (auto mode : {SpgemmMode::InCore, SpgemmMode::ChunkB, SpgemmMode::ChunkAC, SpgemmMode::ZeroCopy}) {
    RaspPlanOptions o;
    o.has_forced_mode = true;
    o.forced_mode = mode;
    o.depth = 3;
    const auto plan = plan_rasp(s, big, p, o);
    CHECK(plan.mode == mode);
    if (mode == SpgemmMode::ChunkB || mode == SpgemmMode::ChunkAC) CHECK(plan.depth == 3);
  }
  CHECK_THROWS(plan_rasp(s, 1000, p));  // far below one row plus the workspace prior
}

TEST(rasp_probe_is_small_first_block) {
  Laplace L(16);
  const CostParams p = small_workspace();
  const auto s = estimate(host_view(L.A), host_view(L.P));
  const double budget = s.b_bytes(0, s.a_cols) + 0.2 * s.a_bytes(0, s.a_rows) + p.ws_base_bytes * 4;
  RaspPlanOptions o;
  o.has_forced_mode = true;
  o.forced_mode = SpgemmMode::ChunkB;
  const auto plan = plan_rasp(s, budget, p, o);
  CHECK(plan.probe);
  CHECK(plan.parts_ac() >= 3);
  const double f0 = double(s.flops(plan.ac_bounds[0], plan.ac_bounds[1]));
  const double f1 = double(s.flops(plan.ac_bounds[1], plan.ac_bounds[2]));
  CHECK(f0 * 4 < f0 + f1);  // probe ~ 1/8 of the original first block
}

TEST(replan_with_better_compression_uses_fewer_blocks) {
  Laplace L(16);
  CostParams p = small_workspace();
  const auto s = estimate(host_view(L.A), host_view(L.P));
  const double budget = s.b_bytes(0, s.a_cols) + 0.15 * (s.a_bytes(0, s.a_rows) + s.c_bytes(0, s.a_rows, 1.0)) +
                        p.ws_base_bytes * 4;
  RaspPlanOptions o;
  o.has_forced_mode = true;
  o.forced_mode = SpgemmMode::ChunkB;
  o.probe = false;
  o.depth = 2;
  const auto plan = plan_rasp(s, budget, p, o);
  CHECK(plan.parts_ac() >= 2);
  const int32_t from = plan.ac_bounds[1];
  CostParams measured = p;
  measured.gamma = 4.0 / 7.0;  // A*P keeps 4 of 7 products per row
  const auto tail = replan_rows(s, budget, plan, measured, from);
  CHECK(tail.front() == from && tail.back() == s.a_rows);
  for (size_t i = 1; i < tail.size(); ++i) CHECK(tail[i] > tail[i - 1]);
  int old_tail = 0;
  for (int32_t b : plan.ac_bounds)
    if (b > from) ++old_tail;
  CHECK(int(tail.size()) - 1 <= old_tail);
  CHECK((replan_rows(s, budget, plan, measured, s.a_rows) == std::vector<int32_t>{s.a_rows}));
}

TEST(needs_replan_threshold) {
  CostParams a, b;
  CHECK(!needs_replan(a, b, 0.2));
  b.flops_per_ms = a.flops_per_ms * 1.19;
  CHECK(!needs_replan(a, b, 0.2));
  b.flops_per_ms = a.flops_per_ms * 1.25;
  CHECK(needs_replan(a, b, 0.2));
  b = a;
  b.gamma = 0.57;
  CHECK(needs_replan(a, b, 0.2));
  b = a;
  b.ws_bytes_per_flop = a.ws_bytes_per_flop * 0.7;
  CHECK(needs_replan(a, b, 0.2));
}

RASP_TEST_MAIN
