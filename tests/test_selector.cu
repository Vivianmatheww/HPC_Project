// Phase 3: GPU converters vs host converters, GPU features vs CPU features, selectors and SpmvPlan.
#include <random>

#include "rasp/features.hpp"
#include "rasp/formats_device.hpp"
#include "rasp/io.hpp"
#include "rasp/plan.hpp"
#include "rasp/reference.hpp"
#include "rasp/selector.hpp"
#include "test_util.hpp"

using namespace rasp;

namespace {

std::vector<HostCsr<double>> test_matrices() {
  using rasp_test::random_csr;
  std::vector<HostCsr<double>> m;
  for (const char* f : {"general_3x3.mtx", "empty_rows.mtx", "pattern.mtx", "one_by_one.mtx", "empty.mtx"})
    m.push_back(read_matrix_market(rasp_test::data_path(f), CacheMode::Off));
  m.push_back(random_csr<double>(1000, 800, [](int32_t r) { return r % 20 == 0 ? 0 : 1 + r % 13; }, 1));
  m.push_back(random_csr<double>(3000, 3000, [](int32_t r) { return r == 7 ? 3000 : 1 + r % 3; }, 2));
  m.push_back(random_csr<double>(5000, 6000, [](int32_t r) { return r == 11 ? 5000 : (r * 7919) % 37; }, 4));
  m.push_back(random_csr<double>(20001, 1000, [](int32_t r) { return r % 97 == 0 ? 60 : 2 + r % 5; }, 5));
  return m;
}

template <class T>
bool same(const DeviceBuffer<T>& d, const std::vector<T>& h) {
  return d.download() == h;
}

}  // namespace

// Runs host() and gpu(); both must refuse (FormatInfeasible) or both must succeed and agree.
template <class H, class G, class Cmp>
void compare(const char* what, H&& host, G&& gpu, Cmp&& cmp) {
  bool host_ok = true, gpu_ok = true;
  decltype(host()) h;
  decltype(gpu()) g;
  try { h = host(); } catch (const FormatInfeasible&) { host_ok = false; }
  try { g = gpu(); } catch (const FormatInfeasible&) { gpu_ok = false; }
  if (host_ok != gpu_ok) rasp_test::fail(__FILE__, __LINE__, std::string(what) + ": only one side refused");
  else if (host_ok && !cmp(h, g)) rasp_test::fail(__FILE__, __LINE__, std::string(what) + ": GPU != host");
}

TEST(gpu_converters_match_host_converters_bitwise) {
  for (const auto& A : test_matrices()) {
    const auto d = DeviceCsr<double>::upload(A);
    compare("ell", [&] { return build_ell(A); }, [&] { return convert_ell_gpu(d); },
            [](auto& h, auto& g) { return g.width == h.width && same(g.col, h.col) && same(g.val, h.val); });
    for (int C : {8, 32})
      for (int sigma : {1, 256, 0})
        compare("sell", [&] { return build_sell(A, C, sigma); }, [&] { return convert_sell_gpu(d, C, sigma); },
                [](auto& h, auto& g) {
                  return g.slices == h.slices && same(g.slice_ptr, h.slice_ptr) && same(g.slice_width, h.slice_width) &&
                         same(g.perm, h.perm) && same(g.col, h.col) && same(g.val, h.val);
                });
    auto arg_eq = [](auto& h, auto& g) {
      return g.groups == int32_t(h.groups.size()) && same(g.threads_end, h.threads_end) && same(g.col, h.col) &&
             same(g.val, h.val);
    };
    for (int D : {1, 4, 32})
      compare("argcsr", [&] { return build_argcsr(A, 128, D); }, [&] { return convert_argcsr_gpu(d, 128, D); }, arg_eq);
    compare("argcsr_local", [&] { return build_argcsr_local(A, 128, 32); },
            [&] { return convert_argcsr_gpu(d, 128, 0, 32); }, arg_eq);
    for (int t : {1, 4})
      compare("binned", [&] { return build_binned(A, t); }, [&] { return convert_binned_gpu(d, t); },
              [](auto& h, auto& g) {
                return g.blocks == h.blocks && g.bin_row_start == h.bin_row_start &&
                       g.bin_block_start == h.bin_block_start && same(g.order, h.order);
              });
  }
}

TEST(gpu_features_match_cpu) {
  FeatureExtractor fx;
  for (const auto& A : test_matrices()) {
    const auto g = fx.extract(DeviceCsr<double>::upload(A));
    const auto c = cpu_features(A);
    const auto a = feature_vector(g, true), b = feature_vector(c, true);
    CHECK(a.size() == feature_names().size());
    for (size_t i = 0; i < a.size(); ++i) CHECK_NEAR(a[i], b[i], 1e-9 * std::max(1.0, std::abs(b[i])));
    CHECK(g.max_len == c.max_len && g.nnz == A.nnz);
  }
  // Hand-checkable case: empty_rows.mtx is 5x4 with rows of length 0,1,0,2,0.
  const auto f = cpu_features(read_matrix_market(rasp_test::data_path("empty_rows.mtx"), CacheMode::Off));
  CHECK(f.max_len == 2);
  CHECK_NEAR(f.empty_frac, 0.6, 1e-12);
  CHECK_NEAR(f.mean_len, 0.6, 1e-12);
  CHECK_NEAR(f.ell_fill, 5.0 * 2 / 3, 1e-12);
  CHECK_NEAR(f.sell32_fill, 32.0 * 2 / 3, 1e-12);
  CHECK_NEAR(f.len_hist[0], 0.8, 1e-12);  // lengths 0,1,0,0 -> bucket "<=1"
  CHECK_NEAR(f.len_hist[1], 0.2, 1e-12);  // length 2
}

TEST(generated_tree_matches_library_features) { CHECK(check_generated_features().empty()); }

TEST(selectors_return_valid_configs) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(0, 1);
  for (int i = 0; i < 2000; ++i) {
    MatrixFeatures f;
    f.rows = int32_t(1 + u(rng) * 1e6);
    f.cols = f.rows;
    f.nnz = int64_t(f.rows * (1 + u(rng) * 50));
    f.mean_len = double(f.nnz) / f.rows;
    f.cv_len = u(rng) * 5;
    f.max_len = int32_t(f.mean_len * (1 + u(rng) * 1000));
    f.long_nnz_frac = u(rng) < 0.1 ? u(rng) : 0;
    f.ell_fill = u(rng) * 100;
    f.sell32_fill = 1 + u(rng) * 10;
    f.len_hist[size_t(i % kLenBuckets)] = 1;
    for (bool fp64 : {false, true}) {
      for (auto c : {select_tree(f, fp64), select_rules(f, fp64)}) CHECK(!c.config.label().empty() && c.predicted_speedup > 0);
      CHECK(!needs_conversion(select_tree(f, fp64, false).config));
      CHECK(!needs_conversion(select_rules(f, fp64, false).config));
    }
  }
}

template <class T>
void plan_suite() {
  const double tol = sizeof(T) == 8 ? 1e-12 : 1e-5;
  for (const auto& A64 : test_matrices()) {
    const auto h = convert_values<T>(A64);
    const auto d = DeviceCsr<T>::upload(h);
    const auto x = random_vector<T>(size_t(h.cols));
    DeviceBuffer<T> dx, dy(size_t(h.rows));
    dx.upload(x);
    const auto ref = cpu_spmv(h, x);
    for (auto sel : {SelectorKind::AlwaysCusparse, SelectorKind::Rules, SelectorKind::Tree})
      for (int64_t n : {int64_t(1), int64_t(1000000)})
        for (bool be : {true, false}) {
          PlanOptions o;
          o.selector = sel;
          o.expected_calls = n;
          o.break_even = be;
          SpmvPlan<T> plan(d, dx.data(), dy.data(), o);
          RASP_CUDA(cudaMemset(dy.data(), 0xff, dy.bytes()));
          plan.run();
          RASP_CUDA(cudaDeviceSynchronize());
          const double err = rel_l2_error(dy.download(), ref);
          if (!(err <= tol))
            rasp_test::fail(__FILE__, __LINE__, std::string(to_string(sel)) + " N=" + std::to_string(n) + " -> " +
                                                    plan.report().config.label() + ": rel err " + std::to_string(err));
          const auto& r = plan.report();
          CHECK(r.plan_ms >= 0 && !r.reason.empty());
          if (sel == SelectorKind::AlwaysCusparse) CHECK(r.config.label() == "cusparse_alg1" && !r.used_features);
          if (be && r.config.label() != r.steady_state.label()) CHECK(!needs_conversion(r.config));
        }
  }
}

TEST(plan_fp64) { plan_suite<double>(); }
TEST(plan_fp32) { plan_suite<float>(); }

TEST(break_even_skips_features_for_one_call_on_small_matrix) {
  auto A = DeviceCsr<double>::upload(rasp_test::random_csr<double>(200, 200, [](int32_t) { return 5; }, 3));
  DeviceBuffer<double> x(200), y(200);
  PlanOptions o;
  o.expected_calls = 1;
  SpmvPlan<double> p(A, x.data(), y.data(), o);
  CHECK(!p.report().used_features && p.report().config.label() == "cusparse_alg1");
}

RASP_TEST_MAIN
