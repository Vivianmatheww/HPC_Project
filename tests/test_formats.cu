// Phase 2 formats and kernels: converter structure (incl. the ArgCSR example from the paper) and every
// kernel configuration against the CPU reference.
#include <algorithm>

#include "rasp/formats.hpp"
#include "rasp/io.hpp"
#include "rasp/reference.hpp"
#include "rasp/spmv.hpp"
#include "test_util.hpp"

using namespace rasp;

namespace {

// Figure 3 of Heller & Oberhuber: rows 0-6 hold one nonzero each, row 7 is full (8 nonzeros).
HostCsr<double> paper_figure3() {
  HostCsr<double> A;
  A.rows = A.cols = 8;
  A.row_ptr = {0};
  for (int r = 0; r < 7; ++r) {
    A.col_idx.push_back(r);
    A.vals.push_back(r + 1);
    A.row_ptr.push_back(int32_t(A.col_idx.size()));
  }
  for (int c = 0; c < 8; ++c) {
    A.col_idx.push_back(c);
    A.vals.push_back(10 + c);
  }
  A.row_ptr.push_back(int32_t(A.col_idx.size()));
  A.nnz = int64_t(A.col_idx.size());
  A.validate();
  return A;
}

template <class T>
double kernel_error(const KernelConfig& cfg, const HostCsr<T>& h) {
  auto d = DeviceCsr<T>::upload(h);
  auto x = random_vector<T>(size_t(h.cols));
  DeviceBuffer<T> dx, dy(size_t(h.rows));
  dx.upload(x);
  RASP_CUDA(cudaMemset(dy.data(), 0xff, dy.bytes()));  // NaN garbage: any row not written is caught
  auto k = make_spmv_kernel(cfg, h, d, dx.data(), dy.data());
  k->run(nullptr);
  RASP_CUDA(cudaDeviceSynchronize());
  RASP_CUDA(cudaGetLastError());
  return rel_l2_error(dy.download(), cpu_spmv(h, x));
}

template <class T>
void all_configs_on(const HostCsr<T>& h, const char* what) {
  const double tol = sizeof(T) == 8 ? 1e-12 : 1e-5;
  for (const auto& cfg : all_configs()) {
    try {
      const double err = kernel_error(cfg, h);
      if (!(err <= tol))
        rasp_test::fail(__FILE__, __LINE__, std::string(what) + " " + cfg.label() + (sizeof(T) == 8 ? " fp64" : " fp32") +
                                                ": rel err " + std::to_string(err));
    } catch (const FormatInfeasible&) {
      // allowed (e.g. ELL on a matrix with one dense row)
    }
  }
}

template <class T>
void suite() {
  all_configs_on(convert_values<T>(paper_figure3()), "figure3");
  for (const char* f : {"general_3x3.mtx", "empty_rows.mtx", "pattern.mtx", "one_by_one.mtx", "empty.mtx"})
    all_configs_on(convert_values<T>(read_matrix_market(rasp_test::data_path(f), CacheMode::Off)), f);
  using rasp_test::random_csr;
  all_configs_on(random_csr<T>(1000, 800, [](int32_t r) { return r % 20 == 0 ? 0 : 1 + r % 13; }, 1), "random");
  all_configs_on(random_csr<T>(3000, 3000, [](int32_t r) { return r == 7 ? 3000 : 1 + r % 3; }, 2), "dense row");
  all_configs_on(random_csr<T>(300, 5000, [](int32_t) { return 400; }, 3), "long rows");
  all_configs_on(random_csr<T>(5000, 5000, [](int32_t r) { return (r * 7919) % 37; }, 4), "irregular");
  // More rows than one ArgCSR group / SELL window, and a row count that is not a multiple of C.
  all_configs_on(random_csr<T>(20001, 1000, [](int32_t r) { return r % 97 == 0 ? 60 : 2 + r % 5; }, 5), "many rows");
}

}  // namespace

TEST(argcsr_reproduces_paper_figure3) {
  const auto G = build_argcsr(paper_figure3(), /*block=*/12, /*desired_chunk=*/32);
  CHECK(G.groups.size() == 1);
  CHECK(G.groups[0].size == 8 && G.groups[0].chunk == 2);
  // Rows 0-6 keep one thread each; row 7 gets 4 (chunk 8 -> 2); the 12th thread stays free.
  CHECK((G.threads_end == std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 11}));
  CHECK(G.col.size() == 24);  // chunk 2 x 12 threads
  int64_t padding = 0;
  for (int32_t c : G.col) padding += c < 0;
  CHECK(padding == 24 - 15);  // 7 short rows pad one slot each + the free thread's 2 slots
  // Row 7's first chunk (thread 7) holds columns 0 and 1, interleaved with stride = block.
  CHECK(G.col[7] == 0 && G.col[7 + 12] == 1);
}

TEST(argcsr_group_budget) {
  // 10 rows of 5 nonzeros, budget desired_chunk*block = 2*8 = 16 -> 3 rows (15) per group, 4 groups.
  auto A = rasp_test::random_csr<double>(10, 20, [](int32_t) { return 5; }, 9);
  const auto G = build_argcsr(A, 8, 2);
  CHECK(G.groups.size() == 4);
  CHECK(G.groups[0].size == 3 && G.groups[3].size == 1);
  // A single row longer than the budget gets a group of its own.
  auto B = rasp_test::random_csr<double>(3, 100, [](int32_t r) { return r == 1 ? 90 : 2; }, 10);
  const auto H = build_argcsr(B, 8, 2);
  CHECK(H.groups.size() == 3 && H.groups[1].size == 1 && H.groups[1].chunk == (90 + 7) / 8);
}

TEST(sell_sorting_and_slices) {
  auto A = rasp_test::random_csr<double>(40, 50, [](int32_t r) { return r % 10; }, 11);
  const auto unsorted = build_sell(A, 8, 1);
  const auto sorted = build_sell(A, 8, 0);
  CHECK(unsorted.slices == 5 && sorted.slices == 5);
  CHECK(sorted.col.size() <= unsorted.col.size());  // sorting can only reduce padding
  for (int i = 0; i < 40; ++i) CHECK(unsorted.perm[size_t(i)] == i);
  // Sorted permutation lists rows by decreasing length.
  for (int i = 1; i < 40; ++i) {
    auto len = [&](int r) { return A.row_ptr[size_t(r) + 1] - A.row_ptr[size_t(r)]; };
    CHECK(len(sorted.perm[size_t(i) - 1]) >= len(sorted.perm[size_t(i)]));
  }
}

TEST(ell_refuses_heavy_padding) {
  auto A = rasp_test::random_csr<double>(1000, 1000, [](int32_t r) { return r == 0 ? 1000 : 1; }, 12);
  CHECK_THROWS(build_ell(A));
  CHECK(build_ell(A, 1e9).width == 1000);
}

TEST(config_labels_round_trip) {
  CHECK(phase2_configs().size() == 33);
  CHECK(extension_configs().size() == 7);
  const auto cfgs = all_configs();
  CHECK(cfgs.size() == 40);
  for (const auto& c : cfgs) CHECK(KernelConfig::parse(c.label()).label() == c.label());
  CHECK(KernelConfig::parse("argcsr_local_B128_R32").family() == "argcsr_local");
  CHECK_THROWS(KernelConfig::parse("csr_binned_t0"));
  CHECK_THROWS(KernelConfig::parse("csr_vector_v3"));
  CHECK_THROWS(KernelConfig::parse("argcsr_B100_D1"));
  CHECK_THROWS(KernelConfig::parse("nonsense"));
}

TEST(binned_bins) {
  CHECK(binned_bin(0, 2) == 0 && binned_bin(1, 2) == 0 && binned_bin(2, 2) == 0);   // 1 lane
  CHECK(binned_bin(3, 2) == 1 && binned_bin(4, 2) == 1);                           // 2 lanes
  CHECK(binned_bin(5, 2) == 2 && binned_bin(64, 2) == 5 && binned_bin(4000, 2) == 5);  // 4 .. 32 lanes (capped)
  CHECK(binned_bin(kLongRow + 1, 2) == kBins - 1);                                 // block per row
  auto A = rasp_test::random_csr<double>(1000, 6000, [](int32_t r) { return r == 5 ? 5000 : r % 40; }, 13);
  const auto H = build_binned(A, 2);
  CHECK(H.bin_row_start.front() == 0 && H.bin_row_start.back() == 1000);
  CHECK(H.bin_row_start[kBins] - H.bin_row_start[kBins - 1] == 1);  // the 5000-long row
  std::vector<int> seen(1000, 0);
  for (int32_t r : H.order) ++seen[size_t(r)];
  CHECK(std::all_of(seen.begin(), seen.end(), [](int c) { return c == 1; }));  // a permutation
  for (int k = 0; k < kBins; ++k)  // stable: rows stay in ascending order inside each bin
    for (int32_t i = H.bin_row_start[size_t(k)] + 1; i < H.bin_row_start[size_t(k) + 1]; ++i)
      CHECK(H.order[size_t(i) - 1] < H.order[size_t(i)]);
}

TEST(argcsr_local_chunk_rule) {
  CHECK(argcsr_local_chunk(4.0, 0.2, 128, 32) == 1);    // 32 rows x 4 = 128 nnz = 1 x 128
  CHECK(argcsr_local_chunk(16.0, 0.2, 128, 32) == 4);
  CHECK(argcsr_local_chunk(16.0, 3.0, 128, 32) == 2);   // irregular: halved
  CHECK(argcsr_local_chunk(1000.0, 0.1, 128, 64) == 32);  // clamped
  CHECK(argcsr_local_chunk(0.0, 0.0, 128, 32) == 1);
  // On a perfectly uniform matrix the local variant equals the global one with the implied D.
  auto A = rasp_test::random_csr<double>(9000, 500, [](int32_t) { return 16; }, 14);
  const auto L = build_argcsr_local(A, 128, 32);
  const auto G = build_argcsr(A, 128, 4);
  CHECK(L.groups.size() == G.groups.size() && L.col == G.col && L.threads_end == G.threads_end);
}

TEST(all_kernels_fp64) { suite<double>(); }
TEST(all_kernels_fp32) { suite<float>(); }

RASP_TEST_MAIN
