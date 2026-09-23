// cuSPARSE SpMV wrapper against the CPU reference, on the edge cases Phase 2 kernels will face.
#include "rasp/cusparse_ref.hpp"
#include "rasp/io.hpp"
#include "rasp/reference.hpp"
#include "test_util.hpp"

using namespace rasp;

namespace {

template <class T>
double tolerance() { return sizeof(T) == 8 ? 1e-12 : 1e-5; }

template <class T>
double spmv_error(const HostCsr<T>& h, SpmvAlg alg) {
  auto d = DeviceCsr<T>::upload(h);
  auto x = random_vector<T>(size_t(h.cols));
  DeviceBuffer<T> dx, dy(size_t(h.rows));
  dx.upload(x);
  RASP_CUDA(cudaMemset(dy.data(), 0xff, dy.bytes()));  // garbage, so a missing write is caught
  CusparseSpmv<T> op(d, alg, dx.data(), dy.data());
  CHECK(op.preprocess_ms() >= 0);
  op.run(nullptr);
  RASP_CUDA(cudaDeviceSynchronize());
  return rel_l2_error(dy.download(), cpu_spmv(h, x));
}

template <class T>
void check_all_algs(const HostCsr<T>& h, const char* what) {
  for (auto alg : {SpmvAlg::CsrAlg1, SpmvAlg::CsrAlg2}) {
    const double err = spmv_error(h, alg);
    if (!(err <= tolerance<T>()))
      rasp_test::fail(__FILE__, __LINE__, std::string(what) + " " + to_string(alg) + (sizeof(T) == 8 ? " fp64" : " fp32") +
                                              ": rel err " + std::to_string(err));
  }
}

template <class T>
void run_suite() {
  for (const char* f : {"general_3x3.mtx", "empty_rows.mtx", "pattern.mtx", "skew.mtx", "one_by_one.mtx"})
    check_all_algs(convert_values<T>(read_matrix_market(rasp_test::data_path(f), CacheMode::Off)), f);

  // Random rectangular matrix with ~5% empty rows.
  check_all_algs(rasp_test::random_csr<T>(1000, 800, [](int32_t r) { return r % 20 == 0 ? 0 : 1 + r % 13; }, 1),
                 "random 1000x800");
  // One dense row among very short rows: the case ArgCSR is designed for.
  check_all_algs(rasp_test::random_csr<T>(2000, 2000, [](int32_t r) { return r == 7 ? 2000 : 1 + r % 3; }, 2),
                 "one dense row");
  // Every row long (warp-per-row regime).
  check_all_algs(rasp_test::random_csr<T>(300, 5000, [](int32_t) { return 400; }, 3), "long rows");
}

}  // namespace

TEST(spmv_fp64) { run_suite<double>(); }
TEST(spmv_fp32) { run_suite<float>(); }

TEST(spmv_empty_matrix_writes_zeros) {
  auto h = convert_values<double>(read_matrix_market(rasp_test::data_path("empty.mtx"), CacheMode::Off));
  auto d = DeviceCsr<double>::upload(h);
  DeviceBuffer<double> dx(3), dy(3);
  RASP_CUDA(cudaMemset(dy.data(), 0xff, dy.bytes()));
  CusparseSpmv<double> op(d, SpmvAlg::CsrAlg1, dx.data(), dy.data());
  op.run(nullptr);
  RASP_CUDA(cudaDeviceSynchronize());
  CHECK((dy.download() == std::vector<double>{0, 0, 0}));
}

TEST(device_csr_round_trip) {
  auto h = rasp_test::random_csr<double>(50, 40, [](int32_t r) { return r % 7; }, 4);
  auto back = DeviceCsr<double>::upload(h).download();
  CHECK(back.row_ptr == h.row_ptr && back.col_idx == h.col_idx && back.vals == h.vals && back.nnz == h.nnz);
}

RASP_TEST_MAIN
