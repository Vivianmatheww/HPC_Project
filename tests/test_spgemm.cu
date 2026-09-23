// cuSPARSE SpGEMM wrapper against a hand-computed product and the CPU reference.
#include "rasp/cusparse_ref.hpp"
#include "rasp/reference.hpp"
#include "test_util.hpp"

using namespace rasp;

namespace {

HostCsr<double> make(int32_t rows, int32_t cols, std::vector<int32_t> rp, std::vector<int32_t> ci, std::vector<double> v) {
  HostCsr<double> A;
  A.rows = rows;
  A.cols = cols;
  A.nnz = int64_t(ci.size());
  A.row_ptr = std::move(rp);
  A.col_idx = std::move(ci);
  A.vals = std::move(v);
  A.validate();
  return A;
}

template <class T>
HostCsr<T> gpu_product(const HostCsr<T>& A, const HostCsr<T>& B, SpgemmInfo* info = nullptr) {
  auto dA = DeviceCsr<T>::upload(A);
  auto dB = DeviceCsr<T>::upload(B);
  return cusparse_spgemm(dA, dB, info).download();
}

}  // namespace

TEST(hand_computed_product) {
  // A = [1 0 2; 0 3 0], B = [1 1; 0 2; 4 0]  ->  C = [9 1; 0 6]
  auto A = make(2, 3, {0, 2, 3}, {0, 2, 1}, {1, 2, 3});
  auto B = make(3, 2, {0, 2, 3, 4}, {0, 1, 1, 0}, {1, 1, 2, 4});
  SpgemmInfo info;
  auto C = gpu_product(A, B, &info);
  C.validate();  // also proves cuSPARSE returns sorted columns
  CHECK(C.rows == 2 && C.cols == 2 && C.nnz == 3 && info.nnz_c == 3);
  CHECK((C.row_ptr == std::vector<int32_t>{0, 2, 3}));
  CHECK((C.col_idx == std::vector<int32_t>{0, 1, 1}));
  CHECK((C.vals == std::vector<double>{9, 1, 6}));
  CHECK(csr_max_rel_diff(C, cpu_spgemm(A, B)) == 0);
}

template <class T>
void random_products(double tol) {
  auto A = rasp_test::random_csr<T>(300, 200, [](int32_t r) { return r % 11 == 0 ? 0 : 1 + r % 9; }, 5);
  auto B = rasp_test::random_csr<T>(200, 250, [](int32_t r) { return r == 3 ? 250 : r % 6; }, 6);
  auto C = gpu_product(A, B);
  C.validate();
  const double d = csr_max_rel_diff(C, cpu_spgemm(A, B));
  if (!(d <= tol)) rasp_test::fail(__FILE__, __LINE__, "random product rel diff " + std::to_string(d));

  auto S = rasp_test::random_csr<T>(400, 400, [](int32_t r) { return 1 + r % 5; }, 7);  // square, A*A
  const double d2 = csr_max_rel_diff(gpu_product(S, S), cpu_spgemm(S, S));
  if (!(d2 <= tol)) rasp_test::fail(__FILE__, __LINE__, "square product rel diff " + std::to_string(d2));
}

TEST(random_products_fp64) { random_products<double>(1e-12); }
TEST(random_products_fp32) { random_products<float>(1e-5); }

TEST(empty_operand_gives_empty_product) {
  auto A = make(2, 3, {0, 2, 3}, {0, 2, 1}, {1, 2, 3});
  auto Z = make(3, 4, {0, 0, 0, 0}, {}, {});
  auto C = gpu_product(A, Z);
  C.validate();
  CHECK(C.rows == 2 && C.cols == 4 && C.nnz == 0);
}

TEST(dimension_mismatch_throws) {
  auto A = DeviceCsr<double>::upload(make(2, 3, {0, 2, 3}, {0, 2, 1}, {1, 2, 3}));
  CHECK_THROWS(cusparse_spgemm(A, A));
}

RASP_TEST_MAIN
