// Out-of-core SpGEMM planning, host only (no GPU needed): the Laplace gallery against gen_laplace.py.
#include "rasp/gallery.hpp"
#include "rasp/io.hpp"
#include "test_util.hpp"

using namespace rasp;

namespace {

void check_same(const HostCsr<double>& a, const HostCsr<double>& b, const char* what) {
  const bool same = a.rows == b.rows && a.cols == b.cols && a.nnz == b.nnz && a.row_ptr == b.row_ptr &&
                    a.col_idx == b.col_idx && a.vals == b.vals;
  if (!same) rasp_test::fail(__FILE__, __LINE__, std::string(what) + " differs from gen_laplace.py");
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

RASP_TEST_MAIN
