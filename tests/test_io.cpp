// Matrix Market reader, binary cache and CSR validation.
#include <filesystem>
#include <unistd.h>

#include "rasp/io.hpp"
#include "test_util.hpp"

using namespace rasp;
using rasp_test::data_path;

namespace {
HostCsr<double> read(const std::string& name) { return read_matrix_market(data_path(name), CacheMode::Off); }

bool same(const HostCsr<double>& a, const HostCsr<double>& b) {
  return a.rows == b.rows && a.cols == b.cols && a.nnz == b.nnz && a.row_ptr == b.row_ptr && a.col_idx == b.col_idx &&
         a.vals == b.vals;
}

std::filesystem::path temp_dir() {
  auto d = std::filesystem::temp_directory_path() / ("rasp_test_io_" + std::to_string(getpid()));
  std::filesystem::create_directories(d);
  return d;
}
}  // namespace

TEST(general_3x3) {
  auto A = read("general_3x3.mtx");
  A.validate();
  CHECK(A.rows == 3 && A.cols == 3 && A.nnz == 5);
  CHECK((A.row_ptr == std::vector<int32_t>{0, 2, 3, 5}));
  CHECK((A.col_idx == std::vector<int32_t>{0, 2, 1, 0, 2}));
  CHECK((A.vals == std::vector<double>{2.0, -1.0, 3.5, 4.0, 0.1}));
}

TEST(symmetric_expansion_matches_general) {
  auto S = read("symmetric.mtx");
  auto G = read("symmetric_expanded.mtx");
  CHECK(S.nnz == 6);  // 4 stored entries, 2 of them off-diagonal -> +2, diagonal not duplicated
  CHECK(same(S, G));
}

TEST(pattern_is_ones_and_rectangular) {
  auto A = read("pattern.mtx");
  CHECK(A.rows == 2 && A.cols == 3 && A.nnz == 3);
  CHECK((A.row_ptr == std::vector<int32_t>{0, 1, 3}));
  CHECK((A.col_idx == std::vector<int32_t>{1, 0, 2}));
  CHECK((A.vals == std::vector<double>{1, 1, 1}));
}

TEST(skew_symmetric_negates_mirror) {
  auto A = read("skew.mtx");
  CHECK(A.nnz == 4);
  CHECK((A.row_ptr == std::vector<int32_t>{0, 2, 3, 4}));
  CHECK((A.col_idx == std::vector<int32_t>{1, 2, 0, 0}));
  CHECK((A.vals == std::vector<double>{-3.0, 1.5, 3.0, -1.5}));
}

TEST(duplicates_are_summed) {
  auto A = read("duplicates.mtx");
  CHECK(A.nnz == 3);
  CHECK((A.row_ptr == std::vector<int32_t>{0, 1, 3}));
  CHECK((A.col_idx == std::vector<int32_t>{0, 0, 1}));
  CHECK((A.vals == std::vector<double>{1.5, 3.0, 2.0}));
}

TEST(empty_rows_including_first_and_last) {
  auto A = read("empty_rows.mtx");
  A.validate();
  CHECK(A.rows == 5 && A.cols == 4 && A.nnz == 3);
  CHECK((A.row_ptr == std::vector<int32_t>{0, 0, 1, 1, 3, 3}));
  CHECK((A.col_idx == std::vector<int32_t>{3, 0, 1}));
}

TEST(integer_field) {
  auto A = read("integer.mtx");
  CHECK((A.vals == std::vector<double>{7, -3}));
}

TEST(one_by_one_and_empty) {
  auto A = read("one_by_one.mtx");
  CHECK(A.rows == 1 && A.nnz == 1 && A.vals[0] == 42.0);
  auto E = read("empty.mtx");
  E.validate();
  CHECK(E.rows == 3 && E.cols == 3 && E.nnz == 0);
  CHECK((E.row_ptr == std::vector<int32_t>{0, 0, 0, 0}));
}

TEST(unsorted_input_is_sorted) {
  auto A = read("unsorted.mtx");
  A.validate();
  CHECK((A.col_idx == std::vector<int32_t>{0, 1, 2, 0, 3}));
  CHECK((A.vals == std::vector<double>{4, 5, 2, 3, 1}));
}

TEST(crlf_line_endings_and_plus_sign) {
  auto A = read("crlf.mtx");
  CHECK(A.nnz == 2);
  CHECK((A.vals == std::vector<double>{1.5, 2.5}));
}

TEST(rejects_unsupported_and_bad_files) {
  CHECK_THROWS(read("complex.mtx"));
  CHECK_THROWS(read("array.mtx"));
  CHECK_THROWS(read("out_of_range.mtx"));
  CHECK_THROWS(read("does_not_exist.mtx"));
}

TEST(binary_cache_round_trip) {
  const auto dir = temp_dir();
  const auto mtx = (dir / "m.mtx").string();
  std::filesystem::copy_file(data_path("unsorted.mtx"), mtx, std::filesystem::copy_options::overwrite_existing);
  auto first = read_matrix_market(mtx, CacheMode::Use);  // parses and writes the cache
  CHECK(std::filesystem::exists(cache_path_for(mtx)));
  auto second = read_matrix_market(mtx, CacheMode::Use);  // served from the cache
  CHECK(same(first, second));
  CHECK(same(first, read("unsorted.mtx")));

  // A corrupt cache is rebuilt, not trusted.
  { FILE* f = std::fopen(cache_path_for(mtx).c_str(), "w"); std::fputs("garbage", f); std::fclose(f); }
  CHECK(same(read_matrix_market(mtx, CacheMode::Use), first));
  std::filesystem::remove_all(dir);
}

TEST(write_then_read_round_trip) {
  const auto dir = temp_dir();
  auto A = read("skew.mtx");
  const auto out = (dir / "w.mtx").string();
  write_matrix_market(out, A);
  CHECK(same(read_matrix_market(out, CacheMode::Off), A));
  save_csr_binary((dir / "w.bin").string(), A);
  CHECK(same(load_csr_binary((dir / "w.bin").string()), A));
  std::filesystem::remove_all(dir);
}

TEST(convert_values_to_float) {
  auto A = read("general_3x3.mtx");
  auto F = convert_values<float>(A);
  CHECK(F.row_ptr == A.row_ptr && F.col_idx == A.col_idx);
  CHECK(F.vals[4] == 0.1f);
}

TEST(validate_catches_bad_csr) {
  auto A = read("general_3x3.mtx");
  auto B = A;
  B.col_idx[1] = 0;  // row 0 columns become {0, 0}
  CHECK_THROWS(B.validate());
  B = A;
  B.row_ptr[3] = 4;  // does not end at nnz
  CHECK_THROWS(B.validate());
  B = A;
  B.col_idx[0] = 3;  // column out of range
  CHECK_THROWS(B.validate());
}

RASP_TEST_MAIN
