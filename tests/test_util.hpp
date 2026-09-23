#pragma once
// Minimal test framework: TEST(name) { ... CHECK(...); } and a main() that runs every test.
// Each test file is its own executable registered with CTest.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "rasp/csr.hpp"

namespace rasp_test {

struct Registry {
  std::vector<std::pair<std::string, std::function<void()>>> tests;
  int failures = 0;
  static Registry& get() {
    static Registry r;
    return r;
  }
};

struct Register {
  Register(const char* name, std::function<void()> fn) { Registry::get().tests.emplace_back(name, std::move(fn)); }
};

inline void fail(const char* file, int line, const std::string& msg) {
  ++Registry::get().failures;
  std::printf("    FAIL %s:%d: %s\n", file, line, msg.c_str());
}

inline std::string data_path(const std::string& name) { return std::string(RASP_TEST_DATA_DIR) + "/" + name; }

// Random CSR with sorted unique columns; row_len(r) gives the requested length of row r.
template <class T>
rasp::HostCsr<T> random_csr(int32_t rows, int32_t cols, const std::function<int(int32_t)>& row_len, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> val(-1, 1);
  rasp::HostCsr<T> A;
  A.rows = rows;
  A.cols = cols;
  A.row_ptr.push_back(0);
  std::vector<int32_t> all(static_cast<size_t>(cols));
  for (int32_t c = 0; c < cols; ++c) all[size_t(c)] = c;
  for (int32_t r = 0; r < rows; ++r) {
    const int len = std::min(row_len(r), int(cols));
    std::vector<int32_t> pick;
    std::sample(all.begin(), all.end(), std::back_inserter(pick), len, rng);  // sorted, unique
    for (int32_t c : pick) {
      A.col_idx.push_back(c);
      A.vals.push_back(T(val(rng)));
    }
    A.row_ptr.push_back(int32_t(A.col_idx.size()));
  }
  A.nnz = int64_t(A.col_idx.size());
  return A;
}

}  // namespace rasp_test

#define TEST(name)                                                              \
  static void name();                                                           \
  static rasp_test::Register name##_registered(#name, name);                    \
  static void name()

#define CHECK(cond)                                                             \
  do {                                                                          \
    if (!(cond)) rasp_test::fail(__FILE__, __LINE__, "CHECK(" #cond ")");       \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                   \
  do {                                                                          \
    const double va_ = double(a), vb_ = double(b);                              \
    if (!(std::abs(va_ - vb_) <= (tol)))                                        \
      rasp_test::fail(__FILE__, __LINE__,                                       \
                      std::string(#a " = ") + std::to_string(va_) + ", " #b " = " + std::to_string(vb_)); \
  } while (0)

#define CHECK_THROWS(expr)                                                      \
  do {                                                                          \
    bool threw_ = false;                                                        \
    try {                                                                       \
      (void)(expr);                                                             \
    } catch (...) {                                                             \
      threw_ = true;                                                            \
    }                                                                           \
    if (!threw_) rasp_test::fail(__FILE__, __LINE__, "expected exception: " #expr); \
  } while (0)

#define RASP_TEST_MAIN                                                          \
  int main() {                                                                  \
    auto& reg = rasp_test::Registry::get();                                     \
    for (auto& [name, fn] : reg.tests) {                                        \
      const int before = reg.failures;                                          \
      try {                                                                     \
        fn();                                                                   \
      } catch (const std::exception& e) {                                       \
        rasp_test::fail(__FILE__, __LINE__, std::string("uncaught: ") + e.what()); \
      }                                                                         \
      std::printf("%s %s\n", reg.failures == before ? "[ ok ]" : "[FAIL]", name.c_str()); \
    }                                                                           \
    std::printf("%zu tests, %d failed checks\n", reg.tests.size(), reg.failures); \
    return reg.failures == 0 ? 0 : 1;                                           \
  }
