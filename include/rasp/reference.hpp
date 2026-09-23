#pragma once
// CPU reference implementations used to check GPU results.
#include <vector>

#include "rasp/csr.hpp"

namespace rasp {

// y = A x, accumulated in double regardless of T.
template <class T>
std::vector<double> cpu_spmv(const HostCsr<T>& A, const std::vector<T>& x);

// ||y - ref||_2 / ||ref||_2 (absolute norm of y if ref is all zeros).
template <class T>
double rel_l2_error(const std::vector<T>& y, const std::vector<double>& ref);

// C = A B with a dense row accumulator. Intended for test-sized matrices.
template <class T>
HostCsr<T> cpu_spgemm(const HostCsr<T>& A, const HostCsr<T>& B);

// Max relative difference between two CSR matrices with identical structure; +inf if structures differ.
template <class T>
double csr_max_rel_diff(const HostCsr<T>& A, const HostCsr<T>& B);

// Vector of n uniform values in [-1, 1] with a fixed seed.
template <class T>
std::vector<T> random_vector(size_t n, unsigned seed = 42);

}  // namespace rasp
