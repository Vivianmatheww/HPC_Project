#pragma once
// Generated matrices for the SpGEMM experiments: paper 1's multigrid workload (Laplace3D with 2x2x2 aggregation).
// Built in memory with exactly the layout of scripts/gen_laplace.py (cell (x, y, z) is row x*n*n + y*n + z; tested
// against its output), so multi-GB problems need no Matrix Market files.
#include "rasp/csr.hpp"

namespace rasp {

// 7-point stencil on an n x n x n grid with Dirichlet boundaries: diagonal 6, neighbours -1. n^3 x n^3, ~7 nnz/row.
HostCsr<double> laplace3d(int n);

// Piecewise-constant aggregation of 2x2x2 cells: n^3 x (n/2)^3, one 1.0 per row (column = the cell's aggregate,
// numbered x/2 * m*m + y/2 * m + z/2 with m = n/2, i.e. the coarse grid in the same layout). n must be even.
HostCsr<double> aggregation(int n);

// Transpose with sorted columns (R = P^T).
template <class T>
HostCsr<T> transpose(const HostCsr<T>& A);

}  // namespace rasp
