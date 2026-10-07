#pragma once
// Device building blocks of the out-of-core SpGEMM executor (exposed so they can be tested on their own). Every
// function only enqueues work on `stream`, except csr_add_symbolic, which must read nnz back.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace rasp {

// y[i] = x[i] - base for i < n: the row pointers of a row block, rebased to start at 0 (cuSPARSE needs that). x may be
// mapped host memory (zero-copy); x == y is allowed.
void rebase_row_ptr(const int32_t* x, int32_t* y, int64_t n, int32_t base, cudaStream_t stream);

// y[i] = x[i] + offset for i < n: moves a block's row pointers to its place in C. y may be mapped host memory;
// x == y is allowed.
void offset_row_ptr(const int32_t* x, int32_t* y, int64_t n, int32_t offset, cudaStream_t stream);

// Row pointers of B rows [k0, k1) as a full-height matrix of `rows` rows (the rows outside the block are empty).
// blk holds the block's own k1 - k0 + 1 row pointers (any base). out[i] is 0 for i <= k0, blk[i - k0] - blk[0] inside
// the block and nnz(block) for i >= k1 (rows + 1 entries). Multiplying A by it skips A's columns outside [k0, k1)
// without splitting A, as paper 1's chunked kernels do.
void expand_row_ptr(const int32_t* blk, int32_t k0, int32_t k1, int32_t rows, int32_t* out, cudaStream_t stream);

// C = X + Y for two CSR matrices with the same rows and sorted columns: the structural union, values added.
// Symbolic: writes C's row pointers (rows + 1) and returns nnz(C); synchronises `stream`. `temp` must hold
// csr_add_temp_bytes(rows).
size_t csr_add_temp_bytes(int32_t rows);
int64_t csr_add_symbolic(int32_t rows, const int32_t* xp, const int32_t* xc, const int32_t* yp, const int32_t* yc,
                         int32_t* cp, void* temp, size_t temp_bytes, cudaStream_t stream);
template <class T>
void csr_add_numeric(int32_t rows, const int32_t* xp, const int32_t* xc, const T* xv, const int32_t* yp,
                     const int32_t* yc, const T* yv, const int32_t* cp, int32_t* cc, T* cv, cudaStream_t stream);

}  // namespace rasp
