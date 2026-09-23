// Adaptive Row-grouped CSR conversion, following Section 3 of Heller & Oberhuber (arXiv:1203.5737).
#include <algorithm>
#include <limits>
#include <queue>

#include "rasp/formats.hpp"

namespace rasp {

template <class T>
ArgCsrHost<T> build_argcsr(const HostCsr<T>& A, int block, int desired_chunk, double max_fill) {
  if (block <= 0 || desired_chunk <= 0) throw std::invalid_argument("build_argcsr: block and chunk must be > 0");
  ArgCsrHost<T> G;
  G.rows = A.rows;
  G.cols = A.cols;
  G.block = block;
  G.desired_chunk = desired_chunk;
  G.threads_end.assign(size_t(A.rows), 0);
  auto len = [&](int32_t r) { return A.row_ptr[size_t(r) + 1] - A.row_ptr[size_t(r)]; };

  // 1. Group formation. The paper closes a group once its nonzeros exceed desiredChunkSize * blockDim or it
  //    would get more than blockDim rows. We close *before* adding the row that would exceed the budget, so a
  //    group never exceeds it unless it holds a single long row.
  const int64_t budget = int64_t(desired_chunk) * block;
  for (int32_t r = 0; r < A.rows;) {
    ArgGroup g{r, 0, 0, 0};
    int64_t nnz = 0;
    while (r < A.rows && g.size < block && (g.size == 0 || nnz + len(r) <= budget)) {
      nnz += len(r);
      ++g.size;
      ++r;
    }
    G.groups.push_back(g);
  }

  // 2. Threads per row and the group's chunk size.
  int64_t stored = 0;
  std::vector<int32_t> threads;
  for (auto& g : G.groups) {
    threads.assign(size_t(g.size), 1);
    auto fill = [&](int i) {
      const int32_t l = len(g.first_row + i);
      return (l + threads[size_t(i)] - 1) / threads[size_t(i)];
    };
    // Max-heap on (chunk filling, row length). Give a spare thread to the fullest chunk until that stops
    // shrinking it: the paper's Figure 3 example leaves one thread free for exactly this reason.
    auto cmp = [&](int a, int b) {
      const int fa = fill(a), fb = fill(b);
      return fa != fb ? fa < fb : len(g.first_row + a) < len(g.first_row + b);
    };
    std::priority_queue<int, std::vector<int>, decltype(cmp)> heap(cmp);
    for (int i = 0; i < g.size; ++i) heap.push(i);
    for (int spare = block - g.size; spare > 0 && !heap.empty(); --spare) {
      const int i = heap.top();
      const int32_t l = len(g.first_row + i), f = fill(i);
      if ((l + threads[size_t(i)]) / (threads[size_t(i)] + 1) >= f) break;  // ceil(l/(t+1)) would not drop
      heap.pop();
      ++threads[size_t(i)];
      heap.push(i);
    }
    g.chunk = 0;
    int32_t prefix = 0;
    for (int i = 0; i < g.size; ++i) {
      g.chunk = std::max(g.chunk, fill(i));
      prefix += threads[size_t(i)];
      G.threads_end[size_t(g.first_row + i)] = prefix;
    }
    g.offset = int32_t(stored);
    stored += int64_t(g.chunk) * block;
    if (stored >= std::numeric_limits<int32_t>::max()) throw FormatInfeasible("ArgCSR exceeds 32-bit offsets");
  }
  if (double(stored) > max_fill * double(std::max<int64_t>(A.nnz, 1)) || double(stored) * double(sizeof(T) + 4) > kMaxFormatBytes)
    throw FormatInfeasible("ArgCSR would store " + std::to_string(stored) + " slots for " + std::to_string(A.nnz) +
                           " nonzeros");

  // 3. Fill chunks: thread t of a row gets that row's elements [q*f, (q+1)*f), f = ceil(len/threads).
  G.col.assign(size_t(stored), -1);
  G.val.assign(size_t(stored), T(0));
  for (auto& g : G.groups)
    for (int i = 0; i < g.size; ++i) {
      const int32_t r = g.first_row + i;
      const int32_t t_begin = i == 0 ? 0 : G.threads_end[size_t(r) - 1];
      const int32_t t_count = G.threads_end[size_t(r)] - t_begin;
      const int32_t l = len(r), f = (l + t_count - 1) / t_count;
      for (int32_t e = 0; e < l; ++e) {
        const int32_t t = t_begin + e / f, slot_in_chunk = e % f;
        const size_t pos = size_t(g.offset) + size_t(slot_in_chunk) * size_t(block) + size_t(t);
        G.col[pos] = A.col_idx[size_t(A.row_ptr[size_t(r)] + e)];
        G.val[pos] = A.vals[size_t(A.row_ptr[size_t(r)] + e)];
      }
    }
  return G;
}

template ArgCsrHost<float> build_argcsr(const HostCsr<float>&, int, int, double);
template ArgCsrHost<double> build_argcsr(const HostCsr<double>&, int, int, double);

}  // namespace rasp
