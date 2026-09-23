// Adaptive Row-grouped CSR conversion, following Section 3 of Heller & Oberhuber (arXiv:1203.5737).
// The layout (groups, threads per row, chunk sizes) depends only on row lengths, so it is computed from
// row_ptr alone and shared by the host converter and the GPU converter (convert_gpu.cu).
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

#include "rasp/formats.hpp"

namespace rasp {

namespace {

// budget_for(r) = desiredChunkSize * block for a group starting at row r.
template <class Budget>
ArgCsrLayout layout_impl(const std::vector<int32_t>& row_ptr, int block, const Budget& budget_for, size_t value_bytes,
                         double max_fill) {
  const int32_t rows = int32_t(row_ptr.size()) - 1;
  const int64_t nnz = row_ptr.back();
  ArgCsrLayout L;
  L.threads_end.assign(size_t(rows), 0);
  L.row_offset.assign(size_t(rows), 0);
  L.row_tbegin.assign(size_t(rows), 0);
  L.row_fill.assign(size_t(rows), 0);
  auto len = [&](int32_t r) { return row_ptr[size_t(r) + 1] - row_ptr[size_t(r)]; };

  // 1. Group formation. The paper closes a group once its nonzeros exceed desiredChunkSize * blockDim or it
  //    would get more than blockDim rows. We close *before* adding the row that would exceed the budget, so a
  //    group never exceeds it unless it holds a single long row.
  for (int32_t r = 0; r < rows;) {
    ArgGroup g{r, 0, 0, 0};
    const int64_t budget = budget_for(r);
    int64_t group_nnz = 0;
    while (r < rows && g.size < block && (g.size == 0 || group_nnz + len(r) <= budget)) {
      group_nnz += len(r);
      ++g.size;
      ++r;
    }
    L.groups.push_back(g);
  }

  // 2. Threads per row and the group's chunk size.
  std::vector<int32_t> threads;
  for (auto& g : L.groups) {
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
    g.offset = int32_t(L.stored);
    int32_t prefix = 0;
    for (int i = 0; i < g.size; ++i) {
      const int32_t r = g.first_row + i;
      g.chunk = std::max(g.chunk, fill(i));
      L.row_tbegin[size_t(r)] = prefix;
      L.row_fill[size_t(r)] = fill(i);
      L.row_offset[size_t(r)] = g.offset;
      prefix += threads[size_t(i)];
      L.threads_end[size_t(r)] = prefix;
    }
    L.stored += int64_t(g.chunk) * block;
    if (L.stored >= std::numeric_limits<int32_t>::max()) throw FormatInfeasible("ArgCSR exceeds 32-bit offsets");
  }
  if (double(L.stored) > max_fill * double(std::max<int64_t>(nnz, 1)) ||
      double(L.stored) * double(value_bytes + 4) > kMaxFormatBytes)
    throw FormatInfeasible("ArgCSR would store " + std::to_string(L.stored) + " slots for " + std::to_string(nnz) +
                           " nonzeros");
  return L;
}

// 3. Fill chunks: thread t of a row gets that row's elements [q*f, (q+1)*f), f = ceil(len/threads).
template <class T>
ArgCsrHost<T> fill_host(const HostCsr<T>& A, int block, int desired_chunk, ArgCsrLayout&& L) {
  ArgCsrHost<T> G;
  G.rows = A.rows;
  G.cols = A.cols;
  G.block = block;
  G.desired_chunk = desired_chunk;
  G.col.assign(size_t(L.stored), -1);
  G.val.assign(size_t(L.stored), T(0));
  for (int32_t r = 0; r < A.rows; ++r) {
    const int32_t l = A.row_ptr[size_t(r) + 1] - A.row_ptr[size_t(r)], f = L.row_fill[size_t(r)];
    for (int32_t e = 0; e < l; ++e) {
      const size_t pos = size_t(L.row_offset[size_t(r)]) + size_t(e % f) * size_t(block) +
                         size_t(L.row_tbegin[size_t(r)] + e / f);
      G.col[pos] = A.col_idx[size_t(A.row_ptr[size_t(r)] + e)];
      G.val[pos] = A.vals[size_t(A.row_ptr[size_t(r)] + e)];
    }
  }
  G.groups = std::move(L.groups);
  G.threads_end = std::move(L.threads_end);
  return G;
}

}  // namespace

ArgCsrLayout argcsr_layout(const std::vector<int32_t>& row_ptr, int block, int desired_chunk, size_t value_bytes,
                           double max_fill) {
  if (block <= 0 || desired_chunk <= 0) throw std::invalid_argument("argcsr: block and chunk must be > 0");
  const int64_t budget = int64_t(desired_chunk) * block;
  return layout_impl(row_ptr, block, [&](int32_t) { return budget; }, value_bytes, max_fill);
}

int argcsr_local_chunk(double mean_len, double cv, int block, int target_rows) {
  const double ideal = double(target_rows) * mean_len / double(block);
  // Nearest power of two in [1, 32]; clamp the exponent before shifting (empty regions give log2 -> -inf).
  const int e = ideal > 0 ? std::clamp(int(std::lround(std::log2(ideal))), 0, 5) : 0;
  int d = 1 << e;
  if (cv > 1.0) d = std::max(1, d / 2);  // very irregular region: smaller groups, less padding
  return d;
}

ArgCsrLayout argcsr_local_layout(const std::vector<int32_t>& row_ptr, int block, int target_rows, int window,
                                 size_t value_bytes, double max_fill) {
  if (block <= 0 || target_rows <= 0 || window <= 0) throw std::invalid_argument("argcsr_local: bad arguments");
  // One desiredChunkSize per window of rows, from that window's own row-length mean and CV.
  const int32_t rows = int32_t(row_ptr.size()) - 1;
  const int32_t windows = (rows + window - 1) / window;
  std::vector<int64_t> budget(size_t(std::max(windows, 1)), block);
  for (int32_t w = 0; w < windows; ++w) {
    const int32_t r0 = w * window, r1 = std::min(rows, r0 + window);
    double sum = 0, sq = 0;
    for (int32_t r = r0; r < r1; ++r) {
      const double l = row_ptr[size_t(r) + 1] - row_ptr[size_t(r)];
      sum += l;
      sq += l * l;
    }
    const double n = r1 - r0, mean = sum / n, var = std::max(0.0, sq / n - mean * mean);
    const double cv = mean > 0 ? std::sqrt(var) / mean : 0.0;
    budget[size_t(w)] = int64_t(argcsr_local_chunk(mean, cv, block, target_rows)) * block;
  }
  return layout_impl(row_ptr, block, [&](int32_t r) { return budget[size_t(r / window)]; }, value_bytes, max_fill);
}

template <class T>
ArgCsrHost<T> build_argcsr(const HostCsr<T>& A, int block, int desired_chunk, double max_fill) {
  return fill_host(A, block, desired_chunk, argcsr_layout(A.row_ptr, block, desired_chunk, sizeof(T), max_fill));
}

template <class T>
ArgCsrHost<T> build_argcsr_local(const HostCsr<T>& A, int block, int target_rows, int window, double max_fill) {
  return fill_host(A, block, 0, argcsr_local_layout(A.row_ptr, block, target_rows, window, sizeof(T), max_fill));
}

template ArgCsrHost<float> build_argcsr(const HostCsr<float>&, int, int, double);
template ArgCsrHost<double> build_argcsr(const HostCsr<double>&, int, int, double);
template ArgCsrHost<float> build_argcsr_local(const HostCsr<float>&, int, int, int, double);
template ArgCsrHost<double> build_argcsr_local(const HostCsr<double>&, int, int, int, double);

}  // namespace rasp
