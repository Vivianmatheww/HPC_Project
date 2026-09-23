#include <algorithm>
#include <limits>
#include <numeric>

#include "rasp/formats.hpp"

namespace rasp {

template <class T>
SellHost<T> build_sell(const HostCsr<T>& A, int C, int sigma, double max_fill) {
  if (C <= 0 || sigma < 0) throw std::invalid_argument("build_sell: C must be > 0 and sigma >= 0");
  SellHost<T> S;
  S.rows = A.rows;
  S.cols = A.cols;
  S.C = C;
  S.sigma = sigma;
  S.slices = (A.rows + C - 1) / C;

  auto len = [&](int32_t r) { return A.row_ptr[size_t(r) + 1] - A.row_ptr[size_t(r)]; };
  S.perm.resize(size_t(A.rows));
  std::iota(S.perm.begin(), S.perm.end(), 0);
  const int64_t window = sigma == 0 ? std::max<int64_t>(A.rows, 1) : sigma;
  if (window > 1)
    for (int64_t w = 0; w < A.rows; w += window) {
      auto first = S.perm.begin() + w, last = S.perm.begin() + std::min<int64_t>(w + window, A.rows);
      std::stable_sort(first, last, [&](int32_t a, int32_t b) { return len(a) > len(b); });
    }

  S.slice_ptr.assign(size_t(S.slices) + 1, 0);
  S.slice_width.assign(size_t(S.slices), 0);
  int64_t stored = 0;
  for (int32_t s = 0; s < S.slices; ++s) {
    int32_t w = 0;
    for (int32_t i = s * C; i < std::min(A.rows, (s + 1) * C); ++i) w = std::max(w, len(S.perm[size_t(i)]));
    S.slice_width[size_t(s)] = w;
    stored += int64_t(w) * C;
    if (stored >= std::numeric_limits<int32_t>::max()) throw FormatInfeasible("SELL exceeds 32-bit offsets");
    S.slice_ptr[size_t(s) + 1] = int32_t(stored);
  }
  if (double(stored) > max_fill * double(std::max<int64_t>(A.nnz, 1)) || double(stored) * double(sizeof(T) + 4) > kMaxFormatBytes)
    throw FormatInfeasible("SELL would store " + std::to_string(stored) + " slots for " + std::to_string(A.nnz) +
                           " nonzeros");

  S.col.assign(size_t(stored), -1);
  S.val.assign(size_t(stored), T(0));
  for (int32_t i = 0; i < A.rows; ++i) {
    const int32_t r = S.perm[size_t(i)], s = i / C, lane = i % C;
    const size_t base = size_t(S.slice_ptr[size_t(s)]) + size_t(lane);
    for (int32_t k = A.row_ptr[size_t(r)], j = 0; k < A.row_ptr[size_t(r) + 1]; ++k, ++j) {
      S.col[base + size_t(j) * size_t(C)] = A.col_idx[size_t(k)];
      S.val[base + size_t(j) * size_t(C)] = A.vals[size_t(k)];
    }
  }
  return S;
}

template SellHost<float> build_sell(const HostCsr<float>&, int, int, double);
template SellHost<double> build_sell(const HostCsr<double>&, int, int, double);

}  // namespace rasp
