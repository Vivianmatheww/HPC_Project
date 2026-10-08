// Memory bandwidth of this CPU for a streaming read (the access pattern of the CG vector kernels and of CSR SpMV's
// values and indices): one thread, then all threads (OpenMP). Supports the platform choice: memory-bound kernels
// scale with bandwidth, not with cores.
//   g++ -O2 -march=native -fopenmp bandwidth.cpp -o bandwidth && ./bandwidth
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

int main() {
  const size_t n = size_t(1) << 26;  // 64 Mi doubles = 512 MiB per array, far beyond the caches
  std::vector<double> a(n, 1.0), b(n, 2.0);
  auto run = [&](int threads) {
    double best = 0, sink = 0;
    for (int rep = 0; rep < 7; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      double s = 0;
#pragma omp parallel for num_threads(threads) reduction(+ : s) schedule(static)
      for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
      const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      best = std::max(best, 2.0 * 8.0 * double(n) / sec / 1e9);
      sink += s;
    }
    std::printf("dot product, %2d thread(s): %6.1f GB/s (best of 7)%s\n", threads, best, sink < 0 ? "!" : "");
  };
  run(1);
  run(omp_get_max_threads());
}
