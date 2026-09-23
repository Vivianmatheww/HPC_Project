#pragma once
// GPU timing harness. Every benchmark in RASP times through time_gpu() so methodology is uniform:
// warm-up runs, then `reps` timed runs with CUDA events, each preceded (outside the timed region) by
// an L2 flush unless disabled. Report the median and IQR, never the mean.
#include <functional>
#include <vector>

#include <cuda_runtime.h>

namespace rasp {

struct TimingStats {
  double median_ms = 0, p25_ms = 0, p75_ms = 0, min_ms = 0, max_ms = 0;
  int reps = 0;
};

struct BenchOptions {
  int warmup = 5;
  int reps = 20;
  bool flush_l2 = true;  // RTX 4050 L2 = 24 MB: small matrices would otherwise run out of cache
};

TimingStats time_gpu(const std::function<void(cudaStream_t)>& fn, cudaStream_t stream, const BenchOptions& opt);

// Evicts L2 by writing a buffer twice the L2 size. Enqueued on `stream`; not timed by time_gpu().
void flush_l2(cudaStream_t stream);

// Percentile of an unsorted sample (linear interpolation), p in [0, 100].
double percentile(std::vector<double> v, double p);

}  // namespace rasp
