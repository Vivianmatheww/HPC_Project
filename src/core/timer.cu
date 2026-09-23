#include <algorithm>
#include <cmath>

#include "rasp/csr.hpp"
#include "rasp/timer.hpp"

namespace rasp {
namespace {

DeviceBuffer<char>& l2_scratch() {
  static DeviceBuffer<char> buf;
  if (buf.size() == 0) {
    int dev = 0, l2 = 0;
    RASP_CUDA(cudaGetDevice(&dev));
    RASP_CUDA(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
    buf.resize(size_t(std::max(l2, 1 << 20)) * 2);
  }
  return buf;
}

struct Events {
  cudaEvent_t start = nullptr, stop = nullptr;
  Events() {
    RASP_CUDA(cudaEventCreate(&start));
    RASP_CUDA(cudaEventCreate(&stop));
  }
  ~Events() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
};

}  // namespace

void flush_l2(cudaStream_t stream) {
  static int value = 0;
  auto& buf = l2_scratch();
  RASP_CUDA(cudaMemsetAsync(buf.data(), (value++) & 0xff, buf.size(), stream));
}

double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  const double pos = std::clamp(p, 0.0, 100.0) / 100.0 * double(v.size() - 1);
  const size_t lo = size_t(std::floor(pos));
  const size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (pos - double(lo)) * (v[hi] - v[lo]);
}

TimingStats time_gpu(const std::function<void(cudaStream_t)>& fn, cudaStream_t stream, const BenchOptions& opt) {
  for (int i = 0; i < opt.warmup; ++i) fn(stream);
  RASP_CUDA(cudaStreamSynchronize(stream));
  RASP_CUDA(cudaGetLastError());

  Events ev;
  std::vector<double> times;
  times.reserve(size_t(std::max(opt.reps, 0)));
  for (int r = 0; r < opt.reps; ++r) {
    if (opt.flush_l2) flush_l2(stream);
    RASP_CUDA(cudaEventRecord(ev.start, stream));
    fn(stream);
    RASP_CUDA(cudaEventRecord(ev.stop, stream));
    RASP_CUDA(cudaEventSynchronize(ev.stop));
    RASP_CUDA(cudaGetLastError());
    float ms = 0;
    RASP_CUDA(cudaEventElapsedTime(&ms, ev.start, ev.stop));
    times.push_back(ms);
  }

  TimingStats s;
  s.reps = int(times.size());
  if (times.empty()) return s;
  s.median_ms = percentile(times, 50);
  s.p25_ms = percentile(times, 25);
  s.p75_ms = percentile(times, 75);
  s.min_ms = *std::min_element(times.begin(), times.end());
  s.max_ms = *std::max_element(times.begin(), times.end());
  return s;
}

}  // namespace rasp
