// UVM / memory-placement probe for the RASP platform study (Phase 0, plan.md §4).
//
// Questions answered:
//   1. Real transfer and device bandwidths (inputs to the SpGEMM cost model).
//   2. When data fits in VRAM: how fast can kernels read device memory vs managed memory (UVM)
//      vs pinned zero-copy host memory, for sequential streams and for random gathers
//      (random gathers mimic SpGEMM's accesses to B; see paper 1, Table 3)?
//   3. Do cudaMemPrefetchAsync / cudaMemAdvise work under WSL2?
//   4. Can managed memory be oversubscribed (bigger than VRAM) under WSL2?
//
// Output: human-readable table on stdout + CSV (default results/platform/uvm_probe.csv).
// Usage:  uvm_probe [--csv path] [--no-oversub]
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                                           \
  do {                                                                                  \
    cudaError_t e_ = (x);                                                               \
    if (e_ != cudaSuccess) {                                                            \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(1);                                                                          \
    }                                                                                   \
  } while (0)

constexpr size_t MiB = size_t(1) << 20;
constexpr size_t GiB = size_t(1) << 30;

// ---------------------------------------------------------------- kernels
__global__ void write_seq(double* a, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    a[i] = double(i & 1023);
}

__device__ inline void reduce_out(double s, double* out) {
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

__global__ void read_seq(const double* __restrict__ a, size_t n, double* out) {
  double s = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    s += a[i];
  reduce_out(s, out);
}

// idx always lives in device memory, so only the placement of `a` is measured.
__global__ void gather(const double* __restrict__ a, const unsigned* __restrict__ idx, size_t m, double* out) {
  double s = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < m; i += size_t(gridDim.x) * blockDim.x)
    s += a[idx[i]];
  reduce_out(s, out);
}

__global__ void copy_kernel(const double* __restrict__ a, double* __restrict__ b, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    b[i] = a[i];
}

// ---------------------------------------------------------------- helpers
struct Row {
  std::string test, location;
  size_t bytes;
  std::string metric;
  double value;
  std::string unit, note;
};
static std::vector<Row> g_rows;

static void record(const std::string& test, const std::string& loc, size_t bytes, const std::string& metric,
                   double value, const std::string& unit, const std::string& note = "") {
  g_rows.push_back({test, loc, bytes, metric, value, unit, note});
  printf("  %-12s %-14s %8.0f MiB  %-22s %12.3f %-8s %s\n", test.c_str(), loc.c_str(), bytes / double(MiB),
         metric.c_str(), value, unit.c_str(), note.c_str());
}

static int g_grid = 0;
constexpr int kBlock = 256;
static double* g_out = nullptr;
static cudaEvent_t g_e0, g_e1;

template <class F>
static float time_once(F&& launch) {
  CK(cudaEventRecord(g_e0));
  launch();
  CK(cudaEventRecord(g_e1));
  CK(cudaEventSynchronize(g_e1));
  CK(cudaGetLastError());
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, g_e0, g_e1));
  return ms;
}

template <class F>
static float time_median(F&& launch, int reps) {
  std::vector<float> t;
  for (int r = 0; r < reps; ++r) t.push_back(time_once(launch));
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

static double gbps(size_t bytes, float ms) { return bytes / (ms * 1e-3) / 1e9; }

static double host_ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static void host_fill(double* a, size_t n) {
  for (size_t i = 0; i < n; ++i) a[i] = double(i & 1023);
}

// Touch one element per 4 KiB page so every page is brought back to the host.
static double host_touch_pages(const double* a, size_t n) {
  double s = 0;
  for (size_t i = 0; i < n; i += 4096 / sizeof(double)) s += a[i];
  return s;
}

// ---------------------------------------------------------------- tests
static void test_transfers() {
  printf("\n[1] Transfer and device bandwidth (median of 10)\n");
  const size_t bytes = 256 * MiB;
  void *d, *h_pin, *h_page;
  CK(cudaMalloc(&d, bytes));
  CK(cudaMallocHost(&h_pin, bytes));
  h_page = malloc(bytes);
  memset(h_pin, 1, bytes);
  memset(h_page, 1, bytes);

  record("transfer", "pinned", bytes, "H2D", gbps(bytes, time_median([&] { CK(cudaMemcpy(d, h_pin, bytes, cudaMemcpyHostToDevice)); }, 10)), "GB/s");
  record("transfer", "pinned", bytes, "D2H", gbps(bytes, time_median([&] { CK(cudaMemcpy(h_pin, d, bytes, cudaMemcpyDeviceToHost)); }, 10)), "GB/s");
  record("transfer", "pageable", bytes, "H2D", gbps(bytes, time_median([&] { CK(cudaMemcpy(d, h_page, bytes, cudaMemcpyHostToDevice)); }, 10)), "GB/s");
  record("transfer", "pageable", bytes, "D2H", gbps(bytes, time_median([&] { CK(cudaMemcpy(h_page, d, bytes, cudaMemcpyDeviceToHost)); }, 10)), "GB/s");

  // Simultaneous H2D + D2H on two streams: with 1 copy engine this is ~ the sum of the two times.
  void *d2, *h_pin2;
  CK(cudaMalloc(&d2, bytes));
  CK(cudaMallocHost(&h_pin2, bytes));
  cudaStream_t s1, s2;
  CK(cudaStreamCreate(&s1));
  CK(cudaStreamCreate(&s2));
  float bidir = time_median([&] {
    CK(cudaMemcpyAsync(d, h_pin, bytes, cudaMemcpyHostToDevice, s1));
    CK(cudaMemcpyAsync(h_pin2, d2, bytes, cudaMemcpyDeviceToHost, s2));
    CK(cudaStreamSynchronize(s1));
    CK(cudaStreamSynchronize(s2));
  }, 10);
  record("transfer", "pinned", 2 * bytes, "H2D+D2H concurrent", gbps(2 * bytes, bidir), "GB/s", "aggregate");
  CK(cudaStreamDestroy(s1));
  CK(cudaStreamDestroy(s2));
  CK(cudaFree(d2));
  CK(cudaFreeHost(h_pin2));

  // Device copy bandwidth (STREAM copy: read + write).
  const size_t cb = 1 * GiB, n = cb / sizeof(double);
  double *a, *b;
  CK(cudaMalloc(&a, cb));
  CK(cudaMalloc(&b, cb));
  write_seq<<<g_grid, kBlock>>>(a, n);
  float ms = time_median([&] { copy_kernel<<<g_grid, kBlock>>>(a, b, n); }, 10);
  record("device", "device", cb, "copy (r+w)", gbps(2 * cb, ms), "GB/s");
  CK(cudaFree(a));
  CK(cudaFree(b));

  CK(cudaFree(d));
  CK(cudaFreeHost(h_pin));
  free(h_page);
}

// Random indices in device memory, shared by all gather tests of a given size.
static unsigned* make_indices(size_t n, size_t m) {
  std::vector<unsigned> h(m);
  std::mt19937_64 rng(42);
  std::uniform_int_distribution<size_t> dist(0, n - 1);
  for (auto& v : h) v = unsigned(dist(rng));
  unsigned* d;
  CK(cudaMalloc(&d, m * sizeof(unsigned)));
  CK(cudaMemcpy(d, h.data(), m * sizeof(unsigned), cudaMemcpyHostToDevice));
  return d;
}

static void test_placement(size_t bytes) {
  const size_t n = bytes / sizeof(double);
  const size_t m = std::min<size_t>(n, 16 * MiB);  // number of random gathers
  printf("\n[2] Placement study, array = %zu MiB, %zu random gathers\n", bytes / MiB, m);
  unsigned* idx = make_indices(n, m);
  cudaError_t err;

  // --- device memory (reference)
  {
    double* a;
    err = cudaMalloc(&a, bytes);
    if (err != cudaSuccess) {
      record("placement", "device", bytes, "alloc", 0, "", cudaGetErrorString(err));
    } else {
      write_seq<<<g_grid, kBlock>>>(a, n);
      record("placement", "device", bytes, "seq read", gbps(bytes, time_median([&] { read_seq<<<g_grid, kBlock>>>(a, n, g_out); }, 10)), "GB/s");
      float ms = time_median([&] { gather<<<g_grid, kBlock>>>(a, idx, m, g_out); }, 10);
      record("placement", "device", bytes, "random gather", m / (ms * 1e-3) / 1e6, "Mgath/s");
      CK(cudaFree(a));
    }
  }

  // --- managed memory (UVM)
  {
    double* a;
    err = cudaMallocManaged(&a, bytes);
    if (err != cudaSuccess) {
      record("placement", "managed", bytes, "alloc", 0, "", cudaGetErrorString(err));
    } else {
      auto t0 = std::chrono::steady_clock::now();
      host_fill(a, n);
      record("placement", "managed", bytes, "host first-touch", host_ms_since(t0), "ms");

      float cold = time_once([&] { read_seq<<<g_grid, kBlock>>>(a, n, g_out); });
      record("placement", "managed", bytes, "seq read (cold)", cold, "ms", "includes host->device migration");
      record("placement", "managed", bytes, "seq read (warm)", gbps(bytes, time_median([&] { read_seq<<<g_grid, kBlock>>>(a, n, g_out); }, 10)), "GB/s");
      float ms = time_median([&] { gather<<<g_grid, kBlock>>>(a, idx, m, g_out); }, 10);
      record("placement", "managed", bytes, "random gather (warm)", m / (ms * 1e-3) / 1e6, "Mgath/s");

      t0 = std::chrono::steady_clock::now();
      volatile double sink = host_touch_pages(a, n);
      (void)sink;
      record("placement", "managed", bytes, "host touch after GPU", host_ms_since(t0), "ms", "device->host migration");

      // After the host touch, does the next kernel pay the migration again?
      float recold = time_once([&] { read_seq<<<g_grid, kBlock>>>(a, n, g_out); });
      record("placement", "managed", bytes, "seq read (re-cold)", recold, "ms", "after host touch");

      int dev = 0;
      cudaMemLocation loc{};
      loc.type = cudaMemLocationTypeDevice;
      loc.id = dev;
      err = cudaMemPrefetchAsync(a, bytes, loc, 0, 0);
      cudaGetLastError();  // clear sticky-free errors so later checks are clean
      record("placement", "managed", bytes, "cudaMemPrefetchAsync", err == cudaSuccess, "ok?", cudaGetErrorString(err));
      err = cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, loc);
      cudaGetLastError();
      record("placement", "managed", bytes, "cudaMemAdvise(PrefLoc)", err == cudaSuccess, "ok?", cudaGetErrorString(err));
      CK(cudaDeviceSynchronize());
      CK(cudaFree(a));
    }
  }

  // --- pinned host memory read directly by the GPU (zero-copy)
  {
    double* h;
    err = cudaHostAlloc(&h, bytes, cudaHostAllocMapped);
    if (err != cudaSuccess) {
      cudaGetLastError();
      record("placement", "zero-copy", bytes, "alloc", 0, "", cudaGetErrorString(err));
    } else {
      host_fill(h, n);
      double* d;
      CK(cudaHostGetDevicePointer(&d, h, 0));
      record("placement", "zero-copy", bytes, "seq read", gbps(bytes, time_median([&] { read_seq<<<g_grid, kBlock>>>(d, n, g_out); }, 3)), "GB/s");
      float ms = time_median([&] { gather<<<g_grid, kBlock>>>(d, idx, m, g_out); }, 3);
      record("placement", "zero-copy", bytes, "random gather", m / (ms * 1e-3) / 1e6, "Mgath/s");
      CK(cudaFreeHost(h));
    }
  }
  CK(cudaFree(idx));
}

static void test_oversubscription(size_t vram) {
  const size_t bytes = size_t(1.25 * double(vram));
  const size_t n = bytes / sizeof(double);
  printf("\n[3] Oversubscription: managed allocation of %.2f GiB (1.25x VRAM)\n", bytes / double(GiB));
  double* a;
  cudaError_t err = cudaMallocManaged(&a, bytes);
  record("oversub", "managed", bytes, "cudaMallocManaged", err == cudaSuccess, "ok?", cudaGetErrorString(err));
  if (err != cudaSuccess) {
    cudaGetLastError();
    return;
  }
  // Written by the GPU so we don't fill host RAM first.
  auto t0 = std::chrono::steady_clock::now();
  write_seq<<<g_grid, kBlock>>>(a, n);
  err = cudaDeviceSynchronize();
  record("oversub", "managed", bytes, "GPU write whole array", err == cudaSuccess ? host_ms_since(t0) : -1, "ms", cudaGetErrorString(err));
  if (err == cudaSuccess) {
    t0 = std::chrono::steady_clock::now();
    read_seq<<<g_grid, kBlock>>>(a, n, g_out);
    err = cudaDeviceSynchronize();
    record("oversub", "managed", bytes, "GPU read whole array", err == cudaSuccess ? gbps(bytes, float(host_ms_since(t0))) : -1, "GB/s", cudaGetErrorString(err));
  }
  cudaGetLastError();
  cudaFree(a);
}

static void write_csv(const std::string& path) {
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  FILE* f = fopen(path.c_str(), "w");
  if (!f) {
    fprintf(stderr, "cannot write %s\n", path.c_str());
    return;
  }
  fprintf(f, "test,location,bytes,metric,value,unit,note\n");
  for (auto& r : g_rows)
    fprintf(f, "%s,%s,%zu,%s,%.6f,%s,\"%s\"\n", r.test.c_str(), r.location.c_str(), r.bytes, r.metric.c_str(), r.value,
            r.unit.c_str(), r.note.c_str());
  fclose(f);
  printf("\nCSV written to %s\n", path.c_str());
}

int main(int argc, char** argv) {
  std::string csv = "results/platform/uvm_probe.csv";
  bool oversub = true;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
    else if (!strcmp(argv[i], "--no-oversub")) oversub = false;
  }

  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  int drv = 0, rt = 0, cma = 0, pma = 0, engines = 0;
  CK(cudaDriverGetVersion(&drv));
  CK(cudaRuntimeGetVersion(&rt));
  CK(cudaDeviceGetAttribute(&cma, cudaDevAttrConcurrentManagedAccess, 0));
  CK(cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess, 0));
  CK(cudaDeviceGetAttribute(&engines, cudaDevAttrAsyncEngineCount, 0));
  printf("GPU %s | %.2f GiB | driver API %d | runtime %d | concurrentManagedAccess=%d pageableMemoryAccess=%d asyncEngines=%d\n",
         p.name, p.totalGlobalMem / double(GiB), drv, rt, cma, pma, engines);

  int per_sm = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, read_seq, kBlock, 0));
  g_grid = p.multiProcessorCount * per_sm;
  CK(cudaMalloc(&g_out, sizeof(double)));
  CK(cudaEventCreate(&g_e0));
  CK(cudaEventCreate(&g_e1));

  test_transfers();
  for (size_t bytes : {256 * MiB, 1 * GiB, 4 * GiB}) test_placement(bytes);
  if (oversub) test_oversubscription(p.totalGlobalMem);

  write_csv(csv);
  return 0;
}
