# Phase 1 Guide — Infrastructure (Weeks 1–2: Sep 28 – Oct 9)

Goal: every later phase should only need to **write a kernel and call the harness**. Reading
matrices, owning GPU memory, timing, checking correctness and writing CSVs are built once, here.

**Deliverable (definition of done):** `./build/bench_spmv --matrices data/suitesparse --out results/phase1_cusparse.csv`
runs cuSPARSE SpMV (ALG1 and ALG2, FP32 and FP64) on ≥ 50 SuiteSparse matrices, checks every
result against a CPU reference, and writes one CSV row per (matrix, config). All unit tests pass
under `ctest`.

---

## Step 0 — Agree on interfaces first (day 1, all three, ~1 hour)

> **Status (Sep 23): implemented.** The real headers are in [include/rasp/](../include/rasp/) and are the
> source of truth. They differ from this sketch in one place: `CusparseSpmv` takes `x` and `y` in its
> constructor (`CusparseSpmv(A, alg, x, y)`, then `run(stream)`), because `cusparseSpMV_preprocess` is
> tied to the vector descriptors. `read_matrix_market` also takes an optional `CacheMode`.

Commit the headers below (declarations only) before anyone implements anything. Then the three of
you can work in parallel without waiting on each other. Names are suggestions; change them
together, not separately.

```cpp
// include/rasp/csr.hpp
namespace rasp {
template <class T> struct HostCsr {            // 0-based, columns sorted within each row, no duplicates
  int32_t rows, cols; int64_t nnz;             // nnz must be < 2^31 (cuSPARSE 32-bit indices); assert it
  std::vector<int32_t> row_ptr, col_idx;       // row_ptr.size() == rows + 1
  std::vector<T> vals;
};
template <class T> class DeviceCsr {           // owns GPU memory: move-only, frees in destructor (RAII)
 public:
  static DeviceCsr upload(const HostCsr<T>&);
  HostCsr<T> download() const;
  int32_t rows() const; int32_t cols() const; int64_t nnz() const;
  const int32_t* row_ptr() const; const int32_t* col_idx() const; const T* vals() const;
};
}  // namespace rasp

// include/rasp/io.hpp
HostCsr<double> read_matrix_market(const std::string& path);   // uses the binary cache (below) when present
void write_matrix_market(const std::string& path, const HostCsr<double>&);
template <class T> HostCsr<T> convert_values(const HostCsr<double>&);   // double -> float for FP32 runs

// include/rasp/timer.hpp
struct TimingStats { double median_ms, p25_ms, p75_ms, min_ms; int reps; };
struct BenchOptions { int warmup = 5; int reps = 20; bool flush_l2 = true; };
TimingStats time_gpu(const std::function<void(cudaStream_t)>& fn, cudaStream_t, const BenchOptions&);

// include/rasp/csv.hpp: one row = key/value pairs; the environment stamp is written into every row
class CsvWriter { public: explicit CsvWriter(const std::string& path);
  void row(const std::vector<std::pair<std::string, std::string>>& fields); };
std::vector<std::pair<std::string, std::string>> environment_stamp();  // gpu, driver, cuda, git hash, time

// include/rasp/cusparse_ref.hpp: the baselines and the correctness reference
enum class SpmvAlg { CsrAlg1, CsrAlg2 };
template <class T> class CusparseSpmv {        // does handle, descriptors and buffer once; run() is the timed part
 public:
  CusparseSpmv(const DeviceCsr<T>&, SpmvAlg);
  void run(const T* x, T* y, cudaStream_t);    // y = A x
};
template <class T> DeviceCsr<T> cusparse_spgemm(const DeviceCsr<T>& A, const DeviceCsr<T>& B);
```

---

## Tasks

| # | Task | Suggested owner | Depends on |
|---|------|-----------------|------------|
| 1 | CMake layout: `rasp` static library + `bench/` + `tests/` (CTest), `-Wall`, `-lineinfo` | you | — |
| 2 | Timer, L2 flush, CSV writer, environment stamp | you | 1 |
| 3 | Matrix Market reader + binary cache + `HostCsr` / `DeviceCsr` | Harsh | 1 |
| 4 | cuSPARSE SpMV wrapper + CPU reference SpMV + unit tests | Harsh | 3 |
| 5 | cuSPARSE SpGEMM wrapper + CPU/SciPy check | Karmanya | 3 |
| 6 | `scripts/gen_laplace.py` (A, R, P) | Karmanya | — |
| 7 | `scripts/fetch_matrices.py` + manifest | Karmanya | — |
| 8 | `bench/bench_spmv.cu`: glue 2 + 3 + 4 → CSV | you | 2, 3, 4 |

Suggested timeline: **Week 1** tasks 1, 3, 6, 7 and the header agreement. **Week 2** tasks 2, 4, 5,
8, then the 50-matrix run and a progress.md entry.

---

## Task details and traps

### 1. CMake
- Build the library from `src/` (`add_library(rasp STATIC ...)`), link `CUDA::cudart CUDA::cusparse`, and link every bench/test against `rasp`.
- Add `-lineinfo` for CUDA so `ncu` maps metrics back to source lines. It costs nothing at runtime.
- Keep `uvm_probe` building as it does now.

### 2. Timer and harness
- Time with CUDA events around each call on the same stream. Do the warm-up runs, then report
  **median and IQR** from ≥ 20 runs, never the mean.
- **L2 trap (important):** the RTX 4050 has a **24 MB L2**. Any matrix + vectors smaller than about
  24 MB stays cached between repetitions and looks faster than it really is. Before each timed
  repetition, write a 48 MB scratch buffer (`cudaMemsetAsync`) to flush L2. Record `flush_l2` in the
  CSV so warm and cold numbers are never mixed.
- Environment stamp: GPU name, driver/runtime version (`cudaDriverGetVersion`), git hash
  (`git rev-parse --short HEAD` at runtime, or baked in by CMake), ISO timestamp, and whether the
  tree was dirty.
- Write CSV numbers at full precision (`%.6g` or more). Plots are made from the CSV, never by hand.

### 3. Matrix Market reader
- Header: `%%MatrixMarket matrix coordinate <real|integer|pattern> <general|symmetric|skew-symmetric>`.
  **Reject** `complex` and `array` with a clear error. Skip `%` comment lines.
- Indices are **1-based** in the file and 0-based in memory.
- `pattern` → every value = 1.0. `symmetric` → for each off-diagonal (i, j) also add (j, i), but add
  diagonal entries only once. `skew-symmetric` → add (j, i) with the negated value.
- Build CSR by counting per row → exclusive prefix sum → scatter. Then **sort columns inside each
  row** and **sum duplicates**; some SuiteSparse files contain them.
- **Binary cache:** text parsing of a 50M-nnz file takes tens of seconds. After the first read,
  write `name.csr.bin` next to it (small header + raw arrays) and load that on later runs. It
  makes the full-dataset sweeps in Phases 2–3 practical.
- Keep matrices that aren't square in the reader, because SpGEMM's R and P are rectangular.

### 4. cuSPARSE SpMV wrapper
- Use the generic API: `cusparseCreateCsr`, `cusparseCreateDnVec`, `cusparseSpMV_bufferSize`,
  and `cusparseSpMV_preprocess` (if present) once in the constructor. Then `run()` calls only
  `cusparseSpMV`. **Time the preprocess separately.** It is a conversion cost, just like our
  formats' conversion (G4).
- `cusparseSetStream(handle, stream)` inside `run()`.
- CPU reference: a plain double-precision loop. Check with the relative L2 error
  `‖y − y_ref‖ / ‖y_ref‖`: ≤ 1e-12 for FP64, ≤ 1e-5 for FP32. Use a random `x` in [−1, 1] with a
  fixed seed, not all ones: all ones hides column-index bugs in symmetric matrices.

### 5. cuSPARSE SpGEMM wrapper
- Sequence: `cusparseSpGEMM_createDescr` → `workEstimation` (call twice: size query, then with
  buffer) → `compute` (twice) → `cusparseSpMatGetSize` for nnz(C) → allocate C →
  `cusparseCsrSetPointers` → `cusparseSpGEMM_copy`.
- Verify small cases against SciPy (`A @ B`) by writing C to `.mtx` and comparing in Python, or
  against a simple CPU hash-map SpGEMM.
- Record the peak extra memory the buffers use. Karmanya needs it for the Phase 4 memory budget.

### 6. `gen_laplace.py`
- A = 7-point 3D Laplacian on an n×n×n grid (`scipy.sparse`: a Kronecker sum of 1D `[-1, 2, -1]`).
- P = piecewise-constant aggregation: group 2×2×2 cells into one aggregate, so P is (n³) × (n/2)³
  with one 1.0 per row. R = Pᵀ.
- Arguments: `--n`, `--out`. Write `.mtx` (use `scipy.io.mmwrite`). Print the sizes in MB, so later
  we can pick n values that give A sizes of 0.5×–4× the fast-memory budget.
- This reproduces paper 1's Laplace3D R×A and A×P workloads.

### 7. `fetch_matrices.py`
- Use `ssgetpy.search(...)` with filters: real, square, `nnz` between 1e4 and 50e6, `limit`. Then
  `download(format='MM', destpath=...)` and extract.
- Always write/append `data/suitesparse/manifest.csv` (group, name, rows, cols, nnz, kind,
  is_spd, path). The selector's grouped cross-validation in Phase 3 needs `group` and `kind`.
- Include the paper-2 named matrices explicitly (rajat23, TSOPF*, case39, Schenk_AFE, GHS_indef, …; list in plan.md §2.4).
- Make it resumable: skip files that already exist.

### 8. `bench_spmv`
- Columns: environment stamp, matrix, group, rows, cols, nnz, precision, kernel, alg, flush_l2,
  median_ms, p25_ms, p75_ms, gflops (= 2·nnz / t), eff_bw_gbs, rel_err, preprocess_ms.
- Effective bandwidth, the minimum bytes moved: `nnz·(sizeof(T)+4) + (rows+1)·4 + cols·sizeof(T) + rows·sizeof(T)`.
  Compare it with the measured **187.8 GB/s**. That percentage is the headline metric in Phases 2–3.
- If a matrix fails (read error, OOM), write a row with an `error` column and continue.

---

## Unit tests (`tests/`, run with `ctest`)
- 3×3 hand-made matrix with known `y`
- a matrix with **empty rows**, including the first and last row
- **one dense row** among very short rows (the case paper 2 is built for)
- a 1×1 matrix, and a matrix with nnz = 0
- symmetric and pattern `.mtx` files → expanded correctly (compare with a hand-expanded general file)
- duplicate entries summed
- binary cache round trip: text → bin → load → identical arrays
- SpGEMM: a small A·B against a hand-computed C

Put tiny `.mtx` fixtures in `tests/data/`. These are committed, unlike `data/`.

---

## Before closing Phase 1
- [x] `ctest` passes on a clean build (`rm -rf build && cmake -B build -G Ninja && cmake --build build && ctest --test-dir build`)
- [x] `results/phase1_cusparse.csv` has ≥ 50 matrices × {FP32, FP64} × {ALG1, ALG2}, all with `rel_err` in tolerance
- [x] A quick plot of GFLOP/s or % of bandwidth vs nnz (`scripts/plot_phase1.py`) so everyone sees the baseline
- [x] progress.md entry with numbers: median % of bandwidth for cuSPARSE, fastest/slowest matrices, ALG1 vs ALG2
