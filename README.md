# RASP — Runtime-Adaptive Sparse Matrix Optimization for GPUs

RASP is a CUDA C++ library and benchmark suite that **inspects a sparse matrix at runtime and
automatically picks how to store and process it on the GPU**:

- **SpMV (y = A·x)** — chooses the storage format (CSR-scalar, CSR-vector, ELL, SELL-C-σ,
  Adaptive Row-grouped CSR, or cuSPARSE) *and* its tuning parameters from cheap GPU-computed
  matrix features, while accounting for conversion cost against the expected number of SpMV calls.
- **SpGEMM (C = A·B)** — chooses between in-core, zero-copy and chunked out-of-core execution
  when the matrices don't fit in GPU memory, using a runtime-calibrated cost model and
  copy/compute overlap.

It builds on two papers, and extends both:

| # | Paper | What we take | What we add |
|---|-------|--------------|-------------|
| 1 | Deveci, Hammond, Wolf, Rajamanickam — *Sparse Matrix-Matrix Multiplication on Multilevel Memory Architectures: Algorithms and Experiments* (Sandia, 2018) — [arXiv:1804.00695](https://arxiv.org/abs/1804.00695) | Chunked SpGEMM (Alg. 2/3) and the chunk-partitioning heuristic (Alg. 4) | Double-buffered copy/compute overlap (their stated future work), a cost model calibrated at runtime instead of a fixed 75/25 split, re-partitioning while running, and a PCIe consumer GPU instead of an NVLink P100 |
| 2 | Heller, Oberhuber — *Adaptive Row-grouped CSR Format for Storing of Sparse Matrices on GPU* (CTU Prague, 2012) — [arXiv:1203.5737](https://arxiv.org/abs/1203.5737) | The ArgCSR format and SpMV kernel | Automatic per-matrix format and parameter selection (they fix `desiredChunkSize` globally), a conversion break-even model, and modern baselines on Ada-generation hardware |

See [project_context.md](project_context.md) for the full paper analysis and how we differ,
[plan.md](plan.md) for the design and schedule, and [progress.md](progress.md) for status.

## Status

**Phases 1–2 implemented; Phase 3 in progress** (candidate set widened with CSR-binned and ArgCSR-local; selector next). See [progress.md](progress.md).

## Target platform

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 4050 Laptop, 6 GB GDDR6 (~192 GB/s), compute capability 8.9 (Ada) |
| Host link | PCIe 4.0 x8 |
| CPU | AMD Ryzen 7 7840HS (16 threads) |
| OS | Ubuntu 22.04 on WSL2 (Windows host has 15.2 GB RAM; WSL gets 7 GB by default, 11 GB after `.wslconfig`) |
| Driver | 615.78 (CUDA 13.4 capable) |
| Toolchain | CUDA Toolkit 13.3, CMake 4.4 + Ninja (from `.venv`), GCC 11.4, Python 3.10 |

## Repository layout (planned; Phase 1 parts exist)

```
HPC_Project/
├── CMakeLists.txt
├── include/rasp/          # public headers: matrix types, spmv(), spgemm(), selector API
├── src/
│   ├── io/                # Matrix Market reader, CSR builder
│   ├── formats/           # CSR -> ELL / SELL-C-σ / ArgCSR converters (on GPU)
│   ├── spmv/              # SpMV kernels, one file per format
│   ├── features/          # GPU feature extraction (row-length stats, padding ratios, ...)
│   ├── selector/          # rule-based + trained decision-tree selector, cost model
│   └── spgemm/            # in-core wrapper, chunked driver, pipeline, placement cost model
├── bench/                 # bench_spmv, bench_spgemm -> CSV in results/
├── apps/                  # cg_solver (SpMV consumer), amg_triple_product (SpGEMM consumer)
├── tests/                 # correctness vs cuSPARSE / CPU reference
├── scripts/               # fetch_matrices.py, gen_laplace.py, train_selector.py, plot_*.py
├── data/                  # (git-ignored) downloaded / generated matrices
├── results/               # raw CSVs + figures
└── docs/                  # paper notes, report, slides
```

## Build and run

```bash
# one-time setup
bash scripts/setup_cuda_wsl.sh                                       # CUDA 13.3 toolkit (needs sudo), adds nvcc to PATH
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # CMake, Ninja, NumPy/SciPy/pandas, sklearn, ssgetpy
source .venv/bin/activate

# build + unit tests
cmake -B build -G Ninja            # Release, sm_89 by default; add -DCMAKE_CUDA_ARCHITECTURES=86 etc. on other GPUs
cmake --build build
ctest --test-dir build --output-on-failure

# data
python scripts/fetch_matrices.py --count 50 --named     # SuiteSparse sample + paper-2 matrices -> data/suitesparse/
python scripts/gen_laplace.py --n 64 128                # paper-1 Laplace3D A, P, R -> data/laplace/

# Phase 1 baseline: cuSPARSE SpMV, FP32/FP64, ALG1/ALG2, checked against a CPU reference
./build/bench_spmv --matrices data/suitesparse --out results/phase1_cusparse.csv
python scripts/plot_phase1.py

# Phase 2: all 33 kernel configs (our formats + cuSPARSE), oracle table and figures
./build/bench_spmv --matrices data/suitesparse --kernels all --out results/phase2_spmv.csv
python scripts/analyze_phase2.py
```

`bench_spmv` flushes the 24 MB L2 before every timed repetition (`--no-flush` to disable) and
reports the median of 20 runs after 5 warm-ups. Matrix Market files are parsed once and cached as
`<file>.csr.bin` next to the `.mtx`.

| Code | What it is |
|------|-----------|
| [include/rasp/](include/rasp/) | Shared interfaces: `HostCsr`/`DeviceCsr`, I/O, timer, CSV, cuSPARSE baselines, CPU references |
| [src/](src/) | Their implementations (`rasp` static library) |
| [bench/bench_spmv.cu](bench/bench_spmv.cu) | SpMV benchmark driver → CSV |
| [bench/platform/uvm_probe.cu](bench/platform/uvm_probe.cu) | Phase 0 memory-placement / UVM probe |
| [tests/](tests/) | Unit tests (`ctest`) and `.mtx` fixtures |
| [scripts/](scripts/) | Data download/generation and plotting |

## Planned API

```cpp
#include <rasp/rasp.hpp>

rasp::CsrMatrix A = rasp::read_matrix_market("data/foo.mtx").to_device();

// SpMV: the plan is built once and reused. The hint says how many calls to expect.
auto plan = rasp::make_spmv_plan(A, {.expected_calls = 500});
plan.run(x, y);                 // uses whichever format/kernel was selected
plan.report();                  // selected format, features, predicted vs measured time

// SpGEMM: works even when A, B, C exceed GPU memory
auto C = rasp::spgemm(A, B, {.fast_mem_budget = 2_GiB});
```

## Team

| Member | Role (proposed) |
|--------|-----------------|
| _(your name)_ | Infrastructure lead: toolchain, benchmark harness, datasets, UVM/placement platform study, integration and evaluation |
| Harsh | Track A: SpMV formats, ArgCSR reproduction, runtime selector, CG demo |
| Karmanya | Track B: out-of-core SpGEMM, chunking and overlap pipeline, AMG triple-product demo |
