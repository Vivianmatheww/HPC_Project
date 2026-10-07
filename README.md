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

**Phases 1–3 done and measured** (SpMV: kernels, paper-2 reproduction, runtime selector with break-even).
**Phases 4–6 implemented, GPU runs pending**: out-of-core SpGEMM (placement study, chunked executor, Alg. 4 baseline,
pipelined planner with calibration and re-planning), the CG and AMG applications, profiling, the LaTeX report and
Beamer slides. The Phase 4–5 numbers come from `scripts/run_phase4.sh` and `scripts/run_phase5.sh` on the RTX 4050
machine. See [progress.md](progress.md).

## Target platform

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 4050 Laptop, 6 GB GDDR6 (~192 GB/s), compute capability 8.9 (Ada) |
| Host link | PCIe 4.0 x8 |
| CPU | AMD Ryzen 7 7840HS (16 threads) |
| OS | Ubuntu 22.04 on WSL2 (Windows host has 15.2 GB RAM; WSL gets 7 GB by default, 11 GB after `.wslconfig`) |
| Driver | 615.78 (CUDA 13.4 capable) |
| Toolchain | CUDA Toolkit 13.3, CMake 4.4 + Ninja (from `.venv`), GCC 11.4, Python 3.10 |

## Repository layout

```
HPC_Project/
├── CMakeLists.txt
├── include/rasp/          # public headers: matrix types, SpMV plan, SpGEMM context and planners
├── src/
│   ├── io/                # Matrix Market reader, CSR builder, binary cache
│   ├── formats/           # CSR -> ELL / SELL-C-σ / ArgCSR(-local) / CSR-binned converters (host and GPU)
│   ├── spmv/              # SpMV kernels behind one interface
│   ├── features/          # GPU feature extraction
│   ├── selector/          # rules + trained decision tree (generated C++), SpmvPlan
│   ├── cusparse/          # cuSPARSE baselines; SpGEMM core on views with caller-provided buffers
│   └── spgemm/            # estimator, planners (Alg. 4, RASP), arena allocator, kernels, executor, gallery
├── bench/                 # bench_spmv, bench_features, bench_plan, bench_spgemm -> CSV in results/
├── apps/                  # cg_solver (SpMV), amg_setup (SpGEMM), rasp_demo (live demo)
├── tests/                 # unit tests (ctest), incl. host-only planner tests that run without a GPU
├── scripts/               # data, training, analysis, run_phase{4,5}.sh, run_all.sh, make_report_assets.py
├── data/                  # (git-ignored) downloaded matrices
├── results/               # raw CSVs + figures
└── docs/                  # phase guides, report/ (LaTeX), slides/ (Beamer)
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

# Phase 3: features + GPU conversion costs, train the selector (regenerates src/selector/tree_generated.hpp),
# rebuild, then evaluate end to end on the held-out set (the selector is frozen since Phase 3; run_all.sh does not retrain)
./build/bench_features --matrices data/suitesparse --out results/phase3/features_train.csv --convert-out results/phase3/gpu_convert_train.csv
python scripts/train_selector.py && cmake --build build
python scripts/fetch_matrices.py --count 40 --seed 1 --out data/heldout --exclude data/suitesparse/manifest.csv
./build/bench_spmv --matrices data/heldout --kernels all --out results/phase3/heldout_spmv.csv
./build/bench_plan --matrices data/heldout --out results/phase3/plan_heldout.csv
python scripts/analyze_phase3.py

# Phase 4: out-of-core SpGEMM (placement study, sweeps over budgets, analysis); QUICK=1 for a 3-minute smoke test
./build/bench_spgemm --study plans --n 128,256 --budget-mb 512     # the plans both planners would make; no GPU needed
bash scripts/run_phase4.sh

# Phase 5: CG over the SPD set, AMG set-up, live-demo transcript, Nsight profiles, analysis
bash scripts/run_phase5.sh

# Phase 6: frozen rerun of everything from a clean build, then the report and slides
git tag v1.0-freeze && bash scripts/run_all.sh
python scripts/make_report_assets.py && (cd docs/report && latexmk -pdf main.tex) && (cd docs/slides && latexmk -pdf slides.tex)
```

Without an NVIDIA GPU you can still compile everything and run the host-only planner tests: install the CUDA 13.3
toolkit wheels into the venv (`pip install nvidia-cuda-nvcc==13.3.73 nvidia-cuda-runtime nvidia-cusparse nvidia-cublas
nvidia-cuda-cccl nvidia-nvtx nvidia-cuda-crt nvidia-nvvm`), add unversioned `lib*.so` links in
`.venv/lib/python3.12/site-packages/nvidia/cu13/lib`, configure with
`-DCMAKE_CUDA_COMPILER=$T/bin/nvcc -DCUDAToolkit_ROOT=$T` (`T` = that `cu13` directory), and run
`LD_LIBRARY_PATH=$T/lib ./build/test_spgemm_plan`.

`bench_spmv` flushes the 24 MB L2 before every timed repetition (`--no-flush` to disable) and
reports the median of 20 runs after 5 warm-ups. Matrix Market files are parsed once and cached as
`<file>.csr.bin` next to the `.mtx`.

| Code | What it is |
|------|-----------|
| [include/rasp/](include/rasp/) | Shared interfaces: `HostCsr`/`DeviceCsr`, I/O, timer, CSV, cuSPARSE baselines, CPU references |
| [src/](src/) | Their implementations (`rasp` static library) |
| [bench/bench_spmv.cu](bench/bench_spmv.cu) | SpMV benchmark driver → CSV |
| [bench/bench_spgemm.cu](bench/bench_spgemm.cu) | SpGEMM placement study, out-of-core sweep, plan dry run |
| [apps/](apps/) | `cg_solver`, `amg_setup`, `rasp_demo` |
| [docs/report/](docs/report/), [docs/slides/](docs/slides/) | LaTeX report and Beamer slides; every number via `\stat{key}` from `scripts/make_report_assets.py` |
| [bench/platform/uvm_probe.cu](bench/platform/uvm_probe.cu) | Phase 0 memory-placement / UVM probe |
| [tests/](tests/) | Unit tests (`ctest`) and `.mtx` fixtures |
| [scripts/](scripts/) | Data download/generation and plotting |

## API

```cpp
#include <rasp/rasp.hpp>

auto A = rasp::DeviceCsr<double>::upload(rasp::read_matrix_market("data/foo.mtx"));

// SpMV (implemented): the plan is built once and reused. The hint says how many calls to expect.
rasp::PlanOptions opt;
opt.expected_calls = 500;
rasp::SpmvPlan<double> plan(A, d_x, d_y, opt);
plan.run(stream);                                  // y = A x with the selected kernel
std::puts(plan.report().to_string().c_str());      // chosen config, features, costs, and why

// SpGEMM (Phase 4): A, B, C in pinned host memory, at most `budget` bytes of device memory
auto Ap = rasp::PinnedCsr<double>::from(hostA), Bp = rasp::PinnedCsr<double>::from(hostB);
rasp::SpgemmContext ctx(512ull << 20);             // budget arena, streams, PCIe calibration
rasp::PinnedCsr<double> C;                         // reuse across calls (pinned allocation is slow)
rasp::SpgemmReport rep;
ctx.multiply(Ap, Bp, C, {}, &rep);                 // RASP: simulated plan, overlap, probe + re-planning
ctx.multiply(Ap, Bp, C, {.planner = rasp::SpgemmPlanner::Alg4}, &rep);   // paper 1, Alg. 4 baseline
std::puts(rep.to_string().c_str());                // mode, partitions, times, bytes moved, peak memory
```

## Team

| Member | Role (proposed) |
|--------|-----------------|
| _(your name)_ | Infrastructure lead: toolchain, benchmark harness, datasets, UVM/placement platform study, integration and evaluation |
| Harsh | Track A: SpMV formats, ArgCSR reproduction, runtime selector, CG demo |
| Karmanya | Track B: out-of-core SpGEMM, chunking and overlap pipeline, AMG triple-product demo |
