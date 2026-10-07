# Progress Log

How to use this file: update the **status table** whenever a phase changes state, and add a
**dated entry** at the top of the log for each work session or weekly meeting. Say what was done,
what we learned (with numbers), what is blocked, and what comes next. Link to CSVs and figures in `results/`.

Legend: ⬜ not started · 🟨 in progress · ✅ done · ⛔ blocked

## Status

| Phase | Weeks | Dates | Status | Notes |
|-------|-------|-------|--------|-------|
| 0 — Setup | 0 | Sep 23 – Sep 27 | ✅ | Toolchain, 11 GB WSL, CMake, UVM probe, git, nsys + ncu. Carried over: teammate setup |
| 1 — Infrastructure | 1–2 | Sep 28 – Oct 9 | ✅ | Done Sep 23 (early). 28 unit tests; cuSPARSE baseline on 60 matrices |
| 2 — SpMV kernels + paper-2 reproduction | 3–4 | Oct 12 – Oct 23 | ✅ | Done Sep 23 (early). 33 configs × 60 matrices; oracle table |
| 3 — Runtime SpMV selector | 5–6 | Oct 26 – Nov 6 | ✅ | Done Sep 24: GPU converters + features, trained tree selector, SpmvPlan; held-out evaluation |
| 4 — Out-of-core adaptive SpGEMM | 7–9 | Nov 9 – Nov 27 | 🟨 | Code complete Oct 8 (branch `phases-4-6`); GPU run pending: `scripts/run_phase4.sh` |
| 5 — Integration + apps | 10 | Nov 30 – Dec 4 | 🟨 | CG, AMG set-up, demo, profiling scripts written; GPU run pending: `scripts/run_phase5.sh` |
| 6 — Evaluation + report | 11–12 | Dec 7 – Dec 18 | 🟨 | `run_all.sh`, report (Phases 1–3 complete) and slides drafted; Phase 4–5 numbers and interpretation pending |

## Goal tracker

| Goal | Target | Current | Status |
|------|--------|---------|--------|
| G1 ArgCSR reproduction | same qualitative trends | chunk-size sensitivity reproduced (rajat23 D=32: 0.78 vs paper 0.81 GFLOPS); "ArgCSR ≫ cuSPARSE" does **not** hold against modern cuSPARSE | ✅ |
| G2 Selector vs oracle | geomean slowdown ≤ 1.10; no large-matrix loss > 5% | held-out: **1.074**; 1.051× over cuSPARSE; 0 large-matrix losses (only 4 large samples) | ✅ |
| G3 Selection overhead | ≤ 2 SpMV | held-out median **1.97** SpMVs (0.28 for > 1M nnz, 2.5 for < 100k); skipped entirely when N is small | 🟨 borderline |
| G4 Break-even aware | ≤ 5% loss at N = 1 | N=1: identical work to cuSPARSE in 80/80 plans (0.996×, timing noise); N=1000: **1.052×** | ✅ |
| G5 Placement study (Table 3) | table | Memory-level probe done (host-side reads 14–28× slower); SpGEMM-level study implemented (`bench_spgemm --study placement`), not yet run | 🟨 |
| G6 Overlapped SpGEMM | ≥ 1.3× over Alg. 4 | implemented (executor, Alg. 4 baseline, simulated planner, re-planning); not yet run | 🟨 |
| G7 CG + AMG demos | working | `apps/cg_solver`, `apps/amg_setup` written with self-checks; not yet run | 🟨 |

## Measured platform numbers (fill in during Phase 0)

| Quantity | Value | How measured |
|----------|-------|--------------|
| Device memory bandwidth (achieved) | **187.8 GB/s** seq read (98% of 192), **172 GB/s** copy r+w | `uvm_probe` |
| H2D pinned bandwidth | **13.4 GB/s** (pageable 11.9) | `uvm_probe`, 256 MiB, median of 10 |
| D2H pinned bandwidth | **13.2 GB/s** (pageable 12.5); H2D+D2H concurrently = 13.3 GB/s total | same |
| `asyncEngineCount` | **1** (H2D and D2H serialise; only copy ‖ compute overlaps) | `cudaDeviceGetAttribute` |
| UVM on WSL2 | **Managed memory stays in host RAM and is read over PCIe (≡ zero-copy).** Warm reads 13.2 GB/s vs 187.8 device. Prefetch/Advise → `invalid device ordinal`. 1.25× VRAM alloc → `out of memory` | `uvm_probe` |
| Nsight Systems in WSL2 | **Works** (nsys 2026.1.3: CUDA trace + `nsys stats`) | saxpy test kernel |
| Nsight Compute in WSL2 | **Works** after enabling counters for all users in the NVIDIA Control Panel (ncu 2026.2.1: SpeedOfLight + MemoryWorkloadAnalysis sections) | `ncu` on saxpy |
| GPU basics | 20 SMs, 6.44 GB, 96-bit bus, 24 MB L2, 99 KB smem/block opt-in | `cudaGetDeviceProperties` |
| Windows host RAM / WSL limit | 15.2 GB / **11 GB + 8 GB swap** (applied) | PowerShell, `free -g`, `swapon` |
| Random-gather penalty, host vs device | 7.8× (256 MiB) · 21× (1 GiB) · 28× (4 GiB) | `uvm_probe`, 16M random 8-byte reads |

## Open questions for the team
- [ ] What is the real submission deadline? (The plan assumes Dec 18, 2026.)
- [x] Team: _(your name)_, Harsh, Karmanya. Proposed roles are in plan.md §5; confirm at the first meeting.
- [ ] Do we have access to a university cluster GPU (native Linux) for the UVM baseline and a cross-GPU check?
- [ ] Is FP64 required by the course, or is FP32 acceptable as the primary precision?
- [ ] Report format and page limit? Is a presentation required?

---

## Log

### 2026-10-08 (Thu) — Phases 4–6 implemented; GPU runs pending
Written on a laptop without an NVIDIA GPU (Harsh). Everything was compiled here with the CUDA 13.3 toolkit from pip
wheels; nothing that launches a kernel has run yet. Branch `phases-4-6`, one commit per step.

**Built**
- **Out-of-core SpGEMM** ([include/rasp/spgemm.hpp](include/rasp/spgemm.hpp), [src/spgemm/](src/spgemm/)):
  - `SpgemmContext(budget)`: one arena of the budget carved by a first-fit allocator (the budget holds exactly; peak
    is recorded), a copy stream and a compute stream, and PCIe calibration at creation.
  - Modes: in-core, zero-copy, ChunkB (paper 1's chunk2, including several B blocks with partial C parked in pinned
    memory) and ChunkAC (chunk1, with on-device partial sums by our own merge kernels). B blocks are full-height
    matrices with empty rows, so A is never split by column.
  - Uploads run depth−1 tiles ahead of cuSPARSE. Per-tile CUDA events give a timeline; NVTX labels the tiles for nsys.
  - Tiles that do not fit are split and redone, so the estimates never affect correctness.
  - The cuSPARSE SpGEMM wrapper was refactored into a core that works on views with caller-provided buffers.
- **Planners** ([include/rasp/spgemm_plan.hpp](include/rasp/spgemm_plan.hpp)), host-only and unit-tested here:
  - The flop and C-size estimator (OpenMP, parallel scan) and paper 1's BinarySearch splitter.
  - A pipeline simulator that models one copy engine.
  - `plan_alg4` (75/25 split, copy volume, depth 1; a depth-2 variant for the ablation).
  - `plan_rasp` simulates every mode × block size × depth 2–3 and gives the streamed side all the free memory. It
    adds a probe block, and re-plans when the measured rate, compression or workspace drifts by more than 20%.
- **Benchmarks:** [bench/bench_spgemm.cu](bench/bench_spgemm.cu) has three studies:
  - `placement`: paper 1's Table 3, each operand in device / pinned / managed memory.
  - `ooc`: Alg. 4, Alg. 4 + overlap, RASP static, RASP, zero-copy, in-core and device-only, in shuffled order;
    checks nnz(C) agreement and 2,000 rows against the CPU.
  - `plans`: a dry run that needs no GPU.
- **Apps:** [apps/cg_solver.cu](apps/cg_solver.cu) (Jacobi-PCG through `SpmvPlan`, iteration cap as the hint),
  [apps/amg_setup.cu](apps/amg_setup.cu) (Galerkin R·(A·P) per level, checked by cross-method agreement and the row-sum
  identity) and [apps/rasp_demo.cu](apps/rasp_demo.cu) (the live demo). The first two have ctest smoke tests.
- **Data and Laplace generation:**
  - [include/rasp/gallery.hpp](include/rasp/gallery.hpp) generates Laplace3D, its aggregation and the transpose in
    memory, identical to `gen_laplace.py` (tested).
  - `fetch_matrices.py --spd` / `--names` fetch the CG set and SNAP graphs.
- **Scripts:**
  - `run_phase4.sh`, `run_phase5.sh`, `profile_kernels.sh` (ncu metrics, nsys traces) and `run_all.sh` (the frozen rerun).
  - `analyze_phase4.py` and `analyze_phase5.py`.
  - `make_report_assets.py`: every report number becomes a `\stat{key}`, and every figure a PDF.
- **Report and slides:** [docs/report/](docs/report/) (13 pages, builds cleanly) and [docs/slides/](docs/slides/)
  (13 frames). Phases 1–3 are complete with numbers generated from the committed CSVs. Phases 4–5 show red `??key`
  markers and `\todo` notes until the runs exist.

**Verified on this laptop (no GPU)**
- The whole project compiles warning-free.
- `test_spgemm_plan`: 20 host-only tests pass (gallery vs `gen_laplace.py`, estimator vs brute force, splitter,
  simulator schedules, a hand-computed Alg. 4 case, plans within budget, probe, re-plan, the allocator stress test).
- `analyze_phase2.py` and `analyze_phase3.py` reproduce `spmv_oracle.csv`, `summary.txt` and `heldout_summary.txt`
  byte for byte from the committed CSVs.
- `analyze_phase4.py` and `analyze_phase5.py` ran end to end on synthetic CSVs that use the benchmarks' exact
  column schemas.
- `bench_spgemm --study plans`, using the prior parameters (these are predictions, not measurements), predicts
  RASP 1.3–2.1× faster than Alg. 4 on Laplace n = 128–256 when the problem exceeds a 512 MiB budget.

**Not verified (needs the GPU machine)**
- `test_spgemm_ooc` and the app smoke tests: every executor mode, planner and depth checked against the CPU.
- Whether cuSPARSE's blocking size queries leave H2D free to overlap compute: `NSYS=1 bash scripts/run_phase4.sh`
  gives the trace.
- cuSPARSE's real SpGEMM workspace size. The priors are guesses; the probe block measures it, and the overflow
  splits cover a wrong guess.
- Pinned memory limits under WSL2 at n = 256 (about 5 GB pinned).

**Problem found:** `.gitignore`'s `data/` also ignored `tests/data/`, so the `.mtx` fixtures of `test_io` and
`test_spmv` were never committed. Those two suites fail on any fresh clone. Fixed the pattern (`/data/`).
**Vivian: please `git add tests/data/*.mtx` from your machine** (only you have the originals).

**Next (GPU machine, in order):**
1. `QUICK=1 bash scripts/run_phase4.sh` (about 3 min: tests plus a small sweep), then the full run. Commit `results/phase4/`.
2. `QUICK=1 bash scripts/run_phase5.sh`, then the full run. Commit `results/phase5/`.
3. `python scripts/make_report_assets.py` and rebuild the report. Fill the `\todo` notes from the summaries.
4. At the freeze: `git tag v1.0-freeze && bash scripts/run_all.sh`.

### 2026-09-24 (Thu) — Phase 3 complete: runtime SpMV selector
**Built**
- **GPU converters** ([src/formats/convert_gpu.cu](src/formats/convert_gpu.cu)) for ELL, SELL-C-σ (σ sort = stable
  64-bit-key CUB radix sort), CSR-binned (radix sort on the bin), ArgCSR(-local) (downloads `row_ptr` only; layout on the
  host because it is sequential; all nonzeros move on the GPU). A test proves them **bit-identical** to the host converters.
  Median conversion cost in cuSPARSE SpMVs, host → GPU: ELL 47 → 14, SELL 47 → 21, ArgCSR 87 → 58, CSR-binned 5 → 6.5.
- **GPU features** ([src/features/features.cu](src/features/features.cu)): one kernel; row-length mean/CV/max, empty
  fraction, ELL and SELL-32 fill, long-row nnz fraction, column span, 8-bucket length histogram (20-feature vector). Matches
  the CPU reference to 1e-12.
- **Selectors** ([src/selector/](src/selector/)): v1 hand-written rules; v2 decision tree trained by
  [scripts/train_selector.py](scripts/train_selector.py) and compiled in as the generated
  [tree_generated.hpp](src/selector/tree_generated.hpp) together with its cost models.
- **`SpmvPlan`** ([include/rasp/plan.hpp](include/rasp/plan.hpp)): feature gate (skip features if
  calls × predicted SpMV time × expected gain < predicted feature time) → tree → conversion break-even → build with the GPU
  converter; falls back if a format is infeasible. `report()` explains every decision.
- [bench/bench_features.cu](bench/bench_features.cu), [bench/bench_plan.cu](bench/bench_plan.cu),
  [scripts/analyze_phase3.py](scripts/analyze_phase3.py); `test_selector` (6 suites, all pass).
- **Held-out set:** 40 new random SuiteSparse matrices (`fetch_matrices.py --seed 1 --exclude ...`), same nnz range,
  no overlap with training.

**Training** (60 matrices × fp32/fp64, grouped 5-fold CV by SuiteSparse group; [report](results/phase3/selector_train.txt))
- Greedy portfolio: cusparse_alg1, csr_vector_v4, csr_binned_t2, csr_scalar, cusparse_alg2. It reaches 1.011 of the full
  40-config oracle.
- Tree: depth 2, 4 leaves (split on log2 max row length, then log2 nnz); CV 1.051 vs oracle.
- The CV also chose a **large-matrix gate** (≥ 1M nnz → cuSPARSE ALG1): it improves *every* metric (CV 1.046, 0 large
  losses vs 5 without). A minimum-expected-gain gate was also tested and rejected by CV (τ = 1.0).
- Cost models: cuSPARSE time = 9.5 µs + bytes / 126 GB/s (median error ×1.06); conversion per family and precision
  (×1.2–1.6); features = 43 µs + 0.23 ns/row.

**Held-out results** ([summary](results/phase3/heldout_summary.txt), [profile](results/phase3/heldout_profile.png),
[N sweep](results/phase3/calls_sweep.png)); all 3,154 SpMV and 1,280 plan results correct

| Steady state (80 samples) | T / T_oracle | within 10% | vs cuSPARSE | > 5% slower than cuSPARSE |
|---|---|---|---|---|
| always cuSPARSE ALG1 | 1.129 | 55% | 1.000× | 0 |
| rules (v1) | 1.128 | 39% | 1.001× | 28 |
| **decision tree (v2)** | **1.074** | **66%** | **1.051×** | 11 (0 large) |
| oracle within portfolio | 1.016 | 92% | 1.111× | 0 |

| Total time vs always-cuSPARSE (geomean [worst]) | N=1 | N=10 | N=100 | N=1000 |
|---|---|---|---|---|
| tree + break-even | 0.996 [0.84] | 0.995 [0.87] | 1.014 [0.78] | **1.052** [0.78] |
| tree, no break-even | 0.601 [0.27] | 0.832 [0.51] | 1.015 | 1.043 |
| rules | 0.988 | 0.998 | 1.004 | 0.994 |

1. **G2 met on unseen matrices:** 1.074 of the oracle (target ≤ 1.10), 1.051× faster than cuSPARSE. Only 4 held-out samples
   are large, so the "never lose on large" half rests mostly on the CV evidence (0/24).
2. **Break-even is what makes the selector safe at small N.** Without it the plan is 40% slower at N=1. With it, all 80 N=1
   plans do exactly cuSPARSE's work (no features), and features are used for 2% of plans at N=10, 89% at N=100 and 100% at N=1000.
3. **G3 borderline:** features cost a median 1.97 SpMVs on held-out (fixed ~40 µs of launch + copy + sync on WSL). They
   are cheap for large matrices (0.28) and expensive for small (2.5), which is why the feature gate skips them for small N.
4. **The hand-written rules (v1) are no better than cuSPARSE.** Fitting thresholds by hand does not transfer; the trained
   tree does.
5. **Where the tree still loses:** DRIVCAV/cavity10/18 and Oberwolfach/piston (fp32, 15–29% slower, picking
   csr_vector_v4). The portfolio oracle (1.016) shows the remaining headroom is in the selector, not the kernels.
6. **Conversion break-even never had to drop a conversion:** the only converting portfolio member (CSR-binned, ~6 SpMVs)
   is always repaid by the time features are worth extracting. The logic matters for the heavy formats (ArgCSR ~58 SpMVs)
   if they enter the portfolio.

**Methodology notes (for the report)**
- Plan-construction timing under WSL has ~5–10 µs position effects. A fixed selector order produced a fake −9% and then a
  fake +9% at N=1. `bench_plan` now shuffles the construction order every repetition (15 reps, fixed seed).
- The feature gate originally used the *maximum* training gain (27%). This extracted features at N=10 for 56% of plans and lost
  ~4%. It now uses the cross-validated *expected* gain (3.9%). This change was prompted by a held-out result; the value itself
  comes only from training-set CV.
- cuSPARSE handle creation: 15 ms the first time in a process, 0.8 µs afterwards, so it does not bias per-plan costs.

### 2026-09-24 (Thu) — Phase 3 step (a): widening the candidate set
**Why:** Phase 2's oracle was only 1.07–1.09× above cuSPARSE, so we added two formats aimed at the gaps it exposed.

**Built**
- **CSR-binned** (`csr_binned_t{1,2,4}`): CSR-vector with the lane count chosen *per row*: v = pow2ceil(len / t), 1–32,
  and a 256-thread block for rows > 4096. Rows are stably grouped by bin; one kernel launch covers all bins
  ([src/formats/binned.cpp](src/formats/binned.cpp), `csr_binned_kernel` in [src/spmv/kernels.cu](src/spmv/kernels.cu)).
- **ArgCSR-local** (`argcsr_local_B{128,256}_R{32,64}`): desiredChunkSize per 4096-row window. Phase 2 data showed
  the best global D tracks mean row length (Spearman ρ = 0.59; CV only −0.28) and keeps a median 41 rows per group
  (IQR 20–75), so each window gets D = pow2_round(R · mean_len / B), halved if CV > 1, clamped to [1, 32]. The
  converter is shared with global ArgCSR; the kernel is unchanged. A test proves that on a uniform matrix the local
  variant equals global ArgCSR with the implied D.
- Tests: `binned_bins`, `argcsr_local_chunk_rule`, and all 40 configs on every edge case (5 suites pass). The rule test
  caught an undefined `1 << -30` for empty regions, which is now fixed.
- `bench_spmv --kernels phase2|extensions|all`; `analyze_phase2.py` merges several CSVs.

**Results** ([results/phase3a_extensions.csv](results/phase3a_extensions.csv): 1,080 runs, all correct; cuSPARSE drift
vs the Phase 2 run: 0.5% geomean, p5–p95 0.98–1.07, so the runs are combined; [figure](results/phase2/perf_profile_extensions.png))

| | FP32 | FP64 |
|---|---|---|
| CSR-binned t=4 vs cuSPARSE ALG1 (geomean) | 0.88× | 0.95× (wins 27/60) |
| ArgCSR-local B128 R32 vs cuSPARSE ALG1 | 0.68× | 0.78× |
| **Oracle vs cuSPARSE, before → after** | **1.068 → 1.072×** | **1.094 → 1.105×** |
| Extensions are the oracle on | 3 / 60 | 6 / 60 |

1. **ArgCSR-local does what it was built for.** One *untuned* config is **1.26× faster than paper 2's default** and
   reaches **0.89 of per-matrix-tuned ArgCSR** (best of 18 B,D per matrix); the best of 4 local configs reaches 0.96.
   Automatic, local parameter choice recovers most of the tuning benefit, which is our improvement over paper 2.
2. **CSR-binned fixes the thread-per-row failure** (ASIC_100k: SELL 48 ms, CSR-vector 0.41 ms → binned 0.24 ms)
   and matches per-matrix-tuned CSR-vector without tuning. It is still 2× behind cuSPARSE on ASIC_100k.
3. **But the oracle headroom barely moves (+0.4–1.0%).** Modern cuSPARSE ALG1 is near-optimal for large matrices, and
   what remains (~7–10% geomean, up to 1.38×) is spread over many formats in small and medium matrices.

**Decision needed (team):** widening the candidates alone does not create a margin over cuSPARSE, so **re-scope G2 to
option (b)**: the selector must be within 10% of the oracle and never lose to cuSPARSE on large matrices. Keep the
widened set as the selector's candidates (ArgCSR-local and CSR-binned are also cheaper to choose, with no per-matrix tuning).

### 2026-09-23 (Wed) — Phase 2 complete: SpMV kernels, ArgCSR reproduction, oracle table
**Built**
- [include/rasp/formats.hpp](include/rasp/formats.hpp) + [src/formats/](src/formats/): host converters for ELL, SELL-C-σ
  and **ArgCSR** (group formation, greedy thread assignment, interleaved chunks; checked exactly against paper 2's Figure 3).
- [include/rasp/spmv.hpp](include/rasp/spmv.hpp) + [src/spmv/](src/spmv/): one `SpmvKernel` interface for all 33 configs
  (cuSPARSE ALG1/2, CSR-scalar, CSR-vector v=2..32, ELL, SELL C∈{8,32} × σ∈{1,256,all}, ArgCSR B∈{64,128,256} × D∈{1..32}).
  The ArgCSR kernel follows the paper's Listing 2.
- `bench_spmv --kernels all|cusparse|<labels>`, `test_formats` (every config × edge cases × fp32/fp64; 5 suites pass),
  [scripts/analyze_phase2.py](scripts/analyze_phase2.py).
- Padding caps: ELL ≤ 3× nnz; SELL/ArgCSR ≤ 64× nnz; any format ≤ 2 GB. (A first run with an 8× cap wrongly refused
  the exact ArgCSR configs paper 2 measured, so it was relaxed and the sweep rerun.)

**Results** (3,960 runs: 3,892 ok, all correct; 68 infeasible, all ELL. Files: [results/phase2_spmv.csv](results/phase2_spmv.csv),
[results/spmv_oracle.csv](results/spmv_oracle.csv), [profile](results/phase2/perf_profile.png),
[chunk sensitivity](results/phase2/argcsr_chunk.png), [summary](results/phase2/summary.txt))

| | FP32 | FP64 |
|---|---|---|
| Oracle (best of 33) vs cuSPARSE ALG1, geomean / max | 1.07× / 1.38× | 1.09× / 1.30× |
| cuSPARSE is the fastest config | 35 / 60 | 27 / 60 |
| CSR-vector (best v) wins | 16 | 26 |
| CSR-scalar / ArgCSR / SELL win | 8 / 0 / 1 | 4 / 2 / 1 |

1. **Paper 2's chunk-size sensitivity reproduces.** rajat23 (FP64, B=128): D=1 8.84 GFLOP/s vs D=32 **0.78**
   (the paper reported 5.1 vs **0.81**); padding grows from 3.3× to 10.2× nnz. af_shell1: D=32 25.3 vs D=1 12.7 (the paper had 18 vs 11,
   same direction).
2. **But a global D=1 (the paper's "robust" choice) is rarely right.** D=1 is the best ArgCSR setting on only 3/60 (FP64)
   and 1/60 (FP32) matrices; the best D spreads over 2–16. Fixing D=1 costs 1.38× geomean vs the per-matrix best D (worst 3.62×,
   nemeth19). This directly supports choosing parameters per matrix.
3. **Paper 2's headline (ArgCSR ≫ cuSPARSE) does not hold against 2026 cuSPARSE.** ArgCSR at the paper default reaches
   0.53× (FP32) / 0.63× (FP64) of cuSPARSE ALG1 (geomean); even tuned it reaches 0.78× / 0.86×. The exceptions are in the family the paper highlighted:
   GHS_indef/ncvxqp9 1.20× faster than cuSPARSE, and torso2 with tuned ArgCSR (D=16) as the overall winner.
4. **Headroom lives in small and medium matrices.** Oracle vs cuSPARSE (FP64): < 100k nnz 1.13×, 100k–1M 1.09×,
   > 1M 1.02×. CSR-vector (small v) is the usual winner below 1M nnz.
5. **Thread-per-row formats break on power-law rows.** ASIC_100k: SELL 48 ms, CSR-scalar 7 ms, cuSPARSE 0.12 ms (a few rows
   with tens of thousands of nonzeros run on one thread). This drags SELL's geomean to 0.18–0.22× (median 0.52×). The
   selector must use max row length / row-length skew as features.
6. **Conversion cost (host)**, in SpMV calls: ArgCSR ~60, ELL ~40, SELL ~15, cuSPARSE preprocess 0.3. Break-even needs
   GPU converters (moved to Phase 3).

**Implications for Phase 3 (decide as a team)**
- With the oracle only 1.07–1.09× above cuSPARSE on average, G2's "beat always-cuSPARSE" leaves a narrow margin. Options:
  widen the candidates (ArgCSR-local with per-region D; CSR-vector with v chosen per row block) or re-scope G2 to "within 10%
  of the oracle and never slower than cuSPARSE on large matrices".
- Feature list confirmed by the data: nnz, rows, mean/max row length, row-length CV/skew, fraction of empty rows.

### 2026-09-23 (Wed) — Phase 1 complete: infrastructure + cuSPARSE baseline
**Built** (see [docs/phase1_guide.md](docs/phase1_guide.md) for the design)
- `rasp` library: [include/rasp/](include/rasp/) + [src/](src/). It has `HostCsr`/`DeviceCsr` (RAII, move-only),
  a Matrix Market reader (real/integer/pattern, general/symmetric/skew, duplicate summing, row sorting, CRLF)
  with an atomic binary cache (`<file>.csr.bin`), a timer with L2 flush + median/IQR, a CSV writer with an
  environment stamp (GPU, driver, CUDA, cuSPARSE, git commit/dirty, host, time), cuSPARSE SpMV (ALG1/ALG2,
  preprocess timed separately) and SpGEMM wrappers, and CPU references.
- Tests: 4 CTest suites, 28 tests, all passing (`ctest --test-dir build`). They include empty rows, a dense row
  among short rows, nnz = 0, rectangular, fp32/fp64, a hand-computed SpGEMM, and cache corruption.
- [bench/bench_spmv.cu](bench/bench_spmv.cu), [scripts/fetch_matrices.py](scripts/fetch_matrices.py),
  [scripts/gen_laplace.py](scripts/gen_laplace.py) (checked: A symmetric, diag 6, P 2×2×2 aggregates, R = Pᵀ),
  [scripts/plot_phase1.py](scripts/plot_phase1.py).
- Data: 60 matrices (50 random real square SuiteSparse, 1e4–5e6 nnz, seed 0 + 10 named in paper 2),
  967 MB, in `data/suitesparse/` with `manifest.csv`.

**Results** ([results/phase1_cusparse.csv](results/phase1_cusparse.csv), [figure](results/phase1/cusparse_bw.png)):
240 rows (60 matrices × FP32/FP64 × ALG1/ALG2), **0 correctness failures, 0 errors**. Timing noise is small:
median IQR/median = 1.9%, worst 11%.

| | median % of 187.8 GB/s | p25–p75 | max | fastest alg (wins) |
|---|---|---|---|---|
| FP64 ALG1 | 44.9 | 18.4–58.3 | 96.1 (af_shell1) | 47 / 60 |
| FP64 ALG2 | 43.2 | 13.0–58.3 | 94.9 | 13 / 60 |
| FP32 ALG1 | 43.4 | 14.9–53.9 | 82.4 | 50 / 60 |
| FP32 ALG2 | 42.5 | 11.7–56.0 | 81.5 | 10 / 60 |

- **Size dominates.** FP64 median %BW by nnz: < 100k → 15.6 (ALG1) / 11.6 (ALG2); 100k–1M → ~51; > 1M → ~73.
  Small matrices are launch/latency-bound: the whole SpMV takes a few µs. Phase 2 comparisons must be read per
  size band, and the selector needs `nnz` / `rows` as features.
- ALG1 (default) usually wins; ALG2 wins on some irregular circuit/power matrices (rajat23 57.1 vs 54.4,
  TSOPF_FS_b39_c7 59.1 vs 55.1, ASIC_100ks 58.0 vs 54.0).
- Worst: Sandia/adder_dcop_* (~8% FP64), which are small circuit matrices. Paper-2 targets for Phase 2:
  ncvxqp9 26.8%, t2d_q9 33.9%, TSOPF_RS_b162_c1 46.1%, ckt11752_dc_1 49.0%.
- `cusparseSpMV_preprocess` is cheap (median 0.015 ms ALG1, 0.011 ms ALG2).
- Reading af_shell1 (17.6 M nnz) from text takes 1.2 s; the binary cache makes later runs near-instant.

**Notes**
- Our nnz can exceed SuiteSparse's listed nnz (af_shell1: 17,588,875 vs 17,562,051) because the reader keeps
  explicitly stored zeros, and SuiteSparse's count excludes them. That's fine for SpMV; the manifest's nnz is metadata only.
- `ssgetpy.download()` sleeps 0.1 s per 4 KB (~40 KB/s), so `fetch_matrices.py` downloads archives itself. "Problem
  sequence" archives are large (case39: 70 MB for a 1 M-nnz matrix) because they bundle the whole sequence.

**Next: Phase 2** (SpMV kernels + paper-2 reproduction): CSR-scalar, CSR-vector, ELL, SELL-C-σ, ArgCSR.

### 2026-09-23 (Wed) — GitHub remote
- Repo pushed to [github.com/Vivianmatheww/HPC_Project](https://github.com/Vivianmatheww/HPC_Project) over SSH (key: `~/.ssh/id_ed25519`, "WSL HPC laptop").
- Next: invite Harsh and Karmanya as collaborators. Decide public vs private visibility (currently public).

### 2026-09-23 (Wed) — Phase 0 closed, Phase 1 started
- Phase 0 is complete apart from two carried-over items that don't block anything: pushing to a
  GitHub remote, and setup notes for Harsh's and Karmanya's machines.
- Wrote [docs/phase1_guide.md](docs/phase1_guide.md): shared headers to agree on first, 8 tasks with
  suggested owners, traps (24 MB L2 caching, MM symmetric expansion, duplicates, 32-bit index
  limit, binary cache), unit-test list, and the Phase 1 exit checklist.
- The platform owner starts Phase 1 now (CMake library layout + timer/CSV harness).

### 2026-09-23 (Wed) — Nsight Compute unblocked
- Enabled GPU performance counters for all users (NVIDIA Control Panel → Developer), then restarted WSL.
  `ncu` now profiles kernels in WSL2 (tested: SpeedOfLight and MemoryWorkloadAnalysis sections).
- **Team note:** `ncu` locks clocks to base and flushes caches by default. Use it for *metrics*
  (bandwidth %, hit rates, stalls), never for timings. Take timings from CUDA events or `nsys`.
  Add `--clock-control none` if you need boost-clock behaviour.

### 2026-09-23 (Wed) — Git + profilers
**Done**
- `git init` on branch `main`; first commit `39f1c87` (docs, setup script, CMake, UVM probe + results).
- Nsight Systems works in WSL2: a saxpy trace gives 1.17 ms/kernel (≈172 GB/s, consistent with the probe).

**Blocked**
- Nsight Compute: `ERR_NVGPUCTRPERM`. Fix on Windows: NVIDIA Control Panel → Desktop menu →
  *Enable Developer Settings* → Developer → *Manage GPU Performance Counters* → *Allow access to
  the GPU performance counters to all users*. Then `wsl --shutdown`.

**Next**
- Create a private GitHub repo, push `main`, and add Harsh and Karmanya as collaborators.

### 2026-09-23 (Wed) — WSL RAM + UVM probe
**Done**
- `.wslconfig` applied: 11 GB RAM (`free -g` shows 10 GiB), 8 GB swap. nvcc still on PATH after the restart.
- Added a minimal [CMakeLists.txt](CMakeLists.txt) (sm_89, C++17, Ninja) and
  [bench/platform/uvm_probe.cu](bench/platform/uvm_probe.cu). Build:
  `cmake -B build -G Ninja && cmake --build build && ./build/uvm_probe`. Raw data is in
  [results/platform/uvm_probe.csv](results/platform/uvm_probe.csv).

**Findings (single session; medians of 10, or of 3 for zero-copy)**

| 4 GiB array | device | managed (UVM) | pinned zero-copy |
|---|---|---|---|
| sequential read, warm | 187.8 GB/s | 13.2 GB/s | 13.4 GB/s |
| random gather | 1,803 M/s | 63 M/s | 64 M/s |

1. **UVM on WSL2 is not real UVM.** Managed memory stays in host RAM and every GPU access goes over
   PCIe. Warm reads run at PCIe speed; a "cold" read costs the same as one PCIe pass (no migration);
   the host reads data back at RAM speed after a kernel; and managed matches zero-copy at every size.
2. `cudaMemPrefetchAsync` and `cudaMemAdvise` return `invalid device ordinal`
   (`concurrentManagedAccess=0`).
3. **No oversubscription:** `cudaMallocManaged` of 1.25× VRAM (7.5 GiB) fails with `out of memory`.
4. **One copy engine confirmed:** H2D and D2H issued together give 13.3 GB/s in total, the same as
   one direction alone.
5. Host-resident data costs 14× for streaming and **8–28× for random gathers**. The penalty grows with
   array size (237 → 64 M gathers/s from 256 MiB to 4 GiB; a GPU TLB-reach effect is the likely cause,
   not yet verified). This lines up with paper 1, Table 3 (B in pinned memory: 7–29×).

**Implications for the project**
- On this platform the "UVM" baseline **is identical to the zero-copy baseline**. Report it that way
  and explain why. Paper 1's "UVM ≈ 30% of HBM when it fits" becomes ~7% here.
- Real page-migrating UVM (paper 1's behaviour) needs a native-Linux GPU. Raise this at the team
  meeting (cluster access question).
- Out-of-core SpGEMM **must** use explicit chunking; no automatic fallback exists. This strengthens
  Track B's motivation.
- Cost-model constants for Karmanya: BW_h2d = 13.4 GB/s, BW_d2h = 13.2 GB/s, no bidirectional
  overlap; device ≈ 188 GB/s read. Transfer:device bandwidth ratio ≈ 1:14.
- B (the randomly accessed operand) must be in device memory: zero-copy B costs up to ~28×.

**Next**
- Rerun the probe 2–3 times on different days to check variance (laptop thermals).
- Test that `nsys`/`ncu` can profile `uvm_probe` in WSL2.
- `git init`, `.gitignore`, private GitHub repo for the team.

### 2026-09-23 (Wed) — CUDA toolkit installed
**Done**
- Fixed a half-upgraded OpenJDK 21 (jdk 21.0.12 / jre 21.0.11 mismatch) with `apt-get install --fix-broken`. It had been blocking every apt install.
- Installed CUDA 13.3 via [scripts/setup_cuda_wsl.sh](scripts/setup_cuda_wsl.sh): nvcc V13.3.73, cuSPARSE 12.8.2 (12802), Nsight Systems 2026.1.3, Nsight Compute 2026.2.1.
- Sanity test (sm_89 kernel + `cusparseCreate`) passes.
- Device attributes: **1 async copy engine**; **`concurrentManagedAccess=0`** on WSL2.

**Implications**
- Track B cost model: with one copy engine, stage time ≈ max(H2D + D2H, compute), not max(H2D, compute, D2H). Update plan.md §2.3.
- UVM baseline: most likely it can only be measured when the problem fits in VRAM (paper 1's "fits in HBM" case). The oversubscribed case must come from the probe or a native-Linux GPU.

**Pending**
- `.wslconfig` (`free -g` still shows 7 GB), then `wsl --shutdown`.

### 2026-09-23 (Wed) — Environment setup
**Done**
- Team fixed at 3: _(your name)_, Harsh, Karmanya. Proposed roles added to [plan.md](plan.md) §5 and [README.md](README.md).
- Created `.venv/` with [requirements.txt](requirements.txt): CMake 4.4.3, Ninja 1.13, NumPy 2.2.6,
  SciPy 1.15.3, pandas, matplotlib, scikit-learn 1.7.2, ssgetpy.
- Wrote [scripts/setup_cuda_wsl.sh](scripts/setup_cuda_wsl.sh). It installs CUDA 13.3 (the newest in NVIDIA's
  wsl-ubuntu repo; the driver supports up to 13.4) and adds it to PATH.
- Windows host RAM is 15.2 GB. Plan: `.wslconfig` with `memory=11GB`, `swap=8GB`.

**Pending (needs the user)**
- Run `bash scripts/setup_cuda_wsl.sh` (needs a sudo password).
- Create `C:\Users\vivia\.wslconfig`, then run `wsl --shutdown`.

**Next**
- Verify `nvcc`, then write the UVM / WSL2 probe (managed memory, oversubscription, prefetch/advise, pinned zero-copy).

### 2026-09-23 (Wed) — Kick-off and planning
**Done**
- Read both papers in full and verified the details used in the plan:
  - arXiv:1804.00695 (Deveci et al., Sandia 2018): chunked out-of-core SpGEMM with kkmem on KNL
    and P100 + NVLink. Alg. 4 uses a fixed 75/25 fast-memory split and a copy-volume-only cost.
    Double buffering is explicitly listed as future work. B in pinned memory costs 7–29×. UVM
    reaches ≤ 30% of HBM.
  - arXiv:1203.5737 (Heller & Oberhuber, 2012): the Adaptive Row-grouped CSR SpMV format on a
    Tesla C2070 with 1,600 matrices. `desiredChunkSize` is a single global value (1 vs 32 gives up
    to a 6× difference, rajat23). The paper recommends testing multiple formats per matrix, but
    does not automate it.
- Defined the project scope and how we differ from both papers (see [project_context.md](project_context.md) §4).
- Wrote [README.md](README.md), [plan.md](plan.md), [project_context.md](project_context.md) and this file.
- Checked the dev machine: RTX 4050 Laptop GPU (6 GB, sm_89, PCIe 4.0 x8), driver 615.78 /
  CUDA 13.4 capable, Ryzen 7 7840HS, Ubuntu 22.04 on WSL2, **7 GB RAM in WSL**, 938 GB free disk, GCC 11.4, Python 3.10.

**Findings / blockers**
- `nvcc` and `cmake` are **not installed**. This blocks all code.
- 7 GB WSL RAM limits out-of-core experiments. The plan uses a fast-memory budget (as paper 1
  did), and we may raise the WSL limit.
- UVM on WSL2 is probably restricted. This must be verified before committing to it as a baseline.

**Next**
- Finish Phase 0 checklist in [plan.md](plan.md) §4 (toolchain, sanity benchmarks, UVM test, git init).
- Answer the open questions above at the first team meeting.
