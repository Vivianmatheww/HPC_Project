# Project Plan — RASP: Runtime-Adaptive Sparse Matrix Optimization for GPUs

> Schedule assumes a **12-week semester, Week 1 = Mon 2026-09-28, final week = 2026-12-14 → 12-18**.
> If the real deadline differs, shift the dates but keep the phase order. Update [progress.md](progress.md) every week.

---

## 1. Goals and success criteria

| ID | Goal | Measured by | Target |
|----|------|-------------|--------|
| G1 | Reproduce paper 2 (ArgCSR) on modern hardware | ArgCSR vs cuSPARSE on the same matrix classes the paper reports (rajat*, TSOPF, AFE, …) | Same qualitative winners and losers |
| G2 | Runtime SpMV selector | Geomean of T_selected / T_oracle over held-out matrices | ≤ 1.10 (within 10% of oracle); beats "always cuSPARSE" and "always ArgCSR-1" |
| G3 | Selection is cheap | Feature extraction + inference time, in units of one CSR SpMV | ≤ 2 SpMVs |
| G4 | Conversion-aware choice | End-to-end time for N ∈ {1, 10, 100, 1000} calls vs any fixed format | Never worse than cuSPARSE CSR by > 5% at N = 1 |
| G5 | Reproduce paper 1's placement study on a PCIe GPU | Our version of their Table 3 (A/B/C each in pinned memory) | Table + explanation |
| G6 | Overlapped, adaptive out-of-core SpGEMM | GFLOP/s at 1×–4× the fast-memory budget vs paper-1 Alg. 4 (no overlap), pinned zero-copy, UVM (if it works on WSL) | ≥ 1.3× over Alg. 4 when problem > budget |
| G7 | End-to-end demos | CG solve (SpMV) and AMG triple product R·A·P (SpGEMM) | Working, timed, in report |

**Minimum viable project** (if time runs out): G1 + G2 + G3 + G5 + chunked SpGEMM with overlap
(G6 without the adaptive re-planning). Everything else is stretch.

---

## 2. System architecture

```
                         ┌──────────────────────── RASP runtime ─────────────────────────┐
  CSR matrix on GPU ───▶ │ Feature extractor (1 kernel + CUB reductions)                  │
                         │   n, nnz, row-len mean/max/std/CV, empty rows, ELL fill,       │
                         │   SELL-32 padding, column span/bandwidth, row-len skew          │
                         │            │                                                    │
                         │            ▼                                                    │
  hints: expected_calls, │ Selector  ─ rules (v1) → decision tree (v2) → online probe (v3) │
  mem budget, precision  │            │  cost = T_convert(f) + N · T_spmv(f)               │
                         │            ▼                                                    │
                         │ SpMV plan: {format, params, kernel}  ──▶ run() many times      │
                         │                                                                 │
  A, B (host or device)─▶│ SpGEMM planner                                                  │
                         │   sizes + flop estimate + calibrated PCIe / kernel rates        │
                         │   → {in-core | zero-copy A/C | chunked(AC-resident|B-resident)} │
                         │   → partitions + pipeline depth; re-plan after each chunk       │
                         │            ▼                                                    │
                         │ Chunk pipeline: H2D(k+1) ‖ compute(k) ‖ D2H(k−1) on streams     │
                         └─────────────────────────────────────────────────────────────────┘
```

### 2.1 SpMV candidate set

| Kernel | Notes | Tunable parameters |
|--------|-------|--------------------|
| CSR-scalar | one thread per row; baseline for very short rows | — |
| CSR-vector | sub-warp of *v* threads per row | v ∈ {2,4,8,16,32}, chosen from mean row length |
| ELL | column-major padded; only if fill ratio ≤ threshold | — (memory cap: skip if padding > 3× nnz) |
| SELL-C-σ | Kreutzer et al. 2014 | C ∈ {8,32}, σ ∈ {1, 32, 256, n} |
| **ArgCSR** | Reimplemented from paper 2, Listing 2 | `desiredChunkSize` ∈ {1,2,4,8,16,32}, blockDim ∈ {64,128,256}; **plus our per-region variant** |
| cuSPARSE | `cusparseSpMV`, `CUSPARSE_SPMV_CSR_ALG1` and `ALG2` | — |

**Our ArgCSR extension ("ArgCSR-local"):** instead of one global `desiredChunkSize`, choose it
per row region from local regularity (the coefficient of variation of row lengths in that window).
This targets paper 2's own finding: regular regions want 32, irregular regions want 1.

### 2.2 Selector (three versions, each shippable)

1. **v1 Rules:** hand-written thresholds on CV, max/mean, ELL fill, etc. Derived from Phase 2 data.
2. **v2 Decision tree:** scikit-learn `DecisionTreeClassifier` (depth ≤ 6), or a regressor per
   candidate predicting GFLOP/s. Trained on the offline benchmark CSV and **exported as generated
   C++ if/else code** (`src/selector/tree_generated.hpp`), so there is no runtime ML dependency.
   Validation is k-fold **grouped by SuiteSparse matrix group**, so matrix families don't leak
   between train and test.
3. **v3 Online probe (stretch):** if `expected_calls` is large, time the top-2/3 predicted
   candidates for a few calls each and keep the fastest.

**Cost model:** choose f = argmin [ T_feat + T_convert(f) + N · T_spmv(f) ]. T_convert is
predicted from nnz and n (linear fit per format), and T_spmv comes from the model. With N = 1
the answer is almost always "stay in CSR", which is the point.

### 2.3 SpGEMM design

- **In-core engine:** cuSPARSE generic SpGEMM (`cusparseSpGEMM_workEstimation` / `_compute` / `_copy`).
  *Stretch:* our own hash-based row-wise kernel (kkmem-style, with a shared-memory accumulator)
  that can skip A's columns outside a B row range, as paper 1 does.
- **Sizing C before computing it:** per-row upper bound = Σ_{k∈A(i,:)} nnz(B(k,:)) (the "flop
  count"), computed by one cheap kernel. Row partitions come from a binary search on its prefix
  sum, as in paper 1's `BinarySearch`.
- **Execution modes chosen by the planner:**
  1. *In-core* — everything fits the budget.
  2. *Zero-copy for A and/or C* — B fits, and A/C are small or rarely accessed (paper 1, Table 3).
  3. *Chunked, B-resident* (`chunk2`) — stream row blocks of A and C through a double buffer.
     This is the common multigrid A×P case.
  4. *Chunked, 2-D* — neither fits. Stream B blocks per resident A/C block and accumulate
     partial C blocks (cuSPARSE SpGEAM, or our kernel's column filter).
- **Pipeline:** two (or three) device buffers, pinned host staging, a copy stream and a compute
  stream synchronised with events. **Measured `asyncEngineCount = 1`** on the RTX 4050 Laptop, so H2D and D2H share one engine and only copy ‖ compute overlaps.
- **Throughput cost model (our replacement for Alg. 4's 75/25 split):**
  stage time ≈ max( bytes_in / BW_h2d + bytes_out / BW_d2h, flops / R_kernel ) with one copy engine
  (max of all three terms with two engines), or the plain sum without overlap. BW is measured at startup with 256 MB pinned transfers. R_kernel starts from
  a calibration SpGEMM, **then is updated after every chunk, and the remaining rows are
  re-partitioned** if the measured rate differs by more than 20% (runtime adaptivity).
- **Baselines:** (a) paper-1 Alg. 4 reimplemented without overlap, (b) everything in pinned
  zero-copy memory, (c) UVM `cudaMallocManaged`, which **on WSL2 behaves exactly like (b)**: host-resident, no migration, no prefetch, no oversubscription (measured, see progress.md 2026-09-23). A true page-migrating UVM baseline needs native Linux. (d) in-core when it fits (upper bound).

### 2.4 Datasets

| Set | Source | Use |
|-----|--------|-----|
| SuiteSparse subset | `ssgetpy`: real, square, 10k ≤ n, nnz ≤ 50M. Start with ~300, grow toward ~1,000–1,600 (paper 2 used 1,600) | SpMV training and evaluation |
| Paper-2 named matrices | rajat23, TSOPF_*, case39, Schenk_AFE/af_shell*, GHS_indef/*, norris/torso2, t2d_q* | G1 reproduction |
| Laplace3D 7-pt + aggregation R, P | Generated by `scripts/gen_laplace.py`, any size | SpGEMM R×A, A×P (paper 1's own workload) |
| Graphs | SNAP (com-LiveJournal, com-Orkut, …) for L×L triangle counting | SpGEMM with irregular structure |
| SPD matrices | SuiteSparse SPD subset | CG end-to-end demo |

---

## 3. Methodology (applies to every experiment)

- Time kernels with CUDA events. Use 5 warm-up runs and report the **median of ≥ 20** with IQR.
  Time conversion and feature extraction separately.
- Check correctness on every run against cuSPARSE (relative L2 error ≤ 1e-12 for FP64,
  1e-5 for FP32). SpGEMM is checked on small cases against a SciPy reference.
- Report GFLOP/s (2·nnz per SpMV) **and** effective bandwidth as a % of the measured device
  bandwidth (from a STREAM-like copy kernel, not the datasheet number).
- Record the environment in every CSV: driver, CUDA version, GPU clocks (via `nvidia-smi`), and git hash.
- Laptop hygiene: AC power, Windows "Best performance" mode, identical conditions, and randomised
  order of configurations to spread thermal drift.
- Use `results/` for raw CSVs and `scripts/plot_*.py` to make figures. Never hand-edit numbers.

---

## 4. Schedule

### Phase 0 — Setup (Week 0: Sep 23 – Sep 27)
- [x] Install CUDA Toolkit 13.x for WSL-Ubuntu (**toolkit only**: `cuda-toolkit-13-x`, never `cuda` or a driver package), CMake ≥ 3.24 (`pip install cmake`), Python deps
- [x] `deviceQuery` and `bandwidthTest` sanity checks. Record `asyncEngineCount`, PCIe H2D/D2H GB/s, device bandwidth
- [x] Test whether `cudaMallocManaged` oversubscription works under WSL2 (decides the UVM baseline)
- [x] Check that Nsight Systems / Nsight Compute can profile from WSL2
- [x] Check Windows host RAM. If ≥ 16 GB, set `.wslconfig` `memory=` higher
- [x] `git init`, `.gitignore` (data/, build/), CMake skeleton that compiles a hello-kernel
- [x] Decide team roles (Track A = SpMV → Harsh, Track B = SpGEMM → Karmanya). *Carried over: deadline, meeting cadence, GitHub remote, teammate setup notes*

### Phase 1 — Infrastructure (Weeks 1–2: Sep 28 – Oct 9)
> Detailed task split, interfaces and traps: [docs/phase1_guide.md](docs/phase1_guide.md)
- [ ] Matrix Market reader (real, integer, pattern, symmetric expansion) → host CSR → device CSR
- [ ] Timer / benchmark harness (warm-up, median, CSV writer, environment stamp)
- [ ] cuSPARSE SpMV + SpGEMM wrappers (these are the baselines and the correctness reference)
- [ ] `fetch_matrices.py` (ssgetpy with filters) and `gen_laplace.py` (A, R, P at chosen size)
- [ ] Unit tests (tiny hand-made matrices, empty rows, a single dense row)
- **Deliverable:** `bench_spmv` runs cuSPARSE on 50 matrices and writes a CSV

### Phase 2 — SpMV kernels and paper-2 reproduction (Weeks 3–4: Oct 12 – Oct 23)
- [ ] CSR-scalar, CSR-vector (v = 2…32), ELL, SELL-C-σ kernels with GPU converters
- [ ] ArgCSR converter + kernel from paper 2 (Listing 2), with parameter sweeps over `desiredChunkSize` and blockDim
- [ ] Reproduce the paper-2 observations: the 1 vs 32 sensitivity (AFE vs rajat23) and circuit vs power-flow winners
- [ ] Exhaustive sweep over the dataset → `results/spmv_oracle.csv` (the training data)
- **Deliverable:** oracle table + "no format wins everywhere" figure (performance profile, as in paper 2 Fig. 4)

### Phase 3 — Runtime SpMV selector (Weeks 5–6: Oct 26 – Nov 6)
- [ ] GPU feature-extraction kernel. Measure its cost in SpMV units (G3)
- [ ] Selector v1 (rules) → v2 (decision tree, exported to C++)
- [ ] Conversion-time model + break-even logic. `make_spmv_plan()` API
- [ ] ArgCSR-local (per-region `desiredChunkSize`) evaluation
- [ ] Evaluation: slowdown vs oracle, confusion matrix, N-sweep (G2, G4)
- **Deliverable:** selector within ~10% of oracle on held-out matrices. **Midpoint check-in / demo.**

### Phase 4 — SpGEMM track (Weeks 7–9: Nov 9 – Nov 27)
- [ ] Week 7: placement study (our Table 3): A, B, C each in pinned zero-copy vs device, on Laplace R×A and A×P (G5). Flop-count / row-size estimator
- [ ] Week 8: chunked driver (chunk1, chunk2, 2-D) + paper-1 Alg. 4 as a baseline. Verify against in-core on sizes that fit
- [ ] Week 9: double-buffered pipeline, runtime bandwidth calibration, throughput cost model, re-planning after each chunk (G6)
- **Deliverable:** GFLOP/s vs problem-size / budget plot with all baselines (paper 1, Figs. 12–13 style)

### Phase 5 — Integration and end-to-end apps (Week 10: Nov 30 – Dec 4)
- [ ] CG solver using `make_spmv_plan` (the iteration count is the natural `expected_calls`)
- [ ] AMG-setup triple product R·A·P through `rasp::spgemm` with a budget
- [ ] Nsight profiles of 2–3 representative kernels (achieved bandwidth, occupancy, stalls)

### Phase 6 — Evaluation and write-up (Weeks 11–12: Dec 7 – Dec 18)
- [ ] Freeze code. Rerun all experiments from scripts on a clean build
- [ ] Final figures: performance profiles, oracle-gap CDF, break-even curves, SpGEMM scaling, overlap timeline (Nsight Systems screenshot)
- [ ] Report: intro, background (both papers), design, evaluation, related work (SMAT, Sedaghati, Zhao, Benatia), limitations, conclusion
- [ ] Slides + live demo (`plan.report()` output on a circuit matrix vs a FEM matrix)

---

## 5. Work split (for a 2–3 person team)

| Track | Owner (proposed) | Phases |
|-------|-------|--------|
| A — SpMV formats + selector | **Harsh** | 2, 3, CG app |
| B — SpGEMM out-of-core | **Karmanya** | 4, AMG app |
| Platform + infrastructure — toolchain, UVM/WSL study, harness, datasets, placement study (G5), integration, plots | **_(your name)_** | 0, 1, 4 (Week 7 study), 5, 6 |
| Report + slides | everyone (each writes their track's sections) | 6 |

Phase 1 is joint work: everyone builds a piece of the harness so all three know the codebase.
After that Tracks A and B run in parallel, and the platform owner keeps both unblocked (datasets,
benchmark runs, CI-style correctness checks). Karmanya can start reading kkmem and cuSPARSE SpGEMM
in Weeks 3–6 and prototype the flop-count estimator early, since Track B's hard part starts in Week 7.

---

## 6. Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| UVM oversubscription not supported on WSL2 | High | Treat UVM as "report where it runs". Pinned zero-copy is the main slow-memory baseline. Optionally rerun on a native Linux / cluster GPU |
| Only 7 GB host RAM | Certain | Fast-memory budget emulation (paper 1 did the same). Raise the `.wslconfig` limit if possible |
| Laptop thermal / clock variance | High | Median of ≥ 20, IQR, interleaved runs, AC power. Report variance |
| ArgCSR reproduction differs from 2012 numbers | Medium | Compare *relative* rankings and trends, not absolute GFLOPS. Document the differences |
| Selector overfits matrix families | Medium | Grouped cross-validation. Keep the tree shallow. Report per-family accuracy |
| Partial-C accumulation in 2-D SpGEMM is slow | Medium | Prioritise the B-resident 1-D path (covers A×P and most cases). 2-D is stretch |
| SuiteSparse download size / time | Medium | nnz filter, download incrementally, cache in `data/`, keep a manifest CSV |
| Scope creep | High | The MVP in §1 is the contract. Stretch items are marked |

---

## 7. Stretch ideas (only after the MVP)
- Own hash-based SpGEMM kernel with a column-range filter (removes the SpGEAM accumulation)
- Mixed-precision SpMV (FP32 values, FP64 accumulate) as another candidate
- Online probing selector (v3) and a bandit-style re-selection for long-running solvers
- Port to a data-centre GPU (A100/H100) to test how well the selector transfers across architectures
