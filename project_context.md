# Project Context

Background for everyone on the team. It covers what the two source papers actually did (checked
against the PDFs), where they stop, and exactly what this project does differently.

---

## 1. The problem in one paragraph

Sparse kernels are **memory-bound and irregular**. The fastest way to run SpMV or SpGEMM on a GPU
depends on the matrix: how evenly nonzeros are spread across rows, how long the longest row is,
how the column indices are laid out, and how big the matrices are compared with GPU memory. No
single format or algorithm wins everywhere. Both source papers show this but still ship a
**static** choice: one format, one tuning parameter, or one fixed memory split for every input.
This project makes those choices **at runtime, per matrix, with cheap measurements**.

---

## 2. Paper 1 — SpGEMM on multilevel memory (arXiv:1804.00695)

**Deveci, Hammond, Wolf, Rajamanickam — Sandia National Laboratories, April 2018 (SAND2018-3428 R).**

### What they did
- **Baseline kernel `kkmem`** (KokkosKernels, the default SpGEMM in Trilinos). It is row-wise and
  hierarchical: each thread (or team) computes one row of C, with vector lanes working inside the
  row. It has a *symbolic* phase that counts nnz per row of C and a *numeric* phase that computes
  the values. It uses a two-level **hashmap accumulator**: level 1 in GPU shared memory, level 2 in
  global memory when level 1 overflows.
- **Hardware:** Intel KNL (MCDRAM + DDR4) and an **NVIDIA P100 connected to a POWER8 over NVLink v1**.
- **Workloads:** multigrid triple products (R×A and A×P) for Laplace3D, BigStar2D, Brick3D and
  Elasticity (A has 7, 13, 27 and 81 nnz/row), weak-scaled so A is 1–32 GB, plus triangle counting
  (twitter-2010, uk-2005, graph500 scale 25).
- **GPU memory analysis (Table 3):** they placed each of A, B, C in host-pinned memory in turn.
  Putting **B** in pinned memory costs **7–29×**. Putting A or C there hurts only when they are
  large or accessed often. UVM reaches as little as **30% of HBM** performance even when the data
  fits, and falls to pinned-memory speed when it doesn't.
- **Chunking algorithm for GPUs:**
  - `kkmemGPUchunk1` (Alg. 2): keep a row block of A and C in fast memory and stream B's row
    blocks. Copy cost = |A| + |C| + |B|·‖P_AC‖.
  - `kkmemGPUchunk2` (Alg. 3): keep a B row block in fast memory and stream A and C.
    Copy cost = |B| + |A|·‖P_B‖ + |C|·(‖P_B‖−1).
  - A is not physically split by column: the kernel skips A's columns that fall outside the current
    B row range.
  - **Alg. 4 (partition heuristic):** reserve **75% / 25%** of fast memory for the resident and the
    streamed operand, find row ranges by binary search, and pick the variant with the lower
    *copy-volume* cost.
- **Results:** "Chunk16" (16 GB budget) reaches 3.1–13.3× over host-pinned memory on R×A. When
  everything fits, chunking is at most 5.7× slower than pure HBM because of copies, and UVM beats
  chunking in that case.

### Where it stops (our opportunities)
1. **No copy/compute overlap.** They write: *"We do not have implementation using these strategies
   [double/triple buffering] but plan to provide an additional analysis…"*. That is exactly the
   cost that made chunking up to 5.7× slower than HBM.
2. **The cost model counts bytes only.** Alg. 4 minimises copy volume. It doesn't model transfer
   bandwidth, kernel throughput, or whether a stage is limited by copying or by computing.
3. **The split is fixed at 75/25** and chosen once before running. Nothing adapts after the first
   chunk shows how fast things actually are.
4. **The hardware is NVLink (~40 GB/s per direction on P100).** Consumer and laptop GPUs use PCIe,
   which has a much worse transfer-to-compute ratio, so the tradeoffs are different and
   still unmeasured.
5. **KNL is obsolete.** Only the GPU half of the paper is relevant today.

---

## 3. Paper 2 — Adaptive Row-grouped CSR (arXiv:1203.5737)

**Heller, Oberhuber — Czech Technical University in Prague, March 2012.** Code in TNL (Template Numerical Library).

### What they did
- **Format:** rows are split into **groups**. Each group is processed by one CUDA block
  (blockDim = number of *chunks* per group). Long rows are split into several equal-size
  **chunks**, so a heavy row gets several threads. Chunks are stored column-major, as in
  Sliced ELLPACK, so memory reads coalesce. Padding is marked with column = −1 and the loop exits early.
- **Conversion from CSR:** greedily close a group when its nnz exceeds
  `desiredChunkSize × blockDim` or it has more than `blockDim` rows. Inside each group, start
  with one thread per row, then repeatedly give a spare thread to the row with the fullest
  chunk. The resulting `chunkSize` is per group. Finally, prefix-sum the thread-to-row mapping.
- **Kernel (Listing 2):** load group info into shared memory, have each thread compute a
  partial dot product over its chunk, then reduce the partial sums per row in shared memory.
- **Setup:** Tesla C2070 (Fermi, 144 GB/s), **double precision**, **~1,600 square matrices**
  (UF/SuiteSparse and the Bai et al. collection), 128 threads/block.
- **Results:** ArgCSR beats CPU CSR on 1,168 of 1,600 matrices (cuSPARSE: 994, RgCSR: 907,
  Hybrid: 726). It is 10× faster than cuSPARSE on circuit matrices (rajat, GHS_indef, IBM EDA),
  but cuSPARSE is ~4× faster on power-flow matrices (TSOPF, case39).

### Where it stops (our opportunities)
1. **`desiredChunkSize` is one global constant.** They show it matters a lot: Schenk AFE runs at
   18 GFLOPS with 32 and 11 with 1, while rajat23 runs at **5.1 GFLOPS with 1 and 0.81 with 32
   (6×)**. They then fix it to 1 for everything because it "seems to be more robust".
2. **No automatic format choice.** They conclude: *"If high performance is the top priority,
   one should test more formats and choose the best one."* That is the core of this project.
3. **Conversion cost is mentioned but never measured.** There is no guidance on when conversion pays off.
4. **2012 baselines and hardware.** Fermi GPUs, the legacy cuSPARSE CSR kernel, no SELL-C-σ, no
   load-balanced CSR (merge-path style, cuSPARSE `CSR_ALG2`).

---

## 4. What this project does differently

| Aspect | Paper 1 (Deveci '18) | Paper 2 (Heller '12) | **RASP (us)** |
|---|---|---|---|
| Kernel | SpGEMM | SpMV | **Both**, under one runtime |
| What adapts | Chunk partition sizes, before running | Chunks per row inside a group | **Format + parameters (SpMV); placement + partitioning + pipeline depth (SpGEMM)** |
| Decision basis | Bytes-copied heuristic, 75/25 split | Fixed global `desiredChunkSize` | **GPU-computed features + cost model calibrated with runtime microbenchmarks** |
| When decided | Once, before running | Offline, by the authors | **At runtime, per matrix; SpGEMM re-plans after each chunk** |
| Conversion cost | n/a | Acknowledged, not measured | **Explicit break-even: pick f minimising T_conv(f) + N·T_spmv(f)** |
| Copy/compute overlap | None (future work) | n/a | **Double-buffered CUDA streams + pinned staging** |
| Hardware | P100 + NVLink, KNL | Tesla C2070 (Fermi) | **RTX 4050 Laptop (Ada, sm_89) over PCIe 4.0 x8, 6 GB** |
| Baselines | kkmem, UVM, pinned | cuSPARSE (2012), HYB, RgCSR | **cuSPARSE generic API (SpMV ALG1/ALG2, SpGEMM), SELL-C-σ, UVM, zero-copy, paper-1 Alg. 4 as a baseline** |
| Evaluation | GFLOP/s | Speedup vs CPU | **GFLOP/s and % of peak bandwidth, slowdown vs oracle, selection overhead, break-even N, end-to-end CG and AMG setup** |

### The contribution in three sentences (for the report and slides)
1. A **lightweight runtime selector** for SpMV. It extracts structural features on the GPU in
   well under the cost of a few SpMVs, predicts the best format *and its parameters* (including
   ArgCSR's `desiredChunkSize`), and weighs conversion cost against the expected number of calls.
2. An **overlapped, self-calibrating out-of-core SpGEMM**. It extends Deveci et al.'s chunking
   with double buffering and a throughput-based cost model that measures PCIe and kernel rates at
   runtime and re-plans partitions as it runs.
3. A **modern re-evaluation** of both papers' claims on a PCIe-attached Ada consumer GPU.

### Honest novelty note
Automatic sparse format selection with ML is an established research area. SMAT (Li et al.,
PLDI 2013), Sedaghati et al. (ICS 2015), Zhao et al. (PPoPP 2018, CNN-based) and Benatia et al.
(ICPP 2016, SVM) all study it. Our novelty is **relative to the two assigned papers**: tuning
ArgCSR's parameters per matrix, a conversion-aware cost model, and bringing adaptivity to paper 1's
out-of-core SpGEMM. Cite this related work in the report instead of claiming format selection itself is new.

---

## 5. Constraints that shape the design

| Constraint | Consequence |
|---|---|
| 6 GB VRAM | Real out-of-core SpGEMM is possible. Most SuiteSparse matrices fit for SpMV. |
| **WSL RAM: raised from 7 GB to 11 GB (+ 8 GB swap)** | Host copies of A, B, C > ~5 GB are impossible. We use a **fast-memory budget** (e.g. 0.5–2 GB) to emulate "doesn't fit", as the paper did with Chunk8/Chunk16. We can raise the WSL limit via `%UserProfile%\.wslconfig` (`memory=`) if the Windows host has more RAM. |
| WSL2 | **Measured:** managed memory (UVM) stays in host RAM and the GPU reads it over PCIe, identical to pinned zero-copy (13.2 vs 13.4 GB/s; device 187.8). Prefetch/Advise are unsupported and allocations > VRAM fail. The UVM baseline therefore equals the zero-copy baseline here; real UVM needs native Linux. |
| Laptop GPU | Clocks and thermals vary. Plug in, use a high-performance power plan, warm up, report the median of ≥ 20 runs with IQR, and interleave configurations. |
| FP64 is 1/64 rate on GeForce | Irrelevant for memory-bound SpMV/SpGEMM, but we report both FP32 and FP64 for SpMV, with FP64 as the primary for comparison with paper 2. |
| PCIe 4.0 x8: **measured 13.4 GB/s** H2D vs **187.8 GB/s** device read | A ~14:1 ratio (the P100 over NVLink was ~18:1 against 732 GB/s). Transfer dominates, so overlap matters. |
| Number of async copy engines | **Measured 1.** H2D and D2H serialise (concurrent aggregate 13.3 GB/s), so the pipeline overlaps copy with compute only. |

---

## 6. Glossary

- **SpMV** — sparse matrix × dense vector. **SpGEMM** — sparse × sparse matrix.
- **CSR** — row pointers + column indices + values. **ELL** — every row padded to max length and
  stored column-major. **SELL-C-σ** — ELL per slice of C rows after sorting rows by length inside
  windows of σ rows (Kreutzer et al., SISC 2014).
- **ArgCSR** — Adaptive Row-grouped CSR (paper 2). **RgCSR** — its non-adaptive predecessor.
- **Oracle** — the best format for a matrix, found by exhaustively timing all candidates. The
  selector is scored against it.
- **Break-even N** — the number of SpMV calls after which converting to a faster format pays off.
- **Fast-memory budget** — an artificial cap on the device memory RASP may use, to emulate
  problems larger than the GPU.
- **Zero-copy** — the GPU reads mapped, pinned host memory directly over PCIe.
