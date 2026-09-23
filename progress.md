# Progress Log

How to use this file: update the **status table** whenever a phase changes state, and add a
**dated entry** at the top of the log for each work session or weekly meeting. Say what was done,
what we learned (with numbers), what is blocked, and what comes next. Link to CSVs and figures in `results/`.

Legend: ⬜ not started · 🟨 in progress · ✅ done · ⛔ blocked

## Status

| Phase | Weeks | Dates | Status | Notes |
|-------|-------|-------|--------|-------|
| 0 — Setup | 0 | Sep 23 – Sep 27 | 🟨 | Toolchain, 11 GB WSL, CMake, UVM probe, git done; ncu counter permission + GitHub remote pending |
| 1 — Infrastructure | 1–2 | Sep 28 – Oct 9 | ⬜ | |
| 2 — SpMV kernels + paper-2 reproduction | 3–4 | Oct 12 – Oct 23 | ⬜ | |
| 3 — Runtime SpMV selector | 5–6 | Oct 26 – Nov 6 | ⬜ | Midpoint demo |
| 4 — Out-of-core adaptive SpGEMM | 7–9 | Nov 9 – Nov 27 | ⬜ | |
| 5 — Integration + apps | 10 | Nov 30 – Dec 4 | ⬜ | |
| 6 — Evaluation + report | 11–12 | Dec 7 – Dec 18 | ⬜ | |

## Goal tracker

| Goal | Target | Current | Status |
|------|--------|---------|--------|
| G1 ArgCSR reproduction | same qualitative trends | — | ⬜ |
| G2 Selector vs oracle | geomean slowdown ≤ 1.10 | — | ⬜ |
| G3 Selection overhead | ≤ 2 SpMV | — | ⬜ |
| G4 Break-even aware | ≤ 5% loss at N = 1 | — | ⬜ |
| G5 Placement study (Table 3) | table | Memory-level probe done (host-side reads 14–28× slower); SpGEMM-level table in Week 7 | 🟨 |
| G6 Overlapped SpGEMM | ≥ 1.3× over Alg. 4 | — | ⬜ |
| G7 CG + AMG demos | working | — | ⬜ |

## Measured platform numbers (fill in during Phase 0)

| Quantity | Value | How measured |
|----------|-------|--------------|
| Device memory bandwidth (achieved) | **187.8 GB/s** seq read (98% of 192), **172 GB/s** copy r+w | `uvm_probe` |
| H2D pinned bandwidth | **13.4 GB/s** (pageable 11.9) | `uvm_probe`, 256 MiB, median of 10 |
| D2H pinned bandwidth | **13.2 GB/s** (pageable 12.5); H2D+D2H concurrently = 13.3 GB/s total | same |
| `asyncEngineCount` | **1** (H2D and D2H serialise; only copy ‖ compute overlaps) | `cudaDeviceGetAttribute` |
| UVM on WSL2 | **Managed memory stays in host RAM and is read over PCIe (≡ zero-copy).** Warm reads 13.2 GB/s vs 187.8 device. Prefetch/Advise → `invalid device ordinal`. 1.25× VRAM alloc → `out of memory` | `uvm_probe` |
| Nsight Systems in WSL2 | **Works** (nsys 2026.1.3: CUDA trace + `nsys stats`) | saxpy test kernel |
| Nsight Compute in WSL2 | **Blocked: `ERR_NVGPUCTRPERM`.** Enable GPU performance counters for all users in the Windows NVIDIA Control Panel | `ncu --section SpeedOfLight` |
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
