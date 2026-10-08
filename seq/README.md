# RASP: sequential implementation and hotspot analysis

`rasp_seq.cpp` is the whole RASP workload on **one CPU thread**, in plain C++17 with no CUDA and no libraries. It is
the baseline the GPU version (the rest of this repository) is compared with, and the program we profiled to find
the hotspots.

| Workload | What it does |
|---|---|
| SpMV | read Matrix Market → build CSR → structural features → convert to ELL and SELL-32-256 → SpMV in every format (the best format is chosen by trying them all) |
| CG | Jacobi-preconditioned conjugate gradients: 1 SpMV, 3 dot products and 3 vector updates per iteration |
| AMG set-up | Galerkin products `A_{l+1} = R_l (A_l P_l)` on every level of a Laplace3D hierarchy; each SpGEMM is a flop estimate, a symbolic pass and a numeric pass (Gustavson) |

## Build and run

```bash
make -C seq                         # rasp_seq (g++ -O2)
./seq/rasp_seq --cg laplace:64 --amg 64                          # no data needed
./seq/rasp_seq --matrices data/seq/Rajat/rajat23.mtx --spmv-reps 50
```

## Hotspot analysis

```bash
python scripts/fetch_matrices.py --count 0 --out data/seq --names Rajat/rajat23 Sandia/ASIC_100k \
    ACUSIM/Pres_Poisson Norris/torso2 Williams/cop20k_A Schenk_AFE/af_shell1
bash seq/profile.sh                 # timed runs + gprof runs of the three workloads, CPU bandwidth (~2 min)
python scripts/analyze_seq.py --tex docs/slides/hotspots/generated --fig-dir docs/slides/hotspots/figures
```

- **gprof** (`make -C seq profile`, built with `-pg`) gives self time and call counts per function. GCC's function
  cloning (`.isra`, `.constprop`, `.part`, `.cold`) is switched off in that build. gprof ignores the cloned symbols
  and would otherwise charge their time to a neighbouring function.
- **Phase timers** inside the program record wall, user and system CPU time (`getrusage`). gprof samples only user
  time, and on WSL2 the kernel time spent zero-filling freshly allocated memory is significant.
- `bandwidth.cpp` measures the streaming bandwidth of one thread and of all threads. It shows the hotspots, which
  are memory-bound, cannot scale with CPU cores.

Results: [results/seq/](../results/seq/) (`summary.txt`, `hotspots.png`, the gprof flat profiles and call graphs).
Slides: [docs/slides/hotspots/hotspots.pdf](../docs/slides/hotspots/hotspots.pdf).
