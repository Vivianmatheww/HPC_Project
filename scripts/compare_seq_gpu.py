#!/usr/bin/env python3
"""Sequential (seq/rasp_seq, one CPU core) vs GPU (RASP) on the same machine, for the three checkpoint workloads.

Inputs (all from one machine, e.g. one scripts/kaggle_run.sh run):
  --seq DIR    seq/profile.sh output: spmv.csv, cg.csv, cg.txt, amg.csv, machine.txt
  --spmv CSV   bench_spmv on the same matrices (data/seq), fp64
  --cg CSV     cg_solver with --laplace 128
  --amg CSV    amg_setup with --n 160 (the size the sequential AMG uses)

    python scripts/compare_seq_gpu.py --seq results/kaggle/seq --spmv results/kaggle/spmv_gpu.csv \\
        --cg results/kaggle/cg_gpu.csv --amg results/kaggle/amg_gpu.csv

SpMV compares time per call (both sides multiply the same matrix), so the speedup does not depend on how either
side counts flops. GPU times include no host-device copies for SpMV and CG (the data stays on the GPU, as in a real
solver); the AMG times do, because amg_setup starts and ends in host memory.
"""
import argparse
import math
import re
from pathlib import Path

import pandas as pd


def geomean(xs):
    xs = [x for x in xs if x > 0 and math.isfinite(x)]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def short_gpu(name):
    return re.sub(r"^NVIDIA (GeForce )?|( Laptop)? GPU$", "", str(name))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", type=Path, required=True)
    ap.add_argument("--spmv", type=Path, required=True)
    ap.add_argument("--cg", type=Path)
    ap.add_argument("--amg", type=Path)
    args = ap.parse_args()

    machine = args.seq / "machine.txt"
    cpu = machine.read_text().strip() if machine.exists() else "unknown CPU"
    gpu = pd.read_csv(args.spmv)
    ok = gpu[(gpu.precision == "fp64") & (gpu.status == "ok")]
    print(f"Same machine. CPU: {cpu}\n             GPU: {short_gpu(gpu.gpu.iloc[0])} (CUDA {gpu.cuda_runtime.iloc[0]}, "
          f"cuSPARSE {gpu.cusparse.iloc[0]})")

    # ---- SpMV: sequential CSR vs cuSPARSE and vs the fastest of our GPU configurations
    seq = pd.read_csv(args.seq / "spmv.csv").query("kind == 'spmv' and name == 'CSR'").set_index("workload")
    n_cfg = ok.config.nunique()
    print(f"\nSpMV, fp64, time per call (sequential: mean of 50 warm calls; GPU: median, L2 flushed)")
    print(f"  {'matrix':<14} {'nnz':>11} {'seq CSR':>10} {'cuSPARSE':>10} {'speedup':>8}   "
          f"{'fastest of ' + str(n_cfg) + ' GPU configs':<34} {'speedup':>8}")
    sp_cus, sp_best = [], []
    for m, g in ok.groupby("matrix"):
        stem = m.split("/")[-1]
        if stem not in seq.index:
            continue
        s_ms = float(seq.loc[stem, "ms_per_call"])
        cus = g[g.config == "cusparse_alg1"]
        best = g.loc[g.median_ms.idxmin()]
        c_ms = float(cus.median_ms.iloc[0]) if len(cus) else float("nan")
        sp_cus.append(s_ms / c_ms)
        sp_best.append(s_ms / best.median_ms)
        print(f"  {stem:<14} {int(g.nnz.iloc[0]):>11,} {s_ms:8.3f}ms {c_ms:8.3f}ms {s_ms / c_ms:7.1f}x   "
              f"{best.config:<22} {best.median_ms:8.3f}ms {s_ms / best.median_ms:7.1f}x")
    print(f"  {'geomean':<14} {'':>11} {'':>10} {'':>10} {geomean(sp_cus):7.1f}x   {'':<34} {geomean(sp_best):7.1f}x")

    # ---- CG on the 128^3 Laplacian
    if args.cg and args.cg.exists():
        ph = pd.read_csv(args.seq / "cg.csv")
        solve_s = float(ph.query("kind == 'phase' and name != 'load / generate matrix'").seconds.sum())
        it = re.search(r"(\d+) iterations", (args.seq / "cg.txt").read_text())
        s_it = int(it.group(1)) if it else 0
        cg = pd.read_csv(args.cg)
        cg = cg[cg.matrix.str.endswith("laplace3d_n128") & (cg.status == "ok")]
        print(f"\nCG, Laplace 128^3 (2.1M unknowns), fp64, Jacobi preconditioner, until |r| <= 1e-8 |b|")
        print(f"  sequential            {solve_s:8.2f} s  {s_it:5d} iterations  {1e3 * solve_s / max(s_it, 1):7.2f} ms/iter")
        for _, r in cg.iterrows():
            per = r.total_ms / max(r.iters, 1)
            print(f"  GPU, {r.selector:<16} {r.total_ms / 1e3:8.2f} s  {int(r.iters):5d} iterations  {per:7.2f} ms/iter"
                  f"  -> {solve_s * 1e3 / r.total_ms:5.1f}x  (per iteration {1e3 * solve_s / max(s_it, 1) / per:5.1f}x)"
                  f"  [{r.config}]")

    # ---- AMG set-up: the SpGEMMs of every level
    if args.amg and args.amg.exists():
        ph = pd.read_csv(args.seq / "amg.csv").query("kind == 'phase'")
        sp_s = float(ph[ph.name.str.startswith("spgemm")].seconds.sum())
        amg = pd.read_csv(args.amg)
        print(f"\nAMG set-up, SpGEMMs R(AP) of every level (the sequential run uses a 160^3 grid)")
        print(f"  sequential, n=160     {sp_s:8.3f} s")
        for (n, method, budget), g in amg[amg.status == "ok"].groupby(["n", "method", "budget_bytes"]):
            tot = g[g.level == -1]
            if tot.empty:
                continue
            ms = float(tot.median_ms.iloc[0])
            vs = f"  -> {sp_s * 1e3 / ms:5.1f}x" if n == 160 else ""
            print(f"  GPU {method:<13} n={n:<4} budget {budget / 2**20:6.0f} MiB  {ms / 1e3:8.3f} s{vs}")
        print("  (GPU times include copying A, P, R in and every product out over PCIe)")


if __name__ == "__main__":
    main()
