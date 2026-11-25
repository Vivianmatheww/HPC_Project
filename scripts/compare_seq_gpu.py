#!/usr/bin/env python3
"""Sequential (seq/rasp_seq, one CPU core) vs GPU (RASP) on the same machine, for the three checkpoint workloads.

Inputs (all from one machine, e.g. one scripts/kaggle_run.sh run):
  --seq DIR    seq/profile.sh output: spmv.csv, cg.csv, cg.txt, amg.csv, machine.txt
  --spmv CSV   bench_spmv on the same matrices (data/seq), fp64
  --spmv-warm CSV  the same with --no-flush (L2 kept warm between calls, as inside a solver loop)
  --cg CSV     cg_solver with --laplace 128
  --amg CSV    amg_setup with --n 160 (the size the sequential AMG uses)
  --omp DIR    seq/scaling.sh output (t<threads>/ per thread count): the OpenMP version on the same CPU

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


def spmv_table(seq, csv, cache):
    """Sequential CSR vs cuSPARSE and vs the fastest of our GPU configurations, time per call."""
    ok = pd.read_csv(csv).query("precision == 'fp64' and status == 'ok'")
    n_cfg = ok.config.nunique()
    print(f"\nSpMV, fp64, time per call (sequential: mean of 50 warm calls; GPU: median, {cache})")
    print(f"  {'matrix':<14} {'nnz':>11} {'seq CSR':>10} {'cuSPARSE':>10} {'speedup':>8}   "
          f"{'fastest of ' + str(n_cfg) + ' GPU configs':<34} {'speedup':>8} {'vs cuSPARSE':>12}")
    sp_cus, sp_best, best_cus = [], [], []
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
        best_cus.append(c_ms / best.median_ms)
        print(f"  {stem:<14} {int(g.nnz.iloc[0]):>11,} {s_ms:8.3f}ms {c_ms:8.3f}ms {s_ms / c_ms:7.1f}x   "
              f"{best.config:<22} {best.median_ms:8.3f}ms {s_ms / best.median_ms:7.1f}x {c_ms / best.median_ms:11.2f}x")
    print(f"  {'geomean':<14} {'':>11} {'':>10} {'':>10} {geomean(sp_cus):7.1f}x   {'':<34} {geomean(sp_best):7.1f}x "
          f"{geomean(best_cus):11.2f}x")
    return geomean(sp_cus)


def seq_times(d):
    """Times of one seq/profile.sh-style run directory (sequential or one OpenMP thread count):
    SpMV CSR ms per call per matrix, the CG solve in seconds, and the AMG SpGEMMs in seconds."""
    spmv = pd.read_csv(d / "spmv.csv").query("kind == 'spmv' and name == 'CSR'").set_index("workload").ms_per_call
    cg = pd.read_csv(d / "cg.csv")
    cg_s = float(cg.query("kind == 'phase' and name != 'load / generate matrix'").seconds.sum())
    amg = pd.read_csv(d / "amg.csv").query("kind == 'phase'")
    amg_s = float(amg[amg.name.str.startswith("spgemm")].seconds.sum())
    return spmv.astype(float), cg_s, amg_s


def omp_table(seq_dir, omp_dir):
    """The OpenMP version (seq/scaling.sh: <omp_dir>/t<threads>/) against the sequential run, per thread count.
    Returns the best speedup over the sequential run and its thread count, per workload."""
    runs = sorted(((int(p.name[1:]), p) for p in omp_dir.glob("t[0-9]*") if (p / "spmv.csv").exists()))
    if not runs:
        return {}
    s_spmv, s_cg, s_amg = seq_times(seq_dir)
    print(f"\nOpenMP version, speedup over the sequential run (same CPU; SpMV: CSR, geomean over "
          f"{len(s_spmv)} matrices)")
    print(f"  {'threads':>7} {'SpMV':>8} {'CG':>8} {'AMG SpGEMM':>11}")
    best = {}
    for t, d in runs:
        o_spmv, o_cg, o_amg = seq_times(d)
        common = s_spmv.index.intersection(o_spmv.index)
        sp = {"spmv": geomean(list(s_spmv[common] / o_spmv[common])), "cg": s_cg / o_cg, "amg": s_amg / o_amg}
        print(f"  {t:>7} {sp['spmv']:7.2f}x {sp['cg']:7.2f}x {sp['amg']:10.2f}x")
        for k, v in sp.items():
            if v > best.get(k, (0, 0))[0]:
                best[k] = (v, t)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", type=Path, required=True)
    ap.add_argument("--spmv", type=Path, required=True)
    ap.add_argument("--spmv-warm", type=Path)
    ap.add_argument("--cg", type=Path)
    ap.add_argument("--amg", type=Path)
    ap.add_argument("--omp", type=Path, help="seq/scaling.sh output: t<threads>/ directories of the OpenMP version")
    args = ap.parse_args()
    summary = {}

    machine = args.seq / "machine.txt"
    cpu = machine.read_text().strip() if machine.exists() else "unknown CPU"
    gpu = pd.read_csv(args.spmv)
    print(f"Same machine. CPU: {cpu}\n             GPU: {short_gpu(gpu.gpu.iloc[0])} (CUDA {gpu.cuda_runtime.iloc[0]}, "
          f"cuSPARSE {gpu.cusparse.iloc[0]})")

    # ---- SpMV, with the GPU's L2 flushed before every call, then (if measured) kept warm
    seq = pd.read_csv(args.seq / "spmv.csv").query("kind == 'spmv' and name == 'CSR'").set_index("workload")
    summary["spmv"] = spmv_table(seq, args.spmv, "L2 flushed")
    if args.spmv_warm and args.spmv_warm.exists():
        spmv_table(seq, args.spmv_warm, "L2 warm, no flush")

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
            if r.selector == "always_cusparse":
                summary["cg"] = solve_s * 1e3 / r.total_ms
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
            if n == 160 and method == "incore":
                summary["amg"] = sp_s * 1e3 / ms
            print(f"  GPU {method:<13} n={n:<4} budget {budget / 2**20:6.0f} MiB  {ms / 1e3:8.3f} s{vs}")
        print("  (GPU times include copying A, P, R in and every product out over PCIe)")

    # ---- the OpenMP version, and one summary line per version
    best = omp_table(args.seq, args.omp) if args.omp and args.omp.exists() else {}
    keys = [("spmv", "SpMV"), ("cg", "CG"), ("amg", "AMG SpGEMM")]
    print("\nSummary: speedup over one core of the same CPU (SpMV: cuSPARSE, L2 flushed; CG: always_cusparse; "
          "AMG: incore, n=160)")
    print(f"  {'':<26}" + "".join(f"{label:>14}" for _, label in keys))
    if best:
        print(f"  {'OpenMP, best thread count':<26}" +
              "".join(f"{f'{best[k][0]:.1f}x ({best[k][1]} t)':>14}" if k in best else f"{'-':>14}" for k, _ in keys))
    print(f"  {'GPU':<26}" + "".join(f"{f'{summary[k]:.1f}x':>14}" if k in summary else f"{'-':>14}" for k, _ in keys))
    if best:
        print(f"  {'GPU over OpenMP':<26}" + "".join(
            f"{f'{summary[k] / best[k][0]:.1f}x':>14}" if k in summary and k in best else f"{'-':>14}" for k, _ in keys))


if __name__ == "__main__":
    main()
