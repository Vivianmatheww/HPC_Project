#!/usr/bin/env python3
"""Thread scaling of the OpenMP version (seq/rasp_omp.cpp), from seq/scaling.sh's output.

Inputs:  --omp DIR [DIR ...]   scaling.sh output: DIR/t<T>/{spmv,cg,amg}.csv, plus spmv_{static,dynamic,guided}.csv
                               at the largest T. Several DIRs are repeated sweeps: the fastest time of each metric is used.
         --seq DIR             seq/profile.sh output (spmv.csv, cg.csv, amg.csv): the sequential run on the same machine
         --bandwidth FILE      seq/bandwidth.txt (default: DIR/bandwidth.txt of --seq); its 1-thread -> all-threads ratio is the
                               expected ceiling of the memory-bound workloads (SpMV, CG)
Outputs: summary.txt, scaling.png|pdf   next to the first --omp directory (its parent)
         --tex DIR                      omp.tex: \\setstat values and tables for the report

    python scripts/analyze_omp.py --omp results/omp_karmanya/omp results/omp_karmanya/omp2 \\
        --seq results/omp_karmanya/seq [--tex docs/report/generated]
"""
import argparse
import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rasp_report import SLOTS, SURFACE, TEXT, TEXT2, TexWriter, geomean, save, style  # noqa: E402

WORKLOADS = [("spmv", "SpMV CSR (geomean of 6 matrices)"), ("cg", "CG solve (Laplace 128³)"),
             ("amg", "AMG SpGEMM (Laplace 160³)")]
SCHEDULES = [("static", "static"), ("dynamic", "dynamic,64"), ("guided", "guided")]


def phase_sum(df, keep):
    df = df[df.kind == "phase"]  # the total row has no name
    ph = df[df.name.map(keep)]
    return float(ph.seconds.sum()), float(ph.user.sum())  # wall and user CPU seconds of the timed phases


def load(d, schedule="spmv"):
    """Timed quantities of one run directory: per-matrix CSR ms, and (wall, user) seconds per workload."""
    s = pd.read_csv(d / f"{schedule}.csv")
    csr = s[(s.kind == "spmv") & (s.name == "CSR")].set_index("workload").ms_per_call.to_dict()
    out = {"spmv_ms": csr, "spmv": phase_sum(s, lambda n: n == "SpMV CSR")}
    if schedule == "spmv":
        out["cg"] = phase_sum(pd.read_csv(d / "cg.csv"), lambda n: n != "load / generate matrix")  # all CG phases
        out["amg"] = phase_sum(pd.read_csv(d / "amg.csv"), lambda n: n.startswith("spgemm"))
    return out


def best(runs):
    """Fastest of repeated runs: per-matrix ms, and the (wall, user) pair of the run with the least wall time."""
    out = {"spmv_ms": {m: min(r["spmv_ms"][m] for r in runs) for m in runs[0]["spmv_ms"]}}
    for w in ("spmv", "cg", "amg"):
        if w in runs[0]:
            out[w] = min((r[w] for r in runs), key=lambda p: p[0])
    out["spread"] = {w: max(r[w][0] for r in runs) / min(r[w][0] for r in runs) - 1 for w in ("spmv", "cg", "amg")
                     if w in runs[0]} if len(runs) > 1 else {}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--omp", type=Path, nargs="+", required=True)
    ap.add_argument("--seq", type=Path, required=True)
    ap.add_argument("--bandwidth", type=Path)
    ap.add_argument("--tex", type=Path)
    args = ap.parse_args()
    tex = TexWriter(args.tex / "omp.tex" if args.tex else None)
    out_dir = args.omp[0].resolve().parent

    seq = load(args.seq)
    threads = sorted({int(p.name[1:]) for d in args.omp for p in d.glob("t[0-9]*") if (p / "cg.csv").exists()})
    par = {T: best([load(d / f"t{T}") for d in args.omp if (d / f"t{T}" / "cg.csv").exists()]) for T in threads}
    tmax = threads[-1]
    seq_t = {"spmv": geomean(list(seq["spmv_ms"].values())), "cg": seq["cg"][0], "amg": seq["amg"][0]}

    def par_t(w, T):
        return geomean(list(par[T]["spmv_ms"].values())) if w == "spmv" else par[T][w][0]

    buf = io.StringIO()
    with redirect_stdout(buf):
        machine = (args.omp[0] / "machine.txt")
        print(f"machine: {machine.read_text().strip() if machine.exists() else '?'}")
        print(f"runs: {len(args.omp)} sweep(s) of threads {threads}, fastest time of each metric")
        for T in threads:
            if par[T]["spread"]:
                print(f"  run-to-run spread at {T} thread(s): " + ", ".join(f"{w} {100 * v:.1f}%" for w, v in par[T]["spread"].items()))
        unit = {"spmv": "ms", "cg": "s", "amg": "s"}

        print("\n== time (speedup vs the sequential run, vs T=1; parallel efficiency = speedup / T; busy = user CPU / wall)")
        head = f"{'':<34}{'seq':>9}" + "".join(f"{f'T={T}':>9}" for T in threads)
        rows = []
        for w, title in WORKLOADS:
            print(f"\n{title}  [{unit[w]}]")
            print(head)
            t = [par_t(w, T) for T in threads]
            sp = [seq_t[w] / x for x in t]
            print(f"{'time':<34}{seq_t[w]:9.3f}" + "".join(f"{x:9.3f}" for x in t))
            print(f"{'speedup vs sequential':<34}{'':>9}" + "".join(f"{x:8.2f}x" for x in sp))
            print(f"{'speedup vs T=1':<34}{'':>9}" + "".join(f"{t[0] / x:8.2f}x" for x in t))
            print(f"{'efficiency (vs seq)':<34}{'':>9}" + "".join(f"{100 * s / T:8.0f}%" for s, T in zip(sp, threads)))
            busy = [par[T][w][1] / par[T][w][0] for T in threads]
            print(f"{'busy cores (user/wall)':<34}{'':>9}" + "".join(f"{b:9.1f}" for b in busy))
            i = int(np.argmax(sp))
            tex.stat(f"omp-{w}-seq", seq_t[w], "{:.2f}")
            tex.stat(f"omp-{w}-best-speedup", sp[i], "{:.1f}")
            tex.stat(f"omp-{w}-best-threads", threads[i])
            tex.stat(f"omp-{w}-speedup-t{tmax}", sp[-1], "{:.1f}")
            rows.append([title.split(" (")[0].replace("³", "$^3$")] + [f"{x:.2f}$\\times$" for x in sp])
        tex.table("omp-scaling", ["workload"] + [f"{T}\\,thr" for T in threads], rows, "l" + "r" * len(threads))

        print("\n== SpMV CSR per matrix  [ms per SpMV; speedup vs sequential]")
        print(f"{'':<16}{'seq':>9}" + "".join(f"{f'T={T}':>9}" for T in threads))
        for m, ms in seq["spmv_ms"].items():
            print(f"{m:<16}{ms:9.3f}" + "".join(f"{par[T]['spmv_ms'][m]:9.3f}" for T in threads))
            print(f"{'  speedup':<16}{'':>9}" + "".join(f"{ms / par[T]['spmv_ms'][m]:8.2f}x" for T in threads))

        print(f"\n== SpMV CSR loop schedule at T={tmax}  [geomean ms; speedup vs sequential]")
        srows = []
        for key, label in SCHEDULES:
            runs = [load(d / f"t{tmax}", f"spmv_{key}") for d in args.omp if (d / f"t{tmax}" / f"spmv_{key}.csv").exists()]
            if not runs:
                continue
            ms = geomean(list(best(runs)["spmv_ms"].values()))
            print(f"  {label:<12}{ms:8.3f} ms  {seq_t['spmv'] / ms:5.2f}x")
            tex.stat(f"omp-spmv-sched-{key}", seq_t["spmv"] / ms, "{:.2f}")
            srows.append([label, f"{ms:.3f}", f"{seq_t['spmv'] / ms:.2f}$\\times$"])
        if srows:
            tex.table("omp-schedule", ["schedule", "ms", "speedup"], srows, "lrr")

        bw = args.bandwidth or args.seq / "bandwidth.txt"
        ceiling = None
        if bw.exists():
            vals = re.findall(r"(\d+) thread\(s\):\s+([\d.]+) GB/s", bw.read_text())
            if len(vals) == 2:
                ceiling = float(vals[1][1]) / float(vals[0][1])
                print(f"\nstreaming bandwidth: 1 thread {vals[0][1]} GB/s, {vals[1][0]} threads {vals[1][1]} GB/s "
                      f"-> expected ceiling of SpMV/CG ~{ceiling:.1f}x")
                tex.stat("omp-bw-ceiling", ceiling, "{:.1f}")
    text = buf.getvalue()
    print(text, end="")
    (out_dir / "summary.txt").write_text(text)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.plot(threads, threads, color=TEXT2, lw=1, ls=":", label="ideal")
    if ceiling:
        ax.axhline(ceiling, color=TEXT2, lw=1, ls="--", label=f"bandwidth ceiling {ceiling:.1f}x")
    for (w, title), c in zip(WORKLOADS, SLOTS):
        ax.plot(threads, [seq_t[w] / par_t(w, T) for T in threads], marker="o", ms=4, color=c, label=title.split(" (")[0])
    ax.set_xlabel("threads", color=TEXT2)
    ax.set_ylabel("speedup vs sequential", color=TEXT2)
    ax.set_xticks(threads)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    for f in save(fig, out_dir / "scaling", ("png", "pdf")):
        print(f"wrote {f}")
    plt.close(fig)
    if (t := tex.write()):
        print(f"wrote {t}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
