#!/usr/bin/env python3
"""Hotspots of the sequential implementation (seq/rasp_seq.cpp), from seq/profile.sh's output.

Inputs:  results/seq/gprof_<workload>/flat.txt   gprof flat profiles (self time per function)
         results/seq/<workload>.csv              per-phase wall / user / system time (optimised build)
         results/seq/bandwidth.txt               streaming bandwidth of this CPU, one thread and all threads
         results/phase2_spmv.csv                 GPU SpMV on the RTX 4050 (Phase 2), for a first comparison
Outputs: results/seq/hotspots.png|pdf            top functions per workload (share of profiled CPU time)
         results/seq/summary.txt
         --tex DIR                               seq.tex: \\setstat values and tables for the slides, plus the data
                                                 files the slides plot with pgfplots (gprof_<w>.csv, speedup.csv,
                                                 cg_traffic.csv)

    python scripts/analyze_seq.py --tex docs/slides/hotspots/generated --fig-dir docs/slides/hotspots/figures
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

from rasp_report import SURFACE, TEXT, TEXT2, TexWriter, geomean, save, style, tex_escape  # noqa: E402

WORKLOADS = [("spmv", "SpMV: 6 matrices x 3 formats"),
             ("cg", "CG solver: Laplace 128\u00b3"),
             ("amg", "AMG set-up: Laplace 160\u00b3")]
# The slides' accent colour; the largest hotspot of each workload is drawn in it, the others muted.
ACCENT, MUTED = "#05668d", "#a9c7d6"
MATRICES = ["Rajat/rajat23", "Sandia/ASIC_100k", "ACUSIM/Pres_Poisson", "Norris/torso2", "Williams/cop20k_A",
            "Schenk_AFE/af_shell1"]
# Labels for the slides' charts (a third of a slide wide); the summary and seq.tex keep the function names.
DISPLAY = {"spmv_csr": "SpMV CSR", "spmv_ell": "SpMV ELL", "spmv_sell": "SpMV SELL", "coo_to_csr": "build CSR",
           "read_matrix_market": "read .mtx", "dot": "dot product", "axpy": "axpy", "jacobi": "Jacobi",
           "xpby": "p-update", "spgemm_numeric": "numeric pass", "spgemm_symbolic": "symbolic pass",
           "spgemm_flops": "flop estimate", "laplace3d": "make Laplace", "std::sort (per row)": "sort rows",
           "std::vector fill": "vector fill", "transpose": "transpose"}
ROW = re.compile(r"^\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(?:(\d+)\s+([\d.]+)\s+([\d.]+)\s+)?(.+?)\s*$")


def short(name):
    name = name.replace("(anonymous namespace)::", "")
    if "std::" in name.split("(")[0]:
        if "sort" in name or "heap" in name:
            return "std::sort (per row)"
        if "fill_assign" in name or "vector(" in name:
            return "std::vector fill"
        return re.sub(r"<.*", "", name)
    return re.sub(r"\(.*", "", name)


def flat_profile(path):
    rows, started = [], False
    for line in path.read_text().splitlines():
        if line.strip().startswith("time"):
            started = True
            continue
        if not started or not line.strip():
            continue
        m = ROW.match(line)
        if m:
            rows.append({"pct": float(m.group(1)), "self_s": float(m.group(3)),
                         "calls": int(m.group(4)) if m.group(4) else 0, "name": short(m.group(7))})
    df = pd.DataFrame(rows)
    return df.groupby("name", as_index=False).agg(pct=("pct", "sum"), self_s=("self_s", "sum"),
                                                  calls=("calls", "sum")).sort_values("pct", ascending=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("results/seq"))
    ap.add_argument("--gpu", type=Path, default=Path("results/phase2_spmv.csv"))
    ap.add_argument("--tex", type=Path)
    ap.add_argument("--fig-dir", type=Path, help="also write hotspots.pdf here (the slides' figure directory)")
    args = ap.parse_args()
    tex = TexWriter(args.tex / "seq.tex" if args.tex else None)
    buf = io.StringIO()
    profiles = {w: flat_profile(args.dir / f"gprof_{w}" / "flat.txt") for w, _ in WORKLOADS}

    with redirect_stdout(buf):
        for w, title in WORKLOADS:
            p = profiles[w]
            print(f"== {title}  (gprof: {p.self_s.sum():.2f} s of samples)")
            for _, r in p.head(7).iterrows():
                print(f"  {r['pct']:5.1f}%  {r.self_s:6.2f} s  {r.calls:>10d} calls  {r['name']}")
            for _, r in p.head(6).iterrows():
                tex.stat(f"seq-{w}-{r['name']}", r["pct"], "{:.0f}")
            def compact(c):
                return "--" if not c else f"{c / 1e6:.1f}M" if c >= 1e6 else f"{c / 1e3:.0f}k" if c >= 1e4 else str(c)
            rows = [[tex_escape(r["name"]), f"{r['pct']:.0f}\\%", compact(r.calls)] for _, r in p.head(5).iterrows()]
            tex.table(f"seq-{w}", ["function", "time", "calls"], rows, "lrr")
            ph = pd.read_csv(args.dir / f"{w}.csv")
            ph = ph[ph.kind == "phase"]
            total = float(pd.read_csv(args.dir / f"{w}.csv").query("kind == 'total'").seconds.iloc[0])
            tex.stat(f"seq-{w}-total-s", total, "{:.1f}")
            print(f"  phases (optimised build, wall {total:.2f} s):")
            for _, r in ph.iterrows():
                print(f"    {r['name']:<34} {r.seconds:7.3f} s {100 * r.seconds / total:5.1f}%  "
                      f"(user {r.user:6.3f}, system {r.sys:6.3f})")
            print()

        # Phase-level figures the slides quote.
        cg = pd.read_csv(args.dir / "cg.csv").query("kind == 'phase'").set_index("name")
        total_cg = cg.seconds.sum()
        tex.stat("seq-cg-spmv-pct", 100 * cg.loc["SpMV (q = A p)", "seconds"] / total_cg, "{:.0f}")
        vec = cg.loc[["dot products", "vector updates (axpy)", "preconditioner (Jacobi)"], "seconds"].sum()
        tex.stat("seq-cg-vector-pct", 100 * vec / total_cg, "{:.0f}")
        dot_calls = cg.loc["dot products", "calls"]
        n = 128 ** 3
        dot_gbs = 2 * 8 * n * dot_calls / cg.loc["dot products", "seconds"] / 1e9  # two vectors of doubles per dot
        tex.stat("seq-dot-gbs", dot_gbs, "{:.1f}")
        print(f"one core streams {dot_gbs:.1f} GB/s in the dot products (two vectors of 8-byte values per call)")

        # Minimum bytes one CG iteration moves (model) vs where its time goes (measured): time follows bytes.
        m = re.search(r"(\d+) rows, nnz (\d+)", (args.dir / "cg.txt").read_text())
        rows_cg, nnz_cg = int(m.group(1)), int(m.group(2))
        spmv_bytes = nnz_cg * 12 + (rows_cg + 1) * 4 + rows_cg * 8 * 2  # values + columns, row pointers, x and y
        vec_bytes = 18 * 8 * rows_cg  # 3 dots x 2 vectors, 2 axpy x 3, Jacobi x 3, p-update x 3 (8-byte entries)
        t_spmv = cg.loc["SpMV (q = A p)", "seconds"]
        cg_traffic = [("SpMV", 100 * spmv_bytes / (spmv_bytes + vec_bytes), 100 * t_spmv / (t_spmv + vec)),
                      ("vector ops", 100 * vec_bytes / (spmv_bytes + vec_bytes), 100 * vec / (t_spmv + vec))]
        for name, b, t in cg_traffic:
            key = "spmv" if name == "SpMV" else "vec"
            tex.stat(f"seq-cg-bytes-{key}-pct", b, "{:.0f}")
            tex.stat(f"seq-cg-time-{key}-pct", t, "{:.0f}")
        tex.stat("seq-cg-iter-mb", (spmv_bytes + vec_bytes) / 1e6, "{:.0f}")
        print(f"CG iteration: SpMV moves {spmv_bytes / 1e6:.0f} MB, vector ops {vec_bytes / 1e6:.0f} MB "
              f"({cg_traffic[0][1]:.0f}% / {cg_traffic[1][1]:.0f}% of bytes; {cg_traffic[0][2]:.0f}% / "
              f"{cg_traffic[1][2]:.0f}% of kernel time)")
        amg = pd.read_csv(args.dir / "amg.csv").query("kind == 'phase'").set_index("name")
        sp = amg.loc[[i for i in amg.index if i.startswith("spgemm")]]
        tex.stat("seq-amg-spgemm-pct", 100 * sp.seconds.sum() / amg.seconds.sum(), "{:.0f}")
        tex.stat("seq-amg-numeric-sys-pct", 100 * amg.loc["spgemm: numeric", "sys"] / amg.loc["spgemm: numeric", "seconds"],
                 "{:.0f}")
        spm = pd.read_csv(args.dir / "spmv.csv")
        ph = spm.query("kind == 'phase'").set_index("name")
        io_s = ph.loc[["read Matrix Market (parse)", "build CSR (sort, merge)"], "seconds"].sum()
        kern = ph.loc[[i for i in ph.index if i.startswith("SpMV")], "seconds"].sum()
        tex.stat("seq-spmv-io-pct", 100 * io_s / ph.seconds.sum(), "{:.0f}")
        tex.stat("seq-spmv-kernel-pct", 100 * kern / ph.seconds.sum(), "{:.0f}")

        # First comparison with the GPU (different machines and methods: see the caption in the slides).
        seq = spm.query("kind == 'spmv' and name == 'CSR'").set_index("workload")
        tex.stat("seq-csr-gflops-min", seq.gflops.min(), "{:.1f}")
        tex.stat("seq-csr-gflops-max", seq.gflops.max(), "{:.1f}")
        gpu = pd.read_csv(args.gpu) if args.gpu.exists() else pd.DataFrame()
        rows, ratios = [], []
        print("\nSpMV fp64, sequential CSR (one i7-1255U core, warm, mean of 50) vs cuSPARSE ALG1 (RTX 4050, cold L2):")
        for m in MATRICES:
            stem = m.split("/")[1]
            if stem not in seq.index or gpu.empty:
                continue
            g = gpu[(gpu.matrix == m) & (gpu.precision == "fp64") & (gpu.config == "cusparse_alg1") & (gpu.status == "ok")]
            if g.empty:
                continue
            s_gf, g_gf = float(seq.loc[stem, "gflops"]), float(g.gflops.iloc[0])
            ratios.append(g_gf / s_gf)
            print(f"  {stem:<14} nnz {int(seq.loc[stem, 'nnz']):>9,}  seq {s_gf:5.2f}  GPU {g_gf:6.2f} GFLOP/s  -> {g_gf / s_gf:5.1f}x")
            rows.append([tex_escape(stem), f"{int(seq.loc[stem, 'nnz']) / 1e6:.2f}M", f"{s_gf:.2f}", f"{g_gf:.1f}",
                         f"{g_gf / s_gf:.1f}$\\times$"])
        bw = args.dir / "bandwidth.txt"
        if bw.exists():
            vals = re.findall(r"(\d+) thread\(s\):\s+([\d.]+) GB/s", bw.read_text())
            if len(vals) == 2:
                (_, one), (threads, allt) = vals
                tex.stat("seq-bw-1", float(one), "{:.1f}")
                tex.stat("seq-bw-all", float(allt), "{:.1f}")
                tex.stat("seq-bw-threads", threads)
                tex.stat("seq-bw-scaling", float(allt) / float(one), "{:.1f}")
                tex.stat("seq-spmv-bound-gflops", 2 * float(one) / 12, "{:.2f}")  # 2 flops per 12 bytes
                print(f"\nstreaming bandwidth: 1 thread {one} GB/s, {threads} threads {allt} GB/s "
                      f"({float(allt) / float(one):.1f}x)")
        if ratios:
            print(f"  geomean {geomean(ratios):.1f}x")
            tex.stat("seq-gpu-speedup-geomean", geomean(ratios), "{:.1f}")
            tex.stat("seq-gpu-speedup-min", min(ratios), "{:.1f}")
            tex.stat("seq-gpu-speedup-max", max(ratios), "{:.1f}")
            tex.table("seq-vs-gpu", ["matrix", "nnz", "seq", "GPU", "ratio"], rows, "lrrrr")
            speedups = sorted(zip([r[0] for r in rows], ratios), key=lambda x: x[1])

    text = buf.getvalue()
    print(text, end="")
    (args.dir / "summary.txt").write_text(text)

    # Data files for the slides' native (pgfplots) charts, next to seq.tex.
    if args.tex:
        args.tex.mkdir(parents=True, exist_ok=True)
        for w, _ in WORKLOADS:
            p = profiles[w].head(5).iloc[::-1]  # ascending: pgfplots draws the first row at the bottom
            lines = ["label,top,rest"]  # the largest share (and any tie) in "top", drawn in the accent colour
            for _, r in p.iterrows():
                hot = p["pct"].max() - r["pct"] < 1
                label = tex_escape(DISPLAY.get(r["name"], r["name"]))
                lines.append(f"{{{label}}},{r['pct'] if hot else 0:.1f},{0 if hot else r['pct']:.1f}")
            (args.tex / f"gprof_{w}.csv").write_text("\n".join(lines) + "\n")
        if ratios:
            (args.tex / "speedup.csv").write_text("label,ratio\n" + "".join(
                f"{{{name}}},{v:.2f}\n" for name, v in speedups))
        (args.tex / "cg_traffic.csv").write_text("label,bytes,time\n" + "".join(
            f"{{{name}}},{b:.1f},{t:.1f}\n" for name, b, t in cg_traffic))

    # Figure: top functions per workload, share of the profiled CPU time.
    # Drawn at about the size it is shown on a 16:9 slide, so the labels stay readable there.
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.75))
    fig.patch.set_facecolor(SURFACE)
    for ax, (w, title) in zip(axes, WORKLOADS):
        style(ax)
        p = profiles[w].head(5).iloc[::-1]
        top = p["pct"].max()
        colors = [ACCENT if top - v < 1.0 else MUTED for v in p["pct"]]  # ties at the top are all highlighted
        bars = ax.barh(p["name"], p["pct"], color=colors, height=0.66)
        for b, v in zip(bars, p["pct"]):
            ax.text(b.get_width() + 1.5, b.get_y() + b.get_height() / 2, f"{v:.0f}%", va="center", fontsize=8.5,
                    color=TEXT)
        ax.set_xlim(0, 62)
        ax.set_title(title, fontsize=9.5, color=TEXT, loc="left", fontweight="bold")
        ax.tick_params(axis="y", labelsize=8.5, length=0)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_xticks([0, 20, 40, 60])
        ax.set_xlabel("% of CPU time (gprof)", fontsize=8, color=TEXT2)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    for f in save(fig, args.dir / "hotspots", ("png", "pdf")) + (save(fig, args.fig_dir / "hotspots", ("pdf",))
                                                                 if args.fig_dir else []):
        print(f"wrote {f}")
    plt.close(fig)
    if (t := tex.write()):
        print(f"wrote {t}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
