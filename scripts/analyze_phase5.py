#!/usr/bin/env python3
"""Phase 5 evaluation: end-to-end applications (G7) and kernel profiles.

Inputs:  results/phase5/cg.csv        cg_solver over the SPD set (selectors in shuffled order, 5 repetitions)
         results/phase5/amg.csv       amg_setup (Galerkin triple products under a budget)
         results/phase5/ncu/*.csv     Nsight Compute raw-page exports (scripts/profile_kernels.sh)
Outputs: results/phase5/cg_speedup.png  CG time-to-solution vs always-cuSPARSE, per matrix
         results/phase5/summary.txt     CG, AMG and profile tables
         --tex DIR                      phase5.tex (report values and tables)

    python scripts/analyze_phase5.py [--fig-dir docs/report/figures --fig-format pdf] [--tex docs/report/generated]
"""
import argparse
import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rasp_report import (METHOD_LABEL, SLOTS, SURFACE, TEXT, TEXT2, TexWriter, geomean, save, style,  # noqa: E402
                         tex_escape)

SELECTORS = ["tree", "tree_exact", "tree_nobe"]
SELECTOR_LABEL = {"tree": "tree, hint = iteration cap", "tree_exact": "tree, hint = exact iterations",
                  "tree_nobe": "tree without break-even"}
SELECTOR_COLOR = {s: SLOTS[i] for i, s in enumerate(SELECTORS)}


def cg(path, out, fig_dir, formats, tex):
    if not path.exists():
        print(f"(no {path})")
        return
    df = pd.read_csv(path)
    print(f"== CG (G7, SpMV side): {df.matrix.nunique()} matrices, {len(df)} rows")
    not_spd = df[df.status == "not_spd"]
    if len(not_spd):
        print(f"  not SPD in floating point (skipped): {', '.join(sorted(not_spd.matrix.unique()))}")
    ok = df[df.status == "ok"].copy()
    fails = ok[ok.check == "FAIL"]
    print(f"  solves: {len(ok)}; converged {int((ok.converged == 1).sum())}, hit the cap {int((ok.converged == 0).sum())}, "
          f"FAILED residual checks {len(fails)}")
    key = ["matrix", "precision"]
    base = ok[ok.selector == "always_cusparse"][key + ["total_ms", "iters", "spmv_share"]].rename(
        columns={"total_ms": "base_ms", "iters": "base_iters", "spmv_share": "base_spmv_share"})
    j = ok.merge(base, on=key)
    j["speedup"] = j["base_ms"] / j["total_ms"]
    print(f"  SpMV share of the solve with cuSPARSE: median {100 * base.base_spmv_share.median():.0f}% "
          f"(the rest: vector updates, reductions, two host synchronisations per iteration)")
    print(f"  iterations: median {base.base_iters.median():.0f}, max {base.base_iters.max():.0f}")
    rows = []
    for s in SELECTORS:
        v = j[j.selector == s]
        if v.empty:
            continue
        g, worst, best = geomean(v.speedup), v.speedup.min(), v.speedup.max()
        feat = 100 * v.used_features.mean()
        changed = 100 * (v.config != "cusparse_alg1").mean()
        print(f"  {SELECTOR_LABEL[s]:<32} speedup vs cuSPARSE: geomean {g:.3f} [{worst:.2f}, {best:.2f}]; "
              f"features used {feat:.0f}%, kernel other than cuSPARSE {changed:.0f}%; >5% slower on "
              f"{int((v.speedup < 0.95).sum())} of {len(v)}")
        rows.append([tex_escape(SELECTOR_LABEL[s]), f"{g:.3f}", f"{worst:.2f}", f"{best:.2f}", f"{feat:.0f}\\%",
                     f"{int((v.speedup < 0.95).sum())} / {len(v)}"])
        tex.stat(f"p5-cg-{s}-geomean", g, "{:.3f}")
        tex.stat(f"p5-cg-{s}-worst", worst)
        tex.stat(f"p5-cg-{s}-best", best)
    tex.table("p5-cg", ["selector", "geomean", "worst", "best", "features", "$>$5\\% slower"], rows)
    tex.stat("p5-cg-matrices", int(j.matrix.nunique()))
    tex.stat("p5-cg-spmv-share", float(100 * base.base_spmv_share.median()), "{:.0f}")
    tex.stat("p5-cg-failures", int(len(fails)))

    # Per-matrix speedups, sorted by nnz.
    order = j[j.selector == "tree"].sort_values("nnz")["matrix"].tolist()
    if not order:
        return
    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    xs = {m: i for i, m in enumerate(order)}
    for k, s in enumerate(SELECTORS):
        v = j[(j.selector == s) & j.matrix.isin(xs)]
        if v.empty:
            continue
        ax.scatter([xs[m] + (k - 1) * 0.18 for m in v.matrix], v.speedup, s=40, color=SELECTOR_COLOR[s],
                   label=SELECTOR_LABEL[s], zorder=3, edgecolors=SURFACE, linewidths=0.8)
    ax.axhline(1.0, color=TEXT2, linewidth=1, zorder=2)
    ax.set_yscale("log", base=2)
    lo, hi = j.speedup.min(), j.speedup.max()
    ticks = [t for t in (0.25, 0.5, 0.67, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2, 3, 4) if lo / 1.1 <= t <= hi * 1.1]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}x" for t in ticks])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([m.split("/")[-1] for m in order], rotation=70, fontsize=7, color=TEXT2)
    ax.set_ylabel("time-to-solution speedup\nvs always cuSPARSE", fontsize=9, color=TEXT2)
    ax.set_title("Jacobi-PCG: plan + solve time, matrices sorted by nnz", fontsize=10, color=TEXT, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    for p in save(fig, out / "cg_speedup", ("png",)) + (save(fig, fig_dir / "p5_cg", formats) if fig_dir else []):
        print(f"  wrote {p}")
    plt.close(fig)


def amg(path, tex):
    if not path.exists():
        print(f"(no {path})")
        return
    df = pd.read_csv(path)
    print(f"\n== AMG setup (G7, SpGEMM side): {len(df)} rows")
    tot = df[df.level == -1]
    bad = df[(df.status == "ok") & (df.check == "FAIL")]
    print(f"  failed checks (coarse operators differ between methods, or Galerkin row sums off): {len(bad)}")
    for n, g in tot.groupby("n"):
        a4 = g[(g.method == "alg4") & (g.status == "ok")]["median_ms"]
        a4 = float(a4.iloc[0]) if len(a4) else np.nan
        print(f"  n = {n}, budget {g[g.method != 'incore'].budget_bytes.max() / 2**20:.0f} MiB:")
        rows = []
        for _, r in g.iterrows():
            if r.status != "ok":
                print(f"    {r.method:<13} {r.status}: {r.error}")
                rows.append([tex_escape(METHOD_LABEL.get(r.method, r.method)), "--", tex_escape(r.status)])
                continue
            sp = a4 / r.median_ms if np.isfinite(a4) else np.nan
            print(f"    {r.method:<13} {r.median_ms:9.1f} ms   {sp:5.2f}x vs Alg. 4")
            rows.append([tex_escape(METHOD_LABEL.get(r.method, r.method)), f"{r.median_ms:.0f}", f"{sp:.2f}$\\times$"])
            tex.stat(f"p5-amg-{n}-{r.method}-ms", float(r.median_ms), "{:.0f}")
            tex.stat(f"p5-amg-{n}-{r.method}-speedup", float(sp))
        tex.table(f"p5-amg-{n}", ["method", "setup (ms)", "vs.\\ Alg.~4"], rows)
        lv = df[(df.n == n) & (df.level >= 0) & df.method.isin(["alg4", "rasp"])]
        if len(lv):
            piv = lv.pivot_table(index=["level", "grid", "product"], columns="method", values="median_ms")
            print("    per product (ms):")
            for idx, r in piv.iterrows():
                print(f"      level {int(idx[0])} grid {int(idx[1]):>4} {idx[2]:<7} " +
                      "  ".join(f"{m} {r[m]:8.2f}" for m in piv.columns if np.isfinite(r[m])))


# Nsight Compute raw pages: header row of metric names, then a row of units, then one row per profiled kernel.
# Older ncu versions spell the time units nsecond/usecond/msecond, newer ones ns/us/ms.
SCALE_NS = {"nsecond": 1, "usecond": 1e3, "msecond": 1e6, "second": 1e9, "ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9}
SCALE_B = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12, "Kibyte": 1024, "Mibyte": 2**20,
           "Gibyte": 2**30}
L2_SECTOR_B = 32
# DRAM counters on the RTX 4050 Laptop GPU (ncu 2025.x): every dram__/gpu__dram_throughput metric has two instances,
# one of which always reads 0 (min = 0, avg = max / 2), so the .avg percentage ncu reports is half the real one; and
# dram__bytes.sum.per_second is ~1/3 of the traffic implied by the timing, as if only one of the three 32-bit memory
# partitions of the 96-bit bus were counted. The L2 miss count does agree with the timing (cuSPARSE on af_shell1:
# 6.86 M sectors x 32 B in 1.18 ms = 185 GB/s, against ~178 GB/s from the bench and 192 GB/s peak), and the L2 hit
# rate (14%) rules out ncu replaying with a warm L2. So the table gives DRAM % from the busy instance (.max) and
# DRAM GB/s from L2 misses; ncu's own dram__bytes figure is printed in summary.txt for reference only.


def ncu_table(ncu_dir, tex):
    files = sorted(ncu_dir.glob("*.csv")) if ncu_dir.exists() else []
    if not files:
        print(f"\n(no Nsight Compute exports in {ncu_dir})")
        return
    print("\n== Nsight Compute (metrics only; ncu runs at base clocks)")

    def col(df, *patterns):
        for p in patterns:
            for c in df.columns:
                if re.fullmatch(p, c):
                    return c
        return None

    rows = []
    for f in files:
        raw = pd.read_csv(f, low_memory=False)
        if raw.empty:
            continue
        units, df = raw.iloc[0], raw.iloc[1:]
        if df.empty:
            continue
        r = df.iloc[0]

        def num(c):
            try:
                return float(str(r[c]).replace(",", "")) if c else np.nan
            except ValueError:
                return np.nan

        name = str(r[col(df, "Kernel Name", "Function Name") or df.columns[0]])
        dur_c = col(df, r"gpu__time_duration\.sum")
        dur_ns = num(dur_c) * SCALE_NS.get(str(units[dur_c]), 1) if dur_c else np.nan
        dram = num(col(df, r"gpu__dram_throughput\.max\.pct_of_peak_sustained_elapsed",
                       r"dram__throughput\.max\.pct_of_peak_sustained_elapsed"))
        mem = num(col(df, r"gpu__compute_memory_throughput\.max\.pct_of_peak_sustained_elapsed"))
        occ = num(col(df, r"sm__warps_active\.avg\.pct_of_peak_sustained_active"))
        theo = num(col(df, r"sm__maximum_warps_per_active_cycle_pct"))
        bw_c = col(df, r"dram__bytes\.sum\.per_second")  # bytes/s; unreliable here, see above
        ncu_gbs = num(bw_c) * SCALE_B.get(str(units[bw_c]).removesuffix("/s"), 1) / 1e9 if bw_c else np.nan
        miss = num(col(df, r"lts__t_sectors_lookup_miss\.sum"))
        hit = num(col(df, r"lts__t_sector_hit_rate\.pct"))
        stalls = {}
        for c in df.columns:
            m = re.fullmatch(r"smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio", c) or \
                re.fullmatch(r"smsp__average_warp_latency_issue_stalled_(\w+)\.ratio", c)
            if m and m.group(1) not in ("not_issued", "selected"):
                v = num(c)
                if np.isfinite(v):
                    stalls[m.group(1)] = v
        top = sorted(stalls.items(), key=lambda kv: -kv[1])[:3]
        short = re.sub(r"\(.*", "", name).split("<")[0].split("::")[-1][:40]
        gbs = miss * L2_SECTOR_B / dur_ns if np.isfinite(miss) and dur_ns > 0 else np.nan  # bytes/ns = GB/s
        print(f"  {f.stem:<28} {short:<30} {dur_ns / 1e3:9.1f} us  DRAM {dram:5.1f}%  mem {mem:5.1f}%  "
              f"occupancy {occ:5.1f}% (theoretical {theo:5.1f}%)  DRAM {gbs:6.1f} GB/s from L2 misses "
              f"(ncu dram__bytes {ncu_gbs:5.1f}), L2 hit {hit:4.1f}%  stalls: "
              + ", ".join(f"{k} {v:.1f}" for k, v in top))
        rows.append([tex_escape(f.stem), tex_escape(short), f"{dur_ns / 1e3:.0f}", f"{dram:.0f}", f"{gbs:.0f}",
                     f"{occ:.0f}", f"{theo:.0f}", tex_escape(", ".join(k.replace("_", " ") for k, _ in top))])
        key = f.stem.replace("_", "-")
        tex.stat(f"p5-ncu-{key}-us", dur_ns / 1e3, "{:.0f}")
        tex.stat(f"p5-ncu-{key}-dram", dram, "{:.0f}")
        tex.stat(f"p5-ncu-{key}-gbs", gbs, "{:.0f}")
        tex.stat(f"p5-ncu-{key}-l2hit", hit, "{:.0f}")
        tex.stat(f"p5-ncu-{key}-occ", occ, "{:.0f}")
    tex.table("p5-ncu", ["profile", "kernel", "$\\mu$s", "DRAM \\%", "GB/s", "occ.\\ \\%", "theor.\\ \\%",
                         "top stalls"], rows, "llrrrrrl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cg", type=Path, default=Path("results/phase5/cg.csv"))
    ap.add_argument("--amg", type=Path, default=Path("results/phase5/amg.csv"))
    ap.add_argument("--ncu", type=Path, default=Path("results/phase5/ncu"))
    ap.add_argument("--out", type=Path, default=Path("results/phase5"))
    ap.add_argument("--fig-dir", type=Path)
    ap.add_argument("--fig-format", default="pdf")
    ap.add_argument("--tex", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tex = TexWriter(args.tex / "phase5.tex" if args.tex else None)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cg(args.cg, args.out, args.fig_dir, tuple(args.fig_format.split(",")), tex)
        amg(args.amg, tex)
        ncu_table(args.ncu, tex)
    text = buf.getvalue()
    print(text, end="")
    (args.out / "summary.txt").write_text(text)
    if (p := tex.write()):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
