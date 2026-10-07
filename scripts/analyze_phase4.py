#!/usr/bin/env python3
"""Phase 4 evaluation: memory placement (G5) and out-of-core SpGEMM (G6).

Inputs:  results/phase4/placement.csv   bench_spgemm --study placement
         results/phase4/ooc.csv         bench_spgemm --study ooc
         results/phase4/chunks.csv      per-tile timelines of the last repetition of every ooc run
Outputs: results/phase4/placement.txt   our version of paper 1, Table 3 (slowdown vs all-device)
         results/phase4/scaling.png     GFLOP/s vs problem size / budget (paper 1, Figs. 12-13 style)
         results/phase4/timeline.png    copy engine and compute timeline, Alg. 4 vs RASP (the overlap)
         results/phase4/summary.txt     ablation, G6 verdict, cost-model accuracy, overheads
         --tex DIR                      phase4.tex: \\setstat values and tables for the report

G5: the placement table and an explanation. G6: RASP >= 1.3x Alg. 4 (geomean) when the problem exceeds the budget.

    python scripts/analyze_phase4.py [--fig-format png,pdf] [--fig-dir docs/report/figures] [--tex docs/report/generated]
"""
import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rasp_report import (GRID, METHOD_COLOR, METHOD_LABEL, METHODS, SLOTS, SURFACE, TEXT, TEXT2, TexWriter,  # noqa: E402
                         geomean, save, style, tex_escape)

MiB = float(1 << 20)
PLACEMENT_ORDER = [
    ("device", "device", "device"), ("device", "device", "pinned"), ("pinned", "device", "device"),
    ("pinned", "device", "pinned"), ("device", "pinned", "device"), ("device", "pinned", "pinned"),
    ("pinned", "pinned", "device"), ("pinned", "pinned", "pinned"), ("managed", "device", "device"),
    ("device", "managed", "device"), ("device", "device", "managed"), ("managed", "managed", "managed")]
SHORT = {"device": "D", "pinned": "P", "managed": "M"}


def placement(path, out, tex):
    if not path.exists():
        print(f"(no {path})")
        return
    df = pd.read_csv(path)
    bad = df[(df["status"] != "ok") | (df["check"] != "pass")]
    print(f"== Placement (G5): {len(df)} runs, {len(bad)} errors or failed checks")
    df = df[(df["status"] == "ok") & (df["check"] == "pass")].copy()
    key = ["workload", "n", "precision"]
    base = df[(df["a_loc"] == "device") & (df["b_loc"] == "device") & (df["c_loc"] == "device")]
    df = df.merge(base[key + ["median_ms"]].rename(columns={"median_ms": "base_ms"}), on=key)
    df["slowdown"] = df["median_ms"] / df["base_ms"]

    # The table: one row per placement; per workload the slowdown at its largest n and the range over all n.
    fp = "fp64" if (df["precision"] == "fp64").any() else df["precision"].iloc[0]
    d = df[df["precision"] == fp]
    workloads = sorted(d["workload"].unique())
    header = ["A", "B", "C"]
    for w in workloads:
        header += [f"{w} n={int(d[d.workload == w]['n'].max())}", f"{w} range"]
    rows, tex_rows = [], []
    for a, b, c in PLACEMENT_ORDER:
        sel = d[(d.a_loc == a) & (d.b_loc == b) & (d.c_loc == c)]
        if sel.empty:
            continue
        cells, tcells = [SHORT[a], SHORT[b], SHORT[c]], [SHORT[a], SHORT[b], SHORT[c]]
        for w in workloads:
            s = sel[sel.workload == w]
            if s.empty:
                cells += ["-", "-"]
                tcells += ["--", "--"]
                continue
            big = s[s.n == s.n.max()]["slowdown"].iloc[0]
            cells += [f"{big:.2f}x", f"{s.slowdown.min():.2f}-{s.slowdown.max():.2f}"]
            tcells += [f"{big:.2f}$\\times$", f"{s.slowdown.min():.1f}--{s.slowdown.max():.1f}"]
        rows.append(cells)
        tex_rows.append(tcells)
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    lines = ["  ".join(str(x).rjust(wd) for x, wd in zip(r, widths)) for r in [header] + rows]
    table = "\n".join(lines)
    print(f"slowdown vs all-device ({fp}; D = device, P = pinned zero-copy, M = managed):")
    print(table)
    (out / "placement.txt").write_text(table + "\n")
    tex_header = ["A", "B", "C"] + [h.replace("range", "(all $n$)") for h in header[3:]]
    tex.table("p4-placement", [tex_escape(h) if "$" not in h else h for h in tex_header], tex_rows, "ccc" + "r" * (len(header) - 3))

    for w in workloads:
        s = d[d.workload == w]
        nmax = s.n.max()
        big = s[s.n == nmax]
        for name, (a, b, c) in {"a": ("pinned", "device", "device"), "b": ("device", "pinned", "device"),
                                "c": ("device", "device", "pinned"), "all": ("pinned", "pinned", "pinned"),
                                "managed": ("managed", "managed", "managed")}.items():
            v = big[(big.a_loc == a) & (big.b_loc == b) & (big.c_loc == c)]["slowdown"]
            if len(v):
                tex.stat(f"p4-place-{w}-{name}", float(v.iloc[0]), "{:.1f}")
        g = s["gamma"].median()
        tex.stat(f"p4-gamma-{w}", float(g))
        print(f"  {w}: nnz(C) / flop upper bound = {g:.3f}; device-resident {s[(s.a_loc == 'device') & (s.b_loc == 'device') & (s.c_loc == 'device')]['gflops'].max():.2f} GFLOP/s at best")
    man = d[(d.a_loc == "managed") & (d.b_loc == "managed") & (d.c_loc == "managed")]
    pin = d[(d.a_loc == "pinned") & (d.b_loc == "pinned") & (d.c_loc == "pinned")]
    mm = man.merge(pin, on=key, suffixes=("_m", "_p"))
    if len(mm):
        r = geomean(mm["median_ms_m"] / mm["median_ms_p"])
        tex.stat("p4-managed-vs-pinned", r)
        print(f"  all-managed / all-pinned time: {r:.3f} (geomean): UVM on WSL2 behaves as zero-copy")


def ooc_frames(path):
    df = pd.read_csv(path)
    for c in ("median_ms", "gflops", "ratio", "budget_bytes", "problem_bytes"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def scaling_figure(ok, out, fig_dir, formats):
    """Per workload: GFLOP/s vs problem/budget at a fixed budget (sizes swept), and vs budget at a fixed size."""
    workloads = sorted(ok["workload"].unique())
    panels = []
    for w in workloads:
        d = ok[(ok.workload == w) & ok.method.isin(METHODS)]
        budgets = d[d.method != "incore"].groupby("budget_bytes")["n"].nunique()
        for b in budgets[budgets >= 2].index:
            panels.append((w, "size", b))
        sizes = d[d.method != "incore"].groupby("n")["budget_bytes"].nunique()
        for n in sizes[sizes >= 3].index:
            panels.append((w, "budget", n))
    if not panels:
        print("(not enough sweep points for the scaling figure)")
        return
    cols = min(len(panels), 2)
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.9 * rows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (w, kind, val) in zip(axes.flat, panels):
        style(ax)
        d = ok[ok.workload == w]
        if kind == "size":
            sel = d[(d.budget_bytes == val) & (d.method != "incore")]
            x, xlabel = "ratio", "problem size / budget  (|A|+|B|+|C|, budget fixed)"
            title = f"{w}: budget {val / MiB:.0f} MiB, grid size swept"
            ref = d[d.method == "device_only"]
            ns = sorted(sel.n.unique())
        else:
            sel = d[(d.n == val) & (d.method != "incore")]
            x, xlabel = "budget_mib", "fast-memory budget (MiB)"
            title = f"{w}: n = {int(val)}, budget swept"
        sel = sel.assign(budget_mib=sel.budget_bytes / MiB)
        ends = []
        for m in METHODS:
            s = sel[sel.method == m].sort_values(x)
            if s.empty:
                continue
            ax.plot(s[x], s["gflops"], color=METHOD_COLOR[m], linewidth=2, marker="o", markersize=5,
                    label=METHOD_LABEL[m], zorder=3)
            if m in ("rasp", "alg4"):  # direct labels for the two lines the figure is about
                ends.append((s.iloc[-1][x], s.iloc[-1]["gflops"], METHOD_LABEL[m].split(" (")[0]))
        if kind == "size":
            # In-core at the same grid sizes (pinned in and out, no overlap): where the problem still fits the device.
            r = d[(d.method == "incore") & d.n.isin(ns)].merge(sel[sel.method == "rasp"][["n", "ratio"]], on="n")
            if len(r):
                r = r.sort_values("ratio_y")
                ax.plot(r["ratio_y"], r["gflops"], color=METHOD_COLOR["incore"], linewidth=1.5, linestyle="--",
                        label=METHOD_LABEL["incore"], zorder=2)
            ax.axvline(1.0, color=GRID, linewidth=1.2, zorder=1)
            # The device-only kernel is an order of magnitude above (no PCIe at all): a number, not a line.
            dev = ref[ref.n.isin(ns)]["gflops"]
            if len(dev):
                title += f"  (device-only kernel: {dev.min():.1f}-{dev.max():.1f} GFLOP/s)"
        else:
            ax.set_xscale("log", base=2)
        ax.set_ylim(bottom=0)
        lo, hi = ax.get_ylim()
        placed = []  # nudge labels apart when the lines end close together
        for xe, ye, text in sorted(ends, key=lambda e: e[1]):
            y = ye
            for py in placed:
                if abs(y - py) < 0.06 * (hi - lo):
                    y = py + 0.06 * (hi - lo)
            placed.append(y)
            ax.text(xe, y, text, transform=offset_copy(ax.transData, fig=fig, x=6, y=0, units="points"), fontsize=8,
                    color=TEXT2, va="center", ha="left", clip_on=False)
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
        ax.set_xlabel(xlabel, fontsize=9, color=TEXT2)
        ax.set_ylabel("GFLOP/s (2 x multiply-adds / wall time)", fontsize=9, color=TEXT2)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    seen = dict()
    for a in axes.flat[:len(panels)]:
        for h, l in zip(*a.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.02 - 0.03 * (len(seen) > 6)))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for p in save(fig, out / "scaling", ("png",)) + (save(fig, fig_dir / "p4_scaling", formats) if fig_dir else []):
        print(f"wrote {p}")
    plt.close(fig)


def timeline_figure(chunks_path, ok, out, fig_dir, formats):
    """Copy engine and compute stream over time for Alg. 4 and RASP on the most out-of-core configuration."""
    if not chunks_path.exists():
        print(f"(no {chunks_path})")
        return
    ch = pd.read_csv(chunks_path)
    both = ok[ok.method.isin(["alg4", "rasp"])].groupby(["workload", "n", "budget_bytes"])["method"].nunique()
    both = both[both == 2].reset_index()
    if both.empty:
        print("(no configuration with both alg4 and rasp for the timeline)")
        return
    both = both.merge(ok[ok.method == "rasp"][["workload", "n", "budget_bytes", "ratio"]],
                      on=["workload", "n", "budget_bytes"])
    pick = both.sort_values(["workload", "ratio"]).groupby("workload").tail(1)
    kinds = [("h2d", "upload (H2D)", SLOTS[0]), ("comp", "cuSPARSE + sums", SLOTS[2]), ("d2h", "download (D2H)", SLOTS[1])]
    fig, axes = plt.subplots(2 * len(pick), 1, figsize=(10, 1.9 * 2 * len(pick)), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    i = 0
    for _, cfg in pick.iterrows():
        sel = ch[(ch.workload == cfg.workload) & (ch.n == cfg.n) & np.isclose(ch.budget_bytes, cfg.budget_bytes)]
        xmax = max(sel[["h2d_end", "comp_end", "d2h_end"]].max().max(), 1e-9)
        for m in ("alg4", "rasp"):
            ax = axes[i, 0]
            i += 1
            style(ax)
            ax.grid(False)
            s = sel[sel.method == m]
            total = ok[(ok.workload == cfg.workload) & (ok.n == cfg.n) & np.isclose(ok.budget_bytes, cfg.budget_bytes) &
                       (ok.method == m)]["exec_ms"]
            for k, label, color in kinds:
                y = 1 if k == "comp" else 0  # one copy engine: uploads and downloads share a row
                st, en = s[f"{k}_start"], s[f"{k}_end"]
                m_ok = (st >= 0) & (en >= st)
                ax.broken_barh(list(zip(st[m_ok], (en - st)[m_ok])), (y - 0.35, 0.7), facecolors=color,
                               edgecolor=SURFACE, linewidth=0.6, label=label)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["copy engine", "compute"], fontsize=8, color=TEXT2)
            ax.set_xlim(0, xmax * 1.02)
            ax.set_title(f"{cfg.workload} n={int(cfg.n)}, budget {cfg.budget_bytes / MiB:.0f} MiB "
                         f"(problem {cfg.ratio:.1f}x budget): {METHOD_LABEL[m]}"
                         + (f", {total.iloc[0]:.0f} ms" if len(total) else ""), fontsize=9, color=TEXT, loc="left")
    axes[-1, 0].set_xlabel("ms since execution started (last repetition)", fontsize=9, color=TEXT2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for p in save(fig, out / "timeline", ("png",)) + (save(fig, fig_dir / "p4_timeline", formats) if fig_dir else []):
        print(f"wrote {p}")
    plt.close(fig)


def ooc(path, chunks_path, out, fig_dir, formats, tex):
    if not path.exists():
        print(f"(no {path})")
        return
    df = ooc_frames(path)
    print(f"\n== Out-of-core (G6): {len(df)} rows")
    errors = df[df.status == "error"]
    failed = df[(df.status == "ok") & (df.check != "pass")]
    infeasible = df[df.status == "infeasible"]
    print(f"  errors {len(errors)}, failed checks {len(failed)}, infeasible {len(infeasible)} "
          f"({', '.join(sorted(infeasible.method.unique())) or '-'})")
    ok = df[(df.status == "ok") & (df.check == "pass")].copy()
    over = ok[(ok.peak_device_bytes > ok.budget_bytes * 1.0001) & (ok.budget_bytes > 0)]
    print(f"  budget exceeded: {len(over)} runs (must be 0)")
    nnz = ok.groupby(["workload", "n", "precision"])["nnz_c"].nunique()
    print(f"  nnz(C) consistent across methods: {bool((nnz == 1).all())}")

    scaling_figure(ok, out, fig_dir, formats)
    timeline_figure(chunks_path, ok, out, fig_dir, formats)

    key = ["workload", "n", "precision", "budget_bytes"]
    a4 = ok[ok.method == "alg4"][key + ["median_ms"]].rename(columns={"median_ms": "alg4_ms"})
    j = ok.merge(a4, on=key)
    j["speedup"] = j["alg4_ms"] / j["median_ms"]
    beyond = j[j.ratio > 1.0]
    print("\n  speedup over Alg. 4 when the problem exceeds the budget (geomean [min, max], #configs):")
    workloads = sorted(beyond.workload.unique())
    rows = []
    for m in ["alg4_overlap", "rasp_static", "rasp", "zerocopy"]:
        cells = [METHOD_LABEL[m]]
        line = f"    {METHOD_LABEL[m]:<32}"
        for w in workloads:
            s = beyond[(beyond.method == m) & (beyond.workload == w)]["speedup"]
            g = geomean(s)
            line += f"  {w}: {g:.2f} [{s.min():.2f}, {s.max():.2f}] n={len(s)}" if len(s) else f"  {w}: -"
            cells.append(f"{g:.2f}$\\times$" if len(s) else "--")
            tex.stat(f"p4-speedup-{m}-{w}", g)
        rows.append(cells)
        print(line)
    tex.table("p4-ablation", ["vs.\\ Alg.~4 (problem $>$ budget)"] + [tex_escape(w) for w in workloads], rows)
    print("\n  G6 (target >= 1.3x over Alg. 4 when problem > budget):")
    for w in workloads:
        g = geomean(beyond[(beyond.method == "rasp") & (beyond.workload == w)]["speedup"])
        verdict = "MET" if g >= 1.3 else "NOT MET"
        tex.stat(f"p4-g6-{w}", g)
        tex.stat(f"p4-g6-{w}-verdict", verdict.lower())
        print(f"    {w}: {g:.2f}x -> {verdict}")

    r = ok[ok.method == "rasp"]
    if len(r):
        acc = (r["predicted_ms"] / r["exec_ms"]).replace([np.inf, -np.inf], np.nan).dropna()
        over_ms = (r["plan_ms"] + r["estimate_ms"]) / r["median_ms"]
        print(f"\n  RASP cost model: predicted / measured execution time median {acc.median():.2f} "
              f"(p10 {acc.quantile(0.1):.2f}, p90 {acc.quantile(0.9):.2f})")
        print(f"  RASP estimate + plan share of total time: median {100 * over_ms.median():.1f}% "
              f"(max {100 * over_ms.max():.1f}%)")
        print(f"  RASP re-plans per run: mean {r.replans.mean():.2f}; overflow splits: mean {r.splits.mean():.2f}")
        tex.stat("p4-model-accuracy", float(acc.median()))
        tex.stat("p4-overhead-pct", float(100 * over_ms.median()), "{:.1f}")
        tex.stat("p4-replans-mean", float(r.replans.mean()))
        dev = ok[ok.method == "device_only"][["workload", "n", "precision", "gflops"]].rename(columns={"gflops": "dev"})
        rr = r.merge(dev, on=["workload", "n", "precision"])
        if len(rr):
            frac = (rr["gflops"] / rr["dev"]).median()
            print(f"  RASP GFLOP/s as a fraction of the device-only kernel: median {frac:.2f}")
            tex.stat("p4-frac-of-device", float(frac))
        print("  measured model parameters (RASP runs, median): "
              f"{r.measured_flops_per_ms.median() / 1e6:.2f} G multiply-adds/s, gamma {r.measured_gamma.median():.3f}, "
              f"workspace {r.measured_ws_per_flop.median():.1f} B/flop; PCIe {r.h2d_gbs.median():.1f} / "
              f"{r.d2h_gbs.median():.1f} GB/s")
        for c in ("measured_gamma", "measured_ws_per_flop", "h2d_gbs"):
            tex.stat(f"p4-{c.replace('_', '-')}", float(r[c].median()))
        tex.stat("p4-rate-gflops", float(2 * r.measured_flops_per_ms.median() / 1e6))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--placement", type=Path, default=Path("results/phase4/placement.csv"))
    ap.add_argument("--ooc", type=Path, default=Path("results/phase4/ooc.csv"))
    ap.add_argument("--chunks", type=Path, default=Path("results/phase4/chunks.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/phase4"))
    ap.add_argument("--fig-dir", type=Path, help="also write the figures here (the report's figure directory)")
    ap.add_argument("--fig-format", default="pdf", help="formats for --fig-dir, e.g. pdf or png,pdf")
    ap.add_argument("--tex", type=Path, help="directory for phase4.tex (report values and tables)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tex = TexWriter(args.tex / "phase4.tex" if args.tex else None)
    formats = tuple(args.fig_format.split(","))

    buf = io.StringIO()
    with redirect_stdout(buf):
        placement(args.placement, args.out, tex)
        ooc(args.ooc, args.chunks, args.out, args.fig_dir, formats, tex)
    text = buf.getvalue()
    print(text, end="")
    (args.out / "summary.txt").write_text(text)
    if (p := tex.write()):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
