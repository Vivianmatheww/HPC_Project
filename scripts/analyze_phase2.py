#!/usr/bin/env python3
"""Phase 2 analysis: oracle table, performance profiles, and the paper-2 (ArgCSR) reproduction.

Inputs:  results/phase2_spmv.csv (bench_spmv --kernels phase2) [+ results/phase3a_extensions.csv], manifest.csv
         Several --csv files are merged; if a (matrix, precision, config) appears twice the first file wins.
Outputs: results/spmv_oracle.csv            best config per (matrix, precision) + best of every family
         results/phase2/perf_profile.png    Dolan-More performance profiles, FP32 and FP64
         results/phase2/argcsr_chunk.png    ArgCSR GFLOP/s vs desiredChunkSize (paper-2 sensitivity)
         results/phase2/perf_profile_extensions.png   CSR-binned / ArgCSR-local vs what they extend (if present)
         results/phase2/summary.txt         the tables printed below
         --tex DIR                          phase2.tex: \\setstat values and tables for the report
         --fig-dir DIR                      the figures again, for the report (--fig-format pdf)

The matrix kinds come from data/suitesparse/manifest.csv; without the data (it is not in git) they are taken from
the committed oracle table instead.

    python scripts/analyze_phase2.py [--csv results/phase2_spmv.csv] [--tex docs/report/generated --fig-dir docs/report/figures]
"""
import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rasp_report import TexWriter, tex_escape  # noqa: E402

# Reference palette (dataviz skill), categorical slots in fixed order, light mode.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

# Profile lines: (name, how to pick the time per matrix). "best" = fastest config of the family (tuned).
PROFILE_LINES = [
    ("cuSPARSE ALG1 (default)", "config", "cusparse_alg1"),
    ("CSR-scalar", "config", "csr_scalar"),
    ("CSR-vector, best v", "family", "csr_vector"),
    ("ELL", "config", "ell"),
    ("SELL-C-σ, best C,σ", "family", "sell"),
    ("ArgCSR B=128 D=1 (paper default)", "config", "argcsr_B128_D1"),
    ("ArgCSR, best B,D", "family", "argcsr"),
]
# Extension profile: each new format next to the fixed and the oracle-tuned version of what it extends.
PROFILE_EXT = [
    ("cuSPARSE ALG1 (default)", "config", "cusparse_alg1"),
    ("CSR-vector, best v (tuned per matrix)", "family", "csr_vector"),
    ("CSR-binned t=4 (no tuning)", "config", "csr_binned_t4"),
    ("ArgCSR B=128 D=1 (paper default)", "config", "argcsr_B128_D1"),
    ("ArgCSR, best B,D (tuned per matrix)", "family", "argcsr"),
    ("ArgCSR-local B=128 R=32 (no tuning)", "config", "argcsr_local_B128_R32"),
]
PAPER2 = ["Rajat/rajat23", "IBM_EDA/ckt11752_dc_1", "Sandia/ASIC_100k", "TSOPF/TSOPF_RS_b162_c1",
          "TSOPF/TSOPF_FS_b39_c7", "QY/case39", "Schenk_AFE/af_shell1", "Norris/torso2", "TKK/t2d_q9",
          "GHS_indef/ncvxqp9"]


def geomean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if len(x) else float("nan")


def load(csv_paths, manifest_path, kinds_from=Path("results/spmv_oracle.csv")):
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    df = df.drop_duplicates(subset=["matrix", "precision", "config"], keep="first")
    bad = df[(df["status"] == "ok") & (df["check"] != "pass")]
    if len(bad):
        raise SystemExit(f"{len(bad)} rows failed the correctness check; fix before analysing:\n{bad[['matrix', 'config', 'precision', 'rel_err']]}")
    ok = df[df["status"] == "ok"].copy()
    if manifest_path.exists():
        man = pd.read_csv(manifest_path)
        man["matrix"] = man["group"] + "/" + man["name"]
        ok = ok.merge(man[["matrix", "kind", "source"]], on="matrix", how="left")
    elif kinds_from.exists():  # no data/ on this machine: the committed oracle table carries the same columns
        man = pd.read_csv(kinds_from)[["matrix", "kind", "source"]].drop_duplicates("matrix")
        ok = ok.merge(man, on="matrix", how="left")
    else:
        ok["kind"] = ok["source"] = ""
    return df, ok


def oracle_table(ok):
    key = ["matrix", "precision"]
    best = ok.loc[ok.groupby(key)["median_ms"].idxmin(), key + ["rows", "cols", "nnz", "kind", "source", "config",
                                                                 "kernel", "median_ms", "gflops", "pct_peak_bw"]]
    best = best.rename(columns={"config": "best_config", "kernel": "best_family", "median_ms": "best_ms",
                                "gflops": "best_gflops", "pct_peak_bw": "best_pct_peak_bw"})
    fam = ok.loc[ok.groupby(key + ["kernel"])["median_ms"].idxmin(), key + ["kernel", "config", "median_ms"]]
    for f, g in fam.groupby("kernel"):
        best = best.merge(g[key + ["config", "median_ms"]].rename(
            columns={"config": f"best_{f}_config", "median_ms": f"best_{f}_ms"}), on=key, how="left")
    for cfg in ("cusparse_alg1", "argcsr_B128_D1"):
        best = best.merge(ok[ok["config"] == cfg][key + ["median_ms"]].rename(columns={"median_ms": f"{cfg}_ms"}),
                          on=key, how="left")
    best["speedup_vs_cusparse_alg1"] = best["cusparse_alg1_ms"] / best["best_ms"]
    best["speedup_vs_best_cusparse"] = best["best_cusparse_ms"] / best["best_ms"]
    return best.sort_values(key).reset_index(drop=True)


def line_times(ok, prec, how, what):
    sub = ok[ok["precision"] == prec]
    sel = sub[sub["config"] == what] if how == "config" else sub[sub["kernel"] == what]
    return sel.groupby("matrix")["median_ms"].min()


def save_fig(fig, out, name, fig_dir=None, formats=("pdf",)):
    fig.savefig(out / name, dpi=150, facecolor=SURFACE)
    if fig_dir:
        fig_dir.mkdir(parents=True, exist_ok=True)
        for ext in formats:
            fig.savefig(fig_dir / f"p2_{Path(name).stem}.{ext}", dpi=150, facecolor=SURFACE, bbox_inches="tight")


def perf_profile(ok, out, lines=PROFILE_LINES, name="perf_profile.png", fig_dir=None, formats=("pdf",)):
    precs = [p for p in ("fp32", "fp64") if p in set(ok["precision"])]
    fig, axes = plt.subplots(1, len(precs), figsize=(6.2 * len(precs), 5.0), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    taus = np.logspace(0, np.log10(8), 400)
    for ax, prec in zip(axes[0], precs):
        best = ok[ok["precision"] == prec].groupby("matrix")["median_ms"].min()
        for (label, how, what), color in zip(lines, SLOTS):
            t = line_times(ok, prec, how, what).reindex(best.index)  # missing (infeasible) -> NaN -> never solved
            ratio = (t / best).to_numpy()
            frac = [(np.nan_to_num(ratio, nan=np.inf) <= tau).mean() for tau in taus]
            ax.step(taus, frac, where="post", color=color, linewidth=2, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xlim(1, 8)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([1, 1.25, 1.5, 2, 3, 4, 8])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
        ax.set_title(f"{prec.upper()}  ({len(best)} matrices)", color=TEXT, fontsize=11, loc="left")
        ax.set_xlabel("within this factor of the fastest configuration", color=TEXT2)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT2, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0][0].set_ylabel("fraction of matrices", color=TEXT2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=4 if len(lines) > 6 else 3, fontsize=9,
               labelcolor=TEXT)
    fig.suptitle("SpMV performance profiles on RTX 4050 Laptop: higher and further left is better",
                 color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save_fig(fig, out, name, fig_dir, formats)
    plt.close(fig)


def chunk_sensitivity(ok, out, matrices=("Schenk_AFE/af_shell1", "Rajat/rajat23"), block=128, prec="fp64",
                      fig_dir=None, formats=("pdf",)):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for m, color in zip(matrices, SLOTS):
        sub = ok[(ok["matrix"] == m) & (ok["precision"] == prec) & ok["config"].str.match(rf"argcsr_B{block}_D\d+$")]
        if sub.empty:
            continue
        sub = sub.assign(D=sub["config"].str.extract(r"_D(\d+)$")[0].astype(int)).sort_values("D")
        ax.plot(sub["D"], sub["gflops"], color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=m.split("/")[1])
        last = sub.iloc[-1]
        ax.annotate(m.split("/")[1], (last["D"], last["gflops"]), xytext=(6, 0), textcoords="offset points",
                    color=TEXT, fontsize=9, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(0.8, 60)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("desiredChunkSize", color=TEXT2)
    ax.set_ylabel(f"GFLOP/s ({prec.upper()})", color=TEXT2)
    ax.set_title(f"ArgCSR sensitivity to desiredChunkSize (block = {block})", color=TEXT, fontsize=11, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=TEXT)
    fig.tight_layout()
    save_fig(fig, out, "argcsr_chunk.png", fig_dir, formats)
    plt.close(fig)


def report(df, ok, oracle, tex):
    n_inf = int((df["status"] == "infeasible").sum())
    n_err = int((df["status"] == "error").sum())
    print(f"{df['matrix'].nunique()} matrices x {df['config'].nunique()} configs x {df['precision'].nunique()} precisions: "
          f"{len(ok)} ok rows (all pass the correctness check), {n_inf} infeasible, {n_err} errors\n")
    tex.stat("p2-matrices", df["matrix"].nunique())
    tex.stat("p2-configs", df["config"].nunique())
    tex.stat("p2-runs", len(ok))
    tex.stat("p2-infeasible", n_inf)

    print("== Which family is fastest (oracle wins) ==")
    wins = oracle.groupby(["precision", "best_family"]).size().unstack(0).fillna(0).astype(int)
    print(wins.to_string(), "\n")
    tex.table("p2-wins", ["fastest family"] + [c.upper() for c in wins.columns],
              [[tex_escape(f)] + [int(v) for v in r] for f, r in wins.iterrows()])

    print("== Speedup of the oracle (best of all configs) ==")
    for prec, g in oracle.groupby("precision"):
        print(f"{prec}: vs cuSPARSE ALG1 geomean {geomean(g['speedup_vs_cusparse_alg1']):.2f}x "
              f"(max {g['speedup_vs_cusparse_alg1'].max():.2f}x); vs best cuSPARSE {geomean(g['speedup_vs_best_cusparse']):.2f}x; "
              f"cuSPARSE is the oracle on {int((g['best_family'] == 'cusparse').sum())}/{len(g)}")
        tex.stat(f"p2-oracle-{prec}", geomean(g["speedup_vs_cusparse_alg1"]))
        tex.stat(f"p2-oracle-{prec}-max", g["speedup_vs_cusparse_alg1"].max())
        tex.stat(f"p2-cusparse-wins-{prec}", int((g["best_family"] == "cusparse").sum()))
    print()

    print("== Each line of the profile vs cuSPARSE ALG1 (geomean speedup; >1 = faster) ==")
    rows = []
    for name, how, what in PROFILE_LINES:
        r = {"line": name}
        for prec in ("fp32", "fp64"):
            base = line_times(ok, prec, "config", "cusparse_alg1")
            t = line_times(ok, prec, how, what).reindex(base.index)
            r[f"{prec} geomean"] = round(geomean(base / t), 3)
            r[f"{prec} feasible"] = int(t.notna().sum())
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False), "\n")
    tex.table("p2-lines", ["configuration", "FP32", "FP64", "feasible"],
              [[tex_escape(r["line"]), f"{r['fp32 geomean']:.2f}$\\times$", f"{r['fp64 geomean']:.2f}$\\times$",
                r["fp64 feasible"]] for r in rows], "lrrr")
    for r in rows:
        key = {"ArgCSR B=128 D=1 (paper default)": "argcsr-default", "ArgCSR, best B,D": "argcsr-tuned",
               "CSR-vector, best v": "csr-vector-best", "SELL-C-σ, best C,σ": "sell-best"}.get(r["line"])
        if key:
            tex.stat(f"p2-{key}-fp32", r["fp32 geomean"])
            tex.stat(f"p2-{key}-fp64", r["fp64 geomean"])

    print("== By size (FP64): oracle speedup vs cuSPARSE ALG1, and most common winning family ==")
    o = oracle[oracle["precision"] == "fp64"].copy()
    o["size"] = pd.cut(o["nnz"], [0, 1e5, 1e6, 1e9], labels=["<100k", "100k-1M", ">1M"])
    by_size = o.groupby("size", observed=True).agg(matrices=("matrix", "size"),
                                                   geomean_speedup=("speedup_vs_cusparse_alg1", geomean),
                                                   top_family=("best_family", lambda s: s.value_counts().index[0]))
    print(by_size.round(2).to_string(), "\n")
    for size, r in by_size.iterrows():
        tex.stat(f"p2-oracle-size-{size}", r["geomean_speedup"])

    print("== Best ArgCSR desiredChunkSize per matrix (a single global value cannot be right) ==")
    arg = ok[ok["kernel"] == "argcsr"].copy()
    arg["D"] = arg["config"].str.extract(r"_D(\d+)$")[0].astype(int)
    arg["B"] = arg["config"].str.extract(r"_B(\d+)_")[0].astype(int)
    bestD = arg.loc[arg.groupby(["matrix", "precision"])["median_ms"].idxmin()]
    print(bestD.groupby(["precision", "D"]).size().unstack(0).fillna(0).astype(int).to_string())
    d1 = arg[arg["B"] == 128].pivot_table(index=["matrix", "precision"], columns="D", values="median_ms")
    ratio = (d1[1] / d1.min(axis=1))
    print(f"fixed D=1 (B=128) vs best D (B=128): geomean slowdown {geomean(ratio):.2f}x, worst {ratio.max():.2f}x "
          f"({ratio.idxmax()[0]}, {ratio.idxmax()[1]})\n")
    tex.stat("p2-d1-slowdown", geomean(ratio))
    tex.stat("p2-d1-worst", ratio.max())
    tex.stat("p2-d1-worst-matrix", tex_escape(ratio.idxmax()[0].split("/")[-1]))
    tex.stat("p2-d1-best-count-fp64", int((bestD[bestD["precision"] == "fp64"]["D"] == 1).sum()))

    print("== Paper-2 check: desiredChunkSize 1 vs 32 (FP64, B=128), GFLOP/s ==")
    for m, claim in (("Schenk_AFE/af_shell1", "paper (AFE): D=32 better, 18 vs 11 GFLOPS"),
                     ("Rajat/rajat23", "paper: D=1 better, 5.1 vs 0.81 GFLOPS (6.3x)")):
        g = arg[(arg["matrix"] == m) & (arg["precision"] == "fp64") & (arg["B"] == 128)].set_index("D")["gflops"]
        if 1 in g and 32 in g:
            print(f"{m}: D=1 {g[1]:.2f}, D=32 {g[32]:.2f} -> D=1/D=32 = {g[1] / g[32]:.2f}   [{claim}]")
            key = m.split("/")[1].replace("_", "")
            tex.stat(f"p2-{key}-d1", g[1])
            tex.stat(f"p2-{key}-d32", g[32])
            st = arg[(arg["matrix"] == m) & (arg["precision"] == "fp64") & (arg["B"] == 128)].set_index("D")["stored_ratio"]
            tex.stat(f"p2-{key}-stored-d1", st[1], "{:.1f}")
            tex.stat(f"p2-{key}-stored-d32", st[32], "{:.1f}")
    print()

    print("== Paper-2 named matrices (FP64): GFLOP/s ==")
    p = oracle[(oracle["precision"] == "fp64") & oracle["matrix"].isin(PAPER2)].set_index("matrix")
    ok64 = ok[ok["precision"] == "fp64"]
    gf = lambda cfg: ok64[ok64["config"] == cfg].set_index("matrix")["gflops"]  # noqa: E731
    fam_gf = lambda fam: ok64[ok64["kernel"] == fam].groupby("matrix")["gflops"].max()  # noqa: E731
    t = pd.DataFrame({"kind": p["kind"], "cusparse_alg1": gf("cusparse_alg1"), "argcsr_B128_D1": gf("argcsr_B128_D1"),
                      "argcsr_best": fam_gf("argcsr"), "best": p["best_gflops"], "best_config": p["best_config"]}).loc[p.index]
    t["argcsr_D1/cusparse"] = t["argcsr_B128_D1"] / t["cusparse_alg1"]
    print(t.round(2).to_string(), "\n")
    tex.table("p2-named", ["matrix", "cuSPARSE", "ArgCSR D=1", "ArgCSR best", "best", "best config"],
              [[tex_escape(m.split("/")[1]), f"{r.cusparse_alg1:.1f}", f"{r.argcsr_B128_D1:.1f}", f"{r.argcsr_best:.1f}",
                f"{r.best:.1f}", tex_escape(r.best_config)] for m, r in t.iterrows()], "lrrrrl")
    for m in ("GHS_indef/ncvxqp9", "Norris/torso2"):
        if m in t.index:
            tex.stat(f"p2-{m.split('/')[1]}-argcsr-vs-cusparse", t.loc[m, "argcsr_best"] / t.loc[m, "cusparse_alg1"])

    if (ok["kernel"] == "argcsr_local").any() or (ok["kernel"] == "csr_binned").any():
        print("== Extensions (vs cuSPARSE ALG1, geomean speedup; >1 = faster) ==")
        rows = []
        for cfg in sorted(ok.loc[ok["kernel"].isin(["csr_binned", "argcsr_local"]), "config"].unique()):
            r = {"config": cfg}
            for prec in ("fp32", "fp64"):
                base = line_times(ok, prec, "config", "cusparse_alg1")
                t = line_times(ok, prec, "config", cfg).reindex(base.index)
                r[prec] = round(geomean(base / t), 3)
                r[f"{prec} wins"] = int((base / t > 1).sum())
            rows.append(r)
            for prec in ("fp32", "fp64"):
                tex.stat(f"p2-ext-{cfg.replace('_', '-')}-{prec}", r[prec])
        print(pd.DataFrame(rows).to_string(index=False))
        arg = ok[ok["kernel"] == "argcsr"]
        k = ["matrix", "precision"]
        tuned = arg.groupby(k)["median_ms"].min()
        d1 = arg[arg["config"] == "argcsr_B128_D1"].set_index(k)["median_ms"]
        loc = ok[ok["config"] == "argcsr_local_B128_R32"].set_index(k)["median_ms"]
        print(f"ArgCSR-local B128 R32 (one untuned config): {geomean(d1 / loc):.2f}x vs paper default D=1; "
              f"reaches {geomean(tuned / loc):.2f} of per-matrix-tuned ArgCSR (18 configs)\n")
        tex.stat("p2-local-vs-default", geomean(d1 / loc))
        tex.stat("p2-local-vs-tuned", geomean(tuned / loc))

    print("== Conversion cost in SpMV calls (median over matrices of convert_ms / median_ms, FP64) ==")
    c = ok[ok["precision"] == "fp64"].assign(calls=lambda d: d["convert_ms"] / d["median_ms"])
    calls = c.groupby("kernel")["calls"].median()
    print(calls.round(1).to_string())
    for k, v in calls.items():
        tex.stat(f"p2-convert-calls-{k.replace('_', '-')}", v, "{:.0f}")
    print("(host-side conversion; cuSPARSE = cusparseSpMV_preprocess on the GPU)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, nargs="+",
                    default=[Path("results/phase2_spmv.csv"), Path("results/phase3a_extensions.csv")])
    ap.add_argument("--manifest", type=Path, default=Path("data/suitesparse/manifest.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/phase2"))
    ap.add_argument("--oracle", type=Path, default=Path("results/spmv_oracle.csv"))
    ap.add_argument("--tex", type=Path, help="directory for phase2.tex (report values and tables)")
    ap.add_argument("--fig-dir", type=Path, help="also write the figures here (the report's figure directory)")
    ap.add_argument("--fig-format", default="pdf")
    args = ap.parse_args()
    tex = TexWriter(args.tex / "phase2.tex" if args.tex else None)
    formats = tuple(args.fig_format.split(","))

    df, ok = load([c for c in args.csv if c.exists()], args.manifest)
    oracle = oracle_table(ok)
    args.out.mkdir(parents=True, exist_ok=True)
    oracle.to_csv(args.oracle, index=False, float_format="%.6g")
    args.csv = [c for c in args.csv if c.exists()]
    perf_profile(ok, args.out, fig_dir=args.fig_dir, formats=formats)
    if (ok["kernel"] == "argcsr_local").any():
        perf_profile(ok, args.out, PROFILE_EXT, "perf_profile_extensions.png", args.fig_dir, formats)
    chunk_sensitivity(ok, args.out, fig_dir=args.fig_dir, formats=formats)

    buf = io.StringIO()
    with redirect_stdout(buf):
        report(df, ok, oracle, tex)
    text = buf.getvalue()
    print(text)
    (args.out / "summary.txt").write_text(text)
    print(f"wrote {args.oracle}, {args.out / 'perf_profile.png'}, {args.out / 'argcsr_chunk.png'}, {args.out / 'summary.txt'}")
    if (p := tex.write()):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
