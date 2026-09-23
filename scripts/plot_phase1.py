#!/usr/bin/env python3
"""Phase 1 baseline figure: cuSPARSE SpMV efficiency (% of measured device bandwidth) vs matrix size.

Reads results/phase1_cusparse.csv (from bench_spmv) and writes results/phase1/cusparse_bw.png, plus a
summary table on stdout (the table view of the figure).

    python scripts/plot_phase1.py [--csv results/phase1_cusparse.csv] [--out results/phase1]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Reference palette (dataviz skill): categorical slots 1-2, text and surface tokens, light mode.
SERIES = {"csr_alg1": "#2a78d6", "csr_alg2": "#eb6834"}
LABEL = {"csr_alg1": "cuSPARSE CSR ALG1", "csr_alg2": "cuSPARSE CSR ALG2"}
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load(path):
    df = pd.read_csv(path)
    ok = df[df["error"].isna() & (df["check"] == "pass")].copy()
    return df, ok


def summary(ok):
    g = ok.groupby(["precision", "alg"])["pct_peak_bw"]
    table = pd.DataFrame({"matrices": g.size(), "median %BW": g.median(), "p25": g.quantile(0.25),
                          "p75": g.quantile(0.75), "max %BW": g.max()}).round(1)
    # Which algorithm wins per matrix (lower median time)?
    wins = (ok.loc[ok.groupby(["matrix", "precision"])["median_ms"].idxmin()]
              .groupby(["precision", "alg"]).size().rename("wins"))
    return table.join(wins).fillna({"wins": 0}).astype({"wins": int})


def plot(ok, out):
    precisions = [p for p in ("fp32", "fp64") if p in set(ok["precision"])]
    fig, axes = plt.subplots(1, len(precisions), figsize=(6 * len(precisions), 4.2), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, prec in zip(axes[0], precisions):
        ax.set_facecolor(SURFACE)
        sub = ok[ok["precision"] == prec]
        for alg, color in SERIES.items():
            s = sub[sub["alg"] == alg]
            if s.empty:
                continue
            ax.scatter(s["nnz"], s["pct_peak_bw"], s=40, color=color, edgecolors=SURFACE, linewidths=1.5,
                       label=LABEL[alg], zorder=3)
        ax.set_xscale("log")
        ax.set_ylim(0, 105)
        ax.axhline(100, color=TEXT2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.text(ax.get_xlim()[0], 101.5, " measured peak 187.8 GB/s", color=TEXT2, fontsize=8, va="bottom")
        ax.set_title(prec.upper(), color=TEXT, fontsize=11, loc="left")
        ax.set_xlabel("nonzeros (log scale)", color=TEXT2)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(colors=TEXT2, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0][0].set_ylabel("effective bandwidth, % of peak", color=TEXT2)
    axes[0][-1].legend(frameon=False, loc="lower right", fontsize=9, labelcolor=TEXT)
    fig.suptitle("cuSPARSE SpMV baseline on RTX 4050 Laptop (L2 flushed, median of repetitions)",
                 color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "cusparse_bw.png", dpi=150, facecolor=SURFACE)
    return out / "cusparse_bw.png"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("results/phase1_cusparse.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/phase1"))
    args = ap.parse_args()
    df, ok = load(args.csv)
    print(f"{df['matrix'].nunique()} matrices, {len(ok)} passing rows, "
          f"{int(df['error'].notna().sum())} error rows, {int((df['check'] == 'FAIL').sum())} correctness failures\n")
    print(summary(ok).to_string(), "\n")
    for prec in sorted(set(ok["precision"])):
        best = ok[ok["precision"] == prec].groupby("matrix")["pct_peak_bw"].max().sort_values()
        print(f"{prec}: lowest %BW {', '.join(f'{m} {v:.1f}' for m, v in best.head(3).items())}")
        print(f"{prec}: highest %BW {', '.join(f'{m} {v:.1f}' for m, v in best.tail(3).items())}")
    print(f"\nfigure: {plot(ok, args.out)}")


if __name__ == "__main__":
    main()
