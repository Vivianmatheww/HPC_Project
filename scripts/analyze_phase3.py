#!/usr/bin/env python3
"""Phase 3 evaluation on the HELD-OUT matrices (never seen by train_selector.py).

Inputs:  results/phase3/heldout_spmv.csv   every config on the held-out set (bench_spmv --kernels all)
         results/phase3/plan_heldout.csv   SpmvPlan end-to-end (bench_plan)
Outputs: results/phase3/heldout_profile.png    steady-state performance profile of the selectors
         results/phase3/calls_sweep.png        total-time speedup vs always-cuSPARSE as a function of N
         results/phase3/heldout_summary.txt

G2: steady-state choice within 10% of the oracle (geomean), never >5% slower than cuSPARSE on large matrices.
G3: feature extraction <= 2 cuSPARSE SpMVs.
G4: total time (plan + N calls) never >5% worse than always-cuSPARSE at N = 1.

    python scripts/analyze_phase3.py
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

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
KEY = ["matrix", "precision"]
PORTFOLIO = ["cusparse_alg1", "csr_vector_v4", "csr_binned_t2", "csr_scalar", "cusparse_alg2"]


def geomean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if len(x) else float("nan")


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spmv", type=Path, default=Path("results/phase3/heldout_spmv.csv"))
    ap.add_argument("--plan", type=Path, default=Path("results/phase3/plan_heldout.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/phase3"))
    args = ap.parse_args()

    s = pd.read_csv(args.spmv)
    assert (s.loc[s["status"] == "ok", "check"] == "pass").all(), "held-out SpMV correctness failures"
    times = s[s["status"] == "ok"].pivot_table(index=KEY, columns="config", values="median_ms")
    oracle = times.min(axis=1)
    cu = times["cusparse_alg1"]
    p = pd.read_csv(args.plan)
    assert (p["check"] == "pass").all(), "plan correctness failures"
    big = pd.Series(p.drop_duplicates(KEY).set_index(KEY)["nnz"] >= 1e6).reindex(times.index)

    def steady(sel):
        """Steady-state config per sample (N = max, where every selector extracts features)."""
        rows = p[(p["selector"] == sel) & (p["calls"] == p["calls"].max())].set_index(KEY)["steady_state"]
        return rows.reindex(times.index)

    def time_of(configs):
        return pd.Series([times.loc[k, c] if c in times.columns and np.isfinite(times.loc[k, c]) else cu[k]
                          for k, c in configs.items()], index=configs.index)

    lines = {
        "always cuSPARSE ALG1": cu,
        "rules (v1)": time_of(steady("rules")),
        "decision tree (v2)": time_of(steady("tree")),
        "oracle within portfolio": times[PORTFOLIO].min(axis=1),
    }

    buf = io.StringIO()
    with redirect_stdout(buf):
        n_mat = times.index.get_level_values(0).nunique()
        print(f"HELD-OUT: {n_mat} matrices x {times.index.get_level_values(1).nunique()} precisions = {len(times)} samples "
              f"({int(big.sum())} with >= 1M nnz); {times.shape[1]} configs; all plan and SpMV results correct\n")
        print(f"best of all 40 configs vs cuSPARSE ALG1: {geomean(cu / oracle):.3f}x geomean, max {(cu / oracle).max():.2f}x\n")

        print("== G2: steady-state selection vs the held-out oracle ==")
        for name, t in lines.items():
            r = t / oracle
            print(f"  {name:26s} T/T_oracle geomean {geomean(r):.4f} | within 10%: {np.mean(r <= 1.10):4.0%} | "
                  f"vs cuSPARSE {geomean(cu / t):.3f}x | >5% slower than cuSPARSE: {int((t > 1.05 * cu).sum())} "
                  f"(large: {int((t[big] > 1.05 * cu[big]).sum())})")
        tree_cfg = steady("tree")
        print("\n  tree picks:", tree_cfg.value_counts().to_dict())
        print("  held-out oracle labels within portfolio:", times[PORTFOLIO].idxmin(axis=1).value_counts().to_dict())
        worst = (lines["decision tree (v2)"] / cu).sort_values(ascending=False).head(5)
        print("  largest tree losses vs cuSPARSE:", {f"{k[0]} {k[1]}": round(v, 3) for k, v in worst.items()})

        print("\n== G3: feature extraction cost (held-out, plans that extracted features) ==")
        f = p[p["used_features"] == 1].drop_duplicates(KEY).set_index(KEY)
        units = (f["feature_ms"] / cu.reindex(f.index))
        print(f"  feature_ms median {f['feature_ms'].median():.4f} ms; in cuSPARSE SpMVs: median {units.median():.2f}, "
              f"p90 {units.quantile(0.9):.2f}; <= 2 for {np.mean(units <= 2):.0%}")
        size = pd.cut(f["nnz"], [0, 1e5, 1e6, 1e9], labels=["<100k", "100k-1M", ">1M"])
        print("  median by size:", units.groupby(size, observed=True).median().round(2).to_dict())

        print("\n== G4: total time (plan + N calls) vs always-cuSPARSE; geomean speedup [worst case] ==")
        base = p[p["selector"] == "always_cusparse"].set_index(KEY + ["calls"])["total_ms"]
        rows = []
        for sel in ("rules", "tree", "tree_nobe"):
            t = p[p["selector"] == sel].set_index(KEY + ["calls"])["total_ms"]
            r = (base / t.reindex(base.index))
            row = {"selector": sel}
            for n, g in r.groupby(level="calls"):
                row[f"N={n}"] = f"{geomean(g):.3f} [{g.min():.2f}]"
            rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False))
        uf = p[p["selector"] == "tree"].groupby("calls").agg(features=("used_features", "mean"),
                                                             steady=("config", lambda c: 0))
        conv_dropped = p[(p["selector"] == "tree") & (p["config"] != p["steady_state"])].groupby("calls").size()
        print("  tree: fraction of plans that extracted features by N:", uf["features"].round(2).to_dict())
        print("  tree: plans where break-even dropped the steady-state conversion, by N:", conv_dropped.to_dict())
        n1 = (base / p[p["selector"] == "tree"].set_index(KEY + ["calls"])["total_ms"].reindex(base.index)).xs(1, level="calls")
        t1 = p[(p["selector"] == "tree") & (p["calls"] == 1)].set_index(KEY)
        b1 = p[(p["selector"] == "always_cusparse") & (p["calls"] == 1)].set_index(KEY)
        same = (t1["config"] == b1["config"].reindex(t1.index)) & (t1["used_features"] == 0)
        print(f"  G4 check at N=1: tree worse than cuSPARSE by >5% on {int((n1 < 1 / 1.05).sum())} of {len(n1)} samples; "
              f"{int(same.sum())} of {len(t1)} N=1 tree plans do exactly the baseline's work (same config, no features),")
        print(f"  so their spread (ratio p5-p95 {n1[same.reindex(n1.index).values].quantile(0.05):.3f}-"
              f"{n1[same.reindex(n1.index).values].quantile(0.95):.3f}) is timing noise of the ~0.05 ms plan construction.")

    text = buf.getvalue()
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "heldout_summary.txt").write_text(text)

    # Figure 1: held-out steady-state profile
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    taus = np.logspace(0, np.log10(2), 300)
    for (name, t), color in zip(lines.items(), SLOTS):
        r = (t / oracle).to_numpy()
        ax.step(taus, [(r <= x).mean() for x in taus], where="post", color=color, linewidth=2, label=name)
    ax.set_xscale("log", base=2)
    ax.set_xlim(1, 2)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1, 1.1, 1.25, 1.5, 2])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_xlabel("within this factor of the fastest of all 40 configs", color=TEXT2)
    ax.set_ylabel("fraction of held-out samples", color=TEXT2)
    ax.set_title(f"Selector quality on {len(times)} held-out samples (steady state)", color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(args.out / "heldout_profile.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Figure 2: total-time speedup vs N
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    base = p[p["selector"] == "always_cusparse"].set_index(KEY + ["calls"])["total_ms"]
    names = {"rules": "rules (v1)", "tree": "decision tree + break-even", "tree_nobe": "decision tree, no break-even"}
    for (sel, label), color in zip(names.items(), SLOTS[1:]):
        t = p[p["selector"] == sel].set_index(KEY + ["calls"])["total_ms"].reindex(base.index)
        g = (base / t).groupby(level="calls")
        ns = np.array(sorted(g.groups))
        ax.plot(ns, [geomean(g.get_group(n)) for n in ns], color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    ax.axhline(1, color=SLOTS[0], linewidth=2, label="always cuSPARSE ALG1")
    ax.set_xscale("log")
    ax.set_xlabel("expected SpMV calls N (plan cost amortised over N calls)", color=TEXT2)
    ax.set_ylabel("total-time speedup vs cuSPARSE (geomean)", color=TEXT2)
    ax.set_title("End-to-end speedup on held-out matrices", color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(args.out / "calls_sweep.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {args.out / 'heldout_profile.png'}, {args.out / 'calls_sweep.png'}, {args.out / 'heldout_summary.txt'}")


if __name__ == "__main__":
    main()
