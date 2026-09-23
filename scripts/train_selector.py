#!/usr/bin/env python3
"""Train the runtime SpMV selector and its cost models; emit src/selector/tree_generated.hpp.

Inputs (training set = data/suitesparse, 60 matrices x fp32/fp64):
  results/phase2_spmv.csv, results/phase3a_extensions.csv   SpMV time of every config (the oracle table)
  results/phase3/features_train.csv                         GPU features per (matrix, precision)
  results/phase3/gpu_convert_train.csv                      GPU conversion time per (matrix, precision, config)
  data/suitesparse/manifest.csv                             SuiteSparse group, for grouped cross-validation

Steps:
  1. Portfolio: greedily pick configs (cuSPARSE ALG1 first) until adding one improves the portfolio's
     best-case geomean slowdown vs the full oracle by < 0.5%, max 8. Labels = best portfolio config.
  2. Decision trees (full portfolio; and no-conversion subset) chosen by grouped 5-fold CV over
     (max_depth, min_samples_leaf, sample weighting). Metric: geomean T_selected / T_oracle.
     Then the minimum expected gain tau, also by grouped CV: the selector leaves cuSPARSE ALG1 only when the
     leaf's training speedup (computed on the training fold only) is >= tau.
  3. Cost models: cuSPARSE time vs bytes, GPU conversion time per (family, precision), feature time vs rows.
  4. Write the C++ header and a report (results/phase3/selector_train.txt).

    python scripts/train_selector.py
"""
import argparse
import itertools
import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

KEY = ["matrix", "precision"]
NO_CONVERSION = ("cusparse", "csr_scalar", "csr_vector")


def geomean(x):
    x = np.asarray(x, float)
    return float(np.exp(np.log(x).mean()))


def load(args):
    t = pd.concat([pd.read_csv(p) for p in args.spmv], ignore_index=True)
    t = t.drop_duplicates(KEY + ["config"])
    t = t[t["status"] == "ok"]
    times = t.pivot_table(index=KEY, columns="config", values="median_ms")  # NaN = infeasible
    feats = pd.read_csv(args.features).set_index(KEY)
    man = pd.read_csv(args.manifest)
    man["matrix"] = man["group"] + "/" + man["name"]
    group = man.set_index("matrix")["group"]
    family = t.drop_duplicates("config").set_index("config")["kernel"]
    conv = pd.read_csv(args.convert)
    return times, feats, group, family, conv, t


def pick_portfolio(times, max_size=8, min_gain=0.005):
    oracle = times.min(axis=1)
    ratio = times.div(oracle, axis=0).fillna(np.inf)
    chosen = ["cusparse_alg1"]
    history = [(chosen[0], geomean(ratio[chosen].min(axis=1)))]
    while len(chosen) < max_size:
        best, best_score = None, history[-1][1]
        for c in times.columns:
            if c in chosen:
                continue
            score = geomean(ratio[chosen + [c]].min(axis=1))
            if score < best_score:
                best, best_score = c, score
        if best is None or history[-1][1] / best_score - 1 < min_gain:
            break
        chosen.append(best)
        history.append((best, best_score))
    return chosen, history


def time_of(times, i, label):
    """SpMV time of `label` on sample i; an infeasible format falls back to cuSPARSE ALG1 (as SpmvPlan does)."""
    t = times.iloc[i][label]
    return times.iloc[i]["cusparse_alg1"] if np.isnan(t) else t


def fit_tree(X, y, w, depth, leaf):
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=0)
    clf.fit(X, y, sample_weight=w)
    return clf


def regret_weight(times, portfolio):
    sub = times[portfolio]
    return np.log(sub.max(axis=1) / sub.min(axis=1)).fillna(0).to_numpy() + 0.01


def cross_validate(X, labels, times, groups, portfolio, grid):
    """Grouped 5-fold CV. Returns (best params, table of all results, out-of-fold predictions for the best)."""
    oracle = times.min(axis=1).to_numpy()
    rows, best = [], None
    for depth, leaf, weighting in grid:
        pred = np.empty(len(labels), dtype=object)
        for tr, te in GroupKFold(n_splits=5).split(X, labels, groups):
            w = regret_weight(times.iloc[tr], portfolio) if weighting == "regret" else None
            clf = fit_tree(X[tr], labels[tr], w, depth, leaf)
            pred[te] = clf.predict(X[te])
        t_pred = np.array([time_of(times, i, p) for i, p in enumerate(pred)])
        score = geomean(t_pred / oracle)
        rows.append({"max_depth": depth, "min_samples_leaf": leaf, "weighting": weighting, "cv_geomean_vs_oracle": score,
                     "within_10pct": float(np.mean(t_pred / oracle <= 1.10))})
        if best is None or score < best[0]:
            best = (score, (depth, leaf, weighting), pred)
    return best[1], pd.DataFrame(rows).sort_values("cv_geomean_vs_oracle"), best[2]


def cv_min_gain(X, labels, times, groups, params, portfolio, taus):
    """Out-of-fold effect of the minimum-gain gate for each tau (leaf speedups from the training fold only)."""
    d, l, wt = params
    oracle = times.min(axis=1).to_numpy()
    cu = times["cusparse_alg1"].to_numpy()
    big = X[:, 1] >= np.log2(1e6)  # log2_nnz
    out = []
    for tau, large_gate in itertools.product(taus, [False, True]):
        t_sel = np.empty(len(labels))
        for tr, te in GroupKFold(n_splits=5).split(X, labels, groups):
            w = regret_weight(times.iloc[tr], portfolio) if wt == "regret" else None
            clf = fit_tree(X[tr], labels[tr], w, d, l)
            sp = leaf_speedups(clf, X[tr], times.iloc[tr])
            pred, leaves = clf.predict(X[te]), clf.apply(X[te])
            for j, i in enumerate(te):
                label = pred[j] if sp.get(leaves[j], 1.0) >= tau else "cusparse_alg1"
                if large_gate and big[i]:
                    label = "cusparse_alg1"
                t_sel[i] = time_of(times, i, label)
        out.append({"tau": tau, "large_gate": large_gate, "cv_geomean_vs_oracle": geomean(t_sel / oracle), "vs_cusparse": geomean(cu / t_sel),
                    "within_10pct": float(np.mean(t_sel / oracle <= 1.1)),
                    "slower_than_cusparse_5pct": int(np.sum(t_sel > 1.05 * cu)),
                    "large_slower_5pct": int(np.sum(t_sel[big] > 1.05 * cu[big]))})
    return pd.DataFrame(out)


def emit_tree(clf, feature_names, leaf_speedup, name):
    t = clf.tree_
    lines = [f"inline TreeResult {name}(const double* f) {{"]

    def rec(node, indent):
        pad = "  " * indent
        if t.children_left[node] == -1:
            label = clf.classes_[int(np.argmax(t.value[node][0]))]
            lines.append(f'{pad}return {{"{label}", {leaf_speedup[node]:.6g}}};  // n={int(t.n_node_samples[node])}')
            return
        fi, thr = int(t.feature[node]), float(t.threshold[node])
        lines.append(f"{pad}if (f[{fi}] <= {thr:.17g}) {{  // {feature_names[fi]}")
        rec(t.children_left[node], indent + 1)
        lines.append(f"{pad}}} else {{")
        rec(t.children_right[node], indent + 1)
        lines.append(f"{pad}}}")

    rec(0, 1)
    lines.append("}")
    return "\n".join(lines)


def leaf_speedups(clf, X, times):
    """Per leaf: geomean over its training samples of T_cuSPARSE_ALG1 / T_predicted_label."""
    leaves = clf.apply(X)
    pred = clf.predict(X)
    out = {}
    for leaf in np.unique(leaves):
        idx = np.where(leaves == leaf)[0]
        s = [times.iloc[i]["cusparse_alg1"] / time_of(times, i, pred[i]) for i in idx]
        out[leaf] = geomean(s)
    return out


def fit_base_model(t):
    cu = t[t["config"] == "cusparse_alg1"]
    vb = np.where(cu["precision"] == "fp64", 8, 4)
    bytes_ = cu["nnz"] * (vb + 4) + (cu["rows"] + 1) * 4 + cu["cols"] * vb + cu["rows"] * vb
    y = cu["median_ms"].to_numpy()
    res = least_squares(lambda p: np.log(np.exp(p[0]) + bytes_.to_numpy() / np.exp(p[1])) - np.log(y),
                        x0=[np.log(0.005), np.log(1.5e8)])
    t0, bpm = np.exp(res.x)
    err = np.exp(np.abs(res.fun))
    return t0, bpm, float(np.median(err)), float(np.quantile(err, 0.9))


def fit_conversion_models(conv):
    ok = conv[(conv["status"] == "ok") & ~conv["kernel"].isin(NO_CONVERSION)]
    models = []
    for (fam, prec), g in ok.groupby(["kernel", "precision"]):
        A = np.column_stack([np.ones(len(g)), g["nnz"], g["rows"]])
        # nnls on relative error: divide rows by y so small and large matrices weigh alike
        y = g["convert_gpu_ms"].to_numpy()
        coef, _ = nnls(A / y[:, None], np.ones(len(y)))
        pred = A @ coef
        models.append((fam, int(prec == "fp64"), *coef, float(np.median(np.maximum(pred / y, y / pred)))))
    return models


def fit_feature_model(feats):
    A = np.column_stack([np.ones(len(feats)), feats["rows"]])
    y = feats["feature_ms"].to_numpy()
    coef, _ = nnls(A / y[:, None], np.ones(len(y)))
    return coef


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spmv", type=Path, nargs="+", default=[Path("results/phase2_spmv.csv"), Path("results/phase3a_extensions.csv")])
    ap.add_argument("--features", type=Path, default=Path("results/phase3/features_train.csv"))
    ap.add_argument("--convert", type=Path, default=Path("results/phase3/gpu_convert_train.csv"))
    ap.add_argument("--manifest", type=Path, default=Path("data/suitesparse/manifest.csv"))
    ap.add_argument("--header", type=Path, default=Path("src/selector/tree_generated.hpp"))
    ap.add_argument("--report", type=Path, default=Path("results/phase3/selector_train.txt"))
    args = ap.parse_args()

    times, feats, group, family, conv, t = load(args)
    feature_names = [c for c in feats.columns if c not in
                     ("feature_ms", "cpu_diff", "rows", "cols", "nnz", "max_len") and not c.startswith(
                         ("gpu", "cc", "driver", "cuda_runtime", "cusparse", "git_", "host", "timestamp"))]
    assert len(feature_names) == 20 and feature_names[0] == "log2_rows" and feature_names[-1] == "fp64", feature_names
    idx = times.index.intersection(feats.index)
    times, feats = times.loc[idx], feats.loc[idx]
    X = feats[feature_names].to_numpy(float)
    groups = np.array([group[m] for m, _ in idx])
    oracle = times.min(axis=1)

    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"training samples: {len(idx)} ({times.index.get_level_values(0).nunique()} matrices x precisions), "
              f"{len(np.unique(groups))} SuiteSparse groups, {times.shape[1]} configs\n")

        # 1. portfolio
        portfolio, history = pick_portfolio(times)
        print("== Portfolio (greedy; geomean slowdown of the portfolio's best vs the full oracle) ==")
        for c, s in history:
            print(f"  + {c:24s} -> {s:.4f}")
        noconv = [c for c in portfolio if family[c] in NO_CONVERSION]
        print(f"no-conversion subset: {noconv}\n")

        labels = times[portfolio].idxmin(axis=1).to_numpy()
        labels_nc = times[noconv].idxmin(axis=1).to_numpy()
        print("label distribution:", pd.Series(labels).value_counts().to_dict(), "\n")

        # 2. trees
        grid = list(itertools.product([2, 3, 4, 5, 6], [2, 4, 8], ["uniform", "regret"]))
        params, table, oof = cross_validate(X, labels, times, groups, portfolio, grid)
        params_nc, _, oof_nc = cross_validate(X, labels_nc, times, groups, noconv, grid)
        print("== Grouped 5-fold CV (full portfolio), top 8 of", len(table), "settings ==")
        print(table.head(8).round(4).to_string(index=False), "\n")

        t_oof = np.array([time_of(times, i, p) for i, p in enumerate(oof)])
        cu = times["cusparse_alg1"].to_numpy()
        single = times.div(oracle, axis=0).apply(geomean).idxmin()
        print("== Out-of-fold comparison (geomean T / T_oracle; lower is better) ==")
        rows = [("always cuSPARSE ALG1", cu), (f"best single config ({single})", times[single].to_numpy()),
                ("oracle within portfolio", times[portfolio].min(axis=1).to_numpy()),
                ("decision tree (CV)", t_oof),
                ("decision tree, no-conversion (CV)", np.array([time_of(times, i, p) for i, p in enumerate(oof_nc)]))]
        for name, tt in rows:
            r = tt / oracle.to_numpy()
            print(f"  {name:36s} {geomean(r):.4f}   within 10%: {np.mean(r <= 1.1):.0%}   "
                  f"vs cuSPARSE: {geomean(cu / tt):.3f}x   slower than cuSPARSE by >5%: {int(np.sum(tt > 1.05 * cu))}")
        big = feats["log2_nnz"].to_numpy() >= np.log2(1e6)
        print(f"  large matrices (>= 1M nnz, n={int(big.sum())}): tree slower than cuSPARSE by >5% on "
              f"{int(np.sum(t_oof[big] > 1.05 * cu[big]))}\n")

        taus = [1.0, 1.02, 1.03, 1.05, 1.08]
        gate = cv_min_gain(X, labels, times, groups, params, portfolio, taus)
        print("== Minimum expected gain tau (grouped CV) ==")
        print(gate.round(4).to_string(index=False))
        # Most accurate tau among those that never lose >5% to cuSPARSE on a large matrix (option (b)).
        safe = gate[gate["large_slower_5pct"] == 0]
        best_gate = (safe if len(safe) else gate).sort_values("cv_geomean_vs_oracle").iloc[0]
        min_gain, large_gate = float(best_gate["tau"]), bool(best_gate["large_gate"])
        expected_gain = max(0.0, 1 - 1 / float(best_gate["vs_cusparse"]))
        print(f"expected (out-of-fold) gain over cuSPARSE for this setting: {expected_gain:.4f}")
        print(f"chosen tau = {min_gain}, large-matrix gate (>= 1M nnz -> cuSPARSE ALG1) = {large_gate}\n")

        conf = pd.crosstab(pd.Series(labels, name="oracle label"), pd.Series(oof, name="predicted (CV)"))
        print("== Confusion matrix (out-of-fold) ==")
        print(conf.to_string(), "\n")

        # 3. final models on all training data
        d, l, wt = params
        clf = fit_tree(X, labels, regret_weight(times, portfolio) if wt == "regret" else None, d, l)
        d2, l2, wt2 = params_nc
        clf_nc = fit_tree(X, labels_nc, regret_weight(times, noconv) if wt2 == "regret" else None, d2, l2)
        print(f"final trees: full {params} ({clf.get_n_leaves()} leaves), no-conversion {params_nc} "
              f"({clf_nc.get_n_leaves()} leaves)")
        imp = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
        print("feature importance (full tree):", {k: round(v, 3) for k, v in imp[imp > 0].items()}, "\n")

        t0, bpm, med_err, p90_err = fit_base_model(t[t.set_index(KEY).index.isin(idx)])
        print(f"cuSPARSE time model: ms = {t0:.4g} + bytes / {bpm:.4g}  (median error x{med_err:.2f}, p90 x{p90_err:.2f})")
        conv_models = fit_conversion_models(conv)
        for m in conv_models:
            print(f"conversion model {m[0]:13s} fp64={m[1]}: ms = {m[2]:.4g} + {m[3]:.3g}*nnz + {m[4]:.3g}*rows  "
                  f"(median error x{m[5]:.2f})")
        fc = fit_feature_model(feats)
        print(f"feature model: ms = {fc[0]:.4g} + {fc[1]:.3g}*rows")
        max_gain = float(max(0.0, (1 - oracle / times["cusparse_alg1"]).max()))
        print(f"max gain over cuSPARSE on training set: {max_gain:.3f} (best config {1 / (1 - max_gain):.2f}x faster)")

    report = buf.getvalue()
    print(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)

    # 4. header
    sp = leaf_speedups(clf, X, times)
    sp_nc = leaf_speedups(clf_nc, X, times)
    names = ", ".join(f'"{n}"' for n in feature_names)
    conv_rows = ",\n    ".join(f'{{"{m[0]}", {m[1]}, {m[2]:.6g}, {m[3]:.6g}, {m[4]:.6g}}}' for m in conv_models)
    header = f"""// GENERATED by scripts/train_selector.py -- do not edit by hand. Regenerate after new training data.
// Portfolio: {', '.join(portfolio)}
// Full tree: max_depth={params[0]}, min_samples_leaf={params[1]}, weighting={params[2]}; grouped-CV geomean
// slowdown vs oracle {table.iloc[0]['cv_geomean_vs_oracle']:.4f}. Leaf values = training geomean speedup vs cuSPARSE ALG1.
#pragma once

namespace rasp::generated {{

inline constexpr int kNumFeatures = {len(feature_names)};
inline constexpr const char* kFeatureNames[kNumFeatures] = {{{names}}};

struct TreeResult {{
  const char* config;
  double speedup;
}};

{emit_tree(clf, feature_names, sp, "tree_full")}

{emit_tree(clf_nc, feature_names, sp_nc, "tree_noconv")}

// cuSPARSE ALG1 time: ms = t0 + bytes / bytes_per_ms
inline constexpr double kBaseT0Ms = {t0:.6g};
inline constexpr double kBaseBytesPerMs = {bpm:.6g};
// GPU conversion time per (family, precision): ms = c0 + c_nnz * nnz + c_rows * rows
struct ConvModel {{
  const char* family;
  int fp64;
  double c0, c_nnz, c_rows;
}};
inline constexpr ConvModel kConvModels[] = {{
    {conv_rows}}};
inline constexpr double kFeatureT0Ms = {fc[0]:.6g};
inline constexpr double kFeaturePerRowMs = {fc[1]:.6g};
inline constexpr double kMaxGain = {max_gain:.6g};
// Out-of-fold expected fraction of time saved vs cuSPARSE ALG1: SpmvPlan extracts features only if
// calls x predicted SpMV time x this exceeds the predicted feature-extraction time.
inline constexpr double kExpectedGain = {expected_gain:.6g};
// Leave cuSPARSE ALG1 only if the tree leaf's training speedup is at least this (chosen by grouped CV).
inline constexpr double kMinGain = {min_gain:.6g};
// If true, matrices with >= 1M nonzeros always use cuSPARSE ALG1 (chosen by grouped CV; option (b) of the plan).
inline constexpr bool kLargeGate = {'true' if large_gate else 'false'};
inline constexpr double kLargeNnz = 1e6;

}}  // namespace rasp::generated
"""
    args.header.write_text(header)
    print(f"wrote {args.header} and {args.report}")


if __name__ == "__main__":
    main()
