#!/usr/bin/env python3
"""Regenerates every number, table and figure the report and slides use, from the CSVs in results/.

    python scripts/make_report_assets.py            # -> docs/report/generated/*.tex, docs/report/figures/*.pdf

Runs analyze_phase2..5.py with --tex/--fig-dir (their own results/ outputs go to a temporary directory, so committed
files are never rewritten), and writes platform.tex (the Phase 0 probe) and phase1.tex (the cuSPARSE baseline) here.
Phases whose CSVs do not exist yet are skipped; the report then shows ??key where their numbers belong.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rasp_report import TexWriter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def platform(tex):
    path = ROOT / "results/platform/uvm_probe.csv"
    if not path.exists():
        return
    p = pd.read_csv(path)

    def val(test, loc, metric, nbytes=None):
        s = p[(p.test == test) & (p.location == loc) & (p.metric == metric)]
        if nbytes is not None:
            s = s[s.bytes == nbytes]
        return float(s.value.iloc[-1]) if len(s) else np.nan

    gib4 = 4 << 30
    tex.stat("pf-h2d", val("transfer", "pinned", "H2D"), "{:.1f}")
    tex.stat("pf-d2h", val("transfer", "pinned", "D2H"), "{:.1f}")
    tex.stat("pf-h2d-pageable", val("transfer", "pageable", "H2D"), "{:.1f}")
    tex.stat("pf-bidir", val("transfer", "pinned", "H2D+D2H concurrent"), "{:.1f}")
    tex.stat("pf-device-read", val("placement", "device", "seq read", gib4), "{:.1f}")
    tex.stat("pf-device-copy", val("device", "device", "copy (r+w)"), "{:.0f}")
    tex.stat("pf-zc-read", val("placement", "zero-copy", "seq read", gib4), "{:.1f}")
    tex.stat("pf-uvm-read", val("placement", "managed", "seq read (warm)", gib4), "{:.1f}")
    dev_g = val("placement", "device", "random gather", gib4)
    zc_g = val("placement", "zero-copy", "random gather", gib4)
    tex.stat("pf-gather-device", dev_g, "{:.0f}")
    tex.stat("pf-gather-zc", zc_g, "{:.0f}")
    tex.stat("pf-gather-penalty", dev_g / zc_g, "{:.0f}")
    tex.stat("pf-stream-penalty", val("placement", "device", "seq read", gib4) / val("placement", "zero-copy", "seq read", gib4),
             "{:.0f}")


def phase1(tex):
    """The Phase 1 baseline CSV (cuSPARSE ALG1/ALG2; it predates the status/config columns of later sweeps)."""
    path = ROOT / "results/phase1_cusparse.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    if "error" in d:
        d = d[d.error.isna()]
    tex.stat("p1-matrices", d.matrix.nunique())
    tex.stat("p1-failures", int((d.check != "pass").sum()))
    for prec in ("fp32", "fp64"):
        for alg in ("csr_alg1", "csr_alg2"):
            s = d[(d.precision == prec) & (d.alg == alg)]
            if len(s):
                tex.stat(f"p1-{prec}-{alg}-bw", s.pct_peak_bw.median(), "{:.0f}")
                tex.stat(f"p1-{prec}-{alg}-bw-max", s.pct_peak_bw.max(), "{:.0f}")
    s = d[(d.precision == "fp64") & (d.alg == "csr_alg1")].copy()
    if len(s):
        s["size"] = pd.cut(s.nnz, [0, 1e5, 1e6, 1e9], labels=["lt100k", "100k-1M", "gt1M"])
        for k, v in s.groupby("size", observed=True).pct_peak_bw.median().items():
            tex.stat(f"p1-fp64-bw-{k}", v, "{:.0f}")
        wins = d.pivot_table(index=["matrix", "precision"], columns="alg", values="median_ms")
        tex.stat("p1-alg1-wins-fp64", int((wins.xs("fp64", level="precision").idxmin(axis=1) == "csr_alg1").sum()))
    iqr = (d.p75_ms - d.p25_ms) / d.median_ms
    tex.stat("p1-iqr-median-pct", 100 * iqr.median(), "{:.1f}")


def conversion(tex):
    """GPU conversion cost (Phase 3) in cuSPARSE ALG1 SpMV calls, median per format family over both precisions."""
    conv = ROOT / "results/phase3/gpu_convert_train.csv"
    spmv = [ROOT / "results/phase2_spmv.csv", ROOT / "results/phase3a_extensions.csv"]
    if not conv.exists() or not spmv[0].exists():
        return
    c = pd.read_csv(conv)
    c = c[c.status == "ok"]
    s = pd.concat([pd.read_csv(f) for f in spmv if f.exists()])
    base = s[(s.status == "ok") & (s.config == "cusparse_alg1")].drop_duplicates(["matrix", "precision"])
    j = c.merge(base[["matrix", "precision", "median_ms"]], on=["matrix", "precision"])
    j["calls"] = j.convert_gpu_ms / j.median_ms
    for fam, v in j.groupby("kernel").calls.median().items():
        tex.stat(f"p3-gpu-convert-calls-{fam}", v, "{:.0f}" if v >= 10 else "{:.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, default=ROOT / "docs/report")
    ap.add_argument("--fig-format", default="pdf")
    args = ap.parse_args()
    gen, figs = args.report / "generated", args.report / "figures"
    gen.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    tex = TexWriter(gen / "platform.tex")
    platform(tex)
    phase1(tex)
    conversion(tex)
    print(f"wrote {tex.write()}")

    with tempfile.TemporaryDirectory() as tmp:
        jobs = [
            ("analyze_phase2.py", ["--out", f"{tmp}/p2", "--oracle", f"{tmp}/oracle.csv"]),
            ("analyze_phase3.py", ["--out", f"{tmp}/p3"]),
            ("analyze_phase4.py", ["--out", f"{tmp}/p4"]),
            ("analyze_phase5.py", ["--out", f"{tmp}/p5"]),
        ]
        for script, extra in jobs:
            cmd = [sys.executable, str(ROOT / "scripts" / script), "--tex", str(gen), "--fig-dir", str(figs),
                   "--fig-format", args.fig_format, *extra]
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            status = "ok" if r.returncode == 0 else f"FAILED ({r.returncode})"
            print(f"{script}: {status}")
            if r.returncode != 0:
                print(r.stdout[-2000:], r.stderr[-3000:])
    missing = [p for p in ("phase2", "phase3", "phase4", "phase5") if not (gen / f"{p}.tex").exists()]
    if missing:
        print(f"not generated yet (no results): {', '.join(missing)}; the report shows ??key for their numbers")


if __name__ == "__main__":
    main()
