#!/usr/bin/env python3
"""Download SuiteSparse matrices for the SpMV experiments.

Selects a reproducible random sample of real, square matrices within an nnz range, plus the
matrices paper 2 (Heller & Oberhuber 2012) discusses by name. Each matrix is saved as
<out>/<group>/<name>.mtx (only the matrix itself; right-hand sides and coordinate files in the
archives are skipped). data/.../manifest.csv records the metadata the selector needs later
(group and kind for grouped cross-validation).

ssgetpy is used only for searching: its own download() sleeps 0.1 s per 4 KB chunk (~40 KB/s),
so this script downloads the archives itself.

Examples:
    python scripts/fetch_matrices.py --count 50 --named          # Phase 1 set
    python scripts/fetch_matrices.py --count 300 --max-nnz 50e6  # Phase 2 sweep
    python scripts/fetch_matrices.py --count 50 --named --dry-run
    python scripts/fetch_matrices.py --count 40 --seed 1 --out data/heldout \
        --exclude data/suitesparse/manifest.csv                   # held-out set for Phase 3
    python scripts/fetch_matrices.py --spd --count 20 --seed 2 --out data/spd \
        --exclude data/suitesparse/manifest.csv data/heldout/manifest.csv   # Phase 5 CG set
    python scripts/fetch_matrices.py --count 0 --names SNAP/com-Amazon SNAP/com-DBLP --out data/snap
"""
import argparse
import csv
import io
import random
import sys
import tarfile
import time
from pathlib import Path

import requests
import ssgetpy

# Matrices named in paper 2's results (circuit wins for ArgCSR; power-flow wins for cuSPARSE/HYB;
# FEM/FD matrices where RgCSR does well), limited to sizes that fit this project's GPU budget.
PAPER2_NAMED = [
    "Rajat/rajat23",            # ArgCSR: desiredChunkSize 1 is 6x faster than 32
    "IBM_EDA/ckt11752_dc_1",    # circuit (IBM EDA)
    "Sandia/ASIC_100k",         # circuit
    "TSOPF/TSOPF_RS_b162_c1",   # power flow: cuSPARSE ~4x faster than ArgCSR
    "TSOPF/TSOPF_FS_b39_c7",    # power flow
    "QY/case39",                # "case39" power-flow set
    "Schenk_AFE/af_shell1",     # AFE: desiredChunkSize 32 beats 1 (18 vs 11 GFLOPS)
    "Norris/torso2",            # RgCSR / Sliced ELL ~2x faster
    "TKK/t2d_q9",               # t2d_q* finite-difference set
    "GHS_indef/ncvxqp9",        # GHS_indef: ArgCSR ~10x cuSPARSE
]

MANIFEST_FIELDS = ["id", "group", "name", "rows", "cols", "nnz", "dtype", "kind", "isspd", "is2d3d",
                   "psym", "nsym", "source", "path"]


def search_pool(min_nnz, max_nnz, square_only):
    pool = ssgetpy.search(nzbounds=(int(min_nnz), int(max_nnz)), dtype="real", limit=100_000)
    return [m for m in pool if not square_only or m.rows == m.cols]


def lookup(spec):
    group, name = spec.split("/")
    hits = [m for m in ssgetpy.search(name, limit=50) if m.group == group and m.name == name]
    return hits[0] if hits else None


def download(m, out_dir, retries=3):
    """Fetch <group>/<name>.tar.gz and extract only <name>/<name>.mtx to out_dir/<group>/<name>.mtx."""
    dest = out_dir / m.group / f"{m.name}.mtx"
    if dest.exists():
        return dest, "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = m.url("MM")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                buf = io.BytesIO()
                for chunk in r.iter_content(chunk_size=1 << 20):
                    buf.write(chunk)
            buf.seek(0)
            with tarfile.open(fileobj=buf, mode="r:gz") as tar:
                member = tar.getmember(f"{m.name}/{m.name}.mtx")
                src = tar.extractfile(member)
                tmp = dest.with_suffix(".mtx.part")
                with open(tmp, "wb") as f:
                    while block := src.read(1 << 20):
                        f.write(block)
                tmp.rename(dest)  # a partial download never looks complete
            return dest, f"{buf.getbuffer().nbytes / 1e6:.1f} MB"
        except (requests.RequestException, tarfile.TarError, KeyError, OSError) as e:
            if attempt == retries:
                raise RuntimeError(f"{m.group}/{m.name}: {e}") from e
            time.sleep(2 * attempt)


def read_manifest(path):
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {(r["group"], r["name"]): r for r in csv.DictReader(f)}


def write_manifest(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for key in sorted(rows):
            w.writerow(rows[key])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/suitesparse", type=Path)
    ap.add_argument("--count", type=int, default=50, help="random matrices to add (named ones are extra)")
    ap.add_argument("--min-nnz", type=float, default=1e4)
    ap.add_argument("--max-nnz", type=float, default=5e6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--named", action="store_true", help="also fetch the paper-2 named matrices")
    ap.add_argument("--allow-rectangular", action="store_true")
    ap.add_argument("--spd", action="store_true", help="sample only symmetric positive definite matrices (CG)")
    ap.add_argument("--names", nargs="*", default=[], help="also fetch these group/name matrices (any dtype)")
    ap.add_argument("--dry-run", action="store_true", help="list the selection without downloading")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="manifest.csv files whose matrices must not be selected (held-out sets)")
    args = ap.parse_args()

    pool = search_pool(args.min_nnz, args.max_nnz, not args.allow_rectangular)
    excluded = set()
    for path in args.exclude:
        excluded |= set(read_manifest(path))
    pool = [m for m in pool if (m.group, m.name) not in excluded and (m.isspd or not args.spd)]
    pool.sort(key=lambda m: m.id)  # stable order so the seed fully determines the sample
    picked = random.Random(args.seed).sample(pool, min(args.count, len(pool)))
    selection = [(m, "random") for m in picked]
    extra = [(spec, "paper2") for spec in PAPER2_NAMED] if args.named else []
    extra += [(spec, "named") for spec in args.names]
    have = {(m.group, m.name) for m, _ in selection}
    for spec, src in extra:
        m = lookup(spec)
        if m is None:
            print(f"warning: {spec} not found in the SuiteSparse index", file=sys.stderr)
        elif (m.group, m.name) not in have:
            selection.append((m, src))
            have.add((m.group, m.name))

    total_nnz = sum(m.nnz for m, _ in selection)
    print(f"pool {len(pool)} matrices -> selected {len(selection)} ({total_nnz / 1e6:.1f} M nnz total)")
    if args.dry_run:
        for m, src in selection:
            print(f"  {src:6s} {m.group}/{m.name:28s} {m.rows:>9d} x {m.cols:<9d} nnz {m.nnz:>10d}  {m.kind}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.csv"
    manifest = read_manifest(manifest_path)
    failures = 0
    for i, (m, src) in enumerate(selection, 1):
        try:
            path, note = download(m, args.out)
        except RuntimeError as e:
            failures += 1
            print(f"[{i}/{len(selection)}] FAILED {e}", file=sys.stderr)
            continue
        print(f"[{i}/{len(selection)}] {m.group}/{m.name} ({m.nnz} nnz) {note}")
        manifest[(m.group, m.name)] = {
            "id": m.id, "group": m.group, "name": m.name, "rows": m.rows, "cols": m.cols, "nnz": m.nnz,
            "dtype": m.dtype, "kind": m.kind, "isspd": int(bool(m.isspd)), "is2d3d": int(bool(m.is2d3d)),
            "psym": m.psym, "nsym": m.nsym, "source": manifest.get((m.group, m.name), {}).get("source", src),
            "path": str(path),
        }
        write_manifest(manifest_path, manifest)  # after every file, so an interrupted run keeps its progress
    print(f"done: {len(manifest)} matrices in {manifest_path}, {failures} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
