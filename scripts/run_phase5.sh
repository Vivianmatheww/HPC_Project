#!/usr/bin/env bash
# Phase 5 on the GPU machine: CG over the SPD set (SpMV side of G7), AMG setup through the out-of-core SpGEMM
# (SpGEMM side of G7), the live-demo transcript, Nsight profiles, analysis. Logged to results/phase5/run.log;
# commit results/phase5/ afterwards (the .ncu-rep / .nsys-rep files stay local).
#
#   bash scripts/run_phase5.sh             # full run (~45-60 min with profiling)
#   QUICK=1 bash scripts/run_phase5.sh     # Laplace-only CG, small AMG, no downloads, no profiling (~3 min)
#   PROFILE=0 bash scripts/run_phase5.sh   # skip Nsight Compute / Systems
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/phase5
mkdir -p "$OUT"
for f in cg.csv amg.csv; do
  if [ -f "$OUT/$f" ]; then
    OLD="$OUT/old-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$OLD"
    mv "$OUT/$f" "$OLD/"
    echo "previous $f moved to $OLD"
  fi
done
exec > >(tee "$OUT/run.log") 2>&1
echo "== Phase 5 run, $(date -Iseconds), git $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
[ -f .venv/bin/activate ] && source .venv/bin/activate
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,temperature.gpu --format=csv

cmake -B build -G Ninja >/dev/null
cmake --build build
ctest --test-dir build --output-on-failure || { echo "TESTS FAILED: fix before benchmarking"; exit 1; }

if [ "${QUICK:-0}" = 1 ]; then
  : >"$OUT/spd_list.txt"
  ./build/cg_solver --laplace 32,64 --precisions fp64 --reps 2 --out "$OUT/cg.csv"
  ./build/amg_setup --n 64 --budget-mb 16 --reps 1 --out "$OUT/amg.csv"
else
  echo "== SPD matrices for CG: every SPD matrix of the earlier sets, plus 20 fetched for Phase 5"
  if [ ! -f data/spd/manifest.csv ]; then
    python scripts/fetch_matrices.py --spd --count 20 --seed 2 --out data/spd \
      --exclude data/suitesparse/manifest.csv data/heldout/manifest.csv
  fi
  python - >"$OUT/spd_list.txt" <<'EOF'
import csv, pathlib
seen = set()
for m in ("data/suitesparse/manifest.csv", "data/heldout/manifest.csv", "data/spd/manifest.csv"):
    p = pathlib.Path(m)
    if not p.exists():
        continue
    for r in csv.DictReader(open(p)):
        if r["isspd"] == "1" and pathlib.Path(r["path"]).exists() and r["path"] not in seen:
            seen.add(r["path"])
            print(r["path"])
EOF
  echo "$(wc -l <"$OUT/spd_list.txt") SPD matrices"
  ./build/cg_solver --list "$OUT/spd_list.txt" --laplace 64,128 --precisions fp64 --reps 5 --out "$OUT/cg.csv"

  echo "== AMG setup (Galerkin triple products) under a fast-memory budget"
  METHODS=alg4,alg4_overlap,rasp_static,rasp,zerocopy,incore
  ./build/amg_setup --n 192 --budget-mb 256 --methods $METHODS --reps 3 --out "$OUT/amg.csv"
  ./build/amg_setup --n 256 --budget-mb 512 --methods $METHODS --reps 3 --out "$OUT/amg.csv"

  echo "== Live-demo transcript (slides)"
  ./build/rasp_demo | tee "$OUT/demo.txt"

  if [ "${PROFILE:-1}" = 1 ]; then
    bash scripts/profile_kernels.sh
  fi
fi

echo "== Analysis"
python scripts/analyze_phase5.py
echo "== Done $(date -Iseconds). Commit results/phase5/."
