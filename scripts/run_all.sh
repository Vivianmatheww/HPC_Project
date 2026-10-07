#!/usr/bin/env bash
# Phase 6 freeze: rerun every experiment from a clean build on the GPU machine and regenerate every number, table
# and figure of the report and slides. The selector stays frozen as trained (src/selector/tree_generated.hpp):
# evaluation is rerun, training is not.
#
#   git tag v1.0-freeze && bash scripts/run_all.sh           # everything (several hours: the Phase 2 sweep dominates)
#   PHASES="4 5 report" bash scripts/run_all.sh              # a subset (phases: 1 2 3 4 5 report)
#   FORCE=1 bash scripts/run_all.sh                          # allow a dirty tree (the CSVs record git_dirty=1)
#
# Earlier outputs are moved to results/archive-<time>/ (git-ignored; git history keeps the committed ones).
# Afterwards: compare against the previous numbers (git diff docs/report/generated/), update progress.md, commit.
set -euo pipefail
cd "$(dirname "$0")/.."
PHASES=${PHASES:-"1 2 3 4 5 report"}
want() { [[ " $PHASES " == *" $1 "* ]]; }

if [ "${FORCE:-0}" != 1 ] && ! git diff --quiet HEAD; then
  echo "refusing to run on a dirty tree (results would be stamped git_dirty=1); commit first or set FORCE=1"
  exit 1
fi
STAMP=$(date +%Y%m%d-%H%M%S)
ARCHIVE="results/archive-$STAMP"
LOG="results/run_all-$STAMP.log"
mkdir -p results
exec > >(tee "$LOG") 2>&1
echo "== run_all $(date -Iseconds), git $(git describe --tags --always --dirty), phases: $PHASES"
[ -f .venv/bin/activate ] && source .venv/bin/activate
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv

archive() {  # move earlier outputs aside: the CSV writer appends
  for f in "$@"; do
    if [ -e "$f" ]; then
      mkdir -p "$ARCHIVE/$(dirname "$f")"
      mv "$f" "$ARCHIVE/$f"
    fi
  done
}

echo "== Clean build and tests"
rm -rf build
cmake -B build -G Ninja >/dev/null
cmake --build build
ctest --test-dir build --output-on-failure || { echo "TESTS FAILED"; exit 1; }

echo "== Data (cached files are kept)"
python scripts/fetch_matrices.py --count 50 --named --out data/suitesparse
python scripts/fetch_matrices.py --count 40 --seed 1 --out data/heldout --exclude data/suitesparse/manifest.csv
python scripts/fetch_matrices.py --spd --count 20 --seed 2 --out data/spd \
  --exclude data/suitesparse/manifest.csv data/heldout/manifest.csv

if want 1; then
  echo "== Phase 1: cuSPARSE baseline"
  archive results/phase1_cusparse.csv
  ./build/bench_spmv --matrices data/suitesparse --out results/phase1_cusparse.csv
  python scripts/plot_phase1.py
fi

if want 2; then
  echo "== Phase 2: every SpMV configuration (+ the Phase 3a extensions)"
  archive results/phase2_spmv.csv results/phase3a_extensions.csv
  ./build/bench_spmv --matrices data/suitesparse --kernels phase2 --out results/phase2_spmv.csv
  ./build/bench_spmv --matrices data/suitesparse --kernels extensions --out results/phase3a_extensions.csv
  python scripts/analyze_phase2.py
fi

if want 3; then
  echo "== Phase 3: features and GPU conversion (training set), held-out sweep and plans (frozen selector)"
  archive results/phase3/features_train.csv results/phase3/gpu_convert_train.csv results/phase3/heldout_spmv.csv \
    results/phase3/plan_heldout.csv
  ./build/bench_features --matrices data/suitesparse --out results/phase3/features_train.csv \
    --convert-out results/phase3/gpu_convert_train.csv
  ./build/bench_spmv --matrices data/heldout --kernels all --out results/phase3/heldout_spmv.csv
  ./build/bench_plan --matrices data/heldout --out results/phase3/plan_heldout.csv
  python scripts/analyze_phase3.py
fi

if want 4; then bash scripts/run_phase4.sh; fi
if want 5; then bash scripts/run_phase5.sh; fi

if want report; then
  echo "== Report and slides"
  python scripts/make_report_assets.py
  (cd docs/report && latexmk -pdf -interaction=nonstopmode main.tex >/dev/null && echo "report: docs/report/main.pdf") || echo "report build failed (see docs/report/main.log)"
  (cd docs/slides && latexmk -pdf -interaction=nonstopmode slides.tex >/dev/null && echo "slides: docs/slides/slides.pdf") || echo "slides build failed"
  missing=$(grep -o "No value for \`[^']*'" docs/report/main.log | sort -u | wc -l || true)
  echo "report values still missing: $missing (grep 'No value for' docs/report/main.log)"
fi
echo "== Done $(date -Iseconds). Log: $LOG"
