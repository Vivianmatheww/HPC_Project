#!/usr/bin/env bash
# Phase 4 on the GPU machine (RTX 4050 laptop): build, all tests, placement study (G5), out-of-core sweep (G6),
# analysis. Everything is logged to results/phase4/run.log; commit results/phase4/ afterwards.
#
#   bash scripts/run_phase4.sh            # full run (~30-40 min)
#   QUICK=1 bash scripts/run_phase4.sh    # small sizes and 3 repetitions: a smoke test of the whole pipeline (~3 min)
#   NSYS=1 bash scripts/run_phase4.sh     # also an Nsight Systems trace of one Alg. 4 vs RASP run (overlap check)
#
# Laptop hygiene (plan.md §3): AC power, Windows "Best performance", nothing else running.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/phase4
mkdir -p "$OUT"
LOG="$OUT/run.log"

# The CSV writer appends: move earlier outputs aside so a rerun never mixes old and new rows.
if ls "$OUT"/*.csv >/dev/null 2>&1; then
  OLD="$OUT/old-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$OLD"
  mv "$OUT"/*.csv "$OLD"/
  echo "previous CSVs moved to $OLD (delete it once the new run is committed)"
fi
exec > >(tee "$LOG") 2>&1
echo "== Phase 4 run, $(date -Iseconds), git $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
[ -f .venv/bin/activate ] && source .venv/bin/activate
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv

cmake -B build -G Ninja >/dev/null
cmake --build build
ctest --test-dir build --output-on-failure || { echo "TESTS FAILED: fix before benchmarking"; exit 1; }

if [ "${QUICK:-0}" = 1 ]; then
  PL_N=48,64; OOC_N=64,96,128; RATIO_N=96; BUDGET=64; REPS=3; WARM=1
else
  PL_N=64,96,128,160; OOC_N=128,160,192,224,256; RATIO_N=192; BUDGET=512; REPS=10; WARM=2
fi

echo "== Plans the planners would make (prior parameters)"
./build/bench_spgemm --study plans --n "$OOC_N" --budget-mb "$BUDGET"

echo "== Placement study (paper 1, Table 3)"
./build/bench_spgemm --study placement --n "$PL_N" --precisions fp64 --reps "$REPS" --warmup "$WARM" --out "$OUT/placement.csv"
./build/bench_spgemm --study placement --n "${PL_N##*,}" --precisions fp32 --reps "$REPS" --warmup "$WARM" --out "$OUT/placement.csv"

echo "== Out-of-core: fixed budget, grid size swept (with in-core and device-only references)"
./build/bench_spgemm --study ooc --workloads AxP,RxA --n "$OOC_N" --budget-mb "$BUDGET" --reps "$REPS" --warmup "$WARM" \
  --out "$OUT/ooc.csv" --chunks-out "$OUT/chunks.csv"

echo "== Out-of-core: fixed size, budget swept"
./build/bench_spgemm --study ooc --workloads AxP,RxA --n "$RATIO_N" --ratios 0.5,1,1.5,2,3,4 \
  --methods alg4,alg4_overlap,rasp_static,rasp,zerocopy --reps "$REPS" --warmup "$WARM" \
  --out "$OUT/ooc.csv" --chunks-out "$OUT/chunks.csv"

if [ "${NSYS:-0}" = 1 ] && command -v nsys >/dev/null; then
  echo "== Nsight Systems trace: H2D on the copy stream must overlap cuSPARSE on the compute stream"
  nsys profile --trace cuda,nvtx --force-overwrite true -o "$OUT/nsys_ooc" \
    ./build/bench_spgemm --study ooc --workloads AxP --n "$RATIO_N" --budget-mb "$BUDGET" --methods alg4,rasp \
    --reps 1 --warmup 0 --out /tmp/rasp_nsys_ooc.csv --chunks-out /tmp/rasp_nsys_chunks.csv
  nsys stats --report cuda_gpu_trace --format csv --output "$OUT/nsys_ooc" "$OUT/nsys_ooc.nsys-rep" >/dev/null || true
  echo "open $OUT/nsys_ooc.nsys-rep in Nsight Systems for the timeline screenshot (report figure)"
fi

echo "== Analysis"
python scripts/analyze_phase4.py
echo "== Done $(date -Iseconds). Commit results/phase4/ (CSVs, PNGs, summary.txt, run.log)."
