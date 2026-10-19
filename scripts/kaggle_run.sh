#!/usr/bin/env bash
# RASP on a Kaggle GPU notebook (Tesla T4): build, run the tests, then the same-machine comparison of the sequential
# version (seq/, one CPU core) with the GPU version on the checkpoint's three workloads (SpMV on the 6 matrices,
# CG on Laplace 128^3, AMG set-up on Laplace 160^3), plus the out-of-core AMG study at n = 192 and the live demo.
#
# In a notebook cell (GPU T4 accelerator, Internet on, the code zip and the matrices zip attached as datasets; both
# zips come from scripts/make_kaggle_zip.sh. Without the matrices zip they are downloaded from SuiteSparse):
#   !bash $(find /kaggle/input -name kaggle_run.sh | head -1) go        # ~40 min; resumes if run again
#   !bash /kaggle/temp/rasp/scripts/kaggle_run.sh status                # one screen: what is done, what failed
#
# Code, build and matrices live in /kaggle/temp (not kept); results go to /kaggle/working/rasp-results/<run>/ and
# /kaggle/working/rasp_<run>.zip, which the notebook's Output panel lets you download. Each finished step leaves a
# marker in <results>/.done, so `go` after a crash or a restart skips what is already done. RUN=name sets <run>
# (default kaggle), ARCH=sm sets the GPU architecture (default 75, the T4), CMAKE_ARGS adds CMake options.
set -uo pipefail
MODE=${1:-go}
RUN=${RUN:-kaggle}
ARCH=${ARCH:-75}
WORK=${WORK:-/kaggle/working}
TMP=${TMP:-/kaggle/temp}
INPUT=${INPUT:-/kaggle/input}
CODE=$TMP/rasp
DATA=$TMP/rasp-data/seq
OUT=$WORK/rasp-results/$RUN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}  # Kaggle gives two T4s; every number here is from one
NAMES="Rajat/rajat23 Sandia/ASIC_100k ACUSIM/Pres_Poisson Norris/torso2 Williams/cop20k_A Schenk_AFE/af_shell1"
mkdir -p "$OUT/.done" "$TMP"

if [ "$MODE" = status ]; then
  echo "results: $OUT"
  for s in deps tests data seq gpu_spmv gpu_cg gpu_amg demo; do
    if [ -f "$OUT/.done/$s" ]; then echo "  done     $s"
    elif [ -f "$OUT/$s.log" ]; then echo "  STARTED  $s (not finished): $(tail -n 1 "$OUT/$s.log")"
    else echo "  pending  $s"; fi
  done
  [ -f "$OUT/comparison.txt" ] && { echo; cat "$OUT/comparison.txt"; }
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader
  df -h "$WORK" | tail -n 1
  exit 0
fi
[ "$MODE" = go ] || { echo "usage: kaggle_run.sh go|status"; exit 2; }

# The dataset is read-only: work on a writable copy. Timestamps are kept, so an unchanged file is not rebuilt.
SRC=$(cd "$(dirname "$0")/.." && pwd)
if [ "$SRC" != "$CODE" ]; then
  mkdir -p "$CODE" && chmod -R u+w "$CODE" && cp -r --preserve=timestamps "$SRC/." "$CODE/" && chmod -R u+w "$CODE" ||
    { echo "!! could not copy the code from $SRC to $CODE"; exit 1; }
fi
cd "$CODE" || exit 1

step() {  # step NAME FUNCTION: run FUNCTION once, its output in $OUT/NAME.log; stop the run if it fails
  local name=$1 fn=$2 t0=$SECONDS
  if [ -f "$OUT/.done/$name" ]; then echo "== $name: already done"; return 0; fi
  echo "== $name: started $(date -u +%H:%M) UTC"
  if "$fn" > "$OUT/$name.log" 2>&1; then
    touch "$OUT/.done/$name"
    echo "   ok in $(( (SECONDS - t0) / 60 )) min $(( (SECONDS - t0) % 60 )) s; last lines of $OUT/$name.log:"
    tail -n 4 "$OUT/$name.log" | sed 's/^/   | /'
  else
    echo "!! $name FAILED after $(( SECONDS - t0 )) s. Last lines of $OUT/$name.log:"
    tail -n 40 "$OUT/$name.log"
    exit 1
  fi
}

deps() {
  python3 -c "import ssgetpy" 2>/dev/null || pip install -q ssgetpy
  command -v ninja >/dev/null || pip install -q ninja
  command -v gprof >/dev/null || { apt-get -qq update && apt-get -qq install -y binutils; }
  command -v gprof
}
build() {  # not a step: incremental, so it is cheap to run every time
  cmake -S . -B "$TMP/rasp-build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
    ${CMAKE_ARGS:-} > "$OUT/build.log" 2>&1 && cmake --build "$TMP/rasp-build" -j "$(nproc)" >> "$OUT/build.log" 2>&1
}
tests() { ctest --test-dir "$TMP/rasp-build" --output-on-failure; }
data() {  # the 6 matrices from an attached dataset (the same files as the laptop runs), else from SuiteSparse
  local src root
  src=$(find "$INPUT" -path "*/Rajat/rajat23.mtx" 2>/dev/null | head -n 1)
  if [ -z "$src" ]; then
    echo "no matrices dataset under $INPUT: downloading from SuiteSparse"
    python3 scripts/fetch_matrices.py --count 0 --out "$DATA" --names $NAMES
    return
  fi
  root=${src%/Rajat/rajat23.mtx}
  echo "matrices from $root"
  mkdir -p "$DATA" && cp -r "$root/." "$DATA/" && chmod -R u+w "$DATA" || return 1
  python3 - "$DATA" <<'PY'  # the manifest's paths point at the laptop's data/seq: point them here
import csv, pathlib, sys
data = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader(open(data / "manifest.csv")))
for r in rows:
    r["path"] = str(data / r["group"] / f"{r['name']}.mtx")
    assert pathlib.Path(r["path"]).exists(), r["path"]
with open(data / "manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} matrices in {data}")
PY
}
seq_run() { OUT=$OUT/seq DATA=$DATA bash seq/profile.sh; }
gpu_spmv() { "$TMP/rasp-build/bench_spmv" --matrices "$DATA" --kernels all --precisions fp64 --out "$OUT/spmv_gpu.csv"; }
gpu_cg() {
  python3 - "$DATA/manifest.csv" > "$OUT/spd_list.txt" <<'EOF'
import csv, sys
for r in csv.DictReader(open(sys.argv[1])):
    if r["isspd"] == "1":
        print(r["path"])
EOF
  "$TMP/rasp-build/cg_solver" --list "$OUT/spd_list.txt" --laplace 128 --precisions fp64 --reps 3 \
    --out "$OUT/cg_gpu.csv"
}
gpu_amg() {
  local B=$TMP/rasp-build/amg_setup
  "$B" --n 160 --budget-mb 8192 --methods incore,rasp --reps 3 --out "$OUT/amg_gpu.csv" &&
  "$B" --n 192 --budget-mb 256 --methods alg4,alg4_overlap,rasp_static,rasp,incore --reps 3 --out "$OUT/amg_gpu.csv"
}
demo() {
  local m="" f
  for f in $NAMES; do m="$m${m:+,}$DATA/$f.mtx"; done
  "$TMP/rasp-build/rasp_demo" --spmv "$m" | tee "$OUT/demo.txt"
}

{
  echo "run $RUN, $(date -u -Iseconds), code $(cat GIT_COMMIT 2>/dev/null || echo unknown)"
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,memory.total --format=csv
  nvcc --version | tail -n 1
  lscpu | sed -n 's/^Model name: *//p'; nproc; free -g | sed -n 2p
} > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"

step deps deps
echo "== build (incremental; log in $OUT/build.log)"
build || { echo "!! build FAILED. Last lines of $OUT/build.log:"; tail -n 40 "$OUT/build.log"; exit 1; }
step tests tests
step data data
step seq seq_run
step gpu_spmv gpu_spmv
step gpu_cg gpu_cg
step gpu_amg gpu_amg
step demo demo

echo "== analysis"
python3 scripts/analyze_seq.py --dir "$OUT/seq" --gpu "$OUT/spmv_gpu.csv" > "$OUT/seq/analysis.log" 2>&1 ||
  echo "   analyze_seq.py failed: see $OUT/seq/analysis.log"
python3 scripts/compare_seq_gpu.py --seq "$OUT/seq" --spmv "$OUT/spmv_gpu.csv" --cg "$OUT/cg_gpu.csv" \
  --amg "$OUT/amg_gpu.csv" | tee "$OUT/comparison.txt"
python3 -c "import shutil, sys; print(shutil.make_archive(sys.argv[1], 'zip', sys.argv[2], sys.argv[3]))" \
  "$WORK/rasp_$RUN" "$WORK/rasp-results" "$RUN"
echo "== Done $(date -u +%H:%M) UTC. Download $WORK/rasp_$RUN.zip from the Output panel."
