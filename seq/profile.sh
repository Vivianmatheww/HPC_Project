#!/usr/bin/env bash
# Hotspot analysis of the sequential implementation: timed runs (optimised build, per-phase wall/user/system time)
# and gprof runs (instrumented build, per-function self time and call counts) of the three workloads.
#
#   bash seq/profile.sh            # ~2 min; needs data/seq/ (python scripts/fetch_matrices.py --count 0 --out data/seq
#                                  #   --names Rajat/rajat23 Sandia/ASIC_100k ACUSIM/Pres_Poisson Norris/torso2
#                                  #   Williams/cop20k_A Schenk_AFE/af_shell1)
# Output: results/seq/{spmv,cg,amg}.{csv,txt}, bandwidth.txt, machine.txt, gprof_<workload>/{flat,callgraph,run}.txt
#   (OUT=dir and DATA=dir override results/seq and data/seq, e.g. for scripts/kaggle_run.sh)
# Then:  python scripts/analyze_seq.py --tex docs/slides/hotspots/generated
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
OUT=$(mkdir -p "${OUT:-$ROOT/results/seq}" && cd "${OUT:-$ROOT/results/seq}" && pwd)
DATA=${DATA:-$ROOT/data/seq}
make -C seq all profile >/dev/null
M=""
for f in Rajat/rajat23 Sandia/ASIC_100k ACUSIM/Pres_Poisson Norris/torso2 Williams/cop20k_A Schenk_AFE/af_shell1; do
  M="$M${M:+,}$DATA/$f.mtx"
done
SPMV=(--matrices "$M" --spmv-reps 50)
CG=(--cg laplace:128 --cg-max-iter 2000)
AMG=(--amg 160)

echo "$(lscpu | sed -n 's/^Model name: *//p'), $(nproc) threads, $(g++ --version | head -1)" > "$OUT/machine.txt"
echo "== machine: $(cat "$OUT/machine.txt")"
echo "== timed runs (optimised build)"
./seq/rasp_seq "${SPMV[@]}" --csv "$OUT/spmv.csv" --label spmv | tee "$OUT/spmv.txt"
./seq/rasp_seq "${CG[@]}" --csv "$OUT/cg.csv" --label cg | tee "$OUT/cg.txt"
./seq/rasp_seq "${AMG[@]}" --csv "$OUT/amg.csv" --label amg | tee "$OUT/amg.txt"

echo "== streaming memory bandwidth (one thread, all threads)"
g++ -O2 -march=native -fopenmp seq/bandwidth.cpp -o seq/bandwidth && ./seq/bandwidth | tee "$OUT/bandwidth.txt"

echo "== gprof runs (instrumented build; AMG repeated 4x for enough samples)"
for w in spmv cg amg; do
  rm -rf "$OUT/gprof_$w" && mkdir -p "$OUT/gprof_$w"
  case $w in
    spmv) args=("${SPMV[@]}") ;;
    cg) args=("${CG[@]}") ;;
    amg) args=("${AMG[@]}" --amg-reps 4) ;;
  esac
  (cd "$OUT/gprof_$w" && "$ROOT/seq/rasp_seq_pg" "${args[@]}" > run.txt && \
   gprof -b -p "$ROOT/seq/rasp_seq_pg" gmon.out > flat.txt && \
   gprof -b -q "$ROOT/seq/rasp_seq_pg" gmon.out > callgraph.txt && rm -f gmon.out)
  echo "-- $w"; sed -n 5,12p "$OUT/gprof_$w/flat.txt" | cut -c1-120
done
