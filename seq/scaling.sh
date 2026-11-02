#!/usr/bin/env bash
# Thread sweep of the OpenMP version: the three timed workloads of seq/profile.sh (same matrices and sizes) at every
# thread count, so the speedup over the sequential run is on the same machine. No gprof (it does not understand threads).
#
#   bash seq/scaling.sh            # ~5-10 min; needs data/seq/ like seq/profile.sh
#   OUT=dir DATA=dir THREADS="1 2 4" bash seq/scaling.sh
# Output: $OUT/machine.txt and $OUT/t<T>/{spmv,cg,amg}.{csv,txt} (machine.txt copied into each t<T>/). At the largest T
#   also $OUT/t<T>/spmv_{static,dynamic,guided}.{csv,txt}: the same SpMV run with that loop schedule (spmv.* is static).
# Then:  python scripts/analyze_omp.py --omp $OUT --seq results/seq
# Defaults: OUT=results/omp, DATA=data/seq, THREADS = 1 2 4 8 ... up to nproc, plus nproc itself.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
OUT=$(mkdir -p "${OUT:-$ROOT/results/omp}" && cd "${OUT:-$ROOT/results/omp}" && pwd)
DATA=${DATA:-$ROOT/data/seq}
if [ -z "${THREADS:-}" ]; then
  N=$(nproc)
  THREADS=""
  for ((t = 1; t < N; t *= 2)); do THREADS="$THREADS $t"; done
  THREADS="$THREADS $N"
fi
make -C seq omp >/dev/null
M=""
for f in Rajat/rajat23 Sandia/ASIC_100k ACUSIM/Pres_Poisson Norris/torso2 Williams/cop20k_A Schenk_AFE/af_shell1; do
  M="$M${M:+,}$DATA/$f.mtx"
done
SPMV=(--matrices "$M" --spmv-reps 50)
CG=(--cg laplace:128 --cg-max-iter 2000)
AMG=(--amg 160)
export OMP_PROC_BIND=close OMP_PLACES=cores  # T threads on T different cores

echo "$(lscpu | sed -n 's/^Model name: *//p'), $(nproc) threads, $(g++ --version | head -1), OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES" > "$OUT/machine.txt"
echo "== machine: $(cat "$OUT/machine.txt")"
MAXT=$(echo $THREADS | tr ' ' '\n' | sort -n | tail -1)
for T in $THREADS; do
  D="$OUT/t$T"
  mkdir -p "$D" && cp "$OUT/machine.txt" "$D/"
  echo "== $T thread(s)"
  ./seq/rasp_omp --threads "$T" "${SPMV[@]}" --csv "$D/spmv.csv" --label "t$T" | tee "$D/spmv.txt"
  ./seq/rasp_omp --threads "$T" "${CG[@]}" --csv "$D/cg.csv" --label "t$T" | tee "$D/cg.txt"
  ./seq/rasp_omp --threads "$T" "${AMG[@]}" --csv "$D/amg.csv" --label "t$T" | tee "$D/amg.txt"
  if [ "$T" = "$MAXT" ]; then
    for sched in dynamic,64 guided; do  # static is spmv.* above
      k=${sched%%,*}
      echo "-- SpMV, schedule $sched"
      ./seq/rasp_omp --threads "$T" --schedule "$sched" "${SPMV[@]}" --csv "$D/spmv_$k.csv" --label "t$T" | tee "$D/spmv_$k.txt"
    done
    cp "$D/spmv.csv" "$D/spmv_static.csv" && cp "$D/spmv.txt" "$D/spmv_static.txt"
  fi
done
