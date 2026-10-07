#!/usr/bin/env bash
# Phase 5 profiles of representative kernels on the GPU machine.
#
# Nsight Compute (metrics only: ncu locks clocks to base and flushes caches, so never use it for timings):
#   spmv_csr_vector_af_shell1    CSR-vector v=4 on af_shell1 (FEM, bandwidth-bound winner)
#   spmv_argcsr_D1_rajat23       ArgCSR B=128, desiredChunkSize 1 on rajat23 (paper 2's robust default)
#   spmv_argcsr_D32_rajat23      ArgCSR B=128, desiredChunkSize 32 on rajat23 (paper 2's 6x slowdown: padding)
#   spmv_cusparse_af_shell1      cuSPARSE ALG1 on af_shell1 (best effort: internal kernel names)
#   spgemm_csr_add_rxa           our partial-product sum kernels inside an out-of-core R*A run
#   spgemm_cusparse_rxa          cuSPARSE SpGEMM kernels inside the same run (best effort)
# Sections: SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, WarpStateStats, LaunchStats. Each report is exported as
# results/phase5/ncu/<name>.csv (raw page); the .ncu-rep files stay local (git-ignored).
#
# Nsight Systems: the out-of-core pipeline (Alg. 4 vs RASP, for the overlap screenshot) and a CG solve.
#
#   bash scripts/profile_kernels.sh
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/phase5
mkdir -p "$OUT/ncu" "$OUT/nsys"
SS=data/suitesparse
SECTIONS="--section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats --section LaunchStats"

profile() {  # name, kernel regex, launches to skip, command...
  local name=$1 regex=$2 skip=$3
  shift 3
  echo "== ncu $name"
  if ncu $SECTIONS --kernel-name "regex:$regex" --launch-skip "$skip" --launch-count 1 -f -o "$OUT/ncu/$name" "$@" \
       >"$OUT/ncu/$name.log" 2>&1; then
    ncu --import "$OUT/ncu/$name.ncu-rep" --csv --page raw >"$OUT/ncu/$name.csv"
  else
    echo "   failed (see $OUT/ncu/$name.log)"
  fi
}

if command -v ncu >/dev/null; then
  # bench_spmv launches every config once to check it, then warm-ups, then timed runs: skip the check launch.
  SPMV="./build/bench_spmv --precisions fp64 --reps 1 --warmup 1 --no-flush --out /tmp/rasp_profile_spmv.csv"
  profile spmv_csr_vector_af_shell1 'csr_vector_kernel' 1 $SPMV --matrices $SS/Schenk_AFE/af_shell1.mtx --kernels csr_vector_v4
  profile spmv_argcsr_D1_rajat23 'argcsr_kernel' 1 $SPMV --matrices $SS/Rajat/rajat23.mtx --kernels argcsr_B128_D1
  profile spmv_argcsr_D32_rajat23 'argcsr_kernel' 1 $SPMV --matrices $SS/Rajat/rajat23.mtx --kernels argcsr_B128_D32
  profile spmv_cusparse_af_shell1 'csrmv|spmv|SpMV' 1 $SPMV --matrices $SS/Schenk_AFE/af_shell1.mtx --kernels cusparse_alg1
  # R*A at n=160 with B (~350 MB) larger than the budget: several B blocks, so partial products are summed.
  SPGEMM="./build/bench_spgemm --study ooc --workloads RxA --n 160 --budget-mb 160 --methods rasp --reps 1 --warmup 0
          --check-rows 10 --out /tmp/rasp_profile_ooc.csv --chunks-out /tmp/rasp_profile_chunks.csv"
  profile spgemm_csr_add_rxa 'add_fill_kernel' 2 $SPGEMM
  profile spgemm_cusparse_rxa 'SpGEMM|spgemm|csrgemm' 5 $SPGEMM
else
  echo "ncu not found: skipping Nsight Compute"
fi

if command -v nsys >/dev/null; then
  echo "== nsys out-of-core pipeline"
  nsys profile --trace cuda,nvtx --force-overwrite true -o "$OUT/nsys/ooc_axp" \
    ./build/bench_spgemm --study ooc --workloads AxP --n 192 --budget-mb 256 --methods alg4,rasp --reps 1 --warmup 0 \
    --out /tmp/rasp_nsys_ooc.csv --chunks-out /tmp/rasp_nsys_chunks.csv >"$OUT/nsys/ooc_axp.log" 2>&1
  nsys stats --report cuda_gpu_trace --format csv --output "$OUT/nsys/ooc_axp" "$OUT/nsys/ooc_axp.nsys-rep" >/dev/null 2>&1
  echo "== nsys CG solve"
  nsys profile --trace cuda,nvtx --force-overwrite true -o "$OUT/nsys/cg_laplace" \
    ./build/cg_solver --laplace 64 --reps 1 --selectors always_cusparse,tree --out /tmp/rasp_nsys_cg.csv \
    >"$OUT/nsys/cg_laplace.log" 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv --output "$OUT/nsys/cg_laplace" "$OUT/nsys/cg_laplace.nsys-rep" \
    >/dev/null 2>&1
  echo "Open $OUT/nsys/ooc_axp.nsys-rep in Nsight Systems and save a screenshot of one Alg. 4 and one RASP run"
  echo "(copy-engine row vs compute row) as docs/report/figures/nsys_overlap.png for the report."
else
  echo "nsys not found: skipping Nsight Systems"
fi
