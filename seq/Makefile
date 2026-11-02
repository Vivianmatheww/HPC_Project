# Sequential RASP (CPU, one thread, no CUDA) and its OpenMP version.
#   make            optimised build                   -> rasp_seq
#   make profile    gprof build (-pg)                  -> rasp_seq_pg  (run it, then: gprof rasp_seq_pg gmon.out)
#   make omp        OpenMP build                       -> rasp_omp
#   make omp-check  rasp_omp --verify at 1, 2 and all threads (under 30 s)
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -march=native

all: rasp_seq

rasp_seq: rasp_seq.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

omp: rasp_omp

rasp_omp: rasp_omp.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $<

# The two matrices are checked too when data/seq/ is there; CG and AMG generate their own inputs.
CHECK_MATRICES = $(wildcard ../data/seq/ACUSIM/Pres_Poisson.mtx ../data/seq/Rajat/rajat23.mtx)
comma = ,
empty =
space = $(empty) $(empty)
omp-check: rasp_omp
	@for t in 1 2 $$(nproc); do \
	  ./rasp_omp --threads $$t --verify $(if $(CHECK_MATRICES),--matrices $(subst $(space),$(comma),$(CHECK_MATRICES))) \
	    --cg laplace:48 --amg 64 | grep -E '^(OpenMP|verify)' || exit 1; \
	done

profile: rasp_seq_pg

# No function cloning or splitting in the profiling build: gprof ignores GCC's ".isra"/".constprop"/".part"/".cold"
# symbols and charges their samples and calls to a neighbouring function.
PGFLAGS = -pg -fno-omit-frame-pointer -fno-ipa-sra -fno-ipa-cp -fno-partial-inlining -fno-reorder-blocks-and-partition

rasp_seq_pg: rasp_seq.cpp
	$(CXX) $(CXXFLAGS) $(PGFLAGS) -o $@ $<

clean:
	rm -f rasp_seq rasp_seq_pg rasp_omp bandwidth gmon.out

.PHONY: all profile omp omp-check clean
