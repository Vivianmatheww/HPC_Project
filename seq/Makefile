# Sequential RASP (CPU, one thread, no CUDA).
#   make            optimised build                   -> rasp_seq
#   make profile    gprof build (-pg)                  -> rasp_seq_pg  (run it, then: gprof rasp_seq_pg gmon.out)
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -march=native

all: rasp_seq

rasp_seq: rasp_seq.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

profile: rasp_seq_pg

# No function cloning or splitting in the profiling build: gprof ignores GCC's ".isra"/".constprop"/".part"/".cold"
# symbols and charges their samples and calls to a neighbouring function.
PGFLAGS = -pg -fno-omit-frame-pointer -fno-ipa-sra -fno-ipa-cp -fno-partial-inlining -fno-reorder-blocks-and-partition

rasp_seq_pg: rasp_seq.cpp
	$(CXX) $(CXXFLAGS) $(PGFLAGS) -o $@ $<

clean:
	rm -f rasp_seq rasp_seq_pg bandwidth gmon.out

.PHONY: all profile clean
