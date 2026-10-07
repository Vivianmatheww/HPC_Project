cmake_minimum_required(VERSION 3.24)

# RTX 4050 Laptop = sm_89. Override with -DCMAKE_CUDA_ARCHITECTURES=... on other GPUs.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

project(rasp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

# Warnings for host code; -lineinfo lets Nsight Compute map metrics to source lines at no runtime cost.
add_compile_options(
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>)

# ---- core library
add_library(rasp STATIC
  src/core/csr.cpp
  src/core/csv.cpp
  src/core/reference.cpp
  src/core/timer.cu
  src/io/matrix_market.cpp
  src/cusparse/cusparse_ref.cu
  src/formats/ell.cpp
  src/formats/sell.cpp
  src/formats/argcsr.cpp
  src/formats/binned.cpp
  src/formats/convert_gpu.cu
  src/spmv/config.cpp
  src/spmv/kernels.cu
  src/features/features.cu
  src/selector/selector.cpp
  src/selector/plan.cu
  src/spgemm/estimate.cpp
  src/spgemm/gallery.cpp
  src/spgemm/planner.cpp
  src/spgemm/range_allocator.cpp)
target_include_directories(rasp PUBLIC include)
target_link_libraries(rasp PUBLIC CUDA::cudart CUDA::cusparse)
# The SpGEMM flop estimator walks every nonzero of A on the host; OpenMP spreads that over the CPU cores.
find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
  target_link_libraries(rasp PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_definitions(rasp PRIVATE RASP_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# ---- benchmarks
foreach(b bench_spmv bench_features bench_plan)
  add_executable(${b} bench/${b}.cu)
  target_link_libraries(${b} PRIVATE rasp)
endforeach()

# platform study (Phase 0)
add_executable(uvm_probe bench/platform/uvm_probe.cu)
target_link_libraries(uvm_probe PRIVATE CUDA::cudart)

# ---- tests (run with: ctest --test-dir build --output-on-failure)
enable_testing()
foreach(t test_io test_spmv test_spgemm test_harness test_formats test_selector test_spgemm_plan)
  file(GLOB src tests/${t}.c*)
  add_executable(${t} ${src})
  target_link_libraries(${t} PRIVATE rasp)
  target_compile_definitions(${t} PRIVATE RASP_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data")
  add_test(NAME ${t} COMMAND ${t})
endforeach()
