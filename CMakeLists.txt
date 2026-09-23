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

# ---- platform study (Phase 0)
add_executable(uvm_probe bench/platform/uvm_probe.cu)
target_link_libraries(uvm_probe PRIVATE CUDA::cudart)
