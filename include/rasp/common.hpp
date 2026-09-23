#pragma once
// Error-checking macros shared by the whole library. Every CUDA / cuSPARSE call goes through these.
#include <cuda_runtime.h>
#include <cusparse.h>

#include <stdexcept>
#include <string>

namespace rasp {

struct CudaError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

[[noreturn]] inline void throw_cuda(cudaError_t e, const char* expr, const char* file, int line) {
  throw CudaError(std::string(file) + ":" + std::to_string(line) + ": " + expr + " -> " + cudaGetErrorString(e));
}

[[noreturn]] inline void throw_cusparse(cusparseStatus_t s, const char* expr, const char* file, int line) {
  throw CudaError(std::string(file) + ":" + std::to_string(line) + ": " + expr + " -> " + cusparseGetErrorString(s));
}

}  // namespace rasp

#define RASP_CUDA(expr)                                                     \
  do {                                                                      \
    cudaError_t rasp_e_ = (expr);                                           \
    if (rasp_e_ != cudaSuccess) ::rasp::throw_cuda(rasp_e_, #expr, __FILE__, __LINE__); \
  } while (0)

#define RASP_CUSPARSE(expr)                                                 \
  do {                                                                      \
    cusparseStatus_t rasp_s_ = (expr);                                      \
    if (rasp_s_ != CUSPARSE_STATUS_SUCCESS) ::rasp::throw_cusparse(rasp_s_, #expr, __FILE__, __LINE__); \
  } while (0)
