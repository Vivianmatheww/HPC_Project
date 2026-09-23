#pragma once
// Runtime SpMV selection: which kernel configuration to use for a matrix, and whether converting to it pays off.
//
//   v0 AlwaysCusparse  baseline
//   v1 Rules           hand-written thresholds derived from the Phase 2 data
//   v2 Tree            decision tree trained by scripts/train_selector.py and compiled in as
//                      src/selector/tree_generated.hpp (no runtime ML dependency)
//
// The cost models (cuSPARSE time, GPU conversion time, feature-extraction time) are fitted by the same script
// and emitted into the generated header, so the selector and its cost model always come from one training run.
#include <string>

#include "rasp/features.hpp"
#include "rasp/spmv.hpp"

namespace rasp {

enum class SelectorKind { AlwaysCusparse, Rules, Tree };
const char* to_string(SelectorKind k);

struct Choice {
  KernelConfig config;
  double predicted_speedup = 1.0;  // vs cuSPARSE ALG1, steady state
  std::string reason;
};

// allow_conversion = false restricts the choice to formats that need no conversion (cuSPARSE, CSR kernels).
Choice select_rules(const MatrixFeatures& f, bool fp64, bool allow_conversion = true);
Choice select_tree(const MatrixFeatures& f, bool fp64, bool allow_conversion = true);

// Formats whose construction does real work (padding, sorting, binning). cuSPARSE preprocessing (~0.3 SpMV)
// and the plain CSR kernels count as free.
bool needs_conversion(const KernelConfig& c);

// Cost models (milliseconds), fitted on the training set.
double predict_base_ms(int64_t rows, int64_t cols, int64_t nnz, bool fp64);  // one cuSPARSE ALG1 SpMV
double predict_convert_ms(const KernelConfig& c, int64_t rows, int64_t nnz, bool fp64);
double predict_feature_ms(int64_t rows);
double max_gain();       // largest 1 - 1/speedup any configuration achieved on the training set
double expected_gain();  // out-of-fold average 1 - 1/speedup of the tree selector (cross-validated)

// The generated tree's feature list must match feature_names(); returns an error message if it does not.
std::string check_generated_features();

}  // namespace rasp
