// Phase 3 end-to-end evaluation of SpmvPlan: plan construction + N calls, per selector.
//
//   bench_plan --matrices <dir-or-.mtx> [--out results/phase3/plan.csv] [--calls 1,10,100,1000]
//              [--selectors always_cusparse,rules,tree,tree_nobe] [--precisions fp32,fp64] [--reps 20]
//
// tree_nobe = decision tree without the break-even logic (always the steady-state choice): the ablation that
// shows what break-even buys at small N. total_ms = plan_ms + N * run_ms, where plan_ms is the median of 15
// constructions (features + selection + GPU conversion, synchronised) and run_ms the median SpMV time.
// Constructions are interleaved across selectors in a freshly shuffled order every repetition (fixed seed):
// host-side timing under WSL shows ~5-10 us position effects, which a fixed order turns into bias.
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "rasp/csv.hpp"
#include "rasp/io.hpp"
#include "rasp/plan.hpp"
#include "rasp/reference.hpp"
#include "rasp/timer.hpp"

namespace fs = std::filesystem;
using namespace rasp;

namespace {

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  for (size_t p = 0; p <= s.size();) {
    const size_t c = s.find(',', p);
    out.push_back(s.substr(p, c - p));
    if (c == std::string::npos) break;
    p = c + 1;
  }
  return out;
}

PlanOptions options_for(const std::string& sel, int64_t n) {
  PlanOptions o;
  o.expected_calls = n;
  if (sel == "always_cusparse") o.selector = SelectorKind::AlwaysCusparse;
  else if (sel == "rules") o.selector = SelectorKind::Rules;
  else if (sel == "tree") o.selector = SelectorKind::Tree;
  else if (sel == "tree_nobe") {
    o.selector = SelectorKind::Tree;
    o.break_even = false;
  } else throw std::invalid_argument("unknown selector " + sel);
  return o;
}

template <class T>
void run(const std::string& name, const HostCsr<double>& A64, const std::vector<std::string>& selectors,
         const std::vector<int64_t>& calls, const BenchOptions& bench, CsvWriter& csv, int& failures) {
  const bool fp64 = sizeof(T) == 8;
  const HostCsr<T> h = convert_values<T>(A64);
  const auto d = DeviceCsr<T>::upload(h);
  const auto x = random_vector<T>(size_t(h.cols));
  DeviceBuffer<T> dx, dy(size_t(h.rows));
  dx.upload(x);
  const auto ref = cpu_spmv(h, x);
  const double tol = fp64 ? 1e-12 : 1e-5;
  { SpmvPlan<T> warm(d, dx.data(), dy.data(), options_for("tree", 1000000)); }  // lazy CUDA/cuSPARSE init

  for (int64_t n : calls) {
    // 15 reps; in each, every selector once, in a shuffled order. Keep the last plan of each selector.
    std::vector<std::vector<double>> plan_ms(selectors.size());
    std::vector<std::unique_ptr<SpmvPlan<T>>> plans(selectors.size());
    std::vector<size_t> order(selectors.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    static std::mt19937 rng(12345);
    for (int rep = 0; rep < 15; ++rep) {
      std::shuffle(order.begin(), order.end(), rng);
      for (size_t si : order) {
        plans[si].reset();
        plans[si] = std::make_unique<SpmvPlan<T>>(d, dx.data(), dy.data(), options_for(selectors[si], n));
        plan_ms[si].push_back(plans[si]->report().plan_ms);
      }
    }
    for (size_t si = 0; si < selectors.size(); ++si) {
      const auto& sel = selectors[si];
      auto& plan = plans[si];
      RASP_CUDA(cudaMemset(dy.data(), 0xff, dy.bytes()));
      plan->run();
      RASP_CUDA(cudaDeviceSynchronize());
      const double err = rel_l2_error(dy.download(), ref);
      const auto t = time_gpu([&](cudaStream_t s) { plan->run(s); }, nullptr, bench);
      const auto& r = plan->report();
      const double pm = percentile(plan_ms[si], 50);
      if (err > tol) {
        ++failures;
        std::printf("    %s %s N=%lld FAILED err %.3g\n", fp64 ? "fp64" : "fp32", sel.c_str(), (long long)n, err);
      }
      Fields row = {{"matrix", name},
                    {"precision", fp64 ? "fp64" : "fp32"},
                    {"rows", std::to_string(h.rows)},
                    {"nnz", std::to_string(h.nnz)},
                    {"selector", sel},
                    {"calls", std::to_string(n)},
                    {"config", r.config.label()},
                    {"steady_state", r.steady_state.label()},
                    {"used_features", r.used_features ? "1" : "0"},
                    {"predicted_speedup", fmt(r.predicted_speedup)},
                    {"predicted_base_ms", fmt(r.predicted_base_ms)},
                    {"feature_ms", fmt(r.feature_ms)},
                    {"convert_ms", fmt(r.convert_ms)},
                    {"plan_ms", fmt(pm)},
                    {"run_ms", fmt(t.median_ms)},
                    {"total_ms", fmt(pm + double(n) * t.median_ms)},
                    {"rel_err", fmt(err)},
                    {"check", err <= tol ? "pass" : "FAIL"},
                    {"reason", r.reason}};
      for (auto& kv : environment_stamp()) row.push_back(kv);
      csv.row(row);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string matrices, out = "results/phase3/plan.csv";
  std::vector<std::string> selectors{"always_cusparse", "rules", "tree", "tree_nobe"}, precisions{"fp32", "fp64"};
  std::vector<int64_t> calls{1, 10, 100, 1000};
  BenchOptions bench;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&] { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (k == "--matrices") matrices = val();
    else if (k == "--out") out = val();
    else if (k == "--selectors") selectors = split(val());
    else if (k == "--precisions") precisions = split(val());
    else if (k == "--reps") bench.reps = std::stoi(val());
    else if (k == "--calls") {
      calls.clear();
      for (auto& c : split(val())) calls.push_back(std::stoll(c));
    } else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      return 2;
    }
  }
  if (matrices.empty()) {
    std::fprintf(stderr, "usage: bench_plan --matrices <dir|file.mtx> [--out f.csv] [--calls 1,10,100,1000] [--selectors ...]\n");
    return 2;
  }
  if (const auto msg = check_generated_features(); !msg.empty()) {
    std::fprintf(stderr, "generated selector does not match the library: %s\n", msg.c_str());
    return 2;
  }
  std::vector<fs::path> files;
  if (fs::is_regular_file(matrices)) files.emplace_back(matrices);
  else
    for (auto& e : fs::recursive_directory_iterator(matrices))
      if (e.is_regular_file() && e.path().extension() == ".mtx") files.push_back(e.path());
  std::sort(files.begin(), files.end());

  CsvWriter csv(out);
  int failures = 0;
  for (size_t i = 0; i < files.size(); ++i) {
    const std::string name = files[i].parent_path().filename().string() + "/" + files[i].stem().string();
    std::printf("[%zu/%zu] %s\n", i + 1, files.size(), name.c_str());
    try {
      const auto A = read_matrix_market(files[i].string());
      for (auto& p : precisions) {
        if (p == "fp64") run<double>(name, A, selectors, calls, bench, csv, failures);
        else run<float>(name, A, selectors, calls, bench, csv, failures);
      }
    } catch (const std::exception& e) {
      ++failures;
      std::printf("    ERROR %s\n", e.what());
    }
  }
  std::printf("\n%zu matrices, %d failures -> %s\n", files.size(), failures, out.c_str());
  return failures ? 1 : 0;
}
