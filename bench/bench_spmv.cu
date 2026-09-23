// SpMV benchmark over a set of Matrix Market files, for any mix of kernel configurations.
//
//   bench_spmv --matrices <dir-or-.mtx> [--out results/phase1_cusparse.csv] [--kernels cusparse|all|<labels>]
//              [--reps 20] [--warmup 5] [--no-flush] [--precisions fp32,fp64] [--max-nnz N] [--peak-bw 187.8]
//
// --kernels: "cusparse" (Phase 1 baseline, default), "all" (the 33 Phase 2 configs) or comma-separated
// labels such as csr_vector_v8,sell_C32_s256,argcsr_B128_D1.
//
// One CSV row per (matrix, precision, config). Every result is checked against the CPU reference
// (relative L2 error; fp64 <= 1e-12, fp32 <= 1e-5). Formats that would need too much padding are recorded
// with status "infeasible"; load/run failures with status "error". The sweep always continues.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "rasp/csv.hpp"
#include "rasp/formats.hpp"
#include "rasp/io.hpp"
#include "rasp/reference.hpp"
#include "rasp/spmv.hpp"
#include "rasp/timer.hpp"

namespace fs = std::filesystem;
using namespace rasp;

namespace {

struct Args {
  std::string matrices, out = "results/phase1_cusparse.csv";
  BenchOptions bench;
  std::vector<std::string> precisions{"fp32", "fp64"};
  std::vector<KernelConfig> kernels{KernelConfig::parse("cusparse_alg1"), KernelConfig::parse("cusparse_alg2")};
  int64_t max_nnz = 0;       // 0 = no limit
  double peak_bw = 187.8;    // GB/s, measured device read bandwidth (uvm_probe, progress.md 2026-09-23)
  bool verbose = false;      // one line per config (default: one line per matrix unless a check fails)
};

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  size_t start = 0;
  while (start <= s.size()) {
    const size_t comma = s.find(',', start);
    out.push_back(s.substr(start, comma - start));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return out;
}

[[noreturn]] void usage(const char* msg) {
  std::fprintf(stderr, "error: %s\nusage: bench_spmv --matrices <dir|file.mtx> [--out f.csv] [--reps N] [--warmup N] "
                       "[--no-flush] [--precisions fp32,fp64] [--kernels cusparse|all|labels] [--max-nnz N] [--peak-bw GB/s]\n", msg);
  std::exit(2);
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) usage(("missing value for " + k).c_str());
      return argv[++i];
    };
    if (k == "--matrices") a.matrices = val();
    else if (k == "--out") a.out = val();
    else if (k == "--reps") a.bench.reps = std::stoi(val());
    else if (k == "--warmup") a.bench.warmup = std::stoi(val());
    else if (k == "--no-flush") a.bench.flush_l2 = false;
    else if (k == "--precisions") a.precisions = split(val());
    else if (k == "--max-nnz") a.max_nnz = int64_t(std::stod(val()));
    else if (k == "--peak-bw") a.peak_bw = std::stod(val());
    else if (k == "-v" || k == "--verbose") a.verbose = true;
    else if (k == "--kernels") {
      const std::string v = val();
      a.kernels.clear();
      if (v == "all") a.kernels = phase2_configs();
      else if (v == "cusparse") a.kernels = {KernelConfig::parse("cusparse_alg1"), KernelConfig::parse("cusparse_alg2")};
      else
        for (auto& s : split(v)) {
          try {
            a.kernels.push_back(KernelConfig::parse(s));
          } catch (const std::exception& e) {
            usage(e.what());
          }
        }
    } else usage(("unknown option " + k).c_str());
  }
  if (a.matrices.empty()) usage("--matrices is required");
  for (auto& p : a.precisions)
    if (p != "fp32" && p != "fp64") usage(("unknown precision " + p).c_str());
  return a;
}

// All .mtx files under a directory (sorted), or the single file given.
std::vector<fs::path> find_matrices(const std::string& root) {
  std::vector<fs::path> files;
  if (fs::is_regular_file(root)) {
    files.emplace_back(root);
  } else if (fs::is_directory(root)) {
    for (auto& e : fs::recursive_directory_iterator(root))
      if (e.is_regular_file() && e.path().extension() == ".mtx") files.push_back(e.path());
    std::sort(files.begin(), files.end());
  } else {
    usage(("not a file or directory: " + root).c_str());
  }
  return files;
}

// Fixed column order so success and error rows share one header.
const std::vector<std::string> kColumns = {
    "matrix", "path", "rows", "cols", "nnz", "precision", "kernel", "config", "flush_l2", "warmup", "reps",
    "median_ms", "p25_ms", "p75_ms", "min_ms", "gflops", "eff_bw_gbs", "pct_peak_bw", "rel_err", "check",
    "convert_ms", "format_bytes", "stored_ratio", "load_ms", "status", "error"};

Fields make_row(const std::map<std::string, std::string>& v) {
  Fields f;
  for (auto& k : kColumns) {
    auto it = v.find(k);
    f.emplace_back(k, it == v.end() ? "" : it->second);
  }
  for (auto& kv : environment_stamp()) f.push_back(kv);
  return f;
}

struct Summary {
  int rows_ok = 0, check_fail = 0, errors = 0, infeasible = 0;
};

template <class T>
void run_precision(const HostCsr<double>& A64, const std::map<std::string, std::string>& base, const Args& args,
                   CsvWriter& csv, Summary& sum) {
  const char* prec = sizeof(T) == 8 ? "fp64" : "fp32";
  const double tol = sizeof(T) == 8 ? 1e-12 : 1e-5;
  const HostCsr<T> h = convert_values<T>(A64);
  const auto d = DeviceCsr<T>::upload(h);
  const auto x = random_vector<T>(size_t(h.cols));
  DeviceBuffer<T> dx, dy(size_t(h.rows));
  dx.upload(x);
  const auto ref = cpu_spmv(h, x);

  // Minimum traffic: values + column indices + row pointers + x + y.
  const double bytes = double(h.nnz) * double(sizeof(T) + sizeof(int32_t)) + double(h.rows + 1) * sizeof(int32_t) +
                       double(h.cols) * sizeof(T) + double(h.rows) * sizeof(T);

  for (const KernelConfig& cfg : args.kernels) {
    auto row = base;
    row["precision"] = prec;
    row["kernel"] = cfg.family();
    row["config"] = cfg.label();
    row["flush_l2"] = args.bench.flush_l2 ? "1" : "0";
    row["warmup"] = std::to_string(args.bench.warmup);
    row["reps"] = std::to_string(args.bench.reps);
    try {
      RASP_CUDA(cudaMemset(dy.data(), 0xff, dy.bytes()));  // NaN garbage: unwritten rows fail the check
      auto k = make_spmv_kernel(cfg, h, d, dx.data(), dy.data());
      k->run(nullptr);
      RASP_CUDA(cudaDeviceSynchronize());
      RASP_CUDA(cudaGetLastError());
      const double err = rel_l2_error(dy.download(), ref);
      const auto t = time_gpu([&](cudaStream_t s) { k->run(s); }, nullptr, args.bench);

      const double sec = t.median_ms * 1e-3;
      const double bw = bytes / sec / 1e9;
      const bool ok = err <= tol;
      row["median_ms"] = fmt(t.median_ms);
      row["p25_ms"] = fmt(t.p25_ms);
      row["p75_ms"] = fmt(t.p75_ms);
      row["min_ms"] = fmt(t.min_ms);
      row["gflops"] = fmt(2.0 * double(h.nnz) / sec / 1e9);
      row["eff_bw_gbs"] = fmt(bw);
      row["pct_peak_bw"] = fmt(100.0 * bw / args.peak_bw);
      row["rel_err"] = fmt(err);
      row["check"] = ok ? "pass" : "FAIL";
      row["convert_ms"] = fmt(k->convert_ms());
      row["format_bytes"] = std::to_string(k->format_bytes());
      row["stored_ratio"] = fmt(h.nnz ? double(k->stored_slots()) / double(h.nnz) : 1.0);
      row["status"] = "ok";
      csv.row(make_row(row));
      ++sum.rows_ok;
      if (!ok) ++sum.check_fail;
      if (!ok || args.verbose)
        std::printf("    %s %-16s %9.4f ms %7.2f GFLOP/s %5.1f%% BW  err %.1e%s\n", prec, cfg.label().c_str(),
                    t.median_ms, 2.0 * double(h.nnz) / sec / 1e9, 100.0 * bw / args.peak_bw, err,
                    ok ? "" : "  <-- FAIL");
    } catch (const FormatInfeasible& e) {
      row["status"] = "infeasible";
      row["error"] = e.what();
      csv.row(make_row(row));
      ++sum.infeasible;
    } catch (const std::exception& e) {
      cudaGetLastError();
      row["status"] = "error";
      row["error"] = e.what();
      csv.row(make_row(row));
      ++sum.errors;
      std::printf("    %s %-16s ERROR %s\n", prec, cfg.label().c_str(), e.what());
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse(argc, argv);
  const auto files = find_matrices(args.matrices);
  if (files.empty()) usage("no .mtx files found");
  CsvWriter csv(args.out);
  Summary sum;
  int skipped = 0;

  const auto& env = environment_stamp();
  std::printf("%s | cuSPARSE %s | git %s | %zu matrices x %zu configs | reps %d, warmup %d, L2 flush %s\n",
              env[0].second.c_str(), env[4].second.c_str(), env[5].second.c_str(), files.size(), args.kernels.size(),
              args.bench.reps, args.bench.warmup, args.bench.flush_l2 ? "on" : "off");

  for (size_t i = 0; i < files.size(); ++i) {
    const auto& path = files[i];
    const std::string name = path.parent_path().filename().string() + "/" + path.stem().string();
    std::map<std::string, std::string> base{{"matrix", name}, {"path", path.string()}};
    try {
      const auto t0 = std::chrono::steady_clock::now();
      const HostCsr<double> A = read_matrix_market(path.string());
      const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (args.max_nnz > 0 && A.nnz > args.max_nnz) {
        ++skipped;
        std::printf("[%zu/%zu] %s  nnz %lld > max, skipped\n", i + 1, files.size(), name.c_str(), (long long)A.nnz);
        continue;
      }
      base["rows"] = std::to_string(A.rows);
      base["cols"] = std::to_string(A.cols);
      base["nnz"] = std::to_string(A.nnz);
      base["load_ms"] = fmt(load_ms);
      base["status"] = "ok";
      std::printf("[%zu/%zu] %s  %d x %d  nnz %lld  (load %.0f ms)\n", i + 1, files.size(), name.c_str(), A.rows, A.cols,
                  (long long)A.nnz, load_ms);
      for (auto& p : args.precisions) {
        if (p == "fp32") run_precision<float>(A, base, args, csv, sum);
        else run_precision<double>(A, base, args, csv, sum);
      }
    } catch (const std::exception& e) {
      cudaGetLastError();
      base["status"] = "error";
      base["error"] = e.what();
      csv.row(make_row(base));
      ++sum.errors;
      std::printf("[%zu/%zu] %s  ERROR %s\n", i + 1, files.size(), name.c_str(), e.what());
    }
  }

  std::printf("\n%d result rows, %d correctness failures, %d infeasible, %d errors, %d skipped -> %s\n", sum.rows_ok,
              sum.check_fail, sum.infeasible, sum.errors, skipped, args.out.c_str());
  return sum.check_fail == 0 && sum.errors == 0 ? 0 : 1;
}
