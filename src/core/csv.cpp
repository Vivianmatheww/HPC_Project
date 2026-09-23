#include <cuda_runtime.h>
#include <cusparse.h>
#include <unistd.h>

#include <array>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "rasp/csv.hpp"

#ifndef RASP_SOURCE_DIR
#define RASP_SOURCE_DIR "."
#endif

namespace fs = std::filesystem;

namespace rasp {
namespace {

std::string join_escaped(const std::vector<std::string>& cells) {
  std::string line;
  for (size_t i = 0; i < cells.size(); ++i) {
    if (i) line += ',';
    line += csv_escape(cells[i]);
  }
  return line;
}

// First line of a shell command's stdout, or "" on failure.
std::string run(const std::string& cmd) {
  std::array<char, 256> buf{};
  std::string out;
  if (FILE* p = popen((cmd + " 2>/dev/null").c_str(), "r")) {
    if (fgets(buf.data(), int(buf.size()), p)) out = buf.data();
    pclose(p);
  }
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
  return out;
}

}  // namespace

std::string csv_escape(const std::string& s) {
  if (s.find_first_of(",\"\n\r") == std::string::npos) return s;
  std::string q = "\"";
  for (char c : s) {
    if (c == '"') q += '"';
    q += c;
  }
  return q + '"';
}

std::string fmt(double v) {
  if (std::isnan(v)) return "nan";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.10g", v);
  return buf;
}

CsvWriter::CsvWriter(std::string path) : path_(std::move(path)) {
  const fs::path p(path_);
  if (p.has_parent_path()) fs::create_directories(p.parent_path());
  if (fs::exists(p) && fs::file_size(p) > 0) {
    std::ifstream in(path_);
    std::getline(in, existing_header_);
  }
  f_ = std::fopen(path_.c_str(), "a");
  if (!f_) throw std::runtime_error("cannot open " + path_ + " for writing");
}

CsvWriter::~CsvWriter() {
  if (f_) std::fclose(f_);
}

void CsvWriter::row(const Fields& fields) {
  std::vector<std::string> keys, values;
  for (auto& [k, v] : fields) {
    keys.push_back(k);
    values.push_back(v);
  }
  if (header_.empty()) {
    header_ = keys;
    const std::string line = join_escaped(header_);
    if (!existing_header_.empty()) {
      if (existing_header_ != line)
        throw std::runtime_error(path_ + ": existing CSV has a different header; use a new output file");
    } else {
      std::fprintf(f_, "%s\n", line.c_str());
    }
  } else if (keys != header_) {
    throw std::runtime_error(path_ + ": row keys differ from the CSV header");
  }
  std::fprintf(f_, "%s\n", join_escaped(values).c_str());
  std::fflush(f_);  // a crash mid-sweep keeps every finished row
}

const Fields& environment_stamp() {
  static const Fields stamp = [] {
    Fields f;
    int dev = 0;
    cudaDeviceProp p{};
    if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess) {
      f.emplace_back("gpu", p.name);
      f.emplace_back("cc", std::to_string(p.major) + "." + std::to_string(p.minor));
    } else {
      f.emplace_back("gpu", "");
      f.emplace_back("cc", "");
    }
    f.emplace_back("driver", run("nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0"));
    int rt = 0;
    cudaRuntimeGetVersion(&rt);
    f.emplace_back("cuda_runtime", std::to_string(rt / 1000) + "." + std::to_string((rt % 1000) / 10));
    int major = 0, minor = 0, patch = 0;
    cusparseGetProperty(MAJOR_VERSION, &major);
    cusparseGetProperty(MINOR_VERSION, &minor);
    cusparseGetProperty(PATCH_LEVEL, &patch);
    f.emplace_back("cusparse", std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch));

    const std::string git = std::string("git -C \"") + RASP_SOURCE_DIR + "\" ";
    f.emplace_back("git_commit", run(git + "rev-parse --short HEAD"));
    f.emplace_back("git_dirty", run(git + "status --porcelain --untracked-files=no").empty() ? "0" : "1");

    char host[256] = {};
    gethostname(host, sizeof(host) - 1);
    f.emplace_back("host", host);
    char ts[64];
    const std::time_t now = std::time(nullptr);
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S%z", std::localtime(&now));
    f.emplace_back("timestamp", ts);
    return f;
  }();
  return stamp;
}

}  // namespace rasp
