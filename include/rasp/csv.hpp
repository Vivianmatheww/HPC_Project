#pragma once
// CSV output for benchmarks. Each row is a list of key/value pairs; the first row fixes the header.
// Opening an existing file appends (resumable sweeps), but only if its header matches.
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace rasp {

using Fields = std::vector<std::pair<std::string, std::string>>;

class CsvWriter {
 public:
  explicit CsvWriter(std::string path);
  ~CsvWriter();
  CsvWriter(const CsvWriter&) = delete;
  CsvWriter& operator=(const CsvWriter&) = delete;

  void row(const Fields& fields);  // throws if keys differ from the header
  const std::string& path() const { return path_; }

 private:
  std::string path_;
  FILE* f_ = nullptr;
  std::vector<std::string> header_;
  std::string existing_header_;  // header line found in an existing file, if any
};

std::string csv_escape(const std::string& s);
std::string fmt(double v);  // full-precision number formatting for CSV cells

// GPU, driver, CUDA and cuSPARSE versions, git commit, dirty flag, timestamp. Computed once.
const Fields& environment_stamp();

}  // namespace rasp
