// Timer, percentile and CSV writer.
#include <filesystem>
#include <fstream>
#include <unistd.h>

#include "rasp/csr.hpp"
#include "rasp/csv.hpp"
#include "rasp/timer.hpp"
#include "test_util.hpp"

using namespace rasp;

TEST(percentile_interpolates) {
  CHECK_NEAR(percentile({4, 1, 3, 2}, 50), 2.5, 1e-12);
  CHECK_NEAR(percentile({4, 1, 3, 2}, 0), 1, 1e-12);
  CHECK_NEAR(percentile({4, 1, 3, 2}, 100), 4, 1e-12);
  CHECK_NEAR(percentile({7}, 25), 7, 1e-12);
  CHECK_NEAR(percentile({}, 50), 0, 1e-12);
}

TEST(time_gpu_stats_are_ordered) {
  DeviceBuffer<char> buf(size_t(64) << 20);
  BenchOptions opt;
  opt.warmup = 2;
  opt.reps = 9;
  int calls = 0;
  auto s = time_gpu([&](cudaStream_t st) { ++calls; RASP_CUDA(cudaMemsetAsync(buf.data(), 1, buf.size(), st)); }, nullptr, opt);
  CHECK(calls == 11);
  CHECK(s.reps == 9);
  CHECK(s.min_ms > 0);
  CHECK(s.min_ms <= s.p25_ms && s.p25_ms <= s.median_ms && s.median_ms <= s.p75_ms && s.p75_ms <= s.max_ms);
  // 64 MiB memset at ~150+ GB/s is well under 10 ms; catches unit mistakes (s vs ms).
  CHECK(s.median_ms < 10);
}

TEST(csv_header_escaping_and_append) {
  const auto path = (std::filesystem::temp_directory_path() / ("rasp_csv_" + std::to_string(getpid()) + ".csv")).string();
  std::filesystem::remove(path);
  {
    CsvWriter w(path);
    w.row({{"name", "a,b"}, {"value", fmt(1.5)}});
    w.row({{"name", "say \"hi\""}, {"value", fmt(2)}});
    CHECK_THROWS(w.row({{"other", "x"}, {"value", "1"}}));
  }
  {
    CsvWriter w(path);  // reopening appends under the same header
    w.row({{"name", "c"}, {"value", fmt(3)}});
  }
  {
    CsvWriter w(path);  // a different header is refused
    CHECK_THROWS(w.row({{"x", "1"}}));
  }
  std::ifstream in(path);
  std::vector<std::string> lines;
  for (std::string l; std::getline(in, l);) lines.push_back(l);
  CHECK(lines.size() == 4);
  if (lines.size() == 4) {
    CHECK(lines[0] == "name,value");
    CHECK(lines[1] == "\"a,b\",1.5");
    CHECK(lines[2] == "\"say \"\"hi\"\"\",2");
    CHECK(lines[3] == "c,3");
  }
  std::filesystem::remove(path);
}

TEST(environment_stamp_has_core_fields) {
  const auto& env = environment_stamp();
  auto get = [&](const std::string& k) {
    for (auto& [key, v] : env)
      if (key == k) return v;
    return std::string("<missing>");
  };
  CHECK(!get("gpu").empty() && get("gpu") != "<missing>");
  CHECK(get("cusparse") != "<missing>");
  CHECK(get("git_commit") != "<missing>");
  CHECK(get("timestamp").size() >= 19);
}

RASP_TEST_MAIN
