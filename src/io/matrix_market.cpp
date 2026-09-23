#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "rasp/io.hpp"

namespace fs = std::filesystem;

namespace rasp {
namespace {

[[noreturn]] void fail(const std::string& path, const std::string& msg) {
  throw std::runtime_error(path + ": " + msg);
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) fail(path, "cannot open file");
  f.seekg(0, std::ios::end);
  const auto size = f.tellg();
  f.seekg(0);
  std::string s(size_t(size), '\0');
  f.read(s.data(), size);
  if (!f) fail(path, "read error");
  return s;
}

// Tokenizer over the whole file. The std::string buffer is NUL-terminated, which strtod relies on.
struct Parser {
  const char* p;
  const char* end;
  const std::string& path;
  int64_t line;

  void skip_ws() {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) {
      if (*p == '\n') ++line;
      ++p;
    }
  }
  [[noreturn]] void error(const char* what) { fail(path, std::string("expected ") + what + " at line " + std::to_string(line)); }

  int64_t integer() {
    skip_ws();
    if (p < end && *p == '+') ++p;
    int64_t v = 0;
    auto [q, ec] = std::from_chars(p, end, v);
    if (ec != std::errc()) error("integer");
    p = q;
    return v;
  }
  double real() {
    skip_ws();
    if (p < end && *p == '+') ++p;
    double v = 0;
    auto [q, ec] = std::from_chars(p, end, v);
    if (ec == std::errc::result_out_of_range) {  // subnormal/overflowing literal: let strtod round it
      char* q2 = nullptr;
      v = std::strtod(p, &q2);
      q = q2;
    } else if (ec != std::errc()) {
      error("number");
    }
    p = q;
    return v;
  }
};

struct Header {
  bool pattern = false;
  enum { General, Symmetric, Skew } symmetry = General;
};

Header parse_banner(const std::string& path, const std::string& banner) {
  std::string lower = banner;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  std::istringstream ss(lower);
  std::string magic, object, format, field, symmetry;
  ss >> magic >> object >> format >> field >> symmetry;
  if (magic != "%%matrixmarket") fail(path, "missing %%MatrixMarket banner");
  if (object != "matrix") fail(path, "unsupported object '" + object + "'");
  if (format == "array") fail(path, "dense 'array' format is not supported");
  if (format != "coordinate") fail(path, "unsupported format '" + format + "'");

  Header h;
  if (field == "pattern") h.pattern = true;
  else if (field != "real" && field != "double" && field != "integer")
    fail(path, "unsupported field '" + field + "' (only real, integer, pattern)");

  if (symmetry == "symmetric") h.symmetry = Header::Symmetric;
  else if (symmetry == "skew-symmetric") h.symmetry = Header::Skew;
  else if (symmetry != "general") fail(path, "unsupported symmetry '" + symmetry + "'");
  return h;
}

// COO -> CSR: bucket by row, then sort each row by column (only rows that need it) and sum duplicates.
HostCsr<double> coo_to_csr(const std::string& path, int32_t rows, int32_t cols, const std::vector<int32_t>& I,
                           const std::vector<int32_t>& J, const std::vector<double>& V) {
  const size_t m = I.size();
  std::vector<int64_t> start(size_t(rows) + 1, 0);
  for (int32_t i : I) ++start[size_t(i) + 1];
  std::partial_sum(start.begin(), start.end(), start.begin());

  std::vector<int32_t> col(m);
  std::vector<double> val(m);
  {
    std::vector<int64_t> next(start.begin(), start.end() - 1);
    for (size_t k = 0; k < m; ++k) {
      const int64_t pos = next[size_t(I[k])]++;
      col[size_t(pos)] = J[k];
      val[size_t(pos)] = V[k];
    }
  }

  HostCsr<double> A;
  A.rows = rows;
  A.cols = cols;
  A.row_ptr.assign(size_t(rows) + 1, 0);
  std::vector<size_t> order;
  int64_t w = 0;  // write position; compaction is in place because w <= row start
  for (int32_t r = 0; r < rows; ++r) {
    const int64_t s = start[size_t(r)], e = start[size_t(r) + 1];
    A.row_ptr[size_t(r)] = int32_t(w);
    bool sorted_unique = true;
    for (int64_t k = s + 1; k < e; ++k)
      if (col[size_t(k)] <= col[size_t(k - 1)]) {
        sorted_unique = false;
        break;
      }
    if (sorted_unique) {
      for (int64_t k = s; k < e; ++k, ++w) {
        col[size_t(w)] = col[size_t(k)];
        val[size_t(w)] = val[size_t(k)];
      }
      continue;
    }
    order.resize(size_t(e - s));
    std::iota(order.begin(), order.end(), size_t(s));
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return col[a] < col[b]; });
    std::vector<std::pair<int32_t, double>> merged;
    merged.reserve(order.size());
    for (size_t k : order) {
      if (!merged.empty() && merged.back().first == col[k]) merged.back().second += val[k];
      else merged.emplace_back(col[k], val[k]);
    }
    for (auto& [c, v] : merged) {
      col[size_t(w)] = c;
      val[size_t(w)] = v;
      ++w;
    }
  }
  A.row_ptr[size_t(rows)] = int32_t(w);
  col.resize(size_t(w));
  val.resize(size_t(w));
  A.col_idx = std::move(col);
  A.vals = std::move(val);
  A.nnz = w;
  (void)path;
  return A;
}

HostCsr<double> parse_matrix_market(const std::string& path) {
  const std::string s = read_file(path);
  const size_t nl = s.find('\n');
  const Header h = parse_banner(path, s.substr(0, nl));

  Parser ps{s.data() + (nl == std::string::npos ? s.size() : nl + 1), s.data() + s.size(), path, 2};
  // Skip comment and blank lines before the size line.
  while (ps.p < ps.end) {
    const char* q = ps.p;
    while (q < ps.end && (*q == ' ' || *q == '\t' || *q == '\r')) ++q;
    if (q < ps.end && (*q == '%' || *q == '\n')) {
      while (q < ps.end && *q != '\n') ++q;
      ps.p = q < ps.end ? q + 1 : q;
      ++ps.line;
    } else {
      break;
    }
  }

  const int64_t rows = ps.integer(), cols = ps.integer(), entries = ps.integer();
  constexpr int64_t kMax = std::numeric_limits<int32_t>::max();
  if (rows < 0 || cols < 0 || entries < 0) fail(path, "negative size");
  if (rows >= kMax || cols >= kMax) fail(path, "dimensions exceed 32-bit indices");
  const int64_t expanded = h.symmetry == Header::General ? entries : 2 * entries;
  if (expanded >= kMax) fail(path, "too many entries for 32-bit CSR (" + std::to_string(expanded) + ")");

  std::vector<int32_t> I, J;
  std::vector<double> V;
  I.reserve(size_t(expanded));
  J.reserve(size_t(expanded));
  V.reserve(size_t(expanded));
  for (int64_t k = 0; k < entries; ++k) {
    const int64_t i = ps.integer(), j = ps.integer();
    const double v = h.pattern ? 1.0 : ps.real();
    if (i < 1 || i > rows || j < 1 || j > cols)
      fail(path, "entry (" + std::to_string(i) + ", " + std::to_string(j) + ") out of range at line " +
                     std::to_string(ps.line));
    I.push_back(int32_t(i - 1));
    J.push_back(int32_t(j - 1));
    V.push_back(v);
    if (h.symmetry != Header::General && i != j) {
      I.push_back(int32_t(j - 1));
      J.push_back(int32_t(i - 1));
      V.push_back(h.symmetry == Header::Skew ? -v : v);
    }
  }
  if (h.symmetry != Header::General && rows != cols) fail(path, "symmetric matrix must be square");
  return coo_to_csr(path, int32_t(rows), int32_t(cols), I, J, V);
}

// ---- binary cache ----------------------------------------------------------------------------
constexpr char kMagic[8] = {'R', 'A', 'S', 'P', 'C', 'S', 'R', '1'};

struct BinHeader {
  char magic[8];
  int32_t rows, cols;
  int64_t nnz;
  int64_t src_size;   // size of the .mtx the cache was built from (0 if none)
  int64_t src_mtime;  // its modification time, in filesystem clock ticks
};

void write_binary(const std::string& path, const HostCsr<double>& A, int64_t src_size, int64_t src_mtime) {
  BinHeader hd{};
  std::memcpy(hd.magic, kMagic, sizeof(kMagic));
  hd.rows = A.rows;
  hd.cols = A.cols;
  hd.nnz = A.nnz;
  hd.src_size = src_size;
  hd.src_mtime = src_mtime;

  const std::string tmp = path + ".tmp";
  FILE* f = std::fopen(tmp.c_str(), "wb");
  if (!f) fail(path, "cannot write");
  bool ok = std::fwrite(&hd, sizeof(hd), 1, f) == 1 &&
            std::fwrite(A.row_ptr.data(), sizeof(int32_t), A.row_ptr.size(), f) == A.row_ptr.size() &&
            std::fwrite(A.col_idx.data(), sizeof(int32_t), A.col_idx.size(), f) == A.col_idx.size() &&
            std::fwrite(A.vals.data(), sizeof(double), A.vals.size(), f) == A.vals.size();
  ok = (std::fclose(f) == 0) && ok;
  if (!ok) {
    std::remove(tmp.c_str());
    fail(path, "short write");
  }
  fs::rename(tmp, path);  // atomic: a crashed run never leaves a truncated cache behind
}

HostCsr<double> read_binary(const std::string& path, BinHeader* out_header) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) fail(path, "cannot open");
  BinHeader hd{};
  HostCsr<double> A;
  bool ok = std::fread(&hd, sizeof(hd), 1, f) == 1 && std::memcmp(hd.magic, kMagic, sizeof(kMagic)) == 0 &&
            hd.rows >= 0 && hd.cols >= 0 && hd.nnz >= 0;
  if (ok) {
    A.rows = hd.rows;
    A.cols = hd.cols;
    A.nnz = hd.nnz;
    A.row_ptr.resize(size_t(hd.rows) + 1);
    A.col_idx.resize(size_t(hd.nnz));
    A.vals.resize(size_t(hd.nnz));
    ok = std::fread(A.row_ptr.data(), sizeof(int32_t), A.row_ptr.size(), f) == A.row_ptr.size() &&
         std::fread(A.col_idx.data(), sizeof(int32_t), A.col_idx.size(), f) == A.col_idx.size() &&
         std::fread(A.vals.data(), sizeof(double), A.vals.size(), f) == A.vals.size();
  }
  std::fclose(f);
  if (!ok) fail(path, "corrupt or truncated CSR binary");
  A.validate();
  if (out_header) *out_header = hd;
  return A;
}

int64_t mtime_ticks(const std::string& path) {
  return int64_t(fs::last_write_time(path).time_since_epoch().count());
}

}  // namespace

std::string cache_path_for(const std::string& mtx_path) { return mtx_path + ".csr.bin"; }

void save_csr_binary(const std::string& path, const HostCsr<double>& A) { write_binary(path, A, 0, 0); }

HostCsr<double> load_csr_binary(const std::string& path) { return read_binary(path, nullptr); }

HostCsr<double> read_matrix_market(const std::string& path, CacheMode cache) {
  if (cache == CacheMode::Off) return parse_matrix_market(path);

  const std::string cpath = cache_path_for(path);
  const int64_t size = int64_t(fs::file_size(path));
  const int64_t mtime = mtime_ticks(path);
  if (fs::exists(cpath)) {
    try {
      BinHeader hd{};
      HostCsr<double> A = read_binary(cpath, &hd);
      if (hd.src_size == size && hd.src_mtime == mtime) return A;
    } catch (const std::exception&) {
      // stale or corrupt cache: fall through and rebuild it
    }
  }
  HostCsr<double> A = parse_matrix_market(path);
  try {
    write_binary(cpath, A, size, mtime);
  } catch (const std::exception&) {
    // read-only directory etc.: the cache is an optimisation only
  }
  return A;
}

void write_matrix_market(const std::string& path, const HostCsr<double>& A) {
  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) fail(path, "cannot write");
  std::vector<char> buf(1 << 20);
  std::setvbuf(f, buf.data(), _IOFBF, buf.size());
  std::fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n%d %d %lld\n", A.rows, A.cols, (long long)A.nnz);
  for (int32_t r = 0; r < A.rows; ++r)
    for (int32_t k = A.row_ptr[size_t(r)]; k < A.row_ptr[size_t(r) + 1]; ++k)
      std::fprintf(f, "%d %d %.17g\n", r + 1, A.col_idx[size_t(k)] + 1, A.vals[size_t(k)]);
  if (std::fclose(f) != 0) fail(path, "write error");
}

}  // namespace rasp
