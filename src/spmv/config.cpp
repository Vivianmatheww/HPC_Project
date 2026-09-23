#include <regex>
#include <stdexcept>

#include "rasp/spmv.hpp"

namespace rasp {

std::string KernelConfig::family() const {
  switch (format) {
    case Format::Cusparse: return "cusparse";
    case Format::CsrScalar: return "csr_scalar";
    case Format::CsrVector: return "csr_vector";
    case Format::Ell: return "ell";
    case Format::Sell: return "sell";
    case Format::ArgCsr: return "argcsr";
  }
  return "?";
}

std::string KernelConfig::label() const {
  switch (format) {
    case Format::Cusparse: return "cusparse_alg" + std::to_string(alg);
    case Format::CsrScalar: return "csr_scalar";
    case Format::CsrVector: return "csr_vector_v" + std::to_string(v);
    case Format::Ell: return "ell";
    case Format::Sell: return "sell_C" + std::to_string(C) + "_s" + (sigma == 0 ? std::string("all") : std::to_string(sigma));
    case Format::ArgCsr: return "argcsr_B" + std::to_string(block) + "_D" + std::to_string(chunk);
  }
  return "?";
}

KernelConfig KernelConfig::parse(const std::string& s) {
  KernelConfig c;
  std::smatch m;
  if (std::regex_match(s, m, std::regex("cusparse_alg([12])"))) {
    c.format = Format::Cusparse;
    c.alg = std::stoi(m[1]);
  } else if (s == "csr_scalar") {
    c.format = Format::CsrScalar;
  } else if (std::regex_match(s, m, std::regex("csr_vector_v(2|4|8|16|32)"))) {
    c.format = Format::CsrVector;
    c.v = std::stoi(m[1]);
  } else if (s == "ell") {
    c.format = Format::Ell;
  } else if (std::regex_match(s, m, std::regex("sell_C(8|32)_s(all|[0-9]+)"))) {
    c.format = Format::Sell;
    c.C = std::stoi(m[1]);
    c.sigma = m[2] == "all" ? 0 : std::stoi(m[2]);
  } else if (std::regex_match(s, m, std::regex("argcsr_B([0-9]+)_D([0-9]+)"))) {
    c.format = Format::ArgCsr;
    c.block = std::stoi(m[1]);
    c.chunk = std::stoi(m[2]);
    if (c.block % 32 != 0 || c.block > 1024 || c.chunk < 1) throw std::invalid_argument("bad argcsr config " + s);
  } else {
    throw std::invalid_argument("unknown kernel config '" + s + "'");
  }
  return c;
}

std::vector<KernelConfig> phase2_configs() {
  std::vector<KernelConfig> out;
  for (const char* s : {"cusparse_alg1", "cusparse_alg2", "csr_scalar"}) out.push_back(KernelConfig::parse(s));
  for (int v : {2, 4, 8, 16, 32}) out.push_back(KernelConfig::parse("csr_vector_v" + std::to_string(v)));
  out.push_back(KernelConfig::parse("ell"));
  for (int C : {8, 32})
    for (const char* sigma : {"1", "256", "all"})
      out.push_back(KernelConfig::parse("sell_C" + std::to_string(C) + "_s" + sigma));
  for (int B : {64, 128, 256})
    for (int D : {1, 2, 4, 8, 16, 32})
      out.push_back(KernelConfig::parse("argcsr_B" + std::to_string(B) + "_D" + std::to_string(D)));
  return out;
}

}  // namespace rasp
