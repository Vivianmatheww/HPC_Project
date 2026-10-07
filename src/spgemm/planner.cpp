// SpGEMM planners (host-only): the device-memory footprint and time model of every execution mode, paper 1's
// Alg. 4 (baseline) and the RASP planner that searches partitions and pipeline depths by simulation.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

#include "rasp/spgemm_plan.hpp"

namespace rasp {

const char* to_string(SpgemmMode m) {
  switch (m) {
    case SpgemmMode::InCore: return "incore";
    case SpgemmMode::ZeroCopy: return "zerocopy";
    case SpgemmMode::ChunkB: return "chunk_b";
    case SpgemmMode::ChunkAC: return "chunk_ac";
  }
  return "?";
}

std::vector<int32_t> split_rows(int32_t begin, int32_t end, const std::function<double(int32_t, int32_t)>& size,
                                double cap) {
  std::vector<int32_t> bounds{begin};
  for (int32_t r0 = begin; r0 < end;) {
    if (size(r0, r0 + 1) > cap) {
      char msg[160];
      std::snprintf(msg, sizeof(msg), "row %d alone needs %.3g MB, more than the %.3g MB available", r0,
                    size(r0, r0 + 1) / 1e6, cap / 1e6);
      throw BudgetTooSmall(msg);
    }
    int32_t lo = r0 + 1, hi = end;  // largest r1 in [lo, hi] with size(r0, r1) <= cap; lo is feasible
    while (lo < hi) {
      const int32_t mid = lo + (hi - lo + 1) / 2;
      if (size(r0, mid) <= cap) lo = mid;
      else hi = mid - 1;
    }
    bounds.push_back(lo);
    r0 = lo;
  }
  return bounds;
}

double simulate(const std::vector<StageCost>& st, int depth) {
  const int K = int(st.size());
  if (K == 0) return 0;
  if (depth <= 1) {
    double t = 0;
    for (const auto& s : st) t += s.in_ms + s.comp_ms + s.out_ms;
    return t;
  }
  const int L = depth - 1;
  std::vector<double> h2d_end(size_t(K), 0), comp_end(size_t(K), 0);
  double copy_free = 0, comp_free = 0;
  auto upload = [&](int j, double issued) {
    copy_free = std::max(copy_free, issued) + st[size_t(j)].in_ms;
    h2d_end[size_t(j)] = copy_free;
  };
  for (int j = 0; j < std::min(L, K); ++j) upload(j, 0);
  for (int k = 0; k < K; ++k) {
    const double host = k == 0 ? 0 : comp_end[size_t(k) - 1];  // the host returns from compute k-1's size queries
    if (k + L < K) upload(k + L, host);
    const double start = std::max({host, h2d_end[size_t(k)], comp_free});
    comp_free = comp_end[size_t(k)] = start + st[size_t(k)].comp_ms;
    if (st[size_t(k)].out_ms > 0) copy_free = std::max(copy_free, comp_end[size_t(k)]) + st[size_t(k)].out_ms;
  }
  return std::max(copy_free, comp_free);
}

bool needs_replan(const CostParams& planned, const CostParams& measured, double threshold) {
  auto off = [&](double a, double b) { return b > 0 && std::abs(a / b - 1.0) > threshold; };
  return off(measured.flops_per_ms, planned.flops_per_ms) || off(measured.gamma, planned.gamma) ||
         off(measured.ws_bytes_per_flop, planned.ws_bytes_per_flop);
}

namespace {

constexpr double kFactors[] = {1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
constexpr size_t kMaxParts = 4096;

double ms_for(double bytes, double gbs) { return bytes / (gbs * 1e6); }

// The footprint and time model shared by both planners and predict().
struct Model {
  const ProblemSizes& s;
  const CostParams& p;

  int32_t a_rows() const { return s.a_rows; }
  int32_t b_rows() const { return s.a_cols; }
  double ws(double flops) const { return p.ws_base_bytes + p.ws_bytes_per_flop * flops; }
  double A(int32_t r0, int32_t r1) const { return s.a_bytes(r0, r1); }
  double C(int32_t r0, int32_t r1) const { return s.c_bytes(r0, r1, p.gamma); }
  double B(int32_t k0, int32_t k1) const { return s.b_bytes(k0, k1); }
  double flops(int32_t r0, int32_t r1) const { return double(s.flops(r0, r1)); }
  double comp(double fl) const { return fl / p.flops_per_ms + p.tile_overhead_ms; }
  double add(int32_t r0, int32_t r1) const { return ms_for(3 * C(r0, r1), p.device_gbs); }  // C_prev + P -> C_new

  double b_frac(int32_t k0, int32_t k1) const {
    const double total = double(s.b_nnz(0, b_rows()));
    return total > 0 ? double(s.b_nnz(k0, k1)) / total : 0;
  }
  double max_b_frac(const std::vector<int32_t>& bb) const {
    double m = 0;
    for (size_t q = 0; q + 1 < bb.size(); ++q) m = std::max(m, b_frac(bb[q], bb[q + 1]));
    return m;
  }
  double max_b_bytes(const std::vector<int32_t>& bb) const {
    double m = 0;
    for (size_t q = 0; q + 1 < bb.size(); ++q) m = std::max(m, B(bb[q], bb[q + 1]));
    return m;
  }
  static bool multi(const std::vector<int32_t>& bb) { return bb.size() > 2; }

  double incore_bytes() const {
    const int32_t n = a_rows();
    return A(0, n) + B(0, b_rows()) + C(0, n) + ws(flops(0, n));
  }

  // Device bytes of one streamed or resident A/C block, by mode (see the executor for what is live at once):
  //   ChunkB 1-D   d A blocks + d C blocks (prefetched uploads, the current C, the C still downloading) + workspace
  //   ChunkB multi d x (A + partial C in) + C_new pending download x d + P + workspace
  //   ChunkAC      A (+ the next A prefetched) + C_acc, P, C_new (+ the previous C downloading) + workspace
  //   ZeroCopy     C's row pointers + workspace (A and C stay in host memory)
  double part_bytes(SpgemmMode mode, int d, bool multi_b, double bfrac, int32_t r0, int32_t r1) const {
    const double a = A(r0, r1), c = C(r0, r1), w = ws(flops(r0, r1) * bfrac);
    const bool overlap = d > 1;
    switch (mode) {
      case SpgemmMode::ChunkB: return multi_b ? d * (a + 2 * c) + c + w : d * (a + c) + w;
      case SpgemmMode::ChunkAC: return (overlap ? 2 : 1) * a + (multi_b ? (overlap ? 4 : 3) : (overlap ? 2 : 1)) * c + w;
      case SpgemmMode::ZeroCopy: return double(r1 - r0 + 1) * 4 + w;
      case SpgemmMode::InCore: return a + c + w;
    }
    return 0;
  }
  // Bytes of the other operand that sit beside those blocks.
  double resident_bytes(SpgemmMode mode, int d, const std::vector<int32_t>& bb) const {
    const double ext = multi(bb) ? s.ext_bytes() : 0;
    switch (mode) {
      case SpgemmMode::ChunkB: return max_b_bytes(bb) + ext;
      case SpgemmMode::ChunkAC: return d * (max_b_bytes(bb) + ext);
      case SpgemmMode::ZeroCopy: return B(0, b_rows());
      case SpgemmMode::InCore: return B(0, b_rows());
    }
    return 0;
  }
  std::function<double(int32_t, int32_t)> part_fn(SpgemmMode mode, int d, const std::vector<int32_t>& bb) const {
    const bool mb = multi(bb);
    const double bf = mode == SpgemmMode::ChunkB || mode == SpgemmMode::ChunkAC ? max_b_frac(bb) : 1.0;
    return [this, mode, d, mb, bf](int32_t r0, int32_t r1) { return part_bytes(mode, d, mb, bf, r0, r1); };
  }
};

// Splits the first block so that it holds ~1/8 of its multiply-adds: a cheap calibration chunk that is also the
// pipeline fill (nothing overlaps the first upload anyway).
void add_probe(const ProblemSizes& s, SpgemmPlan& plan) {
  if (plan.ac_bounds.size() < 2) return;
  const int32_t b0 = plan.ac_bounds[0], b1 = plan.ac_bounds[1];
  if (b1 - b0 < 2) return;
  const int64_t target = s.flops_ptr[size_t(b0)] + s.flops(b0, b1) / 8;
  int32_t r = int32_t(std::lower_bound(s.flops_ptr.begin() + b0 + 1, s.flops_ptr.begin() + b1, target) -
                      s.flops_ptr.begin());
  r = std::clamp(r, b0 + 1, b1 - 1);
  plan.ac_bounds.insert(plan.ac_bounds.begin() + 1, r);
  plan.probe = true;
}

}  // namespace

void predict(const ProblemSizes& s, SpgemmPlan& plan) {
  const Model m{s, plan.params};
  const auto& ab = plan.ac_bounds;
  const auto& bb = plan.b_bounds;
  const int P = plan.parts_ac(), Q = plan.parts_b();
  const double Ball = m.B(0, s.a_cols);
  plan.predicted_ms = plan.copy_bytes = plan.peak_bytes = 0;
  if (P == 0) return;
  const int32_t a0 = ab.front(), a1 = ab.back();
  auto h2d = [&](double b) { return ms_for(b, plan.params.h2d_gbs); };
  auto d2h = [&](double b) { return ms_for(b, plan.params.d2h_gbs); };

  std::vector<StageCost> st;
  switch (plan.mode) {
    case SpgemmMode::InCore: {
      st.push_back({h2d(m.A(a0, a1) + Ball), m.comp(m.flops(a0, a1)), d2h(m.C(a0, a1))});
      plan.predicted_ms = simulate(st, 1);
      plan.copy_bytes = m.A(a0, a1) + Ball + m.C(a0, a1);
      plan.peak_bytes = m.A(a0, a1) + Ball + m.C(a0, a1) + m.ws(m.flops(a0, a1));
      return;
    }
    case SpgemmMode::ZeroCopy: {
      for (int i = 0; i < P; ++i) {
        const int32_t r0 = ab[size_t(i)], r1 = ab[size_t(i) + 1];
        const double pcie = h2d(plan.params.zc_passes_a * m.A(r0, r1) + plan.params.zc_passes_c * m.C(r0, r1));
        st.push_back({i == 0 ? h2d(Ball) : 0, std::max(m.comp(m.flops(r0, r1)), pcie + plan.params.tile_overhead_ms), 0});
        plan.peak_bytes = std::max(plan.peak_bytes, Ball + m.part_bytes(plan.mode, 1, false, 1, r0, r1));
      }
      plan.predicted_ms = simulate(st, 1);
      plan.copy_bytes = Ball;
      return;
    }
    case SpgemmMode::ChunkB: {
      const bool mb = Model::multi(bb);
      const double res = m.resident_bytes(plan.mode, plan.depth, bb);
      for (int q = 0; q < Q; ++q) {
        const int32_t k0 = bb[size_t(q)], k1 = bb[size_t(q) + 1];
        const double bf = m.b_frac(k0, k1);
        st.clear();
        for (int i = 0; i < P; ++i) {
          const int32_t r0 = ab[size_t(i)], r1 = ab[size_t(i) + 1];
          const double in = m.A(r0, r1) + (q > 0 ? m.C(r0, r1) : 0) + (i == 0 ? m.B(k0, k1) : 0);
          const double comp = m.comp(m.flops(r0, r1) * bf) + (q > 0 ? m.add(r0, r1) : 0);
          st.push_back({h2d(in), comp, d2h(m.C(r0, r1))});
          plan.copy_bytes += m.A(r0, r1) + (q > 0 ? m.C(r0, r1) : 0) + m.C(r0, r1);
          plan.peak_bytes = std::max(plan.peak_bytes, res + m.part_bytes(plan.mode, plan.depth, mb, bf, r0, r1));
        }
        plan.copy_bytes += m.B(k0, k1);
        plan.predicted_ms += simulate(st, plan.depth);  // the executor drains between B blocks
      }
      return;
    }
    case SpgemmMode::ChunkAC: {
      const bool mb = Model::multi(bb);
      const double res = m.resident_bytes(plan.mode, plan.depth, bb);
      for (int i = 0; i < P; ++i) {
        const int32_t r0 = ab[size_t(i)], r1 = ab[size_t(i) + 1];
        for (int q = 0; q < Q; ++q) {
          const int32_t k0 = bb[size_t(q)], k1 = bb[size_t(q) + 1];
          const double bf = m.b_frac(k0, k1);
          const double in = m.B(k0, k1) + (q == 0 ? m.A(r0, r1) : 0);
          const double comp = m.comp(m.flops(r0, r1) * bf) + (q > 0 ? m.add(r0, r1) : 0);
          st.push_back({h2d(in), comp, q == Q - 1 ? d2h(m.C(r0, r1)) : 0});
          plan.copy_bytes += in + (q == Q - 1 ? m.C(r0, r1) : 0);
          plan.peak_bytes = std::max(plan.peak_bytes, res + m.part_bytes(plan.mode, plan.depth, mb, bf, r0, r1));
        }
      }
      plan.predicted_ms = simulate(st, plan.depth);
      return;
    }
  }
}

std::string SpgemmPlan::to_string() const {
  char buf[256];
  std::snprintf(buf, sizeof(buf), "%s P_AC=%d P_B=%d depth=%d%s | predicted %.2f ms, copies %.3g GB, peak %.0f MiB | ",
                rasp::to_string(mode), parts_ac(), parts_b(), depth, probe ? " +probe" : "", predicted_ms,
                copy_bytes / 1e9, peak_bytes / double(1 << 20));
  return buf + reason;
}

SpgemmPlan plan_alg4(const ProblemSizes& s, double budget, const CostParams& p, int depth) {
  const Model m{s, p};
  const int32_t n = s.a_rows, nb = s.a_cols;
  const int d = std::max(depth, 1);
  SpgemmPlan plan;
  plan.params = p;
  plan.depth = d;
  if (m.incore_bytes() <= budget) {
    plan.mode = SpgemmMode::InCore;
    plan.depth = 1;
    plan.ac_bounds = {0, n};
    plan.b_bounds = {0, nb};
    plan.reason = "alg4: everything fits the budget";
    predict(s, plan);
    return plan;
  }
  const double Abytes = m.A(0, n), Bbytes = m.B(0, nb), Cbytes = m.C(0, n);
  auto bfn = [&](bool ext) {
    return [&, ext](int32_t k0, int32_t k1) { return m.B(k0, k1) + (ext ? s.ext_bytes() : 0); };
  };

  // chunk2: B blocks resident in 75%, A/C streamed (with the workspace and partial C) in 25%.
  SpgemmPlan c2 = plan;
  double cost2 = std::numeric_limits<double>::infinity();
  try {
    c2.mode = SpgemmMode::ChunkB;
    c2.b_bounds = split_rows(0, nb, bfn(false), 0.75 * budget);
    if (Model::multi(c2.b_bounds)) c2.b_bounds = split_rows(0, nb, bfn(true), 0.75 * budget);
    c2.ac_bounds = split_rows(0, n, m.part_fn(SpgemmMode::ChunkB, d, c2.b_bounds), 0.25 * budget);
    const int PB = c2.parts_b();
    cost2 = Bbytes + Abytes * PB + Cbytes * (PB - 1);
  } catch (const BudgetTooSmall&) {
  }

  // chunk1: A/C blocks resident in 75% (with partial C), B streamed (with the workspace) in 25%.
  SpgemmPlan c1 = plan;
  double cost1 = std::numeric_limits<double>::infinity();
  try {
    c1.mode = SpgemmMode::ChunkAC;
    auto max_flops = [&](const std::vector<int32_t>& ab) {
      double f = 0;
      for (size_t i = 0; i + 1 < ab.size(); ++i) f = std::max(f, m.flops(ab[i], ab[i + 1]));
      return f;
    };
    auto resident = [&](bool multi_b) {  // the A/C block without the workspace, which goes to the streamed share
      return [&, multi_b](int32_t r0, int32_t r1) {
        return m.part_bytes(SpgemmMode::ChunkAC, d, multi_b, 0, r0, r1) - m.ws(0);
      };
    };
    // One B block if d copies of B and the workspace fit the streamed share; otherwise partial sums on the A/C side.
    c1.ac_bounds = split_rows(0, n, resident(false), 0.75 * budget);
    if (d * Bbytes + m.ws(max_flops(c1.ac_bounds)) <= 0.25 * budget) {
      c1.b_bounds = {0, nb};
    } else {
      c1.ac_bounds = split_rows(0, n, resident(true), 0.75 * budget);
      const double fmax = max_flops(c1.ac_bounds);
      c1.b_bounds = split_rows(
          0, nb,
          [&](int32_t k0, int32_t k1) { return d * (m.B(k0, k1) + s.ext_bytes()) + m.ws(fmax * m.b_frac(k0, k1)); },
          0.25 * budget);
    }
    cost1 = Abytes + Cbytes + Bbytes * c1.parts_ac();
  } catch (const BudgetTooSmall&) {
  }

  if (!std::isfinite(cost1) && !std::isfinite(cost2))
    throw BudgetTooSmall("alg4: neither chunk1 nor chunk2 fits a budget of " + std::to_string(budget / 1e6) + " MB");
  plan = cost2 <= cost1 ? c2 : c1;
  char why[160];
  std::snprintf(why, sizeof(why), "alg4: copy volume chunk1 %.3g GB, chunk2 %.3g GB", cost1 / 1e9, cost2 / 1e9);
  plan.reason = why;
  predict(s, plan);
  return plan;
}

namespace {

// The best partition of A/C rows [r_begin, a_rows) for a fixed mode, depth and B partition: block caps from the
// free memory divided by every factor in kFactors; each candidate is simulated.
bool best_rows(const ProblemSizes& s, double budget, SpgemmPlan base, int32_t r_begin, bool probe, SpgemmPlan& best) {
  const Model m{s, base.params};
  const double free = budget - m.resident_bytes(base.mode, base.depth, base.b_bounds);
  if (free <= 0) return false;
  const auto part = m.part_fn(base.mode, base.depth, base.b_bounds);
  bool found = false;
  int worse = 0;  // predicted time is roughly convex in the block count: stop after two steps uphill
  for (double f : kFactors) {
    try {
      base.ac_bounds = split_rows(r_begin, s.a_rows, part, free / f);
    } catch (const BudgetTooSmall&) {
      break;  // smaller blocks cannot fit either
    }
    if (found && size_t(base.parts_ac()) > kMaxParts) break;  // the minimum count is always tried
    SpgemmPlan cand = base;
    if (probe) add_probe(s, cand);
    predict(s, cand);
    if (!found || cand.predicted_ms < best.predicted_ms) {
      best = cand;
      found = true;
      worse = 0;
    } else if (++worse == 2) {
      break;
    }
    if (base.parts_ac() >= s.a_rows - r_begin) break;  // one row per block already
  }
  return found;
}

}  // namespace

SpgemmPlan plan_rasp(const ProblemSizes& s, double budget, const CostParams& p, const RaspPlanOptions& o) {
  const Model m{s, p};
  const int32_t n = s.a_rows, nb = s.a_cols;
  auto allowed = [&](SpgemmMode md) { return !o.has_forced_mode || o.forced_mode == md; };
  const std::vector<int> depths = o.depth > 0 ? std::vector<int>{o.depth} : std::vector<int>{2, 3};

  SpgemmPlan best;
  bool found = false;
  auto consider = [&](const SpgemmPlan& c) {
    if (!found || c.predicted_ms < best.predicted_ms) {
      best = c;
      found = true;
    }
  };
  SpgemmPlan base;
  base.params = p;

  if (allowed(SpgemmMode::InCore) && m.incore_bytes() <= budget) {
    SpgemmPlan c = base;
    c.mode = SpgemmMode::InCore;
    c.ac_bounds = {0, n};
    c.b_bounds = {0, nb};
    predict(s, c);
    consider(c);
  }
  for (int d : depths) {
    // ChunkB 1-D: B resident whole, A/C streamed through all the memory B leaves free.
    if (allowed(SpgemmMode::ChunkB)) {
      SpgemmPlan c = base, out;
      c.mode = SpgemmMode::ChunkB;
      c.depth = d;
      c.b_bounds = {0, nb};
      if (best_rows(s, budget, c, 0, o.probe, out)) consider(out);
      // ChunkB with several B blocks (B larger than the budget allows).
      for (double share : {0.75, 0.5, 0.25, 0.125}) {
        try {
          c.b_bounds = split_rows(0, nb, [&](int32_t k0, int32_t k1) { return m.B(k0, k1) + s.ext_bytes(); },
                                  share * budget);
        } catch (const BudgetTooSmall&) {
          break;
        }
        if (!Model::multi(c.b_bounds)) continue;
        if (best_rows(s, budget, c, 0, false, out)) consider(out);
      }
    }
    // ChunkAC: A/C resident, B blocks streamed (d of them in flight).
    if (allowed(SpgemmMode::ChunkAC)) {
      SpgemmPlan c = base, out;
      c.mode = SpgemmMode::ChunkAC;
      c.depth = d;
      for (double share : {0.5, 0.25, 0.125, 0.0625, 0.03125}) {
        try {
          c.b_bounds = split_rows(0, nb, [&](int32_t k0, int32_t k1) { return m.B(k0, k1); }, share * budget / d);
          if (Model::multi(c.b_bounds))
            c.b_bounds = split_rows(0, nb, [&](int32_t k0, int32_t k1) { return m.B(k0, k1) + s.ext_bytes(); },
                                    share * budget / d);
        } catch (const BudgetTooSmall&) {
          break;
        }
        if (best_rows(s, budget, c, 0, false, out)) consider(out);
      }
    }
  }
  // Zero-copy only when the model is clearly ahead: its constants are the least certain.
  if (allowed(SpgemmMode::ZeroCopy) && (o.allow_zero_copy || o.has_forced_mode)) {
    SpgemmPlan c = base, out;
    c.mode = SpgemmMode::ZeroCopy;
    c.b_bounds = {0, nb};
    if (best_rows(s, budget, c, 0, false, out) && (o.has_forced_mode || !found || out.predicted_ms * 1.25 < best.predicted_ms))
      consider(out);
  }
  if (!found)
    throw BudgetTooSmall("rasp: no SpGEMM plan fits a budget of " + std::to_string(budget / 1e6) + " MB");
  char why[96];
  std::snprintf(why, sizeof(why), "rasp: fastest simulated candidate (%.3g GFLOP/s predicted)",
                best.predicted_ms > 0 ? 2.0 * double(s.flops(0, n)) / (best.predicted_ms * 1e6) : 0.0);
  best.reason = why;
  return best;
}

std::vector<int32_t> replan_rows(const ProblemSizes& s, double budget, const SpgemmPlan& plan, const CostParams& p,
                                 int32_t r_begin) {
  SpgemmPlan base = plan;
  base.params = p;
  base.probe = false;
  SpgemmPlan out;
  if (r_begin >= s.a_rows || !best_rows(s, budget, base, r_begin, false, out)) {
    // Nothing fits under the new estimates: keep the old boundaries from r_begin on (the executor splits on overflow).
    std::vector<int32_t> tail{r_begin};
    for (int32_t b : plan.ac_bounds)
      if (b > r_begin) tail.push_back(b);
    if (tail.back() != s.a_rows) tail.push_back(s.a_rows);
    return tail;
  }
  return out.ac_bounds;
}

}  // namespace rasp
