#include "rasp/range_allocator.hpp"

#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace rasp {

RangeAllocator::RangeAllocator(size_t capacity) : capacity_(capacity / kAlign * kAlign) {
  if (capacity_) free_[0] = capacity_;
}

std::optional<size_t> RangeAllocator::alloc(size_t bytes) {
  const size_t need = std::max<size_t>((bytes + kAlign - 1) / kAlign * kAlign, kAlign);
  for (auto it = free_.begin(); it != free_.end(); ++it) {
    if (it->second < need) continue;
    const size_t off = it->first, rest = it->second - need;
    free_.erase(it);
    if (rest) free_[off + need] = rest;
    used_[off] = need;
    in_use_ += need;
    peak_ = std::max(peak_, in_use_);
    return off;
  }
  return std::nullopt;
}

void RangeAllocator::free(size_t offset) {
  auto u = used_.find(offset);
  if (u == used_.end()) throw std::invalid_argument("RangeAllocator::free: unknown offset");
  size_t off = offset, size = u->second;
  used_.erase(u);
  in_use_ -= size;
  auto next = free_.lower_bound(off);
  if (next != free_.end() && off + size == next->first) {  // merge with the following free range
    size += next->second;
    next = free_.erase(next);
  }
  if (next != free_.begin()) {
    auto prev = std::prev(next);
    if (prev->first + prev->second == off) {  // merge with the preceding free range
      prev->second += size;
      return;
    }
  }
  free_[off] = size;
}

size_t RangeAllocator::largest_free() const {
  size_t m = 0;
  for (const auto& [off, size] : free_) m = std::max(m, size);
  return m;
}

}  // namespace rasp
