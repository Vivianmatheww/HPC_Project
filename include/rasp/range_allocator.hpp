#pragma once
// First-fit allocator over the offsets [0, capacity) with a coalescing free list. The out-of-core SpGEMM executor
// carves every device buffer out of one cudaMalloc of the fast-memory budget with it, so the budget is enforced
// exactly and peak usage is known. Pure host code (tested without a GPU).
#include <cstddef>
#include <map>
#include <optional>

namespace rasp {

class RangeAllocator {
 public:
  static constexpr size_t kAlign = 256;  // cudaMalloc's alignment; keeps every carved buffer coalescing-friendly

  explicit RangeAllocator(size_t capacity = 0);

  // Offset of a free range of at least `bytes` (rounded up to kAlign; 0 counts as kAlign), or nullopt.
  std::optional<size_t> alloc(size_t bytes);
  // Releases a range returned by alloc(); throws std::invalid_argument for anything else.
  void free(size_t offset);

  size_t capacity() const { return capacity_; }
  size_t in_use() const { return in_use_; }
  size_t peak() const { return peak_; }
  size_t largest_free() const;
  size_t live_blocks() const { return used_.size(); }
  void reset_peak() { peak_ = in_use_; }

 private:
  size_t capacity_ = 0, in_use_ = 0, peak_ = 0;
  std::map<size_t, size_t> free_;  // offset -> size, never adjacent (coalesced)
  std::map<size_t, size_t> used_;  // offset -> size
};

}  // namespace rasp
