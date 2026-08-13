#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stop_token>

namespace satview::cpu {

enum class SlotState : std::uint8_t {
  Free,
  Filling,
  Ready,
  InFlight,
};

// A fixed-size ring of pageable host buffers. All slot and queue storage is
// allocated during construction, so lifecycle transitions and lease rollback
// do not allocate.
class PageableRing final {
 private:
  struct Impl;
  enum class LeaseRole : std::uint8_t { None, Filling, Ready };

 public:
  struct Options {
    std::size_t slot_count;
    std::size_t bytes_per_slot;
  };

  struct StateCounts {
    std::size_t free = 0;
    std::size_t filling = 0;
    std::size_t ready = 0;
    std::size_t in_flight = 0;
  };

  class SlotLease final {
   public:
    SlotLease() noexcept = default;
    ~SlotLease() noexcept;
    SlotLease(SlotLease&& other) noexcept;
    SlotLease& operator=(SlotLease&& other) noexcept;
    SlotLease(const SlotLease&) = delete;
    SlotLease& operator=(const SlotLease&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] std::span<std::byte> bytes() noexcept;
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t payload_size() const noexcept;
    [[nodiscard]] std::size_t slot_index() const noexcept;
    void publish_ready(std::size_t bytes_used);
    void mark_consumed();
    void reset() noexcept;

   private:
    friend class PageableRing;
    SlotLease(
        std::shared_ptr<Impl> impl,
        std::size_t index,
        LeaseRole role) noexcept;

    std::shared_ptr<Impl> impl_;
    std::size_t index_ = static_cast<std::size_t>(-1);
    LeaseRole role_ = LeaseRole::None;
  };

  explicit PageableRing(Options options);
  PageableRing(std::size_t slot_count, std::size_t bytes_per_slot);
  ~PageableRing() noexcept;
  PageableRing(PageableRing&& other) noexcept;
  PageableRing& operator=(PageableRing&& other) noexcept;
  PageableRing(const PageableRing&) = delete;
  PageableRing& operator=(const PageableRing&) = delete;

  [[nodiscard]] SlotLease acquire_for_fill(std::stop_token stop = {});
  [[nodiscard]] SlotLease acquire_ready(std::stop_token stop = {});
  [[nodiscard]] SlotLease try_acquire_for_fill();
  [[nodiscard]] SlotLease try_acquire_ready();
  std::size_t reclaim_completed() noexcept;
  [[nodiscard]] std::size_t slot_count() const noexcept;
  [[nodiscard]] std::size_t bytes_per_slot() const noexcept;
  [[nodiscard]] std::size_t total_pinned_bytes() const noexcept;
  [[nodiscard]] StateCounts state_counts() const;

 private:
  std::shared_ptr<Impl> impl_;
};

}  // namespace satview::cpu
