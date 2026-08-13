#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stop_token>

namespace satview::gpu {

enum class SlotState : std::uint8_t {
  Free,
  Filling,
  Ready,
  InFlight,
};

// A fixed-size ring of reusable, page-locked host buffers.
//
// The intended single-reader/single-consumer flow is:
//
//   auto filling = ring.acquire_for_fill(stop);
//   hdf5_read(filling.bytes().data(), filling.capacity());
//   filling.publish_ready(bytes_read);
//
//   auto ready = ring.acquire_ready(stop);
//   cudaMemcpyAsync(device, ready.bytes().data(), ready.bytes().size(),
//                   cudaMemcpyHostToDevice, stream);
//   ready.mark_in_flight(stream);
//
// mark_in_flight records a disabled-timing CUDA event after the copy. Calling
// reclaim_completed (or acquiring another fill slot) queries those events and
// returns completed slots to the free queue without synchronizing the device.
class PinnedRing final {
 private:
  struct Impl;

  enum class LeaseRole : std::uint8_t {
    None,
    Filling,
    Ready,
  };

 public:
  struct Options {
    std::size_t slot_count;
    std::size_t bytes_per_slot;
    unsigned int host_alloc_flags = cudaHostAllocDefault;
    // CPU and experimental backends do not require CUDA page locking or
    // completion events. Their consumers release a Ready slot synchronously
    // with mark_consumed().
    bool page_locked = true;
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

    // For a Filling lease this spans the entire writable slot. For a Ready
    // lease it spans only the byte count supplied to publish_ready.
    [[nodiscard]] std::span<std::byte> bytes() noexcept;
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;

    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t payload_size() const noexcept;
    [[nodiscard]] std::size_t slot_index() const noexcept;

    // Transition Filling -> Ready and make the slot available to the consumer.
    // On error the lease remains valid so the caller may retry or let it roll
    // back to Free.
    void publish_ready(std::size_t bytes_used);

    // Record an event on stream after the caller has enqueued its async H2D
    // copy, then transition Ready -> InFlight. On CUDA error the lease remains
    // valid and is returned to Ready when destroyed.
    void mark_in_flight(cudaStream_t stream);

    // Transition Ready -> Free after a synchronous consumer has finished
    // reading the slot. This is the non-CUDA counterpart to mark_in_flight.
    void mark_consumed();

    // Release without publishing/submitting. A Filling lease returns to Free;
    // a Ready lease returns to the ready queue.
    void reset() noexcept;

   private:
    friend class PinnedRing;

    SlotLease(std::shared_ptr<Impl> impl, std::size_t index,
              LeaseRole role) noexcept;

    std::shared_ptr<Impl> impl_;
    std::size_t index_ = static_cast<std::size_t>(-1);
    LeaseRole role_ = LeaseRole::None;
  };

  explicit PinnedRing(Options options);
  PinnedRing(std::size_t slot_count, std::size_t bytes_per_slot,
             unsigned int host_alloc_flags = cudaHostAllocDefault);
  ~PinnedRing() noexcept;

  PinnedRing(PinnedRing&& other) noexcept;
  PinnedRing& operator=(PinnedRing&& other) noexcept;

  PinnedRing(const PinnedRing&) = delete;
  PinnedRing& operator=(const PinnedRing&) = delete;

  // Blocking acquisitions return an empty lease when stop is requested.
  // acquire_for_fill periodically queries in-flight events while it waits.
  [[nodiscard]] SlotLease acquire_for_fill(std::stop_token stop = {});
  [[nodiscard]] SlotLease acquire_ready(std::stop_token stop = {});

  // Immediate acquisitions return an empty lease when no slot is available.
  // try_acquire_for_fill also performs one nonblocking event-reclaim pass.
  [[nodiscard]] SlotLease try_acquire_for_fill();
  [[nodiscard]] SlotLease try_acquire_ready();

  // Query every in-flight event once. Returns the number of slots transitioned
  // InFlight -> Free. CUDA failures are reported as std::runtime_error.
  std::size_t reclaim_completed();

  [[nodiscard]] std::size_t slot_count() const noexcept;
  [[nodiscard]] std::size_t bytes_per_slot() const noexcept;
  [[nodiscard]] std::size_t total_pinned_bytes() const noexcept;
  [[nodiscard]] StateCounts state_counts() const;

 private:
  std::shared_ptr<Impl> impl_;
};

}  // namespace satview::gpu
