#include "satview/gpu/pinned_ring.hpp"

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace satview::gpu {
namespace {

constexpr auto kEventPollInterval = std::chrono::microseconds{250};
constexpr std::size_t kInvalidSlot = static_cast<std::size_t>(-1);

[[noreturn]] void throw_cuda_error(const char* operation,
                                   const cudaError_t error) {
  const char* const name = cudaGetErrorName(error);
  const char* const description = cudaGetErrorString(error);

  std::string message{"PinnedRing: "};
  message += operation;
  message += " failed";
  if (name != nullptr) {
    message += " (";
    message += name;
    message += ')';
  }
  if (description != nullptr) {
    message += ": ";
    message += description;
  }
  throw std::runtime_error{message};
}

PinnedRing::Options validate_options(PinnedRing::Options options) {
  if (options.slot_count == 0) {
    throw std::invalid_argument{"PinnedRing: slot_count must be greater than 0"};
  }
  if (options.bytes_per_slot == 0) {
    throw std::invalid_argument{
        "PinnedRing: bytes_per_slot must be greater than 0"};
  }
  if (options.bytes_per_slot >
      std::numeric_limits<std::size_t>::max() / options.slot_count) {
    throw std::length_error{
        "PinnedRing: slot_count * bytes_per_slot overflows size_t"};
  }
  return options;
}

class IndexQueue final {
 public:
  explicit IndexQueue(const std::size_t capacity) : storage_(capacity) {}

  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

  void push(const std::size_t value) noexcept {
    assert(size_ < storage_.size());
    if (size_ >= storage_.size()) {
      std::terminate();
    }

    const std::size_t tail = (head_ + size_) % storage_.size();
    storage_[tail] = value;
    ++size_;
  }

  [[nodiscard]] std::size_t pop() noexcept {
    assert(size_ != 0);
    if (size_ == 0) {
      std::terminate();
    }

    const std::size_t value = storage_[head_];
    head_ = (head_ + 1) % storage_.size();
    --size_;
    return value;
  }

 private:
  std::vector<std::size_t> storage_;
  std::size_t head_ = 0;
  std::size_t size_ = 0;
};

}  // namespace

struct PinnedRing::Impl final {
  struct Slot {
    std::byte* data = nullptr;
    cudaEvent_t completion = nullptr;
    std::size_t payload_size = 0;
    SlotState state = SlotState::Free;
    bool leased = false;
  };

  explicit Impl(Options requested_options)
      : options(validate_options(requested_options)),
        total_bytes(options.slot_count * options.bytes_per_slot),
        slots(options.slot_count),
        free_slots(options.slot_count),
        ready_slots(options.slot_count) {
    try {
      if (options.page_locked) {
        void* allocation = nullptr;
        const cudaError_t allocation_result = cudaHostAlloc(
            &allocation, total_bytes, options.host_alloc_flags);
        if (allocation_result != cudaSuccess) {
          throw_cuda_error("cudaHostAlloc", allocation_result);
        }
        allocation_ = static_cast<std::byte*>(allocation);
      } else {
        pageable_allocation_ = std::make_unique<std::byte[]>(total_bytes);
        allocation_ = pageable_allocation_.get();
      }

      for (std::size_t index = 0; index < slots.size(); ++index) {
        Slot& slot = slots[index];
        slot.data = allocation_ + (index * options.bytes_per_slot);

        if (options.page_locked) {
          const cudaError_t event_result = cudaEventCreateWithFlags(
              &slot.completion, cudaEventDisableTiming);
          if (event_result != cudaSuccess) {
            throw_cuda_error("cudaEventCreateWithFlags", event_result);
          }
        }
        free_slots.push(index);
      }
    } catch (...) {
      release_cuda_resources(false);
      throw;
    }
  }

  ~Impl() noexcept { release_cuda_resources(true); }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  [[nodiscard]] std::size_t reclaim_completed_locked() {
    std::size_t reclaimed = 0;

    for (std::size_t index = 0; index < slots.size(); ++index) {
      Slot& slot = slots[index];
      if (slot.state != SlotState::InFlight) {
        continue;
      }

      const cudaError_t query_result = cudaEventQuery(slot.completion);
      if (query_result == cudaErrorNotReady) {
        continue;
      }
      if (query_result != cudaSuccess) {
        throw_cuda_error("cudaEventQuery", query_result);
      }

      slot.state = SlotState::Free;
      slot.payload_size = 0;
      free_slots.push(index);
      ++reclaimed;
    }

    if (reclaimed != 0) {
      changed.notify_all();
    }
    return reclaimed;
  }

  [[nodiscard]] SlotLease take_free_locked(
      const std::shared_ptr<Impl>& self) {
    const std::size_t index = free_slots.pop();
    Slot& slot = slots[index];
    assert(slot.state == SlotState::Free && !slot.leased);

    slot.state = SlotState::Filling;
    slot.leased = true;
    slot.payload_size = 0;
    return SlotLease{self, index, LeaseRole::Filling};
  }

  [[nodiscard]] SlotLease take_ready_locked(
      const std::shared_ptr<Impl>& self) {
    const std::size_t index = ready_slots.pop();
    Slot& slot = slots[index];
    assert(slot.state == SlotState::Ready && !slot.leased);

    slot.leased = true;
    return SlotLease{self, index, LeaseRole::Ready};
  }

  void publish_ready(const std::size_t index,
                     const std::size_t bytes_used) {
    std::lock_guard lock{mutex};
    Slot& slot = slots[index];
    if (slot.state != SlotState::Filling || !slot.leased) {
      throw std::logic_error{
          "PinnedRing: publish_ready requires an active Filling lease"};
    }
    if (bytes_used > options.bytes_per_slot) {
      throw std::length_error{
          "PinnedRing: published byte count exceeds slot capacity"};
    }

    slot.payload_size = bytes_used;
    slot.leased = false;
    slot.state = SlotState::Ready;
    ready_slots.push(index);
    changed.notify_all();
  }

  void mark_in_flight(const std::size_t index, cudaStream_t stream) {
    std::lock_guard lock{mutex};
    Slot& slot = slots[index];
    if (slot.state != SlotState::Ready || !slot.leased) {
      throw std::logic_error{
          "PinnedRing: mark_in_flight requires an active Ready lease"};
    }
    if (!options.page_locked) {
      throw std::logic_error{
          "PinnedRing: pageable slots must use mark_consumed"};
    }

    const cudaError_t record_result =
        cudaEventRecord(slot.completion, stream);
    if (record_result != cudaSuccess) {
      // A transfer may already reference this host slot. The exceptional path
      // must finish that stream before the Ready lease can roll back and be
      // queued again; this is never executed in the normal frame path.
      static_cast<void>(cudaStreamSynchronize(stream));

      throw_cuda_error("cudaEventRecord", record_result);
    }

    slot.leased = false;
    slot.state = SlotState::InFlight;
  }

  void mark_consumed(const std::size_t index) {
    std::lock_guard lock{mutex};
    Slot& slot = slots[index];
    if (slot.state != SlotState::Ready || !slot.leased) {
      throw std::logic_error{
          "PinnedRing: mark_consumed requires an active Ready lease"};
    }
    slot.payload_size = 0;
    slot.leased = false;
    slot.state = SlotState::Free;
    free_slots.push(index);
    changed.notify_all();
  }

  void release_lease(const std::size_t index, const LeaseRole role) noexcept {
    std::lock_guard lock{mutex};
    Slot& slot = slots[index];

    if (role == LeaseRole::Filling) {
      assert(slot.state == SlotState::Filling && slot.leased);
      if (slot.state != SlotState::Filling || !slot.leased) {
        std::terminate();
      }
      slot.payload_size = 0;
      slot.leased = false;
      slot.state = SlotState::Free;
      free_slots.push(index);
    } else if (role == LeaseRole::Ready) {
      assert(slot.state == SlotState::Ready && slot.leased);
      if (slot.state != SlotState::Ready || !slot.leased) {
        std::terminate();
      }
      slot.leased = false;
      ready_slots.push(index);
    } else {
      std::terminate();
    }

    changed.notify_all();
  }

  void release_cuda_resources(const bool wait_for_in_flight) noexcept {
    for (Slot& slot : slots) {
      if (slot.completion == nullptr) {
        continue;
      }
      if (wait_for_in_flight && slot.state == SlotState::InFlight) {
        // Destruction may wait for this ring's own outstanding copies, but it
        // never performs a device-wide synchronization.
        static_cast<void>(cudaEventSynchronize(slot.completion));
      }
      static_cast<void>(cudaEventDestroy(slot.completion));
      slot.completion = nullptr;
    }

    if (allocation_ != nullptr && options.page_locked) {
      static_cast<void>(cudaFreeHost(allocation_));
    }
    pageable_allocation_.reset();
    allocation_ = nullptr;
  }

  const Options options;
  const std::size_t total_bytes;
  std::vector<Slot> slots;
  IndexQueue free_slots;
  IndexQueue ready_slots;
  mutable std::mutex mutex;
  std::condition_variable_any changed;

 private:
  std::byte* allocation_ = nullptr;
  std::unique_ptr<std::byte[]> pageable_allocation_;
};

PinnedRing::SlotLease::SlotLease(std::shared_ptr<Impl> impl,
                                const std::size_t index,
                                const LeaseRole role) noexcept
    : impl_(std::move(impl)), index_(index), role_(role) {}

PinnedRing::SlotLease::~SlotLease() noexcept { reset(); }

PinnedRing::SlotLease::SlotLease(SlotLease&& other) noexcept
    : impl_(std::move(other.impl_)),
      index_(std::exchange(other.index_, kInvalidSlot)),
      role_(std::exchange(other.role_, LeaseRole::None)) {}

PinnedRing::SlotLease& PinnedRing::SlotLease::operator=(
    SlotLease&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  reset();
  impl_ = std::move(other.impl_);
  index_ = std::exchange(other.index_, kInvalidSlot);
  role_ = std::exchange(other.role_, LeaseRole::None);
  return *this;
}

PinnedRing::SlotLease::operator bool() const noexcept {
  return impl_ != nullptr;
}

std::span<std::byte> PinnedRing::SlotLease::bytes() noexcept {
  if (impl_ == nullptr) {
    return {};
  }
  Impl::Slot& slot = impl_->slots[index_];
  const std::size_t length = role_ == LeaseRole::Filling
                                 ? impl_->options.bytes_per_slot
                                 : slot.payload_size;
  return {slot.data, length};
}

std::span<const std::byte> PinnedRing::SlotLease::bytes() const noexcept {
  if (impl_ == nullptr) {
    return {};
  }
  const Impl::Slot& slot = impl_->slots[index_];
  const std::size_t length = role_ == LeaseRole::Filling
                                 ? impl_->options.bytes_per_slot
                                 : slot.payload_size;
  return {slot.data, length};
}

std::size_t PinnedRing::SlotLease::capacity() const noexcept {
  return impl_ == nullptr ? 0 : impl_->options.bytes_per_slot;
}

std::size_t PinnedRing::SlotLease::payload_size() const noexcept {
  return impl_ == nullptr ? 0 : impl_->slots[index_].payload_size;
}

std::size_t PinnedRing::SlotLease::slot_index() const noexcept {
  return index_;
}

void PinnedRing::SlotLease::publish_ready(const std::size_t bytes_used) {
  if (impl_ == nullptr || role_ != LeaseRole::Filling) {
    throw std::logic_error{
        "PinnedRing::SlotLease::publish_ready requires a Filling lease"};
  }

  impl_->publish_ready(index_, bytes_used);
  impl_.reset();
  index_ = kInvalidSlot;
  role_ = LeaseRole::None;
}

void PinnedRing::SlotLease::mark_in_flight(cudaStream_t stream) {
  if (impl_ == nullptr || role_ != LeaseRole::Ready) {
    throw std::logic_error{
        "PinnedRing::SlotLease::mark_in_flight requires a Ready lease"};
  }

  impl_->mark_in_flight(index_, stream);
  impl_.reset();
  index_ = kInvalidSlot;
  role_ = LeaseRole::None;
}

void PinnedRing::SlotLease::mark_consumed() {
  if (impl_ == nullptr || role_ != LeaseRole::Ready) {
    throw std::logic_error{
        "PinnedRing::SlotLease::mark_consumed requires a Ready lease"};
  }

  impl_->mark_consumed(index_);
  impl_.reset();
  index_ = kInvalidSlot;
  role_ = LeaseRole::None;
}

void PinnedRing::SlotLease::reset() noexcept {
  if (impl_ == nullptr) {
    return;
  }

  std::shared_ptr<Impl> impl = std::move(impl_);
  const std::size_t index = std::exchange(index_, kInvalidSlot);
  const LeaseRole role = std::exchange(role_, LeaseRole::None);
  impl->release_lease(index, role);
}

PinnedRing::PinnedRing(Options options)
    : impl_(std::make_shared<Impl>(options)) {}

PinnedRing::PinnedRing(const std::size_t slot_count,
                       const std::size_t bytes_per_slot,
                       const unsigned int host_alloc_flags)
    : PinnedRing(Options{slot_count, bytes_per_slot, host_alloc_flags}) {}

PinnedRing::~PinnedRing() noexcept = default;

PinnedRing::PinnedRing(PinnedRing&& other) noexcept = default;

PinnedRing& PinnedRing::operator=(PinnedRing&& other) noexcept = default;

PinnedRing::SlotLease PinnedRing::acquire_for_fill(std::stop_token stop) {
  std::shared_ptr<Impl> impl = impl_;
  if (impl == nullptr) {
    throw std::logic_error{"PinnedRing: use of a moved-from ring"};
  }

  std::unique_lock lock{impl->mutex};
  for (;;) {
    if (stop.stop_requested()) {
      return {};
    }

    static_cast<void>(impl->reclaim_completed_locked());
    if (stop.stop_requested()) {
      return {};
    }
    if (!impl->free_slots.empty()) {
      return impl->take_free_locked(impl);
    }

    const auto deadline =
        std::chrono::steady_clock::now() + kEventPollInterval;
    if (stop.stop_possible()) {
      static_cast<void>(impl->changed.wait_until(
          lock, stop, deadline,
          [&impl] { return !impl->free_slots.empty(); }));
    } else {
      static_cast<void>(impl->changed.wait_until(
          lock, deadline, [&impl] { return !impl->free_slots.empty(); }));
    }
  }
}

PinnedRing::SlotLease PinnedRing::acquire_ready(std::stop_token stop) {
  std::shared_ptr<Impl> impl = impl_;
  if (impl == nullptr) {
    throw std::logic_error{"PinnedRing: use of a moved-from ring"};
  }
  if (stop.stop_requested()) {
    return {};
  }

  std::unique_lock lock{impl->mutex};
  if (stop.stop_possible()) {
    const bool ready = impl->changed.wait(
        lock, stop, [&impl] { return !impl->ready_slots.empty(); });
    if (!ready) {
      return {};
    }
  } else {
    impl->changed.wait(
        lock, [&impl] { return !impl->ready_slots.empty(); });
  }

  if (stop.stop_requested()) {
    return {};
  }
  return impl->take_ready_locked(impl);
}

PinnedRing::SlotLease PinnedRing::try_acquire_for_fill() {
  std::shared_ptr<Impl> impl = impl_;
  if (impl == nullptr) {
    throw std::logic_error{"PinnedRing: use of a moved-from ring"};
  }

  std::lock_guard lock{impl->mutex};
  static_cast<void>(impl->reclaim_completed_locked());
  if (impl->free_slots.empty()) {
    return {};
  }
  return impl->take_free_locked(impl);
}

PinnedRing::SlotLease PinnedRing::try_acquire_ready() {
  std::shared_ptr<Impl> impl = impl_;
  if (impl == nullptr) {
    throw std::logic_error{"PinnedRing: use of a moved-from ring"};
  }

  std::lock_guard lock{impl->mutex};
  if (impl->ready_slots.empty()) {
    return {};
  }
  return impl->take_ready_locked(impl);
}

std::size_t PinnedRing::reclaim_completed() {
  std::shared_ptr<Impl> impl = impl_;
  if (impl == nullptr) {
    throw std::logic_error{"PinnedRing: use of a moved-from ring"};
  }

  std::lock_guard lock{impl->mutex};
  return impl->reclaim_completed_locked();
}

std::size_t PinnedRing::slot_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->options.slot_count;
}

std::size_t PinnedRing::bytes_per_slot() const noexcept {
  return impl_ == nullptr ? 0 : impl_->options.bytes_per_slot;
}

std::size_t PinnedRing::total_pinned_bytes() const noexcept {
  return impl_ == nullptr || !impl_->options.page_locked
      ? 0
      : impl_->total_bytes;
}

PinnedRing::StateCounts PinnedRing::state_counts() const {
  std::shared_ptr<Impl> impl = impl_;
  if (impl == nullptr) {
    throw std::logic_error{"PinnedRing: use of a moved-from ring"};
  }

  StateCounts counts;
  std::lock_guard lock{impl->mutex};
  for (const Impl::Slot& slot : impl->slots) {
    switch (slot.state) {
      case SlotState::Free:
        ++counts.free;
        break;
      case SlotState::Filling:
        ++counts.filling;
        break;
      case SlotState::Ready:
        ++counts.ready;
        break;
      case SlotState::InFlight:
        ++counts.in_flight;
        break;
    }
  }
  return counts;
}

}  // namespace satview::gpu
