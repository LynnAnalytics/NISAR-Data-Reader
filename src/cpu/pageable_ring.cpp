#include "satview/cpu/pageable_ring.hpp"

#include <cassert>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace satview::cpu {
namespace {

constexpr std::size_t kInvalidSlot = static_cast<std::size_t>(-1);

PageableRing::Options validate_options(PageableRing::Options options) {
  if (options.slot_count == 0 || options.bytes_per_slot == 0) {
    throw std::invalid_argument(
        "PageableRing dimensions must be greater than zero");
  }
  if (options.bytes_per_slot >
      std::numeric_limits<std::size_t>::max() / options.slot_count) {
    throw std::length_error("PageableRing allocation size overflows size_t");
  }
  return options;
}

}  // namespace

struct PageableRing::Impl final {
  struct Slot {
    std::vector<std::byte> storage;
    std::size_t payload_size = 0;
    SlotState state = SlotState::Free;
    bool leased = false;
  };

  explicit Impl(const Options requested)
      : options(validate_options(requested)), slots(options.slot_count) {
    for (std::size_t index = 0; index < slots.size(); ++index) {
      slots[index].storage.resize(options.bytes_per_slot);
      free_slots.push_back(index);
    }
  }

  [[nodiscard]] SlotLease take_free(const std::shared_ptr<Impl>& self) {
    const std::size_t index = free_slots.front();
    free_slots.pop_front();
    Slot& slot = slots[index];
    assert(slot.state == SlotState::Free && !slot.leased);
    slot.state = SlotState::Filling;
    slot.leased = true;
    slot.payload_size = 0;
    return SlotLease{self, index, LeaseRole::Filling};
  }

  [[nodiscard]] SlotLease take_ready(const std::shared_ptr<Impl>& self) {
    const std::size_t index = ready_slots.front();
    ready_slots.pop_front();
    Slot& slot = slots[index];
    assert(slot.state == SlotState::Ready && !slot.leased);
    slot.leased = true;
    return SlotLease{self, index, LeaseRole::Ready};
  }

  void publish_ready(const std::size_t index, const std::size_t bytes_used) {
    std::lock_guard lock{mutex};
    Slot& slot = slots[index];
    if (slot.state != SlotState::Filling || !slot.leased) {
      throw std::logic_error("PageableRing publish requires a Filling lease");
    }
    if (bytes_used > options.bytes_per_slot) {
      throw std::length_error("PageableRing payload exceeds slot capacity");
    }
    slot.payload_size = bytes_used;
    slot.leased = false;
    slot.state = SlotState::Ready;
    ready_slots.push_back(index);
    changed.notify_all();
  }

  void mark_consumed(const std::size_t index) {
    std::lock_guard lock{mutex};
    Slot& slot = slots[index];
    if (slot.state != SlotState::Ready || !slot.leased) {
      throw std::logic_error("PageableRing consume requires a Ready lease");
    }
    slot.payload_size = 0;
    slot.leased = false;
    slot.state = SlotState::Free;
    free_slots.push_back(index);
    changed.notify_all();
  }

  void release(const std::size_t index, const LeaseRole role) noexcept {
    std::lock_guard lock{mutex};
    Slot& slot = slots[index];
    if (role == LeaseRole::Filling) {
      if (slot.state != SlotState::Filling || !slot.leased) {
        std::terminate();
      }
      slot.payload_size = 0;
      slot.leased = false;
      slot.state = SlotState::Free;
      free_slots.push_back(index);
    } else if (role == LeaseRole::Ready) {
      if (slot.state != SlotState::Ready || !slot.leased) {
        std::terminate();
      }
      slot.leased = false;
      ready_slots.push_back(index);
    } else {
      std::terminate();
    }
    changed.notify_all();
  }

  const Options options;
  std::vector<Slot> slots;
  std::deque<std::size_t> free_slots;
  std::deque<std::size_t> ready_slots;
  mutable std::mutex mutex;
  std::condition_variable_any changed;
};

PageableRing::SlotLease::SlotLease(
    std::shared_ptr<Impl> impl,
    const std::size_t index,
    const LeaseRole role) noexcept
    : impl_(std::move(impl)), index_(index), role_(role) {}

PageableRing::SlotLease::~SlotLease() noexcept { reset(); }

PageableRing::SlotLease::SlotLease(SlotLease&& other) noexcept
    : impl_(std::move(other.impl_)),
      index_(std::exchange(other.index_, kInvalidSlot)),
      role_(std::exchange(other.role_, LeaseRole::None)) {}

PageableRing::SlotLease& PageableRing::SlotLease::operator=(
    SlotLease&& other) noexcept {
  if (this != &other) {
    reset();
    impl_ = std::move(other.impl_);
    index_ = std::exchange(other.index_, kInvalidSlot);
    role_ = std::exchange(other.role_, LeaseRole::None);
  }
  return *this;
}

PageableRing::SlotLease::operator bool() const noexcept {
  return impl_ != nullptr;
}

std::span<std::byte> PageableRing::SlotLease::bytes() noexcept {
  if (impl_ == nullptr) {
    return {};
  }
  auto& slot = impl_->slots[index_];
  const std::size_t size = role_ == LeaseRole::Filling
      ? impl_->options.bytes_per_slot
      : slot.payload_size;
  return {slot.storage.data(), size};
}

std::span<const std::byte> PageableRing::SlotLease::bytes() const noexcept {
  if (impl_ == nullptr) {
    return {};
  }
  const auto& slot = impl_->slots[index_];
  const std::size_t size = role_ == LeaseRole::Filling
      ? impl_->options.bytes_per_slot
      : slot.payload_size;
  return {slot.storage.data(), size};
}

std::size_t PageableRing::SlotLease::capacity() const noexcept {
  return impl_ == nullptr ? 0 : impl_->options.bytes_per_slot;
}

std::size_t PageableRing::SlotLease::payload_size() const noexcept {
  return impl_ == nullptr ? 0 : impl_->slots[index_].payload_size;
}

std::size_t PageableRing::SlotLease::slot_index() const noexcept {
  return index_;
}

void PageableRing::SlotLease::publish_ready(const std::size_t bytes_used) {
  if (impl_ == nullptr || role_ != LeaseRole::Filling) {
    throw std::logic_error("PageableRing publish requires a Filling lease");
  }
  impl_->publish_ready(index_, bytes_used);
  impl_.reset();
  index_ = kInvalidSlot;
  role_ = LeaseRole::None;
}

void PageableRing::SlotLease::mark_consumed() {
  if (impl_ == nullptr || role_ != LeaseRole::Ready) {
    throw std::logic_error("PageableRing consume requires a Ready lease");
  }
  impl_->mark_consumed(index_);
  impl_.reset();
  index_ = kInvalidSlot;
  role_ = LeaseRole::None;
}

void PageableRing::SlotLease::reset() noexcept {
  if (impl_ == nullptr) {
    return;
  }
  auto impl = std::move(impl_);
  const auto index = std::exchange(index_, kInvalidSlot);
  const auto role = std::exchange(role_, LeaseRole::None);
  impl->release(index, role);
}

PageableRing::PageableRing(Options options)
    : impl_(std::make_shared<Impl>(options)) {}

PageableRing::PageableRing(
    const std::size_t slot_count, const std::size_t bytes_per_slot)
    : PageableRing(Options{slot_count, bytes_per_slot}) {}

PageableRing::~PageableRing() noexcept = default;
PageableRing::PageableRing(PageableRing&& other) noexcept = default;
PageableRing& PageableRing::operator=(PageableRing&& other) noexcept = default;

PageableRing::SlotLease PageableRing::acquire_for_fill(std::stop_token stop) {
  auto impl = impl_;
  if (impl == nullptr) {
    throw std::logic_error("PageableRing use after move");
  }
  std::unique_lock lock{impl->mutex};
  const bool ready = impl->changed.wait(
      lock, stop, [&impl] { return !impl->free_slots.empty(); });
  return ready && !stop.stop_requested() ? impl->take_free(impl) : SlotLease{};
}

PageableRing::SlotLease PageableRing::acquire_ready(std::stop_token stop) {
  auto impl = impl_;
  if (impl == nullptr) {
    throw std::logic_error("PageableRing use after move");
  }
  std::unique_lock lock{impl->mutex};
  const bool ready = impl->changed.wait(
      lock, stop, [&impl] { return !impl->ready_slots.empty(); });
  return ready && !stop.stop_requested() ? impl->take_ready(impl) : SlotLease{};
}

PageableRing::SlotLease PageableRing::try_acquire_for_fill() {
  auto impl = impl_;
  if (impl == nullptr) {
    throw std::logic_error("PageableRing use after move");
  }
  std::lock_guard lock{impl->mutex};
  return impl->free_slots.empty() ? SlotLease{} : impl->take_free(impl);
}

PageableRing::SlotLease PageableRing::try_acquire_ready() {
  auto impl = impl_;
  if (impl == nullptr) {
    throw std::logic_error("PageableRing use after move");
  }
  std::lock_guard lock{impl->mutex};
  return impl->ready_slots.empty() ? SlotLease{} : impl->take_ready(impl);
}

std::size_t PageableRing::reclaim_completed() noexcept { return 0; }

std::size_t PageableRing::slot_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->options.slot_count;
}

std::size_t PageableRing::bytes_per_slot() const noexcept {
  return impl_ == nullptr ? 0 : impl_->options.bytes_per_slot;
}

std::size_t PageableRing::total_pinned_bytes() const noexcept { return 0; }

PageableRing::StateCounts PageableRing::state_counts() const {
  auto impl = impl_;
  if (impl == nullptr) {
    throw std::logic_error("PageableRing use after move");
  }
  StateCounts counts;
  std::lock_guard lock{impl->mutex};
  for (const auto& slot : impl->slots) {
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

}  // namespace satview::cpu
