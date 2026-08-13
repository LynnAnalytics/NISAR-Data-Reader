#include "satview/cpu/pageable_ring.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <stop_token>
#include <string_view>
#include <thread>

namespace {

struct TestContext {
  int failures = 0;

  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      ++failures;
      std::cerr << "PageableRing test failed: " << message << '\n';
    }
  }
};

template <typename Exception, typename Function>
void expect_throws(
    TestContext& test,
    Function&& function,
    const std::string_view message) {
  bool matched = false;
  try {
    function();
  } catch (const Exception&) {
    matched = true;
  } catch (...) {
  }
  test.expect(matched, message);
}

void expect_counts(
    TestContext& test,
    const satview::cpu::PageableRing& ring,
    const std::size_t free,
    const std::size_t filling,
    const std::size_t ready,
    const std::string_view message) {
  const auto counts = ring.state_counts();
  test.expect(
      counts.free == free && counts.filling == filling &&
          counts.ready == ready && counts.in_flight == 0,
      message);
}

void test_validation_and_lifecycle(TestContext& test) {
  using satview::cpu::PageableRing;

  expect_throws<std::invalid_argument>(
      test,
      [] { static_cast<void>(PageableRing{0, 16}); },
      "zero slot count is rejected");
  expect_throws<std::invalid_argument>(
      test,
      [] { static_cast<void>(PageableRing{1, 0}); },
      "zero slot size is rejected");
  expect_throws<std::length_error>(
      test,
      [] {
        static_cast<void>(PageableRing{
            2, std::numeric_limits<std::size_t>::max()});
      },
      "total allocation overflow is rejected");

  PageableRing ring{2, 16};
  test.expect(
      ring.slot_count() == 2 && ring.bytes_per_slot() == 16,
      "ring reports its fixed dimensions");
  test.expect(
      ring.total_pinned_bytes() == 0 && ring.reclaim_completed() == 0,
      "pageable ring reports no pinned or asynchronous storage");
  expect_counts(test, ring, 2, 0, 0, "all slots start free");

  auto first = ring.try_acquire_for_fill();
  auto second = ring.try_acquire_for_fill();
  test.expect(
      first && second && first.slot_index() != second.slot_index(),
      "fill leases reserve distinct slots");
  test.expect(
      first.capacity() == 16 && first.bytes().size() == 16,
      "filling lease exposes full slot capacity");
  test.expect(
      !ring.try_acquire_for_fill(),
      "no fill lease is returned when the ring is full");

  first.bytes()[0] = std::byte{0x2a};
  first.bytes()[6] = std::byte{0x7f};
  first.publish_ready(7);
  test.expect(!first, "publishing consumes the filling lease");
  expect_counts(test, ring, 0, 1, 1, "publish transitions Filling to Ready");

  auto ready = ring.try_acquire_ready();
  test.expect(
      ready && ready.payload_size() == 7 && ready.bytes().size() == 7,
      "ready lease exposes only the published payload");
  test.expect(
      ready.bytes()[0] == std::byte{0x2a} &&
          ready.bytes()[6] == std::byte{0x7f},
      "published bytes survive the queue transition");
  ready.mark_consumed();
  test.expect(!ready, "consuming clears the ready lease");

  second.reset();
  expect_counts(test, ring, 2, 0, 0, "consume and rollback return slots to Free");
}

void test_rollback_and_reuse(TestContext& test) {
  using satview::cpu::PageableRing;

  PageableRing ring{1, 8};
  {
    auto filling = ring.acquire_for_fill();
    expect_throws<std::length_error>(
        test,
        [&] { filling.publish_ready(9); },
        "oversized publication is rejected");
    test.expect(
        static_cast<bool>(filling),
        "failed publication preserves the filling lease");
    expect_counts(
        test, ring, 0, 1, 0,
        "failed publication preserves Filling state");
  }
  expect_counts(test, ring, 1, 0, 0, "filling lease destructor rolls back");

  auto filling = ring.acquire_for_fill();
  const auto original_index = filling.slot_index();
  filling.bytes()[0] = std::byte{0x55};
  filling.publish_ready(1);
  {
    auto ready = ring.acquire_ready();
    test.expect(
        ready.slot_index() == original_index,
        "ready lease refers to the published slot");
  }
  expect_counts(test, ring, 0, 0, 1, "ready lease destructor requeues Ready");

  auto ready = ring.acquire_ready();
  test.expect(
      ready.bytes()[0] == std::byte{0x55},
      "ready rollback preserves the payload");
  ready.mark_consumed();

  for (std::size_t cycle = 0; cycle < 1'000; ++cycle) {
    auto reused = ring.acquire_for_fill();
    test.expect(
        reused.slot_index() == original_index,
        "single slot remains reusable");
    reused.publish_ready(0);
    auto consumed = ring.acquire_ready();
    consumed.mark_consumed();
  }
  expect_counts(test, ring, 1, 0, 0, "repeated reuse preserves ring invariants");
}

void test_stop_aware_acquisition(TestContext& test) {
  using satview::cpu::PageableRing;

  PageableRing ring{1, 8};
  auto held = ring.acquire_for_fill();
  std::atomic<bool> entered{false};
  std::atomic<bool> returned_empty{false};
  std::jthread waiter([&](const std::stop_token stop) {
    entered.store(true, std::memory_order_release);
    auto blocked = ring.acquire_for_fill(stop);
    returned_empty.store(!blocked, std::memory_order_release);
  });
  while (!entered.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  waiter.request_stop();
  waiter.join();
  test.expect(
      returned_empty.load(std::memory_order_acquire),
      "stopping a blocked fill acquisition returns an empty lease");
  held.reset();

  std::stop_source stopped;
  stopped.request_stop();
  test.expect(
      !ring.acquire_ready(stopped.get_token()),
      "pre-stopped ready acquisition returns an empty lease");
  expect_counts(test, ring, 1, 0, 0, "stopped waits do not alter slot state");
}

}  // namespace

int main() {
  TestContext test;
  test_validation_and_lifecycle(test);
  test_rollback_and_reuse(test);
  test_stop_aware_acquisition(test);
  if (test.failures == 0) {
    std::cout << "All PageableRing tests passed\n";
    return 0;
  }
  std::cerr << test.failures << " PageableRing test(s) failed\n";
  return 1;
}
