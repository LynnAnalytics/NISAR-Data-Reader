#include "overview_worker.hpp"

#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

void expect(
    const bool condition,
    const std::string_view message,
    int& failures) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

}  // namespace

int run_overview_worker_tests(const satview::Hdf5Product& product) {
    int failures = 0;
    satview::viewer::OverviewWorker worker(product);

    constexpr std::uint64_t idle_serial = 41;
    worker.supersede(idle_serial);
    const auto idle = worker.progress();
    expect(
        idle.request_serial == idle_serial && !idle.active,
        "superseding an idle worker advances its inactive progress serial",
        failures);

    worker.set_foreground_active(true);
    constexpr std::uint64_t request_serial = 42;
    constexpr std::size_t layer_index = 7;
    worker.request(satview::viewer::OverviewWorkerRequest{
        .request_serial = request_serial,
        .layer_index = layer_index,
    });
    const auto requested = worker.progress();
    expect(
        requested.request_serial == request_serial &&
            requested.layer_index == layer_index && requested.active,
        "a queued request publishes its active progress generation",
        failures);

    constexpr std::uint64_t replacement_serial = 43;
    worker.supersede(replacement_serial);
    const auto superseded = worker.progress();
    expect(
        superseded.request_serial == replacement_serial &&
            !superseded.active,
        "supersession atomically replaces and deactivates progress",
        failures);

    worker.set_foreground_active(false);
    worker.stop();
    const auto stopped = worker.progress();
    expect(
        stopped.request_serial == replacement_serial && !stopped.active,
        "a released stale request cannot reactivate superseded progress",
        failures);
    expect(
        !worker.try_take_completion().has_value(),
        "a superseded queued request cannot publish a completion",
        failures);
    return failures;
}
