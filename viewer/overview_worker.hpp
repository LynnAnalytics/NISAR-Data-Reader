#pragma once

#include "satview/overview_builder.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace satview::viewer {

struct OverviewWorkerRequest {
    std::uint64_t request_serial = 0;
    std::size_t layer_index = 0;
    overview::OverviewRequest overview_request;
};

struct OverviewWorkerProgress {
    std::uint64_t request_serial = 0;
    std::size_t layer_index = 0;
    overview::OverviewProgress overview_progress;
    bool active = false;
};

struct OverviewWorkerCompletion {
    std::uint64_t request_serial = 0;
    std::size_t layer_index = 0;
    overview::OverviewPrepareStatus status =
        overview::OverviewPrepareStatus::cancelled;
    std::optional<overview::OverviewPlan> plan;
    std::vector<std::byte> science_bytes;
    std::vector<std::byte> mask_bytes;
    double elapsed_milliseconds = 0.0;
    std::string error;

    [[nodiscard]] bool ready() const noexcept {
        return error.empty() && plan.has_value() &&
            status != overview::OverviewPrepareStatus::cancelled;
    }

    [[nodiscard]] bool built() const noexcept {
        return ready() && status == overview::OverviewPrepareStatus::built;
    }
};

// Owns exactly one overview-preparation thread. New requests replace queued
// work and supersede the active serial. The core builder observes that
// supersession through its per-source-chunk cancellation callback.
// The referenced Hdf5Product must outlive the worker; stop() joins all work
// before the product may be moved or destroyed.
class OverviewWorker final {
public:
    explicit OverviewWorker(const Hdf5Product& product);
    ~OverviewWorker();

    OverviewWorker(const OverviewWorker&) = delete;
    OverviewWorker& operator=(const OverviewWorker&) = delete;
    OverviewWorker(OverviewWorker&&) = delete;
    OverviewWorker& operator=(OverviewWorker&&) = delete;

    void request(OverviewWorkerRequest request);
    void supersede(std::uint64_t request_serial) noexcept;
    // Pauses only between source chunks, allowing the latency-sensitive native
    // tile reader to acquire the process-wide HDF5 lock first.
    void set_foreground_active(bool active) noexcept;

    [[nodiscard]] OverviewWorkerProgress progress() const;
    [[nodiscard]] std::optional<OverviewWorkerCompletion>
    try_take_completion();
    void stop() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace satview::viewer
