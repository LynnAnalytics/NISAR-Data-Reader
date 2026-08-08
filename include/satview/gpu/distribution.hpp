#pragma once

#include "satview/distribution.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

#include <cuda_runtime_api.h>

namespace satview::gpu {

struct AsyncDistributionResult {
    std::uint64_t generation = 0;
    DistributionSummary summary;
    float elapsed_milliseconds = 0.0F;
};

// Computes statistics from an already transformed device-resident R32F page.
// enqueue() is stream ordered and asynchronous. poll() performs only an event
// query and consumes a pinned host copy of the fixed-size result when ready.
class AsyncResidentDistribution final {
public:
    AsyncResidentDistribution();
    ~AsyncResidentDistribution() noexcept;

    AsyncResidentDistribution(const AsyncResidentDistribution&) = delete;
    AsyncResidentDistribution& operator=(
        const AsyncResidentDistribution&) = delete;

    [[nodiscard]] bool pending() const noexcept;

    void enqueue(
        const float* device_values,
        std::size_t count,
        cudaStream_t stream,
        std::uint64_t generation);

    [[nodiscard]] bool poll(AsyncDistributionResult& result);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace satview::gpu
