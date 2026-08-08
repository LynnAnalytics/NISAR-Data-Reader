#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

#include <cuda_runtime_api.h>

namespace satview::gpu {

inline constexpr std::size_t kDistributionHistogramBins = 256;
inline constexpr float kMinimumDisplayGamma = 1.0e-4F;

struct DistributionHistogram {
    std::uint64_t finite_count = 0;
    std::uint64_t invalid_count = 0;
    float minimum = std::numeric_limits<float>::quiet_NaN();
    float maximum = std::numeric_limits<float>::quiet_NaN();
    std::array<std::uint64_t, kDistributionHistogramBins> bins{};
};

struct DistributionSummary {
    DistributionHistogram histogram;
    float percentile_1 = std::numeric_limits<float>::quiet_NaN();
    float percentile_2 = std::numeric_limits<float>::quiet_NaN();
    float percentile_50 = std::numeric_limits<float>::quiet_NaN();
    float percentile_98 = std::numeric_limits<float>::quiet_NaN();
    float percentile_99 = std::numeric_limits<float>::quiet_NaN();

    [[nodiscard]] bool has_finite_values() const noexcept;
};

// Converts an exact finite/invalid count, exact extrema, and fixed histogram
// into approximate percentiles. Quantile interpolation is bounded by the
// selected histogram bin and always remains inside [minimum, maximum].
[[nodiscard]] DistributionSummary summarize_distribution(
    const DistributionHistogram& histogram) noexcept;

struct DisplayWindow {
    float low = 0.0F;
    float high = 1.0F;
};

enum class AutoWindowPreset : std::uint8_t {
    full_finite_range,
    percentile_1_99,
    percentile_2_98,
};

// Manual edits have no artificial data limits. Invalid/non-finite edits are
// rejected, and a successful edit always preserves low < high.
// Gamma follows the shader's numerical floor so the typed and rendered values
// remain identical.
[[nodiscard]] bool try_set_window_low(
    DisplayWindow& window, float candidate) noexcept;
[[nodiscard]] bool try_set_window_high(
    DisplayWindow& window, float candidate) noexcept;
[[nodiscard]] bool try_set_gamma(float& gamma, float candidate) noexcept;

// Returns a useful finite step derived from the current window's order of
// magnitude. It is advisory only and never constrains typed input.
[[nodiscard]] float display_control_step(
    const DisplayWindow& window) noexcept;

// Empty/all-invalid summaries return nullopt. Constant distributions are
// expanded by a small finite representable interval.
[[nodiscard]] std::optional<DisplayWindow> auto_window(
    const DistributionSummary& summary,
    AutoWindowPreset preset) noexcept;

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
