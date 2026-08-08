#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

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

[[nodiscard]] bool try_set_window_low(
    DisplayWindow& window, float candidate) noexcept;
[[nodiscard]] bool try_set_window_high(
    DisplayWindow& window, float candidate) noexcept;
[[nodiscard]] bool try_set_gamma(float& gamma, float candidate) noexcept;
[[nodiscard]] float display_control_step(
    const DisplayWindow& window) noexcept;
[[nodiscard]] std::optional<DisplayWindow> auto_window(
    const DistributionSummary& summary,
    AutoWindowPreset preset) noexcept;

}  // namespace satview::gpu
