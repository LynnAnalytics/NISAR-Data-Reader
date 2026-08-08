#include "satview/distribution.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace satview::gpu {
namespace {

[[nodiscard]] std::uint64_t saturating_sum(
    const std::array<std::uint64_t, kDistributionHistogramBins>& bins) noexcept {
  std::uint64_t total = 0;
  for (const std::uint64_t count : bins) {
    if (count > std::numeric_limits<std::uint64_t>::max() - total) {
      return std::numeric_limits<std::uint64_t>::max();
    }
    total += count;
  }
  return total;
}

[[nodiscard]] float approximate_quantile(
    const DistributionHistogram& histogram,
    const double quantile) noexcept {
  if (histogram.finite_count == 0 ||
      !std::isfinite(histogram.minimum) ||
      !std::isfinite(histogram.maximum) ||
      histogram.minimum > histogram.maximum) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (histogram.minimum == histogram.maximum) {
    return histogram.minimum;
  }
  const std::uint64_t histogram_count = saturating_sum(histogram.bins);
  if (histogram_count == 0) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const long double target =
      static_cast<long double>(std::clamp(quantile, 0.0, 1.0)) *
      static_cast<long double>(histogram_count - 1);
  std::uint64_t before = 0;
  std::size_t selected_bin = kDistributionHistogramBins - 1;
  for (std::size_t index = 0; index < kDistributionHistogramBins; ++index) {
    const std::uint64_t count = histogram.bins[index];
    const long double after =
        static_cast<long double>(before) + static_cast<long double>(count);
    if (count != 0 && target < after) {
      selected_bin = index;
      break;
    }
    before = count > std::numeric_limits<std::uint64_t>::max() - before
        ? std::numeric_limits<std::uint64_t>::max()
        : before + count;
  }
  const std::uint64_t bin_count = histogram.bins[selected_bin];
  const long double within = bin_count == 0
      ? 0.5L
      : std::clamp(
            (target - static_cast<long double>(before) + 0.5L) /
                static_cast<long double>(bin_count),
            0.0L,
            1.0L);
  const long double normalized =
      (static_cast<long double>(selected_bin) + within) /
      static_cast<long double>(kDistributionHistogramBins);
  const long double value =
      static_cast<long double>(histogram.minimum) + normalized *
          (static_cast<long double>(histogram.maximum) -
           static_cast<long double>(histogram.minimum));
  return std::clamp(
      static_cast<float>(value), histogram.minimum, histogram.maximum);
}

[[nodiscard]] std::optional<DisplayWindow> finite_window(
    const float first, const float second) noexcept {
  if (!std::isfinite(first) || !std::isfinite(second)) {
    return std::nullopt;
  }
  const float low = std::min(first, second);
  const float high = std::max(first, second);
  if (low < high) {
    return DisplayWindow{low, high};
  }
  const double center = static_cast<double>(low);
  const double delta = std::max(std::abs(center) * 1.0e-6, 1.0e-6);
  const double finite_max =
      static_cast<double>(std::numeric_limits<float>::max());
  const float expanded_low = static_cast<float>(std::max(
      -finite_max, center - delta));
  const float expanded_high = static_cast<float>(std::min(
      finite_max, center + delta));
  if (std::isfinite(expanded_low) && std::isfinite(expanded_high) &&
      expanded_low < expanded_high) {
    return DisplayWindow{expanded_low, expanded_high};
  }
  const float below = std::nextafter(
      low, -std::numeric_limits<float>::infinity());
  const float above = std::nextafter(
      high, std::numeric_limits<float>::infinity());
  if (std::isfinite(below) && below < high) {
    return DisplayWindow{below, high};
  }
  if (std::isfinite(above) && low < above) {
    return DisplayWindow{low, above};
  }
  return std::nullopt;
}

}  // namespace

bool DistributionSummary::has_finite_values() const noexcept {
  return histogram.finite_count != 0 &&
      std::isfinite(histogram.minimum) &&
      std::isfinite(histogram.maximum) &&
      histogram.minimum <= histogram.maximum;
}

DistributionSummary summarize_distribution(
    const DistributionHistogram& histogram) noexcept {
  DistributionSummary result;
  result.histogram = histogram;
  if (!result.has_finite_values()) {
    return result;
  }
  result.percentile_1 = approximate_quantile(histogram, 0.01);
  result.percentile_2 = approximate_quantile(histogram, 0.02);
  result.percentile_50 = approximate_quantile(histogram, 0.50);
  result.percentile_98 = approximate_quantile(histogram, 0.98);
  result.percentile_99 = approximate_quantile(histogram, 0.99);
  return result;
}

bool try_set_window_low(
    DisplayWindow& window, const float candidate) noexcept {
  if (!std::isfinite(candidate) || !std::isfinite(window.high)) {
    return false;
  }
  if (candidate < window.high) {
    window.low = candidate;
    return true;
  }
  const float adjusted_high = std::nextafter(
      candidate, std::numeric_limits<float>::infinity());
  if (!std::isfinite(adjusted_high)) {
    return false;
  }
  window.low = candidate;
  window.high = adjusted_high;
  return true;
}

bool try_set_window_high(
    DisplayWindow& window, const float candidate) noexcept {
  if (!std::isfinite(candidate) || !std::isfinite(window.low)) {
    return false;
  }
  if (window.low < candidate) {
    window.high = candidate;
    return true;
  }
  const float adjusted_low = std::nextafter(
      candidate, -std::numeric_limits<float>::infinity());
  if (!std::isfinite(adjusted_low)) {
    return false;
  }
  window.low = adjusted_low;
  window.high = candidate;
  return true;
}

bool try_set_gamma(float& gamma, const float candidate) noexcept {
  if (!std::isfinite(candidate) || candidate < kMinimumDisplayGamma) {
    return false;
  }
  gamma = candidate;
  return true;
}

float display_control_step(const DisplayWindow& window) noexcept {
  const double low = static_cast<double>(window.low);
  const double high = static_cast<double>(window.high);
  double scale = high - low;
  if (!std::isfinite(scale) || scale <= 0.0) {
    scale = std::max({std::abs(low), std::abs(high), 1.0});
  }
  const double raw_step = scale / 100.0;
  double step = std::pow(10.0, std::floor(std::log10(raw_step)));
  if (!std::isfinite(step) || step <= 0.0) {
    step = static_cast<double>(std::numeric_limits<float>::min());
  }
  const float reference = std::abs(window.low) >= std::abs(window.high)
      ? window.low
      : window.high;
  const float adjacent = std::nextafter(
      reference,
      reference < std::numeric_limits<float>::max()
          ? std::numeric_limits<float>::infinity()
          : -std::numeric_limits<float>::infinity());
  const double ulp = std::abs(
      static_cast<double>(adjacent) - static_cast<double>(reference));
  step = std::max(step, ulp);
  step = std::clamp(
      step,
      static_cast<double>(std::numeric_limits<float>::min()),
      static_cast<double>(std::numeric_limits<float>::max()));
  return static_cast<float>(step);
}

std::optional<DisplayWindow> auto_window(
    const DistributionSummary& summary,
    const AutoWindowPreset preset) noexcept {
  if (!summary.has_finite_values()) {
    return std::nullopt;
  }
  switch (preset) {
    case AutoWindowPreset::full_finite_range:
      return finite_window(
          summary.histogram.minimum, summary.histogram.maximum);
    case AutoWindowPreset::percentile_1_99:
      return finite_window(summary.percentile_1, summary.percentile_99);
    case AutoWindowPreset::percentile_2_98:
      return finite_window(summary.percentile_2, summary.percentile_98);
  }
  return std::nullopt;
}

}  // namespace satview::gpu
