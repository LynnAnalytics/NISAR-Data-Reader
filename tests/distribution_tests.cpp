#include "satview/gpu/distribution.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>

namespace {

void expect(
    const bool condition,
    const std::string_view description,
    int& failures) {
    if (!condition) {
        std::cerr << "Distribution test failed: " << description << '\n';
        ++failures;
    }
}

void test_histogram_summary(int& failures) {
    satview::gpu::DistributionHistogram histogram;
    histogram.finite_count = satview::gpu::kDistributionHistogramBins;
    histogram.invalid_count = 7;
    histogram.minimum = 0.0F;
    histogram.maximum = 256.0F;
    histogram.bins.fill(1);

    const auto summary = satview::gpu::summarize_distribution(histogram);
    expect(summary.has_finite_values(), "uniform histogram is valid", failures);
    expect(summary.histogram.invalid_count == 7, "invalid count retained", failures);
    expect(
        std::abs(summary.percentile_1 - 2.56F) < 1.1F,
        "p1 is inside its histogram error bound",
        failures);
    expect(
        std::abs(summary.percentile_50 - 128.0F) < 1.1F,
        "median is inside its histogram error bound",
        failures);
    expect(
        summary.percentile_1 <= summary.percentile_2 &&
            summary.percentile_2 <= summary.percentile_50 &&
            summary.percentile_50 <= summary.percentile_98 &&
            summary.percentile_98 <= summary.percentile_99,
        "reported percentiles are monotonic",
        failures);
}

void test_empty_and_constant(int& failures) {
    satview::gpu::DistributionHistogram empty;
    empty.invalid_count = 12;
    const auto empty_summary =
        satview::gpu::summarize_distribution(empty);
    expect(
        !empty_summary.has_finite_values(),
        "all-invalid distribution has no finite values",
        failures);
    expect(
        !satview::gpu::auto_window(
             empty_summary,
             satview::gpu::AutoWindowPreset::full_finite_range)
             .has_value(),
        "all-invalid distribution cannot auto-window",
        failures);

    satview::gpu::DistributionHistogram constant;
    constant.finite_count = 20;
    constant.minimum = 42.0F;
    constant.maximum = 42.0F;
    constant.bins[constant.bins.size() / 2] = 20;
    const auto summary = satview::gpu::summarize_distribution(constant);
    expect(
        summary.percentile_1 == 42.0F &&
            summary.percentile_50 == 42.0F &&
            summary.percentile_99 == 42.0F,
        "constant percentiles remain exact",
        failures);
    const auto window = satview::gpu::auto_window(
        summary, satview::gpu::AutoWindowPreset::percentile_1_99);
    expect(
        window.has_value() && window->low < 42.0F &&
            window->high > 42.0F,
        "constant auto-window expands to a useful finite interval",
        failures);
}

void test_extreme_range(int& failures) {
    satview::gpu::DistributionHistogram histogram;
    histogram.finite_count = 2;
    histogram.minimum = -std::numeric_limits<float>::max();
    histogram.maximum = std::numeric_limits<float>::max();
    histogram.bins.front() = 1;
    histogram.bins.back() = 1;
    const auto summary = satview::gpu::summarize_distribution(histogram);
    const auto window = satview::gpu::auto_window(
        summary, satview::gpu::AutoWindowPreset::full_finite_range);
    expect(
        window.has_value() &&
            window->low == -std::numeric_limits<float>::max() &&
            window->high == std::numeric_limits<float>::max(),
        "full-range auto-window preserves finite float extrema",
        failures);
    expect(
        std::isfinite(summary.percentile_1) &&
            std::isfinite(summary.percentile_99),
        "extreme-range percentile interpolation remains finite",
        failures);
}

void test_manual_controls(int& failures) {
    satview::gpu::DisplayWindow window{-10.0F, 10.0F};
    expect(
        satview::gpu::try_set_window_low(window, -20.0F) &&
            window.low == -20.0F && window.high == 10.0F,
        "finite low edit is accepted",
        failures);
    expect(
        !satview::gpu::try_set_window_low(
            window, std::numeric_limits<float>::infinity()) &&
            window.low == -20.0F,
        "non-finite low edit is rejected",
        failures);
    expect(
        satview::gpu::try_set_window_low(window, 10.0F) &&
            window.low == 10.0F && window.low < window.high,
        "crossing low edit advances high by a representable value",
        failures);
    expect(
        satview::gpu::try_set_window_high(window, -5.0F) &&
            window.low < window.high && window.high == -5.0F,
        "crossing high edit retreats low by a representable value",
        failures);

    float gamma = 1.0F;
    expect(
        satview::gpu::try_set_gamma(gamma, 2.25F) && gamma == 2.25F,
        "positive finite gamma is accepted",
        failures);
    expect(
        !satview::gpu::try_set_gamma(gamma, 0.0F) && gamma == 2.25F,
        "zero gamma is rejected",
        failures);
    expect(
        !satview::gpu::try_set_gamma(
            gamma, std::numeric_limits<float>::quiet_NaN()) &&
            gamma == 2.25F,
        "non-finite gamma is rejected",
        failures);
    expect(
        satview::gpu::try_set_gamma(
            gamma, satview::gpu::kMinimumDisplayGamma) &&
            gamma == satview::gpu::kMinimumDisplayGamma,
        "shader gamma floor is accepted exactly",
        failures);
    const float below_gamma_floor = std::nextafter(
        satview::gpu::kMinimumDisplayGamma, 0.0F);
    expect(
        !satview::gpu::try_set_gamma(gamma, below_gamma_floor) &&
            gamma == satview::gpu::kMinimumDisplayGamma,
        "next representable gamma below the shader floor is rejected",
        failures);

    const float step =
        satview::gpu::display_control_step({-35.0F, 5.0F});
    expect(
        std::isfinite(step) && step > 0.0F && step <= 1.0F,
        "window control step is finite and useful",
        failures);
}

}  // namespace

int run_distribution_tests() {
    int failures = 0;
    test_histogram_summary(failures);
    test_empty_and_constant(failures);
    test_extreme_range(failures);
    test_manual_controls(failures);
    if (failures == 0) {
        std::cout << "Distribution tests passed\n";
    }
    return failures;
}
