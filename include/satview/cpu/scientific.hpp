#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace satview::cpu {

struct Complex32 {
    float real = 0.0F;
    float imaginary = 0.0F;
};

enum class ComplexTransform : std::uint8_t {
    amplitude,
    power,
    power_db,
    phase,
    real,
    imaginary,
};

enum class RealTransform : std::uint8_t {
    linear,
    power_db,
};

enum class SpeckleDomain : std::uint8_t {
    amplitude,
    linear_power,
    power_db,
};

enum class SpeckleFilter : std::uint8_t {
    none,
    boxcar,
    lee,
};

inline constexpr float kTransformEpsilon = 1.0e-20F;
inline constexpr std::size_t kHistogramBins = 256;

struct SpeckleOptions {
    SpeckleFilter filter = SpeckleFilter::boxcar;
    SpeckleDomain domain = SpeckleDomain::linear_power;
    std::uint32_t window_size = 5;
    float equivalent_number_of_looks = 1.0F;
    float power_epsilon = kTransformEpsilon;
};

struct Histogram {
    std::uint64_t finite_count = 0;
    std::uint64_t invalid_count = 0;
    float minimum = 0.0F;
    float maximum = 0.0F;
    std::array<std::uint64_t, kHistogramBins> bins{};
};

void transform_complex(
    std::span<const Complex32> input,
    std::span<float> output,
    ComplexTransform transform,
    std::span<const std::uint8_t> validity = {},
    float epsilon = kTransformEpsilon);

void transform_real(
    std::span<const float> input,
    std::span<float> output,
    RealTransform transform,
    std::span<const std::uint8_t> validity = {},
    float epsilon = kTransformEpsilon);

void filter_speckle(
    std::span<const float> input,
    std::span<float> output,
    std::size_t width,
    std::size_t height,
    const SpeckleOptions& options,
    std::span<const std::uint8_t> validity = {});

[[nodiscard]] Histogram build_histogram(std::span<const float> values);

}  // namespace satview::cpu
