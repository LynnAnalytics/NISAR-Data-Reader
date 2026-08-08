#include "satview/cpu/scientific.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace satview::cpu {
namespace {

[[nodiscard]] bool mask_valid(
    const std::span<const std::uint8_t> validity,
    const std::size_t index) noexcept {
    if (validity.empty()) {
        return true;
    }
    const auto value = validity[index];
    return value != 0 && value != 255;
}

void validate_sizes(
    const std::size_t input,
    const std::size_t output,
    const std::span<const std::uint8_t> validity) {
    if (input != output || (!validity.empty() && validity.size() != input)) {
        throw std::invalid_argument("CPU scientific buffer sizes do not match");
    }
}

template <typename Function>
void parallel_ranges(const std::size_t count, Function function) {
    constexpr std::size_t minimum_items_per_worker = 65'536;
    const auto hardware = std::max(1U, std::thread::hardware_concurrency());
    const auto useful = std::max<std::size_t>(
        1, (count + minimum_items_per_worker - 1) /
               minimum_items_per_worker);
    const auto workers = std::min<std::size_t>(hardware, useful);
    if (workers == 1) {
        function(0, count);
        return;
    }

    std::vector<std::jthread> threads;
    threads.reserve(workers);
    for (std::size_t worker = 0; worker < workers; ++worker) {
        const auto begin = count * worker / workers;
        const auto end = count * (worker + 1) / workers;
        threads.emplace_back([=, &function] { function(begin, end); });
    }
}

[[nodiscard]] float robust_magnitude(const Complex32 sample) noexcept {
    const float power = std::fma(
        sample.real, sample.real, sample.imaginary * sample.imaginary);
    if (std::isfinite(power) && power > 0.0F) {
        return std::sqrt(power);
    }
    if (sample.real == 0.0F && sample.imaginary == 0.0F) {
        return 0.0F;
    }
    return std::hypot(sample.real, sample.imaginary);
}

[[nodiscard]] bool source_valid(
    const float sample, const SpeckleDomain domain) noexcept {
    if (!std::isfinite(sample)) {
        return false;
    }
    return domain == SpeckleDomain::power_db || sample >= 0.0F;
}

[[nodiscard]] float normalized_power(
    const float sample,
    const float scale,
    const SpeckleDomain domain) noexcept {
    if (domain == SpeckleDomain::linear_power) {
        return scale == 0.0F ? 0.0F : sample / scale;
    }
    if (domain == SpeckleDomain::amplitude) {
        if (scale == 0.0F) {
            return 0.0F;
        }
        const float ratio = sample / scale;
        return ratio * ratio;
    }
    constexpr float log2_ten_over_ten = 0.3321928094887362F;
    return std::exp2((sample - scale) * log2_ten_over_ten);
}

[[nodiscard]] float from_normalized_power(
    const float normalized,
    const float scale,
    const SpeckleDomain domain,
    const float epsilon) noexcept {
    if (domain == SpeckleDomain::linear_power) {
        return normalized * scale;
    }
    if (domain == SpeckleDomain::amplitude) {
        return std::sqrt(normalized) * scale;
    }
    const float floor_db = 10.0F * std::log10(epsilon);
    return normalized <= 0.0F
        ? floor_db
        : std::max(scale + 10.0F * std::log10(normalized), floor_db);
}

}  // namespace

void transform_complex(
    const std::span<const Complex32> input,
    const std::span<float> output,
    const ComplexTransform transform,
    const std::span<const std::uint8_t> validity,
    const float epsilon) {
    validate_sizes(input.size(), output.size(), validity);
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        throw std::invalid_argument("CPU transform epsilon must be positive");
    }
    const float invalid = std::numeric_limits<float>::quiet_NaN();
    parallel_ranges(input.size(), [&](const std::size_t begin, const std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            const auto sample = input[index];
            if (!mask_valid(validity, index) ||
                !std::isfinite(sample.real) ||
                !std::isfinite(sample.imaginary)) {
                output[index] = invalid;
                continue;
            }
            float value = invalid;
            const float power = std::fma(
                sample.real, sample.real,
                sample.imaginary * sample.imaginary);
            switch (transform) {
                case ComplexTransform::amplitude:
                    value = robust_magnitude(sample);
                    break;
                case ComplexTransform::power:
                    value = power;
                    break;
                case ComplexTransform::power_db:
                    if (std::isfinite(power)) {
                        value = 10.0F * std::log10(std::max(power, epsilon));
                    } else {
                        const float high = std::max(
                            std::abs(sample.real), std::abs(sample.imaginary));
                        const float low = std::min(
                            std::abs(sample.real), std::abs(sample.imaginary));
                        const float ratio = low / high;
                        value = std::max(
                            20.0F * std::log10(high) +
                                10.0F * std::log10(std::fma(ratio, ratio, 1.0F)),
                            10.0F * std::log10(epsilon));
                    }
                    break;
                case ComplexTransform::phase:
                    value = std::atan2(sample.imaginary, sample.real);
                    break;
                case ComplexTransform::real:
                    value = sample.real;
                    break;
                case ComplexTransform::imaginary:
                    value = sample.imaginary;
                    break;
            }
            output[index] = std::isfinite(value) ? value : invalid;
        }
    });
}

void transform_real(
    const std::span<const float> input,
    const std::span<float> output,
    const RealTransform transform,
    const std::span<const std::uint8_t> validity,
    const float epsilon) {
    validate_sizes(input.size(), output.size(), validity);
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        throw std::invalid_argument("CPU transform epsilon must be positive");
    }
    const float invalid = std::numeric_limits<float>::quiet_NaN();
    parallel_ranges(input.size(), [&](const std::size_t begin, const std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            const float sample = input[index];
            if (!mask_valid(validity, index) ||
                !std::isfinite(sample) || sample < 0.0F) {
                output[index] = invalid;
                continue;
            }
            output[index] = transform == RealTransform::linear
                ? sample
                : 10.0F * std::log10(std::max(sample, epsilon));
        }
    });
}

void filter_speckle(
    const std::span<const float> input,
    const std::span<float> output,
    const std::size_t width,
    const std::size_t height,
    const SpeckleOptions& options,
    const std::span<const std::uint8_t> validity) {
    if (width == 0 || height == 0 || width > input.size() / height ||
        width * height != input.size()) {
        throw std::invalid_argument("CPU speckle dimensions are invalid");
    }
    validate_sizes(input.size(), output.size(), validity);
    if (options.filter == SpeckleFilter::none) {
        const float invalid = std::numeric_limits<float>::quiet_NaN();
        parallel_ranges(
            input.size(),
            [&](const std::size_t begin, const std::size_t end) {
                for (std::size_t index = begin; index < end; ++index) {
                    const float sample = input[index];
                    output[index] = mask_valid(validity, index) &&
                            std::isfinite(sample)
                        ? sample
                        : invalid;
                }
            });
        return;
    }
    if ((options.window_size != 3 && options.window_size != 5 &&
         options.window_size != 7) ||
        !std::isfinite(options.equivalent_number_of_looks) ||
        options.equivalent_number_of_looks <= 0.0F ||
        !std::isfinite(options.power_epsilon) ||
        options.power_epsilon <= 0.0F) {
        throw std::invalid_argument("CPU speckle options are invalid");
    }

    const auto radius = static_cast<std::size_t>(options.window_size / 2);
    const float invalid = std::numeric_limits<float>::quiet_NaN();
    parallel_ranges(height, [&](const std::size_t row_begin, const std::size_t row_end) {
        for (std::size_t row = row_begin; row < row_end; ++row) {
            for (std::size_t column = 0; column < width; ++column) {
                const auto center_index = row * width + column;
                const float center_sample = input[center_index];
                if (!mask_valid(validity, center_index) ||
                    !source_valid(center_sample, options.domain)) {
                    output[center_index] = invalid;
                    continue;
                }

                const auto first_row = row > radius ? row - radius : 0;
                const auto last_row = std::min(height - 1, row + radius);
                const auto first_column =
                    column > radius ? column - radius : 0;
                const auto last_column =
                    std::min(width - 1, column + radius);
                float scale = center_sample;
                for (auto y = first_row; y <= last_row; ++y) {
                    for (auto x = first_column; x <= last_column; ++x) {
                        const auto index = y * width + x;
                        const float sample = input[index];
                        if (mask_valid(validity, index) &&
                            source_valid(sample, options.domain)) {
                            scale = std::max(scale, sample);
                        }
                    }
                }

                float mean = 0.0F;
                float m2 = 0.0F;
                std::uint32_t count = 0;
                for (auto y = first_row; y <= last_row; ++y) {
                    for (auto x = first_column; x <= last_column; ++x) {
                        const auto index = y * width + x;
                        const float sample = input[index];
                        if (!mask_valid(validity, index) ||
                            !source_valid(sample, options.domain)) {
                            continue;
                        }
                        const float normalized = normalized_power(
                            sample, scale, options.domain);
                        ++count;
                        const float delta = normalized - mean;
                        mean += delta / static_cast<float>(count);
                        if (options.filter == SpeckleFilter::lee) {
                            m2 = std::fma(delta, normalized - mean, m2);
                        }
                    }
                }

                float filtered = mean;
                if (options.filter == SpeckleFilter::lee) {
                    const float variance =
                        std::max(m2 / static_cast<float>(count), 0.0F);
                    const float noise = mean * mean /
                        options.equivalent_number_of_looks;
                    float weight = 0.0F;
                    if (variance > noise && variance > 0.0F) {
                        weight = (variance - noise) / variance;
                    }
                    weight = std::clamp(weight, 0.0F, 1.0F);
                    const float center = normalized_power(
                        center_sample, scale, options.domain);
                    filtered = std::fma(weight, center - mean, mean);
                }
                filtered = std::clamp(filtered, 0.0F, 1.0F);
                const float result = from_normalized_power(
                    filtered, scale, options.domain, options.power_epsilon);
                output[center_index] =
                    std::isfinite(result) ? result : invalid;
            }
        }
    });
}

Histogram build_histogram(const std::span<const float> values) {
    Histogram result;
    result.minimum = std::numeric_limits<float>::infinity();
    result.maximum = -std::numeric_limits<float>::infinity();
    for (const float value : values) {
        if (std::isfinite(value)) {
            ++result.finite_count;
            result.minimum = std::min(result.minimum, value);
            result.maximum = std::max(result.maximum, value);
        } else {
            ++result.invalid_count;
        }
    }
    if (result.finite_count == 0) {
        result.minimum = std::numeric_limits<float>::quiet_NaN();
        result.maximum = std::numeric_limits<float>::quiet_NaN();
        return result;
    }

    const bool constant = result.minimum == result.maximum;
    const double range = static_cast<double>(result.maximum) - result.minimum;
    for (const float value : values) {
        if (!std::isfinite(value)) {
            continue;
        }
        std::size_t bin = kHistogramBins / 2;
        if (!constant) {
            const double scaled =
                (static_cast<double>(value) - result.minimum) *
                static_cast<double>(kHistogramBins) / range;
            bin = scaled <= 0.0
                ? 0
                : scaled >= static_cast<double>(kHistogramBins)
                    ? kHistogramBins - 1
                    : static_cast<std::size_t>(scaled);
        }
        ++result.bins[bin];
    }
    return result;
}

}  // namespace satview::cpu
