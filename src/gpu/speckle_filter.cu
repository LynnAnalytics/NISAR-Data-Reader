#include "satview/gpu/speckle_filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>

namespace satview::gpu {
namespace {

constexpr unsigned int kBlockSide = 16;
constexpr unsigned int kThreadsPerBlock = kBlockSide * kBlockSide;
constexpr std::size_t kMaxLinearBlocks = 65535;
constexpr std::size_t kMaxGridX = 2147483647ULL;
constexpr std::size_t kMaxGridY = 65535;

[[nodiscard]] constexpr std::size_t ceil_div(
    const std::size_t value,
    const std::size_t divisor) noexcept {
    return value == 0 ? 0 : 1 + ((value - 1) / divisor);
}

[[nodiscard]] bool valid_domain(const SpeckleDomain domain) noexcept {
    switch (domain) {
        case SpeckleDomain::amplitude:
        case SpeckleDomain::linear_power:
        case SpeckleDomain::power_db:
        case SpeckleDomain::signed_value:
        case SpeckleDomain::phase:
            return true;
    }
    return false;
}

[[nodiscard]] bool filterable_domain(const SpeckleDomain domain) noexcept {
    return domain == SpeckleDomain::amplitude ||
           domain == SpeckleDomain::linear_power ||
           domain == SpeckleDomain::power_db;
}

[[nodiscard]] bool supported_window(const std::uint32_t window) noexcept {
    return window == 3 || window == 5 || window == 7;
}

[[nodiscard]] bool positive_finite(const float value) noexcept {
    return std::isfinite(value) && value > 0.0F;
}

__device__ __forceinline__ float quiet_nan() noexcept {
    return __int_as_float(0x7fc00000);
}

__device__ __forceinline__ bool mask_value_is_valid(
    const std::uint8_t value) noexcept {
    return value != 0 && value != 255;
}

template <SpeckleDomain Domain>
__device__ __forceinline__ bool source_is_valid(
    const float sample) noexcept {
    if (!isfinite(sample)) {
        return false;
    }
    if constexpr (
        Domain == SpeckleDomain::amplitude ||
        Domain == SpeckleDomain::linear_power) {
        return sample >= 0.0F;
    } else {
        static_assert(Domain == SpeckleDomain::power_db);
        return true;
    }
}

template <SpeckleDomain Domain>
__device__ __forceinline__ float normalized_power(
    const float sample,
    const float scale) noexcept {
    if constexpr (Domain == SpeckleDomain::linear_power) {
        return scale == 0.0F ? 0.0F : sample / scale;
    } else if constexpr (Domain == SpeckleDomain::amplitude) {
        if (scale == 0.0F) {
            return 0.0F;
        }
        const float ratio = sample / scale;
        return ratio * ratio;
    } else {
        static_assert(Domain == SpeckleDomain::power_db);
        // Relative conversion cannot overflow because scale is the maximum
        // finite dB value in the current neighborhood.
        constexpr float kLog2TenOverTen = 0.3321928094887362F;
        return exp2f((sample - scale) * kLog2TenOverTen);
    }
}

template <SpeckleDomain Domain>
__device__ __forceinline__ float from_normalized_power(
    const float normalized,
    const float scale,
    const float epsilon) noexcept {
    if constexpr (Domain == SpeckleDomain::linear_power) {
        return normalized * scale;
    } else if constexpr (Domain == SpeckleDomain::amplitude) {
        return sqrtf(normalized) * scale;
    } else {
        static_assert(Domain == SpeckleDomain::power_db);
        const float floor_db = 10.0F * log10f(epsilon);
        if (normalized <= 0.0F) {
            return floor_db;
        }
        return fmaxf(
            scale + 10.0F * log10f(normalized),
            floor_db);
    }
}

__device__ __forceinline__ void store_invalid(
    float* const output,
    std::uint8_t* const output_validity,
    const std::size_t index) noexcept {
    output[index] = quiet_nan();
    if (output_validity != nullptr) {
        output_validity[index] = 0;
    }
}

__global__ void passthrough_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const std::size_t count,
    const std::uint8_t* input_validity,
    std::uint8_t* output_validity) {
    const std::size_t first =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t index = first; index < count; index += stride) {
        const float sample = input[index];
        const bool valid =
            (input_validity == nullptr ||
             mask_value_is_valid(input_validity[index])) &&
            isfinite(sample);
        output[index] = valid ? sample : quiet_nan();
        if (output_validity != nullptr) {
            output_validity[index] = static_cast<std::uint8_t>(valid);
        }
    }
}

template <int Radius, SpeckleDomain Domain, SpeckleFilter Filter>
__global__ void neighborhood_filter_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const std::size_t width,
    const std::size_t height,
    const std::uint8_t* input_validity,
    std::uint8_t* output_validity,
    const float equivalent_looks,
    const float power_epsilon) {
    static_assert(Filter == SpeckleFilter::boxcar ||
                  Filter == SpeckleFilter::lee);
    constexpr int kTileSide =
        static_cast<int>(kBlockSide) + 2 * Radius;
    constexpr int kTileElements = kTileSide * kTileSide;

    __shared__ float tile_samples[kTileElements];
    __shared__ std::uint8_t tile_validity[kTileElements];

    const unsigned int linear_thread =
        threadIdx.y * kBlockSide + threadIdx.x;
    const std::int64_t block_origin_x =
        static_cast<std::int64_t>(blockIdx.x) * kBlockSide;
    const std::int64_t block_origin_y =
        static_cast<std::int64_t>(blockIdx.y) * kBlockSide;

    // Cooperatively stage the block and its halo. Out-of-image cells are
    // represented as invalid, which naturally clips edge windows.
    for (int tile_index = static_cast<int>(linear_thread);
         tile_index < kTileElements;
         tile_index += static_cast<int>(kThreadsPerBlock)) {
        const int tile_y = tile_index / kTileSide;
        const int tile_x = tile_index - tile_y * kTileSide;
        const std::int64_t global_x =
            block_origin_x + tile_x - Radius;
        const std::int64_t global_y =
            block_origin_y + tile_y - Radius;

        bool valid =
            global_x >= 0 && global_y >= 0 &&
            static_cast<std::size_t>(global_x) < width &&
            static_cast<std::size_t>(global_y) < height;
        float sample = 0.0F;
        if (valid) {
            const std::size_t index =
                static_cast<std::size_t>(global_y) * width +
                static_cast<std::size_t>(global_x);
            valid =
                input_validity == nullptr ||
                mask_value_is_valid(input_validity[index]);
            if (valid) {
                sample = input[index];
                valid = source_is_valid<Domain>(sample);
            }
        }

        tile_samples[tile_index] = valid ? sample : 0.0F;
        tile_validity[tile_index] = static_cast<std::uint8_t>(valid);
    }
    __syncthreads();

    const std::size_t x =
        static_cast<std::size_t>(blockIdx.x) * kBlockSide + threadIdx.x;
    const std::size_t y =
        static_cast<std::size_t>(blockIdx.y) * kBlockSide + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const std::size_t output_index = y * width + x;
    const int center_x = static_cast<int>(threadIdx.x) + Radius;
    const int center_y = static_cast<int>(threadIdx.y) + Radius;
    const int center_index = center_y * kTileSide + center_x;
    if (tile_validity[center_index] == 0) {
        store_invalid(output, output_validity, output_index);
        return;
    }

    // Normalize relative to the brightest power in this exact neighborhood.
    // Amplitude ratios are squared, power is divided directly, and dB uses a
    // non-positive relative exponent. Thus every statistic stays in [0, 1]
    // even when absolute linear power would overflow or underflow float.
    float scale = tile_samples[center_index];
#pragma unroll
    for (int dy = -Radius; dy <= Radius; ++dy) {
#pragma unroll
        for (int dx = -Radius; dx <= Radius; ++dx) {
            const int local =
                (center_y + dy) * kTileSide + center_x + dx;
            if (tile_validity[local] != 0) {
                scale = fmaxf(scale, tile_samples[local]);
            }
        }
    }

    float mean = 0.0F;
    float m2 = 0.0F;
    unsigned int count = 0;
#pragma unroll
    for (int dy = -Radius; dy <= Radius; ++dy) {
#pragma unroll
        for (int dx = -Radius; dx <= Radius; ++dx) {
            const int local =
                (center_y + dy) * kTileSide + center_x + dx;
            if (tile_validity[local] != 0) {
                const float normalized = normalized_power<Domain>(
                    tile_samples[local], scale);
                ++count;
                const float delta = normalized - mean;
                mean += delta / static_cast<float>(count);
                if constexpr (Filter == SpeckleFilter::lee) {
                    const float delta_after = normalized - mean;
                    m2 = fmaf(delta, delta_after, m2);
                }
            }
        }
    }

    float filtered_normalized = mean;
    if constexpr (Filter == SpeckleFilter::lee) {
        const float variance =
            fmaxf(m2 / static_cast<float>(count), 0.0F);
        const float noise_variance =
            (mean * mean) / equivalent_looks;
        float weight = 0.0F;
        if (variance > noise_variance && variance > 0.0F) {
            weight = (variance - noise_variance) / variance;
        }
        weight = fminf(fmaxf(weight, 0.0F), 1.0F);

        const float center = normalized_power<Domain>(
            tile_samples[center_index], scale);
        filtered_normalized = fmaf(weight, center - mean, mean);
    }
    filtered_normalized =
        fminf(fmaxf(filtered_normalized, 0.0F), 1.0F);

    const float result = from_normalized_power<Domain>(
        filtered_normalized, scale, power_epsilon);
    if (!isfinite(result)) {
        store_invalid(output, output_validity, output_index);
        return;
    }
    output[output_index] = result;
    if (output_validity != nullptr) {
        output_validity[output_index] = 1;
    }
}

[[nodiscard]] cudaError_t launch_passthrough(
    const float* const input,
    float* const output,
    const std::size_t count,
    const SpeckleFilterOptions options) noexcept {
    const std::size_t needed =
        ceil_div(count, static_cast<std::size_t>(kThreadsPerBlock));
    const unsigned int blocks = static_cast<unsigned int>(
        std::min(needed, kMaxLinearBlocks));
    passthrough_kernel<<<blocks, kThreadsPerBlock, 0, options.stream>>>(
        input,
        output,
        count,
        options.validity.input,
        options.validity.output);
    return cudaPeekAtLastError();
}

template <int Radius, SpeckleDomain Domain, SpeckleFilter Filter>
[[nodiscard]] cudaError_t launch_neighborhood(
    const float* const input,
    float* const output,
    const std::size_t width,
    const std::size_t height,
    const SpeckleFilterOptions options) noexcept {
    const dim3 threads{kBlockSide, kBlockSide, 1};
    const dim3 blocks{
        static_cast<unsigned int>(ceil_div(width, kBlockSide)),
        static_cast<unsigned int>(ceil_div(height, kBlockSide)),
        1};
    neighborhood_filter_kernel<Radius, Domain, Filter>
        <<<blocks, threads, 0, options.stream>>>(
            input,
            output,
            width,
            height,
            options.validity.input,
            options.validity.output,
            options.equivalent_number_of_looks,
            options.power_epsilon);
    return cudaPeekAtLastError();
}

template <SpeckleDomain Domain, SpeckleFilter Filter>
[[nodiscard]] cudaError_t launch_window(
    const float* const input,
    float* const output,
    const std::size_t width,
    const std::size_t height,
    const SpeckleFilterOptions options) noexcept {
    switch (options.window_size) {
        case 3:
            return launch_neighborhood<1, Domain, Filter>(
                input, output, width, height, options);
        case 5:
            return launch_neighborhood<2, Domain, Filter>(
                input, output, width, height, options);
        case 7:
            return launch_neighborhood<3, Domain, Filter>(
                input, output, width, height, options);
        default:
            return cudaErrorInvalidValue;
    }
}

template <SpeckleFilter Filter>
[[nodiscard]] cudaError_t launch_domain(
    const float* const input,
    float* const output,
    const std::size_t width,
    const std::size_t height,
    const SpeckleFilterOptions options) noexcept {
    switch (options.domain) {
        case SpeckleDomain::amplitude:
            return launch_window<
                SpeckleDomain::amplitude,
                Filter>(input, output, width, height, options);
        case SpeckleDomain::linear_power:
            return launch_window<
                SpeckleDomain::linear_power,
                Filter>(input, output, width, height, options);
        case SpeckleDomain::power_db:
            return launch_window<
                SpeckleDomain::power_db,
                Filter>(input, output, width, height, options);
        case SpeckleDomain::signed_value:
        case SpeckleDomain::phase:
            return cudaErrorNotSupported;
    }
    return cudaErrorInvalidValue;
}

}  // namespace

cudaError_t launch_speckle_filter(
    const float* const input,
    float* const output,
    const std::size_t width,
    const std::size_t height,
    const SpeckleFilterOptions options) noexcept {
    if (width == 0 || height == 0) {
        return cudaSuccess;
    }
    if (width > std::numeric_limits<std::size_t>::max() / height) {
        return cudaErrorInvalidValue;
    }
    const std::size_t count = width * height;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        input == nullptr || output == nullptr || input == output ||
        (options.validity.input != nullptr &&
         options.validity.input == options.validity.output) ||
        !valid_domain(options.domain)) {
        return cudaErrorInvalidValue;
    }

    switch (options.filter) {
        case SpeckleFilter::none:
            return launch_passthrough(input, output, count, options);
        case SpeckleFilter::boxcar:
        case SpeckleFilter::lee:
            break;
        default:
            return cudaErrorInvalidValue;
    }

    if (!filterable_domain(options.domain)) {
        return cudaErrorNotSupported;
    }
    if (!supported_window(options.window_size)) {
        return cudaErrorInvalidValue;
    }
    if (options.domain == SpeckleDomain::power_db &&
        !positive_finite(options.power_epsilon)) {
        return cudaErrorInvalidValue;
    }

    const std::size_t grid_x = ceil_div(width, kBlockSide);
    const std::size_t grid_y = ceil_div(height, kBlockSide);
    if (grid_x > kMaxGridX || grid_y > kMaxGridY ||
        width > static_cast<std::size_t>(
                    std::numeric_limits<std::int64_t>::max()) ||
        height > static_cast<std::size_t>(
                     std::numeric_limits<std::int64_t>::max())) {
        return cudaErrorInvalidValue;
    }

    if (options.filter == SpeckleFilter::boxcar) {
        return launch_domain<SpeckleFilter::boxcar>(
            input, output, width, height, options);
    }
    if (!positive_finite(options.equivalent_number_of_looks)) {
        return cudaErrorInvalidValue;
    }
    return launch_domain<SpeckleFilter::lee>(
        input, output, width, height, options);
}

}  // namespace satview::gpu
