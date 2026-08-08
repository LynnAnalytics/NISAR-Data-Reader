#include "satview/gpu/transforms.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace satview::gpu {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;
// A grid-stride loop keeps very large rasters supported without launching an
// unnecessarily large grid or querying the device from the launch path.
constexpr std::size_t kMaxBlocks = 65535;

[[nodiscard]] unsigned int block_count(const std::size_t count) noexcept {
    const std::size_t needed = 1 + ((count - 1) / kThreadsPerBlock);
    return static_cast<unsigned int>(std::min(needed, kMaxBlocks));
}

[[nodiscard]] bool valid_epsilon(const float epsilon) noexcept {
    return std::isfinite(epsilon) && epsilon > 0.0F;
}

__device__ __forceinline__ float quiet_nan() noexcept {
    return __int_as_float(0x7fc00000);
}

__device__ __forceinline__ bool finite_complex(const float2 value) noexcept {
    return isfinite(value.x) && isfinite(value.y);
}

// Preserve the original fma/sqrt fast path for ordinary SAR values while
// avoiding intermediate square overflow/underflow for extreme finite inputs.
__device__ __forceinline__ float robust_magnitude(
    const float2 sample) noexcept {
    const float power = fmaf(sample.x, sample.x, sample.y * sample.y);
    if (isfinite(power) && power > 0.0F) {
        return sqrtf(power);
    }
    if (sample.x == 0.0F && sample.y == 0.0F) {
        return 0.0F;
    }
    return hypotf(sample.x, sample.y);
}

template <bool HasInputValidity>
__device__ __forceinline__ bool input_is_valid(
    const std::uint8_t* const validity,
    const std::size_t index) noexcept {
    if constexpr (HasInputValidity) {
        const std::uint8_t value = validity[index];
        return value != 0 && value != 255;
    }
    return true;
}

template <bool HasOutputValidity>
__device__ __forceinline__ void store_result(
    float* const output,
    std::uint8_t* const validity,
    const std::size_t index,
    const float value,
    const bool valid) noexcept {
    output[index] = valid ? value : quiet_nan();
    if constexpr (HasOutputValidity) {
        validity[index] = static_cast<std::uint8_t>(valid);
    }
}

template <GslcTransform Transform>
__device__ __forceinline__ float gslc_value(
    const float2 sample,
    const float epsilon) noexcept {
    if constexpr (Transform == GslcTransform::real) {
        return sample.x;
    } else if constexpr (Transform == GslcTransform::imaginary) {
        return sample.y;
    } else if constexpr (Transform == GslcTransform::phase) {
        return atan2f(sample.y, sample.x);
    } else {
        const float power = fmaf(sample.x, sample.x, sample.y * sample.y);
        if constexpr (Transform == GslcTransform::amplitude) {
            return robust_magnitude(sample);
        } else if constexpr (Transform == GslcTransform::power) {
            return power;
        } else {
            if (!isfinite(power)) {
                const float high =
                    fmaxf(fabsf(sample.x), fabsf(sample.y));
                const float low =
                    fminf(fabsf(sample.x), fabsf(sample.y));
                const float ratio = low / high;
                const float stable_db =
                    20.0F * log10f(high) +
                    10.0F * log10f(fmaf(ratio, ratio, 1.0F));
                return fmaxf(stable_db, 10.0F * log10f(epsilon));
            }
            return 10.0F * log10f(fmaxf(power, epsilon));
        }
    }
}

template <
    GslcTransform Transform,
    bool HasInputValidity,
    bool HasOutputValidity>
__global__ void gslc_transform_kernel(
    const float2* __restrict__ input,
    float* __restrict__ output,
    const std::size_t count,
    const std::uint8_t* input_validity,
    std::uint8_t* output_validity,
    const float epsilon) {
    const std::size_t first =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t index = first; index < count; index += stride) {
        bool valid =
            input_is_valid<HasInputValidity>(input_validity, index);
        float value = quiet_nan();

        if (valid) {
            const float2 sample = input[index];
            valid = finite_complex(sample);
            if (valid) {
                value = gslc_value<Transform>(sample, epsilon);
                valid = isfinite(value);
            }
        }

        store_result<HasOutputValidity>(
            output, output_validity, index, value, valid);
    }
}

template <GcovRealTransform Transform>
__device__ __forceinline__ float gcov_real_value(
    const float sample,
    const float epsilon) noexcept {
    if constexpr (Transform == GcovRealTransform::linear) {
        return sample;
    } else {
        return 10.0F * log10f(fmaxf(sample, epsilon));
    }
}

template <
    GcovRealTransform Transform,
    bool HasInputValidity,
    bool HasOutputValidity>
__global__ void gcov_real_transform_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const std::size_t count,
    const std::uint8_t* input_validity,
    std::uint8_t* output_validity,
    const float epsilon) {
    const std::size_t first =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t index = first; index < count; index += stride) {
        bool valid =
            input_is_valid<HasInputValidity>(input_validity, index);
        float value = quiet_nan();

        if (valid) {
            const float sample = input[index];
            valid = isfinite(sample) && sample >= 0.0F;
            if (valid) {
                value = gcov_real_value<Transform>(sample, epsilon);
                valid = isfinite(value);
            }
        }

        store_result<HasOutputValidity>(
            output, output_validity, index, value, valid);
    }
}

template <GcovComplexTransform Transform>
__device__ __forceinline__ float gcov_complex_value(
    const float2 sample) noexcept {
    if constexpr (Transform == GcovComplexTransform::magnitude) {
        return robust_magnitude(sample);
    } else {
        return atan2f(sample.y, sample.x);
    }
}

template <
    GcovComplexTransform Transform,
    bool HasInputValidity,
    bool HasOutputValidity>
__global__ void gcov_complex_transform_kernel(
    const float2* __restrict__ input,
    float* __restrict__ output,
    const std::size_t count,
    const std::uint8_t* input_validity,
    std::uint8_t* output_validity) {
    const std::size_t first =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t index = first; index < count; index += stride) {
        bool valid =
            input_is_valid<HasInputValidity>(input_validity, index);
        float value = quiet_nan();

        if (valid) {
            const float2 sample = input[index];
            valid = finite_complex(sample);
            if (valid) {
                value = gcov_complex_value<Transform>(sample);
                valid = isfinite(value);
            }
        }

        store_result<HasOutputValidity>(
            output, output_validity, index, value, valid);
    }
}

template <bool HasInputValidity, bool HasOutputValidity>
__global__ void gcov_normalized_correlation_kernel(
    const float2* __restrict__ cij,
    const float* __restrict__ cii,
    const float* __restrict__ cjj,
    float* __restrict__ output,
    const std::size_t count,
    const std::uint8_t* input_validity,
    std::uint8_t* output_validity,
    const float epsilon) {
    const std::size_t first =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t index = first; index < count; index += stride) {
        bool valid =
            input_is_valid<HasInputValidity>(input_validity, index);
        float value = quiet_nan();

        if (valid) {
            const float2 cross = cij[index];
            const float diagonal_i = cii[index];
            const float diagonal_j = cjj[index];
            valid = finite_complex(cross) && isfinite(diagonal_i) &&
                    isfinite(diagonal_j) && diagonal_i >= 0.0F &&
                    diagonal_j >= 0.0F;

            if (valid) {
                const float cross_power =
                    fmaf(cross.x, cross.x, cross.y * cross.y);
                const float diagonal_product = diagonal_i * diagonal_j;
                if (isfinite(cross_power) &&
                    isfinite(diagonal_product)) {
                    const float denominator =
                        sqrtf(fmaxf(diagonal_product, epsilon));
                    value = sqrtf(cross_power) / denominator;
                } else {
                    const double cross_power_wide =
                        static_cast<double>(cross.x) * cross.x +
                        static_cast<double>(cross.y) * cross.y;
                    const double diagonal_product_wide =
                        static_cast<double>(diagonal_i) * diagonal_j;
                    value = static_cast<float>(sqrt(
                        cross_power_wide /
                        fmax(
                            diagonal_product_wide,
                            static_cast<double>(epsilon))));
                }
                valid = isfinite(value);
            }
        }

        store_result<HasOutputValidity>(
            output, output_validity, index, value, valid);
    }
}

template <GslcTransform Transform>
[[nodiscard]] cudaError_t launch_gslc_typed(
    const float2* const input,
    float* const output,
    const std::size_t count,
    const TransformOptions options) noexcept {
    const unsigned int blocks = block_count(count);
    const bool has_input_validity = options.validity.input != nullptr;
    const bool has_output_validity = options.validity.output != nullptr;

    if (has_input_validity && has_output_validity) {
        gslc_transform_kernel<Transform, true, true>
            <<<blocks, kThreadsPerBlock, 0, options.stream>>>(
                input,
                output,
                count,
                options.validity.input,
                options.validity.output,
                options.epsilon);
    } else if (has_input_validity) {
        gslc_transform_kernel<Transform, true, false>
            <<<blocks, kThreadsPerBlock, 0, options.stream>>>(
                input,
                output,
                count,
                options.validity.input,
                nullptr,
                options.epsilon);
    } else if (has_output_validity) {
        gslc_transform_kernel<Transform, false, true>
            <<<blocks, kThreadsPerBlock, 0, options.stream>>>(
                input,
                output,
                count,
                nullptr,
                options.validity.output,
                options.epsilon);
    } else {
        gslc_transform_kernel<Transform, false, false>
            <<<blocks, kThreadsPerBlock, 0, options.stream>>>(
                input, output, count, nullptr, nullptr, options.epsilon);
    }

    return cudaPeekAtLastError();
}

template <GcovRealTransform Transform>
[[nodiscard]] cudaError_t launch_gcov_real_typed(
    const float* const input,
    float* const output,
    const std::size_t count,
    const TransformOptions options) noexcept {
    const unsigned int blocks = block_count(count);
    const bool has_input_validity = options.validity.input != nullptr;
    const bool has_output_validity = options.validity.output != nullptr;

    if (has_input_validity && has_output_validity) {
        gcov_real_transform_kernel<Transform, true, true>
            <<<blocks, kThreadsPerBlock, 0, options.stream>>>(
                input,
                output,
                count,
                options.validity.input,
                options.validity.output,
                options.epsilon);
    } else if (has_input_validity) {
        gcov_real_transform_kernel<Transform, true, false>
            <<<blocks, kThreadsPerBlock, 0, options.stream>>>(
                input,
                output,
                count,
                options.validity.input,
                nullptr,
                options.epsilon);
    } else if (has_output_validity) {
        gcov_real_transform_kernel<Transform, false, true>
            <<<blocks, kThreadsPerBlock, 0, options.stream>>>(
                input,
                output,
                count,
                nullptr,
                options.validity.output,
                options.epsilon);
    } else {
        gcov_real_transform_kernel<Transform, false, false>
            <<<blocks, kThreadsPerBlock, 0, options.stream>>>(
                input, output, count, nullptr, nullptr, options.epsilon);
    }

    return cudaPeekAtLastError();
}

template <GcovComplexTransform Transform>
[[nodiscard]] cudaError_t launch_gcov_complex_typed(
    const float2* const input,
    float* const output,
    const std::size_t count,
    const TransformOptions options) noexcept {
    const unsigned int blocks = block_count(count);
    const bool has_input_validity = options.validity.input != nullptr;
    const bool has_output_validity = options.validity.output != nullptr;

    if (has_input_validity && has_output_validity) {
        gcov_complex_transform_kernel<Transform, true, true>
            <<<blocks, kThreadsPerBlock, 0, options.stream>>>(
                input,
                output,
                count,
                options.validity.input,
                options.validity.output);
    } else if (has_input_validity) {
        gcov_complex_transform_kernel<Transform, true, false>
            <<<blocks, kThreadsPerBlock, 0, options.stream>>>(
                input,
                output,
                count,
                options.validity.input,
                nullptr);
    } else if (has_output_validity) {
        gcov_complex_transform_kernel<Transform, false, true>
            <<<blocks, kThreadsPerBlock, 0, options.stream>>>(
                input,
                output,
                count,
                nullptr,
                options.validity.output);
    } else {
        gcov_complex_transform_kernel<Transform, false, false>
            <<<blocks, kThreadsPerBlock, 0, options.stream>>>(
                input, output, count, nullptr, nullptr);
    }

    return cudaPeekAtLastError();
}

[[nodiscard]] cudaError_t launch_correlation_typed(
    const float2* const cij,
    const float* const cii,
    const float* const cjj,
    float* const output,
    const std::size_t count,
    const TransformOptions options) noexcept {
    const unsigned int blocks = block_count(count);
    const bool has_input_validity = options.validity.input != nullptr;
    const bool has_output_validity = options.validity.output != nullptr;

    if (has_input_validity && has_output_validity) {
        gcov_normalized_correlation_kernel<true, true>
            <<<blocks, kThreadsPerBlock, 0, options.stream>>>(
                cij,
                cii,
                cjj,
                output,
                count,
                options.validity.input,
                options.validity.output,
                options.epsilon);
    } else if (has_input_validity) {
        gcov_normalized_correlation_kernel<true, false>
            <<<blocks, kThreadsPerBlock, 0, options.stream>>>(
                cij,
                cii,
                cjj,
                output,
                count,
                options.validity.input,
                nullptr,
                options.epsilon);
    } else if (has_output_validity) {
        gcov_normalized_correlation_kernel<false, true>
            <<<blocks, kThreadsPerBlock, 0, options.stream>>>(
                cij,
                cii,
                cjj,
                output,
                count,
                nullptr,
                options.validity.output,
                options.epsilon);
    } else {
        gcov_normalized_correlation_kernel<false, false>
            <<<blocks, kThreadsPerBlock, 0, options.stream>>>(
                cij,
                cii,
                cjj,
                output,
                count,
                nullptr,
                nullptr,
                options.epsilon);
    }

    return cudaPeekAtLastError();
}

}  // namespace

cudaError_t launch_gslc_transform(
    const float2* const input,
    float* const output,
    const std::size_t count,
    const GslcTransform transform,
    const TransformOptions options) noexcept {
    if (count == 0) {
        return cudaSuccess;
    }
    if (input == nullptr || output == nullptr) {
        return cudaErrorInvalidValue;
    }

    switch (transform) {
        case GslcTransform::amplitude:
            return launch_gslc_typed<GslcTransform::amplitude>(
                input, output, count, options);
        case GslcTransform::power:
            return launch_gslc_typed<GslcTransform::power>(
                input, output, count, options);
        case GslcTransform::power_db:
            if (!valid_epsilon(options.epsilon)) {
                return cudaErrorInvalidValue;
            }
            return launch_gslc_typed<GslcTransform::power_db>(
                input, output, count, options);
        case GslcTransform::phase:
            return launch_gslc_typed<GslcTransform::phase>(
                input, output, count, options);
        case GslcTransform::real:
            return launch_gslc_typed<GslcTransform::real>(
                input, output, count, options);
        case GslcTransform::imaginary:
            return launch_gslc_typed<GslcTransform::imaginary>(
                input, output, count, options);
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t launch_gcov_real_transform(
    const float* const input,
    float* const output,
    const std::size_t count,
    const GcovRealTransform transform,
    const TransformOptions options) noexcept {
    if (count == 0) {
        return cudaSuccess;
    }
    if (input == nullptr || output == nullptr) {
        return cudaErrorInvalidValue;
    }

    switch (transform) {
        case GcovRealTransform::linear:
            return launch_gcov_real_typed<GcovRealTransform::linear>(
                input, output, count, options);
        case GcovRealTransform::power_db:
            if (!valid_epsilon(options.epsilon)) {
                return cudaErrorInvalidValue;
            }
            return launch_gcov_real_typed<GcovRealTransform::power_db>(
                input, output, count, options);
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t launch_gcov_complex_transform(
    const float2* const input,
    float* const output,
    const std::size_t count,
    const GcovComplexTransform transform,
    const TransformOptions options) noexcept {
    if (count == 0) {
        return cudaSuccess;
    }
    if (input == nullptr || output == nullptr) {
        return cudaErrorInvalidValue;
    }

    switch (transform) {
        case GcovComplexTransform::magnitude:
            return launch_gcov_complex_typed<GcovComplexTransform::magnitude>(
                input, output, count, options);
        case GcovComplexTransform::phase:
            return launch_gcov_complex_typed<GcovComplexTransform::phase>(
                input, output, count, options);
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t launch_gcov_normalized_correlation(
    const float2* const cij,
    const float* const cii,
    const float* const cjj,
    float* const output,
    const std::size_t count,
    const TransformOptions options) noexcept {
    if (count == 0) {
        return cudaSuccess;
    }
    if (cij == nullptr || cii == nullptr || cjj == nullptr ||
        output == nullptr || !valid_epsilon(options.epsilon)) {
        return cudaErrorInvalidValue;
    }

    return launch_correlation_typed(cij, cii, cjj, output, count, options);
}

}  // namespace satview::gpu
