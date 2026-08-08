#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>
#include <vector_types.h>

namespace satview::gpu {

// Small positive floor used by logarithmic and normalized-correlation views.
inline constexpr float kDefaultTransformEpsilon = 1.0e-20F;

// Optional device-resident validity masks.
//
// Each input mask uses one byte per pixel with NISAR-compatible semantics:
// values 1..254 are valid, while 0 and 255 are invalid. A canonical 0/1 mask
// therefore works without conversion. Output masks are canonicalized to 0 or 1.
// Invalid or non-finite pixels produce a quiet NaN. Both pointers may be null.
// inputDataExceptionMask has different semantics and must be normalized first.
//
// The input and output masks may alias for in-place validity refinement.
struct TransformValidity {
    const std::uint8_t* input = nullptr;
    std::uint8_t* output = nullptr;
};

// All pointers in this structure and in the launch APIs are device pointers.
// Launches are asynchronous with respect to the host and are ordered on stream.
// epsilon must be finite and greater than zero for transforms that use it.
struct TransformOptions {
    TransformValidity validity{};
    float epsilon = kDefaultTransformEpsilon;
    cudaStream_t stream = nullptr;
};

enum class GslcTransform : std::uint8_t {
    // sqrt(real^2 + imag^2)
    amplitude,
    // real^2 + imag^2
    power,
    // 10 * log10(max(real^2 + imag^2, epsilon))
    power_db,
    // atan2(imag, real), in radians
    phase,
    real,
    imaginary,
};

enum class GcovRealTransform : std::uint8_t {
    // Preserve a finite, non-negative diagonal covariance value.
    linear,
    // 10 * log10(max(value, epsilon))
    power_db,
};

enum class GcovComplexTransform : std::uint8_t {
    // sqrt(real^2 + imag^2)
    magnitude,
    // atan2(imag, real), in radians
    phase,
};

// Transforms count complex GSLC samples into an R32F-compatible scalar buffer.
// input is float2-compatible (x = real, y = imaginary).
[[nodiscard]] cudaError_t launch_gslc_transform(
    const float2* input,
    float* output,
    std::size_t count,
    GslcTransform transform,
    TransformOptions options = {}) noexcept;

// Transforms count real GCOV diagonal terms into an R32F-compatible buffer.
[[nodiscard]] cudaError_t launch_gcov_real_transform(
    const float* input,
    float* output,
    std::size_t count,
    GcovRealTransform transform,
    TransformOptions options = {}) noexcept;

// Transforms count complex GCOV off-diagonal terms into an R32F-compatible
// scalar buffer. input is float2-compatible (x = real, y = imaginary).
[[nodiscard]] cudaError_t launch_gcov_complex_transform(
    const float2* input,
    float* output,
    std::size_t count,
    GcovComplexTransform transform,
    TransformOptions options = {}) noexcept;

// Computes the scalar magnitude of normalized complex correlation:
//
//     |rho_ij| = |C_ij| / sqrt(max(C_ii * C_jj, epsilon))
//
// C_ij is complex float2-compatible input; C_ii and C_jj are real diagonal
// covariance terms. Negative diagonal terms are invalid. The result is not
// clamped to one so that non-physical source data remains observable.
[[nodiscard]] cudaError_t launch_gcov_normalized_correlation(
    const float2* cij,
    const float* cii,
    const float* cjj,
    float* output,
    std::size_t count,
    TransformOptions options = {}) noexcept;

}  // namespace satview::gpu
