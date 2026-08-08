#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace satview::gpu {

// Speckle filters operate on non-negative SAR power. Amplitude and decibel
// inputs are converted to power before neighborhood statistics are evaluated,
// then converted back to their original display domain. This avoids the
// scientifically incorrect operation of averaging logarithmic dB samples.
enum class SpeckleDomain : std::uint8_t {
    amplitude,
    linear_power,
    power_db,

    // These values let a caller describe the active viewer layer explicitly.
    // They are accepted only by SpeckleFilter::none; neighborhood speckle
    // filtering of phase or signed scalar components is intentionally rejected.
    signed_value,
    phase,
};

enum class SpeckleFilter : std::uint8_t {
    // Copies finite, center-valid samples unchanged while canonicalizing an
    // optional output validity mask. No domain conversion is performed.
    none,

    // Arithmetic mean of valid neighborhood samples in the power domain.
    boxcar,

    // Local-statistics Lee filter in the power domain. The noise coefficient
    // of variation squared is 1 / equivalent_number_of_looks. For local power
    // mean m and population variance v, the deterministic adaptive weight is
    // clamp((v - m*m/ENL) / v, 0, 1), with zero weight when v is zero.
    lee,
};

// NISAR-compatible byte validity: values 1..254 are valid; 0 and 255 are
// invalid. Output validity, when requested, is canonicalized to 0 or 1.
struct SpeckleValidity {
    const std::uint8_t* input = nullptr;
    std::uint8_t* output = nullptr;

    [[nodiscard]] friend constexpr bool operator==(
        const SpeckleValidity&,
        const SpeckleValidity&) noexcept = default;
};

inline constexpr float kDefaultSpecklePowerEpsilon = 1.0e-20F;

struct SpeckleFilterOptions {
    SpeckleFilter filter = SpeckleFilter::none;
    SpeckleDomain domain = SpeckleDomain::linear_power;

    // Boxcar and Lee support practical odd square windows of 3, 5, or 7.
    // Image-edge windows are clipped, not padded.
    std::uint32_t window_size = 5;

    // Lee only. Must be finite and greater than zero. Larger values preserve
    // more local variation because they describe lower speckle variance.
    float equivalent_number_of_looks = 1.0F;

    // power_db only. The converted output is floored to this positive linear
    // power before log10. Must be finite and greater than zero.
    float power_epsilon = kDefaultSpecklePowerEpsilon;

    SpeckleValidity validity{};
    cudaStream_t stream = nullptr;

    // Exact float equality is intentional here: callers can include the
    // options directly in a resident-tile/render identity.
    [[nodiscard]] friend constexpr bool operator==(
        const SpeckleFilterOptions&,
        const SpeckleFilterOptions&) noexcept = default;
};

// All pointers are device pointers. The launch is asynchronous with respect
// to the host and is ordered on options.stream. On success, CUDA execution
// failures remain observable by synchronizing that stream.
//
// input and output must be distinct allocations. If both validity pointers are
// non-null, they must also be distinct. Invalid/non-finite center pixels remain
// invalid and produce a quiet NaN; invalid neighbors are excluded. A valid
// center therefore always contributes at least one sample to its window.
//
// Neighborhood powers are normalized by their local finite maximum before
// statistics are accumulated. Consequently, finite amplitude and dB inputs
// remain filterable even when their absolute linear power would overflow or
// underflow float; no finite source value is silently discarded for that reason.
//
// window_size is measured in the supplied sample grid. On a sparse regional
// LOD page with source stride S, an N x N filter spans N samples separated by S
// native pixels; it is not equivalent to a dense N x N native-pixel filter.
// Callers should expose that effective footprint rather than implying native
// window semantics. Native pages (stride 1) have exact window semantics.
//
// If width or height is zero, this is a no-op and pointers are not inspected.
// Invalid dimensions, unsupported domains, bad options, and immediate CUDA
// launch failures are returned as cudaError_t values without throwing.
[[nodiscard]] cudaError_t launch_speckle_filter(
    const float* input,
    float* output,
    std::size_t width,
    std::size_t height,
    SpeckleFilterOptions options = {}) noexcept;

}  // namespace satview::gpu
