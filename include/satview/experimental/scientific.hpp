#pragma once

#include "satview/cpu/scientific.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace satview::experimental {

enum class InputKind : std::uint8_t {
  complex_float32,
  real_float32,
};

// Host-facing contract for the experimental accelerators. Implementations
// perform the same transform and optional speckle pass as the production CUDA
// path, then copy the final R32F page back for Vulkan staging.
struct PageRequest {
  std::span<const std::byte> science;
  std::span<const std::uint8_t> validity;
  std::size_t width = 0;
  std::size_t height = 0;
  InputKind input_kind = InputKind::real_float32;
  cpu::ComplexTransform complex_transform = cpu::ComplexTransform::amplitude;
  cpu::RealTransform real_transform = cpu::RealTransform::linear;
  bool filter_enabled = false;
  cpu::SpeckleOptions speckle{
      .filter = cpu::SpeckleFilter::none,
  };
};

[[nodiscard]] inline bool valid_filter_configuration(
    const PageRequest& request) noexcept {
  const bool has_filter =
      request.speckle.filter != cpu::SpeckleFilter::none;
  if (request.filter_enabled != has_filter) {
    return false;
  }
  switch (request.speckle.domain) {
    case cpu::SpeckleDomain::amplitude:
    case cpu::SpeckleDomain::linear_power:
    case cpu::SpeckleDomain::power_db:
      break;
    default:
      return false;
  }
  if (!has_filter) {
    return true;
  }
  if (request.speckle.filter != cpu::SpeckleFilter::boxcar &&
      request.speckle.filter != cpu::SpeckleFilter::lee) {
    return false;
  }
  if (request.speckle.window_size != 3 &&
      request.speckle.window_size != 5 &&
      request.speckle.window_size != 7) {
    return false;
  }
  if (!std::isfinite(request.speckle.power_epsilon) ||
      request.speckle.power_epsilon <= 0.0F) {
    return false;
  }
  return request.speckle.filter != cpu::SpeckleFilter::lee ||
      (std::isfinite(request.speckle.equivalent_number_of_looks) &&
       request.speckle.equivalent_number_of_looks > 0.0F);
}

#if defined(SATVIEW_HAS_EXPERIMENTAL_HIP)
[[nodiscard]] bool hip_runtime_available(std::string& reason) noexcept;
void process_hip(const PageRequest& request, std::span<float> output);
#endif

#if defined(SATVIEW_HAS_EXPERIMENTAL_SYCL)
[[nodiscard]] bool sycl_runtime_available(std::string& reason) noexcept;
void process_sycl(const PageRequest& request, std::span<float> output);
#endif

}  // namespace satview::experimental
