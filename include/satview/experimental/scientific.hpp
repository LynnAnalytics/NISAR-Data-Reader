#pragma once

#include "satview/cpu/scientific.hpp"

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
  cpu::SpeckleOptions speckle;
};

#if defined(SATVIEW_HAS_EXPERIMENTAL_HIP)
[[nodiscard]] bool hip_runtime_available(std::string& reason) noexcept;
void process_hip(const PageRequest& request, std::span<float> output);
#endif

#if defined(SATVIEW_HAS_EXPERIMENTAL_SYCL)
[[nodiscard]] bool sycl_runtime_available(std::string& reason) noexcept;
void process_sycl(const PageRequest& request, std::span<float> output);
#endif

}  // namespace satview::experimental
