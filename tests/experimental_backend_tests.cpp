#include "satview/cpu/scientific.hpp"
#include "satview/experimental/scientific.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

void expect(
    const bool condition,
    const std::string_view description,
    int& failures) {
  if (!condition) {
    std::cerr << "Experimental backend test failed: " << description << '\n';
    ++failures;
  }
}

[[nodiscard]] bool equivalent(
    const std::span<const float> actual,
    const std::span<const float> expected) {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (std::isnan(expected[index])) {
      if (!std::isnan(actual[index])) {
        return false;
      }
      continue;
    }
    const float scale = std::max(1.0F, std::abs(expected[index]));
    if (!std::isfinite(actual[index]) ||
        std::abs(actual[index] - expected[index]) > 2.0e-4F * scale) {
      return false;
    }
  }
  return true;
}

template <typename Process>
void test_backend(
    const std::string_view name, Process process, int& failures) {
  constexpr std::size_t width = 5;
  constexpr std::size_t height = 5;
  constexpr std::size_t count = width * height;
  std::array<satview::cpu::Complex32, count> science{};
  std::array<std::uint8_t, count> validity{};
  for (std::size_t index = 0; index < count; ++index) {
    science[index] = {
        0.25F + static_cast<float>(index) * 0.125F,
        -0.5F + static_cast<float>(index % width) * 0.2F};
    validity[index] = index == 7 ? std::uint8_t{255} : std::uint8_t{1};
  }
  std::array<float, count> transformed{};
  std::array<float, count> expected{};
  std::array<float, count> actual{};
  satview::cpu::transform_complex(
      science,
      transformed,
      satview::cpu::ComplexTransform::power_db,
      validity);
  satview::cpu::SpeckleOptions speckle;
  speckle.filter = satview::cpu::SpeckleFilter::lee;
  speckle.domain = satview::cpu::SpeckleDomain::power_db;
  speckle.window_size = 5;
  speckle.equivalent_number_of_looks = 2.0F;
  satview::cpu::filter_speckle(
      transformed, expected, width, height, speckle, validity);

  satview::experimental::PageRequest request;
  request.science = std::as_bytes(std::span(science));
  request.validity = validity;
  request.width = width;
  request.height = height;
  request.input_kind =
      satview::experimental::InputKind::complex_float32;
  request.complex_transform =
      satview::cpu::ComplexTransform::power_db;
  request.filter_enabled = true;
  request.speckle = speckle;
  process(request, actual);
  expect(
      equivalent(actual, expected),
      std::string(name) + " transform/filter parity",
      failures);

  constexpr std::array complex_transforms{
      satview::cpu::ComplexTransform::amplitude,
      satview::cpu::ComplexTransform::power,
      satview::cpu::ComplexTransform::power_db,
      satview::cpu::ComplexTransform::phase,
      satview::cpu::ComplexTransform::real,
      satview::cpu::ComplexTransform::imaginary,
  };
  request.filter_enabled = false;
  for (const auto transform : complex_transforms) {
    satview::cpu::transform_complex(
        science, expected, transform, validity);
    request.complex_transform = transform;
    process(request, actual);
    expect(
        equivalent(actual, expected),
        std::string(name) + " complex transform parity",
        failures);
  }

  std::array<float, count> real_science{};
  for (std::size_t index = 0; index < count; ++index) {
    real_science[index] = 0.25F + static_cast<float>(index);
  }
  real_science[3] = -1.0F;
  real_science[11] = std::numeric_limits<float>::quiet_NaN();
  request.science = std::as_bytes(std::span(real_science));
  request.input_kind = satview::experimental::InputKind::real_float32;
  constexpr std::array real_transforms{
      satview::cpu::RealTransform::linear,
      satview::cpu::RealTransform::power_db,
  };
  for (const auto transform : real_transforms) {
    satview::cpu::transform_real(
        real_science, expected, transform, validity);
    request.real_transform = transform;
    process(request, actual);
    expect(
        equivalent(actual, expected),
        std::string(name) + " real transform parity",
        failures);
  }

  request.science = std::as_bytes(std::span(science));
  request.input_kind = satview::experimental::InputKind::complex_float32;
  request.complex_transform = satview::cpu::ComplexTransform::power;
  request.filter_enabled = true;
  request.speckle.filter = satview::cpu::SpeckleFilter::boxcar;
  request.speckle.domain = satview::cpu::SpeckleDomain::linear_power;
  satview::cpu::transform_complex(
      science,
      transformed,
      satview::cpu::ComplexTransform::power,
      validity);
  satview::cpu::filter_speckle(
      transformed,
      expected,
      width,
      height,
      request.speckle,
      validity);
  process(request, actual);
  expect(
      equivalent(actual, expected),
      std::string(name) + " Boxcar parity",
      failures);
}

}  // namespace

int main() {
  int failures = 0;
#if defined(SATVIEW_HAS_EXPERIMENTAL_HIP)
  {
    std::string reason;
    if (satview::experimental::hip_runtime_available(reason)) {
      test_backend(
          "HIP/ROCm", satview::experimental::process_hip, failures);
    } else {
      std::cout << "HIP/ROCm experimental tests skipped: " << reason << '\n';
    }
  }
#endif
#if defined(SATVIEW_HAS_EXPERIMENTAL_SYCL)
  {
    std::string reason;
    if (satview::experimental::sycl_runtime_available(reason)) {
      test_backend(
          "oneAPI/SYCL", satview::experimental::process_sycl, failures);
    } else {
      std::cout << "oneAPI/SYCL experimental tests skipped: " << reason << '\n';
    }
  }
#endif
  return failures == 0 ? 0 : 1;
}
