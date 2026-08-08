#include "satview/cpu/scientific.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>

namespace {

void expect(
    const bool condition,
    const std::string_view description,
    int& failures) {
  if (!condition) {
    std::cerr << "CPU scientific test failed: " << description << '\n';
    ++failures;
  }
}

[[nodiscard]] bool near(const float left, const float right) {
  const float scale = std::max({1.0F, std::abs(left), std::abs(right)});
  return std::abs(left - right) <= 4.0e-5F * scale;
}

void test_complex_transforms(int& failures) {
  using namespace satview::cpu;
  constexpr std::array input{
      Complex32{3.0F, 4.0F},
      Complex32{0.0F, 0.0F},
      Complex32{-2.0F, 2.0F},
      Complex32{1.0F, -1.0F},
  };
  constexpr std::array<std::uint8_t, 4> validity{1, 1, 255, 0};
  std::array<float, input.size()> output{};

  transform_complex(input, output, ComplexTransform::amplitude, validity);
  expect(near(output[0], 5.0F), "complex amplitude", failures);
  expect(output[1] == 0.0F, "zero complex amplitude", failures);
  expect(std::isnan(output[2]) && std::isnan(output[3]),
         "complex validity mask", failures);

  transform_complex(input, output, ComplexTransform::power, validity);
  expect(near(output[0], 25.0F), "complex power", failures);

  transform_complex(input, output, ComplexTransform::power_db, validity);
  expect(near(output[0], 10.0F * std::log10(25.0F)),
         "complex power dB", failures);
  expect(near(output[1], -200.0F), "complex dB epsilon floor", failures);

  transform_complex(input, output, ComplexTransform::phase, validity);
  expect(near(output[0], std::atan2(4.0F, 3.0F)),
         "complex phase", failures);

  transform_complex(input, output, ComplexTransform::real, validity);
  expect(output[0] == 3.0F, "complex real component", failures);
  transform_complex(input, output, ComplexTransform::imaginary, validity);
  expect(output[0] == 4.0F, "complex imaginary component", failures);
}

void test_real_transform_and_histogram(int& failures) {
  using namespace satview::cpu;
  constexpr std::array input{1.0F, 10.0F, 100.0F, -1.0F, 0.0F};
  std::array<float, input.size()> output{};
  transform_real(input, output, RealTransform::power_db);
  expect(output[0] == 0.0F && near(output[1], 10.0F) &&
             near(output[2], 20.0F),
         "real power dB", failures);
  expect(std::isnan(output[3]) && near(output[4], -200.0F),
         "real invalid and epsilon floor", failures);

  const auto histogram = build_histogram(output);
  expect(histogram.finite_count == 4 && histogram.invalid_count == 1,
         "histogram finite and invalid counts", failures);
  expect(near(histogram.minimum, -200.0F) &&
             near(histogram.maximum, 20.0F),
         "histogram extrema", failures);
  std::uint64_t binned = 0;
  for (const auto count : histogram.bins) {
    binned += count;
  }
  expect(binned == histogram.finite_count,
         "histogram bins retain every finite value", failures);
}

void test_speckle(int& failures) {
  using namespace satview::cpu;
  constexpr std::array<float, 9> impulse{
      1.0F, 1.0F, 1.0F,
      1.0F, 9.0F, 1.0F,
      1.0F, 1.0F, 1.0F,
  };
  std::array<float, impulse.size()> output{};
  SpeckleOptions options;
  options.filter = SpeckleFilter::none;
  filter_speckle(impulse, output, 3, 3, options);
  expect(output == impulse, "no filter is a passthrough", failures);

  options.filter = SpeckleFilter::boxcar;
  options.domain = SpeckleDomain::linear_power;
  options.window_size = 3;
  filter_speckle(impulse, output, 3, 3, options);
  expect(near(output[4], 17.0F / 9.0F),
         "boxcar center is the local power mean", failures);

  constexpr std::array<float, 9> uniform{
      7.0F, 7.0F, 7.0F,
      7.0F, 7.0F, 7.0F,
      7.0F, 7.0F, 7.0F,
  };
  options.filter = SpeckleFilter::lee;
  options.equivalent_number_of_looks = 2.0F;
  filter_speckle(uniform, output, 3, 3, options);
  bool preserved = true;
  for (const float value : output) {
    preserved = preserved && near(value, 7.0F);
  }
  expect(preserved, "Lee filter preserves a uniform field", failures);

  constexpr std::array<std::uint8_t, 9> validity{
      1, 1, 1,
      1, 255, 1,
      1, 1, 1,
  };
  filter_speckle(uniform, output, 3, 3, options, validity);
  expect(std::isnan(output[4]), "speckle mask invalidates its center", failures);
}

}  // namespace

int run_cpu_scientific_tests() {
  int failures = 0;
  test_complex_transforms(failures);
  test_real_transform_and_histogram(failures);
  test_speckle(failures);
  return failures;
}
