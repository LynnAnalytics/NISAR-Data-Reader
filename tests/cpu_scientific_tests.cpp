#include "satview/cpu/scientific.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

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

template <typename Function>
void expect_invalid_argument(Function &&function,
                             const std::string_view description,
                             int &failures) {
  try {
    function();
    expect(false, description, failures);
  } catch (const std::invalid_argument &) {
  } catch (...) {
    expect(false, description, failures);
  }
}

[[nodiscard]] bool exact_float(const float left, const float right) noexcept {
  return std::bit_cast<std::uint32_t>(left) ==
         std::bit_cast<std::uint32_t>(right);
}

[[nodiscard]] satview::cpu::Histogram
reference_build_histogram(const std::span<const float> values) {
  using namespace satview::cpu;
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

[[nodiscard]] bool exact_histogram(const satview::cpu::Histogram &left,
                                   const satview::cpu::Histogram &right) {
  return left.finite_count == right.finite_count &&
         left.invalid_count == right.invalid_count &&
         exact_float(left.minimum, right.minimum) &&
         exact_float(left.maximum, right.maximum) && left.bins == right.bins;
}

[[nodiscard]] bool
reference_mask_valid(const std::span<const std::uint8_t> validity,
                     const std::size_t index) noexcept {
  if (validity.empty()) {
    return true;
  }
  return validity[index] != 0 && validity[index] != 255;
}

[[nodiscard]] bool
reference_source_valid(const float sample,
                       const satview::cpu::SpeckleDomain domain) noexcept {
  return std::isfinite(sample) &&
         (domain == satview::cpu::SpeckleDomain::power_db || sample >= 0.0F);
}

[[nodiscard]] float
reference_normalized_power(const float sample, const float scale,
                           const satview::cpu::SpeckleDomain domain) noexcept {
  using satview::cpu::SpeckleDomain;
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

[[nodiscard]] float
reference_from_normalized_power(const float normalized, const float scale,
                                const satview::cpu::SpeckleDomain domain,
                                const float epsilon) noexcept {
  using satview::cpu::SpeckleDomain;
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

void reference_filter_speckle(const std::span<const float> input,
                              const std::span<float> output,
                              const std::size_t width, const std::size_t height,
                              const satview::cpu::SpeckleOptions &options,
                              const std::span<const std::uint8_t> validity) {
  using satview::cpu::SpeckleFilter;
  const auto radius = static_cast<std::size_t>(options.window_size / 2);
  const float invalid = std::numeric_limits<float>::quiet_NaN();
  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t column = 0; column < width; ++column) {
      const auto center_index = row * width + column;
      const float center_sample = input[center_index];
      if (!reference_mask_valid(validity, center_index) ||
          !reference_source_valid(center_sample, options.domain)) {
        output[center_index] = invalid;
        continue;
      }

      const auto first_row = row > radius ? row - radius : 0;
      const auto last_row = std::min(height - 1, row + radius);
      const auto first_column = column > radius ? column - radius : 0;
      const auto last_column = std::min(width - 1, column + radius);
      float scale = center_sample;
      for (auto y = first_row; y <= last_row; ++y) {
        for (auto x = first_column; x <= last_column; ++x) {
          const auto index = y * width + x;
          const float sample = input[index];
          if (reference_mask_valid(validity, index) &&
              reference_source_valid(sample, options.domain)) {
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
          if (!reference_mask_valid(validity, index) ||
              !reference_source_valid(sample, options.domain)) {
            continue;
          }
          const float normalized =
              reference_normalized_power(sample, scale, options.domain);
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
        const float variance = std::max(m2 / static_cast<float>(count), 0.0F);
        const float noise = mean * mean / options.equivalent_number_of_looks;
        float weight = 0.0F;
        if (variance > noise && variance > 0.0F) {
          weight = (variance - noise) / variance;
        }
        weight = std::clamp(weight, 0.0F, 1.0F);
        const float center =
            reference_normalized_power(center_sample, scale, options.domain);
        filtered = std::fma(weight, center - mean, mean);
      }
      filtered = std::clamp(filtered, 0.0F, 1.0F);
      const float result = reference_from_normalized_power(
          filtered, scale, options.domain, options.power_epsilon);
      output[center_index] = std::isfinite(result) ? result : invalid;
    }
  }
}

[[nodiscard]] std::vector<float>
make_filter_input(const std::size_t count,
                  const satview::cpu::SpeckleDomain domain) {
  std::vector<float> input(count);
  for (std::size_t index = 0; index < count; ++index) {
    const float base =
        static_cast<float>((index * 37 + index / 11) % 4093) / 17.0F;
    input[index] =
        domain == satview::cpu::SpeckleDomain::power_db ? base - 120.0F : base;
  }
  if (count > 19) {
    input[1] = 0.0F;
    input[3] = -1.0F;
    input[5] = std::numeric_limits<float>::denorm_min();
    input[7] = std::numeric_limits<float>::max();
    input[11] = std::numeric_limits<float>::quiet_NaN();
    input[13] = std::numeric_limits<float>::infinity();
    input[17] = -std::numeric_limits<float>::infinity();
    input[count - 1] = std::numeric_limits<float>::max() / 2.0F;
  }
  return input;
}

[[nodiscard]] std::vector<std::uint8_t>
make_filter_validity(const std::size_t count) {
  std::vector<std::uint8_t> validity(count, 1);
  for (std::size_t index = 0; index < count; ++index) {
    if (index % 97 == 0) {
      validity[index] = 0;
    } else if (index % 89 == 0) {
      validity[index] = 255;
    }
  }
  return validity;
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

void test_parallel_histogram_matches_serial_reference(int &failures) {
  using namespace satview::cpu;
  constexpr std::array<std::size_t, 8> sizes{
      0, 1, 255, 65'537, 1'048'575, 1'048'576, 1'048'577, 4'194'304};
  for (const auto size : sizes) {
    std::vector<float> values(size);
    for (std::size_t index = 0; index < size; ++index) {
      const auto centered = static_cast<std::int32_t>(index % 8191) - 4095;
      values[index] = static_cast<float>(centered) / 17.0F;
      if (index % 1009 == 0) {
        values[index] = std::numeric_limits<float>::quiet_NaN();
      } else if (index % 1237 == 0) {
        values[index] = std::numeric_limits<float>::infinity();
      } else if (index % 1429 == 0) {
        values[index] = -std::numeric_limits<float>::infinity();
      }
    }
    if (size > 16) {
      values[1] = std::numeric_limits<float>::lowest();
      values[2] = std::numeric_limits<float>::max();
      values[3] = std::numeric_limits<float>::denorm_min();
      values[4] = -std::numeric_limits<float>::denorm_min();
      values[5] = 0.0F;
      values[6] = -0.0F;
      values[size / 2] = std::numeric_limits<float>::max();
      values[size - 1] = std::numeric_limits<float>::lowest();
    }

    const auto expected = reference_build_histogram(values);
    const auto actual = build_histogram(values);
    expect(exact_histogram(expected, actual),
           "parallel histogram is exactly equal to the serial oracle",
           failures);
  }

  constexpr std::size_t parallel_size = 1'048'576;
  for (const float first_zero : {0.0F, -0.0F}) {
    std::vector<float> values(parallel_size, -first_zero);
    values.front() = first_zero;
    const auto expected = reference_build_histogram(values);
    const auto actual = build_histogram(values);
    expect(exact_histogram(expected, actual),
           "parallel constant histogram preserves signed-zero extrema",
           failures);
  }

  std::vector<float> invalid(parallel_size,
                             std::numeric_limits<float>::quiet_NaN());
  for (std::size_t index = 0; index < invalid.size(); index += 3) {
    invalid[index] = index % 2 == 0
                         ? std::numeric_limits<float>::infinity()
                         : -std::numeric_limits<float>::infinity();
  }
  const auto expected_invalid = reference_build_histogram(invalid);
  const auto actual_invalid = build_histogram(invalid);
  expect(exact_histogram(expected_invalid, actual_invalid),
         "parallel all-invalid histogram matches the serial oracle",
         failures);
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

void test_speckle_overlap_and_options(int &failures) {
  using namespace satview::cpu;
  std::array<float, 10> storage{1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
                                6.0F, 7.0F, 8.0F, 9.0F, 10.0F};
  SpeckleOptions none;
  none.filter = SpeckleFilter::none;
  filter_speckle(std::span<const float>(storage.data(), 9),
                  std::span<float>(storage.data(), 9), 3, 3, none);
  const bool exact_in_place_preserved =
      std::equal(storage.begin(), storage.begin() + 9,
                 std::array{1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
                            6.0F, 7.0F, 8.0F, 9.0F}.begin());
  expect(exact_in_place_preserved,
         "None permits exact in-place input/output", failures);
  expect_invalid_argument(
      [&] {
        filter_speckle(std::span<const float>(storage.data(), 9),
                       std::span<float>(storage.data() + 1, 9), 3, 3, none);
      },
      "None rejects partial input/output overlap", failures);

  for (const auto filter : {SpeckleFilter::boxcar, SpeckleFilter::lee}) {
    SpeckleOptions options;
    options.filter = filter;
    options.window_size = 3;
    options.equivalent_number_of_looks = 2.0F;
    expect_invalid_argument(
        [&] {
          filter_speckle(std::span<const float>(storage.data(), 9),
                         std::span<float>(storage.data(), 9), 3, 3, options);
        },
        filter == SpeckleFilter::boxcar
            ? "Boxcar rejects exact input/output overlap"
            : "Lee rejects exact input/output overlap",
        failures);
    expect_invalid_argument(
        [&] {
          filter_speckle(std::span<const float>(storage.data(), 9),
                         std::span<float>(storage.data() + 1, 9), 3, 3,
                         options);
        },
        filter == SpeckleFilter::boxcar
            ? "Boxcar rejects partial input/output overlap"
            : "Lee rejects partial input/output overlap",
        failures);
  }

  constexpr std::array<float, 9> input{1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
                                       6.0F, 7.0F, 8.0F, 9.0F};
  std::array<float, input.size()> output{};
  SpeckleOptions boxcar;
  boxcar.filter = SpeckleFilter::boxcar;
  boxcar.window_size = 3;
  boxcar.equivalent_number_of_looks = std::numeric_limits<float>::quiet_NaN();
  filter_speckle(input, output, 3, 3, boxcar);
  expect(near(output[4], 5.0F), "Boxcar ignores Lee ENL", failures);

  SpeckleOptions lee = boxcar;
  lee.filter = SpeckleFilter::lee;
  expect_invalid_argument([&] { filter_speckle(input, output, 3, 3, lee); },
                          "Lee rejects non-finite ENL", failures);
  lee.equivalent_number_of_looks = 0.0F;
  expect_invalid_argument([&] { filter_speckle(input, output, 3, 3, lee); },
                          "Lee rejects non-positive ENL", failures);

  SpeckleOptions invalid_domain;
  invalid_domain.filter = SpeckleFilter::boxcar;
  invalid_domain.domain = static_cast<SpeckleDomain>(255);
  expect_invalid_argument(
      [&] { filter_speckle(input, output, 3, 3, invalid_domain); },
      "CPU speckle rejects an out-of-range domain", failures);
}

void test_parallel_speckle_matches_serial_reference(int &failures) {
  using namespace satview::cpu;
  constexpr std::array<std::array<std::size_t, 2>, 4> geometries{{
      {1, 1},
      {19, 1},
      {1, 23},
      {263, 257},
  }};
  constexpr std::array domains{
      SpeckleDomain::amplitude,
      SpeckleDomain::linear_power,
      SpeckleDomain::power_db,
  };
  constexpr std::array filters{
      SpeckleFilter::boxcar,
      SpeckleFilter::lee,
  };
  constexpr std::array<std::uint32_t, 3> windows{3, 5, 7};

  for (const auto geometry : geometries) {
    const auto width = geometry[0];
    const auto height = geometry[1];
    const auto count = width * height;
    const auto validity = make_filter_validity(count);
    for (const auto domain : domains) {
      const auto input = make_filter_input(count, domain);
      for (const auto filter : filters) {
        for (const auto window : windows) {
          SpeckleOptions options;
          options.filter = filter;
          options.domain = domain;
          options.window_size = window;
          options.equivalent_number_of_looks = 3.5F;
          std::vector<float> expected(count);
          std::vector<float> actual(count);
          reference_filter_speckle(input, expected, width, height, options,
                                   validity);
          filter_speckle(input, actual, width, height, options, validity);

          const bool exact =
              std::equal(expected.begin(), expected.end(), actual.begin(),
                         [](const float left, const float right) {
                           return exact_float(left, right);
                         });
          expect(exact,
                 "parallel Boxcar/Lee is bitwise equal to the serial oracle",
                 failures);
        }
      }
    }
  }
}

[[nodiscard]] double percentile(std::vector<double> values,
                                const double fraction) {
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(std::ceil(
                         fraction * static_cast<double>(values.size()))) -
                     1;
  return values[std::min(index, values.size() - 1)];
}

void run_speckle_benchmark_case(const satview::cpu::SpeckleFilter filter,
                                const std::uint32_t window) {
  using Clock = std::chrono::steady_clock;
  using namespace satview::cpu;
  constexpr std::size_t width = 2048;
  constexpr std::size_t height = 2048;
  constexpr std::size_t repetitions = 9;
  const auto input =
      make_filter_input(width * height, SpeckleDomain::linear_power);
  const auto validity = make_filter_validity(input.size());
  std::vector<float> reference(input.size());
  std::vector<float> candidate(input.size());
  SpeckleOptions options;
  options.filter = filter;
  options.domain = SpeckleDomain::linear_power;
  options.window_size = window;
  options.equivalent_number_of_looks = 3.5F;

  reference_filter_speckle(input, reference, width, height, options, validity);
  filter_speckle(input, candidate, width, height, options, validity);
  std::vector<double> baseline_ms;
  std::vector<double> candidate_ms;
  baseline_ms.reserve(repetitions);
  candidate_ms.reserve(repetitions);
  const auto time_baseline = [&] {
    const auto started = Clock::now();
    reference_filter_speckle(input, reference, width, height, options,
                             validity);
    baseline_ms.push_back(
        std::chrono::duration<double, std::milli>(Clock::now() - started)
            .count());
  };
  const auto time_candidate = [&] {
    const auto started = Clock::now();
    filter_speckle(input, candidate, width, height, options, validity);
    candidate_ms.push_back(
        std::chrono::duration<double, std::milli>(Clock::now() - started)
            .count());
  };
  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    if (repetition % 2 == 0) {
      time_baseline();
      time_candidate();
    } else {
      time_candidate();
      time_baseline();
    }
  }

  const bool exact =
      std::equal(reference.begin(), reference.end(), candidate.begin(),
                 [](const float left, const float right) {
                   return exact_float(left, right);
                 });
  std::cout << "CPU speckle benchmark,filter="
            << (filter == SpeckleFilter::boxcar ? "boxcar" : "lee")
            << ",window=" << window << ",pixels=" << input.size()
            << ",samples=" << repetitions
            << ",baseline_p50_ms=" << percentile(baseline_ms, 0.50)
            << ",baseline_p95_ms=" << percentile(baseline_ms, 0.95)
            << ",candidate_p50_ms=" << percentile(candidate_ms, 0.50)
            << ",candidate_p95_ms=" << percentile(candidate_ms, 0.95)
            << ",bitwise_equal=" << (exact ? "yes" : "no") << '\n';
}

void maybe_run_speckle_benchmark() {
#if defined(_MSC_VER)
  std::size_t required = 0;
  if (getenv_s(&required, nullptr, 0, "SATVIEW_CPU_SPECKLE_BENCHMARK") != 0 ||
      required != 2) {
    return;
  }
  std::array<char, 2> enabled{};
  if (getenv_s(&required, enabled.data(), enabled.size(),
               "SATVIEW_CPU_SPECKLE_BENCHMARK") != 0 ||
      std::string_view(enabled.data()) != "1") {
    return;
  }
#else
  const char *const enabled = std::getenv("SATVIEW_CPU_SPECKLE_BENCHMARK");
  if (enabled == nullptr || std::string_view(enabled) != "1") {
    return;
  }
#endif
  run_speckle_benchmark_case(satview::cpu::SpeckleFilter::boxcar, 5);
  run_speckle_benchmark_case(satview::cpu::SpeckleFilter::lee, 7);
}

void maybe_run_histogram_benchmark() {
#if defined(_MSC_VER)
  std::size_t required = 0;
  if (getenv_s(&required, nullptr, 0,
               "SATVIEW_CPU_HISTOGRAM_BENCHMARK") != 0 ||
      required != 2) {
    return;
  }
  std::array<char, 2> enabled{};
  if (getenv_s(&required, enabled.data(), enabled.size(),
               "SATVIEW_CPU_HISTOGRAM_BENCHMARK") != 0 ||
      std::string_view(enabled.data()) != "1") {
    return;
  }
#else
  const char *const enabled =
      std::getenv("SATVIEW_CPU_HISTOGRAM_BENCHMARK");
  if (enabled == nullptr || std::string_view(enabled) != "1") {
    return;
  }
#endif

  using Clock = std::chrono::steady_clock;
  using namespace satview::cpu;
  constexpr std::size_t pixels = 2048 * 2048;
  constexpr std::size_t repetitions = 11;
  std::vector<float> values(pixels);
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = static_cast<float>(
                        static_cast<std::int32_t>(index % 16381) - 8190) /
                    31.0F;
    if (index % 1009 == 0) {
      values[index] = std::numeric_limits<float>::quiet_NaN();
    } else if (index % 1237 == 0) {
      values[index] = std::numeric_limits<float>::infinity();
    }
  }

  auto baseline = reference_build_histogram(values);
  auto candidate = build_histogram(values);
  std::vector<double> baseline_ms;
  std::vector<double> candidate_ms;
  baseline_ms.reserve(repetitions);
  candidate_ms.reserve(repetitions);
  const auto time_baseline = [&] {
    const auto started = Clock::now();
    baseline = reference_build_histogram(values);
    baseline_ms.push_back(
        std::chrono::duration<double, std::milli>(Clock::now() - started)
            .count());
  };
  const auto time_candidate = [&] {
    const auto started = Clock::now();
    candidate = build_histogram(values);
    candidate_ms.push_back(
        std::chrono::duration<double, std::milli>(Clock::now() - started)
            .count());
  };
  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    if (repetition % 2 == 0) {
      time_baseline();
      time_candidate();
    } else {
      time_candidate();
      time_baseline();
    }
  }

  std::cout << "CPU histogram benchmark,pixels=" << values.size()
            << ",samples=" << repetitions
            << ",baseline_p50_ms=" << percentile(baseline_ms, 0.50)
            << ",baseline_p95_ms=" << percentile(baseline_ms, 0.95)
            << ",candidate_p50_ms=" << percentile(candidate_ms, 0.50)
            << ",candidate_p95_ms=" << percentile(candidate_ms, 0.95)
            << ",exact="
            << (exact_histogram(baseline, candidate) ? "yes" : "no")
            << '\n';
}

}  // namespace

int run_cpu_scientific_tests() {
  int failures = 0;
  test_complex_transforms(failures);
  test_real_transform_and_histogram(failures);
  test_parallel_histogram_matches_serial_reference(failures);
  test_speckle(failures);
  test_speckle_overlap_and_options(failures);
  test_parallel_speckle_matches_serial_reference(failures);
  maybe_run_speckle_benchmark();
  maybe_run_histogram_benchmark();
  return failures;
}
