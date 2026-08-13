#include "satview/composite_scientific.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>
#include <utility>

namespace {

using satview::composite::ComparePair;
using satview::composite::PauliPowers;

void expect(
    const bool condition,
    const std::string_view description,
    int& failures) {
  if (!condition) {
    std::cerr << "Composite scientific test failed: " << description << '\n';
    ++failures;
  }
}

[[nodiscard]] bool exact_float(const float first, const float second) noexcept {
  return std::bit_cast<std::uint32_t>(first) ==
         std::bit_cast<std::uint32_t>(second);
}

[[nodiscard]] bool near(
    const float first,
    const float second,
    const float tolerance = 1.0e-6F) noexcept {
  return std::abs(first - second) <= tolerance;
}

void expect_invalid_pauli(
    const PauliPowers& powers,
    const std::string_view description,
    int& failures) {
  expect(!powers.valid && std::isnan(powers.red_double_bounce) &&
             std::isnan(powers.green_volume) &&
             std::isnan(powers.blue_surface),
         description, failures);
}

void test_pauli_oracles(int& failures) {
  using satview::composite::compute_pauli_powers;

  const auto surface = compute_pauli_powers(1.0F, 0.0F, 1.0F, 1.0F, 0.0F);
  expect(surface.valid && exact_float(surface.red_double_bounce, 0.0F) &&
             exact_float(surface.green_volume, 0.0F) &&
             exact_float(surface.blue_surface, 2.0F),
         "surface-scattering oracle maps to blue", failures);

  const auto double_bounce =
      compute_pauli_powers(1.0F, 0.0F, 1.0F, -1.0F, 0.0F);
  expect(double_bounce.valid &&
             exact_float(double_bounce.red_double_bounce, 2.0F) &&
             exact_float(double_bounce.green_volume, 0.0F) &&
             exact_float(double_bounce.blue_surface, 0.0F),
         "double-bounce oracle maps to red", failures);

  const auto volume = compute_pauli_powers(0.0F, 1.0F, 0.0F, 0.0F, 0.0F);
  expect(volume.valid && exact_float(volume.red_double_bounce, 0.0F) &&
             exact_float(volume.green_volume, 2.0F) &&
             exact_float(volume.blue_surface, 0.0F),
         "cross-polarized oracle maps to green", failures);

  const auto mixed = compute_pauli_powers(9.0F, 0.25F, 4.0F, 3.0F, -7.0F);
  expect(mixed.valid && exact_float(mixed.red_double_bounce, 3.5F) &&
             exact_float(mixed.green_volume, 0.5F) &&
             exact_float(mixed.blue_surface, 9.5F),
         "mixed Pauli oracle and finite imaginary covariance", failures);

  const auto signed_powers =
      compute_pauli_powers(1.0F, 0.0F, 1.0F, 2.0F, 0.0F);
  expect(signed_powers.valid &&
             exact_float(signed_powers.red_double_bounce, -1.0F) &&
             exact_float(signed_powers.blue_surface, 3.0F),
         "scalar Pauli result preserves a negative non-PSD channel", failures);
  const auto display =
      satview::composite::pauli_rgb_display_powers(signed_powers);
  expect(display.valid && exact_float(display.red_double_bounce, 0.0F) &&
             exact_float(display.blue_surface, 3.0F),
         "RGB display alone clamps negative Pauli power", failures);
}

void test_pauli_invalid_inputs(int& failures) {
  using satview::composite::compute_pauli_powers;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();

  expect_invalid_pauli(compute_pauli_powers(nan, 0.0F, 0.0F, 0.0F, 0.0F),
                       "NaN diagonal is invalid", failures);
  expect_invalid_pauli(
      compute_pauli_powers(0.0F, infinity, 0.0F, 0.0F, 0.0F),
      "infinite diagonal is invalid", failures);
  expect_invalid_pauli(
      compute_pauli_powers(0.0F, 0.0F, 0.0F, nan, 0.0F),
      "NaN cross-covariance real component is invalid", failures);
  expect_invalid_pauli(
      compute_pauli_powers(0.0F, 0.0F, 0.0F, 0.0F, infinity),
      "infinite cross-covariance imaginary component is invalid", failures);
  expect_invalid_pauli(
      compute_pauli_powers(-1.0F, 0.0F, 0.0F, 0.0F, 0.0F),
      "negative HHHH diagonal is invalid", failures);
  expect_invalid_pauli(
      compute_pauli_powers(0.0F, -1.0F, 0.0F, 0.0F, 0.0F),
      "negative HVHV diagonal is invalid", failures);
  expect_invalid_pauli(
      compute_pauli_powers(0.0F, 0.0F, -1.0F, 0.0F, 0.0F),
      "negative VVVV diagonal is invalid", failures);
  expect_invalid_pauli(
      compute_pauli_powers(std::numeric_limits<float>::max(), 0.0F,
                           std::numeric_limits<float>::max(), 0.0F, 0.0F),
      "overflowing Pauli arithmetic is invalid", failures);

  const auto negative_zero =
      compute_pauli_powers(-0.0F, -0.0F, -0.0F, 0.0F, 0.0F);
  expect(negative_zero.valid, "negative-zero diagonal powers are nonnegative",
         failures);
}

void test_compare_operations(int& failures) {
  using satview::composite::compare_difference;
  using satview::composite::compare_pair;
  using satview::composite::compare_ratio;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float maximum = std::numeric_limits<float>::max();

  const auto pair = compare_pair(3.0F, true, 4.0F, false);
  expect(pair.first_valid && exact_float(pair.first, 3.0F) &&
             !pair.second_valid && std::isnan(pair.second),
         "pair validity remains independent", failures);
  expect(!compare_pair(nan, true, 1.0F, true).first_valid,
         "pair rejects nonfinite values despite caller validity", failures);

  const auto difference = compare_difference(7.0F, true, 2.0F, true);
  expect(difference.valid && exact_float(difference.value, 5.0F),
         "difference oracle", failures);
  const auto self_difference = compare_difference(-3.0F, true, -3.0F, true);
  expect(self_difference.valid && exact_float(self_difference.value, 0.0F),
         "A minus A is positive zero", failures);
  expect(!compare_difference(maximum, true, -maximum, true).valid,
         "overflowing difference is invalid", failures);
  expect(!compare_difference(1.0F, true, 1.0F, false).valid,
         "difference requires two valid inputs", failures);

  const auto ratio = compare_ratio(4.0F, true, 2.0F, true, 1.0e-20F);
  expect(ratio.valid && exact_float(ratio.value, 2.0F), "ratio oracle",
         failures);
  const auto self_ratio = compare_ratio(3.0F, true, 3.0F, true, 1.0e-20F);
  expect(self_ratio.valid && exact_float(self_ratio.value, 1.0F),
         "A divided by A is one", failures);
  expect(!compare_ratio(-1.0F, true, 1.0F, true, 0.0F).valid,
         "ratio rejects a negative numerator", failures);
  expect(!compare_ratio(1.0F, true, 0.0F, true, 0.0F).valid,
         "ratio rejects a denominator at epsilon", failures);
  expect(!compare_ratio(1.0F, true, 1.0F, true, -1.0F).valid,
         "ratio rejects a negative epsilon", failures);
  expect(!compare_ratio(maximum, true, 1.0e-20F, true, 0.0F).valid,
         "ratio rejects a nonfinite quotient", failures);
}

void test_bfloat_conversion(int& failures) {
  using satview::composite::bfloat_bits_to_float;
  using satview::composite::float_to_bfloat_bits;
  using satview::composite::kCanonicalBfloatNaN;

  const std::array<std::pair<float, std::uint16_t>, 9> cases{{
      {0.0F, std::uint16_t{0x0000U}},
      {-0.0F, std::uint16_t{0x8000U}},
      {1.0F, std::uint16_t{0x3f80U}},
      {-2.0F, std::uint16_t{0xc000U}},
      {std::numeric_limits<float>::max(), std::uint16_t{0x7f7fU}},
      {std::ldexp(1.0F, -126), std::uint16_t{0x0080U}},
      {std::numeric_limits<float>::infinity(), std::uint16_t{0x7f80U}},
      {-std::numeric_limits<float>::infinity(), std::uint16_t{0xff80U}},
      {std::numeric_limits<float>::quiet_NaN(), kCanonicalBfloatNaN},
  }};
  for (const auto& [value, expected] : cases) {
    expect(float_to_bfloat_bits(value) == expected,
           "known float-to-bfloat encoding", failures);
  }
  expect(float_to_bfloat_bits(1.00390625F) == 0x3f80U,
         "bfloat halfway rounds to the even lower value", failures);
  expect(float_to_bfloat_bits(1.01171875F) == 0x3f82U,
         "bfloat halfway rounds away from an odd lower value", failures);

  for (std::uint32_t raw = 0U; raw <= 0xffffU; ++raw) {
    const auto bfloat = static_cast<std::uint16_t>(raw);
    const bool is_nan = (bfloat & 0x7f80U) == 0x7f80U &&
                        (bfloat & 0x007fU) != 0U;
    const auto round_trip =
        float_to_bfloat_bits(bfloat_bits_to_float(bfloat));
    expect(round_trip == (is_nan ? kCanonicalBfloatNaN : bfloat),
           "exhaustive bfloat decode-encode round trip", failures);
  }
}

void test_compare_packing(int& failures) {
  using satview::composite::kCanonicalBfloatNaN;
  using satview::composite::pack_compare_pair_bits;
  using satview::composite::pack_compare_pair_r32;
  using satview::composite::unpack_compare_pair_bits;
  using satview::composite::unpack_compare_pair_r32;

  const ComparePair pair{1.0F, -2.0F, true, true};
  const std::uint32_t expected = 0xc0003f80U;
  expect(pack_compare_pair_bits(pair) == expected,
         "compare pair packs first into the low bfloat", failures);
  expect(std::bit_cast<std::uint32_t>(pack_compare_pair_r32(pair)) == expected,
         "compare pair R32 transport preserves every bit", failures);
  const auto unpacked = unpack_compare_pair_bits(expected);
  expect(unpacked.first_valid && unpacked.second_valid &&
             exact_float(unpacked.first, 1.0F) &&
             exact_float(unpacked.second, -2.0F),
         "compare pair unpacks exactly", failures);

  const ComparePair partial{4.0F, 8.0F, false, true};
  const auto partial_bits = pack_compare_pair_bits(partial);
  expect(static_cast<std::uint16_t>(partial_bits) == kCanonicalBfloatNaN,
         "invalid compare channel uses canonical bfloat NaN", failures);
  const auto partial_unpacked = unpack_compare_pair_r32(
      std::bit_cast<float>(partial_bits));
  expect(!partial_unpacked.first_valid && std::isnan(partial_unpacked.first) &&
             partial_unpacked.second_valid &&
             exact_float(partial_unpacked.second, 8.0F),
         "packed pair retains per-channel validity", failures);

  const ComparePair full_range{
      std::numeric_limits<float>::max(), 1.0F, true, true};
  const auto full_range_bits = pack_compare_pair_bits(full_range);
  const auto full_range_unpacked =
      unpack_compare_pair_bits(full_range_bits);
  expect(full_range_unpacked.first_valid &&
             std::isfinite(full_range_unpacked.first) &&
             full_range_unpacked.first > 3.0e38F,
         "compare packing preserves the finite float32 exponent range",
         failures);
  expect(!unpack_compare_pair_bits(0x00007f80U).first_valid,
         "compare decoder rejects an injected bfloat infinity", failures);
}

void test_pauli_packing(int& failures) {
  using namespace satview::composite;
  const PauliPowers powers{1.0F, 1.0e-10F, 1.0e5F, true};
  constexpr std::uint32_t expected =
      682U | (0U << 10U) | (1023U << 20U) | (kPauliRgbValidTag << 30U);
  const auto bits = pack_pauli_rgb_bits(powers);
  expect(bits == expected, "Pauli RGB fixed-range quantization", failures);
  expect(std::bit_cast<std::uint32_t>(pack_pauli_rgb_r32(powers)) == bits,
         "Pauli RGB R32 transport preserves every bit", failures);
  expect(std::isfinite(pack_pauli_rgb_r32(powers)),
         "valid Pauli tag produces a finite transport float", failures);

  const auto unpacked = unpack_pauli_rgb_bits(bits);
  constexpr float half_step =
      0.5F * (kPauliRgbMaximumDb - kPauliRgbMinimumDb) /
      static_cast<float>(kPauliRgbChannelMaximum);
  expect(unpacked.valid && near(unpacked.red, 0.0F, half_step) &&
             near(unpacked.green, kPauliRgbMinimumDb, half_step) &&
             near(unpacked.blue, kPauliRgbMaximumDb, half_step),
         "Pauli RGB decode stays within half a quantization step", failures);

  const PauliPowers clipped{-5.0F, 0.0F, 1.0F, true};
  const auto clipped_db = unpack_pauli_rgb_bits(pack_pauli_rgb_bits(clipped));
  expect(clipped_db.valid &&
             exact_float(clipped_db.red, kPauliRgbMinimumDb),
         "negative display power encodes at the dB floor", failures);

  const PauliPowers invalid{1.0F, 1.0F, 1.0F, false};
  expect(pack_pauli_rgb_bits(invalid) == 0U,
         "invalid Pauli sample has an untagged encoding", failures);
  expect(!unpack_pauli_rgb_bits(0U).valid &&
             !unpack_pauli_rgb_bits(0x40000000U).valid &&
             !unpack_pauli_rgb_bits(0xc0000000U).valid,
         "Pauli decoder rejects every non-valid tag", failures);

  const auto transported = unpack_pauli_rgb_r32(pack_pauli_rgb_r32(powers));
  expect(transported.valid && exact_float(transported.red, unpacked.red) &&
             exact_float(transported.green, unpacked.green) &&
             exact_float(transported.blue, unpacked.blue),
         "Pauli R32 unpack matches bit unpack", failures);
}

}  // namespace

int run_composite_scientific_tests() {
  int failures = 0;
  test_pauli_oracles(failures);
  test_pauli_invalid_inputs(failures);
  test_compare_operations(failures);
  test_bfloat_conversion(failures);
  test_compare_packing(failures);
  test_pauli_packing(failures);
  return failures;
}

#if defined(SATVIEW_COMPOSITE_SCIENTIFIC_STANDALONE)
int main() {
  return run_composite_scientific_tests();
}
#endif
