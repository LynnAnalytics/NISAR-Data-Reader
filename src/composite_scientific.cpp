#include "satview/composite_scientific.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace satview::composite {
namespace {

constexpr std::uint32_t kPauliRedShift = 0U;
constexpr std::uint32_t kPauliGreenShift = 10U;
constexpr std::uint32_t kPauliBlueShift = 20U;
constexpr std::uint32_t kPauliTagShift = 30U;
constexpr std::uint32_t kPauliChannelMask = 0x3ffU;
constexpr std::uint32_t kPauliTagMask = 0x3U;
constexpr std::uint32_t kFloatCanonicalNaN = 0x7fc00000U;

[[nodiscard]] float invalid_float() noexcept {
  return std::bit_cast<float>(kFloatCanonicalNaN);
}

[[nodiscard]] bool finite(const float value) noexcept {
  return std::isfinite(value);
}

[[nodiscard]] bool bfloat_is_finite(const std::uint16_t bits) noexcept {
  return (bits & 0x7f80U) != 0x7f80U;
}

[[nodiscard]] float power_to_db(const float power) noexcept {
  if (power <= 0.0F) {
    return kPauliRgbMinimumDb;
  }
  return std::clamp(
      10.0F * std::log10(power), kPauliRgbMinimumDb, kPauliRgbMaximumDb);
}

[[nodiscard]] std::uint32_t encode_db(const float value) noexcept {
  constexpr float scale = static_cast<float>(kPauliRgbChannelMaximum) /
                          (kPauliRgbMaximumDb - kPauliRgbMinimumDb);
  const float mapped = (std::clamp(value, kPauliRgbMinimumDb,
                                   kPauliRgbMaximumDb) -
                        kPauliRgbMinimumDb) *
                       scale;
  return static_cast<std::uint32_t>(std::floor(mapped + 0.5F));
}

[[nodiscard]] float decode_db(const std::uint32_t code) noexcept {
  constexpr float scale = (kPauliRgbMaximumDb - kPauliRgbMinimumDb) /
                          static_cast<float>(kPauliRgbChannelMaximum);
  return kPauliRgbMinimumDb + static_cast<float>(code) * scale;
}

}  // namespace

PauliPowers compute_pauli_powers(
    const float hhhh,
    const float hvhv,
    const float vvvv,
    const float hhvv_real,
    const float hhvv_imaginary) noexcept {
  const float invalid = invalid_float();
  if (!finite(hhhh) || !finite(hvhv) || !finite(vvvv) ||
      !finite(hhvv_real) || !finite(hhvv_imaginary) || hhhh < 0.0F ||
      hvhv < 0.0F || vvvv < 0.0F) {
    return {invalid, invalid, invalid, false};
  }

  const float diagonal_sum = hhhh + vvvv;
  const float twice_cross = 2.0F * hhvv_real;
  const float red = 0.5F * (diagonal_sum - twice_cross);
  const float green = 2.0F * hvhv;
  const float blue = 0.5F * (diagonal_sum + twice_cross);
  if (!finite(red) || !finite(green) || !finite(blue)) {
    return {invalid, invalid, invalid, false};
  }
  return {red, green, blue, true};
}

PauliPowers pauli_rgb_display_powers(const PauliPowers& powers) noexcept {
  const float invalid = invalid_float();
  if (!powers.valid || !finite(powers.red_double_bounce) ||
      !finite(powers.green_volume) || !finite(powers.blue_surface)) {
    return {invalid, invalid, invalid, false};
  }
  return {
      std::max(powers.red_double_bounce, 0.0F),
      std::max(powers.green_volume, 0.0F),
      std::max(powers.blue_surface, 0.0F),
      true,
  };
}

PauliRgbDb pauli_rgb_db(const PauliPowers& powers) noexcept {
  const auto display = pauli_rgb_display_powers(powers);
  const float invalid = invalid_float();
  if (!display.valid) {
    return {invalid, invalid, invalid, false};
  }
  return {
      power_to_db(display.red_double_bounce),
      power_to_db(display.green_volume),
      power_to_db(display.blue_surface),
      true,
  };
}

ComparePair compare_pair(
    const float first,
    const bool first_valid,
    const float second,
    const bool second_valid) noexcept {
  const bool clean_first = first_valid && finite(first);
  const bool clean_second = second_valid && finite(second);
  const float invalid = invalid_float();
  return {
      clean_first ? first : invalid,
      clean_second ? second : invalid,
      clean_first,
      clean_second,
  };
}

ScalarResult compare_difference(
    const float first,
    const bool first_valid,
    const float second,
    const bool second_valid) noexcept {
  const auto pair = compare_pair(first, first_valid, second, second_valid);
  if (!pair.first_valid || !pair.second_valid) {
    return {invalid_float(), false};
  }
  float difference = pair.first - pair.second;
  if (!finite(difference)) {
    return {invalid_float(), false};
  }
  if (difference == 0.0F) {
    difference = 0.0F;
  }
  return {difference, true};
}

ScalarResult compare_ratio(
    const float numerator,
    const bool numerator_valid,
    const float denominator,
    const bool denominator_valid,
    const float epsilon) noexcept {
  if (!numerator_valid || !denominator_valid || !finite(numerator) ||
      !finite(denominator) || !finite(epsilon) || epsilon < 0.0F ||
      numerator < 0.0F || denominator <= epsilon) {
    return {invalid_float(), false};
  }
  const float ratio = numerator / denominator;
  if (!finite(ratio)) {
    return {invalid_float(), false};
  }
  return {ratio, true};
}

std::uint16_t float_to_bfloat_bits(const float value) noexcept {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t exponent = bits & 0x7f800000U;
  const std::uint32_t mantissa = bits & 0x007fffffU;
  if (exponent == 0x7f800000U) {
    if (mantissa != 0U) {
      return kCanonicalBfloatNaN;
    }
    return static_cast<std::uint16_t>(bits >> 16U);
  }

  const std::uint32_t rounded =
      bits + 0x7fffU + ((bits >> 16U) & 1U);
  const auto encoded = static_cast<std::uint16_t>(rounded >> 16U);
  if (!bfloat_is_finite(encoded)) {
    // Rounding the largest finite float must not turn a valid display sample
    // into infinity/invalid. Saturate at the signed bfloat maximum instead.
    return static_cast<std::uint16_t>(
        (bits >> 16U & 0x8000U) | 0x7f7fU);
  }
  return encoded;
}

float bfloat_bits_to_float(const std::uint16_t bits) noexcept {
  if (!bfloat_is_finite(bits) && (bits & 0x007fU) != 0U) {
    return invalid_float();
  }
  return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

std::uint32_t pack_compare_pair_bits(const ComparePair& pair) noexcept {
  const auto encode = [](const float value, const bool valid) noexcept {
    if (!valid || !finite(value)) {
      return kCanonicalBfloatNaN;
    }
    const auto packed = float_to_bfloat_bits(value);
    return bfloat_is_finite(packed) ? packed : kCanonicalBfloatNaN;
  };
  const std::uint16_t first = encode(pair.first, pair.first_valid);
  const std::uint16_t second = encode(pair.second, pair.second_valid);
  return static_cast<std::uint32_t>(first) |
         (static_cast<std::uint32_t>(second) << 16U);
}

float pack_compare_pair_r32(const ComparePair& pair) noexcept {
  return std::bit_cast<float>(pack_compare_pair_bits(pair));
}

ComparePair unpack_compare_pair_bits(const std::uint32_t bits) noexcept {
  const auto first_bits = static_cast<std::uint16_t>(bits & 0xffffU);
  const auto second_bits = static_cast<std::uint16_t>(bits >> 16U);
  const bool first_valid = bfloat_is_finite(first_bits);
  const bool second_valid = bfloat_is_finite(second_bits);
  return {
      first_valid ? bfloat_bits_to_float(first_bits) : invalid_float(),
      second_valid ? bfloat_bits_to_float(second_bits) : invalid_float(),
      first_valid,
      second_valid,
  };
}

ComparePair unpack_compare_pair_r32(const float transport) noexcept {
  return unpack_compare_pair_bits(std::bit_cast<std::uint32_t>(transport));
}

std::uint32_t pack_pauli_rgb_bits(const PauliPowers& powers) noexcept {
  const auto db = pauli_rgb_db(powers);
  if (!db.valid) {
    return 0U;
  }
  return (encode_db(db.red) << kPauliRedShift) |
         (encode_db(db.green) << kPauliGreenShift) |
         (encode_db(db.blue) << kPauliBlueShift) |
         (kPauliRgbValidTag << kPauliTagShift);
}

float pack_pauli_rgb_r32(const PauliPowers& powers) noexcept {
  return std::bit_cast<float>(pack_pauli_rgb_bits(powers));
}

PauliRgbDb unpack_pauli_rgb_bits(const std::uint32_t bits) noexcept {
  if (((bits >> kPauliTagShift) & kPauliTagMask) != kPauliRgbValidTag) {
    const float invalid = invalid_float();
    return {invalid, invalid, invalid, false};
  }
  return {
      decode_db((bits >> kPauliRedShift) & kPauliChannelMask),
      decode_db((bits >> kPauliGreenShift) & kPauliChannelMask),
      decode_db((bits >> kPauliBlueShift) & kPauliChannelMask),
      true,
  };
}

PauliRgbDb unpack_pauli_rgb_r32(const float transport) noexcept {
  return unpack_pauli_rgb_bits(std::bit_cast<std::uint32_t>(transport));
}

}  // namespace satview::composite
