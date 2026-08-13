#pragma once

#include <cstdint>

namespace satview::composite {

inline constexpr float kPauliRgbMinimumDb = -100.0F;
inline constexpr float kPauliRgbMaximumDb = 50.0F;
inline constexpr std::uint32_t kPauliRgbChannelMaximum = 1023U;
inline constexpr std::uint32_t kPauliRgbValidTag = 0x2U;
inline constexpr std::uint16_t kCanonicalBfloatNaN = 0x7fc0U;

struct ScalarResult {
  float value = 0.0F;
  bool valid = false;
};

struct ComparePair {
  float first = 0.0F;
  float second = 0.0F;
  bool first_valid = false;
  bool second_valid = false;
};

struct PauliPowers {
  float red_double_bounce = 0.0F;
  float green_volume = 0.0F;
  float blue_surface = 0.0F;
  bool valid = false;
};

struct PauliRgbDb {
  float red = 0.0F;
  float green = 0.0F;
  float blue = 0.0F;
  bool valid = false;
};

[[nodiscard]] PauliPowers compute_pauli_powers(
    float hhhh,
    float hvhv,
    float vvvv,
    float hhvv_real,
    float hhvv_imaginary) noexcept;

[[nodiscard]] PauliPowers pauli_rgb_display_powers(
    const PauliPowers& powers) noexcept;

[[nodiscard]] PauliRgbDb pauli_rgb_db(
    const PauliPowers& powers) noexcept;

[[nodiscard]] ComparePair compare_pair(
    float first,
    bool first_valid,
    float second,
    bool second_valid) noexcept;

[[nodiscard]] ScalarResult compare_difference(
    float first,
    bool first_valid,
    float second,
    bool second_valid) noexcept;

[[nodiscard]] ScalarResult compare_ratio(
    float numerator,
    bool numerator_valid,
    float denominator,
    bool denominator_valid,
    float epsilon) noexcept;

[[nodiscard]] std::uint16_t float_to_bfloat_bits(float value) noexcept;
[[nodiscard]] float bfloat_bits_to_float(std::uint16_t bits) noexcept;

[[nodiscard]] std::uint32_t pack_compare_pair_bits(
    const ComparePair& pair) noexcept;
[[nodiscard]] float pack_compare_pair_r32(const ComparePair& pair) noexcept;
[[nodiscard]] ComparePair unpack_compare_pair_bits(std::uint32_t bits) noexcept;
[[nodiscard]] ComparePair unpack_compare_pair_r32(float transport) noexcept;

[[nodiscard]] std::uint32_t pack_pauli_rgb_bits(
    const PauliPowers& powers) noexcept;
[[nodiscard]] float pack_pauli_rgb_r32(const PauliPowers& powers) noexcept;
[[nodiscard]] PauliRgbDb unpack_pauli_rgb_bits(std::uint32_t bits) noexcept;
[[nodiscard]] PauliRgbDb unpack_pauli_rgb_r32(float transport) noexcept;

}  // namespace satview::composite
