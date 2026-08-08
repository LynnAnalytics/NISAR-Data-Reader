#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace satview::viewer {

inline constexpr std::uint32_t kColormapSampleCount = 256;

enum class ColormapId : std::uint32_t {
    grayscale = 0,
    turbo = 1,
    cyclic_phase = 2,
    cmweather_balance = 3,
    cmweather_chase_spectral = 4,
    cmweather_spectral_extended = 5,
    cmweather_plasmidis = 6,
    cmweather_bgyp = 7,
    cmweather_turbone = 8,
    cmweather_cm_depol = 9,
    cmweather_cm_rhohv = 10,
    cmweather_homeyer_rainbow = 11,
    d3_viridis = 12,
    d3_cividis = 13,
    d3_inferno = 14,
    d3_magma = 15,
    d3_plasma = 16,
    d3_rdbu_blue_red = 17,
    d3_puor = 18,
    d3_cubehelix_default = 19,
};

inline constexpr std::array kColormapLabels{
    "Grayscale",
    "Turbo",
    "Cyclic phase",
    "cmweather balance",
    "cmweather ChaseSpectral",
    "cmweather SpectralExtended",
    "cmweather plasmidis",
    "cmweather bgyp",
    "cmweather turbone",
    "cmweather CM_depol",
    "cmweather CM_rhohv",
    "cmweather HomeyerRainbow",
    "D3 Viridis",
    "D3 Cividis",
    "D3 Inferno",
    "D3 Magma",
    "D3 Plasma",
    "D3 RdBu (blue-red)",
    "D3 PuOr",
    "D3 Cubehelix Default",
};

inline constexpr std::uint32_t kColormapCount =
    static_cast<std::uint32_t>(kColormapLabels.size());
inline constexpr std::size_t kColormapRgbaByteCount =
    static_cast<std::size_t>(kColormapCount) * kColormapSampleCount * 4;

[[nodiscard]] constexpr std::size_t colormap_byte_offset(
    const std::uint32_t palette,
    const std::uint32_t sample) noexcept
{
    return (
        static_cast<std::size_t>(palette) * kColormapSampleCount + sample) *
        4;
}

// Row-major 256 x kColormapCount canonical sRGB-encoded RGBA8 pixels. Upload
// unchanged to a VK_FORMAT_R8G8B8A8_UNORM image: the viewer's UNORM swapchain
// target must receive these display-encoded bytes without an automatic decode.
// Palette ID is the texture row; sample with nearest filtering/texelFetch.
[[nodiscard]] std::span<const std::uint8_t> colormap_rgba8() noexcept;

// Clamps both coordinates. This is primarily useful to verify CPU/GPU indexing
// and endpoint behavior without depending on Vulkan.
[[nodiscard]] std::array<std::uint8_t, 4> colormap_sample(
    std::uint32_t palette,
    std::uint32_t sample) noexcept;

} // namespace satview::viewer
