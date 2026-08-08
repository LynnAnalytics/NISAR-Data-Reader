#include "colormaps.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>

namespace {

using Rgba = std::array<std::uint8_t, 4>;

void expect(
    const bool condition,
    const std::string_view description,
    int& failures) {
    if (!condition) {
        std::cerr << "Colormap test failed: " << description << '\n';
        ++failures;
    }
}

void test_metadata_and_indexing(int& failures) {
    using namespace satview::viewer;

    static_assert(static_cast<std::uint32_t>(ColormapId::grayscale) == 0);
    static_assert(static_cast<std::uint32_t>(ColormapId::turbo) == 1);
    static_assert(static_cast<std::uint32_t>(ColormapId::cyclic_phase) == 2);
    static_assert(static_cast<std::uint32_t>(
                      ColormapId::d3_cubehelix_default) == 19);

    expect(kColormapCount == 20, "stable palette count", failures);
    expect(kColormapSampleCount == 256, "stable samples per palette", failures);
    expect(
        kColormapLabels.size() == kColormapCount,
        "every palette ID has one label",
        failures);
    expect(
        kColormapRgbaByteCount == 20ULL * 256ULL * 4ULL,
        "atlas byte count",
        failures);

    const auto pixels = colormap_rgba8();
    expect(
        pixels.size() == kColormapRgbaByteCount,
        "atlas span exposes every RGBA byte",
        failures);
    expect(
        colormap_byte_offset(0, 0) == 0,
        "first atlas offset",
        failures);
    expect(
        colormap_byte_offset(0, 255) + 4 ==
            colormap_byte_offset(1, 0),
        "palette rows are tightly packed",
        failures);
    expect(
        colormap_byte_offset(kColormapCount - 1, 255) + 4 ==
            pixels.size(),
        "last sample ends at the atlas boundary",
        failures);

    constexpr std::array<std::uint32_t, 5> representative_samples{
        0, 1, 127, 128, 255};
    for (std::uint32_t palette = 0; palette < kColormapCount; ++palette) {
        for (const auto sample : representative_samples) {
            const auto offset = colormap_byte_offset(palette, sample);
            const auto rgba = colormap_sample(palette, sample);
            expect(
                rgba == Rgba{
                    pixels[offset],
                    pixels[offset + 1],
                    pixels[offset + 2],
                    pixels[offset + 3],
                },
                "CPU sample helper follows row-major atlas indexing",
                failures);
        }
    }

    expect(
        colormap_sample(
            std::numeric_limits<std::uint32_t>::max(),
            std::numeric_limits<std::uint32_t>::max()) ==
            colormap_sample(kColormapCount - 1, kColormapSampleCount - 1),
        "sample helper clamps both coordinates",
        failures);
}

void test_canonical_pixels(int& failures) {
    using namespace satview::viewer;

    constexpr std::array<Rgba, 20> first{{
        {0, 0, 0, 255},
        {35, 23, 27, 255},
        {255, 64, 64, 255},
        {23, 28, 66, 255},
        {0, 0, 3, 255},
        {20, 16, 26, 255},
        {68, 1, 84, 255},
        {2, 76, 129, 255},
        {0, 0, 0, 255},
        {25, 38, 89, 255},
        {127, 28, 0, 255},
        {0, 108, 198, 255},
        {68, 1, 84, 255},
        {0, 32, 81, 255},
        {0, 0, 4, 255},
        {0, 0, 4, 255},
        {13, 8, 135, 255},
        {5, 48, 97, 255},
        {45, 0, 75, 255},
        {0, 0, 0, 255},
    }};
    constexpr std::array<Rgba, 20> last{{
        {255, 255, 255, 255},
        {144, 12, 0, 255},
        {255, 64, 64, 255},
        {60, 9, 17, 255},
        {64, 0, 76, 255},
        {46, 0, 60, 255},
        {124, 2, 167, 255},
        {252, 149, 253, 255},
        {122, 4, 2, 255},
        {252, 250, 227, 255},
        {27, 52, 153, 255},
        {166, 0, 0, 255},
        {253, 231, 37, 255},
        {253, 234, 69, 255},
        {252, 255, 164, 255},
        {252, 253, 191, 255},
        {240, 249, 33, 255},
        {103, 0, 31, 255},
        {127, 59, 8, 255},
        {255, 255, 255, 255},
    }};

    for (std::uint32_t palette = 0; palette < kColormapCount; ++palette) {
        expect(
            colormap_sample(palette, 0) == first[palette],
            "canonical palette first endpoint",
            failures);
        expect(
            colormap_sample(palette, kColormapSampleCount - 1) ==
                last[palette],
            "canonical palette last endpoint",
            failures);
    }

    for (std::uint32_t sample = 0; sample < kColormapSampleCount; ++sample) {
        const auto grayscale =
            colormap_sample(
                static_cast<std::uint32_t>(ColormapId::grayscale), sample);
        expect(
            grayscale == Rgba{
                static_cast<std::uint8_t>(sample),
                static_cast<std::uint8_t>(sample),
                static_cast<std::uint8_t>(sample),
                255,
            },
            "grayscale is an exact 0..255 ramp",
            failures);
    }

    expect(
        colormap_sample(
            static_cast<std::uint32_t>(ColormapId::cyclic_phase), 0) ==
            colormap_sample(
                static_cast<std::uint32_t>(ColormapId::cyclic_phase),
                kColormapSampleCount - 1),
        "cyclic phase endpoints join exactly",
        failures);

    const auto pixels = colormap_rgba8();
    for (std::size_t index = 3; index < pixels.size(); index += 4) {
        expect(pixels[index] == 255, "every LUT sample is opaque", failures);
    }
}

}  // namespace

int run_colormap_tests() {
    int failures = 0;
    test_metadata_and_indexing(failures);
    test_canonical_pixels(failures);
    return failures;
}
