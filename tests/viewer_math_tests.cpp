#include "satview/viewer_math.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace {

void expect(
    const bool condition,
    const std::string_view description,
    int& failures) {
    if (!condition) {
        std::cerr << "Viewer math test failed: " << description << '\n';
        ++failures;
    }
}

template <typename Exception, typename Function>
void expect_throws(
    Function&& function,
    const std::string_view description,
    int& failures) {
    try {
        function();
        expect(false, description, failures);
    } catch (const Exception&) {
    } catch (...) {
        expect(false, description, failures);
    }
}

void test_mosaic_geometry(int& failures) {
    using satview::viewer::MosaicGeometry;
    using satview::viewer::make_mosaic_geometry;

    constexpr std::uint32_t chunk = 512;
    constexpr std::uint64_t height = 4ULL * chunk + 123;
    constexpr std::uint64_t width = 6ULL * chunk + 77;
    constexpr std::uint64_t rows = 5;
    constexpr std::uint64_t columns = 7;

    expect(
        make_mosaic_geometry(
            2, 3, rows, columns, chunk, chunk, height, width, 1) ==
            MosaicGeometry{
                .start_chunk_row = 2,
                .start_chunk_column = 3,
                .chunk_rows = 1,
                .chunk_columns = 1,
                .pixel_row = 1024,
                .pixel_column = 1536,
                .pixel_height = 512,
                .pixel_width = 512,
            },
        "1x focus maps to exactly one physical chunk",
        failures);

    expect(
        make_mosaic_geometry(
            2, 3, rows, columns, chunk, chunk, height, width, 2) ==
            MosaicGeometry{
                .start_chunk_row = 1,
                .start_chunk_column = 2,
                .chunk_rows = 2,
                .chunk_columns = 2,
                .pixel_row = 512,
                .pixel_column = 1024,
                .pixel_height = 1024,
                .pixel_width = 1024,
            },
        "2x focus includes the preceding and focused chunks",
        failures);

    expect(
        make_mosaic_geometry(
            2, 3, rows, columns, chunk, chunk, height, width, 4) ==
            MosaicGeometry{
                .start_chunk_row = 0,
                .start_chunk_column = 1,
                .chunk_rows = 4,
                .chunk_columns = 4,
                .pixel_row = 0,
                .pixel_column = 512,
                .pixel_height = 2048,
                .pixel_width = 2048,
            },
        "4x focus is centered when away from an edge",
        failures);

    expect(
        make_mosaic_geometry(
            4, 6, rows, columns, chunk, chunk, height, width, 4) ==
            MosaicGeometry{
                .start_chunk_row = 1,
                .start_chunk_column = 3,
                .chunk_rows = 4,
                .chunk_columns = 4,
                .pixel_row = 512,
                .pixel_column = 1536,
                .pixel_height = 1659,
                .pixel_width = 1613,
            },
        "4x bottom-right focus shifts inward and preserves partial edges",
        failures);

    expect(
        make_mosaic_geometry(
            0, 0, rows, columns, chunk, chunk, height, width, 4)
                .start_chunk_row == 0,
        "top edge never underflows",
        failures);

    expect(
        make_mosaic_geometry(
            0, 1, 1, 2, chunk, chunk, 300, 700, 4) ==
            MosaicGeometry{
                .start_chunk_row = 0,
                .start_chunk_column = 0,
                .chunk_rows = 1,
                .chunk_columns = 2,
                .pixel_row = 0,
                .pixel_column = 0,
                .pixel_height = 300,
                .pixel_width = 700,
            },
        "span larger than a small raster uses each available chunk once",
        failures);

    expect_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(make_mosaic_geometry(
                0, 0, rows, columns, chunk, chunk, height, width, 3));
        },
        "unsupported span is rejected",
        failures);
    expect_throws<std::out_of_range>(
        [&] {
            static_cast<void>(make_mosaic_geometry(
                rows, 0, rows, columns, chunk, chunk, height, width, 1));
        },
        "row focus outside the chunk grid is rejected",
        failures);
    expect_throws<std::out_of_range>(
        [&] {
            static_cast<void>(make_mosaic_geometry(
                0, columns, rows, columns, chunk, chunk, height, width, 1));
        },
        "column focus outside the chunk grid is rejected",
        failures);
    expect_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(make_mosaic_geometry(
                0, 0, 0, columns, chunk, chunk, height, width, 1));
        },
        "empty chunk grid is rejected",
        failures);
    expect_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(make_mosaic_geometry(
                0, 0, rows, columns, 0, chunk, height, width, 1));
        },
        "zero chunk extent is rejected",
        failures);
    expect_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(make_mosaic_geometry(
                0, 0, rows + 1, columns, chunk, chunk, height, width, 1));
        },
        "chunk-grid and raster mismatch is rejected",
        failures);
}

void test_checked_mosaic_bytes(int& failures) {
    using satview::viewer::checked_mosaic_bytes;

    constexpr std::size_t full_limit =
        std::numeric_limits<std::size_t>::max();
    expect(
        checked_mosaic_bytes(512, 512, 8, full_limit) == 2'097'152,
        "complex64 1x byte count",
        failures);
    expect(
        checked_mosaic_bytes(2048, 2048, 8, full_limit) == 33'554'432,
        "complex64 4x byte count",
        failures);
    expect(
        checked_mosaic_bytes(2048, 2048, sizeof(float), 16'777'216) ==
            16'777'216,
        "allocation equal to the safety limit is accepted",
        failures);

    expect_throws<std::length_error>(
        [&] {
            static_cast<void>(
                checked_mosaic_bytes(
                    2048, 2048, sizeof(float), 16'777'215));
        },
        "allocation above the safety limit is rejected",
        failures);
    expect_throws<std::length_error>(
        [&] {
            static_cast<void>(checked_mosaic_bytes(
                std::numeric_limits<std::uint64_t>::max(),
                2,
                1,
                full_limit));
        },
        "pixel-count multiplication overflow is rejected",
        failures);
    expect_throws<std::length_error>(
        [&] {
            static_cast<void>(checked_mosaic_bytes(
                1,
                std::numeric_limits<std::size_t>::max(),
                2,
                full_limit));
        },
        "element-size multiplication overflow is rejected",
        failures);
    expect_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                checked_mosaic_bytes(0, 512, sizeof(float), full_limit));
        },
        "zero allocation extent is rejected",
        failures);
    expect_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                checked_mosaic_bytes(512, 512, 0, full_limit));
        },
        "zero element size is rejected",
        failures);
}

void test_maximum_chunk_aligned_extent(int& failures) {
    using satview::viewer::maximum_chunk_aligned_extent;

    expect(
        maximum_chunk_aligned_extent(7, 100, 4) == 12,
        "alignment accounts for expansion on both request boundaries",
        failures);
    expect(
        maximum_chunk_aligned_extent(512, 20'000, 512) == 1024,
        "same-sized science and mask chunks retain a tight staging bound",
        failures);
    expect(
        maximum_chunk_aligned_extent(7, 10, 4) == 10,
        "aligned extent is clipped to the dataset dimension",
        failures);
    expect(
        maximum_chunk_aligned_extent(
            std::numeric_limits<std::uint64_t>::max() - 1,
            std::numeric_limits<std::uint64_t>::max(),
            std::numeric_limits<std::uint64_t>::max()) ==
            std::numeric_limits<std::uint64_t>::max(),
        "alignment bound saturates instead of overflowing",
        failures);
    expect_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                maximum_chunk_aligned_extent(7, 100, 0));
        },
        "zero chunk extent is rejected",
        failures);
}

}  // namespace

int run_viewer_math_tests() {
    int failures = 0;
    test_mosaic_geometry(failures);
    test_checked_mosaic_bytes(failures);
    test_maximum_chunk_aligned_extent(failures);
    return failures;
}
