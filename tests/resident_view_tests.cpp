#include "satview/resident_view.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace {

using satview::viewer::ChunkedRasterGeometry;
using satview::viewer::MosaicGeometry;
using satview::viewer::RasterWindow;
using satview::viewer::ResidentViewSelection;
using satview::viewer::select_resident_view;

void expect(
    const bool condition,
    const std::string_view description,
    int& failures) {
    if (!condition) {
        std::cerr << "Resident view test failed: " << description << '\n';
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

constexpr ChunkedRasterGeometry kRaster{
    .raster_height = 4ULL * 512 + 123,
    .raster_width = 6ULL * 512 + 77,
    .total_chunk_rows = 5,
    .total_chunk_columns = 7,
    .chunk_height = 512,
    .chunk_width = 512,
};

void test_smallest_span_and_boundaries(int& failures) {
    expect(
        select_resident_view(
            kRaster, RasterWindow{600.0, 1100.0, 700.0, 1200.0}) ==
            ResidentViewSelection{
                .requested_span = 1,
                .focus_chunk_row = 1,
                .focus_chunk_column = 2,
                .mosaic =
                    MosaicGeometry{
                        .start_chunk_row = 1,
                        .start_chunk_column = 2,
                        .chunk_rows = 1,
                        .chunk_columns = 1,
                        .pixel_row = 512,
                        .pixel_column = 1024,
                        .pixel_height = 512,
                        .pixel_width = 512,
                    },
            },
        "one-chunk visible window selects a 1x native mosaic",
        failures);

    const auto exact =
        select_resident_view(
            kRaster, RasterWindow{512.0, 1024.0, 1024.0, 1536.0});
    expect(
        exact.has_value() && exact->requested_span == 1 &&
            exact->mosaic.start_chunk_row == 1 &&
            exact->mosaic.start_chunk_column == 2,
        "exact chunk-edge window does not include adjacent chunks",
        failures);

    const auto crosses =
        select_resident_view(
            kRaster,
            RasterWindow{
                std::nextafter(512.0, 0.0),
                1024.0,
                std::nextafter(512.0, 1024.0),
                1536.0});
    expect(
        crosses.has_value() && crosses->requested_span == 2 &&
            crosses->mosaic.start_chunk_row == 0 &&
            crosses->mosaic.start_chunk_column == 1,
        "sub-pixel crossings on both sides select both touched row chunks",
        failures);

    const auto four =
        select_resident_view(
            kRaster, RasterWindow{600.0, 600.0, 1600.0, 1600.0});
    expect(
        four.has_value() && four->requested_span == 4 &&
            four->mosaic.start_chunk_row == 0 &&
            four->mosaic.start_chunk_column == 0,
        "three touched chunks promote directly to the 4x resident span",
        failures);

    const auto too_wide =
        select_resident_view(
            kRaster, RasterWindow{0.0, 0.0, 100.0, 2560.1});
    expect(
        !too_wide.has_value(),
        "window touching five chunks falls back to overview",
        failures);
}

void test_canonical_centering_and_edges(int& failures) {
    const auto two =
        select_resident_view(
            kRaster, RasterWindow{10.0, 600.0, 100.0, 1030.0});
    expect(
        two.has_value() && two->requested_span == 2 &&
            two->mosaic.start_chunk_column == 1 &&
            two->focus_chunk_column == 2,
        "two-chunk range remains fully contained despite an asymmetric window",
        failures);

    const auto vertical_forces_four =
        select_resident_view(
            kRaster, RasterWindow{100.0, 2100.0, 1600.0, 2200.0});
    expect(
        vertical_forces_four.has_value() &&
            vertical_forces_four->requested_span == 4 &&
            vertical_forces_four->mosaic.start_chunk_row == 0 &&
            vertical_forces_four->mosaic.start_chunk_column == 2 &&
            vertical_forces_four->focus_chunk_column == 4,
        "spare resident columns are centered canonically around visibility",
        failures);

    const auto bottom_right = select_resident_view(
        kRaster,
        RasterWindow{
            static_cast<double>(kRaster.raster_height) - 1.0,
            static_cast<double>(kRaster.raster_width) - 1.0,
            static_cast<double>(kRaster.raster_height),
            static_cast<double>(kRaster.raster_width)});
    expect(
        bottom_right.has_value() &&
            bottom_right->requested_span == 1 &&
            bottom_right->mosaic ==
                MosaicGeometry{
                    .start_chunk_row = 4,
                    .start_chunk_column = 6,
                    .chunk_rows = 1,
                    .chunk_columns = 1,
                    .pixel_row = 2048,
                    .pixel_column = 3072,
                    .pixel_height = 123,
                    .pixel_width = 77,
                },
        "bottom-right raster edge maps to the two partial physical chunks",
        failures);

    constexpr ChunkedRasterGeometry tiny{
        .raster_height = 3,
        .raster_width = 5,
        .total_chunk_rows = 1,
        .total_chunk_columns = 1,
        .chunk_height = 512,
        .chunk_width = 512,
    };
    const auto tiny_selection =
        select_resident_view(tiny, RasterWindow{0.0, 0.0, 3.0, 5.0});
    expect(
        tiny_selection.has_value() &&
            tiny_selection->requested_span == 1 &&
            tiny_selection->focus_chunk_row == 0 &&
            tiny_selection->focus_chunk_column == 0 &&
            tiny_selection->mosaic.pixel_height == 3 &&
            tiny_selection->mosaic.pixel_width == 5,
        "tiny raster selects its single partial physical chunk",
        failures);

    constexpr ChunkedRasterGeometry short_axis{
        .raster_height = 300,
        .raster_width = 3ULL * 512,
        .total_chunk_rows = 1,
        .total_chunk_columns = 3,
        .chunk_height = 512,
        .chunk_width = 512,
    };
    const auto short_axis_selection = select_resident_view(
        short_axis, RasterWindow{0.0, 500.0, 300.0, 1030.0});
    expect(
        short_axis_selection.has_value() &&
            short_axis_selection->requested_span == 4 &&
            short_axis_selection->mosaic.chunk_rows == 1 &&
            short_axis_selection->mosaic.chunk_columns == 3 &&
            short_axis_selection->mosaic.pixel_height == 300 &&
            short_axis_selection->mosaic.pixel_width == 1536,
        "requested span larger than a short grid uses available chunks once",
        failures);
}

void test_minimum_guard_span(int& failures) {
    const auto guarded = select_resident_view(
        kRaster,
        RasterWindow{600.0, 1100.0, 700.0, 1200.0},
        4);
    expect(
        guarded.has_value() && guarded->requested_span == 4 &&
            guarded->focus_chunk_row == 2 &&
            guarded->focus_chunk_column == 2 &&
            guarded->mosaic ==
                MosaicGeometry{
                    .start_chunk_row = 0,
                    .start_chunk_column = 0,
                    .chunk_rows = 4,
                    .chunk_columns = 4,
                    .pixel_row = 0,
                    .pixel_column = 0,
                    .pixel_height = 2048,
                    .pixel_width = 2048,
                },
        "minimum span promotes a one-chunk view to a centered 4x guard band",
        failures);

    const auto minimum_two = select_resident_view(
        kRaster,
        RasterWindow{600.0, 1100.0, 700.0, 1200.0},
        2);
    expect(
        minimum_two.has_value() && minimum_two->requested_span == 2 &&
            minimum_two->mosaic.start_chunk_row == 0 &&
            minimum_two->mosaic.start_chunk_column == 1,
        "supported 2x minimum remains available to non-camera callers",
        failures);

    const auto bottom_right = select_resident_view(
        kRaster,
        RasterWindow{
            static_cast<double>(kRaster.raster_height) - 1.0,
            static_cast<double>(kRaster.raster_width) - 1.0,
            static_cast<double>(kRaster.raster_height),
            static_cast<double>(kRaster.raster_width)},
        4);
    expect(
        bottom_right.has_value() && bottom_right->requested_span == 4 &&
            bottom_right->mosaic.start_chunk_row == 1 &&
            bottom_right->mosaic.start_chunk_column == 3 &&
            bottom_right->mosaic.chunk_rows == 4 &&
            bottom_right->mosaic.chunk_columns == 4 &&
            bottom_right->mosaic.pixel_height == 1659 &&
            bottom_right->mosaic.pixel_width == 1613,
        "4x guard band clamps safely against partial bottom-right edges",
        failures);

    constexpr ChunkedRasterGeometry short_grid{
        .raster_height = 300,
        .raster_width = 3ULL * 512,
        .total_chunk_rows = 1,
        .total_chunk_columns = 3,
        .chunk_height = 512,
        .chunk_width = 512,
    };
    const auto guarded_short = select_resident_view(
        short_grid,
        RasterWindow{0.0, 600.0, 300.0, 700.0},
        4);
    expect(
        guarded_short.has_value() &&
            guarded_short->requested_span == 4 &&
            guarded_short->focus_chunk_row == 0 &&
            guarded_short->focus_chunk_column == 2 &&
            guarded_short->mosaic.chunk_rows == 1 &&
            guarded_short->mosaic.chunk_columns == 3 &&
            guarded_short->mosaic.pixel_height == 300 &&
            guarded_short->mosaic.pixel_width == 1536,
        "4x minimum uses every available chunk once on a short grid",
        failures);

    const auto too_wide = select_resident_view(
        kRaster,
        RasterWindow{0.0, 0.0, 100.0, 2560.1},
        4);
    expect(
        !too_wide.has_value(),
        "minimum span never forces native residency past four visible chunks",
        failures);

    expect_throws<std::invalid_argument>(
        [] {
            static_cast<void>(select_resident_view(
                kRaster,
                RasterWindow{0.0, 0.0, 1.0, 1.0},
                0));
        },
        "zero is not a supported minimum resident span",
        failures);
    expect_throws<std::invalid_argument>(
        [] {
            static_cast<void>(select_resident_view(
                kRaster,
                RasterWindow{0.0, 0.0, 1.0, 1.0},
                3));
        },
        "noncanonical minimum resident span is rejected",
        failures);
}
void test_validation(int& failures) {
    expect_throws<std::invalid_argument>(
        [] {
            static_cast<void>(select_resident_view(
                ChunkedRasterGeometry{},
                RasterWindow{0.0, 0.0, 1.0, 1.0}));
        },
        "zero geometry is rejected",
        failures);
    expect_throws<std::invalid_argument>(
        [] {
            auto mismatch = kRaster;
            ++mismatch.total_chunk_rows;
            static_cast<void>(select_resident_view(
                mismatch, RasterWindow{0.0, 0.0, 1.0, 1.0}));
        },
        "chunk-grid mismatch is rejected",
        failures);
    expect_throws<std::invalid_argument>(
        [] {
            static_cast<void>(select_resident_view(
                kRaster,
                RasterWindow{
                    0.0,
                    0.0,
                    std::numeric_limits<double>::infinity(),
                    1.0}));
        },
        "infinite coordinate is rejected",
        failures);
    expect_throws<std::invalid_argument>(
        [] {
            static_cast<void>(select_resident_view(
                kRaster,
                RasterWindow{
                    std::numeric_limits<double>::quiet_NaN(),
                    0.0,
                    1.0,
                    1.0}));
        },
        "NaN coordinate is rejected",
        failures);
    expect_throws<std::invalid_argument>(
        [] {
            static_cast<void>(select_resident_view(
                kRaster, RasterWindow{1.0, 0.0, 1.0, 1.0}));
        },
        "empty window is rejected",
        failures);
    expect_throws<std::invalid_argument>(
        [] {
            static_cast<void>(select_resident_view(
                kRaster, RasterWindow{2.0, 0.0, 1.0, 1.0}));
        },
        "inverted window is rejected",
        failures);
    expect_throws<std::out_of_range>(
        [] {
            static_cast<void>(select_resident_view(
                kRaster, RasterWindow{-1.0, 0.0, 1.0, 1.0}));
        },
        "negative coordinate is rejected",
        failures);
    expect_throws<std::out_of_range>(
        [] {
            static_cast<void>(select_resident_view(
                kRaster,
                RasterWindow{
                    0.0,
                    0.0,
                    static_cast<double>(kRaster.raster_height) + 1.0,
                    1.0}));
        },
        "window beyond raster edge is rejected",
        failures);
}

}  // namespace

int run_resident_view_tests() {
    int failures = 0;
    test_smallest_span_and_boundaries(failures);
    test_canonical_centering_and_edges(failures);
    test_minimum_guard_span(failures);
    test_validation(failures);
    return failures;
}
