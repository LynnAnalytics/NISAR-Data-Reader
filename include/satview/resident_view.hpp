#pragma once

#include "satview/view_navigation.hpp"
#include "satview/viewer_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

namespace satview::viewer {

// Geometry needed to map a continuous raster window to the physical chunk
// grid. Raster dimensions are pixel extents; RasterWindow uses pixel-edge
// coordinates and is interpreted as [begin, end) on each axis.
struct ChunkedRasterGeometry {
    std::uint64_t raster_height = 0;
    std::uint64_t raster_width = 0;
    std::uint64_t total_chunk_rows = 0;
    std::uint64_t total_chunk_columns = 0;
    std::uint32_t chunk_height = 0;
    std::uint32_t chunk_width = 0;

    [[nodiscard]] bool operator==(const ChunkedRasterGeometry&) const =
        default;
};

// A canonical bounded native working set. requested_span is always 1, 2, or
// 4. On rasters with fewer chunks, MosaicGeometry reports the smaller actual
// row/column count. Passing focus_chunk_* and requested_span to
// make_mosaic_geometry reproduces mosaic exactly.
struct ResidentViewSelection {
    std::uint32_t requested_span = 0;
    std::uint64_t focus_chunk_row = 0;
    std::uint64_t focus_chunk_column = 0;
    MosaicGeometry mosaic;

    [[nodiscard]] bool operator==(const ResidentViewSelection&) const =
        default;
};

namespace resident_detail {

[[nodiscard]] inline std::uint64_t first_chunk_for_edge(
    const double edge,
    const std::uint32_t chunk_extent) {
    const auto quotient =
        std::floor(edge / static_cast<double>(chunk_extent));
    if (quotient < 0.0 ||
        quotient >
            static_cast<double>(std::numeric_limits<std::uint64_t>::max())) {
        throw std::out_of_range("visible raster edge cannot map to a chunk");
    }
    return static_cast<std::uint64_t>(quotient);
}

[[nodiscard]] inline std::uint64_t last_chunk_for_edge(
    const double exclusive_edge,
    const std::uint32_t chunk_extent) {
    // Moving one representable value toward -infinity implements the
    // exclusive end without an epsilon. Exact physical chunk boundaries then
    // select the chunk immediately before the boundary.
    const auto inclusive_edge = std::nextafter(
        exclusive_edge, -std::numeric_limits<double>::infinity());
    return first_chunk_for_edge(inclusive_edge, chunk_extent);
}

[[nodiscard]] constexpr std::uint32_t smallest_supported_span(
    const std::uint64_t required_rows,
    const std::uint64_t required_columns) noexcept {
    const auto required = std::max(required_rows, required_columns);
    if (required <= 1) {
        return 1;
    }
    if (required <= 2) {
        return 2;
    }
    if (required <= 4) {
        return 4;
    }
    return 0;
}

[[nodiscard]] constexpr std::uint64_t canonical_start(
    const std::uint64_t first,
    const std::uint64_t last,
    const std::uint64_t total,
    const std::uint32_t requested_span) noexcept {
    const auto actual_span =
        std::min<std::uint64_t>(requested_span, total);
    const auto required_span = last - first + 1;
    const auto spare = actual_span - required_span;
    // Bias an odd spare chunk toward the leading side. This matches the
    // existing even-span focus convention (the focus is the lower/right of
    // the two central chunks).
    const auto leading = spare / 2 + spare % 2;
    const auto preferred = first > leading ? first - leading : 0;
    return std::min(preferred, total - actual_span);
}

}  // namespace resident_detail

// Selects the smallest supported bounded native mosaic that fully contains
// visible and is at least minimum_requested_span. The optional minimum must be
// one of 1, 2, or 4. Returns nullopt when the visible window itself crosses
// more than four chunks on either axis; callers should use an overview then.
//
// Invalid geometry, non-finite coordinates, empty/inverted windows, and
// windows outside the raster are rejected rather than silently clamped.
[[nodiscard]] inline std::optional<ResidentViewSelection>
select_resident_view(
    const ChunkedRasterGeometry& geometry,
    const RasterWindow& visible,
    const std::uint32_t minimum_requested_span = 1) {
    if (minimum_requested_span != 1 &&
        minimum_requested_span != 2 &&
        minimum_requested_span != 4) {
        throw std::invalid_argument(
            "minimum resident span must be one of 1, 2, or 4");
    }
    if (geometry.raster_height == 0 || geometry.raster_width == 0 ||
        geometry.total_chunk_rows == 0 ||
        geometry.total_chunk_columns == 0 ||
        geometry.chunk_height == 0 || geometry.chunk_width == 0) {
        throw std::invalid_argument(
            "resident raster and chunk dimensions must be positive");
    }

    const auto expected_chunk_rows =
        detail::ceil_div(geometry.raster_height, geometry.chunk_height);
    const auto expected_chunk_columns =
        detail::ceil_div(geometry.raster_width, geometry.chunk_width);
    if (geometry.total_chunk_rows != expected_chunk_rows ||
        geometry.total_chunk_columns != expected_chunk_columns) {
        throw std::invalid_argument(
            "resident chunk-grid dimensions do not match the raster");
    }

    if (!std::isfinite(visible.row_begin) ||
        !std::isfinite(visible.column_begin) ||
        !std::isfinite(visible.row_end) ||
        !std::isfinite(visible.column_end)) {
        throw std::invalid_argument(
            "visible raster coordinates must be finite");
    }
    if (!(visible.row_begin < visible.row_end) ||
        !(visible.column_begin < visible.column_end)) {
        throw std::invalid_argument(
            "visible raster window must have positive area");
    }
    if (visible.row_begin < 0.0 || visible.column_begin < 0.0 ||
        visible.row_end > static_cast<double>(geometry.raster_height) ||
        visible.column_end > static_cast<double>(geometry.raster_width)) {
        throw std::out_of_range(
            "visible raster window is outside the raster");
    }

    const auto first_row = resident_detail::first_chunk_for_edge(
        visible.row_begin, geometry.chunk_height);
    const auto first_column = resident_detail::first_chunk_for_edge(
        visible.column_begin, geometry.chunk_width);
    const auto last_row = resident_detail::last_chunk_for_edge(
        visible.row_end, geometry.chunk_height);
    const auto last_column = resident_detail::last_chunk_for_edge(
        visible.column_end, geometry.chunk_width);

    if (first_row >= geometry.total_chunk_rows ||
        last_row >= geometry.total_chunk_rows ||
        first_column >= geometry.total_chunk_columns ||
        last_column >= geometry.total_chunk_columns) {
        throw std::out_of_range(
            "visible raster window cannot map to the chunk grid");
    }

    const auto required_rows = last_row - first_row + 1;
    const auto required_columns = last_column - first_column + 1;
    const auto visible_span = resident_detail::smallest_supported_span(
        required_rows, required_columns);
    if (visible_span == 0) {
        return std::nullopt;
    }
    const auto requested_span =
        std::max(visible_span, minimum_requested_span);

    const auto start_row = resident_detail::canonical_start(
        first_row, last_row, geometry.total_chunk_rows, requested_span);
    const auto start_column = resident_detail::canonical_start(
        first_column,
        last_column,
        geometry.total_chunk_columns,
        requested_span);
    const auto focus_row = std::min(
        geometry.total_chunk_rows - 1,
        start_row + static_cast<std::uint64_t>(requested_span / 2));
    const auto focus_column = std::min(
        geometry.total_chunk_columns - 1,
        start_column + static_cast<std::uint64_t>(requested_span / 2));
    const auto mosaic = make_mosaic_geometry(
        focus_row,
        focus_column,
        geometry.total_chunk_rows,
        geometry.total_chunk_columns,
        geometry.chunk_height,
        geometry.chunk_width,
        geometry.raster_height,
        geometry.raster_width,
        requested_span);

    return ResidentViewSelection{
        .requested_span = requested_span,
        .focus_chunk_row = focus_row,
        .focus_chunk_column = focus_column,
        .mosaic = mosaic,
    };
}

}  // namespace satview::viewer
