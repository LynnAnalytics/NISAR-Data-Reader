#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace satview::viewer {

struct MosaicGeometry {
    std::uint64_t start_chunk_row = 0;
    std::uint64_t start_chunk_column = 0;
    std::uint32_t chunk_rows = 0;
    std::uint32_t chunk_columns = 0;
    std::uint64_t pixel_row = 0;
    std::uint64_t pixel_column = 0;
    std::uint64_t pixel_height = 0;
    std::uint64_t pixel_width = 0;

    [[nodiscard]] bool operator==(const MosaicGeometry&) const = default;
};

[[nodiscard]] constexpr bool is_supported_mosaic_span(
    const std::uint32_t span) noexcept {
    return span == 1 || span == 2 || span == 4;
}

namespace detail {

[[nodiscard]] constexpr std::uint64_t ceil_div(
    const std::uint64_t value,
    const std::uint64_t divisor) {
    return value / divisor + static_cast<std::uint64_t>(value % divisor != 0);
}

[[nodiscard]] constexpr std::uint64_t checked_u64_product(
    const std::uint64_t left,
    const std::uint64_t right) {
    if (left != 0 &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::length_error("mosaic coordinate multiplication overflow");
    }
    return left * right;
}

[[nodiscard]] constexpr std::uint64_t centered_start(
    const std::uint64_t focus,
    const std::uint64_t total,
    const std::uint32_t requested_span,
    const std::uint32_t actual_span) noexcept {
    const std::uint64_t half = requested_span / 2;
    const std::uint64_t preferred = focus > half ? focus - half : 0;
    const std::uint64_t maximum = total - actual_span;
    return std::min(preferred, maximum);
}

}  // namespace detail

// Builds a bounded square-mosaic request around a focus chunk. Near an edge,
// the start moves inward so the requested 1x/2x/4x working set stays full when
// the dataset contains enough chunks. The final physical chunk may be partial.
[[nodiscard]] constexpr MosaicGeometry make_mosaic_geometry(
    const std::uint64_t focus_row,
    const std::uint64_t focus_column,
    const std::uint64_t total_chunk_rows,
    const std::uint64_t total_chunk_columns,
    const std::uint32_t chunk_height,
    const std::uint32_t chunk_width,
    const std::uint64_t raster_height,
    const std::uint64_t raster_width,
    const std::uint32_t span) {
    if (!is_supported_mosaic_span(span)) {
        throw std::invalid_argument("mosaic span must be 1, 2, or 4");
    }
    if (total_chunk_rows == 0 || total_chunk_columns == 0 ||
        chunk_height == 0 || chunk_width == 0 ||
        raster_height == 0 || raster_width == 0) {
        throw std::invalid_argument(
            "mosaic raster, chunk, and chunk-grid dimensions must be positive");
    }
    if (focus_row >= total_chunk_rows ||
        focus_column >= total_chunk_columns) {
        throw std::out_of_range("mosaic focus chunk is outside the grid");
    }

    const auto expected_rows =
        detail::ceil_div(raster_height, chunk_height);
    const auto expected_columns =
        detail::ceil_div(raster_width, chunk_width);
    if (total_chunk_rows != expected_rows ||
        total_chunk_columns != expected_columns) {
        throw std::invalid_argument(
            "mosaic chunk-grid dimensions do not match the raster");
    }

    const auto actual_rows = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(span, total_chunk_rows));
    const auto actual_columns = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(span, total_chunk_columns));
    const auto start_row = detail::centered_start(
        focus_row, total_chunk_rows, span, actual_rows);
    const auto start_column = detail::centered_start(
        focus_column, total_chunk_columns, span, actual_columns);
    const auto pixel_row =
        detail::checked_u64_product(start_row, chunk_height);
    const auto pixel_column =
        detail::checked_u64_product(start_column, chunk_width);
    const auto nominal_height =
        detail::checked_u64_product(actual_rows, chunk_height);
    const auto nominal_width =
        detail::checked_u64_product(actual_columns, chunk_width);

    return MosaicGeometry{
        .start_chunk_row = start_row,
        .start_chunk_column = start_column,
        .chunk_rows = actual_rows,
        .chunk_columns = actual_columns,
        .pixel_row = pixel_row,
        .pixel_column = pixel_column,
        .pixel_height =
            std::min(nominal_height, raster_height - pixel_row),
        .pixel_width =
            std::min(nominal_width, raster_width - pixel_column),
    };
}

// Computes a caller-owned mosaic allocation size without wrapping either the
// pixel count or byte count. The caller supplies the applicable safety limit.
[[nodiscard]] constexpr std::size_t checked_mosaic_bytes(
    const std::uint64_t height,
    const std::uint64_t width,
    const std::size_t element_size,
    const std::size_t limit) {
    if (height == 0 || width == 0 || element_size == 0) {
        throw std::invalid_argument(
            "mosaic allocation dimensions and element size must be positive");
    }

    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    if (height > maximum || width > maximum) {
        throw std::length_error("mosaic pixel count exceeds size_t");
    }
    const auto rows = static_cast<std::size_t>(height);
    const auto columns = static_cast<std::size_t>(width);
    if (columns > maximum / rows) {
        throw std::length_error("mosaic pixel count overflow");
    }
    const auto pixels = rows * columns;
    if (element_size > maximum / pixels) {
        throw std::length_error("mosaic byte count overflow");
    }
    const auto bytes = pixels * element_size;
    if (bytes > limit) {
        throw std::length_error("mosaic allocation exceeds its safety limit");
    }
    return bytes;
}

// Returns a conservative upper bound for the extent produced when an
// arbitrary request of `request` samples is expanded to complete chunks.
// Alignment can add partial chunks before and after the request.
[[nodiscard]] constexpr std::uint64_t maximum_chunk_aligned_extent(
    const std::uint64_t request,
    const std::uint64_t dimension,
    const std::uint64_t chunk_extent) {
    if (request == 0 || dimension == 0 || chunk_extent == 0) {
        throw std::invalid_argument(
            "aligned request, dimension, and chunk extent must be positive");
    }
    if (request >= dimension) {
        return dimension;
    }

    const auto leading_residue = chunk_extent - 1;
    const auto remaining = dimension - request;
    if (leading_residue >= remaining) {
        return dimension;
    }

    // request + (chunk - 1) is now known to be smaller than dimension, so the
    // addition cannot overflow. Rounding that sum up to the chunk extent is
    // the largest aligned window over all possible request-start residues.
    const auto shifted = request + leading_residue;
    const auto chunks = detail::ceil_div(shifted, chunk_extent);
    if (chunks > dimension / chunk_extent) {
        return dimension;
    }
    return std::min(dimension, chunks * chunk_extent);
}

}  // namespace satview::viewer
