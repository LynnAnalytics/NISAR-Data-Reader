#include "satview/hdf5_product.hpp"
#include "satview/viewer_math.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>

namespace {

void expect(
    const bool condition,
    const std::string_view description,
    int& failures) {
    if (!condition) {
        std::cerr << "Real mosaic-plan test failed: " << description << '\n';
        ++failures;
    }
}

}  // namespace

int run_real_mosaic_plan_tests(const satview::Hdf5Product& product) {
    using satview::viewer::checked_mosaic_bytes;
    using satview::viewer::make_mosaic_geometry;

    int failures = 0;
    const auto* science = product.find_dataset(
        "/science/LSAR/GCOV/grids/frequencyB/HHHH");
    const auto* mask = product.find_dataset(
        "/science/LSAR/GCOV/grids/frequencyB/mask");
    expect(
        science != nullptr && mask != nullptr,
        "GCOV B science and exact validity-mask layers exist",
        failures);
    if (science == nullptr || mask == nullptr ||
        !science->chunk_dimensions.has_value()) {
        return failures;
    }

    const auto chunk_height =
        static_cast<std::uint32_t>((*science->chunk_dimensions)[0]);
    const auto chunk_width =
        static_cast<std::uint32_t>((*science->chunk_dimensions)[1]);
    const auto total_rows =
        science->dimensions[0] / chunk_height +
        static_cast<std::uint64_t>(
            science->dimensions[0] % chunk_height != 0);
    const auto total_columns =
        science->dimensions[1] / chunk_width +
        static_cast<std::uint64_t>(
            science->dimensions[1] % chunk_width != 0);

    for (const std::uint32_t span : {1U, 2U, 4U}) {
        const auto geometry = make_mosaic_geometry(
            total_rows - 1,
            total_columns - 1,
            total_rows,
            total_columns,
            chunk_height,
            chunk_width,
            science->dimensions[0],
            science->dimensions[1],
            span);
        const auto plan = product.make_read_plan(
            science->path,
            geometry.pixel_row,
            geometry.pixel_column,
            geometry.pixel_height,
            geometry.pixel_width);
        expect(
            plan.aligned.row == geometry.pixel_row &&
                plan.aligned.column == geometry.pixel_column &&
                plan.aligned.height == geometry.pixel_height &&
                plan.aligned.width == geometry.pixel_width,
            "edge mosaic remains exactly chunk aligned",
            failures);
        expect(
            plan.expected_bytes == checked_mosaic_bytes(
                geometry.pixel_height,
                geometry.pixel_width,
                science->data_type.element_size,
                std::numeric_limits<std::size_t>::max()),
            "read plan and checked allocation agree",
            failures);

        const auto mask_plan = product.make_read_plan(
            mask->path,
            geometry.pixel_row,
            geometry.pixel_column,
            geometry.pixel_height,
            geometry.pixel_width);
        expect(
            mask_plan.aligned.row == plan.aligned.row &&
                mask_plan.aligned.column == plan.aligned.column &&
                mask_plan.aligned.height == plan.aligned.height &&
                mask_plan.aligned.width == plan.aligned.width,
            "exact GCOV B mask matches the science mosaic",
            failures);
    }

    return failures;
}
