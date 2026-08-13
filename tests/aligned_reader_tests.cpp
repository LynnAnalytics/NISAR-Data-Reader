#include "satview/aligned_reader.hpp"
#include "satview/hdf5_product.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kHhhh =
    "/science/LSAR/GCOV/grids/frequencyA/HHHH";
constexpr std::string_view kHvhv =
    "/science/LSAR/GCOV/grids/frequencyA/HVHV";
constexpr std::string_view kMask =
    "/science/LSAR/GCOV/grids/frequencyA/mask";
constexpr std::string_view kFrequencyBHhhh =
    "/science/LSAR/GCOV/grids/frequencyB/HHHH";

void expect(
    const bool condition,
    const std::string_view message,
    int& failures) {
    if (!condition) {
        ++failures;
        std::cerr << "Aligned-reader test failed: " << message << '\n';
    }
}

template <class Operation>
void expect_rejected(
    Operation&& operation,
    const std::string_view message,
    int& failures) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::exception&) {
        rejected = true;
    }
    expect(rejected, message, failures);
}

[[nodiscard]] std::span<const std::byte> source_element(
    const std::vector<std::byte>& bytes,
    const satview::ReadPlan& plan,
    const std::uint64_t row,
    const std::uint64_t column) {
    const auto element =
        (row - plan.aligned.row) * plan.aligned.width +
        column - plan.aligned.column;
    const auto offset =
        static_cast<std::size_t>(element) * plan.data_type.element_size;
    return {
        bytes.data() + offset,
        plan.data_type.element_size};
}

void copy_callback_member(
    const satview::aligned::AlignedBlock& block,
    const std::span<const std::byte> source,
    const std::size_t element_size,
    const std::uint64_t output_columns,
    std::vector<std::byte>& destination) {
    const auto source_row_bytes =
        static_cast<std::size_t>(block.dimensions[1]) * element_size;
    for (std::uint64_t row = 0; row < block.dimensions[0]; ++row) {
        const auto destination_element =
            (block.output_origin[0] + row) * output_columns +
            block.output_origin[1];
        std::memcpy(
            destination.data() +
                static_cast<std::size_t>(destination_element) *
                    element_size,
            source.data() +
                static_cast<std::size_t>(row) * source_row_bytes,
            source_row_bytes);
    }
}

}  // namespace

int run_aligned_reader_tests(const satview::Hdf5Product& product) {
    using namespace satview::aligned;

    int failures = 0;
    const auto* const hhhh = product.find_dataset(kHhhh);
    const auto* const hvhv = product.find_dataset(kHvhv);
    const auto* const mask = product.find_dataset(kMask);
    expect(hhhh != nullptr && hvhv != nullptr && mask != nullptr,
           "required quad-pol GCOV layers are cataloged", failures);
    if (hhhh == nullptr || hvhv == nullptr || mask == nullptr) {
        return failures;
    }

    constexpr satview::Window2D window{
        .row = 510,
        .column = 509,
        .height = 7,
        .width = 8,
    };
    AlignedRequest request{
        .rasters = {
            {.dataset_path = std::string(kHhhh),
             .mask_dataset_path = std::string(kMask)},
            {.dataset_path = std::string(kHvhv),
             .mask_dataset_path = std::string(kMask)},
        },
        .source_window = window,
        .sample_stride = {2, 3},
        .maximum_output_bytes = 1024ULL * 1024ULL,
        .maximum_scratch_bytes = 32ULL * 1024ULL * 1024ULL,
        .prefer_direct_chunk_decode = true,
    };
    const auto plan = make_aligned_plan(product, request);
    expect(plan.layout.output_dimensions ==
               std::array<std::uint64_t, 2>{4, 3},
           "regional lattice uses exact request-anchored ceil dimensions",
           failures);
    expect(plan.layout.source_block_count == 4,
           "boundary-crossing region uses four driver chunks", failures);
    expect(plan.rasters.size() == 2 && plan.masks.size() == 1,
           "shared masks are deduplicated", failures);
    expect(plan.rasters[0].mask_index == std::optional<std::size_t>(0) &&
               plan.rasters[1].mask_index ==
                   std::optional<std::size_t>(0),
           "both science rasters map to the shared mask", failures);
    expect(plan.layout.materialized_bytes == 4 * 3 * (4 + 4 + 1),
           "materialized byte accounting counts unique masks once", failures);
    expect(plan.layout.scratch_bytes <= request.maximum_scratch_bytes,
           "explicit block staging honors its byte budget", failures);

    const auto output_samples = static_cast<std::size_t>(
        plan.layout.output_dimensions[0] *
        plan.layout.output_dimensions[1]);
    std::vector<std::byte> first(output_samples * sizeof(float));
    std::vector<std::byte> second(output_samples * sizeof(float));
    std::vector<std::byte> validity(output_samples);
    std::vector<unsigned int> writes(output_samples, 0);
    std::uint64_t callbacks = 0;
    const auto status = visit_aligned_blocks(
        product,
        plan,
        [&](const AlignedBlock& block) {
            ++callbacks;
            expect(block.rasters.size() == 2 && block.masks.size() == 1,
                   "complete callback contains every planned member",
                   failures);
            expect(block.rasters[0].member_index == 0 &&
                       block.rasters[1].member_index == 1 &&
                       block.masks[0].member_index == 0,
                   "callback member order matches the plan", failures);
            copy_callback_member(
                block, block.rasters[0].samples, sizeof(float),
                plan.layout.output_dimensions[1], first);
            copy_callback_member(
                block, block.rasters[1].samples, sizeof(float),
                plan.layout.output_dimensions[1], second);
            copy_callback_member(
                block, block.masks[0].samples, sizeof(std::uint8_t),
                plan.layout.output_dimensions[1], validity);
            for (std::uint64_t row = 0; row < block.dimensions[0]; ++row) {
                for (std::uint64_t column = 0;
                     column < block.dimensions[1];
                     ++column) {
                    const auto output =
                        (block.output_origin[0] + row) *
                            plan.layout.output_dimensions[1] +
                        block.output_origin[1] + column;
                    ++writes[static_cast<std::size_t>(output)];
                }
            }
            return true;
        });
    expect(status == VisitStatus::completed,
           "complete traversal reports completed", failures);
    expect(callbacks == 4,
           "each sampled driver chunk publishes one complete block", failures);
    for (const auto writes_for_sample : writes) {
        expect(writes_for_sample == 1,
               "every canonical output sample is published exactly once",
               failures);
    }

    const auto hhhh_plan = product.make_read_plan(
        kHhhh, window.row, window.column, window.height, window.width);
    const auto hvhv_plan = product.make_read_plan(
        kHvhv, window.row, window.column, window.height, window.width);
    const auto mask_plan = product.make_read_plan(
        kMask, window.row, window.column, window.height, window.width);
    std::vector<std::byte> hhhh_source(hhhh_plan.expected_bytes);
    std::vector<std::byte> hvhv_source(hvhv_plan.expected_bytes);
    std::vector<std::byte> mask_source(mask_plan.expected_bytes);
    product.read_into(hhhh_plan, hhhh_source);
    product.read_into(hvhv_plan, hvhv_source);
    product.read_into(mask_plan, mask_source);
    for (std::uint64_t output_row = 0;
         output_row < plan.layout.output_dimensions[0];
         ++output_row) {
        for (std::uint64_t output_column = 0;
             output_column < plan.layout.output_dimensions[1];
             ++output_column) {
            const auto row =
                window.row + output_row * request.sample_stride[0];
            const auto column =
                window.column + output_column * request.sample_stride[1];
            const auto destination = static_cast<std::size_t>(
                output_row * plan.layout.output_dimensions[1] +
                output_column);
            expect(std::memcmp(
                       first.data() + destination * sizeof(float),
                       source_element(hhhh_source, hhhh_plan, row, column)
                           .data(),
                       sizeof(float)) == 0,
                   "first raster matches an independent exact HDF5 read",
                   failures);
            expect(std::memcmp(
                       second.data() + destination * sizeof(float),
                       source_element(hvhv_source, hvhv_plan, row, column)
                           .data(),
                       sizeof(float)) == 0,
                   "second raster matches an independent exact HDF5 read",
                   failures);
            expect(std::memcmp(
                       validity.data() + destination,
                       source_element(mask_source, mask_plan, row, column)
                           .data(),
                       sizeof(std::uint8_t)) == 0,
                   "deduplicated mask matches an independent HDF5 read",
                   failures);
        }
    }

    std::uint64_t stopped_callbacks = 0;
    const auto stopped = visit_aligned_blocks(
        product,
        plan,
        [&](const AlignedBlock&) {
            ++stopped_callbacks;
            return false;
        });
    expect(stopped == VisitStatus::visitor_stopped &&
               stopped_callbacks == 1,
           "visitor stop occurs only after one complete block", failures);

    std::uint64_t cancelled_callbacks = 0;
    const auto cancelled = visit_aligned_blocks(
        product,
        plan,
        [&](const AlignedBlock&) {
            ++cancelled_callbacks;
            return true;
        },
        [] { return true; });
    expect(cancelled == VisitStatus::cancelled &&
               cancelled_callbacks == 0,
           "pre-cancellation publishes no partial block", failures);

    auto too_many = request;
    too_many.rasters.resize(5, too_many.rasters.front());
    expect_rejected(
        [&] { static_cast<void>(make_aligned_plan(product, too_many)); },
        "more than four science rasters are rejected", failures);

    auto too_much_output = request;
    too_much_output.maximum_output_bytes =
        plan.layout.materialized_bytes - 1;
    expect_rejected(
        [&] {
            static_cast<void>(
                make_aligned_plan(product, too_much_output));
        },
        "complete-lattice materialization budget is enforced", failures);

    auto too_much_scratch = request;
    too_much_scratch.maximum_scratch_bytes = 1;
    expect_rejected(
        [&] {
            static_cast<void>(
                make_aligned_plan(product, too_much_scratch));
        },
        "explicit per-block scratch budget is enforced", failures);

    if (product.find_dataset(kFrequencyBHhhh) != nullptr) {
        auto cross_frequency = request;
        cross_frequency.rasters[1].dataset_path =
            std::string(kFrequencyBHhhh);
        cross_frequency.rasters[1].mask_dataset_path.reset();
        expect_rejected(
            [&] {
                static_cast<void>(
                    make_aligned_plan(product, cross_frequency));
            },
            "cross-frequency comparison without resampling is rejected",
            failures);
    }

    auto stale = plan;
    ++stale.layout.output_dimensions[0];
    std::uint64_t stale_callbacks = 0;
    expect_rejected(
        [&] {
            static_cast<void>(visit_aligned_blocks(
                product,
                stale,
                [&](const AlignedBlock&) {
                    ++stale_callbacks;
                    return true;
                }));
        },
        "tampered plan is rejected by start-of-read revalidation", failures);
    expect(stale_callbacks == 0,
           "stale plan is rejected before the first callback", failures);

    return failures;
}

#if defined(SATVIEW_ALIGNED_READER_STANDALONE)
int main(const int argument_count, char** const arguments) {
    if (argument_count != 2) {
        std::cerr << "usage: aligned-reader-tests <quad-pol-gcov.h5>\n";
        return 2;
    }
    try {
        const satview::Hdf5Product product(arguments[1]);
        return run_aligned_reader_tests(product);
    } catch (const std::exception& error) {
        std::cerr << "Aligned-reader test setup failed: "
                  << error.what() << '\n';
        return 1;
    }
}
#endif
