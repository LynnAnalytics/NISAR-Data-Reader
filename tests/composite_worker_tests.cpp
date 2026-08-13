#include "satview/analysis_catalog.hpp"
#include "satview/composite_scientific.hpp"
#include "satview/composite_worker.hpp"
#include "satview/cpu/scientific.hpp"
#include "satview/hdf5_product.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::string_view kHhhh =
    "/science/LSAR/GCOV/grids/frequencyA/HHHH";
constexpr std::string_view kHvhv =
    "/science/LSAR/GCOV/grids/frequencyA/HVHV";

void expect(
    const bool condition,
    const std::string_view description,
    int& failures) {
    if (!condition) {
        ++failures;
        std::cerr << "Composite-worker test failed: " << description << '\n';
    }
}

template <class Operation>
void expect_rejected(
    Operation&& operation,
    const std::string_view description,
    int& failures) {
    bool rejected = false;
    try {
        std::forward<Operation>(operation)();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, description, failures);
}

[[nodiscard]] bool same_bits(
    const std::vector<float>& left,
    const std::vector<float>& right) {
    return left.size() == right.size() &&
        std::memcmp(
            left.data(), right.data(), left.size() * sizeof(float)) == 0;
}

[[nodiscard]] satview::composite::PageCompletion build(
    const satview::Hdf5Product& product,
    const satview::composite::PageRequest& request) {
    const auto plan = satview::composite::make_page_plan(product, request);
    return satview::composite::build_page(product, plan);
}

}  // namespace

int run_composite_worker_tests(const satview::Hdf5Product& product) {
    using satview::analysis::CompareMode;
    using satview::composite::PageKind;
    using satview::composite::PageRequest;
    using satview::composite::SourceTransform;

    int failures = 0;
    const auto pauli_capabilities =
        satview::analysis::resolve_pauli_capabilities(product);
    const auto pauli_capability = std::find_if(
        pauli_capabilities.begin(),
        pauli_capabilities.end(),
        [](const satview::analysis::PauliCapability& capability) {
            return capability.frequency == "A";
        });
    expect(
        pauli_capability != pauli_capabilities.end() &&
            pauli_capability->available &&
            pauli_capability->recipe.has_value(),
        "the bundled QPDH frequency A resolves a canonical Pauli recipe",
        failures);
    if (pauli_capability == pauli_capabilities.end() ||
        !pauli_capability->available ||
        !pauli_capability->recipe.has_value()) {
        return failures;
    }

    constexpr satview::Window2D window{
        .row = 8946,
        .column = 9054,
        .height = 2,
        .width = 3,
    };
    PageRequest pauli_request{
        .serial = 1,
        .kind = PageKind::pauli_rgb,
        .pauli = *pauli_capability->recipe,
        .transform = SourceTransform::power_db,
        .source_window = window,
        .sample_stride = {1, 1},
        .maximum_output_bytes = 1024ULL * 1024ULL,
        .maximum_scratch_bytes = 64ULL * 1024ULL * 1024ULL,
        .prefer_direct_chunk_decode = true,
    };
    auto reuse_equivalent = pauli_request;
    reuse_equivalent.serial = 99;
    reuse_equivalent.camera_generation = 42;
    expect(
        satview::composite::same_page_source(
            pauli_request, reuse_equivalent),
        "page reuse identity ignores only the scheduling serial",
        failures);
    reuse_equivalent.sample_stride = {2, 1};
    expect(
        !satview::composite::same_page_source(
            pauli_request, reuse_equivalent),
        "page reuse identity includes the sampling lattice",
        failures);
    const auto pauli_plan =
        satview::composite::make_page_plan(product, pauli_request);
    expect(
        pauli_plan.aligned_plan.layout.output_dimensions ==
            std::array<std::uint64_t, 2>{2, 3},
        "Pauli planning preserves the exact requested sample lattice",
        failures);
    expect(
        pauli_plan.aligned_plan.rasters.size() == 4 &&
            pauli_plan.aligned_plan.rasters[0].dataset_path ==
                pauli_request.pauli->hhhh_dataset_path &&
            pauli_plan.aligned_plan.rasters[1].dataset_path ==
                pauli_request.pauli->hvhv_dataset_path &&
            pauli_plan.aligned_plan.rasters[2].dataset_path ==
                pauli_request.pauli->vvvv_dataset_path &&
            pauli_plan.aligned_plan.rasters[3].dataset_path ==
                pauli_request.pauli->hhvv_dataset_path,
        "Pauli planning binds the four covariance terms in formula order",
        failures);

    const auto first_pauli =
        satview::composite::build_page(product, pauli_plan);
    const auto second_pauli =
        satview::composite::build_page(product, pauli_plan);
    expect(
        first_pauli.ready() && first_pauli.values.size() == 6,
        "Pauli build publishes one packed value per requested pixel",
        failures);
    expect(
        second_pauli.ready() &&
            same_bits(first_pauli.values, second_pauli.values),
        "repeated Pauli builds are bit-deterministic",
        failures);
    const bool any_valid_pauli = std::any_of(
        first_pauli.values.begin(),
        first_pauli.values.end(),
        [](const float packed) {
            return satview::composite::unpack_pauli_rgb_r32(packed).valid;
        });
    expect(
        any_valid_pauli,
        "the center QPDH window contains a valid Pauli display sample",
        failures);

    auto reordered = pauli_request;
    reordered.serial = 2;
    std::swap(
        reordered.pauli->hhhh_dataset_path,
        reordered.pauli->hvhv_dataset_path);
    expect_rejected(
        [&] {
            static_cast<void>(
                satview::composite::make_page_plan(product, reordered));
        },
        "a reordered Pauli recipe is rejected before any raster read",
        failures);

    auto malformed = pauli_request;
    malformed.serial = 3;
    malformed.pauli->frequency = "B";
    expect_rejected(
        [&] {
            static_cast<void>(
                satview::composite::make_page_plan(product, malformed));
        },
        "a malformed Pauli recipe is rejected before any raster read",
        failures);

    const auto compare_capability =
        satview::analysis::resolve_compare_capability(product, kHhhh, kHvhv);
    expect(
        compare_capability.supports(CompareMode::swipe) &&
            compare_capability.supports(CompareMode::difference) &&
            compare_capability.supports(CompareMode::ratio) &&
            compare_capability.recipe.has_value(),
        "the bundled aligned HHHH/HVHV pair supports every worker mode",
        failures);
    if (!compare_capability.recipe.has_value()) {
        return failures;
    }

    PageRequest compare_request{
        .serial = 4,
        .kind = PageKind::compare_pair,
        .compare = *compare_capability.recipe,
        .transform = SourceTransform::linear,
        .source_window = window,
        .sample_stride = {1, 1},
        .maximum_output_bytes = 1024ULL * 1024ULL,
        .maximum_scratch_bytes = 64ULL * 1024ULL * 1024ULL,
        .prefer_direct_chunk_decode = true,
    };
    auto invalid_real_transform = compare_request;
    invalid_real_transform.serial = 40;
    invalid_real_transform.transform = SourceTransform::magnitude;
    expect_rejected(
        [&] {
            static_cast<void>(satview::composite::make_page_plan(
                product, invalid_real_transform));
        },
        "real covariance comparison rejects a complex-only transform",
        failures);
    const auto pair = build(product, compare_request);
    compare_request.serial = 5;
    compare_request.kind = PageKind::compare_difference;
    const auto difference = build(product, compare_request);
    compare_request.serial = 6;
    compare_request.kind = PageKind::compare_ratio_db;
    const auto ratio = build(product, compare_request);
    expect(
        pair.ready() && difference.ready() && ratio.ready() &&
            pair.values.size() == 6 && difference.values.size() == 6 &&
            ratio.values.size() == 6,
        "pair, difference, and ratio builds publish complete pages",
        failures);

    bool any_valid_pair = false;
    const auto sample_count = std::min({
        pair.values.size(), difference.values.size(), ratio.values.size()});
    for (std::size_t index = 0; index < sample_count; ++index) {
        const auto unpacked =
            satview::composite::unpack_compare_pair_r32(pair.values[index]);
        const bool arithmetic_valid =
            unpacked.first_valid && unpacked.second_valid;
        any_valid_pair = any_valid_pair || arithmetic_valid;
        if (!arithmetic_valid) {
            expect(
                std::isnan(difference.values[index]) &&
                    std::isnan(ratio.values[index]),
                "invalid pair samples remain invalid in arithmetic pages",
                failures);
            continue;
        }

        const float expected_difference =
            unpacked.first - unpacked.second;
        const float difference_tolerance = 0.01F * std::max(
            {1.0F, std::abs(unpacked.first), std::abs(unpacked.second)});
        expect(
            std::abs(difference.values[index] - expected_difference) <=
                difference_tolerance,
            "difference agrees with the packed source pair within bfloat precision",
            failures);

        if (unpacked.first >= 0.0F &&
            unpacked.second > satview::cpu::kTransformEpsilon) {
            const float expected_ratio_db = 10.0F * std::log10(std::max(
                unpacked.first / unpacked.second,
                satview::cpu::kTransformEpsilon));
            expect(
                std::abs(ratio.values[index] - expected_ratio_db) <= 0.05F,
                "ratio dB agrees with the packed source pair within bfloat precision",
                failures);
        } else {
            expect(
                std::isnan(ratio.values[index]),
                "non-positive denominators remain invalid for ratio",
                failures);
        }
    }
    expect(
        any_valid_pair,
        "the center QPDH window contains a valid comparison sample",
        failures);

    const auto zero_ratio = satview::composite::compare_ratio(
        0.0F,
        true,
        1.0F,
        true,
        satview::cpu::kTransformEpsilon);
    const float zero_ratio_db = zero_ratio.valid
        ? 10.0F * std::log10(std::max(
              zero_ratio.value, satview::cpu::kTransformEpsilon))
        : std::numeric_limits<float>::quiet_NaN();
    expect(
        zero_ratio.valid && zero_ratio.value == 0.0F &&
            std::isfinite(zero_ratio_db) &&
            zero_ratio_db ==
                10.0F * std::log10(satview::cpu::kTransformEpsilon),
        "a valid zero numerator maps to the finite configured dB floor",
        failures);

    bool cancel_after_block = false;
    std::uint64_t progress_calls = 0;
    const auto cancelled = satview::composite::build_page(
        product,
        pauli_plan,
        [&] { return cancel_after_block; },
        [&](const satview::aligned::AlignedProgress&) {
            ++progress_calls;
            cancel_after_block = true;
        });
    expect(
        progress_calls == 1 && !cancelled.ready() &&
            !cancelled.plan.has_value() && cancelled.values.empty(),
        "cancellation after completed internal work publishes no partial page",
        failures);

    if (failures == 0) {
        std::cout << "Composite worker tests passed\n";
    }
    return failures;
}
