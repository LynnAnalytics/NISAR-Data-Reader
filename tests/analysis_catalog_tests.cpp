#include "satview/analysis_catalog.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using satview::DatasetInfo;
using satview::DatasetRole;
using satview::FrequencyCatalog;
using satview::LayerKind;

void expect(
    const bool condition,
    const std::string_view description,
    int& failures) {
    if (!condition) {
        std::cerr << "Analysis catalog test failed: " << description << '\n';
        ++failures;
    }
}

[[nodiscard]] satview::DataTypeInfo float32_type() {
    return satview::DataTypeInfo{
        .kind = satview::ScalarKind::floating_point,
        .file_element_size = sizeof(float),
        .element_size = sizeof(float),
        .description = "float32",
        .readable = true,
    };
}

[[nodiscard]] satview::DataTypeInfo complex64_type() {
    return satview::DataTypeInfo{
        .kind = satview::ScalarKind::compound_complex,
        .file_element_size = 2 * sizeof(float),
        .element_size = 2 * sizeof(float),
        .complex_layout = satview::ComplexLayout{
            .component_size = sizeof(float),
            .file_real_offset = 0,
            .file_imaginary_offset = sizeof(float),
            .real_member_name = "r",
            .imaginary_member_name = "i",
        },
        .description = "complex64",
        .readable = true,
    };
}

[[nodiscard]] satview::DataTypeInfo uint8_type() {
    return satview::DataTypeInfo{
        .kind = satview::ScalarKind::unsigned_integer,
        .file_element_size = sizeof(std::uint8_t),
        .element_size = sizeof(std::uint8_t),
        .description = "uint8",
        .readable = true,
    };
}

[[nodiscard]] FrequencyCatalog frequency(
    std::string name,
    const std::array<std::uint64_t, 2> dimensions = {40, 60}) {
    FrequencyCatalog result;
    result.name = std::move(name);
    result.group_path = "/science/LSAR/GCOV/grids/frequency" + result.name;
    result.grid.x = satview::AxisInfo{
        .dataset_path = "/science/LSAR/GCOV/grids/xCoordinates",
        .count = dimensions[1],
        .first = 500000.0,
        .last = 500000.0 +
            20.0 * static_cast<double>(dimensions[1] - 1),
        .spacing = 20.0,
        .spacing_consistent_with_endpoints = true,
    };
    result.grid.y = satview::AxisInfo{
        .dataset_path = "/science/LSAR/GCOV/grids/yCoordinates",
        .count = dimensions[0],
        .first = 4000000.0,
        .last = 4000000.0 -
            20.0 * static_cast<double>(dimensions[0] - 1),
        .spacing = -20.0,
        .spacing_consistent_with_endpoints = true,
    };
    result.grid.epsg = 32616;
    result.grid.projection_dataset_path =
        "/science/LSAR/GCOV/grids/projection";
    return result;
}

[[nodiscard]] DatasetInfo covariance(
    const FrequencyCatalog& frequency,
    std::string name,
    const bool complex = false,
    const std::array<std::uint64_t, 2> dimensions = {40, 60}) {
    DatasetInfo result;
    result.path = frequency.group_path + "/" + name;
    result.name = std::move(name);
    result.frequency = frequency.name;
    result.role = DatasetRole::science;
    result.layer_kind = complex
        ? LayerKind::gcov_off_diagonal_covariance
        : LayerKind::gcov_diagonal_covariance;
    result.data_type = complex ? complex64_type() : float32_type();
    result.dimensions = dimensions;
    result.units = "1";
    result.grid_mapping = "projection";
    return result;
}

[[nodiscard]] DatasetInfo mask(const FrequencyCatalog& frequency) {
    DatasetInfo result;
    result.path = frequency.group_path + "/mask";
    result.name = "mask";
    result.frequency = frequency.name;
    result.role = DatasetRole::auxiliary;
    result.layer_kind = LayerKind::mask;
    result.data_type = uint8_type();
    result.dimensions = {
        frequency.grid.y.count, frequency.grid.x.count};
    result.units = "1";
    result.grid_mapping = "projection";
    return result;
}

[[nodiscard]] std::vector<DatasetInfo> quad_pol_layers(
    const FrequencyCatalog& source_frequency) {
    return {
        covariance(source_frequency, "HHHH"),
        covariance(source_frequency, "HHVV", true),
        covariance(source_frequency, "HVHV"),
        covariance(source_frequency, "VVVV"),
        mask(source_frequency),
    };
}

void test_pauli_resolution(int& failures) {
    auto frequency_b = frequency("B");
    auto frequency_a = frequency("A");
    std::vector<FrequencyCatalog> frequencies{
        frequency_b, frequency_a};
    auto datasets = quad_pol_layers(frequencies[1]);
    datasets.push_back(covariance(frequencies[0], "HHHH"));

    const auto capabilities =
        satview::analysis::resolve_pauli_capabilities({frequencies, datasets});
    expect(capabilities.size() == 2, "one Pauli result per frequency", failures);
    expect(
        capabilities[0].frequency == "B" &&
            capabilities[1].frequency == "A",
        "Pauli results preserve frequency catalog order",
        failures);
    expect(
        !capabilities[0].available &&
            capabilities[0].reason.find("missing HVHV") != std::string::npos,
        "incomplete frequency reports an explicit missing-term reason",
        failures);
    expect(
        capabilities[1].available && capabilities[1].recipe.has_value(),
        "complete quad-pol frequency resolves Pauli",
        failures);
    if (capabilities[1].recipe.has_value()) {
        const auto& recipe = *capabilities[1].recipe;
        expect(
            recipe.hhhh_dataset_path ==
                    frequencies[1].group_path + "/HHHH" &&
                recipe.hhvv_dataset_path ==
                    frequencies[1].group_path + "/HHVV",
            "Pauli recipe retains canonical dataset paths",
            failures);
        expect(
            recipe.shared_mask_dataset_path ==
                std::optional<std::string>(
                    frequencies[1].group_path + "/mask"),
            "Pauli recipe resolves the shared sibling mask",
            failures);
    }

    auto wrong_type = datasets;
    for (auto& dataset : wrong_type) {
        if (dataset.name == "HHVV") {
            dataset.data_type = float32_type();
            dataset.layer_kind = LayerKind::gcov_diagonal_covariance;
        }
    }
    const auto wrong_type_result =
        satview::analysis::resolve_pauli_capabilities(
            {std::span<const FrequencyCatalog>(frequencies).subspan(1),
             wrong_type});
    expect(
        wrong_type_result.size() == 1 &&
            !wrong_type_result[0].available &&
            wrong_type_result[0].reason.find("complex64") != std::string::npos,
        "Pauli rejects a non-complex HHVV term",
        failures);

    auto mismatched_grid = quad_pol_layers(frequencies[1]);
    for (auto& dataset : mismatched_grid) {
        if (dataset.name == "VVVV") {
            dataset.grid_mapping = "otherProjection";
        }
    }
    const auto grid_result =
        satview::analysis::resolve_pauli_capabilities(
            {std::span<const FrequencyCatalog>(frequencies).subspan(1),
             mismatched_grid});
    expect(
        grid_result.size() == 1 && !grid_result[0].available &&
            grid_result[0].reason.find("different grid mappings") !=
                std::string::npos,
        "Pauli rejects terms on inconsistent grid mappings",
        failures);

    auto mismatched_units = quad_pol_layers(frequencies[1]);
    for (auto& dataset : mismatched_units) {
        if (dataset.name == "VVVV") {
            dataset.units = "meters";
        }
    }
    const auto pauli_units_result =
        satview::analysis::resolve_pauli_capabilities(
            {std::span<const FrequencyCatalog>(frequencies).subspan(1),
             mismatched_units});
    expect(
        pauli_units_result.size() == 1 &&
            !pauli_units_result[0].available &&
            pauli_units_result[0].reason.find("incompatible units") !=
                std::string::npos,
        "Pauli rejects covariance terms with incompatible units",
        failures);

    auto invalid_mask = quad_pol_layers(frequencies[1]);
    for (auto& dataset : invalid_mask) {
        if (dataset.name == "mask") {
            dataset.data_type = float32_type();
        }
    }
    const auto mask_result =
        satview::analysis::resolve_pauli_capabilities(
            {std::span<const FrequencyCatalog>(frequencies).subspan(1),
             invalid_mask});
    expect(
        mask_result.size() == 1 && !mask_result[0].available &&
            mask_result[0].reason.find("mask") != std::string::npos,
        "Pauli rejects a present but unreadable sibling mask",
        failures);
}

void test_compare_resolution(int& failures) {
    std::vector<FrequencyCatalog> frequencies{
        frequency("A"), frequency("B")};
    auto datasets = quad_pol_layers(frequencies[0]);
    datasets.push_back(covariance(frequencies[1], "HHHH"));

    const auto hhhh = frequencies[0].group_path + "/HHHH";
    const auto hvhv = frequencies[0].group_path + "/HVHV";
    const auto aligned = satview::analysis::resolve_compare_capability(
        {frequencies, datasets}, hhhh, hvhv);
    expect(
        aligned.supports(satview::analysis::CompareMode::side_by_side) &&
            aligned.supports(satview::analysis::CompareMode::swipe) &&
            aligned.supports(satview::analysis::CompareMode::difference) &&
            aligned.supports(satview::analysis::CompareMode::ratio),
        "aligned covariance layers support every compare mode",
        failures);
    expect(
        aligned.recipe.has_value() && aligned.recipe->strictly_aligned &&
            aligned.recipe->shared_mask_dataset_path ==
                std::optional<std::string>(
                    frequencies[0].group_path + "/mask"),
        "aligned compare recipe resolves its shared mask",
        failures);

    const auto cross_frequency =
        satview::analysis::resolve_compare_capability(
            {frequencies, datasets},
            hhhh,
            frequencies[1].group_path + "/HHHH");
    expect(
        cross_frequency.supports(
            satview::analysis::CompareMode::side_by_side),
        "related scalar science layers allow cross-grid side-by-side",
        failures);
    expect(
        !cross_frequency.supports(satview::analysis::CompareMode::swipe) &&
            cross_frequency.reason_for(
                satview::analysis::CompareMode::swipe)
                    .find("same frequency") != std::string_view::npos,
        "cross-frequency swipe is rejected with a precise reason",
        failures);

    auto wrong_units = datasets;
    for (auto& dataset : wrong_units) {
        if (dataset.path == hvhv) {
            dataset.units = "meters";
        }
    }
    const auto units_result =
        satview::analysis::resolve_compare_capability(
            {frequencies, wrong_units}, hhhh, hvhv);
    expect(
        units_result.supports(
            satview::analysis::CompareMode::side_by_side) &&
            !units_result.supports(
                satview::analysis::CompareMode::difference) &&
            units_result.reason_for(
                satview::analysis::CompareMode::difference)
                    .find("compatible units") != std::string_view::npos,
        "unit mismatch keeps side-by-side but disables arithmetic",
        failures);

    auto wrong_dimensions = datasets;
    for (auto& dataset : wrong_dimensions) {
        if (dataset.path == hvhv) {
            dataset.dimensions = {39, 60};
        }
    }
    const auto dimensions_result =
        satview::analysis::resolve_compare_capability(
            {frequencies, wrong_dimensions}, hhhh, hvhv);
    expect(
        dimensions_result.supports(
            satview::analysis::CompareMode::side_by_side) &&
            !dimensions_result.supports(
                satview::analysis::CompareMode::swipe) &&
            dimensions_result.reason_for(
                satview::analysis::CompareMode::swipe)
                    .find("identical non-empty dimensions") !=
                std::string_view::npos,
        "dimension mismatch permits only independent presentation",
        failures);

    auto wrong_mapping = datasets;
    for (auto& dataset : wrong_mapping) {
        if (dataset.path == hvhv) {
            dataset.grid_mapping = "otherProjection";
        }
    }
    const auto mapping_result =
        satview::analysis::resolve_compare_capability(
            {frequencies, wrong_mapping}, hhhh, hvhv);
    expect(
        mapping_result.supports(
            satview::analysis::CompareMode::side_by_side) &&
            !mapping_result.supports(
                satview::analysis::CompareMode::ratio) &&
            mapping_result.reason_for(
                satview::analysis::CompareMode::ratio)
                    .find("same grid mapping") != std::string_view::npos,
        "grid-mapping mismatch disables aligned comparison",
        failures);

    const auto complex_path = frequencies[0].group_path + "/HHVV";
    const auto complex_result =
        satview::analysis::resolve_compare_capability(
            {frequencies, datasets}, hhhh, complex_path);
    expect(
        complex_result.supports(
            satview::analysis::CompareMode::side_by_side) &&
            !complex_result.supports(satview::analysis::CompareMode::swipe),
        "mixed real/complex sources are not advertised as aligned comparisons",
        failures);

    auto second_complex = covariance(frequencies[0], "HVVV", true);
    datasets.push_back(second_complex);
    const auto complex_pair_result =
        satview::analysis::resolve_compare_capability(
            {frequencies, datasets}, complex_path, second_complex.path);
    expect(
        complex_pair_result.supports(
            satview::analysis::CompareMode::swipe),
        "two canonical complex science rasters support aligned comparison",
        failures);

    auto unsupported_type = datasets;
    for (auto& dataset : unsupported_type) {
        if (dataset.path == hvhv) {
            dataset.data_type.element_size = sizeof(double);
            dataset.data_type.file_element_size = sizeof(double);
        }
    }
    const auto type_result =
        satview::analysis::resolve_compare_capability(
            {frequencies, unsupported_type}, hhhh, hvhv);
    expect(
        !type_result.supports(satview::analysis::CompareMode::side_by_side) &&
            type_result.reason_for(
                satview::analysis::CompareMode::side_by_side)
                    .find("readable numeric science raster") !=
                std::string_view::npos,
        "compare resolver rejects storage types the worker cannot transform",
        failures);

    const auto missing_result =
        satview::analysis::resolve_compare_capability(
            {frequencies, datasets}, hhhh, "/missing");
    expect(
        !missing_result.supports(
            satview::analysis::CompareMode::side_by_side) &&
            missing_result.reason_for(
                satview::analysis::CompareMode::side_by_side)
                    .find("not in the catalog") != std::string_view::npos,
        "missing compare input has an explicit path error",
        failures);
}

}  // namespace

int run_analysis_catalog_tests() {
    int failures = 0;
    test_pauli_resolution(failures);
    test_compare_resolution(failures);
    if (failures == 0) {
        std::cout << "Analysis catalog tests passed\n";
    }
    return failures;
}
