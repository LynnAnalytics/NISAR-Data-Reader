#include "satview/hdf5_product.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace {

using nlohmann::json;
using namespace satview;

std::string_view to_string(ProductType value) {
    return value == ProductType::gslc ? "GSLC" : "GCOV";
}

std::string_view to_string(LayerKind value) {
    switch (value) {
        case LayerKind::gslc_polarization: return "gslc_polarization";
        case LayerKind::gcov_diagonal_covariance: return "gcov_diagonal_covariance";
        case LayerKind::gcov_off_diagonal_covariance: return "gcov_off_diagonal_covariance";
        case LayerKind::mask: return "mask";
        case LayerKind::number_of_looks: return "number_of_looks";
        case LayerKind::rtc_gamma_to_sigma_factor: return "rtc_gamma_to_sigma_factor";
        case LayerKind::calibration_lut: return "calibration_lut";
        case LayerKind::auxiliary: return "auxiliary";
    }
    return "unknown";
}

std::string_view to_string(ScalarKind value) {
    switch (value) {
        case ScalarKind::signed_integer: return "signed_integer";
        case ScalarKind::unsigned_integer: return "unsigned_integer";
        case ScalarKind::floating_point: return "floating_point";
        case ScalarKind::compound_complex: return "compound_complex";
        case ScalarKind::unsupported: return "unsupported";
    }
    return "unsupported";
}

std::string_view to_string(StorageLayout value) {
    switch (value) {
        case StorageLayout::contiguous: return "contiguous";
        case StorageLayout::chunked: return "chunked";
        case StorageLayout::compact: return "compact";
        case StorageLayout::virtual_dataset: return "virtual";
        case StorageLayout::unknown: return "unknown";
    }
    return "unknown";
}

json finite_or_symbolic_number(const double value) {
    if (std::isnan(value)) return "NaN";
    if (std::isinf(value)) return std::signbit(value) ? "-Infinity" : "Infinity";
    return value;
}

json fill_value_json(const FillValue& fill) {
    if (fill.numeric) return finite_or_symbolic_number(*fill.numeric);
    if (fill.complex) {
        return {
            {"real", finite_or_symbolic_number(fill.complex->real)},
            {"imaginary", finite_or_symbolic_number(fill.complex->imaginary)},
        };
    }
    return {{"unavailable", true}};
}

json dataset_json(const DatasetInfo& layer) {
    json filters = json::array();
    for (const auto& filter : layer.filters) {
        filters.push_back({
            {"id", filter.id},
            {"name", filter.name},
            {"flags", filter.flags},
            {"client_data", filter.client_data},
        });
    }

    json result = {
        {"path", layer.path},
        {"name", layer.name},
        {"frequency", layer.frequency},
        {"role", layer.role == DatasetRole::science ? "science" : "auxiliary"},
        {"layer_kind", to_string(layer.layer_kind)},
        {"shape", {layer.dimensions[0], layer.dimensions[1]}},
        {"dtype", {
            {"kind", to_string(layer.data_type.kind)},
            {"description", layer.data_type.description},
            {"file_element_bytes", layer.data_type.file_element_size},
            {"canonical_element_bytes", layer.data_type.element_size},
            {"readable", layer.data_type.readable},
        }},
        {"layout", to_string(layer.storage_layout)},
        {"filters", std::move(filters)},
        {"units", layer.units},
        {"long_name", layer.long_name},
    };
    if (layer.chunk_dimensions) {
        result["chunks"] = {(*layer.chunk_dimensions)[0], (*layer.chunk_dimensions)[1]};
    } else {
        result["chunks"] = nullptr;
    }
    result["creation_fill_value"] = layer.creation_fill_value
                                          ? fill_value_json(*layer.creation_fill_value)
                                          : json(nullptr);
    result["fill_value_attribute"] = layer.fill_value_attribute
                                          ? fill_value_json(*layer.fill_value_attribute)
                                          : json(nullptr);
    if (layer.fill_value_attribute) {
        // Backward-compatible alias for the HDF5 _FillValue attribute.
        result["fill_value"] = result["fill_value_attribute"];
    }
    return result;
}

json catalog_json(const Hdf5Product& product) {
    const auto& id = product.identification();
    json frequencies = json::array();
    for (const auto& frequency : product.frequencies()) {
        json layers = json::array();
        for (const auto& layer : frequency.layers) layers.push_back(dataset_json(layer));
        frequencies.push_back({
            {"name", frequency.name},
            {"group_path", frequency.group_path},
            {"polarizations", frequency.polarizations},
            {"covariance_terms", frequency.covariance_terms},
            {"grid", {
                {"rows", frequency.grid.y.count},
                {"columns", frequency.grid.x.count},
                {"x", {{"first", frequency.grid.x.first}, {"last", frequency.grid.x.last}, {"spacing", frequency.grid.x.spacing}}},
                {"y", {{"first", frequency.grid.y.first}, {"last", frequency.grid.y.last}, {"spacing", frequency.grid.y.spacing}}},
                {"epsg", frequency.grid.epsg ? json(*frequency.grid.epsg) : json(nullptr)},
            }},
            {"layers", std::move(layers)},
        });
    }
    return {
        {"file", product.file_path().string()},
        {"product", {
            {"type", to_string(id.product_type)},
            {"granule_id", id.granule_id},
            {"product_version", id.product_version},
            {"specification_version", id.product_specification_version},
            {"instrument", id.instrument_group},
        }},
        {"frequencies", std::move(frequencies)},
    };
}

void print_human(const Hdf5Product& product) {
    const auto& id = product.identification();
    std::cout << to_string(id.product_type) << "  " << id.granule_id << '\n'
              << "  file: " << product.file_path().string() << '\n'
              << "  product version: " << id.product_version
              << "  specification: " << id.product_specification_version << '\n';

    for (const auto& frequency : product.frequencies()) {
        std::cout << "\nFrequency " << frequency.name << '\n'
                  << "  grid: " << frequency.grid.y.count << " rows x "
                  << frequency.grid.x.count << " columns"
                  << "  EPSG:" << (frequency.grid.epsg ? std::to_string(*frequency.grid.epsg) : "unknown") << '\n'
                  << "  x: " << frequency.grid.x.first << " .. " << frequency.grid.x.last
                  << "  spacing " << frequency.grid.x.spacing << '\n'
                  << "  y: " << frequency.grid.y.first << " .. " << frequency.grid.y.last
                  << "  spacing " << frequency.grid.y.spacing << '\n';
        for (const auto& layer : frequency.layers) {
            std::cout << "  - " << layer.name << "  "
                      << layer.dimensions[0] << 'x' << layer.dimensions[1]
                      << "  " << layer.data_type.description;
            if (layer.chunk_dimensions) {
                std::cout << "  chunks " << (*layer.chunk_dimensions)[0]
                          << 'x' << (*layer.chunk_dimensions)[1];
            }
            if (!layer.filters.empty()) {
                std::cout << "  filters";
                for (const auto& filter : layer.filters) std::cout << ' ' << filter.name;
            }
            std::cout << "\n    " << layer.path << '\n';
        }
    }
}

void usage() {
    std::cerr << "Usage: sat-inspect [--json] <NISAR.h5>\n";
}

} // namespace

int main(int argc, char** argv) {
    bool as_json = false;
    std::optional<std::filesystem::path> file;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--json") as_json = true;
        else if (argument == "--help" || argument == "-h") { usage(); return 0; }
        else if (!argument.empty() && argument.front() == '-') { usage(); return 2; }
        else if (file) { usage(); return 2; }
        else file = std::filesystem::path(argument);
    }
    if (!file) { usage(); return 2; }

    try {
        const Hdf5Product product(*file);
        if (as_json) std::cout << catalog_json(product).dump(2) << '\n';
        else print_human(product);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sat-inspect: " << error.what() << '\n';
        return 1;
    }
}
