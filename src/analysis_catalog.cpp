#include "satview/analysis_catalog.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>
#include <utility>

namespace satview::analysis {
namespace {

[[nodiscard]] char lower_ascii(const char value) noexcept {
    return value >= 'A' && value <= 'Z'
        ? static_cast<char>(value - 'A' + 'a')
        : value;
}

[[nodiscard]] std::string trim_ascii(std::string_view value) {
    const auto whitespace = [](const char character) {
        return character == ' ' || character == '\t' ||
               character == '\r' || character == '\n';
    };
    while (!value.empty() && whitespace(value.front())) {
        value.remove_prefix(1);
    }
    while (!value.empty() && whitespace(value.back())) {
        value.remove_suffix(1);
    }
    return std::string(value);
}

[[nodiscard]] std::string normalize_token(std::string_view value) {
    std::string result = trim_ascii(value);
    std::transform(result.begin(), result.end(), result.begin(), lower_ascii);
    return result;
}

[[nodiscard]] bool same_token(
    const std::string_view left,
    const std::string_view right) {
    return normalize_token(left) == normalize_token(right);
}

[[nodiscard]] std::string join_path(
    const std::string_view parent,
    const std::string_view child) {
    std::string result(parent);
    if (result.empty() || result.back() != '/') {
        result.push_back('/');
    }
    result.append(child);
    return result;
}

[[nodiscard]] std::string parent_path(const std::string_view path) {
    const auto separator = path.rfind('/');
    return separator == std::string_view::npos
        ? std::string{}
        : std::string(path.substr(0, separator));
}

[[nodiscard]] const DatasetInfo* find_dataset(
    const CatalogView catalog,
    const std::string_view path) {
    const auto iterator = std::find_if(
        catalog.datasets.begin(),
        catalog.datasets.end(),
        [&](const DatasetInfo& dataset) {
            return dataset.path == path;
        });
    if (iterator != catalog.datasets.end()) {
        return &*iterator;
    }
    for (const auto& frequency : catalog.frequencies) {
        const auto layer = std::find_if(
            frequency.layers.begin(),
            frequency.layers.end(),
            [&](const DatasetInfo& dataset) {
                return dataset.path == path;
            });
        if (layer != frequency.layers.end()) {
            return &*layer;
        }
    }
    return nullptr;
}

[[nodiscard]] const FrequencyCatalog* find_frequency(
    const CatalogView catalog,
    const std::string_view name) {
    const auto iterator = std::find_if(
        catalog.frequencies.begin(),
        catalog.frequencies.end(),
        [&](const FrequencyCatalog& frequency) {
            return same_token(frequency.name, name);
        });
    return iterator == catalog.frequencies.end() ? nullptr : &*iterator;
}

[[nodiscard]] bool is_float32(const DatasetInfo& dataset) noexcept {
    return dataset.data_type.readable &&
           dataset.data_type.kind == ScalarKind::floating_point &&
           dataset.data_type.element_size == sizeof(float);
}

[[nodiscard]] bool is_complex64(const DatasetInfo& dataset) noexcept {
    return dataset.data_type.readable &&
           dataset.data_type.kind == ScalarKind::compound_complex &&
           dataset.data_type.element_size == 2 * sizeof(float) &&
           dataset.data_type.complex_layout.has_value() &&
           dataset.data_type.complex_layout->component_size == sizeof(float);
}

[[nodiscard]] bool is_readable_science_raster(
    const DatasetInfo& dataset) noexcept {
    // The composite worker has exact transform contracts only for the two
    // renderable NISAR storage types. Do not advertise wider numeric catalog
    // types that would become an all-invalid page at execution time.
    return is_float32(dataset) || is_complex64(dataset);
}

void append_issue(std::string& issues, std::string issue) {
    if (!issues.empty()) {
        issues.append("; ");
    }
    issues.append(std::move(issue));
}

[[nodiscard]] bool grid_dimensions_match(
    const FrequencyCatalog& frequency,
    const std::array<std::uint64_t, 2>& dimensions) noexcept {
    return frequency.grid.y.count != 0 && frequency.grid.x.count != 0 &&
           dimensions[0] == frequency.grid.y.count &&
           dimensions[1] == frequency.grid.x.count;
}

[[nodiscard]] bool grid_coordinates_are_usable(
    const FrequencyCatalog& frequency) noexcept {
    const auto usable_axis = [](const AxisInfo& axis) {
        return axis.count != 0 && std::isfinite(axis.first) &&
               std::isfinite(axis.last) && std::isfinite(axis.spacing) &&
               axis.spacing != 0.0;
    };
    return usable_axis(frequency.grid.x) && usable_axis(frequency.grid.y);
}

[[nodiscard]] bool compatible_units(
    const std::string_view left,
    const std::string_view right) {
    const auto normalized_left = normalize_token(left);
    const auto normalized_right = normalize_token(right);
    if (normalized_left == normalized_right) {
        return true;
    }
    const auto unitless = [](const std::string& value) {
        return value == "1" || value == "unitless" ||
               value == "dimensionless";
    };
    return unitless(normalized_left) && unitless(normalized_right);
}

[[nodiscard]] bool side_by_side_is_meaningful(
    const DatasetInfo& first,
    const DatasetInfo& second) {
    if (first.role == DatasetRole::science &&
        second.role == DatasetRole::science) {
        return true;
    }
    if (first.layer_kind == second.layer_kind &&
        first.layer_kind != LayerKind::auxiliary) {
        return true;
    }
    return !normalize_token(first.units).empty() &&
           compatible_units(first.units, second.units);
}

[[nodiscard]] const DatasetInfo* valid_sibling_mask(
    const CatalogView catalog,
    const DatasetInfo& source) {
    const auto parent = parent_path(source.path);
    if (parent.empty()) {
        return nullptr;
    }
    const auto* mask = find_dataset(catalog, join_path(parent, "mask"));
    if (mask == nullptr || mask->name != "mask" ||
        mask->layer_kind != LayerKind::mask ||
        mask->dimensions != source.dimensions ||
        !same_token(mask->frequency, source.frequency) ||
        !mask->data_type.readable ||
        mask->data_type.kind != ScalarKind::unsigned_integer ||
        mask->data_type.element_size != sizeof(std::uint8_t)) {
        return nullptr;
    }
    return mask;
}

[[nodiscard]] std::optional<std::string> shared_sibling_mask(
    const CatalogView catalog,
    const DatasetInfo& first,
    const DatasetInfo& second) {
    const auto* first_mask = valid_sibling_mask(catalog, first);
    const auto* second_mask = valid_sibling_mask(catalog, second);
    if (first_mask == nullptr || second_mask == nullptr ||
        first_mask->path != second_mask->path) {
        return std::nullopt;
    }
    return first_mask->path;
}

[[nodiscard]] CompareModeCapability mode_capability(
    const CompareMode mode,
    const bool available,
    const std::string& reason) {
    return CompareModeCapability{
        .mode = mode,
        .available = available,
        .reason = available ? std::string{} : reason,
    };
}

}  // namespace

CatalogView catalog_view(const Hdf5Product& product) noexcept {
    return CatalogView{
        .frequencies = product.frequencies(),
        .datasets = product.datasets(),
    };
}

std::vector<PauliCapability> resolve_pauli_capabilities(
    const CatalogView catalog) {
    std::vector<PauliCapability> result;
    result.reserve(catalog.frequencies.size());

    constexpr std::array<std::string_view, 4> terms{
        "HHHH", "HVHV", "VVVV", "HHVV"};
    for (const auto& frequency : catalog.frequencies) {
        PauliCapability capability;
        capability.frequency = frequency.name;

        std::array<const DatasetInfo*, terms.size()> sources{};
        std::string issues;
        for (std::size_t index = 0; index < terms.size(); ++index) {
            const auto path = join_path(frequency.group_path, terms[index]);
            sources[index] = find_dataset(catalog, path);
            if (sources[index] == nullptr) {
                append_issue(
                    issues,
                    "missing " + std::string(terms[index]) +
                        " covariance term");
            }
        }

        if (issues.empty()) {
            for (std::size_t index = 0; index < sources.size(); ++index) {
                const auto& source = *sources[index];
                if (source.name != terms[index]) {
                    append_issue(
                        issues,
                        std::string(terms[index]) +
                            " has inconsistent dataset metadata");
                }
                if (!same_token(source.frequency, frequency.name)) {
                    append_issue(
                        issues,
                        std::string(terms[index]) +
                            " does not belong to frequency " +
                            frequency.name);
                }
                if (source.role != DatasetRole::science) {
                    append_issue(
                        issues,
                        std::string(terms[index]) +
                            " is not a science dataset");
                }
            }

            for (std::size_t index = 0; index < 3; ++index) {
                if (sources[index]->layer_kind !=
                        LayerKind::gcov_diagonal_covariance ||
                    !is_float32(*sources[index])) {
                    append_issue(
                        issues,
                        std::string(terms[index]) +
                            " must be a readable float32 diagonal covariance");
                }
            }
            if (sources[3]->layer_kind !=
                    LayerKind::gcov_off_diagonal_covariance ||
                !is_complex64(*sources[3])) {
                append_issue(
                    issues,
                    "HHVV must be a readable complex64 cross covariance");
            }

            const auto dimensions = sources.front()->dimensions;
            if (dimensions[0] == 0 || dimensions[1] == 0) {
                append_issue(issues, "Pauli source dimensions are empty");
            }
            for (std::size_t index = 1; index < sources.size(); ++index) {
                if (sources[index]->dimensions != dimensions) {
                    append_issue(
                        issues,
                        "Pauli covariance terms have different dimensions");
                    break;
                }
            }
            const auto grid_mapping =
                trim_ascii(sources.front()->grid_mapping);
            for (std::size_t index = 1; index < sources.size(); ++index) {
                if (trim_ascii(sources[index]->grid_mapping) != grid_mapping) {
                    append_issue(
                        issues,
                        "Pauli covariance terms use different grid mappings");
                    break;
                }
            }
            for (std::size_t index = 1; index < sources.size(); ++index) {
                if (!compatible_units(
                        sources.front()->units, sources[index]->units)) {
                    append_issue(
                        issues,
                        "Pauli covariance terms use incompatible units");
                    break;
                }
            }
            if (!grid_dimensions_match(frequency, dimensions)) {
                append_issue(
                    issues,
                    "frequency grid dimensions do not match the Pauli sources");
            } else if (!grid_coordinates_are_usable(frequency)) {
                append_issue(
                    issues,
                    "frequency grid coordinate metadata is incomplete");
            }
            const auto mask_path = join_path(frequency.group_path, "mask");
            if (find_dataset(catalog, mask_path) != nullptr &&
                valid_sibling_mask(catalog, *sources.front()) == nullptr) {
                append_issue(
                    issues,
                    "Pauli sibling mask is unreadable or grid-incompatible");
            }
        }

        if (issues.empty()) {
            const auto mask = shared_sibling_mask(
                catalog, *sources[0], *sources[1]);
            const auto all_share_mask = [&]() -> std::optional<std::string> {
                if (!mask.has_value()) {
                    return std::nullopt;
                }
                for (std::size_t index = 2; index < sources.size(); ++index) {
                    const auto* candidate =
                        valid_sibling_mask(catalog, *sources[index]);
                    if (candidate == nullptr || candidate->path != *mask) {
                        return std::nullopt;
                    }
                }
                return mask;
            }();

            capability.available = true;
            capability.recipe = PauliRecipe{
                .frequency = frequency.name,
                .hhhh_dataset_path = sources[0]->path,
                .hvhv_dataset_path = sources[1]->path,
                .vvvv_dataset_path = sources[2]->path,
                .hhvv_dataset_path = sources[3]->path,
                .shared_mask_dataset_path = all_share_mask,
                .dimensions = sources[0]->dimensions,
            };
        } else {
            capability.reason = std::move(issues);
        }
        result.push_back(std::move(capability));
    }
    return result;
}

std::vector<PauliCapability> resolve_pauli_capabilities(
    const Hdf5Product& product) {
    return resolve_pauli_capabilities(catalog_view(product));
}

bool CompareCapability::supports(const CompareMode mode) const noexcept {
    const auto iterator = std::find_if(
        modes.begin(),
        modes.end(),
        [&](const CompareModeCapability& capability) {
            return capability.mode == mode;
        });
    return iterator != modes.end() && iterator->available;
}

std::string_view CompareCapability::reason_for(
    const CompareMode mode) const noexcept {
    const auto iterator = std::find_if(
        modes.begin(),
        modes.end(),
        [&](const CompareModeCapability& capability) {
            return capability.mode == mode;
        });
    return iterator == modes.end() ? std::string_view{} : iterator->reason;
}

CompareCapability resolve_compare_capability(
    const CatalogView catalog,
    const std::string_view first_dataset_path,
    const std::string_view second_dataset_path) {
    CompareCapability result;
    result.first_dataset_path = std::string(first_dataset_path);
    result.second_dataset_path = std::string(second_dataset_path);

    const auto* first = find_dataset(catalog, first_dataset_path);
    const auto* second = find_dataset(catalog, second_dataset_path);
    std::string side_reason;
    if (first == nullptr) {
        append_issue(side_reason, "first dataset path is not in the catalog");
    }
    if (second == nullptr) {
        append_issue(side_reason, "second dataset path is not in the catalog");
    }
    if (first != nullptr && second != nullptr) {
        result.first_dataset_path = first->path;
        result.second_dataset_path = second->path;
        if (first->path == second->path) {
            append_issue(side_reason, "choose two different datasets");
        }
        if (!is_readable_science_raster(*first)) {
            append_issue(
                side_reason,
                "first dataset is not a readable numeric science raster");
        }
        if (!is_readable_science_raster(*second)) {
            append_issue(
                side_reason,
                "second dataset is not a readable numeric science raster");
        }
        if (is_readable_science_raster(*first) &&
            is_readable_science_raster(*second) &&
            !side_by_side_is_meaningful(*first, *second)) {
            append_issue(
                side_reason,
                "datasets do not share a science role, layer kind, or units");
        }
    }

    const bool side_available = side_reason.empty();
    std::string aligned_reason;
    if (!side_available) {
        aligned_reason = "side-by-side prerequisite failed: " + side_reason;
    } else {
        if (first->frequency.empty() || second->frequency.empty() ||
            !same_token(first->frequency, second->frequency)) {
            append_issue(
                aligned_reason,
                "aligned comparison requires the same frequency");
        }
        if (first->dimensions != second->dimensions ||
            first->dimensions[0] == 0 || first->dimensions[1] == 0) {
            append_issue(
                aligned_reason,
                "aligned comparison requires identical non-empty dimensions");
        }
        if (trim_ascii(first->grid_mapping) !=
            trim_ascii(second->grid_mapping)) {
            append_issue(
                aligned_reason,
                "aligned comparison requires the same grid mapping");
        }
        if (!compatible_units(first->units, second->units)) {
            append_issue(
                aligned_reason,
                "aligned comparison requires compatible units");
        }
        if ((first->data_type.kind == ScalarKind::compound_complex) !=
            (second->data_type.kind == ScalarKind::compound_complex)) {
            append_issue(
                aligned_reason,
                "aligned comparison requires both sources to be real or both complex");
        }

        const auto first_mask_path =
            join_path(parent_path(first->path), "mask");
        const auto second_mask_path =
            join_path(parent_path(second->path), "mask");
        const auto* first_declared_mask =
            find_dataset(catalog, first_mask_path);
        const auto* second_declared_mask =
            find_dataset(catalog, second_mask_path);
        const auto* first_mask = valid_sibling_mask(catalog, *first);
        const auto* second_mask = valid_sibling_mask(catalog, *second);
        if ((first_declared_mask != nullptr && first_mask == nullptr) ||
            (second_declared_mask != nullptr && second_mask == nullptr)) {
            append_issue(
                aligned_reason,
                "aligned comparison has an unreadable or grid-incompatible mask");
        } else if ((first_mask == nullptr) != (second_mask == nullptr) ||
                   (first_mask != nullptr &&
                    first_mask->path != second_mask->path)) {
            append_issue(
                aligned_reason,
                "aligned comparison requires the same validity mask policy");
        }

        if (same_token(first->frequency, second->frequency) &&
            !first->frequency.empty()) {
            const auto* frequency = find_frequency(catalog, first->frequency);
            if (frequency == nullptr) {
                append_issue(
                    aligned_reason,
                    "frequency grid metadata is missing from the catalog");
            } else if (!grid_dimensions_match(
                           *frequency, first->dimensions) ||
                       !grid_dimensions_match(
                           *frequency, second->dimensions)) {
                append_issue(
                    aligned_reason,
                    "dataset dimensions do not match the frequency grid");
            } else if (!grid_coordinates_are_usable(*frequency)) {
                append_issue(
                    aligned_reason,
                    "frequency grid coordinate metadata is incomplete");
            }
        }
    }

    const bool aligned_available = aligned_reason.empty();
    result.modes = {
        mode_capability(
            CompareMode::side_by_side, side_available, side_reason),
        mode_capability(
            CompareMode::swipe, aligned_available, aligned_reason),
        mode_capability(
            CompareMode::difference, aligned_available, aligned_reason),
        mode_capability(
            CompareMode::ratio, aligned_available, aligned_reason),
    };

    if (side_available) {
        const auto mask = shared_sibling_mask(catalog, *first, *second);
        result.recipe = CompareRecipe{
            .first_dataset_path = first->path,
            .second_dataset_path = second->path,
            .shared_mask_dataset_path = mask,
            .strictly_aligned = aligned_available,
            .frequency = aligned_available ? first->frequency : std::string{},
            .dimensions = aligned_available
                ? first->dimensions
                : std::array<std::uint64_t, 2>{0, 0},
        };
    }
    return result;
}

CompareCapability resolve_compare_capability(
    const Hdf5Product& product,
    const std::string_view first_dataset_path,
    const std::string_view second_dataset_path) {
    return resolve_compare_capability(
        catalog_view(product), first_dataset_path, second_dataset_path);
}

}  // namespace satview::analysis
