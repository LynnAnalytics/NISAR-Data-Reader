#pragma once

#include "satview/hdf5_product.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace satview::analysis {

struct CatalogView {
    std::span<const FrequencyCatalog> frequencies;
    std::span<const DatasetInfo> datasets;
};

[[nodiscard]] CatalogView catalog_view(
    const Hdf5Product& product) noexcept;

struct PauliRecipe {
    std::string frequency;
    std::string hhhh_dataset_path;
    std::string hvhv_dataset_path;
    std::string vvvv_dataset_path;
    std::string hhvv_dataset_path;
    std::optional<std::string> shared_mask_dataset_path;
    std::array<std::uint64_t, 2> dimensions{0, 0};

    [[nodiscard]] bool operator==(const PauliRecipe&) const = default;
};

struct PauliCapability {
    std::string frequency;
    bool available = false;
    // Empty when available. Unavailable entries contain a deterministic,
    // user-facing explanation instead of leaving callers to infer the cause.
    std::string reason;
    std::optional<PauliRecipe> recipe;
};

// One entry is returned for every frequency, in catalog order.
[[nodiscard]] std::vector<PauliCapability> resolve_pauli_capabilities(
    CatalogView catalog);

[[nodiscard]] std::vector<PauliCapability> resolve_pauli_capabilities(
    const Hdf5Product& product);

enum class CompareMode {
    side_by_side,
    swipe,
    difference,
    ratio,
};

struct CompareModeCapability {
    CompareMode mode = CompareMode::side_by_side;
    bool available = false;
    std::string reason;
};

struct CompareRecipe {
    std::string first_dataset_path;
    std::string second_dataset_path;
    std::optional<std::string> shared_mask_dataset_path;
    // When false, only independent side-by-side presentation is resolved.
    bool strictly_aligned = false;
    std::string frequency;
    std::array<std::uint64_t, 2> dimensions{0, 0};

    [[nodiscard]] bool operator==(const CompareRecipe&) const = default;
};

struct CompareCapability {
    std::string first_dataset_path;
    std::string second_dataset_path;
    std::array<CompareModeCapability, 4> modes{};
    std::optional<CompareRecipe> recipe;

    [[nodiscard]] bool supports(CompareMode mode) const noexcept;
    [[nodiscard]] std::string_view reason_for(
        CompareMode mode) const noexcept;
};

// Paths are resolved exactly and the returned recipe retains their canonical
// catalog spelling. Side-by-side comparison admits semantically related
// real or complex science rasters on different grids. Swipe, difference, and ratio additionally
// require one frequency, identical dimensions/grid metadata, and compatible
// units.
[[nodiscard]] CompareCapability resolve_compare_capability(
    CatalogView catalog,
    std::string_view first_dataset_path,
    std::string_view second_dataset_path);

[[nodiscard]] CompareCapability resolve_compare_capability(
    const Hdf5Product& product,
    std::string_view first_dataset_path,
    std::string_view second_dataset_path);

}  // namespace satview::analysis
