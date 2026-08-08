#pragma once

#include "satview/hdf5_product.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace satview::overview {

inline constexpr std::uint32_t kDefaultOverviewLongEdge = 2048;
inline constexpr std::uint32_t kMaximumOverviewLongEdge = 4096;
inline constexpr std::size_t kDefaultMaximumOverviewBytes =
    256ULL * 1024ULL * 1024ULL;
inline constexpr std::size_t kDefaultMaximumOverviewScratchBytes =
    64ULL * 1024ULL * 1024ULL;

struct OverviewRequest {
    std::string science_dataset_path;
    std::optional<std::string> mask_dataset_path;
    std::uint32_t maximum_long_edge = kDefaultOverviewLongEdge;
    std::size_t maximum_output_bytes = kDefaultMaximumOverviewBytes;
    std::size_t maximum_scratch_bytes =
        kDefaultMaximumOverviewScratchBytes;
    // Optional native-pixel coverage. The default is the complete dataset.
    std::optional<Window2D> source_window;
    // Optional exact row/column sampling strides. When omitted, the planner
    // chooses physically balanced strides that honor maximum_long_edge.
    std::optional<std::array<std::uint64_t, 2>> sample_stride;
};

// Output sample (r, c) is always the exact native source sample at
// source_origin + sample_stride * (r, c). Science and mask use this same
// request-anchored lattice. source_dimensions is the covered regional extent,
// not necessarily the complete dataset shape.
struct OverviewLayout {
    std::array<std::uint64_t, 2> source_origin{0, 0};
    std::array<std::uint64_t, 2> source_dimensions{0, 0};
    std::array<std::uint64_t, 2> sample_stride{1, 1};
    std::array<std::uint64_t, 2> output_dimensions{0, 0};
    DataTypeInfo science_type;
    std::size_t science_bytes = 0;
    std::size_t mask_bytes = 0;
    std::uint64_t source_chunk_count = 0;
    // Per-worker source staging; the builder limits aggregate staging to the
    // request's maximum_scratch_bytes and never uses more than eight workers.
    std::size_t scratch_bytes = 0;

    [[nodiscard]] bool operator==(const OverviewLayout&) const = default;
};

struct OverviewPlan {
    OverviewRequest request;
    std::filesystem::path source_file;
    OverviewLayout layout;
    // Exact source/dataset fingerprint material used to reject a stale plan.
    std::vector<std::byte> validation_metadata;
};

struct OverviewProgress {
    std::uint64_t completed_source_chunks = 0;
    std::uint64_t total_source_chunks = 0;
    std::uint64_t source_bytes_read = 0;

    [[nodiscard]] double fraction() const noexcept;
};

enum class OverviewPrepareStatus {
    built,
    cancelled,
};

using OverviewCancelCallback = std::function<bool()>;
using OverviewProgressCallback =
    std::function<void(const OverviewProgress&)>;

struct OverviewData {
    OverviewPrepareStatus status = OverviewPrepareStatus::cancelled;
    OverviewPlan plan;
    std::vector<std::byte> science_bytes;
    std::vector<std::byte> mask_bytes;

    [[nodiscard]] bool ready() const noexcept {
        return status != OverviewPrepareStatus::cancelled;
    }
};

// Planning is read-only. It validates and canonicalizes optional regional
// coverage, chooses (or validates explicit) bounded row/column integer
// strides, fingerprints the source and dataset metadata, and computes the
// per-worker scratch requirement. The builder bounds aggregate worker staging
// by maximum_scratch_bytes. Regional sampling is anchored at
// source_window.{row,column}; default whole-scene sampling remains anchored at
// the dataset origin.
[[nodiscard]] OverviewPlan make_overview_plan(
    const Hdf5Product& product,
    const OverviewRequest& request);

// Synchronous and intended to run inside a caller-owned worker thread. The
// cancel callback is checked before work and between every source-chunk read.
// Progress is reported after each completed source chunk. Cancellation returns
// no partial output.
[[nodiscard]] OverviewData build_overview(
    const Hdf5Product& product,
    const OverviewPlan& plan,
    OverviewCancelCallback cancel = {},
    OverviewProgressCallback progress = {});

}  // namespace satview::overview
