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

namespace satview::aligned {

inline constexpr std::size_t kDefaultMaximumOutputBytes =
    256ULL * 1024ULL * 1024ULL;
inline constexpr std::size_t kDefaultMaximumScratchBytes =
    64ULL * 1024ULL * 1024ULL;
inline constexpr std::size_t kMaximumScienceRasterCount = 4;

// A science raster may name a sibling uint8 validity mask. Repeated mask
// paths are read once and exposed once in AlignedPlan::masks and each block.
struct RasterRequest {
    std::string dataset_path;
    std::optional<std::string> mask_dataset_path;
};

// Every output sample is the exact native source sample at
// source_window.origin + sample_stride * output_index. No interpolation or
// reprojection is performed. All rasters must therefore belong to one
// frequency grid and have identical dimensions/grid metadata.
struct AlignedRequest {
    std::vector<RasterRequest> rasters;
    std::optional<Window2D> source_window;
    std::array<std::uint64_t, 2> sample_stride{1, 1};
    std::size_t maximum_output_bytes = kDefaultMaximumOutputBytes;
    std::size_t maximum_scratch_bytes = kDefaultMaximumScratchBytes;
    // Direct decoding is used only when Hdf5Product confirms that the exact
    // member ReadPlan is eligible. Other reads retain HDF5's chunk cache.
    bool prefer_direct_chunk_decode = false;
};

struct RasterLayout {
    std::string dataset_path;
    DataTypeInfo data_type;
    std::optional<std::size_t> mask_index;

    [[nodiscard]] bool operator==(const RasterLayout&) const = default;
};

struct MaskLayout {
    std::string dataset_path;
    DataTypeInfo data_type;

    [[nodiscard]] bool operator==(const MaskLayout&) const = default;
};

struct AlignedLayout {
    std::array<std::uint64_t, 2> source_origin{0, 0};
    std::array<std::uint64_t, 2> source_dimensions{0, 0};
    std::array<std::uint64_t, 2> sample_stride{1, 1};
    std::array<std::uint64_t, 2> output_dimensions{0, 0};
    std::array<std::uint64_t, 2> driver_block_dimensions{0, 0};
    std::uint64_t source_block_count = 0;
    // Bytes required to materialize every science raster and every unique
    // mask on the canonical output lattice. The visitor itself does not make
    // this allocation, but planning enforces maximum_output_bytes against it.
    std::size_t materialized_bytes = 0;
    // Explicit reader-owned staging: one reusable native read buffer plus all
    // compact member payloads for the largest callback block. It excludes
    // HDF5's configured chunk cache and decoder-private storage.
    std::size_t scratch_bytes = 0;
    std::size_t maximum_member_read_bytes = 0;
    std::uint64_t maximum_block_sample_count = 0;

    [[nodiscard]] bool operator==(const AlignedLayout&) const = default;
};

struct AlignedPlan {
    AlignedRequest request;
    std::filesystem::path source_file;
    std::string frequency;
    AlignedLayout layout;
    // Raster order matches request.rasters. Mask order is stable first-use
    // order after path deduplication; RasterLayout::mask_index maps the two.
    std::vector<RasterLayout> rasters;
    std::vector<MaskLayout> masks;
    // Source identity plus exact dataset/grid metadata. The reader rebuilds
    // and compares this fingerprint before any source block is published.
    std::vector<std::byte> validation_metadata;
};

struct MemberBlock {
    // Index into AlignedPlan::rasters or AlignedPlan::masks, depending on the
    // containing span. Samples are compact row-major native representations.
    std::size_t member_index = 0;
    std::span<const std::byte> samples;
};

struct AlignedBlock {
    // Output-lattice index and native coordinate of this block's first sample.
    std::array<std::uint64_t, 2> output_origin{0, 0};
    std::array<std::uint64_t, 2> source_sample_origin{0, 0};
    std::array<std::uint64_t, 2> dimensions{0, 0};
    // Both spans and all sample spans are valid only for the visitor call.
    // Raster and mask order exactly matches the corresponding plan vectors.
    std::span<const MemberBlock> rasters;
    std::span<const MemberBlock> masks;
};

struct AlignedProgress {
    std::uint64_t completed_source_blocks = 0;
    std::uint64_t total_source_blocks = 0;
    std::uint64_t source_bytes_read = 0;

    [[nodiscard]] double fraction() const noexcept;
};

enum class VisitStatus {
    completed,
    cancelled,
    visitor_stopped,
};

using CancelCallback = std::function<bool()>;
// Returning false stops cleanly after the complete current block.
using BlockVisitor = std::function<bool(const AlignedBlock&)>;
using ProgressCallback = std::function<void(const AlignedProgress&)>;

// Validates/canonicalizes paths, bounds, byte arithmetic, raster/mask count,
// one-frequency grid identity, and the complete-lattice/output and explicit
// block-staging budgets. The first science raster's native chunk grid drives
// block traversal (a bounded 512x512 fallback is used when it is contiguous).
[[nodiscard]] AlignedPlan make_aligned_plan(
    const Hdf5Product& product,
    const AlignedRequest& request);

// Synchronous and intended for a caller-owned worker thread. A block is
// visited only after every science member and deduplicated mask has been read,
// sampled, and cancellation-checked. Visitor, progress, and cancellation
// callbacks run only while no HDF5 call is active. A stale plan throws before
// the first block callback; cancellation never publishes a partial block.
[[nodiscard]] VisitStatus visit_aligned_blocks(
    const Hdf5Product& product,
    const AlignedPlan& plan,
    BlockVisitor visitor,
    CancelCallback cancel = {},
    ProgressCallback progress = {});

}  // namespace satview::aligned
