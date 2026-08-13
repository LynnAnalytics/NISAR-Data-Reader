#include "satview/overview_builder.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <stop_token>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace satview::overview {
namespace {

constexpr std::size_t kEdgeFingerprintBytes = 64ULL * 1024ULL;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
constexpr std::size_t kMaximumOverviewDecodeWorkers = 8;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error("overview: " + message);
}

[[nodiscard]] std::uint64_t ceil_div(
    const std::uint64_t value,
    const std::uint64_t divisor) {
    if (divisor == 0) {
        fail("division by zero");
    }
    return value / divisor +
        static_cast<std::uint64_t>(value % divisor != 0);
}

[[nodiscard]] std::uint64_t checked_ceil_stride(
    const long double value) {
    const auto maximum = static_cast<long double>(
        std::numeric_limits<std::uint64_t>::max());
    if (!std::isfinite(value) || value <= 0.0L || value > maximum) {
        fail("physical sample stride is not representable");
    }
    const auto rounded = std::ceil(value);
    if (!std::isfinite(rounded) || rounded > maximum) {
        fail("physical sample stride is not representable");
    }
    return std::max<std::uint64_t>(
        1, static_cast<std::uint64_t>(rounded));
}

[[nodiscard]] std::uint64_t maximum_aligned_extent(
    const std::uint64_t request,
    const std::uint64_t dimension,
    const std::uint64_t chunk_extent) {
    if (request == 0 || dimension == 0 || chunk_extent == 0) {
        fail("aligned extent arguments must be positive");
    }
    if (request >= dimension) {
        return dimension;
    }
    const auto leading_residue = chunk_extent - 1;
    const auto remaining = dimension - request;
    if (leading_residue >= remaining) {
        return dimension;
    }
    const auto shifted = request + leading_residue;
    const auto chunks = ceil_div(shifted, chunk_extent);
    if (chunks > dimension / chunk_extent) {
        return dimension;
    }
    return std::min(dimension, chunks * chunk_extent);
}

[[nodiscard]] std::uint64_t checked_u64_add(
    const std::uint64_t left,
    const std::uint64_t right,
    const char* context) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        fail(std::string(context) + " overflow");
    }
    return left + right;
}

[[nodiscard]] std::uint64_t checked_u64_product(
    const std::uint64_t left,
    const std::uint64_t right,
    const char* context) {
    if (left != 0 &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        fail(std::string(context) + " overflow");
    }
    return left * right;
}

[[nodiscard]] std::size_t checked_size(
    const std::uint64_t rows,
    const std::uint64_t columns,
    const std::size_t element_size,
    const char* context) {
    const auto pixels = checked_u64_product(rows, columns, context);
    if (element_size != 0 &&
        pixels > std::numeric_limits<std::size_t>::max() / element_size) {
        fail(std::string(context) + " exceeds size_t");
    }
    return static_cast<std::size_t>(pixels) * element_size;
}

[[nodiscard]] std::uint64_t fnv_update(
    std::uint64_t hash,
    const std::span<const std::byte> bytes) noexcept {
    for (const auto value : bytes) {
        hash ^= static_cast<std::uint8_t>(value);
        hash *= kFnvPrime;
    }
    return hash;
}

class MetadataWriter final {
public:
    void u8(const std::uint8_t value) {
        bytes_.push_back(static_cast<std::byte>(value));
    }

    void u32(const std::uint32_t value) {
        for (std::size_t index = 0; index < sizeof(value); ++index) {
            u8(static_cast<std::uint8_t>(value >> (index * 8)));
        }
    }

    void u64(const std::uint64_t value) {
        for (std::size_t index = 0; index < sizeof(value); ++index) {
            u8(static_cast<std::uint8_t>(value >> (index * 8)));
        }
    }

    void string(const std::string_view value) {
        u64(value.size());
        const auto* begin =
            reinterpret_cast<const std::byte*>(value.data());
        bytes_.insert(bytes_.end(), begin, begin + value.size());
    }

    [[nodiscard]] std::vector<std::byte> finish() && {
        return std::move(bytes_);
    }

private:
    std::vector<std::byte> bytes_;
};

[[nodiscard]] std::string path_utf8(const std::filesystem::path& path) {
    const auto encoded = path.generic_u8string();
    return {
        reinterpret_cast<const char*>(encoded.data()),
        encoded.size()};
}

[[nodiscard]] std::filesystem::path normalized_absolute_path(
    const std::filesystem::path& path) {
    std::error_code error;
    auto result = std::filesystem::weakly_canonical(path, error);
    if (error) {
        error.clear();
        result = std::filesystem::absolute(path, error);
    }
    if (error) {
        fail("could not normalize path: " + path.string());
    }
    return result.lexically_normal();
}

struct SourceIdentity {
    std::uint64_t size = 0;
    std::uint64_t modified = 0;
    std::uint64_t edge_hash = 0;
};

[[nodiscard]] SourceIdentity source_identity(
    const std::filesystem::path& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error) {
        fail("could not stat source file: " + path.string());
    }
    const auto modified = std::filesystem::last_write_time(path, error);
    if (error) {
        fail("could not read source modification time: " + path.string());
    }

    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        fail("could not open source for fingerprinting: " + path.string());
    }
    std::array<std::byte, kEdgeFingerprintBytes> buffer{};
    std::uint64_t hash = kFnvOffset;
    auto read_region = [&](const std::uint64_t offset,
                                 const std::size_t requested) {
        stream.clear();
        stream.seekg(static_cast<std::streamoff>(offset));
        if (!stream) {
            fail("could not seek source while fingerprinting");
        }
        stream.read(
            reinterpret_cast<char*>(buffer.data()),
            static_cast<std::streamsize>(requested));
        const auto received = stream.gcount();
        if (received < 0 ||
            static_cast<std::size_t>(received) != requested) {
            fail("short source read while fingerprinting");
        }
        std::array<std::byte, sizeof(offset)> encoded_offset{};
        for (std::size_t index = 0; index < sizeof(offset); ++index) {
            encoded_offset[index] = static_cast<std::byte>(
                offset >> (index * 8));
        }
        hash = fnv_update(hash, encoded_offset);
        hash = fnv_update(
            hash,
            std::span<const std::byte>(
                buffer.data(), static_cast<std::size_t>(received)));
    };

    const auto first_bytes = static_cast<std::size_t>(
        std::min<std::uint64_t>(size, buffer.size()));
    read_region(0, first_bytes);
    if (size > first_bytes) {
        const auto last_bytes = static_cast<std::size_t>(
            std::min<std::uint64_t>(size - first_bytes, buffer.size()));
        read_region(size - last_bytes, last_bytes);
    }

    return SourceIdentity{
        .size = size,
        .modified = static_cast<std::uint64_t>(
            modified.time_since_epoch().count()),
        .edge_hash = hash,
    };
}

void append_data_type(
    MetadataWriter& writer,
    const DataTypeInfo& type) {
    writer.u32(static_cast<std::uint32_t>(type.kind));
    writer.u32(static_cast<std::uint32_t>(type.file_byte_order));
    writer.u64(type.file_element_size);
    writer.u64(type.element_size);
    writer.u8(static_cast<std::uint8_t>(type.readable));
    writer.string(type.description);
    writer.u8(static_cast<std::uint8_t>(
        type.complex_layout.has_value()));
    if (type.complex_layout.has_value()) {
        const auto& complex = *type.complex_layout;
        writer.u64(complex.component_size);
        writer.u64(complex.file_real_offset);
        writer.u64(complex.file_imaginary_offset);
        writer.string(complex.real_member_name);
        writer.string(complex.imaginary_member_name);
    }
}

void append_dataset(
    MetadataWriter& writer,
    const DatasetInfo& dataset) {
    writer.string(dataset.path);
    writer.u64(dataset.dimensions[0]);
    writer.u64(dataset.dimensions[1]);
    writer.u32(static_cast<std::uint32_t>(dataset.storage_layout));
    writer.u8(static_cast<std::uint8_t>(
        dataset.chunk_dimensions.has_value()));
    if (dataset.chunk_dimensions.has_value()) {
        writer.u64((*dataset.chunk_dimensions)[0]);
        writer.u64((*dataset.chunk_dimensions)[1]);
    }
    append_data_type(writer, dataset.data_type);
    writer.u64(dataset.filters.size());
    for (const auto& filter : dataset.filters) {
        writer.u32(filter.id);
        writer.u32(filter.flags);
        writer.u32(filter.configuration);
        writer.string(filter.name);
        writer.u64(filter.client_data.size());
        for (const auto value : filter.client_data) {
            writer.u32(value);
        }
    }
}

[[nodiscard]] std::vector<std::byte> validation_metadata(
    const std::filesystem::path& source_path,
    const SourceIdentity& identity,
    const DatasetInfo& science,
    const DatasetInfo* const mask,
    const OverviewLayout& layout,
    const std::uint32_t maximum_long_edge) {
    MetadataWriter writer;
    writer.string("satview-nearest-overview-plan-v1");
    writer.string(path_utf8(source_path));
    writer.u64(identity.size);
    writer.u64(identity.modified);
    writer.u64(identity.edge_hash);
    writer.u32(maximum_long_edge);
    writer.u64(layout.source_origin[0]);
    writer.u64(layout.source_origin[1]);
    writer.u64(layout.sample_stride[0]);
    writer.u64(layout.sample_stride[1]);
    writer.u64(layout.output_dimensions[0]);
    writer.u64(layout.output_dimensions[1]);
    append_dataset(writer, science);
    writer.u8(static_cast<std::uint8_t>(mask != nullptr));
    if (mask != nullptr) {
        append_dataset(writer, *mask);
    }
    // Default and explicit whole-scene requests share one canonical identity.
    // Only a true regional window needs the extent discriminator.
    if (layout.source_origin != std::array<std::uint64_t, 2>{0, 0} ||
        layout.source_dimensions != science.dimensions) {
        writer.string("regional-source-window-v1");
        writer.u64(layout.source_dimensions[0]);
        writer.u64(layout.source_dimensions[1]);
    }
    return std::move(writer).finish();
}

[[nodiscard]] bool cancelled(
    const OverviewCancelCallback& callback) {
    return callback && callback();
}

struct SelectedAxis {
    std::uint64_t first_source = 0;
    std::uint64_t destination = 0;
    std::uint64_t count = 0;
};

template <std::size_t ElementBytes>
void copy_strided_elements(
    std::byte* destination,
    const std::byte* source,
    const std::size_t source_stride_bytes,
    const std::size_t count) noexcept {
    for (std::size_t index = 0; index < count; ++index) {
        std::memcpy(destination, source, ElementBytes);
        destination += ElementBytes;
        source += source_stride_bytes;
    }
}

[[nodiscard]] SelectedAxis selected_axis(
    const std::uint64_t start,
    const std::uint64_t extent,
    const std::uint64_t origin,
    const std::uint64_t stride) {
    if (extent == 0) {
        return {};
    }
    const auto end = checked_u64_add(start, extent, "source block end");
    if (start < origin) {
        fail("source block precedes the sampling origin");
    }
    const auto first_index = ceil_div(start - origin, stride);
    const auto last_index = (end - 1 - origin) / stride;
    if (first_index > last_index) {
        return {};
    }
    const auto first_source = checked_u64_add(
        origin,
        checked_u64_product(first_index, stride, "sample coordinate"),
        "sample coordinate");
    return SelectedAxis{
        .first_source = first_source,
        .destination = first_index,
        .count = last_index - first_index + 1,
    };
}

void copy_sampled_block(
    const std::span<std::byte> destination,
    const std::span<const std::byte> source,
    const ReadPlan& source_plan,
    const Window2D& requested_block,
    const OverviewLayout& layout,
    const std::size_t element_size,
    std::uint64_t& samples_written) {
    const auto rows = selected_axis(
        requested_block.row,
        requested_block.height,
        layout.source_origin[0],
        layout.sample_stride[0]);
    const auto columns = selected_axis(
        requested_block.column,
        requested_block.width,
        layout.source_origin[1],
        layout.sample_stride[1]);
    if (rows.count == 0 || columns.count == 0) {
        return;
    }
    const auto expected_destination_bytes = checked_size(
        layout.output_dimensions[0],
        layout.output_dimensions[1],
        element_size,
        "overview destination");
    if (destination.size() != expected_destination_bytes) {
        fail("overview destination does not match its layout");
    }

    const auto last_source_row = checked_u64_add(
        rows.first_source,
        checked_u64_product(
            rows.count - 1,
            layout.sample_stride[0],
            "last sampled row"),
        "last sampled row");
    const auto last_source_column = checked_u64_add(
        columns.first_source,
        checked_u64_product(
            columns.count - 1,
            layout.sample_stride[1],
            "last sampled column"),
        "last sampled column");
    const auto aligned_row_end = checked_u64_add(
        source_plan.aligned.row,
        source_plan.aligned.height,
        "aligned source row end");
    const auto aligned_column_end = checked_u64_add(
        source_plan.aligned.column,
        source_plan.aligned.width,
        "aligned source column end");
    if (rows.first_source < source_plan.aligned.row ||
        columns.first_source < source_plan.aligned.column ||
        last_source_row >= aligned_row_end ||
        last_source_column >= aligned_column_end) {
        fail("sample coordinates escape their HDF5 read plan");
    }

    const auto source_row_bytes = checked_u64_product(
        source_plan.aligned.width,
        element_size,
        "source row bytes");
    const auto source_sample_row_bytes = rows.count > 1
        ? checked_u64_product(
              source_row_bytes,
              layout.sample_stride[0],
              "source sampled-row stride")
        : 0;
    const auto source_sample_column_bytes = columns.count > 1
        ? checked_u64_product(
              layout.sample_stride[1],
              element_size,
              "source sampled-column stride")
        : 0;
    const auto source_start_index = checked_u64_add(
        checked_u64_product(
            rows.first_source - source_plan.aligned.row,
            source_plan.aligned.width,
            "source start row"),
        columns.first_source - source_plan.aligned.column,
        "source start column");
    const auto source_start_bytes = checked_u64_product(
        source_start_index, element_size, "source start bytes");
    const auto destination_start_index = checked_u64_add(
        checked_u64_product(
            rows.destination,
            layout.output_dimensions[1],
            "overview destination row"),
        columns.destination,
        "overview destination column");
    const auto destination_start_bytes = checked_u64_product(
        destination_start_index,
        element_size,
        "overview destination bytes");
    const auto destination_row_bytes = checked_u64_product(
        layout.output_dimensions[1],
        element_size,
        "overview destination row bytes");
    const auto output_row_bytes = checked_size(
        1, columns.count, element_size, "sampled output row");
    const auto final_destination_end = checked_u64_add(
        destination_start_bytes,
        checked_u64_add(
            checked_u64_product(
                rows.count - 1,
                destination_row_bytes,
                "last destination row"),
            output_row_bytes,
            "last destination row end"),
        "overview destination end");
    if (source_start_bytes > source.size() ||
        destination_start_bytes > destination.size() ||
        final_destination_end > destination.size()) {
        fail("sampled block exceeds its source or destination");
    }

    const auto row_count = static_cast<std::size_t>(rows.count);
    const auto column_count = static_cast<std::size_t>(columns.count);
    const auto source_row_step =
        static_cast<std::size_t>(source_sample_row_bytes);
    const auto source_column_step =
        static_cast<std::size_t>(source_sample_column_bytes);
    const auto destination_row_step =
        static_cast<std::size_t>(destination_row_bytes);
    auto* destination_row = destination.data() +
        static_cast<std::size_t>(destination_start_bytes);
    const auto* source_row = source.data() +
        static_cast<std::size_t>(source_start_bytes);
    for (std::size_t row = 0; row < row_count; ++row) {
        if (layout.sample_stride[1] == 1) {
            std::memcpy(destination_row, source_row, output_row_bytes);
        } else {
            switch (element_size) {
            case 1:
                copy_strided_elements<1>(
                    destination_row, source_row, source_column_step,
                    column_count);
                break;
            case 4:
                copy_strided_elements<4>(
                    destination_row, source_row, source_column_step,
                    column_count);
                break;
            case 8:
                copy_strided_elements<8>(
                    destination_row, source_row, source_column_step,
                    column_count);
                break;
            default:
                for (std::size_t column = 0;
                     column < column_count;
                     ++column) {
                    std::memcpy(
                        destination_row + column * element_size,
                        source_row + column * source_column_step,
                        element_size);
                }
                break;
            }
        }
        destination_row += destination_row_step;
        source_row += source_row_step;
    }
    samples_written = checked_u64_add(
        samples_written,
        checked_u64_product(rows.count, columns.count, "sample count"),
        "written sample count");
}

[[nodiscard]] bool build_overview_data(
    const Hdf5Product& product,
    const OverviewPlan& plan,
    const OverviewCancelCallback& cancel,
    const OverviewProgressCallback& progress,
    std::vector<std::byte>& science_output,
    std::vector<std::byte>& mask_output) {
    if (cancelled(cancel)) {
        return false;
    }

    const auto* science = product.find_dataset(
        plan.request.science_dataset_path);
    const auto* mask = plan.request.mask_dataset_path.has_value()
        ? product.find_dataset(*plan.request.mask_dataset_path)
        : nullptr;
    if (science == nullptr ||
        (plan.request.mask_dataset_path.has_value() && mask == nullptr)) {
        fail("planned datasets disappeared before overview build");
    }

    science_output.assign(plan.layout.science_bytes, std::byte{});
    mask_output.assign(plan.layout.mask_bytes, std::byte{});
    const auto block = science->chunk_dimensions.value_or(
        std::array<std::uint64_t, 2>{
            std::min<std::uint64_t>(science->dimensions[0], 512),
            std::min<std::uint64_t>(science->dimensions[1], 512)});
    std::atomic<std::uint64_t> science_samples_written{0};
    std::atomic<std::uint64_t> mask_samples_written{0};
    OverviewProgress current{
        .completed_source_chunks = 0,
        .total_source_chunks = plan.layout.source_chunk_count,
        .source_bytes_read = 0,
    };

    const auto source_row_end = checked_u64_add(
        plan.layout.source_origin[0],
        plan.layout.source_dimensions[0],
        "regional source row end");
    const auto source_column_end = checked_u64_add(
        plan.layout.source_origin[1],
        plan.layout.source_dimensions[1],
        "regional source column end");
    const auto first_chunk_row =
        plan.layout.source_origin[0] -
        plan.layout.source_origin[0] % block[0];
    const auto first_chunk_column =
        plan.layout.source_origin[1] -
        plan.layout.source_origin[1] % block[1];

    const auto chunk_row_count = ceil_div(
        source_row_end - first_chunk_row, block[0]);
    const auto chunk_column_count = ceil_div(
        source_column_end - first_chunk_column, block[1]);
    if (checked_u64_product(
            chunk_row_count,
            chunk_column_count,
            "overview source chunk grid") !=
        plan.layout.source_chunk_count) {
        fail("planned source chunk count changed before overview build");
    }

    const auto representative_plan = product.make_read_plan(
        science->path,
        plan.layout.source_origin[0],
        plan.layout.source_origin[1],
        1,
        1);
    std::size_t worker_count = 1;
    if (product.supports_direct_chunk_decode(representative_plan) &&
        plan.layout.source_chunk_count > 1) {
        const auto hardware_workers = std::max<std::size_t>(
            1, std::thread::hardware_concurrency());
        const auto scratch_workers = std::max<std::size_t>(
            1,
            plan.request.maximum_scratch_bytes /
                plan.layout.scratch_bytes);
        const auto chunk_workers = static_cast<std::size_t>(
            std::min<std::uint64_t>(
                plan.layout.source_chunk_count,
                std::numeric_limits<std::size_t>::max()));
        worker_count = std::min({
            kMaximumOverviewDecodeWorkers,
            hardware_workers,
            scratch_workers,
            chunk_workers});
    }

    std::atomic<std::uint64_t> next_source_chunk{0};
    std::atomic_bool stop_workers{false};
    std::atomic_bool build_cancelled{false};
    std::mutex callback_mutex;
    std::mutex failure_mutex;
    std::exception_ptr first_failure;

    const auto record_failure = [&] {
        std::lock_guard lock(failure_mutex);
        if (first_failure == nullptr) {
            first_failure = std::current_exception();
        }
        stop_workers.store(true, std::memory_order_release);
    };

    const auto build_source_chunks = [&](const std::stop_token stop_token) {
        try {
            std::vector<std::byte> scratch(
                plan.layout.scratch_bytes);
            while (!stop_token.stop_requested() &&
                   !stop_workers.load(std::memory_order_acquire)) {
                std::uint64_t source_chunk_index = 0;
                {
                    std::lock_guard lock(callback_mutex);
                    if (stop_workers.load(
                            std::memory_order_acquire)) {
                        return;
                    }
                    if (cancelled(cancel)) {
                        build_cancelled.store(
                            true, std::memory_order_release);
                        stop_workers.store(
                            true, std::memory_order_release);
                        return;
                    }
                    source_chunk_index = next_source_chunk.fetch_add(
                        1, std::memory_order_relaxed);
                }
                if (source_chunk_index >=
                    plan.layout.source_chunk_count) {
                    return;
                }

                const auto chunk_row_index =
                    source_chunk_index / chunk_column_count;
                const auto chunk_column_index =
                    source_chunk_index % chunk_column_count;
                const auto chunk_row = checked_u64_add(
                    first_chunk_row,
                    checked_u64_product(
                        chunk_row_index,
                        block[0],
                        "overview chunk row"),
                    "overview chunk row");
                const auto chunk_column = checked_u64_add(
                    first_chunk_column,
                    checked_u64_product(
                        chunk_column_index,
                        block[1],
                        "overview chunk column"),
                    "overview chunk column");
                const auto chunk_height = std::min(
                    block[0],
                    science->dimensions[0] - chunk_row);
                const auto chunk_width = std::min(
                    block[1],
                    science->dimensions[1] - chunk_column);
                const auto chunk_row_end = checked_u64_add(
                    chunk_row,
                    chunk_height,
                    "overview chunk row end");
                const auto chunk_column_end = checked_u64_add(
                    chunk_column,
                    chunk_width,
                    "overview chunk column end");
                const auto row = std::max(
                    chunk_row, plan.layout.source_origin[0]);
                const auto column = std::max(
                    chunk_column, plan.layout.source_origin[1]);
                const auto height = std::min(
                    chunk_row_end, source_row_end) - row;
                const auto width = std::min(
                    chunk_column_end, source_column_end) - column;
                const Window2D requested{
                    .row = row,
                    .column = column,
                    .height = height,
                    .width = width,
                };

                std::uint64_t local_science_samples = 0;
                std::uint64_t local_mask_samples = 0;
                std::uint64_t local_source_bytes = 0;
                const auto science_plan = product.make_read_plan(
                    science->path, row, column, height, width);
                if (science_plan.expected_bytes > scratch.size()) {
                    fail(
                        "science read exceeds planned fixed scratch storage");
                }
                product.read_direct_chunk_into(
                    science_plan,
                    std::span<std::byte>(
                        scratch.data(), science_plan.expected_bytes));
                copy_sampled_block(
                    std::span<std::byte>(science_output),
                    std::span<const std::byte>(
                        scratch.data(), science_plan.expected_bytes),
                    science_plan,
                    requested,
                    plan.layout,
                    science->data_type.element_size,
                    local_science_samples);
                local_source_bytes = checked_u64_add(
                    local_source_bytes,
                    science_plan.expected_bytes,
                    "source bytes read");

                if (mask != nullptr) {
                    const auto mask_plan = product.make_read_plan(
                        mask->path, row, column, height, width);
                    if (mask_plan.expected_bytes > scratch.size()) {
                        fail(
                            "mask read exceeds planned fixed scratch storage");
                    }
                    product.read_direct_chunk_into(
                        mask_plan,
                        std::span<std::byte>(
                            scratch.data(), mask_plan.expected_bytes));
                    copy_sampled_block(
                        std::span<std::byte>(mask_output),
                        std::span<const std::byte>(
                            scratch.data(), mask_plan.expected_bytes),
                        mask_plan,
                        requested,
                        plan.layout,
                        mask->data_type.element_size,
                        local_mask_samples);
                    local_source_bytes = checked_u64_add(
                        local_source_bytes,
                        mask_plan.expected_bytes,
                        "source bytes read");
                }

                science_samples_written.fetch_add(
                    local_science_samples,
                    std::memory_order_relaxed);
                mask_samples_written.fetch_add(
                    local_mask_samples,
                    std::memory_order_relaxed);
                {
                    std::lock_guard lock(callback_mutex);
                    if (stop_workers.load(
                            std::memory_order_acquire)) {
                        return;
                    }
                    ++current.completed_source_chunks;
                    current.source_bytes_read = checked_u64_add(
                        current.source_bytes_read,
                        local_source_bytes,
                        "source bytes read");
                    if (progress) {
                        progress(current);
                    }
                    if (cancelled(cancel)) {
                        build_cancelled.store(
                            true, std::memory_order_release);
                        stop_workers.store(
                            true, std::memory_order_release);
                        return;
                    }
                }
            }
        } catch (...) {
            record_failure();
        }
    };

    if (worker_count == 1) {
        build_source_chunks(std::stop_token{});
    } else {
        std::vector<std::jthread> workers;
        workers.reserve(worker_count);
        for (std::size_t worker = 0;
             worker < worker_count;
             ++worker) {
            workers.emplace_back(build_source_chunks);
        }
        for (auto& worker : workers) {
            worker.join();
        }
    }

    {
        std::lock_guard lock(failure_mutex);
        if (first_failure != nullptr) {
            std::rethrow_exception(first_failure);
        }
    }
    if (build_cancelled.load(std::memory_order_acquire) ||
        cancelled(cancel)) {
        return false;
    }

    const auto expected_samples = checked_u64_product(
        plan.layout.output_dimensions[0],
        plan.layout.output_dimensions[1],
        "overview output sample count");
    if (science_samples_written.load(std::memory_order_relaxed) !=
            expected_samples ||
        (mask != nullptr &&
         mask_samples_written.load(std::memory_order_relaxed) !=
             expected_samples)) {
        fail("not every overview sample was written exactly once");
    }
    if (cancelled(cancel)) {
        return false;
    }

    const auto current_plan =
        make_overview_plan(product, plan.request);
    if (current_plan.validation_metadata != plan.validation_metadata ||
        current_plan.layout != plan.layout) {
        fail("source changed while the overview was being built");
    }
    return !cancelled(cancel);
}

[[nodiscard]] bool same_window(
    const Window2D& left,
    const Window2D& right) noexcept {
    return left.row == right.row && left.column == right.column &&
        left.height == right.height && left.width == right.width;
}

[[nodiscard]] bool same_optional_window(
    const std::optional<Window2D>& left,
    const std::optional<Window2D>& right) noexcept {
    if (left.has_value() != right.has_value()) {
        return false;
    }
    return !left.has_value() || same_window(*left, *right);
}

[[nodiscard]] bool same_request(
    const OverviewRequest& left,
    const OverviewRequest& right) {
    return left.science_dataset_path == right.science_dataset_path &&
        left.mask_dataset_path == right.mask_dataset_path &&
        left.maximum_long_edge == right.maximum_long_edge &&
        left.maximum_output_bytes == right.maximum_output_bytes &&
        left.maximum_scratch_bytes == right.maximum_scratch_bytes &&
        same_optional_window(left.source_window, right.source_window) &&
        left.sample_stride == right.sample_stride;
}

}  // namespace

double OverviewProgress::fraction() const noexcept {
    if (total_source_chunks == 0) {
        return 0.0;
    }
    return static_cast<double>(completed_source_chunks) /
        static_cast<double>(total_source_chunks);
}

OverviewPlan make_overview_plan(
    const Hdf5Product& product,
    const OverviewRequest& request) {
    if (request.maximum_long_edge == 0 ||
        request.maximum_long_edge > kMaximumOverviewLongEdge) {
        fail("maximum long edge must be in [1, 4096]");
    }
    if (request.maximum_output_bytes == 0 ||
        request.maximum_scratch_bytes == 0) {
        fail("overview memory limits must be non-zero");
    }

    const auto* science =
        product.find_dataset(request.science_dataset_path);
    if (science == nullptr || !science->data_type.readable ||
        science->dimensions[0] == 0 || science->dimensions[1] == 0 ||
        science->data_type.element_size == 0) {
        fail("science dataset is missing or unreadable");
    }

    const DatasetInfo* mask = nullptr;
    if (request.mask_dataset_path.has_value()) {
        mask = product.find_dataset(*request.mask_dataset_path);
        if (mask == nullptr || !mask->data_type.readable ||
            mask->dimensions != science->dimensions ||
            mask->data_type.kind != ScalarKind::unsigned_integer ||
            mask->data_type.element_size != sizeof(std::uint8_t)) {
            fail(
                "mask must be a readable uint8 dataset with the exact "
                "science dimensions");
        }
    }

    const Window2D complete_source{
        .row = 0,
        .column = 0,
        .height = science->dimensions[0],
        .width = science->dimensions[1],
    };
    const auto source_window =
        request.source_window.value_or(complete_source);
    if (source_window.height == 0 || source_window.width == 0) {
        fail("source window must have non-zero width and height");
    }
    if (source_window.row >= science->dimensions[0] ||
        source_window.column >= science->dimensions[1] ||
        source_window.height >
            science->dimensions[0] - source_window.row ||
        source_window.width >
            science->dimensions[1] - source_window.column) {
        fail("source window is outside the science dataset");
    }

    std::uint64_t row_stride = 0;
    std::uint64_t column_stride = 0;
    if (request.sample_stride.has_value()) {
        row_stride = (*request.sample_stride)[0];
        column_stride = (*request.sample_stride)[1];
        if (row_stride == 0 || column_stride == 0) {
            fail("explicit sample strides must be positive");
        }
    } else {
        const auto* frequency =
            product.find_frequency(science->frequency);
        const auto row_spacing = frequency == nullptr
            ? 0.0
            : std::abs(frequency->grid.y.spacing);
        const auto column_spacing = frequency == nullptr
            ? 0.0
            : std::abs(frequency->grid.x.spacing);
        if (std::isfinite(row_spacing) && row_spacing > 0.0 &&
            std::isfinite(column_spacing) && column_spacing > 0.0) {
            const auto edge =
                static_cast<long double>(request.maximum_long_edge);
            const auto target_spacing = std::max(
                static_cast<long double>(row_spacing) *
                    static_cast<long double>(source_window.height) / edge,
                static_cast<long double>(column_spacing) *
                    static_cast<long double>(source_window.width) / edge);
            row_stride = checked_ceil_stride(
                target_spacing / static_cast<long double>(row_spacing));
            column_stride = checked_ceil_stride(
                target_spacing /
                static_cast<long double>(column_spacing));
        } else {
            const auto fallback_stride = ceil_div(
                std::max(source_window.height, source_window.width),
                request.maximum_long_edge);
            row_stride = fallback_stride;
            column_stride = fallback_stride;
        }
        row_stride = std::max(
            row_stride,
            ceil_div(
                source_window.height, request.maximum_long_edge));
        column_stride = std::max(
            column_stride,
            ceil_div(
                source_window.width, request.maximum_long_edge));
    }

    OverviewLayout layout;
    layout.source_origin = {source_window.row, source_window.column};
    layout.source_dimensions = {
        source_window.height, source_window.width};
    layout.sample_stride = {row_stride, column_stride};
    layout.output_dimensions = {
        ceil_div(source_window.height, row_stride),
        ceil_div(source_window.width, column_stride)};
    if (layout.output_dimensions[0] > request.maximum_long_edge ||
        layout.output_dimensions[1] > request.maximum_long_edge) {
        fail("explicit sample strides exceed the maximum output edge");
    }
    layout.science_type = science->data_type;
    layout.science_bytes = checked_size(
        layout.output_dimensions[0],
        layout.output_dimensions[1],
        science->data_type.element_size,
        "overview science allocation");
    layout.mask_bytes = mask == nullptr
        ? 0
        : checked_size(
              layout.output_dimensions[0],
              layout.output_dimensions[1],
              mask->data_type.element_size,
              "overview mask allocation");
    if (layout.science_bytes >
        request.maximum_output_bytes - std::min(
            request.maximum_output_bytes, layout.mask_bytes)) {
        fail("overview output exceeds its configured memory limit");
    }

    const auto block = science->chunk_dimensions.value_or(
        std::array<std::uint64_t, 2>{
            std::min<std::uint64_t>(science->dimensions[0], 512),
            std::min<std::uint64_t>(science->dimensions[1], 512)});
    const auto source_row_end = checked_u64_add(
        source_window.row,
        source_window.height,
        "regional source row end");
    const auto source_column_end = checked_u64_add(
        source_window.column,
        source_window.width,
        "regional source column end");
    const auto block_rows =
        (source_row_end - 1) / block[0] -
        source_window.row / block[0] + 1;
    const auto block_columns =
        (source_column_end - 1) / block[1] -
        source_window.column / block[1] + 1;
    layout.source_chunk_count = checked_u64_product(
        block_rows, block_columns, "source chunk count");

    const auto science_rows =
        std::min(block[0], science->dimensions[0]);
    const auto science_columns =
        std::min(block[1], science->dimensions[1]);
    layout.scratch_bytes = checked_size(
        science_rows,
        science_columns,
        science->data_type.element_size,
        "source science scratch");
    if (mask != nullptr) {
        auto mask_rows = science_rows;
        auto mask_columns = science_columns;
        if (mask->chunk_dimensions.has_value()) {
            mask_rows = maximum_aligned_extent(
                science_rows,
                mask->dimensions[0],
                (*mask->chunk_dimensions)[0]);
            mask_columns = maximum_aligned_extent(
                science_columns,
                mask->dimensions[1],
                (*mask->chunk_dimensions)[1]);
        }
        layout.scratch_bytes = std::max(
            layout.scratch_bytes,
            checked_size(
                mask_rows,
                mask_columns,
                mask->data_type.element_size,
                "source mask scratch"));
    }
    if (layout.scratch_bytes > request.maximum_scratch_bytes) {
        fail("source chunk exceeds the configured scratch-memory limit");
    }

    OverviewPlan result;
    result.request = request;
    result.request.science_dataset_path = science->path;
    if (mask != nullptr) {
        result.request.mask_dataset_path = mask->path;
    }
    if (same_window(source_window, complete_source)) {
        result.request.source_window.reset();
    } else {
        result.request.source_window = source_window;
    }
    result.request.sample_stride = request.sample_stride;
    result.source_file =
        normalized_absolute_path(product.file_path());
    result.layout = layout;
    result.validation_metadata = validation_metadata(
        result.source_file,
        source_identity(result.source_file),
        *science,
        mask,
        layout,
        request.maximum_long_edge);
    return result;
}

OverviewData build_overview(
    const Hdf5Product& product,
    const OverviewPlan& plan,
    OverviewCancelCallback cancel,
    OverviewProgressCallback progress) {
    const auto canonical = make_overview_plan(product, plan.request);
    if (!same_request(canonical.request, plan.request) ||
        canonical.source_file != plan.source_file ||
        canonical.layout != plan.layout ||
        canonical.validation_metadata != plan.validation_metadata) {
        fail("overview plan is stale or non-canonical");
    }

    OverviewData result;
    result.plan = canonical;
    if (cancelled(cancel)) {
        return result;
    }

    const bool built = build_overview_data(
        product,
        canonical,
        cancel,
        progress,
        result.science_bytes,
        result.mask_bytes);
    if (!built || cancelled(cancel)) {
        std::vector<std::byte>().swap(result.science_bytes);
        std::vector<std::byte>().swap(result.mask_bytes);
        return result;
    }
    if (result.science_bytes.size() != canonical.layout.science_bytes ||
        result.mask_bytes.size() != canonical.layout.mask_bytes) {
        fail("newly prepared overview payload has an invalid byte count");
    }
    result.status = OverviewPrepareStatus::built;
    return result;
}

}  // namespace satview::overview
