#include "satview/aligned_reader.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace satview::aligned {
namespace {

constexpr std::size_t kEdgeFingerprintBytes = 64ULL * 1024ULL;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error("aligned reader: " + message);
}

[[nodiscard]] std::uint64_t checked_add(
    const std::uint64_t left,
    const std::uint64_t right,
    const char* const context) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        fail(std::string(context) + " overflow");
    }
    return left + right;
}

[[nodiscard]] std::uint64_t checked_product(
    const std::uint64_t left,
    const std::uint64_t right,
    const char* const context) {
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
    const char* const context) {
    const auto elements = checked_product(rows, columns, context);
    if (element_size != 0 &&
        elements >
            std::numeric_limits<std::size_t>::max() / element_size) {
        fail(std::string(context) + " exceeds size_t");
    }
    return static_cast<std::size_t>(elements) * element_size;
}

[[nodiscard]] std::size_t checked_size_add(
    const std::size_t left,
    const std::size_t right,
    const char* const context) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        fail(std::string(context) + " exceeds size_t");
    }
    return left + right;
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

[[nodiscard]] bool same_window(
    const Window2D& left,
    const Window2D& right) noexcept {
    return left.row == right.row && left.column == right.column &&
        left.height == right.height && left.width == right.width;
}

[[nodiscard]] bool same_optional_window(
    const std::optional<Window2D>& left,
    const std::optional<Window2D>& right) noexcept {
    return left.has_value() == right.has_value() &&
        (!left.has_value() || same_window(*left, *right));
}

[[nodiscard]] bool same_request(
    const AlignedRequest& left,
    const AlignedRequest& right) {
    if (left.rasters.size() != right.rasters.size() ||
        !same_optional_window(left.source_window, right.source_window) ||
        left.sample_stride != right.sample_stride ||
        left.maximum_output_bytes != right.maximum_output_bytes ||
        left.maximum_scratch_bytes != right.maximum_scratch_bytes ||
        left.prefer_direct_chunk_decode !=
            right.prefer_direct_chunk_decode) {
        return false;
    }
    for (std::size_t index = 0; index < left.rasters.size(); ++index) {
        if (left.rasters[index].dataset_path !=
                right.rasters[index].dataset_path ||
            left.rasters[index].mask_dataset_path !=
                right.rasters[index].mask_dataset_path) {
            return false;
        }
    }
    return true;
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
        fail("could not normalize source path: " + path.string());
    }
    return result.lexically_normal();
}

[[nodiscard]] std::string path_utf8(
    const std::filesystem::path& path) {
    const auto encoded = path.generic_u8string();
    return {
        reinterpret_cast<const char*>(encoded.data()),
        encoded.size()};
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
    const auto read_region = [&](const std::uint64_t offset,
                                 const std::size_t requested) {
        if (offset > static_cast<std::uint64_t>(
                         std::numeric_limits<std::streamoff>::max())) {
            fail("source offset is not representable while fingerprinting");
        }
        stream.clear();
        stream.seekg(static_cast<std::streamoff>(offset));
        if (!stream) {
            fail("could not seek source while fingerprinting");
        }
        stream.read(
            reinterpret_cast<char*>(buffer.data()),
            static_cast<std::streamsize>(requested));
        if (stream.gcount() < 0 ||
            static_cast<std::size_t>(stream.gcount()) != requested) {
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
            std::span<const std::byte>(buffer.data(), requested));
    };

    const auto first_bytes = static_cast<std::size_t>(
        std::min<std::uint64_t>(size, buffer.size()));
    read_region(0, first_bytes);
    if (size > first_bytes) {
        const auto last_bytes = static_cast<std::size_t>(
            std::min<std::uint64_t>(size - first_bytes, buffer.size()));
        read_region(size - last_bytes, last_bytes);
    }
    return {
        .size = size,
        .modified = static_cast<std::uint64_t>(
            modified.time_since_epoch().count()),
        .edge_hash = hash,
    };
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

    void boolean(const bool value) {
        u8(static_cast<std::uint8_t>(value));
    }

    void floating(const double value) {
        u64(std::bit_cast<std::uint64_t>(value));
    }

    void string(const std::string_view value) {
        u64(static_cast<std::uint64_t>(value.size()));
        const auto* const begin =
            reinterpret_cast<const std::byte*>(value.data());
        bytes_.insert(bytes_.end(), begin, begin + value.size());
    }

    void bytes(const std::span<const std::byte> value) {
        u64(static_cast<std::uint64_t>(value.size()));
        bytes_.insert(bytes_.end(), value.begin(), value.end());
    }

    [[nodiscard]] std::vector<std::byte> finish() && {
        return std::move(bytes_);
    }

private:
    std::vector<std::byte> bytes_;
};

void append_data_type(
    MetadataWriter& writer,
    const DataTypeInfo& type) {
    writer.u32(static_cast<std::uint32_t>(type.kind));
    writer.u32(static_cast<std::uint32_t>(type.file_byte_order));
    writer.u64(static_cast<std::uint64_t>(type.file_element_size));
    writer.u64(static_cast<std::uint64_t>(type.element_size));
    writer.boolean(type.file_representation_is_native);
    writer.boolean(type.readable);
    writer.string(type.description);
    writer.boolean(type.complex_layout.has_value());
    if (type.complex_layout.has_value()) {
        const auto& complex = *type.complex_layout;
        writer.u64(static_cast<std::uint64_t>(complex.component_size));
        writer.u64(static_cast<std::uint64_t>(complex.file_real_offset));
        writer.u64(static_cast<std::uint64_t>(complex.file_imaginary_offset));
        writer.string(complex.real_member_name);
        writer.string(complex.imaginary_member_name);
    }
}

void append_fill_value(
    MetadataWriter& writer,
    const std::optional<FillValue>& fill) {
    writer.boolean(fill.has_value());
    if (!fill.has_value()) {
        return;
    }
    writer.bytes(fill->bytes);
    writer.boolean(fill->numeric.has_value());
    if (fill->numeric.has_value()) {
        writer.floating(*fill->numeric);
    }
    writer.boolean(fill->complex.has_value());
    if (fill->complex.has_value()) {
        writer.floating(fill->complex->real);
        writer.floating(fill->complex->imaginary);
    }
}

void append_dataset(
    MetadataWriter& writer,
    const DatasetInfo& dataset) {
    writer.string(dataset.path);
    writer.string(dataset.name);
    writer.string(dataset.frequency);
    writer.u32(static_cast<std::uint32_t>(dataset.role));
    writer.u32(static_cast<std::uint32_t>(dataset.layer_kind));
    append_data_type(writer, dataset.data_type);
    writer.u64(dataset.dimensions[0]);
    writer.u64(dataset.dimensions[1]);
    writer.boolean(dataset.chunk_dimensions.has_value());
    if (dataset.chunk_dimensions.has_value()) {
        writer.u64((*dataset.chunk_dimensions)[0]);
        writer.u64((*dataset.chunk_dimensions)[1]);
    }
    writer.u32(static_cast<std::uint32_t>(dataset.storage_layout));
    writer.u64(static_cast<std::uint64_t>(dataset.filters.size()));
    for (const auto& filter : dataset.filters) {
        writer.u32(filter.id);
        writer.u32(filter.flags);
        writer.u32(filter.configuration);
        writer.string(filter.name);
        writer.u64(static_cast<std::uint64_t>(filter.client_data.size()));
        for (const auto value : filter.client_data) {
            writer.u32(value);
        }
    }
    append_fill_value(writer, dataset.creation_fill_value);
    append_fill_value(writer, dataset.fill_value_attribute);
    writer.string(dataset.units);
    writer.string(dataset.long_name);
    writer.string(dataset.description);
    writer.string(dataset.grid_mapping);
    writer.u64(static_cast<std::uint64_t>(
        dataset.recommended_chunk_cache_bytes));
}

void append_axis(MetadataWriter& writer, const AxisInfo& axis) {
    writer.string(axis.dataset_path);
    writer.u64(axis.count);
    writer.floating(axis.first);
    writer.floating(axis.last);
    writer.floating(axis.spacing);
    writer.boolean(axis.spacing_consistent_with_endpoints);
}

void append_grid(MetadataWriter& writer, const GridInfo& grid) {
    append_axis(writer, grid.x);
    append_axis(writer, grid.y);
    writer.boolean(grid.epsg.has_value());
    if (grid.epsg.has_value()) {
        writer.u32(*grid.epsg);
    }
    writer.string(grid.projection_dataset_path);
}

[[nodiscard]] std::vector<std::byte> validation_metadata(
    const std::filesystem::path& source_file,
    const SourceIdentity& identity,
    const FrequencyCatalog& frequency,
    const AlignedRequest& request,
    const AlignedLayout& layout,
    const std::vector<const DatasetInfo*>& rasters,
    const std::vector<const DatasetInfo*>& masks) {
    MetadataWriter writer;
    writer.string("satview-aligned-multi-raster-plan-v1");
    writer.string(path_utf8(source_file));
    writer.u64(identity.size);
    writer.u64(identity.modified);
    writer.u64(identity.edge_hash);
    writer.string(frequency.name);
    writer.string(frequency.group_path);
    append_grid(writer, frequency.grid);
    writer.u64(layout.source_origin[0]);
    writer.u64(layout.source_origin[1]);
    writer.u64(layout.source_dimensions[0]);
    writer.u64(layout.source_dimensions[1]);
    writer.u64(layout.sample_stride[0]);
    writer.u64(layout.sample_stride[1]);
    writer.u64(layout.output_dimensions[0]);
    writer.u64(layout.output_dimensions[1]);
    writer.u64(static_cast<std::uint64_t>(request.maximum_output_bytes));
    writer.u64(static_cast<std::uint64_t>(request.maximum_scratch_bytes));
    writer.boolean(request.prefer_direct_chunk_decode);
    writer.u64(static_cast<std::uint64_t>(rasters.size()));
    for (const auto* const raster : rasters) {
        append_dataset(writer, *raster);
    }
    writer.u64(static_cast<std::uint64_t>(masks.size()));
    for (const auto* const mask : masks) {
        append_dataset(writer, *mask);
    }
    return std::move(writer).finish();
}

[[nodiscard]] std::uint64_t maximum_aligned_extent(
    const std::uint64_t request_extent,
    const std::uint64_t dimension,
    const std::uint64_t chunk_extent) {
    if (request_extent == 0 || dimension == 0 || chunk_extent == 0) {
        fail("aligned-extent arguments must be positive");
    }
    if (request_extent >= dimension) {
        return dimension;
    }
    const auto shifted = checked_add(
        request_extent, chunk_extent - 1, "aligned extent");
    const auto chunks = ceil_div(shifted, chunk_extent);
    if (chunks > dimension / chunk_extent) {
        return dimension;
    }
    return std::min(dimension, chunks * chunk_extent);
}

[[nodiscard]] bool cancelled(const CancelCallback& callback) {
    return callback && callback();
}

struct SelectedAxis {
    std::uint64_t first_source = 0;
    std::uint64_t destination = 0;
    std::uint64_t count = 0;
};

[[nodiscard]] SelectedAxis selected_axis(
    const std::uint64_t start,
    const std::uint64_t extent,
    const std::uint64_t origin,
    const std::uint64_t stride) {
    if (extent == 0) {
        return {};
    }
    if (start < origin) {
        fail("source block precedes the sampling origin");
    }
    const auto end = checked_add(start, extent, "source block end");
    const auto first_index = ceil_div(start - origin, stride);
    const auto last_index = (end - 1 - origin) / stride;
    if (first_index > last_index) {
        return {};
    }
    return {
        .first_source = checked_add(
            origin,
            checked_product(first_index, stride, "sample coordinate"),
            "sample coordinate"),
        .destination = first_index,
        .count = last_index - first_index + 1,
    };
}

void copy_sampled_member(
    const std::span<std::byte> destination,
    const std::span<const std::byte> source,
    const ReadPlan& read_plan,
    const SelectedAxis& rows,
    const SelectedAxis& columns,
    const std::array<std::uint64_t, 2> stride,
    const std::size_t element_size) {
    const auto expected = checked_size(
        rows.count, columns.count, element_size, "callback block payload");
    if (destination.size() != expected ||
        source.size() != read_plan.expected_bytes ||
        source.size() < element_size ||
        rows.first_source < read_plan.aligned.row ||
        columns.first_source < read_plan.aligned.column) {
        fail("invalid sampled-member copy bounds");
    }
    const auto source_last_row = checked_add(
        rows.first_source,
        checked_product(
            rows.count - 1, stride[0], "last sampled row"),
        "last sampled row");
    const auto source_last_column = checked_add(
        columns.first_source,
        checked_product(
            columns.count - 1, stride[1], "last sampled column"),
        "last sampled column");
    if (source_last_row >=
            checked_add(
                read_plan.aligned.row,
                read_plan.aligned.height,
                "aligned row end") ||
        source_last_column >=
            checked_add(
                read_plan.aligned.column,
                read_plan.aligned.width,
                "aligned column end")) {
        fail("sample lies outside its member read plan");
    }

    auto* output = destination.data();
    for (std::uint64_t row = 0; row < rows.count; ++row) {
        const auto source_row = checked_add(
            rows.first_source,
            checked_product(row, stride[0], "sampled row"),
            "sampled row");
        for (std::uint64_t column = 0;
             column < columns.count;
             ++column) {
            const auto source_column = checked_add(
                columns.first_source,
                checked_product(column, stride[1], "sampled column"),
                "sampled column");
            const auto source_element = checked_add(
                checked_product(
                    source_row - read_plan.aligned.row,
                    read_plan.aligned.width,
                    "source element"),
                source_column - read_plan.aligned.column,
                "source element");
            const auto source_offset = checked_size(
                1, source_element, element_size, "source byte offset");
            if (source_offset > source.size() - element_size) {
                fail("sample byte offset lies outside its member buffer");
            }
            std::memcpy(output, source.data() + source_offset, element_size);
            output += element_size;
        }
    }
}

[[nodiscard]] bool same_plan(
    const AlignedPlan& left,
    const AlignedPlan& right) {
    return same_request(left.request, right.request) &&
        left.source_file == right.source_file &&
        left.frequency == right.frequency &&
        left.layout == right.layout &&
        left.rasters == right.rasters && left.masks == right.masks &&
        left.validation_metadata == right.validation_metadata;
}

}  // namespace

double AlignedProgress::fraction() const noexcept {
    if (total_source_blocks == 0) {
        return 0.0;
    }
    return static_cast<double>(completed_source_blocks) /
        static_cast<double>(total_source_blocks);
}

AlignedPlan make_aligned_plan(
    const Hdf5Product& product,
    const AlignedRequest& request) {
    if (request.rasters.empty() ||
        request.rasters.size() > kMaximumScienceRasterCount) {
        fail("science raster count must be in [1, 4]");
    }
    if (request.maximum_output_bytes == 0 ||
        request.maximum_scratch_bytes == 0) {
        fail("output and scratch byte limits must be non-zero");
    }
    if (request.sample_stride[0] == 0 ||
        request.sample_stride[1] == 0) {
        fail("sample strides must be positive");
    }

    std::vector<const DatasetInfo*> raster_descriptors;
    raster_descriptors.reserve(request.rasters.size());
    std::unordered_set<std::string> raster_paths;
    const DatasetInfo* driver = nullptr;
    const FrequencyCatalog* frequency = nullptr;
    for (const auto& requested : request.rasters) {
        const auto* const descriptor =
            product.find_dataset(requested.dataset_path);
        if (descriptor == nullptr ||
            descriptor->role != DatasetRole::science ||
            descriptor->layer_kind == LayerKind::mask ||
            !descriptor->data_type.readable ||
            descriptor->data_type.element_size == 0 ||
            descriptor->dimensions[0] == 0 ||
            descriptor->dimensions[1] == 0) {
            fail("science raster is missing, empty, or unreadable: " +
                 requested.dataset_path);
        }
        if (!raster_paths.insert(descriptor->path).second) {
            fail("science raster paths must be unique: " + descriptor->path);
        }
        if (driver == nullptr) {
            driver = descriptor;
            frequency = product.find_frequency(descriptor->frequency);
            if (descriptor->frequency.empty() || frequency == nullptr) {
                fail("science raster has no cataloged frequency grid: " +
                     descriptor->path);
            }
            if (frequency->grid.y.count != descriptor->dimensions[0] ||
                frequency->grid.x.count != descriptor->dimensions[1]) {
                fail("science dimensions do not match their frequency grid: " +
                     descriptor->path);
            }
        } else if (descriptor->frequency != driver->frequency ||
                   descriptor->dimensions != driver->dimensions ||
                   descriptor->grid_mapping != driver->grid_mapping) {
            fail(
                "all science rasters must share one grid, dimensions, and "
                "grid-mapping metadata");
        }
        raster_descriptors.push_back(descriptor);
    }
    if (driver == nullptr || frequency == nullptr) {
        fail("science driver was not resolved");
    }

    std::vector<const DatasetInfo*> mask_descriptors;
    std::unordered_map<std::string, std::size_t> mask_indices;
    std::vector<std::optional<std::size_t>> raster_mask_indices;
    raster_mask_indices.reserve(request.rasters.size());
    for (const auto& requested : request.rasters) {
        if (!requested.mask_dataset_path.has_value()) {
            raster_mask_indices.push_back(std::nullopt);
            continue;
        }
        const auto* const mask =
            product.find_dataset(*requested.mask_dataset_path);
        if (mask == nullptr || mask->layer_kind != LayerKind::mask ||
            !mask->data_type.readable ||
            mask->data_type.kind != ScalarKind::unsigned_integer ||
            mask->data_type.element_size != sizeof(std::uint8_t) ||
            mask->frequency != driver->frequency ||
            mask->dimensions != driver->dimensions ||
            mask->grid_mapping != driver->grid_mapping) {
            fail(
                "each mask must be a readable uint8 layer on the exact "
                "science frequency grid: " + *requested.mask_dataset_path);
        }
        const auto found = mask_indices.find(mask->path);
        if (found != mask_indices.end()) {
            raster_mask_indices.push_back(found->second);
            continue;
        }
        const auto index = mask_descriptors.size();
        mask_descriptors.push_back(mask);
        mask_indices.emplace(mask->path, index);
        raster_mask_indices.push_back(index);
    }

    const Window2D complete{
        .row = 0,
        .column = 0,
        .height = driver->dimensions[0],
        .width = driver->dimensions[1],
    };
    const auto source_window = request.source_window.value_or(complete);
    if (source_window.height == 0 || source_window.width == 0 ||
        source_window.row >= driver->dimensions[0] ||
        source_window.column >= driver->dimensions[1] ||
        source_window.height >
            driver->dimensions[0] - source_window.row ||
        source_window.width >
            driver->dimensions[1] - source_window.column) {
        fail("source window lies outside the common science grid");
    }

    AlignedLayout layout;
    layout.source_origin = {source_window.row, source_window.column};
    layout.source_dimensions = {
        source_window.height, source_window.width};
    layout.sample_stride = request.sample_stride;
    layout.output_dimensions = {
        ceil_div(source_window.height, request.sample_stride[0]),
        ceil_div(source_window.width, request.sample_stride[1]),
    };
    layout.driver_block_dimensions =
        driver->chunk_dimensions.value_or(
            std::array<std::uint64_t, 2>{
                std::min<std::uint64_t>(driver->dimensions[0], 512),
                std::min<std::uint64_t>(driver->dimensions[1], 512)});
    if (layout.driver_block_dimensions[0] == 0 ||
        layout.driver_block_dimensions[1] == 0) {
        fail("science driver has an invalid source block shape");
    }

    const auto source_row_end = checked_add(
        source_window.row, source_window.height, "source row end");
    const auto source_column_end = checked_add(
        source_window.column, source_window.width, "source column end");
    const auto first_block_row =
        source_window.row -
        source_window.row % layout.driver_block_dimensions[0];
    const auto first_block_column =
        source_window.column -
        source_window.column % layout.driver_block_dimensions[1];
    const auto block_rows = ceil_div(
        source_row_end - first_block_row,
        layout.driver_block_dimensions[0]);
    const auto block_columns = ceil_div(
        source_column_end - first_block_column,
        layout.driver_block_dimensions[1]);
    layout.source_block_count = checked_product(
        block_rows, block_columns, "source block count");

    std::size_t bytes_per_output_sample = 0;
    for (const auto* const raster : raster_descriptors) {
        bytes_per_output_sample = checked_size_add(
            bytes_per_output_sample,
            raster->data_type.element_size,
            "materialized sample size");
    }
    for (const auto* const mask : mask_descriptors) {
        bytes_per_output_sample = checked_size_add(
            bytes_per_output_sample,
            mask->data_type.element_size,
            "materialized sample size");
    }
    layout.materialized_bytes = checked_size(
        layout.output_dimensions[0],
        layout.output_dimensions[1],
        bytes_per_output_sample,
        "materialized aligned output");
    if (layout.materialized_bytes > request.maximum_output_bytes) {
        fail("aligned output exceeds its configured byte limit");
    }

    const auto maximum_requested_rows = std::min(
        source_window.height, layout.driver_block_dimensions[0]);
    const auto maximum_requested_columns = std::min(
        source_window.width, layout.driver_block_dimensions[1]);
    layout.maximum_block_sample_count = checked_product(
        ceil_div(maximum_requested_rows, request.sample_stride[0]),
        ceil_div(maximum_requested_columns, request.sample_stride[1]),
        "maximum callback block samples");

    const auto account_member_read = [&](const DatasetInfo& member) {
        auto rows = maximum_requested_rows;
        auto columns = maximum_requested_columns;
        if (member.chunk_dimensions.has_value()) {
            rows = maximum_aligned_extent(
                rows, member.dimensions[0],
                (*member.chunk_dimensions)[0]);
            columns = maximum_aligned_extent(
                columns, member.dimensions[1],
                (*member.chunk_dimensions)[1]);
        }
        layout.maximum_member_read_bytes = std::max(
            layout.maximum_member_read_bytes,
            checked_size(
                rows, columns, member.data_type.element_size,
                "maximum member read"));
    };
    for (const auto* const raster : raster_descriptors) {
        account_member_read(*raster);
    }
    for (const auto* const mask : mask_descriptors) {
        account_member_read(*mask);
    }
    const auto compact_block_bytes = checked_size(
        1,
        layout.maximum_block_sample_count,
        bytes_per_output_sample,
        "maximum compact callback block");
    layout.scratch_bytes = checked_size_add(
        layout.maximum_member_read_bytes,
        compact_block_bytes,
        "explicit aligned-reader scratch");
    if (layout.scratch_bytes > request.maximum_scratch_bytes) {
        fail("aligned source block exceeds its configured scratch limit");
    }

    AlignedPlan result;
    result.request = request;
    result.request.rasters.clear();
    result.request.rasters.reserve(raster_descriptors.size());
    result.rasters.reserve(raster_descriptors.size());
    for (std::size_t index = 0;
         index < raster_descriptors.size();
         ++index) {
        const auto* const raster = raster_descriptors[index];
        RasterRequest canonical_request{
            .dataset_path = raster->path,
            .mask_dataset_path = std::nullopt,
        };
        if (raster_mask_indices[index].has_value()) {
            canonical_request.mask_dataset_path =
                mask_descriptors[*raster_mask_indices[index]]->path;
        }
        result.request.rasters.push_back(std::move(canonical_request));
        result.rasters.push_back({
            .dataset_path = raster->path,
            .data_type = raster->data_type,
            .mask_index = raster_mask_indices[index],
        });
    }
    if (same_window(source_window, complete)) {
        result.request.source_window.reset();
    } else {
        result.request.source_window = source_window;
    }
    result.masks.reserve(mask_descriptors.size());
    for (const auto* const mask : mask_descriptors) {
        result.masks.push_back({
            .dataset_path = mask->path,
            .data_type = mask->data_type,
        });
    }
    result.source_file = normalized_absolute_path(product.file_path());
    result.frequency = driver->frequency;
    result.layout = layout;
    result.validation_metadata = validation_metadata(
        result.source_file,
        source_identity(result.source_file),
        *frequency,
        result.request,
        result.layout,
        raster_descriptors,
        mask_descriptors);
    return result;
}

VisitStatus visit_aligned_blocks(
    const Hdf5Product& product,
    const AlignedPlan& plan,
    BlockVisitor visitor,
    CancelCallback cancel,
    ProgressCallback progress) {
    if (!visitor) {
        fail("a block visitor is required");
    }
    if (cancelled(cancel)) {
        return VisitStatus::cancelled;
    }
    const auto canonical = make_aligned_plan(product, plan.request);
    if (!same_plan(canonical, plan)) {
        fail("aligned plan is stale or non-canonical");
    }
    if (cancelled(cancel)) {
        return VisitStatus::cancelled;
    }

    std::vector<std::byte> read_scratch(
        canonical.layout.maximum_member_read_bytes);
    std::vector<std::vector<std::byte>> raster_payloads;
    raster_payloads.reserve(canonical.rasters.size());
    for (const auto& raster : canonical.rasters) {
        raster_payloads.emplace_back(checked_size(
            1,
            canonical.layout.maximum_block_sample_count,
            raster.data_type.element_size,
            "science callback staging"));
    }
    std::vector<std::vector<std::byte>> mask_payloads;
    mask_payloads.reserve(canonical.masks.size());
    for (const auto& mask : canonical.masks) {
        mask_payloads.emplace_back(checked_size(
            1,
            canonical.layout.maximum_block_sample_count,
            mask.data_type.element_size,
            "mask callback staging"));
    }
    std::vector<MemberBlock> raster_views(canonical.rasters.size());
    std::vector<MemberBlock> mask_views(canonical.masks.size());

    const auto source_row_end = checked_add(
        canonical.layout.source_origin[0],
        canonical.layout.source_dimensions[0],
        "source row end");
    const auto source_column_end = checked_add(
        canonical.layout.source_origin[1],
        canonical.layout.source_dimensions[1],
        "source column end");
    const auto first_block_row =
        canonical.layout.source_origin[0] -
        canonical.layout.source_origin[0] %
            canonical.layout.driver_block_dimensions[0];
    const auto first_block_column =
        canonical.layout.source_origin[1] -
        canonical.layout.source_origin[1] %
            canonical.layout.driver_block_dimensions[1];
    const auto block_columns = ceil_div(
        source_column_end - first_block_column,
        canonical.layout.driver_block_dimensions[1]);

    AlignedProgress current{
        .completed_source_blocks = 0,
        .total_source_blocks = canonical.layout.source_block_count,
        .source_bytes_read = 0,
    };
    const auto finish_block = [&]() {
        ++current.completed_source_blocks;
        if (progress) {
            progress(current);
        }
        return !cancelled(cancel);
    };

    for (std::uint64_t block_index = 0;
         block_index < canonical.layout.source_block_count;
         ++block_index) {
        if (cancelled(cancel)) {
            return VisitStatus::cancelled;
        }
        const auto block_row_index = block_index / block_columns;
        const auto block_column_index = block_index % block_columns;
        const auto native_row = checked_add(
            first_block_row,
            checked_product(
                block_row_index,
                canonical.layout.driver_block_dimensions[0],
                "driver block row"),
            "driver block row");
        const auto native_column = checked_add(
            first_block_column,
            checked_product(
                block_column_index,
                canonical.layout.driver_block_dimensions[1],
                "driver block column"),
            "driver block column");
        const auto native_row_end = std::min(
            source_row_end,
            checked_add(
                native_row,
                canonical.layout.driver_block_dimensions[0],
                "driver block row end"));
        const auto native_column_end = std::min(
            source_column_end,
            checked_add(
                native_column,
                canonical.layout.driver_block_dimensions[1],
                "driver block column end"));
        const Window2D requested{
            .row = std::max(native_row, canonical.layout.source_origin[0]),
            .column = std::max(
                native_column, canonical.layout.source_origin[1]),
            .height = native_row_end -
                std::max(native_row, canonical.layout.source_origin[0]),
            .width = native_column_end -
                std::max(
                    native_column, canonical.layout.source_origin[1]),
        };
        const auto rows = selected_axis(
            requested.row,
            requested.height,
            canonical.layout.source_origin[0],
            canonical.layout.sample_stride[0]);
        const auto columns = selected_axis(
            requested.column,
            requested.width,
            canonical.layout.source_origin[1],
            canonical.layout.sample_stride[1]);
        if (rows.count == 0 || columns.count == 0) {
            if (!finish_block()) {
                return VisitStatus::cancelled;
            }
            continue;
        }
        const auto block_samples = checked_product(
            rows.count, columns.count, "callback block sample count");
        if (block_samples >
            canonical.layout.maximum_block_sample_count) {
            fail("callback block exceeds its planned sample capacity");
        }

        const auto read_member = [&](const std::string& path,
                                     const std::size_t element_size,
                                     std::vector<std::byte>& payload) {
            if (cancelled(cancel)) {
                return false;
            }
            const auto member_plan = product.make_read_plan(
                path,
                requested.row,
                requested.column,
                requested.height,
                requested.width);
            if (member_plan.expected_bytes > read_scratch.size()) {
                fail("member read exceeds its planned scratch capacity");
            }
            const auto read_span = std::span<std::byte>(
                read_scratch.data(), member_plan.expected_bytes);
            if (canonical.request.prefer_direct_chunk_decode &&
                product.supports_direct_chunk_decode(member_plan)) {
                product.read_direct_chunk_into(member_plan, read_span);
            } else {
                product.read_into(member_plan, read_span);
            }
            if (cancelled(cancel)) {
                return false;
            }
            current.source_bytes_read = checked_add(
                current.source_bytes_read,
                static_cast<std::uint64_t>(member_plan.expected_bytes),
                "source bytes read");
            const auto active_bytes = checked_size(
                1, block_samples, element_size,
                "active callback payload");
            copy_sampled_member(
                std::span<std::byte>(payload.data(), active_bytes),
                std::span<const std::byte>(
                    read_scratch.data(), member_plan.expected_bytes),
                member_plan,
                rows,
                columns,
                canonical.layout.sample_stride,
                element_size);
            return !cancelled(cancel);
        };

        for (std::size_t index = 0;
             index < canonical.rasters.size();
             ++index) {
            const auto& raster = canonical.rasters[index];
            if (!read_member(
                    raster.dataset_path,
                    raster.data_type.element_size,
                    raster_payloads[index])) {
                return VisitStatus::cancelled;
            }
            const auto active_bytes = checked_size(
                1, block_samples, raster.data_type.element_size,
                "science callback payload");
            raster_views[index] = {
                .member_index = index,
                .samples = std::span<const std::byte>(
                    raster_payloads[index].data(), active_bytes),
            };
        }
        for (std::size_t index = 0;
             index < canonical.masks.size();
             ++index) {
            const auto& mask = canonical.masks[index];
            if (!read_member(
                    mask.dataset_path,
                    mask.data_type.element_size,
                    mask_payloads[index])) {
                return VisitStatus::cancelled;
            }
            const auto active_bytes = checked_size(
                1, block_samples, mask.data_type.element_size,
                "mask callback payload");
            mask_views[index] = {
                .member_index = index,
                .samples = std::span<const std::byte>(
                    mask_payloads[index].data(), active_bytes),
            };
        }
        if (cancelled(cancel)) {
            return VisitStatus::cancelled;
        }

        const AlignedBlock block{
            .output_origin = {rows.destination, columns.destination},
            .source_sample_origin = {
                rows.first_source, columns.first_source},
            .dimensions = {rows.count, columns.count},
            .rasters = raster_views,
            .masks = mask_views,
        };
        const bool continue_visiting = visitor(block);
        if (cancelled(cancel)) {
            return VisitStatus::cancelled;
        }
        if (!finish_block()) {
            return VisitStatus::cancelled;
        }
        if (!continue_visiting) {
            return VisitStatus::visitor_stopped;
        }
    }

    if (current.completed_source_blocks !=
        canonical.layout.source_block_count) {
        fail("source traversal ended before every driver block completed");
    }
    if (cancelled(cancel)) {
        return VisitStatus::cancelled;
    }
    const auto current_plan = make_aligned_plan(product, canonical.request);
    if (!same_plan(current_plan, canonical)) {
        fail("source changed while aligned blocks were being visited");
    }
    return cancelled(cancel)
        ? VisitStatus::cancelled
        : VisitStatus::completed;
}

}  // namespace satview::aligned
