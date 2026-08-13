#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace satview {

class Hdf5Error final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class ProductType {
    gslc,
    gcov,
};

enum class DatasetRole {
    science,
    auxiliary,
};

enum class LayerKind {
    gslc_polarization,
    gcov_diagonal_covariance,
    gcov_off_diagonal_covariance,
    mask,
    number_of_looks,
    rtc_gamma_to_sigma_factor,
    calibration_lut,
    auxiliary,
};

enum class ScalarKind {
    signed_integer,
    unsigned_integer,
    floating_point,
    compound_complex,
    unsupported,
};

enum class ByteOrder {
    little_endian,
    big_endian,
    native,
    not_applicable,
    unknown,
};

enum class StorageLayout {
    contiguous,
    chunked,
    compact,
    virtual_dataset,
    unknown,
};

struct ComplexLayout {
    std::size_t component_size = 0;
    std::size_t file_real_offset = 0;
    std::size_t file_imaginary_offset = 0;
    std::string real_member_name;
    std::string imaginary_member_name;

    [[nodiscard]] bool operator==(const ComplexLayout&) const = default;
};

// Describes the caller-visible representation produced by read_into().  All
// supported scalar values are native-endian.  Complex values are tightly
// interleaved {real, imaginary} float pairs even if the file compound type is
// padded.
struct DataTypeInfo {
    ScalarKind kind = ScalarKind::unsupported;
    ByteOrder file_byte_order = ByteOrder::unknown;
    std::size_t file_element_size = 0;
    std::size_t element_size = 0;
    // True only when the on-disk bytes are exactly the native representation
    // returned by read_into(), including compound member layout.
    bool file_representation_is_native = false;
    std::optional<ComplexLayout> complex_layout;
    std::string description;
    bool readable = false;

    [[nodiscard]] bool operator==(const DataTypeInfo&) const = default;
};

struct FilterInfo {
    unsigned int id = 0;
    unsigned int flags = 0;
    unsigned int configuration = 0;
    std::string name;
    std::vector<unsigned int> client_data;
};

struct ComplexValue {
    double real = 0.0;
    double imaginary = 0.0;
};

struct FillValue {
    // Raw bytes use the same native-endian representation as read_into().
    std::vector<std::byte> bytes;
    std::optional<double> numeric;
    std::optional<ComplexValue> complex;
};

struct DatasetInfo {
    std::string path;
    std::string name;
    std::string frequency; // "A"/"B" when associated with frequencyA/B.
    DatasetRole role = DatasetRole::auxiliary;
    LayerKind layer_kind = LayerKind::auxiliary;
    DataTypeInfo data_type;
    std::array<std::uint64_t, 2> dimensions{0, 0}; // rows, columns
    std::optional<std::array<std::uint64_t, 2>> chunk_dimensions;
    StorageLayout storage_layout = StorageLayout::unknown;
    std::vector<FilterInfo> filters; // HDF5 pipeline order.
    std::optional<FillValue> creation_fill_value;
    std::optional<FillValue> fill_value_attribute;
    std::string units;
    std::string long_name;
    std::string description;
    std::string grid_mapping;
    std::size_t recommended_chunk_cache_bytes = 0;
};

struct AxisInfo {
    std::string dataset_path;
    std::uint64_t count = 0;
    double first = 0.0;
    double last = 0.0;
    double spacing = 0.0;
    // This is a cheap endpoint consistency check, not a scan of the full axis.
    bool spacing_consistent_with_endpoints = false;
};

struct GridInfo {
    AxisInfo x;
    AxisInfo y;
    std::optional<std::uint32_t> epsg;
    std::string projection_dataset_path;
};

struct FrequencyCatalog {
    std::string name; // "A"/"B"
    std::string group_path;
    GridInfo grid;
    std::vector<std::string> polarizations;
    std::vector<std::string> covariance_terms;
    std::vector<DatasetInfo> layers;
};

struct Identification {
    ProductType product_type = ProductType::gslc;
    std::string product_type_text;
    std::string granule_id;
    std::string product_version;
    std::string product_specification_version;
    std::string instrument_group; // "LSAR" or "SSAR"
    std::string identification_group_path;
    std::string product_group_path;
};

struct Window2D {
    std::uint64_t row = 0;
    std::uint64_t column = 0;
    std::uint64_t height = 0;
    std::uint64_t width = 0;
};

struct ReadPlan {
    std::string source_file;
    std::string dataset_path;
    Window2D requested;
    Window2D aligned;
    std::uint64_t requested_row_offset = 0;
    std::uint64_t requested_column_offset = 0;
    std::array<std::uint64_t, 2> dataset_dimensions{0, 0};
    DataTypeInfo data_type;
    std::size_t expected_bytes = 0;
};

struct Hdf5OpenOptions {
    // Maximum target cache per open dataset.  A single source chunk is always
    // made cacheable even when it is larger than this target.
    std::size_t chunk_cache_bytes = 64ULL * 1024ULL * 1024ULL;
    std::size_t chunk_cache_slots = 4099;
    double chunk_cache_preemption = 0.75;
    // Maximum read dataset handles retained in the LRU.  Each handle keeps its
    // independently configured HDF5 raw chunk cache.
    std::size_t read_dataset_cache_entries = 4;
};

class Hdf5Product final {
public:
    explicit Hdf5Product(
        std::filesystem::path file_path,
        Hdf5OpenOptions options = {});
    ~Hdf5Product();

    Hdf5Product(Hdf5Product&&) = delete;
    Hdf5Product& operator=(Hdf5Product&&) = delete;
    Hdf5Product(const Hdf5Product&) = delete;
    Hdf5Product& operator=(const Hdf5Product&) = delete;

    [[nodiscard]] const std::filesystem::path& file_path() const noexcept;
    [[nodiscard]] const Identification& identification() const noexcept;
    [[nodiscard]] const std::vector<FrequencyCatalog>& frequencies() const noexcept;
    [[nodiscard]] const std::vector<DatasetInfo>& datasets() const noexcept;

    [[nodiscard]] const DatasetInfo* find_dataset(std::string_view path) const;
    [[nodiscard]] const FrequencyCatalog* find_frequency(std::string_view name) const;

    // The returned aligned window is expanded to complete source chunks and
    // clipped at the dataset edge.  For contiguous datasets it is identical to
    // the requested window.
    [[nodiscard]] ReadPlan make_read_plan(
        std::string_view dataset_path,
        std::uint64_t row,
        std::uint64_t column,
        std::uint64_t height,
        std::uint64_t width) const;

    // Cheap format/shape predicate for the exact direct-chunk decoder. It does
    // not query whether this particular chunk is physically allocated;
    // read_direct_chunk_into() safely falls back to H5Dread for an absent
    // chunk.
    [[nodiscard]] bool supports_direct_chunk_decode(
        const ReadPlan& plan) const;

    // Reads exactly plan.aligned into caller memory.  The span size must equal
    // plan.expected_bytes.  No raster-sized allocation is made internally.
    void read_into(const ReadPlan& plan, std::span<std::byte> destination) const;

    // Reads through the exact direct-chunk decoder when the plan is eligible,
    // otherwise falls back to read_into(). This is intended for one-pass cold
    // chunk scans; ordinary read_into() retains HDF5's decompressed chunk cache
    // for hot/repeated native-tile reads.
    void read_direct_chunk_into(
        const ReadPlan& plan,
        std::span<std::byte> destination) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace satview
