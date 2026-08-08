#include "satview/hdf5_product.hpp"
#include "satview/overview_builder.hpp"

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

void expect(
    const bool condition,
    const std::string_view message,
    int& failures) {
    if (!condition) {
        ++failures;
        std::cerr << "Overview test failed: " << message << '\n';
    }
}

[[nodiscard]] hid_t checked_h5_id(
    const hid_t value,
    const std::string_view operation) {
    if (value < 0) {
        throw std::runtime_error(
            "synthetic HDF5 fixture failed while " +
            std::string(operation));
    }
    return value;
}

void checked_h5(
    const herr_t value,
    const std::string_view operation) {
    if (value < 0) {
        throw std::runtime_error(
            "synthetic HDF5 fixture failed while " +
            std::string(operation));
    }
}

class H5TestHandle final {
public:
    using Closer = herr_t (*)(hid_t);

    H5TestHandle(const hid_t value, Closer closer) noexcept
        : value_(value), closer_(closer) {}

    ~H5TestHandle() {
        if (value_ >= 0) {
            static_cast<void>(closer_(value_));
        }
    }

    H5TestHandle(const H5TestHandle&) = delete;
    H5TestHandle& operator=(const H5TestHandle&) = delete;
    H5TestHandle(H5TestHandle&& other) noexcept
        : value_(std::exchange(other.value_, H5I_INVALID_HID)),
          closer_(other.closer_) {}

    [[nodiscard]] hid_t get() const noexcept { return value_; }

private:
    hid_t value_ = H5I_INVALID_HID;
    Closer closer_ = nullptr;
};

void create_group(const hid_t file, const char* const path) {
    H5TestHandle group(
        checked_h5_id(
            H5Gcreate2(
                file, path, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
            "creating group"),
        H5Gclose);
}

void write_string(
    const hid_t file,
    const char* const path,
    const std::string_view value) {
    H5TestHandle type(
        checked_h5_id(H5Tcopy(H5T_C_S1), "copying string type"),
        H5Tclose);
    checked_h5(
        H5Tset_size(type.get(), value.size() + 1),
        "sizing string type");
    checked_h5(
        H5Tset_strpad(type.get(), H5T_STR_NULLTERM),
        "setting string padding");
    H5TestHandle space(
        checked_h5_id(H5Screate(H5S_SCALAR), "creating string space"),
        H5Sclose);
    H5TestHandle dataset(
        checked_h5_id(
            H5Dcreate2(
                file,
                path,
                type.get(),
                space.get(),
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT),
            "creating string dataset"),
        H5Dclose);
    std::vector<char> bytes(value.size() + 1, '\0');
    std::memcpy(bytes.data(), value.data(), value.size());
    checked_h5(
        H5Dwrite(
            dataset.get(),
            type.get(),
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            bytes.data()),
        "writing string dataset");
}

void write_axis(
    const hid_t file,
    const char* const path,
    const std::vector<double>& values) {
    const hsize_t count = values.size();
    H5TestHandle space(
        checked_h5_id(
            H5Screate_simple(1, &count, nullptr),
            "creating axis space"),
        H5Sclose);
    H5TestHandle dataset(
        checked_h5_id(
            H5Dcreate2(
                file,
                path,
                H5T_IEEE_F64LE,
                space.get(),
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT),
            "creating axis dataset"),
        H5Dclose);
    checked_h5(
        H5Dwrite(
            dataset.get(),
            H5T_NATIVE_DOUBLE,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            values.data()),
        "writing axis dataset");
}

template <typename Value>
void write_chunked_raster(
    const hid_t file,
    const char* const path,
    const hid_t file_type,
    const hid_t memory_type,
    const std::array<hsize_t, 2> dimensions,
    const std::array<hsize_t, 2> chunks,
    const std::vector<Value>& values) {
    H5TestHandle space(
        checked_h5_id(
            H5Screate_simple(2, dimensions.data(), nullptr),
            "creating raster space"),
        H5Sclose);
    H5TestHandle properties(
        checked_h5_id(
            H5Pcreate(H5P_DATASET_CREATE),
            "creating raster properties"),
        H5Pclose);
    checked_h5(
        H5Pset_chunk(properties.get(), 2, chunks.data()),
        "setting raster chunks");
    checked_h5(
        H5Pset_shuffle(properties.get()),
        "setting raster shuffle filter");
    checked_h5(
        H5Pset_deflate(properties.get(), 1),
        "setting raster DEFLATE filter");
    H5TestHandle dataset(
        checked_h5_id(
            H5Dcreate2(
                file,
                path,
                file_type,
                space.get(),
                H5P_DEFAULT,
                properties.get(),
                H5P_DEFAULT),
            "creating raster dataset"),
        H5Dclose);
    checked_h5(
        H5Dwrite(
            dataset.get(),
            memory_type,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            values.data()),
        "writing raster dataset");
}

class TemporaryDirectory final {
public:
    TemporaryDirectory() {
        const auto stamp = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        path_ = std::filesystem::temp_directory_path() /
            ("satview-overview-test-" + std::to_string(stamp));
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

constexpr std::uint64_t kSyntheticRows = 10;
constexpr std::uint64_t kSyntheticColumns = 13;

[[nodiscard]] float synthetic_science(
    const std::uint64_t row,
    const std::uint64_t column) {
    return static_cast<float>(row * 100 + column) + 0.25F;
}

[[nodiscard]] std::uint8_t synthetic_mask(
    const std::uint64_t row,
    const std::uint64_t column) {
    if (row == 5 && column == 7) {
        return 0;
    }
    if (row == 7 && column == 10) {
        return 255;
    }
    return static_cast<std::uint8_t>(
        1 + (row * kSyntheticColumns + column) % 253);
}

[[nodiscard]] std::filesystem::path make_synthetic_product(
    const std::filesystem::path& directory) {
    const auto path = directory / "synthetic-regional.h5";
    {
        H5TestHandle file(
            checked_h5_id(
                H5Fcreate(
                    path.string().c_str(),
                    H5F_ACC_TRUNC,
                    H5P_DEFAULT,
                    H5P_DEFAULT),
                "creating fixture file"),
            H5Fclose);
        create_group(file.get(), "/science");
        create_group(file.get(), "/science/LSAR");
        create_group(file.get(), "/science/LSAR/identification");
        create_group(file.get(), "/science/LSAR/GCOV");
        create_group(file.get(), "/science/LSAR/GCOV/grids");
        create_group(
            file.get(), "/science/LSAR/GCOV/grids/frequencyA");

        write_string(
            file.get(),
            "/science/LSAR/identification/productType",
            "GCOV");
        write_string(
            file.get(),
            "/science/LSAR/identification/granuleId",
            "synthetic-regional");
        write_string(
            file.get(),
            "/science/LSAR/identification/productVersion",
            "test");
        write_string(
            file.get(),
            "/science/LSAR/GCOV/grids/frequencyA/"
            "listOfCovarianceTerms",
            "HHHH");

        std::vector<double> x(kSyntheticColumns);
        std::vector<double> y(kSyntheticRows);
        for (std::size_t index = 0; index < x.size(); ++index) {
            x[index] = 1'000.0 + 2.0 * static_cast<double>(index);
        }
        for (std::size_t index = 0; index < y.size(); ++index) {
            y[index] = 2'000.0 - 3.0 * static_cast<double>(index);
        }
        write_axis(
            file.get(),
            "/science/LSAR/GCOV/grids/frequencyA/xCoordinates",
            x);
        write_axis(
            file.get(),
            "/science/LSAR/GCOV/grids/frequencyA/yCoordinates",
            y);

        std::vector<float> science(
            static_cast<std::size_t>(
                kSyntheticRows * kSyntheticColumns));
        std::vector<std::uint8_t> mask(science.size());
        for (std::uint64_t row = 0; row < kSyntheticRows; ++row) {
            for (std::uint64_t column = 0;
                 column < kSyntheticColumns;
                 ++column) {
                const auto index = static_cast<std::size_t>(
                    row * kSyntheticColumns + column);
                science[index] = synthetic_science(row, column);
                mask[index] = synthetic_mask(row, column);
            }
        }
        constexpr std::array<hsize_t, 2> dimensions{
            kSyntheticRows, kSyntheticColumns};
        constexpr std::array<hsize_t, 2> chunks{4, 5};
        write_chunked_raster(
            file.get(),
            "/science/LSAR/GCOV/grids/frequencyA/HHHH",
            H5T_IEEE_F32LE,
            H5T_NATIVE_FLOAT,
            dimensions,
            chunks,
            science);
        write_chunked_raster(
            file.get(),
            "/science/LSAR/GCOV/grids/frequencyA/mask",
            H5T_STD_U8LE,
            H5T_NATIVE_UINT8,
            dimensions,
            chunks,
            mask);
    }
    return path;
}

void expect_synthetic_payload(
    const satview::overview::OverviewData& data,
    int& failures) {
    const auto& layout = data.plan.layout;
    for (std::uint64_t output_row = 0;
         output_row < layout.output_dimensions[0];
         ++output_row) {
        for (std::uint64_t output_column = 0;
             output_column < layout.output_dimensions[1];
             ++output_column) {
            const auto source_row = layout.source_origin[0] +
                output_row * layout.sample_stride[0];
            const auto source_column = layout.source_origin[1] +
                output_column * layout.sample_stride[1];
            const auto index = static_cast<std::size_t>(
                output_row * layout.output_dimensions[1] +
                output_column);
            float value = 0.0F;
            std::memcpy(
                &value,
                data.science_bytes.data() + index * sizeof(float),
                sizeof(value));
            expect(
                value == synthetic_science(source_row, source_column),
                "regional science uses its source-window anchored lattice",
                failures);
            expect(
                static_cast<std::uint8_t>(data.mask_bytes[index]) ==
                    synthetic_mask(source_row, source_column),
                "regional mask uses the identical anchored lattice",
                failures);
        }
    }
}

std::span<const std::byte> source_element(
    const std::vector<std::byte>& bytes,
    const satview::ReadPlan& plan,
    const std::uint64_t row,
    const std::uint64_t column) {
    const auto index =
        (row - plan.aligned.row) * plan.aligned.width +
        (column - plan.aligned.column);
    const auto offset =
        static_cast<std::size_t>(index) * plan.data_type.element_size;
    return {
        bytes.data() + offset,
        plan.data_type.element_size};
}

void expect_overview_sample(
    const satview::Hdf5Product& product,
    const satview::overview::OverviewData& data,
    const std::uint64_t output_row,
    const std::uint64_t output_column,
    int& failures) {
    const auto& layout = data.plan.layout;
    const auto source_row = layout.source_origin[0] +
        output_row * layout.sample_stride[0];
    const auto source_column = layout.source_origin[1] +
        output_column * layout.sample_stride[1];
    const auto science_plan = product.make_read_plan(
        data.plan.request.science_dataset_path,
        source_row,
        source_column,
        1,
        1);
    std::vector<std::byte> science(science_plan.expected_bytes);
    product.read_into(science_plan, science);
    const auto direct_science = source_element(
        science, science_plan, source_row, source_column);
    const auto output_index =
        output_row * layout.output_dimensions[1] + output_column;
    const auto overview_science_offset =
        static_cast<std::size_t>(output_index) *
        layout.science_type.element_size;
    expect(
        std::memcmp(
            direct_science.data(),
            data.science_bytes.data() + overview_science_offset,
            layout.science_type.element_size) == 0,
        "overview science is the exact globally anchored native sample",
        failures);

    if (data.plan.request.mask_dataset_path.has_value()) {
        const auto mask_plan = product.make_read_plan(
            *data.plan.request.mask_dataset_path,
            source_row,
            source_column,
            1,
            1);
        std::vector<std::byte> mask(mask_plan.expected_bytes);
        product.read_into(mask_plan, mask);
        const auto direct_mask = source_element(
            mask, mask_plan, source_row, source_column);
        expect(
            std::memcmp(
                direct_mask.data(),
                data.mask_bytes.data() +
                    static_cast<std::size_t>(output_index),
                sizeof(std::uint8_t)) == 0,
            "overview mask uses the exact science sampling coordinate",
            failures);
    }
}

void run_synthetic_regional_tests(int& failures) {
    using namespace satview::overview;

    TemporaryDirectory temporary;
    const satview::Hdf5Product product(
        make_synthetic_product(temporary.path()));
    constexpr std::string_view science_path =
        "/science/LSAR/GCOV/grids/frequencyA/HHHH";
    const auto* science_descriptor = product.find_dataset(science_path);
    expect(
        science_descriptor != nullptr &&
            science_descriptor->data_type.file_representation_is_native &&
            science_descriptor->filters.size() == 2,
        "synthetic science exposes an exact native shuffle/DEFLATE pipeline",
        failures);
    const auto direct_plan = product.make_read_plan(
        science_path, 1, 1, 1, 1);
    const auto multi_chunk_plan = product.make_read_plan(
        science_path, 3, 4, 2, 2);
    const auto edge_plan = product.make_read_plan(
        science_path,
        kSyntheticRows - 1,
        kSyntheticColumns - 1,
        1,
        1);
    expect(
        product.supports_direct_chunk_decode(direct_plan) &&
            !product.supports_direct_chunk_decode(multi_chunk_plan) &&
            product.supports_direct_chunk_decode(edge_plan),
        "direct decoder accepts one full/edge chunk and rejects multichunk reads",
        failures);
    std::vector<std::byte> edge_bytes(edge_plan.expected_bytes);
    product.read_direct_chunk_into(edge_plan, edge_bytes);
    float edge_value = 0.0F;
    const auto edge_offset = static_cast<std::size_t>(
        (edge_plan.requested_row_offset * edge_plan.aligned.width +
         edge_plan.requested_column_offset) * sizeof(float));
    std::memcpy(
        &edge_value,
        edge_bytes.data() + edge_offset,
        sizeof(edge_value));
    expect(
        edge_value == synthetic_science(
            kSyntheticRows - 1, kSyntheticColumns - 1),
        "direct decoder crops a shuffled partial edge chunk exactly",
        failures);
    OverviewRequest request{
        .science_dataset_path =
            "/science/LSAR/GCOV/grids/frequencyA/HHHH",
        .mask_dataset_path =
            "/science/LSAR/GCOV/grids/frequencyA/mask",
        .maximum_long_edge = 16,
        .maximum_output_bytes = 1ULL * 1024ULL * 1024ULL,
        .maximum_scratch_bytes = 1ULL * 1024ULL * 1024ULL,
        .source_window = satview::Window2D{
            .row = 3,
            .column = 4,
            .height = 6,
            .width = 8,
        },
        .sample_stride = std::array<std::uint64_t, 2>{2, 3},
    };
    const auto plan = make_overview_plan(product, request);
    expect(
        plan.layout.source_origin ==
                std::array<std::uint64_t, 2>{3, 4} &&
            plan.layout.source_dimensions ==
                std::array<std::uint64_t, 2>{6, 8} &&
            plan.layout.sample_stride ==
                std::array<std::uint64_t, 2>{2, 3} &&
            plan.layout.output_dimensions ==
                std::array<std::uint64_t, 2>{3, 3},
        "synthetic regional plan preserves exact coverage and strides",
        failures);
    expect(
        plan.layout.source_chunk_count == 9,
        "regional plan counts only its nine intersecting source chunks",
        failures);
    expect(
        plan.request.source_window.has_value() &&
            plan.request.sample_stride == request.sample_stride,
        "regional plan retains canonical request mapping",
        failures);

    std::uint64_t completed = 0;
    const auto built = build_overview(
        product,
        plan,
        {},
        [&](const OverviewProgress& progress) {
            completed = progress.completed_source_chunks;
        });
    expect(
        built.status == OverviewPrepareStatus::built &&
            completed == 9,
        "synthetic regional build scans every intersecting chunk once",
        failures);
    expect_synthetic_payload(built, failures);

    auto unmasked_request = request;
    unmasked_request.mask_dataset_path.reset();
    const auto unmasked_plan =
        make_overview_plan(product, unmasked_request);
    const auto unmasked_built =
        build_overview(product, unmasked_plan);
    expect(
        unmasked_built.status == OverviewPrepareStatus::built &&
            unmasked_built.mask_bytes.empty(),
        "unmasked overview uses no mask allocation",
        failures);

    auto huge_stride_request = request;
    huge_stride_request.source_window = satview::Window2D{
        .row = 1,
        .column = 2,
        .height = 9,
        .width = 11,
    };
    huge_stride_request.sample_stride =
        std::array<std::uint64_t, 2>{
            std::numeric_limits<std::uint64_t>::max(),
            std::numeric_limits<std::uint64_t>::max()};
    const auto huge_stride_plan =
        make_overview_plan(product, huge_stride_request);
    const auto huge_stride_data =
        build_overview(product, huge_stride_plan);
    expect(
        huge_stride_plan.layout.output_dimensions ==
                std::array<std::uint64_t, 2>{1, 1} &&
            huge_stride_data.status == OverviewPrepareStatus::built &&
            huge_stride_data.ready(),
        "a huge explicit stride skips later chunks without coordinate overflow",
        failures);
    expect_synthetic_payload(huge_stride_data, failures);

    const OverviewRequest whole_request{
        .science_dataset_path = request.science_dataset_path,
        .mask_dataset_path = request.mask_dataset_path,
        .maximum_long_edge = 16,
        .maximum_output_bytes = 1ULL * 1024ULL * 1024ULL,
        .maximum_scratch_bytes = 1ULL * 1024ULL * 1024ULL,
    };
    const auto whole_plan = make_overview_plan(product, whole_request);
    expect(
        !whole_plan.request.source_window.has_value() &&
            whole_plan.layout.source_origin ==
                std::array<std::uint64_t, 2>{0, 0},
        "default overview coverage is the whole source",
        failures);

    auto explicit_whole_request = whole_request;
    explicit_whole_request.source_window = satview::Window2D{
        .row = 0,
        .column = 0,
        .height = kSyntheticRows,
        .width = kSyntheticColumns,
    };
    const auto explicit_whole_plan =
        make_overview_plan(product, explicit_whole_request);
    expect(
        !explicit_whole_plan.request.source_window.has_value() &&
            explicit_whole_plan.layout == whole_plan.layout &&
            explicit_whole_plan.validation_metadata ==
                whole_plan.validation_metadata,
        "an explicit complete window canonicalizes to the whole overview",
        failures);

    const auto cancel_plan = make_overview_plan(product, request);
    std::atomic_bool cancel{false};
    const auto cancelled = build_overview(
        product,
        cancel_plan,
        [&] { return cancel.load(); },
        [&](const OverviewProgress& progress) {
            if (progress.completed_source_chunks >= 1) {
                cancel.store(true);
            }
        });
    expect(
        cancelled.status == OverviewPrepareStatus::cancelled &&
            cancelled.science_bytes.empty() &&
            cancelled.mask_bytes.empty(),
        "synthetic regional cancellation returns no partial output",
        failures);

    bool rejected_window = false;
    try {
        auto invalid = request;
        invalid.source_window = satview::Window2D{
            .row = 9, .column = 0, .height = 2, .width = 1};
        static_cast<void>(make_overview_plan(product, invalid));
    } catch (const std::exception&) {
        rejected_window = true;
    }
    expect(rejected_window, "out-of-bounds regional window is rejected", failures);

    bool rejected_stride = false;
    try {
        auto invalid = request;
        invalid.sample_stride = std::array<std::uint64_t, 2>{0, 1};
        static_cast<void>(make_overview_plan(product, invalid));
    } catch (const std::exception&) {
        rejected_stride = true;
    }
    expect(rejected_stride, "zero explicit regional stride is rejected", failures);

    bool rejected_output_edge = false;
    try {
        auto invalid = request;
        invalid.maximum_long_edge = 2;
        static_cast<void>(make_overview_plan(product, invalid));
    } catch (const std::exception&) {
        rejected_output_edge = true;
    }
    expect(
        rejected_output_edge,
        "explicit strides cannot escape the configured output-edge bound",
        failures);

    expect(
        kDefaultOverviewLongEdge == 2048 &&
            kMaximumOverviewLongEdge == 4096 &&
            kDefaultMaximumOverviewBytes >=
                4096ULL * 4096ULL * 9ULL,
        "4096 pages fit the raised default memory budget without changing the 2048 default",
        failures);
}

}  // namespace

int run_overview_tests(const satview::Hdf5Product& product) {
    using namespace satview::overview;

    int failures = 0;
    run_synthetic_regional_tests(failures);
    OverviewRequest request{
        .science_dataset_path =
            "/science/LSAR/GCOV/grids/frequencyB/HHHH",
        .mask_dataset_path =
            "/science/LSAR/GCOV/grids/frequencyB/mask",
        .maximum_long_edge = 128,
        .maximum_output_bytes = 8ULL * 1024ULL * 1024ULL,
        .maximum_scratch_bytes = 8ULL * 1024ULL * 1024ULL,
    };
    const auto plan = make_overview_plan(product, request);
    expect(
        plan.layout.output_dimensions[0] <= request.maximum_long_edge &&
            plan.layout.output_dimensions[1] <=
                request.maximum_long_edge,
        "both planned output axes honor the long-edge bound",
        failures);
    expect(
        plan.layout.source_origin ==
                std::array<std::uint64_t, 2>{0, 0} &&
            plan.layout.sample_stride[0] > 0 &&
            plan.layout.sample_stride[1] > 0,
        "sampling is globally anchored with positive strides",
        failures);
    expect(
        plan.layout.science_bytes + plan.layout.mask_bytes <=
            request.maximum_output_bytes,
        "planned output honors its fixed-memory bound",
        failures);
    expect(
        plan.layout.scratch_bytes <= request.maximum_scratch_bytes,
        "planned source staging honors its fixed-memory bound",
        failures);

    std::uint64_t last_completed = 0;
    std::uint64_t last_bytes = 0;
    const auto built = build_overview(
        product,
        plan,
        {},
        [&](const OverviewProgress& progress) {
            expect(
                progress.completed_source_chunks >= last_completed,
                "build progress is monotonic",
                failures);
            expect(
                progress.source_bytes_read >= last_bytes,
                "source-byte progress is monotonic",
                failures);
            last_completed = progress.completed_source_chunks;
            last_bytes = progress.source_bytes_read;
        });
    expect(
        built.status == OverviewPrepareStatus::built && built.ready(),
        "first preparation builds and returns a ready overview",
        failures);
    expect(
        last_completed == plan.layout.source_chunk_count &&
            last_bytes > 0,
        "build reports every source chunk and decoded source bytes",
        failures);
    expect(
        built.science_bytes.size() == plan.layout.science_bytes &&
            built.mask_bytes.size() == plan.layout.mask_bytes,
        "built payload sizes match the plan",
        failures);
    expect_overview_sample(product, built, 0, 0, failures);
    expect_overview_sample(
        product,
        built,
        std::min<std::uint64_t>(
            17, plan.layout.output_dimensions[0] - 1),
        std::min<std::uint64_t>(
            29, plan.layout.output_dimensions[1] - 1),
        failures);
    expect_overview_sample(
        product,
        built,
        plan.layout.output_dimensions[0] - 1,
        plan.layout.output_dimensions[1] - 1,
        failures);

    const auto repeated = build_overview(product, plan);
    expect(
        repeated.status == OverviewPrepareStatus::built &&
            repeated.science_bytes == built.science_bytes &&
            repeated.mask_bytes == built.mask_bytes,
        "repeated preparation is byte-exact",
        failures);

    OverviewRequest regional_request = request;
    regional_request.maximum_long_edge = 256;
    regional_request.source_window = satview::Window2D{
        .row = 507,
        .column = 509,
        .height = 1'029,
        .width = 1'027,
    };
    regional_request.sample_stride =
        std::array<std::uint64_t, 2>{7, 11};
    const auto regional_plan =
        make_overview_plan(product, regional_request);
    expect(
        regional_plan.layout.source_origin ==
                std::array<std::uint64_t, 2>{507, 509} &&
            regional_plan.layout.source_dimensions ==
                std::array<std::uint64_t, 2>{1'029, 1'027} &&
            regional_plan.layout.sample_stride ==
                std::array<std::uint64_t, 2>{7, 11} &&
            regional_plan.layout.output_dimensions ==
                std::array<std::uint64_t, 2>{147, 94} &&
            regional_plan.layout.source_chunk_count == 9,
        "real regional plan is exact and counts only intersecting chunks",
        failures);
    std::uint64_t regional_completed = 0;
    const auto regional_built = build_overview(
        product,
        regional_plan,
        {},
        [&](const OverviewProgress& progress) {
            regional_completed = progress.completed_source_chunks;
        });
    expect(
        regional_built.status == OverviewPrepareStatus::built &&
            regional_completed == 9,
        "real regional build reads exactly its intersecting chunks",
        failures);
    expect_overview_sample(product, regional_built, 0, 0, failures);
    expect_overview_sample(
        product, regional_built, 73, 47, failures);
    expect_overview_sample(
        product,
        regional_built,
        regional_plan.layout.output_dimensions[0] - 1,
        regional_plan.layout.output_dimensions[1] - 1,
        failures);
    const auto regional_repeated =
        build_overview(product, regional_plan);
    expect(
        regional_repeated.status == OverviewPrepareStatus::built &&
            regional_repeated.science_bytes == regional_built.science_bytes &&
            regional_repeated.mask_bytes == regional_built.mask_bytes,
        "real regional science and mask are byte-exact when rebuilt",
        failures);

    OverviewRequest cancelled_request = regional_request;
    const auto cancelled_plan =
        make_overview_plan(product, cancelled_request);
    std::atomic_bool request_cancel{false};
    const auto cancelled_result = build_overview(
        product,
        cancelled_plan,
        [&] { return request_cancel.load(); },
        [&](const OverviewProgress& progress) {
            if (progress.completed_source_chunks >= 2) {
                request_cancel.store(true);
            }
        });
    expect(
        cancelled_result.status == OverviewPrepareStatus::cancelled &&
            !cancelled_result.ready() &&
            cancelled_result.science_bytes.empty() &&
            cancelled_result.mask_bytes.empty(),
        "real regional cancellation returns no partial payload",
        failures);
    OverviewRequest invalid_mask_request = request;
    invalid_mask_request.mask_dataset_path =
        "/science/LSAR/GCOV/grids/frequencyB/inputDataExceptionMask";
    bool rejected_mismatched_mask = false;
    try {
        static_cast<void>(
            make_overview_plan(product, invalid_mask_request));
    } catch (const std::exception&) {
        rejected_mismatched_mask = true;
    }
    expect(
        rejected_mismatched_mask,
        "dimension-mismatched masks are rejected",
        failures);
    return failures;
}

int run_overview_spacing_tests(const satview::Hdf5Product& product) {
    using namespace satview::overview;

    int failures = 0;
    TemporaryDirectory temporary;
    const OverviewRequest request{
        .science_dataset_path =
            "/science/LSAR/GSLC/grids/frequencyB/HH",
        .mask_dataset_path =
            "/science/LSAR/GSLC/grids/frequencyB/mask",
    };
    const auto plan = make_overview_plan(product, request);
    const auto* frequency = product.find_frequency("B");
    expect(
        frequency != nullptr,
        "GSLC frequency-B metadata exists",
        failures);
    if (frequency != nullptr) {
        const auto row_resolution =
            std::abs(frequency->grid.y.spacing) *
            static_cast<double>(plan.layout.sample_stride[0]);
        const auto column_resolution =
            std::abs(frequency->grid.x.spacing) *
            static_cast<double>(plan.layout.sample_stride[1]);
        const auto tolerance = std::max(
            std::abs(frequency->grid.y.spacing),
            std::abs(frequency->grid.x.spacing));
        expect(
            std::abs(row_resolution - column_resolution) <=
                tolerance + 1.0e-9,
            "independent strides keep physical sample resolution balanced",
            failures);
        expect(
            plan.layout.sample_stride[0] !=
                plan.layout.sample_stride[1],
            "anisotropic frequency-B source spacing uses unequal strides",
            failures);
    }
    expect(
        plan.layout.output_dimensions[0] <= request.maximum_long_edge &&
            plan.layout.output_dimensions[1] <=
                request.maximum_long_edge,
        "physical-spacing plan still honors both output bounds",
        failures);
    return failures;
}
