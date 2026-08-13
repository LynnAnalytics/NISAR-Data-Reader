#include "satview/hdf5_product.hpp"

#include <hdf5.h>
#include <H5Ldevelop.h>
#include <H5PLpublic.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

[[nodiscard]] std::string hdf5_file_name(
    const std::filesystem::path& path) {
#ifdef _WIN32
    const auto utf8 = path.u8string();
    return std::string(
        reinterpret_cast<const char*>(utf8.data()), utf8.size());
#else
    return path.string();
#endif
}

[[nodiscard]] hid_t checked_id(
    const hid_t value,
    std::string_view operation) {
    if (value < 0) {
        throw std::runtime_error(
            "HDF5 boundary fixture failed while " +
            std::string(operation));
    }
    return value;
}

void checked_status(
    const herr_t value,
    std::string_view operation) {
    if (value < 0) {
        throw std::runtime_error(
            "HDF5 boundary fixture failed while " +
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

    [[nodiscard]] hid_t get() const noexcept { return value_; }

private:
    hid_t value_ = H5I_INVALID_HID;
    Closer closer_ = nullptr;
};

class TemporaryDirectory final {
public:
    TemporaryDirectory() {
        static std::atomic_uint64_t counter{0};
        const auto stamp = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        path_ = std::filesystem::temp_directory_path() /
            ("satview-hdf5-boundary-" + std::to_string(stamp) + "-" +
             std::to_string(counter.fetch_add(1)));
        if (!std::filesystem::create_directory(path_)) {
            throw std::runtime_error(
                "could not create HDF5 boundary test directory");
        }
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

void create_group(const hid_t file, const char* path) {
    H5TestHandle group(
        checked_id(
            H5Gcreate2(
                file, path, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
            "creating a group"),
        H5Gclose);
}

void write_string_dataset(
    const hid_t file,
    const char* path,
    const std::string& value,
    const bool variable_length = false) {
    H5TestHandle type(
        checked_id(H5Tcopy(H5T_C_S1), "copying a string datatype"),
        H5Tclose);
    checked_status(
        H5Tset_size(
            type.get(),
            variable_length ? H5T_VARIABLE : value.size() + 1),
        "sizing a string datatype");
    checked_status(
        H5Tset_strpad(type.get(), H5T_STR_NULLTERM),
        "setting string padding");
    H5TestHandle space(
        checked_id(H5Screate(H5S_SCALAR), "creating a string dataspace"),
        H5Sclose);
    H5TestHandle dataset(
        checked_id(
            H5Dcreate2(
                file,
                path,
                type.get(),
                space.get(),
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT),
            "creating a string dataset"),
        H5Dclose);
    if (variable_length) {
        const char* bytes = value.c_str();
        checked_status(
            H5Dwrite(
                dataset.get(),
                type.get(),
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                &bytes),
            "writing a variable-length string dataset");
        return;
    }
    checked_status(
        H5Dwrite(
            dataset.get(),
            type.get(),
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            value.c_str()),
        "writing a fixed-length string dataset");
}

void write_axis(
    const hid_t file,
    const char* path,
    const std::array<double, 2>& values) {
    constexpr hsize_t count = 2;
    H5TestHandle space(
        checked_id(
            H5Screate_simple(1, &count, nullptr),
            "creating an axis dataspace"),
        H5Sclose);
    H5TestHandle dataset(
        checked_id(
            H5Dcreate2(
                file,
                path,
                H5T_IEEE_F64LE,
                space.get(),
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT),
            "creating an axis dataset"),
        H5Dclose);
    checked_status(
        H5Dwrite(
            dataset.get(),
            H5T_NATIVE_DOUBLE,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            values.data()),
        "writing an axis dataset");
}

void write_vlen_string_attribute(
    const hid_t object,
    const char* const name,
    const std::string& value) {
    H5TestHandle type(
        checked_id(H5Tcopy(H5T_C_S1), "copying an attribute datatype"),
        H5Tclose);
    checked_status(
        H5Tset_size(type.get(), H5T_VARIABLE),
        "sizing an attribute datatype");
    H5TestHandle attribute_space(
        checked_id(
            H5Screate(H5S_SCALAR),
            "creating an attribute dataspace"),
        H5Sclose);
    H5TestHandle attribute(
        checked_id(
            H5Acreate2(
                object,
                name,
                type.get(),
                attribute_space.get(),
                H5P_DEFAULT,
                H5P_DEFAULT),
            "creating a variable-length attribute"),
        H5Aclose);
    const char* bytes = value.c_str();
    checked_status(
        H5Awrite(attribute.get(), type.get(), &bytes),
        "writing a variable-length attribute");
}

void write_fixed_string_attribute(
    const hid_t object,
    const char* const name,
    const std::string& value) {
    H5TestHandle type(
        checked_id(H5Tcopy(H5T_C_S1), "copying an attribute datatype"),
        H5Tclose);
    checked_status(
        H5Tset_size(type.get(), value.size() + 1),
        "sizing an attribute datatype");
    checked_status(
        H5Tset_strpad(type.get(), H5T_STR_NULLTERM),
        "setting attribute string padding");
    H5TestHandle attribute_space(
        checked_id(
            H5Screate(H5S_SCALAR),
            "creating an attribute dataspace"),
        H5Sclose);
    H5TestHandle attribute(
        checked_id(
            H5Acreate2(
                object,
                name,
                type.get(),
                attribute_space.get(),
                H5P_DEFAULT,
                H5P_DEFAULT),
            "creating a fixed-length attribute"),
        H5Aclose);
    checked_status(
        H5Awrite(attribute.get(), type.get(), value.c_str()),
        "writing a fixed-length attribute");
}

void write_small_science_dataset(const hid_t file) {
    constexpr std::array<hsize_t, 2> dimensions{2, 2};
    constexpr std::array<float, 4> values{1.0F, 2.0F, 3.0F, 4.0F};
    H5TestHandle space(
        checked_id(
            H5Screate_simple(2, dimensions.data(), nullptr),
            "creating a science dataspace"),
        H5Sclose);
    H5TestHandle dataset(
        checked_id(
            H5Dcreate2(
                file,
                "/science/LSAR/GCOV/grids/frequencyA/HHHH",
                H5T_IEEE_F32LE,
                space.get(),
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT),
            "creating a science dataset"),
        H5Dclose);
    checked_status(
        H5Dwrite(
            dataset.get(),
            H5T_NATIVE_FLOAT,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            values.data()),
        "writing a science dataset");
    write_fixed_string_attribute(dataset.get(), "units", "unitless");
}

[[nodiscard]] std::filesystem::path make_valid_product(
    const std::filesystem::path& path,
    const bool variable_identification = false) {
    const auto name = hdf5_file_name(path);
    H5TestHandle file(
        checked_id(
            H5Fcreate(
                name.c_str(),
                H5F_ACC_TRUNC,
                H5P_DEFAULT,
                H5P_DEFAULT),
            "creating a product file"),
        H5Fclose);
    create_group(file.get(), "/science");
    create_group(file.get(), "/science/LSAR");
    create_group(file.get(), "/science/LSAR/identification");
    create_group(file.get(), "/science/LSAR/GCOV");
    create_group(file.get(), "/science/LSAR/GCOV/grids");
    create_group(file.get(), "/science/LSAR/GCOV/grids/frequencyA");
    write_string_dataset(
        file.get(),
        "/science/LSAR/identification/productType",
        "GCOV",
        variable_identification);
    write_string_dataset(
        file.get(),
        "/science/LSAR/identification/granuleId",
        "boundary-test",
        variable_identification);
    write_string_dataset(
        file.get(),
        "/science/LSAR/identification/productVersion",
        "test",
        variable_identification);
    write_string_dataset(
        file.get(),
        "/science/LSAR/GCOV/grids/frequencyA/listOfCovarianceTerms",
        "HHHH",
        variable_identification);
    write_axis(
        file.get(),
        "/science/LSAR/GCOV/grids/frequencyA/xCoordinates",
        {100.0, 110.0});
    write_axis(
        file.get(),
        "/science/LSAR/GCOV/grids/frequencyA/yCoordinates",
        {200.0, 190.0});
    write_small_science_dataset(file.get());
    return path;
}

template <typename Mutation>
[[nodiscard]] std::filesystem::path make_mutated_product(
    const std::filesystem::path& path,
    Mutation&& mutation) {
    static_cast<void>(make_valid_product(path));
    const auto name = hdf5_file_name(path);
    H5TestHandle file(
        checked_id(
            H5Fopen(name.c_str(), H5F_ACC_RDWR, H5P_DEFAULT),
            "opening a product for mutation"),
        H5Fclose);
    std::forward<Mutation>(mutation)(file.get());
    return path;
}

void expect(
    const bool condition,
    std::string_view message,
    int& failures) {
    if (!condition) {
        ++failures;
        std::cerr << "HDF5 boundary test failed: " << message << '\n';
    }
}

void expect_rejected(
    const std::filesystem::path& path,
    std::string_view expected_message,
    std::string_view test_name,
    int& failures) {
    try {
        const satview::Hdf5Product product(path);
        static_cast<void>(product);
        expect(false, std::string(test_name) + " was accepted", failures);
    } catch (const satview::Hdf5Error& error) {
        expect(
            std::string_view(error.what()).find(expected_message) !=
                std::string_view::npos,
            std::string(test_name) + " reported an unexpected error: " +
                error.what(),
            failures);
    }
}

std::atomic_int user_link_traversals{0};

hid_t user_link_traverse(
    const char*,
    hid_t,
    const void*,
    size_t,
    hid_t,
    hid_t) {
    user_link_traversals.fetch_add(1);
    return H5I_INVALID_HID;
}

void test_unicode_and_plugin_policy(
    const std::filesystem::path& directory,
    int& failures) {
    const auto path = make_valid_product(
        directory / std::filesystem::path(L"product-\u96ea.h5"),
        true);
    const satview::Hdf5Product product(path);
    expect(
        product.identification().granule_id == "boundary-test",
        "a UTF-8 Windows path and normal VLEN metadata open successfully",
        failures);
    const auto* dataset = product.find_dataset(
        "/science/LSAR/GCOV/grids/frequencyA/HHHH");
    expect(
        dataset != nullptr && dataset->units == "unitless",
        "a fixed-length string attribute remains supported",
        failures);

    unsigned plugin_mask = H5PL_ALL_PLUGIN;
    checked_status(
        H5PLget_loading_state(&plugin_mask),
        "reading the dynamic plugin policy");
    expect(
        plugin_mask == 0,
        "opening a product disables all dynamic HDF5 plugins",
        failures);
}

void test_external_link(
    const std::filesystem::path& directory,
    int& failures) {
    const auto path = directory / "external-link.h5";
    const auto name = hdf5_file_name(path);
    {
        H5TestHandle file(
            checked_id(
                H5Fcreate(
                    name.c_str(),
                    H5F_ACC_TRUNC,
                    H5P_DEFAULT,
                    H5P_DEFAULT),
                "creating an external-link fixture"),
            H5Fclose);
        checked_status(
            H5Lcreate_external(
                "missing-external-target.h5",
                "/",
                file.get(),
                "/science",
                H5P_DEFAULT,
                H5P_DEFAULT),
            "creating an external link");
    }
    expect_rejected(
        path,
        "refusing external link",
        "external link fixture",
        failures);
}

void test_user_defined_link(
    const std::filesystem::path& directory,
    int& failures) {
    constexpr auto link_type = static_cast<H5L_type_t>(65);
    const H5L_class_t link_class{
        .version = H5L_LINK_CLASS_T_VERS,
        .id = link_type,
        .comment = "satview boundary test link",
        .create_func = nullptr,
        .move_func = nullptr,
        .copy_func = nullptr,
        .trav_func = user_link_traverse,
        .del_func = nullptr,
        .query_func = nullptr,
    };
    checked_status(
        H5Lregister(&link_class),
        "registering a user-defined link class");
    const auto path = directory / "user-link.h5";
    const auto name = hdf5_file_name(path);
    {
        H5TestHandle file(
            checked_id(
                H5Fcreate(
                    name.c_str(),
                    H5F_ACC_TRUNC,
                    H5P_DEFAULT,
                    H5P_DEFAULT),
                "creating a user-link fixture"),
            H5Fclose);
        checked_status(
            H5Lcreate_ud(
                file.get(),
                "/science",
                link_type,
                nullptr,
                0,
                H5P_DEFAULT,
                H5P_DEFAULT),
            "creating a user-defined link");
    }
    user_link_traversals.store(0);
    expect_rejected(
        path,
        "refusing user-defined link",
        "user-defined link fixture",
        failures);
    expect(
        user_link_traversals.load() == 0,
        "a rejected user-defined link is never traversed",
        failures);
    checked_status(
        H5Lunregister(link_type),
        "unregistering a user-defined link class");
}

void test_external_raw_storage(
    const std::filesystem::path& directory,
    int& failures) {
    const auto path = make_mutated_product(
        directory / "external-storage.h5",
        [](const hid_t file) {
            constexpr std::array<hsize_t, 2> dimensions{2, 2};
            H5TestHandle space(
                checked_id(
                    H5Screate_simple(2, dimensions.data(), nullptr),
                    "creating an external-storage dataspace"),
                H5Sclose);
            H5TestHandle properties(
                checked_id(
                    H5Pcreate(H5P_DATASET_CREATE),
                    "creating external-storage properties"),
                H5Pclose);
            checked_status(
                H5Pset_external(
                    properties.get(),
                    "missing-external.raw",
                    0,
                    4 * sizeof(float)),
                "setting external raw storage");
            H5TestHandle dataset(
                checked_id(
                    H5Dcreate2(
                        file,
                        "/science/LSAR/GCOV/grids/frequencyA/externalData",
                        H5T_IEEE_F32LE,
                        space.get(),
                        H5P_DEFAULT,
                        properties.get(),
                        H5P_DEFAULT),
                    "creating an external-storage dataset"),
                H5Dclose);
        });
    expect_rejected(
        path,
        "refusing dataset with external raw storage",
        "external raw-storage fixture",
        failures);
}

void test_virtual_dataset(
    const std::filesystem::path& directory,
    int& failures) {
    const auto path = make_mutated_product(
        directory / "virtual-dataset.h5",
        [](const hid_t file) {
            constexpr std::array<hsize_t, 2> dimensions{2, 2};
            H5TestHandle virtual_space(
                checked_id(
                    H5Screate_simple(2, dimensions.data(), nullptr),
                    "creating a virtual dataspace"),
                H5Sclose);
            H5TestHandle source_space(
                checked_id(
                    H5Screate_simple(2, dimensions.data(), nullptr),
                    "creating a VDS source dataspace"),
                H5Sclose);
            H5TestHandle properties(
                checked_id(
                    H5Pcreate(H5P_DATASET_CREATE),
                    "creating VDS properties"),
                H5Pclose);
            checked_status(
                H5Pset_virtual(
                    properties.get(),
                    virtual_space.get(),
                    "missing-vds-source.h5",
                    "/data",
                    source_space.get()),
                "setting a virtual-dataset mapping");
            H5TestHandle dataset(
                checked_id(
                    H5Dcreate2(
                        file,
                        "/science/LSAR/GCOV/grids/frequencyA/virtualData",
                        H5T_IEEE_F32LE,
                        virtual_space.get(),
                        H5P_DEFAULT,
                        properties.get(),
                        H5P_DEFAULT),
                    "creating a virtual dataset"),
                H5Dclose);
        });
    expect_rejected(
        path,
        "refusing virtual dataset",
        "virtual-dataset fixture",
        failures);
}

void test_vlen_limit(
    const std::filesystem::path& directory,
    int& failures) {
    const auto path = make_mutated_product(
        directory / "oversized-vlen.h5",
        [](const hid_t file) {
            constexpr const char* product_type_path =
                "/science/LSAR/identification/productType";
            checked_status(
                H5Ldelete(file, product_type_path, H5P_DEFAULT),
                "removing the fixed product type");
            write_string_dataset(
                file,
                product_type_path,
                std::string(1ULL * 1024ULL * 1024ULL + 1, 'G'),
                true);
        });
    expect_rejected(
        path,
        "variable-length string metadata exceeds safety limit",
        "oversized VLEN metadata fixture",
        failures);
}

void test_oversized_vlen_attribute(
    const std::filesystem::path& directory,
    int& failures) {
    const auto path = make_mutated_product(
        directory / "oversized-vlen-attribute.h5",
        [](const hid_t file) {
            constexpr const char* dataset_path =
                "/science/LSAR/GCOV/grids/frequencyA/HHHH";
            H5TestHandle dataset(
                checked_id(
                    H5Dopen2(file, dataset_path, H5P_DEFAULT),
                    "opening a dataset to replace its VLEN attribute"),
                H5Dclose);
            checked_status(
                H5Adelete(dataset.get(), "units"),
                "deleting a VLEN attribute");
            write_vlen_string_attribute(
                dataset.get(),
                "units",
                std::string(2ULL * 1024ULL * 1024ULL, 'u'));
        });
    expect_rejected(
        path,
        "refusing variable-length string attribute",
        "oversized VLEN attribute fixture",
        failures);
}

void test_small_vlen_attribute_policy(
    const std::filesystem::path& directory,
    int& failures) {
    const auto path = make_mutated_product(
        directory / "small-vlen-attribute.h5",
        [](const hid_t file) {
            constexpr const char* dataset_path =
                "/science/LSAR/GCOV/grids/frequencyA/HHHH";
            H5TestHandle dataset(
                checked_id(
                    H5Dopen2(file, dataset_path, H5P_DEFAULT),
                    "opening a dataset to replace its VLEN attribute"),
                H5Dclose);
            checked_status(
                H5Adelete(dataset.get(), "units"),
                "deleting a fixed-length attribute");
            write_vlen_string_attribute(dataset.get(), "units", "unitless");
        });
    expect_rejected(
        path,
        "refusing variable-length string attribute",
        "small VLEN attribute fixture",
        failures);
}

}  // namespace

int main() {
    static_assert(!std::is_move_constructible_v<satview::Hdf5Product>);
    static_assert(!std::is_move_assignable_v<satview::Hdf5Product>);

    int failures = 0;
    try {
        TemporaryDirectory temporary;
        test_unicode_and_plugin_policy(temporary.path(), failures);
        test_external_link(temporary.path(), failures);
        test_user_defined_link(temporary.path(), failures);
        test_external_raw_storage(temporary.path(), failures);
        test_virtual_dataset(temporary.path(), failures);
        test_vlen_limit(temporary.path(), failures);
        test_small_vlen_attribute_policy(temporary.path(), failures);
        test_oversized_vlen_attribute(temporary.path(), failures);
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "HDF5 boundary test fixture failed: "
                  << error.what() << '\n';
    }

    if (failures == 0) {
        std::cout << "All HDF5 boundary tests passed\n";
        return 0;
    }
    std::cerr << failures << " HDF5 boundary test(s) failed\n";
    return 1;
}
