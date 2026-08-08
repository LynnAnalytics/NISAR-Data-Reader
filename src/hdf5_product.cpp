#include "satview/hdf5_product.hpp"

#include <hdf5.h>
#include <zlib.h>

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <list>
#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace satview {
namespace {

// The pinned vcpkg HDF5 build is not thread-safe. The viewer deliberately uses
// one reader thread, and this process-wide lock also keeps multiple product
// objects safe if callers inspect them concurrently.
std::recursive_mutex& global_hdf5_mutex() {
    static std::recursive_mutex mutex;
    return mutex;
}

class H5Handle {
public:
    using Closer = herr_t (*)(hid_t);

    H5Handle() = default;
    H5Handle(hid_t value, Closer closer) noexcept : value_(value), closer_(closer) {}

    ~H5Handle() {
        reset();
    }

    H5Handle(H5Handle&& other) noexcept
        : value_(std::exchange(other.value_, H5I_INVALID_HID)),
          closer_(std::exchange(other.closer_, nullptr)) {}

    H5Handle& operator=(H5Handle&& other) noexcept {
        if (this != &other) {
            reset();
            value_ = std::exchange(other.value_, H5I_INVALID_HID);
            closer_ = std::exchange(other.closer_, nullptr);
        }
        return *this;
    }

    H5Handle(const H5Handle&) = delete;
    H5Handle& operator=(const H5Handle&) = delete;

    [[nodiscard]] hid_t get() const noexcept {
        return value_;
    }

    [[nodiscard]] explicit operator bool() const noexcept {
        return value_ >= 0;
    }

    void reset() noexcept {
        if (value_ >= 0 && closer_ != nullptr) {
            closer_(value_);
        }
        value_ = H5I_INVALID_HID;
        closer_ = nullptr;
    }

private:
    hid_t value_ = H5I_INVALID_HID;
    Closer closer_ = nullptr;
};

[[noreturn]] void fail(std::string_view operation, std::string_view path) {
    std::ostringstream message;
    message << "HDF5 " << operation;
    if (!path.empty()) {
        message << " [" << path << ']';
    }
    throw Hdf5Error(message.str());
}

hid_t checked_id(hid_t value, std::string_view operation, std::string_view path) {
    if (value < 0) {
        fail(operation, path);
    }
    return value;
}

void checked_status(herr_t value, std::string_view operation, std::string_view path) {
    if (value < 0) {
        fail(operation, path);
    }
}

[[nodiscard]] bool link_exists(hid_t location, const std::string& path) {
    htri_t result = -1;
    H5E_BEGIN_TRY {
        result = H5Lexists(location, path.c_str(), H5P_DEFAULT);
    }
    H5E_END_TRY;
    return result > 0;
}

[[nodiscard]] std::string normalize_hdf5_path(std::string_view input) {
    if (input.empty()) {
        return {};
    }

    std::string result;
    result.reserve(input.size() + 1);
    if (input.front() != '/') {
        result.push_back('/');
    }

    bool previous_slash = false;
    for (const char character : input) {
        const bool slash = character == '/';
        if (slash && previous_slash) {
            continue;
        }
        result.push_back(character);
        previous_slash = slash;
    }
    while (result.size() > 1 && result.back() == '/') {
        result.pop_back();
    }
    return result;
}

[[nodiscard]] std::string leaf_name(std::string_view path) {
    const auto position = path.find_last_of('/');
    return std::string(position == std::string_view::npos ? path : path.substr(position + 1));
}

[[nodiscard]] std::string trim_hdf5_string(std::string value) {
    if (const auto null_position = value.find('\0'); null_position != std::string::npos) {
        value.resize(null_position);
    }
    while (!value.empty() &&
           (value.back() == ' ' || value.back() == '\t' || value.back() == '\r' ||
            value.back() == '\n')) {
        value.pop_back();
    }
    return value;
}

[[nodiscard]] std::string uppercase_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char value) {
        return static_cast<char>(std::toupper(value));
    });
    return value;
}

[[nodiscard]] std::string lowercase_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return value;
}

[[nodiscard]] std::size_t checked_size_product(
    std::uint64_t first,
    std::uint64_t second,
    std::size_t third,
    std::string_view context) {
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    if (first != 0 && second > maximum / first) {
        throw Hdf5Error(std::string("byte-count overflow [") + std::string(context) + ']');
    }
    const auto first_two = static_cast<std::size_t>(first * second);
    if (third != 0 && first_two > maximum / third) {
        throw Hdf5Error(std::string("byte-count overflow [") + std::string(context) + ']');
    }
    return first_two * third;
}

[[nodiscard]] std::vector<std::string> read_string_dataset(
    hid_t file,
    const std::string& path,
    bool required) {
    if (!link_exists(file, path)) {
        if (required) {
            fail("missing required string dataset", path);
        }
        return {};
    }

    H5Handle dataset(
        checked_id(H5Dopen2(file, path.c_str(), H5P_DEFAULT), "opening dataset", path),
        H5Dclose);
    H5Handle type(
        checked_id(H5Dget_type(dataset.get()), "getting datatype", path),
        H5Tclose);
    if (H5Tget_class(type.get()) != H5T_STRING) {
        fail("expected string datatype", path);
    }

    H5Handle space(
        checked_id(H5Dget_space(dataset.get()), "getting dataspace", path),
        H5Sclose);
    const auto points = H5Sget_simple_extent_npoints(space.get());
    if (points < 0) {
        fail("getting string count", path);
    }
    constexpr hssize_t maximum_metadata_strings = 1'000'000;
    if (points > maximum_metadata_strings) {
        fail("refusing implausibly large metadata string dataset", path);
    }
    const auto count = static_cast<std::size_t>(points);
    if (count == 0) {
        return {};
    }

    std::vector<std::string> result;
    result.reserve(count);

    if (H5Tis_variable_str(type.get()) > 0) {
        std::vector<char*> values(count, nullptr);
        checked_status(
            H5Dread(
                dataset.get(),
                type.get(),
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                values.data()),
            "reading variable-length strings",
            path);
        for (char* value : values) {
            result.emplace_back(trim_hdf5_string(value == nullptr ? std::string{} : value));
            if (value != nullptr) {
                H5free_memory(value);
            }
        }
        return result;
    }

    const auto width = H5Tget_size(type.get());
    if (width == 0 || (count > (16ULL * 1024ULL * 1024ULL) / width)) {
        fail("invalid or excessive fixed-string metadata size", path);
    }
    std::vector<char> values(count * width);
    checked_status(
        H5Dread(
            dataset.get(),
            type.get(),
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            values.data()),
        "reading fixed-length strings",
        path);
    for (std::size_t index = 0; index < count; ++index) {
        result.emplace_back(trim_hdf5_string(
            std::string(values.data() + index * width, width)));
    }
    return result;
}

[[nodiscard]] std::string read_scalar_string(
    hid_t file,
    const std::string& path,
    bool required) {
    auto values = read_string_dataset(file, path, required);
    if (values.empty()) {
        return {};
    }
    if (values.size() != 1) {
        fail("expected scalar string", path);
    }
    return values.front();
}

[[nodiscard]] std::optional<double> read_scalar_double_if_present(
    hid_t file,
    const std::string& path) {
    if (!link_exists(file, path)) {
        return std::nullopt;
    }
    H5Handle dataset(
        checked_id(H5Dopen2(file, path.c_str(), H5P_DEFAULT), "opening dataset", path),
        H5Dclose);
    H5Handle space(
        checked_id(H5Dget_space(dataset.get()), "getting dataspace", path),
        H5Sclose);
    if (H5Sget_simple_extent_npoints(space.get()) != 1) {
        fail("expected scalar numeric dataset", path);
    }
    double result = 0.0;
    checked_status(
        H5Dread(
            dataset.get(),
            H5T_NATIVE_DOUBLE,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            &result),
        "reading scalar numeric dataset",
        path);
    return result;
}

[[nodiscard]] std::optional<std::uint32_t> read_scalar_u32_if_present(
    hid_t file,
    const std::string& path) {
    if (!link_exists(file, path)) {
        return std::nullopt;
    }
    H5Handle dataset(
        checked_id(H5Dopen2(file, path.c_str(), H5P_DEFAULT), "opening dataset", path),
        H5Dclose);
    H5Handle space(
        checked_id(H5Dget_space(dataset.get()), "getting dataspace", path),
        H5Sclose);
    if (H5Sget_simple_extent_npoints(space.get()) != 1) {
        fail("expected scalar integer dataset", path);
    }
    std::uint32_t result = 0;
    checked_status(
        H5Dread(
            dataset.get(),
            H5T_NATIVE_UINT32,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            &result),
        "reading scalar integer dataset",
        path);
    return result;
}

[[nodiscard]] std::optional<std::uint32_t> read_u32_attribute_if_present(
    hid_t object,
    const std::string& object_path,
    const char* attribute_name) {
    const auto present = H5Aexists(object, attribute_name);
    if (present < 0) {
        fail("checking attribute", object_path + "@" + attribute_name);
    }
    if (present == 0) {
        return std::nullopt;
    }

    const std::string context = object_path + "@" + attribute_name;
    H5Handle attribute(
        checked_id(H5Aopen(object, attribute_name, H5P_DEFAULT), "opening attribute", context),
        H5Aclose);
    H5Handle space(
        checked_id(H5Aget_space(attribute.get()), "getting attribute dataspace", context),
        H5Sclose);
    if (H5Sget_simple_extent_npoints(space.get()) != 1) {
        fail("expected scalar integer attribute", context);
    }
    std::uint32_t result = 0;
    checked_status(
        H5Aread(attribute.get(), H5T_NATIVE_UINT32, &result),
        "reading integer attribute",
        context);
    return result;
}

[[nodiscard]] std::string read_string_attribute_if_present(
    hid_t object,
    const std::string& object_path,
    const char* attribute_name) {
    const auto present = H5Aexists(object, attribute_name);
    if (present < 0) {
        fail("checking attribute", object_path + "@" + attribute_name);
    }
    if (present == 0) {
        return {};
    }

    const std::string context = object_path + "@" + attribute_name;
    H5Handle attribute(
        checked_id(H5Aopen(object, attribute_name, H5P_DEFAULT), "opening attribute", context),
        H5Aclose);
    H5Handle type(
        checked_id(H5Aget_type(attribute.get()), "getting attribute datatype", context),
        H5Tclose);
    if (H5Tget_class(type.get()) != H5T_STRING) {
        return {};
    }
    H5Handle space(
        checked_id(H5Aget_space(attribute.get()), "getting attribute dataspace", context),
        H5Sclose);
    if (H5Sget_simple_extent_npoints(space.get()) != 1) {
        return {};
    }

    if (H5Tis_variable_str(type.get()) > 0) {
        char* value = nullptr;
        checked_status(
            H5Aread(attribute.get(), type.get(), &value),
            "reading string attribute",
            context);
        std::string result = trim_hdf5_string(value == nullptr ? std::string{} : value);
        if (value != nullptr) {
            H5free_memory(value);
        }
        return result;
    }

    const auto width = H5Tget_size(type.get());
    if (width == 0 || width > 16ULL * 1024ULL * 1024ULL) {
        fail("invalid fixed string attribute size", context);
    }
    std::string value(width, '\0');
    checked_status(
        H5Aread(attribute.get(), type.get(), value.data()),
        "reading string attribute",
        context);
    return trim_hdf5_string(std::move(value));
}

[[nodiscard]] ByteOrder byte_order_from_hdf5(H5T_order_t order) {
    switch (order) {
    case H5T_ORDER_LE:
        return ByteOrder::little_endian;
    case H5T_ORDER_BE:
        return ByteOrder::big_endian;
    case H5T_ORDER_NONE:
        return ByteOrder::not_applicable;
    default:
        return ByteOrder::unknown;
    }
}

[[nodiscard]] bool is_real_member_name(std::string name) {
    name = lowercase_ascii(std::move(name));
    return name == "r" || name == "re" || name == "real";
}

[[nodiscard]] bool is_imaginary_member_name(std::string name) {
    name = lowercase_ascii(std::move(name));
    return name == "i" || name == "im" || name == "imag" || name == "imaginary";
}

[[nodiscard]] DataTypeInfo inspect_data_type(hid_t type, const std::string& path) {
    DataTypeInfo result;
    result.file_element_size = H5Tget_size(type);
    if (result.file_element_size == 0) {
        fail("getting datatype size", path);
    }

    const auto type_class = H5Tget_class(type);
    if (type_class == H5T_INTEGER) {
        result.file_byte_order = byte_order_from_hdf5(H5Tget_order(type));
        result.element_size = result.file_element_size;
        const auto sign = H5Tget_sign(type);
        if (sign == H5T_SGN_2) {
            result.kind = ScalarKind::signed_integer;
            result.description = "int" + std::to_string(result.element_size * 8);
        } else if (sign == H5T_SGN_NONE) {
            result.kind = ScalarKind::unsigned_integer;
            result.description = "uint" + std::to_string(result.element_size * 8);
        } else {
            result.description = "unsupported integer sign representation";
            return result;
        }
        result.readable =
            result.element_size == 1 || result.element_size == 2 ||
            result.element_size == 4 || result.element_size == 8;
        return result;
    }

    if (type_class == H5T_FLOAT) {
        result.kind = ScalarKind::floating_point;
        result.file_byte_order = byte_order_from_hdf5(H5Tget_order(type));
        result.element_size = result.file_element_size;
        result.description = "float" + std::to_string(result.element_size * 8);
        result.readable = result.element_size == 4 || result.element_size == 8;
        return result;
    }

    if (type_class != H5T_COMPOUND || H5Tget_nmembers(type) != 2) {
        result.description = "unsupported HDF5 datatype class";
        return result;
    }

    struct Member {
        std::string name;
        std::size_t offset = 0;
        std::size_t size = 0;
        ByteOrder byte_order = ByteOrder::unknown;
        bool floating_point = false;
    };
    std::array<Member, 2> members;
    for (unsigned index = 0; index < members.size(); ++index) {
        char* raw_name = H5Tget_member_name(type, index);
        if (raw_name == nullptr) {
            fail("getting compound member name", path);
        }
        members[index].name = raw_name;
        H5free_memory(raw_name);
        members[index].offset = H5Tget_member_offset(type, index);
        H5Handle member_type(
            checked_id(H5Tget_member_type(type, index), "getting compound member datatype", path),
            H5Tclose);
        members[index].floating_point = H5Tget_class(member_type.get()) == H5T_FLOAT;
        members[index].size = H5Tget_size(member_type.get());
        members[index].byte_order = byte_order_from_hdf5(H5Tget_order(member_type.get()));
    }

    const Member* real = nullptr;
    const Member* imaginary = nullptr;
    for (const auto& member : members) {
        if (is_real_member_name(member.name)) {
            real = &member;
        } else if (is_imaginary_member_name(member.name)) {
            imaginary = &member;
        }
    }
    if (real == nullptr || imaginary == nullptr || !real->floating_point ||
        !imaginary->floating_point || real->size != imaginary->size ||
        (real->size != 4 && real->size != 8)) {
        result.description = "unsupported compound type (expected float {r,i})";
        return result;
    }

    result.kind = ScalarKind::compound_complex;
    result.file_byte_order =
        real->byte_order == imaginary->byte_order ? real->byte_order : ByteOrder::unknown;
    result.element_size = real->size * 2;
    result.complex_layout = ComplexLayout{
        .component_size = real->size,
        .file_real_offset = real->offset,
        .file_imaginary_offset = imaginary->offset,
        .real_member_name = real->name,
        .imaginary_member_name = imaginary->name,
    };
    result.description = real->size == 4 ? "complex64 (compound float32 {r,i})"
                                         : "complex128 (compound float64 {r,i})";
    result.readable = true;
    return result;
}

struct NativeType {
    hid_t id = H5I_INVALID_HID;
    H5Handle owned;
};

[[nodiscard]] NativeType native_memory_type(
    const DataTypeInfo& info,
    const std::string& path) {
    NativeType result;
    if (!info.readable) {
        fail("datatype has no supported native memory representation", path);
    }

    if (info.kind == ScalarKind::signed_integer) {
        switch (info.element_size) {
        case 1:
            result.id = H5T_NATIVE_INT8;
            break;
        case 2:
            result.id = H5T_NATIVE_INT16;
            break;
        case 4:
            result.id = H5T_NATIVE_INT32;
            break;
        case 8:
            result.id = H5T_NATIVE_INT64;
            break;
        default:
            break;
        }
    } else if (info.kind == ScalarKind::unsigned_integer) {
        switch (info.element_size) {
        case 1:
            result.id = H5T_NATIVE_UINT8;
            break;
        case 2:
            result.id = H5T_NATIVE_UINT16;
            break;
        case 4:
            result.id = H5T_NATIVE_UINT32;
            break;
        case 8:
            result.id = H5T_NATIVE_UINT64;
            break;
        default:
            break;
        }
    } else if (info.kind == ScalarKind::floating_point) {
        result.id = info.element_size == 4 ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;
    } else if (info.kind == ScalarKind::compound_complex && info.complex_layout.has_value()) {
        const auto& layout = *info.complex_layout;
        const hid_t component_type =
            layout.component_size == 4 ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;
        result.owned = H5Handle(
            checked_id(
                H5Tcreate(H5T_COMPOUND, info.element_size),
                "creating native complex datatype",
                path),
            H5Tclose);
        checked_status(
            H5Tinsert(
                result.owned.get(),
                layout.real_member_name.c_str(),
                0,
                component_type),
            "inserting real complex member",
            path);
        checked_status(
            H5Tinsert(
                result.owned.get(),
                layout.imaginary_member_name.c_str(),
                layout.component_size,
                component_type),
            "inserting imaginary complex member",
            path);
        result.id = result.owned.get();
    }

    if (result.id < 0) {
        fail("creating native memory datatype", path);
    }
    return result;
}

[[nodiscard]] bool file_representation_is_native(
    const hid_t file_type,
    const DataTypeInfo& info,
    const std::string& path) {
    if (!info.readable) {
        return false;
    }
    const auto native_type = native_memory_type(info, path);
    const auto equal = H5Tequal(file_type, native_type.id);
    if (equal < 0) {
        fail("comparing file and native datatypes", path);
    }
    return equal > 0;
}

template <typename T>
[[nodiscard]] T load_scalar(const std::vector<std::byte>& bytes) {
    T value{};
    std::memcpy(&value, bytes.data(), sizeof(T));
    return value;
}

[[nodiscard]] FillValue decode_fill_value(
    std::vector<std::byte> bytes,
    const DataTypeInfo& info) {
    FillValue result;
    result.bytes = std::move(bytes);
    switch (info.kind) {
    case ScalarKind::signed_integer:
        switch (info.element_size) {
        case 1:
            result.numeric = static_cast<double>(load_scalar<std::int8_t>(result.bytes));
            break;
        case 2:
            result.numeric = static_cast<double>(load_scalar<std::int16_t>(result.bytes));
            break;
        case 4:
            result.numeric = static_cast<double>(load_scalar<std::int32_t>(result.bytes));
            break;
        case 8:
            result.numeric = static_cast<double>(load_scalar<std::int64_t>(result.bytes));
            break;
        }
        break;
    case ScalarKind::unsigned_integer:
        switch (info.element_size) {
        case 1:
            result.numeric = static_cast<double>(load_scalar<std::uint8_t>(result.bytes));
            break;
        case 2:
            result.numeric = static_cast<double>(load_scalar<std::uint16_t>(result.bytes));
            break;
        case 4:
            result.numeric = static_cast<double>(load_scalar<std::uint32_t>(result.bytes));
            break;
        case 8:
            result.numeric = static_cast<double>(load_scalar<std::uint64_t>(result.bytes));
            break;
        }
        break;
    case ScalarKind::floating_point:
        result.numeric = info.element_size == 4
                             ? static_cast<double>(load_scalar<float>(result.bytes))
                             : load_scalar<double>(result.bytes);
        break;
    case ScalarKind::compound_complex:
        if (info.complex_layout->component_size == 4) {
            result.complex = ComplexValue{
                .real = static_cast<double>(load_scalar<float>(result.bytes)),
                .imaginary = static_cast<double>([&] {
                    float value = 0.0F;
                    std::memcpy(
                        &value,
                        result.bytes.data() + info.complex_layout->component_size,
                        sizeof(value));
                    return value;
                }()),
            };
        } else {
            double imaginary = 0.0;
            std::memcpy(
                &imaginary,
                result.bytes.data() + info.complex_layout->component_size,
                sizeof(imaginary));
            result.complex = ComplexValue{
                .real = load_scalar<double>(result.bytes),
                .imaginary = imaginary,
            };
        }
        break;
    case ScalarKind::unsupported:
        break;
    }
    return result;
}

[[nodiscard]] std::optional<FillValue> creation_fill_value(
    hid_t creation_properties,
    const DataTypeInfo& data_type,
    const std::string& path) {
    if (!data_type.readable) {
        return std::nullopt;
    }
    H5D_fill_value_t status = H5D_FILL_VALUE_ERROR;
    checked_status(
        H5Pfill_value_defined(creation_properties, &status),
        "checking creation fill value",
        path);
    if (status == H5D_FILL_VALUE_UNDEFINED) {
        return std::nullopt;
    }

    auto memory_type = native_memory_type(data_type, path);
    std::vector<std::byte> bytes(data_type.element_size);
    checked_status(
        H5Pget_fill_value(creation_properties, memory_type.id, bytes.data()),
        "reading creation fill value",
        path);
    return decode_fill_value(std::move(bytes), data_type);
}

[[nodiscard]] std::optional<FillValue> fill_value_attribute(
    hid_t dataset,
    const DataTypeInfo& data_type,
    const std::string& path) {
    const auto present = H5Aexists(dataset, "_FillValue");
    if (present < 0) {
        fail("checking _FillValue attribute", path);
    }
    if (present == 0 || !data_type.readable) {
        return std::nullopt;
    }

    const std::string context = path + "@_FillValue";
    H5Handle attribute(
        checked_id(H5Aopen(dataset, "_FillValue", H5P_DEFAULT), "opening attribute", context),
        H5Aclose);
    H5Handle space(
        checked_id(H5Aget_space(attribute.get()), "getting attribute dataspace", context),
        H5Sclose);
    if (H5Sget_simple_extent_npoints(space.get()) != 1) {
        fail("expected scalar _FillValue attribute", context);
    }
    auto memory_type = native_memory_type(data_type, context);
    std::vector<std::byte> bytes(data_type.element_size);
    checked_status(
        H5Aread(attribute.get(), memory_type.id, bytes.data()),
        "reading _FillValue attribute",
        context);
    return decode_fill_value(std::move(bytes), data_type);
}

[[nodiscard]] StorageLayout storage_layout(H5D_layout_t layout) {
    switch (layout) {
    case H5D_CONTIGUOUS:
        return StorageLayout::contiguous;
    case H5D_CHUNKED:
        return StorageLayout::chunked;
    case H5D_COMPACT:
        return StorageLayout::compact;
    case H5D_VIRTUAL:
        return StorageLayout::virtual_dataset;
    default:
        return StorageLayout::unknown;
    }
}

[[nodiscard]] std::vector<FilterInfo> inspect_filters(
    hid_t creation_properties,
    const std::string& path) {
    const int count = H5Pget_nfilters(creation_properties);
    if (count < 0) {
        fail("getting filter count", path);
    }

    std::vector<FilterInfo> result;
    result.reserve(static_cast<std::size_t>(count));
    for (unsigned index = 0; index < static_cast<unsigned>(count); ++index) {
        unsigned flags = 0;
        unsigned configuration = 0;
        std::size_t client_value_count = 0;
        auto filter = H5Pget_filter2(
            creation_properties,
            index,
            &flags,
            &client_value_count,
            nullptr,
            0,
            nullptr,
            &configuration);
        if (filter < 0) {
            fail("querying filter", path);
        }

        std::vector<unsigned> client_values(client_value_count);
        std::array<char, 256> name{};
        auto writable_count = client_value_count;
        filter = H5Pget_filter2(
            creation_properties,
            index,
            &flags,
            &writable_count,
            client_values.data(),
            name.size(),
            name.data(),
            &configuration);
        if (filter < 0) {
            fail("reading filter", path);
        }
        client_values.resize(writable_count);
        result.push_back(FilterInfo{
            .id = static_cast<unsigned>(filter),
            .flags = flags,
            .configuration = configuration,
            .name = name.data(),
            .client_data = std::move(client_values),
        });
    }
    return result;
}

[[nodiscard]] std::size_t recommended_cache_size(
    const DatasetInfo& dataset,
    const Hdf5OpenOptions& options) {
    if (!dataset.chunk_dimensions.has_value() || !dataset.data_type.readable ||
        options.chunk_cache_bytes == 0) {
        return 0;
    }
    const auto chunk_bytes = checked_size_product(
        (*dataset.chunk_dimensions)[0],
        (*dataset.chunk_dimensions)[1],
        dataset.data_type.file_element_size,
        dataset.path);
    if (chunk_bytes == 0) {
        return 0;
    }
    return std::max(chunk_bytes, options.chunk_cache_bytes);
}

[[nodiscard]] std::vector<std::string> group_children(
    hid_t file,
    const std::string& group_path) {
    H5Handle group(
        checked_id(H5Gopen2(file, group_path.c_str(), H5P_DEFAULT), "opening group", group_path),
        H5Gclose);
    H5G_info_t info{};
    checked_status(H5Gget_info(group.get(), &info), "getting group information", group_path);

    std::vector<std::string> result;
    if (info.nlinks > 1'000'000) {
        fail("refusing implausibly large group", group_path);
    }
    result.reserve(static_cast<std::size_t>(info.nlinks));
    for (hsize_t index = 0; index < info.nlinks; ++index) {
        const auto name_size = H5Lget_name_by_idx(
            group.get(),
            ".",
            H5_INDEX_NAME,
            H5_ITER_INC,
            index,
            nullptr,
            0,
            H5P_DEFAULT);
        if (name_size < 0) {
            fail("getting child link name size", group_path);
        }
        std::string name(static_cast<std::size_t>(name_size) + 1, '\0');
        const auto copied = H5Lget_name_by_idx(
            group.get(),
            ".",
            H5_INDEX_NAME,
            H5_ITER_INC,
            index,
            name.data(),
            name.size(),
            H5P_DEFAULT);
        if (copied < 0) {
            fail("getting child link name", group_path);
        }
        name.resize(static_cast<std::size_t>(copied));
        result.push_back(std::move(name));
    }
    return result;
}

void collect_dataset_paths(
    hid_t file,
    const std::string& group_path,
    std::vector<std::string>& output,
    unsigned depth = 0) {
    if (depth > 64) {
        fail("group nesting exceeds safety limit", group_path);
    }
    H5Handle group(
        checked_id(H5Gopen2(file, group_path.c_str(), H5P_DEFAULT), "opening group", group_path),
        H5Gclose);
    for (const auto& name : group_children(file, group_path)) {
        H5O_info2_t object_info{};
        checked_status(
            H5Oget_info_by_name3(
                group.get(),
                name.c_str(),
                &object_info,
                H5O_INFO_BASIC,
                H5P_DEFAULT),
            "getting object information",
            group_path + "/" + name);
        const auto path = group_path + "/" + name;
        if (object_info.type == H5O_TYPE_DATASET) {
            output.push_back(path);
        } else if (object_info.type == H5O_TYPE_GROUP) {
            collect_dataset_paths(file, path, output, depth + 1);
        }
    }
}

[[nodiscard]] std::optional<std::string> first_existing_path(
    hid_t file,
    const std::vector<std::string>& candidates) {
    for (const auto& candidate : candidates) {
        if (link_exists(file, candidate)) {
            return candidate;
        }
    }
    return std::nullopt;
}

[[nodiscard]] double read_axis_value(
    hid_t file,
    const std::string& path,
    hsize_t index) {
    H5Handle dataset(
        checked_id(H5Dopen2(file, path.c_str(), H5P_DEFAULT), "opening axis dataset", path),
        H5Dclose);
    H5Handle file_space(
        checked_id(H5Dget_space(dataset.get()), "getting axis dataspace", path),
        H5Sclose);
    const hsize_t count = 1;
    checked_status(
        H5Sselect_hyperslab(
            file_space.get(),
            H5S_SELECT_SET,
            &index,
            nullptr,
            &count,
            nullptr),
        "selecting axis coordinate",
        path);
    H5Handle memory_space(
        checked_id(H5Screate(H5S_SCALAR), "creating axis memory dataspace", path),
        H5Sclose);
    double result = 0.0;
    checked_status(
        H5Dread(
            dataset.get(),
            H5T_NATIVE_DOUBLE,
            memory_space.get(),
            file_space.get(),
            H5P_DEFAULT,
            &result),
        "reading axis coordinate",
        path);
    return result;
}

[[nodiscard]] AxisInfo inspect_axis(
    hid_t file,
    const std::string& axis_path,
    const std::optional<double>& declared_spacing) {
    H5Handle dataset(
        checked_id(
            H5Dopen2(file, axis_path.c_str(), H5P_DEFAULT),
            "opening coordinate axis",
            axis_path),
        H5Dclose);
    H5Handle space(
        checked_id(H5Dget_space(dataset.get()), "getting coordinate dataspace", axis_path),
        H5Sclose);
    const auto rank = H5Sget_simple_extent_ndims(space.get());
    if (rank != 1) {
        fail("expected one-dimensional coordinate axis", axis_path);
    }
    hsize_t count = 0;
    checked_status(
        H5Sget_simple_extent_dims(space.get(), &count, nullptr),
        "getting coordinate axis dimensions",
        axis_path);
    if (count == 0) {
        fail("coordinate axis is empty", axis_path);
    }

    AxisInfo result;
    result.dataset_path = axis_path;
    result.count = count;
    result.first = read_axis_value(file, axis_path, 0);
    result.last = read_axis_value(file, axis_path, count - 1);
    if (declared_spacing.has_value()) {
        result.spacing = *declared_spacing;
    } else if (count > 1) {
        result.spacing = read_axis_value(file, axis_path, 1) - result.first;
    }

    if (count <= 1) {
        result.spacing_consistent_with_endpoints = true;
    } else {
        const double expected_last =
            result.first + result.spacing * static_cast<double>(count - 1);
        const double scale =
            std::max({1.0, std::abs(expected_last), std::abs(result.last)});
        result.spacing_consistent_with_endpoints =
            std::abs(expected_last - result.last) <=
            64.0 * std::numeric_limits<double>::epsilon() * scale *
                static_cast<double>(count);
    }
    return result;
}

[[nodiscard]] GridInfo inspect_grid(
    hid_t file,
    const std::string& product_grid_path,
    const std::string& frequency_path) {
    const auto x_path = first_existing_path(
        file,
        {frequency_path + "/xCoordinates", product_grid_path + "/xCoordinates"});
    const auto y_path = first_existing_path(
        file,
        {frequency_path + "/yCoordinates", product_grid_path + "/yCoordinates"});
    if (!x_path.has_value() || !y_path.has_value()) {
        fail("locating frequency coordinate axes", frequency_path);
    }

    const auto x_spacing_path = first_existing_path(
        file,
        {frequency_path + "/xCoordinateSpacing",
         product_grid_path + "/xCoordinateSpacing"});
    const auto y_spacing_path = first_existing_path(
        file,
        {frequency_path + "/yCoordinateSpacing",
         product_grid_path + "/yCoordinateSpacing"});
    const auto x_spacing = x_spacing_path.has_value()
                               ? read_scalar_double_if_present(file, *x_spacing_path)
                               : std::nullopt;
    const auto y_spacing = y_spacing_path.has_value()
                               ? read_scalar_double_if_present(file, *y_spacing_path)
                               : std::nullopt;

    GridInfo result;
    result.x = inspect_axis(file, *x_path, x_spacing);
    result.y = inspect_axis(file, *y_path, y_spacing);

    const auto projection_path = first_existing_path(
        file,
        {frequency_path + "/projection", product_grid_path + "/projection"});
    if (projection_path.has_value()) {
        result.projection_dataset_path = *projection_path;
        result.epsg = read_scalar_u32_if_present(file, *projection_path);
        if (!result.epsg.has_value()) {
            H5Handle projection(
                checked_id(
                    H5Dopen2(file, projection_path->c_str(), H5P_DEFAULT),
                    "opening projection dataset",
                    *projection_path),
                H5Dclose);
            result.epsg =
                read_u32_attribute_if_present(projection.get(), *projection_path, "epsg_code");
        }
    }
    return result;
}

[[nodiscard]] bool starts_with_path(
    const std::string& path,
    const std::string& prefix) {
    return path.size() > prefix.size() && path.compare(0, prefix.size(), prefix) == 0 &&
           path[prefix.size()] == '/';
}

[[nodiscard]] bool contains_value(
    const std::vector<std::string>& values,
    std::string_view needle) {
    return std::find(values.begin(), values.end(), needle) != values.end();
}

[[nodiscard]] bool is_diagonal_covariance(std::string_view term) {
    return term.size() >= 4 && term.size() % 2 == 0 &&
           term.substr(0, term.size() / 2) == term.substr(term.size() / 2);
}

struct LayerClassification {
    std::string frequency;
    DatasetRole role = DatasetRole::auxiliary;
    LayerKind kind = LayerKind::auxiliary;
};

[[nodiscard]] LayerClassification classify_layer(
    ProductType product_type,
    const std::vector<FrequencyCatalog>& frequencies,
    const std::string& path,
    const DataTypeInfo& data_type) {
    LayerClassification result;
    const auto name = leaf_name(path);
    for (const auto& frequency : frequencies) {
        if (!starts_with_path(path, frequency.group_path)) {
            continue;
        }
        result.frequency = frequency.name;
        if (product_type == ProductType::gslc &&
            contains_value(frequency.polarizations, name)) {
            result.role = DatasetRole::science;
            result.kind = LayerKind::gslc_polarization;
            return result;
        }
        if (product_type == ProductType::gcov &&
            contains_value(frequency.covariance_terms, name)) {
            result.role = DatasetRole::science;
            result.kind = is_diagonal_covariance(name)
                              ? LayerKind::gcov_diagonal_covariance
                              : LayerKind::gcov_off_diagonal_covariance;
            return result;
        }

        if (name == "mask" || name == "inputDataExceptionMask") {
            result.kind = LayerKind::mask;
        } else if (name == "numberOfLooks") {
            result.kind = LayerKind::number_of_looks;
        } else if (name == "rtcGammaToSigmaFactor") {
            result.kind = LayerKind::rtc_gamma_to_sigma_factor;
        }
        return result;
    }

    if (path.find("/calibrationInformation/") != std::string::npos &&
        data_type.kind != ScalarKind::unsupported) {
        result.kind = LayerKind::calibration_lut;
    }
    return result;
}

[[nodiscard]] DatasetInfo inspect_dataset(
    hid_t file,
    const std::string& path,
    ProductType product_type,
    const std::vector<FrequencyCatalog>& frequencies,
    const Hdf5OpenOptions& options) {
    H5Handle dataset(
        checked_id(H5Dopen2(file, path.c_str(), H5P_DEFAULT), "opening dataset", path),
        H5Dclose);
    H5Handle space(
        checked_id(H5Dget_space(dataset.get()), "getting dataspace", path),
        H5Sclose);
    if (H5Sget_simple_extent_ndims(space.get()) != 2) {
        fail("expected a two-dimensional catalog dataset", path);
    }
    std::array<hsize_t, 2> dimensions{};
    checked_status(
        H5Sget_simple_extent_dims(space.get(), dimensions.data(), nullptr),
        "getting dataset dimensions",
        path);

    H5Handle type(
        checked_id(H5Dget_type(dataset.get()), "getting datatype", path),
        H5Tclose);
    H5Handle creation_properties(
        checked_id(H5Dget_create_plist(dataset.get()), "getting creation properties", path),
        H5Pclose);

    DatasetInfo result;
    result.path = path;
    result.name = leaf_name(path);
    result.dimensions = {dimensions[0], dimensions[1]};
    result.data_type = inspect_data_type(type.get(), path);
    result.data_type.file_representation_is_native =
        file_representation_is_native(
            type.get(), result.data_type, path);
    const auto hdf5_layout = H5Pget_layout(creation_properties.get());
    if (hdf5_layout == H5D_LAYOUT_ERROR) {
        fail("getting storage layout", path);
    }
    result.storage_layout = storage_layout(hdf5_layout);
    if (hdf5_layout == H5D_CHUNKED) {
        std::array<hsize_t, 2> chunks{};
        const int chunk_rank =
            H5Pget_chunk(creation_properties.get(),
                         static_cast<int>(chunks.size()), chunks.data());
        if (chunk_rank != 2 || chunks[0] == 0 || chunks[1] == 0) {
            fail("getting two-dimensional chunk layout", path);
        }
        result.chunk_dimensions = std::array<std::uint64_t, 2>{chunks[0], chunks[1]};
    }
    result.filters = inspect_filters(creation_properties.get(), path);
    result.creation_fill_value =
        creation_fill_value(creation_properties.get(), result.data_type, path);
    result.fill_value_attribute =
        fill_value_attribute(dataset.get(), result.data_type, path);
    result.units = read_string_attribute_if_present(dataset.get(), path, "units");
    result.long_name = read_string_attribute_if_present(dataset.get(), path, "long_name");
    result.description =
        read_string_attribute_if_present(dataset.get(), path, "description");
    result.grid_mapping =
        read_string_attribute_if_present(dataset.get(), path, "grid_mapping");

    const auto classification =
        classify_layer(product_type, frequencies, path, result.data_type);
    result.frequency = classification.frequency;
    result.role = classification.role;
    result.layer_kind = classification.kind;
    result.recommended_chunk_cache_bytes = recommended_cache_size(result, options);
    return result;
}

constexpr std::size_t kMaximumDirectChunkBytes =
    16ULL * 1024ULL * 1024ULL;

struct DirectChunkScratch {
    std::vector<std::byte> filtered;
    std::vector<std::byte> reversed_deflate;
};

[[nodiscard]] DirectChunkScratch& direct_chunk_scratch() {
    thread_local DirectChunkScratch scratch;
    return scratch;
}

[[nodiscard]] bool is_deflate_filter(const FilterInfo& filter) noexcept {
    return filter.id == static_cast<unsigned>(H5Z_FILTER_DEFLATE) &&
        filter.client_data.size() == 1 &&
        filter.client_data.front() <= 9;
}

[[nodiscard]] bool direct_filter_pipeline_supported(
    const DatasetInfo& dataset) noexcept {
    if (dataset.filters.size() == 1) {
        return is_deflate_filter(dataset.filters.front());
    }
    if (dataset.filters.size() != 2) {
        return false;
    }
    const auto& shuffle = dataset.filters.front();
    return shuffle.id == static_cast<unsigned>(H5Z_FILTER_SHUFFLE) &&
        shuffle.client_data.size() == 1 &&
        shuffle.client_data.front() ==
            dataset.data_type.file_element_size &&
        is_deflate_filter(dataset.filters.back());
}

[[nodiscard]] bool direct_chunk_shape_supported(
    const DatasetInfo& dataset,
    const ReadPlan& plan) noexcept {
    if (dataset.storage_layout != StorageLayout::chunked ||
        !dataset.chunk_dimensions.has_value() ||
        !dataset.data_type.file_representation_is_native ||
        dataset.data_type.file_element_size == 0 ||
        dataset.data_type.file_element_size !=
            dataset.data_type.element_size ||
        !direct_filter_pipeline_supported(dataset)) {
        return false;
    }
    const auto chunk_rows = (*dataset.chunk_dimensions)[0];
    const auto chunk_columns = (*dataset.chunk_dimensions)[1];
    if (plan.aligned.row % chunk_rows != 0 ||
        plan.aligned.column % chunk_columns != 0 ||
        plan.aligned.height > chunk_rows ||
        plan.aligned.width > chunk_columns) {
        return false;
    }
    if (chunk_rows >
            std::numeric_limits<std::size_t>::max() / chunk_columns) {
        return false;
    }
    const auto chunk_elements =
        static_cast<std::size_t>(chunk_rows * chunk_columns);
    if (dataset.data_type.element_size >
            std::numeric_limits<std::size_t>::max() / chunk_elements) {
        return false;
    }
    const auto chunk_bytes =
        chunk_elements * dataset.data_type.element_size;
    return chunk_bytes != 0 &&
        chunk_bytes <= kMaximumDirectChunkBytes &&
        chunk_bytes <= static_cast<std::size_t>(ULONG_MAX);
}

void copy_cropped_chunk(
    const std::span<const std::byte> source,
    const std::span<std::byte> destination,
    const std::size_t element_size,
    const std::uint64_t chunk_columns,
    const std::uint64_t rows,
    const std::uint64_t columns,
    const std::string& path) {
    const auto source_row_bytes = checked_size_product(
        1, chunk_columns, element_size, path);
    const auto destination_row_bytes = checked_size_product(
        1, columns, element_size, path);
    if (destination.size() != checked_size_product(
            rows, columns, element_size, path) ||
        source.size() < checked_size_product(
            rows, chunk_columns, element_size, path)) {
        fail("validating decoded chunk dimensions", path);
    }
    for (std::size_t row = 0; row < rows; ++row) {
        std::memcpy(
            destination.data() + row * destination_row_bytes,
            source.data() + row * source_row_bytes,
            destination_row_bytes);
    }
}

void unshuffle_cropped_chunk(
    const std::span<const std::byte> shuffled,
    const std::span<std::byte> destination,
    const std::size_t element_size,
    const std::uint64_t chunk_columns,
    const std::uint64_t rows,
    const std::uint64_t columns,
    const std::string& path) {
    if (element_size == 0 || shuffled.size() % element_size != 0 ||
        destination.size() != checked_size_product(
            rows, columns, element_size, path)) {
        fail("validating shuffled chunk dimensions", path);
    }
    const auto elements = shuffled.size() / element_size;
    if (rows != 0 &&
        (rows - 1) * chunk_columns + columns > elements) {
        fail("cropping shuffled edge chunk", path);
    }
    if (element_size == 1) {
        copy_cropped_chunk(
            shuffled, destination, element_size,
            chunk_columns, rows, columns, path);
        return;
    }

    for (std::size_t row = 0; row < rows; ++row) {
        const auto source_row = row * chunk_columns;
        const auto destination_row = row * columns;
        for (std::size_t column = 0; column < columns; ++column) {
            const auto source_element = source_row + column;
            auto* output = destination.data() +
                (destination_row + column) * element_size;
            switch (element_size) {
            case 8:
                output[7] = shuffled[7 * elements + source_element];
                output[6] = shuffled[6 * elements + source_element];
                output[5] = shuffled[5 * elements + source_element];
                output[4] = shuffled[4 * elements + source_element];
                [[fallthrough]];
            case 4:
                output[3] = shuffled[3 * elements + source_element];
                output[2] = shuffled[2 * elements + source_element];
                [[fallthrough]];
            case 2:
                output[1] = shuffled[elements + source_element];
                output[0] = shuffled[source_element];
                break;
            default:
                for (std::size_t byte = 0; byte < element_size; ++byte) {
                    output[byte] = shuffled[byte * elements + source_element];
                }
                break;
            }
        }
    }
}

void decode_direct_chunk(
    DirectChunkScratch& scratch,
    const std::uint32_t filter_mask,
    const DatasetInfo& dataset,
    const ReadPlan& plan,
    const std::span<std::byte> destination) {
    const auto chunk_rows = (*dataset.chunk_dimensions)[0];
    const auto chunk_columns = (*dataset.chunk_dimensions)[1];
    const auto chunk_bytes = checked_size_product(
        chunk_rows,
        chunk_columns,
        dataset.data_type.element_size,
        dataset.path);
    const auto filter_count =
        static_cast<unsigned>(dataset.filters.size());
    const auto valid_mask = (1U << filter_count) - 1U;
    if ((filter_mask & ~valid_mask) != 0) {
        fail("decoding chunk with an invalid filter mask", dataset.path);
    }

    const auto deflate_index = filter_count - 1U;
    const bool deflate_applied =
        (filter_mask & (1U << deflate_index)) == 0;
    if (deflate_applied) {
        scratch.reversed_deflate.resize(chunk_bytes);
        uLongf output_size = static_cast<uLongf>(chunk_bytes);
        const auto status = uncompress(
            reinterpret_cast<Bytef*>(
                scratch.reversed_deflate.data()),
            &output_size,
            reinterpret_cast<const Bytef*>(scratch.filtered.data()),
            static_cast<uLong>(scratch.filtered.size()));
        if (status != Z_OK ||
            output_size != static_cast<uLongf>(chunk_bytes)) {
            fail("reversing the DEFLATE chunk filter", dataset.path);
        }
    } else {
        if (scratch.filtered.size() != chunk_bytes) {
            fail("validating an uncompressed direct chunk", dataset.path);
        }
        scratch.reversed_deflate.assign(
            scratch.filtered.begin(), scratch.filtered.end());
    }

    const bool shuffle_applied = dataset.filters.size() == 2 &&
        (filter_mask & 1U) == 0;
    if (shuffle_applied) {
        unshuffle_cropped_chunk(
            scratch.reversed_deflate,
            destination,
            dataset.data_type.element_size,
            chunk_columns,
            plan.aligned.height,
            plan.aligned.width,
            dataset.path);
    } else {
        copy_cropped_chunk(
            scratch.reversed_deflate,
            destination,
            dataset.data_type.element_size,
            chunk_columns,
            plan.aligned.height,
            plan.aligned.width,
            dataset.path);
    }
}

[[nodiscard]] bool same_read_plan_metadata(
    const ReadPlan& plan,
    const ReadPlan& canonical) noexcept {
    return plan.aligned.row == canonical.aligned.row &&
        plan.aligned.column == canonical.aligned.column &&
        plan.aligned.height == canonical.aligned.height &&
        plan.aligned.width == canonical.aligned.width &&
        plan.requested_row_offset == canonical.requested_row_offset &&
        plan.requested_column_offset == canonical.requested_column_offset &&
        plan.dataset_dimensions == canonical.dataset_dimensions &&
        plan.data_type == canonical.data_type &&
        plan.expected_bytes == canonical.expected_bytes;
}

[[nodiscard]] std::uint64_t align_end(
    std::uint64_t end,
    std::uint64_t quantum,
    std::uint64_t limit) {
    const auto remainder = end % quantum;
    if (remainder == 0) {
        return end;
    }
    const auto addition = quantum - remainder;
    if (addition > limit - end) {
        return limit;
    }
    return end + addition;
}

} // namespace

class Hdf5Product::Impl {
public:
    struct CachedDataset {
        std::string path;
        H5Handle dataset;
        NativeType memory_type;
        // Selections are replaced under read_mutex_. Keeping one file space
        // and the most recent memory shape avoids two HDF5 handle cycles for
        // nearly every chunk without allowing a shape-keyed cache to grow.
        H5Handle file_space;
        H5Handle memory_space;
        std::array<hsize_t, 2> memory_dimensions{};
    };

    Impl(std::filesystem::path file_path, Hdf5OpenOptions options)
        : options_(options) {
        std::scoped_lock hdf5_lock(global_hdf5_mutex());
        if (file_path.empty()) {
            throw Hdf5Error("HDF5 file path is empty");
        }
        if (options_.chunk_cache_slots == 0) {
            throw Hdf5Error("HDF5 chunk cache slot count must be positive");
        }
        if (options_.read_dataset_cache_entries == 0) {
            throw Hdf5Error("HDF5 read dataset cache entry count must be positive");
        }
        if (!std::isfinite(options_.chunk_cache_preemption) ||
            options_.chunk_cache_preemption < 0.0 ||
            options_.chunk_cache_preemption > 1.0) {
            throw Hdf5Error("HDF5 chunk cache preemption must be in [0, 1]");
        }

        std::error_code path_error;
        file_path_ = std::filesystem::absolute(file_path, path_error).lexically_normal();
        if (path_error) {
            file_path_ = std::move(file_path).lexically_normal();
        }
        source_file_ = file_path_.string();
        file_ = H5Handle(
            checked_id(
                H5Fopen(source_file_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT),
                "opening file",
                source_file_),
            H5Fclose);

        try {
            parse_identification();
            parse_frequencies();
            parse_datasets();
        } catch (...) {
            file_.reset();
            throw;
        }
    }

    ~Impl() {
        std::scoped_lock hdf5_lock(global_hdf5_mutex());
        std::scoped_lock read_lock(read_mutex_);
        dataset_cache_.clear();
        file_.reset();
    }

    [[nodiscard]] const DatasetInfo* find_dataset(std::string_view input_path) const {
        const auto path = normalize_hdf5_path(input_path);
        const auto iterator = dataset_index_.find(path);
        return iterator == dataset_index_.end() ? nullptr : &datasets_[iterator->second];
    }

    [[nodiscard]] ReadPlan make_read_plan(
        std::string_view input_path,
        std::uint64_t row,
        std::uint64_t column,
        std::uint64_t height,
        std::uint64_t width) const {
        const auto* dataset = find_dataset(input_path);
        if (dataset == nullptr) {
            throw Hdf5Error(
                "dataset is not a cataloged two-dimensional layer [" +
                normalize_hdf5_path(input_path) + ']');
        }
        if (!dataset->data_type.readable) {
            throw Hdf5Error("dataset datatype is not readable [" + dataset->path + ']');
        }
        if (height == 0 || width == 0) {
            throw Hdf5Error("read window must have non-zero width and height [" + dataset->path +
                            ']');
        }
        if (row >= dataset->dimensions[0] || column >= dataset->dimensions[1] ||
            height > dataset->dimensions[0] - row ||
            width > dataset->dimensions[1] - column) {
            throw Hdf5Error("read window is outside dataset bounds [" + dataset->path + ']');
        }

        ReadPlan result;
        result.source_file = source_file_;
        result.dataset_path = dataset->path;
        result.requested = Window2D{
            .row = row,
            .column = column,
            .height = height,
            .width = width,
        };
        result.aligned = result.requested;
        if (dataset->chunk_dimensions.has_value()) {
            const auto chunk_rows = (*dataset->chunk_dimensions)[0];
            const auto chunk_columns = (*dataset->chunk_dimensions)[1];
            result.aligned.row = row - row % chunk_rows;
            result.aligned.column = column - column % chunk_columns;
            const auto requested_row_end = row + height;
            const auto requested_column_end = column + width;
            const auto aligned_row_end =
                align_end(requested_row_end, chunk_rows, dataset->dimensions[0]);
            const auto aligned_column_end =
                align_end(requested_column_end, chunk_columns, dataset->dimensions[1]);
            result.aligned.height = aligned_row_end - result.aligned.row;
            result.aligned.width = aligned_column_end - result.aligned.column;
        }
        result.requested_row_offset = row - result.aligned.row;
        result.requested_column_offset = column - result.aligned.column;
        result.dataset_dimensions = dataset->dimensions;
        result.data_type = dataset->data_type;
        result.expected_bytes = checked_size_product(
            result.aligned.height,
            result.aligned.width,
            result.data_type.element_size,
            dataset->path);
        return result;
    }

    [[nodiscard]] bool supports_direct_chunk_decode(
        const ReadPlan& plan) const {
        if (plan.source_file != source_file_) {
            return false;
        }
        const auto* descriptor = find_dataset(plan.dataset_path);
        if (descriptor == nullptr) {
            return false;
        }
        try {
            const auto canonical = make_read_plan(
                plan.dataset_path,
                plan.requested.row,
                plan.requested.column,
                plan.requested.height,
                plan.requested.width);
            return same_read_plan_metadata(plan, canonical) &&
                direct_chunk_shape_supported(
                    *descriptor, canonical);
        } catch (const Hdf5Error&) {
            return false;
        }
    }

    void read_into(
        const ReadPlan& plan,
        std::span<std::byte> destination,
        const bool prefer_direct_chunk) const {
        if (plan.source_file != source_file_) {
            throw Hdf5Error("read plan belongs to a different HDF5 product [" +
                            plan.dataset_path + ']');
        }
        const auto canonical = make_read_plan(
            plan.dataset_path,
            plan.requested.row,
            plan.requested.column,
            plan.requested.height,
            plan.requested.width);
        if (!same_read_plan_metadata(plan, canonical)) {
            throw Hdf5Error("read plan metadata does not match the dataset [" +
                            plan.dataset_path + ']');
        }
        if (destination.size() != canonical.expected_bytes) {
            throw Hdf5Error(
                "destination byte count mismatch for HDF5 read: expected " +
                std::to_string(canonical.expected_bytes) + ", received " +
                std::to_string(destination.size()) + " [" + canonical.dataset_path + ']');
        }

        const auto* descriptor = find_dataset(canonical.dataset_path);
        if (descriptor == nullptr) {
            fail("finding planned dataset", canonical.dataset_path);
        }

        std::unique_lock hdf5_lock(global_hdf5_mutex());
        std::unique_lock read_lock(read_mutex_);
        auto cached = std::find_if(
            dataset_cache_.begin(),
            dataset_cache_.end(),
            [&](const CachedDataset& entry) {
                return entry.path == descriptor->path;
            });
        if (cached != dataset_cache_.end()) {
            dataset_cache_.splice(dataset_cache_.begin(), dataset_cache_, cached);
        } else {
            H5Handle access_properties(
                checked_id(
                    H5Pcreate(H5P_DATASET_ACCESS),
                    "creating dataset access properties",
                    descriptor->path),
                H5Pclose);
            if (descriptor->recommended_chunk_cache_bytes > 0) {
                checked_status(
                    H5Pset_chunk_cache(
                        access_properties.get(),
                        options_.chunk_cache_slots,
                        descriptor->recommended_chunk_cache_bytes,
                        options_.chunk_cache_preemption),
                    "configuring raw chunk cache",
                    descriptor->path);
            }

            H5Handle dataset(
                checked_id(
                    H5Dopen2(file_.get(), descriptor->path.c_str(), access_properties.get()),
                    "opening dataset for read",
                    descriptor->path),
                H5Dclose);
            H5Handle file_type(
                checked_id(
                    H5Dget_type(dataset.get()),
                    "getting datatype for read",
                    descriptor->path),
                H5Tclose);
            auto current_type = inspect_data_type(
                file_type.get(), descriptor->path);
            current_type.file_representation_is_native =
                file_representation_is_native(
                    file_type.get(), current_type, descriptor->path);
            if (current_type != canonical.data_type) {
                fail("dataset datatype changed since plan creation", descriptor->path);
            }
            auto memory_type = native_memory_type(current_type, descriptor->path);
            H5Handle file_space(
                checked_id(
                    H5Dget_space(dataset.get()),
                    "getting dataset dataspace for read",
                    descriptor->path),
                H5Sclose);
            dataset_cache_.emplace_front(CachedDataset{
                .path = descriptor->path,
                .dataset = std::move(dataset),
                .memory_type = std::move(memory_type),
                .file_space = std::move(file_space),
            });
            if (dataset_cache_.size() > options_.read_dataset_cache_entries) {
                dataset_cache_.pop_back();
            }
        }
        auto& cached_dataset = dataset_cache_.front();
        const hid_t dataset_id = cached_dataset.dataset.get();
        const hid_t memory_type_id = cached_dataset.memory_type.id;
        const hid_t file_space_id = cached_dataset.file_space.get();

        if (prefer_direct_chunk &&
            direct_chunk_shape_supported(*descriptor, canonical)) {
            const std::array<hsize_t, 2> chunk_offset{
                canonical.aligned.row,
                canonical.aligned.column,
            };
            unsigned info_filter_mask = 0;
            haddr_t chunk_address = HADDR_UNDEF;
            hsize_t stored_size = 0;
            checked_status(
                H5Dget_chunk_info_by_coord(
                    dataset_id,
                    chunk_offset.data(),
                    &info_filter_mask,
                    &chunk_address,
                    &stored_size),
                "querying direct chunk storage",
                descriptor->path);
            if (chunk_address != HADDR_UNDEF && stored_size != 0 &&
                stored_size <= kMaximumDirectChunkBytes &&
                stored_size <= static_cast<hsize_t>(ULONG_MAX)) {
                auto& scratch = direct_chunk_scratch();
                scratch.filtered.resize(
                    static_cast<std::size_t>(stored_size));
                std::size_t buffer_size = scratch.filtered.size();
                std::uint32_t filter_mask = 0;
                checked_status(
                    H5Dread_chunk2(
                        dataset_id,
                        H5P_DEFAULT,
                        chunk_offset.data(),
                        &filter_mask,
                        scratch.filtered.data(),
                        &buffer_size),
                    "reading direct filtered chunk",
                    descriptor->path);
                if (filter_mask !=
                    static_cast<std::uint32_t>(info_filter_mask)) {
                    fail("direct chunk filter mask changed", descriptor->path);
                }
                if (buffer_size > scratch.filtered.size()) {
                    fail(
                        "validating direct filtered chunk size",
                        descriptor->path);
                }
                scratch.filtered.resize(buffer_size);
                read_lock.unlock();
                hdf5_lock.unlock();
                decode_direct_chunk(
                    scratch,
                    filter_mask,
                    *descriptor,
                    canonical,
                    destination);
                return;
            }
        }

        const std::array<hsize_t, 2> start{
            canonical.aligned.row,
            canonical.aligned.column,
        };
        const std::array<hsize_t, 2> count{
            canonical.aligned.height,
            canonical.aligned.width,
        };
        checked_status(
            H5Sselect_hyperslab(
                file_space_id,
                H5S_SELECT_SET,
                start.data(),
                nullptr,
                count.data(),
                nullptr),
            "selecting read hyperslab",
            descriptor->path);
        if (!cached_dataset.memory_space ||
            cached_dataset.memory_dimensions != count) {
            cached_dataset.memory_space = H5Handle(
                checked_id(
                    H5Screate_simple(2, count.data(), nullptr),
                    "creating read memory dataspace",
                    descriptor->path),
                H5Sclose);
            cached_dataset.memory_dimensions = count;
        }
        checked_status(
            H5Dread(
                dataset_id,
                memory_type_id,
                cached_dataset.memory_space.get(),
                file_space_id,
                H5P_DEFAULT,
                destination.data()),
            "reading chunk-aligned hyperslab",
            descriptor->path);
    }

    std::filesystem::path file_path_;
    std::string source_file_;
    Hdf5OpenOptions options_;
    H5Handle file_;
    Identification identification_;
    std::vector<FrequencyCatalog> frequencies_;
    std::vector<DatasetInfo> datasets_;
    std::unordered_map<std::string, std::size_t> dataset_index_;
    mutable std::mutex read_mutex_;
    // Most recently used at the front; least recently used at the back.
    mutable std::list<CachedDataset> dataset_cache_;

private:
    void parse_identification() {
        std::string instrument;
        if (link_exists(file_.get(), "/science/LSAR/identification")) {
            instrument = "LSAR";
        } else if (link_exists(file_.get(), "/science/SSAR/identification")) {
            instrument = "SSAR";
        } else {
            fail("locating NISAR identification group", source_file_);
        }

        const auto identification_path = "/science/" + instrument + "/identification";
        const auto type_text =
            read_scalar_string(file_.get(), identification_path + "/productType", true);
        const auto normalized_type = uppercase_ascii(type_text);
        ProductType product_type;
        if (normalized_type == "GSLC") {
            product_type = ProductType::gslc;
        } else if (normalized_type == "GCOV") {
            product_type = ProductType::gcov;
        } else {
            fail("unsupported NISAR productType '" + type_text + "'", identification_path);
        }

        const std::string product_name =
            product_type == ProductType::gslc ? "GSLC" : "GCOV";
        const auto product_path = "/science/" + instrument + "/" + product_name;
        if (!link_exists(file_.get(), product_path)) {
            fail("missing product group", product_path);
        }

        identification_ = Identification{
            .product_type = product_type,
            .product_type_text = type_text,
            .granule_id =
                read_scalar_string(file_.get(), identification_path + "/granuleId", true),
            .product_version =
                read_scalar_string(file_.get(), identification_path + "/productVersion", true),
            .product_specification_version = read_scalar_string(
                file_.get(),
                identification_path + "/productSpecificationVersion",
                false),
            .instrument_group = instrument,
            .identification_group_path = identification_path,
            .product_group_path = product_path,
        };
    }

    void parse_frequencies() {
        const auto grid_path = identification_.product_group_path + "/grids";
        if (!link_exists(file_.get(), grid_path)) {
            fail("missing geocoded grids group", grid_path);
        }
        for (const auto& child : group_children(file_.get(), grid_path)) {
            if (child.size() <= std::string_view("frequency").size() ||
                child.rfind("frequency", 0) != 0) {
                continue;
            }
            const auto frequency_path = grid_path + "/" + child;
            H5O_info2_t object_info{};
            checked_status(
                H5Oget_info_by_name3(
                    file_.get(),
                    frequency_path.c_str(),
                    &object_info,
                    H5O_INFO_BASIC,
                    H5P_DEFAULT),
                "getting frequency object information",
                frequency_path);
            if (object_info.type != H5O_TYPE_GROUP) {
                continue;
            }

            FrequencyCatalog frequency;
            frequency.name = child.substr(std::string_view("frequency").size());
            frequency.group_path = frequency_path;
            frequency.polarizations = read_string_dataset(
                file_.get(),
                frequency_path + "/listOfPolarizations",
                false);
            frequency.covariance_terms = read_string_dataset(
                file_.get(),
                frequency_path + "/listOfCovarianceTerms",
                false);
            frequency.grid = inspect_grid(file_.get(), grid_path, frequency_path);
            frequencies_.push_back(std::move(frequency));
        }

        std::sort(
            frequencies_.begin(),
            frequencies_.end(),
            [](const FrequencyCatalog& left, const FrequencyCatalog& right) {
                return left.name < right.name;
            });
        if (frequencies_.empty()) {
            fail("no frequency groups found", grid_path);
        }
    }

    void parse_datasets() {
        std::vector<std::string> paths;
        collect_dataset_paths(file_.get(), identification_.product_group_path, paths);
        std::sort(paths.begin(), paths.end());
        for (const auto& path : paths) {
            H5Handle dataset(
                checked_id(H5Dopen2(file_.get(), path.c_str(), H5P_DEFAULT), "opening dataset", path),
                H5Dclose);
            H5Handle space(
                checked_id(H5Dget_space(dataset.get()), "getting dataspace", path),
                H5Sclose);
            if (H5Sget_simple_extent_ndims(space.get()) != 2) {
                continue;
            }
            auto descriptor = inspect_dataset(
                file_.get(),
                path,
                identification_.product_type,
                frequencies_,
                options_);
            const auto index = datasets_.size();
            dataset_index_.emplace(descriptor.path, index);
            datasets_.push_back(std::move(descriptor));
        }

        for (auto& frequency : frequencies_) {
            for (const auto& dataset : datasets_) {
                if (dataset.frequency == frequency.name) {
                    frequency.layers.push_back(dataset);
                }
            }
        }
    }
};

Hdf5Product::Hdf5Product(
    std::filesystem::path file_path,
    Hdf5OpenOptions options)
    : impl_(std::make_unique<Impl>(std::move(file_path), options)) {}

Hdf5Product::~Hdf5Product() = default;
Hdf5Product::Hdf5Product(Hdf5Product&&) noexcept = default;
Hdf5Product& Hdf5Product::operator=(Hdf5Product&&) noexcept = default;

const std::filesystem::path& Hdf5Product::file_path() const noexcept {
    return impl_->file_path_;
}

const Identification& Hdf5Product::identification() const noexcept {
    return impl_->identification_;
}

const std::vector<FrequencyCatalog>& Hdf5Product::frequencies() const noexcept {
    return impl_->frequencies_;
}

const std::vector<DatasetInfo>& Hdf5Product::datasets() const noexcept {
    return impl_->datasets_;
}

const DatasetInfo* Hdf5Product::find_dataset(std::string_view path) const {
    return impl_->find_dataset(path);
}

const FrequencyCatalog* Hdf5Product::find_frequency(std::string_view name) const {
    const auto normalized = uppercase_ascii(std::string(name));
    const auto iterator = std::find_if(
        impl_->frequencies_.begin(),
        impl_->frequencies_.end(),
        [&](const FrequencyCatalog& frequency) {
            return uppercase_ascii(frequency.name) == normalized;
        });
    return iterator == impl_->frequencies_.end() ? nullptr : &*iterator;
}

ReadPlan Hdf5Product::make_read_plan(
    std::string_view dataset_path,
    std::uint64_t row,
    std::uint64_t column,
    std::uint64_t height,
    std::uint64_t width) const {
    return impl_->make_read_plan(dataset_path, row, column, height, width);
}

bool Hdf5Product::supports_direct_chunk_decode(
    const ReadPlan& plan) const {
    return impl_->supports_direct_chunk_decode(plan);
}

void Hdf5Product::read_into(
    const ReadPlan& plan,
    std::span<std::byte> destination) const {
    impl_->read_into(plan, destination, false);
}

void Hdf5Product::read_direct_chunk_into(
    const ReadPlan& plan,
    std::span<std::byte> destination) const {
    impl_->read_into(plan, destination, true);
}

} // namespace satview
