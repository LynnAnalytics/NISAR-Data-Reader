#define NOMINMAX

#include <windows.h>
#include <commdlg.h>

#include "colormap_lut.hpp"
#include "colormaps.hpp"
#include "overview_worker.hpp"

#include "satview/cpu/pageable_ring.hpp"
#include "satview/cpu/scientific.hpp"
#include "satview/distribution.hpp"
#include "satview/experimental/scientific.hpp"
#if defined(SATVIEW_HAS_CUDA)
#include "satview/gpu/distribution.hpp"
#include "satview/gpu/pinned_ring.hpp"
#include "satview/gpu/speckle_filter.hpp"
#include "satview/gpu/transforms.hpp"
#include "satview/gpu/vulkan_interop.hpp"
#endif
#include "satview/hdf5_product.hpp"
#include "satview/resident_view.hpp"
#include "satview/view_navigation.hpp"
#include "satview/viewer_math.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_vulkan.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <vulkan/vulkan.h>

#if defined(SATVIEW_HAS_CUDA)
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
#if defined(SATVIEW_HAS_CUDA)
using ReadRing = satview::gpu::PinnedRing;
#else
using ReadRing = satview::cpu::PageableRing;
#endif
using ReadSlotLease = ReadRing::SlotLease;
using ReadRingState = ReadRing::StateCounts;

constexpr std::uint32_t kContiguousTileExtent = 2048;
constexpr std::size_t kPinnedSlotCount = 3;
constexpr std::size_t kMaximumMosaicChunks = 16;
constexpr std::uint64_t kMaximumTileBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaximumMosaicBytes =
    2ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr std::uint32_t kMinimumSwapchainImages = 2;
constexpr double kPi = 3.141592653589793238462643383279502884;

[[noreturn]] void fail(std::string_view message) {
    throw std::runtime_error(std::string(message));
}
void check_vk(const VkResult result, std::string_view operation) {
    if (result != VK_SUCCESS) {
        std::ostringstream message;
        message << operation << " failed with VkResult " << result;
        throw std::runtime_error(message.str());
    }
}

#if defined(SATVIEW_HAS_CUDA)
void check_cuda(const cudaError_t result, std::string_view operation) {
    if (result != cudaSuccess) {
        std::ostringstream message;
        message << operation << " failed: " << cudaGetErrorString(result);
        throw std::runtime_error(message.str());
    }
}
#endif

enum class BackendPreference : std::uint8_t {
    automatic,
    cuda,
    cpu,
    hip,
    sycl,
};

enum class ComputeBackend : std::uint8_t {
    cuda,
    cpu,
    hip,
    sycl,
};

[[nodiscard]] constexpr std::string_view backend_name(
    const ComputeBackend backend) noexcept {
    switch (backend) {
        case ComputeBackend::cuda:
            return "CUDA";
        case ComputeBackend::cpu:
            return "CPU";
        case ComputeBackend::hip:
            return "HIP/ROCm experimental";
        case ComputeBackend::sycl:
            return "oneAPI/SYCL experimental";
    }
    return "unknown";
}

struct Arguments {
    std::filesystem::path file;
    std::optional<std::uint64_t> frame_limit;
    std::uint32_t zoom_chunks = 1;
    satview::cpu::SpeckleFilter speckle_filter =
        satview::cpu::SpeckleFilter::none;
    std::uint32_t speckle_window = 5;
    float speckle_looks = 1.0F;
    double rotation_degrees = 0.0;
    bool clean_view = false;
    bool smoke_test = false;
    bool fit_scene = false;
    BackendPreference backend = BackendPreference::automatic;
};

[[noreturn]] void usage_error(std::string_view message) {
    std::ostringstream output;
    output << message
           << "\nusage: sat-viewer [--zoom 1|2|4] "
              "[--speckle none|boxcar|lee] [--speckle-window 3|5|7] "
              "[--speckle-looks N] "
              "[--rotation DEGREES] "
              "[--backend auto|cuda|cpu|hip|sycl] "
              "[--clean-view] "
              "[--fit-scene] [--frames N | --smoke-test] "
              "[NISAR-product.h5]";
    throw std::runtime_error(output.str());
}

Arguments parse_arguments(const int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            std::cout
                << "usage: sat-viewer [--zoom 1|2|4] "
                   "[--speckle none|boxcar|lee] [--speckle-window 3|5|7] "
                   "[--speckle-looks N] "
                   "[--rotation DEGREES] "
                   "[--backend auto|cuda|cpu|hip|sycl] "
                   "[--clean-view] "
                   "[--fit-scene] [--frames N | --smoke-test] "
                   "[NISAR-product.h5]\n"
                   "  no path       open the viewer's file picker\n"
                   "  --zoom N      show a centered NxN source-chunk mosaic\n"
                   "  --speckle F   initial filter: none, boxcar, or lee\n"
                   "  --speckle-window N  initial 3x3, 5x5, or 7x7 window\n"
                   "  --speckle-looks N   initial Lee equivalent looks (> 0)\n"
                   "  --rotation N  initial clockwise rotation in degrees\n"
                   "  --backend B   compute backend; auto prefers CUDA\n"
                   "  --clean-view  start with the left controls hidden\n"
                   "  --fit-scene   start with the entire raster visible\n"
                   "  --frames N    exit after N presented frames\n"
                   "  --smoke-test  load the requested startup view, present 4 frames, "
                   "then exit\n";
            std::exit(0);
        }
        if (argument == "--fit-scene") {
            result.fit_scene = true;
            continue;
        }
        if (argument == "--clean-view") {
            result.clean_view = true;
            continue;
        }
        if (argument == "--smoke-test") {
            result.smoke_test = true;
            continue;
        }
        if (argument == "--zoom") {
            if (++index >= argc) {
                usage_error("--zoom requires 1, 2, or 4");
            }
            const std::string_view text(argv[index]);
            std::uint32_t parsed = 0;
            const auto [end, error] = std::from_chars(
                text.data(), text.data() + text.size(), parsed);
            if (text.empty() || error != std::errc{} ||
                end != text.data() + text.size() ||
                !satview::viewer::is_supported_mosaic_span(parsed)) {
                usage_error("--zoom requires 1, 2, or 4");
            }
            result.zoom_chunks = parsed;
            continue;
        }
        if (argument == "--speckle") {
            if (++index >= argc) {
                usage_error("--speckle requires none, boxcar, or lee");
            }
            const std::string_view name(argv[index]);
            if (name == "none") {
                result.speckle_filter = satview::cpu::SpeckleFilter::none;
            } else if (name == "boxcar") {
                result.speckle_filter = satview::cpu::SpeckleFilter::boxcar;
            } else if (name == "lee") {
                result.speckle_filter = satview::cpu::SpeckleFilter::lee;
            } else {
                usage_error("--speckle requires none, boxcar, or lee");
            }
            continue;
        }
        if (argument == "--speckle-window") {
            if (++index >= argc) {
                usage_error("--speckle-window requires 3, 5, or 7");
            }
            const std::string_view text(argv[index]);
            std::uint32_t parsed = 0;
            const auto [end, error] = std::from_chars(
                text.data(), text.data() + text.size(), parsed);
            if (text.empty() || error != std::errc{} ||
                end != text.data() + text.size() ||
                (parsed != 3 && parsed != 5 && parsed != 7)) {
                usage_error("--speckle-window requires 3, 5, or 7");
            }
            result.speckle_window = parsed;
            continue;
        }
        if (argument == "--speckle-looks") {
            if (++index >= argc) {
                usage_error("--speckle-looks requires a finite value > 0");
            }
            const std::string_view text(argv[index]);
            float parsed = 0.0F;
            const auto [end, error] = std::from_chars(
                text.data(), text.data() + text.size(), parsed);
            if (text.empty() || error != std::errc{} ||
                end != text.data() + text.size() ||
                !std::isfinite(parsed) || parsed <= 0.0F) {
                usage_error("--speckle-looks requires a finite value > 0");
            }
            result.speckle_looks = parsed;
            continue;
        }
        if (argument == "--rotation") {
            if (++index >= argc) {
                usage_error("--rotation requires a finite degree value");
            }
            const std::string_view text(argv[index]);
            double parsed = 0.0;
            const auto [end, error] = std::from_chars(
                text.data(), text.data() + text.size(), parsed);
            if (text.empty() || error != std::errc{} ||
                end != text.data() + text.size() ||
                !std::isfinite(parsed)) {
                usage_error("--rotation requires a finite degree value");
            }
            result.rotation_degrees = parsed;
            continue;
        }
        if (argument == "--backend") {
            if (++index >= argc) {
                usage_error("--backend requires auto, cuda, cpu, hip, or sycl");
            }
            const std::string_view name(argv[index]);
            if (name == "auto") {
                result.backend = BackendPreference::automatic;
            } else if (name == "cuda") {
                result.backend = BackendPreference::cuda;
            } else if (name == "cpu") {
                result.backend = BackendPreference::cpu;
            } else if (name == "hip" || name == "rocm") {
                result.backend = BackendPreference::hip;
            } else if (name == "sycl" || name == "oneapi") {
                result.backend = BackendPreference::sycl;
            } else {
                usage_error("--backend requires auto, cuda, cpu, hip, or sycl");
            }
            continue;
        }
        if (argument == "--frames") {
            if (++index >= argc) {
                usage_error("--frames requires a positive integer");
            }
            const std::string_view text(argv[index]);
            std::uint64_t parsed = 0;
            const auto [end, error] = std::from_chars(
                text.data(), text.data() + text.size(), parsed);
            if (text.empty() || error != std::errc{} ||
                end != text.data() + text.size() || parsed == 0) {
                usage_error("--frames requires a positive integer");
            }
            result.frame_limit = parsed;
            continue;
        }
        if (!argument.empty() && argument.front() == '-') {
            usage_error(std::string("unknown option: ") + std::string(argument));
        }
        if (!result.file.empty()) {
            usage_error("only one NISAR product may be opened");
        }
        result.file = std::filesystem::path(argument);
    }
    if (result.smoke_test && result.frame_limit.has_value()) {
        usage_error("--smoke-test and --frames are mutually exclusive");
    }
    if (result.smoke_test && result.file.empty()) {
        usage_error("--smoke-test requires a NISAR product path");
    }
    return result;
}

[[nodiscard]] bool cuda_runtime_available(std::string& reason) noexcept {
#if defined(SATVIEW_HAS_CUDA)
    int device_count = 0;
    const cudaError_t result = cudaGetDeviceCount(&device_count);
    if (result != cudaSuccess) {
        const char* const description = cudaGetErrorString(result);
        reason = description != nullptr
            ? description
            : "CUDA runtime initialization failed";
        static_cast<void>(cudaGetLastError());
        return false;
    }
    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
            properties.major == 12 && cudaSetDevice(device) == cudaSuccess) {
            reason.clear();
            return true;
        }
    }
    static_cast<void>(cudaGetLastError());
    reason = device_count == 0
        ? "no CUDA device was found"
        : "no sm_120-compatible CUDA device was found";
    return false;
#else
    reason = "CUDA support was not compiled";
    return false;
#endif
}

[[nodiscard]] ComputeBackend select_backend(
    const BackendPreference preference) {
    std::string reason;
    switch (preference) {
        case BackendPreference::automatic:
            return cuda_runtime_available(reason)
                ? ComputeBackend::cuda
                : ComputeBackend::cpu;
        case BackendPreference::cuda:
            if (!cuda_runtime_available(reason)) {
                throw std::runtime_error(
                    "CUDA backend is unavailable: " + reason);
            }
            return ComputeBackend::cuda;
        case BackendPreference::cpu:
            return ComputeBackend::cpu;
        case BackendPreference::hip:
#if defined(SATVIEW_HAS_EXPERIMENTAL_HIP)
            if (!satview::experimental::hip_runtime_available(reason)) {
                throw std::runtime_error(
                    "experimental HIP/ROCm backend is unavailable: " + reason);
            }
            return ComputeBackend::hip;
#else
            fail(
                "HIP/ROCm support was not compiled; enable the experimental HIP build flag");
#endif
        case BackendPreference::sycl:
#if defined(SATVIEW_HAS_EXPERIMENTAL_SYCL)
            if (!satview::experimental::sycl_runtime_available(reason)) {
                throw std::runtime_error(
                    "experimental oneAPI/SYCL backend is unavailable: " + reason);
            }
            return ComputeBackend::sycl;
#else
            fail(
                "oneAPI/SYCL support was not compiled; enable the experimental SYCL build flag");
#endif
    }
    fail("invalid compute backend selection");
}

enum class DisplayMode : std::uint8_t {
    amplitude,
    power,
    power_db,
    phase,
    real,
    imaginary,
    linear,
    magnitude,
};

struct SpeckleSettings {
    satview::cpu::SpeckleFilter filter =
        satview::cpu::SpeckleFilter::none;
    std::uint32_t window_size = 5;
    float equivalent_number_of_looks = 1.0F;

    [[nodiscard]] friend constexpr bool operator==(
        const SpeckleSettings&,
        const SpeckleSettings&) noexcept = default;
};

struct ModeChoice {
    DisplayMode mode;
    const char* label;
};

constexpr std::array<ModeChoice, 6> kGslcModes{{
    {DisplayMode::amplitude, "Amplitude |z|"},
    {DisplayMode::power, "Beta-zero power |z|^2"},
    {DisplayMode::power_db, "Beta-zero power (dB)"},
    {DisplayMode::phase, "Phase (radians)"},
    {DisplayMode::real, "Real component"},
    {DisplayMode::imaginary, "Imaginary component"},
}};

constexpr std::array<ModeChoice, 2> kGcovRealModes{{
    {DisplayMode::linear, "Linear covariance"},
    {DisplayMode::power_db, "Covariance (dB)"},
}};

constexpr std::array<ModeChoice, 2> kGcovComplexModes{{
    {DisplayMode::magnitude, "Cross-term magnitude"},
    {DisplayMode::phase, "Cross-term phase"},
}};

std::span<const ModeChoice> modes_for(const satview::DatasetInfo& layer) {
    if (layer.layer_kind == satview::LayerKind::gslc_polarization) {
        return kGslcModes;
    }
    if (layer.data_type.kind == satview::ScalarKind::compound_complex) {
        return kGcovComplexModes;
    }
    return kGcovRealModes;
}

[[nodiscard]] std::optional<satview::cpu::SpeckleDomain>
speckle_domain_for(
    const satview::DatasetInfo& layer,
    const DisplayMode mode) noexcept {
    if (layer.layer_kind == satview::LayerKind::gslc_polarization) {
        switch (mode) {
            case DisplayMode::amplitude:
                return satview::cpu::SpeckleDomain::amplitude;
            case DisplayMode::power:
                return satview::cpu::SpeckleDomain::linear_power;
            case DisplayMode::power_db:
                return satview::cpu::SpeckleDomain::power_db;
            default:
                return std::nullopt;
        }
    }
    if (layer.layer_kind ==
        satview::LayerKind::gcov_diagonal_covariance) {
        if (mode == DisplayMode::linear) {
            return satview::cpu::SpeckleDomain::linear_power;
        }
        if (mode == DisplayMode::power_db) {
            return satview::cpu::SpeckleDomain::power_db;
        }
    }
    return std::nullopt;
}

[[nodiscard]] SpeckleSettings effective_speckle_settings(
    const satview::DatasetInfo& layer,
    const DisplayMode mode,
    const SpeckleSettings settings) noexcept {
    if (settings.filter == satview::cpu::SpeckleFilter::none ||
        !speckle_domain_for(layer, mode).has_value()) {
        return {};
    }
    SpeckleSettings result = settings;
    if (result.filter == satview::cpu::SpeckleFilter::boxcar) {
        result.equivalent_number_of_looks = 1.0F;
    }
    return result;
}

std::string_view layer_kind_name(const satview::LayerKind kind) {
    using enum satview::LayerKind;
    switch (kind) {
        case gslc_polarization:
            return "GSLC complex polarization";
        case gcov_diagonal_covariance:
            return "GCOV diagonal covariance";
        case gcov_off_diagonal_covariance:
            return "GCOV cross covariance";
        case mask:
            return "validity mask";
        case number_of_looks:
            return "number of looks";
        case rtc_gamma_to_sigma_factor:
            return "gamma-to-sigma factor";
        case calibration_lut:
            return "calibration LUT";
        case auxiliary:
            return "auxiliary";
    }
    return "unknown";
}

bool is_renderable(const satview::DatasetInfo& layer) {
    const bool float32 =
        layer.data_type.kind == satview::ScalarKind::floating_point &&
        layer.data_type.element_size == sizeof(float);
    const bool complex64 =
        layer.data_type.kind == satview::ScalarKind::compound_complex &&
        layer.data_type.element_size == sizeof(satview::cpu::Complex32);
    if (layer.layer_kind == satview::LayerKind::gslc_polarization) {
        return complex64;
    }
    if (layer.layer_kind == satview::LayerKind::gcov_off_diagonal_covariance) {
        return complex64;
    }
    return layer.layer_kind == satview::LayerKind::gcov_diagonal_covariance &&
           float32;
}

std::string semantic_layer_label(const satview::DatasetInfo& layer) {
    std::ostringstream label;
    label << (layer.frequency.empty() ? "-" : layer.frequency) << " / "
          << layer.name << " - " << layer_kind_name(layer.layer_kind);
    return label.str();
}

struct LayerView {
    const satview::DatasetInfo* dataset = nullptr;
    const satview::DatasetInfo* validity_mask = nullptr;
    std::string label;
    std::uint32_t tile_height = 0;
    std::uint32_t tile_width = 0;
    std::uint64_t tile_rows = 0;
    std::uint64_t tile_columns = 0;
    double x_spacing = 1.0;
    double y_spacing = 1.0;
};

struct ProductView {
    std::vector<LayerView> layers;
    std::size_t default_layer = 0;
    // One packed science chunk plus its aligned leaf-mask read.
    std::size_t maximum_input_bytes = 0;
    std::size_t maximum_science_mosaic_bytes = 0;
    std::size_t maximum_science_element_size = 0;
    std::uint32_t maximum_tile_height = 0;
    std::uint32_t maximum_tile_width = 0;
    std::uint32_t maximum_mosaic_height = 0;
    std::uint32_t maximum_mosaic_width = 0;
};

ProductView build_product_view(const satview::Hdf5Product& product) {
    ProductView result;
    for (const auto& dataset : product.datasets()) {
        if (!is_renderable(dataset)) {
            continue;
        }

        const std::uint64_t source_height =
            dataset.chunk_dimensions.has_value()
            ? (*dataset.chunk_dimensions)[0]
            : std::min<std::uint64_t>(
                  dataset.dimensions[0], kContiguousTileExtent);
        const std::uint64_t source_width =
            dataset.chunk_dimensions.has_value()
            ? (*dataset.chunk_dimensions)[1]
            : std::min<std::uint64_t>(
                  dataset.dimensions[1], kContiguousTileExtent);
        if (source_height == 0 || source_width == 0 ||
            source_height > std::numeric_limits<std::uint32_t>::max() ||
            source_width > std::numeric_limits<std::uint32_t>::max()) {
            continue;
        }

        const auto plan = product.make_read_plan(
            dataset.path,
            0,
            0,
            std::min(source_height, dataset.dimensions[0]),
            std::min(source_width, dataset.dimensions[1]));
        if (plan.expected_bytes > kMaximumTileBytes) {
            std::ostringstream message;
            message << "source chunk exceeds the 2 GiB safety limit: "
                    << dataset.path;
            throw std::runtime_error(message.str());
        }

        LayerView view;
        view.dataset = &dataset;
        const auto separator = dataset.path.rfind('/');
        const std::string mask_path =
            separator == std::string::npos
            ? std::string{}
            : dataset.path.substr(0, separator + 1) + "mask";
        const auto* validity_mask =
            mask_path.empty() ? nullptr : product.find_dataset(mask_path);
        if (validity_mask != nullptr &&
            validity_mask->name == "mask" &&
            validity_mask->layer_kind == satview::LayerKind::mask &&
            validity_mask->dimensions == dataset.dimensions &&
            validity_mask->data_type.kind ==
                satview::ScalarKind::unsigned_integer &&
            validity_mask->data_type.element_size == sizeof(std::uint8_t) &&
            validity_mask->data_type.readable) {
            view.validity_mask = validity_mask;
        }
        view.label = semantic_layer_label(dataset);
        view.tile_height = static_cast<std::uint32_t>(source_height);
        view.tile_width = static_cast<std::uint32_t>(source_width);
        view.tile_rows =
            (dataset.dimensions[0] + source_height - 1) / source_height;
        view.tile_columns =
            (dataset.dimensions[1] + source_width - 1) / source_width;
        if (const auto* frequency = product.find_frequency(dataset.frequency)) {
            if (std::isfinite(frequency->grid.x.spacing) &&
                frequency->grid.x.spacing != 0.0) {
                view.x_spacing = std::abs(frequency->grid.x.spacing);
            }
            if (std::isfinite(frequency->grid.y.spacing) &&
                frequency->grid.y.spacing != 0.0) {
                view.y_spacing = std::abs(frequency->grid.y.spacing);
            }
        }

        std::size_t maximum_packed_bytes = plan.expected_bytes;
        if (view.validity_mask != nullptr) {
            const auto maximum_mask_extent = [](
                                                 const std::uint64_t request,
                                                 const std::uint64_t dimension,
                                                 const std::optional<
                                                     std::array<
                                                         std::uint64_t,
                                                         2>>& chunks,
                                                 const std::size_t axis) {
                if (!chunks.has_value()) {
                    return std::min(request, dimension);
                }
                return satview::viewer::maximum_chunk_aligned_extent(
                    request, dimension, (*chunks)[axis]);
            };
            const auto mask_rows = maximum_mask_extent(
                source_height,
                view.validity_mask->dimensions[0],
                view.validity_mask->chunk_dimensions,
                0);
            const auto mask_columns = maximum_mask_extent(
                source_width,
                view.validity_mask->dimensions[1],
                view.validity_mask->chunk_dimensions,
                1);
            if (mask_rows != 0 &&
                mask_columns >
                    std::numeric_limits<std::size_t>::max() / mask_rows) {
                fail("validity-mask staging size overflow");
            }
            const auto maximum_mask_bytes =
                static_cast<std::size_t>(mask_rows * mask_columns);
            if (maximum_mask_bytes >
                kMaximumTileBytes - maximum_packed_bytes) {
                fail("science plus validity-mask staging exceeds 2 GiB");
            }
            maximum_packed_bytes += maximum_mask_bytes;
        }
        result.maximum_input_bytes =
            std::max(result.maximum_input_bytes, maximum_packed_bytes);
        result.maximum_tile_height =
            std::max(result.maximum_tile_height, view.tile_height);
        result.maximum_tile_width =
            std::max(result.maximum_tile_width, view.tile_width);
        const auto maximum_mosaic = satview::viewer::make_mosaic_geometry(
            (view.tile_rows - 1) / 2,
            (view.tile_columns - 1) / 2,
            view.tile_rows,
            view.tile_columns,
            view.tile_height,
            view.tile_width,
            dataset.dimensions[0],
            dataset.dimensions[1],
            4);
        if (maximum_mosaic.pixel_height >
                std::numeric_limits<std::uint32_t>::max() ||
            maximum_mosaic.pixel_width >
                std::numeric_limits<std::uint32_t>::max()) {
            fail("four-chunk mosaic exceeds Vulkan image extent limits");
        }
        result.maximum_mosaic_height = std::max(
            result.maximum_mosaic_height,
            static_cast<std::uint32_t>(maximum_mosaic.pixel_height));
        result.maximum_mosaic_width = std::max(
            result.maximum_mosaic_width,
            static_cast<std::uint32_t>(maximum_mosaic.pixel_width));
        result.maximum_science_element_size = std::max(
            result.maximum_science_element_size,
            dataset.data_type.element_size);
        result.layers.push_back(std::move(view));
    }
    if (result.layers.empty()) {
        fail(
            "the product has no float32 GSLC/GCOV science layer supported by "
            "the current CUDA transforms");
    }
    result.maximum_mosaic_height = std::max(
        result.maximum_mosaic_height,
        satview::overview::kMaximumOverviewLongEdge);
    result.maximum_mosaic_width = std::max(
        result.maximum_mosaic_width,
        satview::overview::kMaximumOverviewLongEdge);
    result.maximum_science_mosaic_bytes =
        satview::viewer::checked_mosaic_bytes(
            result.maximum_mosaic_height,
            result.maximum_mosaic_width,
            result.maximum_science_element_size,
            kMaximumMosaicBytes);
    static_cast<void>(satview::viewer::checked_mosaic_bytes(
        result.maximum_mosaic_height,
        result.maximum_mosaic_width,
        sizeof(float),
        kMaximumMosaicBytes));
    static_cast<void>(satview::viewer::checked_mosaic_bytes(
        result.maximum_mosaic_height,
        result.maximum_mosaic_width,
        sizeof(std::uint8_t),
        kMaximumMosaicBytes));

    const bool gslc =
        product.identification().product_type == satview::ProductType::gslc;
    const std::string_view preferred = gslc ? "HH" : "HHHH";
    for (std::size_t index = 0; index < result.layers.size(); ++index) {
        const auto& layer = *result.layers[index].dataset;
        if (layer.frequency == "A" && layer.name == preferred) {
            result.default_layer = index;
            break;
        }
    }
    return result;
}

enum class TileSourceKind : std::uint8_t {
    native_mosaic,
    raw_overview,
};

struct TileRequest {
    std::uint64_t serial = 0;
    // Navigation generation that selected this render request. It is not part
    // of source identity; it only prevents a late result from regressing view.
    std::uint64_t camera_generation = 0;
    std::size_t layer_index = 0;
    std::uint64_t tile_row = 0;
    std::uint64_t tile_column = 0;
    DisplayMode mode = DisplayMode::power_db;
    SpeckleSettings speckle;
    std::uint32_t mosaic_span = 1;
    satview::viewer::MosaicGeometry mosaic;
    TileSourceKind source_kind = TileSourceKind::native_mosaic;
    // Caller-owned stable identity for the raw samples in an overview. Zero is
    // reserved for native mosaics so an uninitialized overview cannot become
    // a false resident-cache hit.
    std::uint64_t overview_identity = 0;
};

[[nodiscard]] bool same_tile_source(
    const TileRequest& left,
    const TileRequest& right) noexcept {
    if (left.layer_index != right.layer_index ||
        left.source_kind != right.source_kind) {
        return false;
    }
    if (left.source_kind == TileSourceKind::raw_overview) {
        return left.overview_identity != 0 &&
            left.overview_identity == right.overview_identity;
    }
    return left.mosaic == right.mosaic;
}

struct ReadCompletion {
    TileRequest request;
    std::uint32_t chunk_index = 0;
    std::uint32_t chunk_count = 0;
    std::uint64_t destination_row = 0;
    std::uint64_t destination_column = 0;
    std::optional<satview::ReadPlan> plan;
    std::optional<satview::ReadPlan> mask_plan;
    double hdf5_milliseconds = 0.0;
    std::string error;
};

[[nodiscard]] ReadRing make_read_ring(
    const std::size_t bytes_per_slot, const bool page_locked) {
#if defined(SATVIEW_HAS_CUDA)
    return ReadRing(ReadRing::Options{
        .slot_count = kPinnedSlotCount,
        .bytes_per_slot = bytes_per_slot,
        .page_locked = page_locked,
    });
#else
    static_cast<void>(page_locked);
    return ReadRing(kPinnedSlotCount, bytes_per_slot);
#endif
}

class TileReader final {
public:
    TileReader(
        const satview::Hdf5Product& product,
        const ProductView& view,
        const bool page_locked)
        : product_(product),
          view_(view),
          ring_(make_read_ring(view.maximum_input_bytes, page_locked)),
          thread_([this](std::stop_token stop) { reader_loop(stop); }) {}

    ~TileReader() {
        stop();
    }

    TileReader(const TileReader&) = delete;
    TileReader& operator=(const TileReader&) = delete;

    void request(TileRequest request) {
        latest_serial_.store(request.serial, std::memory_order_release);
        {
            std::lock_guard lock(mutex_);
            requested_ = std::move(request);
        }
        request_cv_.notify_one();
    }

    void supersede(const std::uint64_t serial) noexcept {
        latest_serial_.store(serial, std::memory_order_release);
    }

    std::optional<ReadCompletion> try_take_completion() {
        std::lock_guard lock(mutex_);
        if (completed_.empty()) {
            return std::nullopt;
        }
        ReadCompletion result = std::move(completed_.front());
        completed_.pop_front();
        return result;
    }

    ReadSlotLease try_take_ready_slot() {
        return ring_.try_acquire_ready();
    }

    std::size_t reclaim_completed() {
        return ring_.reclaim_completed();
    }

    ReadRingState ring_state() const {
        return ring_.state_counts();
    }

    void stop() {
        if (thread_.joinable()) {
            thread_.request_stop();
            request_cv_.notify_all();
            thread_.join();
        }
    }

private:
    [[nodiscard]] bool superseded(
        const std::uint64_t serial) const noexcept {
        return latest_serial_.load(std::memory_order_acquire) != serial;
    }

    void push_completion(ReadCompletion completion) {
        std::lock_guard lock(mutex_);
        completed_.push_back(std::move(completion));
    }

    void reader_loop(const std::stop_token stop) {
        while (!stop.stop_requested()) {
            std::optional<TileRequest> request;
            {
                std::unique_lock lock(mutex_);
                request_cv_.wait(
                    lock,
                    stop,
                    [this] { return requested_.has_value(); });
                if (stop.stop_requested()) {
                    break;
                }
                request = std::exchange(requested_, std::nullopt);
            }

            const auto& layer = view_.layers.at(request->layer_index);
            const auto& dataset = *layer.dataset;
            const auto canonical = satview::viewer::make_mosaic_geometry(
                request->tile_row,
                request->tile_column,
                layer.tile_rows,
                layer.tile_columns,
                layer.tile_height,
                layer.tile_width,
                dataset.dimensions[0],
                dataset.dimensions[1],
                request->mosaic_span);
            if (canonical != request->mosaic) {
                ReadCompletion completion;
                completion.request = *request;
                completion.error = "mosaic request geometry is not canonical";
                push_completion(std::move(completion));
                continue;
            }
            const auto chunk_count = static_cast<std::uint32_t>(
                canonical.chunk_rows * canonical.chunk_columns);
            bool abandon = false;
            for (std::uint32_t local_row = 0;
                 local_row < canonical.chunk_rows && !abandon;
                 ++local_row) {
                for (std::uint32_t local_column = 0;
                     local_column < canonical.chunk_columns;
                     ++local_column) {
                    if (stop.stop_requested() ||
                        superseded(request->serial)) {
                        abandon = true;
                        break;
                    }

                    ReadCompletion completion;
                    completion.request = *request;
                    completion.chunk_index =
                        local_row * canonical.chunk_columns + local_column;
                    completion.chunk_count = chunk_count;
                    try {
                        const auto row =
                            canonical.pixel_row +
                            static_cast<std::uint64_t>(local_row) *
                                layer.tile_height;
                        const auto column =
                            canonical.pixel_column +
                            static_cast<std::uint64_t>(local_column) *
                                layer.tile_width;
                        const auto height = std::min<std::uint64_t>(
                            layer.tile_height,
                            dataset.dimensions[0] - row);
                        const auto width = std::min<std::uint64_t>(
                            layer.tile_width,
                            dataset.dimensions[1] - column);
                        completion.plan = product_.make_read_plan(
                            dataset.path, row, column, height, width);
                        const auto& aligned = completion.plan->aligned;
                        if (aligned.row < canonical.pixel_row ||
                            aligned.column < canonical.pixel_column) {
                            fail("source chunk alignment precedes mosaic bounds");
                        }
                        const auto destination_row =
                            aligned.row - canonical.pixel_row;
                        const auto destination_column =
                            aligned.column - canonical.pixel_column;
                        if (destination_row > canonical.pixel_height ||
                            destination_column > canonical.pixel_width ||
                            aligned.height >
                                canonical.pixel_height - destination_row ||
                            aligned.width >
                                canonical.pixel_width - destination_column) {
                            fail("source chunk alignment escapes mosaic bounds");
                        }
                        completion.destination_row = destination_row;
                        completion.destination_column = destination_column;
                        if (layer.validity_mask != nullptr) {
                            completion.mask_plan = product_.make_read_plan(
                                layer.validity_mask->path,
                                aligned.row,
                                aligned.column,
                                aligned.height,
                                aligned.width);
                        }

                        auto filling = ring_.acquire_for_fill(stop);
                        if (!filling) {
                            abandon = true;
                            break;
                        }
                        if (superseded(request->serial)) {
                            filling.reset();
                            abandon = true;
                            break;
                        }
                        const auto started = Clock::now();
                        product_.read_into(
                            *completion.plan,
                            filling.bytes().first(
                                completion.plan->expected_bytes));
                        std::size_t packed_bytes =
                            completion.plan->expected_bytes;
                        if (completion.mask_plan.has_value()) {
                            if (packed_bytes > filling.capacity() ||
                                completion.mask_plan->expected_bytes >
                                    filling.capacity() - packed_bytes) {
                                fail(
                                    "validity-mask read exceeds pinned slot "
                                    "capacity");
                            }
                            product_.read_into(
                                *completion.mask_plan,
                                filling.bytes().subspan(
                                    packed_bytes,
                                    completion.mask_plan->expected_bytes));
                            packed_bytes +=
                                completion.mask_plan->expected_bytes;
                        }
                        completion.hdf5_milliseconds =
                            std::chrono::duration<double, std::milli>(
                                Clock::now() - started)
                                .count();
                        filling.publish_ready(packed_bytes);
                    } catch (const std::exception& error) {
                        completion.plan.reset();
                        completion.mask_plan.reset();
                        completion.error = error.what();
                        abandon = true;
                    }
                    push_completion(std::move(completion));
                    if (abandon) {
                        break;
                    }
                }
            }
        }
    }

    const satview::Hdf5Product& product_;
    const ProductView& view_;
    ReadRing ring_;
    mutable std::mutex mutex_;
    std::condition_variable_any request_cv_;
    std::atomic<std::uint64_t> latest_serial_{0};
    std::optional<TileRequest> requested_;
    std::deque<ReadCompletion> completed_;
    std::jthread thread_;
};
std::uint32_t find_memory_type(
    const VkPhysicalDevice physical_device,
    const std::uint32_t bits,
    const VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);
    for (std::uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
        if ((bits & (1U << index)) != 0 &&
            (properties.memoryTypes[index].propertyFlags & required) ==
                required) {
            return index;
        }
    }
    fail("no compatible Vulkan device-local memory type");
}

std::filesystem::path executable_directory() {
    std::wstring path(32'768, L'\0');
    const DWORD length = GetModuleFileNameW(
        nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0 || length == path.size()) {
        throw std::runtime_error(
            "could not resolve the sat-viewer executable directory");
    }
    path.resize(length);
    return std::filesystem::path(path).parent_path();
}

std::vector<std::uint32_t> read_spirv(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("could not open shader: " + path.string());
    }
    const auto bytes = stream.tellg();
    if (bytes <= 0 || (static_cast<std::uint64_t>(bytes) % 4) != 0) {
        throw std::runtime_error("invalid SPIR-V byte count: " + path.string());
    }
    std::vector<std::uint32_t> result(
        static_cast<std::size_t>(bytes) / sizeof(std::uint32_t));
    stream.seekg(0);
    stream.read(
        reinterpret_cast<char*>(result.data()),
        static_cast<std::streamsize>(bytes));
    if (!stream) {
        throw std::runtime_error("could not read shader: " + path.string());
    }
    return result;
}

struct DisplayRect {
    float x = 0.0F;
    float y = 0.0F;
    float width = 1.0F;
    float height = 1.0F;
};

// Native pixel-edge coverage carried by the currently published texture.
// `actual_scene` excludes padding beyond the raster. The texture's virtual
// coverage starts at texture_origin and uses independent row/column strides;
// keeping these distinct prevents a partial bottom/right overview from being
// stretched to fill its last power-of-two sample cell.
struct ResidentViewMapping {
    std::size_t layer_index = std::numeric_limits<std::size_t>::max();
    satview::viewer::PixelWindow actual_scene;
    double texture_origin_row = 0.0;
    double texture_origin_column = 0.0;
    std::uint64_t sample_stride_row = 1;
    std::uint64_t sample_stride_column = 1;
    std::uint32_t texture_width = 0;
    std::uint32_t texture_height = 0;
};

struct ViewportNavigationState {
    satview::viewer::Camera2D camera;
    std::size_t layer_index = std::numeric_limits<std::size_t>::max();
    double canvas_width = 0.0;
    double canvas_height = 0.0;
    bool canvas_extent_initialized = false;
    bool initialized = false;
};

struct ViewportUiResult {
    satview::viewer::ScreenViewport canvas;
    satview::viewer::RasterWindow visible;
    satview::viewer::RasterWindow scientific_window;
    DisplayRect scientific_rect{0.0F, 0.0F, 0.0F, 0.0F};
    bool resident_visible = false;
    bool camera_changed = false;
    // Initial, layer, chunk, or footprint framing explicitly requests native
    // data even when the physical-aspect viewport shows blank margins.
    bool native_footprint_requested = false;
    // Discrete navigation commands should bypass a wheel/drag debounce.
    bool immediate_request = false;
};

struct ResidentSamplePushConstants {
    std::array<std::uint32_t, 2> valid_extent{0, 0};
    std::array<float, 2> sample_uv_origin{0.0F, 0.0F};
    std::array<float, 2> sample_uv_extent{0.0F, 0.0F};
};

static_assert(sizeof(ResidentSamplePushConstants) == 24);
static_assert(
    offsetof(ResidentSamplePushConstants, sample_uv_origin) == 8);
static_assert(
    offsetof(ResidentSamplePushConstants, sample_uv_extent) == 16);

struct DisplayPushConstants {
    float low = -35.0F;
    float high = 5.0F;
    float gamma = 1.0F;
    std::uint32_t colormap = 1;
    // Maps the axis-aligned Vulkan draw rectangle into the unrotated raster
    // window. The fragment shader clips the rotated quadrilateral before
    // sampling either resident slot.
    std::array<float, 2> window_uv_origin{0.0F, 0.0F};
    std::array<float, 2> window_uv_dx{1.0F, 0.0F};
    std::array<float, 2> window_uv_dy{0.0F, 1.0F};
    // Physical Vulkan image slots remain immutable across publications.
    std::array<ResidentSamplePushConstants, 2> samples{};
    std::array<float, 2> sample_weights{0.0F, 0.0F};
    std::uint32_t active_slots = 0;
    // Zero is the default validity-aware smooth path; one preserves exact
    // nearest-texel inspection. circular_phase selects angular interpolation.
    std::uint32_t sampling_mode = 0;
    std::uint32_t circular_phase = 0;
    std::uint32_t padding = 0;
};

static_assert(sizeof(DisplayPushConstants) == 112);
static_assert(offsetof(DisplayPushConstants, colormap) == 12);
static_assert(offsetof(DisplayPushConstants, window_uv_origin) == 16);
static_assert(offsetof(DisplayPushConstants, window_uv_dx) == 24);
static_assert(offsetof(DisplayPushConstants, window_uv_dy) == 32);
static_assert(offsetof(DisplayPushConstants, samples) == 40);
static_assert(offsetof(DisplayPushConstants, sample_weights) == 88);
static_assert(offsetof(DisplayPushConstants, active_slots) == 96);
static_assert(offsetof(DisplayPushConstants, sampling_mode) == 100);
static_assert(offsetof(DisplayPushConstants, circular_phase) == 104);

struct TileUpload {
    TileRequest request;
    std::uint64_t cuda_ready_value = 0;
    std::uint64_t vulkan_consumed_value = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::span<const float> host_values;
};

struct DistributionCompletion {
    TileRequest request;
    satview::gpu::DistributionSummary summary;
    float elapsed_milliseconds = 0.0F;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

struct FrameResult {
    bool presented = false;
    std::uint32_t uploaded_slot = std::numeric_limits<std::uint32_t>::max();
    bool upload_submitted = false;
};

class VulkanRenderer final {
public:
    VulkanRenderer(
        SDL_Window* window,
        const std::uint32_t maximum_width,
        const std::uint32_t maximum_height,
        const bool cuda_interop)
        : window_(window),
          maximum_width_(maximum_width),
          maximum_height_(maximum_height),
          cuda_interop_(cuda_interop) {
        try {
            create_instance();
            create_surface();
            select_physical_device();
            create_device();
            create_descriptor_pool();
            create_swapchain();
            create_scientific_resources();
        } catch (...) {
            destroy_all_resources();
            throw;
        }
    }

    ~VulkanRenderer() noexcept {
        destroy_all_resources();
    }

    VulkanRenderer(const VulkanRenderer&) = delete;
    VulkanRenderer& operator=(const VulkanRenderer&) = delete;

    void initialize_imgui() {
        if (!ImGui_ImplSDL3_InitForVulkan(window_)) {
            fail("ImGui SDL3 Vulkan initialization failed");
        }
        ImGui_ImplVulkan_InitInfo information{};
        information.ApiVersion = VK_API_VERSION_1_3;
        information.Instance = instance_;
        information.PhysicalDevice = physical_device_;
        information.Device = device_;
        information.QueueFamily = queue_family_;
        information.Queue = queue_;
        information.DescriptorPool = descriptor_pool_;
        information.MinImageCount = kMinimumSwapchainImages;
        information.ImageCount = window_data_.ImageCount;
        information.PipelineInfoMain.RenderPass = window_data_.RenderPass;
        information.PipelineInfoMain.Subpass = 0;
        information.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        information.CheckVkResultFn = [](VkResult result) {
            if (result < 0) {
                std::cerr << "ImGui Vulkan backend error: " << result << '\n';
            }
        };
        if (!ImGui_ImplVulkan_Init(&information)) {
            ImGui_ImplSDL3_Shutdown();
            fail("ImGui Vulkan renderer initialization failed");
        }
        imgui_initialized_ = true;
    }

    void shutdown_imgui() noexcept {
        if (imgui_initialized_) {
            static_cast<void>(vkDeviceWaitIdle(device_));
            ImGui_ImplVulkan_Shutdown();
            ImGui_ImplSDL3_Shutdown();
            imgui_initialized_ = false;
        }
    }

#if defined(SATVIEW_HAS_CUDA)
    [[nodiscard]] void* cuda_output() const noexcept {
        return exported_buffer_.has_value()
            ? exported_buffer_->cuda_ptr()
            : nullptr;
    }

    [[nodiscard]] satview::gpu::InteropTimeline& timeline() noexcept {
        if (!timeline_.has_value()) {
            fail("CUDA/Vulkan timeline requested by a host-staged backend");
        }
        return *timeline_;
    }
#endif

    [[nodiscard]] std::uint32_t maximum_width() const noexcept {
        return maximum_width_;
    }

    [[nodiscard]] std::uint32_t maximum_height() const noexcept {
        return maximum_height_;
    }

    [[nodiscard]] std::string device_name() const {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical_device_, &properties);
        return properties.deviceName;
    }

    void resize_if_needed() {
        int width = 0;
        int height = 0;
        SDL_GetWindowSizeInPixels(window_, &width, &height);
        if (width <= 0 || height <= 0) {
            return;
        }
        if (!swapchain_rebuild_ && width == window_data_.Width &&
            height == window_data_.Height) {
            return;
        }
        check_vk(vkDeviceWaitIdle(device_), "wait before swapchain resize");
        ImGui_ImplVulkan_SetMinImageCount(kMinimumSwapchainImages);
        ImGui_ImplVulkanH_CreateOrResizeWindow(
            instance_,
            physical_device_,
            device_,
            &window_data_,
            queue_family_,
            nullptr,
            width,
            height,
            kMinimumSwapchainImages,
            0);
        window_data_.FrameIndex = 0;
        swapchain_rebuild_ = false;
    }

    FrameResult render(
        ImDrawData* draw_data,
        const DisplayRect& display_rect,
        const DisplayPushConstants& push,
        const bool image_valid,
        const std::optional<TileUpload>& upload,
        const std::optional<std::uint32_t> upload_slot) {
        FrameResult result;
        if (upload.has_value() != upload_slot.has_value()) {
            fail("Vulkan upload and destination slot must be paired");
        }

        if (draw_data->DisplaySize.x <= 0.0F ||
            draw_data->DisplaySize.y <= 0.0F) {
            return result;
        }

        const bool interop_upload = upload.has_value() &&
            upload->cuda_ready_value != 0;
        if (upload.has_value()) {
            if (interop_upload != cuda_interop_) {
                fail("compute upload does not match the Vulkan backend mode");
            }
            if (!interop_upload) {
                prepare_host_upload(*upload);
            }
        }

        const auto acquired =
            window_data_.FrameSemaphores[window_data_.SemaphoreIndex]
                .ImageAcquiredSemaphore;
        const auto render_complete =
            window_data_.FrameSemaphores[window_data_.SemaphoreIndex]
                .RenderCompleteSemaphore;
        VkResult acquire_result = vkAcquireNextImageKHR(
            device_,
            window_data_.Swapchain,
            UINT64_MAX,
            acquired,
            VK_NULL_HANDLE,
            &window_data_.FrameIndex);
        if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR ||
            acquire_result == VK_SUBOPTIMAL_KHR) {
            swapchain_rebuild_ = true;
        }
        if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
            return result;
        }
        if (acquire_result != VK_SUBOPTIMAL_KHR) {
            check_vk(acquire_result, "acquire swapchain image");
        }

        auto& frame = window_data_.Frames[window_data_.FrameIndex];
        check_vk(
            vkWaitForFences(device_, 1, &frame.Fence, VK_TRUE, UINT64_MAX),
            "wait for frame fence");
        check_vk(vkResetFences(device_, 1, &frame.Fence), "reset frame fence");
        check_vk(
            vkResetCommandPool(device_, frame.CommandPool, 0),
            "reset frame command pool");

        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_vk(
            vkBeginCommandBuffer(frame.CommandBuffer, &begin),
            "begin frame command buffer");

        if (upload.has_value()) {
            record_upload(frame.CommandBuffer, *upload, *upload_slot);
        }
        for (auto& image : scientific_images_) {
            if (image.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
                record_initialize_image(frame.CommandBuffer, image);
            }
        }

        VkRenderPassBeginInfo render_pass{};
        render_pass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass.renderPass = window_data_.RenderPass;
        render_pass.framebuffer = frame.Framebuffer;
        render_pass.renderArea.extent = {
            static_cast<std::uint32_t>(window_data_.Width),
            static_cast<std::uint32_t>(window_data_.Height)};
        render_pass.clearValueCount = 1;
        render_pass.pClearValues = &window_data_.ClearValue;
        vkCmdBeginRenderPass(
            frame.CommandBuffer,
            &render_pass,
            VK_SUBPASS_CONTENTS_INLINE);

        if (image_valid || upload.has_value()) {
            record_scientific_draw(
                frame.CommandBuffer, draw_data, display_rect, push);
        }
        ImGui_ImplVulkan_RenderDrawData(draw_data, frame.CommandBuffer);
        vkCmdEndRenderPass(frame.CommandBuffer);
        check_vk(
            vkEndCommandBuffer(frame.CommandBuffer),
            "end frame command buffer");

        std::array<VkSemaphore, 2> waits{acquired, VK_NULL_HANDLE};
        std::array<VkPipelineStageFlags, 2> wait_stages{
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT};
        std::array<std::uint64_t, 2> wait_values{0, 0};
        std::array<VkSemaphore, 2> signals{render_complete, VK_NULL_HANDLE};
        std::array<std::uint64_t, 2> signal_values{0, 0};
        std::uint32_t wait_count = 1;
        std::uint32_t signal_count = 1;
        if (interop_upload) {
#if defined(SATVIEW_HAS_CUDA)
            waits[1] = timeline_->vk_semaphore();
            wait_values[1] = upload->cuda_ready_value;
            signals[1] = timeline_->vk_semaphore();
            signal_values[1] = upload->vulkan_consumed_value;
            wait_count = 2;
            signal_count = 2;
#else
            fail("CUDA interop upload reached a CUDA-free viewer build");
#endif
        }

        VkTimelineSemaphoreSubmitInfo timeline_submit{};
        timeline_submit.sType =
            VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_submit.waitSemaphoreValueCount = wait_count;
        timeline_submit.pWaitSemaphoreValues = wait_values.data();
        timeline_submit.signalSemaphoreValueCount = signal_count;
        timeline_submit.pSignalSemaphoreValues = signal_values.data();

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.pNext = interop_upload ? &timeline_submit : nullptr;
        submit.waitSemaphoreCount = wait_count;
        submit.pWaitSemaphores = waits.data();
        submit.pWaitDstStageMask = wait_stages.data();
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &frame.CommandBuffer;
        submit.signalSemaphoreCount = signal_count;
        submit.pSignalSemaphores = signals.data();
        check_vk(vkQueueSubmit(queue_, 1, &submit, frame.Fence), "submit frame");
        result.upload_submitted = upload.has_value();
        if (upload_slot.has_value()) {
            result.uploaded_slot = *upload_slot;
        }

        VkPresentInfoKHR present{};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &render_complete;
        present.swapchainCount = 1;
        present.pSwapchains = &window_data_.Swapchain;
        present.pImageIndices = &window_data_.FrameIndex;
        const VkResult present_result = vkQueuePresentKHR(queue_, &present);
        if (present_result == VK_ERROR_OUT_OF_DATE_KHR ||
            present_result == VK_SUBOPTIMAL_KHR) {
            swapchain_rebuild_ = true;
        } else {
            check_vk(present_result, "present swapchain image");
            result.presented = true;
        }
        window_data_.SemaphoreIndex =
            (window_data_.SemaphoreIndex + 1) %
            window_data_.SemaphoreCount;
        return result;
    }

private:
    void create_instance() {
        std::uint32_t extension_count = 0;
        const char* const* sdl_extensions =
            SDL_Vulkan_GetInstanceExtensions(&extension_count);
        if (sdl_extensions == nullptr || extension_count == 0) {
            fail("SDL did not provide Vulkan instance extensions");
        }
        std::vector<const char*> extensions(
            sdl_extensions, sdl_extensions + extension_count);

        VkApplicationInfo application{};
        application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        application.pApplicationName = "NISAR Data Reader";
        application.applicationVersion = VK_MAKE_VERSION(0, 6, 0);
        application.pEngineName = "satview";
        application.engineVersion = VK_MAKE_VERSION(0, 6, 0);
        application.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create.pApplicationInfo = &application;
        create.enabledExtensionCount =
            static_cast<std::uint32_t>(extensions.size());
        create.ppEnabledExtensionNames = extensions.data();
        check_vk(
            vkCreateInstance(&create, nullptr, &instance_),
            "create Vulkan instance");
    }

    void create_surface() {
        if (!SDL_Vulkan_CreateSurface(
                window_, instance_, nullptr, &surface_)) {
            throw std::runtime_error(
                std::string("create Vulkan surface: ") + SDL_GetError());
        }
    }

    void select_physical_device() {
        std::uint32_t device_count = 0;
        check_vk(
            vkEnumeratePhysicalDevices(instance_, &device_count, nullptr),
            "enumerate Vulkan physical devices");
        if (device_count == 0) {
            fail("no Vulkan physical device is available");
        }
        std::vector<VkPhysicalDevice> devices(device_count);
        check_vk(
            vkEnumeratePhysicalDevices(
                instance_, &device_count, devices.data()),
            "enumerate Vulkan physical devices");

#if defined(SATVIEW_HAS_CUDA)
        std::array<char, VK_LUID_SIZE> cuda_luid{};
        unsigned int cuda_node_mask = 0;
        if (cuda_interop_) {
            int cuda_device = 0;
            check_cuda(cudaGetDevice(&cuda_device), "query current CUDA device");
            cudaDeviceProp cuda_properties{};
            check_cuda(
                cudaGetDeviceProperties(&cuda_properties, cuda_device),
                "query CUDA device properties");
            static_assert(sizeof(cuda_properties.luid) == VK_LUID_SIZE);
            std::memcpy(
                cuda_luid.data(), cuda_properties.luid, VK_LUID_SIZE);
            cuda_node_mask = cuda_properties.luidDeviceNodeMask;
        }
#else
        if (cuda_interop_) {
            fail("CUDA interop requested by a CUDA-free viewer build");
        }
#endif

        int best_score = std::numeric_limits<int>::min();
        for (const auto candidate : devices) {
#if defined(SATVIEW_HAS_CUDA)
            if (cuda_interop_) {
                VkPhysicalDeviceIDProperties identity{};
                identity.sType =
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
                VkPhysicalDeviceProperties2 properties{};
                properties.sType =
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                properties.pNext = &identity;
                vkGetPhysicalDeviceProperties2(candidate, &properties);
                if (!identity.deviceLUIDValid ||
                    std::memcmp(
                        identity.deviceLUID,
                        cuda_luid.data(),
                        VK_LUID_SIZE) != 0 ||
                    identity.deviceNodeMask != cuda_node_mask) {
                    continue;
                }
            }
#endif

            std::uint32_t family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(
                candidate, &family_count, nullptr);
            std::vector<VkQueueFamilyProperties> families(family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(
                candidate, &family_count, families.data());
            for (std::uint32_t family = 0; family < family_count; ++family) {
                VkBool32 presentation = VK_FALSE;
                check_vk(
                    vkGetPhysicalDeviceSurfaceSupportKHR(
                        candidate, family, surface_, &presentation),
                    "query Vulkan presentation support");
                if ((families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) !=
                        0 &&
                    presentation == VK_TRUE) {
                    if (cuda_interop_) {
                        physical_device_ = candidate;
                        queue_family_ = family;
                        return;
                    }
                    VkPhysicalDeviceProperties properties{};
                    vkGetPhysicalDeviceProperties(candidate, &properties);
                    int score = 0;
                    if (properties.deviceType ==
                        VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                        score = 200;
                    } else if (properties.deviceType ==
                               VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
                        score = 100;
                    }
                    if (score > best_score) {
                        best_score = score;
                        physical_device_ = candidate;
                        queue_family_ = family;
                    }
                    break;
                }
            }
        }
        if (physical_device_ == VK_NULL_HANDLE) {
            fail(cuda_interop_
                ? "no Vulkan graphics/present device matches the current CUDA device LUID"
                : "no Vulkan graphics/present device is available");
        }
    }

    void create_device() {
        std::vector<const char*> extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        if (cuda_interop_) {
            extensions.insert(
                extensions.end(),
                {VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
                 VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
                 VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
                 VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME});
        }
        const float priority = 1.0F;
        VkDeviceQueueCreateInfo queue{};
        queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue.queueFamilyIndex = queue_family_;
        queue.queueCount = 1;
        queue.pQueuePriorities = &priority;

        VkPhysicalDeviceTimelineSemaphoreFeatures timeline_feature{};
        timeline_feature.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
        timeline_feature.timelineSemaphore = VK_TRUE;

        VkDeviceCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create.pNext = cuda_interop_ ? &timeline_feature : nullptr;
        create.queueCreateInfoCount = 1;
        create.pQueueCreateInfos = &queue;
        create.enabledExtensionCount =
            static_cast<std::uint32_t>(extensions.size());
        create.ppEnabledExtensionNames = extensions.data();
        check_vk(
            vkCreateDevice(physical_device_, &create, nullptr, &device_),
            "create Vulkan device");
        vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
    }

    void create_descriptor_pool() {
        const std::array sizes{
            VkDescriptorPoolSize{
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64},
        };
        VkDescriptorPoolCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        create.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        create.maxSets = 64;
        create.poolSizeCount =
            static_cast<std::uint32_t>(sizes.size());
        create.pPoolSizes = sizes.data();
        check_vk(
            vkCreateDescriptorPool(
                device_, &create, nullptr, &descriptor_pool_),
            "create Vulkan descriptor pool");
    }

    void create_swapchain() {
        VkBool32 presentation = VK_FALSE;
        check_vk(
            vkGetPhysicalDeviceSurfaceSupportKHR(
                physical_device_, queue_family_, surface_, &presentation),
            "query surface support");
        if (presentation != VK_TRUE) {
            fail("selected Vulkan queue cannot present to the SDL window");
        }

        constexpr std::array formats{
            VK_FORMAT_B8G8R8A8_UNORM,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_FORMAT_B8G8R8_UNORM,
            VK_FORMAT_R8G8B8_UNORM,
        };
        window_data_.Surface = surface_;
        window_data_.SurfaceFormat =
            ImGui_ImplVulkanH_SelectSurfaceFormat(
                physical_device_,
                surface_,
                formats.data(),
                static_cast<int>(formats.size()),
                VK_COLORSPACE_SRGB_NONLINEAR_KHR);
        constexpr std::array<VkPresentModeKHR, 2> present_modes{{
            VK_PRESENT_MODE_MAILBOX_KHR,
            VK_PRESENT_MODE_FIFO_KHR,
        }};
        window_data_.PresentMode =
            ImGui_ImplVulkanH_SelectPresentMode(
                physical_device_,
                surface_,
                present_modes.data(),
                static_cast<int>(present_modes.size()));
        window_data_.ClearValue.color.float32[0] = 0.012F;
        window_data_.ClearValue.color.float32[1] = 0.016F;
        window_data_.ClearValue.color.float32[2] = 0.024F;
        window_data_.ClearValue.color.float32[3] = 1.0F;

        int width = 0;
        int height = 0;
        SDL_GetWindowSizeInPixels(window_, &width, &height);
        ImGui_ImplVulkanH_CreateOrResizeWindow(
            instance_,
            physical_device_,
            device_,
            &window_data_,
            queue_family_,
            nullptr,
            width,
            height,
            kMinimumSwapchainImages,
            0);
    }

    struct ScientificImage {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    void create_scientific_resources() {
        const VkDeviceSize output_bytes =
            static_cast<VkDeviceSize>(maximum_width_) *
            static_cast<VkDeviceSize>(maximum_height_) * sizeof(float);
        if (cuda_interop_) {
#if defined(SATVIEW_HAS_CUDA)
            exported_buffer_.emplace(
                physical_device_,
                device_,
                output_bytes,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            timeline_.emplace(physical_device_, device_, 0);
#else
            fail("CUDA interop resources requested by a CUDA-free build");
#endif
        } else {
            VkBufferCreateInfo buffer_create{};
            buffer_create.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buffer_create.size = output_bytes;
            buffer_create.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            buffer_create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            check_vk(
                vkCreateBuffer(
                    device_, &buffer_create, nullptr, &host_staging_buffer_),
                "create host-staged scientific buffer");
            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(
                device_, host_staging_buffer_, &requirements);
            VkMemoryAllocateInfo allocation{};
            allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = find_memory_type(
                physical_device_,
                requirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            check_vk(
                vkAllocateMemory(
                    device_, &allocation, nullptr, &host_staging_memory_),
                "allocate host-staged scientific buffer");
            check_vk(
                vkBindBufferMemory(
                    device_, host_staging_buffer_, host_staging_memory_, 0),
                "bind host-staged scientific buffer");
            check_vk(
                vkMapMemory(
                    device_, host_staging_memory_, 0, output_bytes, 0,
                    &host_staging_mapping_),
                "map host-staged scientific buffer");
            host_staging_capacity_ = output_bytes;
        }

        VkImageCreateInfo image_create{};
        image_create.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_create.imageType = VK_IMAGE_TYPE_2D;
        image_create.format = VK_FORMAT_R32_SFLOAT;
        image_create.extent = {maximum_width_, maximum_height_, 1};
        image_create.mipLevels = 1;
        image_create.arrayLayers = 1;
        image_create.samples = VK_SAMPLE_COUNT_1_BIT;
        image_create.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_create.usage =
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        image_create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        image_create.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        for (auto& image : scientific_images_) {
        check_vk(
                vkCreateImage(device_, &image_create, nullptr, &image.image),
            "create R32F scientific image");

        VkMemoryRequirements requirements{};
            vkGetImageMemoryRequirements(
                device_, image.image, &requirements);
        VkMemoryAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = find_memory_type(
            physical_device_,
            requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check_vk(
            vkAllocateMemory(
                    device_, &allocation, nullptr, &image.memory),
            "allocate scientific image");
        check_vk(
                vkBindImageMemory(device_, image.image, image.memory, 0),
            "bind scientific image");

        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view.image = image.image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = VK_FORMAT_R32_SFLOAT;
        view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.levelCount = 1;
        view.subresourceRange.layerCount = 1;
        check_vk(
                vkCreateImageView(device_, &view, nullptr, &image.view),
            "create scientific image view");
        }

        VkSamplerCreateInfo sampler{};
        sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler.magFilter = VK_FILTER_NEAREST;
        sampler.minFilter = VK_FILTER_NEAREST;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.maxLod = 0.0F;
        check_vk(
            vkCreateSampler(device_, &sampler, nullptr, &sampler_),
            "create scientific sampler");

        colormap_lut_ =
            std::make_unique<satview::viewer::VulkanColormapLut>(
                physical_device_, device_, queue_family_, queue_);

        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
        bindings[0].binding = 0;
        bindings[0].descriptorType =
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[1] = bindings[0];
        bindings[1].binding = 1;
        bindings[2] = bindings[0];
        bindings[2].binding = 2;
        VkDescriptorSetLayoutCreateInfo set_layout{};
        set_layout.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        set_layout.bindingCount =
            static_cast<std::uint32_t>(bindings.size());
        set_layout.pBindings = bindings.data();
        check_vk(
            vkCreateDescriptorSetLayout(
                device_, &set_layout, nullptr, &descriptor_set_layout_),
            "create scientific descriptor layout");

        VkDescriptorSetAllocateInfo set_allocate{};
        set_allocate.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        set_allocate.descriptorPool = descriptor_pool_;
        set_allocate.descriptorSetCount = 1;
        set_allocate.pSetLayouts = &descriptor_set_layout_;
        check_vk(
            vkAllocateDescriptorSets(
                device_, &set_allocate, &descriptor_set_),
            "allocate scientific descriptor");
        std::array<VkDescriptorImageInfo, 3> image_infos{};
        image_infos[0].sampler = sampler_;
        image_infos[0].imageView = scientific_images_[0].view;
        image_infos[0].imageLayout =
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        image_infos[1].sampler = sampler_;
        image_infos[1].imageView = scientific_images_[1].view;
        image_infos[1].imageLayout =
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        image_infos[2].sampler = colormap_lut_->sampler();
        image_infos[2].imageView = colormap_lut_->image_view();
        image_infos[2].imageLayout =
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        std::array<VkWriteDescriptorSet, 3> writes{};
        for (std::size_t index = 0; index < writes.size(); ++index) {
            writes[index].sType =
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[index].dstSet = descriptor_set_;
            writes[index].dstBinding =
                static_cast<std::uint32_t>(index);
            writes[index].descriptorCount = 1;
            writes[index].descriptorType =
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[index].pImageInfo = &image_infos[index];
        }
        vkUpdateDescriptorSets(
            device_,
            static_cast<std::uint32_t>(writes.size()),
            writes.data(),
            0,
            nullptr);

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push.offset = 0;
        push.size = sizeof(DisplayPushConstants);
        VkPipelineLayoutCreateInfo pipeline_layout{};
        pipeline_layout.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout.setLayoutCount = 1;
        pipeline_layout.pSetLayouts = &descriptor_set_layout_;
        pipeline_layout.pushConstantRangeCount = 1;
        pipeline_layout.pPushConstantRanges = &push;
        check_vk(
            vkCreatePipelineLayout(
                device_, &pipeline_layout, nullptr, &pipeline_layout_),
            "create scientific pipeline layout");

        create_scientific_pipeline();
    }

    void create_scientific_pipeline() {
        const auto shader_directory = executable_directory() / "shaders";
        const auto vertex_code =
            read_spirv(shader_directory / "scientific.vert.spv");
        const auto fragment_code =
            read_spirv(shader_directory / "scientific.frag.spv");
        VkShaderModuleCreateInfo vertex_create{};
        vertex_create.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        vertex_create.codeSize =
            vertex_code.size() * sizeof(std::uint32_t);
        vertex_create.pCode = vertex_code.data();
        VkShaderModuleCreateInfo fragment_create = vertex_create;
        fragment_create.codeSize =
            fragment_code.size() * sizeof(std::uint32_t);
        fragment_create.pCode = fragment_code.data();
        VkShaderModule vertex = VK_NULL_HANDLE;
        VkShaderModule fragment = VK_NULL_HANDLE;
        check_vk(
            vkCreateShaderModule(
                device_, &vertex_create, nullptr, &vertex),
            "create vertex shader module");
        try {
            check_vk(
                vkCreateShaderModule(
                    device_, &fragment_create, nullptr, &fragment),
                "create fragment shader module");

            const std::array stages{
                VkPipelineShaderStageCreateInfo{
                    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    nullptr,
                    0,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    vertex,
                    "main",
                    nullptr},
                VkPipelineShaderStageCreateInfo{
                    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    nullptr,
                    0,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    fragment,
                    "main",
                    nullptr},
            };
            VkPipelineVertexInputStateCreateInfo vertex_input{};
            vertex_input.sType =
                VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            VkPipelineInputAssemblyStateCreateInfo assembly{};
            assembly.sType =
                VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo viewport{};
            viewport.sType =
                VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewport.viewportCount = 1;
            viewport.scissorCount = 1;
            VkPipelineRasterizationStateCreateInfo raster{};
            raster.sType =
                VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            raster.cullMode = VK_CULL_MODE_NONE;
            raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            raster.lineWidth = 1.0F;
            VkPipelineMultisampleStateCreateInfo multisample{};
            multisample.sType =
                VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineColorBlendAttachmentState blend_attachment{};
            blend_attachment.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo blend{};
            blend.sType =
                VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            blend.attachmentCount = 1;
            blend.pAttachments = &blend_attachment;
            constexpr std::array dynamic_states{
                VK_DYNAMIC_STATE_VIEWPORT,
                VK_DYNAMIC_STATE_SCISSOR,
            };
            VkPipelineDynamicStateCreateInfo dynamic{};
            dynamic.sType =
                VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamic.dynamicStateCount =
                static_cast<std::uint32_t>(dynamic_states.size());
            dynamic.pDynamicStates = dynamic_states.data();

            VkGraphicsPipelineCreateInfo create{};
            create.sType =
                VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            create.stageCount =
                static_cast<std::uint32_t>(stages.size());
            create.pStages = stages.data();
            create.pVertexInputState = &vertex_input;
            create.pInputAssemblyState = &assembly;
            create.pViewportState = &viewport;
            create.pRasterizationState = &raster;
            create.pMultisampleState = &multisample;
            create.pColorBlendState = &blend;
            create.pDynamicState = &dynamic;
            create.layout = pipeline_layout_;
            create.renderPass = window_data_.RenderPass;
            create.subpass = 0;
            check_vk(
                vkCreateGraphicsPipelines(
                    device_,
                    VK_NULL_HANDLE,
                    1,
                    &create,
                    nullptr,
                    &pipeline_),
                "create scientific graphics pipeline");
        } catch (...) {
            if (fragment != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device_, fragment, nullptr);
            }
            vkDestroyShaderModule(device_, vertex, nullptr);
            throw;
        }
        vkDestroyShaderModule(device_, fragment, nullptr);
        vkDestroyShaderModule(device_, vertex, nullptr);
    }

    void record_initialize_image(
        const VkCommandBuffer command,
        ScientificImage& image) {
        VkImageMemoryBarrier to_sample{};
        to_sample.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_sample.srcAccessMask = 0;
        to_sample.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        to_sample.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_sample.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_sample.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_sample.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_sample.image = image.image;
        to_sample.subresourceRange.aspectMask =
            VK_IMAGE_ASPECT_COLOR_BIT;
        to_sample.subresourceRange.levelCount = 1;
        to_sample.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(
            command,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &to_sample);
        image.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    void prepare_host_upload(const TileUpload& upload) {
        if (host_staging_mapping_ == nullptr ||
            host_staging_buffer_ == VK_NULL_HANDLE) {
            fail("host upload requested without a Vulkan staging buffer");
        }
        const auto required = satview::viewer::checked_mosaic_bytes(
            upload.height,
            upload.width,
            sizeof(float),
            kMaximumMosaicBytes);
        if (required > host_staging_capacity_ ||
            upload.host_values.size_bytes() != required) {
            fail("host scientific upload has an invalid byte count");
        }
        // The fallback backends intentionally use one persistent staging
        // allocation. Waiting only on this renderer queue keeps its ownership
        // simple and avoids a second full-size host allocation.
        check_vk(vkQueueWaitIdle(queue_), "wait before host staging reuse");
        std::memcpy(
            host_staging_mapping_, upload.host_values.data(), required);
    }

    void record_upload(
        const VkCommandBuffer command,
        const TileUpload& upload,
        const std::uint32_t destination_slot) {
        if (destination_slot >= scientific_images_.size()) {
            fail("invalid Vulkan scientific image destination slot");
        }
        auto& image = scientific_images_[destination_slot];
        VkImageMemoryBarrier to_transfer{};
        to_transfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_transfer.srcAccessMask =
            image.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            ? VK_ACCESS_SHADER_READ_BIT
            : 0;
        to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_transfer.oldLayout = image.layout;
        to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_transfer.image = image.image;
        to_transfer.subresourceRange.aspectMask =
            VK_IMAGE_ASPECT_COLOR_BIT;
        to_transfer.subresourceRange.levelCount = 1;
        to_transfer.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(
            command,
            image.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &to_transfer);

        VkBufferImageCopy copy{};
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = 1;
        copy.imageExtent = {upload.width, upload.height, 1};
        VkBuffer source_buffer = host_staging_buffer_;
#if defined(SATVIEW_HAS_CUDA)
        if (cuda_interop_) {
            source_buffer = exported_buffer_->buffer();
        }
#endif
        vkCmdCopyBufferToImage(
            command,
            source_buffer,
            image.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &copy);

        VkImageMemoryBarrier to_sample{};
        to_sample.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_sample.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_sample.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        to_sample.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_sample.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_sample.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_sample.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_sample.image = image.image;
        to_sample.subresourceRange = to_transfer.subresourceRange;
        vkCmdPipelineBarrier(
            command,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &to_sample);
        image.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    void record_scientific_draw(
        const VkCommandBuffer command,
        const ImDrawData* draw_data,
        const DisplayRect& display,
        const DisplayPushConstants& push) {
        const float scale_x = draw_data->FramebufferScale.x;
        const float scale_y = draw_data->FramebufferScale.y;
        const float pixel_x =
            (display.x - draw_data->DisplayPos.x) * scale_x;
        const float pixel_y =
            (display.y - draw_data->DisplayPos.y) * scale_y;
        const float pixel_width = display.width * scale_x;
        const float pixel_height = display.height * scale_y;
        const float target_width = static_cast<float>(window_data_.Width);
        const float target_height = static_cast<float>(window_data_.Height);
        if (!std::isfinite(pixel_x) || !std::isfinite(pixel_y) ||
            !std::isfinite(pixel_width) || !std::isfinite(pixel_height) ||
            pixel_width < 1.0F || pixel_height < 1.0F ||
            target_width < 1.0F || target_height < 1.0F ||
            pixel_x >= target_width || pixel_y >= target_height ||
            pixel_x + pixel_width <= 0.0F ||
            pixel_y + pixel_height <= 0.0F) {
            return;
        }

        const float clipped_left = std::max(pixel_x, 0.0F);
        const float clipped_top = std::max(pixel_y, 0.0F);
        const float clipped_right =
            std::min(pixel_x + pixel_width, target_width);
        const float clipped_bottom =
            std::min(pixel_y + pixel_height, target_height);
        if (clipped_right - clipped_left < 1.0F ||
            clipped_bottom - clipped_top < 1.0F) {
            return;
        }

        const auto x = static_cast<std::int32_t>(std::floor(clipped_left));
        const auto y = static_cast<std::int32_t>(std::floor(clipped_top));
        const auto right = static_cast<std::int32_t>(
            std::ceil(clipped_right));
        const auto bottom = static_cast<std::int32_t>(
            std::ceil(clipped_bottom));
        const auto width = static_cast<std::uint32_t>(right - x);
        const auto height = static_cast<std::uint32_t>(bottom - y);

        const VkViewport viewport{
            pixel_x,
            pixel_y,
            pixel_width,
            pixel_height,
            0.0F,
            1.0F};
        const VkRect2D scissor{
            {x, y},
            {width, height},
        };
        vkCmdBindPipeline(
            command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        vkCmdSetViewport(command, 0, 1, &viewport);
        vkCmdSetScissor(command, 0, 1, &scissor);
        vkCmdBindDescriptorSets(
            command,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout_,
            0,
            1,
            &descriptor_set_,
            0,
            nullptr);
        vkCmdPushConstants(
            command,
            pipeline_layout_,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(push),
            &push);
        vkCmdDraw(command, 3, 1, 0, 0);
    }

    void destroy_all_resources() noexcept {
        if (device_ != VK_NULL_HANDLE) {
            static_cast<void>(vkDeviceWaitIdle(device_));
        }
        destroy_scientific_resources();
        if (window_data_.Swapchain != VK_NULL_HANDLE) {
            ImGui_ImplVulkanH_DestroyWindow(
                instance_, device_, &window_data_, nullptr);
            window_data_.Swapchain = VK_NULL_HANDLE;
        }
        if (descriptor_pool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
            descriptor_pool_ = VK_NULL_HANDLE;
        }
        if (device_ != VK_NULL_HANDLE) {
            vkDestroyDevice(device_, nullptr);
            device_ = VK_NULL_HANDLE;
        }
        if (surface_ != VK_NULL_HANDLE) {
            SDL_Vulkan_DestroySurface(instance_, surface_, nullptr);
            surface_ = VK_NULL_HANDLE;
        }
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
            instance_ = VK_NULL_HANDLE;
        }
    }

    void destroy_scientific_resources() noexcept {
        if (pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, pipeline_, nullptr);
        }
        if (pipeline_layout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        }
        if (descriptor_set_layout_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(
                device_, descriptor_set_layout_, nullptr);
        }
        if (sampler_ != VK_NULL_HANDLE) {
            vkDestroySampler(device_, sampler_, nullptr);
        }
        for (auto& image : scientific_images_) {
            if (image.view != VK_NULL_HANDLE) {
                vkDestroyImageView(device_, image.view, nullptr);
            }
            if (image.image != VK_NULL_HANDLE) {
                vkDestroyImage(device_, image.image, nullptr);
            }
            if (image.memory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, image.memory, nullptr);
            }
            image = ScientificImage{};
        }
        colormap_lut_.reset();
#if defined(SATVIEW_HAS_CUDA)
        timeline_.reset();
        exported_buffer_.reset();
#endif
        if (host_staging_mapping_ != nullptr) {
            vkUnmapMemory(device_, host_staging_memory_);
            host_staging_mapping_ = nullptr;
        }
        if (host_staging_buffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, host_staging_buffer_, nullptr);
            host_staging_buffer_ = VK_NULL_HANDLE;
        }
        if (host_staging_memory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, host_staging_memory_, nullptr);
            host_staging_memory_ = VK_NULL_HANDLE;
        }
        host_staging_capacity_ = 0;
    }

    SDL_Window* window_ = nullptr;
    std::uint32_t maximum_width_ = 0;
    std::uint32_t maximum_height_ = 0;
    bool cuda_interop_ = false;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    std::uint32_t queue_family_ = std::numeric_limits<std::uint32_t>::max();
    VkQueue queue_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    ImGui_ImplVulkanH_Window window_data_;
    bool swapchain_rebuild_ = false;
    bool imgui_initialized_ = false;

#if defined(SATVIEW_HAS_CUDA)
    std::optional<satview::gpu::ExportedBuffer> exported_buffer_;
    std::optional<satview::gpu::InteropTimeline> timeline_;
#endif
    VkBuffer host_staging_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory host_staging_memory_ = VK_NULL_HANDLE;
    void* host_staging_mapping_ = nullptr;
    VkDeviceSize host_staging_capacity_ = 0;
    std::unique_ptr<satview::viewer::VulkanColormapLut> colormap_lut_;
    std::array<ScientificImage, 2> scientific_images_{};
    VkSampler sampler_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

class TilePipeline {
public:
    virtual ~TilePipeline() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual bool page_locked_reads() const noexcept = 0;
    [[nodiscard]] virtual bool timing_pending() const noexcept = 0;
    [[nodiscard]] virtual bool has_resident_source(
        const TileRequest& request) const noexcept = 0;
    virtual bool poll_timing(float& upload_ms, float& processing_ms) = 0;
    [[nodiscard]] virtual std::optional<DistributionCompletion>
    poll_distribution() = 0;
    virtual TileUpload redispatch(
        const TileRequest& request,
        std::uint64_t previous_vulkan_consumed) = 0;
    virtual TileUpload upload_overview(
        const TileRequest& request,
        std::span<const std::byte> science,
        std::optional<std::span<const std::uint8_t>> validity_mask,
        std::uint32_t width,
        std::uint32_t height,
        std::size_t element_size,
        std::uint64_t previous_vulkan_consumed) = 0;
    virtual std::optional<TileUpload> upload_chunk(
        ReadSlotLease ready,
        const ReadCompletion& completion,
        std::uint64_t previous_vulkan_consumed) = 0;
    virtual void discard(ReadSlotLease ready) = 0;
    virtual void set_layers(const ProductView& view) = 0;
};

#if defined(SATVIEW_HAS_CUDA)
class CudaTilePipeline final : public TilePipeline {
public:
    CudaTilePipeline(
        const std::size_t maximum_science_mosaic_bytes,
        VulkanRenderer& renderer)
        : renderer_(renderer),
          science_capacity_(maximum_science_mosaic_bytes) {
        check_cuda(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "create nonblocking CUDA stream");
        try {
            check_cuda(
                cudaMalloc(&science_, science_capacity_),
                "allocate persistent CUDA science mosaic");
            mask_capacity_ =
                satview::viewer::checked_mosaic_bytes(
                    renderer_.maximum_height(),
                    renderer_.maximum_width(),
                    sizeof(std::uint8_t),
                    kMaximumMosaicBytes);
            check_cuda(
                cudaMalloc(
                    reinterpret_cast<void**>(&mask_),
                    mask_capacity_),
                "allocate persistent CUDA validity-mask mosaic");
            filter_capacity_ =
                satview::viewer::checked_mosaic_bytes(
                    renderer_.maximum_height(),
                    renderer_.maximum_width(),
                    sizeof(float),
                    kMaximumMosaicBytes);
            check_cuda(
                cudaMalloc(
                    reinterpret_cast<void**>(&filter_input_),
                    filter_capacity_),
                "allocate persistent CUDA speckle-filter input");
            for (std::size_t index = 0;
                 index < kMaximumMosaicChunks;
                 ++index) {
                check_cuda(
                    cudaEventCreate(&h2d_started_[index]),
                    "create per-chunk H2D start event");
                check_cuda(
                    cudaEventCreate(&h2d_finished_[index]),
                    "create per-chunk H2D finish event");
            }
            check_cuda(
                cudaEventCreate(&kernel_started_),
                "create CUDA timing event");
            check_cuda(
                cudaEventCreate(&kernel_finished_),
                "create CUDA timing event");
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~CudaTilePipeline() override {
        if (stream_ != nullptr) {
            static_cast<void>(cudaStreamSynchronize(stream_));
        }
        cleanup();
    }

    CudaTilePipeline(const CudaTilePipeline&) = delete;
    CudaTilePipeline& operator=(const CudaTilePipeline&) = delete;

    [[nodiscard]] std::string_view name() const noexcept override {
        return "CUDA";
    }

    [[nodiscard]] bool page_locked_reads() const noexcept override {
        return true;
    }

    [[nodiscard]] bool timing_pending() const noexcept override {
        return timing_pending_ || distribution_.pending();
    }

    [[nodiscard]] bool has_resident_source(
        const TileRequest& request) const noexcept override {
        return resident_.has_value() &&
            same_tile_source(resident_->request, request);
    }

    bool poll_timing(float& h2d_ms, float& transform_ms) override {
        if (!timing_pending_) {
            return true;
        }
        const auto query = cudaEventQuery(kernel_finished_);
        if (query == cudaErrorNotReady) {
            return false;
        }
        check_cuda(query, "query CUDA transform completion");
        h2d_ms = 0.0F;
        for (std::size_t index = 0; index < timing_h2d_count_; ++index) {
            float chunk_milliseconds = 0.0F;
            check_cuda(
                cudaEventElapsedTime(
                    &chunk_milliseconds,
                    h2d_started_[index],
                    h2d_finished_[index]),
                "measure per-chunk H2D time");
            h2d_ms += chunk_milliseconds;
        }
        check_cuda(
            cudaEventElapsedTime(
                &transform_ms, kernel_started_, kernel_finished_),
            "measure CUDA transform time");
        timing_pending_ = false;
        return true;
    }

    [[nodiscard]] std::optional<DistributionCompletion>
    poll_distribution() override {
        satview::gpu::AsyncDistributionResult result;
        if (!distribution_.poll(result)) {
            return std::nullopt;
        }
        if (!distribution_request_.has_value() ||
            distribution_request_->generation != result.generation) {
            fail("resident distribution completion identity mismatch");
        }
        DistributionCompletion completion{
            .request = distribution_request_->request,
            .summary = std::move(result.summary),
            .elapsed_milliseconds = result.elapsed_milliseconds,
            .width = distribution_request_->width,
            .height = distribution_request_->height,
        };
        distribution_request_.reset();
        return completion;
    }

    TileUpload redispatch(
        const TileRequest& request,
        const std::uint64_t previous_vulkan_consumed) override {
        if (timing_pending() || !has_resident_source(request)) {
            fail("invalid resident CUDA mosaic redispatch");
        }
        check_cuda(
            cudaEventRecord(h2d_started_[0], stream_),
            "record resident redispatch start");
        check_cuda(
            cudaEventRecord(h2d_finished_[0], stream_),
            "record zero-H2D resident redispatch");
        timing_h2d_count_ = 1;
        resident_->request = request;
        return launch_transform(*resident_, previous_vulkan_consumed);
    }

    TileUpload upload_overview(
        const TileRequest& request,
        const std::span<const std::byte> science,
        const std::optional<std::span<const std::uint8_t>> validity_mask,
        const std::uint32_t width,
        const std::uint32_t height,
        const std::size_t element_size,
        const std::uint64_t previous_vulkan_consumed) override {
        if (request.source_kind != TileSourceKind::raw_overview ||
            request.overview_identity == 0) {
            fail("raw overview upload requires a nonzero overview identity");
        }
        if (timing_pending()) {
            fail("overview source overwritten before transform completion");
        }
        // A camera change can supersede a partially assembled native page.
        // Its H2D copies are ordered on this same stream, so dropping only the
        // obsolete identity is sufficient; the overview copy safely follows
        // any already-enqueued chunk copies.
        assembly_.reset();
        if (width == 0 || height == 0 ||
            width > renderer_.maximum_width() ||
            height > renderer_.maximum_height()) {
            fail("raw overview extent exceeds the persistent GPU page");
        }
        if (request.layer_index >= layer_lookup_.size() ||
            layer_lookup_[request.layer_index] == nullptr) {
            fail("raw overview references an invalid layer");
        }
        const auto& dataset = *layer_lookup_[request.layer_index];
        if (element_size == 0 ||
            dataset.data_type.element_size != element_size) {
            fail("raw overview datatype does not match its layer");
        }

        const auto expected_science_bytes =
            satview::viewer::checked_mosaic_bytes(
                height,
                width,
                element_size,
                kMaximumMosaicBytes);
        if (science.size() != expected_science_bytes ||
            expected_science_bytes > science_capacity_) {
            fail("raw overview science payload exceeds GPU capacity");
        }
        const auto expected_mask_bytes =
            satview::viewer::checked_mosaic_bytes(
                height,
                width,
                sizeof(std::uint8_t),
                kMaximumMosaicBytes);
        if (expected_mask_bytes > mask_capacity_ ||
            (validity_mask.has_value() &&
             validity_mask->size() != expected_mask_bytes)) {
            fail("raw overview validity payload exceeds GPU capacity");
        }

        // The first copy invalidates the previous resident identity before any
        // portion of the persistent private source allocation is overwritten.
        resident_.reset();
        check_cuda(
            cudaEventRecord(h2d_started_[0], stream_),
            "record raw overview H2D start");
        check_cuda(
            cudaMemcpyAsync(
                science_,
                science.data(),
                expected_science_bytes,
                cudaMemcpyHostToDevice,
                stream_),
            "copy raw overview science to CUDA");
        if (validity_mask.has_value()) {
            check_cuda(
                cudaMemcpyAsync(
                    mask_,
                    validity_mask->data(),
                    expected_mask_bytes,
                    cudaMemcpyHostToDevice,
                    stream_),
                "copy raw overview validity mask to CUDA");
        }
        check_cuda(
            cudaEventRecord(h2d_finished_[0], stream_),
            "record raw overview H2D finish");

        // Unlike PinnedRing leases, arbitrary spans carry no asynchronous
        // lifetime token. Complete only the private H2D stage before returning;
        // the transform and Vulkan publication remain asynchronous.
        check_cuda(
            cudaEventSynchronize(h2d_finished_[0]),
            "complete raw overview H2D");

        resident_ = ResidentMosaic{
            .request = request,
            .width = width,
            .height = height,
            .element_size = element_size,
            .has_validity = validity_mask.has_value(),
        };
        timing_h2d_count_ = 1;
        return launch_transform(*resident_, previous_vulkan_consumed);
    }

    std::optional<TileUpload> upload_chunk(
        ReadSlotLease ready,
        const ReadCompletion& completion,
        const std::uint64_t previous_vulkan_consumed) override {
        if (!completion.plan.has_value() || !ready) {
            fail("invalid mosaic chunk upload");
        }
        if (completion.request.source_kind !=
                TileSourceKind::native_mosaic ||
            completion.request.overview_identity != 0) {
            fail("native mosaic upload has an invalid source identity");
        }
        if (timing_pending()) {
            fail("mosaic source overwritten before transform completion");
        }
        const auto& plan = *completion.plan;
        const auto expected_chunks = static_cast<std::uint32_t>(
            completion.request.mosaic.chunk_rows *
            completion.request.mosaic.chunk_columns);
        if (expected_chunks == 0 ||
            expected_chunks > kMaximumMosaicChunks ||
            completion.chunk_count != expected_chunks ||
            completion.chunk_index >= expected_chunks) {
            fail("invalid mosaic chunk ordinal");
        }

        std::size_t packed_bytes = plan.expected_bytes;
        if (completion.mask_plan.has_value()) {
            if (completion.mask_plan->expected_bytes >
                std::numeric_limits<std::size_t>::max() - packed_bytes) {
                fail("packed validity-mask byte count overflow");
            }
            packed_bytes += completion.mask_plan->expected_bytes;
        }
        if (packed_bytes != ready.bytes().size()) {
            fail("pinned payload does not match mosaic chunk plans");
        }

        if (!assembly_.has_value() ||
            assembly_->request.serial != completion.request.serial) {
            if (completion.chunk_index != 0) {
                fail("mosaic assembly did not begin with chunk zero");
            }
            const auto science_bytes =
                satview::viewer::checked_mosaic_bytes(
                    completion.request.mosaic.pixel_height,
                    completion.request.mosaic.pixel_width,
                    plan.data_type.element_size,
                    kMaximumMosaicBytes);
            if (science_bytes > science_capacity_) {
                fail("science mosaic exceeds persistent CUDA allocation");
            }
            // The first write destroys the previous cache identity before any
            // portion of its persistent source mosaic can be overwritten.
            resident_.reset();
            assembly_ = AssemblyState{
                .request = completion.request,
                .element_size = plan.data_type.element_size,
                .has_validity = completion.mask_plan.has_value(),
                .expected_chunks = expected_chunks,
                .uploaded_chunks = 0,
            };
        }
        if (!same_tile_source(
                assembly_->request, completion.request) ||
            assembly_->request.serial != completion.request.serial ||
            assembly_->element_size != plan.data_type.element_size ||
            assembly_->has_validity != completion.mask_plan.has_value() ||
            completion.chunk_index != assembly_->uploaded_chunks) {
            fail("mosaic chunks arrived with inconsistent assembly metadata");
        }

        const auto event_index =
            static_cast<std::size_t>(completion.chunk_index);
        check_cuda(
            cudaEventRecord(h2d_started_[event_index], stream_),
            "record mosaic chunk H2D start");

        const auto mosaic_width =
            static_cast<std::size_t>(
                completion.request.mosaic.pixel_width);
        const auto mosaic_height =
            static_cast<std::size_t>(
                completion.request.mosaic.pixel_height);
        const auto source_width =
            static_cast<std::size_t>(plan.aligned.width);
        const auto source_height =
            static_cast<std::size_t>(plan.aligned.height);
        const auto element_size = plan.data_type.element_size;
        if (source_width >
                std::numeric_limits<std::size_t>::max() / element_size ||
            mosaic_width >
                std::numeric_limits<std::size_t>::max() / element_size) {
            fail("mosaic row pitch overflow");
        }
        const auto source_pitch = source_width * element_size;
        const auto mosaic_pitch = mosaic_width * element_size;
        const auto destination_row =
            static_cast<std::size_t>(completion.destination_row);
        const auto destination_column =
            static_cast<std::size_t>(completion.destination_column);
        if (destination_row > mosaic_height ||
            destination_column > mosaic_width ||
            source_height > mosaic_height - destination_row ||
            source_width > mosaic_width - destination_column) {
            fail("CUDA chunk destination escapes mosaic bounds");
        }
        if (source_height != 0 &&
            (source_pitch >
                 std::numeric_limits<std::size_t>::max() / source_height ||
             source_pitch * source_height != plan.expected_bytes)) {
            fail("science chunk byte count does not match its 2D extent");
        }
        if (destination_row >
                std::numeric_limits<std::size_t>::max() / mosaic_pitch ||
            destination_column >
                std::numeric_limits<std::size_t>::max() / element_size) {
            fail("mosaic science destination offset overflow");
        }
        const auto destination_row_offset = destination_row * mosaic_pitch;
        const auto destination_column_offset =
            destination_column * element_size;
        if (destination_column_offset >
            std::numeric_limits<std::size_t>::max() -
                destination_row_offset) {
            fail("mosaic science destination offset addition overflow");
        }
        const auto destination_offset =
            destination_row_offset + destination_column_offset;
        check_cuda(
            cudaMemcpy2DAsync(
                static_cast<std::byte*>(science_) + destination_offset,
                mosaic_pitch,
                ready.bytes().data(),
                source_pitch,
                source_pitch,
                source_height,
                cudaMemcpyHostToDevice,
                stream_),
            "assemble pinned science chunk into CUDA mosaic");

        if (completion.mask_plan.has_value()) {
            const auto& mask_plan = *completion.mask_plan;
            if (mask_plan.data_type.kind !=
                    satview::ScalarKind::unsigned_integer ||
                mask_plan.data_type.element_size != sizeof(std::uint8_t) ||
                mask_plan.requested.row != plan.aligned.row ||
                mask_plan.requested.column != plan.aligned.column ||
                mask_plan.requested.height != plan.aligned.height ||
                mask_plan.requested.width != plan.aligned.width) {
                fail("validity-mask plan does not match its science chunk");
            }
            if (mask_plan.requested_row_offset >
                    mask_plan.aligned.height ||
                mask_plan.requested_column_offset >
                    mask_plan.aligned.width ||
                plan.aligned.height >
                    mask_plan.aligned.height -
                        mask_plan.requested_row_offset ||
                plan.aligned.width >
                    mask_plan.aligned.width -
                        mask_plan.requested_column_offset) {
                fail("validity-mask crop escapes its aligned source chunk");
            }
            const auto mask_source_offset = static_cast<std::size_t>(
                mask_plan.requested_row_offset *
                    mask_plan.aligned.width +
                mask_plan.requested_column_offset);
            const auto mask_destination_offset =
                destination_row * mosaic_width + destination_column;
            const auto* mask_source =
                reinterpret_cast<const std::uint8_t*>(
                    ready.bytes().data() + plan.expected_bytes) +
                mask_source_offset;
            check_cuda(
                cudaMemcpy2DAsync(
                    mask_ + mask_destination_offset,
                    mosaic_width,
                    mask_source,
                    static_cast<std::size_t>(mask_plan.aligned.width),
                    source_width,
                    source_height,
                    cudaMemcpyHostToDevice,
                    stream_),
                "assemble pinned validity chunk into CUDA mosaic");
        }
        check_cuda(
            cudaEventRecord(h2d_finished_[event_index], stream_),
            "record mosaic chunk H2D finish");
        ready.mark_in_flight(stream_);

        ++assembly_->uploaded_chunks;
        if (assembly_->uploaded_chunks != assembly_->expected_chunks) {
            return std::nullopt;
        }

        resident_ = ResidentMosaic{
            .request = assembly_->request,
            .width = static_cast<std::uint32_t>(
                assembly_->request.mosaic.pixel_width),
            .height = static_cast<std::uint32_t>(
                assembly_->request.mosaic.pixel_height),
            .element_size = assembly_->element_size,
            .has_validity = assembly_->has_validity,
        };
        timing_h2d_count_ = assembly_->expected_chunks;
        assembly_.reset();
        return launch_transform(*resident_, previous_vulkan_consumed);
    }

    void discard(ReadSlotLease ready) override {
        if (ready) {
            ready.mark_in_flight(stream_);
        }
    }

    void set_layers(const ProductView& view) override {
        layer_lookup_.clear();
        layer_lookup_.reserve(view.layers.size());
        for (const auto& layer : view.layers) {
            layer_lookup_.push_back(layer.dataset);
        }
    }

private:
    struct AssemblyState {
        TileRequest request;
        std::size_t element_size = 0;
        bool has_validity = false;
        std::uint32_t expected_chunks = 0;
        std::uint32_t uploaded_chunks = 0;
    };

    struct ResidentMosaic {
        TileRequest request;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::size_t element_size = 0;
        bool has_validity = false;
    };

    struct DistributionRequest {
        std::uint64_t generation = 0;
        TileRequest request;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
    };

    TileUpload launch_transform(
        const ResidentMosaic& resident,
        const std::uint64_t previous_vulkan_consumed) {
        if (previous_vulkan_consumed != 0) {
            renderer_.timeline().enqueue_cuda_wait(
                stream_, previous_vulkan_consumed);
        }
        check_cuda(
            cudaEventRecord(kernel_started_, stream_),
            "record CUDA mosaic transform start");

        const auto count = satview::viewer::checked_mosaic_bytes(
            resident.height,
            resident.width,
            sizeof(std::uint8_t),
            kMaximumMosaicBytes);
        const auto& dataset =
            resident.request.layer_index < layer_lookup_.size()
            ? *layer_lookup_[resident.request.layer_index]
            : throw std::runtime_error("invalid layer dispatch index");
        if (dataset.data_type.element_size != resident.element_size) {
            fail("resident mosaic datatype changed before transform");
        }
        const bool filter_enabled = resident.request.speckle.filter !=
            satview::cpu::SpeckleFilter::none;
        const auto output_bytes = satview::viewer::checked_mosaic_bytes(
            resident.height,
            resident.width,
            sizeof(float),
            kMaximumMosaicBytes);
        if (filter_enabled && output_bytes > filter_capacity_) {
            fail("speckle-filter scratch capacity exceeded");
        }
        auto* const final_output =
            static_cast<float*>(renderer_.cuda_output());
        float* const transform_output =
            filter_enabled ? filter_input_ : final_output;
        satview::gpu::TransformOptions options{};
        options.validity.input = resident.has_validity ? mask_ : nullptr;
        options.stream = stream_;

        cudaError_t launch = cudaErrorInvalidValue;
        if (dataset.layer_kind == satview::LayerKind::gslc_polarization) {
            satview::gpu::GslcTransform transform{};
            switch (resident.request.mode) {
                case DisplayMode::amplitude:
                    transform = satview::gpu::GslcTransform::amplitude;
                    break;
                case DisplayMode::power:
                    transform = satview::gpu::GslcTransform::power;
                    break;
                case DisplayMode::power_db:
                    transform = satview::gpu::GslcTransform::power_db;
                    break;
                case DisplayMode::phase:
                    transform = satview::gpu::GslcTransform::phase;
                    break;
                case DisplayMode::real:
                    transform = satview::gpu::GslcTransform::real;
                    break;
                case DisplayMode::imaginary:
                    transform = satview::gpu::GslcTransform::imaginary;
                    break;
                default:
                    fail("unsupported GSLC display mode");
            }
            launch = satview::gpu::launch_gslc_transform(
                static_cast<const float2*>(science_),
                transform_output,
                count,
                transform,
                options);
        } else if (
            dataset.data_type.kind ==
            satview::ScalarKind::compound_complex) {
            const auto transform =
                resident.request.mode == DisplayMode::phase
                ? satview::gpu::GcovComplexTransform::phase
                : satview::gpu::GcovComplexTransform::magnitude;
            launch = satview::gpu::launch_gcov_complex_transform(
                static_cast<const float2*>(science_),
                transform_output,
                count,
                transform,
                options);
        } else {
            const auto transform =
                resident.request.mode == DisplayMode::power_db
                ? satview::gpu::GcovRealTransform::power_db
                : satview::gpu::GcovRealTransform::linear;
            launch = satview::gpu::launch_gcov_real_transform(
                static_cast<const float*>(science_),
                transform_output,
                count,
                transform,
                options);
        }
        check_cuda(launch, "launch scientific CUDA mosaic transform");
        if (filter_enabled) {
            const auto domain = speckle_domain_for(
                dataset, resident.request.mode);
            if (!domain.has_value()) {
                fail("speckle filter requested for an unsupported display mode");
            }
            satview::gpu::SpeckleFilterOptions filter_options{};
            filter_options.filter =
                resident.request.speckle.filter ==
                    satview::cpu::SpeckleFilter::lee
                ? satview::gpu::SpeckleFilter::lee
                : satview::gpu::SpeckleFilter::boxcar;
            switch (*domain) {
                case satview::cpu::SpeckleDomain::amplitude:
                    filter_options.domain =
                        satview::gpu::SpeckleDomain::amplitude;
                    break;
                case satview::cpu::SpeckleDomain::linear_power:
                    filter_options.domain =
                        satview::gpu::SpeckleDomain::linear_power;
                    break;
                case satview::cpu::SpeckleDomain::power_db:
                    filter_options.domain =
                        satview::gpu::SpeckleDomain::power_db;
                    break;
            }
            filter_options.window_size =
                resident.request.speckle.window_size;
            filter_options.equivalent_number_of_looks =
                resident.request.speckle.equivalent_number_of_looks;
            filter_options.validity.input =
                resident.has_validity ? mask_ : nullptr;
            filter_options.stream = stream_;
            check_cuda(
                satview::gpu::launch_speckle_filter(
                    filter_input_, final_output, resident.width,
                    resident.height, filter_options),
                "launch scientific CUDA speckle filter");
        }
        check_cuda(
            cudaEventRecord(kernel_finished_, stream_),
            "record CUDA scientific processing completion");

        const std::uint64_t distribution_generation =
            next_distribution_generation_++;
        distribution_.enqueue(
            final_output,
            count,
            stream_,
            distribution_generation);
        distribution_request_ = DistributionRequest{
            .generation = distribution_generation,
            .request = resident.request,
            .width = resident.width,
            .height = resident.height,
        };

        const std::uint64_t cuda_ready = next_timeline_value_;
        const std::uint64_t vulkan_consumed = cuda_ready + 1;
        next_timeline_value_ += 2;
        renderer_.timeline().enqueue_cuda_signal(stream_, cuda_ready);
        timing_pending_ = true;
        return {
            .request = resident.request,
            .cuda_ready_value = cuda_ready,
            .vulkan_consumed_value = vulkan_consumed,
            .width = resident.width,
            .height = resident.height,
        };
    }

    void cleanup() noexcept {
        if (kernel_finished_ != nullptr) {
            static_cast<void>(cudaEventDestroy(kernel_finished_));
        }
        if (kernel_started_ != nullptr) {
            static_cast<void>(cudaEventDestroy(kernel_started_));
        }
        for (std::size_t index = 0;
             index < kMaximumMosaicChunks;
             ++index) {
            if (h2d_finished_[index] != nullptr) {
                static_cast<void>(
                    cudaEventDestroy(h2d_finished_[index]));
            }
            if (h2d_started_[index] != nullptr) {
                static_cast<void>(cudaEventDestroy(h2d_started_[index]));
            }
        }
        if (filter_input_ != nullptr) {
            static_cast<void>(cudaFree(filter_input_));
            filter_input_ = nullptr;
        }
        if (mask_ != nullptr) {
            static_cast<void>(cudaFree(mask_));
            mask_ = nullptr;
        }
        if (science_ != nullptr) {
            static_cast<void>(cudaFree(science_));
            science_ = nullptr;
        }
        if (stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(stream_));
            stream_ = nullptr;
        }
    }

    VulkanRenderer& renderer_;
    std::size_t science_capacity_ = 0;
    std::size_t mask_capacity_ = 0;
    std::size_t filter_capacity_ = 0;
    cudaStream_t stream_ = nullptr;
    void* science_ = nullptr;
    std::uint8_t* mask_ = nullptr;
    float* filter_input_ = nullptr;
    std::array<cudaEvent_t, kMaximumMosaicChunks> h2d_started_{};
    std::array<cudaEvent_t, kMaximumMosaicChunks> h2d_finished_{};
    cudaEvent_t kernel_started_ = nullptr;
    cudaEvent_t kernel_finished_ = nullptr;
    std::size_t timing_h2d_count_ = 0;
    bool timing_pending_ = false;
    satview::gpu::AsyncResidentDistribution distribution_;
    std::uint64_t next_distribution_generation_ = 1;
    std::optional<DistributionRequest> distribution_request_;
    std::uint64_t next_timeline_value_ = 1;
    std::vector<const satview::DatasetInfo*> layer_lookup_;
    std::optional<AssemblyState> assembly_;
    std::optional<ResidentMosaic> resident_;
};
#endif

class HostTilePipeline final : public TilePipeline {
public:
    HostTilePipeline(
        const ComputeBackend backend,
        const std::size_t maximum_science_mosaic_bytes,
        const VulkanRenderer& renderer)
        : backend_(backend),
          science_capacity_(maximum_science_mosaic_bytes),
          science_words_((maximum_science_mosaic_bytes + sizeof(float) - 1) /
                         sizeof(float)) {
        if (backend_ == ComputeBackend::cuda) {
            fail("host tile pipeline cannot use the CUDA backend");
        }
        const auto maximum_pixels = satview::viewer::checked_mosaic_bytes(
            renderer.maximum_height(),
            renderer.maximum_width(),
            sizeof(std::uint8_t),
            kMaximumMosaicBytes);
        mask_.resize(maximum_pixels);
        output_.resize(maximum_pixels);
        filter_input_.resize(maximum_pixels);
    }

    [[nodiscard]] std::string_view name() const noexcept override {
        return backend_name(backend_);
    }

    [[nodiscard]] bool page_locked_reads() const noexcept override {
        return false;
    }

    [[nodiscard]] bool timing_pending() const noexcept override {
        return timing_pending_;
    }

    [[nodiscard]] bool has_resident_source(
        const TileRequest& request) const noexcept override {
        return resident_.has_value() &&
            same_tile_source(resident_->request, request);
    }

    bool poll_timing(float& upload_ms, float& processing_ms) override {
        if (!timing_pending_) {
            return true;
        }
        upload_ms = upload_milliseconds_;
        processing_ms = processing_milliseconds_;
        timing_pending_ = false;
        return true;
    }

    [[nodiscard]] std::optional<DistributionCompletion>
    poll_distribution() override {
        return std::exchange(distribution_, std::nullopt);
    }

    TileUpload redispatch(
        const TileRequest& request,
        const std::uint64_t previous_vulkan_consumed) override {
        static_cast<void>(previous_vulkan_consumed);
        if (timing_pending() || !has_resident_source(request)) {
            fail("invalid resident host mosaic redispatch");
        }
        upload_milliseconds_ = 0.0F;
        resident_->request = request;
        return launch_transform(*resident_);
    }

    TileUpload upload_overview(
        const TileRequest& request,
        const std::span<const std::byte> science,
        const std::optional<std::span<const std::uint8_t>> validity_mask,
        const std::uint32_t width,
        const std::uint32_t height,
        const std::size_t element_size,
        const std::uint64_t previous_vulkan_consumed) override {
        static_cast<void>(previous_vulkan_consumed);
        if (request.source_kind != TileSourceKind::raw_overview ||
            request.overview_identity == 0) {
            fail("raw overview upload requires a nonzero overview identity");
        }
        if (timing_pending()) {
            fail("overview source overwritten before host processing completion");
        }
        assembly_.reset();
        if (width == 0 || height == 0) {
            fail("raw overview extent is empty");
        }
        if (request.layer_index >= layer_lookup_.size() ||
            layer_lookup_[request.layer_index] == nullptr) {
            fail("raw overview references an invalid layer");
        }
        const auto& dataset = *layer_lookup_[request.layer_index];
        if (element_size == 0 || dataset.data_type.element_size != element_size) {
            fail("raw overview datatype does not match its layer");
        }
        const auto expected_science_bytes =
            satview::viewer::checked_mosaic_bytes(
                height, width, element_size, kMaximumMosaicBytes);
        const auto expected_mask_bytes =
            satview::viewer::checked_mosaic_bytes(
                height, width, sizeof(std::uint8_t), kMaximumMosaicBytes);
        if (science.size() != expected_science_bytes ||
            expected_science_bytes > science_capacity_ ||
            expected_mask_bytes > mask_.size() ||
            (validity_mask.has_value() &&
             validity_mask->size() != expected_mask_bytes)) {
            fail("raw overview payload exceeds host capacity");
        }

        resident_.reset();
        const auto started = Clock::now();
        std::memcpy(science_bytes().data(), science.data(), science.size());
        if (validity_mask.has_value()) {
            std::memcpy(
                mask_.data(), validity_mask->data(), validity_mask->size());
        }
        upload_milliseconds_ = static_cast<float>(
            std::chrono::duration<double, std::milli>(Clock::now() - started)
                .count());
        resident_ = ResidentMosaic{
            .request = request,
            .width = width,
            .height = height,
            .element_size = element_size,
            .has_validity = validity_mask.has_value(),
        };
        return launch_transform(*resident_);
    }

    std::optional<TileUpload> upload_chunk(
        ReadSlotLease ready,
        const ReadCompletion& completion,
        const std::uint64_t previous_vulkan_consumed) override {
        static_cast<void>(previous_vulkan_consumed);
        if (!completion.plan.has_value() || !ready) {
            fail("invalid host mosaic chunk upload");
        }
        if (completion.request.source_kind != TileSourceKind::native_mosaic ||
            completion.request.overview_identity != 0) {
            fail("native mosaic upload has an invalid source identity");
        }
        if (timing_pending()) {
            fail("mosaic source overwritten before host processing completion");
        }
        const auto& plan = *completion.plan;
        const auto expected_chunks = static_cast<std::uint32_t>(
            completion.request.mosaic.chunk_rows *
            completion.request.mosaic.chunk_columns);
        if (expected_chunks == 0 || expected_chunks > kMaximumMosaicChunks ||
            completion.chunk_count != expected_chunks ||
            completion.chunk_index >= expected_chunks) {
            fail("invalid mosaic chunk ordinal");
        }
        std::size_t packed_bytes = plan.expected_bytes;
        if (completion.mask_plan.has_value()) {
            if (completion.mask_plan->expected_bytes >
                std::numeric_limits<std::size_t>::max() - packed_bytes) {
                fail("packed validity-mask byte count overflow");
            }
            packed_bytes += completion.mask_plan->expected_bytes;
        }
        if (packed_bytes != ready.bytes().size()) {
            fail("host payload does not match mosaic chunk plans");
        }

        if (!assembly_.has_value() ||
            assembly_->request.serial != completion.request.serial) {
            if (completion.chunk_index != 0) {
                fail("mosaic assembly did not begin with chunk zero");
            }
            const auto science_bytes_required =
                satview::viewer::checked_mosaic_bytes(
                    completion.request.mosaic.pixel_height,
                    completion.request.mosaic.pixel_width,
                    plan.data_type.element_size,
                    kMaximumMosaicBytes);
            if (science_bytes_required > science_capacity_) {
                fail("science mosaic exceeds persistent host allocation");
            }
            resident_.reset();
            assembly_ = AssemblyState{
                .request = completion.request,
                .element_size = plan.data_type.element_size,
                .has_validity = completion.mask_plan.has_value(),
                .expected_chunks = expected_chunks,
                .uploaded_chunks = 0,
            };
            assembly_upload_milliseconds_ = 0.0;
        }
        if (!same_tile_source(assembly_->request, completion.request) ||
            assembly_->request.serial != completion.request.serial ||
            assembly_->element_size != plan.data_type.element_size ||
            assembly_->has_validity != completion.mask_plan.has_value() ||
            completion.chunk_index != assembly_->uploaded_chunks) {
            fail("mosaic chunks arrived with inconsistent assembly metadata");
        }

        const auto started = Clock::now();
        const auto mosaic_width = static_cast<std::size_t>(
            completion.request.mosaic.pixel_width);
        const auto mosaic_height = static_cast<std::size_t>(
            completion.request.mosaic.pixel_height);
        const auto source_width = static_cast<std::size_t>(plan.aligned.width);
        const auto source_height = static_cast<std::size_t>(plan.aligned.height);
        const auto element_size = plan.data_type.element_size;
        const auto destination_row = static_cast<std::size_t>(
            completion.destination_row);
        const auto destination_column = static_cast<std::size_t>(
            completion.destination_column);
        if (source_width > std::numeric_limits<std::size_t>::max() /
                element_size ||
            mosaic_width > std::numeric_limits<std::size_t>::max() /
                element_size ||
            destination_row > mosaic_height ||
            destination_column > mosaic_width ||
            source_height > mosaic_height - destination_row ||
            source_width > mosaic_width - destination_column) {
            fail("host chunk destination escapes mosaic bounds");
        }
        const auto source_pitch = source_width * element_size;
        const auto mosaic_pitch = mosaic_width * element_size;
        if (source_height != 0 &&
            (source_pitch > std::numeric_limits<std::size_t>::max() /
                 source_height ||
             source_pitch * source_height != plan.expected_bytes)) {
            fail("science chunk byte count does not match its 2D extent");
        }
        auto destination = science_bytes();
        for (std::size_t row = 0; row < source_height; ++row) {
            const auto destination_offset =
                (destination_row + row) * mosaic_pitch +
                destination_column * element_size;
            std::memcpy(
                destination.data() + destination_offset,
                ready.bytes().data() + row * source_pitch,
                source_pitch);
        }

        if (completion.mask_plan.has_value()) {
            const auto& mask_plan = *completion.mask_plan;
            if (mask_plan.data_type.kind !=
                    satview::ScalarKind::unsigned_integer ||
                mask_plan.data_type.element_size != sizeof(std::uint8_t) ||
                mask_plan.requested.row != plan.aligned.row ||
                mask_plan.requested.column != plan.aligned.column ||
                mask_plan.requested.height != plan.aligned.height ||
                mask_plan.requested.width != plan.aligned.width ||
                mask_plan.requested_row_offset > mask_plan.aligned.height ||
                mask_plan.requested_column_offset > mask_plan.aligned.width ||
                plan.aligned.height > mask_plan.aligned.height -
                    mask_plan.requested_row_offset ||
                plan.aligned.width > mask_plan.aligned.width -
                    mask_plan.requested_column_offset) {
                fail("validity-mask plan does not match its science chunk");
            }
            const auto mask_source_offset = static_cast<std::size_t>(
                mask_plan.requested_row_offset * mask_plan.aligned.width +
                mask_plan.requested_column_offset);
            const auto* const mask_source =
                reinterpret_cast<const std::uint8_t*>(
                    ready.bytes().data() + plan.expected_bytes) +
                mask_source_offset;
            const auto mask_source_pitch = static_cast<std::size_t>(
                mask_plan.aligned.width);
            for (std::size_t row = 0; row < source_height; ++row) {
                std::memcpy(
                    mask_.data() +
                        (destination_row + row) * mosaic_width +
                        destination_column,
                    mask_source + row * mask_source_pitch,
                    source_width);
            }
        }
        ready.mark_consumed();
        assembly_upload_milliseconds_ +=
            std::chrono::duration<double, std::milli>(Clock::now() - started)
                .count();

        ++assembly_->uploaded_chunks;
        if (assembly_->uploaded_chunks != assembly_->expected_chunks) {
            return std::nullopt;
        }
        resident_ = ResidentMosaic{
            .request = assembly_->request,
            .width = static_cast<std::uint32_t>(
                assembly_->request.mosaic.pixel_width),
            .height = static_cast<std::uint32_t>(
                assembly_->request.mosaic.pixel_height),
            .element_size = assembly_->element_size,
            .has_validity = assembly_->has_validity,
        };
        upload_milliseconds_ =
            static_cast<float>(assembly_upload_milliseconds_);
        assembly_.reset();
        return launch_transform(*resident_);
    }

    void discard(ReadSlotLease ready) override {
        if (ready) {
            ready.mark_consumed();
        }
    }

    void set_layers(const ProductView& view) override {
        layer_lookup_.clear();
        layer_lookup_.reserve(view.layers.size());
        for (const auto& layer : view.layers) {
            layer_lookup_.push_back(layer.dataset);
        }
    }

private:
    struct AssemblyState {
        TileRequest request;
        std::size_t element_size = 0;
        bool has_validity = false;
        std::uint32_t expected_chunks = 0;
        std::uint32_t uploaded_chunks = 0;
    };

    struct ResidentMosaic {
        TileRequest request;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::size_t element_size = 0;
        bool has_validity = false;
    };

    [[nodiscard]] std::span<std::byte> science_bytes() noexcept {
        return std::as_writable_bytes(std::span<float>(science_words_));
    }

    [[nodiscard]] static satview::cpu::SpeckleDomain cpu_speckle_domain(
        const satview::cpu::SpeckleDomain domain) {
        switch (domain) {
            case satview::cpu::SpeckleDomain::amplitude:
                return satview::cpu::SpeckleDomain::amplitude;
            case satview::cpu::SpeckleDomain::linear_power:
                return satview::cpu::SpeckleDomain::linear_power;
            case satview::cpu::SpeckleDomain::power_db:
                return satview::cpu::SpeckleDomain::power_db;
        }
        fail("invalid speckle domain");
    }

    [[nodiscard]] satview::experimental::PageRequest make_page_request(
        const ResidentMosaic& resident,
        const satview::DatasetInfo& dataset,
        const std::span<const std::uint8_t> validity) const {
        const auto count = static_cast<std::size_t>(resident.width) *
            static_cast<std::size_t>(resident.height);
        satview::experimental::PageRequest request;
        request.science = std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(science_words_.data()),
            count * resident.element_size);
        request.validity = validity;
        request.width = resident.width;
        request.height = resident.height;
        if (dataset.data_type.kind ==
            satview::ScalarKind::compound_complex) {
            request.input_kind =
                satview::experimental::InputKind::complex_float32;
            if (dataset.layer_kind == satview::LayerKind::gslc_polarization) {
                switch (resident.request.mode) {
                    case DisplayMode::amplitude:
                        request.complex_transform =
                            satview::cpu::ComplexTransform::amplitude;
                        break;
                    case DisplayMode::power:
                        request.complex_transform =
                            satview::cpu::ComplexTransform::power;
                        break;
                    case DisplayMode::power_db:
                        request.complex_transform =
                            satview::cpu::ComplexTransform::power_db;
                        break;
                    case DisplayMode::phase:
                        request.complex_transform =
                            satview::cpu::ComplexTransform::phase;
                        break;
                    case DisplayMode::real:
                        request.complex_transform =
                            satview::cpu::ComplexTransform::real;
                        break;
                    case DisplayMode::imaginary:
                        request.complex_transform =
                            satview::cpu::ComplexTransform::imaginary;
                        break;
                    default:
                        fail("unsupported GSLC host display mode");
                }
            } else {
                request.complex_transform =
                    resident.request.mode == DisplayMode::phase
                    ? satview::cpu::ComplexTransform::phase
                    : satview::cpu::ComplexTransform::amplitude;
            }
        } else {
            request.input_kind = satview::experimental::InputKind::real_float32;
            request.real_transform =
                resident.request.mode == DisplayMode::power_db
                ? satview::cpu::RealTransform::power_db
                : satview::cpu::RealTransform::linear;
        }
        request.filter_enabled = resident.request.speckle.filter !=
            satview::cpu::SpeckleFilter::none;
        if (request.filter_enabled) {
            const auto domain = speckle_domain_for(
                dataset, resident.request.mode);
            if (!domain.has_value()) {
                fail("speckle filter requested for an unsupported display mode");
            }
            request.speckle.filter =
                resident.request.speckle.filter ==
                    satview::cpu::SpeckleFilter::lee
                ? satview::cpu::SpeckleFilter::lee
                : satview::cpu::SpeckleFilter::boxcar;
            request.speckle.domain = cpu_speckle_domain(*domain);
            request.speckle.window_size =
                resident.request.speckle.window_size;
            request.speckle.equivalent_number_of_looks =
                resident.request.speckle.equivalent_number_of_looks;
        }
        return request;
    }

    void process_cpu(
        const satview::experimental::PageRequest& request,
        const std::span<float> output) {
        const auto count = output.size();
        auto transformed = request.filter_enabled
            ? std::span<float>(filter_input_).first(count)
            : output;
        if (request.input_kind ==
            satview::experimental::InputKind::complex_float32) {
            const auto* const input =
                reinterpret_cast<const satview::cpu::Complex32*>(
                    request.science.data());
            satview::cpu::transform_complex(
                std::span<const satview::cpu::Complex32>(input, count),
                transformed,
                request.complex_transform,
                request.validity);
        } else {
            const auto* const input = reinterpret_cast<const float*>(
                request.science.data());
            satview::cpu::transform_real(
                std::span<const float>(input, count),
                transformed,
                request.real_transform,
                request.validity);
        }
        if (request.filter_enabled) {
            satview::cpu::filter_speckle(
                transformed,
                output,
                request.width,
                request.height,
                request.speckle,
                request.validity);
        }
    }

    TileUpload launch_transform(const ResidentMosaic& resident) {
        const auto count = satview::viewer::checked_mosaic_bytes(
            resident.height,
            resident.width,
            sizeof(std::uint8_t),
            kMaximumMosaicBytes);
        const auto& dataset =
            resident.request.layer_index < layer_lookup_.size() &&
                layer_lookup_[resident.request.layer_index] != nullptr
            ? *layer_lookup_[resident.request.layer_index]
            : throw std::runtime_error("invalid layer dispatch index");
        if (dataset.data_type.element_size != resident.element_size ||
            count > output_.size()) {
            fail("resident host mosaic datatype or extent is invalid");
        }
        const auto validity = resident.has_validity
            ? std::span<const std::uint8_t>(mask_).first(count)
            : std::span<const std::uint8_t>{};
        const auto request = make_page_request(resident, dataset, validity);
        const auto output = std::span<float>(output_).first(count);
        const auto started = Clock::now();
        switch (backend_) {
            case ComputeBackend::cpu:
                process_cpu(request, output);
                break;
            case ComputeBackend::hip:
#if defined(SATVIEW_HAS_EXPERIMENTAL_HIP)
                satview::experimental::process_hip(request, output);
                break;
#else
                fail("experimental HIP backend was not compiled");
#endif
            case ComputeBackend::sycl:
#if defined(SATVIEW_HAS_EXPERIMENTAL_SYCL)
                satview::experimental::process_sycl(request, output);
                break;
#else
                fail("experimental SYCL backend was not compiled");
#endif
            case ComputeBackend::cuda:
                fail("CUDA dispatch reached the host tile pipeline");
        }
        processing_milliseconds_ = static_cast<float>(
            std::chrono::duration<double, std::milli>(Clock::now() - started)
                .count());
        timing_pending_ = true;

        const auto cpu_histogram = satview::cpu::build_histogram(output);
        satview::gpu::DistributionHistogram histogram;
        histogram.finite_count = cpu_histogram.finite_count;
        histogram.invalid_count = cpu_histogram.invalid_count;
        histogram.minimum = cpu_histogram.minimum;
        histogram.maximum = cpu_histogram.maximum;
        histogram.bins = cpu_histogram.bins;
        distribution_ = DistributionCompletion{
            .request = resident.request,
            .summary = satview::gpu::summarize_distribution(histogram),
            .elapsed_milliseconds = 0.0F,
            .width = resident.width,
            .height = resident.height,
        };
        return TileUpload{
            .request = resident.request,
            .width = resident.width,
            .height = resident.height,
            .host_values = output,
        };
    }

    ComputeBackend backend_ = ComputeBackend::cpu;
    std::size_t science_capacity_ = 0;
    std::vector<float> science_words_;
    std::vector<std::uint8_t> mask_;
    std::vector<float> output_;
    std::vector<float> filter_input_;
    float upload_milliseconds_ = 0.0F;
    float processing_milliseconds_ = 0.0F;
    double assembly_upload_milliseconds_ = 0.0;
    bool timing_pending_ = false;
    std::vector<const satview::DatasetInfo*> layer_lookup_;
    std::optional<AssemblyState> assembly_;
    std::optional<ResidentMosaic> resident_;
    std::optional<DistributionCompletion> distribution_;
};

class SdlSession final {
public:
    SdlSession() {
        SDL_SetMainReady();
        if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
            throw std::runtime_error(
                std::string("SDL initialization failed: ") + SDL_GetError());
        }
        const float scale = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
        window_ = SDL_CreateWindow(
            "NISAR Data Reader",
            static_cast<int>(1440.0F * scale),
            static_cast<int>(900.0F * scale),
            SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE |
                SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_HIDDEN);
        if (window_ == nullptr) {
            const std::string error = SDL_GetError();
            SDL_Quit();
            throw std::runtime_error("SDL window creation failed: " + error);
        }
    }

    ~SdlSession() {
        if (window_ != nullptr) {
            SDL_DestroyWindow(window_);
        }
        SDL_Quit();
    }

    [[nodiscard]] SDL_Window* window() const noexcept {
        return window_;
    }

    void toggle_fullscreen() {
        const bool fullscreen =
            (SDL_GetWindowFlags(window_) & SDL_WINDOW_FULLSCREEN) != 0;
        if (!SDL_SetWindowFullscreen(window_, !fullscreen)) {
            throw std::runtime_error(
                std::string("fullscreen toggle failed: ") + SDL_GetError());
        }
    }

private:
    SDL_Window* window_ = nullptr;
};

class ImGuiSession final {
public:
    ImGuiSession(SDL_Window* window, VulkanRenderer& renderer)
        : renderer_(renderer) {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        auto& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.ConfigDpiScaleFonts = true;
        ImGui::StyleColorsDark();
        const float scale =
            SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
        ImGui::GetStyle().ScaleAllSizes(scale);
        ImGui::GetStyle().FontScaleDpi = scale;
        renderer_.initialize_imgui();
        SDL_SetWindowPosition(
            window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
        SDL_ShowWindow(window);
    }

    ~ImGuiSession() {
        renderer_.shutdown_imgui();
        ImGui::DestroyContext();
    }

private:
    VulkanRenderer& renderer_;
};

void reset_display_for_mode(
    const DisplayMode mode,
    DisplayPushConstants& display) {
    display.gamma = 1.0F;
    display.circular_phase =
        mode == DisplayMode::phase ? std::uint32_t{1} : std::uint32_t{0};
    switch (mode) {
        case DisplayMode::power_db:
            display.low = -35.0F;
            display.high = 5.0F;
            display.colormap = static_cast<std::uint32_t>(
                satview::viewer::ColormapId::d3_viridis);
            break;
        case DisplayMode::phase:
            display.low = -3.14159265F;
            display.high = 3.14159265F;
            display.colormap = static_cast<std::uint32_t>(
                satview::viewer::ColormapId::cyclic_phase);
            break;
        case DisplayMode::real:
        case DisplayMode::imaginary:
            display.low = -1.0F;
            display.high = 1.0F;
            display.colormap = static_cast<std::uint32_t>(
                satview::viewer::ColormapId::cmweather_balance);
            break;
        default:
            display.low = 0.0F;
            display.high = 1.0F;
            display.colormap = static_cast<std::uint32_t>(
                satview::viewer::ColormapId::d3_viridis);
            break;
    }
}

void apply_display_window(
    DisplayPushConstants& display,
    const satview::gpu::DisplayWindow& window) noexcept {
    display.low = window.low;
    display.high = window.high;
}

void draw_numeric_display_controls(
    const DisplayMode mode,
    DisplayPushConstants& display) {
    satview::gpu::DisplayWindow window{display.low, display.high};
    const float step = satview::gpu::display_control_step(window);
    const float fast_step = std::min(
        step * 10.0F, std::numeric_limits<float>::max());

    float low_candidate = window.low;
    if (ImGui::InputFloat(
            "Low", &low_candidate, step, fast_step, "%.7g",
            ImGuiInputTextFlags_CharsScientific)) {
        if (satview::gpu::try_set_window_low(window, low_candidate)) {
            apply_display_window(display, window);
        }
    }

    float high_candidate = window.high;
    if (ImGui::InputFloat(
            "High", &high_candidate, step, fast_step, "%.7g",
            ImGuiInputTextFlags_CharsScientific)) {
        if (satview::gpu::try_set_window_high(window, high_candidate)) {
            apply_display_window(display, window);
        }
    }

    float gamma_candidate = display.gamma;
    constexpr float gamma_step = 0.05F;
    constexpr float gamma_fast_step = 0.25F;
    if (ImGui::InputFloat(
            "Gamma", &gamma_candidate, gamma_step, gamma_fast_step, "%.6g",
            ImGuiInputTextFlags_CharsScientific)) {
        static_cast<void>(
            satview::gpu::try_set_gamma(display.gamma, gamma_candidate));
    }

    satview::gpu::DisplayWindow dragged{display.low, display.high};
    if (ImGui::DragFloatRange2(
            "Window drag", &dragged.low, &dragged.high, step,
            0.0F, 0.0F, "%.6g", "%.6g")) {
        if (std::isfinite(dragged.low) &&
            std::isfinite(dragged.high) &&
            dragged.low < dragged.high) {
            apply_display_window(display, dragged);
        }
    }
    if (ImGui::Button("Reset display")) {
        reset_display_for_mode(mode, display);
    }
}

[[nodiscard]] bool draw_speckle_controls(
    const satview::DatasetInfo& layer,
    const DisplayMode mode,
    const std::optional<ResidentViewMapping>& resident,
    SpeckleSettings& settings) {
    ImGui::SeparatorText("Speckle reduction");
    constexpr std::array<const char*, 3> filter_labels{{
        "None (exact source)",
        "Boxcar mean",
        "Lee adaptive",
    }};
    constexpr std::array<satview::cpu::SpeckleFilter, 3> filters{{
        satview::cpu::SpeckleFilter::none,
        satview::cpu::SpeckleFilter::boxcar,
        satview::cpu::SpeckleFilter::lee,
    }};
    const bool supported = speckle_domain_for(layer, mode).has_value();
    auto filter_iterator =
        std::find(filters.begin(), filters.end(), settings.filter);
    int filter_index = filter_iterator == filters.end()
        ? 0
        : static_cast<int>(std::distance(filters.begin(), filter_iterator));
    bool changed = false;
    if (!supported) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Combo(
            "Filter", &filter_index, filter_labels.data(),
            static_cast<int>(filter_labels.size()))) {
        settings.filter = filters[static_cast<std::size_t>(filter_index)];
        changed = true;
    }
    if (!supported) {
        ImGui::EndDisabled();
        ImGui::TextDisabled("Unavailable for this layer/mode.");
    }

    const bool filtering = supported &&
        settings.filter != satview::cpu::SpeckleFilter::none;
    if (!filtering) {
        ImGui::BeginDisabled();
    }
    constexpr std::array<const char*, 3> window_labels{{
        "3 x 3", "5 x 5", "7 x 7",
    }};
    constexpr std::array<std::uint32_t, 3> windows{{3, 5, 7}};
    const auto window_iterator =
        std::find(windows.begin(), windows.end(), settings.window_size);
    int window_index = window_iterator == windows.end()
        ? 1
        : static_cast<int>(std::distance(windows.begin(), window_iterator));
    if (ImGui::Combo(
            "Window", &window_index, window_labels.data(),
            static_cast<int>(window_labels.size()))) {
        settings.window_size =
            windows[static_cast<std::size_t>(window_index)];
        changed = true;
    }
    if (!filtering) {
        ImGui::EndDisabled();
    }

    if (filtering && settings.filter == satview::cpu::SpeckleFilter::lee) {
        float looks_candidate = settings.equivalent_number_of_looks;
        const float looks_step = std::max(0.1F, looks_candidate * 0.1F);
        const float looks_fast_step = std::min(
            looks_step * 10.0F, std::numeric_limits<float>::max());
        if (ImGui::InputFloat(
                "Equivalent looks", &looks_candidate, looks_step,
                looks_fast_step, "%.7g",
                ImGuiInputTextFlags_CharsScientific) &&
            std::isfinite(looks_candidate) && looks_candidate > 0.0F) {
            settings.equivalent_number_of_looks = looks_candidate;
            changed = true;
        }
    }

    if (filtering) {
        ImGui::TextDisabled("Filtering uses linear power.");
        if (resident.has_value()) {
            const auto& mapping = *resident;
            const auto radius_span =
                static_cast<std::uint64_t>(settings.window_size - 1);
            const auto native_span = [radius_span](const std::uint64_t stride)
                -> std::optional<std::uint64_t> {
                if (radius_span != 0 && stride >
                    (std::numeric_limits<std::uint64_t>::max() - 1) /
                        radius_span) {
                    return std::nullopt;
                }
                return radius_span * stride + 1;
            };
            const auto native_rows = native_span(mapping.sample_stride_row);
            const auto native_columns =
                native_span(mapping.sample_stride_column);
            if (!native_rows.has_value() || !native_columns.has_value()) {
                ImGui::TextDisabled(
                    "Effective native footprint exceeds the 64-bit display range.");
            } else if (mapping.sample_stride_row == 1 &&
                mapping.sample_stride_column == 1) {
                ImGui::TextDisabled(
                    "%u x %u native-sample window.",
                    settings.window_size, settings.window_size);
            } else {
                ImGui::TextWrapped(
                    "%u x %u samples, stride %llu x %llu; native span %llu x %llu.",
                    settings.window_size, settings.window_size,
                    static_cast<unsigned long long>(
                        mapping.sample_stride_row),
                    static_cast<unsigned long long>(
                        mapping.sample_stride_column),
                    static_cast<unsigned long long>(*native_rows),
                    static_cast<unsigned long long>(*native_columns));
            }
        } else {
            ImGui::TextDisabled(
                "Effective native footprint appears when a page is resident.");
        }
    }
    return changed;
}

void draw_distribution(
    const ProductView& view,
    const std::size_t selected_layer,
    const DisplayMode mode,
    const std::optional<ResidentViewMapping>& resident,
    const DistributionCompletion* distribution,
    DisplayPushConstants& display) {
    ImGui::SeparatorText("Resident distribution");
    if (distribution == nullptr || !resident.has_value() ||
        distribution->request.layer_index != selected_layer ||
        distribution->request.mode != mode) {
        ImGui::TextDisabled("Waiting for resident page...");
        return;
    }

    const auto& mapping = *resident;
    const auto& layer = view.layers.at(selected_layer);
    const bool whole_scene = mapping.actual_scene.row == 0 &&
        mapping.actual_scene.column == 0 &&
        mapping.actual_scene.height == layer.dataset->dimensions[0] &&
        mapping.actual_scene.width == layer.dataset->dimensions[1];
    if (whole_scene) {
        if (distribution->request.source_kind == TileSourceKind::raw_overview) {
            ImGui::TextWrapped(
                "Scope: whole-scene sparse sample, stride %llu x %llu",
                static_cast<unsigned long long>(mapping.sample_stride_row),
                static_cast<unsigned long long>(mapping.sample_stride_column));
        } else {
            ImGui::TextWrapped(
                "Scope: whole-scene native resident, stride %llu x %llu",
                static_cast<unsigned long long>(mapping.sample_stride_row),
                static_cast<unsigned long long>(mapping.sample_stride_column));
        }
    } else {
        ImGui::TextWrapped(
            "%s page: %llu,%llu  %llu x %llu  stride %llu x %llu",
            distribution->request.source_kind == TileSourceKind::native_mosaic
                ? "regional native" : "regional sampled",
            static_cast<unsigned long long>(mapping.actual_scene.row),
            static_cast<unsigned long long>(mapping.actual_scene.column),
            static_cast<unsigned long long>(mapping.actual_scene.height),
            static_cast<unsigned long long>(mapping.actual_scene.width),
            static_cast<unsigned long long>(mapping.sample_stride_row),
            static_cast<unsigned long long>(mapping.sample_stride_column));
    }

    const auto& summary = distribution->summary;
    const auto& histogram = summary.histogram;
    ImGui::Text(
        "Grid %u x %u  |  finite %llu  |  invalid %llu",
        distribution->width,
        distribution->height,
        static_cast<unsigned long long>(histogram.finite_count),
        static_cast<unsigned long long>(histogram.invalid_count));
    ImGui::Text("Statistics GPU %.3f ms", distribution->elapsed_milliseconds);
    if (!summary.has_finite_values()) {
        ImGui::TextDisabled("No finite samples in this resident page.");
        return;
    }

    ImGui::Text("Min %.7g  |  max %.7g", histogram.minimum, histogram.maximum);
    ImGui::Text(
        "p1 %.7g  p2 %.7g  p50 %.7g",
        summary.percentile_1, summary.percentile_2, summary.percentile_50);
    ImGui::Text(
        "p98 %.7g  p99 %.7g", summary.percentile_98, summary.percentile_99);

    std::array<float, satview::gpu::kDistributionHistogramBins> plot{};
    float plot_maximum = 0.0F;
    for (std::size_t index = 0; index < plot.size(); ++index) {
        plot[index] = static_cast<float>(std::log1p(
            static_cast<double>(histogram.bins[index])));
        plot_maximum = std::max(plot_maximum, plot[index]);
    }
    ImGui::PlotHistogram(
        "##resident_distribution", plot.data(),
        static_cast<int>(plot.size()), 0, "log-count histogram", 0.0F,
        std::max(plot_maximum, 1.0F),
        ImVec2(0.0F, 72.0F * ImGui::GetMainViewport()->DpiScale));
    if (ImGui::IsItemHovered()) {
        const ImVec2 item_min = ImGui::GetItemRectMin();
        const ImVec2 item_max = ImGui::GetItemRectMax();
        const float fraction = std::clamp(
            (ImGui::GetIO().MousePos.x - item_min.x) /
                std::max(item_max.x - item_min.x, 1.0F),
            0.0F,
            std::nextafter(1.0F, 0.0F));
        const std::size_t bin = std::min(
            static_cast<std::size_t>(fraction * plot.size()),
            plot.size() - 1);
        const double range = static_cast<double>(histogram.maximum) -
            static_cast<double>(histogram.minimum);
        const double bin_low = static_cast<double>(histogram.minimum) +
            range * static_cast<double>(bin) / plot.size();
        const double bin_high = static_cast<double>(histogram.minimum) +
            range * static_cast<double>(bin + 1) / plot.size();
        ImGui::SetTooltip(
            "[%.7g, %.7g]\n%llu finite samples",
            bin_low, bin_high,
            static_cast<unsigned long long>(histogram.bins[bin]));
    }

    const auto apply_preset = [&](const satview::gpu::AutoWindowPreset preset) {
        const auto window = satview::gpu::auto_window(summary, preset);
        if (window.has_value()) {
            apply_display_window(display, *window);
        }
    };
    if (ImGui::Button("Full finite")) {
        apply_preset(satview::gpu::AutoWindowPreset::full_finite_range);
    }
    ImGui::SameLine();
    if (ImGui::Button("1-99%")) {
        apply_preset(satview::gpu::AutoWindowPreset::percentile_1_99);
    }
    ImGui::SameLine();
    if (ImGui::Button("2-98%")) {
        apply_preset(satview::gpu::AutoWindowPreset::percentile_2_98);
    }
}

[[nodiscard]] satview::viewer::RasterMetrics navigation_raster(
    const LayerView& layer) {
    return satview::viewer::RasterMetrics{
        .rows = layer.dataset->dimensions[0],
        .columns = layer.dataset->dimensions[1],
        .row_spacing = layer.y_spacing,
        .column_spacing = layer.x_spacing,
    };
}

[[nodiscard]] satview::viewer::ScreenViewport navigation_viewport(
    const DisplayRect& rect) {
    return satview::viewer::ScreenViewport{
        rect.x, rect.y, rect.width, rect.height};
}

[[nodiscard]] bool same_camera(
    const satview::viewer::Camera2D& left,
    const satview::viewer::Camera2D& right) noexcept {
    return left.center_row == right.center_row &&
        left.center_column == right.center_column &&
        left.world_units_per_screen_pixel ==
            right.world_units_per_screen_pixel &&
        left.rotation_radians == right.rotation_radians;
}

[[nodiscard]] satview::viewer::Camera2D fit_pixel_window(
    const satview::viewer::PixelWindow& window,
    const satview::viewer::RasterMetrics& raster,
    const satview::viewer::ScreenViewport& viewport,
    const double rotation_radians = 0.0) {
    if (window.height == 0 || window.width == 0) {
        return satview::viewer::fit_camera(
            raster, viewport, rotation_radians);
    }
    const double row_spacing = std::abs(raster.row_spacing);
    const double column_spacing = std::abs(raster.column_spacing);
    const double physical_height =
        static_cast<double>(window.height) * row_spacing;
    const double physical_width =
        static_cast<double>(window.width) * column_spacing;
    const double cosine = std::abs(std::cos(rotation_radians));
    const double sine = std::abs(std::sin(rotation_radians));
    return satview::viewer::clamp_camera(
        satview::viewer::Camera2D{
            static_cast<double>(window.row) +
                0.5 * static_cast<double>(window.height),
            static_cast<double>(window.column) +
                0.5 * static_cast<double>(window.width),
            std::max(
                (sine * physical_width + cosine * physical_height) /
                    viewport.height,
                (cosine * physical_width + sine * physical_height) /
                    viewport.width),
            rotation_radians},
        raster,
        viewport);
}

[[nodiscard]] satview::viewer::Camera2D zoom_with_limit(
    const satview::viewer::Camera2D& camera,
    const satview::viewer::RasterMetrics& raster,
    const satview::viewer::ScreenViewport& viewport,
    const satview::viewer::ScreenPoint anchor,
    double factor) {
    // Thirty-two logical pixels per sample is enough magnification for exact
    // native inspection without letting trackpad deltas drive scale to zero.
    constexpr double maximum_native_magnification = 32.0;
    const double minimum_scale =
        std::max(std::abs(raster.row_spacing),
                 std::abs(raster.column_spacing)) /
        maximum_native_magnification;
    factor = std::max(
        factor,
        minimum_scale / camera.world_units_per_screen_pixel);
    return satview::viewer::zoom_about(
        camera, raster, viewport, anchor, factor);
}

[[nodiscard]] std::uint64_t focus_chunk(
    const double center,
    const std::uint64_t raster_extent,
    const std::uint32_t chunk_extent,
    const std::uint64_t chunk_count) {
    const double last = std::nextafter(
        static_cast<double>(raster_extent), 0.0);
    const auto pixel = static_cast<std::uint64_t>(std::floor(
        std::clamp(center, 0.0, last)));
    return std::min(chunk_count - 1, pixel / chunk_extent);
}

[[nodiscard]] std::optional<ResidentSamplePushConstants>
resident_sample_for_window(
    const ResidentViewMapping& resident,
    const satview::viewer::RasterWindow& window) {
    ResidentSamplePushConstants sample;
    if (resident.sample_stride_row == 0 ||
        resident.sample_stride_column == 0 ||
        resident.texture_width == 0 || resident.texture_height == 0 ||
        !std::isfinite(window.row_begin) ||
        !std::isfinite(window.column_begin) ||
        !std::isfinite(window.row_end) ||
        !std::isfinite(window.column_end) ||
        window.row_end <= window.row_begin ||
        window.column_end <= window.column_begin) {
        return std::nullopt;
    }

    const double actual_row_begin =
        static_cast<double>(resident.actual_scene.row);
    const double actual_column_begin =
        static_cast<double>(resident.actual_scene.column);
    const double actual_row_end = actual_row_begin +
        static_cast<double>(resident.actual_scene.height);
    const double actual_column_end = actual_column_begin +
        static_cast<double>(resident.actual_scene.width);
    const double virtual_row_begin = resident.texture_origin_row;
    const double virtual_column_begin = resident.texture_origin_column;
    const double virtual_row_extent =
        static_cast<double>(resident.texture_height) *
        static_cast<double>(resident.sample_stride_row);
    const double virtual_column_extent =
        static_cast<double>(resident.texture_width) *
        static_cast<double>(resident.sample_stride_column);
    if (!std::isfinite(virtual_row_begin) ||
        !std::isfinite(virtual_column_begin) ||
        !std::isfinite(virtual_row_extent) ||
        !std::isfinite(virtual_column_extent) ||
        virtual_row_extent <= 0.0 || virtual_column_extent <= 0.0) {
        return std::nullopt;
    }

    const double coordinate_scale = std::max({
        1.0,
        std::abs(actual_row_begin),
        std::abs(actual_column_begin),
        std::abs(actual_row_end),
        std::abs(actual_column_end),
        std::abs(virtual_row_begin + virtual_row_extent),
        std::abs(virtual_column_begin + virtual_column_extent)});
    const double tolerance = 64.0 *
        std::numeric_limits<double>::epsilon() * coordinate_scale;
    const bool virtual_intersects_actual =
        actual_row_end > virtual_row_begin - tolerance &&
        actual_column_end > virtual_column_begin - tolerance &&
        actual_row_begin < virtual_row_begin + virtual_row_extent + tolerance &&
        actual_column_begin <
            virtual_column_begin + virtual_column_extent + tolerance;
    const bool intersects_actual =
        window.row_end > actual_row_begin &&
        window.column_end > actual_column_begin &&
        window.row_begin < actual_row_end &&
        window.column_begin < actual_column_end;
    if (!virtual_intersects_actual || !intersects_actual) {
        return std::nullopt;
    }

    sample.valid_extent = {resident.texture_width, resident.texture_height};
    sample.sample_uv_origin = {
        static_cast<float>(
            (window.column_begin - virtual_column_begin) /
            virtual_column_extent),
        static_cast<float>(
            (window.row_begin - virtual_row_begin) / virtual_row_extent)};
    sample.sample_uv_extent = {
        static_cast<float>(
            (window.column_end - window.column_begin) /
            virtual_column_extent),
        static_cast<float>(
            (window.row_end - window.row_begin) / virtual_row_extent)};
    return sample;
}

struct ResidentDrawView {
    satview::viewer::RasterWindow window;
    DisplayRect rect;
    ResidentSamplePushConstants sample;
    std::array<float, 2> window_uv_origin;
    std::array<float, 2> window_uv_dx;
    std::array<float, 2> window_uv_dy;
};

[[nodiscard]] std::optional<ResidentDrawView> resident_draw_view(
    const ResidentViewMapping& resident,
    const satview::viewer::Camera2D& camera,
    const satview::viewer::RasterMetrics& raster,
    const satview::viewer::ScreenViewport& screen,
    const satview::viewer::RasterWindow& visible) {
    if (resident.actual_scene.row >= raster.rows ||
        resident.actual_scene.column >= raster.columns ||
        resident.actual_scene.height >
            raster.rows - resident.actual_scene.row ||
        resident.actual_scene.width >
            raster.columns - resident.actual_scene.column) {
        return std::nullopt;
    }

    const double actual_row_begin =
        static_cast<double>(resident.actual_scene.row);
    const double actual_column_begin =
        static_cast<double>(resident.actual_scene.column);
    const double actual_row_end = actual_row_begin +
        static_cast<double>(resident.actual_scene.height);
    const double actual_column_end = actual_column_begin +
        static_cast<double>(resident.actual_scene.width);
    const double row_begin = std::max(visible.row_begin, actual_row_begin);
    const double column_begin =
        std::max(visible.column_begin, actual_column_begin);
    const double row_end = std::min(visible.row_end, actual_row_end);
    const double column_end =
        std::min(visible.column_end, actual_column_end);
    if (row_end <= row_begin || column_end <= column_begin) {
        return std::nullopt;
    }

    const satview::viewer::RasterWindow window{
        row_begin, column_begin, row_end, column_end};
    const auto sample = resident_sample_for_window(resident, window);
    if (!sample.has_value()) {
        return std::nullopt;
    }

    const std::array raster_corners{
        satview::viewer::RasterPoint{row_begin, column_begin},
        satview::viewer::RasterPoint{row_begin, column_end},
        satview::viewer::RasterPoint{row_end, column_begin},
        satview::viewer::RasterPoint{row_end, column_end}};
    double minimum_x = std::numeric_limits<double>::infinity();
    double minimum_y = std::numeric_limits<double>::infinity();
    double maximum_x = -std::numeric_limits<double>::infinity();
    double maximum_y = -std::numeric_limits<double>::infinity();
    for (const auto corner : raster_corners) {
        const auto point = satview::viewer::raster_to_screen(
            camera, raster, screen, corner);
        minimum_x = std::min(minimum_x, point.x);
        minimum_y = std::min(minimum_y, point.y);
        maximum_x = std::max(maximum_x, point.x);
        maximum_y = std::max(maximum_y, point.y);
    }
    const double left = std::max(screen.x, minimum_x);
    const double top = std::max(screen.y, minimum_y);
    const double right = std::min(screen.x + screen.width, maximum_x);
    const double bottom = std::min(screen.y + screen.height, maximum_y);
    if (!(right > left) || !(bottom > top)) {
        return std::nullopt;
    }

    const DisplayRect rect{
        static_cast<float>(left),
        static_cast<float>(top),
        static_cast<float>(right - left),
        static_cast<float>(bottom - top)};
    const double row_extent = row_end - row_begin;
    const double column_extent = column_end - column_begin;
    const auto window_uv = [&](const double x, const double y) {
        const auto point = satview::viewer::screen_to_raster(
            camera,
            raster,
            screen,
            satview::viewer::ScreenPoint{x, y});
        return std::array<double, 2>{
            (point.column - column_begin) / column_extent,
            (point.row - row_begin) / row_extent};
    };
    const auto origin = window_uv(left, top);
    const auto horizontal = window_uv(right, top);
    const auto vertical = window_uv(left, bottom);
    return ResidentDrawView{
        .window = window,
        .rect = rect,
        .sample = *sample,
        .window_uv_origin = {
            static_cast<float>(origin[0]),
            static_cast<float>(origin[1])},
        .window_uv_dx = {
            static_cast<float>(horizontal[0] - origin[0]),
            static_cast<float>(horizontal[1] - origin[1])},
        .window_uv_dy = {
            static_cast<float>(vertical[0] - origin[0]),
            static_cast<float>(vertical[1] - origin[1])},
    };
}

ViewportUiResult build_ui(
    const satview::Hdf5Product& product,
    const ProductView& view,
    const VulkanRenderer& renderer,
    std::size_t& selected_layer,
    DisplayMode& mode,
    SpeckleSettings& speckle,
    std::uint64_t& tile_row,
    std::uint64_t& tile_column,
    std::uint32_t& mosaic_span,
    ViewportNavigationState& navigation,
    bool& controls_visible,
    const std::optional<ResidentViewMapping>& resident,
    const DistributionCompletion* distribution,
    DisplayPushConstants& display,
    bool& request_tile,
    const bool image_valid,
    const double hdf5_ms,
    const float h2d_ms,
    const float transform_ms,
    const std::string_view compute_backend,
    const std::string& status,
    const ReadRingState ring) {
    ViewportUiResult result;
    bool focus_control_changed = false;
    bool frame_selected_footprint = false;
    bool fit_scene_requested = false;
    bool mode_control_changed = false;
    bool speckle_control_changed = false;
    double button_zoom_factor = 1.0;
    std::optional<double> requested_rotation;
    double button_rotation_delta = 0.0;
    bool rotation_request_immediate = false;

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    auto& io = ImGui::GetIO();
    if (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Tab, false)) {
        controls_visible = !controls_visible;
    }
    const float panel_width = controls_visible
        ? 360.0F * viewport->DpiScale
        : 0.0F;
    if (controls_visible) {
        ImGui::SetNextWindowPos(viewport->Pos);
        ImGui::SetNextWindowSize({panel_width, viewport->Size.y});
        constexpr ImGuiWindowFlags panel_flags =
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoTitleBar;
        ImGui::Begin("NISAR controls", nullptr, panel_flags);
        if (ImGui::SmallButton("Hide [Tab]")) {
            controls_visible = false;
        }
        ImGui::Separator();
        const auto file_name = product.file_path().filename().string();
        ImGui::Text("%s  |  %s",
                    product.identification().product_type_text.c_str(),
                    file_name.c_str());
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", product.file_path().string().c_str());
        }
        ImGui::Text(
            "GPU: %s | %.*s",
            renderer.device_name().c_str(),
            static_cast<int>(compute_backend.size()),
            compute_backend.data());

        const char* selected_label = view.layers[selected_layer].label.c_str();
        if (ImGui::BeginCombo("Semantic layer", selected_label)) {
            for (std::size_t index = 0; index < view.layers.size(); ++index) {
                const bool selected = index == selected_layer;
                if (ImGui::Selectable(view.layers[index].label.c_str(),
                                      selected)) {
                    selected_layer = index;
                    const auto available =
                        modes_for(*view.layers[index].dataset);
                    mode = available.front().mode;
                    if (view.layers[index].dataset->layer_kind ==
                        satview::LayerKind::gslc_polarization) {
                        mode = DisplayMode::power_db;
                    }
                    tile_row = (view.layers[index].tile_rows - 1) / 2;
                    tile_column = (view.layers[index].tile_columns - 1) / 2;
                    navigation.initialized = false;
                    reset_display_for_mode(mode, display);
                    request_tile = true;
                    result.immediate_request = true;
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        const auto choices = modes_for(*view.layers[selected_layer].dataset);
        const auto current = std::find_if(
            choices.begin(), choices.end(),
            [mode](const ModeChoice& choice) { return choice.mode == mode; });
        const char* mode_label =
            current == choices.end() ? choices.front().label : current->label;
        if (ImGui::BeginCombo("Scientific mode", mode_label)) {
            for (const auto& choice : choices) {
                const bool selected = choice.mode == mode;
                if (ImGui::Selectable(choice.label, selected)) {
                    mode_control_changed = choice.mode != mode;
                    mode = choice.mode;
                    reset_display_for_mode(mode, display);
                    request_tile = true;
                    result.immediate_request = true;
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        const auto& colormaps = satview::viewer::kColormapLabels;
        int colormap = static_cast<int>(display.colormap);
        if (ImGui::Combo("Colormap", &colormap, colormaps.data(),
                         static_cast<int>(colormaps.size()))) {
            display.colormap = static_cast<std::uint32_t>(colormap);
        }
        draw_numeric_display_controls(mode, display);
        constexpr std::array<const char*, 2> sampling_labels{{
            "Smooth",
            "Exact Pixels",
        }};
        int sampling_mode = static_cast<int>(display.sampling_mode);
        if (ImGui::Combo("Sampling", &sampling_mode, sampling_labels.data(),
                         static_cast<int>(sampling_labels.size()))) {
            display.sampling_mode = static_cast<std::uint32_t>(sampling_mode);
        }

        const std::optional<ResidentViewMapping> speckle_resident =
            resident.has_value() && resident->layer_index == selected_layer
                ? resident
                : std::nullopt;
        if (draw_speckle_controls(*view.layers[selected_layer].dataset, mode,
                                  speckle_resident, speckle)) {
            speckle_control_changed = true;
            request_tile = true;
            result.immediate_request = true;
        }

        draw_distribution(view, selected_layer, mode, resident, distribution,
                          display);

        ImGui::SeparatorText("Navigation");
        if (ImGui::Button("-##camera_zoom")) {
            button_zoom_factor *= std::sqrt(2.0);
        }
        ImGui::SameLine();
        if (ImGui::Button("+##camera_zoom")) {
            button_zoom_factor /= std::sqrt(2.0);
        }
        ImGui::SameLine();
        if (ImGui::Button("Fit Scene")) {
            fit_scene_requested = true;
        }

        float rotation_radians =
            navigation.initialized
                ? static_cast<float>(navigation.camera.rotation_radians)
                : 0.0F;
        if (ImGui::SliderAngle("Rotation", &rotation_radians, -180.0F, 180.0F,
                               "%.1f deg")) {
            requested_rotation = static_cast<double>(rotation_radians);
        }
        if (ImGui::Button("-90 deg")) {
            button_rotation_delta -= 0.5 * kPi;
            rotation_request_immediate = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("+90 deg")) {
            button_rotation_delta += 0.5 * kPi;
            rotation_request_immediate = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset##rotation")) {
            requested_rotation = 0.0;
            button_rotation_delta = 0.0;
            rotation_request_immediate = true;
        }
        ImGui::TextDisabled("Wheel zoom | drag pan | Q/E rotate | Tab hide");

        ImGui::SeparatorText("Center chunk");
        const auto& layer = view.layers[selected_layer];
        const std::uint64_t one = 1;
        const std::uint64_t zero = 0;
        const std::uint64_t maximum_row = layer.tile_rows - 1;
        const std::uint64_t maximum_column = layer.tile_columns - 1;
        if (ImGui::InputScalar("Chunk row", ImGuiDataType_U64, &tile_row, &one,
                               nullptr, "%llu")) {
            tile_row = std::clamp(tile_row, zero, maximum_row);
            focus_control_changed = true;
            request_tile = true;
            result.immediate_request = true;
        }
        if (ImGui::InputScalar("Chunk column", ImGuiDataType_U64, &tile_column,
                               &one, nullptr, "%llu")) {
            tile_column = std::clamp(tile_column, zero, maximum_column);
            focus_control_changed = true;
            request_tile = true;
            result.immediate_request = true;
        }
        ImGui::Text("Grid: %llu x %llu chunks  |  source %u x %u",
                    static_cast<unsigned long long>(layer.tile_rows),
                    static_cast<unsigned long long>(layer.tile_columns),
                    layer.tile_height, layer.tile_width);

        constexpr std::array footprint_labels{
            "1 x 1 source chunk",
            "2 x 2 source chunks",
            "4 x 4 source chunks",
        };
        constexpr std::array<std::uint32_t, 3> footprint_spans{1, 2, 4};
        const auto footprint_iterator = std::find(
            footprint_spans.begin(), footprint_spans.end(), mosaic_span);
        int footprint_index = static_cast<int>(
            std::distance(footprint_spans.begin(), footprint_iterator));
        if (footprint_iterator == footprint_spans.end()) {
            footprint_index = 0;
        }
        if (ImGui::Combo("Footprint", &footprint_index,
                         footprint_labels.data(),
                         static_cast<int>(footprint_labels.size()))) {
            mosaic_span =
                footprint_spans[static_cast<std::size_t>(footprint_index)];
            frame_selected_footprint = true;
            request_tile = true;
            result.immediate_request = true;
        }
        const auto footprint = satview::viewer::make_mosaic_geometry(
            tile_row, tile_column, layer.tile_rows, layer.tile_columns,
            layer.tile_height, layer.tile_width, layer.dataset->dimensions[0],
            layer.dataset->dimensions[1], mosaic_span);
        ImGui::Text("Native guard: %u x %u chunks, %llu x %llu pixels",
                    footprint.chunk_rows, footprint.chunk_columns,
                    static_cast<unsigned long long>(footprint.pixel_width),
                    static_cast<unsigned long long>(footprint.pixel_height));

        ImGui::SeparatorText("Pipeline");
        if (!status.empty()) {
            ImGui::TextWrapped("%s", status.c_str());
        }
        ImGui::Text("Source/prepare    %8.3f ms", hdf5_ms);
        ImGui::Text("Upload             %8.3f ms", h2d_ms);
        ImGui::Text("Processing         %8.3f ms", transform_ms);
        ImGui::Text("Frame              %8.3f ms",
                    1000.0F / std::max(ImGui::GetIO().Framerate, 0.001F));
        ImGui::Text("Read ring F/R/G/I: %zu/%zu/%zu/%zu", ring.free,
                    ring.ready, ring.filling, ring.in_flight);
        ImGui::End();
    }

    const auto& layer = view.layers[selected_layer];
    const auto footprint = satview::viewer::make_mosaic_geometry(
        tile_row, tile_column, layer.tile_rows, layer.tile_columns,
        layer.tile_height, layer.tile_width, layer.dataset->dimensions[0],
        layer.dataset->dimensions[1], mosaic_span);
    const ImVec2 available_position{viewport->Pos.x + panel_width,
                                    viewport->Pos.y};
    const ImVec2 available_size{std::max(1.0F, viewport->Size.x - panel_width),
                                viewport->Size.y};
    ImGui::SetNextWindowPos(available_position);
    ImGui::SetNextWindowSize(available_size);
    ImGui::SetNextWindowBgAlpha(0.0F);
    constexpr ImGuiWindowFlags viewport_flags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoBringToFrontOnFocus;
    ImGui::Begin("Scientific GPU viewport", nullptr, viewport_flags);

    // Dear ImGui mouse positions and deltas are logical coordinates. Vulkan's
    // framebuffer conversion stays in record_scientific_draw(), so camera math
    // must not apply DpiScale or FramebufferScale a second time.
    const ImVec2 content_size = ImGui::GetContentRegionAvail();
    const ImVec2 canvas_size{std::max(content_size.x, 1.0F),
                             std::max(content_size.y, 1.0F)};
    ImGui::InvisibleButton("##scientific_canvas", canvas_size,
                           ImGuiButtonFlags_MouseButtonLeft |
                               ImGuiButtonFlags_MouseButtonMiddle);
    const ImVec2 canvas_min = ImGui::GetItemRectMin();
    const ImVec2 canvas_max = ImGui::GetItemRectMax();
    const DisplayRect canvas{
        canvas_min.x,
        canvas_min.y,
        std::max(canvas_max.x - canvas_min.x, 1.0F),
        std::max(canvas_max.y - canvas_min.y, 1.0F)};
    const auto screen = navigation_viewport(canvas);
    const auto raster = navigation_raster(layer);
    result.canvas = screen;
    const bool canvas_extent_changed =
        navigation.canvas_extent_initialized &&
        (navigation.canvas_width != screen.width ||
         navigation.canvas_height != screen.height);
    navigation.canvas_width = screen.width;
    navigation.canvas_height = screen.height;
    navigation.canvas_extent_initialized = true;
    const bool canvas_hovered = ImGui::IsItemHovered();
    const bool canvas_active = ImGui::IsItemActive();
    if (canvas_hovered) {
        ImGui::SetItemKeyOwner(ImGuiKey_MouseWheelY);
    }

    auto update_camera = [&](const satview::viewer::Camera2D& candidate,
                             const bool immediate) {
        if (!navigation.initialized ||
            !same_camera(navigation.camera, candidate)) {
            navigation.camera = candidate;
            navigation.initialized = true;
            result.camera_changed = true;
            result.immediate_request = result.immediate_request || immediate;
        }
    };

    if (!navigation.initialized ||
        navigation.layer_index != selected_layer) {
        result.native_footprint_requested = true;
        navigation.layer_index = selected_layer;
        update_camera(
            fit_pixel_window(
                satview::viewer::PixelWindow{
                    footprint.pixel_row,
                    footprint.pixel_column,
                    footprint.pixel_height,
                    footprint.pixel_width},
                raster,
                screen),
            true);
    } else if (canvas_extent_changed) {
        // A resize or DPI/layout change alters the visible source window even
        // when the numeric camera does not move. Re-canonicalize it for the
        // new canvas and always arm the debounced residency scheduler.
        update_camera(
            satview::viewer::clamp_camera(
                navigation.camera, raster, screen),
            false);
        result.camera_changed = true;
    }
    if (focus_control_changed || frame_selected_footprint) {
        result.native_footprint_requested = true;
        update_camera(
            fit_pixel_window(
                satview::viewer::PixelWindow{
                    footprint.pixel_row,
                    footprint.pixel_column,
                    footprint.pixel_height,
                    footprint.pixel_width},
                raster,
                screen,
                navigation.camera.rotation_radians),
            true);
    }
    if (fit_scene_requested) {
        // Fit is also an explicit retry command if the already-fitted view
        // previously failed to load, so keep this hint even without motion.
        result.immediate_request = true;
        result.native_footprint_requested = false;
        update_camera(
            satview::viewer::fit_camera(
                raster, screen, navigation.camera.rotation_radians),
            true);
    }

    const satview::viewer::ScreenPoint canvas_center{
        static_cast<double>(canvas.x) + 0.5 * canvas.width,
        static_cast<double>(canvas.y) + 0.5 * canvas.height};
    if (button_zoom_factor != 1.0) {
        update_camera(
            zoom_with_limit(
                navigation.camera,
                raster,
                screen,
                canvas_center,
                button_zoom_factor),
            true);
    }

    if ((canvas_hovered || canvas_active) && !io.WantTextInput) {
        if (ImGui::IsKeyPressed(ImGuiKey_Q, false)) {
            button_rotation_delta -= 0.5 * kPi;
            rotation_request_immediate = true;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_E, false)) {
            button_rotation_delta += 0.5 * kPi;
            rotation_request_immediate = true;
        }
    }
    if (requested_rotation.has_value() || button_rotation_delta != 0.0) {
        auto rotated = navigation.camera;
        rotated.rotation_radians =
            requested_rotation.value_or(rotated.rotation_radians) +
            button_rotation_delta;
        update_camera(
            satview::viewer::clamp_camera(rotated, raster, screen),
            rotation_request_immediate);
    }

    if (canvas_hovered && io.MouseWheel != 0.0F) {
        update_camera(
            zoom_with_limit(
                navigation.camera,
                raster,
                screen,
                satview::viewer::ScreenPoint{
                    static_cast<double>(io.MousePos.x),
                    static_cast<double>(io.MousePos.y)},
                std::exp2(-0.25 * static_cast<double>(io.MouseWheel))),
            false);
    }
    const bool dragging = canvas_active &&
        (ImGui::IsMouseDown(ImGuiMouseButton_Left) ||
         ImGui::IsMouseDown(ImGuiMouseButton_Middle));
    if (dragging &&
        (io.MouseDelta.x != 0.0F || io.MouseDelta.y != 0.0F)) {
        update_camera(
            satview::viewer::pan_by_screen_delta(
                navigation.camera,
                raster,
                screen,
                static_cast<double>(io.MouseDelta.x),
                static_cast<double>(io.MouseDelta.y)),
            false);
    }
    if (canvas_hovered || canvas_active) {
        ImGui::SetMouseCursor(
            dragging ? ImGuiMouseCursor_ResizeAll : ImGuiMouseCursor_Hand);
    }

    if (result.camera_changed) {
        // The camera is authoritative. Chunk controls mirror the native chunk
        // containing its center; this derived update is only a scheduler hint
        // and deliberately does not submit a read from build_ui().
        tile_row = focus_chunk(
            navigation.camera.center_row,
            raster.rows,
            layer.tile_height,
            layer.tile_rows);
        tile_column = focus_chunk(
            navigation.camera.center_column,
            raster.columns,
            layer.tile_width,
            layer.tile_columns);
    }

    result.visible = satview::viewer::visible_raster_window(
        navigation.camera, raster, screen);

    display.samples = {};
    display.sample_weights = {0.0F, 0.0F};
    display.active_slots = 0;
    if (resident.has_value() && resident->layer_index == selected_layer) {
        if (const auto draw = resident_draw_view(
                *resident,
                navigation.camera,
                raster,
                screen,
                result.visible)) {
            result.scientific_window = draw->window;
            result.scientific_rect = draw->rect;
            display.samples[0] = draw->sample;
            display.window_uv_origin = draw->window_uv_origin;
            display.window_uv_dx = draw->window_uv_dx;
            display.window_uv_dy = draw->window_uv_dy;
            result.resident_visible = true;
        }
    }

    auto* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRect(
        canvas_min,
        canvas_max,
        IM_COL32(50, 61, 78, 180),
        0.0F,
        0,
        1.0F * viewport->DpiScale);
    const auto scene_screen_point = [&](const double row, const double column) {
        const auto point = satview::viewer::raster_to_screen(
            navigation.camera,
            raster,
            screen,
            satview::viewer::RasterPoint{row, column});
        return ImVec2{
            static_cast<float>(point.x),
            static_cast<float>(point.y)};
    };
    const std::array scene_outline{
        scene_screen_point(0.0, 0.0),
        scene_screen_point(0.0, static_cast<double>(raster.columns)),
        scene_screen_point(
            static_cast<double>(raster.rows),
            static_cast<double>(raster.columns)),
        scene_screen_point(static_cast<double>(raster.rows), 0.0)};
    draw_list->AddPolyline(
        scene_outline.data(),
        static_cast<int>(scene_outline.size()),
        IM_COL32(86, 104, 132, 220),
        ImDrawFlags_Closed,
        1.0F * viewport->DpiScale);
    if (!result.resident_visible || !image_valid ||
        mode_control_changed || speckle_control_changed) {
        const char* waiting = result.resident_visible
            ? "Updating visible scene..."
            : "Preparing visible scene...";
        const ImVec2 text = ImGui::CalcTextSize(waiting);
        draw_list->AddText(
            {canvas.x + 0.5F * (canvas.width - text.x),
             canvas.y + 0.5F * (canvas.height - text.y)},
            IM_COL32(180, 190, 208, 255),
            waiting);
    }
    ImGui::End();

    return result;
}

[[nodiscard]] bool same_render_request(
    const TileRequest& left,
    const TileRequest& right) noexcept {
    return same_tile_source(left, right) && left.mode == right.mode &&
        left.speckle == right.speckle;
}

[[nodiscard]] bool request_matches_controls(
    const TileRequest& request,
    const std::size_t layer_index,
    const DisplayMode mode,
    const SpeckleSettings& speckle) noexcept {
    return request.layer_index == layer_index && request.mode == mode &&
        request.speckle == speckle;
}

[[nodiscard]] bool same_overview_window(
    const std::optional<satview::Window2D>& left,
    const std::optional<satview::Window2D>& right) noexcept {
    if (left.has_value() != right.has_value()) {
        return false;
    }
    return !left.has_value() ||
        (left->row == right->row &&
         left->column == right->column &&
         left->height == right->height &&
         left->width == right->width);
}

[[nodiscard]] bool same_overview_request(
    const satview::overview::OverviewRequest& left,
    const satview::overview::OverviewRequest& right) {
    return left.science_dataset_path == right.science_dataset_path &&
        left.mask_dataset_path == right.mask_dataset_path &&
        left.maximum_long_edge == right.maximum_long_edge &&
        left.maximum_output_bytes == right.maximum_output_bytes &&
        left.maximum_scratch_bytes == right.maximum_scratch_bytes &&
        same_overview_window(left.source_window, right.source_window) &&
        left.sample_stride == right.sample_stride;
}

[[nodiscard]] std::string describe_lod_page(
    const satview::overview::OverviewPlan& plan,
    const std::string_view page_state,
    const double density) {
    const auto& layout = plan.layout;
    std::ostringstream description;
    description
        << "LOD page origin " << layout.source_origin[0] << ','
        << layout.source_origin[1] << ", coverage "
        << layout.source_dimensions[0] << 'x'
        << layout.source_dimensions[1] << ", stride "
        << layout.sample_stride[0] << 'x'
        << layout.sample_stride[1] << ", output "
        << layout.output_dimensions[0] << 'x'
        << layout.output_dimensions[1] << ", " << page_state
        << ", density " << density << " texels/framebuffer pixel";
    return description.str();
}

[[nodiscard]] bool mosaic_contains(
    const satview::viewer::MosaicGeometry& mosaic,
    const satview::viewer::RasterWindow& visible) noexcept {
    const double row_begin = static_cast<double>(mosaic.pixel_row);
    const double column_begin = static_cast<double>(mosaic.pixel_column);
    const double row_end = row_begin +
        static_cast<double>(mosaic.pixel_height);
    const double column_end = column_begin +
        static_cast<double>(mosaic.pixel_width);
    return visible.row_begin >= row_begin &&
        visible.column_begin >= column_begin &&
        visible.row_end <= row_end &&
        visible.column_end <= column_end;
}

[[nodiscard]] bool overview_request_remains_usable(
    const satview::overview::OverviewRequest& request,
    const satview::viewer::RasterWindow& visible,
    const satview::viewer::Camera2D& camera,
    const satview::viewer::RasterMetrics& raster,
    const double framebuffer_scale_x,
    const double framebuffer_scale_y,
    const double minimum_density) {
    if (!request.source_window.has_value() ||
        !request.sample_stride.has_value()) {
        return false;
    }
    const auto& source = *request.source_window;
    return satview::viewer::sampled_window_remains_usable(
               {source.row, source.column, source.height, source.width},
               (*request.sample_stride)[0],
               (*request.sample_stride)[1],
               visible,
               camera,
               raster,
               framebuffer_scale_x,
               framebuffer_scale_y,
               minimum_density);
}

[[nodiscard]] bool overview_plan_remains_usable(
    const satview::overview::OverviewPlan& plan,
    const satview::viewer::RasterWindow& visible,
    const satview::viewer::Camera2D& camera,
    const satview::viewer::RasterMetrics& raster,
    const double framebuffer_scale_x,
    const double framebuffer_scale_y,
    const double minimum_density) {
    return satview::viewer::sampled_window_remains_usable(
               {plan.layout.source_origin[0],
                plan.layout.source_origin[1],
                plan.layout.source_dimensions[0],
                plan.layout.source_dimensions[1]},
               plan.layout.sample_stride[0],
               plan.layout.sample_stride[1],
               visible,
               camera,
               raster,
               framebuffer_scale_x,
               framebuffer_scale_y,
               minimum_density);
}

[[nodiscard]] bool resident_contains(
    const ResidentViewMapping& resident,
    const satview::viewer::RasterWindow& visible) noexcept {
    const double row_begin =
        static_cast<double>(resident.actual_scene.row);
    const double column_begin =
        static_cast<double>(resident.actual_scene.column);
    const double row_end = row_begin +
        static_cast<double>(resident.actual_scene.height);
    const double column_end = column_begin +
        static_cast<double>(resident.actual_scene.width);
    return visible.row_begin >= row_begin &&
        visible.column_begin >= column_begin &&
        visible.row_end <= row_end &&
        visible.column_end <= column_end;
}

[[nodiscard]] ResidentViewMapping native_mapping(
    const TileRequest& request) {
    if (request.source_kind != TileSourceKind::native_mosaic ||
        request.mosaic.pixel_width == 0 ||
        request.mosaic.pixel_height == 0 ||
        request.mosaic.pixel_width >
            std::numeric_limits<std::uint32_t>::max() ||
        request.mosaic.pixel_height >
            std::numeric_limits<std::uint32_t>::max()) {
        fail("invalid native resident mapping");
    }
    return ResidentViewMapping{
        .layer_index = request.layer_index,
        .actual_scene = satview::viewer::PixelWindow{
            request.mosaic.pixel_row,
            request.mosaic.pixel_column,
            request.mosaic.pixel_height,
            request.mosaic.pixel_width},
        .texture_origin_row =
            static_cast<double>(request.mosaic.pixel_row),
        .texture_origin_column =
            static_cast<double>(request.mosaic.pixel_column),
        .sample_stride_row = 1,
        .sample_stride_column = 1,
        .texture_width = static_cast<std::uint32_t>(
            request.mosaic.pixel_width),
        .texture_height = static_cast<std::uint32_t>(
            request.mosaic.pixel_height),
    };
}

[[nodiscard]] ResidentViewMapping overview_mapping(
    const satview::viewer::OverviewWorkerCompletion& completion) {
    if (!completion.ready() || !completion.plan.has_value()) {
        fail("overview mapping requires prepared data");
    }
    const auto& layout = completion.plan->layout;
    if (layout.source_dimensions[0] == 0 ||
        layout.source_dimensions[1] == 0 ||
        layout.output_dimensions[0] == 0 ||
        layout.output_dimensions[1] == 0 ||
        layout.sample_stride[0] == 0 ||
        layout.sample_stride[1] == 0 ||
        layout.output_dimensions[0] >
            std::numeric_limits<std::uint32_t>::max() ||
        layout.output_dimensions[1] >
            std::numeric_limits<std::uint32_t>::max()) {
        fail("prepared overview has invalid mapping dimensions");
    }
    return ResidentViewMapping{
        .layer_index = completion.layer_index,
        .actual_scene = satview::viewer::PixelWindow{
            layout.source_origin[0],
            layout.source_origin[1],
            layout.source_dimensions[0],
            layout.source_dimensions[1]},
        .texture_origin_row =
            static_cast<double>(layout.source_origin[0]) + 0.5 -
            0.5 * static_cast<double>(layout.sample_stride[0]),
        .texture_origin_column =
            static_cast<double>(layout.source_origin[1]) + 0.5 -
            0.5 * static_cast<double>(layout.sample_stride[1]),
        .sample_stride_row = layout.sample_stride[0],
        .sample_stride_column = layout.sample_stride[1],
        .texture_width = static_cast<std::uint32_t>(
            layout.output_dimensions[1]),
        .texture_height = static_cast<std::uint32_t>(
            layout.output_dimensions[0]),
    };
}

[[nodiscard]] std::optional<std::filesystem::path> choose_product_file(
    std::string& error) {
    std::array<wchar_t, 32'768> path{};
    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = GetActiveWindow();
    dialog.lpstrFilter =
        L"NISAR HDF5 products (*.h5)\0*.h5\0All files (*.*)\0*.*\0\0";
    dialog.lpstrFile = path.data();
    dialog.nMaxFile = static_cast<DWORD>(path.size());
    dialog.lpstrDefExt = L"h5";
    dialog.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST |
        OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameW(&dialog) != FALSE) {
        error.clear();
        return std::filesystem::path(path.data());
    }
    const DWORD code = CommDlgExtendedError();
    if (code != 0) {
        std::ostringstream message;
        message << "File picker failed (Windows error 0x"
                << std::hex << code << ')';
        error = message.str();
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::filesystem::path> run_file_launcher(
    const std::optional<std::uint64_t> frame_limit,
    const std::string_view initial_error,
    const ComputeBackend backend) {
    SdlSession sdl;
    VulkanRenderer renderer(
        sdl.window(), 1, 1, backend == ComputeBackend::cuda);
    ImGuiSession imgui(sdl.window(), renderer);
    DisplayPushConstants display;
    std::string error(initial_error);
    std::uint64_t presented_frames = 0;

    while (true) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_KEY_DOWN &&
                event.key.key == SDLK_F11 && !event.key.repeat) {
                sdl.toggle_fullscreen();
            }
            if (event.type == SDL_EVENT_QUIT ||
                (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
                 event.window.windowID == SDL_GetWindowID(sdl.window()))) {
                return std::nullopt;
            }
        }
        if ((SDL_GetWindowFlags(sdl.window()) & SDL_WINDOW_MINIMIZED) != 0) {
            SDL_Delay(10);
            continue;
        }

        renderer.resize_if_needed();
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        constexpr ImGuiWindowFlags flags =
            ImGuiWindowFlags_NoDecoration |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoSavedSettings;
        ImGui::Begin("Open NISAR product", nullptr, flags);

        const ImVec2 available = ImGui::GetContentRegionAvail();
        constexpr float panel_width = 360.0F;
        constexpr float panel_height = 140.0F;
        ImGui::SetCursorPos(ImVec2{
            std::max(0.0F, 0.5F * (available.x - panel_width)),
            std::max(0.0F, 0.5F * (available.y - panel_height))});
        ImGui::BeginChild(
            "file_picker",
            ImVec2{panel_width, panel_height},
            ImGuiChildFlags_Borders);
        ImGui::TextUnformatted("Open a NISAR HDF5 product");
        ImGui::Spacing();
        std::optional<std::filesystem::path> selected;
        if (ImGui::Button("Open HDF5 file...", ImVec2{-1.0F, 0.0F})) {
            selected = choose_product_file(error);
        }
        if (!error.empty()) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.0F, 0.45F, 0.4F, 1.0F});
            ImGui::TextWrapped("%s", error.c_str());
            ImGui::PopStyleColor();
        }
        ImGui::EndChild();
        ImGui::End();

        ImGui::Render();
        const FrameResult frame = renderer.render(
            ImGui::GetDrawData(),
            DisplayRect{},
            display,
            false,
            std::nullopt,
            std::nullopt);
        if (frame.presented) {
            ++presented_frames;
        }
        if (selected.has_value()) {
            return selected;
        }
        if (frame_limit.has_value() && presented_frames >= *frame_limit) {
            return std::nullopt;
        }
    }
}

int run(const Arguments& arguments, const ComputeBackend backend) {
    satview::Hdf5Product product(arguments.file);
    const ProductView product_view = build_product_view(product);
    SdlSession sdl;
    VulkanRenderer renderer(
        sdl.window(),
        product_view.maximum_mosaic_width,
        product_view.maximum_mosaic_height,
        backend == ComputeBackend::cuda);
    ImGuiSession imgui(sdl.window(), renderer);
    std::unique_ptr<TilePipeline> pipeline;
    if (backend == ComputeBackend::cuda) {
#if defined(SATVIEW_HAS_CUDA)
        pipeline = std::make_unique<CudaTilePipeline>(
            product_view.maximum_science_mosaic_bytes, renderer);
#else
        fail("CUDA backend selected by a CUDA-free viewer build");
#endif
    } else {
        pipeline = std::make_unique<HostTilePipeline>(
            backend,
            product_view.maximum_science_mosaic_bytes,
            renderer);
    }
    pipeline->set_layers(product_view);
    TileReader reader(product, product_view, pipeline->page_locked_reads());
    satview::viewer::OverviewWorker overview_worker(product);

    std::size_t selected_layer = product_view.default_layer;
    std::size_t scheduled_layer = selected_layer;
    DisplayMode display_mode = DisplayMode::power_db;
    std::uint64_t tile_row =
        (product_view.layers[selected_layer].tile_rows - 1) / 2;
    std::uint64_t tile_column =
        (product_view.layers[selected_layer].tile_columns - 1) / 2;
    std::uint32_t mosaic_span = arguments.zoom_chunks;
    ViewportNavigationState navigation;
    bool controls_visible = !arguments.clean_view;
    DisplayPushConstants display;
    SpeckleSettings speckle_settings{
        .filter = arguments.speckle_filter,
        .window_size = arguments.speckle_window,
        .equivalent_number_of_looks = arguments.speckle_looks,
    };
    reset_display_for_mode(display_mode, display);

    bool done = false;
    bool request_tile = true;
    bool published_image_valid = false;
    bool overview_failed = false;
    bool cli_camera_applied = false;
    std::uint64_t tile_serial = 0;
    std::uint64_t camera_generation = 0;
    std::uint64_t latest_native_serial = 0;
    std::uint64_t overview_serial = 0;
    std::uint64_t last_vulkan_consumed = 0;
    std::optional<TileRequest> queued_request;
    std::optional<TileRequest> active_native_request;
    std::optional<TileRequest> published_request;
    std::optional<TileUpload> pending_upload;
    std::deque<DistributionCompletion> distribution_cache;
    std::optional<ResidentViewMapping> pending_mapping;
    std::optional<ResidentViewMapping> published_mapping;
    std::optional<std::uint32_t> published_slot;
    std::optional<TileRequest> outgoing_request;
    std::optional<ResidentViewMapping> outgoing_mapping;
    std::optional<std::uint32_t> outgoing_slot;
    std::optional<Clock::time_point> crossfade_started;
    constexpr auto crossfade_duration = std::chrono::milliseconds(120);

    std::optional<ReadCompletion> deferred_completion;
    std::optional<satview::viewer::OverviewWorkerCompletion> prepared_overview;
    std::optional<std::uint64_t> overview_job_serial;
    std::optional<std::size_t> overview_job_layer;
    std::optional<satview::overview::OverviewRequest> overview_job_request;
    std::optional<std::size_t> failed_overview_layer;
    std::optional<satview::overview::OverviewRequest> failed_overview_request;
    std::optional<Clock::time_point> request_deadline;
    double hdf5_ms = 0.0;
    double overview_prepare_ms = 0.0;
    double overview_density = 0.0;
    double framebuffer_scale_x = 1.0;
    double framebuffer_scale_y = 1.0;
    float h2d_ms = 0.0F;
    float transform_ms = 0.0F;
    std::string overview_error;
    std::string status = "Opening scene pipeline";
    std::uint32_t image_width = 1;
    std::uint32_t image_height = 1;
    std::uint64_t presented_frames = 0;
    std::uint64_t smoke_image_frames = 0;
    const auto smoke_deadline = Clock::now() +
        (arguments.fit_scene
             ? std::chrono::minutes(5)
             : std::chrono::minutes(1));

    auto cancel_native = [&] {
        ++tile_serial;
        latest_native_serial = tile_serial;
        reader.supersede(latest_native_serial);
        active_native_request.reset();
        overview_worker.set_foreground_active(false);
    };

    auto expected_mask_path = [&](const std::size_t layer_index)
        -> std::optional<std::string> {
        const auto* mask = product_view.layers.at(layer_index).validity_mask;
        if (mask == nullptr) {
            return std::nullopt;
        }
        return mask->path;
    };

    while (!done) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_KEY_DOWN &&
                event.key.key == SDLK_F11 && !event.key.repeat) {
                sdl.toggle_fullscreen();
            }
            if (event.type == SDL_EVENT_QUIT ||
                (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
                 event.window.windowID == SDL_GetWindowID(sdl.window()))) {
                done = true;
            }
        }
        if (done) {
            break;
        }
        if (arguments.smoke_test && Clock::now() > smoke_deadline) {
            fail("smoke test timed out");
        }
        if ((SDL_GetWindowFlags(sdl.window()) & SDL_WINDOW_MINIMIZED) != 0) {
            SDL_Delay(10);
            continue;
        }

        // Release the second image slot on time even if camera motion has made
        // the primary mapping temporarily invisible. Upload scheduling must
        // never depend on the crossfade draw branch running.
        if (crossfade_started.has_value() &&
            Clock::now() - *crossfade_started >= crossfade_duration) {
            outgoing_request.reset();
            outgoing_mapping.reset();
            outgoing_slot.reset();
            crossfade_started.reset();
        }

        renderer.resize_if_needed();
        static_cast<void>(reader.reclaim_completed());
        static_cast<void>(pipeline->poll_timing(h2d_ms, transform_ms));
        if (auto distribution = pipeline->poll_distribution()) {
            distribution_cache.push_back(std::move(*distribution));
            constexpr std::size_t maximum_cached_distributions = 4;
            while (distribution_cache.size() > maximum_cached_distributions) {
                distribution_cache.pop_front();
            }
        }

        if (auto completion = overview_worker.try_take_completion()) {
            const bool expected = overview_job_serial.has_value() &&
                completion->request_serial == *overview_job_serial &&
                overview_job_layer.has_value() &&
                completion->layer_index == *overview_job_layer &&
                overview_job_request.has_value();
            if (expected) {
                const auto completed_request = *overview_job_request;
                overview_job_serial.reset();
                overview_job_layer.reset();
                overview_job_request.reset();
                overview_prepare_ms = completion->elapsed_milliseconds;
                if (!completion->error.empty()) {
                    overview_failed = true;
                    failed_overview_layer = completion->layer_index;
                    failed_overview_request = completed_request;
                    overview_error = completion->error;
                    status = "LOD page failed: " +
                        completion->error;
                } else if (completion->ready() &&
                           completion->plan.has_value()) {
                    const auto& layer =
                        product_view.layers.at(completion->layer_index);
                    const auto expected_mask =
                        expected_mask_path(completion->layer_index);
                    const bool exact = same_overview_request(
                            completion->plan->request, completed_request) &&
                        completion->plan->request.science_dataset_path ==
                            layer.dataset->path &&
                        completion->plan->request.mask_dataset_path ==
                            expected_mask;
                    if (!exact) {
                        status =
                            "Rejected stale LOD page completion; rescheduling";
                        request_deadline = Clock::now();
                    } else {
                        hdf5_ms = completion->elapsed_milliseconds;
                        const auto& layout = completion->plan->layout;
                        const auto raster = navigation_raster(layer);
                        overview_density =
                            satview::viewer::
                                minimum_resident_texels_per_framebuffer_pixel(
                                    navigation.camera,
                                    raster,
                                    layout.sample_stride[0],
                                    layout.sample_stride[1],
                                    framebuffer_scale_x,
                                    framebuffer_scale_y);
                        prepared_overview = std::move(*completion);
                        overview_failed = false;
                        failed_overview_layer.reset();
                        failed_overview_request.reset();
                        overview_error.clear();
                        request_deadline = Clock::now();
                        status = describe_lod_page(
                            *prepared_overview->plan,
                            "memory only",
                            overview_density);
                    }
                }
            }
        }

        constexpr std::size_t maximum_drain =
            2 * kMaximumMosaicChunks;
        for (std::size_t drained = 0;
             drained < maximum_drain;
             ++drained) {
            if (!deferred_completion.has_value()) {
                deferred_completion = reader.try_take_completion();
            }
            if (!deferred_completion.has_value()) {
                break;
            }
            auto& completion = *deferred_completion;
            if (!completion.error.empty()) {
                if (completion.request.serial == latest_native_serial) {
                    active_native_request.reset();
                    overview_worker.set_foreground_active(false);
                    throw std::runtime_error(
                        "native scene read failed: " + completion.error);
                }
                deferred_completion.reset();
                continue;
            }
            if (completion.request.serial == latest_native_serial &&
                (pending_upload.has_value() || pipeline->timing_pending() ||
                 outgoing_slot.has_value())) {
                break;
            }
            auto ready = reader.try_take_ready_slot();
            if (!ready) {
                break;
            }
            if (completion.request.serial != latest_native_serial) {
                pipeline->discard(std::move(ready));
                deferred_completion.reset();
                continue;
            }

            hdf5_ms += completion.hdf5_milliseconds;
            const auto completed_chunks = completion.chunk_index + 1;
            const auto expected_chunks = completion.chunk_count;
            auto upload = pipeline->upload_chunk(
                std::move(ready),
                completion,
                last_vulkan_consumed);
            const TileRequest completion_request = completion.request;
            deferred_completion.reset();
            if (upload.has_value()) {
                pending_mapping = native_mapping(completion_request);
                pending_upload = std::move(*upload);
                active_native_request.reset();
                overview_worker.set_foreground_active(false);
                image_width = pending_upload->width;
                image_height = pending_upload->height;
                status = "Exact native scene ready; publication pending";
                break;
            }
            std::ostringstream progress;
            progress << "Native assembly " << completed_chunks << '/'
                     << expected_chunks << " chunks";
            status = progress.str();
        }

        if (queued_request.has_value() &&
            !active_native_request.has_value() &&
            !pending_upload.has_value() && !pipeline->timing_pending() &&
            !outgoing_slot.has_value()) {
            const TileRequest request = *queued_request;
            if (request.source_kind == TileSourceKind::raw_overview) {
                if (!prepared_overview.has_value() ||
                    !prepared_overview->ready() ||
                    prepared_overview->request_serial !=
                        request.overview_identity ||
                    prepared_overview->layer_index != request.layer_index ||
                    !prepared_overview->plan.has_value()) {
                    queued_request.reset();
                    request_deadline = Clock::now();
                } else {
                    pending_mapping = overview_mapping(*prepared_overview);
                    if (pipeline->has_resident_source(request)) {
                        pending_upload = pipeline->redispatch(
                            request, last_vulkan_consumed);
                        status =
                            "Resident LOD page reused; processing queued";
                    } else {
                        const auto& plan = *prepared_overview->plan;
                        std::optional<std::span<const std::uint8_t>> mask;
                        if (!prepared_overview->mask_bytes.empty()) {
                            mask = std::span<const std::uint8_t>(
                                reinterpret_cast<const std::uint8_t*>(
                                    prepared_overview->mask_bytes.data()),
                                prepared_overview->mask_bytes.size());
                        }
                        pending_upload = pipeline->upload_overview(
                            request,
                            prepared_overview->science_bytes,
                            mask,
                            static_cast<std::uint32_t>(
                                plan.layout.output_dimensions[1]),
                            static_cast<std::uint32_t>(
                                plan.layout.output_dimensions[0]),
                            plan.layout.science_type.element_size,
                            last_vulkan_consumed);
                        const auto raster = navigation_raster(
                            product_view.layers.at(request.layer_index));
                        overview_density = satview::viewer::
                            minimum_resident_texels_per_framebuffer_pixel(
                                navigation.camera,
                                raster,
                                plan.layout.sample_stride[0],
                                plan.layout.sample_stride[1],
                                framebuffer_scale_x,
                                framebuffer_scale_y);
                        status = describe_lod_page(
                            plan,
                            "memory only, publication queued",
                            overview_density);
                    }
                    image_width = pending_upload->width;
                    image_height = pending_upload->height;
                    queued_request.reset();
                }
            } else if (pipeline->has_resident_source(request)) {
                pending_mapping = native_mapping(request);
                pending_upload = pipeline->redispatch(
                    request, last_vulkan_consumed);
                image_width = pending_upload->width;
                image_height = pending_upload->height;
                active_native_request.reset();
                overview_worker.set_foreground_active(false);
                queued_request.reset();
                status =
                    "Resident native scene reused; processing queued";
            } else {
                reader.request(request);
                active_native_request = request;
                overview_worker.set_foreground_active(true);
                queued_request.reset();
                hdf5_ms = 0.0;
                status = "Streaming exact native HDF5 chunks and masks";
            }
        }

        const auto worker_progress = overview_worker.progress();
        if (worker_progress.active &&
            overview_job_serial.has_value() &&
            worker_progress.request_serial == *overview_job_serial &&
            worker_progress.layer_index == selected_layer &&
            overview_job_request.has_value() &&
            !active_native_request.has_value()) {
            std::ostringstream progress;
            progress << "Preparing LOD page ";
            if (overview_job_request->source_window.has_value()) {
                const auto& window =
                    *overview_job_request->source_window;
                progress << "origin " << window.row << ',' << window.column
                         << ", coverage " << window.height << 'x'
                         << window.width;
            } else {
                progress << "for the complete raster";
            }
            if (overview_job_request->sample_stride.has_value()) {
                progress << ", stride "
                         << (*overview_job_request->sample_stride)[0] << 'x'
                         << (*overview_job_request->sample_stride)[1];
            } else {
                progress << ", physical auto-stride";
            }
            progress << ": "
                     << worker_progress.overview_progress.completed_source_chunks
                     << '/' << worker_progress.overview_progress.total_source_chunks
                     << " chunks (" << static_cast<int>(std::round(
                            100.0 * worker_progress.overview_progress.fraction()))
                     << "%)";
            status = progress.str();
        }

        std::optional<ResidentViewMapping> frame_mapping;
        const TileRequest* frame_request = nullptr;
        if (published_image_valid && published_request.has_value() &&
            published_mapping.has_value() &&
            published_request->layer_index == selected_layer &&
            published_request->mode == display_mode) {
            frame_mapping = published_mapping;
            frame_request = &*published_request;
        } else if (pending_upload.has_value() &&
                   pending_mapping.has_value() &&
                   pending_upload->request.layer_index == selected_layer &&
                   pending_upload->request.mode == display_mode) {
            // Keep a valid old resident on screen while the inactive slot is
            // uploaded. First publication still uses the pending mapping.
            frame_mapping = pending_mapping;
            frame_request = &pending_upload->request;
        }

        const DistributionCompletion* frame_distribution = nullptr;
        if (frame_request != nullptr) {
            const auto match = std::find_if(
                distribution_cache.rbegin(),
                distribution_cache.rend(),
                [&](const DistributionCompletion& candidate) {
                    return same_render_request(
                        candidate.request, *frame_request);
                });
            if (match != distribution_cache.rend()) {
                frame_distribution = &*match;
            }
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        auto ui = build_ui(
            product,
            product_view,
            renderer,
            selected_layer,
            display_mode,
            speckle_settings,
            tile_row,
            tile_column,
            mosaic_span,
            navigation,
            controls_visible,
            frame_mapping,
            frame_distribution,
            display,
            request_tile,
            frame_mapping.has_value(),
            hdf5_ms,
            h2d_ms,
            transform_ms,
            pipeline->name(),
            status,
            reader.ring_state());

        const SpeckleSettings effective_speckle =
            effective_speckle_settings(
                *product_view.layers[selected_layer].dataset,
                display_mode, speckle_settings);
        const auto framebuffer_scale = ImGui::GetIO().DisplayFramebufferScale;
        framebuffer_scale_x =
            std::isfinite(framebuffer_scale.x) && framebuffer_scale.x > 0.0F
            ? static_cast<double>(framebuffer_scale.x)
            : 1.0;
        framebuffer_scale_y =
            std::isfinite(framebuffer_scale.y) && framebuffer_scale.y > 0.0F
            ? static_cast<double>(framebuffer_scale.y)
            : 1.0;

        auto desired_visible = ui.visible;
        bool force_native_footprint = ui.native_footprint_requested;
        if (!cli_camera_applied) {
            const auto raster = navigation_raster(
                product_view.layers[selected_layer]);
            const double rotation_radians =
                arguments.rotation_degrees * (kPi / 180.0);
            if (arguments.fit_scene) {
                navigation.camera = satview::viewer::fit_camera(
                    raster, ui.canvas, rotation_radians);
                force_native_footprint = false;
            } else if (arguments.rotation_degrees != 0.0) {
                navigation.camera.rotation_radians = rotation_radians;
                navigation.camera = satview::viewer::clamp_camera(
                    navigation.camera, raster, ui.canvas);
            }
            if (arguments.fit_scene || arguments.rotation_degrees != 0.0) {
                navigation.initialized = true;
                navigation.layer_index = selected_layer;
                desired_visible = satview::viewer::visible_raster_window(
                    navigation.camera, raster, ui.canvas);
                ui.camera_changed = true;
                ui.immediate_request = true;
            }
            cli_camera_applied = true;
        }

        if (ui.camera_changed) {
            ++camera_generation;
        }

        // A rapid gesture reversal can move the camera back into the outgoing
        // guard while the newly published page no longer covers it. Restore
        // the still-resident outgoing page immediately instead of showing a
        // gap and blocking correction until the fade timer expires.
        const bool published_covers_reversed_view =
            published_image_valid && published_request.has_value() &&
            published_mapping.has_value() &&
            request_matches_controls(
                *published_request, selected_layer, display_mode,
                effective_speckle) &&
            resident_contains(*published_mapping, desired_visible);
        const bool outgoing_covers_reversed_view =
            outgoing_request.has_value() && outgoing_mapping.has_value() &&
            outgoing_slot.has_value() &&
            request_matches_controls(
                *outgoing_request, selected_layer, display_mode,
                effective_speckle) &&
            resident_contains(*outgoing_mapping, desired_visible);
        if (!published_covers_reversed_view &&
            outgoing_covers_reversed_view) {
            published_request = *outgoing_request;
            published_mapping = *outgoing_mapping;
            published_slot = *outgoing_slot;
            outgoing_request.reset();
            outgoing_mapping.reset();
            outgoing_slot.reset();
            crossfade_started.reset();
            status = "LOD transition reversed; prior guard restored";
        }

        // A page can finish immediately before this frame's wheel or drag
        // input. Do not let that stale result replace a fully covering,
        // sharper resident. Same-level pages remain publishable so guarded pan
        // can advance without waiting for an exact request-identity match.
        if (pending_upload.has_value() && pending_mapping.has_value()) {
            const bool pending_matches_controls =
                request_matches_controls(
                    pending_upload->request, selected_layer, display_mode,
                    effective_speckle);
            const bool published_covers_current =
                published_image_valid && published_request.has_value() &&
                published_mapping.has_value() &&
                request_matches_controls(
                    *published_request, selected_layer, display_mode,
                    effective_speckle) &&
                resident_contains(*published_mapping, desired_visible);
            const bool pending_covers_current =
                resident_contains(*pending_mapping, desired_visible);
            const bool pending_is_not_coarser =
                !published_mapping.has_value() ||
                (pending_mapping->sample_stride_row <=
                     published_mapping->sample_stride_row &&
                 pending_mapping->sample_stride_column <=
                     published_mapping->sample_stride_column);
            const bool stale_regression =
                pending_upload->request.camera_generation !=
                    camera_generation &&
                published_covers_current &&
                (!pending_covers_current || !pending_is_not_coarser);
            if (!pending_matches_controls || stale_regression) {
                pending_upload.reset();
                pending_mapping.reset();
                status = "Stale GPU result skipped; current resident retained";
            }
        }

        const auto now = Clock::now();
        if (selected_layer != scheduled_layer) {
            scheduled_layer = selected_layer;
            cancel_native();
            queued_request.reset();
            outgoing_request.reset();
            outgoing_mapping.reset();
            outgoing_slot.reset();
            crossfade_started.reset();

            ++overview_serial;
            overview_worker.supersede(overview_serial);
            overview_job_serial.reset();
            overview_job_layer.reset();
            overview_job_request.reset();
            prepared_overview.reset();
            overview_failed = false;
            failed_overview_layer.reset();
            failed_overview_request.reset();
            overview_error.clear();
            overview_density = 0.0;
            hdf5_ms = 0.0;
            request_deadline = now;
        }

        if (ui.camera_changed || ui.immediate_request || request_tile) {
            if (ui.immediate_request || request_tile) {
                request_deadline = now;
                overview_failed = false;
                failed_overview_layer.reset();
                failed_overview_request.reset();
            } else {
                // Arm once and sample the latest camera when the deadline
                // expires. Continuous wheel/drag input can no longer postpone
                // residency work indefinitely.
                if (!request_deadline.has_value()) {
                    request_deadline =
                        now + std::chrono::milliseconds(33);
                }
            }
        }
        request_tile = false;

        if (request_deadline.has_value() &&
            now >= *request_deadline) {
            const auto& layer = product_view.layers[selected_layer];
            const satview::viewer::ChunkedRasterGeometry geometry{
                .raster_height = layer.dataset->dimensions[0],
                .raster_width = layer.dataset->dimensions[1],
                .total_chunk_rows = layer.tile_rows,
                .total_chunk_columns = layer.tile_columns,
                .chunk_height = layer.tile_height,
                .chunk_width = layer.tile_width,
            };

            std::optional<satview::viewer::MosaicGeometry>
                forced_native;
            if (force_native_footprint) {
                forced_native = satview::viewer::make_mosaic_geometry(
                    tile_row,
                    tile_column,
                    layer.tile_rows,
                    layer.tile_columns,
                    layer.tile_height,
                    layer.tile_width,
                    layer.dataset->dimensions[0],
                    layer.dataset->dimensions[1],
                    mosaic_span);
            }
            const TileRequest* guard = nullptr;
            const auto consider_guard = [&](const TileRequest* request) {
                if (guard == nullptr && request != nullptr &&
                    request->source_kind == TileSourceKind::native_mosaic &&
                    request->layer_index == selected_layer &&
                    mosaic_contains(request->mosaic, desired_visible)) {
                    guard = request;
                }
            };
            if (pending_upload.has_value()) {
                consider_guard(&pending_upload->request);
            }
            if (queued_request.has_value()) {
                consider_guard(&*queued_request);
            }
            if (active_native_request.has_value()) {
                consider_guard(&*active_native_request);
            }
            if (published_request.has_value()) {
                consider_guard(&*published_request);
            }

            std::optional<satview::viewer::ResidentViewSelection> native;
            if (!forced_native.has_value() && guard == nullptr) {
                // Automatic camera motion reserves the full bounded native
                // working set. Explicit chunk footprints bypass this selector.
                native = satview::viewer::select_resident_view(
                    geometry, desired_visible, 4);
            }

            const auto raster = navigation_raster(layer);
            const satview::viewer::GuardedOverviewOptions lod_options{
                .maximum_level = 63,
                .page_extent = 512,
                .maximum_resident_extent = 4096,
                .target_texels_per_logical_pixel =
                    1.25 * std::max(
                        framebuffer_scale_x, framebuffer_scale_y),
                .guard_fraction = 0.25,
            };
            const auto guarded_lod =
                satview::viewer::make_guarded_overview_request(
                    navigation.camera, raster, ui.canvas, lod_options);
            constexpr double lod_reuse_minimum_density = 1.0;

            satview::overview::OverviewRequest desired_overview_request;
            desired_overview_request.science_dataset_path =
                layer.dataset->path;
            desired_overview_request.mask_dataset_path =
                expected_mask_path(selected_layer);
            desired_overview_request.maximum_long_edge = 4096;
            desired_overview_request.maximum_output_bytes =
                satview::overview::kDefaultMaximumOverviewBytes;

            const auto& lod_source = guarded_lod.resident_source_window;
            const bool lod_covers_complete_raster =
                lod_source.row == 0 && lod_source.column == 0 &&
                lod_source.height == raster.rows &&
                lod_source.width == raster.columns;
            if (!lod_covers_complete_raster) {
                desired_overview_request.source_window = satview::Window2D{
                    .row = lod_source.row,
                    .column = lod_source.column,
                    .height = lod_source.height,
                    .width = lod_source.width,
                };
                desired_overview_request.sample_stride =
                    std::array<std::uint64_t, 2>{
                        guarded_lod.visible.sample_stride,
                        guarded_lod.visible.sample_stride};
            }

            const auto prepared_overview_matches = [&] {
                return prepared_overview.has_value() &&
                    prepared_overview->ready() &&
                    prepared_overview->layer_index == selected_layer &&
                    prepared_overview->plan.has_value() &&
                    (same_overview_request(
                         prepared_overview->plan->request,
                         desired_overview_request) ||
                     overview_plan_remains_usable(
                         *prepared_overview->plan,
                         desired_visible,
                         navigation.camera,
                         raster,
                         framebuffer_scale_x,
                         framebuffer_scale_y,
                         lod_reuse_minimum_density));
            };
            const auto active_overview_matches = [&] {
                return overview_job_serial.has_value() &&
                    overview_job_layer == selected_layer &&
                    overview_job_request.has_value() &&
                    (same_overview_request(
                         *overview_job_request,
                         desired_overview_request) ||
                     overview_request_remains_usable(
                         *overview_job_request,
                         desired_visible,
                         navigation.camera,
                         raster,
                         framebuffer_scale_x,
                         framebuffer_scale_y,
                         lod_reuse_minimum_density));
            };
            const auto failed_overview_matches = [&] {
                return overview_failed &&
                    failed_overview_layer == selected_layer &&
                    failed_overview_request.has_value() &&
                    same_overview_request(
                        *failed_overview_request,
                        desired_overview_request);
            };
            const auto cancel_active_overview = [&] {
                if (!overview_job_serial.has_value()) {
                    return;
                }
                ++overview_serial;
                overview_worker.supersede(overview_serial);
                overview_job_serial.reset();
                overview_job_layer.reset();
                overview_job_request.reset();
            };
            const auto ensure_overview_job = [&] {
                if (overview_job_serial.has_value() &&
                    !active_overview_matches()) {
                    cancel_active_overview();
                }
                if (overview_failed && !failed_overview_matches()) {
                    overview_failed = false;
                    failed_overview_layer.reset();
                    failed_overview_request.reset();
                    overview_error.clear();
                }
                if (prepared_overview_matches() ||
                    active_overview_matches() ||
                    failed_overview_matches()) {
                    return false;
                }

                ++overview_serial;
                overview_job_serial = overview_serial;
                overview_job_layer = selected_layer;
                overview_job_request = desired_overview_request;
                overview_worker.request(
                    satview::viewer::OverviewWorkerRequest{
                        .request_serial = overview_serial,
                        .layer_index = selected_layer,
                        .overview_request = desired_overview_request,
                    });
                return true;
            };

            const auto target_is_scheduled =
                [&](const TileRequest& candidate) {
                    // The last future operation wins. A queued correction
                    // follows an unavoidable pending publication; only when
                    // neither exists can an active read or published image
                    // satisfy the target.
                    if (queued_request.has_value()) {
                        return same_render_request(
                            *queued_request, candidate);
                    }
                    if (pending_upload.has_value()) {
                        return same_render_request(
                            pending_upload->request, candidate);
                    }
                    if (active_native_request.has_value()) {
                        return same_render_request(
                            *active_native_request, candidate);
                    }
                    return published_image_valid &&
                        published_request.has_value() &&
                        same_render_request(
                            *published_request, candidate);
                };
            if (forced_native.has_value() || guard != nullptr ||
                native.has_value()) {
                TileRequest candidate;
                candidate.camera_generation = camera_generation;
                candidate.layer_index = selected_layer;
                candidate.mode = display_mode;
                candidate.speckle = effective_speckle;
                candidate.source_kind = TileSourceKind::native_mosaic;
                if (forced_native.has_value()) {
                    candidate.tile_row = tile_row;
                    candidate.tile_column = tile_column;
                    candidate.mosaic_span = mosaic_span;
                    candidate.mosaic = *forced_native;
                } else if (guard != nullptr) {
                    candidate.tile_row = guard->tile_row;
                    candidate.tile_column = guard->tile_column;
                    candidate.mosaic_span = guard->mosaic_span;
                    candidate.mosaic = guard->mosaic;
                } else {
                    candidate.tile_row = native->focus_chunk_row;
                    candidate.tile_column = native->focus_chunk_column;
                    candidate.mosaic_span = native->requested_span;
                    candidate.mosaic = native->mosaic;
                }

                const bool active_source_matches =
                    active_native_request.has_value() &&
                    same_tile_source(*active_native_request, candidate);
                if (active_source_matches) {
                    // Mode/filter settings are CUDA choices, not HDF5 source
                    // identity. Keep the reader serial authoritative until its
                    // source is resident, then apply the latest settings in-order.
                    candidate.serial = active_native_request->serial;
                    const bool correction_is_queued =
                        queued_request.has_value() &&
                        queued_request->source_kind ==
                            TileSourceKind::native_mosaic &&
                        queued_request->serial ==
                            active_native_request->serial &&
                        same_tile_source(
                            *queued_request, *active_native_request);
                    if (same_render_request(
                            candidate, *active_native_request)) {
                        if (correction_is_queued) {
                            queued_request.reset();
                        }
                        status =
                            "Exact native source read retained for current processing";
                    } else if (!correction_is_queued ||
                               !same_render_request(
                                   *queued_request, candidate)) {
                        queued_request = candidate;
                        status =
                            "Native source read retained; latest processing queued";
                    }
                } else if (!target_is_scheduled(candidate)) {
                    ++tile_serial;
                    candidate.serial = tile_serial;
                    latest_native_serial = tile_serial;
                    reader.supersede(latest_native_serial);
                    active_native_request.reset();
                    overview_worker.set_foreground_active(true);
                    queued_request = candidate;
                    status = "Scheduling exact native resident scene";
                }

                const double visible_rows =
                    desired_visible.row_end - desired_visible.row_begin;
                const double visible_columns =
                    desired_visible.column_end - desired_visible.column_begin;
                const bool prefetch_lod = !forced_native.has_value() &&
                    (visible_rows >
                         2.0 * static_cast<double>(layer.tile_height) ||
                     visible_columns >
                         2.0 * static_cast<double>(layer.tile_width));
                if (prefetch_lod) {
                    if (ensure_overview_job()) {
                        status = active_native_request.has_value()
                            ? "Exact native read retained; guarded LOD prefetch queued"
                            : "Exact native resident; prefetching guarded LOD page";
                    }
                } else {
                    cancel_active_overview();
                }
            } else {
                if (active_native_request.has_value() ||
                    (queued_request.has_value() &&
                     queued_request->source_kind ==
                         TileSourceKind::native_mosaic)) {
                    cancel_native();
                    if (queued_request.has_value() &&
                        queued_request->source_kind ==
                            TileSourceKind::native_mosaic) {
                        queued_request.reset();
                    }
                }

                if (!prepared_overview_matches() &&
                    queued_request.has_value() &&
                    queued_request->source_kind == TileSourceKind::raw_overview) {
                    queued_request.reset();
                }
                const bool started_lod_job = ensure_overview_job();
                if (prepared_overview_matches()) {
                    TileRequest candidate;
                    candidate.camera_generation = camera_generation;
                    candidate.layer_index = selected_layer;
                    candidate.tile_row = tile_row;
                    candidate.tile_column = tile_column;
                    candidate.mode = display_mode;
                    candidate.speckle = effective_speckle;
                    candidate.mosaic_span = 1;
                    candidate.source_kind = TileSourceKind::raw_overview;
                    candidate.overview_identity =
                        prepared_overview->request_serial;
                    if (!target_is_scheduled(candidate)) {
                        ++tile_serial;
                        candidate.serial = tile_serial;
                        latest_native_serial = tile_serial;
                        reader.supersede(latest_native_serial);
                        queued_request = candidate;
                        const auto& plan = *prepared_overview->plan;
                        overview_density = satview::viewer::
                            minimum_resident_texels_per_framebuffer_pixel(
                                navigation.camera,
                                raster,
                                plan.layout.sample_stride[0],
                                plan.layout.sample_stride[1],
                                framebuffer_scale_x,
                                framebuffer_scale_y);
                        status = describe_lod_page(
                            plan,
                            "memory only, GPU publication queued",
                            overview_density);
                    }
                } else if (failed_overview_matches()) {
                    status = "LOD page failed; Fit Scene retries: " +
                        overview_error;
                } else if (started_lod_job) {
                    status = lod_covers_complete_raster
                        ? "Preparing full-source physical-density LOD page"
                        : "Preparing guarded regional LOD page";
                }
            }
            request_deadline.reset();
        }

        std::optional<std::uint32_t> upload_destination_slot;
        if (pending_upload.has_value()) {
            if (published_slot.has_value() && *published_slot >= 2) {
                fail("published Vulkan scientific image slot is invalid");
            }
            upload_destination_slot = published_slot.has_value()
                ? std::uint32_t{1} - *published_slot
                : std::uint32_t{0};
        }

        const auto current_raster = navigation_raster(
            product_view.layers[selected_layer]);
        const bool published_draw_compatible =
            published_image_valid && published_request.has_value() &&
            published_mapping.has_value() && published_slot.has_value() &&
            published_request->layer_index == selected_layer &&
            published_request->mode == display_mode;
        const bool pending_draw_compatible =
            pending_upload.has_value() && pending_mapping.has_value() &&
            upload_destination_slot.has_value() &&
            pending_upload->request.layer_index == selected_layer &&
            pending_upload->request.mode == display_mode;
        const bool outgoing_draw_compatible =
            outgoing_request.has_value() && outgoing_mapping.has_value() &&
            outgoing_slot.has_value() &&
            outgoing_request->layer_index == selected_layer &&
            outgoing_request->mode == display_mode;

        std::optional<ResidentDrawView> published_draw;
        std::optional<ResidentDrawView> pending_draw;
        std::optional<ResidentDrawView> outgoing_draw;
        if (published_draw_compatible) {
            published_draw = resident_draw_view(
                *published_mapping,
                navigation.camera,
                current_raster,
                ui.canvas,
                desired_visible);
        }
        if (pending_draw_compatible) {
            pending_draw = resident_draw_view(
                *pending_mapping,
                navigation.camera,
                current_raster,
                ui.canvas,
                desired_visible);
        }
        if (outgoing_draw_compatible) {
            outgoing_draw = resident_draw_view(
                *outgoing_mapping,
                navigation.camera,
                current_raster,
                ui.canvas,
                desired_visible);
        }

        const TileRequest* draw_source = nullptr;
        std::optional<std::uint32_t> primary_slot;
        const ResidentDrawView* primary_draw = nullptr;
        bool primary_is_published = false;
        const bool published_covers_current =
            published_draw_compatible &&
            resident_contains(*published_mapping, desired_visible);
        const bool pending_covers_current =
            pending_draw_compatible &&
            resident_contains(*pending_mapping, desired_visible);
        const bool prefer_pending = pending_draw.has_value() &&
            (!published_draw.has_value() ||
             (!published_covers_current && pending_covers_current));
        if (prefer_pending) {
            draw_source = &pending_upload->request;
            primary_slot = upload_destination_slot;
            primary_draw = &*pending_draw;
        } else if (published_draw.has_value()) {
            draw_source = &*published_request;
            primary_slot = published_slot;
            primary_draw = &*published_draw;
            primary_is_published = true;
        } else if (pending_draw.has_value()) {
            draw_source = &pending_upload->request;
            primary_slot = upload_destination_slot;
            primary_draw = &*pending_draw;
        } else if (outgoing_draw.has_value()) {
            draw_source = &*outgoing_request;
            primary_slot = outgoing_slot;
            primary_draw = &*outgoing_draw;
        }

        const bool draw_valid = draw_source != nullptr &&
            primary_slot.has_value() && *primary_slot < 2 &&
            primary_draw != nullptr;
        DisplayRect draw_rect{0.0F, 0.0F, 0.0F, 0.0F};
        ResidentSamplePushConstants primary_sample;
        display.samples = {};
        display.sample_weights = {0.0F, 0.0F};
        display.active_slots = 0;
        ui.resident_visible = draw_valid;
        if (draw_valid) {
            ui.scientific_window = primary_draw->window;
            ui.scientific_rect = primary_draw->rect;
            draw_rect = primary_draw->rect;
            primary_sample = primary_draw->sample;
            display.window_uv_origin = primary_draw->window_uv_origin;
            display.window_uv_dx = primary_draw->window_uv_dx;
            display.window_uv_dy = primary_draw->window_uv_dy;
        }

        const ResidentViewMapping* secondary_mapping = nullptr;
        std::optional<std::uint32_t> secondary_slot;
        float primary_weight = 1.0F;
        float secondary_weight = 0.0F;
        const auto transition_compatible = [](
            const TileRequest& left, const TileRequest& right) {
            return left.layer_index == right.layer_index &&
                left.mode == right.mode;
        };

        if (draw_valid && primary_is_published &&
            !pending_upload.has_value() &&
            outgoing_request.has_value() &&
            outgoing_mapping.has_value() &&
            outgoing_slot.has_value() &&
            crossfade_started.has_value() &&
            published_request.has_value() &&
            transition_compatible(
                *outgoing_request, *published_request)) {
            const double elapsed = std::chrono::duration<double>(
                Clock::now() - *crossfade_started).count();
            const double duration =
                std::chrono::duration<double>(crossfade_duration).count();
            const double linear = std::clamp(
                duration > 0.0 ? elapsed / duration : 1.0, 0.0, 1.0);
            if (linear < 1.0) {
                const double eased = linear * linear * (3.0 - 2.0 * linear);
                secondary_mapping = &*outgoing_mapping;
                secondary_slot = outgoing_slot;
                primary_weight = static_cast<float>(eased);
                secondary_weight = 1.0F - primary_weight;
            } else {
                outgoing_request.reset();
                outgoing_mapping.reset();
                outgoing_slot.reset();
                crossfade_started.reset();
            }
        }

        if (draw_valid) {
            display.samples[*primary_slot] = primary_sample;
            display.sample_weights[*primary_slot] = primary_weight;
            display.active_slots |= std::uint32_t{1} << *primary_slot;
            if (secondary_mapping != nullptr && secondary_slot.has_value() &&
                *secondary_slot < 2 && *secondary_slot != *primary_slot) {
                const auto sample = resident_sample_for_window(
                    *secondary_mapping, ui.scientific_window);
                if (sample.has_value()) {
                    display.samples[*secondary_slot] = *sample;
                    display.sample_weights[*secondary_slot] =
                        secondary_weight;
                    display.active_slots |=
                        std::uint32_t{1} << *secondary_slot;
                }
            }
        } else {
            draw_rect = DisplayRect{0.0F, 0.0F, 0.0F, 0.0F};
        }

        ImGui::Render();
        const FrameResult frame = renderer.render(
            ImGui::GetDrawData(),
            draw_rect,
            display,
            draw_valid,
            pending_upload,
            upload_destination_slot);
        const bool smoke_frame_valid = draw_valid &&
            (!arguments.fit_scene ||
             draw_source->source_kind == TileSourceKind::raw_overview);
        if (frame.upload_submitted) {
            if (!pending_upload.has_value() ||
                !pending_mapping.has_value() ||
                !upload_destination_slot.has_value() ||
                frame.uploaded_slot != *upload_destination_slot ||
                frame.uploaded_slot >= 2) {
                fail("renderer submitted an upload without its slot mapping");
            }
            const bool retain_outgoing = published_image_valid &&
                published_request.has_value() &&
                published_mapping.has_value() &&
                published_slot.has_value() &&
                *published_slot != frame.uploaded_slot &&
                transition_compatible(
                    *published_request, pending_upload->request);
            if (retain_outgoing) {
                outgoing_request = *published_request;
                outgoing_mapping = *published_mapping;
                outgoing_slot = *published_slot;
                crossfade_started = Clock::now();
            } else {
                outgoing_request.reset();
                outgoing_mapping.reset();
                outgoing_slot.reset();
                crossfade_started.reset();
            }

            last_vulkan_consumed =
                pending_upload->vulkan_consumed_value;
            published_request = pending_upload->request;
            published_mapping = pending_mapping;
            published_slot = frame.uploaded_slot;
            published_image_valid = true;
            pending_upload.reset();
            pending_mapping.reset();
            status = request_matches_controls(
                    *published_request, selected_layer, display_mode,
                    effective_speckle)
                ? ""
                : "Obsolete GPU publication completed; scheduling current view";
        }
        if (frame.presented) {
            ++presented_frames;
            if (smoke_frame_valid) {
                ++smoke_image_frames;
            }
        }

        if (arguments.smoke_test) {
            if (smoke_image_frames >= 4) {
                done = true;
            } else if (Clock::now() > smoke_deadline) {
                fail(
                    "smoke test timed out before four valid scene frames "
                    "were presented");
            }
        } else if (arguments.frame_limit.has_value() &&
                   presented_frames >= *arguments.frame_limit) {
            done = true;
        }
    }

    overview_worker.set_foreground_active(false);
    overview_worker.stop();
    reader.stop();
    if (arguments.smoke_test) {
        if (!published_request.has_value() ||
            !published_mapping.has_value()) {
            fail("smoke test ended without a published scene");
        }
        std::cout << "sat-viewer smoke test passed: "
                  << image_width << 'x' << image_height << ' ';
        if (published_request->source_kind ==
            TileSourceKind::raw_overview) {
            const auto& mapping = *published_mapping;
            const auto raster = navigation_raster(
                product_view.layers.at(published_request->layer_index));
            const double density = satview::viewer::
                minimum_resident_texels_per_framebuffer_pixel(
                    navigation.camera,
                    raster,
                    mapping.sample_stride_row,
                    mapping.sample_stride_column,
                    framebuffer_scale_x,
                    framebuffer_scale_y);
            std::cout
                << "native-HDF5 LOD page, origin "
                << mapping.actual_scene.row << ','
                << mapping.actual_scene.column << ", coverage "
                << mapping.actual_scene.height << 'x'
                << mapping.actual_scene.width << ", stride "
                << mapping.sample_stride_row << 'x'
                << mapping.sample_stride_column << ", output "
                << mapping.texture_height << 'x'
                << mapping.texture_width << ", "
                << "memory only"
                << " in " << overview_prepare_ms << " ms, density "
                << density << " texels/framebuffer pixel, ";
        } else {
            const auto& mosaic = published_request->mosaic;
            std::cout << mosaic.chunk_rows << 'x'
                      << mosaic.chunk_columns
                      << " exact native chunk footprint, HDF5 total "
                      << hdf5_ms << " ms, ";
        }
        std::cout << smoke_image_frames
                  << " post-upload frames, upload " << h2d_ms
                  << " ms, " << pipeline->name() << ' '
                  << transform_ms << " ms\n";
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        Arguments arguments = parse_arguments(argc, argv);
        const ComputeBackend backend = select_backend(arguments.backend);
        if (!arguments.file.empty()) {
            return run(arguments, backend);
        }

        std::string open_error;
        while (true) {
            const auto selected = run_file_launcher(
                arguments.frame_limit, open_error, backend);
            if (!selected.has_value()) {
                return 0;
            }
            arguments.file = *selected;
            try {
                return run(arguments, backend);
            } catch (const std::exception& error) {
                open_error = error.what();
                arguments.file.clear();
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "sat-viewer: " << error.what() << '\n';
        return 1;
    }
}


