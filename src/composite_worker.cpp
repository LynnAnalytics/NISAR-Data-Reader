#include "satview/composite_worker.hpp"

#include "satview/composite_scientific.hpp"
#include "satview/cpu/scientific.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace satview::composite {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] std::size_t checked_pixel_count(
    const aligned::AlignedPlan& plan) {
    const auto rows = plan.layout.output_dimensions[0];
    const auto columns = plan.layout.output_dimensions[1];
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    if (rows == 0 || columns == 0 || rows > maximum || columns > maximum ||
        static_cast<std::size_t>(rows) >
            maximum / static_cast<std::size_t>(columns)) {
        throw std::length_error("composite output dimensions overflow");
    }
    return static_cast<std::size_t>(rows) *
        static_cast<std::size_t>(columns);
}

[[nodiscard]] bool valid_mask(
    const aligned::AlignedBlock& block,
    const aligned::AlignedPlan& plan,
    const std::size_t raster,
    const std::size_t sample) {
    const auto mask_index = plan.rasters.at(raster).mask_index;
    if (!mask_index.has_value()) {
        return true;
    }
    const auto bytes = block.masks[*mask_index].samples;
    const auto value = std::to_integer<std::uint8_t>(bytes[sample]);
    return value != 0 && value != 255;
}

[[nodiscard]] float read_float(
    const std::span<const std::byte> bytes,
    const std::size_t index) noexcept {
    float value = 0.0F;
    std::memcpy(&value, bytes.data() + index * sizeof(float), sizeof(value));
    return value;
}

[[nodiscard]] cpu::Complex32 read_complex(
    const std::span<const std::byte> bytes,
    const std::size_t index) noexcept {
    cpu::Complex32 value;
    std::memcpy(
        &value,
        bytes.data() + index * sizeof(cpu::Complex32),
        sizeof(value));
    return value;
}

[[nodiscard]] float canonical_nan() noexcept {
    return std::bit_cast<float>(std::uint32_t{0x7fc00000U});
}

[[nodiscard]] float finite_or_nan(const float value) noexcept {
    return std::isfinite(value) ? value : canonical_nan();
}

[[nodiscard]] float robust_magnitude(const cpu::Complex32 value) noexcept {
    const float power = std::fma(
        value.real, value.real, value.imaginary * value.imaginary);
    if (std::isfinite(power) && power > 0.0F) {
        return std::sqrt(power);
    }
    if (value.real == 0.0F && value.imaginary == 0.0F) {
        return 0.0F;
    }
    return std::hypot(value.real, value.imaginary);
}

[[nodiscard]] float transform_sample(
    const std::span<const std::byte> bytes,
    const DataTypeInfo& type,
    const std::size_t index,
    const SourceTransform transform,
    const bool mask_valid) noexcept {
    if (!mask_valid) {
        return canonical_nan();
    }
    if (type.kind == ScalarKind::floating_point &&
        type.element_size == sizeof(float)) {
        const float value = read_float(bytes, index);
        if (!std::isfinite(value) || value < 0.0F) {
            return canonical_nan();
        }
        if (transform == SourceTransform::power_db) {
            return 10.0F * std::log10(
                std::max(value, cpu::kTransformEpsilon));
        }
        return value;
    }
    if (type.kind != ScalarKind::compound_complex ||
        type.element_size != sizeof(cpu::Complex32)) {
        return canonical_nan();
    }
    const auto value = read_complex(bytes, index);
    if (!std::isfinite(value.real) || !std::isfinite(value.imaginary)) {
        return canonical_nan();
    }
    const float power = std::fma(
        value.real, value.real, value.imaginary * value.imaginary);
    float result = canonical_nan();
    switch (transform) {
        case SourceTransform::amplitude:
        case SourceTransform::magnitude:
            result = robust_magnitude(value);
            break;
        case SourceTransform::power:
        case SourceTransform::linear:
            result = power;
            break;
        case SourceTransform::power_db:
            if (std::isfinite(power)) {
                result = 10.0F * std::log10(
                    std::max(power, cpu::kTransformEpsilon));
            } else {
                const float high = std::max(
                    std::abs(value.real), std::abs(value.imaginary));
                const float low = std::min(
                    std::abs(value.real), std::abs(value.imaginary));
                const float ratio = low / high;
                result = std::max(
                    20.0F * std::log10(high) +
                        10.0F * std::log10(
                            std::fma(ratio, ratio, 1.0F)),
                    10.0F * std::log10(cpu::kTransformEpsilon));
            }
            break;
        case SourceTransform::phase:
            result = std::atan2(value.imaginary, value.real);
            break;
        case SourceTransform::real:
            result = value.real;
            break;
        case SourceTransform::imaginary:
            result = value.imaginary;
            break;
    }
    return finite_or_nan(result);
}

[[nodiscard]] analysis::CompareMode compare_mode_for(
    const PageKind kind) {
    switch (kind) {
        case PageKind::compare_pair:
            // The packed pair shares one sampling lattice; independent-grid
            // side-by-side rendering is deliberately outside this worker.
            return analysis::CompareMode::swipe;
        case PageKind::compare_difference:
            return analysis::CompareMode::difference;
        case PageKind::compare_ratio_db:
            return analysis::CompareMode::ratio;
        case PageKind::pauli_rgb:
            break;
    }
    throw std::invalid_argument("Pauli pages do not have a compare mode");
}

[[nodiscard]] PageRequest canonical_request(
    const Hdf5Product& product,
    const PageRequest& request) {
    auto canonical = request;
    if (request.kind == PageKind::pauli_rgb) {
        if (!request.pauli.has_value() || request.compare.has_value()) {
            throw std::invalid_argument(
                "Pauli pages require exactly one Pauli recipe");
        }
        const auto capabilities =
            analysis::resolve_pauli_capabilities(product);
        const auto match = std::find_if(
            capabilities.begin(), capabilities.end(),
            [&](const analysis::PauliCapability& capability) {
                return capability.available && capability.recipe.has_value() &&
                    *capability.recipe == *request.pauli;
            });
        if (match == capabilities.end()) {
            throw std::invalid_argument(
                "Pauli recipe does not match an available canonical recipe");
        }
        canonical.pauli = *match->recipe;
        return canonical;
    }

    if (!request.compare.has_value() || request.pauli.has_value()) {
        throw std::invalid_argument(
            "comparison pages require exactly one comparison recipe");
    }
    const auto capability = analysis::resolve_compare_capability(
        product,
        request.compare->first_dataset_path,
        request.compare->second_dataset_path);
    const auto mode = compare_mode_for(request.kind);
    if (!capability.supports(mode) || !capability.recipe.has_value() ||
        !capability.recipe->strictly_aligned ||
        *capability.recipe != *request.compare) {
        throw std::invalid_argument(
            "comparison recipe is not a supported canonical aligned pair");
    }
    const auto first = product.find_dataset(
        capability.recipe->first_dataset_path);
    const auto second = product.find_dataset(
        capability.recipe->second_dataset_path);
    if (first == nullptr || second == nullptr) {
        throw std::invalid_argument("comparison source left the product catalog");
    }
    const bool first_complex =
        first->data_type.kind == ScalarKind::compound_complex;
    const bool second_complex =
        second->data_type.kind == ScalarKind::compound_complex;
    if (first_complex != second_complex) {
        throw std::invalid_argument(
            "comparison requires sources with the same scalar/complex type");
    }
    if (!first_complex &&
        request.transform != SourceTransform::linear &&
        request.transform != SourceTransform::power_db) {
        throw std::invalid_argument(
            "real covariance comparison does not support that transform");
    }
    canonical.compare = *capability.recipe;
    return canonical;
}

[[nodiscard]] aligned::AlignedRequest aligned_request_for(
    const PageRequest& request) {
    aligned::AlignedRequest aligned_request;
    aligned_request.source_window = request.source_window;
    aligned_request.sample_stride = request.sample_stride;
    aligned_request.maximum_output_bytes = request.maximum_output_bytes;
    aligned_request.maximum_scratch_bytes = request.maximum_scratch_bytes;
    aligned_request.prefer_direct_chunk_decode =
        request.prefer_direct_chunk_decode;
    if (request.kind == PageKind::pauli_rgb) {
        if (!request.pauli.has_value()) {
            throw std::invalid_argument("Pauli page is missing its recipe");
        }
        const auto& recipe = *request.pauli;
        for (const auto& path : {
                 recipe.hhhh_dataset_path,
                 recipe.hvhv_dataset_path,
                 recipe.vvvv_dataset_path,
                 recipe.hhvv_dataset_path}) {
            aligned_request.rasters.push_back({
                .dataset_path = path,
                .mask_dataset_path = recipe.shared_mask_dataset_path,
            });
        }
    } else {
        if (!request.compare.has_value() ||
            !request.compare->strictly_aligned) {
            throw std::invalid_argument(
                "composite comparison requires an aligned recipe");
        }
        aligned_request.rasters = {
            {.dataset_path = request.compare->first_dataset_path,
             .mask_dataset_path =
                 request.compare->shared_mask_dataset_path},
            {.dataset_path = request.compare->second_dataset_path,
             .mask_dataset_path =
                 request.compare->shared_mask_dataset_path},
        };
    }
    return aligned_request;
}

}  // namespace

bool same_page_source(
    const PageRequest& left,
    const PageRequest& right) {
    const auto same_window = left.source_window.row == right.source_window.row &&
        left.source_window.column == right.source_window.column &&
        left.source_window.height == right.source_window.height &&
        left.source_window.width == right.source_window.width;
    return left.kind == right.kind && left.transform == right.transform &&
        left.pauli == right.pauli &&
        left.compare == right.compare && same_window &&
        left.sample_stride == right.sample_stride &&
        left.maximum_output_bytes == right.maximum_output_bytes &&
        left.maximum_scratch_bytes == right.maximum_scratch_bytes &&
        left.prefer_direct_chunk_decode == right.prefer_direct_chunk_decode;
}

PagePlan make_page_plan(
    const Hdf5Product& product,
    const PageRequest& request) {
    if (request.serial == 0 || request.source_window.height == 0 ||
        request.source_window.width == 0 || request.sample_stride[0] == 0 ||
        request.sample_stride[1] == 0) {
        throw std::invalid_argument("composite page request is invalid");
    }
    auto canonical = canonical_request(product, request);
    auto aligned_plan = aligned::make_aligned_plan(
        product, aligned_request_for(canonical));
    const auto pixels = checked_pixel_count(aligned_plan);
    if (pixels > request.maximum_output_bytes / sizeof(float)) {
        throw std::length_error(
            "composite page exceeds its output byte budget");
    }
    return PagePlan{std::move(canonical), std::move(aligned_plan)};
}

PageCompletion build_page(
    const Hdf5Product& product,
    const PagePlan& plan,
    aligned::CancelCallback cancel,
    aligned::ProgressCallback progress) {
    PageCompletion completion{.serial = plan.request.serial, .plan = plan};
    const auto started = Clock::now();
    const auto rebuilt = make_page_plan(product, plan.request);
    if (rebuilt.request.serial != plan.request.serial ||
        !same_page_source(rebuilt.request, plan.request) ||
        rebuilt.aligned_plan.validation_metadata !=
            plan.aligned_plan.validation_metadata ||
        !(rebuilt.aligned_plan.layout == plan.aligned_plan.layout)) {
        throw std::runtime_error("composite page plan changed before build");
    }
    const auto pixels = checked_pixel_count(plan.aligned_plan);
    std::vector<float> output(pixels, canonical_nan());
    const auto columns = plan.aligned_plan.layout.output_dimensions[1];
    const auto status = aligned::visit_aligned_blocks(
        product,
        plan.aligned_plan,
        [&](const aligned::AlignedBlock& block) {
            const auto block_columns = block.dimensions[1];
            const auto block_pixels = static_cast<std::size_t>(
                block.dimensions[0] * block.dimensions[1]);
            for (std::size_t sample = 0; sample < block_pixels; ++sample) {
                const auto local_row = sample /
                    static_cast<std::size_t>(block_columns);
                const auto local_column = sample %
                    static_cast<std::size_t>(block_columns);
                const auto destination = static_cast<std::size_t>(
                    (block.output_origin[0] + local_row) * columns +
                    block.output_origin[1] + local_column);
                if (plan.request.kind == PageKind::pauli_rgb) {
                    bool valid = true;
                    for (std::size_t raster = 0; raster < 4; ++raster) {
                        valid = valid && valid_mask(
                            block, plan.aligned_plan, raster, sample);
                    }
                    const float hhhh = read_float(
                        block.rasters[0].samples, sample);
                    const float hvhv = read_float(
                        block.rasters[1].samples, sample);
                    const float vvvv = read_float(
                        block.rasters[2].samples, sample);
                    const auto hhvv = read_complex(
                        block.rasters[3].samples, sample);
                    const auto powers = valid
                        ? compute_pauli_powers(
                              hhhh, hvhv, vvvv, hhvv.real, hhvv.imaginary)
                        : PauliPowers{};
                    output[destination] = pack_pauli_rgb_r32(powers);
                    continue;
                }

                const bool first_valid = valid_mask(
                    block, plan.aligned_plan, 0, sample);
                const bool second_valid = valid_mask(
                    block, plan.aligned_plan, 1, sample);
                const auto first = transform_sample(
                    block.rasters[0].samples,
                    plan.aligned_plan.rasters[0].data_type,
                    sample,
                    plan.request.kind == PageKind::compare_ratio_db
                        ? SourceTransform::linear
                        : plan.request.transform,
                    first_valid);
                const auto second = transform_sample(
                    block.rasters[1].samples,
                    plan.aligned_plan.rasters[1].data_type,
                    sample,
                    plan.request.kind == PageKind::compare_ratio_db
                        ? SourceTransform::linear
                        : plan.request.transform,
                    second_valid);
                if (plan.request.kind == PageKind::compare_pair) {
                    output[destination] = pack_compare_pair_r32(compare_pair(
                        first, std::isfinite(first),
                        second, std::isfinite(second)));
                } else if (plan.request.kind == PageKind::compare_difference) {
                    output[destination] = compare_difference(
                        first, std::isfinite(first),
                        second, std::isfinite(second)).value;
                } else {
                    const auto ratio = compare_ratio(
                        first, std::isfinite(first),
                        second, std::isfinite(second),
                        cpu::kTransformEpsilon);
                    output[destination] = ratio.valid
                        ? 10.0F * std::log10(std::max(
                              ratio.value, cpu::kTransformEpsilon))
                        : canonical_nan();
                }
            }
            return true;
        },
        cancel,
        progress);
    if (status != aligned::VisitStatus::completed ||
        (cancel && cancel())) {
        completion.plan.reset();
        completion.values.clear();
        return completion;
    }
    completion.values = std::move(output);
    completion.elapsed_milliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - started)
            .count();
    return completion;
}

class PageWorker::Impl final {
public:
    explicit Impl(const Hdf5Product& product) : product_(product) {
        thread_ = std::jthread([this](const std::stop_token stop) {
            worker_loop(stop);
        });
    }

    ~Impl() { stop(); }

    void request(PageRequest request) {
        std::lock_guard lock(mutex_);
        if (stopped_ || request.serial == 0 ||
            request.serial <= latest_serial_.load(std::memory_order_acquire)) {
            throw std::invalid_argument(
                "composite request serials must increase monotonically");
        }
        latest_serial_.store(request.serial, std::memory_order_release);
        requested_ = std::move(request);
        completed_.reset();
        progress_ = PageProgress{.serial = requested_->serial, .active = true};
        cv_.notify_one();
    }

    void supersede(const std::uint64_t serial) noexcept {
        std::lock_guard lock(mutex_);
        if (stopped_ ||
            serial <= latest_serial_.load(std::memory_order_acquire)) {
            return;
        }
        latest_serial_.store(serial, std::memory_order_release);
        requested_.reset();
        completed_.reset();
        progress_ = PageProgress{.serial = serial, .active = false};
        cv_.notify_all();
    }

    [[nodiscard]] PageProgress progress() const {
        std::lock_guard lock(mutex_);
        return progress_;
    }

    [[nodiscard]] std::optional<PageCompletion> take() {
        std::lock_guard lock(mutex_);
        return std::exchange(completed_, std::nullopt);
    }

    void stop() noexcept {
        std::lock_guard stop_lock(stop_mutex_);
        {
            std::lock_guard lock(mutex_);
            if (stopped_) {
                return;
            }
            stopped_ = true;
            requested_.reset();
            progress_.active = false;
        }
        if (thread_.joinable()) {
            thread_.request_stop();
            cv_.notify_all();
            thread_.join();
        }
    }

private:
    [[nodiscard]] bool stale(const std::uint64_t serial) const noexcept {
        return latest_serial_.load(std::memory_order_acquire) != serial;
    }

    void worker_loop(const std::stop_token stop) {
        while (!stop.stop_requested()) {
            std::optional<PageRequest> request;
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, stop, [this] { return requested_.has_value(); });
                if (stop.stop_requested()) {
                    return;
                }
                request = std::exchange(requested_, std::nullopt);
            }
            PageCompletion completion{.serial = request->serial};
            try {
                auto plan = make_page_plan(product_, *request);
                completion = build_page(
                    product_,
                    plan,
                    [this, stop, serial = request->serial] {
                        return stop.stop_requested() || stale(serial);
                    },
                    [this, serial = request->serial](
                        const aligned::AlignedProgress& value) {
                        std::lock_guard lock(mutex_);
                        if (!stopped_ && !stale(serial) &&
                            progress_.serial == serial) {
                            progress_.aligned_progress = value;
                            progress_.active = true;
                        }
                    });
            } catch (const std::exception& error) {
                completion.error = error.what();
            } catch (...) {
                completion.error = "unknown composite build failure";
            }
            if (stop.stop_requested() || stale(request->serial) ||
                (!completion.ready() && completion.error.empty())) {
                continue;
            }
            std::lock_guard lock(mutex_);
            if (!stopped_ && !stale(request->serial)) {
                progress_.active = false;
                completed_ = std::move(completion);
            }
        }
    }

    const Hdf5Product& product_;
    mutable std::mutex mutex_;
    std::mutex stop_mutex_;
    std::condition_variable_any cv_;
    std::atomic<std::uint64_t> latest_serial_{0};
    bool stopped_ = false;
    std::optional<PageRequest> requested_;
    std::optional<PageCompletion> completed_;
    PageProgress progress_;
    std::jthread thread_;
};

PageWorker::PageWorker(const Hdf5Product& product)
    : impl_(std::make_unique<Impl>(product)) {}

PageWorker::~PageWorker() { stop(); }

void PageWorker::request(PageRequest request) {
    impl_->request(std::move(request));
}

void PageWorker::supersede(const std::uint64_t serial) noexcept {
    impl_->supersede(serial);
}

PageProgress PageWorker::progress() const { return impl_->progress(); }

std::optional<PageCompletion> PageWorker::try_take_completion() {
    return impl_->take();
}

void PageWorker::stop() noexcept {
    if (impl_) {
        impl_->stop();
    }
}

}  // namespace satview::composite
