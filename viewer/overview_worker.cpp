#include "overview_worker.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

namespace satview::viewer {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] std::size_t checked_product(
    const std::uint64_t left,
    const std::uint64_t right,
    const std::string_view description) {
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    if (left > maximum || right > maximum) {
        throw std::length_error(
            std::string(description) + " exceeds addressable memory");
    }
    const auto left_size = static_cast<std::size_t>(left);
    const auto right_size = static_cast<std::size_t>(right);
    if (left_size != 0 && right_size > maximum / left_size) {
        throw std::length_error(
            std::string(description) + " multiplication overflow");
    }
    return left_size * right_size;
}

void validate_bounded_plan(const overview::OverviewPlan& plan) {
    const auto& request = plan.request;
    const auto& layout = plan.layout;
    const auto rows = layout.output_dimensions[0];
    const auto columns = layout.output_dimensions[1];
    if (rows == 0 || columns == 0 ||
        layout.source_dimensions[0] == 0 ||
        layout.source_dimensions[1] == 0 ||
        layout.sample_stride[0] == 0 ||
        layout.sample_stride[1] == 0 ||
        layout.science_type.element_size == 0 ||
        layout.source_chunk_count == 0) {
        throw std::runtime_error(
            "overview plan contains an empty or invalid layout");
    }
    if (request.maximum_long_edge == 0 ||
        rows > request.maximum_long_edge ||
        columns > request.maximum_long_edge) {
        throw std::length_error(
            "overview plan exceeds its requested long-edge limit");
    }

    const auto pixels =
        checked_product(rows, columns, "overview pixel count");
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    if (layout.science_type.element_size > maximum / pixels) {
        throw std::length_error("overview science byte count overflow");
    }
    const auto expected_science =
        pixels * layout.science_type.element_size;
    if (layout.science_bytes != expected_science) {
        throw std::runtime_error(
            "overview plan science byte count does not match its layout");
    }

    const auto expected_mask =
        request.mask_dataset_path.has_value() ? pixels : 0;
    if (layout.mask_bytes != expected_mask) {
        throw std::runtime_error(
            "overview plan mask byte count does not match its layout");
    }
    if (layout.science_bytes > request.maximum_output_bytes ||
        layout.mask_bytes >
            request.maximum_output_bytes - layout.science_bytes) {
        throw std::length_error(
            "overview plan exceeds its bounded output allocation");
    }
    if (layout.scratch_bytes > request.maximum_scratch_bytes) {
        throw std::length_error(
            "overview plan exceeds its bounded scratch allocation");
    }
}

void validate_ready_data(const overview::OverviewData& data) {
    if (!data.ready()) {
        return;
    }
    validate_bounded_plan(data.plan);
    if (data.science_bytes.size() != data.plan.layout.science_bytes ||
        data.mask_bytes.size() != data.plan.layout.mask_bytes) {
        throw std::runtime_error(
            "prepared overview payload does not match its plan");
    }
}

}  // namespace

class OverviewWorker::Impl final {
public:
    explicit Impl(const Hdf5Product& product)
        : product_(product) {
        thread_ = std::jthread(
            [this](const std::stop_token stop) {
                worker_loop(stop);
            });
    }

    ~Impl() {
        stop();
    }

    void request(OverviewWorkerRequest request) {
        {
            std::lock_guard lock(mutex_);
            if (stopped_) {
                throw std::logic_error(
                    "cannot request an overview after worker shutdown");
            }
            if (request.request_serial == 0 ||
                (last_submitted_serial_.has_value() &&
                 request.request_serial <= *last_submitted_serial_)) {
                throw std::invalid_argument(
                    "overview request serials must increase monotonically");
            }
            last_submitted_serial_ = request.request_serial;
            latest_serial_.store(
                request.request_serial, std::memory_order_release);
            requested_ = std::move(request);
            completed_.reset();
            progress_ = OverviewWorkerProgress{
                .request_serial = requested_->request_serial,
                .layer_index = requested_->layer_index,
                .active = true,
            };
        }
        request_cv_.notify_one();
    }

    void supersede(const std::uint64_t request_serial) noexcept {
        {
            std::lock_guard lock(mutex_);
            if (stopped_) {
                return;
            }
            latest_serial_.store(
                request_serial, std::memory_order_release);
            if (requested_.has_value() &&
                requested_->request_serial != request_serial) {
                requested_.reset();
            }
            if (progress_.request_serial != request_serial) {
                progress_.active = false;
            }
            if (completed_.has_value() &&
                completed_->request_serial != request_serial) {
                completed_.reset();
            }
        }
        request_cv_.notify_all();
    }

    void set_foreground_active(const bool active) noexcept {
        {
            std::lock_guard lock(mutex_);
            if (stopped_) {
                return;
            }
            foreground_active_.store(active, std::memory_order_release);
        }
        request_cv_.notify_all();
    }

    [[nodiscard]] OverviewWorkerProgress progress() const {
        std::lock_guard lock(mutex_);
        return progress_;
    }

    [[nodiscard]] std::optional<OverviewWorkerCompletion>
    try_take_completion() {
        std::lock_guard lock(mutex_);
        return std::exchange(completed_, std::nullopt);
    }

    void stop() noexcept {
        std::lock_guard stop_lock(stop_mutex_);
        bool join = false;
        {
            std::lock_guard lock(mutex_);
            if (stopped_) {
                return;
            }
            stopped_ = true;
            requested_.reset();
            progress_.active = false;
            join = thread_.joinable();
        }
        if (join) {
            thread_.request_stop();
            request_cv_.notify_all();
            thread_.join();
        }
    }

private:
    [[nodiscard]] bool superseded(
        const std::uint64_t request_serial) const noexcept {
        return latest_serial_.load(std::memory_order_acquire) !=
            request_serial;
    }

    [[nodiscard]] bool wait_for_foreground(
        const std::stop_token stop,
        const std::uint64_t request_serial) {
        std::unique_lock lock(mutex_);
        request_cv_.wait(
            lock,
            stop,
            [this, request_serial] {
                return !foreground_active_.load(
                           std::memory_order_acquire) ||
                    superseded(request_serial);
            });
        return !stop.stop_requested() && !superseded(request_serial);
    }

    void publish_progress(
        const OverviewWorkerRequest& request,
        const overview::OverviewProgress& progress) {
        if (superseded(request.request_serial)) {
            return;
        }
        std::lock_guard lock(mutex_);
        if (!stopped_ &&
            progress_.request_serial == request.request_serial) {
            progress_.overview_progress = progress;
            progress_.active = true;
        }
    }

    void publish_completion(OverviewWorkerCompletion completion) {
        if (superseded(completion.request_serial)) {
            return;
        }
        std::lock_guard lock(mutex_);
        if (stopped_ || superseded(completion.request_serial)) {
            return;
        }
        progress_.request_serial = completion.request_serial;
        progress_.layer_index = completion.layer_index;
        progress_.active = false;
        completed_ = std::move(completion);
    }

    void worker_loop(const std::stop_token stop) {
        while (!stop.stop_requested()) {
            std::optional<OverviewWorkerRequest> request;
            {
                std::unique_lock lock(mutex_);
                request_cv_.wait(
                    lock,
                    stop,
                    [this] { return requested_.has_value(); });
                if (stop.stop_requested()) {
                    return;
                }
                request = std::exchange(requested_, std::nullopt);
            }

            if (superseded(request->request_serial) ||
                !wait_for_foreground(stop, request->request_serial)) {
                continue;
            }

            const auto started = Clock::now();
            OverviewWorkerCompletion completion{
                .request_serial = request->request_serial,
                .layer_index = request->layer_index,
            };
            try {
                auto plan = overview::make_overview_plan(
                    product_, request->overview_request);
                validate_bounded_plan(plan);
                completion.plan = plan;
                if (stop.stop_requested() ||
                    superseded(request->request_serial)) {
                    continue;
                }

                auto data = overview::build_overview(
                    product_,
                    plan,
                    [this, stop, serial = request->request_serial] {
                        return stop.stop_requested() ||
                            superseded(serial);
                    },
                    [this, stop, request = *request](
                        const overview::OverviewProgress& progress) {
                        publish_progress(request, progress);
                        static_cast<void>(wait_for_foreground(
                            stop, request.request_serial));
                    });
                if (!data.ready()) {
                    if (stop.stop_requested() ||
                        superseded(request->request_serial)) {
                        continue;
                    }
                    throw std::runtime_error(
                        "overview preparation was cancelled unexpectedly");
                }
                validate_ready_data(data);
                if (data.plan.validation_metadata !=
                        plan.validation_metadata ||
                    data.plan.layout != plan.layout) {
                    throw std::runtime_error(
                        "prepared overview does not match the requested plan");
                }

                completion.status = data.status;
                completion.plan = std::move(data.plan);
                completion.science_bytes =
                    std::move(data.science_bytes);
                completion.mask_bytes = std::move(data.mask_bytes);
            } catch (const std::exception& error) {
                completion.error = error.what();
            } catch (...) {
                completion.error =
                    "unknown exception while preparing overview";
            }

            completion.elapsed_milliseconds =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - started)
                    .count();
            publish_completion(std::move(completion));
        }
    }

    const Hdf5Product& product_;
    mutable std::mutex mutex_;
    std::mutex stop_mutex_;
    std::condition_variable_any request_cv_;
    std::atomic<std::uint64_t> latest_serial_{0};
    std::optional<std::uint64_t> last_submitted_serial_;
    bool stopped_ = false;
    std::atomic<bool> foreground_active_{false};
    std::optional<OverviewWorkerRequest> requested_;
    OverviewWorkerProgress progress_;
    std::optional<OverviewWorkerCompletion> completed_;
    std::jthread thread_;
};

OverviewWorker::OverviewWorker(const Hdf5Product& product)
    : impl_(std::make_unique<Impl>(product)) {}

OverviewWorker::~OverviewWorker() {
    stop();
}

void OverviewWorker::request(OverviewWorkerRequest request) {
    impl_->request(std::move(request));
}

void OverviewWorker::supersede(
    const std::uint64_t request_serial) noexcept {
    impl_->supersede(request_serial);
}

void OverviewWorker::set_foreground_active(const bool active) noexcept {
    impl_->set_foreground_active(active);
}

OverviewWorkerProgress OverviewWorker::progress() const {
    return impl_->progress();
}

std::optional<OverviewWorkerCompletion>
OverviewWorker::try_take_completion() {
    return impl_->try_take_completion();
}

void OverviewWorker::stop() noexcept {
    if (impl_) {
        impl_->stop();
    }
}

}  // namespace satview::viewer
