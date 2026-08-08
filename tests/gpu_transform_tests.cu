#include "satview/gpu/pinned_ring.hpp"
#include "satview/gpu/transforms.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace {

template <typename T>
class DeviceBuffer final {
public:
    explicit DeviceBuffer(const std::size_t count) {
        status_ = cudaMalloc(
            reinterpret_cast<void**>(&data_), count * sizeof(T));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    [[nodiscard]] T* get() const noexcept {
        return data_;
    }

    [[nodiscard]] cudaError_t status() const noexcept {
        return status_;
    }

private:
    T* data_ = nullptr;
    cudaError_t status_ = cudaSuccess;
};

[[nodiscard]] bool nearly_equal(const float actual, const float expected) {
    const float scale = std::max(1.0F, std::abs(expected));
    return std::abs(actual - expected) <= 3.0e-5F * scale;
}

void expect(
    const bool condition,
    const std::string_view description,
    int& failures) {
    if (!condition) {
        std::cerr << "CUDA transform test failed: " << description << '\n';
        ++failures;
    }
}

bool expect_cuda(
    const cudaError_t status,
    const std::string_view operation,
    int& failures) {
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << "CUDA transform test failed in " << operation << ": "
              << cudaGetErrorString(status) << '\n';
    ++failures;
    return false;
}

template <typename Exception, typename Callable>
void expect_throws(
    Callable&& callable,
    const std::string_view description,
    int& failures) {
    bool caught_expected = false;
    try {
        std::forward<Callable>(callable)();
    } catch (const Exception&) {
        caught_expected = true;
    } catch (...) {
    }
    expect(caught_expected, description, failures);
}

void expect_counts(
    const satview::gpu::PinnedRing::StateCounts counts,
    const std::size_t free,
    const std::size_t filling,
    const std::size_t ready,
    const std::size_t in_flight,
    const std::string_view description,
    int& failures) {
    expect(
        counts.free == free && counts.filling == filling &&
            counts.ready == ready && counts.in_flight == in_flight,
        description,
        failures);
}

void CUDART_CB wait_for_gate(void* context) {
    const auto* const open =
        static_cast<const std::atomic<bool>*>(context);
    while (!open->load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

[[nodiscard]] float expected_gslc(
    const float2 sample,
    const satview::gpu::GslcTransform transform,
    const float epsilon) {
    const double power =
        static_cast<double>(sample.x) * sample.x +
        static_cast<double>(sample.y) * sample.y;
    switch (transform) {
        case satview::gpu::GslcTransform::amplitude:
            return static_cast<float>(std::sqrt(power));
        case satview::gpu::GslcTransform::power:
            return static_cast<float>(power);
        case satview::gpu::GslcTransform::power_db:
            return static_cast<float>(10.0 * std::log10(
                std::max(power, static_cast<double>(epsilon))));
        case satview::gpu::GslcTransform::phase:
            return std::atan2(sample.y, sample.x);
        case satview::gpu::GslcTransform::real:
            return sample.x;
        case satview::gpu::GslcTransform::imaginary:
            return sample.y;
    }
    return std::numeric_limits<float>::quiet_NaN();
}

int test_gslc(const cudaStream_t stream) {
    constexpr std::size_t count = 259;
    constexpr float epsilon = 1.0e-12F;
    std::vector<float2> input(count, float2{1.0F, 0.0F});
    std::vector<std::uint8_t> input_mask(count, 1);
    std::vector<float> output(count);
    std::vector<std::uint8_t> output_mask(count);

    input[0] = float2{3.0F, 4.0F};
    input[1] = float2{-3.0F, 4.0F};
    input[2] = float2{2.0F, -1.0F};
    input[3] =
        float2{std::numeric_limits<float>::quiet_NaN(), 1.0F};
    input[4] = float2{1.0F, 2.0F};
    input[5] = float2{0.0F, 0.0F};
    input[6] =
        float2{std::numeric_limits<float>::max(), 0.0F};
    input[258] = float2{5.0F, 12.0F};
    input_mask[1] = 254;
    input_mask[2] = 0;
    input_mask[4] = 255;

    DeviceBuffer<float2> device_input(count);
    DeviceBuffer<std::uint8_t> device_input_mask(count);
    DeviceBuffer<float> device_output(count);
    DeviceBuffer<std::uint8_t> device_output_mask(count);
    int failures = 0;

    if (!expect_cuda(device_input.status(), "cudaMalloc GSLC input", failures) ||
        !expect_cuda(
            device_input_mask.status(), "cudaMalloc GSLC mask", failures) ||
        !expect_cuda(
            device_output.status(), "cudaMalloc GSLC output", failures) ||
        !expect_cuda(
            device_output_mask.status(),
            "cudaMalloc GSLC output mask",
            failures)) {
        return failures;
    }

    if (!expect_cuda(
            cudaMemcpyAsync(
                device_input.get(),
                input.data(),
                count * sizeof(float2),
                cudaMemcpyHostToDevice,
                stream),
            "copy GSLC input",
            failures) ||
        !expect_cuda(
            cudaMemcpyAsync(
                device_input_mask.get(),
                input_mask.data(),
                count * sizeof(std::uint8_t),
                cudaMemcpyHostToDevice,
                stream),
            "copy GSLC mask",
            failures)) {
        return failures;
    }

    const std::array transforms{
        satview::gpu::GslcTransform::amplitude,
        satview::gpu::GslcTransform::power,
        satview::gpu::GslcTransform::power_db,
        satview::gpu::GslcTransform::phase,
        satview::gpu::GslcTransform::real,
        satview::gpu::GslcTransform::imaginary,
    };

    for (const auto transform : transforms) {
        satview::gpu::TransformOptions options;
        options.validity.input = device_input_mask.get();
        options.validity.output = device_output_mask.get();
        options.epsilon = epsilon;
        options.stream = stream;

        if (!expect_cuda(
                satview::gpu::launch_gslc_transform(
                    device_input.get(),
                    device_output.get(),
                    count,
                    transform,
                    options),
                "launch GSLC transform",
                failures)) {
            continue;
        }
        if (!expect_cuda(
                cudaMemcpyAsync(
                    output.data(),
                    device_output.get(),
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream),
                "copy GSLC output",
                failures) ||
            !expect_cuda(
                cudaMemcpyAsync(
                    output_mask.data(),
                    device_output_mask.get(),
                    count * sizeof(std::uint8_t),
                    cudaMemcpyDeviceToHost,
                    stream),
                "copy GSLC output mask",
                failures) ||
            !expect_cuda(
                cudaStreamSynchronize(stream),
                "synchronize GSLC stream",
                failures)) {
            continue;
        }

        for (std::size_t index = 0; index < count; ++index) {
            const bool valid_mask =
                input_mask[index] != 0 && input_mask[index] != 255;
            const bool finite = std::isfinite(input[index].x) &&
                                std::isfinite(input[index].y);
            const float expected =
                expected_gslc(input[index], transform, epsilon);
            const bool expected_valid = valid_mask && finite && std::isfinite(expected);
            expect(
                output_mask[index] ==
                    static_cast<std::uint8_t>(expected_valid),
                "GSLC validity byte",
                failures);
            if (expected_valid) {
                expect(
                    nearly_equal(
                        output[index],
                        expected),
                    "GSLC numerical result",
                    failures);
            } else {
                expect(std::isnan(output[index]), "GSLC invalid NaN", failures);
            }
        }
    }

    satview::gpu::TransformOptions alias_options;
    alias_options.validity.input = device_input_mask.get();
    alias_options.validity.output = device_input_mask.get();
    alias_options.stream = stream;
    expect_cuda(
        satview::gpu::launch_gslc_transform(
            device_input.get(),
            device_output.get(),
            count,
            satview::gpu::GslcTransform::real,
            alias_options),
        "launch GSLC with aliased validity mask",
        failures);
    expect_cuda(
        cudaMemcpyAsync(
            output_mask.data(),
            device_input_mask.get(),
            count * sizeof(std::uint8_t),
            cudaMemcpyDeviceToHost,
            stream),
        "copy aliased GSLC validity mask",
        failures);
    expect_cuda(
        cudaStreamSynchronize(stream),
        "synchronize aliased GSLC validity mask",
        failures);
    for (std::size_t index = 0; index < count; ++index) {
        const bool finite = std::isfinite(input[index].x) &&
                            std::isfinite(input[index].y);
        const bool expected_valid = input_mask[index] != 0 &&
                                    input_mask[index] != 255 && finite;
        expect(
            output_mask[index] ==
                static_cast<std::uint8_t>(expected_valid),
            "GSLC in-place validity mask",
            failures);
    }

    expect(
        satview::gpu::launch_gslc_transform(
            nullptr,
            nullptr,
            0,
            satview::gpu::GslcTransform::amplitude) == cudaSuccess,
        "zero-count GSLC launch",
        failures);

    satview::gpu::TransformOptions bad_options;
    bad_options.epsilon = 0.0F;
    expect(
        satview::gpu::launch_gslc_transform(
            device_input.get(),
            device_output.get(),
            count,
            satview::gpu::GslcTransform::power_db,
            bad_options) == cudaErrorInvalidValue,
        "GSLC rejects invalid epsilon",
        failures);
    return failures;
}

int test_gcov(const cudaStream_t stream) {
    constexpr std::size_t count = 8;
    constexpr float epsilon = 1.0e-12F;
    const std::array<float, count> diagonal{
        25.0F,
        0.0F,
        -1.0F,
        std::numeric_limits<float>::quiet_NaN(),
        100.0F,
        4.0F,
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
    };
    const std::array<float2, count> cross{
        float2{3.0F, 4.0F},
        float2{0.0F, 0.0F},
        float2{1.0F, 0.0F},
        float2{std::numeric_limits<float>::quiet_NaN(), 1.0F},
        float2{1.0F, 1.0F},
        float2{5.0F, 12.0F},
        float2{std::numeric_limits<float>::max(), 0.0F},
        float2{
            std::numeric_limits<float>::max() / 2.0F,
            std::numeric_limits<float>::max() / 2.0F},
    };
    const std::array<std::uint8_t, count> input_mask{
        1, 254, 1, 1, 255, 1, 1, 1};
    const std::array<float, count> diagonal_j{
        4.0F, 0.0F, 4.0F, 1.0F, 1.0F, 169.0F,
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    std::array<float, count> output{};
    std::array<std::uint8_t, count> output_mask{};

    DeviceBuffer<float> device_diagonal(count);
    DeviceBuffer<float> device_diagonal_j(count);
    DeviceBuffer<float2> device_cross(count);
    DeviceBuffer<std::uint8_t> device_input_mask(count);
    DeviceBuffer<float> device_output(count);
    DeviceBuffer<std::uint8_t> device_output_mask(count);
    int failures = 0;

    const std::array allocation_status{
        device_diagonal.status(),
        device_diagonal_j.status(),
        device_cross.status(),
        device_input_mask.status(),
        device_output.status(),
        device_output_mask.status(),
    };
    for (const cudaError_t status : allocation_status) {
        if (!expect_cuda(status, "cudaMalloc GCOV buffer", failures)) {
            return failures;
        }
    }

    expect_cuda(
        cudaMemcpyAsync(
            device_diagonal.get(),
            diagonal.data(),
            sizeof(diagonal),
            cudaMemcpyHostToDevice,
            stream),
        "copy GCOV diagonal",
        failures);
    expect_cuda(
        cudaMemcpyAsync(
            device_diagonal_j.get(),
            diagonal_j.data(),
            sizeof(diagonal_j),
            cudaMemcpyHostToDevice,
            stream),
        "copy GCOV second diagonal",
        failures);
    expect_cuda(
        cudaMemcpyAsync(
            device_cross.get(),
            cross.data(),
            sizeof(cross),
            cudaMemcpyHostToDevice,
            stream),
        "copy GCOV cross term",
        failures);
    expect_cuda(
        cudaMemcpyAsync(
            device_input_mask.get(),
            input_mask.data(),
            sizeof(input_mask),
            cudaMemcpyHostToDevice,
            stream),
        "copy GCOV mask",
        failures);

    satview::gpu::TransformOptions options;
    options.validity.input = device_input_mask.get();
    options.validity.output = device_output_mask.get();
    options.epsilon = epsilon;
    options.stream = stream;

    for (const auto transform :
         {satview::gpu::GcovRealTransform::linear,
          satview::gpu::GcovRealTransform::power_db}) {
        expect_cuda(
            satview::gpu::launch_gcov_real_transform(
                device_diagonal.get(),
                device_output.get(),
                count,
                transform,
                options),
            "launch GCOV real transform",
            failures);
        expect_cuda(
            cudaMemcpyAsync(
                output.data(),
                device_output.get(),
                sizeof(output),
                cudaMemcpyDeviceToHost,
                stream),
            "copy GCOV real output",
            failures);
        expect_cuda(
            cudaMemcpyAsync(
                output_mask.data(),
                device_output_mask.get(),
                sizeof(output_mask),
                cudaMemcpyDeviceToHost,
                stream),
            "copy GCOV real validity",
            failures);
        expect_cuda(
            cudaStreamSynchronize(stream),
            "synchronize GCOV real transform",
            failures);

        for (std::size_t index = 0; index < count; ++index) {
            const bool valid =
                input_mask[index] != 0 && input_mask[index] != 255 &&
                std::isfinite(diagonal[index]) && diagonal[index] >= 0.0F;
            expect(
                output_mask[index] == static_cast<std::uint8_t>(valid),
                "GCOV real validity byte",
                failures);
            if (!valid) {
                expect(
                    std::isnan(output[index]),
                    "GCOV invalid diagonal NaN",
                    failures);
            } else {
                const float expected =
                    transform == satview::gpu::GcovRealTransform::linear
                    ? diagonal[index]
                    : 10.0F *
                        std::log10(std::max(diagonal[index], epsilon));
                expect(
                    nearly_equal(output[index], expected),
                    "GCOV real numerical result",
                    failures);
            }
        }
    }

    for (const auto transform :
         {satview::gpu::GcovComplexTransform::magnitude,
          satview::gpu::GcovComplexTransform::phase}) {
        expect_cuda(
            satview::gpu::launch_gcov_complex_transform(
                device_cross.get(),
                device_output.get(),
                count,
                transform,
                options),
            "launch GCOV complex transform",
            failures);
        expect_cuda(
            cudaMemcpyAsync(
                output.data(),
                device_output.get(),
                sizeof(output),
                cudaMemcpyDeviceToHost,
                stream),
            "copy GCOV complex output",
            failures);
        expect_cuda(
            cudaMemcpyAsync(
                output_mask.data(),
                device_output_mask.get(),
                sizeof(output_mask),
                cudaMemcpyDeviceToHost,
                stream),
            "copy GCOV complex validity",
            failures);
        expect_cuda(
            cudaStreamSynchronize(stream),
            "synchronize GCOV complex transform",
            failures);

        if (transform == satview::gpu::GcovComplexTransform::magnitude) {
            expect(nearly_equal(output[0], 5.0F), "GCOV magnitude", failures);
            expect(
                output_mask[6] == 1 &&
                    output[6] == std::numeric_limits<float>::max(),
                "GCOV magnitude avoids intermediate square overflow",
                failures);
            const double half_max =
                static_cast<double>(std::numeric_limits<float>::max()) / 2.0;
            expect(
                output_mask[7] == 1 &&
                    nearly_equal(
                        output[7],
                        static_cast<float>(std::hypot(half_max, half_max))),
                "GCOV two-component extreme magnitude",
                failures);
        } else {
            expect(
                nearly_equal(output[0], std::atan2(4.0F, 3.0F)),
                "GCOV phase",
                failures);
        }
        expect(
            output_mask[4] == 0 && std::isnan(output[4]),
            "GCOV complex NISAR fill mask",
            failures);
    }

    expect_cuda(
        satview::gpu::launch_gcov_normalized_correlation(
            device_cross.get(),
            device_diagonal.get(),
            device_diagonal_j.get(),
            device_output.get(),
            count,
            options),
        "launch normalized correlation",
        failures);
    expect_cuda(
        cudaMemcpyAsync(
            output.data(),
            device_output.get(),
            sizeof(output),
            cudaMemcpyDeviceToHost,
            stream),
        "copy normalized correlation",
        failures);
    expect_cuda(
        cudaMemcpyAsync(
            output_mask.data(),
            device_output_mask.get(),
            sizeof(output_mask),
            cudaMemcpyDeviceToHost,
            stream),
        "copy correlation validity",
        failures);
    expect_cuda(
        cudaStreamSynchronize(stream),
        "synchronize normalized correlation",
        failures);

    expect(nearly_equal(output[0], 0.5F), "normalized correlation value", failures);
    expect(nearly_equal(output[1], 0.0F), "correlation epsilon floor", failures);
    expect(
        output_mask[2] == 0 && std::isnan(output[2]),
        "negative correlation diagonal invalid",
        failures);
    expect(
        output_mask[4] == 0 && std::isnan(output[4]),
        "correlation NISAR fill invalid",
        failures);
    expect(nearly_equal(output[5], 0.5F), "second correlation value", failures);

    expect(
        output_mask[6] == 1 && nearly_equal(output[6], 1.0F),
        "extreme normalized correlation",
        failures);
    expect(
        output_mask[7] == 1 &&
            nearly_equal(output[7], std::sqrt(0.5F)),
        "two-component extreme normalized correlation",
        failures);
    return failures;
}

int test_pinned_ring(const cudaStream_t stream) {
    using satview::gpu::PinnedRing;

    int failures = 0;
    expect_throws<std::invalid_argument>(
        [] {
            const PinnedRing ring{
                PinnedRing::Options{0, 64, cudaHostAllocDefault}};
            static_cast<void>(ring);
        },
        "PinnedRing rejects zero slots",
        failures);
    expect_throws<std::invalid_argument>(
        [] {
            const PinnedRing ring{
                PinnedRing::Options{1, 0, cudaHostAllocDefault}};
            static_cast<void>(ring);
        },
        "PinnedRing rejects zero-sized slots",
        failures);
    expect_throws<std::length_error>(
        [] {
            const PinnedRing ring{PinnedRing::Options{
                2,
                std::numeric_limits<std::size_t>::max(),
                cudaHostAllocDefault}};
            static_cast<void>(ring);
        },
        "PinnedRing rejects total-size overflow",
        failures);

    {
        PinnedRing ring{PinnedRing::Options{
            .slot_count = 1,
            .bytes_per_slot = 64,
            .page_locked = false,
        }};
        expect(
            ring.total_pinned_bytes() == 0,
            "pageable ring reports no CUDA-pinned bytes",
            failures);
        auto fill = ring.try_acquire_for_fill();
        fill.bytes()[0] = std::byte{0x5A};
        fill.publish_ready(1);
        auto ready = ring.try_acquire_ready();
        expect(
            ready && ready.bytes()[0] == std::byte{0x5A},
            "pageable ring preserves published bytes",
            failures);
        ready.mark_consumed();
        expect_counts(
            ring.state_counts(), 1, 0, 0, 0,
            "synchronous consumption returns a pageable slot to Free",
            failures);
    }

    {
        PinnedRing ring{2, 256};
        expect(
            ring.slot_count() == 2 && ring.bytes_per_slot() == 256 &&
                ring.total_pinned_bytes() == 512,
            "PinnedRing reports fixed allocation dimensions",
            failures);
        expect_counts(
            ring.state_counts(), 2, 0, 0, 0,
            "PinnedRing starts with every slot free", failures);

        auto first = ring.try_acquire_for_fill();
        auto second = ring.try_acquire_for_fill();
        expect(
            static_cast<bool>(first) && static_cast<bool>(second),
            "PinnedRing acquires available fill slots",
            failures);
        expect_counts(
            ring.state_counts(), 0, 2, 0, 0,
            "fill leases transition slots to Filling", failures);

        const std::size_t first_index = first.slot_index();
        second = std::move(first);
        expect(
            !first && second && second.slot_index() == first_index,
            "fill lease move assignment transfers ownership",
            failures);
        expect_counts(
            ring.state_counts(), 1, 1, 0, 0,
            "move assignment rolls the replaced fill lease back to Free",
            failures);
        second.reset();
        expect_counts(
            ring.state_counts(), 2, 0, 0, 0,
            "reset rolls a fill lease back to Free", failures);
    }

    constexpr std::size_t slot_bytes = 256;
    constexpr std::size_t payload_bytes = 37;
    PinnedRing ring{1, slot_bytes};
    auto filling = ring.try_acquire_for_fill();
    expect(
        filling && filling.capacity() == slot_bytes &&
            filling.bytes().size() == slot_bytes &&
            filling.payload_size() == 0,
        "fill lease exposes the entire pinned slot",
        failures);
    expect_counts(
        ring.state_counts(), 0, 1, 0, 0,
        "acquiring the only slot transitions Free to Filling", failures);

    for (std::size_t index = 0; index < payload_bytes; ++index) {
        filling.bytes()[index] =
            static_cast<std::byte>((index * 29U + 7U) & 0xFFU);
    }

    expect_throws<std::length_error>(
        [&filling] { filling.publish_ready(slot_bytes + 1); },
        "PinnedRing rejects a payload larger than the slot",
        failures);
    expect(
        filling && filling.capacity() == slot_bytes,
        "failed publish keeps the fill lease valid",
        failures);
    filling.publish_ready(payload_bytes);
    expect(!filling, "publish consumes the fill lease", failures);
    expect_counts(
        ring.state_counts(), 0, 0, 1, 0,
        "publish transitions Filling to Ready", failures);

    auto ready = ring.try_acquire_ready();
    expect(
        ready && ready.capacity() == slot_bytes &&
            ready.payload_size() == payload_bytes &&
            ready.bytes().size() == payload_bytes,
        "ready lease exposes exactly the published payload",
        failures);
    for (std::size_t index = 0; index < payload_bytes; ++index) {
        expect(
            ready.bytes()[index] ==
                static_cast<std::byte>((index * 29U + 7U) & 0xFFU),
            "ready payload preserves pinned bytes",
            failures);
    }

    auto moved_ready = std::move(ready);
    expect(
        !ready && moved_ready,
        "ready lease move construction transfers ownership",
        failures);
    moved_ready.reset();
    expect_counts(
        ring.state_counts(), 0, 0, 1, 0,
        "reset rolls a ready lease back to the ready queue", failures);
    ready = ring.try_acquire_ready();
    expect(
        ready && ready.payload_size() == payload_bytes,
        "rolled-back ready payload remains available",
        failures);

    DeviceBuffer<std::byte> device_payload(payload_bytes);
    if (!expect_cuda(
            device_payload.status(),
            "cudaMalloc PinnedRing payload",
            failures)) {
        return failures;
    }

    cudaStream_t gate_stream = nullptr;
    cudaEvent_t gate_event = nullptr;
    std::atomic<bool> gate_open{false};
    bool gate_enqueued = false;
    bool stream_wait_enqueued = false;

    bool setup_ok = expect_cuda(
        cudaStreamCreateWithFlags(&gate_stream, cudaStreamNonBlocking),
        "create PinnedRing gate stream",
        failures);
    if (setup_ok) {
        setup_ok = expect_cuda(
            cudaEventCreateWithFlags(
                &gate_event, cudaEventDisableTiming),
            "create PinnedRing gate event",
            failures);
    }
    if (setup_ok) {
        setup_ok = expect_cuda(
            cudaLaunchHostFunc(gate_stream, wait_for_gate, &gate_open),
            "enqueue PinnedRing host gate",
            failures);
        gate_enqueued = setup_ok;
    }
    if (setup_ok) {
        setup_ok = expect_cuda(
            cudaEventRecord(gate_event, gate_stream),
            "record PinnedRing gate event",
            failures);
    }
    if (setup_ok) {
        setup_ok = expect_cuda(
            cudaStreamWaitEvent(stream, gate_event, 0),
            "wait for PinnedRing gate",
            failures);
        stream_wait_enqueued = setup_ok;
    }
    if (setup_ok) {
        setup_ok = expect_cuda(
            cudaMemcpyAsync(
                device_payload.get(),
                ready.bytes().data(),
                payload_bytes,
                cudaMemcpyHostToDevice,
                stream),
            "copy PinnedRing payload",
            failures);
    }

    if (setup_ok) {
        try {
            ready.mark_in_flight(stream);
        } catch (const std::exception& error) {
            std::cerr << "CUDA transform test failed in "
                         "PinnedRing::mark_in_flight: "
                      << error.what() << '\n';
            ++failures;
            setup_ok = false;
        }
    }

    if (setup_ok) {
        expect(!ready, "submission consumes the ready lease", failures);
        expect_counts(
            ring.state_counts(), 0, 0, 0, 1,
            "submission transitions Ready to InFlight", failures);
        expect(
            ring.reclaim_completed() == 0,
            "event reclamation is nonblocking while stream work is pending",
            failures);
        expect(
            !ring.try_acquire_for_fill(),
            "pending in-flight work cannot be reused for filling",
            failures);
    }

    gate_open.store(true, std::memory_order_release);
    if (gate_enqueued) {
        expect_cuda(
            cudaStreamSynchronize(gate_stream),
            "release PinnedRing gate stream",
            failures);
    }
    if (stream_wait_enqueued) {
        expect_cuda(
            cudaStreamSynchronize(stream),
            "complete PinnedRing transfer stream",
            failures);
    }
    if (setup_ok) {
        expect(
            ring.reclaim_completed() == 1,
            "completed stream work is reclaimed by event query",
            failures);
        expect_counts(
            ring.state_counts(), 1, 0, 0, 0,
            "reclamation transitions InFlight to Free", failures);
    } else {
        ready.reset();
    }

    if (gate_event != nullptr) {
        expect_cuda(
            cudaEventDestroy(gate_event),
            "destroy PinnedRing gate event",
            failures);
    }
    if (gate_stream != nullptr) {
        expect_cuda(
            cudaStreamDestroy(gate_stream),
            "destroy PinnedRing gate stream",
            failures);
    }

    return failures;
}
}  // namespace

int run_gpu_transform_tests() {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status == cudaErrorNoDevice ||
        count_status == cudaErrorInsufficientDriver || device_count == 0) {
        (void)cudaGetLastError();
        std::cout << "CUDA transform tests skipped: no CUDA device\n";
        return 0;
    }
    if (count_status != cudaSuccess) {
        std::cerr << "CUDA transform tests could not query the device: "
                  << cudaGetErrorString(count_status) << '\n';
        return 1;
    }

    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
        cudaSuccess) {
        std::cerr << "CUDA transform tests could not create a stream\n";
        return 1;
    }

    const int failures =
        test_gslc(stream) + test_gcov(stream) + test_pinned_ring(stream);
    const cudaError_t destroy_status = cudaStreamDestroy(stream);
    if (destroy_status != cudaSuccess) {
        std::cerr << "CUDA transform test stream destruction failed: "
                  << cudaGetErrorString(destroy_status) << '\n';
        return failures + 1;
    }
    return failures;
}
