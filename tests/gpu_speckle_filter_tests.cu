#include "satview/gpu/speckle_filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace {

using satview::gpu::SpeckleDomain;
using satview::gpu::SpeckleFilter;
using satview::gpu::SpeckleFilterOptions;

static_assert(std::is_trivially_copyable_v<SpeckleFilterOptions>);

struct TestContext {
    int failures = 0;

    void expect(const bool condition, const std::string_view message) {
        if (!condition) {
            ++failures;
            std::cerr << "FAIL: " << message << '\n';
        }
    }

    void expect_cuda(
        const cudaError_t status,
        const std::string_view operation) {
        if (status != cudaSuccess) {
            ++failures;
            std::cerr << "FAIL: " << operation << ": "
                      << cudaGetErrorString(status) << '\n';
        }
    }
};

void require_cuda(
    const cudaError_t status,
    const std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(const std::size_t count) : count_(count) {
        if (count_ != 0) {
            require_cuda(
                cudaMalloc(
                    reinterpret_cast<void**>(&pointer_),
                    count_ * sizeof(T)),
                "cudaMalloc");
        }
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            static_cast<void>(cudaFree(pointer_));
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    [[nodiscard]] T* get() noexcept {
        return pointer_;
    }

    [[nodiscard]] const T* get() const noexcept {
        return pointer_;
    }

    [[nodiscard]] std::size_t bytes() const noexcept {
        return count_ * sizeof(T);
    }

private:
    T* pointer_ = nullptr;
    std::size_t count_ = 0;
};

class Stream {
public:
    Stream() {
        require_cuda(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
    }

    ~Stream() {
        if (stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(stream_));
        }
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept {
        return stream_;
    }

private:
    cudaStream_t stream_ = nullptr;
};

struct FilterResult {
    std::vector<float> values;
    std::vector<std::uint8_t> validity;
};

[[nodiscard]] bool mask_valid(
    const std::vector<std::uint8_t>& validity,
    const std::size_t index) {
    if (validity.empty()) {
        return true;
    }
    return validity[index] != 0 && validity[index] != 255;
}

[[nodiscard]] bool cpu_to_power(
    const float sample,
    const SpeckleDomain domain,
    double& power) {
    if (!std::isfinite(sample)) {
        return false;
    }
    switch (domain) {
        case SpeckleDomain::linear_power:
            if (sample < 0.0F) {
                return false;
            }
            power = sample;
            return true;
        case SpeckleDomain::amplitude:
            if (sample < 0.0F) {
                return false;
            }
            power = static_cast<double>(sample) * sample;
            return std::isfinite(power);
        case SpeckleDomain::power_db:
            power = std::pow(10.0, static_cast<double>(sample) * 0.1);
            return std::isfinite(power);
        case SpeckleDomain::signed_value:
        case SpeckleDomain::phase:
            return false;
    }
    return false;
}

[[nodiscard]] float cpu_from_power(
    const double power,
    const SpeckleFilterOptions options) {
    switch (options.domain) {
        case SpeckleDomain::linear_power:
            return static_cast<float>(power);
        case SpeckleDomain::amplitude:
            return static_cast<float>(std::sqrt(power));
        case SpeckleDomain::power_db:
            return static_cast<float>(
                10.0 * std::log10(std::max(
                           power,
                           static_cast<double>(options.power_epsilon))));
        case SpeckleDomain::signed_value:
        case SpeckleDomain::phase:
            break;
    }
    return std::numeric_limits<float>::quiet_NaN();
}

[[nodiscard]] FilterResult cpu_reference(
    const std::vector<float>& input,
    const std::vector<std::uint8_t>& input_validity,
    const std::size_t width,
    const std::size_t height,
    const SpeckleFilterOptions options) {
    const std::size_t count = width * height;
    FilterResult result{
        std::vector<float>(
            count, std::numeric_limits<float>::quiet_NaN()),
        std::vector<std::uint8_t>(count, 0)};

    if (options.filter == SpeckleFilter::none) {
        for (std::size_t index = 0; index < count; ++index) {
            if (mask_valid(input_validity, index) &&
                std::isfinite(input[index])) {
                result.values[index] = input[index];
                result.validity[index] = 1;
            }
        }
        return result;
    }

    const int radius = static_cast<int>(options.window_size / 2);
    std::vector<double> power(count);
    std::vector<std::uint8_t> power_valid(count, 0);
    for (std::size_t index = 0; index < count; ++index) {
        power_valid[index] = static_cast<std::uint8_t>(
            mask_valid(input_validity, index) &&
            cpu_to_power(input[index], options.domain, power[index]));
    }

    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const std::size_t center_index = y * width + x;
            if (power_valid[center_index] == 0) {
                continue;
            }

            double sum = 0.0;
            double sum_squared = 0.0;
            std::size_t local_count = 0;
            for (int dy = -radius; dy <= radius; ++dy) {
                const std::int64_t local_y =
                    static_cast<std::int64_t>(y) + dy;
                if (local_y < 0 ||
                    static_cast<std::size_t>(local_y) >= height) {
                    continue;
                }
                for (int dx = -radius; dx <= radius; ++dx) {
                    const std::int64_t local_x =
                        static_cast<std::int64_t>(x) + dx;
                    if (local_x < 0 ||
                        static_cast<std::size_t>(local_x) >= width) {
                        continue;
                    }
                    const std::size_t local_index =
                        static_cast<std::size_t>(local_y) * width +
                        static_cast<std::size_t>(local_x);
                    if (power_valid[local_index] != 0) {
                        const double value = power[local_index];
                        sum += value;
                        sum_squared += value * value;
                        ++local_count;
                    }
                }
            }

            const double mean = sum / static_cast<double>(local_count);
            double filtered = mean;
            if (options.filter == SpeckleFilter::lee) {
                const double variance = std::max(
                    sum_squared / static_cast<double>(local_count) -
                        mean * mean,
                    0.0);
                const double noise_variance =
                    (mean * mean) / options.equivalent_number_of_looks;
                double weight = 0.0;
                if (variance > noise_variance && variance > 0.0) {
                    weight = (variance - noise_variance) / variance;
                }
                weight = std::clamp(weight, 0.0, 1.0);
                filtered =
                    mean + weight * (power[center_index] - mean);
                filtered = std::max(filtered, 0.0);
            }

            result.values[center_index] =
                cpu_from_power(filtered, options);
            result.validity[center_index] = 1;
        }
    }
    return result;
}

[[nodiscard]] FilterResult run_gpu(
    const std::vector<float>& input,
    const std::vector<std::uint8_t>& input_validity,
    const std::size_t width,
    const std::size_t height,
    SpeckleFilterOptions options,
    const cudaStream_t stream) {
    const std::size_t count = width * height;
    DeviceBuffer<float> device_input(count);
    DeviceBuffer<float> device_output(count);
    DeviceBuffer<std::uint8_t> device_input_validity(
        input_validity.empty() ? 0 : count);
    DeviceBuffer<std::uint8_t> device_output_validity(count);

    require_cuda(
        cudaMemcpyAsync(
            device_input.get(),
            input.data(),
            device_input.bytes(),
            cudaMemcpyHostToDevice,
            stream),
        "copy filter input");
    if (!input_validity.empty()) {
        require_cuda(
            cudaMemcpyAsync(
                device_input_validity.get(),
                input_validity.data(),
                device_input_validity.bytes(),
                cudaMemcpyHostToDevice,
                stream),
            "copy filter validity");
    }

    options.validity.input =
        input_validity.empty() ? nullptr : device_input_validity.get();
    options.validity.output = device_output_validity.get();
    options.stream = stream;
    require_cuda(
        satview::gpu::launch_speckle_filter(
            device_input.get(),
            device_output.get(),
            width,
            height,
            options),
        "launch speckle filter");

    FilterResult result{
        std::vector<float>(count),
        std::vector<std::uint8_t>(count)};
    require_cuda(
        cudaMemcpyAsync(
            result.values.data(),
            device_output.get(),
            device_output.bytes(),
            cudaMemcpyDeviceToHost,
            stream),
        "copy filter output");
    require_cuda(
        cudaMemcpyAsync(
            result.validity.data(),
            device_output_validity.get(),
            device_output_validity.bytes(),
            cudaMemcpyDeviceToHost,
            stream),
        "copy output validity");

    // This checks deferred execution errors as well as proving the operation
    // participates correctly in non-default-stream ordering.
    require_cuda(cudaStreamSynchronize(stream), "synchronize speckle filter");
    return result;
}

[[nodiscard]] bool close_float(
    const float actual,
    const float expected,
    const float relative = 3.0e-5F,
    const float absolute = 2.0e-6F) {
    if (std::isnan(expected)) {
        return std::isnan(actual);
    }
    if (!std::isfinite(actual) || !std::isfinite(expected)) {
        return actual == expected;
    }
    return std::abs(actual - expected) <=
           absolute + relative * std::abs(expected);
}

void expect_matches(
    TestContext& test,
    const FilterResult& actual,
    const FilterResult& expected,
    const std::string_view description,
    const float relative = 3.0e-5F,
    const float absolute = 2.0e-6F) {
    test.expect(
        actual.values.size() == expected.values.size() &&
            actual.validity.size() == expected.validity.size(),
        std::string(description) + " sizes");
    if (actual.values.size() != expected.values.size() ||
        actual.validity.size() != expected.validity.size()) {
        return;
    }

    for (std::size_t index = 0; index < actual.values.size(); ++index) {
        if (actual.validity[index] != expected.validity[index]) {
            test.expect(
                false,
                std::string(description) + " validity at " +
                    std::to_string(index));
            return;
        }
        if (!close_float(
                actual.values[index],
                expected.values[index],
                relative,
                absolute)) {
            std::cerr << "DETAIL: " << description << " index " << index
                      << ": actual=" << actual.values[index]
                      << ", expected=" << expected.values[index] << '\n';
            test.expect(
                false,
                std::string(description) + " values");
            return;
        }
    }
}

void test_option_identity(TestContext& test) {
    SpeckleFilterOptions first;
    const SpeckleFilterOptions same = first;
    test.expect(first == same, "speckle options compare equal by value");
    first.equivalent_number_of_looks = 4.0F;
    test.expect(first != same, "Lee ENL participates in render identity");
}

void test_passthrough(TestContext& test, const cudaStream_t stream) {
    const std::vector<float> input{
        1.0F,
        -2.0F,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -0.0F,
        8.0F};
    const std::vector<std::uint8_t> validity{1, 254, 1, 1, 255, 0};
    SpeckleFilterOptions options;
    options.filter = SpeckleFilter::none;
    options.domain = SpeckleDomain::phase;
    options.window_size = 2;
    options.equivalent_number_of_looks =
        std::numeric_limits<float>::quiet_NaN();

    const auto actual =
        run_gpu(input, validity, 3, 2, options, stream);
    const auto expected =
        cpu_reference(input, validity, 3, 2, options);
    expect_matches(test, actual, expected, "None pass-through");
    test.expect(
        std::signbit(actual.values[4]) == false &&
            std::isnan(actual.values[4]),
        "masked pass-through sample becomes NaN");
}

void test_masks_borders_and_nonfinite(
    TestContext& test,
    const cudaStream_t stream) {
    constexpr std::size_t width = 4;
    constexpr std::size_t height = 3;
    const std::vector<float> input{
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, std::numeric_limits<float>::quiet_NaN(), -7.0F, 8.0F,
        9.0F, 10.0F, 11.0F, 12.0F};
    const std::vector<std::uint8_t> validity{
        1, 1, 1, 255,
        1, 1, 1, 1,
        0, 254, 1, 1};

    for (const SpeckleFilter filter :
         {SpeckleFilter::boxcar, SpeckleFilter::lee}) {
        SpeckleFilterOptions options;
        options.filter = filter;
        options.domain = SpeckleDomain::linear_power;
        options.window_size = 3;
        options.equivalent_number_of_looks = 2.5F;
        const auto actual =
            run_gpu(input, validity, width, height, options, stream);
        const auto expected =
            cpu_reference(input, validity, width, height, options);
        expect_matches(
            test,
            actual,
            expected,
            filter == SpeckleFilter::boxcar
                ? "Boxcar mask and clipped borders"
                : "Lee mask and clipped borders");

        test.expect(
            actual.validity[0] == 1,
            "top-left clipped neighborhood remains valid");
        test.expect(
            actual.validity[5] == 0 &&
                actual.validity[6] == 0 &&
                actual.validity[8] == 0,
            "nonfinite, negative-power, and masked centers stay invalid");
    }

    std::vector<std::uint8_t> all_invalid(width * height, 0);
    SpeckleFilterOptions all_invalid_options;
    all_invalid_options.filter = SpeckleFilter::boxcar;
    all_invalid_options.window_size = 7;
    const auto all_invalid_actual = run_gpu(
        input,
        all_invalid,
        width,
        height,
        all_invalid_options,
        stream);
    test.expect(
        std::all_of(
            all_invalid_actual.validity.begin(),
            all_invalid_actual.validity.end(),
            [](const std::uint8_t value) { return value == 0; }),
        "all-invalid windows produce invalid outputs");
    test.expect(
        std::all_of(
            all_invalid_actual.values.begin(),
            all_invalid_actual.values.end(),
            [](const float value) { return std::isnan(value); }),
        "all-invalid windows produce NaNs");
}

void test_constant_and_windows(
    TestContext& test,
    const cudaStream_t stream) {
    constexpr std::size_t width = 8;
    constexpr std::size_t height = 6;
    const std::vector<float> input(width * height, 7.25F);
    for (const auto filter :
         {SpeckleFilter::boxcar, SpeckleFilter::lee}) {
        for (const std::uint32_t window : {3U, 5U, 7U}) {
            SpeckleFilterOptions options;
            options.filter = filter;
            options.window_size = window;
            options.equivalent_number_of_looks = 4.0F;
            const auto actual =
                run_gpu(input, {}, width, height, options, stream);
            test.expect(
                std::all_of(
                    actual.values.begin(),
                    actual.values.end(),
                    [](const float value) {
                        return close_float(value, 7.25F, 1.0e-6F, 1.0e-6F);
                    }),
                "constant image is invariant for every supported window");
            test.expect(
                std::all_of(
                    actual.validity.begin(),
                    actual.validity.end(),
                    [](const std::uint8_t value) { return value == 1; }),
                "constant image validity remains canonical");
        }
    }
}

void test_impulse_and_enl(
    TestContext& test,
    const cudaStream_t stream) {
    constexpr std::size_t side = 9;
    constexpr std::size_t center = 4 * side + 4;
    std::vector<float> input(side * side, 1.0F);
    input[center] = 100.0F;

    SpeckleFilterOptions boxcar;
    boxcar.filter = SpeckleFilter::boxcar;
    boxcar.window_size = 5;
    const auto box =
        run_gpu(input, {}, side, side, boxcar, stream);

    SpeckleFilterOptions lee = boxcar;
    lee.filter = SpeckleFilter::lee;
    lee.equivalent_number_of_looks = 1.0F;
    const auto lee_one =
        run_gpu(input, {}, side, side, lee, stream);
    lee.equivalent_number_of_looks = 8.0F;
    const auto lee_eight =
        run_gpu(input, {}, side, side, lee, stream);

    test.expect(
        close_float(box.values[center], 124.0F / 25.0F),
        "Boxcar attenuates a center impulse to its local mean");
    test.expect(
        box.values[center] < lee_one.values[center] &&
            lee_one.values[center] < lee_eight.values[center] &&
            lee_eight.values[center] < input[center],
        "Lee adaptively preserves impulse contrast and responds to ENL");

    const auto expected =
        cpu_reference(input, {}, side, side, lee);
    expect_matches(test, lee_eight, expected, "Lee impulse CPU reference");
}

void test_domains(TestContext& test, const cudaStream_t stream) {
    constexpr std::size_t width = 5;
    constexpr std::size_t height = 4;
    const std::vector<float> power{
        0.05F, 0.10F, 0.15F, 0.20F, 0.25F,
        0.40F, 0.55F, 0.70F, 0.85F, 1.00F,
        1.20F, 1.40F, 1.60F, 1.80F, 2.00F,
        2.50F, 3.00F, 3.50F, 4.00F, 4.50F};
    std::vector<float> amplitude(power.size());
    std::vector<float> db(power.size());
    for (std::size_t index = 0; index < power.size(); ++index) {
        amplitude[index] = std::sqrt(power[index]);
        db[index] = 10.0F * std::log10(power[index]);
    }

    for (const auto filter :
         {SpeckleFilter::boxcar, SpeckleFilter::lee}) {
        SpeckleFilterOptions linear_options;
        linear_options.filter = filter;
        linear_options.domain = SpeckleDomain::linear_power;
        linear_options.window_size = 5;
        linear_options.equivalent_number_of_looks = 3.0F;
        const auto linear = run_gpu(
            power, {}, width, height, linear_options, stream);

        SpeckleFilterOptions amplitude_options = linear_options;
        amplitude_options.domain = SpeckleDomain::amplitude;
        const auto amplitude_result = run_gpu(
            amplitude, {}, width, height, amplitude_options, stream);

        SpeckleFilterOptions db_options = linear_options;
        db_options.domain = SpeckleDomain::power_db;
        const auto db_result =
            run_gpu(db, {}, width, height, db_options, stream);

        bool domains_agree = true;
        for (std::size_t index = 0; index < power.size(); ++index) {
            const float amplitude_as_power =
                amplitude_result.values[index] *
                amplitude_result.values[index];
            const float db_as_power =
                std::pow(10.0F, db_result.values[index] * 0.1F);
            domains_agree =
                domains_agree &&
                close_float(amplitude_as_power, linear.values[index],
                            8.0e-5F, 3.0e-6F) &&
                close_float(db_as_power, linear.values[index],
                            8.0e-5F, 3.0e-6F);
        }
        test.expect(
            domains_agree,
            filter == SpeckleFilter::boxcar
                ? "Boxcar filters amplitude and dB in linear power"
                : "Lee filters amplitude and dB in linear power");

        const auto db_expected =
            cpu_reference(db, {}, width, height, db_options);
        expect_matches(
            test,
            db_result,
            db_expected,
            filter == SpeckleFilter::boxcar
                ? "Boxcar dB CPU reference"
                : "Lee dB CPU reference",
            8.0e-5F,
            4.0e-5F);
    }
}

void test_extreme_finite_values(
    TestContext& test,
    const cudaStream_t stream) {
    constexpr std::size_t side = 3;
    constexpr float high = std::numeric_limits<float>::max();
    const std::vector<float> amplitude{
        high, high * 0.75F, high * 0.50F,
        high * 0.25F, high * 0.125F, high * 0.0625F,
        high * 0.50F, high * 0.75F, high};
    const std::vector<float> high_db{
        900.0F, 890.0F, 880.0F,
        870.0F, 860.0F, 850.0F,
        880.0F, 890.0F, 900.0F};
    const std::vector<float> constant_extreme(side * side, high);

    for (const SpeckleFilter filter :
         {SpeckleFilter::boxcar, SpeckleFilter::lee}) {
        SpeckleFilterOptions options;
        options.filter = filter;
        options.window_size = 3;
        options.equivalent_number_of_looks = 3.0F;

        options.domain = SpeckleDomain::amplitude;
        const auto amplitude_actual = run_gpu(
            amplitude, {}, side, side, options, stream);
        const auto amplitude_expected = cpu_reference(
            amplitude, {}, side, side, options);
        expect_matches(
            test,
            amplitude_actual,
            amplitude_expected,
            filter == SpeckleFilter::boxcar
                ? "Boxcar preserves finite extreme amplitudes"
                : "Lee preserves finite extreme amplitudes",
            8.0e-5F,
            0.0F);

        options.domain = SpeckleDomain::power_db;
        const auto db_actual = run_gpu(
            high_db, {}, side, side, options, stream);
        const auto db_expected = cpu_reference(
            high_db, {}, side, side, options);
        expect_matches(
            test,
            db_actual,
            db_expected,
            filter == SpeckleFilter::boxcar
                ? "Boxcar handles dB whose absolute power exceeds float"
                : "Lee handles dB whose absolute power exceeds float",
            3.0e-6F,
            1.0e-3F);

        // FLT_MAX dB cannot be converted to absolute power even in double,
        // but relative normalization makes a homogeneous field invariant.
        const auto db_extreme = run_gpu(
            constant_extreme, {}, side, side, options, stream);
        test.expect(
            std::all_of(
                db_extreme.values.begin(),
                db_extreme.values.end(),
                [](const float value) { return value == high; }) &&
                std::all_of(
                    db_extreme.validity.begin(),
                    db_extreme.validity.end(),
                    [](const std::uint8_t value) { return value == 1; }),
            "finite FLT_MAX dB remains finite and valid");
    }
}

void test_determinism(TestContext& test, const cudaStream_t stream) {
    constexpr std::size_t width = 19;
    constexpr std::size_t height = 17;
    std::vector<float> input(width * height);
    std::vector<std::uint8_t> validity(width * height, 1);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] =
            0.01F + static_cast<float>((index * 37U) % 101U) / 17.0F;
        if (index % 23U == 0) {
            validity[index] = 255;
        }
    }
    SpeckleFilterOptions options;
    options.filter = SpeckleFilter::lee;
    options.window_size = 7;
    options.equivalent_number_of_looks = 2.75F;
    const auto first =
        run_gpu(input, validity, width, height, options, stream);
    const auto second =
        run_gpu(input, validity, width, height, options, stream);
    test.expect(
        std::memcmp(
            first.values.data(),
            second.values.data(),
            first.values.size() * sizeof(float)) == 0 &&
            first.validity == second.validity,
        "repeated Lee launches are bitwise deterministic");
}

void test_validation(TestContext& test) {
    DeviceBuffer<float> input(16);
    DeviceBuffer<float> output(16);
    DeviceBuffer<std::uint8_t> input_mask(16);
    DeviceBuffer<std::uint8_t> output_mask(16);

    SpeckleFilterOptions options;
    options.filter = SpeckleFilter::boxcar;

    test.expect(
        satview::gpu::launch_speckle_filter(
            nullptr, nullptr, 0, 9, options) == cudaSuccess,
        "zero width is a pointer-free no-op");
    test.expect(
        satview::gpu::launch_speckle_filter(
            nullptr, nullptr, 9, 0, options) == cudaSuccess,
        "zero height is a pointer-free no-op");
    test.expect(
        satview::gpu::launch_speckle_filter(
            nullptr, output.get(), 4, 4, options) ==
            cudaErrorInvalidValue,
        "null input is rejected");
    test.expect(
        satview::gpu::launch_speckle_filter(
            input.get(), nullptr, 4, 4, options) ==
            cudaErrorInvalidValue,
        "null output is rejected");
    test.expect(
        satview::gpu::launch_speckle_filter(
            input.get(), input.get(), 4, 4, options) ==
            cudaErrorInvalidValue,
        "in-place data filtering is rejected");
    options.validity.input = input_mask.get();
    options.validity.output = input_mask.get();
    test.expect(
        satview::gpu::launch_speckle_filter(
            input.get(), output.get(), 4, 4, options) ==
            cudaErrorInvalidValue,
        "in-place validity filtering is rejected");
    options.validity.output = output_mask.get();

    for (const std::uint32_t bad_window : {0U, 1U, 2U, 4U, 9U}) {
        options.window_size = bad_window;
        test.expect(
            satview::gpu::launch_speckle_filter(
                input.get(), output.get(), 4, 4, options) ==
                cudaErrorInvalidValue,
            "unsupported speckle window is rejected");
    }
    options.window_size = 3;

    options.filter = SpeckleFilter::lee;
    for (const float bad_enl :
         {0.0F,
          -1.0F,
          std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::quiet_NaN()}) {
        options.equivalent_number_of_looks = bad_enl;
        test.expect(
            satview::gpu::launch_speckle_filter(
                input.get(), output.get(), 4, 4, options) ==
                cudaErrorInvalidValue,
            "non-positive or non-finite Lee ENL is rejected");
    }
    options.equivalent_number_of_looks = 1.0F;

    options.domain = SpeckleDomain::power_db;
    for (const float bad_epsilon :
         {0.0F,
          -1.0F,
          std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::quiet_NaN()}) {
        options.power_epsilon = bad_epsilon;
        test.expect(
            satview::gpu::launch_speckle_filter(
                input.get(), output.get(), 4, 4, options) ==
                cudaErrorInvalidValue,
            "non-positive or non-finite dB epsilon is rejected");
    }
    options.power_epsilon =
        satview::gpu::kDefaultSpecklePowerEpsilon;

    for (const auto unsupported :
         {SpeckleDomain::phase, SpeckleDomain::signed_value}) {
        options.domain = unsupported;
        test.expect(
            satview::gpu::launch_speckle_filter(
                input.get(), output.get(), 4, 4, options) ==
                cudaErrorNotSupported,
            "phase and signed neighborhood filtering are unsupported");
    }

    options.domain = static_cast<SpeckleDomain>(255);
    test.expect(
        satview::gpu::launch_speckle_filter(
            input.get(), output.get(), 4, 4, options) ==
            cudaErrorInvalidValue,
        "unknown speckle domain is rejected");
    options.domain = SpeckleDomain::linear_power;
    options.filter = static_cast<SpeckleFilter>(255);
    test.expect(
        satview::gpu::launch_speckle_filter(
            input.get(), output.get(), 4, 4, options) ==
            cudaErrorInvalidValue,
        "unknown speckle filter is rejected");

    options.filter = SpeckleFilter::boxcar;
    test.expect(
        satview::gpu::launch_speckle_filter(
            input.get(),
            output.get(),
            std::numeric_limits<std::size_t>::max(),
            2,
            options) == cudaErrorInvalidValue,
        "dimension multiplication overflow is rejected");
    test.expect(
        satview::gpu::launch_speckle_filter(
            input.get(), output.get(), 16, 1048561, options) ==
            cudaErrorInvalidValue,
        "unsupported CUDA grid height is rejected");
}

void test_launch_error_reporting(TestContext& test) {
    DeviceBuffer<float> input(16);
    DeviceBuffer<float> output(16);
    cudaStream_t stale_stream = nullptr;
    test.expect_cuda(
        cudaStreamCreateWithFlags(
            &stale_stream, cudaStreamNonBlocking),
        "create launch-error stream");
    if (stale_stream == nullptr) {
        return;
    }
    test.expect_cuda(
        cudaStreamDestroy(stale_stream),
        "destroy launch-error stream");

    static_cast<void>(cudaGetLastError());
    SpeckleFilterOptions options;
    options.filter = SpeckleFilter::boxcar;
    options.window_size = 3;
    options.stream = stale_stream;
    const cudaError_t launch_status =
        satview::gpu::launch_speckle_filter(
            input.get(), output.get(), 4, 4, options);
    test.expect(
        launch_status != cudaSuccess,
        "immediate CUDA launch errors are returned to the caller");
    static_cast<void>(cudaGetLastError());
}

[[nodiscard]] float benchmark_filter(
    const std::size_t side,
    const SpeckleFilter filter,
    const SpeckleDomain domain,
    const cudaStream_t stream) {
    const std::size_t count = side * side;
    std::vector<float> host_input(count);
    for (std::size_t index = 0; index < count; ++index) {
        const float ramp =
            static_cast<float>((index * 48271ULL) % 8191ULL) / 8191.0F;
        const float power = 0.01F + ramp * ramp * 4.0F;
        if (domain == SpeckleDomain::amplitude) {
            host_input[index] = std::sqrt(power);
        } else if (domain == SpeckleDomain::power_db) {
            host_input[index] = 10.0F * std::log10(power);
        } else {
            host_input[index] = power;
        }
    }
    DeviceBuffer<float> input(count);
    DeviceBuffer<float> output(count);
    DeviceBuffer<std::uint8_t> validity(count);
    require_cuda(
        cudaMemcpyAsync(
            input.get(),
            host_input.data(),
            input.bytes(),
            cudaMemcpyHostToDevice,
            stream),
        "copy benchmark input");
    require_cuda(
        cudaMemsetAsync(validity.get(), 1, validity.bytes(), stream),
        "initialize benchmark validity");

    SpeckleFilterOptions options;
    options.filter = filter;
    options.domain = domain;
    options.window_size = 5;
    options.equivalent_number_of_looks = 3.0F;
    options.validity.input = validity.get();
    options.stream = stream;
    for (int warmup = 0; warmup < 3; ++warmup) {
        require_cuda(
            satview::gpu::launch_speckle_filter(
                input.get(), output.get(), side, side, options),
            "warm speckle benchmark");
    }
    require_cuda(
        cudaStreamSynchronize(stream),
        "synchronize speckle benchmark warmup");

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    require_cuda(cudaEventCreate(&begin), "create benchmark begin event");
    try {
        require_cuda(cudaEventCreate(&end), "create benchmark end event");
        constexpr int iterations = 20;
        require_cuda(
            cudaEventRecord(begin, stream),
            "record benchmark begin");
        for (int iteration = 0; iteration < iterations; ++iteration) {
            require_cuda(
                satview::gpu::launch_speckle_filter(
                    input.get(), output.get(), side, side, options),
                "launch timed speckle benchmark");
        }
        require_cuda(
            cudaEventRecord(end, stream),
            "record benchmark end");
        require_cuda(
            cudaEventSynchronize(end),
            "synchronize benchmark end");
        float total_milliseconds = 0.0F;
        require_cuda(
            cudaEventElapsedTime(
                &total_milliseconds, begin, end),
            "measure speckle benchmark");
        require_cuda(cudaEventDestroy(end), "destroy benchmark end event");
        require_cuda(
            cudaEventDestroy(begin), "destroy benchmark begin event");
        return total_milliseconds / iterations;
    } catch (...) {
        if (end != nullptr) {
            static_cast<void>(cudaEventDestroy(end));
        }
        static_cast<void>(cudaEventDestroy(begin));
        throw;
    }
}

void run_benchmarks(const cudaStream_t stream) {
    std::cout << "Speckle GPU kernel benchmarks (5x5, event-timed):\n";
    for (const std::size_t side : {2048U, 4096U}) {
        for (const SpeckleDomain domain :
             {SpeckleDomain::amplitude,
              SpeckleDomain::linear_power,
              SpeckleDomain::power_db}) {
            for (const SpeckleFilter filter :
                 {SpeckleFilter::boxcar, SpeckleFilter::lee}) {
                const float milliseconds =
                    benchmark_filter(side, filter, domain, stream);
                const double gigapixels_per_second =
                    (static_cast<double>(side) * side) /
                    (static_cast<double>(milliseconds) * 1.0e6);
                const char* const domain_name =
                    domain == SpeckleDomain::amplitude ? "amplitude" :
                    domain == SpeckleDomain::linear_power ? "power" : "dB";
                std::cout
                    << "  " << side << 'x' << side << ' '
                    << (filter == SpeckleFilter::boxcar ? "Boxcar" : "Lee")
                    << ' ' << domain_name << ": " << milliseconds << " ms, "
                    << gigapixels_per_second << " Gpix/s\n";
            }
        }
    }
}

}  // namespace

int main(const int argc, char** const argv) {
    bool run_benchmark = true;
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) == "--no-benchmark") {
            run_benchmark = false;
        } else {
            std::cerr << "Unknown argument: " << argv[index] << '\n';
            return 2;
        }
    }

    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status == cudaErrorNoDevice ||
        count_status == cudaErrorInsufficientDriver ||
        device_count == 0) {
        static_cast<void>(cudaGetLastError());
        std::cout << "CUDA speckle tests skipped: no CUDA device\n";
        return 0;
    }
    if (count_status != cudaSuccess) {
        std::cerr << "FAIL: cudaGetDeviceCount: "
                  << cudaGetErrorString(count_status) << '\n';
        return 1;
    }

    try {
        TestContext test;
        Stream stream;
        test_option_identity(test);
        test_passthrough(test, stream.get());
        test_masks_borders_and_nonfinite(test, stream.get());
        test_constant_and_windows(test, stream.get());
        test_impulse_and_enl(test, stream.get());
        test_domains(test, stream.get());
        test_extreme_finite_values(test, stream.get());
        test_determinism(test, stream.get());
        test_validation(test);
        test_launch_error_reporting(test);
        if (test.failures != 0) {
            std::cerr << test.failures
                      << " CUDA speckle-filter test(s) failed\n";
            return 1;
        }

        if (run_benchmark) {
            run_benchmarks(stream.get());
        }
        std::cout << "All CUDA speckle-filter tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: CUDA speckle-filter test threw: "
                  << error.what() << '\n';
        return 1;
    }
}
