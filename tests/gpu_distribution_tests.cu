#include "satview/gpu/distribution.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {

#ifndef SATVIEW_REQUIRE_ACCELERATOR_TESTS
constexpr int kTestSkipped = 77;
#endif

void expect(
    const bool condition,
    const std::string_view description,
    int& failures) {
    if (!condition) {
        std::cerr << "GPU distribution test failed: "
                  << description << '\n';
        ++failures;
    }
}

bool expect_cuda(
    const cudaError_t result,
    const std::string_view operation,
    int& failures) {
    if (result == cudaSuccess) {
        return true;
    }
    std::cerr << "GPU distribution test CUDA failure in "
              << operation << ": " << cudaGetErrorString(result) << '\n';
    ++failures;
    return false;
}

[[nodiscard]] bool equal_float(const float left, const float right) {
    return left == right || (std::isnan(left) && std::isnan(right));
}

[[nodiscard]] bool same_summary(
    const satview::gpu::DistributionSummary& left,
    const satview::gpu::DistributionSummary& right) {
    return left.histogram.finite_count == right.histogram.finite_count &&
        left.histogram.invalid_count == right.histogram.invalid_count &&
        equal_float(left.histogram.minimum, right.histogram.minimum) &&
        equal_float(left.histogram.maximum, right.histogram.maximum) &&
        left.histogram.bins == right.histogram.bins &&
        equal_float(left.percentile_1, right.percentile_1) &&
        equal_float(left.percentile_2, right.percentile_2) &&
        equal_float(left.percentile_50, right.percentile_50) &&
        equal_float(left.percentile_98, right.percentile_98) &&
        equal_float(left.percentile_99, right.percentile_99);
}

__global__ void fill_full_page(float* values, const std::size_t count) {
    const std::size_t first =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = first; index < count; index += stride) {
        values[index] = index % 257 == 0
            ? __uint_as_float(0x7FC00000U)
            : static_cast<float>(index % 1024);
    }
}

void test_mixed_extrema(cudaStream_t stream, int& failures) {
    const std::vector<float> input{
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        -std::numeric_limits<float>::max(),
        -2.0F,
        -1.0F,
        -0.0F,
        0.0F,
        1.0F,
        2.0F,
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::infinity(),
    };
    float* device = nullptr;
    if (!expect_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device),
                input.size() * sizeof(float)),
            "allocate mixed input",
            failures)) {
        return;
    }
    if (!expect_cuda(
            cudaMemcpyAsync(
                device,
                input.data(),
                input.size() * sizeof(float),
                cudaMemcpyHostToDevice,
                stream),
            "copy mixed input",
            failures)) {
        static_cast<void>(cudaFree(device));
        return;
    }

    satview::gpu::AsyncResidentDistribution distribution;
    distribution.enqueue(device, input.size(), stream, 91);
    if (!expect_cuda(
            cudaStreamSynchronize(stream),
            "complete mixed distribution",
            failures)) {
        static_cast<void>(cudaFree(device));
        return;
    }
    satview::gpu::AsyncDistributionResult result;
    expect(
        distribution.poll(result),
        "completed mixed result is pollable",
        failures);
    expect(result.generation == 91, "generation is preserved", failures);
    expect(
        result.summary.histogram.finite_count == 8 &&
            result.summary.histogram.invalid_count == 3,
        "finite and invalid values are counted exactly",
        failures);
    expect(
        result.summary.histogram.minimum ==
                -std::numeric_limits<float>::max() &&
            result.summary.histogram.maximum ==
                std::numeric_limits<float>::max(),
        "finite extrema are exact across the full float range",
        failures);
    std::uint64_t histogram_count = 0;
    for (const auto count : result.summary.histogram.bins) {
        histogram_count += count;
    }
    expect(
        histogram_count == 8,
        "GPU histogram contains every finite value exactly once",
        failures);
    expect(
        std::isfinite(result.summary.percentile_1) &&
            std::isfinite(result.summary.percentile_99),
        "wide-range percentiles remain finite",
        failures);
    const auto canonical =
        satview::gpu::summarize_distribution(result.summary.histogram);
    expect(
        same_summary(result.summary, canonical),
        "async result uses the canonical host summarizer",
        failures);
    expect(
        result.elapsed_milliseconds >= 0.0F,
        "GPU timing is available",
        failures);
    static_cast<void>(cudaFree(device));
}

void test_constant_and_invalid(cudaStream_t stream, int& failures) {
    const std::vector<float> input{
        7.5F,
        7.5F,
        7.5F,
        std::numeric_limits<float>::quiet_NaN(),
    };
    float* device = nullptr;
    if (!expect_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device),
                input.size() * sizeof(float)),
            "allocate constant input",
            failures)) {
        return;
    }
    expect_cuda(
        cudaMemcpyAsync(
            device,
            input.data(),
            input.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream),
        "copy constant input",
        failures);

    satview::gpu::AsyncResidentDistribution distribution;
    distribution.enqueue(device, input.size(), stream, 92);
    expect_cuda(
        cudaStreamSynchronize(stream),
        "complete constant distribution",
        failures);
    satview::gpu::AsyncDistributionResult result;
    expect(
        distribution.poll(result),
        "completed constant result is pollable",
        failures);
    expect(
        result.summary.histogram.finite_count == 3 &&
            result.summary.histogram.invalid_count == 1,
        "constant distribution counts are exact",
        failures);
    expect(
        result.summary.histogram.minimum == 7.5F &&
            result.summary.histogram.maximum == 7.5F &&
            result.summary.percentile_50 == 7.5F,
        "constant distribution summary remains exact",
        failures);
    static_cast<void>(cudaFree(device));
}

void test_full_resident_page(cudaStream_t stream, int& failures) {
    constexpr std::size_t extent = 4096;
    constexpr std::size_t count = extent * extent;
    float* device = nullptr;
    if (!expect_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device), count * sizeof(float)),
            "allocate full resident page",
            failures)) {
        return;
    }
    fill_full_page<<<4096, 256, 0, stream>>>(device, count);
    if (!expect_cuda(
            cudaPeekAtLastError(),
            "fill full resident page",
            failures)) {
        static_cast<void>(cudaFree(device));
        return;
    }

    satview::gpu::AsyncResidentDistribution distribution;
    distribution.enqueue(device, count, stream, 93);
    if (!expect_cuda(
            cudaStreamSynchronize(stream),
            "complete full-page distribution",
            failures)) {
        static_cast<void>(cudaFree(device));
        return;
    }
    satview::gpu::AsyncDistributionResult result;
    expect(
        distribution.poll(result),
        "completed full-page result is pollable",
        failures);
    constexpr std::uint64_t invalid = (count + 256) / 257;
    expect(
        result.summary.histogram.invalid_count == invalid &&
            result.summary.histogram.finite_count == count - invalid,
        "full-page finite and invalid counts are exact",
        failures);
    expect(
        result.summary.histogram.minimum == 0.0F &&
            result.summary.histogram.maximum == 1023.0F,
        "full-page extrema are exact",
        failures);
    std::uint64_t histogram_count = 0;
    for (const auto bin_count : result.summary.histogram.bins) {
        histogram_count += bin_count;
    }
    expect(
        histogram_count == count - invalid,
        "full-page histogram accounts for every finite sample",
        failures);
    std::cout << "gpu-distribution-4096x4096-ms="
              << result.elapsed_milliseconds << '\n';
    static_cast<void>(cudaFree(device));
}

}  // namespace

int run_gpu_distribution_tests() {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status == cudaErrorNoDevice ||
        count_status == cudaErrorInsufficientDriver ||
        device_count == 0) {
        static_cast<void>(cudaGetLastError());
        std::cout << "GPU distribution tests skipped: no CUDA device\n";
#ifdef SATVIEW_REQUIRE_ACCELERATOR_TESTS
        std::cerr << "GPU distribution certification requires a usable "
                     "CUDA device\n";
        return 1;
#else
        return kTestSkipped;
#endif
    }
    if (count_status != cudaSuccess) {
        std::cerr << "GPU distribution tests could not query the device: "
                  << cudaGetErrorString(count_status) << '\n';
        return 1;
    }

    int failures = 0;
    cudaStream_t stream = nullptr;
    if (!expect_cuda(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            "create stream",
            failures)) {
        return failures;
    }
    test_mixed_extrema(stream, failures);
    test_constant_and_invalid(stream, failures);
    test_full_resident_page(stream, failures);
    expect_cuda(cudaStreamDestroy(stream), "destroy stream", failures);
    if (failures == 0) {
        std::cout << "GPU distribution tests passed\n";
    }
    return failures;
}
