#include "satview/gpu/distribution.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace satview::gpu {
namespace {

constexpr unsigned int kBlockSize = 256;
constexpr unsigned int kMaximumReductionBlocks = 4096;
constexpr unsigned int kMaximumHistogramBlocks = 1024;

struct DeviceHistogram {
    unsigned long long finite_count;
    unsigned long long invalid_count;
    unsigned int minimum_ordered;
    unsigned int maximum_ordered;
    unsigned long long bins[kDistributionHistogramBins];
};

static_assert(sizeof(unsigned long long) == sizeof(std::uint64_t));

[[noreturn]] void throw_cuda_error(
    const char* operation, const cudaError_t error) {
    const char* const description = cudaGetErrorString(error);
    throw std::runtime_error(
        std::string(operation) + " failed: " +
        (description != nullptr ? description : "unknown CUDA error"));
}

void check_cuda(const char* operation, const cudaError_t error) {
    if (error != cudaSuccess) {
        throw_cuda_error(operation, error);
    }
}

__device__ unsigned int ordered_float(const float value) {
    const unsigned int bits = __float_as_uint(value);
    return (bits & 0x80000000U) != 0U
        ? ~bits
        : bits ^ 0x80000000U;
}

[[nodiscard]] float float_from_ordered(const unsigned int ordered) noexcept {
    const unsigned int bits = (ordered & 0x80000000U) != 0U
        ? ordered ^ 0x80000000U
        : ~ordered;
    return std::bit_cast<float>(bits);
}

__global__ void initialize_histogram(DeviceHistogram* result) {
    result->minimum_ordered = 0xFFFFFFFFU;
    result->maximum_ordered = 0U;
}

__global__ void reduce_distribution(
    const float* values,
    const std::size_t count,
    DeviceHistogram* result) {
    __shared__ unsigned int minimums[kBlockSize];
    __shared__ unsigned int maximums[kBlockSize];
    __shared__ unsigned long long finite_counts[kBlockSize];
    __shared__ unsigned long long invalid_counts[kBlockSize];

    unsigned int local_minimum = 0xFFFFFFFFU;
    unsigned int local_maximum = 0U;
    unsigned long long local_finite = 0;
    unsigned long long local_invalid = 0;

    const std::size_t first =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = first; index < count; index += stride) {
        const float value = values[index];
        if (isfinite(value)) {
            const unsigned int ordered = ordered_float(value);
            local_minimum = min(local_minimum, ordered);
            local_maximum = max(local_maximum, ordered);
            ++local_finite;
        } else {
            ++local_invalid;
        }
    }

    minimums[threadIdx.x] = local_minimum;
    maximums[threadIdx.x] = local_maximum;
    finite_counts[threadIdx.x] = local_finite;
    invalid_counts[threadIdx.x] = local_invalid;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset != 0; offset /= 2) {
        if (threadIdx.x < offset) {
            minimums[threadIdx.x] =
                min(minimums[threadIdx.x], minimums[threadIdx.x + offset]);
            maximums[threadIdx.x] =
                max(maximums[threadIdx.x], maximums[threadIdx.x + offset]);
            finite_counts[threadIdx.x] += finite_counts[threadIdx.x + offset];
            invalid_counts[threadIdx.x] +=
                invalid_counts[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (finite_counts[0] != 0) {
            atomicMin(&result->minimum_ordered, minimums[0]);
            atomicMax(&result->maximum_ordered, maximums[0]);
            atomicAdd(&result->finite_count, finite_counts[0]);
        }
        if (invalid_counts[0] != 0) {
            atomicAdd(&result->invalid_count, invalid_counts[0]);
        }
    }
}

__global__ void build_histogram(
    const float* values,
    const std::size_t count,
    DeviceHistogram* result) {
    __shared__ unsigned int local_bins[kDistributionHistogramBins];
    local_bins[threadIdx.x] = 0;
    __syncthreads();

    if (result->finite_count != 0) {
        const float minimum = __uint_as_float(
            (result->minimum_ordered & 0x80000000U) != 0U
                ? result->minimum_ordered ^ 0x80000000U
                : ~result->minimum_ordered);
        const float maximum = __uint_as_float(
            (result->maximum_ordered & 0x80000000U) != 0U
                ? result->maximum_ordered ^ 0x80000000U
                : ~result->maximum_ordered);
        const float float_range = maximum - minimum;
        const bool constant = maximum == minimum;
        const bool wide_range = !constant && !isfinite(float_range);
        const float float_scale = !constant && !wide_range
            ? static_cast<float>(kDistributionHistogramBins) / float_range
            : 0.0F;
        const double double_scale = wide_range
            ? static_cast<double>(kDistributionHistogramBins) /
                (static_cast<double>(maximum) -
                 static_cast<double>(minimum))
            : 0.0;

        const std::size_t first =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t stride =
            static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (std::size_t index = first; index < count; index += stride) {
            const float value = values[index];
            if (!isfinite(value)) {
                continue;
            }

            unsigned int bin = static_cast<unsigned int>(
                kDistributionHistogramBins / 2);
            if (!constant) {
                const double scaled = wide_range
                    ? (static_cast<double>(value) -
                       static_cast<double>(minimum)) * double_scale
                    : static_cast<double>((value - minimum) * float_scale);
                if (scaled <= 0.0) {
                    bin = 0;
                } else if (
                    scaled >=
                    static_cast<double>(kDistributionHistogramBins)) {
                    bin = static_cast<unsigned int>(
                        kDistributionHistogramBins - 1);
                } else {
                    bin = static_cast<unsigned int>(scaled);
                }
            }
            atomicAdd(&local_bins[bin], 1U);
        }
    }
    __syncthreads();

    if (local_bins[threadIdx.x] != 0) {
        atomicAdd(
            &result->bins[threadIdx.x],
            static_cast<unsigned long long>(local_bins[threadIdx.x]));
    }
}

[[nodiscard]] unsigned int block_count(
    const std::size_t count, const unsigned int maximum) noexcept {
    if (count == 0) {
        return 0;
    }
    const std::size_t requested =
        1 + (count - 1) / static_cast<std::size_t>(kBlockSize);
    return static_cast<unsigned int>(std::min<std::size_t>(
        requested, static_cast<std::size_t>(maximum)));
}

}  // namespace

struct AsyncResidentDistribution::Impl {
    DeviceHistogram* device = nullptr;
    DeviceHistogram* host = nullptr;
    cudaEvent_t started = nullptr;
    cudaEvent_t finished = nullptr;
    bool is_pending = false;
    std::uint64_t generation = 0;

    Impl() {
        try {
            check_cuda(
                "allocate distribution device result",
                cudaMalloc(
                    reinterpret_cast<void**>(&device),
                    sizeof(DeviceHistogram)));
            check_cuda(
                "allocate pinned distribution host result",
                cudaMallocHost(
                    reinterpret_cast<void**>(&host),
                    sizeof(DeviceHistogram)));
            check_cuda(
                "create distribution start event",
                cudaEventCreate(&started));
            check_cuda(
                "create distribution completion event",
                cudaEventCreate(&finished));
        } catch (...) {
            release();
            throw;
        }
    }

    ~Impl() noexcept {
        if (is_pending && finished != nullptr) {
            static_cast<void>(cudaEventSynchronize(finished));
        }
        release();
    }

    void release() noexcept {
        if (finished != nullptr) {
            static_cast<void>(cudaEventDestroy(finished));
            finished = nullptr;
        }
        if (started != nullptr) {
            static_cast<void>(cudaEventDestroy(started));
            started = nullptr;
        }
        if (host != nullptr) {
            static_cast<void>(cudaFreeHost(host));
            host = nullptr;
        }
        if (device != nullptr) {
            static_cast<void>(cudaFree(device));
            device = nullptr;
        }
    }
};

AsyncResidentDistribution::AsyncResidentDistribution()
    : impl_(std::make_unique<Impl>()) {}

AsyncResidentDistribution::~AsyncResidentDistribution() noexcept = default;

bool AsyncResidentDistribution::pending() const noexcept {
    return impl_->is_pending;
}

void AsyncResidentDistribution::enqueue(
    const float* device_values,
    const std::size_t count,
    const cudaStream_t stream,
    const std::uint64_t generation) {
    if (impl_->is_pending) {
        throw std::logic_error(
            "resident distribution overwritten before completion");
    }
    if (count != 0 && device_values == nullptr) {
        throw std::invalid_argument(
            "resident distribution requires device values");
    }

    check_cuda(
        "record distribution start",
        cudaEventRecord(impl_->started, stream));
    check_cuda(
        "clear distribution accumulator",
        cudaMemsetAsync(
            impl_->device, 0, sizeof(DeviceHistogram), stream));
    initialize_histogram<<<1, 1, 0, stream>>>(impl_->device);
    check_cuda(
        "launch distribution initialization",
        cudaPeekAtLastError());

    const unsigned int reduction_blocks =
        block_count(count, kMaximumReductionBlocks);
    if (reduction_blocks != 0) {
        reduce_distribution<<<
            reduction_blocks, kBlockSize, 0, stream>>>(
                device_values, count, impl_->device);
        check_cuda(
            "launch resident distribution reduction",
            cudaPeekAtLastError());

        const unsigned int histogram_blocks =
            block_count(count, kMaximumHistogramBlocks);
        build_histogram<<<
            histogram_blocks, kBlockSize, 0, stream>>>(
                device_values, count, impl_->device);
        check_cuda(
            "launch resident distribution histogram",
            cudaPeekAtLastError());
    }

    check_cuda(
        "copy resident distribution summary",
        cudaMemcpyAsync(
            impl_->host,
            impl_->device,
            sizeof(DeviceHistogram),
            cudaMemcpyDeviceToHost,
            stream));
    check_cuda(
        "record distribution completion",
        cudaEventRecord(impl_->finished, stream));
    impl_->generation = generation;
    impl_->is_pending = true;
}

bool AsyncResidentDistribution::poll(AsyncDistributionResult& result) {
    if (!impl_->is_pending) {
        return false;
    }
    const cudaError_t query = cudaEventQuery(impl_->finished);
    if (query == cudaErrorNotReady) {
        return false;
    }
    check_cuda("query distribution completion", query);

    DistributionHistogram histogram;
    histogram.finite_count =
        static_cast<std::uint64_t>(impl_->host->finite_count);
    histogram.invalid_count =
        static_cast<std::uint64_t>(impl_->host->invalid_count);
    if (histogram.finite_count != 0) {
        histogram.minimum =
            float_from_ordered(impl_->host->minimum_ordered);
        histogram.maximum =
            float_from_ordered(impl_->host->maximum_ordered);
    }
    for (std::size_t index = 0;
         index < kDistributionHistogramBins;
         ++index) {
        histogram.bins[index] =
            static_cast<std::uint64_t>(impl_->host->bins[index]);
    }

    float elapsed_milliseconds = 0.0F;
    check_cuda(
        "measure resident distribution",
        cudaEventElapsedTime(
            &elapsed_milliseconds, impl_->started, impl_->finished));
    result = AsyncDistributionResult{
        .generation = impl_->generation,
        .summary = summarize_distribution(histogram),
        .elapsed_milliseconds = elapsed_milliseconds,
    };
    impl_->is_pending = false;
    return true;
}

}  // namespace satview::gpu
