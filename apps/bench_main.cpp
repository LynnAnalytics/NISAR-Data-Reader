#include "satview/hdf5_product.hpp"

#ifdef SATVIEW_HAS_CUDA
#include "satview/gpu/pinned_ring.hpp"
#include "satview/gpu/transforms.hpp"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <latch>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stop_token>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kMaximumBenchmarkChunks = 1'000'000;
constexpr std::size_t kPipelineRingSlots = 3;
constexpr std::size_t kPipelineTimingBatchChunks = 64;

struct Arguments {
    std::filesystem::path file;
    std::string layer;
    std::size_t chunks = 32;
    std::size_t warmup = 4;
    bool csv = false;
    bool pipeline = false;
};

void usage() {
    std::cerr
        << "Usage: sat-bench <NISAR.h5> [--layer PATH] [--chunks N] "
           "[--warmup N] [--pipeline] [--csv]\n"
        << "  --chunks: decimal integer in [1, 1000000]\n"
        << "  --warmup: decimal integer in [0, 1000000]\n";
}

std::size_t parse_bounded_count(std::string_view text,
                                std::string_view option,
                                bool allow_zero) {
    if (text.empty() ||
        !std::all_of(text.begin(), text.end(), [](const char character) {
            return character >= '0' && character <= '9';
        })) {
        throw std::invalid_argument(
            std::string(option) + " requires an unsigned decimal integer");
    }

    std::uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value, 10);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument(
            std::string(option) + " value is outside the supported integer range");
    }
    if (!allow_zero && value == 0) {
        throw std::invalid_argument(std::string(option) + " must be at least 1");
    }
    if (value > kMaximumBenchmarkChunks) {
        throw std::invalid_argument(
            std::string(option) + " must not exceed " +
            std::to_string(kMaximumBenchmarkChunks));
    }
    return static_cast<std::size_t>(value);
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--layer") {
            if (index + 1 >= argc) throw std::invalid_argument("--layer requires a path");
            result.layer = argv[++index];
        } else if (argument == "--chunks") {
            if (index + 1 >= argc) throw std::invalid_argument("--chunks requires a value");
            result.chunks = parse_bounded_count(argv[++index], "--chunks", false);
        } else if (argument == "--warmup") {
            if (index + 1 >= argc) throw std::invalid_argument("--warmup requires a value");
            result.warmup = parse_bounded_count(argv[++index], "--warmup", true);
        } else if (argument == "--pipeline") {
            result.pipeline = true;
        } else if (argument == "--csv") {
            result.csv = true;
        } else if (argument == "--help" || argument == "-h") {
            usage();
            std::exit(0);
        } else if (!argument.empty() && argument.front() == '-') {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        } else if (result.file.empty()) {
            result.file = std::filesystem::path(argument);
        } else {
            throw std::invalid_argument("multiple input files");
        }
    }
    if (result.file.empty()) throw std::invalid_argument("missing input file");
    return result;
}

double percentile(const std::vector<double>& values, double fraction) {
    if (values.empty()) return 0.0;
    const auto index = static_cast<std::size_t>(std::ceil(fraction * values.size()) - 1.0);
    return values[std::min(index, values.size() - 1)];
}

const satview::DatasetInfo& choose_layer(const satview::Hdf5Product& product,
                                         const std::string& requested) {
    if (!requested.empty()) {
        const auto* found = product.find_dataset(requested);
        if (found == nullptr) throw std::runtime_error("layer not found: " + requested);
        if (!found->data_type.readable) throw std::runtime_error("layer datatype is not readable");
        return *found;
    }
    for (const auto& dataset : product.datasets()) {
        if (dataset.role == satview::DatasetRole::science && dataset.data_type.readable) return dataset;
    }
    throw std::runtime_error("product has no readable science layer");
}

struct Sample {
    double hdf5_ms = 0.0;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    std::size_t bytes = 0;
};

#ifdef SATVIEW_HAS_CUDA
void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

cudaError_t launch_selected_transform(const satview::DatasetInfo& layer,
                                      const void* device_input,
                                      float* device_output,
                                      std::size_t count,
                                      cudaStream_t stream) {
    if (layer.layer_kind == satview::LayerKind::gslc_polarization) {
        return satview::gpu::launch_gslc_transform(
            static_cast<const float2*>(device_input), device_output, count,
            satview::gpu::GslcTransform::power_db, {.stream = stream});
    }
    if (layer.data_type.kind == satview::ScalarKind::compound_complex) {
        return satview::gpu::launch_gcov_complex_transform(
            static_cast<const float2*>(device_input), device_output, count,
            satview::gpu::GcovComplexTransform::magnitude, {.stream = stream});
    }
    if (layer.data_type.kind == satview::ScalarKind::floating_point) {
        return satview::gpu::launch_gcov_real_transform(
            static_cast<const float*>(device_input), device_output, count,
            satview::gpu::GcovRealTransform::power_db, {.stream = stream});
    }
    return cudaErrorInvalidValue;
}

class DeviceAllocation final {
public:
    DeviceAllocation(std::size_t bytes, const char* operation) {
        check_cuda(cudaMalloc(&data_, bytes), operation);
    }

    ~DeviceAllocation() noexcept {
        if (data_ != nullptr) static_cast<void>(cudaFree(data_));
    }

    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    [[nodiscard]] void* get() noexcept { return data_; }
    [[nodiscard]] const void* get() const noexcept { return data_; }

private:
    void* data_ = nullptr;
};

class CudaStream final {
public:
    CudaStream() {
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags");
    }

    ~CudaStream() noexcept {
        if (stream_ != nullptr) {
            // Drain queued work before destruction so it no longer references
            // caller-owned buffers during unwind.
            static_cast<void>(cudaStreamSynchronize(stream_));
            static_cast<void>(cudaStreamDestroy(stream_));
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

    void synchronize(const char* operation) const {
        check_cuda(cudaStreamSynchronize(stream_), operation);
    }

private:
    cudaStream_t stream_ = nullptr;
};

class CudaEvent final {
public:
    CudaEvent() {
        check_cuda(cudaEventCreate(&event_), "cudaEventCreate");
    }

    ~CudaEvent() noexcept {
        if (event_ != nullptr) static_cast<void>(cudaEventDestroy(event_));
    }

    CudaEvent(CudaEvent&& other) noexcept
        : event_(std::exchange(other.event_, nullptr)) {}

    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_ != nullptr) static_cast<void>(cudaEventDestroy(event_));
            event_ = std::exchange(other.event_, nullptr);
        }
        return *this;
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

    void record(cudaStream_t stream, const char* operation) const {
        check_cuda(cudaEventRecord(event_, stream), operation);
    }

    void synchronize(const char* operation) const {
        check_cuda(cudaEventSynchronize(event_), operation);
    }

    [[nodiscard]] float elapsed_from(const CudaEvent& start,
                                     const char* operation) const {
        float milliseconds = 0.0F;
        check_cuda(cudaEventElapsedTime(&milliseconds, start.get(), event_), operation);
        return milliseconds;
    }

private:
    cudaEvent_t event_ = nullptr;
};

struct PipelineEvents {
    CudaEvent h2d_start;
    CudaEvent h2d_stop;
    CudaEvent kernel_start;
    CudaEvent kernel_stop;
};
#endif

} // namespace

int main(int argc, char** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        const auto open_start = Clock::now();
        satview::Hdf5Product product(arguments.file);
        const double open_ms = std::chrono::duration<double, std::milli>(Clock::now() - open_start).count();
        const auto& layer = choose_layer(product, arguments.layer);
        if (!layer.chunk_dimensions) throw std::runtime_error("benchmark requires a chunked layer");

#ifdef SATVIEW_HAS_CUDA
        const bool is_float32 =
            layer.data_type.kind == satview::ScalarKind::floating_point &&
            layer.data_type.element_size == sizeof(float);
        const bool is_complex64 =
            layer.data_type.kind == satview::ScalarKind::compound_complex &&
            layer.data_type.element_size == sizeof(float) * 2;
        if (!is_float32 && !is_complex64) {
            throw std::runtime_error(
                "CUDA benchmark supports only float32 and complex64 layers");
        }
#else
        if (arguments.pipeline) {
            throw std::runtime_error("--pipeline requires a CUDA-enabled build");
        }
#endif

        const auto chunk_height = (*layer.chunk_dimensions)[0];
        const auto chunk_width = (*layer.chunk_dimensions)[1];
        if (arguments.warmup >
            std::numeric_limits<std::size_t>::max() - arguments.chunks) {
            throw std::overflow_error("warmup + chunks overflows size_t");
        }
        const std::size_t total_iterations = arguments.warmup + arguments.chunks;
        if (chunk_height == 0 || chunk_width == 0) {
            throw std::runtime_error("benchmark layer has a zero-sized source chunk");
        }
        if (layer.dimensions[0] == 0 || layer.dimensions[1] == 0) {
            throw std::runtime_error("benchmark layer has an empty raster dimension");
        }

        const auto chunks_down =
            layer.dimensions[0] / chunk_height +
            static_cast<std::uint64_t>(layer.dimensions[0] % chunk_height != 0);
        const auto chunks_across =
            layer.dimensions[1] / chunk_width +
            static_cast<std::uint64_t>(layer.dimensions[1] % chunk_width != 0);
        if (chunks_down >
            std::numeric_limits<std::uint64_t>::max() / chunks_across) {
            throw std::overflow_error("source chunk-grid size overflows uint64_t");
        }
        const std::uint64_t source_chunk_count = chunks_down * chunks_across;

        if (chunk_height >
            std::numeric_limits<std::uint64_t>::max() / chunk_width) {
            throw std::overflow_error("source chunk element count overflows uint64_t");
        }
        const std::uint64_t chunk_elements_u64 = chunk_height * chunk_width;
        if (chunk_elements_u64 > std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error("source chunk element count overflows size_t");
        }
        const std::size_t chunk_elements =
            static_cast<std::size_t>(chunk_elements_u64);
        if (layer.data_type.element_size == 0 ||
            chunk_elements > std::numeric_limits<std::size_t>::max() /
                                 layer.data_type.element_size) {
            throw std::overflow_error("source chunk byte count overflows size_t");
        }
        const std::size_t maximum_bytes =
            chunk_elements * layer.data_type.element_size;
#ifdef SATVIEW_HAS_CUDA
        if (chunk_elements >
            std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::overflow_error("output chunk byte count overflows size_t");
        }
        const std::size_t output_bytes = chunk_elements * sizeof(float);
#endif

        const auto make_plan = [&](const std::size_t iteration) {
            const std::uint64_t linear =
                static_cast<std::uint64_t>(iteration) % source_chunk_count;
            const std::uint64_t chunk_row = linear / chunks_across;
            const std::uint64_t chunk_column = linear % chunks_across;
            auto plan = product.make_read_plan(
                layer.path, chunk_row * chunk_height, chunk_column * chunk_width, 1, 1);
            if (plan.aligned.height == 0 || plan.aligned.width == 0 ||
                plan.aligned.height >
                    std::numeric_limits<std::uint64_t>::max() /
                        plan.aligned.width) {
                throw std::runtime_error("HDF5 read plan has invalid dimensions");
            }
            const std::uint64_t plan_elements_u64 =
                plan.aligned.height * plan.aligned.width;
            if (plan_elements_u64 > chunk_elements_u64 ||
                plan_elements_u64 > std::numeric_limits<std::size_t>::max()) {
                throw std::runtime_error("HDF5 read plan exceeds one source chunk");
            }
            const std::size_t plan_elements =
                static_cast<std::size_t>(plan_elements_u64);
            if (plan_elements > std::numeric_limits<std::size_t>::max() /
                                    layer.data_type.element_size ||
                plan.expected_bytes !=
                    plan_elements * layer.data_type.element_size ||
                plan.expected_bytes > maximum_bytes) {
                throw std::runtime_error("HDF5 read plan byte count is inconsistent");
            }
            return plan;
        };
#ifdef SATVIEW_HAS_CUDA
        const auto plan_element_count = [](const satview::ReadPlan& plan) {
            return static_cast<std::size_t>(
                plan.aligned.height * plan.aligned.width);
        };
#endif

#ifdef SATVIEW_HAS_CUDA
        int device = 0;
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, device),
                   "cudaGetDeviceProperties");
        if (properties.major != 12) {
            throw std::runtime_error("sat-bench requires an sm_120-class GPU");
        }

        // Declaration order is intentional: reverse destruction first drains
        // the stream, then releases pinned staging, then frees device memory.
        // This also covers an exception after a copy was queued but before its
        // staging lease could be marked in flight.
        DeviceAllocation device_input(maximum_bytes, "cudaMalloc input");
        DeviceAllocation device_output(output_bytes, "cudaMalloc output");
        satview::gpu::PinnedRing ring(kPipelineRingSlots, maximum_bytes);
        CudaStream stream;

        if (arguments.pipeline) {
            // Warm HDF5's raw-chunk cache and the CUDA context outside the
            // measured interval. There is one stream drain for the full warmup.
            for (std::size_t iteration = 0; iteration < arguments.warmup;
                 ++iteration) {
                const auto plan = make_plan(iteration);
                auto filling = ring.acquire_for_fill();
                product.read_into(
                    plan, filling.bytes().first(plan.expected_bytes));
                filling.publish_ready(plan.expected_bytes);
                auto ready = ring.acquire_ready();
                check_cuda(
                    cudaMemcpyAsync(device_input.get(), ready.bytes().data(),
                                    plan.expected_bytes, cudaMemcpyHostToDevice,
                                    stream.get()),
                    "warmup cudaMemcpyAsync");
                ready.mark_in_flight(stream.get());
                check_cuda(
                    launch_selected_transform(
                        layer, device_input.get(),
                        static_cast<float*>(device_output.get()),
                        plan_element_count(plan), stream.get()),
                    "warmup scientific transform launch");
            }
            stream.synchronize("wait pipeline warmup");
            static_cast<void>(ring.reclaim_completed());

            // All measured read plans and result storage are allocated before
            // the reader starts. CUDA timing resources are a constant-size
            // two-bank pool independent of --chunks.
            std::vector<satview::ReadPlan> plans;
            plans.reserve(arguments.chunks);
            std::vector<Sample> pipeline_samples(arguments.chunks);
            for (std::size_t index = 0; index < arguments.chunks; ++index) {
                plans.push_back(make_plan(arguments.warmup + index));
                pipeline_samples[index].bytes = plans.back().expected_bytes;
            }

            constexpr std::size_t timing_bank_count = 2;
            std::vector<PipelineEvents> timing_events(
                timing_bank_count * kPipelineTimingBatchChunks);
            struct TimingBankState {
                std::size_t sample_begin = 0;
                std::size_t count = 0;
                bool active = false;
            };
            std::array<TimingBankState, timing_bank_count> timing_banks{};
            const auto event_at = [&](const std::size_t bank,
                                      const std::size_t slot)
                -> PipelineEvents& {
                return timing_events[
                    bank * kPipelineTimingBatchChunks + slot];
            };
            const auto harvest_bank = [&](const std::size_t bank) {
                auto& state = timing_banks[bank];
                for (std::size_t slot = 0; slot < state.count; ++slot) {
                    auto& timing = event_at(bank, slot);
                    auto& sample =
                        pipeline_samples[state.sample_begin + slot];
                    sample.h2d_ms = timing.h2d_stop.elapsed_from(
                        timing.h2d_start, "pipeline H2D elapsed");
                    sample.kernel_ms = timing.kernel_stop.elapsed_from(
                        timing.kernel_start, "pipeline kernel elapsed");
                }
                state = {};
            };

            std::stop_source stop_source;
            std::exception_ptr reader_error;
            std::latch start_gate(1);
            bool gate_open = false;
            const auto open_gate = [&]() noexcept {
                if (!gate_open) {
                    gate_open = true;
                    start_gate.count_down();
                }
            };
            std::jthread reader([&] {
                start_gate.wait();
                try {
                    const auto stop = stop_source.get_token();
                    for (std::size_t index = 0; index < plans.size(); ++index) {
                        auto filling = ring.acquire_for_fill(stop);
                        if (!filling) return;
                        const auto read_start = Clock::now();
                        product.read_into(
                            plans[index],
                            filling.bytes().first(plans[index].expected_bytes));
                        pipeline_samples[index].hdf5_ms =
                            std::chrono::duration<double, std::milli>(
                                Clock::now() - read_start)
                                .count();
                        filling.publish_ready(plans[index].expected_bytes);
                    }
                } catch (...) {
                    reader_error = std::current_exception();
                    static_cast<void>(stop_source.request_stop());
                }
            });

            double pipeline_ms = 0.0;
            try {
                const auto pipeline_start = Clock::now();
                open_gate();
                const auto stop = stop_source.get_token();
                std::size_t consumed = 0;
                std::size_t batch_number = 0;
                std::size_t final_bank = 0;
                std::size_t final_bank_count = 0;
                Clock::time_point pipeline_complete{};

                while (consumed < plans.size()) {
                    const std::size_t bank =
                        batch_number % timing_bank_count;
                    auto& bank_state = timing_banks[bank];
                    if (bank_state.active) {
                        event_at(bank, bank_state.count - 1)
                            .kernel_stop.synchronize(
                                "wait reusable pipeline timing bank");
                        harvest_bank(bank);
                    }

                    const std::size_t batch_count = std::min(
                        kPipelineTimingBatchChunks,
                        plans.size() - consumed);
                    std::size_t queued = 0;
                    for (; queued < batch_count; ++queued) {
                        auto ready = ring.acquire_ready(stop);
                        if (!ready) break;
                        const std::size_t sample_index = consumed + queued;
                        const auto& plan = plans[sample_index];
                        auto& timing = event_at(bank, queued);

                        timing.h2d_start.record(
                            stream.get(), "record pipeline H2D start");
                        check_cuda(
                            cudaMemcpyAsync(
                                device_input.get(), ready.bytes().data(),
                                plan.expected_bytes, cudaMemcpyHostToDevice,
                                stream.get()),
                            "pipeline cudaMemcpyAsync");
                        timing.h2d_stop.record(
                            stream.get(), "record pipeline H2D stop");
                        ready.mark_in_flight(stream.get());
                        // Start after PinnedRing's completion marker so the
                        // transform duration excludes staging bookkeeping.
                        timing.kernel_start.record(
                            stream.get(), "record pipeline kernel start");
                        check_cuda(
                            launch_selected_transform(
                                layer, device_input.get(),
                                static_cast<float*>(device_output.get()),
                                plan_element_count(plan), stream.get()),
                            "pipeline scientific transform launch");
                        timing.kernel_stop.record(
                            stream.get(), "record pipeline kernel stop");
                    }

                    if (queued != batch_count) {
                        static_cast<void>(stop_source.request_stop());
                        if (reader.joinable()) reader.join();
                        if (reader_error) {
                            std::rethrow_exception(reader_error);
                        }
                        throw std::runtime_error(
                            "pipeline reader stopped before all chunks were published");
                    }

                    bank_state.sample_begin = consumed;
                    bank_state.count = batch_count;
                    bank_state.active = true;
                    final_bank = bank;
                    final_bank_count = batch_count;
                    consumed += batch_count;
                    ++batch_number;
                }

                // The last event is ordered after every operation on the one
                // benchmark stream, so this single final wait completes both
                // active timing banks without a device-wide synchronization.
                event_at(final_bank, final_bank_count - 1)
                    .kernel_stop.synchronize("wait pipeline completion");
                pipeline_complete = Clock::now();
                for (std::size_t bank = 0; bank < timing_bank_count; ++bank) {
                    if (timing_banks[bank].active) harvest_bank(bank);
                }
                pipeline_ms = std::chrono::duration<double, std::milli>(
                    pipeline_complete - pipeline_start)
                                  .count();

                if (reader.joinable()) reader.join();
                if (reader_error) std::rethrow_exception(reader_error);
            } catch (...) {
                open_gate();
                static_cast<void>(stop_source.request_stop());
                static_cast<void>(cudaStreamSynchronize(stream.get()));
                try {
                    static_cast<void>(ring.reclaim_completed());
                } catch (...) {
                    // Preserve the original benchmark failure.
                }
                if (reader.joinable()) reader.join();
                throw;
            }

            if (!(pipeline_ms > 0.0)) {
                throw std::runtime_error(
                    "pipeline duration was not positive at clock resolution");
            }
            std::vector<double> pipeline_hdf5;
            std::vector<double> pipeline_h2d;
            std::vector<double> pipeline_kernel;
            pipeline_hdf5.reserve(pipeline_samples.size());
            pipeline_h2d.reserve(pipeline_samples.size());
            pipeline_kernel.reserve(pipeline_samples.size());
            long double pipeline_bytes = 0.0L;
            for (const auto& sample : pipeline_samples) {
                pipeline_hdf5.push_back(sample.hdf5_ms);
                pipeline_h2d.push_back(sample.h2d_ms);
                pipeline_kernel.push_back(sample.kernel_ms);
                pipeline_bytes += static_cast<long double>(sample.bytes);
            }
            std::sort(pipeline_hdf5.begin(), pipeline_hdf5.end());
            std::sort(pipeline_h2d.begin(), pipeline_h2d.end());
            std::sort(pipeline_kernel.begin(), pipeline_kernel.end());

            const double pipeline_seconds = pipeline_ms / 1000.0;
            const double chunks_per_second =
                static_cast<double>(pipeline_samples.size()) /
                pipeline_seconds;
            const double pipeline_gib_per_second =
                static_cast<double>(pipeline_bytes / 1073741824.0L) /
                pipeline_seconds;

            if (arguments.csv) {
                std::cout
                    << "file,layer,mode,chunks,timing_banks,timing_bank_chunks,"
                       "open_ms,pipeline_ms,pipeline_chunks_s,"
                       "pipeline_logical_gib_s,hdf5_p50_ms,hdf5_p95_ms,"
                       "h2d_p50_ms,h2d_p95_ms,kernel_p50_ms,kernel_p95_ms\n"
                    << '"' << arguments.file.string() << "\",\"" << layer.path
                    << "\",pipeline," << pipeline_samples.size() << ','
                    << timing_bank_count << ','
                    << kPipelineTimingBatchChunks << ',' << open_ms << ','
                    << pipeline_ms << ',' << chunks_per_second << ','
                    << pipeline_gib_per_second << ','
                    << percentile(pipeline_hdf5, 0.50) << ','
                    << percentile(pipeline_hdf5, 0.95) << ','
                    << percentile(pipeline_h2d, 0.50) << ','
                    << percentile(pipeline_h2d, 0.95) << ','
                    << percentile(pipeline_kernel, 0.50) << ','
                    << percentile(pipeline_kernel, 0.95) << '\n';
            } else {
                std::cout
                    << "Layer: " << layer.path << "\n"
                    << "Mode: overlapped pipeline (1 HDF5 reader, "
                    << kPipelineRingSlots << " pinned slots)\n"
                    << "Timing pool: " << timing_bank_count << " x "
                    << kPipelineTimingBatchChunks
                    << " chunks (ping-pong; no per-chunk waits)\n"
                    << "Open/catalog: " << open_ms << " ms\n"
                    << "Pipeline wall time: " << pipeline_ms << " ms\n"
                    << "Pipeline throughput: " << chunks_per_second
                    << " chunks/s, " << pipeline_gib_per_second
                    << " logical GiB/s\n"
                    << "HDF5 read+decode p50/p95: "
                    << percentile(pipeline_hdf5, 0.50) << " / "
                    << percentile(pipeline_hdf5, 0.95) << " ms\n"
                    << "Pinned H2D p50/p95: "
                    << percentile(pipeline_h2d, 0.50) << " / "
                    << percentile(pipeline_h2d, 0.95) << " ms\n"
                    << "CUDA transform p50/p95: "
                    << percentile(pipeline_kernel, 0.50) << " / "
                    << percentile(pipeline_kernel, 0.95) << " ms\n";
            }
            return 0;
        }

        CudaEvent h2d_start;
        CudaEvent h2d_stop;
        CudaEvent kernel_start;
        CudaEvent kernel_stop;
#else
        std::vector<std::byte> host(maximum_bytes);
#endif

        std::vector<Sample> samples;
        samples.reserve(arguments.chunks);
        for (std::size_t iteration = 0; iteration < total_iterations;
             ++iteration) {
            const auto plan = make_plan(iteration);
            Sample sample;
            sample.bytes = plan.expected_bytes;

#ifdef SATVIEW_HAS_CUDA
            auto filling = ring.acquire_for_fill();
            const auto read_start = Clock::now();
            product.read_into(
                plan, filling.bytes().first(plan.expected_bytes));
            sample.hdf5_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - read_start)
                                 .count();
            filling.publish_ready(plan.expected_bytes);
            auto ready = ring.acquire_ready();

            h2d_start.record(stream.get(), "record H2D start");
            check_cuda(
                cudaMemcpyAsync(device_input.get(), ready.bytes().data(),
                                plan.expected_bytes, cudaMemcpyHostToDevice,
                                stream.get()),
                "cudaMemcpyAsync");
            h2d_stop.record(stream.get(), "record H2D stop");
            ready.mark_in_flight(stream.get());
            kernel_start.record(stream.get(), "record kernel start");
            check_cuda(
                launch_selected_transform(
                    layer, device_input.get(),
                    static_cast<float*>(device_output.get()),
                    plan_element_count(plan), stream.get()),
                "scientific transform launch");
            kernel_stop.record(stream.get(), "record kernel stop");
            kernel_stop.synchronize("wait benchmark sample");
            sample.h2d_ms =
                h2d_stop.elapsed_from(h2d_start, "H2D elapsed");
            sample.kernel_ms =
                kernel_stop.elapsed_from(kernel_start, "kernel elapsed");
            static_cast<void>(ring.reclaim_completed());
#else
            const auto read_start = Clock::now();
            product.read_into(
                plan, std::span<std::byte>(host.data(), plan.expected_bytes));
            sample.hdf5_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - read_start)
                                 .count();
#endif
            if (iteration >= arguments.warmup) samples.push_back(sample);
        }
        std::vector<double> hdf5, h2d, kernel;
        long double total_bytes = 0.0L;
        for (const auto& sample : samples) {
            hdf5.push_back(sample.hdf5_ms);
            h2d.push_back(sample.h2d_ms);
            kernel.push_back(sample.kernel_ms);
            total_bytes += static_cast<long double>(sample.bytes);
        }
        std::sort(hdf5.begin(), hdf5.end());
        std::sort(h2d.begin(), h2d.end());
        std::sort(kernel.begin(), kernel.end());
        const double total_hdf5_seconds = std::accumulate(hdf5.begin(), hdf5.end(), 0.0) / 1000.0;
        const double gib_per_second = total_hdf5_seconds > 0.0
            ? static_cast<double>(total_bytes / 1073741824.0L) / total_hdf5_seconds : 0.0;

        if (arguments.csv) {
            std::cout << "file,layer,chunks,open_ms,hdf5_p50_ms,hdf5_p95_ms,hdf5_gib_s,h2d_p50_ms,h2d_p95_ms,kernel_p50_ms,kernel_p95_ms\n"
                      << '"' << arguments.file.string() << "\",\"" << layer.path << "\"," << samples.size() << ','
                      << open_ms << ',' << percentile(hdf5, 0.50) << ',' << percentile(hdf5, 0.95) << ',' << gib_per_second << ','
                      << percentile(h2d, 0.50) << ',' << percentile(h2d, 0.95) << ','
                      << percentile(kernel, 0.50) << ',' << percentile(kernel, 0.95) << '\n';
        } else {
            std::cout << "Layer: " << layer.path << "\n"
                      << "Open/catalog: " << open_ms << " ms\n"
                      << "HDF5 read+decode p50/p95: " << percentile(hdf5, 0.50) << " / " << percentile(hdf5, 0.95) << " ms\n"
                      << "HDF5 effective throughput: " << gib_per_second << " GiB/s\n";
#ifdef SATVIEW_HAS_CUDA
            std::cout << "Pinned H2D p50/p95: " << percentile(h2d, 0.50) << " / " << percentile(h2d, 0.95) << " ms\n"
                      << "CUDA transform p50/p95: " << percentile(kernel, 0.50) << " / " << percentile(kernel, 0.95) << " ms\n";
#endif
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sat-bench: " << error.what() << '\n';
        usage();
        return 1;
    }
}
