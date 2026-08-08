#include "satview/experimental/scientific.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace satview::experimental {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;

struct DeviceComplex {
  float real;
  float imaginary;
};

static_assert(sizeof(DeviceComplex) == sizeof(cpu::Complex32));

[[noreturn]] void throw_hip(const char* operation, const hipError_t error) {
  throw std::runtime_error(
      std::string(operation) + ": " + hipGetErrorString(error));
}

void check_hip(const hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw_hip(operation, error);
  }
}

class DeviceAllocation final {
 public:
  explicit DeviceAllocation(const std::size_t bytes) {
    check_hip(hipMalloc(&data_, bytes), "HIP allocation");
  }

  ~DeviceAllocation() {
    if (data_ != nullptr) {
      static_cast<void>(hipFree(data_));
    }
  }

  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  [[nodiscard]] void* get() const noexcept { return data_; }

 private:
  void* data_ = nullptr;
};

void validate_request(
    const PageRequest& request, const std::span<float> output) {
  if (request.width == 0 || request.height == 0 ||
      request.width > std::numeric_limits<std::size_t>::max() /
          request.height) {
    throw std::invalid_argument("HIP page dimensions are invalid");
  }
  const std::size_t count = request.width * request.height;
  const std::size_t element_size =
      request.input_kind == InputKind::complex_float32
      ? sizeof(DeviceComplex)
      : sizeof(float);
  if (count > std::numeric_limits<std::size_t>::max() / element_size ||
      request.science.size() != count * element_size ||
      output.size() != count ||
      (!request.validity.empty() && request.validity.size() != count)) {
    throw std::invalid_argument("HIP page buffers do not match");
  }
  if (request.filter_enabled) {
    if ((request.speckle.filter != cpu::SpeckleFilter::boxcar &&
         request.speckle.filter != cpu::SpeckleFilter::lee) ||
        (request.speckle.window_size != 3 &&
         request.speckle.window_size != 5 &&
         request.speckle.window_size != 7) ||
        !std::isfinite(request.speckle.power_epsilon) ||
        request.speckle.power_epsilon <= 0.0F ||
        (request.speckle.filter == cpu::SpeckleFilter::lee &&
         (!std::isfinite(request.speckle.equivalent_number_of_looks) ||
          request.speckle.equivalent_number_of_looks <= 0.0F))) {
      throw std::invalid_argument("HIP speckle options are invalid");
    }
  }
}

__device__ __forceinline__ float quiet_nan() noexcept {
  return nanf("");
}

__device__ __forceinline__ bool mask_valid(
    const std::uint8_t* validity, const std::size_t index) noexcept {
  if (validity == nullptr) {
    return true;
  }
  const std::uint8_t value = validity[index];
  return value != 0 && value != 255;
}

__device__ __forceinline__ float robust_magnitude(
    const DeviceComplex sample) noexcept {
  const float power =
      fmaf(sample.real, sample.real, sample.imaginary * sample.imaginary);
  if (isfinite(power) && power > 0.0F) {
    return sqrtf(power);
  }
  if (sample.real == 0.0F && sample.imaginary == 0.0F) {
    return 0.0F;
  }
  return hypotf(sample.real, sample.imaginary);
}

__global__ void transform_complex_kernel(
    const DeviceComplex* input,
    float* output,
    const std::size_t count,
    const cpu::ComplexTransform transform,
    const std::uint8_t* validity,
    const float epsilon) {
  const std::size_t first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = first; index < count; index += stride) {
    const DeviceComplex sample = input[index];
    if (!mask_valid(validity, index) || !isfinite(sample.real) ||
        !isfinite(sample.imaginary)) {
      output[index] = quiet_nan();
      continue;
    }
    const float power =
        fmaf(sample.real, sample.real, sample.imaginary * sample.imaginary);
    float value = quiet_nan();
    switch (transform) {
      case cpu::ComplexTransform::amplitude:
        value = robust_magnitude(sample);
        break;
      case cpu::ComplexTransform::power:
        value = power;
        break;
      case cpu::ComplexTransform::power_db:
        if (isfinite(power)) {
          value = 10.0F * log10f(fmaxf(power, epsilon));
        } else {
          const float high =
              fmaxf(fabsf(sample.real), fabsf(sample.imaginary));
          const float low =
              fminf(fabsf(sample.real), fabsf(sample.imaginary));
          const float ratio = low / high;
          value = fmaxf(
              20.0F * log10f(high) +
                  10.0F * log10f(fmaf(ratio, ratio, 1.0F)),
              10.0F * log10f(epsilon));
        }
        break;
      case cpu::ComplexTransform::phase:
        value = atan2f(sample.imaginary, sample.real);
        break;
      case cpu::ComplexTransform::real:
        value = sample.real;
        break;
      case cpu::ComplexTransform::imaginary:
        value = sample.imaginary;
        break;
    }
    output[index] = isfinite(value) ? value : quiet_nan();
  }
}

__global__ void transform_real_kernel(
    const float* input,
    float* output,
    const std::size_t count,
    const cpu::RealTransform transform,
    const std::uint8_t* validity,
    const float epsilon) {
  const std::size_t first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = first; index < count; index += stride) {
    const float sample = input[index];
    if (!mask_valid(validity, index) || !isfinite(sample) || sample < 0.0F) {
      output[index] = quiet_nan();
      continue;
    }
    output[index] = transform == cpu::RealTransform::linear
        ? sample
        : 10.0F * log10f(fmaxf(sample, epsilon));
  }
}

__device__ __forceinline__ bool source_valid(
    const float sample, const cpu::SpeckleDomain domain) noexcept {
  return isfinite(sample) &&
      (domain == cpu::SpeckleDomain::power_db || sample >= 0.0F);
}

__device__ __forceinline__ float normalized_power(
    const float sample,
    const float scale,
    const cpu::SpeckleDomain domain) noexcept {
  if (domain == cpu::SpeckleDomain::linear_power) {
    return scale == 0.0F ? 0.0F : sample / scale;
  }
  if (domain == cpu::SpeckleDomain::amplitude) {
    if (scale == 0.0F) {
      return 0.0F;
    }
    const float ratio = sample / scale;
    return ratio * ratio;
  }
  constexpr float kLog2TenOverTen = 0.3321928094887362F;
  return exp2f((sample - scale) * kLog2TenOverTen);
}

__device__ __forceinline__ float from_normalized_power(
    const float normalized,
    const float scale,
    const cpu::SpeckleDomain domain,
    const float epsilon) noexcept {
  if (domain == cpu::SpeckleDomain::linear_power) {
    return normalized * scale;
  }
  if (domain == cpu::SpeckleDomain::amplitude) {
    return sqrtf(normalized) * scale;
  }
  const float floor_db = 10.0F * log10f(epsilon);
  return normalized <= 0.0F
      ? floor_db
      : fmaxf(scale + 10.0F * log10f(normalized), floor_db);
}

__global__ void speckle_kernel(
    const float* input,
    float* output,
    const std::size_t width,
    const std::size_t height,
    const std::uint8_t* validity,
    const cpu::SpeckleOptions options) {
  const std::size_t first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t count = width * height;
  const std::size_t radius = options.window_size / 2;
  for (std::size_t index = first; index < count; index += stride) {
    const std::size_t row = index / width;
    const std::size_t column = index - row * width;
    const float center_sample = input[index];
    if (!mask_valid(validity, index) ||
        !source_valid(center_sample, options.domain)) {
      output[index] = quiet_nan();
      continue;
    }
    const std::size_t first_row = row > radius ? row - radius : 0;
    const std::size_t last_row =
        row + radius < height ? row + radius : height - 1;
    const std::size_t first_column =
        column > radius ? column - radius : 0;
    const std::size_t last_column =
        column + radius < width ? column + radius : width - 1;
    float scale = center_sample;
    for (std::size_t y = first_row; y <= last_row; ++y) {
      for (std::size_t x = first_column; x <= last_column; ++x) {
        const std::size_t neighbor = y * width + x;
        const float sample = input[neighbor];
        if (mask_valid(validity, neighbor) &&
            source_valid(sample, options.domain)) {
          scale = fmaxf(scale, sample);
        }
      }
    }
    float mean = 0.0F;
    float m2 = 0.0F;
    unsigned int valid_count = 0;
    for (std::size_t y = first_row; y <= last_row; ++y) {
      for (std::size_t x = first_column; x <= last_column; ++x) {
        const std::size_t neighbor = y * width + x;
        const float sample = input[neighbor];
        if (!mask_valid(validity, neighbor) ||
            !source_valid(sample, options.domain)) {
          continue;
        }
        const float normalized =
            normalized_power(sample, scale, options.domain);
        ++valid_count;
        const float delta = normalized - mean;
        mean += delta / static_cast<float>(valid_count);
        if (options.filter == cpu::SpeckleFilter::lee) {
          m2 = fmaf(delta, normalized - mean, m2);
        }
      }
    }
    float filtered = mean;
    if (options.filter == cpu::SpeckleFilter::lee) {
      const float variance =
          fmaxf(m2 / static_cast<float>(valid_count), 0.0F);
      const float noise =
          mean * mean / options.equivalent_number_of_looks;
      float weight = 0.0F;
      if (variance > noise && variance > 0.0F) {
        weight = (variance - noise) / variance;
      }
      weight = fminf(fmaxf(weight, 0.0F), 1.0F);
      const float center =
          normalized_power(center_sample, scale, options.domain);
      filtered = fmaf(weight, center - mean, mean);
    }
    filtered = fminf(fmaxf(filtered, 0.0F), 1.0F);
    const float result = from_normalized_power(
        filtered, scale, options.domain, options.power_epsilon);
    output[index] = isfinite(result) ? result : quiet_nan();
  }
}

[[nodiscard]] unsigned int block_count(const std::size_t count) noexcept {
  const std::size_t requested =
      (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  return static_cast<unsigned int>(std::min<std::size_t>(requested, 65535));
}

}  // namespace

bool hip_runtime_available(std::string& reason) noexcept {
  int count = 0;
  const hipError_t result = hipGetDeviceCount(&count);
  if (result == hipSuccess && count > 0) {
    reason.clear();
    return true;
  }
  reason = result == hipSuccess
      ? "no HIP device was found"
      : hipGetErrorString(result);
  static_cast<void>(hipGetLastError());
  return false;
}

void process_hip(
    const PageRequest& request, const std::span<float> output) {
  validate_request(request, output);
  const std::size_t count = request.width * request.height;
  DeviceAllocation device_science(request.science.size());
  DeviceAllocation device_output(count * sizeof(float));
  std::optional<DeviceAllocation> device_scratch;
  std::optional<DeviceAllocation> device_validity;
  if (request.filter_enabled) {
    device_scratch.emplace(count * sizeof(float));
  }
  if (!request.validity.empty()) {
    device_validity.emplace(count * sizeof(std::uint8_t));
  }
  check_hip(
      hipMemcpy(
          device_science.get(), request.science.data(), request.science.size(),
          hipMemcpyHostToDevice),
      "copy scientific page to HIP");
  if (device_validity.has_value()) {
    check_hip(
        hipMemcpy(
            device_validity->get(), request.validity.data(), count,
            hipMemcpyHostToDevice),
        "copy validity page to HIP");
  }
  auto* const transformed = static_cast<float*>(
      request.filter_enabled ? device_scratch->get() : device_output.get());
  const auto* const validity = device_validity.has_value()
      ? static_cast<const std::uint8_t*>(device_validity->get())
      : nullptr;
  const unsigned int blocks = block_count(count);
  if (request.input_kind == InputKind::complex_float32) {
    hipLaunchKernelGGL(
        transform_complex_kernel,
        dim3(blocks),
        dim3(kThreadsPerBlock),
        0,
        0,
        static_cast<const DeviceComplex*>(device_science.get()),
        transformed,
        count,
        request.complex_transform,
        validity,
        cpu::kTransformEpsilon);
  } else {
    hipLaunchKernelGGL(
        transform_real_kernel,
        dim3(blocks),
        dim3(kThreadsPerBlock),
        0,
        0,
        static_cast<const float*>(device_science.get()),
        transformed,
        count,
        request.real_transform,
        validity,
        cpu::kTransformEpsilon);
  }
  check_hip(hipGetLastError(), "launch HIP scientific transform");
  if (request.filter_enabled) {
    hipLaunchKernelGGL(
        speckle_kernel,
        dim3(blocks),
        dim3(kThreadsPerBlock),
        0,
        0,
        transformed,
        static_cast<float*>(device_output.get()),
        request.width,
        request.height,
        validity,
        request.speckle);
    check_hip(hipGetLastError(), "launch HIP speckle filter");
  }
  check_hip(
      hipMemcpy(
          output.data(), device_output.get(), output.size_bytes(),
          hipMemcpyDeviceToHost),
      "copy HIP scientific page to host");
}

}  // namespace satview::experimental
