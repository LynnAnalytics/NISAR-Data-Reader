#include "satview/experimental/scientific.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace satview::experimental {
namespace {

void validate_request(
    const PageRequest& request, const std::span<float> output) {
  if (request.width == 0 || request.height == 0 ||
      request.width > std::numeric_limits<std::size_t>::max() /
          request.height) {
    throw std::invalid_argument("SYCL page dimensions are invalid");
  }
  const std::size_t count = request.width * request.height;
  const std::size_t element_size =
      request.input_kind == InputKind::complex_float32
      ? sizeof(cpu::Complex32)
      : sizeof(float);
  if (count > std::numeric_limits<std::size_t>::max() / element_size ||
      request.science.size() != count * element_size ||
      output.size() != count ||
      (!request.validity.empty() && request.validity.size() != count)) {
    throw std::invalid_argument("SYCL page buffers do not match");
  }
  if (!valid_filter_configuration(request)) {
    throw std::invalid_argument("SYCL speckle options are invalid");
  }
}

template <typename T>
class DeviceAllocation final {
 public:
  DeviceAllocation(sycl::queue& queue, const std::size_t count)
      : queue_(&queue), data_(sycl::malloc_device<T>(count, queue)) {
    if (data_ == nullptr) {
      throw std::runtime_error("oneAPI/SYCL device allocation failed");
    }
  }

  ~DeviceAllocation() {
    if (data_ != nullptr) {
      sycl::free(data_, *queue_);
    }
  }

  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  [[nodiscard]] T* get() const noexcept { return data_; }

 private:
  sycl::queue* queue_ = nullptr;
  T* data_ = nullptr;
};

}  // namespace

bool sycl_runtime_available(std::string& reason) noexcept {
  try {
    const auto devices =
        sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) {
      reason = "no SYCL GPU device was found";
      return false;
    }
    reason.clear();
    return true;
  } catch (const sycl::exception& error) {
    reason = error.what();
    return false;
  } catch (...) {
    reason = "SYCL runtime initialization failed";
    return false;
  }
}

void process_sycl(
    const PageRequest& request, const std::span<float> output) {
  validate_request(request, output);
  try {
    sycl::queue queue{
        sycl::gpu_selector_v,
        sycl::property::queue::in_order{}};
    const std::size_t count = request.width * request.height;
    DeviceAllocation<std::byte> device_science(queue, request.science.size());
    DeviceAllocation<float> device_output(queue, count);
    std::optional<DeviceAllocation<float>> device_scratch;
    std::optional<DeviceAllocation<std::uint8_t>> device_validity;
    if (request.filter_enabled) {
      device_scratch.emplace(queue, count);
    }
    if (!request.validity.empty()) {
      device_validity.emplace(queue, count);
    }
    queue.memcpy(
        device_science.get(), request.science.data(), request.science.size());
    if (device_validity.has_value()) {
      queue.memcpy(
          device_validity->get(), request.validity.data(), count);
    }

    auto* const transformed = request.filter_enabled
        ? device_scratch->get()
        : device_output.get();
    const auto* const validity = device_validity.has_value()
        ? device_validity->get()
        : nullptr;
    constexpr float invalid = std::numeric_limits<float>::quiet_NaN();
    constexpr float epsilon = cpu::kTransformEpsilon;

    if (request.input_kind == InputKind::complex_float32) {
      const auto* const input =
          reinterpret_cast<const cpu::Complex32*>(device_science.get());
      const auto transform = request.complex_transform;
      queue.parallel_for(sycl::range<1>(count), [=](const sycl::id<1> id) {
        const std::size_t index = id[0];
        const auto sample = input[index];
        const bool mask_is_valid = validity == nullptr ||
            (validity[index] != 0 && validity[index] != 255);
        if (!mask_is_valid || !sycl::isfinite(sample.real) ||
            !sycl::isfinite(sample.imaginary)) {
          transformed[index] = invalid;
          return;
        }
        const float power = sycl::fma(
            sample.real, sample.real,
            sample.imaginary * sample.imaginary);
        float value = invalid;
        switch (transform) {
          case cpu::ComplexTransform::amplitude:
            if (sycl::isfinite(power) && power > 0.0F) {
              value = sycl::sqrt(power);
            } else if (sample.real == 0.0F && sample.imaginary == 0.0F) {
              value = 0.0F;
            } else {
              value = sycl::hypot(sample.real, sample.imaginary);
            }
            break;
          case cpu::ComplexTransform::power:
            value = power;
            break;
          case cpu::ComplexTransform::power_db:
            if (sycl::isfinite(power)) {
              value = 10.0F * sycl::log10(sycl::fmax(power, epsilon));
            } else {
              const float high = sycl::fmax(
                  sycl::fabs(sample.real), sycl::fabs(sample.imaginary));
              const float low = sycl::fmin(
                  sycl::fabs(sample.real), sycl::fabs(sample.imaginary));
              const float ratio = low / high;
              value = sycl::fmax(
                  20.0F * sycl::log10(high) +
                      10.0F * sycl::log10(
                          sycl::fma(ratio, ratio, 1.0F)),
                  10.0F * sycl::log10(epsilon));
            }
            break;
          case cpu::ComplexTransform::phase:
            value = sycl::atan2(sample.imaginary, sample.real);
            break;
          case cpu::ComplexTransform::real:
            value = sample.real;
            break;
          case cpu::ComplexTransform::imaginary:
            value = sample.imaginary;
            break;
        }
        transformed[index] = sycl::isfinite(value) ? value : invalid;
      });
    } else {
      const auto* const input =
          reinterpret_cast<const float*>(device_science.get());
      const auto transform = request.real_transform;
      queue.parallel_for(sycl::range<1>(count), [=](const sycl::id<1> id) {
        const std::size_t index = id[0];
        const float sample = input[index];
        const bool mask_is_valid = validity == nullptr ||
            (validity[index] != 0 && validity[index] != 255);
        if (!mask_is_valid || !sycl::isfinite(sample) || sample < 0.0F) {
          transformed[index] = invalid;
          return;
        }
        transformed[index] = transform == cpu::RealTransform::linear
            ? sample
            : 10.0F * sycl::log10(sycl::fmax(sample, epsilon));
      });
    }

    if (request.filter_enabled) {
      const auto options = request.speckle;
      const std::size_t width = request.width;
      const std::size_t height = request.height;
      auto* const filtered_output = device_output.get();
      queue.parallel_for(sycl::range<1>(count), [=](const sycl::id<1> id) {
        const std::size_t index = id[0];
        const std::size_t row = index / width;
        const std::size_t column = index - row * width;
        const float center_sample = transformed[index];
        const bool center_mask_valid = validity == nullptr ||
            (validity[index] != 0 && validity[index] != 255);
        const bool center_source_valid = sycl::isfinite(center_sample) &&
            (options.domain == cpu::SpeckleDomain::power_db ||
             center_sample >= 0.0F);
        if (!center_mask_valid || !center_source_valid) {
          filtered_output[index] = invalid;
          return;
        }
        const std::size_t radius = options.window_size / 2;
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
            const float sample = transformed[neighbor];
            const bool neighbor_mask_valid = validity == nullptr ||
                (validity[neighbor] != 0 && validity[neighbor] != 255);
            const bool neighbor_source_valid = sycl::isfinite(sample) &&
                (options.domain == cpu::SpeckleDomain::power_db ||
                 sample >= 0.0F);
            if (neighbor_mask_valid && neighbor_source_valid) {
              scale = sycl::fmax(scale, sample);
            }
          }
        }

        float mean = 0.0F;
        float m2 = 0.0F;
        std::uint32_t valid_count = 0;
        for (std::size_t y = first_row; y <= last_row; ++y) {
          for (std::size_t x = first_column; x <= last_column; ++x) {
            const std::size_t neighbor = y * width + x;
            const float sample = transformed[neighbor];
            const bool neighbor_mask_valid = validity == nullptr ||
                (validity[neighbor] != 0 && validity[neighbor] != 255);
            const bool neighbor_source_valid = sycl::isfinite(sample) &&
                (options.domain == cpu::SpeckleDomain::power_db ||
                 sample >= 0.0F);
            if (!neighbor_mask_valid || !neighbor_source_valid) {
              continue;
            }
            float normalized = 0.0F;
            if (options.domain == cpu::SpeckleDomain::linear_power) {
              normalized = scale == 0.0F ? 0.0F : sample / scale;
            } else if (options.domain == cpu::SpeckleDomain::amplitude) {
              const float ratio = scale == 0.0F ? 0.0F : sample / scale;
              normalized = ratio * ratio;
            } else {
              constexpr float log2_ten_over_ten = 0.3321928094887362F;
              normalized = sycl::exp2(
                  (sample - scale) * log2_ten_over_ten);
            }
            ++valid_count;
            const float delta = normalized - mean;
            mean += delta / static_cast<float>(valid_count);
            if (options.filter == cpu::SpeckleFilter::lee) {
              m2 = sycl::fma(delta, normalized - mean, m2);
            }
          }
        }

        float filtered = mean;
        if (options.filter == cpu::SpeckleFilter::lee) {
          const float variance = sycl::fmax(
              m2 / static_cast<float>(valid_count), 0.0F);
          const float noise =
              mean * mean / options.equivalent_number_of_looks;
          float weight = 0.0F;
          if (variance > noise && variance > 0.0F) {
            weight = (variance - noise) / variance;
          }
          weight = sycl::fmin(sycl::fmax(weight, 0.0F), 1.0F);
          float center = 0.0F;
          if (options.domain == cpu::SpeckleDomain::linear_power) {
            center = scale == 0.0F ? 0.0F : center_sample / scale;
          } else if (options.domain == cpu::SpeckleDomain::amplitude) {
            const float ratio = scale == 0.0F
                ? 0.0F
                : center_sample / scale;
            center = ratio * ratio;
          } else {
            constexpr float log2_ten_over_ten = 0.3321928094887362F;
            center = sycl::exp2(
                (center_sample - scale) * log2_ten_over_ten);
          }
          filtered = sycl::fma(weight, center - mean, mean);
        }
        filtered = sycl::fmin(sycl::fmax(filtered, 0.0F), 1.0F);
        float result = 0.0F;
        if (options.domain == cpu::SpeckleDomain::linear_power) {
          result = filtered * scale;
        } else if (options.domain == cpu::SpeckleDomain::amplitude) {
          result = sycl::sqrt(filtered) * scale;
        } else {
          const float floor_db =
              10.0F * sycl::log10(options.power_epsilon);
          result = filtered <= 0.0F
              ? floor_db
              : sycl::fmax(
                    scale + 10.0F * sycl::log10(filtered), floor_db);
        }
        filtered_output[index] =
            sycl::isfinite(result) ? result : invalid;
      });
    }

    queue.memcpy(output.data(), device_output.get(), output.size_bytes())
        .wait_and_throw();
  } catch (const sycl::exception& error) {
    throw std::runtime_error(
        std::string("oneAPI/SYCL scientific processing failed: ") +
        error.what());
  }
}

}  // namespace satview::experimental
