#pragma once

#if !defined(_WIN32)
#error "satview::gpu Vulkan/CUDA interop is available only on Windows"
#endif

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

#include <cstdint>

namespace satview::gpu {

// A Vulkan-owned, device-local buffer allocation imported and mapped by CUDA.
//
// Construction requires the current CUDA device to identify the same adapter
// (LUID, node mask, and UUID) as physical_device. The VkDevice must have been
// created with VK_KHR_external_memory_win32 enabled.
//
// A typical zero-copy ownership sequence is:
//
//   ExportedBuffer pixels{physical, device, bytes, usage};
//   InteropTimeline ownership{physical, device};
//
//   // Vulkan submission uses pixels.buffer(), then signals timeline value 1.
//   ownership.enqueue_cuda_wait(stream, 1);
//   launch_cuda_work(pixels.cuda_ptr(), stream);
//   ownership.enqueue_cuda_signal(stream, 2);
//   // The next Vulkan submission waits for timeline value 2 before reading.
//
// Use monotonically increasing timeline values. Appropriate Vulkan pipeline
// barriers are still required around Vulkan accesses. Before destroying either
// object, the owner must ensure its queued Vulkan and CUDA work has completed;
// destructors deliberately perform no device-wide or queue-wide synchronization.
class ExportedBuffer final {
 public:
  ExportedBuffer(VkPhysicalDevice physical_device, VkDevice device,
                 VkDeviceSize size, VkBufferUsageFlags usage);
  ~ExportedBuffer() noexcept;

  ExportedBuffer(ExportedBuffer&& other) noexcept;
  ExportedBuffer& operator=(ExportedBuffer&& other) noexcept;

  ExportedBuffer(const ExportedBuffer&) = delete;
  ExportedBuffer& operator=(const ExportedBuffer&) = delete;

  [[nodiscard]] VkBuffer buffer() const noexcept;
  [[nodiscard]] VkDeviceSize size() const noexcept;
  [[nodiscard]] void* cuda_ptr() const noexcept;

 private:
  void release() noexcept;

  VkDevice device_ = VK_NULL_HANDLE;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  cudaExternalMemory_t cuda_memory_ = nullptr;
  void* cuda_ptr_ = nullptr;
  VkDeviceSize size_ = 0;
  int cuda_device_ = -1;
};

// A Vulkan timeline semaphore imported into CUDA.
//
// The VkDevice must have timeline semaphores enabled as well as
// VK_KHR_external_semaphore_win32. Waits and signals are enqueued on the
// supplied CUDA stream and do not block the host.
class InteropTimeline final {
 public:
  InteropTimeline(VkPhysicalDevice physical_device, VkDevice device,
                  std::uint64_t initial_value = 0);
  ~InteropTimeline() noexcept;

  InteropTimeline(InteropTimeline&& other) noexcept;
  InteropTimeline& operator=(InteropTimeline&& other) noexcept;

  InteropTimeline(const InteropTimeline&) = delete;
  InteropTimeline& operator=(const InteropTimeline&) = delete;

  void enqueue_cuda_wait(cudaStream_t stream, std::uint64_t value);
  void enqueue_cuda_signal(cudaStream_t stream, std::uint64_t value);

  [[nodiscard]] VkSemaphore vk_semaphore() const noexcept;

 private:
  void release() noexcept;

  VkDevice device_ = VK_NULL_HANDLE;
  VkSemaphore semaphore_ = VK_NULL_HANDLE;
  cudaExternalSemaphore_t cuda_semaphore_ = nullptr;
  int cuda_device_ = -1;
};

}  // namespace satview::gpu
