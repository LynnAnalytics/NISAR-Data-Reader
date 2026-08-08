#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include "satview/gpu/vulkan_interop.hpp"

#include <windows.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace satview::gpu {
namespace {

constexpr VkExternalMemoryHandleTypeFlagBits kExternalMemoryHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
constexpr VkExternalSemaphoreHandleTypeFlagBits kExternalSemaphoreHandleType =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

[[nodiscard]] const char* vk_result_name(const VkResult result) noexcept {
  switch (result) {
    case VK_SUCCESS:
      return "VK_SUCCESS";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
      return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
      return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST:
      return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED:
      return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_TOO_MANY_OBJECTS:
      return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
      return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FEATURE_NOT_PRESENT:
      return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
      return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_UNKNOWN:
      return "VK_ERROR_UNKNOWN";
    default:
      return "unrecognized VkResult";
  }
}

[[noreturn]] void throw_vk_error(const char* operation,
                                 const VkResult result) {
  std::string message{"Vulkan/CUDA interop: "};
  message += operation;
  message += " failed (";
  message += vk_result_name(result);
  message += ", ";
  message += std::to_string(static_cast<int>(result));
  message += ')';
  throw std::runtime_error{message};
}

void check_vk(const char* operation, const VkResult result) {
  if (result != VK_SUCCESS) {
    throw_vk_error(operation, result);
  }
}

[[noreturn]] void throw_cuda_error(const char* operation,
                                   const cudaError_t error) {
  const char* const name = cudaGetErrorName(error);
  const char* const description = cudaGetErrorString(error);

  std::string message{"Vulkan/CUDA interop: "};
  message += operation;
  message += " failed";
  if (name != nullptr) {
    message += " (";
    message += name;
    message += ')';
  }
  if (description != nullptr) {
    message += ": ";
    message += description;
  }
  throw std::runtime_error{message};
}

void check_cuda(const char* operation, const cudaError_t error) {
  if (error != cudaSuccess) {
    throw_cuda_error(operation, error);
  }
}

class UniqueWin32Handle final {
 public:
  UniqueWin32Handle() noexcept = default;
  ~UniqueWin32Handle() noexcept { reset(); }

  UniqueWin32Handle(const UniqueWin32Handle&) = delete;
  UniqueWin32Handle& operator=(const UniqueWin32Handle&) = delete;

  [[nodiscard]] HANDLE* put() noexcept {
    reset();
    return &handle_;
  }

  [[nodiscard]] HANDLE get() const noexcept { return handle_; }

  void reset() noexcept {
    if (handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE) {
      static_cast<void>(CloseHandle(handle_));
    }
    handle_ = nullptr;
  }

 private:
  HANDLE handle_ = nullptr;
};

class UniqueVkBuffer final {
 public:
  explicit UniqueVkBuffer(const VkDevice device) noexcept : device_(device) {}
  ~UniqueVkBuffer() noexcept { reset(); }

  UniqueVkBuffer(const UniqueVkBuffer&) = delete;
  UniqueVkBuffer& operator=(const UniqueVkBuffer&) = delete;

  [[nodiscard]] VkBuffer* put() noexcept {
    reset();
    return &buffer_;
  }

  [[nodiscard]] VkBuffer get() const noexcept { return buffer_; }

  [[nodiscard]] VkBuffer release() noexcept {
    return std::exchange(buffer_, VK_NULL_HANDLE);
  }

  void reset() noexcept {
    if (device_ != VK_NULL_HANDLE && buffer_ != VK_NULL_HANDLE) {
      vkDestroyBuffer(device_, buffer_, nullptr);
    }
    buffer_ = VK_NULL_HANDLE;
  }

 private:
  VkDevice device_ = VK_NULL_HANDLE;
  VkBuffer buffer_ = VK_NULL_HANDLE;
};

class UniqueVkDeviceMemory final {
 public:
  explicit UniqueVkDeviceMemory(const VkDevice device) noexcept
      : device_(device) {}
  ~UniqueVkDeviceMemory() noexcept { reset(); }

  UniqueVkDeviceMemory(const UniqueVkDeviceMemory&) = delete;
  UniqueVkDeviceMemory& operator=(const UniqueVkDeviceMemory&) = delete;

  [[nodiscard]] VkDeviceMemory* put() noexcept {
    reset();
    return &memory_;
  }

  [[nodiscard]] VkDeviceMemory get() const noexcept { return memory_; }

  [[nodiscard]] VkDeviceMemory release() noexcept {
    return std::exchange(memory_, VK_NULL_HANDLE);
  }

  void reset() noexcept {
    if (device_ != VK_NULL_HANDLE && memory_ != VK_NULL_HANDLE) {
      vkFreeMemory(device_, memory_, nullptr);
    }
    memory_ = VK_NULL_HANDLE;
  }

 private:
  VkDevice device_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
};

class UniqueVkSemaphore final {
 public:
  explicit UniqueVkSemaphore(const VkDevice device) noexcept
      : device_(device) {}
  ~UniqueVkSemaphore() noexcept { reset(); }

  UniqueVkSemaphore(const UniqueVkSemaphore&) = delete;
  UniqueVkSemaphore& operator=(const UniqueVkSemaphore&) = delete;

  [[nodiscard]] VkSemaphore* put() noexcept {
    reset();
    return &semaphore_;
  }

  [[nodiscard]] VkSemaphore get() const noexcept { return semaphore_; }

  [[nodiscard]] VkSemaphore release() noexcept {
    return std::exchange(semaphore_, VK_NULL_HANDLE);
  }

  void reset() noexcept {
    if (device_ != VK_NULL_HANDLE && semaphore_ != VK_NULL_HANDLE) {
      vkDestroySemaphore(device_, semaphore_, nullptr);
    }
    semaphore_ = VK_NULL_HANDLE;
  }

 private:
  VkDevice device_ = VK_NULL_HANDLE;
  VkSemaphore semaphore_ = VK_NULL_HANDLE;
};

class CleanupCudaDeviceScope final {
 public:
  explicit CleanupCudaDeviceScope(const int target_device) noexcept {
    if (target_device < 0 ||
        cudaGetDevice(&previous_device_) != cudaSuccess) {
      return;
    }

    active_ = true;
    if (previous_device_ == target_device) {
      return;
    }

    if (cudaSetDevice(target_device) == cudaSuccess) {
      restore_ = true;
    } else {
      active_ = false;
    }
  }

  ~CleanupCudaDeviceScope() noexcept {
    if (restore_) {
      static_cast<void>(cudaSetDevice(previous_device_));
    }
  }

  CleanupCudaDeviceScope(const CleanupCudaDeviceScope&) = delete;
  CleanupCudaDeviceScope& operator=(const CleanupCudaDeviceScope&) = delete;

  [[nodiscard]] bool active() const noexcept { return active_; }

 private:
  int previous_device_ = -1;
  bool active_ = false;
  bool restore_ = false;
};

class UniqueCudaExternalMemory final {
 public:
  explicit UniqueCudaExternalMemory(const int cuda_device) noexcept
      : cuda_device_(cuda_device) {}
  ~UniqueCudaExternalMemory() noexcept { reset(); }

  UniqueCudaExternalMemory(const UniqueCudaExternalMemory&) = delete;
  UniqueCudaExternalMemory& operator=(const UniqueCudaExternalMemory&) =
      delete;

  [[nodiscard]] cudaExternalMemory_t* put() noexcept {
    reset();
    return &memory_;
  }

  [[nodiscard]] cudaExternalMemory_t get() const noexcept { return memory_; }

  [[nodiscard]] cudaExternalMemory_t release() noexcept {
    return std::exchange(memory_, nullptr);
  }

  void reset() noexcept {
    if (memory_ == nullptr) {
      return;
    }
    CleanupCudaDeviceScope cuda_scope{cuda_device_};
    if (cuda_scope.active()) {
      static_cast<void>(cudaDestroyExternalMemory(memory_));
    }
    memory_ = nullptr;
  }

 private:
  int cuda_device_ = -1;
  cudaExternalMemory_t memory_ = nullptr;
};

class UniqueCudaMappedBuffer final {
 public:
  explicit UniqueCudaMappedBuffer(const int cuda_device) noexcept
      : cuda_device_(cuda_device) {}
  ~UniqueCudaMappedBuffer() noexcept { reset(); }

  UniqueCudaMappedBuffer(const UniqueCudaMappedBuffer&) = delete;
  UniqueCudaMappedBuffer& operator=(const UniqueCudaMappedBuffer&) = delete;

  [[nodiscard]] void** put() noexcept {
    reset();
    return &pointer_;
  }

  [[nodiscard]] void* get() const noexcept { return pointer_; }

  [[nodiscard]] void* release() noexcept {
    return std::exchange(pointer_, nullptr);
  }

  void reset() noexcept {
    if (pointer_ == nullptr) {
      return;
    }
    CleanupCudaDeviceScope cuda_scope{cuda_device_};
    if (cuda_scope.active()) {
      static_cast<void>(cudaFree(pointer_));
    }
    pointer_ = nullptr;
  }

 private:
  int cuda_device_ = -1;
  void* pointer_ = nullptr;
};

class UniqueCudaExternalSemaphore final {
 public:
  explicit UniqueCudaExternalSemaphore(const int cuda_device) noexcept
      : cuda_device_(cuda_device) {}
  ~UniqueCudaExternalSemaphore() noexcept { reset(); }

  UniqueCudaExternalSemaphore(const UniqueCudaExternalSemaphore&) = delete;
  UniqueCudaExternalSemaphore& operator=(
      const UniqueCudaExternalSemaphore&) = delete;

  [[nodiscard]] cudaExternalSemaphore_t* put() noexcept {
    reset();
    return &semaphore_;
  }

  [[nodiscard]] cudaExternalSemaphore_t get() const noexcept {
    return semaphore_;
  }

  [[nodiscard]] cudaExternalSemaphore_t release() noexcept {
    return std::exchange(semaphore_, nullptr);
  }

  void reset() noexcept {
    if (semaphore_ == nullptr) {
      return;
    }
    CleanupCudaDeviceScope cuda_scope{cuda_device_};
    if (cuda_scope.active()) {
      static_cast<void>(cudaDestroyExternalSemaphore(semaphore_));
    }
    semaphore_ = nullptr;
  }

 private:
  int cuda_device_ = -1;
  cudaExternalSemaphore_t semaphore_ = nullptr;
};

[[nodiscard]] int validate_matching_devices(
    const VkPhysicalDevice physical_device,
    const bool require_timeline_support) {
  if (physical_device == VK_NULL_HANDLE) {
    throw std::invalid_argument{
        "Vulkan/CUDA interop: physical_device must not be VK_NULL_HANDLE"};
  }

  VkPhysicalDeviceIDProperties identifiers{};
  identifiers.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

  VkPhysicalDeviceProperties2 properties{};
  properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties.pNext = &identifiers;
  vkGetPhysicalDeviceProperties2(physical_device, &properties);

  int cuda_device = -1;
  check_cuda("cudaGetDevice", cudaGetDevice(&cuda_device));

  cudaDeviceProp cuda_properties{};
  check_cuda("cudaGetDeviceProperties",
             cudaGetDeviceProperties(&cuda_properties, cuda_device));

  static_assert(VK_LUID_SIZE == sizeof(cuda_properties.luid));
  static_assert(VK_UUID_SIZE == sizeof(cuda_properties.uuid.bytes));

  if (identifiers.deviceLUIDValid != VK_TRUE) {
    throw std::runtime_error{
        "Vulkan/CUDA interop: Vulkan did not report a valid Windows adapter "
        "LUID (CUDA/Vulkan interop requires a WDDM-visible matching device)"};
  }

  if (std::memcmp(identifiers.deviceLUID, cuda_properties.luid,
                  VK_LUID_SIZE) != 0) {
    throw std::runtime_error{
        "Vulkan/CUDA interop: selected VkPhysicalDevice LUID does not match "
        "the current CUDA device"};
  }

  if (identifiers.deviceNodeMask != cuda_properties.luidDeviceNodeMask) {
    throw std::runtime_error{
        "Vulkan/CUDA interop: Vulkan and CUDA identify different nodes of "
        "the selected Windows adapter"};
  }

  if (std::memcmp(identifiers.deviceUUID, cuda_properties.uuid.bytes,
                  VK_UUID_SIZE) != 0) {
    throw std::runtime_error{
        "Vulkan/CUDA interop: selected VkPhysicalDevice UUID does not match "
        "the current CUDA device"};
  }

  if (require_timeline_support &&
      cuda_properties.timelineSemaphoreInteropSupported == 0) {
    throw std::runtime_error{
        "Vulkan/CUDA interop: current CUDA device does not support external "
        "timeline semaphore interop"};
  }

  return cuda_device;
}

void validate_external_buffer_support(
    const VkPhysicalDevice physical_device,
    const VkBufferUsageFlags usage) {
  VkPhysicalDeviceExternalBufferInfo query{};
  query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
  query.usage = usage;
  query.handleType = kExternalMemoryHandleType;

  VkExternalBufferProperties properties{};
  properties.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
  vkGetPhysicalDeviceExternalBufferProperties(physical_device, &query,
                                              &properties);

  const VkExternalMemoryProperties& external =
      properties.externalMemoryProperties;
  if ((external.externalMemoryFeatures &
       VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) == 0 ||
      (external.compatibleHandleTypes & kExternalMemoryHandleType) == 0) {
    throw std::runtime_error{
        "Vulkan/CUDA interop: OPAQUE_WIN32 export is not supported for the "
        "requested Vulkan buffer usage"};
  }
}

void validate_external_timeline_support(
    const VkPhysicalDevice physical_device) {
  VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features{};
  timeline_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;

  VkPhysicalDeviceFeatures2 features{};
  features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features.pNext = &timeline_features;
  vkGetPhysicalDeviceFeatures2(physical_device, &features);
  if (timeline_features.timelineSemaphore != VK_TRUE) {
    throw std::runtime_error{
        "Vulkan/CUDA interop: VkPhysicalDevice does not support timeline "
        "semaphores"};
  }

  VkPhysicalDeviceExternalSemaphoreInfo query{};
  query.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO;
  VkSemaphoreTypeCreateInfo timeline_info{};
  timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  query.pNext = &timeline_info;
  query.handleType = kExternalSemaphoreHandleType;

  VkExternalSemaphoreProperties properties{};
  properties.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES;
  vkGetPhysicalDeviceExternalSemaphoreProperties(physical_device, &query,
                                                 &properties);

  if ((properties.externalSemaphoreFeatures &
       VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) == 0 ||
      (properties.compatibleHandleTypes & kExternalSemaphoreHandleType) ==
          0) {
    throw std::runtime_error{
        "Vulkan/CUDA interop: exportable OPAQUE_WIN32 semaphores are not "
        "supported by the selected VkPhysicalDevice"};
  }
}

[[nodiscard]] std::uint32_t find_device_local_memory_type(
    const VkPhysicalDevice physical_device,
    const std::uint32_t allowed_types) {
  VkPhysicalDeviceMemoryProperties properties{};
  vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);

  for (std::uint32_t index = 0; index < properties.memoryTypeCount;
       ++index) {
    const bool allowed = (allowed_types & (std::uint32_t{1} << index)) != 0;
    const bool device_local =
        (properties.memoryTypes[index].propertyFlags &
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    if (allowed && device_local) {
      return index;
    }
  }

  throw std::runtime_error{
      "Vulkan/CUDA interop: no compatible device-local Vulkan memory type "
      "is available for the exported buffer"};
}

template <typename Function>
[[nodiscard]] Function load_device_function(const VkDevice device,
                                            const char* name) {
  const auto function =
      reinterpret_cast<Function>(vkGetDeviceProcAddr(device, name));
  if (function == nullptr) {
    std::string message{"Vulkan/CUDA interop: Vulkan device function "};
    message += name;
    message +=
        " is unavailable; enable the matching VK_KHR_*_win32 extension "
        "when creating VkDevice";
    throw std::runtime_error{message};
  }
  return function;
}

}  // namespace

ExportedBuffer::ExportedBuffer(const VkPhysicalDevice physical_device,
                               const VkDevice device,
                               const VkDeviceSize size,
                               const VkBufferUsageFlags usage) {
  if (device == VK_NULL_HANDLE) {
    throw std::invalid_argument{
        "ExportedBuffer: device must not be VK_NULL_HANDLE"};
  }
  if (size == 0) {
    throw std::invalid_argument{
        "ExportedBuffer: size must be greater than zero"};
  }
  if (usage == 0) {
    throw std::invalid_argument{
        "ExportedBuffer: usage must contain at least one flag"};
  }

  const int cuda_device =
      validate_matching_devices(physical_device, false);
  validate_external_buffer_support(physical_device, usage);

  // Locals own every resource until construction is complete. Their
  // declaration order makes unwinding release a CUDA mapping before its
  // imported memory, and the Vulkan buffer before its bound allocation.
  UniqueVkDeviceMemory memory_owner{device};
  UniqueVkBuffer buffer_owner{device};
  UniqueCudaExternalMemory cuda_memory_owner{cuda_device};
  UniqueCudaMappedBuffer cuda_mapping_owner{cuda_device};

  VkExternalMemoryBufferCreateInfo external_buffer{};
  external_buffer.sType =
      VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
  external_buffer.handleTypes = kExternalMemoryHandleType;

  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.pNext = &external_buffer;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  check_vk("vkCreateBuffer",
           vkCreateBuffer(device, &buffer_info, nullptr,
                          buffer_owner.put()));

  VkMemoryRequirements memory_requirements{};
  vkGetBufferMemoryRequirements(device, buffer_owner.get(),
                                &memory_requirements);

  VkMemoryDedicatedAllocateInfo dedicated_info{};
  dedicated_info.sType =
      VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
  dedicated_info.buffer = buffer_owner.get();

  VkExportMemoryAllocateInfo export_info{};
  export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
  export_info.pNext = &dedicated_info;
  export_info.handleTypes = kExternalMemoryHandleType;

  VkMemoryAllocateInfo allocation_info{};
  allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocation_info.pNext = &export_info;
  allocation_info.allocationSize = memory_requirements.size;
  allocation_info.memoryTypeIndex = find_device_local_memory_type(
      physical_device, memory_requirements.memoryTypeBits);
  check_vk("vkAllocateMemory",
           vkAllocateMemory(device, &allocation_info, nullptr,
                            memory_owner.put()));

  check_vk("vkBindBufferMemory",
           vkBindBufferMemory(device, buffer_owner.get(),
                              memory_owner.get(), 0));

  const auto get_memory_handle =
      load_device_function<PFN_vkGetMemoryWin32HandleKHR>(
          device, "vkGetMemoryWin32HandleKHR");

  VkMemoryGetWin32HandleInfoKHR handle_info{};
  handle_info.sType =
      VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
  handle_info.memory = memory_owner.get();
  handle_info.handleType = kExternalMemoryHandleType;

  UniqueWin32Handle exported_handle;
  check_vk("vkGetMemoryWin32HandleKHR",
           get_memory_handle(device, &handle_info,
                             exported_handle.put()));

  cudaExternalMemoryHandleDesc import_info{};
  import_info.type = cudaExternalMemoryHandleTypeOpaqueWin32;
  import_info.handle.win32.handle = exported_handle.get();
  import_info.handle.win32.name = nullptr;
  import_info.size =
      static_cast<unsigned long long>(memory_requirements.size);
  import_info.flags = cudaExternalMemoryDedicated;

  const cudaError_t import_result =
      cudaImportExternalMemory(cuda_memory_owner.put(), &import_info);
  // CUDA retains its own reference but never takes ownership of an NT
  // OPAQUE_WIN32 handle. Close it after both success and failure.
  exported_handle.reset();
  check_cuda("cudaImportExternalMemory", import_result);

  cudaExternalMemoryBufferDesc mapping_info{};
  mapping_info.offset = 0;
  mapping_info.size = static_cast<unsigned long long>(size);
  mapping_info.flags = 0;
  check_cuda(
      "cudaExternalMemoryGetMappedBuffer",
      cudaExternalMemoryGetMappedBuffer(cuda_mapping_owner.put(),
                                        cuda_memory_owner.get(),
                                        &mapping_info));

  device_ = device;
  size_ = size;
  cuda_device_ = cuda_device;
  memory_ = memory_owner.release();
  buffer_ = buffer_owner.release();
  cuda_memory_ = cuda_memory_owner.release();
  cuda_ptr_ = cuda_mapping_owner.release();
}

ExportedBuffer::~ExportedBuffer() noexcept { release(); }

ExportedBuffer::ExportedBuffer(ExportedBuffer&& other) noexcept
    : device_(std::exchange(other.device_, VK_NULL_HANDLE)),
      buffer_(std::exchange(other.buffer_, VK_NULL_HANDLE)),
      memory_(std::exchange(other.memory_, VK_NULL_HANDLE)),
      cuda_memory_(std::exchange(other.cuda_memory_, nullptr)),
      cuda_ptr_(std::exchange(other.cuda_ptr_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      cuda_device_(std::exchange(other.cuda_device_, -1)) {}

ExportedBuffer& ExportedBuffer::operator=(ExportedBuffer&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  release();
  device_ = std::exchange(other.device_, VK_NULL_HANDLE);
  buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
  memory_ = std::exchange(other.memory_, VK_NULL_HANDLE);
  cuda_memory_ = std::exchange(other.cuda_memory_, nullptr);
  cuda_ptr_ = std::exchange(other.cuda_ptr_, nullptr);
  size_ = std::exchange(other.size_, 0);
  cuda_device_ = std::exchange(other.cuda_device_, -1);
  return *this;
}

VkBuffer ExportedBuffer::buffer() const noexcept { return buffer_; }

VkDeviceSize ExportedBuffer::size() const noexcept { return size_; }

void* ExportedBuffer::cuda_ptr() const noexcept { return cuda_ptr_; }

void ExportedBuffer::release() noexcept {
  {
    CleanupCudaDeviceScope cuda_scope{cuda_device_};
    if (cuda_scope.active()) {
      // CUDA requires mapped pointers to be released before the imported
      // external-memory object is destroyed.
      if (cuda_ptr_ != nullptr) {
        static_cast<void>(cudaFree(cuda_ptr_));
      }
      if (cuda_memory_ != nullptr) {
        static_cast<void>(cudaDestroyExternalMemory(cuda_memory_));
      }
    }
  }

  cuda_ptr_ = nullptr;
  cuda_memory_ = nullptr;
  cuda_device_ = -1;

  if (device_ != VK_NULL_HANDLE) {
    if (buffer_ != VK_NULL_HANDLE) {
      vkDestroyBuffer(device_, buffer_, nullptr);
    }
    if (memory_ != VK_NULL_HANDLE) {
      vkFreeMemory(device_, memory_, nullptr);
    }
  }

  buffer_ = VK_NULL_HANDLE;
  memory_ = VK_NULL_HANDLE;
  device_ = VK_NULL_HANDLE;
  size_ = 0;
}

InteropTimeline::InteropTimeline(
    const VkPhysicalDevice physical_device, const VkDevice device,
    const std::uint64_t initial_value) {
  if (device == VK_NULL_HANDLE) {
    throw std::invalid_argument{
        "InteropTimeline: device must not be VK_NULL_HANDLE"};
  }

  const int cuda_device =
      validate_matching_devices(physical_device, true);
  validate_external_timeline_support(physical_device);

  UniqueVkSemaphore semaphore_owner{device};
  UniqueCudaExternalSemaphore cuda_semaphore_owner{cuda_device};

  VkSemaphoreTypeCreateInfo timeline_info{};
  timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timeline_info.initialValue = initial_value;

  VkExportSemaphoreCreateInfo export_info{};
  export_info.sType =
      VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
  export_info.pNext = &timeline_info;
  export_info.handleTypes = kExternalSemaphoreHandleType;

  VkSemaphoreCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  create_info.pNext = &export_info;
  check_vk("vkCreateSemaphore",
           vkCreateSemaphore(device, &create_info, nullptr,
                             semaphore_owner.put()));

  const auto get_semaphore_handle =
      load_device_function<PFN_vkGetSemaphoreWin32HandleKHR>(
          device, "vkGetSemaphoreWin32HandleKHR");

  VkSemaphoreGetWin32HandleInfoKHR handle_info{};
  handle_info.sType =
      VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
  handle_info.semaphore = semaphore_owner.get();
  handle_info.handleType = kExternalSemaphoreHandleType;

  UniqueWin32Handle exported_handle;
  check_vk("vkGetSemaphoreWin32HandleKHR",
           get_semaphore_handle(device, &handle_info,
                                exported_handle.put()));

  cudaExternalSemaphoreHandleDesc import_info{};
  import_info.type =
      cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
  import_info.handle.win32.handle = exported_handle.get();
  import_info.handle.win32.name = nullptr;
  import_info.flags = 0;

  const cudaError_t import_result =
      cudaImportExternalSemaphore(cuda_semaphore_owner.put(), &import_info);
  // TimelineSemaphoreWin32 also leaves NT HANDLE ownership with the app.
  exported_handle.reset();
  check_cuda("cudaImportExternalSemaphore", import_result);

  device_ = device;
  cuda_device_ = cuda_device;
  semaphore_ = semaphore_owner.release();
  cuda_semaphore_ = cuda_semaphore_owner.release();
}

InteropTimeline::~InteropTimeline() noexcept { release(); }

InteropTimeline::InteropTimeline(InteropTimeline&& other) noexcept
    : device_(std::exchange(other.device_, VK_NULL_HANDLE)),
      semaphore_(std::exchange(other.semaphore_, VK_NULL_HANDLE)),
      cuda_semaphore_(std::exchange(other.cuda_semaphore_, nullptr)),
      cuda_device_(std::exchange(other.cuda_device_, -1)) {}

InteropTimeline& InteropTimeline::operator=(
    InteropTimeline&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  release();
  device_ = std::exchange(other.device_, VK_NULL_HANDLE);
  semaphore_ = std::exchange(other.semaphore_, VK_NULL_HANDLE);
  cuda_semaphore_ = std::exchange(other.cuda_semaphore_, nullptr);
  cuda_device_ = std::exchange(other.cuda_device_, -1);
  return *this;
}

void InteropTimeline::enqueue_cuda_wait(const cudaStream_t stream,
                                        const std::uint64_t value) {
  if (cuda_semaphore_ == nullptr) {
    throw std::logic_error{
        "InteropTimeline: enqueue_cuda_wait called on a moved-from object"};
  }

  cudaExternalSemaphoreWaitParams parameters{};
  parameters.params.fence.value = value;
  parameters.flags = 0;
  check_cuda(
      "cudaWaitExternalSemaphoresAsync",
      cudaWaitExternalSemaphoresAsync(&cuda_semaphore_, &parameters, 1,
                                      stream));
}

void InteropTimeline::enqueue_cuda_signal(const cudaStream_t stream,
                                          const std::uint64_t value) {
  if (cuda_semaphore_ == nullptr) {
    throw std::logic_error{
        "InteropTimeline: enqueue_cuda_signal called on a moved-from object"};
  }

  cudaExternalSemaphoreSignalParams parameters{};
  parameters.params.fence.value = value;
  parameters.flags = 0;
  check_cuda(
      "cudaSignalExternalSemaphoresAsync",
      cudaSignalExternalSemaphoresAsync(&cuda_semaphore_, &parameters, 1,
                                        stream));
}

VkSemaphore InteropTimeline::vk_semaphore() const noexcept {
  return semaphore_;
}

void InteropTimeline::release() noexcept {
  {
    CleanupCudaDeviceScope cuda_scope{cuda_device_};
    if (cuda_scope.active() && cuda_semaphore_ != nullptr) {
      // The caller must have completed all queued waits/signals first.
      static_cast<void>(cudaDestroyExternalSemaphore(cuda_semaphore_));
    }
  }

  cuda_semaphore_ = nullptr;
  cuda_device_ = -1;

  if (device_ != VK_NULL_HANDLE && semaphore_ != VK_NULL_HANDLE) {
    vkDestroySemaphore(device_, semaphore_, nullptr);
  }

  semaphore_ = VK_NULL_HANDLE;
  device_ = VK_NULL_HANDLE;
}

}  // namespace satview::gpu
