#pragma once

#include <cstdint>

#include <vulkan/vulkan.h>

namespace satview::viewer {

class VulkanColormapLut final {
public:
    VulkanColormapLut(
        VkPhysicalDevice physical_device,
        VkDevice device,
        std::uint32_t queue_family,
        VkQueue queue);
    ~VulkanColormapLut() noexcept;

    VulkanColormapLut(const VulkanColormapLut&) = delete;
    VulkanColormapLut& operator=(const VulkanColormapLut&) = delete;

    [[nodiscard]] VkImageView image_view() const noexcept {
        return image_view_;
    }

    [[nodiscard]] VkSampler sampler() const noexcept {
        return sampler_;
    }

private:
    void destroy() noexcept;

    VkDevice device_ = VK_NULL_HANDLE;
    VkImage image_ = VK_NULL_HANDLE;
    VkDeviceMemory image_memory_ = VK_NULL_HANDLE;
    VkImageView image_view_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
};

} // namespace satview::viewer
