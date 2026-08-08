#include "colormap_lut.hpp"

#include "colormaps.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace satview::viewer {
namespace {

void check_vk(const VkResult result, const char* operation)
{
    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with VkResult " +
            std::to_string(static_cast<int>(result)));
    }
}

std::uint32_t find_memory_type(
    const VkPhysicalDevice physical_device,
    const std::uint32_t allowed_types,
    const VkMemoryPropertyFlags required_properties)
{
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);
    for (std::uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
        if ((allowed_types & (1u << index)) != 0u &&
            (properties.memoryTypes[index].propertyFlags &
             required_properties) == required_properties) {
            return index;
        }
    }
    throw std::runtime_error("no compatible Vulkan colormap memory type");
}

} // namespace

VulkanColormapLut::VulkanColormapLut(
    const VkPhysicalDevice physical_device,
    const VkDevice device,
    const std::uint32_t queue_family,
    const VkQueue queue)
    : device_(device)
{
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;

    const auto destroy_temporary = [&]() noexcept {
        if (command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, command_pool, nullptr);
            command_pool = VK_NULL_HANDLE;
        }
        if (staging_buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, staging_buffer, nullptr);
            staging_buffer = VK_NULL_HANDLE;
        }
        if (staging_memory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, staging_memory, nullptr);
            staging_memory = VK_NULL_HANDLE;
        }
    };

    try {
        VkImageCreateInfo image_create{};
        image_create.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_create.imageType = VK_IMAGE_TYPE_2D;
        image_create.format = VK_FORMAT_R8G8B8A8_UNORM;
        image_create.extent = {
            kColormapSampleCount,
            kColormapCount,
            1,
        };
        image_create.mipLevels = 1;
        image_create.arrayLayers = 1;
        image_create.samples = VK_SAMPLE_COUNT_1_BIT;
        image_create.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_create.usage =
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        image_create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        image_create.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        check_vk(
            vkCreateImage(device_, &image_create, nullptr, &image_),
            "create colormap LUT image");

        VkMemoryRequirements image_requirements{};
        vkGetImageMemoryRequirements(device_, image_, &image_requirements);
        VkMemoryAllocateInfo image_allocation{};
        image_allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        image_allocation.allocationSize = image_requirements.size;
        image_allocation.memoryTypeIndex = find_memory_type(
            physical_device,
            image_requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check_vk(
            vkAllocateMemory(
                device_, &image_allocation, nullptr, &image_memory_),
            "allocate colormap LUT image");
        check_vk(
            vkBindImageMemory(device_, image_, image_memory_, 0),
            "bind colormap LUT image");

        const auto pixels = colormap_rgba8();
        const auto upload_size = static_cast<VkDeviceSize>(pixels.size_bytes());

        VkBufferCreateInfo buffer_create{};
        buffer_create.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_create.size = upload_size;
        buffer_create.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        buffer_create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check_vk(
            vkCreateBuffer(
                device_, &buffer_create, nullptr, &staging_buffer),
            "create colormap LUT staging buffer");

        VkMemoryRequirements staging_requirements{};
        vkGetBufferMemoryRequirements(
            device_, staging_buffer, &staging_requirements);
        VkMemoryAllocateInfo staging_allocation{};
        staging_allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        staging_allocation.allocationSize = staging_requirements.size;
        staging_allocation.memoryTypeIndex = find_memory_type(
            physical_device,
            staging_requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check_vk(
            vkAllocateMemory(
                device_, &staging_allocation, nullptr, &staging_memory),
            "allocate colormap LUT staging buffer");
        check_vk(
            vkBindBufferMemory(device_, staging_buffer, staging_memory, 0),
            "bind colormap LUT staging buffer");

        void* mapped = nullptr;
        check_vk(
            vkMapMemory(
                device_, staging_memory, 0, upload_size, 0, &mapped),
            "map colormap LUT staging buffer");
        std::memcpy(mapped, pixels.data(), pixels.size_bytes());
        vkUnmapMemory(device_, staging_memory);

        VkCommandPoolCreateInfo pool_create{};
        pool_create.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_create.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pool_create.queueFamilyIndex = queue_family;
        check_vk(
            vkCreateCommandPool(
                device_, &pool_create, nullptr, &command_pool),
            "create colormap LUT command pool");

        VkCommandBufferAllocateInfo command_allocate{};
        command_allocate.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command_allocate.commandPool = command_pool;
        command_allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_allocate.commandBufferCount = 1;
        VkCommandBuffer command = VK_NULL_HANDLE;
        check_vk(
            vkAllocateCommandBuffers(
                device_, &command_allocate, &command),
            "allocate colormap LUT command buffer");

        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_vk(
            vkBeginCommandBuffer(command, &begin),
            "begin colormap LUT upload");

        VkImageMemoryBarrier to_transfer{};
        to_transfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_transfer.srcAccessMask = 0;
        to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_transfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_transfer.image = image_;
        to_transfer.subresourceRange.aspectMask =
            VK_IMAGE_ASPECT_COLOR_BIT;
        to_transfer.subresourceRange.levelCount = 1;
        to_transfer.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(
            command,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &to_transfer);

        VkBufferImageCopy copy{};
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = 1;
        copy.imageExtent = {
            kColormapSampleCount,
            kColormapCount,
            1,
        };
        vkCmdCopyBufferToImage(
            command,
            staging_buffer,
            image_,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &copy);

        VkImageMemoryBarrier to_sample{};
        to_sample.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_sample.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_sample.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        to_sample.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_sample.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_sample.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_sample.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_sample.image = image_;
        to_sample.subresourceRange = to_transfer.subresourceRange;
        vkCmdPipelineBarrier(
            command,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &to_sample);

        check_vk(
            vkEndCommandBuffer(command),
            "end colormap LUT upload");
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        check_vk(
            vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE),
            "submit colormap LUT upload");
        check_vk(vkQueueWaitIdle(queue), "wait for colormap LUT upload");

        destroy_temporary();

        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = image_;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = VK_FORMAT_R8G8B8A8_UNORM;
        view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.levelCount = 1;
        view.subresourceRange.layerCount = 1;
        check_vk(
            vkCreateImageView(device_, &view, nullptr, &image_view_),
            "create colormap LUT image view");

        VkSamplerCreateInfo sampler_create{};
        sampler_create.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler_create.magFilter = VK_FILTER_NEAREST;
        sampler_create.minFilter = VK_FILTER_NEAREST;
        sampler_create.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler_create.addressModeU =
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_create.addressModeV =
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_create.addressModeW =
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_create.maxLod = 0.0F;
        check_vk(
            vkCreateSampler(device_, &sampler_create, nullptr, &sampler_),
            "create colormap LUT sampler");
    } catch (...) {
        destroy_temporary();
        destroy();
        throw;
    }
}

VulkanColormapLut::~VulkanColormapLut() noexcept
{
    destroy();
}

void VulkanColormapLut::destroy() noexcept
{
    if (sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_, sampler_, nullptr);
        sampler_ = VK_NULL_HANDLE;
    }
    if (image_view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_, image_view_, nullptr);
        image_view_ = VK_NULL_HANDLE;
    }
    if (image_ != VK_NULL_HANDLE) {
        vkDestroyImage(device_, image_, nullptr);
        image_ = VK_NULL_HANDLE;
    }
    if (image_memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, image_memory_, nullptr);
        image_memory_ = VK_NULL_HANDLE;
    }
}

} // namespace satview::viewer
