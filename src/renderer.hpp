#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tri {

class VulkanError : public std::runtime_error {
public:
    explicit VulkanError(const std::string &what) : std::runtime_error(what) {}
};

struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;

    static vk::VertexInputBindingDescription bindingDescription();
    static std::vector<vk::VertexInputAttributeDescription>
    attributeDescriptions();
};

class Renderer {
public:
    Renderer();
    ~Renderer();

    Renderer(const Renderer &) = delete;
    Renderer &operator=(const Renderer &) = delete;

    void run();

private:
    static constexpr uint32_t kWidth = 800;
    static constexpr uint32_t kHeight = 600;
    static constexpr uint32_t kIndexCount = 3;
    static constexpr uint32_t kExitAfterFrames = 200;

    GLFWwindow *window_ = nullptr;

    // The raii wrappers delete their default constructor when
    // VULKAN_HPP_NO_STRUCT_CONSTRUCTORS is set, so they are copy-initialised
    // from nullptr (a null handle) and filled in later.
    vk::raii::Context context_;
    vk::raii::Instance instance_ = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger_ = nullptr;
    vk::raii::SurfaceKHR surface_ = nullptr;
    vk::raii::PhysicalDevice physicalDevice_ = nullptr;
    vk::raii::Device device_ = nullptr;
    vk::raii::Queue graphicsQueue_ = nullptr;
    uint32_t graphicsQueueFamily_ = 0;

    vk::raii::SwapchainKHR swapchain_ = nullptr;
    std::vector<vk::Image> swapchainImages_;
    vk::SurfaceFormatKHR surfaceFormat_;
    vk::Extent2D extent_;
    std::vector<vk::raii::ImageView> imageViews_;

    vk::raii::PipelineLayout pipelineLayout_ = nullptr;
    vk::raii::ShaderModule vertModule_ = nullptr;
    vk::raii::ShaderModule fragModule_ = nullptr;
    vk::raii::Pipeline pipeline_ = nullptr;

    vk::raii::CommandPool commandPool_ = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers_;

    vk::raii::Buffer vertexBuffer_ = nullptr;
    vk::raii::DeviceMemory vertexBufferMemory_ = nullptr;
    vk::raii::Buffer indexBuffer_ = nullptr;
    vk::raii::DeviceMemory indexBufferMemory_ = nullptr;
    vk::raii::Buffer stagingBuffer_ = nullptr;
    vk::raii::DeviceMemory stagingBufferMemory_ = nullptr;
    void *stagingMapped_ = nullptr;
    vk::DeviceSize stagingCapacity_ = 0;

    vk::raii::Semaphore imageAvailable_ = nullptr;
    vk::raii::Semaphore renderFinished_ = nullptr;
    vk::raii::Fence inFlight_ = nullptr;
    vk::FenceCreateInfo fenceCreateInfo_;

    bool framebufferResized_ = false;

    void createWindow();
    void createInstance();
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapchain();
    void createImageViews();
    void createPipelines();
    void createCommandPool();
    void createCommandBuffers();
    void createBuffers();
    void createSyncObjects();

    void destroySwapchain();
    void recreateSwapchain();

    void drawFrame();
    void recordCommandBuffer(uint32_t imageIndex);
    void saveScreenshot(const std::string &path);

    vk::raii::Buffer createBuffer(vk::DeviceSize size,
                                  vk::BufferUsageFlags usage,
                                  vk::MemoryPropertyFlags props,
                                  vk::raii::DeviceMemory &outMemory);
    void copyToDevice(const void *data,
                      vk::DeviceSize size,
                      const vk::raii::Buffer &dst);
    uint32_t findMemoryType(uint32_t typeFilter,
                            vk::MemoryPropertyFlags props) const;

    bool deviceSuitable(const vk::raii::PhysicalDevice &dev) const;
    std::optional<uint32_t>
    findGraphicsQueue(const vk::raii::PhysicalDevice &dev) const;
    std::vector<const char *> requiredInstanceExtensions() const;
    std::vector<const char *> requiredDeviceExtensions() const;
    vk::SurfaceFormatKHR chooseSurfaceFormat(
        const std::vector<vk::SurfaceFormatKHR> &formats) const;
    vk::PresentModeKHR choosePresentMode(
        const std::vector<vk::PresentModeKHR> &modes) const;
    vk::Extent2D chooseExtent(const vk::SurfaceCapabilitiesKHR &caps) const;

    static void framebufferResizeCallback(GLFWwindow *w, int width, int height);
    static VKAPI_ATTR vk::Bool32 VKAPI_CALL
    debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                  vk::DebugUtilsMessageTypeFlagsEXT type,
                  const vk::DebugUtilsMessengerCallbackDataEXT *data,
                  void *userData);
    static void glfwErrorCallback(int code, const char *desc);
};

} // namespace tri
