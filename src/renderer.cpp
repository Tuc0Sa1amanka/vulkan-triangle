#include "renderer.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string_view>

namespace tri {

namespace {

// Bytes per pixel for the swapchain formats chooseSurfaceFormat can return.
std::optional<uint32_t> bytesPerPixel(vk::Format format) {
    switch (format) {
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eB8G8R8A8Srgb:
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Srgb:
        return 4;
    case vk::Format::eA8B8G8R8UnormPack32:
    case vk::Format::eA8B8G8R8SrgbPack32:
        return 4;
    case vk::Format::eA2R10G10B10UnormPack32:
    case vk::Format::eA2B10G10R10UnormPack32:
    case vk::Format::eR16G16B16A16Sfloat:
    case vk::Format::eR16G16B16A16Unorm:
    case vk::Format::eR32G32B32A32Sfloat:
        return 8;
    default:
        return std::nullopt;
    }
}

std::vector<char> readFile(const std::string &path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file) {
        throw VulkanError("cannot open " + path);
    }
    const auto size = file.tellg();
    if (size <= 0) {
        throw VulkanError("empty file " + path);
    }
    file.seekg(0);

    std::vector<char> data(static_cast<size_t>(size));
    if (!file.read(data.data(), size)) {
        throw VulkanError("cannot read " + path);
    }
    return data;
}

} // namespace

vk::VertexInputBindingDescription Vertex::bindingDescription() {
    return vk::VertexInputBindingDescription{
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = vk::VertexInputRate::eVertex,
    };
}

std::vector<vk::VertexInputAttributeDescription>
Vertex::attributeDescriptions() {
    return {
        vk::VertexInputAttributeDescription{
            .location = 0,
            .binding = 0,
            .format = vk::Format::eR32G32Sfloat,
            .offset = offsetof(Vertex, pos),
        },
        vk::VertexInputAttributeDescription{
            .location = 1,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = offsetof(Vertex, color),
        },
    };
}

Renderer::Renderer(uint32_t maxFrames) : maxFrames_(maxFrames) {
    createWindow();
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createImageViews();
    createPipelines();
    createCommandPool();
    createCommandBuffers();
    createBuffers();
    createSyncObjects();
}

Renderer::~Renderer() {
    device_.waitIdle();

    if (stagingMapped_ != nullptr) {
        stagingBufferMemory_.unmapMemory();
        stagingMapped_ = nullptr;
    }

    destroySwapchain();

    // The VkSurfaceKHR must go away while the GLFW window is still alive,
    // otherwise the driver tears down its window handles first.
    surface_ = nullptr;

    glfwDestroyWindow(window_);
    glfwTerminate();
}

void Renderer::glfwErrorCallback(int code, const char *desc) {
    std::cerr << "[glfw] error " << code << ": " << desc << "\n";
}

void Renderer::createWindow() {
    glfwSetErrorCallback(&glfwErrorCallback);

    // GLFW 3.4 can use either X11 or Wayland and picks Wayland whenever
    // WAYLAND_DISPLAY is set. Some builds ignore the GLFW_PLATFORM variable,
    // so the same choice is passed explicitly as an init hint. Forcing X11
    // under WSLg is useful because XWayland windows are easier to inspect.
    if (const char *requested = std::getenv("GLFW_PLATFORM")) {
        const std::string_view platform(requested);
        if (platform == "x11") {
            glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
        } else if (platform == "wayland") {
            glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_WAYLAND);
        } else {
            std::cerr << "[win] ignoring unknown GLFW_PLATFORM '" << platform
                      << "'\n";
        }
    }

    if (glfwInit() != GLFW_TRUE) {
        throw VulkanError("glfwInit failed");
    }

    // A Vulkan window must not create an OpenGL context.
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window_ = glfwCreateWindow(static_cast<int>(kWidth),
                               static_cast<int>(kHeight),
                               "vulkan-triangle",
                               nullptr,
                               nullptr);
    if (window_ == nullptr) {
        glfwTerminate();
        throw VulkanError("glfwCreateWindow failed");
    }

    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, &Renderer::framebufferResizeCallback);

    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(window_, &framebufferWidth, &framebufferHeight);

    const int platform = glfwGetPlatform();
    const char *platformName = platform == GLFW_PLATFORM_WAYLAND ? "wayland"
                               : platform == GLFW_PLATFORM_X11    ? "x11"
                               : platform == GLFW_PLATFORM_NULL    ? "null"
                                                                   : "unknown";
    std::cout << "[win] platform=" << platformName << " window=" << kWidth << "x"
              << kHeight << " framebuffer=" << framebufferWidth << "x"
              << framebufferHeight << "\n";
}

std::vector<const char *> Renderer::requiredInstanceExtensions() {
    uint32_t count = 0;
    const char **raw = glfwGetRequiredInstanceExtensions(&count);
    if (raw == nullptr) {
        throw VulkanError(
            "GLFW required no instance extensions; is Vulkan support compiled "
            "into this GLFW build?");
    }

    std::vector<const char *> result(raw, raw + count);

    // The messenger is optional: without the extension the program still runs,
    // it just gets no validation output.
    bool debugUtils = false;
    for (const auto &ext : context_.enumerateInstanceExtensionProperties()) {
        if (std::strcmp(ext.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) ==
            0) {
            debugUtils = true;
            break;
        }
    }
    if (debugUtils) {
        result.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        debugUtilsAvailable_ = true;
    } else {
        std::cerr << "[vulkan] " << VK_EXT_DEBUG_UTILS_EXTENSION_NAME
                  << " unavailable, validation messages disabled\n";
    }
    return result;
}

void Renderer::createInstance() {
    const vk::ApplicationInfo appInfo{
        .pApplicationName = "vulkan-triangle",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "none",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_3,
    };

    const auto extensions = requiredInstanceExtensions();

    // Chaining the messenger into pNext catches messages during instance
    // creation, which a standalone messenger would miss. It is only legal
    // while VK_EXT_debug_utils is being enabled, so it stays out of pNext
    // otherwise.
    const vk::DebugUtilsMessengerCreateInfoEXT debugInfo{
        .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose,
        .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        .pfnUserCallback = &Renderer::debugCallback,
    };

    const vk::InstanceCreateInfo createInfo{
        .pNext = debugUtilsAvailable_ ? &debugInfo : nullptr,
        .pApplicationInfo = &appInfo,
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    instance_ = vk::raii::Instance(context_, createInfo);
}

void Renderer::setupDebugMessenger() {
    if (!debugUtilsAvailable_) {
        return;
    }

    const vk::DebugUtilsMessengerCreateInfoEXT info{
        .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose,
        .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        .pfnUserCallback = &Renderer::debugCallback,
    };
    debugMessenger_ = vk::raii::DebugUtilsMessengerEXT(instance_, info);
}

VKAPI_ATTR vk::Bool32 VKAPI_CALL
Renderer::debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                        vk::DebugUtilsMessageTypeFlagsEXT,
                        const vk::DebugUtilsMessengerCallbackDataEXT *data,
                        void *) {
    if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose) {
        return VK_FALSE;
    }
    std::cerr << "[vulkan] "
              << (data != nullptr && data->pMessage ? data->pMessage : "")
              << "\n";
    return VK_FALSE;
}

void Renderer::createSurface() {
    VkSurfaceKHR raw = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(*instance_, window_, nullptr, &raw) !=
        VK_SUCCESS) {
        throw VulkanError("glfwCreateWindowSurface failed");
    }
    surface_ = vk::raii::SurfaceKHR(instance_, raw);
}

std::optional<uint32_t>
Renderer::findGraphicsQueue(const vk::raii::PhysicalDevice &dev) const {
    const auto families = dev.getQueueFamilyProperties();
    for (uint32_t i = 0; i < families.size(); ++i) {
        if (families[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            return i;
        }
    }
    return std::nullopt;
}

std::vector<const char *> Renderer::requiredDeviceExtensions() const {
    return {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
}

bool Renderer::deviceSuitable(const vk::raii::PhysicalDevice &dev) const {
    // Dynamic rendering (used for the render pass) requires Vulkan 1.3.
    if (dev.getProperties().apiVersion < VK_API_VERSION_1_3) {
        return false;
    }

    if (!findGraphicsQueue(dev).has_value()) {
        return false;
    }

    const auto available = dev.enumerateDeviceExtensionProperties();
    for (const char *needed : requiredDeviceExtensions()) {
        bool found = false;
        for (const auto &ext : available) {
            if (std::strcmp(ext.extensionName, needed) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

void Renderer::pickPhysicalDevice() {
    auto devices = instance_.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw VulkanError("no Vulkan physical devices found");
    }

    for (auto &dev : devices) {
        if (deviceSuitable(dev)) {
            physicalDevice_ = std::move(dev);
            std::cout << "[device] "
                      << physicalDevice_.getProperties().deviceName << "\n";
            return;
        }
    }
    throw VulkanError("no suitable physical device found");
}

void Renderer::createLogicalDevice() {
    const auto family = findGraphicsQueue(physicalDevice_);
    if (!family.has_value()) {
        throw VulkanError("graphics queue family not found");
    }
    graphicsQueueFamily_ = *family;

    const float priority = 1.0f;
    const vk::DeviceQueueCreateInfo queueInfo{
        .queueFamilyIndex = graphicsQueueFamily_,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };

    const auto extensions = requiredDeviceExtensions();

    // dynamicRendering was promoted to core in 1.3, but promoted features are
    // not enabled implicitly.
    const vk::PhysicalDeviceVulkan13Features features13{
        .dynamicRendering = vk::True,
    };

    const vk::DeviceCreateInfo createInfo{
        .pNext = &features13,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueInfo,
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    device_ = vk::raii::Device(physicalDevice_, createInfo);
    graphicsQueue_ = vk::raii::Queue(device_, graphicsQueueFamily_, 0);
}

vk::SurfaceFormatKHR Renderer::chooseSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR> &formats) const {
    // saveScreenshot writes 8-bit PPM, so an 8-bit-per-channel format is
    // strongly preferred over e.g. a float or 10-bit one. Anything wider is
    // still accepted for rendering; only the screenshot is refused.
    const auto usable = [](const vk::SurfaceFormatKHR &f) {
        const auto bpp = bytesPerPixel(f.format);
        return bpp.has_value() && *bpp == 4;
    };

    for (const auto &format : formats) {
        if (format.format == vk::Format::eB8G8R8A8Srgb &&
            format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return format;
        }
    }
    for (const auto &format : formats) {
        if (format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear &&
            usable(format)) {
            return format;
        }
    }
    for (const auto &format : formats) {
        if (usable(format)) {
            return format;
        }
    }
    if (formats.empty()) {
        throw VulkanError("surface reports no formats");
    }
    return formats.front();
}

vk::PresentModeKHR Renderer::choosePresentMode(
    const std::vector<vk::PresentModeKHR> &modes) const {
    const auto it =
        std::find(modes.begin(), modes.end(), vk::PresentModeKHR::eMailbox);
    if (it != modes.end()) {
        return *it;
    }
    // FIFO is the only mode guaranteed to be supported by the spec.
    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D
Renderer::chooseExtent(const vk::SurfaceCapabilitiesKHR &caps) const {
    if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return caps.currentExtent;
    }

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);

    return vk::Extent2D{
        .width = std::clamp(static_cast<uint32_t>(width),
                            caps.minImageExtent.width,
                            caps.maxImageExtent.width),
        .height = std::clamp(static_cast<uint32_t>(height),
                             caps.minImageExtent.height,
                             caps.maxImageExtent.height),
    };
}

void Renderer::destroySwapchain() {
    commandBuffers_.clear();
    imageViews_.clear();
    swapchain_ = nullptr;
    pipeline_ = nullptr;
    pipelineLayout_ = nullptr;
    vertModule_ = nullptr;
    fragModule_ = nullptr;
}

void Renderer::createSwapchain() {
    const auto caps = physicalDevice_.getSurfaceCapabilitiesKHR(*surface_);

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) {
        imageCount = caps.maxImageCount;
    }

    const auto formats = physicalDevice_.getSurfaceFormatsKHR(*surface_);
    const auto presentModes =
        physicalDevice_.getSurfacePresentModesKHR(*surface_);

    surfaceFormat_ = chooseSurfaceFormat(formats);
    extent_ = chooseExtent(caps);
    const auto presentMode = choosePresentMode(presentModes);

    const vk::SwapchainCreateInfoKHR info{
        .surface = *surface_,
        .minImageCount = imageCount,
        .imageFormat = surfaceFormat_.format,
        .imageColorSpace = surfaceFormat_.colorSpace,
        .imageExtent = extent_,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = vk::SharingMode::eExclusive,
        .preTransform = caps.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = presentMode,
        .clipped = VK_TRUE,
    };

    swapchain_ = vk::raii::SwapchainKHR(device_, info);
    swapchainImages_ = swapchain_.getImages();
}

void Renderer::createImageViews() {
    imageViews_.reserve(swapchainImages_.size());
    for (const vk::Image image : swapchainImages_) {
        const vk::ImageViewCreateInfo info{
            .image = image,
            .viewType = vk::ImageViewType::e2D,
            .format = surfaceFormat_.format,
            .subresourceRange = vk::ImageSubresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        imageViews_.push_back(vk::raii::ImageView(device_, info));
    }
}

void Renderer::createPipelines() {
    const auto vertCode =
        readFile(std::string(SHADER_DIR) + "/triangle_vert.spv");
    const auto fragCode =
        readFile(std::string(SHADER_DIR) + "/triangle_frag.spv");

    const vk::ShaderModuleCreateInfo vertInfo{
        .codeSize = vertCode.size(),
        .pCode = reinterpret_cast<const uint32_t *>(vertCode.data()),
    };
    vertModule_ = vk::raii::ShaderModule(device_, vertInfo);

    const vk::ShaderModuleCreateInfo fragInfo{
        .codeSize = fragCode.size(),
        .pCode = reinterpret_cast<const uint32_t *>(fragCode.data()),
    };
    fragModule_ = vk::raii::ShaderModule(device_, fragInfo);

    const std::array<vk::PipelineShaderStageCreateInfo, 2> stages{
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = *vertModule_,
            .pName = "main",
        },
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = *fragModule_,
            .pName = "main",
        },
    };

    const auto binding = Vertex::bindingDescription();
    const auto attributes = Vertex::attributeDescriptions();

    const vk::PipelineVertexInputStateCreateInfo vertexInput{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount =
            static_cast<uint32_t>(attributes.size()),
        .pVertexAttributeDescriptions = attributes.data(),
    };

    const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
    };

    const vk::Viewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(extent_.width),
        .height = static_cast<float>(extent_.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    const vk::Rect2D scissor{
        .offset = {0, 0},
        .extent = extent_,
    };
    const vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    const vk::PipelineRasterizationStateCreateInfo rasterization{
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .lineWidth = 1.0f,
    };

    const vk::PipelineMultisampleStateCreateInfo multisample{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };

    const vk::PipelineColorBlendAttachmentState blendAttachment{
        .blendEnable = VK_FALSE,
        .colorWriteMask = vk::ColorComponentFlagBits::eR |
                          vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB |
                          vk::ColorComponentFlagBits::eA,
    };
    const vk::PipelineColorBlendStateCreateInfo colorBlend{
        .attachmentCount = 1,
        .pAttachments = &blendAttachment,
    };

    // The shaders declare no bindings, but Vulkan still requires a valid
    // pipeline layout handle.
    const vk::PipelineLayoutCreateInfo layoutInfo{};
    pipelineLayout_ = vk::raii::PipelineLayout(device_, layoutInfo);

    // With no VkRenderPass, dynamic rendering needs to know the attachment
    // formats up front.
    const vk::PipelineRenderingCreateInfo renderingInfo{
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &surfaceFormat_.format,
    };

    const vk::GraphicsPipelineCreateInfo pipelineInfo{
        .pNext = &renderingInfo,
        .stageCount = static_cast<uint32_t>(stages.size()),
        .pStages = stages.data(),
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pColorBlendState = &colorBlend,
        .layout = *pipelineLayout_,
    };

    pipeline_ = vk::raii::Pipeline(device_, nullptr, pipelineInfo);
}

void Renderer::createCommandPool() {
    const vk::CommandPoolCreateInfo info{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = graphicsQueueFamily_,
    };
    commandPool_ = vk::raii::CommandPool(device_, info);
}

void Renderer::createCommandBuffers() {
    const vk::CommandBufferAllocateInfo info{
        .commandPool = *commandPool_,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    };
    commandBuffers_ = device_.allocateCommandBuffers(info);
}

uint32_t Renderer::findMemoryType(uint32_t typeFilter,
                                  vk::MemoryPropertyFlags props) const {
    const auto memProps = physicalDevice_.getMemoryProperties();
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeFilter & (1u << i)) == 0) {
            continue;
        }
        if ((memProps.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    throw VulkanError("suitable memory type not found");
}

vk::raii::Buffer Renderer::createBuffer(vk::DeviceSize size,
                                        vk::BufferUsageFlags usage,
                                        vk::MemoryPropertyFlags props,
                                        vk::raii::DeviceMemory &outMemory) {
    const vk::BufferCreateInfo info{
        .size = size,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    auto buffer = vk::raii::Buffer(device_, info);

    const auto reqs = buffer.getMemoryRequirements();
    const uint32_t typeIndex = findMemoryType(reqs.memoryTypeBits, props);

    const vk::MemoryAllocateInfo allocInfo{
        .allocationSize = reqs.size,
        .memoryTypeIndex = typeIndex,
    };
    outMemory = vk::raii::DeviceMemory(device_, allocInfo);

    buffer.bindMemory(*outMemory, 0);
    return buffer;
}

void Renderer::copyToDevice(const void *data,
                            vk::DeviceSize size,
                            const vk::raii::Buffer &dst) {
    if (size > stagingCapacity_) {
        throw VulkanError("staging buffer too small");
    }
    std::memcpy(stagingMapped_, data, size);

    const vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *commandPool_,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    };
    auto cmdBuffers = device_.allocateCommandBuffers(allocInfo);

    cmdBuffers[0].begin({});
    const std::array<vk::BufferCopy, 1> regions{
        vk::BufferCopy{.srcOffset = 0, .dstOffset = 0, .size = size},
    };
    cmdBuffers[0].copyBuffer(*stagingBuffer_, *dst, regions);
    cmdBuffers[0].end();

    graphicsQueue_.submit(vk::SubmitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &*cmdBuffers[0],
    });
    graphicsQueue_.waitIdle();
}

void Renderer::createBuffers() {
    constexpr std::array<Vertex, 3> vertices{
        Vertex{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
        Vertex{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
        Vertex{{0.0f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    };
    constexpr std::array<uint16_t, 3> indices{0, 1, 2};

    constexpr vk::DeviceSize vertexSize = sizeof(Vertex) * vertices.size();
    constexpr vk::DeviceSize indexSize = sizeof(uint16_t) * indices.size();

    // One staging buffer serves both copies: each is followed by a queue wait,
    // so the copies can never overlap.
    stagingCapacity_ = std::max(vertexSize, indexSize);
    stagingBuffer_ = createBuffer(
        stagingCapacity_,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBufferMemory_);
    stagingMapped_ = stagingBufferMemory_.mapMemory(0, stagingCapacity_);

    vertexBuffer_ = createBuffer(
        vertexSize,
        vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        vertexBufferMemory_);
    copyToDevice(vertices.data(), vertexSize, vertexBuffer_);

    indexBuffer_ = createBuffer(
        indexSize,
        vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eIndexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        indexBufferMemory_);
    copyToDevice(indices.data(), indexSize, indexBuffer_);
}

void Renderer::createSyncObjects() {
    const vk::SemaphoreCreateInfo semInfo{};
    imageAvailable_ = vk::raii::Semaphore(device_, semInfo);
    renderFinished_ = vk::raii::Semaphore(device_, semInfo);

    // Signalled, so the first frame does not wait on a fence that was never
    // submitted to.
    inFlight_ = vk::raii::Fence(device_, vk::FenceCreateInfo{
                                          .flags = vk::FenceCreateFlagBits::eSignaled,
                                      });
}

void Renderer::recordCommandBuffer(uint32_t imageIndex) {
    // begin() implicitly resets the buffer, so one allocation is reused for
    // every frame.
    commandBuffers_[0].begin({});

    // Swapchain images are handed over in eUndefined, so each frame has to
    // transition the image before the dynamic render pass starts.
    const vk::ImageMemoryBarrier2 toAttachment{
        .srcStageMask = vk::PipelineStageFlagBits2::eNone,
        .srcAccessMask = vk::AccessFlagBits2::eNone,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = swapchainImages_[imageIndex],
        .subresourceRange = vk::ImageSubresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .levelCount = 1,
            .layerCount = 1,
        },
    };
    commandBuffers_[0].pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &toAttachment,
    });

    vk::ClearValue clearValue{};
    clearValue.color = vk::ClearColorValue(
        std::array<float, 4>{0.02f, 0.02f, 0.06f, 1.0f});

    const vk::RenderingAttachmentInfo colorAttachment{
        .imageView = *imageViews_[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clearValue,
    };
    const vk::RenderingInfo renderingInfo{
        .renderArea = vk::Rect2D{{0, 0}, extent_},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachment,
    };

    commandBuffers_[0].beginRendering(renderingInfo);
    commandBuffers_[0].bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline_);

    commandBuffers_[0].setViewport(0, vk::Viewport{
                                       .x = 0.0f,
                                       .y = 0.0f,
                                       .width = static_cast<float>(extent_.width),
                                       .height = static_cast<float>(extent_.height),
                                       .minDepth = 0.0f,
                                       .maxDepth = 1.0f,
                                   });
    commandBuffers_[0].setScissor(0, vk::Rect2D{{0, 0}, extent_});

    commandBuffers_[0].bindVertexBuffers(0, *vertexBuffer_, {0});
    commandBuffers_[0].bindIndexBuffer(*indexBuffer_, 0, vk::IndexType::eUint16);
    commandBuffers_[0].drawIndexed(kIndexCount, 1, 0, 0, 0);

    commandBuffers_[0].endRendering();
    commandBuffers_[0].end();
}

void Renderer::saveScreenshot(const std::string &path) {
    if (swapchainImages_.empty() || extent_.width == 0 || extent_.height == 0) {
        throw VulkanError("nothing rendered yet");
    }

    // Derive the row pitch from the format instead of assuming 4 bytes per
    // pixel: a surface that only offers a wider format would otherwise size
    // the readback buffer too small and overrun it while writing the PPM.
    // Only 8-bit channels can be handed to PPM without a colour conversion,
    // so anything wider is refused rather than silently mis-decoded.
    const auto bpp = bytesPerPixel(surfaceFormat_.format);
    if (!bpp.has_value() || *bpp != 4) {
        throw VulkanError(
            "screenshot needs an 8-bit-per-channel swapchain format, got " +
            std::string(vk::to_string(surfaceFormat_.format)));
    }

    const vk::DeviceSize bufferSize =
        static_cast<vk::DeviceSize>(extent_.width) * extent_.height * *bpp;

    vk::raii::DeviceMemory readbackMemory = nullptr;
    auto readback = createBuffer(bufferSize,
                                 vk::BufferUsageFlagBits::eTransferDst,
                                 vk::MemoryPropertyFlagBits::eHostVisible |
                                     vk::MemoryPropertyFlagBits::eHostCoherent,
                                 readbackMemory);

    // The image written by the most recent drawFrame, not necessarily [0] —
    // the swapchain rotates its images, and with eUndefined transitions the
    // other ones hold stale or undefined contents.
    const uint32_t imageIndex = lastImageIndex_ % swapchainImages_.size();
    const vk::Image image = swapchainImages_[imageIndex];

    auto cmd = device_.allocateCommandBuffers(
        vk::CommandBufferAllocateInfo{
            .commandPool = *commandPool_,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        });

    // eColorAttachmentOptimal -> eTransferSrcOptimal
    const vk::ImageMemoryBarrier2 toTransfer{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .newLayout = vk::ImageLayout::eTransferSrcOptimal,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = vk::ImageSubresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .levelCount = 1,
            .layerCount = 1,
        },
    };
    // eTransferSrcOptimal -> eColorAttachmentOptimal, so the next frame can
    // keep using the image.
    const vk::ImageMemoryBarrier2 back{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = vk::ImageSubresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .levelCount = 1,
            .layerCount = 1,
        },
    };

    cmd[0].begin({});
    cmd[0].pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &toTransfer,
    });
    const vk::BufferImageCopy region{
        .bufferOffset = 0,
        .imageSubresource = vk::ImageSubresourceLayers{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .imageOffset = {0, 0, 0},
        .imageExtent = vk::Extent3D{extent_.width, extent_.height, 1},
    };
    cmd[0].copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal,
                             *readback, region);
    cmd[0].pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &back,
    });
    cmd[0].end();

    graphicsQueue_.submit(vk::SubmitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &*cmd[0],
    });
    graphicsQueue_.waitIdle();

    void *mapped = readbackMemory.mapMemory(0, bufferSize);

    // PPM wants 8-bit RGB; the swapchain format is BGRA on virtually every
    // desktop implementation.
    const bool bgra = surfaceFormat_.format == vk::Format::eB8G8R8A8Srgb ||
                      surfaceFormat_.format == vk::Format::eB8G8R8A8Unorm ||
                      surfaceFormat_.format == vk::Format::eA8B8G8R8SrgbPack32 ||
                      surfaceFormat_.format == vk::Format::eA8B8G8R8UnormPack32;

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw VulkanError("cannot write " + path);
    }
    out << "P6\n" << extent_.width << " " << extent_.height << "\n255\n";

    const auto *pixels = static_cast<const uint8_t *>(mapped);
    for (uint32_t y = 0; y < extent_.height; ++y) {
        for (uint32_t x = 0; x < extent_.width; ++x) {
            const size_t i =
                (static_cast<size_t>(y) * extent_.width + x) * *bpp;
            const uint8_t r = bgra ? pixels[i + 2] : pixels[i + 0];
            const uint8_t g = pixels[i + 1];
            const uint8_t b = bgra ? pixels[i + 0] : pixels[i + 2];
            out.put(static_cast<char>(r));
            out.put(static_cast<char>(g));
            out.put(static_cast<char>(b));
        }
    }

    readbackMemory.unmapMemory();
}

void Renderer::drawFrame() {
    const auto [acquireResult, imageIndex] =
        swapchain_.acquireNextImage(UINT64_MAX, *imageAvailable_);

    if (acquireResult == vk::Result::eErrorOutOfDateKHR) {
        recreateSwapchain();
        return;
    }
    if (acquireResult != vk::Result::eSuccess &&
        acquireResult != vk::Result::eSuboptimalKHR) {
        throw VulkanError("acquireNextImage failed");
    }

    if (device_.waitForFences(*inFlight_, VK_TRUE, UINT64_MAX) !=
        vk::Result::eSuccess) {
        throw VulkanError("waitForFences failed");
    }

    // Reset the fence for the next frame. vk::raii::Fence has no reset(), and
    // vk::raii::Device::resetFences returns void — it reports failure by
    // throwing, so there is no result to compare here.
    device_.resetFences(*inFlight_);

    recordCommandBuffer(imageIndex);
    lastImageIndex_ = imageIndex;

    const vk::PipelineStageFlags waitStage =
        vk::PipelineStageFlagBits::eColorAttachmentOutput;

    graphicsQueue_.submit(
        vk::SubmitInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*imageAvailable_,
            .pWaitDstStageMask = &waitStage,
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffers_[0],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*renderFinished_,
        },
        *inFlight_);

    const vk::SwapchainKHR raw = *swapchain_;
    const auto presentResult = graphicsQueue_.presentKHR(vk::PresentInfoKHR{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*renderFinished_,
        .swapchainCount = 1,
        .pSwapchains = &raw,
        .pImageIndices = &imageIndex,
    });

    if (presentResult == vk::Result::eErrorOutOfDateKHR ||
        presentResult == vk::Result::eSuboptimalKHR) {
        recreateSwapchain();
    } else if (presentResult != vk::Result::eSuccess) {
        throw VulkanError("presentKHR failed");
    }
}

void Renderer::recreateSwapchain() {
    device_.waitIdle();

    destroySwapchain();
    createSwapchain();
    createImageViews();
    createPipelines();
    createCommandBuffers();

    // The new swapchain may have a different image count, and none of its
    // images has been rendered into yet.
    lastImageIndex_ = 0;
    framebufferResized_ = false;
}

void Renderer::framebufferResizeCallback(GLFWwindow *window, int, int) {
    auto *renderer = static_cast<Renderer *>(glfwGetWindowUserPointer(window));
    if (renderer != nullptr) {
        renderer->framebufferResized_ = true;
    }
}

void Renderer::run() {
    const double start = glfwGetTime();
    uint32_t frames = 0;

    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();

        if (framebufferResized_) {
            recreateSwapchain();
        }

        drawFrame();

        if (maxFrames_ != 0 && ++frames >= maxFrames_) {
            const double elapsed = glfwGetTime() - start;
            std::cout << "[perf] " << frames << " frames in " << elapsed << "s ("
                      << static_cast<double>(frames) / elapsed << " fps)\n";
            break;
        }
    }

    device_.waitIdle();

    try {
        saveScreenshot("screenshot.ppm");
        std::cout << "[shot] wrote screenshot.ppm (" << extent_.width << "x"
                  << extent_.height << ")\n";
    } catch (const std::exception &error) {
        std::cerr << "[shot] " << error.what() << "\n";
    }
}

} // namespace tri
