#include "renderer.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>

namespace tri {

namespace {

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

Renderer::Renderer() {
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
    glfwDestroyWindow(window_);
    glfwTerminate();
}

void Renderer::glfwErrorCallback(int code, const char *desc) {
    std::cerr << "[glfw] error " << code << ": " << desc << "\n";
}

void Renderer::createWindow() {
    glfwSetErrorCallback(&glfwErrorCallback);

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
}

std::vector<const char *> Renderer::requiredInstanceExtensions() const {
    uint32_t count = 0;
    const char **raw = glfwGetRequiredInstanceExtensions(&count);

    std::vector<const char *> result(raw, raw + count);
    result.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
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
    // creation, which a standalone messenger would miss.
    const vk::DebugUtilsMessengerCreateInfoEXT debugInfo{
        .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose,
        .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        .pfnUserCallback = &Renderer::debugCallback,
    };

    const vk::InstanceCreateInfo createInfo{
        .pNext = &debugInfo,
        .pApplicationInfo = &appInfo,
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    instance_ = vk::raii::Instance(context_, createInfo);
}

void Renderer::setupDebugMessenger() {
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
    for (const auto &format : formats) {
        if (format.format == vk::Format::eB8G8R8A8Srgb &&
            format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return format;
        }
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

    const vk::FenceCreateInfo fenceInfo{
        .flags = vk::FenceCreateFlagBits::eSignaled,
    };
    inFlight_ = vk::raii::Fence(device_, fenceInfo);
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
    inFlight_ = nullptr;

    recordCommandBuffer(imageIndex);

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

        if (++frames >= kExitAfterFrames) {
            const double elapsed = glfwGetTime() - start;
            std::cout << "[perf] " << frames << " frames in " << elapsed << "s ("
                      << static_cast<double>(frames) / elapsed << " fps)\n";
            break;
        }
    }

    device_.waitIdle();
}

} // namespace tri
