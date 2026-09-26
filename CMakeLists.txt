cmake_minimum_required(VERSION 3.24)
project(vulkan-triangle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan 1.3 REQUIRED)
find_package(glfw3 3.4 REQUIRED)
find_package(glm REQUIRED)
find_program(GLSLC_EXECUTABLE glslc REQUIRED)

set(SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_DIR})

set(SHADER_OUTPUTS)
foreach(shader triangle.vert triangle.frag)
    string(REPLACE "." "_" stem ${shader})
    set(dst ${SHADER_DIR}/${stem}.spv)
    add_custom_command(
        OUTPUT ${dst}
        COMMAND ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader} -o ${dst}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}
        COMMENT "glslc ${shader}"
        VERBATIM
    )
    list(APPEND SHADER_OUTPUTS ${dst})
endforeach()

add_custom_target(shaders DEPENDS ${SHADER_OUTPUTS})

add_executable(vulkan_triangle
    src/main.cpp
    src/renderer.cpp
)

target_link_libraries(vulkan_triangle PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
)

target_compile_definitions(vulkan_triangle PRIVATE
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    SHADER_DIR="${SHADER_DIR}"
)

target_compile_options(vulkan_triangle PRIVATE -Wall -Wextra -Wpedantic)

add_dependencies(vulkan_triangle shaders)
