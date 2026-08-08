if(NOT TARGET satview::gpu)
  message(FATAL_ERROR "The native viewer requires SATVIEW_ENABLE_CUDA=ON")
endif()

find_package(SDL3 CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(imgui CONFIG REQUIRED)

find_program(
  SATVIEW_GLSLANG_VALIDATOR
  NAMES glslangValidator
  HINTS
    "${VCPKG_INSTALLED_DIR}/${VCPKG_HOST_TRIPLET}/tools/glslang"
    "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/glslang"
    "$ENV{VULKAN_SDK}/Bin"
  REQUIRED
)

set(SATVIEW_SHADER_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SATVIEW_SHADER_DIRECTORY}")

set(SATVIEW_VERTEX_SOURCE "${CMAKE_SOURCE_DIR}/shaders/scientific.vert")
set(SATVIEW_FRAGMENT_SOURCE "${CMAKE_SOURCE_DIR}/shaders/scientific.frag")
set(SATVIEW_VERTEX_SPIRV "${SATVIEW_SHADER_DIRECTORY}/scientific.vert.spv")
set(SATVIEW_FRAGMENT_SPIRV "${SATVIEW_SHADER_DIRECTORY}/scientific.frag.spv")

# Compile the checked-in scientific shaders offline; no runtime compiler is linked.
add_custom_command(
  OUTPUT "${SATVIEW_VERTEX_SPIRV}"
  COMMAND
    "${SATVIEW_GLSLANG_VALIDATOR}"
    --target-env vulkan1.2
    -V
    -S vert
    -o "${SATVIEW_VERTEX_SPIRV}"
    "${SATVIEW_VERTEX_SOURCE}"
  DEPENDS "${SATVIEW_VERTEX_SOURCE}"
  VERBATIM
  COMMENT "Compiling scientific vertex shader"
)

add_custom_command(
  OUTPUT "${SATVIEW_FRAGMENT_SPIRV}"
  COMMAND
    "${SATVIEW_GLSLANG_VALIDATOR}"
    --target-env vulkan1.2
    -V
    -S frag
    -o "${SATVIEW_FRAGMENT_SPIRV}"
    "${SATVIEW_FRAGMENT_SOURCE}"
  DEPENDS "${SATVIEW_FRAGMENT_SOURCE}"
  VERBATIM
  COMMENT "Compiling scientific fragment shader"
)

add_custom_target(
  satview-viewer-shaders
  DEPENDS "${SATVIEW_VERTEX_SPIRV}" "${SATVIEW_FRAGMENT_SPIRV}"
)

add_executable(
  sat-viewer
  main.cpp
  colormap_lut.cpp
  colormaps.cpp
  overview_worker.cpp
  "${CMAKE_SOURCE_DIR}/src/gpu/vulkan_interop.cpp"
  "${SATVIEW_VERTEX_SPIRV}"
  "${SATVIEW_FRAGMENT_SPIRV}"
)
add_dependencies(sat-viewer satview-viewer-shaders)

add_custom_command(
  TARGET sat-viewer
  POST_BUILD
  COMMAND
    "${CMAKE_COMMAND}" -E make_directory
    "$<TARGET_FILE_DIR:sat-viewer>/shaders"
  COMMAND
    "${CMAKE_COMMAND}" -E copy_if_different
    "${SATVIEW_VERTEX_SPIRV}" "${SATVIEW_FRAGMENT_SPIRV}"
    "$<TARGET_FILE_DIR:sat-viewer>/shaders"
  VERBATIM
)

target_link_libraries(
  sat-viewer
  PRIVATE
    satview::core
    satview::gpu
    SDL3::SDL3
    Vulkan::Vulkan
    imgui::imgui
    Comdlg32
)
target_compile_features(sat-viewer PRIVATE cxx_std_20)
target_compile_definitions(
  sat-viewer
  PRIVATE
    VK_USE_PLATFORM_WIN32_KHR=1
    SDL_MAIN_HANDLED=1
)

if(MSVC)
  target_compile_options(sat-viewer PRIVATE /bigobj)
endif()

install(
  TARGETS sat-viewer
  RUNTIME DESTINATION .
)

if(MSVC)
  install(
    DIRECTORY "$<TARGET_FILE_DIR:sat-viewer>/"
    DESTINATION .
    FILES_MATCHING PATTERN "*.dll"
  )
endif()

install(
  FILES
    "${SATVIEW_VERTEX_SPIRV}"
    "${SATVIEW_FRAGMENT_SPIRV}"
  DESTINATION shaders
)
