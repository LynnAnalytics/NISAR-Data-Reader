cmake_minimum_required(VERSION 4.2)

project(
  NISARDataReader
  VERSION 0.6.0
  DESCRIPTION "CUDA-first native NISAR GSLC/GCOV viewer with CPU fallback"
  LANGUAGES CXX
)

option(SATVIEW_ENABLE_CUDA "Build CUDA scientific transforms" ON)
option(
  SATVIEW_EXPERIMENTAL_BACKENDS
  "Allow explicitly selected experimental compute backends"
  OFF
)
option(
  SATVIEW_ENABLE_EXPERIMENTAL_HIP
  "Build the experimental AMD HIP/ROCm compute backend"
  OFF
)
option(
  SATVIEW_ENABLE_EXPERIMENTAL_SYCL
  "Build the experimental Intel oneAPI/SYCL compute backend"
  OFF
)
option(SATVIEW_BUILD_TESTS "Build unit and integration tests" ON)
option(SATVIEW_BUILD_VIEWER "Build the Vulkan desktop viewer" ON)

if(
  (SATVIEW_ENABLE_EXPERIMENTAL_HIP OR
   SATVIEW_ENABLE_EXPERIMENTAL_SYCL) AND
  NOT SATVIEW_EXPERIMENTAL_BACKENDS
)
  message(FATAL_ERROR
    "Experimental HIP/SYCL backends require "
    "SATVIEW_EXPERIMENTAL_BACKENDS=ON"
  )
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
    $<$<COMPILE_LANGUAGE:CXX>:/permissive->
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
  )
else()
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  )
endif()

find_package(hdf5 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(ZLIB REQUIRED)

set(SATVIEW_CUDA_AVAILABLE OFF)
if(SATVIEW_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit 13.3 QUIET)
    if(CUDAToolkit_FOUND)
      set(SATVIEW_CUDA_AVAILABLE ON)
    else()
      message(WARNING
        "CUDA Toolkit 13.3 was not found; building the CPU fallback"
      )
    endif()
  else()
    message(STATUS
      "No CUDA compiler was found; building the CPU fallback"
    )
  endif()
endif()

add_library(satview_core
  src/cpu/pageable_ring.cpp
  src/cpu/scientific.cpp
  src/hdf5_product.cpp
  src/overview_builder.cpp
  src/view_navigation.cpp
)
add_library(satview::core ALIAS satview_core)

target_include_directories(
  satview_core
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_link_libraries(
  satview_core
  PUBLIC
    hdf5::hdf5-shared
  PRIVATE
    ZLIB::ZLIB
)
target_compile_features(satview_core PUBLIC cxx_std_20)

if(SATVIEW_CUDA_AVAILABLE)
  add_library(satview_gpu
    src/gpu/distribution.cu
    src/gpu/transforms.cu
    src/gpu/speckle_filter.cu
    src/gpu/pinned_ring.cpp
  )
  add_library(satview::gpu ALIAS satview_gpu)

  target_include_directories(
    satview_gpu
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  )
  target_link_libraries(
    satview_gpu
    PUBLIC
      CUDA::cudart
  )
  target_compile_features(satview_gpu PUBLIC cxx_std_20)
  set_target_properties(
    satview_gpu
    PROPERTIES
      CUDA_ARCHITECTURES 120
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON
      POSITION_INDEPENDENT_CODE ON
  )
else()
  target_sources(satview_core PRIVATE src/distribution.cpp)
endif()

add_executable(sat-inspect apps/inspect_main.cpp)
target_link_libraries(sat-inspect PRIVATE satview::core nlohmann_json::nlohmann_json)

add_executable(sat-bench apps/bench_main.cpp)
target_link_libraries(sat-bench PRIVATE satview::core)
if(SATVIEW_CUDA_AVAILABLE)
  target_link_libraries(sat-bench PRIVATE satview::gpu)
  target_compile_definitions(sat-bench PRIVATE SATVIEW_HAS_CUDA=1)
endif()

install(TARGETS sat-inspect sat-bench RUNTIME DESTINATION .)
install(FILES LICENSE THIRD_PARTY_NOTICES.md DESTINATION .)
install(
  FILES
    README.md
    ARCHITECTURE.md
    PERFORMANCE.md
    COLORMAP_ATTRIBUTION.md
  DESTINATION docs
)

if(MSVC)
  # vcpkg's app-local deployment places the configuration-correct HDF5/zlib
  # DLLs beside the executable. Install those rather than hard-coding release
  # paths, so Debug packages do not mix incompatible runtimes.
  install(
    DIRECTORY "$<TARGET_FILE_DIR:sat-inspect>/"
    DESTINATION .
    FILES_MATCHING PATTERN "*.dll"
  )
  set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ".")
  include(InstallRequiredSystemLibraries)
endif()

if(SATVIEW_EXPERIMENTAL_BACKENDS AND SATVIEW_ENABLE_EXPERIMENTAL_HIP)
  check_language(HIP)
  if(NOT CMAKE_HIP_COMPILER)
    message(FATAL_ERROR
      "The experimental HIP backend requires a ROCm HIP compiler. "
      "Set CMAKE_HIP_COMPILER to amdclang++ from the ROCm SDK."
    )
  endif()
  enable_language(HIP)
  find_package(hip CONFIG REQUIRED)
  add_library(satview_hip_backend src/experimental/hip/scientific.hip)
  add_library(satview::hip_backend ALIAS satview_hip_backend)
  target_include_directories(satview_hip_backend PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  )
  target_link_libraries(satview_hip_backend PUBLIC hip::host)
  target_compile_definitions(
    satview_hip_backend PUBLIC SATVIEW_HAS_EXPERIMENTAL_HIP=1
  )
  set_target_properties(satview_hip_backend PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
  )
endif()

if(SATVIEW_EXPERIMENTAL_BACKENDS AND SATVIEW_ENABLE_EXPERIMENTAL_SYCL)
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    message(FATAL_ERROR
      "The experimental oneAPI/SYCL backend requires the IntelLLVM "
      "compiler. Configure with icx plus Ninja on Windows."
    )
  endif()
  find_package(IntelSYCL REQUIRED)
  add_library(satview_sycl_backend src/experimental/sycl/scientific.cpp)
  add_library(satview::sycl_backend ALIAS satview_sycl_backend)
  target_include_directories(satview_sycl_backend PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  )
  target_compile_definitions(
    satview_sycl_backend PUBLIC SATVIEW_HAS_EXPERIMENTAL_SYCL=1
  )
  target_compile_features(satview_sycl_backend PUBLIC cxx_std_20)
  add_sycl_to_target(
    TARGET satview_sycl_backend
    SOURCES src/experimental/sycl/scientific.cpp
  )
endif()

if(SATVIEW_BUILD_VIEWER)
  add_subdirectory(viewer)
endif()

if(SATVIEW_BUILD_TESTS)
  include(CTest)
  add_executable(satview-tests
    tests/colormap_tests.cpp
    tests/cpu_scientific_tests.cpp
    tests/distribution_tests.cpp
    tests/overview_builder_tests.cpp
    tests/real_mosaic_plan_tests.cpp
    tests/resident_view_tests.cpp
    tests/test_main.cpp
    tests/viewer_math_tests.cpp
    viewer/colormaps.cpp
  )
  target_include_directories(satview-tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/viewer"
  )
  target_link_libraries(satview-tests PRIVATE satview::core)
  if(SATVIEW_CUDA_AVAILABLE)
    target_link_libraries(satview-tests PRIVATE satview::gpu)
    target_sources(satview-tests PRIVATE
      tests/gpu_distribution_tests.cu
      tests/gpu_transform_tests.cu
    )
    target_compile_definitions(satview-tests PRIVATE SATVIEW_HAS_CUDA=1)
    set_source_files_properties(
      tests/gpu_distribution_tests.cu
      tests/gpu_transform_tests.cu
      PROPERTIES LANGUAGE CUDA
    )
    set_target_properties(satview-tests PROPERTIES CUDA_ARCHITECTURES 120)

    add_executable(
      satview-speckle-tests
      tests/gpu_speckle_filter_tests.cu
    )
    target_link_libraries(satview-speckle-tests PRIVATE satview::gpu)
    target_compile_features(satview-speckle-tests PRIVATE cxx_std_20)
    set_source_files_properties(
      tests/gpu_speckle_filter_tests.cu
      PROPERTIES LANGUAGE CUDA
    )
    set_target_properties(
      satview-speckle-tests
      PROPERTIES
        CUDA_ARCHITECTURES 120
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
    add_test(NAME satview-speckle-tests COMMAND satview-speckle-tests)
  endif()
  target_compile_definitions(
    satview-tests PRIVATE SATVIEW_DEFAULT_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/Test Data"
  )
  add_test(NAME satview-tests COMMAND satview-tests)

  add_executable(satview-navigation-tests
    tests/view_navigation_tests.cpp
  )
  target_link_libraries(satview-navigation-tests PRIVATE satview::core)
  add_test(NAME satview-navigation-tests COMMAND satview-navigation-tests)

  if(TARGET satview::hip_backend OR TARGET satview::sycl_backend)
    add_executable(
      satview-experimental-backend-tests
      tests/experimental_backend_tests.cpp
    )
    target_link_libraries(
      satview-experimental-backend-tests PRIVATE satview::core
    )
    if(TARGET satview::hip_backend)
      target_link_libraries(
        satview-experimental-backend-tests PRIVATE satview::hip_backend
      )
    endif()
    if(TARGET satview::sycl_backend)
      target_link_libraries(
        satview-experimental-backend-tests PRIVATE satview::sycl_backend
      )
    endif()
    add_test(
      NAME satview-experimental-backend-tests
      COMMAND satview-experimental-backend-tests
    )
  endif()
endif()
