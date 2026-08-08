cmake_minimum_required(VERSION 4.2)

project(
  NISARDataReader
  VERSION 0.5.4
  DESCRIPTION "GPU-first native NISAR GSLC/GCOV viewer"
  LANGUAGES CXX
)

option(SATVIEW_ENABLE_CUDA "Build CUDA scientific transforms" ON)
option(SATVIEW_BUILD_TESTS "Build unit and integration tests" ON)
option(SATVIEW_BUILD_VIEWER "Build the Vulkan desktop viewer" ON)

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

add_library(satview_core
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

if(SATVIEW_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit 13.3 REQUIRED)

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
endif()

add_executable(sat-inspect apps/inspect_main.cpp)
target_link_libraries(sat-inspect PRIVATE satview::core nlohmann_json::nlohmann_json)

add_executable(sat-bench apps/bench_main.cpp)
target_link_libraries(sat-bench PRIVATE satview::core)
if(SATVIEW_ENABLE_CUDA)
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

if(SATVIEW_BUILD_VIEWER)
  add_subdirectory(viewer)
endif()

if(SATVIEW_BUILD_TESTS)
  include(CTest)
  add_executable(satview-tests
    tests/colormap_tests.cpp
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
  if(SATVIEW_ENABLE_CUDA)
    target_link_libraries(satview-tests PRIVATE satview::gpu)
    target_sources(satview-tests PRIVATE
      tests/distribution_tests.cpp
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
endif()

