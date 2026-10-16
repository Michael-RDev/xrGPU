cmake_minimum_required(VERSION 3.24)
project(xrGPU LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT ANDROID OR NOT ANDROID_ABI STREQUAL "arm64-v8a")
  message(FATAL_ERROR "xrGPU native app targets Android arm64-v8a. See docs/setup.md.")
endif()
set(EXECUTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/deps/executorch")
if(NOT PYTHON_EXECUTABLE)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "Export environment")
endif()
set(EXECUTORCH_BUILD_VULKAN ON CACHE BOOL "" FORCE)
set(EXECUTORCH_ENABLE_EVENT_TRACER ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_DEVTOOLS ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_XNNPACK ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXECUTOR_RUNNER OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_VULKAN_SHADER_COMPILE_NTHREADS 4 CACHE STRING "" FORCE)
add_subdirectory(artifacts/deps/executorch et)
target_include_directories(vulkan_backend PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_sources(vulkan_backend PRIVATE runtime/native/model_ops.cpp)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_API_LAYERS OFF CACHE BOOL "" FORCE)
set(BUILD_LOADER ON CACHE BOOL "" FORCE)
set(DYNAMIC_LOADER ON CACHE BOOL "" FORCE)
add_subdirectory(artifacts/deps/openxr xr)

file(GLOB_RECURSE revision_sources CONFIGURE_DEPENDS
  kernels/*.glsl kernels/*.yaml kernels/*.json kernels/*.h devices/*.h
  runtime/native/*.h runtime/native/*.cpp
  runtime/patches/*.patch app/src/main/*.java app/src/main/*.xml
  app/src/debug/*.java app/src/debug/*.xml)
list(APPEND revision_sources "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
  "${CMAKE_CURRENT_SOURCE_DIR}/app/build.gradle" "${CMAKE_CURRENT_SOURCE_DIR}/dependencies.json"
  "${CMAKE_CURRENT_SOURCE_DIR}/runtime/windows-toolchain.json")
list(SORT revision_sources)
set(revision_material "")
foreach(source IN LISTS revision_sources)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source}")
  file(SHA256 "${source}" digest)
  string(APPEND revision_material "${digest}")
endforeach()
string(SHA256 kernel_revision "${revision_material}")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/dependencies.json" dependencies)
string(JSON et_revision GET "${dependencies}" executorch commit)
file(WRITE "${CMAKE_BINARY_DIR}/build-identity.json"
  "{\"kernel_revision\":\"${kernel_revision}\",\"runtime_revision\":\"${et_revision}\"}\n")

add_library(xrgpu SHARED runtime/native/jni.cpp runtime/native/scene.cpp
  "${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c")
target_include_directories(xrgpu PRIVATE runtime/native "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/deps"
  "${EXECUTORCH_ROOT}/third-party/json/single_include"
  "${EXECUTORCH_ROOT}/backends/vulkan/third-party/Vulkan-Headers/include"
  "${ANDROID_NDK}/sources/android/native_app_glue")
target_compile_definitions(xrgpu PRIVATE XR_USE_PLATFORM_ANDROID XR_USE_GRAPHICS_API_OPENGL_ES
  XRGPU_KERNEL_REVISION="${kernel_revision}" XRGPU_ET_REVISION="${et_revision}")
target_link_libraries(xrgpu PRIVATE executorch extension_module extension_tensor
  portable_ops_lib vulkan_backend xnnpack_backend openxr_loader android log EGL GLESv3 dl)
target_link_options(xrgpu PRIVATE "-Wl,-u,ANativeActivity_onCreate" "-Wl,-z,max-page-size=16384")
install(TARGETS xrgpu openxr_loader LIBRARY DESTINATION lib/arm64-v8a)
