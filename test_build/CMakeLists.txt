cmake_minimum_required(VERSION 3.18)
project(hpdexc LANGUAGES CXX)

# ========== 基础配置 ==========
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # 保证可用于共享库
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 默认 Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")

# ========== Python & pybind11 ==========
set(Python3_FIND_STRATEGY LOCATION)
set(Python3_FIND_IMPLEMENTATIONS CPython)
if(NOT DEFINED Python3_EXECUTABLE)
  if(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python")
  endif()
endif()
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# ========== OpenMP ==========
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
endif()

# ========== Highway (Google VQSort) ==========
include(FetchContent)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG 1.1.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(highway)

# ========== indicators ==========
FetchContent_Declare(
  indicators
  GIT_REPOSITORY https://github.com/p-ranav/indicators.git
  GIT_TAG v2.3
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(indicators)

# ========== Python 扩展模块 ==========
# 入口文件必须是 module.cpp
pybind11_add_module(kernel src/cpp/module.cpp)

# 其余 cpp 文件单独加进来，保证有独立的 .o
target_sources(kernel PRIVATE
    src/cpp/mannwhitneyu.cpp
    src/cpp/reader.cpp
    src/cpp/sort.cpp
)

target_include_directories(kernel PRIVATE 
  ${CMAKE_SOURCE_DIR}/src/cpp
  ${CMAKE_SOURCE_DIR}/src/cpp/include
)

set_target_properties(kernel PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/python/hpdex/backend
)

# ========== VQSort static/动态模式开关 ==========
option(HPDEXC_VQSORT_STATIC "Use VQSort static inline mode" ON)
if(HPDEXC_VQSORT_STATIC)
    target_compile_definitions(kernel PRIVATE VQSORT_ONLY_STATIC=1)
    message(STATUS "VQSort: static inline mode enabled (VQSORT_ONLY_STATIC=1)")
else()
    target_compile_definitions(kernel PRIVATE VQSORT_ONLY_STATIC=0)
    message(STATUS "VQSort: dynamic dispatch mode enabled (VQSORT_ONLY_STATIC=0)")
endif()

# ========== 编译模式 ==========
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(kernel PRIVATE -g -O0 -DDEBUG -fno-omit-frame-pointer -gdwarf-4)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-fstandalone-debug" COMPILER_SUPPORTS_STANDALONE_DEBUG)
    if(COMPILER_SUPPORTS_STANDALONE_DEBUG)
        target_compile_options(kernel PRIVATE -fstandalone-debug)
    endif()
    message(STATUS "Building in Debug mode for LLDB/GDB")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(kernel PRIVATE -g -O2 -DNDEBUG)
    message(STATUS "Building in RelWithDebInfo mode")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(kernel PRIVATE -O3 -march=native -DNDEBUG)
    message(STATUS "Building in Release mode")
endif()

# ========== OpenMP 多线程选项 ==========
option(HPDEXC_ENABLE_MULTITHREAD "Enable OpenMP multithreading support" ON)
if(HPDEXC_ENABLE_MULTITHREAD)
    target_compile_definitions(kernel PRIVATE HPDEXC_ST=0)
    message(STATUS "Multithreading enabled (HPDEXC_ST=0)")
else()
    target_compile_definitions(kernel PRIVATE HPDEXC_ST=1)
    message(STATUS "Multithreading disabled (HPDEXC_ST=1)")
endif()

# ========== 其它编译选项 ==========
target_compile_definitions(kernel PRIVATE HPDEXC_WITH_PYBIND=1)
target_compile_definitions(kernel PRIVATE HPDEXC_WITH_INDICATORS=1)
target_compile_definitions(kernel PRIVATE HPDEXC_HAS_F16=0)
target_compile_definitions(kernel PRIVATE HPDEXC_ENABLE_ALL_OPTIMIZATIONS=1)

message(STATUS "Python bindings enabled (HPDEXC_WITH_PYBIND=1)")
message(STATUS "Indicators progress bar enabled (HPDEXC_WITH_INDICATORS=1)")
message(STATUS "Float16 disabled (HPDEXC_HAS_F16=0)")
message(STATUS "All optimizations enabled (HPDEXC_ENABLE_ALL_OPTIMIZATIONS=1)")

# ========== 链接依赖库 ==========
target_link_libraries(kernel PRIVATE hwy hwy_contrib indicators::indicators)

if(APPLE)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(kernel PRIVATE OpenMP::OpenMP_CXX)
  else()
    target_include_directories(kernel PRIVATE /opt/homebrew/opt/libomp/include)
    target_link_directories(kernel PRIVATE /opt/homebrew/opt/libomp/lib)
    target_link_libraries(kernel PRIVATE omp)
    target_compile_options(kernel PRIVATE -Xpreprocessor -fopenmp)
  endif()
endif()

# ========== Debug: compile_commands.json symlink ==========
option(HPDEXC_LINK_COMPILE_COMMANDS "Symlink compile_commands.json at source root" ON)
if(HPDEXC_LINK_COMPILE_COMMANDS)
  add_custom_target(link_compile_commands ALL
      COMMAND ${CMAKE_COMMAND} -E create_symlink
              ${CMAKE_BINARY_DIR}/compile_commands.json
              ${CMAKE_SOURCE_DIR}/compile_commands.json
      COMMENT "Creating compile_commands.json symlink for clangd/IDE"
  )
endif()

# ========== 调试信息 ==========
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
get_target_property(KERNEL_DEFS kernel COMPILE_DEFINITIONS)
message(STATUS "kernel COMPILE_DEFINITIONS = ${KERNEL_DEFS}")
