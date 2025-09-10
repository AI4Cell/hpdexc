cmake_minimum_required(VERSION 3.18)
project(hpdexc LANGUAGES CXX)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 生成 compile_commands.json (供 clangd)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 使用 Make 生成器（与 Makefile 兼容）
if(NOT CMAKE_GENERATOR)
  set(CMAKE_GENERATOR "Unix Makefiles" CACHE INTERNAL "" FORCE)
endif()

# pybind11 & Python
# 优先使用传入的 Python3 可执行文件（例如 -DPython3_EXECUTABLE=.venv/bin/python）
# 并强制以 LOCATION 策略查找，避免选择系统 Framework Python
set(Python3_FIND_STRATEGY LOCATION)
set(Python3_FIND_IMPLEMENTATIONS CPython)
if(NOT DEFINED Python3_EXECUTABLE)
  if(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python")
  endif()
endif()
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# OpenMP（macOS 需 libomp）
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
endif()

# ========= VQSort (Google Highway) =========
# 暂时注释掉 Highway 集成，先使用标准库实现
include(FetchContent)
# 关闭 Highway 的测试与示例，避免引入 googletest 配置
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG 1.1.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(highway)

# 头文件包含路径将在target_include_directories中设置

# ========= Reader (HDF5) helper =========
add_library(reader_obj OBJECT
  src/cpp/reader.cpp
)

# 设置头文件包含路径
target_include_directories(reader_obj PRIVATE 
  ${CMAKE_SOURCE_DIR}/src/cpp
  ${CMAKE_SOURCE_DIR}/src/cpp/include
)

# macOS 上链接 HDF5（借助 Homebrew 前缀）
if(APPLE)
  target_include_directories(reader_obj PRIVATE /opt/homebrew/opt/hdf5/include)
  target_link_directories(reader_obj PRIVATE /opt/homebrew/opt/hdf5/lib)
endif()

set(HPDEXC_SOURCES
  src/cpp/mannwhitney.cpp
  src/cpp/module.cpp
  src/cpp/reader.cpp
)

pybind11_add_module(kernel ${HPDEXC_SOURCES})

# 设置头文件包含路径
target_include_directories(kernel PRIVATE 
  ${CMAKE_SOURCE_DIR}/src/cpp
  ${CMAKE_SOURCE_DIR}/src/cpp/include
)

# 将编译产物直接输出到 Python 包目录，便于 import
set_target_properties(kernel PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/python/hpdex/backend
)

# 链接 HDF5（如果需要）
if(APPLE)
  target_link_libraries(kernel PRIVATE hdf5)
  target_link_directories(kernel PRIVATE /opt/homebrew/opt/hdf5/lib)
endif()

# 编译选项
target_compile_options(kernel PRIVATE -O3 -march=native)

# 禁用Float16支持以避免符号链接问题
target_compile_definitions(kernel PRIVATE HPDEXC_HAS_F16=0)

# 启用OpenMP多线程支持
target_compile_definitions(kernel PRIVATE HPDEXC_ST=0)

# 安装扩展到 Python 包目录 `hpdexc/`（可选）
install(TARGETS kernel DESTINATION ${CMAKE_SOURCE_DIR}/src/python/hpdex/backend)

# 条件编译选项
option(HPDEXC_ENABLE_ASSERTS "Enable runtime asserts in C++ code" ON)
option(HPDEXC_ENABLE_TEST_MACROS "Enable test-only macros in C++ code" OFF)
option(HPDEXC_FORCE_NO_INPLACE "Disallow any inplace operation at runtime" OFF)
option(HPDEXC_FORCE_ALWAYS_INPLACE "Force all operations to be inplace (fail if not possible)" OFF)
option(HPDEXC_DEBUG_POLICY "Enable policy decision logging" OFF)

target_compile_definitions(kernel
  PRIVATE
    $<$<BOOL:${HPDEXC_ENABLE_ASSERTS}>:HPDEXC_ENABLE_ASSERTS=1>
    $<$<NOT:$<BOOL:${HPDEXC_ENABLE_ASSERTS}>>:HPDEXC_ENABLE_ASSERTS=0>
    $<$<BOOL:${HPDEXC_ENABLE_TEST_MACROS}>:HPDEXC_ENABLE_TEST_MACROS=1>
    $<$<NOT:$<BOOL:${HPDEXC_ENABLE_TEST_MACROS}>>:HPDEXC_ENABLE_TEST_MACROS=0>
    $<$<BOOL:${HPDEXC_FORCE_NO_INPLACE}>:HPDEXC_FORCE_NO_INPLACE=1>
    $<$<BOOL:${HPDEXC_FORCE_ALWAYS_INPLACE}>:HPDEXC_FORCE_ALWAYS_INPLACE=1>
    $<$<BOOL:${HPDEXC_DEBUG_POLICY}>:HPDEXC_DEBUG_POLICY=1>
)

# 链接 Highway
target_link_libraries(kernel PRIVATE hwy)
target_link_libraries(kernel PRIVATE hwy_contrib)

# macOS 上的 OpenMP 兼容处理（尽量单处设置）
if(APPLE)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(kernel PRIVATE OpenMP::OpenMP_CXX)
  else()
    message(STATUS "Using Homebrew libomp")
    target_include_directories(kernel PRIVATE /opt/homebrew/opt/libomp/include)
    target_link_directories(kernel PRIVATE /opt/homebrew/opt/libomp/lib)
    target_link_libraries(kernel PRIVATE omp)
    target_compile_options(kernel PRIVATE -Xpreprocessor -fopenmp)
  endif()
endif()

# ========= 测试模块 =========
# 如果需要测试模块，可以在这里添加
# pybind11_add_module(_c_test src/cpp/module_test.cpp)
# set_target_properties(_c_test PROPERTIES
#   LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/python/hpdex/backen
# )