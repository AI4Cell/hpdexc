cmake_minimum_required(VERSION 3.18)
project(hpdexc LANGUAGES CXX)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 确保C++17特性可用
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # 确保GNU C++标准库特性
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
    # 设置C++17特性
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif()

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

# ========= Progress Bar Library =========
# 添加 indicators 进度条库
FetchContent_Declare(
  indicators
  GIT_REPOSITORY https://github.com/p-ranav/indicators.git
  GIT_TAG v2.3
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(indicators)


set(HPDEXC_SOURCES
  src/cpp/mannwhitney.cpp
  src/cpp/module.cpp
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

if(CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# 编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(kernel PRIVATE -g -O0 -DDEBUG -fno-omit-frame-pointer)
    # 为LLDB优化调试信息（检查编译器是否支持-fstandalone-debug）
    target_compile_options(kernel PRIVATE -gdwarf-4)
    # 检查GCC版本是否支持-fstandalone-debug（需要GCC 12+）
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-fstandalone-debug" COMPILER_SUPPORTS_STANDALONE_DEBUG)
    if(COMPILER_SUPPORTS_STANDALONE_DEBUG)
        target_compile_options(kernel PRIVATE -fstandalone-debug)
    endif()
    message(STATUS "Building in Debug mode for LLDB/GDB")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(kernel PRIVATE -g -O2 -DNDEBUG)
    message(STATUS "Building in RelWithDebInfo mode")
else()
    target_compile_options(kernel PRIVATE -O3 -march=native -DNDEBUG)
    message(STATUS "Building in Release mode")
endif()

# OpenMP多线程支持（可通过编译选项控制）
option(HPDEXC_ENABLE_MULTITHREAD "Enable OpenMP multithreading support" ON)
if(HPDEXC_ENABLE_MULTITHREAD)
    target_compile_definitions(kernel PRIVATE HPDEXC_ST=0)
    message(STATUS "Multithreading enabled (HPDEXC_ST=0)")
else()
    target_compile_definitions(kernel PRIVATE HPDEXC_ST=1)
    message(STATUS "Multithreading disabled (HPDEXC_ST=1)")
endif()

# 安装扩展到 Python 包目录 `hpdexc/`
install(TARGETS kernel DESTINATION ${CMAKE_SOURCE_DIR}/src/python/hpdex/backend)

# 条件编译选项
option(HPDEXC_ST "Force disable multithreading" OFF)

target_compile_definitions(kernel
  PRIVATE
      $<$<BOOL:${HPDEXC_ST}>:HPDEXC_ST=1>
)

# 禁用Float16支持以避免符号链接问题
target_compile_definitions(kernel PRIVATE HPDEXC_HAS_F16=0)

# 链接 Highway
target_link_libraries(kernel PRIVATE hwy)
target_link_libraries(kernel PRIVATE hwy_contrib)

# 链接 indicators 进度条库
target_link_libraries(kernel PRIVATE indicators::indicators)

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

# ========= 调试配置 =========
# 为调试创建符号链接，方便LLDB找到符号
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_custom_target(debug_setup ALL
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_SOURCE_DIR}/build/compile_commands.json
        ${CMAKE_SOURCE_DIR}/compile_commands.json
        COMMENT "Creating compile_commands.json symlink for debugging"
    )
    add_dependencies(debug_setup kernel)
endif()
