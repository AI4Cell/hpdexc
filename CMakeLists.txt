cmake_minimum_required(VERSION 3.18)
project(hpdexc LANGUAGES CXX)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 生成 compile_commands.json (供 clangd)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 使用 Ninja 更快
if(NOT CMAKE_GENERATOR)
  set(CMAKE_GENERATOR "Ninja" CACHE INTERNAL "" FORCE)
endif()

# pybind11 & Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# OpenMP（macOS 需 libomp）
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
endif()

# 头文件目录（与你的结构对应）
include_directories(
  ${CMAKE_SOURCE_DIR}/src/cpp/include
)

# 源文件（根据你已有文件命名）
set(HPDEXC_SOURCES
  src/cpp/core/wilcoxon.cpp
  src/cpp/matrix/dense.cpp
  src/cpp/matrix/sparse.cpp
  src/cpp/common.cpp
  src/cpp/bindings/module.cpp
)

# 生成 Python 扩展模块：hpdexc (import hpdexc)
pybind11_add_module(hpdexc ${HPDEXC_SOURCES})

# 编译选项
target_compile_options(hpdexc PRIVATE -O3 -march=native)

# macOS 上的 OpenMP 兼容处理
if(OpenMP_CXX_FOUND)
  target_link_libraries(hpdexc PRIVATE OpenMP::OpenMP_CXX)
else()
  # 备用：AppleClang 手动链接 libomp（若 find_package 失败）
  if(APPLE)
    target_compile_options(hpdexc PRIVATE -Xpreprocessor -fopenmp)
    target_link_libraries(hpdexc PRIVATE omp)
  endif()
endif()
