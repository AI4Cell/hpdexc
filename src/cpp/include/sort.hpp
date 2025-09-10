#pragma once

#include "common.hpp"
#include <vector>
#include <string>
// #include <hwy/contrib/sort/vqsort.h>  // 暂时注释掉

// 前向声明
namespace pybind11 {
    class module_;
}

namespace hpdexc {

// 主要排序函数声明
template<typename T>
HPDEXC_FORCE_INLINE void sort(T* data, size_t n, const std::string& kind, bool reverse);

// 原地排序（类似ndarray.sort()）
template<typename T>
HPDEXC_FORCE_INLINE void sort_inplace(T* data, size_t n, const std::string& kind, bool reverse);

// 返回排序后副本的函数
template<typename T>
HPDEXC_FORCE_INLINE std::vector<T> sort_copy(const T* data, size_t n, const std::string& kind, bool reverse);

// 多线程排序
template<typename T>
HPDEXC_FORCE_INLINE void parallel_sort(T* data, size_t n, const std::string& kind, bool reverse, int threads);

// 与numpy兼容的排序函数
template<typename T>
HPDEXC_FORCE_INLINE void numpy_compatible_sort(T* data, size_t n, const std::string& kind, bool reverse);

// 内部实现函数（供高级用户使用）
template<typename T>
HPDEXC_FORCE_INLINE void quicksort_impl(T* data, size_t n);

template<typename T>
HPDEXC_FORCE_INLINE void heapsort_impl(T* data, size_t n);

template<typename T>
HPDEXC_FORCE_INLINE void insertion_sort(T* data, size_t n);

template<typename T>
HPDEXC_FORCE_INLINE void radixsort_impl(T* data, size_t n);

// VQSort 实现（Google Highway 的高性能排序）
template<typename T>
HPDEXC_FORCE_INLINE void vqsort_impl(T* data, size_t n);

template<typename T>
HPDEXC_FORCE_INLINE void adaptive_sort_impl(T* data, size_t n, const std::string& kind, bool reverse);

// Python绑定函数声明
void bind_sort_functions(pybind11::module_& m);

} // namespace hpdexc