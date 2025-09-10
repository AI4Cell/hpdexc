#pragma once

#include <cstddef>
#include <type_traits>
#include <cmath>

/// @brief intel SSE2 支持
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

/// @brief OpenMP 支持
#ifndef HPDEXC_ST
#define HPDEXC_ST 0  // 启用多线程支持
#endif

#if defined(_OPENMP) && !HPDEXC_ST
  #include <omp.h>
  // 若头文件未定义保护宏，这里提供后备。统一用它屏蔽/启用所有并行区。
  #ifndef HPDEXC_OMP_GUARD
  #define HPDEXC_OMP_GUARD(x) (x)
  #endif
#else
  inline int  omp_get_max_threads() { return 1; }
  inline int  omp_get_thread_num()  { return 0; }
  inline int  omp_in_parallel()     { return 0; }
  #ifndef HPDEXC_OMP_GUARD
  #define HPDEXC_OMP_GUARD(x) (0)
  #endif
#endif

//================= 优化宏 =================
#if defined(_MSC_VER)
  #define HPDEXC_FORCE_INLINE __forceinline
  #define HPDEXC_RESTRICT     __restrict
#else
  #define HPDEXC_FORCE_INLINE inline __attribute__((always_inline))
  #define HPDEXC_RESTRICT     __restrict__
#endif

#if defined(__clang__) || defined(__GNUC__)
  #define HPDEXC_LIKELY(x)   (__builtin_expect(!!(x), 1))
  #define HPDEXC_UNLIKELY(x) (__builtin_expect(!!(x), 0))
  #define HPDEXC_PREFETCH_R(p,loc) __builtin_prefetch((p), 0, (loc))
  #define HPDEXC_PREFETCH_W(p,loc) __builtin_prefetch((p), 1, (loc))
  template<class T>
  HPDEXC_FORCE_INLINE T* hpdexc_assume_aligned(T* p, std::size_t) {
    return reinterpret_cast<T*>(__builtin_assume_aligned(p, 64));
  }
#else
  #define HPDEXC_LIKELY(x)   (x)
  #define HPDEXC_UNLIKELY(x) (x)
  #define HPDEXC_PREFETCH_R(p,loc) ((void)0)
  #define HPDEXC_PREFETCH_W(p,loc) ((void)0)
  template<class T>
  HPDEXC_FORCE_INLINE T* hpdexc_assume_aligned(T* p, std::size_t) { return p; }
#endif

/// @brief 导出宏
#ifndef HPDEXC_EXPORT
#  ifdef _WIN32
#    define HPDEXC_EXPORT __declspec(dllexport)
#  else
#    define HPDEXC_EXPORT __attribute__((visibility("default")))
#  endif
#endif

#if !defined(HPDEXC_HAS_F16)
  #if defined(__FLT16_MANT_DIG__) || defined(__F16C__) || (defined(_MSC_VER) && defined(_M_FP16))
    #define HPDEXC_HAS_F16 1
    using fp16 = _Float16;
  #else
    #define HPDEXC_HAS_F16 0
  #endif
#endif

#if HPDEXC_HAS_F16
#define HPDEXC_DTYPE_DISPATCH(DO) \
DO(double)\
DO(float)\
DO(fp16)\
DO(int64_t)\
DO(int32_t)\
DO(uint64_t)\
DO(uint32_t)\
DO(int16_t)\
DO(uint16_t)
#else
#define HPDEXC_DTYPE_DISPATCH(DO) \
DO(double)\
DO(float)\
DO(int64_t)\
DO(int32_t)\
DO(uint64_t)\
DO(uint32_t)\
DO(int16_t)\
DO(uint16_t)
#endif


namespace hpdexc {

constexpr size_t MannWhitneyProgressDefaultBufSize = 1024;
constexpr size_t MannWhitneyGCountDefaultBufSize = 1'000'000;
constexpr size_t MannWhitneyMaxStackGroups = 1024;
constexpr size_t MannWhitneyMaxStackNnz = 20'000;
constexpr size_t MannWhitneyMaxStackColCap = 20'000;
constexpr size_t MannWhitneyStackGroupLimit = 128;
constexpr size_t kSortThresholdsDefault_small_cut = 1u<<12;
constexpr size_t kSortThresholdsDefault_mid_cut = 1u<<20;  // 恢复原始阈值
constexpr size_t kSortThresholdsDefault_runs_cap = 1u<<12;
constexpr size_t kSortThresholdsDefault_run_len_hot = 256;
constexpr size_t SortTArMinN = 1u << 15;
constexpr size_t SortPrefetchDist = 64;


template<typename T>
inline bool is_nan(const T&) {
    if constexpr (std::is_floating_point_v<T>) return false; // 让重载分派到下面
    else return false;
}
inline bool is_nan(const float& x)  { return std::isnan(x); }
inline bool is_nan(const double& x) { return std::isnan(x); }
#if defined(__FLT16_MANT_DIG__) || defined(__F16C__) || (defined(_MSC_VER) && defined(_M_FP16))
inline bool is_nan(const _Float16& x) { return std::isnan((double)x); }
#endif

template<typename T>
inline bool is_inf(const T&) {
    if constexpr (std::is_floating_point_v<T>) return false; // 让重载分派到下面
    else return false;
}
inline bool is_inf(const float& x)  { return std::isinf(x); }
inline bool is_inf(const double& x) { return std::isinf(x); }
#if defined(__FLT16_MANT_DIG__) || defined(__F16C__) || (defined(_MSC_VER) && defined(_M_FP16))
inline bool is_inf(const _Float16& x) { return std::isinf((double)x); }
#endif

template<typename T>
static inline T cdf(T z) {
    constexpr T INV_SQRT2 = 0.707106781186547524400844362104849039;
    return 0.5 * std::erfc(-z * INV_SQRT2);
}

template<typename T>
static inline T sf(T z) {
    constexpr T INV_SQRT2 = 0.707106781186547524400844362104849039;
    return 0.5 * std::erfc( z * INV_SQRT2);
}

}