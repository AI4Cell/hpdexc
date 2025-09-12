#pragma once

#include <cstddef>
#include <type_traits>
#include <cmath>
#include <memory>
#include <algorithm>


// ======================= SIMD / 平台支持 =======================
#if defined(__SSE2__)
  #include <emmintrin.h>
#endif


// ======================= 默认条件宏 =======================
// 启用hwy
#ifndef HPDEXC_HWY
#define HPDEXC_HWY 1
#endif

// 强制vqsort静态模式
#ifndef VQSORT_ONLY_STATIC
#define VQSORT_ONLY_STATIC 1
#endif

// 启用 VQSort
#ifndef HPDEXC_VQSORT
#define HPDEXC_VQSORT 1
#endif

// 启用 OpenMP 多线程支持
#ifndef HPDEXC_OMP
#define HPDEXC_OMP 1
#endif

// 启用 fp16
#ifndef HPDEXC_FP16
#define HPDEXC_FP16 0
#endif

// pybind11
#ifndef HPDEXC_PYBIND
#define HPDEXC_PYBIND 1
#endif

// 对齐精度
#ifndef HPDEXC_ALIGN_SIZE
#define HPDEXC_ALIGN_SIZE 64
#endif

// 预取距离
#ifndef HPDEXC_PREFETCH_DIST
#define HPDEXC_PREFETCH_DIST 32
#endif


// ======================= 头文件 ==========================
// hwy::
#if HPDEXC_HWY
  #include "hwy/highway.h"
  #if HPDEXC_VQSORT
    #if VQSORT_ONLY_STATIC
      #include "hwy/contrib/sort/vqsort-inl.h"
    #else
      #include "hwy/contrib/sort/vqsort.h"
    #endif
  #endif
#endif

// omp::
#if HPDEXC_OMP
  #include <omp.h>
#endif

// pybind::
#if HPDEXC_PYBIND
  #include <pybind11/pybind11.h>
  #include <pybind11/numpy.h>
  #include <pybind11/stl.h>
  namespace py = pybind11;
#endif


// ======================= 条件宏接口 =======================
#if VQSORT_ONLY_STATIC
  #define VQSORT(ptr, n, order) hwy::HWY_NAMESPACE::VQSortStatic((ptr), (n), (order))
#else
  #define VQSORT(ptr, n, order) hwy::VQSort((ptr), (n), (order))
#endif

// Sort 封装
#if HPDEXC_HWY && HPDEXC_VQSORT
#define SORT(ptr, n) VQSORT((ptr), (n), hwy::SortAscending{})
#define SORT_R(ptr, n) VQSORT((ptr), (n), hwy::SortDescending{})
#else
#define SORT(ptr, n) std::sort((ptr), (ptr) + (n))
#define SORT_R(ptr, n) std::sort((ptr), (ptr) + (n), std::greater<T>())
#endif

// 多线程支持
#if HPDEXC_OMP && defined(_OPENMP)
  #define PARALLEL_REGION(threads) \
    _Pragma("omp parallel num_threads(threads)")

  // 把 SCHEDULE 直接拼接成 schedule(SCHEDULE)
  #define PARALLEL_FOR(SCHEDULE, LOOP) \
    _Pragma(HPDEXC_STRINGIFY(omp for schedule(SCHEDULE))) \
    LOOP

  #define THREAD_ID() omp_get_thread_num()
  #define MAX_THREADS() omp_get_max_threads()

  // 辅助宏：把参数转成字符串
  #define HPDEXC_STRINGIFY(x) HPDEXC_STRINGIFY_IMPL(x)
  #define HPDEXC_STRINGIFY_IMPL(x) #x
#else
  #define PARALLEL_REGION(threads)
  #define PARALLEL_FOR(SCHEDULE, LOOP) LOOP
  #define THREAD_ID() 0
  #define MAX_THREADS() 1
#endif

// fp16 支持
#if HPDEXC_FP16
  #if defined(__FLT16_MANT_DIG__) || defined(__F16C__) || (defined(_MSC_VER) && defined(_M_FP16))
    using fp16 = _Float16;
  #else
    #define HPDEXC_FP16 0
  #endif
#endif


// ======================= 工具 =======================
// inline
#if defined(_MSC_VER)
  #define FORCE_INLINE __forceinline
  #define RESTRICT     __restrict
#else
  #define FORCE_INLINE inline __attribute__((always_inline))
  #define RESTRICT     __restrict__
#endif

#if defined(_MSC_VER)
  #define MALLOC_ALIGNED(ptr, size) \
    do { ptr = static_cast<void*>(_aligned_malloc((size), HPDEXC_ALIGN_SIZE)); } while(0)
  #define FREE_ALIGNED(ptr) \
    do { _aligned_free((ptr)); } while(0)
#else
  #define MALLOC_ALIGNED(ptr, size) \
    do { if (posix_memalign(&(ptr), HPDEXC_ALIGN_SIZE, (size)) != 0) ptr = nullptr; } while(0)
  #define FREE_ALIGNED(ptr) \
    do { free((ptr)); } while(0)
#endif

// assume_aligned
#if defined(__clang__) || defined(__GNUC__)
  #define ASSUME_ALIGNED(p, N) reinterpret_cast<decltype(p)>(__builtin_assume_aligned((p), (N)))
#else
  #define ASSUME_ALIGNED(p, N) (p)
#endif

// 分支预测
#if defined(__clang__) || defined(__GNUC__)
  #define LIKELY(x)   (__builtin_expect(!!(x), 1))
  #define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
  #define LIKELY(x)   (x)
  #define UNLIKELY(x) (x)
#endif

// 预取
#if defined(__clang__) || defined(__GNUC__)
  #define PREFETCH_R(p,loc) __builtin_prefetch((p), 0, (loc))
  #define PREFETCH_W(p,loc) __builtin_prefetch((p), 1, (loc))
#else
  #define PREFETCH_R(p,loc) ((void)0)
  #define PREFETCH_W(p,loc) ((void)0)
#endif

// 导出
#if defined(_MSC_VER)
  #define HPDEXC_EXPORT __declspec(dllexport)
#else
  #define HPDEXC_EXPORT __attribute__((visibility("default")))
#endif


// ======================= 分派工具 =======================
#if HPDEXC_FP16
#define DTYPE_DISPATCH(DO) \
    DO(double)   \
    DO(float)    \
    DO(fp16)     \
    DO(int64_t)  \
    DO(int32_t)  \
    DO(uint64_t) \
    DO(uint32_t) \
    DO(int16_t)  \
    DO(uint16_t)
#else
#define DTYPE_DISPATCH(DO) \
    DO(double)   \
    DO(float)    \
    DO(int64_t)  \
    DO(int32_t)  \
    DO(uint64_t) \
    DO(uint32_t) \
    DO(int16_t)  \
    DO(uint16_t)
#endif


// ======================= AlignedVector =======================
namespace hpdexc {

template<class T, size_t ALIGN = HPDEXC_ALIGN_SIZE>
class AlignedVector {
public:
  using value_type = T;
  static_assert((ALIGN & (ALIGN - 1)) == 0, "ALIGN must be power of two");

  // ---------- 构造/析构/赋值 ----------
  FORCE_INLINE AlignedVector() noexcept : ptr_(nullptr), size_(0), cap_(0) {}

  explicit AlignedVector(size_t n) : ptr_(nullptr), size_(0), cap_(0) { resize(n); }
  AlignedVector(size_t n, const T& val) : ptr_(nullptr), size_(0), cap_(0) { resize(n, val); }

  AlignedVector(const AlignedVector& o) : ptr_(nullptr), size_(0), cap_(0) {
      reserve(o.size_);
      uninitialized_copy(o.ptr_, o.size_);
      size_ = o.size_;
  }
  AlignedVector& operator=(const AlignedVector& o) {
      if (LIKELY(this != &o)) {
          clear();
          reserve(o.size_);
          uninitialized_copy(o.ptr_, o.size_);
          size_ = o.size_;
      }
      return *this;
  }

  AlignedVector(AlignedVector&& o) noexcept
      : ptr_(o.ptr_), size_(o.size_), cap_(o.cap_) { o.ptr_ = nullptr; o.size_ = 0; o.cap_ = 0; }

  AlignedVector& operator=(AlignedVector&& o) noexcept {
      if (LIKELY(this != &o)) {
          destroy_and_free();
          ptr_ = o.ptr_; size_ = o.size_; cap_ = o.cap_;
          o.ptr_ = nullptr; o.size_ = 0; o.cap_ = 0;
      }
      return *this;
  }

  ~AlignedVector() { destroy_and_free(); }

  // ---------- 基本访问 ----------
  FORCE_INLINE T*       data()        noexcept { return ptr_; }
  FORCE_INLINE const T* data()  const noexcept { return ptr_; }

  // 编译器假定对齐，有利于自动向量化/对齐加载
  FORCE_INLINE T*       aligned_data()        noexcept { return ASSUME_ALIGNED(ptr_, ALIGN); }
  FORCE_INLINE const T* aligned_data()  const noexcept { return ASSUME_ALIGNED(ptr_, ALIGN); }

  FORCE_INLINE size_t size()     const noexcept { return size_; }
  FORCE_INLINE size_t capacity() const noexcept { return cap_;  }
  FORCE_INLINE bool   empty()    const noexcept { return size_ == 0; }

  FORCE_INLINE T&       operator[](size_t i)       noexcept { return ptr_[i]; }
  FORCE_INLINE const T& operator[](size_t i) const noexcept { return ptr_[i]; }

  FORCE_INLINE T*       begin()        noexcept { return ptr_; }
  FORCE_INLINE T*       end()          noexcept { return ptr_ + size_; }
  FORCE_INLINE const T* begin()  const noexcept { return ptr_; }
  FORCE_INLINE const T* end()    const noexcept { return ptr_ + size_; }

  // ---------- 容量控制 ----------
  // 至少预留 n 容量，保留内容
  FORCE_INLINE void reserve(size_t n) {
      if (LIKELY(n <= cap_)) return;
      reallocate(n);
  }

  // 预留至少 n（带增长策略），避免频繁重分配
  FORCE_INLINE void reserve_at_least(size_t n) {
      if (LIKELY(n <= cap_)) return;
      size_t new_cap = cap_ ? std::max(n, cap_ * 2) : std::max<size_t>(n, 8);
      reallocate(new_cap);
  }

  // 精确收缩到 size_
  void shrink_to_fit() {
      if (LIKELY(size_ == cap_)) return;
      if (UNLIKELY(size_ == 0)) { destroy_and_free(); return; }
      reallocate(size_);
  }

  // ---------- 改变大小 ----------
  // 默认构造/销毁（非平凡类型会逐个调用构造/析构）
  void resize(size_t n) {
      reserve(n);
      if (n > size_) default_construct(size_, n);
      else           destroy_range(n, size_);
      size_ = n;
  }

  // 填充值扩容
  void resize(size_t n, const T& val) {
      reserve(n);
      if (n > size_) uninitialized_fill(size_, n, val);
      else           destroy_range(n, size_);
      size_ = n;
  }

  // 高性能：不构造对象，仅改变 size（仅限可平凡默认构造的类型）
  template<class U=T, std::enable_if_t<std::is_trivially_default_constructible_v<U>, int> = 0>
  FORCE_INLINE void resize_uninitialized(size_t n) {
      reserve(n);
      size_ = n; // 不触发构造；调用者需保证后续会写满
  }

  // 清空内容（保留容量）
  FORCE_INLINE void clear() noexcept {
      destroy_range(0, size_);
      size_ = 0;
  }

  // ---------- 修改 ----------
  FORCE_INLINE void push_back(const T& v) {
      if (UNLIKELY(size_ == cap_)) reserve_at_least(size_ + 1);
      new (ptr_ + size_) T(v);
      ++size_;
  }
  FORCE_INLINE void push_back(T&& v) {
      if (UNLIKELY(size_ == cap_)) reserve_at_least(size_ + 1);
      new (ptr_ + size_) T(std::move(v));
      ++size_;
  }

  // 不检查容量的 push（配合 writable_span 使用，最激进）
  FORCE_INLINE void push_back_unchecked(const T& v) {
      new (ptr_ + size_) T(v);
      ++size_;
  }
  FORCE_INLINE void push_back_unchecked(T&& v) {
      new (ptr_ + size_) T(std::move(v));
      ++size_;
  }

  // 获取可写入区（不改 size），适合“复写不 fill”的场景
  // 返回保证对齐的指针
  FORCE_INLINE T* writable_span(size_t n_required) {
      reserve_at_least(n_required);
      return aligned_data();
  }

  // 提交写入长度（与 writable_span 配合使用）
  FORCE_INLINE void commit_size(size_t n) {
      // 调用者需保证 n <= capacity
      size_ = n;
  }

  // 对 trivially copyable 数值类型：快速置零
  template<class U=T, std::enable_if_t<std::is_trivially_copyable_v<U>, int> = 0>
  FORCE_INLINE void memset_zero() {
      if (LIKELY(ptr_ && size_)) std::memset(ptr_, 0, size_ * sizeof(T));
  }

private:
  T*     ptr_;
  size_t size_;
  size_t cap_;

  // ---------- 内部：分配/释放 ----------
  FORCE_INLINE void* aligned_malloc_bytes(size_t bytes) {
      void* raw = nullptr;
      MALLOC_ALIGNED(raw, bytes); // nullptr on failure (POSIX) / throws later
      return raw;
  }
  FORCE_INLINE static void aligned_free(void* p) {
      if (LIKELY(p)) FREE_ALIGNED(p);
  }

  void allocate(size_t n) {
      if (UNLIKELY(n == 0)) { ptr_ = nullptr; cap_ = 0; return; }
      void* raw = aligned_malloc_bytes(n * sizeof(T));
      if (UNLIKELY(!raw)) throw std::bad_alloc();
      ptr_ = static_cast<T*>(raw);
      cap_ = n;
  }

  // 带内容迁移的重分配：POD 走 memcpy；否则 move
  void reallocate(size_t new_cap) {
      T* old = ptr_;
      const size_t old_size = size_;

      allocate(new_cap); // 更新 ptr_/cap_

      if constexpr (std::is_trivially_copyable_v<T>) {
          // 对齐提示 + 预取
          T* dst = aligned_data();
          const T* src = old;
          if (LIKELY(old && old_size)) {
              PREFETCH_R(src, 3);
              std::memcpy(dst, src, old_size * sizeof(T));
          }
      } else {
          // 非平凡类型：move 构造
          for (size_t i = 0; i < old_size; ++i) {
              if (LIKELY((i + 64) < old_size)) PREFETCH_R(old + i + 64, 1);
              new (ptr_ + i) T(std::move_if_noexcept(old[i]));
          }
          if (LIKELY(old)) {
              if constexpr (!std::is_trivially_destructible_v<T>) {
                  for (size_t i = 0; i < old_size; ++i) old[i].~T();
              }
          }
      }

      if (LIKELY(old)) aligned_free(old);
      size_ = old_size;
  }

  FORCE_INLINE void destroy_and_free() {
      if (LIKELY(ptr_)) {
          destroy_range(0, size_);
          aligned_free(ptr_);
          ptr_ = nullptr; size_ = 0; cap_ = 0;
      }
  }

  // ---------- 内部：构造/销毁/拷贝 ----------
  FORCE_INLINE void destroy_range(size_t from, size_t to) {
      if constexpr (!std::is_trivially_destructible_v<T>) {
          for (size_t i = from; i < to; ++i) ptr_[i].~T();
      }
  }

  FORCE_INLINE void default_construct(size_t from, size_t to) {
      if constexpr (std::is_trivially_default_constructible_v<T>) {
          // 无操作：跳过构造（更快）
      } else {
          for (size_t i = from; i < to; ++i) new (ptr_ + i) T();
      }
  }

  FORCE_INLINE void uninitialized_fill(size_t from, size_t to, const T& val) {
      for (size_t i = from; i < to; ++i) new (ptr_ + i) T(val);
  }

  FORCE_INLINE void uninitialized_copy(const T* src, size_t n) {
      if constexpr (std::is_trivially_copyable_v<T>) {
          if (LIKELY(n)) {
              PREFETCH_R(src, 3);
              std::memcpy(ptr_, src, n * sizeof(T));
          }
      } else {
          for (size_t i = 0; i < n; ++i) {
              if (LIKELY((i + 64) < n)) PREFETCH_R(src + i + 64, 1);
              new (ptr_ + i) T(src[i]);
          }
      }
  }
};

} // namespace hpdexc

// ======================= 数学工具 =======================
namespace hpdexc {

template<typename T>
inline bool is_nan(const T&) {
    if constexpr (std::is_floating_point_v<T>) return false;
    else return false;
}
inline bool is_nan(const float& x)  { return std::isnan(x); }
inline bool is_nan(const double& x) { return std::isnan(x); }
#if HPDEXC_HAS_F16
inline bool is_nan(const _Float16& x) { return std::isnan((double)x); }
#endif

template<typename T>
inline bool is_inf(const T&) {
    if constexpr (std::is_floating_point_v<T>) return false;
    else return false;
}
inline bool is_inf(const float& x)  { return std::isinf(x); }
inline bool is_inf(const double& x) { return std::isinf(x); }
#if HPDEXC_HAS_F16
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
    return 0.5 * std::erfc(z * INV_SQRT2);
}

} // namespace hpdexc
