#pragma once
#include "view.hpp"
#include <vector>
#include <cstdint>
#include <cstddef>

namespace hpdexc {

/*
接口结构体
*/

/// @brief Mann-Whitney U 检验的运行选项
template<typename T>
struct MannWhitneyOption {
    bool ref_sorted;   ///< 是否保证 ref 组已排序（若 true 可跳过排序阶段）
    bool tar_sorted;   ///< 是否保证 tar 组已排序

    bool tie_correction;   ///< 是否应用 ties 修正项（处理重复值时修正方差）
    bool use_continuity;   ///< 是否做连续性校正（在正态近似时 ±0.5 调整）

    bool use_sparse_value;    ///< 是否使用稀疏值
    enum SparseValueMinmax {
        none = 0,    ///< 不确保稀疏值是最大或者最小值
        min = 1,     ///< 确保稀疏值是最小值
        max = 2      ///< 确保稀疏值是最大值
    } is_spare_minmax;    ///< 是否确保稀疏值是最大或者最小值
    T sparse_value;    ///< 特殊"稀疏值"：与该值相等的观测将被忽略

    /// @brief 备择假设类型
    enum Alternative {
        less      = 0,  ///< 单尾检验，假设 ref < tar
        greater   = 1,  ///< 单尾检验，假设 ref > tar
        two_sided = 2  ///< 双尾检验，假设 ref 与 tar 有差异
    } alternative;

    /// @brief 正态近似
    enum Method {
        autometic  = 0, ///< 与 scipy 决策方法一致
        exact      = 1, ///< 精确方法（此选项忽略 ties 修正项）
        asymptotic = 2, ///< 正态分布渐近近似
    } method;


    /// @brief 直方图参数
    struct HistogramOption {
        uint8_t  use_histogram;
        uint64_t max_bins         = 1'000'000;
        uint64_t mem_budget_bytes = 128ull << 20; ///< 128 MB 适应 L3 缓存
    } histogram;
};


/// @brief 
struct MannWhitneyResult {
    // 持有内存，避免悬垂
    std::vector<double> U_buf;
    std::vector<double> P_buf;

    // 暴露视图（指向上面的缓冲）
    DenseViewMut U;
    DenseViewMut P;

    MannWhitneyResult() : U(nullptr, 0, 0, 0), P(nullptr, 0, 0, 0) {}

    void set(std::size_t rows, std::size_t cols){
        U_buf.assign(rows * cols, 0.0);
        P_buf.assign(rows * cols, 1.0);
        // 视图指向内部缓冲；stride 取列数（行主序）
        U = DenseViewMut(U_buf.data(), rows, cols, cols);
        P = DenseViewMut(P_buf.data(), rows, cols, cols);
    }
};

template<typename T>
MannWhitneyResult mannwhitney(const CscView& A, const MannWhitneyOption<T>& opt,
                              const DenseView& group_id,
                              const std::size_t n_targets, int threads = 1, void* progress_ptr = nullptr);

}