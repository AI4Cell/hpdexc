#pragma once
#include <tuple>
#include <cstddef>
#include "tensor.hpp"

namespace hpdexc {

// 返回类型: (U, P)，均为 [cols, n_targets]，行主序
using MannWhitneyResult = std::tuple<tensor::Ndarray<double>, tensor::Ndarray<double>>;

template<class T>
struct MannWhitneyuOption {
    bool ref_sorted;
    bool tar_sorted;
    bool tie_correction;
    bool use_continuity;

    // 稀疏值类型：是否当作极小/极大/中间
    enum SparseValueMinmax {
        none = 0,
        strict_minor = 1,
        strict_major = 2,
        medium = 3,
    } sparse_type;
    T sparse_value;

    // 假设检验的方向
    enum Alternative { less = 0, greater = 1, two_sided = 2 } alternative;

    // 计算方法
    enum Method { automatic = 0, exact = 1, asymptotic = 2 } method;
};

/// @brief Mann-Whitney U test for multiple targets vs reference
/// @throws std::invalid_argument if group_id length != A.rows()
template<class T, class Idx>
auto mannwhitneyu(
    const tensor::Csc<T, Idx>& A,
    const MannWhitneyuOption<T>& opt,
    const tensor::Vector<int32_t>& group_id,
    std::size_t n_targets,
    int threads = -1,              // -1: use all available threads
    std::size_t* progress_ptr = nullptr // optional, must be >= threads length
) -> MannWhitneyResult;


/// @brief Compute per-group mean values across columns
/// @throws std::invalid_argument if group_id length != A.rows()
template<class T, class Idx>
auto group_mean(
    const tensor::Csc<T,Idx>& A,
    const tensor::Vector<int32_t>& group_id,
    std::size_t n_groups,
    int threads = -1,
    size_t* progress_ptr = nullptr
) -> tensor::Ndarray<double>;

} // namespace hpdexc
