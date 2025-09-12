#include "mannwhitneyu.hpp"
#include "common.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <limits>


namespace hpdexc {

// tools
template<class T>
static FORCE_INLINE bool 
is_valid_value(const T& v) {
    return !is_nan(v) && !is_inf(v);
}

template<class T>
static FORCE_INLINE typename MannWhitneyuOption<T>::Method 
choose_method(size_t n1, size_t n2, bool has_tie) {
    return (n1 > 8 && n2 > 8) || has_tie
        ? MannWhitneyuOption<T>::Method::asymptotic
        : MannWhitneyuOption<T>::Method::exact;
}

// ====== 正态近似 p 值（SciPy 对齐） ======
static FORCE_INLINE double normal_sf(double z) {
    // 上尾概率：P(Z >= z)
    return 0.5 * std::erfc(z / std::sqrt(2.0));
}

template<class T>
static FORCE_INLINE double
p_asymptotic_two_sided(
    double U1,
    size_t n1,
    size_t n2,
    double tie_sum,
    bool use_continuity
) {
    const double N  = static_cast<double>(n1 + n2);
    const double mu = 0.5 * static_cast<double>(n1) * static_cast<double>(n2);
    const double denom = N * (N - 1.0);
    const double base = static_cast<double>(n1) * static_cast<double>(n2) / 12.0;
    const double var = (denom > 0.0)
        ? base * (N + 1.0 - tie_sum / denom)
        : static_cast<double>(n1) * static_cast<double>(n2) * (N + 1.0) / 12.0;
    if (var <= 0.0) return 1.0;

    const double sd = std::sqrt(var);
    const double cc = use_continuity ? 0.5 : 0.0;
    const double z  = (std::abs(U1 - mu) - cc) / sd;
    return 2.0 * normal_sf(z);
}

template<class T>
static FORCE_INLINE double
p_asymptotic_greater(
    double U1,
    size_t n1,
    size_t n2,
    double tie_sum,
    bool use_continuity
) {
    const double N  = static_cast<double>(n1 + n2);
    const double mu = 0.5 * static_cast<double>(n1) * static_cast<double>(n2);
    const double denom = N * (N - 1.0);
    const double base = static_cast<double>(n1) * static_cast<double>(n2) / 12.0;
    const double var = (denom > 0.0)
        ? base * (N + 1.0 - tie_sum / denom)
        : static_cast<double>(n1) * static_cast<double>(n2) * (N + 1.0) / 12.0;
    if (var <= 0.0) return 1.0;

    const double sd = std::sqrt(var);
    const double cc = use_continuity ? 0.5 : 0.0;
    const double z  = (U1 - mu - cc) / sd;
    return normal_sf(z);
}

template<class T>
static FORCE_INLINE double
p_asymptotic_less(
    double U1,
    size_t n1,
    size_t n2,
    double tie_sum,
    bool use_continuity
) {
    const double N  = static_cast<double>(n1 + n2);
    const double mu = 0.5 * static_cast<double>(n1) * static_cast<double>(n2);
    const double denom = N * (N - 1.0);
    const double base = static_cast<double>(n1) * static_cast<double>(n2) / 12.0;
    const double var = (denom > 0.0)
        ? base * (N + 1.0 - tie_sum / denom)
        : static_cast<double>(n1) * static_cast<double>(n2) * (N + 1.0) / 12.0;
    if (var <= 0.0) return 1.0;

    const double sd = std::sqrt(var);
    const double cc = use_continuity ? 0.5 : 0.0;
    const double z  = (U1 - mu + cc) / sd;
    // 左尾：P(U1 <= u) = Φ(z) = 1 - SF(z)
    return 1.0 - normal_sf(z);
}


// exact p calculation (no ties), SciPy-aligned SF; ultra-optimized
template<class T>
static FORCE_INLINE double
p_exact(double U, size_t n1, size_t n2) {
    using U64 = unsigned long long;

    const size_t Umax = n1 * n2;
    if UNLIKELY(Umax == 0) return 1.0;

    // clamp & floor like SciPy
    const double U_clip = std::max(0.0, std::min(static_cast<double>(Umax), U));
    const size_t u_stat = static_cast<size_t>(std::floor(U_clip));

    // DP buffers: only write 0..up each iteration; avoid O(SZ) clears
    const size_t SZ = Umax + 1;
    hpdexc::AlignedVector<U64, 64> dp(SZ);   // uninitialized payload ok:我们只会读到上界以内
    hpdexc::AlignedVector<U64, 64> ndp(SZ);

    U64* RESTRICT dp_cur = dp.aligned_data();
    U64* RESTRICT dp_nxt = ndp.aligned_data();

    // init: dp_cur[0]=1; 其余不关心（不会被读到）
    dp_cur[0] = 1ULL;

    // 主 DP：窗口法
    for (size_t i = 1; i <= n1; ++i) {
        const size_t prev_up = (i - 1) * n2;  // 上一层可达最大 U
        const size_t up      =  i      * n2;  // 本层可达最大 U

        PREFETCH_W(dp_nxt + HPDEXC_PREFETCH_DIST, 1);

        U64 win = 0ULL;

        // 第一段：u ∈ [0, min(prev_up, up)]，需要累加 dp_cur[u]
        size_t u = 0;
        const size_t bound1 = prev_up < up ? prev_up : up;
        for (; u <= bound1; ++u) {
            PREFETCH_R(dp_cur + u + HPDEXC_PREFETCH_DIST, 1);
            win += dp_cur[u];
            if (u >= n2 + 1) {
                // (u - (n2+1)) ≤ prev_up - 1，必在已写范围内
                win -= dp_cur[u - (n2 + 1)];
            }
            dp_nxt[u] = win;
        }

        // 第二段：u ∈ [prev_up+1, up]，加数项为 0，仅做滑窗减法
        for (; u <= up; ++u) {
            if (u >= n2 + 1) win -= dp_cur[u - (n2 + 1)];
            dp_nxt[u] = win;
        }

        // 交换缓冲：下一轮读 dp_cur，写 dp_nxt
        U64* tmp = dp_cur; dp_cur = dp_nxt; dp_nxt = tmp;
    }

    // 归一化 (double)，与 SciPy 一致
    // 这里 dp_cur 已包含 i==n1 的最终分布，0..Umax 都已写入
    double total = 0.0;
    {
        // 顺序累加 + 轻预取
        for (size_t u = 0; u <= Umax; ++u) {
            PREFETCH_R(dp_cur + u + HPDEXC_PREFETCH_DIST, 1);
            total += static_cast<double>(dp_cur[u]);
        }
    }

    // SciPy 的 SF 对称优化
    const size_t kc    = Umax - u_stat;
    const size_t small = (u_stat < kc ? u_stat : kc);

    double cdf_small = 0.0;
    for (size_t u = 0; u <= small; ++u) {
        PREFETCH_R(dp_cur + u + HPDEXC_PREFETCH_DIST, 1);
        cdf_small += static_cast<double>(dp_cur[u]);
    }
    const double pmf_small = static_cast<double>(dp_cur[small]);

    // SF(k) = P(U >= k)，含等号
    double sf_ge;
    if (u_stat <= kc) {
        // 左半边：SF(u) = 1 - CDF(u) + PMF(u)
        sf_ge = 1.0 - cdf_small / total + pmf_small / total;
    } else {
        // 右半边：SF(u) = CDF(kc)
        sf_ge = cdf_small / total;
    }
    return sf_ge;
}


// sparse strict minor/major
static FORCE_INLINE void 
sparse_strict_minor_major_core(
    const bool          position_head,
    const size_t        ref_sp_cnt,
    const size_t        tar_sp_cnt,
    const size_t        N,
    double&             R1g,
    double&             tie_sum_g,
    bool&               has_tie_g,
    size_t&             grank_g
){
    const size_t sp_tie = ref_sp_cnt + tar_sp_cnt;
    const double tt = (sp_tie);
    const double mask = (sp_tie > 1) ? 1.0 : 0.0;
    tie_sum_g += ((tt*tt*tt - tt) * mask);
    has_tie_g  = has_tie_g || (sp_tie > 1);

    const double m   = position_head ? 1.0 : 0.0;
    const double nm  = 1.0 - m;
    const double ldN = (double)N;
    const double ldS = (double)sp_tie;
    const double start = m*1.0 + nm*(ldN - ldS + 1.0);
    const double end   = m*ldS  + nm*(ldN);
    const double avg   = (start + end) * 0.5;

    R1g    += (double)ref_sp_cnt * avg;
    grank_g += sp_tie;
}

// === tar 合并策略（两套）==================================================

struct TarMergeDense {
template<class T>
FORCE_INLINE void operator()(const T& val,
                                size_t count,
                                bool&  have_run,
                                T&     run_val,
                                size_t& run_len,
                                double& tie_sum_g,
                                bool&   has_tie_g,
                                size_t& grank_g) const {
    if UNLIKELY(!count) return;
    // ties 频繁：继续同值 run 的概率更高
    if LIKELY(have_run && !(run_val < val) && !(val < run_val)) {
    run_len += count;
    } else {
    if (run_len > 1) {
        const double tt = static_cast<double>(run_len);
        tie_sum_g += (tt*tt*tt - tt);
        has_tie_g  = true;
    }
    run_val  = val;
    run_len  = count;
    have_run = true;
    }
    grank_g += count;
}
};

struct TarMergeSparse {
template<class T>
FORCE_INLINE void operator()(const T& val,
                                size_t count,
                                bool&  have_run,
                                T&     run_val,
                                size_t& run_len,
                                double& tie_sum_g,
                                bool&   has_tie_g,
                                size_t& grank_g) const {
    if UNLIKELY(!count) return;
    // ties 罕见：相等命中概率低
    if UNLIKELY(have_run && !(run_val < val) && !(val < run_val)) {
    run_len += count;
    } else {
    if (run_len > 1) {
        const double tt = static_cast<double>(run_len);
        tie_sum_g += (tt*tt*tt - tt);
        has_tie_g  = true;
    }
    run_val  = val;
    run_len  = count;
    have_run = true;
    }
    grank_g += count;
}
};


// === 核心实现：模板化 merge 策略 =========================================
template<class T, class TarMerge>
static FORCE_INLINE void sparse_medium_core_impl(
    const T* RESTRICT col_val,
    const size_t* RESTRICT off,
    const size_t* RESTRICT gnnz,
    const size_t* RESTRICT sparse_value_cnt,
    const size_t  G,
    const T* RESTRICT refv,
    const size_t nref_exp,
    const T      sparse_value,
    size_t* RESTRICT tar_ptrs_local,
    size_t* RESTRICT grank,
    size_t* RESTRICT tar_eq,
    size_t* RESTRICT sp_left,
    bool*   RESTRICT have_run,
    T*      RESTRICT run_val,
    size_t* RESTRICT run_len,
    double* RESTRICT R1,
    double* RESTRICT tie_sum,
    bool*   RESTRICT has_tie,
    const TarMerge& merge
){
    // 对齐提示（只做一次）
    col_val = ASSUME_ALIGNED(col_val, HPDEXC_ALIGN_SIZE);
    off     = ASSUME_ALIGNED(off,     HPDEXC_ALIGN_SIZE);
    gnnz    = ASSUME_ALIGNED(gnnz,    HPDEXC_ALIGN_SIZE);
    refv    = ASSUME_ALIGNED(refv,    HPDEXC_ALIGN_SIZE);

    // 小工具：flush 当前组的 tar-only run
    auto flush_run = [&](size_t g){
        if (run_len[g] > 1) {
            const double tt = static_cast<double>(run_len[g]);
            tie_sum[g] += (tt*tt*tt - tt);
            has_tie[g]  = true;
        }
        run_len[g]  = 0;
        have_run[g] = false;
    };

    size_t i = 0;
    while (i < nref_exp || sp_left[0] > 0) {
        PREFETCH_R(refv + i + HPDEXC_PREFETCH_DIST, 1);

        // 取本轮参与归并的“参考值 vref”与 ref 的等值块大小 ref_tie
        T      vref;
        size_t ref_tie = 0;

        if (sp_left[0] > 0 && (i >= nref_exp || !(refv[i] < sparse_value))) {
            // 稀疏值 <= 当前参考值：把稀疏块和可能相等的 ref 一起并成一个 tie
            vref = sparse_value;
            size_t k_exp = 0;
            while (i + k_exp < nref_exp &&
                    !(refv[i + k_exp] < vref) && !(vref < refv[i + k_exp])) {
                ++k_exp;
            }
            ref_tie = sp_left[0] + k_exp;
            sp_left[0] = 0;
            i += k_exp;
        } else {
            // 采用当前 ref 值作为 vref，并收集它的等值块长度
            vref = (i < nref_exp) ? refv[i] : sparse_value;
            const size_t ref_start = i;
            while (i + 1 < nref_exp &&
                    !(refv[i + 1] < vref) && !(vref < refv[i + 1])) {
                ++i;
            }
            ref_tie = i - ref_start + 1;
            ++i;
        }

        // 归并每个 target 组中 < vref 的部分，以及与 vref 相等的显式/稀疏块
        for (size_t g = 1; g < G; ++g) {
            size_t& tp        = tar_ptrs_local[g];
            const size_t gend = off[g] + gnnz[g];

            PREFETCH_R(col_val + tp + HPDEXC_PREFETCH_DIST, 1);

            while (true) {
                const bool has_exp = (tp < gend) && (col_val[tp] < vref);
                const bool has_sp  = (sp_left[g] > 0) && (sparse_value < vref);
                if (!has_exp && !has_sp) break;

                if (has_exp && has_sp) {
                    const T ev = col_val[tp];
                    if (ev < sparse_value) {
                        size_t j = tp + 1;
                        while (j < gend &&
                                !(col_val[j] < ev) && !(ev < col_val[j])) ++j;
                        const size_t blk = j - tp;
                        merge(ev, blk, have_run[g], run_val[g], run_len[g],
                            tie_sum[g], has_tie[g], grank[g]);
                        tp = j;
                    } else if (sparse_value < ev) {
                        const size_t blk = sp_left[g];
                        merge(sparse_value, blk, have_run[g], run_val[g], run_len[g],
                            tie_sum[g], has_tie[g], grank[g]);
                        sp_left[g] = 0;
                    } else {
                        // ev == sparse_value：先并显式块，再并剩余稀疏块
                        size_t j = tp + 1;
                        while (j < gend &&
                                !(col_val[j] < sparse_value) && !(sparse_value < col_val[j])) ++j;
                        const size_t blk_exp = j - tp;
                        merge(sparse_value, blk_exp, have_run[g], run_val[g], run_len[g],
                            tie_sum[g], has_tie[g], grank[g]);
                        tp = j;
                        if (sp_left[g] > 0) {
                            const size_t blk_sp = sp_left[g];
                            merge(sparse_value, blk_sp, have_run[g], run_val[g], run_len[g],
                                tie_sum[g], has_tie[g], grank[g]);
                            sp_left[g] = 0;
                        }
                    }
                } else if (has_exp) {
                    const T ev = col_val[tp];
                    size_t j = tp + 1;
                    while (j < gend &&
                            !(col_val[j] < ev) && !(ev < col_val[j])) ++j;
                    const size_t blk = j - tp;
                    merge(ev, blk, have_run[g], run_val[g], run_len[g],
                        tie_sum[g], has_tie[g], grank[g]);
                    tp = j;
                } else {
                    // 只剩稀疏块
                    const size_t blk = sp_left[g];
                    merge(sparse_value, blk, have_run[g], run_val[g], run_len[g],
                        tie_sum[g], has_tie[g], grank[g]);
                    sp_left[g] = 0;
                }
            }
            // 清理本轮 tar-only run（确保下一轮 vref 的统计独立）
            flush_run(g);

            // 统计等于 vref 的 target 显式块
            size_t eq = 0;
            while (tp < gend && !(col_val[tp] < vref) && !(vref < col_val[tp])) {
                ++tp; ++eq;
            }
            // 若 target 还有等于 vref 的稀疏块，也并入
            if (sp_left[g] > 0 && !(sparse_value < vref) && !(vref < sparse_value)) {
                eq += sp_left[g];
                sp_left[g] = 0;
            }
            tar_eq[g] = eq;
        }

        // 合并更新：为每个目标组写入 ref_tie 个 vref 的平均秩到 R1
        for (size_t g = 1; g < G; ++g) {
            const double rrcur   = static_cast<double>(grank[g]);
            const size_t t       = ref_tie + tar_eq[g];
            const double rrnext  = rrcur + static_cast<double>(t);
            const double avg_rank= (rrcur + rrnext + 1.0) * 0.5;

            R1[g]    += static_cast<double>(ref_tie) * avg_rank;
            grank[g]  = static_cast<size_t>(rrnext);

            if UNLIKELY(t > 1) {
                const double tt = static_cast<double>(t);
                tie_sum[g] += (tt*tt*tt - tt);
                has_tie[g]  = true;
            }
            tar_eq[g] = 0;
        }
    }

    // 扫尾：处理每个组在末尾残留的 tar-only run（含稀疏剩余）
    for (size_t g = 1; g < G; ++g) {
        size_t& tp  = tar_ptrs_local[g];
        const size_t gend = off[g] + gnnz[g];

        have_run[g] = false;
        run_len[g]  = 0;

        while (tp < gend || sp_left[g] > 0) {
            bool has_exp = (tp < gend);
            bool has_sp  = (sp_left[g] > 0);

            T     cand;
            bool  take_sp = false;
            if (!has_exp) { cand = sparse_value; take_sp = true; }
            else if (!has_sp) { cand = col_val[tp]; }
            else {
                const T ev = col_val[tp];
                if (sparse_value < ev) { cand = sparse_value; take_sp = true; }
                else                   { cand = ev; }
            }

            if (take_sp) {
                merge(cand, sp_left[g], have_run[g], run_val[g], run_len[g],
                    tie_sum[g], has_tie[g], grank[g]);
                sp_left[g] = 0;
            } else {
                size_t j = tp + 1;
                while (j < gend && !(col_val[j] < cand) && !(cand < col_val[j])) ++j;
                const size_t blk = j - tp;
                merge(cand, blk, have_run[g], run_val[g], run_len[g],
                    tie_sum[g], has_tie[g], grank[g]);
                tp = j;
            }
        }
        // 最后 run 的结算
        if (run_len[g] > 1) {
            const double tt = static_cast<double>(run_len[g]);
            tie_sum[g] += (tt*tt*tt - tt);
            has_tie[g]  = true;
        }
        run_len[g]  = 0;
        have_run[g] = false;
    }
}


// === 两个无分支包装（外层按列/按组选择其中之一调用）======================
template<class T>
static FORCE_INLINE void sparse_medium_core_dense(
    const T* RESTRICT col_val,
    const size_t* RESTRICT off,
    const size_t* RESTRICT gnnz,
    const size_t* RESTRICT sparse_value_cnt,
    const size_t  G,
    const T* RESTRICT refv,
    const size_t nref_exp,
    const T      sparse_value,
    size_t* RESTRICT tar_ptrs_local,
    size_t* RESTRICT grank,
    size_t* RESTRICT tar_eq,
    size_t* RESTRICT sp_left,
    bool*   RESTRICT have_run,
    T*      RESTRICT run_val,
    size_t* RESTRICT run_len,
    double* RESTRICT R1,
    double* RESTRICT tie_sum,
    bool*   RESTRICT has_tie
){
    TarMergeDense merger{};
    sparse_medium_core_impl<T>(col_val, off, gnnz, sparse_value_cnt, G, refv, nref_exp,
                                sparse_value, tar_ptrs_local, grank, tar_eq, sp_left,
                                have_run, run_val, run_len, R1, tie_sum, has_tie, merger);
}

template<class T>
static FORCE_INLINE void sparse_medium_core_sparse(
    const T* RESTRICT col_val,
    const size_t* RESTRICT off,
    const size_t* RESTRICT gnnz,
    const size_t* RESTRICT sparse_value_cnt,
    const size_t  G,
    const T* RESTRICT refv,
    const size_t nref_exp,
    const T      sparse_value,
    size_t* RESTRICT tar_ptrs_local,
    size_t* RESTRICT grank,
    size_t* RESTRICT tar_eq,
    size_t* RESTRICT sp_left,
    bool*   RESTRICT have_run,
    T*      RESTRICT run_val,
    size_t* RESTRICT run_len,
    double* RESTRICT R1,
    double* RESTRICT tie_sum,
    bool*   RESTRICT has_tie
){
    TarMergeSparse merger{};
    sparse_medium_core_impl<T>(col_val, off, gnnz, sparse_value_cnt, G, refv, nref_exp,
                                sparse_value, tar_ptrs_local, grank, tar_eq, sp_left,
                                have_run, run_val, run_len, R1, tie_sum, has_tie, merger);
}


// 非稀疏分支：仅显式值参与；已初始化缓冲，计算 R1 与 tie_sum（不做 fill/init）
template<class T>
static FORCE_INLINE void sparse_none_core(
    const T*       RESTRICT col_val,         // 所有组拼在一起的有序值区间，按 off/gnnz 划分
    const size_t*  RESTRICT off,             // 每组起始偏移
    const size_t*  RESTRICT gnnz,            // 每组“显式”样本数（已排除非法+稀疏）
    const size_t            G,               // 组数（含参考组），约定 g=0 是参考
    const T*       RESTRICT refv,            // 参考组 col_val + off[0]
    const size_t            nrefcol,         // 参考组显式样本数 gnnz[0]
    size_t*        RESTRICT tar_ptrs_local,  // 初值设为 off[g]
    size_t*        RESTRICT grank,           // 初值 0
    size_t*        RESTRICT tie_cnt,         // 初值 0（记录 tar-only 连续相等长度-1）
    size_t*        RESTRICT tar_eq,          // 临时：本轮等于 vref 的 target 显式计数
    double*        RESTRICT R1,              // 初值 0
    double*        RESTRICT tie_sum,         // 初值 0
    bool*          RESTRICT has_tie          // 初值 false
){
    // 对齐提示（一次即可）
    col_val = ASSUME_ALIGNED(col_val, HPDEXC_ALIGN_SIZE);
    off     = ASSUME_ALIGNED(off,     HPDEXC_ALIGN_SIZE);
    gnnz    = ASSUME_ALIGNED(gnnz,    HPDEXC_ALIGN_SIZE);
    refv    = ASSUME_ALIGNED(refv,    HPDEXC_ALIGN_SIZE);

    // tar-only 并列 run 结算
    auto flush_tar_run = [&](size_t g){
        if (tie_cnt[g] > 0) {
            const double t = static_cast<double>(tie_cnt[g] + 1);
            tie_sum[g] += t*t*t - t;
            tie_cnt[g]  = 0;
            has_tie[g]  = true;
        }
    };

    size_t i = 0;
    while (i < nrefcol) {
        PREFETCH_R(refv + i + HPDEXC_PREFETCH_DIST, 1);

        const T vref = refv[i];

        // (1) 推进所有 < vref 的 target，期间把 tar-only 的并列块累加进 tie_sum
        for (size_t g = 1; g < G; ++g) {
            size_t& tp        = tar_ptrs_local[g];
            const size_t gbeg = off[g];
            const size_t gend = off[g] + gnnz[g];

            PREFETCH_R(col_val + tp + HPDEXC_PREFETCH_DIST, 1);

            while (tp < gend && col_val[tp] < vref) {
                if (tp > gbeg) {
                    // 等价判断：!(a<b) && !(b<a)
                    if (!(col_val[tp] < col_val[tp-1]) && !(col_val[tp-1] < col_val[tp])) {
                        ++tie_cnt[g];   // 连续相等长度-1
                    } else {
                        flush_tar_run(g);
                    }
                }
                ++tp;
                ++grank[g];  // 已有元素个数 = 当前 rank-1
            }
            flush_tar_run(g);

            // (2) 统计等于 vref 的 target 显式个数
            size_t eq = 0;
            while (tp < gend && !(col_val[tp] < vref) && !(vref < col_val[tp])) {
                ++tp; ++eq;
            }
            tar_eq[g] = eq;
        }

        // (3) ref 的等值块
        const size_t ref_start = i;
        while ((i + 1) < nrefcol && !(refv[i + 1] < vref) && !(vref < refv[i + 1])) ++i;
        const size_t ref_tie = i - ref_start + 1;

        // (4) 合并：为每个 target，把 ref_tie 个 vref 的平均秩叠加到 R1[g]
        for (size_t g = 1; g < G; ++g) {
            const double rrcur    = static_cast<double>(grank[g]);      // 当前 rank-1
            const size_t t        = ref_tie + tar_eq[g];                // 本轮 total tie 大小
            const double rrnext   = rrcur + static_cast<double>(t);
            const double avg_rank = (rrcur + rrnext + 1.0) * 0.5;

            R1[g]     += static_cast<double>(ref_tie) * avg_rank;
            grank[g]   = static_cast<size_t>(rrnext);

            if UNLIKELY(t > 1) {
                const double tt = static_cast<double>(t);
                tie_sum[g] += (tt*tt*tt - tt);
                has_tie[g]  = true;
            }
            tie_cnt[g] = 0;     // 清理 tar-only 暂存
            tar_eq[g]  = 0;     // 清理本轮等值计数
        }
        ++i;
    }

    // (5) 扫尾：target 剩余的 tar-only 并列块
    for (size_t g = 1; g < G; ++g) {
        size_t& tp  = tar_ptrs_local[g];
        const size_t gend = off[g] + gnnz[g];

        while (tp < gend) {
            size_t j = tp + 1;
            // 扫一个 run
            while (j < gend && !(col_val[j] < col_val[tp]) && !(col_val[tp] < col_val[j])) ++j;
            const size_t block_len = j - tp;
            if (block_len > 1) {
                const double tt = static_cast<double>(block_len);
                tie_sum[g] += (tt*tt*tt - tt);
                has_tie[g]  = true;
            }
            tp = j;
        }
    }
}

// 非稀疏分支（无 tie 修正）：只计算 R1（不含任何 fill/init）。
template<class T>
static FORCE_INLINE void sparse_core_without_tie(
    const T*      RESTRICT col_val,        // 拼好的所有组值，按 off/gnnz 划分
    const size_t* RESTRICT off,            // 每组起始偏移
    const size_t* RESTRICT gnnz,           // 每组显式样本数（已剔除非法/稀疏）
    const size_t           G,           // 组数（g=0 为参考组）
    const T*      RESTRICT refv,           // col_val + off[0]
    const size_t           nrefcol,     // gnnz[0]
    size_t*       RESTRICT tar_ptrs_local, // 初值 off[g]
    size_t*       RESTRICT grank,          // 初值 0
    size_t*       RESTRICT tar_eq,         // 临时缓冲（本轮等于 vref 的 target 个数）
    double*       RESTRICT R1              // 初值 0
){
    // 对齐假设（帮助自动向量化）
    col_val = ASSUME_ALIGNED(col_val, HPDEXC_ALIGN_SIZE);
    off     = ASSUME_ALIGNED(off,     HPDEXC_ALIGN_SIZE);
    gnnz    = ASSUME_ALIGNED(gnnz,    HPDEXC_ALIGN_SIZE);
    refv    = ASSUME_ALIGNED(refv,    HPDEXC_ALIGN_SIZE);

    size_t i = 0;
    while (i < nrefcol) {
        PREFETCH_R(refv + i + HPDEXC_PREFETCH_DIST, 1);

        const T vref = refv[i];

        // 1) 对每个 target：推进所有 < vref 的元素；统计 == vref 的个数
        for (size_t g = 1; g < G; ++g) {
            size_t& tp        = tar_ptrs_local[g];
            const size_t gend = off[g] + gnnz[g];

            PREFETCH_R(col_val + tp + HPDEXC_PREFETCH_DIST, 1);

            // < vref 的都推进，并把数量累加到 grank[g]
            while (tp < gend && col_val[tp] < vref) {
                ++tp;
                ++grank[g];
            }

            // 统计 == vref 的个数
            size_t eq = 0;
            while (tp < gend && !(col_val[tp] < vref) && !(vref < col_val[tp])) {
                ++tp;
                ++eq;
            }
            tar_eq[g] = eq; // 留给本轮合并使用
        }

        // 2) 参考组 ref 的等值块（连续的 vref）
        const size_t ref_start = i;
        while ((i + 1) < nrefcol && !(refv[i + 1] < vref) && !(vref < refv[i + 1])) ++i;
        const size_t ref_tie = i - ref_start + 1;

        // 3) 合并更新（把 ref_tie 个 vref 的平均秩叠到每个 target 的 R1[g]）
        for (size_t g = 1; g < G; ++g) {
            const double rrcur     = static_cast<double>(grank[g]);
            const size_t  t        = ref_tie + tar_eq[g];           // 本轮合并块大小（>= 1）
            const double  rrnext   = rrcur + static_cast<double>(t);
            const double  avg_rank = (rrcur + rrnext + 1.0) * 0.5;  // 闭区间平均秩

            R1[g]    += static_cast<double>(ref_tie) * avg_rank;
            grank[g]  = static_cast<size_t>(rrnext);
            tar_eq[g] = 0; // 清理本轮临时
        }
        ++i;
    }
}


// 主算子
// 主算子（收口版）
template<class T, class Idx>
MannWhitneyResult
mannwhitneyu(
    const tensor::Csc<T, Idx>& A,
    const MannWhitneyuOption<T>& opt,
    const tensor::Vector<int32_t>& group_id,
    size_t n_targets,
    int threads,
    size_t* RESTRICT progress_ptr
) {
    // ---------- 形状/早退 ----------
    if UNLIKELY(group_id.size() != A.rows()) {
        throw std::invalid_argument("[mannwhitney] group_id must be length = A.rows");
    }
    if UNLIKELY(n_targets == 0) {
        return {
            tensor::Ndarray<double>::zeros({A.cols(), 0}),
            tensor::Ndarray<double>::zeros({A.cols(), 0})
        };
    }

    const bool use_sparse_value =
        opt.sparse_type != MannWhitneyuOption<T>::SparseValueMinmax::none;

    // ---------- 进度指针 ----------
    AlignedVector<size_t> progress_dummy;
    const int n_threads_runtime = (threads < 0 ? MAX_THREADS() : threads);

    progress_dummy.resize((size_t)std::max(1, n_threads_runtime), 0);
    // 注意：不能是 const；下面会 ++
    size_t* RESTRICT progress_safe =
        progress_ptr ? progress_ptr : progress_dummy.data();

    // ---------- 基本元数据 ----------
    const size_t R = A.rows();
    const size_t C = A.cols();

    // 底层指针 + 对齐假设
    const T*        RESTRICT vals    = ASSUME_ALIGNED(A.data(),         HPDEXC_ALIGN_SIZE);
    const Idx*      RESTRICT indptr  = ASSUME_ALIGNED(A.indptr(),       HPDEXC_ALIGN_SIZE);
    const Idx*      RESTRICT indices = ASSUME_ALIGNED(A.indices(),      HPDEXC_ALIGN_SIZE);
    const int32_t*  RESTRICT gid     = ASSUME_ALIGNED(group_id.data(),  HPDEXC_ALIGN_SIZE);


    // ---------- 每组总样本量（含稀疏+显式+非法） ----------
    const size_t G = n_targets + 1;
    AlignedVector<size_t> gcount(G, 0);
    for (size_t r = 0; r < R; ++r) {
        PREFETCH_R(gid + r + HPDEXC_PREFETCH_DIST, 1);
        const int32_t g = gid[r];
        if (g >= 0 && (size_t)g < G) ++gcount[(size_t)g];
    }

    // 组段偏移
    AlignedVector<size_t> off(G + 1, 0);
    for (size_t g = 1; g <= G; ++g) off[g] = off[g - 1] + gcount[g - 1];
    const size_t total_cap = off[G];

    // ---------- 结果 ----------
    auto U1_out = tensor::Ndarray<double>::zeros({C, n_targets});
    auto P_out  = tensor::Ndarray<double>::zeros({C, n_targets});
    double* RESTRICT U1_buf = U1_out.data();
    double* RESTRICT P_buf  = P_out.data();

    // ---------- 并行 ----------
    PARALLEL_REGION(threads)
{
    const size_t tid = (size_t)THREAD_ID();
    // 线程私有缓冲
    AlignedVector<T>        col_val(total_cap);           // 拼接后的每列按组分段的显式值
    AlignedVector<size_t> gnnz(G, 0);                // 每组显式非零（或非稀疏）数
    AlignedVector<size_t> invalid_cnt(G, 0);         // 每组非法值数（NaN/Inf）
    AlignedVector<size_t> sparse_cnt(G, 0);          // 每组稀疏值计数（由 gcount - gnnz - invalid 得出）

    AlignedVector<double> R1(G, 0.0);                // 参考组的秩和（对每个 target 分别累加）
    AlignedVector<double> tie_sum(G, 0.0);           // 每个 target 的 tie 修正项
    AlignedVector<bool>   has_tie(G, false);         // 是否存在并列
    AlignedVector<size_t> grank(G, 0);               // 当前秩（已推进的元素数）
    AlignedVector<size_t> tar_ptrs(G, 0);            // 各 target 在 col_val 的游标
    AlignedVector<size_t> tar_eq(G, 0);              // 本轮等于 vref 的 target 个数
    AlignedVector<size_t> tie_cnt(G, 0);             // tar-only 连续相同计数（-1 形式）
    AlignedVector<size_t> n2_eff(G, 0);              // target 有效样本量
    AlignedVector<double> U1_tmp(G, 0.0);            // 临时 U1（R1->U1）

    // 遍历列
    PARALLEL_FOR(dynamic, {for (std::ptrdiff_t cc = 0; cc < (std::ptrdiff_t)C; ++cc) {
        const size_t c = (size_t)cc;

        // 清零本列状态
        std::fill(gnnz.begin(),       gnnz.end(),       0);
        std::fill(invalid_cnt.begin(),invalid_cnt.end(),0);

        // 读取本列范围
        const Idx p0 = indptr[c];
        const Idx p1 = indptr[c + 1];

        // 1) 扫描列，按组收集显式值（同时统计非法）
        for (Idx p = p0; p < p1; ++p) {
            const Idx r_next = indices[p + HPDEXC_PREFETCH_DIST];
            PREFETCH_R(gid + (size_t)r_next, 1);
            

            const Idx rIdx = indices[p];
            if UNLIKELY(rIdx < 0 || (size_t)rIdx >= R) continue;

            const size_t r = (size_t)rIdx;
            const int32_t g = gid[r];
            if UNLIKELY(g < 0 || (size_t)g >= G) continue;

            const T v = vals[p];
            if UNLIKELY(!is_valid_value(v)) {
                ++invalid_cnt[(size_t)g];
                continue; // 非法值直接丢弃
            }

            const size_t gi  = (size_t)g;
            const size_t dst = off[gi] + gnnz[gi]++;
            col_val[dst] = v;
        }

        // 2) 组内排序（只有不保证已排序时才排序）
        if (gnnz[0] > 1 && !opt.ref_sorted)
            std::sort(col_val.data() + off[0],
                        col_val.data() + off[0] + gnnz[0]);
        for (size_t g = 1; g < G; ++g) {
            if (gnnz[g] > 1 && !opt.tar_sorted)
                std::sort(col_val.data() + off[g],
                            col_val.data() + off[g] + gnnz[g]);
        }

        // 3) 计算每组“稀疏值”数量（只有 use_sparse_value 时用到）
        if (use_sparse_value) {
            for (size_t g = 0; g < G; ++g) {
                const size_t tot = gcount[g];
                const size_t bad = invalid_cnt[g];
                sparse_cnt[g] = (tot > gnnz[g] + bad) ? (tot - gnnz[g] - bad) : 0;
            }
        } else {
            std::fill(sparse_cnt.begin(), sparse_cnt.end(), 0);
        }

        // 4) 选择合并核：计算 R1/tie_sum/has_tie
        std::fill(R1.begin(),      R1.end(),      0.0);
        std::fill(tie_sum.begin(), tie_sum.end(), 0.0);
        std::fill(has_tie.begin(), has_tie.end(), false);
        std::fill(grank.begin(),   grank.end(),   0);
        std::fill(tar_eq.begin(),  tar_eq.end(),  0);
        std::fill(tie_cnt.begin(), tie_cnt.end(), 0);

        std::copy(off.begin(), off.begin() + G, tar_ptrs.begin());

        const T* RESTRICT refv     = col_val.data() + off[0];
        const size_t nref_exp = gnnz[0];

        if (!use_sparse_value) {
            // 非稀疏：是否需要 tie 修正
            if (!opt.tie_correction) {
                sparse_core_without_tie<T>(
                    col_val.data(), off.data(), gnnz.data(), G,
                    refv, nref_exp,
                    tar_ptrs.data(), grank.data(),
                    tar_eq.data(), R1.data()
                );
            } else {
                sparse_none_core<T>(
                    col_val.data(), off.data(), gnnz.data(), G,
                    refv, nref_exp,
                    tar_ptrs.data(), grank.data(),
                    tie_cnt.data(), tar_eq.data(),
                    R1.data(), tie_sum.data(), has_tie.data()
                );
            }
        } else {
            // 稀疏值参与
            const auto st = opt.sparse_type;
            if ((opt.tie_correction || opt.method != MannWhitneyuOption<T>::Method::exact) &&
                st == MannWhitneyuOption<T>::SparseValueMinmax::strict_minor)
            {
                // 稀疏块在序列头
                for (size_t g = 1; g < G; ++g) {
                    const size_t ref_sp = sparse_cnt[0];
                    const size_t tar_sp = sparse_cnt[g];
                    const size_t sp_tie = ref_sp + tar_sp;
                    if (sp_tie > 1) {
                        const double tt = (double)sp_tie;
                        tie_sum[g] += (tt*tt*tt - tt);
                        has_tie[g]  = true;
                    }
                    grank[g] = sp_tie;
                    if (ref_sp > 0) {
                        const double avg = (1.0 + (double)sp_tie) * 0.5;
                        R1[g] = (double)ref_sp * avg;
                    }
                }
                sparse_none_core<T>(
                    col_val.data(), off.data(), gnnz.data(), G,
                    refv, nref_exp,
                    tar_ptrs.data(), grank.data(),
                    tie_cnt.data(), tar_eq.data(),
                    R1.data(), tie_sum.data(), has_tie.data()
                );
            }
            else if ((opt.tie_correction || opt.method != MannWhitneyuOption<T>::Method::exact) &&
                        st == MannWhitneyuOption<T>::SparseValueMinmax::strict_major)
            {
                // 稀疏块在序列尾
                sparse_none_core<T>(
                    col_val.data(), off.data(), gnnz.data(), G,
                    refv, nref_exp,
                    tar_ptrs.data(), grank.data(),
                    tie_cnt.data(), tar_eq.data(),
                    R1.data(), tie_sum.data(), has_tie.data()
                );
                for (size_t g = 1; g < G; ++g) {
                    const size_t ref_sp = sparse_cnt[0];
                    const size_t tar_sp = sparse_cnt[g];
                    const size_t sp_tie = ref_sp + tar_sp;
                    if (sp_tie > 1) {
                        const double tt = (double)sp_tie;
                        tie_sum[g] += (tt*tt*tt - tt);
                        has_tie[g]  = true;
                    }
                    grank[g] += sp_tie;
                    if (ref_sp > 0) {
                        const size_t N = gnnz[0] + sparse_cnt[0] + gnnz[g] + sparse_cnt[g];
                        const size_t start = N - sp_tie + 1;
                        const size_t end   = N;
                        const double avg = ( (double)start + (double)end ) * 0.5;
                        R1[g] += (double)ref_sp * avg;
                    }
                }
            }
            else {
                // 稀疏在中间：与显式块归并
                AlignedVector<bool>     have_run(G, false);
                AlignedVector<T>        run_val(G);
                AlignedVector<size_t> run_len(G, 0);
                // 根据列密度选择 dense / sparse 归并器（可用更聪明的启发式）
                size_t col_exp_total = 0;
                for (size_t g = 0; g < G; ++g) col_exp_total += gnnz[g];
                const bool very_sparse = (col_exp_total * 2 < total_cap);

                if (very_sparse) {
                    sparse_medium_core_sparse<T>(
                        col_val.data(), off.data(), gnnz.data(), sparse_cnt.data(), G,
                        refv, nref_exp, opt.sparse_value,
                        tar_ptrs.data(), grank.data(), tar_eq.data(),
                        sparse_cnt.data(), // 这里作为 sp_left
                        have_run.data(), run_val.data(), run_len.data(),
                        R1.data(), tie_sum.data(), has_tie.data()
                    );
                } else {
                    sparse_medium_core_dense<T>(
                        col_val.data(), off.data(), gnnz.data(), sparse_cnt.data(), G,
                        refv, nref_exp, opt.sparse_value,
                        tar_ptrs.data(), grank.data(), tar_eq.data(),
                        sparse_cnt.data(), // 这里作为 sp_left
                        have_run.data(), run_val.data(), run_len.data(),
                        R1.data(), tie_sum.data(), has_tie.data()
                    );
                }
            }
        }

        // 5) 有效样本量 + R1->U1
        size_t n1_eff = gnnz[0];
        if (use_sparse_value) {
            const size_t spr =
                (gcount[0] > gnnz[0] + invalid_cnt[0]) ? (gcount[0] - gnnz[0] - invalid_cnt[0]) : 0;
            n1_eff += spr;
            for (size_t g = 1; g < G; ++g) {
                const size_t bad = invalid_cnt[g];
                const size_t sp  = (gcount[g] > gnnz[g] + bad) ? (gcount[g] - gnnz[g] - bad) : 0;
                n2_eff[g] = gnnz[g] + sp;
            }
        } else {
            for (size_t g = 1; g < G; ++g) n2_eff[g] = gnnz[g];
        }

        // 基本有效性检查（与早先逻辑一致：样本太少抛错）
        if UNLIKELY(n1_eff < 2) {
            throw std::runtime_error("Sample too small for reference at column " + std::to_string(c));
        }
        for (size_t g = 1; g < G; ++g) {
            if UNLIKELY(n2_eff[g] < 2) {
                throw std::runtime_error("Sample too small for group " + std::to_string(g) +
                                            " at column " + std::to_string(c));
            }
        }

        const double base = (double)n1_eff * ((double)n1_eff + 1.0) * 0.5;
        for (size_t g = 1; g < G; ++g) {
            U1_tmp[g] = R1[g] - base;
        }

        // 6) 写 U（输出 U1；P 的 U 使用原始 U1，方向在 p 函数内部处理）
        for (size_t g = 1; g < G; ++g) {
            U1_buf[c * n_targets + (g - 1)] = U1_tmp[g];
        }

        // 7）计算 P
        if UNLIKELY(opt.method == MannWhitneyuOption<T>::Method::exact) {
            bool any_tie = false;
            if (!use_sparse_value) {
                auto has_neighbor_equal = [&](size_t g)->bool{
                    const size_t n = gnnz[g];
                    const T* a = col_val.data() + off[g];
                    for (size_t i = 1; i < n; ++i) {
                        if (!(a[i-1] < a[i]) && !(a[i] < a[i-1])) return true;
                    }
                    return false;
                };
                any_tie = has_neighbor_equal(0);
                for (size_t g = 1; g < G && !any_tie; ++g) any_tie |= has_neighbor_equal(g);
            } else {
                for (size_t g = 1; g < G; ++g) if UNLIKELY(has_tie[g]) { any_tie = true; break; }
            }
        
            for (size_t g = 1; g < G; ++g) {
                const double U1v = U1_tmp[g];
                double p;
                if UNLIKELY(any_tie || has_tie[g]) {
                    if LIKELY(opt.alternative == MannWhitneyuOption<T>::Alternative::two_sided) {
                        p = p_asymptotic_two_sided<T>(U1v, n1_eff, n2_eff[g], tie_sum[g], opt.use_continuity);
                    } else if UNLIKELY(opt.alternative == MannWhitneyuOption<T>::Alternative::less) {
                        p = p_asymptotic_less<T>(U1v, n1_eff, n2_eff[g], tie_sum[g], opt.use_continuity);
                    } else { // greater
                        p = p_asymptotic_greater<T>(U1v, n1_eff, n2_eff[g], tie_sum[g], opt.use_continuity);
                    }
                } else {
                    if LIKELY(opt.alternative == MannWhitneyuOption<T>::Alternative::two_sided) {
                        const double U2 = double(n1_eff) * double(n2_eff[g]) - U1v;
                        const double p_right = p_exact<T>(std::max(U1v, U2), n1_eff, n2_eff[g]);
                        p = 2.0 * p_right;
                    } else if UNLIKELY(opt.alternative == MannWhitneyuOption<T>::Alternative::greater) {
                        p = p_exact<T>(U1v, n1_eff, n2_eff[g]);
                    } else { // less
                        const double U2 = double(n1_eff) * double(n2_eff[g]) - U1v;
                        p = p_exact<T>(U2, n1_eff, n2_eff[g]);
                    }
                }
                P_buf[c * n_targets + (g - 1)] = std::clamp(p, 0.0, 1.0);
            }
        } else if UNLIKELY(opt.method == MannWhitneyuOption<T>::Method::asymptotic) {
            for (size_t g = 1; g < G; ++g) {
                const double U1v = U1_tmp[g];
                double p;
                if UNLIKELY(opt.alternative == MannWhitneyuOption<T>::Alternative::greater) {
                    p = p_asymptotic_greater<T>(U1v, n1_eff, n2_eff[g], tie_sum[g], opt.use_continuity);
                } else if UNLIKELY(opt.alternative == MannWhitneyuOption<T>::Alternative::less) {
                    p = p_asymptotic_less<T>(U1v, n1_eff, n2_eff[g], tie_sum[g], opt.use_continuity);
                } else { // two-sided
                    p = p_asymptotic_two_sided<T>(U1v, n1_eff, n2_eff[g], tie_sum[g], opt.use_continuity);
                }
                P_buf[c * n_targets + (g - 1)] = std::clamp(p, 0.0, 1.0);
            }
        } else { // automatic — LIKELY path
            for (size_t g = 1; g < G; ++g) {
                const double U1v = U1_tmp[g];
                const bool asym = (choose_method<T>(n1_eff, n2_eff[g], has_tie[g])
                                   == MannWhitneyuOption<T>::Method::asymptotic);
                double p;
                if LIKELY(asym) {
                    if LIKELY(opt.alternative == MannWhitneyuOption<T>::Alternative::two_sided) {
                        p = p_asymptotic_two_sided<T>(U1v, n1_eff, n2_eff[g], tie_sum[g], opt.use_continuity);
                    } else if UNLIKELY(opt.alternative == MannWhitneyuOption<T>::Alternative::less) {
                        p = p_asymptotic_less<T>(U1v, n1_eff, n2_eff[g], tie_sum[g], opt.use_continuity);
                    } else {
                        p = p_asymptotic_greater<T>(U1v, n1_eff, n2_eff[g], tie_sum[g], opt.use_continuity);
                    }
                } else {
                    if LIKELY(opt.alternative == MannWhitneyuOption<T>::Alternative::two_sided) {
                        const double U2 = double(n1_eff) * double(n2_eff[g]) - U1v;
                        const double p_right = p_exact<T>(std::max(U1v, U2), n1_eff, n2_eff[g]);
                        p = 2.0 * p_right;
                    } else if UNLIKELY(opt.alternative == MannWhitneyuOption<T>::Alternative::greater) {
                        p = p_exact<T>(U1v, n1_eff, n2_eff[g]);
                    } else {
                        const double U2 = double(n1_eff) * double(n2_eff[g]) - U1v;
                        p = p_exact<T>(U2, n1_eff, n2_eff[g]);
                    }
                }
                P_buf[c * n_targets + (g - 1)] = std::clamp(p, 0.0, 1.0);
            }
        }

        // 进度（每列+1）
        ++progress_safe[tid];
    }}) // parallel for loop

} // parallel region

    return { U1_out, P_out };
}


// group mean
template<class T, class Idx>
tensor::Ndarray<double> group_mean(
    const tensor::Csc<T,Idx>& A,
    const tensor::Vector<int32_t>& group_id,
    size_t n_groups,
    int threads,
    size_t* RESTRICT progress_ptr
) {
    const size_t R = A.rows();
    const size_t C = A.cols();

    if (group_id.size() != R) {
        throw std::invalid_argument("[group_mean] group_id length mismatch with matrix rows");
    }

    // 结果矩阵 (n_groups × C)
    auto means = tensor::Ndarray<double>::zeros({n_groups, C});
    double* mean_buf = means.data();

    // 每组样本量（包含隐含稀疏值）
    AlignedVector<size_t> group_size(n_groups, 0);
    for (size_t r = 0; r < R; ++r) {
        int g = group_id.data()[r];
        if (g >= 0 && (size_t)g < n_groups) {
            ++group_size[g];
        }
    }

    size_t dummy = 0;
    size_t* RESTRICT prog = progress_ptr ? progress_ptr : &dummy;

    const T* RESTRICT vals   = A.data();
    const Idx* RESTRICT indptr = A.indptr();
    const Idx* RESTRICT ridx   = A.indices();

    PARALLEL_REGION(threads)
{
    AlignedVector<double> sum_local(n_groups, 0.0);
    AlignedVector<size_t> count_local(n_groups, 0);

    PARALLEL_FOR(dynamic, {for (std::ptrdiff_t cc = 0; cc < (std::ptrdiff_t)C; ++cc) {
        const size_t c = (size_t)cc;
        std::fill(sum_local.begin(), sum_local.end(), 0.0);
        std::fill(count_local.begin(), count_local.end(), 0);

        const Idx p0 = indptr[c];
        const Idx p1 = indptr[c+1];
        for (Idx p = p0; p < p1; ++p) {
            const size_t r = (size_t)ridx[p];
            int g = group_id.data()[r];
            if (g < 0 || (size_t)g >= n_groups) continue;
            sum_local[g] += (double)vals[p];
            count_local[g] ++;
        }

        // 计算均值（使用实际有效值计数，而不是总组大小）
        for (size_t g = 0; g < n_groups; ++g) {
            size_t valid_count = count_local[g];
            if (valid_count == 0) {
                mean_buf[g * C + c] = std::numeric_limits<double>::quiet_NaN();
            } else {
                mean_buf[g * C + c] = sum_local[g] / (double)valid_count;
            }
        }

        // 进度
        const size_t tid = (size_t)THREAD_ID();
        prog[tid] ++;
    }}) // parallel for loop
} // parallel region

    return means;
}


// 显式实例化
#define INST_MANNWHITNEYU(T, Idx) \
    template MannWhitneyResult mannwhitneyu<T, Idx>( \
        const tensor::Csc<T, Idx>& A, \
        const MannWhitneyuOption<T>& opt, \
        const tensor::Vector<int32_t>& group_id, \
        size_t n_targets, \
        int threads, \
        size_t* RESTRICT progress_ptr);
#define DO_IDX(T)\
    INST_MANNWHITNEYU(T, int32_t); \
    INST_MANNWHITNEYU(T, int64_t)

DTYPE_DISPATCH(DO_IDX);
#undef DO_IDX
#undef INST_MANNWHITNEYU

#define INST_GROUP_MEAN(T, Idx) \
    template tensor::Ndarray<double> group_mean<T, Idx>( \
        const tensor::Csc<T, Idx>& A, \
        const tensor::Vector<int32_t>& group_id, \
        size_t n_groups, \
        int threads, \
        size_t* RESTRICT progress_ptr);
#define DO_IDX(T)\
    INST_GROUP_MEAN(T, int32_t); \
    INST_GROUP_MEAN(T, int64_t)
DTYPE_DISPATCH(DO_IDX);
#undef DO_IDX
#undef INST_GROUP_MEAN

}


