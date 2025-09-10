#include "mannwhitney.hpp"
#include "common.hpp"

#include <algorithm>
#include <hwy/highway.h>
#include <hwy/contrib/sort/vqsort.h>
#include <cstddef>
#include <cstring>
#include <vector>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif


namespace hpdexc {

// 统一排序包装：优先使用 Highway VQSort（升序），否则回退到 std::sort
template<typename T>
static HPDEXC_FORCE_INLINE void vqsort_or_std(T* data, size_t n) {
    if (n <= 1) return;
    if constexpr (std::is_same_v<T, float>) {
        hwy::VQSort(data, n, hwy::SortAscending{});
    } else if constexpr (std::is_same_v<T, double>) {
        hwy::VQSort(data, n, hwy::SortAscending{});
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        hwy::VQSort(data, n, hwy::SortAscending{});
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        hwy::VQSort(data, n, hwy::SortAscending{});
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        hwy::VQSort(data, n, hwy::SortAscending{});
    } else if constexpr (std::is_same_v<T, int16_t>) {
        hwy::VQSort(data, n, hwy::SortAscending{});
    } else if constexpr (std::is_same_v<T, int32_t>) {
        hwy::VQSort(data, n, hwy::SortAscending{});
    } else if constexpr (std::is_same_v<T, int64_t>) {
        hwy::VQSort(data, n, hwy::SortAscending{});
    } else {
        std::sort(data, data + n);
    }
}

// 计算 p（仅 asymptotic；对齐 SciPy _get_mwu_z + sf）
template<typename T>
static HPDEXC_FORCE_INLINE double mwu_p_asymptotic(
    double U, 
    size_t n1,
    size_t n2, 
    long double tie_sum,
    bool use_continuity
){
    const long double N   = (long double)n1 + (long double)n2;
    const long double mu  = (long double)n1 * (long double)n2 * 0.5L;

    // 方差，带 tie 修正
    long double var;
    const long double denom = N * (N - 1.0L);
    if HPDEXC_LIKELY(denom > 0.0L) {
        var = (long double)n1 * (long double)n2 / 12.0L *
              ((long double)N + 1.0L - tie_sum / denom);
    } else {
        var = (long double)n1 * (long double)n2 *
              ((long double)N + 1.0L) / 12.0L;
    }
    if HPDEXC_UNLIKELY(var <= 0.0L) return 1.0;

    const long double sd = std::sqrt(var);

    // continuity correction
    long double numerator = (long double)U - mu;
    if (use_continuity) numerator -= 0.5L;

    const double z = numerator / sd;
    double p = sf(z);

    return p;
}


// ---- Exact p-value for Mann–Whitney U (no ties), SciPy-aligned ----
// 规则：SF(k)=P(U >= k)（含等号）；利用对称性从较小一侧求和；
// alternative: less→用U2；greater→用U1；two_sided→max(U1,U2)并×2（上限1）
// Exact p-value for Mann–Whitney U (no ties), SciPy-aligned
template<typename T>
static HPDEXC_FORCE_INLINE double mwu_p_exact_no_tie(
    double U,       // 已经根据 alternative 挑选好的 U
    size_t n1,
    size_t n2
){
    const size_t Umax = n1 * n2;
    if HPDEXC_UNLIKELY(Umax == 0) return 1.0;

    // 钳制 U 到 [0, Umax]，并取整（使用 floor 确保与 SciPy 一致）
    const double U_clip = std::max(0.0, std::min((double)Umax, U));
    const size_t u_stat = (size_t)std::floor(U_clip);

    // ---------- DP：构造 U 的计数分布 ----------
    const size_t SZ = Umax + 1;
    unsigned long long dp_buf[MannWhitneyMaxStackNnz];
    unsigned long long* dp = (SZ <= MannWhitneyMaxStackNnz) ? dp_buf : new unsigned long long[SZ];
    unsigned long long ndp_buf[MannWhitneyMaxStackNnz];
    unsigned long long* ndp = (SZ <= MannWhitneyMaxStackNnz) ? ndp_buf : new unsigned long long[SZ];

    std::fill(dp, dp + SZ, 0ULL);
    dp[0] = 1ULL;

    for (size_t i = 1; i <= n1; ++i) {
        std::fill(ndp, ndp + SZ, 0ULL);
        unsigned long long win = 0ULL;
        const size_t up = i * n2;
        for (size_t u = 0; u <= up; ++u) {
            win += dp[u];
            if (u >= n2 + 1) win -= dp[u - (n2 + 1)];
            ndp[u] = win;
        }
        std::swap(dp, ndp);
    }

    if (SZ > MannWhitneyMaxStackNnz) {
        delete[] dp;
        delete[] ndp;
    }

    long double total = 0.0L;
    for (size_t u = 0; u <= Umax; ++u) total += (long double)dp[u];

    // ---------- SciPy 的 SF 对称优化 ----------
    const size_t kc    = Umax - u_stat;
    const size_t small = (u_stat < kc ? u_stat : kc);

    long double cdf_small = 0.0L;
    for (size_t u = 0; u <= small; ++u) cdf_small += (long double)dp[u];
    const long double pmf_small = (long double)dp[small];

    long double sf_ge;
    if (u_stat <= kc) {
        // 左半边：SF(u_stat) = 1 - CDF(u_stat) + PMF(u_stat)
        sf_ge = 1.0L - cdf_small/total + pmf_small/total;
    } else {
        // 右半边：SF(u_stat) = CDF(kc)
        sf_ge = cdf_small/total;
    }

    return (double)sf_ge;
}


// 统一稀疏块：min/max 通用、无分支、使用 "+=" 叠加
static HPDEXC_FORCE_INLINE void mwu_apply_sparse_block(
    bool         position_head,
    size_t       ref_sp_cnt,
    size_t       tar_sp_cnt,
    size_t       N,
    long double& R1g,
    long double& tie_sum_g,
    bool&        has_tie_g,
    size_t&      grank_g
){
    const size_t sp_tie = ref_sp_cnt + tar_sp_cnt;
    const long double tt = (long double)sp_tie;
    const long double mask = (sp_tie > 1) ? 1.0L : 0.0L;
    tie_sum_g += (tt*tt*tt - tt) * mask;
    has_tie_g  = has_tie_g || (sp_tie > 1);

    const long double m   = position_head ? 1.0L : 0.0L;
    const long double nm  = 1.0L - m;
    const long double ldN = (long double)N;
    const long double ldS = (long double)sp_tie;
    const long double start = m*1.0L + nm*(ldN - ldS + 1.0L);
    const long double end   = m*ldS  + nm*(ldN);
    const long double avg   = (start + end) * 0.5L;

    R1g    += (long double)ref_sp_cnt * avg;
    grank_g += sp_tie;
}

// tar-only 并列合并器：合并同值 run，按需推进 grank
template<class T>
static HPDEXC_FORCE_INLINE void mwu_emit_tar_block_merge(
    const T&     val,
    size_t       count,
    bool&        have_run,
    T&           run_val,
    size_t&      run_len,
    long double& tie_sum_g,
    bool&        has_tie_g,
    size_t&      grank_g
){
    if (!count) return;
    if (have_run && !(run_val < val) && !(val < run_val)) {
        run_len += count;
    } else {
        if (run_len > 1) {
            const long double tt = (long double)run_len;
            tie_sum_g += tt*tt*tt - tt;
            has_tie_g  = true;
        }
        run_val  = val;
        run_len  = count;
        have_run = true;
    }
    grank_g += count;
}

// none（稀疏值既非最小也非最大）核心：已初始化缓冲，计算 R1 与 tie_sum（不含任何 fill/init）。
// 需要调用侧：
//   - 提供 tar_ptrs_local[g]=off[g]，grank[g]=0，tar_eq[g]=0；
//   - sp_left[g] = sparse_value_cnt[g] 的工作拷贝（或直接在此处拷贝）；
//   - have_run[g]=false，run_len[g]=0（run_val[g] 无需初始化）。
template<typename T>
static HPDEXC_FORCE_INLINE void mwu_scan_sparse_none_core(
    const T* __restrict col_val,
    const size_t* __restrict off,
    const size_t* __restrict gnnz,
    const size_t* __restrict sparse_value_cnt,
    const size_t  G,
    const T* __restrict refv,
    const size_t nref_exp,
    const T      sparse_value,
    size_t* __restrict tar_ptrs_local,
    size_t* __restrict grank,
    size_t* __restrict tar_eq,
    size_t* __restrict sp_left,
    bool*   __restrict have_run,
    T*      __restrict run_val,
    size_t* __restrict run_len,
    long double* __restrict R1,
    long double* __restrict tie_sum,
    bool* __restrict has_tie
){
    auto flush_run = [&](size_t g){
        if (run_len[g] > 1) {
            const long double tt = (long double)run_len[g];
            tie_sum[g] += tt*tt*tt - tt;
            has_tie[g]  = true;
        }
        run_len[g]  = 0;
        have_run[g] = false;
    };

    size_t i = 0;
    while (i < nref_exp || sp_left[0] > 0) {
        T vref;
        size_t ref_tie = 0;

        if (sp_left[0] > 0 && (i >= nref_exp || !(refv[i] < sparse_value))) {
            vref = sparse_value;
            size_t k_exp = 0;
            while (i + k_exp < nref_exp && !(refv[i + k_exp] < vref) && !(vref < refv[i + k_exp])) ++k_exp;
            ref_tie = sp_left[0] + k_exp;
            sp_left[0] = 0;
            i += k_exp;
        } else {
            vref = (i < nref_exp) ? refv[i] : sparse_value;
            size_t ref_start = i;
            while (i + 1 < nref_exp && !(refv[i + 1] < vref) && !(vref < refv[i + 1])) ++i;
            ref_tie = i - ref_start + 1;
            ++i;
        }

        for (size_t g = 1; g < G; ++g) {
            size_t& tp        = tar_ptrs_local[g];
            const size_t gend = off[g] + gnnz[g];

            while (true) {
                const bool has_exp = (tp < gend) && (col_val[tp] < vref);
                const bool has_sp  = (sp_left[g] > 0) && (sparse_value < vref);
                if (!has_exp && !has_sp) break;

                if (has_exp && has_sp) {
                    const T ev = col_val[tp];
                    if (ev < sparse_value) {
                        size_t j = tp + 1;
                        while (j < gend && !(col_val[j] < ev) && !(ev < col_val[j])) ++j;
                        const size_t blk = j - tp;
                        mwu_emit_tar_block_merge(ev, blk, have_run[g], run_val[g], run_len[g],
                                                 tie_sum[g], has_tie[g], grank[g]);
                        tp = j;
                    } else if (sparse_value < ev) {
                        const size_t blk = sp_left[g];
                        mwu_emit_tar_block_merge(sparse_value, blk, have_run[g], run_val[g], run_len[g],
                                                 tie_sum[g], has_tie[g], grank[g]);
                        sp_left[g] = 0;
                    } else {
                        size_t j = tp + 1;
                        while (j < gend && !(col_val[j] < sparse_value) && !(sparse_value < col_val[j])) ++j;
                        const size_t blk_exp = j - tp;
                        mwu_emit_tar_block_merge(sparse_value, blk_exp, have_run[g], run_val[g], run_len[g],
                                                 tie_sum[g], has_tie[g], grank[g]);
                        tp = j;
                        if (sp_left[g] > 0) {
                            const size_t blk_sp = sp_left[g];
                            mwu_emit_tar_block_merge(sparse_value, blk_sp, have_run[g], run_val[g], run_len[g],
                                                     tie_sum[g], has_tie[g], grank[g]);
                            sp_left[g] = 0;
                        }
                    }
                } else if (has_exp) {
                    const T ev = col_val[tp];
                    size_t j = tp + 1;
                    while (j < gend && !(col_val[j] < ev) && !(ev < col_val[j])) ++j;
                    const size_t blk = j - tp;
                    mwu_emit_tar_block_merge(ev, blk, have_run[g], run_val[g], run_len[g],
                                             tie_sum[g], has_tie[g], grank[g]);
                    tp = j;
                } else {
                    const size_t blk = sp_left[g];
                    mwu_emit_tar_block_merge(sparse_value, blk, have_run[g], run_val[g], run_len[g],
                                             tie_sum[g], has_tie[g], grank[g]);
                    sp_left[g] = 0;
                }
            }
            flush_run(g);

            size_t eq = 0;
            while (tp < gend && !(col_val[tp] < vref) && !(vref < col_val[tp])) { ++tp; ++eq; }
            if (sp_left[g] > 0 && !(sparse_value < vref) && !(vref < sparse_value)) {
                eq += sp_left[g];
                sp_left[g] = 0;
            }
            tar_eq[g] = eq;
        }

        for (size_t g = 1; g < G; ++g) {
            const long double rrcur   = (long double)grank[g];
            const size_t t            = ref_tie + tar_eq[g];
            const long double rrnext  = rrcur + (long double)t;
            const long double avg_rank= (rrcur + rrnext + 1.0L) * 0.5L;

            R1[g]    += (long double)ref_tie * avg_rank;
            grank[g]  = (size_t)rrnext;

            if HPDEXC_UNLIKELY(t > 1) {
                const long double tt = (long double)t;
                tie_sum[g] += tt*tt*tt - tt;
                has_tie[g]  = true;
            }
            tar_eq[g] = 0;
        }
    }

    for (size_t g = 1; g < G; ++g) {
        size_t& tp        = tar_ptrs_local[g];
        const size_t gend = off[g] + gnnz[g];

        have_run[g] = false;
        run_len[g]  = 0;

        while (tp < gend || sp_left[g] > 0) {
            bool has_exp = (tp < gend);
            bool has_sp  = (sp_left[g] > 0);

            T cand; bool take_sp = false;
            if (!has_exp) { cand = sparse_value; take_sp = true; }
            else if (!has_sp) { cand = col_val[tp]; }
            else {
                const T ev = col_val[tp];
                if (sparse_value < ev) { cand = sparse_value; take_sp = true; }
                else                   { cand = ev; }
            }

            if (take_sp) {
                mwu_emit_tar_block_merge(cand, sp_left[g], have_run[g], run_val[g], run_len[g],
                                         tie_sum[g], has_tie[g], grank[g]);
                sp_left[g] = 0;
            } else {
                size_t j = tp + 1;
                while (j < gend && !(col_val[j] < cand) && !(cand < col_val[j])) ++j;
                const size_t blk = j - tp;
                mwu_emit_tar_block_merge(cand, blk, have_run[g], run_val[g], run_len[g],
                                         tie_sum[g], has_tie[g], grank[g]);
                tp = j;
            }
        }
        flush_run(g);
    }
}

// 非稀疏分支：已初始化的缓冲，计算 R1 与 tie_sum（不含任何 fill/init）。
template<typename T>
static HPDEXC_FORCE_INLINE void mwu_scan_dense_core(
    const T* __restrict col_val,
    const size_t* __restrict off,
    const size_t* __restrict gnnz,
    const size_t G,
    const T* __restrict refv,
    const size_t nrefcol,
    size_t* __restrict tar_ptrs_local,
    size_t* __restrict grank,
    size_t* __restrict tie_cnt,
    size_t* __restrict tar_eq,
    long double* __restrict R1,
    long double* __restrict tie_sum,
    bool* __restrict has_tie
){

    // 内联结算 tar-only 并列块的工具
    size_t i = 0;
    auto flush = [&](size_t g){
        if (tie_cnt[g] > 0) {
            const long double t = (long double)(tie_cnt[g] + 1);
            tie_sum[g] += t*t*t - t;
            tie_cnt[g] = 0;
            has_tie[g] = true;
        }
    };
    
    while (i < nrefcol) {
        const T vref = (T)refv[i];
        // 1) 推进所有 < vref 的 tar，并结算 tar-only tie
        for (size_t g = 1; g < G; ++g) {
            size_t& tp        = tar_ptrs_local[g];
            const size_t gbeg = off[g];
            const size_t gend = off[g] + gnnz[g];
            while (tp < gend && col_val[tp] < vref) {
                if (tp > gbeg) {
                    if (!(col_val[tp] < col_val[tp-1]) && !(col_val[tp-1] < col_val[tp])) {
                        ++tie_cnt[g];
                    } else {
                        flush(g);
                    }
                }
                ++tp;
                ++grank[g];
            }
            flush(g);
            // 2) 统计等于 vref 的 tar 个数（不动 grank，只推进 tp）
            tar_eq[g] = 0;
            while (tp < gend && !(col_val[tp] < vref) && !(vref < col_val[tp])) {
                ++tp;
                ++tar_eq[g];
            }
        }
        // 3) ref 等值块
        const size_t ref_start = i;
        while (i + 1 < nrefcol && !(refv[i + 1] < vref) && !(vref < refv[i + 1])) ++i;
        const size_t ref_tie = i - ref_start + 1;
        // 4) 合并更新 R1/grank/tie_sum
        for (size_t g = 1; g < G; ++g) {
            const long double rrcur  = (long double)grank[g];
            const size_t t           = ref_tie + tar_eq[g];
            const long double rrnext = rrcur + (long double)t;
            const long double avg_rank = (rrcur + rrnext + 1.0L) * 0.5L;
            R1[g]    += (long double)ref_tie * avg_rank;
            grank[g]  = (size_t)rrnext;
            if HPDEXC_UNLIKELY(t > 1) {
                const long double tt = (long double)t;
                tie_sum[g] += tt*tt*tt - tt;
                has_tie[g] = true;
            }
            tie_cnt[g] = 0;
            tar_eq[g]  = 0;
        }
        ++i;
    }
    // 5) 扫尾：所有剩余 tar-only 并列块（位于所有 ref 之后）
    for (size_t g = 1; g < G; ++g) {
        size_t& tp       = tar_ptrs_local[g];
        const size_t gend= off[g] + gnnz[g];
        while (tp < gend) {
            size_t j = tp + 1;
            while (j < gend && !(col_val[j] < col_val[tp]) && !(col_val[tp] < col_val[j])) ++j;
            const size_t block_len = j - tp;
            if (block_len > 1) {
                const long double tt = (long double)block_len;
                tie_sum[g] += tt*tt*tt - tt;
                has_tie[g] = true;
            }
            tp = j;
        }
    }
}

// 非稀疏分支（无 tie 修正）：已初始化的缓冲，只计算 R1（不含任何 fill/init）。
// 要求：各组段 [off[g], off[g]+gnnz[g]) 已升序；refv 指向组0段；tar_ptrs_local[g]=off[g]；grank[g]=0。
template<typename T>
static HPDEXC_FORCE_INLINE void mwu_scan_dense_core_notie(
    const T* __restrict col_val,
    const size_t* __restrict off,
    const size_t* __restrict gnnz,
    const size_t G,
    const T* __restrict refv,
    const size_t nrefcol,
    size_t* __restrict tar_ptrs_local, // 初值 off[g]
    size_t* __restrict grank,          // 初值 0
    size_t* __restrict tar_eq,         // 临时缓冲，长度 >= G，用于存放每个 g 的等值个数
    long double* __restrict R1         // 初值 0
){
    size_t i = 0;
    while (i < nrefcol) {
        const T vref = refv[i];

        // 1) 对每个 target：推进所有 < vref 的元素（推进 grank），并统计 == vref 的个数（不推进 grank）
        for (size_t g = 1; g < G; ++g) {
            size_t& tp        = tar_ptrs_local[g];
            const size_t gend = off[g] + gnnz[g];

            while (tp < gend && col_val[tp] < vref) {
                ++tp;
                ++grank[g];
            }
            size_t eq = 0;
            while (tp < gend && !(col_val[tp] < vref) && !(vref < col_val[tp])) {
                ++tp;
                ++eq;
            }
            tar_eq[g] = eq;
        }

        // 2) ref 等值块 [ref_start..i]
        const size_t ref_start = i;
        while (i + 1 < nrefcol && !(refv[i + 1] < vref) && !(vref < refv[i + 1])) ++i;
        const size_t ref_tie = i - ref_start + 1;

        // 3) 合并 ref_tie 与 tar_eq，统一用平均秩公式更新 R1 与 grank
        for (size_t g = 1; g < G; ++g) {
            const long double rrcur    = (long double)grank[g];
            const size_t t             = ref_tie + tar_eq[g];
            const long double rrnext   = rrcur + (long double)t;
            const long double avg_rank = (rrcur + rrnext + 1.0L) * 0.5L;

            R1[g]   += (long double)ref_tie * avg_rank;
            grank[g] = (size_t)rrnext;
            tar_eq[g] = 0;
        }

        ++i;
    }
}

template<typename T>
static HPDEXC_FORCE_INLINE bool is_valid_value(const T& v) {
    return !is_nan(v) && !is_inf(v);
}

template<typename T>
HPDEXC_FORCE_INLINE typename MannWhitneyOption<T>::Method 
mannwhitney_choose_method(size_t n1, size_t n2, bool has_tie) {
    return n1 > 8 && n2 > 8 || has_tie ? MannWhitneyOption<T>::Method::asymptotic : MannWhitneyOption<T>::Method::exact;
}

// ---------- 主算子 ----------
template<typename T>
MannWhitneyResult
mannwhitney(
    const CscView& A,
    const MannWhitneyOption<T>& opt,
    const DenseView& group_id,   // ref=0, tar=1.., -1 忽略（1D 向量）
    const size_t n_targets,
    int threads,
    void* progress_ptr
){
    // === 形状校验：group_id 必须是一维，长度等于行数 ===
    if HPDEXC_UNLIKELY(!(group_id.cols == 1 && group_id.rows == A.rows)) {
        throw std::invalid_argument(
            "[mannwhitney] group_id must be a 1D array of length = A.rows"
        );
    }

    // 检查进度缓冲区大小
    if HPDEXC_UNLIKELY(threads > MannWhitneyProgressDefaultBufSize) {
        throw std::runtime_error("progress buffer too small for provided `threads`; Max: " + std::to_string(MannWhitneyProgressDefaultBufSize));
    }
    
    size_t progress_buf[MannWhitneyProgressDefaultBufSize];
    if (progress_ptr == nullptr) {
        progress_ptr = progress_buf; // 避免悬空指针
    }

    const size_t R = A.rows;
    const size_t C = A.cols;
    const T* __restrict data = A.data_ptr<T>();

    // === 统计组数与各组总样本量（全局） ===
    const size_t G = n_targets + 1; // 0=ref, 1..n_targets=tar
    size_t gcount_buf[MannWhitneyMaxStackGroups];
    size_t* gcount = (G <= MannWhitneyMaxStackGroups) ? gcount_buf : new size_t[G];
    std::fill(gcount, gcount + G, size_t(0));
    for (size_t r = 0; r < R; ++r) {
        const int32_t g = static_cast<const int32_t*>(group_id.data)[r * group_id.stride]; // ★ 正确索引
        if (g >= 0 && (size_t)g < G) ++gcount[(size_t)g];
    }
    const size_t nref = gcount[0];

    // === 组段偏移与列容量上界 ===
    size_t off_buf[MannWhitneyMaxStackGroups + 1];
    size_t* off = (G <= MannWhitneyMaxStackGroups) ? off_buf : new size_t[G + 1];

    // === 结果缓冲（持有存储，返回时以视图暴露） ===
    MannWhitneyResult result;
    result.set(n_targets, C);  // 分配并初始化 U/P

    if (n_targets == 0) {
        if (G > MannWhitneyMaxStackGroups) delete[] gcount;
        if (G > MannWhitneyMaxStackGroups) delete[] off;
        return result;
    } // 无目标组，直接返回空结果

    off[0] = 0;
    for (size_t g=1; g<=G; ++g) off[g] = off[g-1] + gcount[g-1];
    const size_t total_cap = off[G];
    // off[G] 表示col中，每一组占据段的offset

    #pragma omp parallel num_threads(threads) if HPDEXC_OMP_GUARD(threads > 1)
    {
        // === 线程私有缓冲 ===
        size_t      gnnz_buf[MannWhitneyMaxStackGroups];
        long double R1_buf [MannWhitneyMaxStackGroups];
        size_t      tar_ptrs_buf[MannWhitneyMaxStackGroups];
        long double tie_sum_buf[MannWhitneyMaxStackGroups];
        bool        has_tie_buf[MannWhitneyMaxStackGroups];

        size_t*      gnnz     = (G <= MannWhitneyMaxStackGroups) ? gnnz_buf     : new size_t[G];
        long double* R1      = (G <= MannWhitneyMaxStackGroups) ? R1_buf      : new long double[G];
        size_t*      tar_ptrs= (G <= MannWhitneyMaxStackGroups) ? tar_ptrs_buf: new size_t[G];
        long double* tie_sum = (G <= MannWhitneyMaxStackGroups) ? tie_sum_buf : new long double[G];
        bool*        has_tie = (G <= MannWhitneyMaxStackGroups) ? has_tie_buf : new bool[G];

        size_t      tie_cnt_buf[MannWhitneyMaxStackGroups];
        size_t*     tie_cnt = (G <= MannWhitneyMaxStackGroups) ? tie_cnt_buf : new size_t[G];
        size_t      grank_buf[MannWhitneyMaxStackGroups];
        size_t*     grank   = (G <= MannWhitneyMaxStackGroups) ? grank_buf   : new size_t[G];
        size_t      tar_eq_buf[MannWhitneyMaxStackGroups];
        size_t*     tar_eq  = (G <= MannWhitneyMaxStackGroups) ? tar_eq_buf  : new size_t[G];
        size_t      tar_ptrs_local_buf[MannWhitneyMaxStackGroups];
        size_t*     tar_ptrs_local = (G <= MannWhitneyMaxStackGroups) ? tar_ptrs_local_buf : new size_t[G];

        size_t      invalid_value_cnt_buf[MannWhitneyMaxStackGroups];
        size_t      sparse_value_cnt_buf[MannWhitneyMaxStackGroups];
        size_t*     invalid_value_cnt = (G <= MannWhitneyMaxStackGroups) ? invalid_value_cnt_buf : new size_t[G];
        size_t*     sparse_value_cnt = (G <= MannWhitneyMaxStackGroups) ? sparse_value_cnt_buf : new size_t[G];

        // n2_eff 缓冲区：移到线程级别，避免循环内分配
        size_t      n2_eff_buf[MannWhitneyMaxStackGroups];
        size_t*     n2_eff = (G <= MannWhitneyMaxStackGroups) ? n2_eff_buf : new size_t[G];

        // 单独的 U 缓冲区，避免与 R1 aliasing
        long double U_buf[MannWhitneyMaxStackGroups];
        long double U1_buf[MannWhitneyMaxStackGroups];
        long double* U = (G <= MannWhitneyMaxStackGroups) ? U_buf : new long double[G];
        long double* U1 = (G <= MannWhitneyMaxStackGroups) ? U1_buf : new long double[G];

        // 大缓冲（分段存每组取出的值）- 按实际需要分配
        T col_val_buf[MannWhitneyMaxStackColCap];
        T* col_val = nullptr; // 将在循环内按需分配

        // === 每个线程处理分配到的列 ===
        #pragma omp for schedule(dynamic)
        for (std::ptrdiff_t cc = 0; cc < (std::ptrdiff_t)C; ++cc) {
            const size_t c = (size_t)cc;

            // 计算当前列的实际容量
            size_t col_cap = 0;
            for (size_t g = 0; g < G; ++g) {
                col_cap += gcount[g];
            }
            
            // 按实际需要分配 col_val
            if (col_cap <= MannWhitneyMaxStackColCap) {
                col_val = col_val_buf;
            } else {
                col_val = new T[col_cap];
            }

        // 1) 扫描该列，按组收集值
        const int64_t p0 = A.indptr_at(c);
        const int64_t p1 = A.indptr_at(c+1);
        std::fill(gnnz, gnnz+G, 0);
        std::fill(invalid_value_cnt, invalid_value_cnt + G, 0);
        for (int64_t p = p0; p < p1; ++p) {
            const int64_t r64 = A.row_at(p);
            if HPDEXC_UNLIKELY(r64 < 0 || (size_t)r64 >= R) continue; // 保护：跳过非法行索引
            const size_t r = (size_t)r64;
            const int32_t g = static_cast<const int32_t*>(group_id.data)[r * group_id.stride];
            if HPDEXC_UNLIKELY(g < 0 || (size_t)g >= G) continue; // 忽略无效组

            const T v = (T)data[p];
            if HPDEXC_UNLIKELY(!is_valid_value(v)) { ++invalid_value_cnt[(size_t)g]; continue; }

            const size_t gi  = (size_t)g;
            const size_t idx = off[gi] + gnnz[gi]++;
            col_val[idx] = v;
        }
        
        // 2) 各组内部排序
        const size_t ref_valid_value = opt.use_sparse_value ? R - invalid_value_cnt[0] : gnnz[0];
        if HPDEXC_UNLIKELY(ref_valid_value < 2) {
            throw std::runtime_error("Sample is too small for reference group at column " + std::to_string(c));
        }
        if HPDEXC_LIKELY(!opt.ref_sorted && gnnz[0] > 1) {
            vqsort_or_std(&col_val[off[0]], gnnz[0]);
        }
        for (size_t g=1; g<G; ++g) {
            const size_t valid_value = opt.use_sparse_value ? R - invalid_value_cnt[g] : gnnz[g];
            if HPDEXC_UNLIKELY(valid_value < 2) {
                throw std::runtime_error("Sample is too small for group " + std::to_string(g) + " at column " + std::to_string(c));
            }
            if HPDEXC_LIKELY(!opt.tar_sorted && gnnz[g] > 1) {
                vqsort_or_std(&col_val[off[g]], gnnz[g]);
            }
        }

        // 3) 合并扫描：计算 R1[g], tie_sum[g]
        std::fill(has_tie, has_tie + G, false);

        // 非稀疏：分 tie 修正与否。
        if (!opt.use_sparse_value) {
            std::fill(R1,      R1      + G, 0.0L);
            std::fill(has_tie, has_tie + G, false);
            std::fill(tie_sum, tie_sum + G, 0.0L);

            const T* refv        = col_val + off[0];
            const size_t nrefcol = gnnz[0];

            if (!opt.tie_correction) {
                // 不做 tie 修正：仅计算 R1
                std::fill(grank,   grank   + G, size_t(0));
                std::fill(tar_eq,  tar_eq  + G, size_t(0));
                std::copy(off, off + G, tar_ptrs_local);

                mwu_scan_dense_core_notie<T>(
                    col_val,
                    off,
                    gnnz,
                    G,
                    refv,
                    nrefcol,
                    tar_ptrs_local,
                    grank,
                    tar_eq,
                    R1
                );
            } else {
                // 做 tie 修正：计算 R1 与 tie_sum
                std::fill(tie_cnt, tie_cnt + G, size_t(0));
                std::fill(grank,   grank   + G, size_t(0));
                std::fill(tar_eq,  tar_eq  + G, size_t(0));
                std::copy(off, off + G, tar_ptrs_local);

                mwu_scan_dense_core<T>(
                    col_val,
                    off,
                    gnnz,
                    G,
                    refv,
                    nrefcol,
                    tar_ptrs_local,
                    grank,
                    tie_cnt,
                    tar_eq,
                    R1,
                    tie_sum,
                    has_tie
                );
            }
        }
        
        // 稀疏最小：稀疏块在序列最前端
        else if (opt.use_sparse_value && (opt.tie_correction || opt.method != MannWhitneyOption<T>::Method::exact) && opt.is_spare_minmax == MannWhitneyOption<T>::SparseValueMinmax::min) {
            std::fill(tie_sum, tie_sum + G, 0.0L);
            std::fill(R1, R1 + G, 0.0L);

            const T* refv = col_val + off[0];
            const size_t nrefcol = gnnz[0];

            // 稀疏元素数量（零视为最小，在序列最前端形成一个合并块）
            for (size_t g = 0; g < G; ++g) {
                const size_t bad = invalid_value_cnt[g];
                sparse_value_cnt[g] = (gcount[g] > gnnz[g] + bad) ? (gcount[g] - gnnz[g] - bad) : 0;
            }

            // 预先合并零块的影响
            for (size_t g = 1; g < G; ++g) {
                const size_t ref_sp_cnt = sparse_value_cnt[0];
                const size_t tar_sp_cnt = sparse_value_cnt[g];
                const size_t sp_tie = ref_sp_cnt + tar_sp_cnt;
                if (sp_tie > 1) {
                    const long double tt = (long double)sp_tie;
                    tie_sum[g] += tt*tt*tt - tt;
                    has_tie[g] = true;
                }
                grank[g] = sp_tie; // 排名游标从零块之后开始
                if (ref_sp_cnt > 0) {
                    const long double avg_sp_rank = (1.0L + (long double)sp_tie) * 0.5L;
                    R1[g] = (long double)ref_sp_cnt * avg_sp_rank;
                }
            }

            // 初始化本地游标并扫描非零段
            std::fill(tie_cnt, tie_cnt + G, size_t(0));
            std::fill(tar_eq,  tar_eq  + G, size_t(0));
            std::copy(off, off + G, tar_ptrs_local);

            mwu_scan_dense_core<T>(
                col_val,
                off,
                gnnz,
                G,
                refv,
                nrefcol,
                tar_ptrs_local,
                grank,
                tie_cnt,
                tar_eq,
                R1,
                tie_sum,
                has_tie
            );

        }
        // 稀疏最大：稀疏块在序列末端
        else if (opt.use_sparse_value && 
                    (opt.tie_correction || opt.method != MannWhitneyOption<T>::Method::exact) && 
                    opt.is_spare_minmax == MannWhitneyOption<T>::SparseValueMinmax::max) {
            std::fill(tie_sum, tie_sum + G, 0.0L);
            std::fill(R1, R1 + G, 0.0L);

            const T* refv = col_val + off[0];
            const size_t nrefcol = gnnz[0];

            // 稀疏元素数量（零视为最大，在序列末端形成一个合并块）
            for (size_t g = 0; g < G; ++g) {
                const size_t bad = invalid_value_cnt[g];
                sparse_value_cnt[g] = (gcount[g] > gnnz[g] + bad) ? (gcount[g] - gnnz[g] - bad) : 0;
            }

            // 先只扫描非稀疏段
            std::fill(tie_cnt, tie_cnt + G, size_t(0));
            std::fill(grank,   grank   + G, size_t(0));
            std::fill(tar_eq,  tar_eq  + G, size_t(0));
            std::copy(off, off + G, tar_ptrs_local);

            mwu_scan_dense_core<T>(
                col_val,
                off,
                gnnz,
                G,
                refv,
                nrefcol,
                tar_ptrs_local,
                grank,
                tie_cnt,
                tar_eq,
                R1,
                tie_sum,
                has_tie
            );

            // 扫尾：统一处理末端的稀疏块
            for (size_t g = 1; g < G; ++g) {
                const size_t ref_sp_cnt = sparse_value_cnt[0];
                const size_t tar_sp_cnt = sparse_value_cnt[g];
                const size_t sp_tie     = ref_sp_cnt + tar_sp_cnt;
                if (sp_tie > 1) {
                    const long double tt = (long double)sp_tie;
                    tie_sum[g] += tt*tt*tt - tt;
                    has_tie[g] = true;
                }
                grank[g] += sp_tie;
                if (ref_sp_cnt > 0) {
                    const size_t N = gnnz[0] + sparse_value_cnt[0] + gnnz[g] + sparse_value_cnt[g];
                    const size_t start_rank = N - sp_tie + 1;
                    const size_t end_rank   = N;
                    const long double avg_rank = (start_rank + end_rank) * 0.5L;
                    R1[g] += (long double)ref_sp_cnt * avg_rank;
                }
            }

        }
        // 稀疏 none：稀疏值位于中间，需与显式归并
        else if (opt.use_sparse_value && (opt.tie_correction || opt.method != MannWhitneyOption<T>::Method::exact) && opt.is_spare_minmax == MannWhitneyOption<T>::SparseValueMinmax::none) {
            std::fill(tie_sum, tie_sum + G, 0.0L);
            std::fill(R1,      R1      + G, 0.0L);
            std::fill(has_tie, has_tie + G, false);

            const T* refv        = col_val + off[0];
            const size_t nrefcol = gnnz[0];

            for (size_t g = 0; g < G; ++g) {
                const size_t bad = invalid_value_cnt[g];
                sparse_value_cnt[g] = (gcount[g] > gnnz[g] + bad) ? (gcount[g] - gnnz[g] - bad) : 0;
            }

            // 工作缓冲
            size_t sp_left_buf[MannWhitneyMaxStackGroups];
            size_t* sp_left = (G <= MannWhitneyMaxStackGroups) ? sp_left_buf : new size_t[G];
            bool   have_run_buf[MannWhitneyMaxStackGroups];
            bool*  have_run = (G <= MannWhitneyMaxStackGroups) ? have_run_buf : new bool[G];
            T      run_val_buf[MannWhitneyMaxStackGroups];
            T*     run_val  = (G <= MannWhitneyMaxStackGroups) ? run_val_buf  : new T[G];
            size_t run_len_buf[MannWhitneyMaxStackGroups];
            size_t* run_len = (G <= MannWhitneyMaxStackGroups) ? run_len_buf : new size_t[G];

            for (size_t g = 0; g < G; ++g) {
                sp_left[g]  = sparse_value_cnt[g];
                have_run[g] = false;
                run_len[g]  = 0;
            }

            std::fill(grank,   grank   + G, size_t(0));
            std::fill(tar_eq,  tar_eq  + G, size_t(0));
            std::copy(off, off + G, tar_ptrs_local);

            mwu_scan_sparse_none_core<T>(
                col_val,
                off,
                gnnz,
                sparse_value_cnt,
                G,
                refv,
                nrefcol,
                opt.sparse_value,
                tar_ptrs_local,
                grank,
                tar_eq,
                sp_left,
                have_run,
                run_val,
                run_len,
                R1,
                tie_sum,
                has_tie
            );

            if (G > MannWhitneyMaxStackGroups) { delete[] sp_left; delete[] have_run; delete[] run_val; delete[] run_len; }
        }

        // 计算有效样本数（显式 + 稀疏 − 非法）并进行 R1->U1 转换
        size_t n1_eff = gnnz[0];
        if HPDEXC_LIKELY(opt.use_sparse_value) {
            // 重新计算稀疏数，避免依赖分支内局部变量
            const size_t spr = (gcount[0] > gnnz[0] + invalid_value_cnt[0]) ? (gcount[0] - gnnz[0] - invalid_value_cnt[0]) : 0;
            n1_eff += spr;
            for (size_t g = 1; g < G; ++g) {
                const size_t bad = invalid_value_cnt[g];
                const size_t sp = (gcount[g] > gnnz[g] + bad) ? (gcount[g] - gnnz[g] - bad) : 0;
                n2_eff[g] = gnnz[g] + sp;
            }
        } else {
            for (size_t g = 1; g < G; ++g) n2_eff[g] = gnnz[g];
        }

        // 检查边界情况：如果 n1_eff 或 n2_eff 为 0，直接设置 U=0, p=1.0
        for (size_t g = 1; g < G; ++g) {
            if HPDEXC_UNLIKELY(n1_eff == 0 || n2_eff[g] == 0) {
                U[g] = 0.0L;
                result.U_buf[c * n_targets + (g-1)] = 0.0;
                result.P_buf[c * n_targets + (g-1)] = 1.0;
                continue;
            }
        }

        // R1 -> U1 转换（跳过已处理的边界情况）
        const long double base = (long double)n1_eff * ((long double)n1_eff + 1.0L) * 0.5L;
        double f = 1.0;
        if (opt.alternative == MannWhitneyOption<T>::Alternative::less) {
            for (size_t g = 1; g < G; ++g) {
                U1[g] = R1[g] - base;
                const long double U2 = (long double)n1_eff * (long double)n2_eff[g] - U1[g];
                U[g] = U2;
            }
        } else if (opt.alternative == MannWhitneyOption<T>::Alternative::greater) {
            for (size_t g = 1; g < G; ++g) {
                U1[g] = R1[g] - base;
                U[g] = U1[g];
            }
        } else {
            for (size_t g = 1; g < G; ++g) {
                U1[g] = R1[g] - base;
                const long double U2 = (long double)n1_eff * (long double)n2_eff[g] - U1[g];
                U[g] = std::max(U1[g], U2);
            }
            f = 2.0; // 与 scipy 对齐
        }
        
        // 5) 计算 P
        if (opt.method == MannWhitneyOption<T>::Method::exact) {
            for (size_t g = 1; g < G; ++g) {
                result.U_buf[c * n_targets + (g-1)] = U1[g];
                double P = mwu_p_exact_no_tie<T>(U[g], n1_eff, n2_eff[g]);
                P *= f;
                P = std::clamp(P, 0.0, 1.0);
                result.P_buf[c * n_targets + (g-1)] = P;
            }
        } else if (opt.method == MannWhitneyOption<T>::Method::asymptotic) {
            for (size_t g = 1; g < G; ++g) {
                result.U_buf[c * n_targets + (g-1)] = U1[g];
                double P = mwu_p_asymptotic<T>(U[g], n1_eff, n2_eff[g], tie_sum[g], opt.use_continuity);
                P *= f;
                P = std::clamp(P, 0.0, 1.0);
                result.P_buf[c * n_targets + (g-1)] = P;
            }

        } else { // autometic
            for (size_t g = 1; g < G; ++g) {
                result.U_buf[c * n_targets + (g-1)] = U1[g];
                const bool asym = mannwhitney_choose_method<T>(n1_eff, n2_eff[g], has_tie[g]) == MannWhitneyOption<T>::Method::asymptotic;
                double P = asym ?
                    mwu_p_asymptotic<T>(U[g], n1_eff, n2_eff[g], tie_sum[g], opt.use_continuity) :
                    mwu_p_exact_no_tie<T>(U[g], n1_eff, n2_eff[g]);
                P *= f;
                P = std::clamp(P, 0.0, 1.0);
                result.P_buf[c * n_targets + (g-1)] = P;
            }
        }

        // 释放当前列的 col_val（如果是在堆上分配的）
        if HPDEXC_UNLIKELY(col_val != col_val_buf) {
            delete[] col_val;
        }

        // 更新进度
        ((size_t*)progress_ptr)[omp_get_thread_num()]++;

        } // omp for end

        // === 每个线程释放一次 ===
        if HPDEXC_UNLIKELY(G > MannWhitneyMaxStackGroups) {
            delete[] gnnz; delete[] R1; delete[] tar_ptrs; delete[] tie_sum; delete[] has_tie;
            delete[] tie_cnt; delete[] grank; delete[] tar_eq; delete[] tar_ptrs_local; delete[] invalid_value_cnt; delete[] sparse_value_cnt;
            delete[] n2_eff; delete[] U; delete[] U1;
        }
    }

    // 8）清理缓冲
    if HPDEXC_UNLIKELY(G > MannWhitneyMaxStackGroups) {delete[] gcount; delete[] off;}

    
    return result;
}


template<typename T>
inline MannWhitneyResult
mannwhitney_hist(
    const CscView& A,
    const MannWhitneyOption<T>& opt,
    const DenseView& group_id,
    const size_t n_targets,
    int threads = 1,
    void* progress_ptr = nullptr
) {
    // TODO: 实现直方图版本的 Mann-Whitney U 测试
    MannWhitneyResult result;
    result.set(n_targets, A.cols);
    return result;
}

#define INSTANTIATE_MANNWHITNEY(T) \
template MannWhitneyResult mannwhitney<T>(const CscView& A, const MannWhitneyOption<T>& opt, const DenseView& group_id, const size_t n_targets, int threads, void* progress_ptr); \
template MannWhitneyResult mannwhitney_hist<T>(const CscView& A, const MannWhitneyOption<T>& opt, const DenseView& group_id, const size_t n_targets, int threads, void* progress_ptr);

HPDEXC_DTYPE_DISPATCH(INSTANTIATE_MANNWHITNEY);


}