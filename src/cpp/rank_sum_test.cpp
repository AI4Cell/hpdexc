// 简化版 Rank-Sum（Mann–Whitney U）计算的测试实现
// 仅处理二组：ref 与 tar；提供非稀疏分支，便于与 naive 实现做一致性校验。

#include <vector>
#include <algorithm>
#include <cstddef>
#include <type_traits>
#include "include/rank_sum_core.hpp"
#include "include/common.hpp"
#include "include/mannwhitney.hpp"

template<class T>
struct RankSumResult {
    long double U_ref;     // ref(组0) 的 U
    long double tie_sum;   // sum_over_ties (t^3 - t)
    std::size_t n_ref;
    std::size_t n_tar;
    bool has_tie;
};

// 非稀疏分支：输入两组数值；若未排序则内部排序；
// 使用与 src/cpp/mannwhitney.cpp 中“非稀疏+tie 修正”一致的扫描逻辑，
// 特别包含：对每个 vref 周期对 tar 等值段的精确计数 tar_eq 修复。
template<class T>
RankSumResult<T> rank_sum_nonsparse(
    const std::vector<T>& ref_in,
    const std::vector<T>& tar_in,
    bool ref_sorted,
    bool tar_sorted
){
    static_assert(std::is_arithmetic<T>::value, "T must be numeric");

    // 复制并按需排序（稳定排序非必需，这里按值升序即可）
    std::vector<T> ref = ref_in;
    std::vector<T> tar = tar_in;
    if (!ref_sorted && ref.size() > 1) std::sort(ref.begin(), ref.end());
    if (!tar_sorted && tar.size() > 1) std::sort(tar.begin(), tar.end());

    const std::size_t nref = ref.size();
    const std::size_t ntar = tar.size();

    long double R1 = 0.0L;
    long double tie_sum = 0.0L;
    bool has_tie = false;

    // 用 core 统一实现
    std::vector<T> col;
    col.reserve(nref + ntar);
    for (auto v : ref) col.push_back(v);
    for (auto v : tar) col.push_back(v);

    // 构造视图：G=2 组，off[0]=0, off[1]=nref；gnnz[0]=nref, gnnz[1]=ntar
    size_t off[3]  = {0, nref, nref + ntar};
    size_t gnnz[2] = {nref, ntar};

    long double R1_groups[2]   = {0.0L, 0.0L};
    long double tie_groups[2]  = {0.0L, 0.0L};
    bool has_tie_groups[2]     = {false, false};
    size_t tar_ptrs_local[2]   = {0, off[1]};
    size_t grank_arr[2]        = {0, 0};
    size_t tie_cnt_arr[2]      = {0, 0};
    size_t tar_eq_arr[2]       = {0, 0};

    hpdexc_detail::mwu_scan_dense_core<T>(
        col.data(),
        off,
        gnnz,
        2,
        col.data() + off[0],
        nref,
        tar_ptrs_local,
        grank_arr,
        tie_cnt_arr,
        tar_eq_arr,
        R1_groups,
        tie_groups,
        has_tie_groups
    );

    R1    = R1_groups[1];
    tie_sum = tie_groups[1];
    has_tie = has_tie_groups[1];

    const long double U_ref = R1 - static_cast<long double>(nref) * static_cast<long double>(nref + 1) * 0.5L;
    return RankSumResult<T>{U_ref, tie_sum, nref, ntar, has_tie};
}

// （测试版）不计算 p，只测试 R1/U 与 tie_sum 的一致性

// 统一稀疏块：min/max 通用、无分支、使用 "+=" 叠加
static inline void mwu_apply_sparse_block(
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
static inline void mwu_emit_tar_block_merge(
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
//   - sp_left[g] = sparse_cnt[g] 的工作拷贝（或直接在此处拷贝）；
//   - have_run[g]=false，run_len[g]=0（run_val[g] 无需初始化）。
template<typename T>
static inline void mwu_scan_sparse_none_core(
    const T* __restrict col_val,
    const size_t* __restrict off,
    const size_t* __restrict gnnz,
    const size_t* __restrict sparse_cnt,
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

            if (t > 1) {
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
static inline void mwu_scan_dense_core(
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
    auto flush = [&](size_t g){
        if (tie_cnt[g] > 0) {
            const long double t = (long double)(tie_cnt[g] + 1);
            tie_sum[g] += t*t*t - t;
            tie_cnt[g] = 0;
            has_tie[g] = true;
        }
    };
    size_t i = 0;
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
            if (t > 1) {
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
static inline void mwu_scan_dense_core_notie(
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
static inline bool is_valid_value(const T& v) {
    return !hpdexc::is_nan(v) && !hpdexc::is_inf(v);
}

template<typename T>
inline typename hpdexc::MannWhitneyOption<T>::Method 
mannwhitney_choose_method(size_t n1, size_t n2, bool has_tie) {
    return n1 > 8 && n2 > 8 || has_tie ? hpdexc::MannWhitneyOption<T>::Method::asymptotic : hpdexc::MannWhitneyOption<T>::Method::exact;
}

// ---------- 主算子 ----------
namespace hpdexc {

template<typename T>
MannWhitneyResult
mannwhitney(
    const CscView& A,
    const MannWhitneyOption<T>& opt,
    const DenseView& group_id,   // ref=0, tar=1.., -1 忽略（1D 向量）
    const size_t n_targets,
    int threads
){
    // === 形状校验：group_id 必须是一维，长度等于行数 ===
    if (!(group_id.cols == 1 && group_id.rows == A.rows)) {
        throw std::invalid_argument(
            "[mannwhitney] group_id must be a 1D array of length = A.rows"
        );
    }

    const size_t R = A.rows;
    const size_t C = A.cols;
    const T* __restrict data = A.data_ptr<T>();

    // === 统计组数与各组总样本量（全局） ===
    const size_t G = n_targets + 1; // 0=ref, 1..n_targets=tar
    size_t gcount_buf[hpdexc::MannWhitneyMaxStackGroups];
    size_t* gcount = (G <= hpdexc::MannWhitneyMaxStackGroups) ? gcount_buf : new size_t[G];
    std::fill(gcount, gcount + G, size_t(0));
    for (size_t r = 0; r < R; ++r) {
        const int32_t g = static_cast<const int32_t*>(group_id.data)[r * group_id.stride]; // ★ 正确索引
        if (g >= 0 && (size_t)g < G) ++gcount[(size_t)g];
    }
    const size_t nref = gcount[0];

    // === 组段偏移与列容量上界 ===
    size_t off_buf[hpdexc::MannWhitneyMaxStackGroups + 1];
    size_t* off = (G <= hpdexc::MannWhitneyMaxStackGroups) ? off_buf : new size_t[G + 1];

    // === 结果缓冲（持有存储，返回时以视图暴露） ===
    hpdexc::MannWhitneyResult result;
    result.set(n_targets, C);  // 分配并初始化 U/P

    if (n_targets == 0) {
        if (G > hpdexc::MannWhitneyMaxStackGroups) delete[] gcount;
        if (G > hpdexc::MannWhitneyMaxStackGroups) delete[] off;
        return result;
    } // 无目标组，直接返回空结果

    off[0] = 0;
    for (size_t g=1; g<=G; ++g) off[g] = off[g-1] + gcount[g-1];
    const size_t total_cap = off[G];
    // off[G] 表示col中，每一组占据段的offset

    #pragma omp parallel for schedule(dynamic) num_threads(threads) if(threads>1)
    for (std::ptrdiff_t cc = 0; cc < (std::ptrdiff_t)C; ++cc) {
        const size_t c = (size_t)cc;

        // 线程私有小缓冲
        size_t      gnnz_buf[hpdexc::MannWhitneyMaxStackGroups];
        long double R1_buf [hpdexc::MannWhitneyMaxStackGroups];
        size_t      tar_ptrs_buf[hpdexc::MannWhitneyMaxStackGroups];
        long double tie_sum_buf[hpdexc::MannWhitneyMaxStackGroups];
        bool        has_tie_buf[hpdexc::MannWhitneyMaxStackGroups];

        size_t*      gnnz     = (G <= hpdexc::MannWhitneyMaxStackGroups) ? gnnz_buf     : new size_t[G];
        long double* R1      = (G <= hpdexc::MannWhitneyMaxStackGroups) ? R1_buf      : new long double[G];
        size_t*      tar_ptrs= (G <= hpdexc::MannWhitneyMaxStackGroups) ? tar_ptrs_buf: new size_t[G];
        long double* tie_sum = (G <= hpdexc::MannWhitneyMaxStackGroups) ? tie_sum_buf : new long double[G];
        bool*        has_tie = (G <= hpdexc::MannWhitneyMaxStackGroups) ? has_tie_buf : new bool[G];

        size_t      tie_cnt_buf[hpdexc::MannWhitneyMaxStackGroups];
        size_t*     tie_cnt = (G <= hpdexc::MannWhitneyMaxStackGroups) ? tie_cnt_buf : new size_t[G];
        size_t      grank_buf[hpdexc::MannWhitneyMaxStackGroups];
        size_t*     grank   = (G <= hpdexc::MannWhitneyMaxStackGroups) ? grank_buf   : new size_t[G];
        size_t      tar_eq_buf[hpdexc::MannWhitneyMaxStackGroups];
        size_t*     tar_eq  = (G <= hpdexc::MannWhitneyMaxStackGroups) ? tar_eq_buf  : new size_t[G];
        size_t      tar_ptrs_local_buf[hpdexc::MannWhitneyMaxStackGroups];
        size_t*     tar_ptrs_local = (G <= hpdexc::MannWhitneyMaxStackGroups) ? tar_ptrs_local_buf : new size_t[G];

        // 工作缓冲：先用于 invalid 计数，后复用为 sparse_cnt
        size_t      invalid_value_cnt_sparse_cnt_buf[hpdexc::MannWhitneyMaxStackGroups];
        size_t*     invalid_value_cnt = (G <= hpdexc::MannWhitneyMaxStackGroups) ? invalid_value_cnt_sparse_cnt_buf : new size_t[G];

        long double* U1 = R1; // 复用

        // 大缓冲（分段存每组取出的值）
        T col_val_buf[hpdexc::MannWhitneyMaxStackColCap]; // 用于存储列，默认大小是
        T* col_val = (total_cap <= hpdexc::MannWhitneyMaxStackColCap) ? col_val_buf : new T[total_cap];

        // 1) 扫描该列，按组收集值
        const int64_t p0 = A.indptr_at(c);
        const int64_t p1 = A.indptr_at(c+1);
        std::fill(gnnz, gnnz+G, 0);
        std::fill(invalid_value_cnt, invalid_value_cnt + G, 0);
        for (int64_t p = p0; p < p1; ++p) {
            const int64_t r64 = A.row_at(p);
            if (r64 < 0 || (size_t)r64 >= R) continue; // 保护：跳过非法行索引
            const size_t r = (size_t)r64;
            const int32_t g = static_cast<const int32_t*>(group_id.data)[r * group_id.stride];
            if (g < 0 || (size_t)g >= G) continue; // 忽略无效组

            const T v = (T)data[p];
            if (!is_valid_value(v)) { ++invalid_value_cnt[(size_t)g]; continue; }

            const size_t gi  = (size_t)g;
            const size_t idx = off[gi] + gnnz[gi]++;
            col_val[idx] = v;
        }
        
        // 2) 各组内部排序
        if (!opt.ref_sorted && gnnz[0] > 1) {
            std::sort(&col_val[off[0]], &col_val[off[0]] + gnnz[0]);
        }
        for (size_t g=1; g<G; ++g) {
            if (!opt.tar_sorted && gnnz[g] > 1) {
                std::sort(&col_val[off[g]], &col_val[off[g]] + gnnz[g]);
            }
        }

        // 3) 合并扫描：计算 R1[g], tie_sum[g]
        std::fill(has_tie, has_tie + G, false);

        // 使用lambda表达式节藕，将tie计算分离
        auto scan_sorted = [&](const T* refv, const size_t nrefcol) {

        };
        

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
        else if (opt.use_sparse_value && (opt.tie_correction || opt.method != hpdexc::MannWhitneyOption<T>::Method::exact) && opt.is_spare_minmax == hpdexc::MannWhitneyOption<T>::SparseValueMinmax::min) {
            std::fill(tie_sum, tie_sum + G, 0.0L);
            std::fill(R1, R1 + G, 0.0L);

            const T* refv = col_val + off[0];
            const size_t nrefcol = gnnz[0];

            // 稀疏元素数量（零视为最小，在序列最前端形成一个合并块）
            // 复用 invalid_value_cnt 作为 sparse_cnt
            size_t* sparse_cnt = invalid_value_cnt;
            for (size_t g = 0; g < G; ++g) {
                const size_t bad = invalid_value_cnt[g];
                sparse_cnt[g] = (gcount[g] > gnnz[g] + bad) ? (gcount[g] - gnnz[g] - bad) : 0;
            }

            // 预先合并零块的影响
            for (size_t g = 1; g < G; ++g) {
                const size_t ref_sp_cnt = sparse_cnt[0];
                const size_t tar_sp_cnt = sparse_cnt[g];
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

            // sparse_cnt 与 invalid_value_cnt 复用，同一指针，统一在尾部释放
        }
        // 稀疏最大：稀疏块在序列末端
        else if (opt.use_sparse_value && 
                    (opt.tie_correction || opt.method != hpdexc::MannWhitneyOption<T>::Method::exact) && 
                    opt.is_spare_minmax == hpdexc::MannWhitneyOption<T>::SparseValueMinmax::max) {
            std::fill(tie_sum, tie_sum + G, 0.0L);
            std::fill(R1, R1 + G, 0.0L);

            const T* refv = col_val + off[0];
            const size_t nrefcol = gnnz[0];

            // 稀疏元素数量（零视为最大，在序列末端形成一个合并块）
            // 复用 invalid_value_cnt 作为 sparse_cnt
            size_t* sparse_cnt = invalid_value_cnt;
            for (size_t g = 0; g < G; ++g) {
                const size_t bad = invalid_value_cnt[g];
                sparse_cnt[g] = (gcount[g] > gnnz[g] + bad) ? (gcount[g] - gnnz[g] - bad) : 0;
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
                const size_t ref_sp_cnt = sparse_cnt[0];
                const size_t tar_sp_cnt = sparse_cnt[g];
                const size_t sp_tie     = ref_sp_cnt + tar_sp_cnt;
                if (sp_tie > 1) {
                    const long double tt = (long double)sp_tie;
                    tie_sum[g] += tt*tt*tt - tt;
                    has_tie[g] = true;
                }
                grank[g] += sp_tie;
                if (ref_sp_cnt > 0) {
                    const size_t N = gnnz[0] + sparse_cnt[0] + gnnz[g] + sparse_cnt[g];
                    const size_t start_rank = N - sp_tie + 1;
                    const size_t end_rank   = N;
                    const long double avg_rank = (start_rank + end_rank) * 0.5L;
                    R1[g] += (long double)ref_sp_cnt * avg_rank;
                }
            }

            // sparse_cnt 与 invalid_value_cnt 复用，同一指针，统一在尾部释放
        }
        // 稀疏 none：稀疏值位于中间，需与显式归并
        else if (opt.use_sparse_value && (opt.tie_correction || opt.method != hpdexc::MannWhitneyOption<T>::Method::exact) && opt.is_spare_minmax == hpdexc::MannWhitneyOption<T>::SparseValueMinmax::none) {
            std::fill(tie_sum, tie_sum + G, 0.0L);
            std::fill(R1,      R1      + G, 0.0L);
            std::fill(has_tie, has_tie + G, false);

            const T* refv        = col_val + off[0];
            const size_t nrefcol = gnnz[0];

            // 复用 invalid_value_cnt 作为 sparse_cnt
            size_t* sparse_cnt = invalid_value_cnt;
            for (size_t g = 0; g < G; ++g) {
                const size_t bad = invalid_value_cnt[g];
                sparse_cnt[g] = (gcount[g] > gnnz[g] + bad) ? (gcount[g] - gnnz[g] - bad) : 0;
            }

            // 工作缓冲
            size_t sp_left_buf[hpdexc::MannWhitneyMaxStackGroups];
            size_t* sp_left = (G <= hpdexc::MannWhitneyMaxStackGroups) ? sp_left_buf : new size_t[G];
            bool   have_run_buf[hpdexc::MannWhitneyMaxStackGroups];
            bool*  have_run = (G <= hpdexc::MannWhitneyMaxStackGroups) ? have_run_buf : new bool[G];
            T      run_val_buf[hpdexc::MannWhitneyMaxStackGroups];
            T*     run_val  = (G <= hpdexc::MannWhitneyMaxStackGroups) ? run_val_buf  : new T[G];
            size_t run_len_buf[hpdexc::MannWhitneyMaxStackGroups];
            size_t* run_len = (G <= hpdexc::MannWhitneyMaxStackGroups) ? run_len_buf : new size_t[G];

            for (size_t g = 0; g < G; ++g) {
                sp_left[g]  = sparse_cnt[g];
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
                sparse_cnt,
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

            if (G > hpdexc::MannWhitneyMaxStackGroups) { /* sparse_cnt 复用 invalid_value_cnt，统一释放 */ delete[] sp_left; delete[] have_run; delete[] run_val; delete[] run_len; }
        }

        // 计算有效样本数（显式 + 稀疏 − 非法）并进行 R1->U1 转换
        size_t n1_eff = gnnz[0];
        size_t n2_eff_buf[hpdexc::MannWhitneyMaxStackGroups];
        size_t* n2_eff = (G <= hpdexc::MannWhitneyMaxStackGroups) ? n2_eff_buf : new size_t[G];
        if (opt.use_sparse_value) {
            // 重新计算稀疏数，避免依赖分支内局部变量
            for (size_t g = 0; g < G; ++g) {
                const size_t bad = invalid_value_cnt[g];
                const size_t sp = (gcount[g] > gnnz[g] + bad) ? (gcount[g] - gnnz[g] - bad) : 0;
                if (g == 0) n1_eff += sp;
                else        n2_eff[g] = gnnz[g] + sp;
            }
        } else {
            for (size_t g = 1; g < G; ++g) n2_eff[g] = gnnz[g];
        }

        // R1 -> U1 转换并仅写出 U，不计算 P
        const long double base = (long double)n1_eff * ((long double)n1_eff + 1.0L) * 0.5L;
        for (size_t g = 1; g < G; ++g) {
            const long double U = R1[g] - base;
            result.U_buf[c * n_targets + (g-1)] = U;
            result.P_buf[c * n_targets + (g-1)] = 0.0; // 测试模式：不计算 P
        }


        if (G > hpdexc::MannWhitneyMaxStackGroups) {
            delete [] n2_eff;
            delete[] gnnz; delete[] R1; delete[] tar_ptrs; delete[] tie_sum; delete[] has_tie;
            delete[] tie_cnt; delete[] grank; delete[] tar_eq; delete[] tar_ptrs_local; delete[] invalid_value_cnt;
        }
        if (total_cap > hpdexc::MannWhitneyMaxStackColCap) delete[] col_val; // 始终按分配方式释放
    }

    // 8）清理缓冲
    if (G > MannWhitneyMaxStackGroups) delete[] gcount;
    if (G > MannWhitneyMaxStackGroups) delete[] off;
    
    return result;
}

} // namespace hpdexc

namespace hpdexc {

template<typename T>
inline MannWhitneyResult
mannwhitney_hist(
    const CscView& A,
    const MannWhitneyOption<T>& opt,
    const DenseView& group_id,
    const size_t n_targets,
    int threads = 1
) {
    // TODO: 实现直方图版本的 Mann-Whitney U 测试
    MannWhitneyResult result;
    result.set(n_targets, A.cols);
    return result;
}

#define INSTANTIATE_MANNWHITNEY(T) \
template MannWhitneyResult mannwhitney<T>(const CscView& A, const MannWhitneyOption<T>& opt, const DenseView& group_id, const size_t n_targets, int threads); \
template MannWhitneyResult mannwhitney_hist<T>(const CscView& A, const MannWhitneyOption<T>& opt, const DenseView& group_id, const size_t n_targets, int threads);

HPDEXC_DTYPE_DISPATCH(INSTANTIATE_MANNWHITNEY);

} // namespace hpdexc
