#pragma once
#include <cstddef>

namespace hpdexc_detail {

// 非稀疏分支核心：已初始化的缓冲，计算 R1 与 tie_sum（不做 fill/init）
template<typename T>
inline void mwu_scan_dense_core(
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
    // 结算 tar-only 并列块
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
            tar_eq[g] = 0;
            while (tp < gend && !(col_val[tp] < vref) && !(vref < col_val[tp])) {
                ++tp;
                ++tar_eq[g];
            }
        }
        const size_t ref_start = i;
        while (i + 1 < nrefcol && !(refv[i + 1] < vref) && !(vref < refv[i + 1])) ++i;
        const size_t ref_tie = i - ref_start + 1;
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
    for (size_t g = 1; g < G; ++g) {
        size_t tp        = tar_ptrs_local[g];
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
inline void mwu_scan_dense_core_notie(
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

} // namespace hpdexc_detail


