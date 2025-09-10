#include "common.hpp"
#include "view.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace hpdexc {

// 缺失的结构体定义
struct MwuStatsOptions {
    bool calc_tie = false;
    bool ref_sorted = false;
    bool tar_sorted = false;
    uint64_t hist_max_bins = 1000000;
    uint64_t hist_mem_budget_bytes = 128 * 1024 * 1024; // 128 MB
};

struct MwuStatsOutCsc {
    std::vector<double> U1;
    std::vector<int64_t> t_indptr;
    std::vector<int64_t> t_indices;
    std::vector<double> t_data;
    std::vector<uint8_t> tie_data;
    int n1 = 0;
    int n2 = 0;
};

// 简单的排序函数实现
template<typename KeyT, typename ValueT>
void adaptive_sort_by(KeyT* keys, ValueT* values, size_t n, bool stable, int threads) {
    // 简单的快速排序实现
    if (n <= 1) return;
    
    // 创建索引数组
    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; ++i) {
        indices[i] = i;
    }
    
    // 按keys排序索引
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return keys[a] < keys[b];
    });
    
    // 重新排列keys和values
    std::vector<KeyT> temp_keys(n);
    std::vector<ValueT> temp_values(n);
    
    for (size_t i = 0; i < n; ++i) {
        temp_keys[i] = keys[indices[i]];
        temp_values[i] = values[indices[i]];
    }
    
    // 复制回原数组
    for (size_t i = 0; i < n; ++i) {
        keys[i] = temp_keys[i];
        values[i] = temp_values[i];
    }
}

// ====== f16 与 traits ======
#if defined(__FLT16_MANT_DIG__) || defined(__F16C__) || (defined(_MSC_VER) && defined(_M_FP16))
  using f16 = _Float16;
  static constexpr bool kSortHasF16 = true;
#else
  struct f16 {};
  static constexpr bool kSortHasF16 = false;
#endif

template<class T> struct mwu_type_traits {
    static constexpr bool is_float    = std::is_floating_point_v<T>;
    static constexpr bool is_signed   = std::is_signed_v<T> && !is_float;
    static constexpr bool is_unsigned = std::is_unsigned_v<T> && !is_float;
    static constexpr bool is_integer  = is_signed || is_unsigned;
};
template<class T> inline constexpr bool is_integer_v = mwu_type_traits<T>::is_integer;

// ====== 小工具 ======
static inline void count_groups(const uint8_t* ref_mask, const uint8_t* tar_mask,
                                std::size_t rows, int& n1, int& n2) {
    n1 = n2 = 0;
    for (std::size_t r=0;r<rows;++r) {
        n1 += (ref_mask && ref_mask[r]) ? 1 : 0;
        n2 += (tar_mask && tar_mask[r]) ? 1 : 0;
    }
}
template<class T>
static inline bool is_nan_T(T) { return false; }
static inline bool is_nan_T(float x)   { return std::isnan(x); }
static inline bool is_nan_T(double x)  { return std::isnan(x); }
#if defined(__FLT16_MANT_DIG__) || defined(__F16C__) || (defined(_MSC_VER) && defined(_M_FP16))
static inline bool is_nan_T(hpdexc::f16 x) { return std::isnan((float)x); }
#endif

// 键类型：除 f16 可能降级为 float 外，其它均用原生 T 作为排序键
template<class T> struct sort_key_t { using type = T; };
template<> struct sort_key_t<hpdexc::f16> { using type = std::conditional_t<kSortHasF16, hpdexc::f16, float>; };

// ====== 列内合并扫描（键为 KeyT，行号 int64_t）======
template<class KeyT>
static inline void scan_two_sorted_groups(
    const KeyT* ref_v, const int64_t* ref_r, std::size_t nref,
    const KeyT* tar_v, const int64_t* tar_r, std::size_t ntar,
    std::size_t zeros_ref, std::size_t zeros_tar,
    bool need_rank,
    double& U1_col,
    std::vector<int64_t>& out_rows,
    std::vector<double>&  out_ranks,
    std::vector<uint8_t>& out_ties
) {
    const double n1d = double(nref + zeros_ref);
    const double n2d = double(ntar + zeros_tar);
    if (n1d <= 0.0 || n2d <= 0.0) { U1_col = std::numeric_limits<double>::quiet_NaN(); return; }

    const std::size_t zeros_total = zeros_ref + zeros_tar;
    double R_tar = 0.0;
    std::size_t seen = 0;

    if (zeros_total > 0) {
        const double avg0 = (double(zeros_total) + 1.0)*0.5;
        R_tar += double(zeros_tar) * avg0;
    }

    std::size_t i=0, j=0;
    while (i<nref || j<ntar) {
        KeyT v;
        bool take_r=false, take_t=false;
        if (i<nref && j<ntar) {
            v = (ref_v[i] <= tar_v[j]) ? ref_v[i] : tar_v[j];
            take_r = (ref_v[i] == v);
            take_t = (tar_v[j] == v);
        } else if (i<nref) { v = ref_v[i]; take_r=true; }
        else               { v = tar_v[j]; take_t=true; }

        std::size_t cr=0, ct=0;
        if (take_r) { while (i<nref && !(ref_v[i] < v) && !(v < ref_v[i])) { ++cr; ++i; } }
        if (take_t) { while (j<ntar && !(tar_v[j] < v) && !(v < tar_v[j])) { ++ct; ++j; } }

        const std::size_t L = cr + ct;
        const double avg_rank = double(zeros_total) + double(seen) + (double(L)+1.0)*0.5;
        R_tar += double(ct) * avg_rank;

        if (need_rank) {
            const uint8_t tie = (L > 1) ? 1u : 0u;
            for (std::size_t k=0;k<ct;++k) {
                out_rows.push_back(tar_r[(j-ct)+k]);
                out_ranks.push_back(avg_rank);
                out_ties.push_back(tie);
            }
            for (std::size_t k=0;k<cr;++k) {
                out_rows.push_back(ref_r[(i-cr)+k]);
                out_ranks.push_back(avg_rank);
                out_ties.push_back(tie);
            }
        }
        seen += L;
    }

    const double U2 = R_tar - n2d * (n2d + 1.0)*0.5;
    U1_col = n1d * n2d - U2;
}

// ===================================================================
// 固定算子 1：Sort+Scan（适用所有 u/i/f 16/32/64）
// 将列拆为 ref/tar 两组，各自排序后两路扫描合成 U1；
// 当 calc_tie=true 时，回填每个被选中且非零/非NaN元素的平均秩与是否并列。
// ===================================================================
template<class T>
void mwu_stats_csc_sort(const CscView& A,
                        const uint8_t* ref_mask,
                        const uint8_t* tar_mask,
                        const MwuStatsOptions& opt,
                        MwuStatsOutCsc& out)
{
    using Key = typename sort_key_t<T>::type;
    out = MwuStatsOutCsc{};
    out.U1.resize((size_t)A.cols);

    count_groups(ref_mask, tar_mask, (size_t)A.rows, out.n1, out.n2);
    if (out.n1==0 || out.n2==0) {
        std::fill(out.U1.begin(), out.U1.end(), std::numeric_limits<double>::quiet_NaN());
        return;
    }

    if (opt.calc_tie) {
        out.t_indptr.assign((size_t)A.cols+1, 0);
        for (std::size_t c=0;c<(std::size_t)A.cols;++c) {
            const int64_t st=A.indptr_at(c), ed=A.indptr_at(c+1);
            const T* v = A.data_ptr<T>() + st;
            std::size_t kcol=0;
            for (int64_t p=st;p<ed;++p) {
                const int64_t r = A.row_at((size_t)p);
                if ((uint64_t)r >= (uint64_t)A.rows) continue;
                const bool a = ref_mask && ref_mask[(size_t)r];
                const bool b = tar_mask && tar_mask[(size_t)r];
                if (!(a ^ b)) continue;
                T x = v[(size_t)(p-st)];
                if constexpr (!mwu_type_traits<T>::is_float) {
                    if (x==T(0)) continue;
                } else {
                    if (is_nan_T(x) || x==T(0)) continue;
                }
                ++kcol;
            }
            out.t_indptr[c+1] = out.t_indptr[c] + (int64_t)kcol;
        }
        const std::size_t tnnz = (std::size_t)out.t_indptr.back();
        out.t_indices.resize(tnnz);
        out.t_data.resize(tnnz);
        out.tie_data.resize(tnnz);
    }

    std::vector<Key>     ref_vals, tar_vals, vals;
    std::vector<int64_t> ref_rows, tar_rows;
    std::vector<uint8_t> flags;

    std::vector<int64_t> tmp_rows;
    std::vector<double>  tmp_ranks;
    std::vector<uint8_t> tmp_ties;

    for (std::size_t c=0;c<(std::size_t)A.cols;++c) {
        const int64_t st=A.indptr_at(c), ed=A.indptr_at(c+1);
        const T* v = A.data_ptr<T>() + st;

        ref_vals.clear(); ref_rows.clear();
        tar_vals.clear(); tar_rows.clear();
        vals.clear(); flags.clear();
        tmp_rows.clear(); tmp_ranks.clear(); tmp_ties.clear();

        std::size_t nz_ref=0, nz_tar=0;

        if (!opt.calc_tie) {
            for (int64_t p=st;p<ed;++p) {
                const int64_t r = A.row_at((size_t)p);
                if ((uint64_t)r >= (uint64_t)A.rows) continue;
                const bool a = ref_mask && ref_mask[(size_t)r];
                const bool b = tar_mask && tar_mask[(size_t)r];
                if (!(a ^ b)) continue;
                T xT = v[(size_t)(p-st)];
                if constexpr (!mwu_type_traits<T>::is_float) {
                    if (xT==T(0)) continue;
                } else {
                    if (is_nan_T(xT) || xT==T(0)) continue;
                }
                Key x = (Key)xT;
                vals.push_back(x);
                flags.push_back(b ? 1u : 0u);
                if (b) ++nz_tar; else ++nz_ref;
            }

            const std::size_t zeros_ref = (std::size_t)out.n1 - nz_ref;
            const std::size_t zeros_tar = (std::size_t)out.n2 - nz_tar;
            const std::size_t zeros_total = zeros_ref + zeros_tar;

            if (!vals.empty()) {
                adaptive_sort_by<Key,uint8_t>(vals.data(), flags.data(), vals.size(), /*stable=*/false, /*threads=*/1);
            }

            double R_tar = 0.0;
            if (zeros_total>0) {
                const double avg0 = (double(zeros_total)+1.0)*0.5;
                R_tar += (double)zeros_tar * avg0;
            }
            std::size_t s=0;
            while (s<vals.size()) {
                std::size_t e=s+1;
                const Key vv = vals[s];
                while (e<vals.size() && !(vals[e] < vv) && !(vv < vals[e])) ++e;
                const std::size_t L = e-s;
                std::size_t tcnt=0;
                for (std::size_t i=s;i<e;++i) tcnt += (flags[i]!=0);
                const double avg_rank = (double)zeros_total + (double)s + (double(L)+1.0)*0.5;
                R_tar += (double)tcnt * avg_rank;
                s=e;
            }
            const double n1d=(double)out.n1, n2d=(double)out.n2;
            const double U2 = R_tar - n2d*(n2d+1.0)*0.5;
            out.U1[c] = n1d*n2d - U2;
            continue;
        }

        for (int64_t p=st;p<ed;++p) {
            const int64_t r = A.row_at((size_t)p);
            if ((uint64_t)r >= (uint64_t)A.rows) continue;
            const bool a = ref_mask && ref_mask[(size_t)r];
            const bool b = tar_mask && tar_mask[(size_t)r];
            if (!(a ^ b)) continue;
            T xT = v[(size_t)(p-st)];
            if constexpr (!mwu_type_traits<T>::is_float) {
                if (xT==T(0)) continue;
            } else {
                if (is_nan_T(xT) || xT==T(0)) continue;
            }
            Key x = (Key)xT;
            if (b) { tar_vals.push_back(x); tar_rows.push_back(r); ++nz_tar; }
            else   { ref_vals.push_back(x); ref_rows.push_back(r); ++nz_ref; }
        }

        const std::size_t zeros_ref = (std::size_t)out.n1 - nz_ref;
        const std::size_t zeros_tar = (std::size_t)out.n2 - nz_tar;

        if (!opt.tar_sorted && !tar_vals.empty()) {
            adaptive_sort_by<Key,int64_t>(tar_vals.data(), tar_rows.data(), tar_vals.size(), /*stable=*/false, /*threads=*/1);
        }
        if (!opt.ref_sorted && !ref_vals.empty()) {
            adaptive_sort_by<Key,int64_t>(ref_vals.data(), ref_rows.data(), ref_vals.size(), /*stable=*/false, /*threads=*/1);
        }

        double U1_col = std::numeric_limits<double>::quiet_NaN();
        tmp_rows.reserve((size_t)(out.t_indptr[c+1]-out.t_indptr[c]));
        tmp_ranks.reserve(tmp_rows.capacity());
        tmp_ties.reserve(tmp_rows.capacity());

        scan_two_sorted_groups<Key>(ref_vals.data(), ref_rows.data(), ref_vals.size(),
                                    tar_vals.data(), tar_rows.data(), tar_vals.size(),
                                    zeros_ref, zeros_tar,
                                    /*need_rank=*/true,
                                    U1_col,
                                    tmp_rows, tmp_ranks, tmp_ties);

        const std::size_t base = (std::size_t)out.t_indptr[c];
        std::vector<size_t> order(tmp_rows.size());
        for (size_t i=0;i<order.size();++i) order[i]=i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b){
            return tmp_rows[a] < tmp_rows[b];
        });
        for (size_t k=0;k<order.size();++k) {
            const size_t i = order[k];
            out.t_indices[base + k] = tmp_rows[i];
            out.t_data[base + k]    = tmp_ranks[i];
            out.tie_data[base + k]  = tmp_ties[i];
        }
        out.U1[c] = U1_col;
    }
}

// ----------------- 小工具：拷贝到 C 缓冲 -----------------
static inline void copy_vec_double(double* dst, const std::vector<double>& src) {
    if (!dst || src.empty()) return;
    std::memcpy(dst, src.data(), src.size() * sizeof(double));
}
static inline void copy_vec_i64(int64_t* dst, const std::vector<int64_t>& src) {
    if (!dst || src.empty()) return;
    std::memcpy(dst, src.data(), src.size() * sizeof(int64_t));
}
static inline void copy_vec_u8(uint8_t* dst, const std::vector<uint8_t>& src) {
    if (!dst || src.empty()) return;
    std::memcpy(dst, src.data(), src.size() * sizeof(uint8_t));
}

// ===================================================================
// 固定算子 1：Sort+Scan（整数专用包装）
// ===================================================================
template<class T>
static inline void run_sort_scan_int(const CscView& mat,
                              const uint8_t* ref_mask, const uint8_t* tar_mask,
                                     bool calc_tie, bool ref_sorted, bool tar_sorted,
                              double* U1,
                              int64_t* t_indptr, int64_t* t_indices,
                              double* t_data, uint8_t* tie_data,
                              int* n1_out, int* n2_out)
{
    static_assert(std::is_integral_v<T>, "run_sort_scan_int<T>: integer types only");

    MwuStatsOptions opt{};
    opt.calc_tie   = calc_tie;
    opt.ref_sorted = ref_sorted;
    opt.tar_sorted = tar_sorted;

    MwuStatsOutCsc out;
    mwu_stats_csc_sort<T>(mat, ref_mask, tar_mask, opt, out);

    copy_vec_double(U1, out.U1);
    if (n1_out) *n1_out = out.n1;
    if (n2_out) *n2_out = out.n2;

    if (calc_tie) {
        copy_vec_i64(t_indptr, out.t_indptr);
        const std::size_t tnnz = out.t_indptr.empty() ? 0u : (std::size_t)out.t_indptr.back();
        if (tnnz) {
            if (t_indices) std::memcpy(t_indices, out.t_indices.data(), tnnz * sizeof(int64_t));
            if (t_data)    std::memcpy(t_data,    out.t_data.data(),    tnnz * sizeof(double));
            if (tie_data)  std::memcpy(tie_data,  out.tie_data.data(),  tnnz * sizeof(uint8_t));
        }
    }
}

// ===================================================================
// 固定算子 1（浮点包装）：Sort+Scan（f16/f32/f64）
// ===================================================================
template<class T>
static inline void run_sort_scan_fp(const CscView& mat,
                                 const uint8_t* ref_mask, const uint8_t* tar_mask,
                                 bool calc_tie, bool ref_sorted, bool tar_sorted,
                                 double* U1,
                                 int64_t* t_indptr, int64_t* t_indices,
                                 double* t_data, uint8_t* tie_data,
                                 int* n1_out, int* n2_out)
{
    MwuStatsOptions opt{};
    opt.calc_tie   = calc_tie;
    opt.ref_sorted = ref_sorted;
    opt.tar_sorted = tar_sorted;

    MwuStatsOutCsc out;
    mwu_stats_csc_sort<T>(mat, ref_mask, tar_mask, opt, out);

    copy_vec_double(U1, out.U1);
    if (n1_out) *n1_out = out.n1;
    if (n2_out) *n2_out = out.n2;

    if (calc_tie) {
        copy_vec_i64(t_indptr, out.t_indptr);
        const std::size_t tnnz = out.t_indptr.empty() ? 0u : (std::size_t)out.t_indptr.back();
        if (tnnz) {
            if (t_indices) std::memcpy(t_indices, out.t_indices.data(), tnnz * sizeof(int64_t));
            if (t_data)    std::memcpy(t_data,    out.t_data.data(),    tnnz * sizeof(double));
            if (tie_data)  std::memcpy(tie_data,  out.tie_data.data(),  tnnz * sizeof(uint8_t));
        }
    }
}

// ===================================================================
// 固定算子 2：Histogram（仅整型；支持 bucket_size>0）
// - 三遍：统计 vmin/vmax/nz → 计桶 → 回填（可选）
// - 桶宽 = bucket_size；bin = floor((x - vmin)/bucket_size)
// - 当 R 或内存超限时返回该列 U1=NaN（固定算子不回退到排序路径）
// ===================================================================
template<class T>
static inline std::enable_if_t<std::is_integral_v<T>, void>
mwu_stats_csc_hist_stride(const CscView& A,
                          const uint8_t* ref_mask,
                          const uint8_t* tar_mask,
                          const MwuStatsOptions& opt,
                          uint64_t bucket_size,        // 桶宽（值域步长），>=1
                          MwuStatsOutCsc& out)
{
    out = MwuStatsOutCsc{};
    out.U1.resize((size_t)A.cols);

    count_groups(ref_mask, tar_mask, (size_t)A.rows, out.n1, out.n2);
    if (out.n1==0 || out.n2==0) {
        std::fill(out.U1.begin(), out.U1.end(), std::numeric_limits<double>::quiet_NaN());
        return;
    }
    if (bucket_size == 0) bucket_size = 1;

    // 需要 t/tie 时，先数每列选中且非零的元素个数，构 indptr
    if (opt.calc_tie) {
        out.t_indptr.assign((size_t)A.cols+1, 0);
        for (std::size_t c=0;c<(std::size_t)A.cols;++c) {
            const int64_t st=A.indptr_at(c), ed=A.indptr_at(c+1);
            const T* v=A.data_ptr<T>() + st;
            std::size_t kcol=0;
            for (int64_t p=st;p<ed;++p) {
                const int64_t r=A.row_at((size_t)p);
                if ((uint64_t)r >= (uint64_t)A.rows) continue;
                const bool a=ref_mask && ref_mask[(size_t)r];
                const bool b=tar_mask && tar_mask[(size_t)r];
                if (!(a ^ b)) continue;
                const T x=v[(size_t)(p-st)];
                if (x==T(0)) continue; // 0 由 zero-run 处理
                ++kcol;
            }
            out.t_indptr[c+1] = out.t_indptr[c] + (int64_t)kcol;
        }
        const std::size_t tnnz=(std::size_t)out.t_indptr.back();
        out.t_indices.resize(tnnz);
        out.t_data.resize(tnnz);
        out.tie_data.resize(tnnz);
    }

    // 工作区（每列复用）
    std::vector<uint32_t> hist_cnt; // 每桶元素数
    std::vector<uint32_t> hist_tar; // 每桶 tar 元素数

    for (std::size_t c=0;c<(std::size_t)A.cols;++c) {
        const int64_t st=A.indptr_at(c), ed=A.indptr_at(c+1);
        const T* v=A.data_ptr<T>() + st;

        // 第一遍：统计 vmin/vmax/nz
        bool has_val=false;
        T vmin=std::numeric_limits<T>::max();
        T vmax=std::numeric_limits<T>::min();
        std::size_t nz_ref=0, nz_tar=0;

        for (int64_t p=st;p<ed;++p) {
            const int64_t r=A.row_at((size_t)p);
            if ((uint64_t)r >= (uint64_t)A.rows) continue;
            const bool a=ref_mask && ref_mask[(size_t)r];
            const bool b=tar_mask && tar_mask[(size_t)r];
            if (!(a ^ b)) continue;
            const T x=v[(size_t)(p-st)];
            if (x==T(0)) continue;
            has_val=true;
            if (x<vmin) vmin=x;
            if (x>vmax) vmax=x;
            if (b) ++nz_tar; else ++nz_ref;
        }

        const std::size_t zeros_ref = (std::size_t)out.n1 - nz_ref;
        const std::size_t zeros_tar = (std::size_t)out.n2 - nz_tar;
        const std::size_t zeros_total = zeros_ref + zeros_tar;

        if (!has_val) {
            // 仅零值 run
            double R_tar = 0.0;
            if (zeros_total>0) {
                const double avg0=(double(zeros_total)+1.0)*0.5;
                R_tar += (double)zeros_tar * avg0;
            }
            const double n1d=(double)out.n1, n2d=(double)out.n2;
            const double U2=R_tar - n2d*(n2d+1.0)*0.5;
            out.U1[c]=n1d*n2d - U2;
            continue;
        }

        // 桶数 R_bins
        uint64_t R_bins = 0;
#if defined(__SIZEOF_INT128__)
            if constexpr (std::is_signed_v<T>) {
                __int128 diff = ( (__int128)vmax - (__int128)vmin );
                if (diff < 0) diff = 0;
                __int128 bins = diff / (__int128)bucket_size + 1;
                if (bins > (__int128)std::numeric_limits<uint64_t>::max()) {
                    out.U1[c] = std::numeric_limits<double>::quiet_NaN();
                    continue;
                }
                R_bins = (uint64_t)bins;
            } else {
            __uint128_t diff = ( (__uint128_t)vmax - (__uint128_t)vmin );
                __uint128_t bins = diff / (__uint128_t)bucket_size + 1;
                if (bins > (__uint128_t)std::numeric_limits<uint64_t>::max()) {
                    out.U1[c] = std::numeric_limits<double>::quiet_NaN();
                    continue;
                }
                R_bins = (uint64_t)bins;
            }
#else
        {
            long double diff = (long double)((long double)vmax - (long double)vmin);
            if (diff < 0) diff = 0;
            long double bins = diff / (long double)bucket_size + 1.0L;
            if (bins > (long double)std::numeric_limits<uint64_t>::max()) {
                out.U1[c] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            R_bins = (uint64_t)bins;
        }
#endif

        if (R_bins == 0 || R_bins > opt.hist_max_bins) {
            out.U1[c] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }

        // 预算检查
#if defined(__SIZEOF_INT128__)
        __uint128_t need = (__uint128_t)R_bins * (2u * (unsigned)sizeof(uint32_t));
        if (opt.calc_tie) {
            need += (__uint128_t)R_bins * ((unsigned)sizeof(double) + (unsigned)sizeof(uint8_t));
        }
        if (need > (__uint128_t)opt.hist_mem_budget_bytes) {
            out.U1[c] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
#else
        long double need = (long double)R_bins * (2.0L * (long double)sizeof(uint32_t));
        if (opt.calc_tie) {
            need += (long double)R_bins * ((long double)sizeof(double) + (long double)sizeof(uint8_t));
        }
        if (need > (long double)opt.hist_mem_budget_bytes) {
            out.U1[c] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
#endif

        // 第二遍：计桶
        hist_cnt.assign((size_t)R_bins, 0u);
        hist_tar.assign((size_t)R_bins, 0u);

        for (int64_t p=st;p<ed;++p) {
            const int64_t r=A.row_at((size_t)p);
            if ((uint64_t)r >= (uint64_t)A.rows) continue;
            const bool a=ref_mask && ref_mask[(size_t)r];
            const bool b=tar_mask && tar_mask[(size_t)r];
            if (!(a ^ b)) continue;
            const T x=v[(size_t)(p-st)];
            if (x==T(0)) continue;

            uint64_t bkt;
#if defined(__SIZEOF_INT128__)
            if constexpr (std::is_signed_v<T>) {
                __int128 diff = (__int128)x - (__int128)vmin;
                if (diff < 0) diff = 0;
                bkt = (uint64_t)((__int128)diff / (__int128)bucket_size);
            } else {
                __uint128_t diff = (__uint128_t)x - (__uint128_t)vmin;
                bkt = (uint64_t)( diff / (__uint128_t)bucket_size );
            }
#else
            long double diff = (long double)x - (long double)vmin;
            if (diff < 0) diff = 0;
            bkt = (uint64_t)( diff / (long double)bucket_size );
#endif
            if (bkt >= R_bins) bkt = R_bins - 1;
            hist_cnt[(size_t)bkt] += 1u;
            hist_tar[(size_t)bkt] += (b ? 1u : 0u);
        }

        // 合成 R_tar / U1
        double R_tar = 0.0;
        if (zeros_total>0) {
            const double avg0=(double(zeros_total)+1.0)*0.5;
            R_tar += (double)zeros_tar * avg0;
        }
        std::size_t seen=0;
        for (size_t bkt=0;bkt<R_bins;++bkt) {
            const uint32_t L = hist_cnt[bkt];
            if (!L) continue;
            const uint32_t t = hist_tar[bkt];
            const double avg_rank = (double)zeros_total + (double)seen + (double(L)+1.0)*0.5;
            R_tar += (double)t * avg_rank;
            seen += L;
        }
        {
            const double n1d=(double)out.n1, n2d=(double)out.n2;
            const double U2=R_tar - n2d*(n2d+1.0)*0.5;
            out.U1[c]=n1d*n2d - U2;
        }

        // 第三遍：回填 t/tie（仅当需要），并按 row 升序
        if (opt.calc_tie) {
            // 每桶 avg_rank 与 tie 标记
            std::vector<double>  bucket_avg(R_bins, 0.0);
            std::vector<uint8_t> bucket_tie(R_bins, 0u);
            seen = 0;
            for (size_t bkt=0;bkt<R_bins;++bkt) {
                const uint32_t L = hist_cnt[bkt];
                if (!L) continue;
                const double avg_rank = (double)zeros_total + (double)seen + (double(L)+1.0)*0.5;
                bucket_avg[bkt] = avg_rank;
                bucket_tie[bkt] = (L>1) ? 1u : 0u;
                seen += L;
            }

            const std::size_t base = (std::size_t)out.t_indptr[c];
            size_t cursor = 0;
            for (int64_t p=st;p<ed;++p) {
                const int64_t r=A.row_at((size_t)p);
                if ((uint64_t)r >= (uint64_t)A.rows) continue;
                const bool a=ref_mask && ref_mask[(size_t)r];
                const bool b=tar_mask && tar_mask[(size_t)r];
                if (!(a ^ b)) continue;
                const T x=v[(size_t)(p-st)];
                if (x==T(0)) continue;

                uint64_t bkt;
#if defined(__SIZEOF_INT128__)
                if constexpr (std::is_signed_v<T>) {
                    __int128 diff = (__int128)x - (__int128)vmin;
                    if (diff < 0) diff = 0;
                    bkt = (uint64_t)((__int128)diff / (__int128)bucket_size);
                } else {
                    __uint128_t diff = (__uint128_t)x - (__uint128_t)vmin;
                    bkt = (uint64_t)( diff / (__uint128_t)bucket_size );
                }
#else
                long double diff = (long double)x - (long double)vmin;
                if (diff < 0) diff = 0;
                bkt = (uint64_t)( diff / (long double)bucket_size );
#endif
                if (bkt >= R_bins) bkt = R_bins - 1;
                out.t_indices[base + cursor] = r;
                out.t_data[base + cursor]    = bucket_avg[(size_t)bkt];
                out.tie_data[base + cursor]  = bucket_tie[(size_t)bkt];
                ++cursor;
            }
            // —— 行号排序，保证 CSC 规范（indices 升序）
            struct Node { int64_t row; double rank; uint8_t tie; };
            std::vector<Node> nodes(cursor);
            for (size_t k=0;k<cursor;++k) {
                nodes[k] = { out.t_indices[base+k], out.t_data[base+k], out.tie_data[base+k] };
            }
            std::sort(nodes.begin(), nodes.end(), [](const Node& a, const Node& b){ return a.row < b.row; });
            for (size_t k=0;k<cursor;++k) {
                out.t_indices[base+k] = nodes[k].row;
                out.t_data[base+k]    = nodes[k].rank;
                out.tie_data[base+k]  = nodes[k].tie;
            }
        }
    }
}

} // namespace hpdexc


// =================== C ABI：Sort+Scan（整数） ===================
extern "C" {

#define HPDEXC_SORT_SCAN_INT_C_API(NAME, TCpp) \
void NAME(const hpdexc::CscView& mat, \
          const uint8_t* ref_mask, const uint8_t* tar_mask, \
          bool calc_tie, bool ref_sorted, bool tar_sorted, \
          double* U1, \
          int64_t* t_indptr, int64_t* t_indices, double* t_data, uint8_t* tie_data, \
          int* n1_out, int* n2_out) { \
    hpdexc::run_sort_scan_int<TCpp>(mat, ref_mask, tar_mask, \
                                    calc_tie, ref_sorted, tar_sorted, \
                                    U1, t_indptr, t_indices, t_data, tie_data, \
                                    n1_out, n2_out); \
}

HPDEXC_SORT_SCAN_INT_C_API(hpdexc_mwu_stats_i16_csc,  int16_t)
HPDEXC_SORT_SCAN_INT_C_API(hpdexc_mwu_stats_i32_csc,  int32_t)
HPDEXC_SORT_SCAN_INT_C_API(hpdexc_mwu_stats_i64_csc,  int64_t)
HPDEXC_SORT_SCAN_INT_C_API(hpdexc_mwu_stats_u16_csc,  uint16_t)
HPDEXC_SORT_SCAN_INT_C_API(hpdexc_mwu_stats_u32_csc,  uint32_t)
HPDEXC_SORT_SCAN_INT_C_API(hpdexc_mwu_stats_u64_csc,  uint64_t)

#undef HPDEXC_SORT_SCAN_INT_C_API
} // extern "C"


// =================== C ABI：Sort+Scan（浮点） ===================
extern "C" {

#define HPDEXC_SORT_SCAN_FP_C_API(NAME, TCpp) \
void NAME(const hpdexc::CscView& mat, \
          const uint8_t* ref_mask, const uint8_t* tar_mask, \
          bool calc_tie, bool ref_sorted, bool tar_sorted, \
          double* U1, \
          int64_t* t_indptr, int64_t* t_indices, double* t_data, uint8_t* tie_data, \
          int* n1_out, int* n2_out) { \
    hpdexc::run_sort_scan_fp<TCpp>(mat, ref_mask, tar_mask, \
                                calc_tie, ref_sorted, tar_sorted, \
                                U1, t_indptr, t_indices, t_data, tie_data, \
                                n1_out, n2_out); \
}

HPDEXC_SORT_SCAN_FP_C_API(hpdexc_mwu_stats_f32_csc,  float)
HPDEXC_SORT_SCAN_FP_C_API(hpdexc_mwu_stats_f64_csc,  double)

#undef HPDEXC_SORT_SCAN_FP_C_API

} // extern "C"


// =================== C ABI：Sort+Scan（f16，带回退） ===================
extern "C" {

#if defined(__FLT16_MANT_DIG__) || defined(__F16C__) || (defined(_MSC_VER) && defined(_M_FP16))
void hpdexc_mwu_stats_f16_csc(const hpdexc::CscView& mat,
                              const uint8_t* ref_mask, const uint8_t* tar_mask,
                              bool calc_tie, bool ref_sorted, bool tar_sorted,
                              double* U1,
                              int64_t* t_indptr, int64_t* t_indices,
                              double* t_data, uint8_t* tie_data,
                              int* n1_out, int* n2_out)
{
    hpdexc::run_sort_scan_fp<hpdexc::f16>(mat, ref_mask, tar_mask,
                                          calc_tie, ref_sorted, tar_sorted,
                                          U1, t_indptr, t_indices, t_data, tie_data,
                                          n1_out, n2_out);
}
#else
void hpdexc_mwu_stats_f16_csc(const hpdexc::CscView& mat,
                              const uint8_t* ref_mask, const uint8_t* tar_mask,
                              bool calc_tie, bool ref_sorted, bool tar_sorted,
                              double* U1,
                              int64_t* t_indptr, int64_t* t_indices,
                              double* t_data, uint8_t* tie_data,
                              int* n1_out, int* n2_out)
{
    (void)ref_mask; (void)tar_mask; (void)calc_tie; (void)ref_sorted; (void)tar_sorted;
    (void)t_indptr; (void)t_indices; (void)t_data; (void)tie_data;
    if (U1) {
        const std::size_t cols = (std::size_t)mat.cols;
        for (std::size_t i=0;i<cols;++i) U1[i] = std::numeric_limits<double>::quiet_NaN();
    }
    if (n1_out) *n1_out = 0;
    if (n2_out) *n2_out = 0;
}
#endif

} // extern "C"


// =================== C ABI：Histogram（整数） ===================
// 说明：接受 bucket_size/max_bins/mem_budget_bytes；不依赖 sorted 参数。
extern "C" {

#define HPDEXC_HIST_C_API(NAME, TCpp) \
void NAME(const hpdexc::CscView& mat, \
          const uint8_t* ref_mask, const uint8_t* tar_mask, \
          bool calc_tie, \
          uint64_t bucket_size, uint64_t max_bins, uint64_t mem_budget_bytes, \
          double* U1, \
          int64_t* t_indptr, int64_t* t_indices, double* t_data, uint8_t* tie_data, \
          int* n1_out, int* n2_out) { \
    hpdexc::MwuStatsOptions opt{}; \
    opt.calc_tie = calc_tie; \
    opt.hist_max_bins = max_bins; \
    opt.hist_mem_budget_bytes = mem_budget_bytes; \
    hpdexc::MwuStatsOutCsc out; \
    hpdexc::mwu_stats_csc_hist_stride<TCpp>(mat, ref_mask, tar_mask, opt, bucket_size, out); \
    if (U1) std::memcpy(U1, out.U1.data(), out.U1.size()*sizeof(double)); \
    if (n1_out) *n1_out = out.n1; \
    if (n2_out) *n2_out = out.n2; \
    if (calc_tie) { \
        if (t_indptr) std::memcpy(t_indptr, out.t_indptr.data(), out.t_indptr.size()*sizeof(int64_t)); \
        const std::size_t tnnz = out.t_indptr.empty()?0u:(std::size_t)out.t_indptr.back(); \
        if (tnnz) { \
            if (t_indices) std::memcpy(t_indices, out.t_indices.data(), tnnz*sizeof(int64_t)); \
            if (t_data)    std::memcpy(t_data,    out.t_data.data(),   tnnz*sizeof(double)); \
            if (tie_data)  std::memcpy(tie_data,  out.tie_data.data(), tnnz*sizeof(uint8_t)); \
        } \
    } \
}

HPDEXC_HIST_C_API(hpdexc_mwu_stats_i16_csc_hist,  int16_t)
HPDEXC_HIST_C_API(hpdexc_mwu_stats_i32_csc_hist,  int32_t)
HPDEXC_HIST_C_API(hpdexc_mwu_stats_i64_csc_hist,  int64_t)
HPDEXC_HIST_C_API(hpdexc_mwu_stats_u16_csc_hist,  uint16_t)
HPDEXC_HIST_C_API(hpdexc_mwu_stats_u32_csc_hist,  uint32_t)
HPDEXC_HIST_C_API(hpdexc_mwu_stats_u64_csc_hist,  uint64_t)

#undef HPDEXC_HIST_C_API

} // extern "C"
