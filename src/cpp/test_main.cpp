#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <random>
#include <stdexcept>
#include <iostream>
#include <chrono>

// 引入被测实现
#include "rank_sum_test.cpp"

template<class T>
struct NaiveUTieResult {
    long double U_ref;               // ref(组0) 的 U
    std::vector<uint8_t> is_tie;     // 与排序后序列等长，处于并列块则为1
    long double tie_sum;             // 校验用：sum_over_ties (t^3 - t)
    std::size_t n_ref;               // 补齐后的 ref 个数
    std::size_t n_tar;               // 补齐后的 tar 个数
};

// 朴素一致性校验实现：
// - 若 use_sparse=true：将 ref/tar 各自补齐到长度 col，填充值为 sparse_value，然后再排序。
// - 排序键：先按 value，再按 group（0=ref, 1=tar），以保证稳定可重复。
// - 扫描时按“块”处理：块内所有值相等。用 r=块内ref数, t=块内tar数。
//   U_ref 累加：U += r * tar_seen + r * t * 0.5    （等价平均秩法）
//   其中 tar_seen 为严格小于当前块值的 tar 总数。
template<class T>
NaiveUTieResult<T> naive_u_and_tie(
    const std::vector<T>& ref_in,
    const std::vector<T>& tar_in,
    std::size_t col,          // 仅在 use_sparse=true 时使用：每组补齐到 col
    bool use_sparse,
    T sparse_value = T{}      // 稀疏填充值（如 0）
){
    static_assert(std::is_arithmetic<T>::value, "T must be numeric");

    // 1) 复制并按需补齐
    std::vector<T> ref = ref_in, tar = tar_in;
    if (use_sparse) {
        if (ref.size() < col) ref.insert(ref.end(), col - ref.size(), sparse_value);
        if (tar.size() < col) tar.insert(tar.end(), col - tar.size(), sparse_value);
        if (ref.size() > col) ref.resize(col);   // 防御式：超过则截断
        if (tar.size() > col) tar.resize(col);
    }
    const std::size_t n_ref = ref.size();
    const std::size_t n_tar = tar.size();

    // 2) 合并 + sort_by(组掩码)
    struct Item {
        T val;
        uint8_t grp; // 0=ref, 1=tar
    };
    std::vector<Item> a;
    a.reserve(n_ref + n_tar);
    for (const auto& v : ref) a.push_back({v, 0});
    for (const auto& v : tar) a.push_back({v, 1});

    // 按 (value, group) 排序，保证稳定与可重复；组次序仅为确定性，不影响 U 的正确性
    std::stable_sort(a.begin(), a.end(),
                     [](const Item& x, const Item& y){
                         if (x.val < y.val) return true;
                         if (y.val < x.val) return false;
                         return x.grp < y.grp; // value 相等时，ref 在前
                     });

    // 3) 一次扫描得到 U_ref 与 is_tie、tie_sum
    std::vector<uint8_t> is_tie(a.size(), 0);
    long double U = 0.0L;
    long double tie_sum = 0.0L;
    std::size_t tar_seen = 0;

    for (std::size_t i = 0; i < a.size(); ) {
        std::size_t j = i + 1;
        // 找到相等值的块 [i, j)
        while (j < a.size() && !(a[i].val < a[j].val) && !(a[j].val < a[i].val)) ++j;

        // 统计本块两组计数
        std::size_t r = 0, t = 0;
        for (std::size_t k = i; k < j; ++k) {
            if (a[k].grp == 0) ++r; else ++t;
        }

        // 标记并列
        const std::size_t block_len = j - i;
        if (block_len > 1) {
            for (std::size_t k = i; k < j; ++k) is_tie[k] = 1;
            const long double tt = static_cast<long double>(block_len);
            tie_sum += tt*tt*tt - tt;
        }

        // 累加 U_ref（ref 与更小 tar 的配对 + 与相等 tar 的半配对）
        if (r) {
            U += static_cast<long double>(r) * static_cast<long double>(tar_seen)
               + static_cast<long double>(r) * static_cast<long double>(t) * 0.5L;
        }

        // 更新 tar_seen，进入下块
        tar_seen += t;
        i = j;
    }

    return NaiveUTieResult<T>{
        /*U_ref*/  U,
        /*is_tie*/ std::move(is_tie),
        /*tie_sum*/tie_sum,
        /*n_ref*/  n_ref,
        /*n_tar*/  n_tar
    };
}

// 稀疏优化版朴素实现：默认稀疏值=0 且为最小值，避免显式补零
template<class T>
NaiveUTieResult<T> naive_u_sparse_zero_min_optimized(
    std::vector<T> ref_nz,           // 非零（可未排序）
    std::vector<T> tar_nz,           // 非零（可未排序）
    std::size_t col                  // 每组长度（含零）
){
    static_assert(std::is_arithmetic<T>::value, "T must be numeric");
    const std::size_t nnz_ref = ref_nz.size();
    const std::size_t nnz_tar = tar_nz.size();
    if (nnz_ref > col || nnz_tar > col) throw std::runtime_error("nnz > col");
    const std::size_t sp_ref = col - nnz_ref;
    const std::size_t sp_tar = col - nnz_tar;
    std::sort(ref_nz.begin(), ref_nz.end());
    std::sort(tar_nz.begin(), tar_nz.end());

    long double U = 0.0L;
    long double tie_sum = 0.0L;
    std::size_t tar_seen = 0;

    // 零块（最前端）
    if (sp_ref > 0 || sp_tar > 0) {
        const std::size_t block_len = sp_ref + sp_tar;
        if (block_len > 1) { const long double tt = (long double)block_len; tie_sum += tt*tt*tt - tt; }
        if (sp_ref > 0) {
            U += (long double)sp_ref * (long double)tar_seen
               + (long double)sp_ref * (long double)sp_tar * 0.5L;
        }
        tar_seen += sp_tar;
    }

    // 归并非零
    std::size_t i = 0, j = 0;
    while (i < nnz_ref || j < nnz_tar) {
        T v; bool from_ref = false, from_tar = false;
        if (i < nnz_ref && j < nnz_tar) {
            if (ref_nz[i] < tar_nz[j]) { v = ref_nz[i]; from_ref = true; }
            else if (tar_nz[j] < ref_nz[i]) { v = tar_nz[j]; from_tar = true; }
            else { v = ref_nz[i]; from_ref = from_tar = true; }
        } else if (i < nnz_ref) { v = ref_nz[i]; from_ref = true; }
        else { v = tar_nz[j]; from_tar = true; }

        std::size_t r = 0, t = 0;
        if (from_ref) { do { ++r; ++i; } while (i < nnz_ref && !(ref_nz[i] < v) && !(v < ref_nz[i])); }
        if (from_tar) { do { ++t; ++j; } while (j < nnz_tar && !(tar_nz[j] < v) && !(v < tar_nz[j])); }

        const std::size_t block_len = r + t;
        if (block_len > 1) { const long double tt = (long double)block_len; tie_sum += tt*tt*tt - tt; }
        if (r > 0) {
            U += (long double)r * (long double)tar_seen
               + (long double)r * (long double)t * 0.5L;
        }
        tar_seen += t;
    }

    return NaiveUTieResult<T>{ U, /*is_tie*/{}, tie_sum, col, col };
}

// 稀疏优化版朴素实现（通用稀疏值 + min/max/none 行为，对齐算子接口）
template<class T>
NaiveUTieResult<T> naive_u_sparse_value_optimized(
    std::vector<T> ref_nz,
    std::vector<T> tar_nz,
    std::size_t col,
    T sparse_value,
    typename hpdexc::MannWhitneyOption<T>::SparseValueMinmax minmax
){
    static_assert(std::is_arithmetic<T>::value, "T must be numeric");
    const std::size_t nnz_ref = ref_nz.size();
    const std::size_t nnz_tar = tar_nz.size();
    if (nnz_ref > col || nnz_tar > col) throw std::runtime_error("nnz > col");
    const std::size_t sp_ref = col - nnz_ref;
    const std::size_t sp_tar = col - nnz_tar;
    std::sort(ref_nz.begin(), ref_nz.end());
    std::sort(tar_nz.begin(), tar_nz.end());

    auto merge_blocks = [&](const T* a, std::size_t na, const T* b, std::size_t nb,
                            long double& U, long double& tie_sum, std::size_t& tar_seen){
        std::size_t i=0, j=0;
        while (i<na || j<nb) {
            T v; bool from_a=false, from_b=false;
            if (i<na && j<nb) {
                if (a[i] < b[j]) { v=a[i]; from_a=true; }
                else if (b[j] < a[i]) { v=b[j]; from_b=true; }
                else { v=a[i]; from_a=from_b=true; }
            } else if (i<na) { v=a[i]; from_a=true; }
            else { v=b[j]; from_b=true; }
            std::size_t r=0, t=0;
            if (from_a) { do { ++r; ++i; } while (i<na && !(a[i] < v) && !(v < a[i])); }
            if (from_b) { do { ++t; ++j; } while (j<nb && !(b[j] < v) && !(v < b[j])); }
            const std::size_t bl = r + t;
            if (bl > 1) { const long double tt=(long double)bl; tie_sum += tt*tt*tt - tt; }
            if (r > 0) {
                U += (long double)r * (long double)tar_seen
                   + (long double)r * (long double)t * 0.5L;
            }
            tar_seen += t;
        }
    };

    long double U = 0.0L;
    long double tie_sum = 0.0L;
    std::size_t tar_seen = 0;

    if (minmax == hpdexc::MannWhitneyOption<T>::SparseValueMinmax::min) {
        // 稀疏块在最前
        if (sp_ref > 0 || sp_tar > 0) {
            const std::size_t bl = sp_ref + sp_tar;
            if (bl > 1) { const long double tt=(long double)bl; tie_sum += tt*tt*tt - tt; }
            if (sp_ref > 0) {
                U += (long double)sp_ref * (long double)tar_seen
                   + (long double)sp_ref * (long double)sp_tar * 0.5L;
            }
            tar_seen += sp_tar;
        }
        merge_blocks(ref_nz.data(), nnz_ref, tar_nz.data(), nnz_tar, U, tie_sum, tar_seen);
    } else if (minmax == hpdexc::MannWhitneyOption<T>::SparseValueMinmax::max) {
        // 稀疏块在最后
        merge_blocks(ref_nz.data(), nnz_ref, tar_nz.data(), nnz_tar, U, tie_sum, tar_seen);
        if (sp_ref > 0 || sp_tar > 0) {
            const std::size_t bl = sp_ref + sp_tar;
            if (bl > 1) { const long double tt=(long double)bl; tie_sum += tt*tt*tt - tt; }
            if (sp_ref > 0) {
                // 稀疏块位于末尾，此时 tar_seen 已统计所有非零 tar
                U += (long double)sp_ref * (long double)tar_seen
                   + (long double)sp_ref * (long double)sp_tar * 0.5L;
            }
            tar_seen += sp_tar;
        }
    } else { // none：稀疏值在中间
        // 按 sparse_value 拆分非零：<sv 与 >sv
        auto split_less = [&](const std::vector<T>& v){
            return (std::size_t)(std::lower_bound(v.begin(), v.end(), sparse_value,
                [](const T& x, const T& y){ return x < y; }) - v.begin());
        };
        const std::size_t ar = split_less(ref_nz);
        const std::size_t br = split_less(tar_nz);
        // 先处理 < sv
        merge_blocks(ref_nz.data(), ar, tar_nz.data(), br, U, tie_sum, tar_seen);
        // 处理中间稀疏块
        if (sp_ref > 0 || sp_tar > 0) {
            const std::size_t bl = sp_ref + sp_tar;
            if (bl > 1) { const long double tt=(long double)bl; tie_sum += tt*tt*tt - tt; }
            if (sp_ref > 0) {
                U += (long double)sp_ref * (long double)tar_seen
                   + (long double)sp_ref * (long double)sp_tar * 0.5L;
            }
            tar_seen += sp_tar;
        }
        // 处理 > sv
        merge_blocks(ref_nz.data()+ar, nnz_ref-ar, tar_nz.data()+br, nnz_tar-br, U, tie_sum, tar_seen);
    }

    return NaiveUTieResult<T>{ U, /*is_tie*/{}, tie_sum, col, col };
}

// 简单一致性测试：随机或固定用例对比 rank_sum_nonsparse 与 naive 实现
static void test_consistency_basic() {
    {
        // 用例1：包含单个 tar 等于 ref 的场景（回归验证）
        std::vector<double> ref{1.0, 2.0, 2.0, 4.0};
        std::vector<double> tar{0.5, 2.0, 3.0};

        auto naive = naive_u_and_tie(ref, tar, 0, false, 0.0);
        auto fast = rank_sum_nonsparse<double>(ref, tar, false, false);

        if (std::llround(fast.U_ref * 1e12) != std::llround(naive.U_ref * 1e12)) {
            throw std::runtime_error("U_ref mismatch in basic case");
        }
        if (std::llround(fast.tie_sum) != std::llround(naive.tie_sum)) {
            throw std::runtime_error("tie_sum mismatch in basic case");
        }
    }

    {
        // 用例2：无并列
        std::vector<int> ref{1,3,5};
        std::vector<int> tar{2,4,6};
        auto naive = naive_u_and_tie(ref, tar, 0, false, 0);
        auto fast = rank_sum_nonsparse<int>(ref, tar, false, false);
        if (fast.U_ref != naive.U_ref) throw std::runtime_error("U_ref mismatch (no tie)");
        if (fast.tie_sum != naive.tie_sum) throw std::runtime_error("tie_sum mismatch (no tie)");
    }
}

// ---------------- 数据生成器 ----------------
template<class T>
struct DataGenSpec {
    // 规模与稀疏
    std::size_t col = 0;        // use_sparse=true 时有效
    std::size_t n_ref = 0;      // 非稀疏使用；若 use_sparse=true 则代表 ref 的 nnz
    std::size_t n_tar = 0;      // 非稀疏使用；若 use_sparse=true 则代表 tar 的 nnz
    bool use_sparse = false;
    T sparse_value = T{};       // 稀疏填充值

    // 值域与并列控制
    T min_val = T(0);
    T max_val = T(1);
    bool integer_values = false;  // 若为 true，从整数区间采样，提升并列概率
    int levels = 0;               // >0 时强制将值量化到该等级数，制造并列

    // 随机性
    unsigned seed = 42;
};

template<class T>
static void generate_data(const DataGenSpec<T>& spec,
                          std::vector<T>& ref,
                          std::vector<T>& tar) {
    std::mt19937 rng(spec.seed);

    auto gen_value = [&](std::size_t count) {
        std::vector<T> v;
        v.reserve(count);
        if (spec.integer_values) {
            long long lo = static_cast<long long>(spec.min_val);
            long long hi = static_cast<long long>(spec.max_val);
            if (hi < lo) std::swap(lo, hi);
            std::uniform_int_distribution<long long> dist(lo, hi);
            for (std::size_t i = 0; i < count; ++i) v.push_back(static_cast<T>(dist(rng)));
        } else {
            std::uniform_real_distribution<double> dist(static_cast<double>(spec.min_val), static_cast<double>(spec.max_val));
            for (std::size_t i = 0; i < count; ++i) v.push_back(static_cast<T>(dist(rng)));
        }
        if (spec.levels > 0) {
            // 量化到指定等级
            const double L = static_cast<double>(spec.levels);
            const double a = static_cast<double>(spec.min_val);
            const double b = static_cast<double>(spec.max_val);
            const double w = (b - a) / L;
            for (auto& x : v) {
                double dx = static_cast<double>(x);
                if (w > 0) {
                    double k = std::floor((dx - a) / w + 0.5);
                    if (k < 0) k = 0; if (k > L) k = L;
                    dx = a + k * w;
                }
                x = static_cast<T>(dx);
            }
        }
        return v;
    };

    if (!spec.use_sparse) {
        ref = gen_value(spec.n_ref);
        tar = gen_value(spec.n_tar);
    } else {
        // 稀疏：只生成 nnz，pad 在使用时完成
        ref = gen_value(spec.n_ref);
        tar = gen_value(spec.n_tar);
    }
}

template<class T>
static void run_param_case(const DataGenSpec<T>& spec) {
    std::vector<T> ref, tar;
    generate_data(spec, ref, tar);

    if (!spec.use_sparse) {
        // 非稀疏：直接对比
        auto naive = naive_u_and_tie(ref, tar, 0, false, T{});
        auto fast  = rank_sum_nonsparse<T>(ref, tar, false, false);

        if (std::llround(fast.U_ref * 1e12) != std::llround(naive.U_ref * 1e12))
            throw std::runtime_error("U_ref mismatch (dense)");
        if (std::llround(fast.tie_sum) != std::llround(naive.tie_sum))
            throw std::runtime_error("tie_sum mismatch (dense)");
    } else {
        // 稀疏：pad 到 col 再比较
        if (spec.col == 0) throw std::runtime_error("spec.col must be > 0 for sparse case");
        std::vector<T> ref_pad = ref;
        std::vector<T> tar_pad = tar;
        if (ref_pad.size() < spec.col) ref_pad.insert(ref_pad.end(), spec.col - ref_pad.size(), spec.sparse_value);
        if (tar_pad.size() < spec.col) tar_pad.insert(tar_pad.end(), spec.col - tar_pad.size(), spec.sparse_value);
        if (ref_pad.size() > spec.col) ref_pad.resize(spec.col);
        if (tar_pad.size() > spec.col) tar_pad.resize(spec.col);

        auto naive = naive_u_and_tie(ref, tar, spec.col, true, spec.sparse_value);
        auto fast  = rank_sum_nonsparse<T>(ref_pad, tar_pad, false, false);

        if (std::llround(fast.U_ref * 1e12) != std::llround(naive.U_ref * 1e12))
            throw std::runtime_error("U_ref mismatch (sparse)");
        if (std::llround(fast.tie_sum) != std::llround(naive.tie_sum))
            throw std::runtime_error("tie_sum mismatch (sparse)");
    }
}

// 随机扫测：在多组参数上循环测试若干次
template<class T>
static void run_random_sweep(unsigned base_seed,
                             int repeats,
                             bool use_sparse,
                             T min_val,
                             T max_val,
                             bool integer_values,
                             int levels,
                             std::size_t n_ref,
                             std::size_t n_tar,
                             std::size_t col = 0,
                             T sparse_value = T{}) {
    for (int k = 0; k < repeats; ++k) {
        DataGenSpec<T> s;
        s.seed = base_seed + k;
        s.use_sparse = use_sparse;
        s.min_val = min_val;
        s.max_val = max_val;
        s.integer_values = integer_values;
        s.levels = levels;
        s.n_ref = n_ref;
        s.n_tar = n_tar;
        s.col = col;
        s.sparse_value = sparse_value;
        run_param_case<T>(s);
    }
    std::cout << "[PASS] sweep type=" << (std::is_integral<T>::value?"int":"float")
              << " sparse=" << (use_sparse?1:0)
              << " reps=" << repeats << "\n";
}

int main() {
    test_consistency_basic();

    // 参数化生成若干组用例
    {
        DataGenSpec<double> s; s.seed=123; s.n_ref=100; s.n_tar=120; s.use_sparse=false; s.min_val=0.0; s.max_val=10.0; s.integer_values=false; s.levels=0;
        run_param_case(s);
    }
    {
        DataGenSpec<double> s; s.seed=456; s.n_ref=80; s.n_tar=80; s.use_sparse=false; s.min_val=0.0; s.max_val=1.0; s.integer_values=false; s.levels=8; // 制造并列
        run_param_case(s);
    }
    {
        DataGenSpec<int> s; s.seed=789; s.n_ref=60; s.n_tar=95; s.use_sparse=false; s.min_val=0; s.max_val=5; s.integer_values=true; // 高并列
        run_param_case(s);
    }
    {
        // 小范围大量正数（整数）：大量并列，用于压力测试 tie
        DataGenSpec<int> s; s.seed=20240910; s.n_ref=5000; s.n_tar=6000; s.use_sparse=false; s.min_val=0; s.max_val=10; s.integer_values=true; // 仅4个取值
        run_param_case(s);
    }
    {
        // 小范围大量正数（浮点量化）：量化到4级，制造大量并列
        DataGenSpec<double> s; s.seed=20240911; s.n_ref=4000; s.n_tar=4000; s.use_sparse=false; s.min_val=0.0; s.max_val=1.0; s.integer_values=false; s.levels=4;
        run_param_case(s);
    }
    {
        // 稀疏：零值为稀疏填充值
        DataGenSpec<double> s; s.seed=321; s.use_sparse=true; s.col=200; s.n_ref=40; s.n_tar=50; s.sparse_value=0.0; s.min_val=0.1; s.max_val=3.0; s.integer_values=false; s.levels=0;
        run_param_case(s);
    }
    {
        // 稀疏 + 等级量化提升并列
        DataGenSpec<double> s; s.seed=654; s.use_sparse=true; s.col=256; s.n_ref=30; s.n_tar=30; s.sparse_value=0.0; s.min_val=0.0; s.max_val=1.0; s.integer_values=false; s.levels=16;
        run_param_case(s);
    }

    // 随机扫测：稠密整数/浮点，各5轮
    run_random_sweep<int>(9000, 5, false, 0, 9, true, 0, 500, 600);
    run_random_sweep<double>(9100, 5, false, 0.0, 1.0, false, 8, 400, 450);
    // 随机扫测：稀疏（零为稀疏），各5轮
    run_random_sweep<double>(9200, 5, true, 0.1, 3.0, false, 0, 40, 50, 200, 0.0);

    // 全相等（大量并列）
    {
        std::vector<int> ref(1000, 5), tar(800, 5);
        auto naive = naive_u_and_tie(ref, tar, 0, false, 0);
        auto fast  = rank_sum_nonsparse<int>(ref, tar, true, true);
        if (fast.U_ref != naive.U_ref || fast.tie_sum != naive.tie_sum)
            throw std::runtime_error("mismatch (all equal)");
        std::cout << "[PASS] all equal dense n_ref=1000 n_tar=800\n";
    }

    // 完全分离（ref 全大于 tar）：tie_sum=0，U_ref=n_ref* n_tar
    {
        DataGenSpec<int> s1; s1.n_ref=500; s1.n_tar=600; s1.use_sparse=false; s1.min_val=100; s1.max_val=200; s1.integer_values=true;
        DataGenSpec<int> s2 = s1; s2.min_val=0; s2.max_val=10; // tar 低、ref 高
        std::vector<int> ref, tar;
        generate_data(s1, ref, tar);
        std::vector<int> tar_low; std::vector<int> dummy;
        generate_data(s2, tar_low, dummy);
        tar.swap(tar_low);
        auto naive = naive_u_and_tie(ref, tar, 0, false, 0);
        auto fast  = rank_sum_nonsparse<int>(ref, tar, false, false);
        if (fast.U_ref != naive.U_ref || fast.tie_sum != naive.tie_sum)
            throw std::runtime_error("mismatch (disjoint ranges)");
        std::cout << "[PASS] disjoint ranges dense n_ref=500 n_tar=600\n";
    }

    // 极端稀疏：col 很大，nnz 很小，sparse_value=0
    {
        DataGenSpec<double> s; s.seed=1357; s.use_sparse=true; s.col=5000; s.n_ref=10; s.n_tar=12; s.sparse_value=0.0; s.min_val=0.1; s.max_val=2.0;
        run_param_case(s);
    }

    // 追加测试：小范围整数（高并列）
    {
        DataGenSpec<int> s; s.seed=20250910; s.n_ref=2000; s.n_tar=2000; s.use_sparse=false; s.min_val=0; s.max_val=3; s.integer_values=true;
        run_param_case(s);
        std::cout << "[PASS] extra small-range ints tie-heavy" << std::endl;
    }

    // 追加测试：稀疏三种情况（min/max/none）对齐 naive（通过 pad 方式）
    {
        // min 情况：sparse_value 最小（0.0）
        DataGenSpec<double> s; s.seed=20250911; s.use_sparse=true; s.col=300; s.n_ref=40; s.n_tar=45; s.sparse_value=0.0; s.min_val=0.2; s.max_val=1.8;
        run_param_case(s);
        std::cout << "[PASS] sparse min" << std::endl;
    }
    {
        // max 情况：sparse_value 远大于显式值
        DataGenSpec<double> s; s.seed=20250912; s.use_sparse=true; s.col=300; s.n_ref=35; s.n_tar=50; s.sparse_value=1000.0; s.min_val=0.2; s.max_val=1.8;
        run_param_case(s);
        std::cout << "[PASS] sparse max" << std::endl;
    }
    {
        // none 情况：sparse_value 处于中间
        DataGenSpec<double> s; s.seed=20250913; s.use_sparse=true; s.col=256; s.n_ref=30; s.n_tar=30; s.sparse_value=0.5; s.min_val=0.0; s.max_val=1.0; s.integer_values=false; s.levels=0;
        run_param_case(s);
        std::cout << "[PASS] sparse none" << std::endl;
    }

    // 追加测试：包含 invalid (NaN/Inf) 的情况（通过 CSC + mannwhitney 过滤 invalid 验证）
    {
        using T = double;
        std::vector<T> ref{0.3, std::numeric_limits<T>::infinity(), 0.7, std::numeric_limits<T>::quiet_NaN()};
        std::vector<T> tar{0.2, 0.7, std::numeric_limits<T>::quiet_NaN(), 1.1};
        // 过滤出有效值进行 naive 参照
        std::vector<T> ref_ok, tar_ok;
        for (auto v: ref) if (!std::isnan(v) && !std::isinf(v)) ref_ok.push_back(v);
        for (auto v: tar) if (!std::isnan(v) && !std::isinf(v)) tar_ok.push_back(v);
        auto naive = naive_u_and_tie(ref_ok, tar_ok, 0, false, T{});

        // 构造 1 列 CSC，将有效与无效一起放入，交由 mannwhitney 侧过滤
        // 行按 [ref..., tar...] 排列
        const size_t R = ref.size() + tar.size();
        const size_t C = 1;
        std::vector<int64_t> indptr{0, (int64_t)R};
        std::vector<int64_t> indices(R);
        for (size_t r=0; r<R; ++r) indices[r] = (int64_t)r;
        std::vector<T> data; data.reserve(R);
        for (auto v: ref) data.push_back(v);
        for (auto v: tar) data.push_back(v);
        hpdexc::CscView A(indptr.data(), indices.data(), data.data(), R, C, R, true);
        std::vector<int32_t> gid(R, 0);
        for (size_t r=ref.size(); r<R; ++r) gid[r] = 1; // 后半为 tar 组1
        hpdexc::DenseView group_id(gid.data(), R, 1, 1);

        hpdexc::MannWhitneyOption<T> opt{};
        opt.ref_sorted=false; opt.tar_sorted=false;
        opt.tie_correction=true; opt.use_continuity=false;
        opt.use_sparse_value=false; opt.is_spare_minmax=hpdexc::MannWhitneyOption<T>::SparseValueMinmax::none; opt.sparse_value=T{};
        opt.alternative=hpdexc::MannWhitneyOption<T>::Alternative::two_sided; opt.method=hpdexc::MannWhitneyOption<T>::Method::asymptotic;

        auto res = hpdexc::mannwhitney<T>(A, opt, group_id, /*n_targets=*/1, /*threads=*/1);
        const long double U = res.U_buf[0];
        if (std::llround(U * 1e12) != std::llround(naive.U_ref * 1e12))
            throw std::runtime_error("mismatch (invalid filtering)");
        std::cout << "[PASS] invalid values filtered" << std::endl;
    }

    // 追加测试：多组（ref=0，tar=1..k），逐一与 naive 对比
    {
        using T = double;
        // 三组：0=ref, 1=tarA, 2=tarB
        std::vector<T> ref{0.2, 0.5, 0.7, 1.0};
        std::vector<T> tarA{0.1, 0.6, 0.9};
        std::vector<T> tarB{0.3, 0.4, 1.2, 1.3};
        // 构成 1 列 CSC，行顺序拼接三组
        const size_t R = ref.size() + tarA.size() + tarB.size();
        const size_t C = 1;
        std::vector<int64_t> indptr{0, (int64_t)R};
        std::vector<int64_t> indices(R);
        for (size_t r=0; r<R; ++r) indices[r] = (int64_t)r;
        std::vector<T> data; data.reserve(R);
        data.insert(data.end(), ref.begin(), ref.end());
        data.insert(data.end(), tarA.begin(), tarA.end());
        data.insert(data.end(), tarB.begin(), tarB.end());
        hpdexc::CscView A(indptr.data(), indices.data(), data.data(), R, C, R, true);
        std::vector<int32_t> gid(R, 0);
        for (size_t r=ref.size(); r<ref.size()+tarA.size(); ++r) gid[r] = 1;
        for (size_t r=ref.size()+tarA.size(); r<R; ++r) gid[r] = 2;
        hpdexc::DenseView group_id(gid.data(), R, 1, 1);

        hpdexc::MannWhitneyOption<T> opt{};
        opt.ref_sorted=false; opt.tar_sorted=false; opt.tie_correction=true; opt.use_continuity=false;
        opt.use_sparse_value=false; opt.is_spare_minmax=hpdexc::MannWhitneyOption<T>::SparseValueMinmax::none; opt.sparse_value=T{};
        opt.alternative=hpdexc::MannWhitneyOption<T>::Alternative::two_sided; opt.method=hpdexc::MannWhitneyOption<T>::Method::asymptotic;

        auto res = hpdexc::mannwhitney<T>(A, opt, group_id, /*n_targets=*/2, /*threads=*/1);
        // 对比 naive：ref vs tarA, ref vs tarB
        auto nA = naive_u_and_tie(ref, tarA, 0, false, T{});
        auto nB = naive_u_and_tie(ref, tarB, 0, false, T{});
        const long double U_A = res.U_buf[0];
        const long double U_B = res.U_buf[1];
        if (std::llround(U_A * 1e12) != std::llround(nA.U_ref * 1e12))
            throw std::runtime_error("mismatch (multi-group A)");
        if (std::llround(U_B * 1e12) != std::llround(nB.U_ref * 1e12))
            throw std::runtime_error("mismatch (multi-group B)");
        std::cout << "[PASS] multi-group U matches naive pairwise" << std::endl;
    }

    // ---------------- 压力测试：不同稀疏度（默认稀疏值=0，作为最小值） ----------------
    {
        using T = double;
        auto time_ms = [](auto&& fn){
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };
        auto benchmark_once = [&](std::size_t col, double density, unsigned seed){
            const size_t R = col * 2; // ref 与 tar 各 col
            const size_t C = 1;
            std::mt19937 rng(seed);
            std::uniform_int_distribution<size_t> dist_idx(0, col - 1);
            std::uniform_real_distribution<double> dist_val(0.1, 3.0);
            const size_t nnz_ref = (size_t)std::llround(density * (double)col);
            const size_t nnz_tar = (size_t)std::llround(density * (double)col);
            // 选位置（不重复）
            std::vector<size_t> pos_ref, pos_tar; pos_ref.reserve(nnz_ref); pos_tar.reserve(nnz_tar);
            std::vector<uint8_t> used_ref(col,0), used_tar(col,0);
            while (pos_ref.size() < nnz_ref) { size_t p=dist_idx(rng); if(!used_ref[p]){used_ref[p]=1; pos_ref.push_back(p);} }
            while (pos_tar.size() < nnz_tar) { size_t p=dist_idx(rng); if(!used_tar[p]){used_tar[p]=1; pos_tar.push_back(p);} }
            std::sort(pos_ref.begin(), pos_ref.end());
            std::sort(pos_tar.begin(), pos_tar.end());
            // 组装 CSC（单列）
            std::vector<int64_t> indptr{0, (int64_t)(nnz_ref + nnz_tar)};
            std::vector<int64_t> indices; indices.reserve(nnz_ref + nnz_tar);
            std::vector<T> data; data.reserve(nnz_ref + nnz_tar);
            for (size_t p: pos_ref){ indices.push_back((int64_t)p); data.push_back((T)dist_val(rng)); }
            for (size_t p: pos_tar){ indices.push_back((int64_t)(col + p)); data.push_back((T)dist_val(rng)); }
            hpdexc::CscView A(indptr.data(), indices.data(), data.data(), R, C, indices.size(), true);
            std::vector<int32_t> gid(R, 0); for (size_t r=col; r<R; ++r) gid[r]=1;
            hpdexc::DenseView group_id(gid.data(), R, 1, 1);
            hpdexc::MannWhitneyOption<T> opt{};
            opt.ref_sorted=false; opt.tar_sorted=false; opt.tie_correction=true; opt.use_continuity=false;
            opt.use_sparse_value=true; opt.is_spare_minmax=hpdexc::MannWhitneyOption<T>::SparseValueMinmax::min; opt.sparse_value=(T)0.5; // 非零取[0.1,3.0]，选0.5确保为最小值
            opt.alternative=hpdexc::MannWhitneyOption<T>::Alternative::two_sided; opt.method=hpdexc::MannWhitneyOption<T>::Method::asymptotic;
            // 提取显式非零（用于朴素优化版）
            std::vector<T> ref_nz(data.begin(), data.begin()+ (ptrdiff_t)nnz_ref);
            std::vector<T> tar_nz(data.begin()+ (ptrdiff_t)nnz_ref, data.end());
            // 重复计时
            const int reps = 5;
            double t_fast=0.0, t_naive=0.0;
            for (int k=0;k<reps;++k){ t_fast += time_ms([&]{ auto r = hpdexc::mannwhitney<T>(A,opt,group_id,1,1); 
                                                              if (k==0) {
                                                                  auto naive = naive_u_sparse_value_optimized<T>(ref_nz, tar_nz, col, opt.sparse_value, opt.is_spare_minmax);
                                                                  if (std::llround(r.U_buf[0]*1e12) != std::llround(naive.U_ref*1e12))
                                                                      throw std::runtime_error("mismatch (sparsity sweep U)");
                                                              }
                                                            }); }
            for (int k=0;k<reps;++k){ t_naive+= time_ms([&]{ auto n = naive_u_sparse_value_optimized<T>(ref_nz, tar_nz, col, opt.sparse_value, opt.is_spare_minmax); (void)n; }); }
            return std::pair<double,double>(t_fast/reps, t_naive/reps);
        };
        std::cout << "\n[Benchmark] sparsity sweep (sparse=0 as min)" << std::endl;
        const size_t col = 20000; // 每组样本数
        std::vector<double> densities{0.005, 0.01, 0.02, 0.05, 0.10, 0.20};
        unsigned seed = 20250914;
        for (size_t i=0;i<densities.size();++i){
            auto d = densities[i];
            auto pr = benchmark_once(col, d, seed + (unsigned)i);
            std::cout << "density=" << d
                      << " fast(ms)=" << pr.first
                      << " naive(ms)=" << pr.second
                      << " speedup=" << (pr.second>0? (pr.second/pr.first):0.0)
                      << std::endl;
        }
    }

    // ---------------- 压力测试：稠密矩阵（两组，大规模以压低调度开销） ----------------
    {
        using T = double;
        auto time_ms = [](auto&& fn){
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };
        std::mt19937 rng(20250915);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        const size_t n_ref = 200000; // 20万
        const size_t n_tar = 200000; // 20万
        std::vector<T> ref(n_ref), tar(n_tar);
        for (auto& v: ref) v = (T)dist(rng);
        for (auto& v: tar) v = (T)dist(rng);

        // 计时
        const int reps = 3;
        double t_fast = 0.0, t_naive = 0.0;
        for (int k=0;k<reps;++k) t_fast  += time_ms([&]{ auto r = rank_sum_nonsparse<T>(ref, tar, false, false); (void)r; });
        for (int k=0;k<reps;++k) t_naive += time_ms([&]{ auto n = naive_u_and_tie<T>(ref, tar, 0, false, T{}); (void)n; });
        t_fast  /= reps; t_naive /= reps;
        std::cout << "\n[Benchmark] dense two-group n_ref=" << n_ref << " n_tar=" << n_tar
                  << " fast(ms)=" << t_fast << " naive(ms)=" << t_naive
                  << " speedup=" << (t_naive>0? (t_naive/t_fast):0.0) << std::endl;
    }

    // ---------------- 压力测试：稠密多组（一次扫描 vs 逐对 naive） ----------------
    {
        using T = double;
        auto time_ms = [](auto&& fn){
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };
        std::mt19937 rng(20250916);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        const size_t groups = 6; // 1个ref + 5个tar
        const size_t n_per_group = 50000; // 每组5万
        const size_t R = groups * n_per_group;
        const size_t C = 1;

        // 构造稠密一列 CSC（所有行都显式存储）
        std::vector<int64_t> indptr{0, (int64_t)R};
        std::vector<int64_t> indices(R);
        for (size_t r=0;r<R;++r) indices[r] = (int64_t)r;
        std::vector<T> data(R);
        for (auto& v: data) v = (T)dist(rng);
        hpdexc::CscView A(indptr.data(), indices.data(), data.data(), R, C, R, true);
        std::vector<int32_t> gid(R, 0);
        for (size_t g=1; g<groups; ++g) {
            const size_t start = g * n_per_group;
            for (size_t r=0; r<n_per_group; ++r) gid[start + r] = (int32_t)g;
        }
        hpdexc::DenseView group_id(gid.data(), R, 1, 1);

        hpdexc::MannWhitneyOption<T> opt{};
        opt.ref_sorted=false; opt.tar_sorted=false; opt.tie_correction=true; opt.use_continuity=false;
        opt.use_sparse_value=false; opt.is_spare_minmax=hpdexc::MannWhitneyOption<T>::SparseValueMinmax::none; opt.sparse_value=T{};
        opt.alternative=hpdexc::MannWhitneyOption<T>::Alternative::two_sided; opt.method=hpdexc::MannWhitneyOption<T>::Method::asymptotic;

        // 一次扫描（多组）
        const int reps = 3;
        double t_once = 0.0;
        for (int k=0;k<reps;++k) t_once += time_ms([&]{ auto r = hpdexc::mannwhitney<T>(A, opt, group_id, groups-1, 1); (void)r; });
        t_once /= reps;

        // 逐对 naive：ref vs 每个 tar
        std::vector<T> ref(n_per_group);
        std::copy(data.begin(), data.begin()+ (ptrdiff_t)n_per_group, ref.begin());
        double t_naive_sum = 0.0;
        for (size_t g=1; g<groups; ++g) {
            std::vector<T> tar(n_per_group);
            const size_t start = g * n_per_group;
            std::copy(data.begin()+ (ptrdiff_t)start, data.begin()+ (ptrdiff_t)(start + n_per_group), tar.begin());
            double t_naive = 0.0;
            for (int k=0;k<reps;++k) t_naive += time_ms([&]{ auto n = naive_u_and_tie<T>(ref, tar, 0, false, T{}); (void)n; });
            t_naive /= reps;
            t_naive_sum += t_naive;
        }

        std::cout << "\n[Benchmark] dense multi-group ref=1 tar=" << (groups-1)
                  << " once(ms)=" << t_once
                  << " naive_sum(ms)=" << t_naive_sum
                  << " speedup=" << (t_naive_sum>0? (t_naive_sum/t_once):0.0)
                  << std::endl;
    }

    // 微小样本：n_ref=1, n_tar=1（不同与相等）
    {
        std::vector<int> ref{3}, tar{7};
        auto naive = naive_u_and_tie(ref, tar, 0, false, 0);
        auto fast  = rank_sum_nonsparse<int>(ref, tar, true, true);
        if (fast.U_ref != naive.U_ref || fast.tie_sum != naive.tie_sum)
            throw std::runtime_error("mismatch (1v1 distinct)");
        std::cout << "[PASS] 1v1 distinct\n";
    }
    {
        std::vector<int> ref{5}, tar{5};
        auto naive = naive_u_and_tie(ref, tar, 0, false, 0);
        auto fast  = rank_sum_nonsparse<int>(ref, tar, true, true);
        if (fast.U_ref != naive.U_ref || fast.tie_sum != naive.tie_sum)
            throw std::runtime_error("mismatch (1v1 equal)");
        std::cout << "[PASS] 1v1 equal\n";
    }

    // 已排序输入一致性：sorted=true 与 false 结果一致
    {
        DataGenSpec<double> s; s.seed=97531; s.n_ref=300; s.n_tar=250; s.use_sparse=false; s.min_val=0.0; s.max_val=5.0; s.integer_values=false; s.levels=7;
        std::vector<double> ref, tar; generate_data(s, ref, tar);
        auto fast_unsorted = rank_sum_nonsparse<double>(ref, tar, false, false);
        std::sort(ref.begin(), ref.end()); std::sort(tar.begin(), tar.end());
        auto fast_sorted   = rank_sum_nonsparse<double>(ref, tar, true, true);
        if (std::llround(fast_sorted.U_ref * 1e12) != std::llround(fast_unsorted.U_ref * 1e12))
            throw std::runtime_error("mismatch (sorted flag U)");
        if (std::llround(fast_sorted.tie_sum) != std::llround(fast_unsorted.tie_sum))
            throw std::runtime_error("mismatch (sorted flag tie)");
        std::cout << "[PASS] sorted flag consistency\n";
    }

    // ---------------- 基因差异表达场景：100组 × 每组5000样本 × 20000列，稀疏度 0.1/0.2/0.3 ----------------
    {
        using T = double;
        const size_t groups = 100;
        const size_t n_per_group = 5000;
        const size_t R = groups * n_per_group;
        const size_t C = 20000;
        std::vector<int32_t> gid(R, 0);
        for (size_t g=1; g<groups; ++g) {
            const size_t start = g * n_per_group;
            for (size_t r=0; r<n_per_group; ++r) gid[start + r] = (int32_t)g;
        }
        hpdexc::DenseView group_id(gid.data(), R, 1, 1);

        hpdexc::MannWhitneyOption<T> opt{};
        opt.ref_sorted=false; opt.tar_sorted=false; opt.tie_correction=true; opt.use_continuity=false;
        opt.use_sparse_value=true; opt.is_spare_minmax=hpdexc::MannWhitneyOption<T>::SparseValueMinmax::min; opt.sparse_value=T(0);
        opt.alternative=hpdexc::MannWhitneyOption<T>::Alternative::two_sided; opt.method=hpdexc::MannWhitneyOption<T>::Method::asymptotic;

        std::mt19937 rng(20250917);
        std::uniform_real_distribution<double> dist_val(0.1, 3.0); // 非零为正
        std::uniform_int_distribution<size_t> dist_idx(0, n_per_group - 1);

        auto time_ms = [](auto&& fn){
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };

        auto build_one_col = [&](double density,
                                  std::vector<int64_t>& indptr,
                                  std::vector<int64_t>& indices,
                                  std::vector<T>& data,
                                  // 可选输出：用于一致性抽检（ref 与某个 tar 的非零）
                                  std::vector<T>* ref_nz_out,
                                  std::vector<T>* tar_nz_out,
                                  size_t tar_group)->void {
            indptr.clear(); indices.clear(); data.clear();
            indptr.push_back(0);
            // 为每组生成 nnz 位置与数值
            for (size_t g=0; g<groups; ++g) {
                const size_t nnz_g = (size_t)std::llround(density * (double)n_per_group);
                std::vector<size_t> pos; pos.reserve(nnz_g);
                std::vector<uint8_t> used(n_per_group, 0);
                while (pos.size() < nnz_g) {
                    size_t p = dist_idx(rng);
                    if (!used[p]) { used[p]=1; pos.push_back(p); }
                }
                std::sort(pos.begin(), pos.end());
                for (size_t p: pos) {
                    indices.push_back((int64_t)(g * n_per_group + p));
                    T v = (T)dist_val(rng);
                    data.push_back(v);
                    if (ref_nz_out && g==0) ref_nz_out->push_back(v);
                    if (tar_nz_out && g==tar_group) tar_nz_out->push_back(v);
                }
            }
            indptr.push_back((int64_t)indices.size());
        };

        auto run_density = [&](double density){
            // 计时：按列逐一构造 1 列 CSC 并调用算子（threads=1），统计总用时与吞吐
            const int reps = 1;
            double ms_total = 0.0;
            // 抽检一致性：只对第一列做一次 ref vs tar=1 的朴素比对
            bool checked = false;
            for (int rep=0; rep<reps; ++rep) {
                ms_total += time_ms([&]{
                    for (size_t c=0; c<C; ++c) {
                        std::vector<int64_t> indptr; indptr.reserve(2);
                        std::vector<int64_t> indices; indices.reserve((size_t)(density*R)+16);
                        std::vector<T> data; data.reserve((size_t)(density*R)+16);
                        std::vector<T> ref_nz, tar_nz;
                        std::vector<T>* pref = nullptr; std::vector<T>* ptar = nullptr;
                        if (!checked) { pref=&ref_nz; ptar=&tar_nz; }
                        build_one_col(density, indptr, indices, data, pref, ptar, /*tar_group=*/1);
                        hpdexc::CscView A(indptr.data(), indices.data(), data.data(), R, 1, indices.size(), true);
                        auto res = hpdexc::mannwhitney<T>(A, opt, group_id, groups-1, /*threads=*/1);
                        if (!checked) {
                            auto naive = naive_u_sparse_value_optimized<T>(ref_nz, tar_nz, n_per_group, /*sparse_value*/T(0), hpdexc::MannWhitneyOption<T>::SparseValueMinmax::min);
                            const long double U_ref_tar1 = res.U_buf[0];
                            if (std::llround(U_ref_tar1*1e12) != std::llround(naive.U_ref*1e12))
                                throw std::runtime_error("mismatch (DE large multi-group, sanity check)");
                            checked = true;
                        }
                    }
                });
            }
            const double ms_avg = ms_total / reps;
            const double cols_per_s = (ms_avg>0) ? (1000.0 * (double)C / ms_avg) : 0.0;
            std::cout << "[DE Benchmark] groups=100 n_per_group=5000 cols=" << C
                      << " density=" << density
                      << " time(ms)=" << ms_avg
                      << " throughput(cols/s)=" << cols_per_s
                      << std::endl;
        };

        std::cout << "\n[Benchmark] DE scenario (single-thread full-batch; groups=100, n_per_group=5000, cols=20000, density=0.2)" << std::endl;
        // 我们算子：整批、单线程
        {
            const double density = 0.20;
            auto time_ms = [](auto&& fn){ auto t0=std::chrono::high_resolution_clock::now(); fn(); auto t1=std::chrono::high_resolution_clock::now(); return std::chrono::duration<double, std::milli>(t1-t0).count(); };
            double ms_total = 0.0;
            ms_total += time_ms([&]{
                for (size_t c=0; c<C; ++c) {
                    std::vector<int64_t> indptr; indptr.reserve(2);
                    std::vector<int64_t> indices; indices.reserve((size_t)(density*R)+16);
                    std::vector<T> data; data.reserve((size_t)(density*R)+16);
                    build_one_col(density, indptr, indices, data, nullptr, nullptr, /*tar_group=*/1);
                    hpdexc::CscView A(indptr.data(), indices.data(), data.data(), R, 1, indices.size(), true);
                    (void)hpdexc::mannwhitney<T>(A, opt, group_id, groups-1, /*threads=*/1);
                }
            });
            std::cout << "[DE Benchmark][ours full-batch] total(ms)=" << ms_total
                      << " throughput(cols/s)=" << (ms_total>0? 1000.0*(double)C/ms_total : 0.0) << std::endl;
        }

        // 朴素算子：整批、单线程
        {
            const double density = 0.20;
            auto time_ms = [](auto&& fn){ auto t0=std::chrono::high_resolution_clock::now(); fn(); auto t1=std::chrono::high_resolution_clock::now(); return std::chrono::duration<double, std::milli>(t1-t0).count(); };
            double ms_total = 0.0;
            ms_total += time_ms([&]{
                for (size_t c=0; c<C; ++c) {
                    std::vector<int64_t> indptr; std::vector<int64_t> indices; std::vector<T> data;
                    std::vector<T> ref_nz, tar_nz;
                    build_one_col(density, indptr, indices, data, &ref_nz, nullptr, /*tar_group=*/1);
                    for (size_t g=1; g<groups; ++g) {
                        tar_nz.clear(); tar_nz.reserve((size_t)std::llround(density*n_per_group));
                        for (size_t k=0; k<indices.size(); ++k) {
                            size_t row = (size_t)indices[k];
                            size_t grp = row / n_per_group;
                            if (grp == g) tar_nz.push_back(data[k]);
                        }
                        (void)naive_u_sparse_value_optimized<T>(ref_nz, tar_nz, n_per_group, /*sparse_value*/T(0), hpdexc::MannWhitneyOption<T>::SparseValueMinmax::min);
                    }
                }
            });
            std::cout << "[DE Benchmark][naive full-batch] total(ms)=" << ms_total
                      << " throughput(cols/s)=" << (ms_total>0? 1000.0*(double)C/ms_total : 0.0) << std::endl;
        }

        // 保留：naive 单列微基准（用于常数项感知）
        std::cout << "\n[Benchmark] DE scenario (naive micro, one column pairwise repeats)" << std::endl;
        // 仅跑 naive（density=0.20，一列，逐对 ref vs tar），放大量程以避免计时粒度误差
        {
            const double density = 0.20;
            std::vector<int64_t> indptr; std::vector<int64_t> indices; std::vector<T> data;
            std::vector<T> ref_nz, tar_nz;
            build_one_col(density, indptr, indices, data, &ref_nz, nullptr, /*tar_group=*/1);
            auto time_ms = [](auto&& fn){ auto t0=std::chrono::high_resolution_clock::now(); fn(); auto t1=std::chrono::high_resolution_clock::now(); return std::chrono::duration<double, std::milli>(t1-t0).count(); };
            const int repeats_outer = 5;     // 外层重复若干次取平均
            const int repeats_inner = 50;    // 内层重复调用，放大量程
            // 预先收集每个 tar 组的非零，避免把数据收集时间算进来
            std::vector<std::vector<T>> tar_nz_list(groups);
            for (size_t g=1; g<groups; ++g) {
                auto& tz = tar_nz_list[g]; tz.clear(); tz.reserve((size_t)std::llround(density*n_per_group));
                for (size_t k=0; k<indices.size(); ++k) {
                    size_t row = (size_t)indices[k];
                    size_t grp = row / n_per_group;
                    if (grp == g) tz.push_back(data[k]);
                }
                std::sort(tz.begin(), tz.end());
            }
            double ms_naive_sum = 0.0;
            for (int r=0; r<repeats_outer; ++r) {
                ms_naive_sum += time_ms([&]{
                    volatile long double guard_acc = 0.0L; // 防止编译器消除
                    for (size_t g=1; g<groups; ++g) {
                        const auto& tz = tar_nz_list[g];
                        for (int t=0; t<repeats_inner; ++t) {
                            auto n = naive_u_sparse_value_optimized<T>(ref_nz, std::vector<T>(tz.begin(), tz.end()), n_per_group, /*sparse_value*/T(0), hpdexc::MannWhitneyOption<T>::SparseValueMinmax::min);
                            guard_acc += n.U_ref;
                        }
                    }
                    (void)guard_acc;
                });
            }
            const double ms_avg = ms_naive_sum / repeats_outer;
            const double per_eval_ms = ms_avg / (double)((groups-1) * repeats_inner);
            std::cout << "[DE Benchmark][naive only] density=0.2 one-col pairwise total(ms)=" << ms_avg
                      << " per_eval(ms)=" << per_eval_ms << std::endl;
        }
    }

    std::cout << "All tests passed" << std::endl;
    return 0;
}
