import numpy as np
import scipy as sp
from hpdex.backend.kernel import mannwhitney as mwu_hpdex
from scipy.stats import mannwhitneyu as mwu_scipy

def test_scipy(A, group_id):
    # 调换顺序：让 SciPy 使用与 hpdex 相同的组别顺序
    # hpdex 中：组0是参考组，组1是目标组
    # 所以 SciPy 应该用 group_id == 1 作为第一个参数，group_id == 0 作为第二个参数
    return mwu_scipy(A[group_id == 0, :], A[group_id == 1, :], alternative='two-sided',
                         use_continuity=True,
                         method='asymptotic', axis=0)

def test_hpdex(A, group_id, sp_val):
    A = sp.sparse.csc_matrix(A)
    result = mwu_hpdex(A, group_id=group_id,
        n_targets=1,            # !!! 核心：G-1
        tie_correction=True,            # 几乎无并列，开关不影响，但与 SciPy 对齐
        use_continuity=True,            # 与 SciPy 对齐
        alternative=2,                  # 0:less,1:greater,2:two-sided
        method=2,                       # 0:auto,1:exact,2:asymptotic
        ref_sorted=False,
        tar_sorted=False,
        use_sparse_value=True,         # 关键：不启用稀疏值
        is_sparse_minmax=False,         # 无关
        sparse_value=sp_val,              # 无关
        use_histogram=False,
        max_bins=65536,
        mem_budget_bytes=1 << 30,
        threads=1,
        progress=None,
    )
    return result.U, result.P

def make_data(shape, minmax, spz):
    data = np.random.rand(*shape)
    mask = np.random.choice(shape[0], size=int(shape[0] * spz), replace=False)  # 随机选择 spz 比例的索引
    data[mask] = np.nan
    group_id = np.concatenate((np.ones(shape[0] // 2, dtype=np.int32), np.zeros(shape[0] // 2, dtype=np.int32)))
    return data, group_id

def dense_to_csc(dense, sp_val):
    rows, cols = dense.shape
    data = []
    indices = []
    indptr = [0]
    
    for j in range(cols):  # 按列扫描
        for i in range(rows):
            if dense[i, j] != 0:  # 非零才存
                data.append(dense[i, j])
                indices.append(i)
        indptr.append(len(data))
    
    return sp.sparse.csc_matrix((np.array(data), np.array(indices), np.array(indptr)), shape=dense.shape)

def make_sparse_data(shape, minmax, spz, sp_val):
    data = np.random.rand(*shape)
    mask = np.random.randn() < spz
    data[mask] = sp_val if sp_val is not None else np.nan
    group_id = np.concatenate((np.ones(shape[0] // 2, dtype=np.int32), np.zeros(shape[0] // 2, dtype=np.int32)))
    
    sparse_data = dense_to_csc(data, sp_val)
    return data, sparse_data, group_id

def test(shape, minmax, spz, sp_val, repeats):
    print(f"testing {shape} with {minmax} {spz} {sp_val} {repeats} times")
    import time
    hpdex_time = 0
    scipy_time = 0
    max_U_diff = 0
    max_P_diff = 0
    for _ in range(repeats):
        dense_data, sparse_data, group_id = make_sparse_data(shape, minmax, spz, sp_val)
        start_hpdex = time.time()
        hU, hP = test_hpdex(sparse_data, group_id, 0)
        end_hpdex = time.time()
        hpdex_time += end_hpdex - start_hpdex

        start_scipy = time.time()
        U, P = test_scipy(dense_data, group_id)
        end_scipy = time.time()
        scipy_time += end_scipy - start_scipy
        
        max_U_diff = max(max_U_diff, np.max(np.abs(hU - U)))
        max_P_diff = max(max_P_diff, np.max(np.abs(hP - P)))
        
    return hpdex_time, scipy_time, max_U_diff, max_P_diff
        
    


print("\n=== 结果比较 ===")

ht, st, du, dp = test((1000, 1000), (-5000, 5000), 0.0, 0, 1)
print("hpdex time:", ht)
print("scipy time:", st)
print("max U diff:", du)
print("max P diff:", dp)
print("speedup: ", st / ht, "x")