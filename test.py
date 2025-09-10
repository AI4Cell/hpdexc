# test_dense_align.py
import numpy as np
import scipy as sp
from scipy.stats import mannwhitneyu as mwu_scipy
from hpdex.backend.kernel import mannwhitney as mwu_hpdex

def run_dense_alignment(R=400, C=50, G=4, seed=0):
    """
    R: 总样本数（行）
    C: 特征/列数
    G: 组数，0 为 ref，其余 1..G-1 为 tar（因此 n_targets = G - 1）
    """
    rng = np.random.default_rng(seed)

    # ------- 构造致密正数矩阵（无 0） -------
    X = rng.random((R, C)) + 1e-6  # 全为正数，几乎无并列

    # ------- 构造分组：尽量均匀 -------
    # group_id: shape (R,)
    base = R // G
    extras = R % G
    counts = [base + (1 if i < extras else 0) for i in range(G)]
    group_id = np.concatenate([np.full(c, i, dtype=np.int32) for i, c in enumerate(counts)])
    rng.shuffle(group_id)  # 打乱行的组别分布

    # ------- SciPy 参考结果（逐列） -------
    # shape: (n_targets, C) 其中 n_targets = G - 1
    n_targets = G - 1
    U_scipy = np.empty((n_targets, C), dtype=float)
    P_scipy = np.empty((n_targets, C), dtype=float)

    ref = X[group_id == 0, :]  # (n_ref, C)
    for i in range(1, G):
        tar = X[group_id == i, :]  # (n_tar, C)
        # SciPy 返回 shape (C,)；我们堆叠为 (n_targets, C)
        U, P = mwu_scipy(
            ref, tar,
            axis=1,                      # 按列比较
            alternative="two-sided",
            method="asymptotic",
            use_continuity=True,
        )
        U_scipy[i-1, :] = U
        P_scipy[i-1, :] = P

    # ------- hpdex 结果（传入 CSC，但不启用稀疏值逻辑） -------
    A = sp.sparse.csc_matrix(X)  # 只是接口需要 CSC；我们不使用稀疏填充值
    result = mwu_hpdex(
        A,
        group_id=np.asarray(group_id, dtype=np.int32),
        n_targets=n_targets,            # !!! 核心：G-1
        tie_correction=True,            # 几乎无并列，开关不影响，但与 SciPy 对齐
        use_continuity=True,            # 与 SciPy 对齐
        alternative=2,                  # 0:greater,1:less,2:two-sided
        method=2,                       # 0:auto,1:exact,2:asymptotic
        ref_sorted=False,
        tar_sorted=False,
        use_sparse_value=False,         # 关键：不启用稀疏值
        is_sparse_minmax=False,         # 无关
        sparse_value=None,              # 无关
        use_histogram=False,
        max_bins=65536,
        mem_budget_bytes=1 << 30,
        threads=1,
        progress=None,
    )
    U_hpdex = np.asarray(result.U)
    P_hpdex = np.asarray(result.P)

    # ------- 形状校验 -------
    print("SciPy U shape:", U_scipy.shape)
    print("hpdex U shape:", U_hpdex.shape)
    assert U_scipy.shape == U_hpdex.shape == (n_targets, C)

    # ------- 数值对比 -------
    # 对于无并列的致密连续数据，U 应该完全一致；P 在数值误差内一致
    u_diff = np.max(np.abs(U_scipy - U_hpdex))
    p_diff = np.max(np.abs(P_scipy - P_hpdex))
    print("U diff (max abs):", u_diff)
    print("P diff (max abs):", p_diff)

    # 给点头几列看看
    for t in range(min(3, n_targets)):
        print(f"\nTarget {t+1}:")
        print("SciPy U[0:5]:", U_scipy[t, :5])
        print("hpdex U[0:5]:", U_hpdex[t, :5])
        print("SciPy P[0:5]:", P_scipy[t, :5])
        print("hpdex P[0:5]:", P_hpdex[t, :5])

    # 严格断言（根据需要调节阈值）
    # U 在无并列场景通常能完全一致；若你的平台/排序稳定性导致稀有 0.5 误差，可放宽到 1e-12
    assert np.allclose(U_scipy, U_hpdex, atol=0, rtol=0), "U mismatch"
    # P 值允许极小的数值误差
    assert np.allclose(P_scipy, P_hpdex, atol=1e-12, rtol=1e-12), "P mismatch"

    print("\n✅ Dense alignment test PASSED.")

if __name__ == "__main__":
    run_dense_alignment(R=400, C=50, G=4, seed=0)
