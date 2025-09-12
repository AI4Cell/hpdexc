import numpy as np
import scipy as sp
from scipy.stats import mannwhitneyu as scipy_mwu
from hpdex.backend.kernel import mannwhitneyu as hpdex_mwu

# 随机生成数据
R, C = 1000, 1000   # 100 rows × 100 cols
dense = np.random.rand(R, C)

# 分组标签 (前 50 行 = ref, 后 50 行 = tar)
gid = np.array([0] * (R // 2) + [1] * (R // 2), dtype=np.int32)
n_targets = 1

# --- SciPy 版本 ---
Us, Ps = scipy_mwu(
    dense[:R // 2, :],  # ref
    dense[R // 2:, :],  # tar
    method="asymptotic",
    use_continuity=True,
    alternative="two-sided"
)

# --- hpdex 版本 ---
A = sp.sparse.csc_matrix(dense)
Uh, Ph = hpdex_mwu(
    A, gid, n_targets,
    threads=-1,
    ref_sorted=False,
    tar_sorted=False,
    tie_correction=True,
    use_continuity=True,
    sparse_type=0,       # none
    sparse_value=0.0,
    alternative=2,       # two-sided
    method=2             # asymptotic
)

# --- 对比结果 ---
print("Max U difference:", np.abs(Us - Uh.T).max())
print("Max P difference:", np.abs(Ps - Ph.T).max())

print(np.finfo(float).eps)