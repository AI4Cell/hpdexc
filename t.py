import numpy as np
import scipy as sp
from hpdex.backend.kernel import mannwhitney as mwu_hpdex

d = np.random.randint(1, 20, (6, 6))
A = sp.sparse.csc_matrix(d)
group_id = np.random.randint(0, 2, 6)

result = mwu_hpdex(
        A,
        group_id=np.asarray(group_id, dtype=np.int32),
        n_targets=1,            # !!! 核心：G-1
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
print(d)
print(A)

print(result.U)
print(result.P)

