import numpy as np
from tqdm import tqdm
from .backend import kernel  # 你的 pybind11 模块
from scipy.sparse import csc_matrix

def mannwhitneyu_test(sp_csc, gid, n_targets=4):
    """
    测试 Mann-Whitney U 统计检验
    
    Args:
        sp_csc: scipy.sparse.csc_matrix 格式的稀疏矩阵
        gid: group_id 数组，int32 类型
        n_targets: 目标数量，默认4
    """
    cols = sp_csc.shape[1]
    total = cols * n_targets

    pbar = tqdm(total=total, desc="MWU", mininterval=0.1)

    def on_progress(done, total_):
        # 在 C++ 回调线程中执行（已持有 GIL）
        pbar.n = min(done, total_)
        pbar.refresh()

    U, P = kernel.mannwhitneyu(
        sp_csc, gid.astype(np.int32), n_targets,
        tie_correction=True, use_continuity=True,
        alternative=2, method=0,
        ref_sorted=False, tar_sorted=False,
        sparse_minmax=0, sparse_value=None,
        threads=8,
        progress="cb",                # 用内部缓冲，仅回调
        on_progress=on_progress,
        progress_interval_ms=100
    )
    pbar.close()
    return U, P