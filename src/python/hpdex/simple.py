from __future__ import annotations

from typing import Any

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats

from hpdex.backen import _c_test as C


def _ensure_csc(matrix: sp.spmatrix) -> sp.csc_matrix:
    if sp.isspmatrix_csc(matrix):
        return matrix  # type: ignore[return-value]
    if sp.isspmatrix_csr(matrix):
        return matrix.tocsc(copy=False)  # type: ignore[return-value]
    return sp.csc_matrix(matrix)


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    q_sorted = q[order]
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    out = np.empty_like(q_sorted)
    out[order.argsort()] = np.clip(q_sorted, 0.0, 1.0)
    return out


def wilcoxon_fdr_csc(
    A: sp.spmatrix,
    ref_mask: np.ndarray,
    tar_mask: np.ndarray,
    *,
    tie_correct: bool = True,
    omp_threads: int | None = None,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """简单版：一次性CSC算子 + z/p + BH FDR。

    - 输入可以是 CSR/CSC，其它稀疏类型将被转换为 CSC
    - 可通过 `omp_threads` 设置 OMP 线程数（例如 4/8），未设置保持环境默认
    - 返回每列（特征）的 U1、z、p_value、fdr
    """

    if omp_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(int(omp_threads))

    A_csc = _ensure_csc(A)

    # 调用 C++ 算子，一次性计算所有列 U1
    out = C.mwu_stats_csc(A_csc, ref_mask.astype(np.uint8), tar_mask.astype(np.uint8), calc_tie=tie_correct)
    U1 = np.asarray(out["U1"], dtype=np.float64)

    ref_b = ref_mask.astype(bool)
    tar_b = tar_mask.astype(bool)

    # 逐列计算 z / p（演示用实现，后续可在 C 端优化合并）
    z_vals = np.empty(A_csc.shape[1], dtype=np.float64)
    p_vals = np.empty(A_csc.shape[1], dtype=np.float64)
    for j in range(A_csc.shape[1]):
        col = A_csc.getcol(j)
        dense_col = np.asarray(col.toarray()).ravel()
        xr = dense_col[ref_b]
        xt = dense_col[tar_b]
        n1 = xr.size
        n2 = xt.size
        if n1 == 0 or n2 == 0:
            z_vals[j] = np.nan
            p_vals[j] = np.nan
            continue

        if tie_correct:
            x = np.concatenate([xr, xt])
            _, counts = np.unique(x, return_counts=True)
            t_corr = np.sum(counts ** 3 - counts) / ((n1 + n2) * (n1 + n2 - 1)) if (n1 + n2) > 1 else 0.0
        else:
            t_corr = 0.0

        mean_u = n1 * n2 / 2.0
        var_u = n1 * n2 * (n1 + n2 + 1 - t_corr) / 12.0
        if var_u <= 0:
            z_vals[j] = np.nan
            p_vals[j] = np.nan
            continue

        z = (float(U1[j]) - mean_u) / np.sqrt(var_u)
        p = float(2.0 * stats.norm.sf(abs(z)))
        z_vals[j] = z
        p_vals[j] = p

    fdr = _bh_fdr(p_vals)

    if feature_names is None:
        feature_names = [str(i) for i in range(A_csc.shape[1])]

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "U1": U1,
            "z": z_vals,
            "p_value": p_vals,
            "fdr": fdr,
        }
    )
    return df


