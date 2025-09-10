from __future__ import annotations

from typing import Any, Iterable

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
    # Convert any other sparse type to CSC
    return sp.csc_matrix(matrix)


def _z_from_u_with_ties(u: float, x_ref: np.ndarray, x_tar: np.ndarray) -> float:
    n1 = x_ref.size
    n2 = x_tar.size
    if n1 == 0 or n2 == 0:
        return float("nan")
    x = np.concatenate([x_ref, x_tar])
    # tie correction using counts of equal values
    _, counts = np.unique(x, return_counts=True)
    t_corr = np.sum(counts ** 3 - counts) / ((n1 + n2) * (n1 + n2 - 1)) if (n1 + n2) > 1 else 0.0
    mean_u = n1 * n2 / 2.0
    var_u = n1 * n2 * (n1 + n2 + 1 - t_corr) / 12.0
    if var_u <= 0:
        return float("nan")
    return float((u - mean_u) / np.sqrt(var_u))


def sparse_wilcoxon_de(
    adata: Any,
    groups: list[str] | None = None,
    reference: str = "non-targeting",
    groupby_key: str = "target_gene",
    tie_correct: bool = True,
) -> pd.DataFrame:
    """Minimal sparse Wilcoxon rank-sum DE using CSC operator.

    - Operates directly on `adata.X` as sparse
    - Builds masks per group vs reference and calls `mwu_v1_stats_csc` once per comparison
    - Computes z and two-sided p-values using normal approximation with tie correction
    """

    X = adata.X  # type: ignore[attr-defined]
    if not sp.issparse(X):
        # allow dense input but convert to sparse for our operator
        X = sp.csr_matrix(X)
    A_csc = _ensure_csc(X)

    # collect groups
    obs_series = adata.obs[groupby_key]  # type: ignore[index]
    unique_targets = np.asarray(pd.Series(obs_series).unique())
    if groups is not None:
        unique_targets = np.asarray([g for g in unique_targets if g in set(groups + [reference])])

    # features list
    feature_names: Iterable[str] = getattr(adata.var, "index", None)  # type: ignore[attr-defined]
    if feature_names is None:
        # fallback: numeric indices as names
        feature_names = [str(i) for i in range(A_csc.shape[1])]
    feature_names = np.asarray(list(feature_names))

    results: list[dict[str, Any]] = []

    # precompute ref mask for reference label
    ref_mask_bool = (obs_series == reference).to_numpy()
    ref_mask = ref_mask_bool.astype(np.uint8)

    for target in unique_targets:
        if target == reference:
            continue
        tar_mask_bool = (obs_series == target).to_numpy()
        tar_mask = tar_mask_bool.astype(np.uint8)

        # call C++ operator once over all columns
        out = C.mwu_stats_csc(A_csc, ref_mask, tar_mask, calc_tie=tie_correct)
        U1 = np.asarray(out["U1"], dtype=np.float64)
        n1 = int(out["n1"])  # ref count
        n2 = int(out["n2"])  # tar count

        # compute z and p for each feature using tie correction via numpy
        # obtain column values cheaply from CSC
        for col_idx in range(A_csc.shape[1]):
            col = A_csc.getcol(col_idx)
            dense_col = np.asarray(col.toarray()).ravel()
            xr = dense_col[ref_mask_bool]
            xt = dense_col[tar_mask_bool]

            z = _z_from_u_with_ties(float(U1[col_idx]), xr, xt)
            if np.isfinite(z):
                p = float(2.0 * stats.norm.sf(abs(z)))
            else:
                p = float("nan")

            results.append(
                {
                    "group": target,
                    "feature": feature_names[col_idx],
                    "n_ref": n1,
                    "n_tar": n2,
                    "U1": float(U1[col_idx]),
                    "z": z,
                    "p_value": p,
                }
            )

    df = pd.DataFrame(results)
    if not df.empty:
        df["fdr"] = stats.false_discovery_control(df["p_value"].values, method="bh")  # type: ignore[attr-defined]
    return df

