import pandas as pd
import anndata as ad
import numpy as np
import scipy as sp
from scipy.stats import false_discovery_control
from .stats import mannwhitneyu


def _fold_change(
    μ_tgt: float,
    μ_ref: float,
    clip_value: float | int | None = 20,
) -> float:
    """Calculate the fold change between two means."""
    # The fold change is infinite so clip to default value
    if μ_ref == 0:
        return np.nan if clip_value is None else clip_value

    # The fold change is zero so clip to 1 / default value
    if μ_tgt == 0:
        return 0 if clip_value is None else 1 / clip_value

    # Return the fold change
    return μ_tgt / μ_ref


def _percent_change(
    μ_tgt: float,
    μ_ref: float,
) -> float:
    """Calculate the percent change between two means."""
    if μ_ref == 0:
        return np.nan
    return (μ_tgt - μ_ref) / μ_ref


def parallel_differential_expression(
    adata: ad.AnnData,
    groups: list[str] | None = None,
    reference: str | None = "non-targeting", # None -> pd.na
    groupby_key: str = "target_gene",
    threads: int = 8,
    metric: str = "wilcoxon",
    tie_correction: bool = True,
    use_continuity: bool = True,
    clip_value: float | int | None = 20.0,
    show_progress: bool = False,
) -> pd.DataFrame:
    if groupby_key not in adata.obs.columns:
        raise ValueError(f"Groupby key {groupby_key} not found in adata.obs")
    if reference is not None and reference not in adata.obs[groupby_key].unique().tolist():
        raise ValueError(f"Reference group {reference} not found in adata.obs[{groupby_key}]")
    
    supported_metrics = ["wilcoxon"]

    if groups is None:
        groups = adata.obs[groupby_key].unique().tolist()
    else:
        groups = list(set(groups))
        
    if reference is not None and reference in groups:
        groups.remove(reference)

    obs = adata.obs
    if reference is None:
        obs = adata.obs.copy()
        if "non-targeting" not in obs[groupby_key].unique():
            obs[groupby_key].cat.add_categories("non-targeting")
        obs[groupby_key].fillna("non-targeting", inplace=True)
        reference = "non-targeting"

    match metric:
        case "wilcoxon":
            group_str2id = {group: i for i, group in enumerate(groups, start=1)}
            group_str2id[reference] = 0
            group_id = np.array([-1] * adata.n_obs).astype(np.int32)
            for group in groups:
                mask = obs[groupby_key] == group
                group_id[mask] = group_str2id[group]
            
            matrix = adata.X
            if not isinstance(matrix, sp.sparse.csc_matrix):
                matrix = sp.sparse.csc_matrix(matrix)
            U, P = mannwhitneyu(matrix,
                                group_id,
                                len(groups),
                                tie_correction,
                                use_continuity,
                                "two_sided",
                                "auto",
                                False,
                                False,
                                True,
                                False,
                                0, # sparse see as 0 in de analysis
                                threads,
                                show_progress=show_progress)
            
            results = []
            reference_mask = (obs[groupby_key] == reference).values
            
            for group_idx, group in enumerate(groups, start=1):
                target_mask = (obs[groupby_key] == group).values
                
                for var_idx, var_name in enumerate(adata.var_names):
                    # calculate mean
                    x_tgt = matrix[target_mask, var_idx]
                    x_ref = matrix[reference_mask, var_idx]
                    
                    μ_tgt = x_tgt.mean()
                    μ_ref = x_ref.mean()
                    
                    # fold change and percent change
                    fc = _fold_change(μ_tgt, μ_ref, clip_value=clip_value)
                    pcc = _percent_change(μ_tgt, μ_ref)
                    
                    # 获取对应的U和P值
                    u_val = U[group_idx - 1, var_idx] if U.ndim > 1 else U[var_idx]
                    p_val = P[group_idx - 1, var_idx] if P.ndim > 1 else P[var_idx]
                    
                    results.append({
                        "target": group,
                        "reference": reference,
                        "feature": var_name,
                        "target_mean": μ_tgt,
                        "reference_mean": μ_ref,
                        "percent_change": pcc,
                        "fold_change": fc,
                        "log2_fold_change": np.log2(fc),
                        "p_value": p_val,
                        "statistic": u_val
                    })
            
            # 转换为DataFrame
            df = pd.DataFrame(results)
            
            # 进行FDR校正
            if not df.empty:
                df["q_value"] = false_discovery_control(df["p_value"].values)
            
            return df
            
        case _:
            raise ValueError(f"Invalid metric: {metric}")