import pandas as pd
import anndata as ad
import numpy as np
import scipy as sp
from scipy.stats import false_discovery_control
from .stats import mannwhitneyu_test as mannwhitneyu


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


def _vectorized_fold_change(
    μ_tgt: np.ndarray,
    μ_ref: np.ndarray,
    clip_value: float | int | None = 20,
) -> np.ndarray:
    """Vectorized calculation of fold change between two arrays of means."""
    result = np.zeros_like(μ_tgt)
    
    # Handle zero reference values
    zero_ref_mask = μ_ref == 0
    result[zero_ref_mask] = np.nan if clip_value is None else clip_value
    
    # Handle zero target values
    zero_tgt_mask = μ_tgt == 0
    result[zero_tgt_mask] = 0 if clip_value is None else 1 / clip_value
    
    # Calculate normal fold change for non-zero values
    normal_mask = ~(zero_ref_mask | zero_tgt_mask)
    result[normal_mask] = μ_tgt[normal_mask] / μ_ref[normal_mask]
    
    return result


def _vectorized_percent_change(
    μ_tgt: np.ndarray,
    μ_ref: np.ndarray,
) -> np.ndarray:
    """Vectorized calculation of percent change between two arrays of means."""
    result = np.zeros_like(μ_tgt)
    
    # Handle zero reference values
    zero_ref_mask = μ_ref == 0
    result[zero_ref_mask] = np.nan
    
    # Calculate normal percent change for non-zero values
    normal_mask = ~zero_ref_mask
    result[normal_mask] = (μ_tgt[normal_mask] - μ_ref[normal_mask]) / μ_ref[normal_mask]
    
    return result


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
            
            return pd.DataFrame(
                {
                    "target": groups,
                    "reference": reference,
                    "feature": adata.var_names,
                    "p_value": P,
                    "statistic": U
                }
            )
            
            # 预计算所有mask，避免重复计算
            reference_mask = (obs[groupby_key] == reference).values
            group_masks = {}
            for group in groups:
                group_masks[group] = (obs[groupby_key] == group).values
            
            # 使用预分配数组来构建结果，避免重复的字典创建
            n_genes = len(adata.var_names)
            n_groups = len(groups)
            total_results = n_groups * n_genes
            
            # 预分配结果数组
            targets = np.empty(total_results, dtype=object)
            references = np.empty(total_results, dtype=object)
            features = np.empty(total_results, dtype=object)
            target_means_arr = np.empty(total_results, dtype=float)
            reference_means_arr = np.empty(total_results, dtype=float)
            percent_changes_arr = np.empty(total_results, dtype=float)
            fold_changes_arr = np.empty(total_results, dtype=float)
            log2_fold_changes_arr = np.empty(total_results, dtype=float)
            p_values_arr = np.empty(total_results, dtype=float)
            statistics_arr = np.empty(total_results, dtype=float)
            
            # 预计算reference均值（只需要计算一次）
            reference_means = np.array(matrix[reference_mask, :].mean(axis=0)).flatten()
            
            result_idx = 0
            for group_idx, group in enumerate(groups, start=1):
                target_mask = group_masks[group]
                
                # 批量计算该组所有基因的均值
                target_means = np.array(matrix[target_mask, :].mean(axis=0)).flatten()
                
                # 向量化计算fold change和percent change
                fold_changes = _vectorized_fold_change(target_means, reference_means, clip_value)
                percent_changes = _vectorized_percent_change(target_means, reference_means)
                
                # 批量获取U和P值
                if U.ndim > 1:
                    u_vals = U[group_idx - 1, :]
                    p_vals = P[group_idx - 1, :]
                else:
                    u_vals = U
                    p_vals = P
                
                # 批量填充结果数组
                end_idx = result_idx + n_genes
                targets[result_idx:end_idx] = group
                references[result_idx:end_idx] = reference
                features[result_idx:end_idx] = adata.var_names
                target_means_arr[result_idx:end_idx] = target_means
                reference_means_arr[result_idx:end_idx] = reference_means
                percent_changes_arr[result_idx:end_idx] = percent_changes
                fold_changes_arr[result_idx:end_idx] = fold_changes
                
                # 安全计算log2 fold change
                log2_fc = np.where(fold_changes > 0, np.log2(fold_changes), np.nan)
                log2_fold_changes_arr[result_idx:end_idx] = log2_fc
                p_values_arr[result_idx:end_idx] = p_vals
                statistics_arr[result_idx:end_idx] = u_vals
                
                result_idx = end_idx
            
            # 转换为DataFrame
            df = pd.DataFrame({
                "target": targets,
                "reference": references,
                "feature": features,
                "target_mean": target_means_arr,
                "reference_mean": reference_means_arr,
                "percent_change": percent_changes_arr,
                "fold_change": fold_changes_arr,
                "log2_fold_change": log2_fold_changes_arr,
                "p_value": p_values_arr,
                "statistic": statistics_arr
            })
            
            # 进行FDR校正
            if not df.empty:
                df["q_value"] = false_discovery_control(df["p_value"].values)
            
            return df
            
        case _:
            raise ValueError(f"Invalid metric: {metric}")