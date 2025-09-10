from typing import Literal, Any
from .backend import mannwhitney
import scipy as sp
import numpy as np

def mannwhitneyu(
    matrix: sp.sparse.csc_matrix,
    group_id: np.array,
    n_targets: int,
    tie_correction: bool = True,
    use_continuity: bool = True,
    alternative: Literal["less", "greater", "two_sided"] = "two_sided",
    method: Literal["auto", "exact", "asymptotic"] = "asymptotic",
    ref_sorted: bool =False,
    tar_sorted: bool =False,
    use_sparse_value: bool =True,
    is_sparse_minmax: bool =False,
    sparse_value: Any = 0,
    threads: int = 1,
    show_progress: bool = False
):
    match alternative:
        case "less":
            alternative = 0
        case "greater":
            alternative = 1
        case "two_sided":
            alternative = 2
        case _:
            raise ValueError(f"Invalid alternative: {alternative}")
    
    match method:
        case "auto":
            method = 0
        case "exact":
            method = 1
        case "asymptotic":
            method = 2
        case _:
            raise ValueError(f"Invalid method: {method}")
        
    result = mannwhitney(
        csc_matrix=matrix,
        group_id=group_id,
        n_targets=n_targets,
        tie_correction=tie_correction,
        use_continuity=use_continuity,
        alternative=alternative,
        method=method,
        threads=threads,
        ref_sorted=ref_sorted,
        tar_sorted=tar_sorted,
        use_sparse_value=use_sparse_value,
        is_sparse_minmax=is_sparse_minmax,
        sparse_value=sparse_value,
        show_progress=show_progress
    )
    
    return result.U, result.P