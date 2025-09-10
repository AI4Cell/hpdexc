from typing import Literal, Any
from .backend.kernel import Alternative, Method, mannwhitney
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
    sparse_value: Any = 0
):
    match alternative:
        case "less":
            alternative = Alternative.less
        case "greater":
            alternative = Alternative.greater
        case "two_sided":
            alternative = Alternative.two_sided
        case _:
            raise ValueError(f"Invalid alternative: {alternative}")
    
    match method:
        case "auto":
            method = Method.auto
        case "exact":
            method = Method.exact
        case "asymptotic":
            method = Method.asymptotic
        case _:
            raise ValueError(f"Invalid method: {method}")
        
    result = mannwhitney(
        matrix,
        group_id,
        n_targets,
        tie_correction,
        use_continuity,
        alternative,
        method,
    )
    
    return result.U, result.P