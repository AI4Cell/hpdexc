from typing import Any, Optional, Union
import numpy as np
import scipy.sparse as sp

def adaptive_sort(
    a: np.ndarray,
    reverse: bool = False,
    prefer_stable: bool = False,
    threads: int = -1
) -> None: ...

def mannwhitney(
    sparse_matrix: sp.csc_matrix,
    group_id: np.ndarray,
    n_targets: int,
    tie_correction: bool = True,
    use_continuity: bool = True,
    alternative: int = 2,
    method: int = 0,
    ref_sorted: bool = False,
    tar_sorted: bool = False,
    use_sparse_value: bool = False,
    is_sparse_minmax: bool = False,
    sparse_value: Optional[Union[int, float]] = None,
    use_histogram: bool = False,
    max_bins: int = 65536,
    mem_budget_bytes: int = 1073741824,
    threads: int = -1
) -> Any: ...

__all__ = ["adaptive_sort", "mannwhitney"]


