from typing import Any

import scipy as sp
import numpy as np


def mannwhitney(
    csc_matrix: sp.sparse.csc_matrix, 
    group_id: np.array,
    n_targets: int,
    tie_correction: bool = True,
    use_continuity: bool = True,
    alternative: int = 2,
    method: int = 2,
    ref_sorted: bool = False,
    tar_sorted: bool = False,
    use_sparse_value: bool = True,
    is_sparse_minmax: bool = False,
    sparse_value: Any =None,
    use_histogram: bool = False,
    max_bins: int = 65536,
    mem_budget_bytes: int = 1 << 30,
    threads: int = -1,
    progress: Any = None,
):
    ...

__all__ = ["mannwhitney"]