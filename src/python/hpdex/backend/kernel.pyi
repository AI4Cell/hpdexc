from typing import Any, Callable, Optional, Tuple, Union

import scipy as sp
import numpy as np

# 暴露的 C++ 类型（基于 dtype 动态命名）
# 这些类型名称会根据实际的数据类型自动生成，例如：
# - Ndarray_float64, Ndarray_float32, Ndarray_int64, Ndarray_int32 等
# - Vector_int32, Vector_int64
# - Csc_float64_int32, Csc_float64_int64, Csc_float32_int32, Csc_float32_int64
# - MannWhitneyuOption_float64, MannWhitneyuOption_float32

# 通用类型声明（具体类型会在运行时根据 dtype 动态创建）
class Ndarray: ...  # 通用 Ndarray 类型
class Vector: ...   # 通用 Vector 类型  
class Csc: ...      # 通用 Csc 类型
class Csr: ...      # 通用 Csr 类型
class MannWhitneyuOption: ...  # 通用 MannWhitneyuOption 类型

# 常见的具体类型（用于类型提示）
class Ndarray_float64(Ndarray): ...  # double
class Ndarray_float32(Ndarray): ...  # float
class Ndarray_int64(Ndarray): ...    # int64_t
class Ndarray_int32(Ndarray): ...    # int32_t
class Ndarray_uint64(Ndarray): ...   # uint64_t
class Ndarray_uint32(Ndarray): ...   # uint32_t
class Ndarray_int16(Ndarray): ...    # int16_t
class Ndarray_uint16(Ndarray): ...   # uint16_t

class Vector_int32(Vector): ...
class Vector_int64(Vector): ...

class Csc_float64_int32(Csc): ...    # double, int32_t
class Csc_float64_int64(Csc): ...    # double, int64_t
class Csc_float32_int32(Csc): ...    # float, int32_t
class Csc_float32_int64(Csc): ...    # float, int64_t

class Csr_float64_int32(Csr): ...    # double, int32_t
class Csr_float64_int64(Csr): ...    # double, int64_t
class Csr_float32_int32(Csr): ...    # float, int32_t
class Csr_float32_int64(Csr): ...    # float, int64_t

class MannWhitneyuOption_float64(MannWhitneyuOption): ...  # double
class MannWhitneyuOption_float32(MannWhitneyuOption): ...  # float


def mannwhitneyu(
    csc_matrix: sp.sparse.csc_matrix,
    group_id: np.ndarray,
    n_targets: int,
    tie_correction: bool = True,
    use_continuity: bool = True,
    alternative: int = 2,
    method: int = 0,
    ref_sorted: bool = False,
    tar_sorted: bool = False,
    sparse_minmax: int = 0,
    sparse_value: Optional[Any] = None,
    threads: int = -1,
    progress: Optional[Union[np.ndarray, str]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
    progress_prefix: str = "progress",
    progress_interval_ms: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mann-Whitney U test for CSC sparse matrices.
    
    Args:
        csc_matrix: CSC sparse matrix
        group_id: Group ID array (int32)
        n_targets: Number of targets
        tie_correction: Apply tie correction
        use_continuity: Use continuity correction
        alternative: Alternative hypothesis (0=less, 1=greater, 2=two_sided)
        method: Method (0=automatic, 1=exact, 2=asymptotic)
        ref_sorted: Reference group is sorted
        tar_sorted: Target group is sorted
        sparse_minmax: Sparse value handling (0=none, 1=strict_minor, 2=strict_major, 3=medium)
        sparse_value: Sparse value to use
        threads: Number of threads (-1 for auto)
        progress: Progress mode (None, numpy array, "bar", or "cb")
        on_progress: Progress callback function
        progress_prefix: Progress bar prefix
        progress_interval_ms: Progress update interval in milliseconds
        
    Returns:
        Tuple of (U_statistics, p_values) arrays
    """
    ...

# Tensor 创建函数（类似 NumPy 接口）
def zeros(shape: list, dtype: str = "float64") -> np.ndarray:
    """
    Create a tensor filled with zeros.
    
    Args:
        shape: Shape of the tensor
        dtype: Data type ("float64", "float32", "int64", "int32", etc.)
        
    Returns:
        NumPy array filled with zeros
    """
    ...

def ones(shape: list, dtype: str = "float64") -> np.ndarray:
    """
    Create a tensor filled with ones.
    
    Args:
        shape: Shape of the tensor
        dtype: Data type ("float64", "float32", "int64", "int32", etc.)
        
    Returns:
        NumPy array filled with ones
    """
    ...

def full(shape: list, value: Any, dtype: str = "float64") -> np.ndarray:
    """
    Create a tensor filled with a specific value.
    
    Args:
        shape: Shape of the tensor
        value: Fill value
        dtype: Data type ("float64", "float32", "int64", "int32", etc.)
        
    Returns:
        NumPy array filled with the specified value
    """
    ...

def array(arr: np.ndarray) -> np.ndarray:
    """
    Create a tensor from a NumPy array.
    
    Args:
        arr: Input NumPy array
        
    Returns:
        Tensor created from the input array
    """
    ...

# Vector 创建函数
def vector_zeros(size: int, dtype: str = "int32") -> np.ndarray:
    """
    Create a vector filled with zeros.
    
    Args:
        size: Size of the vector
        dtype: Data type ("int32", "int64")
        
    Returns:
        NumPy array (1D) filled with zeros
    """
    ...

def vector_ones(size: int, dtype: str = "int32") -> np.ndarray:
    """
    Create a vector filled with ones.
    
    Args:
        size: Size of the vector
        dtype: Data type ("int32", "int64")
        
    Returns:
        NumPy array (1D) filled with ones
    """
    ...

def vector_full(size: int, value: Any, dtype: str = "int32") -> np.ndarray:
    """
    Create a vector filled with a specific value.
    
    Args:
        size: Size of the vector
        value: Fill value
        dtype: Data type ("int32", "int64")
        
    Returns:
        NumPy array (1D) filled with the specified value
    """
    ...

# 稀疏矩阵转换函数
def from_scipy_csc(csc_matrix: sp.sparse.csc_matrix) -> Union[Csc, Any]:
    """
    Convert scipy.sparse.csc_matrix to internal Csc tensor.
    
    Args:
        csc_matrix: SciPy CSC sparse matrix
        
    Returns:
        Internal Csc tensor object
    """
    ...

def from_scipy_csr(csr_matrix: sp.sparse.csr_matrix) -> Union[Csr, Any]:
    """
    Convert scipy.sparse.csr_matrix to internal Csr tensor.
    
    Args:
        csr_matrix: SciPy CSR sparse matrix
        
    Returns:
        Internal Csr tensor object
    """
    ...

def to_scipy_csc(csc_obj: Union[Csc, Any]) -> sp.sparse.csc_matrix:
    """
    Convert internal Csc tensor to scipy.sparse.csc_matrix.
    
    Args:
        csc_obj: Internal Csc tensor object
        
    Returns:
        SciPy CSC sparse matrix
    """
    ...

def to_scipy_csr(csr_obj: Union[Csr, Any]) -> sp.sparse.csr_matrix:
    """
    Convert internal Csr tensor to scipy.sparse.csr_matrix.
    
    Args:
        csr_obj: Internal Csr tensor object
        
    Returns:
        SciPy CSR sparse matrix
    """
    ...

def dense_to_csc(arr: np.ndarray, sparse_value: Optional[Any] = None) -> Union[Csc, Any]:
    """
    Convert dense NumPy array to CSC sparse tensor.
    
    Args:
        arr: Dense NumPy array (2D)
        sparse_value: Value to treat as sparse (default: 0)
        
    Returns:
        Internal Csc tensor object
    """
    ...

def dense_to_csr(arr: np.ndarray, sparse_value: Optional[Any] = None) -> Union[Csr, Any]:
    """
    Convert dense NumPy array to CSR sparse tensor.
    
    Args:
        arr: Dense NumPy array (2D)
        sparse_value: Value to treat as sparse (default: 0)
        
    Returns:
        Internal Csr tensor object
    """
    ...

def group_mean(csc_matrix: Union[Csc, Any], group_id: np.ndarray, n_groups: int, 
               threads: int = 1, progress: Optional[Union[np.ndarray, Any]] = None) -> np.ndarray:
    """
    Calculate group means for sparse CSC matrix.
    
    Args:
        csc_matrix: Internal Csc tensor object
        group_id: 1D numpy array of group IDs (int32)
        n_groups: Number of groups
        threads: Number of threads to use (default: 1)
        progress: Optional progress tracking array (size_t type)
        
    Returns:
        NumPy array of group means (n_groups × n_features)
    """
    ...


__all__ = [
    "mannwhitneyu",
    # Tensor 创建函数（类似 NumPy）
    "zeros", "ones", "full", "array",
    # Vector 创建函数
    "vector_zeros", "vector_ones", "vector_full",
    # 稀疏矩阵转换函数
    "from_scipy_csc", "from_scipy_csr", "to_scipy_csc", "to_scipy_csr",
    "dense_to_csc", "dense_to_csr",
    # 统计分析函数
    "group_mean",
    # 通用类型
    "Ndarray", "Vector", "Csc", "Csr", "MannWhitneyuOption",
    # 具体类型（基于 dtype 动态命名）
    "Ndarray_float64", "Ndarray_float32", "Ndarray_int64", "Ndarray_int32",
    "Ndarray_uint64", "Ndarray_uint32", "Ndarray_int16", "Ndarray_uint16",
    "Vector_int32", "Vector_int64",
    "Csc_float64_int32", "Csc_float64_int64", "Csc_float32_int32", "Csc_float32_int64",
    "Csr_float64_int32", "Csr_float64_int64", "Csr_float32_int32", "Csr_float32_int64",
    "MannWhitneyuOption_float64", "MannWhitneyuOption_float32"
]