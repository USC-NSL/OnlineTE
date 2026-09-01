import numpy as np
from .. import get_global_precision
from .types import *
from .sparse.types import *
from numpy.typing import DTypeLike
from typing import Any, Callable, Tuple, Optional

cpu_dump: Callable[[str, CPUArray], None] = lambda path, data: np.save(path, data, allow_pickle=True)
"""Replacement for Joblib `dump`, it seems to not do what I expect it to"""

cpu_double_array: Callable[[Any], DoublePrecisionCPUArray] = lambda input: np.array(input, dtype=np.float64)
"""Create a copy of an array-like thing that is always double precision, regardles of global data type"""
cpu_double_zeros: Callable[[Tuple[int]], DoublePrecisionCPUArray] = lambda shape: np.zeros(shape=shape, dtype=np.float64)
"""Always returns zero array with double precision, regardless of global data type"""

cpu_int_array: Callable[[Any], IntegerCPUArray] = lambda input: np.array(input, dtype=np.int32)
"""Create a copy of an array-like thing that is always `np.int32`, regardles of global data type"""
cpu_int_zeros: Callable[[Tuple[int]], IntegerCPUArray] = lambda shape: np.zeros(shape=shape, dtype=np.int32)
"""Always returns zero array with `np.int32`, regardless of global data type"""
cpu_int_fill: Callable[[Tuple[int], int], IntegerCPUArray] = lambda shape, fill: np.full(shape=shape, fill_value=fill, dtype=np.int32)
"""Always returns zero array with `np.int32`, regardless of global data type"""
cpu_bool_zeros: Callable[[Tuple[int]], BooleanCPUArray] = lambda shape: np.zeros(shape=shape, dtype=bool)
"""Always returns zero array with Boolean values, regardless of global data type"""


def cpu_mmap(path: str, shape: Tuple[int], mode: str, dtype: Optional[DTypeLike] = None):
    """Alias for MMAP"""
    if dtype is None:
        return np.lib.format.open_memmap(shape=shape, filename=path, mode=mode, dtype=get_global_precision())
    else:
        return np.lib.format.open_memmap(shape=shape, filename=path, mode=mode, dtype=dtype)


__all__ = [
    'cpu_dump', 'cpu_cast_float',
    'cpu_double_array', 'cpu_double_zeros',
    'cpu_int_array', 'cpu_int_zeros', 'cpu_int_fill',
    'cpu_bool_zeros',
    'cpu_mmap'
]