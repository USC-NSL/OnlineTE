import numpy as np
from .. import get_global_precision
from .types import *
from numpy.typing import DTypeLike
from typing import Tuple, Optional


def cpu_frombuffer(buffer: bytes, shape: Tuple[int], dtype: Optional[DTypeLike] = None) -> CPUArray:
    """Alias for `np.frombuffer`, but expects the shape as an argument"""
    if dtype is None:
        return np.frombuffer(buffer, dtype=get_global_precision()).reshape(shape)
    else:
        return np.frombuffer(buffer, dtype=dtype).reshape(shape)


def cpu_frombuffer_serial(buffer: bytes, dtype: Optional[DTypeLike] = None) -> CPUArray:
    """Alias for `np.frombuffer`. Always returns a 1D array"""
    if dtype is None:
        return np.frombuffer(buffer, dtype=get_global_precision())
    else:
        return np.frombuffer(buffer, dtype=dtype)