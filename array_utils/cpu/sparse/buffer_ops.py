import numpy as np
import scipy.sparse as sp
from .types import *
from ..buffer_ops import cpu_frombuffer_serial
from numpy.typing import DTypeLike
from typing import Tuple, Optional


def cpu_csr_frombuffer(buffer: bytes, shape: Tuple[int], lens: Tuple[int], dtype: Optional[DTypeLike] = None) -> CPUCSRArray:
    """Given the 1D buffer of a CSR array, rebuilds it from the given lengths and shape"""
    assert len(lens) == 3
    d_len, i_len, p_len = lens
    assert len(buffer) == d_len + i_len + p_len
    data = cpu_frombuffer_serial(buffer[:d_len], dtype)
    indices = cpu_frombuffer_serial(buffer[d_len:d_len+i_len], np.int64)
    pointers = cpu_frombuffer_serial(buffer[d_len+i_len:], np.int64)
    return sp.csr_array((data, indices, pointers), shape=shape)
