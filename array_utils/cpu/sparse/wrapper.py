import numpy as np
import scipy.sparse as sp
from .types import *
from .. import _CPU_DTYPE
from typing import List, Callable, Any, Tuple


def cpu_coo_array(rows: List[int], cols: List[int], data: List[Any], shape: Tuple[int]) -> CPUCOOArray:
    if _CPU_DTYPE != np.float16:
        return sp.coo_array((data, (rows, cols)), shape=shape, dtype=_CPU_DTYPE)
    raise ValueError('Sparse half-precision operations not yet supported!')

cpu_coo_to_csr: Callable[[CPUCOOArray], CPUCSRArray] = lambda inp: inp.tocsr() if isinstance(inp, CPUCOOArray) else inp
cpu_coo_to_csc: Callable[[CPUCOOArray], CPUCSCArray] = lambda inp: inp.tocsc() if isinstance(inp, CPUCOOArray) else inp
