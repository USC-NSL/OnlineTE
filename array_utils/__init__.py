import numpy as np
from typing import Optional
from numpy.typing import DTypeLike

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)
"""This is mostly to catch overflow, they can be devistating!"""


"""
For very large topologies, GPU memory becomes very tight.
We need to switch to half precision for such cases.
"""

DOUBLE_PRECISION = 'double'  # float 64
SINGLE_PRECISION = 'single'  # float 32
HALF_PRECISION = 'half'      # float 16


_GLOBAL_PRECISION: Optional[DTypeLike] = None
"""
Any algorithm that uses array features MUST explicitly set this, otherwise it will
get an error later (which we actually want, since it was probably never intended
to be this way)
"""

def set_global_precision(precision: str):
    global _GLOBAL_PRECISION
    if _GLOBAL_PRECISION == precision:
        return
    assert _GLOBAL_PRECISION is None
    _GLOBAL_PRECISION =  ({
        DOUBLE_PRECISION: np.float64,
        SINGLE_PRECISION: np.float32,
        HALF_PRECISION: np.float16
    })[precision]


def get_global_precision():
    global _GLOBAL_PRECISION
    if _GLOBAL_PRECISION is None:
        set_global_precision(DOUBLE_PRECISION)
    return _GLOBAL_PRECISION


__all__ = [
    'DOUBLE_PRECISION', 'SINGLE_PRECISION', 'HALF_PRECISION',
    'set_global_precision', 'get_global_precision'
]
