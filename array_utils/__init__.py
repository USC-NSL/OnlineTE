from typing import Optional


"""
For very large topologies, GPU memory becomes very tight.
We need to switch to half precision for such cases.
"""

DOUBLE_PRECISION = 'double'  # float 64
SINGLE_PRECISION = 'single'  # float 32
HALF_PRECISION = 'half'      # float 16


_GLOBAL_PRECISION: Optional[str] = None
"""
Any algorithm that uses array features MUST explicitly set this, otherwise it will
get an error later (which we actually want, since it was probably never intended
to be this way)
"""

def set_global_precision(precision: str):
    global _GLOBAL_PRECISION
    assert _GLOBAL_PRECISION is None
    _GLOBAL_PRECISION = precision


def get_global_precision():
    global _GLOBAL_PRECISION
    assert _GLOBAL_PRECISION is not None
    return _GLOBAL_PRECISION


__all__ = [
    'DOUBLE_PRECISION', 'SINGLE_PRECISION', 'HALF_PRECISION',
    'set_global_precision', 'get_global_precision'
]
