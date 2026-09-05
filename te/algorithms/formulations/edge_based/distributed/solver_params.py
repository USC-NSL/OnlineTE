from typing import Optional, Literal
from dataclasses import dataclass
from te.algorithms.base import SolverParams
from array_utils import SINGLE_PRECISION
from utils.logging import as_warning


@dataclass(frozen=True)
class EdgeBasedOnlineTEParameters(SolverParams):
    OuterLoopRounds: Optional[int] = 100
    """Number of outer loop iterations"""
    InnerLoopRounds: int = 3
    """Number of inner loop iterations"""
    Rho: float = 1.0
    """Outer ADMM step size"""
    Eta: float = 0.2
    """Inner ADMM step size"""
    Gamma: float = 1.0
    """Step size for solving the switch-level problems"""
    SwitchIterations: int = 2
    """Number of iterations for each switch-level problem"""
    Precision: Literal['double', 'single', 'half'] = SINGLE_PRECISION
    """Floating point operation precision"""
    ScaleWithCapacity: bool = False
    """Scale everything with link capacities"""
    Beta: Optional[float] = None
    """
    L1 norm penalty coefficient for sparsity.
    When `None`, a PGD algorithm on a dense assignment matrix is
    used to solve inner loop problems.
    If not `None`, then an alternating shrinkage algorithm is
    used to solve the inner loop problems instead.
    """

    def __post_init__(self):
        if self.Beta is not None:
            assert self.Beta > 0, "L1 penalty coefficient must be strictly greater than 0"
        if self.Rho > self.Eta:
            as_warning(f"Outer ADMM step size (`Rho`) = {self.Rho} is strictly larger "
                       f"than inner ADMM step size (`Eta`) = {self.Eta}.\nThis is almost never beneficial.")
