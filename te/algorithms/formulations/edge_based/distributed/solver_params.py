import math
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
    SwitchFeasibilityTolerance: float = 1e-5
    """Absolute dense switch feasibility tolerance, in solver units"""
    SwitchOptimalityTolerance: float = 1e-5
    """Dense switch complementarity gap tolerance, averaged over commodities"""
    SwitchMaxIterations: int = 1000
    """Maximum gradient steps per dense switch update; warn if unconverged"""

    def __post_init__(self):
        if self.SwitchMaxIterations <= 0:
            raise ValueError('SwitchMaxIterations must be positive')
        if not math.isfinite(self.SwitchFeasibilityTolerance) or self.SwitchFeasibilityTolerance <= 0:
            raise ValueError('SwitchFeasibilityTolerance must be positive and finite')
        if not math.isfinite(self.SwitchOptimalityTolerance) or self.SwitchOptimalityTolerance <= 0:
            raise ValueError('SwitchOptimalityTolerance must be positive and finite')
        if self.Beta is None and not 0 < self.Gamma <= 1:
            raise ValueError('Dense switch Gamma must satisfy 0 < Gamma <= 1')
        if self.Beta is not None:
            assert self.Beta > 0, "L1 penalty coefficient must be strictly greater than 0"
        if self.Rho > self.Eta:
            as_warning(f"Outer ADMM step size (`Rho`) = {self.Rho} is strictly larger "
                       f"than inner ADMM step size (`Eta`) = {self.Eta}.\nThis is almost never beneficial.")
