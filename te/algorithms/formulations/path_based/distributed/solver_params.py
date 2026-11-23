import math
from typing import Optional, Literal
from dataclasses import dataclass
from te.algorithms.base import SolverParams
from array_utils import SINGLE_PRECISION
from utils.logging import as_warning


@dataclass(frozen=True)
class PathBasedOnlineTEParameters(SolverParams):
    OuterLoopRounds: Optional[int] = 200
    """Number of outer loop iterations"""
    InnerLoopRounds: int = 5
    """Number of inner loop iterations"""
    Rho: float = 1.0
    """Outer ADMM step size"""
    Eta: float = 0.2
    """Inner ADMM step size"""
    Gamma: float = 0.1
    """Step size for solving the switch-level problems"""
    SwitchMaxIterations: int = 1000
    """Maximum gradient steps per switch update; warn if unconverged"""
    SwitchOptimalityTolerance: float = 1e-5
    """Switch Frank-Wolfe gap tolerance, averaged over commodities"""
    Precision: Literal['double', 'single', 'half'] = SINGLE_PRECISION
    """Floating point operation precision"""
    ScaleWithCapacity: bool = True
    """Normalize internal edge flows by capacity; report assignments in physical units"""
    PathFile: Optional[str] = None
    """Path to a `PathProvider` object to use for paths"""
    NumberOfPathsPerCommodity: int = 8
    """Max number of available paths for each commodity"""
    AdjustGamma: bool = True
    """Whether to adjust PGD step size based on path lengths"""
    def __post_init__(self):
        if self.SwitchMaxIterations <= 0:
            raise ValueError('SwitchMaxIterations must be positive')
        if not math.isfinite(self.SwitchOptimalityTolerance) or self.SwitchOptimalityTolerance <= 0:
            raise ValueError('SwitchOptimalityTolerance must be positive and finite')
        if self.Rho > self.Eta:
            as_warning(f"Outer ADMM step size (`Rho`) = {self.Rho} is strictly larger "
                       f"than inner ADMM step size (`Eta`) = {self.Eta}.\nThis is almost never beneficial.")
