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
    SwitchIterations: int = 5
    """Number of iterations for each switch-level problem"""
    Precision: Literal['double', 'single', 'half'] = SINGLE_PRECISION
    """Floating point operation precision"""
    ScaleWithCapacity: bool = False
    """Scale everything with link capacities"""
    PathFile: Optional[str] = None
    """Path to a `PathProvider` object to use for paths"""
    NumberOfPathsPerCommodity: int = 8
    """Max number of available paths for each commodity"""
    AdjustGamma: bool = True
    """Whether to adjust PGD step size based on path lengths"""
    def __post_init__(self):
        if self.Rho > self.Eta:
            as_warning(f"Outer ADMM step size (`Rho`) = {self.Rho} is strictly larger "
                       f"than inner ADMM step size (`Eta`) = {self.Eta}.\nThis is almost never beneficial.")
