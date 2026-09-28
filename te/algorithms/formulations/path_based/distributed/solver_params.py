from enum import StrEnum
import math
from typing import Optional, Literal
from dataclasses import dataclass
from te.algorithms.base import SolverParams
from array_utils import HALF_PRECISION, SINGLE_PRECISION
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
    """Maximum number of iterations for each switch-level problem"""
    SwitchOptimalityTolerance: Optional[float] = None
    """Relative Wolfe-gap tolerance for switch-level path QPs"""
    Precision: Literal['double', 'single'] = SINGLE_PRECISION
    """Floating point operation precision"""
    ScaleWithCapacity: bool = False
    """Scale everything with link capacities"""
    PathFile: Optional[str] = None
    """Path to a `PathProvider` object to use for paths"""
    NumberOfPathsPerCommodity: int = 8
    """Max number of available paths for each commodity"""
    AdjustGamma: bool = True
    """Whether to adjust PGD step size based on path lengths"""
    KernelThreads: Optional[int] = None
    """Native kernel threads; defaults to available worker CPUs capped at four"""
    def __post_init__(self):
        if self.Precision == HALF_PRECISION:
            raise ValueError(
                "The distributed native path solver supports only single and double precision"
            )
        if self.KernelThreads is not None and self.KernelThreads <= 0:
            raise ValueError("KernelThreads must be a positive integer or None")
        if self.SwitchIterations <= 0:
            raise ValueError("SwitchIterations must be a positive integer")
        if self.SwitchOptimalityTolerance is not None and (
            not math.isfinite(self.SwitchOptimalityTolerance) or
            not 0 < self.SwitchOptimalityTolerance < 1
        ):
            raise ValueError(
                "SwitchOptimalityTolerance must be finite and strictly between zero and one"
            )
        if self.Rho > self.Eta:
            as_warning(f"Outer ADMM step size (`Rho`) = {self.Rho} is strictly larger "
                       f"than inner ADMM step size (`Eta`) = {self.Eta}.\nThis is almost never beneficial.")


class WorkerUpdateData(StrEnum):
    DEMANDS = "demands"
    """Key for total demand satisfied until now"""
    NORM_F_DELTA_X = "norm_f_delta_X"
    """
    Key for the frobenius norm of `X^{m+1} - X^{m}` on the
    switch side.
    """
    DUAL_OBJ = "dual_obj"
    """Key for dual objective estimate"""
