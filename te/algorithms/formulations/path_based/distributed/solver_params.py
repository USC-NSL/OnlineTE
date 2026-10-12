from enum import StrEnum
import math
import warnings
import numpy as np
from typing import Optional, Literal
from dataclasses import dataclass
from te.algorithms.base import SolverParams
from array_utils import HALF_PRECISION, SINGLE_PRECISION
from utils.logging import as_warning


@dataclass(frozen=True)
class PathBasedOnlineTEParameters(SolverParams):
    OuterLoopRounds: Optional[int] = None
    """Deprecated compatibility option; outer rounds are convergence/timeout driven"""
    InnerLoopRounds: int = 5
    """Number of inner loop iterations"""
    Rho: float = 1.0
    """Outer ADMM step size"""
    Eta: float = 0.2
    """Inner ADMM step size"""
    Gamma: float = 0.5
    """Step size for solving the switch-level problems"""
    MaxSwitchIterations: int = 1000
    """Safety cap for tolerance-driven switch-level problems"""
    SwitchOptimalityTolerance: float = 1e-4
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
    Timeout: Optional[float] = 20.0
    """Warm-start time budget in seconds, checked after complete outer rounds"""
    def __post_init__(self):
        if self.OuterLoopRounds is not None:
            warnings.warn("OuterLoopRounds is deprecated and ignored; use Timeout", FutureWarning, stacklevel=2)
        for name in ("InnerLoopRounds", "MaxSwitchIterations", "NumberOfPathsPerCommodity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("Rho", "Eta", "Gamma"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.Timeout is not None and (not math.isfinite(self.Timeout) or self.Timeout <= 0):
            raise ValueError("Timeout must be positive and finite or None")
        if self.Precision == HALF_PRECISION:
            raise ValueError(
                "The distributed native path solver supports only single and double precision"
            )
        if self.KernelThreads is not None and self.KernelThreads <= 0:
            raise ValueError("KernelThreads must be a positive integer or None")
        if self.MaxSwitchIterations <= 0:
            raise ValueError("SwitchIterations must be a positive integer")
        if self.SwitchOptimalityTolerance is None or (
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
    """Max-Flow objective value; omitted for MLU"""
    NORM_F_DELTA_X = "norm_f_delta_X"
    """
    Key for the frobenius norm of `X^{m+1} - X^{m}` on the
    switch side.
    """
    DUAL_OBJ = "dual_obj"
    """Key for dual objective estimate"""


def certificate_prices(outer_dual, rho, capacities, objective, num_commodities):
    """Dual-feasible prices in packed edge units; no demands are needed.

    MLU normalization removes the controller's objective multiplier alpha.
    Max-Flow prices use the native dual kernel's 1/K objective convention.
    """
    from te.algorithms.base import TEObjective
    prices = np.maximum(np.asarray(outer_dual, dtype=np.float64) * rho, 0.0)
    if not np.all(np.isfinite(prices)):
        raise FloatingPointError("Non-finite certificate prices")
    if objective == TEObjective.MLU:
        norm = float(prices @ capacities)
        if norm > 0:
            prices /= norm
        # Casting must not turn a feasible MLU dual into an infeasible one.
        prices = np.asarray(prices, dtype=outer_dual.dtype)
        norm = float(prices.astype(np.float64) @ capacities.astype(np.float64))
        if norm > 1:
            prices *= np.nextafter(prices.dtype.type(1 / norm), prices.dtype.type(0))
    elif objective == TEObjective.MAX_FLOW:
        prices = np.asarray(prices / num_commodities, dtype=outer_dual.dtype)
    else:
        raise NotImplementedError(objective)
    return np.ascontiguousarray(prices)


@dataclass(frozen=True)
class PathBasedSimplifiedOnlineTEParameters(SolverParams):
    Timeout: Optional[float] = 20.0
    """Timeout for warm-starts"""
    Rho: float = 1.0
    """(Initial) ADMM penalty parameter"""
    Gamma: float = 0.5
    """(Scaled) step size for solving the switch-level problems"""
    MaxSwitchIterations: int = 1000
    """Maximum number of iterations for each switch-level problem"""
    Precision: Literal['double', 'single'] = SINGLE_PRECISION
    """Floating point operation precision"""
    NumberOfPathsPerCommodity: int = 8
    """Max number of available paths for each commodity"""
    PathFile: Optional[str] = None
    """Path to a `PathProvider` object to use for paths"""
    KernelThreads: Optional[int] = None
    """Native kernel threads; defaults to available worker CPUs capped at four"""
    SwitchOptimalityTolerance: float = 1e-4
    def __post_init__(self):
        if self.Precision == HALF_PRECISION:
            raise ValueError(
                "The distributed native path solver supports only single and double precision"
            )
        if self.KernelThreads is not None and self.KernelThreads <= 0:
            raise ValueError("KernelThreads must be a positive integer or None")
        if self.MaxSwitchIterations <= 0:
            raise ValueError("MaxSwitchIterations must be a positive integer")
        if self.Gamma > 1.0:
            as_warning("Scaling the PGD step size to be larger than 1 may cause "
                       "zig-zags! Make sure you test cold-start behavior first!")
