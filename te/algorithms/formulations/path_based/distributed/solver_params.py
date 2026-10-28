from enum import StrEnum
import math
import warnings
import numpy as np
from typing import Optional, Literal
from dataclasses import dataclass
from te.algorithms.base import SolverParams
from array_utils import HALF_PRECISION, SINGLE_PRECISION
from utils.logging import as_warning


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


@dataclass(frozen=True)
class PathBasedOnlineTEParameters(SolverParams):
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
        if self.SwitchOptimalityTolerance is None or (
            not math.isfinite(self.SwitchOptimalityTolerance) or
            not 0 < self.SwitchOptimalityTolerance < 1
        ):
            raise ValueError(
                "SwitchOptimalityTolerance must be finite and strictly between zero and one"
            )
