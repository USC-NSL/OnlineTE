import te.constants
from typing import Optional, Literal
from dataclasses import dataclass
from multiprocessing import cpu_count
from te.algorithms.base import SolverParams
from utils.logging import as_warning
from array_utils import SINGLE_PRECISION


from utils.gurobi_utils import GurobiSolverParams # noqa


@dataclass(frozen=True)
class PDLPParams(SolverParams):
    """
    Solver parameters for `ortools.pdlp`.
    This is usally our preferred method for the controller backend, as 
    updating the objective is _MUCH_ faster than Gurobi when solving QPs.

    Attributes
    ----------
    Threads: int
        Number of threads to use for the PDHG backend.
    Presolve: bool
        Invoke `ortools.glop` to do a presolve on the problem.
        For very large problems, this is almost never worth it, and for
        smaller ones it is rather unpredicitable.
    ConvTol: float
        Objective convergence tolerance.
    FeasibilityTol: float
        Constraint violation tolerance. 
    """
    Threads: int = min(cpu_count(), 8)
    Presolve: bool = False
    ConvTol: float = te.constants.DEFAULT_OPTIMALITY_TOLERANCE
    FeasibilityTol: float = te.constants.DEFAULT_FEASIBILITY_TOLERANCE


@dataclass(frozen=True)
class GPUParams(SolverParams):
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
    Beta: Optional[float] = None
    """
    L1 norm penalty coefficient for sparsity.
    When `None`, a PGD algorithm on a dense assignment matrix is
    used to solve inner loop problems.
    If not `None`, then an alternating shrinkage algorithm is
    used to solve the inner loop problems instead.
    """
    SwitchIterations: int = 2
    """Number of iterations for each switch-level problem"""
    ConvTol: float = 1e-3
    """Objective convergence tolerance"""
    Precision: Literal['double', 'single', 'half'] = SINGLE_PRECISION
    """Floating point operation precision"""
    TMSeed: int = te.constants.DEFAULT_SEED
    """Traffic matrix RNG seed"""

    def __post_init__(self):
        if self.Beta is not None:
            assert self.Beta > 0, "L1 penalty coefficient must be strictly greater than 0"
        if self.Rho > self.Eta:
            as_warning(f"Outer ADMM step size (`Rho`) = {self.Rho} is strictly larger "
                       f"than inner ADMM step size (`Eta`) = {self.Eta}.\nThis is almost never beneficial.")
