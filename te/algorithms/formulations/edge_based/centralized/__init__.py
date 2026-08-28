import enum
import gurobipy
import te.constants
from typing import Optional, Literal
from dataclasses import dataclass
from multiprocessing import cpu_count
from te.algorithms.base import SolverParams
from utils.exceptions import SolutionInterrupted
from utils.logging import as_warning, as_fail, as_info
from te.algorithms.array_utils import SINGLE_PRECISION


class GurobiMethod(enum.Enum):
    BARRIER = gurobipy.GRB.METHOD_BARRIER
    SIM_PRIMAL = gurobipy.GRB.METHOD_PRIMAL
    SIM_DUAL = gurobipy.GRB.METHOD_DUAL

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class GurobiSolverParams(SolverParams):
    """
    Solver parameters for Gurobi.

    Attributes
    ----------
    Method: GurobiMethod
        The Gurobi solver method to be used
    Crossover: int
        Whether to use Simplex crossover after finishing Barrier iterations. Since
        Gurobi epxects `int`s, we use `int` instead of `bool`.
    NumericFocus: int
        How careful should Gurobi be about numerical errors. See Gurobi docs for
        how to set them. We have noted that Dual Simplex in particular might need
        to have this with a higher value when solving MLU on large topologies.
    ConvTol: float
        Objective convergence tolerance (used for _ALL_ algorithms, not just Barrier).
    FeasibilityTol: float
        Constraint violation tolerance. 
    Presolve: int
        Whether to allow for Gurobi to apply presolve to the model. In our experience,
        with Barrier in particular, it is not worth it and it is much better to go
        directly to the solver.
    Threads: int
        Number of threads to use for Barrier/Concurrent solver. Simplex methods do
        not benefit from multiple threads.
    LogFile:
        Output log file for Gurobi.
    """
    # Method: int = te.constants.DEFAULT_SOLVER_METHOD
    Method: GurobiMethod = GurobiMethod.BARRIER
    Crossover: int = te.constants.DEFAULT_CROSSOVER
    NumericFocus: int = te.constants.DEFAULT_NUMERIC_FOCUS
    # ConvTol: float = te.constants.DEFAULT_OPTIMALITY_TOLERANCE
    # FeasibilityTol: float = te.constants.DEFAULT_FEASIBILITY_TOLERANCE
    Presolve: int = te.constants.DEFAULT_PRESOLVE
    Threads: int = min(cpu_count(), 8)
    LogFile: str = te.constants.DEFAULT_GUROBI_LOG_FILE


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


def make_model(name: str, params: SolverParams, feasibility_tolerance: float,
               optimality_tolerance: float , env: Optional[gurobipy.Env], 
               verbose: bool = True, **kwargs):
    assert issubclass(params.__class__, GurobiSolverParams)
    model = gurobipy.Model(name=name, env=env)
    model.Params.Method = params.Method.value
    model.Params.Crossover = params.Crossover
    model.Params.NumericFocus = params.NumericFocus

    # We set _both_ of these to the same value to make sure that both
    # Barrier and Simplex converge to within the same tolerance.
    model.Params.BarConvTol = optimality_tolerance
    model.Params.OptimalityTol = optimality_tolerance

    model.Params.FeasibilityTol = feasibility_tolerance
    model.Params.LogFile = params.LogFile
    model.Params.Presolve = params.Presolve

    model.Params.Threads = params.Threads

    if len(kwargs) > 0:
        for k, v in kwargs.items():
            setattr(model.Params, k, v)

    if verbose:
        print(as_info(
            "Created Gurobi Model With:\n"
            f"\tMethod: {params.Method}\n"
            f"\tSimplex Optimality Tolerance (OptimalityTol): {model.Params.OptimalityTol}\n"
            f"\tBarrier Optimality Tolerance (BarConvTol): {model.Params.BarConvTol}\n"
            f"\tCosntraint Feasibility Tolerance (FeasibilityTol): {model.Params.FeasibilityTol}\n"
        ))
    
    if model.Params.OptimalityTol != model.Params.BarConvTol:
        print(as_warning(f'Simplex and Barrier have different convergence tolerances. Make sure this is actualy intended to be!!'))

    return model


def optimize_or_scream(model: gurobipy.Model):
    """Solve a Gurobi model. Throw an error if the model ends up in any non-optimal state"""
    model.optimize()
    if model.Status != gurobipy.GRB.OPTIMAL:
        if model.Status == gurobipy.GRB.INTERRUPTED:
            raise SolutionInterrupted
        else:
            raise RuntimeError(as_fail(f"Optimizing model {model.ModelName} returned non-optimal status: {model.Status}"))