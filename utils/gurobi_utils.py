import enum
import gurobipy
import te.constants
from dataclasses import dataclass
from te.algorithms.base import SolverParams
from typing import Optional
from .logging import as_warning, as_fail, as_info
from .exceptions import SolutionInterrupted


class GurobiMethod(enum.Enum):
    BARRIER = gurobipy.GRB.METHOD_BARRIER
    """
    Barrier is an interior point method. It fully takes advantage of
    all the cores in the system, but cannot be warm-started. You will
    pay the cost of a full solve, even if the problem didn't change
    at all.
    """
    SIM_PRIMAL = gurobipy.GRB.METHOD_PRIMAL
    """
    Primal Simplex solver. Can actually run into numerical issues for
    the max-flow problem. We recommend Dual Simplex the most. Regardless,
    it warm-starts easily, but for large problems will take a long time
    to solve on cold start.
    """
    SIM_DUAL = gurobipy.GRB.METHOD_DUAL
    """
    Dual Simplex solver. It warm-starts beutifully and generates very nice
    and sparse solutions. Similar to Primal Simplex, it can take a very long
    time to solve cold start.
    """
    COMPOSITE = gurobipy.GRB.METHOD_BARRIER
    """
    Probably the best scheme for our setting. The composite solver instructs
    our code to:
    - On the first run, use Barrier with Crossover enabled. This runs
      Dual Simplex iterations in the end to obtain a basic solution.
      The hope is that generating a feasible solution using the Barrier
      solution skips the majority of the Simplex iterations that we would
      have to do for cold start.
    - On every subsequent call, use Dual Simplex to warm-start.
    """

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
    Presolve: int
        Whether to allow for Gurobi to apply presolve to the model. In our experience,
        with Barrier in particular, it is not worth it and it is much better to go
        directly to the solver.
    Threads: int
        Number of threads to use for Barrier/Concurrent solver. Simplex methods do
        not benefit from multiple threads.
        We usually limit this to 8. For most of our problems, that seems to be the
        sweet-spot.
    LogFile:
        Output log file for Gurobi.
    """
    Method: GurobiMethod = GurobiMethod.BARRIER
    Crossover: int = te.constants.DEFAULT_CROSSOVER
    NumericFocus: int = te.constants.DEFAULT_NUMERIC_FOCUS
    Presolve: int = te.constants.DEFAULT_PRESOLVE
    Threads: int = min(te.constants.NUM_PROCS, 8)
    LogFile: str = te.constants.DEFAULT_GUROBI_LOG_FILE

    def __post_init__(self):
        if self.Method == GurobiMethod.COMPOSITE:
            assert self.Crossover > 0, "When using the composite solver, crossover _MUST_ be enabled!"


def make_model(
    name: str, params: GurobiSolverParams, feasibility_tolerance: float,
    optimality_tolerance: float , env: Optional[gurobipy.Env], 
    verbose: bool = True, **kwargs
):
    model = gurobipy.Model(name=name, env=env)
    model.Params.Method = params.Method.value
    model.Params.Crossover = params.Crossover
    model.Params.NumericFocus = params.NumericFocus

    if not verbose:
        model.Params.OutputFlag = 0

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

    if params.Method == GurobiMethod.COMPOSITE:
        print(as_info('COMPOSITE solver: First call with use barrier with crossover.'))

    return model


def optimize_or_scream(model: gurobipy.Model, warm_start: bool = False):
    """Solve a Gurobi model. Throw an error if the model ends up in any non-optimal state"""
    if warm_start:
        if model.Params.Method == gurobipy.GRB.METHOD_BARRIER:
            print(as_info('Switching the solver from BARRIER to DUAL SIMPLEX for warm-start.'))
            model.Params.Method = gurobipy.GRB.METHOD_DUAL
            model.update()
    model.optimize()
    if model.Status != gurobipy.GRB.OPTIMAL:
        if model.Status == gurobipy.GRB.INTERRUPTED:
            raise SolutionInterrupted
        else:
            raise RuntimeError(as_fail(f"Optimizing model {model.ModelName} returned non-optimal status: {model.Status}"))


__all__ = ['GurobiSolverParams', 'GurobiMethod', 'make_model', 'optimize_or_scream']