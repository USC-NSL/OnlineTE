import numpy as np
import te.constants
import scipy.sparse as sp
from dataclasses import dataclass
from te.algorithms.base import SolverParams
from ortools.pdlp import solve_log_pb2
from ortools.pdlp import solvers_pb2
from ortools.pdlp.python import pdlp
from .logging import as_fail


@dataclass(frozen=True)
class PDLPSolverParams(SolverParams):
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
    """
    Threads: int = min(te.constants.NUM_PROCS, 8)
    Presolve: bool = False


@dataclass
class ConstraintVector:
    """
    Encodes as 3-tuple, the elements of:
    - constraint coefficients
    - constraint lower bounds
    - constraint upper bounds
    Coefficients are kept as a sparse matrix, but bounds are dense.
    """
    coeffs: sp.lil_matrix
    lowers: np.ndarray
    uppers: np.ndarray

    @classmethod
    def allocate(cls, n_vars, n_constraints):
        return cls(
            sp.lil_matrix((n_constraints, n_vars)),
            np.zeros((n_constraints,)), np.zeros((n_constraints,))
        )
    
    def attach_to_program(self, lp: pdlp.QuadraticProgram):
        lp.constraint_matrix = self.coeffs.tocsc()
        lp.constraint_lower_bounds = self.lowers
        lp.constraint_upper_bounds = self.uppers

    def update_bounds(self, lp: pdlp.QuadraticProgram):
        lp.constraint_lower_bounds = self.lowers
        lp.constraint_upper_bounds = self.uppers


def solve_qp_or_scream(
    qp: pdlp.QuadraticProgram,
    params: PDLPSolverParams,
    feasibility_tolerance: float,
    optimality_tolerance: float,
    verbose: bool = True
) -> pdlp.SolverResult:
    PDHG_PARAMS = solvers_pb2.PrimalDualHybridGradientParams()
    PDHG_PARAMS.termination_criteria\
        .simple_optimality_criteria\
        .eps_optimal_relative = optimality_tolerance
    PDHG_PARAMS.termination_criteria\
        .eps_primal_infeasible = feasibility_tolerance
    PDHG_PARAMS.termination_criteria\
        .eps_dual_infeasible = feasibility_tolerance
    PDHG_PARAMS.termination_criteria.time_sec_limit = np.inf
    PDHG_PARAMS.num_threads = params.Threads
    PDHG_PARAMS.presolve_options.use_glop = params.Presolve
    PDHG_PARAMS.verbosity_level = 3 if verbose else 0

    try:
        result: pdlp.SolverResult = pdlp.primal_dual_hybrid_gradient(qp, PDHG_PARAMS)
        if result.solve_log.termination_reason == solve_log_pb2.TERMINATION_REASON_OPTIMAL:
            return result
        raise RuntimeError(as_fail(
            f"PDLP returned non-optimal status: {solve_log_pb2.TerminationReason.Name(result.solve_log.termination_reason)}"
        ))
    except Exception as e:
        raise RuntimeError(as_fail(f"Error while solving: {e}"))