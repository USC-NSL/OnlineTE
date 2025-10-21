import numpy as np
import scipy.sparse
from dataclasses import dataclass
from typing import Optional
from ortools.pdlp import solve_log_pb2
from ortools.pdlp import solvers_pb2
from ortools.pdlp.python import pdlp
from te.algorithms.base import SolverParams
from te.algorithms.utils import as_info
from te.algorithms.array_utils.cpu_utils import CPUArray, cpu_array, cpu_cast_float
from .mlu_base import ControllerMLUSolver, ControllerMLUException
from te.algorithms.statistics.helpers import record_cpu_runtime


@dataclass
class PDLPMLUParams(SolverParams):
    Threads: int
    Presolve: bool
    ConvTol: float
    Rho: float
    Alpha: float

    def __post_init__(self):
        self._left_column_share = 0.5

"""
Just like Gurobi, PDLP expects double-precision arrays ...
"""


class PDLPMLU(ControllerMLUSolver):
    def __init__(self, num_edges: int, capacities: CPUArray, solver_params: PDLPMLUParams):
        self._num_edges: int = num_edges
        self._capacities: np.ndarray = np.array(capacities, dtype=np.float64)
        self._solver_params = solver_params

        self._current_F: np.ndarray = None
        self._solved: bool = False
        self._lp: Optional[pdlp.QuadraticProgram] = None
        self._pdlp_params: Optional[solvers_pb2.PrimalDualHybridGradientParams] = None

        # `Z` has length `N` and we need one more variable for `u`
        self._NUM_VARIABLES: int = num_edges + 1
        # `N` capacity constraints are needed
        self._NUM_CONSTRAINTS: int = num_edges

        self._last_result: Optional[pdlp.SolverResult] = None

    @property
    def num_edges(self) -> int:
        return self._num_edges
    @property
    def capacities(self) -> CPUArray:
        return self._capacities
    @property
    def solver_params(self) -> PDLPMLUParams:
        return self._solver_params
    @property
    def is_solved(self) -> bool:
        return self._solved
    
    def _get_variable_lower_bound_vector(self) -> np.ndarray:
        out = np.full((self._NUM_VARIABLES,), -np.inf)
        out[-1] = 0
        return out

    def _get_variable_upper_bound_vector(self) -> np.ndarray:
        out = np.full((self._NUM_VARIABLES,), np.inf)
        out[-1] = 1.0
        return out

    def _get_capacity_constraint_matrix(self) -> np.ndarray:
        coeff = np.zeros(shape=(self.num_edges, self._NUM_VARIABLES))

        for e in range(self.num_edges):
            coeff[e, e] = 1.0
            coeff[e, -1] = -self.capacities[e]
        
        return coeff

    def _get_capacity_constraint_lower_bound(self) -> np.ndarray:
        return np.full(shape=(self._NUM_CONSTRAINTS,), fill_value=-np.inf)
    
    def _get_capacity_constraint_upper_bound(self) -> np.ndarray:
        return np.zeros(shape=(self._NUM_CONSTRAINTS,))
    
    def _get_objective_matrix_diagonal(self) -> np.ndarray:
        d = np.full((self._NUM_VARIABLES,), fill_value=self._solver_params.Rho)
        d[-1] = 0
        return d
    
    def _get_objective_vector(self) -> np.ndarray:
        out = np.zeros((self._NUM_VARIABLES,))
        out[:-1] = -self._current_F * self._solver_params.Rho
        out[-1] = self._solver_params.Alpha
        return out

    def update_F_m(self, new_F: CPUArray):
        self._current_F = np.array(new_F, dtype=np.float64)
        self._solved = False
        self._lp.objective_vector = self._get_objective_vector()
    
    def _make_variables(self):
        assert self._lp is None

        LP = pdlp.QuadraticProgram()
        SOLVER_PARAMS: PDLPMLUParams = self._solver_params
        PDHG_PARAMS = solvers_pb2.PrimalDualHybridGradientParams()

        optimality_criteria = PDHG_PARAMS.termination_criteria.simple_optimality_criteria
        optimality_criteria.eps_optimal_relative = SOLVER_PARAMS.ConvTol
        PDHG_PARAMS.termination_criteria.time_sec_limit = np.inf
        PDHG_PARAMS.num_threads = SOLVER_PARAMS.Threads
        PDHG_PARAMS.presolve_options.use_glop = SOLVER_PARAMS.Presolve
        PDHG_PARAMS.verbosity_level = 0

        self._lp = LP
        print(as_info(f"PDLP objective convergence tolerance: {SOLVER_PARAMS.ConvTol}"))
        self._pdlp_params = PDHG_PARAMS
    
    def _add_constraints(self):
        assert self._lp is not None
        LP = self._lp
        # Lower and upper variable bounds
        LP.variable_lower_bounds = self._get_variable_lower_bound_vector()
        LP.variable_upper_bounds = self._get_variable_upper_bound_vector()
        # Capacity constraints
        LP.constraint_matrix = scipy.sparse.csc_matrix(self._get_capacity_constraint_matrix())
        LP.constraint_lower_bounds = self._get_capacity_constraint_lower_bound()
        LP.constraint_upper_bounds = self._get_capacity_constraint_upper_bound()
    
    def _add_objective(self):
        assert self._lp is not None
        LP = self._lp
        LP.set_objective_matrix_diagonal(self._get_objective_matrix_diagonal())
        LP.objective_offset = 0

    @property
    def current_u(self) -> float:
        return cpu_cast_float(self._last_result.primal_solution[-1])
    @property
    def current_Z(self) -> CPUArray:
        return cpu_array(self._last_result.primal_solution[:-1])
    
    def close(self):
        pass

    def reset(self, with_params):
        self._lp = None
        if with_params:
            self._pdlp_params = None
    
    @record_cpu_runtime('PDLP-MLU')
    def solve(self):
        assert not self._solved
        result: pdlp.SolverResult = pdlp.primal_dual_hybrid_gradient(self._lp, self._pdlp_params)
        if result.solve_log.termination_reason != solve_log_pb2.TERMINATION_REASON_OPTIMAL:
            raise ControllerMLUException(
                'PDLP', 
                RuntimeError(f"Solution did not terminate optimally. Reason: {solve_log_pb2.TerminationReason.Name(result.solve_log.termination_reason)}")
            )
        self._last_result = result
        self._solved = True
