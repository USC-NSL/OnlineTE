import numpy as np
from typing import Optional
from ortools.pdlp import solve_log_pb2
from ortools.pdlp import solvers_pb2
from ortools.pdlp.python import pdlp
from te.algorithms.base import TEObjective
from array_utils.cpu.types import *
from utils.pdlp_utils import *


class CapacityQP:
    def __init__(
        self,
        rho: float,         
        num_edges: int,
        num_commodities: int,
        feasibility_tolerance: float,
        optimality_tolerance: float,
        objective: TEObjective = TEObjective.MLU,
        capacities: Optional[CPUArray] = None,
        mean_scaled: bool = False
    ):
        self._rho: float = rho
        self._num_edges: int = num_edges
        self._num_commodities: int = num_commodities
        self._objective = objective
        self._feasibility_tolerance = feasibility_tolerance
        self._optimality_tolerance = optimality_tolerance
        self._capacities = np.array(capacities, dtype=np.float64) if capacities is not None else None
        self._mean_scaled = mean_scaled

        self._current_F: np.ndarray = None
        self._solved: bool = False
        self._lp: Optional[pdlp.QuadraticProgram] = None
        self._pdlp_params: Optional[solvers_pb2.PrimalDualHybridGradientParams] = \
            get_pdhg_params(
                PDLPSolverParams(Threads=1, Presolve=False),
                feasibility_tolerance=feasibility_tolerance,
                optimality_tolerance=optimality_tolerance,
                verbose=False
            )

        # `Z` has length `N * d` and we need one more variable for `u`
        self._NUM_VARIABLES: int = num_edges + 1
        # `N` capacity constraints are needed
        self._NUM_CONSTRAINTS: int = num_edges

        self._last_result: Optional[pdlp.SolverResult] = None

    @property
    def num_edges(self) -> int:
        return self._num_edges
    @property
    def objective_type(self) -> TEObjective:
        return self._objective
    @property
    def is_solved(self) -> bool:
        return self._solved
    @property
    def is_mlu(self) -> bool:
        return self._objective == TEObjective.MLU
    @property
    def current_u(self) -> float:
        return cpu_cast_float(self._last_result.primal_solution[-1])
    @property
    def current_Z(self) -> CPUArray:
        return cpu_array(self._last_result.primal_solution[:-1])
    @property
    def current_lambda(self) -> CPUArray:
        if self._last_result is None:
            return cpu_zeros((self.num_edges,))
        return cpu_array(self._last_result.dual_solution)
    @property
    def objective_value(self) -> float:
        if self._objective == TEObjective.MLU:
            return self.current_u if not self._mean_scaled else \
                self.current_u / self._num_commodities
    
    def _get_variable_lower_bound_vector(self) -> np.ndarray:
        out = np.full((self._NUM_VARIABLES,), -np.inf)
        if self.is_mlu:
            out[-1] = 0
        else:
            out[-1] = 1.0
        return out

    def _get_variable_upper_bound_vector(self) -> np.ndarray:
        out = np.full((self._NUM_VARIABLES,), np.inf)
        if not self.is_mlu:
            # Fix MLU to 1 for Max-Flow
            out[-1] = 1.0
        return out

    def _get_capacity_constraint(self) -> ConstraintVector:
        constraints = ConstraintVector.allocate(self._NUM_VARIABLES, self._NUM_CONSTRAINTS)
        caps = self._capacities

        N = self.num_edges
        for e in range(N):
            constraints.coeffs[e, e] = self._num_commodities
            constraints.coeffs[e, -1] = -caps[e]
        constraints.lowers.fill(-np.inf)
        return constraints
    
    def _get_objective_matrix_diagonal(self) -> np.ndarray:
        # d = np.full((self._NUM_VARIABLES,), fill_value=self._rho * self._num_commodities)
        d = np.full((self._NUM_VARIABLES,), fill_value=self._rho)
        d[-1] = 0
        return d
    
    def _get_objective_vector(self) -> np.ndarray:
        out = np.zeros((self._NUM_VARIABLES,))
        # out[:-1] = -self._current_F * self._rho * self._num_commodities
        out[:-1] = -self._current_F * self._rho
        if self.is_mlu:
            out[-1] = 1
        return out

    def update_F_m(self, new_F: CPUArray, rho: Optional[float] = None):
        if rho is not None:
            self._rho = rho
        self._current_F = np.array(new_F, dtype=np.float64).flatten()
        self._solved = False
        self._lp.set_objective_matrix_diagonal(self._get_objective_matrix_diagonal())
        self._lp.objective_vector = self._get_objective_vector()

    def build(self):
        assert self._lp is None
        LP = pdlp.QuadraticProgram()
        LP.variable_lower_bounds = self._get_variable_lower_bound_vector()
        LP.variable_upper_bounds = self._get_variable_upper_bound_vector()
        constraints = self._get_capacity_constraint()
        constraints.attach_to_program(LP)
        LP.set_objective_matrix_diagonal(self._get_objective_matrix_diagonal())
        LP.objective_offset = 0
        self._lp = LP
    
    def solve(self):
        assert not self._solved
        result: pdlp.SolverResult = pdlp.primal_dual_hybrid_gradient(self._lp, self._pdlp_params)
        if result.solve_log.termination_reason != solve_log_pb2.TERMINATION_REASON_OPTIMAL:
            raise RuntimeError(
                f"Solution did not terminate optimally.\n"
                f"Reason: {solve_log_pb2.TerminationReason.Name(result.solve_log.termination_reason)}"
            )
        self._last_result = result
        self._solved = True

    def close(self):
        pass
