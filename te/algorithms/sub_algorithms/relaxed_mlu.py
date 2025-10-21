import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from te.algorithms.base import SolverParams
from te.algorithms.array_utils.cpu_utils import CPUArray, cpu_zeros


@dataclass
class RelaxedMLUSolverParams(SolverParams):
    Rho: float
    Alpha: float
    Gamma: float
    GDIters: int


class RelaxedMLU:
    """
    Implements the dual optimization problem of the QP relation of MLU.
    The QP relaxation is:

        min alpha * u^2 + | sum_k X_k - Z + r |_2^2
       s.t. 0 <= u <= 1
            Z <= uC
    
    Note that we have `alpha * u^2` instead of `alpha * u`. This helps in two ways:
        - It aids convergence, especially if we are far from the optimal `u`
        - It makes the dual problem much easier to solve, as it get rids of the linear
          constraint that remains on the dual variables of `u`.
          Without this, we would have to use method of multipliers or ADMM to solve this.
    """
    BETA = 0
    def __init__(self, num_edges: int, capacities: CPUArray, solver_params: RelaxedMLUSolverParams):
        self._num_edges: int = num_edges
        self._capacities: CPUArray = capacities
        self._solver_params = solver_params

        self._v_minus: Optional[float] = None
        self._v_plus: Optional[float] = None
        self._tau: Optional[CPUArray] = None
        self._last_v_minus: Optional[float] = None
        self._last_v_plus: Optional[float] = None
        self._last_tau: Optional[CPUArray] = None

        self._recent_u: Optional[float] = None
        self._recent_Z: Optional[CPUArray] = None
        
        self._solved: bool = False
        self._current_F: Optional[CPUArray] = None
    
    def initiate(self, Z_start: CPUArray):
        self._solved = False
        self._v_minus = 0
        self._v_plus = 0
        self._tau = cpu_zeros((self._num_edges,))
        self._last_v_minus = 0
        self._last_v_plus = 0
        self._last_tau = cpu_zeros((self._num_edges,))
        self._current_F = Z_start
        self._recent_Z = Z_start
        self._recent_u = min(1, np.max(np.divide(Z_start, self._capacities)))
    
    @property
    def num_edges(self) -> int:
        return self._num_edges
    @property
    def capacities(self) -> CPUArray:
        return self._capacities
    @property
    def solver_params(self) -> RelaxedMLUSolverParams:
        return self._solver_params
    
    @property
    def current_u(self) -> float:
        return self._recent_u
    @property
    def current_Z(self) -> CPUArray:
        return self._recent_Z
    
    @staticmethod
    def pgd_update(C: np.ndarray, F: np.ndarray, RHO: float, ALPHA: float, 
                   GAMMA: float,
                   TAU: np.ndarray, V_MINUS: float, V_PLUS: float) -> Tuple[np.ndarray, float, float]:
        grad_v_minus = (np.dot(TAU, C) + V_MINUS - V_PLUS) / (2 * ALPHA)
        grad_v_plus = -grad_v_minus
        grad_tau = C * grad_v_minus + TAU / RHO - F

        next_v_minus = V_MINUS - GAMMA * grad_v_minus
        next_v_plus = V_PLUS - GAMMA * grad_v_plus
        next_tau = TAU - GAMMA * grad_tau

        return next_tau, next_v_minus, next_v_plus

    @staticmethod
    def nesterov_pgd_update(C: np.ndarray, F: np.ndarray, RHO: float, ALPHA: float, 
                            GAMMA: float, BETA: float,
                            TAU_OLD: np.ndarray, V_MINUS_OLD: float, V_PLUS_OLD: float,
                            TAU: np.ndarray, V_MINUS: float, V_PLUS: float) -> Tuple[np.ndarray, float, float]:
        v_minus_nesterov = V_MINUS + BETA * (V_MINUS - V_MINUS_OLD)
        v_plus_nesterov = V_PLUS + BETA * (V_PLUS - V_PLUS_OLD)
        tau_nesterov = TAU + BETA * (TAU - TAU_OLD)
        
        grad_v_minus = (np.dot(tau_nesterov, C) + v_minus_nesterov - v_plus_nesterov) / (2 * ALPHA)
        grad_v_plus = -grad_v_minus
        grad_tau = C * grad_v_minus + tau_nesterov / RHO - F

        next_v_minus = V_MINUS - GAMMA * grad_v_minus
        next_v_plus = V_PLUS - GAMMA * grad_v_plus
        next_tau = TAU - GAMMA * grad_tau

        return next_tau, next_v_minus, next_v_plus
    
    def solve(self):
        assert not self._solved

        PARAMS = self._solver_params
        C = self._capacities
        TAU_OLD = self._last_tau
        V_PLUS_OLD = self._last_v_plus
        V_MINUS_OLD = self._last_v_minus
        TAU = self._tau
        V_PLUS = self._v_plus
        V_MINUS = self._v_minus
        F_M = self._current_F

        for _ in range(PARAMS.GDIters):
            # TAU, V_MINUS, V_PLUS = self.pgd_update(C, F_M, PARAMS.Rho, PARAMS.Alpha, PARAMS.Gamma, TAU, V_MINUS, V_PLUS)
            TAU_NEXT, V_MINUS_NEXT, V_PLUS_NEXT = self.nesterov_pgd_update(
                C, F_M, PARAMS.Rho, PARAMS.Alpha, 
                PARAMS.Gamma, self.BETA, 
                TAU_OLD, V_MINUS_OLD, V_PLUS_OLD,
                TAU, V_MINUS, V_PLUS)
            
            self._last_tau = TAU
            self._last_v_minus = V_MINUS
            self._last_v_plus = V_PLUS
            self._tau = TAU_NEXT
            self._v_minus = V_MINUS_NEXT
            self._v_plus = V_PLUS_NEXT
        
        self._recent_u = (np.dot(self._tau, C) + self._v_minus - self._v_plus) / (2 * PARAMS.Alpha)
        self._recent_Z = F_M - self._tau / PARAMS.Rho
        assert self._recent_Z.shape == (self.num_edges,)
        self._solved = True

    def update_solver(self, new_F: CPUArray):
        self._current_F = new_F
        self._solved = False