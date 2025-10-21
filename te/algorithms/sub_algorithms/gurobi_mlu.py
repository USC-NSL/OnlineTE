import gurobipy
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from gurobipy import GRB
from te.algorithms.base import SolverParams, GurobiSolverParams
from te.algorithms.array_utils.cpu_utils import CPUArray, cpu_array, cpu_cast_float
from te.algorithms.utils import optimize_or_scream, make_model, as_info
from .mlu_base import ControllerMLUSolver, ControllerMLUException
from te.algorithms.statistics.helpers import record_cpu_runtime


@dataclass
class GurobiMLUParams(SolverParams):
    Rho: float
    Alpha: float
    GurobiParams: GurobiSolverParams

    def __post_init__(self):
        self._left_column_share = 0.5


class GurobiMLU(ControllerMLUSolver):
    def __init__(self, num_edges: int, capacities: CPUArray, solver_params: GurobiMLUParams):
        self._num_edges: int = num_edges
        self._capacities: CPUArray = capacities
        self._solver_params = solver_params

        self._env: gurobipy.Env = None
        self._double_precision_capacities: np.ndarray = np.array(capacities, dtype=np.float64)
        self._model_controller: Optional[gurobipy.Model] = None
        self._objective_controller: Optional[gurobipy.QuadExpr] = None
        self._Z_e: Optional[gurobipy.tupledict] = None
        self._utility: Optional[gurobipy.Var] = None
        self._utility_bound_constraints: Tuple[gurobipy.Constr, gurobipy.Constr] = None
        """Gives the dual variables `v_neg` and `v_pos`"""
        self._capacity_constraints: List[gurobipy.Constr] = None
        """Gives the dual variables `tau_e`, a vector of length `n`"""

        self._current_F: np.ndarray = None
        self._solved: bool = False
        self._current_u: Optional[float] = None
        self._current_Z: Optional[CPUArray] = None

    @property
    def num_edges(self) -> int:
        return self._num_edges
    @property
    def capacities(self) -> CPUArray:
        return self._capacities
    @property
    def solver_params(self) -> GurobiSolverParams:
        return self._solver_params
    @property
    def is_solved(self) -> bool:
        return self._solved

    def _make_variables(self):
        assert self._model_controller is None
        
        NUM_EDGES = self.num_edges

        ENV = gurobipy.Env()
        ENV.setParam('OutputFlag', 0)
        ENV.start()
        self._env = ENV

        PARAMS = self._solver_params
        MODEL_CONTROLLER: gurobipy.Model = \
            make_model('EdgeBasedDistributedTE_Controller', params=PARAMS.GurobiParams, env=ENV)
        
        self._Z_e = MODEL_CONTROLLER.addVars(NUM_EDGES, lb=float('-inf'), vtype=GRB.CONTINUOUS, name='Z_E')
        self._utility = MODEL_CONTROLLER.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='U')

        print(as_info(f"Gurobi objective convergence tolerance: {PARAMS.GurobiParams.ConvTol}"))
        self._model_controller = MODEL_CONTROLLER

    def _add_constraints(self):
        assert self._model_controller is not None

        C = self._capacities
        Z_E = self._Z_e
        UTILITY = self._utility
        MODEL_CONTROLLER = self._model_controller

        # Utilization bound constraints
        u_low = MODEL_CONTROLLER.addConstr(UTILITY >= 0)
        u_high = MODEL_CONTROLLER.addConstr(-UTILITY >= -1)
        self._utility_bound_constraints = (u_low, u_high)

        capacity_constraints: List[gurobipy.Constr] = []
        for e in range(self.num_edges):
            capacity_constraints.append(MODEL_CONTROLLER.addConstr(UTILITY * C[e] >= Z_E[e]))
        self._capacity_constraints = capacity_constraints
    
    def _add_objective(self):
        """
        Gurobi doesn't allow the objective to be changed bit by bit ...
        We have no choice to build it from scratch in `_update_controller_objective`.
        """
        pass

    def _update_controller_objective(self):
        NUM_EDGES = self._num_edges
        UTILITY = self._utility
        Z_E = self._Z_e
        F_M_E = self._current_F
        RHO = self._solver_params.Rho
        MODEL_CONTROLLER = self._model_controller
        ALPHA = self._solver_params.Alpha
        
        OBJECTIVE_CONTROLLER = gurobipy.QuadExpr()
        OBJECTIVE_CONTROLLER.addTerms(ALPHA, UTILITY)
        for e in range(NUM_EDGES):
            OBJECTIVE_CONTROLLER += (RHO/2) * (F_M_E[e] - Z_E[e]) ** 2
        MODEL_CONTROLLER.setObjective(OBJECTIVE_CONTROLLER, GRB.MINIMIZE)
        self._objective_controller = OBJECTIVE_CONTROLLER

    @property
    def current_u(self) -> float:
        return self._current_u
    @property
    def current_Z(self) -> CPUArray:
        return self._current_Z
    
    def close(self):
        if self._model_controller:
            self._model_controller.close()
        if self._env:
            self._env.close()
    
    def reset(self, with_params: False):
        self._model_controller.reset()
        if with_params:
            self._model_controller.resetParams()
    
    def update_F_m(self, new_F: CPUArray):
        self._current_F = np.array(new_F, dtype=np.float64)
        self._solved = False
        self._update_controller_objective()    

    @record_cpu_runtime('Gurobi-MLU')
    def solve(self):
        assert self._solved is False
        try:
            optimize_or_scream(self._model_controller)
        except RuntimeError as e:
            raise ControllerMLUException('Gurobi', e)
        U = self._utility
        Z_E = self._Z_e
        self._current_u = cpu_cast_float(U.X)
        self._current_Z = cpu_array([Z_E[e].X for e in range(self.num_edges)])
        self._solved = True
