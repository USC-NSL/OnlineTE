import gurobipy
import numpy as np
from typing import List, Optional
from gurobipy import GRB
from te.algorithms.base import *
from te.traffic_models.base import *
from utils.logging import as_info, ShortTQDM
from utils.gurobi_utils import make_model
from te.path_providers import *
from te.path_providers.sparse_ops import path_based_to_edge_based_nnz
from . import GurobiPathBasedSolverParams


class GurobiPathBasedTE(TELP[GurobiPathBasedSolverParams]):
    def __init__(self, problem_description: TEProblemDescription, solver_params: GurobiPathBasedSolverParams) -> None:
        super().__init__(problem_description, solver_params)
        self._env: Optional[gurobipy.Env] = None
        self._model: Optional[gurobipy.Model] = None
        self._Y_tk: Optional[gurobipy.tupledict] = None
        self._utility: Optional[gurobipy.Var] = None
        self._objective: Optional[gurobipy.LinExpr] = None
        self._X_ek: Optional[np.ndarray] = None
        self._splits: Optional[np.ndarray] = None
        self._path_object: Optional[PathProvider] = None
        self._demand_constraints: Optional[List[gurobipy.Constr]] = None
        self._total_flow: Optional[gurobipy.LinExpr] = None
        
        self._initialize()
    
    def _initialize(self):
        path = self._solver_params.path_file
        if path is not None:
            self._path_object = PathProvider.load()
        else:
            self._path_object = build_provider(
                T=self._solver_params.max_num_paths_per_commodity,
                graph=self._graph,
                per_commodity_provider=get_scheme(),
                edge_indexing=self._edge_indexing
            )
    
    @property
    def alg_name(self) -> str:
        return 'Path-Based Gurobi'

    @property
    def current_objective(self) -> float:
        return abs(self._objective.getValue())

    def _set_X_ek(self):
        ROWS = self._path_object.rows
        COLS = self._path_object.cols
        K, N, T = self._path_object.shape
        ASSIGNMENTS = self._Y_tk
        DEMANDS = traffic_to_demands(self._current_TM)
        Y_TK = np.ndarray(shape=(T, K))
        for k in range(K):
            for t in range(T):
                Y_TK[t, k] = ASSIGNMENTS[(t, k)].X
        self._splits = Y_TK
        self._X_ek = path_based_to_edge_based_nnz(
            Y_TK, ROWS, COLS, N, DEMANDS
        )
    
    def _make_variables(self):
        assert self._model is None and self._Y_tk is None

        K = self.number_of_commodities
        T = self._solver_params.max_num_paths_per_commodity
        
        ENV = gurobipy.Env()
        ENV.start()
        self._env = ENV
        MODEL = make_model(
            name='PathBasedTE', params=self._solver_params,
            feasibility_tolerance=self._problem_description.eval_params.feasibility_tolerance,
            optimality_tolerance=self._problem_description.eval_params.optimality_tolerance,
            verbose=self._problem_description.eval_params.verbose,
            env=ENV
        )
        self._model = MODEL

        print(as_info("Adding tunnel assignment variables"))
        self._Y_tk = MODEL.addVars(T, K, lb=0.0, vtype=GRB.CONTINUOUS, name='Y')
        if self.objective == TEObjective.MLU:
            self._utility = MODEL.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='U')
    
    def _add_constraints(self):
        assert self._model is not None and self._Y_tk is not None

        MODEL = self._model
        Y_TK = self._Y_tk
        CAPS = self._capacities
        ROWS = self._path_object.rows
        COLS = self._path_object.cols
        BETA_K = self._path_object.beta
        K, N, T = self._path_object.shape

        # Capacity constraint
        print(as_info("Adding capacity constraints"))
        total_flows = [gurobipy.LinExpr() for _ in range(N)]
        for k in ShortTQDM(range(K)):
            rows = ROWS[k]
            cols = COLS[k]
            nnz = len(rows)
            for i in range(nnz):
                e = rows[i]
                t = cols[i]
                total_flows[e].addTerms(1, Y_TK[(t, k)])

        match self.objective:
            case TEObjective.MLU:
                self._capacity_constraints = [
                    MODEL.addConstr(total_flow <= self._utility * CAPS[e]) \
                        for e, total_flow in enumerate(total_flows)
                ]
            case _ :
                self._capacity_constraints = [
                    MODEL.addConstr(total_flow <= CAPS[e]) \
                        for e, total_flow in enumerate(total_flows)
                ]

        # Demand constraint
        print(as_info("Adding demand constraints"))
        demand_constraints: List[gurobipy.Constr] = []
        for k in ShortTQDM(range(K)):
            total_assignment = gurobipy.LinExpr()
            for t in range(T):
                total_assignment.addTerms(1, Y_TK[(t, k)])
            demand_constraints.append(MODEL.addConstr(total_assignment == 1))
        self._demand_constraints = demand_constraints
        
        # Number of paths constraint
        print(as_info("Adding path availability constraints"))
        for k in ShortTQDM(range(K)):
            for t in range(BETA_K[k], T):
                MODEL.addConstr(Y_TK[(t, k)] == 0)

        # Total flow objective
        if self.objective == TEObjective.MAX_FLOW:
            total_flow = gurobipy.LinExpr()
            for k in range(K):
                for t in range(T):
                    total_flow.addTerms(-1, Y_TK[(t, k)])
            self._total_flow = total_flow

    def _add_objective(self):
        assert self._model is not None and \
                self._Y_tk is not None and \
                self._objective is None
        
        MODEL = self._model

        match self.objective:
            case TEObjective.MLU: self._objective = gurobipy.LinExpr(1.0, self._utility)
            case TEObjective.MAX_FLOW: self._objective = self._total_flow
            case _ : raise ValueError
        
        MODEL.setObjective(self._objective, GRB.MINIMIZE)
    
    def close(self):
        self._model.close()
        self._env.close()
    
    def _solve_for_tm(self, tm: np.ndarray):
        self._model.optimize()
        if self._model.Status == gurobipy.GRB.OPTIMAL:
            self._set_X_ek()
        else:
            raise RuntimeError(f"Problem when solving. Gurobi status: {self._model.Status}")
    
    def _update_constraits(self, tm: np.ndarray):
        assert self._demand_constraints is not None
        COMMODITIES = traffic_to_commodity(tm)
        for constraint, commodity in zip(self._demand_constraints, COMMODITIES):
            constraint.RHS = commodity.demand

    def _update_objective(self, tm: np.ndarray):
        pass


import jsonargparse

def centralized_gurobi_solver_params_parser() -> jsonargparse.ArgumentParser:
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(GurobiPathBasedSolverParams, 'SolverParams', help='Gurobi Solver Params')
    return parser


def parse_centralized_gurobi_solver_params(args: jsonargparse.Namespace) -> GurobiPathBasedSolverParams:
    return GurobiPathBasedSolverParams.make_from_args(args.SolverParams)
