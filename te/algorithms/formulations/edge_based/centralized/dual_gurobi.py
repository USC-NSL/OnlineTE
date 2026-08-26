import gurobipy
import numpy as np
import networkx as nx
from typing import List, Tuple, Optional
from collections import defaultdict
from gurobipy import GRB, GurobiError
from topologies.utils import get_graph_M_matrix
from te.algorithms.base import *
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, Commodity
from te.algorithms.solution import EdgeBasedMinimizeMaximumUtilitySolution
from te.algorithms.sub_algorithms.link_capacity_test import check_capacity_constraint
from te.algorithms.sub_algorithms.flow_conservation_test import check_flow_conservation
from utils.logging import as_info, as_fail, as_success, as_warning, ShortTQDM, ShortTQDMEnumerate
from . import GurobiSolverParams, make_model


class DualGurobiTE(TELP):
    """
    Similar to `CentralizedEdgeBasedLP`, but it also cleanly returns all the dual variables
    and explicitly checks the dual infesibility.
    This is mostly used for debug, but the output solution can be helpful for exact warm starts.
    """
    def __init__(self, problem_description: TEProblemDescription, solver_params: GurobiSolverParams) -> None:
        super().__init__(problem_description, solver_params)
        self._graph = problem_description.Graph
        self._traffic = problem_description.TM
        self._M: np.ndarray = get_graph_M_matrix(self._graph)
        self._c_e: np.ndarray = np.array([item[-1] for item in self._graph.edges.data('capacity')])
        self._solver_params: GurobiSolverParams = solver_params
        self._env: gurobipy.Env = None
        self._model: gurobipy.Model = None
        self._flows: gurobipy.tupledict = None
        self._utility: gurobipy.Var = None
        self._objective: gurobipy.LinExpr = None
        self._commodity_list: List[Commodity] = traffic_to_commodity(self._traffic)
        self._utility_bound_constraints: Tuple[gurobipy.Constr, gurobipy.Constr] = None
        """Gives the dual variables `v_neg` and `v_pos`"""
        self._flow_bound_constraints: List[List[gurobipy.Constr]] = None
        """Gives the dual variables `lambda_ek`, same shape as `X_ek`"""
        self._demand_constraints: List[List[gurobipy.Constr]] = None
        """Gives the dual variables `r_mk`, a matrix of shape `m x K`"""
        self._capacity_constraints: List[gurobipy.Constr] = None
        """Gives the dual variables `tau_e`, a vector of length `n`"""
        self._X_ek: np.ndarray = None
        
        self._report_problem_size()
    
    @property
    def alg_name(self) -> str:
        return 'Centralized'
    
    @property
    def graph(self) -> nx.DiGraph:
        return self._graph
    
    @property
    def traffic(self) -> TrafficMatrixBase:
        return self._traffic
    
    @property
    def commodity_list(self) -> List[Commodity]:
        return self._commodity_list

    @property
    def objective_value(self) -> float:
        return self._utility.X
    
    @property
    def objective_trace(self) -> Optional[List[float]]:
        # TODO: Anyway to get this from Gurobi?
        return None
    
    @property
    def assignments(self) -> np.ndarray:
        assert self._X_ek is not None
        return self._X_ek

    def _report_problem_size(self):
        M = len(self._graph.nodes)
        N = len(self._graph.edges)
        K = len(self._commodity_list)

        print(as_info(f"Graph Size: {M} nodes | {N} edges"))
        print(as_info(f"Number of commodities: {K}"))

    def initialize_to(self, solution: EdgeBasedMinimizeMaximumUtilitySolution):
        assert self._model is not None and self._flows is not None
        solution.initiate_model_from_basis(self._model)

    def _set_X_ek(self):
        K = len(self._commodity_list)
        N = len(self._graph.edges)
        FLOWS = self._flows
        X_EK = np.ndarray(shape=(N, K))
        for k in range(K):
            for e in range(N):
                X_EK[e, k] = FLOWS[(e, k)].X
        self._X_ek = X_EK
    
    def _make_variables(self):
        assert self._model is None and self._flows is None
        K = len(self._commodity_list)
        N = len(self._graph.edges)
        
        ENV = gurobipy.Env()
        ENV.start()
        self._env = ENV
        MODEL = make_model(name='DualCheckingEdgeBasedTE', params=self._solver_params, env=ENV)
        self._model = MODEL

        print(as_info("Adding commodity assignment variables"))
        self._flows = MODEL.addVars(N, K, lb=float('-inf'), vtype=GRB.CONTINUOUS, name='X')
        self._utility = MODEL.addVar(lb=float('-inf'), vtype=GRB.CONTINUOUS, name='U')

    def _add_constraints(self):
        assert self._model is not None and self._flows is not None

        K = len(self._commodity_list)
        MODEL = self._model
        GRAPH = self._graph
        FLOWS = self._flows
        UTILITY = self._utility
        COMMODITIES = self._commodity_list

        # Utilization bound constraints
        u_low = MODEL.addConstr(UTILITY >= 0)
        u_high = MODEL.addConstr(-UTILITY >= -1)
        self._utility_bound_constraints = (u_low, u_high)

        # Flow bound constraints
        print(as_info("Adding flow bound constraints"))
        bounds = []
        for e, edge in ShortTQDMEnumerate(self._graph.edges()):
            ls = []
            for k, commodity in enumerate(self._commodity_list):
                SOURCE = commodity.source
                DESTINATION = commodity.destination
                if edge[0] == DESTINATION:
                    ls.append(MODEL.addConstr(self._flows[(e, k)] == 0))
                elif edge[1] == SOURCE:
                    ls.append(MODEL.addConstr(self._flows[(e, k)] == 0))
                else:
                    ls.append(MODEL.addConstr(self._flows[(e, k)] >= 0 ))
            bounds.append(ls)
        self._flow_bound_constraints = bounds

        # Capacity constraint
        print(as_info("Adding capacity constraints"))
        capacity_constraints: List[gurobipy.Constr] = []
        for e, (_, _, c_e) in ShortTQDMEnumerate(GRAPH.edges.data('capacity')):
            total_flow = gurobipy.LinExpr()
            for k in range(K):
                total_flow.addTerms(1, FLOWS[(e, k)])
            capacity_constraints.append(MODEL.addConstr(UTILITY * c_e >= total_flow))
        self._capacity_constraints = capacity_constraints

        demand_constraints = []
        M = self._M
        print(as_info("Adding demand/flow-conservation constraints"))
        for v in ShortTQDM(range(self._graph.number_of_nodes())):
            ls = []
            for k, commodity in enumerate(COMMODITIES):
                if v == commodity.source:
                    B_vk = commodity.demand
                elif v == commodity.destination:
                    B_vk = -commodity.demand
                else:
                    B_vk = 0
                
                expr = gurobipy.LinExpr()
                for e in range(self._graph.number_of_edges()):
                    if M[v, e] != 0:
                        expr.addTerms(M[v, e], FLOWS[(e, k)])
                ls.append(MODEL.addConstr(expr == B_vk))
            demand_constraints.append(ls)
        self._demand_constraints = demand_constraints

    def _add_objective(self):
        assert self._model is not None and \
                self._flows is not None and \
                self._objective is None
        
        MODEL = self._model

        # For now, let's minimize maximum link utilization
        self._objective = self._utility
        MODEL.setObjective(self._objective, GRB.MINIMIZE)
    
    def close(self):
        self._model.close()
    
    def make_lp(self):
        self._make_variables()
        self._add_constraints()
        self._add_objective()
    
    def reset(self, with_params: False):
        self._model.reset()
        if with_params:
            self._model.resetParams()
    
    def solve(self, params: SolverParams = None) -> float:
        self.check_result = None
        if params:
            self.reset(with_params=True)
            self._params = params
            for key, value in self.solver_params._asdict().items():
                self._model.setParam(key, value)
        try:
            self._model.optimize()
            if self._model.Status == gurobipy.GRB.OPTIMAL:
                self._set_X_ek()
                return self._model.Runtime
            return -1
        except GurobiError as e:
            print(as_fail(f'Error code {e.errno}: {e}'))
            return -1
    
    def get_dual_var(self, constraint: gurobipy.Constr) -> float:
        if self._solver_params.Method == GRB.METHOD_BARRIER:
            return getattr(constraint, 'BarPi')
        else:
            return getattr(constraint, 'Pi')

    def check(self):
        eval_params = self._problem_description.EvalParams
        unsat_ratio, unsat_commodities, total_satisfcation = check_flow_conservation(
            self._X_ek, self._graph, self._commodity_list,
            eval_params
        )
        congested_ratio, congested_links = check_capacity_constraint(
            self._X_ek, self._graph, self._commodity_list,
            eval_params
        )
        self.check_result = TECheckResult(
            unsat_ratio=unsat_ratio,
            congested_ratio=congested_ratio,
            unsat_commodities=unsat_commodities,
            congested_links=congested_links,
            total_satisfcation=total_satisfcation
        )

        M_MAT = self._M
        C_E = self._c_e
        N = self.graph.number_of_edges()
        M = self.graph.number_of_nodes()
        K = len(self._commodity_list)
        COMMODITIES = self._commodity_list

        v_minus = self.get_dual_var(self._utility_bound_constraints[0])
        v_plus = self.get_dual_var(self._utility_bound_constraints[1])

        tau_e = np.array([self.get_dual_var(const) for const in self._capacity_constraints])

        lambda_ek = np.zeros(shape=(N, K))
        for e in range(N):
            for k in range(K):
                lambda_ek[e, k] = self.get_dual_var(self._flow_bound_constraints[e][k])
        
        r_vk = np.zeros(shape=(M, K))
        for v in range(M):
            for k in range(K):
                r_vk[v, k] = self.get_dual_var(self._demand_constraints[v][k])

        dual_inf_1 = 1 - v_minus + v_plus - np.dot(tau_e, C_E)
        dual_inf_2 = np.zeros(shape=(N, K))
        for e in range(N):
            for k in range(K):
                dual_inf_2[e, k] = tau_e[e] - lambda_ek[e, k] - np.dot(M_MAT[:, e], r_vk[:, k])

        dual_feasible_1 = abs(dual_inf_1) < self._solver_params.FeasibilityTol
        dual_inf_2 = np.linalg.norm(dual_inf_2)
        dual_feasible_2 = abs(dual_inf_2) < self._solver_params.FeasibilityTol
        if dual_feasible_1 and dual_feasible_2:
            print(as_success('Dual Feasibility Holds'))
        else:
            print(as_fail('Solution Is Dual Infeasible'))
            if not dual_feasible_1:
                print(as_warning(f'Dual Infeasibility I: {dual_inf_1}'))
            if not dual_feasible_2:
                print(as_warning(f'Dual Infeasibility II: {dual_inf_2}'))
        
        holder = 0
        for v in range(self._graph.number_of_nodes()):
            for k, commodity in enumerate(COMMODITIES):
                if v == commodity.source:
                    holder += r_vk[v, k] * commodity.demand
                elif v == commodity.destination:
                    holder -= r_vk[v, k] * commodity.demand
                else:
                    pass
        dual_objective = -v_plus + holder
        dual_objective_gap = abs(dual_objective - self._model.ObjBound) / self._model.ObjBound
        if dual_objective_gap < self._solver_params.ConvTol:
            print(as_success('Dual Objective Convergance Holds'))
        else:
            print(as_fail('Dual Objective Convergance Does Not Hold'))
            print(as_warning(f'Dual Objective Gap: {dual_objective_gap}'))
    
    def get_solution_commodity_list(self) -> List[Tuple[Commodity, Commodity]]:
        assert self._X_ek is not None

        COMMODITIES = self._commodity_list
        GRAPH = self._graph
        X = self._X_ek

        ls = []
        for k, commodity in enumerate(COMMODITIES):
            flow_out = defaultdict(list)
            flow_in = defaultdict(list)
            for e, edge in enumerate(GRAPH.edges()):
                flow_out[edge[0]].append(X[e, k])
                flow_in[edge[1]].append(X[e, k])
            commodity_sent = Commodity(
                source=commodity.source, destination=commodity.destination,
                demand=sum(flow_out[commodity.source])
            )
            commodity_received = Commodity(
                source=commodity.source, destination=commodity.destination,
                demand=sum(flow_in[commodity.destination])
            )
            ls.append((commodity_sent, commodity_received))
        return ls

    def update_traffic_matrix(self, tm: TrafficMatrixBase):
        # First, record the new commodity list
        COMMODITIES = traffic_to_commodity(tm)
        self._commodity_list = COMMODITIES

        # Now, update demand constraints
        for constraints, commodity in zip(self._demand_constraints, COMMODITIES):
            DEMAND = commodity.demand
            source_constraint, destination_constraint = constraints
            source_constraint.RHS = DEMAND
            destination_constraint.RHS = DEMAND
        
        # Record the new TM
        self._traffic = tm
    
    def add_solution_elements(self, solution: TESolution):
        solution.add_solution_element(self._utility, 'utility')
        solution.add_solution_element(self._flows, 'assignments')
        # solution.add_solution_element(self._capacity_constraints, 'capacity_constraints')


import argparse
from .gurobi import centralized_gurobi_solver_params_parser, parse_centralized_gurobi_solver_params

def centralized_dual_gurobi_solver_params_parser(parser: argparse.ArgumentParser):
    centralized_gurobi_solver_params_parser(parser)


def parse_centralized_dual_gurobi_solver_params(
    parser: argparse.ArgumentParser, 
    args: Optional[argparse.Namespace] = None
) -> Tuple[GurobiSolverParams, argparse.Namespace]:
    return parse_centralized_gurobi_solver_params(parser, args)
