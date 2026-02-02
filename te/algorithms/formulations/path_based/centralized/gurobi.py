import gurobipy
import numpy as np
import networkx as nx
from typing import List, Tuple, Optional
from collections import defaultdict
from gurobipy import GRB, GurobiError
from te.algorithms.base import *
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, Commodity
from te.algorithms.solution import EdgeBasedMinimizeMaximumUtilitySolution
from te.algorithms.sub_algorithms.link_capacity_test import check_capacity_constraint
from te.algorithms.sub_algorithms.flow_conservation_test import check_flow_conservation
from utils.logging import as_info, as_fail, ShortTQDMEnumerate, ShortTQDM
from te.algorithms.sub_algorithms.paths import TShortestPaths, path_based_to_edge_based_nnz, get_or_make_path_object_for_topology_name
from ...edge_based.centralized import make_model
from . import GurobiPathBasedSolverParams


class GurobiPathBasedTE(TrafficEngineeringLP):
    def __init__(self, problem_description: TrafficEngineeringProblemDescription, solver_params: GurobiPathBasedSolverParams) -> None:
        super().__init__(problem_description, solver_params)
        self._graph = problem_description.Graph
        self._traffic = problem_description.TM
        self._solver_params: GurobiPathBasedSolverParams = solver_params
        self._env: Optional[gurobipy.Env] = None
        self._model: Optional[gurobipy.Model] = None
        self._Y_tk: Optional[gurobipy.tupledict] = None
        self._utility: Optional[gurobipy.Var] = None
        self._objective: Optional[gurobipy.LinExpr] = None
        self._commodity_list: List[Commodity] = traffic_to_commodity(self._traffic)
        self._X_ek: Optional[np.ndarray] = None
        self._splits: Optional[np.ndarray] = None
        self._capacities: List[float] = [c_e for _, _, c_e in self._graph.edges.data('capacity')]

        self._path_object: Optional[TShortestPaths] = None
        self._demands: np.ndarray = np.array([commodity.demand for commodity in self._commodity_list])
        
        self._report_problem_size()
        self._initialize()
    
    def _initialize(self):
        self._path_object = get_or_make_path_object_for_topology_name(
            topo_name=self._problem_description.EvalParams.TopologyName,
            T=self._solver_params.NumberOfPathsPerCommodity,
            edge_disjoint=False
        )
    
    @property
    def alg_name(self) -> str:
        return 'Path-Based Gurobi'
    
    @property
    def graph(self) -> nx.DiGraph:
        return self._graph
    
    @property
    def traffic(self) -> TrafficMatrixBase:
        return self._traffic

    @property
    def params(self) -> SolverParams:
        return self._solver_params
    
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
        raise NotImplementedError

    def _set_X_ek(self):
        ALPHA_K = self._path_object.alpha
        K, N, T = ALPHA_K.shape
        ASSIGNMENTS = self._Y_tk
        DEMANDS = self._demands
        Y_TK = np.ndarray(shape=(T, K))
        for k in range(K):
            for t in range(T):
                Y_TK[t, k] = ASSIGNMENTS[(t, k)].X
        self._splits = Y_TK
        self._X_ek = path_based_to_edge_based_nnz(Y_TK, ALPHA_K.rows, ALPHA_K.cols, N, DEMANDS)
    
    def _make_variables(self):
        assert self._model is None and self._Y_tk is None

        K = len(self._commodity_list)
        T = self._solver_params.NumberOfPathsPerCommodity
        
        ENV = gurobipy.Env()
        ENV.start()
        self._env = ENV
        MODEL = make_model(name='PathBasedTE', params=self._solver_params, env=ENV)
        self._model = MODEL

        print(as_info("Adding tunnel assignment variables"))
        self._Y_tk = MODEL.addVars(T, K, lb=0.0, vtype=GRB.CONTINUOUS, name='Y')
        # self._utility = MODEL.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='U')
        # TODO: Keep the upper bound?
        self._utility = MODEL.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='U')
    
    def _add_constraints(self):
        assert self._model is not None and self._Y_tk is not None

        MODEL = self._model
        Y_TK = self._Y_tk
        CAPS = self._capacities
        ALPHA_K = self._path_object.alpha
        BETA_K = self._path_object.beta
        UTILITY = self._utility
        COMMODITIES = self._commodity_list
        K, N, T = ALPHA_K.shape

        # Capacity constraint
        print(as_info("Adding capacity constraints"))
        total_flows = [gurobipy.LinExpr() for _ in range(N)]
        for k, commodity in ShortTQDMEnumerate(COMMODITIES):
            D_K = commodity.demand
            rows = ALPHA_K.rows[k]
            cols = ALPHA_K.cols[k]
            nnz = len(rows)
            for i in range(nnz):
                e = rows[i]
                t = cols[i]
                total_flows[e].addTerms(D_K, Y_TK[(t, k)])
        self._capacity_constraints = [MODEL.addConstr(total_flow <= UTILITY * CAPS[e]) for e, total_flow in enumerate(total_flows)]

        # Demand constraint
        print(as_info("Adding demand constraints"))
        for k in ShortTQDM(range(K)):
            total_assignment = gurobipy.LinExpr()
            for t in range(T):
                total_assignment.addTerms(1, Y_TK[(t, k)])
            MODEL.addConstr(total_assignment - 1 == 0)
        
        # Number of paths constraint
        print(as_info("Adding path availability constraints"))
        for k in ShortTQDM(range(K)):
            for t in range(BETA_K[k], T):
                MODEL.addConstr(Y_TK[(t, k)] == 0)

    def _add_objective(self):
        assert self._model is not None and \
                self._Y_tk is not None and \
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
            for key, value in self.params._asdict().items():
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

    def check(self):
        unsat_ratio, unsat_commodities, total_satisfcation = check_flow_conservation(
            self._X_ek, self._graph, self._commodity_list,
            self._problem_description.EvalParams
        )
        congested_ratio, congested_links = check_capacity_constraint(
            self._X_ek, self._graph, self._commodity_list,
            self._problem_description.EvalParams
        )
        self.check_result = TrafficEngineeringLPCheckResult(
            unsat_ratio=unsat_ratio,
            congested_ratio=congested_ratio,
            unsat_commodities=unsat_commodities,
            congested_links=congested_links,
            density=np.count_nonzero(self._X_ek) / self._X_ek.size,
            total_satisfcation=total_satisfcation
        )
    
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
        raise NotImplementedError
    
    def add_solution_elements(self, solution: TrafficEngineeringLPSolution):
        # print(self._splits)
        raise NotImplementedError


import jsonargparse

def centralized_gurobi_solver_params_parser() -> jsonargparse.ArgumentParser:
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(GurobiPathBasedSolverParams, 'SolverParams', help='Gurobi Solver Params')
    return parser


def parse_centralized_gurobi_solver_params(args: jsonargparse.Namespace) -> GurobiPathBasedSolverParams:
    return GurobiPathBasedSolverParams.make_from_args(args.SolverParams)
