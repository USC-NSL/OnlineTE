import gurobipy
import numpy as np
import networkx as nx
from typing import List, Tuple, Optional
from collections import defaultdict
from gurobipy import GRB, GurobiError
from te.algorithms.base import *
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, Commodity
from te.algorithms.solution import EdgeBasedMinimizeMaximumUtilitySolution
from utils.logging import as_info, as_fail, ShortTQDMEnumerate
from . import GurobiSolverParams, make_model
from ..base import EdgeBasedTEBase


class GurobiTE(EdgeBasedTEBase):
    """
    An honest implementation of edge-based MLU with Gurobi.
    Becomes too sluggish for very large topologies, but solutions look very nice.
    """
    def __init__(self, problem_description: TrafficEngineeringProblemDescription, solver_params: GurobiSolverParams) -> None:
        super().__init__(problem_description, solver_params)
        self._graph = problem_description.Graph
        self._traffic = problem_description.TM
        self._solver_params: GurobiSolverParams = solver_params
        self._env: Optional[gurobipy.Env] = None
        self._model: Optional[gurobipy.Model] = None
        self._flows: Optional[gurobipy.tupledict] = None
        self._utility: Optional[gurobipy.Var] = None
        self._objective: Optional[gurobipy.LinExpr] = None
        self._commodity_list: List[Commodity] = traffic_to_commodity(self._traffic)
        self._demand_constraints: Optional[List[Tuple[gurobipy.Constr, gurobipy.Constr]]] = None
        self._capacity_constraints: Optional[gurobipy.tupledict] = None
        self._X_ek: Optional[np.ndarray] = None

        self._demand_objective: Optional[gurobipy.LinExpr] = None
        self._regularizer_objective: Optional[gurobipy.LinExpr] = None
        
        self._report_problem_size()
    
    @property
    def alg_name(self) -> str:
        return 'Centralized-Gurobi'
    
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
        if self._problem_description.is_mlu:
            return self._utility.X
        else:
            return -self._objective.getValue()
    
    @property
    def objective_trace(self) -> Optional[List[float]]:
        # TODO: Anyway to get this from Gurobi?
        return None
    
    @property
    def assignments(self) -> np.ndarray:
        assert self._X_ek is not None
        return self._X_ek

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

        """
        As per our formulation, the commodity matrix is of the form X_{ke},
        where each column is the split of a commodity `k` over the set of 
        edges in the graph.
        For `N` edges and `K` commodities, `X` would be a `N x K` matrix, where
        `X[e]` gives the flow of each commodity on edge `e`.
        """

        K = len(self._commodity_list)
        N = len(self._graph.edges)
        
        ENV = gurobipy.Env()
        ENV.start()
        self._env = ENV
        MODEL = make_model(name='EdgeBasedTE', params=self._solver_params, env=ENV)
        self._model = MODEL

        # This implicitly encodes the condition for `X_{ke} >= 0`
        print(as_info("Adding commodity assignment variables"))
        self._flows = MODEL.addVars(N, K, lb=0.0, vtype=GRB.CONTINUOUS, name='X')
        # (MLU Only) Link utilization upper bound, may not be important based on what we need
        if self._problem_description.is_mlu:
            self._utility = MODEL.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='U')
    
    def _add_constraints(self):
        assert self._model is not None and self._flows is not None

        K = len(self._commodity_list)
        MODEL = self._model
        GRAPH = self._graph
        FLOWS = self._flows
        UTILITY = self._utility
        COMMODITIES = self._commodity_list

        # Capacity constraint
        print(as_info("Adding capacity constraints"))
        capacity_constraints: List[gurobipy.Constr] = []
        for e, (_, _, c_e) in ShortTQDMEnumerate(GRAPH.edges.data('capacity')):
            total_flow = gurobipy.LinExpr()
            for k in range(K):
                total_flow.addTerms(1, FLOWS[(e, k)])
            if self._problem_description.is_mlu:
                capacity_constraints.append(MODEL.addConstr(total_flow <= UTILITY * c_e))
            else:
                capacity_constraints.append(MODEL.addConstr(total_flow <= c_e))
        self._capacity_constraints = capacity_constraints

        """
        We use the expanded version of the constraint `MX_k = b_k.d_k`, which needs
        that for each node `v`, we have:

            - `sum(flow_in[k]) == sum(flow_out[k])` if the node is transit for `k`.
            - `sum(flow_out[k]) - sum(flow_in[k]) == +d_k` if the node is `src_k`.
            - `sum(flow_out[k]) - sum(flow_in[k]) == -d_k` if the node is `dst_k`.
        
        Now, the above combined with the capacity constraint may not be feasible at
        all, thus, we should probably relax the constraint such that the demand at
        the destination is less than or equal to `d_k` instead.
        With this:

            - `sum(flow_out[k]) == sum(flow_in[k])` if the node is transit for `k`.
            - `sum(flow_out[k]) - sum(flow_in[k]) <= +d_k` if the node is `src_k`.

        And instead, we force conservation from source to destination by:
        
                `sum(flow_out[src_k][k]) == sum(flow_in[dst_k][k])`
        
        We use the second form to make sure that the problem is always feasible.
        For the case of MLU, we expect that some problems can ideed be infeasible.
        """

        demand_constraints = []
        # For the case of Max-Flow / Max-Concurrent-Flow, we should just build the
        # objective now so that we won't traverse the commodity list again ...
        demand_objective = gurobipy.LinExpr()
        total_demand = 0
        print(as_info("Adding demand/flow-conservation constraints"))
        for k, commodity in ShortTQDMEnumerate(COMMODITIES):
            SOURCE = commodity.source
            DESTINATION = commodity.destination
            DEMAND = commodity.demand
            
            total_demand += DEMAND
            flow_out = defaultdict(gurobipy.LinExpr)
            flow_in = defaultdict(gurobipy.LinExpr)
            for e, edge in enumerate(GRAPH.edges()):
                flow_out[edge[0]].addTerms(1, FLOWS[(e, k)])
                flow_in[edge[1]].addTerms(1, FLOWS[(e, k)])
                if edge[0] == DESTINATION:
                    MODEL.addConstr(FLOWS[(e, k)] == 0)
                if edge[1] == SOURCE:
                    MODEL.addConstr(FLOWS[(e, k)] == 0)
            
            source_constraint = None
            destination_constraint = None
            for v in GRAPH.nodes():
                if v == SOURCE:
                    # Demand constraint from source
                    if self._problem_description.EvalParams.Objective == TEObjective.MLU:
                        source_constraint = MODEL.addConstr(flow_out[v] - flow_in[v] == DEMAND)
                    elif self._problem_description.EvalParams.Objective == TEObjective.MAX_FLOW:
                        source_constraint = MODEL.addConstr(flow_out[v] - flow_in[v] <= DEMAND)
                    else:
                        raise NotImplementedError
                    
                    # Update objective for non-MLU case
                    if self._problem_description.EvalParams.Objective != TEObjective.MLU:
                        demand_objective.add(flow_out[v], -1)
                elif v == DESTINATION:
                    # Demand constraint in destination
                    if self._problem_description.EvalParams.Objective == TEObjective.MLU:
                        destination_constraint = MODEL.addConstr(flow_in[v] - flow_out[v] == DEMAND)
                    elif self._problem_description.EvalParams.Objective == TEObjective.MAX_FLOW:
                        destination_constraint = MODEL.addConstr(flow_in[v] - flow_out[v] <= DEMAND)
                    else:
                        raise NotImplementedError
                else:
                    # Flow conservation in transit
                    MODEL.addConstr(flow_out[v] == flow_in[v])
            demand_constraints.append((source_constraint, destination_constraint))
        self._demand_constraints = demand_constraints

        # The regularizer objective is needed to ensure loops do not happen!
        # We use a linear regularizer for this.
        regularizer_objective = gurobipy.LinExpr()
        for k in range(K):
            for e in range(GRAPH.number_of_edges()):
                regularizer_objective.addTerms(0.01/total_demand, FLOWS[(e, k)])

        self._demand_objective = demand_objective
        self._regularizer_objective = regularizer_objective

    def _add_objective(self):
        assert self._model is not None and \
            self._flows is not None and \
            self._objective is None
        
        MODEL = self._model

        if self._problem_description.is_mlu:
            self._objective = self._utility + self._regularizer_objective
        else:
            self._objective = self._demand_objective + self._regularizer_objective
        MODEL.setObjective(self._objective, GRB.MINIMIZE)
    
    def close(self):
        self._model.close()
    
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
    
    def add_solution_elements(self, solution: TrafficEngineeringLPSolution):
        solution.add_solution_element(self._utility, 'utility')
        solution.add_solution_element(self._flows, 'assignments')
        # solution.add_solution_element(self._capacity_constraints, 'capacity_constraints')


import jsonargparse

def centralized_gurobi_solver_params_parser() -> jsonargparse.ArgumentParser:
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(GurobiSolverParams, 'SolverParams', help='Gurobi Solver Params')
    return parser


def parse_centralized_gurobi_solver_params(args: jsonargparse.Namespace) -> GurobiSolverParams:
    return GurobiSolverParams.make_from_args(args.SolverParams)
