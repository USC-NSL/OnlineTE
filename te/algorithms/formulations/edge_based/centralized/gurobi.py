import gurobipy
import numpy as np
from typing import List, Tuple, Optional
from collections import defaultdict
from gurobipy import GRB
from te.algorithms.base import *
from te.traffic_models.base import *
from utils.logging import as_info, ShortTQDMEnumerate, ShortTQDM
from utils.gurobi_utils import *
from te.algorithms.objective_evaluators import get_total_routed_flow
from topologies.utils import get_graph_M_matrix


class GurobiTE(TELP[GurobiSolverParams]):
    """
    An honest implementation of edge-based MLU with Gurobi.
    Becomes too sluggish for very large topologies, but solutions look very nice.
    """
    def __init__(self, problem_description: TEProblemDescription, solver_params: GurobiSolverParams) -> None:
        super().__init__(problem_description, solver_params)
        self._env: Optional[gurobipy.Env] = None
        self._model: Optional[gurobipy.Model] = None
        self._flows: Optional[gurobipy.tupledict] = None
        self._utility: Optional[gurobipy.Var] = None
        self._objective: Optional[gurobipy.LinExpr] = None
        self._demand_constraints: Optional[List[Tuple[gurobipy.Constr, gurobipy.Constr]]] = None
        self._capacity_constraints: Optional[gurobipy.tupledict] = None

        self._demand_objective: Optional[gurobipy.LinExpr] = None
        self._regularizer_objective: Optional[gurobipy.LinExpr] = None
    
    @property
    def alg_name(self) -> str:
        return 'Centralized-Gurobi'

    @property
    def current_objective(self) -> float:
        return abs(self._objective.getValue())

    def _set_X_ek(self):
        K = self.number_of_commodities
        N = self.number_of_edges
        FLOWS = self._flows
        X_EK = np.ndarray(shape=(N, K))
        for k in range(K):
            for e in range(N):
                X_EK[e, k] = FLOWS[(e, k)].X
        self._X_ek = X_EK
        print(f"Total routed flow: {get_total_routed_flow(self._X_ek, get_graph_M_matrix(self._graph))}")
    
    def _make_variables(self):
        assert self._model is None and self._flows is None

        """
        As per our formulation, the commodity matrix is of the form X_{ke},
        where each column is the split of a commodity `k` over the set of 
        edges in the graph.
        For `N` edges and `K` commodities, `X` would be a `N x K` matrix, where
        `X[e]` gives the flow of each commodity on edge `e`.
        """

        K = self.number_of_commodities
        N = self.number_of_edges
        
        ENV = gurobipy.Env()
        ENV.start()
        self._env = ENV
        MODEL = make_model(
            name='EdgeBasedTE', params=self._solver_params,
            feasibility_tolerance=self._problem_description.eval_params.feasibility_tolerance,
            optimality_tolerance=self._problem_description.eval_params.optimality_tolerance,
            verbose=self._problem_description.eval_params.verbose,
            env=ENV
        )
        self._model = MODEL

        # This implicitly encodes the condition for `X_{ke} >= 0`
        print(as_info("Adding commodity assignment variables"))
        self._flows = MODEL.addVars(N, K, lb=0.0, vtype=GRB.CONTINUOUS, name='X')
        # (MLU Only) Link utilization upper bound
        if self.objective == TEObjective.MLU:
            self._utility = MODEL.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='U')
    
    def _add_constraints(self):
        assert self._model is not None and self._flows is not None

        K = self.number_of_commodities
        M = self.number_of_nodes
        N = self.number_of_edges
        MODEL = self._model
        GRAPH = self._graph
        FLOWS = self._flows
        # We build the model with a demand of clear for every pair
        DEMAND = 1

        # Capacity constraint
        print(as_info("Adding capacity constraints"))
        capacity_constraints: List[gurobipy.Constr] = []
        for e, (_, _, c_e) in ShortTQDMEnumerate(GRAPH.edges.data('capacity')):
            total_flow = gurobipy.LinExpr()
            for k in range(K):
                total_flow.addTerms(1, FLOWS[(e, k)])
            match self.objective:
                case TEObjective.MLU:
                    capacity_constraints.append(MODEL.addConstr(total_flow <= self._utility * c_e))
                case _ :
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
        print(as_info("Adding demand/flow-conservation constraints"))
        k = 0
        for od_pair in ShortTQDM(np.ndindex((M, M)), M**2):
            SOURCE, DESTINATION = od_pair
            if SOURCE == DESTINATION:
                continue
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
                    match self.objective:
                        case TEObjective.MLU:
                            source_constraint = MODEL.addConstr(flow_out[v] - flow_in[v] == DEMAND)
                        case TEObjective.MAX_FLOW:
                            source_constraint = MODEL.addConstr(flow_out[v] - flow_in[v] <= DEMAND)
                            demand_objective.add(flow_out[v], -1)
                        case _ : raise ValueError
                elif v == DESTINATION:
                    # Demand constraint in destination
                    match self.objective:
                        case TEObjective.MLU:
                            destination_constraint = MODEL.addConstr(flow_in[v] - flow_out[v] == DEMAND)
                        case TEObjective.MAX_FLOW:
                            destination_constraint = MODEL.addConstr(flow_in[v] - flow_out[v] <= DEMAND)
                        case _ : raise ValueError
                else:
                    # Flow conservation in transit
                    MODEL.addConstr(flow_out[v] == flow_in[v])
            demand_constraints.append((source_constraint, destination_constraint))
            k += 1
        self._demand_constraints = demand_constraints

        # The regularizer objective is needed to ensure loops do not happen!
        # We use a linear regularizer for this.
        regularizer_objective = gurobipy.LinExpr()
        for k in range(K):
            for e in range(N):
                regularizer_objective.addTerms(1, FLOWS[(e, k)])

        self._demand_objective = demand_objective
        self._regularizer_objective = regularizer_objective

    def _add_objective(self):
        assert self._model is not None and \
            self._flows is not None and \
            self._objective is None
        
        MODEL = self._model
        match self.objective:
            case TEObjective.MLU:
                self._objective = gurobipy.LinExpr(1.0, self._utility)
                MODEL.setObjectiveN(self._objective, index=0, priority=10, name='MLU')
                # MODEL.setObjectiveN(self._regularizer_objective, index=1, priority=1, name='Loop')
            case TEObjective.MAX_FLOW:
                self._objective = self._demand_objective
                MODEL.setObjectiveN(self._objective, index=0, priority=10, name='MaxFlow')
                # MODEL.setObjectiveN(self._regularizer_objective, index=1, priority=1, name='Loop')
            case _ : raise ValueError
    
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
        for constraints, commodity in zip(self._demand_constraints, COMMODITIES):
            DEMAND = commodity.demand
            source_constraint, destination_constraint = constraints
            source_constraint.RHS = DEMAND
            destination_constraint.RHS = DEMAND

    def _update_objective(self, tm: np.ndarray):
        pass


import jsonargparse

def centralized_gurobi_solver_params_parser() -> jsonargparse.ArgumentParser:
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(GurobiSolverParams, 'SolverParams', help='Gurobi Solver Params')
    return parser


def parse_centralized_gurobi_solver_params(args: jsonargparse.Namespace) -> GurobiSolverParams:
    return GurobiSolverParams.make_from_args(args.SolverParams)
