import gurobipy
import networkx as nx
from typing import List, Tuple, Optional
from collections import defaultdict
from gurobipy import GRB, GurobiError, quicksum
from te.algorithms.base import TrafficEngineeringLP, SolverParams, GurobiSolverParams
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, Commodity
from topologies.utils import get_edge_indexing
from te.algorithms.utils import check_centralized_flow_conservation, check_capacity_constraint, make_model


class CentralizedEdgeBasedLP(TrafficEngineeringLP):
    def __init__(self, graph: nx.DiGraph, traffic: TrafficMatrixBase, solver_params: GurobiSolverParams) -> None:
        super().__init__()
        self._graph = graph
        self._edge_indexing = get_edge_indexing(graph)
        self._traffic = traffic
        self._solver_params: GurobiSolverParams = solver_params
        self._model: gurobipy.Model = None
        self._flows: gurobipy.MVar = None
        self._utility: gurobipy.Var = None
        self._objective: gurobipy.LinExpr = None
        self._commodity_list: List[Commodity] = traffic_to_commodity(self._traffic)
    
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
    
    def _make_variables(self):
        assert self._model is None and self._flows is None
        self._model = make_model(name='EdgeBasedTE', params=self._solver_params)

        """
        As per our formulation, the commodity matrix is of the form X_{ke},
        where each column is the split of a commodity `k` over the set of 
        edges in the graph.
        For `N` edges and `K` commodities, `X` would be a `N x K` matrix, where
        `X[e]` gives the flow of each commodity on edge `e`.
        """

        K = len(self._commodity_list)
        N = len(self._graph.edges)
        MODEL = self._model

        # This implicitly encodes the condition for `X_{ke} >= 0`
        self._flows = MODEL.addVars(N, K, lb=0.0, vtype=GRB.CONTINUOUS, name='X')
        # Link utilization upper bound, may not be important based on what we need
        self._utility = MODEL.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='U')
    
    def _add_constraints(self):
        assert self._model is not None and self._flows is not None

        MODEL = self._model
        GRAPH = self._graph
        FLOWS = self._flows
        UTILITY = self._utility
        COMMODITIES = self._commodity_list

        # Capacity constraint
        MODEL.addConstrs(
            FLOWS.sum(e, '*') / c_e <= UTILITY \
                for e, (_, _, c_e) in \
                    enumerate(GRAPH.edges.data('capacity'))
        )

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
        """

        for k, commodity in enumerate(COMMODITIES):
            SOURCE = commodity.source
            DESTINATION = commodity.destination
            DEMAND = commodity.demand

            flow_out = defaultdict(list)
            flow_in = defaultdict(list)
            for e, edge in enumerate(GRAPH.edges()):
                flow_out[edge[0]].append(FLOWS[e, k])
                flow_in[edge[1]].append(FLOWS[e, k])

            for v in GRAPH.nodes():
                if v == SOURCE:
                    # Demand constraint from source
                    MODEL.addConstr(quicksum(flow_out[v]) == DEMAND)
                    MODEL.addConstr(quicksum(flow_in[v]) == 0)
                elif v == DESTINATION:
                    # Demand constraint in destination
                    MODEL.addConstr(quicksum(flow_in[v]) == DEMAND)
                    MODEL.addConstr(quicksum(flow_out[v]) == 0)
                else:
                    # Flow conservation in transit
                    MODEL.addConstr(quicksum(flow_out[v]) == quicksum(flow_in[v]))

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
        if params:
            self.reset(with_params=True)
            self._params = params
            for key, value in self.params._asdict().items():
                self._model.setParam(key, value)
        try:
            self._model.optimize()
            if self._model.Status == gurobipy.GRB.OPTIMAL:
                return self._model.Runtime
            return -1
        except GurobiError as e:
            print(f'Error code {e.errno}: {e}')
            return -1

    def check(self, feasibility_tol: Optional[float] = None, feasibility_ratio: Optional[float] = None):
        assert (feasibility_tol is None) ^ (feasibility_ratio is None)
        PARAMS: GurobiSolverParams = self._solver_params
        check_centralized_flow_conservation(
            self._flows, self._graph, self._commodity_list,
            PARAMS.FeasibilityTol
        )
        check_capacity_constraint(
            self._flows, self._graph, self._commodity_list,
            feasibility_tol=feasibility_tol, feasibility_ratio=feasibility_ratio
        )
    
    def get_solution_commodity_list(self) -> List[Tuple[Commodity, Commodity]]:
        COMMODITIES = self._commodity_list
        FLOWS = self._flows
        GRAPH = self._graph
        INDICES = self._edge_indexing

        return [
            (
                Commodity(
                    source=commodity.source,
                    destination=commodity.destination,
                    demand=sum([
                        FLOWS[INDICES[(v, commodity.destination)], i].X \
                            for v in GRAPH.predecessors(commodity.destination)
                    ])
                ),
                Commodity(
                    source=commodity.source,
                    destination=commodity.destination,
                    demand=sum([
                        FLOWS[INDICES[(commodity.source, v)], i].X \
                            for v in GRAPH.successors(commodity.source)
                    ])
                )
            )
            for i, commodity in enumerate(COMMODITIES)
        ]
