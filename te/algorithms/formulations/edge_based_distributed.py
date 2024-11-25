import tqdm
import gurobipy
import numpy as np
import networkx as nx
import te.constants
from typing import List, Dict, Tuple
from dataclasses import dataclass
from gurobipy import GRB, GurobiError, quicksum
from te.algorithms.base import TrafficEngineeringLP, GurobiSolverParams, SolverParams
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, Commodity
from topologies.utils import get_edge_indexing, get_node_and_out_edge_index_mapping

@dataclass
class DistributedSolverParams(GurobiSolverParams):
    NumberOfEpochs: int = 1000
    EpsilonOE: float = te.constants.DEFAULT_EPSILON_OE
    EpsilonKE: float = te.constants.DEFAULT_EPSILON_KE
    Alpha: float = te.constants.DEFAULT_ALPHA
    Beta: float = te.constants.DEFAULT_BETA
    Seed: int = te.constants.DEFAULT_SEED


class DistributedEdgeBasedLP(TrafficEngineeringLP):
    def __init__(self, graph: nx.DiGraph, traffic: TrafficMatrixBase, solver_params: DistributedSolverParams) -> None:
        super().__init__()
        self._graph = graph
        self._edge_indexing = get_edge_indexing(graph)
        self._out_edge_mapping = get_node_and_out_edge_index_mapping(graph)
        self._in_edge_mapping: Dict[int, List[Tuple[int, int]]] = None
        self._out_degrees = {k: v for k, v in graph.out_degree()}
        self._traffic = traffic
        self._solver_params = solver_params
        self._rng = np.random.default_rng(seed=solver_params.Seed)
        self._commodity_list = traffic_to_commodity(self._traffic)
        self._env: gurobipy.Env = None
        self._model_controller: gurobipy.Model = None
        self._models_nodes: List[gurobipy.Model] = None
        self._objective_controller: gurobipy.QuadExpr = None
        self._objectives_nodes: List[gurobipy.QuadExpr] = None
        self._flows_ke: List[gurobipy.MVar] = None
        self._flows_oe: List[gurobipy.MVar] = None
        self._utility = None
        self._objective_trace = []
        
        self._lambda: List[np.ndarray] = None
        self._mu: np.ndarray = None

        self._prepare_in_edge_mapping()
        self._initiate_dual_weights()

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
        if self._objective_controller and self._objectives_nodes:
            return self._objective_controller.getValue() + sum([
                obj.getValue() for obj in self._objectives_nodes
            ])
        return None
    
    @property
    def objective_trace(self) -> List[float]:
        return self._objective_trace
    
    def _prepare_in_edge_mapping(self):
        """
        This creates a mapping from node index to a list of
        tuples of (pred_node_index, pred_edge_out_index).
        """

        M = len(self._graph.nodes)
        GRAPH = self._graph
        IN_EDGE_MAPPING: Dict[int, List[Tuple[int, int]]] = dict()
        
        for v in range(M):
            IN_EDGE_MAPPING[v] = list()
            for pred_v in GRAPH.predecessors(v):
                for i, (_, dst) in enumerate(GRAPH.out_edges(pred_v, data=False)):
                    if v == dst:
                        IN_EDGE_MAPPING[v].append([pred_v, i])

        self._in_edge_mapping = IN_EDGE_MAPPING
    
    def _f_oe(self, x_oe: gurobipy.Var) -> gurobipy.QuadExpr:
        return self._solver_params.EpsilonOE * x_oe ** 2
    
    def _f_ke(self, x_ke: gurobipy.Var) -> gurobipy.QuadExpr:
        return self._solver_params.EpsilonKE * x_ke ** 2
    
    def _initiate_dual_weights(self):
        """
        The dual weights are the Lagrange multipliers for the relaxed
        problem. These weights are:
            - `lambda`. In the original formulation, it is indexed per
              edge, but here, we index it two level:
                - Level 1 is with node index
                - Level 2 is with the out-going edge index
            - `mu`, a `K` by `M` matrix
        We initialize all of these with a uniform, random value.
        """

        M = len(self._graph.nodes)
        K = len(self._commodity_list)

        self._lambda = [
            self._rng.random(size=(self._out_degrees[i]))
                for i in range(M)
        ]
        self._mu = self._rng.random(size=(K, M))

    def _make_variables(self):
        """
        In this formulation, we have a total of `M+1` models.
            - One controller model, involves `N+1` variables:
                - `X_OE`, saved as attribute `_flows_oe`, which sums the total flow
                  on any edge `e`. It is indexed in 2 levels, the first one is the
                  node index of the source, and the second is the outgoing edge index.
                - `U`. saved as attribute `_utility`, the upper bound on the utility
                  of the network.
            - `M` node models (one per node), with `d_out` variables, where `d_out` is the
              out-degree of the node.
              All variables for this are saved in a list, `_flows_ke`, where each 
              element (indexed by node index in the graph), is a `d_out` by `K` matrix.
        """

        assert self._model_controller is None and self._models_nodes is None
        
        M = len(self._graph.nodes)
        K = len(self._commodity_list)
        OUT_DEGREES = self._out_degrees
        GRAPH = self._graph

        ENV = gurobipy.Env()
        ENV.setParam('OutputFlag', 0)
        ENV.start()
        self._env = ENV

        self._model_controller = gurobipy.Model('EdgeBasedDistributedTE_Controller', env=ENV)
        self._models_nodes = [
            gurobipy.Model(f'EdgeBasedDistributedTE_Node_{i}', env=ENV) for i in range(M)
        ]
    
        MODEL_CONTROLLER = self._model_controller
        MODEL_NODES = self._models_nodes

        self._flows_oe = [
            MODEL_CONTROLLER.addVars(OUT_DEGREES[v], lb=0.0, vtype=GRB.CONTINUOUS, name=f'X_OE_{v}') \
                for v in range(M)
        ]
        self._flows_ke = [
            node_model.addVars(GRAPH.out_degree(i), K, lb=0.0, vtype=GRB.CONTINUOUS, name=f'X_KE_{i}') \
                for i, node_model in enumerate(MODEL_NODES)
        ]
        self._utility = MODEL_CONTROLLER.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='U')

    def _add_constraints(self):
        assert self._model_controller is not None and self._models_nodes is not None

        M = len(self._graph.nodes)
        MODEL_CONTROLLER = self._model_controller
        GRAPH = self._graph
        X_OE = self._flows_oe
        UTILITY = self._utility

        # Capacity constraint
        for v in range(M):
            MODEL_CONTROLLER.addConstrs(
                # X_OE[v][i] <= c_e
                X_OE[v][i] / c_e <= UTILITY
                    for i, (_, _, c_e) in enumerate(GRAPH.out_edges(v, data='capacity'))
            )

        # The node problem is NOT constrained beyond `X_KE >= 0` !
    
    def _update_controller_objective(self):
        M = len(self._graph.nodes)
        OUT_DEGREES = self._out_degrees
        MODEL_CONTROLLER = self._model_controller
        UTILITY = self._utility
        X_OE = self._flows_oe
        LAMBDA = self._lambda

        """
        Controller objective is:

            u^2 + \sum_{v \in V} \sum_{e \in E_v^{out}} f_oe(X_oe) - \lambda_e X_{oe}
        """

        OBJECTIVE_CONTROLLER = \
            UTILITY ** 2 + \
            quicksum([
                quicksum([
                    self._f_oe(X_OE[v][i]) - LAMBDA[v][i] * X_OE[v][i] \
                        for i in range(OUT_DEGREES[v])
                ]) for v in range(M)
            ])
        MODEL_CONTROLLER.setObjective(OBJECTIVE_CONTROLLER, GRB.MINIMIZE)
        self._objective_controller = OBJECTIVE_CONTROLLER
    
    def _update_node_objectives(self):
        M = len(self._graph.nodes)
        K = len(self._commodity_list)
        OUT_DEGREES = self._out_degrees
        OUT_EDGE_MAPPING = self._out_edge_mapping
        MODEL_NODES = self._models_nodes
        X_KE = self._flows_ke
        LAMBDA = self._lambda
        MU = self._mu

        """
        Node objective is:

            \sum_k \sum_v \sum_{e \in E_v^{out}} (
                f_{ke}(X_{ke}) + X_{ke} (\lambda_e + mu_{kv} - mu_{kv'}) 
            )
        """

        OBJECTIVE_NODES = [
            quicksum([
                quicksum([
                    self._f_ke(X_KE[v][i, k]) + X_KE[v][i, k] * (
                        LAMBDA[v][i] + MU[k, v] - MU[k, OUT_EDGE_MAPPING[(v, i)][1]]
                    ) for i in range(OUT_DEGREES[v])
                ]) for k in range(K)
            ]) for v in range(M)
        ]
        assert len(OBJECTIVE_NODES) == len(MODEL_NODES)
        for node_obj, node_model in zip(OBJECTIVE_NODES, MODEL_NODES):
            node_model.setObjective(node_obj, GRB.MINIMIZE)
        self._objectives_nodes = OBJECTIVE_NODES
    
    def _add_objective(self):
        assert self._model_controller is not None and self._models_nodes is not None

        self._update_controller_objective()
        self._update_node_objectives()
    
    def _update_lambda(self):
        assert self._model_controller is not None and self._models_nodes is not None

        """
        The update rule for lambda_e is:
            \lambda_e \gets [\lambda_e + alpha * (\sum_k X_{ke} - X_{oe})]
        """

        M = len(self._graph.nodes)
        OUT_DEGREES = self._out_degrees
        X_OE = self._flows_oe
        X_KE = self._flows_ke
        LAMBDA = self._lambda
        PARAMS = self._solver_params

        for v in range(M):
            for i in range(OUT_DEGREES[v]):
                LAMBDA[v][i] += PARAMS.Alpha * (
                    X_KE[v].sum(i, '*').getValue() - X_OE[v][i].X
                )
                if LAMBDA[v][i] < 0:
                    LAMBDA[v][i] = 0
    
    def _update_mu(self, v: int):
        assert self._model_controller is not None and self._models_nodes is not None

        """
        The update rule for mu_{kv} is:
        """

        X_KE = self._flows_ke
        COMMODITIES = self._commodity_list
        IN_EDGE_MAPPING = self._in_edge_mapping
        PARAMS = self._solver_params
        MU = self._mu

        for k, commodity in enumerate(COMMODITIES):
            flow_out_k = X_KE[v].sum('*', k).getValue()
            flow_in_k = quicksum([
                X_KE[v_prime][i, k]
                    for v_prime, i in IN_EDGE_MAPPING[v]
            ]).getValue()

            if v == commodity.source:
                MU[k, v] += PARAMS.Beta * (flow_out_k - flow_in_k - commodity.demand)
            elif v == commodity.destination:
                MU[k, v] += PARAMS.Beta * (flow_out_k - flow_in_k + commodity.demand)
            else:
                MU[k, v] += PARAMS.Beta * (flow_out_k - flow_in_k)
    
    def close(self):
        self._model_controller.close()
        for model in self._models_nodes:
            model.close()
        if self._env:
            self._env.close()
    
    def make_lp(self):
        self._make_variables()
        self._add_constraints()
        self._add_objective()
    
    def reset(self, with_params: False):
        self._model_controller.reset()
        if with_params:
            self._model_controller.resetParams()
        for model in self._models_nodes:
            model.reset()
            if with_params:
                model.resetParams()
    
    def solve(self, params: SolverParams = None) -> float:
        assert params is None
        
        NODE_MODELS = self._models_nodes
        CONTROLLER_MODEL = self._model_controller
        M = len(NODE_MODELS)

        total_runtime = 0
        try:
            # with ProcessingPool(8) as pool:
            for _ in tqdm.tqdm(range(self._solver_params.NumberOfEpochs)):
                t_nodes = []
                # First, update `X_{ke}` at each node (along with `mu`)
                for model in NODE_MODELS:
                    model.optimize()
                for v in range(M):
                    t_nodes.append(NODE_MODELS[v].Runtime)
                    self._update_mu(v)
                
                # Now, update `lambda` and the controller objective
                CONTROLLER_MODEL.optimize()
                self._update_lambda()

                # Finally, update objectives and prepare for next epoch
                self._update_controller_objective()
                self._update_node_objectives()

                total_runtime += (max(t_nodes) + CONTROLLER_MODEL.Runtime)
                self._objective_trace.append(self._utility.X)
            return total_runtime
        except GurobiError as e:
            print(f'Error code {e.errno}: {e}')
            return -1

    def get_solution_commodity_list(self) -> List[Commodity]:
        COMMODITIES = self._commodity_list
        X_KE = self._flows_ke
        X_OE = self._flows_oe
        OUT_DEGREE = self._out_degrees

        for v in range(len(X_OE)):
            for i in range(OUT_DEGREE[v]):
                print(X_OE[v][i].X)

        # TODO: WHY DO WE NEED `max` !?

        return [
            Commodity(
                source=commodity.source,
                destination=commodity.destination,
                demand=X_KE[commodity.source].sum('*', i).getValue()
            )
            for i, commodity in enumerate(COMMODITIES)
        ]
    
    # def get_ratio_of_unsatisfied_demands(self, params: DistributedSolverParams, solution: List[Commodity] = None) -> float:
    #     COMMODITIES = self._commodity_list
    #     K = len(COMMODITIES)
    #     if solution is None:
    #         solution = self.get_solution_commodity_list()
    #     assert len(solution) == K

    #     k = 0
    #     for actual, ideal in zip(solution, COMMODITIES):
    #         assert actual.source == ideal.source
    #         assert actual.destination == ideal.destination
    #         if abs(actual.demand - ideal.demand) > params.FeasibilityTol * 2:
    #             print(f"COULD NOT SATISFY {actual.source} -> {actual.destination}: {actual.demand} vs {ideal.demand}")
    #             k += 1

    #     return k / K
