import tqdm
import gurobipy
import numpy as np
import networkx as nx
import te.constants
from collections import defaultdict
from typing import List, Dict, Tuple
from dataclasses import dataclass
from gurobipy import GRB, GurobiError, quicksum
from te.algorithms.base import TrafficEngineeringLP, GurobiSolverParams, SolverParams
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, Commodity
from topologies.utils import (get_edge_indexing, get_node_and_out_edge_index_mapping, 
                              get_in_edge_mapping, get_edge_to_out_index_mapping)

@dataclass
class DistributedADMMSolverParams(GurobiSolverParams):
    NumberOfEpochs: int = 1000
    EpsilonKE: float = 1e-2
    Rho: float = te.constants.DEFAULT_RHO
    Beta: float = te.constants.DEFAULT_BETA
    NumberOfNetworkUpdates: int = te.constants.DEFAULT_NUMBER_OF_NETWORK_UPDATES
    Seed: int = te.constants.DEFAULT_SEED


class DistributedEdgeBasedADMMLP(TrafficEngineeringLP):
    def __init__(self, graph: nx.DiGraph, traffic: TrafficMatrixBase, solver_params: DistributedADMMSolverParams) -> None:
        super().__init__()
        self._graph = graph
        self._traffic = traffic
        self._solver_params = solver_params
        self._rng = np.random.default_rng(seed=solver_params.Seed)
        self._commodity_list = traffic_to_commodity(self._traffic)

        self._edge_indexing = get_edge_indexing(graph)
        self._out_edge_mapping = get_node_and_out_edge_index_mapping(graph)
        self._in_edge_mapping: Dict[int, List[Tuple[int, int]]] = get_in_edge_mapping(graph)
        self._edge_out_indexing = get_edge_to_out_index_mapping(graph)
        self._out_degrees = {k: v for k, v in graph.out_degree()}

        self._env: gurobipy.Env = None

        self._model_controller: gurobipy.Model = None
        self._model_nodes: List[gurobipy.Model] = None
        self._objective_controller: gurobipy.QuadExpr = None
        self._objective_nodes: List[gurobipy.QuadExpr] = None
        
        self._X_ke: List[gurobipy.MVar] = None
        self._X_oe: List[gurobipy.MVar] = None
        self._Z_oe: List[np.ndarray] = None
        self._utility: gurobipy.Var = None
        
        self._re: List[np.ndarray] = None
        self._mu: np.ndarray = None

        # self._residual: Dict[Tuple[int, int], gurobipy.LinExpr] = None

        self._objective_trace = []

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
        raise NotImplementedError
    
    @property
    def objective_trace(self) -> List[float]:
        return self._objective_trace

    def _f_ke(self, x_ke: gurobipy.Var) -> gurobipy.QuadExpr:
        return self._solver_params.EpsilonKE * x_ke ** 2
    
    def _initiate_dual_weights(self):
        M = len(self._graph.nodes)
        K = len(self._commodity_list)

        self._re = [
            self._rng.random(size=(self._out_degrees[i]))
                for i in range(M)
        ]
        self._mu = self._rng.random(size=(K, M))

    def _make_variables(self):
        """
        This is again very similar to primal/dual ascent, but the additional
        variable here would be `Z_oe`.
        """

        assert self._model_controller is None and \
               self._model_nodes is None
        
        M = len(self._graph.nodes)
        K = len(self._commodity_list)
        OUT_DEGREES = self._out_degrees

        ENV = gurobipy.Env()
        ENV.setParam('OutputFlag', 0)
        ENV.start()
        self._env = ENV

        self._model_controller = gurobipy.Model('EdgeBasedDistributedTE_Controller', env=ENV)
        self._model_nodes = [
            gurobipy.Model(f'EdgeBasedDistributedTE_Node_{i}', env=ENV) for i in range(M)
        ]
    
        MODEL_CONTROLLER = self._model_controller
        MODEL_NODES = self._model_nodes
        
        self._X_oe = [
            MODEL_CONTROLLER.addVars(OUT_DEGREES[v], lb=0.0, vtype=GRB.CONTINUOUS, name=f'X_OE_{v}') \
                for v in range(M)
        ]
        self._utility = MODEL_CONTROLLER.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='U')

        # This need not be managed by Gurobi, since each step has a closed form solution
        self._Z_oe = [np.zeros((OUT_DEGREES[v],)) for v in range(M)]

        self._X_ke = [
            node_model.addVars(OUT_DEGREES[v], K, lb=0.0, vtype=GRB.CONTINUOUS, name=f'X_KE_{v}') \
                for v, node_model in enumerate(MODEL_NODES)
        ]

        # self._make_residual()
    
    # def _make_residual(self):
    #     assert self._model_controller is not None and \
    #            self._model_nodes is not None
        
    #     M = len(self._graph.nodes)
    #     K = len(self._commodity_list)
    #     COMMODITIES = self._commodity_list
    #     OUT_DEGREES = self._out_degrees
    #     OUT_EDGE_MAPPING = self._out_edge_mapping
    #     X_KE = self._X_ke

    #     flow_out = {k: defaultdict(list) for k in range(K)}
    #     flow_in = {k: defaultdict(list) for k in range(K)}
    #     residual: Dict[Tuple[int, int], gurobipy.LinExpr] = dict()

    #     for k, commodity in enumerate(COMMODITIES):
    #         SOURCE = commodity.source
    #         DESTINATION = commodity.destination
    #         DEMAND = commodity.demand

    #         for v in range(M):
    #             for i in range(OUT_DEGREES[v]):
    #                 dst = OUT_EDGE_MAPPING[(v, i)][1]
    #                 flow_out[k][v].append(X_KE[v][i, k])
    #                 flow_in[k][dst].append(X_KE[v][i, k])
        
    #         for v in range(M):
    #             if v == SOURCE:
    #                 # Demand constraint from source
    #                 residual[(k, v)] = quicksum(flow_out[v]) - DEMAND
    #             elif v == DESTINATION:
    #                 # Demand constraint in destination
    #                 residual[(k, v)] = DEMAND - quicksum(flow_in[v])
    #             else:
    #                 # Flow conservation in transit
    #                 residual[(k, v)] = quicksum(flow_out[v]) - quicksum(flow_in[v])
        
    #     self._residual = residual

    def _add_constraints(self):
        """
        This is exactly the same as before.
        """

        assert self._model_controller is not None and \
               self._model_nodes is not None

        M = len(self._graph.nodes)
        GRAPH = self._graph
        OUT_INDEX_MAPPING = self._edge_out_indexing
        X_OE = self._X_oe
        X_KE = self._X_ke
        UTILITY = self._utility
        COMMODITIES = self._commodity_list
        MODEL_CONTROLLER = self._model_controller
        MODEL_NODES = self._model_nodes

        # Capacity constraint
        for v in range(M):
            MODEL_CONTROLLER.addConstrs(
                X_OE[v][i] / c_e <= UTILITY
                    for i, (_, _, c_e) in enumerate(GRAPH.out_edges(v, data='capacity'))
            )
        
        # Demand constraints
        for k, commodity in enumerate(COMMODITIES):
            SOURCE = commodity.source
            DESTINATION = commodity.destination
            DEMAND = commodity.demand

            flow_out = defaultdict(list)
            flow_in = defaultdict(list)
            for edge in GRAPH.edges():
                v = edge[0]
                i = OUT_INDEX_MAPPING[edge]
                flow_out[edge[0]].append(X_KE[v][i, k])
                flow_in[edge[1]].append(X_KE[v][i, k])
            
            for v in GRAPH.nodes():
                node_model = MODEL_NODES[v]
                if v == SOURCE:
                    # Demand constraint from source
                    node_model.addConstr(quicksum(flow_out[v]) == DEMAND)
                    node_model.addConstr(quicksum(flow_in[v]) == 0)
                elif v == DESTINATION:
                    # Demand constraint in destination
                    node_model.addConstr(quicksum(flow_in[v]) == DEMAND)
                    node_model.addConstr(quicksum(flow_out[v]) == 0)
                else:
                    # Flow conservation in transit
                    node_model.addConstr(quicksum(flow_out[v]) == quicksum(flow_in[v]))
    
    def _update_controller_objective(self):
        M = len(self._graph.nodes)
        OUT_DEGREES = self._out_degrees
        UTILITY = self._utility
        X_OE = self._X_oe
        Z_OE = self._Z_oe
        RE = self._re
        RHO = self._solver_params.Rho
        MODEL_CONTROLLER = self._model_controller

        """
        Controller objective is:
            u + rho/2 sum_e (X_oe - Z_oe + r_e)^2
        """

        OBJECTIVE_CONTROLLER = \
            UTILITY + \
            RHO/2 * quicksum([
                quicksum([
                    (X_OE[v][i] - Z_OE[v][i] + RE[v][i]) ** 2
                        for i in range(OUT_DEGREES[v])
                ]) for v in range(M)
            ])
        
        MODEL_CONTROLLER.setObjective(OBJECTIVE_CONTROLLER, GRB.MINIMIZE)
        self._objective_controller = OBJECTIVE_CONTROLLER
    
    def _update_node_objective(self):
        M = len(self._graph.nodes)
        K = len(self._commodity_list)
        OUT_DEGREES = self._out_degrees
        Z_OE = self._Z_oe
        X_KE = self._X_ke
        RE = self._re
        MU = self._mu
        RHO = self._solver_params.Rho
        BETA = self._solver_params.Beta
        # RESIDUAL = self._residual
        MODEL_NODES = self._model_nodes

        """
        The node objective is:
            \sum_{e \in E_v^{out}} (
                \sum_k (f_ke(X_ke) + rho/2 * (\sum_k X_ke - Z_oe - r_e)^2 
            ) + 
            \sum_k (
                \mu_kv * (
                    \sum_{e \in E_v^{out}}{X_ke} - \sum_{e \in E_v^{in}}{X_ke} - b_k d_k
                )
            ) + 
            \sum_k (
                beta/2 * (
                    \sum_{e \in E_v^{out}}{X_ke} - \sum_{e \in E_v^{in}}{X_ke} - b_k d_k
                )^2
            )
        """

        OBJECTIVE_NODES = [
            quicksum([
                RHO/2 * (X_KE[v].sum(i, '*') - Z_OE[v][i] - RE[v][i]) ** 2
                for i in range(OUT_DEGREES[v])
            ])
            # ]) + quicksum([
            #     MU[k, v] * RESIDUAL[(k, v)] for k in range(K)
            # ]) + quicksum([
            #     BETA/2 * RESIDUAL[(k, v)] ** 2 for k in range(K)
            # ])
            for v in range(M)
        ]
        assert len(OBJECTIVE_NODES) == len(MODEL_NODES)
        for node_obj, node_model in zip(OBJECTIVE_NODES, MODEL_NODES):
            node_model.setObjective(node_obj, GRB.MINIMIZE)
        self._objectives_nodes = OBJECTIVE_NODES
    
    def _add_objective(self):
        assert self._model_controller is not None and \
               self._model_nodes is not None

        self._update_controller_objective()
        self._update_node_objective()
    
    def _update_re(self):
        assert self._model_controller is not None and \
               self._model_nodes is not None

        """
        The update rule for r_e is:
            r_e \gets r_e + (X_oe - \sum_k X_ke)/2
        """

        M = len(self._graph.nodes)
        OUT_DEGREES = self._out_degrees
        X_OE = self._X_oe
        X_KE = self._X_ke
        RE = self._re

        for v in range(M):
            for i in range(OUT_DEGREES[v]):
                RE[v][i] += (
                    X_OE[v][i].X - X_KE[v].sum(i, '*').getValue()
                ) / 2
    
    def _update_Z_oe(self):
        assert self._model_controller is not None and \
               self._model_nodes is not None

        """
        The update rule for lambda_e is:
            Z_oe \gets (X_oe + \sum_k X_ke)/2
        """

        M = len(self._graph.nodes)
        OUT_DEGREES = self._out_degrees
        X_KE = self._X_ke
        X_OE = self._X_oe
        Z_OE = self._Z_oe

        for v in range(M):
            for i in range(OUT_DEGREES[v]):
                Z_OE[v][i] = (
                    X_OE[v][i].X + X_KE[v].sum(i, '*').getValue()
                ) / 2

    # def _update_mu(self, v: int):
    #     assert self._model_controller is not None and \
    #            self._model_nodes is not None
        
    #     PARAMS = self._solver_params
    #     MU = self._mu
    #     RESIDUAL = self._residual

    #     for (k, v), value in RESIDUAL.items():
    #         MU[(k, v)] += PARAMS.Beta * value.getValue()
    
    def close(self):
        self._model_controller.close()
        for node_model in self._model_nodes:
            node_model.close()
        if self._env:
            self._env.close()
    
    def make_lp(self):
        self._make_variables()
        self._add_constraints()
        self._add_objective()
    
    def reset(self, with_params: False):
        self._model_controller.reset()
        for node_model in self._model_nodes:
            node_model.reset()
        if with_params:
            self._model_controller.resetParams()
            for node_model in self._model_nodes:
                node_model.resetParams()
    
    def solve(self, params: SolverParams = None) -> float:
        assert params is None
        
        MODEL_CONTROLLER = self._model_controller
        MODEL_NODES = self._model_nodes
        PARAMS = self._solver_params

        total_runtime = 0
        t_nodes = {v: [] for v in range(len(MODEL_NODES))}

        try:
            for _ in tqdm.tqdm(range(PARAMS.NumberOfEpochs)):
                # First, concurrently solve both controller and network models
                MODEL_CONTROLLER.optimize()
                if MODEL_CONTROLLER.Status != GRB.OPTIMAL:
                    raise RuntimeError(f"Optimization for controller returned non-optimal status: {MODEL_CONTROLLER.Status}")
                # for _ in range(PARAMS.NumberOfNetworkUpdates):
                for v, node_model in enumerate(MODEL_NODES):
                    node_model.optimize()
                    if node_model.Status != GRB.OPTIMAL:
                        raise RuntimeError(f"Optimization for node {v} returned non-optimal status: {node_model.Status}")
                    t_nodes[v].append(node_model.Runtime)
                # for v in range(len(MODEL_NODES)):
                #     self._update_mu(v)
                # self._update_node_objective()

                # Second, update Z_oe and r_e
                self._update_Z_oe()
                self._update_re()

                # Update the objective and start again
                self._update_controller_objective()
                self._update_node_objective()

                # Houskeeping
                self._objective_trace.append(self._utility.X)
                # print(f"Controller solver time: {MODEL_CONTROLLER.Runtime}")
                total_node = max(sum(t_nodes[v]) for v in range(len(MODEL_NODES)))
                # print(f"Node solver time: {total_node}")
                total_runtime += max(MODEL_CONTROLLER.Runtime, total_node)
                for k in t_nodes.keys():
                    t_nodes[k].clear()

            return total_runtime
        except GurobiError as e:
            print(f'Error code {e.errno}: {e}')
            return -1

    def get_solution_commodity_list(self) -> List[Commodity]:
        COMMODITIES = self._commodity_list
        X_KE = self._X_ke
        X_OE = self._X_oe
        Z_OE = self._Z_oe
        OUT_DEGREE = self._out_degrees

        for v in range(len(X_OE)):
            for i in range(OUT_DEGREE[v]):
                print(f"X = {X_OE[v][i].X} vs. Z = {Z_OE[v][i]}")

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


class SemiDistributedEdgeBasedADMMLP(TrafficEngineeringLP):
    def __init__(self, graph: nx.DiGraph, traffic: TrafficMatrixBase, solver_params: DistributedADMMSolverParams) -> None:
        super().__init__()
        self._graph = graph
        self._traffic = traffic
        self._solver_params = solver_params
        self._rng = np.random.default_rng(seed=solver_params.Seed)
        self._commodity_list = traffic_to_commodity(self._traffic)

        self._edge_indexing = get_edge_indexing(graph)
        self._out_edge_mapping = get_node_and_out_edge_index_mapping(graph)
        self._in_edge_mapping: Dict[int, List[Tuple[int, int]]] = get_in_edge_mapping(graph)
        self._edge_out_indexing = get_edge_to_out_index_mapping(graph)
        self._out_degrees = {k: v for k, v in graph.out_degree()}

        self._env: gurobipy.Env = None

        self._model_controller: gurobipy.Model = None
        self._model_network: gurobipy.Model = None
        self._objective_controller: gurobipy.QuadExpr = None
        self._objective_network: gurobipy.QuadExpr = None
        
        self._X_ke: List[gurobipy.MVar] = None
        self._X_oe: List[gurobipy.MVar] = None
        self._Z_oe: List[np.ndarray] = None
        self._utility: gurobipy.Var = None
        
        self._re: List[np.ndarray] = None

        self._objective_trace = []

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
        raise NotImplementedError
    
    @property
    def objective_trace(self) -> List[float]:
        return self._objective_trace
    
    def _initiate_dual_weights(self):
        M = len(self._graph.nodes)

        self._re = [
            self._rng.random(size=(self._out_degrees[i]))
                for i in range(M)
        ]

    def _make_variables(self):
        """
        This is again very similar to primal/dual ascent, but the additional
        variable here would be `Z_oe`.
        """

        assert self._model_controller is None and \
               self._model_network is None
        
        M = len(self._graph.nodes)
        K = len(self._commodity_list)
        OUT_DEGREES = self._out_degrees

        ENV = gurobipy.Env()
        ENV.setParam('OutputFlag', 0)
        ENV.start()
        self._env = ENV

        self._model_controller = gurobipy.Model('EdgeBasedDistributedTE_Controller', env=ENV)
        self._model_network = gurobipy.Model('EdgeBasedDistributedTE_Network', env=ENV)
    
        MODEL_CONTROLLER = self._model_controller
        MODEL_NETWORK = self._model_network
        
        self._X_oe = [
            MODEL_CONTROLLER.addVars(OUT_DEGREES[v], lb=0.0, vtype=GRB.CONTINUOUS, name=f'X_OE_{v}') \
                for v in range(M)
        ]
        self._utility = MODEL_CONTROLLER.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='U')

        # This need not be managed by Gurobi, since each step has a closed form solution
        self._Z_oe = [np.zeros((OUT_DEGREES[v],)) for v in range(M)]

        self._X_ke = [
            MODEL_NETWORK.addVars(OUT_DEGREES[v], K, lb=0.0, vtype=GRB.CONTINUOUS, name=f'X_KE_{v}') \
                for v in range(M)
        ]

    def _add_constraints(self):
        """
        This is exactly the same as before.
        """

        assert self._model_controller is not None and \
               self._model_network is not None

        M = len(self._graph.nodes)
        GRAPH = self._graph
        X_OE = self._X_oe
        X_KE = self._X_ke
        UTILITY = self._utility
        COMMODITIES = self._commodity_list
        MODEL_CONTROLLER = self._model_controller
        MODEL_NETWORK = self._model_network
        OUT_INDEX_MAPPING = self._edge_out_indexing

        # Capacity constraint
        for v in range(M):
            MODEL_CONTROLLER.addConstrs(
                X_OE[v][i] / c_e <= UTILITY
                    for i, (_, _, c_e) in enumerate(GRAPH.out_edges(v, data='capacity'))
            )
        
        # Network model constraints (i.e. demand satisfaction)
        for k, commodity in enumerate(COMMODITIES):
            SOURCE = commodity.source
            DESTINATION = commodity.destination
            DEMAND = commodity.demand

            flow_out = defaultdict(list)
            flow_in = defaultdict(list)
            for edge in GRAPH.edges():
                v = edge[0]
                i = OUT_INDEX_MAPPING[edge]
                flow_out[edge[0]].append(X_KE[v][i, k])
                flow_in[edge[1]].append(X_KE[v][i, k])

            for v in GRAPH.nodes():
                if v == SOURCE:
                    # Demand constraint from source
                    MODEL_NETWORK.addConstr(quicksum(flow_out[v]) == DEMAND)
                    MODEL_NETWORK.addConstr(quicksum(flow_in[v]) == 0)
                elif v == DESTINATION:
                    # Demand constraint in destination
                    MODEL_NETWORK.addConstr(quicksum(flow_in[v]) == DEMAND)
                    MODEL_NETWORK.addConstr(quicksum(flow_out[v]) == 0)
                else:
                    # Flow conservation in transit
                    MODEL_NETWORK.addConstr(quicksum(flow_out[v]) == quicksum(flow_in[v]))
    
    def _update_controller_objective(self):
        M = len(self._graph.nodes)
        OUT_DEGREES = self._out_degrees
        UTILITY = self._utility
        X_OE = self._X_oe
        Z_OE = self._Z_oe
        RE = self._re
        RHO = self._solver_params.Rho
        MODEL_CONTROLLER = self._model_controller

        """
        Controller objective is:
            u + rho/2 sum_e (X_oe - Z_oe + r_e)^2
        """

        OBJECTIVE_CONTROLLER = \
            UTILITY + \
            RHO/2 * quicksum([
                quicksum([
                    (X_OE[v][i] - Z_OE[v][i] + RE[v][i]) ** 2
                        for i in range(OUT_DEGREES[v])
                ]) for v in range(M)
            ])
        
        MODEL_CONTROLLER.setObjective(OBJECTIVE_CONTROLLER, GRB.MINIMIZE)
        self._objective_controller = OBJECTIVE_CONTROLLER
    
    def _update_network_objective(self):
        M = len(self._graph.nodes)
        OUT_DEGREES = self._out_degrees
        Z_OE = self._Z_oe
        X_KE = self._X_ke
        RE = self._re
        RHO = self._solver_params.Rho
        MODEL_NETWORK = self._model_network

        """
        The node objective is:
            rho/2 * \sum_e (\sum_k X_ke - Z_oe - r_e)^2
        """

        OBJECTIVE_NETWORK = \
            RHO/2 * quicksum([
                quicksum([
                    (X_KE[v].sum(i, '*') - Z_OE[v][i] - RE[v][i]) ** 2
                        for i in range(OUT_DEGREES[v])
                ]) for v in range(M)
            ])

        MODEL_NETWORK.setObjective(OBJECTIVE_NETWORK, GRB.MINIMIZE)
        self._objective_network = OBJECTIVE_NETWORK
    
    def _add_objective(self):
        assert self._model_controller is not None and \
               self._model_network is not None

        self._update_controller_objective()
        self._update_network_objective()
    
    def _update_re(self):
        assert self._model_controller is not None and \
               self._model_network is not None

        """
        The update rule for r_e is:
            r_e \gets r_e + (X_oe - \sum_k X_ke)/2
        """

        M = len(self._graph.nodes)
        OUT_DEGREES = self._out_degrees
        X_OE = self._X_oe
        X_KE = self._X_ke
        RE = self._re

        for v in range(M):
            for i in range(OUT_DEGREES[v]):
                RE[v][i] += (
                    X_OE[v][i].X - X_KE[v].sum(i, '*').getValue()
                ) / 2
    
    def _update_Z_oe(self):
        assert self._model_controller is not None and \
               self._model_network is not None

        """
        The update rule for lambda_e is:
            Z_oe \gets (X_oe + \sum_k X_ke)/2
        """

        M = len(self._graph.nodes)
        OUT_DEGREES = self._out_degrees
        X_KE = self._X_ke
        X_OE = self._X_oe
        Z_OE = self._Z_oe

        for v in range(M):
            for i in range(OUT_DEGREES[v]):
                Z_OE[v][i] = (
                    X_OE[v][i].X + X_KE[v].sum(i, "*").getValue()
                ) / 2
    
    def close(self):
        self._model_controller.close()
        self._model_network.close()
        if self._env:
            self._env.close()
    
    def make_lp(self):
        self._make_variables()
        self._add_constraints()
        self._add_objective()
    
    def reset(self, with_params: False):
        self._model_controller.reset()
        self._model_network.reset()
        if with_params:
            self._model_controller.resetParams()
            self._model_network.resetParams()
    
    def solve(self, params: SolverParams = None) -> float:
        assert params is None
        
        CONTROLLER_MODEL = self._model_controller
        NETWORK_MODEL = self._model_network

        total_runtime = 0
        try:
            for _ in tqdm.tqdm(range(self._solver_params.NumberOfEpochs)):
            # for _ in range(self._solver_params.NumberOfEpochs):
                # First, concurrently solve both controller and network models
                CONTROLLER_MODEL.optimize()
                NETWORK_MODEL.optimize()

                # Second, update Z_oe and r_e
                self._update_Z_oe()
                self._update_re()

                # Update the objective and start again
                self._update_controller_objective()
                self._update_network_objective()

                # Houskeeping
                self._objective_trace.append(self._utility.X)
                total_runtime += max(CONTROLLER_MODEL.Runtime, NETWORK_MODEL.Runtime)

            return total_runtime
        except GurobiError as e:
            print(f'Error code {e.errno}: {e}')
            return -1

    def get_solution_commodity_list(self) -> List[Commodity]:
        COMMODITIES = self._commodity_list
        X_KE = self._X_ke
        X_OE = self._X_oe
        Z_OE = self._Z_oe
        OUT_DEGREE = self._out_degrees

        for v in range(len(X_OE)):
            for i in range(OUT_DEGREE[v]):
                print(f"X = {X_OE[v][i].X} vs. Z = {Z_OE[v][i]}")

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
