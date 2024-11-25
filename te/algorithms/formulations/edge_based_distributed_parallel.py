import tqdm
import gurobipy
import numpy as np
import networkx as nx
import te.constants
from typing import List, Dict, Tuple
from dataclasses import dataclass
from gurobipy import GRB, GurobiError, quicksum
from te.algorithms.base import TrafficEngineeringLP, SolverParams, GurobiSolverParams
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, Commodity
from topologies.utils import get_node_and_out_edge_index_mapping, get_in_edge_mapping

# Shared between the controller and node solvers
@dataclass
class DistributedParallelSolverParams(GurobiSolverParams):
    NumberOfEpochs: int = 1000
    Seed: int = te.constants.DEFAULT_SEED

# Specific to the centralized controller solver
@dataclass
class DistributedParallelSolverControllerParams(SolverParams):
    EpsilonOE: float = te.constants.DEFAULT_EPSILON_OE 
    Alpha: float = te.constants.DEFAULT_ALPHA

# Specific to node solvers
@dataclass
class DistributedParallelSolverNodeParams(SolverParams):
    EpsilonKE: float = te.constants.DEFAULT_EPSILON_KE
    Beta: float = te.constants.DEFAULT_BETA


class NodeLP(TrafficEngineeringLP):
    def __init__(self, node_index: int, commodity_list: List[Commodity], out_degree: int, 
                 solver_params: DistributedParallelSolverParams, 
                 node_params: DistributedParallelSolverNodeParams) -> None:
        super().__init__()
        self._node_index: int = node_index
        self._commodity_list = commodity_list
        self._out_degree = out_degree
        self._solver_params: DistributedParallelSolverParams = solver_params
        self._node_params: DistributedParallelSolverNodeParams = node_params

        self._rng = np.random.default_rng(seed=solver_params.Seed)
        self._env: gurobipy.Env = None
        self._node_model: gurobipy.Model = None
        self._node_objective: gurobipy.QuadExpr = None
        self._flows_ke: gurobipy.MVar = None
        self._mu_kv: np.ndarray = None

    @property
    def graph(self) -> nx.DiGraph:
        raise ValueError("Shouldn't be used ...")
    @property
    def traffic(self) -> TrafficMatrixBase:
        raise ValueError("Shouldn't be used ...")
    @property
    def params(self) -> SolverParams:
        return self._node_params
    @property
    def commodity_list(self) -> List[Commodity]:
        return self._commodity_list
    @property
    def objective_trace(self) -> List[float]:
        raise ValueError("Shouldn't be used ...")
    @property
    def objective_value(self) -> float:
        if self._node_objective:
            return self._node_objective.getValue()
        return None
    
    def _f_ke(self, x_ke: gurobipy.Var) -> gurobipy.QuadExpr:
        return self._node_params.EpsilonKE * x_ke ** 2

    def _initiate_mu(self):
        K = len(self._commodity_list)
        self._mu_kv = self._rng.random(size=(K,))

    def _make_variables(self):
        assert self._node_model is None
        
        K = len(self._commodity_list)
        V = self._node_index
        OUT_DEGREE = self._out_degree

        ENV = gurobipy.Env()
        ENV.setParam('OutputFlag', 0)
        ENV.start()
        self._env = ENV

        self._node_model = gurobipy.Model(f'EdgeBasedDistributedTE_Node_{V}', env=ENV)
        MODEL_NODE = self._node_model

        self._flows_ke = MODEL_NODE.addVars(OUT_DEGREE, K, lb=0.0, vtype=GRB.CONTINUOUS, name=f'X_KE_{V}')

    def _add_constraints(self):
        assert self._node_model is not None
        # The node problem is NOT constrained beyond `X_KE >= 0` !
        pass

    def _update_node_objective(self, lambda_e: np.ndarray, mu_prime: np.ndarray):
        assert self._node_model is not None
        
        K = len(self._commodity_list)
        OUT_DEGREE = self._out_degree
        NODE_MODEL = self._node_model
        X_KE = self._flows_ke
        LAMBDA_E = lambda_e
        MU_KV = self._mu_kv
        MU_PRIME = mu_prime

        assert np.shape(LAMBDA_E) == (OUT_DEGREE,)
        assert np.shape(MU_PRIME) == (OUT_DEGREE, K)

        OBJECTIVE_NODE = \
            quicksum([
                quicksum([
                    self._f_ke(X_KE[i, k]) + X_KE[i, k] * (
                        LAMBDA_E[i] + MU_KV[k] - MU_PRIME[i, k]
                    ) for i in range(OUT_DEGREE)
                ]) for k in range(K)
            ])
        self._node_objective = OBJECTIVE_NODE

        NODE_MODEL.setObjective(OBJECTIVE_NODE, GRB.MINIMIZE)

    def _add_objective(self, lambda_e: List[float], mu_prime: np.ndarray):
        self._update_node_objective(lambda_e, mu_prime)

    def _update_mu(self, flows_in: List[float]):
        assert self._node_model is not None

        V = self._node_index
        X_KE = self._flows_ke
        COMMODITIES = self._commodity_list
        K = len(COMMODITIES)
        PARAMS = self._node_params
        MU_KV = self._mu_kv

        assert (len(flows_in) == K)
        FLOWS_IN = flows_in

        for k, commodity in enumerate(COMMODITIES):
            flow_out_k = X_KE.sum('*', k).getValue()
            flow_in_k = FLOWS_IN[k]

            if V == commodity.source:
                MU_KV[k] += PARAMS.Beta * (flow_out_k - flow_in_k - commodity.demand)
            elif V == commodity.destination:
                MU_KV[k] += PARAMS.Beta * (flow_out_k - flow_in_k + commodity.demand)
            else:
                MU_KV[k] += PARAMS.Beta * (flow_out_k - flow_in_k)

    def close(self):
        if self._node_model:
            self._node_model.close()
        if self._env:
            self._env.close()

    def make_lp(self, lambda_e: List[float], mu_prime: List[float]):
        self._make_variables()
        self._add_constraints()
        self._add_objective(lambda_e, mu_prime)

    def reset(self, with_params: False):
        self._node_model.reset()
        if with_params:
            self._node_model.resetParams()

    def solve(self, params: SolverParams = None) -> float:
        assert params is None

        self._node_model.optimize()
        return self._node_model.Runtime
    
    @property
    def x_ke(self) -> np.ndarray:
        K = len(self._commodity_list)
        OUT_DEGREE = self._out_degree
        X_KE = self._flows_ke

        res = np.array([[X_KE[i, k].X for k in range(K)] for i in range(OUT_DEGREE)])
        assert res.shape == (OUT_DEGREE, K)
        return res
    
    @property
    def mu_kv(self) -> np.ndarray:
        return self._mu_kv.copy()


class ControllerLP(TrafficEngineeringLP):
    def __init__(self, graph: nx.DiGraph, commodity_list: List[Commodity],
                 solver_params: DistributedParallelSolverParams, 
                 controller_params: DistributedParallelSolverControllerParams) -> None:
        super().__init__()
        self._graph = graph
        self._commodity_list = commodity_list
        self._solver_params = solver_params
        self._controller_params = controller_params
        
        self._env: gurobipy.Env = None
        self._rng = np.random.default_rng(seed=solver_params.Seed)
        self._controller_model: gurobipy.Model = None
        self._controller_objective: gurobipy.QuadExpr = None
        self._flows_oe: List[gurobipy.MVar] = None
        self._utility = None
        self._lambda_ve: List[np.ndarray] = None

        self._out_degrees = {k: v for k, v in graph.out_degree()}

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph
    @property
    def traffic(self) -> TrafficMatrixBase:
        raise ValueError("Shouldn't be used ...")
    @property
    def params(self) -> SolverParams:
        return self._controller_params
    @property
    def commodity_list(self) -> List[Commodity]:
        return self._commodity_list
    @property
    def objective_trace(self) -> List[float]:
        raise ValueError("Shouldn't be used ...")
    @property
    def objective_value(self) -> float:
        if self._controller_objective:
            return self._controller_objective.getValue()
        return None
    
    def _f_oe(self, x_oe: gurobipy.Var) -> gurobipy.QuadExpr:
        return self._controller_params.EpsilonOE * x_oe ** 2
    
    def _initiate_lambda_ve(self):
        M = len(self._graph.nodes)
        self._lambda_ve = [
            self._rng.random(size=(self._out_degrees[i]))
                for i in range(M)
        ]

    def _make_variables(self):
        assert self._controller_model is None
        
        M = len(self._graph.nodes)
        OUT_DEGREES = self._out_degrees

        ENV = gurobipy.Env()
        ENV.setParam('OutputFlag', 0)
        ENV.start()
        self._env = ENV

        self._controller_model = gurobipy.Model('EdgeBasedDistributedTE_Controller', env=ENV)
        CONTROLLER_MODEL = self._controller_model

        self._flows_oe = [
            CONTROLLER_MODEL.addVars(OUT_DEGREES[v], lb=0.0, vtype=GRB.CONTINUOUS, name=f'X_OE_{v}') \
                for v in range(M)
        ]
        self._utility = CONTROLLER_MODEL.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='U')

    def _add_constraints(self):
        assert self._controller_model is not None

        M = len(self._graph.nodes)
        CONTROLLER_MODEL = self._controller_model
        GRAPH = self._graph
        X_OE = self._flows_oe
        UTILITY = self._utility

        # Capacity constraint
        for v in range(M):
            CONTROLLER_MODEL.addConstrs(
                # X_OE[v][i] <= c_e
                X_OE[v][i] / c_e <= UTILITY
                    for i, (_, _, c_e) in enumerate(GRAPH.out_edges(v, data='capacity'))
            )
    
    def _update_controller_objective(self):
        M = len(self._graph.nodes)
        OUT_DEGREES = self._out_degrees
        CONTROLLER_MODEL = self._controller_model
        UTILITY = self._utility
        X_OE = self._flows_oe
        LAMBDA_VE = self._lambda_ve

        CONTROLLER_OBJECTIVE = \
            UTILITY ** 2 + \
            quicksum([
                quicksum([
                    self._f_oe(X_OE[v][i]) - LAMBDA_VE[v][i] * X_OE[v][i] \
                        for i in range(OUT_DEGREES[v])
                ]) for v in range(M)
            ])
        
        self._controller_objective = CONTROLLER_OBJECTIVE
        CONTROLLER_MODEL.setObjective(CONTROLLER_OBJECTIVE, GRB.MINIMIZE)
    
    def _add_objective(self):
        assert self._controller_model is not None
        self._update_controller_objective()
    
    def _update_lambda(self, flows_ve: List[np.ndarray]):
        assert self._controller_model is not None

        M = len(self._graph.nodes)
        assert len(flows_ve) == M

        OUT_DEGREES = self._out_degrees
        X_OE = self._flows_oe
        LAMBDA_VE = self._lambda_ve
        CONTROLLER_PARAMS = self._controller_params

        for v in range(M):
            for i in range(OUT_DEGREES[v]):
                LAMBDA_VE[v][i] += CONTROLLER_PARAMS.Alpha * (
                    flows_ve[v][i] - X_OE[v][i].X
                )
                if LAMBDA_VE[v][i] < 0:
                    LAMBDA_VE[v][i] = 0
    
    def close(self):
        if self._controller_model:
            self._controller_model.close()
        if self._env:
            self._env.close()
    
    def make_lp(self):
        self._make_variables()
        self._add_constraints()
        self._add_objective()
    
    def reset(self, with_params: False):
        self._controller_model.reset()
        if with_params:
            self._controller_model.resetParams()

    def solve(self, params: SolverParams = None) -> float:
        assert params is None

        self._controller_model.optimize()
        return self._controller_model.Runtime
    
    @property
    def lambda_ve(self) -> List[np.ndarray]:
        return self._lambda_ve.copy()


class DistributedParallelEdgeBasedLP(TrafficEngineeringLP):
    def __init__(self, graph: nx.DiGraph, traffic: TrafficMatrixBase, 
                 solver_params: DistributedParallelSolverParams,
                 controller_params: DistributedParallelSolverControllerParams,
                 node_params: DistributedParallelSolverNodeParams) -> None:
        super().__init__()
        self._graph = graph
        self._traffic = traffic
        self._solver_params = solver_params
        self._controller_params = controller_params
        self._node_params = node_params
        self._commodity_list = traffic_to_commodity(self._traffic)

        self._out_degrees: Dict[int, int] = {k: v for k, v in graph.out_degree()}
        self._objective_trace: List[float] = []

        self._controller_lp = ControllerLP(
            graph, self._commodity_list, solver_params,
            controller_params
        )
        self._node_lps = [
            NodeLP(v, self._commodity_list, self._out_degrees[v],
                   solver_params, node_params) for v in range(len(graph.nodes))
        ]

        self._in_edge_mapping: Dict[int, List[Tuple[int, int]]] = get_in_edge_mapping(graph)
        self._out_edge_mapping: Dict[Tuple[int, int], Tuple[int, int]] = get_node_and_out_edge_index_mapping(graph)

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
        return self._controller_lp.objective_value + \
            sum([node_lp.objective_value for node_lp in self._node_lps])
    
    @property
    def objective_trace(self) -> List[float]:
        return self._objective_trace

    def _initiate_dual_weights(self):
        self._controller_lp._initiate_lambda_ve()
        for node_lp in self._node_lps:
            node_lp._initiate_mu()

    def _make_variables(self):
        raise ValueError("Shouldn't be called ...")

    def _add_constraints(self):
        raise ValueError("Shouldn't be called ...")

    def _add_objective(self):
        raise ValueError("Shouldn't be called ...")
    
    def close(self):
        self._controller_lp.close()
        for node_lp in self._node_lps:
            node_lp.close()
    
    def make_lp(self):
        # First, initiate `mu` and `lambda`
        self._initiate_dual_weights()
        # Make controller LP
        self._controller_lp.make_lp()
        # Get current values of `lambda` and `mu`
        lambda_ve = self._controller_lp.lambda_ve
        mu_kv = [node_lp.mu_kv for node_lp in self._node_lps]
        # Make node LPs
        for v, node_lp in enumerate(self._node_lps):
            node_lp.make_lp(lambda_e=lambda_ve[v], mu_prime=self.get_mu_prime(mu_kv, v))
        
        # Just some sanity checks
        assert self._controller_lp._controller_objective is not None
        for node_lp in self._node_lps:
            assert node_lp._node_objective is not None
    
    def reset(self, with_params: False):
        self._controller_lp.reset(with_params=with_params)
        for node_lp in self._node_lps:
            node_lp.reset(with_params=with_params)
    
    def solve(self, params: SolverParams = None) -> float:
        assert params is None
        
        NODE_LPS = self._node_lps
        CONTROLLER_LP = self._controller_lp

        total_runtime = 0
        t_nodes = []
        try:
            for _ in tqdm.tqdm(range(self._solver_params.NumberOfEpochs)):
                # Clear up collected runtime samples ...
                t_nodes.clear()

                # First, solve node LPs
                for node_lp in NODE_LPS:
                    t_nodes.append(node_lp.solve())
                
                # Get updated `x_ke` ...
                X_KE = self.get_x_ke()

                # Now, we need to update instances of `mu`
                for V, node_lp in enumerate(NODE_LPS):
                    flows_in = self.get_flows_in(X_KE, V)
                    node_lp._update_mu(flows_in)
                
                # Now, solve the controller LP
                t_controller = CONTROLLER_LP.solve()

                # Get `flows_ve` and update `lambda`
                flows_ve = self.get_flows_ve(X_KE)
                CONTROLLER_LP._update_lambda(flows_ve)

                # Now, we need to update the objectives. Controller is easy ...
                CONTROLLER_LP._update_controller_objective()

                # For nodes, we need `lambda` and `mu_prime`
                LAMBDA_VE = self.get_lambda_ve()
                MU_KV = self.get_mu_kv()
                for V, node_lp in enumerate(NODE_LPS):
                    MU_PRIME = self.get_mu_prime(MU_KV, V)
                    node_lp._update_node_objective(LAMBDA_VE[V], MU_PRIME)

                total_runtime += (max(t_nodes) + t_controller)
                self._objective_trace.append(CONTROLLER_LP._utility.X)
            return total_runtime
        except GurobiError as e:
            print(f'Error code {e.errno}: {e}')
            return -1
    
    def get_flows_in(self, x_ke: List[np.ndarray], v: int) -> np.ndarray:
        X_KE = x_ke
        K = len(self.commodity_list)
        IN_EDGE_MAPPING = self._in_edge_mapping
        res = np.sum([X_KE[v_prime][i, :] for v_prime, i in IN_EDGE_MAPPING[v]], axis=0)
        assert np.shape(res) == (K,)

        return res
    
    def get_flows_ve(self, x_ke: List[np.ndarray]) -> List[np.ndarray]:
        X_KE = x_ke
        M = len(self._graph.nodes)

        return [np.sum(X_KE[V], axis=0) for V in range(M)]


    def get_mu_prime(self, mu_kv: List[np.ndarray], v: int) -> np.ndarray:
        MU_KV = mu_kv
        K = len(self.commodity_list)
        OUT_DEGREE = self._out_degrees[v]
        OUT_EDGE_MAPPING = self._out_edge_mapping

        res = np.array([MU_KV[OUT_EDGE_MAPPING[(v, i)][1]] for i in range(OUT_DEGREE)])
        assert np.shape(res) == (OUT_DEGREE, K)

        return res
    
    def get_x_ke(self) -> List[np.ndarray]:
        return [node_lp.x_ke for node_lp in self._node_lps]
    
    def get_lambda_ve(self) -> List[np.ndarray]:
        return self._controller_lp.lambda_ve
    
    def get_mu_kv(self) -> List[np.ndarray]:
        return [node_lp.mu_kv for node_lp in self._node_lps]

    def get_solution_commodity_list(self) -> List[Commodity]:
        COMMODITIES = self._commodity_list
        X_KE = [node_lp._flows_ke for node_lp in self._node_lps]
        X_OE = self._controller_lp._flows_oe
        OUT_DEGREE = self._out_degrees

        for v in range(len(X_OE)):
            for i in range(OUT_DEGREE[v]):
                print(X_OE[v][i].X)

        return [
            Commodity(
                source=commodity.source,
                destination=commodity.destination,
                demand=X_KE[commodity.source].sum('*', i).getValue()
            )
            for i, commodity in enumerate(COMMODITIES)
        ]
