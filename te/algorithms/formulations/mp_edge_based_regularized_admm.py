import time
import tqdm
import gurobipy
import numpy as np
import networkx as nx
import te.constants
import multiprocessing
import protos.regularized_admm.regularized_admm_pb2 as lp_messages
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from gurobipy import GRB, GurobiError
from te.algorithms.base import TrafficEngineeringLP, GurobiSolverParams, SolverParams
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, Commodity
from topologies.utils import (get_edge_indexing, get_graph_M_matrix, 
                              get_adjacency_null_space, get_feasible_flow_assignment)
from te.algorithms.utils import (check_centralized_flow_conservation,
                                 optimize_or_scream)


@dataclass
class MultiProcessesorRegularizedADMMSolverParams(GurobiSolverParams):
    NumberOfNodeProcesses: int = 1
    NumberOfEpochs: int = 1000
    NumberOfNetworkUpdates: int = te.constants.DEFAULT_NUMBER_OF_NETWORK_UPDATES
    Rho: float = te.constants.DEFAULT_RHO
    Eta: float = te.constants.DEFAULT_ETA
    Epsilon: float = te.constants.DEFAULT_EPSILON_KE
    Seed: int = te.constants.DEFAULT_SEED


class ControllerModel(TrafficEngineeringLP):
    def __init__(self, graph: nx.DiGraph, solver_params: MultiProcessesorRegularizedADMMSolverParams):
        super().__init__()
        self._graph = graph
        self._solver_params = solver_params
        self._rng = np.random.default_rng(seed=solver_params.Seed)

        self._NUM_EDGES: Optional[int] = None
        self._env: gurobipy.Env = None

        self._model_controller: Optional[gurobipy.Model] = None
        self._objective_controller: Optional[gurobipy.QuadExpr] = None

        # Both just vectors of length `n`
        self._Xo_e_start: Optional[np.ndarray] = None
        self._Xo_e: Optional[gurobipy.tupledict] = None
        self._Zo_e: Optional[np.ndarray] = None
        # Just a variable between 0 and 1
        self._utility: Optional[gurobipy.Var] = None
        # Residual of outer ADMM. A vector of length `n`.
        self._r_e: Optional[np.ndarray] = None

        self._initialize_variables_and_residuals()

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph
    @property
    def traffic(self) -> TrafficMatrixBase:
        raise NotImplementedError
    @property
    def params(self) -> SolverParams:
        return self._solver_params
    @property
    def commodity_list(self) -> List[Commodity]:
        raise NotImplementedError
    @property
    def objective_value(self) -> float:
        raise NotImplementedError
    @property
    def objective_trace(self) -> List[float]:
        raise NotImplementedError
    
    def _initialize_variables_and_residuals(self):
        self._NUM_EDGES = len(self._graph.edges())

        NUM_EDGES = self._NUM_EDGES

        self._r_e = np.zeros(shape=(NUM_EDGES,))
        self._Zo_e = np.copy(self._Xo_e_start)

    def _make_variables(self):
        assert self._model_controller is None
        
        NUM_EDGES = self._NUM_EDGES

        ENV = gurobipy.Env()
        ENV.setParam('OutputFlag', 0)
        ENV.start()
        self._env = ENV

        self._model_controller = gurobipy.Model('EdgeBasedDistributedTE_Controller', env=ENV)

        MODEL_CONTROLLER = self._model_controller
        
        self._Xo_e = MODEL_CONTROLLER.addVars(NUM_EDGES, lb=0.0, vtype=GRB.CONTINUOUS, name=f'XO_E')
        self._utility = MODEL_CONTROLLER.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='U')
        # Set starting values ...
        self._Xo_e.Start = self._Xo_e_start

    def _get_F(self) -> np.ndarray:
        return self._Zo_e + self._r_e - self._Xo_e_start
    
    def _add_constraints(self):
        assert self._model_controller is not None

        GRAPH = self._graph
        XO_E = self._Xo_e
        UTILITY = self._utility
        MODEL_CONTROLLER = self._model_controller

        # Capacity constraint
        MODEL_CONTROLLER.addConstrs(
            XO_E[i] / c_e <= UTILITY
                for i, (_, _, c_e) in enumerate(GRAPH.edges(data='capacity'))
        )

    def _update_controller_objective(self):
        NUM_EDGES = self._NUM_EDGES
        UTILITY = self._utility
        XO_E = self._Xo_e
        ZO_E = self._Zo_e
        R_E = self._r_e
        RHO = self._solver_params.Rho
        MODEL_CONTROLLER = self._model_controller

        """
        Controller objective is:
            u + rho/2 sum_e (X_oe - Z_oe + r_e)^2
        """

        OBJECTIVE_CONTROLLER = gurobipy.QuadExpr()
        OBJECTIVE_CONTROLLER.addTerms(1, UTILITY)
        for e in range(NUM_EDGES):
            x = XO_E[e]
            c = R_E[e] - ZO_E[e]
            OBJECTIVE_CONTROLLER.addTerms(RHO/2, x, x)
            OBJECTIVE_CONTROLLER.addTerms(RHO * c, x)
        
        MODEL_CONTROLLER.setObjective(OBJECTIVE_CONTROLLER, GRB.MINIMIZE)
        self._objective_controller = OBJECTIVE_CONTROLLER

    def _add_objective(self):
        assert self._model_controller is not None

        self._update_controller_objective()

    def _update_Zo_e(self, X_KE_SUM_E):
        assert self._model_controller is not None

        """
        The update rule for Zo_e is:
            Zo_e \gets (X_oe + \sum_k X_ke)/2
        """

        XO_E = self._Xo_e
        NUM_EDGES = self._NUM_EDGES
        Zo_e = np.zeros((NUM_EDGES,))
        for e in range(NUM_EDGES):
            Zo_e[e] = (XO_E[e].X + X_KE_SUM_E[e]) / 2
        self._Zo_e = Zo_e
    
    def _update_r_e(self, X_KE_SUM_E):
        assert self._model_controller is not None

        """
        The update rule for r_e is:
            r_e \gets r_e + (X_oe - \sum_k X_ke)/2
        """

        R_E = self._r_e
        XO_E = self._Xo_e
        NUM_EDGES = self._NUM_EDGES

        r_e = np.zeros((NUM_EDGES,))
        for e in range(NUM_EDGES):
            r_e[e] = R_E[e] + (XO_E[e].X - X_KE_SUM_E[e]) / 2
        self._r_e = r_e
    
    def close(self):
        if self._model_controller:
            self._model_controller.close()
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

    def solve(self, params: SolverParams = None) -> float:
        assert params is None
        
        MODEL_CONTROLLER = self._model_controller
        optimize_or_scream(MODEL_CONTROLLER)
        return MODEL_CONTROLLER.Runtime

    def check(self):
        NUM_EDGES = self._NUM_EDGES
        PARAMS = self._solver_params

        # Are outer ADMM pairs in consensus?
        XO_E = self._Xo_e
        ZO_E = self._Zo_e
        for e in range(NUM_EDGES):
            primal = XO_E[e].X
            pair = ZO_E[e]
            primal_str = str(np.round(primal, 4))
            pair_str = str(np.round(pair, 4))
            assert abs(primal - pair) < 2*PARAMS.FeasibilityTol, \
                f"Edge {e} --> Outer ADMM pairing is not in consensus with primal variable: {primal_str} vs {pair_str}"


class NodeModel(TrafficEngineeringLP):
    def __init__(self, K: int, commodity_mapping: Dict[int, int], commodities: List[Commodity], 
                 X_ek_start: np.ndarray, NULL_M: np.ndarray, solver_params: MultiProcessesorRegularizedADMMSolverParams):
        super().__init__()
        self._commodity_mapping = commodity_mapping
        self._commodity_list = commodities
        self._NULL_M: np.ndarray = NULL_M
        self._solver_params = solver_params

        self._K = K
        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None

        self._env: gurobipy.Env = None

        self._model_nodes: Optional[List[gurobipy.Model]] = None
        self._objective_nodes: Optional[List[gurobipy.QuadExpr]] = None

        # This need not be managed by Gurobi
        self._X_ek_start: np.ndarray = X_ek_start
        # Slice of global value. A `T x K` matrix that we treat as a list of `K` vectors of length `T`
        self._Y_tk: Optional[List[gurobipy.tupledict]] = None

        self._initialize_variables_and_residuals()

    @property
    def graph(self) -> nx.DiGraph:
        raise NotImplementedError
    @property
    def traffic(self) -> TrafficMatrixBase:
        raise NotImplementedError
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
        raise NotImplementedError

    def _initialize_variables_and_residuals(self):
        NUM_EDGES, T = np.shape(self._NULL_M)
        self._T = T
        self._NUM_EDGES = NUM_EDGES
        
        self._u_t_scattered = np.zeros(shape=(T,))
        self._P_bar_t_scattered = np.zeros((T,))
    
    def _update_u_t(self, new_u_t: np.ndarray):
        self._u_t_scattered = new_u_t
    
    def _update_P_bar_t_scattered(self, new_P_bar_t: np.ndarray):
        self._P_bar_t_scattered = new_P_bar_t
    
    def _make_variables(self):
        assert self._model_nodes is None
        
        T = self._T
        K_SLICE = len(self._commodity_list)

        ENV = gurobipy.Env()
        ENV.setParam('OutputFlag', 0)
        ENV.start()
        self._env = ENV

        self._model_nodes = [
            gurobipy.Model(f'EdgeBasedDistributedTE_Commodity_{k}', env=ENV) 
                for k in range(K_SLICE)
        ]
    
        MODEL_NODES = self._model_nodes

        self._Y_tk = [
            model.addVars(T, lb=-float('inf'), vtype=GRB.CONTINUOUS, name=f'Y_{k}') 
                for k, model in enumerate(MODEL_NODES)
        ]
    
    def _get_X_ek_local(self, e: int, k: int) -> gurobipy.LinExpr:
        assert k >= 0 and k < len(self._commodity_list)
        T = self._T
        NULL_M = self._NULL_M
        Y_K_t = self._Y_tk[k]
        exp = gurobipy.LinExpr()
        exp.addConstant(self._X_ek_start[e, k])
        for t in range(T):
            exp.addTerms(NULL_M[e, t], Y_K_t[t])
        return exp

    def _get_X_ek_remote(self, e: int, k_remote: int) -> gurobipy.LinExpr:
        k = self._commodity_mapping.get(k_remote)
        assert k is not None
        return self._get_X_ek_local(e, k)

    def _get_X_k_sum(self) -> np.ndarray:
        K_MAP = self._commodity_mapping
        NUM_EDGES = self._NUM_EDGES
        return np.array([
            np.sum([
                self._get_X_ek_local(e, K_MAP[k]).getValue() if k in K_MAP else 0
                    for k in range(self._K)
            ]) for e in range(NUM_EDGES)
        ])

    def _get_Y_k_old_local(self, k: int) -> np.ndarray:
        assert k >= 0 and k < len(self._commodity_list)
        T = self._T
        try:
            return np.array([self._Y_tk[k][t].X for t in range(T)])
        except AttributeError:
            return np.zeros((T,))
    
    def _get_Y_sum_local(self) -> np.ndarray:
        K_SLICE = len(self._commodity_list)
        return np.sum([self._get_Y_k_old_local(k) for k in range(K_SLICE)], axis=0)

    def _add_constraints(self):
        assert self._model_nodes is not None

        NUM_EDGES = self._NUM_EDGES
        MODEL_NODES = self._model_nodes
        
        # Non-negativity constraint for node models
        for k, node_model in enumerate(MODEL_NODES):
            for e in range(NUM_EDGES):
                node_model.addConstr(0 <= self._get_X_ek_local(e, k))

    def _update_node_objective(self, Y_BAR_T_scattered, P_BAR_T_scattered, U_T_scattered):
        T = self._T
        K_SLICE = len(self._commodity_list)
        NUM_EDGES = self._NUM_EDGES
        EPSILON = self._solver_params.Epsilon
        ETA = self._solver_params.Eta

        Y_TK = self._Y_tk
        NULL_M = self._NULL_M
        X_EK_0 = self._X_ek_start
        Y_TK_old = [self._get_Y_k_old_local(k) for k in range(K_SLICE)]
        MODEL_NODES = self._model_nodes

        """
        The node objective for commodity `k` is:

            (\epsilon/2) || X_k^0 + NULL_M @ Y_k ||_2^2 + 
            (\eta/2) || Y_k - Y_k^(old) + Y_bar - P_bar + u ||_2^2
        
        We will benefit greatly from openning this expression and writing it out as
        incremental terms rather than `quicksum`, as it makes it much faster.

        To this end, the first expression (the regularizer) can be expanded to (ignoring constants!):

            (\epsilon/2) (\sum_t (Y_kt^2) + 2 \sum_e X_ke^(0) \sum_t NULL_M_et Y_tk))
        
        (Note that columns of `NULL_M` were orthonormal).
        The section expression is just:

            (\eta/2) (\sum_t (Y_kt^2) + 2*\sum_t (Y_kt)(u_t - P_bar_t + Y_bar_t - Y_k^(old)_t))
        """

        OBJECTIVE_NODES = [gurobipy.QuadExpr() for _ in range(K_SLICE)]
        for k, obj in enumerate(OBJECTIVE_NODES):
            for t in range(T):
                y = Y_TK[k][t]
                c = U_T_scattered[t] - P_BAR_T_scattered[t] + Y_BAR_T_scattered[t] - Y_TK_old[k][t]
                obj.addTerms((EPSILON + ETA)/2, y, y)
                obj.addTerms(ETA * c, y)
                for e in range(NUM_EDGES):
                    x = X_EK_0[e, k]
                    n = NULL_M[e, t]
                    obj.addTerms(ETA * x * n, y)
                    
        assert len(OBJECTIVE_NODES) == len(MODEL_NODES)
        for node_obj, node_model in zip(OBJECTIVE_NODES, MODEL_NODES):
            node_model.setObjective(node_obj, GRB.MINIMIZE)
        self._objectives_nodes = OBJECTIVE_NODES

    def _add_objective(self):
        assert self._model_nodes is not None
        self._update_node_objective()

    def close(self):
        if self._model_nodes:
            for node_model in self._model_nodes:
                node_model.close()
        if self._env:
            self._env.close()
    
    def make_lp(self):
        self._make_variables()
        self._add_constraints()
        self._add_objective()

    def reset(self, with_params: False):
        for node_model in self._model_nodes:
            node_model.reset()
        if with_params:
            for node_model in self._model_nodes:
                node_model.resetParams()

    def _make_X_ek(self):
        K_SLICE = len(self._commodity_list)
        NUM_EDGE = self._NUM_EDGES
        X_EK = np.zeros((NUM_EDGE, K_SLICE))
        for e in range(NUM_EDGE):
            for k in range(K_SLICE):
                X_EK[e, k] = self._get_X_ek_local(e, k).getValue()
        return X_EK

    def solve(self, params: SolverParams = None) -> float:
        assert params is None
        
        MODEL_NODES = self._model_nodes
        runtimes = []
        for k, node_model in enumerate(MODEL_NODES):
            optimize_or_scream(node_model)
            runtimes.append(node_model.Runtime)
        return max(runtimes)
    
    def check(self, Y_BAR_T_scattered, P_BAR_T_scattered):
        T = self._T
        PARAMS = self._solver_params
        
        # Are inner ADMM pairs in consensus?
        for t in range(T):
            primal = Y_BAR_T_scattered[t]
            pair = P_BAR_T_scattered[t]
            primal_str = str(np.round(primal, 4))
            pair_str = str(np.round(pair, 4))
            assert abs(primal - pair) < 2*PARAMS.FeasibilityTol, \
                f"Axis {t} --> Inner ADMM pairing is not in consensus with primal variable: {primal_str} vs {pair_str}"


class MultiProcessorRegularizedADMMLP(TrafficEngineeringLP):
    def __init__(self, graph: nx.DiGraph, traffic: TrafficMatrixBase, solver_params: MultiProcessesorRegularizedADMMSolverParams) -> None:
        super().__init__()
        self._graph = graph
        self._M = get_graph_M_matrix(graph)
        self._traffic = traffic
        self._solver_params = solver_params
        self._rng = np.random.default_rng(seed=solver_params.Seed)
        self._commodity_list = traffic_to_commodity(self._traffic)

        self._edge_indexing = get_edge_indexing(graph)
        self._NULL_M: np.ndarray = None
        
        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None
        self._BASE_NUM_NODE_LPS: Optional[int] = None
        self._REM_NUM_NODE_LPS: Optional[int] = None
        self._commodity_mapping: Optional[Dict[int, Dict[int, int]]] = None
        self._commodity_slices: Optional[Dict[int, List[Commodity]]] = None

        # This need not be managed by Gurobi
        self._X_ek_start: Optional[np.ndarray] = None
        self._X_ek: Optional[np.ndarray] = None
        # Running sum, collected from nodes
        self._X_k_sum_e: Optional[np.ndarray] = None
        # ADMM running average variable. A vector of length `T`.
        self._Y_bar_t: Optional[np.ndarray] = None
        self._P_bar_t: Optional[np.ndarray] = None
        # Residual of inner ADMM. A vector of length `T`.
        self._u_t: Optional[np.ndarray] = None

        self._controller_lp: Optional[ControllerModel] = None
        self._node_lps: Optional[List[NodeModel]] = None

        self._X_ek: Optional[np.ndarray] = None

        self._objective_trace = []

        self._set_initial_feasible_solution()
        self._set_NULL_M()
        self._initialize_variables_and_residuals()
        self._partition_commodities()
        self._report_problem_size()

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

    def _set_initial_feasible_solution(self):
        self._X_ek_start = get_feasible_flow_assignment(self._graph, self._commodity_list)
        self._Xo_e_start = np.sum(self._X_ek_start, axis=1)
        # Just a sanity check ...
        check_centralized_flow_conservation(self._X_ek_start, self._graph, self._commodity_list, self._solver_params.FeasibilityTol)
    
    def _set_NULL_M(self):
        M = self._M
        assert len(M.shape) == 2
        m, n = M.shape
        assert m < n
        N = get_adjacency_null_space(M)
        T = N.shape[1]
        # TODO: This is off by 1, since the columns of `M` are not independent
        # assert T == (n - m), f'{n}, {m}, {T}'
        assert np.allclose(np.matmul(N.T, N) - np.eye(T), 0)
        self._NULL_M = N
        self._T = T
        self._NUM_EDGES = n
    
    def _initialize_variables_and_residuals(self):
        T = self._T

        self._u_t = np.zeros(shape=(T,))
        self._P_bar_t = np.zeros((T,))
    
    def _partition_commodities(self):
        assert self._BASE_NUM_NODE_LPS and self._REM_NUM_NODE_LPS is None
        K = self._commodity_list
        NUM_PROCS = self._solver_params.NumberOfNodeProcesses
        BASE_NUM_NODE_LPS = K // NUM_PROCS
        REM_NUM_NODE_LPS = K % NUM_PROCS
        
        # For now, just a simple rolling assignment
        commodity_mapping = dict()
        commodity_slices = dict()
        commodity_counter = 0
        for node_index in range(self._solver_params.NumberOfNodeProcesses):
            n = BASE_NUM_NODE_LPS+1 if node_index+1 <= REM_NUM_NODE_LPS else BASE_NUM_NODE_LPS
            commodity_mapping[node_index] = {commodity_counter + i: i for i in range(n)} 
            commodity_counter += n
            commodity_slices[node_index] = list(range(n))

        self._BASE_NUM_NODE_LPS = BASE_NUM_NODE_LPS
        self._REM_NUM_NODE_LPS = REM_NUM_NODE_LPS
        self._commodity_mapping = commodity_mapping
        self._commodity_slices = commodity_slices
    
    def _make_models(self):
        assert self._controller_lp is None and self._node_lps is None
        K = len(self._commodity_list)
        K_MAP = self._commodity_mapping
        K_SLICES = self._commodity_slices
        NUM_PROCS = self._solver_params.NumberOfNodeProcesses
        NULL_M = self._NULL_M
        self._controller_lp = ControllerModel(self._graph, self._solver_params)
        self._node_lps = [
            NodeModel(K, K_MAP[node_index], K_SLICES[node_index], self._X_ek_start[:, K_SLICES[node_index]], NULL_M, self._solver_params)
            for node_index in range(NUM_PROCS)
        ]
    
    def _report_problem_size(self):
        M = len(self._graph.nodes)
        N = len(self._graph.edges)
        T = self._T
        K = len(self._commodity_list)

        print(f"Graph Size: {M} nodes | {N} edges")
        print(f"Number of commodities: {K}")
        print(f"Nullity of commodity assignment matrix: {T}")
        print("-"*60)
        print("CONTROLLER PROBLEM:\n" +
              f"\t TOTAL NUMBER OF VARIABLES: {N + 1}\n"
              f"\t TOTAL NUMBER OF CONSTRAINTS: {N + 1}\n")
        print("-"*60)
        print("NODE PROBLEM:\n" +
              f"\t NUMBER OF INDEPENDENT QPs PER NODE: {M - 1}\n"
              f"\t NUMBER OF VARIABLES PER QP PER NODE: {T}\n"
              f"\t NUMBER CONSTRAINTS PER QP PER NODE: {T}\n")
        
    def _make_variables(self):
        assert self._controller_lp is not None and \
               self._node_lps is not None
        
        self._controller_lp._make_variables()
        for node_lp in self._node_lps:
            node_lp._make_variables()
    
    def _get_node_index_for_commodity(self, k: int) -> int:
        BASE_NUM_NODE_LPS = self._BASE_NUM_NODE_LPS
        REM_NUM_NODE_LPS = self._REM_NUM_NODE_LPS
        cutoff = REM_NUM_NODE_LPS * (BASE_NUM_NODE_LPS+1)
        if k < cutoff:
            return k // (BASE_NUM_NODE_LPS+1)
        return REM_NUM_NODE_LPS + (k - cutoff) // BASE_NUM_NODE_LPS

    def _add_constraints(self):
        assert self._controller_lp is not None and \
               self._node_lps is not None

        self._controller_lp._add_constraints()
        for node_lp in self._node_lps:
            node_lp._add_constraints()
    
    def _update_controller_objective(self):
        assert self._controller_lp is not None
        self._controller_lp._update_controller_objective()
    
    def _update_node_objectives(self):
        assert self._node_lps is not None
        Y_BAR_T = self._Y_bar_t
        P_BAR_T = self._P_bar_t
        U_T = self._u_t
        for node_lp in self._node_lps:
            node_lp._update_node_objective(Y_BAR_T, P_BAR_T, U_T)
    
    def _add_objective(self):
        assert self._controller_lp is not None and \
               self._node_lps is not None

        self._update_controller_objective()
        self._update_node_objectives()
    
    def _gather_Y_bar(self):
        assert self._node_lps is not None
        K = len(self._commodity_list)

        self._Y_bar_t = np.sum([
            node_lp._get_Y_sum_local()
            for node_lp in self._node_lps
        ], axis=0) / K
    
    def _gather_X_k_sum(self):
        assert self._node_lps is not None
        self._X_k_sum_e = np.sum([
            node_lp._get_X_k_sum() for node_lp in self._node_lps
        ], axis=0)
    
    def _update_P_bar(self):
        assert self._controller_lp is not None and \
               self._node_lps is not None and \
               self._Y_bar_t is not None
        
        """
        The update rule for `P_bar` is:

            P_bar \gets (NULL_M^T F + (\eta/\rho) (u + Y_bar)) / (K + (\eta/\rho))
        """

        K = len(self._commodity_list)
        ETA = self._solver_params.Eta
        RHO = self._solver_params.Rho
        U_T = self._u_t
        Y_BAR_T = self._Y_bar_t
        F_E = self._controller_lp._get_F()
        NULL_M = self._NULL_M

        self._P_bar_t = (NULL_M.T @ F_E + (ETA/RHO) * (U_T + Y_BAR_T)) / (K + (ETA/RHO))
    
    def _update_u_t(self):
        assert self._controller_lp is not None and \
               self._node_lps is not None and \
               self._Y_bar_t is not None
        
        """
        The update rule for `u` is:

            u \gets (u + Y_bar - P_bar)
        """

        U_T = self._u_t
        Y_BAR_T = self._Y_bar_t
        P_BAR_T = self._P_bar_t

        self._u_t = (U_T + Y_BAR_T - P_BAR_T)

    def _update_Zo_e(self):
        assert self._controller_lp is not None and \
               self._node_lps is not None

        """
        The update rule for Zo_e is:
            Zo_e \gets (X_oe + \sum_k X_ke)/2
        """

        XO_E = self._controller_lp._Xo_e
        NUM_EDGES = self._NUM_EDGES
        X_KE_SUM_E = self._X_k_sum_e
        Zo_e = np.zeros((NUM_EDGES,))
        for e in range(NUM_EDGES):
            Zo_e[e] = (XO_E[e].X + X_KE_SUM_E[e]) / 2
        self._Zo_e = Zo_e
    
    def _update_r_e(self):
        assert self._controller_lp is not None and \
               self._node_lps is not None

        """
        The update rule for r_e is:
            r_e \gets r_e + (X_oe - \sum_k X_ke)/2
        """

        R_E = self._r_e
        XO_E = self._controller_lp._Xo_e
        NUM_EDGES = self._NUM_EDGES
        X_KE_SUM_E = self._X_k_sum_e
        r_e = np.zeros((NUM_EDGES,))
        for e in range(NUM_EDGES):
            r_e[e] = R_E[e] + (XO_E[e].X - X_KE_SUM_E[e]) / 2
        self._r_e = r_e
    
    def close(self):
        if self._controller_lp:
            self._controller_lp.close()
        if self._node_lps:
            for node_model in self._node_lps:
                node_model.close()
    
    def make_lp(self):
        t_start = time.time()
        print("Starting to create the model")
        self._make_variables()
        self._add_constraints()
        self._add_objective()
        print(f"Built model in {str(np.round(time.time() - t_start, 2))} seconds.")
    
    def reset(self, with_params: False):
        self._controller_lp.reset(with_params=with_params)
        for node_model in self._node_lps:
            node_model.reset(with_params=with_params)

    def _gather_X_ek(self):
        self._X_ek = np.hstack([
            node_lp._make_X_ek()
            for node_lp in self._node_lps
        ])
    
    def solve(self, params: SolverParams = None) -> float:
        assert params is None
        
        MODEL_CONTROLLER = self._controller_lp
        MODEL_NODES = self._node_lps
        NUM_PROCS = len(MODEL_NODES)
        PARAMS = self._solver_params

        total_runtime = 0

        try:
            for _ in tqdm.tqdm(range(PARAMS.NumberOfEpochs)):
                t_nodes: Dict[int, List[float]] = defaultdict(list)

                # First, let the controller decide what the utilization is
                t_controller = MODEL_CONTROLLER.solve()
                # Now, do in-network optimization
                for _ in range(PARAMS.NumberOfNetworkUpdates):
                    for node_index, node_model in enumerate(MODEL_NODES):
                        t_nodes[node_index].append(node_model.solve())
                    # Gather updates for inner ADMM step
                    self._gather_Y_bar()
                    # Finish inner ADMM step
                    self._update_P_bar()
                    self._update_u_t()
                    # Scatter updates to nodes
                    self._update_node_objectives()
                # Gather updates for outer ADMM step
                self._gather_X_k_sum()
                # Finish outer ADMM step
                self._update_Zo_e()
                self._update_r_e()
                # Update the objectives and start again
                self._update_controller_objective()
                self._update_node_objectives()

                # Houskeeping
                self._objective_trace.append(self._controller_lp._utility.X)
                total_runtime += t_controller + max(sum(t_nodes[node_index]) for node_index in range(NUM_PROCS))
            
            # Build flow assignments
            self._gather_X_ek()
            return total_runtime
        except GurobiError as e:
            print(f'Error code {e.errno}: {e}')
            return -1
    
    def check(self):
        PARAMS = self._solver_params
        Y_BAR_T = self._Y_bar_t
        P_BAR_T = self._P_bar_t

        # Check outer ADMM consensus
        self._controller_lp.check()
        # Check inner ADMM consensus
        for node_lp in self._node_lps:
            node_lp.check(Y_BAR_T, P_BAR_T)
        # Now, check flow conservation ...
        X_EK = self._X_ek
        check_centralized_flow_conservation(X_EK, self._graph, self._commodity_list, PARAMS.FeasibilityTol)

    def get_solution_commodity_list(self) -> List[Tuple[Commodity, Commodity]]:
        COMMODITIES = self._commodity_list
        X_EK = self._X_ek
        GRAPH = self._graph
        INDICES = self._edge_indexing

        return [
            (
                Commodity(
                    source=commodity.source,
                    destination=commodity.destination,
                    demand=sum([
                        X_EK[INDICES[(v, commodity.destination)], i] \
                            for v in GRAPH.predecessors(commodity.destination)
                    ])
                ),
                Commodity(
                    source=commodity.source,
                    destination=commodity.destination,
                    demand=sum([
                        X_EK[INDICES[(commodity.source, v)], i] \
                            for v in GRAPH.successors(commodity.source)
                    ])
                )
            )
            for i, commodity in enumerate(COMMODITIES)
        ]
