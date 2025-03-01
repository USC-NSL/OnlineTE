import grpc
import time
import tqdm
import signal
import gurobipy
import numpy as np
import networkx as nx
import te.constants
import multiprocessing
import protos.regularized_admm.regularized_admm_pb2 as regularized_admm_messages
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from gurobipy import GRB, GurobiError
from te.algorithms.base import TrafficEngineeringLP, GurobiSolverParams, SolverParams
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, Commodity
from topologies.utils import (get_edge_indexing, get_graph_M_matrix, 
                              get_adjacency_null_space, get_feasible_flow_assignment)
from te.algorithms.utils import (check_centralized_flow_conservation,
                                 check_capacity_constraint,
                                 optimize_or_scream)
from protos.regularized_admm.regularized_admm_pb2_grpc import (
    RegularizedADMMSolverServicer, RegularizedADMMSolverStub, 
    add_RegularizedADMMSolverServicer_to_server
)
from google.protobuf.empty_pb2 import Empty


@dataclass
class MultiProcessesorRegularizedADMMSolverParams(GurobiSolverParams):
    NumberOfNodeProcesses: int = 1
    NumberOfEpochs: int = 1000
    NumberOfNetworkUpdates: int = te.constants.DEFAULT_NUMBER_OF_NETWORK_UPDATES
    Rho: float = te.constants.DEFAULT_RHO
    Eta: float = te.constants.DEFAULT_ETA
    Epsilon: float = te.constants.DEFAULT_EPSILON_KE
    Seed: int = te.constants.DEFAULT_SEED


@dataclass
class RegularizedADMMNodeModelRPCParams:
    ip: str
    port: int
    number_of_workers: int


@dataclass
class RegularizedADMMRPCParams:
    ip_list: Union[str, List[str]] = "localhost"
    port_list: Union[int, List[int]] = 13000
    number_of_workers_per_node: int = 1
    number_of_controller_workers: int = 1


def array_to_serialized_message(array: np.ndarray) -> regularized_admm_messages.SerializedNumpyArrayMessage:
    return regularized_admm_messages.SerializedNumpyArrayMessage(array=array.tobytes(), dims=list(array.shape))
def serialized_message_to_array(message: regularized_admm_messages.SerializedNumpyArrayMessage) -> np.ndarray:
    return np.reshape(np.frombuffer(message.array), tuple(message.dims))


class ControllerModel(TrafficEngineeringLP):
    def __init__(self, graph: nx.DiGraph, solver_params: MultiProcessesorRegularizedADMMSolverParams,
                 Xo_e_start: np.ndarray):
        super().__init__()
        self._graph = graph
        self._solver_params = solver_params
        self._rng = np.random.default_rng(seed=solver_params.Seed)

        self._NUM_EDGES: Optional[int] = None
        self._env: gurobipy.Env = None

        self._model_controller: Optional[gurobipy.Model] = None
        self._objective_controller: Optional[gurobipy.QuadExpr] = None

        # Both just vectors of length `n`
        self._Xo_e_start: np.ndarray = Xo_e_start
        self._Xo_e: Optional[gurobipy.tupledict] = None
        self._Zo_e: Optional[np.ndarray] = None
        # Just a variable between 0 and 1
        self._utility: Optional[gurobipy.Var] = None
        # Residual of outer ADMM. A vector of length `n`.
        self._r_e: Optional[np.ndarray] = None

        self._initialize_variables_and_residuals()

    @property
    def alg_name(self) -> str:
        return 'Multi-Process, Node-Distributed ADMM'
    @property
    def graph(self) -> nx.DiGraph:
        return self._graph
    @property
    def traffic(self) -> TrafficMatrixBase:
        raise ValueError("Shouldn't be used ...")
    @property
    def params(self) -> SolverParams:
        return self._solver_params
    @property
    def commodity_list(self) -> List[Commodity]:
        raise ValueError("Shouldn't be used ...")
    @property
    def objective_value(self) -> float:
        raise ValueError("Shouldn't be used ...")
    @property
    def objective_trace(self) -> Optional[List[float]]:
        raise ValueError("Shouldn't be used ...")
    
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
    
    def get_solution_commodity_list(self):
        raise ValueError("Shouldn't be used ...")


class NodeModel(TrafficEngineeringLP):
    def __init__(self, index: int, K: int, commodities: List[Commodity], X_ek_start: np.ndarray, 
                 NULL_M: np.ndarray, solver_params: MultiProcessesorRegularizedADMMSolverParams,
                 rpc_params: RegularizedADMMNodeModelRPCParams):
        super().__init__()
        self.index = index
        self._commodity_list = commodities
        self._NULL_M: np.ndarray = NULL_M
        self._solver_params = solver_params
        self._rpc_params = rpc_params

        self._K = K
        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None

        self._env: gurobipy.Env = None

        self._model_nodes: Optional[List[gurobipy.Model]] = None
        self._objective_nodes: Optional[List[gurobipy.QuadExpr]] = None

        # These need not be managed by Gurobi
        self._X_ek_start: np.ndarray = X_ek_start
        self._u_t_scattered: Optional[np.ndarray] = None
        self._P_bar_t_scattered: Optional[np.ndarray] = None
        self._Y_bar_t_scattered: Optional[np.ndarray] = None

        # Slice of global value. A `T x K` matrix that we treat as a list of `K` vectors of length `T`
        self._Y_tk: Optional[List[gurobipy.tupledict]] = None

        # The gRPC server
        self._is_active: bool = False
        self._server: Optional[grpc.Server] = None
        self._listener: Optional[NodeModelListener] = None

        self._initialize_variables_and_residuals()
        self._initialize_listener()

        for sig in ('TERM', 'INT'):
            signal.signal(getattr(signal, 'SIG'+sig), self.int_handler)
        
        self._start_listener()
    
    @staticmethod
    def spawn_and_wait(index: int, K: int, commodities: List[Commodity], X_ek_start: np.ndarray, 
                       NULL_M: np.ndarray, solver_params: MultiProcessesorRegularizedADMMSolverParams,
                       rpc_params: RegularizedADMMNodeModelRPCParams):
        node_model = NodeModel(index, K, commodities, X_ek_start, NULL_M, solver_params, rpc_params)
        node_model.wait()
    
    @classmethod
    def spawn(cls, index: int, K: int, commodities: List[Commodity], X_ek_start: np.ndarray, 
                   NULL_M: np.ndarray, solver_params: MultiProcessesorRegularizedADMMSolverParams,
                   rpc_params: RegularizedADMMNodeModelRPCParams) -> multiprocessing.Process:
        proc = multiprocessing.Process(target=cls.spawn_and_wait, args=(index, K, commodities, X_ek_start, NULL_M, solver_params, rpc_params))
        proc.start()
        return proc

    @property
    def rpc_params(self) -> RegularizedADMMNodeModelRPCParams:
        return self._rpc_params

    @property
    def graph(self) -> nx.DiGraph:
        raise ValueError("Shouldn't be used ...")
    @property
    def traffic(self) -> TrafficMatrixBase:
        raise ValueError("Shouldn't be used ...")
    @property
    def params(self) -> SolverParams:
        return self._solver_params
    @property
    def commodity_list(self) -> List[Commodity]:
        return self._commodity_list
    @property
    def objective_value(self) -> float:
        raise ValueError("Shouldn't be used ...")
    @property
    def objective_trace(self) -> Optional[List[float]]:
        raise ValueError("Shouldn't be used ...")

    def _initialize_variables_and_residuals(self):
        assert self._u_t_scattered is None
        assert self._P_bar_t_scattered is None
        assert self._Y_bar_t_scattered is None
        
        NUM_EDGES, T = np.shape(self._NULL_M)
        self._T = T
        self._NUM_EDGES = NUM_EDGES
        
        self._u_t_scattered = np.zeros(shape=(T,))
        self._P_bar_t_scattered = np.zeros(shape=(T,))
        self._Y_bar_t_scattered = np.zeros(shape=(T,))

    def _initialize_listener(self):
        assert self._server is None and self._listener is None

        IP = self._rpc_params.ip
        PORT = self._rpc_params.port
        NUM_WORKERS = self._rpc_params.number_of_workers
        self._server = grpc.server(thread_pool=ThreadPoolExecutor(max_workers=NUM_WORKERS))
        self._listener = NodeModelListener(self)
        add_RegularizedADMMSolverServicer_to_server(self._listener, self._server)
        addr = ":".join([IP, str(PORT)])
        self._server.add_insecure_port(addr)

        print(f"[NODE {self.index}] Initialized listener at address {addr}")
    
    def _start_listener(self):
        assert self._server is not None and self._listener is not None
        self._server.start()
        self._is_active = True

        print(f"[NODE {self.index}] Listener started")
    
    def _stop_listener(self):
        self._is_active = False
        if self._server is not None:
            self._server.stop(1)
    
    def wait(self):
        if self._server is not None:
            print(f"[NODE {self.index}] Will now wait for termination.")
            self._server.wait_for_termination()
        print(f"[NODE {self.index}] Will soon terminate")
    
    def int_handler(self, _, __):
        self._stop_listener()
        try:
            self.close()
        finally:
            pass
    
    def _update_u_t_scattered(self, new_u_t: np.ndarray):
        self._u_t_scattered = new_u_t
    def _update_P_bar_t_scattered(self, new_P_bar_t: np.ndarray):
        self._P_bar_t_scattered = new_P_bar_t
    def _update_Y_bar_t_scattered(self, new_Y_bar_t: np.ndarray):
        self._Y_bar_t_scattered = new_Y_bar_t
    
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
        T = self._T
        NULL_M = self._NULL_M
        Y_K_t = self._Y_tk[k]
        exp = gurobipy.LinExpr()
        exp.addConstant(self._X_ek_start[e, k])
        for t in range(T):
            exp.addTerms(NULL_M[e, t], Y_K_t[t])
        return exp

    def _get_X_k_sum(self) -> np.ndarray:
        K_SLICE = len(self._commodity_list)
        NUM_EDGES = self._NUM_EDGES
        return np.array([
            np.sum([
                self._get_X_ek_local(e, k).getValue() for k in range(K_SLICE)
            ]) for e in range(NUM_EDGES)
        ])

    def _get_Y_k_old_local(self, k: int) -> np.ndarray:
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
        # print(f"[NODE {self.index}] Adding constraints to models")

        NUM_EDGES = self._NUM_EDGES
        MODEL_NODES = self._model_nodes
        
        # Non-negativity constraint for node models
        for k, node_model in enumerate(MODEL_NODES):
            for e in range(NUM_EDGES):
                node_model.addConstr(0 <= self._get_X_ek_local(e, k))

    def _update_node_objective(self, Y_BAR_T_scattered, P_BAR_T_scattered, U_T_scattered):
        assert Y_BAR_T_scattered is not None
        assert P_BAR_T_scattered is not None
        assert U_T_scattered is not None

        # print(f"[NODE {self.index}] Updating objectives")

        self._update_P_bar_t_scattered(P_BAR_T_scattered)
        self._update_Y_bar_t_scattered(Y_BAR_T_scattered)
        self._update_u_t_scattered(U_T_scattered)

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
        # For this first call, we can just pass in zero arrays
        zero = np.zeros((self._T,))
        self._update_node_objective(zero, zero, zero)

    def close(self):
        if self._model_nodes:
            for node_model in self._model_nodes:
                node_model.close()
        if self._env:
            self._env.close()
    
    def make_lp(self):
        # print(f"[NODE {self.index}] Making problem model")
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

    def get_solution_commodity_list(self):
        raise ValueError("Shouldn't be used ...")


class NodeModelListener(RegularizedADMMSolverServicer):
    def __init__(self, model: NodeModel):
        super().__init__()
        self._model = model
    
    def Optimize(self, request, context):
        runtime = self._model.solve()
        return regularized_admm_messages.OptimizationRuntime(runtime=runtime)

    def GatherXEK(self, request, context):
        return array_to_serialized_message(self._model._make_X_ek())
    
    def GatherYSum(self, request, context):
        return array_to_serialized_message(self._model._get_Y_sum_local())
    
    def GatherXKSum(self, request, context):
        return array_to_serialized_message(self._model._get_X_k_sum())
    
    def ScatterInnerADMMLoopUpdates(self, request: regularized_admm_messages.NodeObjectiveUpdateMessage, context):
        Y_BAR_T = serialized_message_to_array(request.Y_BAR_T)
        P_BAR_T = serialized_message_to_array(request.P_BAR_T)
        U_T = serialized_message_to_array(request.U_T)
        self._model._update_node_objective(Y_BAR_T, P_BAR_T, U_T)
        return Empty()
    
    def Close(self, request: regularized_admm_messages.CloseMessage, context):
        if request.shutdown:
            self._model._stop_listener()
        self._model.close()
        return Empty()
    
    def MakeProblem(self, request, context):
        self._model.make_lp()
        return Empty()
    
    def Reset(self, request: regularized_admm_messages.ResetMessage, context):
        self._model.reset(request.with_params)
        return Empty()
    
    def CheckProblem(self, request: regularized_admm_messages.ProblemCheckRequest, context):
        Y_BAR_T = serialized_message_to_array(request.Y_BAR_T)
        P_BAR_T = serialized_message_to_array(request.P_BAR_T)
        try:
            self._model.check(Y_BAR_T, P_BAR_T)
            result = regularized_admm_messages.ProblemCheckResult(ok=True)
        except AssertionError as e:
            result = regularized_admm_messages.ProblemCheckResult(ok=False, message=str(e))
        finally:
            return result


class MultiProcessorRegularizedADMMLP(TrafficEngineeringLP):
    def __init__(self, graph: nx.DiGraph, traffic: TrafficMatrixBase, 
                 solver_params: MultiProcessesorRegularizedADMMSolverParams,
                 rpc_params: RegularizedADMMRPCParams) -> None:
        super().__init__()
        self._graph = graph
        self._M = get_graph_M_matrix(graph)
        self._traffic = traffic
        self._solver_params = solver_params
        self._rpc_params = rpc_params
        self._rng = np.random.default_rng(seed=solver_params.Seed)
        self._commodity_list = traffic_to_commodity(self._traffic)

        self._edge_indexing = get_edge_indexing(graph)
        self._NULL_M: np.ndarray = None
        
        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None
        self._BASE_NUM_NODE_LPS: Optional[int] = None
        self._REM_NUM_NODE_LPS: Optional[int] = None
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
        self._node_lps: Optional[List[multiprocessing.Process]] = None
        self._node_channels: Optional[List[grpc.Channel]] = None
        self._node_stubs: Optional[List[RegularizedADMMSolverStub]] = None

        # Thread pool for handling broadcasts
        self._broadcast_thread_pool = ThreadPoolExecutor(max_workers=rpc_params.number_of_controller_workers)

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
        return self._controller_lp._utility.X
    
    @property
    def objective_trace(self) -> Optional[List[float]]:
        return self._objective_trace

    def _set_initial_feasible_solution(self):
        self._X_ek_start = get_feasible_flow_assignment(self._graph, self._commodity_list)
        # check_centralized_flow_conservation(self._X_ek_start, self._graph, self._commodity_list, self._solver_params.FeasibilityTol)
    
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
        self._P_bar_t = np.zeros(shape=(T,))
        self._Y_bar_t = np.zeros(shape=(T,))
    
    def _partition_commodities(self):
        assert self._BASE_NUM_NODE_LPS is None and self._REM_NUM_NODE_LPS is None
        K = len(self._commodity_list)
        NUM_PROCS = self._solver_params.NumberOfNodeProcesses
        BASE_NUM_NODE_LPS = K // NUM_PROCS
        REM_NUM_NODE_LPS = K % NUM_PROCS
        
        # For now, just a simple rolling assignment
        commodity_slices = dict()
        commodity_counter = 0
        for node_index in range(self._solver_params.NumberOfNodeProcesses):
            n = BASE_NUM_NODE_LPS+1 if node_index+1 <= REM_NUM_NODE_LPS else BASE_NUM_NODE_LPS
            commodity_slices[node_index] = [commodity_counter + i for i in range(n)]
            commodity_counter += n

        print(f"Total number of commodities: {K}")
        print(f"Number of node processes: {NUM_PROCS}")
        print(f"BASE / REM: {BASE_NUM_NODE_LPS}: {REM_NUM_NODE_LPS}")

        self._BASE_NUM_NODE_LPS = BASE_NUM_NODE_LPS
        self._REM_NUM_NODE_LPS = REM_NUM_NODE_LPS
        self._commodity_slices = commodity_slices
    
    def _make_models(self):
        assert self._controller_lp is None and self._node_lps is None
        K = len(self._commodity_list)
        K_SLICES = self._commodity_slices
        NUM_PROCS = self._solver_params.NumberOfNodeProcesses
        NULL_M = self._NULL_M
        RPC_PARAMS = self._rpc_params
        self._controller_lp = ControllerModel(self._graph, self._solver_params, np.sum(self._X_ek_start, axis=1))
        node_rpc_params = [
            RegularizedADMMNodeModelRPCParams(
                ip = (RPC_PARAMS.ip_list if isinstance(RPC_PARAMS.ip_list, str) else RPC_PARAMS.ip_list[node_index]),
                port = (RPC_PARAMS.port_list + node_index if isinstance(RPC_PARAMS.port_list, int) else RPC_PARAMS.port_list[node_index]),
                number_of_workers = RPC_PARAMS.number_of_workers_per_node
            ) for node_index in range(NUM_PROCS)
        ]
        self._node_lps = [
            NodeModel.spawn(
                index=node_index, K=K, commodities=K_SLICES[node_index], 
                X_ek_start=self._X_ek_start[:, K_SLICES[node_index]], NULL_M=NULL_M, 
                solver_params=self._solver_params, rpc_params=node_rpc_params[node_index])
            for node_index in range(NUM_PROCS)
        ]
        self._node_channels = [
            grpc.insecure_channel(target=":".join([rpc_param.ip, str(rpc_param.port)]))
            for rpc_param in node_rpc_params
        ]
        self._node_stubs = [RegularizedADMMSolverStub(ch) for ch in self._node_channels]
    
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
              f"\t NUMBER OF INDEPENDENT QPs PER NODE: {K // M}\n"
              f"\t NUMBER OF VARIABLES PER QP PER NODE: {T}\n"
              f"\t NUMBER CONSTRAINTS PER QP PER NODE: {T}\n")
        
    def _make_variables(self):
        assert self._controller_lp is not None and \
               self._node_lps is not None
        
        self._controller_lp._make_variables()
        # make node variables later
    
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
        # make node constraints later
    
    def _update_controller_objective(self):
        assert self._controller_lp is not None
        self._controller_lp._update_controller_objective()
    
    def _update_node_objectives(self):
        assert self._node_lps is not None
        Y_BAR_T = array_to_serialized_message(self._Y_bar_t)
        P_BAR_T = array_to_serialized_message(self._P_bar_t)
        U_T = array_to_serialized_message(self._u_t)
        update_request = regularized_admm_messages.NodeObjectiveUpdateMessage(
            Y_BAR_T=Y_BAR_T, P_BAR_T=P_BAR_T, U_T=U_T
        )
        # TODO: Make async
        wait([
            self._broadcast_thread_pool.submit(node_stub.ScatterInnerADMMLoopUpdates, update_request) 
            for node_stub in self._node_stubs
        ])
        
    
    def _add_objective(self):
        assert self._controller_lp is not None and \
               self._node_lps is not None

        self._update_controller_objective()
        # Make objective for nodes later
    
    def _gather_Y_bar(self):
        assert self._node_lps is not None
        K = len(self._commodity_list)

        # TODO: Make async
        # Y_SUM_gathered = [
        #     serialized_message_to_array(node_stub.GatherYSum(Empty()))
        #     for node_stub in self._node_stubs
        # ]
        Y_SUM_gathered = self._broadcast_thread_pool.map(
            lambda node_stub: serialized_message_to_array(node_stub.GatherYSum(Empty())), self._node_stubs
        )
        self._Y_bar_t = np.sum(Y_SUM_gathered, axis=0) / K
    
    def _gather_X_k_sum(self):
        assert self._node_lps is not None

        # TODO: Make async
        # X_k_sum_gathered = [
        #     serialized_message_to_array(node_stub.GatherXKSum(Empty()))
        #     for node_stub in self._node_stubs
        # ]
        X_k_sum_gathered = self._broadcast_thread_pool.map(
            lambda node_stub: serialized_message_to_array(node_stub.GatherXKSum(Empty())), self._node_stubs
        )
        self._X_k_sum_e = np.sum(X_k_sum_gathered, axis=0)
    
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
        self._controller_lp._Zo_e = Zo_e
    
    def _update_r_e(self):
        assert self._controller_lp is not None and \
               self._node_lps is not None

        """
        The update rule for r_e is:
            r_e \gets r_e + (X_oe - \sum_k X_ke)/2
        """

        R_E = self._controller_lp._r_e
        XO_E = self._controller_lp._Xo_e
        NUM_EDGES = self._NUM_EDGES
        X_KE_SUM_E = self._X_k_sum_e
        r_e = np.zeros((NUM_EDGES,))
        for e in range(NUM_EDGES):
            r_e[e] = R_E[e] + (XO_E[e].X - X_KE_SUM_E[e]) / 2
        self._controller_lp._r_e = r_e
    
    def close(self):
        if self._controller_lp:
            self._controller_lp.close()
        if self._node_lps:
            for node_stub in self._node_stubs:
                node_stub.Close(regularized_admm_messages.CloseMessage(shutdown=True))
            for node_proc in self._node_lps:
                node_proc.join()
    
    def make_lp(self):
        t_start = time.time()
        print("Spawning processes ...")
        self._make_models()
        print("Starting to create the model")
        self._make_variables()
        self._add_constraints()
        self._add_objective()
        wait([
            self._broadcast_thread_pool.submit(node_stub.MakeProblem, Empty()) 
            for node_stub in self._node_stubs
        ])
        print(f"Built model in {str(np.round(time.time() - t_start, 2))} seconds.")
    
    def reset(self, with_params: False):
        self._controller_lp.reset(with_params=with_params)
        for node_stub in self._node_stubs:
            node_stub.Reset(regularized_admm_messages.ResetMessage(with_params=with_params))

    def _gather_X_ek(self):
        # TODO: Make async
        # stack = [serialized_message_to_array(node_stub.GatherXEK(Empty())) for node_stub in self._node_stubs]
        stack = self._broadcast_thread_pool.map(
            lambda node_stub: serialized_message_to_array(node_stub.GatherXEK(Empty())),
            self._node_stubs
        )
        self._X_ek = np.hstack(stack)
    
    def solve(self, params: SolverParams = None) -> float:
        assert params is None
        
        MODEL_CONTROLLER = self._controller_lp
        NODE_STUBS = self._node_stubs
        NUM_PROCS = len(NODE_STUBS)
        PARAMS = self._solver_params

        total_runtime = 0

        try:
            for _ in tqdm.tqdm(range(PARAMS.NumberOfEpochs)):
                t_nodes: Dict[int, List[float]] = defaultdict(list)

                # First, let the controller decide what the utilization is
                t_controller = MODEL_CONTROLLER.solve()
                # Now, do in-network optimization
                for _ in range(PARAMS.NumberOfNetworkUpdates):
                    for node_index, node_stub in enumerate(NODE_STUBS):
                        t_nodes[node_index].append(node_stub.Optimize(Empty()).runtime)
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
                max_node_time = max(sum(t_nodes[node_index]) for node_index in range(NUM_PROCS))
                total_runtime += t_controller + max_node_time
            
            # Build flow assignments
            self._gather_X_ek()
            return total_runtime
        except GurobiError as e:
            print(f'Error code {e.errno}: {e}')
            return -1
    
    def check(self, feasibility_tol: Optional[float] = None, feasibility_ratio: Optional[float] = None):
        PARAMS = self._solver_params
        Y_BAR_T = array_to_serialized_message(self._Y_bar_t)
        P_BAR_T = array_to_serialized_message(self._P_bar_t)
        check_request = regularized_admm_messages.ProblemCheckRequest(Y_BAR_T, P_BAR_T)
        # Check outer ADMM consensus
        self._controller_lp.check()
        # Check inner ADMM consensus
        for node_index, node_stub in enumerate(self._node_stubs):
            result = node_stub.CheckProblem(check_request)
            if not result.ok:
                print(f"Node {node_index} has not converged correctly.\n{result.message}")
        # Now, check flow conservation ...
        X_EK = self._X_ek
        check_centralized_flow_conservation(X_EK, self._graph, self._commodity_list, PARAMS.FeasibilityTol)
        check_capacity_constraint(
            X_EK, self._graph, self._commodity_list, 
            feasibility_tol=feasibility_tol, feasibility_ratio=feasibility_ratio
        )

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
