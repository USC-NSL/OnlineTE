import grpc
import time
import grpc._channel
import tqdm
import gurobipy
import numpy as np
import networkx as nx
import te.constants
from typing import List, Tuple, Optional
from gurobipy import GRB, GurobiError
from concurrent.futures import ThreadPoolExecutor, wait
from te.algorithms.base import TrafficEngineeringLP, SolverParams
from te.algorithms.solution import GurobiEdgeBasedMinimizeMaximumUtilitySolution
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, Commodity
from topologies.utils import (get_edge_indexing, get_graph_M_matrix, 
                              get_adjacency_null_space, get_feasible_flow_assignment)
from te.algorithms.utils import check_capacity_constraint, optimize_or_scream, make_model, as_fail
from te.algorithms.formulations.edge_based_distributed_admm import DistributedADMMSolverParams, DistributedADMMControllerRPCParams
from te.algorithms.formulations.edge_based_distributed_admm.utils import serialized_message_to_array, array_to_serialized_message
import protos.distributed_lp.distributed_lp_pb2 as distributed_lp_messages
from protos.distributed_lp.distributed_lp_pb2_grpc import DistributedADMMSolverStub
from google.protobuf.empty_pb2 import Empty



class ControllerNode(TrafficEngineeringLP):
    def __init__(self, graph: nx.DiGraph, traffic: TrafficMatrixBase, solver_params: DistributedADMMSolverParams,
                 rpc_params: DistributedADMMControllerRPCParams) -> None:
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
        self._NNT_M: np.ndarray = None
        
        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None

        self._env: gurobipy.Env = None

        self._model_controller: Optional[gurobipy.Model] = None
        self._objective_controller: Optional[gurobipy.QuadExpr] = None

        self._capacities: Optional[np.ndarray] = None
        self._c_norm: Optional[float] = None

        self._X_ek: Optional[np.ndarray] = None
        self._X_ek_start: Optional[np.ndarray] = None
        self._Xo_e_start: Optional[np.ndarray] = None
        self._Xo_e: Optional[gurobipy.tupledict] = None
        self._Zo_e: Optional[np.ndarray] = None
        self._utility: Optional[gurobipy.Var] = None
        self._r_e: Optional[np.ndarray] = None

        self._P_bar_t: Optional[np.ndarray] = None
        self._Y_bar_t: Optional[np.ndarray] = None
        self._u_t: Optional[np.ndarray] = None

        self._worker_channels: List[grpc.Channel] = [
            grpc.insecure_channel(target=":".join([ip, str(port)]))
                for ip, port in self._rpc_params.addr_list
        ]
        self._worker_stubs: List[DistributedADMMSolverStub] = [
            DistributedADMMSolverStub(ch) for ch in self._worker_channels
        ]
        self._broadcast_thread_pool = ThreadPoolExecutor(max_workers=rpc_params.num_threads)

        self._objective_trace = []
        self._objective_gap_trace = []

        self._set_initial_feasible_solution()
        self._set_NULL_M()
        self._initialize_variables_and_residuals()
        self._report_problem_size()

    @property
    def alg_name(self) -> str:
        return 'Multi-Proces Unregulated ADMM'

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
        return self._objective_trace

    @property
    def objective_gap_trace(self) -> Optional[List[float]]:
        return self._objective_gap_trace
    
    @property
    def assignments(self) -> np.ndarray:
        assert self._X_ek is not None
        return self._X_ek

    def _set_initial_feasible_solution(self):
        self._X_ek_start = get_feasible_flow_assignment(self._graph, self._commodity_list)
        self._Xo_e_start = np.sum(self._X_ek_start, axis=1)
    
    def _set_NULL_M(self):
        M = self._M
        assert len(M.shape) == 2
        m, n = M.shape
        assert m < n
        N = get_adjacency_null_space(M)
        T = N.shape[1]
        assert np.allclose(np.matmul(N.T, N) - np.eye(T), 0)
        self._NULL_M = N
        self._NNT_M = N @ N.T
        self._T = T
        self._NUM_EDGES = n
    
    def _initialize_variables_and_residuals(self):
        T = self._T
        NUM_EDGES = self._NUM_EDGES
        self._capacities = np.array([item[-1] for item in self._graph.edges(data='capacity')])
        self._c_norm = np.linalg.norm(self._capacities)
        self._r_e = np.zeros(shape=(NUM_EDGES,))
        self._u_t = np.zeros(shape=(T,))
        self._Zo_e = np.copy(self._Xo_e_start)
        self._P_bar_t = np.zeros((T,))
        self._Y_bar_t = np.zeros((T,))
        self._X_ek = np.copy(self._X_ek_start)
    
    def is_node_ready(self, worker_id: int) -> bool:
        try:
            return self._worker_stubs[worker_id].QueryState(Empty()).ready
        except grpc._channel._InactiveRpcError:
            return False
    
    def are_network_nodes_ready(self) -> bool:
        return all(self._broadcast_thread_pool.map(
            self.is_node_ready, range(self._solver_params.NumWorkers)
        ))

    def _initialize_worker_nodes(self):
        NUM_WORKERS = self._solver_params.NumWorkers
        NULL_M = self._NULL_M
        X_EK_START_CHUNKS = np.array_split(self._X_ek_start, NUM_WORKERS, axis=1)
        WORKERS = self._worker_stubs
        wait([
            self._broadcast_thread_pool.submit(stub.InitializeWorkerNode, 
                                               distributed_lp_messages.InitMessage(
                                                   NULL_M=array_to_serialized_message(NULL_M),
                                                   X_EK_START=array_to_serialized_message(X_EK_START_CHUNKS[i])
                                               ))
                for i, stub in enumerate(WORKERS)
        ])
    
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

    def initialize_to(self, assignment: np.ndarray):
        raise NotImplementedError
        
    def _make_variables(self):
        assert self._model_controller is None
        
        NUM_EDGES = self._NUM_EDGES

        ENV = gurobipy.Env()
        ENV.setParam('OutputFlag', 0)
        ENV.start()
        self._env = ENV

        PARAMS = self._solver_params
        MODEL_CONTROLLER: gurobipy.Model = \
            make_model('EdgeBasedDistributedTE_Controller', params=PARAMS, env=ENV)
        
        self._Xo_e = MODEL_CONTROLLER.addVars(NUM_EDGES, lb=0.0, vtype=GRB.CONTINUOUS, name='XO_E')
        self._utility = MODEL_CONTROLLER.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='U')

        self._model_controller = MODEL_CONTROLLER
    
    def _get_F(self) -> np.ndarray:
        return self._Zo_e + self._r_e - self._Xo_e_start
    
    def _set_X_ek(self):
        serialized_chunks = self._broadcast_thread_pool.map(
            lambda stub: stub.RequestChunk(Empty()), self._worker_stubs
        )
        self._X_ek = self._X_ek_start + self._NULL_M @ np.hstack([
            serialized_message_to_array(chunk) for chunk in serialized_chunks
        ])
    
    def _add_constraints(self):
        assert self._model_controller is not None

        GRAPH = self._graph
        XO_E = self._Xo_e
        UTILITY = self._utility
        MODEL_CONTROLLER = self._model_controller

        for i, (_, _, c_e) in enumerate(GRAPH.edges(data='capacity')):
            MODEL_CONTROLLER.addConstr(XO_E[i] / c_e <= UTILITY)
    
    def _update_controller_objective(self):
        NUM_EDGES = self._NUM_EDGES
        UTILITY = self._utility
        XO_E = self._Xo_e
        ZO_E = self._Zo_e
        R_E = self._r_e
        RHO = self._solver_params.Rho
        MODEL_CONTROLLER = self._model_controller
        
        OBJECTIVE_CONTROLLER = gurobipy.QuadExpr()
        OBJECTIVE_CONTROLLER.addTerms(self._c_norm * np.sqrt(NUM_EDGES), UTILITY)
        for e in range(NUM_EDGES):
            OBJECTIVE_CONTROLLER += (RHO/2) * (XO_E[e] - ZO_E[e] + R_E[e]) ** 2
        MODEL_CONTROLLER.setObjective(OBJECTIVE_CONTROLLER, GRB.MINIMIZE)
        self._objective_controller = OBJECTIVE_CONTROLLER
    
    def _add_objective(self):
        assert self._model_controller is not None

        self._update_controller_objective()

    def _do_network_update(self, epoch: int):
        message = distributed_lp_messages.NetworkUpdateRequest(epoch=epoch)
        serialized_y_bar_chunks = self._broadcast_thread_pool.map(
            lambda stub: stub.DoNetworkUpdate(message), self._worker_stubs)
        self._Y_bar_t = np.mean([serialized_message_to_array(chunk) for chunk in serialized_y_bar_chunks], axis=0)
    
    def _update_P_bar(self):
        assert self._model_controller is not None

        K = len(self._commodity_list)
        ETA = self._solver_params.Eta
        RHO = self._solver_params.Rho
        U_T = self._u_t
        Y_BAR_T = self._Y_bar_t
        F_E = self._get_F()
        NULL_M = self._NULL_M
        P_BAR_T = (NULL_M.T @ F_E + (ETA/RHO) * (U_T + Y_BAR_T)) / (K + (ETA/RHO))
        self._P_bar_t = P_BAR_T
    
    def _update_u_t(self):
        assert self._model_controller is not None

        U_T = self._u_t
        Y_BAR_T = self._Y_bar_t
        P_BAR_T = self._P_bar_t

        self._u_t = U_T + (Y_BAR_T - P_BAR_T)
    
    def _reconvene_network_updates(self):
        self._update_P_bar()
        self._update_u_t()
        message = distributed_lp_messages.UpdateMessage(
            P_bar_t=array_to_serialized_message(self._P_bar_t),
            u_t = array_to_serialized_message(self._u_t)
        )
        wait([
            self._broadcast_thread_pool.submit(
                lambda: stub.UpdateWorkerNode(message)
            )
            for stub in self._worker_stubs
        ])
        print(self._P_bar_t)

    def _update_Zo_e(self):
        assert self._model_controller is not None

        NUM_EDGES = self._NUM_EDGES
        XO_E = self._Xo_e
        XO_E_ = np.array([XO_E[e].X for e in range(NUM_EDGES)])
        serialized_chunks = self._broadcast_thread_pool.map(
            lambda stub: stub.RequestAggregate(Empty()), self._worker_stubs)
        X_KE_SUM_E = np.sum([serialized_message_to_array(chunk) for chunk in serialized_chunks], axis=0)
        Zo_e = (XO_E_ + X_KE_SUM_E) / 2
        self._Zo_e = Zo_e

    def _update_r_e(self):
        assert self._model_controller is not None

        R_E = self._r_e
        XO_E = self._Xo_e
        NUM_EDGES = self._NUM_EDGES
        XO_E_ = np.array([XO_E[e].X for e in range(NUM_EDGES)])
        serialized_chunks = self._broadcast_thread_pool.map(
            lambda stub: stub.RequestAggregate(Empty()), self._worker_stubs)
        X_KE_SUM_E = np.sum([serialized_message_to_array(chunk) for chunk in serialized_chunks], axis=0)
        self._r_e = R_E + (XO_E_ - X_KE_SUM_E) /2
    
    def _close_node(self, worker_id: int):
        try:
            self._worker_stubs[worker_id].Close(Empty())
        except:
            pass

    def close(self):
        wait([
            self._broadcast_thread_pool.submit(lambda: self._close_node(i))
                for i in range(len(self._worker_stubs))
        ], timeout=5)
        if self._model_controller:
            self._model_controller.close()
        if self._env:
            self._env.close()
    
    def make_lp(self):
        t_start = time.time()
        print("Starting to create the model")
        self._make_variables()
        self._add_constraints()
        self._add_objective()
        print(f"Built model in {str(np.round(time.time() - t_start, 2))} seconds.")
        self._initialize_worker_nodes()
    
    def reset(self, with_params: False):
        self._model_controller.reset()
        if with_params:
            self._model_controller.resetParams()
    
    def solve(self, params: SolverParams = None) -> float:
        assert params is None
        
        MODEL_CONTROLLER = self._model_controller
        PARAMS = self._solver_params
        
        try:
            t = time.time()
            for epoch in tqdm.tqdm(range(PARAMS.NumberOfEpochs)):
                optimize_or_scream(MODEL_CONTROLLER)
                for i in reversed(range(PARAMS.NumberOfNetworkUpdates)):
                    self._do_network_update(epoch)
                    if i > 0:
                        self._reconvene_network_updates()

                self._update_Zo_e()
                self._update_r_e()
                self._reconvene_network_updates()
                self._update_controller_objective()

                self._objective_trace.append(self._utility.X)
            self._set_X_ek()
            return time.time() - t
        except GurobiError as e:
            print(f'Error code {e.errno}: {e}')
            return -1
    
    def check(self, feasibility_tol: Optional[float] = None, feasibility_ratio: Optional[float] = None):
        NUM_EDGES = self._NUM_EDGES
        T = self._T
        PARAMS = self._solver_params

        # TODO: This is not numerically stable ...
        def in_consensus(primal, pair):
            if abs(primal - pair) < te.constants.FLOAT_RES:
                return True
            if feasibility_tol is not None:
                return abs(primal - pair) < feasibility_tol
            return abs((primal - pair) / (primal + te.constants.FLOAT_RES)) < feasibility_ratio

        # Are outer ADMM pairs in consensus?
        XO_E = self._Xo_e
        ZO_E = self._Zo_e
        for e in range(NUM_EDGES):
            primal = XO_E[e].X
            pair = ZO_E[e]
            primal_str = str(np.round(primal, 4))
            pair_str = str(np.round(pair, 4))
            if not in_consensus(primal, pair):
                print(as_fail(f"Edge {e} --> Outer ADMM pairing is not in consensus with primal variable: {primal_str} vs {pair_str}"))
        
        # Are inner ADMM pairs in consensus?
        Y_BAR_T = self._Y_bar_t
        P_BAR_T = self._P_bar_t
        for t in range(T):
            primal = Y_BAR_T[t]
            pair = P_BAR_T[t]
            primal_str = str(np.round(primal, 4))
            pair_str = str(np.round(pair, 4))
            if not in_consensus(primal, pair):
                print(as_fail(f"Axis {t} --> Inner ADMM pairing is not in consensus with primal variable: {primal_str} vs {pair_str}"))
        
        # Now, check flow conservation ...
        X_EK = self._X_ek
        # check_centralized_flow_conservation(X_EK, self._graph, self._commodity_list, PARAMS.FeasibilityTol)
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
    
    def update_traffic_matrix(self, tm):
        raise NotImplementedError
    
    def initialize_to(self, solution: GurobiEdgeBasedMinimizeMaximumUtilitySolution):
        raise NotImplementedError
    
    def set_target(self, solution: GurobiEdgeBasedMinimizeMaximumUtilitySolution):
        raise NotImplementedError
    
    def add_solution_elements(self, solution):
        raise NotImplementedError
