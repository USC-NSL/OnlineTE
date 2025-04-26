import time
import tqdm
import signal
import gurobipy
import numpy as np
import networkx as nx
from typing import Optional, List, Tuple
from gurobipy import GRB, GurobiError
from utils.logging import as_warning, as_info, log_subsection_separator
from te.algorithms.base import TrafficEngineeringLP, SolverParams
from te.algorithms.solution import EdgeBasedMinimizeMaximumUtilitySolution
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, Commodity
from topologies.utils import get_edge_indexing, get_graph_M_matrix, get_adjacency_null_space
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.cpu_utils import (CPUArray, DoublePrecisionCPUArray, 
                                                 cpu_array, cpu_zeros, cpu_double_array, 
                                                 set_cpu_float_precision)
from te.algorithms.statistics.helpers import record_cpu_runtime, record_return_value
from te.algorithms.utils import (check_capacity_constraint, optimize_or_scream, make_model, 
                                 get_solution_maximum_utilization)
from .controller_backends import get_backend
from .controller_backends.base import ControllerCommunicationBackendBase, NetworkUpdate
from . import AsynchronousADMMSolverParams, AsynchronousADMMControllerRPCParams
from te.algorithms.sub_algorithms.feasible_assignment import get_feasible_flow_assignment
from te.algorithms.sub_algorithms.admm_consensus_test import (outer_admm_consensus_test, 
                                                              inner_admm_consensus_test)
from te.algorithms.sub_algorithms.flow_conservation_test import check_flow_conservation


class ControllerNode(TrafficEngineeringLP):    
    def __init__(self, graph: nx.DiGraph, traffic: TrafficMatrixBase, 
                 solver_params: AsynchronousADMMSolverParams,
                 rpc_params: AsynchronousADMMControllerRPCParams) -> None:
        super().__init__()
        self._graph = graph
        self._M = get_graph_M_matrix(graph)
        self._traffic = traffic
        self._solver_params = solver_params
        self._rpc_params = rpc_params
        self._rng = np.random.default_rng(seed=solver_params.Seed)
        self._commodity_list = traffic_to_commodity(self._traffic)

        self._edge_indexing = get_edge_indexing(graph)
        self._NULL_M: CPUArray = None
        self._NNT_M: CPUArray = None
        
        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None

        self._env: gurobipy.Env = None

        self._model_controller: Optional[gurobipy.Model] = None
        self._objective_controller: Optional[gurobipy.QuadExpr] = None

        self._capacities: Optional[DoublePrecisionCPUArray] = None
        self._c_norm: Optional[float] = None

        self._X_ek: Optional[CPUArray] = None
        self._X_ek_start: Optional[CPUArray] = None
        self._Xo_e_start: Optional[CPUArray] = None
        self._Xo_e_assigned: Optional[CPUArray] = None
        self._Xo_e: Optional[gurobipy.tupledict] = None
        self._Zo_e: Optional[CPUArray] = None
        self._utility: Optional[gurobipy.Var] = None
        self._r_e: Optional[CPUArray] = None

        self._P_bar_t: Optional[CPUArray] = None
        self._Y_bar_t: Optional[CPUArray] = None
        self._u_t: Optional[CPUArray] = None
        self._partitioned_Y_bar: Optional[List[CPUArray]] = None

        self._backend: Optional[ControllerCommunicationBackendBase] = None

        self._objective_trace: List[Tuple[float, float]] = []
        self._objective_gap_trace = []

        assert rpc_params.NumWorkers >= solver_params.Upsilon, \
            'The controller update set cannot be larger than number of workers!: '\
            f'{solver_params.Upsilon} > {rpc_params.NumWorkers}'
        
        self._is_active = False
        self._die_on_next_int = False
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.die)

        set_global_precision(solver_params.Precision)
        set_cpu_float_precision()
    
    def initialize(self):
        self._set_initial_feasible_solution()
        self._backend = get_backend(self._rpc_params)
        self._backend.Upsilon = self._solver_params.Upsilon
        self._set_NULL_M()
        self._initialize_variables_and_residuals()
        self._partitioned_Y_bar = \
            [cpu_zeros((self._T,)) for _ in range(self._rpc_params.NumWorkers)]
        self._report_problem_size()
        self._backend.start()
        self._is_active = True
    
    @property
    def alg_name(self) -> str:
        return 'Asynchronous ADMM'

    @property
    def is_active(self) -> bool:
        return self._is_active
    
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
    def objective_trace(self) -> Optional[List[Tuple[float, float]]]:
        return self._objective_trace

    @property
    def objective_gap_trace(self) -> Optional[List[float]]:
        return self._objective_gap_trace
    
    @property
    def assignments(self) -> np.ndarray:
        assert self._X_ek is not None
        return self._X_ek

    def stop(self, _, __):
        if self._die_on_next_int:
            signal.raise_signal(signal.SIGTERM)
        else:
            print(as_warning('SIGINT: Stopping worker. Invoke again to kill the process.'))
            if self._backend:
                self._backend.stop()
            self._is_active = False
            self._die_on_next_int = True
    
    def die(self, _, __):
        print(as_warning('SIGTERM: Killing the worker.'))
        if self._backend:
            self._backend.die()
        self._is_active = False

    @record_cpu_runtime('Feasible-Assignment')
    def _set_initial_feasible_solution(self):
        self._X_ek_start = get_feasible_flow_assignment(self._graph, self._commodity_list)
        self._Xo_e_start = np.sum(self._X_ek_start, axis=1)
    
    def _set_NULL_M(self):
        M = self._M
        _, n = M.shape
        N = cpu_array(get_adjacency_null_space(M))
        T = N.shape[1]
        self._NULL_M = N
        self._NNT_M = N @ N.T
        self._T = T
        self._NUM_EDGES = n
    
    def _initialize_variables_and_residuals(self):
        T = self._T
        NUM_EDGES = self._NUM_EDGES
        self._capacities = cpu_double_array(
            [item[-1] for item in self._graph.edges(data='capacity')])
        self._c_norm = np.linalg.norm(self._capacities)
        self._r_e = cpu_zeros((NUM_EDGES,))
        self._u_t = cpu_zeros((T,))
        self._Zo_e = cpu_array(self._Xo_e_start)
        self._P_bar_t = cpu_zeros((T,))
        self._Y_bar_t = cpu_zeros((T,))
        self._X_ek = cpu_array(self._X_ek_start)
    
    def are_network_nodes_ready(self) -> bool:
        return self._backend.are_network_nodes_ready()
    
    def _report_problem_size(self):
        M = len(self._graph.nodes)
        N = len(self._graph.edges)
        T = self._T
        K = len(self._commodity_list)

        print(as_info(log_subsection_separator()))
        print(as_info(f"Graph Size: {M} nodes | {N} edges"))
        print(as_info(f"Number of commodities: {K}"))
        print(as_info(f"Nullity of commodity assignment matrix: {T}"))
        print(as_info(log_subsection_separator()))

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
            make_model('EdgeBasedDistributedTE_Controller', 
                       params=PARAMS, env=ENV, BarConvTol=PARAMS.BigGamma)
        
        self._Xo_e = MODEL_CONTROLLER.addVars(NUM_EDGES, lb=0.0, vtype=GRB.CONTINUOUS, name='XO_E')
        self._utility = MODEL_CONTROLLER.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='U')

        self._model_controller = MODEL_CONTROLLER
    
    def _get_F(self) -> np.ndarray:
        return self._Zo_e + self._r_e - self._Xo_e_start
    
    def _set_X_ek(self):
        result = self._backend.get_X_ek(
            basis=self._NULL_M, initial_feasible_solution=self._X_ek_start)
        if result is None:
            print(as_warning('Some worker nodes are not available. Unable to construct solution.'))
        else:
            self._X_ek = result
    
    def _add_constraints(self):
        assert self._model_controller is not None

        GRAPH = self._graph
        XO_E = self._Xo_e
        UTILITY = self._utility
        MODEL_CONTROLLER = self._model_controller

        for i, (_, _, c_e) in enumerate(GRAPH.edges(data='capacity')):
            MODEL_CONTROLLER.addConstr(XO_E[i] / c_e <= UTILITY)
    
    @record_cpu_runtime('Controller-Update')
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
    
    @record_cpu_runtime('Update-Zo-Re')
    def _update_Zo_e_and_r_e(self):
        assert self._model_controller is not None

        NUM_EDGES = self._NUM_EDGES
        R_E = self._r_e
        XO_E = self._Xo_e
        XO_E_ = cpu_array([XO_E[e].X for e in range(NUM_EDGES)])
        X_KE_SUM_E = self._Xo_e_start + len(self._commodity_list) * self._NULL_M @ self._Y_bar_t

        Zo_e = (XO_E_ + X_KE_SUM_E) / 2
        self._Zo_e = Zo_e
        self._r_e = R_E + (XO_E_ - X_KE_SUM_E) / 2
        self._Xo_e_assigned = X_KE_SUM_E

    def close(self):
        self._backend.close()
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
        self._backend.initialize_worker_nodes(
            self._solver_params,
            self._NULL_M, 
            self._X_ek_start
        )
        self._backend.set_active_commodity_count(len(self._commodity_list))
    
    def reset(self, with_params: False):
        self._model_controller.reset()
        if with_params:
            self._model_controller.resetParams()

    @record_return_value('PGD-Runtime')
    def _consume_updates(self, updates: List[Tuple[int, NetworkUpdate]]) -> int:
        runtimes = []
        for worker_id, update in updates:
            runtime, Y_bar = update
            self._partitioned_Y_bar[worker_id] = Y_bar
            runtimes.append(runtime)
        self._Y_bar_t = np.mean(self._partitioned_Y_bar, axis=0)
        self._update_P_bar()
        self._update_u_t()
        return max(runtimes)
    
    def _wait_for_minimum_updates(self) -> bool:
        gathered_updates = self._backend.get_network_updates()
        if len(gathered_updates) < self._solver_params.Upsilon:
            # This only happens if the solution is interrupted
            print(as_warning('Solution interrupted, will no longer consume updates.'))
            return False
        self._consume_updates(gathered_updates)
        return True
    
    @record_cpu_runtime('Solve')
    def solve(self, params = None):
        MODEL_CONTROLLER = self._model_controller
        PARAMS = self._solver_params
        EPOCHS = params if params is not None else PARAMS.NumberOfEpochs
        
        try:
            t = time.time()
            for _ in tqdm.tqdm(range(EPOCHS), bar_format='{l_bar}{bar:36}{r_bar}{bar:-36b}'):
            # for epoch in range(EPOCHS):
                if not self._is_active:
                    break
                optimize_or_scream(MODEL_CONTROLLER)
                if not self._wait_for_minimum_updates():
                    break
                self._update_Zo_e_and_r_e()
                self._update_controller_objective()
                self._backend.update_network_nodes(self._P_bar_t, self._Y_bar_t, self._u_t)
                
                self._objective_trace.append((
                    self._utility.X, 
                    get_solution_maximum_utilization(self._Xo_e_assigned, self._graph)
                ))
            self._set_X_ek()
            return time.time() - t
        except GurobiError as e:
            print(f'Error code {e.errno}: {e}')
            return -1

    
    def check(self, feasibility_tol: Optional[float] = None, 
              feasibility_ratio: Optional[float] = None, report: bool = False):
        NUM_EDGES = self._NUM_EDGES

        # Are outer ADMM pairs in consensus?
        XO_E = np.array([self._Xo_e[e].X for e in range(NUM_EDGES)])
        ZO_E = self._Zo_e
        outer_admm_consensus_test(XO_E, ZO_E, feasibility_tol, feasibility_ratio, report)
        
        # Are inner ADMM pairs in consensus?
        Y_BAR_T = self._Y_bar_t
        P_BAR_T = self._P_bar_t
        inner_admm_consensus_test(Y_BAR_T, P_BAR_T, feasibility_tol, feasibility_ratio, report)
        
        # Now, check flow conservation ...
        X_EK = self._X_ek
        check_flow_conservation(X_EK, self._graph, self._commodity_list, feasibility_tol, 
                                feasibility_ratio, report=report)
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
        self._traffic = tm
        self._commodity_list = traffic_to_commodity(self._traffic)
        self._set_initial_feasible_solution()
        self._Zo_e = cpu_array(self._Xo_e_start)
        self._backend.update_demands(self._X_ek_start)
    
    def initialize_to(self, solution: EdgeBasedMinimizeMaximumUtilitySolution):
        raise NotImplementedError
    
    def set_target(self, solution: EdgeBasedMinimizeMaximumUtilitySolution):
        raise NotImplementedError
    
    def add_solution_elements(self, solution: EdgeBasedMinimizeMaximumUtilitySolution):
        solution.add_solution_element(self._utility, name='utility')
        solution.add_solution_element(self._X_ek, name='assignments')
