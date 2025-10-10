import time
import signal
import gurobipy
import numpy as np
import networkx as nx
import asyncio.exceptions
from collections import defaultdict
from typing import List, Tuple, Optional
from gurobipy import GRB, GurobiError
from te.algorithms.base import (TrafficEngineeringLP, SolverParams, TrafficEngineeringLPCheckResult, TrafficEngineeringLPEvaluationParams, 
                                TrafficEngineeringLPObjectiveTrace)
from te.algorithms.solution import EdgeBasedMinimizeMaximumUtilitySolution
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, Commodity
from topologies.utils import get_graph_M_matrix, get_adjacency_null_space, get_commodity_in_out_mask
from topologies.utils import get_sparse_null_space, get_symbolic_graph_M_matrix
from utils.exceptions import SolutionInterrupted
from utils.logging import as_info, as_warning, log_subsection_separator, ShortTQDM
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.cpu_utils import (CPUArray, DoublePrecisionCPUArray, BooleanCPUArray,
                                                 cpu_array, cpu_zeros, cpu_double_array, 
                                                 set_cpu_float_precision)
from te.algorithms.utils import optimize_or_scream, make_model, get_solution_maximum_utilization
from te.algorithms.sub_algorithms.feasible_assignment import get_feasible_flow_assignment
from te.algorithms.sub_algorithms.admm_consensus_test import outer_admm_consensus_test, inner_admm_consensus_test, norm_in_consensus
from te.algorithms.sub_algorithms.link_capacity_test import check_capacity_constraint
from te.algorithms.sub_algorithms.flow_conservation_test import check_flow_conservation
from te.algorithms.statistics.helpers import record_cpu_runtime, record_return_value
from . import DistributedADMMSolverParams, DistributedADMMControllerRPCParams
from .controller_backends import get_backend, ControllerCommunicationBackendBase


class ControllerNode(TrafficEngineeringLP):
    def __init__(self, graph: nx.DiGraph, traffic: TrafficMatrixBase, solver_params: DistributedADMMSolverParams,
                 rpc_params: DistributedADMMControllerRPCParams) -> None:
        super().__init__()
        self._graph = graph
        self._M = get_graph_M_matrix(graph)
        self._symbolic_M = get_symbolic_graph_M_matrix(graph)
        self._traffic = traffic
        self._solver_params = solver_params
        self._rpc_params = rpc_params
        self._rng = np.random.default_rng(seed=solver_params.Seed)
        self._commodity_list = traffic_to_commodity(self._traffic)

        self._NULL_M: CPUArray = None
        self._NNT_M: CPUArray = None
        
        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None
        self._M_MASK: Optional[BooleanCPUArray] = None

        self._env: gurobipy.Env = None

        self._model_controller: Optional[gurobipy.Model] = None
        self._objective_controller: Optional[gurobipy.QuadExpr] = None

        self._capacities: Optional[DoublePrecisionCPUArray] = None
        self._c_norm: Optional[float] = None
        self._alpha: Optional[float] = None

        self._X_ek: Optional[CPUArray] = None
        self._X_ek_start: Optional[CPUArray] = None
        self._Z_e_start: Optional[CPUArray] = None
        self._Z_e: Optional[gurobipy.tupledict] = None
        self._X_ek_sum_e: Optional[CPUArray] = None
        self._utility: Optional[gurobipy.Var] = None
        self._r_e: Optional[CPUArray] = None

        self._utility_bound_constraints: Tuple[gurobipy.Constr, gurobipy.Constr] = None
        """Gives the dual variables `v_neg` and `v_pos`"""
        self._capacity_constraints: List[gurobipy.Constr] = None
        """Gives the dual variables `tau_e`, a vector of length `n`"""

        self._P_bar_t: Optional[CPUArray] = None
        self._Y_bar_t: Optional[CPUArray] = None
        self._u_t: Optional[CPUArray] = None

        self._backend: Optional[ControllerCommunicationBackendBase] = None

        self._objective_trace: TrafficEngineeringLPObjectiveTrace = TrafficEngineeringLPObjectiveTrace(['Perceived Utilization', 'Actual Utilization'])
        self._objective_gap_trace = []

        # These we call right now, as opposed to doing them under `initialize`
        set_global_precision(solver_params.Precision)
        set_cpu_float_precision()

        self._die_on_next_int = False
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.die)
    
    def initialize(self):
        # First, set the initial feasible solutions.
        # We will do this before spawning the backend, since if we use `gRPC`, 
        # this function may invoke `fork` which causes `gRPC` to spam warnings.
        self._set_initial_feasible_solution()
        # Now, create the backend
        self._backend = get_backend(self._rpc_params)
        # Initialize the algorithm
        self._set_NULL_M()
        self._initialize_variables_and_residuals()
        # Report what we are dealing with
        self._report_problem_size()
        # Now, start the backend
        self._backend.start()
    
    def stop(self, _, __):
        if self._die_on_next_int:
            signal.raise_signal(signal.SIGTERM)
        else:
            print(as_warning('SIGINT: Stopping solver. Invoke again to kill the process.'))
            if self._backend:
                self._backend.stop()
            self._die_on_next_int = True
            raise SolutionInterrupted
    
    def die(self, _, __):
        print(as_warning('SIGTERM: Killing the solver.'))
        if self._backend:
            self._backend.die()

    @property
    def alg_name(self) -> str:
        return 'Distributed Unregulated ADMM'

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
    def objective_trace(self) -> Optional[TrafficEngineeringLPObjectiveTrace]:
        return self._objective_trace

    @property
    def objective_gap_trace(self) -> Optional[List[float]]:
        return self._objective_gap_trace
    
    @property
    def assignments(self) -> np.ndarray:
        assert self._X_ek is not None
        return self._X_ek

    @record_cpu_runtime('Feasible-Assignment')
    def _set_initial_feasible_solution(self):
        self._X_ek_start = get_feasible_flow_assignment(self._graph, self._commodity_list)
        self._Z_e_start = np.sum(self._X_ek_start, axis=1)
        
        # TODO: Are these safe?
        # if self._T is not None:
        #     T = self._T
        #     NUM_EDGES = self._NUM_EDGES
        #     self._r_e = cpu_zeros((NUM_EDGES,))
        #     self._u_t = cpu_zeros((T,))
    
    def _set_NULL_M(self):
        M = self._M
        assert len(M.shape) == 2
        m, n = M.shape
        assert m < n
        N = cpu_array(get_adjacency_null_space(M))
        # N = cpu_array(get_sparse_null_space(self._symbolic_M))
        T = N.shape[1]
        self._NULL_M = N
        self._NNT_M = N @ N.T
        self._T = T
        self._NUM_EDGES = n
        self._M_MASK = get_commodity_in_out_mask(self.graph, self.commodity_list)
    
    def _get_Z_value(self) -> CPUArray:
        try:
            return cpu_array([self._Z_e[e].X for e in range(self._NUM_EDGES)])
        except AttributeError:
            return cpu_array(self._Z_e_start)
    
    def _initialize_variables_and_residuals(self):
        T = self._T
        NUM_EDGES = self._NUM_EDGES
        self._capacities = cpu_double_array([item[-1] for item in self._graph.edges(data='capacity')])
        self._c_norm = np.linalg.norm(self._capacities)
        self._r_e = cpu_zeros((NUM_EDGES,))
        self._u_t = cpu_zeros((T,))
        self._P_bar_t = cpu_zeros((T,))
        self._Y_bar_t = cpu_zeros((T,))
        self._X_ek = cpu_array(self._X_ek_start)
        self._X_ek_sum_e = cpu_array(self._Z_e_start)
    
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
        print(as_info("CONTROLLER PROBLEM:\n" +
              f"\t TOTAL NUMBER OF VARIABLES: {N + 1}\n"
              f"\t TOTAL NUMBER OF CONSTRAINTS: {N + 1}"))
        print(as_info(log_subsection_separator()))
        print(as_info("NODE PROBLEM:\n" +
              f"\t NUMBER OF INDEPENDENT QPs PER NODE: {M - 1}\n"
              f"\t NUMBER OF VARIABLES PER QP PER NODE: {T}\n"
              f"\t NUMBER CONSTRAINTS PER QP PER NODE: {T}"))
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
            make_model('EdgeBasedDistributedTE_Controller', params=PARAMS, env=ENV, BarConvTol=PARAMS.BigGamma)
        
        # self._Z_e = MODEL_CONTROLLER.addVars(NUM_EDGES, lb=0.0, vtype=GRB.CONTINUOUS, name='Z_E')
        self._Z_e = MODEL_CONTROLLER.addVars(NUM_EDGES, lb=float('-inf'), vtype=GRB.CONTINUOUS, name='Z_E')
        # self._utility = MODEL_CONTROLLER.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='U')
        self._utility = MODEL_CONTROLLER.addVar(lb=float('-inf'), vtype=GRB.CONTINUOUS, name='U')

        self._model_controller = MODEL_CONTROLLER
    
    def _get_F(self) -> np.ndarray:
        return self._get_Z_value() - self._Z_e_start - self._r_e
    
    def _set_X_ek(self):
        self._X_ek = self._backend.get_X_ek(basis=self._NULL_M, initial_feasible_solution=self._X_ek_start)
        # self._X_ek = np.multiply(
        #     self._backend.get_X_ek(basis=self._NULL_M, initial_feasible_solution=self._X_ek_start),
        #     self._M_MASK
        # )
    
    def _add_constraints(self):
        assert self._model_controller is not None

        GRAPH = self._graph
        Z_E = self._Z_e
        UTILITY = self._utility
        MODEL_CONTROLLER = self._model_controller

        # Utilization bound constraints
        u_low = MODEL_CONTROLLER.addConstr(UTILITY >= 0)
        u_high = MODEL_CONTROLLER.addConstr(-UTILITY >= -1)
        self._utility_bound_constraints = (u_low, u_high)

        # for i, (_, _, c_e) in enumerate(GRAPH.edges(data='capacity')):
        #     MODEL_CONTROLLER.addConstr(Z_E[i] / c_e <= UTILITY)
        capacity_constraints: List[gurobipy.Constr] = []
        for e, (_, _, c_e) in enumerate(GRAPH.edges.data('capacity')):
            capacity_constraints.append(MODEL_CONTROLLER.addConstr(UTILITY * c_e >= Z_E[e]))
        self._capacity_constraints = capacity_constraints
    
    @record_cpu_runtime('Controller-Update')
    def _update_controller_objective(self):
        NUM_EDGES = self._NUM_EDGES
        UTILITY = self._utility
        Z_E = self._Z_e
        X_EK_SUM_E = self._X_ek_sum_e
        R_E = self._r_e
        RHO = self._solver_params.Rho
        MODEL_CONTROLLER = self._model_controller
        ALPHA = self._c_norm * np.sqrt(NUM_EDGES)
        
        OBJECTIVE_CONTROLLER = gurobipy.QuadExpr()
        OBJECTIVE_CONTROLLER.addTerms(ALPHA, UTILITY)
        for e in range(NUM_EDGES):
            OBJECTIVE_CONTROLLER += (RHO/2) * (X_EK_SUM_E[e] - Z_E[e] + R_E[e]) ** 2
        MODEL_CONTROLLER.setObjective(OBJECTIVE_CONTROLLER, GRB.MINIMIZE)
        self._alpha = ALPHA
        self._objective_controller = OBJECTIVE_CONTROLLER
    
    def _add_objective(self):
        assert self._model_controller is not None

        self._update_controller_objective()

    @record_return_value('QP-Runtime')
    @record_cpu_runtime('Network-Update')
    def _do_network_update(self, epoch: int):
        if self._solver_params.NumberOfLocalUpdates > 0:
            max_run, self._Y_bar_t = self._backend.do_network_update(epoch, self._get_F())
        else:
            max_run, self._Y_bar_t = self._backend.do_network_update(epoch)
        return max_run
    
    def _update_P_bar(self):
        assert self._model_controller is not None

        K = len(self._commodity_list)
        ETA = self._solver_params.Eta
        RHO = self._solver_params.Rho
        U_T = self._u_t
        Y_BAR_T = self._Y_bar_t
        F_E = self._get_F()
        NULL_M = self._NULL_M
        # P_BAR_T = (NULL_M.T @ F_E + (ETA/RHO) * (U_T + Y_BAR_T)) / (K + (ETA/RHO))
        # TODO: See https://github.com/USC-NSL/DistributedTE/issues/29
        P_BAR_T = (NULL_M.T @ F_E / K + (ETA/RHO) * (U_T + Y_BAR_T)) / (1 + (ETA/RHO))
        self._P_bar_t = P_BAR_T
    
    def _update_u_t(self):
        assert self._model_controller is not None

        U_T = self._u_t
        Y_BAR_T = self._Y_bar_t
        P_BAR_T = self._P_bar_t

        self._u_t = U_T + (Y_BAR_T - P_BAR_T)
    
    @record_cpu_runtime('Update-Reconvene')
    def _reconvene_network_updates(self) -> bool:
        self._update_P_bar()
        self._update_u_t()
        self._backend.reconvene_network_updates(
            P_bar_t=self._P_bar_t,
            Y_bar_t=self._Y_bar_t,
            u_t=self._u_t
        )
        return norm_in_consensus(self._P_bar_t, self._Y_bar_t, 5e-4)
    
    @record_cpu_runtime('Update-X-EK-SUM')
    def _update_X_ek_sum(self):
        self._X_ek_sum_e = self._Z_e_start + len(self._commodity_list) * self._NULL_M @ self._Y_bar_t
    
    def _reset_u_t(self):
        self._u_t = cpu_zeros((self._NULL_M.shape[1],))
        self._backend.reset_inner_dual_variable()
    
    @record_cpu_runtime('Update-Re')
    def _update_r_e(self):
        assert self._model_controller is not None

        R_E = self._r_e
        Z_E = self._get_Z_value()
        X_EK_SUM_E = self._X_ek_sum_e
        self._r_e = R_E + (X_EK_SUM_E - Z_E)

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
            self._X_ek_start,
            self._M_MASK
        )
        self._backend.set_active_commodity_count(len(self._commodity_list))
    
    def reset(self, with_params: False):
        self._model_controller.reset()
        if with_params:
            self._model_controller.resetParams()
    
    def check_stopping_criterion(self):
        K = len(self.commodity_list)
        NUM_EDGES = self._NUM_EDGES
        T = self._T
        C_E = self._capacities
        Z_E = self._get_Z_value()
        V_MINUS = self._utility_bound_constraints[0].BarPi
        V_PLUS = self._utility_bound_constraints[1].BarPi
        TAU_E = np.array([const.BarPi for const in self._capacity_constraints])
        R_E = self._r_e
        UTILIZATION = self._utility.X
        ALPHA = self._alpha
        X_EK_START = self._X_ek_start
        X_EK_SUM_START = self._Z_e_start
        X_EK_SUM_E = self._X_ek_sum_e
        Y_TK = self._backend.get_Y_tk()
        LAMBDA_EK = self._backend.get_Lambda_ek()
        Y_BAR_T = self._Y_bar_t
        P_BAR_T = self._P_bar_t
        N = self._NULL_M

        primal_inf_1 = np.linalg.norm(X_EK_SUM_E - Z_E) / NUM_EDGES
        primal_inf_2 = np.linalg.norm(Y_BAR_T - P_BAR_T) / T
        # WHY IS IT MINUS????
        dual_inf_1 = np.dot(np.abs(Z_E), np.abs(-TAU_E + R_E)) / NUM_EDGES
        dual_inf_2 = np.sum(np.abs(np.array([np.dot(N @ Y_TK[:, k], LAMBDA_EK[:, k] + R_E) for k in range(K)]))) / (K * NUM_EDGES)
        dual_inf_3 = np.abs(ALPHA - V_MINUS + V_PLUS - np.dot(TAU_E, C_E))
        objective_gap = np.abs(V_PLUS + np.sum([np.dot(X_EK_START[:, k], LAMBDA_EK[:, k] + R_E) for k in range(K)]) - ALPHA * UTILIZATION) / (ALPHA * UTILIZATION)

        # print(f"PRIMAL INF I: {str(round(primal_inf_1, 4))}")
        # print(f"PRIMAL INF II: {str(round(primal_inf_2, 4))}")
        # print(f"DUAL INF I: {str(round(dual_inf_1, 4))}")
        # print(f"DUAL INF II: {str(round(dual_inf_2, 4))}")
        # print(f"DUAL INF III: {str(round(dual_inf_3, 4))}")
        print(f"OBJECTIVE GAP: {str(round(objective_gap, 4))} (EXPECTED: {(np.abs(UTILIZATION/0.5592 - 1))})")
    
    @record_cpu_runtime('Solve')
    def solve(self, params: Optional[int] = None) -> float:
        self.check_result = None
        MODEL_CONTROLLER = self._model_controller
        PARAMS = self._solver_params
        EPOCHS = params if params is not None else PARAMS.NumberOfEpochs
        SHIFT = 0 if params is None else PARAMS.NumberOfEpochs // 2

        try:
            t = time.time()
            optimize_or_scream(MODEL_CONTROLLER)
            self._update_r_e()
            for epoch in ShortTQDM(range(EPOCHS)):
            # for epoch in range(EPOCHS):
                # self._reset_u_t()
                for i in reversed(range(PARAMS.NumberOfNetworkUpdates)):
                    self._do_network_update(epoch + SHIFT)
                    if i > 0 and self._reconvene_network_updates():
                        break
                self._reconvene_network_updates()
                self._update_X_ek_sum()
                self._update_controller_objective()
                optimize_or_scream(MODEL_CONTROLLER)
                self._update_r_e()

                self._objective_trace.append(float(self._utility.X), float(get_solution_maximum_utilization(self._X_ek_sum_e, self._graph)))
                # self.check_stopping_criterion()
            self._set_X_ek()
            return time.time() - t
        except GurobiError as e:
            print(f'Error code {e.errno}: {e}')
            return -1
        except SolutionInterrupted:
            self._set_X_ek()
            return time.time() - t
        except asyncio.exceptions.CancelledError:
            return -1
        
        # try:
        #     t = time.time()
        #     for epoch in ShortTQDM(range(EPOCHS)):
        #     # for epoch in range(EPOCHS):
        #         # self._reset_u_t()
        #         for i in reversed(range(PARAMS.NumberOfNetworkUpdates)):
        #             self._do_network_update(epoch + SHIFT)
        #             if i > 0 and self._reconvene_network_updates():
        #                 break
        #             print(f'Inner distance: {str(round(np.linalg.norm(self._Y_bar_t - self._P_bar_t), 4))}')
        #         self._reconvene_network_updates()
        #         print(f'Inner distance: {str(round(np.linalg.norm(self._Y_bar_t - self._P_bar_t), 4))}')
        #         self._update_X_ek_sum()
        #         self._update_controller_objective()
        #         optimize_or_scream(MODEL_CONTROLLER)
        #         self._update_r_e()

        #         self._objective_trace.append(self._utility.X, get_solution_maximum_utilization(self._X_ek_sum_e, self._graph))
        #     self._set_X_ek()
        #     return time.time() - t
        # except GurobiError as e:
        #     print(f'Error code {e.errno}: {e}')
        #     return -1
        # except SolutionInterrupted:
        #     self._set_X_ek()
        #     return time.time() - t
        # except asyncio.exceptions.CancelledError:
        #     return -1
    
    def check(self, eval_params: TrafficEngineeringLPEvaluationParams):
        NUM_EDGES = self._NUM_EDGES

        # Are outer ADMM pairs in consensus?
        X_EK_SUM_E = self._X_ek_sum_e
        Z_E = self._get_Z_value()
        outer_admm_consensus_test(X_EK_SUM_E, Z_E, eval_params=eval_params)
        
        # Are inner ADMM pairs in consensus?
        Y_BAR_T = self._Y_bar_t
        P_BAR_T = self._P_bar_t
        inner_admm_consensus_test(Y_BAR_T, P_BAR_T, eval_params=eval_params)
        
        # Now, check flow conservation ...
        X_EK = self._X_ek
        unsat_ratio, unsat_commodities = check_flow_conservation(
            X_EK, self._graph, self._commodity_list, eval_params=eval_params)
        congested_ratio, congested_links = check_capacity_constraint(
            X_EK, self._graph, self._commodity_list, eval_params=eval_params)
        self.check_result = TrafficEngineeringLPCheckResult(
            unsat_ratio=unsat_ratio,
            congested_ratio=congested_ratio,
            unsat_commodities=unsat_commodities,
            congested_links=congested_links
        )

    def get_solution_commodity_list(self) -> List[Tuple[Commodity, Commodity]]:
        assert self._X_ek is not None

        COMMODITIES = self._commodity_list
        GRAPH = self._graph
        X = self._X_ek

        ls = []
        for k, commodity in enumerate(COMMODITIES):
            flow_out = defaultdict(list)
            flow_in = defaultdict(list)
            for e, edge in enumerate(GRAPH.edges()):
                flow_out[edge[0]].append(X[e, k])
                flow_in[edge[1]].append(X[e, k])
            commodity_sent = Commodity(
                source=commodity.source, destination=commodity.destination,
                demand=sum(flow_out[commodity.source])
            )
            commodity_received = Commodity(
                source=commodity.source, destination=commodity.destination,
                demand=sum(flow_in[commodity.destination])
            )
            ls.append((commodity_sent, commodity_received))
        return ls
    
    def update_traffic_matrix(self, tm):
        # First, record the matrix and the new commodities
        self._traffic = tm
        self._commodity_list = traffic_to_commodity(self._traffic)
        # Get a new feasible solution (if the matrix did not change too much),
        # then this also will not change too much.
        self._set_initial_feasible_solution()
        # Send it to the backend
        self._backend.update_demands(self._X_ek_start)
    
    def initialize_to(self, solution: EdgeBasedMinimizeMaximumUtilitySolution):
        raise NotImplementedError
    
    def set_target(self, solution: EdgeBasedMinimizeMaximumUtilitySolution):
        raise NotImplementedError
    
    def add_solution_elements(self, solution: EdgeBasedMinimizeMaximumUtilitySolution):
        solution.add_solution_element(self._utility, name='utility')
        solution.add_solution_element(self._X_ek, name='assignments')
