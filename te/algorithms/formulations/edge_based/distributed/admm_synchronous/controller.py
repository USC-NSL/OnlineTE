import time
import signal
import numpy as np
import networkx as nx
import asyncio.exceptions
from collections import defaultdict
from typing import List, Tuple, Optional
from ..base import ControllerNodeBase
from te.algorithms.base import (SolverParams, TrafficEngineeringLPCheckResult, 
                                TrafficEngineeringLPEvaluationParams, TrafficEngineeringLPObjectiveTrace)
from te.algorithms.solution import EdgeBasedMinimizeMaximumUtilitySolution
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, Commodity
from topologies.utils import get_graph_M_matrix, get_adjacency_null_space, get_commodity_in_out_mask
from topologies.utils import get_symbolic_graph_M_matrix
from utils.exceptions import SolutionInterrupted
from utils.logging import as_info, as_warning, log_subsection_separator, ShortTQDM
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.cpu_utils import (CPUArray, BooleanCPUArray,
                                                 cpu_array, cpu_zeros, cpu_double_array, 
                                                 set_cpu_float_precision)
from te.algorithms.utils import get_solution_maximum_utilization
from te.algorithms.sub_algorithms.feasible_assignment import get_feasible_flow_assignment
from te.algorithms.sub_algorithms.admm_consensus_test import outer_admm_consensus_test, inner_admm_consensus_test, norm_in_consensus
from te.algorithms.sub_algorithms.link_capacity_test import check_capacity_constraint
from te.algorithms.sub_algorithms.flow_conservation_test import check_flow_conservation
from te.algorithms.statistics.helpers import record_cpu_runtime, record_return_value
from . import SynchADMMSolverParams
from .. import ControllerRPCParams
from .base import SynchADMMControllerBackendBase
from ..base import ControllerNodeParams
from te.algorithms.sub_algorithms.mlu_backends.base import ControllerMLUSolver, ControllerMLUException


class SynchADMMControllerNode(ControllerNodeBase):
    def __init__(self, params: ControllerNodeParams) -> None:
        self._graph = params.graph
        self._M = get_graph_M_matrix(params.graph)
        self._symbolic_M = get_symbolic_graph_M_matrix(params.graph)
        self._traffic = params.traffic
        self._solver_params = params.solver_params
        self._rpc_params = params.rpc_params
        self._rng = np.random.default_rng(seed=params.solver_params.TMSeed)
        self._commodity_list = traffic_to_commodity(self._traffic)

        self._NULL_M: CPUArray = None
        self._NNT_M: CPUArray = None
        
        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None
        self._M_MASK: Optional[BooleanCPUArray] = None

        self._capacities: Optional[CPUArray] = None
        self._c_norm: Optional[float] = None
        self._alpha: Optional[float] = None

        self._mlu_solver_cls: type[ControllerMLUSolver] = params.mlu_backend
        self._mlu_params: SolverParams = params.mlu_params
        self._mlu_solver: Optional[ControllerMLUSolver] = None

        self._X_ek: Optional[CPUArray] = None
        self._X_ek_start: Optional[CPUArray] = None
        self._Z_e_start: Optional[CPUArray] = None
        self._X_ek_sum_e: Optional[CPUArray] = None
        self._r_e: Optional[CPUArray] = None

        self._P_bar_t: Optional[CPUArray] = None
        self._Y_bar_t: Optional[CPUArray] = None
        self._u_t: Optional[CPUArray] = None

        self._backend: SynchADMMControllerBackendBase = params.communication_backend(params.rpc_params)
        self._backend.start()

        self._objective_trace: TrafficEngineeringLPObjectiveTrace = \
            TrafficEngineeringLPObjectiveTrace(['Perceived Utilization', 'Actual Utilization'])
        self._objective_gap_trace = []

        # These we call right now, as opposed to doing them under `initialize`
        set_global_precision(params.solver_params.Precision)
        set_cpu_float_precision()

        self._die_on_next_int = False
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.die)
    
    def initialize(self):
        # First, set the initial feasible solutions.
        # We will do this before spawning the backend, since if we use `gRPC`, 
        # this function may invoke `fork` which causes `gRPC` to spam warnings.
        self._set_initial_feasible_solution()
        # Initialize the algorithm
        self._set_NULL_M()
        self._initialize_variables_and_residuals()
        # Report what we are dealing with
        self._report_problem_size()
    
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
        return 'Distributed Synchronous ADMM'

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
        return self._mlu_solver.current_u
    
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
    
    def _set_NULL_M(self):
        M = self._M
        assert len(M.shape) == 2
        m, n = M.shape
        assert m < n
        N = cpu_array(get_adjacency_null_space(M))
        T = N.shape[1]
        self._NULL_M = N
        self._NNT_M = N @ N.T
        self._T = T
        self._NUM_EDGES = n
        self._M_MASK = get_commodity_in_out_mask(self.graph, self.commodity_list)
    
    def _get_Z_value(self) -> CPUArray:
        return self._mlu_solver.current_Z
    
    def _initialize_variables_and_residuals(self):
        T = self._T
        NUM_EDGES = self._NUM_EDGES
        self._capacities = cpu_double_array([item[-1] for item in self._graph.edges(data='capacity')])
        self._c_norm = np.linalg.norm(self._capacities)
        self._alpha = self._c_norm * np.sqrt(NUM_EDGES)
        self._r_e = cpu_zeros((NUM_EDGES,))
        self._u_t = cpu_zeros((T,))
        self._P_bar_t = cpu_zeros((T,))
        self._Y_bar_t = cpu_zeros((T,))
        self._X_ek = cpu_array(self._X_ek_start)
        self._X_ek_sum_e = cpu_array(self._Z_e_start)
        # The objective convergance tolerance for the MLU problem _MUST_ stricter than the
        # tolerance for the distributed algorithm itself.
        assert self._solver_params.ConvTol >= self._mlu_params.ConvTol, \
            f"{self._solver_params.ConvTol} < {self._mlu_params.ConvTol}"
        # TODO: Find a better way to handle `Rho` and `Alpha` here ...
        self._mlu_params._Rho = self._solver_params.Rho
        self._mlu_params._Alpha = self._alpha
        self._mlu_solver = self._mlu_solver_cls(NUM_EDGES, self._capacities, self._mlu_params)
    
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
        assert self._mlu_solver is not None
        self._mlu_solver._make_variables()
        
    
    def _get_F(self) -> np.ndarray:
        return self._get_Z_value() - self._Z_e_start - self._r_e
    
    def _set_X_ek(self):
        self._X_ek = self._backend.get_X_ek(basis=self._NULL_M, initial_feasible_solution=self._X_ek_start)
    
    def _add_constraints(self):
        assert self._mlu_solver is not None
        self._mlu_solver._add_constraints()
    
    @record_cpu_runtime('Controller-Update')
    def _update_controller_objective(self):
        assert self._mlu_solver is not None
        X_EK_SUM_E = self._X_ek_sum_e
        R_E = self._r_e
        self._mlu_solver.update_F_m(X_EK_SUM_E + R_E)
    
    def _add_objective(self):
        assert self._mlu_solver is not None
        self._mlu_solver._add_objective()

    @record_return_value('PGD-Runtime')
    @record_cpu_runtime('Network-Update')
    def _do_network_update(self, epoch: int):
        max_run, self._Y_bar_t = self._backend.do_network_update(epoch)
        return max_run
    
    def _update_P_bar(self):
        assert self._mlu_solver is not None

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
        assert self._mlu_solver is not None

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
    
    @record_cpu_runtime('Update-Re')
    def _update_r_e(self):
        assert self._mlu_solver is not None

        R_E = self._r_e
        Z_E = self._get_Z_value()
        X_EK_SUM_E = self._X_ek_sum_e
        self._r_e = R_E + (X_EK_SUM_E - Z_E)

    def close(self):
        self._backend.close()
        if self._mlu_solver is not None:
            self._mlu_solver.close()
    
    def make_lp(self):
        self.initialize()
        t_start = time.time()
        self._make_variables()
        self._add_constraints()
        self._add_objective()
        print(as_info(f"Built model in {str(np.round(time.time() - t_start, 2))} seconds."))
        self._backend.initialize_worker_nodes(
            self._solver_params,
            self._NULL_M, 
            self._X_ek_start,
            self._M_MASK
        )
        self._backend.set_active_commodity_count(len(self._commodity_list))
    
    def reset(self, with_params: False):
        self._mlu_solver.reset(with_params)
    
    @record_cpu_runtime('Solve')
    def solve(self, params: Optional[int] = None) -> float:
        self.check_result = None
        MODEL_CONTROLLER = self._mlu_solver
        PARAMS = self._solver_params
        EPOCHS = params if params is not None else PARAMS.NumberOfEpochs
        SHIFT = 0 if params is None else PARAMS.NumberOfEpochs // 2

        try:
            t = time.time()
            self._update_controller_objective()
            MODEL_CONTROLLER.solve()
            self._update_r_e()
            for epoch in ShortTQDM(range(EPOCHS)):
                for i in reversed(range(PARAMS.NumberOfNetworkUpdates)):
                    self._do_network_update(epoch + SHIFT)
                    if i > 0 and self._reconvene_network_updates():
                        break
                self._reconvene_network_updates()
                self._update_X_ek_sum()
                self._update_controller_objective()
                MODEL_CONTROLLER.solve()
                self._update_r_e()

                self._objective_trace.append(
                    float(self.objective_value), 
                    float(get_solution_maximum_utilization(self._X_ek_sum_e, self._graph))
                )
            self._set_X_ek()
            return time.time() - t
        except ControllerMLUException as e:
            print(f'MLU solver failed: {e}')
            return -1
        except SolutionInterrupted:
            self._set_X_ek()
            return time.time() - t
        except asyncio.exceptions.CancelledError:
            return -1

    def check(self, eval_params: TrafficEngineeringLPEvaluationParams):
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
        solution.add_solution_element(self.objective_value, name='utility')
        solution.add_solution_element(self._X_ek, name='assignments')
