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
from topologies.utils import get_graph_M_matrix
from utils.exceptions import SolutionInterrupted
from utils.logging import as_info, as_warning, log_subsection_separator, ShortTQDM
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.cpu_utils import (CPUArray, DoublePrecisionCPUArray, 
                                                 cpu_array, cpu_zeros, cpu_double_array, 
                                                 set_cpu_float_precision)
from te.algorithms.utils import optimize_or_scream, make_model, get_solution_maximum_utilization
from te.algorithms.sub_algorithms.admm_consensus_test import outer_admm_consensus_test, inner_admm_consensus_test, norm_in_consensus
from te.algorithms.sub_algorithms.link_capacity_test import check_capacity_constraint
from te.algorithms.sub_algorithms.flow_conservation_test import check_flow_conservation
from te.algorithms.sub_algorithms.paths import TShortestPaths
from te.algorithms.statistics.helpers import record_cpu_runtime, record_return_value
from . import PathBasedDistributedADMMSolverParams, PathBasedDistributedADMMControllerRPCParams
from .controller_backends import get_backend, ControllerCommunicationBackendBase


class ControllerNode(TrafficEngineeringLP):
    def __init__(self, graph: nx.DiGraph, traffic: TrafficMatrixBase, 
                 solver_params: PathBasedDistributedADMMSolverParams,
                 rpc_params: PathBasedDistributedADMMControllerRPCParams) -> None:
        super().__init__()
        self._graph = graph
        self._M = get_graph_M_matrix(graph)
        self._traffic = traffic
        self._solver_params = solver_params
        self._rpc_params = rpc_params
        self._rng = np.random.default_rng(seed=solver_params.Seed)
        self._commodity_list = traffic_to_commodity(self._traffic)

        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None

        self._env: gurobipy.Env = None

        self._model_controller: Optional[gurobipy.Model] = None
        self._objective_controller: Optional[gurobipy.QuadExpr] = None

        self._capacities: Optional[DoublePrecisionCPUArray] = None
        self._c_norm: Optional[float] = None

        # Path configurations
        self._path_object: Optional[TShortestPaths] = None
        # Real-time demands
        # In an online setting, these will be moved completely into the worker nodes
        self._D_k: Optional[CPUArray] = None

        self._X_ek: Optional[CPUArray] = None
        self._Z_e: Optional[gurobipy.tupledict] = None
        self._utility: Optional[gurobipy.Var] = None
        self._r_e: Optional[CPUArray] = None

        self._utility_bound_constraints: Tuple[gurobipy.Constr, gurobipy.Constr] = None
        """Gives the dual variables `v_neg` and `v_pos`"""
        self._capacity_constraints: List[gurobipy.Constr] = None
        """Gives the dual variables `tau_e`, a vector of length `n`"""

        self._P_bar_e: Optional[CPUArray] = None
        self._X_bar_e: Optional[CPUArray] = None
        self._u_e: Optional[CPUArray] = None

        self._backend: Optional[ControllerCommunicationBackendBase] = None

        self._objective_trace: TrafficEngineeringLPObjectiveTrace = TrafficEngineeringLPObjectiveTrace(['Perceived Utilization', 'Actual Utilization'])
        self._objective_gap_trace = []

        # These we call right now, as opposed to doing them under `initialize`
        set_global_precision(solver_params.Precision)
        set_cpu_float_precision()

        self._die_on_next_int = False
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.die)
    
    @record_cpu_runtime('Initialization')
    def initialize(self):
        # First, load all path configurations
        self._path_object = TShortestPaths.load(graph=self._graph, T=self._solver_params.NumberOfPathsPerCommodity,
                                                topo_name=self._solver_params.TopologyName)
        # Set the demands
        # TODO: Fix this, it should invoke CPU_ARRAY!
        self._D_k: np.ndarray = np.array([commodity.demand for commodity in self._commodity_list])
        # Now, create the backend
        self._backend = get_backend(self._rpc_params)
        # Initialize the algorithm
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
        return 'Path Based Distributed Unregulated ADMM'

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
    
    def _get_Z_value(self) -> CPUArray:
        try:
            return cpu_array([self._Z_e[e].X for e in range(self._NUM_EDGES)])
        except AttributeError:
            return cpu_zeros((self._NUM_EDGES,))
    
    def _initialize_variables_and_residuals(self):
        NUM_EDGES = self.graph.number_of_edges()
        self._NUM_EDGES = NUM_EDGES
        self._capacities = cpu_double_array([item[-1] for item in self._graph.edges(data='capacity')])
        self._c_norm = np.linalg.norm(self._capacities)
        self._r_e = cpu_zeros((NUM_EDGES,))
        self._u_e = cpu_zeros((NUM_EDGES,))
        self._X_bar_e = cpu_zeros((NUM_EDGES,))
        self._P_bar_e = cpu_zeros((NUM_EDGES,))
        self._X_ek_sum_e = cpu_zeros((NUM_EDGES,))
    
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
        # TODO: Get back `BigGamma` for the solver parameters ...
        MODEL_CONTROLLER: gurobipy.Model = \
            make_model('PathBasedDistributedTE_Controller', params=PARAMS, env=ENV, BarConvTol=1e-6)
        
        # self._Z_e = MODEL_CONTROLLER.addVars(NUM_EDGES, lb=0.0, vtype=GRB.CONTINUOUS, name='Z_E')
        self._Z_e = MODEL_CONTROLLER.addVars(NUM_EDGES, lb=float('-inf'), vtype=GRB.CONTINUOUS, name='Z_E')
        # self._utility = MODEL_CONTROLLER.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='U')
        self._utility = MODEL_CONTROLLER.addVar(lb=float('-inf'), vtype=GRB.CONTINUOUS, name='U')

        self._model_controller = MODEL_CONTROLLER
    
    def _set_X_ek(self):
        self._X_ek = self._backend.get_X_ek(alpha=self._path_object.alpha, demands=self._D_k)
    
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

        capacity_constraints: List[gurobipy.Constr] = []
        for e, (_, _, c_e) in enumerate(GRAPH.edges.data('capacity')):
            capacity_constraints.append(MODEL_CONTROLLER.addConstr(UTILITY * c_e >= Z_E[e]))
        self._capacity_constraints = capacity_constraints
    
    def _get_X_ek_sum_e(self) -> CPUArray:
        return len(self._commodity_list) * self._X_bar_e
    
    @record_cpu_runtime('Controller-Update')
    def _update_controller_objective(self):
        NUM_EDGES = self._NUM_EDGES
        UTILITY = self._utility
        Z_E = self._Z_e
        X_EK_SUM_E = self._get_X_ek_sum_e()
        R_E = self._r_e
        RHO = self._solver_params.Rho
        MODEL_CONTROLLER = self._model_controller
        BIG_GAMMA = self._c_norm * np.sqrt(NUM_EDGES)
        
        OBJECTIVE_CONTROLLER = gurobipy.QuadExpr()
        OBJECTIVE_CONTROLLER.addTerms(BIG_GAMMA, UTILITY)
        for e in range(NUM_EDGES):
            OBJECTIVE_CONTROLLER += (RHO/2) * (X_EK_SUM_E[e] - Z_E[e] + R_E[e]) ** 2
        MODEL_CONTROLLER.setObjective(OBJECTIVE_CONTROLLER, GRB.MINIMIZE)
        self._objective_controller = OBJECTIVE_CONTROLLER
    
    def _add_objective(self):
        assert self._model_controller is not None

        self._update_controller_objective()

    @record_return_value('QP-Runtime')
    @record_cpu_runtime('Network-Update')
    def _do_network_update(self, epoch: int):
        max_run, self._X_bar_e = self._backend.do_network_update(epoch)
        return max_run
    
    def _update_P_bar(self):
        assert self._model_controller is not None

        K = len(self._commodity_list)
        ETA = self._solver_params.Eta
        RHO = self._solver_params.Rho
        R_E = self._r_e
        U_E = self._u_e
        X_BAR_E = self._X_bar_e
        Z_E = self._get_Z_value()
        P_BAR_E = (Z_E - R_E + (ETA/RHO) * (X_BAR_E + U_E)) / (K + (ETA/RHO))
        self._P_bar_e = P_BAR_E
    
    def _update_u_e(self):
        assert self._model_controller is not None

        U_E = self._u_e
        X_BAR_E = self._X_bar_e
        P_BAR_E = self._P_bar_e

        self._u_e = U_E + (X_BAR_E - P_BAR_E)
    
    @record_cpu_runtime('Update-Reconvene')
    def _reconvene_network_updates(self) -> bool:
        self._update_P_bar()
        self._update_u_e()
        self._backend.reconvene_network_updates(
            P_bar_e=self._P_bar_e,
            X_bar_e=self._X_bar_e,
            u_e=self._u_e
        )
        return norm_in_consensus(self._X_bar_e, self._P_bar_e, 5e-4)
    
    @record_cpu_runtime('Update-Re')
    def _update_r_e(self):
        assert self._model_controller is not None

        R_E = self._r_e
        Z_E = self._get_Z_value()
        X_EK_SUM_E = self._get_X_ek_sum_e()
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
            self._path_object.alpha,
            self._path_object.beta,
            self._D_k
        )
        self._backend.set_active_commodity_count(len(self._commodity_list))
    
    def reset(self, with_params: False):
        self._model_controller.reset()
        if with_params:
            self._model_controller.resetParams()
    
    def check_stopping_criterion(self):
        raise NotImplementedError
    
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
                for i in reversed(range(PARAMS.NumberOfNetworkUpdates)):
                    self._do_network_update(epoch + SHIFT)
                    if i > 0 and self._reconvene_network_updates():
                        break
                self._reconvene_network_updates()
                self._update_controller_objective()
                optimize_or_scream(MODEL_CONTROLLER)
                self._update_r_e()

                self._objective_trace.append(
                    self._utility.X, 
                    get_solution_maximum_utilization(self._get_X_ek_sum_e(), self._graph)
                )
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

    def check(self, eval_params: TrafficEngineeringLPEvaluationParams):
        # Are outer ADMM pairs in consensus?
        X_EK_SUM_E = self._get_X_ek_sum_e()
        Z_E = self._get_Z_value()
        outer_admm_consensus_test(X_EK_SUM_E, Z_E, eval_params=eval_params)
        
        # Are inner ADMM pairs in consensus?
        X_BAR_E = self._X_bar_e
        P_BAR_E = self._P_bar_e
        inner_admm_consensus_test(X_BAR_E, P_BAR_E, eval_params=eval_params)
        
        # Now, check flow conservation ...
        X_EK = self.assignments
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
        raise NotImplementedError
    
    def initialize_to(self, solution: EdgeBasedMinimizeMaximumUtilitySolution):
        raise NotImplementedError
    
    def set_target(self, solution: EdgeBasedMinimizeMaximumUtilitySolution):
        raise NotImplementedError
    
    def add_solution_elements(self, solution: EdgeBasedMinimizeMaximumUtilitySolution):
        raise NotImplementedError
