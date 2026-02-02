import time
import queue
import numpy as np
import networkx as nx
import asyncio.exceptions
from collections import defaultdict
from typing import List, Tuple, Optional, Set
from ..base import DistributedSolverNodeParams, DistributedSolverNodeBase
from te.algorithms.base import *
from te.algorithms.solution import EdgeBasedMinimizeMaximumUtilitySolution
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, Commodity
from topologies.utils import get_graph_M_matrix, get_adjacency_null_space, get_commodity_in_out_mask
from topologies.utils import get_symbolic_graph_M_matrix
from utils.exceptions import SolutionInterrupted
from utils.logging import as_info, as_success, log_subsection_separator, ShortTQDM
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.cpu_utils import (CPUArray, BooleanCPUArray,
                                                 cpu_array, cpu_zeros, cpu_double_array, 
                                                 set_cpu_float_precision)
from te.algorithms.utils import get_solution_maximum_utilization
from te.algorithms.sub_algorithms.feasible_assignment import get_feasible_flow_assignment
from te.algorithms.sub_algorithms.admm_consensus_test import outer_admm_consensus_test, inner_admm_consensus_test
from te.algorithms.sub_algorithms.link_capacity_test import check_capacity_constraint
from te.algorithms.sub_algorithms.flow_conservation_test import check_flow_conservation
from te.algorithms.statistics.helpers import record_cpu_runtime, record_return_value
from . import HierarchicalADMMSolverParams
from .base import MasterCommunicationBackendBase
from te.algorithms.sub_algorithms.mlu_backends.base import ControllerMLUSolver, ControllerMLUException


class MasterNode(TrafficEngineeringLP, DistributedSolverNodeBase):
    def __init__(self, params: DistributedSolverNodeParams, mlu_cls: ControllerMLUSolver, 
                 mlu_params: SolverParams, partitions: List[Tuple[int, int]]) -> None:
        super().__init__(
            problem_description=params.ProblemDescription,
            solver_params=params.SolverParams_,
            node_params=params
        )
        self._graph = params.ProblemDescription.Graph
        self._M = get_graph_M_matrix(self._graph)
        self._symbolic_M = get_symbolic_graph_M_matrix(self._graph)
        self._traffic = params.ProblemDescription.TM
        self._solver_params: HierarchicalADMMSolverParams = params.SolverParams_
        self._rpc_params = params.RPCParams_
        self._rng = np.random.default_rng(seed=self._solver_params.TMSeed)
        self._commodity_list = traffic_to_commodity(self._traffic)

        self._NULL_M: CPUArray = None
        self._NNT_M: CPUArray = None
        
        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None
        self._M_MASK: Optional[BooleanCPUArray] = None

        self._capacities: Optional[CPUArray] = None
        self._c_norm: Optional[float] = None
        self._alpha: Optional[float] = None

        self._mlu_solver_cls: type[ControllerMLUSolver] = mlu_cls
        self._mlu_params: SolverParams = mlu_params
        self._mlu_solver: Optional[ControllerMLUSolver] = None

        self._domain_partitions: List[Tuple[int, int]] = partitions

        self._X_ek: Optional[CPUArray] = None
        self._X_dek_start: Optional[List[CPUArray]] = None
        self._Z_de_start: Optional[CPUArray] = None
        self._X_dek_sum_de: Optional[CPUArray] = None
        self._Z_de: Optional[CPUArray] = None
        self._r_de: Optional[CPUArray] = None

        self._domain_updates_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._arrived_domains: Set[int] = set()
        self._arrived_updates: List[Tuple[int, CPUArray]] = list()
        self._timer: List[int] = [0 for _ in range(self._solver_params.NumberOfDomains)]
        self._clock: int = 0

        self.backend: MasterCommunicationBackendBase = params.CommunicationBackendCLS(params.RPCParams_)
        self.backend.start()

        self._objective_trace: TrafficEngineeringLPObjectiveTrace = \
            TrafficEngineeringLPObjectiveTrace(['Perceived Utilization', 'Actual Utilization'])
        self._objective_gap_trace = []

        set_global_precision(self._solver_params.Precision)
        set_cpu_float_precision()

    def can_update(self) -> bool:
        return len(self._arrived_domains) >= self._solver_params.MasterUpdateBarrierSize and \
            max(self._timer) < self._solver_params.MasterUpdateMaxLag

    def increment_timer(self):
        for i in set(range(self._solver_params.NumberOfDomains)).difference(self._arrived_domains):
            self._timer[i] += 1
    
    def increment_clock(self):
        self._clock += 1
    
    def initialize(self):
        print(as_info("Waiting for peers to become reachable"))
        while self.backend.is_alive and not self.backend.are_all_peers_reachable():
            time.sleep(1)
        if not self.backend.is_alive:
            raise SolutionInterrupted
        print(as_success("All peer nodes are reachable"))
        set_global_precision(self._solver_params.Precision)
        set_cpu_float_precision()
        self._set_initial_feasible_solution()
        self._set_NULL_M()
        self._initialize_variables_and_residuals()
        self._report_problem_size()
        self.backend.initialize_domain_peers(
            self._solver_params, 
            self._NULL_M, 
            self._X_dek_start, 
            self._M_MASK
        )

    def run(self):
        self.solve()

    @property
    def alg_name(self) -> str:
        return 'Edge-Based Hierarchical'

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
        X_EK_START = get_feasible_flow_assignment(self._graph, self._commodity_list)
        self._X_dek_start = [
            cpu_array(X_EK_START[:, item[0]:item[1]]) for item in self._domain_partitions
        ]
        self._Z_de_start = cpu_array([
            np.sum(self._X_dek_start[d, :, :], axis=1) for d in range(self._solver_params.NumberOfDomains)
        ])
    
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
    
    def _initialize_variables_and_residuals(self):
        NUM_EDGES = self._NUM_EDGES
        self._capacities = cpu_double_array([item[-1] for item in self._graph.edges(data='capacity')])
        self._c_norm = np.linalg.norm(self._capacities)
        self._alpha = self._c_norm * np.sqrt(NUM_EDGES)
        self._r_de = cpu_zeros((self._solver_params.NumberOfDomains, NUM_EDGES))
        self._X_dek_sum_de = cpu_array(self._Z_de_start)
        self._Z_de = cpu_array(self._Z_de_start)
        # The objective convergance tolerance for the MLU problem _MUST_ be stricter than the
        # tolerance for the distributed algorithm itself.
        assert self._solver_params.ConvTol >= self._mlu_params.ConvTol, \
            f"{self._solver_params.ConvTol} < {self._mlu_params.ConvTol}"
        # TODO: Find a better way to handle `Rho` and `Alpha` here ...
        self._mlu_params._Rho = self._solver_params.Rho
        self._mlu_params._Alpha = self._alpha
        self._mlu_solver = self._mlu_solver_cls(NUM_EDGES, self._capacities, self._mlu_params, self._solver_params.NumberOfDomains)

    def are_peer_network_nodes_ready(self):
        return self.backend.are_all_peers_reachable()
    
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
    
    def _set_X_ek(self):
        self._X_ek = self.backend.collect_X_ek()
    
    def _add_constraints(self):
        assert self._mlu_solver is not None
        self._mlu_solver._add_constraints()
    
    @record_cpu_runtime('Master-Update')
    def _update_controller_objective(self):
        assert self._mlu_solver is not None
        assert len(self._arrived_updates) > 0
        X_DEK_SUM_DE = self._X_dek_sum_de
        R_DE = self._r_de
        for domain_id, domain_aggregate in self._arrived_updates:
            X_DEK_SUM_DE[domain_id, :] = domain_aggregate
        self._mlu_solver.update_F_m(X_DEK_SUM_DE + R_DE)
    
    def _add_objective(self):
        assert self._mlu_solver is not None
        self._mlu_solver._add_objective()

    def close(self):
        self.backend.close_domains()
        self.backend.close()
        if self._mlu_solver is not None:
            self._mlu_solver.close()
    
    def make_lp(self):
        self.initialize()
        t_start = time.time()
        self._make_variables()
        self._add_constraints()
        self._add_objective()
        print(as_info(f"Built model in {str(np.round(time.time() - t_start, 2))} seconds."))
        self.backend.initialize_domain_peers(
            self._solver_params,
            self._NULL_M, 
            self._X_dek_sum_de,
            self._M_MASK
        )
    
    def reset(self, with_params: False):
        self._mlu_solver.reset(with_params)

    def notify_domains_and_reset(self):
        self.backend.notify_arrived_peers(self._arrived_updates, self._Z_de)
        self._arrived_updates.clear()
        self._arrived_domains.clear()

    def wait_until_update(self) -> bool:
        while self.backend.is_alive:
            try:
                item: Tuple[int, CPUArray] = self._domain_updates_queue.get(timeout=1.0)
                worker_id: int = item[0]
                self._arrived_domains.add(worker_id)
                self._arrived_updates.append(item)
                self._timer[worker_id] = 0
            except queue.Empty:
                pass
            finally:
                if self.can_update():
                    return True
        return False
    
    def record_objective(self):
        self._objective_trace.append(
            float(self.objective_value), 
            float(get_solution_maximum_utilization(np.sum(self._X_dek_sum_de, axis=0), self._graph))
        )
    
    @record_cpu_runtime('Solve')
    def solve(self, params: Optional[int] = None) -> float:
        self.check_result = None
        MODEL_CONTROLLER = self._mlu_solver
        PARAMS = self._solver_params
        EPOCHS = params if params is not None else PARAMS.NumberOfMasterUpdates

        try:
            t = time.time()
            for _ in ShortTQDM(range(EPOCHS)):
                if not self.wait_until_update():
                    return -1
                self.increment_timer()
                self._update_controller_objective()
                MODEL_CONTROLLER.solve()
                self._Z_de = MODEL_CONTROLLER.current_Z
                self.notify_domains_and_reset()
                self.record_objective()
                self.increment_clock()
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
        X_EK_SUM_E = np.sum(self._X_dek_sum_de, axis=0)
        Z_E = np.sum(self._Z_de, axis=0)
        outer_admm_consensus_test(X_EK_SUM_E, Z_E, eval_params=eval_params)
        
        # Are inner ADMM pairs in consensus?
        Y_BAR_T, P_BAR_T = self.backend.get_admm_consensus_variables()
        inner_admm_consensus_test(Y_BAR_T, P_BAR_T, eval_params=eval_params)
        
        # Now, check flow conservation ...
        X_EK = self._X_ek
        unsat_ratio, unsat_commodities, total_satisfcation = check_flow_conservation(
            X_EK, self._graph, self._commodity_list, eval_params=eval_params)
        congested_ratio, congested_links = check_capacity_constraint(
            X_EK, self._graph, self._commodity_list, eval_params=eval_params)
        self.check_result = TrafficEngineeringLPCheckResult(
            unsat_ratio=unsat_ratio,
            congested_ratio=congested_ratio,
            unsat_commodities=unsat_commodities,
            congested_links=congested_links,
            total_satisfcation=total_satisfcation
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
        solution.add_solution_element(self.objective_value, name='utility')
        solution.add_solution_element(self._X_ek, name='assignments')


# if __name__ == '__main__':
#     import argparse
#     from typing import List
#     from .. import DEFAULT_RPC_PORT
#     from .master_backends.asynchronous_grpc_backend import AsynchronousgRPCMasterBackend, AsynchronousgRPCMasterBackendParams
#     from te.algorithms.sub_algorithms.mlu_backends.aggregate import add_mlu_backend_subparser, parse_mlu_backend_params

#     parser = argparse.ArgumentParser('Spawn A Master Controller')
#     parser.add_argument('--partitions', action='append', type=int, help='Number of domain workers for each domain', required=True)
#     parser.add_argument('--peers', nargs='+', help='List of peer addresses (master and other domains)')
#     parser.add_argument('--local', action='store_true', help='Assume everything is run locally')

#     add_mlu_backend_subparser(parser)
#     MLU_BACKEND_PARAMS, MLUCLS, args = parse_mlu_backend_params(parser=parser)

#     def addr_str_to_tuple(addr_str: str) -> Tuple[str, int]:
#         ls = addr_str.split(':')
#         assert len(ls) == 2
#         return ls[0], int(ls[1])

#     PARTITIONS = args.partitions
#     num_domains = len(PARTITIONS)
#     peers: Optional[List[str]] = args.peers
#     if peers is not None:
#         num_peers = len(peers)
#         assert num_peers == num_domains+1
#     else:
#         num_peers = num_domains+1

#     if args.local:
#         PEER_ADDRS = tuple([('localhost', DEFAULT_RPC_PORT+i) for i in range(num_peers)])
#     else:
#         if peers is None:
#             PEER_ADDRS = tuple([('controller', DEFAULT_RPC_PORT)] + [(f'd{i}', DEFAULT_RPC_PORT) for i in range(num_domains)])
#         else:
#             PEER_ADDRS = tuple(map(addr_str_to_tuple, peers))
#     rpc_params = AsynchronousgRPCMasterBackendParams(Index=0, Peers=PEER_ADDRS)
#     rpc_cls = AsynchronousgRPCMasterBackend
#     print(f'RPC Parameters:\n{rpc_params.str_all()}')
#     MasterNode.spawn_and_run(P2PNodeParams(
#         mlu_backend=MLUCLS, mlu_params=MLU_BACKEND_PARAMS,
#         communication_backend=rpc_cls,
#         rpc_params=rpc_params
#     ), PARTITIONS)
