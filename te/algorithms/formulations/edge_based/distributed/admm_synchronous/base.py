import networkx as nx
from abc import abstractmethod
from typing import Tuple, Callable
from ..base import CommunicationBackendBase
from array_utils.cpu.types import *
from te.algorithms.base import SolverParams


class SynchADMMControllerBackendBase(CommunicationBackendBase):
    # For the synchronous case, there is only one controller!
    def are_all_peers_reachable(self):
        return True
    
    @abstractmethod
    def initialize_worker_nodes(
        self,
        solver_params: SolverParams,
        topology: nx.DiGraph
    ):
        """
        The nodes need to know a few thing before they can start.
        In particular, they need the full solver parameters and the
        graph before they can start.
        The worker nodes are expected to only answer this RPC when
        all work on their side is finished. This at least includes:
        - Reading the adjacency matrix from the graph and taking the
          SVD to find the null-space basis.
        - Reading the commodity mask for loop-avoidance on endpoints.
        - Caching the pseudo-inverse basis of the adjacency matrix
          to quickly find feasible solutions on demand updates (see
          the note on `get_feasible_assignment` for why we have to
          do this).
        """

    @abstractmethod
    def update_demands(self, demands: CPUArray) -> CPUArray:
        """
        Given new demands, update all nodes so that we can restart a solve.
        Must return the new sharing mean (the mean of the assignments) after
        the update is done.
        """
    
    @abstractmethod
    def get_X_ek(self) -> CPUArray:
        """Get the final solution array (X_ek)"""
    
    @abstractmethod
    def get_X_ek_sum(self) -> CPUArray:
        """Get the total flow over each edge"""
    
    @abstractmethod
    def do_network_update(self, epoch: int) -> Tuple[int, CPUArray]:
        """Do network update for a given epoch and return the aggregate"""
    
    @abstractmethod
    def reconvene_network_updates(self, sharing_mean_1: CPUArray, sharing_mean_2: CPUArray, sharing_dual: CPUArray):
        """Finalize network updates for a single inner ADMM iteration"""


class SynchADMMWorkerBackendBase(CommunicationBackendBase):
    # These workers never directly reach-out to any node, they are passive
    def are_all_peers_reachable(self):
        return True
    def are_all_workers_reachable(self):
        return True

    @property
    def set_solver_parameters(self) -> Callable[[SolverParams, int], None]:
        return self._set_solver_parameters
    @set_solver_parameters.setter
    def set_solver_parameters(self, f: Callable[[SolverParams, int], None]):
        self._set_solver_parameters = f

    @property
    def set_topology(self) -> Callable[[nx.DiGraph], None]:
        return self._set_topology
    @set_topology.setter
    def set_topology(self, f: Callable[[nx.DiGraph], None]):
        self._set_topology = f
    
    @property
    def do_inner_loop_update(self) -> Callable[[int], Tuple[int, CPUArray]]:
        return self._do_inner_loop_update
    @do_inner_loop_update.setter
    def do_inner_loop_update(self, f: Callable[[int], Tuple[int, CPUArray]]):
        self._do_inner_loop_update = f
    
    @property
    def update_cached_values(self) -> Callable[[CPUArray, CPUArray, CPUArray], None]:
        return self._update_cached_values
    @update_cached_values.setter
    def update_cached_values(self, f: Callable[[CPUArray, CPUArray, CPUArray], None]):
        self._update_cached_values = f
    
    @property
    def report_chunk(self) -> Callable[[None], CPUArray]:
        return self._report_chunk
    @report_chunk.setter
    def report_chunk(self, f: Callable[[None], CPUArray]):
        self._report_chunk = f
    
    @property
    def report_aggregate(self) -> Callable[[None], CPUArray]:
        return self._report_aggregate
    @report_aggregate.setter
    def report_aggregate(self, f: Callable[[None], CPUArray]):
        self._report_aggregate = f

    @property
    def update_demands(self) -> Callable[[CPUArray], CPUArray]:
        return self._update_demands
    @update_demands.setter
    def update_demands(self, f: Callable[[CPUArray], CPUArray]):
        self._update_demands = f
