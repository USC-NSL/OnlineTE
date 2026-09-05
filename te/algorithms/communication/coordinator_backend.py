import networkx as nx
from abc import abstractmethod
from typing import Tuple, Optional
from .base import CommunicationBackendBase
from array_utils.cpu.types import *
from te.algorithms.base import SolverParams, TEObjective


class CoordinatorBackendBase[P: SolverParams](CommunicationBackendBase[P]):
    @abstractmethod
    def initialize_worker_nodes(
        self,
        solver_params: P,
        topology: nx.DiGraph,
        objective: TEObjective
    ):
        """
        The nodes need to know a few thing before they can start.
        In particular, they need the full solver parameters and the
        graph before they can start.
        The worker nodes are expected to only answer this RPC when
        all work on their side is finished so the coordinator can
        proceed.
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
    def do_network_update(self, epoch: int) -> Tuple[int, CPUArray, Optional[float]]:
        """Do network update for a given epoch and return the aggregate"""
    
    @abstractmethod
    def reconvene_network_updates(self, sharing_mean_1: CPUArray, sharing_mean_2: CPUArray, sharing_dual: CPUArray):
        """Finalize network updates for a single inner ADMM iteration"""
