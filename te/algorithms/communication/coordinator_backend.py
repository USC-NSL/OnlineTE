import networkx as nx
from abc import abstractmethod
from typing import Tuple, Optional, Dict
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
    def do_network_update(
        self,
        sharing_gap: CPUArray,
        sharing_dual: CPUArray,
        sharing_rho: float
    ) -> Tuple[int, CPUArray, Dict[str, float]]:
        """
        Do network update for a given epoch.
        Returns how long the update took (in micro-seconds),
        the new sharing mean, as well as a dict of arbitrary
        data.

        Note
        ----
        Each worker returns its own dict object. It is _ASSUMED_
        that we can just sum up the value of each key to create
        the new values and return a dict object with those values
        instead.
        """
