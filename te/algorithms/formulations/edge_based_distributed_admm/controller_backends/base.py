from typing import Dict, Type
from abc import ABC, abstractmethod
from te.algorithms.array_utils.cpu_utils import CPUArray
from te.algorithms.base import SolverParams
import te.algorithms.formulations.edge_based_distributed_admm.controller_backends


class ControllerCommunicationBackendBase(ABC):
    @classmethod
    @abstractmethod
    def backend_name(cls) -> str:
        """Name of the communication backend to use for controller scatter/gather operations"""

    @property
    @abstractmethod
    def number_of_nodes(self) -> int:
        """Number of nodes attached to this backend"""

    @abstractmethod
    def are_network_nodes_ready(self) -> bool:
        """Check if all network nodes are ready"""

    @abstractmethod
    def initialize_worker_nodes(self, solver_params: SolverParams, basis: CPUArray, initial_feasible_solution: CPUArray):
        """Initialize worker nodes with solver parameters and initial feasible solution (X_ek_0)"""
    
    @abstractmethod
    def get_X_ek(self, basis: CPUArray, initial_feasible_solution: CPUArray) -> CPUArray:
        """Get the final solution array (X_ek)"""
    
    @abstractmethod
    def get_X_ek_sum(self) -> CPUArray:
        """Get the total flow over each edge"""
    
    @abstractmethod
    def do_network_update(self, epoch: int) -> CPUArray:
        """Do network update for a given epoch and return the aggregate"""
    
    @abstractmethod
    def reconvene_network_updates(self, P_bar_t: CPUArray, Y_bar_t: CPUArray, u_t: CPUArray):
        """Finalize network updates for a single inner ADMM iteration"""
    
    @abstractmethod
    def close(self):
        """Close and cleanup all nodes attached to this backend"""


_BACKENDS: Dict[str, Type[ControllerCommunicationBackendBase]] = dict()


def controller_communication_backend(cls: Type[ControllerCommunicationBackendBase]) -> ControllerCommunicationBackendBase:
    """Decorator that registers any communication backend for simple use"""
    global _BACKENDS

    assert issubclass(cls, ControllerCommunicationBackendBase)
    tpe = cls.backend_name()
    assert tpe not in _BACKENDS
    _BACKENDS[tpe] = cls
    return cls
