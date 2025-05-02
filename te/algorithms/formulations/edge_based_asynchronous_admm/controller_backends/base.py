import signal
from dataclasses import dataclass
from typing import Dict, Type, List, Optional
from abc import ABC, abstractmethod
from te.algorithms.array_utils.cpu_utils import CPUArray
from te.algorithms.base import SolverParams
from .. import AsynchronousADMMControllerRPCParams, NetworkUpdate, ControllerUpdate


class ControllerCommunicationBackendBase(ABC):
    @abstractmethod
    def __init__(self):
        self.is_alive = False
        self.killed = False
    
    @classmethod
    @abstractmethod
    def backend_name(cls) -> str:
        """Name of the communication backend to use for controller scatter/gather operations"""

    @property
    @abstractmethod
    def number_of_nodes(self) -> int:
        """Number of nodes attached to this backend"""

    @property
    def Upsilon(self) -> int:
        """Minimum number of updateable switches before we can unblock the controller"""
        return self._upsilon
    @Upsilon.setter
    def Upsilon(self, upsilon: int):
        self._upsilon = upsilon
    
    @property
    def is_alive(self) -> bool:
        return self._is_alive
    @is_alive.setter
    def is_alive(self, alive: bool):
        self._is_alive = alive

    @property
    def killed(self) -> bool:
        return self._killed
    @killed.setter
    def killed(self, kill: bool):
        self._killed = kill
    
    @abstractmethod
    def start(self):
        pass
    
    @abstractmethod
    def stop(self):
        pass
    
    @abstractmethod
    def die(self):
        pass

    def register_signal_handler(self):
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.die)

    @abstractmethod
    def update_network_nodes(self, update: ControllerUpdate):
        """Broadcast an update message to the network nodes"""
    
    @abstractmethod
    def get_network_updates(self) -> List[NetworkUpdate]:
        """Get a list of updateable switch number to consumable network updates"""

    @abstractmethod
    def are_network_nodes_ready(self) -> bool:
        """Check if all network nodes are ready"""

    @abstractmethod
    def initialize_worker_nodes(self, solver_params: SolverParams, basis: CPUArray, initial_feasible_solution: CPUArray,
                                mask: Optional[CPUArray] = None):
        """Initialize worker nodes with solver parameters and initial feasible solution (X_ek_0) and optional mask"""

    @abstractmethod
    def update_demands(self, updated_feasible_solution: CPUArray):
        """Update the initial feasible solution (X_ek_0)"""
    
    @abstractmethod
    def get_X_ek(self, basis: CPUArray, initial_feasible_solution: CPUArray) -> CPUArray:
        """Get the final solution array (X_ek)"""
    
    @abstractmethod
    def set_active_commodity_count(self, K: int):
        """Set total number of active commodities in the network (needed for local updates)"""
    
    @abstractmethod
    def close(self):
        """Close and cleanup all nodes attached to this backend"""


_BACKENDS: Dict[str, Type[ControllerCommunicationBackendBase]] = dict()
_PARAMS: Dict[str, Type[AsynchronousADMMControllerRPCParams]] = dict()


def controller_communication_backend(cls: Type[ControllerCommunicationBackendBase]) -> ControllerCommunicationBackendBase:
    """Decorator that registers any communication backend for simple use"""
    global _BACKENDS

    assert issubclass(cls, ControllerCommunicationBackendBase)
    tpe = cls.backend_name()
    assert tpe not in _BACKENDS
    _BACKENDS[tpe] = cls
    return cls


def controller_communication_backend_params(cls: Type[AsynchronousADMMControllerRPCParams]) -> AsynchronousADMMControllerRPCParams:
    """Decorator that registers any communication backend parameters for simple use"""
    global _PARAMS

    assert issubclass(cls, AsynchronousADMMControllerRPCParams)
    tpe = cls.Backend
    assert tpe not in _PARAMS
    _PARAMS[tpe] = cls
    return cls
