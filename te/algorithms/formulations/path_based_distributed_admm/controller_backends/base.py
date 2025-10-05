import signal
from typing import Dict, Type, Tuple, Optional
from abc import ABC, abstractmethod
from te.algorithms.array_utils.cpu_utils import CPUArray
from te.algorithms.base import SolverParams
from .. import PathBasedDistributedADMMControllerRPCParams


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
    def is_alive(self) -> bool:
        """Is the backend serving/receiving new messages?"""
        return self._is_alive
    @is_alive.setter
    def is_alive(self, alive: bool):
        self._is_alive = alive

    @property
    def killed(self) -> bool:
        """Set to `True` when `die` is called"""
        return self._killed
    @killed.setter
    def killed(self, kill: bool):
        self._killed = kill
    
    @abstractmethod
    def start(self):
        """Start the backend. After this command is executed, it should be ready to accept messages"""
    
    @abstractmethod
    def stop(self):
        """
        Stop serving new requests. This function MUST be idempotent.
        It may NOT close connections to the nodes, as the controller may
        want to do a final back-and-forth in the end.
        Optionally, it will attach to the `SIGINT` signal handler.
        """
    
    @abstractmethod
    def die(self):
        """
        Completely close the backend, no questions asked.
        Allowed to cancel inflight RPCs and leave incomplete requests
        unfinished.
        Optionally, may be invoked with `SIGTERM`.
        """

    @abstractmethod
    def are_network_nodes_ready(self) -> bool:
        """Check if all network nodes are ready"""

    @abstractmethod
    def initialize_worker_nodes(self, solver_params: SolverParams, alpha: CPUArray, beta: CPUArray, demands: CPUArray):
        """Initialize worker nodes with solver parameters and path configurations (i.e. alpha_ket, beta_k, D_k)"""

    @abstractmethod
    def update_demands(self, updated_demands: CPUArray):
        """Update D_k"""
    
    @abstractmethod
    def get_X_ek(self, alpha: CPUArray, demands: CPUArray) -> CPUArray:
        """Get the final solution array (X_ek)"""
    
    @abstractmethod
    def do_network_update(self, epoch: int) -> Tuple[int, CPUArray]:
        """Do network update for a given epoch and return the aggregate"""
    
    @abstractmethod
    def reconvene_network_updates(self, X_bar_e: CPUArray, P_bar_e: CPUArray, u_e: CPUArray):
        """Finalize network updates for a single inner ADMM iteration"""
    
    @abstractmethod
    def set_active_commodity_count(self, K: int):
        """Set total number of active commodities in the network (needed for local updates)"""
    
    @abstractmethod
    def close(self):
        """
        Close and cleanup all nodes attached to this backend.
        If `die` was called, its behavior may be changed.
        """
    
    def register_signal_handler(self):
        """Delegate signal handling to the backend, otherwise, the controller/worker should do it"""
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.die)
    
    def reset_inner_dual_variable(self):
        pass


_BACKENDS: Dict[str, Type[ControllerCommunicationBackendBase]] = dict()
_PARAMS: Dict[str, Type[PathBasedDistributedADMMControllerRPCParams]] = dict()


def controller_communication_backend(cls: Type[ControllerCommunicationBackendBase]) -> ControllerCommunicationBackendBase:
    """Decorator that registers any communication backend for simple use"""
    global _BACKENDS

    assert issubclass(cls, ControllerCommunicationBackendBase)
    tpe = cls.backend_name()
    assert tpe not in _BACKENDS
    _BACKENDS[tpe] = cls
    return cls


def controller_communication_backend_params(cls: Type[PathBasedDistributedADMMControllerRPCParams]) -> PathBasedDistributedADMMControllerRPCParams:
    """Decorator that registers any communication backend parameters for simple use"""
    global _PARAMS

    assert issubclass(cls, PathBasedDistributedADMMControllerRPCParams)
    tpe = cls.Backend
    assert tpe not in _PARAMS
    _PARAMS[tpe] = cls
    return cls
