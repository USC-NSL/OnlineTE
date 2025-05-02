import signal
from abc import abstractmethod, ABC
from typing import Type, Dict, Callable, List, Optional
from te.algorithms.array_utils.cpu_utils import CPUArray
from te.algorithms.base import SolverParams
from .. import AsynchronousADMMWorkerRPCParams, ControllerUpdate, NetworkUpdate


class WorkerNodeCommunicationBackendBase(ABC):
    @abstractmethod
    def __init__(self):
        self.is_alive = False
        self.killed = False
    
    @classmethod
    @abstractmethod
    def backend_name(cls) -> str:
        """Name of the communication backend to use for responding to controller RPCs"""

    @property
    @abstractmethod
    def worker_id(self) -> int:
        """ID of the worker attached to this backend"""
    
    @property
    def Sigma(self) -> int:
        """Maximum number of local iterations without using new controller updates"""
        return self._Sigma
    @Sigma.setter
    def Sigma(self, sigma: int):
        self._Sigma = sigma
    
    @property
    def WorkerBatchSize(self) -> int:
        """Maximum number of consecutive controller updates that can be consumed at once"""
        return self._WorkerBatchSize
    @WorkerBatchSize.setter
    def WorkerBatchSize(self, size: int):
        self._WorkerBatchSize = size
    
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

    @abstractmethod
    def wait_for_close(self) -> bool:
        """
        This method is called when the worker ends the solution procedure and just waits for the
        controller to give the go on quitting (this is done by calling the `Close` RPC).
        Optionally, it may have a timeout, afterwhich the worker will quit anyway.
        On timeout, returns `True`, else `False`.
        """

    def register_signal_handler(self):
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.die)

    @abstractmethod
    def wait_until_initialized(self) -> bool:
        """Wait until we can start with the algorithm or are interrupted"""

    @abstractmethod
    def gather_updates(self, block = False) -> Optional[List[ControllerUpdate]]:
        """
        If immediately available, gathter up to `WorkerBatchSize` updates from the queue.
        If no update is available and `block` is False, return an empty list.
        If no update is available and `block` is True, block until an update arrives.
        If interrupted, return None.
        """
    
    @abstractmethod
    def send_update_to_controller(self, update: NetworkUpdate):
        """Send an update to the controller"""

    @property
    def set_initial_feasible_solution(self) -> Callable[[CPUArray], None]:
        return self._set_initial_feasible_solution
    @set_initial_feasible_solution.setter
    def set_initial_feasible_solution(self, f: Callable[[CPUArray], None]):
        self._set_initial_feasible_solution = f

    @property
    def set_mask(self) -> Callable[[CPUArray], None]:
        return self._set_mask
    @set_mask.setter
    def set_mask(self, f: Callable[[CPUArray], None]):
        self._set_mask = f

    @property
    def set_null_space_basis(self) -> Callable[[CPUArray], None]:
        return self._set_null_space_basis
    @set_null_space_basis.setter
    def set_null_space_basis(self, f: Callable[[CPUArray], None]):
        self._set_null_space_basis = f

    @property
    def set_active_commodity_count(self) -> Callable[[int], None]:
        return self._set_active_commodity_count
    @set_active_commodity_count.setter
    def set_active_commodity_count(self, f: Callable[[int], None]):
        self._set_active_commodity_count = f
    
    @property
    def report_chunk(self) -> Callable[[None], CPUArray]:
        return self._report_chunk
    @report_chunk.setter
    def report_chunk(self, f: Callable[[None], CPUArray]):
        self._report_chunk = f
    
    @property
    def is_worker_node_ready(self) -> bool:
        return self._is_worker_node_ready
    @is_worker_node_ready.setter
    def is_worker_node_ready(self, ready: bool):
        self._is_worker_node_ready = ready
    
    @property
    def set_solver_parameters(self) -> Callable[[SolverParams], None]:
        return self._set_solver_parameters
    @set_solver_parameters.setter
    def set_solver_parameters(self, f: Callable[[SolverParams], None]):
        self._set_solver_parameters = f

    @property
    def is_initialized(self) -> Callable[[None], bool]:
        """Are we ready to start the algorithm?"""
        return self._is_initialized
    @is_initialized.setter
    def is_initialized(self, f: Callable[[None], bool]):
        self._is_initialized = f
    
    @property
    def close(self) -> Callable[[None], None]:
        return self._close
    @close.setter
    def close(self, f: Callable[[None], None]):
        self._close = f


_BACKENDS: Dict[str, Type[WorkerNodeCommunicationBackendBase]] = dict()
_PARAMS: Dict[str, Type[AsynchronousADMMWorkerRPCParams]] = dict()


def worker_node_communication_backend(cls: Type[WorkerNodeCommunicationBackendBase]) -> WorkerNodeCommunicationBackendBase:
    """Decorator that registers any communication backend for simple use"""
    global _BACKENDS

    assert issubclass(cls, WorkerNodeCommunicationBackendBase)
    tpe = cls.backend_name()
    assert tpe not in _BACKENDS
    _BACKENDS[tpe] = cls
    return cls


def worker_communication_backend_params(cls: Type[AsynchronousADMMWorkerRPCParams]) -> AsynchronousADMMWorkerRPCParams:
    """Decorator that registers any communication backend parameters for simple use"""
    global _PARAMS

    assert issubclass(cls, AsynchronousADMMWorkerRPCParams)
    tpe = cls.Backend
    assert tpe not in _PARAMS
    _PARAMS[tpe] = cls
    return cls
