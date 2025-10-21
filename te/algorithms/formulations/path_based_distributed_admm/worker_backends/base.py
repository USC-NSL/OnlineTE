import signal
from typing import Dict, Type, Callable, Tuple
from abc import ABC, abstractmethod
from te.algorithms.array_utils.cpu_utils import CPUArray, BooleanCPUArray, IntegerCPUArray
from te.algorithms.base import SolverParams


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
        Stop accepting scattered updates.
        The backend must still accept RPCs, since the controller
        may want to get some information before it exits.
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
    def wait(self):
        """Wait until `stop` is called or the server is terminated"""

    @abstractmethod
    def close(self):
        """
        Close and cleanup all nodes attached to this backend.
        If `die` was called, its behavior may be changed.
        """

    @property
    def set_alpha(self) -> Callable[[BooleanCPUArray], None]:
        return self._set_alpha
    @set_alpha.setter
    def set_alpha(self, f: Callable[[BooleanCPUArray], None]):
        self._set_alpha = f

    @property
    def set_beta(self) -> Callable[[IntegerCPUArray], None]:
        return self._set_beta
    @set_beta.setter
    def set_beta(self, f: Callable[[IntegerCPUArray], None]):
        self._set_beta = f

    @property
    def set_demands(self) -> Callable[[CPUArray], None]:
        return self._set_demands
    @set_demands.setter
    def set_demands(self, f: Callable[[CPUArray], None]):
        self._set_demands = f
    
    @property
    def do_inner_loop_update(self) -> Callable[[int], Tuple[int, CPUArray]]:
        return self._do_inner_loop_update
    @do_inner_loop_update.setter
    def do_inner_loop_update(self, f: Callable[[int], Tuple[int, CPUArray]]):
        self._do_inner_loop_update = f

    @property
    def set_active_commodity_count(self) -> Callable[[int], None]:
        return self._set_active_commodity_count
    @set_active_commodity_count.setter
    def set_active_commodity_count(self, f: Callable[[int], None]):
        self._set_active_commodity_count = f
    
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
    
    def register_signal_handler(self):
        """Delegate signal handling to the backend, otherwise, the controller/worker should do it"""
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.die)


_BACKENDS: Dict[str, Type[WorkerNodeCommunicationBackendBase]] = dict()


def worker_node_communication_backend(cls: Type[WorkerNodeCommunicationBackendBase]) -> WorkerNodeCommunicationBackendBase:
    """Decorator that registers any communication backend for simple use"""
    global _BACKENDS

    assert issubclass(cls, WorkerNodeCommunicationBackendBase)
    tpe = cls.backend_name()
    assert tpe not in _BACKENDS
    _BACKENDS[tpe] = cls
    return cls
