from typing import Dict, Type, Callable, Tuple, Optional
from abc import ABC, abstractmethod
from te.algorithms.array_utils.cpu_utils import CPUArray
from te.algorithms.base import SolverParams


class WorkerNodeCommunicationBackendBase(ABC):
    @classmethod
    @abstractmethod
    def backend_name(cls) -> str:
        """Name of the communication backend to use for responding to controller RPCs"""

    @property
    @abstractmethod
    def worker_id(self) -> int:
        """ID of the worker attached to this backend"""
    
    @abstractmethod
    def start(self):
        """Start listening on the backend (should be non-blocking)"""
    
    @abstractmethod
    def stop(self):
        """Stop listening on the backend. May or may not cancel inflight RPCs"""
    
    @abstractmethod
    def wait(self):
        """Wait until `stop` is called or the server is terminated"""

    @property
    def set_initial_feasible_solution(self) -> Callable[[CPUArray], None]:
        return self._set_initial_feasible_solution
    @set_initial_feasible_solution.setter
    def set_initial_feasible_solution(self, f: Callable[[CPUArray], None]):
        self._set_initial_feasible_solution = f

    @property
    def set_null_space_basis(self) -> Callable[[CPUArray], None]:
        return self._set_null_space_basis
    @set_null_space_basis.setter
    def set_null_space_basis(self, f: Callable[[CPUArray], None]):
        self._set_null_space_basis = f
    
    @property
    def do_inner_loop_update(self) -> Callable[[int, Optional[CPUArray]], Tuple[int, CPUArray]]:
        return self._do_inner_loop_update
    @do_inner_loop_update.setter
    def do_inner_loop_update(self, f: Callable[[int, Optional[CPUArray]], Tuple[int, CPUArray]]):
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
    def report_aggregate(self) -> Callable[[None], CPUArray]:
        return self._report_aggregate
    @report_aggregate.setter
    def report_aggregate(self, f: Callable[[None], CPUArray]):
        self._report_aggregate = f
    
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
    def close(self) -> Callable[[None], None]:
        return self._close
    @close.setter
    def close(self, f: Callable[[None], None]):
        self._close = f


_BACKENDS: Dict[str, Type[WorkerNodeCommunicationBackendBase]] = dict()


def worker_node_communication_backend(cls: Type[WorkerNodeCommunicationBackendBase]) -> WorkerNodeCommunicationBackendBase:
    """Decorator that registers any communication backend for simple use"""
    global _BACKENDS

    assert issubclass(cls, WorkerNodeCommunicationBackendBase)
    tpe = cls.backend_name()
    assert tpe not in _BACKENDS
    _BACKENDS[tpe] = cls
    return cls
