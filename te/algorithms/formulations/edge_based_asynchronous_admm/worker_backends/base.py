from abc import abstractmethod, ABC
from typing import Type, Dict, Callable, List, Tuple, Optional
from te.algorithms.array_utils.cpu_utils import CPUArray
from te.algorithms.base import SolverParams
from .. import AsynchronousADMMWorkerRPCParams


class WorkerNodeCommunicationBackendBase(ABC):
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
    
    @abstractmethod
    def stop(self):
        """Stop collecting updates (MUST be idempotent)"""

    @abstractmethod
    def wait_until_initialized(self) -> bool:
        """Wait until we can start with the algorithm or are interrupted"""

    @abstractmethod
    def gather_updates(self, block = False) -> Optional[List[Tuple[CPUArray, CPUArray, CPUArray]]]:
        """
        If immediately available, gathter up to `Upsilon` updates from the queue.
        If no update is available and `block` is False, return an empty list.
        If no update is available and `block` is True, block until an update arrives.
        If interrupted, return None.
        """

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
