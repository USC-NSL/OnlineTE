import signal
import networkx as nx
from typing import Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from . import WorkerRPCParams, ControllerRPCParams
from te.algorithms.base import SolverParams, TrafficEngineeringLP, TrafficMatrixBase
from te.algorithms.sub_algorithms.mlu_backends.base import ControllerMLUSolver


class _CommunicationBackendBase(ABC):
    @classmethod
    @abstractmethod
    def backend_name(cls) -> str:
        """Name of this backend"""

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
    def close(self):
        """
        Close and cleanup all nodes attached to this backend.
        If `die` was called, its behavior may be changed.
        """
    
    def register_signal_handler(self):
        """Delegate signal handling to the backend, otherwise, the controller/worker should do it"""
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.die)


class ControllerCommunicationBackendBase(_CommunicationBackendBase):
    @abstractmethod
    def __init__(self, rpc_params: ControllerRPCParams):
        super().__init__()

    @property
    @abstractmethod
    def number_of_nodes(self) -> int:
        """Number of nodes attached to this backend"""

    @abstractmethod
    def are_network_nodes_ready(self) -> bool:
        """Check if all network nodes are ready"""


class WorkerCommunicationBackendBase(_CommunicationBackendBase):
    @abstractmethod
    def __init__(self, rpc_params: WorkerRPCParams):
        super().__init__()


@dataclass
class WorkerNodeParams:
    communication_backend: type[WorkerCommunicationBackendBase]
    rpc_params: WorkerRPCParams


class WorkerNodeBase(ABC):
    @abstractmethod
    def __init__(params: WorkerNodeParams, *args, **kwargs):
        pass

    @abstractmethod
    def initialize(self):
        """
        Initialize the worker node by setting the known solver parameters and configuring the
        communication backend to wait for the controller node to interact with it.
        """
    
    @abstractmethod
    def wait(self):
        """
        Wait until the communication backend terminates.
        """
    
    @abstractmethod
    def close(self):
        """
        Close and cleanup the communication backend and this object.
        """
    
    @classmethod
    @abstractmethod
    def spawn_and_wait(cls, params: WorkerNodeParams, *args, **kwargs):
        """
        Spawn a fresh instance of this worker node and instruct it to `wait`.
        """


@dataclass
class ControllerNodeParams:
    graph: nx.DiGraph
    traffic: TrafficMatrixBase
    solver_params: SolverParams
    mlu_backend: type[ControllerMLUSolver]
    mlu_params: SolverParams
    communication_backend: type[ControllerCommunicationBackendBase]
    rpc_params: ControllerRPCParams


class ControllerNodeBase(TrafficEngineeringLP):
    @abstractmethod
    def __init__(params: ControllerNodeParams, *args, **kwargs):
        pass

    @abstractmethod
    def are_network_nodes_ready(self) -> bool:
        """
        Returns `True`, if all the designated worker nodes for this controller are active.
        """