import signal
import contextlib
import te.constants
from typing import Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from te.algorithms.base import SolverParams, TrafficEngineeringProblemDescription
from utils.exceptions import SolutionInterrupted
from utils.logging import as_warning


"""
We model a distributed setting as follows:
- Our model is hierarchical, which can be though of as nodes within layers going
  from 0 to `n-1` in some `n` level topology.
- Each node in level `i` controls a partition of nodes in level `i+1` as its _worker_
  ndoes. It is allowed to invoke RPCs on them without any prior notice.
- Nodes within the same level are considered _peers_, which can communicate with each
  other in an asynchronous manner or not all (depending on how the user wants them to
  look).
- Each node has a communication backend, which abstract away the operation of sending 
  arrays over the wire from the solvers implemented within the node.
  These backends will be invoked to exchange messages between solver nodes, and may use any
  particular implementation to do it (e.g. gRPC, MPI, IP Multicast, etc.).

Here, we define the base of such classes, which are written in a generic manner so that
they fit into any distributed solver.
"""


@dataclass
class RPCParams(SolverParams):
    PeerIndex: int = 0
    Peers: Tuple[Tuple[str, int]] = (("localhost", te.constants.DEFAULT_RPC_PORT),)
    Workers: Tuple[Tuple[str, int]] = tuple()
    
    def __post_init__(self):
        self.left_column_share = 0.2
    
    def get_bind_address(self) -> Tuple[str, int]:
        if len(self.Peers) == 1:
            return self.Peers[0]
        assert self.PeerIndex < len(self.Peers)
        return self.Peers[self.PeerIndex]


class CommunicationBackendBase(ABC):
    """
    All backends inherit this. Implements simple properties for
    checking if the backend is doing anything and to delegate signal
    handlers.

    Note
    ----
    Usually, intrrupting the backends is much harder than the solvers,
    as the solvers usually stop the moment a signal arrives. The backends
    however can be interrupted while handling IO, and thus must be handled
    with a bit more care.

    It is for this reason that users may explicitly decide to delegate the
    signal handlers completely to the backends by calling `register_signal_handler`;
    Doing so means that whenever the solvers receive an interrupt, they can be sure
    that the communication backend has already received and processed it.
    """

    @abstractmethod
    def __init__(self, rpc_params: RPCParams):
        super().__init__()
        self._rpc_params = rpc_params
    
    @property
    def rpc_params(self) -> RPCParams:
        return self._rpc_params

    @property
    def peer_id(self) -> int:
        """Unique ID of this peer within its level"""
        return self._rpc_params.PeerIndex

    @property
    def number_of_peers(self) -> int:
        """Number of peers in the current level (including this node!)"""
        return len(self._rpc_params.Peers)
    
    @property
    def number_of_workers(self) -> int:
        """Number of workers for this node"""
        return len(self._rpc_params.Workers)

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

    @classmethod
    @abstractmethod
    def backend_name(cls) -> str:
        """Name of this backend"""
    
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
    def wait(self):
        """
        Put the calling thread to sleep until the backend is finished
        and closed.
        """
    
    @abstractmethod
    def close(self):
        """
        Close and cleanup all nodes attached to this backend.
        If `die` was called, its behavior may be changed.
        """

    @abstractmethod
    def are_all_peers_reachable(self) -> bool:
        """Check if all peers are ready"""

    @abstractmethod
    def are_all_workers_reachable(self) -> bool:
        """Check if all network nodes for this peer are ready"""
    
    def register_signal_handler(self):
        """Delegate signal handling to the backend, otherwise, the controller/worker should do it"""
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.die)


@dataclass
class DistributedSolverNodeParams:
    """
    Dataclass that defines basic parameters for creating a distributed solver node.
    Besides the usual input (problem description and solver parameters), the distributed
    solver will require the backend class and its parameters as well.
    """
    CommunicationBackendCLS: type[CommunicationBackendBase]
    RPCParams_: RPCParams
    SolverParams_: Optional[SolverParams] = None
    ProblemDescription: Optional[TrafficEngineeringProblemDescription] = None


class DistributedSolverNodeBase(ABC):
    @abstractmethod
    def __init__(self, node_params: DistributedSolverNodeParams):
        self._node_params = node_params
        # First interrupt is graceful, the next one kills the process no questions asked ...
        self._die_on_next_int = False
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.die)
    
    @property
    def node_params(self) -> DistributedSolverNodeParams:
        return self._node_params
    
    @property
    def backend(self) -> CommunicationBackendBase:
        """
        Communication backend instance for this node.
        """
        assert self._backend is not None
        return self._backend
    @backend.setter
    def backend(self, backend: CommunicationBackendBase):
        self._backend = backend

    @abstractmethod
    def initialize(self):
        """
        Initialize the solver node.
        MUST be called before `run`.
        """
    
    # @abstractmethod
    # def wait(self):
    #     """
    #     Wait until the communication backend terminates.
    #     """

    @abstractmethod
    def run(self):
        """
        Begin procedure for this peer and respond to messages from other peers.
        This is a blocking call and will use the calling thread for execution.
        """
    
    @abstractmethod
    def close(self):
        """
        Close and cleanup the communication backend and this object.
        """

    def stop(self, _, __):
        if self._die_on_next_int:
            signal.raise_signal(signal.SIGTERM)
        else:
            print(as_warning('SIGINT: Stopping solver. Invoke again to kill the process.'))
            if self.backend:
                self.backend.stop()
            self._die_on_next_int = True
            raise SolutionInterrupted
    
    def die(self, _, __):
        print(as_warning('SIGTERM: Killing the solver.'))
        if self.backend:
            self.backend.die()

    def are_all_peers_reachable(self) -> bool:
        return self.backend.are_all_peers_reachable()
    
    def are_all_workers_reachable(self) -> bool:
        return self.backend.are_all_workers_reachable()
    
    @classmethod
    def spawn_and_run(cls, node_params: DistributedSolverNodeParams, *args, **kwargs):
        """
        Spawn a fresh instance of this worker node and instruct it to `run`.
        """
        with contextlib.closing(cls(node_params, *args, **kwargs)) as node:
            node.initialize()
            node.run()


__all__ = [
    'CommunicationBackendBase', 'RPCParams', 
    'DistributedSolverNodeBase', 'DistributedSolverNodeParams'
]