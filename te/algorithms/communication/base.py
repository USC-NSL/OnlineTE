import signal
import importlib
import contextlib
import te.constants
import networkx as nx
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
from te.algorithms.base import SolverParams
from utils.exceptions import SolutionInterrupted
from utils.logging import as_warning
from array_utils.cpu.types import CPUArray, cpu_array
from topologies.utils import get_edge_indexing
from google.protobuf.message import Message
from google.protobuf.any_pb2 import Any
from google.protobuf.json_format import MessageToDict
from utils.logging import as_fail


"""
We model a distributed setting as follows:
- Our model is hierarchical, which can be thought of as nodes within layers going
  from 0 to `n-1` in some `n` level control plane topology.
- Each node in level `i` controls a partition of nodes in level `i+1` as its _worker_
  ndoes. It is allowed to invoke RPCs on them without any prior notice.
- Nodes within the same level are considered _peers_, which can communicate with each
  other in an asynchronous manner or not all (depending on how the user wants them to
  interact).
- Each node has a communication backend, which abstract away the operation of sending 
  arrays over the wire from the solvers implemented within the node.
  These backends will be invoked to exchange messages between solver nodes, and may use any
  particular implementation to do it (e.g. gRPC, MPI, IP Multicast, etc.).

Here, we define the base of such classes, which are written in a generic manner so that
they fit into any distributed solver.
"""


@dataclass(frozen=True)
class RPCParams(SolverParams):
    PeerIndex: int = 0
    """Index of _this_ peer within its network"""
    Peers: Tuple[Tuple[str, int]] = (("localhost", te.constants.DEFAULT_RPC_PORT),)
    """Address list (as a tuple) of _all_ peers within this network"""
    Workers: Tuple[Tuple[str, int],...] = tuple()
    """Address list of all worker nodes for this peer"""
    _left_column_share = 0.2
    
    def get_bind_address(self) -> Tuple[str, int]:
        if len(self.Peers) == 1:
            return self.Peers[0]
        assert self.PeerIndex < len(self.Peers)
        return self.Peers[self.PeerIndex]


class CommunicationBackendBase[P: SolverParams](ABC):
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

    Types
    -----
    `P`: A subclass of `SolverType`. The acutal solver parameters used by the
         solver class.
    """

    @abstractmethod
    def __init__(self, rpc_params: RPCParams, solver_params_cls: type[P]):
        super().__init__()
        self._rpc_params = rpc_params
        self._solver_params_cls = solver_params_cls
        self.set_solver_param_message_type()
    
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

    @property
    def solver_param_object_type(self) -> type[P]:
        return self._solver_params_cls

    def set_solver_param_message_type(self):
        """
        Resolves the concrete class bound to solver parameters message at runtime.
        It is _expected_ that this wire type is protocol buffer implementation of
        the same name under `protos.solver_params`.
        """
        param_object_type = self.solver_param_object_type
        try:
            self._solver_params_message_cls = getattr(
                importlib.import_module('protos.solver_params.solver_params_pb2'),
                param_object_type.__name__
            )
        except ImportError as e:
            raise RuntimeError(f'No `solver_params` module exists under `protos`. '
                               'Did you compile them?') from e
        except AttributeError as e:
            raise RuntimeError(f'Solver parameters {param_object_type.__name__} does '
                               'not have a protobuff message in `protos`!') from e
    @property
    def solver_params_message_cls(self) -> type[Message]:
        return self._solver_params_message_cls

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
    
    # TODO: There seems to be a case where the signal is processed
    #       multiple times. Find out what causes that ...
    # def register_signal_handler(self):
    #     """Delegate signal handling to the backend, otherwise, the controller/worker should do it"""
    #     signal.signal(signal.SIGINT, self.stop)
    #     signal.signal(signal.SIGTERM, self.die)

    def serialize_solver_params(self, solver_params: P) -> Message:
        """
        Serialize the solver parameters such that we can pack it into
        a field in a `core_messages.SolverParameters` message.
        """
        return self.solver_params_message_cls(**solver_params.child_fields)

    def deserialize_solver_params(self, buf: Any) -> Optional[P]:
        """
        Given an `parameter` field from a `core_messages.SolverParameters`,
        deserialize it into an instance of solver paramters given as `P`.
        If the buffer cannot be unpacked, return None.
        """
        message_class = self.solver_params_message_cls
        param_class = self.solver_param_object_type
        if buf.Is(message_class.DESCRIPTOR):
            new_params = message_class()
            buf.Unpack(new_params)
            return param_class(**MessageToDict(new_params))
        print(as_fail(f"Failed to parse solver parameters from coordinator!"))


@dataclass
class DistributedSolverNodeParams:
    """
    Dataclass that defines basic parameters for creating a distributed solver node.
    Besides the usual input (problem description and solver parameters), the distributed
    solver will require the backend class and its parameters as well.
    """
    CommunicationBackendCLS: type[CommunicationBackendBase]
    RPCParams_: RPCParams


class DistributedSolverNodeBase(ABC):
    @abstractmethod
    def __init__(self, node_params: DistributedSolverNodeParams, **kwargs):
        super().__init__(**kwargs)
        self._node_params = node_params
        # First interrupt is graceful, the next one kills the process no questions asked ...
        self._die_on_next_int = False
        # Currently, we can only set this _ONCE_. This is because changing the number of
        # workers requires a multi-cast within the network that is quite difficult.
        # As such, for now, in case of node failure, we just restart the workers rather
        # than even bothering to send the information across the network.
        # Fixing this is a big TODO on our plate ...
        self._number_of_workers: Optional[int] = None
        # These fields are set automatically the moment the coordinator reveals the
        # solver parameters and the topology to the workers
        self._graph: Optional[nx.DiGraph] = None
        self._indexing: Dict[Tuple[int, int], int] = None
        self._capacities: Optional[CPUArray] = None
        self._total_commodity_count: Optional[int] = None
        self._assigned_commodity_count: Optional[int] = None
        self._assigned_commodity_start_id: Optional[int] = None

        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.die)
    
    @property
    def node_params(self) -> DistributedSolverNodeParams:
        return self._node_params

    @property
    def worker_id(self) -> int:
        return self._node_params.RPCParams_.PeerIndex

    @property
    def number_of_workers(self) -> int:
        return self._number_of_workers
    @number_of_workers.setter
    def number_of_workers(self, val: int):
        assert self._number_of_workers is None
        self._number_of_workers = val

    @property
    def graph(self) -> nx.DiGraph:
        assert self._graph is not None
        return self._graph
    @graph.setter
    def graph(self, G: nx.DiGraph):
        self._graph = G
        self._capacities = cpu_array([c_e for _, _, c_e in G.edges(data='capacity')])
        self._indexing = get_edge_indexing(G)
        num_endpoints = G.number_of_nodes()
        self._total_commodity_count = num_endpoints * (num_endpoints - 1)
        assert self._total_commodity_count % self.number_of_workers == 0
        self._assigned_commodity_count = self._total_commodity_count // self.number_of_workers
        self._assigned_commodity_start_id = self.worker_id * self._assigned_commodity_count

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()
    @property
    def capacities(self) -> CPUArray:
        return self._capacities
    @property
    def edge_indexing(self) -> Dict[Tuple[int, int], int]:
        return self._indexing
    @property
    def total_commodity_count(self) -> int:
        return self._total_commodity_count
    @property
    def assigned_commodity_count(self) -> int:
        return self._assigned_commodity_count
    @property
    def assigned_commodity_start_id(self) -> int:
        return self._assigned_commodity_start_id
    @property
    def assigned_commodity_end_id(self) -> int:
        return self._assigned_commodity_start_id + self._assigned_commodity_count
    
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


@dataclass(frozen=True)
class PrettyAddressList(SolverParams):
    Addresses: Tuple[Tuple[str, int]]
    _left_column_share = 0.2


__all__ = [
    'CommunicationBackendBase', 'RPCParams', 
    'DistributedSolverNodeBase', 'DistributedSolverNodeParams',
    'PrettyAddressList'
]