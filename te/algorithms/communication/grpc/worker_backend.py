import grpc
import te.constants
from typing import Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from ..base import RPCParams
from ..worker_backend import WorkerBackendBase
from te.algorithms.base import SolverParams
from array_utils.cpu.grpc_utils import *

import protos.array.array_pb2 as array_messages
import protos.graph.graph_pb2 as graph_messages
import protos.core.core_pb2 as core_messages
from protos.core.core_pb2_grpc import (
    OnlineTECoreServicer, add_OnlineTECoreServicer_to_server
)
from google.protobuf.empty_pb2 import Empty


@dataclass(frozen=True)
class gRPCWorkerBackendParams(RPCParams):
    NumThreads: int = 1
    """Number of threads in the gRPC server pool"""
    _left_column_share = 0.2


class gRPCWorkerBackend[P: SolverParams](WorkerBackendBase[P]):
    def __init__(self,
        rpc_params: gRPCWorkerBackendParams,
        solver_params_cls: SolverParams
    ):
        super().__init__(rpc_params, solver_params_cls)

        self._server: Optional[grpc.Server] = None
        self._listener: Optional[OnlineTEWorkerNodeListener] = None

        self._initialize_listener()

    @classmethod
    def backend_name(cls) -> str:
        return 'gRPC'
    
    @property
    def worker_id(self) -> int:
        return self._rpc_params.PeerIndex

    @property
    def number_of_peers(self):
        return super().number_of_peers

    def _initialize_listener(self):
        assert self._server is None and self._listener is None
        RPC_PARAMS: gRPCWorkerBackendParams = self._rpc_params
        IP, PORT = RPC_PARAMS.get_bind_address()
        self._server = grpc.server(thread_pool=ThreadPoolExecutor(max_workers=RPC_PARAMS.NumThreads))
        self._listener = OnlineTEWorkerNodeListener(self)
        add_OnlineTECoreServicer_to_server(self._listener, self._server)
        addr = ":".join([IP, str(PORT)])
        self._server.add_insecure_port(addr)
    
    def start(self):
        assert self._server is not None and self._listener is not None
        self._server.start()
        self.is_alive = True
        self.killed = False

    def stop(self):
        self.is_alive = False
    
    def die(self):
        self.is_alive = False
        if self._server is not None:
            self._server.stop(1)
        self.killed = True

    def wait(self):
        if self._server is not None:
            self._server.wait_for_termination()
    
    def close(self):
        if not self.killed:
            self._server.stop(1)


class OnlineTEWorkerNodeListener(OnlineTECoreServicer):
    def __init__(self, backend: gRPCWorkerBackend):
        super().__init__()
        self._backend = backend
        self._id = backend.worker_id
    
    def SetTopology(self, request: graph_messages.Topology, context):
        self._backend.set_topology(
            serialized_message_to_graph(request)
        )
        return Empty()

    def SetDemands(self, request: array_messages.Chunk, context):
        X_bar = self._backend.update_demands(
            serialized_message_to_array(request)
        )
        return array_to_serialized_message(X_bar)
    
    def DoNetworkUpdate(self, request: core_messages.NetworkUpdateRequest, context):
        runtime, means = self._backend.do_inner_loop_update(request.epoch)
        return core_messages.NetworkUpdateResponse(
            runtime_ns=runtime, means=array_to_serialized_message(means)
        )
    
    def UpdateWorkerNode(self, request: core_messages.UpdateMessage, context):
        self._backend.update_cached_values(
            serialized_message_to_array(request.sharing_bias)
        )
        return Empty()
    
    def RequestChunk(self, request, context):
        return chunk_big_array(self._backend.report_chunk(), te.constants.GRPC_ARRAY_STREAM_MAX_LEN)
    
    def RequestAggregate(self, request, context):
        return array_to_serialized_message(self._backend.report_aggregate())
    
    def QueryState(self, request, context):
        return core_messages.State(ready=self._backend.is_alive)
    
    def SetSolverParameters(self, request: core_messages.SolverParameters, context):
        num_workers = request.num_workers
        new_params = self._backend.deserialize_solver_params(request.parameters)
        if new_params is None:
            # Deserialization failed! Let the sender know!
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Solver parametrs do not match what was expected")
        else:
            self._backend.set_solver_parameters(new_params, num_workers)
        return Empty()
    
    def Close(self, request, context):
        self._backend.close()
        return Empty()


__all__ = ['gRPCWorkerBackendParams', 'gRPCWorkerBackend', 'OnlineTEWorkerNodeListener']