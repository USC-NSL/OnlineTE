import grpc
import protos.path_based_distributed_lp.path_based_distributed_lp_pb2 as distributed_lp_messages
from typing import Optional, Iterator
from concurrent.futures import ThreadPoolExecutor
from .base import WorkerNodeCommunicationBackendBase, worker_node_communication_backend
from te.algorithms.formulations.edge_based_distributed_admm.utils import (
    serialized_message_to_array, array_to_serialized_message,
    rebuild_chunked_array, chunk_big_array, get_optional_field,
    GRPC_ARRAY_STREAM_MAX_LEN)
from .. import PathBasedDistributedADMMSolverParams, PathBasedDistributedADMMWorkerRPCParams

import protos.array.array_pb2 as array_messages
from protos.path_based_distributed_lp.path_based_distributed_lp_pb2_grpc import (
    PathBasedDistributedADMMSolverServicer, add_PathBasedDistributedADMMSolverServicer_to_server)
from google.protobuf.empty_pb2 import Empty


@worker_node_communication_backend
class SynchronousgRPCBackend(WorkerNodeCommunicationBackendBase):
    def __init__(self, rpc_params: PathBasedDistributedADMMWorkerRPCParams):
        super().__init__()
        self._rpc_params = rpc_params

        self._server: Optional[grpc.Server] = None
        self._listener: Optional[NetworkWorkerNodeListener] = None

        self._initialize_listener()

    @classmethod
    def backend_name(cls) -> str:
        return 'gRPC-synchronous'
    
    @property
    def worker_id(self) -> int:
        return self._rpc_params.WorkerID

    def _initialize_listener(self):
        assert self._server is None and self._listener is None
        RPC_PARAMS = self._rpc_params
        IP = RPC_PARAMS.IP
        PORT = RPC_PARAMS.Port
        self._server = grpc.server(thread_pool=ThreadPoolExecutor(max_workers=RPC_PARAMS.NumThreads))
        self._listener = NetworkWorkerNodeListener(self)
        add_PathBasedDistributedADMMSolverServicer_to_server(self._listener, self._server)
        addr = ":".join([IP, str(PORT)])
        self._server.add_insecure_port(addr)
    
    def start(self):
        assert self._server is not None and self._listener is not None
        self._server.start()
        self.is_alive = True

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


class NetworkWorkerNodeListener(PathBasedDistributedADMMSolverServicer):
    def __init__(self, backend: SynchronousgRPCBackend):
        super().__init__()
        self._backend = backend
        self._id = backend.worker_id
    
    def SetAlpha(self, request_iterator: Iterator[array_messages.Chunk], context):
        self._backend.set_alpha(rebuild_chunked_array(request_iterator))
        return Empty()
    def SetBeta(self, request_iterator: Iterator[array_messages.Chunk], context):
        self._backend.set_beta(rebuild_chunked_array(request_iterator))
        return Empty()
    def SetDemands(self, request_iterator: Iterator[array_messages.Chunk], context):
        self._backend.set_demands(rebuild_chunked_array(request_iterator))
        return Empty()
    
    def DoNetworkUpdate(self, request: distributed_lp_messages.NetworkUpdateRequest, context):
        runtime, means = self._backend.do_inner_loop_update(request.epoch)
        return distributed_lp_messages.NetworkUpdateResponse(
            runtime_ns=runtime, means=array_to_serialized_message(means)
        )
    
    def UpdateWorkerNode(self, request: distributed_lp_messages.UpdateMessage, context):
        self._backend.update_cached_values(
            serialized_message_to_array(request.X_bar_e),
            serialized_message_to_array(request.P_bar_e),
            serialized_message_to_array(request.u_e)
        )
        return Empty()
    
    def RequestChunk(self, request, context):
        return chunk_big_array(self._backend.report_chunk(), GRPC_ARRAY_STREAM_MAX_LEN)
    
    def RequestAggregate(self, request, context):
        return array_to_serialized_message(self._backend.report_aggregate())
    
    def QueryState(self, request, context):
        return distributed_lp_messages.State(ready=self._backend.is_alive)
    
    def SetSolverParameters(self, request: distributed_lp_messages.SolverParameters, context):
        new_params = PathBasedDistributedADMMSolverParams()
        for field in new_params.child_fields.keys():
            setattr(new_params, field, getattr(request, field))
        self._backend.set_solver_parameters(new_params)
        return Empty()
    
    def Close(self, request, context):
        self._backend.close()
        return Empty()
    
    def SetActiveCommodityCount(self, request: distributed_lp_messages.ActiveCommodityCount, context):
        self._backend.set_active_commodity_count(request.TotalNumberOfCommodities)
        return Empty()
    