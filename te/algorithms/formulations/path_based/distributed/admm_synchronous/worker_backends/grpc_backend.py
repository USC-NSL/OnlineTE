import grpc
import asyncio
import protos.path_based_distributed_lp.path_based_distributed_lp_pb2 as distributed_lp_messages
from typing import Optional, Iterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from ..base import SynchADMMWorkerBackendBase
from .. import SynchADMMSolverParams
from te.algorithms.formulations.edge_based.distributed.base import RPCParams
from te.algorithms.formulations.edge_based.distributed.utils import *

import protos.array.array_pb2 as array_messages
from protos.path_based_distributed_lp.path_based_distributed_lp_pb2_grpc import DistributedADMMSolverServicer, add_DistributedADMMSolverServicer_to_server
from google.protobuf.empty_pb2 import Empty


@dataclass
class gRPCWorkerBackendParams(RPCParams):
    NumThreads: int = 1
    """Number of threads in the gRPC server pool"""
    
    def __post_init__(self):
        self.left_column_share = 0.2


class gRPCWorkerBackend(SynchADMMWorkerBackendBase):
    def __init__(self, rpc_params: gRPCWorkerBackendParams):
        super().__init__(rpc_params)

        self._server: Optional[grpc.Server] = None
        self._listener: Optional[NetworkWorkerNodeListener] = None

        self._initialize_listener()

    @classmethod
    def backend_name(cls) -> str:
        return 'gRPC'
    
    @property
    def worker_id(self) -> int:
        return self._rpc_params.PeerIndex

    def _initialize_listener(self):
        assert self._server is None and self._listener is None
        RPC_PARAMS: gRPCWorkerBackendParams = self._rpc_params
        IP, PORT = RPC_PARAMS.get_bind_address()
        self._server = grpc.server(thread_pool=ThreadPoolExecutor(max_workers=RPC_PARAMS.NumThreads))
        self._listener = NetworkWorkerNodeListener(self)
        add_DistributedADMMSolverServicer_to_server(self._listener, self._server)
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


class NetworkWorkerNodeListener(DistributedADMMSolverServicer):
    def __init__(self, backend: gRPCWorkerBackend):
        super().__init__()
        self._backend = backend
        self._id = backend.worker_id
    
    def SetAlphaShape(self, request: array_messages.ArrayShape, context):
        self._backend.set_path_mask_shape(tuple(request.dims))
        return Empty()
    def SetAlphaRows(self, request_iterator: Iterator[array_messages.SerializedNumpyArrayMessage], context):
        self._backend.set_path_mask_rows(serialized_message_to_array_list(request_iterator))
        return Empty()
    def SetAlphaCols(self, request_iterator: Iterator[array_messages.SerializedNumpyArrayMessage], context):
        self._backend.set_path_mask_cols(serialized_message_to_array_list(request_iterator))
        return Empty()
    def SetBeta(self, request_iterator: Iterator[array_messages.Chunk], context):
        self._backend.set_path_count(rebuild_chunked_array(request_iterator))
        return Empty()
    def SetCapacities(self, request: array_messages.SerializedNumpyArrayMessage, context):
        self._backend.set_capacities(serialized_message_to_array(request))
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
            serialized_message_to_array(request.sharing_bias)
        )
        return Empty()
    
    def RequestChunk(self, request, context):
        return chunk_big_array(self._backend.report_chunk(), GRPC_ARRAY_STREAM_MAX_LEN)
    
    def QueryState(self, request, context):
        return distributed_lp_messages.State(ready=self._backend.is_alive)
    
    def SetSolverParameters(self, request: distributed_lp_messages.SolverParameters, context):
        new_params = SynchADMMSolverParams()
        for field in new_params.child_fields.keys():
            if request.HasField(field):
                setattr(new_params, field, getattr(request, field))
            else:
                setattr(new_params, field, None)
        self._backend.set_solver_parameters(new_params)
        return Empty()
    
    def Close(self, request, context):
        self._backend.close()
        return Empty()
    
    def SetActiveCommodityCount(self, request: distributed_lp_messages.ActiveCommodityCount, context):
        self._backend.set_active_commodity_count(request.TotalNumberOfCommodities)
        return Empty()
    
    def JITWarmStart(self, request, context):
        self._backend.jit_warmstart()
        return Empty()
