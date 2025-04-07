import grpc
import signal
import inspect
import protos.distributed_lp.distributed_lp_pb2 as distributed_lp_messages
from typing import Optional, Iterator
from concurrent.futures import ThreadPoolExecutor
from .base import WorkerNodeCommunicationBackendBase, worker_node_communication_backend
from .. import DistributedADMMSolverParams, DistributedADMMWorkerRPCParams
from ..utils import (serialized_message_to_array, array_to_serialized_message,
                     rebuild_chunked_array, chunk_big_array,
                     GRPC_ARRAY_STREAM_MAX_LEN)
from protos.distributed_lp.distributed_lp_pb2_grpc import DistributedADMMSolverServicer, add_DistributedADMMSolverServicer_to_server
from google.protobuf.empty_pb2 import Empty


@worker_node_communication_backend
class SynchronousgRPCBackend(WorkerNodeCommunicationBackendBase):
    def __init__(self, rpc_params: DistributedADMMWorkerRPCParams):
        super().__init__()
        self._rpc_params = rpc_params

        self._server: Optional[grpc.Server] = None
        self._listener: Optional[NetworkWorkerNodeListener] = None

        self._initialize_listener()

        for sig in ('TERM', 'INT'):
            signal.signal(getattr(signal, 'SIG'+sig), self.int_handler)
        
        self.start()

        self.close = lambda: self.int_handler(None, None)
    
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
        add_DistributedADMMSolverServicer_to_server(self._listener, self._server)
        addr = ":".join([IP, str(PORT)])
        self._server.add_insecure_port(addr)

    def int_handler(self, _, __):
        try:
            self.stop()
        except:
            pass
    
    def start(self):
        assert self._server is not None and self._listener is not None
        self._server.start()
        self.is_worker_node_ready = True

    def stop(self):
        self.is_worker_node_ready = False
        if self._server is not None:
            self._server.stop(1)
    
    def wait(self):
        if self._server is not None:
            self._server.wait_for_termination()


class NetworkWorkerNodeListener(DistributedADMMSolverServicer):
    def __init__(self, backend: SynchronousgRPCBackend):
        super().__init__()
        self._backend = backend
        self._id = backend.worker_id
    
    def SetInitialFeasibleSolution(self, request_iterator: Iterator[distributed_lp_messages.Chunk], context):
        print(f'[START] {inspect.currentframe().f_code.co_name}')
        self._backend.set_initial_feasible_solution(rebuild_chunked_array(request_iterator))
        print(f'[END] {inspect.currentframe().f_code.co_name}')
        return Empty()

    def SetNullSpaceBasis(self, request_iterator: Iterator[distributed_lp_messages.Chunk], context):
        print(f'[START] {inspect.currentframe().f_code.co_name}')
        self._backend.set_null_space_basis(rebuild_chunked_array(request_iterator))
        print(f'[END] {inspect.currentframe().f_code.co_name}')
        return Empty()
    
    def DoNetworkUpdate(self, request: distributed_lp_messages.NetworkUpdateRequest, context):
        print(f'[START] {inspect.currentframe().f_code.co_name}')
        res = array_to_serialized_message(self._backend.do_inner_loop_update(request.epoch))
        print(f'[END] {inspect.currentframe().f_code.co_name}')
        return res
    
    def UpdateWorkerNode(self, request: distributed_lp_messages.UpdateMessage, context):
        print(f'[START] {inspect.currentframe().f_code.co_name}')
        self._backend.update_cached_values(
            serialized_message_to_array(request.u_t),
            serialized_message_to_array(request.P_bar_t),
            serialized_message_to_array(request.Y_bar_t)
        )
        print(f'[END] {inspect.currentframe().f_code.co_name}')
        return Empty()
    
    def RequestChunk(self, request, context):
        print(f'[START] {inspect.currentframe().f_code.co_name}')
        res = chunk_big_array(self._backend.report_chunk(), GRPC_ARRAY_STREAM_MAX_LEN)
        print(f'[END] {inspect.currentframe().f_code.co_name}')
        return res
    
    def RequestAggregate(self, request, context):
        print(f'[START] {inspect.currentframe().f_code.co_name}')
        res = array_to_serialized_message(self._backend.report_aggregate())
        print(f'[END] {inspect.currentframe().f_code.co_name}')
        return res
    
    def QueryState(self, request, context):
        print(f'[START] {inspect.currentframe().f_code.co_name}')
        res = distributed_lp_messages.State(ready=self._backend.is_worker_node_ready)
        print(f'[END] {inspect.currentframe().f_code.co_name}')
        return res
    
    def SetSolverParameters(self, request: distributed_lp_messages.SolverParameters, context):
        print(f'[START] {inspect.currentframe().f_code.co_name}')
        new_params = DistributedADMMSolverParams()
        for field in new_params.child_fields.keys():
            setattr(new_params, field, getattr(request, field))
        self._backend.set_solver_parameters(new_params)
        print(f'[END] {inspect.currentframe().f_code.co_name}')
        return Empty()
    
    def Close(self, request, context):
        print(f'[START] {inspect.currentframe().f_code.co_name}')
        self._backend.close()
        print(f'[END] {inspect.currentframe().f_code.co_name}')
        return Empty()
