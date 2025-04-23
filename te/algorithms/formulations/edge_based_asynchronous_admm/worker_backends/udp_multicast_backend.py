import grpc
import signal
import socket
import asyncio
from dataclasses import dataclass
from typing import Optional, Iterator, ClassVar, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from te.algorithms.array_utils.cpu_utils import CPUArray
from .base import WorkerNodeCommunicationBackendBase, worker_node_communication_backend, worker_communication_backend_params
from .. import AsynchronousADMMSolverParams, AsynchronousADMMWorkerRPCParams
from ...edge_based_distributed_admm.utils import (serialized_message_to_array, array_to_serialized_message,
                                                  rebuild_chunked_array, chunk_big_array, get_optional_field,
                                                  GRPC_ARRAY_STREAM_MAX_LEN)
from ..controller_backends.udp_multicast_backend import TLVRPCMessages

import protos.distributed_lp.distributed_lp_pb2 as distributed_lp_messages
from protos.distributed_lp.distributed_lp_pb2_grpc import DistributedADMMSolverServicer, add_DistributedADMMSolverServicer_to_server
from google.protobuf.empty_pb2 import Empty


@worker_communication_backend_params
@dataclass
class MulticastWorkerBackendParams(AsynchronousADMMWorkerRPCParams):
    Backend: ClassVar[str] = 'multicast'
    ScatterAddress: str = '224.0.0.10'
    ControllerHost: str = socket.gethostname()
    ControllerPort: int = 11000
    TTL: int = 2
    ScatterPort: int = 12000
    SocketTimeout: float = 5
    
    def __post_init__(self):
        self.left_column_share = 0.2



@worker_node_communication_backend
class MulticastBackend(WorkerNodeCommunicationBackendBase):
    def __init__(self, rpc_params: AsynchronousADMMWorkerRPCParams):
        super().__init__()
        self._rpc_params = rpc_params

        self._server: Optional[grpc.Server] = None
        self._listener: Optional[NetworkWorkerNodeListener] = None

        for sig in ('TERM', 'INT'):
            signal.signal(getattr(signal, 'SIG'+sig), self.int_handler)
        
        self._gather_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._gather_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._gather_socket.settimeout(rpc_params.SocketTimeout)
        self.SCATTER_ADDRESS = (rpc_params.ScatterAddress, rpc_params.ScatterPort)
        self.CONTROLLER_ADDRESS = (rpc_params.ControllerHost, rpc_params.ControllerPort)
        self._gather_socket.bind(self.SCATTER_ADDRESS)
        self._event_loop = asyncio.get_event_loop()

        self._initialize_listener()
        self._server.start()
        self._update_queue: asyncio.Queue = asyncio.Queue()
        self._gatherer_loop = self._event_loop.create_task(self.gatherer_loop())
        self.is_worker_node_ready = True
    
    @classmethod
    def backend_name(cls) -> str:
        return 'multicast'
    
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
        self.is_worker_node_ready = False

    def stop(self):
        self.is_worker_node_ready = False
        if self._server is not None:
            self._server.stop(1)
    
    async def gatherer_loop(self):
        buffer = b''
        while self.is_worker_node_ready:
            try:
                packet = self._gather_socket.recv(10240)
                buffer += packet
                update = TLVRPCMessages.get_packet_rpc_message(buffer)
                if update is not None:
                    update_type, consumed_length, request = update
                    assert update_type == TLVRPCMessages.UpdateNetworkNodes
                    self._update_queue.put_nowait((
                        serialized_message_to_array(request.u_t),
                        serialized_message_to_array(request.P_bar_t),
                        serialized_message_to_array(request.Y_bar_t)
                    ))
                    buffer = buffer[consumed_length:]
            except socket.timeout:
                pass
    
    def gather_updates(self, block = False) -> Optional[List[Tuple[CPUArray, CPUArray, CPUArray]]]:
        if not self.is_worker_node_ready:
            return
        if self._update_queue.qsize() == 0 and block:
            while self.is_worker_node_ready:
                try:
                    # TODO: Fix the rest!
                    self._event_loop.run_until_complete(asyncio.wait_for(self._update_queue.get(), self._rpc_params.QueueTimeout))
                except asyncio.TimeoutError:
                    pass

    async def aclose(self):
        await self._gatherer_loop
    
    def close(self):
        self.stop()
        self._event_loop.run_until_complete(self.aclose())
        self._gather_socket.close()


class NetworkWorkerNodeListener(DistributedADMMSolverServicer):
    def __init__(self, backend: MulticastBackend):
        super().__init__()
        self._backend = backend
        self._id = backend.worker_id
    
    def SetInitialFeasibleSolution(self, request_iterator: Iterator[distributed_lp_messages.Chunk], context):
        self._backend.set_initial_feasible_solution(rebuild_chunked_array(request_iterator))
        return Empty()

    def SetNullSpaceBasis(self, request_iterator: Iterator[distributed_lp_messages.Chunk], context):
        self._backend.set_null_space_basis(rebuild_chunked_array(request_iterator))
        return Empty()
    
    def DoNetworkUpdate(self, request: distributed_lp_messages.NetworkUpdateRequest, context):
        raise NotImplementedError('This should NEVER be invoked!')
    
    def UpdateWorkerNode(self, request: distributed_lp_messages.UpdateMessage, context):
        raise NotImplementedError('This should NEVER be invoked!')
    
    def RequestChunk(self, request, context):
        return chunk_big_array(self._backend.report_chunk(), GRPC_ARRAY_STREAM_MAX_LEN)
    
    def RequestAggregate(self, request, context):
        return array_to_serialized_message(self._backend.report_aggregate())
    
    def QueryState(self, request, context):
        return distributed_lp_messages.State(ready=self._backend.is_worker_node_ready)
    
    def SetSolverParameters(self, request: distributed_lp_messages.SolverParameters, context):
        new_params = AsynchronousADMMSolverParams()
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
