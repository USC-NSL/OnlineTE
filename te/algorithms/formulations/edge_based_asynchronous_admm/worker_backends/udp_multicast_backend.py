import grpc
import time
import signal
import socket
import asyncio
from dataclasses import dataclass
from typing import Optional, Iterator, ClassVar, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from utils.logging import as_warning
from te.algorithms.array_utils.cpu_utils import CPUArray
from .base import WorkerNodeCommunicationBackendBase, worker_node_communication_backend, worker_communication_backend_params
from .. import AsynchronousADMMSolverParams, AsynchronousADMMWorkerRPCParams
from ...edge_based_distributed_admm.utils import (serialized_message_to_array, array_to_serialized_message,
                                                  rebuild_chunked_array, chunk_big_array,
                                                  GRPC_ARRAY_STREAM_MAX_LEN)
from ..controller_backends.udp_multicast_backend import TLVRPCMessages

import protos.array.array_pb2 as array_messages
import protos.asynchronous_lp.asynchronous_lp_pb2 as asynchronous_lp_messages
from protos.asynchronous_lp.asynchronous_lp_pb2_grpc import AsynchronousADMMSolverServicer, add_AsynchronousADMMSolverServicer_to_server
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
        self._gather_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._gather_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._gather_socket.settimeout(rpc_params.SocketTimeout)
        self.SCATTER_ADDRESS = (rpc_params.ScatterAddress, rpc_params.ScatterPort)
        self.CONTROLLER_ADDRESS = (rpc_params.ControllerHost, rpc_params.ControllerPort)
        self._gather_socket.bind(self.SCATTER_ADDRESS)
        self._event_loop = asyncio.get_event_loop()

        self._initialize_listener()
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
        add_AsynchronousADMMSolverServicer_to_server(self._listener, self._server)
        addr = ":".join([IP, str(PORT)])
        self._server.add_insecure_port(addr)
        self._server.start()

    def int_handler(self, _, __):
        self.stop()
    
    def register_signal_handler(self):
        for sig in ('TERM', 'INT'):
            signal.signal(getattr(signal, 'SIG'+sig), self.int_handler)

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
                    assert update_type == TLVRPCMessages.ControllerUpdate and \
                           isinstance(request, asynchronous_lp_messages.ControllerMessage)
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
        if self._update_queue.qsize() == 0:
            # No update is immediately available
            if block:
                while self.is_worker_node_ready:
                    # Keep polling the queue for an update
                    try:
                        update = self._event_loop.run_until_complete(
                            asyncio.wait_for(self._update_queue.get(), self._rpc_params.QueueTimeout))
                        return [update]
                    except asyncio.TimeoutError:
                        pass
            else:
                return []
        else:
            # Updates are immediately available
            return [
                self._update_queue.get_nowait()
                for _ in range(min(self._update_queue.qsize(), self.WorkerBatchSize))
            ]
    
    def wait_until_initialized(self):
        while self.is_worker_node_ready:
            if not self.is_initialized():
                time.sleep(self._rpc_params.QueueTimeout)
            else:
                return True
        return False
    
    def send_update_to_controller(self, runtime: int, Y_bar: CPUArray):
        message = asynchronous_lp_messages.NetworkUpdateResponse(
            worker_id=self.worker_id, runtime_ns=runtime,
            means=array_to_serialized_message(Y_bar)
        )
        self._gather_socket.sendto(TLVRPCMessages.serialize_network_update(message), self.CONTROLLER_ADDRESS)

    async def aclose(self):
        await self._gatherer_loop
    
    def close(self):
        self.stop()
        self._event_loop.run_until_complete(self.aclose())
        self._gather_socket.close()


class NetworkWorkerNodeListener(AsynchronousADMMSolverServicer):
    def __init__(self, backend: MulticastBackend):
        super().__init__()
        self._backend = backend
        self._id = backend.worker_id
    
    def SetInitialFeasibleSolution(self, request_iterator: Iterator[array_messages.Chunk], context):
        self._backend.set_initial_feasible_solution(rebuild_chunked_array(request_iterator))
        return Empty()

    def SetNullSpaceBasis(self, request_iterator: Iterator[array_messages.Chunk], context):
        self._backend.set_null_space_basis(rebuild_chunked_array(request_iterator))
        return Empty()
    
    def RequestChunk(self, request, context):
        return chunk_big_array(self._backend.report_chunk(), GRPC_ARRAY_STREAM_MAX_LEN)
    
    def RequestAggregate(self, request, context):
        return array_to_serialized_message(self._backend.report_aggregate())
    
    def QueryState(self, request, context):
        return asynchronous_lp_messages.State(ready=self._backend.is_worker_node_ready)
    
    def SetSolverParameters(self, request: asynchronous_lp_messages.SolverParameters, context):
        new_params = AsynchronousADMMSolverParams()
        for field in new_params.child_fields.keys():
            setattr(new_params, field, getattr(request, field))
        self._backend.set_solver_parameters(new_params)
        return Empty()
    
    def Close(self, request, context):
        self._backend.close()
        return Empty()

    def SetActiveCommodityCount(self, request: asynchronous_lp_messages.ActiveCommodityCount, context):
        self._backend.set_active_commodity_count(request.TotalNumberOfCommodities)
        return Empty()
