import grpc
import time
import socket
import asyncio
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Iterator, ClassVar, List
from concurrent.futures import ThreadPoolExecutor
from .base import WorkerNodeCommunicationBackendBase, worker_node_communication_backend, worker_communication_backend_params
from .. import (AsynchronousADMMSolverParams, AsynchronousADMMWorkerRPCParams, NetworkUpdate, 
                ControllerUpdate, TLVRPCMessages)
from ...edge_based_distributed_admm.utils import (serialized_message_to_array, array_to_serialized_message,
                                                  rebuild_chunked_array, chunk_big_array,
                                                  GRPC_ARRAY_STREAM_MAX_LEN)

import protos.array.array_pb2 as array_messages
import protos.asynchronous_lp.asynchronous_lp_pb2 as asynchronous_lp_messages
from protos.asynchronous_lp.asynchronous_lp_pb2_grpc import (AsynchronousADMMSolverServicer, 
                                                             add_AsynchronousADMMSolverServicer_to_server)
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
    SocketTimeout: float = 1.0
    
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
        self._update_sem = threading.Semaphore(value=0)
        self._update_queue = deque()
        self._gatherer_loop: Optional[threading.Thread] = None
    
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
    
    def start(self):
        self._server.start()
        self._gatherer_loop = threading.Thread(target=self.gatherer_loop)
        self.is_alive = True
        self._gatherer_loop.start()
    
    def stop(self):
        self.is_alive = False
    
    def die(self):
        self.is_alive = False
        if self._server is not None:
            self._server.stop(1)
        self.killed = True
    
    def gatherer_loop(self):
        buffer = b''
        try:
            while self.is_alive:
                try:
                    packet = self._gather_socket.recv(10240)
                    buffer += packet
                    update = TLVRPCMessages.get_packet_rpc_message(buffer)
                    if update is not None:
                        update_type, consumed_length, request = update
                        assert update_type == TLVRPCMessages.ControllerUpdateType and \
                            isinstance(request, asynchronous_lp_messages.ControllerMessage)
                        if self.worker_id in request.Workers:
                            self._update_queue.append(ControllerUpdate(
                                workers=request.Workers,
                                P_bar_t=serialized_message_to_array(request.P_bar_t),
                                P_bar_sample=serialized_message_to_array(request.P_t_sample_mean),
                                Y_bar_sample=serialized_message_to_array(request.Y_t_sample_mean),
                                u_bar_sample=serialized_message_to_array(request.u_t_sample_mean),
                                sample_size=request.sample_size
                            ))
                        else:
                            self._update_queue.append(ControllerUpdate(
                                workers=request.Workers,
                                P_bar_t=serialized_message_to_array(request.P_bar_t),
                                P_bar_sample=None,
                                Y_bar_sample=None,
                                u_bar_sample=None,
                                sample_size=request.sample_size
                            ))
                        self._update_sem.release()
                        buffer = buffer[consumed_length:]
                except socket.timeout:
                    pass
        except OSError as e:
            print(f'Error in gatherer loop: {e}')
        finally:
            if self._gather_socket:
                self._gather_socket.close()
                self._gather_socket = None
    
    def gather_updates(self, block = False) -> Optional[List[ControllerUpdate]]:
        if not self.is_alive:
            return
        updates = []
        if not self._update_sem.acquire(blocking=False):
            # No update is immediately available
            if block:
                while self.is_alive:
                    # Keep polling the queue for an update
                    if self._update_sem.acquire(timeout=self._rpc_params.QueueTimeout):
                        updates.append(self._update_queue.popleft())
                        break
        else:
            while self._update_sem.acquire(blocking=False) and len(updates) < self.WorkerBatchSize:
                updates.append(self._update_queue.popleft())
        return updates
    
    def wait_until_initialized(self):
        while self.is_alive:
            if not self.is_initialized():
                time.sleep(self._rpc_params.QueueTimeout)
            else:
                return True
        return False
    
    def send_update_to_controller(self, update: NetworkUpdate):
        message = asynchronous_lp_messages.SwitchMessage(
            worker_id=update.worker_id, runtime_ns=update.runtime,
            Y_bar_w=array_to_serialized_message(update.Y_bar_w),
            P_bar_w=array_to_serialized_message(update.P_bar_w),
            u_w=array_to_serialized_message(update.u_w),
            Xo_w=array_to_serialized_message(update.Xo_w)
        )
        self._gather_socket.sendto(TLVRPCMessages.serialize_network_update(message), self.CONTROLLER_ADDRESS)
    
    def wait_for_close(self):
        if not self.killed:
            return self._server.wait_for_termination(timeout=self._rpc_params.QuitTimeout)

    def close(self):
        self.stop()
        if self._gatherer_loop is not None:
            self._gatherer_loop.join()
        if not self.killed:
            self._server.stop(1)
        if self._gather_socket is not None:
            self._gather_socket.close()


class NetworkWorkerNodeListener(AsynchronousADMMSolverServicer):
    def __init__(self, backend: MulticastBackend):
        super().__init__()
        self._backend: WorkerNodeCommunicationBackendBase = backend
        self._id = self._backend.worker_id
    
    def SetInitialFeasibleSolution(self, request_iterator: Iterator[array_messages.Chunk], context):
        self._backend.set_initial_feasible_solution(rebuild_chunked_array(request_iterator))
        return Empty()

    def SetMask(self, request_iterator: Iterator[array_messages.Chunk], context):
        self._backend.set_mask(rebuild_chunked_array(request_iterator))
        return Empty()

    def SetNullSpaceBasis(self, request_iterator: Iterator[array_messages.Chunk], context):
        self._backend.set_null_space_basis(rebuild_chunked_array(request_iterator))
        return Empty()
    
    def RequestChunk(self, request, context):
        return chunk_big_array(self._backend.report_chunk(), GRPC_ARRAY_STREAM_MAX_LEN)
    
    def QueryState(self, request, context):
        return asynchronous_lp_messages.State(ready=self._backend.is_alive)
    
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
