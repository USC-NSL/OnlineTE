import grpc
import socket
import struct
import asyncio
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, ClassVar, Optional, Tuple, Any
from te.algorithms.array_utils.cpu_utils import CPUArray
from .. import AsynchronousADMMControllerRPCParams, AsynchronousADMMSolverParams
from .base import (controller_communication_backend, controller_communication_backend_params, 
                   ControllerCommunicationBackendBase, NetworkUpdate)
from ...edge_based_distributed_admm.utils import (serialized_message_to_array, array_to_serialized_message,
                                                  chunk_big_array, async_rebuild_chunked_array,
                                                  GRPC_ARRAY_STREAM_MAX_LEN)

import protos.asynchronous_lp.asynchronous_lp_pb2 as asynchronous_lp_messages
from protos.asynchronous_lp.asynchronous_lp_pb2_grpc import AsynchronousADMMSolverStub
from google.protobuf.empty_pb2 import Empty


@controller_communication_backend_params
@dataclass
class MulticastControllerBackendParams(AsynchronousADMMControllerRPCParams):
    Backend: ClassVar[str] = 'multicast'
    ScatterAddress: str = '224.0.0.10'
    TTL: int = 2
    ScatterPort: int = 12000
    SocketTimeout: float = 1.0
    Hostname: str = socket.gethostname()
    ListenPort: int = 11000
    
    def __post_init__(self):
        self.left_column_share = 0.2


class TLVRPCMessages:
    ControllerUpdate = 0x00
    NetworkUpdate = 0x01
    
    HEADER_FORMAT = "!HI"
    HEADER_LENGTH = struct.calcsize(HEADER_FORMAT)

    @classmethod
    def get_packet_header(cls, packet: bytes) -> Optional[bytes]:
        if len(packet) >= cls.HEADER_LENGTH:
            return packet[:cls.HEADER_LENGTH]
    
    @classmethod
    def get_packet_rpc_message(cls, packet: bytes) -> Optional[Tuple[int, int, Any]]:
        """
        Assuming `packet` has at least one finished packet, return the RPC message
        type, the length of the packet and its protobuff representation.
        """
        header = cls.get_packet_header(packet)
        if header is not None:
            message_type, message_len = struct.unpack_from(cls.HEADER_FORMAT, header)
            if len(packet) >= message_len:
                message_serialized = packet[cls.HEADER_LENGTH:message_len]
                if message_type == cls.NetworkUpdate:
                    message = asynchronous_lp_messages.SwitchMessage.FromString(message_serialized)
                elif message_type == cls.ControllerUpdate:
                    message = asynchronous_lp_messages.ControllerMessage.FromString(message_serialized)
                else:
                    raise ValueError(f'Unexpected update message type: {message_type}')
                return (message_type, message_len, message)

    @classmethod
    def serialize_controller_update(cls, message: asynchronous_lp_messages.ControllerMessage) -> bytes:
        body = message.SerializeToString()
        header = struct.pack(cls.HEADER_FORMAT, cls.ControllerUpdate, len(body) + cls.HEADER_LENGTH)
        return header + body
    
    @classmethod
    def serialize_network_update(cls, message: asynchronous_lp_messages.SwitchMessage) -> bytes:
        body = message.SerializeToString()
        header = struct.pack(cls.HEADER_FORMAT, cls.NetworkUpdate, len(body) + cls.HEADER_LENGTH)
        return header + body


@controller_communication_backend
class MulticastBackend(ControllerCommunicationBackendBase):
    def __init__(self, rpc_params: AsynchronousADMMControllerRPCParams):
        super().__init__()
        self._rpc_params = rpc_params

        self._worker_channels: List[grpc.Channel] = [
            grpc.aio.insecure_channel(target=":".join([ip, str(port)]))
                for ip, port in self._rpc_params.AddressList
        ]
        self._worker_stubs: List[AsynchronousADMMSolverStub] = [
            AsynchronousADMMSolverStub(ch) for ch in self._worker_channels
        ]
        self._scatter_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._scatter_socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self._rpc_params.TTL)
        self._scatter_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.SCATTER_ADDRESS = (rpc_params.ScatterAddress, rpc_params.ScatterPort)
        self.LISTEN_ADDRESS = (socket.gethostbyname(rpc_params.Hostname), rpc_params.ListenPort)
        self._scatter_socket.bind(self.LISTEN_ADDRESS)
        self._scatter_socket.settimeout(rpc_params.SocketTimeout)
        self._event_loop = asyncio.get_event_loop()

        self._update_sem = threading.Semaphore(value=0)
        self._update_queue = deque()
        self._gatherer_loop: Optional[threading.Thread] = None

        self._gathered_workers: Optional[List[int]] = None
    
    @classmethod
    def backend_name(self) -> str:
        return MulticastControllerBackendParams.Backend
    
    @property
    def number_of_nodes(self) -> int:
        return self._rpc_params.NumWorkers
    
    def start(self):
        self._gatherer_loop = threading.Thread(target=self.gatherer_loop)
        self.is_alive = True
        self._gatherer_loop.start()
    
    def stop(self):
        self.is_alive = False
    
    def die(self):
        self.is_alive = False
        try:
            for task in asyncio.all_tasks():
                task.cancel()
        except RuntimeError:
            pass
        self.killed = True

    def stop(self):
        self.is_alive = False
    
    def gatherer_loop(self):
        buffer = b''
        try:
            while self.is_alive:
                try:
                    packet = self._scatter_socket.recv(10240)
                    buffer += packet
                    update = TLVRPCMessages.get_packet_rpc_message(buffer)
                    if update is not None:
                        update_type, consumed_length, request = update
                        assert update_type == TLVRPCMessages.NetworkUpdate and \
                            isinstance(request, asynchronous_lp_messages.SwitchMessage)
                        self._update_queue.append((
                            request.worker_id, 
                            (request.runtime_ns, serialized_message_to_array(request.means))
                        ))
                        self._update_sem.release()
                        buffer = buffer[consumed_length:]
                except socket.timeout:
                    pass
        except OSError as e:
            print(f'Error in gatherer loop: {e}')
        finally:
            if self._scatter_socket:
                self._scatter_socket.close()
                self._scatter_socket = None

    def get_network_updates(self) -> List[Tuple[int, NetworkUpdate]]:
        gathered_updates = []
        workers = []
        while self.is_alive:
            if not self._update_sem.acquire(timeout=self._rpc_params.QueueTimeout):
                continue
            worker, update = self._update_queue.popleft()
            gathered_updates.append((worker, update))
            workers.append(worker)
            if len(gathered_updates) >= self.Upsilon:
                assert self._gathered_workers is None
                self._gathered_workers = workers
                break
        return gathered_updates

    async def is_node_ready(self, worker_id: int) -> bool:
        try:
            res: asynchronous_lp_messages.State = await self._worker_stubs[worker_id].QueryState(Empty())
            return res.ready
        except grpc.aio._call.AioRpcError:
            return False
    
    async def _are_network_nodes_ready(self) -> bool:
        results = await asyncio.gather(
            *[self.is_node_ready(i) for i in range(self.number_of_nodes)],
            loop=self._event_loop
        )
        return all(results)
    
    def are_network_nodes_ready(self):
        return self._event_loop.run_until_complete(self._are_network_nodes_ready())

    async def _initialize_worker_nodes(self, solver_params: AsynchronousADMMSolverParams, basis: CPUArray, 
                                       initial_feasible_solution: CPUArray):
        NUM_WORKERS = self.number_of_nodes
        NULL_M = basis
        X_EK_START_CHUNKS = np.array_split(initial_feasible_solution, NUM_WORKERS, axis=1)
        WORKERS = self._worker_stubs

        params = asynchronous_lp_messages.SolverParameters(**solver_params.child_fields)
        await asyncio.gather(*[stub.SetSolverParameters(params) for stub in WORKERS])
        await asyncio.gather(*[
            stub.SetInitialFeasibleSolution(chunk_big_array(X_EK_START_CHUNKS[i], GRPC_ARRAY_STREAM_MAX_LEN))
            for i, stub in enumerate(WORKERS)
        ])
        await asyncio.gather(*[
            stub.SetNullSpaceBasis(chunk_big_array(NULL_M, GRPC_ARRAY_STREAM_MAX_LEN))
            for stub in WORKERS
        ])
    
    def initialize_worker_nodes(self, solver_params: AsynchronousADMMSolverParams, basis: CPUArray, 
                                initial_feasible_solution: CPUArray):
        self._event_loop.run_until_complete(self._initialize_worker_nodes(solver_params, basis, initial_feasible_solution))

    async def _update_demands(self, updated_feasible_solution: CPUArray):
        X_EK_START_CHUNKS = np.array_split(updated_feasible_solution, self.number_of_nodes, axis=1)
        WORKERS = self._worker_stubs
        await asyncio.gather(*[
            stub.SetInitialFeasibleSolution(chunk_big_array(X_EK_START_CHUNKS[i], GRPC_ARRAY_STREAM_MAX_LEN))
            for i, stub in enumerate(WORKERS)
        ])
    
    def update_demands(self, updated_feasible_solution: CPUArray):
        self._event_loop.run_until_complete(self._update_demands(updated_feasible_solution))
    
    async def _get_X_ek(self, basis: CPUArray, initial_feasible_solution: CPUArray):
        chunks = await asyncio.gather(*[
            async_rebuild_chunked_array(stub.RequestChunk(Empty()))
            for stub in self._worker_stubs
        ])
        return initial_feasible_solution + basis @ np.hstack(list(chunks))

    def get_X_ek(self, basis: CPUArray, initial_feasible_solution: CPUArray):
        try:
            return self._event_loop.run_until_complete(self._get_X_ek(basis, initial_feasible_solution))
        except grpc.aio._call.AioRpcError:
            return None
    
    def update_network_nodes(self, P_bar_t: CPUArray, Y_bar_t: CPUArray, u_t: CPUArray):
        assert self._gathered_workers is not None
        message = asynchronous_lp_messages.ControllerMessage(
            P_bar_t = array_to_serialized_message(P_bar_t),
            Y_bar_t = array_to_serialized_message(Y_bar_t),
            u_t = array_to_serialized_message(u_t),
            Workers = self._gathered_workers
        )
        self._gathered_workers = None
        self._scatter_socket.sendto(TLVRPCMessages.serialize_controller_update(message), self.SCATTER_ADDRESS)

    async def _close_node(self, worker_id: int):
        try:
            await self._worker_stubs[worker_id].Close(Empty())
        except:
            pass
    
    async def aclose(self):
        await asyncio.wait(
            [asyncio.create_task(self._close_node(i)) for i in range(self.number_of_nodes)],
            timeout=self._rpc_params.SocketTimeout
        )
    
    def close(self):
        self.stop()
        self._event_loop.run_until_complete(self.aclose())
        if self._gatherer_loop is not None:
            self._gatherer_loop.join()
        if self._scatter_socket is not None:
            self._scatter_socket.close()

    async def _set_active_commodity_count(self, K: int):
        message = asynchronous_lp_messages.ActiveCommodityCount(TotalNumberOfCommodities=K)
        await asyncio.gather(*[stub.SetActiveCommodityCount(message) for stub in self._worker_stubs])
    
    def set_active_commodity_count(self, K: int):
        self._event_loop.run_until_complete(self._set_active_commodity_count(K))
