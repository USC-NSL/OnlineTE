import grpc
import socket
import asyncio
import numpy as np
from typing import List, ClassVar
from dataclasses import dataclass
from te.algorithms.array_utils.cpu_utils import CPUArray
from .. import DistributedADMMControllerRPCParams, DistributedADMMSolverParams
from .base import (ControllerCommunicationBackendBase, controller_communication_backend,
                   controller_communication_backend_params)
from ..utils import (serialized_message_to_array, array_to_serialized_message,
                     chunk_big_array, async_rebuild_chunked_array,
                     GRPC_ARRAY_STREAM_MAX_LEN)
from utils.logging import as_success, as_fail

import protos.distributed_lp.distributed_lp_pb2 as distributed_lp_messages
from protos.distributed_lp.distributed_lp_pb2_grpc import DistributedADMMSolverStub
from google.protobuf.empty_pb2 import Empty


@controller_communication_backend_params
@dataclass
class MulticastControllerBackendParams(DistributedADMMControllerRPCParams):
    Backend: ClassVar[str] = 'multicast'
    ScatterAddress: str = '224.0.0.10'
    HostName: str = socket.gethostname()
    TTL: int = 2
    ScatterPort: int = 12000
    Timeout: float = 5
    
    def __post_init__(self):
        self.left_column_share = 0.2


@controller_communication_backend
class MulticastBackend(ControllerCommunicationBackendBase):
    def __init__(self, rpc_params: DistributedADMMControllerRPCParams):
        super().__init__()
        self._rpc_params = rpc_params

        self._worker_channels: List[grpc.Channel] = [
            grpc.aio.insecure_channel(target=":".join([ip, str(port)]))
                for ip, port in self._rpc_params.AddressList
        ]
        self._worker_stubs: List[DistributedADMMSolverStub] = [
            DistributedADMMSolverStub(ch) for ch in self._worker_channels
        ]
        self._scatter_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._scatter_socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self._rpc_params.TTL)
        self.SCATTER_ADDRESS = (rpc_params.ScatterAddress, rpc_params.ScatterPort)
        self._event_loop = asyncio.get_event_loop()

        self.is_active = False
        self._gethered_results = []
        self._gather_done = asyncio.Event()

        self._report_queue = asyncio.Queue()
        self._report_task = self._event_loop.create_task(self.report_updates())
        self._xid = 0
    
    @classmethod
    def backend_name(self) -> str:
        return MulticastControllerBackendParams.Backend
    
    @property
    def number_of_nodes(self) -> int:
        return self._rpc_params.NumWorkers
    
    @property
    def current_xid(self) -> int:
        return self._xid
    
    def update_xid(self):
        self._xid = self._xid + 1
    
    async def report_updates(self):
        events_to_print = []
        while True:
            try:
                event = self._report_queue.get_nowait()
                if event is None:
                    if len(events_to_print) > 0:
                        print('\n'.join(events_to_print))
                    break
                events_to_print.append(event)
            except asyncio.QueueEmpty:
                if len(events_to_print) > 0:
                    print('\n'.join(events_to_print))
                    events_to_print.clear()
                await asyncio.sleep(3.0)

    async def is_node_ready(self, worker_id: int) -> bool:
        try:
            res = await self._worker_stubs[worker_id].QueryState(Empty())
            return res.ready
        except grpc.aio._call.AioRpcError:
            return False
    
    async def _are_network_nodes_ready(self) -> bool:
        results = await asyncio.gather(*[self.is_node_ready(i) for i in range(self.number_of_nodes)])
        return all(results)
    
    def are_network_nodes_ready(self):
        return self._event_loop.run_until_complete(self._are_network_nodes_ready())

    async def _initialize_worker_nodes(self, solver_params: DistributedADMMSolverParams, basis: CPUArray, 
                                       initial_feasible_solution: CPUArray):
        NUM_WORKERS = self.number_of_nodes
        NULL_M = basis
        X_EK_START_CHUNKS = np.array_split(initial_feasible_solution, NUM_WORKERS, axis=1)
        WORKERS = self._worker_stubs

        # Update solver parameters
        params = distributed_lp_messages.SolverParameters(**solver_params.child_fields)
        await asyncio.gather(*[stub.SetSolverParameters(params) for stub in WORKERS])

        # Now, send the initial solution
        await asyncio.gather(*[
            stub.SetInitialFeasibleSolution(chunk_big_array(X_EK_START_CHUNKS[i], GRPC_ARRAY_STREAM_MAX_LEN))
            for i, stub in enumerate(WORKERS)
        ])

        # Finally, the rest of the things to know ...
        await asyncio.gather(*[
            stub.SetNullSpaceBasis(chunk_big_array(NULL_M, GRPC_ARRAY_STREAM_MAX_LEN))
            for stub in WORKERS
        ])
    
    def initialize_worker_nodes(self, solver_params: DistributedADMMSolverParams, basis: CPUArray, 
                                initial_feasible_solution: CPUArray):
        self._event_loop.run_until_complete(self._initialize_worker_nodes(solver_params, basis, initial_feasible_solution))
    
    async def _get_X_ek(self, basis: CPUArray, initial_feasible_solution: CPUArray):
        chunks = await asyncio.gather(*[
            async_rebuild_chunked_array(stub.RequestChunk(Empty()))
            for stub in self._worker_stubs
        ])
        return initial_feasible_solution + basis @ np.hstack(list(chunks))

    def get_X_ek(self, basis: CPUArray, initial_feasible_solution: CPUArray):
        return self._event_loop.run_until_complete(self._get_X_ek(basis, initial_feasible_solution))
    
    async def _get_X_ek_sum(self):
        serialized_chunks = await asyncio.gather(*[
            stub.RequestAggregate(Empty()) for stub in self._worker_stubs
        ])
        return np.sum([serialized_message_to_array(chunk) for chunk in serialized_chunks], axis=0)
    
    def get_X_ek_sum(self):
        return self._event_loop.run_until_complete(self._get_X_ek_sum())

    async def _do_network_update(self, message: distributed_lp_messages.NetworkUpdateRequest):
        self._scatter_socket.sendto(message.SerializeToString(), self.SCATTER_ADDRESS)
        responses = [None for _ in range(self.number_of_nodes)]
        for _ in range(self.number_of_nodes):
            res = distributed_lp_messages.NetworkUpdateResponse.FromString(
                await self._event_loop.sock_recv(self._scatter_socket, 10240)
            )
            responses[res.worker_id] = res
            if res.xid == self.current_xid:
                self._report_queue.put_nowait(as_success(f'[NET-UPDATE] XID={res.xid} | WORKER={res.worker_id}'))
            else:
                self._report_queue.put_nowait(as_fail(f'[NET-UPDATE] XID={res.xid} (SHOULD BE {self.current_xid}) | WORKER={res.worker_id}'))
        runtimes, serialized_y_bar_chunks = zip(*list([(res.runtime_ns, res.means) for res in responses]))
        return max(runtimes), np.mean([serialized_message_to_array(chunk) for chunk in serialized_y_bar_chunks], axis=0)
    
    def do_network_update(self, epoch: int):
        message = distributed_lp_messages.NetworkUpdateRequest(epoch=epoch, xid=self.current_xid)
        return self._event_loop.run_until_complete(self._do_network_update(message))
    
    def _reconvene_network_updates(self, message: distributed_lp_messages.UpdateMessage):
        self._scatter_socket.sendto(message.SerializeToString(), self.SCATTER_ADDRESS)
    
    def reconvene_network_updates(self, P_bar_t: CPUArray, Y_bar_t: CPUArray, u_t: CPUArray):
        message = distributed_lp_messages.UpdateMessage(
            P_bar_t = array_to_serialized_message(P_bar_t),
            Y_bar_t = array_to_serialized_message(Y_bar_t),
            u_t = array_to_serialized_message(u_t),
            xid = self.current_xid
        )
        self._reconvene_network_updates(message)
        self.update_xid()

    async def _close_node(self, worker_id: int):
        try:
            await self._worker_stubs[worker_id].Close(Empty())
        except:
            pass
    
    async def aclose(self):
        await self._report_task
        await asyncio.wait(
            [asyncio.create_task(self._close_node(i)) for i in range(self.number_of_nodes)],
            timeout=self._rpc_params.Timeout
        )
    
    def close(self):
        self._report_queue.put_nowait(None)
        async def _helper():
            await self._report_task
        self._event_loop.run_until_complete(_helper())
        self._event_loop.run_until_complete(self.aclose())
        self._scatter_socket.close()
