import grpc
import queue
import asyncio
import numpy as np
from typing import List, Optional, Iterator
from dataclasses import dataclass
from te.algorithms.array_utils.cpu_utils import CPUArray, BooleanCPUArray
from .. import HierarchicalADMMSolverParams
from ... import P2PRPCParams
from ..base import DomainControllerCommunicationBackendBase
from ...utils import (serialized_message_to_array, array_to_serialized_message,
                      chunk_big_array, async_rebuild_chunked_array, rebuild_chunked_array,
                      GRPC_ARRAY_STREAM_MAX_LEN)

import protos.array.array_pb2 as array_messages
import protos.hierarchical_lp.hierarchical_lp_pb2 as hierarchical_lp_messages
from protos.hierarchical_lp.hierarchical_lp_pb2_grpc import (MasterSolverServicer, add_MasterSolverServicer_to_server, 
                                                             AsynchronousDomainNotificationStub, DomainSolverStub)
from google.protobuf.empty_pb2 import Empty


@dataclass
class AsynchronousgRPCDomainControllerBackendParams(P2PRPCParams):
    MasterPeerID: int = 0
    Threads: int = 1
    Timeout: float = 5
    
    def __post_init__(self):
        self.left_column_share = 0.2


class AsynchronousgRPCDomainControllerBackend(DomainControllerCommunicationBackendBase):
    def __init__(self, rpc_params: AsynchronousgRPCDomainControllerBackendParams):
        self._rpc_params = rpc_params

        self._worker_channels: List[grpc.Channel] = [
            grpc.aio.insecure_channel(target=":".join([ip, str(port)]))
                for ip, port in self._rpc_params.Workers
        ]
        self._worker_stubs: List[DomainSolverStub] = [
            DomainSolverStub(ch) for ch in self._worker_channels
        ]
        # Domain controllers need not maintain connection with other domain controllers ...
        # As such, no peer channels are needed here, except for the master
        MASTER_ADDR = self._rpc_params.Peers[self._rpc_params.MasterPeerID]
        self._master_channel: grpc.Channel = \
            grpc.aio.insecure_channel(target=":".join([
                MASTER_ADDR[0], str(MASTER_ADDR[1])
            ]))
        self._master_stub: AsynchronousDomainNotificationStub = AsynchronousDomainNotificationStub(self._master_channel)
        self._master_update_queue: queue.Queue = queue.Queue()
        self._server: Optional[grpc.Server] = None
        self._listener: Optional[MasterSolverListener] = None
        self._event_loop = asyncio.get_event_loop()
    
    def start(self):
        self.is_alive = True
        self.killed = False
    
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
    
    @classmethod
    def backend_name(self) -> str:
        return "gRPC-asynchronous"
    
    @property
    def number_of_peers(self) -> int:
        return len(self._rpc_params.Peers)
    
    @property
    def number_of_nodes(self) -> int:
        return len(self._rpc_params.Workers)
    
    @property
    def peer_id(self) -> int:
        return self._rpc_params.Index

    async def is_node_ready(self, worker_id: int) -> bool:
        if not self.is_alive:
            return False
        try:
            res = await self._worker_stubs[worker_id].QueryState(Empty())
            return res.ready
        except grpc.aio._call.AioRpcError:
            return False
    
    async def _are_network_nodes_ready(self) -> bool:
        results = await asyncio.gather(*[self.is_node_ready(i) for i in range(self.number_of_nodes)])
        return all(results)
    
    async def _is_master_ready(self) -> bool:
        if not self.is_alive:
            return False
        try:
            res: hierarchical_lp_messages.State = await self._master_stub.QueryState()
            return res.ready
        except grpc.aio._call.AioRpcError:
            return False
    
    def are_all_peers_reachable(self):
        if not self.is_alive:
            return None
        return self._event_loop.run_until_complete(self._is_master_ready())

    async def _update_master(self, X_ek_sum: CPUArray, r_e: CPUArray):
        await self._master_stub.EnqueueDomainUpdate(
            hierarchical_lp_messages.MasterUpdateMessage(
                X_dek_sum_de=array_to_serialized_message(X_ek_sum),
                r_de=array_to_serialized_message(r_e)
            )
        )
    
    def update_master(self, X_ek_sum: CPUArray, r_e: CPUArray):
        self._event_loop.run_until_complete(self._update_master(X_ek_sum, r_e))
    
    def wait_for_master_update(self) -> bool:
        while self.is_alive:
            try:
                Z_DE = self._master_update_queue.get(timeout=self._rpc_params.Timeout)
                self.record_master_update(Z_DE)
                return True
            except queue.Empty:
                pass
        return False
    
    def are_all_workers_ready(self):
        if not self.is_alive:
            return None
        return self._event_loop.run_until_complete(self._are_network_nodes_ready())

    async def _initialize_worker_nodes(self, solver_params: HierarchicalADMMSolverParams, basis: CPUArray, 
                                       initial_feasible_solution: CPUArray, in_out_mask: Optional[BooleanCPUArray] = None):
        NUM_WORKERS = self.number_of_nodes
        NULL_M = basis
        X_EK_START_CHUNKS = np.array_split(initial_feasible_solution, NUM_WORKERS, axis=1)
        MASK_EK_CHUNKS = None if in_out_mask is None else np.array_split(in_out_mask, NUM_WORKERS, axis=1)
        WORKERS = self._worker_stubs

        # Update solver parameters
        params = hierarchical_lp_messages.SolverParameters(**solver_params.child_fields)
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

        # If it exists, send the mask as well
        if MASK_EK_CHUNKS is not None:
            await asyncio.gather(*[
                stub.SetCommodityInOutMask(chunk_big_array(MASK_EK_CHUNKS[i], GRPC_ARRAY_STREAM_MAX_LEN, dtype=bool)) 
                for i, stub in enumerate(WORKERS)
            ])
    
    def initialize_worker_nodes(self, solver_params: hierarchical_lp_messages, basis: CPUArray, 
                                initial_feasible_solution: CPUArray, in_out_mask: Optional[BooleanCPUArray] = None):
        self._event_loop.run_until_complete(self._initialize_worker_nodes(solver_params, basis, initial_feasible_solution, in_out_mask))
    
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
        return self._event_loop.run_until_complete(self._get_X_ek(basis, initial_feasible_solution))
    
    async def _get_X_ek_sum(self):
        serialized_chunks = await asyncio.gather(*[
            stub.RequestAggregate(Empty()) for stub in self._worker_stubs
        ])
        return np.sum([serialized_message_to_array(chunk) for chunk in serialized_chunks], axis=0)
    
    def get_X_ek_sum(self):
        return self._event_loop.run_until_complete(self._get_X_ek_sum())
    
    async def _do_network_update(self, message: hierarchical_lp_messages.NetworkUpdateRequest):
        responses = await asyncio.gather(*[
            stub.DoNetworkUpdate(message) for stub in self._worker_stubs
        ])
        runtimes, serialized_y_bar_chunks = zip(*list([(res.runtime_ns, res.means) for res in responses]))
        return max(runtimes), np.mean([serialized_message_to_array(chunk) for chunk in serialized_y_bar_chunks], axis=0)
    
    def do_network_update(self, epoch: int, F_e: Optional[CPUArray] = None):
        message = hierarchical_lp_messages.NetworkUpdateRequest(epoch=epoch, F_e=array_to_serialized_message(F_e))
        return self._event_loop.run_until_complete(self._do_network_update(message))
    
    async def _reconvene_network_updates(self, message: hierarchical_lp_messages.UpdateMessage):
        await asyncio.gather(*[
            stub.UpdateWorkerNode(message) for stub in self._worker_stubs
        ])
    
    def reconvene_network_updates(self, P_bar_t: CPUArray, Y_bar_t: CPUArray, u_t: CPUArray):
        message = hierarchical_lp_messages.UpdateMessage(
            P_bar_t = array_to_serialized_message(P_bar_t),
            Y_bar_t = array_to_serialized_message(Y_bar_t),
            u_t = array_to_serialized_message(u_t)
        )
        self._event_loop.run_until_complete(self._reconvene_network_updates(message))

    async def _close_node(self, worker_id: int):
        try:
            await self._worker_stubs[worker_id].Close(Empty())
        except:
            pass
    
    async def aclose(self):
        await asyncio.wait(
            [asyncio.create_task(self._close_node(i)) for i in range(self.number_of_nodes)],
            timeout=self._rpc_params.Timeout
        )
    
    def close(self):
        if not self.killed:
            self._event_loop.run_until_complete(self.aclose())


class MasterSolverListener(MasterSolverServicer):
    def __init__(self, backend: AsynchronousgRPCDomainControllerBackend):
        super().__init__()
        self._backend = backend
    
    def QueryState(self, request, context):
        return hierarchical_lp_messages.State(self._backend.is_alive and self._backend.are_all_workers_ready())

    def Close(self, request, context):
        self._backend.close()
        return Empty()

    def SetInitialFeasibleSolution(self, request_iterator: Iterator[array_messages.Chunk], context):
        self._backend.set_initial_feasible_solution(rebuild_chunked_array(request_iterator))
        return Empty()

    def SetNullSpaceBasis(self, request_iterator: Iterator[array_messages.Chunk], context):
        self._backend.set_null_space_basis(rebuild_chunked_array(request_iterator))
        return Empty()
    
    def SetCommodityInOutMask(self, request_iterator: Iterator[array_messages.Chunk], context):
        self._backend.set_commodity_in_out_mask(rebuild_chunked_array(request_iterator))
        return Empty()

    def SetSolverParameters(self, request: hierarchical_lp_messages.SolverParameters, context):
        new_params = HierarchicalADMMSolverParams()
        for field in new_params.child_fields.keys():
            setattr(new_params, field, getattr(request, field))
        self._backend.set_solver_parameters(new_params)
        return Empty()
    
    def RequestXEK(self, request, context):
        return chunk_big_array(self._backend.collect_X_ek(), GRPC_ARRAY_STREAM_MAX_LEN)
    
    def RequestConsensusVariables(self, request, context):
        primal, pair = self._backend.get_admm_consensus_variables()
        return hierarchical_lp_messages.DomainConsensusVariablesMessage(
            Y_bar_t=array_to_serialized_message(primal),
            P_bar_t=array_to_serialized_message(pair)
        )

    def NotifyArrivedDomain(self, request: hierarchical_lp_messages.DomainUpdateMessage, context):
        Z_DE = serialized_message_to_array(request.Z_de)
        self._backend._master_update_queue.put_nowait(Z_DE)
        return Empty()
    