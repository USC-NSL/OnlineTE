import grpc
import asyncio
import numpy as np
from typing import List, ClassVar, Optional
from dataclasses import dataclass
from te.algorithms.array_utils.cpu_utils import CPUArray, BooleanCPUArray, IntegerCPUArray
from .. import PathBasedDistributedADMMControllerRPCParams, PathBasedDistributedADMMSolverParams
from .base import (ControllerCommunicationBackendBase, controller_communication_backend,
                   controller_communication_backend_params)
from te.algorithms.formulations.edge_based_distributed_admm.utils import (
    serialized_message_to_array, array_to_serialized_message,
    chunk_big_array, async_rebuild_chunked_array,
    GRPC_ARRAY_STREAM_MAX_LEN)
from te.algorithms.sub_algorithms.paths import path_based_to_edge_based

import protos.path_based_distributed_lp.path_based_distributed_lp_pb2 as distributed_lp_messages
from protos.path_based_distributed_lp.path_based_distributed_lp_pb2_grpc import PathBasedDistributedADMMSolverStub
from google.protobuf.empty_pb2 import Empty


@controller_communication_backend_params
@dataclass
class AsynchronousgRPCControllerBackendParams(PathBasedDistributedADMMControllerRPCParams):
    Backend: ClassVar[str] = 'gRPC-asynchronous'
    Timeout: float = 5
    
    def __post_init__(self):
        self.left_column_share = 0.2


@controller_communication_backend
class AsynchronousgRPCBackend(ControllerCommunicationBackendBase):
    def __init__(self, rpc_params: PathBasedDistributedADMMControllerRPCParams):
        super().__init__()
        self._rpc_params = rpc_params

        self._worker_channels: List[grpc.Channel] = [
            grpc.aio.insecure_channel(target=":".join([ip, str(port)]))
                for ip, port in self._rpc_params.AddressList
        ]
        self._worker_stubs: List[PathBasedDistributedADMMSolverStub] = [
            PathBasedDistributedADMMSolverStub(ch) for ch in self._worker_channels
        ]
        self._event_loop = asyncio.get_event_loop()
    
    def start(self):
        self.is_alive = True
    
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
        return AsynchronousgRPCControllerBackendParams.Backend
    
    @property
    def number_of_nodes(self) -> int:
        return self._rpc_params.NumWorkers

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
    
    def are_network_nodes_ready(self):
        if not self.is_alive:
            return None
        return self._event_loop.run_until_complete(self._are_network_nodes_ready())

    async def _initialize_worker_nodes(self, solver_params: PathBasedDistributedADMMSolverParams, 
                                       alpha: BooleanCPUArray, beta: IntegerCPUArray, demands: CPUArray):
        NUM_WORKERS = self.number_of_nodes
        ALPHA_KET_CHUNKS = np.array_split(alpha, NUM_WORKERS, axis=0)
        BETA_K_CHUNKS = np.array_split(beta, NUM_WORKERS, axis=0)
        D_K_CHUNKS = np.array_split(demands, NUM_WORKERS, axis=0)
        WORKERS = self._worker_stubs

        # Update solver parameters
        params = distributed_lp_messages.SolverParameters(**solver_params.child_fields)
        await asyncio.gather(*[stub.SetSolverParameters(params) for stub in WORKERS])

        # Now, send the path and demand configurations
        await asyncio.gather(*[
            stub.SetAlpha(chunk_big_array(ALPHA_KET_CHUNKS[i], GRPC_ARRAY_STREAM_MAX_LEN, dtype=bool))
            for i, stub in enumerate(WORKERS)
        ])
        await asyncio.gather(*[
            stub.SetBeta(chunk_big_array(BETA_K_CHUNKS[i], GRPC_ARRAY_STREAM_MAX_LEN, dtype=np.int32))
            for i, stub in enumerate(WORKERS)
        ])
        await asyncio.gather(*[
            stub.SetDemands(chunk_big_array(D_K_CHUNKS[i], GRPC_ARRAY_STREAM_MAX_LEN))
            for i, stub in enumerate(WORKERS)
        ])
    
    def initialize_worker_nodes(self, solver_params: PathBasedDistributedADMMSolverParams,
                                alpha: BooleanCPUArray, beta: IntegerCPUArray, demands: CPUArray):
        self._event_loop.run_until_complete(self._initialize_worker_nodes(solver_params, alpha, beta, demands))
    
    async def _update_demands(self, updated_demands: CPUArray):
        D_K_CHUNKS = np.array_split(updated_demands, self.number_of_nodes, axis=0)
        WORKERS = self._worker_stubs
        await asyncio.gather(*[
            stub.SetDemands(chunk_big_array(D_K_CHUNKS[i], GRPC_ARRAY_STREAM_MAX_LEN))
            for i, stub in enumerate(WORKERS)
        ])
    
    def update_demands(self, updated_demands: CPUArray):
        self._event_loop.run_until_complete(self._update_demands(updated_demands))
    
    async def _get_X_ek(self, alpha: CPUArray, demands: CPUArray):
        chunks = await asyncio.gather(*[
            async_rebuild_chunked_array(stub.RequestChunk(Empty()))
            for stub in self._worker_stubs
        ])
        return path_based_to_edge_based(np.hstack(list(chunks)), alpha, demands)

    def get_X_ek(self, alpha: CPUArray, demands: CPUArray):
        return self._event_loop.run_until_complete(self._get_X_ek(alpha, demands))
    
    async def _do_network_update(self, message: distributed_lp_messages.NetworkUpdateRequest):
        responses = await asyncio.gather(*[
            stub.DoNetworkUpdate(message) for stub in self._worker_stubs
        ])
        runtimes, serialized_x_bar_chunks = zip(*list([(res.runtime_ns, res.means) for res in responses]))
        return max(runtimes), np.mean([serialized_message_to_array(chunk) for chunk in serialized_x_bar_chunks], axis=0)
    
    def do_network_update(self, epoch: int):
        message = distributed_lp_messages.NetworkUpdateRequest(epoch=epoch)
        return self._event_loop.run_until_complete(self._do_network_update(message))
    
    async def _reconvene_network_updates(self, message: distributed_lp_messages.UpdateMessage):
        await asyncio.gather(*[
            stub.UpdateWorkerNode(message) for stub in self._worker_stubs
        ])
    
    def reconvene_network_updates(self, X_bar_e: CPUArray, P_bar_e: CPUArray, u_e: CPUArray):
        message = distributed_lp_messages.UpdateMessage(
            X_bar_e = array_to_serialized_message(X_bar_e),
            P_bar_e = array_to_serialized_message(P_bar_e),
            u_e = array_to_serialized_message(u_e)
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
    
    async def _set_active_commodity_count(self, K: int):
        message = distributed_lp_messages.ActiveCommodityCount(TotalNumberOfCommodities=K)
        await asyncio.gather(*[stub.SetActiveCommodityCount(message) for stub in self._worker_stubs])
    
    def set_active_commodity_count(self, K: int):
        self._event_loop.run_until_complete(self._set_active_commodity_count(K))

    async def _reset_inner_dual_variable(self):
        await asyncio.gather(*[stub.ResetInnerDualVariable(Empty()) for stub in self._worker_stubs])

    def reset_inner_dual_variable(self):
        self._event_loop.run_until_complete(self._reset_inner_dual_variable())
