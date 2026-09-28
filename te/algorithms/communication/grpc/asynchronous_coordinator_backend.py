"""
gRPC is one way or another always involved in most of our implementations.
We have an IP multicast backend which has much less overhead, but it is not
included in the Artifact Evaluation.

This provides a comfortable base to build solver-specific backends from and
handles the majority of the dity-work for setting up nodes.
"""


import grpc
import asyncio
import numpy as np
import networkx as nx
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from array_utils.cpu.types import *
from array_utils.cpu.grpc_utils import *
from te.algorithms.base import SolverParams, TEObjective
from ..base import RPCParams, OBJECTIVE_TO_ID
from ..coordinator_backend import CoordinatorBackendBase
from .partial_barrier import PartialBarrier

import protos.core.core_pb2 as core_messages
from protos.core.core_pb2_grpc import OnlineTECoreStub
from google.protobuf.empty_pb2 import Empty


@dataclass(frozen=True)
class AsynchronousgRPCCoordinatorBackendParams(RPCParams):
    Timeout: float = 5
    """Timeout for all asynchronous `wait` calls"""
    BarrierSize: Optional[int] = None
    """Partial barrier size"""
    MaxLag: int = 1
    """Partial barrier maximum lag"""
    _left_column_share = 0.2


class AsynchronousgRPCCoordinatorBackend[P: SolverParams](CoordinatorBackendBase[P]):
    def __init__(self,
        rpc_params: AsynchronousgRPCCoordinatorBackendParams,
        solver_params_cls: SolverParams
    ):
        super().__init__(rpc_params, solver_params_cls)

        # Defer channel and stub creation until the async peer check loop
        self._worker_channels: List[Optional[grpc.Channel]] = [None] * self.number_of_workers
        self._worker_stubs: List[Optional[OnlineTECoreStub]] = [None] * self.number_of_workers
        self._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._event_loop)
        # The partial barrier for asynchronous broadcasts
        self._barrier = PartialBarrier[
            core_messages.NetworkUpdateRequest,
            core_messages.NetworkUpdateResponse,
            Tuple[int, CPUArray, Dict[str, float]],
        ](
            number_of_endpoints=self.number_of_workers,
            min_arrival=rpc_params.BarrierSize,
            max_lag=rpc_params.MaxLag,
            event_loop=self._event_loop
        )

    def are_all_peers_reachable(self):
        if len(self._rpc_params.Peers) <= 1:
            return True
        raise NotImplementedError
    
    def start(self):
        self._barrier.start_barrier()
        self.is_alive = True
        self.killed = False
    
    def stop(self):
        self.is_alive = False
        self._barrier.break_barrier()
    
    def die(self):
        self.is_alive = False
        self._barrier.break_barrier()
        try:
            for task in asyncio.all_tasks():
                task.cancel()
        except RuntimeError:
            pass
        self.killed = True
    
    def wait(self):
        pass
    
    @classmethod
    def backend_name(self) -> str:
        return "gRPC-asynchronous"

    async def is_node_ready(self, worker_id: int) -> bool:
        if not self.is_alive:
            return False
        try:
            if self._worker_stubs[worker_id] is None:
                ip, port = self._rpc_params.Workers[worker_id]
                ch = grpc.aio.insecure_channel(target=":".join([ip, str(port)]))
                self._worker_channels[worker_id] = ch
                self._worker_stubs[worker_id] = OnlineTECoreStub(ch)
            
            res = await self._worker_stubs[worker_id].QueryState(Empty())
            return res.ready
        except grpc.aio._call.AioRpcError:
            return False
    
    async def _are_all_workers_reachable(self) -> List[int]:
        results = await asyncio.gather(*[self.is_node_ready(i) for i in range(self.number_of_workers)])
        return [i for i, val in enumerate(results) if not val]
    
    def are_all_workers_reachable(self) -> List[int]:
        if not self.is_alive:
            return None
        return self._event_loop.run_until_complete(self._are_all_workers_reachable())

    async def _initialize_worker_nodes(self,
        solver_params: P,
        graph: nx.DiGraph,
        objecitve: TEObjective
    ):
        WORKERS = self._worker_stubs
        # Set solver parameters
        params = core_messages.SolverParameters()
        params.parameters.Pack(self.serialize_solver_params(solver_params))
        params.num_workers = self.number_of_workers
        params.objective = OBJECTIVE_TO_ID[objecitve]
        await asyncio.gather(*[stub.SetSolverParameters(params) for stub in WORKERS])
        # Set topology
        topology = graph_to_serialized_message(graph)
        await asyncio.gather(*[stub.SetTopology(topology) for stub in WORKERS])
    
    def initialize_worker_nodes(self,
        solver_params: P,
        graph: nx.DiGraph,
        objecitve: TEObjective
    ):
        self._event_loop.run_until_complete(self._initialize_worker_nodes(
            solver_params, graph, objecitve
        ))
    
    async def _update_demands(self, demands: CPUArray):
        WORKERS = self._worker_stubs
        NUM_WORKERS  = self.number_of_workers
        DEMAND_SPLITS = np.array_split(demands, NUM_WORKERS)
        serialized_chunks = await asyncio.gather(*[
            stub.SetDemands(array_to_serialized_message(DEMAND_SPLITS[i]))
                for i, stub in enumerate(WORKERS)
        ])
        return np.mean([serialized_message_to_array(chunk) for chunk in serialized_chunks], axis=0)
    
    def update_demands(self, demands: CPUArray) -> CPUArray:
        return self._event_loop.run_until_complete(self._update_demands(demands))
    
    async def _get_X_ek(self):
        chunks = await asyncio.gather(*[
            async_rebuild_chunked_array(stub.RequestChunk(Empty()))
            for stub in self._worker_stubs
        ])
        return np.hstack(list(chunks))

    def get_X_ek(self):
        return self._event_loop.run_until_complete(self._get_X_ek())

    async def _stub_net_update(self, node_id: int, request: core_messages.NetworkUpdateRequest):
        return await self._worker_stubs[node_id].DoNetworkUpdate(request)

    def _deserialize_update_response(self, message: core_messages.NetworkUpdateResponse) -> Tuple[int, CPUArray, Dict[str, float]]:
        return message.runtime_us, serialized_message_to_array(message.means), message.data

    def do_network_update(
        self,
        sharing_gap: CPUArray,
        sharing_dual: CPUArray,
        sharing_rho: float
    ) -> Tuple[int, CPUArray, Dict[str, float]]:
        message = core_messages.NetworkUpdateRequest(
            sharing_gap=array_to_serialized_message(sharing_gap),
            sharing_dual=array_to_serialized_message(sharing_dual),
            sharing_rho=sharing_rho
        )
        updates = self._barrier.scatter_gather(
            message=message,
            node_coroutine=self._stub_net_update,
            store_operation=self._deserialize_update_response 
        )
        runtimes, means, data = zip(*updates)
        combined_data = defaultdict(float)
        for d in data:
            for k, v in d.items():
                combined_data[k] += v
        return max(runtimes), np.mean(means, axis=0), combined_data

    async def _close_node(self, worker_id: int):
        try:
            await self._worker_stubs[worker_id].Close(Empty())
        except:
            pass
    
    async def aclose(self):
        await asyncio.wait(
            [asyncio.create_task(self._close_node(i)) for i in range(self.number_of_workers)],
            timeout=self._rpc_params.Timeout
        )
    
    def close(self):
        if not self.killed:
            self._event_loop.run_until_complete(self.aclose())


__all__ = ['AsynchronousgRPCCoordinatorBackendParams', 'AsynchronousgRPCCoordinatorBackend']