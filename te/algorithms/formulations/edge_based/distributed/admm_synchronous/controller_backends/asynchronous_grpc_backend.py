import grpc
import asyncio
import numpy as np
import te.constants
import networkx as nx
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from array_utils.cpu.types import *
from array_utils.cpu.grpc_utils import *
from .. import SynchADMMSolverParams
from ...base import RPCParams
from ..base import SynchADMMControllerBackendBase

import protos.distributed_lp.distributed_lp_pb2 as distributed_lp_messages
from protos.distributed_lp.distributed_lp_pb2_grpc import DistributedADMMSolverStub
from google.protobuf.empty_pb2 import Empty


@dataclass(frozen=True)
class AsynchronousgRPCControllerBackendParams(RPCParams):
    Timeout: float = 5
    """Timeout for all asynchronous `wait` calls"""
    _left_column_share = 0.2

class AsynchronousgRPCControllerBackend(SynchADMMControllerBackendBase):
    def __init__(self, rpc_params: AsynchronousgRPCControllerBackendParams):
        super().__init__(rpc_params)

        # Defer channel and stub creation until the async peer check loop
        self._worker_channels: List[Optional[grpc.Channel]] = [None] * self.number_of_workers
        self._worker_stubs: List[Optional[DistributedADMMSolverStub]] = [None] * self.number_of_workers
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
                self._worker_stubs[worker_id] = DistributedADMMSolverStub(ch)
            
            res = await self._worker_stubs[worker_id].QueryState(Empty())
            return res.ready
        except grpc.aio._call.AioRpcError:
            return False
    
    async def _are_all_workers_reachable(self) -> bool:
        results = await asyncio.gather(*[self.is_node_ready(i) for i in range(self.number_of_workers)])
        return all(results)
    
    def are_all_workers_reachable(self):
        if not self.is_alive:
            return None
        return self._event_loop.run_until_complete(self._are_all_workers_reachable())

    async def _initialize_worker_nodes(self, solver_params: SynchADMMSolverParams, graph: nx.DiGraph):
        WORKERS = self._worker_stubs
        # Set solver parameters
        params = distributed_lp_messages.SolverParameters(**solver_params.child_fields)
        params.NumWorkers = self.number_of_workers
        await asyncio.gather(*[stub.SetSolverParameters(params) for stub in WORKERS])
        # Set topology
        topology = graph_to_serialized_message(graph)
        await asyncio.gather(*[stub.SetTopology(topology) for stub in WORKERS])
    
    def initialize_worker_nodes(self,
        solver_params: SynchADMMSolverParams,
        graph: nx.DiGraph
    ):
        self._event_loop.run_until_complete(self._initialize_worker_nodes(solver_params, graph))
    
    async def _update_demands(self, demands: CPUArray):
        WORKERS = self._worker_stubs
        NUM_WORKERS  = self.number_of_workers
        DEMAND_SPLITS = np.array_split(demands, NUM_WORKERS)
        serialized_chunks = await asyncio.gather(*[
            stub.SetDemands(array_to_serialized_message(DEMAND_SPLITS[i]))
                for i, stub in enumerate(WORKERS)
        ])
        return np.sum([serialized_message_to_array(chunk) for chunk in serialized_chunks], axis=0)
    
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
    
    async def _get_X_ek_sum(self):
        serialized_chunks = await asyncio.gather(*[
            stub.RequestAggregate(Empty()) for stub in self._worker_stubs
        ])
        return np.sum([serialized_message_to_array(chunk) for chunk in serialized_chunks], axis=0)
    
    def get_X_ek_sum(self):
        return self._event_loop.run_until_complete(self._get_X_ek_sum())
    
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
    
    def reconvene_network_updates(self, sharing_mean_1: CPUArray, sharing_mean_2: CPUArray, sharing_dual: CPUArray):
        message = distributed_lp_messages.UpdateMessage(
            sharing_bias=array_to_serialized_message(
                sharing_mean_1 - sharing_mean_2 + sharing_dual
            )
        )
        self._event_loop.run_until_complete(self._reconvene_network_updates(message))

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


import jsonargparse
from ..worker_backends.grpc_backend import gRPCWorkerBackendParams, gRPCWorkerBackend

def add_asyn_grpc_params(parser: jsonargparse.ArgumentParser):
    parser.add_class_arguments(AsynchronousgRPCControllerBackendParams, 'AsyngRPC', 
                               help='Asynchronous gRPC Communication Backend Parameters')

def parse_asyn_grpc_params(
    args: jsonargparse.Namespace,
    controller_addr: Tuple[str, int],
    worker_addr_list: List[Tuple[str, int]],
) -> AsynchronousgRPCControllerBackendParams:
    args.AsyngRPC.Peers = tuple([controller_addr])
    args.AsyngRPC.Workers = tuple(worker_addr_list)
    return AsynchronousgRPCControllerBackendParams.make_from_args(args.AsyngRPC)

def generate_asyn_grpc_worker_params(
    controller_params: AsynchronousgRPCControllerBackendParams
) -> Tuple[List[gRPCWorkerBackendParams], type[gRPCWorkerBackend]]:
    return [gRPCWorkerBackendParams(
        PeerIndex=i, Peers=tuple([addr]),
        NumThreads=1
    ) for i, addr in enumerate(controller_params.Workers)], gRPCWorkerBackend


__all__ = [
    'AsynchronousgRPCControllerBackend', 
    'parse_asyn_grpc_params', 'add_asyn_grpc_params', 'generate_asyn_grpc_worker_params'
]