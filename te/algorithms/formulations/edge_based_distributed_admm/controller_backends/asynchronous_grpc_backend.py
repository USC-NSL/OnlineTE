import grpc
import asyncio
import numpy as np
from typing import List
from te.algorithms.array_utils.cpu_utils import CPUArray
from te.algorithms.formulations.edge_based_distributed_admm import DistributedADMMControllerRPCParams, DistributedADMMSolverParams
from te.algorithms.formulations.edge_based_distributed_admm.controller_backends.base import (ControllerCommunicationBackendBase, 
                                                                                             controller_communication_backend)
from te.algorithms.formulations.edge_based_distributed_admm.utils import (serialized_message_to_array, array_to_serialized_message,
                                                                          chunk_big_array, async_rebuild_chunked_array,
                                                                          GRPC_ARRAY_STREAM_MAX_LEN)

import protos.distributed_lp.distributed_lp_pb2 as distributed_lp_messages
from protos.distributed_lp.distributed_lp_pb2_grpc import DistributedADMMSolverStub
from google.protobuf.empty_pb2 import Empty


@controller_communication_backend
class AsynchronousgRPCBackend(ControllerCommunicationBackendBase):
    def __init__(self, rpc_params: DistributedADMMControllerRPCParams, number_of_nodes: int):
        super().__init__()
        self._rpc_params = rpc_params
        self._number_of_nodes = number_of_nodes

        self._worker_channels: List[grpc.Channel] = [
            grpc.aio.insecure_channel(target=":".join([ip, str(port)]))
                for ip, port in self._rpc_params.addr_list
        ]
        self._worker_stubs: List[DistributedADMMSolverStub] = [
            DistributedADMMSolverStub(ch) for ch in self._worker_channels
        ]
        self._event_loop = asyncio.get_event_loop()
    
    @classmethod
    def backend_name(self) -> str:
        return 'gRPC-asynchronous'
    
    @property
    def number_of_nodes(self) -> int:
        return self._number_of_nodes

    async def is_node_ready(self, worker_id: int) -> bool:
        try:
            res = await self._worker_stubs[worker_id].QueryState(Empty())
            return res.ready
        except grpc._channel._InactiveRpcError:
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
        init = distributed_lp_messages.InitMessage(NULL_M=array_to_serialized_message(NULL_M))
        await asyncio.gather(*[stub.InitializeWorkerNode(init) for stub in WORKERS])
    
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
        serialized_y_bar_chunks = await asyncio.gather(*[
            stub.DoNetworkUpdate(message) for stub in self._worker_stubs
        ])
        return np.mean([serialized_message_to_array(chunk) for chunk in serialized_y_bar_chunks], axis=0)
    
    def do_network_update(self, epoch: int):
        message = distributed_lp_messages.NetworkUpdateRequest(epoch=epoch)
        return self._event_loop.run_until_complete(self._do_network_update(message))
    
    async def _reconvene_network_updates(self, message: distributed_lp_messages.UpdateMessage):
        await asyncio.gather(*[
            stub.UpdateWorkerNode(message) for stub in self._worker_stubs
        ])
    
    def reconvene_network_updates(self, P_bar_t: CPUArray, Y_bar_t: CPUArray, u_t: CPUArray):
        message = distributed_lp_messages.UpdateMessage(
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
            timeout=5
        )
    
    def close(self):
        self._event_loop.run_until_complete(self.aclose())
