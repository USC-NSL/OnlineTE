import grpc
import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor, wait
from utils.logging import as_info
from te.algorithms.array_utils.cpu_utils import CPUArray
from te.algorithms.formulations.edge_based_distributed_admm import DistributedADMMControllerRPCParams, DistributedADMMSolverParams
from te.algorithms.formulations.edge_based_distributed_admm.controller_backends.base import (ControllerCommunicationBackendBase, 
                                                                                             controller_communication_backend)
from te.algorithms.formulations.edge_based_distributed_admm.utils import (serialized_message_to_array, array_to_serialized_message,
                                                                          chunk_big_array, rebuild_chunked_array,
                                                                          GRPC_ARRAY_STREAM_MAX_LEN)

import protos.distributed_lp.distributed_lp_pb2 as distributed_lp_messages
from protos.distributed_lp.distributed_lp_pb2_grpc import DistributedADMMSolverStub
from google.protobuf.empty_pb2 import Empty


@controller_communication_backend
class SynchronousgRPCBackend(ControllerCommunicationBackendBase):
    def __init__(self, rpc_params: DistributedADMMControllerRPCParams, number_of_nodes: int):
        super().__init__()
        self._rpc_params = rpc_params
        self._number_of_nodes = number_of_nodes

        self._worker_channels: List[grpc.Channel] = [
            grpc.insecure_channel(target=":".join([ip, str(port)]))
                for ip, port in self._rpc_params.addr_list
        ]
        self._worker_stubs: List[DistributedADMMSolverStub] = [
            DistributedADMMSolverStub(ch) for ch in self._worker_channels
        ]
        self._broadcast_thread_pool = ThreadPoolExecutor(max_workers=rpc_params.num_threads)
        as_info(f'Will use a broadcast thread pool of size: {rpc_params.num_threads}')
    
    @classmethod
    def backend_name(self) -> str:
        return 'gRPC-synchronous'
    
    @property
    def number_of_nodes(self) -> int:
        return self._number_of_nodes

    def is_node_ready(self, worker_id: int) -> bool:
        try:
            return self._worker_stubs[worker_id].QueryState(Empty()).ready
        except grpc._channel._InactiveRpcError:
            return False
    
    def are_network_nodes_ready(self):
        return all(self._broadcast_thread_pool.map(
            self.is_node_ready, range(self.number_of_nodes)
        ))
    
    def initialize_worker_nodes(self, solver_params: DistributedADMMSolverParams, basis: CPUArray, 
                                initial_feasible_solution: CPUArray):
        NUM_WORKERS = self.number_of_nodes
        NULL_M = basis
        X_EK_START_CHUNKS = np.array_split(initial_feasible_solution, NUM_WORKERS, axis=1)
        WORKERS = self._worker_stubs

        # Update solver parameters
        wait([
            self._broadcast_thread_pool.submit(stub.SetSolverParameters, 
                distributed_lp_messages.SolverParameters(**solver_params.child_fields))
                for stub in WORKERS
        ])

        # Now, send the initial solution
        wait([
            self._broadcast_thread_pool.submit(
                stub.SetInitialFeasibleSolution, chunk_big_array(X_EK_START_CHUNKS[i], GRPC_ARRAY_STREAM_MAX_LEN)
            ) for i, stub in enumerate(WORKERS)
        ])
          
        # Finally, the rest of the things to know ...
        wait([
            self._broadcast_thread_pool.submit(stub.InitializeWorkerNode, 
                                               distributed_lp_messages.InitMessage(
                                                   NULL_M=array_to_serialized_message(NULL_M)
                                               ))
                for stub in WORKERS
        ])        

    def get_X_ek(self, basis: CPUArray, initial_feasible_solution: CPUArray):
        chunks = self._broadcast_thread_pool.map(
            lambda stub: rebuild_chunked_array(stub.RequestChunk(Empty())), self._worker_stubs
        )
        return initial_feasible_solution + basis @ np.hstack(list(chunks))
    
    def get_X_ek_sum(self):
        serialized_chunks = self._broadcast_thread_pool.map(
            lambda stub: stub.RequestAggregate(Empty()), self._worker_stubs)
        return np.sum([serialized_message_to_array(chunk) for chunk in serialized_chunks], axis=0)
    
    def do_network_update(self, epoch: int):
        message = distributed_lp_messages.NetworkUpdateRequest(epoch=epoch)
        serialized_y_bar_chunks = self._broadcast_thread_pool.map(
            lambda stub: stub.DoNetworkUpdate(message), self._worker_stubs)
        return np.mean([serialized_message_to_array(chunk) for chunk in serialized_y_bar_chunks], axis=0)
    
    def reconvene_network_updates(self, P_bar_t: CPUArray, Y_bar_t: CPUArray, u_t: CPUArray):
        message = distributed_lp_messages.UpdateMessage(
            P_bar_t = array_to_serialized_message(P_bar_t),
            Y_bar_t = array_to_serialized_message(Y_bar_t),
            u_t = array_to_serialized_message(u_t)
        )
        wait([
            self._broadcast_thread_pool.submit(stub.UpdateWorkerNode, message)
                for stub in self._worker_stubs
        ])

    def _close_node(self, worker_id: int):
        try:
            self._worker_stubs[worker_id].Close(Empty())
        except:
            pass
    
    def close(self):
        wait([
            self._broadcast_thread_pool.submit(lambda worker_id: self._close_node(worker_id), i)
                for i in range(len(self._worker_stubs))
        ], timeout=5)
