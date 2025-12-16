import grpc
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, wait
from utils.logging import as_info
from te.algorithms.array_utils.cpu_utils import CPUArray, BooleanCPUArray
from .. import SynchADMMSolverParams
from ..base import SynchADMMWorkerBackendBase
from ...base import RPCParams
from ...utils import *

import protos.distributed_lp.distributed_lp_pb2 as distributed_lp_messages
from protos.distributed_lp.distributed_lp_pb2_grpc import DistributedADMMSolverStub
from google.protobuf.empty_pb2 import Empty


@dataclass
class SynchronousgRPCControllerBackendParams(RPCParams):
    NumThreads: int = 1
    """Number of threads in the gRPC server pool"""
    
    def __post_init__(self):
        self.left_column_share = 0.2


class SynchronousgRPCControllerBackend(SynchADMMWorkerBackendBase):
    def __init__(self, rpc_params: SynchronousgRPCControllerBackendParams):
        self._rpc_params = rpc_params

        self._worker_channels: List[grpc.Channel] = [
            grpc.insecure_channel(target=":".join([ip, str(port)]))
                for ip, port in self._rpc_params.Workers
        ]
        self._worker_stubs: List[DistributedADMMSolverStub] = [
            DistributedADMMSolverStub(ch) for ch in self._worker_channels
        ]
        self._broadcast_thread_pool = ThreadPoolExecutor(max_workers=rpc_params.NumThreads)
        as_info(f'Will use a broadcast thread pool of size: {rpc_params.NumThreads}')
    
    @classmethod
    def backend_name(self) -> str:
        return "gRPC-synchronous"
    
    @property
    def number_of_nodes(self) -> int:
        return len(self._rpc_params.Workers)

    def start(self):
        self.is_alive = True
        self.killed = False
    
    def stop(self):
        self.is_alive = False
    
    def die(self):
        self._broadcast_thread_pool.shutdown()
        self.killed = True

    def is_node_ready(self, worker_id: int) -> bool:
        try:
            return self._worker_stubs[worker_id].QueryState(Empty()).ready
        except grpc._channel._InactiveRpcError:
            return False
    
    def are_network_nodes_ready(self):
        if not self.is_alive:
            return False
        return all(self._broadcast_thread_pool.map(
            self.is_node_ready, range(self.number_of_nodes)
        ))
    
    def initialize_worker_nodes(self, solver_params: SynchADMMSolverParams, basis: CPUArray, 
                                initial_feasible_solution: CPUArray, in_out_mask: Optional[BooleanCPUArray] = None):
        NUM_WORKERS = self.number_of_nodes
        NULL_M = basis
        X_EK_START_CHUNKS = np.array_split(initial_feasible_solution, NUM_WORKERS, axis=1)
        MASK_EK_CHUNKS = None if in_out_mask is None else np.array_split(in_out_mask, NUM_WORKERS, axis=1)
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
            self._broadcast_thread_pool.submit(stub.SetNullSpaceBasis, 
                                               chunk_big_array(NULL_M, GRPC_ARRAY_STREAM_MAX_LEN))
                for stub in WORKERS
        ]) 

        # If it exists, send the mask as well
        if MASK_EK_CHUNKS is not None:
            wait([
                self._broadcast_thread_pool.submit(
                    stub.SetCommodityInOutMask, chunk_big_array(MASK_EK_CHUNKS[i], GRPC_ARRAY_STREAM_MAX_LEN, dtype=bool) 
                ) for i, stub in enumerate(WORKERS)
            ])

    def update_demands(self, is_sparse: bool, updated_feasible_solution: CPUArray):
        X_EK_START_CHUNKS = np.array_split(updated_feasible_solution, self.number_of_nodes, axis=1)
        WORKERS = self._worker_stubs
        wait([
            self._broadcast_thread_pool.submit(
                stub.SetInitialFeasibleSolution, chunk_big_array(X_EK_START_CHUNKS[i], GRPC_ARRAY_STREAM_MAX_LEN)
            ) for i, stub in enumerate(WORKERS)
        ])

    def get_X_ek(self, is_sparse: bool, basis: CPUArray, initial_feasible_solution: CPUArray):
        chunks = self._broadcast_thread_pool.map(
            lambda stub: rebuild_chunked_array(stub.RequestChunk(Empty())), self._worker_stubs
        )
        if is_sparse:
            return np.hstack(list(chunks))
        else:
            return initial_feasible_solution + basis @ np.hstack(list(chunks))
    
    def get_X_ek_sum(self):
        serialized_chunks = self._broadcast_thread_pool.map(
            lambda stub: stub.RequestAggregate(Empty()), self._worker_stubs)
        return np.sum([serialized_message_to_array(chunk) for chunk in serialized_chunks], axis=0)
    
    def do_network_update(self, epoch: int):
        message = distributed_lp_messages.NetworkUpdateRequest(epoch=epoch)
        responses = self._broadcast_thread_pool.map(
            lambda stub: stub.DoNetworkUpdate(message), self._worker_stubs)
        runtimes, serialized_y_bar_chunks = zip(*list([(res.runtime_ns, res.means) for res in responses]))
        return max(runtimes), np.mean([serialized_message_to_array(chunk) for chunk in serialized_y_bar_chunks], axis=0)
    
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
        if not self.killed:
            wait([
                self._broadcast_thread_pool.submit(lambda worker_id: self._close_node(worker_id), i)
                    for i in range(len(self._worker_stubs))
            ], timeout=5)

    def set_active_commodity_count(self, K: int):
        message = distributed_lp_messages.ActiveCommodityCount(TotalNumberOfCommodities=K)
        wait([
            self._broadcast_thread_pool.submit(stub.SetActiveCommodityCount, message)
                for stub in self._worker_stubs
        ])


import jsonargparse
from ..worker_backends.grpc_backend import gRPCWorkerBackendParams, gRPCWorkerBackend

def add_syn_grpc_params(parser: jsonargparse.ArgumentParser):
    parser.add_class_arguments(SynchronousgRPCControllerBackendParams, 'SyngRPC',
                               help='Synchronous gRPC Communication Backend Parameters')

def parse_syn_grpc_params(args: jsonargparse.Namespace) -> SynchronousgRPCControllerBackendParams:
    return SynchronousgRPCControllerBackendParams.make_from_args(args)

def generate_syn_grpc_worker_params(
    controller_params: SynchronousgRPCControllerBackendParams
) -> Tuple[List[gRPCWorkerBackendParams], type[gRPCWorkerBackend]]:
    return [gRPCWorkerBackendParams(
        PeerIndex=i, Peers=tuple([addr]),
        NumThreads=1
    ) for i, addr in enumerate(controller_params.Workers)], gRPCWorkerBackend


__all__ = [
    'SynchronousgRPCControllerBackend', 
    'parse_syn_grpc_params', 'add_syn_grpc_params', 'generate_syn_grpc_worker_params'
]