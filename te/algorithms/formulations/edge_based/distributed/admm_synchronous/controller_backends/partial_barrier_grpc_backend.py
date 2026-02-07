import grpc
import asyncio
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Set, Dict
from te.algorithms.array_utils.cpu_utils import CPUArray, BooleanCPUArray, CPUCSRArray, CPUCSCArray, get_global_precision
from .. import SynchADMMSolverParams
from ...base import RPCParams
from ..base import SynchADMMControllerBackendBase
from ...utils import *
from utils.logging import as_fail

import protos.distributed_lp.distributed_lp_pb2 as distributed_lp_messages
from protos.distributed_lp.distributed_lp_pb2_grpc import DistributedADMMSolverStub
from google.protobuf.empty_pb2 import Empty


@dataclass
class PartialBarriergRPCControllerBackendParams(RPCParams):
    Timeout: float = 5
    """Timeout for all asynchronous `wait` calls"""
    BarrierSize: int = 1
    MaxLag: int = 1
    
    def __post_init__(self):
        self.left_column_share = 0.2


class PartialBarriergRPCControllerBackend(SynchADMMControllerBackendBase):
    def __init__(self, rpc_params: PartialBarriergRPCControllerBackendParams):
        super().__init__(rpc_params)

        self._ASYN_PARAM_K = rpc_params.BarrierSize
        self._ASYN_PARAM_TAU = rpc_params.MaxLag

        self._worker_channels: List[grpc.Channel] = [
            grpc.aio.insecure_channel(target=":".join([ip, str(port)]))
                for ip, port in self._rpc_params.Workers
        ]
        self._worker_stubs: List[DistributedADMMSolverStub] = [
            DistributedADMMSolverStub(ch) for ch in self._worker_channels
        ]
        self._event_loop = asyncio.get_event_loop()
        self._clock: int = 0
        """Local clock that keeps track of request timers"""
        self._pending_tasks: Dict[int, Tuple[asyncio.Task, int]] = dict()
        """Maps worker node ID to a pair of task and start clock"""
        self._scatter_tasks: List[asyncio.Task] = []
        """List of scatter tasks to be awaited when reconvening the network"""
        self._arrival_set: Set[int] = set()
        """Set of node IDs that arrived since last iteration"""
        self._means: List[CPUArray] = []
        self._initiated: bool = False
    
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
        return "gRPC-parbar"

    async def is_node_ready(self, worker_id: int) -> bool:
        if not self.is_alive:
            return False
        try:
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

    async def _initialize_worker_nodes(self, solver_params: SynchADMMSolverParams, basis: CPUArray, 
                                       initial_feasible_solution: Union[CPUCSRArray, CPUCSCArray, CPUArray],
                                       in_out_mask: Optional[BooleanCPUArray] = None):
        NUM_WORKERS = self.number_of_workers
        NULL_M = basis
        NUM_COLS = initial_feasible_solution.shape[1]
        assert NUM_COLS % NUM_WORKERS == 0
        CHUNK_INDICES = np.array_split(np.arange(NUM_COLS), NUM_WORKERS)
        X_EK_START_CHUNKS = [initial_feasible_solution[:, chunk[0]:chunk[-1]+1] for chunk in CHUNK_INDICES]
        MASK_EK_CHUNKS = None if in_out_mask is None else np.array_split(in_out_mask, NUM_WORKERS, axis=1)
        WORKERS = self._worker_stubs

        # First, update known means (we need this because we may have to reuse previous means ...)
        self._means = [np.mean(chunk, axis=1) for chunk in X_EK_START_CHUNKS]

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

        # If it exists, send the mask as well
        if MASK_EK_CHUNKS is not None:
            await asyncio.gather(*[
                stub.SetCommodityInOutMask(chunk_big_array(MASK_EK_CHUNKS[i], GRPC_ARRAY_STREAM_MAX_LEN, dtype=bool)) 
                for i, stub in enumerate(WORKERS)
            ])
    
    def initialize_worker_nodes(self, solver_params: SynchADMMSolverParams, basis: CPUArray, 
                                initial_feasible_solution: Union[CPUCSRArray, CPUCSCArray, CPUArray],
                                in_out_mask: Optional[BooleanCPUArray] = None):
        self._event_loop.run_until_complete(self._initialize_worker_nodes(solver_params, basis, initial_feasible_solution, in_out_mask))
    
    async def _update_demands(self, updated_feasible_solution: CPUArray):
        X_EK_START_CHUNKS = np.array_split(updated_feasible_solution, self.number_of_workers, axis=1)
        WORKERS = self._worker_stubs
        await asyncio.gather(*[
            stub.SetInitialFeasibleSolution(chunk_big_array(X_EK_START_CHUNKS[i], GRPC_ARRAY_STREAM_MAX_LEN))
            for i, stub in enumerate(WORKERS)
        ])
    
    def update_demands(self, updated_feasible_solution: CPUArray):
        self._event_loop.run_until_complete(self._update_demands(updated_feasible_solution))
    
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

    async def _stub_net_update(self, node_id: int, request: distributed_lp_messages.NetworkUpdateRequest):
        return await self._worker_stubs[node_id].DoNetworkUpdate(request)

    async def _partial_barrier_update(self, message: distributed_lp_messages.NetworkUpdateRequest):
        # Broadcast to any non-pending node and increment the local clock
        for node_id in range(self.number_of_workers):
            if node_id not in self._pending_tasks:
                task = asyncio.create_task(self._stub_net_update(node_id, message))
                self._pending_tasks[node_id] = (task, int(self._clock))
        self._clock += 1
        
        # Gather finished responses since last time
        current_batch_responses: List[Tuple[distributed_lp_messages.NetworkUpdateResponse, int]] = []
        while self.is_alive:
            finished_this_loop = [
                node_id for node_id, (task, _) in self._pending_tasks.items() if task.done()
            ]

            for node_id in finished_this_loop:
                task, _ = self._pending_tasks.pop(node_id)
                try:
                    res = await task
                    if res is not None:
                        current_batch_responses.append((res, node_id))
                except Exception as e:
                    print(as_fail(f'Failure while awaiting finished update on node {node_id}: {e}'))

            # Evaluate Exit Conditions
            # Condition A: Count >= n
            # Condition B: Max age <= T
            stale_nodes = [
                node_id for node_id, (_, start_time) in self._pending_tasks.items()
                if (self._clock - start_time) > self._ASYN_PARAM_TAU
            ]
            # if len(stale_nodes) > 0:
            #     print(f"STALE NODES: {stale_nodes}")

            if len(current_batch_responses) >= self._ASYN_PARAM_K and not stale_nodes:
                break

            # Efficiency: If we aren't done, wait for the next task to finish
            # We only wait on tasks that are actually running
            if self._pending_tasks:
                tasks_to_watch = [t for t, _ in self._pending_tasks.values()]
                await asyncio.wait(tasks_to_watch, return_when=asyncio.FIRST_COMPLETED)
            else:
                # Safety break if no tasks are left but conditions aren't met
                break
        
        runtimes = []
        for response, node_id in current_batch_responses:
            runtimes.append(response.runtime_ns)
            self._means[node_id] = serialized_message_to_array(response.means)
            self._arrival_set.add(node_id)
        return max(runtimes), np.mean(self._means, axis=0)
    
    def do_network_update(self, epoch: int):
        message = distributed_lp_messages.NetworkUpdateRequest(epoch=epoch)
        res = self._event_loop.run_until_complete(self._partial_barrier_update(message))
        return res
    
    async def _stub_net_reconvene(self, node_id: int, request: distributed_lp_messages.UpdateMessage):
        return await self._worker_stubs[node_id].UpdateWorkerNode(request)
    
    async def _reconvene_network_updates(self, message: distributed_lp_messages.UpdateMessage):
        if not self._initiated:
            await asyncio.gather(*[
                stub.UpdateWorkerNode(message) for stub in self._worker_stubs
            ])
            self._initiated = True
        else:
            pending_scatters = []
            for task in self._scatter_tasks:
                if task.done():
                    try:
                        await task
                    except Exception as e:
                        print(as_fail(f'Failure while awaiting scatter update: {e}'))
                else:
                    pending_scatters.append(task)
            for node_id in self._arrival_set:
                pending_scatters.append(asyncio.create_task(self._stub_net_reconvene(node_id, message)))
            self._scatter_tasks = pending_scatters
            self._arrival_set.clear()
    
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
    
    async def _set_active_commodity_count(self, K: int):
        message = distributed_lp_messages.ActiveCommodityCount(TotalNumberOfCommodities=K)
        await asyncio.gather(*[stub.SetActiveCommodityCount(message) for stub in self._worker_stubs])
    
    def set_active_commodity_count(self, K: int):
        self._event_loop.run_until_complete(self._set_active_commodity_count(K))


import jsonargparse
from ..worker_backends.grpc_backend import gRPCWorkerBackendParams, gRPCWorkerBackend

def add_parbar_grpc_params(parser: jsonargparse.ArgumentParser):
    parser.add_class_arguments(PartialBarriergRPCControllerBackendParams, 'ParBargRPC', 
                               help='Partial Barrier gRPC Communication Backend Parameters')

def parse_parbar_grpc_params(args: jsonargparse.Namespace) -> PartialBarriergRPCControllerBackendParams:
    return PartialBarriergRPCControllerBackendParams.make_from_args(args.ParBargRPC)

def generate_parbar_grpc_worker_params(
    controller_params: PartialBarriergRPCControllerBackendParams
) -> Tuple[List[gRPCWorkerBackendParams], type[gRPCWorkerBackend]]:
    return [gRPCWorkerBackendParams(
        PeerIndex=i, Peers=tuple([addr]),
        NumThreads=1
    ) for i, addr in enumerate(controller_params.Workers)], gRPCWorkerBackend


__all__ = [
    'PartialBarriergRPCControllerBackend', 
    'parse_parbar_grpc_params', 'add_parbar_grpc_params', 'generate_parbar_grpc_worker_params'
]