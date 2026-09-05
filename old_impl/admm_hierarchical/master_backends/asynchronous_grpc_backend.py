import grpc
import asyncio
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from te.algorithms.array_utils.cpu_utils import CPUArray, BooleanCPUArray
from .. import HierarchicalADMMSolverParams
from ... import P2PRPCParams
from ..base import MasterCommunicationBackendBase
from ...utils import (serialized_message_to_array, array_to_serialized_message,
                      chunk_big_array, async_rebuild_chunked_array,
                      GRPC_ARRAY_STREAM_MAX_LEN)

import protos.hierarchical_lp.hierarchical_lp_pb2 as hierarchical_lp_messages
from protos.hierarchical_lp.hierarchical_lp_pb2_grpc import (MasterSolverStub, AsynchronousDomainNotificationServicer,
                                                             add_AsynchronousDomainNotificationServicer_to_server)
from google.protobuf.empty_pb2 import Empty


@dataclass
class AsynchronousgRPCMasterBackendParams(P2PRPCParams):
    Timeout: float = 5
    Threads: int = 1
    
    def __post_init__(self):
        self.left_column_share = 0.2


class AsynchronousgRPCMasterBackend(MasterCommunicationBackendBase):
    def __init__(self, rpc_params: AsynchronousgRPCMasterBackendParams):
        self._rpc_params = rpc_params
        
        PEER_ADDRS = self._rpc_params.Peers
        self._domain_channels: List[grpc.Channel] = [
            grpc.aio.insecure_channel(target=":".join([PEER_ADDRS[i][0], str(PEER_ADDRS[i][1])]))
                for i in range(len(PEER_ADDRS)) if i != self._rpc_params.Index
        ]
        self._domain_stubs: List[MasterSolverStub] = [
            MasterSolverStub(ch) for ch in self._domain_channels
        ]
        self._server: Optional[grpc.Server] = None
        self._listener: Optional[AsynchronousDomainNotificationListener] = None
        self._event_loop = asyncio.get_event_loop()

    def _initialize_listener(self):
        assert self._server is None and self._listener is None
        RPC_PARAMS = self._rpc_params
        IP, PORT = RPC_PARAMS.Peers[self.peer_id]
        self._server = grpc.server(thread_pool=ThreadPoolExecutor(max_workers=RPC_PARAMS.Threads))
        self._listener = AsynchronousDomainNotificationListener(self)
        add_AsynchronousDomainNotificationServicer_to_server(self._listener, self._server)
        addr = ":".join([IP, str(PORT)])
        self._server.add_insecure_port(addr)
    
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
    def peer_id(self) -> int:
        return self._rpc_params.Index
    
    @property
    def number_of_peers(self) -> int:
        return len(self._rpc_params.Peers)
    
    @property
    def number_of_domains(self) -> int:
        return self.number_of_peers - 1

    async def is_domain_ready(self, domain_id: int) -> bool:
        if not self.is_alive:
            return False
        try:
            res: hierarchical_lp_messages.State = await self._domain_stubs[domain_id].QueryState(Empty())
            return res.ready
        except grpc.aio._call.AioRpcError:
            return False
    
    async def _are_peers_ready(self) -> bool:
        results = await asyncio.gather(*[self.is_domain_ready(i) for i in range(self.number_of_domains)])
        return all(results)
    
    def are_all_peers_reachable(self):
        return self._event_loop.run_until_complete(self._are_peers_ready())
    
    def are_all_workers_ready(self) -> bool:
        # Master node has no workers, so this returns `True` regardless
        return True

    async def _initialize_domain_peers(self, solver_params: HierarchicalADMMSolverParams, 
                                       basis: CPUArray, initial_feasible_solution: List[CPUArray], 
                                       in_out_mask: List[BooleanCPUArray]):
        DOMAINS = self._domain_stubs

        # Update solver parameters
        params = hierarchical_lp_messages.SolverParameters(**solver_params.child_fields)
        await asyncio.gather(*[stub.SetSolverParameters(params) for stub in DOMAINS])
        await asyncio.gather(*[
            stub.SetInitialFeasibleSolution(chunk_big_array(initial_feasible_solution[i], GRPC_ARRAY_STREAM_MAX_LEN))
            for i, stub in enumerate(DOMAINS)
        ])
        await asyncio.gather(*[
            stub.SetNullSpaceBasis(chunk_big_array(basis, GRPC_ARRAY_STREAM_MAX_LEN))
            for stub in DOMAINS
        ])
        await asyncio.gather(*[
            stub.SetCommodityInOutMask(chunk_big_array(in_out_mask[i], GRPC_ARRAY_STREAM_MAX_LEN, dtype=bool)) 
            for i, stub in enumerate(DOMAINS)
        ])
    
    def initialize_domain_peers(self, solver_params: HierarchicalADMMSolverParams, basis: CPUArray, 
                                initial_feasible_solution: List[CPUArray], 
                                in_out_mask: List[BooleanCPUArray]):
        self._event_loop.run_until_complete(self._initialize_domain_peers(solver_params, basis, initial_feasible_solution, in_out_mask))
    
    async def _collect_X_ek(self) -> CPUArray:
        return await asyncio.gather(*[
            async_rebuild_chunked_array(stub.RequestXEK(Empty()))
            for stub in self._domain_stubs
        ])

    def collect_X_ek(self) -> CPUArray:
        return self._event_loop.run_until_complete(self._collect_X_ek())

    async def _get_admm_consensus_variables(self) -> Tuple[CPUArray, CPUArray]:
        ls: List[hierarchical_lp_messages.DomainConsensusVariablesMessage] = await \
            asyncio.gather(*[stub.RequestConsensusVariables(Empty()) for stub in self._domain_stubs])
        primal_pair_list: Tuple[List[CPUArray], List[CPUArray]] = \
            list(zip(*[(serialized_message_to_array(item.Y_bar_t), serialized_message_to_array(item.P_bar_t)) for item in ls]))
        return np.hstack(primal_pair_list[0]), np.hstack(primal_pair_list[1])

    def get_admm_consensus_variables(self) -> Tuple[CPUArray, CPUArray]:
        return self._event_loop.run_until_complete(self._get_admm_consensus_variables())
    
    async def _notify_arrived_peers(self, arrival_list: List[Tuple[int, CPUArray]], z_de: CPUArray):
        DOMAINS = self._domain_stubs
        await asyncio.gather(*[
            DOMAINS[domain_id].NotifyArrivedDomain(
                hierarchical_lp_messages.DomainUpdateMessage(
                    array_to_serialized_message(z_de[domain_id, :])
                )
            ) for domain_id, _ in arrival_list
        ])

    def notify_arrived_peers(self, arrival_list: List[Tuple[int, CPUArray]], z_de: CPUArray):
        self._event_loop.run_until_complete(arrival_list, z_de)

    async def _close_domain(self, domain_id: int):
        try:
            await self._domain_stubs[domain_id].Close(Empty())
        except:
            pass
    
    async def aclose(self):
        await asyncio.wait(
            [asyncio.create_task(self._close_domain(i)) for i in range(self.number_of_domains)],
            timeout=self._rpc_params.Timeout
        )
    
    def close_domains(self):
        if not self.killed:
            self._event_loop.run_until_complete(self.aclose())
    
    def close(self):
        self.close_domains()


class AsynchronousDomainNotificationListener(AsynchronousDomainNotificationServicer):
    def __init__(self, backend: AsynchronousgRPCMasterBackend):
        super().__init__()
        self._backend = backend
    
    def QueryState(self, request, context):
        return hierarchical_lp_messages.State(self._backend.is_alive)

    def EnqueueDomainUpdate(self, request: hierarchical_lp_messages.MasterUpdateMessage, context):
        self._backend.enqueue_domain_update(
            serialized_message_to_array(request.X_dek_sum_de), 
            serialized_message_to_array(request.r_de)
        )
        return Empty()
