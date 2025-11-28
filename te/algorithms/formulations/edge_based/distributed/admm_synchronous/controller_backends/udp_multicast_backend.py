import grpc
import socket
import struct
import asyncio
import numpy as np
import te.constants
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass
from utils.exceptions import SolutionInterrupted
from te.algorithms.array_utils.cpu_utils import CPUArray, BooleanCPUArray
from .. import SynchADMMSolverParams
from ...base import RPCParams
from ..base import SynchADMMControllerBackendBase
from ...utils import *
from utils.logging import as_warning

import protos.distributed_lp.distributed_lp_pb2 as distributed_lp_messages
from protos.distributed_lp.distributed_lp_pb2_grpc import DistributedADMMSolverStub
from google.protobuf.empty_pb2 import Empty


@dataclass
class MulticastControllerBackendParams(RPCParams):
    ScatterAddress: str = te.constants.DEFAULT_SCATTER_ADDRESS
    """Multicast group address to scatter to all worker nodes"""
    ScatterPort: int = te.constants.DEFAULT_SCATTER_PORT
    """UDP port to bind for multicasting"""
    TTL: int = 2
    """UDP packet TTL. Should be at least 2."""
    Timeout: float = 1
    """Timeout when waiting for gather phases"""
    UpdateCopyCount: int = 3
    """Number of copies to send when scattering when packet loss is likely"""
    
    def __post_init__(self):
        self.left_column_share = 0.2


class TLVRPCMessages:
    DoInnerLoops = 0x00
    UpdateNetworkNodes = 0x01
    
    HEADER_FORMAT = "!HI"
    HEADER_LENGTH = struct.calcsize(HEADER_FORMAT)

    @classmethod
    def get_packet_header(cls, packet: bytes) -> Optional[bytes]:
        if len(packet) >= cls.HEADER_LENGTH:
            return packet[:cls.HEADER_LENGTH]
    
    @classmethod
    def get_packet_rpc_message(cls, packet: bytes) -> Optional[Tuple[int, int, Any]]:
        """
        Assuming `packet` has at least one finished packet, return the RPC message
        type, the length of the packet and its protobuff representation.
        """
        header = cls.get_packet_header(packet)
        if header is not None:
            message_type, message_len = struct.unpack_from(cls.HEADER_FORMAT, header)
            if len(packet) >= message_len:
                message_serialized = packet[cls.HEADER_LENGTH:message_len]
                if message_type == cls.DoInnerLoops:
                    message = distributed_lp_messages.NetworkUpdateRequest.FromString(message_serialized)
                elif message_type == cls.UpdateNetworkNodes:
                    message = distributed_lp_messages.UpdateMessage.FromString(message_serialized)
                else:
                    raise ValueError(f'Unexpected RPC message type: {message_type}')
                return (message_type, message_len, message)

    @classmethod
    def serialize_do_inner_loop(cls, message: distributed_lp_messages.NetworkUpdateRequest) -> bytes:
        body = message.SerializeToString()
        header = struct.pack(cls.HEADER_FORMAT, cls.DoInnerLoops, len(body) + cls.HEADER_LENGTH)
        return header + body
    
    @classmethod
    def serialize_update_network_nodes(cls, message: distributed_lp_messages.UpdateMessage) -> bytes:
        body = message.SerializeToString()
        header = struct.pack(cls.HEADER_FORMAT, cls.UpdateNetworkNodes, len(body) + cls.HEADER_LENGTH)
        return header + body


class MulticastControllerBackend(SynchADMMControllerBackendBase):
    """
    Implements the UDP multicast backend for screaming and receiving updates from worker nodes.
    Our updates are small, but come at high frequency; We can also assume that the network is
    somewhat stable.
    
    These facts combined, make this backend _MUCH_ more efficient that gRPC. The only real
    price payed here is that since this is plain IP multicast, it is unreliable and we can
    get packet loss.

    Assuming packet loss is _rare_, our best effort solution is to make this backend idempotent
    and thus just retry if things go south. To this end, this backend expects messages to have
    a transaction ID (`xid`) field (in particular `NetworkUpdateRequest` and `UpdateMessage`
    have a `xid` field).
    
    This backend maintains a local `xid` value that starts from 0 (it can be accessed by
    `MulticastControllerBackend.current_xid`). The procedure for using and updating this value is as
    follows:
        
    - When `do_network_update` is called, the current XID is stamped on the request and
    sent out. The workers implicitly ACK the request by giving a `NetworkUpdateResponse`
    message. If not all make it, we timeout and send the update again.
    A worker thus must not accept an update if the `xid` value stamped on the update
    is not larger than its own, and upon seeing a new `xid`, the worker must update its
    own `xid` value to match that.
    As the controller algorithm is synchronous, we will _NOT_ move on until all workers
    have responded.
        
    - When `reconvene_network_updates` is called, we increment our `xid` and then we 
    send out multiple copies of our update instead of just one (we pace them so that 
    it is not a sudden burst).
    We _hope_ that at least one copy reaches each node, as the author is lazy.
    """
    def __init__(self, rpc_params: MulticastControllerBackendParams):
        self._rpc_params = rpc_params

        self._worker_channels: List[grpc.Channel] = [
            grpc.aio.insecure_channel(target=":".join([ip, str(port)]))
                for ip, port in self._rpc_params.Workers
        ]
        self._worker_stubs: List[DistributedADMMSolverStub] = [
            DistributedADMMSolverStub(ch) for ch in self._worker_channels
        ]
        self._scatter_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._scatter_socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self._rpc_params.TTL)
        self._scatter_socket.settimeout(rpc_params.Timeout)
        self.SCATTER_ADDRESS = (rpc_params.ScatterAddress, rpc_params.ScatterPort)
        self._event_loop = asyncio.get_event_loop()

        self._gethered_results = []
        self._gather_done = asyncio.Event()

        self._xid = 0
    
    @classmethod
    def backend_name(self) -> str:
        return "multicast"
    
    @property
    def number_of_nodes(self) -> int:
        return self.number_of_workers
    
    @property
    def current_xid(self) -> int:
        return self._xid
    
    def update_xid(self):
        self._xid = self._xid + 1

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
        self._scatter_socket.close()
        self.killed = True

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
        if not self.is_alive:
            return False
        return self._event_loop.run_until_complete(self._are_network_nodes_ready())

    async def _initialize_worker_nodes(self, solver_params: SynchADMMSolverParams, basis: CPUArray, 
                                       initial_feasible_solution: CPUArray, in_out_mask: Optional[BooleanCPUArray] = None):
        NUM_WORKERS = self.number_of_nodes
        NULL_M = basis
        X_EK_START_CHUNKS = np.array_split(initial_feasible_solution, NUM_WORKERS, axis=1)
        MASK_EK_CHUNKS = None if in_out_mask is None else np.array_split(in_out_mask, NUM_WORKERS, axis=1)
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

        # If it exists, send the mask as well
        if MASK_EK_CHUNKS is not None:
            await asyncio.gather(*[
                stub.SetCommodityInOutMask(chunk_big_array(MASK_EK_CHUNKS[i], GRPC_ARRAY_STREAM_MAX_LEN, dtype=bool)) 
                for i, stub in enumerate(WORKERS)
            ])
    
    def initialize_worker_nodes(self, solver_params: SynchADMMSolverParams, basis: CPUArray, 
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
    
    def do_network_update(self, epoch: int, F_e: Optional[CPUArray] = None):
        self.update_xid()
        message = distributed_lp_messages.NetworkUpdateRequest(epoch=epoch, xid=self.current_xid, 
                                                               F_e=array_to_serialized_message(F_e))
        packet = TLVRPCMessages.serialize_do_inner_loop(message)
        self._scatter_socket.sendto(packet, self.SCATTER_ADDRESS)
        responses = [None for _ in range(self.number_of_nodes)]
        remaining_workers = self.number_of_nodes
        while self.is_alive and remaining_workers > 0:
            try:
                # TODO: For now, assume the response fits in a single packet, but that may not be the case ...
                res = distributed_lp_messages.NetworkUpdateResponse.FromString(
                    self._scatter_socket.recv(40960))
                responses[res.worker_id] = res
                remaining_workers -= 1
            except socket.timeout:
                # This could be a lost packet ...
                # Since the update is idempotent, we can just send it again
                print(as_warning(f"Timeout on network gather ({remaining_workers}/{self.number_of_nodes} workers remaining)"))
                self._scatter_socket.sendto(packet, self.SCATTER_ADDRESS)
        if not self.is_alive:
            raise SolutionInterrupted
        runtimes, serialized_y_bar_chunks = zip(*list([(res.runtime_ns, res.means) for res in responses]))
        return max(runtimes), np.mean([serialized_message_to_array(chunk) for chunk in serialized_y_bar_chunks], axis=0)
    
    def reconvene_network_updates(self, P_bar_t: CPUArray, Y_bar_t: CPUArray, u_t: CPUArray):
        self.update_xid()
        message = distributed_lp_messages.UpdateMessage(
            P_bar_t = array_to_serialized_message(P_bar_t),
            Y_bar_t = array_to_serialized_message(Y_bar_t),
            u_t = array_to_serialized_message(u_t),
            xid = self.current_xid
        )
        for _ in range(self._rpc_params.UpdateCopyCount):
            self._scatter_socket.sendto(TLVRPCMessages.serialize_update_network_nodes(message), self.SCATTER_ADDRESS)

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
            self._scatter_socket.close()

    async def _set_active_commodity_count(self, K: int):
        message = distributed_lp_messages.ActiveCommodityCount(TotalNumberOfCommodities=K)
        await asyncio.gather(*[stub.SetActiveCommodityCount(message) for stub in self._worker_stubs])
    
    def set_active_commodity_count(self, K: int):
        self._event_loop.run_until_complete(self._set_active_commodity_count(K))


import jsonargparse
from ..worker_backends.udp_multicast_backend import MulticastWorkerBackendParams, MulticastWorkerBackend

def add_mcast_params(parser: jsonargparse.ArgumentParser):
    parser.add_class_arguments(MulticastControllerBackendParams, 'Mcast',
                               help='IP Multicast Communication Backend Parameters')

def parse_mcast_params(args: jsonargparse.Namespace) -> MulticastControllerBackendParams:
    return MulticastControllerBackendParams.make_from_args(args)

def generate_mcast_worker_params(
    controller_params: MulticastControllerBackendParams
) -> Tuple[List[MulticastWorkerBackendParams], type[MulticastWorkerBackend]]:
    return [MulticastWorkerBackendParams(
        PeerIndex=i, Peers=tuple([addr]),
        ScatterAddress=controller_params.ScatterAddress,
        ScatterPort=controller_params.ScatterPort,
        TTL=controller_params.TTL
    ) for i, addr in enumerate(controller_params.Workers)], MulticastWorkerBackend


__all__ = [
    'MulticastControllerBackend', 
    'add_mcast_params', 'parse_mcast_params', 'generate_mcast_worker_params'
]