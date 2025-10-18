import grpc
import socket
import asyncio
import threading
import protos.distributed_lp.distributed_lp_pb2 as distributed_lp_messages
from typing import Optional, Iterator
from concurrent.futures import ThreadPoolExecutor
from .base import WorkerNodeCommunicationBackendBase, worker_node_communication_backend
from .. import DistributedADMMSolverParams, DistributedADMMWorkerRPCParams
from ..utils import (serialized_message_to_array, array_to_serialized_message,
                     rebuild_chunked_array, chunk_big_array, get_optional_field,
                     GRPC_ARRAY_STREAM_MAX_LEN)
from ..controller_backends.udp_multicast_backend import TLVRPCMessages

import protos.array.array_pb2 as array_messages
from protos.distributed_lp.distributed_lp_pb2_grpc import DistributedADMMSolverServicer, add_DistributedADMMSolverServicer_to_server
from google.protobuf.empty_pb2 import Empty


@worker_node_communication_backend
class MulticastBackend(WorkerNodeCommunicationBackendBase):
    def __init__(self, rpc_params: DistributedADMMWorkerRPCParams):
        super().__init__()
        self._rpc_params = rpc_params

        self._server: Optional[grpc.Server] = None
        self._listener: Optional[NetworkWorkerNodeListener] = None
        self._initialize_listener()

        self._gather_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._gather_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # TODO: Don't hardcode these ...
        self._gather_socket.settimeout(5.0)
        self._gather_socket.bind(('224.0.0.10', 12000))
        
        self._handler_loop: Optional[threading.Thread] = None
        self._xid = None
    
    @classmethod
    def backend_name(cls) -> str:
        return 'multicast'
    
    @property
    def worker_id(self) -> int:
        return self._rpc_params.WorkerID

    @property
    def current_xid(self) -> int:
        return self._xid
    
    def update_xid(self):
        self._xid = self._xid + 1

    def _initialize_listener(self):
        assert self._server is None and self._listener is None
        RPC_PARAMS = self._rpc_params
        IP = RPC_PARAMS.IP
        PORT = RPC_PARAMS.Port
        self._server = grpc.server(thread_pool=ThreadPoolExecutor(max_workers=RPC_PARAMS.NumThreads))
        self._listener = NetworkWorkerNodeListener(self)
        add_DistributedADMMSolverServicer_to_server(self._listener, self._server)
        addr = ":".join([IP, str(PORT)])
        self._server.add_insecure_port(addr)
    
    def start(self):
        assert self._server is not None and self._listener is not None
        self._handler_loop = threading.Thread(target=self.gather_updates)
        self._server.start()
        self.is_alive = True
        self._handler_loop.start()

    def stop(self):
        self.is_alive = False
    
    def die(self):
        self.stop()
        if self._server is not None:
            self._server.stop(1)
        self.killed = True
    
    def wait(self):
        if self._server is not None:
            if self._handler_loop:
                self._handler_loop.join()
            if not self.is_alive:
                self._server.stop(1)
            self._server.wait_for_termination()
    
    def gather_updates(self):
        buffer = b''
        try:
            while self.is_alive:
                try:
                    packet, addr = self._gather_socket.recvfrom(10240)
                    buffer += packet
                    update = TLVRPCMessages.get_packet_rpc_message(buffer)
                    if update is not None:
                        update_type, consumed_length, request = update
                        if update_type == TLVRPCMessages.DoInnerLoops:
                            if self.current_xid == None:
                                self._xid = request.xid
                            F_e = serialized_message_to_array(get_optional_field(request, 'F_e'))
                            runtime, means = self.do_inner_loop_update(request.epoch, F_e)
                            response = distributed_lp_messages.NetworkUpdateResponse(
                                worker_id=self.worker_id, runtime_ns=runtime, 
                                means=array_to_serialized_message(means),
                                xid=request.xid
                            )
                            self._gather_socket.sendto(response.SerializeToString(), addr)
                        elif update_type == TLVRPCMessages.UpdateNetworkNodes:
                            self.update_cached_values(
                                serialized_message_to_array(request.u_t),
                                serialized_message_to_array(request.P_bar_t),
                                serialized_message_to_array(request.Y_bar_t)
                            )
                            self.update_xid()
                        else:
                            raise ValueError(f'Unexpected update type: {update_type}')
                        buffer = buffer[consumed_length:]
                except socket.timeout:
                    pass
        except OSError as e:
            print(f'Error in gatherer loop: {e}')
        finally:
            if self._gather_socket:
                self._gather_socket.close()

    def close(self):
        self.stop()
        if not self.killed:
            self._server.stop(1)


class NetworkWorkerNodeListener(DistributedADMMSolverServicer):
    def __init__(self, backend: MulticastBackend):
        super().__init__()
        self._backend = backend
        self._id = backend.worker_id
    
    def SetInitialFeasibleSolution(self, request_iterator: Iterator[array_messages.Chunk], context):
        self._backend.set_initial_feasible_solution(rebuild_chunked_array(request_iterator))
        return Empty()

    def SetNullSpaceBasis(self, request_iterator: Iterator[array_messages.Chunk], context):
        self._backend.set_null_space_basis(rebuild_chunked_array(request_iterator))
        return Empty()

    def SetCommodityInOutMask(self, request_iterator: Iterator[array_messages.Chunk], context):
        self._backend.set_commodity_in_out_mask(rebuild_chunked_array(request_iterator))
        return Empty()
    
    def DoNetworkUpdate(self, request: distributed_lp_messages.NetworkUpdateRequest, context):
        raise NotImplementedError('This should NEVER be invoked!')
    
    def UpdateWorkerNode(self, request: distributed_lp_messages.UpdateMessage, context):
        raise NotImplementedError('This should NEVER be invoked!')
    
    def RequestChunk(self, request, context):
        return chunk_big_array(self._backend.report_chunk(), GRPC_ARRAY_STREAM_MAX_LEN)
    
    def RequestAggregate(self, request, context):
        return array_to_serialized_message(self._backend.report_aggregate())
    
    def QueryState(self, request, context):
        return distributed_lp_messages.State(ready=self._backend.is_alive)
    
    def SetSolverParameters(self, request: distributed_lp_messages.SolverParameters, context):
        new_params = DistributedADMMSolverParams()
        for field in new_params.child_fields.keys():
            setattr(new_params, field, getattr(request, field))
        self._backend.set_solver_parameters(new_params)
        return Empty()
    
    def Close(self, request, context):
        self._backend.close()
        return Empty()

    def SetActiveCommodityCount(self, request: distributed_lp_messages.ActiveCommodityCount, context):
        self._backend.set_active_commodity_count(request.TotalNumberOfCommodities)
        return Empty()
