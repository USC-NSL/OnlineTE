import sys
import grpc
import signal
import contextlib
import numpy as np
from typing import Optional, Iterator
from concurrent.futures import ThreadPoolExecutor
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.cpu_utils import cpu_zeros, cpu_array, set_cpu_float_precision
from . import DistributedADMMSolverParams, DistributedADMMWorkerRPCParams
from te.algorithms.sub_algorithms.pgd import do_plain_pgd_with_step_reduction
from .utils import (serialized_message_to_array, array_to_serialized_message,
                    rebuild_chunked_array, chunk_big_array,
                    get_optional_field, GRPC_ARRAY_STREAM_MAX_LEN)
import protos.distributed_lp.distributed_lp_pb2 as distributed_lp_messages
from protos.distributed_lp.distributed_lp_pb2_grpc import DistributedADMMSolverServicer, add_DistributedADMMSolverServicer_to_server
from google.protobuf.empty_pb2 import Empty


class NetworkWorkerNode:
    def __init__(self, worker_id: int, rpc_params: DistributedADMMWorkerRPCParams, 
                 solver_params: Optional[DistributedADMMSolverParams] = None):
        self.worker_id = worker_id
        self._rpc_params = rpc_params
        self._solver_params = solver_params
        self._ready: bool = False

        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None
        self._CHUNK_LEN: Optional[int] = None
        self._NULL_M: Optional[np.ndarray] = None
        self._NNT_M: Optional[np.ndarray] = None
        self._X_ek_start_chunk: Optional[np.ndarray] = None
        self._Y_tk_chunk: Optional[np.ndarray] = None
        self._lambda_ek_chunk: Optional[np.ndarray] = None
        self._Y_bar_t_cached: Optional[np.ndarray] = None
        self._P_bar_t_cached: Optional[np.ndarray] = None
        self._u_t_cached: Optional[np.ndarray] = None

        self._server: Optional[grpc.Server] = None
        self._listener: Optional[NetworkWorkerNodeListener] = None
        self._initialize_listener()

        for sig in ('TERM', 'INT'):
            signal.signal(getattr(signal, 'SIG'+sig), self.int_handler)
        
        self._start_listener()

    def _initialize_listener(self):
        assert self._server is None and self._listener is None
        RPC_PARAMS = self._rpc_params
        IP = RPC_PARAMS.ip
        PORT = RPC_PARAMS.port
        self._server = grpc.server(thread_pool=ThreadPoolExecutor(max_workers=RPC_PARAMS.num_threads))
        self._listener = NetworkWorkerNodeListener(self)
        add_DistributedADMMSolverServicer_to_server(self._listener, self._server)
        addr = ":".join([IP, str(PORT)])
        self._server.add_insecure_port(addr)

        # print(f"[NODE {self.worker_id}] Initialized listener at address {addr}")

    def _start_listener(self):
        assert self._server is not None and self._listener is not None
        self._server.start()
        self._is_active = True

        # print(f"[NODE {self.worker_id}] Listener started")
    
    def _stop_listener(self):
        self._is_active = False
        if self._server is not None:
            self._server.stop(1)
    
    def wait(self):
        if self._server is not None:
            # print(f"[NODE {self.worker_id}] Will now wait for termination.")
            self._server.wait_for_termination()
        # print(f"[NODE {self.worker_id}] Will soon terminate")
    
    def close(self):
        self.int_handler(None, None)

    def int_handler(self, _, __):
        try:
            self._ready = False
            self._stop_listener()
        except:
            pass
    
    def set_initial_feasible_solution(self, X: np.ndarray):
        self._X_ek_start_chunk = X
        self._NUM_EDGES, self._CHUNK_LEN = self._X_ek_start_chunk.shape
    
    def initialize(self, N: np.ndarray, P_bar_t: Optional[np.ndarray] = None, 
                   Y_bar_t: Optional[np.ndarray] = None, u_t: Optional[np.ndarray] = None):
        assert self._X_ek_start_chunk is not None
        CHUNK_LEN = self._CHUNK_LEN
        self._NULL_M = N
        self._NNT_M = N @ N.T
        T = self._NULL_M.shape[1]
        self._T = T
        self._Y_tk_chunk = cpu_zeros((T, CHUNK_LEN))
        self._lambda_ek_chunk = cpu_zeros(self._X_ek_start_chunk.shape)
        self._Y_bar_t_cached: Optional[np.ndarray] = Y_bar_t if Y_bar_t is not None else cpu_zeros((T,))
        self._P_bar_t_cached: Optional[np.ndarray] = P_bar_t if P_bar_t is not None else cpu_zeros((T,))
        self._u_t_cached: Optional[np.ndarray] = u_t if u_t is not None else cpu_zeros((T,))

    def _get_current_C(self) -> np.ndarray:
        Y_TK = self._Y_tk_chunk
        Y_BAR = self._Y_bar_t_cached
        P_BAR = self._P_bar_t_cached
        U_T = self._u_t_cached
        # print(f'Y MEAN (BEFORE): {str(round(np.mean(Y_TK), 8))}')
        # print(f'Y BAR MEAN: {str(round(np.mean(Y_BAR), 8))}')
        # print(f'P BAR MEAN: {str(round(np.mean(P_BAR), 8))}')
        # print(f'U MEAN: {str(round(np.mean(U_T), 8))}')
        return Y_TK - np.expand_dims(Y_BAR - P_BAR + U_T, axis=1)

    def do_inner_loop_update(self, epoch: int) -> np.ndarray:
        GAMMA = self._solver_params.Gamma
        KAPPA = self._solver_params.Kappa
        PGD_ITERS = self._solver_params.PGDIterations
        NULL_M = self._NULL_M
        NNT_M = self._NNT_M
        X_EK_START_CHUNK = self._X_ek_start_chunk
        LAMBDA_EK_CHUNK = self._lambda_ek_chunk
        C_TK_CHUNK = self._get_current_C()
        # self._lambda_ek_chunk, self._Y_tk_chunk = \
        #     do_pgd_with_backtracking(LAMBDA_EK_CHUNK, X_EK_START_CHUNK, NNT_M, NULL_M, C_TK_CHUNK,
        #                              PGD_ITERS, 0.5, 3)
        # print(f'X0 MEAN: {str(round(np.mean(X_EK_START_CHUNK), 8))}')
        # print(f'C MEAN: {str(round(np.mean(C_TK_CHUNK), 8))}')
        # print(f'LAMBDA MEAN (BEFORE): {str(round(np.mean(LAMBDA_EK_CHUNK), 8))}')
        self._lambda_ek_chunk, self._Y_tk_chunk = \
            do_plain_pgd_with_step_reduction(LAMBDA_EK_CHUNK, X_EK_START_CHUNK, NNT_M, NULL_M, C_TK_CHUNK, GAMMA, 
                                             PGD_ITERS, KAPPA, epoch)
        # print(f'Y MEAN: {str(round(np.mean(self._Y_tk_chunk), 8))}')
        # print(f'LAMBDA MEAN (AFTER): {str(round(np.mean(self._lambda_ek_chunk), 8))}')
        return np.mean(self._Y_tk_chunk, axis=1)

    def update_cached_values(self, u_t: np.ndarray, P_bar_t: np.ndarray, Y_bar_t: np.ndarray):
        self._u_t_cached = u_t
        self._P_bar_t_cached = P_bar_t
        self._Y_bar_t_cached = Y_bar_t
    
    def report_chunk(self) -> np.ndarray:
        return cpu_array(self._Y_tk_chunk)
    
    def report_aggregate(self) -> np.ndarray:
        return np.sum(self._X_ek_start_chunk + self._NULL_M @ self._Y_tk_chunk, axis=1)

    @staticmethod
    def spawn_and_wait(worker_id: int, rpc_params: DistributedADMMWorkerRPCParams, 
                       solver_params: Optional[DistributedADMMSolverParams] = None):
        with contextlib.closing(NetworkWorkerNode(worker_id, rpc_params, solver_params)) as worker:
            worker._ready = True
            worker.wait()


class NetworkWorkerNodeListener(DistributedADMMSolverServicer):
    def __init__(self, node: NetworkWorkerNode):
        super().__init__()
        self._worker_node = node
        self._id = node.worker_id
    
    def SetInitialFeasibleSolution(self, request_iterator: Iterator[distributed_lp_messages.Chunk], context):
        self._worker_node.set_initial_feasible_solution(
            X=rebuild_chunked_array(request_iterator)
        )
        return Empty()
    
    def InitializeWorkerNode(self, request: distributed_lp_messages.InitMessage, context):
        self._worker_node.initialize(
            N=serialized_message_to_array(request.NULL_M),
            P_bar_t=serialized_message_to_array(get_optional_field(request, 'P_bar_t')),
            u_t=serialized_message_to_array(get_optional_field(request, 'u_t')),
            Y_bar_t=serialized_message_to_array(get_optional_field(request, 'Y_bar_t'))
        )
        return Empty()
    
    def DoNetworkUpdate(self, request: distributed_lp_messages.NetworkUpdateRequest, context):
        return array_to_serialized_message(self._worker_node.do_inner_loop_update(epoch=request.epoch))
    
    def UpdateWorkerNode(self, request: distributed_lp_messages.UpdateMessage, context):
        self._worker_node.update_cached_values(
            u_t=serialized_message_to_array(request.u_t),
            P_bar_t=serialized_message_to_array(request.P_bar_t),
            Y_bar_t=serialized_message_to_array(request.Y_bar_t)
        )
        return Empty()
    
    def RequestChunk(self, request, context):
        return chunk_big_array(self._worker_node.report_chunk(), GRPC_ARRAY_STREAM_MAX_LEN)
    
    def RequestAggregate(self, request, context):
        return array_to_serialized_message(self._worker_node.report_aggregate())
    
    def QueryState(self, request, context):
        return distributed_lp_messages.State(ready=self._worker_node._ready)
    
    def SetSolverParameters(self, request: distributed_lp_messages.SolverParameters, context):
        new_params = DistributedADMMSolverParams()
        for field in new_params.child_fields.keys():
            setattr(new_params, field, getattr(request, field))
        self._worker_node._solver_params = new_params
        set_global_precision(precision=new_params.Precision)
        set_cpu_float_precision()
        return Empty()
    
    def Close(self, request, context):
        self._worker_node.close()
        return Empty()


if __name__ == '__main__':
    import socket
    from utils.logging import as_fail
    worker_id = int(sys.argv[1])
    if worker_id < 0:
        print(as_fail('Worker ID was not properly initialized!'), file=sys.stderr)
        sys.exit(-1)
    else:
        rpc_params = DistributedADMMWorkerRPCParams(ip=socket.gethostbyname(socket.gethostname()), port=13000 + worker_id)
        NetworkWorkerNode.spawn_and_wait(worker_id, rpc_params)
