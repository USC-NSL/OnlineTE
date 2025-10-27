import sys
import time
import signal
import contextlib
import numpy as np
from typing import Optional, Tuple
from utils.logging import as_warning
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.cpu_utils import CPUArray, BooleanCPUArray, cpu_zeros, cpu_array, set_cpu_float_precision
from ..base import WorkerNodeBase
from . import SynchADMMSolverParams
from .base import SynchADMMWorkerBackendBase
from ..base import WorkerNodeParams
from te.algorithms.sub_algorithms.pgd import do_plain_pgd_with_step_reduction, do_nesterov_pgd


class SynchADMMWorkerNode(WorkerNodeBase):
    def __init__(self, params: WorkerNodeParams, solver_params: Optional[SynchADMMSolverParams] = None):
        self.worker_id = params.rpc_params.WorkerID
        self._rpc_params = params.rpc_params
        self._solver_params = solver_params
        self._ready: bool = False
        
        self._K: Optional[int] = None
        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None
        self._CHUNK_LEN: Optional[int] = None
        self._NULL_M: Optional[CPUArray] = None
        self._NNT_M: Optional[CPUArray] = None
        self._MASK_M_chunk: Optional[BooleanCPUArray] = None
        self._X_ek_start_chunk: Optional[CPUArray] = None
        self._Y_bar_t_cached: Optional[CPUArray] = None
        self._P_bar_t_cached: Optional[CPUArray] = None
        self._u_t_cached: Optional[CPUArray] = None
        self._Y_tk_chunk: Optional[CPUArray] = None
        self._lambda_ek_chunk: Optional[CPUArray] = None

        assert issubclass(params.communication_backend, SynchADMMWorkerBackendBase)
        self._backend: SynchADMMWorkerBackendBase = params.communication_backend(params.rpc_params)
        self._backend.start()
        
        self._die_on_next_int = False
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.die)

    def stop(self, _, __):
        if self._die_on_next_int:
            signal.raise_signal(signal.SIGTERM)
        else:
            print(as_warning('SIGINT: Stopping worker. Invoke again to kill the process.'))
            if self._backend:
                self._backend.stop()
            self._die_on_next_int = True
    
    def die(self, _, __):
        print(as_warning('SIGTERM: Killing the worker.'))
        if self._backend:
            self._backend.die()

    def initialize(self):
        self._backend.set_initial_feasible_solution = self.set_initial_feasible_solution
        self._backend.set_null_space_basis = self.set_null_space_basis
        self._backend.set_commodity_in_out_mask = self.set_commodity_in_out_mask
        self._backend.set_solver_parameters = self.set_solver_parameters
        self._backend.update_cached_values = self.update_cached_values
        self._backend.report_chunk = self.report_chunk
        self._backend.report_aggregate = self.report_aggregate
        self._backend.set_active_commodity_count = self.set_active_commodity_count
        self._backend.do_inner_loop_update = self.do_inner_loop_pgd_update
    
    def wait(self):
        self._backend.wait()
    
    def set_initial_feasible_solution(self, X: CPUArray):
        self._X_ek_start_chunk = X
        self._NUM_EDGES, self._CHUNK_LEN = self._X_ek_start_chunk.shape
    
    def set_null_space_basis(self, NULL_M: CPUArray):
        self._NULL_M = NULL_M
        assert self._X_ek_start_chunk is not None
        CHUNK_LEN = self._CHUNK_LEN
        self._NULL_M = NULL_M
        self._NNT_M = NULL_M @ NULL_M.T
        T = self._NULL_M.shape[1]
        self._T = T
        self._Y_tk_chunk = cpu_zeros((T, CHUNK_LEN))
        self._Y_bar_t_cached: Optional[CPUArray] = cpu_zeros((T,))
        self._P_bar_t_cached: Optional[CPUArray] = cpu_zeros((T,))
        self._u_t_cached: Optional[CPUArray] = cpu_zeros((T,))
        self._lambda_ek_chunk = cpu_zeros(self._X_ek_start_chunk.shape)
    
    def set_commodity_in_out_mask(self, MASK_M: BooleanCPUArray):
        self._MASK_M_chunk = MASK_M
        N, K = MASK_M.shape
        assert self._NUM_EDGES == N
        assert self._CHUNK_LEN == K

    def _get_current_C(self) -> CPUArray:
        Y_TK = self._Y_tk_chunk
        Y_BAR = self._Y_bar_t_cached
        P_BAR = self._P_bar_t_cached
        U_T = self._u_t_cached
        
        return Y_TK - np.expand_dims(Y_BAR - P_BAR + U_T, axis=1)
    
    def set_solver_parameters(self, new_params: SynchADMMSolverParams):
        self._solver_params = new_params
        set_global_precision(precision=new_params.Precision)
        set_cpu_float_precision()

    def do_inner_loop_pgd_update(self, epoch: int, F_e: Optional[CPUArray] = None) -> Tuple[int, CPUArray]:
        GAMMA = self._solver_params.Gamma
        KAPPA = self._solver_params.Kappa
        PGD_ITERS = self._solver_params.PGDIterations
        NULL_M = self._NULL_M
        NNT_M = self._NNT_M
        X_EK_START_CHUNK = self._X_ek_start_chunk
        M_MASK_CHUNK = self._MASK_M_chunk
        LAMBDA_EK_CHUNK = self._lambda_ek_chunk
        C_TK_CHUNK = self._get_current_C()
        
        start = time.perf_counter_ns()
        self._lambda_ek_chunk, self._Y_tk_chunk = \
            do_plain_pgd_with_step_reduction(LAMBDA_EK_CHUNK, X_EK_START_CHUNK, NNT_M, NULL_M, C_TK_CHUNK, GAMMA, 
                                                PGD_ITERS, KAPPA, epoch, M_MASK_CHUNK)
        # self._lambda_ek_chunk, self._Y_tk_chunk = \
        #     do_nesterov_pgd(LAMBDA_EK_CHUNK, X_EK_START_CHUNK, NNT_M, NULL_M, C_TK_CHUNK, GAMMA, 
        #                     PGD_ITERS, epoch, M_MASK_CHUNK)
        means = np.mean(self._Y_tk_chunk, axis=1)
        return time.perf_counter_ns() - start, means
    
    def set_active_commodity_count(self, K: int):
        self._K = K

    def update_cached_values(self, u_t: CPUArray, P_bar_t: CPUArray, Y_bar_t: CPUArray):
        self._u_t_cached = u_t
        self._P_bar_t_cached = P_bar_t
        self._Y_bar_t_cached = Y_bar_t
    
    def report_chunk(self) -> CPUArray:
        return cpu_array(self._Y_tk_chunk)
    
    def report_aggregate(self) -> CPUArray:
        return np.sum(self._X_ek_start_chunk + self._NULL_M @ self._Y_tk_chunk, axis=1)

    def close(self):
        self._backend.close()

    @classmethod
    def spawn_and_wait(cls, params: WorkerNodeParams, solver_params: Optional[SynchADMMSolverParams] = None):
        with contextlib.closing(cls(params, solver_params)) as worker:
            worker.initialize()
            worker.wait()


if __name__ == '__main__':
    import socket
    import argparse
    from utils.logging import as_fail
    from .. import DEFAULT_RPC_PORT
    from .worker_backends.udp_multicast_backend import MulticastWorkerBackend, MulticastWorkerBackendParams
    from .worker_backends.grpc_backend import gRPCWorkerBackend, gRPCWorkerBackendParams

    parser =argparse.ArgumentParser('Spawn A Worker Node')
    parser.add_argument('worker_id', type=int, help='Worker ID')
    parser.add_argument('--multicast', action='store_true', help='Use UDP Multicast backend')
    parser.add_argument('--hostname', help='Hostname to use')
    args = parser.parse_args()

    worker_id = args.worker_id
    if worker_id < 0:
        print(as_fail('Worker ID was not properly initialized!'), file=sys.stderr)
        sys.exit(-1)
    else:
        hostname = args.hostname if args.hostname is not None else f'n{worker_id}'
        if not args.multicast:
            rpc_params = gRPCWorkerBackendParams(
                IP=socket.gethostbyname(hostname), 
                Port=DEFAULT_RPC_PORT + worker_id,
                WorkerID=worker_id
            )
            rpc_cls = gRPCWorkerBackend
        else:
            rpc_params = MulticastWorkerBackendParams(
                IP=socket.gethostbyname(hostname), 
                Port=DEFAULT_RPC_PORT + worker_id,
                WorkerID=worker_id
            )
            rpc_cls = MulticastWorkerBackend
        print(f'RPC Parameters:\n{rpc_params}')
        SynchADMMWorkerNode.spawn_and_wait(WorkerNodeParams(
            communication_backend=rpc_cls,
            rpc_params=rpc_params
        ))
