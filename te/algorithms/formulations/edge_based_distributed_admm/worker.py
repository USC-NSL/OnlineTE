import sys
import time
import contextlib
import numpy as np
from typing import Optional, Tuple
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.cpu_utils import CPUArray, cpu_zeros, cpu_array, set_cpu_float_precision
from . import DistributedADMMSolverParams, DistributedADMMWorkerRPCParams
from .worker_backends.base import WorkerNodeCommunicationBackendBase
from .worker_backends.synchronous_grpc_backend import SynchronousgRPCBackend
from .worker_backends.udp_multicast_backend import MulticastBackend
from te.algorithms.sub_algorithms.pgd import do_plain_pgd_with_step_reduction


class NetworkWorkerNode:
    def __init__(self, rpc_params: DistributedADMMWorkerRPCParams, 
                 solver_params: Optional[DistributedADMMSolverParams] = None):
        self.worker_id = rpc_params.WorkerID
        self._rpc_params = rpc_params
        self._solver_params = solver_params
        self._ready: bool = False

        self._K: Optional[int] = None
        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None
        self._CHUNK_LEN: Optional[int] = None
        self._NULL_M: Optional[CPUArray] = None
        self._NNT_M: Optional[CPUArray] = None
        self._X_ek_start_chunk: Optional[CPUArray] = None
        self._Y_bar_t_cached: Optional[CPUArray] = None
        self._P_bar_t_cached: Optional[CPUArray] = None
        self._u_t_cached: Optional[CPUArray] = None
        self._Y_tk_chunk: Optional[CPUArray] = None

        # Specific to the PGD solution
        self._lambda_ek_chunk: Optional[CPUArray] = None
        
        # Specific to the ADMM solution
        self._S_ek_chunk: Optional[CPUArray] = None
        self._t_ek_chunk: Optional[CPUArray] = None

        self._backend = None
    
    def initialize(self):
        if self._rpc_params.Multicast:
            self._backend: WorkerNodeCommunicationBackendBase = MulticastBackend(rpc_params)
        else:
            self._backend: WorkerNodeCommunicationBackendBase = SynchronousgRPCBackend(rpc_params)
        self._backend.set_initial_feasible_solution = self.set_initial_feasible_solution
        self._backend.set_null_space_basis = self.set_null_space_basis
        self._backend.set_solver_parameters = self.set_solver_parameters
        self._backend.update_cached_values = self.update_cached_values
        self._backend.report_chunk = self.report_chunk
        self._backend.report_aggregate = self.report_aggregate
        self._backend.set_active_commodity_count = self.set_active_commodity_count
    
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

        if self._solver_params.QPMethod == 'PGD':
            self._lambda_ek_chunk = cpu_zeros(self._X_ek_start_chunk.shape)
        elif self._solver_params.QPMethod == 'ADMM':
            self._S_ek_chunk = np.copy(self._X_ek_start_chunk)
            self._t_ek_chunk = cpu_zeros(self._X_ek_start_chunk.shape)

    def _get_current_C(self) -> CPUArray:
        Y_TK = self._Y_tk_chunk
        Y_BAR = self._Y_bar_t_cached
        P_BAR = self._P_bar_t_cached
        U_T = self._u_t_cached
        
        return Y_TK - np.expand_dims(Y_BAR - P_BAR + U_T, axis=1)
    
    def set_solver_parameters(self, new_params: DistributedADMMSolverParams):
        self._solver_params = new_params
        set_global_precision(precision=new_params.Precision)
        set_cpu_float_precision()
        self._backend.do_inner_loop_update = {
            'PGD': self.do_inner_loop_pgd_update,
            'ADMM': self.do_inner_loop_admm_update
        }[self._solver_params.QPMethod]

    def do_inner_loop_pgd_update(self, epoch: int, F_e: Optional[CPUArray] = None) -> Tuple[int, CPUArray]:
        assert self._solver_params.QPMethod == 'PGD'
        GAMMA = self._solver_params.Gamma
        KAPPA = self._solver_params.Kappa
        LOCAL_ITERS = self._solver_params.NumberOfLocalUpdates
        PGD_ITERS = self._solver_params.QPIterations
        NULL_M = self._NULL_M
        NNT_M = self._NNT_M
        X_EK_START_CHUNK = self._X_ek_start_chunk
        LAMBDA_EK_CHUNK = self._lambda_ek_chunk
        C_TK_CHUNK = self._get_current_C()
        
        start = time.perf_counter_ns()
        local_iters = 0
        while True:
            self._lambda_ek_chunk, self._Y_tk_chunk = \
                do_plain_pgd_with_step_reduction(LAMBDA_EK_CHUNK, X_EK_START_CHUNK, NNT_M, NULL_M, C_TK_CHUNK, GAMMA, 
                                                 PGD_ITERS, KAPPA, epoch)
            means = np.mean(self._Y_tk_chunk, axis=1)
            if F_e is not None and local_iters < LOCAL_ITERS:
                self.do_local_update(F_e, means)
                local_iters += 1
            else:
                break
        return time.perf_counter_ns() - start, means

    def do_inner_loop_admm_update(self, epoch: int, F_e: Optional[CPUArray] = None) -> Tuple[int, CPUArray]:
        assert self._solver_params.QPMethod == 'ADMM'
        GAMMA = self._solver_params.Gamma
        ADMM_ITERS = self._solver_params.QPIterations
        NULL_M = self._NULL_M
        Y_TK = self._Y_tk_chunk
        X_EK_START_CHUNK = self._X_ek_start_chunk
        C_TK_CHUNK = self._get_current_C()
        S_EK_CHUNK = self._S_ek_chunk
        T_EK_CHUNK = self._t_ek_chunk
        
        start = time.perf_counter_ns()
        for _ in range(ADMM_ITERS):
            Y_TK = (C_TK_CHUNK - GAMMA * NULL_M.T @ (X_EK_START_CHUNK + T_EK_CHUNK - S_EK_CHUNK)) / (1 + GAMMA)
            S_EK_CHUNK = np.clip(X_EK_START_CHUNK + NULL_M @ Y_TK + T_EK_CHUNK, a_min=0, a_max=None)
            T_EK_CHUNK = T_EK_CHUNK + (X_EK_START_CHUNK + NULL_M @ Y_TK - S_EK_CHUNK)
        self._Y_tk_chunk = Y_TK
        self._S_ek_chunk = S_EK_CHUNK
        self._t_ek_chunk = T_EK_CHUNK
        return time.perf_counter_ns() - start, np.mean(self._Y_tk_chunk, axis=1)
    
    def do_local_update(self, F_e: CPUArray, means: CPUArray):
        K = self._K
        ETA = self._solver_params.Eta
        RHO = self._solver_params.Rho
        self._P_bar_t_cached = self._P_bar_t_cached + (ETA/RHO) / (K + ETA/RHO) * (means - self._Y_bar_t_cached)
        self._Y_bar_t_cached = means
        self._u_t_cached = self._u_t_cached + (means - self._P_bar_t_cached)
    
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

    @staticmethod
    def spawn_and_wait(rpc_params: DistributedADMMWorkerRPCParams, 
                       solver_params: Optional[DistributedADMMSolverParams] = None):
        with contextlib.closing(NetworkWorkerNode(rpc_params, solver_params)) as worker:
            worker.initialize()
            worker.wait()


if __name__ == '__main__':
    import socket
    import argparse
    from utils.logging import as_fail

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
        rpc_params = DistributedADMMWorkerRPCParams(
            IP=socket.gethostbyname(hostname), Port=13000 + worker_id,
            WorkerID=worker_id, Multicast=args.multicast
        )
        print(f'RPC Parameters:\n{rpc_params}')
        NetworkWorkerNode.spawn_and_wait(rpc_params)
