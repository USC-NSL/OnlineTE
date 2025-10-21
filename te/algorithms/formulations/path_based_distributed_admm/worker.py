import sys
import time
import signal
import contextlib
import numpy as np
from typing import Optional, Tuple
from utils.logging import as_warning
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.cpu_utils import (CPUArray, IntegerCPUArray, BooleanCPUArray, 
                                                 cpu_zeros, cpu_array, set_cpu_float_precision)
from . import PathBasedDistributedADMMSolverParams, PathBasedDistributedADMMWorkerRPCParams
from .worker_backends.base import WorkerNodeCommunicationBackendBase
from .worker_backends.synchronous_grpc_backend import SynchronousgRPCBackend
from te.algorithms.sub_algorithms.pgd import do_plain_path_based_pgd_with_step_reduction


class NetworkWorkerNode:
    def __init__(self, rpc_params: PathBasedDistributedADMMWorkerRPCParams, 
                 solver_params: Optional[PathBasedDistributedADMMSolverParams] = None):
        self.worker_id = rpc_params.WorkerID
        self._rpc_params = rpc_params
        self._solver_params = solver_params
        self._ready: bool = False

        self._K: Optional[int] = None
        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None
        self._CHUNK_LEN: Optional[int] = None
        self._alpha_ket_chunk: Optional[BooleanCPUArray] = None
        self._beta_k_chunk: Optional[IntegerCPUArray] = None
        self._D_k_chunk: Optional[CPUArray] = None
        
        self._X_bar_e_cached: Optional[CPUArray] = None
        self._P_bar_e_cached: Optional[CPUArray] = None
        self._u_e_cached: Optional[CPUArray] = None
        self._X_ek_chunk: Optional[CPUArray] = None
        self._Y_tk_chunk: Optional[CPUArray] = None

        self._backend = None
        
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
        self._backend: WorkerNodeCommunicationBackendBase = SynchronousgRPCBackend(self._rpc_params)
        self._backend.set_alpha = self.set_alpha
        self._backend.set_beta = self.set_beta
        self._backend.set_demands = self.set_demands

        self._backend.set_solver_parameters = self.set_solver_parameters
        self._backend.update_cached_values = self.update_cached_values
        self._backend.report_chunk = self.report_chunk
        self._backend.do_inner_loop_update = self.do_inner_loop_update
        self._backend.set_active_commodity_count = self.set_active_commodity_count

        self._backend.start()
    
    def wait(self):
        self._backend.wait()
    
    def set_alpha(self, alpha: BooleanCPUArray):
        self._alpha_ket_chunk = alpha
        K, N, T = alpha.shape
        
        self._CHUNK_LEN = K 
        self._NUM_EDGES = N 
        self._T = T
        
        self._X_bar_e_cached = cpu_zeros((N,))
        self._P_bar_e_cached = cpu_zeros((N,))
        self._u_e_cached = cpu_zeros((N,))
        self._X_ek_chunk = cpu_zeros((N, K))
        self._Y_tk_chunk = cpu_zeros((T, K))

    def set_beta(self, beta: IntegerCPUArray):
        self._beta_k_chunk = beta
    def set_demands(self, demands: CPUArray):
        self._D_k_chunk = demands

    def _get_current_C(self) -> CPUArray:
        X_EK = self._X_ek_chunk
        X_BAR_E = self._X_bar_e_cached
        P_BAR_E = self._P_bar_e_cached
        U_E = self._u_e_cached
        
        return X_EK - np.expand_dims(X_BAR_E - P_BAR_E + U_E, axis=1)
    
    def _get_scaled_alpha_ket(self) -> CPUArray:
        # TODO: Should was just make this an attribute?
        return self._alpha_ket_chunk * self._D_k_chunk[:, np.newaxis, np.newaxis]
    
    def set_solver_parameters(self, new_params: PathBasedDistributedADMMSolverParams):
        self._solver_params = new_params
        set_global_precision(precision=new_params.Precision)
        set_cpu_float_precision()

    def do_inner_loop_update(self, epoch: int) -> Tuple[int, CPUArray]:
        GAMMA = self._solver_params.Gamma
        KAPPA = self._solver_params.Kappa
        PGD_ITERS = self._solver_params.QPIterations
        C_EK_CHUNK = self._get_current_C()
        Y_TK_CHUNK = self._Y_tk_chunk
        SCALED_ALPHA_KET_CHUNK = self._get_scaled_alpha_ket()
        BETA_K_CHUNK = self._beta_k_chunk
        
        start = time.perf_counter_ns()
        self._Y_tk_chunk = \
            do_plain_path_based_pgd_with_step_reduction(Y_TK_CHUNK, SCALED_ALPHA_KET_CHUNK, C_EK_CHUNK, 
                                                        BETA_K_CHUNK, GAMMA, PGD_ITERS, KAPPA, epoch)
        means = np.mean(np.einsum("ijk,ki->ji", SCALED_ALPHA_KET_CHUNK, self._Y_tk_chunk), axis=1)
        return time.perf_counter_ns() - start, means
    
    def set_active_commodity_count(self, K: int):
        self._K = K
    
    def reset_inner_dual_variable(self):
        self._u_t = cpu_zeros((self._T,))
        self._u_t_cached = cpu_zeros((self._T,))

    def update_cached_values(self, X_bar_e: CPUArray, P_bar_e: CPUArray, u_e: CPUArray):
        self._X_bar_e_cached = X_bar_e
        self._P_bar_e_cached = P_bar_e
        self._u_e_cached = u_e
    
    def report_chunk(self) -> CPUArray:
        return cpu_array(self._Y_tk_chunk)
    
    def close(self):
        self._backend.close()

    @staticmethod
    def spawn_and_wait(rpc_params: PathBasedDistributedADMMWorkerRPCParams, 
                       solver_params: Optional[PathBasedDistributedADMMSolverParams] = None):
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
        assert not args.multicast, 'Multicast not yet implemented for this'
        hostname = args.hostname if args.hostname is not None else f'n{worker_id}'
        rpc_params = PathBasedDistributedADMMWorkerRPCParams(
            IP=socket.gethostbyname(hostname), Port=13000 + worker_id,
            WorkerID=worker_id, Multicast=args.multicast
        )
        print(f'RPC Parameters:\n{rpc_params}')
        NetworkWorkerNode.spawn_and_wait(rpc_params)
