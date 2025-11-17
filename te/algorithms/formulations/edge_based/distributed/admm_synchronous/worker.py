import sys
import time
import numpy as np
from typing import Optional, Tuple
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.cpu_utils import CPUArray, BooleanCPUArray, cpu_zeros, cpu_array, set_cpu_float_precision
from ..base import DistributedSolverNodeBase, DistributedSolverNodeParams
from . import SynchADMMSolverParams
from .base import SynchADMMWorkerBackendBase
from te.algorithms.sub_algorithms.pgd import do_plain_pgd_with_step_reduction, do_nesterov_pgd


class SynchADMMWorkerNode(DistributedSolverNodeBase):
    def __init__(self, params: DistributedSolverNodeParams):
        super().__init__(params)
        self.worker_id = params.RPCParams_.PeerIndex
        self._solver_params: Optional[SynchADMMSolverParams] = None
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

        assert issubclass(params.CommunicationBackendCLS, SynchADMMWorkerBackendBase)
        self.backend: SynchADMMWorkerBackendBase = params.CommunicationBackendCLS(params.RPCParams_)
        self.backend.start()

    def initialize(self):
        self.backend.set_initial_feasible_solution = self.set_initial_feasible_solution
        self.backend.set_null_space_basis = self.set_null_space_basis
        self.backend.set_commodity_in_out_mask = self.set_commodity_in_out_mask
        self.backend.set_solver_parameters = self.set_solver_parameters
        self.backend.update_cached_values = self.update_cached_values
        self.backend.report_chunk = self.report_chunk
        self.backend.report_aggregate = self.report_aggregate
        self.backend.set_active_commodity_count = self.set_active_commodity_count
        self.backend.do_inner_loop_update = self.do_inner_loop_pgd_update
    
    def run(self):
        self.backend.wait()
    
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
        self.backend.close()


if __name__ == '__main__':
    import argparse
    import te.constants
    from utils.logging import as_fail
    from .worker_backends.udp_multicast_backend import MulticastWorkerBackend, MulticastWorkerBackendParams
    from .worker_backends.grpc_backend import gRPCWorkerBackend, gRPCWorkerBackendParams

    parser =argparse.ArgumentParser('Spawn A Worker Node')
    parser.add_argument('worker_id', type=int, help='Worker ID')
    parser.add_argument('--multicast', action='store_true', help='Use UDP Multicast backend')
    parser.add_argument('--hostname', help='Hostname to use')
    parser.add_argument('--port', type=int, help='Port number to bind to')
    parser.add_argument('--local', action='store_true', help='Assume everything is run locally')
    args = parser.parse_args()

    worker_id: int = args.worker_id
    hostname: Optional[str] = args.hostname
    port: Optional[int] = args.port
    if worker_id < 0:
        print(as_fail('Worker ID was not properly initialized!'), file=sys.stderr)
        sys.exit(-1)

    if args.local:
        hostname = "localhost"
        if port is None:
            port = te.constants.DEFAULT_RPC_PORT + worker_id + 1
    else:
        if hostname is None:
            hostname = f'n{worker_id}'
        if port is None:
            port = te.constants.DEFAULT_RPC_PORT

    if not args.multicast:
        rpc_params = gRPCWorkerBackendParams(
            PeerIndex=worker_id, Peers=tuple([(hostname, port)])
        )
        rpc_cls = gRPCWorkerBackend
    else:
        rpc_params = MulticastWorkerBackendParams(
            PeerIndex=worker_id, Peers=tuple([(hostname, port)])
        )
        rpc_cls = MulticastWorkerBackend
    
    print(f'RPC Parameters:\n{rpc_params.str_all()}')
    SynchADMMWorkerNode.spawn_and_run(DistributedSolverNodeParams(
        CommunicationBackendCLS=rpc_cls, RPCParams_=rpc_params
    ))
