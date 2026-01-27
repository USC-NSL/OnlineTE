import sys
import time
import numpy as np
from typing import Optional, Tuple
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.cpu_utils import (CPUArray, IntegerCPUArray, BooleanCPUArray, cpu_cast_float,
                                                 cpu_zeros, cpu_array, set_cpu_float_precision)
from te.algorithms.formulations.edge_based.distributed.base import DistributedSolverNodeBase, DistributedSolverNodeParams
from . import SynchADMMSolverParams
from .base import SynchADMMWorkerBackendBase
# from .worker_backends.grpc_backend import gRPCWorkerBackend
from te.algorithms.sub_algorithms.pgd import do_path_based_pgd, do_path_based_maxflow_pgd
from te.algorithms.sub_algorithms.paths import path_based_to_edge_based


class DenseSolver:
    def __init__(self, alpha: BooleanCPUArray, beta: IntegerCPUArray, demands: CPUArray,
                 pgd_step: float, pgd_iters: int, eta: float):
        self._alpha = alpha
        self._beta = beta
        self._demands = demands
        self._pgd_step = pgd_step
        self._pgd_iters = pgd_iters
        self._eta = eta

        K, _, T = alpha.shape
        self._K = K
        self._T = T
        self._A_ktt = (demands[:, np.newaxis, np.newaxis] ** 2) * np.einsum('kij,kih->kjh', self._alpha, self._alpha)
        self._Y_tk = cpu_zeros((T, K))
        self._Y_tk[0, :] = 1
        self._X_ek = path_based_to_edge_based(self._Y_tk, self._alpha, self._demands)

    def _get_current_C(self, sharing_bias: CPUArray) -> CPUArray:
        term1 = np.einsum('kij,jk->ik', self._A_ktt, self._Y_tk)
        term2 = np.einsum('k,kij,i->jk', self._demands, self._alpha, sharing_bias)
        return term1 - term2

    @property
    def X_ek(self) -> CPUArray:
        return self._X_ek
    
    def update(self, sharing_bias: CPUArray) -> CPUArray:
        # self._Y_tk = do_path_based_pgd(
        #     y_block=self._Y_tk,
        #     A_block=self._A_ktt,
        #     C_block=self._get_current_C(sharing_bias),
        #     beta_block=self._beta,
        #     step_size=self._pgd_step,
        #     n_iter=self._pgd_iters
        # )
        self._Y_tk = do_path_based_maxflow_pgd(
            y_block=self._Y_tk,
            A_block=self._A_ktt,
            C_block=self._get_current_C(sharing_bias),
            D_block=self._demands,
            beta_block=self._beta,
            step_size=self._pgd_step,
            n_iter=self._pgd_iters,
            eta=self._eta
        )
        self._X_ek = path_based_to_edge_based(self._Y_tk, self._alpha, self._demands)
        return self._X_ek


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
        self._alpha_ket_chunk: Optional[BooleanCPUArray] = None
        self._beta_k_chunk: Optional[IntegerCPUArray] = None
        self._D_k_chunk: Optional[CPUArray] = None
        
        self._sharing_bias_cached: Optional[CPUArray] = None
        self._dense_solver: Optional[DenseSolver] = None

        assert issubclass(params.CommunicationBackendCLS, SynchADMMWorkerBackendBase)
        self.backend: SynchADMMWorkerBackendBase = params.CommunicationBackendCLS(params.RPCParams_)
        self.backend.start()

    def initialize(self):
        self.backend.set_path_mask = self.set_alpha
        self.backend.set_path_count = self.set_beta
        self.backend.set_demands = self.set_demands
        self.backend.set_solver_parameters = self.set_solver_parameters
        self.backend.update_cached_values = self.update_cached_values
        self.backend.report_chunk = self.report_chunk
        self.backend.set_active_commodity_count = self.set_active_commodity_count
        self.backend.do_inner_loop_update = self.do_inner_loop_pgd_update
    
    def run(self):
        self.backend.wait()
    
    def set_alpha(self, alpha: BooleanCPUArray):
        self._alpha_ket_chunk = alpha
        K, N, T = alpha.shape
        
        self._CHUNK_LEN = K 
        self._NUM_EDGES = N 
        self._T = T

    def set_beta(self, beta: IntegerCPUArray):
        self._beta_k_chunk = beta
    def set_demands(self, demands: CPUArray):
        self._D_k_chunk = demands
        self._dense_solver = DenseSolver(
            self._alpha_ket_chunk, self._beta_k_chunk, self._D_k_chunk, 
            self._solver_params.Gamma, self._solver_params.SwitchIterations,
            self._solver_params.Eta
        )
    def set_solver_parameters(self, new_params: SynchADMMSolverParams):
        self._solver_params = new_params
        set_global_precision(precision=new_params.Precision)
        set_cpu_float_precision()

    def do_inner_loop_pgd_update(self, epoch: int) -> Tuple[int, CPUArray]:
        start = time.perf_counter_ns()
        X_EK = self._dense_solver.update(self._sharing_bias_cached)
        return time.perf_counter_ns() - start, np.mean(X_EK, axis=1)
    
    def set_active_commodity_count(self, K: int):
        self._K = K

    def update_cached_values(self, sharing_bias: CPUArray):
        self._sharing_bias_cached = sharing_bias
    
    def report_chunk(self) -> CPUArray:
        return self._dense_solver.X_ek
    
    def report_aggregate(self) -> CPUArray:
        raise ValueError("This should NOT be used!")

    def close(self):
        self.backend.close()


# if __name__ == '__main__':
#     import socket
#     import argparse
#     from utils.logging import as_fail

#     parser =argparse.ArgumentParser('Spawn A Worker Node')
#     parser.add_argument('worker_id', type=int, help='Worker ID')
#     parser.add_argument('--multicast', action='store_true', help='Use UDP Multicast backend')
#     parser.add_argument('--hostname', help='Hostname to use')
#     args = parser.parse_args()

#     worker_id = args.worker_id
#     if worker_id < 0:
#         print(as_fail('Worker ID was not properly initialized!'), file=sys.stderr)
#         sys.exit(-1)
#     else:
#         assert not args.multicast, 'Multicast not yet implemented for this'
#         hostname = args.hostname if args.hostname is not None else f'n{worker_id}'
#         rpc_params = PathBasedDistributedADMMWorkerRPCParams(
#             IP=socket.gethostbyname(hostname), Port=13000 + worker_id,
#             WorkerID=worker_id, Multicast=args.multicast
#         )
#         print(f'RPC Parameters:\n{rpc_params}')
#         NetworkWorkerNode.spawn_and_wait(rpc_params)
