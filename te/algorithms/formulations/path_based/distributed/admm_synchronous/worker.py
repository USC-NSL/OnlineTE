import sys
import time
import numpy as np
from typing import Optional, Tuple, List
from numba.typed import List as NumbaList
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.cpu_utils import CPUArray, IntegerCPUArray, cpu_zeros, cpu_array, set_cpu_float_precision
from te.algorithms.formulations.edge_based.distributed.base import DistributedSolverNodeBase, DistributedSolverNodeParams
from . import SynchADMMSolverParams
from .base import SynchADMMWorkerBackendBase
# from .worker_backends.grpc_backend import gRPCWorkerBackend
from te.algorithms.sub_algorithms.pgd import do_path_based_pgd, do_path_based_maxflow_pgd
from te.algorithms.sub_algorithms.paths import path_based_to_edge_based_nnz, path_based_to_edge_based_mean_nnz, warm_start_jit


class DenseSolver:
    def __init__(self, alpha_shape: Tuple[int, int, int], alpha_cols: NumbaList, alpha_rows: NumbaList, 
                 beta: IntegerCPUArray, demands: CPUArray, pgd_step: float, pgd_iters: int, eta: float):
        self._alpha_shape = alpha_shape
        self._alpha_rows = alpha_rows
        self._alpha_cols = alpha_cols
        self._beta = beta
        self._demands = demands
        self._pgd_step = pgd_step
        self._pgd_iters = pgd_iters
        self._eta = eta

        K, N, T = alpha_shape
        self._K = K
        self._N = N
        self._T = T
        self._Y_tk = cpu_zeros((T, K))
        self._initialize_splits()
        self._Y_tk_old = cpu_array(self._Y_tk)
    
    def _initialize_splits(self):
        initial_values = self._demands / self._beta
        mask = np.arange(self._T)[:, None] < self._beta
        self._Y_tk[mask] = np.repeat(initial_values, self._beta)

    @property
    def X_ek(self) -> CPUArray:
        return path_based_to_edge_based_nnz(self._Y_tk, self._alpha_rows, self._alpha_cols, self._N, self._demands)
    
    def update(self, sharing_bias: CPUArray) -> CPUArray:
        new_Y = do_path_based_pgd(
            y_block=self._Y_tk,
            y_block_old=self._Y_tk_old,
            alpha_rows=self._alpha_rows,
            alpha_cols=self._alpha_cols,
            sharing_bias=sharing_bias,
            beta_block=self._beta,
            demand_block=self._demands,
            num_edges=self._N,
            num_paths=self._T,
            step_size=self._pgd_step,
            n_iter=self._pgd_iters
        )
        np.copyto(self._Y_tk_old, self._Y_tk)
        np.copyto(self._Y_tk, new_Y)
        return path_based_to_edge_based_mean_nnz(self._Y_tk, self._alpha_rows, self._alpha_cols,
                                                 self._N, self._demands)


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
        self._alpha_shape: Optional[Tuple[int, int, int]] = None
        self._alpha_rows_chunk: Optional[NumbaList[IntegerCPUArray]] = None
        self._alpha_cols_chunk: Optional[NumbaList[IntegerCPUArray]] = None
        self._beta_k_chunk: Optional[IntegerCPUArray] = None
        self._D_k_chunk: Optional[CPUArray] = None
        
        self._sharing_bias_cached: Optional[CPUArray] = None
        self._dense_solver: Optional[DenseSolver] = None

        assert issubclass(params.CommunicationBackendCLS, SynchADMMWorkerBackendBase)
        self.backend: SynchADMMWorkerBackendBase = params.CommunicationBackendCLS(params.RPCParams_)
        self.backend.start()

    def initialize(self):
        self.backend.set_path_mask_shape = self.set_alpha_shape
        self.backend.set_path_mask_rows = self.set_alpha_rows
        self.backend.set_path_mask_cols = self.set_alpha_cols
        self.backend.set_path_count = self.set_beta
        self.backend.set_demands = self.set_demands
        self.backend.set_solver_parameters = self.set_solver_parameters
        self.backend.update_cached_values = self.update_cached_values
        self.backend.report_chunk = self.report_chunk
        self.backend.set_active_commodity_count = self.set_active_commodity_count
        self.backend.do_inner_loop_update = self.do_inner_loop_pgd_update
        self.backend.jit_warmstart = self.jit_warmstart
    
    def run(self):
        self.backend.wait()
    
    def set_alpha_shape(self, shape: Tuple[int,...]):
        assert len(shape) == 3
        self._alpha_shape = shape
    def set_alpha_rows(self, rows: List[IntegerCPUArray]):
        self._alpha_rows_chunk = NumbaList(rows)
    def set_alpha_cols(self, cols: List[IntegerCPUArray]):
        self._alpha_cols_chunk = NumbaList(cols)
    def set_beta(self, beta: IntegerCPUArray):
        self._beta_k_chunk = beta
    def set_demands(self, demands: CPUArray):
        self._D_k_chunk = demands
        self._dense_solver = DenseSolver(
            self._alpha_shape, self._alpha_cols_chunk, self._alpha_rows_chunk,
            self._beta_k_chunk, self._D_k_chunk, 
            self._solver_params.Gamma, self._solver_params.SwitchIterations,
            self._solver_params.Eta
        )
    def set_solver_parameters(self, new_params: SynchADMMSolverParams):
        self._solver_params = new_params
        set_global_precision(precision=new_params.Precision)
        set_cpu_float_precision()

    def do_inner_loop_pgd_update(self, epoch: int) -> Tuple[int, CPUArray]:
        start = time.perf_counter_ns()
        mean = self._dense_solver.update(self._sharing_bias_cached)
        return (time.perf_counter_ns() - start) // 1000, mean
    
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
    
    def jit_warmstart(self):
        warm_start_jit(self._alpha_rows_chunk, self._alpha_cols_chunk, self._alpha_shape, self._beta_k_chunk)


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
