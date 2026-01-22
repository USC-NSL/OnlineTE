import sys
import time
import numpy as np
from typing import Optional, Tuple, Union
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.cpu_utils import CPUArray, BooleanCPUArray, CPUCSRArray, CPUCSCArray, cpu_zeros, cpu_array, set_cpu_float_precision
from ..base import DistributedSolverNodeBase, DistributedSolverNodeParams
from . import SynchADMMSolverParams
from .base import SynchADMMWorkerBackendBase
from te.algorithms.sub_algorithms.pgd import do_memory_efficient_pgd
from te.algorithms.sub_algorithms.lasso import sparse_range_lasso
from te.algorithms.sub_algorithms.feasible_assignment import InitialSolutionType


class DenseSolver:
    def __init__(self, X_0: Union[CPUCSRArray, CPUCSCArray, CPUArray], NNT: CPUArray, mask: BooleanCPUArray, 
                 pgd_step: float, pgd_iters: int):
        self._X_0 = X_0
        self._NNT = NNT
        self._mask = mask
        self._lambda_ek = cpu_zeros(X_0.shape)
        self._X_ek = cpu_array(X_0)
        self._pgd_step = pgd_step
        self._pgd_iters = pgd_iters

    def _get_current_C(self, sharing_bias: CPUArray) -> CPUArray:
        return self._NNT @ (self._X_ek - self._X_0 - np.expand_dims(sharing_bias, axis=1)) + self._X_0

    @property
    def X_ek(self) -> CPUArray:
        return self._X_ek
    
    def update(self, sharing_bias: CPUArray) -> CPUArray:
        self._lambda_ek = do_memory_efficient_pgd(
            lambda_block=self._lambda_ek, 
            x_block=self._X_ek,
            nnt=self._NNT,
            bias=sharing_bias,
            x_block_0=self._X_0,
            step_size=self._pgd_step, 
            n_iter=self._pgd_iters, 
            mask=self._mask
        )
        self._X_ek += self._NNT @ (self._lambda_ek - np.expand_dims(sharing_bias, axis=1))
        return self._X_ek



class SparseSolver:
    def __init__(self, X_0: Union[CPUCSRArray, CPUCSCArray, CPUArray], NNT: CPUArray, mask: BooleanCPUArray, 
                 admm_step: float, admm_iters: int, l1_threshold: float):
        self._X_0 = X_0
        self._NNT = NNT
        self._mask = mask
        self._admm_step = admm_step
        self._admm_iters = admm_iters
        self._l1_threshold = l1_threshold
        self._X_ek = cpu_array(X_0)
        self._Z_ek = cpu_array(X_0)
        self._L_ek = cpu_zeros(X_0.shape)
    
    @property
    def X_ek(self) -> CPUArray:
        return self._Z_ek

    def _get_current_C(self, sharing_bias: CPUArray) -> CPUArray:
        return self._X_ek - np.expand_dims(sharing_bias, axis=1)
    
    def update(self, sharing_bias: CPUArray) -> CPUArray:
        self._X_ek, self._Z_ek, self._L_ek = sparse_range_lasso(
            X_block=self._X_ek, Z_block=self._Z_ek, L_block=self._L_ek,
            X_block_0=self._X_0, NNT=self._NNT, C_block=self._get_current_C(sharing_bias),
            gamma=self._admm_step, epsilon=self._l1_threshold, n_iter=self._admm_iters,
            mask=self._mask
        )
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
        self._NULL_M: Optional[CPUArray] = None
        self._NNT_M: Optional[CPUArray] = None
        self._MASK_M_chunk: Optional[BooleanCPUArray] = None

        self._X_ek_start_chunk: Optional[Union[CPUCSRArray, CPUCSCArray, CPUArray]] = None
        self._sharing_bias_cached: Optional[CPUArray] = None

        self._dense_solver: Optional[DenseSolver] = None
        self._sparse_solver: Optional[SparseSolver] = None

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
    
    def set_initial_feasible_solution(self, X: Union[CPUCSRArray, CPUCSCArray, CPUArray]):
        self._X_ek_start_chunk = X
        self._NUM_EDGES, self._CHUNK_LEN = self._X_ek_start_chunk.shape
    
    def set_null_space_basis(self, NULL_M: CPUArray):
        self._NULL_M = NULL_M
        assert self._X_ek_start_chunk is not None
        self._NULL_M = NULL_M
        self._NNT_M = NULL_M @ NULL_M.T
    
    def set_commodity_in_out_mask(self, MASK_M: BooleanCPUArray):
        self._MASK_M_chunk = MASK_M
        N, K = MASK_M.shape
        assert self._NUM_EDGES == N
        assert self._CHUNK_LEN == K
        if self._solver_params.Beta is None:
            self._dense_solver = DenseSolver(
                self._X_ek_start_chunk, self._NNT_M, self._MASK_M_chunk, 
                self._solver_params.Gamma, self._solver_params.SwitchIterations
            )
        else:
            self._sparse_solver = SparseSolver(
                self._X_ek_start_chunk, self._NNT_M, self._MASK_M_chunk,
                self._solver_params.Gamma, self._solver_params.SwitchIterations,
                self._solver_params.Beta
            )
    
    def set_solver_parameters(self, new_params: SynchADMMSolverParams):
        # TODO: This is so unclean ... there should be a better way
        if isinstance(new_params.X0Type, str):
            new_params.X0Type = InitialSolutionType(new_params.X0Type)
        
        self._solver_params = new_params
        set_global_precision(precision=new_params.Precision)
        set_cpu_float_precision()

    # TODO: Record the last epoch that we managed to handle ...
    def do_inner_loop_pgd_update(self, epoch: int) -> Tuple[int, CPUArray]:
        start = time.perf_counter_ns()
        if self._solver_params.Beta is None:
            X_EK = self._dense_solver.update(self._sharing_bias_cached)
        else:
            X_EK = self._sparse_solver.update(self._sharing_bias_cached)
        means = np.mean(X_EK, axis=1)
        return (time.perf_counter_ns() - start) // 1000, means
    
    def set_active_commodity_count(self, K: int):
        self._K = K

    def update_cached_values(self, sharing_bias: CPUArray):
        self._sharing_bias_cached = sharing_bias
    
    def report_chunk(self) -> CPUArray:
        if self._solver_params.Beta is None:
            return self._dense_solver.X_ek
        else:
            return self._sparse_solver.X_ek
    
    def report_aggregate(self) -> CPUArray:
        raise ValueError("This should NOT be used!")

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
