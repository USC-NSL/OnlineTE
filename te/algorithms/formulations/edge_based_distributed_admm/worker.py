import sys
import contextlib
import numpy as np
from typing import Optional
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.cpu_utils import cpu_zeros, cpu_array, set_cpu_float_precision
from . import DistributedADMMSolverParams, DistributedADMMWorkerRPCParams
from .worker_backends.base import WorkerNodeCommunicationBackendBase
from .worker_backends.synchronous_grpc_backend import SynchronousgRPCBackend
from te.algorithms.sub_algorithms.pgd import do_plain_pgd_with_step_reduction


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

        self._backend: WorkerNodeCommunicationBackendBase = SynchronousgRPCBackend(rpc_params)
        self._backend.set_initial_feasible_solution = self.set_initial_feasible_solution
        self._backend.set_null_space_basis = self.set_null_space_basis
        self._backend.do_inner_loop_update = self.do_inner_loop_update
        self._backend.set_solver_parameters = self.set_solver_parameters
        self._backend.update_cached_values = self.update_cached_values
        self._backend.report_chunk = self.report_chunk
        self._backend.report_aggregate = self.report_aggregate
    
    def wait(self):
        self._backend.wait()
    
    def set_initial_feasible_solution(self, X: np.ndarray):
        self._X_ek_start_chunk = X
        self._NUM_EDGES, self._CHUNK_LEN = self._X_ek_start_chunk.shape
    
    def set_null_space_basis(self, NULL_M: np.ndarray):
        self._NULL_M = NULL_M
        assert self._X_ek_start_chunk is not None
        CHUNK_LEN = self._CHUNK_LEN
        self._NULL_M = NULL_M
        self._NNT_M = NULL_M @ NULL_M.T
        T = self._NULL_M.shape[1]
        self._T = T
        self._Y_tk_chunk = cpu_zeros((T, CHUNK_LEN))
        self._lambda_ek_chunk = cpu_zeros(self._X_ek_start_chunk.shape)
        self._Y_bar_t_cached: Optional[np.ndarray] = cpu_zeros((T,))
        self._P_bar_t_cached: Optional[np.ndarray] = cpu_zeros((T,))
        self._u_t_cached: Optional[np.ndarray] = cpu_zeros((T,))

    def _get_current_C(self) -> np.ndarray:
        Y_TK = self._Y_tk_chunk
        Y_BAR = self._Y_bar_t_cached
        P_BAR = self._P_bar_t_cached
        U_T = self._u_t_cached
        
        return Y_TK - np.expand_dims(Y_BAR - P_BAR + U_T, axis=1)
    
    def set_solver_parameters(self, new_params: DistributedADMMSolverParams):
        self._solver_params = new_params
        set_global_precision(precision=new_params.Precision)
        set_cpu_float_precision()

    def do_inner_loop_update(self, epoch: int) -> np.ndarray:
        GAMMA = self._solver_params.Gamma
        KAPPA = self._solver_params.Kappa
        PGD_ITERS = self._solver_params.PGDIterations
        NULL_M = self._NULL_M
        NNT_M = self._NNT_M
        X_EK_START_CHUNK = self._X_ek_start_chunk
        LAMBDA_EK_CHUNK = self._lambda_ek_chunk
        C_TK_CHUNK = self._get_current_C()
        
        self._lambda_ek_chunk, self._Y_tk_chunk = \
            do_plain_pgd_with_step_reduction(LAMBDA_EK_CHUNK, X_EK_START_CHUNK, NNT_M, NULL_M, C_TK_CHUNK, GAMMA, 
                                             PGD_ITERS, KAPPA, epoch)

        return np.mean(self._Y_tk_chunk, axis=1)

    def update_cached_values(self, u_t: np.ndarray, P_bar_t: np.ndarray, Y_bar_t: np.ndarray):
        self._u_t_cached = u_t
        self._P_bar_t_cached = P_bar_t
        self._Y_bar_t_cached = Y_bar_t
    
    def report_chunk(self) -> np.ndarray:
        return cpu_array(self._Y_tk_chunk)
    
    def report_aggregate(self) -> np.ndarray:
        return np.sum(self._X_ek_start_chunk + self._NULL_M @ self._Y_tk_chunk, axis=1)
    
    def close(self):
        self._backend.close()

    @staticmethod
    def spawn_and_wait(worker_id: int, rpc_params: DistributedADMMWorkerRPCParams, 
                       solver_params: Optional[DistributedADMMSolverParams] = None):
        with contextlib.closing(NetworkWorkerNode(worker_id, rpc_params, solver_params)) as worker:
            worker.wait()


if __name__ == '__main__':
    import socket
    from utils.logging import as_fail
    worker_id = int(sys.argv[1])
    if worker_id < 0:
        print(as_fail('Worker ID was not properly initialized!'), file=sys.stderr)
        sys.exit(-1)
    else:
        # rpc_params = DistributedADMMWorkerRPCParams(IP=socket.gethostbyname(socket.gethostname()), Port=13000 + worker_id)
        rpc_params = DistributedADMMWorkerRPCParams(IP='localhost', Port=13000 + worker_id)
        NetworkWorkerNode.spawn_and_wait(worker_id, rpc_params)
