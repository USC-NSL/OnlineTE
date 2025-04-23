import threading
from typing import List, Tuple
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.cpu_utils import CPUArray, set_cpu_float_precision
from ..edge_based_distributed_admm.worker import NetworkWorkerNode as SynchronousNetworkWorkerNode
from .worker_backends.base import WorkerNodeCommunicationBackendBase
from . import AsynchronousADMMSolverParams


class NetworkWorkerNode(SynchronousNetworkWorkerNode):
    def __init__(self, rpc_params, solver_params = None):
        self._is_active: bool = False
        self._quit_event: threading.Event = threading.Event()
        super().__init__(rpc_params, solver_params)
    
    def initialize(self):
        assert self._rpc_params.Multicast
        self._backend: WorkerNodeCommunicationBackendBase = MulticastBackend(rpc_params)
        self._backend.set_initial_feasible_solution = self.set_initial_feasible_solution
        self._backend.set_null_space_basis = self.set_null_space_basis
        self._backend.set_solver_parameters = self.set_solver_parameters
        self._backend.report_chunk = self.report_chunk
        self._backend.set_active_commodity_count = self.set_active_commodity_count
        self._is_active = True
        self._quit_event.clear()

    def set_solver_parameters(self, new_params: AsynchronousADMMSolverParams):
        self._solver_params = new_params
        self._backend.set_update_batch_size(new_params.WorkerBatchSize)
        set_global_precision(precision=new_params.Precision)
        set_cpu_float_precision()

    def consume_batch_update(self, batch: List[Tuple[CPUArray, CPUArray, CPUArray]]):
        """
        We need to think about this.
        Currently, the best thing that we might be able to do is to just pick the
        most recent update.
        """
        self._u_t_cached, self._P_bar_t_cached, self._Y_bar_t_cached = batch[-1]
    
    def solve(self):
        while self._is_active:
            # First, update yourself and generate a notification for the controller
            if self._solver_params.QPMethod == 'PGD':
                # The epoch here should NOT be used for step reduction.
                # TODO: We need to substitute this with something, it will not work
                # well without it.
                self.do_inner_loop_pgd_update(0)
            elif self._solver_params.QPMethod == 'ADMM':
                self.do_inner_loop_pgd_update(0)
            else:
                raise ValueError(f'Unexpected QP method: {self._solver_params.QPMethod}')
            
            controller_update_batch = self._backend.gather_updates()
            if not self._is_active:
                break
            if controller_update_batch is not None:
                self.consume_batch_update(controller_update_batch)

