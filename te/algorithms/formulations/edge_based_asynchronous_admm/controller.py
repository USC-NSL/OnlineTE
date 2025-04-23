import time
import tqdm
import numpy as np
from typing import NoReturn, Optional, List, Tuple
from gurobipy import GurobiError
from utils.logging import as_warning
from utils.exceptions import Unreachable
from te.algorithms.array_utils.cpu_utils import CPUArray, cpu_zeros
from te.algorithms.sub_algorithms.admm_consensus_test import norm_in_consensus
from te.algorithms.statistics.helpers import record_cpu_runtime, record_return_value
from te.algorithms.utils import optimize_or_scream, get_solution_maximum_utilization
from ..edge_based_distributed_admm.controller import ControllerNode as SynchronousControllerNode
from .controller_backends import get_backend
from .controller_backends.base import ControllerCommunicationBackendBase, NetworkUpdate
from . import AsynchronousADMMSolverParams


class ControllerNode(SynchronousControllerNode):
    def __init__(self, graph, traffic, solver_params: AsynchronousADMMSolverParams, rpc_params):
        self._solver_params: AsynchronousADMMSolverParams = solver_params
        super().__init__(graph, traffic, solver_params, rpc_params)
        assert rpc_params.NumWorkers >= solver_params.Upsilon, \
            'The controller update set cannot be larger than number of workers!: '\
            f'{solver_params.Upsilon} > {rpc_params.NumWorkers}'
        
        self._partitioned_Y_bar: Optional[List[CPUArray]] = None

    def initialize(self):
        self._set_initial_feasible_solution()
        self._backend: ControllerCommunicationBackendBase = get_backend(self._rpc_params)
        self._backend.Upsilon = self._solver_params.Upsilon
        self._set_NULL_M()
        self._initialize_variables_and_residuals()
        self._partitioned_Y_bar = \
            [cpu_zeros((self._T,)) for _ in range(self._rpc_params.NumWorkers)]
        self._report_problem_size()
    
    @property
    def alg_name(self) -> str:
        return 'Asynchronous ADMM'

    def _reconvene_network_updates(self) -> NoReturn:
        raise Unreachable
    def _do_network_update(self, epoch) -> NoReturn:
        raise Unreachable
    
    @record_return_value('PGD-Runtime')
    def _consume_updates(self, updates: List[Tuple[int, NetworkUpdate]]) -> int:
        runtimes = []
        for worker_id, update in updates:
            runtime, Y_bar = update
            self._partitioned_Y_bar[worker_id] = Y_bar
            runtimes.append(runtime)
        self._Y_bar_t = np.mean(self._partitioned_Y_bar, axis=0)
        return max(runtimes)
    
    def _wait_for_minimum_updates(self) -> bool:
        gathered_updates = self._backend.get_network_updates()
        if len(gathered_updates) < self._solver_params.Upsilon:
            # This only happens if the solution is interrupted
            print(as_warning('Solution interrupted, will no longer to updates.'))
            return False
        self._consume_updates(gathered_updates)
        return True
    
    @record_cpu_runtime('Solve')
    def solve(self, params = None):
        MODEL_CONTROLLER = self._model_controller
        PARAMS = self._solver_params
        EPOCHS = params if params is not None else PARAMS.NumberOfEpochs
        
        try:
            t = time.time()
            for epoch in tqdm.tqdm(range(EPOCHS), bar_format='{l_bar}{bar:36}{r_bar}{bar:-36b}'):
                optimize_or_scream(MODEL_CONTROLLER)
                if not self._wait_for_minimum_updates():
                    break
                self._update_Zo_e_and_r_e()
                self._update_controller_objective()

                self._objective_trace.append((self._utility.X, get_solution_maximum_utilization(self._Xo_e_assigned, self._graph)))
            self._set_X_ek()
            return time.time() - t
        except GurobiError as e:
            print(f'Error code {e.errno}: {e}')
            return -1
