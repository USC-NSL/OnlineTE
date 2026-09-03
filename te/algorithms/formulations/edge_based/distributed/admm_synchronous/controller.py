import time
import numpy as np
import asyncio.exceptions
from typing import Optional
from te.algorithms.base import *
from te.traffic_models.base import traffic_to_demands
from topologies.utils import get_graph_null_space_basis
from utils.exceptions import SolutionInterrupted
from utils.logging import as_info, as_fail, as_success, as_warning, ShortTQDM
from array_utils import set_global_precision
from array_utils.cpu.types import *
# TODO: Finish the `SharingWrapper` for the inner loop
from te.algorithms.sub_algorithms.admm import ADMMWrapper
from . import SynchADMMSolverParams
from te.algorithms.communication import *
from te.algorithms.sub_algorithms.mlu_backends.base import ControllerMLUSolver, ControllerMLUException


class SynchADMMControllerNode(TELP[SynchADMMSolverParams], DistributedSolverNodeBase):
    def __init__(
        self, 
        problem_description: TEProblemDescription,
        solver_params: SynchADMMSolverParams,
        node_params: DistributedSolverNodeParams, 
        mlu_cls: type[ControllerMLUSolver], 
        mlu_params: SolverParams
    ) -> None:
        super().__init__(
            problem_description=problem_description,
            solver_params=solver_params,
            node_params=node_params
        )
        # Topology info
        self._NULL_M: Optional[CPUArray] = None
        self._T: Optional[int] = None
        # MLU backend solver
        self._alpha: Optional[float] = None
        self._mlu_solver_cls: type[ControllerMLUSolver] = mlu_cls
        self._mlu_params: SolverParams = mlu_params
        self._mlu_solver: Optional[ControllerMLUSolver] = None
        # Outer ADMM
        self._X_ek_sum_e: Optional[CPUArray] = None
        self._outer_admm_wrapper: Optional[ADMMWrapper] = None
        # Inner ADMM
        self._sharing_mean_1: Optional[CPUArray] = None
        self._sharing_mean_2: Optional[CPUArray] = None
        self._sharing_dual: Optional[CPUArray] = None
        # Communication backend
        self.backend: CoordinatorBackendBase = \
            node_params.CommunicationBackendCLS(node_params.RPCParams_)
        # self.backend.register_signal_handler()
        self.backend.start()
        # These we call right now, as opposed to doing them under `initialize`
        set_global_precision(self._solver_params.Precision)
        # TODO: Add back MaxFlow after refactor is done
        assert self.objective == TEObjective.MLU

        self.initialize()
    
    def initialize(self):
        print(as_info("Waiting for workers to become reachable"))
        counter = 0
        while self.backend.is_alive and len(unreachables := self.are_all_workers_reachable()) > 0:
            time.sleep(1)
            counter += 1
            if counter >= 5:
                print(as_warning(f"Unreachable Nodes: {unreachables}"))
                counter = 0
        if not self.backend.is_alive:
            raise SolutionInterrupted
        print(as_success("All worker nodes are reachable"))

        # Initialize the algorithm
        # First, set the null space basis
        self._set_graph_matrices()
        # Reveal the topology to the worker nodes
        self.backend.initialize_worker_nodes(
            self._solver_params,
            self._graph
        )
        # Finalize all controller states
        self._initialize_variables_and_residuals()

    @property
    def alg_name(self) -> str:
        return 'Distributed Synchronous ADMM'
    
    @property
    def current_objective(self) -> float:
        return self._mlu_solver.current_u
    
    def _set_graph_matrices(self):
        self._NULL_M = cpu_array(get_graph_null_space_basis(
            self._graph,
            self._capacities if self._solver_params.ScaleWithCapacity else None
        ))
        self._T = self._NULL_M.shape[1]

    def _initialize_variables_and_residuals(self):
        N = self.number_of_edges
        self._alpha = 1 if self._solver_params.ScaleWithCapacity else np.linalg.norm(self._capacities)**2 / np.sqrt(N)
        self._mlu_solver = self._mlu_solver_cls(
            N, 
            np.ones_like(self._capacities) \
            if self._solver_params.ScaleWithCapacity else \
            self._capacities,
            self._mlu_params,
            # The feasibility and optimality tolerances for the inner MLU problem
            # most be tighter!
            self._problem_description.eval_params.feasibility_tolerance * 0.1,
            self._problem_description.eval_params.optimality_tolerance * 0.1
        )
        self._mlu_solver.rho = self._solver_params.Rho
        self._mlu_solver.alpha = self._alpha
        # Initialize the consensus wrapper
        self._outer_admm_wrapper = ADMMWrapper(
            N, self._solver_params.Rho,
            adaptive_tau=2, adaptive_mu=5, adaptive_T=2
        )
        # Initialize the sharing wrapper
        self._sharing_dual = cpu_zeros((self.number_of_edges,))

    def _get_Z_value(self) -> CPUArray:
        return self._mlu_solver.current_Z
        
    def _make_variables(self):
        assert self._mlu_solver is not None
        self._mlu_solver._make_variables()
        
    def _get_F(self) -> np.ndarray:
        return self._outer_admm_wrapper.get_X_step_bias()
    
    def _set_X_ek(self):
        self._X_ek = self.backend.get_X_ek()
    
    def _add_constraints(self):
        assert self._mlu_solver is not None
        self._mlu_solver._add_constraints()
    
    # @record_cpu_runtime('Controller-Update')
    def _update_controller_objective(self):
        assert self._mlu_solver is not None
        self._mlu_solver.update_F_m(
            -self._outer_admm_wrapper.get_Z_step_bias(),
            self._outer_admm_wrapper.step_size
        )
    
    def _add_objective(self):
        assert self._mlu_solver is not None
        self._mlu_solver._add_objective()

    # @record_return_value('PGD-Runtime')
    # @record_cpu_runtime('Network-Update')
    def _do_network_update(self, epoch: int):
        # TODO: Try to make the workers return aggregate flows as well
        # max_run, self._Y_bar_t = self.backend.do_network_update(epoch)
        max_run, self._sharing_mean_1 = self.backend.do_network_update(epoch)
        return max_run * 1000

    # @record_cpu_runtime('Sharing-Mean')
    def _update_sharing_mean(self):
        assert self._mlu_solver is not None

        K = self.number_of_commodities
        ETA = self._solver_params.Eta
        RHO = self._solver_params.Rho
        U_E = self._sharing_dual
        X_BAR_E = self._sharing_mean_1
        F_E = self._get_F()
        self._sharing_mean_2 = (F_E / K + (ETA/RHO) * (U_E + X_BAR_E)) / (1 + (ETA/RHO))

    def _update_sharing_dual(self):
        assert self._mlu_solver is not None

        self._sharing_dual += (self._sharing_mean_1 - self._sharing_mean_2)

    # @record_cpu_runtime('Update-Reconvene')
    def _reconvene_network_updates(self) -> bool:
        self._update_sharing_mean()
        self._update_sharing_dual()
        self.backend.reconvene_network_updates(
            sharing_mean_1=self._sharing_mean_1,
            sharing_mean_2=self._sharing_mean_2,
            sharing_dual=self._sharing_dual
        )
        # TODO: How safe is this?
        # return norm_in_consensus(self._P_bar_t, self._Y_bar_t, 5e-4)
        return False
    
    # @record_cpu_runtime('Update-X-EK-SUM')
    def _update_X_ek_sum(self):
        self._X_ek_sum_e = self.number_of_commodities * self._sharing_mean_1
        self._outer_admm_wrapper.record_X_update(self._X_ek_sum_e)
    
    # @record_cpu_runtime('Update-Re')
    def _update_r_e(self):
        assert self._mlu_solver is not None

        self._outer_admm_wrapper.record_Z_update(self._get_Z_value())
        self._outer_admm_wrapper.update_dual_var(True)

    def close(self):
        self.backend.close()
        if self._mlu_solver is not None:
            self._mlu_solver.close()

    def _outer_inf_bound(self) -> float:
        if self._solver_params.ScaleWithCapacity:
            return self._problem_description.eval_params.optimality_tolerance
        return self.unscaled_outer_inf_bound

    def _solve_for_tm(self, tm: np.ndarray):
        MODEL_CONTROLLER = self._mlu_solver
        PARAMS = self._solver_params

        try:
            self._update_controller_objective()
            MODEL_CONTROLLER.solve()
            self._update_r_e()
            progress_bar = ShortTQDM(range(PARAMS.OuterLoopRounds))
            for epoch in progress_bar:
                for i in reversed(range(PARAMS.InnerLoopRounds)):
                    self._do_network_update(epoch)
                    if i > 0 and self._reconvene_network_updates():
                        break
                self._reconvene_network_updates()
                # for _ in range(PARAMS.InnerLoopRounds):
                #     self._do_network_update(epoch)
                #     self._reconvene_network_updates()
                self._update_X_ek_sum()
                self._update_controller_objective()
                MODEL_CONTROLLER.solve()
                self._update_r_e()
                if self._solver_params.ScaleWithCapacity:
                    max_util = float(np.max(self.number_of_commodities * self._sharing_mean_1))
                else:
                    max_util = float(np.max(self.number_of_commodities * self._sharing_mean_1 / self._capacities))
                # Inner loop infeasibility is usually very small, no need to bother with it!
                err = self._outer_admm_wrapper.infeasibility
                progress_bar.set_postfix({
                    'Cont. Util.': f'{self._mlu_solver.current_u:.4f}',
                    'Net. Util.': f'{max_util:.4f}',
                    'Outer Inf.': f'{err:.4f}',
                    'Outer Step.': f'{self._outer_admm_wrapper.step_size:.2f}'
                })
                if err < self._outer_inf_bound():
                    print(as_success("Crossed the convergance bound. Breaking early ..."))
                    progress_bar._pbar.close()
                    break
            if not self._problem_description.eval_params.skip_checks:
                self._set_X_ek()
        except ControllerMLUException as e:
            raise RuntimeError(as_fail(f'MLU solver failed: {e}'))
        except SolutionInterrupted:
            if not self._problem_description.eval_params.skip_checks:
                self._set_X_ek()
        except asyncio.exceptions.CancelledError:
            pass

    def run(self):
        self.solve()

    def _update_constraits(self, tm: np.ndarray):
        demands = cpu_array(traffic_to_demands(tm))
        # First, update demands so that nodes can set X_0
        # Nodes will return `X_bar` which is `sharing_mean_1`
        X_bar = self.backend.update_demands(demands)

        if self._X_ek_sum_e is None:
            # On initialization, we are seeing all of this for the first
            # time, so we should initialize the outer ADMM wrapper
            self._X_ek_sum_e = X_bar * self.number_of_commodities
            self._outer_admm_wrapper.initialize(self._X_ek_sum_e)
        else:
            # On subsequent iterations, only record an X-update
            self._X_ek_sum_e = X_bar * self.number_of_commodities
            self._outer_admm_wrapper.record_X_update(self._X_ek_sum_e)

        # Update _LOCAL_  sharing mean. The switches will report
        # the other one after an update.
        self._sharing_mean_2 = cpu_array(X_bar)

    def _update_objective(self, tm: np.ndarray):
        pass
