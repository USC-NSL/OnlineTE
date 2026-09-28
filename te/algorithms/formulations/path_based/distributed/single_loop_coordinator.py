import time
import numpy as np
import asyncio.exceptions
from dataclasses import replace
from typing import Optional
from te.algorithms.base import *
from te.traffic_models.base import traffic_to_demands
from utils.exceptions import SolutionInterrupted
from utils.logging import as_info, as_success, as_warning, ShortTQDM, TQDMSpinner
from array_utils import set_global_precision
from array_utils.cpu.types import *
from te.algorithms.sub_algorithms.admm import DistributedSharingWrapper, AdaptiveStepSizeParams, NesterovAccelerationParams
from te.algorithms.communication import *
from te.algorithms.sub_algorithms.capacity_qp import CapacityQP
from .solver_params import PathBasedOnlineTEParameters, WorkerUpdateData


class OnlineTECoordinator(TELP[PathBasedOnlineTEParameters], DistributedSolverNodeBase):
    def __init__(
        self, 
        problem_description: TEProblemDescription,
        solver_params: PathBasedOnlineTEParameters,
        node_params: DistributedSolverNodeParams,
        *args, **kwargs
    ) -> None:
        super().__init__(
            problem_description=problem_description,
            solver_params=solver_params,
            node_params=node_params
        )
        # MLU QP solver
        self._mlu_solver: CapacityQP = CapacityQP(
            rho=solver_params.Rho,
            num_edges=self.num_edges,
            num_commodities=self.number_of_commodities,
            feasibility_tolerance=0.01 * self.feasibility_tolerance,
            optimality_tolerance=0.01 * self.optimality_tolerance,
            objective=self.objective,
            capacities=self._capacities,
            mean_scaled=True
        )
        # ADMM
        self._sharing_wrapper: Optional[DistributedSharingWrapper] = None
        # Communication backend
        self.backend: CoordinatorBackendBase = \
            node_params.CommunicationBackendCLS[PathBasedOnlineTEParameters](
                rpc_params=node_params.RPCParams_,
                solver_params_cls=PathBasedOnlineTEParameters
            )
        self.backend.start()
        # These we call right now, as opposed to doing them under `initialize`
        set_global_precision(self._solver_params.Precision)
        self._skip_loop_check = True
        self._total_flow = 0

        self._primal_objective = 0
        self._dual_objective = 0

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
        self.backend.initialize_worker_nodes(
            self._get_worker_solver_params(),
            self._graph,
            self.objective
        )
        # Finalize all controller states
        self._initialize_variables_and_residuals()

    def _get_worker_solver_params(self) -> PathBasedOnlineTEParameters:
        return replace(
            self._solver_params,
            SwitchOptimalityTolerance=0.9 * self.optimality_tolerance
        )

    @property
    def alg_name(self) -> str:
        return 'Path Based OnlineTE'

    @property
    def current_objective(self) -> float:
        return self._primal_objective

    def _initialize_variables_and_residuals(self):
        def constant_step_size():
            while True: yield 0.1
        # Build the MLU QP
        self._mlu_solver.build()
        # Create the ADMM wrapper
        self._sharing_wrapper = DistributedSharingWrapper(
            (self.number_of_edges, self.number_of_commodities),
            self._solver_params.Rho,
            # adaptation=AdaptiveStepSizeParams(
            #     infeasibility_bound=5,
            #     multiplicative_factor=2,
            #     strike_count=1
            # ),
            # acceleration=NesterovAccelerationParams()
            # acceleration=NesterovAccelerationParams(restart_threshold=0.5)
            acceleration=NesterovAccelerationParams(step_sizes=constant_step_size)
        )

    def _get_Z_value(self) -> CPUArray:
        return self._mlu_solver.current_Z

    def _get_lambda(self) -> CPUArray:
        return self._mlu_solver.current_lambda
        
    def _make_variables(self):
        pass
    def _add_constraints(self):
        pass
    def _add_objective(self):
        pass
    
    def _set_X_ek(self):
        """Store the public assignment in physical flow units.

        With capacity scaling enabled, worker-side path operators return
        ``y[e, k] = x[e, k] / C[e]`` so the outer ADMM and MLU controller
        operate against unit capacities.  ``TELP.current_assignment`` is
        consumed by physical-flow checks, however, so undo that change of
        variables only at this reporting boundary.
        """
        self._X_ek = self.backend.get_X_ek()

    def _do_network_update(self):
        SHARING = self._sharing_wrapper
        _, X_bar, data = self.backend.do_network_update(
            sharing_gap=SHARING.X_mean - SHARING.Z_mean,
            sharing_dual=SHARING.dual_var,
            sharing_rho=SHARING.step_size
        )
        if self.objective == TEObjective.MAX_FLOW:
            self._total_flow = data[WorkerUpdateData.DEMANDS]
        self._sharing_wrapper.record_mean_X_update(
            next_X_mean=X_bar,
            delta_X_frobenius=data[WorkerUpdateData.NORM_F_DELTA_X]
        )
        self._dual_objective = data[WorkerUpdateData.DUAL_OBJ]
    
    def _do_coordinator_update(self):
        MLU = self._mlu_solver
        SHARING = self._sharing_wrapper
        # Update `Z_bar`
        MLU.update_F_m(
            SHARING.get_Z_step_bias(),
            SHARING.step_size
        )
        # Z-step and its update
        MLU.solve()
        SHARING.record_mean_Z_update(next_Z_mean=MLU.current_Z)
        # Dual update and round finalization
        SHARING.update_dual_var()
        SHARING.finalize_round()
        # Set primal objective
        match self.objective:
            case TEObjective.MLU: self._primal_objective = float(np.max(
                self.number_of_commodities * SHARING.X_mean / self.capacities
            ))
            case TEObjective.MAX_FLOW: self._primal_objective = self._total_flow
            case _: raise NotImplementedError

    def close(self):
        self.backend.close()
        if self._mlu_solver is not None:
            self._mlu_solver.close()

    def _solve_for_tm(self, tm: np.ndarray):
        PARAMS = self._solver_params

        try:
            progress_bar = ShortTQDM(range(PARAMS.OuterLoopRounds))\
                # if not self.first_solve else TQDMSpinner('Cold Start.')
            for i in progress_bar:
                self._do_network_update()
                self._do_coordinator_update()

                p = self._primal_objective
                d = self._dual_objective
                if p != 0:
                    gap = abs(d - p) / p
                else:
                    gap = np.inf
                # inf = self._sharing_wrapper.infeasibility
                # feasible = self._sharing_wrapper.is_feasible(
                #     self.feasibility_tolerance,
                #     self.optimality_tolerance
                # )

                if i % 1 == 0:
                    progress_bar.set_postfix({
                        'Obj. Val.': f'{self._primal_objective:.4f}',
                        'Obj. Gap': f'{gap*100:.2f}%',
                        # 'ADMM Inf.': f'{inf:.4f} ({feasible})',
                        'Outer Step.': f'{self._sharing_wrapper.step_size:.2f}'
                    })

                # if (2*gap < self.optimality_tolerance) and feasible:
                if (gap < self.optimality_tolerance):
                    progress_bar._pbar.close()
                    if self.first_solve:
                        print(as_success("Cold start finished!"))
                    else:
                        print(as_success("Crossed the convergance bound. Breaking early ..."))
                    break
                progress_bar.update()
            if not self._problem_description.eval_params.skip_checks:
                self._set_X_ek()
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
        # Nodes will return `X_bar`
        X_bar = self.backend.update_demands(demands)
        
        if self._first_solve:
            # On initialization, we are seeing all of this for the first
            # time, so we should initialize the outer ADMM wrapper
            self._sharing_wrapper.initialize(X_bar)
        else:
            # On subsequent iterations, only record an X-update
            # Use `np.inf` as the frobenius norm here (essentially,
            # meaning that the infeasibility is infinite). This is a
            # deliberate conservative choice.
            self._sharing_wrapper.record_mean_X_update(X_bar, np.inf)

    def _update_objective(self, tm: np.ndarray):
        pass
