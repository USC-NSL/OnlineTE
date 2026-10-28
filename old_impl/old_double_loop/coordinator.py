import time
import numpy as np
import asyncio.exceptions
from typing import Optional, Tuple
from te.algorithms.base import *
from te.traffic_models.base import traffic_to_demands
from utils.exceptions import SolutionInterrupted
from utils.logging import as_info, as_success, as_warning, as_fail, TQDMSpinner, TQDMTimer
from array_utils import set_global_precision
from array_utils.cpu.types import *
from te.algorithms.sub_algorithms.admm import VectorConsensusWrapper, DistributedSharingWrapper, AdaptiveStepSizeParams
from te.algorithms.communication import *
from te.algorithms.sub_algorithms.capacity_qp import CapacityQP, CapacityQPMode
from .solver_params import PathBasedOnlineTEParameters, WorkerUpdateData


class OnlineTECoordinator(TELP[PathBasedOnlineTEParameters], DistributedSolverNodeBase):
    def __init__(
        self, 
        problem_description: TEProblemDescription,
        solver_params: PathBasedOnlineTEParameters,
        node_params: DistributedSolverNodeParams
    ) -> None:
        super().__init__(
            problem_description=problem_description,
            solver_params=solver_params,
            node_params=node_params
        )
        # MLU QP solver
        self._mlu_solver: CapacityQP = CapacityQP(
            # mode=CapacityQPMode.CONSENSUS,
            mode=CapacityQPMode.SHARING,
            rho=solver_params.Rho,
            num_edges=self.num_edges,
            num_commodities=self.number_of_commodities,
            feasibility_tolerance=0.01 * self.feasibility_tolerance,
            optimality_tolerance=0.01 * self.optimality_tolerance,
            objective=self.objective,
            capacities=None,
            scratch=0
        )
        # Outer ADMM
        self._outer_admm_wrapper: Optional[VectorConsensusWrapper] = None
        # Inner ADMM
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
        self._nested = True

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
            self._solver_params,
            self._graph,
            self.objective
        )
        # Finalize all controller states
        self._initialize_variables_and_residuals()

    @property
    def alg_name(self) -> str:
        return 'Path Based OnlineTE'

    @property
    def current_objective(self) -> float:
        return self._primal_objective

    def _initialize_variables_and_residuals(self):
        # Build the MLU QP
        self._mlu_solver.build()
        # Initialize the consensus wrapper
        self._outer_admm_wrapper = VectorConsensusWrapper(
            self.number_of_edges, self._solver_params.Rho,
            adaptation=AdaptiveStepSizeParams(
                infeasibility_bound=5,
                multiplicative_factor=2,
                strike_count=3
            )
        )
        # Initialize the sharing wrapper
        self._sharing_wrapper = DistributedSharingWrapper(
            (self.number_of_edges, self.number_of_commodities),
            self._solver_params.Eta,
            adaptation=AdaptiveStepSizeParams(
                infeasibility_bound=5,
                multiplicative_factor=2,
                strike_count=3
            )
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
        
    def _get_F(self) -> np.ndarray:
        # Changed for sharing scaling factor
        return -self._outer_admm_wrapper.get_X_step_bias()
    
    def _set_X_ek(self):
        """Store the public assignment in physical flow units.

        With capacity scaling enabled, worker-side path operators return
        ``y[e, k] = x[e, k] / C[e]`` so the outer ADMM and MLU controller
        operate against unit capacities.  ``TELP.current_assignment`` is
        consumed by physical-flow checks, however, so undo that change of
        variables only at this reporting boundary.
        """
        self._X_ek = self.backend.get_X_ek() * self._capacities[:, None]

    def _do_single_loop_network_update(self):
        SHARING = self._sharing_wrapper
        _, X_bar, data = self.backend.do_network_update(
            sharing_bias=SHARING.get_X_step_bias(),
            edge_dual=SHARING.unscaled_dual_var,
            sharing_rho=SHARING.step_size,
            nested=False
        )
        if self.objective == TEObjective.MAX_FLOW:
            self._total_flow = data[WorkerUpdateData.DEMANDS]
        SHARING.record_mean_X_update(
            next_X_mean=X_bar,
            delta_X_frobenius=data[WorkerUpdateData.NORM_F_DELTA_X]
        )
        match self.objective:
            case TEObjective.MLU:
                self._dual_objective = data[WorkerUpdateData.DUAL_OBJ]
            case TEObjective.MAX_FLOW:
                self._dual_objective = - data[WorkerUpdateData.DUAL_OBJ] - np.sum(SHARING.unscaled_dual_var)
            case _: raise NotImplementedError

    def _do_double_loop_network_update(self):
        SHARING = self._sharing_wrapper
        CONSENSUS = self._outer_admm_wrapper
        K = self.number_of_commodities
        F_E = self._get_F()
        RHO = CONSENSUS.step_size
        ETA = SHARING.step_size
        for _ in range(self._solver_params.InnerLoopRounds):
            _, X_bar, data = self.backend.do_network_update(
                sharing_bias=SHARING.get_X_step_bias(),
                # edge_dual=CONSENSUS.unscaled_dual_var,
                edge_dual=SHARING.unscaled_dual_var / self.number_of_commodities,
                sharing_rho=SHARING.step_size,
                nested=True
            )
            if self.objective == TEObjective.MAX_FLOW:
                self._total_flow = data[WorkerUpdateData.DEMANDS]
            SHARING.record_mean_X_update(X_bar, data[WorkerUpdateData.NORM_F_DELTA_X])
            # Changed for sharing scaling factor
            SHARING.record_mean_Z_update(
                (F_E + (ETA / RHO) * (SHARING.scaled_dual_var + SHARING.X_mean)) / (K + (ETA / RHO))
                # (F_E / K + (ETA / RHO) * (SHARING.scaled_dual_var + SHARING.X_mean)) / (1 + (ETA / RHO))
            )
            SHARING.update_dual_var()
            SHARING.finalize_round()
        match self.objective:
            case TEObjective.MLU:
                self._dual_objective = data[WorkerUpdateData.DUAL_OBJ]
            case TEObjective.MAX_FLOW:
                self._dual_objective = - data[WorkerUpdateData.DUAL_OBJ] - np.sum(SHARING.unscaled_dual_var)
            case _: raise NotImplementedError

    def _do_network_update(self):
        if self._nested:
            self._do_double_loop_network_update()
        else:
            self._do_single_loop_network_update()

    # def _do_network_update(self):
    #     SHARING = self._sharing_wrapper
    #     CONSENSUS = self._outer_admm_wrapper
    #     K = self.number_of_commodities
    #     F_E = self._get_F()
    #     RHO = CONSENSUS.step_size
    #     ETA = SHARING.step_size
    #     for _ in range(self._solver_params.InnerLoopRounds):
    #         _, X_bar, data = self.backend.do_network_update(
    #             sharing_bias=SHARING.get_X_step_bias(),
    #             edge_dual=CONSENSUS.unscaled_dual_var,
    #             sharing_rho=SHARING.step_size,
    #             nested=True
    #         )
    #         if self.objective == TEObjective.MAX_FLOW:
    #             self._total_flow = data[WorkerUpdateData.DEMANDS]
    #         SHARING.record_mean_X_update(X_bar, data[WorkerUpdateData.NORM_F_DELTA_X])
    #         # Changed for sharing scaling factor
    #         SHARING.record_mean_Z_update(
    #             (F_E / K + (ETA / RHO) * (SHARING.scaled_dual_var + SHARING.X_mean)) / (1 + (ETA / RHO))
    #         )
    #         SHARING.update_dual_var()
    #         SHARING.finalize_round()
    #     match self.objective:
    #         case TEObjective.MLU:
    #             self._dual_objective = data[WorkerUpdateData.DUAL_OBJ]
    #         case TEObjective.MAX_FLOW:
    #             self._dual_objective = - data[WorkerUpdateData.DUAL_OBJ] - np.sum(SHARING.unscaled_dual_var)
    #         case _: raise NotImplementedError

    def _do_single_loop_coordinator_update(self):
        MLU = self._mlu_solver
        SHARING = self._sharing_wrapper
        # Update `Z_bar`
        MLU.update_F_m(
            SHARING.get_Z_step_bias(),
            SHARING.step_size
        )
        # Z-step and its update
        MLU.solve()
        # SHARING.record_mean_Z_update(next_Z_mean=MLU.current_Z / self.number_of_commodities)
        SHARING.record_mean_Z_update(next_Z_mean=MLU.current_Z)
        # Dual update and round finalization
        SHARING.update_dual_var()
        SHARING.finalize_round()
        # Set primal objective
        match self.objective:
            case TEObjective.MLU:
                total_assignment = self.number_of_commodities * SHARING.X_mean
                self._primal_objective = float(np.max(total_assignment))
            case TEObjective.MAX_FLOW:
                # The Max-Flow objective is scaled by number of commodities.
                self._primal_objective = -self._total_flow / self.number_of_commodities
            case _: raise NotImplementedError

    def _do_double_loop_coordinator_update(self):
        MLU = self._mlu_solver
        CONSENSUS = self._outer_admm_wrapper
        CONSENSUS.record_X_update(
            self.number_of_commodities * self._sharing_wrapper.X_mean
        )
        MLU.update_F_m(
            # Changed for sharing scaling factor
            CONSENSUS.get_Z_step_bias() / self.number_of_commodities,
            CONSENSUS.step_size
        )
        MLU.solve()
        CONSENSUS.record_Z_update(next_Z=MLU.current_Z * self.number_of_commodities)
        CONSENSUS.update_dual_var()
        CONSENSUS.finalize_round()
        # Set primal objective
        match self.objective:
            case TEObjective.MLU:
                self._primal_objective = float(np.max(CONSENSUS.X))
            case TEObjective.MAX_FLOW:
                # The Max-Flow objective is scaled by number of commodities.
                self._primal_objective = -self._total_flow / self.number_of_commodities
            case _: raise NotImplementedError

    # def _do_coordinator_update(self):
    #     MLU = self._mlu_solver
    #     CONSENSUS = self._outer_admm_wrapper
    #     CONSENSUS.record_X_update(
    #         self.number_of_commodities * self._sharing_wrapper.X_mean
    #     )
    #     MLU.update_F_m(
    #         # Changed for sharing scaling factor
    #         CONSENSUS.get_Z_step_bias(),
    #         CONSENSUS.step_size
    #     )
    #     MLU.solve()
    #     CONSENSUS.record_Z_update(next_Z=MLU.current_Z)
    #     CONSENSUS.update_dual_var()
    #     CONSENSUS.finalize_round()
    #     # Set primal objective
    #     match self.objective:
    #         case TEObjective.MLU:
    #             self._primal_objective = float(np.max(CONSENSUS.X))
    #         case TEObjective.MAX_FLOW:
    #             # The Max-Flow objective is scaled by number of commodities.
    #             self._primal_objective = -self._total_flow / self.number_of_commodities
    #         case _: raise NotImplementedError

    def _do_coordinator_update(self):
        if self._nested:
            self._do_double_loop_coordinator_update()
        else:
            self._do_single_loop_coordinator_update()

    def switch_to_nested(self):
        assert not self._nested
        # Warm-start the consensus wrapper
        self._outer_admm_wrapper.warm_start(
            self._sharing_wrapper.X_mean * self.number_of_commodities,
            self._sharing_wrapper.Z_mean * self.number_of_commodities,
            self._sharing_wrapper.unscaled_dual_var * self.number_of_commodities
        )
        self._nested = True

    def switch_from_nested(self):
        # assert self._nested
        self._nested = False

    def close(self):
        self.backend.close()
        if self._mlu_solver is not None:
            self._mlu_solver.close()

    def check_admm_parimal_feasibility(self) -> bool:
        if self._nested:
            return self._outer_admm_wrapper.is_primal_feasible(
                eps_abs=self.feasibility_tolerance,
                eps_rel=self.optimality_tolerance
            )
        else:
            return self._sharing_wrapper.is_primal_feasible(
                eps_abs=self.feasibility_tolerance,
                eps_rel=self.optimality_tolerance
            )

    def check_congestion(self) -> float:
        target = self._mlu_solver.current_u
        actual = float(np.max(self._sharing_wrapper.X_mean) * self.number_of_commodities)
        return actual - target

    def _single_solve_step(self) -> Tuple[float, float, float, bool, float]:
        # time.sleep(0.1)
        CONSENSUS = self._outer_admm_wrapper
        self._do_network_update()
        self._do_coordinator_update()

        p = self._primal_objective
        d = self._dual_objective
        if p != 0:
            gap = abs(d - p) / abs(p)
        else:
            gap = np.inf
        admm_primal_feasible = self.check_admm_parimal_feasibility()
        congestion = self.check_congestion()
        # if gap <= self.optimality_tolerance and admm_primal_feasible and not self._nested:
        #     print(as_success(f'Objective within tolerance! Polishing ...'))
        #     self.switch_to_nested()
        return p, gap, CONSENSUS.step_size, admm_primal_feasible, congestion

    def _cold_start(self) -> bool:
        with TQDMSpinner('Cold Start.') as progress_bar:
            while self.backend.is_alive:
                val, gap, step, admm_primal_feasible, congestion = self._single_solve_step()
                progress_bar.set_postfix({
                    'Obj. Val.': f'{val:.4f}',
                    'Obj. Gap': f'{gap*100:.2f}%',
                    'ADMM Penalty.': f'{step:.2f}',
                    'Congestion': f'{congestion*100:.2f}%',
                    'NESTED': self._nested
                })
                if (gap < self.optimality_tolerance and \
                    admm_primal_feasible and \
                    congestion < self.feasibility_tolerance):
                    return True
                progress_bar.update()
        return False

    def _warm_start(self) -> bool:
        with TQDMTimer(timeout=self._solver_params.Timeout, desc="Warm Start.") as progress_bar:
            while self.backend.is_alive and progress_bar.update():
                val, gap, step, admm_primal_feasible, congestion = self._single_solve_step()
                progress_bar.set_postfix({
                    'Obj. Val.': f'{val:.4f}',
                    'Obj. Gap': f'{gap*100:.2f}%',
                    'ADMM Penalty.': f'{step:.2f}',
                    'Congestion': f'{congestion*100:.2f}%'
                })
                if (gap < self.optimality_tolerance and \
                    admm_primal_feasible and \
                    congestion < self.feasibility_tolerance):
                    return True
        return not progress_bar.timed_out

    def _solve_for_tm(self, tm: np.ndarray):
        try:
            if self.first_solve:
                if self._cold_start():
                    print(as_success('Cold start finished!'))
                else:
                    print((as_fail('Cold start failed or interrupted!')))
            else:
                if self._warm_start():
                    print(as_success('Warm start finished!'))
                else:
                    print((as_warning('Timeout on warm start!')))
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
        # Nodes will return `X_bar` which is `sharing_mean_1`
        X_bar = self.backend.update_demands(demands)

        if self._first_solve:
            # On initialization, we are seeing all of this for the first
            # time, so we should initialize the outer ADMM wrapper
            self._outer_admm_wrapper.initialize(X_bar * self.number_of_commodities)
            self._sharing_wrapper.initialize(X_bar)
        else:
            # Demand changes are external to ADMM: refresh the X reference without
            # resetting the accumulated Z means or either dual variable.
            self._sharing_wrapper.refresh_mean_X(X_bar)

        # Switch to the single-loop solver
        self.switch_from_nested()
        # Reset the barrier
        self.backend.reset_barrier()

    def _update_objective(self, tm: np.ndarray):
        pass
