"""Double-loop algebra, certificates, lifecycle, and native worker integration."""
import asyncio
import contextlib
import io
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import networkx as nx
import numpy as np
import array_utils
from scipy.optimize import linprog
from google.protobuf.any_pb2 import Any

from array_utils import set_global_precision
from protos.solver_params.solver_params_pb2 import PathBasedOnlineTEParameters as ParamsMessage
from te.algorithms.base import TEObjective, TEProblemDescription, TEEvaluationParams
from te.algorithms.communication.base import DistributedSolverNodeParams, RPCParams
from te.algorithms.communication.coordinator_backend import CoordinatorBackendBase
from te.algorithms.communication.worker_backend import WorkerBackendBase
from te.algorithms.communication.grpc.asynchronous_coordinator_backend import (
    AsynchronousgRPCCoordinatorBackend, AsynchronousgRPCCoordinatorBackendParams,
)
from te.algorithms.formulations.path_based.distributed.coordinator import (
    OnlineTECoordinator, PathConsensusWrapper, DoubleLoopSolveInfo,
)
from te.algorithms.formulations.path_based.distributed.worker import OnlineTEWorkerNode, DenseSolver
from te.algorithms.formulations.path_based.distributed.solver_params import (
    PathBasedOnlineTEParameters, WorkerUpdateData, certificate_prices,
)
from te.algorithms.sub_algorithms.admm import DistributedSharingWrapper
from te.algorithms.sub_algorithms.pgd import do_packed_path_based_nesterov_pgd
from te.path_providers import build_provider, get_scheme
from te.path_providers.packed_paths import PackedPathBatch
from te.traffic_models.base import traffic_to_demands
from topologies.utils import get_edge_indexing
from utils.exceptions import SolutionInterrupted


def test_graph():
    graph = nx.DiGraph()
    for edge, capacity in zip(((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)),
                              (2., 3., 5., 7., 11., 13.)):
        graph.add_edge(*edge, capacity=capacity)
    return graph


def test_matrices():
    first = np.asarray([[0., 4., 6.], [8., 0., 2.], [3., 5., 0.]])
    return [first, first * 1.1, first * np.asarray([[1., .9, 1.2], [1.1, 1., .8], [.7, 1., 1.]])]


class InlineWorkerBackend[P](WorkerBackendBase[P]):
    """Use the real worker callbacks, with deterministic in-process transport."""
    def __init__(self, rpc_params, solver_params_cls):
        super().__init__(rpc_params, solver_params_cls)
    def start(self): self.is_alive = True; self.killed = False
    def stop(self): self.is_alive = False
    def die(self): self.stop(); self.killed = True
    def wait(self): pass
    def close(self): self.stop()
    def are_all_peers_reachable(self): return True
    @classmethod
    def backend_name(cls): return "inline-worker-test"


class InlineCoordinatorBackend[P](CoordinatorBackendBase[P]):
    def __init__(self, rpc_params, solver_params_cls):
        super().__init__(rpc_params, solver_params_cls)
        self.workers = []
        self.calls = 0

    def start(self): self.is_alive = True; self.killed = False
    def stop(self): self.is_alive = False
    def die(self): self.stop(); self.killed = True
    def wait(self): pass
    def close(self):
        for worker in self.workers: worker.close()
        self.stop()
    def are_all_peers_reachable(self): return True
    def are_all_workers_reachable(self): return []
    @classmethod
    def backend_name(cls): return "inline-coordinator-test"

    def initialize_worker_nodes(self, solver_params, topology, objective):
        for i in range(self.number_of_workers):
            worker = OnlineTEWorkerNode(DistributedSolverNodeParams(
                CommunicationBackendCLS=InlineWorkerBackend,
                RPCParams_=RPCParams(PeerIndex=i, Peers=self.rpc_params.Workers),
            ))
            worker.initialize()
            worker.set_solver_parameters(solver_params, self.number_of_workers, objective)
            worker.set_topology(topology)
            self.workers.append(worker)

    def update_demands(self, demands):
        return np.mean([w.update_demands(chunk) for w, chunk in
                        zip(self.workers, np.array_split(demands, self.number_of_workers))], axis=0)

    def do_network_update(self, sharing_gap, sharing_dual, sharing_rho):
        self.calls += 1
        updates = [w.do_inner_loop_pgd_update(sharing_gap, sharing_dual, sharing_rho)
                   for w in self.workers]
        data = {key: sum(u[2][key] for u in updates) for key in updates[0][2]}
        return max(u[0] for u in updates), np.mean([u[1] for u in updates], axis=0), data

    def get_X_ek(self):
        return np.hstack([worker.report_chunk() for worker in self.workers])


def make_coordinator(objective=TEObjective.MLU, scaled=True, precision="double", **kwargs):
    # Production nodes use separate processes; isolate precision per test instance.
    array_utils._GLOBAL_PRECISION = None
    graph = test_graph()
    problem = TEProblemDescription(objective, TEEvaluationParams(), graph, [])
    params = PathBasedOnlineTEParameters(ScaleWithCapacity=scaled, Precision=precision,
                                       KernelThreads=1, **kwargs)
    coordinator = OnlineTECoordinator(problem, params, DistributedSolverNodeParams(
        CommunicationBackendCLS=InlineCoordinatorBackend,
        RPCParams_=RPCParams(Workers=(("inline", 1), ("inline", 2))),
    ))
    coordinator.make_lp()
    return coordinator


def centralized_objective(graph, demands, objective):
    """Independent path-flow LP with precisely the same candidate path policy."""
    provider = build_provider(T=8, graph=graph, per_commodity_provider=get_scheme(),
                              edge_indexing=get_edge_indexing(graph))
    batch = PackedPathBatch.from_path_provider(provider, np.asarray([c for _, _, c in graph.edges(data="capacity")]), False, 1)
    edge = np.zeros((batch.num_edges, batch.num_paths))
    demand = np.zeros((batch.num_commodities, batch.num_paths))
    for k in range(batch.num_commodities):
        start, end = batch.commodity_path_offsets[k:k+2]
        demand[k, start:end] = 1
        for p in range(start, end):
            edge[batch.path_edges[batch.path_edge_offsets[p]:batch.path_edge_offsets[p+1]], p] = 1
    caps = np.asarray([c for _, _, c in graph.edges(data="capacity")])
    if objective == TEObjective.MLU:
        result = linprog(np.r_[np.zeros(batch.num_paths), 1.],
                         A_ub=np.column_stack([edge, -caps]), b_ub=np.zeros(batch.num_edges),
                         A_eq=np.column_stack([demand, np.zeros(batch.num_commodities)]), b_eq=demands,
                         bounds=(0, None), method="highs")
    else:
        result = linprog(-np.ones(batch.num_paths), A_ub=np.vstack([edge, demand]),
                         b_ub=np.r_[caps, demands], bounds=(0, None), method="highs")
    if not result.success:
        raise AssertionError(result.message)
    return result.fun if objective == TEObjective.MLU else -result.fun


class DoubleLoopPathTests(unittest.TestCase):
    def setUp(self):
        self.precision_patch = patch.object(array_utils, "_GLOBAL_PRECISION", None)
        self.precision_patch.start()
        set_global_precision("double")
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.precision_patch.stop()

    def test_implicit_consensus_matches_explicit_residuals(self):
        wrapper = PathConsensusWrapper(2, .7)
        initial, x, z = np.asarray([2., 3.]), np.asarray([4., 5.]), np.asarray([3., 4.])
        wrapper.initialize(initial)
        wrapper.record_X_update(x)
        np.testing.assert_allclose(wrapper.get_Z_step_bias(), x)
        wrapper.record_Z_update(z)
        wrapper.update_dual_var()
        wrapper.finalize_round()
        self.assertAlmostEqual(wrapper.primal_infeasibility, np.linalg.norm(x-z))
        self.assertAlmostEqual(wrapper.dual_infeasibility, .7*np.linalg.norm(z-initial))
        np.testing.assert_allclose(-wrapper.get_X_step_bias(), z-(x-z))
        self.assertEqual(wrapper._A, True)

    def test_corrected_sharing_mean_solves_dense_subproblem(self):
        coordinator = make_coordinator()
        try:
            coordinator._update_constraits(test_matrices()[0])
            outer, sharing = coordinator._outer_admm_wrapper, coordinator._sharing_wrapper
            sharing._dual_var[:] = np.arange(6)*.01
            new_mean = sharing.X_mean + .02
            sharing.record_mean_X_update(new_mean, 6*.02**2*6)
            target = -outer.get_X_step_bias()
            eta, rho, K = coordinator.solver_params.Eta, coordinator.solver_params.Rho, 6
            # Solve the Zbar quadratic's first-order system directly.
            expected = np.linalg.solve(np.eye(6)*(rho*K*K+eta*K),
                                       rho*K*target+eta*K*(new_mean+sharing.dual_var))
            coordinator._update_sharing_mean()
            np.testing.assert_allclose(sharing.Z_mean, expected, rtol=1e-13)
            self.assertEqual(sharing.step_size, eta)
        finally: coordinator.close()

    def test_demand_refresh_preserves_duals_and_invalidates_diagnostics(self):
        coordinator = make_coordinator()
        try:
            coordinator._update_constraits(test_matrices()[0])
            sharing = coordinator._sharing_wrapper
            sharing._dual_var[:] = .2
            before_z = sharing.Z_mean.copy()
            coordinator._outer_admm_wrapper._dual_var[:] = .3
            coordinator._update_constraits(test_matrices()[1])
            np.testing.assert_array_equal(sharing.dual_var, .2)
            np.testing.assert_array_equal(sharing.Z_mean, before_z)
            np.testing.assert_array_equal(coordinator._outer_admm_wrapper.dual_var, .3)
            with self.assertRaises(RuntimeError): _ = sharing.primal_infeasibility
            self.assertEqual(sharing._round_stage, "finalized")
        finally: coordinator.close()

    def test_params_validation_and_proto_roundtrip(self):
        params = PathBasedOnlineTEParameters(Timeout=12.5)
        transport = object.__new__(InlineCoordinatorBackend)
        transport._solver_params_cls = PathBasedOnlineTEParameters
        transport.set_solver_param_message_type()
        packed = Any()
        packed.Pack(transport.serialize_solver_params(params))
        self.assertEqual(transport.deserialize_solver_params(packed), params)
        self.assertEqual(ParamsMessage.DESCRIPTOR.fields_by_name["Timeout"].number, 14)
        for name, value in (("Eta", 0.), ("Rho", np.nan), ("Gamma", -1.),
                            ("Timeout", float("inf")), ("Timeout", 0.),
                            ("InnerLoopRounds", 0), ("SwitchOptimalityTolerance", None)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                PathBasedOnlineTEParameters(**{name: value})
        self.assertIsNone(PathBasedOnlineTEParameters(Timeout=None).Timeout)
        with self.assertWarns(FutureWarning): PathBasedOnlineTEParameters(OuterLoopRounds=1)

    def test_worker_proximal_reference_and_zero_demand(self):
        coordinator = make_coordinator(TEObjective.MAX_FLOW)
        try:
            worker = coordinator.backend.workers[0]
            worker.update_demands(np.zeros(3))
            self.assertTrue(np.all(np.isfinite(worker._dense_solver._pgd_steps)))
            _, mean, data = worker.do_inner_loop_pgd_update(np.zeros(6), np.zeros(6), 1.)
            self.assertEqual(data[WorkerUpdateData.DEMANDS], 0)
            np.testing.assert_array_equal(mean, 0.)
            worker.update_demands(np.ones(3))
            solver = worker._dense_solver
            for _ in range(2):
                previous = solver._Y_tk.copy()
                solver.update(np.zeros(6))
                np.testing.assert_array_equal(solver._Y_tk_old, previous)
        finally: coordinator.close()

    def test_worker_response_contains_only_solve_and_certificate_data(self):
        coordinator = make_coordinator(SwitchIterations=1, SwitchOptimalityTolerance=1e-14)
        try:
            coordinator._update_constraits(test_matrices()[0])
            worker = coordinator.backend.workers[0]
            _, _, data = worker.do_inner_loop_pgd_update(np.linspace(-2, 3, 6), np.zeros(6), 1.)
            self.assertEqual(set(data), {WorkerUpdateData.NORM_F_DELTA_X, WorkerUpdateData.DUAL_OBJ})
        finally: coordinator.close()

    def test_certificate_prices_are_feasible_in_both_units(self):
        for dtype in (np.float32, np.float64):
            for scaled in (False, True):
                caps = np.ones(6, dtype=dtype) if scaled else np.arange(1, 7, dtype=dtype)
                for objective in (TEObjective.MLU, TEObjective.MAX_FLOW):
                    prices = certificate_prices(np.arange(-2, 4, dtype=dtype), .7, caps, objective, 6)
                    self.assertTrue(np.all(prices >= 0))
                    if objective == TEObjective.MLU:
                        self.assertLessEqual(float(prices.astype(float) @ caps.astype(float)), 1.)
                    else:
                        np.testing.assert_allclose(prices*6, np.maximum(np.arange(-2, 4)*.7, 0), rtol=1e-6)
                np.testing.assert_array_equal(certificate_prices(np.zeros(6, dtype=dtype), 1., caps, TEObjective.MLU, 6), 0.)

    def test_warm_timeout_finishes_round_and_keeps_state(self):
        coordinator = make_coordinator(Timeout=.1)
        try:
            coordinator._update_constraits(test_matrices()[0])
            coordinator._first_solve = False
            with patch("te.algorithms.formulations.path_based.distributed.coordinator.time.monotonic",
                       side_effect=[0., .2, .2, .2]), patch.object(coordinator, "_check_convergence", return_value=False):
                coordinator._solve_for_tm(test_matrices()[0])
            self.assertEqual(coordinator.last_solve_info.termination_reason, "timeout")
            self.assertEqual(coordinator.last_solve_info.outer_rounds, 1)
            self.assertEqual(coordinator.backend.calls, coordinator.solver_params.InnerLoopRounds)
            self.assertFalse(coordinator._failed_round)
            self.assertEqual(coordinator._outer_admm_wrapper._round_stage, "finalized")
            coordinator._update_constraits(test_matrices()[1])
        finally: coordinator.close()

    def test_unlimited_warm_start_and_early_convergence(self):
        coordinator = make_coordinator(Timeout=None)
        try:
            coordinator._update_constraits(test_matrices()[0])
            coordinator._first_solve = False
            with patch.object(coordinator, "_check_convergence", return_value=True):
                coordinator._solve_for_tm(test_matrices()[0])
            self.assertEqual(coordinator.last_solve_info.termination_reason, "converged")
            self.assertEqual(coordinator.last_solve_info.outer_rounds, 1)
        finally: coordinator.close()

    def test_worker_failure_requires_restart(self):
        coordinator = make_coordinator()
        try:
            coordinator._update_constraits(test_matrices()[0])
            with patch.object(coordinator.backend, "do_network_update", side_effect=RuntimeError("worker failed")):
                with self.assertRaisesRegex(RuntimeError, "worker failed"):
                    coordinator._solve_for_tm(test_matrices()[0])
            self.assertEqual(coordinator.last_solve_info.termination_reason, "backend_failure")
            with self.assertRaisesRegex(RuntimeError, "restart"):
                coordinator._update_constraits(test_matrices()[1])
        finally: coordinator.close()

    def test_partial_barrier_rejected_before_start(self):
        with self.assertRaisesRegex(ValueError, "full worker barrier"):
            OnlineTECoordinator(TEProblemDescription(TEObjective.MLU, TEEvaluationParams(), test_graph(), []),
                PathBasedOnlineTEParameters(), DistributedSolverNodeParams(
                    CommunicationBackendCLS=AsynchronousgRPCCoordinatorBackend,
                    RPCParams_=AsynchronousgRPCCoordinatorBackendParams(Workers=(("localhost", 1), ("localhost", 2)), BarrierSize=1),
                ))

    def test_full_barrier_never_reuses_missing_response(self):
        backend = object.__new__(AsynchronousgRPCCoordinatorBackend)
        backend._rpc_params = AsynchronousgRPCCoordinatorBackendParams(Workers=(("x", 1), ("x", 2)))
        async def update(i, message): return None if i else Mock()
        backend._stub_net_update = update
        with self.assertRaisesRegex(RuntimeError, "Missing worker"):
            asyncio.run(backend._full_network_update(Mock()))

    def test_sequences_match_independent_lp_and_pass_sanity_checks(self):
        for objective in (TEObjective.MLU, TEObjective.MAX_FLOW):
            for scaled in (False, True):
                for precision in ("single", "double"):
                    with self.subTest(objective=objective, scaled=scaled, precision=precision):
                        coordinator = make_coordinator(objective, scaled, precision, Timeout=20.)
                        try:
                            # Bound this test's cold start without changing the solver policy.
                            original = coordinator._single_solve_step
                            def bounded_round():
                                if coordinator.last_solve_info.outer_rounds >= 5000:
                                    self.fail("Cold start did not converge in 5000 outer rounds")
                                return original()
                            coordinator._single_solve_step = bounded_round
                            for tm in test_matrices():
                                reference = centralized_objective(coordinator.graph, traffic_to_demands(tm), objective)
                                coordinator._current_TM = tm
                                coordinator.solve_for_tm(tm)
                                info = coordinator.last_solve_info
                                self.assertEqual(info.termination_reason, "converged")
                                self.assertLessEqual(abs(coordinator.current_objective-reference)/reference, .01)
                                check = coordinator.check_results[-1]
                                self.assertEqual(check.leaks[1], [])
                                if objective == TEObjective.MAX_FLOW:
                                    self.assertEqual(check.congestions[1], [])
                                    self.assertEqual(check.satisfaction[1], [])
                                self.assertEqual(coordinator.backend.calls % coordinator.solver_params.InnerLoopRounds, 0)
                                self.assertLessEqual(info.objective_gap, .01)
                        finally: coordinator.close()

    def test_pdlp_controller_sequence(self):
        for objective in (TEObjective.MLU, TEObjective.MAX_FLOW):
            coordinator = make_coordinator(objective)
            try:
                original = coordinator._single_solve_step
                def bounded_round():
                    if coordinator.last_solve_info.outer_rounds >= 5000:
                        self.fail("PDLP-backed cold start did not converge")
                    return original()
                coordinator._single_solve_step = bounded_round
                for tm in test_matrices():
                    coordinator._current_TM = tm
                    coordinator.solve_for_tm(tm)
                    self.assertEqual(coordinator.last_solve_info.termination_reason, "converged")
                    self.assertEqual(coordinator.check_results[-1].leaks[1], [])
                    if objective == TEObjective.MAX_FLOW:
                        self.assertEqual(coordinator.check_results[-1].congestions[1], [])
            finally: coordinator.close()


if __name__ == "__main__":
    unittest.main()
