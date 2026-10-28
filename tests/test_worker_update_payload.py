"""Verify that path-worker replies contain only algorithm/certificate inputs."""
import unittest
from unittest.mock import Mock, patch

import array_utils
import numpy as np

from array_utils.cpu.grpc_utils import array_to_serialized_message, serialized_message_to_array
from protos.core.core_pb2 import NetworkUpdateRequest
from te.algorithms.base import TEObjective
from te.algorithms.communication.grpc.worker_backend import OnlineTEWorkerNodeListener
from te.algorithms.formulations.path_based.distributed.worker import OnlineTEWorkerNode, DenseSolver
from te.algorithms.formulations.path_based.distributed.solver_params import PathBasedOnlineTEParameters, WorkerUpdateData
from te.path_providers.packed_paths import PackedPathBatch
from tests.test_packed_path_kernels import dense_alpha, make_provider


class WorkerUpdatePayloadTests(unittest.TestCase):
    def setUp(self):
        self.precision = patch.object(array_utils, "_GLOBAL_PRECISION", np.float64)
        self.precision.start()
        self.addCleanup(self.precision.stop)

    def make_worker(self, objective, iterations=1000):
        worker = object.__new__(OnlineTEWorkerNode)
        worker._objective = objective
        worker._capacities = np.asarray([2., 3., 5., 7., 11., 13.])
        worker._total_commodity_count = 3
        batch = PackedPathBatch.from_path_provider(make_provider(), worker._capacities, True, 1)
        kwargs = dict(demands=np.asarray([1., 1.5, 2.]), path_batch=batch,
                      pgd_step=.5, pgd_iters=iterations, capacities=worker._capacities,
                      objective=objective, optimality_tolerance=1e-4)
        worker._solver_params = PathBasedOnlineTEParameters()
        worker._dense_solver = DenseSolver(**kwargs, total_numbar_of_commodities=3)
        return worker

    def test_worker_sends_required_fields_and_reports_matching_iterates(self):
        provider = make_provider()
        for objective in (TEObjective.MLU, TEObjective.MAX_FLOW):
            for iterations in (1, 1000):
                worker = self.make_worker(objective, iterations)
                solver = worker._dense_solver
                batch = solver._paths
                blocks = [dense_alpha(provider, k, np.float64) * batch.edge_scale[:, None]
                          for k in range(batch.num_commodities)]
                factors = np.zeros(batch.num_edges)
                backend = Mock(worker_id=0)
                backend.do_inner_loop_update = worker.do_inner_loop_pgd_update
                listener = OnlineTEWorkerNodeListener(backend)
                for round_id in range(3):
                    with self.subTest(objective=objective, iterations=iterations, round=round_id):
                        previous = solver._Y_tk.copy()
                        previous_assignment = np.column_stack([
                            block @ previous[start:end] * solver._demands[k]
                            for k, (block, start, end) in enumerate(zip(
                                blocks, batch.commodity_path_offsets[:-1], batch.commodity_path_offsets[1:]))
                        ])
                        expected_feasible = previous * np.asarray([
                            np.min(factors[batch.path_edges[start:end]])
                            for start, end in zip(batch.path_edge_offsets[:-1], batch.path_edge_offsets[1:])
                        ])
                        response = listener.DoNetworkUpdate(NetworkUpdateRequest(
                            sharing_bias=array_to_serialized_message(np.linspace(-.1, .2, 6)),
                            edge_dual=array_to_serialized_message(np.full(6, .1)),
                            sharing_rho=1.,
                            nested=False,
                            load_factor=array_to_serialized_message(factors),
                        ), Mock())
                        expected = {WorkerUpdateData.NORM_F_DELTA_X, WorkerUpdateData.DUAL_OBJ}
                        if objective == TEObjective.MAX_FLOW:
                            expected.add(WorkerUpdateData.DEMANDS)
                        self.assertEqual(set(response.data), expected)
                        # A default-zero runtime field occupies no protobuf bytes.
                        self.assertEqual({field.name for field, _ in response.ListFields()}, {"means", "data"})
                        self.assertTrue(np.all(np.isfinite(list(response.data.values()))))
                        assignment = np.column_stack([
                            block @ solver._Y_tk[start:end] * solver._demands[k]
                            for k, (block, start, end) in enumerate(zip(
                                blocks, batch.commodity_path_offsets[:-1], batch.commodity_path_offsets[1:]))
                        ])
                        mean = serialized_message_to_array(response.means)
                        np.testing.assert_allclose(mean, assignment.mean(axis=1), rtol=1e-12)
                        self.assertAlmostEqual(response.data[WorkerUpdateData.NORM_F_DELTA_X],
                                               np.sum((assignment - previous_assignment) ** 2))
                        np.testing.assert_array_equal(solver._Y_tk_old, solver._Y_tk)
                        if objective == TEObjective.MAX_FLOW:
                            feasible_assignment = np.column_stack([
                                block @ expected_feasible[start:end] * solver._demands[k]
                                for k, (block, start, end) in enumerate(zip(
                                    blocks, batch.commodity_path_offsets[:-1], batch.commodity_path_offsets[1:]))
                            ])
                            np.testing.assert_allclose(worker.report_chunk(), feasible_assignment, rtol=1e-12)
                            expected_flow = np.dot(expected_feasible, np.repeat(solver._demands, batch.path_counts))
                            self.assertAlmostEqual(response.data[WorkerUpdateData.DEMANDS], expected_flow)
                            if round_id > 0:
                                self.assertGreater(expected_flow, 0)
                        else:
                            np.testing.assert_allclose(worker.report_chunk(), assignment, rtol=1e-12)
                        # The next request carries factors for this round's iterate.
                        factors = np.ones(batch.num_edges)
                        np.divide(1, mean * worker._total_commodity_count, out=factors, where=mean > 0)
                        np.clip(factors, 0, 1, out=factors)


if __name__ == "__main__":
    unittest.main()
