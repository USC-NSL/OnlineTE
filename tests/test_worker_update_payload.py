"""Verify that path-worker replies contain only algorithm/certificate inputs."""
import unittest
from unittest.mock import Mock, patch

import array_utils
import numpy as np

from array_utils.cpu.grpc_utils import array_to_serialized_message
from protos.core.core_pb2 import NetworkUpdateRequest
from te.algorithms.base import TEObjective, TEProblemDescription, TEEvaluationParams
from te.algorithms.communication.grpc.worker_backend import OnlineTEWorkerNodeListener
from te.algorithms.formulations.path_based.distributed.coordinator import OnlineTECoordinator, PathConsensusWrapper
from te.algorithms.formulations.path_based.distributed.worker import OnlineTEWorkerNode, DenseSolver
from te.algorithms.formulations.path_based.distributed.simple_worker import SimpleOnlineTEWorkerNode, DenseSolver as SimpleDenseSolver
from te.algorithms.formulations.path_based.distributed.solver_params import PathBasedOnlineTEParameters, WorkerUpdateData
from te.algorithms.sub_algorithms.admm import DistributedSharingWrapper
from te.path_providers.packed_paths import PackedPathBatch
from tests.test_packed_path_kernels import make_provider
from tests.test_double_loop_path import test_graph


class WorkerUpdatePayloadTests(unittest.TestCase):
    def setUp(self):
        self.precision = patch.object(array_utils, "_GLOBAL_PRECISION", np.float64)
        self.precision.start()
        self.addCleanup(self.precision.stop)

    def make_worker(self, cls, objective, iterations=1000):
        worker = object.__new__(cls)
        worker._objective = objective
        worker._capacities = np.asarray([2., 3., 5., 7., 11., 13.])
        worker._total_commodity_count = 3
        batch = PackedPathBatch.from_path_provider(make_provider(), worker._capacities, True, 1)
        kwargs = dict(demands=np.asarray([1., 1.5, 2.]), path_batch=batch,
                      pgd_step=.5, pgd_iters=iterations, capacities=worker._capacities,
                      objective=objective, optimality_tolerance=1e-4)
        if cls is OnlineTEWorkerNode:
            worker._solver_params = PathBasedOnlineTEParameters(ScaleWithCapacity=True)
            worker._dense_solver = DenseSolver(**kwargs, eta=.2, adjust_step_size=True)
        else:
            worker._dense_solver = SimpleDenseSolver(**kwargs, rho=1., total_numbar_of_commodities=3)
        return worker

    def test_both_workers_send_only_required_fields(self):
        for cls in (OnlineTEWorkerNode, SimpleOnlineTEWorkerNode):
            for objective in (TEObjective.MLU, TEObjective.MAX_FLOW):
                for iterations in (1, 1000):
                    with self.subTest(worker=cls.__name__, objective=objective, iterations=iterations):
                        worker = self.make_worker(cls, objective, iterations)
                        backend = Mock(worker_id=0)
                        backend.do_inner_loop_update = worker.do_inner_loop_pgd_update
                        listener = OnlineTEWorkerNodeListener(backend)
                        response = listener.DoNetworkUpdate(NetworkUpdateRequest(
                            sharing_gap=array_to_serialized_message(np.linspace(-.1, .2, 6)),
                            sharing_dual=array_to_serialized_message(np.full(6, .1)),
                            sharing_rho=1.,
                        ), Mock())
                        expected = {WorkerUpdateData.NORM_F_DELTA_X, WorkerUpdateData.DUAL_OBJ}
                        if objective == TEObjective.MAX_FLOW:
                            expected.add(WorkerUpdateData.DEMANDS)
                        self.assertEqual(set(response.data), expected)
                        # A default-zero runtime field occupies no protobuf bytes.
                        self.assertEqual({field.name for field, _ in response.ListFields()}, {"means", "data"})
                        self.assertTrue(np.all(np.isfinite(list(response.data.values()))))

    def test_double_coordinator_accepts_same_minimal_payload(self):
        for objective in (TEObjective.MLU, TEObjective.MAX_FLOW):
            with self.subTest(objective=objective):
                coordinator = object.__new__(OnlineTECoordinator)
                coordinator._problem_description = TEProblemDescription(objective, TEEvaluationParams(), test_graph(), [])
                coordinator._graph = coordinator._problem_description.graph
                coordinator._outer_admm_wrapper = PathConsensusWrapper(6, 1.)
                coordinator._outer_admm_wrapper.initialize(np.zeros(6))
                coordinator._sharing_wrapper = DistributedSharingWrapper((6, 6), .2)
                coordinator._sharing_wrapper.initialize(np.zeros(6))
                data = {WorkerUpdateData.NORM_F_DELTA_X: 2.25, WorkerUpdateData.DUAL_OBJ: .1}
                if objective == TEObjective.MAX_FLOW:
                    data[WorkerUpdateData.DEMANDS] = 2.
                coordinator.backend = Mock()
                coordinator.backend.do_network_update.return_value = (0, np.full(6, .25), data)
                dual = np.full(6, .1)
                self.assertEqual(coordinator._do_network_update(dual), .1)
                request = coordinator.backend.do_network_update.call_args.kwargs
                self.assertEqual(set(request), {"sharing_gap", "sharing_dual", "sharing_rho"})
                np.testing.assert_allclose(request["sharing_gap"] + request["sharing_dual"], 0.)
                np.testing.assert_allclose(coordinator._sharing_wrapper.X_mean, .25)


if __name__ == "__main__":
    unittest.main()
