import pickle
import unittest
from unittest.mock import Mock

import networkx as nx
import numpy as np
from numba.typed import List as NumbaList
from google.protobuf.json_format import MessageToDict

from protos.solver_params.solver_params_pb2 import PathBasedOnlineTEParameters as PathParamsMessage
from te.algorithms.formulations.path_based.distributed.packed_paths import PackedPathBatch
from te.algorithms.formulations.path_based.distributed.coordinator import OnlineTECoordinator
from te.algorithms.formulations.path_based.distributed.solver_params import PathBasedOnlineTEParameters
from te.algorithms.formulations.path_based.distributed.worker import DenseSolver
from te.algorithms.base import TEObjective
from te.algorithms.sub_algorithms.pgd import (
    do_packed_path_based_maxflow_pgd,
    do_packed_path_based_nesterov_pgd,
    do_path_based_maxflow_pgd,
    do_path_based_nesterov_pgd,
)
from te.algorithms.sub_algorithms.constraint_sanity_checks import (
    check_capacity_constraint,
    check_flow_satisfaction,
)
from te.traffic_models.base import traffic_to_commodity
from topologies.utils import get_edge_indexing
from te.path_providers import PathProvider
from te.path_providers.sparse_ops import (
    path_based_to_edge_based_mean_nnz,
    path_based_to_edge_based_nnz,
)


def make_provider() -> PathProvider:
    paths = (
        ((0, 1),),
        ((1, 2), (2, 3)),
        ((0, 4), (4, 5), (0, 5)),
    )
    rows = []
    cols = []
    for commodity_paths in paths:
        commodity_rows = []
        commodity_cols = []
        for t, edges in enumerate(commodity_paths):
            commodity_rows.extend(edges)
            commodity_cols.extend([t] * len(edges))
        rows.append(np.asarray(commodity_rows, dtype=np.int32))
        cols.append(np.asarray(commodity_cols, dtype=np.int32))
    return PathProvider(
        shape=(3, 6, 3),
        rows=rows,
        cols=cols,
        beta=np.asarray([1, 2, 3], dtype=np.int32),
    )


def dense_alpha(provider: PathProvider, k: int, dtype: np.dtype) -> np.ndarray:
    beta = int(provider.beta[k])
    alpha = np.zeros((provider.shape[1], beta), dtype=dtype)
    alpha[provider.rows[k], provider.cols[k]] = 1
    return alpha


def unpack_blocks(batch: PackedPathBatch) -> list[np.ndarray]:
    blocks = []
    for k, beta in enumerate(batch.path_counts):
        start = batch.gram_offsets[k]
        end = batch.gram_offsets[k + 1]
        blocks.append(batch.gram_values[start:end].reshape(int(beta), int(beta)))
    return blocks


def reference_simplex(values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_values)
    indices = np.arange(1, len(values) + 1)
    rho = np.nonzero(sorted_values * indices > cumulative - 1)[0][-1]
    theta = (cumulative[rho] - 1) / (rho + 1)
    return np.maximum(values - theta, 0)


def packed_to_padded(values: np.ndarray, batch: PackedPathBatch) -> np.ndarray:
    padded = np.zeros((int(batch.path_counts.max()), batch.num_commodities), dtype=values.dtype)
    for k in range(batch.num_commodities):
        start, end = batch.commodity_path_offsets[k:k + 2]
        padded[:end - start, k] = values[start:end]
    return padded


def padded_to_packed(values: np.ndarray, batch: PackedPathBatch) -> np.ndarray:
    packed = np.empty(batch.num_paths, dtype=values.dtype)
    for k in range(batch.num_commodities):
        start, end = batch.commodity_path_offsets[k:k + 2]
        packed[start:end] = values[:end - start, k]
    return packed


class PackedPathKernelTests(unittest.TestCase):
    def _make_batch(self, dtype, scaled=False):
        capacities = np.asarray([2, 3, 5, 7, 11, 13], dtype=dtype)
        return PackedPathBatch.from_path_provider(
            make_provider(), capacities, scaled, kernel_threads=2
        ), capacities

    def test_gram_matches_dense_reference(self):
        provider = make_provider()
        for dtype in (np.float32, np.float64):
            for scaled in (False, True):
                with self.subTest(dtype=dtype, scaled=scaled):
                    batch, capacities = self._make_batch(dtype, scaled)
                    edge_scale = 1 / capacities if scaled else np.ones_like(capacities)
                    for k, actual in enumerate(unpack_blocks(batch)):
                        alpha = dense_alpha(provider, k, dtype)
                        expected = alpha.T @ ((edge_scale * edge_scale)[:, None] * alpha)
                        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_linear_flow_initialization_and_total(self):
        provider = make_provider()
        for dtype in (np.float32, np.float64):
            for scaled in (False, True):
                with self.subTest(dtype=dtype, scaled=scaled):
                    batch, capacities = self._make_batch(dtype, scaled)
                    demands = np.asarray([1.5, 2.0, 0.75], dtype=dtype)
                    bias = np.asarray([0.2, -0.1, 0.4, 0.7, -0.2, 0.3], dtype=dtype)
                    edge_scale = 1 / capacities if scaled else np.ones_like(capacities)

                    expected_linear = []
                    for k in range(batch.num_commodities):
                        alpha = dense_alpha(provider, k, dtype)
                        expected_linear.extend(demands[k] * (alpha.T @ (edge_scale * bias)))
                    np.testing.assert_allclose(
                        batch.linear_term(bias, demands),
                        np.asarray(expected_linear, dtype=dtype),
                        rtol=1e-6,
                        atol=1e-7,
                    )

                    splits = batch.initialize_splits(capacities)
                    for k in range(batch.num_commodities):
                        start, end = batch.commodity_path_offsets[k:k + 2]
                        bottlenecks = []
                        for path in range(start, end):
                            e0, e1 = batch.path_edge_offsets[path:path + 2]
                            bottlenecks.append(np.min(capacities[batch.path_edges[e0:e1]]))
                        expected = np.asarray(bottlenecks, dtype=dtype)
                        expected /= expected.sum()
                        np.testing.assert_allclose(splits[start:end], expected, rtol=1e-6)

                    expected_edge = np.zeros((batch.num_edges, batch.num_commodities), dtype=dtype)
                    for k in range(batch.num_commodities):
                        start, end = batch.commodity_path_offsets[k:k + 2]
                        alpha = dense_alpha(provider, k, dtype)
                        expected_edge[:, k] = edge_scale * demands[k] * (alpha @ splits[start:end])
                    actual_edge = batch.paths_to_edge(splits, demands)
                    np.testing.assert_allclose(actual_edge, expected_edge, rtol=1e-6, atol=1e-7)
                    np.testing.assert_allclose(
                        batch.paths_to_edge_mean(splits, demands),
                        expected_edge.mean(axis=1),
                        rtol=1e-6,
                        atol=1e-7,
                    )
                    self.assertAlmostEqual(
                        batch.total_flow(splits, demands),
                        float(demands.sum()),
                        places=5 if dtype == np.float32 else 12,
                    )

    def test_projected_step_matches_dense_reference(self):
        rng = np.random.default_rng(12345)
        for dtype in (np.float32, np.float64):
            batch, _ = self._make_batch(dtype, scaled=True)
            z = rng.normal(size=batch.num_paths).astype(dtype)
            y_old = rng.normal(size=batch.num_paths).astype(dtype)
            linear = rng.normal(size=batch.num_paths).astype(dtype)
            demands = rng.uniform(0.5, 2.0, size=batch.num_commodities).astype(dtype)
            steps = rng.uniform(0.01, 0.1, size=batch.num_commodities).astype(dtype)
            blocks = unpack_blocks(batch)

            for orthant in (False, True):
                with self.subTest(dtype=dtype, orthant=orthant):
                    actual = np.empty_like(z)
                    batch.projected_qp_step(
                        z, y_old, linear, demands, steps, orthant, actual
                    )
                    expected = np.empty_like(z)
                    for k, block in enumerate(blocks):
                        start, end = batch.commodity_path_offsets[k:k + 2]
                        trial = z[start:end] - steps[k] * (
                            demands[k] ** 2 * block @ (z[start:end] - y_old[start:end])
                            + linear[start:end]
                        )
                        if orthant:
                            trial = np.maximum(trial, 0)
                            expected[start:end] = (
                                trial if trial.sum() <= 1 else reference_simplex(trial)
                            )
                        else:
                            expected[start:end] = reference_simplex(trial)
                    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
                    for k in range(batch.num_commodities):
                        start, end = batch.commodity_path_offsets[k:k + 2]
                        self.assertTrue(np.all(actual[start:end] >= 0))
                        segment_sum = actual[start:end].sum()
                        if orthant:
                            self.assertLessEqual(float(segment_sum), 1.0 + 1e-6)
                        else:
                            self.assertAlmostEqual(float(segment_sum), 1.0, places=5)

    def test_edge_conversion_matches_legacy_numba(self):
        provider = make_provider()
        rows = NumbaList(provider.rows)
        cols = NumbaList(provider.cols)
        demands = np.asarray([1.0, 1.5, 2.0], dtype=np.float64)
        for scaled in (False, True):
            batch, capacities = self._make_batch(np.float64, scaled)
            values = batch.initialize_splits(capacities)
            padded = packed_to_padded(values, batch)
            conditional_capacity = capacities if scaled else None
            expected = path_based_to_edge_based_nnz(
                padded, rows, cols, batch.num_edges, demands, conditional_capacity
            )
            expected_mean = path_based_to_edge_based_mean_nnz(
                padded, rows, cols, batch.num_edges, demands, conditional_capacity
            )
            np.testing.assert_allclose(batch.paths_to_edge(values, demands), expected)
            np.testing.assert_allclose(
                batch.paths_to_edge_mean(values, demands), expected_mean
            )

    def test_capacity_scaled_representation_preserves_mlu(self):
        unscaled, capacities = self._make_batch(np.float64, scaled=False)
        scaled, _ = self._make_batch(np.float64, scaled=True)
        demands = np.asarray([1.0, 1.5, 2.0], dtype=np.float64)

        unscaled_splits = unscaled.initialize_splits(capacities)
        scaled_splits = scaled.initialize_splits(capacities)
        physical_assignment = unscaled.paths_to_edge(unscaled_splits, demands)
        normalized_assignment = scaled.paths_to_edge(scaled_splits, demands)

        np.testing.assert_allclose(unscaled_splits, scaled_splits)
        np.testing.assert_allclose(
            normalized_assignment * capacities[:, None],
            physical_assignment,
        )
        physical_mlu = np.max(physical_assignment.sum(axis=1) / capacities)
        normalized_mlu = np.max(normalized_assignment.sum(axis=1))
        self.assertAlmostEqual(physical_mlu, normalized_mlu)

    def test_coordinator_reports_physical_assignment_after_capacity_scaling(self):
        normalized_assignment = np.asarray(
            [[0.5, 0.0], [0.0, 0.8]], dtype=np.float64
        )
        capacities = np.asarray([2.0, 5.0], dtype=np.float64)
        coordinator = object.__new__(OnlineTECoordinator)
        coordinator._solver_params = PathBasedOnlineTEParameters(ScaleWithCapacity=True)
        coordinator._capacities = capacities
        coordinator.backend = Mock()
        coordinator.backend.get_X_ek.return_value = normalized_assignment

        coordinator._set_X_ek()

        np.testing.assert_allclose(
            coordinator.current_assignment,
            normalized_assignment * capacities[:, None],
        )

        graph = nx.DiGraph()
        graph.add_edge(0, 1, capacity=capacities[0])
        graph.add_edge(1, 0, capacity=capacities[1])
        physical_demands = np.asarray([[0.0, 1.0], [4.0, 0.0]])
        congestion_ratio, _ = check_capacity_constraint(
            coordinator.current_assignment, graph, feasibility_tolerance=1e-10
        )
        self.assertEqual(congestion_ratio, 0)
        self.assertEqual(
            check_flow_satisfaction(
                coordinator.current_assignment,
                graph,
                traffic_to_commodity(physical_demands),
                feasibility_tolerance=1e-10,
                edge_indexing=get_edge_indexing(graph),
            ),
            [],
        )

    def test_power_method_matches_current_recurrence(self):
        for dtype in (np.float32, np.float64):
            batch, _ = self._make_batch(dtype)
            demands = np.asarray([1.0, 1.5, 2.0], dtype=dtype)
            actual = batch.estimate_lipschitz(demands)
            expected = []
            for k, block in enumerate(unpack_blocks(batch)):
                matrix = demands[k] ** 2 * block
                vector = np.ones(block.shape[0], dtype=dtype)
                for _ in range(20):
                    result = matrix @ vector
                    vector = result / np.linalg.norm(vector)
                result = matrix @ vector
                expected.append(np.dot(result, vector) / np.dot(vector, vector))
            np.testing.assert_allclose(actual, np.asarray(expected), rtol=2e-6, atol=1e-7)

    def test_multi_iteration_pgd_matches_legacy_padded_solver(self):
        provider = make_provider()
        batch, _ = self._make_batch(np.float64, scaled=False)
        rng = np.random.default_rng(7)
        y = batch.initialize_splits(np.asarray([2, 3, 5, 7, 11, 13], dtype=np.float64))
        y_old = y + rng.normal(scale=0.01, size=y.shape)
        demands = np.asarray([1.0, 1.5, 2.0], dtype=np.float64)
        sharing_bias = rng.normal(size=batch.num_edges)
        steps = np.asarray([0.03, 0.02, 0.01], dtype=np.float64)
        padded_y = packed_to_padded(y, batch)
        padded_old = packed_to_padded(y_old, batch)
        rows = NumbaList(provider.rows)
        cols = NumbaList(provider.cols)

        legacy_mlu = do_path_based_nesterov_pgd(
            padded_y.copy(), padded_old.copy(), rows, cols, sharing_bias,
            provider.beta, demands, batch.num_edges, provider.shape[2], steps, 3, None,
        )
        packed_mlu = do_packed_path_based_nesterov_pgd(
            y.copy(), y_old.copy(), batch, batch.linear_term(sharing_bias, demands),
            demands, steps, 3,
        )
        np.testing.assert_allclose(
            packed_mlu, padded_to_packed(legacy_mlu, batch), rtol=1e-11, atol=1e-12
        )

        eta = 0.4
        legacy_maxflow = do_path_based_maxflow_pgd(
            padded_y.copy(), padded_old.copy(), rows, cols, sharing_bias,
            provider.beta, demands, batch.num_edges, provider.shape[2], steps, 3, eta, None,
        )
        packed_maxflow = do_packed_path_based_maxflow_pgd(
            y.copy(), y_old.copy(), batch,
            batch.linear_term(sharing_bias, demands, maxflow_shift=1 / eta),
            demands, steps, 3,
        )
        np.testing.assert_allclose(
            packed_maxflow,
            padded_to_packed(legacy_maxflow, batch),
            rtol=1e-11,
            atol=1e-12,
        )

    def test_path_provider_pickle_remains_compatible(self):
        provider = make_provider()
        loaded = pickle.loads(pickle.dumps(provider))
        batch = PackedPathBatch.from_path_provider(
            loaded, np.ones(loaded.shape[1], dtype=np.float64), False, 1
        )
        np.testing.assert_array_equal(batch.path_counts, provider.beta)

    def test_half_precision_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "float32 and float64"):
            PackedPathBatch.from_path_provider(
                make_provider(), np.ones(6, dtype=np.float16), False, 1
            )

    def test_dense_solver_integration_for_both_objectives(self):
        batch, capacities = self._make_batch(np.float64)
        demands = np.asarray([1.0, 1.5, 2.0], dtype=np.float64)
        bias = np.asarray([0.1, -0.2, 0.3, 0.05, -0.1, 0.2], dtype=np.float64)
        for objective in (TEObjective.MLU, TEObjective.MAX_FLOW):
            with self.subTest(objective=objective):
                solver = DenseSolver(
                    demands=demands,
                    path_batch=batch,
                    pgd_step=0.01,
                    pgd_iters=3,
                    eta=0.4,
                    adjust_step_size=False,
                    capacities=capacities,
                    objective=objective,
                )
                initial = solver.X_ek
                self.assertEqual(initial.shape, (batch.num_edges, batch.num_commodities))
                mean = solver.update(bias)
                self.assertEqual(mean.shape, (batch.num_edges,))
                np.testing.assert_allclose(mean, solver.X_ek.mean(axis=1), rtol=1e-12)
                self.assertGreaterEqual(solver.total_flow, 0)

    def test_kernel_thread_parameter_is_serializable(self):
        params = PathBasedOnlineTEParameters(KernelThreads=3)
        self.assertEqual(params.KernelThreads, 3)
        message = PathParamsMessage(**params.child_fields)
        self.assertEqual(MessageToDict(message)["KernelThreads"], 3)
        with self.assertRaisesRegex(ValueError, "positive"):
            PathBasedOnlineTEParameters(KernelThreads=0)
        with self.assertRaisesRegex(ValueError, "single and double"):
            PathBasedOnlineTEParameters(Precision="half")


if __name__ == "__main__":
    unittest.main()
