import pickle
import unittest
from unittest.mock import Mock, patch

import networkx as nx
import numpy as np
from numba.typed import List as NumbaList
from google.protobuf.json_format import MessageToDict

from protos.solver_params.solver_params_pb2 import PathBasedOnlineTEParameters as PathParamsMessage
from te.path_providers.packed_paths import PackedPathBatch
from te.algorithms.formulations.path_based.distributed.simple_worker import (
    DenseSolver as SimpleDenseSolver,
)
from te.algorithms.formulations.path_based.distributed.coordinator import OnlineTECoordinator
from te.algorithms.formulations.path_based.distributed.single_loop_coordinator import (
    SimpleOnlineTECoordinator as SingleLoopOnlineTECoordinator,
)
from te.algorithms.formulations.path_based.distributed.solver_params import PathBasedOnlineTEParameters, PathBasedSimplifiedOnlineTEParameters
from te.algorithms.formulations.path_based.distributed.worker import (
    DenseSolver,
    OnlineTEWorkerNode,
)
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

                    splits = batch.initialize_splits(capacities, TEObjective.MLU)
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

    def test_assignment_delta_norm_squared_matches_dense_edge_difference(self):
        provider = make_provider()
        rng = np.random.default_rng(20260921)
        for dtype in (np.float32, np.float64):
            for scaled in (False, True):
                with self.subTest(dtype=dtype, scaled=scaled):
                    batch, _ = self._make_batch(dtype, scaled)
                    current = rng.normal(size=batch.num_paths).astype(dtype)
                    previous = rng.normal(size=batch.num_paths).astype(dtype)
                    demands = rng.uniform(0.1, 3.0, size=batch.num_commodities).astype(dtype)

                    current_edge = batch.paths_to_edge(current, demands)
                    previous_edge = batch.paths_to_edge(previous, demands)
                    expected = float(np.sum(
                        (current_edge.astype(np.float64) - previous_edge.astype(np.float64)) ** 2
                    ))
                    actual = batch.assignment_delta_norm_squared(
                        current, previous, demands
                    )
                    tolerance = 2e-6 if dtype == np.float32 else 1e-12
                    self.assertAlmostEqual(actual, expected, delta=tolerance * max(1.0, expected))
                    self.assertEqual(
                        batch.assignment_delta_norm_squared(current, current, demands),
                        0.0,
                    )

                    one_thread = PackedPathBatch.from_path_provider(
                        provider,
                        np.asarray([2, 3, 5, 7, 11, 13], dtype=dtype),
                        scaled,
                        kernel_threads=1,
                    )
                    self.assertEqual(
                        actual,
                        one_thread.assignment_delta_norm_squared(current, previous, demands),
                    )

    def test_dual_objective_term_matches_path_enumeration(self):
        provider = make_provider()
        for dtype in (np.float32, np.float64):
            for scaled in (False, True):
                with self.subTest(dtype=dtype, scaled=scaled):
                    batch, capacities = self._make_batch(dtype, scaled)
                    demands = np.asarray([1.5, 2.0, 0.75], dtype=dtype)
                    edge_duals = np.asarray(
                        [0.3, -0.2, 0.5, -0.7, 0.1, 0.4], dtype=dtype
                    )
                    edge_scale = 1 / capacities if scaled else np.ones_like(capacities)
                    expected = 0.0
                    for k in range(batch.num_commodities):
                        alpha = dense_alpha(provider, k, dtype)
                        path_values = alpha.T @ (edge_scale * edge_duals)
                        expected += float(demands[k]) * float(np.min(path_values))

                    actual = batch.dual_objective_term(edge_duals, demands, TEObjective.MLU, 3)
                    tolerance = 2e-6 if dtype == np.float32 else 1e-12
                    self.assertAlmostEqual(actual, expected, delta=tolerance * max(1.0, abs(expected)))
                    self.assertEqual(
                        batch.dual_objective_term(np.zeros_like(edge_duals), demands, TEObjective.MLU, 3),
                        0.0,
                    )

                    one_thread = PackedPathBatch.from_path_provider(
                        provider, capacities, scaled, kernel_threads=1
                    )
                    self.assertEqual(
                        actual,
                        one_thread.dual_objective_term(edge_duals, demands, TEObjective.MLU, 3),
                    )

    def test_diagnostic_kernel_inputs_are_validated(self):
        batch, _ = self._make_batch(np.float32)
        values = np.zeros(batch.num_paths, dtype=np.float32)
        demands = np.ones(batch.num_commodities, dtype=np.float32)
        edge_duals = np.zeros(batch.num_edges, dtype=np.float32)

        with self.assertRaisesRegex(TypeError, "current must have dtype"):
            batch.assignment_delta_norm_squared(
                values.astype(np.float64), values, demands
            )
        with self.assertRaisesRegex(ValueError, "previous must have shape"):
            batch.assignment_delta_norm_squared(values, values[:-1], demands)
        with self.assertRaisesRegex(ValueError, "edge_duals must be native-endian and C-contiguous"):
            noncontiguous = np.zeros(batch.num_edges * 2, dtype=np.float32)[::2]
            batch.dual_objective_term(noncontiguous, demands, TEObjective.MLU, 3)
        with self.assertRaisesRegex(ValueError, "demands must have shape"):
            batch.dual_objective_term(edge_duals, demands[:-1], TEObjective.MLU, 3)

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
                    relative_gaps = batch.projected_qp_step(
                        z, y_old, linear, demands, steps, orthant,
                        np.arange(batch.num_commodities, dtype=np.int64), actual,
                    )
                    self.assertEqual(relative_gaps.dtype, np.float64)
                    self.assertEqual(relative_gaps.shape, (batch.num_commodities,))
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
                        candidate = actual[start:end].astype(np.float64)
                        previous = y_old[start:end].astype(np.float64)
                        q_gradient = (
                            float(demands[k]) ** 2
                            * blocks[k].astype(np.float64)
                            @ (candidate - previous)
                        )
                        gradient = q_gradient + linear[start:end].astype(np.float64)
                        primal = (
                            0.5 * float((candidate - previous) @ q_gradient)
                            + float(linear[start:end].astype(np.float64) @ candidate)
                        )
                        linear_minimum = float(np.min(gradient))
                        if orthant:
                            linear_minimum = min(0.0, linear_minimum)
                        gap = max(0.0, float(gradient @ candidate) - linear_minimum)
                        lower_bound = primal - gap
                        expected_relative_gap = gap / max(
                            1.0, abs(primal), abs(lower_bound)
                        )
                        tolerance = 3e-6 if dtype == np.float32 else 1e-12
                        self.assertAlmostEqual(
                            relative_gaps[k],
                            expected_relative_gap,
                            delta=tolerance,
                        )
                        self.assertTrue(np.all(actual[start:end] >= 0))
                        segment_sum = actual[start:end].sum()
                        if orthant:
                            self.assertLessEqual(float(segment_sum), 1.0 + 1e-6)
                        else:
                            self.assertAlmostEqual(float(segment_sum), 1.0, places=5)

    def test_projected_step_wolfe_gap_is_exact_at_known_linear_optima(self):
        for dtype in (np.float32, np.float64):
            batch, _ = self._make_batch(dtype)
            demands = np.zeros(batch.num_commodities, dtype=dtype)
            steps = np.ones(batch.num_commodities, dtype=dtype)
            y_old = np.zeros(batch.num_paths, dtype=dtype)

            simplex_z = np.zeros(batch.num_paths, dtype=dtype)
            simplex_linear = np.empty(batch.num_paths, dtype=dtype)
            orthant_z = np.zeros(batch.num_paths, dtype=dtype)
            orthant_linear = np.ones(batch.num_paths, dtype=dtype)
            for k in range(batch.num_commodities):
                start, end = batch.commodity_path_offsets[k:k + 2]
                simplex_z[start] = 1
                simplex_linear[start:end] = np.arange(end - start, dtype=dtype)

            output = np.empty_like(simplex_z)
            simplex_gaps = batch.projected_qp_step(
                simplex_z, y_old, simplex_linear, demands, steps, False,
                np.arange(batch.num_commodities, dtype=np.int64), output,
            )
            np.testing.assert_allclose(simplex_gaps, 0.0, atol=1e-12)

            orthant_output = np.empty_like(orthant_z)
            orthant_gaps = batch.projected_qp_step(
                orthant_z, y_old, orthant_linear, demands, steps, True,
                np.arange(batch.num_commodities, dtype=np.int64), orthant_output,
            )
            np.testing.assert_allclose(orthant_output, 0.0, atol=1e-12)
            np.testing.assert_allclose(orthant_gaps, 0.0, atol=1e-12)

            nonoptimal_z = np.empty_like(simplex_z)
            for k in range(batch.num_commodities):
                start, end = batch.commodity_path_offsets[k:k + 2]
                nonoptimal_z[start:end] = 1 / (end - start)
            small_steps = np.full(batch.num_commodities, 1e-3, dtype=dtype)
            nonoptimal_output = np.empty_like(nonoptimal_z)
            nonoptimal_gaps = batch.projected_qp_step(
                nonoptimal_z,
                y_old,
                simplex_linear,
                demands,
                small_steps,
                False,
                np.arange(batch.num_commodities, dtype=np.int64),
                nonoptimal_output,
            )
            self.assertTrue(np.any(nonoptimal_gaps > 0))

    def test_packed_pgd_stops_on_gap_or_iteration_cap(self):
        y = np.asarray([0.5, 0.5], dtype=np.float64)
        demands = np.asarray([1.0], dtype=np.float64)
        steps = np.asarray([0.1], dtype=np.float64)
        linear = np.zeros_like(y)
        batch = Mock()
        batch.num_commodities = 1

        def converged_step(**kwargs):
            np.copyto(kwargs["output"], kwargs["z"])
            return np.zeros(1, dtype=np.float64)

        batch.projected_qp_step.side_effect = converged_step
        result = do_packed_path_based_nesterov_pgd(
            y, y.copy(), batch, linear, demands, steps, 7,
            optimality_tolerance=1e-3,
        )
        np.testing.assert_allclose(result, y)
        self.assertEqual(batch.projected_qp_step.call_count, 1)

        def unconverged_step(**kwargs):
            np.copyto(kwargs["output"], kwargs["z"])
            return np.ones(1, dtype=np.float64)

        batch.reset_mock()
        batch.projected_qp_step.side_effect = unconverged_step
        do_packed_path_based_maxflow_pgd(
            y, y.copy(), batch, linear, demands, steps, 7,
            optimality_tolerance=1e-3,
        )
        self.assertEqual(batch.projected_qp_step.call_count, 7)

    def test_edge_conversion_matches_legacy_numba(self):
        provider = make_provider()
        rows = NumbaList(provider.rows)
        cols = NumbaList(provider.cols)
        demands = np.asarray([1.0, 1.5, 2.0], dtype=np.float64)
        for scaled in (False, True):
            batch, capacities = self._make_batch(np.float64, scaled)
            values = batch.initialize_splits(capacities, TEObjective.MLU)
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

        unscaled_splits = unscaled.initialize_splits(capacities, TEObjective.MLU)
        scaled_splits = scaled.initialize_splits(capacities, TEObjective.MLU)
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

    def test_capacity_scaling_rejects_invalid_capacities(self):
        valid = np.asarray([2, 3, 5, 7, 11, 13], dtype=np.float64)
        for invalid in (0.0, -1.0, np.inf, np.nan):
            with self.subTest(invalid=invalid):
                capacities = valid.copy()
                capacities[2] = invalid
                with self.assertRaisesRegex(
                    ValueError, "finite, strictly positive"
                ):
                    PackedPathBatch.from_path_provider(
                        make_provider(), capacities, True, kernel_threads=1
                    )

        # The validation belongs to the reciprocal-capacity coordinate
        # transform and must not change the legacy unscaled path.
        capacities = valid.copy()
        capacities[2] = 0
        batch = PackedPathBatch.from_path_provider(
            make_provider(), capacities, False, kernel_threads=1
        )
        np.testing.assert_allclose(batch.edge_scale, 1)

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
            (1.0, []),
        )

    def test_single_loop_coordinator_reports_assignments_in_physical_units(self):
        normalized_assignment = np.asarray(
            [[0.5, 0.0], [0.0, 0.8]], dtype=np.float64
        )
        capacities = np.asarray([2.0, 5.0], dtype=np.float64)

        for scaled in (False, True):
            with self.subTest(scaled=scaled):
                coordinator = object.__new__(SingleLoopOnlineTECoordinator)
                coordinator._solver_params = PathBasedSimplifiedOnlineTEParameters()
                coordinator._capacities = capacities
                coordinator.backend = Mock()
                coordinator.backend.get_X_ek.return_value = normalized_assignment

                coordinator._set_X_ek()

                # The simplified solver always uses reciprocal-capacity coordinates.
                expected = normalized_assignment * capacities[:, None]
                np.testing.assert_allclose(coordinator.current_assignment, expected)
                np.testing.assert_allclose(
                    coordinator.backend.get_X_ek.return_value,
                    normalized_assignment,
                )

    def test_single_loop_coordinator_computes_mlu_in_internal_coordinates(self):
        capacities = np.asarray([2.0, 4.0], dtype=np.float64)
        graph = nx.DiGraph()
        graph.add_nodes_from((0, 1))

        for scaled, mean in (
            (False, np.asarray([0.4, 1.6])),
            (True, np.asarray([0.2, 0.4])),
        ):
            with self.subTest(scaled=scaled):
                coordinator = object.__new__(SingleLoopOnlineTECoordinator)
                coordinator._solver_params = PathBasedSimplifiedOnlineTEParameters()
                coordinator._capacities = capacities
                coordinator._graph = graph
                coordinator._problem_description = Mock()
                coordinator._problem_description.objective = TEObjective.MLU
                coordinator._mlu_solver = Mock()
                coordinator._mlu_solver.current_Z = np.zeros_like(capacities)
                coordinator._sharing_wrapper = Mock()
                coordinator._sharing_wrapper.X_mean = mean if scaled else mean / capacities

                coordinator._do_coordinator_update()

                self.assertAlmostEqual(coordinator.current_objective, 0.8)

        coordinator._problem_description.objective = TEObjective.MAX_FLOW
        coordinator._total_flow = 3.25
        coordinator._do_coordinator_update()
        self.assertEqual(coordinator.current_objective, -3.25 / 2)

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

    def test_maxflow_edge_step_and_gap_match_dense_objective(self):
        rng = np.random.default_rng(20261009)
        for dtype in (np.float32, np.float64):
            for scaled in (False, True):
                with self.subTest(dtype=dtype, scaled=scaled):
                    batch, _ = self._make_batch(dtype, scaled)
                    stored_gram = batch.gram_values.copy()
                    z = rng.normal(size=batch.num_paths).astype(dtype)
                    old = rng.uniform(0, 0.2, size=batch.num_paths).astype(dtype)
                    linear = rng.normal(size=batch.num_paths).astype(dtype)
                    demands = np.asarray([0.7, 1.5, 2.0], dtype=dtype)
                    steps = np.asarray([0.04, 0.03, 0.02], dtype=dtype)
                    output = np.empty_like(z)
                    gaps = batch.projected_qp_step(
                        z, old, linear, demands, steps, True,
                        np.arange(batch.num_commodities, dtype=np.int64), output,
                    )
                    for k, gram in enumerate(unpack_blocks(batch)):
                        start, end = batch.commodity_path_offsets[k:k + 2]
                        hessian = float(demands[k]) ** 2 * gram.astype(np.float64)
                        delta = z[start:end].astype(np.float64) - old[start:end]
                        trial = z[start:end] - steps[k] * (hessian @ delta + linear[start:end])
                        trial = np.maximum(trial, 0)
                        expected = trial if trial.sum() <= 1 else reference_simplex(trial)
                        np.testing.assert_allclose(output[start:end], expected, rtol=3e-6, atol=3e-7)

                        candidate = output[start:end].astype(np.float64)
                        delta = candidate - old[start:end]
                        gradient = hessian @ delta + linear[start:end]
                        objective = 0.5 * delta @ hessian @ delta + linear[start:end] @ candidate
                        gap = max(0.0, gradient @ candidate - min(0.0, gradient.min()))
                        expected_gap = gap / max(1.0, abs(objective), abs(objective - gap))
                        self.assertAlmostEqual(gaps[k], expected_gap, delta=3e-12)
                    # Edge-load residuals and dual reporting must keep the edge-only Gram.
                    np.testing.assert_array_equal(batch.gram_values, stored_gram)

    def test_maxflow_lipschitz_matches_edge_hessian_eigenvalues(self):
        for dtype in (np.float32, np.float64):
            for scaled in (False, True):
                with self.subTest(dtype=dtype, scaled=scaled):
                    batch, _ = self._make_batch(dtype, scaled)
                    demands = np.asarray([0.7, 1.5, 2.0], dtype=dtype)
                    actual = batch.estimate_lipschitz(demands)
                    expected = [
                        float(demands[k]) ** 2 * np.linalg.eigvalsh(gram.astype(np.float64))[-1]
                        for k, gram in enumerate(unpack_blocks(batch))
                    ]
                    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
                    np.testing.assert_array_equal(batch.estimate_lipschitz(np.zeros_like(demands)), 0)

    def test_maxflow_edge_pgd_reaches_known_interior_optimum(self):
        for dtype in (np.float32, np.float64):
            for scaled in (False, True):
                with self.subTest(dtype=dtype, scaled=scaled):
                    batch, _ = self._make_batch(dtype, scaled)
                    demands = np.asarray([0.7, 1.5, 2.0], dtype=dtype)
                    old = np.empty(batch.num_paths, dtype=dtype)
                    optimum = np.empty_like(old)
                    linear = np.empty_like(old)
                    for k, gram in enumerate(unpack_blocks(batch)):
                        start, end = batch.commodity_path_offsets[k:k + 2]
                        old[start:end] = (0.1 + 0.1 * k) / (end - start)
                        optimum[start:end] = (0.2 + 0.3 * k) / (end - start)
                        linear[start:end] = -demands[k] ** 2 * gram @ (optimum[start:end] - old[start:end])
                    steps = 1 / batch.estimate_lipschitz(demands)
                    tolerance = 1e-6 if dtype == np.float32 else 1e-10
                    result = do_packed_path_based_maxflow_pgd(
                        old.copy(), old, batch, linear, demands, steps, 500,
                        optimality_tolerance=tolerance,
                    )
                    np.testing.assert_allclose(result, optimum, rtol=2e-4, atol=2e-5)
                    output = np.empty_like(result)
                    gaps = batch.projected_qp_step(
                        optimum, old, linear, demands, steps, True,
                        np.arange(batch.num_commodities, dtype=np.int64), output,
                    )
                    np.testing.assert_allclose(output, optimum, rtol=2e-6, atol=2e-7)
                    np.testing.assert_allclose(gaps, 0.0, atol=tolerance)

    def test_maxflow_edge_step_respects_active_commodities(self):
        batch, _ = self._make_batch(np.float64, scaled=True)
        demands = np.ones(batch.num_commodities)
        old = np.zeros(batch.num_paths)
        linear = -np.ones(batch.num_paths)
        output = np.full(batch.num_paths, -123.0)
        batch.projected_qp_step(
            old, old, linear, demands, np.full(batch.num_commodities, 0.1), True,
            np.asarray([1], dtype=np.int64), output,
        )
        np.testing.assert_array_equal(output, [-123.0, 0.1, 0.1, -123.0, -123.0, -123.0])

    def test_workers_use_edge_curvature_on_demand_updates(self):
        batch, capacities = self._make_batch(np.float64, scaled=True)
        demands = np.asarray([1.0, 1.5, 2.0])
        for objective in (TEObjective.MLU, TEObjective.MAX_FLOW):
            for worker_kind in (DenseSolver, SimpleDenseSolver):
                with self.subTest(objective=objective, worker=worker_kind.__name__):
                    kwargs = dict(
                        demands=demands, path_batch=batch, pgd_step=0.5,
                        pgd_iters=1, capacities=capacities, objective=objective,
                        optimality_tolerance=1e-10,
                    )
                    if worker_kind is DenseSolver:
                        kwargs.update(eta=0.4, adjust_step_size=True)
                    else:
                        kwargs.update(rho=0.4, total_numbar_of_commodities=3)
                    solver = worker_kind(**kwargs)
                    for next_demands in (demands, demands * 2):
                        solver.set_demands(next_demands)
                        expected = np.asarray([
                            float(next_demands[k]) ** 2 * np.linalg.eigvalsh(
                                gram
                            )[-1]
                            for k, gram in enumerate(unpack_blocks(batch))
                        ])
                        np.testing.assert_allclose(solver._pgd_steps, 0.5 / expected, rtol=1e-7)
                    if worker_kind is DenseSolver:
                        mean = solver.update(np.zeros(batch.num_edges))
                    else:
                        mean = solver.update(np.zeros(batch.num_edges), 0.4)
                    self.assertTrue(np.all(np.isfinite(mean)))

    def test_multi_iteration_pgd_matches_legacy_padded_solver(self):
        provider = make_provider()
        batch, _ = self._make_batch(np.float64, scaled=False)
        rng = np.random.default_rng(7)
        y = batch.initialize_splits(
            np.asarray([2, 3, 5, 7, 11, 13], dtype=np.float64), TEObjective.MLU
        )
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
            optimality_tolerance=1e-14,
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
            optimality_tolerance=1e-14,
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
        demands = np.asarray([1.0, 1.5, 2.0], dtype=np.float64)
        bias = np.asarray([0.1, -0.2, 0.3, 0.05, -0.1, 0.2], dtype=np.float64)
        for scaled in (False, True):
            batch, capacities = self._make_batch(np.float64, scaled=scaled)
            for objective in (TEObjective.MLU, TEObjective.MAX_FLOW):
                with self.subTest(scaled=scaled, objective=objective):
                    solver = DenseSolver(
                        demands=demands,
                        path_batch=batch,
                        pgd_step=0.01,
                        pgd_iters=3,
                        eta=0.4,
                        adjust_step_size=False,
                        capacities=capacities,
                        objective=objective,
                        optimality_tolerance=1e-10,
                    )
                    initial = solver.X_ek
                    self.assertEqual(initial.shape, (batch.num_edges, batch.num_commodities))
                    expected_initial_flow = float(demands.sum()) if objective == TEObjective.MLU else 0.0
                    self.assertAlmostEqual(solver.total_flow, expected_initial_flow)
                    mean = solver.update(bias)
                    self.assertEqual(mean.shape, (batch.num_edges,))
                    np.testing.assert_allclose(mean, solver.X_ek.mean(axis=1), rtol=1e-12)
                    self.assertGreaterEqual(solver.total_flow, 0)

    def test_worker_forwards_capacity_scaling_to_packed_paths(self):
        graph = nx.DiGraph()
        for edge, capacity in zip(
            ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)),
            (2.0, 3.0, 5.0, 7.0, 11.0, 13.0),
        ):
            graph.add_edge(*edge, capacity=capacity)

        for scaled in (False, True):
            with self.subTest(scaled=scaled):
                worker = object.__new__(OnlineTEWorkerNode)
                worker._solver_params = PathBasedOnlineTEParameters(
                    ScaleWithCapacity=scaled
                )
                worker._objective = TEObjective.MLU
                worker._number_of_workers = 1
                worker._node_params = Mock()
                worker._node_params.RPCParams_.PeerIndex = 0
                worker._create_local_path_object = Mock(
                    side_effect=lambda: setattr(worker, "_path_object", make_provider())
                )
                packed = Mock()

                with patch.object(
                    PackedPathBatch, "from_path_provider", return_value=packed
                ) as build_batch, patch(
                    "te.algorithms.formulations.path_based.distributed.worker.DenseSolver"
                ) as dense_solver:
                    worker.set_topology(graph)

                build_batch.assert_called_once()
                call = build_batch.call_args.kwargs
                self.assertEqual(call["scale_with_capacity"], scaled)
                np.testing.assert_allclose(
                    call["capacities"],
                    np.asarray([2.0, 5.0, 3.0, 11.0, 7.0, 13.0]),
                )
                dense_solver.assert_called_once()

    def test_kernel_thread_parameter_is_serializable(self):
        params = PathBasedOnlineTEParameters(
            KernelThreads=3,
            SwitchOptimalityTolerance=1e-3,
        )
        self.assertEqual(params.KernelThreads, 3)
        message = PathParamsMessage(**params.child_fields)
        self.assertEqual(MessageToDict(message)["KernelThreads"], 3)
        self.assertAlmostEqual(
            MessageToDict(message)["SwitchOptimalityTolerance"],
            1e-3,
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            PathBasedOnlineTEParameters(KernelThreads=0)
        with self.assertRaisesRegex(ValueError, "SwitchIterations"):
            PathBasedOnlineTEParameters(SwitchIterations=0)
        for invalid in (0.0, 1.0, np.nan):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "SwitchOptimalityTolerance"
            ):
                PathBasedOnlineTEParameters(SwitchOptimalityTolerance=invalid)
        with self.assertRaisesRegex(ValueError, "single and double"):
            PathBasedOnlineTEParameters(Precision="half")


if __name__ == "__main__":
    unittest.main()
