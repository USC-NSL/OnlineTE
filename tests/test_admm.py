import unittest

import numpy as np

from te.algorithms.sub_algorithms.admm import (
    ADMMMode,
    ADMMWrapper,
    AdaptiveStepSizeParams,
    DistributedSharingWrapper,
    NesterovAccelerationParams,
    SimpleStepSizeAdapter,
)


def constant_half_steps():
    while True:
        yield 0.5


class StepSizeAdapterTests(unittest.TestCase):
    def test_requires_consecutive_strikes_in_the_same_direction(self):
        params = AdaptiveStepSizeParams(2, 2, 2)
        adapter = SimpleStepSizeAdapter(4.0, params)
        dual = np.array([8.0])

        self.assertIsNone(adapter.adapt_step_size(5.0, 1.0, dual))
        self.assertIsNone(adapter.adapt_step_size(1.0, 5.0, dual))
        self.assertEqual(adapter._rho, 4.0)
        np.testing.assert_allclose(dual, [8.0])

        self.assertEqual(adapter.adapt_step_size(1.0, 5.0, dual), 2.0)
        np.testing.assert_allclose(dual, [16.0])

    def test_balanced_round_resets_strikes(self):
        params = AdaptiveStepSizeParams(2, 2, 2)
        adapter = SimpleStepSizeAdapter(1.0, params)
        dual = np.array([2.0])

        self.assertIsNone(adapter.adapt_step_size(3.0, 1.0, dual))
        self.assertIsNone(adapter.adapt_step_size(1.0, 1.0, dual))
        self.assertIsNone(adapter.adapt_step_size(3.0, 1.0, dual))
        self.assertEqual(adapter.adapt_step_size(3.0, 1.0, dual), 2.0)
        np.testing.assert_allclose(dual, [1.0])


class ADMMWrapperTests(unittest.TestCase):
    def make_wrapper(self, A, B, C, **kwargs):
        wrapper = ADMMWrapper(**kwargs)
        wrapper.set_matrices(
            np.asarray(A, dtype=float),
            np.asarray(B, dtype=float),
            np.asarray(C, dtype=float),
        )
        return wrapper

    def test_mode_selection_and_parameter_validation(self):
        self.assertEqual(ADMMWrapper(1.0).mode, ADMMMode.VANILLA)
        self.assertEqual(
            ADMMWrapper(1.0, acceleration=NesterovAccelerationParams()).mode,
            ADMMMode.ACCELERATED,
        )
        self.assertEqual(
            ADMMWrapper(
                1.0,
                acceleration=NesterovAccelerationParams(restart_threshold=0.5),
            ).mode,
            ADMMMode.ACCELERATED_RESTART,
        )
        self.assertEqual(ADMMWrapper(1.0, alpha=1.5).mode, ADMMMode.OVER_RELAXED)

        with self.assertRaises(ValueError):
            ADMMWrapper(0.0)
        with self.assertRaises(ValueError):
            NesterovAccelerationParams(restart_threshold=1.0)
        with self.assertRaises(ValueError):
            ADMMWrapper(1.0, alpha=1.5, acceleration=NesterovAccelerationParams())

    def test_vanilla_vector_round_matches_direct_formulas(self):
        A = np.diag([2.0, 3.0])
        B = np.diag([-1.0, -2.0])
        C = np.array([1.0, -1.0])
        X0 = np.array([1.0, 2.0])
        Z0 = np.array([3.0, 4.0])
        X1 = np.array([2.0, -1.0])
        Z1 = np.array([1.0, 5.0])
        rho = 2.0
        wrapper = self.make_wrapper(A, B, C, rho=rho)
        wrapper.initialize(X0, Z0)

        np.testing.assert_allclose(wrapper.get_X_step_bias(), B @ Z0 - C)
        wrapper.record_X_update(X1)
        np.testing.assert_allclose(wrapper.get_Z_step_bias(), A @ X1 - C)
        wrapper.record_Z_update(Z1)

        residual = A @ X1 + B @ Z1 - C
        wrapper.update_dual_var()
        np.testing.assert_allclose(wrapper.dual_var, residual)
        wrapper.finalize_round()

        self.assertAlmostEqual(wrapper.primal_infeasibility, np.linalg.norm(residual))
        expected_dual = rho * np.linalg.norm(A.T @ B @ (Z1 - Z0))
        self.assertGreater(expected_dual, 0)
        self.assertAlmostEqual(wrapper.dual_infeasibility, expected_dual)

    def test_batched_matrix_round_matches_direct_formulas(self):
        A = np.array([[1.0, 2.0], [0.0, 1.0]])
        B = np.array([[-1.0, 0.0], [1.0, -1.0]])
        C = np.array([[1.0, 2.0], [3.0, 4.0]])
        X0 = np.array([[1.0, 0.0], [0.0, 1.0]])
        Z0 = np.array([[2.0, 1.0], [1.0, 3.0]])
        X1 = np.array([[3.0, 1.0], [2.0, -1.0]])
        Z1 = np.array([[1.0, 4.0], [2.0, 0.0]])
        wrapper = self.make_wrapper(A, B, C, rho=1.5)
        wrapper.initialize(X0, Z0)

        wrapper.record_X_update(X1)
        np.testing.assert_allclose(wrapper.get_Z_step_bias(), A @ X1 - C)
        wrapper.record_Z_update(Z1)
        wrapper.update_dual_var()
        wrapper.finalize_round()

        residual = A @ X1 + B @ Z1 - C
        np.testing.assert_allclose(wrapper.dual_var, residual)
        self.assertAlmostEqual(wrapper.primal_infeasibility, np.linalg.norm(residual))
        self.assertAlmostEqual(
            wrapper.dual_infeasibility,
            1.5 * np.linalg.norm(A.T @ B @ (Z1 - Z0)),
        )

    def test_distinct_X_and_Z_shapes_require_explicit_Z_start(self):
        wrapper = self.make_wrapper(
            np.array([[1.0], [2.0]]),
            np.array([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]]),
            np.zeros(2),
            rho=1.0,
        )
        with self.assertRaises(ValueError):
            wrapper.initialize(np.array([1.0]))

        Z0 = np.array([1.0, 2.0, 3.0])
        Z1 = np.array([3.0, 2.0, 1.0])
        wrapper.initialize(np.array([1.0]), Z0)
        wrapper.record_X_update(np.array([2.0]))
        wrapper.record_Z_update(Z1)
        wrapper.update_dual_var()
        wrapper.finalize_round()
        self.assertGreater(wrapper.dual_infeasibility, 0)

    def test_over_relaxation_matches_direct_formulas(self):
        A = np.diag([2.0, 3.0])
        B = -np.eye(2)
        C = np.array([1.0, -2.0])
        X0 = np.array([0.0, 0.0])
        Z0 = np.array([1.0, 2.0])
        X1 = np.array([2.0, -1.0])
        Z1 = np.array([3.0, 4.0])
        alpha = 1.5
        wrapper = self.make_wrapper(A, B, C, rho=1.0, alpha=alpha)
        wrapper.initialize(X0, Z0)
        wrapper.record_X_update(X1)

        relaxed = alpha * (A @ X1) + (alpha - 1) * (B @ Z0 - C)
        np.testing.assert_allclose(wrapper.get_Z_step_bias(), relaxed - C)
        wrapper.record_Z_update(Z1)
        wrapper.update_dual_var()
        np.testing.assert_allclose(wrapper.dual_var, relaxed + B @ Z1 - C)

    def test_acceleration_advances_extrapolated_states(self):
        acceleration = NesterovAccelerationParams(step_sizes=constant_half_steps)
        wrapper = self.make_wrapper(
            np.eye(2), -np.eye(2), np.zeros(2), rho=1.0, acceleration=acceleration
        )
        Z0 = np.array([1.0, 2.0])
        wrapper.initialize(np.zeros(2), Z0)

        X1 = np.array([2.0, 3.0])
        Z1 = np.array([1.5, 2.5])
        wrapper.record_X_update(X1)
        wrapper.record_Z_update(Z1)
        wrapper.update_dual_var()
        wrapper.finalize_round()

        dual1 = X1 - Z1
        Z_hat1 = Z1 + 0.5 * (Z1 - Z0)
        dual_hat1 = dual1 + 0.5 * dual1
        np.testing.assert_allclose(wrapper.get_X_step_bias(), -Z_hat1 + dual_hat1)

        X2 = np.array([3.0, 4.0])
        Z2 = np.array([2.0, 3.0])
        wrapper.record_X_update(X2)
        np.testing.assert_allclose(wrapper.get_Z_step_bias(), X2 + dual_hat1)
        wrapper.record_Z_update(Z2)
        wrapper.update_dual_var()
        expected_dual2 = dual_hat1 + X2 - Z2
        np.testing.assert_allclose(wrapper.dual_var, expected_dual2)
        wrapper.finalize_round()
        self.assertAlmostEqual(wrapper.dual_infeasibility, np.linalg.norm(Z2 - Z_hat1))

        Z_hat2 = Z2 + 0.5 * (Z2 - Z1)
        dual_hat2 = expected_dual2 + 0.5 * (expected_dual2 - dual1)
        np.testing.assert_allclose(wrapper.get_X_step_bias(), -Z_hat2 + dual_hat2)

    def test_restart_uses_squared_merit_and_resets_momentum(self):
        acceleration = NesterovAccelerationParams(
            step_sizes=constant_half_steps,
            restart_threshold=0.5,
        )
        wrapper = self.make_wrapper(
            [[1.0]], [[-1.0]], [0.0], rho=1.0, acceleration=acceleration
        )
        wrapper.initialize(np.array([0.0]), np.array([0.0]))

        for value in (1.0, 2.1):
            state = np.array([value])
            wrapper.record_X_update(state)
            wrapper.record_Z_update(state)
            wrapper.update_dual_var()
            wrapper.finalize_round()

        # Round-two merit is 0.6**2, so it must not restart against 0.5 * 1.0.
        np.testing.assert_allclose(wrapper.get_X_step_bias(), [-2.65])

        state = np.array([5.0])
        wrapper.record_X_update(state)
        wrapper.record_Z_update(state)
        wrapper.update_dual_var()
        wrapper.finalize_round()
        np.testing.assert_allclose(wrapper.get_X_step_bias(), [-5.0])

    def test_adaptive_acceleration_rescales_all_dual_momentum(self):
        acceleration = NesterovAccelerationParams(step_sizes=constant_half_steps)
        adaptation = AdaptiveStepSizeParams(1.0, 2.0, 1)
        wrapper = self.make_wrapper(
            [[1.0]],
            [[-1.0]],
            [0.0],
            rho=1.0,
            acceleration=acceleration,
            adaptation=adaptation,
        )
        wrapper.initialize(np.array([0.0]), np.array([0.0]))
        wrapper.record_X_update(np.array([2.0]))
        wrapper.record_Z_update(np.array([0.0]))
        wrapper.update_dual_var()

        self.assertEqual(wrapper.finalize_round(), 2.0)
        np.testing.assert_allclose(wrapper.dual_var, [1.0])
        np.testing.assert_allclose(wrapper.get_X_step_bias(), [1.5])

        wrapper.record_X_update(np.array([0.0]))
        wrapper.record_Z_update(np.array([0.0]))
        wrapper.update_dual_var()
        wrapper.finalize_round()
        # The prior dual state was also rescaled from 2.0 to 1.0.
        np.testing.assert_allclose(wrapper.get_X_step_bias(), [1.75])

    def test_reinitialize_resets_rho_histories_and_round_state(self):
        adaptation = AdaptiveStepSizeParams(1.0, 2.0, 1)
        wrapper = self.make_wrapper(
            [[1.0]], [[-1.0]], [0.0], rho=1.0, adaptation=adaptation
        )
        wrapper.initialize(np.array([0.0]))
        wrapper.record_X_update(np.array([2.0]))
        wrapper.record_Z_update(np.array([0.0]))
        wrapper.update_dual_var()
        wrapper.finalize_round()
        self.assertEqual(wrapper.step_size, 2.0)

        wrapper.initialize(np.array([3.0]))
        self.assertEqual(wrapper.step_size, 1.0)
        np.testing.assert_allclose(wrapper.dual_var, [0.0])
        with self.assertRaises(RuntimeError):
            _ = wrapper.primal_infeasibility

    def test_invalid_round_order_and_shapes_are_rejected(self):
        wrapper = self.make_wrapper(np.eye(2), -np.eye(2), np.zeros(2), rho=1.0)
        with self.assertRaises(RuntimeError):
            wrapper.get_X_step_bias()
        wrapper.initialize(np.zeros(2))
        with self.assertRaises(RuntimeError):
            wrapper.record_Z_update(np.zeros(2))
        with self.assertRaises(ValueError):
            wrapper.record_X_update(np.zeros(3))
        wrapper.record_X_update(np.ones(2))
        with self.assertRaises(RuntimeError):
            wrapper.finalize_round()


class DistributedSharingWrapperTests(unittest.TestCase):
    def test_mode_selection_and_positional_adaptation_compatibility(self):
        adaptation = AdaptiveStepSizeParams(2.0, 2.0, 2)
        vanilla = DistributedSharingWrapper((2, 3), 1.0, adaptation)
        accelerated = DistributedSharingWrapper(
            (2, 3),
            1.0,
            acceleration=NesterovAccelerationParams(step_sizes=constant_half_steps),
        )
        restarted = DistributedSharingWrapper(
            (2, 3),
            1.0,
            acceleration=NesterovAccelerationParams(
                step_sizes=constant_half_steps,
                restart_threshold=0.5,
            ),
        )

        self.assertEqual(vanilla.mode, ADMMMode.VANILLA)
        self.assertEqual(accelerated.mode, ADMMMode.ACCELERATED)
        self.assertEqual(restarted.mode, ADMMMode.ACCELERATED_RESTART)

    def test_round_matches_reconstructed_full_residuals(self):
        X0 = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
        X1 = np.array([[2.0, 4.0, 7.0], [1.0, 6.0, 5.0]])
        Z_mean1 = np.array([4.0, 3.0])
        rho = 2.0
        wrapper = DistributedSharingWrapper(X0.shape, rho)
        wrapper.initialize(X0.mean(axis=1))

        np.testing.assert_allclose(wrapper.get_X_step_bias(), np.zeros(X0.shape[0]))
        delta_X_sq = float(np.sum((X1 - X0) ** 2))
        wrapper.record_mean_X_update(X1.mean(axis=1), delta_X_sq)
        np.testing.assert_allclose(wrapper.get_Z_step_bias(), X1.mean(axis=1))
        wrapper.record_mean_Z_update(Z_mean1)
        wrapper.update_dual_var()
        wrapper.finalize_round()

        Z0 = X0
        Z1 = Z_mean1[:, None] + X1 - X1.mean(axis=1)[:, None]
        np.testing.assert_allclose(wrapper.dual_var, X1.mean(axis=1) - Z_mean1)
        self.assertAlmostEqual(wrapper.primal_infeasibility, np.linalg.norm(X1 - Z1))
        self.assertAlmostEqual(wrapper.dual_infeasibility, rho * np.linalg.norm(Z1 - Z0))

    def test_primal_residual_has_full_frobenius_scaling(self):
        wrapper = DistributedSharingWrapper((1, 4), 1.0)
        wrapper.initialize(np.array([0.0]))
        wrapper.record_mean_X_update(np.array([2.0]), 16.0)
        wrapper.record_mean_Z_update(np.array([0.0]))
        wrapper.update_dual_var()
        wrapper.finalize_round()
        self.assertEqual(wrapper.primal_infeasibility, 4.0)

    def test_accelerated_rounds_match_reconstructed_mean_state(self):
        X0 = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
        X1 = np.array([[2.0, 4.0, 7.0], [1.0, 6.0, 5.0]])
        X2 = np.array([[3.0, 5.0, 6.0], [2.0, 4.0, 7.0]])
        Z_mean1 = np.array([4.0, 3.0])
        Z_mean2 = np.array([5.0, 4.0])
        rho = 2.0
        wrapper = DistributedSharingWrapper(
            X0.shape,
            rho,
            acceleration=NesterovAccelerationParams(step_sizes=constant_half_steps),
        )
        X_mean0 = X0.mean(axis=1)
        X_mean1 = X1.mean(axis=1)
        X_mean2 = X2.mean(axis=1)
        wrapper.initialize(X_mean0)

        wrapper.record_mean_X_update(X_mean1, float(np.sum((X1 - X0) ** 2)))
        np.testing.assert_allclose(wrapper.get_Z_step_bias(), X_mean1)
        wrapper.record_mean_Z_update(Z_mean1)
        wrapper.update_dual_var()
        dual1 = X_mean1 - Z_mean1
        np.testing.assert_allclose(wrapper.dual_var, dual1)
        wrapper.finalize_round()

        Z_mean_hat1 = Z_mean1 + 0.5 * (Z_mean1 - X_mean0)
        dual_hat1 = dual1 + 0.5 * dual1
        np.testing.assert_allclose(
            wrapper.get_X_step_bias(),
            X_mean1 - Z_mean_hat1 + dual_hat1,
        )

        wrapper.record_mean_X_update(X_mean2, float(np.sum((X2 - X1) ** 2)))
        np.testing.assert_allclose(wrapper.get_Z_step_bias(), X_mean2 + dual_hat1)
        wrapper.record_mean_Z_update(Z_mean2)
        wrapper.update_dual_var()
        dual2 = dual_hat1 + X_mean2 - Z_mean2
        np.testing.assert_allclose(wrapper.dual_var, dual2)
        wrapper.finalize_round()

        Z_hat1 = Z_mean_hat1[:, None] + X1 - X_mean1[:, None]
        Z2 = Z_mean2[:, None] + X2 - X_mean2[:, None]
        self.assertAlmostEqual(wrapper.primal_infeasibility, np.linalg.norm(X2 - Z2))
        self.assertAlmostEqual(
            wrapper.dual_infeasibility,
            rho * np.linalg.norm(Z2 - Z_hat1),
        )

        Z_mean_hat2 = Z_mean2 + 0.5 * (Z_mean2 - Z_mean1)
        dual_hat2 = dual2 + 0.5 * (dual2 - dual1)
        np.testing.assert_allclose(
            wrapper.get_X_step_bias(),
            X_mean2 - Z_mean_hat2 + dual_hat2,
        )

    def test_accelerated_restart_uses_squared_full_state_merit(self):
        wrapper = DistributedSharingWrapper(
            (1, 1),
            1.0,
            acceleration=NesterovAccelerationParams(
                step_sizes=constant_half_steps,
                restart_threshold=0.5,
            ),
        )
        wrapper.initialize(np.array([0.0]))

        previous = 0.0
        for value in (1.0, 2.1):
            state = np.array([value])
            wrapper.record_mean_X_update(state, (value - previous) ** 2)
            wrapper.record_mean_Z_update(state)
            wrapper.update_dual_var()
            wrapper.finalize_round()
            previous = value

        # The second merit is 0.6**2, so it stays below 0.5 * 1.0.
        np.testing.assert_allclose(wrapper.get_X_step_bias(), [-0.55])

        X_state = np.array([5.0])
        Z_state = np.array([4.0])
        wrapper.record_mean_X_update(X_state, (5.0 - previous) ** 2)
        wrapper.record_mean_Z_update(Z_state)
        wrapper.update_dual_var()
        wrapper.finalize_round()
        np.testing.assert_allclose(wrapper.get_X_step_bias(), [2.0])

    def test_accelerated_adaptation_rescales_all_dual_momentum(self):
        wrapper = DistributedSharingWrapper(
            (1, 1),
            1.0,
            adaptation=AdaptiveStepSizeParams(1.0, 2.0, 1),
            acceleration=NesterovAccelerationParams(step_sizes=constant_half_steps),
        )
        wrapper.initialize(np.array([0.0]))
        wrapper.record_mean_X_update(np.array([2.0]), 4.0)
        wrapper.record_mean_Z_update(np.array([0.0]))
        wrapper.update_dual_var()

        self.assertEqual(wrapper.finalize_round(), 2.0)
        np.testing.assert_allclose(wrapper.dual_var, [1.0])
        np.testing.assert_allclose(wrapper.get_X_step_bias(), [3.5])

        wrapper.record_mean_X_update(np.array([0.0]), 4.0)
        wrapper.record_mean_Z_update(np.array([0.0]))
        wrapper.update_dual_var()
        wrapper.finalize_round()
        np.testing.assert_allclose(wrapper.get_X_step_bias(), [1.75])

    def test_accelerated_reinitialize_resets_momentum_and_restart_history(self):
        wrapper = DistributedSharingWrapper(
            (1, 1),
            1.0,
            acceleration=NesterovAccelerationParams(
                step_sizes=constant_half_steps,
                restart_threshold=0.5,
            ),
        )
        wrapper.initialize(np.array([0.0]))
        wrapper.record_mean_X_update(np.array([1.0]), 1.0)
        wrapper.record_mean_Z_update(np.array([0.5]))
        wrapper.update_dual_var()
        wrapper.finalize_round()

        wrapper.initialize(np.array([3.0]))
        np.testing.assert_allclose(wrapper.get_X_step_bias(), [0.0])
        np.testing.assert_allclose(wrapper.dual_var, [0.0])
        with self.assertRaises(RuntimeError):
            _ = wrapper.primal_infeasibility

    def test_accelerated_invalid_dual_radicand_is_rejected(self):
        wrapper = DistributedSharingWrapper(
            (1, 2),
            1.0,
            acceleration=NesterovAccelerationParams(step_sizes=constant_half_steps),
        )
        wrapper.initialize(np.array([0.0]))
        wrapper.record_mean_X_update(np.array([1.0]), 1.0)
        wrapper.record_mean_Z_update(np.array([0.0]))
        wrapper.update_dual_var()
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            wrapper.finalize_round()

    def test_tiny_negative_dual_radicand_is_clamped(self):
        wrapper = DistributedSharingWrapper((1, 2), 1.0)
        wrapper.initialize(np.array([0.0]))
        wrapper.record_mean_X_update(np.array([1.0]), 2.0 - 1e-15)
        wrapper.record_mean_Z_update(np.array([0.0]))
        wrapper.update_dual_var()
        wrapper.finalize_round()
        self.assertEqual(wrapper.dual_infeasibility, 0.0)

    def test_materially_negative_dual_radicand_is_rejected(self):
        wrapper = DistributedSharingWrapper((1, 2), 1.0)
        wrapper.initialize(np.array([0.0]))
        wrapper.record_mean_X_update(np.array([1.0]), 1.0)
        wrapper.record_mean_Z_update(np.array([0.0]))
        wrapper.update_dual_var()
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            wrapper.finalize_round()

    def test_reinitialize_resets_adaptation_and_histories(self):
        adaptation = AdaptiveStepSizeParams(1.0, 2.0, 1)
        wrapper = DistributedSharingWrapper((1, 2), 1.0, adaptation)
        wrapper.initialize(np.array([0.0]))
        wrapper.record_mean_X_update(np.array([1.0]), 2.0)
        wrapper.record_mean_Z_update(np.array([0.0]))
        wrapper.update_dual_var()
        wrapper.finalize_round()
        self.assertEqual(wrapper.step_size, 2.0)

        wrapper.initialize(np.array([3.0]))
        self.assertEqual(wrapper.step_size, 1.0)
        np.testing.assert_allclose(wrapper.dual_var, [0.0])
        with self.assertRaises(RuntimeError):
            _ = wrapper.dual_infeasibility

    def test_invalid_inputs_and_round_order_are_rejected(self):
        with self.assertRaises(ValueError):
            DistributedSharingWrapper((2, 0), 1.0)
        wrapper = DistributedSharingWrapper((2, 3), 1.0)
        with self.assertRaises(ValueError):
            wrapper.initialize(np.zeros(3))
        wrapper.initialize(np.zeros(2))
        with self.assertRaises(RuntimeError):
            wrapper.record_mean_Z_update(np.zeros(2))
        with self.assertRaises(ValueError):
            wrapper.record_mean_X_update(np.zeros(2), -1.0)


if __name__ == "__main__":
    unittest.main()
