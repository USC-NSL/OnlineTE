import numpy as np
from enum import StrEnum
from dataclasses import dataclass
from typing import Optional, Callable, Iterator, Sequence, Tuple, List
from array_utils.cpu.types import *
from .nesterov import NesterovExtrapolationWrapper, default_nesterov_step_size


@dataclass(frozen=True)
class NesterovAccelerationParams:
    step_sizes: Callable[[], Iterator[float]] = default_nesterov_step_size
    restart_threshold: Optional[float] = None

    def __post_init__(self):
        if self.restart_threshold is not None and not 0 < self.restart_threshold < 1:
            raise ValueError("restart_threshold must be strictly between 0 and 1")


@dataclass(frozen=True)
class AdaptiveStepSizeParams:
    infeasibility_bound: float
    multiplicative_factor: float
    strike_count: int

    def __post_init__(self):
        if self.infeasibility_bound < 1:
            raise ValueError("infeasibility_bound must be at least 1")
        if self.multiplicative_factor < 1:
            raise ValueError("multiplicative_factor must be at least 1")
        if self.strike_count < 1:
            raise ValueError("strike_count must be at least 1")


class SimpleStepSizeAdapter:
    def __init__(self, initial_step_size: float, params: Optional[AdaptiveStepSizeParams] = None):
        if not np.isfinite(initial_step_size) or initial_step_size <= 0:
            raise ValueError("initial_step_size must be positive and finite")
        self._rho = initial_step_size
        self._params = params
        self._tokens = 0
        self._adapt_direction = 0

    def _how_adapt(self, p_inf: float, d_inf: float) -> int:
        if self._params is None:
            return 0
        MU = self._params.infeasibility_bound
        if p_inf > MU * d_inf:
            return 1
        elif d_inf > MU * p_inf:
            return -1
        return 0

    def adapt_step_size(self, p_inf: float, d_inf: float, dual_var: CPUArray) -> Optional[float]:
        if self._params is None:
            return
        adapt = self._how_adapt(p_inf, d_inf)
        if adapt == 0:
            self._tokens = 0
            self._adapt_direction = 0
            return

        if adapt != self._adapt_direction:
            self._adapt_direction = adapt
            self._tokens = 1
        else:
            self._tokens += 1

        if (self._tokens >= self._params.strike_count):
            self._tokens = 0
            self._adapt_direction = 0
            TAU = self._params.multiplicative_factor
            if adapt > 0:
                self._rho *= TAU
                dual_var /= TAU
                return self._rho
            elif adapt < 0:
                self._rho /= TAU
                dual_var *= TAU
                return self._rho

    def reset(self, new_rho: float, dual_var: CPUArray):
        if not np.isfinite(new_rho) or new_rho <= 0:
            raise ValueError("new_rho must be positive and finite")
        self._tokens = 0
        self._adapt_direction = 0
        dual_var *= (self._rho / new_rho)
        self._rho = new_rho


class ADMMMode(StrEnum):
    VANILLA = "Vanilla"
    """Simple and honest ADMM."""
    ACCELERATED = "Accelerated"
    """
    Applies a Nesterov-style acceleration (see Goldstein et al., 2014)
    to the second state and dual variable. Essentially, given extrapolated
    second state `Z_hat^{m}` and dual variable `dual_hat^{m}` from the last
    ADMM update iteration `m`:

    - **`X` update:** Penalized with `|| A . X + B . Z_hat^{m} - C + dual_hat^{m} ||_2^2`
    - **`Z` update:** Penalized with `|| A . X^{m+1} + B . Z - C + dual_hat^{m} ||_2^2`
    - **`dual` update:** `dual^{m+1} = dual_hat^{m} + (A . X^{m+1} + B . Z^{m+1} - C)

    This is then followed by extrapolation:

    - `Z_hat^{m+1} = Z^{m+1} + nu^{m+1} (Z^{m+1} - Z^{m})
    - `dual_hat^{m+1} = dual^{m+1} + nu^{m+1} (dual^{m+1} - dual^{m})

    For a step size `nu`. The step size can either be given by the user
    directly, or calculated according to Nesterov's original method which
    needs:

        nu^{m+1} = (1 + sqrt{1 + 4 . nu^{m}**2}) / 2

    With `nu^{0} = 1`.
    """
    OVER_RELAXED = "Over-Relaxed"
    """
    Given a parameter `alpha`, it generates a relaxed state after the
    `X` update:

        S_alpha^{k+1} = alpha . A . X^{k+1} + (alpha - 1) (B . Z^{k} - C)

    This state replaces `AX^{k+1}` in the `Z` step and dual update:

    - **`Z` update:** Penalized with `|| S_alpha^{k+1} + B . Z - C + dual_hat^{m} ||_2^2`
    - **`dual` update:** `dual^{m+1} = dual_hat^{m} + (S_alpha^{k+1} + B . Z^{m+1} - C)
    """
    ACCELERATED_RESTART = "Accelerated With Restart"
    """
    Acceleration is only guaranteed to converge (and give `O(1/k^2)` convergence)
    under strong convexity. If that is not the case, a restart scheme is often
    needed to prevent oscillations.

    Restart means to set `nu` to 1 and forget all state for extrapolated `Z` and
    dual variable. The criteria for determining when to restart are not really
    set in stone.
    Once scheme is to monitor the total infeasibilities before applying/updating
    momentum for the next iteration (i.e. after the dual update).

    We look at:

        res^{m+1} = ||A . X^{m+1} + B . Z^{m+1} - C||_2^2 / rho +
                    ||B . (Z^{m+1} - Z_hat^{m})||_2^2 . rho

    Note that the first term is essentially just scaled and squared ADMM primal
    infeasibility, but the second term is _NOT_ the dual infeasibility (it
    is mssing a projection `A^T`).
    Given a user-specified parameter `0 < kappa < 1`, we determine the need for a
    restart by checking:

        res^{m+1} > kappa . res^{m}
    """
    # ACCELERATED_OVER_RELAXED = "Accelerated-Over-Relaxed"
    """
    TODO: Just combining the two as-is causes a lot of problems! We should be careful.
          I am removing this option for the time being, I am not confident in it.
    """


class ADMMWrapper:
    """
    A wrapper for a generic ADMM algorithm.
    The main purpose is to handle Nesterov-style acceleration or over-relaxation
    without having to track what is happening to the biases used for each step.
    It also keeps the dual variable hidden, updating it automatically.

    Currently, this only works with `Numpy`. When needed, will make it compatible
    with `Cupy` as well.

    Citations
    ---------
    The description of how over-relaxation and momentum factors into the biases
    comes from the following sources:

    - Tavakoli, M., Jakob, F., Carnevale, G., Notarstefano, G., & Iannelli, A. (2025). 
      Accelerated ADMM: Automated Parameter Tuning and Improved Linear Convergence. arXiv preprint arXiv:2511.21210.
    - Goldstein, T., O'Donoghue, B., Setzer, S., & Baraniuk, R. (2014). 
      Fast alternating direction optimization methods. SIAM Journal on Imaging Sciences, 7(3), 1588-1623.

    Notation
    --------
    ADMM is usually applied to a constrain of the form `AX + BZ = C` and a dual variable
    named `dual`.
    """
    def __init__(
        self,
        rho: float,
        alpha: Optional[float] = None,
        acceleration: Optional[NesterovAccelerationParams] = None,
        adaptation: Optional[AdaptiveStepSizeParams] = None
    ):
        if not np.isfinite(rho) or rho <= 0:
            raise ValueError("rho must be positive and finite")
        self._set_mode(alpha, acceleration)

        # ADMM state variables
        self._initial_rho = rho
        """Current ADMM step size."""
        self._X: Optional[CPUArray] = None
        """First state variable"""
        self._Z: Optional[CPUArray] = None
        """Second state variable"""
        self._Z_old: Optional[CPUArray] = None
        """
        The previous second state variable (we use this for checking the dual
        infeasibility).
        Note that:
        - When there is no acceleration, this is just the copy of the previous
          second state variable.
        - When there is acceleration, this is the previous _extrapolated_ second
          state variable.
        """
        self._dual_var: Optional[CPUArray] = None
        """The ADMM dual variable"""
        self._A: Optional[CPUArray] = None
        self._B: Optional[CPUArray] = None
        self._C: Optional[CPUArray] = None
        self._pinf_list: List[float] = []
        """Cached primal infeasibilities"""
        self._dinf_list: List[float] = []
        """Cached dual infeasibilities"""

        # Relaxation specific state
        self._alpha = alpha
        """
        Relxation coefficient.
        If it is `None`, then it is interpreted as it being 1.
        """
        self._relaxed_state: Optional[CPUArray] = None
        """Cached relaxation state (only used when over-rexlation is enabled)"""

        # Acceleration specific state
        self._acceleration = acceleration
        """Acceleration params"""
        self._momentum_inf_list: List[float] = []
        """Cached momentum infeasibilities (only used for restarts)"""
        self._Z_nesterov: Optional[NesterovExtrapolationWrapper] = None
        """The Nesterov extrapolation wrapper for the second state variable"""
        self._dual_nesterov: Optional[NesterovExtrapolationWrapper] = None
        """The Nesterov extrapolation wrapper for the dual variable"""

        # Step size adaptation
        self._step_adapter: SimpleStepSizeAdapter = \
            SimpleStepSizeAdapter(
                initial_step_size=rho,
                params=adaptation
            )

        # House-keeping variables
        self._finalized = True
        """
        A simple state variable to make sure changes are finalized before
        starting the next iteration.
        """
        self._iterate: int = 0
        """The current step iteration count"""
        self._initialized = False
        self._round_stage = "uninitialized"

    def _set_shapes(
        self,
        A_shape: Sequence[int],
        B_shape: Sequence[int],
        C_shape: Sequence[int]
    ):
        if len(A_shape) != 2 or len(B_shape) != 2:
            raise ValueError("A and B must be two-dimensional matrices")
        if len(C_shape) not in (1, 2):
            raise ValueError("C must be a vector or a two-dimensional matrix")
        if A_shape[0] != C_shape[0] or B_shape[0] != C_shape[0]:
            raise ValueError("A, B, and C must have the same constraint dimension")
        if len(C_shape) == 1:
            self._X_shape: Tuple[int] = (A_shape[1],)
            self._Z_shape: Tuple[int] = (B_shape[1],)
        else:
            self._X_shape: Tuple[int] = (A_shape[1], C_shape[1])
            self._Z_shape: Tuple[int] = (B_shape[1], C_shape[1])
        self._dual_shape = C_shape

    def _set_mode(
        self,
        alpha: Optional[float] = None,
        acceleration_params: Optional[NesterovAccelerationParams] = None
    ):
        if alpha is None:
            if acceleration_params is not None:
                if acceleration_params.restart_threshold is None:
                    self._mode = ADMMMode.ACCELERATED
                else:
                    self._mode = ADMMMode.ACCELERATED_RESTART
            else:
                self._mode = ADMMMode.VANILLA
        else:
            if acceleration_params is not None:
                raise ValueError("Acceleration and over-relaxation cannot be combined")
            self._mode = ADMMMode.OVER_RELAXED

    def set_matrices(self, A: CPUArray, B: CPUArray, C: CPUArray):
        self._set_shapes(A.shape, B.shape, C.shape)
        self._A = cpu_array(A)
        """
        State projection matrix 1.
        """
        self._B = cpu_array(B)
        """
        State projection matrix 2.
        """
        self._C = cpu_array(C)
        """
        Constraint matrix.
        """
        self._initialized = False
        self._round_stage = "uninitialized"
    
    def initialize(self, X_start: CPUArray, Z_start: Optional[CPUArray] = None):
        """Initialize the algorithm and its dual variable iterates"""
        if self._A is None or self._B is None or self._C is None:
            raise RuntimeError("set_matrices must be called before initialize")
        if X_start.shape != self._X_shape:
            raise ValueError(f"X_start must have shape {self._X_shape}")
        self._X = cpu_array(X_start)
        if Z_start is not None:
            if Z_start.shape != self._Z_shape:
                raise ValueError(f"Z_start must have shape {self._Z_shape}")
            self._Z = cpu_array(Z_start)
        else:
            if self._X_shape != self._Z_shape:
                raise ValueError("Z_start is required when X and Z have different shapes")
            self._Z = cpu_array(X_start)
        self._dual_var = cpu_zeros(self._dual_shape)
        self._step_adapter.reset(self._initial_rho, self._dual_var)
        self._iterate = 0
        self._Z_old = cpu_array(self._Z)
        self._pinf_list.clear()
        self._dinf_list.clear()
        self._momentum_inf_list.clear()
        # Set the mode specific properties
        match self._mode:
            case ADMMMode.ACCELERATED | ADMMMode.ACCELERATED_RESTART:
                self._dual_nesterov = NesterovExtrapolationWrapper(self._dual_var, self._acceleration.step_sizes)
                self._Z_nesterov = NesterovExtrapolationWrapper(self._Z, self._acceleration.step_sizes)
            case ADMMMode.OVER_RELAXED:
                self._relaxed_state = np.empty_like(self._dual_var)
            case _: pass
        self._finalized = True
        self._initialized = True
        self._round_stage = "finalized"

    def _require_initialized(self):
        if not self._initialized:
            raise RuntimeError("ADMMWrapper has not been initialized")

    def _require_stage(self, expected: str):
        self._require_initialized()
        if self._round_stage != expected:
            raise RuntimeError(
                f"Expected ADMM round stage '{expected}', got '{self._round_stage}'"
            )
    
    @property
    def mode(self) -> ADMMMode:
        return self._mode
    
    @property
    def dual_var(self) -> CPUArray:
        self._require_initialized()
        return self._dual_var
    @property
    def X(self) -> CPUArray:
        self._require_initialized()
        return self._X
    @property
    def Z(self) -> CPUArray:
        self._require_initialized()
        return self._Z
    @property
    def step_size(self) -> float:
        return self._step_adapter._rho

    def _mul_A(self, thing: CPUArray) -> CPUArray:
        return self._A @ thing

    def _mul_rho_A_t(self, thing: CPUArray) -> CPUArray:
        return self.step_size * (self._A.T @ thing)

    def _mul_B(self, thing: CPUArray) -> CPUArray:
        return self._B @ thing
    
    def _add_C(self, thing: CPUArray) -> CPUArray:
        return thing + self._C

    def _sub_C(self, thing: CPUArray) -> CPUArray:
        return thing - self._C

    def _set_relaxed_state(self, X: CPUArray):
        if self._mode != ADMMMode.OVER_RELAXED:
            raise RuntimeError("Relaxed state is only available in over-relaxed mode")
        self._relaxed_state = \
            self._alpha * self._mul_A(X) + \
            (self._alpha - 1) * self._sub_C(self._mul_B(self._Z))
    
    def get_X_step_bias(self) -> CPUArray:
        """
        Return `b` such that the ADMM penalty is:
            || AX + b ||_2^2
        """
        self._require_stage("finalized")
        match self._mode:
            case ADMMMode.ACCELERATED | ADMMMode.ACCELERATED_RESTART:
                Z_term = self._mul_B(self._Z_nesterov.extrapolated_state)
                dual_term = self._dual_nesterov.extrapolated_state
            case _:
                Z_term = self._mul_B(self._Z)
                dual_term = self._dual_var
        return self._sub_C(Z_term + dual_term)

    def get_Z_step_bias(self) -> CPUArray:
        """
        Return `b` such that the ADMM penalty is:
            || BZ + b ||_2^2
        """
        self._require_stage("x_recorded")
        match self._mode:
            case ADMMMode.ACCELERATED | ADMMMode.ACCELERATED_RESTART:
                X_term = self._mul_A(self._X)
                dual_term = self._dual_nesterov.extrapolated_state
            case ADMMMode.OVER_RELAXED:
                X_term = self._relaxed_state
                dual_term = self._dual_var
            case _:
                X_term = self._mul_A(self._X)
                dual_term = self._dual_var
        return self._sub_C(X_term + dual_term)
    
    def record_X_update(self, next_X: CPUArray):
        """Record the X-update step output"""
        self._require_stage("finalized")
        if next_X.shape != self._X_shape:
            raise ValueError(f"next_X must have shape {self._X_shape}")
        self._X = cpu_array(next_X)
        if self._mode == ADMMMode.OVER_RELAXED:
            self._set_relaxed_state(next_X)
        self._finalized = False
        self._round_stage = "x_recorded"
    
    def record_Z_update(self, next_Z: CPUArray):
        """Record the Z-update step output"""
        self._require_stage("x_recorded")
        if next_Z.shape != self._Z_shape:
            raise ValueError(f"next_Z must have shape {self._Z_shape}")
        match self._mode:
            case ADMMMode.ACCELERATED | ADMMMode.ACCELERATED_RESTART:
                np.copyto(self._Z_old, self._Z_nesterov.extrapolated_state)
            case _: np.copyto(self._Z_old, self._Z)
        self._Z = cpu_array(next_Z)
        self._round_stage = "z_recorded"

    def update_dual_var(self):
        """
        Update the dual variable.
        """
        self._require_stage("z_recorded")
        match self._mode:
            case ADMMMode.ACCELERATED | ADMMMode.ACCELERATED_RESTART:
                self._dual_var = \
                    self._dual_nesterov.extrapolated_state + \
                    self._sub_C(self._mul_A(self._X) + self._mul_B(self._Z))
            case ADMMMode.OVER_RELAXED:
                self._dual_var += \
                    self._relaxed_state + \
                    self._sub_C(self._mul_B(self._Z))
            case _:
                self._dual_var += \
                    self._sub_C(self._mul_A(self._X) + self._mul_B(self._Z))
        self._round_stage = "dual_updated"

    def _record_infeasibilities(self):
        rho = self.step_size
        r = self._get_primal_residual()
        p_inf = float(np.linalg.norm(r))
        s, m = self._get_dual_and_momentum_residual()
        d_inf = float(np.linalg.norm(s))
        m_inf = p_inf**2 / rho + rho * float(np.linalg.norm(m))**2
        self._pinf_list.append(p_inf)
        self._momentum_inf_list.append(m_inf)
        self._dinf_list.append(d_inf)

    def _update_momentum_or_restart(self):
        match self._mode:
            case ADMMMode.ACCELERATED:
                self._Z_nesterov.update(self._Z)
                self._dual_nesterov.update(self._dual_var)
            case ADMMMode.ACCELERATED_RESTART:
                if len(self._momentum_inf_list) >= 2:
                    last_inf = self._momentum_inf_list[-2]
                    current_inf = self._momentum_inf_list[-1]
                    if current_inf > self._acceleration.restart_threshold * last_inf:
                        self._dual_nesterov.restart(self._dual_var)
                        self._Z_nesterov.restart(self._Z)
                        self._momentum_inf_list.clear()
                        return
                self._Z_nesterov.update(self._Z)
                self._dual_nesterov.update(self._dual_var)
            case _: pass

    def finalize_round(self) -> Optional[float]:
        """
        Finalize the ADMM round.
        If needed, prepares the step size and momentum terms for the next
        iteration.
        When the step size is modified, this function will return the
        new value.
        """
        self._require_stage("dual_updated")
        self._record_infeasibilities()
        # Update momentum state or restart
        match self._mode:
            case ADMMMode.ACCELERATED | ADMMMode.ACCELERATED_RESTART:
                self._update_momentum_or_restart()
            case _: pass
        self._iterate += 1
        self._finalized = True
        self._round_stage = "finalized"
        old_rho = self.step_size
        new_rho = self._step_adapter.adapt_step_size(
            self.primal_infeasibility,
            self.dual_infeasibility,
            self._dual_var
        )
        if new_rho is not None and self._dual_nesterov is not None:
            self._dual_nesterov.rescale(old_rho / new_rho)
        return new_rho
    
    def _get_primal_residual(self) -> CPUArray:
        return self._sub_C(self._mul_A(self._X) + self._mul_B(self._Z))

    def _get_momentum_residual(self) -> CPUArray:
        return self._mul_B(self._Z - self._Z_old)
    
    def _get_dual_and_momentum_residual(self) -> Tuple[CPUArray, CPUArray]:
        momentum_residual = self._get_momentum_residual()
        dual_residual = self._mul_rho_A_t(momentum_residual)
        return dual_residual, momentum_residual

    @property
    def primal_infeasibility(self) -> float:
        if not self._pinf_list:
            raise RuntimeError("No ADMM round has been finalized")
        return self._pinf_list[-1]

    @property
    def dual_infeasibility(self) -> float:
        if not self._dinf_list:
            raise RuntimeError("No ADMM round has been finalized")
        return self._dinf_list[-1]
    
    @property
    def infeasibility(self) -> float:
        p_inf = self.primal_infeasibility
        d_inf = self.dual_infeasibility
        return p_inf + d_inf


"""
The wrapper for an ADMM sharing problem:

    minimize sum_k f_k(X_k) + g(sum_k X_k)

Which is rewritten as:

    minimize sum_k f_k(X_k) + g(sum_k Z_k)
        s.t. X - Z = 0

Which admits `A = I`, `B = -I` and `C = 0`. In this setting, the
over-relaxation state is:

    alpha . X^{m+1} + (1 - alpha) Z^{m}

Which means that the Z and dual updates become:

    Z^{m+1} = argmin g(sum_k Z_k) + sum_k rho/2 || alpha . X_k^{m+1} + (1 - alpha) Z_k^{m} - Z_k + D_k^{m} ||_2^2
    D^{m+1} = D^{m} + (alpha . X^{m+1} + (1 - alpha) Z^{m} - Z^{m+1})

A nice result in this setting though, is that the `Z^{m+1}` satisfies:

    Z_k^{m+1} = mean(Z^{m+1}) + (1 - alpha) (Z_k^{m} - mean(Z^{m})) + alpha (X_k^{m+1} - mean(X^{m+1}))

And the columns of the dual variable are in consensus, hence we need only store one column, which is
updated as:

    D^{m+1} = D^{m} + alpha mean(X^{m+1}) + (1 - alpha) mean(Z^{m}) - mean(Z^{m+1})

Where `mean(.)` is mean over the columns.

Using this relation, we can re-write the X-update as:

    X_k^{m+1} = argmin f_k(X_k) + rho/2 || X_k - mean(Z^{m}) - (1 - alpha) (Z_k^{m-1} - mean(Z^{m-1})) - alpha (X_k^{m} - mean(X^{m+1})) + D^{m} ||_2^2
                = argmin f_k(X_k) + rho/2 || X_k - alpha . X_k^{m} - (1 - alpha) . Z_k^{m-1} - mean(Z^{m}) + (1 - alpha) . mean(Z^{m-1}) + alpha . mean(X^{m}) + D^{m} ||_2^2

A particulaly annoying thing in this setting is that the X-update depends on the last _two_ `Z` updates rather than
just the last one. As such, over-relaxation is usually not worth it as it effectively erases the benefit of having
just one dual variable.
As such, our implementation currently avoid over-relaxation for this setting.

Without relaxation, the problem becomes:

    X_k^{m+1} = argmin f_k(X_k) + rho/2 || X_k - X_k^{m} + mean(X^{m}) - mean(Z^{m}) + D^{m} ||_2^2
    mean(Z^{m+1}) = argmin g(K Z_bar) + K rho/2 || mean(X^{m+1}) - Z_bar + D^{m} ||_2^2
    Z^{m+1} = mean(Z^{m+1}) + X^{m+1} - mean(X^{m+1})
    D^{m+1} = D^{m} + (mean(X^{m+1}) - mean(Z^{m+1}))

When acceleration is enabled, this wrapper accelerates only the coordinator-held
mean of Z and the consensus dual. It reconstructs the extrapolated columns as:

    Z_hat_k^{m} = mean(Z_hat^{m}) + X_k^{m} - mean(X^{m})

The centered column deviations are deliberately not extrapolated. The updates are:

    X_k^{m+1} = argmin f_k(X_k) + rho/2 ||X_k - X_k^m
        + mean(X^m) - mean(Z_hat^m) + D_hat^m||_2^2
    mean(Z^{m+1}) = argmin g(K Z_bar) + K rho/2
        ||mean(X^{m+1}) - Z_bar + D_hat^m||_2^2
    D^{m+1} = D_hat^m + mean(X^{m+1}) - mean(Z^{m+1})

This is reduced mean-state acceleration, not full-column accelerated ADMM.

The infeasibilities at the end of step `m+1` in this setting are:

    primal infeasibility = sqrt(K) || mean(X^{m+1}) - mean(Z^{m+1}) ||_2
      dual infeasibility = rho ||X^{m+1} - X^{m} - mean(X^{m+1})
                                + mean(X^{m}) + mean(Z^{m+1}) - mean(Z^{m})||_F
"""

class DistributedSharingWrapper:
    """
    This is a wrapper for the sharing problem that we usually solve.
    In this setting:
    - Individual X columns are in remote nodes, and out of reach. We
      can only work with their mean values.
    - The Z_bar-update and dual variable updates are done locally.
    - Optional acceleration applies only to Z_bar and the consensus dual.


    Note
    ----
    One particular point to consider is how we calculate the dual
    infeasibility.
    Let `Delta^{m+1}(.)` be the different between an iterate value
    at iteration `m+1` and `m`.
    Recall that the dual residual takes the form:

        s_k^{m+1} = -rho ( Delta^{m+1}(X_k) - Delta^{m+1}(mean(X)) + Delta^{m+1}(mean(Z)) )

    The first term is not available on the central node, unless we take
    another round-trip to the worker nodes.
    Note that the infeasibility is:

        rho sqrt(
            || Delta^{m+1}(X) ||^2_F +
            K || Delta^{m+1}(mean(X)) - Delta^{m+1}(mean(Z)) ||_2^2 +
            - 2 sum_k < Delta^{m+1}(X_k), Delta^{m+1}(mean(X)) - Delta^{m+1}(mean(Z)) >
        )

    The first term is available only on the worker nodes, but its value
    is known at the end of each X-update. Hence, the worker node can
    report it to the central node.
    The second term is readily known on the central node after the update
    to the mean of Z.
    The final term can be expanded as:

        -2K <Delta^{m+1}(mean(X)), Delta^{m+1}(mean(X)) - Delta^{m+1}(mean(Z))>

    Which when combined, gives the infeasibility as:

        rho sqrt(
            || Delta^{m+1}(X) ||^2_F +
            K || Delta^{m+1} (mean(Z)) ||_2^2 - K || Delta^{m+1} (mean(X)) ||_2^2
        )

    With acceleration, `Delta(mean(Z))` above is measured from the previous
    extrapolated mean, rather than the previous unshifted mean.

    As such, if the worker nodes promise to also tell us the Frobenius norm
    of the final X iterates, we can certify feasibility on the central node.
    """
    def __init__(
        self,
        X_shape: Tuple[int, int],
        rho: float,
        adaptation: Optional[AdaptiveStepSizeParams] = None,
        acceleration: Optional[NesterovAccelerationParams] = None
    ):
        if len(X_shape) != 2 or X_shape[0] <= 0 or X_shape[1] <= 0:
            raise ValueError("X_shape must contain two positive dimensions")
        if not np.isfinite(rho) or rho <= 0:
            raise ValueError("rho must be positive and finite")
        self._X_shape = X_shape
        self._initial_rho = rho
        self._dual_shape = (X_shape[0],)
        self._K = X_shape[1]
        self._acceleration = acceleration
        if acceleration is None:
            self._mode = ADMMMode.VANILLA
        elif acceleration.restart_threshold is None:
            self._mode = ADMMMode.ACCELERATED
        else:
            self._mode = ADMMMode.ACCELERATED_RESTART

        self._X_mean: Optional[CPUArray] = None
        self._X_mean_old: Optional[CPUArray] = None
        self._delta_X_frobenius: Optional[float] = None
        self._Z_mean: Optional[CPUArray] = None
        self._Z_mean_old: Optional[CPUArray] = None
        self._dual_var: Optional[CPUArray] = None
        self._Z_mean_nesterov: Optional[NesterovExtrapolationWrapper] = None
        self._dual_nesterov: Optional[NesterovExtrapolationWrapper] = None

        self._pinf_list: List[float] = []
        self._dinf_list: List[float] = []
        self._momentum_inf_list: List[float] = []

        # Step size adaptation
        self._step_adapter: SimpleStepSizeAdapter = \
            SimpleStepSizeAdapter(
                initial_step_size=rho,
                params=adaptation
            )
        self._initialized = False
        self._round_stage = "uninitialized"

    def _require_initialized(self):
        if not self._initialized:
            raise RuntimeError("DistributedSharingWrapper has not been initialized")

    def _require_stage(self, expected: str):
        self._require_initialized()
        # if self._round_stage != expected:
        #     raise RuntimeError(
        #         f"Expected sharing round stage '{expected}', got '{self._round_stage}'"
        #     )

    @property
    def mode(self) -> ADMMMode:
        return self._mode
    @property
    def dual_var(self) -> CPUArray:
        self._require_initialized()
        return self._dual_var
    @property
    def X_mean(self) -> CPUArray:
        self._require_initialized()
        return self._X_mean
    @property
    def Z_mean(self) -> CPUArray:
        self._require_initialized()
        return self._Z_mean
    @property
    def step_size(self) -> float:
        return self._step_adapter._rho
    @property
    def primal_infeasibility(self) -> float:
        if not self._pinf_list:
            raise RuntimeError("No sharing round has been finalized")
        return self._pinf_list[-1]
    @property
    def dual_infeasibility(self) -> float:
        if not self._dinf_list:
            raise RuntimeError("No sharing round has been finalized")
        return self._dinf_list[-1]
    @property
    def infeasibility(self) -> float:
        p_inf = self.primal_infeasibility
        d_inf = self.dual_infeasibility
        return p_inf + d_inf

    def initialize(self, X_mean: CPUArray):
        if X_mean.shape != self._dual_shape:
            raise ValueError(f"X_mean must have shape {self._dual_shape}")
        self._X_mean = cpu_array(X_mean)
        self._X_mean_old = cpu_array(X_mean)
        self._Z_mean = cpu_array(X_mean)
        self._Z_mean_old = cpu_array(X_mean)
        self._dual_var = cpu_zeros(self._dual_shape)
        self._step_adapter.reset(self._initial_rho, self._dual_var)
        self._delta_X_frobenius = None
        self._pinf_list.clear()
        self._dinf_list.clear()
        self._momentum_inf_list.clear()
        if self._acceleration is not None:
            self._Z_mean_nesterov = NesterovExtrapolationWrapper(
                self._Z_mean, self._acceleration.step_sizes
            )
            self._dual_nesterov = NesterovExtrapolationWrapper(
                self._dual_var, self._acceleration.step_sizes
            )
        else:
            self._Z_mean_nesterov = None
            self._dual_nesterov = None
        self._initialized = True
        self._round_stage = "finalized"

    def get_X_step_bias(self) -> CPUArray:
        """
        Return `b` such that the ADMM penalty is:
            || X - X^{m} + b ||_2^2
        """
        self._require_stage("finalized")
        if self._acceleration is None:
            Z_mean = self._Z_mean
            dual_var = self._dual_var
        else:
            Z_mean = self._Z_mean_nesterov.extrapolated_state
            dual_var = self._dual_nesterov.extrapolated_state
        return self._X_mean - Z_mean + dual_var

    def get_Z_step_bias(self) -> CPUArray:
        """
        Return `b` such that the ADMM penalty is:
            || -Z_bar + b ||_2^2
        """
        self._require_stage("x_recorded")
        dual_var = self._dual_var if self._acceleration is None \
            else self._dual_nesterov.extrapolated_state
        return self._X_mean + dual_var

    def record_mean_X_update(self, next_X_mean: CPUArray, delta_X_frobenius: float):
        """Record the X update and its squared full-matrix Frobenius change."""
        self._require_stage("finalized")
        if next_X_mean.shape != self._dual_shape:
            raise ValueError(f"next_X_mean must have shape {self._dual_shape}")
        if np.isnan(delta_X_frobenius) or delta_X_frobenius < 0:
            raise ValueError("delta_X_frobenius must be a nonnegative squared norm")
        self._X_mean_old = self._X_mean
        self._X_mean = cpu_array(next_X_mean)
        self._delta_X_frobenius = float(delta_X_frobenius)
        self._round_stage = "x_recorded"

    def record_mean_Z_update(self, next_Z_mean: CPUArray):
        """Record the Z-update step output through the mean"""
        self._require_stage("x_recorded")
        if next_Z_mean.shape != self._dual_shape:
            raise ValueError(f"next_Z_mean must have shape {self._dual_shape}")
        if self._acceleration is None:
            self._Z_mean_old = self._Z_mean
        else:
            np.copyto(
                self._Z_mean_old,
                self._Z_mean_nesterov.extrapolated_state
            )
        self._Z_mean = cpu_array(next_Z_mean)
        self._round_stage = "z_recorded"

    def update_dual_var(self):
        """Update the dual variable"""
        self._require_stage("z_recorded")
        if self._acceleration is None:
            self._dual_var += (self._X_mean - self._Z_mean)
        else:
            self._dual_var = \
                self._dual_nesterov.extrapolated_state + \
                (self._X_mean - self._Z_mean)
        self._round_stage = "dual_updated"

    def _record_infeasibilities(self):
        p_inf = self._K**0.5 * float(np.linalg.norm(self._X_mean - self._Z_mean))
        z_delta_sq = float(np.linalg.norm(self._Z_mean - self._Z_mean_old)**2)
        x_delta_sq = float(np.linalg.norm(self._X_mean - self._X_mean_old)**2)
        radicand = self._delta_X_frobenius + self._K * (z_delta_sq - x_delta_sq)
        scale = self._delta_X_frobenius + self._K * (z_delta_sq + x_delta_sq)
        dtype = self._X_mean.dtype
        epsilon = np.finfo(dtype).eps \
            if np.issubdtype(dtype, np.floating) else np.finfo(float).eps
        tolerance = 64 * epsilon * max(1.0, float(scale))
        if radicand < -tolerance:
            raise ValueError(
                "Distributed dual residual squared is negative; "
                "delta_X_frobenius is inconsistent with the reported means"
            )
        momentum_inf_squared = max(0.0, radicand)
        d_inf = self.step_size * float(np.sqrt(momentum_inf_squared))
        merit = p_inf**2 / self.step_size + self.step_size * momentum_inf_squared
        self._pinf_list.append(p_inf)
        self._dinf_list.append(d_inf)
        self._momentum_inf_list.append(merit)

    def _update_momentum_or_restart(self):
        if self._mode == ADMMMode.ACCELERATED_RESTART \
            and len(self._momentum_inf_list) >= 2:
            previous = self._momentum_inf_list[-2]
            current = self._momentum_inf_list[-1]
            if current > self._acceleration.restart_threshold * previous:
                self._Z_mean_nesterov.restart(self._Z_mean)
                self._dual_nesterov.restart(self._dual_var)
                self._momentum_inf_list.clear()
                return
        self._Z_mean_nesterov.update(self._Z_mean)
        self._dual_nesterov.update(self._dual_var)

    def finalize_round(self) -> Optional[float]:
        """Finalize the ADMM round"""
        self._require_stage("dual_updated")
        self._record_infeasibilities()
        if self._acceleration is not None:
            self._update_momentum_or_restart()
        old_rho = self.step_size
        new_rho = self._step_adapter.adapt_step_size(
            p_inf=self.primal_infeasibility,
            d_inf=self.dual_infeasibility,
            dual_var=self._dual_var
        )
        if new_rho is not None and self._dual_nesterov is not None:
            self._dual_nesterov.rescale(old_rho / new_rho)
        self._round_stage = "finalized"
        return new_rho

    def is_feasible(self, eps_abs: float, eps_rel: float) -> bool:
        N, K = self._X_shape
        eps_primal = eps_abs * np.sqrt(N) + eps_rel * max([
            np.linalg.norm(self._X_mean),
            np.linalg.norm(self._Z_mean)
        ]) * np.sqrt(K)
        eps_dual = eps_abs * np.sqrt(N) + eps_rel * np.linalg.norm(self._dual_var) * np.sqrt(K)
        return self.primal_infeasibility <= eps_primal and \
            self.dual_infeasibility <= eps_dual
