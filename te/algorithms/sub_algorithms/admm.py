import enum
import math
import numpy as np
from typing import Optional, Tuple
from te.algorithms.array_utils.cpu_utils import CPUArray, cpu_array, cpu_zeros


class ADMMMode(str, enum.Enum):
    VANILLA = "Vanilla"
    ACCELERATED = "Accelerated"
    OVER_RELAXED = "Over-Relaxed"
    ACCELERATED_OVER_RELAXED = "Accelerated-Over-Relaxed"


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
    """
    def __init__(self, n: int, rho: float, 
                 A: Optional[CPUArray] = None, B: Optional[CPUArray] = None, C: Optional[CPUArray] = None,
                 alpha: Optional[float] = None, nesterov: bool = False, nesterov_coeff: Optional[float] = None):
        # The ADMM mode is determined by the parameters
        if alpha is None:
            if nesterov is False:
                self._mode: ADMMMode = ADMMMode.VANILLA
            else:
                self._mode: ADMMMode = ADMMMode.ACCELERATED
        else:
            if nesterov is False:
                self._mode: ADMMMode = ADMMMode.OVER_RELAXED
            else:
                self._mode: ADMMMode = ADMMMode.ACCELERATED_OVER_RELAXED

        self._n = n
        self._rho = rho
        # If this is `None`, it is interpreted the same as being identity
        self._A = A
        # If this is `None`, it is interpreted the same as being _MINUS_ identity
        self._B = B
        # If this is `None`, it is interpreted as being all zeros
        self._C = C
        # If this is `None`, then it is interpreted as it being 1
        self._alpha = alpha
        # Nesterov momentum coefficients
        self._nu: float = 1
        if nesterov_coeff is not None:
            self._nesterov_coeff: float = nesterov_coeff
            self._nesterov_coeff_auto = False
        else:
            self._nesterov_coeff_auto = True

        # Dual variable, its last iterate and its Nesterov shifted value
        self._dual_var: Optional[CPUArray] = None
        self._dual_var_old: Optional[CPUArray] = None
        self._dual_var_nesterov: Optional[CPUArray] = None

        # Primal variable
        self._X: Optional[CPUArray] = None
        # Pair variable, its last iterate and its Nesterov shifted value
        self._Z: Optional[CPUArray] = None
        self._Z_old: Optional[CPUArray] = None
        self._Z_nesterov: Optional[CPUArray] = None
    
    def initialize(self, X_start: CPUArray, Z_start: Optional[CPUArray] = None):
        """Initialize the algorithm and its dual variable iterates"""
        self._X = X_start
        if Z_start is not None:
            self._Z = Z_start
        else:
            self._Z = cpu_array(X_start)
        self._Z_nesterov = cpu_array(self._Z)
        self._dual_var = cpu_zeros((self._n,))
        self._dual_var_old = cpu_zeros((self._n,))
        self._dual_var_nesterov = cpu_zeros((self._n,))
    
    @property
    def mode(self) -> ADMMMode:
        return self._mode
    
    # These return the most recent iterate
    @property
    def dual_var(self) -> CPUArray:
        assert self._dual_var is not None
        return self._dual_var
    @property
    def X(self) -> CPUArray:
        assert self._X is not None
        return self._X
    @property
    def Z(self) -> CPUArray:
        assert self._Z is not None
        return self._Z

    @property
    def nesterov_coeff(self) -> float:
        return self._nesterov_coeff

    def _update_nesterov_coeff(self):
        if not self._nesterov_coeff_auto:
            assert self._nesterov_coeff is not None
        else:
            self._nu = (1 + math.sqrt(1 + 4 * math.pow(self._nu, 2))) / 2
            self._nesterov_coeff = (1 - self._nu) / (1 + self._nu)
    
    def _mul_A(self, thing: CPUArray, over_relax: bool) -> CPUArray:
        """
        - Get `A @ thing` when `over_relax` is `False`
        - Get `alpha * A @ thing` when `over_relax` is `True`
        - If `A` is `None`, then assume it is identity.
        """
        if over_relax and self._A is not None:
            return self._alpha * self._A @ thing
        elif over_relax and self._A is None:
            return self._alpha * thing
        elif not over_relax and self._A is not None:
            return self._A @ thing
        return thing

    def _mul_B(self, thing: CPUArray, over_relax: bool) -> CPUArray:
        """
        - Get `B @ thing` when `over_relax` is `False`
        - Get `(1 - alpha) * B @ thing` when `over_relax` is `True`
        - If `B` is `None`, then assume it is identity.
        """
        if over_relax and self._B is not None:
            return (1 - self._alpha) * self._B @ thing
        elif over_relax and self._B is None:
            return - (1 - self._alpha) * thing
        elif not over_relax and self._B is not None:
            return self._B @ thing
        return -thing
    
    def _add_C(self, thing: CPUArray, over_relax: bool) -> CPUArray:
        """
        - Get `C + thing` when `over_relax` is `False`
        - Get `alpha * C + thing` when `over_relax` is `True`
        - If `C` is `None`, then assume it is all zeros.
        """
        if self._C is not None:
            if over_relax:
                return self._alpha * self._C + thing
            return self._C + thing
        return thing
    
    def get_X_step_bias(self) -> CPUArray:
        """
        Return `b` such that the ADMM penalty is:
            || AX - b ||_2^2
        """
        if self._mode == ADMMMode.ACCELERATED or self._mode == ADMMMode.ACCELERATED_OVER_RELAXED:
            return self._add_C(- self._mul_B(self._Z_nesterov, False) - self._dual_var_nesterov, False)
        return self._add_C(- self._mul_B(self._Z, False) - self._dual_var, False)

    def get_Z_step_bias(self) -> CPUArray:
        """
        Return `b` such that the ADMM penalty is:
            || BZ - b ||_2^2
        """
        if self._mode == ADMMMode.VANILLA:
            return self._add_C(- self._mul_A(self._X, False) - self._dual_var, False)
        elif self._mode == ADMMMode.ACCELERATED:
            return self._add_C(- self._mul_A(self._X, False) - self._dual_var_nesterov, False)
        elif self._mode == ADMMMode.OVER_RELAXED:
            return self._add_C(- self._mul_A(self._X, True) + self._mul_B(self._Z, True) - self._dual_var, True)
        return self._add_C(- self._mul_A(self._X, True) + self._mul_B(self._Z_nesterov, True) - self._dual_var_nesterov, True)
    
    def record_X_update(self, next_X: CPUArray):
        """Record the X-update step output"""
        self._X = next_X
    
    def record_Z_update(self, next_Z: CPUArray):
        """Record the Z-update step output"""
        self._Z_old = cpu_array(self._Z)
        self._Z = next_Z
    
    def _dual_var_bias(self) -> CPUArray:
        if self._mode == ADMMMode.VANILLA:
            return self._add_C(- self._mul_A(self._X, False) - self._mul_B(self._Z, False) - self._dual_var, False)
        elif self._mode == ADMMMode.ACCELERATED:
            return self._add_C(- self._mul_A(self._X, False) - self._mul_B(self._Z, False) - self._dual_var_nesterov, False)
        elif self._mode == ADMMMode.OVER_RELAXED:
            return self._add_C(- self._mul_A(self._X, True) + self._mul_B(self._Z_old, True) - self._mul_B(self._Z, False) - self._dual_var, True)
        return self._add_C(- self._mul_A(self._X, True) + self._mul_B(self._Z_nesterov, True) - self._mul_B(self._Z, False) - self._dual_var_nesterov, True)

    def update_dual_var(self, finalize: bool = False):
        """
        Update the dual variable.
        Optionally, `finalize = True` will finalize the current ADMM round
        by updating the Nesterov coefficient and momentum shifts.
        """
        self.record_dual_update(-self._dual_var_bias())
        if finalize:
            self.finalize_round()
    
    def record_dual_update(self, next_dual_var: CPUArray):
        """Record the dual variable update step output"""
        if self._mode == ADMMMode.ACCELERATED or self._mode == ADMMMode.ACCELERATED_OVER_RELAXED:
            self._dual_var_old = cpu_array(self._dual_var)
        self._dual_var = next_dual_var
    
    def finalize_round(self):
        """
        Finalize the ADMM round.
        Only effects the case where acelerated ADMM is used.
        """
        if self._mode == ADMMMode.ACCELERATED or self._mode == ADMMMode.ACCELERATED_OVER_RELAXED:
            self._update_nesterov_coeff()
            self._Z_nesterov = self._Z + self.nesterov_coeff * (self._Z - self._Z_old)
            self._dual_var_nesterov = self._dual_var + self.nesterov_coeff * (self._dual_var - self._dual_var_old)
    
    def _get_primal_residual(self) -> CPUArray:
        return -self._add_C(- self._mul_A(self._X, False) - self._mul_B(self._Z, False), False)
    
    def _get_dual_residual(self) -> CPUArray:
        if self._A is None and self._B is None:
            return self._rho * (self._Z - self._Z_old)
        elif self._A is None and self._B is not None:
            return self._rho * self._B @ (self._Z - self._Z_old)
        elif self._A is not None and self._B is None:
            return self._rho * self._A.T @ (self._Z - self._Z_old)
        return self._rho * self._A.T @ self._B @ (self._Z - self._Z_old)
    
    def primal_infeasibility(self) -> float:
        r = self._get_primal_residual()
        return float(np.linalg.norm(r) / math.sqrt(len(r)))
    def dual_infeasibility(self) -> float:
        s = self._get_dual_residual()
        return float(np.linalg.norm(s) / math.sqrt(len(s)))
    
    @property
    def infeasibility(self) -> float:
        p_inf = self.primal_infeasibility()
        d_inf = self.dual_infeasibility()
        # print(f"======= P: {str(round(p_inf, 4))} \t D: {str(round(d_inf, 4))}")
        return p_inf + d_inf
        # return self.primal_infeasibility() + self.dual_infeasibility()


class SharingWrapper:
    """
    A specialization of `ADMMWrapper`, specifically for ADMM _sharing_ problems
    as defined in the cited sources.

    Citations
    ---------
    - Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). 
      Distributed optimization and statistical learning via the alternating direction method of multipliers. 
      Foundations and Trends® in Machine learning, 3(1), 1-122.
    """
    def __init__(self, n: int, rho: float, alpha: Optional[float] = None, nesterov: bool = False, 
                 nesterov_coeff: Optional[float] = None):
        # The ADMM mode is determined by the parameters
        if alpha is None:
            if nesterov is False:
                self._mode: ADMMMode = ADMMMode.VANILLA
            else:
                self._mode: ADMMMode = ADMMMode.ACCELERATED
        else:
            if nesterov is False:
                self._mode: ADMMMode = ADMMMode.OVER_RELAXED
            else:
                self._mode: ADMMMode = ADMMMode.ACCELERATED_OVER_RELAXED

        self._n = n
        self._rho = rho
        # If this is `None`, then it is interpreted as it being 1
        self._alpha = alpha
        # Nesterov momentum coefficients
        self._nu: float = 1
        if nesterov_coeff is not None:
            self._nesterov_coeff: float = nesterov_coeff
            self._nesterov_coeff_auto = False
        else:
            self._nesterov_coeff_auto = True

        # Dual variable, its last iterate and its Nesterov shifted value
        self._dual_var: Optional[CPUArray] = None
        self._dual_var_old: Optional[CPUArray] = None
        self._dual_var_nesterov: Optional[CPUArray] = None

        # Primal variable mean
        self._X: Optional[CPUArray] = None
        self._X_mean: Optional[CPUArray] = None
        # Pair variable mean, its last iterate and its Nesterov shifted value
        self._Z: Optional[CPUArray] = None
        self._Z_old: Optional[CPUArray] = None
        self._Z_nesterov: Optional[CPUArray] = None
        self._Z_mean: Optional[CPUArray] = None
        self._Z_mean_old: Optional[CPUArray] = None
        self._Z_mean_nesterov: Optional[CPUArray] = None
    
    def initialize(self, X: Optional[CPUArray] = None, Z: Optional[CPUArray] = None, 
                   X_mean: Optional[CPUArray] = None, Z_mean: Optional[CPUArray] = None):
        """
        Initialize the algorithm and its dual variable iterates.
        Pass at most _ONE_ of either the main iterates or the mean value, NOT both
        at the same time!
        """
        if X is not None:
            assert X_mean is None
            self._X = X
            self._X_mean = np.mean(X, axis=1)
        elif X_mean is not None:
            self._X_mean = X_mean
        
        if Z is not None:
            assert Z_mean is None
            self._Z = Z
            self._Z_mean = np.mean(Z, axis=1)
        elif X is not None:
            self._Z = cpu_array(X)
            self._Z_mean = cpu_array(self._X_mean)
        
        if self._Z is not None:
            self._Z_nesterov = cpu_array(self._Z)
        if self._Z_mean is not None:
            self._Z_mean_nesterov = cpu_array(self._Z_mean)
        
        self._dual_var = cpu_zeros((self._n,))
        self._dual_var_old = cpu_zeros((self._n,))
        self._dual_var_nesterov = cpu_zeros((self._n,))
    
    @property
    def mode(self) -> ADMMMode:
        return self._mode
    
    # These return the most recent iterate
    @property
    def dual_var(self) -> CPUArray:
        assert self._dual_var is not None
        return self._dual_var
    @property
    def X(self) -> CPUArray:
        assert self._X is not None
        return self._X
    @property
    def X_mean(self) -> CPUArray:
        assert self._X_mean is not None
        return self._X_mean
    @property
    def Z(self) -> CPUArray:
        assert self._Z is not None
        return self._Z
    @property
    def Z_mean(self) -> CPUArray:
        assert self._Z_mean is not None
        return self._Z_mean

    @property
    def nesterov_coeff(self) -> float:
        return self._nesterov_coeff

    def _update_nesterov_coeff(self):
        if not self._nesterov_coeff_auto:
            assert self._nesterov_coeff is not None
        else:
            self._nu = (1 + math.sqrt(1 + 4 * math.pow(self._nu, 2))) / 2
            self._nesterov_coeff = (1 - self._nu) / (1 + self._nu)
    
    def get_X_step_bias(self) -> CPUArray:
        """
        Return the array `b` such that the columns `b_k` give ADMM penalty as:
            sum_k || X_k - b_k ||_2^2
        """
        ALPHA = self._alpha
        if self._mode == ADMMMode.VANILLA:
            return self._X - \
                np.expand_dims(self._X_mean + self._Z_mean - self._dual_var, axis=1)
        elif self._mode == ADMMMode.ACCELERATED:
            return self._X - \
                np.expand_dims(self._X_mean + self._Z_mean - self._dual_var_nesterov, axis=1)
        elif self._mode == ADMMMode.OVER_RELAXED:
            return ALPHA * self._X + (1 - ALPHA) * self._Z - \
                np.expand_dims(ALPHA * self._X_mean + (2 - ALPHA) * self._Z_mean - self._dual_var, axis=1)
        return ALPHA * self._X + (1 - ALPHA) * self._Z - \
            np.expand_dims(ALPHA * self._X_mean + (1 - ALPHA) * self._Z_mean_nesterov + self._Z_mean - self._dual_var, axis=1)

    def get_Z_mean_step_bias(self) -> CPUArray:
        """
        Return `b` such that the ADMM penalty is:
            || Z_mean - b ||_2^2
        """
        ALPHA = self._alpha
        if self._mode == ADMMMode.VANILLA:
            return self._X_mean + self._dual_var
        elif self._mode == ADMMMode.ACCELERATED:
            return self._X_mean + self._dual_var_nesterov
        elif self._mode == ADMMMode.OVER_RELAXED:
            return ALPHA * self._X_mean + (1 - ALPHA) * self._Z_mean + self._dual_var
        return ALPHA * self._X_mean + (1 - ALPHA) * self._Z_mean_nesterov + self._dual_var_nesterov
    
    def record_X_update(self, next_X: CPUArray):
        """Record the X-update step output"""
        self._X = next_X
        self._X_mean = np.mean(next_X, axis=1)
    
    def record_Z_update(self, next_Z: CPUArray):
        """Record the Z-update step output"""
        self._Z_old = cpu_array(self._Z)
        self._Z_mean_old = cpu_array(self._Z_mean)
        self._Z = next_Z
        self._Z_mean = np.mean(next_Z, axis=1)

    def record_X_mean_update(self, next_X_mean: CPUArray):
        """Record the mean X-update step output"""
        self._X_mean_old = cpu_array(self._X_mean)
        self._X_mean = next_X_mean

    def record_Z_mean_update(self, next_Z_mean: CPUArray):
        """Record the mean Z-update step output"""
        self._Z_mean_old = cpu_array(self._Z_mean)
        self._Z_mean = next_Z_mean

    def update_dual_var(self, finalize: bool = False):
        """
        Update the dual variable.
        Optionally, `finalize = True` will finalize the current ADMM round
        by updating the Nesterov coefficient and momentum shifts.
        """

        ALPHA = self._alpha
        if self._mode == ADMMMode.VANILLA:
            next_dual_var = self._dual_var + self._X_mean - self._Z_mean
        elif self._mode == ADMMMode.ACCELERATED:
            next_dual_var = self._dual_var_nesterov + self._X_mean - self._Z_mean
        elif self._mode == ADMMMode.OVER_RELAXED:
            next_dual_var = self._dual_var + ALPHA * self._X_mean + (1 - ALPHA) - self._Z_mean
        else:
            next_dual_var = self._dual_var_nesterov + ALPHA * self._X_mean + (1 - ALPHA) * self._Z_mean_nesterov - self._Z_mean
    
        self.record_dual_update(next_dual_var)
        if finalize:
            self.finalize_round()
    
    def record_dual_update(self, next_dual_var: CPUArray):
        """Record the dual variable update step output"""
        if self._mode == ADMMMode.ACCELERATED or self._mode == ADMMMode.ACCELERATED_OVER_RELAXED:
            self._dual_var_old = cpu_array(self._dual_var)
        self._dual_var = next_dual_var
    
    def finalize_round(self):
        """
        Finalize the ADMM round.
        Only effects the case where acelerated ADMM is used.
        """
        if self._mode == ADMMMode.ACCELERATED or self._mode == ADMMMode.ACCELERATED_OVER_RELAXED:
            self._update_nesterov_coeff()
            if self._Z is not None:
                self._Z_nesterov = self._Z + self.nesterov_coeff * (self._Z - self._Z_old)
                self._Z_mean_nesterov = np.mean(self._Z_nesterov, axis=1)
            else:
                self._Z_mean_nesterov = self._Z_mean + self.nesterov_coeff * (self._Z_mean - self._Z_mean_old)
            self._dual_var_nesterov = self._dual_var + self.nesterov_coeff * (self._dual_var - self._dual_var_old)
    
    def _get_primal_residual(self) -> CPUArray:
        return self._X - self._Z
    
    def _get_dual_residual(self) -> CPUArray:
        return self._rho * (self._Z - self._Z_old)
    
    def primal_infeasibility(self) -> float:
        r = self._get_primal_residual()
        return float(np.linalg.norm(r) / math.sqrt(r.size))
    def dual_infeasibility(self) -> float:
        s = self._get_dual_residual()
        return float(np.linalg.norm(s) / math.sqrt(s.size))
    
    @property
    def infeasibility(self) -> float:
        p_inf = self.primal_infeasibility()
        d_inf = self.dual_infeasibility()
        # print(f"======= P: {str(round(p_inf, 4))} \t D: {str(round(d_inf, 4))}")
        return p_inf + d_inf
        # return self.primal_infeasibility() + self.dual_infeasibility()