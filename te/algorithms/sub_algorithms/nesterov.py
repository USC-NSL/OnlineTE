import numpy as np
from typing import Optional, Callable, Iterator
from array_utils.cpu.types import CPUArray, cpu_array, cpu_cast_float


def default_nesterov_step_size() -> Iterator[float]:
    nu = 1
    while True:
        next_nu = cpu_cast_float((1 + (1 + 4 * nu**2)**0.5) / 2)
        yield (nu - 1) / next_nu
        nu = next_nu


class NesterovExtrapolationWrapper:
    """
    A trivial wrapper for nesterov momentum extrapolation.
    The starting state array can be modified in-place or otherwise
    completely changed.

    Mode Of Operation
    -----------------
    Let `S_{k}` be the state variable being tracked. We work with `S_{k-1}` and
    `E_{k}`, where `S_{k-1}` is the old state and `E_{k}` is the extrapolated state
    that is a function of `S_{k}` and `S_{k-1}`.

    Let the updated state be `S_{k}`, then:
    - At the time of the update, the wrapper maintains `S_{k-1}` and `E_{k-1}`.
      In this state, the old state is correct but `E_{k-1}` is stale.
    - After the update, the wrapper maintains `S_{k}` and `E_{k}`, the old
      state is no longer here, but extrapolated state is up to date.
    """
    def __init__(self, starting_state: CPUArray, steps: Optional[Callable[[], Iterator[float]]] = None):
        self._previous_state = cpu_array(starting_state)
        self._step_gen: Callable[[], Iterator[float]] = steps if steps is not None else default_nesterov_step_size
        self._steps: Iterator[float] = self._step_gen()
        self._extrapolated_state: CPUArray = cpu_array(starting_state)

    @property
    def extrapolated_state(self) -> CPUArray:
        return self._extrapolated_state

    @property
    def old_state(self) -> CPUArray:
        return self._previous_state

    def restart(self, new_state: CPUArray):
        """
        Forget all previous momentum and start from a given state.
        If no new state has been given, just use the current state.
        """
        np.copyto(self._previous_state, new_state)
        np.copyto(self._extrapolated_state, new_state)
        self._steps = self._step_gen()

    def rescale(self, factor: float):
        """Rescale every tracked state without resetting the momentum sequence."""
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Nesterov state scale factor must be positive and finite")
        self._previous_state *= factor
        self._extrapolated_state *= factor

    def update(self, new_state: CPUArray):
        """Update the extrapolated state."""
        ext = self._extrapolated_state
        prev = self._previous_state
        step_size = next(self._steps)
        np.subtract(new_state, prev, out=ext)
        ext *= step_size
        ext += new_state
        # Copy the new state, into the previous state for next iteration
        np.copyto(prev, new_state)
