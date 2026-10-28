# Network Path Providers

This defines a unified provider for working with, manipulating and calculating things over network paths.

- `base.py` defines the main API of this module, the `PathProvider` dataclass and the `build_provider` function. These are the means by which are path-based solvers store path inputs. `PathProvider` maintains path information as a sparse boolean array $\alpha^{(k)}_{et}$, where for commodity index $k$, if path index $t$ crosses edge index $e$, it records a `True` value. This is generic enough to describe any path configuration.

- `packed_paths.py` defines the formatting for actually computing with paths. The main trouble that this is trying to solve, is that fact that not every commodity will have the same number of paths available to it, and hence the split matrix $Y_{tk}$ is a ragged array. We handle operations on this array with a native C++ backend and this file provides the Python API for calling those functions.

- `schemes.py` defines path generation schemes. We currently only support shortest path and edge-disjoint path iteration.
- `sparse_ops.py` is a legacy implementation of sparse matrix operations directly on `PathProvider` objects (as opposed to using `PackedPathBatch` as it is defined in `packed_paths.py`). This implementation uses `Numba` to JIT compile Python (as opposed to using C++ kernels). This exists merely to support some tests and `packed_paths.py` is a better implementation.