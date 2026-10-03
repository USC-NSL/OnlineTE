from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from te.path_providers import PathProvider


_NATIVE_IMPORT_ERROR: Optional[ImportError] = None
try:
    from . import _path_kernels as _native
except ImportError as exc:  # Keep unrelated solver imports working.
    _native = None
    _NATIVE_IMPORT_ERROR = exc


def require_native_kernels():
    if _native is None:
        raise RuntimeError(
            "The distributed path solver requires the native path kernels. "
            "Install a C++ compiler with OpenMP support and run "
            "`.onlinete-venv\\Scripts\\python.exe -m pip install -e .` "
            "(or `python -m pip install -e .` on Linux)."
        ) from _NATIVE_IMPORT_ERROR
    return _native


def resolve_kernel_threads(requested: Optional[int]) -> int:
    if requested is not None:
        if requested <= 0:
            raise ValueError("KernelThreads must be a positive integer or None")
        return requested
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(1, min(4, available))


def _require_float_array(array: np.ndarray, name: str, length: int, dtype: np.dtype) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must have shape ({length},)")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")
    if not array.flags.c_contiguous or not array.dtype.isnative:
        raise ValueError(f"{name} must be native-endian and C-contiguous")
    return array


def _typed_kernel(kernels, name: str, dtype: np.dtype):
    return getattr(kernels, f"{name}_{dtype.name}")


@dataclass(frozen=True)
class PackedPathBatch:
    """Contiguous, unpadded representation of a worker's assigned paths."""

    num_edges: int
    commodity_path_offsets: np.ndarray
    path_edge_offsets: np.ndarray
    path_edges: np.ndarray
    gram_offsets: np.ndarray
    gram_values: np.ndarray
    edge_scale: np.ndarray
    kernel_threads: int

    def __post_init__(self):
        if self.num_edges <= 0:
            raise ValueError("num_edges must be positive")
        if self.kernel_threads <= 0:
            raise ValueError("kernel_threads must be positive")
        for name, array, dtype in (
            ("commodity_path_offsets", self.commodity_path_offsets, np.dtype(np.int64)),
            ("path_edge_offsets", self.path_edge_offsets, np.dtype(np.int64)),
            ("path_edges", self.path_edges, np.dtype(np.int32)),
            ("gram_offsets", self.gram_offsets, np.dtype(np.int64)),
        ):
            if array.ndim != 1 or array.dtype != dtype or not array.flags.c_contiguous:
                raise TypeError(f"{name} must be a contiguous 1-D {dtype} array")
        if self.gram_values.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("gram_values must use float32 or float64")
        if self.edge_scale.dtype != self.gram_values.dtype:
            raise TypeError("edge_scale and gram_values must have the same dtype")
        if len(self.commodity_path_offsets) < 2:
            raise ValueError("at least one commodity is required")
        if self.commodity_path_offsets[0] != 0 or np.any(np.diff(self.commodity_path_offsets) <= 0):
            raise ValueError("every commodity must own at least one path")
        if self.path_edge_offsets[0] != 0 or np.any(np.diff(self.path_edge_offsets) <= 0):
            raise ValueError("every active path must contain at least one edge")
        if self.gram_offsets[0] != 0 or np.any(np.diff(self.gram_offsets) <= 0):
            raise ValueError("gram offsets must be strictly increasing from zero")
        if self.commodity_path_offsets[-1] + 1 != len(self.path_edge_offsets):
            raise ValueError("path offsets do not match the packed path count")
        if self.path_edge_offsets[-1] != len(self.path_edges):
            raise ValueError("edge offsets do not match the packed edge count")
        if self.gram_offsets[-1] != len(self.gram_values):
            raise ValueError("gram offsets do not match the packed Gram values")
        if len(self.edge_scale) != self.num_edges:
            raise ValueError("edge_scale length does not match num_edges")
        if np.any(self.path_edges < 0) or np.any(self.path_edges >= self.num_edges):
            raise ValueError("path edge index is outside the topology")

        betas = np.diff(self.commodity_path_offsets)
        if not np.array_equal(np.diff(self.gram_offsets), betas * betas):
            raise ValueError("each packed Gram block must contain beta_k squared values")

    @property
    def dtype(self) -> np.dtype:
        return self.gram_values.dtype

    @property
    def num_commodities(self) -> int:
        return len(self.commodity_path_offsets) - 1

    @property
    def num_paths(self) -> int:
        return int(self.commodity_path_offsets[-1])

    @property
    def path_counts(self) -> np.ndarray:
        return np.diff(self.commodity_path_offsets)

    @classmethod
    def from_path_provider(
        cls,
        provider: "PathProvider",
        capacities: np.ndarray,
        scale_with_capacity: bool,
        kernel_threads: Optional[int] = None,
    ) -> "PackedPathBatch":
        kernels = require_native_kernels()
        if capacities.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("The native path solver supports only float32 and float64")
        if capacities.ndim != 1 or not capacities.flags.c_contiguous:
            raise ValueError("capacities must be a contiguous 1-D array")

        commodity_count, num_edges, _ = provider.shape
        if commodity_count != len(provider.rows) or commodity_count != len(provider.cols):
            raise ValueError("PathProvider commodity metadata is inconsistent")
        if num_edges != len(capacities):
            raise ValueError("PathProvider edge count does not match capacities")
        if scale_with_capacity and (
            not np.all(np.isfinite(capacities)) or np.any(capacities <= 0)
        ):
            raise ValueError(
                "capacity scaling requires finite, strictly positive capacities"
            )

        betas = np.asarray(provider.beta, dtype=np.int64)
        if betas.shape != (commodity_count,) or np.any(betas <= 0):
            raise ValueError("every assigned commodity must have at least one path")

        commodity_offsets = np.empty(commodity_count + 1, dtype=np.int64)
        commodity_offsets[0] = 0
        np.cumsum(betas, out=commodity_offsets[1:])
        packed_path_count = int(commodity_offsets[-1])

        total_edges = sum(len(rows) for rows in provider.rows)
        path_edge_offsets = np.empty(packed_path_count + 1, dtype=np.int64)
        path_edge_offsets[0] = 0
        path_edges = np.empty(total_edges, dtype=np.int32)
        path_cursor = 0
        edge_cursor = 0

        for k in range(commodity_count):
            rows = np.asarray(provider.rows[k], dtype=np.int32)
            cols = np.asarray(provider.cols[k], dtype=np.int32)
            beta = int(betas[k])
            if rows.ndim != 1 or cols.ndim != 1 or len(rows) != len(cols):
                raise ValueError(f"invalid COO path data for commodity {k}")
            if np.any(cols < 0) or np.any(cols >= beta):
                raise ValueError(f"path index outside beta for commodity {k}")
            for local_path in range(beta):
                edges = rows[cols == local_path]
                if len(edges) == 0:
                    raise ValueError(f"commodity {k}, path {local_path} is empty")
                next_edge_cursor = edge_cursor + len(edges)
                path_edges[edge_cursor:next_edge_cursor] = edges
                edge_cursor = next_edge_cursor
                path_cursor += 1
                path_edge_offsets[path_cursor] = edge_cursor

        if edge_cursor != total_edges or path_cursor != packed_path_count:
            raise RuntimeError("failed to pack all PathProvider entries")

        gram_offsets = np.empty(commodity_count + 1, dtype=np.int64)
        gram_offsets[0] = 0
        np.cumsum(betas * betas, out=gram_offsets[1:])
        gram_values = np.empty(int(gram_offsets[-1]), dtype=capacities.dtype)
        if scale_with_capacity:
            edge_scale = np.reciprocal(capacities)
        else:
            edge_scale = np.ones(num_edges, dtype=capacities.dtype)
        edge_scale = np.ascontiguousarray(edge_scale)
        threads = resolve_kernel_threads(kernel_threads)

        _typed_kernel(kernels, "build_gram", capacities.dtype)(
            commodity_offsets,
            path_edge_offsets,
            path_edges,
            edge_scale,
            gram_offsets,
            gram_values,
            threads,
        )
        return cls(
            num_edges=num_edges,
            commodity_path_offsets=commodity_offsets,
            path_edge_offsets=path_edge_offsets,
            path_edges=path_edges,
            gram_offsets=gram_offsets,
            gram_values=gram_values,
            edge_scale=edge_scale,
            kernel_threads=threads,
        )

    def linear_term(
        self,
        sharing_bias: np.ndarray,
        demands: np.ndarray,
        maxflow_shift: float = 0.0,
    ) -> np.ndarray:
        kernels = require_native_kernels()
        _require_float_array(sharing_bias, "sharing_bias", self.num_edges, self.dtype)
        _require_float_array(demands, "demands", self.num_commodities, self.dtype)
        output = np.empty(self.num_paths, dtype=self.dtype)
        _typed_kernel(kernels, "linear_term", self.dtype)(
            self.commodity_path_offsets,
            self.path_edge_offsets,
            self.path_edges,
            self.edge_scale,
            sharing_bias,
            demands,
            self.dtype.type(maxflow_shift),
            output,
            self.kernel_threads,
        )
        return output

    def block_gram_matvec(self, values: np.ndarray, demands: np.ndarray) -> np.ndarray:
        kernels = require_native_kernels()
        _require_float_array(values, "values", self.num_paths, self.dtype)
        _require_float_array(demands, "demands", self.num_commodities, self.dtype)
        output = np.empty_like(values)
        _typed_kernel(kernels, "block_gram_matvec", self.dtype)(
            self.commodity_path_offsets,
            self.gram_offsets,
            self.gram_values,
            values,
            demands,
            output,
            self.kernel_threads,
        )
        return output

    def projected_qp_step(
        self,
        z: np.ndarray,
        y_old: np.ndarray,
        linear: np.ndarray,
        demands: np.ndarray,
        step_sizes: np.ndarray,
        orthant: bool,
        output: np.ndarray,
    ) -> np.ndarray:
        kernels = require_native_kernels()
        for name, array in (("z", z), ("y_old", y_old), ("linear", linear), ("output", output)):
            _require_float_array(array, name, self.num_paths, self.dtype)
        _require_float_array(demands, "demands", self.num_commodities, self.dtype)
        _require_float_array(step_sizes, "step_sizes", self.num_commodities, self.dtype)
        relative_gaps = np.empty(self.num_commodities, dtype=np.float64)
        _typed_kernel(kernels, "projected_qp_step", self.dtype)(
            self.commodity_path_offsets,
            self.gram_offsets,
            self.gram_values,
            z,
            y_old,
            linear,
            demands,
            step_sizes,
            orthant,
            output,
            relative_gaps,
            self.kernel_threads,
        )
        # if not np.all(np.isfinite(relative_gaps)):
        #     raise FloatingPointError("Projected QP step produced a non-finite Wolfe gap")
        return relative_gaps

    def estimate_lipschitz(self, demands: np.ndarray, iterations: int = 20) -> np.ndarray:
        kernels = require_native_kernels()
        _require_float_array(demands, "demands", self.num_commodities, self.dtype)
        output = np.empty(self.num_commodities, dtype=self.dtype)
        _typed_kernel(kernels, "estimate_lipschitz", self.dtype)(
            self.commodity_path_offsets,
            self.gram_offsets,
            self.gram_values,
            demands,
            iterations,
            output,
            self.kernel_threads,
        )
        return output

    def initialize_splits(self, capacities: np.ndarray) -> np.ndarray:
        kernels = require_native_kernels()
        _require_float_array(capacities, "capacities", self.num_edges, self.dtype)
        output = np.empty(self.num_paths, dtype=self.dtype)
        _typed_kernel(kernels, "initialize_splits", self.dtype)(
            self.commodity_path_offsets,
            self.path_edge_offsets,
            self.path_edges,
            capacities,
            output,
            self.kernel_threads,
        )
        return output

    def paths_to_edge(self, values: np.ndarray, demands: np.ndarray) -> np.ndarray:
        kernels = require_native_kernels()
        _require_float_array(values, "values", self.num_paths, self.dtype)
        _require_float_array(demands, "demands", self.num_commodities, self.dtype)
        output = np.zeros((self.num_edges, self.num_commodities), dtype=self.dtype)
        _typed_kernel(kernels, "paths_to_edge", self.dtype)(
            self.commodity_path_offsets,
            self.path_edge_offsets,
            self.path_edges,
            self.edge_scale,
            values,
            demands,
            output,
            self.kernel_threads,
        )
        return output

    def paths_to_edge_mean(self, values: np.ndarray, demands: np.ndarray) -> np.ndarray:
        kernels = require_native_kernels()
        _require_float_array(values, "values", self.num_paths, self.dtype)
        _require_float_array(demands, "demands", self.num_commodities, self.dtype)
        output = np.zeros(self.num_edges, dtype=self.dtype)
        _typed_kernel(kernels, "paths_to_edge_mean", self.dtype)(
            self.commodity_path_offsets,
            self.path_edge_offsets,
            self.path_edges,
            self.edge_scale,
            values,
            demands,
            output,
            self.kernel_threads,
        )
        return output

    def total_flow(self, values: np.ndarray, demands: np.ndarray) -> float:
        kernels = require_native_kernels()
        _require_float_array(values, "values", self.num_paths, self.dtype)
        _require_float_array(demands, "demands", self.num_commodities, self.dtype)
        return float(
            _typed_kernel(kernels, "total_flow", self.dtype)(
                self.commodity_path_offsets,
                values,
                demands,
                self.kernel_threads,
            )
        )

    def assignment_delta_norm_squared(
        self,
        current: np.ndarray,
        previous: np.ndarray,
        demands: np.ndarray,
    ) -> float:
        """Return the local squared Frobenius norm of an assignment update."""
        kernels = require_native_kernels()
        _require_float_array(current, "current", self.num_paths, self.dtype)
        _require_float_array(previous, "previous", self.num_paths, self.dtype)
        _require_float_array(demands, "demands", self.num_commodities, self.dtype)
        return float(
            _typed_kernel(kernels, "assignment_delta_norm_squared", self.dtype)(
                self.commodity_path_offsets,
                self.gram_offsets,
                self.gram_values,
                current,
                previous,
                demands,
                self.kernel_threads,
            )
        )

    def dual_objective_term(
        self,
        edge_duals: np.ndarray,
        demands: np.ndarray,
    ) -> float:
        """Return this batch's path-minimization term in the dual objective."""
        kernels = require_native_kernels()
        _require_float_array(edge_duals, "edge_duals", self.num_edges, self.dtype)
        _require_float_array(demands, "demands", self.num_commodities, self.dtype)
        return float(
            _typed_kernel(kernels, "dual_objective_term", self.dtype)(
                self.commodity_path_offsets,
                self.path_edge_offsets,
                self.path_edges,
                self.edge_scale,
                edge_duals,
                demands,
                self.kernel_threads,
            )
        )


__all__ = ["PackedPathBatch", "require_native_kernels", "resolve_kernel_threads"]
