"""Reproducible hot-loop benchmark for packed path-QP kernels."""

import argparse
import statistics
import time

import numpy as np
from numba import set_num_threads
from numba.typed import List as NumbaList

from array_utils import SINGLE_PRECISION, set_global_precision
from te.algorithms.formulations.path_based.distributed.packed_paths import PackedPathBatch
from te.algorithms.sub_algorithms.pgd import (
    do_packed_path_based_nesterov_pgd,
    do_path_based_nesterov_pgd,
)
from te.path_providers import PathProvider


def make_provider(
    commodity_count: int,
    num_edges: int,
    max_paths: int,
    path_length: int,
    seed: int,
) -> PathProvider:
    rng = np.random.default_rng(seed)
    betas = 1 + np.arange(commodity_count, dtype=np.int32) % max_paths
    rows = []
    cols = []
    for beta in betas:
        commodity_rows = np.empty(int(beta) * path_length, dtype=np.int32)
        commodity_cols = np.repeat(np.arange(beta, dtype=np.int32), path_length)
        for path in range(int(beta)):
            start = path * path_length
            commodity_rows[start:start + path_length] = rng.choice(
                num_edges, size=path_length, replace=False
            )
        rows.append(commodity_rows)
        cols.append(commodity_cols)
    return PathProvider(
        shape=(commodity_count, num_edges, max_paths),
        rows=rows,
        cols=cols,
        beta=betas,
    )


def packed_to_padded(values: np.ndarray, batch: PackedPathBatch, max_paths: int) -> np.ndarray:
    output = np.zeros((max_paths, batch.num_commodities), dtype=values.dtype)
    for k in range(batch.num_commodities):
        start, end = batch.commodity_path_offsets[k:k + 2]
        output[:end - start, k] = values[start:end]
    return output


def padded_to_packed(values: np.ndarray, batch: PackedPathBatch) -> np.ndarray:
    output = np.empty(batch.num_paths, dtype=values.dtype)
    for k in range(batch.num_commodities):
        start, end = batch.commodity_path_offsets[k:k + 2]
        output[start:end] = values[:end - start, k]
    return output


def median_runtime(function, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commodities", type=int, default=4096)
    parser.add_argument("--edges", type=int, default=1800)
    parser.add_argument("--paths", type=int, default=16)
    parser.add_argument("--path-length", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--minimum-speedup", type=float, default=2.0)
    args = parser.parse_args()

    set_global_precision(SINGLE_PRECISION)
    set_num_threads(args.threads)
    dtype = np.dtype(np.float32)
    provider = make_provider(
        args.commodities, args.edges, args.paths, args.path_length, args.seed
    )
    capacities = np.ones(args.edges, dtype=dtype)

    setup_start = time.perf_counter()
    batch = PackedPathBatch.from_path_provider(
        provider, capacities, False, kernel_threads=args.threads
    )
    setup_time = time.perf_counter() - setup_start

    rng = np.random.default_rng(args.seed + 1)
    y = np.empty(batch.num_paths, dtype=dtype)
    for k in range(batch.num_commodities):
        start, end = batch.commodity_path_offsets[k:k + 2]
        y[start:end] = dtype.type(1 / (end - start))
    y_old = y + rng.normal(0, 1e-3, size=batch.num_paths).astype(dtype)
    demands = rng.uniform(0.5, 2.0, size=args.commodities).astype(dtype)
    sharing_bias = rng.normal(0, 0.1, size=args.edges).astype(dtype)
    steps = np.full(args.commodities, 0.01, dtype=dtype)
    linear = batch.linear_term(sharing_bias, demands)

    padded_y = packed_to_padded(y, batch, args.paths)
    padded_old = packed_to_padded(y_old, batch, args.paths)
    rows = NumbaList(provider.rows)
    cols = NumbaList(provider.cols)

    def legacy_update():
        return do_path_based_nesterov_pgd(
            padded_y.copy(), padded_old, rows, cols, sharing_bias,
            provider.beta, demands, args.edges, args.paths, steps,
            args.iterations, None,
        )

    def packed_update():
        return do_packed_path_based_nesterov_pgd(
            y.copy(), y_old, batch, linear, demands, steps, args.iterations
        )

    # Warm both the Numba specialization and native code paths.
    legacy_result = legacy_update()
    packed_result = packed_update()
    np.testing.assert_allclose(
        packed_result,
        padded_to_packed(legacy_result, batch),
        rtol=2e-4,
        atol=2e-5,
    )

    legacy_seconds = median_runtime(legacy_update, args.repeats)
    packed_seconds = median_runtime(packed_update, args.repeats)
    linear_seconds = median_runtime(
        lambda: batch.linear_term(sharing_bias, demands), args.repeats
    )
    report_seconds = median_runtime(
        lambda: batch.paths_to_edge_mean(y, demands), args.repeats
    )
    speedup = legacy_seconds / packed_seconds

    padded_state_bytes = padded_y.nbytes
    packed_state_bytes = y.nbytes
    gram_bytes = batch.gram_values.nbytes
    print(f"commodities={args.commodities} paths<={args.paths}")
    print(f"threads={args.threads} pgd_iterations={args.iterations}")
    print(f"native_setup_seconds={setup_time:.6f}")
    print(f"native_linear_seconds={linear_seconds:.6f}")
    print(f"native_report_mean_seconds={report_seconds:.6f}")
    print(f"legacy_hot_loop_seconds={legacy_seconds:.6f}")
    print(f"packed_hot_loop_seconds={packed_seconds:.6f}")
    print(f"hot_loop_speedup={speedup:.3f}x")
    print(f"padded_state_bytes={padded_state_bytes}")
    print(f"packed_state_bytes={packed_state_bytes}")
    print(f"packed_gram_bytes={gram_bytes}")

    if speedup < args.minimum_speedup:
        raise SystemExit(
            f"speedup {speedup:.3f}x is below required {args.minimum_speedup:.3f}x"
        )


if __name__ == "__main__":
    main()
