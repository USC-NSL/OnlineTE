import time
import numpy as np
from numba import set_num_threads
from te.algorithms.array_utils import set_global_precision, SINGLE_PRECISION
from te.algorithms.array_utils.cpu_utils import cpu_array, set_cpu_float_precision
from topologies.utils import load_zoo_topology
from te.algorithms.sub_algorithms.paths import *
from .worker import DenseSolver


def jit_tests():
    set_num_threads(2)
    set_global_precision(SINGLE_PRECISION)
    set_cpu_float_precision()
    topo_name = 'Kdl'
    # SEED = 5932763
    SEED = None
    T = 8
    rng = np.random.default_rng(SEED)
    g = load_zoo_topology(topo_name)
    obj = get_or_make_path_object_for_topology_name(
        topo_name=topo_name, T=T, edge_disjoint=False
    )
    _, N, T = obj.alpha.shape
    K = (g.number_of_nodes() - 1) * 3
    Y = cpu_array(rng.random(size=(T, K)))
    D = cpu_array(rng.random(size=(K,)))
    ALPHA_DENSE = obj.alpha.as_array(k_end=K)
    C = cpu_array(np.ones((N,)))
    X_dense = path_based_to_edge_based_dense(Y, ALPHA_DENSE, D)
    X_mean_dense = np.mean(X_dense, axis=1)
    X_proj_dense = path_based_projection_dense(Y, ALPHA_DENSE, D)
    from numba.typed import List
    rows = List(obj.alpha.rows[:K])
    cols = List(obj.alpha.cols[:K])
    beta = obj.beta[:K]

    print("JIT warm start ...")
    warm_start_jit(rows, cols, (K, N, T), beta)
    print("JIT done!")

    t_start = time.perf_counter()
    for _ in range(100):
        X_nnz = path_based_to_edge_based_nnz(Y, rows, cols, N, D)
    print(f"NNZ took: {(time.perf_counter() - t_start) * 1000 / 100} ms")
    print(f"Err: {np.max(np.abs(X_nnz - X_dense))}")

    t_start = time.perf_counter()
    for _ in range(100):
        X_mean_nnz = path_based_to_edge_based_mean_nnz(Y, rows, cols, N, D)
    print(f"NNZ Mean took: {(time.perf_counter() - t_start) * 1000 / 100} ms")
    print(f"Err: {np.max(np.abs(X_mean_nnz - X_mean_dense))}")

    t_start = time.perf_counter()
    for _ in range(100):
        X_proj_nnz = path_based_projection_nnz(Y, rows, cols, N, D)
    print(f"NNZ proj took: {(time.perf_counter() - t_start) * 1000 / 100} ms")
    print(f"Err: {np.max(np.abs(X_proj_nnz - X_proj_dense))}")

    t_start = time.perf_counter()
    for _ in range(100):
        X_proj_indirect_nnz = path_based_transpose_product_nnz(X_nnz, rows, cols, 8, D)
    print(f"NNZ indirect proj took: {(time.perf_counter() - t_start) * 1000 / 100} ms")
    print(f"Err: {np.max(np.abs(X_proj_indirect_nnz - X_proj_dense))}")

    t_start = time.perf_counter()
    for _ in range(100):
        total_flow = get_initial_total_flow_nnz(rows, beta, (K, N, T), D)
    print(f"Total flow took: {(time.perf_counter() - t_start) * 1000 / 100} ms")

    t_start = time.perf_counter()
    for _ in range(100):
        eigen_ups = path_based_eigen_upper_nnz(cols, T)
    print(f"Eigen up took: {(time.perf_counter() - t_start) * 1000 / 100} ms")
    print(f"MIN: {eigen_ups.min()} | MAX: {eigen_ups.max()}")

    t_start = time.perf_counter()
    for _ in range(100):
        X_proj_nnz = path_based_projection_nnz(Y, rows, cols, N, D, C)
    print(f"NNZ (capped) proj took: {(time.perf_counter() - t_start) * 1000 / 100} ms")
    print(f"Err: {np.max(np.abs(X_proj_nnz - X_proj_dense))}")

    t_start = time.perf_counter()
    for _ in range(100):
        eigs = path_based_power_method(rows, cols, (K, N, T))
    print(f"Eigens took: {(time.perf_counter() - t_start) * 1000 / 100} ms")
    print(f"lambda_10: {eigs[10]}")


if __name__ == '__main__':
    # jit_tests()
    # make solver
    set_num_threads(2)
    set_global_precision(SINGLE_PRECISION)
    set_cpu_float_precision()
    # topo_name = 'Kdl'
    topo_name = 'Cogentco'
    # topo_name = 'Colt'
    # topo_name = 'TataNld'
    # topo_name = 'Interoute'
    # topo_name = ''
    # SEED = 5932763
    SEED = 12345
    RNG = np.random.default_rng(SEED+1)
    T = 16
    rng = np.random.default_rng(SEED)
    g = load_zoo_topology(topo_name)
    obj = get_or_make_path_object_for_topology_name(
        topo_name=topo_name, T=T, edge_disjoint=False
    )
    _, N, T = obj.alpha.shape
    K = (g.number_of_nodes())
    Y = cpu_array(rng.random(size=(T, K)))
    D = cpu_array(rng.random(size=(K,)))
    ALPHA_DENSE = obj.alpha.as_array(k_end=K)
    C = cpu_array(np.ones((N,)))
    from numba.typed import List
    rows = List(obj.alpha.rows[:K])
    cols = List(obj.alpha.cols[:K])
    beta = obj.beta[:K]
    solver = DenseSolver(
        alpha_shape=(K, N, T), alpha_cols=cols, alpha_rows=rows, beta=beta,
        demands=D, pgd_step=1.0, pgd_iters=2, eta=0.1, adjust_step_size=True
    )
    NUM=30
    warm_start_jit(rows, cols, (K, N, T), beta)
    print("WARMED!")
    BIAS = cpu_array(RNG.random(size=(N,)))
    def test():
        start = time.perf_counter_ns()
        mean = solver.update(BIAS)
        return (time.perf_counter_ns() - start) // 1e3
    
    times = []
    for i in range(NUM):
        t = test()
        # t = test_f(solver, X_f=X_F, bias_f=BIAS_F)
        # t = test_2(solver, X=X, bias=BIAS)
        if i > 0:
            times.append(t)
    print(times)
    mean = np.mean(times)
    print(f"Mean Per Iter: {mean} ms")
