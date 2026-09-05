import time
import numpy as np
from typing import Union
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.cpu_utils import CPUArray, BooleanCPUArray, CPUCSRArray, CPUCSCArray, cpu_zeros, cpu_array, set_cpu_float_precision
# from te.algorithms.sub_algorithms.pgd import do_memory_efficient_pgd, do_memory_efficient_pgd_2
from te.algorithms.sub_algorithms.pgd import do_memory_efficient_pgd
from te.algorithms.sub_algorithms.feasible_assignment import InitialSolutionType


class DenseSolver:
    def __init__(self, X_0: Union[CPUCSRArray, CPUCSCArray, CPUArray], NNT: CPUArray, mask: BooleanCPUArray, 
                 pgd_step: float, pgd_iters: int):
        self._X_0 = X_0
        self._NNT = NNT
        self._mask = mask
        self._lambda_ek = cpu_zeros(X_0.shape)
        self._X_ek = cpu_array(X_0)
        self._pgd_step = pgd_step
        self._pgd_iters = pgd_iters
        self._buffer = np.empty_like(X_0)

    def _get_current_C(self, sharing_bias: CPUArray) -> CPUArray:
        return self._NNT @ (self._X_ek - self._X_0 - np.expand_dims(sharing_bias, axis=1)) + self._X_0

    @property
    def X_ek(self) -> CPUArray:
        return self._X_ek

    def set_X_ek(self, X_ek: CPUArray):
        self._X_ek = X_ek

    def update(self, sharing_bias: CPUArray, c_block: CPUArray) -> CPUArray:
        self._lambda_ek = do_memory_efficient_pgd(
            lambda_block=self._lambda_ek, 
            c_block=c_block,
            nnt=self._NNT,
            step_size=self._pgd_step, 
            n_iter=self._pgd_iters, 
            mask=self._mask,
            buffer=self._buffer
        )
        self._X_ek += self._NNT @ (self._lambda_ek - np.expand_dims(sharing_bias, axis=1))
        return self._X_ek

    def update_minimum(self, bias: CPUArray) -> CPUArray:
        self._lambda_ek = do_memory_efficient_pgd(
            lambda_block=self._lambda_ek, 
            x_block=self._X_ek,
            nnt=self._NNT,
            bias=bias,
            x_block_0=self._X_0,
            step_size=self._pgd_step, 
            n_iter=self._pgd_iters, 
            mask=self._mask
        )
        # do_memory_efficient_pgd(
        #     lambda_block=self._lambda_ek, 
        #     c_block=c_block,
        #     nnt=self._NNT,
        #     step_size=self._pgd_step, 
        #     n_iter=self._pgd_iters, 
        #     mask=self._mask
        # )
        return self._X_ek

    # def update_minimum_2(self, bias: CPUArray) -> CPUArray:
    #     do_memory_efficient_pgd_2(
    #         lambda_block=self._lambda_ek, 
    #         x_block=self._X_ek,
    #         nnt=self._NNT,
    #         bias=bias,
    #         x_block_0=self._X_0,
    #         step_size=self._pgd_step, 
    #         n_iter=self._pgd_iters,
    #         mask=self._mask
    #     )
    #     return self._X_ek


if __name__ == '__main__':
    set_global_precision('single')
    set_cpu_float_precision()
    from te.algorithms.sub_algorithms.feasible_assignment import get_feasible_flow_assignment, InitialSolutionType
    from topologies.utils import load_zoo_topology, get_adjacency_null_space, get_graph_M_matrix, get_commodity_in_out_mask, get_edge_indexing
    from te.traffic_models.base import traffic_to_commodity
    from te.traffic_models.models import UniformTrafficMatrix, UniformTrafficMatrixParams
    
    def make_solver(topo: str, parts: int, tm_seed: int, pgd_step: float, pgd_iters: int) -> DenseSolver:
        g = load_zoo_topology(topo)
        print(f'{g.number_of_nodes()} | {g.number_of_edges()} | {g.number_of_nodes() * g.number_of_edges()}')
        M = get_graph_M_matrix(g)
        N = get_adjacency_null_space(M)
        NNT = cpu_array(N @ N.T)
        INDICES = get_edge_indexing(g)
        TM = UniformTrafficMatrix(seed=tm_seed, params=UniformTrafficMatrixParams(g.number_of_nodes(), 0.0, 1.0))
        COMMODITIES = traffic_to_commodity(TM)
        NCOLS = len(COMMODITIES) // parts
        COMMODITIES = COMMODITIES[:NCOLS]
        del TM
        MASK = get_commodity_in_out_mask(g, COMMODITIES, INDICES)
        X0 = get_feasible_flow_assignment(g, COMMODITIES, InitialSolutionType.PSEUDO_INVERSE)
        assert MASK.shape[1] == NCOLS
        assert X0.shape[1] == NCOLS
        return DenseSolver(X_0=X0, NNT=NNT, mask=MASK, pgd_step=pgd_step, pgd_iters=pgd_iters)

    def test_1(solver: DenseSolver, X: CPUArray, bias: CPUArray):
        solver.set_X_ek(X)
        start = time.perf_counter_ns()
        # c_block = solver._get_current_C(bias)
        # solver.update_minimum(c_block)
        solver.update_minimum(bias)
        # means = np.mean(X_EK, axis=1)
        return (time.perf_counter_ns() - start) // 1e3

    # def test_2(solver: DenseSolver, X: CPUArray, bias: CPUArray):
    #     solver.set_X_ek(X)
    #     start = time.perf_counter_ns()
    #     solver.update_minimum_2(bias)
    #     # means = np.mean(X_EK, axis=1)
    #     return (time.perf_counter_ns() - start) // 1e6
    
    # TOPO = 'Kdl'
    # PARTS = 753
    TOPO = 'Cogentco'
    PARTS = 197
    # TOPO = 'Interoute'
    # PARTS = 110
    # TOPO = 'Colt'
    # PARTS = 153
    # TOPO = 'DialtelecomCz'
    # PARTS = 194
    # TOPO = 'TataNld'
    # TOPO = 'Uninett2010'
    PARTS = 62
    SEED = 12345
    GAMMA = 1.0
    ITERS = 1
    NUM = 30
    RNG = np.random.default_rng(SEED+1)

    solver = make_solver(TOPO, PARTS, SEED, GAMMA, ITERS)
    X = cpu_array(RNG.random(size=solver._X_0.shape))
    BIAS = cpu_array(RNG.random(size=(solver._X_0.shape[0],)))
    print(f"Shape of assignment: {X.shape}")
    times = []
    for i in range(NUM):
        t = test_1(solver, X=X, bias=BIAS)
        # t = test_f(solver, X_f=X_F, bias_f=BIAS_F)
        # t = test_2(solver, X=X, bias=BIAS)
        times.append(t)
    print(times)
    mean = np.mean(times)
    print(f"Mean Per Iter: {mean / (ITERS)} us")
