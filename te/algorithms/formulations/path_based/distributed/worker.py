import time
import numpy as np
import networkx as nx
from typing import Optional, Tuple
from numba.typed import List as NumbaList
from array_utils import set_global_precision
from array_utils.cpu.types import *
from array_utils.cpu.wrapper import cpu_fill
from te.algorithms.communication import *
from te.algorithms.base import TEObjective
from te.algorithms.sub_algorithms.pgd import do_path_based_maxflow_pgd, do_path_based_nesterov_pgd
from te.path_providers import *
from te.path_providers.sparse_ops import *
from utils.logging import as_warning
from .solver_params import PathBasedOnlineTEParameters


class DenseSolver:
    def __init__(self,
        demands: CPUArray,
        alpha_shape: Tuple[int, int, int],
        alpha_cols: NumbaList,
        alpha_rows: NumbaList, 
        beta: IntegerCPUArray,
        pgd_step: float, pgd_iters: int,
        eta: float,
        adjust_step_size: bool,
        capacities: CPUArray,
        scale_with_capacity: bool,
        objective: TEObjective
    ):
        self._demands: CPUArray = demands
        K, N, T = alpha_shape
        self._alpha_shape = alpha_shape
        self._alpha_rows = alpha_rows
        self._alpha_cols = alpha_cols
        self._beta = beta
        self._adjust_step_size = adjust_step_size
        self._pgd_iters = pgd_iters
        self._eta = eta
        self._scale_with_capacity = scale_with_capacity
        self._capacities = capacities
        self._objective = objective

        self._pgd_step_0 = pgd_step
        self._pgd_steps = pgd_step if not adjust_step_size else \
            cpu_array(pgd_step / path_based_power_method(
                alpha_rows, alpha_cols, alpha_shape,
                demands, self.conditional_capacity
            ))

        self._K = K
        self._N = N
        self._T = T
        self._Y_tk = cpu_zeros((T, K))
        self._initialize_splits()
        self._Y_tk_old = cpu_array(self._Y_tk)
    
    def _initialize_splits(self):
        self._Y_tk = get_path_split_with_capacity(
            self._Y_tk,self._alpha_rows,
            self._alpha_cols, self._capacities
        )

    @property
    def conditional_capacity(self) -> Optional[CPUArray]:
        return self._capacities if self._scale_with_capacity else None

    @property
    def X_ek(self) -> CPUArray:
        return path_based_to_edge_based_nnz(
            self._Y_tk,self._alpha_rows, self._alpha_cols,
            self._N, self._demands,
            self.conditional_capacity
        )

    @property
    def X_bar(self) -> CPUArray:
        return path_based_to_edge_based_mean_nnz(
            self._Y_tk, self._alpha_rows, self._alpha_cols,
            self._N, self._demands,
            self.conditional_capacity
        )

    @property
    def total_flow(self) -> float:
        return np.sum(np.multiply(np.sum(self._Y_tk, axis=0), self._demands))

    def set_demands(self, demands: CPUArray):
        self._demands = demands
        self._pgd_steps = self._pgd_step_0 if not self._adjust_step_size else \
            cpu_array(self._pgd_step_0 / path_based_power_method(
                self._alpha_rows, self._alpha_cols, self._alpha_shape,
                demands, self.conditional_capacity
            ))
    
    def update(self, sharing_bias: CPUArray) -> CPUArray:
        new_Y_old = cpu_array(self._Y_tk)
        if self._objective == TEObjective.MLU:
            self._Y_tk = do_path_based_nesterov_pgd(
                y_block=self._Y_tk,
                y_block_old=self._Y_tk_old,
                alpha_rows=self._alpha_rows,
                alpha_cols=self._alpha_cols,
                sharing_bias=sharing_bias,
                beta_block=self._beta,
                demand_block=self._demands,
                num_edges=self._N,
                num_paths=self._T,
                step_sizes=self._pgd_steps,
                n_iter=self._pgd_iters,
                capacities=self.conditional_capacity
            )
        elif self._objective == TEObjective.MAX_FLOW:
            self._Y_tk = do_path_based_maxflow_pgd(
                y_block=self._Y_tk,
                y_block_old=self._Y_tk_old,
                alpha_rows=self._alpha_rows,
                alpha_cols=self._alpha_cols,
                sharing_bias=sharing_bias,
                beta_block=self._beta,
                demand_block=self._demands,
                num_edges=self._N,
                num_paths=self._T,
                step_sizes=self._pgd_steps,
                n_iter=self._pgd_iters,
                eta=self._eta,
                capacities=self.conditional_capacity
            )
        else:
            raise ValueError
        self._Y_tk_old = new_Y_old
        return path_based_to_edge_based_mean_nnz(
            self._Y_tk, self._alpha_rows, self._alpha_cols,
            self._N, self._demands, self.conditional_capacity
        )


class OnlineTEWorkerNode(DistributedSolverNodeBase):
    def __init__(self, params: DistributedSolverNodeParams):
        super().__init__(params)
        self._solver_params: Optional[PathBasedOnlineTEParameters] = None
        self._objective: Optional[TEObjective] = None
        self._ready: bool = False

        self._T: Optional[int] = None
        self._alpha_shape: Optional[Tuple[int, int, int]] = None
        self._alpha_rows_chunk: Optional[NumbaList[IntegerCPUArray]] = None
        self._alpha_cols_chunk: Optional[NumbaList[IntegerCPUArray]] = None
        self._beta_k_chunk: Optional[IntegerCPUArray] = None

        self._sharing_bias_cached: Optional[CPUArray] = None

        self._dense_solver: Optional[DenseSolver] = None

        assert issubclass(
            params.CommunicationBackendCLS, WorkerBackendBase)
        self.backend: WorkerBackendBase =\
            params.CommunicationBackendCLS[PathBasedOnlineTEParameters](
                rpc_params=params.RPCParams_,
                solver_params_cls=PathBasedOnlineTEParameters
            )
        self.backend.start()

    def initialize(self):
        self.backend.set_solver_parameters = self.set_solver_parameters
        self.backend.set_topology = self.set_topology
        self.backend.do_inner_loop_update = self.do_inner_loop_pgd_update
        self.backend.update_cached_values = self.update_cached_values
        self.backend.report_chunk = self.report_chunk
        self.backend.update_demands = self.update_demands
    
    def run(self):
        self.backend.wait()

    def set_solver_parameters(self,
        new_params: PathBasedOnlineTEParameters,
        num_workers: int,
        objective: TEObjective
    ):
        self._solver_params = new_params
        self._objective = objective
        self.number_of_workers = num_workers
        set_global_precision(precision=new_params.Precision)

    def _set_chunk_alignment(self, num_endpoints: int):
        self._K = num_endpoints * (num_endpoints - 1)
        assert self._K % self._backend.number_of_peers == 0
        self._CHUNK_LEN = self._K // self._backend.number_of_peers
        self._K_START = self.worker_id * self._CHUNK_LEN

    def _get_local_path_file_name(self) -> Optional[str]:
        """
        An annoying thing with this setting is that we may have to
        regenerate path files for different number of worker nodes.
        We have to do this since the number of commodities assigned
        to a worker (and hence available paths) change depending on
        the number of workers.
        To make subsequent runs easier, the path file name used shall
        be:
        ```
        <Coordinator Path File Name Without Extension>_<worker_id>_<_CHUNK_LEN>.pkl
        ```
        """
        coordinator_path_file = self._solver_params.PathFile
        if coordinator_path_file is not None:
            name, ext = coordinator_path_file.split('.', maxsplit=1)
            return f'{name}_{self.worker_id}_{self.assigned_commodity_count}.{ext}'

    def _create_local_path_object(self):
        self._path_object = build_provider(
            T=self._solver_params.NumberOfPathsPerCommodity,
            graph=self._graph,
            # TODO: The scheme may have to be a solver parameter ...
            per_commodity_provider=get_scheme(),
            edge_indexing=self._indexing,
            commodity_id_start=self.assigned_commodity_start_id,
            commodity_id_end=self.assigned_commodity_end_id
        )

    def set_topology(self, graph: nx.DiGraph):
        self.graph = graph
        path = self._get_local_path_file_name()
        if path is None:
            # We were not given any path file, just build it
            self._create_local_path_object()
        else:
            # We have a path file, try to load it
            try:
                self._path_object = PathProvider.load(path)
                K, N, _ = self._path_object.shape
                assert K == self.assigned_commodity_count,\
                    'Commodity count does not match! This is the wrong path file!'
                assert N == self._graph.number_of_edges(),\
                    'Number of edges do not match! This is the wrong path file!'
            except FileNotFoundError:
                # This path file does not exist. For now, our policy is to
                # build and store it locally, as it makes experiments easy.
                print(as_warning(f'Path file {path} does not exist. Will make one.'))
                self._create_local_path_object()
                # Save it for future use!
                self._path_object.save(path)
        warm_start_jit()
        self._sharing_bias_cached = cpu_zeros((graph.number_of_edges(),))
        self._dense_solver = DenseSolver(
            demands=cpu_fill((self.assigned_commodity_count,), 1),
            alpha_shape=self._path_object.shape,
            alpha_cols=NumbaList(self._path_object.cols),
            alpha_rows=NumbaList(self._path_object.rows),
            beta=self._path_object.beta,
            pgd_step=self._solver_params.Gamma,
            pgd_iters=self._solver_params.SwitchIterations,
            eta=self._solver_params.Eta,
            adjust_step_size=self._solver_params.AdjustGamma,
            capacities=self._capacities,
            scale_with_capacity=self._solver_params.ScaleWithCapacity,
            objective=self._objective
        )

    def do_inner_loop_pgd_update(self, epoch: int) -> Tuple[int, CPUArray, Optional[float]]:
        start = time.perf_counter_ns()
        mean = self._dense_solver.update(self._sharing_bias_cached)
        total_flow = self._dense_solver.total_flow
        return (time.perf_counter_ns() - start) // 1000, mean, total_flow

    def update_cached_values(self, sharing_bias: CPUArray):
        self._sharing_bias_cached = sharing_bias
    
    def report_chunk(self) -> CPUArray:
        return self._dense_solver.X_ek
    
    def report_aggregate(self) -> CPUArray:
        raise ValueError("This should NOT be used!")

    def update_demands(self, demands: CPUArray) -> CPUArray:
        self._dense_solver.set_demands(demands)
        return self._dense_solver.X_bar

    def close(self):
        self.backend.close()
