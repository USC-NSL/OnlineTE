import time
import networkx as nx
from typing import Optional, Tuple, Dict
from array_utils import set_global_precision
from array_utils.cpu.types import *
from array_utils.cpu.wrapper import cpu_fill, cpu_copyto
from te.algorithms.communication import *
from te.algorithms.base import TEObjective
from te.algorithms.sub_algorithms.pgd import (
    do_packed_path_based_maxflow_pgd,
    do_packed_path_based_nesterov_pgd,
)
from te.path_providers import *
from utils.logging import as_warning
from .packed_paths import PackedPathBatch
from .solver_params import PathBasedSimplifiedOnlineTEParameters, WorkerUpdateData


class DenseSolver:
    def __init__(self,
        demands: CPUArray,
        path_batch: PackedPathBatch,
        pgd_step: float, pgd_iters: int,
        capacities: CPUArray,
        objective: TEObjective,
        rho: float,
        total_numbar_of_commodities: int,
        optimality_tolerance: Optional[float] = None
    ):
        self._paths = path_batch
        self._demands: CPUArray = cpu_array(demands)
        self._pgd_iters = pgd_iters
        self._optimality_tolerance = optimality_tolerance
        self._capacities = cpu_array(capacities)
        self._objective = objective
        self._rho = rho

        self._pgd_step_0 = pgd_step
        self._total_numbar_of_commodities = total_numbar_of_commodities
        self._N = path_batch.num_edges
        self._Y_tk = cpu_zeros((path_batch.num_paths,))
        self._initialize_splits()
        self._Y_tk_old = cpu_array(self._Y_tk)
        self._update_step_sizes()
    
    def _initialize_splits(self):
        self._Y_tk = self._paths.initialize_splits(self._capacities, self._objective)

    def _update_step_sizes(self):
        self._pgd_steps = cpu_array(
            self._pgd_step_0 / self._paths.estimate_lipschitz(self._demands)
        )

    @property
    def X_ek(self) -> CPUArray:
        return self._paths.paths_to_edge(self._Y_tk, self._demands)

    @property
    def X_bar(self) -> CPUArray:
        return self._paths.paths_to_edge_mean(self._Y_tk, self._demands)

    @property
    def total_flow(self) -> float:
        return self._paths.total_flow(self._Y_tk, self._demands)

    @property
    def norm_F_delta_X(self) -> float:
        return self._paths.assignment_delta_norm_squared(
            current=self._Y_tk,
            previous=self._Y_tk_old,
            demands=self._demands
        )

    def dual_objective_term(self, sharing_dual: CPUArray) -> float:
        return self._paths.dual_objective_term(
            edge_duals=sharing_dual,
            demands=self._demands,
            objective=self._objective,
            total_num_commodities=self._total_numbar_of_commodities
        )

    def set_demands(self, demands: CPUArray):
        self._demands = cpu_array(demands)
        self._update_step_sizes()
    
    def update(self, sharing_bias: CPUArray, sharing_rho: float) -> CPUArray:
        sharing_bias = cpu_array(sharing_bias)
        self._rho = sharing_rho
        if self._objective == TEObjective.MLU:
            linear_term = self._paths.linear_term(sharing_bias, self._demands)
            self._Y_tk = do_packed_path_based_nesterov_pgd(
                y_block=self._Y_tk,
                y_block_old=self._Y_tk_old,
                path_batch=self._paths,
                linear_term=linear_term,
                demand_block=self._demands,
                step_sizes=self._pgd_steps,
                n_iter=self._pgd_iters,
                optimality_tolerance=self._optimality_tolerance,
            )
        elif self._objective == TEObjective.MAX_FLOW:
            linear_term = self._paths.linear_term(
                sharing_bias,
                self._demands,
                maxflow_shift=1 / (self._rho * self._total_numbar_of_commodities),
            )
            self._Y_tk = do_packed_path_based_maxflow_pgd(
                y_block=self._Y_tk,
                y_block_old=self._Y_tk_old,
                path_batch=self._paths,
                linear_term=linear_term,
                demand_block=self._demands,
                step_sizes=self._pgd_steps,
                n_iter=self._pgd_iters,
                optimality_tolerance=self._optimality_tolerance,
            )
        else:
            raise ValueError
        return self._paths.paths_to_edge_mean(self._Y_tk, self._demands)

    def update_Y_tk_old(self):
        cpu_copyto(self._Y_tk, self._Y_tk_old)


class SimpleOnlineTEWorkerNode(DistributedSolverNodeBase):
    def __init__(self, params: DistributedSolverNodeParams):
        super().__init__(params)
        self._solver_params: Optional[PathBasedSimplifiedOnlineTEParameters] = None
        self._objective: Optional[TEObjective] = None
        self._ready: bool = False

        self._sharing_bias_cached: Optional[CPUArray] = None
        self._sharing_dual_cached: Optional[CPUArray] = None
        self._sharing_rho_cached: Optional[float] = None

        self._dense_solver: Optional[DenseSolver] = None

        assert issubclass(
            params.CommunicationBackendCLS, WorkerBackendBase)
        self.backend: WorkerBackendBase =\
            params.CommunicationBackendCLS[PathBasedSimplifiedOnlineTEParameters](
                rpc_params=params.RPCParams_,
                solver_params_cls=PathBasedSimplifiedOnlineTEParameters
            )
        self.backend.start()

    def initialize(self):
        self.backend.set_solver_parameters = self.set_solver_parameters
        self.backend.set_topology = self.set_topology
        self.backend.do_inner_loop_update = self.do_inner_loop_pgd_update
        self.backend.report_chunk = self.report_chunk
        self.backend.update_demands = self.update_demands
    
    def run(self):
        self.backend.wait()

    def set_solver_parameters(self,
        new_params: PathBasedSimplifiedOnlineTEParameters,
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
        self._sharing_bias_cached = cpu_zeros((graph.number_of_edges(),))
        self._sharing_dual_cached = cpu_zeros((graph.number_of_edges(),))
        self._sharing_rho_cached = self._solver_params.Rho
        path_batch = PackedPathBatch.from_path_provider(
            provider=self._path_object,
            capacities=self._capacities,
            scale_with_capacity=True,
            kernel_threads=self._solver_params.KernelThreads,
        )
        self._dense_solver = DenseSolver(
            demands=cpu_fill((self.assigned_commodity_count,), 1),
            path_batch=path_batch,
            pgd_step=self._solver_params.Gamma,
            pgd_iters=self._solver_params.MaxSwitchIterations,
            optimality_tolerance=self._solver_params.SwitchOptimalityTolerance,
            capacities=self._capacities,
            objective=self._objective,
            rho=self._sharing_rho_cached,
            total_numbar_of_commodities=self.total_commodity_count
        )

    def do_inner_loop_pgd_update(
        self,
        sharing_gap: CPUArray,
        sharing_dual: CPUArray,
        sharing_rho: float
    ) -> Tuple[int, CPUArray, Dict[str, float]]:
        # Update cached values
        self._sharing_bias_cached = sharing_gap + sharing_dual
        self._sharing_dual_cached = sharing_dual * sharing_rho
        self._sharing_rho_cached = sharing_rho
        # self._sharing_dual_cached = sharing_dual * sharing_rho / self.total_commodity_count
        # X-step
        DENSE = self._dense_solver
        start = time.perf_counter_ns()
        mean = DENSE.update(self._sharing_bias_cached, self._sharing_rho_cached)
        total_routed_flow = DENSE.total_flow
        norm_F_delta_X = DENSE.norm_F_delta_X
        dual_objective = DENSE.dual_objective_term(
            self._sharing_dual_cached
        )
        DENSE.update_Y_tk_old()
        return (time.perf_counter_ns() - start) // 1000, mean, {
            WorkerUpdateData.DEMANDS: total_routed_flow,
            WorkerUpdateData.NORM_F_DELTA_X: norm_F_delta_X,
            WorkerUpdateData.DUAL_OBJ: dual_objective
        }
    
    def report_chunk(self) -> CPUArray:
        return self._dense_solver.X_ek

    def update_demands(self, demands: CPUArray) -> CPUArray:
        self._dense_solver.set_demands(demands)
        return self._dense_solver.X_bar

    def close(self):
        self.backend.close()
