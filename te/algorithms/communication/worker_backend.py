import networkx as nx
from typing import Tuple, Callable
from .base import CommunicationBackendBase
from array_utils.cpu.types import *
from te.algorithms.base import SolverParams


class WorkerBackendBase[P: SolverParams](CommunicationBackendBase):
    # These workers never directly reach-out to any node, they are passive
    def are_all_peers_reachable(self):
        return True
    def are_all_workers_reachable(self):
        return True

    @property
    def set_solver_parameters(self) -> Callable[[P, int], None]:
        return self._set_solver_parameters
    @set_solver_parameters.setter
    def set_solver_parameters(self, f: Callable[[P, int], None]):
        self._set_solver_parameters = f

    @property
    def set_topology(self) -> Callable[[nx.DiGraph], None]:
        return self._set_topology
    @set_topology.setter
    def set_topology(self, f: Callable[[nx.DiGraph], None]):
        self._set_topology = f
    
    @property
    def do_inner_loop_update(self) -> Callable[[int], Tuple[int, CPUArray]]:
        return self._do_inner_loop_update
    @do_inner_loop_update.setter
    def do_inner_loop_update(self, f: Callable[[int], Tuple[int, CPUArray]]):
        self._do_inner_loop_update = f
    
    @property
    def update_cached_values(self) -> Callable[[CPUArray, CPUArray, CPUArray], None]:
        return self._update_cached_values
    @update_cached_values.setter
    def update_cached_values(self, f: Callable[[CPUArray, CPUArray, CPUArray], None]):
        self._update_cached_values = f
    
    @property
    def report_chunk(self) -> Callable[[None], CPUArray]:
        return self._report_chunk
    @report_chunk.setter
    def report_chunk(self, f: Callable[[None], CPUArray]):
        self._report_chunk = f
    
    @property
    def report_aggregate(self) -> Callable[[None], CPUArray]:
        return self._report_aggregate
    @report_aggregate.setter
    def report_aggregate(self, f: Callable[[None], CPUArray]):
        self._report_aggregate = f

    @property
    def update_demands(self) -> Callable[[CPUArray], CPUArray]:
        return self._update_demands
    @update_demands.setter
    def update_demands(self, f: Callable[[CPUArray], CPUArray]):
        self._update_demands = f
