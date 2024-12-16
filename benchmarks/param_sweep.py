import contextlib
import matplotlib.pyplot as plt
from typing import List
from te.traffic_models import get_traffic_model
from te.traffic_models.models import UniformTrafficMatrixParams
from te.algorithms.formulations.edge_based_distributed_admm_debug import (
    SemiDistributedEdgeBasedADMMLP, DistributedADMMDebugSolverParams
)
from topologies.utils import (
    load_zoo_topology, get_capacity_lower_bound,
    set_edge_capacity_to
)


def admm_rho_param_sweep(rho_list: List[float], num_epochs: int = 50):
    graph = load_zoo_topology('Twaren')
    
    tm_params = UniformTrafficMatrixParams(
        n = len(graph.nodes), min = 0.0, max = 1.0
    )
    tm = get_traffic_model('Uniform')(seed=12345, params=tm_params)
    
    c_min = get_capacity_lower_bound(graph, tm)
    set_edge_capacity_to(graph, c_min*3)
    print(f"Capacity lower bound is: {c_min}")

    objective_traces: List[List[float]] = []
    for rho in rho_list:
        solver_params = DistributedADMMDebugSolverParams(
            NumberOfEpochs=num_epochs, Rho=rho
        )
        with contextlib.closing(SemiDistributedEdgeBasedADMMLP(graph, tm, solver_params)) as lp:
            lp.make_lp()
            lp.solve()
            objective_traces.append(lp.objective_trace)
    
    for rho, trace in zip(rho_list, objective_traces):
        print(f"Trace for rho = {rho}:\n\t{trace}")
    plt.figure()
    for trace in objective_traces:
        plt.plot(trace)
    plt.legend([f'Rho = {rho}' for rho in rho_list])
    plt.show()


if __name__ == '__main__':
    admm_rho_param_sweep([1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1], 50)
