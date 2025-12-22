import numpy as np
import networkx as nx
from te.traffic_models.models import CustomTrafficMatrix
from te.algorithms.base import *
from te.algorithms.formulations.edge_based.distributed.admm_synchronous import SynchADMMSolverParams
from te.algorithms.formulations.edge_based.distributed.admm_synchronous.helper import *


# def get_topo() -> nx.DiGraph:
#     g = nx.DiGraph([
#         (0, 1), (0, 2), (2, 1),
#         (1, 0), (2, 0), (1, 2)
#     ])
#     nx.set_edge_attributes(g, 1.0, name='capacity')
#     return g
# def get_topo() -> nx.DiGraph:
#     g = nx.DiGraph([
#         (0, 1), (0, 2), (1, 3), (2, 3), (1, 2),
#         (1, 0), (2, 0), (3, 1), (3, 2), (2, 1)
#     ])
#     nx.set_edge_attributes(g, {
#         (0, 1): 8, (0, 2): 8, (1, 3): 6, (2, 3): 6, (1, 2): 4,
#         (1, 0): 8, (2, 0): 8, (3, 1): 6, (3, 2): 6, (2, 1): 4
#     }, name='capacity')
#     return g
# def get_topo() -> nx.DiGraph:
#     g = nx.DiGraph([
#         (0, 1), (1, 2), (1, 3), (2, 3),
#         (1, 0), (2, 1), (3, 1), (3, 2)
#     ])
#     nx.set_edge_attributes(g, {
#         (0, 1): 2, (1, 2): 2, (1, 3): 2, (2, 3): 2,
#         (1, 0): 2, (2, 1): 2, (3, 1): 2, (3, 2): 2
#     }, name='capacity')
#     return g
def get_topo() -> nx.DiGraph:
    g = nx.DiGraph([
        (0, 1), (0, 2), (1, 3), (2, 3), (0, 3),
        (1, 0), (2, 0), (3, 1), (3, 2), (3, 0)
    ])
    nx.set_edge_attributes(g, {
        (0, 1): 2, (0, 2): 2, (1, 3): 2, (2, 3): 2, (0, 3): 4,
        (1, 0): 2, (2, 0): 2, (3, 1): 2, (3, 2): 2, (3, 0): 4
    }, name='capacity')
    return g


# def get_tm() -> CustomTrafficMatrix:
#     return CustomTrafficMatrix(
#         np.array([
#             [0, 1.0, 0.0],
#             [0, 0, 0],
#             [0, 0, 0]
#         ])
#     )
def get_tm() -> CustomTrafficMatrix:
    return CustomTrafficMatrix(
        np.array([
            [0, 0, 0, 3.5],
            [0, 0, 0, 2.5],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
    )


def get_problem_description() -> TrafficEngineeringProblemDescription:
    return TrafficEngineeringProblemDescription(
        EvalParams=TrafficEngineeringLPEvaluationParams(
            TopologyName='toy-1',
            Seed=0,
            Objective=TEObjective.MLU,
            ScaleFactor=1.0,
            FeasibilityTolerance=1e-4
        ),
        Graph=get_topo(),
        TM=get_tm()
    )


if __name__ == '__main__':
    """
    These parameters worked well!
    ```
    python3 -m benchmarks.toy_example --num-workers 1 --local ^
        --mlu_backend pdlp --SolverParams.InnerLoopRounds 10 ^
        --SolverParams.OuterLoopRounds 10 --SolverParams.Beta 1e-4 --SolverParams.Rho 2.0
    ```
    python3 -m benchmarks.toy_example --num-workers 1 --local --mlu_backend pdlp --SolverParams.InnerLoopRounds 10 --SolverParams.OuterLoopRounds 10 --SolverParams.Beta 1e-4 --SolverParams.Rho 2.0

    Essentially, we are increasing the inner loop iteration so much that the
    inner loop (i.e. satisfying demands) is solved to a very high degree
    of optimality. When this is the case, it is possible to increase the ADMM
    step size without causing huge oscilations, and make the algorithm finish
    within fewer outer loop iterations.
    """
    problem = get_problem_description()
    parser = distributed_synchronous_admm_parser()
    args = parser.parse_args()
    solver = parse_distributed_synchronous_admm(args)
    spawn_distributed_synchronous_solver(problem, solver, True)
