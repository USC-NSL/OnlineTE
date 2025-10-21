from gurobipy import GRB
from te.algorithms.formulations.aggregate import (
    CentralizedEdgeBasedLP, GurobiSolverParams,
    UnregulatedADMMLP, UnregulatedADMMSolverParams,
    RegularizedADMMLP, RegularizedADMMSolverParams,
    GPUUnregulatedADMMLP, GPUUnregulatedADMMSolverParams,
    MultiGPUUnregulatedADMMLP, MultiGPUUnregulatedADMMSolverParams,
    CentralizedPathBasedLP, CentralizedPathBasedSolverParams
)
from te.algorithms.formulations.edge_based_centralized import PDLPParams
from te.algorithms.formulations.edge_based_centralized.edge_based_centralized_pdlp import CentralizedEdgeBasedPDLP
from te.algorithms.base import TrafficEngineeringLPEvaluationParams
from te.algorithms.utils import test_mlu
from te.algorithms.solution import EdgeBasedMinimizeMaximumUtilitySolutionParams, default_solution_name
from topologies.utils import get_uniform_tm_problem_with_capacity_heuristic

import warnings
warnings.filterwarnings("error")

RNG_SEED = 12345
FEASIBILITY_RATIO = 1e-2
FLOAT_RES = 1e-4

SMALL_TOPOLOGY = 'Claranet'
SMALL_MEDIUM_TOPOLOGY = 'Forthnet'
MEDIUM_TOPOLOGY = 'Interoute'
MEDIUM_LARGE_TOPOLOGY = 'Cogentco'
HUGE_TOPOLOGY = 'Kdl'
ARTIFICIAL_MEDIUM_TOPOLOGY_1 = 'Artificial-200'
ARTIFICIAL_MEDIUM_TOPOLOGY_2 = 'Artificial-300'
ARTIFICIAL_MEDIUM_TOPOLOGY_4 = 'Artificial-400'


NUMBER_OF_PATHS = 16


SMALL_EVAL_PARAMS = TrafficEngineeringLPEvaluationParams(
    TopologyName='Claranet', Seed=RNG_SEED, PrintReports=True,
    FeasibilityRatio=FEASIBILITY_RATIO,
    FloatResolution=FLOAT_RES
)
SMALL_MEDIUM_EVAL_PARAMS = TrafficEngineeringLPEvaluationParams(
    TopologyName='Forthnet', Seed=RNG_SEED, PrintReports=True,
    FeasibilityRatio=FEASIBILITY_RATIO,
    FloatResolution=FLOAT_RES
)
MEDIUM_EVAL_PARAMS = TrafficEngineeringLPEvaluationParams(
    TopologyName='Interoute', Seed=RNG_SEED, PrintReports=False,
    FeasibilityRatio=FEASIBILITY_RATIO,
    FloatResolution=FLOAT_RES
)
MEDIUM_LARGE_EVAL_PARAMS = TrafficEngineeringLPEvaluationParams(
    TopologyName='Cogentco', Seed=RNG_SEED, PrintReports=False,
    FeasibilityRatio=FEASIBILITY_RATIO,
    FloatResolution=FLOAT_RES,
    ScaleFactor=20.0
)
LARGE_EVAL_PARAMS = TrafficEngineeringLPEvaluationParams(
    TopologyName='Kdl', Seed=RNG_SEED, PrintReports=False,
    FeasibilityRatio=FEASIBILITY_RATIO,
    FloatResolution=FLOAT_RES,
    # ScaleFactor=20.0
)


def centralized_edge_based_test(eval_params: TrafficEngineeringLPEvaluationParams, method: int = GRB.METHOD_BARRIER, 
                                crossover: bool = False):
    c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(eval_params.TopologyName, eval_params.Seed, scale_factor=eval_params.ScaleFactor)
    print(f"Network link capacity is: {str(round(c, 2))}")

    solver_params = GurobiSolverParams(ConvTol=1e-6, FeasibilityTol=1e-8, Threads=8, Method=method, Crossover=crossover)
    solution_params = None
    if eval_params.SaveSol:
        solution_params = EdgeBasedMinimizeMaximumUtilitySolutionParams(
            seed=eval_params.Seed, topology_name=eval_params.TopologyName, capacity=c,
            tm_model_name=tm.type(), tm_model_params=tm.params,
            path=None, sol_name=default_solution_name(
                topology_name=eval_params.TopologyName, rng_seed=eval_params.Seed, tm_type=tm.type(),
                method=method, crossover=crossover
            )
        )
    test_mlu(CentralizedEdgeBasedLP, graph, tm, solver_params, eval_params, solution_params=solution_params)


def centralized_edge_based_pdlp_test(eval_params: TrafficEngineeringLPEvaluationParams, num_threads: int):
    c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(eval_params.TopologyName, eval_params.Seed, scale_factor=eval_params.ScaleFactor)
    print(f"Network link capacity is: {str(round(c, 2))}")

    solver_params = PDLPParams(Threads=num_threads, Presolve=False, ConvTol=1e-4)
    solution_params = None
    if eval_params.SaveSol:
        raise NotImplementedError
    test_mlu(CentralizedEdgeBasedPDLP, graph, tm, solver_params, eval_params, solution_params=solution_params)


def centralized_path_based_test(eval_params: TrafficEngineeringLPEvaluationParams, method: int = GRB.METHOD_BARRIER, 
                                crossover: bool = False):
    c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(eval_params.TopologyName, eval_params.Seed, scale_factor=eval_params.ScaleFactor)
    print(f"Network link capacity is: {str(round(c, 2))}")

    solver_params = CentralizedPathBasedSolverParams(
        ConvTol=1e-6, FeasibilityTol=1e-8, Threads=8, Method=method, Crossover=crossover,
        NumberOfPathsPerCommodity=NUMBER_OF_PATHS,
        TopologyName=eval_params.TopologyName
    )
    solution_params = None
    if eval_params.SaveSol:
        raise NotImplementedError
        # solution_params = EdgeBasedMinimizeMaximumUtilitySolutionParams(
        #     seed=eval_params.Seed, topology_name=eval_params.TopologyName, capacity=c,
        #     tm_model_name=tm.type(), tm_model_params=tm.params,
        #     path=None, sol_name=default_solution_name(
        #         topology_name=eval_params.TopologyName, rng_seed=eval_params.Seed, tm_type=tm.type(),
        #         method=method, crossover=crossover
        #     )
        # )
    test_mlu(CentralizedPathBasedLP, graph, tm, solver_params, eval_params, solution_params=solution_params)


def regularized_admm_test(eval_params: TrafficEngineeringLPEvaluationParams):
    c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(eval_params.TopologyName, eval_params.Seed, scale_factor=eval_params.ScaleFactor)
    print(f"Network link capacity is: {str(round(c, 2))}")
    n = graph.number_of_nodes()

    solver_params = RegularizedADMMSolverParams(
        NumberOfEpochs=50,
        NumberOfNetworkUpdates=2,
        PGDIterations=3,
        Gamma=None,
        Epsilon=1e-4,
        Eta=10/(n**2),
        Rho=1/(n**2),
        PGDConvTol=1e-4,
        NumWorkers=1,
        UseVariableRho=True,
        BigTheta=1e-6,
        BigGamma=1e-7,
        BlockMode=True,
        CheckBlockConv=False,
        Seed=RNG_SEED
    )
    test_mlu(RegularizedADMMLP, graph, tm, solver_params, eval_params)


def unregulated_admm_test(eval_params: TrafficEngineeringLPEvaluationParams):
    c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(eval_params.TopologyName, eval_params.Seed, scale_factor=eval_params.ScaleFactor)
    print(f"Network link capacity is: {str(round(c, 2))}")
    n = graph.number_of_nodes()

    solver_params = UnregulatedADMMSolverParams(
        NumberOfEpochs=100,
        NumberOfNetworkUpdates=2,
        PGDIterations=2,
        Gamma=None,
        # Eta=1e-4,
        # Rho=1e-5,
        # Eta=10/n**2,
        # Rho=1.125/n**2,
        Eta=8,
        Rho=1,
        PGDConvTol=1e-4,
        NumWorkers=8,
        UseVariableRho=True,
        BigTheta=1e-6,
        BigGamma=1e-7,
        BlockMode=True,
        CheckBlockConv=False,
        Seed=RNG_SEED
    )
    test_mlu(UnregulatedADMMLP, graph, tm, solver_params, eval_params)


def gpu_unregulated_admm_test(eval_params: TrafficEngineeringLPEvaluationParams):
    c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(eval_params.TopologyName, eval_params.Seed, scale_factor=eval_params.ScaleFactor)
    print(f"Network link capacity is: {str(round(c, 2))}")
    m = graph.number_of_nodes()
    n = graph.number_of_edges()

    solver_params = GPUUnregulatedADMMSolverParams(
        NumberOfEpochs=100,
        NumberOfNetworkUpdates=2,
        PGDIterations=2,
        Gamma=2,
        # Eta=10/(n * c ** 2),
        # Rho=50/(n * c ** 2),
        # Eta=1/m**2,
        # Rho=0.25/m**2,
        Eta=8,
        Rho=1,
        UseVariableRho=False,
        BigTheta=1e-6,
        BigGamma=1e-7,
        Kappa=0.1,
        FloatPrecision='half',
        Seed=RNG_SEED
    )
    test_mlu(GPUUnregulatedADMMLP, graph, tm, solver_params, eval_params)


def multi_gpu_unregulated_admm_test(eval_params: TrafficEngineeringLPEvaluationParams):
    c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(eval_params.TopologyName, eval_params.Seed, scale_factor=eval_params.ScaleFactor)
    print(f"Network link capacity is: {str(round(c, 2))}")
    m = graph.number_of_nodes()
    n = graph.number_of_edges()

    solver_params = MultiGPUUnregulatedADMMSolverParams(
        NumberOfEpochs=250,
        NumberOfNetworkUpdates=3,
        PGDIterations=3,
        Gamma=1,
        # Eta=10/(n * c ** 2),
        # Rho=5e-2/(n * c ** 2),
        Eta=8,
        Rho=1e-1,
        BigTheta=1e-6,
        BigGamma=1e-7,
        Kappa=0.1,
        FloatPrecision='half',
        Seed=RNG_SEED
    )
    test_mlu(MultiGPUUnregulatedADMMLP, graph, tm, solver_params, eval_params)


if __name__ == '__main__':
    # centralized_edge_based_test(SMALL_EVAL_PARAMS(SaveSol=True), method=GRB.METHOD_DUAL)
    # centralized_edge_based_test(SMALL_EVAL_PARAMS(SaveSol=True), method=GRB.METHOD_BARRIER, crossover=False)
    # centralized_edge_based_test(SMALL_EVAL_PARAMS, method=GRB.METHOD_BARRIER, crossover=False)
    # centralized_edge_based_test(SMALL_MEDIUM_EVAL_PARAMS)
    # centralized_edge_based_test(MEDIUM_EVAL_PARAMS)
    # centralized_edge_based_test(HUGE_TOPOLOGY, RNG_SEED, scale_factor=200)
    # regularized_admm_test(HUGE_TOPOLOGY, RNG_SEED, scale_factor=200)
    # regularized_admm_test(SMALL_EVAL_PARAMS)
    # unregulated_admm_test(SMALL_EVAL_PARAMS)
    # unregulated_admm_test(SMALL_MEDIUM_EVAL_PARAMS)
    # unregulated_admm_test(MEDIUM_EVAL_PARAMS)
    # gpu_unregulated_admm_test(SMALL_EVAL_PARAMS)
    # gpu_unregulated_admm_test(SMALL_MEDIUM_EVAL_PARAMS)
    # gpu_unregulated_admm_test(MEDIUM_EVAL_PARAMS)
    # gpu_unregulated_admm_test(HUGE_TOPOLOGY, RNG_SEED, report=False)
    # gpu_unregulated_admm_test(ARTIFICIAL_MEDIUM_TOPOLOGY_1, RNG_SEED)
    # gpu_unregulated_admm_test(ARTIFICIAL_MEDIUM_TOPOLOGY_2, RNG_SEED)
    # multi_gpu_unregulated_admm_test(SMALL_TOPOLOGY, RNG_SEED)
    # multi_gpu_unregulated_admm_test(SMALL_MEDIUM_TOPOLOGY, RNG_SEED)
    # multi_gpu_unregulated_admm_test(MEDIUM_TOPOLOGY, RNG_SEED)
    # multi_gpu_unregulated_admm_test(ARTIFICIAL_MEDIUM_TOPOLOGY_1, RNG_SEED)
    # multi_gpu_unregulated_admm_test(HUGE_TOPOLOGY, RNG_SEED, report=False, scale_factor=200)
    # centralized_path_based_test(SMALL_EVAL_PARAMS, method=GRB.METHOD_DUAL)
    # centralized_path_based_test(SMALL_MEDIUM_EVAL_PARAMS, method=GRB.METHOD_DUAL)
    # centralized_path_based_test(MEDIUM_EVAL_PARAMS, method=GRB.METHOD_DUAL)
    # centralized_path_based_test(LARGE_EVAL_PARAMS, method=GRB.METHOD_DUAL)
    # centralized_edge_based_pdlp_test(SMALL_EVAL_PARAMS, num_threads=1)
    # centralized_edge_based_pdlp_test(SMALL_MEDIUM_EVAL_PARAMS, num_threads=2)
    centralized_edge_based_pdlp_test(MEDIUM_LARGE_EVAL_PARAMS, num_threads=4)
    # centralized_edge_based_pdlp_test(LARGE_EVAL_PARAMS, num_threads=16)
