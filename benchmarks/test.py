from te.algorithms.base import GurobiSolverParams
from te.algorithms.formulations.edge_based_centralized import CentralizedEdgeBasedLP
from te.algorithms.formulations.edge_based_regularized_admm import RegularizedADMMLP, RegularizedADMMSolverParams
from te.algorithms.formulations.edge_based_unregulated_admm import UnregulatedADMMLP, UnregulatedADMMSolverParams
from te.algorithms.formulations.edge_based_unregulated_admm_gpu import GPUUnregulatedADMMLP, GPUUnregulatedADMMSolverParams
from te.algorithms.utils import test_mlu
from topologies.utils import get_uniform_tm_problem_with_capacity_heuristic

RNG_SEED = 12345

FEASIBILITY_TOL = None
FEASIBILITY_RATIO = 1e-2

SMALL_TOPOLOGY = 'Claranet'
SMALL_MEDIUM_TOPOLOGY = 'Forthnet'
MEDIUM_TOPOLOGY = 'Interoute'
HUGE_TOPOLOGY = 'Kdl'


def centralized_test(topology: str, seed: int, **kwargs):
    c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(topology, seed)
    print(f"Network link capacity is: {str(round(c, 2))}")

    solver_params = GurobiSolverParams(ConvTol=1e-6, FeasibilityTol=1e-8)
    test_mlu(CentralizedEdgeBasedLP, graph, tm, solver_params, feasibility_tol=FEASIBILITY_TOL, 
             feasibility_ratio=FEASIBILITY_RATIO, **kwargs)


def regularized_admm_test(topology: str, seed: int, **kwargs):
    c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(topology, seed)
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
    test_mlu(RegularizedADMMLP, graph, tm, solver_params, feasibility_tol=FEASIBILITY_TOL, 
             feasibility_ratio=FEASIBILITY_RATIO, **kwargs)


def unregulated_admm_test(topology: str, seed: int, **kwargs):
    c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(topology, seed)
    print(f"Network link capacity is: {str(round(c, 2))}")
    n = graph.number_of_nodes()

    solver_params = UnregulatedADMMSolverParams(
        NumberOfEpochs=100,
        NumberOfNetworkUpdates=2,
        PGDIterations=2,
        Gamma=None,
        # Eta=1e-4,
        # Rho=1e-5,
        Eta=10/n**2,
        Rho=1.125/n**2,
        PGDConvTol=1e-4,
        NumWorkers=8,
        UseVariableRho=True,
        BigTheta=1e-6,
        BigGamma=1e-7,
        BlockMode=True,
        CheckBlockConv=False,
        Seed=RNG_SEED
    )
    test_mlu(UnregulatedADMMLP, graph, tm, solver_params, feasibility_tol=FEASIBILITY_TOL, 
             feasibility_ratio=FEASIBILITY_RATIO, **kwargs)


def gpu_unregulated_admm_test(topology: str, seed: int, **kwargs):
    c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(topology, seed)
    print(f"Network link capacity is: {str(round(c, 2))}")
    n = graph.number_of_nodes()

    solver_params = GPUUnregulatedADMMSolverParams(
        NumberOfEpochs=150,
        NumberOfNetworkUpdates=3,
        PGDIterations=3,
        Gamma=2,
        Eta=1/n**2,
        Rho=0.25/n**2,
        UseVariableRho=False,
        BigTheta=1e-6,
        BigGamma=1e-7,
        Kappa=0.1,
        FloatPrecision='half',
        Seed=RNG_SEED
    )
    test_mlu(GPUUnregulatedADMMLP, graph, tm, solver_params, feasibility_tol=FEASIBILITY_TOL, 
             feasibility_ratio=FEASIBILITY_RATIO, **kwargs)


if __name__ == '__main__':
    # centralized_test(SMALL_MEDIUM_TOPOLOGY, RNG_SEED)
    # centralized_test(MEDIUM_TOPOLOGY, RNG_SEED)
    # centralized_test(HUGE_TOPOLOGY, RNG_SEED)
    # regularized_admm_test(HUGE_TOPOLOGY, RNG_SEED)
    # regularized_admm_test(SMALL_TOPOLOGY, RNG_SEED)
    # unregulated_admm_test(SMALL_TOPOLOGY, RNG_SEED, trace_out_path=None)
    # unregulated_admm_test(SMALL_MEDIUM_TOPOLOGY, RNG_SEED, trace_out_path=None)
    # unregulated_admm_test(MEDIUM_TOPOLOGY, RNG_SEED)
    # gpu_unregulated_admm_test(SMALL_TOPOLOGY, RNG_SEED)
    # gpu_unregulated_admm_test(SMALL_MEDIUM_TOPOLOGY, RNG_SEED)
    gpu_unregulated_admm_test(MEDIUM_TOPOLOGY, RNG_SEED)
    # gpu_unregulated_admm_test(HUGE_TOPOLOGY, RNG_SEED)
