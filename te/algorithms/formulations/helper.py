import contextlib
import jsonargparse
from typing import Optional, Tuple
from te.algorithms.base import *
from topologies.utils import load_zoo_topology, set_random_capacities
from te.traffic_models.generators import attach_TM_class_parser, parse_and_get_TM
from te.algorithms.utils import get_solution_confusion_matrix
from utils.logging import as_info, as_fail, as_warning, log_subsection_title, str_round


def solve_te_and_check(
    problem: TEProblemDescription, 
    solver_cls: type[TELP], 
    solver_params: SolverParams, 
    *args, **kwargs
) -> TETracer:
    """
    Create the TE LP instance, solve it, and finally check it.
    
    Arguments
    ---------
    problem: TrafficEngineeringProblemDescription
        Full TE problem input and evaluation description
    solver_cls: type[TELP]
        TE solver class to instantiate
    solver_params: SolverParams
        TE solver parameters
    args, kwargs:
        Extra parameters to pass to `solver_cls` constructor.
    
    Returns
    -------
    trace: Optional[TEObjectiveTrace]
        The objective trace (along with any debug info) recorded as
        the algorithm was executed.
    """
    with contextlib.closing(solver_cls(problem, solver_params, *args, **kwargs)) as lp:
        print(as_info(f"Solving With: {lp.alg_name}"))
        print(as_info(f"Evaluating With Parameters:\n{problem.eval_params}"))
        print(as_info(f"Solving With Parameters:\n{solver_params.str_all()}"))
        print(as_info(log_subsection_title("MAKING TE LP")))
        lp.make_lp()
        print(as_info(log_subsection_title(f"SOLVING WITH: {lp.alg_name}")))
        lp.solve()
        
        # get_solution_confusion_matrix(lp, problem.eval_params)
        
        # stats = stringify_collected_stats()
        # if stats is not None:
        #     print(as_info(stats))
        
        return lp.tracer


def te_problem_description_parser(prog_name: str) -> jsonargparse.ArgumentParser:
    """
    Helper utility that creates an argument parser for defining a random TE problem.

    Arguments
    ---------
    prog_name: str
        Problem name to see when `help` is issued.
    
    Returns
    -------
    parser: jsonargparse.ArgumentParser
        A partially completed argument parser. The fields that it contains 
        are described below.

    General Parameters
    ------------------
    `topo`: str
        The topology name (must in the Internet Topology Zoo)
    `tm-seed`: int
        The RNG seed used to generate the TM
    `objective`: TEObjective
        TE objective to solve for
    `tm-path`: Optional[str]
        Path to a TM file that will be loaded as a file-backed matrix
    
    Runtime Parameters
    ------------------
    `feas-tol`: float
        Contraint feasibility absolute tolerance
    `conv-tol`: float
        Objective value relative convergance tolerance
    `scale-factor`: float
        Link capacity scaling factor
    `report-unsat`: bool
        Whether or not to output the details of unsatisfied demands or
        congested links.
    
    Solution Output Parameters
    --------------------------
    `save-sol`: bool
        Whether or not to save the solution output
    `path-sol`: str
        Path to the directory to output the solution
    `name-sol`: str
        Name of teh output solution file
    """
    parser = jsonargparse.ArgumentParser(prog_name)

    parser.add_argument('--config', action='config')

    # Objective
    parser.add_argument('--objective', help='TE Objective', type=TEObjective, default=TEObjective.MLU)
    
    # Topology params
    topo_group = parser.add_argument_group('Topology')
    topo_group.add_argument('--topo-path', help='Path to a topology file', type=Optional[str])
    topo_group.add_argument('--topo-name', help='Name of a topology from the zoo', type=Optional[str])

    # RNG seeds if we have to generate anything
    rng_seed_group = parser.add_argument_group('RNG Seeds')
    rng_seed_group.add_argument('--tm-seed', help='RNG seed for generating traffic matrix sequences')
    rng_seed_group.add_argument('--topo-seed', help='RNG seed for generating topology capacities')

    # Evaluation parameters
    parser.add_class_arguments(TEEvaluationParams, 'EvaluationParams', help='TE Evaluation Parameters')

    # TM class selection
    attach_TM_class_parser(parser)

    return parser


def parse_te_problem_description_args(parser: jsonargparse.ArgumentParser) -> Tuple[
    TEProblemDescription,
    jsonargparse.Namespace]:
    """
    Parse all the default arguments needed for the TE problem.

    Arguments
    ---------
    parser: `jsonargparse.ArgumentParser`
        The argument parser (assumed produced with `te_argparser`)
    
    Returns
    -------
    problem_description: TrafficEngineeringProblemDescription
        Full description of our TE problem to pass to our solvers
    args: jsonargparse.Namespace
        The namespace object of parsed arguments to further process
    """
    args = parser.parse_args()

    # Objective
    objective = args.objective

    # Load the topology
    if args.topo_path is not None:
        # graph = load_topo_from_path(args.topo_path)
        # print(as_info(f'Loaded topology from path {args.topo_path}'))
        raise NotImplementedError
    elif args.topo_name is not None:
        graph = load_zoo_topology(args.topo_name)
        print(as_info(f'Loaded zoo topology {args.topo_name}'))
    else:
        raise ValueError('No topology provided!')
    # Assign the capacities
    if not all('capacity' in graph[u][v] for u, v in graph.edges()):
        print(as_warning('Some edges have missing capacities. Will generate random capacity values for all edges'))
        set_random_capacities(graph, args.topo_seed)

    # Seeds
    tm_seed = args.tm_seed
    topo_seed = args.topo_seed
    print(as_info(f'RNG Seeds:\tTopo:{topo_seed}\tTM:{tm_seed}'))

    # Evaluation parameters
    eval_params = TEEvaluationParams.make_from_args(args.EvaluationParams)
    print(as_info(f"Evaluation Parameters:\n{eval_params}"))

    # Traffic matrix
    tm_generator = parse_and_get_TM(
        tm_seed=tm_seed, tm_count=eval_params.sequence_length,
        scale_factor=eval_params.scale_factor, graph=graph,
        args=args
    )
    print(as_info(f"Using TM Class `{tm_generator.type()}` With Parameters:\n{tm_generator.params}"))

    problem_description = TEProblemDescription(
        objective=objective, eval_params=eval_params,
        graph=graph, tm_generator=tm_generator
    )
    
    return problem_description, args


__all__ = [
    'solve_te_and_check',
    'parse_te_problem_description_args', 
    'te_problem_description_parser'
]