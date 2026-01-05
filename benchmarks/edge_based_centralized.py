from te.algorithms.formulations.helper import *
from te.algorithms.formulations.edge_based.centralized.aggregate import *
from te.algorithms.sub_algorithms.mlu_backends.aggregate import *


if __name__ == '__main__':
    # Problem description parser
    parser = te_problem_description_parser('Edge-Based Centralized TE')
    # Solver params parsers
    solver_subparser = parser.add_subcommands(dest='solver', help='The solver to use', required=True)
    solver_subparser.add_subcommand('gurobi', centralized_gurobi_solver_params_parser(), help='Options for the Gurobi solver')
    # solver_subparser.add_subcommand('gurobi-dual', centralized_dual_gurobi_solver_params_parser(), help='Options for the Dual Gorobi solver')
    solver_subparser.add_subcommand('pdlp', centralized_pdlp_solver_params_parser(), help='Options for the PDLP solver')
    if GPUADMMTE is not None:
        solver_subparser.add_subcommand('gpuadmm', centralized_gpuadmm_solver_params_parser(), help='Options for the GPU accelerated ADMM solver')

    problem, args = parse_te_problem_description_args(parser)
    solver_subparser_args = getattr(args, args.solver)
    if args.solver == 'gurobi':
        GUROBI_PARAMS = parse_centralized_gurobi_solver_params(solver_subparser_args)
        solve_te_and_check(problem, GurobiTE, GUROBI_PARAMS)
    # elif args.solver == 'gurobi-dual':
    #     GUROBI_PARAMS = parse_centralized_dual_gurobi_solver_params(solver_subparser_args)
    #     solve_te_and_check(problem, DualGurobiTE, GUROBI_PARAMS)
    elif args.solver =='pdlp':
        PDLP_PARAMS = parse_centralized_pdlp_solver_params(solver_subparser_args)
        solve_te_and_check(problem, PDLPTE, PDLP_PARAMS)
    elif args.solver == 'gpuadmm':
        GPUADMM_PARAMS = parse_centralized_gpuadmm_solver_params(solver_subparser_args)
        solve_te_and_check(problem, GPUADMMTE, GPUADMM_PARAMS, PDLPMLU, PDLPMLUParams(Threads=1))
    else:
        raise ValueError(f'Invalid solver name: {args.solver}.\nAvailable solvers are: {list(AVAILABLE_SOLVERS.keys())}')
