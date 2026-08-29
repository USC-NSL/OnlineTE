from te.algorithms.formulations.helper import *
from te.algorithms.formulations.path_based.centralized.aggregate import *


if __name__ == '__main__':
    # Problem description parser
    parser = te_problem_description_parser('Path-Based Centralized TE')
    # Solver params parsers
    solver_subparser = parser.add_subcommands(dest='solver', help='The solver to use', required=True)
    solver_subparser.add_subcommand('gurobi', centralized_gurobi_solver_params_parser(), help='Options for the Gurobi solver')
    # solver_subparser.add_subcommand('pdlp', centralized_pdlp_solver_params_parser(), help='Options for the PDLP solver')

    problem, args = parse_te_problem_description_args(parser)
    solver_subparser_args = getattr(args, args.solver)
    if args.solver == 'gurobi':
        GUROBI_PARAMS = parse_centralized_gurobi_solver_params(solver_subparser_args)
        solve_te_and_check(problem, GurobiPathBasedTE, GUROBI_PARAMS)
    # elif args.solver == 'pdlp':
    #     PDLP_PARAMS = parse_centralized_pdlp_solver_params(solver_subparser_args)
    #     solve_te_and_check(problem, PDLPPathBasedTE, PDLP_PARAMS)
    else:
        raise ValueError(f'Invalid solver name: {args.solver}.\nAvailable solvers are: {list(AVAILABLE_SOLVERS.keys())}')
