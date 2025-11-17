from te.algorithms.formulations.helper import *
from te.algorithms.formulations.edge_based.helper import *
from te.algorithms.formulations.edge_based.centralized.aggregate import *


if __name__ == '__main__':
    # Problem description parser
    parser = mlu_problem_description_parser('Edge-Based Centralized TE')
    # Solver params parsers
    solver_subparser = parser.add_subparsers(dest='solver', help='The solver to use', required=True)
    gurobi_params_parser = solver_subparser.add_parser('gurobi', help='Options for the Gurobi solver')
    centralized_gurobi_solver_params_parser(gurobi_params_parser)
    dual_gurobi_params_parser = solver_subparser.add_parser('gurobi-dual', help='Options for the Dual Gurobi solver')
    centralized_dual_gurobi_solver_params_parser(dual_gurobi_params_parser)
    pdlp_params_parser = solver_subparser.add_parser('pdlp', help='Options for the PDLP solver')
    centralized_pdlp_solver_params_parser(pdlp_params_parser)

    problem, args = parse_mlu_problem_description_args(parser)
    if args.solver == 'gurobi':
        GUROBI_PARAMS, _ = parse_centralized_gurobi_solver_params(gurobi_params_parser, args)
        solve_te_and_check(problem, GurobiTE, GUROBI_PARAMS)
    elif args.solver == 'gurobi-dual':
        GUROBI_PARAMS, _ = parse_centralized_dual_gurobi_solver_params(dual_gurobi_params_parser, args)
        solve_te_and_check(problem, DualGurobiTE, GUROBI_PARAMS)
    elif args.solver =='pdlp':
        PDLP_PARAMS, _ = parse_centralized_pdlp_solver_params(pdlp_params_parser, args)
        solve_te_and_check(problem, DualGurobiTE, PDLP_PARAMS)
    else:
        raise ValueError(f'Invalid solver name: {args.solver}.\nAvailable solvers are: {list(AVAILABLE_SOLVERS.keys())}')
