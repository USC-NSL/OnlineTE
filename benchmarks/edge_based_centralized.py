from te.algorithms.formulations.helper_base import mlu_helper, mlu_argparser, mlu_parse_args
from te.algorithms.formulations.edge_based.centralized.aggregate import *


if __name__ == '__main__':
    parser = mlu_argparser('Edge-Based Centralized TE')
    
    solver_subparser = parser.add_subparsers(dest='solver', help='The solver to use', required=True)
    
    GUROBI_PARAMS = GurobiSolverParams()
    gurobi_params_parser = solver_subparser.add_parser('gurobi', help='Options for the Gurobi solver')
    gurobi_params_parser.add_argument('--method', help='Gurobi method to use', choices=list(METHOD_MAP.keys()), default=METHOD_MAP_REVERSE[GUROBI_PARAMS.Method])
    gurobi_params_parser.add_argument('--focus', help='Gurobi numeric focus', type=int, choices=[0, 1, 2, 3], default=GUROBI_PARAMS.NumericFocus)
    gurobi_params_parser.add_argument('--presolve', help='Perform presolve', action='store_true')
    gurobi_params_parser.add_argument('--crossover', action='store_true', help='(BARRIER only) perform crossover after barrier solver ends')
    gurobi_params_parser.add_argument('--threads', help='(BARRIER only) Max number of threads', type=int, default=GUROBI_PARAMS.Threads)
    gurobi_params_parser.add_argument('--log-to', help='Log file path', default=GUROBI_PARAMS.LogFile)

    dual_gurobi_params_parser = solver_subparser.add_parser('gurobi-dual', help='Options for the Dual Gurobi solver')
    dual_gurobi_params_parser.add_argument('--method', help='Gurobi method to use', choices=list(METHOD_MAP.keys()), default=METHOD_MAP_REVERSE[GUROBI_PARAMS.Method])
    dual_gurobi_params_parser.add_argument('--focus', help='Gurobi numeric focus', type=int, choices=[0, 1, 2, 3], default=GUROBI_PARAMS.NumericFocus)
    dual_gurobi_params_parser.add_argument('--presolve', help='Perform presolve', action='store_true')
    dual_gurobi_params_parser.add_argument('--crossover', action='store_true', help='(BARRIER only) perform crossover after barrier solver ends')
    dual_gurobi_params_parser.add_argument('--threads', help='(BARRIER only) Max number of threads', type=int, default=GUROBI_PARAMS.Threads)
    dual_gurobi_params_parser.add_argument('--log-to', help='Log file path', default=GUROBI_PARAMS.LogFile)

    PDLP_PARAMS = PDLPParams()
    pdlp_params_parser = solver_subparser.add_parser('pdlp', help='Options for the PDLP solver')
    pdlp_params_parser.add_argument('--threads', type=int, help='Number of threads to use for PDHG', default=PDLP_PARAMS.Threads)
    pdlp_params_parser.add_argument('--presolve', help='Perform presolve', action='store_true')

    eval_params, solution_params, warm_start_params, args = mlu_parse_args(parser)

    if args.solver == 'gurobi' or args.solver == 'gurobi-dual':
        GUROBI_PARAMS.Method = METHOD_MAP[args.method]
        GUROBI_PARAMS.NumericFocus = args.focus
        GUROBI_PARAMS.FeasibilityTol = args.feas_tol
        GUROBI_PARAMS.ConvTol = args.conv_tol
        GUROBI_PARAMS.Presolve = args.presolve
        GUROBI_PARAMS.Crossover = args.crossover
        GUROBI_PARAMS.Threads = args.threads
        GUROBI_PARAMS.LogFile = args.log_to
    elif args.solver =='pdlp':
        PDLP_PARAMS.Threads = args.threads
        PDLP_PARAMS.FeasibilityTol = args.feas_tol
        PDLP_PARAMS.ConvTol = args.conv_tol
        PDLP_PARAMS.Presolve = args.presolve
    else:
        raise ValueError(f'Invalid solver name: {args.solver}.\nAvailable solvers are: {list(AVAILABLE_SOLVERS.keys())}')

    if args.solver == 'gurobi':
        mlu_helper(GurobiTE, GUROBI_PARAMS, eval_params, warm_start_params, solution_params)
    elif args.solver == 'gurobi-dual':
        mlu_helper(DualGurobiTE, GUROBI_PARAMS, eval_params, warm_start_params, solution_params)
    else:
        assert args.solver == 'pdlp'
        mlu_helper(PDLPTE, PDLP_PARAMS, eval_params, warm_start_params, solution_params)
