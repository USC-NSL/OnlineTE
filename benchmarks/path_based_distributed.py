from te.algorithms.formulations.helper import *
from te.algorithms.formulations.online_te import *
from te.algorithms.formulations.path_based.distributed import *


if __name__ == '__main__':
    # Problem description parser
    parser = te_problem_description_parser('Path-Based Distributed TE')

    # Solver params parsers
    solver_subparser = parser.add_subcommands(dest='solver', help='The solver to use', required=True)
    solver_subparser.add_subcommand(
        "online_te", online_te_parser('Path-based OnlineTE', PathBasedOnlineTEParameters, False),
        help='Options for OnlineTE solver'
    )
    solver_subparser.add_subcommand(
        "simple", online_te_parser('Simplified path-based OnlineTE', PathBasedSimplifiedOnlineTEParameters, True),
        help='Options for simplified OnlineTE solver'
    )

    # Get TE problem description
    problem, args = parse_te_problem_description_args(parser)
    solver_subparser_args = getattr(args, args.solver)
    if args.solver == 'online_te':
        solver = parse_online_te_config(
            args=solver_subparser_args,
            solver_param_cls=PathBasedOnlineTEParameters,
            coordinator_cls=OnlineTECoordinator,
            worker_cls=OnlineTEWorkerNode,
            simplified=False
        )
        spawn_online_te_solver(problem, solver, solver_subparser_args.local)
    elif args.solver == 'simple':
        solver = parse_online_te_config(
            args=solver_subparser_args,
            solver_param_cls=PathBasedSimplifiedOnlineTEParameters,
            coordinator_cls=SimpleOnlineTECoordinator,
            worker_cls=SimpleOnlineTEWorkerNode,
            simplified=True
        )
        spawn_online_te_solver(problem, solver, solver_subparser_args.local)
    else:
        raise ValueError(f'Unknown solver implementation: {args.solver}')
