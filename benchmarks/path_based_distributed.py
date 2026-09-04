from te.algorithms.formulations.helper import *
from te.algorithms.formulations.online_te import *
from te.algorithms.formulations.path_based.distributed import *


if __name__ == '__main__':
    # Problem description parser
    parser = te_problem_description_parser('Path-Based Distributed TE')

    # Solver params parsers
    solver_subparser = parser.add_subcommands(dest='solver', help='The solver to use', required=True)
    solver_subparser.add_subcommand(
        "online_te", online_te_parser('Path-based OnlineTE', PathBasedOnlineTEParameters),
        help='Options for OnlineTE solver'
    )

    # Get TE problem description
    problem, args = parse_te_problem_description_args(parser)
    solver_subparser_args = getattr(args, args.solver)
    if args.solver == 'online_te':
        solver = parse_online_te_config(
            args=solver_subparser_args,
            solver_param_cls=PathBasedOnlineTEParameters,
            coordinator_cls=OnlineTECoordinator,
            worker_cls=OnlineTEWorkerNode
        )
        spawn_online_te_solver(problem, solver, solver_subparser_args.local)
    else:
        raise ValueError(f'Unknown solver implementation: {args.solver}')
