from te.algorithms.formulations.edge_based.helper import *
# TODO: Replace this with the `distributed.aggregate import *` when that file is ready ...
from te.algorithms.formulations.edge_based.distributed.admm_synchronous.helper import *


if __name__ == '__main__':
    # Problem description parser
    parser = te_problem_description_parser('Edge-Based Distributed TE')
    # Solver params parsers
    solver_subparser = parser.add_subcommands(dest='solver', help='The solver to use', required=True)
    solver_subparser.add_subcommand("synch", distributed_synchronous_admm_parser(), help='Options for the synchronous solver')
    # TODO: Add 'hierarchical' solver

    # Get TE problem description
    problem, args = parse_te_problem_description_args(parser)
    solver_subparser_args = getattr(args, args.solver)
    if args.solver == 'synch':
        solver = parse_distributed_synchronous_admm(solver_subparser_args)
        spawn_distributed_synchronous_solver(problem, solver, solver_subparser_args.local)
    else:
        raise ValueError(f'Unknown solver implementation: {args.solver}')
