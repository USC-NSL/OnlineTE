import argparse
from typing import Tuple, Optional, List
from .base import ControllerMLUSolver, SolverParams
from .gurobi_mlu import GurobiMLUParams, GurobiMLU
from .pdlp_mlu import PDLPMLUParams, PDLPMLU
from te.algorithms.formulations.edge_based.centralized import METHOD_MAP, METHOD_MAP_REVERSE


def add_mlu_backend_subparser(parser: argparse.ArgumentParser) -> List[argparse.ArgumentParser]:
    mlu_subparser = parser.add_subparsers(dest='mlu_backend', help='MLU backend to use', required=True)

    # NOTE: The `_Rho` and `_Alpha` fields of the backends are always set by the solver itself
    #       in general, a field that begins with an underscore shouldn't be assigned to directly by the user

    GUROBI_MLU_PARAMS = GurobiMLUParams()
    gurobi_mlu_solver_params_parser = mlu_subparser.add_parser('gurobi', help='Options for the Gurobi MLU backend')
    gurobi_mlu_solver_params_parser.add_argument('--method', help='Gurobi method to use', choices=list(METHOD_MAP.keys()), default=METHOD_MAP_REVERSE[GUROBI_MLU_PARAMS.Method])
    gurobi_mlu_solver_params_parser.add_argument('--focus', help='Gurobi numeric focus', type=int, choices=[0, 1, 2, 3], default=GUROBI_MLU_PARAMS.NumericFocus)
    gurobi_mlu_solver_params_parser.add_argument('--presolve', help='Perform presolve', action='store_true')
    gurobi_mlu_solver_params_parser.add_argument('--crossover', action='store_true', help='(BARRIER only) perform crossover after barrier solver ends')
    gurobi_mlu_solver_params_parser.add_argument('--threads', help='(BARRIER only) Max number of threads', type=int, default=GUROBI_MLU_PARAMS.Threads)
    gurobi_mlu_solver_params_parser.add_argument('--log-to', help='Log file path', default=GUROBI_MLU_PARAMS.LogFile)
    gurobi_mlu_solver_params_parser.add_argument('--conv-tol', help='Objective convergence tolerance', type=float, default=GUROBI_MLU_PARAMS.ConvTol)

    PDLP_MLU_PARAMS = PDLPMLUParams()
    pdlp_mlu_solver_params_parser = mlu_subparser.add_parser('pdlp', help='Options for the PDLP MLU backend')
    pdlp_mlu_solver_params_parser.add_argument('--threads', type=int, help='Number of threads to use for PDHG', default=PDLP_MLU_PARAMS.Threads)
    pdlp_mlu_solver_params_parser.add_argument('--presolve', help='Perform presolve', action='store_true')
    pdlp_mlu_solver_params_parser.add_argument('--conv-tol', help='Objective convergence tolerance', type=float, default=PDLP_MLU_PARAMS.ConvTol)

    return [gurobi_mlu_solver_params_parser, pdlp_mlu_solver_params_parser]


def parse_mlu_backend_params(
    args: Optional[argparse.Namespace] = None, 
    parser: Optional[argparse.ArgumentParser] = None
) -> Tuple[SolverParams, type[ControllerMLUSolver], argparse.Namespace]:
    if args is None:
        assert parser is not None
        args = parser.parse_args()

    if args.mlu_backend == 'gurobi':
        GUROBI_MLU_PARAMS = GurobiMLUParams()
        GUROBI_MLU_PARAMS.Method = METHOD_MAP[args.method]
        GUROBI_MLU_PARAMS.NumericFocus = args.focus
        GUROBI_MLU_PARAMS.Presolve = args.presolve
        GUROBI_MLU_PARAMS.Crossover = args.crossover
        GUROBI_MLU_PARAMS.Threads = args.threads
        GUROBI_MLU_PARAMS.LogFile = args.log_to
        GUROBI_MLU_PARAMS.ConvTol = args.conv_tol
        return GUROBI_MLU_PARAMS, GurobiMLU, args
    elif args.mlu_backend == 'pdlp':
        PDLP_MLU_PARAMS = PDLPMLUParams()
        PDLP_MLU_PARAMS.Threads = args.threads
        PDLP_MLU_PARAMS.Presolve = args.presolve
        PDLP_MLU_PARAMS.ConvTol = args.conv_tol
        return PDLP_MLU_PARAMS, PDLPMLU, args
    else:
        raise ValueError(f'Unknown MLU backend name: {args.mlu_backend}')


__all__ = [
    'GurobiMLUParams', 'PDLPMLUParams',
    'GurobiMLU', 'PDLPMLU',
    'METHOD_MAP', 'METHOD_MAP_REVERSE',
    'add_mlu_backend_subparser', 'parse_mlu_backend_params'
]