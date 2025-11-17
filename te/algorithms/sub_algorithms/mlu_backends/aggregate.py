import argparse
from typing import Tuple, Optional, List
from .base import ControllerMLUSolver, SolverParams
from .gurobi_mlu import GurobiMLUParams, GurobiMLU, gurobi_mlu_solver_params_parser, parse_gurobi_mlu_solver_params
from .pdlp_mlu import PDLPMLUParams, PDLPMLU, pdlp_mlu_solver_params_parser, parse_pdlp_mlu_solver_params
from te.algorithms.formulations.edge_based.centralized import METHOD_MAP, METHOD_MAP_REVERSE


def add_mlu_backend_subparser(parser: argparse.ArgumentParser) -> List[argparse.ArgumentParser]:
    mlu_subparser = parser.add_subparsers(dest='mlu_backend', help='MLU backend to use', required=True)

    # NOTE: The `_Rho` and `_Alpha` fields of the backends are always set by the solver itself
    #       in general, a field that begins with an underscore shouldn't be assigned to directly by the user

    gurobi_mlu_solver_params_subparser = mlu_subparser.add_parser('gurobi', help='Options for the Gurobi MLU backend')
    gurobi_mlu_solver_params_parser(gurobi_mlu_solver_params_subparser)
    pdlp_mlu_solver_params_subparser = mlu_subparser.add_parser('pdlp', help='Options for the PDLP MLU backend')
    pdlp_mlu_solver_params_parser(pdlp_mlu_solver_params_subparser)

    return [gurobi_mlu_solver_params_subparser, pdlp_mlu_solver_params_subparser]


def parse_mlu_backend_params(
    args: Optional[argparse.Namespace] = None, 
    parser: Optional[argparse.ArgumentParser] = None
) -> Tuple[SolverParams, type[ControllerMLUSolver], argparse.Namespace]:
    """
    Parse all MLU backend arguments and output the parameters and class for
    the selected backend.

    Arguments
    ---------
    args: Optional[argparse.Namespace] = Non
        The parsed arguments from the parser
    parser: Optional[argparse.ArgumentParser] = None
        The full parser object, which we use to parse the arguments
    
    Returns
    -------
    MLU_PARAMS: SolverParams
        Appropriate solver parameters for the selected backend (i.e. 
        `gurobi`, `pdlp`, etc.)
    MLUCLS: type[ControllerMLUSolver]
        Appropriate solver class that implements the backend for the
        above parameters.
    args: argparse.Namespace
        Remaning parsed arguments (will be the same as `args` if it is
        not `None`)
    
    Notes
    -----
    If `args` and `parser` are both `None`, this raises an `AssertionError`. 
    Specify `args` to prevent the arguments from being parsed again.
    """
    if args is None:
        assert parser is not None
        args = parser.parse_args()

    if args.mlu_backend == 'gurobi':
        GUROBI_MLU_PARAMS, _ = parse_gurobi_mlu_solver_params(parser, args)
        return GUROBI_MLU_PARAMS, GurobiMLU, args
    elif args.mlu_backend == 'pdlp':
        PDLP_MLU_PARAMS, _ = parse_pdlp_mlu_solver_params(parser, args)
        return PDLP_MLU_PARAMS, PDLPMLU, args
    else:
        raise ValueError(f'Unknown MLU backend name: {args.mlu_backend}')


__all__ = [
    'GurobiMLUParams', 'PDLPMLUParams',
    'GurobiMLU', 'PDLPMLU',
    'METHOD_MAP', 'METHOD_MAP_REVERSE',
    'add_mlu_backend_subparser', 'parse_mlu_backend_params'
]