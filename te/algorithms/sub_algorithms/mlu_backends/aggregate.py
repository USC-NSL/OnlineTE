import jsonargparse
from typing import Tuple
from .base import ControllerMLUSolver, SolverParams
from .gurobi_mlu import GurobiMLUParams, GurobiMLU, add_gurobi_mlu_solver_params_parser, parse_gurobi_mlu_solver_params
from .pdlp_mlu import PDLPMLUParams, PDLPMLU, add_pdlp_mlu_solver_params_parser, parse_pdlp_mlu_solver_params


def add_mlu_backend_parser(parser: jsonargparse.ArgumentParser):
    parser.add_argument('--mlu_backend', choices=['gurobi', 'pdlp'], default='pdlp')
    add_gurobi_mlu_solver_params_parser(parser)
    add_pdlp_mlu_solver_params_parser(parser)


def parse_mlu_backend_params(args: jsonargparse.Namespace) -> Tuple[SolverParams, type[ControllerMLUSolver]]:
    """
    Parse all MLU backend arguments and output the parameters and class for
    the selected backend.

    Arguments
    ---------
    args: jsonargparse.Namespace
        The parsed arguments from the parser
    
    Returns
    -------
    MLU_PARAMS: SolverParams
        Appropriate solver parameters for the selected backend (i.e. 
        `gurobi`, `pdlp`, etc.)
    MLUCLS: type[ControllerMLUSolver]
        Appropriate solver class that implements the backend for the
        above parameters.
    """
    if args.mlu_backend == 'gurobi':
        GUROBI_MLU_PARAMS = parse_gurobi_mlu_solver_params(args)
        return GUROBI_MLU_PARAMS, GurobiMLU
    elif args.mlu_backend == 'pdlp':
        PDLP_MLU_PARAMS = parse_pdlp_mlu_solver_params(args)
        return PDLP_MLU_PARAMS, PDLPMLU
    else:
        raise ValueError(f'Unknown MLU backend name: {args.mlu_backend}')


__all__ = [
    'GurobiMLUParams',
    'PDLPMLUParams',
    'GurobiMLU',
    'PDLPMLU',
    'ControllerMLUSolver',
    'add_mlu_backend_parser', 'parse_mlu_backend_params'
]