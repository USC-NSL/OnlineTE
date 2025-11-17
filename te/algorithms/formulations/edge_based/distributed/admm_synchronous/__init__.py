import argparse
import te.constants
from typing import Optional, Literal, Tuple
from dataclasses import dataclass
from te.algorithms.base import SolverParams
from te.algorithms.array_utils import SINGLE_PRECISION

import warnings
warnings.filterwarnings("error")
"""This is mostly to catch overflow, they can be devistating!"""

@dataclass
class SynchADMMSolverParams(SolverParams):
    NumberOfEpochs: Optional[int] = 100
    NumberOfNetworkUpdates: int = 3
    Rho: float = 1.0
    Eta: float = 0.5
    Gamma: float = 1.0
    Kappa: float = 0.01
    PGDIterations: int = 2
    ConvTol: float = 1e-3
    Precision: Literal['double', 'single', 'half'] = SINGLE_PRECISION
    TMSeed: int = te.constants.DEFAULT_SEED

    def __post_init__(self):
        self._left_column_share = 0.5


def distributed_synchronous_admm_solver_params_parser(parser: argparse.ArgumentParser):
    SYNCH_ADMM_PARAMS = SynchADMMSolverParams()
    parser.add_argument('--epochs', type=int, default=SYNCH_ADMM_PARAMS.NumberOfEpochs, 
                            help='Number of epochs')
    parser.add_argument('--updates', type=int, default=SYNCH_ADMM_PARAMS.NumberOfNetworkUpdates, 
                            help='Number of consecutive network updates')
    parser.add_argument('--rho', type=float, default=SYNCH_ADMM_PARAMS.Rho, 
                            help='Outer ADMM step size')
    parser.add_argument('--eta', type=float, default=SYNCH_ADMM_PARAMS.Eta, 
                            help='Inner ADMM step size')
    parser.add_argument('--gamma', type=float, default=SYNCH_ADMM_PARAMS.Gamma, 
                            help='Projected Gradient Descent step size')
    parser.add_argument('--kappa', type=float, default=SYNCH_ADMM_PARAMS.Kappa, 
                            help='Projected Gradient Descent step size reduction factor')
    parser.add_argument('--pgd-iters', type=int, default=SYNCH_ADMM_PARAMS.PGDIterations, 
                            help='Number of iterations for each of the inner loop PGD solvers per update')
    parser.add_argument('--precision', choices=['half', 'single', 'double'], default=SYNCH_ADMM_PARAMS.Precision,
                            help='Floating point operation precision')


def parse_distributed_synchronous_admm_solver_params(
    parser: argparse.ArgumentParser, 
    args: Optional[argparse.Namespace] = None
) -> Tuple[SynchADMMSolverParams, argparse.Namespace]:
    if args is None:
        args = parser.parse_args()
    
    SYNCH_ADMM_PARAMS = SynchADMMSolverParams()
    SYNCH_ADMM_PARAMS.NumberOfEpochs = args.epochs
    SYNCH_ADMM_PARAMS.NumberOfNetworkUpdates = args.updates
    SYNCH_ADMM_PARAMS.Rho = args.rho
    SYNCH_ADMM_PARAMS.Eta = args.eta
    SYNCH_ADMM_PARAMS.Gamma = args.gamma
    SYNCH_ADMM_PARAMS.Kappa = args.kappa
    SYNCH_ADMM_PARAMS.PGDIterations = args.pgd_iters
    SYNCH_ADMM_PARAMS.Precision = args.precision
    
    return SYNCH_ADMM_PARAMS, args
