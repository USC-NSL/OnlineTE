# import argparse
import jsonargparse
import te.constants
from typing import Optional, Literal
from dataclasses import dataclass
from te.algorithms.base import SolverParams
from te.algorithms.array_utils import SINGLE_PRECISION

import warnings
warnings.filterwarnings("error")
"""This is mostly to catch overflow, they can be devistating!"""

@dataclass
class SynchADMMSolverParams(SolverParams):
    NumberOfEpochs: Optional[int] = 100
    """Number of epochs"""
    NumberOfNetworkUpdates: int = 3
    """Number of consecutive network updates"""
    Rho: float = 1.0
    """Outer ADMM step size"""
    Eta: float = 0.5
    """Inner ADMM step size"""
    Gamma: float = 1.0
    """Projected Gradient Descent step size"""
    Kappa: float = 0.01
    """Projected Gradient Descent step size reduction factor"""
    PGDIterations: int = 2
    """Number of iterations for each of the inner loop PGD solvers per update"""
    ConvTol: float = 1e-3
    """Objective convergence tolerance"""
    Precision: Literal['double', 'single', 'half'] = SINGLE_PRECISION
    """Floating point operation precision"""
    TMSeed: int = te.constants.DEFAULT_SEED
    """Traffic matrix RNG seed"""

    def __post_init__(self):
        self._left_column_share = 0.5


def add_synch_solver_params_parser(parser: jsonargparse.ArgumentParser):
    parser.add_class_arguments(SynchADMMSolverParams, 'SolverParams', help='Synchronous ADMM Algorithm Parameters')

def parse_synch_solver_params(args: jsonargparse.Namespace) -> SynchADMMSolverParams:
    return SynchADMMSolverParams.make_from_args(args.SolverParams)
