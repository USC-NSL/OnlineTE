import jsonargparse
import te.constants
from typing import Optional, Literal
from dataclasses import dataclass
from te.algorithms.base import SolverParams
from te.algorithms.array_utils import SINGLE_PRECISION
from numba.core.errors import NumbaTypeSafetyWarning

import warnings
warnings.filterwarnings("error")
# TODO: This warning is raised in all of the sparse algorithms under
#       `sub_algorithms/paths` for handling path mask multiplication
#       and edge-based mean. See how we can resolve it.
warnings.simplefilter('ignore', category=NumbaTypeSafetyWarning)
"""This is mostly to catch overflow, they can be devistating!"""

@dataclass
class SynchADMMSolverParams(SolverParams):
    NumberOfPathsPerCommodity: int = 16
    """Max number of available paths for each commodity"""
    OuterLoopRounds: Optional[int] = 200
    """Number of outer loop iterations"""
    InnerLoopRounds: int = 5
    """Number of inner loop iterations"""
    Rho: float = 1.0
    """Outer ADMM step size"""
    Eta: float = 0.2
    """Inner ADMM step size"""
    Gamma: float = 0.1
    """Step size for solving the switch-level problems"""
    SwitchIterations: int = 5
    """Number of iterations for each switch-level problem"""
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
