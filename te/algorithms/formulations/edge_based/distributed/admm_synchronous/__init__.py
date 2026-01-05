import jsonargparse
import te.constants
from typing import Optional, Literal
from dataclasses import dataclass
from te.algorithms.base import SolverParams
from te.algorithms.array_utils import SINGLE_PRECISION
from utils.logging import as_warning

import warnings
warnings.filterwarnings("error")
"""This is mostly to catch overflow, they can be devistating!"""

@dataclass
class SynchADMMSolverParams(SolverParams):
    OuterLoopRounds: Optional[int] = 100
    """Number of outer loop iterations"""
    InnerLoopRounds: int = 3
    """Number of inner loop iterations"""
    Rho: float = 1.0
    """Outer ADMM step size"""
    Eta: float = 0.2
    """Inner ADMM step size"""
    Gamma: float = 1.0
    """Step size for solving the switch-level problems"""
    Beta: Optional[float] = None
    """
    L1 norm penalty coefficient for sparsity.
    When `None`, a PGD algorithm on a dense assignment matrix is
    used to solve inner loop problems.
    If not `None`, then an alternating shrinkage algorithm is
    used to solve the inner loop problems instead.
    """
    UseSparseBasis: bool = False
    """Use a sparse null space basis, but sacrifice orthonormality"""
    SwitchIterations: int = 2
    """Number of iterations for each switch-level problem"""
    ConvTol: float = 1e-3
    """Objective convergence tolerance"""
    Precision: Literal['double', 'single', 'half'] = SINGLE_PRECISION
    """Floating point operation precision"""
    TMSeed: int = te.constants.DEFAULT_SEED
    """Traffic matrix RNG seed"""

    def __post_init__(self):
        self._left_column_share = 0.5
        if self.Beta is not None:
            assert self.Beta > 0, "L1 penalty coefficient must be strictly greater than 0"
        if self.Rho > self.Eta:
            as_warning(f"Outer ADMM step size (`Rho`) = {self.Rho} is strictly larger "
                       f"than inner ADMM step size (`Eta`) = {self.Eta}.\nThis is almost never beneficial.")


def add_synch_solver_params_parser(parser: jsonargparse.ArgumentParser):
    parser.add_class_arguments(SynchADMMSolverParams, 'SolverParams', help='Synchronous ADMM Algorithm Parameters')

def parse_synch_solver_params(args: jsonargparse.Namespace) -> SynchADMMSolverParams:
    return SynchADMMSolverParams.make_from_args(args.SolverParams)
