from typing import Tuple, Dict
from te.algorithms.base import SolverParams, TrafficEngineeringLP
from . import METHOD_MAP, METHOD_MAP_REVERSE, GurobiSolverParams, PDLPParams
from .gurobi import GurobiTE
from .pdlp import PDLPTE
from .dual_gurobi import DualGurobiTE


AVAILABLE_SOLVERS: Dict[str, Tuple[type[TrafficEngineeringLP], type[SolverParams]]] = {
    'gurobi': (GurobiTE, GurobiSolverParams),
    'pdlp': (PDLPTE, PDLPParams),
    'gurobi-dual': (DualGurobiTE, GurobiSolverParams)
}
"""
Avaialble solvers are:
    - **gurobi**: (`GurobiTE`, `GurobiSolverParams`)
    - **pdlp**: (`PDLPTE`, `PDLPParams`)
    - **gurobi-dual**: (`DualGurobiTE`, `GurobiSolverParams`)
"""


__all__ = [
    'AVAILABLE_SOLVERS',
    'METHOD_MAP', 'METHOD_MAP_REVERSE',
    'GurobiTE', 'GurobiSolverParams',
    'PDLPTE', 'PDLPParams',
    'DualGurobiTE'
]