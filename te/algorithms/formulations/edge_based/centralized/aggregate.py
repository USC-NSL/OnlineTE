from typing import Tuple, Dict
from te.algorithms.base import SolverParams, TrafficEngineeringLP
from . import METHOD_MAP, METHOD_MAP_REVERSE, GurobiSolverParams, PDLPParams
from .gurobi import GurobiTE, centralized_gurobi_solver_params_parser, parse_centralized_gurobi_solver_params
from .pdlp import PDLPTE, centralized_pdlp_solver_params_parser, parse_centralized_pdlp_solver_params
from .dual_gurobi import DualGurobiTE, centralized_dual_gurobi_solver_params_parser, parse_centralized_dual_gurobi_solver_params


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
    'GurobiTE', 'GurobiSolverParams', 'centralized_gurobi_solver_params_parser', 'parse_centralized_gurobi_solver_params',
    'PDLPTE', 'PDLPParams', 'centralized_pdlp_solver_params_parser', 'parse_centralized_pdlp_solver_params',
    'DualGurobiTE', 'centralized_dual_gurobi_solver_params_parser', 'parse_centralized_dual_gurobi_solver_params'
]