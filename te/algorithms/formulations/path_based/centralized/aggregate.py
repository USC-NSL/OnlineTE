from typing import Tuple, Dict
from te.algorithms.base import SolverParams, TrafficEngineeringLP
from . import GurobiPathBasedSolverParams, PDLPPathBasedSolverParams
from .gurobi import GurobiPathBasedTE, centralized_gurobi_solver_params_parser, parse_centralized_gurobi_solver_params
from .pdlp import PDLPPathBasedTE, centralized_pdlp_solver_params_parser, parse_centralized_pdlp_solver_params


AVAILABLE_SOLVERS: Dict[str, Tuple[type[TrafficEngineeringLP], type[SolverParams]]] = {
    'gurobi': (GurobiPathBasedTE, GurobiPathBasedSolverParams),
    'pdlp': (PDLPPathBasedTE, PDLPPathBasedSolverParams),
    # 'gurobi-dual': (DualGurobiTE, GurobiSolverParams)
}
"""
Avaialble solvers are:
    - **gurobi**: (`GurobiTE`, `GurobiSolverParams`)
    - **pdlp**: (`PDLPTE`, `PDLPParams`)
    - **gurobi-dual**: (`DualGurobiTE`, `GurobiSolverParams`)
"""


__all__ = [
    'AVAILABLE_SOLVERS',
    'GurobiPathBasedTE', 'GurobiPathBasedSolverParams', 'centralized_gurobi_solver_params_parser', 'parse_centralized_gurobi_solver_params',
    'PDLPPathBasedTE', 'PDLPPathBasedSolverParams', 'centralized_pdlp_solver_params_parser', 'parse_centralized_pdlp_solver_params',
    # 'DualGurobiTE', 'centralized_dual_gurobi_solver_params_parser', 'parse_centralized_dual_gurobi_solver_params'
]