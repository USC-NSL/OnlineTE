from typing import Tuple, Dict
from te.algorithms.base import SolverParams, TELP
from . import GurobiSolverParams, PDLPParams, GPUParams
from .gurobi import GurobiTE, centralized_gurobi_solver_params_parser, parse_centralized_gurobi_solver_params
# from .pdlp import PDLPTE, centralized_pdlp_solver_params_parser, parse_centralized_pdlp_solver_params
# from .dual_gurobi import DualGurobiTE, centralized_dual_gurobi_solver_params_parser, parse_centralized_dual_gurobi_solver_params

# try:
#     from .admm_gpu import GPUADMMTE, centralized_gpuadmm_solver_params_parser, parse_centralized_gpuadmm_solver_params
# except ModuleNotFoundError:
#     GPUADMMTE = None
#     def centralized_gpuadmm_solver_params_parser(*args, **kwargs):
#         raise ValueError("No GPU/CUDA backend available")
#     def parse_centralized_gpuadmm_solver_params(*args, **kwargs):
#         raise ValueError("No GPU/CUDA backend available")


AVAILABLE_SOLVERS: Dict[str, Tuple[type[TELP], type[SolverParams]]] = {
    'gurobi': (GurobiTE, GurobiSolverParams),
    # 'pdlp': (PDLPTE, PDLPParams),
    # 'gurobi-dual': (DualGurobiTE, GurobiSolverParams),
    # 'gpuadmm': (GPUADMMTE, GPUParams)
}
"""
Avaialble solvers are:
    - **gurobi**: (`GurobiTE`, `GurobiSolverParams`)
    - **pdlp**: (`PDLPTE`, `PDLPParams`)
    - **gurobi-dual**: (`DualGurobiTE`, `GurobiSolverParams`)
    - **gpuadmm**: ('GPUADMMTE`, `GPUParams`)
"""


__all__ = [
    'AVAILABLE_SOLVERS',
    'GurobiTE', 'GurobiSolverParams', 'centralized_gurobi_solver_params_parser', 'parse_centralized_gurobi_solver_params',
    # 'PDLPTE', 'PDLPParams', 'centralized_pdlp_solver_params_parser', 'parse_centralized_pdlp_solver_params',
    # 'DualGurobiTE', 'centralized_dual_gurobi_solver_params_parser', 'parse_centralized_dual_gurobi_solver_params',
    # 'GPUADMMTE', 'centralized_gpuadmm_solver_params_parser', 'parse_centralized_gpuadmm_solver_params'
]