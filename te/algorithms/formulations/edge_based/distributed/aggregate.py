from typing import Tuple, Dict
from te.algorithms.base import SolverParams
from .base import DistributedSolverNodeBase
from .admm_synchronous import (SynchADMMSolverParams, parse_distributed_synchronous_admm_solver_params, 
                               distributed_synchronous_admm_solver_params_parser)
from .admm_synchronous.controller import SynchADMMControllerNode
from .admm_synchronous.worker import SynchADMMWorkerNode
from .admm_synchronous.helper import add_admm_synch_communication_backend_subparser, parse_add_admm_synch_communication_backend_params
from .admm_hierarchical import HierarchicalADMMSolverParams
from .admm_hierarchical.master import MasterNode
from .admm_hierarchical.domain import DomainControllerNode
from .admm_hierarchical.worker import DomainWorkerNode
from .helper import (single_controller_multiprocess_mlu_helper, distributed_mlu_helper, distributed_mlu_argparser, 
                     distributed_mlu_parse_args, hierarchical_multiprocess_mlu_helper,
                     DistributedMLUHelperParams, SingleControllerMultiprocessMLUHelperParams, HierarchicalMultiprocessMLUHelperParams)


DistributedSolver = Tuple[
    type[DistributedSolverNodeBase], 
    type[DistributedSolverNodeBase], 
    type[SolverParams]
]

HierarchicalSolver = Tuple[
    type[DistributedSolverNodeBase], 
    type[DistributedSolverNodeBase],
    type[DistributedSolverNodeBase], 
    type[SolverParams]
]


AVAILABLE_SOLVERS: Dict[str, DistributedSolver] = {
    'admm-synch': (SynchADMMControllerNode, SynchADMMWorkerNode, SynchADMMSolverParams),
    'admm-hier':  (MasterNode, DomainControllerNode, DomainWorkerNode, HierarchicalADMMSolverParams)
}
"""
Avaialble solvers are:
    - **admm-synch**: (`SynchADMMControllerNode`, `SynchADMMWorkerNode`, `SynchADMMSolverParams`)
"""


__all__ = [
    'AVAILABLE_SOLVERS',
    'SynchADMMControllerNode', 'SynchADMMWorkerNode', 
    'parse_distributed_synchronous_admm_solver_params', 'distributed_synchronous_admm_solver_params_parser',
    'MasterNode', 'DomainControllerNode', 'DomainWorkerNode',
    'SynchADMMSolverParams', 'HierarchicalADMMSolverParams',
    # Generic solver function
    'distributed_mlu_helper',
    # Local version of the distributed solver
    'single_controller_multiprocess_mlu_helper', 
    # Local version of the hierarchical solver
    'hierarchical_multiprocess_mlu_helper',
    # Argument parsers
    'distributed_mlu_argparser', 'distributed_mlu_parse_args',
    'add_admm_synch_communication_backend_subparser', 'parse_add_admm_synch_communication_backend_params',
    # Parameter bundles
    'DistributedMLUHelperParams',
    'SingleControllerMultiprocessMLUHelperParams',
    'HierarchicalMultiprocessMLUHelperParams'   
]