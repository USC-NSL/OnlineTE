from typing import Tuple, Dict
from te.algorithms.base import SolverParams
from . import ControllerRPCParams, WorkerRPCParams
from .base import ControllerNodeBase, WorkerNodeBase, ControllerNodeParams, WorkerNodeParams
from .admm_synchronous import SynchADMMSolverParams
from .admm_synchronous.controller import SynchADMMControllerNode
from .admm_synchronous.worker import SynchADMMWorkerNode
from .admm_synchronous.helper import add_admm_synch_communication_backend_subparser, parse_add_admm_synch_communication_backend_params
from .helper import (multiprocess_mlu_helper, distributed_mlu_helper, distributed_mlu_argparser, distributed_mlu_parse_args, 
                     MultiprocessMLUHelperParams, DistributedMLUHelperParams)


DistributedSolver = Tuple[type[ControllerNodeBase], type[WorkerNodeBase], type[ControllerRPCParams], type[WorkerRPCParams], type[SolverParams]]


AVAILABLE_SOLVERS: Dict[str, DistributedSolver] = {
    'admm-synch': (SynchADMMControllerNode, SynchADMMWorkerNode, ControllerRPCParams, WorkerRPCParams, SynchADMMSolverParams)
}
"""
Avaialble solvers are:
    - **admm-synch**: (`SynchADMMControllerNode`, `SynchADMMWorkerNode`, `ControllerRPCParams`, `WorkerRPCParams`, `SynchADMMSolverParams`)
"""


__all__ = [
    'AVAILABLE_SOLVERS',
    'SynchADMMControllerNode', 'SynchADMMWorkerNode', 
    'ControllerNodeParams', 'WorkerNodeParams',
    'ControllerRPCParams', 'WorkerRPCParams', 'SynchADMMSolverParams',
    'multiprocess_mlu_helper', 'distributed_mlu_helper',
    'distributed_mlu_argparser', 'distributed_mlu_parse_args',
    'add_admm_synch_communication_backend_subparser', 'parse_add_admm_synch_communication_backend_params',
    'MultiprocessMLUHelperParams', 'DistributedMLUHelperParams'
]