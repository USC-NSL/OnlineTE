"""
A helper sript for spawning a distributed solver using our
nested ADMM backbone. It parses and stores:
- Algorithm Parameters
- MLU solver backend parameters
- Communication backend parameters
"""


import jsonargparse
import multiprocessing
import concurrent.futures
from dataclasses import dataclass
from typing import List, Optional
from utils.logging import as_info, as_warning, log_section_title
from te.algorithms.base import *
from te.algorithms.communication.base import *
from te.algorithms.communication.helper import *
from te.algorithms.formulations.helper import *
from te.algorithms.sub_algorithms.mlu_backends.aggregate import *


@dataclass
class OnlineTESolverDescription:
    """
    Helper Dataclass for a solver on top of the OnlineTE backbone.
    This completely describes the nested ADMM solver, and once combined
    with the problem description, can be used to spawn a full instance
    of a solver and see how it works.
    """
    AlgorithmParams: SolverParams
    MasterCLS: type[TELP]
    MasterBackendCLS: type[CommunicationBackendBase]
    MasterRPCParams: RPCParams
    MLUCLS: type[ControllerMLUSolver]
    MLUParams: SolverParams
    WorkerCLS: type[DistributedSolverNodeBase]
    WorkerBackendCLS: type[CommunicationBackendBase]
    WorkerRPCParamList: List[RPCParams]


def online_te_parser(
    name: str,
    solver_param_cls: type[SolverParams]
) -> jsonargparse.ArgumentParser:
    parser = jsonargparse.ArgumentParser(name)
    parser.add_class_arguments(solver_param_cls, 'SolverParams', help='Algorithm parameters')
    add_mlu_backend_parser(parser)
    single_controller_topology_address_parser(parser)
    add_communication_backend_params_parser(parser)
    return parser


def parse_online_te_config(
    args: jsonargparse.Namespace,
    solver_param_cls: type[SolverParams],
    coordinator_cls: type[TELP],
    worker_cls: type[DistributedSolverNodeBase]
) -> OnlineTESolverDescription:
    solver_params = solver_param_cls.make_from_args(args.SolverParams)
    mlu_backend_params, mlu_backend_cls = parse_mlu_backend_params(args)
    master_addr, worker_addr_list = parse_single_controller_topology_address_parser(args)
    comm_backend_description = parse_communication_backend_params(master_addr, worker_addr_list, args)
    return OnlineTESolverDescription(
        AlgorithmParams=solver_params,
        MasterCLS=coordinator_cls,
        MasterBackendCLS=comm_backend_description.ControllerBackendCLS,
        MasterRPCParams=comm_backend_description.ControllerBackendParams,
        MLUCLS=mlu_backend_cls,
        MLUParams=mlu_backend_params,
        WorkerCLS=worker_cls,
        WorkerBackendCLS=comm_backend_description.WorkerBackendCLS,
        WorkerRPCParamList=comm_backend_description.WorkerBackendParams
    )


def spawn_online_te_solver(
    problem: TEProblemDescription, 
    solver: OnlineTESolverDescription,
    is_local: bool = False
) -> Optional[TETracer]:
    def _spawn_online_te_solver() -> Optional[TETracer]:
        print(as_info(
            f'Using master node communication backend `{solver.MasterBackendCLS.backend_name()}` with parameters:\n'+
            solver.MasterRPCParams.str_all()
        ))
        node_params = DistributedSolverNodeParams(
            CommunicationBackendCLS=solver.MasterBackendCLS,
            RPCParams_=solver.MasterRPCParams
        )
        return solve_te_and_check(
            problem, 
            solver.MasterCLS,
            solver.AlgorithmParams,
            node_params,
            solver.MLUCLS,
            solver.MLUParams
        )

    num_workers = len(solver.MasterRPCParams.Workers)
    # Number of workers that we want to spawn must match the number of workers
    # controlled by our controller node.
    assert len(solver.WorkerRPCParamList) == num_workers
    if is_local:
        print(as_warning(log_section_title("LOCAL EXPERIMENT")))
        print(as_info(
            f'A total of {num_workers} worker nodes will be spawned with addresses:\n'+
            PrettyAddressList(tuple([p.Peers[0] for p in solver.WorkerRPCParamList])).str_all()
        ))
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers, 
            mp_context=multiprocessing.get_context(method='spawn')
        ) as network_pool:
            # Worker nodes need no problem description and solver inputs, the
            # central controller will tell them all they need.
            for worker_rpc_params in solver.WorkerRPCParamList:
                network_pool.submit(
                    solver.WorkerCLS.spawn_and_run, 
                    DistributedSolverNodeParams(
                        CommunicationBackendCLS=solver.WorkerBackendCLS,
                        RPCParams_=worker_rpc_params
                    )
                )
            return _spawn_online_te_solver()
    else:
        return _spawn_online_te_solver()


__all__ = [
    'online_te_parser',
    'parse_online_te_config',
    'OnlineTESolverDescription',
    'spawn_online_te_solver'
]