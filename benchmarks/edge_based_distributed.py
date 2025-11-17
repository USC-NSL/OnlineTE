from te.algorithms.formulations.helper import *
from te.algorithms.formulations.edge_based.helper import *
from te.algorithms.sub_algorithms.mlu_backends.aggregate import *
from te.algorithms.formulations.edge_based.distributed.aggregate import *


if __name__ == '__main__':
    # Problem description parser
    parser = distributed_mlu_argparser('Edge-Based Distributed TE')
    # Solver params parsers
    solver_subparser = parser.add_subparsers(dest='solver', help='The solver to use', required=True)
    synchronous_solver_params_subparser = solver_subparser.add_parser('synch', help='Options for the synchronous solver')
    distributed_synchronous_admm_solver_params_parser(synchronous_solver_params_subparser)

    for upto_mlu_backend_parser in add_mlu_backend_subparser(synchronous_solver_params_subparser):
        add_admm_synch_communication_backend_subparser(upto_mlu_backend_parser)

    num_workers, addr_list, problem, args = distributed_mlu_parse_args(parser)
    if args.solver == 'synch':
        ALGORITHM_SOLVER_PARAMS, _ = parse_distributed_synchronous_admm_solver_params(synchronous_solver_params_subparser, args)
        CONTROLLER_CLS = SynchADMMControllerNode
        WORKER_CLS = SynchADMMWorkerNode
    else:
        raise ValueError(f'Unknown solver name {args.solver}')
    
    MLU_BACKEND_PARAMS, MLU_BACKEND_CLS, _ = parse_mlu_backend_params(args)

    CONTROLLER_RPC_PARAMS, CONTROLLER_COMMUNICATION_CLS, \
    WORKER_RPC_PARAMS, WORKER_COMMUNICATION_CLS, _ = \
        parse_add_admm_synch_communication_backend_params(num_workers, addr_list, args)
    
    if args.local:
        single_controller_multiprocess_mlu_helper(
            problem=problem,
            distributed_solver=SingleControllerMultiprocessMLUHelperParams(
                MasterCLS=CONTROLLER_CLS,
                MasterBackendCLS=CONTROLLER_COMMUNICATION_CLS,
                MasterRPCParams=CONTROLLER_RPC_PARAMS,
                MLUCLS=MLU_BACKEND_CLS,
                MLUParams=MLU_BACKEND_PARAMS, 
                WorkerCLS=WORKER_CLS,
                WorkerBackendCLS=WORKER_COMMUNICATION_CLS,
                WorkerRPCParamList=WORKER_RPC_PARAMS
            ),
            solver_params=ALGORITHM_SOLVER_PARAMS
        )
    else:
        distributed_mlu_helper(
            problem=problem,
            distributed_solver=DistributedMLUHelperParams(
                MasterCLS=CONTROLLER_CLS,
                MasterBackendCLS=CONTROLLER_COMMUNICATION_CLS,
                MasterRPCParams=CONTROLLER_RPC_PARAMS,
                MLUCLS=MLU_BACKEND_CLS,
                MLUParams=MLU_BACKEND_PARAMS
            ),
            solver_params=ALGORITHM_SOLVER_PARAMS
        )
