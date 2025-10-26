from te.algorithms.sub_algorithms.mlu_backends.aggregate import *
from te.algorithms.formulations.edge_based.distributed.aggregate import *


if __name__ == '__main__':
    parser = distributed_mlu_argparser('Edge-Based Distributed TE')

    solver_subparser = parser.add_subparsers(dest='solver', help='The solver to use', required=True)

    SYNCH_ADMM_PARAMS = SynchADMMSolverParams()
    synchronous_solver_params_parser = solver_subparser.add_parser('synch', help='Options for the synchronous ADMM solver')
    synchronous_solver_params_parser.add_argument('--epochs', type=int, default=SYNCH_ADMM_PARAMS.NumberOfEpochs, 
                                     help='Number of epochs')
    synchronous_solver_params_parser.add_argument('--updates', type=int, default=SYNCH_ADMM_PARAMS.NumberOfNetworkUpdates, 
                                     help='Number of consecutive network updates')
    synchronous_solver_params_parser.add_argument('--rho', type=float, default=SYNCH_ADMM_PARAMS.Rho, 
                                     help='Outer ADMM step size')
    synchronous_solver_params_parser.add_argument('--eta', type=float, default=SYNCH_ADMM_PARAMS.Eta, 
                                     help='Inner ADMM step size')
    synchronous_solver_params_parser.add_argument('--gamma', type=float, default=SYNCH_ADMM_PARAMS.Gamma, 
                                     help='Projected Gradient Descent step size')
    synchronous_solver_params_parser.add_argument('--kappa', type=float, default=SYNCH_ADMM_PARAMS.Kappa, 
                                     help='Projected Gradient Descent step size reduction factor')
    synchronous_solver_params_parser.add_argument('--pgd-iters', type=int, default=SYNCH_ADMM_PARAMS.PGDIterations, 
                                     help='Number of iterations for each of the inner loop PGD solvers per update')
    synchronous_solver_params_parser.add_argument('--precision', choices=['half', 'single', 'double'], default=SYNCH_ADMM_PARAMS.Precision,
                                     help='Floating point operation precision')
    
    for upto_mlu_backend_parser in add_mlu_backend_subparser(synchronous_solver_params_parser):
        add_admm_synch_communication_backend_subparser(upto_mlu_backend_parser)

    num_workers, addr_list, eval_params, solution_params, warm_start_params, args = distributed_mlu_parse_args(parser)

    if args.solver == 'synch':
        SYNCH_ADMM_PARAMS.NumberOfEpochs = args.epochs
        SYNCH_ADMM_PARAMS.NumberOfNetworkUpdates = args.updates
        SYNCH_ADMM_PARAMS.Rho = args.rho
        SYNCH_ADMM_PARAMS.Eta = args.eta
        SYNCH_ADMM_PARAMS.Gamma = args.gamma
        SYNCH_ADMM_PARAMS.Kappa = args.kappa
        SYNCH_ADMM_PARAMS.PGDIterations = args.pgd_iters
        SYNCH_ADMM_PARAMS.Precision = args.precision
        ALGORITHM_SOLVER_PARAMS = SYNCH_ADMM_PARAMS
        CONTROLLER_CLS = SynchADMMControllerNode
        WORKER_CLS = SynchADMMWorkerNode
    else:
        raise ValueError(f'Unknown solver name {args.solver}')
    
    MLU_BACKEND_PARAMS, MLU_BACKEND_CLS, _ = parse_mlu_backend_params(args)

    CONTROLLER_RPC_PARAMS, CONTROLLER_COMMUNICATION_CLS, \
    WORKER_RPC_PARAMS, WORKER_COMMUNICATION_CLS, _ = \
        parse_add_admm_synch_communication_backend_params(num_workers, addr_list, args)
    
    if args.local:
        multiprocess_mlu_helper(MultiprocessMLUHelperParams(
            ControllerCLS=CONTROLLER_CLS,
            MLUCLS=MLU_BACKEND_CLS,
            ControllerBackendCLS=CONTROLLER_COMMUNICATION_CLS,
            AlgorithmSolverParams=ALGORITHM_SOLVER_PARAMS,
            MLUParams=MLU_BACKEND_PARAMS,
            ControllerRPCParams=CONTROLLER_RPC_PARAMS,
            EvalParams=eval_params,
            WarmstartParams=warm_start_params,
            SolutionParams=solution_params,
            WorkerCLS=WORKER_CLS,
            WorkerBackendCLS=WORKER_COMMUNICATION_CLS,
            WorkerRPCParamList=WORKER_RPC_PARAMS
        ))
    else:
        distributed_mlu_helper(DistributedMLUHelperParams(
            ControllerCLS=CONTROLLER_CLS,
            MLUCLS=MLU_BACKEND_CLS,
            ControllerBackendCLS=CONTROLLER_COMMUNICATION_CLS,
            AlgorithmSolverParams=ALGORITHM_SOLVER_PARAMS,
            MLUParams=MLU_BACKEND_PARAMS,
            ControllerRPCParams=CONTROLLER_RPC_PARAMS,
            EvalParams=eval_params,
            WarmstartParams=warm_start_params,
            SolutionParams=solution_params
        ))
