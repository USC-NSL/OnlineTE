import time
import argparse
import contextlib
import multiprocessing
import concurrent.futures
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union
from te.traffic_models.converters import SampledConverter
from . import ControllerRPCParams, WorkerRPCParams, DEFAULT_RPC_PORT
from .base import (WorkerNodeBase, ControllerNodeBase, WorkerCommunicationBackendBase, ControllerCommunicationBackendBase, 
                   ControllerNodeParams, WorkerNodeParams)
from te.algorithms.base import (SolverParams, TrafficEngineeringLPEvaluationParams, 
                                TrafficEngineeringLPSolutionParams, TrafficEngineeringLPWarmStartParams)
from te.algorithms.utils import get_solution_confusion_matrix, stringify_collected_stats
from te.algorithms.solution import EdgeBasedMinimizeMaximumUtilitySolutionParams, EdgeBasedMinimizeMaximumUtilitySolution
from topologies.utils import get_uniform_tm_problem_with_capacity_heuristic
from utils.logging import as_info, as_warning, as_success, log_subsection_title, log_section_title
from te.algorithms.formulations.helper_base import mlu_solve_and_check, mlu_argparser, mlu_parse_args
from te.algorithms.sub_algorithms.mlu_backends.base import ControllerMLUSolver


@dataclass
class DistributedMLUHelperParams:
    ControllerCLS: type[ControllerNodeBase]
    MLUCLS: type[ControllerMLUSolver]
    ControllerBackendCLS: type[ControllerCommunicationBackendBase]
    AlgorithmSolverParams: SolverParams
    MLUParams: SolverParams
    ControllerRPCParams: ControllerRPCParams
    EvalParams: TrafficEngineeringLPEvaluationParams
    WarmstartParams: Optional[TrafficEngineeringLPWarmStartParams]
    SolutionParams: Optional[TrafficEngineeringLPSolutionParams]


@dataclass
class MultiprocessMLUHelperParams(DistributedMLUHelperParams):
    WorkerCLS: type[WorkerNodeBase]
    WorkerBackendCLS: type[WorkerCommunicationBackendBase]
    WorkerRPCParamList: List[WorkerRPCParams]


def multiprocess_mlu_helper(params: MultiprocessMLUHelperParams):
    assert params.ControllerRPCParams.NumWorkers == len(params.WorkerRPCParamList)
    print(as_warning(log_section_title("LOCAL EXPERIMENT")))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=params.ControllerRPCParams.NumWorkers, 
        mp_context=multiprocessing.get_context(method='spawn')
    ) as network_pool:
        for worker_rpc_params in params.WorkerRPCParamList:
            network_pool.submit(params.WorkerCLS.spawn_and_wait, WorkerNodeParams(
                communication_backend=params.WorkerBackendCLS,
                rpc_params=worker_rpc_params
            ))
        distributed_mlu_helper(params)


def distributed_mlu_helper(params: Union[DistributedMLUHelperParams, MultiprocessMLUHelperParams]):
    c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(
        params.EvalParams.TopologyName, params.EvalParams.Seed, 
        scale_factor=params.EvalParams.ScaleFactor
    )

    if params.EvalParams.SaveSol:
        mlu_solution_params = EdgeBasedMinimizeMaximumUtilitySolutionParams(
            seed=params.EvalParams.Seed, 
            topology_name=params.EvalParams.TopologyName, 
            capacity=c,
            tm_model_name=tm.type(), 
            tm_model_params=tm.params,
            path=params.SolutionParams.Path, 
            sol_name=params.SolutionParams.Name
        )
    else:
        mlu_solution_params = None

    if params.WarmstartParams is not None:
        print(as_info(log_section_title("MLU PROBLEM (WITH WARM-START)")))
        # TODO: Implement different converter passing here ...
        converter = SampledConverter(
            seed=params.WarmstartParams.ConverterSeed,
            params=params.WarmstartParams.ConverterParams
        )
    else:
        print(as_info(log_section_title("MLU PROBLEM")))
        converter = None
    
    print(as_info(f"Network link capacity is: {str(round(c, 2))}"))

    with contextlib.closing(params.ControllerCLS(ControllerNodeParams(
        graph=graph, traffic=tm, solver_params=params.AlgorithmSolverParams,
        mlu_backend=params.MLUCLS, mlu_params=params.MLUParams, communication_backend=params.ControllerBackendCLS,
        rpc_params=params.ControllerRPCParams
    ))) as lp:
        print(as_info(f"Solving With Algorithm: {lp.alg_name}"))
        print(as_info(f"Algorithm Parameters:\n{params.AlgorithmSolverParams}"))
        print(as_info(f"Using MLU Backend: {params.MLUCLS.name()}"))
        print(as_info(f"MLU Backend Parameters:\n{params.MLUParams.stringify_up_to_level(1)}"))
        print(as_info(f"Communication Backend `{params.ControllerBackendCLS.backend_name()}` " +
                      f"With Parameters:\n{params.ControllerRPCParams.stringify_up_to_level(1)}"))
        print(as_info(f"Evaluating With Parameters:\n{params.EvalParams}"))
        print(as_info("Waiting For Network Nodes ..."))
        while True:
            time.sleep(1)
            ready = lp.are_network_nodes_ready()
            if ready is True:
                print(as_success("All Network Nodes Ready"))
                break
            elif ready is None:
                print(as_warning("Aborting"))
                return
        print(as_info(log_subsection_title("MAKING TE LP")))
        lp.make_lp()
        print(as_info(log_subsection_title(f"SOLVING WITH: {lp.alg_name}")))
        mlu_solve_and_check(lp, params.EvalParams)
        
        if mlu_solution_params:
            if converter is None:
                solution = EdgeBasedMinimizeMaximumUtilitySolution(params=mlu_solution_params)
                lp.add_solution_elements(solution)
                solution.dump_elements()
                solution.dump(name=mlu_solution_params.Name)
            else:
                raise NotImplementedError('Will not save solution for warm-tests for now ... (takes too much space!)')
        
        if converter is not None:
            converted_tm = tm
            for i in range(params.WarmstartParams.WarmIters):
                print(as_info(log_subsection_title(f"WARM-START ITERATION {i}")))
                converted_tm = converter.convert(tm)
                lp.update_traffic_matrix(converted_tm)
                mlu_solve_and_check(lp, params.EvalParams)
        
        get_solution_confusion_matrix(lp, params.EvalParams)

        stats = stringify_collected_stats()
        if stats is not None:
            print(as_info(stats))


def distributed_mlu_argparser(prog_name: str) -> argparse.ArgumentParser:
    parser = mlu_argparser(prog_name)
    parser.add_argument('--num-workers', type=int, help='Number of workers to invoke', required=True)
    
    host_params_group = parser.add_argument_group('Remote Host Parameters')
    host_params_group.add_argument('--hosts', nargs='*', default=[], 
                                   help='List of remote hosts to connect to. If empty, defaults to `n0`, `n1`, `n2`, etc.')
    host_params_group.add_argument('--local', action='store_true', 
                                   help='Perform the test on local network. Overrides `hosts` option value')

    return parser


def distributed_mlu_parse_args(parser: argparse.ArgumentParser) -> Tuple[
    int, Tuple[Tuple[str, int], ...],
    TrafficEngineeringLPEvaluationParams, 
    Optional[TrafficEngineeringLPSolutionParams],
    Optional[TrafficEngineeringLPWarmStartParams],
    argparse.Namespace]:
    """
    Parse all the default arguments needed for the distributed MLU problem.

    Arguments
    ---------
    parser: `argparse.ArgumentParser`
        The argument parser (assumed produced with `distributed_mlu_argparser`)
    
    Returns
    -------
    num_workers: int
        Number of distributed workers involved in the problem
    addr_list: Tuple[Tuple[str, int], ...]
        A tuple (yes, it is a named a list because of reasons ...) of all worker
        node addresses. If empty, hostnames of the form `n0`, `n1`, etc. will be
        used, each bound to port `DEFAULT_RPC_PORT`.
        
        If run locally, the ports will be incremented by one for each host.
    eval_params: TrafficEngineeringLPEvaluationParams
        The TE problem evaluation parameters
    solution_params: Optional[TrafficEngineeringLPSolutionParams]
        Solution output parameters
    warmstart_params: Optional[TrafficEngineeringLPWarmStartParams]
        Warm-start parameters
    args: argparse.Namespace
        The namespace object of parsed arguments to further process
    """
    eval_params, solution_params, warm_start_params, args = mlu_parse_args(parser)
    num_workers = args.num_workers
    if args.local:
        addr_list = tuple([('localhost', DEFAULT_RPC_PORT + i) for i in range(num_workers)])
    else:
        if len(args.hosts) == 0:
            # Use `ni` as hosts ...
            addr_list = tuple([(f'n{i}', DEFAULT_RPC_PORT) for i in range(num_workers)])
        else:
            assert len(args.hosts) == num_workers
            addr_list = tuple([(host, DEFAULT_RPC_PORT) for host in args.hosts])
    
    return num_workers, addr_list, eval_params, solution_params, warm_start_params, args
