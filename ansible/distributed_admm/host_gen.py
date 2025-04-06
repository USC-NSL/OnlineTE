import os
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
import sys
import argparse
sys.path.append(os.path.join(THIS_DIR, '../..'))
from te.algorithms.formulations.edge_based_distributed_admm import DistributedADMMSolverParams
from utils.logging import as_warning


WORKER_NODE_NAME_FORMAT = 'n{index}'
CONTROLLER_NODE_NAME = 'controller'
LOCALHOST_NODE_REFERENCE = 'localhost'

double_quoted = lambda s: f'\"{str(s)}\"'

solver_params = DistributedADMMSolverParams()


CONTROLLER_PARAMETER_LIST = {
    "epochs": double_quoted(solver_params.NumberOfEpochs),
    "updates": double_quoted(solver_params.NumberOfNetworkUpdates),
    "pgd_iters": double_quoted(solver_params.PGDIterations),
    "pgd_step": double_quoted(solver_params.Gamma),
    "pgd_reduction": double_quoted(solver_params.Kappa),
    "admm_outer": double_quoted(solver_params.Rho),
    "admm_inner": double_quoted(solver_params.Eta),
    "controller_opt_tol": double_quoted(solver_params.BigGamma),
    "precision": double_quoted(solver_params.Precision)
}


def generate_host_file(number_of_nodes: int, paths: str):
    all_group = [LOCALHOST_NODE_REFERENCE, CONTROLLER_NODE_NAME] + \
        [WORKER_NODE_NAME_FORMAT.format(index=i) for i in range(number_of_nodes)]
    workers_group = [
        ' '.join([
            WORKER_NODE_NAME_FORMAT.format(index=i),
            f'worker_id={i}'
        ])
        for i in range(number_of_nodes)
    ]
    controller_group = [CONTROLLER_NODE_NAME]
    controller_group_vars = [f'{k}={v}' for k, v in CONTROLLER_PARAMETER_LIST.items()]
    all_vars = ['access_token=`FILL ME`', f'num_hosts={number_of_nodes}']

    all_group_str = '\n'.join(['[all]'] + all_group)
    workers_group_str = '\n'.join(['[workers_group]'] + workers_group)
    controller_group_str = '\n'.join(['[controller_group]'] + controller_group)
    controller_group_vars_str = '\n'.join(['[controller_group:vars]'] + controller_group_vars)
    all_vars_str = '\n'.join(['[all:vars]'] + all_vars)

    host_file_str = '\n\n'.join([
        all_group_str, workers_group_str, controller_group_str,
        controller_group_vars_str, all_vars_str
    ])

    with open(paths, 'w') as f:
        f.write(host_file_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate simple host file for Ansible')
    parser.add_argument('n', type=int, help='Number of worker nodes')
    parser.add_argument('--path', type=str, default='hosts', help='Output file path')
    args = parser.parse_args()
    generate_host_file(args.n, args.path)
    print(as_warning('Remember to set the access token for the Github repo!'))
