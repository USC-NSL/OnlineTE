"""
Quickly set up a `hosts` file for Ansible on the XDC node in SPHERE so that it can interact
with the materialization.
We assume that in a topology containing `N+1` nodes, each node is named `n0` up to `nN` and
the controller node is named `controller`.
"""

import os
import dotenv
import argparse
from typing import Optional
from utils.logging import as_fail, as_warning, as_success
from __init__ import ROOT_PATH


WORKER_NODE_NAME_FORMAT = 'n{index}'
CONTROLLER_NODE_NAME = 'controller'
LOCALHOST_NODE_REFERENCE = 'localhost'
ANSIBLE_DIR = 'ansible'


def generate_host_file(number_of_nodes: int, path: Optional[str] = None):
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
    all_vars = [f'num_hosts={number_of_nodes}', 'multicast=0']

    all_group_str = '\n'.join(['[all]'] + all_group)
    workers_group_str = '\n'.join(['[workers_group]'] + workers_group)
    controller_group_str = '\n'.join(['[controller_group]'] + controller_group)
    all_vars_str = '\n'.join(['[all:vars]'] + all_vars)

    host_file_str = '\n\n'.join([all_group_str, workers_group_str, controller_group_str, all_vars_str])

    if path is None:
        path = os.path.join(ROOT_PATH, ANSIBLE_DIR, 'hosts')

    with open(path, 'w') as f:
        f.write(host_file_str)
    print(as_success(f'Generated inventory file at: {path}'))


def check_env_file():
    env_file = os.path.join(ROOT_PATH, ANSIBLE_DIR, 'sphere_env.env')
    assert dotenv.load_dotenv(env_file), as_fail('No SPHERE environment vairable file found! Aborting ...')
    _git_token = os.environ.get('GIT_ACCESS_TOKEN')
    if _git_token is None or _git_token == '':
        print(as_warning('Remember to set the git access token!'))
    print(as_success(f'Found SPHERE .env file at: {env_file}'))


def generate_service_file():
    service_path = os.path.join(ROOT_PATH, os.environ.get('SERVICE_FILE'))
    exec_path = os.path.join(
        os.environ.get('SPHERE_HOME'),
        os.environ.get('LAUNCH_FILE')
    )
    service_working_dir = os.environ.get('SPHERE_HOME')
    lines = [
        '[Unit]',
        'Description=TE Worker Node Service',
        '',
        '[Service]',
        f'WorkingDirectory={service_working_dir}',
        f'ExecStart={exec_path}',
        '',
        '[Install]',
        'WantedBy=default.target'
    ]
    with open(service_path, 'w') as f:
        f.write('\n'.join(lines))
    print(as_success(f'Wrote out TE service file at: {service_path}'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate simple host file for Ansible')
    parser.add_argument('n', type=int, help='Number of worker nodes')
    parser.add_argument('--path', type=str, help='Output file path')
    args = parser.parse_args()
    check_env_file()
    generate_host_file(args.n, args.path)
    generate_service_file()
