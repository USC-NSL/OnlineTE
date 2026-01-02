# Ansible Scripts

This directory contains Ansible scripts for different scenarios. Most of these scripts are to be mainly used with the [SPHERE](https://sphere-project.net/) testbed for setting up compute nodes.

The environment file `sphere_env.env` contains the necessary things setup on the XDC node; modify it to suit the
needs of your current SPHERE project.

Usually, the process for a fresh materilization (i.e. one that has literarily no configuration done on it) is as follows:
```sh
# Setup the environment variables for SPHERE
chmod +x set_env.sh && ./set_env.sh
# On a materialization with <N> switches, first generate the hosts file
python3 -m distributed_admm/host_gen <N>
# Prepare each node by installing all dependencies
ansible-playbook -i distributed_admm/hosts distributed_admm/prepare.yaml
# Start <n> worker nodes (n <= N) for one experiment
ansible-playbook -i distributed_admm/hosts distributed_admm/wait_for_controller --extra-vars "upto=<n>"
```