# Ansible Scripts

This directory contains Ansible scripts for different scenarios. Most of these scripts are to be mainly used with the [SPHERE](https://sphere-project.net/) testbed for setting up compute nodes.

Usually, the process for a fresh materilization (i.e. one that has literarily no configuration done on it) is as follows:
```
# On a materialization with <N> switches, first generate the hosts file
python3 host_gen <N>
# Prepare each node by installing all dependencies
ansible-playbook -i hosts prepare.yaml
# Start <n> worker nodes (n <= N) for one experiment
ansible-playbook -i hosts wait_for_controller --extra-vars "upto=<n>"
```