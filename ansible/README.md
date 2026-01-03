# Ansible Scripts

This directory contains Ansible scripts for different scenarios. Most of these scripts are to be mainly used with the [SPHERE](https://sphere-project.net/) testbed for setting up compute nodes.

The environment file `sphere_env.env` contains the necessary things setup on the XDC node; modify it to suit the
needs of your current SPHERE project.

Usually, the process for a fresh materilization (i.e. one that has literarily no configuration done on it) is as follows (executed from the root of the repository):
- First, get Ansible and create a configuration file (if you have Ansible already installed, then make sure you skip this, as it also modifies your home directory `ansible.cfg` file):
```sh
# Installs Ansible, as well as a boilerplate `ansible.cfg` file.
# The configuration mathces the one found in:
#   https://mergetb.gitlab.io/testbeds/sphere/sphere-docs/docs/experimentation/experiment-automation/
# We increase the maximum fork to `10`, as 5 is too small ...
./ansible/install_ansible.sh
```
- Now, the environment on the XDC needs to be prepared. A template ENV file `sphere_env.example` is given for this:
```sh
cp ansible/sphere_env.example ansible/sphere_env.env
# Modify the file to suit your needs (in particular, the username
# and project paths and the Git access token)
```
- Make all definitions known so that Ansible can use them
```sh
set -a
source ansible/sphere_env.env
set +a
```
- Create the Ansible inventory file. The number of nodes must be known and passed to the script.
```sh
# On a materialization with <N> switches
python3 -m ansible.host_gen <N>
```
- Prepare the remote ndoes.
```sh
cd ansible
# Prepare each node by installing all dependencies
ansible-playbook -i hosts distributed_admm/prepare.yaml
# Start <n> worker nodes (n <= N) for one experiment
ansible-playbook -i hosts distributed_admm/wait_for_controller --extra-vars "upto=<n>"
```