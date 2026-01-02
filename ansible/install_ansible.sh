#!/bin/bash

sudo apt update
sudo apt install software-properties-common python3-dev
sudo add-apt-repository --yes --update ppa:ansible/ansible
sudo apt install ansible -y

cat > ~/.ansible.cfg << EOF
[defaults]
# don't check experiment node keys, if this is not set, you will have to
# explicitly accept the SSH key for each experiment node you run Ansible
# against
host_key_checking = False

# configure up to 10 hosts in parallel
forks = 10

# tmp directory on the local non-shared filesystem. Useful when running ansible
# on multiple separate XDCs
local_tmp = /tmp/ansible/tmp

[ssh_connection]

# connection optimization that increases speed significantly
pipelining = True

# control socket directory on the local non-shared filesystem. Useful when
# running ansible on multiple separate XDCs
control_path_dir = /tmp/ansible/cp
EOF