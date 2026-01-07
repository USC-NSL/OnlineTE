#!/bin/bash

sudo apt update
sudo apt install -y software-properties-common python3-dev
sudo add-apt-repository --yes --update ppa:ansible/ansible
sudo apt install -y ansible

cat > ~/.ansible.cfg << EOF
[defaults]
# don't check experiment node keys, if this is not set, you will have to
# explicitly accept the SSH key for each experiment node you run Ansible
# against
host_key_checking = False
interpreter_python = auto_silent

# configure up to 10 hosts in parallel
forks = 10

# tmp directory on the local non-shared filesystem. Useful when running ansible
# on multiple separate XDCs
local_tmp = /tmp/ansible/tmp

# fact caching to file instead of in memory
fact_caching = jsonfile

# set directory/location of fact cache files
fact_caching_connection = /tmp/.ansible/fc

# ask ansible to gather facts only when necessary
gathering = smart

[ssh_connection]

# connection optimization that increases speed significantly
pipelining = True

# control socket directory on the local non-shared filesystem. Useful when
# running ansible on multiple separate XDCs
control_path_dir = /tmp/ansible/cp
EOF
