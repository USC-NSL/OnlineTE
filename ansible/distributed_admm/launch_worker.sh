#!/bin/bash

cd /home/aghavidel/DistributedTE

if [[ -z "${WORKER_ID}" ]]; then
  >&2 echo "WORKER_ID environment variable is not set. Will refuse to proceed."
  exit -1
fi

/usr/bin/python3 -m te.algorithms.formulations.edge_based_distributed_admm.worker "${WORKER_ID}"
