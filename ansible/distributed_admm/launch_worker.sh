#!/bin/bash

cd /home/aghavidel/DistributedTE

if [[ -z "${WORKER_ID}" ]]; then
  >&2 echo "WORKER_ID environment variable is not set. Will refuse to proceed."
  exit -1
fi

_term() {
  kill -15 "$child" 2>/dev/null
}
_kill() {
  kill -9 "$child" 2>/dev/null
}

trap _term SIGTERM
trap _kill SIGKILL

/usr/bin/python3 -m te.algorithms.formulations.edge_based_distributed_admm.worker "${WORKER_ID}" &
child=$!
wait "$child"
