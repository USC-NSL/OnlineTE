#!/bin/bash

cd /home/aghavidel/DistributedTE

if [[ -z "${WORKER_ID}" ]]; then
  >&2 echo "WORKER_ID environment variable is not set. Will refuse to proceed."
  exit -1
fi

_term() {
  kill -15 "$child" 2>/dev/null
  echo "Worker was TERMINATED."
}
_kill() {
  kill -9 "$child" 2>/dev/null
  echo "Worker was KILLED."
}

trap _term SIGTERM
trap _kill SIGKILL

if [ "${TE_MULTICAST}" = "0" ]; then
  echo "Using gRPC backend"
  /usr/bin/python3 -m te.algorithms.formulations.edge_based_distributed_admm.worker "${WORKER_ID}" &
else
/usr/bin/python3 -m te.algorithms.formulations.edge_based_distributed_admm.worker "${WORKER_ID}" --multicast &
  echo "Using UDP multicast backend"
fi

child=$!
wait "$child"

exit 0
