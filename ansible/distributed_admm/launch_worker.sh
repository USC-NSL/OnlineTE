#!/bin/bash
# Called by the service script to set up the TE worker listener (i.e. switches)

if [[ -z "${SPHERE_HOME}" ]]; then
    echo "Must set \`SPHERE_HOME\`" 1>&2
    exit 1
fi

if [[ -z "${VENV_HOME}" ]]; then
    echo "Must set \`VENV_HOME\`" 1>&2
    exit 1
fi

pushd "${SPHERE_HOME}"

if [[ -z "${WORKER_ID}" ]]; then
  >&2 echo "WORKER_ID environment variable is not set. Will refuse to proceed."
  exit -1
fi
if [[ -z "${SOLVER_TYPE}" ]]; then
  >&2 echo "SOLVER_TYPE environment variable is not set. Will refuse to proceed."
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

if [ "${SOLVER_TYPE}" = "edge-sync"]; then
  if [ "${TE_MULTICAST}" = "0" ]; then
    echo "Using gRPC backend"
    "${VENV_HOME}/bin/python" -m te.algorithms.formulations.edge_based.distributed.admm_synchronous.worker "${WORKER_ID}" &
  else
    echo "Using UDP multicast backend"
    "${VENV_HOME}/bin/python" -m te.algorithms.formulations.edge_based.distributed.admm_synchronous.worker "${WORKER_ID}" --multicast &
  fi
elif [ "${SOLVER_TYPE}" = "path-sync" ]; then
  if [ "${TE_MULTICAST}" = "0" ]; then
    echo "Using gRPC backend"
    "${VENV_HOME}/bin/python" -m te.algorithms.formulations.path_based.distributed.admm_synchronous.worker "${WORKER_ID}" &
  else
    >&2 echo "UDP multicast backend not ready!"
    exit -1
    # "${VENV_HOME}/bin/python" -m te.algorithms.formulations.path_based.distributed.admm_synchronous.worker "${WORKER_ID}" --multicast &
  fi
else
  >&2 echo "Unknown solver type `${SOLVER_TYPE}`"
  exit -1
fi

popd

child=$!
wait "$child"

exit 0
