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

"${VENV_HOME}/bin/python" -m benchmarks.spawn_worker &

popd

child=$!
wait "$child"

exit 0
