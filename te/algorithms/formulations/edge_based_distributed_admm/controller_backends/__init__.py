from typing import Type
from te.algorithms.formulations.edge_based_distributed_admm.controller_backends.base import (
    ControllerCommunicationBackendBase, _BACKENDS
)

from . import synchronous_grpc_backend
from . import asynchronous_grpc_backend


def get_backend(name: str) -> Type[ControllerCommunicationBackendBase]:
    global _BACKENDS

    assert name in _BACKENDS, f'No communication backend `{name}` has been registered'

    return _BACKENDS[name]


def list_backends():
    return list(_BACKENDS.keys())
