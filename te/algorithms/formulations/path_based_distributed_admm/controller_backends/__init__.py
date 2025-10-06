from typing import Type
from .base import ControllerCommunicationBackendBase, _BACKENDS, _PARAMS
from .. import PathBasedDistributedADMMControllerRPCParams

from . import asynchronous_grpc_backend


def get_backend(params: PathBasedDistributedADMMControllerRPCParams) -> ControllerCommunicationBackendBase:
    global _BACKENDS
    name = params.Backend
    assert name in _BACKENDS, f'No communication backend `{name}` has been registered'

    return _BACKENDS[name](params)


def get_backend_params(name: str) -> Type[PathBasedDistributedADMMControllerRPCParams]:
    global _PARAMS
    assert name in _PARAMS, f'No communication backend `{name}` has been registered'

    return _PARAMS[name]


def list_backends():
    return list(_BACKENDS.keys())
