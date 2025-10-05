from typing import Type
from .base import ControllerCommunicationBackendBase, _BACKENDS, _PARAMS
from .. import DistributedADMMControllerRPCParams

from . import synchronous_grpc_backend
from . import asynchronous_grpc_backend
from . import udp_multicast_backend


def get_backend(params: DistributedADMMControllerRPCParams) -> ControllerCommunicationBackendBase:
    global _BACKENDS
    name = params.Backend
    assert name in _BACKENDS, f'No communication backend `{name}` has been registered'

    return _BACKENDS[name](params)


def get_backend_params(name: str) -> Type[DistributedADMMControllerRPCParams]:
    global _PARAMS
    assert name in _PARAMS, f'No communication backend `{name}` has been registered'

    return _PARAMS[name]


def list_backends():
    return list(_BACKENDS.keys())
