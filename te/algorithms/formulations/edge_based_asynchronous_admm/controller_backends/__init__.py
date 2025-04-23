from typing import Type
from .base import ControllerCommunicationBackendBase, _BACKENDS, _PARAMS
from .. import AsynchronousADMMControllerRPCParams

from . import udp_multicast_backend


def get_backend(params: AsynchronousADMMControllerRPCParams) -> ControllerCommunicationBackendBase:
    global _BACKENDS
    name = params.Backend
    assert name in _BACKENDS, f'No communication backend `{name}` has been registered'

    return _BACKENDS[name](params)


def get_backend_params(name: str) -> Type[AsynchronousADMMControllerRPCParams]:
    global _PARAMS
    assert name in _PARAMS, f'No communication backend `{name}` has been registered'

    return _PARAMS[name]


def list_backends():
    return list(_BACKENDS.keys())
