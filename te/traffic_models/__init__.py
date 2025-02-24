from typing import Type
from te.traffic_models.base import TrafficMatrixBase, _MODELS, _PARAMS


def get_traffic_model(name: str) -> Type[TrafficMatrixBase]:
    global _MODELS

    assert name in _MODELS, f'No traffic model `{name}` has been registered'

    return _MODELS[name]


def get_traffic_model_params(name: str) -> Type:
    global _PARAMS

    assert name in _PARAMS, f'No traffic model parameter class `{name}` has been registered'

    return _PARAMS[name]


def list_traffic_models():
    return list(_MODELS.keys())
