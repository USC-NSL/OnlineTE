from typing import Type
from .base import (
    TrafficMatrixBase, TrafficMatrixConverterBase, 
    TrafficMatrixParamsBase, TrafficMatrixConverterParamsBase,
    _MODELS, _PARAMS, _CONVERTERS, _CONVERTER_PARAMS
)


def get_traffic_model(name: str) -> Type[TrafficMatrixBase]:
    global _MODELS

    assert name in _MODELS, f'No traffic model `{name}` has been registered'

    return _MODELS[name]


def get_traffic_model_params(name: str) -> Type[TrafficMatrixParamsBase]:
    global _PARAMS

    assert name in _PARAMS, f'No traffic model parameter class `{name}` has been registered'

    return _PARAMS[name]


def get_traffic_converter(name: str) -> Type[TrafficMatrixConverterBase]:
    global _CONVERTERS

    assert name in _CONVERTERS, f'No traffic model converter class `{name}` has been registered'

    return _CONVERTERS[name]


def get_traffic_converter_params(name: str) -> Type[TrafficMatrixConverterParamsBase]:
    global _CONVERTER_PARAMS

    assert name in _CONVERTER_PARAMS, f'No traffic model converter parameter class `{name}` has been registered'

    return _CONVERTER_PARAMS[name]


def list_traffic_models():
    return list(_MODELS.keys())
