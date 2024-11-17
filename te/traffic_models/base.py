import os
import pickle
import numpy as np
import te.constants
import te.traffic_models
from te import TE_PATH
from typing import Dict, Type, List
from abc import ABC, abstractmethod
from collections import namedtuple


class TrafficMatrixBase(ABC):
    TM_FILE_NAME_FORMAT = "{name}_{seed}.pkl"

    def __init__(self, tm: np.ndarray=None, seed: int=None, params=None):
        self.seed = seed
        self.params = params

        self._rng = np.random.default_rng(self.seed)

        if tm is None:
            # If no TM is given, make one from scratch
            self._make_tm()
        else:
            # If a TM is given, it must be square
            assert (len(tm.shape) == 2 and tm.shape[0] == tm.shape[1])
            self.tm = tm
        
        self.fname = self.TM_FILE_NAME_FORMAT.format(name=self.name, seed=self.seed)

    def serialize(self, path=None):
        """
        Serialize matrix and pickle it.
        The object that we save is of the form:
        
        ```
        {
            'tm': 'The matrix array'
            'type': 'A string showing the type of the matrix'
            'params': 'The parameters of this specific traffic'
            'seed': 'The RNG seed used to generate the matrix'
        }
        ```
        """
        if not path:
            path = os.path.join(TE_PATH, te.constants.TM_DIR)
        
        if not os.path.exists(path):
            os.makedirs(path)

        file_path = os.path.join(path, self.fname)

        with open(file_path, 'wb') as f:
            pickle.dump({
                'tm': self.tm,
                'type': self.type(),
                'params': self.params,
                'seed': self.seed
            }, f)

    @staticmethod
    def deserialize(path: str):
        """
        Unpickle the matrix in the given path.
        The models must be registered in `_MODELS` for this to work.
        """
        global _MODELS

        with open(path, 'rb') as f:
            tm_object = pickle.load(f)
        
        assert isinstance(tm_object, dict)
        assert set(tm_object.keys()) == {'tm', 'type', 'params', 'seed'}

        tm_class = _MODELS[tm_object['type']]
        
        return tm_class(
            tm=tm_object['tm'],
            seed=tm_object['seed'],
            params=tm_object['params']
        )

    @abstractmethod
    def _make_tm(self):
        """
        Makes a TM with the given seed and parameters.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Full name of the matrix (including with the parameters).
        """
        pass

    @classmethod
    @abstractmethod
    def type(cls) -> str:
        """
        The type of this traffic matrix.
        """
        pass


"""
This maps the type name of a traffic matrix to the class that implements
it. All traffic matrices that we want to consider MUST be registered in
this before we want deserialize them.
"""
_MODELS: Dict[str, Type[TrafficMatrixBase]] = dict()


def traffic_matrix(cls: Type[TrafficMatrixBase]) -> TrafficMatrixBase:
    """Decorator that registers any Traffic Matrix class for use"""
    global _MODELS

    assert issubclass(cls, TrafficMatrixBase)
    tpe = cls.type()
    assert tpe not in _MODELS
    _MODELS[tpe] = cls

    return cls

# A Commodity is just a tuple of source, destination and demand.
Commodity = namedtuple('Commodity', ['source', 'destination', 'demand'])


def traffic_to_commodity(tm: TrafficMatrixBase) -> List[Commodity]:
    """Convert a traffic matrix to a list of commodities"""
    TM = tm.tm
    return [
        Commodity(src_idx, dst_idx, TM[src_idx, dst_idx]) \
            for src_idx, dst_idx in np.ndindex(TM.shape) \
            if src_idx != dst_idx
    ]
