"""
Besides our base classes, this code essentially mirrors that of `NCFlow`.
Please see: https://github.com/netcontract/ncflow/blob/master/lib/traffic_matrix.py
"""

import os
import pickle
import numpy as np
import networkx as nx
import te.constants
from te import TE_PATH
from numpy.typing import DTypeLike
from dataclasses import dataclass
from typing import Tuple, Optional, Type
from .base import TrafficMatrixBase, TrafficMatrixParamsBase, traffic_matrix, traffic_matrix_param


"""
Certain distributions depend very much on the graph that we use
for the traffic matrix itself.
For these distributions, we need to include the graph object (i.e. a
`nx.DiGraph` object) with the parameters.
"""


@traffic_matrix_param('Uniform')
@dataclass
class UniformTrafficMatrixParams(TrafficMatrixParamsBase):
    """
    Parameters defining a random, uniform traffic matrix.

    Attributes
    ----------
    n: int
        Number of nodes
    min: float
        Minimum admissible demand value
    max: float
        Maximum admissible demand value
    """
    n: int
    min: float
    max: float

    def __post_init__(self):
        assert self.min >= 0 and self.max >= 0 and self.max >= self.min


@traffic_matrix_param('Exponential')
@dataclass
class ExponentialTrafficMatrixParams(TrafficMatrixParamsBase):
    """
    Parameters defining an exponential traffic matrix.
    The matrix is defined by a mean `beta` and a decay of `gamma`.
    Demand between two nodes a distance of `d` hops apart is
    sampled from `Exp(beta * (gamma)^d)`.

    Attributes
    ----------
    graph: nx.DiGraph
        Topology graph object. We need distances between
        nodes, so that is why we need it.
    beta: float
        Distribution mean; can be arbitrary positive value.
    gamma: float
        Distribution decay; must be in `(0, 1]`
    """
    graph: nx.DiGraph
    beta: float
    gamma: float

    def __post_init__(self):
        assert self.beta > 0 and self.gamma > 0 and self.gamma <= 1


@traffic_matrix_param('Bimodal')
@dataclass
class BimodalTrafficMatrixParams(TrafficMatrixParamsBase):
    """
    Parameters defining an bimodal traffic matrix.
    A given fraction of the entries are sampled from one uniform distribution,
    and the rest from another one.

    Attributes
    ----------
    n: int
        Number of nodes for our matrix.
    range1: Tuple[float, float]
        First distribution min and max values.
    range2: Tuple[float, float]
        Second distribution min and max values.
    fraction: float
        What fraction of entries belong to the first distribution.
        Must be in `(0, 1)`.
    """
    n: int
    range1: Tuple[float, float]
    range2: Tuple[float, float]
    fraction: float

    def __post_init__(self):
        assert self.range1[0] <= self.range1[1]
        assert self.range2[0] <= self.range2[1]
        assert self.fraction > 0 and self.fraction < 1


class CustomTrafficMatrix(TrafficMatrixBase):
    def __init__(self, tm: Optional[np.ndarray] = None, seed: Optional[int] = None, 
                 params: Optional[Type[TrafficMatrixParamsBase]] = None):
        assert tm is not None
        assert params is None and seed is None
        np.fill_diagonal(tm, 0.0)
        self.tm = tm
    
    def _make_tm(self):
        pass

    @property
    def name(self) -> str:
        return "Custom"
    
    @property
    def type(self) -> str:
        return "Custom"


@traffic_matrix
class UniformTrafficMatrix(TrafficMatrixBase):
    def __init__(self, tm: Optional[np.ndarray] = None, seed: Optional[int] = None, 
                 params: Optional[UniformTrafficMatrixParams] = None):
        super().__init__(tm, seed, params)

    @classmethod
    def type(cls) -> str:
        return 'Uniform'
    
    @property
    def name(self) -> str:
        PARAMS: UniformTrafficMatrixParams = self.params

        return f'Uniform_{PARAMS.n}_{PARAMS.min}_{PARAMS.max}'
    
    def _make_tm(self):
        assert self.params is not None
        PARAMS: UniformTrafficMatrixParams = self.params

        tm = self._rng.random(
            size=(PARAMS.n, PARAMS.n),
            dtype=np.float32
        ) * (PARAMS.max - PARAMS.min) + PARAMS.min
        np.fill_diagonal(tm, 0.0)

        self.tm = tm
    

@traffic_matrix
class ExponentialTrafficMatrix(TrafficMatrixBase):
    def __init__(self, tm: Optional[np.ndarray] = None, seed: Optional[int] = None, 
                 params: Optional[ExponentialTrafficMatrixParams] = None):
        super().__init__(tm, seed, params)
    
    @classmethod
    def type(cls) -> str:
        return 'Exponential'
    
    @property
    def name(self) -> str:
        PARAMS: ExponentialTrafficMatrixParams = self.params
        GRAPH: nx.DiGraph = PARAMS.graph

        return f'Exponential_G({len(GRAPH.nodes)},{len(GRAPH.edges)})_{PARAMS.beta}_{PARAMS.gamma}'
    
    def _make_tm(self):
        assert self.params is not None
        PARAMS: ExponentialTrafficMatrixParams = self.params
        GRAPH: nx.DiGraph = PARAMS.graph
        NUMBER_OF_NODES = len(GRAPH.nodes)

        distances = np.zeros((NUMBER_OF_NODES, NUMBER_OF_NODES), dtype=np.int32)
        for src, dist_dict in nx.shortest_path_length(GRAPH):
            for target, dist in dist_dict.items():
                distances[src, target] = dist

        tm = np.array([
            [self._rng.exponential(PARAMS.beta * (PARAMS.gamma**dist)) for dist in row] \
                for row in distances
            ], dtype=np.float32)
        np.fill_diagonal(self.tm, 0.0)

        self.tm = tm


@traffic_matrix
class BimodalTrafficMatrix(TrafficMatrixBase):
    def __init__(self, tm: Optional[np.ndarray] = None, seed: Optional[int] = None, 
                 params: Optional[BimodalTrafficMatrixParams] = None):
        super().__init__(tm, seed, params)

    @classmethod
    def type(cls) -> str:
        return 'Bimodal'
    
    @property
    def name(self) -> str:
        PARAMS: BimodalTrafficMatrixParams = self.params

        return f'Bimodal_R1({PARAMS.range1[0]},{PARAMS.range1[1]})_R2({PARAMS.range2[0]},{PARAMS.range2[1]})_{PARAMS.fraction}'
    
    def _make_tm(self):
        assert self.params is not None
        PARAMS: BimodalTrafficMatrixParams = self.params
        num_nodes = PARAMS.n

        tm = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        inds = self._rng.choice(
            a=[False, True],
            size=(num_nodes, num_nodes),
            p=[PARAMS.fraction, 1 - PARAMS.fraction]
        )
        tm[inds] = np.random.uniform(PARAMS.range1[0], PARAMS.range1[1], np.sum(inds))
        tm[~inds] = np.random.uniform(PARAMS.range2[0], PARAMS.range2[1], np.sum(~inds))
        np.fill_diagonal(tm, 0.0)

        self.tm = tm


@traffic_matrix_param('File')
@dataclass
class FilebackedTrafficMatrixParams(TrafficMatrixParamsBase):
    """
    Parameters defining a traffic matrix backed by a file.

    Attributes
    ----------
    path: str
        Path to load the traffic matrix from
    scale: Optional[float]
        If present, the traffic matrix will be divided by this value
        element-wise. If `None`, the matrix is normalized between
        0 and 1 by dividing by the maximum value.
        Defaults to `1.0`.
    dtype: DTypeLike
        Data type to cast the array into after reading it.
        Defaults to `np.float64`.
    """
    path: str
    scale: Optional[float] = 1.0
    dtype: DTypeLike = np.float64


@traffic_matrix
class FilebackedTrafficMatrix(TrafficMatrixBase):
    def __init__(self, tm: Optional[np.ndarray] = None, seed: Optional[int] = None, 
                 params: Optional[FilebackedTrafficMatrixParams] = None):
        super().__init__(tm, seed, params)
        self._expected_cap: Optional[float] = None
        self._initial_max: Optional[float] = None

    @classmethod
    def type(cls) -> str:
        return 'Filebacked'
    
    @property
    def name(self) -> str:
        PARAMS: FilebackedTrafficMatrixParams = self.params

        return f'Filebacked_{PARAMS.scale}_{str(PARAMS.dtype)}_{PARAMS.path}'
    
    def _make_tm(self):
        assert self.params is not None
        PARAMS: FilebackedTrafficMatrixParams = self.params
        path = PARAMS.path
        if not os.path.exists(path):
            path = os.path.join(TE_PATH, te.constants.TM_DIR, path)
            
        with open(path, "rb") as tm_file:
            tm = pickle.load(tm_file)
            print(tm)
            assert isinstance(tm, np.ndarray), 'TM object is not a Numpy array'
            np.fill_diagonal(tm, 0)
            self._initial_max = tm.max()
            if PARAMS.scale is not None:
                tm = tm / PARAMS.scale
            else:
                tm = tm / tm.max()
            self.tm = tm.astype(PARAMS.dtype)
            # TODO: Change this. It is here to match TEAL
            self._expected_cap = 5000 / self._initial_max
    
    @property
    def expected_cap(self) -> float:
        return self._expected_cap
