"""
For all intents and purposes, a Traffic Matrix (TM) is _just_ a Numpy array.
However, it is rare that we want to evaluate just one traffic matrix, and
in fact will likely test on many in a sequence.
As such, we define a TM _generator_, `TMGenerator`, which is essentially just a Python 
`generator` that iterates on some sequence of traffic matrices. We do this since
loading many traffic matrices at once can be costly and the generator can afford to be
lazy.
"""

import argparse
import numpy as np
import networkx as nx
from numpy.typing import DTypeLike
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, ClassVar, Tuple, Iterator, Optional


@dataclass(frozen=True, kw_only=True)
class TMGeneratorParams:
    """
    A base class for all TM parameters.
    
    Parameters
    ----------
    seed: Optional[int]
        A random number generator seed that can be used when generating
        matrices of this class.
    count: int
        Number of traffic matrices to generate.
    dtype: numpy.DTypeLike
        The data-type of the matrix.
    scale_factor: float
        The traffic matrix scale factor. During iteration, any TM we
        generate is scaled by this amount before output.
    """
    seed: Optional[int] = field(default=None, metadata={'help': argparse.SUPPRESS})
    count: int = field(default=1, metadata={'help': argparse.SUPPRESS})
    dtype: DTypeLike = field(default=np.float32, metadata={'help': argparse.SUPPRESS})
    scale_factor: float = field(default=1.0, metadata={'help': argparse.SUPPRESS})


class TMGenerator(ABC):
    """
    ABC that implements a sequence of traffic matrices generated from a
    given family (e.g. uniform).
    The general pattern of use is to just iterate on the generator one
    at a time.
    These objects have no memory and thus it should be possible to iterate
    and exhaust them, and then get the exact same sequence of matrices upon
    reiteration.
    """
    _TYPE: ClassVar[str] = ''

    def __init__(self, params: TMGeneratorParams):
        self._params = params
        assert len(self._TYPE) > 0

    @abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]:
        """This object can be iterated to generate traffic matrices."""
        pass

    def get_rng(self):
        return np.random.default_rng(self.params.seed)

    @property
    def params(self) -> TMGeneratorParams:
        """Generator parameters."""
        return self._params

    @classmethod
    def type(cls) -> str:
        """The type of this traffic matrix."""
        return cls._TYPE


@dataclass(frozen=True)
class Commodity:
    """A Commodity is just a tuple of source, destination and demand."""
    source: int
    destination: int
    demand: float


def traffic_to_commodity(tm: np.ndarray) -> List[Commodity]:
    """Convert a traffic matrix to a list of commodities"""
    return [
        Commodity(src_idx, dst_idx, tm[src_idx, dst_idx]) \
            for src_idx, dst_idx in np.ndindex(tm.shape) \
            if src_idx != dst_idx
    ]


def traffic_to_list_of_tuples(tm: np.ndarray) -> List[Tuple[float, int, int]]:
    """Convert a traffic matrix to a list of tuples"""
    return [
        (src_idx, dst_idx, tm[src_idx, dst_idx]) \
            for src_idx, dst_idx in np.ndindex(tm.shape) \
            if src_idx != dst_idx
    ]


def edge_based_to_commodities(
    assignments: np.ndarray, commodities: List[Commodity],
    graph: nx.DiGraph
) -> List[Tuple[Commodity, Commodity]]:
    """
    A handy function that takes the edge-based assignment and returns the list
    of _routed_ commodities. We need this to check if the amount a node sends
    or receives is correctly balanced.

    Returns
    -------
    ls: List[Tuple[Commodity, Commodity]]
        List of tuples, containing the amount sent and received respectively.
    """
    ls = []
    for k, commodity in enumerate(commodities):
        flow_out = defaultdict(list)
        flow_in = defaultdict(list)
        for e, edge in enumerate(graph.edges()):
            flow_out[edge[0]].append(assignments[e, k])
            flow_in[edge[1]].append(assignments[e, k])
        commodity_sent = Commodity(
            source=commodity.source,
            destination=commodity.destination,
            demand=sum(flow_out[commodity.source])
        )
        commodity_received = Commodity(
            source=commodity.source,
            destination=commodity.destination,
            demand=sum(flow_in[commodity.destination])
        )
        ls.append((commodity_sent, commodity_received))
    return ls



__all__ = ['TMGenerator', 'TMGeneratorParams', 'Commodity',
           'traffic_to_commodity', 'traffic_to_list_of_tuples',
           'edge_based_to_commodities']