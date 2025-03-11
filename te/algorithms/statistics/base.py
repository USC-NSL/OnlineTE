import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional


class RuntimeStatisticsBase(ABC):
    """A base class for things that we record during runtime"""

    @property
    @abstractmethod
    def values(self) -> List:
        """The associated values gathered under this element"""
    
    @abstractmethod
    def add_value(self, value):
        """Add a new value to this statistic element"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this statistics element"""


class StatisticsCollectorBase(ABC):
    """A base class for things that gather runtime statistics about an algorithm"""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this statistics collector instance"""

    @property
    @abstractmethod
    def elements(self) -> List[RuntimeStatisticsBase]:
        """Return statistic elements managed under this collector"""
    
    @abstractmethod
    def get_element_by_name(self, name: str) -> RuntimeStatisticsBase:
        """Return a statistic element given its name"""

    @abstractmethod
    def add_element(self, element: RuntimeStatisticsBase):
        """Add a new statistics element to the collector"""
    
    @abstractmethod
    def has_element(self, element: Union[str, RuntimeStatisticsBase]) -> bool:
        """Return True/False whether an element is managed by this collector"""

    def add_value(self, element_name: str, value):
        """Add a value to the given statistics element"""
        self.get_element_by_name(element_name).add_value(value)


class RuntimeTrace(RuntimeStatisticsBase):
    """Records list of arbitrary values"""
    
    LINE = "+" + "-"*97 + "+"
    BORDER = "+" + "-"*32 + "+" + "-"*64 + "+"
    NUMERICAL_PRINT_FORMAT = "| {:^30} | MIN: {:^15} Max: {:^15} Med: {:^15} |"
    CENTER_PRINT_FORMAT = "| {:^95} |"

    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self._values: List = list()
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def values(self) -> List:
        return self._values
    
    def add_value(self, value):
        self._values.append(value)
    
    @classmethod
    def get_or_create(cls, name: str, collector: StatisticsCollectorBase):
        """
        Return a trace of a given name in the associated collector,
        or create it if it does not exist.
        """
        if not collector.has_element(name):
            collector.add_element(cls(name))
        return collector.get_element_by_name(name)

    def __str__(self) -> str:
        vals = self.values
        if all([isinstance(v, (int, float)) for v in vals]):
            return self.NUMERICAL_PRINT_FORMAT.format(self.name, np.min(vals), np.max(vals), np.median(vals))
        raise NotImplementedError('Not handling non-numeric lists yet!')


class DictionaryStatisticsCollector(StatisticsCollectorBase):
    """Collector that indexes elements by name in a dictionary"""

    def __init__(self, name: str):
        super().__init__()
        self._dict: Dict[str, RuntimeStatisticsBase] = dict()
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def elements(self) -> List[RuntimeStatisticsBase]:
        return list(self._dict.values())
    
    def add_value(self, element_name, value):
        RuntimeTrace.get_or_create(element_name, self).add_value(value)
    
    def get_element_by_name(self, name) -> RuntimeStatisticsBase:
        return self._dict[name]

    def add_element(self, element):
        assert element.name not in self._dict
        self._dict[element.name] = element
    
    def has_element(self, element: Union[str, RuntimeStatisticsBase]) -> bool:
        if isinstance(element, str):
            return element in self._dict
        elif issubclass(element.__class__, RuntimeStatisticsBase):
            return element.name in self._dict
        raise ValueError(f"Unexpected element type: {type(element)}")
    
    def __str__(self):
        header = '\n'.join([RuntimeTrace.LINE, 
                            RuntimeTrace.CENTER_PRINT_FORMAT.format(self.name),
                            RuntimeTrace.BORDER])
        values = '\n'.join([str(e) for e in self.elements])
        footer = RuntimeTrace.LINE
        return '\n'.join([header, values, footer])


"""
We will create a global collector for all instances to use when
needed.
"""
_GLOBAL_COLLECTOR: StatisticsCollectorBase = DictionaryStatisticsCollector('GlobalCollector')

def get_global_collector() -> StatisticsCollectorBase:
    return _GLOBAL_COLLECTOR

def get_global_statistics() -> Optional[str]:
    g = get_global_collector()
    if len(g.elements) > 0:
        return str(g)
