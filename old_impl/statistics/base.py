import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Callable, Any


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
    
    @abstractmethod
    def str_with_format(self, format: Optional[Callable[[Any], str]] = None) -> str:
        """Stringify the values with a format (if available)"""


class StatisticsCollectorBase(ABC):
    """A base class for things that gather runtime statistics about an algorithm"""

    LINE: str = "+" + "-"*32 + "+"
    """A class attribute that keeps a separator between tables"""

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
    
    @abstractmethod
    def set_format(self, format: Optional[Callable[[Any], str]]):
        """Set formatting for all values under this collector"""

    @property
    @abstractmethod
    def str_body(self) -> Optional[str]:
        """Stringify this collector, without header/footer. Return `None` if we have nothing to report"""
    
    @property
    def str_with_header(self) -> Optional[str]:
        body = self.str_body
        if body:
            return '\n'.join([self.LINE, body])
    
    @property
    def str_with_footer(self) -> str:
        body = self.str_body
        if body:
            return '\n'.join([body, self.LINE])
    
    def __str__(self):
        with_header = self.str_with_header
        if with_header:
            return '\n'.join([with_header, self.LINE])


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
        return self.str_with_format(format=lambda item: str(item))
    
    def str_with_format(self, format = None):
        if format is None:
            return str(self)
        vals = self.values
        if all([isinstance(v, (int, float)) for v in vals]):
            return self.NUMERICAL_PRINT_FORMAT.format(self.name, format(np.min(vals)), format(np.max(vals)), format(np.median(vals)))
        raise NotImplementedError('Not handling non-numeric lists yet!')


class DictionaryStatisticsCollector(StatisticsCollectorBase):
    """Collector that indexes elements by name in a dictionary"""

    LINE = RuntimeTrace.LINE

    def __init__(self, name: str):
        super().__init__()
        self._dict: Dict[str, RuntimeStatisticsBase] = dict()
        self._name = name
        self._format: Optional[Callable[[Any], str]] = None
    
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
    
    def set_format(self, format):
        self._format = format
    
    @property
    def str_body(self) -> str:
        header = '\n'.join([RuntimeTrace.CENTER_PRINT_FORMAT.format(self.name),
                            RuntimeTrace.BORDER])
        values = '\n'.join([e.str_with_format(format=self._format) for e in self.elements])
        return '\n'.join([header, values])


"""
We will create a global collector for all instances to use when
needed.
We care in particular about two runtime metrics.
    1. Function execution times, to determine stragglers
    2. Memory consumption, sampled at certain locations
We create global collectors for each ...
"""
_GLOBAL_EXECUTION_TIME_COLLECTOR: StatisticsCollectorBase = DictionaryStatisticsCollector('Execution-Time')
_GLOBAL_MEMORY_USAGE_COLLECTOR: StatisticsCollectorBase = DictionaryStatisticsCollector('Memory-Usage')

"""This will contain the list of collectors"""
_COLLECTORS: List[StatisticsCollectorBase] = [_GLOBAL_EXECUTION_TIME_COLLECTOR, _GLOBAL_MEMORY_USAGE_COLLECTOR]

def add_collector(collector: StatisticsCollectorBase):
    global _COLLECTORS
    _COLLECTORS.append(collector)

def get_global_execution_time_collector() -> StatisticsCollectorBase:
    return _GLOBAL_EXECUTION_TIME_COLLECTOR

def get_global_memory_usage_collecotor() -> StatisticsCollectorBase:
    return _GLOBAL_MEMORY_USAGE_COLLECTOR

def stringify_collected_stats() -> Optional[str]:
    ls = list(filter(lambda col: len(col.elements) > 0, _COLLECTORS))
    if len(ls) == 1:
        return str(ls[0])
    elif len(ls) == 2:
        return '\n'.join([ls[0].str_with_header, str(ls[1])])
    elif len(ls) > 2:
        return '\n'.join([ls[0].str_with_header] + [col.str_body for col in ls[1:-1]] + [ls[-1].str_with_footer])
    
def _decimal_str(value: float) -> str:
    if value < 10:
        return str(round(value, 2))
    elif value < 100:
        return str(round(value, 1))
    else:
        return str(round(value))

def format_runtime(runtime_ns: Union[int, float]) -> str:
    runtime_ns = int(runtime_ns)
    if runtime_ns < int(1e3):
        return f'{str(runtime_ns)} ns'
    elif runtime_ns < int(1e6):
        runtime_us = runtime_ns / 1e3
        return f'{_decimal_str(runtime_us)} us'
    elif runtime_ns < int(1e9):
        runtime_ms = runtime_ns / 1e6
        return f'{_decimal_str(runtime_ms)} ms'
    else:
        runtime_s = runtime_ns / 1e9
        return f'{_decimal_str(runtime_s)} s'

def format_memory_usage(mem_bytes: int) -> str:
    if mem_bytes < (1 << 10):
        return f'{mem_bytes} B'
    elif mem_bytes < (1 << 20):
        mem_kB = mem_bytes // (1 << 10)
        return f'{mem_kB} kB'
    elif mem_bytes < (1 << 30):
        mem_MB = mem_bytes // (1 << 20)
        return f'{mem_MB} MB'
    else:
        mem_GB = mem_bytes // (1 << 30)
        return f'{mem_GB} GB'


_GLOBAL_EXECUTION_TIME_COLLECTOR.set_format(format=format_runtime)
_GLOBAL_MEMORY_USAGE_COLLECTOR.set_format(format=format_memory_usage)
