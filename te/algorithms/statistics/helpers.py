import time
from typing import Dict, Callable, Optional, Any
from te.algorithms.statistics.base import StatisticsCollectorBase, get_global_collector
from te.algorithms.gpu_utils import (synchronize_to_all, synchronize_to_device, get_total_reserved_gpu_memory_usage, 
                                     get_total_used_gpu_memory_usage)


def before_and_after_helper(collector: StatisticsCollectorBase, element_name: str, 
                            f_after: Callable[[Any], Any],
                            f_before: Optional[Callable[[], Dict[str, Any]]] = None):
    """
    Given a statistics collector and a runtime element name, this decorator
    executes `f_before`, then the decorated function, and finally `f_after` and
    then adds the return value of `f_after` to the trace.
    """
    def inner(f):
        def wrapper(*args, **kwargs):
            if f_before is not None:
                after_kwargs = f_before()
            else:
                after_kwargs = dict()
            res = f(*args, **kwargs)
            value = f_after(**after_kwargs)
            collector.add_value(element_name, value)
            return res
        return wrapper
    return inner


record_time_ns = lambda: {'start': time.perf_counter_ns()}


def get_elapsed_time_ns(start: int) -> int:
    return time.perf_counter_ns() - start


def synchronize_and_get_elapsed_time_ns(start: int, dev: Optional[int] = None):
    if dev is not None:
        synchronize_to_device(dev)
    else:
        synchronize_to_all()
    return time.perf_counter_ns() - start


# CPU / GPU runtime

def record_cpu_runtime(element_name: str, collector: Optional[StatisticsCollectorBase] = None):
    if collector is None:
        collector = get_global_collector()
    return before_and_after_helper(collector, element_name, get_elapsed_time_ns, record_time_ns)

def record_gpu_runtime(element_name: str, dev: Optional[int] = None, collector: Optional[StatisticsCollectorBase] = None):
    if collector is None:
        collector = get_global_collector()
    def _synchronize_and_get_elapsed_time_ns(start: int):
        return synchronize_and_get_elapsed_time_ns(start=start, dev=dev)
    return before_and_after_helper(collector, element_name, _synchronize_and_get_elapsed_time_ns, record_time_ns)


# GPU memory usage

def record_reserved_gpu_memory(element_name: str, collector: Optional[StatisticsCollectorBase] = None):
    if collector is None:
        collector = get_global_collector()
    return before_and_after_helper(collector, element_name, get_total_reserved_gpu_memory_usage)
def record_used_gpu_memory(element_name: str, collector: Optional[StatisticsCollectorBase] = None):
    if collector is None:
        collector = get_global_collector()
    return before_and_after_helper(collector, element_name, get_total_used_gpu_memory_usage)
