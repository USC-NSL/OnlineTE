import os
import sys
import math
import shutil
import multiprocessing
from typing import List, Tuple
from utils.logging import as_warning


NUM_PROCS = max(multiprocessing.cpu_count() - 4, 1)
"""
Maximum number of processes that will be spawned to do any parallel task.
We avoid going up to exactly the CPU count, since that will bring a lot of
contention and cause problems.
We leave 4 cores alone at all times for other things.
"""


class TempHelper:
    """Simple class for working with temporary files"""
    
    @staticmethod
    def create_temp_folder(name: str) -> str:
        TMP = os.environ['TEMP'] if sys.platform == 'win32' else '/tmp'
        path = os.path.join(TMP, name)
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        return path
    
    def __init__(self, temp_folder: str):
        self.temp_folder = temp_folder
        self.temp_path = self.create_temp_folder(temp_folder)
    
    def get_file_path(self, name: str) -> str:
        return os.path.join(self.temp_path, name)

    def close(self):
        try:
            shutil.rmtree(self.temp_path)
        except Exception as e: # noqa
            print(as_warning(f'Could not cleanup temp file: {e}'))
            pass


def get_number_of_required_workers(number_of_columns: int, max_num_workers: int, max_column_per_workers: int) -> int:
    """
    Get number of workers required to handle a 2D matrix with a given number of columns.
    We limit the number of workers to a max value, and also the number of columns that should
    be assigned to each one.
    The latter will be relaxed if the number of columns is too high.
    """
    return min(max_num_workers, math.ceil(number_of_columns / max_column_per_workers))


def get_slice_size(number_of_columns: int, max_num_workers: int, max_column_per_workers: int) -> int:
    """
    Get the appropriate number of columns to chop a 2D matrix with a given
    number of columns into.
    """
    return int(number_of_columns // get_number_of_required_workers(number_of_columns, max_num_workers, max_column_per_workers))


def get_slice_starts_and_exclusive_ends(number_of_commodities: int, max_num_workers: int, 
                                        max_column_per_workers: int) -> List[Tuple[int, int]]:
    """
    Get (inclusive) begin and (exclusive) end of column-wise slice of an array
    with a given number of columns.
    """
    slice_size = get_slice_size(number_of_commodities, max_num_workers, max_column_per_workers)
    number_of_slices = math.ceil(number_of_commodities / slice_size)
    return [(slice_size * i, min(slice_size * (i+1), number_of_commodities)) for i in range(number_of_slices)]
