import tqdm
import numpy as np
import te.constants
from typing import List, Iterable, Optional, Dict


class ANSIColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


as_bold = lambda msg: f"{ANSIColors.BOLD}{msg}{ANSIColors.ENDC}"
as_warning = lambda msg: f"{ANSIColors.BOLD}{ANSIColors.WARNING}{msg}{ANSIColors.ENDC}"
as_info = lambda msg: f"{ANSIColors.BOLD}{ANSIColors.OKBLUE}{msg}{ANSIColors.ENDC}"
as_success = lambda msg: f"{ANSIColors.BOLD}{ANSIColors.OKGREEN}{msg}{ANSIColors.ENDC}"
as_fail = lambda msg: f"{ANSIColors.BOLD}{ANSIColors.FAIL}{msg}{ANSIColors.ENDC}"


def str_round(value, digits: int) -> str:
    """For float16, `np.round` / `round` can easily return `inf`. So we cast to `float32` always"""
    val32 = np.float32(value)
    return str(round(val32, digits))


def list_round(values: List, digits: int) -> List[str]:
    return [str_round(value, digits) for value in values]


LINE_SEPARATOR_LENGTH = 82

def log_section_title(title: str) -> str:
    assert len(title) < (LINE_SEPARATOR_LENGTH - 2)
    left_padding = (LINE_SEPARATOR_LENGTH - (len(title) + 2)) // 2
    right_padding = LINE_SEPARATOR_LENGTH - (left_padding + len(title) + 2)
    return '\n'.join([
        '=' * LINE_SEPARATOR_LENGTH,
        '=' * left_padding + f' {title} ' + '=' * right_padding,
        '=' * LINE_SEPARATOR_LENGTH
    ])

def log_subsection_title(title: str) -> str:
    assert len(title) < (LINE_SEPARATOR_LENGTH - 2)
    left_padding = (LINE_SEPARATOR_LENGTH - (len(title) + 2)) // 2
    right_padding = LINE_SEPARATOR_LENGTH - (left_padding + len(title) + 2)
    return '=' * left_padding + f' {title} ' + '=' * right_padding

_LOG_SUBSECTION_SEPARATOR = '-' * LINE_SEPARATOR_LENGTH

def log_subsection_separator() -> str:
    return _LOG_SUBSECTION_SEPARATOR


TQDM_PBAR_LEN = 36

class ShortTQDM:
    @classmethod
    def pbar_format(cls) -> str:
        return '{l_bar}{bar:36}{r_bar}{bar:-36b}'

    def __init__(self, object: Iterable, length: Optional[int] = None):
        if hasattr(object, '__len__'):
            self._len = len(object)
        else:
            assert length is not None
            self._len = length
        if te.constants.SHOW_PROGRESS_BAR:
            self._pbar = tqdm.tqdm(object, bar_format=self.pbar_format(), total=self._len)
        self._object = iter(object)
    
    def __len__(self) -> int:
        return self._len
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            obj = next(self._object)
            if te.constants.SHOW_PROGRESS_BAR:
                self._pbar.update()
            return obj
        except StopIteration:
            if te.constants.SHOW_PROGRESS_BAR:
                self._pbar.close()
            raise StopIteration
    
    def set_postfix(self, data: Dict):
        self._pbar.set_postfix(data)
    
    def write(self, message: str):
        self._pbar.write(message)
    

class ShortTQDMEnumerate(ShortTQDM):
    def __init__(self, object: List):
        self._len = len(object)
        self._object = enumerate(object)
        if te.constants.SHOW_PROGRESS_BAR:
            self._pbar = tqdm.tqdm(object, bar_format=self.pbar_format(), total=self._len)
