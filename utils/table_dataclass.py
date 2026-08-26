import enum
import inspect
import argparse
import dataclasses
import jsonargparse
from abc import ABC
from itertools import count
from typing import List, Dict, Any, Union, Tuple
from utils.logging import LINE_SEPARATOR_LENGTH


class TableDataclass(ABC):
    """
    ABC for any set of parameters that we want to bundle together and make
    known to the user.
    By default, it supports a pretty-print such that a table is shown for
    a given object when stringified.

    Every inheritence of this class, adds an extra `depth` to it.
    Depth 0 is always the parameters introduced by the latest inheritence, and
    inner depths go into the fields inherited by the parents.

    For examples:
    ```
    @dataclass
    class A(TableDataclass):
        field1: str
    
    @dataclass
    class B(TableDataclass):
        field2: str
    
    B_obj = B('a', 'b')
    print(B_obj.child_fields)               # returns `{'field2': 'a'}
    print(B_obj.get_fields_up_to_level(1))  # returns `{'field2': 'a', 'field1': 'b'}
    ```

    TODO: Force this to always check if we are implementing a `dataclass`.
    """
    _left_column_share = 0.5
    PRINT_FORMAT = "| {:^{left_padding}} | {:^{right_padding}} |"

    @property
    def left_column_share(self) -> float:
        return self._left_column_share
    @left_column_share.setter
    def left_column_share(self, value: float):
        assert (value > 0) and (value < 1)
        self._left_column_share = value
    @property
    def left_column_padding(self) -> int:
        return int(self.left_column_share * (LINE_SEPARATOR_LENGTH - 5))
    @property
    def right_column_padding(self) -> int:
        return LINE_SEPARATOR_LENGTH - 7 - self.left_column_padding
    @property
    def line_padding(self) -> int:
        return LINE_SEPARATOR_LENGTH - 2
    @property
    def line(self) -> str:
        return "+" + "-"*self.line_padding + "+"

    @classmethod
    def field_names(cls) -> List[str]:
        if cls.__base__ == ABC:
            return []
        return [item.name for item in dataclasses.fields(cls)]
    
    @property
    def child_fields(self) -> Dict[str, Any]:
        return self.get_fields_up_to_level(0)
    
    def get_fields_up_to_level(self, level: int):
        ancestor_class = self.__class__
        if level < 0:
            it = count()
        else:
            it = range(level+1)
        for i in it:
            ancestor_class = ancestor_class.__base__
            if ancestor_class == ABC:
                ancestor_fields = []
                break
            else:
                assert issubclass(ancestor_class, TableDataclass)
            if i == level:
                ancestor_fields = ancestor_class.field_names()
        child_dict = self.__dict__.copy()
        for key in ancestor_fields:
            child_dict.pop(key)
        keys = list(child_dict.keys())
        for key in keys:
            if key.startswith('_'):
                child_dict.pop(key, None)
        return child_dict
    
    @classmethod
    def _list_or_tuple_to_str(cls, items: Union[List, Tuple]) -> Union[str, List[str]]:
        if len(items) == 0:
            if isinstance(items, list):
                return '[]'
            else:
                return '()'
        example = items[0]
        if isinstance(example, (int, float, bool, str)):
            # Multiple things on a single line ...
            return ', '.join([str(item) for item in items])
        elif isinstance(example, tuple):
            # Sequence of tuples ...
            return [f'({", ".join([str(e) for e in item])})' for item in items]
        else:
            raise ValueError(f'Sequence element type unexpected: {example}')
    
    @classmethod
    def _param_to_str(cls, value) -> Union[str, List[str]]:
        if isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            return f'{value:.2e}'
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, str):
            return value
        elif isinstance(value, (tuple, list)):
            return cls._list_or_tuple_to_str(value)
        elif value is None:
            return "None"
        elif dataclasses.is_dataclass(value):
            return f"<{value.__class__.__name__}>"
        elif inspect.isclass(value):
            return f"[{value.__class__.__name__}]"
        elif isinstance(value, enum.Enum):
            return str(value)
        raise ValueError(f'Unexpected instance: {type(value)}')
    
    def _field_to_string(self, key: str, value: Any):
        value_str = self._param_to_str(value)
        if isinstance(value_str, str):
            return self.PRINT_FORMAT.format(
                key, value_str,
                left_padding=self.left_column_padding,
                right_padding=self.right_column_padding
            )
        elif isinstance(value_str, list):
            result = '\n'.join(
                [self.PRINT_FORMAT.format(
                    key, value_str[0],
                    left_padding=self.left_column_padding,
                    right_padding=self.right_column_padding)] + 
                [self.PRINT_FORMAT.format(
                    '.', value_line,
                    left_padding=self.left_column_padding,
                    right_padding=self.right_column_padding
                ) for value_line in value_str[1:]]
            )
            return result

    def __str__(self) -> str:
        return self.stringify_up_to_level(0)
    
    def stringify_up_to_level(self, level: int) -> str:
        return '\n'.join(
            [self.line] +
            [self._field_to_string(key, value)
                for key, value in self.get_fields_up_to_level(level).items()] +
            [self.line]
        )
    
    def str_all(self) -> str:
        return self.stringify_up_to_level(-1)
    
    @classmethod
    def make_from_args(cls, namespace: Union[argparse.Namespace, jsonargparse.Namespace]):
        params = dict()
        for name in cls.field_names():
            if name in namespace:
                params[name] = namespace[name]
        return cls(**params)