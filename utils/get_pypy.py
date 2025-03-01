import os
import sys


def get_pypy_interpreter_path() -> str:
    if sys.platform == 'linux':
        return '/usr/bin/pypy3'
    return os.path.join(os.environ['PYPY_HOME'], 'pypy.exe')
