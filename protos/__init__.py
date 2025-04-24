import os
import subprocess
from typing import Tuple

PROTO_PATH = os.path.dirname(os.path.realpath(__file__))

COMPILE_PROTO = f'python3 -m grpc_tools.protoc -I {PROTO_PATH} ' \
                '--python_out={pyout} --pyi_out={pyiout} --grpc_python_out={grpcout} {protobuf}'

DEPENDENCIES = [
    'array'
]

PROTOS = [
    'distributed_lp',
    'regularized_admm',
    'asynchronous_lp'
]

def make_proto_dir(proto: str) -> Tuple[str, str]:
    assert not proto.endswith('.proto')

    dir_path = os.path.join(PROTO_PATH, proto)
    protobuf_path = os.path.join(PROTO_PATH, f'{proto}.proto')
    assert os.path.exists(protobuf_path)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    
    return dir_path, protobuf_path

def compile_proto(proto: str, with_dependency: bool = False):
    dir_path, protobuf_path = make_proto_dir(proto)
    cmd = COMPILE_PROTO.format(
        pyout=dir_path, grpcout=dir_path, pyiout=dir_path,
        protobuf=protobuf_path
    )

    subprocess.call(cmd.split())

    # Make an empty `__init__` for the proto dir
    # Since we import as modules, let's add the proto path to `sys.path`
    init_ = os.path.join(dir_path, '__init__.py')
    with open(init_, 'w') as init_file:
        lines = ['import os', 'import sys', '', 'PROTO_DIR = os.path.abspath(os.path.dirname(__file__))', 
                 'sys.path.append(PROTO_DIR)']
        dependencies = [] if not with_dependency else \
            [f'sys.path.append(os.path.join(PROTO_DIR, "../{dep}"))' for dep in DEPENDENCIES]
        init_file.write('\n'.join(lines + dependencies))


if __name__ == '__main__':
    # First, compile dependencies
    for proto in DEPENDENCIES:
        print(f'Compiling protobuf dependency `{proto}`')
        compile_proto(proto)
    # Now, the rest
    for proto in PROTOS:
        print(f'Compiling protobuf `{proto}`')
        compile_proto(proto, with_dependency=True)
    
