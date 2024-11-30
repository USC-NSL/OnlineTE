import os
import subprocess
from typing import Tuple

PROTO_PATH = os.path.dirname(os.path.realpath(__file__))

COMPILE_PROTO = f'python3 -m grpc_tools.protoc -I {PROTO_PATH} ' \
                '--python_out={pyout} --pyi_out={pyiout} --grpc_python_out={grpcout} {protobuf}'

PROTOS = [
    'distributed_lp.proto'
]

def make_proto_dir(proto: str) -> Tuple[str, str]:
    assert proto.endswith('.proto')
    proto_name = proto[:-len('.proto')]
    assert len(proto_name) > 0

    dir_path = os.path.join(PROTO_PATH, proto_name)
    protobuf_path = os.path.join(PROTO_PATH, proto)
    assert os.path.exists(protobuf_path)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    
    return dir_path, protobuf_path

def compile_proto(proto: str):
    dir_path, protobuf_path = make_proto_dir(proto)
    cmd = COMPILE_PROTO.format(
        pyout=dir_path, grpcout=dir_path, pyiout=dir_path,
        protobuf=protobuf_path
    )

    subprocess.call(cmd.split())


if __name__ == '__main__':
    for proto in PROTOS:
        print(f'Compiling protobuf `{proto}`')
        compile_proto(proto)
    
