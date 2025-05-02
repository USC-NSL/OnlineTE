import struct
import te.constants
from dataclasses import dataclass
from typing import Tuple, ClassVar, Literal, Optional, List, Any
from te.algorithms.base import SolverParams, GurobiSolverParams
from te.algorithms.array_utils import SINGLE_PRECISION
from te.algorithms.array_utils.cpu_utils import CPUArray

import protos.asynchronous_lp.asynchronous_lp_pb2 as asynchronous_lp_messages


@dataclass
class AsynchronousADMMSolverParams(GurobiSolverParams):
    NumberOfEpochs: int = 100
    InnerIterations: int = 4
    Rho: float = 1.0
    Eta: float = 8.0
    QPStep: float = 1.0
    QPIterations: int = 5
    BigGamma: float = 1e-7
    Precision: Literal['double', 'single', 'half'] = SINGLE_PRECISION
    Seed: int = te.constants.DEFAULT_SEED
    Upsilon: int = 1
    WorkerBatchSize: int = 1
    Sigma: int = 1
    ADMMConvTol: float = 1e-6


@dataclass
class AsynchronousADMMControllerRPCParams(SolverParams):
    AddressList: Tuple[Tuple[str, int]] = (("localhost", 13000),)
    NumWorkers: int = 2
    Backend: ClassVar[str] = ""
    QueueTimeout: float = 1.0

    def __post_init__(self):
        self.left_column_share = 0.2


@dataclass
class AsynchronousADMMWorkerRPCParams(SolverParams):
    IP: str = "localhost"
    Port: int = 13000
    NumThreads: int = 1
    WorkerID: int = 0
    Backend: ClassVar[str] = ""
    QueueTimeout: float = 1.0
    QuitTimeout: Optional[float] = 30.0
    
    def __post_init__(self):
        self.left_column_share = 0.5


@dataclass
class NetworkUpdate:
    """
    A `NetworkUpdate` type is a tuple containing the following:
        1. Runtime of the inner loop algorithm on the switch
        2. Mean value of null-space shift `Y_k` and its consensus variable `P_k` 
        for all commodities in its partition
        3. The dual variable `u_w`.
        4. The total flow of its partition (the sum of `S_k` for all commodities)
        5. ID of the worker who sent the update
    """
    worker_id: int
    runtime: int
    Y_bar_w: CPUArray
    P_bar_w: CPUArray
    u_w: CPUArray
    Xo_w: CPUArray


@dataclass
class ControllerUpdate:
    """
    A `ControllerUpdate` type is a tuple containing the following:
        1. List of workers that it effects.
        2. Most recent global mean value (`P_bar_t`)
        3. Sample means that generated it (`P_bar_sample`, 'Y_bar_sample` and `u_bar_sample`)
        4. Sample size (number of commodities it effects)
    """
    workers: List[int]
    P_bar_t: CPUArray
    P_bar_sample: CPUArray
    Y_bar_sample: CPUArray
    u_bar_sample: CPUArray
    sample_size: int


class TLVRPCMessages:
    ControllerUpdateType = 0x00
    NetworkUpdateType = 0x01
    
    HEADER_FORMAT = "!HI"
    HEADER_LENGTH = struct.calcsize(HEADER_FORMAT)

    @classmethod
    def get_packet_header(cls, packet: bytes) -> Optional[bytes]:
        if len(packet) >= cls.HEADER_LENGTH:
            return packet[:cls.HEADER_LENGTH]
    
    @classmethod
    def get_packet_rpc_message(cls, packet: bytes) -> Optional[Tuple[int, int, Any]]:
        """
        Assuming `packet` has at least one finished packet, return the RPC message
        type, the length of the packet and its protobuff representation.
        """
        header = cls.get_packet_header(packet)
        if header is not None:
            message_type, message_len = struct.unpack_from(cls.HEADER_FORMAT, header)
            if len(packet) >= message_len:
                message_serialized = packet[cls.HEADER_LENGTH:message_len]
                if message_type == cls.NetworkUpdateType:
                    message = asynchronous_lp_messages.SwitchMessage.FromString(message_serialized)
                elif message_type == cls.ControllerUpdateType:
                    message = asynchronous_lp_messages.ControllerMessage.FromString(message_serialized)
                else:
                    raise ValueError(f'Unexpected update message type: {message_type}')
                return (message_type, message_len, message)

    @classmethod
    def serialize_controller_update(cls, message: asynchronous_lp_messages.ControllerMessage) -> bytes:
        body = message.SerializeToString()
        header = struct.pack(cls.HEADER_FORMAT, cls.ControllerUpdateType, len(body) + cls.HEADER_LENGTH)
        return header + body
    
    @classmethod
    def serialize_network_update(cls, message: asynchronous_lp_messages.SwitchMessage) -> bytes:
        body = message.SerializeToString()
        header = struct.pack(cls.HEADER_FORMAT, cls.NetworkUpdateType, len(body) + cls.HEADER_LENGTH)
        return header + body
