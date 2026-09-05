import struct
from typing import Optional, Tuple, Any
import protos.distributed_lp.distributed_lp_pb2 as distributed_lp_messages


class TLVRPCMessages:
    DoInnerLoops = 0x00
    UpdateNetworkNodes = 0x01
    
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
                if message_type == cls.DoInnerLoops:
                    message = distributed_lp_messages.NetworkUpdateRequest.FromString(message_serialized)
                elif message_type == cls.UpdateNetworkNodes:
                    message = distributed_lp_messages.UpdateMessage.FromString(message_serialized)
                else:
                    raise ValueError(f'Unexpected RPC message type: {message_type}')
                return (message_type, message_len, message)

    @classmethod
    def serialize_do_inner_loop(cls, message: distributed_lp_messages.NetworkUpdateRequest) -> bytes:
        body = message.SerializeToString()
        header = struct.pack(cls.HEADER_FORMAT, cls.DoInnerLoops, len(body) + cls.HEADER_LENGTH)
        return header + body
    
    @classmethod
    def serialize_update_network_nodes(cls, message: distributed_lp_messages.UpdateMessage) -> bytes:
        body = message.SerializeToString()
        header = struct.pack(cls.HEADER_FORMAT, cls.UpdateNetworkNodes, len(body) + cls.HEADER_LENGTH)
        return header + body