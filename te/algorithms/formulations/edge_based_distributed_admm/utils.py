import numpy as np
import protos.distributed_lp.distributed_lp_pb2 as distributed_lp_messages
from typing import Optional


def array_to_serialized_message(array: Optional[np.ndarray]) -> Optional[distributed_lp_messages.SerializedNumpyArrayMessage]:
    if array is not None:
        return distributed_lp_messages.SerializedNumpyArrayMessage(array=array.tobytes(), dims=list(array.shape))

def serialized_message_to_array(message: Optional[distributed_lp_messages.SerializedNumpyArrayMessage]) -> Optional[np.ndarray]:
    if message is not None:
        return np.reshape(np.frombuffer(message.array), tuple(message.dims))

get_optional_field = lambda request, field_name: getattr(request, field_name) if request.HasField(field_name) else None
