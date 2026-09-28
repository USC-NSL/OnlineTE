import asyncio
from typing import List, Optional, Set, Dict, Tuple, Callable, Awaitable
from array_utils.cpu.types import *
from array_utils.cpu.grpc_utils import *
from utils.logging import as_fail, as_warning


class PartialBarrier[ScatterType, GatherType, StoreType]:
    """
    Implements a Partial Barrier for a scatter-gather operation.
    The idea is to have a separate partial barrier object for
    every scatter-gather operation type when needed.

    Operation
    ---------
    The barrier maintains internal state for tracking the progress
    of each broadcast operation. Each endpoint will have only at
    most one in-progress task at a time.

    The set of nodes with no pending request is called the "arrival set".
    
    - At each generation, the barrier broadcasts **only to the arrival set**
    - The barrier then waits until all the following are true:
        - The arrival set has at least `min_arrivals` members
        - No endpoint exists that has not arrived in `max_lag` iterations
          or more
        - No endpoint exists that has not arrived even once
    - When the barrier breaks, responses are stored in an internal dict object
      for each node.

    Scatter Operation
    -----------------
    A request of type `ScatterType` is broadcast to the arrival set. Currently, before
    the scatter operation is done, the barrier quickly checks for any newly arrived
    nodes and quickly adds them to the arrival set, which means that they will also
    receive the scatter request.
    
    Gather Operation
    ----------------
    For gather operations, a response of type `GatherType` will be accumulated over
    time until the barrier breaks.
    The user can provide a store operation callable that takes each `GatherType`
    object as input and stores the output (e.g. deserializing data from the response
    and storing the result).

    Types
    -----
    - `ScatterType` is the type of the message we scatter to the current arrival set.
    - `GatherType` is the type of received messages from each endpoint.
    - `StoreType` is the type of the data stored for each endpoint during gather. This
      can simply be `GatherType`, indicating that responses are stored without modification.
    - The store operation (if exists), takes a `GatherType` as input as returns a `StoreType`
      as output.
    """
    def __init__(self,
        number_of_endpoints: int,
        min_arrival: Optional[int],
        max_lag: int,
        event_loop: asyncio.AbstractEventLoop
    ):
        self._number_of_endpoints = number_of_endpoints
        self._min_arrival = min_arrival if min_arrival is not None else number_of_endpoints
        self._max_lag = max_lag
        self._event_loop = event_loop

        self._clock: int = 0
        """Local clock that keeps track of request arrivals"""
        self._gather_tasks: Dict[int, Tuple[asyncio.Task, int]] = dict()
        """Maps endpoint ID to a pair of task and start clock"""
        self._scatter_tasks: List[asyncio.Task] = []
        """List of scatter tasks to be awaited on next iteration"""
        self._arrival_set: Set[int] = set([i for i in range(number_of_endpoints)])
        """Set of node IDs that arrived since last iteration"""
        self._storage: List[Optional[StoreType]] = [None] * number_of_endpoints
        """List of the last stored value for each endpoint"""
        self._initial_response_set: Set[int] = {i for i in range(number_of_endpoints)}
        """A house-keeping attribute to quickly check if we have at least one response"""

        self._active: bool = True

    @property
    def number_of_endpoints(self) -> int:
        return self._number_of_endpoints
    @property
    def min_arrival(self) -> int:
        return self._min_arrival
    @property
    def max_lag(self) -> int:
        return self._max_lag

    def start_barrier(self):
        self._active = True
        if self._min_arrival == self._number_of_endpoints:
            print(as_warning(f'Partial Barrier will operate synchronosuly'))
        else:
            print(as_warning(f'Partial Barrier operates asynchronously'))

    def break_barrier(self):
        self._active = False

    async def _scatter_gather(self,
        message: ScatterType,
        node_coroutine: Callable[
            [int, ScatterType],
            Awaitable[GatherType]
        ],
        store_operation: Callable[[GatherType], StoreType]
    ):
        # Broadcast to any arrived node and increment the local clock
        for node_id in range(self.number_of_endpoints):
            if node_id in self._arrival_set:
                task = asyncio.create_task(node_coroutine(node_id, message))
                self._gather_tasks[node_id] = (task, int(self._clock))
                self._arrival_set.discard(node_id)
        self._clock += 1
        
        # Gather finished responses until barrier breaks
        current_batch_responses: List[Tuple[GatherType, int]] = []
        while self._active:
            finished_this_loop = [
                node_id for node_id, (task, _) in self._gather_tasks.items() if task.done()
            ]

            for node_id in finished_this_loop:
                task, _ = self._gather_tasks.pop(node_id)
                try:
                    res = await task
                    if res is not None:
                        current_batch_responses.append((res, node_id))
                except Exception as e:
                    print(as_fail(f'Failure while awaiting finished update on node {node_id}: {e}'))

            # Evaluate the barrier condition
            stale_nodes = [
                node_id for node_id, (_, start_time) in self._gather_tasks.items()
                if (self._clock - start_time) > self._max_lag
            ]

            if (
                len(self._arrival_set) >= self._min_arrival and \
                not stale_nodes and \
                len(self._initial_response_set) == 0
            ):
                break

            # If we aren't done, wait for the next task to finish
            # We only wait on tasks that are actually running
            if self._gather_tasks:
                tasks_to_watch = [t for t, _ in self._gather_tasks.values()]
                await asyncio.wait(tasks_to_watch, return_when=asyncio.FIRST_COMPLETED)
            else:
                # Safety break if no tasks are left but conditions aren't met
                break

        # Store the result
        for response, node_id in current_batch_responses:
            self._storage[node_id] = store_operation(response)
            if len(self._initial_response_set) > 0:
                self._initial_response_set.discard(node_id)
            assert node_id not in self._arrival_set
            self._arrival_set.add(node_id)
        return self._storage

    def scatter_gather(self,
        message: ScatterType,
        node_coroutine: Callable[
            [int, ScatterType],
            Awaitable[GatherType]
        ],
        store_operation: Callable[[StoreType], StoreType]
    ) -> List[StoreType]:
        return self._event_loop.run_until_complete(self._scatter_gather(
            message=message, node_coroutine=node_coroutine,
            store_operation=store_operation
        ))
    