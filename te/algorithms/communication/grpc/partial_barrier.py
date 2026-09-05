import asyncio
from typing import List, Optional, Any, Set, Dict, Tuple, Callable, Awaitable
from array_utils.cpu.types import *
from array_utils.cpu.grpc_utils import *
from utils.logging import as_fail, as_warning


class PartialBarrier[GatherRequest, GatherResponse, StoreType, ScatterRequest]:
    """
    Implements a Partial Barrier for a scatter-gather operation.
    The idea is to have a separate partial barrier object for
    every scatter-gather operation type when needed.

    Operation
    ---------
    The barrier maintains internal state for tracking the progress
    of each broadcast operation. Each endpoint will have only at
    most one in-progress task at a time.
    
    - At each generation, the barrier broadcasts **only to non-pending nodes**
    - The barrier then waits until all the following are true:
        - At least `min_arrival` endpoints have responded
        - No endpoint exists that has not responded in `max_lag` iterations
          or more
        - No endpoint exists that has not responded even once
    - When the barrier breaks, responses are stored in an internal dict object
      for each node.
    
    Gather Operation
    ----------------
    For gather operations, a request of type `GatherType` will be broadcast
    to all non-pending endpoints. We accumulate responses of type `GatherResponse`
    until the barrier opens.
    The user can provide a store operation callable that takes each `GatherResponse`
    object as input and stores the output (e.g. deserializing data from the response
    and storing the result).
    When a gather operation starts again, **the set of previous arrivals will be cleared**.

    Scatter Operation
    -----------------
    For a scatter operation, the barrier no longer waits and merely broadcasts to
    some nodes.
    - The barrier **does not wait for individual scatter operations to finish**.
    - Scatter operations **only target arrived nodes from a previous gather operation**.
    - Each scatter task is only awaited when the next scatter request has been
      issued.

    Types
    -----
    - `GatherRequest` is the type of gather request that we broadcast to
      all endpoits.
    - `GatherResponse` is the type of received messages from each endpoint
    - `StoreType` is the type of the data stored for each endpoint during gather.
      When this is `GatherResponse`, we just store the original response and leave
      any processing to the caller.
    - `ScatterRequest` is the type of value that we scatter to all arrived nodes.
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
        self._arrival_set: Set[int] = set()
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

    async def _gather(self,
        message: GatherRequest,
        node_coroutine: Callable[
            [int, GatherRequest],
            Awaitable[GatherResponse]
        ],
        store_operation: Callable[[GatherResponse], StoreType]
    ):
        # Clear arrival set
        self._arrival_set.clear()

        # Broadcast to any non-pending node and increment the local clock
        for node_id in range(self.number_of_endpoints):
            if node_id not in self._gather_tasks:
                task = asyncio.create_task(node_coroutine(node_id, message))
                self._gather_tasks[node_id] = (task, int(self._clock))
        self._clock += 1
        
        # Gather finished responses since last time
        current_batch_responses: List[Tuple[GatherResponse, int]] = []
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
                len(current_batch_responses) >= self._min_arrival and \
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
            self._arrival_set.add(node_id)
        return self._storage

    def gather(self,
        message: GatherRequest,
        node_coroutine: Callable[
            [int, GatherRequest],
            Awaitable[GatherResponse]
        ],
        store_operation: Callable[[GatherResponse], StoreType]
    ) -> List[StoreType]:
        return self._event_loop.run_until_complete(self._gather(
            message=message, node_coroutine=node_coroutine,
            store_operation=store_operation
        ))

    async def _scatter(self,
        message: ScatterRequest,
        node_coroutine: Callable[
            [int, ScatterRequest],
            Awaitable[Any]
        ]
    ):
        assert len(self._arrival_set) > 0
        # Check for previous scatter
        pending_scatters = []
        for task in self._scatter_tasks:
            if task.done():
                try:
                    await task
                except Exception as e:
                    print(as_fail(f'Failure while awaiting scatter update: {e}'))
            else:
                pending_scatters.append(task)

        # Scatter to the new arrivals
        for node_id in self._arrival_set:
            pending_scatters.append(asyncio.create_task(node_coroutine(node_id, message)))
        self._scatter_tasks = pending_scatters

    def scatter(self,
        message: ScatterRequest,
        node_coroutine: Callable[
            [int, ScatterRequest],
            Awaitable[Any]
        ]
    ):
        self._event_loop.run_until_complete(self._scatter(
            message=message,
            node_coroutine=node_coroutine
        ))
    