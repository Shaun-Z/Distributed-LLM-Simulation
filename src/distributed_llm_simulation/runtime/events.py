import heapq
import itertools
import math
from dataclasses import dataclass, field


@dataclass(order=True, frozen=True)
class Event:
    time_s: float
    priority: int
    seq: int
    kind: str = field(compare=False)
    activity_id: str | None = field(default=None, compare=False)
    generation: int = field(default=0, compare=False)
    payload: object = field(default=None, compare=False)


class EventQueue:
    def __init__(self):
        self.heap = []
        self.sequence = itertools.count()

    def push(self, time_s, kind, activity_id=None, generation=0, payload=None):
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError(f"invalid event time {time_s}")
        priority = {
            "PREDICTED": 0,
            "PHASE_READY": 1,
            "CAPACITY_CHANGE": 2,
            "ARRIVAL": 3,
            "INIT": 4,
        }[kind]
        event = Event(time_s, priority, next(self.sequence), kind, activity_id, generation, payload)
        heapq.heappush(self.heap, event)
        return event

    def pop(self):
        return heapq.heappop(self.heap)

    def __bool__(self):
        return bool(self.heap)
