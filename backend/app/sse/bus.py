"""In-memory pub/sub bus that bridges sync producers to async SSE consumers.

Topics are arbitrary strings — by convention:

- ``run:<run_id>``        — events for a single test run (logs, results, status)
- ``project:<project_id>`` — project-level events (suite created, doc uploaded)

Producer side
-------------
Background threads (Playwright runner, agent nodes) call ``bus.publish(topic,
type, data)``. ``publish`` is sync and thread-safe — it grabs a lock,
appends to history, then schedules ``put_nowait`` on every subscriber's queue
via ``loop.call_soon_threadsafe``.

Consumer side
-------------
SSE endpoints (FastAPI) call ``async for event in bus.topic(topic).subscribe():``.
History is replayed first (so a reconnecting client can pick up where it
left off using ``since_seq``), then live events stream in.

Cleanup
-------
When the consumer's coroutine ends (client disconnects, connection drops,
``"done"`` received and client closes), the subscriber removes itself in a
``finally`` block. No leaks. Empty topics still hold history for late
subscribers; call ``bus.remove(topic)`` after a run is fully done with.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Event:
    type: str
    data: dict[str, Any]
    seq: int
    timestamp: float = field(default_factory=time.time)


class TopicBus:
    """A single pub/sub topic with bounded replayable history."""

    def __init__(self, max_history: int = 5000):
        self._history: deque[Event] = deque(maxlen=max_history)
        self._seq: int = 0
        self._subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self._lock = threading.Lock()

    @property
    def last_seq(self) -> int:
        return self._seq

    @property
    def history_size(self) -> int:
        return len(self._history)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def publish(self, event_type: str, data: dict[str, Any] | None = None) -> Event:
        """Thread-safe publish. Callable from sync OR async code."""
        with self._lock:
            self._seq += 1
            event = Event(type=event_type, data=data or {}, seq=self._seq)
            self._history.append(event)

            dead: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
            for loop, queue in self._subscribers:
                if loop.is_closed():
                    dead.append((loop, queue))
                    continue
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, event)
                except RuntimeError:
                    # loop stopped between is_closed() check and call_soon
                    dead.append((loop, queue))
                except Exception as e:
                    logger.warning("SSE publish to subscriber failed: %s", e)
                    dead.append((loop, queue))

            for d in dead:
                try:
                    self._subscribers.remove(d)
                except ValueError:
                    pass

        return event

    async def subscribe(self, since_seq: int = 0) -> AsyncIterator[Event]:
        """Yield missed history (events with seq > since_seq), then live events."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Event] = asyncio.Queue()
        entry = (loop, queue)

        with self._lock:
            for event in self._history:
                if event.seq > since_seq:
                    queue.put_nowait(event)
            self._subscribers.append(entry)

        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            with self._lock:
                try:
                    self._subscribers.remove(entry)
                except ValueError:
                    pass


class EventBus:
    """Registry of TopicBus by topic key."""

    def __init__(self):
        self._topics: dict[str, TopicBus] = {}
        self._lock = threading.Lock()

    def topic(self, key: str) -> TopicBus:
        with self._lock:
            if key not in self._topics:
                self._topics[key] = TopicBus()
            return self._topics[key]

    def publish(
        self, topic: str, event_type: str, data: dict[str, Any] | None = None,
    ) -> Event:
        return self.topic(topic).publish(event_type, data)

    def remove(self, topic: str) -> None:
        """Drop a topic and its history. Use only after a run is fully done."""
        with self._lock:
            self._topics.pop(topic, None)

    def list_topics(self) -> list[dict[str, Any]]:
        """Diagnostic snapshot — topic key, last seq, history size, subscriber count."""
        with self._lock:
            return [
                {
                    "topic": key,
                    "last_seq": tb.last_seq,
                    "history_size": tb.history_size,
                    "subscribers": tb.subscriber_count,
                }
                for key, tb in self._topics.items()
            ]


_bus = EventBus()


def get_bus() -> EventBus:
    return _bus
