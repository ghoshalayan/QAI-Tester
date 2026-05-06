"""SSE response helper — bridges :class:`TopicBus` to ``EventSourceResponse``.

Wire format (per SSE spec, what the browser sees):

    event: log
    id: 12
    data: {"type":"log","data":{"message":"Hello"},"seq":12,"timestamp":1715000000.123}

Use ``EventSource.addEventListener('log', ...)`` on the frontend to dispatch by
type. The ``id`` field is the bus sequence number, so on reconnect the browser
can pass ``Last-Event-ID`` and we replay only the missed tail.
"""

from __future__ import annotations

import json

from fastapi import Request
from sse_starlette.sse import EventSourceResponse

from app.sse.bus import get_bus


def sse_for_topic(
    request: Request,
    topic: str,
    since_seq: int = 0,
    ping: int = 15,
) -> EventSourceResponse:
    """Stream events from ``topic`` to the client as Server-Sent Events.

    Honors ``Last-Event-ID`` header — if the browser passed it, we resume from
    that seq; the explicit ``since_seq`` query param wins if both are given.
    """
    last_event_id = request.headers.get("last-event-id")
    if last_event_id and since_seq == 0:
        try:
            since_seq = int(last_event_id)
        except ValueError:
            since_seq = 0

    topic_bus = get_bus().topic(topic)

    async def gen():
        async for event in topic_bus.subscribe(since_seq=since_seq):
            yield {
                "event": event.type,
                "id": str(event.seq),
                "data": json.dumps(
                    {
                        "type": event.type,
                        "data": event.data,
                        "seq": event.seq,
                        "timestamp": event.timestamp,
                    }
                ),
            }

    return EventSourceResponse(gen(), ping=ping)
