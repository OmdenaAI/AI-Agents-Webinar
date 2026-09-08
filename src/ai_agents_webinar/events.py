"""
The event stream the audience UI consumes.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from typing import Any, Iterator

# Enough backlog that someone opening the page mid-scenario sees how it got here.
BACKLOG = 200

_lock = threading.Lock()
_recent: deque[dict] = deque(maxlen=BACKLOG)
_subscribers: set[queue.Queue] = set()


def publish(event: dict) -> dict:
    """Fan out one event. Never raises — a broken display must not stop a run."""
    with _lock:
        _recent.append(event)
        targets = list(_subscribers)
    for q in targets:
        try:
            q.put_nowait(event)
        except queue.Full:
            pass  
    return event


def recent() -> list[dict]:
    with _lock:
        return list(_recent)


def subscribe(maxsize: int = 500) -> tuple[queue.Queue, Any]:
    q: queue.Queue = queue.Queue(maxsize=maxsize)
    with _lock:
        _subscribers.add(q)

    def unsubscribe() -> None:
        with _lock:
            _subscribers.discard(q)

    return q, unsubscribe


def stream(timeout: float = 15.0) -> Iterator[dict | None]:
    """Yields events, or None as a keepalive so a proxy will not close the pipe."""
    q, unsubscribe = subscribe()
    try:
        for event in recent():
            yield event
        while True:
            try:
                yield q.get(timeout=timeout)
            except queue.Empty:
                yield None
    finally:
        unsubscribe()


def reset() -> None:
    """Clear the backlog between scenarios so the screen starts clean."""
    with _lock:
        _recent.clear()
