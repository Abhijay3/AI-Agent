"""Tracks the single authenticated local desktop agent connection and
routes capability requests to it.

Architecture: Browser -> FastAPI backend -> [this hub] -> local desktop
agent process running on the user's own Mac (see desktop_agent/agent.py).
The backend is a cloud container with no way to reach into a home
network, so the agent connects OUTBOUND to the backend's WebSocket
endpoint and authenticates with DESKTOP_AGENT_KEY — a secret that lives
only in the backend's environment and the agent's local environment, and
is never sent to or held by the browser. The browser only ever talks to
the backend over its normal per-user session token; it has no path to the
desktop agent that bypasses this hub.

Only one agent connection is trusted at a time — a new authenticated
connection replaces whatever was there before, so there's never ambiguity
about which machine a request is actually going to.

The hub only ever asks the agent to run one of the capabilities the agent
itself declared in its hello message (see desktop_agent/capabilities.py on
the agent side) — it has no way to send it an arbitrary command, and
nothing here executes a shell command directly.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger("acme_support_agent")

DESKTOP_AGENT_KEY = os.environ.get("DESKTOP_AGENT_KEY")

# How long we'll wait for the agent to answer a single capability request
# before giving up — an agent that's connected but hung, or a network
# blip, must not hang the caller's HTTP request forever.
REQUEST_TIMEOUT_SECONDS = 15


class DesktopAgentHub:
    def __init__(self) -> None:
        self._socket = None
        self._connected_at: Optional[float] = None
        self._agent_info: dict = {}
        self._pending: dict = {}
        # The event loop that owns this hub's async state — captured on
        # register() (called from the WS handler, so it's always the real
        # running loop). call_sync() needs this to safely bridge in from a
        # different thread; asyncio.Future results can't be delivered
        # correctly across threads/loops without going through it.
        self._loop = None

    def is_connected(self) -> bool:
        return self._socket is not None

    def status(self) -> dict:
        return {
            "connected": self.is_connected(),
            "connected_since": self._connected_at,
            "agent": dict(self._agent_info) if self.is_connected() else None,
        }

    async def register(self, socket, agent_info: dict) -> None:
        # Replace, don't stack — an old connection left open after a new
        # one authenticates would just be a stale, unusable duplicate.
        if self._socket is not None and self._socket is not socket:
            await self._close_quietly(self._socket)
        self._socket = socket
        self._connected_at = time.time()
        self._agent_info = agent_info
        self._pending = {}
        self._loop = asyncio.get_running_loop()

    async def unregister(self, socket) -> None:
        if self._socket is not socket:
            return
        self._socket = None
        self._connected_at = None
        self._agent_info = {}
        # Nothing is left to answer any request still in flight.
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ConnectionError("Desktop agent disconnected"))
        self._pending = {}

    async def handle_message(self, raw: str) -> None:
        try:
            message = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("desktop agent sent a malformed frame: %r", raw[:200])
            return
        request_id = message.get("request_id")
        future = self._pending.pop(request_id, None) if request_id else None
        if future is not None and not future.done():
            future.set_result(message)

    async def call(self, capability: str, params: Optional[dict] = None) -> dict:
        if not self.is_connected():
            raise ConnectionError("Desktop agent is not connected")
        request_id = f"{time.monotonic_ns():x}"
        future = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._socket.send_text(json.dumps({
                "request_id": request_id,
                "capability": capability,
                "params": params or {},
            }))
            return await asyncio.wait_for(future, timeout=REQUEST_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Desktop agent didn't respond to '{capability}' in time")
        finally:
            self._pending.pop(request_id, None)

    def call_sync(self, capability: str, params: Optional[dict] = None) -> dict:
        """Synchronous bridge for code that isn't running on this hub's
        event loop — e.g. a ThreadPoolExecutor worker in agent_core's
        tool-calling loop, where every existing tool handler is a plain
        blocking function, not a coroutine. asyncio.Future results can't be
        delivered correctly across threads/event loops on their own
        (resolving one from the wrong thread doesn't reliably wake up
        whatever is awaiting it) — run_coroutine_threadsafe is the correct,
        supported way to schedule real async work from another thread and
        block that thread for the result.

        concurrent.futures.TimeoutError and the builtin TimeoutError are
        the same class in Python 3.11+ but distinct ones in 3.9 (this
        project's local dev version) — catch both explicitly so the
        caller only ever needs to handle one, version-independent
        TimeoutError.
        """
        if self._loop is None:
            raise ConnectionError("Desktop agent is not connected")
        future = asyncio.run_coroutine_threadsafe(self.call(capability, params), self._loop)
        try:
            # A few seconds longer than call()'s own internal timeout, so
            # that timeout (which raises a properly-messaged TimeoutError
            # from inside the coroutine) fires first in normal operation —
            # this outer one is only a defensive backstop.
            return future.result(timeout=REQUEST_TIMEOUT_SECONDS + 5)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Desktop agent didn't respond to '{capability}' in time")

    async def _close_quietly(self, socket) -> None:
        try:
            await socket.close(code=4409, reason="Replaced by a newer connection")
        except Exception:
            pass


# One hub for the whole process — this app is single-operator (one Mac,
# one agent), so a module-level singleton is the right amount of state,
# not a premature abstraction for multi-tenant agent management.
hub = DesktopAgentHub()
