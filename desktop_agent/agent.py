"""Local desktop agent — run this ON YOUR OWN MAC, never in the cloud.

Architecture: Browser -> FastAPI backend (Render) -> desktop_agent_hub ->
this process -> macOS. The backend is a cloud container with no way to
reach into your home network, so this agent connects OUTBOUND to the
backend's WebSocket endpoint and authenticates with DESKTOP_AGENT_KEY, a
secret that must match what the backend has configured and must never be
committed or shared. Once connected, the backend can ask this process to
run one of the capabilities registered in capabilities.py — nothing else.
This process never accepts or evaluates an arbitrary command string.

Required environment variables (set them in a local .env next to this
file, or export them in your shell):
  DESKTOP_AGENT_KEY   shared secret — must match the backend's env var of
                       the same name. Generate one with, e.g.:
                       python3 -c "import secrets; print(secrets.token_urlsafe(32))"
  BACKEND_WS_URL       e.g. ws://localhost:8000/desktop-agent/ws while
                       developing, or wss://your-app.onrender.com/desktop-agent/ws
                       once deployed. Defaults to the local dev URL.

Run with: python3 desktop_agent/agent.py
"""

import asyncio
import json
import logging
import os
import platform

import websockets
from dotenv import load_dotenv

from capabilities import CAPABILITIES

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("desktop_agent")

DESKTOP_AGENT_KEY = os.environ.get("DESKTOP_AGENT_KEY")
BACKEND_WS_URL = os.environ.get("BACKEND_WS_URL", "ws://localhost:8000/desktop-agent/ws")
RECONNECT_DELAY_SECONDS = 5


async def handle_request(websocket, message: dict) -> None:
    request_id = message.get("request_id")
    capability_name = message.get("capability")
    params = message.get("params") or {}

    handler = CAPABILITIES.get(capability_name)
    if handler is None:
        response = {"request_id": request_id, "ok": False, "error": f"Unknown capability '{capability_name}'"}
    else:
        try:
            result = handler(**params)
            response = {"request_id": request_id, "ok": True, "result": result}
        except Exception as e:
            logger.exception("capability '%s' failed", capability_name)
            response = {"request_id": request_id, "ok": False, "error": str(e)}

    await websocket.send(json.dumps(response))


async def run_once() -> None:
    hello = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "capabilities": list(CAPABILITIES),
    }
    async with websockets.connect(
        BACKEND_WS_URL,
        additional_headers={"Authorization": f"Bearer {DESKTOP_AGENT_KEY}"},
    ) as websocket:
        await websocket.send(json.dumps(hello))
        logger.info("connected to %s as %s", BACKEND_WS_URL, hello["hostname"])
        async for raw in websocket:
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("ignoring malformed message from backend: %r", raw[:200])
                continue
            await handle_request(websocket, message)


async def main() -> None:
    if not DESKTOP_AGENT_KEY:
        logger.error("DESKTOP_AGENT_KEY is not set — refusing to start without authentication configured.")
        return
    while True:
        try:
            await run_once()
        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            logger.warning("disconnected (%s) — reconnecting in %ss", e, RECONNECT_DELAY_SECONDS)
        except Exception:
            logger.exception("unexpected error — reconnecting in %ss", RECONNECT_DELAY_SECONDS)
        await asyncio.sleep(RECONNECT_DELAY_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
