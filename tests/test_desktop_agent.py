import asyncio
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import api
import desktop_agent_hub
from desktop_agent_hub import DesktopAgentHub


class FakeSocket:
    def __init__(self):
        self.sent = []
        self.closed_with = None

    async def send_text(self, text):
        self.sent.append(text)

    async def close(self, code=None, reason=None):
        self.closed_with = code


# ---------- DesktopAgentHub, in isolation (no network/ASGI involved) ----------


def test_hub_status_when_never_connected():
    hub = DesktopAgentHub()
    assert hub.status() == {"connected": False, "connected_since": None, "agent": None}


def test_hub_register_marks_connected_with_agent_info():
    hub = DesktopAgentHub()
    socket = FakeSocket()
    asyncio.run(hub.register(socket, {"hostname": "abhis-mac"}))
    status = hub.status()
    assert status["connected"] is True
    assert status["agent"]["hostname"] == "abhis-mac"
    assert status["connected_since"] is not None


def test_hub_unregister_marks_disconnected():
    hub = DesktopAgentHub()
    socket = FakeSocket()
    asyncio.run(hub.register(socket, {"hostname": "x"}))
    asyncio.run(hub.unregister(socket))
    assert hub.status() == {"connected": False, "connected_since": None, "agent": None}


def test_hub_unregister_ignores_a_stale_socket():
    # A previously-replaced connection's own disconnect handler firing late
    # must not clobber whatever is currently registered.
    hub = DesktopAgentHub()
    old, new = FakeSocket(), FakeSocket()
    asyncio.run(hub.register(old, {"hostname": "old"}))
    asyncio.run(hub.register(new, {"hostname": "new"}))
    asyncio.run(hub.unregister(old))
    assert hub.status()["connected"] is True
    assert hub.status()["agent"]["hostname"] == "new"


def test_hub_register_closes_the_previous_connection():
    hub = DesktopAgentHub()
    old, new = FakeSocket(), FakeSocket()
    asyncio.run(hub.register(old, {"hostname": "old"}))
    asyncio.run(hub.register(new, {"hostname": "new"}))
    assert old.closed_with == 4409


def test_hub_call_raises_when_not_connected():
    hub = DesktopAgentHub()
    with pytest.raises(ConnectionError):
        asyncio.run(hub.call("ping"))


def test_hub_call_times_out_when_agent_never_responds(monkeypatch):
    monkeypatch.setattr(desktop_agent_hub, "REQUEST_TIMEOUT_SECONDS", 0.05)
    hub = DesktopAgentHub()
    socket = FakeSocket()
    asyncio.run(hub.register(socket, {"hostname": "x"}))
    with pytest.raises(TimeoutError):
        asyncio.run(hub.call("ping"))


def test_hub_call_resolves_once_a_matching_response_arrives():
    hub = DesktopAgentHub()
    socket = FakeSocket()
    asyncio.run(hub.register(socket, {"hostname": "x"}))

    async def scenario():
        call_task = asyncio.create_task(hub.call("ping", {"a": 1}))
        await asyncio.sleep(0.01)  # let call() send its request and start waiting
        sent = json.loads(socket.sent[0])
        assert sent["capability"] == "ping"
        assert sent["params"] == {"a": 1}
        await hub.handle_message(json.dumps({
            "request_id": sent["request_id"], "ok": True, "result": {"message": "pong"},
        }))
        return await call_task

    result = asyncio.run(scenario())
    assert result["ok"] is True
    assert result["result"] == {"message": "pong"}


def test_hub_handle_message_ignores_malformed_json():
    hub = DesktopAgentHub()
    # Must not raise — a corrupt frame from the agent shouldn't take down
    # the connection or crash whatever's awaiting a different request.
    asyncio.run(hub.handle_message("not json"))


def test_hub_handle_message_ignores_unknown_request_id():
    hub = DesktopAgentHub()
    asyncio.run(hub.handle_message(json.dumps({"request_id": "does-not-exist", "ok": True})))


def test_hub_unregister_fails_pending_calls():
    hub = DesktopAgentHub()
    socket = FakeSocket()
    asyncio.run(hub.register(socket, {"hostname": "x"}))

    async def scenario():
        call_task = asyncio.create_task(hub.call("ping"))
        await asyncio.sleep(0.01)
        await hub.unregister(socket)
        with pytest.raises(ConnectionError):
            await call_task

    asyncio.run(scenario())


# ---------- The real WebSocket + REST endpoints, end to end ----------


def _wait_until(predicate, timeout=2.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture(autouse=True)
def desktop_agent_test_key(monkeypatch):
    monkeypatch.setattr(api, "DESKTOP_AGENT_KEY", "secret123")
    yield
    # Endpoint tests connect real (in-process) sockets — make sure a test
    # that fails mid-connection can't leave the hub "connected" for the
    # next test to trip over.
    asyncio.run(api.desktop_agent_hub.unregister(api.desktop_agent_hub._socket))


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    yield
    api.app.dependency_overrides.clear()


def test_ws_rejects_connection_with_no_token():
    client = TestClient(api.app)
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/desktop-agent/ws"):
            pass
    assert exc_info.value.code == 4401


def test_ws_rejects_connection_with_wrong_token():
    client = TestClient(api.app)
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/desktop-agent/ws", headers={"Authorization": "Bearer wrong"}):
            pass
    assert exc_info.value.code == 4401


def test_desktop_agent_status_requires_auth():
    client = TestClient(api.app)
    res = client.get("/desktop-agent/status")
    assert res.status_code == 401


def test_desktop_agent_status_reports_not_connected_by_default():
    api.app.dependency_overrides[api.require_user] = lambda: "test-user"
    client = TestClient(api.app)
    res = client.get("/desktop-agent/status")
    assert res.status_code == 200
    assert res.json()["connected"] is False


def test_desktop_agent_ping_returns_503_when_not_connected():
    api.app.dependency_overrides[api.require_user] = lambda: "test-user"
    client = TestClient(api.app)
    res = client.post("/desktop-agent/ping")
    assert res.status_code == 503


def test_ws_connect_and_hello_updates_status_then_reverts_on_disconnect():
    api.app.dependency_overrides[api.require_user] = lambda: "test-user"
    with TestClient(api.app) as client:
        with client.websocket_connect("/desktop-agent/ws", headers={"Authorization": "Bearer secret123"}) as ws:
            ws.send_text(json.dumps({"hostname": "abhis-mac", "platform": "macOS-14", "capabilities": ["ping"]}))
            assert _wait_until(lambda: api.desktop_agent_hub.is_connected())

            status = client.get("/desktop-agent/status").json()
            assert status["connected"] is True
            assert status["agent"]["hostname"] == "abhis-mac"
            assert status["agent"]["capabilities"] == ["ping"]

        assert _wait_until(lambda: not api.desktop_agent_hub.is_connected())


def test_desktop_agent_ping_round_trip_through_a_real_socket():
    # This exercises the whole real path (WS auth, hello, hub state, an
    # HTTP request blocking on a live reply from a connected socket) rather
    # than mocking any piece of it — the closest thing to a live desktop
    # agent this test suite can run without an actual Mac process attached.
    api.app.dependency_overrides[api.require_user] = lambda: "test-user"
    with TestClient(api.app) as client:
        with client.websocket_connect("/desktop-agent/ws", headers={"Authorization": "Bearer secret123"}) as ws:
            ws.send_text(json.dumps({"hostname": "abhis-mac", "platform": "macOS-14", "capabilities": ["ping"]}))
            assert _wait_until(lambda: api.desktop_agent_hub.is_connected())

            result_holder = {}

            def do_ping():
                result_holder["res"] = client.post("/desktop-agent/ping")

            thread = threading.Thread(target=do_ping)
            thread.start()

            incoming = json.loads(ws.receive_text())
            assert incoming["capability"] == "ping"
            ws.send_text(json.dumps({
                "request_id": incoming["request_id"],
                "ok": True,
                "result": {"message": "pong", "hostname": "abhis-mac"},
            }))

            thread.join(timeout=5)
            res = result_holder["res"]
            assert res.status_code == 200
            assert res.json()["result"] == {"message": "pong", "hostname": "abhis-mac"}


def test_desktop_agent_ping_times_out_if_agent_never_replies(monkeypatch):
    monkeypatch.setattr(desktop_agent_hub, "REQUEST_TIMEOUT_SECONDS", 0.2)
    api.app.dependency_overrides[api.require_user] = lambda: "test-user"
    with TestClient(api.app) as client:
        with client.websocket_connect("/desktop-agent/ws", headers={"Authorization": "Bearer secret123"}) as ws:
            ws.send_text(json.dumps({"hostname": "abhis-mac", "platform": "macOS-14", "capabilities": ["ping"]}))
            assert _wait_until(lambda: api.desktop_agent_hub.is_connected())

            res = client.post("/desktop-agent/ping")
            assert res.status_code == 504
