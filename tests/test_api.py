import glob
import json
import os

import pytest
from fastapi.testclient import TestClient

import api
from tools import PDF_UPLOAD_DIR


@pytest.fixture(autouse=True)
def cleanup_test_uploads():
    yield
    for path in glob.glob(os.path.join(PDF_UPLOAD_DIR, "*_test_*.pdf")):
        os.remove(path)


def make_client(monkeypatch):
    monkeypatch.setattr(api, "run_turn", lambda messages, user_id: "mocked reply")
    monkeypatch.setattr(api, "load_history", lambda session_id: [])
    monkeypatch.setattr(api, "save_history", lambda session_id, messages: None)
    return TestClient(api.app)


def test_health_needs_no_auth(monkeypatch):
    client = make_client(monkeypatch)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_chat_rejects_missing_api_key(monkeypatch):
    client = make_client(monkeypatch)
    res = client.post("/chat", json={"session_id": "s1", "user_id": "u1", "message": "hi"})
    assert res.status_code == 401


def test_chat_rejects_wrong_api_key(monkeypatch):
    client = make_client(monkeypatch)
    res = client.post(
        "/chat",
        json={"session_id": "s1", "user_id": "u1", "message": "hi"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert res.status_code == 401


def test_chat_accepts_correct_api_key(monkeypatch):
    client = make_client(monkeypatch)
    res = client.post(
        "/chat",
        json={"session_id": "s1", "user_id": "u1", "message": "hi"},
        headers={"X-API-Key": "test-app-key"},
    )
    assert res.status_code == 200
    assert res.json() == {"reply": "mocked reply"}


def test_index_injects_api_key(monkeypatch):
    client = make_client(monkeypatch)
    res = client.get("/")
    assert res.status_code == 200
    assert "test-app-key" in res.text
    assert "__APP_API_KEY__" not in res.text


def test_chat_stream_rejects_missing_api_key(monkeypatch):
    client = make_client(monkeypatch)
    res = client.post("/chat/stream", json={"session_id": "s1", "user_id": "u1", "message": "hi"})
    assert res.status_code == 401


def test_chat_stream_yields_ndjson_events(monkeypatch):
    client = make_client(monkeypatch)

    def fake_stream_turn(messages, user_id):
        yield {"event": "tool_call", "tool": "calculator"}
        yield {"event": "token", "text": "Hello"}
        yield {"event": "token", "text": " world"}

    monkeypatch.setattr(api, "stream_turn", fake_stream_turn)

    res = client.post(
        "/chat/stream",
        json={"session_id": "s1", "user_id": "u1", "message": "hi"},
        headers={"X-API-Key": "test-app-key"},
    )
    assert res.status_code == 200
    events = [json.loads(line) for line in res.text.strip().split("\n") if line.strip()]
    assert events == [
        {"event": "tool_call", "tool": "calculator"},
        {"event": "token", "text": "Hello"},
        {"event": "token", "text": " world"},
    ]


def test_chat_stream_reports_error_event_on_exception(monkeypatch):
    client = make_client(monkeypatch)

    def fake_stream_turn(messages, user_id):
        yield {"event": "token", "text": "partial"}
        raise RuntimeError("boom")

    monkeypatch.setattr(api, "stream_turn", fake_stream_turn)

    res = client.post(
        "/chat/stream",
        json={"session_id": "s1", "user_id": "u1", "message": "hi"},
        headers={"X-API-Key": "test-app-key"},
    )
    assert res.status_code == 200
    events = [json.loads(line) for line in res.text.strip().split("\n") if line.strip()]
    assert events[0] == {"event": "token", "text": "partial"}
    assert events[-1]["event"] == "error"


def test_upload_rejects_missing_api_key(monkeypatch):
    client = make_client(monkeypatch)
    res = client.post("/upload", files={"file": ("test_doc.pdf", b"%PDF-1.4 fake", "application/pdf")})
    assert res.status_code == 401


def test_upload_rejects_non_pdf(monkeypatch):
    client = make_client(monkeypatch)
    res = client.post(
        "/upload",
        headers={"X-API-Key": "test-app-key"},
        files={"file": ("test_doc.txt", b"not a pdf", "text/plain")},
    )
    assert res.status_code == 400


def test_upload_accepts_pdf_and_saves_it(monkeypatch):
    client = make_client(monkeypatch)
    res = client.post(
        "/upload",
        headers={"X-API-Key": "test-app-key"},
        files={"file": ("test_report.pdf", b"%PDF-1.4 fake content", "application/pdf")},
    )
    assert res.status_code == 200
    filename = res.json()["filename"]
    assert filename.endswith("_test_report.pdf")
    assert os.path.exists(os.path.join(PDF_UPLOAD_DIR, filename))


def test_upload_sanitizes_path_traversal_in_filename(monkeypatch):
    client = make_client(monkeypatch)
    res = client.post(
        "/upload",
        headers={"X-API-Key": "test-app-key"},
        files={"file": ("../../test_evil.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert res.status_code == 200
    filename = res.json()["filename"]
    saved_path = os.path.abspath(os.path.join(PDF_UPLOAD_DIR, filename))
    assert os.path.commonpath([saved_path, PDF_UPLOAD_DIR]) == PDF_UPLOAD_DIR
    assert ".." not in filename


def test_history_rejects_missing_api_key(monkeypatch):
    client = make_client(monkeypatch)
    res = client.get("/history/s1")
    assert res.status_code == 401


def test_history_filters_to_plain_text_turns(monkeypatch):
    client = make_client(monkeypatch)
    monkeypatch.setattr(
        api,
        "load_history",
        lambda session_id: [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
            {"role": "tool", "tool_call_id": "x", "content": "42"},
            {"role": "assistant", "content": "The answer is 42."},
        ],
    )
    res = client.get("/history/s1", headers={"X-API-Key": "test-app-key"})
    assert res.status_code == 200
    assert res.json() == {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "The answer is 42."},
        ]
    }


def test_delete_history_requires_auth_and_clears(monkeypatch):
    client = make_client(monkeypatch)
    calls = []
    monkeypatch.setattr(api, "delete_history", lambda session_id: calls.append(session_id))

    res = client.delete("/history/s1")
    assert res.status_code == 401
    assert calls == []

    res = client.delete("/history/s1", headers={"X-API-Key": "test-app-key"})
    assert res.status_code == 200
    assert calls == ["s1"]


def test_list_memories_requires_auth(monkeypatch):
    client = make_client(monkeypatch)
    res = client.get("/memories", params={"user_id": "u1"})
    assert res.status_code == 401


def test_list_memories_returns_scoped_facts(monkeypatch):
    client = make_client(monkeypatch)
    seen_user_ids = []

    def fake_get_all_memories(user_id):
        seen_user_ids.append(user_id)
        return {"name": "Abhi", "role": "full stack developer"}

    monkeypatch.setattr(api, "get_all_memories", fake_get_all_memories)

    res = client.get(
        "/memories",
        params={"user_id": "u1"},
        headers={"X-API-Key": "test-app-key"},
    )
    assert res.status_code == 200
    assert res.json() == {
        "memories": [
            {"key": "name", "value": "Abhi"},
            {"key": "role", "value": "full stack developer"},
        ]
    }
    assert seen_user_ids == ["u1"]


def test_delete_memory_requires_auth_and_scopes_to_user(monkeypatch):
    client = make_client(monkeypatch)
    calls = []
    monkeypatch.setattr(api, "forget_about_me", lambda user_id, key: calls.append((user_id, key)))

    res = client.delete("/memories/name", params={"user_id": "u1"})
    assert res.status_code == 401
    assert calls == []

    res = client.delete(
        "/memories/name",
        params={"user_id": "u1"},
        headers={"X-API-Key": "test-app-key"},
    )
    assert res.status_code == 200
    assert calls == [("u1", "name")]
