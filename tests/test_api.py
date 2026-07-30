from fastapi.testclient import TestClient

import api


def make_client(monkeypatch):
    monkeypatch.setattr(api, "run_turn", lambda messages: "mocked reply")
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
    res = client.post("/chat", json={"session_id": "s1", "message": "hi"})
    assert res.status_code == 401


def test_chat_rejects_wrong_api_key(monkeypatch):
    client = make_client(monkeypatch)
    res = client.post(
        "/chat",
        json={"session_id": "s1", "message": "hi"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert res.status_code == 401


def test_chat_accepts_correct_api_key(monkeypatch):
    client = make_client(monkeypatch)
    res = client.post(
        "/chat",
        json={"session_id": "s1", "message": "hi"},
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
