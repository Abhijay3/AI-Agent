import glob
import json
import os
import sqlite3

import pytest
from fastapi.testclient import TestClient

import api
from setup_db import DB_PATH
from tools import PDF_UPLOAD_DIR

TEST_USER_ID = "test-user-id"
TEST_EMAILS = [
    "signup-test@example.com",
    "duplicate-test@example.com",
    "login-test@example.com",
    "me-test@example.com",
    "logout-test@example.com",
]


@pytest.fixture(autouse=True)
def cleanup_test_uploads():
    yield
    for path in glob.glob(os.path.join(PDF_UPLOAD_DIR, "*_test_*.pdf")):
        os.remove(path)


@pytest.fixture(autouse=True)
def cleanup_test_users():
    yield
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.executemany("DELETE FROM users WHERE email = ?", [(e,) for e in TEST_EMAILS])
        conn.commit()
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    yield
    api.app.dependency_overrides.clear()


def _insert_ticket(subject, status="open", name="Test User", email="test@example.com"):
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO tickets (name, email, subject, description, status) VALUES (?, ?, ?, ?, ?)",
            (name, email, subject, "Test description.", status),
        )
        ticket_id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()
    return ticket_id


def make_client(monkeypatch, authenticated=True):
    """Stubs the model/history layer. By default also bypasses real session
    token parsing (dependency_overrides) so most tests don't need to run a
    real signup/login round trip just to reach an authenticated route —
    the auth mechanism itself is covered separately by the /auth/* tests
    below, and by tools.py's own per-user isolation tests."""
    # The rate limiter's storage lives in real Redis, shared across every
    # test in the run — without resetting it, tests that hit a tightly
    # limited route (e.g. 5/minute on signup) multiple times start failing
    # each other depending on run order.
    api.limiter.reset()
    monkeypatch.setattr(api, "run_turn", lambda messages, user_id: "mocked reply")
    monkeypatch.setattr(api, "load_history", lambda user_id, session_id: [])
    monkeypatch.setattr(api, "save_history", lambda user_id, session_id, messages: None)
    if authenticated:
        api.app.dependency_overrides[api.require_user] = lambda: TEST_USER_ID
    return TestClient(api.app)


def test_health_needs_no_auth(monkeypatch):
    client = make_client(monkeypatch)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_index_serves_html_with_no_key_placeholder(monkeypatch):
    client = make_client(monkeypatch)
    res = client.get("/")
    assert res.status_code == 200
    assert "__APP_API_KEY__" not in res.text


# ---------- /auth/* : real integration tests, no dependency override ----------


def test_signup_creates_user_and_returns_token(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    res = client.post("/auth/signup", json={"email": "signup-test@example.com", "password": "hunter22"})
    assert res.status_code == 200
    data = res.json()
    assert data["email"] == "signup-test@example.com"
    assert data["token"]
    assert data["user_id"]


def test_signup_rejects_duplicate_email(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    client.post("/auth/signup", json={"email": "duplicate-test@example.com", "password": "hunter22"})
    res = client.post("/auth/signup", json={"email": "duplicate-test@example.com", "password": "different1"})
    assert res.status_code == 409


def test_signup_rejects_short_password(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    res = client.post("/auth/signup", json={"email": "signup-test@example.com", "password": "short"})
    assert res.status_code == 422


def test_signup_rejects_invalid_email(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    res = client.post("/auth/signup", json={"email": "not-an-email", "password": "hunter22"})
    assert res.status_code == 422


def test_login_succeeds_with_correct_password(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    client.post("/auth/signup", json={"email": "login-test@example.com", "password": "correct-horse"})
    res = client.post("/auth/login", json={"email": "login-test@example.com", "password": "correct-horse"})
    assert res.status_code == 200
    assert res.json()["token"]


def test_login_rejects_wrong_password(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    client.post("/auth/signup", json={"email": "login-test@example.com", "password": "correct-horse"})
    res = client.post("/auth/login", json={"email": "login-test@example.com", "password": "wrong-password"})
    assert res.status_code == 401


def test_login_rejects_unknown_email(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    res = client.post("/auth/login", json={"email": "nobody-here@example.com", "password": "whatever1"})
    assert res.status_code == 401


def test_me_requires_valid_token(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    res = client.get("/auth/me")
    assert res.status_code == 401

    res = client.get("/auth/me", headers={"Authorization": "Bearer garbage-token"})
    assert res.status_code == 401

    signup_res = client.post("/auth/signup", json={"email": "me-test@example.com", "password": "hunter22"})
    token = signup_res.json()["token"]
    res = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert res.json()["email"] == "me-test@example.com"


def test_logout_invalidates_token(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    signup_res = client.post("/auth/signup", json={"email": "logout-test@example.com", "password": "hunter22"})
    token = signup_res.json()["token"]

    res = client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200

    res = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 401


# ---------- /chat, /chat/stream ----------


def test_chat_rejects_missing_auth(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    res = client.post("/chat", json={"session_id": "s1", "message": "hi"})
    assert res.status_code == 401


def test_chat_rejects_invalid_token(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    res = client.post(
        "/chat",
        json={"session_id": "s1", "message": "hi"},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert res.status_code == 401


def test_chat_accepts_authenticated_request(monkeypatch):
    client = make_client(monkeypatch)
    res = client.post("/chat", json={"session_id": "s1", "message": "hi"})
    assert res.status_code == 200
    assert res.json() == {"reply": "mocked reply"}


def test_chat_stream_rejects_missing_auth(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    res = client.post("/chat/stream", json={"session_id": "s1", "message": "hi"})
    assert res.status_code == 401


def test_chat_stream_yields_ndjson_events(monkeypatch):
    client = make_client(monkeypatch)

    def fake_stream_turn(messages, user_id):
        yield {"event": "tool_call", "tool": "calculator"}
        yield {"event": "token", "text": "Hello"}
        yield {"event": "token", "text": " world"}

    monkeypatch.setattr(api, "stream_turn", fake_stream_turn)

    res = client.post("/chat/stream", json={"session_id": "s1", "message": "hi"})
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

    res = client.post("/chat/stream", json={"session_id": "s1", "message": "hi"})
    assert res.status_code == 200
    events = [json.loads(line) for line in res.text.strip().split("\n") if line.strip()]
    assert events[0] == {"event": "token", "text": "partial"}
    assert events[-1]["event"] == "error"


# ---------- /upload ----------


def test_upload_rejects_missing_auth(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    res = client.post("/upload", files={"file": ("test_doc.pdf", b"%PDF-1.4 fake", "application/pdf")})
    assert res.status_code == 401


def test_upload_rejects_non_pdf(monkeypatch):
    client = make_client(monkeypatch)
    res = client.post(
        "/upload",
        files={"file": ("test_doc.txt", b"not a pdf", "text/plain")},
    )
    assert res.status_code == 400


def test_upload_accepts_pdf_and_saves_it(monkeypatch):
    client = make_client(monkeypatch)
    res = client.post(
        "/upload",
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
        files={"file": ("../../test_evil.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert res.status_code == 200
    filename = res.json()["filename"]
    saved_path = os.path.abspath(os.path.join(PDF_UPLOAD_DIR, filename))
    assert os.path.commonpath([saved_path, PDF_UPLOAD_DIR]) == PDF_UPLOAD_DIR
    assert ".." not in filename


# ---------- /history ----------


def test_history_rejects_missing_auth(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    res = client.get("/history/s1")
    assert res.status_code == 401


def test_history_filters_to_plain_text_turns(monkeypatch):
    client = make_client(monkeypatch)
    monkeypatch.setattr(
        api,
        "load_history",
        lambda user_id, session_id: [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
            {"role": "tool", "tool_call_id": "x", "content": "42"},
            {"role": "assistant", "content": "The answer is 42."},
        ],
    )
    res = client.get("/history/s1")
    assert res.status_code == 200
    assert res.json() == {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "The answer is 42."},
        ]
    }


def test_delete_history_requires_auth_and_clears(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    calls = []
    monkeypatch.setattr(api, "delete_history", lambda user_id, session_id: calls.append((user_id, session_id)))

    res = client.delete("/history/s1")
    assert res.status_code == 401
    assert calls == []

    api.app.dependency_overrides[api.require_user] = lambda: TEST_USER_ID
    res = client.delete("/history/s1")
    assert res.status_code == 200
    assert calls == [(TEST_USER_ID, "s1")]


# ---------- /memories ----------


def test_list_memories_requires_auth(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    res = client.get("/memories")
    assert res.status_code == 401


def test_list_memories_returns_scoped_facts(monkeypatch):
    client = make_client(monkeypatch)
    seen_user_ids = []

    def fake_get_all_memories(user_id):
        seen_user_ids.append(user_id)
        return {"name": "Abhi", "role": "full stack developer"}

    monkeypatch.setattr(api, "get_all_memories", fake_get_all_memories)

    res = client.get("/memories")
    assert res.status_code == 200
    assert res.json() == {
        "memories": [
            {"key": "name", "value": "Abhi"},
            {"key": "role", "value": "full stack developer"},
        ]
    }
    assert seen_user_ids == [TEST_USER_ID]


def test_delete_memory_requires_auth_and_scopes_to_user(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    calls = []
    monkeypatch.setattr(api, "forget_about_me", lambda user_id, key: calls.append((user_id, key)))

    res = client.delete("/memories/name")
    assert res.status_code == 401
    assert calls == []

    api.app.dependency_overrides[api.require_user] = lambda: TEST_USER_ID
    res = client.delete("/memories/name")
    assert res.status_code == 200
    assert calls == [(TEST_USER_ID, "name")]


# ---------- /admin (separate operator key, unchanged) ----------


def test_admin_page_injects_api_key(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    res = client.get("/admin")
    assert res.status_code == 200
    assert "test-app-key" in res.text
    assert "__APP_API_KEY__" not in res.text


def test_list_tickets_requires_auth(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    res = client.get("/admin/tickets")
    assert res.status_code == 401


def test_list_tickets_returns_and_filters_by_status(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    open_id = _insert_ticket("Open ticket", status="open")
    resolved_id = _insert_ticket("Resolved ticket", status="resolved")

    res = client.get("/admin/tickets", headers={"X-API-Key": "test-app-key"})
    assert res.status_code == 200
    ids = [t["id"] for t in res.json()["tickets"]]
    assert open_id in ids
    assert resolved_id in ids

    res = client.get(
        "/admin/tickets", params={"status": "open"}, headers={"X-API-Key": "test-app-key"}
    )
    ids = [t["id"] for t in res.json()["tickets"]]
    assert open_id in ids
    assert resolved_id not in ids


def test_update_ticket_status_requires_auth_and_updates(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    ticket_id = _insert_ticket("To resolve", status="open")

    res = client.post(f"/admin/tickets/{ticket_id}/status", json={"status": "resolved"})
    assert res.status_code == 401

    res = client.post(
        f"/admin/tickets/{ticket_id}/status",
        json={"status": "resolved"},
        headers={"X-API-Key": "test-app-key"},
    )
    assert res.status_code == 200

    res = client.get(
        "/admin/tickets", params={"status": "resolved"}, headers={"X-API-Key": "test-app-key"}
    )
    ids = [t["id"] for t in res.json()["tickets"]]
    assert ticket_id in ids


def test_update_ticket_status_unknown_ticket_is_404(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    res = client.post(
        "/admin/tickets/999999/status",
        json={"status": "resolved"},
        headers={"X-API-Key": "test-app-key"},
    )
    assert res.status_code == 404


def test_update_ticket_status_rejects_invalid_status(monkeypatch):
    client = make_client(monkeypatch, authenticated=False)
    ticket_id = _insert_ticket("Invalid status test", status="open")

    res = client.post(
        f"/admin/tickets/{ticket_id}/status",
        json={"status": "not-a-real-status"},
        headers={"X-API-Key": "test-app-key"},
    )
    assert res.status_code == 422
