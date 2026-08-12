import os
import threading
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("TAVILY_API_KEY", "test-key")

import tools  # noqa: E402
from tools import (  # noqa: E402
    _is_public_http_url,
    browse_webpage,
    calculator,
    check_order_status,
    create_support_ticket,
    forget_about_me,
    forget_all_about_me,
    get_all_memories,
    get_all_memories_detailed,
    get_current_time,
    open_url,
    read_pdf,
    remember_about_me,
    run_sql_query,
    web_search,
)


def test_calculator_basic_ops():
    assert calculator("add", 2, 3) == 5
    assert calculator("subtract", 5, 3) == 2
    assert calculator("multiply", 4, 3) == 12
    assert calculator("divide", 10, 2) == 5


def test_calculator_divide_by_zero():
    with pytest.raises(ValueError):
        calculator("divide", 1, 0)


def test_get_current_time_defaults_to_utc():
    result = get_current_time()
    assert "UTC" in result


def test_get_current_time_accepts_iana_timezone():
    result = get_current_time("Asia/Kolkata")
    assert "Asia/Kolkata" in result


def test_get_current_time_rejects_unknown_timezone():
    with pytest.raises(ValueError):
        get_current_time("Not/ARealZone")


def test_open_url_allows_public_url():
    result = open_url("https://example.com")
    assert "https://example.com" in result


def test_open_url_blocks_non_public_url():
    with pytest.raises(ValueError):
        open_url("http://127.0.0.1:8000")


def test_open_url_blocks_non_http_scheme():
    with pytest.raises(ValueError):
        open_url("javascript:alert(1)")


def test_run_sql_query_blocks_non_select():
    for bad in ["DROP TABLE products", "DELETE FROM products", "PRAGMA table_info(products)"]:
        with pytest.raises(ValueError):
            run_sql_query(bad)


def test_run_sql_query_blocks_reading_other_tables():
    # sqlite_master isn't 'products', the authorizer should deny it even
    # though the string starts with SELECT.
    with pytest.raises(ValueError):
        run_sql_query("SELECT name FROM sqlite_master")


def test_run_sql_query_allows_products_select():
    result = run_sql_query("SELECT name, price FROM products LIMIT 1")
    assert "name=" in result


def test_run_sql_query_blocks_orders_and_tickets_tables():
    # Order/ticket data must only be reachable through the dedicated,
    # identity-checked tools below, never the generic SQL tool.
    with pytest.raises(ValueError):
        run_sql_query("SELECT * FROM orders")
    with pytest.raises(ValueError):
        run_sql_query("SELECT * FROM tickets")
    with pytest.raises(ValueError):
        run_sql_query("SELECT * FROM memories")


def test_check_order_status_matches_id_and_email():
    result = check_order_status(1001, "alex@example.com")
    assert "Laptop" in result
    assert "shipped" in result


def test_check_order_status_wrong_email_is_generic():
    result = check_order_status(1001, "wrong@example.com")
    assert "No order found" in result
    assert "Laptop" not in result


def test_check_order_status_unknown_id():
    result = check_order_status(999999, "alex@example.com")
    assert "No order found" in result


def test_create_support_ticket_returns_id_and_persists():
    result = create_support_ticket(
        "Jamie", "jamie@example.com", "Broken keyboard", "Keys don't register."
    )
    assert result.startswith("Support ticket #")
    assert "jamie@example.com" in result


def test_read_pdf_blocks_path_traversal():
    with pytest.raises(ValueError):
        read_pdf("../.env")
    with pytest.raises(ValueError):
        read_pdf("/etc/passwd")


def test_read_pdf_allows_uploads_dir():
    text = read_pdf("sample.pdf")
    assert text.strip() != ""


def test_browse_webpage_blocks_private_hosts():
    for url in ["http://localhost:8000", "http://127.0.0.1", "http://169.254.169.254/latest/meta-data/"]:
        assert _is_public_http_url(url) is False


def test_browse_webpage_blocks_non_http_scheme():
    assert _is_public_http_url("file:///etc/passwd") is False


def test_browse_webpage_allows_public_host():
    assert _is_public_http_url("https://example.com") is True


TEST_USER = "test-user-1"
OTHER_USER = "test-user-2"


@pytest.fixture(autouse=True)
def cleanup_test_memories():
    yield
    for user in [TEST_USER, OTHER_USER]:
        for key in ["test_name", "test_role", "test_key"]:
            forget_about_me(user, key)


def test_remember_about_me_persists_and_normalizes_key():
    result = remember_about_me(TEST_USER, "Test Name", "Abhi")
    assert result == "Remembered: test_name = Abhi"
    assert get_all_memories(TEST_USER)["test_name"] == "Abhi"


def test_remember_about_me_overwrites_existing_key():
    remember_about_me(TEST_USER, "test_role", "backend developer")
    remember_about_me(TEST_USER, "test_role", "full stack developer")
    assert get_all_memories(TEST_USER)["test_role"] == "full stack developer"


def test_forget_about_me_removes_key():
    remember_about_me(TEST_USER, "test_key", "some value")
    assert "test_key" in get_all_memories(TEST_USER)

    result = forget_about_me(TEST_USER, "test_key")
    assert result == "Forgot 'test_key'."
    assert "test_key" not in get_all_memories(TEST_USER)


def test_forget_about_me_unknown_key_is_graceful():
    result = forget_about_me(TEST_USER, "nonexistent_key_xyz")
    assert "Nothing was remembered" in result


def test_memories_are_isolated_per_user():
    remember_about_me(TEST_USER, "test_name", "Abhi")
    remember_about_me(OTHER_USER, "test_name", "Someone Else")

    assert get_all_memories(TEST_USER)["test_name"] == "Abhi"
    assert get_all_memories(OTHER_USER)["test_name"] == "Someone Else"

    forget_about_me(TEST_USER, "test_name")
    assert "test_name" not in get_all_memories(TEST_USER)
    # forgetting for one user must not touch the other user's memory
    assert get_all_memories(OTHER_USER)["test_name"] == "Someone Else"


def test_remember_about_me_stores_category():
    remember_about_me(TEST_USER, "test_name", "Abhi", "name")
    detailed = {m["key"]: m for m in get_all_memories_detailed(TEST_USER)}
    assert detailed["test_name"]["category"] == "name"
    assert detailed["test_name"]["value"] == "Abhi"


def test_remember_about_me_defaults_to_other_category():
    remember_about_me(TEST_USER, "test_key", "some value")
    detailed = {m["key"]: m for m in get_all_memories_detailed(TEST_USER)}
    assert detailed["test_key"]["category"] == "other"


def test_remember_about_me_rejects_unknown_category():
    remember_about_me(TEST_USER, "test_key", "some value", "not_a_real_category")
    detailed = {m["key"]: m for m in get_all_memories_detailed(TEST_USER)}
    assert detailed["test_key"]["category"] == "other"


def test_remember_about_me_update_overwrites_category():
    remember_about_me(TEST_USER, "test_key", "first value", "project")
    remember_about_me(TEST_USER, "test_key", "second value", "preference")
    detailed = {m["key"]: m for m in get_all_memories_detailed(TEST_USER)}
    assert detailed["test_key"]["value"] == "second value"
    assert detailed["test_key"]["category"] == "preference"


def test_forget_all_about_me_clears_only_that_user():
    try:
        remember_about_me(TEST_USER, "test_name", "Abhi")
        remember_about_me(TEST_USER, "test_role", "developer")
        remember_about_me(OTHER_USER, "test_name", "Someone Else")

        deleted = forget_all_about_me(TEST_USER)

        assert deleted == 2
        assert get_all_memories(TEST_USER) == {}
        # other user's memories must survive
        assert get_all_memories(OTHER_USER)["test_name"] == "Someone Else"
    finally:
        forget_all_about_me(OTHER_USER)


class FakeTavilyClient:
    def __init__(self, api_key):
        pass

    def search(self, query, **kwargs):
        return {
            "answer": "42",
            "results": [{"title": "The Answer", "url": "https://example.com/x", "content": "It's 42."}],
        }


class FailingTavilyClient:
    def __init__(self, api_key):
        pass

    def search(self, query, **kwargs):
        raise RuntimeError("network down")


def test_web_search_includes_synthesized_answer_and_sources(monkeypatch):
    monkeypatch.setattr(tools, "TavilyClient", FakeTavilyClient)

    result = web_search("what is the answer")

    assert "Tavily's synthesized answer: 42" in result
    assert "Sources:" in result
    assert "https://example.com/x" in result


def test_web_search_wraps_tavily_failures_as_value_error(monkeypatch):
    # A Tavily outage, timeout, or bad API key must never crash the whole
    # turn — it should surface as the same kind of friendly ValueError every
    # other tool uses, so agent_core can hand it back to the model gracefully.
    monkeypatch.setattr(tools, "TavilyClient", FailingTavilyClient)

    with pytest.raises(ValueError):
        web_search("what is the answer")


class FakeChromiumLaunchFails:
    def launch(self, timeout=None):
        from playwright.sync_api import Error as PlaywrightError

        raise PlaywrightError("Timeout exceeded while launching browser")


class FakePlaywrightContext:
    def __enter__(self):
        from types import SimpleNamespace

        return SimpleNamespace(chromium=FakeChromiumLaunchFails())

    def __exit__(self, *args):
        return False


def test_browse_webpage_wraps_playwright_failures_as_value_error(monkeypatch):
    monkeypatch.setattr(tools, "sync_playwright", lambda: FakePlaywrightContext())

    with pytest.raises(ValueError):
        browse_webpage("https://example.com")


def test_browse_webpage_caps_concurrent_browser_launches(monkeypatch):
    # Each browse_webpage launches a full Chromium process — on a small
    # instance that's exactly what OOM-crashed the whole server once
    # already under real concurrent load. The module-wide semaphore is
    # supposed to bound how many launches run at once regardless of how
    # many chats are concurrent; prove it actually does, not just that the
    # code exists.
    lock = threading.Lock()
    state = {"current": 0, "max_seen": 0}

    class FakePage:
        def goto(self, url, timeout=None, wait_until=None):
            pass

        def inner_text(self, selector):
            return "page text"

    class FakeBrowser:
        def new_page(self):
            return FakePage()

        def close(self):
            pass

    class FakeChromium:
        def launch(self, timeout=None):
            with lock:
                state["current"] += 1
                state["max_seen"] = max(state["max_seen"], state["current"])
            time.sleep(0.2)
            with lock:
                state["current"] -= 1
            return FakeBrowser()

    class FakePlaywrightContextSlow:
        def __enter__(self):
            return SimpleNamespace(chromium=FakeChromium())

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(tools, "sync_playwright", lambda: FakePlaywrightContextSlow())
    # Real module-wide semaphore, default cap of 2 — not mocked, so this
    # test exercises the actual production limit.
    monkeypatch.setattr(tools, "_BROWSER_SLOTS", threading.Semaphore(2))

    threads = [threading.Thread(target=browse_webpage, args=("https://example.com",)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state["max_seen"] <= 2
