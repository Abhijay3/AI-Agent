import os

import pytest

os.environ.setdefault("TAVILY_API_KEY", "test-key")

from tools import (  # noqa: E402
    _is_public_http_url,
    calculator,
    check_order_status,
    create_support_ticket,
    forget_about_me,
    get_all_memories,
    read_pdf,
    remember_about_me,
    run_sql_query,
)


def test_calculator_basic_ops():
    assert calculator("add", 2, 3) == 5
    assert calculator("subtract", 5, 3) == 2
    assert calculator("multiply", 4, 3) == 12
    assert calculator("divide", 10, 2) == 5


def test_calculator_divide_by_zero():
    with pytest.raises(ValueError):
        calculator("divide", 1, 0)


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
