import os

import pytest

os.environ.setdefault("TAVILY_API_KEY", "test-key")

from tools import (  # noqa: E402
    _is_public_http_url,
    calculator,
    check_order_status,
    create_support_ticket,
    read_pdf,
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
