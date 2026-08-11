import ipaddress
import os
import socket
import sqlite3
import threading
from datetime import datetime
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright
from pypdf import PdfReader
from tavily import TavilyClient

from setup_db import DB_PATH

PDF_UPLOAD_DIR = os.path.abspath("uploads")

# Each browse_webpage call launches a full Chromium process — on a small
# instance (e.g. Render's free 512MB tier) a handful of concurrent chats
# each triggering one at the same time is enough to OOM-kill the whole
# server (this actually happened earlier: /health itself started 502ing
# under load). Capping how many can run at once, process-wide, bounds
# worst-case memory regardless of how many chats are concurrent.
_BROWSER_SLOTS = threading.Semaphore(2)
_BROWSER_WAIT_TIMEOUT = 20  # seconds to wait for a free slot before giving up


def calculator(operation: str, a: float, b: float) -> float:
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    raise ValueError(f"Unknown operation: {operation}")


def get_current_time(timezone: str = "UTC") -> str:
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        raise ValueError(f"Unknown timezone {timezone!r} — use an IANA name like 'Asia/Kolkata' or 'UTC'.")
    return datetime.now(tz).strftime(f"%A, %B %d, %Y, %I:%M %p ({timezone})")


def get_weather(city: str) -> str:
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=10,
    ).json()

    results = geo.get("results")
    if not results:
        raise ValueError(f"Could not find location: {city}")

    lat = results[0]["latitude"]
    lon = results[0]["longitude"]

    forecast = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "current_weather": True},
        timeout=10,
    ).json()

    current = forecast["current_weather"]
    return (
        f"{city}: {current['temperature']}°C, "
        f"wind {current['windspeed']} km/h"
    )


def _sql_authorizer(action, arg1, arg2, db_name, trigger_name):
    # Belt-and-suspenders on top of the read-only connection: only allow
    # SELECT-shaped access, and only to the products table.
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if arg1 == "products" else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def run_sql_query(query: str) -> str:
    if not query.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    # Open read-only at the OS level, and restrict what the query can touch
    # via an authorizer, so a crafted query can't write, attach another DB,
    # or read tables outside 'products' even if it slips past the prefix check.
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.set_authorizer(_sql_authorizer)
    try:
        cur = conn.cursor()
        cur.execute(query)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()
    except sqlite3.DatabaseError as e:
        raise ValueError(f"Query rejected: {e}")
    finally:
        conn.close()

    if not rows:
        return "No results."

    return "\n".join(
        ", ".join(f"{col}={val}" for col, val in zip(columns, row))
        for row in rows
    )


def check_order_status(order_id: int, email: str) -> str:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT product_name, quantity, status, order_date FROM orders "
            "WHERE id = ? AND customer_email = ?",
            (order_id, email),
        )
        row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        # Deliberately vague: don't reveal whether the order exists under a
        # different email, since that would let one caller enumerate orders.
        return f"No order found with ID {order_id} for that email address."

    product_name, quantity, status, order_date = row
    return (
        f"Order #{order_id}: {quantity}x {product_name}, "
        f"status: {status}, ordered on {order_date}."
    )


def create_support_ticket(name: str, email: str, subject: str, description: str) -> str:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO tickets (name, email, subject, description) VALUES (?, ?, ?, ?)",
            (name, email, subject, description),
        )
        ticket_id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()

    return f"Support ticket #{ticket_id} created. Our team will follow up at {email}."


def remember_about_me(user_id: str, key: str, value: str) -> str:
    normalized_key = key.strip().lower().replace(" ", "_")
    if not normalized_key:
        raise ValueError("Key must not be empty.")

    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO memories (user_id, key, value, updated_at) VALUES (?, ?, ?, datetime('now')) "
            "ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (user_id, normalized_key, value),
        )
        conn.commit()
    finally:
        conn.close()

    return f"Remembered: {normalized_key} = {value}"


def forget_about_me(user_id: str, key: str) -> str:
    normalized_key = key.strip().lower().replace(" ", "_")
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM memories WHERE user_id = ? AND key = ?", (user_id, normalized_key))
        deleted = cur.rowcount
        conn.commit()
    finally:
        conn.close()

    if deleted:
        return f"Forgot '{normalized_key}'."
    return f"Nothing was remembered for '{normalized_key}'."


def get_all_memories(user_id: str) -> dict:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM memories WHERE user_id = ? ORDER BY key", (user_id,))
        rows = cur.fetchall()
    finally:
        conn.close()
    return dict(rows)


def read_pdf(path: str) -> str:
    # Resolve against the uploads dir and make sure the result doesn't
    # escape it (blocks "../../.env"-style traversal and absolute paths).
    # os.path.join would let an absolute `path` discard PDF_UPLOAD_DIR
    # outright, so strip any leading slash before joining.
    resolved = os.path.abspath(os.path.join(PDF_UPLOAD_DIR, path.lstrip("/\\")))
    if os.path.commonpath([resolved, PDF_UPLOAD_DIR]) != PDF_UPLOAD_DIR:
        raise ValueError("Path must be inside the uploads directory.")

    if not os.path.exists(resolved):
        raise ValueError(f"File not found: {path}")

    reader = PdfReader(resolved)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    if not text.strip():
        raise ValueError("No extractable text found in this PDF.")

    return text


def web_search_with_sources(query: str, max_results: int = 5) -> tuple:
    """Returns (text_for_model, sources) — sources is a list of
    {"title", "url"} dicts, separate from the text so a caller (the UI)
    can show them as proper citations instead of parsing them back out of
    the prose the model reads."""
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    try:
        response = tavily_client.search(
            query,
            search_depth="basic",  # fast tier; "advanced" roughly doubles latency
            max_results=max_results,
            include_answer="basic",  # ask Tavily for a pre-synthesized answer, not just raw snippets
            timeout=10,
        )
    except Exception as e:
        # Network error, bad API key, Tavily downtime, or our own timeout —
        # all look the same to the caller: search isn't available right now.
        raise ValueError("Web search is temporarily unavailable.") from e

    results = response.get("results", [])
    answer = response.get("answer")

    if not results and not answer:
        return "No results found.", []

    parts = []
    if answer:
        parts.append(f"Tavily's synthesized answer: {answer}")
    if results:
        parts.append(
            "Sources:\n\n" + "\n\n".join(f"{r['title']}\n{r['url']}\n{r['content']}" for r in results)
        )
    sources = [{"title": r["title"], "url": r["url"]} for r in results]
    return "\n\n".join(parts), sources


def web_search(query: str, max_results: int = 5) -> str:
    text, _sources = web_search_with_sources(query, max_results)
    return text


def _is_public_http_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        addresses = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        return False
    for family, _, _, _, sockaddr in addresses:
        ip = ipaddress.ip_address(sockaddr[0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


def open_url(url: str) -> str:
    # The backend has no way to open anything on the user's actual machine
    # (it's a cloud container, not the user's Mac) — what it CAN honestly do
    # is validate the link and hand it to the frontend, which opens it in a
    # new browser tab client-side. Same public-URL check as browse_webpage:
    # not for SSRF here (nothing is fetched server-side), but so the model
    # can't be steered into pointing the user's browser at an internal/
    # private address.
    if not _is_public_http_url(url):
        raise ValueError(f"Refusing to open non-public URL: {url}")
    return f"Opened {url} in a new browser tab."


def browse_webpage(url: str) -> str:
    # Block SSRF: refuse anything that isn't a public http(s) host, so the
    # model can't be steered into fetching internal services or metadata
    # endpoints via this server-side browser.
    if not _is_public_http_url(url):
        raise ValueError(f"Refusing to browse non-public URL: {url}")

    if not _BROWSER_SLOTS.acquire(timeout=_BROWSER_WAIT_TIMEOUT):
        raise ValueError("Too many page-reading requests right now — please try again shortly.")

    try:
        # This is the slowest tool available (a full browser launch), and the
        # main cause of the UI getting stuck on "Browsing the page..." — bound
        # both the launch and the navigation explicitly instead of relying on
        # Playwright's much longer defaults (30s launch, 30s goto).
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(timeout=8000)
                try:
                    page = browser.new_page()
                    # domcontentloaded (not the default "load") returns as soon
                    # as the DOM is ready, without waiting for every image/ad/
                    # script to finish — the visible text we want is there by then.
                    page.goto(url, timeout=8000, wait_until="domcontentloaded")
                    text = page.inner_text("body")
                finally:
                    browser.close()
        except PlaywrightError as e:
            raise ValueError("That page took too long to load or couldn't be reached.") from e
    finally:
        _BROWSER_SLOTS.release()

    return text.strip()[:3000]
