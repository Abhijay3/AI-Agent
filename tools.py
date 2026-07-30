import ipaddress
import os
import socket
import sqlite3
from urllib.parse import urlparse

import requests
from playwright.sync_api import sync_playwright
from pypdf import PdfReader
from tavily import TavilyClient

PDF_UPLOAD_DIR = os.path.abspath("uploads")


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
    conn = sqlite3.connect("file:store.db?mode=ro", uri=True)
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


def web_search(query: str, max_results: int = 3) -> str:
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = tavily_client.search(query, max_results=max_results)
    results = response.get("results", [])

    if not results:
        return "No results found."

    return "\n\n".join(
        f"{r['title']}\n{r['url']}\n{r['content']}" for r in results
    )


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


def browse_webpage(url: str) -> str:
    # Block SSRF: refuse anything that isn't a public http(s) host, so the
    # model can't be steered into fetching internal services or metadata
    # endpoints via this server-side browser.
    if not _is_public_http_url(url):
        raise ValueError(f"Refusing to browse non-public URL: {url}")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(url, timeout=15000)
            text = page.inner_text("body")
        finally:
            browser.close()

    return text.strip()[:3000]
