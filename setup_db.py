import os
import sqlite3

DB_PATH = os.environ.get("DB_PATH", "store.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    price REAL NOT NULL,
    stock INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY,
    customer_email TEXT NOT NULL,
    product_name TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    status TEXT NOT NULL,
    order_date TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    subject TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS memories (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

PRODUCTS = [
    ("Laptop", 799.99, 12),
    ("Wireless Mouse", 19.99, 150),
    ("Mechanical Keyboard", 89.99, 40),
    ("Monitor 27in", 249.99, 25),
    ("USB-C Hub", 34.99, 60),
]

ORDERS = [
    (1001, "alex@example.com", "Laptop", 1, "shipped", "2026-07-22"),
    (1002, "alex@example.com", "Wireless Mouse", 2, "delivered", "2026-07-10"),
    (1003, "priya@example.com", "Mechanical Keyboard", 1, "processing", "2026-07-28"),
    (1004, "jordan@example.com", "Monitor 27in", 1, "cancelled", "2026-07-15"),
]


def ensure_seeded(path: str = DB_PATH) -> None:
    """Create tables if missing and seed sample rows only into empty
    tables. Safe to call on every app startup — never touches existing
    data, so it won't wipe real orders/tickets on a redeploy."""
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.executescript(SCHEMA)

        cur.execute("SELECT COUNT(*) FROM products")
        if cur.fetchone()[0] == 0:
            cur.executemany(
                "INSERT INTO products (name, price, stock) VALUES (?, ?, ?)",
                PRODUCTS,
            )

        cur.execute("SELECT COUNT(*) FROM orders")
        if cur.fetchone()[0] == 0:
            cur.executemany(
                "INSERT INTO orders "
                "(id, customer_email, product_name, quantity, status, order_date) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ORDERS,
            )
        conn.commit()
    finally:
        conn.close()


def reset_db(path: str = DB_PATH) -> None:
    """Drop everything and reseed from scratch. Destructive — for local
    development resets only; never called automatically at startup."""
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS products")
        cur.execute("DROP TABLE IF EXISTS orders")
        cur.execute("DROP TABLE IF EXISTS tickets")
        cur.execute("DROP TABLE IF EXISTS memories")
        conn.commit()
    finally:
        conn.close()
    ensure_seeded(path)


if __name__ == "__main__":
    reset_db()
    print(f"{DB_PATH} reset with fresh products, orders, and tickets tables")
