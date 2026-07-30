import sqlite3

conn = sqlite3.connect("store.db")
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS products")
cur.execute("""
    CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        price REAL NOT NULL,
        stock INTEGER NOT NULL
    )
""")

cur.executemany(
    "INSERT INTO products (name, price, stock) VALUES (?, ?, ?)",
    [
        ("Laptop", 799.99, 12),
        ("Wireless Mouse", 19.99, 150),
        ("Mechanical Keyboard", 89.99, 40),
        ("Monitor 27in", 249.99, 25),
        ("USB-C Hub", 34.99, 60),
    ],
)

cur.execute("DROP TABLE IF EXISTS orders")
cur.execute("""
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        customer_email TEXT NOT NULL,
        product_name TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        status TEXT NOT NULL,
        order_date TEXT NOT NULL
    )
""")

cur.executemany(
    "INSERT INTO orders (id, customer_email, product_name, quantity, status, order_date) VALUES (?, ?, ?, ?, ?, ?)",
    [
        (1001, "alex@example.com", "Laptop", 1, "shipped", "2026-07-22"),
        (1002, "alex@example.com", "Wireless Mouse", 2, "delivered", "2026-07-10"),
        (1003, "priya@example.com", "Mechanical Keyboard", 1, "processing", "2026-07-28"),
        (1004, "jordan@example.com", "Monitor 27in", 1, "cancelled", "2026-07-15"),
    ],
)

cur.execute("DROP TABLE IF EXISTS tickets")
cur.execute("""
    CREATE TABLE tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        subject TEXT NOT NULL,
        description TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'open',
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
""")

conn.commit()
conn.close()
print("store.db created with products, orders, and tickets tables")
