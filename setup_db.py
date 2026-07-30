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

conn.commit()
conn.close()
print("store.db created with sample products table")
