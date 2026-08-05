import secrets
import sqlite3

import bcrypt

from memory import redis_client
from setup_db import DB_PATH

SESSION_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_user(email: str, password: str) -> int:
    normalized_email = email.strip().lower()
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                (normalized_email, hash_password(password)),
            )
        except sqlite3.IntegrityError:
            raise ValueError("An account with that email already exists.")
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_user_by_email(email: str):
    """Returns (id, email, password_hash) or None."""
    normalized_email = email.strip().lower()
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, email, password_hash FROM users WHERE email = ?",
            (normalized_email,),
        )
        return cur.fetchone()
    finally:
        conn.close()


def get_user_email(user_id: str):
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute("SELECT email FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    redis_client.set(f"session:{token}", str(user_id), ex=SESSION_TTL_SECONDS)
    return token


def get_user_id_for_token(token: str):
    return redis_client.get(f"session:{token}")


def delete_session(token: str) -> None:
    redis_client.delete(f"session:{token}")
