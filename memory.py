import json
import os

import redis

# Redis stores data as key -> value, in RAM, on a separate server process.
# Unlike our JSON file, it supports many keys at once (one per user/session),
# concurrent access from multiple processes, and optional expiry (TTL).
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
# Local/Docker Compose Redis has no auth or TLS, so REDIS_HOST alone is
# enough. Hosted Redis (e.g. Upstash, used for the free Render deployment)
# requires a full connection URL with TLS and a password baked in — set
# REDIS_URL for that case instead of REDIS_HOST.
REDIS_URL = os.environ.get("REDIS_URL")
redis_client = (
    redis.from_url(REDIS_URL, decode_responses=True)
    if REDIS_URL
    else redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
)


def load_history(user_id: str, session_id: str) -> list:
    raw = redis_client.get(f"chat:{user_id}:{session_id}")
    if raw is None:
        return []
    return json.loads(raw)


def save_history(user_id: str, session_id: str, messages: list) -> None:
    serializable = [
        m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m
        for m in messages
    ]
    redis_client.set(f"chat:{user_id}:{session_id}", json.dumps(serializable))


def delete_history(user_id: str, session_id: str) -> None:
    redis_client.delete(f"chat:{user_id}:{session_id}")
