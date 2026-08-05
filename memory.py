import json
import os

import redis

# Redis stores data as key -> value, in RAM, on a separate server process.
# Unlike our JSON file, it supports many keys at once (one per user/session),
# concurrent access from multiple processes, and optional expiry (TTL).
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)


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
