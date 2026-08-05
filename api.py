import json
import logging
import os
import re
import sqlite3
import uuid
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.concurrency import iterate_in_threadpool

from agent_core import run_turn, stream_turn
from memory import REDIS_HOST, delete_history, load_history, save_history
from rag import ingest_docs
from schemas import (
    ChatRequest,
    ChatResponse,
    HistoryMessage,
    HistoryResponse,
    MemoriesResponse,
    MemoryItem,
    TicketItem,
    TicketsResponse,
    TicketStatusUpdate,
    UploadResponse,
)
from setup_db import DB_PATH, ensure_seeded
from tools import PDF_UPLOAD_DIR, forget_about_me, get_all_memories

SESSION_ID_PATH = Path(..., min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
USER_ID_QUERY = Query(..., min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
MEMORY_KEY_PATH = Path(..., min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
TICKET_ID_PATH = Path(..., ge=1)
TICKET_STATUS_QUERY = Query(None, pattern=r"^(open|resolved)$")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("acme_support_agent")

APP_API_KEY = os.environ["APP_API_KEY"]
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8000").split(",")

app = FastAPI(title="Acme Corp AI Customer Support Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

limiter = Limiter(key_func=get_remote_address, storage_uri=f"redis://{REDIS_HOST}:6379")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str = Security(api_key_header)) -> None:
    if key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.on_event("startup")
def seed_knowledge_base() -> None:
    ingest_docs()
    ensure_seeded()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    with open("static/index.html") as f:
        html = f.read()
    # Inject the API key server-side at request time rather than baking it
    # into the checked-in static file. Note: this is a soft gate — anyone
    # viewing page source on this served page can still read the key. It
    # stops casual/direct API abuse, not a determined attacker on a public
    # frontend. For real multi-user auth, replace with per-user login.
    html = html.replace("__APP_API_KEY__", APP_API_KEY)
    return HTMLResponse(content=html)


@app.get("/admin", response_class=HTMLResponse)
def admin() -> HTMLResponse:
    with open("static/admin.html") as f:
        html = f.read()
    # Same soft-gate tradeoff as index(): the key is embedded in this page
    # server-side rather than committed to the static file. Anyone with the
    # shared APP_API_KEY can reach this page directly by URL — there's no
    # separate staff-only credential. Fine for a single-operator deployment;
    # would need real per-user auth before handing this to multiple staff.
    html = html.replace("__APP_API_KEY__", APP_API_KEY)
    return HTMLResponse(content=html)


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("20/minute")
def chat(request: Request, body: ChatRequest) -> ChatResponse:
    messages = load_history(body.session_id)
    messages.append({"role": "user", "content": body.message})

    try:
        reply = run_turn(messages, body.user_id)
    except Exception:
        logger.exception("run_turn failed for session_id=%s", body.session_id)
        raise HTTPException(status_code=502, detail="Upstream model request failed")

    save_history(body.session_id, messages)
    return ChatResponse(reply=reply)


@app.post("/chat/stream", dependencies=[Depends(require_api_key)])
@limiter.limit("20/minute")
async def chat_stream(request: Request, body: ChatRequest) -> StreamingResponse:
    messages = load_history(body.session_id)
    messages.append({"role": "user", "content": body.message})

    def event_source():
        try:
            for event in stream_turn(messages, body.user_id):
                yield json.dumps(event) + "\n"
        except Exception:
            logger.exception("stream_turn failed for session_id=%s", body.session_id)
            yield json.dumps({"event": "error", "message": "Upstream model request failed"}) + "\n"
        finally:
            save_history(body.session_id, messages)

    return StreamingResponse(
        iterate_in_threadpool(event_source()),
        media_type="application/x-ndjson",
    )


@app.post("/upload", response_model=UploadResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("10/minute")
async def upload(request: Request, file: UploadFile) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Strip any path components and unsafe characters, then prefix with a
    # random id so two uploads with the same name can't collide or overwrite
    # each other.
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(file.filename))
    stored_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    dest_path = os.path.join(PDF_UPLOAD_DIR, stored_name)

    size = 0
    try:
        with open(dest_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File too large (max 10MB).")
                out.write(chunk)
    except HTTPException:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise

    return UploadResponse(filename=stored_name)


@app.get(
    "/history/{session_id}",
    response_model=HistoryResponse,
    dependencies=[Depends(require_api_key)],
)
def get_history(session_id: str = SESSION_ID_PATH) -> HistoryResponse:
    raw = load_history(session_id)
    # Stored history includes intermediate tool-call/tool-result messages
    # (needed by run_turn, not meaningful to show); only surface plain
    # user/assistant text turns for display.
    messages = [
        HistoryMessage(role=m["role"], content=m["content"])
        for m in raw
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) and m["content"]
    ]
    return HistoryResponse(messages=messages)


@app.delete("/history/{session_id}", dependencies=[Depends(require_api_key)])
def clear_history(session_id: str = SESSION_ID_PATH) -> dict:
    delete_history(session_id)
    return {"status": "deleted"}


@app.get("/memories", response_model=MemoriesResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
def list_memories(request: Request, user_id: str = USER_ID_QUERY) -> MemoriesResponse:
    memories = get_all_memories(user_id)
    return MemoriesResponse(memories=[MemoryItem(key=k, value=v) for k, v in memories.items()])


@app.delete("/memories/{key}", dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
def delete_memory(request: Request, key: str = MEMORY_KEY_PATH, user_id: str = USER_ID_QUERY) -> dict:
    forget_about_me(user_id, key)
    return {"status": "deleted"}


@app.get("/admin/tickets", response_model=TicketsResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
def list_tickets(request: Request, status: Optional[str] = TICKET_STATUS_QUERY) -> TicketsResponse:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        query = "SELECT id, name, email, subject, description, status, created_at FROM tickets"
        params: tuple = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY created_at DESC"
        cur.execute(query, params)
        rows = cur.fetchall()
    finally:
        conn.close()

    tickets = [
        TicketItem(id=r[0], name=r[1], email=r[2], subject=r[3], description=r[4], status=r[5], created_at=r[6])
        for r in rows
    ]
    return TicketsResponse(tickets=tickets)


@app.post("/admin/tickets/{ticket_id}/status", dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
def update_ticket_status(
    request: Request, body: TicketStatusUpdate, ticket_id: int = TICKET_ID_PATH
) -> dict:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("UPDATE tickets SET status = ? WHERE id = ?", (body.status, ticket_id))
        updated = cur.rowcount
        conn.commit()
    finally:
        conn.close()

    if not updated:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return {"status": "updated"}
