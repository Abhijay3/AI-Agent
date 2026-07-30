import logging
import os

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agent_core import run_turn
from memory import REDIS_HOST, load_history, save_history
from rag import ingest_docs
from schemas import ChatRequest, ChatResponse
from setup_db import ensure_seeded

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


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("20/minute")
def chat(request: Request, body: ChatRequest) -> ChatResponse:
    messages = load_history(body.session_id)
    messages.append({"role": "user", "content": body.message})

    try:
        reply = run_turn(messages)
    except Exception:
        logger.exception("run_turn failed for session_id=%s", body.session_id)
        raise HTTPException(status_code=502, detail="Upstream model request failed")

    save_history(body.session_id, messages)
    return ChatResponse(reply=reply)
