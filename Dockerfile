FROM python:3.11-slim

WORKDIR /app

# Keep browser binaries and the embedding model cache under /app so they
# end up owned by appuser below, instead of /root's home (which a non-root
# user can't read, and Path.home()-based caches like chromadb's default to).
ENV PLAYWRIGHT_BROWSERS_PATH=/app/.playwright-browsers
ENV HOME=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install --with-deps chromium

# Pre-download the embedding model into the image so container startup
# doesn't have to fetch it over the network on first request (was ~75s).
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warm'])"

COPY agent_core.py api.py auth.py memory.py rag.py schemas.py setup_db.py tools.py ./
COPY docs/ ./docs/
COPY static/ ./static/
COPY uploads/ ./uploads/

# store.db is deliberately NOT copied in: it's seeded at startup (see
# setup_db.ensure_seeded, called from api.py) into /app/data, which is
# volume-mounted in docker-compose.yml so orders/tickets survive rebuilds.
RUN mkdir -p /app/data \
    && useradd --no-create-home --home-dir /app --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
