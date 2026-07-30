FROM python:3.11-slim

WORKDIR /app

# Keep browser binaries under /app so they end up owned by appuser below,
# instead of the default /root/.cache which a non-root user can't read.
ENV PLAYWRIGHT_BROWSERS_PATH=/app/.playwright-browsers

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install --with-deps chromium

# Pre-download the embedding model into the image so container startup
# doesn't have to fetch it over the network on first request (was ~75s).
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warm'])"

COPY agent_core.py api.py memory.py rag.py schemas.py tools.py ./
COPY docs/ ./docs/
COPY static/ ./static/
COPY uploads/ ./uploads/
COPY store.db ./store.db

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
