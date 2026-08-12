import os

# Set before any app module is imported, since several modules read
# required keys from os.environ at import time.
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault("APP_API_KEY", "test-app-key")

# Several tests (test_tools.py) hit the DB directly rather than going
# through the FastAPI app, so they never trigger api.py's startup-event
# migration. Run it here so the schema is current before anything queries
# it — otherwise a checked-out store.db predating a later column addition
# fails with "no such column" regardless of what's committed.
from setup_db import ensure_seeded  # noqa: E402

ensure_seeded()
