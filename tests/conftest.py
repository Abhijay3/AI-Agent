import os

# Set before any app module is imported, since several modules read
# required keys from os.environ at import time.
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault("APP_API_KEY", "test-app-key")
