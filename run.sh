#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
(sleep 1 && open http://127.0.0.1:8000) &
uvicorn api:app --reload
