"""
config.py

Loads secrets/settings from environment (.env). Import this early
(e.g. from main.py) so GROQ_API_KEY etc. are available before any
client is constructed.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
