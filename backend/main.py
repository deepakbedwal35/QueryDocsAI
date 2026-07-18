"""
main.py

FastAPI app entrypoint. Wires up routers and CORS.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import backend.config  # noqa: F401  — loads .env before anything else

from backend.routes.ask import router as ask_router

app = FastAPI(title="Ask My Papers API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask_router)
