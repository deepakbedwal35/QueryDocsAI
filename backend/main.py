"""
main.py

FastAPI app entrypoint. Wires up routers and CORS.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.ask import router as ask_router

app = FastAPI(title="Ask My Papers API")

# Dev-friendly CORS. Tighten this to your actual frontend origin(s)
# before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask_router)
