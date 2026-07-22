"""
main.py

FastAPI app entrypoint. Wires up routers, CORS, and DB init.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# import config  # noqa: F401  — loads .env before anything else

from backend.routes.ask import router as ask_router
from backend.routes.chats import router as chats_router
from backend.routes.documents import router as documents_router

app = FastAPI(title="QueryDocsAI")

# Dev-friendly CORS. Tighten this to your actual frontend origin(s)
# before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


app.include_router(ask_router)
app.include_router(chats_router)
app.include_router(documents_router)
