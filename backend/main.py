"""
main.py

FastAPI app entrypoint. Wires up routers, CORS, and DB init.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Must be imported first: this triggers load_dotenv(), so GROQ_API_KEY
# etc. are in os.environ before groq_client or anything else reads them.
import backend.config  # noqa: F401
from backend.db.session import init_db
from backend.routes.ask import router as ask_router
from backend.routes.chats import router as chats_router

app = FastAPI(title="Ask My Papers API")

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
