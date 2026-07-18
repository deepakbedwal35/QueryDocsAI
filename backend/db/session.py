"""
db/session.py

SQLAlchemy engine + session setup. SQLite for now — swapping to
Postgres later is just changing DATABASE_URL, since everything else
uses the SQLAlchemy ORM rather than raw SQL.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./ask_my_papers.db")

# check_same_thread=False is only needed for SQLite (FastAPI handles
# requests across threads); harmless no-op arg for other engines.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def get_db():
    """FastAPI dependency: yields a DB session, closes it after the request."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Call once at startup (fine for SQLite/dev;
    swap for Alembic migrations if this grows into production)."""
    from backend.db import models  # noqa: F401 (ensure models are registered)

    Base.metadata.create_all(bind=engine)
