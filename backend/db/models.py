"""
db/models.py

SQLAlchemy ORM models.

Devices -> Chats -> Messages, with ON DELETE CASCADE so deleting a
chat removes its messages, and deleting a device removes its chats
(and transitively their messages).

Chats -> Documents, with ON DELETE CASCADE so deleting a chat
removes its uploaded documents (and their Qdrant points via
application-level cascade).

`citations` is stored as a JSON column on Message rather than a
separate table, since citations are always read/written as a unit
with their parent message and never queried independently.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.session import Base, utcnow


def _uuid() -> str:
    return str(uuid.uuid4())


class Device(Base):
    __tablename__ = "devices"

    device_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    chats: Mapped[list["Chat"]] = relationship(
        back_populates="device", cascade="all, delete-orphan"
    )


class Chat(Base):
    __tablename__ = "chats"

    chat_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    device_id: Mapped[str] = mapped_column(
        String, ForeignKey("devices.device_id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String, default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    device: Mapped["Device"] = relationship(back_populates="chats")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="chat",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    documents: Mapped[list["Document"]] = relationship(
        back_populates="chat",
        cascade="all, delete-orphan",
        order_by="Document.uploaded_at",
    )


class Message(Base):
    __tablename__ = "messages"

    message_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    chat_id: Mapped[str] = mapped_column(
        String, ForeignKey("chats.chat_id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    answer_found: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    chat: Mapped["Chat"] = relationship(back_populates="messages")


class Document(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    chat_id: Mapped[str] = mapped_column(
        String, ForeignKey("chats.chat_id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String)
    page_count: Mapped[int | None] = mapped_column(nullable=True)
    chunk_count: Mapped[int] = mapped_column(default=0)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    chat: Mapped["Chat"] = relationship(back_populates="documents")
