"""
routes/chats.py

CRUD for chats, scoped to the requesting device (via X-Device-Id).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.db.deps import get_current_device
from backend.db.models import Chat, Device
from backend.db.session import get_db
from backend.models.schemas import ChatCreate, ChatRename, ChatSummary, MessageOut

router = APIRouter(prefix="/chats", tags=["chats"])


def _get_owned_chat(chat_id: str, device: Device, db: Session) -> Chat:
    chat = db.get(Chat, chat_id)
    if chat is None or chat.device_id != device.device_id:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.post("", response_model=ChatSummary)
def create_chat(
    body: ChatCreate,
    device: Device = Depends(get_current_device),
    db: Session = Depends(get_db),
) -> Chat:
    chat = Chat(device_id=device.device_id, title=body.title)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


@router.get("", response_model=list[ChatSummary])
def list_chats(
    device: Device = Depends(get_current_device),
    db: Session = Depends(get_db),
) -> list[Chat]:
    return (
        db.query(Chat)
        .filter(Chat.device_id == device.device_id)
        .order_by(Chat.updated_at.desc())
        .all()
    )


@router.patch("/{chat_id}", response_model=ChatSummary)
def rename_chat(
    chat_id: str,
    body: ChatRename,
    device: Device = Depends(get_current_device),
    db: Session = Depends(get_db),
) -> Chat:
    chat = _get_owned_chat(chat_id, device, db)
    chat.title = body.title
    db.commit()
    db.refresh(chat)
    return chat


@router.delete("/{chat_id}", status_code=204)
def delete_chat(
    chat_id: str,
    device: Device = Depends(get_current_device),
    db: Session = Depends(get_db),
) -> None:
    chat = _get_owned_chat(chat_id, device, db)
    db.delete(chat)
    db.commit()


@router.get("/{chat_id}/messages", response_model=list[MessageOut])
def get_chat_messages(
    chat_id: str,
    device: Device = Depends(get_current_device),
    db: Session = Depends(get_db),
) -> list:
    chat = _get_owned_chat(chat_id, device, db)
    return chat.messages
