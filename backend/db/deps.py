"""
db/deps.py

FastAPI dependency for device identification. No login — the
frontend generates a UUID on first load (stored in localStorage) and
sends it as X-Device-Id on every request. This dependency looks up
that device, creating it on first sight.
"""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from backend.db.models import Device
from backend.db.session import get_db, utcnow


def get_current_device(
    x_device_id: str | None = Header(default=None, alias="X-Device-Id"),
    db: Session = Depends(get_db),
) -> Device:
    if not x_device_id:
        raise HTTPException(
            status_code=400,
            detail="Missing X-Device-Id header. The frontend must generate and "
            "send a device id on every request.",
        )

    device = db.get(Device, x_device_id)
    if device is None:
        device = Device(device_id=x_device_id)
        db.add(device)
        db.commit()
        db.refresh(device)
    else:
        device.last_seen_at = utcnow()
        db.commit()

    return device
