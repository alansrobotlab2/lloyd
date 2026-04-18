"""Session metadata persistence and active-stream registry.

Session data lives as one JSON file per session under `SESSIONS_DIR`. The
`_active_streams` dict tracks live SSE streams so the cancel endpoint can
signal the streaming generator to stop between SDK messages — importers
must access the same dict object, not a copy.
"""

import asyncio
import json
from datetime import datetime

from app.paths import SESSIONS_DIR

_active_streams: dict[str, asyncio.Event] = {}


def _save_session_meta(session_id: str, model: str, preview: str = ""):
    """Save session metadata to JSON file."""
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    now = datetime.now().isoformat()
    if meta_path.exists():
        data = json.loads(meta_path.read_text())
        data["last_active"] = now
        if preview:
            data["preview"] = preview[:60]
        data["message_count"] = data.get("message_count", 0) + 1
    else:
        data = {
            "session_id": session_id,
            "model": model,
            "created_at": now,
            "last_active": now,
            "preview": preview[:60],
            "message_count": 1,
            "messages": [],
            "platform": "mission-control",
        }
    meta_path.write_text(json.dumps(data, indent=2))


def _append_messages(session_id: str, new_messages: list[dict]):
    """Append messages to session metadata file."""
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    if not meta_path.exists():
        return
    data = json.loads(meta_path.read_text())
    msgs = data.get("messages", [])
    msgs.extend(new_messages)
    data["messages"] = msgs
    data["last_active"] = datetime.now().isoformat()
    meta_path.write_text(json.dumps(data, indent=2))
