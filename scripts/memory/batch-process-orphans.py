#!/usr/bin/env python3
"""Batch process ORPHAN_FILE items in groundskeeper queue.

Stamps up to 25 pending orphans that already sit in an organised directory.
Until backlog #899 this wrote the queue with a bare ``json.dump`` and appended
no log row at all, which is why every one of the ~31k rows in
``groundskeeper-log.jsonl`` carried a single reason while 25 queue items
carried this script's reason: the second writer was invisible to the log. It now
shares the survey's guarded writer and logs each item it stamps.

The queue path is overridable with GROUNDKEEPER_QUEUE so the guard can be
tested without touching the live queue.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.groundskeeper.queue_io import (  # noqa: E402
    PROCESS_LOG_NAME,
    append_process_log,
    write_queue_atomic,
)
from app.paths import PIPELINE_DIR  # noqa: E402

QUEUE_PATH = Path(
    os.environ.get(
        "GROUNDKEEPER_QUEUE", str(PIPELINE_DIR / "groundskeeper-queue.json")
    )
)
LOG_PATH = QUEUE_PATH.parent / PROCESS_LOG_NAME

REASON = "legitimate organized project file in folder hierarchy"
BATCH_CAP = 25
ORGANIZED_DIRS = [
    "projects/", "agents/", "knowledge/", "work/", "skills/",
    "templates/", "backlog/", "memory/", "personal/", "lloyd/",
]


def main() -> int:
    # Read queue
    with open(QUEUE_PATH, "r") as f:
        queue = json.load(f)

    # Process up to 25 pending ORPHAN_FILE items, one log row each.
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for item in queue["items"]:
        if len(rows) >= BATCH_CAP:
            break
        if item.get("status") == "pending" and item.get("type") == "ORPHAN_FILE":
            # Check if it's in an organized directory
            source = item.get("source_file", "")
            if any(source.startswith(d) for d in ORGANIZED_DIRS):
                item["status"] = "skipped"
                item["reason"] = REASON
                item["processed_at"] = timestamp
                rows.append(
                    {
                        "item_id": item.get("id"),
                        "type": item["type"],
                        "status": "skipped",
                        "reason": item["reason"],
                        "processed_at": timestamp,
                    }
                )
    processed = len(rows)

    # Update queue metadata
    queue["items_processed"] = processed

    # Guarded write first: the log may only record stamps the queue actually
    # carries, so rows land after the rename that made them durable.
    write_queue_atomic(QUEUE_PATH, queue, stamped=processed)
    append_process_log(LOG_PATH, rows)

    print(f"Processed {processed} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
