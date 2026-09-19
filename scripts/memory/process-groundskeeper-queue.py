#!/usr/bin/env python3
"""Process groundskeeper queue - mark pending ORPHAN_FILE items as skipped (survey bug).

One-off repair for the period when the survey had no hub-page awareness and
every organised project file read as an orphan. The repair outlived the bug and
its caller is still unidentified (backlog #899), so it is not deleted; it now
reaches the queue only through the shared guarded writer and logs each item it
stamps with that item's own reason.

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

QUEUE_PATH = Path(
    os.environ.get(
        "GROUNDKEEPER_QUEUE", "/home/alansrobotlab/lloyd/_pipeline/groundskeeper-queue.json"
    )
)
LOG_PATH = QUEUE_PATH.parent / PROCESS_LOG_NAME

REASON = "hub-page-linked-survey-bug"


def main() -> int:
    # Read queue
    with open(QUEUE_PATH, "r") as f:
        queue = json.load(f)

    # Stamp every pending ORPHAN_FILE and keep one log row per stamped item.
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for item in queue["items"]:
        if item.get("status") == "pending" and item.get("type") == "ORPHAN_FILE":
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
    processed_count = len(rows)

    # Update queue metadata
    queue["items_processed"] = processed_count

    # Guarded write first: the log may only record stamps the queue actually
    # carries, so rows land after the rename that made them durable.
    write_queue_atomic(QUEUE_PATH, queue, stamped=processed_count)
    append_process_log(LOG_PATH, rows)

    print("Groundskeeper Run Summary")
    print(f"- Processed: {processed_count}")
    print(f"- Skipped: {processed_count} ({REASON})")
    print(f"- Types: ORPHAN_FILE: {processed_count}")
    print("- Health Score: N/A (known survey bug)")
    print("SIGNAL:TASK_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
