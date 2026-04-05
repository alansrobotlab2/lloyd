#!/usr/bin/env python3
"""Process groundskeeper queue - mark ORPHAN_FILE items as skipped due to survey bug."""
import json
from datetime import datetime, timezone

queue_path = "/home/alansrobotlab/lloyd/_pipeline/groundskeeper-queue.json"
log_path = "/home/alansrobotlab/lloyd/_pipeline/groundskeeper-log.jsonl"

# Read queue
with open(queue_path, 'r') as f:
    queue = json.load(f)

# Process items
processed_count = 0
skipped_count = 0
timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

for item in queue["items"]:
    if item.get("status") == "pending" and item.get("type") == "ORPHAN_FILE":
        item["status"] = "skipped"
        item["reason"] = "hub-page-linked-survey-bug"
        item["processed_at"] = timestamp
        processed_count += 1
        
        # Log result
        log_entry = json.dumps({
            "item_id": item["id"],
            "type": item["type"],
            "status": "skipped",
            "reason": "hub-page-linked-survey-bug",
            "processed_at": timestamp
        })
        with open(log_path, 'a') as log:
            log.write(log_entry + "\n")

# Update queue metadata
queue["items_processed"] = processed_count

# Write updated queue
with open(queue_path, 'w') as f:
    json.dump(queue, f, indent=2)

print(f"Groundskeeper Run Summary")
print(f"- Processed: {processed_count}")
print(f"- Skipped: {processed_count} (hub-page-linked-survey-bug)")
print(f"- Types: ORPHAN_FILE: {processed_count}")
print(f"- Health Score: N/A (known survey bug)")
print(f"SIGNAL:TASK_COMPLETE")
