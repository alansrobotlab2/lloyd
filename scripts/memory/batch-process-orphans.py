#!/usr/bin/env python3
"""Batch process ORPHAN_FILE items in groundskeeper queue."""
import json
from datetime import datetime, timezone

# Read queue
with open('/home/alansrobotlab/obsidian/agents/lloyd/groundskeeper-queue.json', 'r') as f:
    queue = json.load(f)

# Process up to 25 pending ORPHAN_FILE items
timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
processed = 0

for item in queue['items']:
    if processed >= 25:
        break
    if item.get('status') == 'pending' and item.get('type') == 'ORPHAN_FILE':
        # Check if it's in an organized directory
        source = item.get('source_file', '')
        organized_dirs = ['projects/', 'agents/', 'knowledge/', 'work/', 'skills/', 'templates/', 'backlog/', 'memory/', 'personal/', 'lloyd/']
        
        is_organized = any(source.startswith(d) for d in organized_dirs)
        
        if is_organized:
            item['status'] = 'skipped'
            item['reason'] = 'legitimate organized project file in folder hierarchy'
            item['processed_at'] = timestamp
            processed += 1

# Update queue metadata
queue['items_processed'] = processed

# Write updated queue
with open('/home/alansrobotlab/obsidian/agents/lloyd/groundskeeper-queue.json', 'w') as f:
    json.dump(queue, f, indent=2)

print(f'Processed {processed} items')
