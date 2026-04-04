#!/bin/bash
# Process all remaining documents in batches of 10

SCRIPT_DIR="$HOME/obsidian/agents/memory/scripts/next-gen-memory"
LOG_FILE="$HOME/obsidian/memory/fact-extraction-loop.log"

echo "Starting batch processing at $(date)" > "$LOG_FILE"

# Count remaining docs
PROGRESS_FILE="$HOME/obsidian/memory/fact-extraction-progress.json"
REMAINING=$(cat "$PROGRESS_FILE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
processed = set(d['processed_docs'])
import os
from pathlib import Path
docs = list(Path.home().joinpath('obsidian').rglob('*.md'))
docs = [d for d in docs if all(p not in str(d) for p in ['node_modules', '.venv', 'facts', '.cache', '__pycache__'])]
remaining = [d for d in docs if str(d) not in processed]
print(len(remaining))
")

echo "Remaining documents: $REMAINING" | tee -a "$LOG_FILE"

BATCH_SIZE=10
BATCH_NUM=0

while true; do
    # Check remaining
    REMAINING=$(cat "$PROGRESS_FILE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
processed = set(d['processed_docs'])
import os
from pathlib import Path
docs = list(Path.home().joinpath('obsidian').rglob('*.md'))
docs = [d for d in docs if all(p not in str(d) for p in ['node_modules', '.venv', 'facts', '.cache', '__pycache__'])]
remaining = [d for d in docs if str(d) not in processed]
print(len(remaining))
")
    
    if [ "$REMAINING" -le 0 ]; then
        echo "All documents processed!" | tee -a "$LOG_FILE"
        break
    fi
    
    BATCH_NUM=$((BATCH_NUM + 1))
    echo "" | tee -a "$LOG_FILE"
    echo "=== Batch $BATCH_NUM - Processing next 10 docs ===" | tee -a "$LOG_FILE"
    
    cd "$SCRIPT_DIR"
    timeout 60 python3 run_batch_10.py 2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 124 ]; then
        echo "WARNING: Batch timed out" | tee -a "$LOG_FILE"
    fi
    
    sleep 1
done

echo "" | tee -a "$LOG_FILE"
echo "=== COMPLETION SUMMARY ===" | tee -a "$LOG_FILE"
cat "$PROGRESS_FILE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Total processed: {len(d[\"processed_docs\"])} documents')
print(f'Total facts: {d[\"total_facts\"]}')
print(f'Errors: {d[\"errors\"]}')
print(f'Completed at: {d[\"timestamp\"]}')
" | tee -a "$LOG_FILE"

