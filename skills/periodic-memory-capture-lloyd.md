---
category: autonomy
description: "Lloyd periodic transcript capture with watermark-based skip logic: extracts
  new Lloyd interactions from session JSON files every hour and appends to daily notes.
  Runs via autonomy scheduler hourly."
name: periodic-memory-capture-lloyd
segment: periodic-memory-capture-lloyd
tags:
- periodic-memory-capture-lloyd
type: notes
metadata:
---


# Periodic Memory Capture

**Watermark-gated execution:** Check for new sessions before processing.

## Pre-Flight: Watermark Check

**CRITICAL: Check watermark BEFORE doing any work.**

1. Read `~/lloyd/_pipeline/autonomy-watermarks.json`
2. Check `memory_capture.last_session_count`
3. Count current Lloyd session files:
   ```bash
   ls ~/lloyd/sessions/*.json 2>/dev/null | wc -l
   ```
4. Compare: If count hasn't changed since last run, write "No new sessions" and **EXIT EARLY**
5. Update watermark after successful capture

```bash
python3 << 'PYEOF'
import json
import os
import subprocess

# Read watermarks
watermark_path = os.path.expanduser("~/lloyd/_pipeline/autonomy-watermarks.json")
try:
    with open(watermark_path, 'r') as f:
        watermarks = json.load(f)
except FileNotFoundError:
    watermarks = {"memory_capture": {"last_session_count": 0}}

last_count = watermarks.get("memory_capture", {}).get("last_session_count", 0)

# Count current Lloyd session files
result = subprocess.run(
    'ls ~/lloyd/sessions/*.json 2>/dev/null | wc -l',
    shell=True, capture_output=True, text=True
)
current_count = int(result.stdout.strip()) if result.returncode == 0 else 0

# Compare
if current_count == last_count:
    print("NO_NEW_SESSIONS")
    exit(0)
else:
    print("HAS_NEW_SESSIONS")
    exit(1)
PYEOF
```

If "NO_NEW_SESSIONS" is printed, log "No new sessions since last run" and exit. Otherwise, proceed.

## Step 1: Extract Transcript

Run `python3 ~/lloyd/scripts/memory/extract-transcript.py --instance lloyd` via Bash.

If output is empty or no new content found, reply 'no new content' and stop.

## Step 2: Read Today's Daily Note

Use PST timezone for the date. Read the daily note using vault_get:
```
vault_get(path="memory/YYYY-MM-DD.md")
```

If the file doesn't exist, that's fine — you'll create it in step 4.

## Step 3: Extract & Deduplicate

Extract from the transcript: decisions, facts, context shifts, project updates, preferences, action/result pairs. Format as timestamped entries with session links. Only extract entries NOT already present in the daily note.

## Step 4: Write New Entries

Write ONLY the new entries by appending to the file using Bash with shell append:
```bash
cat >> ~/obsidian/memory/YYYY-MM-DD.md << 'MEMENTRY'
---

### Session HH:MM PDT — Title

<content>
MEMENTRY
```

If the daily note file doesn't exist yet, create it first:
```bash
cat > ~/obsidian/memory/YYYY-MM-DD.md << 'MEMENTRY'
---
segment: agents
---

# YYYY-MM-DD Daily Notes

## Sessions
MEMENTRY
```

Then append entries separately.

## Post-Flight: Watermark Update

After successful capture, update watermark:

```bash
python3 << 'PYEOF'
import json
import os
import subprocess

# Read watermarks
watermark_path = os.path.expanduser("~/lloyd/_pipeline/autonomy-watermarks.json")
try:
    with open(watermark_path, 'r') as f:
        watermarks = json.load(f)
except FileNotFoundError:
    watermarks = {}

# Count current Lloyd session files
result = subprocess.run(
    'ls ~/lloyd/sessions/*.json 2>/dev/null | wc -l',
    shell=True, capture_output=True, text=True
)
current_count = int(result.stdout.strip()) if result.returncode == 0 else 0

# Update watermark
from datetime import datetime
now = datetime.now().isoformat()
if "memory_capture" not in watermarks:
    watermarks["memory_capture"] = {}
watermarks["memory_capture"]["last_run"] = now
watermarks["memory_capture"]["last_session_count"] = current_count

# Write back
os.makedirs(os.path.dirname(watermark_path), exist_ok=True)
with open(watermark_path, 'w') as f:
    json.dump(watermarks, f, indent=2)

print("Watermark updated successfully")
PYEOF
```

## Consolidation Awareness

When `vault_search` returns >=4 results, the response includes a `consolidated_summary` field containing a pre-synthesized summary. Prefer using `consolidated_summary` for context synthesis instead of reading each result individually — it's faster and already deduplicated.

If `vault_search` is used in this skill, always include `scope="memory"`.

## Critical Rules

- **NEVER** use vault_write for the daily notes file — it does a full file replace and can destroy existing content.
- **ALWAYS** use shell append (`cat >>` or `echo >>`) for adding entries.
- Only use `cat >` (overwrite) when creating a new file that doesn't exist yet.
- Do not duplicate entries already in the file.
- **Watermark check at start:** If no new sessions, EXIT EARLY without processing.
- **Update watermark after successful capture.**
- **Skip already-captured sessions:** Sessions with `"captured": true` in their JSON have already been summarized by the post-session capture hook. Check this flag and skip those sessions unless the summary looks incomplete.

## Path Rule

Always use `memory/` as the path prefix for vault_get. The actual file on disk is at `~/obsidian/memory/`.