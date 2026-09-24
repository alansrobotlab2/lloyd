# Intelligence Pipeline MVP

A modular pipeline for discovering, filtering, and scoring intelligence items from multiple sources.

## Architecture

```
intel_pipeline/
├── __init__.py            # Package init
├── __main__.py            # CLI: --scan / --score / --write / --date
├── _paths.py              # Feeds and vault paths, derived from app.paths
├── models.py              # Data models (FeedItem, ScoredItem)
├── state.py               # Seen-set and raw JSONL helpers
├── profile.py             # Interest profile loading
├── scoring.py             # Two-stage scoring pipeline
├── vault_writer.py        # Digests into the vault, with the write guards
└── scanners/
    ├── __init__.py        # Scanners package init
    ├── github_scanner.py  # GitHub releases/commits/issues (config/github-repos.yml)
    └── youtube_scanner.py # YouTube channel RSS (config/youtube-channels.yml)
```

## Data Models

### FeedItem
Base item from any feed source:
- `id`: Unique identifier
- `source`: Source name (e.g., "arxiv", "hackernews")
- `title`: Item title
- `url`: Link to original item
- `summary`: Brief description
- `discovered_at`: ISO-8601 timestamp
- `published`: ISO-8601 date the SOURCE published the item, when it says so
  (`atom:published` for YouTube). Empty means the source served no date — which is
  normal for GitHub and is treated as inside the freshness window, never as old.
- `authors`: List of authors
- `source_tags`: Tags from source

### ScoredItem
Extended FeedItem with scoring:
- `relevance`: 1-10 score
- `urgency`: urgent|morning|weekly|low
- `why`: Explanation for scoring
- `projects`: Matched projects
- `category`: Primary category

## Write guards (`vault_writer.py`)

Two conditions stop a write, on both routes into the vault
(`write_all_to_vault` and `write_item_to_vault`):

- **`MAX_ITEM_AGE_DAYS = 30`** — an item whose own `published` date is older than this
  is held out of the digests, with the held count and the oldest age printed. An item
  with no `published` is always written, so a source that stops serving dates cannot
  zero the writer. Without it, the 2026-09-22 run filed 102 videos as the day's
  `URGENT` news, 28 of them more than 30 days old and the oldest 453 days (backlog
  #1379).
- **`STATE_LOSS_MIN_SCORED_ITEMS = 100`** — combined with an *empty* `seen` set in
  `scanner-state.json`, this is state loss, not a busy day: the run prints
  `STATE LOSS: ...` and writes nothing. The day it fired measured 258 scored items
  from 1007 raw against 9 from 49 normally. Restore the state from
  `~/.lloyd-data-snapshots` (`scripts/backup/restore-data.sh`) and re-run `--write`.

## Configuration

Interest profile stored in `~/obsidian/interests.md` (markdown format):

```markdown
### Humanoid Robotics
- **Weight:** 0.9 (high priority)
- **Keywords:** gr00t, isaac lab, humanoid, locomotion, bipedal
- **Projects:** Alfie, Yoshi
- **Depth:** deep
```

## State Management

Under `~/lloyd-data`, not the code tree and not the vault — runtime data moved out of
`~/lloyd` on 2026-09-22 so no `git clean` or fixture teardown aimed at the code can
reach it again. `intel_pipeline/_paths.py` derives all three from `app.paths`:

- **Seen items**: `~/lloyd-data/_pipeline/vault-derived/memory/feeds/scanner-state.json`
- **Raw items**: `~/lloyd-data/_pipeline/vault-derived/memory/feeds/raw/YYYY-MM-DD.jsonl`
- **Scoring output**: `~/lloyd-data/_pipeline/vault-derived/memory/feeds/intel-YYYY-MM-DD.jsonl`

## Usage

### CLI Entry Points

Run as a module from the package directory (this is what autonomy task #30 does;
with no flag it scans, scores and writes in one pass):
```bash
cd ~/lloyd/scripts/intel-pipeline
python -m intel_pipeline                       # scan + score + write for today
python -m intel_pipeline --scan                # scanners only
python -m intel_pipeline --score --write --date 2026-09-22
```

### Programmatic Usage

```python
from intel_pipeline.scanners.github_scanner import scan_github_repos
from intel_pipeline.scanners.youtube_scanner import scan_youtube_channels
from intel_pipeline.scoring import run_scoring_pipeline
from intel_pipeline.profile import load_profile

# Load profile
profile = load_profile()

# Scan sources (each reads its config/*.yml and saves its own raw JSONL)
github_items = scan_github_repos()
youtube_items, coverage = scan_youtube_channels()

# Combine and score
all_items = github_items + youtube_items
scored = run_scoring_pipeline(all_items, profile)

# Process results
for item in scored:
    if item.urgency == "urgent":
        print(f"URGENT: {item.title}")
```

## Scoring Pipeline

### Stage 1: Filter
- Match items against interest profile keywords
- Filter out irrelevant items

### Stage 2: Score
- Calculate relevance (1-10)
- Determine urgency level
- Generate explanation
- Match to projects
- Assign category

## Requirements

```bash
pip install -r requirements.txt
```

## Development

Install dependencies:
```bash
pip install pyyaml requests
```

Run tests (they live in the repo's suite, not in this directory):
```bash
cd ~/lloyd && .venvs/lloyd/bin/python -m pytest tests/test_intel_pipeline_*.py
```
