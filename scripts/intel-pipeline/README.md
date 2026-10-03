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
- `grade_source`: why the item has the score it has — `model` (the stage-2 call
  returned a usable grade), `no_usable_grade` (it was asked and nothing usable came
  back), `call_cap` (eligible, but `LLM_MAX_CALLS` came first — the writer refuses
  this one, #1380), or `keyword` (never eligible: engine off, or below the keyword
  threshold)

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

## Entry bodies (`body.py`)

A feed description becomes a digest entry's body only after `strip_link_footer` has been
through it, and since backlog #2143 that includes **removing every block that carries a
URL** — a block being a run of non-blank lines, removed whole, including the sentence that
introduced the link. Whatever is left is the body; if nothing is left, the entry carries the
scorer's `why` instead.

It keys on a URL being present, never on the wording around it, because the corpus is
open-set: the four shape rules that came before it (a separator rule, a footer label from a
closed list, an arrow prefix, a short colon label over two or more link lines) each
recognised one measured channel's style, and each was walked through by the next one. The
row that filed #2143 had a 79-character label over a single bare UTM link — too long for
the label rule, too short for the run rule — and a second published row was an ad whose
footer had already been stripped above it.

Two consequences to know before wondering where a description went:

- A URL glued into a line of prose takes that line's whole block with it — 39 of the 55
  URL-carrying youtube rows in `feeds/raw/2026-*.jsonl` write their links that way
  (`CPU: Ryzen 9800X3D https://amzn.to/40Nor9v` is the shape) — and a description that held
  nothing but link blocks (12 of those 55) ends up carrying `why`.
- GitHub is untouched: `_entry_body` only strips non-GitHub text, so an issue or PR body
  keeps its links, which is what `clean_body`/`clip_body` exist to publish.

The entry always carries its own `[Link](https://www.youtube.com/watch?v=…)` added by the
writer, so an entry that loses a channel's link does not lose its source.

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
- **First grade per item id**: `~/lloyd-data/_pipeline/vault-derived/memory/feeds/grades.jsonl`
  (`_paths.GRADE_STORE`). Appended when stage 2 gets a usable grade and never
  rewritten, so the first grade an id received is the grade it keeps. An eligible item
  whose id is already there is not put to the model again — it comes back with the
  stored grade and `grade_source=model`, costing no call and not freeing a call for
  anything else, so which ids `--max-calls` bites is unchanged (#2081). This is why it
  exists: greedy + seed (`#1232`) did not pin the answer, because the movement is not
  sampling — the same prompt bytes graded 3 five times in a row and then 4 once other
  prompts were interleaved — and `RELEVANCE_FLOOR = 4` with a strict `<` makes 4-vs-3
  write-vs-refuse, while `--score` rewrites the day file and the writer dedupes by id,
  so the day file kept the newest grade and the vault the first. Backlog #2139.
  Consequence: a grade is a point-in-time judgement, so re-weighting `interests.md`
  does not re-grade ids that already have one. Delete the file to re-grade.

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

#### Who gets a model call (`LLM_MAX_CALLS`, backlog #2081)
`stage2_score` buys at most `LLM_MAX_CALLS = 40` model grades, and
`scoring.stage2_allocation_order` decides who they go to. The key, highest priority
first: **feed source** (`SOURCE_ALLOCATION_RANK`: youtube before github), then the
stage-1 `keyword_score` descending, then position. Items the keyword stage rated
below `LLM_KEYWORD_THRESHOLD` never get a call wherever they land, and the returned
list stays in the caller's order — allocation decides who is asked, not the shape of
the day file.

Source is first because it is the cheapest stage-1 signal that discriminates. On
2026-10-02, the first day the cap bound, 20 of the 33 graded GitHub items came in
below the writer's relevance floor and wrote nothing, against 1 of the 7 graded
videos — and the budget, spent in the day file's row order (the raw file is appended
to, so it is the order the day's scan passes found things), was what left the two
YouTube videos at the tail of the day ungraded. Keyword score is second
because it carries no ordering information at the top of the scale today: it is
`max(topic weight)` and every one of those 42 survivors scored exactly 1.0, so
ordering by it alone refuses the same items. Weights are a human decision (#1380,
ruled 2026-09-27) and this key reads them rather than inventing replacements.

What does not change: an item the cap never reached keeps
`grade_source = "call_cap"` and `vault_writer.refused_by_call_cap` still refuses to
write it (#1380). The `=== Top Results ===` block applies that same predicate, so a
refused item cannot headline a run.

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
