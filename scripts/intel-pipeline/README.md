# Intelligence Pipeline MVP

A modular pipeline for discovering, filtering, and scoring intelligence items from multiple sources.

## Architecture

```
intel_pipeline/
├── __init__.py          # Package init
├── models.py            # Data models (FeedItem, ScoredItem)
├── state.py             # State management helpers
├── profile.py           # Interest profile loading
├── scoring.py           # Two-stage scoring pipeline
└── scanners/
    ├── __init__.py      # Scanners package init
    ├── arxiv_scanner.py # arXiv API scanner
    └── hn_scanner.py    # Hacker News scanner
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
- `authors`: List of authors
- `source_tags`: Tags from source

### ScoredItem
Extended FeedItem with scoring:
- `relevance`: 1-10 score
- `urgency`: urgent|morning|weekly|low
- `why`: Explanation for scoring
- `projects`: Matched projects
- `category`: Primary category

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

- **Seen items**: `~/obsidian/memory/feeds/scanner-state.json`
- **Raw items**: `~/obsidian/memory/feeds/raw/YYYY-MM-DD.jsonl`
- **Scoring output**: `~/obsidian/memory/feeds/intel-YYYY-MM-DD.jsonl`

## Usage

### CLI Entry Points

Run as module:
```bash
cd ~/obsidian/agents/lloyd/intel-pipeline
python -m intel_pipeline.scanners.arxiv_scanner
python -m intel_pipeline.scanners.hn_scanner
```

### Programmatic Usage

```python
from intel_pipeline.scanners.arxiv_scanner import scan_arxiv
from intel_pipeline.scanners.hn_scanner import scan_hn
from intel_pipeline.scoring import run_scoring_pipeline
from intel_pipeline.profile import load_profile

# Load profile
profile = load_profile()

# Scan sources
arxiv_items = scan_arxiv("all:reinforcement learning", max_results=10)
hn_items = scan_hn("machine learning", max_results=10)

# Combine and score
all_items = arxiv_items + hn_items
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

Run tests:
```bash
python -m pytest
```
