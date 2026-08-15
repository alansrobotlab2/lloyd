---
segment: agents
generated: 2026-08-14 02:00 PST
data_range: 2026-08-11 to 2026-08-13
---

# Signal Report — 2026-08-14

## Queued Signals (met threshold)

### Explicit (act on first occurrence)

| ID | Date | Type | Category | Description | Source |
|----|------|------|----------|-------------|--------|
| 1  | 2026-08-12 | correction | tool-use | youtube-transcript-api failed with API signature error; 5 retries before eventual success — tool reliability issue | extraction |
| 2  | 2026-08-12 | correction | tool-use | yt-dlp failed due to missing JS runtime (no node installed) — wrong tool choice for environment | extraction |

### Inferred (met 2+ threshold)

| ID | Date | Type | Category | Description | Frequency | Source |
|----|------|------|----------|-------------|-----------|--------|
| 1  | 2026-08-11/13 | pattern | memory-capture | No daily notes written for 2 consecutive days — memory capture gap | 2x | daily notes |

## Pending Signals (below threshold)

- [No daily note written for 2026-08-10 — single occurrence, monitor] — [1 occurrence, monitor]

## Tool Failure Patterns

- **Tool:** youtube-transcript-api — **Error type:** API signature error, failed on first 5 attempts before eventually succeeding — **Occurrences:** 5 retries (1 session) — **Recommendation:** Add skill-level guardrail: try youtube-transcript-api first, fall back to browser_evaluate with page.transcriptExtractor if API fails; encode correct instantiation pattern to avoid signature errors
- **Tool:** yt-dlp — **Error type:** No JS runtime available (node not installed) — **Occurrences:** 1 (1 session) — **Recommendation:** Avoid yt-dlp in environments without node; default to youtube-transcript-api or browser-based extraction; add environment check before attempting yt-dlp

## Positive Patterns to Reinforce

- **Pattern:** Browser-based extraction for SPA product pages — **Evidence:** 1 successful session (2026-08-12, 0 errors, 5 tools) — **Action:** Encode as skill: when http_fetch returns JS-rendered content, switch to browser_navigate + browser_snapshot + browser_evaluate with page.transcriptExtractor
- **Pattern:** Ambient signal suppression — **Evidence:** 1 successful session (2026-08-12, ambient signals correctly suppressed via ambient_decide) — **Action:** Reinforce in operating contract: ambient_decide should be called for all ambient signals to filter non-notable content
- **Pattern:** Fallback chain for YouTube transcripts — **Evidence:** 1 successful session (2026-08-12, youtube-transcript-api succeeded after yt-dlp failure) — **Action:** Encode as skill: try youtube-transcript-api first, then browser_evaluate with page.transcriptExtractor, then VTT parsing as final fallback
