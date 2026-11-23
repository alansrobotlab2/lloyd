"""What kind of work a backlog item is, and what selection does with that.

Until 2026-11-22 the loop picked its next item with no notion of whether the
item was meant to move a measured behaviour, repair something, or keep the
books. Triage now says which, in two front-matter keys on the item:

    class:  improvement | fix | maintenance | simplify
    metric: <the bench an `improvement` is expected to move>

* `improvement` — expected to move a named, measured bench (`metric`).
* `fix`         — something is broken and this repairs it.
* `simplify`    — same behaviour, fewer lines. Deletion is creditable work:
  the round must remove more non-test Python lines than it adds.
* `maintenance` — everything else (tests, docs, bookkeeping, plumbing).

Both keys are optional everywhere they are read. An item triaged before this
module existed, a group-triage umbrella (its schema carries no class), a
verdict whose model left the line out, and an unknown word all read as
"unclassified", and an unclassified item is ordered exactly as it always was.

Everything here is a leaf: nothing at module level imports `backlog`,
`scorecard` or `app.config`, because `backlog` imports this module at its
foot. The hooks elsewhere are single lines, kept line-neutral on purpose —
`backlog.py`, `gate.py` and both worker sources are pinned by line number from
tests and docs:

    backlog.py     foot: `extend_triage_schema(TRIAGE_VERDICT_SCHEMA)`;
                   `select_confirmed._key`: `selection_rank_of(item)`
    autotriage.py  foot: `PROMPT = splice_triage_prompt(PROMPT)`; `_FIELD`;
                   both parse paths; `write_to_item` after `record_verdict`
    autocode.py    `_build_prompt`: `contract_note(candidate)`
    gate.py        `rung_preflight`: `gate_data(...)` (observe-only)
    scorecard.py   `_tree_growth`: `deleted_*` and `net_non_test`

Long version: `architecture/backlog.md` "Item class", `architecture/automod.md`
§3.2e.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger("lloyd-automod.item_class")

CLASSES: tuple[str, ...] = ("improvement", "fix", "maintenance", "simplify")
IMPROVEMENT, SIMPLIFY = "improvement", "simplify"
#: What the grammar offers a turn whose verdict block carried no CLASS line (a
#: session that started before the prompt changed, a retiring verdict). Never
#: stored: `clean_class` reads it as unclassified.
NO_CLASS = "none"

#: The grammar's ceiling on `metric`, and what the reader keeps. The cap sits
#: strictly above the slice, like every other leaf of the triage schema (#2444).
METRIC_MAX = 120
METRIC_KEEP = 80

FM_CLASS, FM_METRIC = "class", "metric"

_NO_VALUE = frozenset({"", "none", "n/a", "na", "-", "null", "unknown", "unclassified"})


def clean_class(value: Any) -> str:
    """One of `CLASSES`, or "" for anything else (absent, `none`, a typo)."""
    s = str(value or "").strip().strip("`'\"<>.").lower()
    return s if s in CLASSES else ""


def clean_metric(value: Any) -> str:
    """A bench name as one short line, or "" for a placeholder."""
    s = " ".join(str(value or "").split()).strip("`'\"<> ")
    return "" if s.lower().rstrip(".") in _NO_VALUE else s[:METRIC_KEEP]


def _fields(cls: Any, metric: Any) -> dict[str, str]:
    """`{}` when there is no class — so a verdict that names none parses to
    exactly the dict it parsed to before these keys existed. `metric` is kept
    only for `improvement`: on any other class it means nothing."""
    c = clean_class(cls)
    if not c:
        return {}
    m = clean_metric(metric) if c == IMPROVEMENT else ""
    return {FM_CLASS: c, **({FM_METRIC: m} if m else {})}


def class_from_structured(obj: Mapping[str, Any]) -> dict[str, str]:
    """The finalizer object's class fields; `{}` when it carries none."""
    return _fields(obj.get("class"), str(obj.get("metric") or "")[:METRIC_KEEP])


def class_from_fields(fields: Mapping[str, list[str]]) -> dict[str, str]:
    """The text block's `CLASS:` / `METRIC:` lines; `{}` when there are none.
    First line only: neither value is prose."""
    def first(key: str) -> str:
        return (fields.get(key) or [""])[0]
    return _fields(first("CLASS"), first("METRIC"))


# ── the schema ──────────────────────────────────────────────────────────────

def extend_triage_schema(schema: dict) -> dict:
    """Add `class` and `metric` to the single-item triage grammar, in place.

    Both are in `required`, because `tests/test_structured_verdict.py` pins that
    this schema requires every property it declares (the request goes out
    `strict: true`). What keeps them OPTIONAL is the value space and the reader:
    `class` may be `none`, `metric` may be empty, and `class_from_structured`
    reads a missing key, a `none` and an unknown word the same way — as no
    class. Idempotent, so a reload cannot double the `required` entries.
    """
    props = schema.setdefault("properties", {})
    props.setdefault("class", {
        "type": "string", "enum": [*CLASSES, NO_CLASS],
        "description": ("For `confirmed`: what kind of work this is. `improvement` = "
                        "expected to move a named measured bench; `fix` = repairs "
                        "something broken; `simplify` = same behaviour, fewer lines; "
                        "`maintenance` = everything else. `none` when the verdict "
                        "block states no class.")})
    props.setdefault("metric", {
        "type": "string", "maxLength": METRIC_MAX,
        "description": ("For class `improvement`: the bench it is expected to move, "
                        "by name. Empty otherwise.")})
    required = schema.setdefault("required", [])
    for key in ("class", "metric"):
        if key not in required:
            required.append(key)
    return schema


# ── the prompt ──────────────────────────────────────────────────────────────

def manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "eval" / "fitness" / "manifest.yaml"


def bench_names(path: Path | None = None) -> list[str]:
    """Bench names from `eval/fitness/manifest.yaml`, or [] when there is no
    such file (then `metric` is free text). Tolerant of the two obvious shapes
    — a `benches:`/`metrics:` mapping or list of `{name: …}` — because the
    manifest is another change's file and this must not break on its layout."""
    path = path or manifest_path()
    try:
        import yaml
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - absent or unreadable: free text
        return []
    if not isinstance(doc, dict):
        return []
    node = next((doc[k] for k in ("benches", "metrics", "benchmarks") if k in doc), None)
    names: list[str] = []
    if isinstance(node, dict):
        names = [str(k) for k in node]
    elif isinstance(node, list):
        names = [str(e.get("name") or e.get("id") or "") if isinstance(e, dict) else str(e)
                 for e in node]
    return [n for n in (re.sub(r"[{}\s]+", " ", n).strip() for n in names) if n][:40]


TRIAGE_PARAGRAPH = (
    "- **Say what kind of work it is** (CLASS, for a `confirmed` item): "
    "`improvement` = expected to move a measured bench, which METRIC names%s; "
    "`fix` = something is broken and this repairs it; `simplify` = same "
    "behaviour, fewer lines (its round must remove more non-test Python lines "
    "than it adds); `maintenance` = everything else. No bench that would show "
    "the gain means it is not an `improvement`.\n")
TRIAGE_BLOCK_LINES = (
    "CLASS: <if confirmed: one of improvement|fix|maintenance|simplify; otherwise the word none>\n"
    "METRIC: <if improvement: the bench it should move, by name; otherwise the word none>\n")
_PARAGRAPH_ANCHOR = "\nFinish with exactly this block and nothing after it:"
_BLOCK_ANCHOR = "CHECK: <the command or method you ran, one line>\n"


def splice_triage_prompt(prompt: str, benches: Iterable[str] | None = None) -> str:
    """The single-item triage prompt with the class paragraph and block lines.

    Spliced rather than typed into the template so `autotriage.py` keeps its
    line numbers. Both anchors must be present: a prompt reworded past them
    raises at import, which every test of the source then reports, instead of
    shipping a prompt that silently never asks. The result still goes through
    `str.format`, so nothing added here may carry a brace.
    """
    if TRIAGE_BLOCK_LINES in prompt:
        return prompt
    names = list(bench_names() if benches is None else benches)
    hint = f" (one of: {', '.join(names)})" if names else ""
    for anchor in (_PARAGRAPH_ANCHOR, _BLOCK_ANCHOR):
        if prompt.count(anchor) != 1:
            raise ValueError(f"triage prompt anchor not found exactly once: {anchor!r}")
    out = prompt.replace(_PARAGRAPH_ANCHOR, TRIAGE_PARAGRAPH % hint + _PARAGRAPH_ANCHOR)
    return out.replace(_BLOCK_ANCHOR, TRIAGE_BLOCK_LINES + _BLOCK_ANCHOR)


# ── the item ────────────────────────────────────────────────────────────────

def write_to_item(item: Any, parsed: Mapping[str, Any], *, closed: bool = False) -> dict[str, str]:
    """Put a `confirmed` verdict's class on the item; return the ledger fields.

    Runs after `record_verdict`, through `update_frontmatter` — the writer that
    takes the item's lock and re-dumps every key it parsed. `{}` and no write
    for a verdict with no class, one that is not `confirmed`, or an item the
    verdict closed. A `metric` left over from an earlier triage is removed when
    this one names none. Fail-open: a write that raises costs the two keys and
    a warning, never the triage.
    """
    fields = {k: str(parsed[k]) for k in (FM_CLASS, FM_METRIC) if parsed.get(k)}
    if not fields.get(FM_CLASS) or closed or parsed.get("verdict") != "confirmed":
        return {}
    try:
        from scripts.automod import backlog as B
        B.update_frontmatter(item.path, {FM_CLASS: fields[FM_CLASS],
                                         FM_METRIC: fields.get(FM_METRIC)})
    except Exception as exc:  # noqa: BLE001
        logger.warning("backlog #%s: class not written (%s: %s)",
                       getattr(item, "id", "?"), type(exc).__name__, exc)
    return fields


def class_of(item: Any) -> tuple[str, str]:
    """`(class, metric)` off the item's front matter, ("", "") when unclassified.
    Through `backlog.front_matter_of`, which parses a file version once."""
    try:
        from scripts.automod import backlog as B
        fm = B.front_matter_of(item.path) or {}
    except Exception:  # noqa: BLE001
        return "", ""
    got = _fields(fm.get(FM_CLASS), fm.get(FM_METRIC))
    return got.get(FM_CLASS, ""), got.get(FM_METRIC, "")


# ── selection ───────────────────────────────────────────────────────────────

#: A recorded growth figure older than this is not the trailing week any more.
#: Housekeeping records a row every 23 h (`scorecard.RECORD_EVERY_S`).
GROWTH_MAX_AGE_S = 3 * 86400


@dataclass(frozen=True)
class Prefs:
    """What selection is asked to prefer. All off is the shipped state."""
    prefer_improvement: bool = False
    net_lines_budget: int | None = None
    #: Net non-test lines the tree grew over the trailing 7 days, from the
    #: newest recorded scorecard row; None when no fresh row carries it.
    growth: int | None = None

    @property
    def over_budget(self) -> bool:
        return (self.net_lines_budget is not None and self.growth is not None
                and self.growth > self.net_lines_budget)

    @property
    def active(self) -> bool:
        return self.prefer_improvement or self.over_budget


def selection_rank(cls: str, metric: str, prefs: Prefs,
                   known_metrics: Iterable[str] = ()) -> int:
    """An item's place inside its priority band. Lower is sooner.

    0 for every item while nothing is preferred, so the key it sits in sorts
    exactly as it did. Otherwise `simplify` first while the tree is over its
    net-lines budget, then an `improvement` that names a metric when
    `prefer_improvement` is on, then everything else. An `improvement` with no
    metric is not preferred: the class is a claim, the bench is what could
    check it. Where a manifest lists the benches, the name must be one of them.
    """
    if not prefs.active:
        return 0
    if prefs.over_budget and cls == SIMPLIFY:
        return 0
    if prefs.prefer_improvement and cls == IMPROVEMENT and metric:
        known = {k.lower() for k in known_metrics}
        if not known or metric.lower() in known:
            return 1
    return 2


def growth_from_row(row: Mapping[str, Any] | None, *, now: float,
                    max_age_s: float = GROWTH_MAX_AGE_S) -> int | None:
    """Net non-test lines per 7 days out of one scorecard row, or None.

    None — never 0 — when the row is missing, stale, unmeasured, or predates
    `net_non_test` (rows before 2026-11-22 split only the additions, and
    additions minus ALL deletions would credit test deletions to the code)."""
    growth = (row or {}).get("tree_growth")
    if not isinstance(growth, dict) or not growth.get("measured") or "net_non_test" not in growth:
        return None
    try:
        from datetime import datetime
        at = datetime.fromisoformat(str((row or {}).get("computed_at"))).timestamp()
        days = float((row or {}).get("since_days") or 7)
        if now - at > max_age_s or days <= 0:
            return None
        return int(round(int(growth["net_non_test"]) * 7.0 / days))
    except (ValueError, TypeError):
        return None


def prefs_from(cfg: Mapping[str, Any] | None, row: Mapping[str, Any] | None = None, *,
               now: float | None = None) -> Prefs:
    """`automod.selection` plus the newest scorecard row, as `Prefs`. Pure."""
    sel = ((cfg or {}).get("automod") or {}).get("selection") or {}
    budget = sel.get("net_lines_budget")
    if isinstance(budget, bool) or not isinstance(budget, (int, float)):
        budget = None
    growth = None
    if budget is not None:
        growth = growth_from_row(row, now=time.time() if now is None else now)
    return Prefs(prefer_improvement=sel.get("prefer_improvement") is True,
                 net_lines_budget=None if budget is None else int(budget), growth=growth)


def newest_scorecard_row(path: Path | None = None) -> dict | None:
    """The last readable row of `scorecard.jsonl`, from a tail read. The
    growth figure comes from here — a recorded number — and never from `git
    log` in the selection path, which runs on every worker tick."""
    if path is None:
        from scripts.automod import scorecard as SC
        path = SC.scorecard_path()
    try:
        with Path(path).open("rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 262144))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            return row
    return None


def current_prefs() -> Prefs:
    """The live preference: `app.config.CONFIG` (read at boot, so a change
    needs a backend restart) and, only when a budget is set, the scorecard
    tail. Any failure reads as nothing preferred."""
    try:
        from app.config import CONFIG
        sel = (CONFIG.get("automod") or {}).get("selection") or {}
        row = newest_scorecard_row() if sel.get("net_lines_budget") is not None else None
        return prefs_from(CONFIG, row)
    except Exception:  # noqa: BLE001
        return Prefs()


def selection_rank_of(item: Any, prefs: Prefs | None = None) -> int:
    """`select_confirmed`'s hook. With nothing preferred it returns 0 without
    opening the item, so the shipped configuration costs one dict lookup."""
    prefs = current_prefs() if prefs is None else prefs
    if not prefs.active:
        return 0
    cls, metric = class_of(item)
    return selection_rank(cls, metric, prefs, bench_names())


# ── simplify: deletion as creditable work ───────────────────────────────────

_TEST_PATH = re.compile(r"(^|/)tests?/|(^|/)test_[^/]*\.py$|(^|/)conftest\.py$")


def net_lines(diff_numstat: str) -> dict[str, int]:
    """Non-test Python lines a `git diff --numstat` adds and deletes.

    `net_removed` is deleted minus added, so a positive number is a smaller
    tree. Tests, prose, config and binary rows (`-\\t-\\tpath`) are not counted:
    a simplification that deletes its own tests has not made the code smaller.
    """
    out = {"added": 0, "deleted": 0, "files": 0}
    for line in (diff_numstat or "").splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or not (parts[0].isdigit() and parts[1].isdigit()):
            continue
        path = parts[2].strip()
        if not path.endswith(".py") or _TEST_PATH.search(path):
            continue
        out["added"] += int(parts[0])
        out["deleted"] += int(parts[1])
        out["files"] += 1
    out["net_removed"] = out["deleted"] - out["added"]
    return out


SIMPLIFY_NOTE = (
    "\n\n    Implicit check (this item is `class: simplify` — same behaviour, fewer "
    "lines): net non-test Python lines removed must be positive. Count it before "
    "you gate (`git diff --numstat <base>...HEAD`, `.py` files outside tests: "
    "deleted minus added) and state the number in your summary; the gate's "
    "preflight records it as `simplify.net_removed`. Not a numbered clause: do "
    "not report it as one.")


def contract_note(item: Any) -> str:
    """What the implement prompt appends to the contract for a `simplify` item;
    "" for every other item, so their prompt is byte-identical."""
    return SIMPLIFY_NOTE if class_of(item)[0] == SIMPLIFY else ""


def gate_data(item_id: Any, base: str, worktree: Any) -> dict[str, Any]:
    """`{"simplify": {...}}` for a round on a `simplify` item, else `{}`.

    OBSERVE-ONLY: merged into the preflight rung's data so the number is on
    the ledger; it never fails the rung. `ok` is what a blocking check would
    have answered. Never raises — an instrument that cannot refuse must not
    be able to crash the rung it rides on either.
    """
    try:
        from scripts.automod import backlog as B
        item = B.item_by_id(int(item_id)) if item_id else None
        if item is None or class_of(item)[0] != SIMPLIFY:
            return {}
        r = subprocess.run(["git", "-C", str(worktree), "diff", "--numstat", "--no-renames",
                            f"{base}...HEAD"], capture_output=True, text=True,
                           timeout=60, check=False)
        if r.returncode != 0:
            return {"simplify": {"status": "unevaluated", "observe_only": True}}
        counts = net_lines(r.stdout)
        return {"simplify": {**counts, "ok": counts["net_removed"] > 0, "observe_only": True}}
    except Exception:  # noqa: BLE001
        return {}
