"""Report-only count of conflicting look-alikes in a turn's merged prefetch (#3801, step 4).

NVIDIA's *You don't need long context* result is a claim about softmax
competition: a handful of near-copies of the evidence can halve its attention
score, and past a threshold the reasoning falls off a cliff. `distractK` in
`eval/run_context_rot_eval.py` measures that cliff on a synthetic stimulus
(digit permutations of one relay's port). This file is the only thing that ties
that synthetic k to the harness: over the turns the engine actually received,
how often does the merged prefetch top-K hold **two or more items that resolve
to the same entity while carrying different values**? That is the class the
standing problem is made of — sibling notes with thin facts, an oblique
reference landing on the wrong row — and `eval/dup_detect.py`'s nightly
`dup_slot_share` cannot see it, because a same-form-different-value pair scores
LOW on its predicate (identical twins are the measured-clean 0.005 case).

**Nothing here changes assembly.** No rung, no cap, no reorder: the module
parses the `<context>` block a turn was already sent and reports a number.
`--help` prints and `main()` writes nothing to disk. If the measured cliff turns
out to sit at a k real turns produce, the collapse/disambiguation change is its
own proposal with this distribution as its evidence — that is the item's own
scope line, and it is why this is a counter and not a guard.

WHAT THE HISTOGRAM IS, AND WHAT IT IS NOT. `max_items_for_one_entity` counts
items landing on ONE entity, which is the harness's own shape: a relay note, its
sibling notes and a stale fact all arriving on one topic. The synthetic arm is
NOT that shape — a `distractK` cell attributes its k look-alikes to k DISTINCT
sibling relays (that is what keeps the family under the duplicate threshold;
pinned by `test_every_lookalike_is_a_wrong_answer_and_none_of_them_is_the_answer`
in `tests/test_context_rot_eval.py`), so the model there sees k items on k
different entities plus one needle. The two quantities share the prompt-item
axis, not the entity axis. Reading the measured curve beside this histogram has
to name that asymmetry, or it overstates the harness's exposure.

Two rules carry the whole meaning, and both are stated so a later reader can
re-decide them:

* **Same entity.** A `<facts>` line asserts its entity (`- [Entity] text`), so
  the surface it names is the entity, canonicalised through the KG alias table
  when the store answers. A `<vault-context>` hit asserts only a file, and a file
  stem is not an entity claim: it counts only when the KG resolves it, and
  otherwise lands in `n_unresolved` rather than being guessed into a conflict.
  `app/prefetch.py` already dedupes vault hits **by file**, so two items can
  never share a path — the conflict this counts is one entity reached by two
  different routes, which is exactly what the per-file cap cannot bound.
* **Different value.** Two items on one entity with byte-identical normalised
  text are a duplicate (the #2271 class) and are not a conflict. Anything else
  is: the same entity stated two ways is what steals attention mass from the
  statement that happens to be current.

THE VAULT ROUTE IS DELIBERATELY NARROW, AND THE NUMBER IS A FLOOR. A `<facts>`
line arrives with its entity already chosen by the assembler, so that route is
exact. A vault hit is a chunk of prose, and the only sound entity it carries is
the one its own file stem names in the KG. The tempting widening — resolve a note
to every entity with a fact whose `source_doc` is that note — was measured and
rejected: `facts_idx` maps 910 distinct documents to 16,474 entities, and a daily
note alone cites dozens, so two daily notes would "conflict" on a dozen shared
entities that have nothing to do with the excerpt. That would manufacture the
very conflicts this counter is meant to count honestly.

Measured over the 120 newest transcripts and the 91 turns that carried a prefetch
section (2026-11-05, live store, `--days 3 --limit 120`): 238 items, all
`<vault-context>`, none of whose file stems named a KG entity — coverage 0.0,
0 turns with a conflict. So the conflict count is a floor on a sample that could
not see the class, and `resolution_coverage` is printed beside it for exactly
that reason. A run over a window holding interactive turns (whose prompts carry
`<facts>` lines, which resolve by declaration) is the measurement the deferred
verdict needs; this file is the instrument, not the result.

    .venvs/lloyd/bin/python eval/prefetch_conflict_count.py                  # last 14 days
    .venvs/lloyd/bin/python eval/prefetch_conflict_count.py --days 2 --limit 40
    .venvs/lloyd/bin/python eval/prefetch_conflict_count.py --json           # machine-readable
    .venvs/lloyd/bin/python eval/prefetch_conflict_count.py --fixture-check   # no sessions read
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSIONS = Path.home() / "lloyd-data" / "sessions"

#: The two injected sections whose items reach a prompt as ranked top-K lists.
SECTIONS = ("facts", "vault-context")

# `- [Entity] fact text (confidence: 0.9)`  — app/prefetch.py:858
FACT_LINE = re.compile(r"^-\s+\[(?P<entity>[^\]]+)\]\s+(?P<text>.*?)(?:\s*\(confidence:[^)]*\))?\s*$")
# `- **Title** (score: 1.00, file: a/b.md, line: 52): excerpt`  — app/prefetch.py:1353,
# and the same line with `, semantic hit from the previous turn's query` before the colon.
VAULT_LINE = re.compile(
    r"^-\s+\*\*(?P<title>[^*]+)\*\*\s+\((?P<meta>[^)]*)\):\s*(?P<excerpt>.*)$")
VAULT_FILE = re.compile(r"file:\s*(?P<file>[^,]+)")

#: A value is its text with case and whitespace folded and this many characters
#: kept: long excerpts that agree on their first 200 folded characters are the
#: same statement truncated, and calling that a conflict would inflate the count.
VALUE_MAX = 200

Resolve = Optional[Callable[[str], Optional[str]]]


@dataclass(frozen=True)
class Item:
    """One prefetched thing as the prompt carried it: where it came from, what
    entity it lands on (`None` = unresolved), and the value it states."""
    section: str
    source: str
    entity: Optional[str]
    value: str

    def as_dict(self) -> dict[str, Any]:
        return {"section": self.section, "source": self.source,
                "entity": self.entity, "value": self.value}


def value_of(text: str) -> str:
    """The comparison key for a stated value: case- and whitespace-folded, head
    clipped to `VALUE_MAX`. Identical keys mean a duplicate, not a conflict."""
    return " ".join((text or "").split()).casefold()[:VALUE_MAX]


def kg_resolver() -> Callable[[str], Optional[str]]:
    """Surface form -> canonical entity, through the live KG; `None` when the
    store does not know the surface.

    `kg_store.store()` raises rather than creating a database when nothing has
    been configured (`app/kg_store.py:1660`), which is the behaviour wanted here:
    a report-only counter must never be the thing that opens a store. A resolver
    that raises for any other reason degrades to "knows nothing", and every item
    then lands unresolved — the report says so in `n_unresolved` rather than
    inventing an entity."""
    try:
        from app import kg_store
        store = kg_store.store()
    except Exception:                                   # noqa: BLE001
        return lambda name: None

    def resolve(name: str) -> Optional[str]:
        for step in (getattr(store, "aliases", None), getattr(store, "entities", None)):
            fn = getattr(step, "resolve", None) or getattr(step, "lookup", None)
            if fn is None:
                continue
            try:
                got = fn(name)
            except Exception:                           # noqa: BLE001
                return None
            if got:
                return str(got)
        return None

    return resolve


def _section_lines(block: str, name: str) -> Iterable[str]:
    """Lines inside `<name>…</name>`, in document order — top-K order, which is
    the order the model reads them in."""
    open_, close_ = f"<{name}>", f"</{name}>"
    i = block.find(open_)
    if i < 0:
        return ()
    j = block.find(close_, i)
    body = block[i + len(open_):] if j < 0 else block[i + len(open_):j]
    return (ln for ln in body.splitlines() if ln.startswith("- "))


def parse_items(block: str, *, resolve: Resolve = None) -> list[Item]:
    """Every fact and vault hit a rendered `<context>` block carries, in order.

    `resolve` is the entity canonicaliser; `None` means no canonicalisation,
    which is what the fixture tests inject so a number never depends on the live
    KG. A fact line keeps its declared entity even unresolved — the line asserts
    it — while a vault hit without a KG answer becomes `entity=None`."""
    out: list[Item] = []
    for section in SECTIONS:
        for line in _section_lines(block, section):
            if section == "facts":
                m = FACT_LINE.match(line)
                if not m:
                    continue
                surface = m.group("entity").strip()
                out.append(Item("facts", surface, (resolve or (lambda n: None))(surface) or surface,
                                value_of(m.group("text"))))
                continue
            m = VAULT_LINE.match(line)
            if not m:
                continue
            fm = VAULT_FILE.search(m.group("meta") or "")
            path = (fm.group("file").strip() if fm else "")
            stem = Path(path).stem if path else m.group("title").strip()
            out.append(Item("vault-context", path or m.group("title").strip(),
                            (resolve or (lambda n: None))(stem),
                            value_of(m.group("excerpt"))))
    return out


def conflicts(items: list[Item]) -> dict[str, Any]:
    """The per-turn answer: does one entity appear with two or more different
    values, and how bad is the crowding.

    `max_distinct_values_one_entity` is the quantity meant to be read against
    the runner's k: the number of conflicting statements one entity arrives with.
    `max_items_for_one_entity` is its superset (duplicates included) and is
    reported beside it so a reader can see how much of the crowding is the #2271
    duplicate class rather than this one. Neither is the same variable as k — see
    WHAT THE HISTOGRAM IS above."""
    by_entity: dict[str, list[Item]] = {}
    for it in items:
        if it.entity:
            by_entity.setdefault(it.entity, []).append(it)
    conflicting: dict[str, Any] = {}
    for entity, group in by_entity.items():
        values = sorted({g.value for g in group})
        if len(values) > 1:
            conflicting[entity] = {"items": len(group), "distinct_values": len(values),
                                   "sources": sorted(g.source for g in group)}
    return {
        "n_items": len(items),
        "n_resolved": sum(1 for i in items if i.entity),
        "n_unresolved": sum(1 for i in items if not i.entity),
        "n_entities": len(by_entity),
        "max_items_for_one_entity": max((len(g) for g in by_entity.values()), default=0),
        "max_distinct_values_one_entity": max((len({g.value for g in grp})
                                               for grp in by_entity.values()), default=0),
        "conflicting_entities": sorted(conflicting),
        "conflict_details": conflicting,
        "conflict": bool(conflicting),
    }


def turn_report(block: str, *, resolve: Resolve = None) -> dict[str, Any]:
    """One turn's report: the items its prompt carried and whether they conflict.

    A block with no fact or vault section reports `has_prefetch: False` and is
    excluded from the distribution by `main()` — a turn that was sent no
    retrieved context is not evidence that conflicts are rare."""
    items = parse_items(block, resolve=resolve)
    return {"has_prefetch": bool(items), "items": [i.as_dict() for i in items],
            **conflicts(items)}


def _message_text(msg: dict) -> str:
    """A stored message as prose: a plain string, or the joined `text` parts of a
    list-content message. Saved sessions on this box carry the injected block on
    list-content messages far more often than on a plain string (measured over
    the 200 newest transcripts: 154 list-content `subliminal`, 7 list-content
    `user`), so reading only `str` content would report an empty distribution and
    read as "conflicts are rare"."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return ""


def _iter_turn_blocks(path: Path) -> Iterable[tuple[str, str]]:
    """(turn reference, message text) for every message in a saved session that
    carries a prefetch section.

    The reference names the message index and role, because the block reaches the
    model on whichever message the harness attached it to — `user` on an
    interactive turn, `subliminal` on an injected one — and the report has to say
    which it counted."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:                                   # noqa: BLE001
        return ()
    msgs = doc.get("messages") if isinstance(doc, dict) else None
    for i, m in enumerate(msgs or []):
        if not isinstance(m, dict):
            continue
        text = _message_text(m)
        if any(f"<{s}>" in text for s in SECTIONS):
            yield f"{path.name}#msg{i}:{m.get('role')}", text


def collect(sessions_dir: Path = DEFAULT_SESSIONS, *, days: int = 14,
            limit: int = 0, resolve: Resolve = None,
            now: Optional[datetime] = None) -> dict[str, Any]:
    """The distribution over recent real turns. Read-only; nothing is written.

    `limit` bounds how many session FILES are read (newest first), not how many
    turns: one session holds many turns and the turn count is what `n_turns`
    reports."""
    resolve = resolve if resolve is not None else kg_resolver()
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    files = sorted(sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if limit:
        files = files[:limit]
    turns = 0
    n_items = n_resolved = n_unresolved = 0
    values_hist: Counter[int] = Counter()   # the axis read against the measured curve
    items_hist: Counter[int] = Counter()    # the same crowding, duplicates included
    per_section: Counter[str] = Counter()
    conflicting: list[dict[str, Any]] = []
    for f in files:
        if datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc) < cutoff:
            continue
        for ref, text in _iter_turn_blocks(f):
            rep = turn_report(text, resolve=resolve)
            if not rep["has_prefetch"]:
                continue
            turns += 1
            n_items += rep["n_items"]
            n_resolved += rep["n_resolved"]
            n_unresolved += rep["n_unresolved"]
            values_hist[rep["max_distinct_values_one_entity"]] += 1
            items_hist[rep["max_items_for_one_entity"]] += 1
            for it in rep["items"]:
                per_section[it["section"]] += 1
            if rep["conflict"]:
                conflicting.append({"turn": ref,
                                    "entities": rep["conflicting_entities"],
                                    "max_distinct_values_one_entity":
                                        rep["max_distinct_values_one_entity"]})
    return {"sessions_read": len(files), "n_turns": turns,
            "n_items": n_items, "n_resolved": n_resolved, "n_unresolved": n_unresolved,
            "resolution_coverage": round(n_resolved / n_items, 4) if n_items else None,
            "n_turns_with_conflict": len(conflicting),
            "share_with_conflict": round(len(conflicting) / turns, 4) if turns else None,
            "values_histogram": {str(k): v for k, v in sorted(values_hist.items())},
            "max_values_seen": max(values_hist, default=0),
            "items_histogram": {str(k): v for k, v in sorted(items_hist.items())},
            "max_items_seen": max(items_hist, default=0),
            "items_by_section": dict(per_section),
            "conflicting_turns": conflicting[:40]}


#: The turn this counter is specified against. Every shape in it is reachable in
#: a real rendered block: two facts on one entity carrying different claims, the
#: same fact twice (a duplicate, which is #2271's class and NOT a conflict), and
#: two vault hits from two different files that the KG resolves to one entity —
#: one entity reached by two routes, which the per-file cap in
#: `app/prefetch.py:_merge_vault_results` cannot bound, because it dedupes by
#: path and the paths differ.
FIXTURE = (
    "<context>\n"
    "<facts>\n"
    "- [Voice Pipeline] The barge-in gate ships behind VOICE_BARGEIN=1 (confidence: 0.9)\n"
    "- [Voice Pipeline] Barge-in is ON since 2026-09-24; the click-only ruling is retired (confidence: 0.8)\n"
    "- [djev] djev runs DiffusionGemma on GPU 2 (confidence: 0.9)\n"
    "- [djev] djev runs DiffusionGemma on GPU 2 (confidence: 0.7)\n"
    "</facts>\n"
    "<vault-context>\n"
    "- **Wake-word retrain** (score: 1.00, file: memory/2026/wake-word-retrain.md, line: 12): two acoustic models, retrained 09-18\n"
    "- **Wake word audio** (score: 0.90, file: knowledge/ops/wake-word-audio.md, line: 4): six phrases, two acoustic models\n"
    "- **Relay inventory** (score: 0.80, file: knowledge/ops/relay-inventory.md, line: 9): the relay listens on 7425\n"
    "</vault-context>\n</context>"
)

#: What the KG says about the fixture's surfaces: two note stems, one entity.
#: Injected rather than read live, so a test number never depends on the store.
FIXTURE_KG = {"wake-word-retrain": "Wake-Word & Audio",
              "wake-word-audio": "Wake-Word & Audio",
              "Voice Pipeline": "Voice Pipeline", "djev": "djev"}


def fixture_report(*, resolve: Resolve = None) -> dict[str, Any]:
    """The fixture turn as the counter sees it. With no resolver the two facts on
    `Voice Pipeline` conflict and the vault hits stay unresolved (counted, never
    guessed); with `FIXTURE_KG` the two wake-word notes join them, because the KG
    says they are one entity saying two things."""
    return turn_report(FIXTURE, resolve=resolve)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="read only the N newest session files")
    ap.add_argument("--no-kg", action="store_true",
                    help="skip entity canonicalisation: only declared fact entities can conflict")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--fixture-check", action="store_true",
                    help="print the fixture report and read no sessions")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fixture_check:
        print(json.dumps(fixture_report(), indent=1))
        return 0
    rep = collect(args.sessions_dir, days=args.days, limit=args.limit,
                  resolve=(lambda n: None) if args.no_kg else None)
    if args.json:
        print(json.dumps(rep, indent=1))
        return 0
    print(f"turns with injected facts/vault context: {rep['n_turns']} "
          f"(sessions read: {rep['sessions_read']}, window: {args.days}d)")
    print(f"turns holding >=2 items on one entity with different values: "
          f"{rep['n_turns_with_conflict']} "
          f"(share: {rep['share_with_conflict']})")
    print(f"max conflicting values on one entity, histogram: {rep['values_histogram']} "
          f"(largest seen: {rep['max_values_seen']}) — this is the axis the curve is read on")
    print(f"max items on one entity (duplicates included), histogram: "
          f"{rep['items_histogram']} (largest seen: {rep['max_items_seen']})")
    print(f"items by section: {rep['items_by_section']}")
    print(f"items: {rep['n_items']} resolved to an entity: {rep['n_resolved']} "
          f"(coverage {rep['resolution_coverage']}), unresolved: {rep['n_unresolved']} — "
          f"an unresolved item can never be counted either way, so the conflict number "
          f"above is a floor on that sample, not a rate")
    for t in rep["conflicting_turns"][:12]:
        print(f"  {t['turn']}: {t['entities']} "
              f"values={t['max_distinct_values_one_entity']}")
    print("report-only: nothing here changes prefetch or assembly (#3801 step 4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
