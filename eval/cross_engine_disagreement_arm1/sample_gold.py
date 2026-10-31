#!/usr/bin/env python3
"""Build arm 1's labelled gold sample from the djev shadow store — backlog #2824.

Reads `~/.local/state/lloyd-djev/shadow.jsonl` and the artifacts the seams read, and
writes `gold.jsonl` plus `sample.json` beside it. Nothing here is wired to anything:
the runner scores the file only when asked for the surface by name, and no live path
imports this module.

READ-ONLY BY CONSTRUCTION. The store is opened `"r"` and streamed through a bounded
deque — it is ~218k rows and roughly a gigabyte, so it is never loaded whole, and no
path in this file opens it for anything but reading. `tests/test_cross_engine_arm1_gold.py`
pins that over a real run against a read-only copy: the file's sha256 must be
identical afterwards.

WHAT MAKES A ROW ADMISSIBLE (both must hold, or the row is dropped and counted):

  1. its canvas re-derives. `app/djev_shadow` records `inputs_digest` = sha1 of
     `json.dumps([state, questions], sort_keys=True)` over exactly the text and the
     typed question set the seam sent, so rebuilding the state from the artifacts and
     re-digesting it proves the sample asks the same words. A mismatch is a dropped
     row, never a relabelled one.
  2. `rubric.py` yields a label for it from a named artifact. The common outcome for
     an `action_review` row is no label at all: most reviewed calls name nothing the
     operator's task text names, and that absence is not a warrant for `unrelated`.

STRATA, for a fixed `--seed` (clause 1). `dedupe` and `entity` take every admissible
row in the `--window`-row tail. `action_review` is split on `djev.min_label_mass` at
0.9 — the seam's own trust axis — with the `<0.9` pool capped at `MIN_LOW = 30`, which
is `MIN_NON_ABSTAINING`: the thin-confidence cell can then clear the measurement floor
on its own rather than being drowned by the confident majority. The `>=0.9` pool takes
the remainder to exactly `TOTAL`. A `review` stratum is declared and, because the
store has never had a `review` seam, contributes 0 rows and does not abort.

Usage:
    .venvs/lloyd/bin/python -m eval.cross_engine_disagreement_arm1.sample_gold --dry-run
    .venvs/lloyd/bin/python -m eval.cross_engine_disagreement_arm1.sample_gold
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Same guard `eval/djev/replay.py` sets: importing the harness modules must never let
# a recording fire while a reader is walking the store it reads.
os.environ.setdefault("LLOYD_DJEV_SHADOW", "0")

from app.djev_shadow import _digest                       # noqa: E402  (the ONE digest impl)
from app.paths import production_data_root                # noqa: E402  (data root, not the checkout)
from app.data_root import vault_root
from eval.djev import schemas                             # noqa: E402
from eval.cross_engine_disagreement_arm1 import rubric    # noqa: E402

SEAM_ACTION_REVIEW = "action_review"
SEAM_DEDUPE = "dedupe"
SEAM_ENTITY = "entity"
SEAM_REVIEW = "review"

#: The store. Read-only: this is the path every consumer of the shadow seam shares.
SHADOW_LOG = Path.home() / ".local" / "state" / "lloyd-djev" / "shadow.jsonl"
SESSIONS_DIR = production_data_root() / "sessions"
#: The vault identity record lives in the runtime data root. `app.paths.VAULT_FACTS_ROOT`
#: anchors to whichever tree imported it — inside a worktree that is an empty facts
#: directory, and `app/paths.py` prints a notice saying so — so the root is named from
#: the production data root, the same way `SESSIONS_DIR` above is.
FACTS_ROOT = production_data_root() / "_pipeline" / "vault-derived" / "facts"
CANARY_ROWS = _REPO / "eval/measurements/injection-canary/rows.jsonl"
BACKLOG_DIR = vault_root() / "backlog"
GOLD_PATH = _HERE / "gold.jsonl"
SAMPLE_PATH = _HERE / "sample.json"

#: The 0.9 the seam's own trust axis is drawn at, and the item's split point.
LABEL_MASS_SPLIT = 0.9
TOTAL = 120
MIN_LOW = 30
DEFAULT_SEED = 20261016
DEFAULT_WINDOW = 20_000
#: Admissibility is unknown before a row is opened, so each stratum walks a seeded
#: order and stops when its quota is met. This bounds the work a run does when the
#: early rows all fail reconstruction.
OVERSAMPLE = 6

#: The strata, in fill order. `review` is the seam named in the item's recipe; the
#: store has never carried it, so its pool is empty and the run must say so rather
#: than raise.
STRATA = ("action_review_low", "action_review_high", "dedupe", "entity", SEAM_REVIEW)


def stratum_of(row: dict) -> str | None:
    seam = str(row.get("seam") or "")
    if seam == SEAM_ACTION_REVIEW:
        mass = (row.get("djev") or {}).get("min_label_mass")
        if mass is None:
            return None
        return ("action_review_high" if float(mass) >= LABEL_MASS_SPLIT
                else "action_review_low")
    if seam in (SEAM_DEDUPE, SEAM_ENTITY, SEAM_REVIEW):
        return seam
    return None


# ── reading the store ───────────────────────────────────────────────────────

def _strip_heavy(row: dict) -> dict:
    """Drop the log-probability dump, which is most of the file's bytes and nothing
    to a sampler: the decision is in `djev.answers`, the trust in `min_label_mass`."""
    djev = row.get("djev")
    if isinstance(djev, dict) and "chunks" in djev:
        row = {**row, "djev": {k: v for k, v in djev.items() if k != "chunks"}}
    return row


def iter_store_tail(path: Path = SHADOW_LOG, window: int | None = DEFAULT_WINDOW
                    ) -> Iterator[dict[str, Any]]:
    """The last `window` rows, streamed, oldest first. Read-only.

    A `deque(maxlen=window)` rather than a list: the file grows ~10k rows a day, so a
    reader that held it whole would grow with it. A malformed line is skipped, not
    fatal — the recorder writes best-effort JSON and an unparseable line is a dropped
    shadow write, never a reason a sampler cannot run.

    `window=None` streams the WHOLE store instead. Re-selection needs it: at ~10k rows
    a day a fixed 20,000-row tail stops containing a shipped sample's ids in about two
    days, so a sampler that could only see a fresh tail could never re-derive the file
    it shipped. The whole file is streamed line by line, never held as one list.
    """
    if window is None:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:                            # noqa: BLE001 — see above
                    continue
                if isinstance(row, dict):
                    yield _strip_heavy(row)
        return
    buf: deque[str] = deque(maxlen=max(0, int(window)))
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                buf.append(line)
    for line in buf:
        try:
            row = json.loads(line)
        except Exception:                                    # noqa: BLE001 — see docstring
            continue
        if isinstance(row, dict):
            yield _strip_heavy(row)


def store_witness(path: Path = SHADOW_LOG, window: int | None = DEFAULT_WINDOW
                  ) -> tuple[list[dict], dict]:
    rows = list(iter_store_tail(path, window))
    witness = {"path": str(path), "bytes": (path.stat().st_size if path.exists() else 0),
               "window_rows_read": len(rows),
               "first_ts": _iso(rows[0].get("ts")) if rows else None,
               "last_ts": _iso(rows[-1].get("ts")) if rows else None}
    return rows, witness


def _iso(ts: Any) -> str | None:
    try:
        return datetime.fromtimestamp(float(ts), timezone.utc).isoformat(timespec="seconds")
    except Exception:                                        # noqa: BLE001
        return None


# ── seam-side reconstruction ────────────────────────────────────────────────

def _session_turns(sid: str, sessions_dir: Path | None = None
                   ) -> dict[str, dict] | None:
    """`{turn_id: {"text": the operator's own text, "calls": [(id, name, rendered)]}}`.

    The reviewer is installed per turn with that turn's request text
    (`app/routers/messages.py:521`) and its `prior_calls` counts calls made *in* that
    turn (`app/harness/action_review.py:264`), so a canvas can only be rebuilt from
    the turn that owns the call. Sessions here are routinely multi-turn: reading the
    first user message of a four-turn session and the fourth turn's calls describes a
    page the seam never saw, which is why this groups by `turn_id` instead.
    """
    base = Path(sessions_dir or SESSIONS_DIR)
    for cand in (base / f"{sid}.json", base / f"{sid}.json.gz"):
        if not cand.exists():
            continue
        raw = (gzip.open(cand, "rt", encoding="utf-8", errors="replace")
               if cand.suffix == ".gz" else cand.open("r", encoding="utf-8",
                                                      errors="replace"))
        with raw as fh:
            doc = json.load(fh)
        turns: dict[str, dict] = {}
        for msg in (doc.get("messages") or []):
            tid = str(msg.get("turn_id", 0))
            turn = turns.setdefault(tid, {"text": None, "calls": []})
            if msg.get("role") == "user" and turn["text"] is None:
                content = msg.get("content")
                turn["text"] = ("".join(str(b.get("text") or "")
                                        for b in content if isinstance(b, dict))
                                if isinstance(content, list) else str(content or ""))
            for call in (msg.get("tool_calls") or []):
                fn = call.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:                        # noqa: BLE001
                        args = {}
                if isinstance(args, dict):
                    # The harness re-keys the canvas by the ORIGINAL arguments,
                    # `summary` excluded (app/harness/action_review.py:241), and
                    # truncates each rendering to ARG_CHARS.
                    args = {k: v for k, v in args.items() if k != "summary"}
                turn["calls"].append((str(call.get("id") or ""),
                                      str(fn.get("name") or ""),
                                      _render(args)))
        return turns
    return None


def _render(args: Any) -> str:
    """`app.harness.action_review.args_rendering`, imported not re-typed."""
    from app.harness.action_review import args_rendering
    return args_rendering(args)


def rebuild_action_review(row: dict, session_cache: dict,
                          sessions_dir: Path | None = None) -> dict | None:
    """The canvas this row was recorded against, verified by its own digest.

    Every piece of it is the shipped builder rather than a re-typed copy —
    `canvas_state`, `args_rendering`, `questions()` and `djev_shadow._digest` —
    because a re-typed list is exactly the drift `eval/djev/schemas.py:311-316`
    records biting a first implementation. The digest comparison is the proof; a row
    that does not digest is returned as None with the reason.
    """
    from app.harness.action_review import canvas_state, questions as ar_questions
    meta = row.get("meta") or {}
    sid, cid = str(meta.get("session_id") or ""), str(meta.get("call_id") or "")
    if not sid or not cid:
        return None
    if sid not in session_cache:
        session_cache[sid] = _session_turns(sid, sessions_dir)
    turns = session_cache[sid]
    if not turns:
        return None
    tid = str(row.get("turn_id", meta.get("turn_id", ""))) or None
    turn = turns.get(tid) if tid else None
    if turn is None:
        # `turn_id` is not in the row's own meta, so find the turn that holds the
        # call: the tool_call_id the recorder's `call_id` is is unique per turn.
        turn = next((t for t in turns.values()
                     if any(c[0] == cid for c in t["calls"])), None)
    if turn is None or not turn["text"] or not turn["calls"]:
        return None
    pos = next((i for i, (c_id, _, _) in enumerate(turn["calls"]) if c_id == cid), None)
    if pos is None:
        return None
    _, name, rendered = turn["calls"][pos]
    state = canvas_state(turn["text"], [(n, r) for (_, n, r) in turn["calls"][:pos]],
                         (name, rendered))
    qs = ar_questions()
    if _digest(state, qs) != str(row.get("inputs_digest") or ""):
        return None
    qid, spec = next(iter(qs.items()))
    return {"state": state, "question_id": qid, "instructions": spec["instructions"],
            "options": dict(spec["criteria"]), "asked_from_canvas": "whole",
            "task_text": turn["text"], "tool": name, "call_text": rendered,
            "session_file": str(Path(sessions_dir or SESSIONS_DIR) / f"{sid}.json")}


def _backlog_head(path: Path) -> dict | None:
    from agent_mcp import backlog_similar as SIM
    cfg = _dedupe_cfg()
    return SIM._head(path, head_bytes=int(cfg["head_bytes"]),
                     body_chars=int(cfg["body_chars"]))


def _dedupe_cfg() -> dict:
    from agent_mcp import backlog_similar as SIM
    return SIM.dedupe_config()


def _item_body(description_head: str) -> str:
    """The body as written minus the H1 the writer adds.

    `backlog_write_task` is handed `name` and `description` and the seam put the raw
    description on the canvas; on disk the same text sits under an `# <name>` heading
    with an activity section appended below it. This takes the description back out.
    Reconstruction is only ever trusted when the digest agrees, so a wrong guess here
    drops the row rather than mislabelling it.
    """
    body = []
    for line in str(description_head or "").splitlines():
        s = line.strip()
        if s.startswith("# ") or s.startswith("## Activity") or s.startswith("---"):
            break
        body.append(line)
    return "\n".join(body).strip()


def _find_backlog_item_by_title(title: str, backlog_dir: Path | None = None
                                ) -> Path | None:
    """The item whose on-disk title is exactly `title`, or None if not unique.

    The dedupe row records the new item's `name` in `meta` and nothing else about it,
    so a title match is the only join back to its description — and it is only honest
    while it is unique, which is what this checks. Two matches is None: a sample that
    guessed which item was meant would be labelling a pair it never saw.
    """
    want = str(title or "").strip()
    if not want:
        return None
    hits = []
    for path in sorted((backlog_dir or BACKLOG_DIR).glob("*.md")):
        head = _backlog_head(path)
        if head and str(head.get("title") or "").strip() == want:
            hits.append(path)
    return hits[0] if len(hits) == 1 else None


def rebuild_dedupe(row: dict, ctx: dict | None = None) -> dict | None:
    """One pair from a dedupe canvas: the would-be item beside one candidate.

    The candidate heads come off disk through the same `backlog_similar._head` the
    recorder's callable used, and the whole canvas is verified by digest before a
    slice of it is offered to the engines. `pair_state` is imported, never re-typed.
    """
    from eval.djev.schemas import pair_state
    meta = row.get("meta") or {}
    cands = [c for c in (meta.get("candidates") or []) if c.get("id")]
    merged = (row.get("actual") or {}).get("merged_into")
    if not cands or merged is None:
        # No recorded merge means no rule verdict to warrant anything (rubric
        # `label_dedupe`), so these rows are not admissible at all — and building
        # their canvas would be wasted work.
        return None
    picked = next((c for c in cands if int(c["id"]) == int(merged)), None)
    if picked is None:
        return None
    name = str(meta.get("name") or "")
    board = Path((ctx or {}).get("backlog_dir") or BACKLOG_DIR)
    item_path = _find_backlog_item_by_title(name, board)
    if item_path is None:
        return None
    item_head = _backlog_head(item_path) or {}
    description = _item_body(str(item_head.get("text") or ""))
    q = schemas.DEDUPE.spec["same_finding"]
    slices, qs = {}, {}
    for cand in cands:
        cid = int(cand["id"])
        cpath = next(iter(sorted(board.glob(f"{cid}-*.md"))), None)
        if cpath is None:
            return None
        chead = _backlog_head(cpath) or {}
        cid_key = f"cand{cid}"
        qs[cid_key] = {"type": q["type"],
                       "instructions": q["instructions"] + f" (Item B is backlog #{cid}.)",
                       "criteria": dict(q["criteria"])}
        slices[cid_key] = pair_state(name, description,
                                     f"#{cid} {chead.get('title') or ''}",
                                     str(chead.get("text") or ""))
    state = "\n\n---\n\n".join(slices[k] for k in qs)
    if _digest(state, qs) != str(row.get("inputs_digest") or ""):
        return None
    key = f"cand{int(merged)}"
    return {"state": slices[key], "question_id": key,
            "instructions": qs[key]["instructions"],
            "options": dict(qs[key]["criteria"]), "asked_from_canvas": "pair_slice",
            "item_file": str(item_path)}


def _overview_aliases(name: str, root: Path) -> tuple[str, ...]:
    """The `aliases:` field of a name's overview front matter, if it has one.

    Read here rather than borrowed because nothing else reads it: measured
    2026-10-16, not one of the ~1200 overviews under the facts root carries the field,
    which is why `rubric` has no `same` rule for the entity seam. It is read only to
    REFUSE a `different` label in the case where the vault does record the two names as
    one, so a field a human adds later cannot quietly make a committed gold wrong.
    """
    path = root / name / f"{name}-overview.md"
    if not path.exists():
        return ()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ()
    if not text.startswith("---"):
        return ()
    block = text.split("---", 2)[1] if len(text.split("---", 2)) >= 3 else ""
    m = re.search(r"^aliases:\s*(.*)$", block, re.M)
    if not m:
        return ()
    inline = m.group(1).strip()
    if inline.startswith("["):
        return tuple(a.strip().strip("'\"") for a in inline.strip("[]").split(",")
                     if a.strip())
    out = []
    for line in block[m.end():].splitlines():
        s = line.strip()
        if not s.startswith("- "):
            break
        out.append(s[2:].strip().strip("'\""))
    return tuple(x for x in out if x)


def rebuild_entity(row: dict, ctx: dict) -> dict | None:
    """The SAME/DIFFERENT pair's canvas, from the two overview files on file now."""
    from scripts.memory.entity_semantic_gate import entity_definition
    root = Path((ctx or {}).get("facts_root") or FACTS_ROOT)
    meta = row.get("meta") or {}
    a, b = str(meta.get("a") or ""), str(meta.get("b") or "")
    if not a or not b:
        return None
    da, db = entity_definition(a, root), entity_definition(b, root)
    if not da or not db:
        return None
    state = schemas.pair_state(a, da, b, db)
    q = schemas.ENTITY.spec["same_entity"]
    qs = {"same_entity": {"type": q["type"], "instructions": q["instructions"],
                          "criteria": dict(q["criteria"])}}
    if _digest(state, qs) != str(row.get("inputs_digest") or ""):
        return None
    return {"state": state, "question_id": "same_entity",
            "instructions": q["instructions"], "options": dict(q["criteria"]),
            "asked_from_canvas": "whole", "name_a": a, "name_b": b,
            "def_a": da, "def_b": db, "facts_root": str(root),
            "aliases_a": _overview_aliases(a, root),
            "aliases_b": _overview_aliases(b, root),
            "path_a": str(root / a / f"{a}-overview.md"),
            "path_b": str(root / b / f"{b}-overview.md")}


def rebuild_action_review_row(row: dict, ctx: dict) -> dict | None:
    return rebuild_action_review(row, ctx["sessions"], ctx.get("sessions_dir"))


#: seam -> the rebuild for it. `review` is deliberately absent: no seam of that name
#: has ever written a row, and the run reports that stratum as empty (clause 1)
#: instead of raising on a builder that cannot exist.
BUILDERS = {SEAM_ACTION_REVIEW: rebuild_action_review_row,
            SEAM_DEDUPE: rebuild_dedupe,
            SEAM_ENTITY: rebuild_entity}


# ── labelling ───────────────────────────────────────────────────────────────

def canary_tokens(canary_file: Path | None = None) -> tuple[str, ...]:
    """Every canary string the injection corpus defines, from the repo copy.

    Read-only, and read once. These are the only strings in this sampler's world that
    exist by construction inside fetched content and nowhere else, which is what makes
    one appearing in a call's arguments a warrant for `injected` rather than a hunch.
    """
    out: list[str] = []
    canary_file = Path(canary_file or CANARY_ROWS)
    if not canary_file.exists():
        return ()
    with canary_file.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except Exception:                                # noqa: BLE001
                continue
            tok = str(row.get("canary") or "").strip()
            if tok:
                out.append(tok)
    return tuple(dict.fromkeys(out))


def label(row: dict, rebuilt: dict, tokens: tuple[str, ...]) -> tuple[str, str] | None:
    seam = str(row.get("seam"))
    if seam == SEAM_ACTION_REVIEW:
        return rubric.label_action_review(
            rebuilt["task_text"], rebuilt["tool"], rebuilt["call_text"],
            session_file=rebuilt["session_file"], canary_tokens=tokens)
    if seam == SEAM_DEDUPE:
        merged = (row.get("actual") or {}).get("merged_into")
        return rubric.label_dedupe(row, int(merged)) if merged is not None else None
    if seam == SEAM_ENTITY:
        return rubric.label_entity(
            rebuilt["name_a"], rebuilt["name_b"], rebuilt["def_a"], rebuilt["def_b"],
            rebuilt["path_a"], rebuilt["path_b"],
            rebuilt["aliases_a"], rebuilt["aliases_b"])
    return None


# ── the sample ──────────────────────────────────────────────────────────────

def build_sample(*, seed: int = DEFAULT_SEED, window: int | None = DEFAULT_WINDOW,
                 store: Path = SHADOW_LOG, total: int = TOTAL,
                 min_low: int = MIN_LOW,
                 sessions_dir: Path | None = None,
                 facts_root: Path | None = None,
                 backlog_dir: Path | None = None,
                 canary_file: Path | None = None,
                 ids: tuple[str, ...] | Sequence[str] | None = None
                 ) -> tuple[list[dict], dict]:
    """The 120 gold rows and the sidecar that records how they were chosen.

    Two modes. With `ids=None` (the default) this draws a fresh sample: read a window,
    keep every admissible dedupe/entity/review row, then the low-trust action_review
    half up to `min_low`, then fill to `total` from the high-trust half. With `ids` set
    it RE-SELECTS: the pools are filtered to exactly those `inputs_digest`s, no quota
    and no top-up applies, and whatever the store still yields is what comes back —
    rows that rotated out appear in the sidecar's `ids_missing` instead of being
    replaced by a fresh draw. Re-selection is the mode that reproduces a shipped
    `gold.jsonl`, because a shipped sample's window has moved on since it was drawn.

    `window=None` reads the whole store rather than a tail, which is what re-selection
    needs: at ~10k rows/day a fixed 20,000-row tail stops containing a sample's ids in
    about two days.

    This function computes and reports; it never refuses and never writes. What a sample
    may not become a FILE — short of `total`, or a replay that lost a recorded id — is
    refused by `write`, the one surface that can produce a file, so the strata stats and
    `ids_missing` stay readable to a caller that has to report the gap. A caller that
    only wants the numbers (the CLI's `--dry-run`) simply does not call `write`.
    """
    rows, witness = store_witness(store, window)
    pools: dict[str, list[dict]] = {s: [] for s in STRATA}
    for row in rows:
        s = stratum_of(row)
        if s is not None:
            pools[s].append(row)
    wanted = tuple(ids) if ids else None
    if wanted is not None:
        want_set = set(wanted)
        for stratum in pools:
            pools[stratum] = [r for r in pools[stratum]
                              if str(r.get("inputs_digest") or "") in want_set]

    rng = random.Random(seed)
    # Session transcripts are the expensive artifact — one multi-megabyte read per
    # session, and a session holds dozens of reviewed calls — so they are cached for
    # the run and read at most once each.
    ctx: dict[str, Any] = {"sessions": {}, "sessions_dir": sessions_dir or SESSIONS_DIR,
                           "facts_root": facts_root or FACTS_ROOT,
                           "backlog_dir": backlog_dir or BACKLOG_DIR,
                           "canary_file": canary_file or CANARY_ROWS}
    tokens = canary_tokens(ctx["canary_file"])
    taken: dict[str, list[dict]] = {s: [] for s in STRATA}
    stats: dict[str, dict[str, int]] = {}

    def admit(stratum: str, want: int, pool: list[dict]) -> None:
        """Walk the pool in a seeded order, keeping rows that clear both gates.

        Order is `rng.sample` of the pool sorted by `inputs_digest`, so the choice
        cannot depend on the file's line order or on dict iteration: the same store
        slice, the same seed, the same 120 digests. `want=0` means "every admissible
        row", which is the dedupe and entity strata's rule.
        """
        base = sorted(pool, key=lambda r: str(r.get("inputs_digest") or ""))
        order = (rng.sample(base, len(base)) if want and len(base) > want * OVERSAMPLE
                 else list(base))
        st = {"pool": len(pool), "quota_want": want, "attempted": 0, "no_canvas": 0,
              "no_label": 0, "admitted": 0}
        stats[stratum] = st
        for row in order:
            if want and st["admitted"] >= want:
                break
            seam = str(row.get("seam") or "")
            builder = BUILDERS.get(seam)
            if builder is None:
                # `review` lands here: no seam of that name has ever written a row,
                # and no canvas builder exists for it. That is a stratum that
                # contributes 0 rows, not an error — the item's recipe named it, so
                # the run reports it empty rather than raising on it.
                st["no_canvas"] += 1
                continue
            st["attempted"] += 1
            rebuilt = builder(row, ctx)
            if rebuilt is None:
                st["no_canvas"] += 1
                continue
            got = label(row, rebuilt, tokens)
            if got is None:
                st["no_label"] += 1
                continue
            rule_id, evidence = got
            taken[stratum].append(_gold_row(row, rebuilt, rule_id, evidence, stratum))
            st["admitted"] += 1

    # Every ADMISSIBLE row of the small strata first — the item's phrase is "every
    # dedupe/entity row in the last 20,000", and ADMISSIBLE is the word that matters: in
    # the window the shipped sample was drawn from, 40 dedupe rows are in it and none of
    # the 40 records `actual.merged_into`, which is the only thing `rubric.label_dedupe`
    # can read, so every admissible dedupe row is an empty set and the stratum contributes
    # 0. `want=0` is "take what is admissible", never "take until the quota".
    admit("dedupe", 0, pools["dedupe"])
    admit("entity", 0, pools["entity"])
    admit(SEAM_REVIEW, 0, pools[SEAM_REVIEW])
    if wanted is None:
        # Then the seam's distrustful half, capped so it can stand as its own cell.
        low_want = min(min_low, max(0, total - len(taken["dedupe"])
                                   - len(taken["entity"])))
        admit("action_review_low", low_want, pools["action_review_low"])
        # Then the confident majority takes whatever is left.
        room = total - sum(len(v) for v in taken.values())
        admit("action_review_high", max(0, room), pools["action_review_high"])
    else:
        # Re-selection: the pools already hold only the recorded ids, so each stratum's
        # quota IS its recorded count. Capping them at the fresh-draw quotas would
        # truncate a reproduced sample down to 120's arithmetic — 60 of 90 high rows —
        # and print a number that looks like a reproduction.
        admit("action_review_low", len(pools["action_review_low"]),
              pools["action_review_low"])
        admit("action_review_high", len(pools["action_review_high"]),
              pools["action_review_high"])

    # Strata in fill order, rows within a stratum by `inputs_digest`. The SELECTION is
    # the seeded shuffle (`_take`); this is presentation, and it exists so that the
    # recorded-ids replay emits the identical BYTES and not merely the identical SET.
    # A fresh draw walks a stratum in `rng.sample` order and a replay walks it in
    # recorded order, so without this the two agreed on all 120 digests while differing
    # at 70 line positions — enough to make "re-run it and diff the file" fail on a
    # correct reproduction, which is the check the item's own verification step names.
    gold = [g for s in STRATA
            for g in sorted(taken[s], key=lambda r: str(r["inputs_digest"]))]
    # A stratum whose rows rotated out of the store simply comes back short, and the
    # ids that went missing are named. Nothing backfills them: a sample that quietly
    # replaced a missing id with a fresh draw would print `rows: 120` while reproducing
    # nothing, which is the one outcome this mode exists to make impossible.
    ids_missing = (list(dict.fromkeys(
        i for i in wanted
        if i not in {g["inputs_digest"] for g in gold}))
        if wanted is not None else None)
    labelled_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for g in gold:
        g["labelled_at"] = labelled_at
        for req in rubric.REQUIRED_FIELDS:
            if not g.get(req):
                raise SystemExit(f"internal: gold row {g.get('inputs_digest')} came "
                                 f"out without {req!r}")
        rubric.validate_gold_row(g)
    side = {
        "seed": seed, "total_requested": total, "window_rows": window,
        "label_mass_split": LABEL_MASS_SPLIT, "min_low": min_low,
        "rubric_sha256": rubric.rubric_sha256(), "labelled_by": rubric.LABELLED_BY,
        "labelled_at": labelled_at, "store": witness,
        "strata": {s: dict(stats.get(s, {})) for s in STRATA},
        "inputs_digests": [g["inputs_digest"] for g in gold],
        "seams": {s: sum(1 for g in gold if g["seam"] == s)
                  for s in (SEAM_ACTION_REVIEW, SEAM_DEDUPE, SEAM_ENTITY)},
        "gold_labels": {lb: sum(1 for g in gold if g["gold"] == lb)
                        for lb in sorted({g["gold"] for g in gold})},
        # Re-selection mode only; both null in the fresh-draw mode that ships a sample.
        # `ids_missing` is the difference between the ids asked for and the ids that
        # came back, so a re-run against a rotated store reports the gap instead of
        # papering over it with a fresh draw.
        "reselect_requested": list(wanted) if wanted is not None else None,
        "ids_missing": ids_missing,
    }
    return gold, side


def _gold_row(row: dict, rebuilt: dict, rule_id: str, evidence: str,
              stratum: str) -> dict:
    return {
        "inputs_digest": str(row.get("inputs_digest") or ""),
        "seam": str(row.get("seam") or ""),
        "schema_hash": str(row.get("schema") or ""),
        "stratum": stratum,
        "question_id": rebuilt["question_id"],
        "question": rebuilt["instructions"],
        "options": rebuilt["options"],
        "state": rebuilt["state"],
        "asked_from_canvas": rebuilt["asked_from_canvas"],
        "gold": rubric.RULES[rule_id]["label"],
        "warrant": rubric.build_warrant(rule_id, evidence),
        "labelled_by": rubric.LABELLED_BY,
        "rubric_sha256": rubric.rubric_sha256(),
        "recorded_ts": _iso(row.get("ts")),
        "recorded_meta": {k: v for k, v in (row.get("meta") or {}).items()
                          if k in ("session_id", "source", "tool", "a", "b", "name")},
        "provenance": _provenance(row, rebuilt),
    }


def _provenance(row: dict, rebuilt: dict) -> dict:
    out = {"canvas_verified_by": "app.djev_shadow._digest over [state, questions], "
                                 "compared to the row's own inputs_digest"}
    if "session_file" in rebuilt:
        out["session_file"] = rebuilt["session_file"]
    if "item_file" in rebuilt:
        out["item_file"] = rebuilt["item_file"]
    if "facts_root" in rebuilt:
        out["facts_root"] = rebuilt["facts_root"]
    merged = (row.get("actual") or {}).get("merged_into")
    if merged is not None:
        out["actual_merged_into"] = merged
    return out


def write_refusal(gold: list[dict], side: dict) -> str | None:
    """Why this sample may not become a file, or None when it may.

    Both checks are about the FILE rather than the draw, which is why they sit here and
    not in `build_sample`: `build_sample` also serves a caller that only has to report
    a gap (a replay against a store that lost an id prints `ids_missing` and exits
    non-zero), and refusing inside it made that report unreachable — the call raised
    before the caller could read the sidecar. `write` is the only call that can put a
    gold file on disk, so refusing here refuses on every path that ships one, including
    `main`. `tests/test_cross_engine_arm1_gold.py` pins both branches.
    """
    wanted = side.get("reselect_requested")
    if wanted is not None:
        missing = side.get("ids_missing") or []
        if missing:
            return (f"re-selection reproduced {len(gold)} of {len(wanted)} recorded "
                    f"ids; missing {len(missing)}: {missing[:5]}. Nothing was "
                    "written: a gold file with substitutions in it is not the file "
                    "this sampler shipped.")
        return None
    total = int(side.get("total_requested") or 0)
    if len(gold) != total:
        return (f"only {len(gold)} rows in the store slice cleared both admission "
                f"gates, need exactly {total}; per-stratum: "
                f"{json.dumps({s: (side.get('strata') or {}).get(s) for s in STRATA})}"
                ". The sample is not written short: a gold file that is not the size "
                "the surface was designed for would silently move every denominator.")
    return None


def write(gold: list[dict], side: dict, gold_path: Path = GOLD_PATH,
          sample_path: Path = SAMPLE_PATH) -> None:
    refused = write_refusal(gold, side)
    if refused:
        raise SystemExit(refused)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    with gold_path.open("w", encoding="utf-8") as fh:
        for g in gold:
            fh.write(json.dumps(g, ensure_ascii=False, sort_keys=True) + "\n")
    with sample_path.open("w", encoding="utf-8") as fh:
        json.dump(side, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--store", type=Path, default=SHADOW_LOG)
    ap.add_argument("--window", type=int, default=None,
                    help="tail rows of the store to sample from; 0 reads the whole "
                         f"store (default {DEFAULT_WINDOW} for a fresh draw, 0 with "
                         "--ids-from)")
    ap.add_argument("--ids-from", type=Path, default=None,
                    help="re-select exactly the `inputs_digests` recorded in a "
                         "previous run's sidecar instead of drawing a fresh sample. "
                         "This is what re-derives a shipped gold.jsonl: the store grows "
                         "~10k rows a day, so the window a shipped sample was drawn "
                         "from is not the window a fresh run would see. Implies "
                         "--window 0. Exits non-zero, writing nothing, if any recorded "
                         "id is gone from the store.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--total", type=int, default=TOTAL)
    ap.add_argument("--min-low", type=int, default=MIN_LOW)
    ap.add_argument("--out", type=Path, default=GOLD_PATH)
    ap.add_argument("--sidecar", type=Path, default=SAMPLE_PATH)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the strata stats and write nothing")
    args = ap.parse_args(argv)
    wanted = None
    if args.ids_from:
        recorded = json.loads(args.ids_from.read_text(encoding="utf-8"))
        wanted = tuple(recorded.get("inputs_digests") or ())
        if not wanted:
            raise SystemExit(f"--ids-from {args.ids_from}: no inputs_digests recorded")
    # `0` and "unset while re-selecting" both mean the whole store; a fresh draw keeps
    # the bounded tail, which is what the shipped sample was drawn from.
    window = args.window
    if window is None:
        window = None if wanted is not None else DEFAULT_WINDOW
    elif window == 0:
        window = None
    gold, side = build_sample(seed=args.seed, window=window,
                              store=args.store, total=args.total, min_low=args.min_low,
                              ids=wanted)
    print(json.dumps({k: side[k] for k in ("seed", "store", "strata", "seams",
                                           "gold_labels", "rubric_sha256")},
                     indent=2, sort_keys=True))
    if side.get("ids_missing"):
        print(f"ids missing from the store: {len(side['ids_missing'])}")
    print(f"rows: {len(gold)}")
    if args.dry_run:
        return 0
    write(gold, side, gold_path=args.out, sample_path=args.sidecar)
    print(f"wrote {args.out} ({len(gold)} rows) and {args.sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
