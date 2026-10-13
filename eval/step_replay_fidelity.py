"""Reconstruction fidelity for recorded worker steps — the #2636 step-1 gate, measured.

What this answers
-----------------
Item #2636 proposes replaying one recorded LLM step (the outgoing messages array
as it stood before one model call) against a prompt variant, so a SKILL.md edit
can be A/B-ed without burning a whole worker run. Its own step 1 is a gate in
front of all of that: *"Reconstruct the messages array as it stood before the
note-writing call from `~/lloyd-data/sessions/*.json` and check byte-level
fidelity ... If the reconstructed prompt cannot be shown to match what was sent,
stop and record that — the item becomes a capture feature, not an eval."*

That gate has been an assumption everywhere it was quoted, because showing the
match needs a number nobody had computed. This module computes it. For each
recorded step of a recurring source it answers one question with a comparison,
not a claim: **can the bytes that are on disk right now rebuild an outgoing
messages array that hashes to what the request manifest says was sent?**

Why the manifest is the only possible yardstick
-----------------------------------------------
`app/component_manifest.py` (#581) writes one NDJSON line per outgoing request
holding `sha256:` plus `bytes` for every part of it — the tools array, each
`prompt_builder` component by dict key, the prefetch block, and **each message in
order** (`_build_line`'s `message_rows`, over `canonical_json(message)` of the
payload's own dict). It is digests and sizes only: there is deliberately no blob
store and no `rebuild_request`, because retaining the rendered prompt — vault
text, email bodies, user messages — was judged "a strictly larger
confidentiality problem than a diff needs" and its one would-be consumer (#564)
was retired. So the store can tell you exactly what a request hashed to and
nothing about what it said, which makes it a perfect *oracle* and a useless
*source*. This probe therefore takes the bytes from the session JSON and the
answer from the manifest, and reports where the two disagree.

What it measured on the step triage quoted
------------------------------------------
`20261012_163343_youtubediges_ad09` step 1, manifest line
`r-20261012T233353-bd4704268b` (day file `2026-10-12.ndjson`, re-read for this
sentence): 2 messages, message 0 `role=system` at 45,080 bytes, tools array
27,936 bytes over 22 definitions, and four components — `SOUL.md` 6,994,
`memories` 20,494, `skills_index` 12,456, `harness_hints` 4,516. Neither the
45,080-byte system prompt nor the 27,936-byte tools array is anywhere in the
session file, and `probe` reports the step `mismatched` at message index 0 with
first moved component `skills_index` (12,456 recorded against 12,443 rebuilt)
while `SOUL.md` and `memories` still hash equal to the recorded request — the
session's platform is `worker`, and `harness.worker_prompt.drop_memory_files`
makes that block `MEMORY.md` without `USER.md`, which is why the rebuild is asked
for the session's own platform and not a default. So for
a real recorded step the fidelity answer is "no, not byte-exact", and it is a
named block that moved rather than an unexplained difference — which is the
distinction step 1 asked for.

Per source, `--today 2026-10-13`, components rebuilt from the live vault:
`report --source youtube-digest --max-sessions 8` → 8 sessions, 78 recorded steps
inside the 14-day window, **0 matched**, 78 mismatched, 2.34 s wall;
`report --source autocode --max-sessions 4` → 4 sessions, 532 steps, **0 matched**,
3.66 s. Not one step in either source rebuilds byte-exact, so the item's step-1
fork applies and it is a capture feature until the rendered payload is retained.
The counts grow between readings because these are live sessions — the autocode
figure was 295 an hour earlier — which is why `report` prints the window it was
measured in and `--today` pins one.

And the gap is not only the system slot. Across the witness session's day file,
**0 of 302 non-system message rows** hash equal to the reconstruction this
module's rules produce (`WIRE_KEYS` over `is_history_row`): the witness step's own
user row is 32,974 recorded bytes against 29,972 rebuilt, 3,002 bytes short. So the
capture feature step 1 points at cannot be "keep the system prompt" — it has to be
"keep the outgoing `messages` array", which is the confidentiality question the item
already defers rather than answers. Pinned by
`tests/test_step_replay_fidelity.py::test_no_history_row_of_the_witness_session_round_trips`.

Which component is named depends on which vault the rebuild could read, and the
probe prints that path rather than hiding it: with the live vault visible the
first moved component is `skills_index`; run from an automod round, whose sibling
`obsidian/` does not exist, it is `SOUL.md`, because the identity block is absent
rather than changed. Both are `mismatched` at message index 0 either way, and the
fidelity answer does not turn on the difference.

What the reconstruction is, and what it is not
----------------------------------------------
The reconstruction is the honest best effort of a would-be replayer holding only
committed artifacts, and it is documented per rule in `reconstruct_messages`
below. It is deliberately NOT the production turn-start stack:
`app.compaction.load_and_compact_session` runs a microcompact pre-pass and, over
threshold, an LLM summarisation — reproducing a step through it would mean
issuing model requests to find out whether one could be avoided, and its
summarise layer is not byte-reproducible anyway. So the probe rebuilds:

* the conversation messages, from `app.compaction.is_history_row` (the one
  definition of a row the turn-start stack keeps) reduced to `WIRE_KEYS`;
* a system message **only when the manifest line's message[0] is `role=system`**,
  because `app/harness/loop.py` prepends one and a replay has to send something
  there — and the only text available to put in it is the live rebuild from
  `app.prompt_builder.build_system_prompt`, since the session JSON stores no
  system row at all.

A `matched` verdict therefore means something strong: the stored rows, reduced
and ordered by these rules, hash equal to the request that actually went out.
A `mismatched` verdict is not a bare failure: the report names the first
component in the manifest's own order whose digest moved between the recorded
request and the live rebuild (`first_moved_component`), because "the prompt did
not rebuild" is only actionable as "THIS part of the prompt did not rebuild".

Strictly read-only
------------------
Nothing here issues a request or writes a byte, and the two halves are proven
differently because one of them is not what it looks like.

No import STATEMENT in this file names an engine module — no `app.harness.client`,
no `httpx`, no `openai` — and the closure walk in `tests/test_step_replay_fidelity.py`
proves which loader paths exist at all. It also proves the limit of that claim: the
mandated component rebuild calls `build_system_prompt`, which resolves
`app.harness.rpc_policy`, so CPython executes `app/harness/__init__.py` ->
`run_query` -> `app.harness.client`, whose own module level imports `httpx`. So the
engine client IS present in this process, loaded by the package initialiser, on a
path the clause requires. What is true, and what is pinned, is that nothing on it is
called: `stream_chat` is never referenced, `component_manifest.record_request` is
never called (grep it; the assertion counts call syntax), so the writer thread that
owns the manifest store is never started, and a probe run under a `sys.audit` hook
opens no socket, builds no HTTP request, spawns no child, and still prints its
verdict.

Write-identity is pinned by hashing every file under the sessions, event-log and
request-manifest trees before and after a probe and a report run, with the run
driven through the REAL component rebuild rather than a pinned one — the rebuild is
the only call in here that reaches another subsystem's state, so a byte-identity
result obtained without it would be proving an easier run than the probe performs.

Cost, since the item asks for it rather than assuming it
--------------------------------------------------------
The manifest is ~330 MB of NDJSON per day file, so a report reads each dated
file it needs exactly ONCE and keeps only the lines whose `session_id` is one it
was asked about (`_lines_by_session`, one pass per file for the whole corpus).
Sessions outside the retention window are never opened for manifest work at all:
their day file is gone, which is the whole measurement of that bucket. No model
request is issued by any code path here, so the cost of a report is wall-clock
disk read and no tokens; `report` prints seconds and sessions scanned.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import re
import sys

HERE = Path(__file__).resolve().parent
LLOYD_HOME = HERE.parent
sys.path.insert(0, str(LLOYD_HOME))

from app import component_manifest  # noqa: E402
from app.compaction import is_history_row  # noqa: E402
from app.data_root import ACCOUNT_HOME, production_data_root  # noqa: E402

#: Key this probe hands `build_system_prompt` as its `session_id` so the live
#: component rebuild reaches the #581 registry. Real session ids are
#: `<YYYYMMDD>_<HHMMSS>_<source>_<hex>`, so this cannot collide with one, and a
#: rebuild must never overwrite what a live session handed the registry.
PROBE_REGISTRY_KEY = "step-replay-fidelity-probe"

#: The keys of an OpenAI message dict that a persisted row can supply. A stored
#: row carries `id`, `timestamp`, `turn_id`, `stats`, `reasoning` and
#: `subliminal` on top of what goes on the wire; a replay that re-sent those
#: would not be reproducing the request, so the candidate drops them.
WIRE_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name")

#: `<YYYY-MM-DD>.ndjson` — the store's dated-file shape, spelled here rather than
#: reached through `component_manifest._DAY_FILE_RE` because that name is private.
#: The cost of the copy is drift, so `tests/test_step_replay_fidelity.py` pins
#: this pattern against the writer's: a store that starts naming its files
#: another way fails that test instead of quietly making every step of every
#: corpus read as "outside the retention window".
_DAY_FILE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.ndjson$")

#: One model decision per assistant row. Used only to count steps that CANNOT be
#: enumerated from the manifest because the session's day file has been pruned;
#: a step that is still on disk is always counted from its manifest line.
_ASSISTANT_ROLES = ("assistant",)


# ── the two sides of the comparison ───────────────────────────────────────


def message_digests(messages: Iterable[dict[str, Any]]) -> list[str]:
    """Per-message `sha256:` over `canonical_json(message)`, in order.

    This is `_build_line`'s own recipe, called here on the reconstruction so the
    two sides of the comparison are the same function of their input. Reusing
    `component_manifest.canonical_json`/`digest_text` is the point: a probe that
    serialised its own way would report a mismatch that only exists in the probe.
    """
    return [component_manifest.digest_text(component_manifest.canonical_json(m))
            for m in messages]


def message_digests_from_line(line: dict[str, Any]) -> list[str]:
    """The recorded per-message digests of a manifest line, in index order."""
    rows = sorted(line.get("messages") or [], key=lambda r: r.get("index", 0))
    return [str(r.get("sha256", "")) for r in rows]


def array_sha256(digests: list[str]) -> str:
    """One digest for a whole messages array: the digest of its ordered digests.

    The comparison the item's step-1 gate asks for is per step, so per step there
    is exactly one number on each side to quote. Hashing the ordered list is a
    positional comparison in one value: two arrays that agree message-for-message
    in order hash equal, and one reordered message moves it.
    """
    return component_manifest.digest_obj(digests)


# ── reading the session side ──────────────────────────────────────────────


def sessions_dir() -> Path:
    """Where the session JSON a probe reads lives: the live data root, always.

    `production_data_root()`, not `app.paths.SESSIONS_DIR`, because this is a
    reader and `app.paths` is anchored to the tree that imported it — inside an
    automod worktree that is `<round>/home/lloyd/.lloyd-data/sessions`, which
    `app.paths` itself refuses for exactly this reason ("scanning them is a
    measurement of nothing, not a clean result"). A fidelity probe run from a
    round has to measure the sessions that are actually on the box, or the number
    it prints is a zero that reads like a result.
    """
    return production_data_root() / "sessions"


def manifest_store() -> Path:
    """The request-manifest store root, anchored on the account, not on `$HOME`.

    Same reader rule as `sessions_dir()`: inside a round `$HOME` is the round-home
    symlink farm, so the writer's `store_root()` — which falls back to
    `Path.home()` — would point at a directory that does not exist and every step
    would come back `no_manifest_line`. `$LLOYD_MANIFEST_STORE` still wins, which
    is both the writer's own first override and the hook a test uses to point the
    probe at a fixture store; the default reuses
    `component_manifest.DEFAULT_SUBDIR`, so a rename of the directory moves this
    resolution with it rather than leaving the probe reading a stale name.
    """
    override = (os.environ.get("LLOYD_MANIFEST_STORE")
                or str(component_manifest._cfg().get("store_dir") or ""))
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_STATE_HOME") or ""
    state = Path(xdg).expanduser() if xdg else (ACCOUNT_HOME / ".local" / "state")
    return state / component_manifest.DEFAULT_SUBDIR


def _session_path(session_id: str, root: Path) -> Path | None:
    """The session file, plain or gzipped, or None when there is none."""
    for candidate in (root / f"{session_id}.json", root / f"{session_id}.json.gz"):
        if candidate.is_file():
            return candidate
    return None


def read_session(session_id: str, *, root: "Path | None" = None) -> dict[str, Any] | None:
    """One session file as a dict, or None. Read-only, and never raises.

    Gzipped files are read with `gzip` because the retention sweep compresses
    sessions as they age, and a probe that silently skipped them would report
    "no recorded steps" for exactly the corpus a long-window eval would want.
    """
    path = _session_path(session_id, root or sessions_dir())
    if path is None:
        return None
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
                return json.load(fh)
        with open(path, encoding="utf-8", errors="replace") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def source_of(session_id: str) -> str:
    """The recurring source a session belongs to, from `component_manifest`.

    The same four-underscore-parts rule the manifest line itself uses, so a
    report's `--source` and the `source` field on the line it scores cannot
    disagree about which job a session belongs to.
    """
    return component_manifest.source_of_session(session_id)


def source_matches(token: str, requested: str) -> bool:
    """Whether a session id's source token is the source `report --source` named.

    Two spellings have to work, and the gap between them is the id scheme, not a
    choice here: `source_of_session` returns the third underscore-part of the id,
    which `sessions_io` truncated to 12 characters — which is why the task file
    `youtube-digest.md` produces the token `youtubediges` while the manifest's own
    corpus map keys it as `youtube-digest`. Matching on a normalised prefix covers
    both without hard-coding either list: `youtubedigest`.startswith(`youtubediges`).
    """
    if not token:
        return False
    normalised = re.sub(r"[^a-z0-9]", "", (requested or "").lower())
    return bool(normalised) and (normalised == token or normalised.startswith(token))


def source_sessions(source: str, *, root: "Path | None" = None,
                    limit: int = 25) -> list[str]:
    """The newest `limit` session ids whose source is `source`.

    Newest first, by name: session ids begin with a UTC timestamp, so the name
    sort IS the time sort and does not depend on mtimes a restore may have
    rewritten. `limit` bounds the work the same way the item bounds its own
    corpus — an unbounded sweep over the sessions directory is the "always-on
    nightly cost" this instrument is told not to become.
    """
    root = Path(root) if root else sessions_dir()
    try:
        names = sorted([p.name for p in root.iterdir()
                        if p.suffix in (".json", ".gz")], reverse=True)
    except OSError:
        return []
    out: list[str] = []
    for name in names:
        session_id = name[:-3] if name.endswith(".gz") else name
        session_id = session_id[:-5] if session_id.endswith(".json") else session_id
        if not source_matches(source_of(session_id), source):
            continue
        out.append(session_id)
        if len(out) >= max(0, limit):
            break
    return out


def utc_day_of(session: dict[str, Any]) -> str:
    """The UTC date a session began, as `YYYY-MM-DD`.

    `created_at` is stored with the box's own offset (`2026-10-12T16:33:43-07:00`),
    which is local wall clock; the manifest store's day files are named in UTC. A
    naive parse here would ask the 23:33-UTC request for the file named after the
    16:33 it looks like, and report a pruned file where there is a live one — the
    timestamp-in-a-machine-facing-payload failure #1668 names, in the one place
    this item could have made it silently understate its own corpus.
    """
    raw = str(session.get("created_at") or "")
    try:
        stamp = datetime.fromisoformat(raw)
    except ValueError:
        return ""
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).date().isoformat()


# ── reading the manifest side ─────────────────────────────────────────────


def day_files(store: "Path | None" = None) -> "dict[str, Path]":
    """`{YYYY-MM-DD: path}` for every dated file present in the store."""
    manifests = (Path(store) if store else manifest_store()) / "manifests"
    out: "dict[str, Path]" = {}
    try:
        for path in manifests.iterdir():
            match = _DAY_FILE_RE.match(path.name)
            if match and path.is_file():
                out[match.group(1)] = path
    except OSError:
        return {}
    return out


def retention_window(store: "Path | None" = None, *,
                     today: "str | None" = None) -> "tuple[str, str]":
    """`(cutoff_day, newest_day)` of files `prune_store` would still keep.

    Computed the way the writer computes it — `today - (retention_days - 1)`, so
    a 14-day window holds 14 dated files including today's — because "outside the
    retention window" has to mean what the sweep means, or the report counts its
    corpus against a window the store does not implement.
    """
    days = component_manifest.retention_days()
    newest = today or datetime.now(timezone.utc).date().isoformat()
    try:
        anchor = datetime.strptime(newest, "%Y-%m-%d").date()
    except ValueError:
        anchor = datetime.now(timezone.utc).date()
        newest = anchor.isoformat()
    cutoff = anchor - timedelta(days=max(days, 1) - 1)
    return cutoff.isoformat(), newest


def _lines_by_session(session_ids: "set[str]", path: Path) -> "dict[str, list[dict[str, Any]]]":
    """Every manifest line in one day file whose session is wanted, grouped by session.

    One pass per file for the whole corpus: a day file is ~330 MB of NDJSON, so a
    report that re-opened it per session would read 330 MB times the corpus size
    to answer a question it can answer in one read.
    """
    out: "dict[str, list[dict[str, Any]]]" = {sid: [] for sid in session_ids}
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    line = json.loads(raw)
                except ValueError:
                    continue
                sid = line.get("session_id")
                if sid in out:
                    out[sid].append(line)
    except OSError:
        return {}
    return out


def step_label(line: dict[str, Any]) -> str:
    """What a step is called: its `iteration`, or `finalizer` when it has none.

    `iteration` counts the loop's own requests, and the one request a turn makes
    after the loop — `app/harness/finalizer.py::run_finalizer` — records
    `iteration: null`. That is a model decision point like any other, and dropping
    it would quietly shrink the corpus by one replayable step per turn.
    """
    iteration = line.get("iteration")
    return "finalizer" if iteration is None else str(iteration)


def _line_for_step(lines: list[dict[str, Any]], step: str) -> dict[str, Any] | None:
    """The recorded line for `step`, first wins when several share a label."""
    for line in lines:
        if step_label(line) == str(step):
            return line
    return None


# ── the reconstruction ────────────────────────────────────────────────────


def live_component_digests(platform: str = "") -> "dict[str, dict[str, Any]]":
    """`{name: {sha256, bytes}}` for the components the tree builds RIGHT NOW.

    Goes through the real builder rather than re-reading SOUL.md, MEMORY.md and
    `skills/` here, because the question is not whether a file changed but
    whether the *component* the model was handed changes — and a component is
    what `build_system_prompt` renders, wrappers and all (`<memory>…</memory>`,
    the skills-index note, the harness hints' joined paragraphs). Two copies of
    that rendering would disagree, and the disagreement would be reported as a
    prompt change.

    What the rebuild cannot carry is stated rather than hidden: `goal`, `plan`
    and `todos` blocks come from session state the probe does not have, and a
    worker turn's `memories` may have been a frozen snapshot
    (`app/memory_snapshot.py`) rather than the live files. Those components then
    read as moved, which is the true answer to "could a replay re-send it" and
    the reason this function's output is only ever compared, never trusted alone.
    """
    return _rebuild(platform)["components"]


def first_moved_component(line: dict[str, Any],
                         rebuilt: "dict[str, dict[str, Any]]") -> str:
    """The first component in the manifest's own order whose digest moved.

    Order matters: the manifest lists components in render order, so "the first
    one that moved" is the first thing a replay would have to fix, and it is also
    the answer to the question #520's prefix-miss reporting cannot answer alone.
    A component the rebuild did not produce at all counts as moved — an absent
    block is not the same bytes as a present one, whatever its prose says.
    Returns `""` when every recorded component hashes equal, which is the case
    that says the mismatch is in the conversation, not the system prompt.
    """
    for row in line.get("components") or []:
        name = str(row.get("name", ""))
        pair = rebuilt.get(name)
        if pair is None or str(pair.get("sha256", "")) != str(row.get("sha256", "")):
            return name
    return ""


def reconstruct_messages(session: dict[str, Any], line: dict[str, Any], *,
                         system_prompt: "str | None" = None) -> list[dict[str, Any]]:
    """The probe's candidate for the messages array `line` describes.

    Rule 1 — the conversation. `app.compaction.is_history_row` is the turn-start
    stack's own definition of a row that re-enters the prompt, so the candidate
    is built from the same rows the production reader keeps, in stored order,
    reduced to `WIRE_KEYS`. A row the session stores under another role —
    `subliminal`, `thinking` — is not a message the engine received under that
    role, and this is exactly where fidelity is expected to break: turn-scoped
    injections ride in on the user message at send time and come back as their
    own rows, so the stored user row is smaller than the one that went out.

    Rule 2 — how many. `line` recorded `N` messages, so the candidate takes the
    first `N - 1` history rows after the system slot. A truncated history is
    still a candidate: the comparison, not this slice, decides whether it is the
    right one.

    Rule 3 — the system slot. `system_prompt` overrides the text in it; with no
    override every production call has nothing else to put there. `run_query` prepends
    `{"role": "system", "content": options.system_prompt}` as message 0, and the
    session JSON stores no system row at all. So the candidate puts the live
    rebuild there when the manifest line's message 0 really is `role=system`, and
    puts nothing there when it is not. `system_prompt=None` means "use the live
    rebuild"; the caller may pass text (a test, or a frozen snapshot a future
    capture feature would read) to pin that slot.
    """
    recorded = line.get("messages") or []
    wants_system = bool(recorded) and str(recorded[0].get("role")) == "system"
    rows = [r for r in (session.get("messages") or []) if is_history_row(r)]
    take = len(recorded) - (1 if wants_system else 0)
    out: list[dict[str, Any]] = []
    if wants_system:
        text = system_prompt
        if text is None:
            text = _live_system_prompt(str(session.get("platform") or ""))
        out.append({"role": "system", "content": text})
    for row in rows[:max(take, 0)]:
        out.append({k: row[k] for k in WIRE_KEYS if k in row})
    return out


def _live_system_prompt(platform: str) -> str:
    """The system prompt the current tree would build for this platform.

    Kept separate from `live_component_digests` because it needs the rendered
    string and that one needs the dict; both rebuild once per process (see
    `_REBUILD`), so a report over 25 sessions does not read the vault 25 times.
    """
    return _rebuild(platform)["prompt"]


_REBUILD: "dict[str, dict[str, Any]]" = {}


def _rebuild(platform: str) -> "dict[str, Any]":
    """Memoised live rebuild: `{"prompt": str, "components": {name: pair}}`."""
    cached = _REBUILD.get(platform)
    if cached is not None:
        return cached
    from app import prompt_builder
    prompt = prompt_builder.build_system_prompt(session_id=PROBE_REGISTRY_KEY,
                                                platform=platform)
    entry = component_manifest.components_for(PROBE_REGISTRY_KEY)
    # The identity files are read from `LLOYD_HOME.parent / "obsidian"`, and
    # `LLOYD_HOME` is derived from the tree that imported `prompt_builder` — so a
    # probe run inside an automod round reads `<round>/home/obsidian`, where
    # SOUL.md, MEMORY.md and `skills/` are not the shipped ones and every identity
    # component can report as "moved" when nothing about the shipped prompt
    # changed. The record therefore carries the vault it read and `format_probe`
    # prints it: an instrument whose mismatch depends on which tree it was invoked
    # from has to say so, or its number is unreadable — the anchored-tree trap
    # `app.paths.describe_tree()` warns about, in the one place this probe can
    # still fall into it.
    from app import prompt_builder as _pb
    built = {"prompt": prompt,
             "vault_root": str(_pb.LLOYD_HOME.parent / "obsidian"),
             "components": dict(entry.get("components") or {})}
    _REBUILD[platform] = built
    return built


# ── the probe ─────────────────────────────────────────────────────────────


def probe_step(session_id: str, step: str, *,
               root: "Path | None" = None,
               store: "Path | None" = None,
               rebuilt: "dict[str, dict[str, Any]] | None" = None,
               system_prompt: "str | None" = None,
               today: "str | None" = None
               ) -> dict[str, Any]:
    """Compare one recorded step with what the bytes on disk can rebuild.

    Returns one record per step: `verdict` (`matched`, `mismatched`,
    `no_session`, `no_manifest_line`), `request_id` — the manifest line the step
    was compared against, which is what makes the number attributable rather than
    anecdotal — and the two `array_sha256` values that were compared, so a reader
    can re-run the hash and see which side moved. `moved_component` carries the
    first component in the recorded order whose digest differs from the live
    rebuild, `""` when none does.

    `rebuilt` lets a caller supply the component digests instead of rebuilding
    them from the vault: a test needs that to make a moved component
    deterministic, and nothing else in here reads the vault.

    `today` anchors the retention window the `outside_window` flag is measured
    against. A measurement quoted in a note has to stay checkable after the
    sweep pruned the day files it was taken against, and the same rule applies to
    a test: without an anchor, the fixture's own dates decide when it starts
    failing.
    """
    started = time.perf_counter()
    record: dict[str, Any] = {
        "session_id": session_id,
        "source": source_of(session_id),
        "step": str(step),
        "request_id": "",
        "verdict": "",
        "recorded_sha256": "",
        "reconstructed_sha256": "",
        "recorded_messages": 0,
        "reconstructed_messages": 0,
        "first_diff_index": None,
        "moved_component": "",
        "day": "",
        "vault_read": "",
        "seconds": 0.0,
    }
    session = read_session(session_id, root=root)
    if session is None:
        record["verdict"] = "no_session"
        record["seconds"] = round(time.perf_counter() - started, 4)
        return record
    record["day"] = utc_day_of(session)
    store_root = Path(store) if store else manifest_store()
    files = day_files(store_root)
    day_path = files.get(record["day"])
    if day_path is None:
        # The day file is gone (pruned, or predates #581), so the step has no
        # oracle: nothing here can call that a match or a miss.
        record["verdict"] = "no_manifest_line"
        record["outside_window"] = (
            record["day"] < retention_window(store_root, today=today)[0])
        record["seconds"] = round(time.perf_counter() - started, 4)
        return record
    lines = _lines_by_session({session_id}, day_path).get(session_id) or []
    line = _line_for_step(lines, str(step))
    if line is None:
        record["verdict"] = "no_manifest_line"
        record["available_steps"] = sorted(step_label(l) for l in lines)
        record["seconds"] = round(time.perf_counter() - started, 4)
        return record

    record["request_id"] = str(line.get("request_id", ""))
    record["recorded_messages"] = len(line.get("messages") or [])
    if rebuilt is None:
        full = _rebuild(str(session.get("platform") or ""))
        rebuilt = full["components"]
        record["vault_read"] = full["vault_root"]
    # Named after the branch that guarantees it is not the `None` the parameter
    # allows, rather than asserting the parameter's own name is narrowed: `_rebuild`
    # is monkeypatched across this module's tests, so its return is only ever as
    # tight as its own annotation, and a re-assignment back to `rebuilt` left pyright
    # reading line 646's `components` as possibly-None.
    components: "dict[str, dict[str, Any]]" = (
        rebuilt if rebuilt is not None else _rebuild(
            str(session.get("platform") or ""))["components"])
    if record["recorded_messages"] == 0:
        # Nothing recorded means nothing to compare; say so rather than reporting
        # two empty arrays as an agreement.
        record["verdict"] = "mismatched"
        record["reason"] = "the manifest line recorded no messages to compare"
        record["moved_component"] = first_moved_component(line, components)
        record["seconds"] = round(time.perf_counter() - started, 4)
        return record

    messages = reconstruct_messages(session, line, system_prompt=system_prompt)
    reconstructed = message_digests(messages)
    recorded = message_digests_from_line(line)
    record["reconstructed_messages"] = len(reconstructed)
    record["recorded_sha256"] = array_sha256(recorded)
    record["reconstructed_sha256"] = array_sha256(reconstructed)
    if record["verdict"] == "mismatched":
        record["vault_read"] = record.get("vault_read") or _vault_read()
    if record["recorded_sha256"] == record["reconstructed_sha256"]:
        record["verdict"] = "matched"
    else:
        record["verdict"] = "mismatched"
        record["first_diff_index"] = next(
            (i for i, (a, b) in enumerate(zip(recorded, reconstructed)) if a != b),
            None if len(recorded) == len(reconstructed) else min(len(recorded),
                                                                 len(reconstructed)))
        record["moved_component"] = first_moved_component(line, components)
    record["seconds"] = round(time.perf_counter() - started, 4)
    return record


def _vault_read() -> str:
    """The vault the memoised rebuild read, for a record that did not rebuild."""
    for built in _REBUILD.values():
        return str(built.get("vault_root", ""))
    return ""


def format_probe(record: dict[str, Any]) -> str:
    """One step, printed. The request id and both digests are always in it.

    A verdict with no numbers beside it is the thing #2636's whole step-1 gate
    exists to stop, so no branch of this printer is allowed to omit the digests
    it compared or the line it compared them against.
    """
    head = (f"{record['session_id']} step={record['step']} "
            f"verdict={record['verdict'] or 'unset'}")
    if record["verdict"] in ("matched", "mismatched"):
        return (f"{head}\n"
                f"  compared against manifest request_id={record['request_id']}"
                f" (day {record['day']})\n"
                f"  recorded      {record['recorded_sha256']}"
                f"  [{record['recorded_messages']} messages]\n"
                f"  reconstructed {record['reconstructed_sha256']}"
                f"  [{record['reconstructed_messages']} messages]\n"
                + _moved_line(record))
    detail = f"  day file for {record['day'] or 'an unreadable created_at'}: "
    if record["verdict"] == "no_manifest_line":
        detail += ("absent from the store" if not record.get("available_steps")
                   else "no line for this step; recorded steps: "
                        + ", ".join(record["available_steps"]))
    else:
        detail = f"  no session file for {record['session_id']}"
    return f"{head}\n{detail}"


def _moved_line(record: dict[str, Any]) -> str:
    """Why a mismatch is blamed on a component, or on the conversation."""
    where = (f"first differing message index {record['first_diff_index']}"
             if record.get("first_diff_index") is not None else "message count moved")
    name = record.get("moved_component") or ""
    vault = f" (component rebuild read vault {record.get('vault_read') or 'not rebuilt here'})"
    if name:
        return (f"  mismatch: {where}; first component whose hash moved: "
                f"{name}{vault}")
    return (f"  mismatch: {where}; every recorded component hashes equal to the "
            f"live rebuild, so the movement is in the conversation{vault}")


# ── the report ────────────────────────────────────────────────────────────


def _tokens_on_disk(root: Path, cap: int = 40) -> "list[str]":
    """Distinct source tokens in the sessions directory, for an empty report.

    Read off the file names, not off the matched set: a report that matched
    nothing has no matched set to describe, and `steps_matched=0` printed alone
    cannot tell a typo in `--source` from a source whose steps are all pruned —
    the two readings send a reader in opposite directions.
    """
    tokens: set[str] = set()
    try:
        names = [p.name for p in Path(root).iterdir()]
    except OSError:
        return []
    for name in names:
        session_id = name[:-3] if name.endswith(".gz") else name
        session_id = session_id[:-5] if session_id.endswith(".json") else session_id
        token = source_of(session_id)
        if token and token != "chat":
            tokens.add(token)
        if len(tokens) >= cap:
            break
    return sorted(tokens)


def report_source(source: str, *, root: "Path | None" = None,
                  store: "Path | None" = None, limit: int = 25,
                  today: "str | None" = None,
                  system_prompt: "str | None" = None) -> dict[str, Any]:
    """Every replayable-or-not step of one recurring source, counted.

    A step is one recorded outgoing model request, so wherever the manifest
    survives the steps are counted from their manifest lines — a line IS the
    request. Where the day file is gone the session's own assistant rows stand in
    as the step count (`_ASSISTANT_ROLES`), because a step whose line has been
    pruned cannot be enumerated from the store at all and the alternative to
    counting it from the session is to not report it, which is how a corpus size
    becomes an assumption. The two buckets are printed separately and never
    merged: "N steps are replayable" and "N steps were measured" are different
    sentences, and the item's cost clause and owed retention decision (#2636's
    own open question about a >=30-step corpus against
    `DEFAULT_RETENTION_DAYS = 14`) are both answered by which of them is big.

    Every session is read exactly once, and each needed manifest day file is read
    exactly once for the whole corpus — see `_lines_by_session`.
    """
    started = time.perf_counter()
    root_path = Path(root) if root else sessions_dir()
    store_path = Path(store) if store else manifest_store()
    cutoff, newest = retention_window(store_path, today=today)
    files = day_files(store_path)

    sessions: "dict[str, dict[str, Any]]" = {}
    unreadable = 0
    for sid in source_sessions(source, root=root_path, limit=limit):
        session = read_session(sid, root=root_path)
        if session is None:
            unreadable += 1
            continue
        sessions[sid] = session

    days = {sid: utc_day_of(session) for sid, session in sessions.items()}
    inside = {sid: day for sid, day in days.items()
              if day >= cutoff and day in files}
    lines_by_session: "dict[str, list[dict[str, Any]]]" = {sid: [] for sid in inside}
    for day in sorted(set(inside.values())):
        wanted = {sid for sid, sid_day in inside.items() if sid_day == day}
        for sid, lines in _lines_by_session(wanted, files[day]).items():
            lines_by_session[sid] = lines

    rows: list[dict[str, Any]] = []
    # Three buckets, not two, because they mean opposite things. A step outside
    # the retention window is a corpus the sweep took; a step whose day is inside
    # the window and still has no manifest line is a corpus that was never
    # recorded — a disabled manifest, or a session that predates #581. Merging
    # them into `steps_skipped_outside_window` inflates the outside count with
    # steps no retention decision ever removed, which is the number clause 4
    # exists to print honestly.
    steps_outside = 0
    steps_no_lines = 0
    sessions_outside = 0
    sessions_no_lines = 0
    # Which vault the component rebuild could actually see. A mismatch is only
    # interpretable against it: `skills_index` moving is a different fact when the
    # rebuild read the live vault (the skills really changed since the step) than
    # when it read a round's sibling vault (the rebuild was reading other text
    # entirely), and the two are indistinguishable in the record without this.
    vault_roots: list[str] = []
    for sid, session in sessions.items():
        day = days[sid]
        assistant_rows = sum(1 for m in (session.get("messages") or [])
                             if m.get("role") in _ASSISTANT_ROLES)
        if day < cutoff:
            sessions_outside += 1
            steps_outside += assistant_rows
            continue
        if day not in files or not lines_by_session[sid]:
            sessions_no_lines += 1
            steps_no_lines += assistant_rows
            continue
        built = _rebuild(str(session.get("platform") or ""))
        if built["vault_root"] not in vault_roots:
            vault_roots.append(built["vault_root"])
        for line in lines_by_session[sid]:
            rows.append(_probe_record_from_line(sid, session, line,
                                                built["components"],
                                                system_prompt=system_prompt,
                                                vault_read=built["vault_root"]))

    matched = sum(1 for r in rows if r["verdict"] == "matched")
    out: dict[str, Any] = {
        "source": source,
        "retention_days": component_manifest.retention_days(),
        "window_cutoff": cutoff,
        "window_newest": newest,
        "vault_roots": vault_roots,
        "sessions_seen": len(sessions) + unreadable,
        "source_tokens_seen": _tokens_on_disk(root_path),
        "sessions_unreadable": unreadable,
        "sessions_probed": len({r["session_id"] for r in rows}),
        "sessions_skipped_outside_window": sessions_outside,
        "sessions_in_window_without_manifest_lines": sessions_no_lines,
        "steps_probed": len(rows),
        "steps_matched": matched,
        "steps_mismatched": len(rows) - matched,
        "steps_skipped_outside_window": steps_outside,
        "steps_in_window_without_manifest_lines": steps_no_lines,
        "moved_components": {},
        "rows": rows,
        "seconds": round(time.perf_counter() - started, 2),
    }
    for r in rows:
        name = r.get("moved_component") or "(none: every component hashes equal)"
        out["moved_components"][name] = out["moved_components"].get(name, 0) + 1
    return out


def _probe_record_from_line(session_id: str, session: dict[str, Any],
                            line: dict[str, Any],
                            rebuilt: "dict[str, dict[str, Any]]", *,
                            system_prompt: "str | None" = None,
                            vault_read: str = "") -> dict[str, Any]:
    """One report row: a line already in hand, so no day file is re-read for it."""
    messages = reconstruct_messages(session, line, system_prompt=system_prompt)
    reconstructed = message_digests(messages)
    recorded = message_digests_from_line(line)
    moved = first_moved_component(line, rebuilt)
    first_diff = next((i for i, (a, b) in enumerate(zip(recorded, reconstructed))
                       if a != b), None)
    if first_diff is None and len(recorded) != len(reconstructed):
        first_diff = min(len(recorded), len(reconstructed))
    matched = recorded == reconstructed
    return {
        "session_id": session_id,
        "source": source_of(session_id),
        "step": step_label(line),
        "request_id": str(line.get("request_id", "")),
        "verdict": "matched" if matched else "mismatched",
        "recorded_sha256": array_sha256(recorded),
        "reconstructed_sha256": array_sha256(reconstructed),
        "recorded_messages": len(recorded),
        "reconstructed_messages": len(reconstructed),
        "first_diff_index": None if matched else first_diff,
        "moved_component": "" if matched else moved,
        "vault_read": vault_read,
        "day": str(utc_day_of(session)),
        "seconds": 0.0,
    }


def format_report(report: dict[str, Any]) -> str:
    """The printed measurement: counts first, then every step that produced them."""
    lines = [
        f"source={report['source']} retention_days={report['retention_days']} "
        f"window={report['window_cutoff']}..{report['window_newest']}"
        # Named on the same line as the counts it qualifies, because every
        # `moved_component=` below is only readable against it. Printed as a list
        # even though a run normally has one: `_rebuild` memoises per platform, so a
        # corpus spanning two platforms really can have two.
        + " vault_read=" + (", ".join(report["vault_roots"])
                            or "(no step was rebuilt: nothing was compared)")
        + ("" if report["sessions_seen"] else
           "  (no session matched; tokens present: "
           + ", ".join(report["source_tokens_seen"]) + ")"),
        f"sessions_seen={report['sessions_seen']} "
        f"sessions_probed={report['sessions_probed']} "
        f"sessions_unreadable={report['sessions_unreadable']} "
        f"sessions_skipped_outside_window={report['sessions_skipped_outside_window']} "
        f"sessions_in_window_without_manifest_lines="
        f"{report['sessions_in_window_without_manifest_lines']}",
        f"steps_probed={report['steps_probed']} "
        f"steps_matched={report['steps_matched']} "
        f"steps_mismatched={report['steps_mismatched']} "
        f"steps_skipped_outside_window={report['steps_skipped_outside_window']} "
        f"steps_in_window_without_manifest_lines="
        f"{report['steps_in_window_without_manifest_lines']}",
        f"seconds={report['seconds']} (wall clock, no model request issued)",
    ]
    if report["moved_components"]:
        lines.append("first component whose hash moved, by step:")
        for name, count in sorted(report["moved_components"].items(),
                                  key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  {count:>4}  {name}")
    lines.append("per step:")
    for row in report["rows"]:
        lines.append("  " + _row_line(row))
    return "\n".join(lines)


def _row_line(row: dict[str, Any]) -> str:
    """One report row in one line, digests included — a count without them is unrepeatable."""
    moved = f" moved_component={row['moved_component']}" if row["moved_component"] else ""
    return (f"{row['session_id']} step={row['step']} request={row['request_id']} "
            f"verdict={row['verdict']} first_diff={row['first_diff_index']}"
            f"{moved} recorded={row['recorded_sha256'][:19]} "
            f"reconstructed={row['reconstructed_sha256'][:19]}")


# ── CLI ───────────────────────────────────────────────────────────────────


def main(argv: "list[str] | None" = None) -> int:
    """`probe --session <id> --step <n>` and `report --source <src>`.

    Read-only by construction: no subcommand takes an output path, and nothing
    here calls `record_request`, so the manifest writer thread is never started.
    """
    parser = argparse.ArgumentParser(
        prog="step_replay_fidelity",
        description="Measure whether a recorded step's outgoing messages array "
                    "can be rebuilt byte-for-byte from the bytes on disk (#2636).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    probe = sub.add_parser("probe", help="compare one recorded step with the rebuild")
    probe.add_argument("--session", required=True, help="session id")
    probe.add_argument("--step", required=True,
                       help="the manifest line's `iteration`, or `finalizer`")
    probe.add_argument("--json", action="store_true", help="print the record as JSON")

    report = sub.add_parser("report", help="count replayable steps per source")
    report.add_argument("--source", required=True,
                        help="recurring source, e.g. youtube-digest")
    report.add_argument("--max-sessions", type=int, default=25,
                        dest="max_sessions",
                        help="newest sessions of that source to look at")
    report.add_argument("--json", action="store_true", help="print as JSON")

    for spec in (probe, report):
        spec.add_argument("--sessions-dir", default="", dest="sessions_dir",
                          help="override the sessions directory")
        spec.add_argument("--store", default="",
                          help="override the request-manifest store root")
        spec.add_argument("--today", default="", dest="today",
                          help="anchor the retention window on this date "
                               "(YYYY-MM-DD) instead of today")

    args = parser.parse_args(argv)
    root = Path(args.sessions_dir) if args.sessions_dir else None
    store = Path(args.store) if args.store else None

    if args.cmd == "probe":
        record = probe_step(args.session, args.step, root=root, store=store,
                            today=args.today or None)
        print(json.dumps(record, indent=2) if args.json else format_probe(record))
        return 0 if record["verdict"] in ("matched", "mismatched") else 2

    out = report_source(args.source, root=root, store=store,
                        limit=args.max_sessions, today=args.today or None)
    print(json.dumps(out, indent=2) if args.json else format_report(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
