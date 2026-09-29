"""Content-addressed manifest for every outgoing model request (#581).

What this records
-----------------
Every request the harness sends to an inference engine is described, before it
is sent, as an **ordered list of the hashes of its parts**: the model id, the
chat template that will render it, each `prompt_builder` component by its dict
key, the tools array as one canonical-JSON hash plus one hash per advertised
tool definition, the prefetched context block, and each message in order — each
digest paired with the byte count of what it names. One NDJSON line per request
goes to the store. This is the representation Louf's runtime uses ("what you see
when you're using Codex is kind of a lie" — compaction, provider quirks and
unreturned thinking mean the transcript is not the request), and of the three
things his store buys — traceability, run-to-run diffing, replay — this records
the first two. Replay is deliberately not bought here; see the policy below.

Why it exists
-------------
`prompt_builder.build_system_prompt` already assembled a *named component dict*
and `measure_prompt` already reported each component's size — content was never
hashed, so no request could be compared with another. Session JSON keeps only
`{id, role, content, timestamp}` per message: no system prompt, no tools array,
no injected skill block. Item #520 (prefix-cache misses) can now name *which*
component's hash moved at which iteration instead of ablating; `app/prefix_miss.py`
already says *that* a break happened, and says nothing about which component —
this is the half that does. This is instrumentation: it does not move the hit
rate, and a landing here is not progress on #520.

Retention policy (the decision, in writing)
-------------------------------------------
**Digests and byte counts only. No content is retained — not components, not
messages, not the rendered prompt.** Concretely:

* A manifest line carries `sha256:<hex>` plus `bytes` per component, per tool
  definition, per message, and for the tools array and prefetch block. It never
  carries the text those digests name, so a sentinel string placed inside a
  component cannot be found in the store —
  `tests/test_component_manifest.py::test_no_written_line_carries_component_text`
  pins that, and it is the clause the rest of this section follows from.
* **No blob store and no `rebuild_request`.** Hashes without the bytes cannot
  rebuild a request, so nothing here can replay one. That is the narrowed scope
  this item was triaged to: its only would-be consumer (#564, differential
  replay of recent real turns) is retired, and retained component blobs —
  vault text, email bodies, user messages, deduplicated and addressable by
  content — are a strictly larger confidentiality problem than a diff needs. If
  a replay consumer returns, the blob store is a separate decision with its own
  policy, not an extension of this one.
* The **rendered prompt** — the chat-template output, with template markers,
  tool markup and the whole concatenated history — is never hashed and never
  written. No digest in a manifest line is `sha256(rendered_prompt)`: they are
  digests of strictly smaller parts.
* The manifest lines are the whole store. One per request, per iteration: measured
  at 22,737 bytes with 131 advertised tools and a full component set, of which
  133 digests for the tools array alone account for most of it — so budget tens of
  KB per iteration, not a few. They are what
  `scripts/meta_review/prompt_diff.py` reads.
* **Dated files are deleted after `harness.component_manifest.retention_days`
  (default 14; `0` disables).** The writer thread sweeps `manifests/` once per
  process day and removes every `<YYYY-MM-DD>.ndjson` older than the window,
  ageing a file by the date in its name rather than its mtime; nothing else in the
  tree deletes anything. This is a confidentiality window as much as a disk one:
  an unbounded store of digests of vault text, email bodies and user messages was
  the debt #1604 named — 2.4 GB over 9 days on 2026-09-27, with 2.8 TB free on the
  filesystem, so the bytes were never the problem.
* **Every line carries `build_ms`** — the milliseconds spent building that line,
  timed in `record_request` around `_build_line` alone. The cost of a line was
  asserted (~40 KB of canonical JSON to hash per request) and never measured; with
  the field present a p50 is computable from one day's file with no other source.
* The store lives OUTSIDE every git-tracked tree — `~/.local/state/lloyd-
  request-manifests` by default (`$LLOYD_MANIFEST_STORE` or
  `harness.component_manifest.store_dir` override it;
  `tests/test_component_manifest.py::test_the_store_resolves_outside_every_git_tree`
  refuses to accept a path inside one). Manifest lines carry digests of vault
  text and of user messages, and a length plus a hash is still a fingerprint of
  confidential content, so the store inherits the vault's confidentiality
  boundary. `POLICY.md`, holding this policy, is written into the store root on
  first use so the bytes carry their own rules, and refreshed on the next recorded
  request whenever the copy on disk says something other than what this module
  does — a notice that only ever gets written once freezes as a description of the
  build that happened to create the directory (#1781).

Never on the token path
-----------------------
`record_request` hashes inline — the snapshot must be the bytes being sent, and
the loop rewrites past messages in place (`_prune_reasoning`, tool-result
spill), so hashing later in a thread would describe a different request — and
then hands the finished line to a writer thread. No file I/O happens in the
stream. The hash work is one canonical dump per message plus one per tools
array; the per-definition tool digests, which are the expensive part with 131
tools, are computed once per tools-array *generation* and reused while the
array's own hash is unchanged. A failure at any point is counted in `stats()`
and swallowed: a manifest that cannot be written must neither delay nor fail the
turn, and must never raise into `stream_chat`'s caller.
`tests/test_component_manifest.py::test_a_broken_writer_still_completes_the_turn`
drives three real iterations with the append point raising `OSError` and asserts
both halves of that: the turn completes, and `stats()["write_errors"]` is the
only thing that noticed.

Seams
-----
Three process boundaries this module sits across, each with the test that
crosses it rather than a grep that suggests it:

* `prompt_builder` writes components into this module's per-session registry
  and `app/harness/client.py` reads it from inside the loop's task — a
  write-in-one-module/read-in-another boundary with no call chain between them,
  pinned by `test_the_component_dict_prompt_builder_built_is_the_one_recorded`.
* The hashing runs in the calling task but the writing runs in a daemon thread
  of the backend process, so "it was counted" and "it is on disk" are different
  facts; every test here calls `flush()` before reading the file, and the test
  that asserts `recorded` also asserts the line count it implies.
* Five send sites reach this module from five different callers, three of them
  in other processes' code paths (the inner-voice observer, the compaction
  critic, the post-capture secondary-engine jobs). Each is exercised by driving
  the real send function against a stub engine in
  `tests/test_component_manifest_send_sites.py`, not by grepping for
  `record_request(`.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
from collections import OrderedDict
from datetime import date as _day, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("lloyd-server")

#: Line format tag. Bump it if the shape changes so a reader can tell.
SCHEMA = "lloyd.component-manifest/1"

#: The policy, verbatim, as it is written into the store root.
RETENTION_POLICY = (
    "# Request-manifest retention policy (#581)\n"
    "\n"
    "Digests and byte counts only. No content is retained here - not components,\n"
    "not messages, not the rendered prompt.\n"
    "\n"
    "* `manifests/<YYYY-MM-DD>.ndjson` - one line per outgoing model request:\n"
    "  request_id, session, iteration, model, provider/chat-template id, and for\n"
    "  each component (model id, tools array, each advertised tool definition,\n"
    "  each prompt_builder component by dict key, the prefetch block, each\n"
    "  message in order) a `sha256:` digest and its byte count. Nothing else.\n"
    "* There is no blob directory and no `rebuild_request`: hashes without the\n"
    "  bytes cannot replay a request, and that is the narrowed scope this item\n"
    "  was triaged to. A replay consumer is a separate decision with its own\n"
    "  confidentiality policy.\n"
    "* The rendered prompt (chat-template output: markers, tool markup, the whole\n"
    "  concatenated history) is never hashed and never written here.\n"
    "* Dated files are DELETED after harness.component_manifest.retention_days\n"
    "  (default 14; 0 disables). The writer thread sweeps manifests/ once per\n"
    "  process day and removes every <YYYY-MM-DD>.ndjson older than the window,\n"
    "  ageing a file by the date in its name, not its mtime. Nothing else in the\n"
    "  store is ever removed - including this file.\n"
    "\n"
    "The digests are of vault text, email bodies and user messages, and a length\n"
    "plus a hash is still a fingerprint of confidential content. This directory is\n"
    "outside every git-tracked tree; that is necessary and not sufficient - treat\n"
    "it with the same confidentiality as ~/obsidian.\n"
)
#: `RETENTION_POLICY` as bytes, so the steady-state check `_write_policy` runs on
#: every recorded request is one read and one byte comparison — no second copy of
#: the text to keep in step, and no encode per request.
_POLICY_BYTES = RETENTION_POLICY.encode("utf-8")
POLICY_FILENAME = "POLICY.md"
DEFAULT_SUBDIR = "lloyd-request-manifests"

#: Distinct sessions whose components are remembered. Count-bounded on purpose:
#: an autonomy round runs for hours on one session and its entry must outlive
#: the turn, so an age-based expiry would silently blank long turns.
MAX_SESSIONS = 128

_UNRECORDED = "unrecorded"

# ── config ────────────────────────────────────────────────────────────────


def _cfg() -> dict[str, Any]:
    try:
        from app.config import CONFIG
        return dict((CONFIG.get("harness") or {}).get("component_manifest") or {})
    except Exception:
        return {}


def enabled() -> bool:
    """Master switch. On by default: this is the record everything else reads."""
    return bool(_cfg().get("enabled", True))


def store_root() -> Path:
    """Where the manifest lines live. Never inside a git-tracked tree.

    Precedence: `$LLOYD_MANIFEST_STORE`, then
    `harness.component_manifest.store_dir`, then `$XDG_STATE_HOME` (or
    `~/.local/state`) + `lloyd-request-manifests` — the same neighbourhood the
    automod audit trail already uses (`~/.local/state/lloyd-automod`).
    `git_tree_containing()` is the check a reviewer can re-run; the test asserts
    the resolved default, not just this sentence.
    """
    cfg = _cfg()
    override = os.environ.get("LLOYD_MANIFEST_STORE") or str(cfg.get("store_dir") or "")
    if override:
        return Path(override).expanduser()
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state).expanduser() / DEFAULT_SUBDIR


def git_tree_containing(path: "Path | str") -> str:
    """Nearest ancestor of `path` that is a git working tree, or `""`.

    Walks upward looking for a `.git` directory or file (a worktree's `.git` is
    a file, so `is_dir()` alone would miss this very checkout's sibling
    worktrees). The store must resolve to a path with no such ancestor: a
    manifest line is a fingerprint of vault text, and the one guarantee the
    retention policy can actually enforce mechanically is that it never lands in
    a tree someone commits.
    """
    try:
        cur = Path(path).expanduser().absolute()
    except (OSError, ValueError):
        return ""
    for candidate in (cur, *cur.parents):
        try:
            if (candidate / ".git").exists():
                return str(candidate)
        except OSError:
            continue
    return ""


# ── retention over the day files ──────────────────────────────────────────
#
# The store grew with age and nothing bounded it: 2.4 GB across 9 dated files on
# 2026-09-27 (45 MB → 531 MB/day), still being written, with no deletion anywhere
# in the tree (#1604). Bytes were not the argument — `df` had 2.8 T free. What was
# unbounded is retention of `sha256:` digests of vault text, email bodies and user
# messages, which this module's own `POLICY.md` says to treat "with the same
# confidentiality as ~/obsidian". The window is that confidentiality decision
# expressed as a file age.
#
# It lives in the writer rather than in `scripts/groundskeeper/retention-sweep.py`
# (which has never heard of this store) because the sweep is weekly and this store
# takes a day's worth of lines per run of the box; a round-landable question that
# belongs to the directory's owner belongs here. Idempotent, so it costs nothing
# that several processes each run their own writer thread against one store.

#: Used when `harness.component_manifest.retention_days` is absent or unparseable.
#: 14 is what the item asks for, and it is already wider than what the only reader
#: wants: `scripts/meta_review/prompt_diff.py` opens at most the last two dated
#: files, so the window bounds the reader's work as well as the store.
DEFAULT_RETENTION_DAYS = 14

#: `<YYYY-MM-DD>.ndjson` and nothing else. `_day_file` writes exactly this shape,
#: so a name that does not match it is not this module's file and is never
#: removed — which is also why `POLICY.md` and anything a human drops in the store
#: survive every sweep.
_DAY_FILE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})\.ndjson$")


def retention_days() -> int:
    """The age window in days; `0` or negative disables pruning entirely.

    Read per call like the rest of the config here, so a process that started
    before the key existed and one started after agree on one store. The value goes
    through `str()` on purpose: YAML hands over a string as often as a number, and a
    bare `int(raw)` would read a mistyped `false` as the 0 that means "delete
    everything", where `int("False")` has to fail and fall back to the default.
    """
    raw = _cfg().get("retention_days")
    if raw is None:
        return DEFAULT_RETENTION_DAYS
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_RETENTION_DAYS


def prune_store(*, store: "Path | str | None" = None,
                today: "str | None" = None) -> "dict[str, int]":
    """Delete dated day files older than `retention_days()`. Never raises.

    Age is the date *in the name*, not mtime: a file is appended to for its own
    day and renamed by nothing, so the name is the record's age and a restored or
    clock-skewed mtime cannot make an old file look young. The cutoff is
    `today - (days - 1)`, so the default window keeps exactly 14 dated files
    including today's. `today` is a parameter so a test can hold it fixed instead
    of depending on when it ran.

    Only files inside `manifests/` whose name parses as a date are eligible; this
    never walks anywhere else, and `2026-13-99.ndjson` is not a day so it is not
    ours to delete. Returns `{"files": deleted, "bytes": freed, "errors": n}` — an
    unlistable directory is one error and deletes nothing.
    """
    out = {"files": 0, "bytes": 0, "errors": 0}
    days = retention_days()
    if days <= 0:
        return out
    root = Path(store) if store else store_root()
    manifests = root / "manifests"
    if not manifests.is_dir():
        return out
    try:
        newest = datetime.strptime(today or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                                   "%Y-%m-%d").date()
    except ValueError:
        return out
    cutoff = newest - timedelta(days=days - 1)
    try:
        paths = list(manifests.iterdir())
    except OSError:
        out["errors"] += 1
        return out
    for path in paths:
        match = _DAY_FILE_RE.match(path.name)
        if match is None:
            continue
        try:
            file_day = _day(int(match.group(1)), int(match.group(2)),
                            int(match.group(3)))
        except ValueError:
            continue
        if file_day >= cutoff:
            continue
        try:
            out["bytes"] += path.stat().st_size
            path.unlink()
            out["files"] += 1
        except OSError:
            out["errors"] += 1
    return out


def _prune_and_count(*, today: "str | None" = None) -> "dict[str, int]":
    """`prune_store` plus the counters and the one log line a report reads.

    Retention is never allowed to reach a request or kill the writer thread: a
    store that cannot be swept simply keeps growing, which is the old behaviour,
    not an outage. `prune_errors` is the counter that says it still is.
    """
    try:
        result = prune_store(today=today)
    except Exception as exc:  # noqa: BLE001 — retention never breaks a turn
        _bump("prune_errors")
        logger.warning("component_manifest: %s while pruning the manifest store — "
                       "nothing was deleted and the store keeps growing",
                       type(exc).__name__)
        return {"files": 0, "bytes": 0, "errors": 1}
    _bump("pruned_files", int(result.get("files", 0)))
    _bump("pruned_bytes", int(result.get("bytes", 0)))
    _bump("prune_errors", int(result.get("errors", 0)))
    if result["files"]:
        logger.info("component_manifest: pruned %d manifest day file(s) older than "
                    "%d day(s) from %s (%d byte(s) freed)", result["files"],
                    retention_days(), store_root() / "manifests", result["bytes"])
    return result


# ── hashing ───────────────────────────────────────────────────────────────


def canonical_json(obj: Any) -> str:
    """The one serialization every digest in this module uses.

    Sorted keys and no whitespace: two requests that differ only in the order
    dict keys happened to be built in must not read as a prompt change, and the
    diff CLI has to hash the same array the same way twice to say "no, that one
    is identical".
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def digest_obj(obj: Any) -> str:
    return digest_text(canonical_json(obj))


def _size(text: str) -> int:
    return len(text.encode("utf-8", "replace"))


# ── per-session component registry ────────────────────────────────────────
#
# `prompt_builder.build_system_prompt` has the named component dict and no
# knowledge of the request; `app/harness/client.py` has the request and no
# knowledge of the components. Nothing calls across the gap, so the handoff is
# this table, keyed by the session id both sides already have. Peeking (not
# popping) is deliberate: a turn has many iterations and its component dict is
# built once per turn.

_registry: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_registry_lock = threading.Lock()


def note_components(session_id: str, components: "dict[str, str] | None", *,
                    prefetch_text: str | None = None) -> None:
    """Remember the component dict for `session_id`. Never raises.

    Called from `prompt_builder` with the same ordered dict that produced the
    system prompt, and from the chat router with the prefetched context block.
    """
    if not session_id or not enabled():
        return
    try:
        with _registry_lock:
            entry = _registry.setdefault(session_id, {})
            if components is not None:
                entry["components"] = dict(components)
            if prefetch_text is not None:
                entry["prefetch"] = prefetch_text
            entry["at"] = time.time()
            _registry.move_to_end(session_id)
            while len(_registry) > MAX_SESSIONS:
                _registry.popitem(last=False)
    except Exception as exc:  # noqa: BLE001 — a lost record is never an error
        logger.debug("component_manifest: registry write failed: %s", exc)


def note_prefetch(session_id: str, text: str) -> None:
    """Remember the injected context block for `session_id`."""
    note_components(session_id, None, prefetch_text=text or "")


def components_for(session_id: str) -> dict[str, Any]:
    """The remembered `{components, prefetch, at}` for a session, or `{}`."""
    if not session_id:
        return {}
    with _registry_lock:
        entry = _registry.get(session_id)
        return dict(entry) if entry else {}


def _reset_registry() -> None:
    with _registry_lock:
        _registry.clear()


# ── provider + chat template identity ─────────────────────────────────────
#
# The item's open question: a manifest can only rebuild a request byte-exactly
# if the provider and the chat-template version are recorded. Both fields are
# written on every line from day one; the digest is real whenever the template
# file is reachable, and says plainly why it is not when it is not.

_template_cache: "dict[str, dict[str, Any]]" = {}
_template_lock = threading.Lock()


def provider_for(base_url: str) -> dict[str, Any]:
    """Which configured engine slot a request is heading to.

    Matched on `base_url` against `models.*`, the same table
    `app/model_identity.py` verifies at boot. `unrecorded` is written rather
    than a guess: a wrong engine label in the record is worse than an empty one.
    """
    target = (base_url or "").rstrip("/")
    try:
        from app.config import CONFIG
        models = CONFIG.get("models") or {}
    except Exception:
        models = {}
    for name, cfg in (models or {}).items():
        if not isinstance(cfg, dict):
            continue
        url = str(cfg.get("base_url") or "").rstrip("/")
        if url and url == target:
            return {"slot": name, "base_url": url,
                    "engine": str(cfg.get("engine") or "vllm"),
                    "expect_model": str(cfg.get("expect_model") or ""),
                    "source": "config models match on base_url"}
    return {"slot": _UNRECORDED, "base_url": target, "engine": "",
            "expect_model": "", "source": "no config model at this base_url"}


def chat_template(model: str = "", base_url: str = "") -> dict[str, Any]:
    """Content hash of the template that will render this request, or why not.

    Looked up once per (model, slot) and cached, because this runs inline in the
    request-build path: an explicit `harness.component_manifest.chat_template_path`,
    else `chat_template.jinja` / `.json` / `.jsonl` under
    `harness.component_manifest.model_dir`. The engine's own served path is not
    probed from here — that is an HTTP call, and nothing in this module does I/O
    slower than reading one small file.
    """
    slot = provider_for(base_url).get("slot") or _UNRECORDED
    key = f"{model}|{slot}"
    with _template_lock:
        cached = _template_cache.get(key)
    if cached is not None:
        return cached
    cfg = _cfg()
    candidates: list[Path] = []
    explicit = str(cfg.get("chat_template_path") or "")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    model_dir = str(cfg.get("model_dir") or "")
    if model_dir:
        root = Path(model_dir).expanduser()
        candidates += [root / name for name in
                       ("chat_template.jinja", "chat_template.json",
                        "chat_template.jsonl")]
    found: Path | None = None
    for path in candidates:
        try:
            if path.is_file():
                found = path
                break
        except OSError:
            continue
    if found is not None:
        try:
            result = {"id": digest_text(found.read_text(encoding="utf-8",
                                                         errors="replace")),
                      "source": str(found)}
        except OSError as exc:
            result = {"id": _UNRECORDED, "source": f"unreadable {found}: {exc}"}
    else:
        result = {
            "id": _UNRECORDED,
            "source": "no chat template file found; set harness.component_manifest."
                      "chat_template_path or model_dir",
        }
    with _template_lock:
        _template_cache[key] = result
    return result


def _reset_template_cache() -> None:
    with _template_lock:
        _template_cache.clear()


# ── the writer thread ─────────────────────────────────────────────────────

_queue: "queue.Queue[str | None]" = queue.Queue()
_writer_started = False
_writer_lock = threading.Lock()
_stats_lock = threading.Lock()
# `write_errors` is the clause the acceptance names; `recorded` and
# `lines_written` are the pair that proves the writer thread actually drained
# what it was handed, which one counter alone cannot say.
_stats = {"recorded": 0, "write_errors": 0, "hash_errors": 0,
          "lines_written": 0, "bytes_written": 0,
          "pruned_files": 0, "pruned_bytes": 0, "prune_errors": 0}


def _bump(key: str, amount: int = 1) -> None:
    with _stats_lock:
        _stats[key] = _stats.get(key, 0) + amount


def stats() -> dict[str, int]:
    """Counters a report reads. `write_errors` is the one the acceptance names."""
    with _stats_lock:
        return dict(_stats)


def reset_stats() -> None:
    """Zero the counters. Start-of-process and tests."""
    with _stats_lock:
        for key in _stats:
            _stats[key] = 0


def _ensure_writer() -> None:
    global _writer_started
    with _writer_lock:
        if _writer_started:
            return
        threading.Thread(target=_writer_loop, name="component-manifest",
                         daemon=True).start()
        _writer_started = True


#: How long the writer may sit on an empty queue before it looks at the calendar.
#: The prune needs a clock and this thread is the only thing that ever touches the
#: store: parked in a blocking `get()` it would never wake on a date change, so a
#: process that outlives midnight would keep appending to a file the window has
#: already passed — and on this box the backend runs for weeks. 15 minutes is a
#: date-string compare per wake, which is not a cost worth a smaller number.
_IDLE_POLL_SECONDS = 900.0


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _sweep_if_new_day(pruned_on: str) -> str:
    """Sweep the store if the date has moved since `pruned_on`; return the new marker.

    The one trigger retention has. `pruned_on` is the caller's own memory, not a
    module global: every process that sends a request runs its own writer thread
    against the one store, and each only needs to avoid sweeping twice *itself* —
    a sweep is idempotent, so the cost per process is one directory listing a day.
    Called with `""` it always sweeps, which is what a test that wants a sweep now,
    rather than at midnight, uses.
    """
    today = _today()
    if today != pruned_on:
        _prune_and_count(today=today)
    return today


def _writer_loop() -> None:
    # One sweep per process-day. It runs before the first `get()`, so a store that
    # filled while the box was down is bounded the moment it is next opened, and
    # again on the first wake after midnight, so a backend that runs for weeks does
    # not keep appending into a file the window has passed.
    pruned_on = _sweep_if_new_day("")
    while True:
        try:
            item = _queue.get(timeout=_IDLE_POLL_SECONDS)
        except queue.Empty:
            pruned_on = _sweep_if_new_day(pruned_on)
            continue
        try:
            if item is None:
                return
            line_text = item
            root = store_root()
            try:
                _write_policy(root)
                _append_line(root, line_text)
            except Exception as exc:  # noqa: BLE001 — counted, never raised
                _bump("write_errors")
                logger.warning(
                    "component_manifest: %s while writing a manifest line to %s "
                    "(the request it describes was unaffected; counted in "
                    "component_manifest.stats() as write_errors)",
                    type(exc).__name__, root)
        finally:
            _queue.task_done()


def _write_policy(root: Path) -> None:
    """Put the policy in the store root, and refresh it when it has gone stale.

    Write-if-stale, not write-if-absent (#1781). The notice is compared against
    `_POLICY_BYTES` rather than merely checked for existence, because existence is
    what made the contradiction permanent: the store created on 2026-09-19 holds a
    1,193-byte notice whose only difference from shipped `RETENTION_POLICY` is the
    bullet saying dated files are DELETED, so the file sitting in the store
    described a store nothing pruned while `prune_store` beside it deleted one
    daily, and no request could ever correct it.

    The steady-state cost is one read of a ~1.5 KB file on the path that is already
    appending a line under `flock`, and a notice that matches is written zero times
    — not an mtime bump — so nothing here scales with the size of a request.
    `tests/test_component_manifest_retention.py::test_a_current_policy_notice_is_not_rewritten_by_recorded_requests`
    holds that line.

    Anything that stops the notice being read counts as stale and is replaced: a
    document nobody can open is not doing its job, and `POLICY.md` is this module's
    own file. The refresh is still best-effort, because the notice must never cost a
    request its manifest line — an `OSError` here is counted and logged and
    `_append_line` runs regardless.
    """
    policy = root / POLICY_FILENAME
    try:
        if policy.read_bytes() == _POLICY_BYTES:
            return
    except OSError:
        pass  # absent, unreadable, or not a file: every one is a notice to rewrite
    try:
        root.mkdir(parents=True, exist_ok=True)
        policy.write_bytes(_POLICY_BYTES)
    except OSError as exc:  # noqa: BLE001 — the notice loses, the line does not
        _bump("write_errors")
        logger.warning(
            "component_manifest: could not refresh %s (%s: %s) — the policy notice "
            "in the store may be stale; the manifest line it sits beside was "
            "unaffected", policy, type(exc).__name__, exc)


def _day_file(root: Path) -> Path:
    """Today's NDJSON path. One function so the reader and the writer cannot disagree."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return root / "manifests" / f"{day}.ndjson"


def _append_line(root: Path, line_text: str) -> None:
    """The single append point. Tests make THIS one raise OSError.

    Several OS processes append to one day file: the backend, `agent_mcp/main.py`,
    `workers/sources/*.py` and the automod round subprocess each run their own
    writer thread and each open this file per line, so two writers are never in the
    same address space and no queue can order them. What makes one record land whole
    is therefore two things, and both are done here rather than inherited from
    whoever owns the stdlib defaults:

    * `fcntl.flock(LOCK_EX)` over the whole record. An advisory lock taken on the
      open descriptor serialises appenders that share nothing but the path, which is
      exactly the set that exists here.
    * one `os.write` of the encoded line on an `O_APPEND` descriptor, so the kernel
      takes the end-of-file offset and the bytes together, and a short write is
      resumed from where it stopped instead of silently truncating the record.

    A buffered text-mode `fh.write()` is deliberately not used. A line here measured
    22,737 bytes with 131 advertised tools — five times `PIPE_BUF` (4096 on this
    box) and nearly three times the 8 KiB stdio buffer — so whether a record reached
    the file as one `write(2)` or as several depended on how CPython's text layer
    chose to chunk that particular string. A guarantee that holds because the buffer
    happened to flush in one piece is not a guarantee.
    """
    path = _day_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (line_text + "\n").encode("utf-8")
    fd = os.open(path,
                 os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        view = memoryview(payload)
        while view:
            view = view[os.write(fd, view):]
    finally:
        # Closing releases the lock; unlocking first keeps a held lock from
        # outliving this call if the retry loop above ever raises.
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    _bump("lines_written", 1)
    _bump("bytes_written", len(line_text) + 1)


def flush(timeout: float = 5.0) -> bool:
    """Block until the writer has drained everything queued so far.

    For tests and shutdown; the request path never calls it.
    """
    if not _writer_started:
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if getattr(_queue, "unfinished_tasks", 0) == 0:
            return True
        time.sleep(0.005)
    return False


# ── tools-array generation memo ───────────────────────────────────────────
#
# 131 tool definitions per request is the expensive part — ~40 KB of canonical
# JSON to serialize and hash on every iteration. Their digests are computed once
# per *generation*, whenever the array's own hash changes, and reused while it
# does not, which is every iteration of a turn that neither adds nor removes a
# tool. This is the item's own cost question answered in code: the array's
# single hash still goes on every line, so a per-request change is still caught.

_tools_memo: dict[str, Any] = {"array_sha": "", "block": None}
_tools_lock = threading.Lock()


def _tools_block(tools: "list[dict[str, Any]] | None") -> "dict[str, Any] | None":
    """Hash the tools array, and per definition when the array is new."""
    if not tools:
        return None
    array_text = canonical_json(tools)
    array_sha = digest_text(array_text)
    with _tools_lock:
        if _tools_memo["array_sha"] == array_sha and _tools_memo["block"]:
            return _tools_memo["block"]
    definitions: list[dict[str, Any]] = []
    for tool in tools:
        name = ""
        if isinstance(tool, dict):
            fn = tool.get("function")
            name = str(tool.get("name")
                       or (fn.get("name") if isinstance(fn, dict) else "")
                       or "")
        text = canonical_json(tool)
        digest = digest_text(text)
        definitions.append({"name": name, "sha256": digest, "bytes": _size(text)})
    block = {"sha256": array_sha, "bytes": _size(array_text),
             "count": len(tools), "definitions": definitions}
    with _tools_lock:
        _tools_memo["array_sha"] = array_sha
        _tools_memo["block"] = block
    return block


def _reset_tools_memo() -> None:
    with _tools_lock:
        _tools_memo["array_sha"] = ""
        _tools_memo["block"] = None


# ── recording ─────────────────────────────────────────────────────────────


def source_of_session(session_id: str) -> str:
    """`chat` for a chat session, the worker source name for a background one.

    A background session id has four underscore-separated parts
    (`20260910_120001_autocode_9f2a`) and a chat's has three — the same fast
    path `prefix_miss.label_for_session` and `sessions_io` take, so a line says
    who sent the request without another lookup.
    """
    parts = (session_id or "").split("_")
    if len(parts) >= 4 and parts[2]:
        return parts[2]
    return "chat"


def _request_id() -> str:
    return "r-%s-%s" % (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
                        os.urandom(5).hex())


def record_request(*, base_url: str, model: str, payload: dict[str, Any],
                   session_id: str = "", iteration: "int | None" = None,
                   send_site: str = "") -> "dict[str, Any] | None":
    """Manifest the request that is about to be sent; return the line.

    Called with the payload already built and BEFORE the connection opens, so
    the digests describe the bytes going on the wire. Hashing is inline (the
    loop mutates message dicts in place afterwards, so a deferred snapshot would
    not be this request); the file write goes to the writer thread. Never raises
    and never reports a failure to the caller: on any problem it counts it in
    `stats()` and returns None, and the request proceeds. What it hands the
    writer is a serialized line and nothing else — no bytes of the request
    itself, which is the whole retention policy in one sentence.
    """
    if not enabled():
        return None
    try:
        # The cost of describing a request is not free: 131 tool definitions is
        # ~40 KB of canonical JSON to hash, and the memo only helps while the array
        # is unchanged. Nothing measured this before (#1604), so the timer starts
        # here — around the build and nothing else, which is why it cannot live in
        # `_build_line`, and the queue hand-off and `_bump` below are outside it.
        started = time.perf_counter()
        line = _build_line(base_url=base_url, model=model, payload=payload,
                           session_id=session_id, iteration=iteration,
                           send_site=send_site)
        line["build_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        _ensure_writer()
        _queue.put(canonical_json(line))
        _bump("recorded")
        return line
    except Exception as exc:  # noqa: BLE001 — the turn is not the manifest's
        _bump("hash_errors")
        logger.warning("component_manifest: could not build a manifest line at %s "
                       "(%s: %s) — request unaffected",
                       send_site or "?", type(exc).__name__, exc)
        return None


def _build_line(*, base_url: str, model: str, payload: dict[str, Any],
                session_id: str, iteration: "int | None",
                send_site: str) -> dict[str, Any]:
    """Describe the payload as digests and sizes. Keeps no text but the digests.

    `params` is the deliberately non-content-bearing half of the payload — every
    key except `messages` and `tools`, which is sampling parameters plus the
    model id. It is retained verbatim because a diff that cannot say
    "temperature moved" is not a diff, and because none of it is content;
    `test_no_written_line_carries_component_text` is what stops it from quietly
    growing into a transcript.
    """
    message_rows: list[dict[str, Any]] = []
    for index, message in enumerate(payload.get("messages") or []):
        text = canonical_json(message)
        digest = digest_text(text)
        role = str(message.get("role")) if isinstance(message, dict) else ""
        message_rows.append({"index": index, "role": role, "sha256": digest,
                             "bytes": _size(text)})

    tools = _tools_block(payload.get("tools"))

    entry = components_for(session_id)
    component_rows = []
    for name, text in (entry.get("components") or {}).items():
        body = text or ""
        component_rows.append({"name": name, "sha256": digest_text(body),
                               "bytes": _size(body)})

    scalar = {k: v for k, v in payload.items() if k not in ("messages", "tools")}
    line: dict[str, Any] = {
        "schema": SCHEMA,
        "ts": time.time(),
        "request_id": _request_id(),
        "session_id": session_id or "",
        "source": source_of_session(session_id),
        "iteration": iteration,
        "send_site": send_site or "",
        "model": model or "",
        "base_url": base_url or "",
        "provider": provider_for(base_url),
        "chat_template": chat_template(model, base_url),
        "components": component_rows,
        # Why this is turn-scoped: `build_system_prompt` runs once per user turn,
        # so every iteration of that turn carries the same component dict; the
        # per-iteration re-anchor rides in on a message, and the message digests
        # are what catch it moving.
        "components_captured": "turn_start" if component_rows else _UNRECORDED,
        "messages": message_rows,
        "params": scalar,
    }
    if tools is not None:
        line["tools"] = tools
    prefetch = entry.get("prefetch")
    if prefetch:
        line["prefetch"] = {"sha256": digest_text(prefetch),
                            "bytes": _size(prefetch)}
    return line


# ── reading ───────────────────────────────────────────────────────────────


def read_manifest(request_id: str, *, store: "Path | str | None" = None,
                  date: str = "") -> "dict[str, Any] | None":
    """Fetch one manifest line by `request_id`.

    The newest day file is scanned first and a full scan is the fallback; ids
    carry no date, so a caller that knows one passes it.
    """
    root = Path(store) if store else store_root()
    manifests = root / "manifests"
    files: list[Path] = []
    if date:
        files.append(manifests / f"{date}.ndjson")
    if manifests.is_dir():
        files += sorted(manifests.glob("*.ndjson"), reverse=True)
    seen: set[str] = set()
    for path in files:
        key = str(path)
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        try:
            with open(path, encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        line = json.loads(raw)
                    except ValueError:
                        continue
                    if line.get("request_id") == request_id:
                        return line
        except OSError:
            continue
    return None


# Deliberately absent: `load_blob` and `rebuild_request`. The store holds no
# bytes, so a manifest addresses a request it cannot reproduce — which is the
# narrowed scope #581 was triaged to once its only replay consumer (#564) was
# retired. The diff needs only digests; a replay consumer would need the
# retained-blob decision made again, in the open, with a confidentiality policy
# of its own.


# ── diffing ───────────────────────────────────────────────────────────────


def component_positions(line: dict[str, Any]) -> "list[tuple[str, str, dict[str, Any]]]":
    """A manifest line as an ordered list of `(position, digest, detail)`.

    Order is render order, not dict order: Qwen's template puts the tools array
    inside the system message (which is why `finalizer.py` insists on the
    identical array), so `tools` precedes the prompt components, which precede
    the conversation. Positions after that are messages, by index.

    The tools array is ONE position. Its per-definition digests are recorded, and
    the definitions that moved are named in the diff row's `moved` detail, but they
    are not positions of their own: the array digest can only move when a
    definition does, so emitting both would report one change twice, and the
    question this answers — "was it just my message, or did the model get a
    different tool?" — wants one name. The name is still said, which is the whole
    reason the per-definition digests are recorded.
    """
    rows: "list[tuple[str, str, dict[str, Any]]]" = []
    tools = line.get("tools")
    if tools:
        rows.append(("tools", tools["sha256"],
                     {"bytes": tools.get("bytes"), "count": tools.get("count")}))
    for row in line.get("components") or []:
        rows.append((f"system.{row['name']}", row["sha256"], {"bytes": row.get("bytes")}))
    prefetch = line.get("prefetch")
    if prefetch:
        rows.append(("prefetch", prefetch["sha256"], {"bytes": prefetch.get("bytes")}))
    for row in line.get("messages") or []:
        rows.append((f"message[{row['index']}]", row["sha256"],
                     {"role": row.get("role"), "bytes": row.get("bytes")}))
    return rows


def diff_manifests(a: dict[str, Any], b: dict[str, Any]) -> "list[dict[str, Any]]":
    """What changed between two requests, in position order.

    The diff Louf demos — "was it just my message? did I give the model a
    different skill, a different tool?" A changed count of messages shows up as
    added or removed positions, which is how a turn that appended one tool
    result reads, as distinct from a turn whose tools array moved.
    """
    pa, pb = component_positions(a), component_positions(b)
    da = {name: (digest, detail) for name, digest, detail in pa}
    db = {name: (digest, detail) for name, digest, detail in pb}
    out: list[dict[str, Any]] = []
    for name, digest, detail in pa:
        other = db.get(name)
        if other is None:
            out.append({"position": name, "status": "removed", "a": digest,
                        "b": None, "a_detail": detail, "b_detail": None})
        elif other[0] != digest:
            out.append({"position": name, "status": "changed", "a": digest,
                        "b": other[0], "a_detail": detail, "b_detail": other[1]})
    for name, digest, detail in pb:
        if name not in da:
            out.append({"position": name, "status": "added", "a": None,
                        "b": digest, "a_detail": None, "b_detail": detail})
    for row in out:
        if row["position"] == "tools":
            moved = moved_definitions(a.get("tools"), b.get("tools"))
            if moved:
                row["moved"] = moved
    return out


def moved_definitions(tools_a: "dict[str, Any] | None",
                      tools_b: "dict[str, Any] | None") -> "list[str]":
    """Which tool definitions differ, read off the two arrays' per-definition digests.

    Names only: the point is to say `Bash changed`, not what changed in it, which a
    store that keeps no bytes could not say anyway. A tool present on one side only
    is named `added` or `removed`, because "did the model get a different tool?"
    includes a tool that was not there before.
    """
    da = {d.get("name"): d.get("sha256") for d in (tools_a or {}).get("definitions") or []}
    db = {d.get("name"): d.get("sha256") for d in (tools_b or {}).get("definitions") or []}
    out: list[str] = []
    for name in list(da) + [n for n in db if n not in da]:
        if da.get(name) == db.get(name):
            continue
        if name in da and name in db:
            out.append(f"{name} changed")
        else:
            out.append(f"{name} {'added' if name in db else 'removed'}")
    return out


def engine_changes(a: dict[str, Any], b: dict[str, Any]) -> "list[str]":
    """Which engine slot or chat template differs, as its own sentences.

    Deliberately not positions: the cached prefix is per-engine whether or not a
    byte of the prompt moved, so "this request went somewhere else" is context for
    reading the component list, not a component in it. Only each field's naming key
    is compared — the rest of those dicts is provenance about how the value was
    obtained (`source`, the base_url it matched on), and two requests that agree on
    the slot but disagree on where the template file was read from are not two
    different prompts.
    """
    out = []
    for field, key in (("provider", "slot"), ("chat_template", "id")):
        val_a = (a.get(field) or {}).get(key)
        val_b = (b.get(field) or {}).get(key)
        if val_a != val_b:
            out.append(f"{field}: {_short_str(val_a)} -> {_short_str(val_b)}")
    return out


def _short_str(value: object) -> str:
    """An identity field's naming key as short text; an unknown half reads as `—`."""
    if value in (None, ""):
        return "—"
    text = str(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _short(digest: "str | None") -> str:
    if not digest:
        return "—"
    return digest.replace("sha256:", "")[:12]


def format_diff(a: dict[str, Any], b: dict[str, Any]) -> str:
    """The CLI's rendering: one line per differing position, in position order."""
    diffs = diff_manifests(a, b)
    head = (f"{a.get('request_id')} (iteration {a.get('iteration')}, "
            f"{a.get('send_site')}) vs {b.get('request_id')} "
            f"(iteration {b.get('iteration')}, {b.get('send_site')})")
    if not diffs and not engine_changes(a, b):
        return head + "\nno component differs: the two requests hash identically"
    lines = [head, *engine_changes(a, b)]
    for row in diffs:
        a_detail, b_detail = row["a_detail"] or {}, row["b_detail"] or {}
        detail = ""
        if row.get("moved"):
            moved = row["moved"]
            detail += ("  moved: " + ", ".join(moved[:6])
                       + (" ..." if len(moved) > 6 else ""))
        if a_detail.get("bytes") is not None or b_detail.get("bytes") is not None:
            detail += (f"  [{a_detail.get('bytes', '-')} -> "
                       f"{b_detail.get('bytes', '-')} bytes]")
        lines.append(f"{row['status']:>7}  {row['position']}: "
                     f"{_short(row['a'])} -> {_short(row['b'])}{detail}")
    return "\n".join(lines)
