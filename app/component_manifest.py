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
* The store lives OUTSIDE every git-tracked tree — `~/.local/state/lloyd-
  request-manifests` by default (`$LLOYD_MANIFEST_STORE` or
  `harness.component_manifest.store_dir` override it;
  `tests/test_component_manifest.py::test_the_store_resolves_outside_every_git_tree`
  refuses to accept a path inside one). Manifest lines carry digests of vault
  text and of user messages, and a length plus a hash is still a fingerprint of
  confidential content, so the store inherits the vault's confidentiality
  boundary. `POLICY.md`, holding this policy, is written into the store root on
  first use so the bytes carry their own rules.

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
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
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
    "\n"
    "The digests are of vault text, email bodies and user messages, and a length\n"
    "plus a hash is still a fingerprint of confidential content. This directory is\n"
    "outside every git-tracked tree; that is necessary and not sufficient - treat\n"
    "it with the same confidentiality as ~/obsidian.\n"
)
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
          "lines_written": 0, "bytes_written": 0}


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


def _writer_loop() -> None:
    while True:
        item = _queue.get()
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
    if not (root / POLICY_FILENAME).exists():
        root.mkdir(parents=True, exist_ok=True)
        (root / POLICY_FILENAME).write_text(RETENTION_POLICY, encoding="utf-8")


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
        line = _build_line(base_url=base_url, model=model, payload=payload,
                           session_id=session_id, iteration=iteration,
                           send_site=send_site)
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
