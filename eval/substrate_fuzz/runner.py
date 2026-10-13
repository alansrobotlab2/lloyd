#!/usr/bin/env python3
"""Arm A: drive every corpus task through the real dispatch seam, with no model in the loop.

One task's route, in order, is the thing this module exists to test. The corpus row's steps
reach :func:`app.harness.loop._execute_tool_call` as a `tc` dict; the seam parses the arguments
into a dict, applies the tool sandbox, hands the call to a pool, flattens what came back with
the harness's own `_flatten_result`, measures `raw_chars`, runs
`fallback_for_empty_result`, runs `maybe_spill`, and builds the `tool_result` event whose
``content`` is the text that would go into ``messages``. Arm A reads that event. It never
re-implements any step in that chain, and there is no completion call anywhere in it: the
model name on the :class:`RunOptions` it passes is a sentinel that no server would answer.

Two measurements per step, kept apart on purpose:

* the **tool's own bytes** — the text the pool handed back before the harness shaped it;
* the **shown bytes** — the text the event carries, which is what a model would read.

For an ordinary result those are the same bytes, and saying so is worth nothing. They differ
exactly where this instrument is pointed: a result over `SPILL_THRESHOLD_CHARS` (50,000 on this
box) is replaced by a `<persisted-output>` pointer, and a control task's seeded break lands
between the two. That is Orellana's first reported failure — an image uploaded and downloaded
came back with different bytes, "the encoding layer at the MCP layer was incorrect" — in a form
this box can reproduce on purpose and byte-compare.

So every failure gets one layer from ``{model, interface, host, harness}`` and the evidence
that chose it, with no `unknown` bucket to park an awkward case in. An `interface` verdict
requires the two digests to be present in the record: without them the claim is not
provable, and this module refuses to emit it rather than downgrade it to a layer nobody
proved. Where the tool's side was never captured, every tool-side field reads
:data:`NOT_MEASURED` instead of borrowing the shown bytes — a field that defaults to the other
side of the comparison would make the requirement impossible to fail.

Run it headless (no GPU, no cloud, one box):

    cd ~/lloyd && .venvs/lloyd/bin/python -m eval.substrate_fuzz.runner
    cd ~/lloyd && .venvs/lloyd/bin/python -m eval.substrate_fuzz.runner --domains size_spill
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import shutil
import sys
import tempfile
import time
import types
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx  # noqa: E402  (the module arm A would use for anything network-shaped)

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app.harness.errors import ToolDispatchError  # noqa: E402
from app.harness.loop import _execute_tool_call  # noqa: E402
from app.harness.mcp_pool import _flatten_result  # noqa: E402
from app.harness.options import RunOptions  # noqa: E402
from app.harness.protected_paths import referenced_paths, write_deny_reason  # noqa: E402
from app.harness.tool_result_spill import PERSISTED_OUTPUT_TAG  # noqa: E402
from app.paths import DATA_ROOT, EVAL_SUBSTRATE_FUZZ_DIR, SESSIONS_DIR  # noqa: E402
from eval.substrate_fuzz.corpus import (  # noqa: E402
    LAYERS, CorpusError, apply_control, load_corpus, sha,
    _break_bytes as _corpus_break_bytes,
)

__all__ = ["NO_MODEL", "NOT_MEASURED", "LAYERS", "StepResult", "AttributionError",
           "ProtectedTarget", "attribute", "check_postcondition", "run_task", "run_corpus",
           "write_artifact", "prune_artifacts", "artifact_path", "scratch_root",
           "assert_confined", "main"]

#: The model field of the :class:`RunOptions` arm A passes. It is not a model: it is a string
#: no endpoint would recognise, so a leak into a completion would return an error rather than
#: burn a GPU, and `test_the_runner_uses_the_real_seam_and_not_a_stub_of_it` reads this field
#: to prove that no model reached the loop.
NO_MODEL = "substrate-fuzz-arm-a-no-model"

#: Every tool-side field of a step whose tool response was not captured. Never the shown
#: bytes, never an empty digest: the record has to be able to say it did not look.
NOT_MEASURED = "not measured"

_ENV_ROOT = "SUBFUZZ_ROOT"

# ── scratch roots ───────────────────────────────────────────────────────────

#: Where a run's scratch tree goes when the caller does not name one. `DATA_ROOT` when it is
#: a real data root; when `app.paths` is anchored to a git worktree (an automod round's tree,
#: where `DATA_ROOT` resolves inside the code tree) it is the system temp dir instead, because
#: an instrument that writes its scratch into the tree under review dirties the thing it is
#: reviewing. `$HOME` is never a root: a scratch directory under `$HOME` is one rename from
#: the vault.
def _runs_base() -> Path:
    env = os.environ.get(_ENV_ROOT)
    if env:
        return Path(env)
    base = Path(DATA_ROOT)
    try:
        base.relative_to(_REPO.resolve())
    except ValueError:
        pass
    else:
        base = Path(tempfile.gettempdir()) / "lloyd-substrate-fuzz"
    return base / "eval" / "substrate-fuzz" / "runs"


def scratch_root(task_id: str = "task") -> Path:
    """A private, empty, removable scratch root for one task.

    Refuses to build one inside a protected tree or at `$HOME` itself. The caller that wants
    to inspect the tree afterwards passes its own root to :func:`run_task` instead and owns
    its cleanup.
    """
    base = _runs_base()
    base.mkdir(parents=True, exist_ok=True)
    resolved = base.resolve()
    deny = write_deny_reason(str(resolved))
    if deny:
        raise ProtectedTarget(f"{base}: the scratch root itself is not a safe place to "
                              f"write ({deny})")
    if resolved == Path.home().resolve():
        raise ProtectedTarget(f"{base}: a scratch root at $HOME is one rename from the vault")
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)[:60] or "task"
    return Path(tempfile.mkdtemp(prefix=f"{safe}-", dir=str(base)))


def _new_session_id(task_id: str) -> str:
    """A session id for the spill sidecar, so a run's spill files are one known directory.

    It is a real session id shape rather than a name that collides with a live session,
    because `maybe_spill` writes under `SESSIONS_DIR` no matter who called it, and arm A
    removes that directory by name at the end of the task.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)[:60] or "task"
    return f"substrate-fuzz-{safe}-{secrets.token_hex(4)}"


# ── the guard ───────────────────────────────────────────────────────────────

class ProtectedTarget(RuntimeError):
    """A task's own target resolves into a tree arm A must not write into. Raised before the
    step is dispatched, never after it ran."""


class AttributionError(RuntimeError):
    """A failure this module is not allowed to report. Always names what is missing, and the
    record of a run that hits it is a defect against the runner, never a layer."""


_TOOL_KEY = "_substrate_fuzz_tool"

_PATH_ARGS = {
    "write": ("file_path", "path", "notebook_path", "cwd", "output_path"),
    "read": ("file_path", "path", "notebook_path", "cwd"),
}


def tool_of(args: dict | None) -> str:
    """The tool an argument bundle belongs to, as recorded by the caller.

    The guard is handed a step's args, not its step, so the tool name travels in the bundle
    under a reserved key the seam never sees: `Bash` is the only tool whose destinations are
    inside a string, and reaching them needs the extractor rather than a key lookup."""
    return str((args or {}).get(_TOOL_KEY, ""))


#: The four trees the item names, spelled the way the item spells them. Kept beside the
#: guard rather than imported because the harness's own table is not this list: it polices
#: the vault, the harness checkout and the interpreter homes, and does not name
#: `~/.openclaw` at all. A guard that quietly inherits a narrower list than its contract
#: reports "refused" for the shapes it happens to share and dispatches the rest.
_ITEM_ROOTS = {name: Path(name).expanduser() for name in (
    "~/obsidian", "~/lloyd", "~/.openclaw", "~/.venvs",
)}


def assert_confined(task, root: Path, args: dict | None = None,
                    fixture_rel: str | None = None) -> None:
    """Refuse, before dispatch, any target of this task that resolves into a protected tree.

    The list is the item's, verbatim: `~/obsidian`, `~/lloyd`, `~/.openclaw`, `~/.venvs`. The
    reasons come from :mod:`app.harness.protected_paths`, the same table the harness gates
    writes with — a second list would be a second truth to keep in step, and the second one
    always drifts.

    Relative targets are resolved against the run root, which is where a corpus row's relative
    names actually land. Symlinks are deliberately not resolved here: resolving them would
    refuse an in-root symlink the corpus *plants on purpose* to ask whether an escape works,
    and the answer to that question is the substrate's, not this guard's.

    Anything not named by one of these arguments — a path inside a `Bash` command string, a
    `Grep` pattern — is the tool sandbox's business, and arm A reports what the sandbox
    decided instead of pre-empting it.
    """
    checks: list[tuple[str, Path]] = []
    for item in task.setup:
        checks.append(("setup", root / str(item["path"])))
    if fixture_rel:
        checks.append(("fixture", root / fixture_rel))
    if task.post.kind in ("file_sha256", "file_length", "file_exists") and task.post.path:
        checks.append(("postcondition", root / str(task.post.path)))
    if tool_of(args) == "Bash" and isinstance(args.get("command"), str):
        for raw in referenced_paths(str(args["command"]), cwd=str(root)):
            expanded = str(Path(str(raw)).expanduser())
            checks.append(("bash-operand",
                           Path(expanded) if expanded.startswith("/") else root / expanded))
    # The table is keyed by TOOL and holds the argument names, so the lookup goes
    # tool -> keys. Iterating the table's keys and testing `key not in args` reads as a
    # per-argument check while comparing the literal strings "write"/"read" against the
    # step's argument names, which no step ever carries: the branch then ran for nothing
    # and the guard's own positive control passed by being dead. What it let through was
    # measured, not theorised — a run of `tests/test_substrate_fuzz_safety.py` on the
    # previous round's head created `~/lloyd/app/planted.py` and
    # `~/obsidian/knowledge/planted.md` for real, because the harness's own write
    # deny-set does not cover `~/lloyd/app` and this guard was the only thing in front
    # of the dispatch.
    modes = _PATH_ARGS.get(tool_of(args).lower(), ())
    for key in modes:
        if not args or not isinstance(args.get(key), str):
            continue
        raw = str(args[key])
        # Expand BEFORE the join. `Path.expanduser` only acts on a leading `~`, and joining
        # first buries it mid-path, where it stays literal: the guard then resolves
        # `~/obsidian/x` to a directory under the run root, finds nothing protected, and
        # dispatches the write it was written to refuse.
        expanded = str(Path(raw).expanduser())
        target = Path(expanded) if expanded.startswith("/") else root / expanded
        checks.append((f"arg:{key}", target))
    for origin, target in checks:
        # Expand first, then the table. `~` is how a person — and this corpus row — names a
        # protected tree, and `write_deny_reason` compares against absolute roots, so an
        # unexpanded `~/obsidian/...` slides through a guard that reads as checking the home
        # directory. Relative names resolve against the run root, which is where they land.
        expanded = Path(str(target)).expanduser()
        absolute = expanded if str(expanded).startswith("/") else root / expanded
        deny = write_deny_reason(str(absolute))
        if deny:
            raise ProtectedTarget(f"{task.id}: {origin} target {target} refused before "
                                  f"dispatch — {deny}")
        # The table and this item's list are not the same list, and the item's is the one the
        # acceptance names. `PROTECTED_WRITE_ROOTS` covers the vault, the harness and the
        # interpreter homes but not `~/.openclaw`; a target landing in a root the harness does
        # not police is still a target this instrument was told to refuse. Checked by
        # containment rather than by string match so a sibling like `~/lloyd-work` cannot
        # shadow `~/lloyd` in either direction.
        resolved = absolute if absolute.is_absolute() else Path.cwd() / absolute
        for name, blocked in _ITEM_ROOTS.items():
            if resolved == blocked or blocked in resolved.parents:
                raise ProtectedTarget(
                    f"{task.id}: {origin} target {target} resolves inside {name} "
                    f"({blocked}), which arm A never writes into")


# ── the pool arm A hands the seam ───────────────────────────────────────────

_AGGR = None


def _aggregator():
    """`agent_mcp.main`, imported once, lazily.

    The aggregator's own `call_tool` is what resolves a tool name to its handler, and it is
    the same dispatcher a live turn reaches through the pool, so the handler, its sandbox and
    its argument checks are production's. Imported lazily because importing it logs about
    missing MCP bridges, and a module that prints warnings on import makes every test noisy.
    """
    global _AGGR
    if _AGGR is None:
        import agent_mcp.main as agg
        _AGGR = agg
    return _AGGR


def _broken_result(text: str) -> Any:
    """A control's broken response, in the shape the aggregator's own SDK returns.

    `_flatten_result` reads `getattr(item, "text")` off the content blocks, so a namespace
    with a `text` is what the harness will flatten without noticing the difference — which is
    the point: the break enters as bytes at the boundary, not as a patched expectation."""
    return types.SimpleNamespace(
        content=[types.SimpleNamespace(type="text", text=text)],
        isError=False, is_error=False)


class _RecordingPool:
    """The pool the seam calls: it serves the real tool and keeps both sides of the answer.

    `call_tool` is the seam's one process boundary — production's pool reaches an MCP server
    over stdio or the in-process aggregator, and this reaches the same in-process aggregator —
    so the text returned here is the text `_flatten_result` will flatten. Capturing it is what
    makes an `interface` verdict a comparison instead of an opinion.

    A control's seeded break is applied to that returned text and nothing else: not to the
    file on disk, not to the task's expectation, not to the runner's bookkeeping. The tool
    has already run and its effects are on disk by then; what the break changes is the payload
    on its way to the model, which is precisely where the fault the corpus is looking for
    lives. `control_fired` says the break was applied to a response, so a control whose break
    could not apply to the payload it got is reported as dark rather than as a pass.
    """

    #: Tests flip this on the class to make the tool's side uncapturable, which is what a tool
    #: whose response object cannot be inspected looks like from the harness's side. The
    #: clause-4 refusal and the `not measured` fields are only testable because it is one
    #: switch and not a deleted line.
    record_tool_side = True

    def __init__(self, control: tuple[str, str] | None = None,
                 meta: dict[str, Any] | None = None) -> None:
        self.control_tool, self.control_kind = control if control else (None, None)
        self.meta = dict(meta or {})
        self.control_fired: set[str] = set()
        self.control_failed: set[str] = set()
        self.last_body: str | None = None
        self.last_is_error: bool | None = None
        self.calls: list[str] = []

    def has(self, name: str) -> bool:
        return True

    async def call_tool(self, name: str, arguments: dict, **kwargs: Any) -> Any:
        self.calls.append(name)
        meta = dict(self.meta)
        meta.update(kwargs.get("meta") or {})
        result = await _aggregator().call_tool(name, dict(arguments), meta)
        # The pool flattens in production, so the pool flattens here: `MCPPool.call_tool`
        # hands the seam `{"content": str, "is_error": bool}` built by this same function,
        # and the seam's `raw_chars = len(content)` is a length of THAT text.
        flat = _flatten_result(result)
        body = flat.get("content")
        self.last_is_error = bool(flat.get("is_error"))
        self.last_body = body if isinstance(body, str) else None
        if self.control_tool and name == self.control_tool and self.last_body is not None:
            broken = apply_control(str(self.control_kind), self.last_body)
            if broken is None:
                self.control_failed.add(name)
            else:
                self.control_fired.add(name)
                return {"content": broken, "is_error": self.last_is_error}
        return flat


def _tool_side_recorded() -> bool:
    """Whether the tool's own bytes are being captured this run.

    Two spellings, one switch. The class attribute is the declaration; the dotted string
    is how a reader outside this module names it — `getattr(runner, "_RecordingPool.record_tool_side")`
    — and since `monkeypatch` resolves a dotted name against the object it was handed, the
    module answers that spelling here rather than letting the patch land nowhere. A run with
    the switch off is what a tool whose response object cannot be inspected looks like from
    the harness's side, which is the only way to test that the runner says `not measured`
    instead of quietly borrowing the shown bytes for the tool's side of the comparison.
    """
    flag = globals().get(_FLAG_NAME)
    return _RecordingPool.record_tool_side if flag is None else bool(flag)


def __getattr__(name: str) -> Any:
    """Answer the dotted spelling of the recording switch (see `_tool_side_recorded`)."""
    if name == _FLAG_NAME:
        return _RecordingPool.record_tool_side
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_FLAG_NAME = "_RecordingPool.record_tool_side"


def _break_bytes(payload: bytes, kind: str) -> bytes | None:
    """The seeded break at the byte level, through the corpus's one implementation.

    `corpus._break_bytes` takes the break name first; this takes the payload first, because
    a caller holding bytes should not have to know the vocabulary's argument order. It is a
    re-export, not a second implementation: an injection written here and an expectation
    written there drift apart silently, and the control then reports a break nobody injected.
    """
    return _corpus_break_bytes(kind, payload)


# ── steps ───────────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """One dispatched step, with the tool's bytes and the shown bytes kept apart.

    `shown_text` is the `content` field of the `tool_result` event: the text that would be
    appended to `messages`. `effective_bytes` is what a reader of that text would end up
    holding once a spill pointer is followed, which is the payload a postcondition compares
    against. `tool_bytes` is what the tool answered first. The three are equal for an ordinary
    result and differ only for the two shapes arm A exists to catch: a clipped result and a
    broken payload.
    """

    tool: str
    args: dict
    ok: bool
    raw_chars: int | None
    shown_text: str
    tool_bytes: bytes | None
    spill_path: Path | None = None
    spill_bytes: bytes | None = None
    spill_resolved: bool | None = None
    error_text: str = ""
    duration_ms: float = 0.0
    control: str | None = None
    control_fired: bool = False
    control_failed: bool = False
    error_class: str | None = None

    @property
    def tool_recorded(self) -> bool:
        return self.tool_bytes is not None

    @property
    def tool_sha256(self) -> str:
        return NOT_MEASURED if self.tool_bytes is None else sha(self.tool_bytes)

    @property
    def shown_text_sha256(self) -> str:
        return sha(self.shown_bytes)

    @property
    def shown_bytes(self) -> bytes:
        return self.shown_text.encode("utf-8", "surrogateescape")

    @property
    def effective_bytes(self) -> bytes:
        """What the model would actually have: the resolved result if there was one, else the
        shown text. On a spilled step whose pointer does not resolve, this is the pointer —
        and a postcondition that compares against it fails, which is the correct verdict."""
        if self.spill_bytes is not None:
            return self.spill_bytes
        return self.shown_bytes

    @property
    def shown_sha256(self) -> str:
        return sha(self.effective_bytes)

    @property
    def shown_truncated(self) -> bool:
        """Whether the shown text is a stand-in rather than the payload: a spill pointer, or
        the harness's own truncation marker."""
        return (PERSISTED_OUTPUT_TAG in self.shown_text
                or "[... output truncated" in self.shown_text)

    def head(self, data: bytes) -> str:
        """The first 64 bytes, escaped, for a human reading a verdict beside its evidence."""
        return data[:64].decode("utf-8", "replace").replace("\x00", "\\0")[:80]

    @property
    def tool_head(self) -> str:
        return NOT_MEASURED if self.tool_bytes is None else self.head(self.tool_bytes)

    @property
    def shown_head(self) -> str:
        return self.head(self.effective_bytes)

    @property
    def interface_pair(self) -> dict:
        """The two sides of the comparison an `interface` verdict is allowed to rest on.

        Emitted only when both digests exist: :func:`attribute` refuses the verdict otherwise,
        so a record carrying this pair is a record whose claim a reader can re-check with two
        `sha256sum` runs.
        """
        return {
            "tool_sha256": self.tool_sha256,
            "shown_sha256": self.shown_sha256,
            "tool_head": self.tool_head,
            "shown_head": self.shown_head,
            "raw_chars": self.raw_chars,
            "spill_path": None if self.spill_path is None else str(self.spill_path),
            "spill_resolved": self.spill_resolved,
        }


_SAVED_TO = re.compile(r"Full output saved to: (\S+)")


def _spill_resolution(shown_text: str, recorded_body: str | None,
                      ok: bool) -> tuple[Path | None, bytes | None, bool | None]:
    """Follow the harness's own pointer, by the path the pointer states.

    `maybe_spill` writes the full result to a file and puts that file's absolute path inside
    the pointer text (`Full output saved to: /…/<call-id>.txt`). Reading it back is therefore
    a claim the shown text itself makes and this function can check, with no assumption about
    how the path is named — which is the whole reason the resolution is worth having: an
    instrument that *derived* the path from the session id would confirm a sidecar exists even
    if the pointer named somewhere else, and pointers are exactly where #1052 went wrong.

    Returns `(path, bytes, resolved)`. `resolved` is None for a step that was never spilled,
    so the record cannot say `false` about a pointer that was never offered.
    """
    if not ok or not isinstance(shown_text, str) or PERSISTED_OUTPUT_TAG not in shown_text:
        return None, None, None
    if recorded_body is None:
        # Spilled, and the full text is unknown because the tool's side was not captured.
        # Still resolve the pointer the harness printed, so the artifact can name the file.
        pass
    match = _SAVED_TO.search(shown_text)
    if not match:
        return None, None, False
    path = Path(match.group(1))
    try:
        data = path.read_bytes()
    except OSError:
        return path, None, False
    return path, data, True


_SAVE_KEYS = ("file_path", "path")


def _declared_save(step: dict) -> tuple[Path | None, str | None]:
    """A `Write` step's target and its `content`, when it declares both.

    A stored writer's file survives a response-level break untouched, so a control on such a
    step can be graded honestly two ways: same digest means the file is byte-identical while
    the payload is not, which is an interface fault; a different digest means the file itself
    changed, which is not an interface fault. Deciding that needs the target, and only the
    corpus row knows it."""
    args = step.get("args") or {}
    for key in _SAVE_KEYS:
        target = args.get(key)
        if isinstance(target, str) and isinstance(args.get("content"), str):
            return Path(target) if target.startswith("/") else None, args["content"]
    return None, None


def _sub(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        for token, text in mapping.items():
            value = value.replace(token, text)
        return value
    if isinstance(value, dict):
        return {k: _sub(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_sub(v, mapping) for v in value]
    return value


def _args_for(task, step: dict, root: Path) -> tuple[str, dict, Path | None]:
    """The tool name, its arguments with `{root}` and `{fixture}` substituted, and its target.

    Substitution is on purpose: a corpus row that carried a machine-specific absolute path
    would be a row that could not run on another box, and a row that relied on an implicit
    working directory would be testing whatever directory happened to be current."""
    fixture_rel = next((str(item["path"]) for item in task.setup if item.get("fixture")), None)
    mapping = {"{root}": str(root)}
    if fixture_rel:
        mapping["{fixture}"] = str(root / fixture_rel)
    args = _sub(dict(step.get("args") or {}), mapping)
    target: Path | None = None
    for key in _SAVE_KEYS:
        if isinstance(args.get(key), str):
            raw = args[key]
            target = Path(raw) if raw.startswith("/") else root / raw
            break
    return str(step["tool"]), args, target


def _run_step(task, step: dict, root: Path, session_id: str,
              control: tuple[str, str] | None, index: int) -> StepResult:
    """One step, through the seam, synchronously.

    `asyncio.run` per step rather than one loop per task: a tool that hangs then costs one
    step and its traceback names the step, where one loop over a whole task would put the
    traceback inside whichever step happened to be awaiting.
    """
    tool, args, target = _args_for(task, step, root)
    call_id = f"s{index}"
    pool = _RecordingPool(control=control,
                          meta={"lloyd/session_id": session_id, "lloyd/call_id": call_id})
    # No `cwd` on the options: `RunOptions` has no such field, and the seam does not
    # consult one — the working directory a step runs in is the `cwd` argument the corpus
    # row puts in the tool's own arguments, which is where the model sets it too.
    options = RunOptions(model=NO_MODEL, max_turns=1, session_id=session_id)
    # The seam reads `tc["_args_dict"]` (the already-parsed arguments the loop put there at
    # 3155 for its own retry path) and `tc["function"]["arguments"]` as a JSON string, and it
    # takes the session correlation out of `_meta` rather than the arguments. All three are
    # supplied exactly as the loop's `_call_to_tc` leaves them: the arguments travel as text,
    # so a corpus row cannot smuggle a key past the inputSchema the way a dict would.
    tc = {"id": call_id, "type": "function",
          "function": {"name": tool, "arguments": json.dumps(args)},
          "_args_dict": dict(args),
          "_meta": {"lloyd/session_id": session_id, "lloyd/call_id": call_id},
          "lloyd/call_id": call_id}
    t0 = time.perf_counter()
    event = asyncio.run(_execute_tool_call(tc=tc, pool=pool, options=options,
                                          session_id=session_id))
    duration = (time.perf_counter() - t0) * 1000.0
    shown = event.get("content")
    shown_text = shown if isinstance(shown, str) else json.dumps(shown, sort_keys=True)
    recorded = pool.last_body if _tool_side_recorded() else None
    tool_bytes: bytes | None = (None if recorded is None
                                else recorded.encode("utf-8", "surrogateescape"))
    spill_path, spill_bytes, resolved = _spill_resolution(shown_text, pool.last_body,
                                                          not event.get("is_error"))
    return StepResult(
        tool=tool, args=args, ok=not event.get("is_error"),
        raw_chars=event.get("raw_chars"), shown_text=shown_text, tool_bytes=tool_bytes,
        spill_path=spill_path, spill_bytes=spill_bytes, spill_resolved=resolved,
        error_text="" if not event.get("is_error") else shown_text,
        duration_ms=duration, control=task.control if control else None,
        control_fired=tool in pool.control_fired, control_failed=tool in pool.control_failed,
        error_class=event.get("error_class"))


# ── the postcondition ───────────────────────────────────────────────────────

def check_postcondition(task, root: Path, step: StepResult | None
                        ) -> tuple[bool, str]:
    """The task's mechanical claim, decided with no judgement and no model.

    Four things are ever compared: a digest, a byte length, an exit shape, and whether a path
    exists under the run root. A kind the runner does not implement raises rather than passing:
    a decoder that does not know its own postcondition is the #644 defect class, and it is the
    only way a green board can arrive by accident here.
    """
    post = task.post
    if step is None:
        return False, "no step ran, so the postcondition was never measured"
    kind = post.kind

    if kind == "exit_shape":
        want = bool(post.is_error)
        got = not step.ok
        return (want == got,
                f"is_error={got} (expected {want})"
                + (f"; error_text={step.error_text[:200]!r}" if got else ""))

    if kind in ("file_exists", "file_length", "file_sha256"):
        target = root / str(post.path)
        if not target.exists():
            return False, f"{post.path}: no such file under the run root {root}"
        if kind == "file_exists":
            return True, f"{post.path} exists ({target.stat().st_size} bytes)"
        data = target.read_bytes()
        if kind == "file_length":
            want = int(post.expected)
            return (len(data) == want,
                    f"{post.path}: {len(data)} bytes on disk, expected {want} "
                    f"(diff {len(data) - want:+d})")
        got = sha(data)
        return (got == post.expected,
                f"{post.path}: sha256 {got} (expected {post.expected}; {len(data)} bytes)")

    if kind in ("stdout_sha256", "stdout_length"):
        data = step.effective_bytes
        if kind == "stdout_length":
            want = int(post.expected)
            return (len(data) == want,
                    f"payload is {len(data)} bytes, expected {want} (diff {len(data) - want:+d}; "
                    f"raw_chars={step.raw_chars}, spill_resolved={step.spill_resolved})")
        got = sha(data)
        return (got == post.expected,
                f"payload sha256 {got} (expected {post.expected}; {len(data)} bytes, "
                f"raw_chars={step.raw_chars}, spill_resolved={step.spill_resolved})")

    if kind == "shown_contains":
        needle = str(post.expected)
        if needle in step.shown_text:
            return True, f"needle {needle[:40]!r} is in the shown text"
        if step.spill_bytes is not None and needle in step.spill_bytes.decode(
                "utf-8", "surrogateescape"):
            # The needle is in the result but not in what the model was shown. For a task whose
            # claim is about the pointer that is the pointer working; for a task whose claim
            # was that the answer is visible, it is the clipping. The row says which it is by
            # what it asks for, so the check stays mechanical either way.
            return (bool(step.spill_resolved),
                    f"needle {needle[:40]!r} is in the resolved result "
                    f"({len(step.spill_bytes)} bytes at "
                    f"{step.spill_path.name if step.spill_path else '?'}) but not in the "
                    f"{len(step.shown_bytes)} shown bytes; pointer resolved="
                    f"{step.spill_resolved}")
        return False, (f"needle {needle[:40]!r} absent from the shown text "
                       f"({len(step.shown_bytes)} bytes) and from any resolved result")

    raise CorpusError(f"{task.id}: postcondition kind {kind!r} is not implemented by the "
                      "runner, which would report it green from a decoder that does not know "
                      "it — the #644 defect class")


# ── attribution ─────────────────────────────────────────────────────────────

_HOST_MARKERS = ("read-only file system", "permission denied", "operation not permitted",
                 "no space left", "disk quota", "text file busy", "too many open files",
                 "resource temporarily unavailable", "cannot allocate memory", "memoryerror",
                 "argument list too long", "signal 9", "signal 11", "exit code 137",
                 "command not found", "no such file or directory")
_HARNESS_MARKERS = ("tool call denied", "blocked until", "protected path", "not permitted for",
                    "denied by policy", "sandbox", "not available on this box",
                    "is not allowed")
_INTERFACE_MARKERS = ("codec can't decode", "surrogate", "unicodeescape", "json parse",
                      "invalid start character", "expecting value", "unexpected end of data")


def _marker(text: str, markers: tuple[str, ...]) -> str | None:
    low = (text or "").lower()
    for m in markers:
        if m in low:
            return m
    return None


def _expected_limit(task, step: StepResult) -> bool:
    """Whether a red here is a substrate doing the right thing rather than a defect.

    Two shapes qualify, and only these. A task that declares `host` among its expected layers
    and asks for an error: refusing a read-only write and reporting `is_error` is the correct
    answer from a substrate with edges. A step that returned an error the tool sandbox raised
    while the task's claim is about a file the step was supposed to create: the sandbox
    refusing is also a correct answer, and the layer field still says which guard refused it.
    Anything else that goes red is a defect, including a step that refused when the row said
    it would work — that is a substrate that changed under the corpus, and the item's own rule
    is that a limit is a finding, not an excuse.
    """
    if not step.ok and task.post.kind == "exit_shape" and "host" in task.expected_layers:
        return True
    if (not step.ok and task.post.kind in ("file_sha256", "file_length", "file_exists")
            and _marker(step.error_text, _HARNESS_MARKERS) is not None):
        return True
    return False


def attribute(task, step: StepResult, *, root: Path | None = None) -> dict:
    """One layer from the fixed vocabulary, the rule that chose it, and its evidence.

    Rules run in order and the first match wins, quoting the measurement it read:

    1. the tool's own bytes differ from the bytes that would reach `messages` — that
       disagreement is by construction the boundary between the two, so it is `interface`;
    2. the tool's answer carries a guard's own refusal text — `harness`;
    3. the tool's answer carries a syscall-level error and the file it was supposed to write
       is not on disk — `host`;
    4. the tool's answer carries a syscall-level error while the file it claims is present —
       `host` as well, because the kernel spoke;
    5. the payload does not match the corpus's own digest and the answer arrived without
       error — `model`, in the corpus's sense: the bytes handed to the model are not the
       bytes the task was authored against;
    6. nothing matched — :class:`AttributionError`, which is the runner's defect and is
       reported as one, never an `unknown` bucket.

    An `interface` verdict requires the byte pair. A row that declares `interface` as its only
    expected layer and whose tool side was never captured raises too: the only layer that row
    can prove is the one nobody measured, and silently filing it under some other layer is how
    a proven interface break and an unmeasurable one come to read the same in the artifact.
    """
    pair = step.interface_pair
    wants_interface = list(task.expected_layers) == ["interface"]
    if wants_interface and not step.tool_recorded:
        raise AttributionError(
            f"{task.id}: an interface verdict requires the tool's own bytes beside the bytes "
            f"that would reach messages, and the tool side is {NOT_MEASURED}; the row declares "
            "interface as its only expected layer, so there is no other verdict its evidence "
            "could support")

    if task.is_control and step.tool_recorded and step.tool_bytes != step.effective_bytes:
        same_on_disk = "not compared"
        if root is not None:
            save_target, save_content = _declared_save({"tool": step.tool, "args": step.args})
            if save_target is not None and save_content is not None:
                want = sha(save_content.encode("utf-8", "surrogateescape"))
                try:
                    same_on_disk = (sha(save_target.read_bytes()) == want)
                except OSError:
                    same_on_disk = "unreadable"
        return {"layer": "interface", "rule": "1: tool bytes differ from shown bytes",
                "evidence": (f"tool sha256 {step.tool_sha256} != shown sha256 "
                             f"{step.shown_sha256} ({len(step.tool_bytes or b'')} bytes "
                             f"answered, {len(step.effective_bytes)} bytes shown, "
                             f"raw_chars={step.raw_chars}, file unchanged={same_on_disk})"),
                "interface_pair": pair}

    refused = step.error_text
    if refused:
        hit = _marker(refused, _HARNESS_MARKERS)
        if hit:
            return {"layer": "harness", "rule": f"2: refusal text names a guard ({hit!r})",
                    "evidence": refused[:400], "interface_pair": pair}
        hit = _marker(refused, _HOST_MARKERS)
        if hit:
            on_disk = None
            if root is not None and task.post.kind in ("file_sha256", "file_length",
                                                       "file_exists"):
                on_disk = (root / str(task.post.path)).exists()
            return {"layer": "host",
                    "rule": f"3/4: syscall-level error text ({hit!r}); "
                            f"target present={on_disk}",
                    "evidence": refused[:400], "interface_pair": pair}
        hit = _marker(refused, _INTERFACE_MARKERS)
        if hit:
            if not step.tool_recorded:
                raise AttributionError(
                    f"{task.id}: the answer names a codec or parse fault ({hit!r}), which is "
                    f"an interface claim, but the tool side is {NOT_MEASURED} so the bytes "
                    "cannot be put beside the ones that reached messages")
            return {"layer": "interface",
                    "rule": f"3/4: codec or parse fault in the answer ({hit!r})",
                    "evidence": refused[:400], "interface_pair": pair}
        raise AttributionError(
            f"{task.id}: the tool answered with an error that names no guard, no syscall and "
            f"no codec, so its layer is not provable: {refused[:300]!r}")

    ok_, detail = check_postcondition(task, root if root is not None else Path.cwd(), step)
    if not ok_:
        return {"layer": "model", "rule": "5: the payload the model was given is not the "
                                          "payload the corpus row was authored against",
                "evidence": detail, "interface_pair": pair}
    raise AttributionError(f"{task.id}: asked to attribute a result that did not fail "
                           f"({detail}) — a record with a layer and no failure would let a "
                           "passing task look like a defect, so the runner is blamed instead")


# ── one task ────────────────────────────────────────────────────────────────

def _do_setup(root: Path, task, fixture_bytes: bytes | None) -> None:
    """Lay down what the row assumes exists, under the run root only, and check our own write.

    The digest re-check is arm A measuring itself, not the substrate: the row's bytes hashed
    to this value in the loader, so the only way to fail here is that the write did not land
    as asked — and every later comparison would then be measuring a fixture that was never
    on disk."""
    for item in task.setup:
        rel = str(item["path"])
        target = root / rel
        if item.get("dir"):
            target.mkdir(parents=True, exist_ok=True)
            if item.get("read_only"):
                os.chmod(target, 0o500)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if item.get("fixture"):
            if fixture_bytes is None:
                raise CorpusError(f"{task.id}: setup names a fixture the row does not carry")
            target.write_bytes(fixture_bytes)
        elif isinstance(item.get("content"), str):
            # surrogateescape both ways: a NUL or a lone surrogate IS the payload under test,
            # and a setup writer that rejected it would fail the task for the runner's own
            # encoder before any substrate had been touched.
            target.write_bytes(item["content"].encode("utf-8", "surrogateescape"))
        else:
            raise CorpusError(f"{task.id}: setup entry {rel!r} carries neither content nor a "
                              "fixture flag")
        if item.get("mode"):
            os.chmod(target, int(str(item["mode"]), 8))
        if item.get("sha256"):
            got = sha(target.read_bytes())
            if got != item["sha256"]:
                raise RuntimeError(
                    f"{task.id}: setup {rel!r} was written as {got} but the row declares "
                    f"{item['sha256']} — the bytes every expected value was computed from are "
                    "not on disk")


def _control_of(task) -> tuple[str, str] | None:
    """Which tool's payload this control breaks, and with what.

    The LAST step, because that is the step the postcondition reads: `check_postcondition`
    is handed the final `StepResult`, so a break applied to an earlier step's payload cannot
    redden the shipped claim no matter how faithfully it fires. The first shipped filesystem
    control was written that way — its base wrote a file and read it back, the break landed on
    the `Write` summary, and the digest of the read-back stayed green, which is a dead detector
    wearing a control's clothes.
    """
    if not task.is_control:
        return None
    return str(task.steps[-1]["tool"]), str(task.control)


def _control_break_verified(task, control: tuple[str, str] | None,
                            last: "StepResult | None") -> bool:
    """Did this control's break actually change the bytes the tool answered with?

    Judged on the payload the run OBSERVED — the tool's own recorded answer on the graded
    step — and nowhere else. The first version derived the payload from the postcondition
    instead: `bytes.fromhex(post.expected)` is the 32 bytes of the expected DIGEST, not the
    task's payload, so the break was applied to a hash nobody sent and the answer measured
    whether a digest happens to contain a NUL byte. It reported `verified` for controls whose
    break could not have fired and `not verified` for one that demonstrably reddened its
    postcondition — `byte_roundtrip controls 0 verified` on the full arm-A run, over a control
    that had just produced an `interface` verdict. Since that count is the clause-3 evidence a
    green board rests on ("0 failures with a live control is evidence; without one it is
    nothing"), a count that cannot be wrong is a green test asserting nothing — #644's exact
    objection, sitting inside the instrument built to answer it.

    `None` when no step ran, and `False` whenever the tool's side was never recorded: an
    unmeasured payload is reported as unverified rather than assumed to have been broken.
    """
    if not control or last is None:
        return False
    original = last.tool_bytes if last.tool_recorded else None
    if not original:
        return False
    broken = apply_control(control[1],
                           original.decode("utf-8", "surrogateescape"))
    if broken is None:
        return False
    return sha(broken.encode("utf-8", "surrogateescape")) != sha(original)


def _step_dict(sr: StepResult) -> dict:
    """A recorded step. Both sides of the comparison, by digest and by head, with an
    unmeasured side saying `not measured` in every tool-side field rather than quoting the
    shown text.

    The shown side is named twice because a spilled step has two shown texts: `shown_text` is
    the event's content verbatim (a pointer), and `shown` is what the model ends up holding
    once that pointer is followed. On an ordinary step the two digests are equal; on a spilled
    one they cannot be, and keeping them apart is why the spill round trip is checkable at all.
    """
    return {
        "tool": sr.tool, "args": sr.args, "ok": sr.ok,
        "duration_ms": round(sr.duration_ms, 1),
        "raw_chars": sr.raw_chars,
        "tool_recorded": sr.tool_recorded,
        "tool_sha256": sr.tool_sha256,
        "tool_bytes": sr.tool_sha256,
        "tool_head": (f"tool sha256 {sr.tool_sha256}" if sr.tool_recorded else NOT_MEASURED),
        "tool_repr": sr.tool_head,
        "shown_text": sr.shown_text_sha256,
        "shown_text_sha256": sr.shown_text_sha256,
        "shown_text_head": f"shown text sha256 {sr.shown_text_sha256}",
        "shown_text_repr": sr.head(sr.shown_bytes),
        "shown_sha256": sr.shown_sha256,
        "shown_head": f"shown sha256 {sr.shown_sha256}",
        "shown_repr": sr.shown_head,
        "shown_truncated": sr.shown_truncated,
        "spill": {"path": None if sr.spill_path is None else str(sr.spill_path),
                  "resolves": sr.spill_resolved,
                  "bytes": None if sr.spill_bytes is None else len(sr.spill_bytes)},
        "error_text": sr.error_text[:400],
        "error_class": sr.error_class,
        "control": sr.control, "control_fired": sr.control_fired,
        "control_failed": sr.control_failed,
    }


def _cleanup(root: Path, session_id: str, root_owned: bool, record: dict) -> None:
    """Remove what the run made, and record what removal found.

    The spill sidecar is removed by name, never by wildcard: it lives under the live sessions
    tree, and a glob aimed near it could reach a session this box is serving right now. A
    failed removal is written into the record, because "arm A leaves nothing behind" is the
    kind of claim that goes false quietly."""
    spill_dir = Path(SESSIONS_DIR) / f"{session_id}.tool-results"
    record["spill_removed"] = False
    if spill_dir.exists():
        try:
            shutil.rmtree(spill_dir)
            record["spill_removed"] = True
        except OSError as exc:
            record["spill_removal_error"] = str(exc)
    if root_owned:
        shutil.rmtree(root, ignore_errors=True)
        if root.exists():
            record["root_removal_error"] = f"{root} still exists"


def run_task(task, root: Path | None = None, keep_root: bool = False,
             *, fixture_bytes: bytes | None = "sentinel") -> dict:
    """One corpus task, driven end-to-end through the seam. Returns a record; never raises for
    a substrate reason.

    Two roots and two lifetimes. `root=None` builds a private scratch root that this call
    removes at the end: arm A writes files, spawns children and takes memory on a box that is
    serving a person, and a store its runs leave behind would itself need a retention rule. A
    root handed in by a caller is the caller's tree and nothing in it is removed — passing one
    is how a test reads the run-root files afterwards.

    `fixture_bytes=None` overrides the row's own fixture with no fixture at all, which is how
    a caller makes a fixture-reading task disagree with its expectation on purpose.

    Two outcomes are deliberately NOT returned as a verdict, and both are the instrument's
    own failure rather than the substrate's:

    * a control whose seeded break never applied to the payload it was given raises
      :class:`CorpusError` — the row is mis-authored (its step cannot carry the break), so
      the task is skipped and named, never counted in `k / N`;
    * a control whose break DID apply and whose postcondition still passed is recorded RED
      with `rule: "control stayed green"` and `layer: "harness"` — the one value of the four
      that owns this instrument, since clause 4 allows no `unknown` and a record with no layer
      is either a skip or a violation. It is not raised, because raising turns it into a skip,
      and a skip is excluded from `k / N` — which is precisely how a dead detector would
      disappear from the report while the domain still printed a number. The node that pins
      the difference is
      `tests/test_substrate_fuzz_controls.py::test_a_control_that_stays_green_is_a_test_failure`.

    :class:`AttributionError` propagates out of this function: it means the runner could not
    build a provable layer, and a caller that asked for one task's verdict should hear it,
    not read a layer nobody proved. :func:`run_corpus` catches it and files the task with
    `failure: null` under `unattributed`, because one unattributable row must not stop the
    other fifty-one — and that count is itself reported as a defect against this module.
    """
    fixture = task.fixture if fixture_bytes == "sentinel" else fixture_bytes
    root_owned = root is None
    root = Path(root) if root is not None else scratch_root(task.id)
    session_id = _new_session_id(task.id)
    control = _control_of(task)
    rec: dict[str, Any] = {
        "id": task.id, "task_id": task.id, "domain": task.domain,
        "is_control": task.is_control, "control_for": task.control_for,
        "control": task.control, "expected_layers": list(task.expected_layers),
        "run_root": str(root), "root_owned": root_owned, "session_id": session_id,
        "steps": [], "skipped": None, "ok": None, "evidence": "",
        "layer": None, "rule": None, "defect": None, "expected_limit": None,
        "interface_pair": None, "failure": None,
        # Filled in after the steps run: it is a judgement about the payload the run observed,
        # and before dispatch there is no such thing.
        "control_break_verified": False,
    }
    t0 = time.perf_counter()
    _rail_http(True)
    try:
        fixture_rel = next((str(i["path"]) for i in task.setup if i.get("fixture")), None)
        for step in task.steps:
            tool, args, _target = _args_for(task, step, root)
            assert_confined(task, root, {**args, _TOOL_KEY: tool}, fixture_rel)
        assert_confined(task, root, None, fixture_rel)
        _do_setup(root, task, fixture)
        last = None
        for index, step in enumerate(task.steps):
            sr = _run_step(task, step, root, session_id, control, index)
            rec["steps"].append(_step_dict(sr))
            last = sr
        if last is None:                                    # pragma: no cover — loader refuses
            raise CorpusError(f"{task.id}: no steps ran")
        rec["control_break_verified"] = _control_break_verified(task, control, last)
        if control and not any(s["control_fired"] for s in rec["steps"]):
            raise CorpusError(
                f"{task.id}: control {control[1]!r} was never applied to the {control[0]!r} "
                "response, so this domain's detector is dark and a green board would be a "
                "claim about a detector that never lit")
        ok, detail = check_postcondition(task, root, last)
        rec["ok"], rec["evidence"] = ok, detail
        expected_limit = _expected_limit(task, last)
        rec["expected_limit"] = expected_limit
        if ok:
            rec["defect"] = False
            if control:
                # The break was applied and the shipped postcondition still passed, so this
                # domain's detector cannot fail. That is a defect, and it is filed under the
                # one layer of the four that owns this instrument — `harness` — because the
                # vocabulary has no fifth value and clause 4 allows no `unknown`: a record
                # with no layer is either a skip (which is excluded from `k / N` and so
                # invisible) or a violation. `controls_verified` staying 0 for the domain is
                # the second reading of the same fact.
                rec["ok"] = False
                rec["defect"] = True
                rec["layer"] = "harness"
                rec["rule"] = "control stayed green"
                rec["evidence"] = (f"control {control[1]!r} left the shipped postcondition "
                                   "green, so this domain's detector cannot fail")
                _failure_from(rec, task)
        else:
            verdict = attribute(task, last, root=root)
            rec["layer"], rec["rule"] = verdict["layer"], verdict["rule"]
            rec["interface_pair"] = verdict["interface_pair"]
            rec["defect"] = not expected_limit
            _failure_from(rec, task)
    except ProtectedTarget as exc:
        rec["skipped"] = "protected-target"
        rec["evidence"] = str(exc)
    except AttributionError as exc:
        # The runner could not build a provable layer. It emits NO failure object rather than
        # one carrying a layer it did not prove — `validate_failure` is exactly the rail that
        # forbids a defect with no layer, and the record keeps `failure: null` so a reader
        # cannot mistake this for an attributed verdict. The id is what gets published, under
        # `unattributed` (clause 4: "Unattributable failures are a defect in the runner, not an
        # `unknown` bucket to park them in"), and :func:`run_corpus` turns the raise into that
        # record so one row cannot stop the run.
        rec["ok"] = False
        rec["layer"], rec["rule"] = None, "unattributable"
        rec["defect"] = True
        rec["evidence"] = str(exc)
        # The half-built record travels on the exception so a fleet run can book the row
        # without this module carrying a second copy of the record shape — two shapes drift,
        # and the drift would show up as a task that is in `failed_task_ids` but not in any
        # domain's tally.
        exc.record = rec                                  # type: ignore[attr-defined]
        raise
    except CorpusError as exc:
        rec["skipped"] = "corpus"
        rec["evidence"] = str(exc)
    except (ToolDispatchError, OSError, RuntimeError) as exc:                # noqa: BLE001
        rec["ok"] = False
        rec["layer"], rec["rule"] = "harness", "runner or dispatch exception"
        rec["defect"] = True
        rec["evidence"] = f"{type(exc).__name__}: {exc}"
        _failure_from(rec, task)
    finally:
        _rail_http(False)
        rec["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        _cleanup(root, session_id, root_owned and not keep_root, rec)
    return rec


def _failure_from(rec: dict, task) -> None:
    """The failure record: one layer, its rule, its evidence, and the row's own claim beside
    it. Every red record has one; a record whose layer is None is counted as `unattributed`,
    which is a defect in this module and is reported as one rather than parked."""
    rec["failure"] = {
        "task_id": rec["id"], "domain": rec["domain"], "layer": rec["layer"],
        "rule": rec["rule"], "evidence": rec["evidence"][:600],
        "defect": rec["defect"], "expected_limit": rec["expected_limit"],
        "expected_layers": list(task.expected_layers),
        "interface_pair": rec["interface_pair"],
    }


# ── the run ─────────────────────────────────────────────────────────────────

def run_corpus(tasks=None, *, domains=None, limit: int | None = None,
               keep_root: bool = False, root: Path | None = None) -> dict:
    """Arm A over the corpus (or over the rows a test hands in): per-domain `k / N`, every
    failing task id, one layer per failure, and the control tally that makes a green board
    mean something.

    `domains` restricts to named domains; `limit` caps the count and takes each sampled-out
    control's base with it, so a sampled run cannot report a control it did not run. `root`
    drives every task in one caller-owned directory instead of a private scratch root.
    """
    tasks = load_corpus() if tasks is None else list(tasks)
    if domains:
        wanted = set(domains)
        tasks = [t for t in tasks if t.domain in wanted]
    if limit:
        kept = {t.id for t in tasks[:limit]}
        tasks = [t for t in tasks[:limit] if not t.is_control or t.control_for in kept]
    started = time.time()
    roots = (Path(root) if root is not None else None)
    records: list[dict] = []
    for t in tasks:
        try:
            records.append(run_task(t, root=roots,
                                    keep_root=True if root is not None else keep_root))
        except AttributionError as exc:
            # One row the runner cannot prove must not stop the other fifty-one, and it must
            # not vanish either: the record `run_task` built on the way out is booked like any
            # other, and its id lands in `unattributed` because `layer` is null there. That
            # count is a defect against this module, not a bucket for hard cases.
            rec = getattr(exc, "record", None)
            if rec is None:                                # pragma: no cover — run_task sets it
                raise
            records.append(rec)

    domains_out: dict[str, dict] = {}
    per_domain: dict[str, str] = {}
    failed_task_ids: list[str] = []
    unattributed: list[str] = []
    layers = {layer: 0 for layer in LAYERS}
    skipped: list[str] = []
    verified = 0
    for r in records:
        d = domains_out.setdefault(r["domain"], {
            "n": 0, "passed": 0, "k": 0, "failing_task_ids": [], "controls": 0,
            "controls_red": 0, "controls_verified": 0, "skipped": 0})
        d["n"] += 1
        if r["is_control"]:
            d["controls"] += 1
            if r["control_break_verified"]:
                d["controls_verified"] += 1
                verified += 1
        if r["skipped"]:
            d["skipped"] += 1
            skipped.append(r["id"])
            continue
        if r["ok"]:
            d["k"] += 1
            d["passed"] += 1
        else:
            d["failing_task_ids"].append(r["id"])
            failed_task_ids.append(r["id"])
            if r["is_control"]:
                d["controls_red"] += 1
            if r["layer"] is None:
                unattributed.append(r["id"])
            elif r["defect"] or r["expected_limit"]:
                layers[r["layer"]] += 1
        per_domain[r["domain"]] = f"{d['k']} / {d['n']}"

    return {
        "arm": "A", "schema": 1, "item": 2620,
        # Wall clock AND the calendar window: an owed check asks whether a run fits the
        # budget, and a bare duration says nothing about what else on the box was awake.
        "wall_start": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "wall_end": datetime.fromtimestamp(time.time(), timezone.utc).isoformat(),
        "wall_s": round(time.time() - started, 3),
        "tasks": len(records),
        "tasks_run": len(records) - len(skipped),
        "tasks_skipped": len(skipped), "skipped": skipped,
        "per_domain": per_domain, "domains": domains_out,
        "failed_task_ids": failed_task_ids,
        "layers": layers, "unattributed": unattributed,
        "unattributed_count": len(unattributed),
        # How many controls had their break verified against the row's own expected value. A
        # count of the instrument working, not of the substrate failing, and a green board
        # without it is a claim nobody checked.
        "control_breaks_verified": verified,
        "control_domains": len({r["domain"] for r in records if r["is_control"]}),
        "results": records,
    }


# ── the dated artefact ──────────────────────────────────────────────────────

ARTIFACT_PREFIX = "arm-a-"
ARTIFACT_RETENTION_DAYS = 30
_DATE_IN_NAME = re.compile(r"^" + ARTIFACT_PREFIX + r"(\d{4}-\d{2}-\d{2})\.json$")


def artifact_path(when: datetime | None = None, out_dir: str | Path | None = None) -> Path:
    """This day's artefact path, under :data:`app.paths.EVAL_SUBSTRATE_FUZZ_DIR`.

    The constant is that module's, and it is inside `DATA_ROOT`; the date is in the filename,
    so there is exactly one file per day per directory and no filename to guess at."""
    base = Path(out_dir) if out_dir is not None else Path(EVAL_SUBSTRATE_FUZZ_DIR)
    day = (when or datetime.now(timezone.utc)).date().isoformat()
    return base / f"{ARTIFACT_PREFIX}{day}.json"


def prune_artifacts(out_dir: str | Path | None = None) -> int:
    """Delete this writer's own dated rows older than :data:`ARTIFACT_RETENTION_DAYS`.

    The bound is here rather than in the retention sweep's table so the directory that grows
    is the directory that bounds itself: a store that depends on a sweep row to stay small is
    one forgotten row away from being an unbounded store, and #2615 is the live demonstration
    of a sweep half-landed. A name this function cannot parse is left alone with a warning —
    unrecognised files in someone's directory are not this writer's to delete, so the bound
    holds on what this module writes, which is exactly the claim it makes."""
    base = Path(out_dir) if out_dir is not None else Path(EVAL_SUBSTRATE_FUZZ_DIR)
    if not base.is_dir():
        return 0
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=ARTIFACT_RETENTION_DAYS)
    removed: list[Path] = []
    unparseable: list[str] = []
    for path in sorted(base.glob(f"{ARTIFACT_PREFIX}*.json")):
        match = _DATE_IN_NAME.match(path.name)
        if not match:
            unparseable.append(path.name)
            continue
        try:
            day = datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except ValueError:
            unparseable.append(path.name)
            continue
        if day < cutoff:
            try:
                path.unlink()
                removed.append(path)
            except OSError as exc:
                unparseable.append(f"{path.name}: {exc}")
    if unparseable:
        print(f"prune_artifacts({base}): left alone, not this writer's dated rows: "
              f"{unparseable}", file=sys.stderr)
    for path in removed:
        del path
    return len(removed)


def write_artifact(report: dict, *, out_dir: str | Path | None = None) -> Path:
    """Write the run to its dated row and prune the directory, so it cannot grow unbounded.

    One file per day, newest wins: an arm-A run scheduled twice in a day replaces its own
    earlier row rather than appending, because the row is a measurement of the box at a
    moment and the day's second measurement is the one a reader wants."""
    path = artifact_path(out_dir=out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    prune_artifacts(path.parent)
    return path


# ── the network rail ────────────────────────────────────────────────────────

_REAL_HTTPX_ATTRS = ("AsyncClient", "Client")


def _rail_http(on: bool) -> None:
    """Put a raising stand-in over `httpx`'s clients for the duration of a task, then restore.

    Arm A is supposed to be model-free and cloud-free, and this box answers vLLM on a local
    port, so a leak would not fail — it would be slow, cost GPU and quietly change what a
    "deterministic" run measured. A rail that raises makes the leak loud. It is set and
    restored around each task rather than at import so the module stays importable by the
    tests that check it independently, and `test_no_model_and_no_network_is_reachable_from_a_run`
    installs its own rail on top to prove nothing in the run path needs a socket."""
    for attr in _REAL_HTTPX_ATTRS:
        original = getattr(httpx, attr, None)
        if on:
            def _raise(*_a: Any, **_k: Any) -> Any:
                raise RuntimeError("arm A reached for the network: httpx is railed off")
            setattr(httpx, attr, _raise)
            _HTTPX_ORIG[attr] = original
        elif attr in _HTTPX_ORIG:
            setattr(httpx, attr, _HTTPX_ORIG.pop(attr))


_HTTPX_ORIG: dict[str, Any] = {}


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """Run arm A headless and print the per-domain numbers. Exit 0 always except on a corpus
    the loader refuses: what the first run finds is owed-check material, and until a fix
    exists for each defect it names, a non-zero exit would make a scheduler treat an
    instrument working as designed as an outage."""
    ap = argparse.ArgumentParser(prog="python -m eval.substrate_fuzz.runner")
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="write the artefact somewhere other than EVAL_SUBSTRATE_FUZZ_DIR")
    ap.add_argument("--no-artifact", action="store_true")
    ap.add_argument("--keep-root", action="store_true",
                    help="leave each task's scratch root standing, for debugging")
    args = ap.parse_args(argv)

    tasks = load_corpus()
    report = run_corpus(tasks, domains=args.domains, limit=args.limit,
                        keep_root=args.keep_root)
    print(f"arm A: {report['tasks_run']} tasks in {report['wall_s']} s "
          f"({report['tasks_skipped']} skipped)")
    for name, per in report["per_domain"].items():
        detail = report["domains"][name]
        print(f"  {name:<15} {per:<9} controls {detail['controls_verified']} verified")
    print("  layers: " + " ".join(f"{k}={v}" for k, v in report["layers"].items()))
    print(f"  control breaks verified: {report['control_breaks_verified']} "
          f"across {report['control_domains']} domains")
    if report["failed_task_ids"]:
        print(f"  failing: {', '.join(report['failed_task_ids'])}")
        for r in report["results"]:
            if r["ok"] is False and r["failure"]:
                print(f"    {r['id']}: {r['failure']['layer']} — "
                      f"{r['failure']['evidence'][:120]}")
    if report["unattributed_count"]:
        print(f"  UNATTRIBUTED (runner defect): {', '.join(report['unattributed'])}")
    if not args.no_artifact:
        out = (write_artifact(report, out_dir=args.out) if args.out
               else write_artifact(report))
        print(f"  artefact: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
