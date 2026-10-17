"""Arm A's corpus as data, and the loader that refuses a task it cannot check (#2620).

The format
----------
`corpus.json` is one object holding 51 real tasks over 8 substrate domains plus 8 positive
controls (one per domain). Every task is `steps` plus exactly one `postcondition` whose
expected value is a literal in the file, so decidable by a program: a sha256, a byte count,
an exit shape, a file-exists-under-the-run-root — the item's four — plus the stdout and
spill-resolved readings of the same idea for what a tool *answers* rather than what it
leaves on disk.

No postcondition kind reads a document, calls a model, or asks anything a person would
judge. The two that look closest to reading prose — `shown_contains` and its absence —
compare a byte substring, and are used only on text this corpus itself chose.

`gen_corpus.py` beside this module is the committed program that emits the file with those
literals computed from fixtures it builds itself. That is the only reason the digests are
trustworthy and nobody had to type one, and it is why hand-editing `corpus.json` is the
wrong place to change a task.

Why the loader refuses
----------------------
An empty result is the worst thing an instrument like this can produce, because it looks
like a clean bill of health. Two kinds of emptiness are refused here rather than reported
downstream, and the item's own recorded wounds are the reason for both:

* **a task with no mechanically checkable postcondition** — #644's triage ruled that
  auto-generated rows are "another way to produce green tests that assert nothing". A
  corpus whose rows can be written without a postcondition reproduces that defect at
  authoring time, so the absence is a load failure naming the task id.
* **a control that cannot fail** — `_control_can_redden` decides it, and both of its routes
  are load-time raises: a postcondition that reads no bytes cannot be reddened by a seeded
  byte break, and a base task that lays the checked file down without a `content` argument
  gives the break nothing to land on. Either one reports "detector alive" forever without
  ever having been tested.

Every refusal carries the task id: a report that says "corpus invalid" without naming a
task is a report nobody can act on.

Size floors are a property of the shipped corpus only. A test that hands the loader two
rows to watch it refuse one of them is not declaring a corpus of two.
"""

from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CORPUS_VERSION = 1
CORPUS_PATH = Path(__file__).resolve().parent / "corpus.json"

#: The whole point of arm A: no LLM grading anywhere, so a pass is arithmetic. The first
#: four are the kinds the item enumerates; the last three are the same idea applied to what
#: a tool ANSWERS instead of what it leaves on disk, which is where the interface layer
#: lives.
POSTCONDITION_KINDS = (
    "file_sha256",     # {path, expected hex}   — a file under the run root, from disk
    "file_length",     # {path, expected int}   — bytes on disk
    "file_exists",     # {path}                 — the path exists under the run root
    "exit_shape",      # {is_error bool}        — what KIND of answer came back
    "stdout_sha256",   # {expected hex}         — the step's payload, spill-resolved
    "stdout_length",   # {expected int}         — its length in bytes
    "shown_contains",  # {expected str}         — the shown bytes carry this substring
)
#: The kinds a byte-level control can redden. `exit_shape` reads a boolean and
#: `file_exists` an inode: no seeded byte break touches either.
BYTE_READING_KINDS = ("file_sha256", "file_length", "stdout_sha256",
                      "stdout_length", "shown_contains")
#: The three breaks the runner can inject at the tool boundary. Each is implemented in
#: `runner.py`, and each is deliberately length-changing, so a byte count is part of the
#: evidence and a control that merely shuffled bytes slips past nothing.
CONTROLS = ("strip_at_nul", "clip_to_prefix", "flip_one_byte")
CONTROL_AT = ("tool_boundary",)
LAYERS = ("model", "interface", "host", "harness")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

#: Floors over the SHIPPED corpus: the item's own numbers.
MIN_TASKS = 40
MIN_DOMAINS = 6

#: The tool shapes arm A drives. A task naming another is a corpus bug: the point of arm A
#: is the substrate under the tools that are there, not an error message about one that is
#: not. `Task` is listed because an arm-B task would name it; arm A never dispatches it.
KNOWN_TOOLS = ("Bash", "Read", "Write", "Edit", "Glob", "Grep",
               "browser_navigate", "http_fetch", "Task")

#: The eight substrate domains. `size_spill` and `refusal_shapes` are Lloyd's own — the
#: spill layer and the harness's guards are part of the boundary arm A exists to
#: byte-compare — and the rest are the substrate primitives the video names.
DOMAINS = ("byte_roundtrip", "filesystem", "process", "size_spill", "binaries",
           "unicode_text", "resource_limits", "refusal_shapes")

__all__ = ["CORPUS_PATH", "CORPUS_VERSION", "CONTROLS", "CONTROL_AT",
           "BYTE_READING_KINDS", "DOMAINS", "LAYERS", "MIN_DOMAINS", "MIN_TASKS",
           "POSTCONDITION_KINDS", "CorpusError", "Post", "Task", "load_corpus",
           "corpus_stats", "corpus_summary", "fixture_sha256s",
           "postcondition_from_spec", "require_floors"]


class CorpusError(Exception):
    """The corpus is not loadable, or not checkable. Always names the task or the count."""


@dataclass
class Post:
    """A validated postcondition. Attribute access, so a reader cannot ask for a typo'd
    field and get the `None` a dict would hand back."""

    kind: str
    path: str | None = None
    expected: Any = None
    is_error: bool | None = None


@dataclass
class Task:
    """One corpus row, validated. `raw` is an untouched copy, so a test can copy a real
    task, change one field, and hand it back to the loader."""

    id: str
    domain: str
    title: str
    reason: str
    steps: list[dict]
    post: Post
    raw: dict
    setup: list[dict] = field(default_factory=list)
    expected_layers: tuple[str, ...] = ()
    control_for: str | None = None
    control: str | None = None
    control_at: str | None = None
    keep_root: bool = False

    @property
    def is_control(self) -> bool:
        return self.control_for is not None

    @property
    def postcondition(self) -> dict:
        """The row's own postcondition as data — which is what a test edits to see what the
        runner and the loader each refuse."""
        return self.raw["postcondition"]

    @property
    def fixture_sha256(self) -> str | None:
        """The digest the row DECLARED for its fixture: a claim, which `_parse_task` checks
        against the bytes and `fixture_sha256s()` recomputes."""
        return self.raw.get("fixture_sha256")

    @property
    def fixture(self) -> bytes | None:
        """The fixture bytes, or None. Base64 in the row (field `fixture`) so the corpus
        stays one JSON file and a NUL or a PNG never has to survive being pasted into a
        text editor."""
        b = self.raw.get("fixture")
        return None if b is None else base64.b64decode(b)

    def fixture_bytes(self) -> bytes | None:
        """The same bytes, for a caller that reads better as a call."""
        return self.fixture

    @property
    def expected(self) -> Any:
        return self.post.expected

    @property
    def expected_path(self) -> str | None:
        return self.post.path


def sha(b: bytes) -> str:
    """sha256, hex — the corpus's only unit of mechanical truth, and shared with the runner
    so the digest a postcondition declares and the digest a run records are computed by the
    same line of code."""
    return hashlib.sha256(b).hexdigest()


def control_arg(task: "Task") -> str:
    """What the step was ASKED to do, before any break — the value a `host` verdict compares
    the box against.

    A tool that overrode the break wrote exactly this, and the file it left behind is the
    proof. That is why the `host` rule needs a stored-target step (`Write`/`Edit`) and not a
    shell command: `bash -c 'printf x > f'` can write anything or nothing, so "the box still
    holds the asked-for value" has no referent there, and the verdict would be a number
    invented rather than observed."""
    for step in reversed(task.steps):
        args = step.get("args") or {}
        for key in ("content", "new_string", "text"):
            if isinstance(args.get(key), str):
                return args[key]
        if isinstance(args.get("command"), str):
            return args["command"]
    return ""


def _break_bytes(control: str, original: bytes) -> bytes | None:
    """A control's seeded break, applied to the ORIGINAL bytes. `None` means the break
    cannot be applied to this payload at all.

    One function, two callers: `apply_control` (what the runner's pool injects at the
    boundary) and the clause-3 test, which applies it to the bytes the tool SHOULD have
    produced and asserts the two digests differ. A break implemented twice is a control whose
    shipped behaviour and whose test disagree, which is a green detector that breaks nothing.

    `None` is an answer rather than a shrug, and the runner treats it as a build failure: a
    control whose break never fired leaves its domain's detector dark while the report still
    counts the row as a control — the green-board-with-a-dead-detector outcome clause 3 exists
    to prevent. All three transformations act on the payload alone: never on a task id, never
    on the file on disk. That is what makes them tool-boundary breaks rather than a break in
    the runner's own bookkeeping, which would prove the bookkeeping works and nothing about
    the substrate.
    """
    if control not in CONTROLS:
        raise CorpusError(f"control {control!r} is outside the closed vocabulary; the "
                          f"vocabulary is {CONTROLS} — a control that invents a break nobody "
                          "implemented would report itself as exercised")
    if control == "flip_one_byte":
        if not original:
            return None
        out = bytearray(original)
        out[0] ^= 0x01
        return bytes(out)
    if control == "clip_to_prefix":
        keep = max(1, len(original) // 2)
        return original[:keep] if keep < len(original) else None
    # Orellana's own fault: a NUL read as a terminator, silently shortening a payload that
    # still writes and still returns without complaint.
    cut = original.find(b"\x00")
    return None if cut < 0 else original[:cut]


def apply_control(control: str, text: str) -> str | None:
    """`_break_bytes` over a text response, so the runner's pool can break what the tool
    answered without a second implementation of the break.

    The encode is surrogateescape in and surrogateescape out, so a payload carrying an
    unpaired surrogate or a split multi-byte sequence comes back byte-identical apart from
    the break itself — a control that re-encoded its own payload would be reporting its
    encoder's behaviour instead of the boundary's."""
    broken = _break_bytes(control, text.encode("utf-8", "surrogateescape"))
    return None if broken is None else broken.decode("utf-8", "surrogateescape")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorpusError(message)


def postcondition_from_spec(spec: Any) -> Post:
    """Validate one postcondition and return it typed.

    An absent or non-object postcondition is the clause-1 refusal: a task carrying no
    expected value a program can decide cannot fail, which is the property the item makes
    mandatory and the trap #644's triage named.
    """
    _require(isinstance(spec, dict),
             "task has no mechanical postcondition: it carries no expected value a program "
             "can decide, so nothing in it can ever fail (#2620 clause 1)")
    kind = spec.get("kind")
    _require(isinstance(kind, str) and kind in POSTCONDITION_KINDS,
             f"unsupported postcondition kind {kind!r}: not one of {POSTCONDITION_KINDS} "
             "— an unimplemented kind is reported green by a decoder that does not know "
             "it, which is the #644 defect class")
    if kind in ("file_sha256", "file_length", "file_exists"):
        path = spec.get("path")
        _require(isinstance(path, str) and path.strip() != "",
                 f"postcondition kind {kind} needs a 'path' to check")
        parts = Path(path).parts
        _require(not path.startswith("/") and ".." not in parts and "." not in parts,
                 f"postcondition path {path!r} must stay inside the run root — a mechanical "
                 "check that can read any file on the box reports someone else's "
                 "filesystem as the run's result")
    if kind == "file_sha256":
        _require(isinstance(spec.get("expected"), str) and HEX64.match(spec["expected"]),
                 f"file_sha256 expected must be 64-hex lowercase, got "
                 f"{spec.get('expected')!r}")
    if kind == "file_length":
        _require(isinstance(spec.get("expected"), int)
                 and not isinstance(spec["expected"], bool),
                 "file_length expected must be an int")
    if kind == "stdout_sha256":
        _require(isinstance(spec.get("expected"), str) and HEX64.match(spec["expected"]),
                 f"stdout_sha256 expected must be 64-hex lowercase, got "
                 f"{spec.get('expected')!r}")
    if kind == "stdout_length":
        _require(isinstance(spec.get("expected"), int)
                 and not isinstance(spec["expected"], bool),
                 "stdout_length expected must be an int")
    if kind == "shown_contains":
        _require(isinstance(spec.get("expected"), str) and spec["expected"] != "",
                 "shown_contains needs a non-empty expected substring")
    if kind == "exit_shape":
        _require(isinstance(spec.get("is_error"), bool),
                 "exit_shape needs a boolean is_error")
    return Post(kind=kind, path=spec.get("path"), expected=spec.get("expected"),
                is_error=spec.get("is_error"))


def _no_bytes_reason(kind: str) -> str | None:
    """Why a postcondition of this kind cannot be reddened by a byte break, or None."""
    if kind in BYTE_READING_KINDS:
        return None
    return (f"postcondition kind {kind!r} reads no bytes, so no seeded byte break can "
            "redden it — a control that cannot fail is a build failure (#2620 clause 3)")


#: A `shown_contains` needle is a claim about what an answer will SAY, and for most needles
#: the corpus authors that text itself. One class is not authorable: the string an
#: operating-system error prints is decided by the errno behind it, and an errno appears in
#: an answer only when the condition behind it exists. #2929's `rf-readonly-dir-error-text`
#: pinned `Read-only file system` (EROFS — a read-only MOUNT) on a mode-500 directory
#: (EACCES), so the needle could never fire however the substrate behaved and the row read
#: as a refusal-shape defect that no substrate can cause.
#:
#: The needle set is derived and therefore closed: every string below is this machine's own
#: `os.strerror` output, so a new errno joins the class without anyone editing a list. Only
#: two things are hand-written — which errno needs which CAUSE, and how a row is recognised
#: as staging that cause.
def _os_error_needles() -> dict[str, int]:
    """needle (lower-cased) -> the errno whose `strerror` it is.

    :data:`errno.errorcode`'s KEYS are the error numbers and its values the names, which is
    the direction its own documentation states and the one that tripped the first version of
    this function: iterating the values handed back names, `os.strerror("EACCES")` raised
    `TypeError`, the `except` swallowed it, and the map came out EMPTY — which made every
    needle in the corpus look fine. A guard whose input map is empty reports a clean board, so
    `test_the_os_error_needle_map_is_populated_before_it_is_trusted` pins its size (#2929).
    """
    out: dict[str, int] = {}
    for code in sorted(errno.errorcode):
        try:
            out[os.strerror(code).lower()] = code
        except (OSError, OverflowError, TypeError, ValueError):
            continue
    return out


OS_ERROR_NEEDLES = _os_error_needles()
ERRNO_NAMES = dict(errno.errorcode)

#: The errno a needle can name, and the shapes that raise it. An errno missing from this
#: table has no staging vocabulary at all, so any needle naming its message is refused: the
#: honest default, because a needle nothing can stage is a claim about a string and not
#: about the substrate. `read_only_mount` deliberately has no producer either — arm A runs
#: inside an ordinary writable tree and cannot mount anything read-only, which is exactly
#: why the shipped row had to stop asserting it.
_ERRNO_CAUSES: dict[int, tuple[str, ...]] = {
    errno.EACCES: ("mode_without_write", "protected_path"),
    errno.EPERM: ("mode_without_write", "protected_path"),
    errno.EROFS: ("read_only_mount",),
    errno.ENOSPC: ("disk_full",),
    errno.ENOMEM: ("memory_ceiling",),
    errno.E2BIG: ("argv_ceiling",),
    errno.EMFILE: ("fd_exhaustion",),
    errno.ENFILE: ("fd_exhaustion",),
    errno.EAGAIN: ("fd_exhaustion",),
    errno.ETXTBSY: ("concurrent_writer",),
}

#: ENOENT is the errno a step raises for free: naming a path nothing creates IS the shape,
#: and a row that reads a file its own step never wrote needs no setup entry to be able to
#: fail. Refusing its needle would teach an author to write the message in a title instead.
_UNSTAGED_OK_ERRNOS = frozenset({errno.ENOENT})

#: A write into one of these is refused by the operating system for an unprivileged uid,
#: which is the condition behind the EACCES text of `rf-write-into-proc-error-text`.
_PROTECTED_WRITE_PREFIXES = ("/proc/", "/sys/", "/boot/", "/usr/lib/", "/dev/")

#: A single command line this long is the argument-list ceiling: `res-oversized-arg-list`
#: carries one of 200,022 characters, so a row that wants to assert E2BIG's text has to
#: actually carry an argument list of the size that raises it.
_ARGV_CEILING_CHARS = 100_000


def _mode_lacks_write(item: dict) -> bool:
    """Whether a setup entry declares a mode this uid cannot write through."""
    mode = item.get("mode")
    if mode is None:
        return bool(item.get("read_only"))
    try:
        bits = int(str(mode), 8)
    except (TypeError, ValueError):
        return False
    return not bits & stat.S_IWUSR


def _step_strings(steps: list[dict]) -> list[str]:
    return [v for s in steps for v in (s.get("args") or {}).values() if isinstance(v, str)]


#: A shell output redirect — `> path`, never `2>` or `&>`, which are how the shipped rows
#: send their stderr to /dev/null and do not claim anything about writing to it.
_OUTPUT_REDIRECT = re.compile(r"(?<![0-9&|])>\s*(\S+)")


def _absolute_write_targets(steps: list[dict]) -> list[str]:
    out: list[str] = []
    for text in _step_strings(steps):
        for target in _OUTPUT_REDIRECT.findall(text):
            if target.startswith("/"):
                out.append(target)
    return out


def _staged_causes(setup: list[dict], steps: list[dict]) -> tuple[set[str], list[str]]:
    """The error conditions this row really creates, plus what was measured and rejected.

    A cause counts only when the condition behind it exists, so a needle asking for a
    permission refusal is not "staged" by a step writing to a path that was never there — the
    kernel answers ENOENT for that, and the row would sit red blaming the substrate for a
    filename. The protected-path reading is therefore a probe of this machine, which is the
    honest instrument: the claim is about the machine.
    """
    causes: set[str] = set()
    notes: list[str] = []
    if any(_mode_lacks_write(item) for item in setup):
        causes.add("mode_without_write")
    for target in _absolute_write_targets(steps):
        if not target.startswith(_PROTECTED_WRITE_PREFIXES):
            continue
        if not os.path.exists(target):
            notes.append(f"{target} does not exist, so a write there is refused with ENOENT "
                         "before any permission check and can never print a permission error")
        elif os.access(target, os.W_OK):
            notes.append(f"{target} is writable by uid {os.getuid()}, so no permission refusal "
                         "is available to print")
        else:
            causes.add("protected_path")
    for text in _step_strings(steps):
        if "setrlimit" in text or "RLIMIT_" in text:
            causes.add("memory_ceiling")
        if len(text) > _ARGV_CEILING_CHARS:
            causes.add("argv_ceiling")
    return causes, notes


def errno_needle_reason(post: Post, setup: list[dict], steps: list[dict]) -> str | None:
    """Why this row's `shown_contains` needle cannot be produced by its own setup or step,
    or None when nothing about it is impossible.

    The question is the row's own capability, never the substrate's behaviour: a needle that
    needs an errno no part of the row stages stays red for every substrate on earth, which is
    the difference between an instrument and a wish. Called by `_parse_task` at load time, and
    by the corpus test that scans the shipped rows — one measurement, both sides (#2929).
    """
    if post.kind != "shown_contains" or not isinstance(post.expected, str):
        return None
    code = OS_ERROR_NEEDLES.get(post.expected.lower())
    if code is None or code in _UNSTAGED_OK_ERRNOS:
        return None
    named = ERRNO_NAMES.get(code, f"errno {code}")
    causes = _ERRNO_CAUSES.get(code)
    if causes is None:
        return (f"shown_contains needle {post.expected!r} is what {named} prints, and no "
                "setup or step shape in this corpus stages that condition, so the needle can "
                "never fire and the row would report a defect no substrate can cause")
    staged, notes = _staged_causes(setup, steps)
    if staged & set(causes):
        return None
    return (f"shown_contains needle {post.expected!r} is what {named} prints, which needs one "
            f"of {sorted(causes)}; this row stages "
            f"{sorted(staged) if staged else 'nothing'}"
            + (f"; measured: {'; '.join(notes)}" if notes else "")
            + ", so the needle can never fire")


def _control_can_redden(control: Task, base: Task) -> str | None:
    """Why this control could NOT redden its own postcondition, or None if it can.

    The one question clause 3 asks, with both routes to a detector that can never light up
    answered in the same function so a fix to one cannot leave the other behind:

    * a postcondition that reads no bytes (`_no_bytes_reason`);
    * a base task that lays the checked file down without a `content` argument, so the
      break lands on an argument that does not exist and the file-side digest stays green
      however hard the control tries.
    """
    reason = _no_bytes_reason(control.post.kind)
    if reason:
        return reason
    if control.post.kind in ("file_sha256", "file_length"):
        target = str(control.post.path)
        writes_it = any(s.get("tool") == "Write"
                        and isinstance((s.get("args") or {}).get("content"), str)
                        and target in str((s.get("args") or {}).get("file_path", ""))
                        for s in base.steps)
        if not writes_it:
            return (f"its base task {base.id} writes {target!r} without a `content` "
                    "argument, so the break has nothing to land on and the file-side "
                    "digest can never go red")
    return None


def _parse_step(step: Any, tid: str, index: int) -> dict:
    _require(isinstance(step, dict), f"corpus task {tid}: step {index} is not an object")
    tool = step.get("tool")
    _require(isinstance(tool, str) and tool in KNOWN_TOOLS,
             f"corpus task {tid}: step {index} names tool {tool!r}, outside the arm-A "
             f"surface {KNOWN_TOOLS}")
    args = step.get("args", {})
    _require(isinstance(args, dict),
             f"corpus task {tid}: step {index} args are not an object")
    return {"tool": tool, "args": args}


def _check_setup(item: Any, tid: str) -> None:
    _require(isinstance(item, dict), f"corpus task {tid}: a setup entry is not an object")
    path = item.get("path")
    _require(isinstance(path, str) and path.strip() != "",
             f"corpus task {tid}: a setup entry names no path")
    parts = Path(path).parts
    _require(not path.startswith("/") and ".." not in parts and "." not in parts,
             f"corpus task {tid}: setup path {path!r} must stay inside the run root")
    if item.get("dir"):
        return
    content = item.get("content")
    if "sha256" in item and isinstance(content, str):
        # Arm A's own writes get their digest checked too: a fixture the corpus
        # mis-transcribes makes every task that reads it red for the wrong reason.
        want = item["sha256"]
        got = hashlib.sha256(content.encode("utf-8", "surrogateescape")).hexdigest()
        _require(got == want, f"corpus task {tid}: setup {path!r} declares sha256 {want!r} "
                              f"but its content hashes to {got}")


def _parse_task(row: Any, seen_ids: set[str] | None = None) -> Task:
    """Validate one row and return it.

    `seen_ids` is every id already accepted, which is what makes a duplicate id and a
    dangling `control_for` detectable at all: a control that points at a typo'd id is
    counted in its domain's coverage and checks nothing.
    """
    seen = set() if seen_ids is None else seen_ids
    if not isinstance(row, dict):
        raise CorpusError(f"corpus row is not an object ({type(row).__name__}); every row "
                          "must be a task")
    tid = row.get("id")
    _require(isinstance(tid, str) and tid.strip() != "",
             f"corpus row has no id (keys: {sorted(row)[:6]})")
    _require(tid not in seen, f"corpus task {tid}: id given twice — a control and its base "
                              "would collapse onto one result")
    _require(isinstance(row.get("setup"), (list, type(None))),
             f"corpus task {tid}: setup must be a list")
    for item in row.get("setup") or []:
        _check_setup(item, tid)
    domain = row.get("domain")
    _require(isinstance(domain, str) and domain in DOMAINS,
             f"corpus task {tid}: domain {domain!r} is not one of {DOMAINS}")
    _require(isinstance(row.get("title"), str) and row["title"].strip() != "",
             f"corpus task {tid} has no title")
    _require(isinstance(row.get("reason"), str) and row["reason"].strip() != "",
             f"corpus task {tid} has no reason — a task nobody can justify is a task "
             "nobody will fix when it goes red")
    steps = row.get("steps")
    _require(isinstance(steps, list) and steps, f"corpus task {tid} has no steps")
    parsed_steps = [_parse_step(s, tid, i) for i, s in enumerate(steps)]

    fixture = row.get("fixture")
    if fixture is not None:
        # Before the postcondition, because the bytes are the more basic claim: a task
        # whose fixture does not match its declared digest has expected values computed
        # from something else, and that is worth naming even if the row is broken twice.
        _require(isinstance(fixture, str),
                 f"corpus task {tid}: fixture is not base64 text")
        declared = row.get("fixture_sha256")
        _require(isinstance(declared, str) and bool(HEX64.match(declared)),
                 f"corpus task {tid} carries a fixture with no declared fixture_sha256, "
                 "so nothing checks the bytes its expected values were computed from")
        got = hashlib.sha256(base64.b64decode(fixture)).hexdigest()
        _require(got == declared,
                 f"corpus task {tid}: fixture_sha256 {declared} does not match the fixture "
                 f"it travels with, which hashes to {got}")

    post = postcondition_from_spec(row.get("postcondition"))

    needle_reason = errno_needle_reason(post, row.get("setup") or [], parsed_steps)
    _require(needle_reason is None, f"corpus task {tid}: {needle_reason}")

    control_for = row.get("control_for")
    control = row.get("control")
    control_at = row.get("control_at")
    if control_for is not None:
        _require(control_for in seen,
                 f"corpus control {tid} names control_for {control_for!r}, which is not a "
                 "task in this corpus — a control pointing at a typo'd id is counted as "
                 "coverage and checks nothing")
        _require(control in CONTROLS,
                 f"corpus control {tid}: control {control!r} is outside the fixed "
                 f"vocabulary; the vocabulary is {CONTROLS} — a control that invents a "
                 "break nobody implemented reports itself as exercised")
        _require(control_at in CONTROL_AT,
                 f"corpus control {tid}: control_at {control_at!r} is not one of "
                 f"{CONTROL_AT} — a break inside the runner's own bookkeeping tests the "
                 "bookkeeping and not the substrate")
        reason = _no_bytes_reason(post.kind)
        _require(reason is None, f"corpus control {tid}: {reason}")
    else:
        _require(control is None,
                 f"corpus task {tid} declares control {control!r} without control_for")

    for layer in row.get("expected_layers", []) or []:
        _require(layer in LAYERS,
                 f"corpus task {tid}: expected layer {layer!r} is outside the fixed "
                 f"vocabulary; the vocabulary is {LAYERS} — there is no unknown bucket, on "
                 "either side")

    seen.add(tid)
    return Task(id=tid, domain=domain, title=row["title"], reason=row["reason"],
                steps=parsed_steps, post=post, raw=json.loads(json.dumps(row)),
                setup=list(row.get("setup") or []),
                expected_layers=tuple(row.get("expected_layers", []) or []),
                control_for=control_for, control=control, control_at=control_at,
                keep_root=bool(row.get("keep_root", False)))


def _check_control_pairs(tasks: list[Task]) -> None:
    """Ask `_control_can_redden` of every control now that its base task is loadable."""
    by_id = {t.id: t for t in tasks}
    for t in tasks:
        if not t.is_control:
            continue
        reason = _control_can_redden(t, by_id[t.control_for])
        _require(reason is None, f"corpus control {t.id}: {reason}")


def _check_floors(tasks: list[Task]) -> None:
    stats = corpus_stats(tasks)
    _require(stats["tasks"] >= MIN_TASKS,
             f"corpus has {stats['tasks']} real tasks; the item requires >= {MIN_TASKS} — a "
             "corpus under that number is a handful of hand-picked shapes wearing a "
             "corpus's name")
    _require(stats["domains"] >= MIN_DOMAINS,
             f"corpus spans {stats['domains']} domains; the item requires >= "
             f"{MIN_DOMAINS} — one narrow area finds only the limits already known")
    _require(not stats["domains_without_control"],
             f"domains with no positive control: {stats['domains_without_control']} — a "
             "domain reporting zero failures without a control that was shown to light up "
             "reports nothing (#2620 clause 3, and #644's stated objection)")


def load_corpus(source: str | Path | list[dict] | None = None, *,
                check_cross_row: bool = True) -> list[Task]:
    """Load and validate a corpus.

    `source` is None (the shipped `corpus.json`), a path, or an already-parsed list of rows
    — the tests hand rows in to watch the loader refuse them. The size floors are asserted
    only for the shipped corpus: they are a claim about THAT file, and refusing a two-row
    test corpus for having two rows would make the per-row refusals untestable.
    """
    if source is None:
        doc = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    elif isinstance(source, list):
        doc = {"version": CORPUS_VERSION, "item": 2620, "tasks": source}
    else:
        doc = json.loads(Path(source).read_text(encoding="utf-8"))
    _require(isinstance(doc, dict), "the corpus document is not a JSON object")
    rows = doc.get("tasks")
    _require(isinstance(rows, list) and rows, "the corpus document has no 'tasks' list")
    _require(doc.get("version") == CORPUS_VERSION,
             f"corpus version {doc.get('version')!r} != {CORPUS_VERSION}: the runner's "
             "decoders and the file disagree about what a task means")
    _require(doc.get("item") == 2620,
             f"corpus item {doc.get('item')!r} is not #2620 — this loader is arm A's")
    seen: set[str] = set()
    tasks = [_parse_task(row, seen_ids=seen) for row in rows]
    _check_control_pairs(tasks)
    if source is None:
        _check_floors(tasks)
    return tasks


def require_floors(tasks: list[Task]) -> list[Task]:
    """Re-assert the shipped corpus's floors over an already-parsed list, for a caller that
    loaded by path and still wants the claim checked (`runner.main` does)."""
    _check_floors(tasks)
    return tasks


def corpus_stats(tasks: list[Task]) -> dict:
    """The counts clause 1 is stated in: real tasks, domain count, control count, and which
    domains have no control. Every refusal message here quotes one of these."""
    real = [t for t in tasks if not t.is_control]
    controls = [t for t in tasks if t.is_control]
    domains = sorted({t.domain for t in real})
    covered = {t.domain for t in controls}
    return {"tasks": len(real), "controls": len(controls), "domains": len(domains),
            "domain_names": domains,
            "domains_without_control": sorted(set(domains) - covered)}


def corpus_summary(tasks: list[Task]) -> dict:
    """Per-domain real/control counts, so a reader can check the shape without opening the
    JSON."""
    out: dict[str, dict[str, int]] = {}
    for t in tasks:
        d = out.setdefault(t.domain, {"real": 0, "controls": 0})
        d["controls" if t.is_control else "real"] += 1
    for d in out.values():
        d["total"] = d["real"] + d["controls"]
    return dict(sorted(out.items()))


def fixture_sha256s(tasks: list[Task]) -> dict[str, str]:
    """`task_id -> sha256 of its own fixture`, recomputed from the bytes rather than read
    from the declaration the loader already cross-checked.

    A digest in a data file is a claim; this is the function that turns it back into a
    measurement, which is what the corpus test compares the expected values against.
    """
    out: dict[str, str] = {}
    for t in tasks:
        raw = t.fixture
        if raw is not None:
            out[t.id] = hashlib.sha256(raw).hexdigest()
    return out
