"""Deterministic structural vet of a change set (backlog #679).

The gate asks whether the system still works: compileall, an import smoke,
pyflakes, the full suite, a canary boot, a real turn, a retrieval eval. Nothing
in that ladder asks the other question — is this change set *structurally
sane*? A file emptied by a botched redirect that still imports, a compiled blob
dropped into the tree by a build the agent ran, a diff two orders of magnitude
bigger than anything this repo has landed: each one preserves behaviour
exactly, which is why every behavioural rung is green on it. The class is not
hypothetical — `lloyd/MEMORY.md` was truncated to one line and committed on
2026-09-10, and no test suite could have caught it
(`~/obsidian/knowledge/software/memory-md-clobber-2026-09-10.md`).

PatchPilot (Form3) runs this pass between its agent and its deterministic
controller, before anything is committed. It is cheap and deterministic, which
is the only reason it can run on every change.

Three checks, in the item's own order of confidence:

  * `empty_file` — non-empty at the merge base, zero bytes at HEAD.
  * `binary_artifact` — newly added, NUL byte in its first bytes, outside
    :data:`BINARY_ALLOWLIST_GLOBS`.
  * `diff_too_large` — total changed lines over the ceiling.

NOT here, deliberately:

  * Out-of-scope paths (the item's check (c)). That already runs
    deterministically at landing — `rung_preflight` calls `spec.check_scope` on
    the enumerated `changed_paths`, and `spec.classify` refuses every denied OR
    unlisted path — so a second owner for one property would only add a way for
    the two to disagree.
  * Any LLM call. The vet's entire value is that it cannot be argued with.
  * Newly *added* empty files. PatchPilot's motivating case is an agent that
    piped bash badly and produced a zero-byte file, but the clause pinned for
    this round is base-relative, and a file that was never non-empty is a
    different finding. Recorded as a follow-up on #679, not silently widened.
  * Symlinks and per-file size ceilings (the item's step 1(d)), which the
    triage left out of the contract.

Stdlib only, and it never raises: a git read that fails returns
`status="unevaluated"` with the reason, never an empty violation list, because
a check that could not run must not be reported as clean.

Run it by hand over any tree::

    .venvs/lloyd/bin/python -m scripts.automod.vet <base-ref> [worktree]
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

#: Ceiling on total changed lines (`git diff --numstat`, added + deleted, all
#: files). Measured on `main` 2026-09-24: the largest of the last 300 commits is
#: 6,103 lines (an eval JSON artifact landing) and the largest authored change
#: in that window is 4,807, while an ordinary round lands 500-1,500. So 12,000
#: is roughly twice the largest change this repo has actually landed, and a
#: sixth of the 70,000-line dependency bump that motivated the item.
#: Overridable as `automod.gate.max_diff_lines`; the default lives HERE because
#: the loop may not write `config.yaml`, and a check whose only default sits in
#: a file nobody is allowed to edit is a check that is off.
DEFAULT_MAX_DIFF_LINES = 12_000

#: Bytes of a newly-added blob sniffed for a NUL. Git's own binary heuristic
#: looks at the first 8,000 bytes of a blob, so 8 KiB is the same judgement
#: rounded to a page: a file this check calls binary and git calls text is a
#: file whose NUL sits past the first page, which no compiled artifact does.
BINARY_SNIFF_BYTES = 8192

#: The trees that legitimately carry binary blobs, read off `main` rather than
#: invented: `git ls-files` holds 17 binary-shaped tracked files, and these are
#: the three of those trees that a round is ALLOWED to write. The rest
#: (`agent-services/models/**` is unlisted, `chrome-extension/**` is unlisted)
#: are refused earlier by `spec.check_scope`, so listing them here would only
#: let the vet disagree with the rung that already said no.
#:
#: The check is path-scoped on purpose. A repo-wide "no NUL bytes" rule is false
#: on this tree by construction — five tracked `.onnx` models (largest 2,276
#: KiB), `web/public/{apple-touch-icon.png,lloyd.jpg}`, and four `.wav` voice
#: fixtures under `tests/fixtures/voice/` that the wake-word tests read.
BINARY_ALLOWLIST_GLOBS: tuple[str, ...] = (
    "web/public/**",
    "tests/fixtures/**",
    "agent-services/models/**",
)

#: Violation `kind`s, so a caller can bucket them without string-matching.
EMPTY_FILE = "empty_file"
BINARY_ARTIFACT = "binary_artifact"
DIFF_TOO_LARGE = "diff_too_large"

#: `VetResult.status` values. `unevaluated` is a third answer alongside
#: "clean" and "dirty", and it has to be nameable: `git diff` failing and the
#: change set being genuinely clean both produce an empty violation list.
EVALUATED = "evaluated"
UNEVALUATED = "unevaluated"


def _cfg(key: str, default):
    """`automod.gate.<key>`, or `default`. Never raises.

    Same shape as `gate._gate_cfg`: lazy, guarded, and it reads the live
    `CONFIG` object so a test can substitute a dict. Importing it at module
    scope would put the application inside a module the gate imports to judge
    that application.
    """
    try:
        from app.config import CONFIG
        return ((CONFIG.get("automod") or {}).get("gate") or {}).get(key, default)
    except Exception:
        return default


def max_diff_lines() -> int:
    """The line ceiling in force: config override, else the code default."""
    raw = _cfg("max_diff_lines", DEFAULT_MAX_DIFF_LINES)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        # A ceiling nobody can parse is not a ceiling; the code default is.
        return DEFAULT_MAX_DIFF_LINES
    return value if value > 0 else DEFAULT_MAX_DIFF_LINES


@dataclass(frozen=True)
class Violation:
    """One structural defect in the change set. `path` is "" for whole-diff kinds."""

    kind: str
    path: str
    detail: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "path": self.path, "detail": self.detail}

    def label(self) -> str:
        return f"{self.kind}:{self.path}" if self.path else self.kind


@dataclass
class VetResult:
    """What the vet concluded — including whether it got to conclude anything."""

    status: str = EVALUATED
    violations: list[Violation] = field(default_factory=list)
    reason: str = ""
    totals: dict = field(default_factory=dict)

    @property
    def evaluated(self) -> bool:
        return self.status == EVALUATED

    def to_dict(self) -> dict:
        out: dict = {"status": self.status,
                     "violations": [v.to_dict() for v in self.violations],
                     "totals": dict(self.totals)}
        if self.reason:
            out["reason"] = self.reason
        return out


def _unevaluated(reason: str) -> VetResult:
    """A vet that could not complete, which is never the same as a clean vet."""
    return VetResult(status=UNEVALUATED, violations=[], reason=reason[:400])


def _git(root: Path, *args: str, stdin: bytes | None = None,
         timeout: float = 120.0) -> tuple[int, str, str]:
    """`git` in `root`, returning (returncode, stdout, stderr) — never raises.

    Every git read the vet depends on goes through here so one failure mode —
    a missing repo, a bad ref, a tree moved out from under the call — is one
    guarded path instead of five.
    """
    try:
        r = subprocess.run(["git", "-C", str(root), *args], input=stdin,
                           capture_output=True, timeout=timeout, check=False)
        return r.returncode, r.stdout.decode("utf-8", "replace"), \
            r.stderr.decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 — the caller reports it as unevaluated
        return 127, "", f"{type(exc).__name__}: {exc}"


def _paths_and_sizes(root: Path, ref: str) -> dict[str, tuple[int, str]] | None:
    """`path -> (blob size, blob sha)` for every tracked file at `ref`.

    `git ls-tree -r --long` answers for the whole tree in one call, which is why
    this is a tree read and not one `cat-file` per changed path. The size in
    column 4 is the blob's byte count, so "0 bytes at HEAD" needs no read of the
    file's contents at all — the check that has to catch an emptied 400 KB file
    cannot itself be the expensive one.

    Returns None when the read failed; `{}` means the ref has no files, which a
    caller treats as its own answer.
    """
    rc, out, err = _git(root, "ls-tree", "-r", "--long", ref)
    if rc != 0:
        return None
    sizes: dict[str, tuple[int, str]] = {}
    for line in out.splitlines():
        # `git ls-tree -l` prints `<mode> <type> <sha> <size>` then ONE tab and
        # the path (measured: `100755 blob e69de29…       0\tapp/__init__.py`,
        # the size right-padded with spaces inside the meta field). So the tab
        # is the path boundary and the path is the whole remainder — splitting
        # on whitespace would corrupt every path containing a space, and the
        # quoted form a non-ASCII path arrives in simply does not match.
        meta, sep, path = line.partition("\t")
        if not sep or not path:
            continue
        fields = meta.split()
        if len(fields) != 4:
            continue
        try:
            size = int(fields[3])
        except ValueError:
            continue  # a tree/commit entry (-r should mean none) is not a file
        sizes[path] = (size, fields[2])
    return sizes


def _name_status(root: Path, spec: str) -> list[tuple[str, str]] | None:
    """Changed (status, path) pairs for `spec`. None if the read failed.

    `--no-renames` is deliberate and not an optimisation: rename detection is
    `diff.renames`-config dependent, so with it on, the same committed change
    would be `A new/path` here and `R100 old new` there, and the binary check
    would fire on a moved `.onnx` in one environment and not another. A rename
    is reported here as a delete plus an add, which is the deterministic
    reading and the one the size and NUL checks can actually act on.
    """
    rc, out, err = _git(root, "diff", "--name-status", "--no-renames", spec)
    if rc != 0:
        return None
    entries: list[tuple[str, str]] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status, path = parts[0].strip(), parts[-1]
        if status and path:
            entries.append((status[0], path))
    return entries


def _changed_lines(root: Path, spec: str) -> int | None:
    """Total added + deleted lines for `spec`, or None if the read failed.

    Binary files report `-\t-\t` and contribute nothing: their line count is not
    a quantity, and a 2 MB blob is caught by the NUL check rather than here.
    Deleted files count, since a 70,000-line *deletion* is exactly as much
    untrusted input as an insertion.
    """
    rc, out, err = _git(root, "diff", "--numstat", "--no-renames", spec)
    if rc != 0:
        return None
    total = 0
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        for field_text in (parts[0], parts[1]):
            try:
                total += int(field_text)
            except ValueError:
                continue
    return total


def _has_nul(root: Path, blob: str) -> bool | None:
    """True if the blob's first `BINARY_SNIFF_BYTES` bytes contain a NUL.

    `None` means the content could not be read — clause 4's rule, and the reason
    the object's size is asked for first. Reading only the head of a stream
    cannot report its own failure: closing the pipe after 8 KiB leaves git dying
    of SIGPIPE, so a non-zero exit status here would mean "a large object", not
    "unreadable", and the empty output of a failed `cat-file` would be read as an
    empty blob and answered `False` — a missing object reported as text, which is
    precisely the shape this module forbids. `cat-file -s` is bounded, fully
    consumed, and its exit status means something: a blob git cannot resolve is
    `None` before any streaming starts, and a `size > 0` blob from which no bytes
    arrive is likewise `None`, never `False`.
    """
    rc, out, _err = _git(root, "cat-file", "-s", blob)
    if rc != 0:
        return None
    try:
        size = int(out.strip())
    except ValueError:
        return None
    if size == 0:
        return False                       # an empty blob has no NUL: a real answer
    proc = None
    try:
        proc = subprocess.Popen(
            ["git", "-C", str(root), "cat-file", "blob", blob],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.stdout is None:
            return None
        data = proc.stdout.read(BINARY_SNIFF_BYTES)
        if not data:
            # The object has `size` bytes and we were handed none: a failed read.
            return None
        return b"\x00" in data
    except Exception:  # noqa: BLE001 — the caller reports it as unevaluated
        return None
    finally:
        if proc is not None:
            for stream in (proc.stdout, proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:  # noqa: BLE001
                    pass
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass


def binary_allowed(path: str,
                   allowlist: tuple[str, ...] = BINARY_ALLOWLIST_GLOBS) -> bool:
    """Whether `path` sits in a tree that legitimately holds binaries."""
    return any(fnmatch(path, glob) for glob in allowlist)


def vet_change_set(base: str, worktree, *,
                   max_lines: int | None = None,
                   binary_allowlist: tuple[str, ...] = BINARY_ALLOWLIST_GLOBS) -> VetResult:
    """Structurally vet the change set from `base` to `worktree`'s HEAD.

    `base` is any rev; the comparison runs from `git merge-base base HEAD`, i.e.
    exactly what `git diff base...HEAD` judges, which is what the gate promotes.
    Returns a :class:`VetResult`; a failed git read is `status="unevaluated"`
    with the reason, never an empty violation list.
    """
    root = Path(worktree)
    if not str(base or "").strip():
        return _unevaluated("no base ref given")
    if not root.is_dir():
        return _unevaluated(f"no worktree at {root}")
    if not root.joinpath(".git").exists():
        return _unevaluated(f"{root} is not a git repository")

    ceiling = max_lines if max_lines is not None else max_diff_lines()

    rc, out, err = _git(root, "merge-base", str(base), "HEAD")
    if rc != 0:
        return _unevaluated(f"merge-base {base}...HEAD failed: {err.strip()[:180]}")
    merge_base = out.strip()
    if not merge_base:
        return _unevaluated(f"merge-base {base}...HEAD returned no commit")
    spec = f"{merge_base}..HEAD"

    entries = _name_status(root, spec)
    if entries is None:
        return _unevaluated(f"git diff --name-status {spec} failed")
    lines = _changed_lines(root, spec)
    if lines is None:
        return _unevaluated(f"git diff --numstat {spec} failed")
    base_tree = _paths_and_sizes(root, merge_base)
    if base_tree is None:
        return _unevaluated(f"git ls-tree {merge_base[:12]} failed")
    head_tree = _paths_and_sizes(root, "HEAD")
    if head_tree is None:
        return _unevaluated("git ls-tree HEAD failed")

    violations: list[Violation] = []

    # (a) emptied files. Relative to the merge base or it alarms on the
    # zero-byte tracked files `main` already carries — the package `__init__.py`
    # markers plus `scripts/intel-pipeline/config/interest-profile.yml`, empty
    # since the initial snapshot, a number that moves whenever a package is
    # added and is asserted by shape in tests/test_automod_vet.py rather than
    # pinned here. Only files the diff touched are considered, which is what
    # keeps a mode-only change to an empty file out of it.
    for status, path in entries:
        if status == "D":
            continue
        at_head = head_tree.get(path)
        if at_head is None:
            # A diff entry with no HEAD blob: the tree moved while the vet read
            # it, or the entry is a delete the caller's `status` missed. Either
            # way the sizes no longer describe one commit, so say so.
            return _unevaluated(f"{path} is in the diff but not in HEAD's tree")
        at_base = base_tree.get(path)
        base_size = at_base[0] if at_base else 0
        if at_head[0] == 0 and base_size > 0:
            violations.append(Violation(
                EMPTY_FILE, path,
                f"non-empty ({base_size} B) at base {merge_base[:8]}, "
                f"0 bytes at HEAD"))

    # (b) stray compiled/binary artifacts in a newly-added file.
    for status, path in entries:
        if status != "A" or binary_allowed(path, binary_allowlist):
            continue
        blob = head_tree[path][1]
        has_nul = _has_nul(root, blob)
        if has_nul is None:
            return _unevaluated(f"could not read blob for added path {path}")
        if has_nul:
            violations.append(Violation(
                BINARY_ARTIFACT, path,
                f"newly added, NUL byte in its first {BINARY_SNIFF_BYTES} bytes, "
                f"{head_tree[path][0]} B, not under the binary allowlist"))

    # (e) an unbounded diff. The review rung caps what the grader *sees* at
    # `DIFF_CAP_CHARS` and stamps `base_truncated=true`; it never refuses, so a
    # 70,000-line change set used to land with the reviewer reading ~60 KB of
    # it. Nothing else in the ladder bounds total churn either.
    if lines > ceiling:
        violations.append(Violation(
            DIFF_TOO_LARGE, "",
            f"{lines} changed lines over the ceiling of {ceiling} "
            f"(automod.gate.max_diff_lines, default {DEFAULT_MAX_DIFF_LINES})"))

    return VetResult(
        status=EVALUATED, violations=violations,
        totals={"files": len(entries), "changed_lines": lines,
                "max_diff_lines": ceiling, "merge_base": merge_base})


if __name__ == "__main__":  # pragma: no cover - manual invocation
    import sys

    _base = sys.argv[1] if len(sys.argv) > 1 else "HEAD~1"
    _wt = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd()
    _res = vet_change_set(_base, _wt)
    print(json.dumps(_res.to_dict(), indent=2))
    sys.exit(2 if not _res.evaluated else (1 if _res.violations else 0))
