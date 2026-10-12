#!/usr/bin/env python3
"""Validate the nightly reflection handoff markdown artifact.

Purpose: Catch structural drift in the knowledge-handoff-YYYY-MM-DD.md emitted
by the Knowledge Analysis stage BEFORE the Write stage tries to consume it.

Exit code 0 = valid; 1 = missing required sections, malformed bullets, a
frontmatter `generated_at` that claims a time after the file was written, or a
`producing run record` finding: the autonomy run that wrote this artifact died
(`status: failed`) and no successful run has replaced the file since (#2614,
`check_producer_status`).

Intentionally forgiving on STRUCTURE: the 35B local model drifts on exact phrasing,
so we check for required section *presence* and flag parse issues — we do NOT reject
on cosmetic variation. We are equally silent about a stamp that merely LAGS its
file's mtime: that is the ordinary amend-in-place case and most of the corpus is
that way. Only a stamp claiming to exist before the file does is a finding
(`check_clock`, #2610).

What structure cannot answer, the producer's record answers instead. The chain
consumed a dead run's placeholder skeleton for 21 hours because a structurally
perfect document is exactly what a mid-write infra death leaves behind
(`run_42_20261011_050938`, #2614), so the one thing checked outside the file is
whether the run that produced it reported success — see the block above
`check_producer_status` for the predicate, the two things deliberately not treated
as signals, and why this file cannot import `app.autonomy` to ask.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# How far a `generated_at` may sit ahead of the file's own mtime before the stamp
# is dishonest rather than merely a few seconds early in the same call sequence.
# A tool call cannot interpolate a shell value into its own arguments, so the
# instant has to come from the call BEFORE the write — a `date -u` in the
# preceding call is the honest shape, and it lands a few seconds short of the
# write it describes. Zero tolerance would fail every real artifact for harness
# latency; 120 s is generous against what this exists to catch, which is a clock
# typed from a guess: +1063, +937 and +430 s in the three handoffs it would have
# failed, minutes off.
CLOCK_AHEAD_MAX_SECONDS = 120

# Required top-level (## H2) sections
REQUIRED_H2 = [
    # At least one `## Person: <name>` must exist — checked separately
    "## Vault Propagations",
    "## Tool Patterns — Failures",
    "## Tool Patterns — Successes",
    "## Conversation Patterns",
    "## Priority Order",
]

# Required H3 subsections under each `## Person:` block
REQUIRED_PERSON_H3 = [
    "### Mental Model — Decision Patterns",
    "### Mental Model — Communication Preferences",
    "### Mental Model — Technical Preferences",
    "### Mental Model — Project Prioritization",
    "### MEMORY.md Additions",
    "### MEMORY.md Removals",
    "### MEMORY.md Relocations",
    "### Profile Updates",
    "### Missing Files",
]

# Bullet patterns we care about (for sanity, not strict enforcement)
FREQUENCY_BARE_INT = re.compile(r"^\s*frequency:\s*(\d+)\s*$")
FREQUENCY_WITH_PAREN = re.compile(r"^\s*frequency:\s*(\d+)\s*[(\[].*")


def split_person_blocks(text: str) -> list[tuple[str, str]]:
    """Return list of (person_name, block_text) from ## Person: <name> sections."""
    blocks: list[tuple[str, str]] = []
    pattern = re.compile(r"^## Person:\s*(.+?)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        # Stop at next ## H2 (non-Person)
        tail = text[start:end]
        next_h2 = re.search(r"^##\s+(?!#)", tail, re.MULTILINE)
        if next_h2 and not tail[next_h2.start() : next_h2.start() + 10].startswith(
            "## Person:"
        ):
            tail = tail[: next_h2.start()]
        blocks.append((name, tail))
    return blocks


GENERATED_AT_KEY = re.compile(r"^generated_at:[ \t]*(.+?)[ \t]*$", re.MULTILINE)


def _parse_instant(raw: str) -> "datetime | None":
    """Parse a frontmatter instant, or None when it is not one.

    Three shapes occur in the live corpus: `...Z`, `...+00:00`, and (from a
    hand-typed pass) no offset at all. `datetime.fromisoformat` takes the first two
    natively on 3.12; `Z` is normalised because it is what most handoffs emit. A
    naive instant is read as UTC rather than local — this box runs at -0700, so
    reading it as local would move every honest stamp by seven hours and invent a
    failure out of a correct file, and UTC is what the key has always meant here.
    """
    text = raw.strip()
    if not text:
        return None
    if text[-1] in "zZ":
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def check_clock(path: Path) -> list[str]:
    """One finding when `generated_at` claims a time after the file exists.

    A comparison rather than a fourth prose rule, because the rules already
    existed and stopped nothing: three instructions across the reflection skills
    say never to type a clock by hand, and the validator still printed
    `[ok] … structure valid` and exited 0 over a stamp 1063 s inside its own file's
    future (#2610).

    Silent in the other three cases, each of which would be a fresh false failure.
    No `generated_at` key at all — structure does not require it, and signals
    reports name the same idea `generated:`. A stamp that will not parse — there is
    no instant there, so there is no claim to judge. And a stamp BEHIND its mtime by
    any amount, which is the ordinary amend-in-place and addendum-after-write case:
    nine of the fifteen real handoffs sit that way, so firing backwards would
    retroactively fail most of the corpus and stop the nightly chain. Bounding the
    stale direction (measured at -82,928 s in a signals report) is a separate
    ruling owed on #2610, not this check.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        # An unreadable file is main()'s own report; a clock finding on top of that
        # would be two messages about one broken read.
        return []
    match = GENERATED_AT_KEY.search(text)
    if not match:
        return []
    raw = match.group(1)
    generated = _parse_instant(raw)
    if generated is None:
        return []
    mtime = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
    delta = int((generated - mtime).total_seconds())
    if delta <= CLOCK_AHEAD_MAX_SECONDS:
        return []
    return [
        f"frontmatter `generated_at` is {delta:+d} s ahead of the file's own mtime, "
        f"past the {CLOCK_AHEAD_MAX_SECONDS} s tolerance: generated_at={raw} vs "
        f"mtime={mtime.strftime('%Y-%m-%dT%H:%M:%SZ')}. A file cannot be generated "
        f"after it was last written, so that stamp was typed rather than read. Take "
        f"`date -u` in the call immediately before the write — a tool call cannot "
        f"interpolate a shell value into its own arguments, so the preceding call is "
        f"the closest honest instant — and re-validate."
    ]


# ── the producer-status join: who wrote this, and did that run report OK? ─────
#
# Everything above asks the artifact about itself, and a run that died mid-write
# answers all of it correctly. `run_42_20261011_050938` (task #42, Knowledge
# Analysis) died of a ConnectTimeout at 342.9 s — `status: failed`,
# `failure_kind: infra` — after CREATING
# `~/lloyd-data/_pipeline/reflection/knowledge-handoff-2026-10-11.md` at 05:12:21Z
# with three pattern sections still at `- (none)`; its changes manifest holds that
# one `create` entry at `"reverted_at": null`, so the skeleton sat at the canonical
# path as the newest artifact in the window. This validator printed
# `[ok] knowledge-handoff-2026-10-11.md — structure valid` and exited 0 over it
# (#39's own note: "modified 2026-10-10 22:12:21 -0700, inside the 1,440-minute
# freshness window. Validation: pass"), the cycle concluded there was nothing to
# write, and the real handoff arrived at 11:23:17Z — 3 h 08 m 28 s after the write
# that should have consumed it, unread ~21 h after that (#2614).
#
# The three-way signature of a corpse is not in the file — a mid-write death leaves
# `status: in-progress`, which is what a healthy in-flight handoff says too — it is
# across the two stores the producer writes:
#
#   the file's mtime  <=  the newest run record's completed_at, that record's status
#   is `failed`,  AND  no successful run closed after the file's last write.
#
# The third clause is what makes it safe, and it is the one #2417's
# `_artifact_produced_during_run` already relied on: `mtime <= started_at` is the
# proven-absent half. A file touched at 11:20:00Z cannot have come from a run that
# closed at 05:15:21Z, so a retry in flight — whose own record does not exist yet,
# so the newest record is still the earlier death — is not refused for a file it is
# legitimately writing; #42's Step 0/Step 3 self-validation runs in exactly that
# state. And a death LATER than a successful write is silent for the same reason the
# `depends_on` rule releases a `failed` upstream whose last success names a real
# artifact (`app/autonomy.py:1962-1966`): "a run died" is not the predicate, "no
# successful run can claim this file" is.
#
# What is deliberately NOT a signal, both for stated reasons: `- (none)` content (a
# quiet day legitimately writes it — the analysis skill documents that a skeleton
# exits 0, and the report names emptiness as the bad fix), and any producer-side
# removal of a dead run's writes (no honest discriminator exists at death time;
# `tests/test_revert_by_run_id.py::test_autonomy_reverts_only_from_the_explicit_entry_point`
# pins the no-unattended-revert posture, which this file does not touch).
#
# `app.autonomy` cannot be imported from here: it pulls `agent_mcp._shared`, which
# needs `mcp`, and the nightly chain runs this script with `/usr/bin/python3`. So the
# two questions `app.autonomy.newest_run_record` (#2417) already asks are read again
# here over the same `autonomy-runs/<task>/run_*.md` files, and
# `tests/test_validate_handoff.py::test_the_validator_reads_the_newest_record_ex_as_app_autonomy_does`
# pins the reading to the module's own — same status word, same instant, same run id
# — so this stays a projection and not a second opinion. The bound that comes with
# it: like `newest_run_record`, this sees the plain `run_*.md` files and nothing the
# task manifest ledger holds but a worktree cleaned up (#2619), so a producer whose
# newest record lives only in a ledger answers "no verdict" rather than a refusal.

FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# The `output_artifact` placeholder the scheduler expands per run (`_artifact_candidates`
# in app/autonomy.py), so a declared template names a FAMILY of paths and `{date}` is a
# date wildcard — in either the local or the UTC spelling.
DATE_PLACEHOLDER = "{date}"
# A task file is `<id>-<name>.md`: the same filter `_iter_task_files_with_ledgers`
# applies, which is what skips `_config.md`, reports and notes on the board.
TASK_FILE_RE = re.compile(r"\d+-")
# The class marker the knowledge-write job routes on, and the only thing that tells
# this refusal apart from a structure or clock one in the printed list.
PRODUCER_MARKER = "producing run record"


def _utc_instant(dt: datetime) -> str:
    """One instant spelled the way `check_clock` spells an mtime: `%…T…Z`."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _front_matter_fields(text: str) -> dict[str, str]:
    """The leading `---` … `---` block as flat `key: value` pairs.

    Every key this join reads (`id`, `status`, `started_at`, `completed_at`,
    `output_artifact`) is a scalar, so there is no YAML and no venv import: nested
    structure is not part of the question. A value loses the quotes the writer adds —
    `_write_run_record` emits `completed_at: '2026-…+00:00'`, and `app.autonomy`'s own
    `_parse_iso` strips them too.
    """
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return {}
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        if key and key not in fields:      # first wins, as a YAML reader's top level does
            fields[key] = value.strip().strip("'\"")
    return fields


def _app_data_root():
    """The `app.data_root` module, or None when it cannot be reached.

    `app.data_root` and not `app.paths`: the latter imports the venv (see the block
    below). It is stdlib-only, so this works under `/usr/bin/python3`. The import is
    lazy and its failures are swallowed into a None because the structure checks must
    still run from any cwd with any repo layout — losing the producer question is
    clause 4's "no verdict", losing the validator entirely is not this file's call.
    """
    try:
        repo_root = str(Path(__file__).resolve().parents[1])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from app import data_root
        return data_root
    except Exception:
        return None


def _data_root() -> Path | None:
    """The runtime data root for THIS tree, via the stdlib-only resolver.

    Rule 3 of that resolver keeps a non-production checkout's data inside itself
    (`.lloyd-data/`), which is what a worktree wants; `DataRootMissing` — a production
    tree whose root lost its marker — answers None rather than raising, so a caller
    gets the same silence it would get from a board it could not read.
    """
    module = _app_data_root()
    if module is None:
        return None
    try:
        return module.resolve_data_root_for_tree(Path(__file__).resolve().parents[1])
    except Exception:
        return None


def _vault_root() -> Path | None:
    """The vault holding `autonomy/`, honouring `app.data_root.VAULT_ROOT_ENV`.

    Read per call, not once: `vault_root()` resolves `LLOYD_VAULT_ROOT` (or
    `Path.home()/obsidian`) at call time, which is the seam the tests redirect through.
    """
    module = _app_data_root()
    if module is None:
        return None
    try:
        return module.vault_root()
    except Exception:
        return None


def iter_tasks(*, autonomy_dir: Path | None = None) -> list[dict]:
    """Every task on the board that declares an `output_artifact`.

    `autonomy_dir` defaults to `<vault>/autonomy` — where the task files live, which
    is NOT under the data root, so the join needs both roots. An unreadable board is
    an empty list: clause 4's "no verdict", not a refusal.
    """
    board = autonomy_dir
    if board is None:
        vault = _vault_root()
        if vault is None:
            return []
        board = Path(vault) / "autonomy"
    tasks: list[dict] = []
    try:
        paths = sorted(Path(board).glob("*.md"))
    except OSError:
        return []
    for path in paths:
        if not TASK_FILE_RE.match(path.name):
            continue
        try:
            fields = _front_matter_fields(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if not fields.get("output_artifact"):
            continue
        fields.setdefault("task_file", str(path))
        tasks.append(fields)
    return tasks


def producing_task_for(artifact, *, autonomy_dir: Path | None = None) -> dict | None:
    """The task that declares `artifact` as its `output_artifact`, or None.

    The reverse of `_artifact_candidates`, which goes task → candidate paths: a board
    stores the TEMPLATE (`knowledge-handoff-{date}.md`) and the file on disk is one of
    its expansions, so `{date}` becomes a date wildcard. An unbound `{date}` would be
    `.*`, which matches any path containing the template's literal halves — so the
    wildcard is a date, deliberately: one task's `{date}` cannot swallow another's
    artifact. A literal `output_artifact` (no placeholder) matches only itself.
    """
    target = Path(artifact).resolve()
    for task in iter_tasks(autonomy_dir=autonomy_dir):
        template = str(task["output_artifact"]).strip()
        if not template:
            continue
        if DATE_PLACEHOLDER not in template:
            if Path(template).resolve() == target:
                return task
            continue
        stem, _, tail = template.partition(DATE_PLACEHOLDER)
        try:
            pattern = re.compile(rf"^{re.escape(stem)}\d{{4}}-\d{{2}}-\d{{2}}{re.escape(tail)}$")
        except re.error:
            continue
        if pattern.match(str(target)):
            return task
    return None


def run_records(task_id, *, runs_dir: Path | None = None) -> list[dict]:
    """`autonomy-runs/<task>/run_*.md` newest first, each with its own front matter.

    The same directory walk `app.autonomy._newest_run_record` performs and the same
    tiebreaks: newest by `completed_at` or `started_at`, falling back to the file's
    mtime for a record mid-flight with no stamp, `run_id` from the front matter or the
    filename stem. It does NOT skip a record for having no readable `status:` —
    `_newest_run_record` skips those and a caller that then asks "what succeeded"
    would silently miss it — it keeps them and lets `status` read `""`, so the caller
    answers "the newest record reports nothing", which is the same non-verdict.
    """
    root = runs_dir
    if root is None:
        data = _data_root()
        if data is None:
            return []
        root = Path(data) / "autonomy-runs"
    task_dir = Path(root) / str(task_id)
    found: list[dict] = []
    try:
        paths = sorted(p for p in task_dir.glob("run_*.md") if not p.name.startswith("."))
    except OSError:
        return []
    for path in paths:
        try:
            fields = _front_matter_fields(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        # `_newest_run_record`'s own order: completed_at, else started_at, else the
        # file's mtime — and the same reader for the same stamps, not a second one.
        when = _parse_instant(
            str(fields.get("completed_at") or fields.get("started_at") or ""))
        if when is None:
            try:
                when = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue
        found.append({
            "path": path,
            "run_id": fields.get("run_id") or path.stem,
            "status": str(fields.get("status") or "").strip(),
            "when": when,
        })
    found.sort(key=lambda r: r["when"], reverse=True)
    return found


def check_producer_status(artifact, *, autonomy_dir: Path | None = None,
                          runs_dir: Path | None = None) -> list[str]:
    """Findings for "the run that produced this artifact is dead and nothing replaced it".

    Call this with the artifact path; the return is a list of findings in the same
    shape as `check_clock`'s, so a caller can print it, extend a list with it, or act
    on the class — the phrase `producing run record` is the class marker the knowledge
    write job routes on. Empty list means no claim against the file, and it is silent
    for every one of these, each of which is clause 4's "validates as today": no task
    declares the path; the producing task has no run records; the newest record has no
    `status:` to read; the roots cannot be resolved. Absence is never evidence of a
    failure — a first run's artifact would otherwise refuse itself.
    """
    path = Path(artifact)
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return []
    task = producing_task_for(path, autonomy_dir=autonomy_dir)
    if task is None:
        return []
    records = run_records(task.get("id") or "?", runs_dir=runs_dir)
    if not records:
        return []
    newest = records[0]
    # An unreadable newest record is a missing answer, not a failed one.
    if newest["status"] != "failed":
        return []
    # The bound: a file newer than this run's own close cannot be its output, so the
    # newest death is somebody else's news (see the header — the retry-in-flight case).
    if mtime > newest["when"]:
        return []
    claimant = next((r for r in records
                     if r["status"] == "success" and r["when"] > mtime), None)
    if claimant is not None:
        return []
    lag = int((newest["when"] - mtime).total_seconds())
    return [
        f"{PRODUCER_MARKER} {newest['path'].name} (task #{task.get('id')}) reads "
        f"status: failed, closed at {_utc_instant(newest['when'])} — "
        f"{lag}s after this artifact was last written at {_utc_instant(mtime)} — and "
        f"no successful run closed after that write. This is the dead run's partial "
        f"output, not a handoff: treat it as absent and take the no-handoff path. "
        f"Producer's record: {newest['path']}"
    ]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_handoff.py <path-to-handoff.md>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1]).expanduser()
    if not path.exists():
        print(f"[fail] handoff artifact not found: {path}", file=sys.stderr)
        return 1

    text = path.read_text(encoding="utf-8")
    issues: list[str] = []

    # Frontmatter sanity
    if not text.lstrip().startswith("---"):
        issues.append("missing YAML frontmatter delimiter at top (`---`)")

    # H1 title check
    if not re.search(r"^# Knowledge Handoff", text, re.MULTILINE):
        issues.append("missing `# Knowledge Handoff` H1 header")

    # At least one Person block
    person_blocks = split_person_blocks(text)
    if not person_blocks:
        issues.append("no `## Person: <name>` section found")

    # Required H2 sections
    for heading in REQUIRED_H2:
        if heading not in text:
            issues.append(f"missing required section: `{heading}`")

    # Per-person H3 sections
    for name, block in person_blocks:
        for h3 in REQUIRED_PERSON_H3:
            if h3 not in block:
                issues.append(f"person `{name}`: missing `{h3}`")

    # Frequency fields must be bare integers (catches "frequency: 4 (tasks ...)" drift)
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("frequency:"):
            if FREQUENCY_WITH_PAREN.match(line):
                issues.append(
                    f"line {lineno}: `frequency:` has non-integer trailing text "
                    f"({line.strip()!r}). Move narrative to a separate bullet."
                )
            elif not FREQUENCY_BARE_INT.match(line):
                issues.append(
                    f"line {lineno}: `frequency:` is not a bare integer "
                    f"({line.strip()!r})"
                )

    # The clock last and into the same list, not printed and returned early: a
    # file with a missing section AND a future stamp has two problems, and
    # reporting one teaches the reader to expect one (#2610).
    issues.extend(check_clock(path))

    # …and the producer's terminal record beside it, for the same reason. The
    # knowledge-write job's Step 1 treats these two classes differently — a
    # structural or clock failure is flagged and processed best-effort, a
    # producer-status refusal means the artifact is not a handoff at all — so they
    # share a list and are told apart by the `producing run record` marker.
    issues.extend(check_producer_status(path))

    if issues:
        print(f"[fail] {len(issues)} validation issue(s) in {path.name}:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print(f"[ok] {path.name} — structure valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
