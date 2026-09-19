#!/usr/bin/env python3
"""skill_verdicts.py — a keyed ledger of skill-candidate verdicts (backlog #530).

The nightly skill pipeline adjudicates candidates and then forgets it. Every
disposition is written down — `REVIEW-LOG.md`, and a `status:` line in each
candidate's frontmatter — and none of it is read back before the next run.
`mine-trajectories.py` regenerates a date-stamped candidate file per pattern each
night, so yesterday's `reviewed_no_skill` never reaches today's input, and the
consolidator's own skip list (`nightly-skill-consolidation/SKILL.md` 1.3) named only
`consolidated` and `noise` while the statuses actually in use were
`rejected_false_positive` (12 files) and `reviewed_no_skill` (22). The result is on
disk: all seven error patterns rejected on 2026-09-06 came back as fresh files on
2026-09-09 and were hand-rejected again, one `review_reason` at a time. Run #57
named this in its own words on 2026-09-01: "the miner regenerated both files as
`pending_review`, silently wiping the previous review — that's the real process
bug here."

WikiSkill's Skill Proposer solves the same problem with one read: its step 2 is
"read the impact log to see what was tried before, including the full context of
rejected proposals", and its step 5 permits "no action at all" as a legal answer.
Its load-bearing ablation — the wiki helps only when the *proposer* reads it, and
handing it to the executor makes things worse — says the value is in accumulated,
already-adjudicated knowledge, not in rewriting skills more often. So this ledger is
a development-side instrument. It must never be injected into a runtime prompt.

Two invariants:

* **Append-only, latest-wins per key.** A verdict is never edited or deleted; a
  reopen is a new line. Rollback is asymmetric the way WikiSkill's is: a skill can
  be reverted, the record of why it was rejected never is.
* **`evidence_cmd` is required.** A prose verdict is an assertion the next run can
  only inherit. A re-executable check is what lets a later run *falsify* it, which is
  the weakness #525 records on the worker-evidence path.
* **The ledger exists in two trees (#772).** This store lives under `_pipeline/`,
  which `.gitignore` excludes from every repo on the box and no backup job reads, so
  one truncated append or one `rm -rf _pipeline` used to erase every decision with no
  way to re-derive the reasons — they are prose that lives nowhere else. `record`
  therefore appends the identical line to a durable copy under the vault root (the
  15-minute vault snapshotter covers it), seeding that copy from this file the first
  time it is written so no pre-existing verdict is missing, and copies beside it any
  script a stored `evidence_cmd` names that no repo already tracks. `check` reads the
  mirror when this file is gone and says it did, so a wiped ledger is an announced
  incident rather than `skipped_by_verdict: 0`. Seeding and appending keep the copy a
  superset of the live file; neither keeps the *live* file honest, so `check` also
  compares the two latest-per-key tables every run and prints `MIRROR_MISSING`,
  `LEDGER_LOST` or `LEDGER_MIRROR_CONFLICT` per disagreeing key
  (`check_against_mirror`). Without that, a copy that quietly fell behind — or a live
  ledger that quietly lost one line while still existing — prints precisely the counts
  two healthy trees would.

Stickiness is capped, because a verdict that outlives its evidence becomes a veto on
real work: a terminal verdict reopens itself `REOPEN_AFTER_DAYS` after it was
decided, or immediately once the pattern's live `occurrences` exceed
`REOPEN_OCCURRENCE_GROWTH` times the count at the time of the decision.

Field names deliberately match the target-keyed decision ledger proposed in #588
(`pattern_key` there is its `target_key`), which explicitly defers to this item for
the skill-candidate domain and is meant to absorb this store, not compete with it.
Do not build a second one.

Usage:
    skill_verdicts.py check   --candidates DIR    # what must not be proposed again
    skill_verdicts.py record  --pattern K --verdict V --reason R --evidence-cmd C
    skill_verdicts.py seed    --candidates DIR    # harvest existing dispositions once
    skill_verdicts.py list                        # latest-wins view
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Append-only. `~/lloyd/_pipeline` is gitignored, same as `REVIEW-LOG.md` beside it.
DEFAULT_STORE = Path.home() / "lloyd" / "_pipeline" / "skills" / "reviews" / "verdicts.jsonl"
DEFAULT_CANDIDATES = Path.home() / "lloyd" / "_pipeline" / "skills" / "candidates"

# The durable copy (#772), derived from the vault root like every other vault writer
# (`scripts/skill_lint.py:34`, `app/paths.py:10`). Deliberately NOT under
# `_pipeline/`: a mirror inside the tree it protects goes down with it. A path under
# `skills/<slug>/` would be load-validated by the vault lander as a skill, so this
# non-prose payload gets its own directory under `memory/`, where `memory/facts/`
# already sets the precedent for machine-written data files.
MIRROR_FILENAME = "verdicts.jsonl"
DEFAULT_MIRROR = Path.home() / "obsidian" / "memory" / "skill-verdicts" / MIRROR_FILENAME

# How long one stored `evidence_cmd` may take before `check` stops waiting for it.
# Measured on the live ledger 2026-09-19: all 75 blocking keys re-execute in 2.0 s
# together, so this bounds a hung grep, not a slow pipeline.
EVIDENCE_TIMEOUT_SECONDS = 15

# A path token inside a stored `evidence_cmd` naming a script (clause 5, #772).
_SCRIPT_TOKEN_RE = re.compile(r"([^\s'\"`;|&()<>]+[^\s'\"`;|&()<>]*\.(?:py|sh))")

# Bash's own complaint that it could not parse the command (`bash: -c: line 1:
# unexpected EOF while looking for matching '"'`). Matched on the `bash: -c:` prefix so a
# tool inside the command that exits 2 for its own reasons is still read as a result.
_BASH_PARSE_ERROR_RE = re.compile(r"bash: -c: line \d+: (unexpected EOF|syntax error)")

# Verdicts that mean "this pattern has been adjudicated; do not propose it again".
# `reviewed_no_skill` covers "real signal, an installed skill already owns it", which
# is the most common one on this board and the one that kept coming back.
TERMINAL_VERDICTS = frozenset({
    "rejected_false_positive",
    "reviewed_no_skill",
    "rejected_unverifiable",
    "rejected_artifact_class",
    "archived_content",
})

# A terminal verdict that has aged out or whose evidence has grown >10x is reopened
# rather than trusted. 60 days is roughly two full re-emergence cycles on this board.
REOPEN_AFTER_DAYS = 60
REOPEN_OCCURRENCE_GROWTH = 10.0

# The status the miner writes instead of `pending_review` when a key is terminal.
SUPERSEDED_STATUS = "superseded_by_verdict"

# Frontmatter keys, in order. `pattern_key` is the join key everywhere.
FIELDS = (
    "pattern_key",
    "verdict",
    "reason",
    "evidence_cmd",
    "occurrences_at_decision",
    "decided_at",
    "decided_by",
    "source_candidate",
)

_STATUS_RE = re.compile(r"^status:\s*(.+?)\s*$", re.MULTILINE)
_PATTERN_RE = re.compile(r"^pattern:\s*(\S+)", re.MULTILINE)
_OCCURRENCES_RE = re.compile(r"^occurrences:\s*(\d+)", re.MULTILINE)


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(value: str) -> datetime | None:
    """Parse a decided_at stamp. Returns None if unparseable — a verdict whose date
    cannot be read is treated as un-expiring rather than silently reopened, because
    the reopen path is the one that can re-mint a rejected skill."""
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        stamp = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def store_path(explicit: str | None = None) -> Path:
    """Explicit path, else $SKILL_VERDICTS_STORE, else the default. The env var exists
    so the nightly job (and a test) can point the ledger somewhere else without a
    code change — `mine-trajectories.py` calls this with no argument."""
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("SKILL_VERDICTS_STORE")
    return Path(env).expanduser() if env else DEFAULT_STORE


def mirror_path(explicit: str | None = None) -> Path:
    """Explicit path, else $SKILL_VERDICTS_MIRROR, else the vault copy (#772).

    Derived from the vault root, not from the store: the mirror has to survive whatever
    happens to `_pipeline`, so it can never be computed from it. This is the *path*
    accessor; `_mirror_target` is the one that decides whether a given ledger has a
    mirror at all, and it is what stops a scratch `--store` from being written through
    the vault copy of the production ledger.
    """
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("SKILL_VERDICTS_MIRROR")
    return Path(env).expanduser() if env else DEFAULT_MIRROR


def _mirror_target(live: Path) -> Path | None:
    """The durable copy that belongs to `live`, or None when this ledger has none.

    Without an explicit `$SKILL_VERDICTS_MIRROR`, the vault copy mirrors the **default**
    ledger and nothing else. That guard is what keeps a test, or a runbook that points
    `--store` at a scratch file, from appending scratch decisions — or reading them back —
    through the real vault copy of the production verdicts. An explicit env path means the
    caller named both ends and gets exactly that pairing.
    """
    env = os.environ.get("SKILL_VERDICTS_MIRROR")
    if env:
        return Path(env).expanduser()
    return DEFAULT_MIRROR if live == DEFAULT_STORE else None


def resolve_verdict_source(store: Path | str | None = None) -> tuple[Path, bool]:
    """Where to read verdicts from now: `(path, fell_back_to_mirror)`.

    The live ledger when it exists — including when it exists but is empty, since an
    emptied file is a different event from a deleted one and must not be papered over.
    When it is gone, the durable copy (#772): a wiped `_pipeline` used to make every
    rejected pattern look freshly-adjudicated-free, and `check` printed
    `skipped_by_verdict: 0` as though the pipeline had never said no to anything. The
    caller prints the fact — `cmd_check` puts it on stdout beside the counts, where the
    nightly will see it — and a ledger missing from *both* trees returns the live path
    with nothing loaded, which is the honest empty state.
    """
    live = store_path(store)
    if live.exists():
        return live, False
    durable = _mirror_target(live)
    return (durable, True) if durable is not None and durable.is_file() else (live, False)


def _seed_mirror(live: Path, durable: Path) -> None:
    """Copy every line the live ledger holds into a mirror that holds none (#772).

    Seeding is what makes the mirror complete for verdicts recorded *before* this
    change existed; it is a one-time copy of an append-only file, so the mirror keeps
    its own history of lines the live ledger may later lose.
    """
    if durable.exists() or not live.exists():
        return
    durable.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(live, durable)


def _mirror_script(script: Path, mirror_dir: Path) -> None:
    """Copy a script a stored `evidence_cmd` names into the mirror directory (#772).

    The ledger's re-executability is a second artifact: 20 of its 81 keys invoke a
    script under `_pipeline/skills/tools/`, so a wipe of that tree would leave the
    verdict binding while the check meant to overturn it could no longer run — the
    fail-shut half of #772. Only scripts with no second copy are mirrored: a tracked
    file (the ledger also names `scripts/mine-trajectories.py`) is already durable, and
    copying it here would put a second, forkable copy of a live module in the vault.
    """
    if not script.is_file():
        return
    if _git_tracked(script):
        return
    dest = mirror_dir / script.name
    if dest.is_file() and dest.read_bytes() == script.read_bytes():
        return
    mirror_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(script, dest)


def _git_tracked(path: Path) -> bool:
    """Whether some git repo already versions `path`. Unaskable means untracked, so an
    absent git binary errs toward mirroring rather than toward a single copy."""
    try:
        return subprocess.run(
            ["git", "-C", str(path.parent), "ls-files", "--error-unmatch", path.name],
            capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def named_scripts(evidence_cmd: str) -> list[Path]:
    """Existing script files a stored `evidence_cmd` names (#772 clause 5).

    Tokenised on shell metacharacters rather than parsed as a shell: the commands in
    the ledger are greps and `python3 <script>` calls, and a real parser would be a
    bigger surface than the thing it reads. Only paths that are files right now are
    returned — a command may name a snapshot chosen at run time that no longer exists.
    """
    out: list[Path] = []
    for token in _SCRIPT_TOKEN_RE.findall(evidence_cmd or ""):
        candidate = Path(token).expanduser()
        if candidate.is_file() and candidate not in out:
            out.append(candidate)
    return out


def mirror_for(store: Path) -> Path | None:
    """Open the durable copy that belongs to `store`, seeding it, or None if it has none.

    Call this **before** appending the new line to the ledger, then append the same line
    to the returned path: seeding after the ledger grew would copy the new line as
    history and then write it a second time, so the mirror would hold a duplicate of the
    newest verdict and no duplicate of anything else — a divergence invisible to
    `check`, which reads only the last line per key.

    Mirroring never refuses the record: the ledger is the live artifact, and a vault that
    cannot be written costs the second copy, not the decision — which is what the
    fallback line in `check` and the restore README are for.
    """
    durable = _mirror_target(store)
    if durable is None:
        return None
    durable.parent.mkdir(parents=True, exist_ok=True)
    _seed_mirror(store, durable)
    return durable



def load_verdicts(store: Path) -> dict[str, dict]:
    """Latest line per `pattern_key`. Missing store is an empty ledger, not an error;
    an unparseable line is skipped with a warning rather than dropping the ledger."""
    verdicts: dict[str, dict] = {}
    path = Path(store)
    if not path.exists():
        return verdicts
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print(f"  warn: {path}:{lineno} is not JSON, skipped", file=sys.stderr)
            continue
        key = (row.get("pattern_key") or "").strip()
        if not key:
            print(f"  warn: {path}:{lineno} has no pattern_key, skipped", file=sys.stderr)
            continue
        verdicts[key] = row  # append-only file, so last write is the live verdict
    return verdicts


def divergence_lines(live_table: dict[str, dict], durable_table: dict[str, dict],
                     durable: Path) -> list[str]:
    """One line per key where the two trees no longer hold the same latest verdict.

    Three shapes, because the ledger is append-only latest-wins and a disagreement has
    three causes: the durable copy never received a line (`MIRROR_MISSING` — a writer
    appended to the live file without going through `record`, which is the only
    two-tree writer), the live file *lost* a line the copy still holds
    (`LEDGER_LOST` — the file's only documented writer appends, but that is a convention
    and not an enforcement, and `resolve_verdict_source` cannot see the gap while the file
    still exists), or both trees hold the key with
    different content (`LEDGER_MIRROR_CONFLICT` — one side took a reopen the other did
    not). An empty list is the only healthy answer, and it is also the case that makes
    this worth printing: two silently-diverging copies are exactly the state #772
    describes, where a restore would look like it worked.
    """
    lines: list[str] = []
    for key in sorted(set(live_table) | set(durable_table)):
        live_row, durable_row = live_table.get(key), durable_table.get(key)
        if durable_row is None:
            lines.append(
                f"MIRROR_MISSING {key} :: {live_row.get('verdict')} decided "
                f"{live_row.get('decided_at')} is in the live ledger but not in the "
                f"durable copy {durable}")
        elif live_row is None:
            lines.append(
                f"LEDGER_LOST {key} :: {durable_row.get('verdict')} decided "
                f"{durable_row.get('decided_at')} is in the durable copy {durable} "
                f"but the live ledger no longer holds it")
        elif live_row != durable_row:
            lines.append(
                f"LEDGER_MIRROR_CONFLICT {key} :: live={live_row.get('verdict')} "
                f"durable={durable_row.get('verdict')} decided "
                f"{live_row.get('decided_at')}/{durable_row.get('decided_at')}")
    return lines


def check_against_mirror(live: Path, live_table: dict[str, dict]) -> list[str]:
    """The live ledger's own copy, read as a second opinion rather than as a fallback.

    `resolve_verdict_source` answers from the mirror only when the live file is *entirely*
    absent, so a one-line disappearance prints the same counts as if nothing happened
    (#772). This is the companion that has no fallback in it: it compares the two trees
    on every run and returns what to print. No mirror for this ledger, or none on disk
    yet, is not a divergence — it is the pre-#772 single-copy state, and `check` would
    otherwise alarm on every scratch `--store`.
    """
    durable = _mirror_target(live)
    if durable is None or durable == live or not durable.is_file():
        return []
    return divergence_lines(live_table, load_verdicts(durable), durable)


def is_terminal(row: dict | None) -> bool:
    return bool(row) and (row.get("verdict") or "").strip() in TERMINAL_VERDICTS


def reopen_reason(row: dict, occurrences: int = 0, now: datetime | None = None) -> str | None:
    """Why a terminal verdict no longer binds, or None while it still does."""
    now = now or datetime.now(tz=timezone.utc)
    decided = _parse_ts(row.get("decided_at") or "")
    if decided is None:
        return None
    if now - decided > timedelta(days=REOPEN_AFTER_DAYS):
        return f"expired after {REOPEN_AFTER_DAYS}d"
    baseline = row.get("occurrences_at_decision") or 0
    try:
        baseline = int(baseline)
    except (TypeError, ValueError):
        baseline = 0
    if baseline > 0 and occurrences > baseline * REOPEN_OCCURRENCE_GROWTH:
        return (
            f"occurrences grew {occurrences}>{int(baseline * REOPEN_OCCURRENCE_GROWTH)}"
            f" (>10x the {baseline} at decision)"
        )
    return None


def terminal_verdict(
    pattern_key: str,
    store: Path | str | None = None,
    occurrences: int = 0,
    verdicts: dict[str, dict] | None = None,
    now: datetime | None = None,
) -> dict | None:
    """The binding terminal verdict for `pattern_key`, or None.

    None means "propose freely": no line, a non-terminal line, or a line the reopen
    rules have lifted. A ledger missing from *both* trees returns None for every key;
    a ledger missing from the live tree alone is read from the durable copy (#772), so
    a wiped `_pipeline` no longer re-mints every rejected pattern through the miner.
    """
    table = (verdicts if verdicts is not None
             else load_verdicts(resolve_verdict_source(store)[0]))
    row = table.get((pattern_key or "").strip())
    if not is_terminal(row):
        return None
    if reopen_reason(row, occurrences=occurrences, now=now) is not None:
        return None
    return row


def record_verdict(
    store: Path | str | None = None,
    pattern_key: str = "",
    verdict: str = "",
    reason: str = "",
    evidence_cmd: str = "",
    occurrences: int = 0,
    decided_by: str = "agent",
    source_candidate: str = "",
    decided_at: str | None = None,
) -> dict:
    """Append one decision. Never mutates an earlier line.

    `reason` and `evidence_cmd` are required: without the second, a verdict is an
    assertion a later run can only inherit.

    The line lands in two trees (#772): the ledger, then the durable copy, which is
    seeded from the ledger first so a verdict recorded before the mirror existed is in
    it too. A script the new `evidence_cmd` names is copied beside the mirror.
    Mirroring never refuses the record: the ledger is the live artifact, and a vault
    that cannot be written must not lose a decision.
    """
    if not (pattern_key := pattern_key.strip()):
        raise ValueError("pattern_key is required")
    if not (verdict := verdict.strip()):
        raise ValueError("verdict is required")
    if not (reason := reason.strip()):
        raise ValueError(
            "reason is required — a verdict with no stated reason becomes a bare veto"
        )
    if not (evidence_cmd := evidence_cmd.strip()):
        raise ValueError(
            "evidence_cmd is required — a verdict with no re-executable check cannot "
            "be falsified by a later run, only inherited (#525)"
        )
    row = {
        "pattern_key": pattern_key,
        "verdict": verdict,
        "reason": " ".join(reason.split()),
        "evidence_cmd": evidence_cmd,
        "occurrences_at_decision": int(occurrences or 0),
        "decided_at": decided_at or now_iso(),
        "decided_by": decided_by,
        "source_candidate": source_candidate,
    }
    path = store_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    # Seed first, then append to both (`mirror_for` says why the order is load-bearing).
    durable = mirror_for(path)
    for target in (path, durable):
        if target is None:
            continue
        with target.open("a", encoding="utf-8") as fh:
            fh.write(line)
    if durable is not None:
        for script in named_scripts(evidence_cmd):
            _mirror_script(script, durable.parent)
    return row


def read_candidate(file: Path) -> tuple[str, str, int]:
    """(pattern_key, status_token, occurrences) from one candidate's frontmatter."""
    text = Path(file).read_text(encoding="utf-8", errors="replace")
    # Frontmatter only: `status:` also appears in mined body examples.
    head = text.split("---", 2)[1] if text.startswith("---") else text[:2000]
    pattern = _PATTERN_RE.search(head)
    status = _STATUS_RE.search(head)
    occ = _OCCURRENCES_RE.search(head)
    status_token = status.group(1).split()[0].strip("_*` ") if status else ""
    return (
        pattern.group(1) if pattern else "",
        status_token,
        int(occ.group(1)) if occ else 0,
    )


# `bash`'s own "command not found" status, reused as the answer "this stored check
# cannot be executed at all" — as opposed to rc 1, which is the check running and
# reporting its assertion false, which is a falsification and not a defect.
UNRUNNABLE = 127


def evidence_cmd_status(row: dict, timeout: int = EVIDENCE_TIMEOUT_SECONDS):
    """Re-execute a verdict's stored check: `(rc, detail)`, rc `UNRUNNABLE` for a command
    that cannot produce a verdict at all — absent, unparseable, unstartable, or hung.

    Only `UNRUNNABLE` is reported by `check`. The distinction is the point: a falsifier
    that exits 1 says the verdict's grounds no longer hold and is the ledger working,
    while one that exits 127, or whose `No such file or directory` says the script the
    command names is gone, means the verdict is binding on a check nobody can run any
    more — which is invisible from the ledger alone (#772: `scan_candidates` never
    executed these commands, so a wiped falsifier silenced nothing and the verdict kept
    suppressing candidates).
    """
    cmd = (row.get("evidence_cmd") or "").strip()
    if not cmd:
        return UNRUNNABLE, "no evidence_cmd recorded"
    try:
        proc = subprocess.run(["bash", "-c", cmd], capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # Reported as unrunnable rather than passed over: a check that never finishes
        # disarms a verdict exactly as quietly as one that cannot start, and the bounded
        # timeout is worthless if nothing is printed when it fires.
        return UNRUNNABLE, f"still running after {timeout}s"
    except OSError as exc:  # bash itself unusable: no verdict either way
        return UNRUNNABLE, str(exc)
    if proc.returncode == UNRUNNABLE:
        return UNRUNNABLE, (proc.stderr.strip().splitlines() or ["exit 127"])[0]
    missing = next((ln for ln in proc.stderr.splitlines()
                    if "No such file or directory" in ln), "")
    if missing:
        return UNRUNNABLE, missing.strip()
    # A command the shell itself refuses to parse can never run either, and this is not
    # speculation about an exit code: as measured on the live ledger 2026-09-19, the key
    # `seq-2-read-edit` — whose falsifier the 09-15 triage confirmed re-executed at rc 0 —
    # exits 2 with `bash: -c: line 1: unexpected EOF while looking for matching '"'`,
    # because the stored string lost a quote. Detecting only exit 127 would leave the one
    # falsifier that is actually broken on this box silent, which is the fail-shut case
    # clause 4 exists to surface. Keyed on bash's own prefix, not on rc 2 at large: a tool
    # inside the command may exit 2 for its own reasons, and that is a result, not a fault.
    if proc.returncode == 2 and _BASH_PARSE_ERROR_RE.search(proc.stderr):
        return UNRUNNABLE, (proc.stderr.strip().splitlines() or ["bash parse error"])[0]
    return proc.returncode, ""


def scan_candidates(candidates_dir: Path, store: Path | str | None = None, *,
                    table: dict[str, dict] | None = None) -> list[dict]:
    """One row per candidate file: its key, whether a verdict blocks it, and why.

    `table` lets a caller that already resolved the verdict source (see
    `resolve_verdict_source`, and `cmd_check`) hand the loaded ledger over instead of
    reading the default store a second time.
    """
    rows = []
    if table is None:
        table = load_verdicts(resolve_verdict_source(store)[0])
    for file in sorted(Path(candidates_dir).glob("candidate-*.md")):
        key, status, occurrences = read_candidate(file)
        if not key:
            continue
        row = table.get(key)
        terminal = is_terminal(row)
        lift = reopen_reason(row, occurrences=occurrences) if terminal else None
        rows.append({
            "file": file.name,
            "pattern_key": key,
            "status": status,
            "occurrences": occurrences,
            "verdict": (row or {}).get("verdict", ""),
            "reason": (row or {}).get("reason", ""),
            "decided_at": (row or {}).get("decided_at", ""),
            "blocked": bool(terminal and lift is None),
            "reopened": lift or "",
        })
    return rows


def cmd_check(args: argparse.Namespace) -> int:
    live = store_path(args.store)
    source, fell_back = resolve_verdict_source(args.store)
    table = load_verdicts(source)
    rows = scan_candidates(Path(args.candidates).expanduser(), table=table)
    skipped = [r for r in rows if r["blocked"]]
    for row in rows:
        if row["blocked"]:
            print(
                f"SKIP {row['pattern_key']} :: {row['verdict']} :: {row['reason']} "
                f"(decided {row['decided_at']}, file {row['file']})"
            )
        elif row["reopened"]:
            print(f"REOPEN {row['pattern_key']} :: {row['reopened']}")
        else:
            print(f"PROCEED {row['pattern_key']}")
    # Both findings print above the totals, and the totals stay the last line: the
    # runbook and the nightly greps read the counts off `splitlines()[-1]`, so a
    # warning that displaced them would break a reader that is not looking for it.
    if not fell_back:  # when the copy *is* the source there is nothing to compare it with
        for line in check_against_mirror(live, table):
            print(line)
    for key in sorted({r["pattern_key"] for r in skipped}):
        rc, detail = evidence_cmd_status(table.get(key) or {})
        if rc == UNRUNNABLE:
            print(f"EVIDENCE_CMD_UNRUNNABLE {key} :: {detail}")
    if fell_back:
        print(f"verdict source: {source} (live ledger {live} is absent)")
    print(f"checked: {len(rows)}  skipped_by_verdict: {len(skipped)}")
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    try:
        row = record_verdict(
            store=args.store,
            pattern_key=args.pattern,
            verdict=args.verdict,
            reason=args.reason,
            evidence_cmd=args.evidence_cmd,
            occurrences=args.occurrences,
            decided_by=args.decided_by,
            source_candidate=args.source_candidate,
        )
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(f"recorded {row['pattern_key']} -> {row['verdict']} at {row['decided_at']}")
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    """Harvest dispositions already on disk into the ledger, once.

    Reads the `status:` line of each candidate — the same hand-written verdicts the
    runbook used to re-make every night — and gives each one its re-executable check:
    the grep that re-derives that status from that file. Duplicate keys keep the
    newest file's disposition. Skips keys already in the ledger, so re-seeding after
    the key derivation changes is a re-run, not a cleanup.
    """
    table = load_verdicts(resolve_verdict_source(args.store)[0])
    newest: dict[str, tuple[Path, str, int]] = {}
    for file in sorted(Path(args.candidates).expanduser().glob("candidate-*.md")):
        key, status, occurrences = read_candidate(file)
        if key and is_terminal({"verdict": status}):
            prev = newest.get(key)
            if prev is None or file.name > prev[0].name:
                newest[key] = (file, status, occurrences)
    seeded = 0
    for key, (file, status, occurrences) in sorted(newest.items()):
        if key in table:
            print(f"  keep: {key} already in the ledger ({table[key]['verdict']})")
            continue
        text = file.read_text(encoding="utf-8", errors="replace")
        head = text.split("---", 2)[1] if text.startswith("---") else ""
        raw = _STATUS_RE.search(head)
        reason = raw.group(1).strip() if raw else status
        reason = reason.split("—", 1)[1].strip() if "—" in reason else reason
        record_verdict(
            store=args.store,
            pattern_key=key,
            verdict=status,
            reason=reason or status,
            evidence_cmd=f"grep -m1 '^status:' {file}",
            occurrences=occurrences,
            decided_by="seed-from-candidates",
            source_candidate=file.name,
        )
        seeded += 1
        print(f"  seed: {key} -> {status}")
    print(f"seeded: {seeded}  ledger now: {len(load_verdicts(resolve_verdict_source(args.store)[0]))} keys")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    table = load_verdicts(resolve_verdict_source(args.store)[0])
    for key in sorted(table):
        row = table[key]
        print(f"{key}\t{row.get('verdict')}\t{row.get('decided_at')}\t{row.get('reason')}")
    print(f"keys: {len(table)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def with_store(subparser):
        """`--store` per subcommand rather than on the root parser: the runbooks write
        `skill_verdicts.py check --candidates DIR --store PATH`, and a root-level option
        is only accepted *before* the subcommand, which is not how anybody types it."""
        subparser.add_argument(
            "--store", default=None, help=f"verdict ledger path (default {DEFAULT_STORE})")
        return subparser

    p_check = with_store(sub.add_parser("check", help="which candidate patterns a verdict blocks"))
    p_check.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    p_check.set_defaults(func=cmd_check)

    p_record = with_store(sub.add_parser("record", help="append one decision, including 'no action'"))
    p_record.add_argument("--pattern", required=True, help="pattern key, e.g. Edit/not_found")
    p_record.add_argument("--verdict", required=True)
    p_record.add_argument("--reason", required=True)
    p_record.add_argument("--evidence-cmd", dest="evidence_cmd", required=True)
    p_record.add_argument("--occurrences", type=int, default=0)
    p_record.add_argument("--decided-by", dest="decided_by", default="agent")
    p_record.add_argument("--source-candidate", dest="source_candidate", default="")
    p_record.set_defaults(func=cmd_record)

    p_seed = with_store(sub.add_parser("seed", help="harvest existing candidate dispositions once"))
    p_seed.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    p_seed.set_defaults(func=cmd_seed)

    p_list = with_store(sub.add_parser("list", help="latest-wins view of the ledger"))
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
