#!/usr/bin/env python3
"""Does any job that owes the bench-card copy duty have a bench drop it did not copy?

One command answers #4554's owed RE-ARM row, which quotes #4185's ruling: RE-ARM the
door's hard refusal only on a measured fire — `>=1 BENCH ASSERTION DROP on a job's
stderr whose run record does NOT carry the copied line` (`scripts/automod/owed.py:78`).
Answering that by hand means grepping 1,147 run records, and the grep is a trap: every
one of the 42 `BENCH ASSERTION` mentions in that corpus is a sentence saying the marker
did NOT appear (``no `BENCH ASSERTION DROP` — re-derived``, ``No `BENCH ASSERTION DROP`
printed``, ``**No `BENCH ASSERTION:` line printed**``, ``**0 `BENCH ASSERTION:`
lines**``), so a plain marker grep reports a fire on the first real drop and then
forever. This counts affirmative occurrences only and pairs them per job.

Two patterns, never collapsed. `scripts/util/bench_assertion_findings.py:229` records
that the door's prefix and the rung's are separate on purpose ("a caller that greps the
door's stderr for the rung's decision must not find a bookkeeping line wearing the
decision's prefix"), so these stay two independent regexes and
`test_the_two_patterns_never_match_the_others_form` pins that neither matches the
other's landed text:

  FIRE — the door's per-path drop line, emitted by `printf '%s: BENCH ASSERTION DROP:
         %s is out of this commit — %s\\n'` at `scripts/util/vault-commit.sh:628-629` in
         the `BENCH_RC -eq 1` branch (`:616-632`), which `git reset`s the path and does
         NOT exit — drop-and-report. `FIRE_RE` is pinned to the same prefix
         `tests/test_vault_commit_bench_assertion_refusal.py:271` pins on the emitter
         side, so a reworded emitter turns this witness red rather than blind.
  COPY — the rung's copy line, `BENCH ASSERTION: N bench card(s) added under bench/ …
         Copy this exact line into the invoking job's run record`
         (`bench_assertion_findings.py:477-484`, to stderr). Anchored at line start and
         on the card count, which is why the two shapes the corpus actually holds —
         `` `BENCH ASSERTION:` `` inside a sentence and ``**0 `BENCH ASSERTION:`
         lines**`` — cannot match it.

A pattern match is still only a hit when the line CARRIES the marker (`marker_hit`): the
text before the match must hold no negation cue, and for the FIRE side that lead must be
decoration and cue words only — one word like `echo` or a `file:line` makes the line a
quotation, and 45 sessions under ~/lloyd-data/sessions/ contain the door's line precisely
that way, as a re-published source quote. The two sides skew OPPOSITE on purpose: a
`DROPPED_UNCOPIED` re-arms a hard refusal, so the fire side is strict about what counts as
a fire and the copy side is permissive about what counts as a copy, which is the pairing
that makes "uncopied" the hardest of the three verdicts to reach.

The cue test runs on the LEAD ONLY, never on the tail — the emitter's own sentence ends
with "A line that must not appear in this commit's paths has also been removed", so a
trailing-`not` scan would negate every real fire the door ever wrote (checked against
`scripts/util/vault-commit.sh:628-629`). A whole fenced code block is skipped too (a run
record printing the format as an example is documentation, not an observation), and those
mentions are counted and printed as `fenced_mentions` rather than dropped quietly.

Scope is the duty set, not every job that can reach the door: 10 skill files across 8
skills invoke `vault-commit.sh`, and only 2 of them (`skills/autonomy-data-pipeline/
step1-fact-extraction.md:325`, `skills/nightly-reflection-signals/SKILL.md:383`) carry
the copy duty. A drop on a duty-less job satisfies "does NOT carry the copied line" with
no guard defect, so it is printed as `NO_DUTY` and never armed. #4185's ruling re-arms a
hard refusal; only an uncopied drop by a job that was told to copy is evidence for it.

Honest about its stores, because finding 4 is real: a drop's only durable trace is the
copy this condition asks to be ABSENT. The door's stderr reaches nothing that outlives
the call — `~/lloyd-data/logs/server.err` carries zero `vault-commit.sh:` lines ever,
`workers.db`'s `runs` table has no autonomy rows at all (`run_24_%` → 0 of 11,202), and
`vault-commit.sh:619` `git reset`s the dropped path so `git status --porcelain --
lloyd/bench/` is clean by the time anyone looks. So the per-job verdict comes from the
one store that can carry it — the autonomy run records — and every other store prints
its own count, attributed to itself, never becoming a job's zero. Until the vault's copy
duty also names the door's line (the HUMAN_CLAUSES ruling owed-check owes at #4554's
row, not this script's to invent), `DROPPED_UNCOPIED` is reachable only from a run record
quoting the door's line at line start: the watch is correct and near-silent by
construction, which is what the printed per-store counts and their provenance are for.

Where clause 5 overrides clause 1: if the per-job store itself is unreadable, a duty job
gets `STORE_UNREADABLE` and no zero, and `answerable=no` on the headline, rather than
`NO_DROPS` — "I could not look" must never print as "nothing happened".

Usage:
    python3 scripts/util/bench_drop_watch.py
    python3 scripts/util/bench_drop_watch.py --runs-dir … --server-err …
        --workers-db … --vault … --skills-dir … --autonomy-dir … --no-git

Exit status: 1 when a duty-carrying job is `DROPPED_UNCOPIED`, else 0. An unreadable
store never moves it and never contributes a zero — it prints UNREADABLE, because a
store nobody could open is not a store that reported nothing.
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

#: The door's drop line, tied to the prefix `tests/test_vault_commit_bench_assertion_refusal.py:271`
#: pins on the emitter side. Column-0 anchored because the emitter writes it there.
FIRE_RE = re.compile(r"^vault-commit\.sh:[ \t]+BENCH[ \t]+ASSERTION[ \t]+DROP\b")

#: The rung's copy line, anchored at line start (minus markdown decoration) and on the
#: card count. See the module docstring's COPY paragraph.
#: The decoration class is wider than the door's: the duty says copy the LINE verbatim,
#: and a job may file it as a bullet, a quote, a numbered finding, or in bold, so list
#: markers (including `3.`), emphasis and backticks may stand ahead of it. What may not
#: stand ahead of it is a sentence — see `marker_hit`'s lead rules.
COPY_RE = re.compile(
    r"^(?:[ \t>*_~`\[\]{}|]|\d+[.)]|- )*BENCH ASSERTION:[ \t]+\d+[ \t]+bench[ \t]+card"
)

#: What a run record can say about a marker without the marker being absent. Measured on
#: the 42-line corpus: 22 `` no `BENCH ASSERTION``, 12 ``No `BENCH ASSERTION``, 1
#: `no BENCH ASSERTION` bare, 4 backticked `` `BENCH ASSERTION:` `` mentions — all
#: negations. `0` is a cue because ``**0 `BENCH ASSERTION:` lines**`` negates with a
#: digit, and a cue-word-only guard would call that line a fire.
_CUE_RE = re.compile(
    r"\b(?:no|not|never|none|zero|without|neither|absent|missing|lacks?)\b|\b0\b",
    re.IGNORECASE,
)

#: The lead a carried marker line may have: whitespace, markdown decoration, and cue
#: words — and nothing else. Any other word (`echo`, `prints`, `grep`, a path) means the
#: line is quoting or discussing the marker, so it is not a hit. This is what excludes
#: the affirmative-looking `vault-commit.sh: BENCH ASSERTION DROP` hits under
#: ~/lloyd-data/sessions/, which are every one a source quote such as
#: `echo "vault-commit.sh: BENCH ASSERTION DROP: …`.
_ALLOWED_LEAD_RE = re.compile(
    r"^(?:[ \t>*_~`'\"\[\](){}.,:;|+\-]"
    r"|\b(?:no|not|never|none|zero|without|neither|absent|missing|0)\b[ \t:]*)*$"
)

#: A markdown fence. Content between two of these is an example, not an observation.
_FENCE_RE = re.compile(r"^[ \t]*(?:```|~~~)")

DROPPED_COPIED = "DROPPED_COPIED"
DROPPED_UNCOPIED = "DROPPED_UNCOPIED"
NO_DROPS = "NO_DROPS"
NO_DUTY = "NO_DUTY"
#: A store could not be read. Never a zero, and never silent.
STORE_UNREADABLE = "STORE_UNREADABLE"
#: A store this run was TOLD not to open (`--no-git`). Kept distinct from UNREADABLE on
#: purpose: a store nobody attempted is a choice the caller made, and printing it with the
#: same word would let the flag bury a store that genuinely failed.
STORE_SKIPPED = "STORE_SKIPPED"

#: Both markers, for the cheap "did this blob mention it at all" prefilter only. The
#: verdict always comes from FIRE_RE / COPY_RE.
MENTION_RE = re.compile(r"BENCH[ \t]+ASSERTION", re.IGNORECASE)

#: The duty: a line that mentions the marker and a copy verb together. Both landed
#: sentences match — "A `BENCH ASSERTION:` line on that stderr is copied verbatim into
#: this run's record" and "A line starting `BENCH ASSERTION:` is copied **verbatim** into
#: the same section" — and nothing else in the vault's skills does.
_DUTY_VERB_RE = re.compile(r"\b(?:copied|copy|copies)\b", re.IGNORECASE)


@dataclass
class Counts:
    """Affirmative hits only. `fenced` counts candidates inside a code block: skipped,
    but never silently."""

    fires: int = 0
    copies: int = 0
    fenced: int = 0

    def add(self, other: "Counts") -> None:
        self.fires += other.fires
        self.copies += other.copies
        self.fenced += other.fenced


@dataclass
class Store:
    """One store's report: what was read, what it held, and what it can prove."""

    name: str
    readable: bool
    detail: str
    counts: Counts = field(default_factory=Counts)
    #: Whether a hit here can be pinned to one job. Run records can; server.err,
    #: workers.db and `git status` cannot, so their counts print as the store's own.
    attributable: bool = False
    unreadable_reason: str = ""
    #: Told not to open this one (`--no-git`). Prints as SKIPPED, not UNREADABLE: a store
    #: nobody attempted is a caller's choice, and reusing the failure word would let the
    #: flag bury a store that genuinely could not be read.
    skipped: bool = False


def marker_hit(line: str, pattern: re.Pattern[str], strict_lead: bool = True):
    """The match if `line` CARRIES this marker, else None.

    Always: the pattern must match and the text before it (`the lead`) must hold no
    negation cue — that is what makes the 42 corpus lines count zero.

    With `strict_lead`, the lead must additionally be decoration and cue words ONLY: one
    real word (`echo`, `prints`, `grep`, `skills/foo/SKILL.md:67`) means the line is
    quoting or discussing the marker, which is what excludes every affirmative-looking
    `vault-commit.sh: BENCH ASSERTION DROP` hit under ~/lloyd-data/sessions/ — all of them
    source quotes like `echo "vault-commit.sh: BENCH ASSERTION DROP: …`.

    The two markers deliberately use OPPOSITE lead policies, because the two errors cost
    different amounts. A `DROPPED_UNCOPIED` re-arms a hard refusal, so arming must be the
    hard verdict to reach: the FIRE side stays strict (a fire needs a carried line, not a
    discussion of one) and the COPY side is permissive (a copy counts wherever it sits in
    the record, even mid-sentence, since a job satisfying its duty may write the line into
    a paragraph). Both skews point the same way — more evidence required before the watch
    says "uncopied".
    """
    match = pattern.search(line)
    if match is None:
        return None
    lead = line[: match.start()]
    if strict_lead and not _ALLOWED_LEAD_RE.match(lead):
        return None
    if _CUE_RE.search(lead):
        return None
    return match


def classify_line(line: str) -> tuple[bool, bool]:
    """(is_fire, is_copy), each decided independently — nothing assumes they exclude one
    another; the caller sums the two counts separately."""
    fire = marker_hit(line, FIRE_RE, strict_lead=True)
    copy = marker_hit(line, COPY_RE, strict_lead=False)
    return bool(fire), bool(copy)


def scan_lines(lines) -> Counts:
    """Affirmative fire/copy counts over lines; a fenced block is documentation, not an
    observation, so nothing inside it is counted — and what was skipped is COUNTED as
    skipped (`fenced`) rather than dropped quietly, because a format example that one day
    reads like a fire must still be visible in the output."""
    out = Counts()
    fenced = False
    for line in lines:
        if _FENCE_RE.match(line):
            fenced = not fenced
            continue
        if fenced:
            if MENTION_RE.search(line):
                out.fenced += 1
            continue
        fire, copy = classify_line(line)
        out.fires += int(fire)
        out.copies += int(copy)
    return out


def scan_text(text: str) -> Counts:
    return scan_lines(text.splitlines())


def data_root() -> Path:
    """`$LLOYD_DATA` when set, else `~/lloyd-data` — the root `app.paths` resolves.

    Derived rather than imported: `app.paths` drags the whole app in and prints its
    worktree-anchored warning, and a watch that runs at a scheduled recheck should answer
    a question about files without booting the app.
    """
    env = os.environ.get("LLOYD_DATA")
    return Path(env).expanduser() if env else Path.home() / "lloyd-data"


def duty_files(skills_dir: Path) -> list[str]:
    """Skill-relative files whose text carries the copy duty, derived by scan.

    Derived, not hard-coded: the duty belongs to the vault, so if the ruling on #4555
    extends these sentences to name the door's line, or a third skill takes the duty, the
    watch follows without a code change.
    """
    out: list[str] = []
    if not skills_dir.is_dir():
        return out
    for path in sorted(skills_dir.glob("*/**/*.md")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if MENTION_RE.search(line) and _DUTY_VERB_RE.search(line):
                out.append(str(path.relative_to(skills_dir)))
                break
    return out


def duty_skills(skills_dir: Path) -> list[str]:
    """Distinct skill dirs carrying the duty — the set jobs are matched against."""
    return sorted({p.split("/")[0] for p in duty_files(skills_dir)})


def live_jobs(autonomy_dir: Path) -> dict[int, dict[str, str]]:
    """task id -> {name, skill} for live autonomy task files.

    `_archived/` is not read: an archived task cannot run, so it cannot owe a copy.
    """
    out: dict[int, dict[str, str]] = {}
    if not autonomy_dir.is_dir():
        return out
    for path in sorted(autonomy_dir.glob("*.md")):
        if path.name.startswith("_"):
            continue
        try:
            parts = path.read_text(encoding="utf-8", errors="replace").split("---")
            meta = yaml.safe_load(parts[1]) if len(parts) > 1 else None
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(meta, dict) or meta.get("id") is None:
            continue
        try:
            tid = int(meta["id"])
        except (TypeError, ValueError):
            continue
        out[tid] = {
            "name": str(meta.get("name") or path.stem),
            "skill": str(meta.get("skill_name") or ""),
        }
    return out


def read_run_records(runs_dir: Path) -> tuple[Store, dict[int, Counts]]:
    """The one store that can attribute a drop to a job: `autonomy-runs/*/run_*.md`."""
    store = Store("run_records", True, f"root={runs_dir}", attributable=True)
    per_job: dict[int, Counts] = {}
    if not runs_dir.is_dir():
        store.readable = False
        store.unreadable_reason = "no such directory"
        store.detail = f"root={runs_dir}"
        return store, per_job
    files = sorted(runs_dir.glob("*/run_*.md"))
    read = 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        read += 1
        tid = int(path.parent.name) if path.parent.name.isdigit() else -1
        hit = scan_text(text)
        per_job.setdefault(tid, Counts()).add(hit)
        store.counts.add(hit)
    store.detail = f"root={runs_dir} records={read}/{len(files)}"
    return store, per_job


def read_server_err(path: Path) -> Store:
    """`logs/server.err`. Print both the raw mention count and the affirmative one: the
    item's own premise ("server.err carries 0 door lines at all") is the RAW count, and
    0 affirmative alone would not distinguish an empty store from a store whose every
    line is a negation.
    """
    store = Store("server_err", True, f"path={path}")
    if not path.is_file():
        store.readable = False
        store.unreadable_reason = "no such file"
        store.detail = f"path={path}"
        return store
    mentions = 0
    counts = Counts()
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if MENTION_RE.search(line):
                    mentions += 1
                fire, copy = classify_line(line)
                counts.fires += int(fire)
                counts.copies += int(copy)
    except OSError as exc:
        store.readable = False
        store.unreadable_reason = f"{type(exc).__name__}: {exc}"
        store.detail = f"path={path}"
        return store
    store.counts = counts
    store.detail = f"path={path} mentions={mentions}"
    return store


def read_workers_db(path: Path) -> Store:
    """Counts over `runs.summary` only, read-only.

    `summary` and nothing else: the blob columns are what put `~/lloyd-data/workers.db`
    over the 1,048,576 B witness ceiling, and a watch that orders 11k rows across five
    columns to count a literal is the wrong instrument. `runs` also holds no autonomy rows
    at all, so this store's counts are the store's own and can never be a job's zero.
    """
    store = Store("workers_db", True, f"path={path}")
    if not path.is_file():
        store.readable = False
        store.unreadable_reason = "no such file"
        store.detail = f"path={path}"
        return store
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        store.readable = False
        store.unreadable_reason = f"{type(exc).__name__}: {exc}"
        store.detail = f"path={path}"
        return store
    try:
        rows = con.execute(
            "select summary from runs where summary like '%BENCH ASSERTION%'"
        ).fetchall()
        total = con.execute("select count(*) from runs").fetchone()[0]
        counts = Counts()
        for (summary,) in rows:
            counts.add(scan_text(summary or ""))
        store.counts = counts
        store.detail = f"path={path} runs={total} summary_candidates={len(rows)}"
    except sqlite3.Error as exc:
        store.readable = False
        store.unreadable_reason = f"{type(exc).__name__}: {exc}"
        store.detail = f"path={path}"
    finally:
        con.close()
    return store


def read_vault_dirty(vault: Path) -> Store:
    """`git status --porcelain -- lloyd/bench/`: a drop leaves its path dirty.

    Independent witness, never a verdict: `vault-commit.sh:619` `git reset`s the path, so
    it stays dirty only until something else lands it, and the command cannot say which
    job dropped anything.
    """
    store = Store("vault_dirty", True, f"vault={vault}")
    if not (vault / ".git").exists():
        store.readable = False
        store.unreadable_reason = "no .git"
        return store
    try:
        proc = subprocess.run(
            ["git", "-C", str(vault), "status", "--porcelain", "--", "lloyd/bench/"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        store.readable = False
        store.unreadable_reason = f"{type(exc).__name__}"
        return store
    if proc.returncode != 0:
        store.readable = False
        store.unreadable_reason = (
            f"git exited {proc.returncode}: {proc.stderr.strip()[:120]}"
        )
        return store
    paths = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    store.detail = f"vault={vault} bench_dirty_paths={len(paths)}"
    return store


def status_for(counts: Counts, carries_duty: bool, store_readable: bool) -> tuple[str, bool]:
    """(token, armed). Only a duty-carrying job's uncopied drop arms the exit status.

    `NO_DROPS` requires the per-job store to have been readable — clause 5's precedence
    over clause 1, spelled out in the module docstring.
    """
    if not store_readable:
        return STORE_UNREADABLE, False
    if not carries_duty:
        return NO_DUTY, False
    if counts.fires == 0:
        return NO_DROPS, False
    if counts.copies >= counts.fires:
        return DROPPED_COPIED, False
    return DROPPED_UNCOPIED, True


def build(runs_dir: Path, server_err: Path, workers_db: Path, vault: Path,
          skills_dir: Path, autonomy_dir: Path,
          use_git: bool = True) -> dict[str, object]:
    """Read every store, pair drops with copies per job, and prepare the report."""
    skills = duty_skills(skills_dir)
    task_jobs = live_jobs(autonomy_dir)
    rr_store, per_job = read_run_records(runs_dir)
    stores = [rr_store, read_server_err(server_err), read_workers_db(workers_db)]
    if use_git:
        stores.append(read_vault_dirty(vault))
    else:
        stores.append(Store("vault_dirty", True, f"vault={vault}", skipped=True))

    duty_lines: list[str] = []
    duty_less: list[str] = []
    armed = 0
    # Print a job when it carries the duty or its records hold a marker: 54 task dirs of
    # NO_DROPS is noise, and a job with no records cannot have dropped anything.
    tids = sorted(
        set(per_job) | {t for t, m in task_jobs.items() if m["skill"] in skills}
    )
    for tid in tids:
        meta = task_jobs.get(tid, {"name": f"task-{tid}-no-file", "skill": "?"})
        carries_duty = meta["skill"] in skills
        counts = per_job.get(tid, Counts())
        token, is_armed = status_for(counts, carries_duty, rr_store.readable)
        armed += int(is_armed)
        line = (
            f"{token:16s} task={tid} job={meta['name']} skill={meta['skill'] or '?'} "
            f"drops={counts.fires} copies={counts.copies} store=run_records"
        )
        if carries_duty:
            duty_lines.append(line)
        elif counts.fires or counts.copies:
            duty_less.append(line + " note=marker_on_a_job_with_no_copy_duty_not_armed")
    return {
        "skills": skills,
        "duty_files": duty_files(skills_dir),
        "duty_lines": duty_lines,
        "duty_less": duty_less,
        "stores": stores,
        "armed": armed,
        "answerable": rr_store.readable,
    }


def render(report: dict[str, object]) -> list[str]:
    """The output lines. One store block per store, each naming itself."""
    out = [
        f"bench_drop_watch duty_skills={len(report['skills'])} "
        f"duty_files={', '.join(report['duty_files']) or 'NONE_watch_is_blind'} "
        f"answerable={'yes' if report['answerable'] else 'no'}"
    ]
    if not report["duty_lines"]:
        out.append(
            f"{NO_DUTY:16s} note=no live autonomy task names a duty-carrying skill; "
            f"the watch has nothing to grade"
        )
    out.extend(report["duty_lines"])
    out.extend(report["duty_less"])
    for st in report["stores"]:
        if not st.readable:
            out.append(
                f"store {st.name}: UNREADABLE ({st.unreadable_reason}) "
                f"{st.detail} — contributes no zero to any job"
            )
        elif st.skipped:
            out.append(
                f"store {st.name}: {STORE_SKIPPED} (--no-git was passed) "
                f"{st.detail} — not opened, so it contributes nothing either way"
            )
        else:
            out.append(
                f"store {st.name}: {st.detail} fires={st.counts.fires} "
                f"copies={st.counts.copies} fenced_mentions={st.counts.fenced} "
                f"per_job={'yes' if st.attributable else 'no'}"
            )
    out.append(f"exit={'1' if report['armed'] else '0'} armed_dropped_uncopied={report['armed']}")
    return out


def _vault_default() -> Path:
    """The vault root, from `app.data_root.vault_root()` — not recomputed here.

    The resolver is the only module allowed to turn a home directory into a vault path
    (`tests/test_no_runtime_paths_in_code.py` refuses a second spelling), and its reason is
    this script's exact hazard: it takes `--skills-dir` and `--autonomy-dir` from the vault,
    so a default computed from the wrong root points the duty scan at the live vault while
    the caller thinks it sandboxed itself. `vault_root()` honours `$LLOYD_VAULT_ROOT` then
    `$LLOYD_VAULT` then `~/.config/lloyd/vault_root` then `~/obsidian`, and is imported
    lazily because a watch that answers a question about files should not need the app.

    The fallback repeats both env names for the same reason the rail checks for them: an
    importer that fails must not leave a caller reading a silent `~/obsidian`.
    """
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from app.data_root import vault_root

        return vault_root()
    except Exception:  # noqa: BLE001 - no app importable: fall back, loudly named below
        env = os.environ.get("LLOYD_VAULT_ROOT") or os.environ.get("LLOYD_VAULT")
        return Path(env).expanduser() if env else Path.home() / "obsidian"


def main(argv: list[str] | None = None) -> int:
    root = data_root()
    vault_default = _vault_default()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", type=Path, default=root / "autonomy-runs")
    ap.add_argument("--server-err", type=Path, default=root / "logs" / "server.err")
    ap.add_argument("--workers-db", type=Path, default=root / "workers.db")
    ap.add_argument("--vault", type=Path, default=vault_default)
    ap.add_argument("--skills-dir", type=Path, default=vault_default / "skills")
    ap.add_argument("--autonomy-dir", type=Path, default=vault_default / "autonomy")
    ap.add_argument("--no-git", action="store_true",
                    help="skip the `git status` witness (for a vault-less fixture)")
    args = ap.parse_args(argv)

    report = build(
        runs_dir=args.runs_dir,
        server_err=args.server_err,
        workers_db=args.workers_db,
        vault=args.vault,
        skills_dir=args.skills_dir,
        autonomy_dir=args.autonomy_dir,
        use_git=not args.no_git,
    )
    for line in render(report):
        print(line)
    return 1 if report["armed"] else 0


if __name__ == "__main__":
    sys.exit(main())
