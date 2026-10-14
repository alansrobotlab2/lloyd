#!/usr/bin/env python3
"""Learned per-task execution baselines and a read-only structural-conformance check (#673).

The method is the one in the JP Morgan "learned execution graphs" talk: model each run as
the ordered sequence of steps it traversed, learn the shape from history, and compare every
run against what the *other* runs of that task did. A step that every other run took and
this one did not is a **structural** deviation, visible even though latency was fine and the
run reported `success`. That is the failure family this exists for: the worker ledger holds
one terminal status per run (`runs.status` is `success|failed|skipped`, and no `meta_json`
key is per-stage), so a run that silently skips a mandatory step lands in `completed` and
nothing out-of-band notices. See the `dream-consolidation` lock-gate pass (09-03 to 09-09,
gate permanently open), the 09-08 dependency inversion, and the 09-01/09-02 missing-handoff
family — all found by manual forensics after the fact.

Two rules this deliberately does NOT do:

* **It does not alert.** It emits reports; wiring them to a surface is a human decision
  (#673's own "do not alert yet", and the guardian's record that every rollback so far was
  a false positive). There is no mail, HTTP or notification import in this file on purpose,
  and `tests/test_step_conformance.py` pins that. The one escalation #2104 clause 4 added is
  opt-in (`replay --escalate-writes`) and is not an alert either: it files a backlog **draft**
  — a queue entry a human adjudicates — for a flagged run that never wrote an artifact every
  other run of its task wrote, and it is the only third store this module may ever write. The
  flag is off by default, so a replay nobody asked to escalate still writes nothing anywhere
  (`tests/test_step_conformance_escalation.py::test_replay_without_the_flag_files_nothing`).
* **It does not re-implement the dependency gate.** That is #558's decision, live at
  `app/autonomy.py::_is_dependency_met`. Where the 09-08 inversion shows up here it shows up as
  a *missing consumption step in a trace* — the structural signature — never as a verdict
  about whether a task was allowed to run.

Signal sources, both already on disk and joined by `runs.meta_json.session_id` (== the
trajectory row's `session_key`; 411 of 438 `scheduled-task` runs joined over 2026-09-14 ->
09-21):

* `_pipeline/trajectories/YYYY-MM-DD.jsonl` — ordered per-step trace per session
  (`session_source: autonomy-task:N`, `tools[]` with `name`, `params_summary`, `sequence`).
  Written nightly by `scripts/extract-trajectories.py` (autonomy task #56).
* `workers.db` `runs` — the recorded status, timings and session ids.

Both are opened read-only: the sqlite handle uses `mode=ro` and the JSONL files are only
ever read, so a replay cannot perturb the corpus it is judging
(`tests/test_step_conformance.py::test_replay_leaves_both_stores_byte_identical_and_exits_zero`
hashes a real corpus across a full subprocess replay to prove that).

Step identity. A step is not just a tool name: `write:<basename>@<dir>` / `read:<basename>`
are first-class steps, evidenced either by a file-tool's `file_path`/`path` or by a Bash
command that redirects into a path or runs a writer (`touch|tee|cp|mv|install`). Tool
substitution (`Write` today, `Bash > file` tomorrow) therefore does not move the baseline,
which is the main false-alarm lever here. Basenames have date stamps elided (`knowledge-handoff-
2026-09-20.md` -> `knowledge-handoff-<date>.md`) so a nightly artifact is one step, not 365.

An *artifact* step is additionally checked against where it was written (#2502 clause 2): the
id stays `write:<basename>` — widening it would rename every artifact step in the published
corpus and make the labelled `validate` numbers incomparable with the ones measured before —
and the captured path is kept beside it, so a write of the right filename at a directory the
task never writes that artifact to does not discharge the expectation. That is the shape
nothing caught on 2026-10-09, when task 39 left `_pipeline/reflection/conversation-patterns-latest.md`
untouched, wrote the same name into the vault instead, and turned up in a later baseline's
`learned_from` as a run that took the step. `read:` steps stay name-only: a stage may
legitimately read its input from somewhere else, and there is no escalation riding on a
missing read to justify the over-fire risk.

Tuning, and what it cost to pick these numbers. All figures are `replay --days 14` against
the live corpus on 2026-09-21: 2,083 scored runs across 33 learned task baselines, of which
1,570 recorded `success`.

* `support=1.0` — expect a step only where every *other* run of that task has it: 13 flagged
  runs (0.62 %). Relaxing to 0.9 flags 88 (4.22 %) and 0.8 flags 124 (5.95 %), so strict stays
  the default. Stated plainly, the cost of strict: two runs that skip the *same* step cancel
  each other's expectation, so a persistent skip erases its own alarm, and `--support` is the
  knob that buys recall back at the rate above.
* `min_runs=5` — at 3, a task whose few runs all differ flags all of its own runs (task 85 did
  exactly that with four runs): 17/2,087 (0.81 %) at 3 vs 13/2,083 (0.62 %) at 5, and one more
  task baseline appears. Below 3 the published set is the other run's trace verbatim, which is
  not a baseline.
* `grace_seconds` — his late-span problem. A run that finished inside the window is reported
  `pending`, not a deviation, and becomes one only on a re-check after the window lapses, so
  the cost of the window is exactly one nightly cycle of delay. Honest measurement: on the
  14-day live replay it held 0 of the 13 flagged runs (all had finished >6 h before the pass)
  and the mechanism engaged only when the support was loosened (2 pending runs at 0.8). The
  window is therefore pinned by tests, not yet by a field number; it is also unvalidated
  against a trace store that lags the ledger, which is what it is for.
* Position is never a deviation, only an observation. A step that shows up far later than
  its learned median is listed in `late_steps` and scores clean, because the 1-of-7-node
  late-span case is the named false-positive generator and reordering a nightly chain is
  legal while dropping a stage is not.

Usage (read-only; nothing here restarts or writes to the corpus):

    python3 scripts/step_conformance.py replay --days 14
    python3 scripts/step_conformance.py learn --out /tmp/step-baseline.json
    python3 scripts/step_conformance.py validate            # offline injected-fault fixtures
    python3 scripts/step_conformance.py validate --fixtures DIR --json /tmp/validate.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.paths import PIPELINE_DIR, WORKERS_DB  # noqa: E402

DEFAULT_TRAJECTORIES = str(PIPELINE_DIR / "trajectories")
DEFAULT_DB = str(WORKERS_DB)
# Empty means "the whole ledger": the structural check is driven by the *trace* store, and
# baselines are keyed by the trace's `session_source`, so no run is ever judged against
# another pipeline's shape (his per-client rule) whether or not the ledger is filtered here.
# What `--source` actually changes is coverage of the recorded status: measured 09-21 over 14
# days, restricting it to `scheduled-task` left 1,660 of 2,096 traces with no ledger row and
# so with `run_status=None`, vs 387 unfiltered — the flag count itself was identical (13).
DEFAULT_SOURCE = ""
DEFAULT_GRACE_SECONDS = 6 * 3600
DEFAULT_MIN_RUNS = 5
DEFAULT_SUPPORT = 1.0

DATE_RE = re.compile(r"20\d{2}-?\d{2}-?\d{2}")
REDIRECT_RE = re.compile(r">+\s*([^\s;&|`)\"']+)")
WRITER_RE = re.compile(r"(?:^|[;&|]\s*)(?:touch|tee|cp|mv|install)\b")
WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit", "vault_write", "write_file"}
READ_TOOLS = {"Read", "vault_read", "Grep", "Glob", "note_read"}
# Basenames that name nothing: a trace that "wrote" one of these proved nothing, and the
# first pass over live data produced a `write:null` expected step out of exactly one.
GENERIC_BASENAMES = {"", "null", "none", "dev", "stdout", "stderr", "input", "output"}
# The constructs that make a capture a fragment rather than a path — see
# `_unclosed_shell_word`, which is the only reader and states the measured case. A *closed*
# expansion (`${OUT}`, `$VAR`) is not in here on purpose: an unexpanded variable directory still
# leaves a basename that names the artifact, and rejecting those would shrink the learned set.
SHELL_TOKEN_RE = re.compile(r"\$\(|`|\$\{[^}]*\Z")

# How much of an artifact's directory the step identity carries (#2502 clause 2). Two, not
# the whole path and not one. Whole paths are what `normalized_dir` refuses, for the data-home
# root move it names. One component is too few because a directory called `reflection` is not
# evidence of being *the* reflection directory: `/a/reflection/x.md` and
# `/b/obsidian/reflection/x.md` would be the same step, and a run that moved a report into
# some other `reflection` folder — the shape this clause exists to catch — would stay quiet.
# Two names an artifact's own folder and the folder that holds it, which is as deep as any
# real report path here differs.
DIR_COMPONENTS = 2

# Why a deviation is what it is, and which of them may file (#2502 clause 3).
# `missing_structural_step` is the shape the detector has always escalated: no write of that
# artifact anywhere in the trace, and nothing on disk at the learned path from this run.
# `artifact_written_at_other_path` files too — the content moved, which is a finding about the
# task, not a parsing gap — and says so in its own words.
# `disk_witnessed_write_not_in_trace` never files: the artifact at the learned path *was*
# modified inside the run's own window, so the claim "this run dropped the stage" is false
# whatever the trace's grammar captured. It stays reported, because the trace not showing a
# write the task's other runs show is a parsing gap worth a look.
REASON_MISSING = "missing_structural_step"
REASON_OTHER_PATH = "artifact_written_at_other_path"
REASON_WITNESSED = "disk_witnessed_write_not_in_trace"
ESCALABLE_REASONS = (REASON_MISSING, REASON_OTHER_PATH)

#: The margin on the disk witness's window. `started_at` is when the queue claimed the run and
#: `completed_at` is the queue's own write, so either can sit minutes from the tool call that
#: touched the file, and a trace may span a resume. Six hours, the same figure the pending
#: window uses above — a nightly cycle plus the model's slack, and deliberately not a day: the
#: real cases left the pointer one to two days stale, which this catches while a same-day
#: re-run of the stage is forgiven.
WITNESS_GRACE_SECONDS = DEFAULT_GRACE_SECONDS


def _expand(path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()


def _unclosed_shell_word(text: str) -> bool:
    """True when a captured string is not a complete shell word.

    `REDIRECT_RE` ends a capture at whitespace, `;`, `&`, `|`, a backtick or a quote — it cannot
    express quoting, so a redirect into a quoted path whose name contains a substitution is
    captured up to the first space *inside* that substitution. What comes back is therefore a
    fragment that still carries an open construct: `$(…` with no `)`, `${…` with no `}`, a lone
    backtick, or a `(` left unbalanced. Measured on the 14-day live replay of 2026-10-03, the
    family is exactly one name — `write:preflight-$(date`, from `echo hi >
    "/tmp/preflight-$(date +%F).md"` — 4 of the 22 deviations over 3,783 scored runs, sitting in
    task 38's learned `expected_steps` so every run of that task was flagged against an artifact
    nothing ever wrote."""
    return bool(SHELL_TOKEN_RE.search(text)) or text.count("(") != text.count(")")


def normalized_basename(raw) -> str:
    """Basename of a path-ish value, with a date stamp elided so a nightly file is one step.

    Empty for a name that is not a name: a value in `GENERIC_BASENAMES`, and a capture that is
    not a complete shell word (#2104 clause 1, `_unclosed_shell_word`). The test is on the whole
    captured word rather than the basename, because a truncation in the directory part means the
    shell never wrote the path the basename appears to name; a *closed* expansion is not a
    truncation, so `${OUT}/knowledge-handoff-2026-09-20.md` still contributes
    `knowledge-handoff-<date>.md` — banning `$` outright would shrink the learned set in silence.
    """
    text = str(raw or "").strip().strip("\"'")
    if not text or _unclosed_shell_word(text):
        return ""
    base = os.path.basename(text.rstrip("/"))
    return DATE_RE.sub("<date>", base)


def normalized_dir(raw) -> str:
    """The `DIR_COMPONENTS` path components nearest a file, date stamps elided.

    The directory half of an artifact step's identity (#2502 clause 2), and deliberately not
    the whole path, because of the data-home move of 2026-09-22 (`architecture/data-home.md`):
    a baseline that spans that date holds runs that wrote the same reflection report under two
    different roots, and whole-path identity would read that as every run of the task dropping
    its stage. Two components is still enough to separate the case the clause is about: the
    2026-10-09 run wrote `knowledge/conversation/conversation-patterns-latest.md`, whose nearest
    components are `knowledge/conversation`, into a baseline whose other runs wrote it under
    `_pipeline/reflection/`.

    Empty for a directory that cannot be resolved, which is a deliberate third state and not a
    failure: a bare filename has no directory, and a path whose directory is still an
    unexpanded expansion (`${OUT}/report.md`, which the module's own comment on
    `normalized_basename` protects as a learned basename) names a directory this host cannot
    see. Both come back empty, and `dir_matches` reads empty as *no evidence*, not as
    evidence of a different place — an unknown directory must not refute a run's write, which
    is the same reasoning that kept `$` out of the basename reject list.
    """
    text = str(raw or "").strip().strip("\"'")
    if not text or _unclosed_shell_word(text):
        return ""
    parts = [p for p in text.rstrip("/").replace("\\", "/").split("/") if p]
    parts = parts[:-1]                      # drop the filename itself
    tail = parts[-DIR_COMPONENTS:]
    if any("$" in p for p in tail):         # unresolvable: no claim about where it is
        return ""
    return "/".join(DATE_RE.sub("<date>", p) for p in tail)


#: The keys a dispatched call may name its file with, in precedence order:
#: `Write`/`Edit`/`Read` use `file_path`, the vault tools `path`, the notebook
#: editor `notebook_path`. First truthy one wins, which is the order the tools
#: themselves are read in.
PATH_KEYS = ("file_path", "path", "notebook_path")


def _path_target(params) -> str:
    """The file a non-Bash call names, or "". `params` may be anything a trace holds."""
    params = params if isinstance(params, dict) else {}
    for key in PATH_KEYS:
        target = params.get(key)
        if target:
            return str(target)
    return ""


def write_targets(name, params) -> list:
    """Every path capture in one dispatched call that means "this call writes here".

    The one grammar for both readers of a run's writes: `steps_from_tools` below
    (the replay's step ids) and `app.run_acceptance.DispatchTrace` (which feeds
    the in-run acceptance check `artifact_written`, #2706). ONE function is the
    point: two copies of this grammar is how the in-run grader and the T+1 replay
    start answering different questions about the same run — which is exactly how
    task 94's 2026-10-13 run came to pass every declared check while writing
    nothing.

    A file tool contributes its target only when the tool is a writer (a `Read`
    of a path is not a write of it). A `Bash` call contributes each redirect
    target, plus, for `touch|tee|cp|mv|install`, the first path-looking token
    after the writer word. That last rule is carried as published, including that
    a `cp SRC DST` contributes SRC: it is the corpus as measured — task 94's
    learned `write:referential-integrity-latest.md` comes from its
    `cp …/referential-integrity-latest.md …/$(date -u +%F).md`, whose *source* is
    the artifact. Making it read the destination would rename published step ids
    and make the labelled `validate` numbers incomparable with the ones already
    published, so it is a decision for a person, not a side effect of this round.
    """
    out: list = []
    params = params if isinstance(params, dict) else {}
    target = _path_target(params)
    if target and name in WRITE_TOOLS:
        out.append(target)
    if name == "Bash":
        command = str(params.get("command") or "")
        for match in REDIRECT_RE.finditer(command):
            out.append(match.group(1))
        writer = WRITER_RE.search(command)
        if writer:
            for token in command[writer.start():].split()[1:]:
                if token.startswith(("-", ">")):
                    continue
                # Only an explicit path is evidence: a bare `cp foo` would name a file
                # that lives in the caller's cwd, which a basename cannot resolve.
                if "/" not in token:
                    continue
                # The scan keeps going past a token whose basename the sanitizer
                # blanks or files as generic, and stops at the first usable one. The
                # predicate has to be this exact one and not merely `"/" in token`,
                # because it selects WHICH token is the write: the corpus has a
                # nightly `… | tee /tmp/preflight-$(date +%s).txt; … ls -t
                # /tmp/preflight-*.txt` whose first slashed token normalizes to "" —
                # break there and the published `write:preflight-*.txt` step for the
                # 2026-09-25 reflection-signals run disappears. Re-derived over all
                # 8,770 trajectory runs on 2026-10-14: this line is what made the
                # extraction byte-identical to the one it replaced.
                base = normalized_basename(token)
                if base and base not in GENERIC_BASENAMES:
                    out.append(token)
                    break
    return out


def artifact_basenames(name, params) -> list:
    """The normalized artifact basenames one dispatched call writes.

    `write_targets`' captures through `normalized_basename` and minus
    `GENERIC_BASENAMES` — the same two filters that make a step id mean something
    (`normalized_basename`), so a run that "wrote" `null` proves nothing here
    either. This is what `artifact_written` is graded against (#2706 clause 4):
    comparing on these means a run's `foo-2026-10-14.md` and a declared
    `foo-<date>.md` are one name to the grader and to the replay alike.
    """
    out: list = []
    for raw in write_targets(name, params):
        base = normalized_basename(raw)
        if base and base not in GENERIC_BASENAMES and base not in out:
            out.append(base)
    return out


def corroborated(counts: dict) -> list:
    """The artifact paths a task's baseline treats as *where that artifact lives*.

    A path is learned if more than one of the task's runs wrote the artifact there; a path only
    one run wrote is that run's behaviour, not the task's, and is exactly what #2502 clause 2
    has to be able to refuse — task 39's history has five runs at `_pipeline/reflection/…` and
    the 2026-10-09 run at a vault path, and folding both in as equally "learned" would make the
    clause check nothing. If no path reaches two runs the location is uncorroborated, so every
    path is kept, which means no location is enforced: the conservative half of the trade,
    because a baseline of two disagreeing runs cannot say where an artifact belongs, and a
    detector that invents one fires on the run it happened to score first. A tie between two
    locations that both recurred keeps both, which is the real case of a task whose data root
    moved (2026-09-22) — both spellings are its history, and neither is a deviation.

    Best-supported first, so `[0]` is the artifact's learned path for the one question that
    asks for a single answer (`learned_artifact_path`, and with it the disk witness).
    """
    kept = [path for path, seen in counts.items() if seen >= 2]
    if not kept:
        return list(counts)
    return sorted(kept, key=lambda path: -counts[path])


def learned_artifact_path(artifact_paths, step: str) -> str:
    """The one path a learned artifact step names: the best-supported location the task's own
    runs wrote it to, `""` if history recorded none.

    Singular on purpose (#2502 clause 3): the disk witness answers "did this run produce the
    artifact it was supposed to produce", and that is a question about the file the task writes,
    not about every file that has ever carried the basename. `TaskBaseline.artifact_paths` keeps
    the full corroborated list because `same_name_elsewhere` legitimately needs every recorded
    location to say where a swapped write went; widening a witness across those paths would
    excuse a deviation on a file that is not this artifact.
    """
    paths = (artifact_paths or {}).get(step) or []
    return paths[0] if paths else ""


def artifact_basename(step: str) -> str:
    """The basename an artifact step names: `write:report-latest.md` -> `report-latest.md`."""
    return step[len(ESCALATE_PREFIX):] if step.startswith(ESCALATE_PREFIX) else ""


def dir_matches(learned: str, observed: str) -> bool:
    """Whether a write's directory contradicts the one the expectation was learned at.

    A mismatch needs two named directories, and that asymmetry is the whole design. Either
    side empty means no evidence, and no evidence never refutes a run's write: a bare
    filename, or a capture whose directory is still an unexpanded expansion, has not shown
    the artifact living anywhere else. This is the same reasoning that keeps `$` out of the
    basename reject list — a truncated capture must not manufacture a deviation — and it is
    why the comparison is suffix-wise: a learned `_pipeline/reflection` and a captured
    `lloyd-data/_pipeline/reflection` are the same directory seen from different depths, and
    the data-home move of 2026-09-22 means a real baseline holds both spellings.

    What it cannot do is catch a run whose write is at a *shorter* path than the learned one
    with a different tail. That is the cost of the suffix rule, and the disk witness beside it
    (#2502 clause 3) is what catches the case the grammar misses.
    """
    if not learned or not observed:
        return True
    return learned.endswith(observed) or observed.endswith(learned)


def same_name_elsewhere(step: str, write_paths, artifact_paths) -> list[str]:
    """Paths this run wrote under the artifact's basename, at a directory its task's history
    does not write that artifact to. Reported on the deviation so the draft says where the
    content went: for the 2026-10-09 run that is the vault path, and the difference between
    naming it and not naming it is the difference between a human adjudicating in one read and
    one going back to `git log`. An artifact with no learned paths has no expectation about
    where it lives, so nothing counts as elsewhere."""
    base = artifact_basename(step)
    learned = (artifact_paths or {}).get(step) or []
    if not base or not learned:
        return []
    out = {
        path
        for path in (write_paths or {}).get(base, ())
        if not any(dir_matches(normalized_dir(here), normalized_dir(path)) for here in learned)
    }
    return sorted(out)


def _satisfied_steps(baseline, trace) -> set:
    """Which of a trace's step ids count as taken, given where its writes actually landed.

    Every step that is not an artifact write is satisfied by its own presence, exactly as
    before #2502. An artifact write additionally needs its directory to be one the task's own
    history writes that artifact to (#2502 clause 2). The directory is deliberately *not* part
    of the step id: `write:<basename>` is the name a tool call gives a file, and widening the
    id would rename every artifact step in every baseline, invalidate the `validate` corpus's
    labelled fault labels, and make the false-alarm rate measured before this change
    incomparable with the rate after it. So the id stays a basename and the directory is
    checked against the captured paths here: the right filename at the wrong path leaves the
    step unsatisfied, which is what makes the 2026-10-09 vault-path swap visible at all, and
    `same_name_elsewhere` beside it is what tells a reader that this was a swap and not a skip.

    An artifact the baseline has no path for — every run that wrote it came through a capture
    shape that names no path — keeps the old behaviour, satisfied by its name alone. Refusing
    it would penalise the corpus for the shape of its traces, not the behaviour of its runs.
    """
    present = set(trace.steps)
    for step in list(present):
        learned = (baseline.artifact_paths or {}).get(step)
        if not (step.startswith(ESCALATE_PREFIX) and learned):
            continue
        observed = (trace.write_paths or {}).get(artifact_basename(step), ())
        if any(dir_matches(normalized_dir(here), normalized_dir(there))
               for here in learned for there in observed):
            continue
        present.discard(step)
    return present


def artifact_disk_witness(path, window_start, window_end, *, grace_seconds: int) -> dict | None:
    """Evidence that the artifact at its learned path was modified inside the run's own window,
    or None — the check that decides whether a missing-`write:` deviation may file (#2502
    clause 3).

    `path` is the single path `learned_artifact_path` names, and one path is the whole of the
    clause: the claim being answered is "this run produced no artifact", and a scan across every
    location the basename has ever appeared in would excuse a deviation on a file that is not
    this artifact. It is deliberately a stat and not a read, because one mtime answers that claim.

    The window is `[run start − grace, run completed + grace]`, both edges, because what is being
    excluded is an artifact last touched by some *earlier* night and a one-sided "newer than
    nothing" test is satisfied by a pointer that has been stale for a week. Measured on the two
    real cases the clause was written for: `_pipeline/reflection/conversation-patterns-latest.md`
    carried mtime 2026-10-08T11:40:05Z against the flagged run that completed
    2026-10-07T08:23:08Z — after it, because a later night wrote it — and the same mtime against
    the 2026-10-09 run's window, which closed before it. Neither is a write by the run that is
    being accused, and neither is witnessed.

    None for no path, an unreadable file, or an mtime outside the window: every one of those
    leaves the deviation exactly where it was. The record names the path read, the mtime and the
    window, so an answer is checkable from the report alone.
    """
    start, end = window_start, window_end
    if start is not None:
        start = start - timedelta(seconds=grace_seconds)
    if end is not None:
        end = end + timedelta(seconds=grace_seconds)
    if not path or start is None or end is None or end < start:
        return None
    try:
        stamp = datetime.fromtimestamp(_expand(path).stat().st_mtime, timezone.utc)
    except OSError:          # absent, or a path this host cannot stat: no witness
        return None
    if not start <= stamp <= end:
        return None
    return {
        "path": str(path),
        "mtime": stamp.isoformat(),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "grace_seconds": int(grace_seconds),
    }


def steps_from_tools(tools, write_paths: dict | None = None) -> list[str]:
    """Ordered, de-duplicated step ids for one run's tool calls, in `sequence` order.

    Three step kinds: `tool:<name>` for every call, plus `read:<basename>` and
    `write:<basename>` where the call names a file or the Bash command writes one.

    Pass a dict as `write_paths` and it comes back mapping each artifact basename to the
    captured paths behind it — the evidence both halves of #2502 read: the directory check in
    `_satisfied_steps` and the disk witness, each of which needs the path a step id does not
    carry. Collected in the same pass rather than by a second extractor, so the two can never
    disagree about which capture was a write.
    """
    steps: list[str] = []
    seen: set[str] = set()

    def add(step: str) -> None:
        if step and step not in seen:
            seen.add(step)
            steps.append(step)

    def add_write(base: str, raw) -> None:
        # The id is the basename alone — see `_satisfied_steps` for why the directory is
        # checked against these captured paths at match time instead of being folded in here.
        add(f"write:{base}")
        if write_paths is not None:
            seen_paths = write_paths.setdefault(base, [])
            if str(raw) not in seen_paths:
                seen_paths.append(str(raw))

    for tool in sorted(tools, key=lambda t: t.get("sequence") or 0):
        name = tool.get("name") or ""
        params = tool.get("params_summary") or {}
        add(f"tool:{name}")
        target = _path_target(params)
        if target and name in READ_TOOLS:
            base = normalized_basename(target)
            if base and base not in GENERIC_BASENAMES:
                add(f"read:{base}")
        for raw in write_targets(name, params):
            base = normalized_basename(raw)
            if base and base not in GENERIC_BASENAMES:
                add_write(base, raw)
    return steps


@dataclass
class Trace:
    """One run's ordered step trace, as recovered from the trajectory store."""

    session_key: str
    session_source: str
    steps: list[str]
    trace_ts: str | None = None
    #: Artifact step id → the paths captured behind it (#2502 clause 3). The step id says what
    #: was written; the witness needs the path, and the path has to survive the loader for the
    #: escalation half of the pipeline to be able to ask the disk anything.
    write_paths: dict = field(default_factory=dict)

    @property
    def task_number(self) -> int | None:
        _, _, tail = (self.session_source or "").rpartition(":")
        try:
            return int(tail)
        except ValueError:
            return None


@dataclass
class RunRecord:
    """The ledger row joined to a trace. `status` is what the run *recorded*, never inferred."""

    run_id: str
    task_id: str | None
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    session_ids: tuple[str, ...] = ()


@dataclass
class TaskBaseline:
    """What the runs of one `session_source` actually did, learned from history."""

    session_source: str
    task_id: int | None
    n_runs: int
    run_ids: list[str]
    steps_present: dict[str, int] = field(default_factory=dict)
    positions: dict[str, float] = field(default_factory=dict)
    expected_steps: list[str] = field(default_factory=list)
    #: Artifact step id → the paths the runs that took it wrote it to, in first-seen order
    #: (#2502 clause 3). This is "the learned path" the disk witness reads: a list and not a
    #: single value because a task's own history can name more than one — task 39's does, one
    #: side and the other of the 2026-09-22 data-home move — and any of them being touched in
    #: the run's window is a witness. Not in `to_json`: it is corpus-derived evidence that
    #: would go stale against a moved disk, and nothing reads the published baseline file for
    #: it.
    artifact_paths: dict[str, list[str]] = field(default_factory=dict)

    def expected_for(self, run_id: str, support: float = DEFAULT_SUPPORT) -> set[str]:
        """Steps every *other* run of this task has — leave-one-out, so a corpus cannot
        vouch for itself: the run that skipped a stage is in the history too, and counting it
        toward its own expectation would hide the skip. The subtraction is per step (does
        *this* run have it), never "the run is in the corpus, so it contributed everything" —
        that variant scores every skip clean, which is how such a check dies quietly."""
        others = self.n_runs - 1
        if others < 1:
            return set()
        need = _expected_threshold(self.n_runs, support)
        try:
            position = self.run_ids.index(run_id)
        except ValueError:
            position = None  # a run outside the corpus vouched for nothing here
        out: set[str] = set()
        for step, count in self.steps_present.items():
            flags = self.step_present_in.get(step, ())
            mine = 1 if position is not None and position < len(flags) and flags[position] else 0
            if count - mine >= need:
                out.add(step)
        return out

    def ordered(self, steps) -> list[str]:
        """Order a step set the way the baseline was learned: by learned median position."""
        return sorted(steps, key=lambda s: (self.positions.get(s, 1e9), s))

    def learned_from(self, step: str, exclude: str | None = None) -> list[str]:
        """The runs the expectation for `step` came from — the evidence, not a score."""
        out = []
        for run_id, present in zip(self.run_ids, self.step_present_in.get(step, ())):
            if present and run_id != exclude:
                out.append(run_id)
        return out

    step_present_in: dict[str, tuple[bool, ...]] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "session_source": self.session_source,
            "task_id": self.task_id,
            "n_runs": self.n_runs,
            "expected_steps": self.expected_steps,
            "learned_from": self.run_ids,
            "positions": {s: self.positions[s] for s in self.expected_steps},
            "runs_with_step": {s: self.steps_present[s] for s in self.expected_steps},
        }


@dataclass
class Deviation:
    """One structural deviation, carrying the evidence behind it and never a bare score."""

    task_id: int | None
    session_source: str
    run_id: str
    session_key: str
    run_status: str | None
    step: str
    kind: str
    state: str
    grace_seconds: int
    run_completed_at: str | None
    checked_at: str
    expected_steps: list[str]
    learned_from: list[str]
    learned_from_n: int
    baseline_runs: int
    #: Which of the three cases this is (`REASON_*`), and the two pieces of evidence #2502
    #: added: the disk witness that may excuse the deviation, and the paths the same-named
    #: artifact was written to instead. Defaults keep a deviation built from an older report
    #: file scoring-compatible: no reason recorded, nothing witnessed, nothing elsewhere.
    reason: str = REASON_MISSING
    witnessed: bool = False
    disk_witness: dict | None = None
    written_elsewhere: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "task_id": self.task_id,
            "session_source": self.session_source,
            "run_id": self.run_id,
            "session_key": self.session_key,
            "run_status": self.run_status,
            "step": self.step,
            "kind": self.kind,
            "state": self.state,
            "grace_seconds": self.grace_seconds,
            "run_completed_at": self.run_completed_at,
            "checked_at": self.checked_at,
            "expected_steps": list(self.expected_steps),
            "learned_from": list(self.learned_from),
            "learned_from_n": self.learned_from_n,
            "baseline_runs": self.baseline_runs,
            "reason": self.reason,
            "witnessed": self.witnessed,
            "disk_witness": dict(self.disk_witness) if self.disk_witness else None,
            "written_elsewhere": list(self.written_elsewhere),
        }


@dataclass
class RunReport:
    session_source: str
    task_id: int | None
    run_id: str
    session_key: str
    run_status: str | None
    deviations: list[Deviation] = field(default_factory=list)
    pending: list[Deviation] = field(default_factory=list)
    late_steps: list[dict] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.deviations)

    def to_json(self) -> dict:
        return {
            "session_source": self.session_source,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "session_key": self.session_key,
            "run_status": self.run_status,
            "deviations": [d.to_json() for d in self.deviations],
            "pending": [d.to_json() for d in self.pending],
            "late_steps": list(self.late_steps),
        }


@dataclass
class Corpus:
    """Learned baselines keyed by `session_source`, plus the traces they were learned from."""

    baselines: dict[str, TaskBaseline] = field(default_factory=dict)
    unscored: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "schema": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tasks": [b.to_json() for b in self.baselines.values()],
            "unscored_sessions": sorted(self.unscored),
        }


def _parse_ts(value, *, assume_local: bool = False):
    """Parse an ISO timestamp. Naive values are trajectory timestamps, which are written in
    machine-local time (a `session_key` of `20260921_060002_*` belongs to a 13:00Z run), so
    the caller says which side of the seam it is on."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.astimezone() if assume_local else parsed.replace(tzinfo=timezone.utc)
    return parsed


def load_traces(trajectory_dir, *, session_keys=None, since: datetime | None = None) -> list[Trace]:
    """Read the trajectory JSONL store into one `Trace` per session. Read-only."""
    root = _expand(trajectory_dir)
    wanted = set(session_keys) if session_keys is not None else None
    per_session: dict[str, list[dict]] = collections.defaultdict(list)
    sources: dict[str, str] = {}
    stamps: dict[str, str] = {}
    for path in sorted(root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = row.get("session_key")
                source = row.get("session_source") or ""
                if not key or not source:
                    continue
                if wanted is not None and key not in wanted:
                    continue
                ts = row.get("timestamp")
                if since is not None:
                    when = _parse_ts(ts, assume_local=True)
                    if when is None or when < since:
                        continue
                per_session[key].append(row)
                sources.setdefault(key, source)
                if ts and (key not in stamps or str(ts) < stamps[key]):
                    stamps[key] = str(ts)

    traces: list[Trace] = []
    for key, rows in per_session.items():
        tools = []
        for row in sorted(rows, key=lambda r: str(r.get("timestamp") or "")):
            tools.extend(row.get("tools") or [])
        paths: dict[str, list[str]] = {}
        steps = steps_from_tools(tools, paths)
        traces.append(
            Trace(
                session_key=key,
                session_source=sources[key],
                steps=steps,
                trace_ts=stamps.get(key),
                write_paths=paths,
            )
        )
    return traces


def load_runs(db_path, *, source: str = DEFAULT_SOURCE, since: datetime | None = None) -> list[RunRecord]:
    """Read `runs` from the worker ledger. Opened `mode=ro`: a replay cannot write the ledger."""
    path = _expand(db_path)
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        sql = "SELECT run_id, task_id, status, started_at, completed_at, meta_json FROM runs"
        clauses, args = [], []
        if source:
            clauses.append("source = ?")
            args.append(source)
        if since is not None:
            clauses.append("started_at >= ?")
            args.append(since.astimezone(timezone.utc).isoformat())
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY started_at"
        out: list[RunRecord] = []
        for row in conn.execute(sql, args):
            meta = {}
            try:
                meta = json.loads(row["meta_json"] or "{}")
            except (json.JSONDecodeError, TypeError):
                meta = {}
            ids = list(meta.get("session_ids") or [])
            if meta.get("session_id"):
                ids.append(meta["session_id"])
            seen: set[str] = set()
            uniq = tuple(i for i in ids if i and not (i in seen or seen.add(i)))
            out.append(
                RunRecord(
                    run_id=row["run_id"],
                    task_id=row["task_id"],
                    status=row["status"],
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                    session_ids=uniq,
                )
            )
        return out
    finally:
        conn.close()


def join_runs(traces: list[Trace], runs: list[RunRecord]):
    """Pair each trace with its ledger row. Returns (joined, unjoined) where `joined` is a
    list of `(trace, run_or_None)`. A trace with no ledger row is still scorable — its id is
    its session_key and its recorded status is `None`, which is reported, never guessed."""
    by_session: dict[str, RunRecord] = {}
    for run in runs:
        for sid in run.session_ids:
            by_session.setdefault(sid, run)
    joined, unjoined = [], []
    for trace in traces:
        run = by_session.get(trace.session_key)
        if run is None:
            unjoined.append(trace)
        joined.append((trace, run))
    return joined, unjoined


def _identity(trace: Trace, run: RunRecord | None) -> str:
    return run.run_id if run is not None else trace.session_key


def learn(traces_with_runs, *, min_runs: int = DEFAULT_MIN_RUNS, support: float = DEFAULT_SUPPORT) -> Corpus:
    """Learn the per-task ordered step set from history.

    Baselines are per `session_source` — never one global shape — because "POST for
    real-time payments" and "POST for wire payments" need separate numbers, and here task 68
    (mail/calendar) and task 38 (nightly signals) share nothing. A task with fewer than
    `min_runs` runs gets no baseline and lands in `Corpus.unscored` instead of a guess.
    """
    grouped: dict[str, dict[str, list[str]]] = collections.defaultdict(dict)
    # Artifact step → paths, per task, unioned over every run that took the step. Collected
    # beside the step counts rather than from them: the counts are what the expectation is,
    # and this is only the evidence of where the artifact lives.
    step_paths: dict[str, dict[str, list[str]]] = collections.defaultdict(dict)
    unscored: list[str] = []
    for trace, run in traces_with_runs:
        grouped[trace.session_source][_identity(trace, run)] = trace.steps
        for base, captured in (trace.write_paths or {}).items():
            # Keyed by step id, because that is what a deviation names and what a lookup at
            # match time has to find. Counted per run (a run that wrote the same artifact
            # twice still says "here" once), because the question asked of this below is how
            # many runs wrote it there, not how many writes there were.
            counts = step_paths[trace.session_source].setdefault(f"write:{base}", {})
            for raw in dict.fromkeys(captured):
                counts[raw] = counts.get(raw, 0) + 1
    corpus = Corpus()
    for source, step_sets in grouped.items():
        if len(step_sets) < min_runs:
            unscored.extend(step_sets.keys())
            continue
        counts: collections.Counter = collections.Counter()
        positions: dict[str, list[int]] = collections.defaultdict(list)
        for steps in step_sets.values():
            counts.update(set(steps))
            for index, step in enumerate(steps):
                positions[step].append(index)
        run_ids = list(step_sets.keys())
        present_in = {
            step: tuple(step in step_sets[rid] for rid in run_ids) for step in counts
        }
        median = {step: _median(pos) for step, pos in positions.items()}
        # The published set is the union of the per-run leave-one-out expectations: a step
        # every run but at most one took. The one run that skipped it is the reason the
        # published set and the deviation it receives agree instead of contradicting.
        expected = {
            step
            for step, n in counts.items()
            if n >= _expected_threshold(len(run_ids), support)
        }
        baseline = TaskBaseline(
            session_source=source,
            task_id=_task_number(source),
            n_runs=len(run_ids),
            run_ids=run_ids,
            steps_present=dict(counts),
            positions=median,
            step_present_in=present_in,
            artifact_paths={step: corroborated(counts)
                            for step, counts in step_paths.get(source, {}).items()},
        )
        baseline.expected_steps = baseline.ordered(expected)
        corpus.baselines[source] = baseline
    corpus.unscored = sorted(unscored)
    return corpus


def _task_number(session_source: str) -> int | None:
    _, _, tail = (session_source or "").rpartition(":")
    try:
        return int(tail)
    except ValueError:
        return None


def _expected_threshold(n_runs: int, support: float) -> int:
    """How many runs must carry a step for it to be expected of one more run.

    At strict `support` that is `n_runs - 1`: *expected of this run* means *every other run
    of this task has it*, which is the clause #673 is graded on and the only rule that lets a
    run be judged against a history it is itself part of.
    """
    others = n_runs - 1
    if others < 1:
        return 1
    return others if support >= 1.0 else max(1, math.ceil(support * others))


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return 0.0
    if n % 2:
        return float(ordered[n // 2])
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def _end_time(run: RunRecord | None, trace: Trace, now: datetime):
    if run is not None:
        ended = _parse_ts(run.completed_at) or _parse_ts(run.started_at)
        if ended is not None:
            return ended
    return _parse_ts(trace.trace_ts, assume_local=True) or now


def score_run(
    trace: Trace,
    run: RunRecord | None,
    corpus: Corpus,
    *,
    now: datetime | None = None,
    grace_seconds: int = DEFAULT_GRACE_SECONDS,
    support: float = DEFAULT_SUPPORT,
) -> RunReport | None:
    """Score one run against its task's learned shape.

    A run recorded `success` is scored, not skipped — that status is exactly what the
    detector exists to doubt, and it is copied onto every deviation so a reader can see the
    disagreement instead of having it smoothed over. Runs outside the learned set return
    `None` (no baseline yet), which is reported as *no baseline*, never as a clean run.
    """
    baseline = corpus.baselines.get(trace.session_source)
    if baseline is None:
        return None
    checked = now or datetime.now(timezone.utc)
    run_id = _identity(trace, run)
    status = run.status if run is not None else None
    report = RunReport(
        session_source=trace.session_source,
        task_id=baseline.task_id,
        run_id=run_id,
        session_key=trace.session_key,
        run_status=status,
    )
    expected = baseline.expected_for(run_id, support=support)
    # Not `set(trace.steps)`: an artifact write counts only if it landed where that task's own
    # history lands it (#2502 clause 2).
    present = _satisfied_steps(baseline, trace)
    index_of = {step: i for i, step in enumerate(trace.steps)}
    ended = _end_time(run, trace, checked)
    inside_grace = (checked - ended) < timedelta(seconds=grace_seconds)
    started = (_parse_ts(run.started_at) if run is not None else None) or ended
    for step in baseline.ordered(expected - present):
        state = "pending" if inside_grace else "deviation"
        # #2502 clause 2: an artifact expectation is about the artifact at the path the task's
        # history writes it to, so a write of the same filename somewhere else is a deviation
        # of its own kind, and an unmet one. It does not satisfy the step, and the run that
        # made it is named in `written_elsewhere` so the reader sees where the content went.
        elsewhere = same_name_elsewhere(step, trace.write_paths, baseline.artifact_paths)
        # #2502 clause 3: before this may become a draft, ask the disk. The artifact's learned
        # paths are the ones this task's own history wrote it to; one of them touched inside
        # this run's window disproves "this run produced no artifact", whatever the trace's
        # grammar captured — and `steps_from_tools` reads only a redirect or a writer token, so
        # a `python3 - <<'EOF'` write is invisible to it and has always been.
        witness = None
        if step.startswith(ESCALATE_PREFIX):
            witness = artifact_disk_witness(
                learned_artifact_path(baseline.artifact_paths, step), started, ended,
                grace_seconds=grace_seconds,
            )
        if witness is not None:
            reason = REASON_WITNESSED
        elif elsewhere:
            reason = REASON_OTHER_PATH
        else:
            reason = REASON_MISSING
        deviation = Deviation(
            task_id=baseline.task_id,
            session_source=trace.session_source,
            run_id=run_id,
            session_key=trace.session_key,
            run_status=status,
            step=step,
            kind="structural",
            state=state,
            grace_seconds=grace_seconds,
            run_completed_at=ended.isoformat(),
            checked_at=checked.isoformat(),
            expected_steps=list(baseline.expected_steps),
            learned_from=baseline.learned_from(step, exclude=run_id),
            learned_from_n=len(baseline.learned_from(step, exclude=run_id)),
            baseline_runs=baseline.n_runs,
            reason=reason,
            witnessed=witness is not None,
            disk_witness=witness,
            written_elsewhere=elsewhere,
        )
        (report.pending if state == "pending" else report.deviations).append(deviation)
    for step in baseline.ordered(present & expected):
        learned_position = baseline.positions.get(step)
        if learned_position is None:
            continue
        seen_at = index_of[step]
        if seen_at > learned_position:
            report.late_steps.append(
                {"step": step, "seen_at": seen_at, "learned_position": learned_position}
            )
    return report


def replay(
    *,
    trajectory_dir=DEFAULT_TRAJECTORIES,
    db_path=DEFAULT_DB,
    source: str = DEFAULT_SOURCE,
    days: int | None = None,
    now: datetime | None = None,
    grace_seconds: int = DEFAULT_GRACE_SECONDS,
    min_runs: int = DEFAULT_MIN_RUNS,
    support: float = DEFAULT_SUPPORT,
) -> dict:
    """Replay history through the conformance check. Read-only on both stores."""
    checked = now or datetime.now(timezone.utc)
    since = checked - timedelta(days=days) if days else None
    traces = load_traces(trajectory_dir, since=since)
    runs = load_runs(db_path, source=source, since=since)
    joined, unjoined = join_runs(traces, runs)
    corpus = learn(joined, min_runs=min_runs, support=support)
    reports = []
    for trace, run in joined:
        report = score_run(trace, run, corpus, now=checked, grace_seconds=grace_seconds, support=support)
        if report is not None:
            reports.append(report)
    return {
        "trajectory_dir": str(_expand(trajectory_dir)),
        "db": str(_expand(db_path)),
        "source": source,
        "days": days,
        "checked_at": checked.isoformat(),
        "grace_seconds": grace_seconds,
        "min_runs": min_runs,
        "support": support,
        "runs_in_ledger": len(runs),
        "traces_loaded": len(traces),
        "traces_without_ledger_row": len(unjoined),
        "runs_scored": len(reports),
        "tasks_scored": len(corpus.baselines),
        "tasks_without_baseline": len(corpus.unscored),
        "runs_flagged": sum(1 for r in reports if r.flagged),
        "runs_pending": sum(1 for r in reports if r.pending and not r.flagged),
        "deviations": sum(len(r.deviations) for r in reports),
        "pending": sum(len(r.pending) for r in reports),
        # Of the deviations above, how many the disk spoke about (#2502 clause 3): witnessed
        # ones are reported and never filed, other-path ones name a write that went somewhere
        # else. Both are subtotals of `deviations`, never additional ones, so a reader adding
        # the two must not read them as more findings.
        "deviations_witnessed": sum(
            1 for r in reports for d in r.deviations if d.witnessed
        ),
        "deviations_written_elsewhere": sum(
            1 for r in reports for d in r.deviations if d.written_elsewhere
        ),
        # The rate a reader compares against the acceptance bound, stated as what it is. On
        # unlabelled traffic a flag may be a true positive, so `flagged_rate` is an *upper
        # bound* on the false-alarm rate and `*_flagged` below counts true positives and false
        # alarms together; the labelled corpus in `validate` is the only place the false-alarm
        # rate is actually measured.
        "flagged_rate": (
            sum(1 for r in reports if r.flagged) / len(reports) if reports else 0.0
        ),
        "success_recorded_runs": sum(1 for r in reports if r.run_status == "success"),
        "success_recorded_quiet": sum(
            1 for r in reports if r.run_status == "success" and not r.flagged
        ),
        "success_recorded_flagged": sum(
            1 for r in reports if r.run_status == "success" and r.flagged
        ),
        "tasks": sorted(corpus.baselines),
        "published": corpus.to_json(),
        "baseline_digest": hashlib.sha256(
            json.dumps(corpus.to_json()["tasks"], sort_keys=True).encode()
        ).hexdigest()[:16],
        "reports": [r.to_json() for r in reports],
        "corpus": corpus,
    }


# ---------------------------------------------------------------------------
# Offline injected-fault fixtures
#
# His validation story was millions of traces with anomalies injected *before* going live,
# so acceptance here is against labelled fixtures, not a vibe: three faults that re-create
# Lloyd's own confirmed incidents, plus benign controls that vary order and add steps and
# must stay quiet. Built as real files — trajectory JSONL plus a real sqlite `runs` table —
# and read back through the same read-only loaders production uses, so the fixtures exercise
# the join and not just the comparison.
# ---------------------------------------------------------------------------

FIXTURE_SOURCE = "scheduled-task"

FIXTURE_TASKS = {
    # Incident 1, the shape of `dream-consolidation` 09-03 -> 09-09: the job stamps a
    # presence/lock file through a shell redirect, and the file's directory no longer exists.
    "autonomy-task:47": {
        "clean": [
            {"tool": "Read", "file_path": "/vault/skills/dream-consolidation/SKILL.md"},
            {"tool": "Bash", "command": "date -u +%FT%TZ > /vault/agents/lloyd/.consolidate-lock"},
            {"tool": "Read", "file_path": "/lloyd/_pipeline/reflection/signals-latest.md"},
            {"tool": "Write", "file_path": "/lloyd/_pipeline/reflection/dream-report-latest.md"},
        ],
        "fault_step": "write:.consolidate-lock",
    },
    # Incident 2, the shape of the 09-08 ordering inversion: the dependent ran while its
    # upstream artifact did not exist, so the consumption step never happened. Structural
    # signature = the read is missing; whether the run was *allowed* is #558's gate, not this.
    "autonomy-task:39": {
        "clean": [
            {"tool": "Read", "file_path": "/lloyd/_pipeline/reflection/knowledge-handoff-2026-09-20.md"},
            {"tool": "Bash", "command": "wc -c /lloyd/_pipeline/reflection/knowledge-handoff-2026-09-20.md"},
            {"tool": "Write", "file_path": "/lloyd/_pipeline/reflection/knowledge-latest.md"},
            {"tool": "Bash", "command": "cd /lloyd && git add _pipeline/reflection/knowledge-latest.md"},
        ],
        "fault_step": "read:knowledge-handoff-<date>.md",
    },
    # Incident 3, the shape of the 09-01/09-02 missing-handoff family: the upstream stage
    # produced no handoff artifact and the chain still reported green.
    "autonomy-task:40": {
        "clean": [
            {"tool": "Read", "file_path": "/lloyd/_pipeline/reflection/knowledge-latest.md"},
            {"tool": "Write", "file_path": "/lloyd/_pipeline/reflection/knowledge-handoff-2026-09-20.md"},
            {"tool": "Bash", "command": "wc -c /lloyd/_pipeline/reflection/knowledge-handoff-2026-09-20.md"},
            {"tool": "Bash", "command": "cd /lloyd && git add -A"},
        ],
        "fault_step": "write:knowledge-handoff-<date>.md",
    },
}

# Benign variation a conformance check must survive: an extra step, a different order, and
# the same steps with the artifact step emitted late (his 1-of-7-node late span).
BENIGN_VARIANTS = {
    "plain": lambda calls: list(calls),
    # A call no clean run makes: an extra step must never be a deviation, only a missing one.
    "extra": lambda calls: calls + [{"tool": "vault_search"}],
    "reordered": lambda calls: list(reversed(calls)),
    "late": lambda calls: calls[1:2] + calls[2:] + calls[0:1],
}


def _tool_call_from_spec(spec: dict, sequence: int) -> dict:
    name = spec["tool"]
    params = {"summary": f"fixture step {sequence}"}
    if "file_path" in spec:
        params["file_path"] = spec["file_path"]
    if "command" in spec:
        params["command"] = spec["command"]
    return {
        "name": name,
        "params_summary": params,
        "result_summary": "OK: 42 chars",
        "is_error": False,
        "error_source": None,
        "exit_code": None,
        "sequence": sequence,
    }


def _trajectory_row(session_key: str, source: str, calls: list[dict], stamp: str) -> dict:
    return {
        "session_key": session_key,
        "agent_id": "lloyd",
        "session_class": "autonomy",
        "session_source": source,
        "timestamp": stamp,
        "tool_count": len(calls),
        "error_count": 0,
        "has_errors": False,
        "tools": [_tool_call_from_spec(c, i) for i, c in enumerate(calls, start=1)],
    }


def synth_fixtures(dest, *, per_task_benign: int = 6, day: str = "2026-09-20") -> dict:
    """Write a labelled fixture corpus (trajectory JSONL + `runs.db`) under `dest`.

    Returns the labels: which run ids are injected faults (and the step each one omits) and
    which are benign controls. Benign runs carry `status='success'` — a failed run is not
    evidence of anything here, since the whole point is runs that reported success.
    """
    root = _expand(dest)
    traj = root / "trajectories"
    traj.mkdir(parents=True, exist_ok=True)
    db_path = root / "runs.db"
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE runs (run_id TEXT PRIMARY KEY, queue_id TEXT, source TEXT, task_id TEXT,"
        " status TEXT, started_at TEXT, completed_at TEXT, duration_seconds REAL, summary TEXT,"
        " artifact_path TEXT, response_json TEXT, meta_json TEXT, claims_json TEXT)"
    )
    rows = []
    labels = {"injected": [], "benign": []}
    conn.commit()

    for source, spec in FIXTURE_TASKS.items():
        task_id = _task_number(source)
        clean = list(spec["clean"])
        fault_step = spec["fault_step"]
        for i in range(per_task_benign):
            variant = list(BENIGN_VARIANTS[["plain", "extra", "reordered", "late", "plain", "extra"][i % 6]](clean))
            session_key = f"2026092{i}_0{i}0000_autonomy_b{i}{source[-2:]}"
            stamp = f"2026-09-{20 - i}T0{i}:00:00"
            rows.append(_trajectory_row(session_key, source, variant, stamp))
            # Labels are *run ids*, the key the join scores under — labelling by session key
            # would look up nothing and score every benign run "quiet" by construction.
            labels["benign"].append(f"run_benign_{source[-2:]}_{i}")
            conn.execute(
                "INSERT INTO runs (run_id, source, task_id, status, started_at, completed_at, meta_json)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    f"run_benign_{source[-2:]}_{i}",
                    FIXTURE_SOURCE,
                    str(task_id),
                    "success",
                    f"{day}T{i}:00:00+00:00",
                    f"{day}T{i}:09:00+00:00",
                    json.dumps({"session_id": session_key, "session_ids": [session_key]}),
                ),
            )
        # The injected fault: the same run, with one step never taken, still recorded success.
        faulty = [c for c in clean if fault_step not in set(steps_from_tools([_tool_call_from_spec(c, 1)]))]
        session_key = f"20260920_050000_autonomy_fault{source[-2:]}"
        rows.append(_trajectory_row(session_key, source, faulty, "2026-09-20T05:00:00"))
        labels["injected"].append(
            {
                "run_id": f"run_fault_{source[-2:]}",
                "session_key": session_key,
                "session_source": source,
                "task_id": task_id,
                "expected_missing_step": fault_step,
                "incident": {
                    "autonomy-task:47": "presence/lock stamp never written (09-03 -> 09-09 gate)",
                    "autonomy-task:39": "dependent ran before its upstream artifact existed (09-08)",
                    "autonomy-task:40": "nightly reflection handoff never written (09-01/09-02)",
                }[source],
            }
        )
        conn.execute(
            "INSERT INTO runs (run_id, source, task_id, status, started_at, completed_at, meta_json)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                f"run_fault_{source[-2:]}",
                FIXTURE_SOURCE,
                str(task_id),
                "success",
                f"{day}T05:00:00+00:00",
                f"{day}T05:12:00+00:00",
                json.dumps({"session_id": session_key, "session_ids": [session_key]}),
            ),
        )
    conn.commit()
    conn.close()

    day_file = traj / f"{day}.jsonl"
    with day_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    labels["trajectory_dir"] = str(traj)
    labels["db"] = str(db_path)
    (root / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return labels


def validate_fixtures(
    *,
    trajectory_dir,
    db_path,
    labels: dict,
    now: datetime | None = None,
    grace_seconds: int = DEFAULT_GRACE_SECONDS,
    min_runs: int = DEFAULT_MIN_RUNS,
    support: float = DEFAULT_SUPPORT,
) -> dict:
    """Score a labelled fixture corpus. Detection is counted only when the deviation *names*
    the step the fault removed; the benign corpus is scored in the same pass so a
    positives-only tuning cannot report a pass."""
    checked = now or datetime.now(timezone.utc)
    traces = load_traces(trajectory_dir)
    runs = load_runs(db_path)
    joined, _unjoined = join_runs(traces, runs)
    corpus = learn(joined, min_runs=min_runs, support=support)
    by_id: dict[str, tuple[Trace, RunRecord | None]] = {}
    for trace, run in joined:
        by_id[_identity(trace, run)] = (trace, run)

    detected, missed = [], []
    for fault in labels["injected"]:
        pair = by_id.get(fault["run_id"])
        hit = None
        if pair is not None:
            report = score_run(pair[0], pair[1], corpus, now=checked, grace_seconds=grace_seconds, support=support)
            if report:
                for deviation in report.deviations:
                    if deviation.step == fault["expected_missing_step"]:
                        hit = deviation
                        break
        if hit is None:
            missed.append(fault)
        else:
            detected.append((fault, hit))
    benign_quiet, benign_flagged = 0, []
    for run_id in labels["benign"]:
        pair = by_id.get(run_id)
        if pair is None:
            continue
        report = score_run(pair[0], pair[1], corpus, now=checked, grace_seconds=grace_seconds, support=support)
        if report and report.flagged:
            benign_flagged.append(
                {
                    "run_id": run_id,
                    "steps": [d.step for d in report.deviations],
                    "run_status": report.run_status,
                }
            )
        else:
            benign_quiet += 1
    n_benign = len(labels["benign"])
    return {
        "checked_at": checked.isoformat(),
        "detected": [
            {"run_id": fault["run_id"], "step": dev.step, "incident": fault["incident"],
             "run_status": dev.run_status, "deviation": dev.to_json()}
            for fault, dev in detected
        ],
        "missed": [dict(f) for f in missed],
        "benign_flagged": benign_flagged,
        "benign_quiet": benign_quiet,
        "counts": {
            "detected": len(detected),
            "injected": len(labels["injected"]),
            "false_alarms": len(benign_flagged),
            "benign": n_benign,
            "benign_scored": benign_quiet + len(benign_flagged),
        },
        "corpus": corpus,
    }


def summary_line(counts: dict) -> str:
    """The one line a reader has to see: positives beside negatives, never a lone number."""
    det, inj = counts["detected"], counts["injected"]
    fa, benign = counts["false_alarms"], counts["benign"]
    scored = counts.get("benign_scored", benign)
    rate = (fa / scored * 100.0) if scored else float("nan")
    # A corpus that scored no benign run cannot pass however well it caught the faults: the
    # denominator is the whole reason the rate is printed, and `scored != benign` means a
    # labelled control never joined a trace — the other way a benign corpus goes quiet.
    verdict = "PASS" if (inj and det == inj and fa == 0 and scored == benign and scored > 0) else "FAIL"
    return (
        f"detection={det}/{inj} ({det / inj * 100.0 if inj else float('nan'):.1f}%)  "
        f"false_alarms={fa}/{scored} ({rate:.1f}%)  "
        f"benign_labelled={benign}  verdict={verdict}"
    )


def _hash_inputs(paths) -> dict:
    out = {}
    for path in paths:
        p = _expand(path)
        if p.is_file():
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ---------------------------------------------------------------------------
# The one escalation (#2104 clause 4): a missing *written* artifact files a draft
# ---------------------------------------------------------------------------
#
# Why it exists: the detector shipped as a report, and a report nobody reads every morning is
# an undetected failure. Why it is a draft and not an alert: every rollback on the guardian's
# record was a false positive, so the surface has to be a queue entry a human adjudicates at
# their desk rather than one that pages anybody — and `draft` is the status the autocode loop
# never reads, which is the difference between queueing a judgement and queueing work nobody
# made.
#
# Which deviations qualify, measured on the 14-day live replay of 2026-10-03 (11 flagged runs,
# 22 deviations). A missing `write:` is the only one of the three step kinds that proves a stage
# did not happen, because it names an artifact that does not exist. A missing `tool:` is one
# call replaced by another, which is legal and common — `tool:TodoWrite` alone is 11 of those 22
# deviations, so escalating it would file on 9 of the 11 flagged runs. A missing `read:` is the
# same shape one step over: a read is evidence of *how* a stage ran, not that it ran, and task
# 42's live `expected_steps` that day were nothing but `read:` ids, so escalating reads would
# pile a second draft on top of task 40's genuinely missing write.
#
# Which *runs* qualify is a second gate, added by #2477: the run that recorded green. On the
# 14-day live replay of 2026-10-09 `runs_flagged=12/6413` with `flagged_among_them=9`, so 3 of
# the 12 flagged runs sit on a non-success status — and the set that qualified to file was
# narrower still and entirely aborted: 2 targets, both `run_status='failed'` (task 24 missing
# `write:nightly_extraction.lock`, filed as #2238 and closed `stale`; task 38 missing
# `write:signals-latest.md`, filed as #2473). Both drafts asserted a dropped stage over an abort
# the ledger recorded with a cause, and both were adjudicated by hand.
ESCALATE_PREFIX = "write:"
# The tag the write-time merge rule and `expire_stale_spawns` both act on: an escalation is
# loop output, so it must be mergeable and it must expire if nobody ever reads it.
ESCALATION_TAG = "spawned-by-step-conformance"
ESCALATION_BOARD = "lloyd"      # Alan's own code board — the one autotriage reads
ESCALATION_STATUS = "draft"     # never `up_next`: a candidate is not a task


def escalates(step: str) -> bool:
    """True for the one deviation shape that files, and false for the other two.

    One positive gate, on purpose: `write:` is the only prefix that names an artifact, so it is
    the only one that can be *missing* in the evidential sense. What a run did instead of
    calling a tool, or which file it read on the way to the same output, is named by its own
    prefix and stays a report line. The measurement above is why a broader rule is not merely
    noisier: escalating `tool:` would have filed on 9 of the 11 flagged runs of the 2026-10-03
    replay."""
    return step.startswith(ESCALATE_PREFIX)


#: The terminal statuses the fleet already records with a cause (#2477) — the same pair the
#: queue's own rollup bills as failures (`workers/queue.py:1301`:
#: `SUM(status IN ('failed','interrupted'))`). The measured vocabulary of the 14-day ledger at
#: 2026-10-09 is `success` 5008, `skipped` 1923, `failed` 440, `interrupted` 3, and `skipped`
#: is not here because it is not a failure (`workers/pool.py:41-42`).
ABORTED_RUN_STATUSES = frozenset({"failed", "interrupted"})


def recorded_failure(run_status: str | None) -> bool:
    """True when the ledger already explains why an artifact is absent (#2477).

    A run the fleet recorded as `failed` or `interrupted` never had the chance to write its
    artifact, so the absence is a consequence of the abort and `workers.db` carries the cause —
    `#2473`'s run died of `StreamStalledError: stream produced no data for 60s after 564
    line(s)` with 11 tool calls and 0 errors, mid-skill before the pointer publish. Filing a
    draft on top of that asks a human to re-derive, from the same row the escalation already
    joined, the one judgement the ledger has already made. `success` is deliberately absent: a
    run that reported green while dropping its stage is the class this detector exists to doubt.
    So is `None` — the status of a trace with no ledger row, common under `--source` — because
    an unknown status is not a recorded failure, and a file-only-on-success rule would silently
    drop that coverage."""
    return run_status in ABORTED_RUN_STATUSES


def _backlog_store():
    """The shared backlog writer, imported lazily.

    `scripts/automod/backlog.py::new_item` is the Python half of the board the MCP tool, the
    HTTP route and the red-tree filer all write through — id allocation `max + 1` under
    `O_EXCL`, front matter and tags by the create-time rule, and the store's one clock. This
    module is loaded by `importlib` from a path *and* run by the nightly subprocess, so the
    import lives inside the call and costs a replay that never escalates nothing. It is also the
    only write this file makes outside its own report: see
    `tests/test_step_conformance_escalation.py::test_the_draft_is_the_only_file_the_escalation_creates`."""
    from scripts.automod import backlog

    return backlog


def escalation_targets(result: dict) -> list[dict]:
    """The deviations of one replay that qualify, at most one per run.

    Read off `replay`'s own reports, so what files is exactly what printed: a deviation with a
    `state` of `deviation` (a `pending` one is inside the grace window and not yet a finding)
    whose step names a written artifact, for a run the ledger did not already record as a
    failure (#2477 — `recorded_failure`), and one whose artifact the disk did not just update
    at its learned path (#2502 clause 3 — `REASON_WITNESSED`). This is the escalation gate, not
    the scoring gate: an aborted run's missing `write:`, and a witnessed one, stay in
    `result["reports"]`, in `print_replay`'s output and in `runs_flagged`, and only the draft is
    suppressed. The run ids the expectation was learned from ride along, because the draft is a
    handoff and the evidence is the handoff.

    The witness is the third suppression, and it suppresses a claim rather than a signal: the
    draft would assert that this run produced no artifact, and an mtime inside the run's own
    window at the path that task's history writes it to disproves that sentence. What survives
    in the report is the true residue — the trace names no write where the task's other runs
    name one, which is a gap in what `steps_from_tools` can read and is printed, not filed."""
    out: list[dict] = []
    for report in result.get("reports") or []:
        if recorded_failure(report.get("run_status")):
            continue
        for deviation in report.get("deviations") or []:
            step = str(deviation.get("step") or "")
            if not escalates(step):
                continue
            if deviation.get("witnessed"):
                continue
            out.append({
                "reason": deviation.get("reason") or REASON_MISSING,
                "written_elsewhere": list(deviation.get("written_elsewhere") or []),
                "disk_witness": deviation.get("disk_witness"),
                "task_id": report.get("task_id"),
                "session_source": report.get("session_source") or "",
                "run_id": report.get("run_id") or "",
                "session_key": report.get("session_key") or "",
                "run_status": report.get("run_status"),
                "step": step,
                "learned_from": list(deviation.get("learned_from") or []),
                "learned_from_n": deviation.get("learned_from_n", 0),
                "baseline_runs": deviation.get("baseline_runs", 0),
                "run_completed_at": deviation.get("run_completed_at"),
            })
            break  # one escalation per run, whatever else that run is missing
    return out


def escalation_name(target: dict) -> str:
    """The draft's title, and the key its dedupe turns on.

    `new_item` slugifies the name and keeps the first 50 characters, and a full run id is longer
    than the budget leaves room for after the prefix — so the name keeps the run id's leading
    `run_<source>_<date>_<time>`, which is unique per run. The dedupe below compares full names,
    never filenames, so it does not depend on that truncation."""
    artifact = target["step"].split(":", 1)[1]
    return (f"step-conformance: task {target.get('task_id')} run {target['run_id']}"
            f" never wrote {artifact}")


def escalation_body(target: dict) -> str:
    """The draft's prose: every identifier needed to reproduce the judgement, and the reason it
    is a candidate rather than a finding."""
    learned = ", ".join(target["learned_from"]) or "(none recorded)"
    return "\n".join([
        "The read-only step-conformance replay (`scripts/step_conformance.py replay`) flagged"
        " this run: it recorded a terminal status while a step that every *other* run of its"
        " task took — writing an artifact — never happened. The artifact is the evidence, which"
        " is why this is the one deviation shape the replay escalates.",
        "",
        "It is a **candidate**, not a finding. The corpus is unlabelled real traffic, so the"
        " false-alarm rate here is UNMEASURED; the labelled rate comes from the detector's"
        " `validate` subcommand. A renamed output, a deliberately dropped stage, or a baseline"
        " still learning from a thin history all look like this. Adjudicate before promoting"
        " it: a draft is the status the autocode loop never reads, which is the point.",
        "",
        f"- task_id: {target.get('task_id')}",
        f"- session_source: {target['session_source']}",
        f"- run_id: {target['run_id']}",
        f"- session_key: {target['session_key']}",
        f"- run_status: {target['run_status']}",
        f"- missing_step: {target['step']}",
        f"- reason: {target.get('reason') or REASON_MISSING}",
        # #2502: the two things that make the difference between a human adjudicating this in
        # one read and going back to `stat` and `git log` to work out which of the two cases it
        # is. The path the artifact is expected at, and — when there is one — the path the same
        # filename was written to instead, which is the whole finding for the 2026-10-09 shape.
        "- written_elsewhere: "
        + (", ".join(target.get("written_elsewhere") or []) or "(none: no same-named write"
                                                       " anywhere else in this run)"),
        f"- learned_from_n: {target['learned_from_n']} of {target['baseline_runs']} baseline runs",
        f"- learned_from: {learned}",
        f"- run_completed_at: {target.get('run_completed_at')}",
        "",
        "Reproduce it read-only (a replay writes nothing unless `--escalate-writes` is passed,"
        " and this item is what that flag writes):",
        "",
        "```",
        f"python3 scripts/step_conformance.py replay --days 14 | grep -F {target['run_id']!r}",
        "```",
    ])


def file_escalations(result: dict, *, backlog_dir=None, dry_run: bool = False) -> list[dict]:
    """File one backlog draft per escalating run and return what was filed.

    Three rules the tests read off the disk. One draft per run: a re-check days later sees the
    same deviation and files nothing, because the dedupe compares the *name this escalation would
    file* against every item already on the board — so two runs of one task file twice (both are
    evidence) while one run never files twice. `dry_run` prints the same answer without writing.
    And the only write is the item: no ledger handle, no statement, no notification — see
    `tests/test_step_conformance_escalation.py::test_the_escalation_path_makes_no_alert_call_and_writes_no_store`.
    """
    targets = escalation_targets(result)
    if not targets:
        return []
    store = _backlog_store()
    root = _expand(backlog_dir) if backlog_dir else store.BACKLOG_DIR
    filed: list[dict] = []
    known = {item.name for item in store.all_items(None, backlog_dir=root)}
    for target in targets:
        name = escalation_name(target)
        if name in known:
            continue
        if dry_run:
            filed.append({**target, "name": name, "path": "", "dry_run": True})
            continue
        item = store.new_item(
            name,
            escalation_body(target),
            priority="low",
            tags=(ESCALATION_TAG,),
            status=ESCALATION_STATUS,
            board=ESCALATION_BOARD,
            backlog_dir=root,
        )
        known.add(name)
        filed.append({**target, "name": name, "path": str(item.path), "item_id": item.id})
    return filed


def print_escalations(filed: list[dict], result: dict, *, dry_run: bool) -> None:
    """The escalation block. The count goes beside its denominators — qualifying deviations and
    flagged runs — never as a lone number, because a filer that filed 1 of 0 qualifying deviations
    would be a filer silently doing nothing."""
    targets = escalation_targets(result)
    print(
        f"\nescalations_filed={len(filed)}/{len(targets)}"
        f" escalations_deduped={len(targets) - len(filed)}"
        f" runs_flagged={result['runs_flagged']}"
        f" dry_run={str(dry_run).lower()}"
        f" — non-alerting: the only write is a backlog draft"
        f" (board={ESCALATION_BOARD}, status={ESCALATION_STATUS}, tag={ESCALATION_TAG})"
    )
    for entry in filed:
        print(f"  - task {entry.get('task_id')} {entry['run_id']} missing {entry['step']!r}"
              f" -> {entry['path'] or '(not written: dry run)'}")


def print_replay(result: dict) -> None:
    print(f"# step conformance replay (read-only) — {result['trajectory_dir']} + {result['db']}")
    print(
        f"checked_at={result['checked_at']} days={result['days']} grace_s={result['grace_seconds']}"
        f" min_runs={result['min_runs']} support={result['support']}"
    )
    print(
        f"ledger_runs={result['runs_in_ledger']} traces={result['traces_loaded']}"
        f" traces_unjoined={result['traces_without_ledger_row']}"
    )
    print(
        f"tasks_scored={result['tasks_scored']} tasks_without_baseline={result['tasks_without_baseline']}"
        f" runs_scored={result['runs_scored']}"
    )
    flagged = [r for r in result["reports"] if r["deviations"]]
    print(
        f"runs_flagged={result['runs_flagged']}/{result['runs_scored']}"
        f" ({result['runs_flagged'] / result['runs_scored'] * 100.0 if result['runs_scored'] else 0.0:.2f}%)"
        f" runs_pending={result['runs_pending']} deviations={result['deviations']} pending={result['pending']}"
        f" disk_witnessed={result['deviations_witnessed']}"
        f" written_elsewhere={result['deviations_written_elsewhere']}"
    )
    print(
        f"success_recorded_runs={result['success_recorded_runs']}"
        f" flagged_among_them={result['success_recorded_flagged']}"
        f" quiet_among_them={result['success_recorded_quiet']}"
        f" flag_rate_of_scored={result['flagged_rate'] * 100.0:.2f}%"
    )
    print(
        "NOTE: unlabelled real traffic — the false-alarm rate on this corpus is UNMEASURED, a"
        " flagged run is a *candidate* skipped step for a human to adjudicate and may be a true"
        " positive. The labelled rate comes from `validate`. Nothing here alerts."
    )
    for report in flagged:
        print(
            f"\n- task {report['task_id']} {report['run_id']} status={report['run_status']}"
            f" session={report['session_key']}"
        )
        for dev in report["deviations"]:
            print(
                f"    missing {dev['step']!r} (structural, reason={dev.get('reason')}) —"
                f" expected in {dev['learned_from_n']}"
                f" of {dev['baseline_runs']} runs; learned_from={dev['learned_from'][:5]}"
            )
            if dev.get("written_elsewhere"):
                print(f"    written at another path instead: {dev['written_elsewhere']}")
            witness = dev.get("disk_witness")
            if witness:
                print(
                    f"    disk witness: {witness['path']} mtime={witness['mtime']} inside"
                    f" [{witness['window_start']}, {witness['window_end']}]"
                    f" ±{witness['grace_seconds']}s — not escalated: the artifact moved during"
                    " this run, the trace just never named the write"
                )
            print(f"    expected_steps={dev['expected_steps']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--trajectories", default=DEFAULT_TRAJECTORIES)
    common.add_argument("--db", default=DEFAULT_DB)
    common.add_argument("--source", default=DEFAULT_SOURCE)
    common.add_argument("--days", type=int, default=None)
    common.add_argument("--grace-seconds", type=int, default=DEFAULT_GRACE_SECONDS)
    common.add_argument("--min-runs", type=int, default=DEFAULT_MIN_RUNS)
    common.add_argument("--support", type=float, default=DEFAULT_SUPPORT)

    p_replay = sub.add_parser("replay", parents=[common], help="read-only replay over real history")
    p_replay.add_argument("--json", default=None, help="write the full report here (outside the corpus)")
    # Off by default: a human running the replay to look at something must not be able to put
    # work on the board by looking at it. The nightly task names the flag on purpose.
    p_replay.add_argument("--escalate-writes", action="store_true",
                          help="file a backlog DRAFT for each flagged run missing a written "
                               "artifact (non-alerting; the only write outside --json)")
    p_replay.add_argument("--dry-run", action="store_true",
                          help="with --escalate-writes: print what would be filed, write nothing")
    p_replay.add_argument("--backlog-dir", default=None,
                          help="board directory to file into (default: the store's own)")

    p_learn = sub.add_parser("learn", parents=[common], help="publish the learned baseline")
    p_learn.add_argument("--out", default=None)

    p_val = sub.add_parser("validate", parents=[common], help="offline injected-fault fixtures")
    p_val.add_argument("--fixtures", default=None, help="fixture dir; built into a temp dir if omitted")
    p_val.add_argument("--json", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "replay":
        result = replay(
            trajectory_dir=args.trajectories,
            db_path=args.db,
            source=args.source,
            days=args.days,
            grace_seconds=args.grace_seconds,
            min_runs=args.min_runs,
            support=args.support,
        )
        print_replay(result)
        if args.escalate_writes:
            filed = file_escalations(
                result, backlog_dir=args.backlog_dir, dry_run=args.dry_run
            )
            print_escalations(filed, result, dry_run=args.dry_run)
        if args.json:
            payload = {k: v for k, v in result.items() if k != "corpus"}
            _expand(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"full report written to {args.json}")
        return 0

    if args.cmd == "learn":
        since = datetime.now(timezone.utc) - timedelta(days=args.days) if args.days else None
        traces = load_traces(args.trajectories, since=since)
        runs = load_runs(args.db, source=args.source, since=since)
        joined, _ = join_runs(traces, runs)
        corpus = learn(joined, min_runs=args.min_runs, support=args.support)
        payload = json.dumps(corpus.to_json(), indent=2)
        if args.out:
            _expand(args.out).write_text(payload, encoding="utf-8")
            print(f"published baseline for {len(corpus.baselines)} tasks to {args.out}")
        else:
            print(payload)
        return 0

    # validate
    labels = None
    if args.fixtures:
        root = _expand(args.fixtures)
        traj = root / "trajectories"
        db = root / "runs.db"
        label_file = root / "labels.json"
        if not (traj.is_dir() and db.is_file() and label_file.is_file()):
            print(
                f"error: {root} is not a fixture corpus — it needs trajectories/, runs.db"
                " and labels.json. Run `validate` without --fixtures to build one.",
                file=sys.stderr,
            )
            return 2
        labels = json.loads(label_file.read_text(encoding="utf-8"))
        owned = False
    else:
        tmp = Path(tempfile.mkdtemp(prefix="step-conformance-fixtures-"))
        labels = synth_fixtures(tmp)
        traj, db = Path(labels["trajectory_dir"]), Path(labels["db"])
        owned = True

    before = _hash_inputs([db, *sorted(traj.glob("*.jsonl"))])
    result = validate_fixtures(
        trajectory_dir=traj,
        db_path=db,
        labels=labels,
        grace_seconds=args.grace_seconds,
        min_runs=args.min_runs,
        support=args.support,
    )
    after = _hash_inputs([db, *sorted(traj.glob("*.jsonl"))])

    print("# injected-fault validation (offline fixtures)")
    for entry in result["detected"]:
        print(
            f"  DETECTED {entry['run_id']} ({entry['incident']}): missing {entry['step']!r}"
            f" while status={entry['run_status']}"
        )
    for fault in result["missed"]:
        print(f"  MISSED   {fault['run_id']} ({fault['incident']}): {fault['expected_missing_step']!r}")
    for entry in result["benign_flagged"]:
        print(f"  FALSE    {entry['run_id']} status={entry['run_status']}: {entry['steps']}")
    print(f"  inputs_unchanged={before == after}")
    print(summary_line(result["counts"]))
    if args.json:
        payload = {k: v for k, v in result.items() if k != "corpus"}
        payload["counts"]["inputs_unchanged"] = before == after
        _expand(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if owned:
        import shutil

        shutil.rmtree(traj.parent, ignore_errors=True)
    counts = result["counts"]
    return 0 if counts["detected"] == counts["injected"] and counts["false_alarms"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
