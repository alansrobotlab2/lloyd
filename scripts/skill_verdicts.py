#!/usr/bin/env python3
"""skill_verdicts.py — a keyed ledger of skill-candidate verdicts (backlog #530).

The nightly skill pipeline adjudicates candidates and then forgets it. Every
disposition used to be written down — in a markdown review log, and in a `status:` line
in each candidate's frontmatter — and none of it was read back before the next run.
This module replaced the log with two JSONL surfaces: the live ledger
(`_pipeline/skills/reviews/verdicts.jsonl` under the data root) and its durable mirror
(`~/obsidian/memory/skill-verdicts/verdicts.jsonl`), and there is no markdown view of
either — nothing here writes one (#1795).
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

Eight invariants:

* **Append-only, latest-wins per key.** A verdict is never edited or deleted; a
  reopen is a new line. Rollback is asymmetric the way WikiSkill's is: a skill can
  be reverted, the record of why it was rejected never is.
* **A repair may never change what a verdict blocks (#1588).** The moved data root
  killed falsifiers without touching the decisions they grounded, and `repair` is the
  machine correcting its own bookkeeping, so it runs unattended. Its one safety line is
  terminality: a correction keeps the verdict, and a disposal of grounds that retention
  deleted may relabel a terminal verdict as terminal (`rejected_unverifiable`) but must
  never mint a block a non-terminal verdict never had.
* **`evidence_cmd` is required.** A prose verdict is an assertion the next run can
  only inherit. A re-executable check is what lets a later run *falsify* it, which is
  the weakness #525 records on the worker-evidence path.
* **A new line never weakens the row it supersedes (#736 clause 1).**
  `occurrences_at_decision` is the baseline the growth reopen measures against, and
  `reopen_reason` requires a baseline above 0 — so a correction line that stores 0
  permanently disarms the >10x reopen for that key while looking like a routine
  append. Appending is the only sanctioned way to fix a wrong phrase in a prior line,
  so the act of correcting prose was itself what disarmed the cap: 10 live keys read
  `occurrences_at_decision: 0` on 2026-09-15 for exactly that reason. An omitted
  `--occurrences` therefore carries the superseded row's count forward; an explicit
  `0` is still honoured, because `cmd_seed` passes one deliberately for a merged
  candidate whose units cannot be compared.
* **`evidence_cmd` has to observe something (#736 clauses 2-3).** Presence was the
  only requirement, so a command that runs and prints nothing satisfied it: `true`,
  `false`, `test $(…) -eq 0`, and a `grep -rl` for a string the target file does not
  contain are all silent on success. 12 of the ledger's 41 live keys printed nothing
  when every one was executed on 2026-09-13. `record` now runs the command once,
  refuses an append whose combined stdout+stderr is empty, and stores the first line
  it printed as `evidence_observed` — a required field becomes a required observation,
  and a later run sees the measurement that justified the decision beside the command
  that re-makes it.
* **A stored falsifier has to read the same way it will be re-read (#2103).** An
  `evidence_observed` of `0` is two different facts and nothing in the row tells them apart:
  the owner lost the section, or the grep was case-sensitive and the owner's own nightly
  rewrite re-cased the sentence — `grep -c` prints `0` and exits 1, and no rc is stored at all.
  So the classifier asks a case-sensitive literal grep the same question with `-i` added and
  reports `STRANDED_CASE` when that is the difference between absent and present, which
  `audit` publishes as one line per key plus its own tally and `check` names in its existing
  `EVIDENCE_CMD_*` shape; and `record` refuses to mint a new case-sensitive literal grep of a
  `skills/**/SKILL.md` at all, because that file is the one its owner's refresh edits under the
  verdict, and the 24 such falsifiers already in the ledger are history no refusal reaches.
* **A field the record names is a field the record has to carry (#2103).** `record` refuses a
  command that exits 0 and whose stored line holds a `name=` with nothing after it — the shape
  that put `echo 'newest_bucket= newest_rows= prev_rows= input_rows=2812'` into the durable
  ledger with its denominator real and the three numbers that make that run's window auditable
  simply absent, because all four of those words are field names and three had lost their
  value. #2052's rail reads the count, so it passed; `audit` re-executes the same half-empty
  line and reports the ledger clean over it.
* **The ledger exists in two trees (#772).** This store lives under `_pipeline/`,
  which `.gitignore` excludes from every repo on the box and no backup job reads, so
  one truncated append or one `rm -rf _pipeline` used to erase every decision with no
  way to re-derive the reasons — they are prose that lives nowhere else. `record`
  therefore appends the identical line to a durable copy under the vault root (the
  15-minute vault snapshotter covers it), seeding that copy from this file the first
  time it is written so no pre-existing verdict is missing, and copies beside it any
  script a stored `evidence_cmd` names that no repo already tracks. `check` reads the
  mirror when this file is gone and says it did, so a wiped ledger is an announced
  incident rather than `skipped_by_verdict: 0` — and when it is gone while the corpus
  still holds a candidate whose `status:` only this ledger mints, `check` prints
  `LEDGER_ABSENT` and exits non-zero (#736 clause 4), because a fresh install and a
  wiped decision history otherwise print the identical quiet night. Seeding and
  appending keep the copy a
  superset of the live file; neither keeps the *live* file honest, so `check` also
  compares the two latest-per-key tables every run and prints `MIRROR_MISSING`,
  `LEDGER_LOST`, `LEDGER_MIRROR_LAGGED` (the copy simply never got the newest append) or
  `LEDGER_MIRROR_CONFLICT` (the two trees hold *different verdicts*) per diverging key,
  then one tally naming both classes against `keys_compared:` (`check_against_mirror`).
  Without that, a copy that quietly fell behind — or a live ledger that quietly lost one
  line while still existing — prints precisely the counts two healthy trees would. The lag
  and the disagreement are separate labels because merging them cost a day: on
  2026-09-28 the 79 lines `check` printed were 54 of the first and 25 of the second, all
  one shape, so the case the detector was built for was invisible inside it (#1717).
  `sync` is the bring-up that closes a gap once one exists, and it refuses to write a
  ledger that has no copy of its own through the production one.

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
    skill_verdicts.py audit                       # re-execute every stored check (#1533)
    skill_verdicts.py repair                      # append the corrections the moved data
                                                  # root broke, and name the ones only a
                                                  # re-derivation can fix (#1588)
    skill_verdicts.py sync                        # bring the durable copy up to the live
                                                  # ledger; idempotent, refuses a --store
                                                  # with no copy of its own (#1717)
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.paths import LIVE_CHECKOUT, PIPELINE_DIR  # noqa: E402

# Append-only, under the data root's `_pipeline/`. This JSONL and the durable copy
# below are the ledger's only two surfaces; there is no markdown view beside them.
DEFAULT_STORE = PIPELINE_DIR / "skills" / "reviews" / "verdicts.jsonl"
DEFAULT_CANDIDATES = PIPELINE_DIR / "skills" / "candidates"

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
#
# The last three are the values the runbooks themselves tell a run to write, which is
# what made their absence a defect and not a spare corner of the vocabulary:
# `nightly-skill-consolidation` Phase 5.1 routes `consolidated` and `noise` to this
# ledger precisely because "the frontmatter status is per-file and gets regenerated with
# the next dated snapshot; the ledger is what survives", and `trajectory-skill-mining`
# step 3.5 records `reviewed_authored` once a skill has been authored from the pattern.
# `record_verdict` validates only non-emptiness, so it accepted all three, appended the
# line, and `is_terminal` then honoured nothing: 5 `reviewed_authored` rows and 1 `noise`
# row bound no key (#830 triage, 2026-09-18), while the 248 of 1,132 keys whose `noise`
# lived only in frontmatter were re-emitted `pending_review` by the next mining run. A
# decision that binds nothing is worse than no decision, because the next run reads the
# ledger and sees a verdict already recorded.
#
# Stickiness is what the reopen rules below exist for. An authored pattern is *more*
# disposed than a `reviewed_no_skill` one, and a key is coarse (`Bash/timeout`, not one
# signature), so making these terminal must not bury a genuinely new failure mode that
# later shares the key — the 60-day expiry and the >10x growth trigger are what lift it,
# and they are unchanged. `proposed` stays non-terminal by design: a patch below the
# auto-apply threshold has to keep accumulating evidence (Phase 5.1).
TERMINAL_VERDICTS = frozenset({
    "rejected_false_positive",
    "reviewed_no_skill",
    "rejected_unverifiable",
    "rejected_artifact_class",
    "archived_content",
    "reviewed_authored",
    "noise",
    "consolidated",
})

# A terminal verdict that has aged out or whose evidence has grown >10x is reopened
# rather than trusted. 60 days is roughly two full re-emergence cycles on this board.
REOPEN_AFTER_DAYS = 60
REOPEN_OCCURRENCE_GROWTH = 10.0

# The status the miner writes instead of `pending_review` when a key is terminal.
SUPERSEDED_STATUS = "superseded_by_verdict"

# A candidate `status:` that only this ledger can mint: the original terminal verdicts,
# plus `superseded_by_verdict`, which `mine-trajectories.py` writes *because* a verdict
# blocked the key. Finding one in the corpus while the store is absent is therefore proof
# that a ledger existed and is gone, which is the difference between an incident and a
# fresh install (#736 clause 4).
#
# Deliberately a literal, not `TERMINAL_VERDICTS | {SUPERSEDED_STATUS}`: #830 widened that
# set with three values a runbook writes straight into a candidate's own frontmatter about
# its own content — `status: noise` on 930 of the live corpus's files, 3 files carrying
# `status: reviewed_authored` — and the great majority of those never had a ledger row to
# lose. Counting them as ledger-minted would turn every scratch run over that corpus into
# a false `LEDGER_ABSENT` alarm, and an alarm that always fires is the alarm that gets
# skipped. A future terminal verdict that is likewise hand-written in frontmatter belongs
# here and nowhere else in this file; `tests/test_skill_verdicts.py` pins that widening
# `TERMINAL_VERDICTS` does not silently widen this set.
LEDGER_MINTED_STATUSES = frozenset({
    "rejected_false_positive",
    "reviewed_no_skill",
    "rejected_unverifiable",
    "rejected_artifact_class",
    "archived_content",
})
MINTED_BY_LEDGER = LEDGER_MINTED_STATUSES | {SUPERSEDED_STATUS}

# Frontmatter keys, in order. `pattern_key` is the join key everywhere.
FIELDS = (
    "pattern_key",
    "verdict",
    "reason",
    "evidence_cmd",
    "evidence_observed",
    "occurrences_at_decision",
    "decided_at",
    "decided_by",
    "source_candidate",
)

_STATUS_RE = re.compile(r"^status:\s*(.+?)\s*$", re.MULTILINE)
_PATTERN_RE = re.compile(r"^pattern:\s*(\S+)", re.MULTILINE)
_OCCURRENCES_RE = re.compile(r"^occurrences:\s*(\d+)", re.MULTILINE)
# Emitted by `mine-trajectories.py` since #515: how many mined signature buckets one
# candidate file stands for. Absent means one bucket (a pre-#515 file, or a success or
# sequence candidate), which is the behaviour that predates the field.
_SIGNATURE_BUCKETS_RE = re.compile(r"^signature_buckets:\s*(\d+)", re.MULTILINE)


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


#: Prefix of the line a single-tree write prints (#1717). Named as a constant because the
#: nightly greps it: 79 rows written on 2026-09-27 landed in the live ledger alone and
#: nothing on any surface said so, so the first requirement is a string to grep for.
MIRROR_NOT_WRITTEN = "MIRROR_NOT_WRITTEN"


def mirror_skip_notice(live: Path) -> str:
    """The line naming the durable copy a write to `live` did NOT reach (#1717 clause 2).

    `mirror_for`'s own policy is record-anyway-and-say-so, never refuse: the ledger is the
    live artifact and an unwritable vault costs the second copy, not the decision. That
    policy is only honest if the *saying* half happens, and until #1717 it did not — the
    2026-09-27 repair pass appended 79 rows through this branch in silence, so the vault
    half of #772 (the reasons and falsifiers "that live nowhere else") was the half that
    went missing, and `check` spent its divergence alarm on the resulting lag.
    """
    return (f"{MIRROR_NOT_WRITTEN} {live} :: the durable copy {DEFAULT_MIRROR} was NOT "
            f"written, because this ledger is not the default {DEFAULT_STORE} and "
            f"$SKILL_VERDICTS_MIRROR is unset (`_mirror_target` refuses to push an "
            "off-default ledger through the production vault copy). The live row is "
            "recorded; it exists in one tree only (#1717)")


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


#: The two labels for "both trees hold this key and their latest rows differ" (#1717
#: clause 3). They were one label until #1717, and that merger is why the real ones went
#: unseen: on 2026-09-28 the live ledger and its vault copy printed 79 identically-shaped
#: `LEDGER_MIRROR_CONFLICT` lines, 54 of which were the copy merely keeping behind (same
#: verdict, older stamp, the append never reached it) and 25 of which were the two trees
#: holding *different decisions* for one key — the case the detector exists to find,
#: byte-indistinguishable from the other 54.
LAGGED = "LAGGED"
DISAGREEING = "DISAGREEING"


def divergence_class(live_row: dict, durable_row: dict) -> str:
    """`LAGGED` or `DISAGREEING` for one key both trees hold, judged on the verdict first.

    Same verdict with the live row stamped *newer* is plain lag: `record` appends to both
    trees, so a copy that never got the append is behind and nothing was decided
    differently. Everything else that differs is `DISAGREEING` and gets the loud label — a
    different verdict, a durable copy stamped *newer* than live (which no append-only
    writer produces), or a pair whose stamps cannot be read, since a row whose date is
    unparseable cannot be *proved* to be lag. The asymmetry is the safety of the split:
    `LAGGED` is the class `sync` copies unattended, so only what the evidence supports may
    land there, and an unexplained difference never does.
    """
    if live_row.get("verdict") != durable_row.get("verdict"):
        return DISAGREEING
    live_at = _parse_ts(live_row.get("decided_at") or "")
    durable_at = _parse_ts(durable_row.get("decided_at") or "")
    if live_at is not None and durable_at is not None and live_at > durable_at:
        return LAGGED
    return DISAGREEING


def divergence_report(live_table: dict[str, dict], durable_table: dict[str, dict],
                      durable: Path) -> dict:
    """Every key where the two trees diverge, as lines *and* the counts that size them.

    `keys_compared` is the denominator (#1717 clause 4): the number of keys both trees were
    asked about. `LAGGED`, `DISAGREEING`, `missing` and `lost` are the classes. Counts next
    to the lines are the point — 79 look-alike lines cannot be sized by a reader and hid
    the 25 lines inside them that meant something else, and a wall of them stays hidden no
    matter who reads it, while `lagged: 54 disagreeing: 25` cannot.

    Four shapes, because the ledger is append-only latest-wins and a divergence has four
    causes: the durable copy never received a line (`MIRROR_MISSING` — a writer appended to
    the live file without going through `record`, which is the only two-tree writer), the
    live file *lost* a line the copy still holds (`LEDGER_LOST` — the file's only documented
    writer appends, but that is a convention and not an enforcement, and
    `resolve_verdict_source` cannot see the gap while the file still exists), or both trees
    hold the key with different content, split here into `LEDGER_MIRROR_LAGGED` (same
    verdict, the copy simply never got the newest append — a `sync` fixes it) and
    `LEDGER_MIRROR_DISAGREEING` (a different decision, or a difference that cannot be
    explained as lag, which one side has to justify). An empty `lines` is the only healthy
    answer, and it is also the case that makes this worth printing: two silently-diverging
    copies are exactly the state #772 describes, where a restore would look like it worked.
    """
    lines: list[str] = []
    counts = {"keys_compared": len(set(live_table) | set(durable_table)),
              LAGGED: 0, DISAGREEING: 0, "missing": 0, "lost": 0}
    for key in sorted(set(live_table) | set(durable_table)):
        live_row, durable_row = live_table.get(key), durable_table.get(key)
        if durable_row is None:
            counts["missing"] += 1
            lines.append(
                f"MIRROR_MISSING {key} :: {live_row.get('verdict')} decided "
                f"{live_row.get('decided_at')} is in the live ledger but not in the "
                f"durable copy {durable}")
        elif live_row is None:
            counts["lost"] += 1
            lines.append(
                f"LEDGER_LOST {key} :: {durable_row.get('verdict')} decided "
                f"{durable_row.get('decided_at')} is in the durable copy {durable} "
                f"but the live ledger no longer holds it")
        elif live_row != durable_row:
            kind = divergence_class(live_row, durable_row)
            counts[kind] += 1
            if kind == LAGGED:
                lines.append(
                    f"LEDGER_MIRROR_{LAGGED} {key} :: {live_row.get('verdict')} in both "
                    f"trees, durable copy is behind: live decided "
                    f"{live_row.get('decided_at')}, durable "
                    f"{durable_row.get('decided_at')} (`sync` copies it)")
            else:
                # The `LEDGER_MIRROR_CONFLICT` token is kept: the nightly and this item's
                # own acceptance grep it, and re-naming it would make that grep go quiet
                # for the wrong reason. What changes is that it now fires on this class
                # only, so its count means *disagreements* rather than *any difference*.
                lines.append(
                    f"LEDGER_MIRROR_CONFLICT {key} :: {DISAGREEING} "
                    f"live={live_row.get('verdict')} "
                    f"durable={durable_row.get('verdict')} decided "
                    f"{live_row.get('decided_at')}/{durable_row.get('decided_at')}")
    counts["lines"] = lines
    counts["durable"] = durable
    return counts


def divergence_lines(live_table: dict[str, dict], durable_table: dict[str, dict],
                     durable: Path) -> list[str]:
    """`divergence_report`'s lines alone, for a caller that only prints them."""
    return divergence_report(live_table, durable_table, durable)["lines"]


def mirror_tally_line(report: dict) -> str:
    """The one-line tally that puts the two classes beside their denominator (#1717 clause 4).

    Shared by `check` and `sync` so the two surfaces cannot state the same comparison in two
    shapes and have one of them believed. The class counts come *before* the paths because
    the numbers are what a reader sizes the finding by: on 2026-09-28 `check` emitted 79
    lines under one label and a reader could not tell whether that was one problem or 79,
    which is how 25 verdict disagreements sat inside it unnoticed for a day.
    """
    return (f"ledger_mirror: keys_compared: {report['keys_compared']}  "
            f"lagged: {report[LAGGED]}  disagreeing: {report[DISAGREEING]}  "
            f"missing: {report['missing']}  lost: {report['lost']}  "
            f"durable: {report['durable']}")


def check_against_mirror_report(live: Path, live_table: dict[str, dict]) -> dict | None:
    """`divergence_report` for this ledger's own copy, or None when there is nothing to compare.

    No mirror for this ledger, or none on disk yet, is not a divergence — it is the pre-#772
    single-copy state, and `check` would otherwise alarm on every scratch `--store`. None,
    as distinct from an empty report, is what lets `check` say *not compared* rather than
    printing a denominator of `keys_compared: 0` for a comparison it never ran.
    """
    durable = _mirror_target(live)
    if durable is None or durable == live or not durable.is_file():
        return None
    return divergence_report(live_table, load_verdicts(durable), durable)


def check_against_mirror(live: Path, live_table: dict[str, dict]) -> list[str]:
    """The live ledger's own copy, read as a second opinion rather than as a fallback.

    `resolve_verdict_source` answers from the mirror only when the live file is *entirely*
    absent, so a one-line disappearance prints the same counts as if nothing happened
    (#772). This is the companion that has no fallback in it: it compares the two trees on
    every run and returns what to print. `check` calls the `_report` variant for the counts
    as well; this one stays the lines-only accessor for a caller that prints them.
    """
    report = check_against_mirror_report(live, live_table)
    return report["lines"] if report else []


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


def carried_forward_occurrences(store: Path | str | None, pattern_key: str) -> int:
    """The count the row about to be appended would otherwise lose (#736 clause 1).

    Read through `resolve_verdict_source`, not the live file alone: after a wipe the row
    this append supersedes exists only in the durable copy, and a baseline recovered from
    there is still the baseline the reopen rule needs. No row for the key — a genuinely
    new pattern — is the one case where 0 is the honest answer, because there is nothing
    to disarm.
    """
    prior = load_verdicts(resolve_verdict_source(store)[0]).get(pattern_key.strip())
    if not prior:
        return 0
    try:
        return int(prior.get("occurrences_at_decision") or 0)
    except (TypeError, ValueError):
        return 0


def record_verdict(
    store: Path | str | None = None,
    pattern_key: str = "",
    verdict: str = "",
    reason: str = "",
    evidence_cmd: str = "",
    occurrences: int | None = None,
    decided_by: str = "agent",
    source_candidate: str = "",
    decided_at: str | None = None,
    *,
    require_input_rows: bool = True,
    require_body_scope: bool = True,
) -> dict:
    """Append one decision. Never mutates an earlier line.

    `reason` and `evidence_cmd` are required: without the second, a verdict is an
    assertion a later run can only inherit. The command is *executed* before anything is
    written and an append whose combined output is empty is refused (#736 clause 2), and
    the first line it printed is stored as `evidence_observed` (clause 3). Emptiness is not
    runnability: an append whose command `evidence_cmd_status` — the classifier `check` and
    `audit` report from — calls UNRUNNABLE is refused too, because a command that never
    started still prints its own failure and quotes it as a measurement (#1586). The
    classifier's fourth shape, a check that outlives its timeout, never reaches that test:
    `run_evidence` returns empty output for it, so the emptiness guard above is what
    refuses it.

    `occurrences` is `None` when the caller did not name a count, which is not the same
    fact as `0`: an omitted count is carried forward from the row this line supersedes,
    so a correction appended without it cannot disarm the growth reopen (#736 clause 1).
    An explicit `0` is stored as given — `cmd_seed` passes one for a merged candidate
    whose units are not comparable with a one-bucket baseline.

    The third refusal is the undeclared denominator (#2052): a command that ran, exited 0,
    and printed no `input_rows=<N>` is refused too, because 0 of the live ledger's 109
    latest-wins keys declared the field on 2026-10-02, which left #2048's zero-denominator
    rail with nothing to read on any key and `audit` printing `undeclared 109` as a figure
    it could only publish, never act on. Refusing at the mint is what makes the figure
    fall, and it is keyed on rc 0 for the same reason EMPTY_INPUT is: a falsifier exiting
    nonzero beside a printed count is the ledger answering no, not a vacuous success.
    `require_input_rows=False` is for the mint paths where this module wrote the command
    text itself — `cmd_repair`'s root-move substitution, its disposal tombstone, and
    `cmd_seed`'s harvest grep — so nobody is refused over a field no author was ever asked
    to write; those rows stay `undeclared` in `audit`, which is where the mandate keeps
    counting them. It is keyword-only and unreachable from the CLI, so `record` cannot
    talk itself out of the rule it is being refused by.

    Two refusals joined on 2026-10-03, both on the same evidence that the four above read
    as healthy (#2103). A command that exits 0 and whose stored first line carries a `name=`
    field with nothing after it is refused, naming the field — the shape that let
    `echo 'newest_bucket= newest_rows= prev_rows= input_rows=2812'` into the durable record
    with a real denominator and no numbers, because each of those four words is a field name
    and three of them lost their value. And a command that greps a `skills/<name>/SKILL.md`
    path case-sensitively is refused before it runs, naming `grep -i` as the fix: that file is
    the one its owner's own nightly rewrite edits, so a falsifier minted against one casing
    re-executes as `0` the night the casing moves and the ledger cannot tell that from the
    owner deleting the section. Neither is keyed on `require_input_rows`: a mint that is
    exempt from the denominator convention is not exempt from writing an observation that
    means something.

    The line lands in two trees (#772): the ledger, then the durable copy, which is
    seeded from the ledger first so a verdict recorded before the mirror existed is in
    it too. A script the new `evidence_cmd` names is copied beside the mirror.
    Mirroring never refuses the record: the ledger is the live artifact, and a vault
    that cannot be written must not lose a decision. It never goes *quiet* either
    (#1717): when there is no durable copy for this ledger, the skip is named on stderr
    by `mirror_skip_notice` before the function returns, so a pass that wrote 79 rows
    into one tree cannot report a clean night. `repair_verdicts` carries the same fact in
    its returned tally as `mirror_skipped`, which is what `cmd_repair` prints and counts.
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
    if occurrences is None:
        occurrences = carried_forward_occurrences(store, pattern_key)
    # The casing mandate is decided from the command text alone, before it is executed,
    # because the defect is the shape of the read and not its answer: a case-sensitive
    # literal grep of an installed `SKILL.md` is born falsifiable by somebody else's nightly
    # whether or not it matches tonight. Keyed on nothing else — not `require_input_rows`,
    # not the verdict, not the rc — because every flag that could relax it is a way for a
    # mint to talk itself out of the rule it exists to enforce.
    if (skill_file := case_sensitive_skill_md_grep(evidence_cmd)) is not None:
        raise ValueError(
            f"evidence_cmd greps {skill_file} case-sensitively, and an owner-coverage "
            "falsifier cannot survive that: the file it names is the one its owner's own "
            "nightly rewrite edits. #2103 measured the invalidation end to end — "
            "`seq-2-edit-err-read` was minted 2026-09-12 with `grep -c \"one mechanism, two "
            "symptoms\"` and re-executed 2026-10-03 as `0`, because the installed skill's "
            "09-29 refresh made the sentence read `One mechanism, two symptoms.` A stored `0` "
            "is then indistinguishable from the owner DELETING the section, and the rc is no "
            "help: `grep -c` prints `0` and exits 1, no rc is stored in the row, and `audit` "
            "re-runs the same case-sensitive read and reports the key healthy. Read it "
            "case-insensitively — `grep -ci` — so the falsifier measures the coverage and not "
            "the casing. Rows stored before this mandate stay as the recorded history they "
            "are: it is a mint-time rule, not a rewrite of the ledger, and a stored "
            "case-sensitive grep is named by `audit` and `check` the night its own re-read "
            "comes back absent while `-i` matches (#2103)"
        )
    if require_body_scope and (leaking := candidate_body_defect(evidence_cmd)) is not None:
        # The other half of #2103's lesson, on the other file of the pair. A stored falsifier's
        # operand is a LIVE file that outlives its snapshot, and a candidate snapshot's front
        # matter is written by the ledger rather than by the traffic: the mint re-injects the
        # previous decision's own `verdict_reason` into every later dated snapshot for that key
        # (`status_block` in `mine-trajectories.py`). A count over the whole file therefore
        # counts the prose that justified the last verdict, and can only rise — #2166 measured
        # whole=3 body=2 on `candidate-edit-logic-20261004.md`, the third match that night's own
        # reason, at line 17. Refusing the shape here is the point rather than a courtesy: four
        # keys were hand-repaired in the ledger tonight, which proves a run CAN write the scoped
        # form, and a convention for doing it is re-made every night by whichever run notices.
        if leaking[0] == "whole_file":
            raise ValueError(
                f"evidence_cmd counts a candidate file with a whole-file grep: `{leaking[1]}`. "
                "A candidate snapshot's front matter is written by the ledger, not by the "
                "traffic — the mint re-injects the previous decision's own `verdict_reason` into "
                "it — so this count includes the prose that justified the last verdict and rises "
                "every time the key is re-recorded over itself. Body-scope it: strip the front "
                "matter once (`b=$(awk '/^---/{n++; next} n>=2' \"$f\")`) and count that "
                "(`printf '%s\\n' \"$b\" | grep -ci '<pattern>'`), or keep the read inside a "
                "front-matter field with `-m1` and an anchored pattern like `^sessions:`. A "
                "count is taken over the record set, never over narrative prose.")
        raise ValueError(
            f"evidence_cmd defines a front-matter strip and never reads it: {leaking[1]}. Every "
            "field it then emits is a front-matter field or an owner-skill grep, so the key ends "
            "up with no candidate-body falsifier at all while its own first clause tells the "
            "next reader its counts are body-scoped — `Bash/timeout` in tonight's ledger is that "
            "key, and its verbatim run still prints only `owner_pattern3=… owner_timeout_words=… "
            "owner_literal_phrase=…`, no `body_*` field. Either consume the stripped copy "
            "(`body_<field>=$(printf '%s\\n' \"$b\" | grep -ci '<pattern>')`) or drop the strip "
            "and stop claiming the scoping.")
    rc, observed = run_evidence(evidence_cmd)
    if not observed:
        raise ValueError(
            "evidence_cmd observed nothing: it ran (exit code "
            f"{rc}) and printed neither stdout nor stderr, so it cannot falsify this "
            f"verdict, only decorate it — {evidence_cmd!r}. Print the measurement the "
            "decision rests on (a `grep -c`, an `echo` of the counts you compared); "
            "`true`, `false` and `test $(…) -eq 0` are silent on success, which is why "
            "12 of the ledger's 41 live keys were unfalsifiable prose with a `cmd` key "
            "attached (#736 clause 2)"
        )
    # Output is not runnability. `run_evidence` quotes whatever the command printed, and a
    # command that never started prints too — its own failure — so the emptiness test above
    # accepted three shapes that cannot falsify anything: an absent file, a command bash
    # cannot parse, and a not-found binary, each recorded with `evidence_observed` holding
    # the crash and then honoured by `check` for its full 60 days while its own
    # re-execution prints EVIDENCE_CMD_UNRUNNABLE (#1586: the write path was minting the
    # exact ledger state #1533 measures with `audit`). Refuse precisely what the ledger's
    # own readers already report, by asking that classifier rather than inventing a second
    # rule — which costs a second execution of the check, because the classifier needs the
    # exit code and the whole stderr and `run_evidence` keeps only one line of one stream.
    # That is bounded and cheap next to what it prevents: the live ledger's 170 rows fall on
    # 15 dates, busiest night 21, and every one of them is read by every later night for its
    # full 60 days. `_run_stored_check` rather than its two-value wrapper `evidence_cmd_status`
    # because the third value is the denominator parse, and the two rails below must decide
    # from one read of one stream: an rc-0 command that prints its findings header before
    # `input_rows=0` would be recorded by a mint reading only the first line while `audit`,
    # scanning the whole stdout, would report the zero it was shown (#2052 clause 3 — one
    # measurement, both sides).
    status_rc, status_detail, declared = _run_stored_check({"evidence_cmd": evidence_cmd})
    if status_rc == UNRUNNABLE:
        raise ValueError(
            "evidence_cmd is UNRUNNABLE — it cannot run at all, so it falsifies nothing "
            f"and this verdict would be born uncheckable: {status_detail!r} — "
            f"{evidence_cmd!r}. `check` and `audit` report the same UNRUNNABLE for it "
            "(absent file, exit 127, or a command bash refuses to parse), which is a "
            "different fact from a check that runs and answers no: a falsifier exiting 1 "
            "beside a printed count is a verdict this function records. When the "
            "measurement *is* the absence of a file, print the absence so a later run can "
            f"re-check it — `test ! -f <path> && echo absent` — instead of letting grep's "
            "error stand in for the observation (#1586)"
        )
    if status_rc == EMPTY_INPUT:
        # The other born-uncheckable shape, and the one no exit code reports: the command
        # ran, exited 0, and declared that it read no input, so its output is a statement
        # about the empty set and no future run can falsify the verdict it grounds. Refused
        # at the mint, not repaired afterwards, because afterwards is `audit`, which sees a
        # zero-denominator key as a *finding* on a ledger that already made its decision —
        # and #1588's repair pass, keyed on UNRUNNABLE, cannot reach it at all (#2048).
        raise ValueError(
            f"evidence_cmd declares an empty input — it printed {status_detail!r} over "
            f"zero input rows, so it measured nothing and falsifies nothing: "
            f"{evidence_cmd!r}. Point it at the input it is meant to read, or record the "
            "verdict without a claim that a check backs it. Printing no "
            f"{INPUT_ROWS_FIELD} field at all is now refused too, one guard below — the "
            "silence and the declared zero are the same unusable evidence (#2052)"
        )
    if require_input_rows and status_rc == 0 and declared is None:
        # The silence, and the reason #2048's rail could not do its job: a declared zero is a
        # claim `audit` can tally and a repair can act on, while an absent field is nothing to
        # read. Measured on the live ledger 2026-10-02: 109 latest-wins keys, 0 of them
        # declaring `input_rows=` in either `evidence_cmd` or `evidence_observed`, so
        # `audit` printed `denominators: empty_input 0 undeclared 109` with every one of those
        # 109 re-executing at rc 0 — a rail over a denominator nobody publishes. Refuse the
        # silence at the mint, where a command can still be written differently, and leave the
        # 109 legacy rows alone: re-judging them is `audit`'s non-fault denominator, not a
        # refusal it can retroactively issue (#2052 clause 4). Keyed on rc 0 for the same
        # reason EMPTY_INPUT is — a falsifier exiting 1 beside its count is the ledger working,
        # and requiring a declaration from the answer "no" would teach `check` to distrust a
        # real falsification.
        raise ValueError(
            f"evidence_cmd declares no input count — it ran and exited 0 without printing an "
            f"{INPUT_ROWS_FIELD}=<N> field, so no later run can tell a measurement taken over "
            f"a real corpus from the same numbers printed over none: {evidence_cmd!r}. Print "
            'the count beside the values it grounds — `echo "matched=3 '
            f'{INPUT_ROWS_FIELD}=$(ls -1 "$DIR" | wc -l)"` — and an absence probe prints the '
            f"inputs it inspected, not the empty listing, so its zero stays the visible claim "
            f"EMPTY_INPUT refuses rather than an invisible one. `audit` has been publishing "
            f"`undeclared` for a field nothing in the code or the convention asked for: 0 of "
            f"the live ledger's 109 latest-wins keys declared {INPUT_ROWS_FIELD}= on "
            f"2026-10-02 (#2052)"
        )
    if rc == 0 and (blank := empty_valued_fields(observed)):
        # The half-empty line, and the reason every rail above let it through: the command
        # ran, exited 0, printed a line, and declared a real denominator — all four rails read
        # that as healthy evidence, while three of the line's fields carried their names and no
        # values. Measured on the live ledger the night #2103 was filed: the run-level key
        # `run:2026-10-03-nightly-mining` was minted with `newest_bucket= newest_rows=
        # prev_rows= input_rows=2812`, the values lost to an unquoted multi-word shell variable
        # under the mining writer's `sh`, and because `input_rows=2812` matched the miner's own
        # printed `Window: 2812 row(s)` exactly, #2052's denominator rail passed it and the row
        # was accepted — the three numbers that make that run's window auditable were simply
        # absent from the durable record, and `audit`, which re-executes the same half-empty
        # line, reported the ledger clean over it. Four such rows were already in the ledger
        # from 2026-09-27 (`seq-2-bash-fs-edit` and three others, each printing `snapshot= …
        # sessions= `), so the shape is the writer's, not a one-off.
        #
        # Keyed on rc 0 for the reason the rails above are: a falsifier that exits nonzero
        # beside a blank field is answering no, which is the ledger working, and the rc here is
        # `run_evidence`'s rather than the classifier's because the pair being tested is the
        # pair about to be written — this execution's exit status and this execution's first
        # line, one measurement, not two executions that may print different numbers. Mint-only
        # like #2052's declaration rule: the five rows already in the ledger are history that
        # no refusal can retroactively issue, and `audit` is the surface that reads them.
        raise ValueError(
            f"evidence_cmd exited 0 and printed a field with no value — "
            f"{', '.join(name + '=' for name in blank)} — so the durable record would carry "
            f"the field names and lose the numbers beside them: {observed!r} over "
            f"{evidence_cmd!r}. Quote what you interpolate (`echo \"newest_bucket=$bucket\"`) "
            "or print the measurement as a count; a field that prints empty is a field the "
            "next run cannot re-check, and it is not the same defect as a missing field or a "
            "declared zero, both of which are refused above (#2103)"
        )
    row = {
        "pattern_key": pattern_key,
        "verdict": verdict,
        "reason": " ".join(reason.split()),
        "evidence_cmd": evidence_cmd,
        "evidence_observed": observed,
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
    # One and the same file under two names is not a second tree: appending twice to it
    # would put the newest verdict on disk twice and leave every later reader counting a
    # duplicate, which is what `check_against_mirror`'s `durable == live` guard already
    # declines to call a divergence.
    same_file = durable is not None and Path(durable).resolve() == path.resolve()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
    if durable is not None and not same_file:
        with durable.open("a", encoding="utf-8") as fh:
            fh.write(line)
    elif durable is None:
        # #1717: the half that used to be missing. `mirror_for` returning None is a
        # legitimate state — an off-default `--store` with no `$SKILL_VERDICTS_MIRROR` must
        # not push scratch rows through the production vault copy — and the policy is to
        # record anyway. But the 2026-09-27 repair pass took this branch for all 79 of its
        # appends and printed nothing, so the reasons and falsifiers #772 keeps a second
        # copy *for* ended up in one tree, and `check` spent its divergence alarm on the
        # lag. Saying so costs one line and is what makes the silence unforgivable again.
        print(mirror_skip_notice(path), file=sys.stderr)
    if durable is not None and not same_file:
        for script in named_scripts(evidence_cmd):
            _mirror_script(script, durable.parent)
    return row


def read_candidate(file: Path) -> tuple[str, str, int, int]:
    """(pattern_key, status_token, occurrences, signature_buckets) from one candidate.

    `signature_buckets` is 1 for a candidate that stands for one mined pattern: a
    pre-#515 file, a success pattern, or an error key with a single signature behind
    it. It is >1 only for a merged error emission unit, whose `occurrences:` is the sum
    over every bucket behind the key rather than one pattern's count — which is why
    `scan_candidates` treats the two cases differently.
    """
    text = Path(file).read_text(encoding="utf-8", errors="replace")
    # Frontmatter only: `status:` also appears in mined body examples.
    head = text.split("---", 2)[1] if text.startswith("---") else text[:2000]
    pattern = _PATTERN_RE.search(head)
    status = _STATUS_RE.search(head)
    occ = _OCCURRENCES_RE.search(head)
    buckets = _SIGNATURE_BUCKETS_RE.search(head)
    status_token = status.group(1).split()[0].strip("_*` ") if status else ""
    return (
        pattern.group(1) if pattern else "",
        status_token,
        int(occ.group(1)) if occ else 0,
        max(1, int(buckets.group(1)) if buckets else 1),
    )


# `bash`'s own "command not found" status, reused as the answer "this stored check
# cannot be executed at all" — as opposed to rc 1, which is the check running and
# reporting its assertion false, which is a falsification and not a defect.
UNRUNNABLE = 127


#: An `evidence_observed` value is a quotation, not a log. The first line is what
#: identifies the measurement; a falsifier that prints a 200-row diff would bury the
#: ledger line it is quoted in.
EVIDENCE_OBSERVED_MAX = 200

#: The fifth answer a stored falsifier gives, and the only one no exit code carries: the
#: command ran, exited 0, and declared that it read nothing — `input_rows=0` on its own
#: stdout. It is neither of the two states a reader already knows: not rc 0 (a healthy
#: observation — #2048's executed proof prints `0` over a directory that does not exist
#: and `audit` booked it as healthy), and not `UNRUNNABLE` (the command ran, and
#: `cmd_repair` must not be handed it, since #1588's repair pass is driven by UNRUNNABLE
#: keys and cannot reach this class even in principle). It is a string and not an int
#: precisely so no tally keyed on an exit status can absorb it: `audit`'s `unrunnable:`
#: figure is the nightly's published health number, and folding a zero denominator into
#: it would report the blind spot as the thing it already measures. Compare it with
#: `==` against this name, never by truthiness or ordering.
EMPTY_INPUT = "empty_input"

#: The sixth answer a stored falsifier gives, and like `EMPTY_INPUT` no exit code carries
#: it: the command ran, answered "absent", and would answer "present" if it read
#: case-insensitively. A verdict that rests on it is not wrong about the corpus — it is
#: measuring its own grep. #2103's witness: the key `seq-2-edit-err-read` carried
#: `grep -c "one mechanism, two symptoms" …/file-mutation-safety/SKILL.md`, minted 2026-09-12
#: when the sentence existed with that casing, and re-executed 2026-10-03 as `0` because the
#: owner skill's own nightly rewrite made the sentence read `One mechanism, two symptoms.` —
#: 24 of the ledger's 27 latest-wins falsifiers that name a `SKILL.md` read it
#: case-sensitively that night. It is a string and not an int for the same reason
#: `EMPTY_INPUT` is, so no
#: tally keyed on an exit status can absorb it, and `audit`'s published `unrunnable:` figure
#: stays the count of checks that cannot execute. Compare with `==` against this name.
STRANDED_CASE = "stranded_case"

#: The field a falsifier declares its input count in, in the `name=value` shape Phase 0.6
#: of `nightly-skill-consolidation` already asks a recorded command's output for. stdout
#: only, because that is where a measurement goes — `run_evidence` reads it first too —
#: and a grep *pattern* containing the field name must not decide the state. The leading
#: boundary keeps `total_input_rows=7` from answering for `input_rows`; first match wins,
#: so a command that prints its denominator before its findings declares the one it means.
INPUT_ROWS_FIELD = "input_rows"
_INPUT_ROWS_RE = re.compile(r"(?:^|[\s,;:/(=])" + INPUT_ROWS_FIELD + r"\s*=\s*(\d+)")


def declared_denominator(stdout: str) -> int | None:
    """What a command's stdout says about its own input: the count, or None for no claim.

    None is not zero. A command that declares nothing may well have read plenty — 109 of
    the live ledger's latest-wins keys print a bare count tonight and `audit` still counts
    them `undeclared`, a non-fault denominator, because those rows are history and a later
    read cannot refuse a decision already made. What changed on 2026-10-02 is the mint: the
    ruling on the authoring convention was made, and `record_verdict` refuses to create a
    new one of them (#2052), so the figure can only fall as keys get re-recorded. Zero is a
    claim: the command saw no input, so whatever it printed next describes an empty set.
    """
    found = _INPUT_ROWS_RE.search(stdout or "")
    return int(found.group(1)) if found else None


#: The only `grep` flag letters the recogniser below accepts, because every one of them
#: takes no argument: a token made solely of these letters is unambiguously a flag cluster,
#: so the token after the cluster is the pattern and the tokens after that are files. One
#: argument-taking flag (`-f FILE`, `-m N`, `-e PATTERN`, `-A N`) would make the parser guess
#: which operand is the pattern, and a wrong guess here re-cases the wrong string.
_GREP_ARG_FREE_FLAGS = "bchilnoqrsvwx"

#: A `grep` pattern is a literal string only if it holds none of these. A pattern with a
#: metacharacter is a regular expression, and `grep -i '^status:'` matches lines no literal
#: read does — so the re-read would be a different measurement, not the same one case-folded.
_GREP_META_RE = re.compile(r"[\.^$*+?()\[\]{}|\\]")

#: The shell structures that make a command more than one `grep`. The live ledger's 27 keys
#: that name a `SKILL.md` include 8 of the shape `S=…/SKILL.md; printf 'a=%s b=%s' "$(grep
#: -c 'lit' "$S")"` — a variable assignment and a command substitution whose output is one
#: field among several. Adding `-i` inside one of those is a rewrite of a command this module
#: cannot read, so those keys are left to their own readings.
_SHELL_COMPOSITION_RE = re.compile(r"[<>&|;$`*\n?]")

#: A file operand this module is willing to touch: a plain path. A glob would be expanded by
#: the shell into N files, and `grep -c` over N files prints N prefixed counts, which is not
#: the single number an absent-read is decided from; a `$VAR` operand expands to something the
#: stored text does not say, so what it would match case-insensitively is unknowable here.
_PLAIN_PATH_RE = re.compile(r"[A-Za-z0-9_./~+:@-]+")

#: A `SKILL.md` inside a `skills/` directory, matched on the raw stored token so both
#: spellings the ledger uses for it — `/home/<user>/obsidian/skills/x/SKILL.md` and
#: `~/obsidian/skills/x/SKILL.md` — land the same way. This is the owner-coverage shape #2103
#: is about: the file is the one an owner's own nightly rewrite edits under the verdict.
_SKILL_MD_OPERAND_RE = re.compile(r"(?:^|/)skills/[^/]+/SKILL\.md$")

#: The same shape over raw command text, for the mint-side check — deliberately wider than
#: `_SKILL_MD_OPERAND_RE`, and it is safe to be: this rail decides, it never rewrites the
#: command, so it does not need to know the operand is a plain path the way the strand rail
#: does before it re-executes one. Any token ending in `SKILL.md` counts, including one built
#: through a variable — `K=…/skills; grep -Fc 'command (string) is required'
#: "$K/tool-parameter-validation/SKILL.md"` is one of the ledger's own latest-wins falsifiers,
#: and a mint rule keyed on a literal `skills/` prefix would be a rule that `$K` walks through,
#: which is the same hole #2052's `--no-input-rows` flag was.
_SKILL_MD_PATH_RE = re.compile(r"[^\s'\"]+SKILL\.md")

#: Every `grep` invocation in a command, whether or not this module can parse it.
_GREP_MENTION_RE = re.compile(r"\bgrep\b")

#: The flag run immediately after the word `grep`: zero or more `-flag` tokens. Group 1
#: being empty means the very next token is the pattern, which is a case-sensitive read.
_GREP_FLAGS_AFTER_RE = re.compile(r"\s*((?:-[A-Za-z]+[ \t]+)*)")

#: A shell assignment — `rc=$?`, `S=~/obsidian/skills/x/SKILL.md`. One of the two wrapper
#: statements a denominator-carrying falsifier is written with since #2052, and a statement
#: that cannot change what grep matched.
_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")

#: `exit` of a captured status or a literal number: the other half of that wrapper.
_EXIT_STATUS_RE = re.compile(r"exit\s+(?:\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|\d+)")

#: A `name=` field carrying no value: the name, an `=`, then end of line or the next field
#: boundary. `input_rows=2812` is not it; `newest_bucket=` at end of line, or `newest_bucket= `
#: before the next field, is.
_EMPTY_VALUED_FIELD_RE = re.compile(r"(?:^|[\s,;(/])([A-Za-z_][A-Za-z0-9_-]*)=(?=$|[\s,;)])")


def _parse_grep_invocation(toks: list[str]):
    """`(flags, pattern, operands)` for token list `toks` beginning with `grep`, else None.

    The token list is already one shell segment, so this answers only whether THAT invocation
    is a literal read of plain paths: flags drawn solely from `_GREP_ARG_FREE_FLAGS` (one
    argument-taking flag — `-m N`, `-f FILE`, `-e PATTERN`, `-A N` — would make the parser
    guess which token is the pattern, and a wrong guess re-cases the wrong string), no
    `--long` option, a pattern with no regex metacharacter, and at least one plain-path
    operand. Grep with no operand reads stdin, where the child inherits whatever the nightly's
    pipe holds, so that is refused too.
    """
    rest = toks[1:]
    flags: list[str] = []
    while rest and rest[0].startswith("-") and rest[0] != "-":
        flag = rest.pop(0)
        if flag.startswith("--") or any(ch not in _GREP_ARG_FREE_FLAGS for ch in flag[1:]):
            return None
        flags.append(flag[1:])
    if not rest:
        return None
    pattern, operands = rest[0], rest[1:]
    if not pattern or _GREP_META_RE.search(pattern):
        return None
    if not operands or any(not _PLAIN_PATH_RE.fullmatch(op) for op in operands):
        return None
    return flags, pattern, operands


def _grep_readings(evidence_cmd: str):
    """`(greps, variant)` for a command this module can read as a whole, else None.

    `greps` holds one `(flags, pattern, operands)` per `grep` invocation; `variant` is the
    same command text with `-i` inserted into every case-sensitive one and is otherwise
    byte-for-byte the stored command. A falsifier is written one of two ways now that #2052
    demands a denominator — a bare `grep -c 'lit' path`, or that grep with an `rc=$?` capture
    and an `echo input_rows=N` beside it — and BOTH have to be readable, because a mandate or
    a re-read that reached only the bare form could be minted around by appending the very
    declaration the other rail demands. So a `;`/`&&`/`||`-separated command is read segment
    by segment and the whole command is refused unless every segment is a plain literal grep,
    an assignment, an `echo`/`printf`, an `exit` of a captured status, or empty.

    Refused outright, and therefore left to its own readings rather than rewritten on a guess:
    a command substitution (`$( )`, a backquote), a redirect, a subshell parenthesis, a `test`
    or `if`, a pipeline stage that is neither echo nor printf, and a `grep` this parser will
    not sign off on. The live ledger holds 8 keys of the refused shape (`S=…/SKILL.md; printf
    'a=%s' "$(grep -c 'lit' "$S")"`); they are history, and the mint rule below is what stops
    new ones arriving.
    """
    bounds: list[tuple[int, int]] = []
    start, i, quote = 0, 0, ""
    while i < len(evidence_cmd):
        ch = evidence_cmd[i]
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch in "<>()" or evidence_cmd.startswith("$(", i) or ch == "`":
            return None
        elif ch in ";\n|&":
            bounds.append((start, i))
            start = i + 1
        i += 1
    if quote:
        return None  # an unbalanced quote: bash cannot run this either
    bounds.append((start, len(evidence_cmd)))

    greps: list[tuple[list[str], str, list[str]]] = []
    rewrites: list[tuple[int, int, str]] = []
    for lo, hi in bounds:
        while lo < hi and evidence_cmd[lo] in " \t":
            lo += 1
        while hi > lo and evidence_cmd[hi - 1] in " \t":
            hi -= 1
        body = evidence_cmd[lo:hi]
        if not body:
            continue
        try:
            toks = shlex.split(body)
        except ValueError:
            return None
        if not toks:
            return None
        if toks[0] in ("echo", "printf") or _ASSIGNMENT_RE.match(body):
            continue
        if toks[0] == "exit" and _EXIT_STATUS_RE.fullmatch(body):
            continue
        if toks[0] != "grep" or (parsed := _parse_grep_invocation(toks)) is None:
            return None
        flags, pattern, operands = parsed
        greps.append((flags, pattern, operands))
        if not any("i" in flag for flag in flags):
            # Splice the flag into the segment's own text rather than re-joining its tokens.
            # `shlex.join(["grep", "-i", …])` single-quotes every operand that contains a
            # metacharacter, and bash does not expand a tilde or a variable inside single
            # quotes — so the re-read of the ledger's tilde-spelled falsifiers, which is how
            # most of them are written (`grep -c 'lit' ~/obsidian/skills/x/SKILL.md`), would
            # die on a file literally named `'~/obsidian/…'` and the strand would go
            # undiagnosed on exactly the corpus this rail exists for. Inserting at the end of
            # the flag run leaves every operand byte-identical to the stored spelling, shell
            # expansions included; `-i` goes immediately after the word `grep`, which GNU grep
            # accepts before any other flag, so the only bytes that move are the three inserted
            # ones and the published line reads `grep -i -c …` — the command an author would
            # write by hand to fix the key.
            rewrites.append((lo, hi, f"{body[:len('grep')]} -i{body[len('grep'):]}".strip()))
    if not greps:
        return None
    variant = evidence_cmd
    for lo, hi, text in reversed(rewrites):
        variant = variant[:lo] + text + variant[hi:]
    return greps, variant


def case_insensitive_reread(evidence_cmd: str) -> str | None:
    """The same check with `grep -i` added, or None when there is nothing to re-case.

    None three ways, each one a reason the re-read would prove nothing: the command is not
    one this module can read as a whole (`_grep_readings`), it holds no `grep` at all, or
    every `grep` in it already reads case-insensitively — which is the fix, not the defect,
    and also what bounds the recursion in `_run_stored_check`. `-i` is inserted as its own
    token in front of a re-quoted invocation, so the variant differs from the stored command
    by exactly one argument per grep and the wrapper around it (`rc=$?`, the denominator
    `echo`, `exit $rc`) survives byte for byte.

    This is the only place a re-read is computed, and both read surfaces ask it:
    `_run_stored_check` for a key already in the ledger, which is what makes `audit`'s
    `STRANDED_CASE` set and `check`'s line the same classifier's answer rather than two
    readings of the same row. It is deliberately NARROWER than the mint rule in
    `case_sensitive_skill_md_grep`, and the asymmetry has a reason: this one executes what it
    builds, so an unreadable command must be left alone; that one only refuses a write and
    names the exact edit that clears it, so being wide there costs an author one `-i` and
    closes the bypass the narrow form would leave open.
    """
    readings = _grep_readings(evidence_cmd)
    if readings is None:
        return None
    _greps, variant = readings
    return variant if variant != evidence_cmd else None


def _reads_case_insensitively(evidence_cmd: str) -> bool:
    """True when some `grep` mention in the command carries an `i` in its flag run.

    The exemption both case rails honour, decided from the flag run alone rather than from a
    full parse, because the flag run is legible exactly where the parse is not: the ledger's
    repaired key `seq-2-edit-err-read` interleaves `$(grep -ci …)` with `$(grep -c …)` and a
    `grep -m1 '^sessions:' $(ls -t …)` that no forward parse can finish, and a rule that could
    only be evaluated on a fully parseable command would refuse the one mint that has actually
    written the case-insensitive read down. Coarse by design, and coarse the same way in both
    rails: `case_insensitive_reread` asks no second question of a command that already reads
    case-insensitively, so a key can never be exempt at the mint and stranded at the audit.
    """
    # Same comment convention as `_grep_readings`: a trailing comment is not a reading, so a
    # note that mentions `grep -i` cannot exempt the command it describes.
    code = evidence_cmd.split("#")[0]
    for mention in _GREP_MENTION_RE.finditer(code):
        flags = _GREP_FLAGS_AFTER_RE.match(code, mention.end())
        if flags and any("i" in f for f in flags.group(1).split()):
            return True
    return False


def case_sensitive_skill_md_grep(evidence_cmd: str) -> str | None:
    """The `SKILL.md` path this command greps case-sensitively, or None.

    Owner-coverage is the one falsifier shape whose target is edited by somebody else's
    nightly: `autonomy-83`'s skills refresh re-writes an installed `SKILL.md` whenever a
    measurement under it changes, and #2103's witness key is the sentence that refresh
    re-cased — `grep -c "one mechanism, two symptoms"` minted 2026-09-12 against a file whose
    sentence read `One mechanism, two symptoms.` by 2026-10-03. The verdict then reads as
    though the owner had DELETED its section, and a later run that trusts it re-adjudicates or
    reopens a closed key on an artefact of casing. Measured on the live ledger
    2026-10-03: 27 latest-wins falsifiers name a `SKILL.md`, and this rail refuses 24 of them
    as new mints — the other 3 read something case-insensitively and are exempt. The mandate
    lands where it can still be obeyed, at the mint, and leaves those 24 rows as history.

    A command that reads anything case-insensitively is exempt (see
    `_reads_case_insensitively`): that is the ledger's repair idiom, `owner_section_ci=`
    beside `owner_section_case_sensitive=`, recording the artefact next to the coverage claim.
    Beyond that exemption there are two readings, widest first, because a mandate here is only
    as good as its narrowest bypass. Where `_grep_readings` can parse the command its verdict is
    used exactly; where it cannot (a command substitution, a `test`, a `$VAR` operand) the
    command is decided from its text and the tie goes against the mint: a path of that shape
    plus a `grep` whose flag run carries no `i` is refused. Both routes name the same remedy —
    read it case-insensitively — so the cost of the wide one is that an author adds the flag the
    refusal is asking for, while the cost of the narrow one is that `grep -c 'lit'
    skills/x/SKILL.md; echo 'input_rows=$(wc -l < f)'` walks straight past it.
    """
    path = _SKILL_MD_PATH_RE.search(evidence_cmd)
    if path is None or _reads_case_insensitively(evidence_cmd):
        return None
    readings = _grep_readings(evidence_cmd)
    if readings is not None:
        for flags, _pattern, operands in readings[0]:
            if "i" in "".join(flags):
                continue
            named = next((op for op in operands if _SKILL_MD_OPERAND_RE.search(op)), None)
            if named:
                return named
        return None
    return path.group(0)


_CANDIDATE_PATH_RE = re.compile(r"candidate-[^\s'\"]*\.md")
# A read aimed at the candidates *directory* is a read of a candidate file as surely as one
# naming it: the ledger's `seq-*` keys reach their snapshot through `$C/$N`, a directory
# variable and a filename variable, and a rule keyed on the word `candidate-…md` reads that
# operand as pointing nowhere. It is the same prose leak arriving by a longer path.
_CANDIDATE_DIR_RE = re.compile(r"skills/candidates(?![A-Za-z0-9_-])")
# A pipeline whose first stage emits *pathnames* — `ls $C | grep -cE "^candidate-x-[0-9]{8}\.md$"`,
# which is how four `seq-*` keys count their own dated snapshots — is a file count with a grep
# instead of a `wc -l`. No file's contents pass through it, so no verdict_reason can inflate it,
# and refusing it would refuse `dated_corpus_files=` itself, the shape `repair` writes into every
# tombstone. `cat`, `printf`, `head`, `tail`, `sed`, `awk` and `tr` all move file contents and are
# deliberately absent from this list.
_PATH_STREAM_CMDS = frozenset({"ls", "find", "basename", "dirname", "realpath"})
# The strip the ledger's repaired keys use: an `awk` program that counts `---` fences and
# emits only what comes after the second one. All three tokens are required — `^---` alone
# is any rule about a horizontal rule, and `n>=2` is what makes it a front-matter skip.
_FM_STRIP_RE = re.compile(r"\bawk\b[^\n]*\^---[^\n]*n\s*>=\s*2")
_ASSIGN_START_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)=")


def _skip_quoted(cmd: str, i: int) -> int:
    """Index just past the quote opening at `i`, or the end of the string if unclosed."""
    quote = cmd[i]
    j = i + 1
    while j < len(cmd):
        if cmd[j] == "\\" and quote == '"':
            j += 2
            continue
        if cmd[j] == quote:
            return j + 1
        j += 1
    return len(cmd)


def _shell_boundaries(cmd: str) -> list[tuple[int, str, int]]:
    """The `(position, kind, paren-depth)` of every separator and closing paren in `cmd`.

    Recorded at every depth rather than only at depth 0, because the reads this module
    decides over live inside command substitutions: `body_x=$(printf '%s\\n' "$b" | grep -ci
    lit)` is one assignment whose whole point is the pipe one level in, and a scanner that
    saw only depth-0 separators could not tell that grep's stdin from its file operand.
    A `$(` counts as an opening paren, and quotes are skipped whole, so neither a `;` inside
    a quoted pattern nor a `|` inside `$(ls -t … | head -1)` is mistaken for a boundary.
    """
    out: list[tuple[int, str, int]] = []
    depth, i, n = 0, 0, len(cmd)
    while i < n:
        ch = cmd[i]
        if ch in "'\"":
            i = _skip_quoted(cmd, i)
            continue
        if ch == "$" and cmd.startswith("$(", i):
            depth += 1
            i += 2
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            out.append((i, ")", max(0, depth - 1)))
            depth = max(0, depth - 1)
            i += 1
            continue
        if ch == "|":
            if cmd.startswith("||", i):
                out.append((i, "||", depth))
                i += 2
            else:
                out.append((i, "|", depth))
                i += 1
            continue
        if cmd.startswith("&&", i):
            out.append((i, "&&", depth))
            i += 2
            continue
        if depth == 0 and ch in ";\n":
            out.append((i, ";", 0))
        i += 1
    return out


def _paren_depth(cmd: str, pos: int) -> int:
    """The parenthesis depth at `pos` — the nesting level of the pipeline it sits in."""
    depth, i, n = 0, 0, min(pos, len(cmd))
    while i < n:
        ch = cmd[i]
        if ch in "'\"":
            i = _skip_quoted(cmd, i)
            continue
        if ch == "$" and cmd.startswith("$(", i):
            depth += 1
            i += 2
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        i += 1
    return depth


def _shell_tokens(text: str) -> list[tuple[str, int, int]]:
    """`(word, start, end)` for each word of `text`, quotes stripped, `$`-substitutions whole."""
    out: list[tuple[str, int, int]] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] in " \t\n":
            i += 1
            continue
        start = i
        while i < n and text[i] not in " \t\n":
            ch = text[i]
            if ch in "'\"":
                i = _skip_quoted(text, i)
                continue
            if ch == "$" and text.startswith("$(", i):
                depth = 0
                while i < n:
                    if text[i] in "'\"":
                        i = _skip_quoted(text, i)
                        continue
                    if text[i] == "(":
                        depth += 1
                    elif text[i] == ")":
                        depth -= 1
                        if depth == 0:
                            i += 1
                            break
                    i += 1
                continue
            i += 1
        out.append((text[start:i], start, i))
    return out


def _shell_assignments(cmd: str) -> dict[str, tuple[int, int]]:
    """Every `NAME=` assignment, mapped to the `(start, end)` of its value.

    The value runs to the next depth-0 separator, so `f=$(ls -t …/candidate-*.md | head -1)`
    is read as one binding despite the pipe inside its substitution — that shape is how every
    candidate file in the ledger gets named, and a scanner that stopped at the inner pipe would
    bind `f` to the word `$(ls`.
    """
    bounds = _shell_boundaries(cmd)
    found: dict[str, tuple[int, int]] = {}
    for m in _ASSIGN_START_RE.finditer(cmd):
        start = m.end()
        later = [pos for pos, kind, depth in bounds
                 if pos > start and depth == 0 and kind in (";", "|", "||", "&&")]
        found.setdefault(m.group(1), (start, later[0] if later else len(cmd)))
    return found


def _grep_reads(cmd: str, _depth: int = 0) -> list[tuple[str, bool, str, str, str]]:
    """Every `grep` in `cmd` as `(invocation text, counts?, operand text, stdin source text)`.

    Each invocation's own region runs from the word `grep` to the first separator at its own
    nesting level or shallower — so the operands of a grep inside `$( … )` stop at that
    substitution's closing paren and do not swallow the assignment that follows. A grep with
    no operand tokens reads its pipeline's stdout, and for those the source stage is returned
    as well: `printf '%s\\n' "$f" | grep -ci lit` believes it is scoping a read and is not,
    which is the false repair this has to catch as surely as the whole-file one.
    """
    bounds = _shell_boundaries(cmd)
    reads: list[tuple[str, bool, str, str, str]] = []
    for m in _GREP_MENTION_RE.finditer(cmd):
        start, depth = m.start(), _paren_depth(cmd, m.start())
        stops = [pos for pos, _kind, bd in bounds if pos > start and bd <= depth]
        # A `)` that takes the depth back below this grep's is the `$( … )` it lives in
        # closing, and the invocation cannot reach past it. `_split_shell` records only
        # `;`, `|`, `&&` and newline as boundaries, so without this the operands of
        # `snap=$(grep -cE '^p$') err=$(grep -c lit $f)` run on through the paren and swallow
        # the NEXT substitution — the first candidate read in the command then reports as
        # reading the last one's file, and every read before it is invisible to the rule.
        stops += [pos for pos in range(start, len(cmd))
                  if cmd[pos] == ")" and _paren_depth(cmd, pos) <= depth]
        end = min(stops) if stops else len(cmd)
        region = cmd[start:end]
        toks = _shell_tokens(region)
        if not toks or toks[0][0] != "grep":
            continue  # the word appears inside a pattern or an argument, not as a command
        idx = 1
        flags = []
        while idx < len(toks) and toks[idx][0].startswith("-") and not toks[idx][0].startswith("$"):
            flags.append(toks[idx][0])
            idx += 1
        if idx >= len(toks):
            reads.append((region, False, "", "", ""))
            continue
        # The pattern is an argument, never a target, and it is what this rule reads NEXT:
        # an anchored `^key:` pattern can only match a metadata line at column 0, which is the
        # scoping clause 3 exempts, and telling it from `grep -c lit $f` needs the pattern
        # itself rather than the invocation's text.
        pattern = _unquote(toks[idx][0]) if idx < len(toks) else ""
        idx += 1
        counting = any(tok.startswith("-") and "c" in tok for tok in flags)
        # Token offsets are relative to `region`, not to `cmd`: slicing `cmd` at one of them
        # would start mid-path and hand back a fragment of the pattern's tail.
        reading = region[toks[idx][1]:] if idx < len(toks) else ""
        # The stage piped into this grep, if any. Judged whether or not the grep also names a
        # file, because the ledger greps streams both ways: `ls $C | grep -cE '^x\.md$'` counts
        # pathnames with a pattern that is not a file, and `grep -c lit $f | wc -l` would read
        # the file whatever the pipe says.
        source = ""
        prior = [p for p, kind, bd in bounds if p < start and bd == depth and kind == "|"]
        if prior:
            pipe_at = prior[-1]
            before = [p for p, _kind, bd in bounds if p < pipe_at and bd <= depth]
            source = cmd[(max(before) + 1) if before else 0:pipe_at]
            stage = _shell_tokens(source)
            if stage and stage[0][0] in _PATH_STREAM_CMDS:
                source = ""            # the stream carries pathnames, not any file's contents
        reads.append((region, counting, reading, source, pattern))
    if _depth < 4:
        # `echo "x=$(grep -c lit $f)"` is one opaque quoted word to a region split, and bash
        # runs the substitution anyway. Every `seq-*` row in tonight's ledger hides its
        # candidate reads inside exactly that, so a reader that stops at the outer region finds
        # a falsifier that greps nothing. A grep assembled into a variable and `eval`ed is
        # unseen by this and by every other rail in the file; that mint has left all of them.
        # `$( … )` anywhere in the text, including inside a double-quoted word — which is
        # where every candidate read in tonight's `seq-*` rows sits: `echo "err=$(grep -c lit
        # $C/$N)"` is one opaque token to a word splitter, and a reader that stops at the
        # token finds a falsifier that greps nothing at all.
        covered = [(cmd.index(r), cmd.index(r) + len(r)) for r, *_x in reads]
        for open_at in (i for i, ch in enumerate(cmd) if cmd.startswith("$(", i)):
            if any(lo <= open_at < hi for lo, hi in covered):
                continue      # this level already reported the greps inside it
            stop = _matching_paren(cmd, open_at + 1)
            if stop > open_at:
                reads += _grep_reads(cmd[open_at + 2:stop], _depth + 1)
    return reads


def _matching_paren(cmd: str, open_at: int) -> int:
    """Index of the `)` closing the `(` at `open_at`, or -1 when it is never closed.

    Nesting-aware and quote-aware, so `$(echo "x=$(date)")` closes at the OUTER paren and the
    inner substitution is left to the recursive call that opens it in turn.
    """
    depth = 0
    i = open_at
    while i < len(cmd):
        ch = cmd[i]
        if ch in "'\"":
            i = _skip_quoted(cmd, i)
            continue
        if ch == "$" and cmd[i + 1:i + 2] == "(":
            depth += 1
            i += 2
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _names_a_variable(text: str, names: set[str]) -> bool:
    """True when `$NAME` or `${NAME}` for some NAME in `names` appears in `text`."""
    return any(re.search(rf"\$\{{?{re.escape(name)}\}}?(?![A-Za-z0-9_])", text)
               for name in names)


#: An anchored `^<key>:` pattern: a read for one of a file's metadata fields. It can only
#: match at column 0, and `mine-trajectories.status_block()` writes the verdict's prose onto
#: the `verdict_reason:` line, so no anchored read of a DIFFERENT key can reach that prose.
#: The one shape that would defeat it is a folded multi-line YAML value whose continuation
#: lines begin at column 0 with another key's name — `status_block` writes single-line values
#: (`f"verdict_reason: {reason}"`), and an author who changes that must revisit this line.
_FRONT_MATTER_KEY_RE = re.compile(r"\^[A-Za-z_][A-Za-z0-9_]*:")


def _unquote(token: str) -> str:
    """A shell word with its own surrounding quotes removed, if it had a matched pair.

    `_shell_tokens` keeps the quoting bytes, because the operand text has to stay verbatim
    for the tilde-spelling and path-shape tests to read it as the ledger wrote it. A PATTERN
    is the opposite case: the question asked of it is whether it begins with `^`, and
    `'^sessions:'` begins with a quote character.
    """
    if len(token) >= 2 and token[0] in "'\"" and token[-1] == token[0]:
        return token[1:-1]
    return token


def _targets_front_matter(pattern: str) -> bool:
    """True when the grep is asking for a metadata FIELD rather than counting prose.

    Clause 3's exemption, decided on the pattern alone: `grep -m1 '^occurrences:' $f` and
    `grep -m1 '^sessions:' $f` name the front matter deliberately, and `cmd_seed` and
    `cmd_repair` both mint `grep -m1 '^status:'` as their own template. A rule that refused
    those would refuse the ledger's two most common reads and the module's own mint paths.
    The anchor is doing the work, not `-m1`: an unanchored `grep -c 'sessions' $f` counts the
    word wherever the examples mention it, which is exactly the leak.
    """
    return bool(_FRONT_MATTER_KEY_RE.match(pattern))


def _reads_a_candidate(text: str, cand_vars: set[str]) -> bool:
    """True when `text` names something that actually resolves to a candidate file.

    A SINGLE-quoted substitution is skipped as inert, for the reason `_tilde_spelled_operands`
    already gives: bash does not expand `'$(ls -t …/candidate-x-*.md | head -1)'`, it hands
    `grep` a file literally named that, so such a read never opens a candidate file at all and
    the whole-file rule has nothing to say about it. `audit` reports those under UNRUNNABLE and
    the #2103 re-anchor rewrites them; refusing them here would be a second verdict on a shape
    the mint rail cannot fix, and the nightly minter emits it. A DOUBLE-quoted one is the
    opposite case and is not skipped: bash expands `"$( … )"`, so that read reaches the file.
    """
    for word, _start, _end in _shell_tokens(text):
        if word[:1] == "'" and "$(" in word:
            continue
        if (_CANDIDATE_PATH_RE.search(word) or _CANDIDATE_DIR_RE.search(word)
                or _names_a_variable(word, cand_vars)):
            return True
    return False


#: The candidate-input question `cmd_audit` answers from the row text alone (#2339).
#:
#: `_CANDIDATE_PATH_RE` already finds a `candidate-…md` token anywhere in a stored command; what it
#: cannot say is whether that token still NAMES a file. The dated-snapshot pruning in the candidate
#: directory does exactly that to a row: the pattern a key was recorded against stays in the ledger
#: after the file it was recorded against is gone, and the command that counts it then expands
#: `$C/$N` to the bare directory, prints `grep: …: Is a directory`, emits the count `0` and exits 0
#: through its trailing `echo`. `unrunnable: 0` is true of such a ledger and tells nobody anything,
#: so only the row and the directory can answer, and they answer here.
#:
#: Nothing at mint time consults this, deliberately: a command recorded while its snapshot exists is
#: honest, and the pruning that removes the snapshot later is not something its writer could have been
#: refused for, and 25 of the ledger's 116 current rows name a candidate file by path, glob or dated
#: pattern: a mint-time rule keyed on that would refuse a large slice of the ledger to prevent
#: something that only happens to a row later. #2333 rejects the mint-time half of itself on the same
#: grounds and asks for this half only.
#:
#: A token with none of `_CAND_META_RE`'s characters is a LITERAL filename, and a literal that has
#: gone is loud: `grep` prints `No such file or directory` and `UNRUNNABLE` decides it by executing.
#: This class exists for the silent death — a GLOB or a dated ERE that keeps its exact shape while
#: every file it ever named is pruned, which is the `$C/$N`-expands-to-the-directory case no
#: execution can see because the compound command still ends in `echo`.
_CAND_META_RE = re.compile(r"[*?\[{(|+]")
_CAND_DIR_FROM_TEXT_RE = re.compile(r"(/[^\s'\"`;|&()<>$]*)$")


def _candidate_dirs_assigned_in(cmd: str) -> list[str]:
    r"""Absolute directories this command assigns that ARE a candidates directory.

    The seq-shaped rows keep the directory in a variable (`C=/…/skills/candidates`) and reach the
    file only through `$C/$N`, so a bare pattern like `candidate-<key>-[0-9]{8}\.md` has nowhere to
    be resolved except the assignment the same command makes. `skills/candidates` is recognised by
    the existing `_CANDIDATE_DIR_RE`; the leading `/` is required because a relative `candidates` in
    a command whose cwd nobody knows is not a claim about any directory on this box.
    """
    out: list[str] = []
    for name, (start, end) in _shell_assignments(cmd).items():
        value = cmd[start:end]
        if value.startswith("/") and _CANDIDATE_DIR_RE.search(value) and value not in out:
            out.append(value)
    return out


def _candidate_listing(directory: str) -> list[str] | None:
    """The file names in `directory`, or None when it cannot be listed (missing, or not one)."""
    try:
        return sorted(p.name for p in Path(directory).expanduser().iterdir())
    except OSError:
        return None


def _pattern_names_any(pattern: str, listing: list[str]) -> bool | None:
    r"""Does one stored candidate NAME pattern name any file in `listing`? None if it cannot be said.

    A ledger pattern arrives in whichever dialect it was written in: a glob for `grep -E`/`ls`
    (`candidate-*-20260920.md`), an ERE for `grep -xE` (`candidate-…-[0-9]{8}\.md`), or a literal
    dated file, and `^`/`$` anchors appear on some of them. A pattern that mixes glob and regex
    metacharacters is left UNDECIDED rather than guessed at: translating `[0-9]{8}` as an fnmatch
    character class plus a literal `{8}` would answer "no files match" for a directory full of
    files that do, and a detector that manufactures the fabricated zero it exists to report is
    worse than the blind spot it closes.
    """
    pattern = pattern.lstrip("^").rstrip("$")
    if "$" in pattern or "`" in pattern:
        return None                      # still holds a variable: no file is being claimed
    globby = "*" in pattern or "?" in pattern
    if globby and "[" in pattern:
        return None
    if globby:
        return any(fnmatch.fnmatch(n, pattern) for n in listing)
    try:
        rx = re.compile(pattern)
    except re.error:
        return None
    return any(rx.fullmatch(n) for n in listing)


def candidate_input_lost(evidence_cmd: str) -> str | None:
    """A sentence naming what resolves to nothing, when a row's candidate input is off the disk.

    Decided from the stored command and a directory listing, because the execution cannot say it: a
    row whose dated snapshot has been pruned runs clean, prints a zero it did not earn, and exits 0.
    None — no claim — whenever the command leaves the question open, which is every shape where the
    answer would otherwise be a guess:

    * it names no `candidate-…md` token at all (an owner-skill grep: no input to have lost);
    * the only directory the name could resolve in is itself a variable (`$C/candidate-x-*.md` with
      no `C=` to ground it), or the pattern's own name still holds one (`candidate-$k-*.md`, which
      `ledger/falsifier_input_lost_for_non_error_seq_keys` stores tonight — resolving that token
      means matching the literal `$k` against a filename, finding nothing, and calling the result a
      finding);
    * the pattern is undecidable by `_pattern_names_any`, or any candidate token in the command DOES
      resolve — a command with three globs and one live snapshot is reading its input.

    A literal dated path that no longer exists is not answered here either: that one prints `No such
    file or directory` and belongs to `UNRUNNABLE`, which decides it by executing.

    One row shape is exempt, and it is exempt for what it CLAIMS rather than how it spells it: a
    disposal tombstone written by `repair_verdicts`. That row exists precisely because the corpus is
    gone, so its check counts the absence and prints it — `dated_corpus_files=0 pinned_input='<dir>
    <pattern>'` — and `audit` reading it as INPUT_LOST makes the ledger permanently unrestorable: the
    #2433 measurement was `run:2026-09-20-nightly-mining` named by `audit` every night over a
    disposal that had already been appended for it, which is why `repair --dispose-unverifiable` had
    nothing left to do and `input_lost` could never reach 0. The alternative — pinning an input the
    predicate cannot see — would retire the count by hiding the corpus from the detector, which is
    the blind spot arriving laundered rather than closed. What the exemption cannot buy is a dodge:
    `_DISPOSAL_TOMBSTONE_RE` is the template's own frame with a backreference, so an exempt row must
    actually count the corpus it pins, and a command that pins one path while counting another is
    still named.
    """
    if _DISPOSAL_TOMBSTONE_RE.fullmatch((evidence_cmd or "").strip()):
        return None
    specs = list(_CANDIDATE_PATH_RE.finditer(evidence_cmd))
    if not specs:
        return None
    assigned_dirs = _candidate_dirs_assigned_in(evidence_cmd)
    listings: dict[str, list[str] | None] = {}

    def listing_of(directory: str) -> list[str] | None:
        if directory not in listings:
            listings[directory] = _candidate_listing(directory)
        return listings[directory]

    for match in specs:
        pattern = match.group(0)
        if "$" in pattern or "`" in pattern:
            return None                       # undecidable name: the whole command is unprovable
        if not _CAND_META_RE.search(pattern):
            continue     # a literal dated path is UNRUNNABLE's, decided by running it
        head = _CAND_DIR_FROM_TEXT_RE.search(evidence_cmd[:match.start()])
        own_dir = head.group(1) if head and len(head.group(1)) > 1 else None
        where = [own_dir] if own_dir else assigned_dirs
        if not where:
            return None
        why: list[str] = []
        for directory in where:
            listing = listing_of(directory)
            if listing is None:
                why.append(f"{pattern} in {directory} (no such directory)")
                continue
            named = _pattern_names_any(pattern, listing)
            if named is None:
                return None
            if named:
                break                         # this token resolves: its input is on the disk
            why.append(f"{pattern} in {directory} "
                       f"({len(listing)} files there, none matching)")
        else:
            return "; ".join(why)
    return None


def candidate_body_defect(evidence_cmd: str) -> tuple[str, str] | None:
    """`(kind, detail)` for a falsifier whose candidate-file counts are not body-scoped, or None.

    A candidate snapshot's front matter is written by the ledger, not by the traffic: `status_block`
    in `mine-trajectories.py` re-injects the decision's own `verdict_reason` (and its
    `evidence_cmd`) into the front matter of every later dated snapshot for that key, so a `grep -c`
    over the whole file counts the decision's prose among the examples that justified it. #2166
    measured it on `candidate-edit-logic-20261004.md`: `whole=3 body=2` for the worktree-path
    pattern, the surplus match the stored reason at line 17. It fails in only one direction — the
    prose can add matches and never remove one — so a key whose falsifier greps the whole file
    looks more alive every time a verdict is re-recorded over it, which is the 2026-09-21 class
    rule "bind a count to the record set, never to narrative prose" arriving on the write side.

    `whole_file` is that leak: a `grep` carrying `-c` whose read target is a candidate file, either
    a variable bound to one (`f=$(ls -t …/candidate-*.md | head -1)`) or a literal candidate path,
    without a front-matter-stripped copy in between. Only a *count* is refused: a read that
    deliberately targets front matter — `grep -m1 -i '^occurrences:' $f`, `grep -m1 '^sessions:'` —
    asks that file a front-matter question on purpose, prints no count, and stays the ledger's
    normal way of reading `occurrences` and `sessions`. The same rule catches the false repair
    `printf '%s\\n' "$f" | grep -ci lit`, where the pipe is real but its source is the unstripped
    file: what is decided is the target of the read, not the presence of an awk near the grep.

    `dead_strip` is its mirror and the reason a per-invocation rule is not enough: the command
    defines the strip (`b=$(awk '/^---/{n++; next} n>=2' $f)`) and then never reads `$b`, so every
    field it emits is a front-matter field or an owner-skill grep. `Bash/timeout` in tonight's
    ledger is exactly that — its verbatim run emits `owner_pattern3`, `owner_timeout_words`,
    `owner_literal_phrase` and nothing else — which leaves the key with no candidate-body falsifier
    while its first clause tells the next reader that its counts are body-scoped. Looking repaired
    is the failure, not a loud one.

    Returns the first defect found, whole-file first: a command with both problems is refusing
    evidence before it is wasting one.
    """
    assigns = _shell_assignments(evidence_cmd)
    cand_vars = {name for name, (a, b) in assigns.items()
                 if _CANDIDATE_PATH_RE.search(evidence_cmd[a:b])
                 or _CANDIDATE_DIR_RE.search(evidence_cmd[a:b])}
    body_vars = {name for name, (a, b) in assigns.items()
                 if _FM_STRIP_RE.search(evidence_cmd[a:b])}
    for region, counting, reading, _source, pattern in _grep_reads(evidence_cmd):
        if not counting or not _reads_a_candidate(reading, cand_vars):
            continue
        if _targets_front_matter(pattern):
            continue
        if _names_a_variable(reading, body_vars):
            continue
        return "whole_file", region.strip()
    for name, (a, b) in assigns.items():
        if name not in body_vars:
            continue
        rest = evidence_cmd[:a] + evidence_cmd[b:]
        if not _names_a_variable(rest, {name}):
            return "dead_strip", f"{name}=$(awk '/^---/…' …) defined and never read"
    return None


def empty_valued_fields(observed: str) -> list[str]:
    """The `name=` fields in one recorded evidence line that carry no value, in order.

    The shape #2103 measured on the live ledger: `echo 'newest_bucket= newest_rows=
    prev_rows= input_rows=2812'` — an unquoted multi-word shell variable under the mining
    writer's `sh`, which printed the field *names*, lost their values, exited 0, and
    declared a real denominator. Every rail then in place read that line as healthy: the
    emptiness guard saw a printed line, `UNRUNNABLE` saw a command that ran, `EMPTY_INPUT`
    saw a nonzero denominator, and `audit` re-executes the same half-empty line and reports
    it clean. The durable record of the run's window was three numbers short of being
    auditable while its `input_rows=2812` matched the miner's own printed count exactly.

    Deliberately literal, and no wider: only a bare `name=` at a field boundary followed by
    end-of-line or the next boundary is a missing value. A quoted empty (`note=''`) and a
    value spelled as whitespace are different defects and this function does not claim them.
    The false-positive cost is a line of prose that happens to hold `word= ` before a space;
    rc 0 has to be true as well, and an author who meets the refusal can put a value in the
    field or print the measurement as a count.
    """
    return _EMPTY_VALUED_FIELD_RE.findall(observed or "")


def printed_count_is_zero(stdout: str) -> bool:
    """Whether a check's whole answer was the single number `0`.

    `grep -c` prints `0` and exits 1 when it matches nothing, so an rc-keyed rail cannot
    tell this from a healthy count and #2103's witness read as a measurement rather than a
    non-answer. Only the first non-empty line counts, and only when it is nothing but `0`:
    a findings header followed by a zero is the command reporting something else.
    """
    first = next((ln.strip() for ln in (stdout or "").splitlines() if ln.strip()), "")
    return first == "0"


def run_evidence(evidence_cmd: str, timeout: int = EVIDENCE_TIMEOUT_SECONDS) -> tuple[int, str]:
    """Execute a check the way `record` needs it: `(rc, observed)`, `observed` being the
    first non-empty line of its real output, truncated to `EVIDENCE_OBSERVED_MAX` and
    empty when the command printed nothing at all.

    Two differences from `evidence_cmd_status`, which answers a different question. That
    one asks whether an *already stored* check can still run, so it keeps the exit code
    and reports `UNRUNNABLE`; this one asks whether a check is worth storing, so it keeps
    the output. rc 1 beside a printed count is a falsification and a perfectly good
    verdict; rc 0 with no output is the vacuous assertion #736 is about, which is why
    emptiness, not rc, is what `record_verdict` refuses on. Its one other refusal is
    `evidence_cmd_status`'s verdict on the same command — a shape that cannot run is
    refused for being unrunnable, never for its exit code (#1586) — and this function's
    return shape and stderr fallback are what make that second refusal safe for a check
    that legitimately reports through stderr.

    stdin is DEVNULL, never inherited: `grep -c x` with no file blocks on stdin forever,
    and `record` runs synchronously inside a nightly turn — an inherited terminal would
    hang the job that is trying to write a verdict. With DEVNULL that command sees EOF at
    once and its emptiness is the answer the clause wants.
    """
    try:
        proc = subprocess.run(["bash", "-c", evidence_cmd], capture_output=True,
                              text=True, stdin=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        # Nothing printed within the bound is the same observation as nothing printed at
        # all: the caller refuses, and the rc names the shape of the silence.
        return UNRUNNABLE, ""
    except OSError as exc:  # no usable bash: the check cannot observe anything
        raise ValueError(f"evidence_cmd could not be executed at all: {exc}") from exc
    stdout, stderr = proc.stdout or "", proc.stderr or ""
    if not (stdout + stderr).strip():
        return proc.returncode, ""
    # stdout first: that is where a measurement goes. stderr is the only answer for a
    # command that reports through it, and quoting it is what makes the falsifier's own
    # failure visible at the moment the decision is made rather than months later.
    source = stdout if stdout.strip() else stderr
    line = next(ln.strip() for ln in source.splitlines() if ln.strip())
    return proc.returncode, (line[:EVIDENCE_OBSERVED_MAX] + "…"
                             if len(line) > EVIDENCE_OBSERVED_MAX else line)


def _run_stored_check(row: dict, timeout: int = EVIDENCE_TIMEOUT_SECONDS):
    """Execute a stored falsifier once: `(state, detail, declared_denominator)`.

    `state` is a real exit status, `UNRUNNABLE`, `EMPTY_INPUT` or `STRANDED_CASE`. The
    third value is `declared_denominator()` read over the run's stdout and it is returned
    on every path, the unrunnable ones included, because `audit` tallies declarations over
    the ledger's keys rather than over its successes — and `cmd_audit` calls *this* function
    so that a full audit still executes each stored command exactly once. `evidence_cmd_status`
    below is this with the third value dropped, which is what every other reader wants.
    """
    cmd = (row.get("evidence_cmd") or "").strip()
    if not cmd:
        return UNRUNNABLE, "no evidence_cmd recorded", None
    try:
        proc = subprocess.run(["bash", "-c", cmd], capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # Reported as unrunnable rather than passed over: a check that never finishes
        # disarms a verdict exactly as quietly as one that cannot start, and the bounded
        # timeout is worthless if nothing is printed when it fires.
        return UNRUNNABLE, f"still running after {timeout}s", None
    except OSError as exc:  # bash itself unusable: no verdict either way
        return UNRUNNABLE, str(exc), None
    declared = declared_denominator(proc.stdout)
    if proc.returncode == UNRUNNABLE:
        return UNRUNNABLE, (proc.stderr.strip().splitlines() or ["exit 127"])[0], declared
    missing = next((ln for ln in proc.stderr.splitlines()
                    if "No such file or directory" in ln), "")
    if missing:
        return UNRUNNABLE, missing.strip(), declared
    # A command the shell itself refuses to parse can never run either, and this is not
    # speculation about an exit code: as measured on the live ledger 2026-09-19, the key
    # `seq-2-read-edit` — whose falsifier the 09-15 triage confirmed re-executed at rc 0 —
    # exits 2 with `bash: -c: line 1: unexpected EOF while looking for matching '"'`,
    # because the stored string lost a quote. Detecting only exit 127 would leave the one
    # falsifier that is actually broken on this box silent, which is the fail-shut case
    # clause 4 exists to surface. Keyed on bash's own prefix, not on rc 2 at large: a tool
    # inside the command may exit 2 for its own reasons, and that is a result, not a fault.
    if proc.returncode == 2 and _BASH_PARSE_ERROR_RE.search(proc.stderr):
        return UNRUNNABLE, (proc.stderr.strip().splitlines() or ["bash parse error"])[0], declared
    if proc.returncode == 0 and declared == 0:
        # Every rail above is silent here — the command started, parsed, and exited 0 — and
        # its own stdout says it read nothing. `print(sum(...))` over a glob that matched no
        # file prints `0` byte-identically to a healthy count, which is why an exit status
        # can never decide this and why the field has to be the command's own claim: the
        # live ledger held 28 latest-wins keys whose every referenced absolute path is gone
        # and `audit` reported `unrunnable: 0` for all of them (#2048's executed proof).
        # Keyed on rc 0 only: a nonzero rc beside `input_rows=0` is the falsifier answering
        # no, which is the ledger working, and relabelling it would teach `check` to stop
        # trusting a real falsification.
        return EMPTY_INPUT, f"{INPUT_ROWS_FIELD}=0", declared
    if (variant := case_insensitive_reread(cmd)) is not None and (
            proc.returncode != 0 or printed_count_is_zero(proc.stdout)):
        # The answer no exit code reports, and the reason a rail keyed on rc 0 would miss it:
        # a case-sensitive `grep -c` that matches nothing exits **1** and prints `0` (measured,
        # #2103), which every rail above reads as the ledger answering no. So ask the same
        # command the question a second time with `-i` added: when absent becomes present, the
        # verdict is not measuring the corpus at all, it is measuring its own grep. The string
        # is there in another casing, which is exactly what an owner's nightly rewrite of its
        # own `SKILL.md` does to a falsifier minted against the old one — `seq-2-edit-err-read`
        # read `0` on 2026-10-03 for that reason, six weeks after it was minted against a
        # sentence that then read with the intended casing. Ordered after `EMPTY_INPUT`
        # because a run that read no input has nothing to be stranded about, and after the
        # `UNRUNNABLE` rails because a command that cannot execute is a different repair. The
        # second execution happens only on this branch, so a key whose first read found its
        # string still costs `audit` exactly one run, and `case_insensitive_reread` returns
        # None for a command that already reads case-insensitively — which is what bounds the
        # recursion here rather than letting a re-read re-read itself.
        again, _again_detail, _again_declared = _run_stored_check(
            {"evidence_cmd": variant}, timeout=timeout)
        if again == 0:
            return (STRANDED_CASE,
                    "the stored grep is case-sensitive and read absent while the same command "
                    f"with `-i` matches: {variant} — the string is present in another casing, "
                    "so this key measures its own grep and not the corpus; re-anchor it with "
                    "`-i` (#2103)",
                    declared)
    return proc.returncode, "", declared


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

    The third state is `EMPTY_INPUT`: an rc-0 run whose stdout declares `input_rows=0`,
    so it observed an empty input set. Neither of the other two, and named so a caller
    need not re-read the detail to tell them apart — a command exiting 0 over the files it
    believes it read is the shape that kept being honoured for 60 days on a check that
    could not see them, and `record_verdict` refuses to mint one while `audit` publishes
    it. The `UNRUNNABLE` rails are checked first and unchanged, so a command whose input
    file is gone is still reported as unrunnable, not as empty input: those two facts have
    different repairs, and only one of them is a path (#1588's repair pass reads
    `UNRUNNABLE` and never sees this state, which is the whole reason #2048 is a separate
    item from it).

    The fourth is `STRANDED_CASE`: the command ran and answered "absent", and answers
    "present" with `grep -i` added, so what it observed is its own casing rather than the
    corpus (#2103). `check` reports it on the same surface as the other two, one
    `EVIDENCE_CMD_STRANDED_CASE` line per blocked key, because the whole cost of this state
    is that a nightly reading the stored `0` concludes the owner LOST the section and
    re-adjudicates a closed key.
    """
    state, detail, _declared = _run_stored_check(row, timeout)
    return state, detail


# --------------------------------------------------------------- root-move repair --

#: The five answers `falsifier_repair` gives about a key. The first four are about a check
#: `audit` called UNRUNNABLE, and only `ROOT_MOVED` is a machine's business: the other two
#: name an input that no longer exists, or a field that never held a command, and both need a
#: re-derivation. The fifth arrives at a row `audit` called INPUT_LOST while `check` called it
#: healthy, and it is here because the two surfaces disagree on live keys today.
RUNNABLE = "runnable"
ROOT_MOVED = "root_moved"
NEEDS_RERECORD = "needs_rerecord"
UNPARSEABLE = "unparseable"

#: The fifth class, and the one the four above cannot reach: a stored falsifier that still
#: EXECUTES — so `evidence_cmd_status` has nothing to say about it — while the candidate snapshot
#: its own pattern names has been pruned, so what it prints is a zero it did not read. `audit`
#: named 6 such keys on the live ledger and exited 1; `repair` on the same 118-key ledger printed
#: six zeroes and exit 0, because `falsifier_repair` answered `RUNNABLE` for every row that was
#: not UNRUNNABLE and `repair_verdicts` drops that class. The one writer that could retire a
#: blind falsifier therefore reported there was nothing to do, which is the finding #2433 is
#: about. Decided by `candidate_input_lost` — `audit`'s predicate, called and not re-implemented:
#: a second predicate would give the pass and the audit two numbers, and #1588 already measured
#: what happens when one of them gets believed.
INPUT_LOST_CORPUS = "input_lost_corpus"

#: The `NEEDS_RERECORD` bracket for that class: `NEEDS_RERECORD <key> [input_lost]`. Named apart
#: from the class word because the bracket is published text a nightly greps, and `audit`'s
#: uppercase `INPUT_LOST <key> :: <detail>` line is a different surface's line about the same key.
INPUT_LOST_SHAPE = "input_lost"

#: The two `NEEDS_RERECORD` shapes a re-anchor can actually fix, kept as words because
#: they are what `reanchor_verdict` is offered for: the dead root is reachable by editing
#: the command, unlike a corpus retention deleted. Which one fired is worth printing —
#: a quoted glob and a nested string need different edits and look identical in `audit`.
NESTED_ROOT = "nested_root"        # rewritten command still PRINTS the dead root
QUOTED_GLOB = "quoted_glob"        # the rewritten detail still contains a `*`
DATED_CORPUS = "dated_corpus"      # the named input is simply gone: retention's doing
REANCHORABLE = (NESTED_ROOT, QUOTED_GLOB)

#: What a disposal relabels a *terminal* verdict to. Terminal, like the verdict it
#: supersedes, so honouring it keeps blocking exactly what it blocked before; and a
#: different word, so the ledger stops claiming a decision whose grounds can be re-run.
#: The non-terminal half of the rule is `disposal_verdict`.
UNVERIFIABLE_VERDICT = "rejected_unverifiable"

#: Tag every row this pass appends, so a reader can tell bookkeeping from a decision
#: an agent made about a candidate.
REPAIR_DECIDED_BY = "root-move-repair"
DISPOSE_DECIDED_BY = "root-move-repair-disposal"

#: A re-anchor's command was authored by hand against an artifact that still exists, so
#: it is neither the substitution nor a disposal. Named as a constant because the ledger
#: rows written on 2026-09-27 already carry this exact string: it is the value that makes
#: them reproducible, not a label invented after them.
REANCHOR_DECIDED_BY = "root-move-repair-reanchor"

#: A classifier detail shaped `grep: /path: No such file or directory` or
#: `ls: cannot access '/path*': No such file or directory` — the input is named, which
#: is what makes an absence check writable for it.
_MISSING_INPUT_RE = re.compile(
    r":\s*(?:cannot access\s*)?['\"]?([^'\"\n|&;]+?)['\"]?:\s*No such file or directory")

#: The check a disposal leaves behind: count the corpus the verdict was measured on.
#: A count and not an assertion — if a restore ever brings the dated files back the
#: number goes nonzero and the disposal is falsified by its own check, rather than by
#: somebody remembering to look. Braces are filled with a path already screened for
#: shell metacharacters by `disposable_input`.
TOMBSTONE_TEMPLATE = ('echo "dated_corpus_files=$(ls -1 \'{input}\' 2>/dev/null | wc -l)'
                      ' pinned_input=\'{input}\'"')

DISPOSAL_NOTE = (
    "#1588 disposal: the check above was pinned to {input}, which retention has pruned, so "
    "those grounds can no longer be re-executed by anyone; the appended check counts that "
    "corpus absent instead, and a nonzero count falsifies this disposal and reopens the "
    "decision. Verdict {kept}.")

#: A command that IS `TOMBSTONE_TEMPLATE`, recognised from the template's own frame so the two
#: cannot drift apart, with a backreference pinning the counted path and the published
#: `pinned_input=` to the same string. `candidate_input_lost` exempts exactly this shape: a
#: disposal's claim is the absence of its corpus, so classifying it as a falsifier that lost its
#: input is a verdict on the ledger's bookkeeping, not on a decision nobody can re-run (#2433).
_DISPOSAL_TOMBSTONE_PARTS = TOMBSTONE_TEMPLATE.split("{input}")
_DISPOSAL_TOMBSTONE_RE = re.compile(
    re.escape(_DISPOSAL_TOMBSTONE_PARTS[0])
    + r"(?P<pin>[^'\"\n]*)"
    + re.escape(_DISPOSAL_TOMBSTONE_PARTS[1])
    + r"(?P=pin)"
    + re.escape(_DISPOSAL_TOMBSTONE_PARTS[2]) + r"\Z")

#: The characters no pinned path may contain in either tombstone builder: each one can close
#: the quote the template splices it into, or start a substitution inside it. A backslash is
#: refused outright by `disposable_input` (#1588, and `tests/test_skill_verdicts.py` pins it).
#: `input_lost_input` screens it one notch narrower, and cannot do otherwise: the pattern half
#: of its pin arrives in the dialect the ledger stored it in, and every `grep -xE` key spells
#: its extension `[0-9]{8}\.md` — refusing that backslash would refuse all four live `seq-*`
#: keys and leave `audit`'s figure unreachable forever (owed #2433 clause 1). What stays
#: refused is a backslash escaping anything but a regex metacharacter, which is the class that
#: actually breaks the quoting: `C=/data/a\ b/skills/candidates` pins a `\ ` whose backslash
#: survives into `ls "…"` and eats the character after it.
UNSAFE_PIN_CHARS = "'\"$`"
PIN_SAFE_ESCAPES = ".*+?[]{}()|"


def pin_is_unsafe(pinned: str, *, backslash: str = "any") -> bool:
    """Whether `pinned` would let ledger DATA choose what the tombstone executes.

    `backslash="any"` is #1588's rule, character for character as `disposable_input` applied
    it before #2433; `backslash="unsafe"` exempts a backslash whose next character is one of
    `PIN_SAFE_ESCAPES`, the reading `input_lost_input` needs. Kept as one function so the two
    builders cannot drift into two different ideas of what is safe to splice.
    """
    if any(c in pinned for c in UNSAFE_PIN_CHARS):
        return True
    if backslash == "any":
        return "\\" in pinned
    i = 0
    while True:
        i = pinned.find("\\", i)
        if i < 0:
            return False
        if i + 1 >= len(pinned) or pinned[i + 1] not in PIN_SAFE_ESCAPES:
            return True
        i += 2


#: One clause of a `candidate_input_lost` detail, verbatim in its two shapes: `<pattern> in
#: <directory> (N files there, none matching)` or `(no such directory)`. The pattern is that
#: function's own token — `candidate-[^\s'"]*\.md`, so it carries no space and no quote — which
#: is what makes `\S+` unambiguous for it; the directory is the greedy remainder up to the last
#: ` (`, because an assigned directory is raw command text and may itself hold anything the
#: quoting rules allow.
_INPUT_LOST_DETAIL_RE = re.compile(
    r"^(?P<pattern>\S+) in (?P<directory>.+) "
    r"\((?:no such directory|\d+ files? there, none matching)\)$")


def dead_root_spellings(live: Path | None = None) -> tuple[tuple[str, str], ...]:
    """Each spelling the data root used to have, paired with where it lives today.

    `_pipeline/` used to sit inside the code tree; since 2026-09-22 it is a sibling of it
    (`architecture/data-home.md`), and the stored falsifiers that name the old location
    have reported `UNRUNNABLE` ever since — 79 of the ledger's 104 keys on 2026-09-27.

    Assembled from `LIVE_CHECKOUT` and `PIPELINE_DIR.name` rather than written as a
    literal, because a home-rooted runtime path in tracked code is precisely what
    `tests/test_no_runtime_paths_in_code.py` refuses: the *name* of a dead path is not a
    reason to reintroduce the bug that test exists to catch.
    """
    target = str(PIPELINE_DIR if live is None else live)
    in_tree = LIVE_CHECKOUT / PIPELINE_DIR.name
    # Absolute first — it is the longer spelling, and both land on the same directory.
    return ((str(in_tree), target), (f"~/{in_tree.parent.name}/{in_tree.name}", target))


def rewrite_dead_roots(cmd: str, *, rewrites: list[tuple[str, str]] | None = None
                       ) -> tuple[str, bool]:
    """`cmd` with every dead root spelling replaced, and whether anything changed."""
    out = cmd
    for old, new in (rewrites or dead_root_spellings()):
        out = out.replace(old, new)
    return out, out != cmd


def missing_input(detail: str) -> str:
    """The path a classifier detail names as absent, or '' when it names none.

    A glob spelling survives here (`…/candidate-*-20260920.md`) because bash never
    expands a quoted glob and neither does `ls`'s complaint about it: the literal,
    asterisks and all, is what the stored command looked for.
    """
    m = _MISSING_INPUT_RE.search(detail)
    return m.group(1) if m else ""


def disposable_input(detail: str) -> str:
    """`missing_input(detail)` when it is safe to embed in a check, else ''.

    A path carrying a quote, backslash or `$` would let the *ledger data* choose what
    the tombstone executes. Those keys are reported instead of disposed.
    """
    pinned = missing_input(detail)
    if not pinned or pin_is_unsafe(pinned):
        return ""
    return pinned


def input_lost_input(detail: str) -> str:
    """The corpus path an `INPUT_LOST_CORPUS` detail names, or '' when none is safe to pin.

    The other builder, `disposable_input`, reads a `No such file or directory` line, and there
    is no such line to read here: the whole point of #2339's class is that the command executes
    and says nothing of the kind, so its detail is the classifier's own sentence — `<pattern>
    in <directory> (<why>)` — and the path has to be rebuilt from its two halves.

    Rebuilt, never copied: the row's `evidence_cmd` is not a path, it is the compound command
    (`C=<dir>; N=$(ls $C | grep -xE "<pattern>"); …`), and lifting text out of it would hand the
    tombstone a string the ledger authored rather than the one this module measured. The
    directory and the pattern are what `candidate_input_lost` resolved the file *in* and *as*,
    which is the claim the appended count has to keep. Joining them with `/` reproduces the
    pattern's own spelling in the ledger's directory, trailing separator normalised so
    `/…/candidates/` + `candidate-*-20260920.md` does not become a doubled slash.

    Screened like the other builder, with the one documented exception in
    `UNSAFE_PIN_CHARS`: an assigned directory is raw command text, so
    `C=/data/a\\ b/skills/candidates` arrives with its backslash-space intact and would be
    spliced into the `bash -c` string below — such a key comes back as work, not as a tombstone.
    """
    first = (detail or "").split("; ")[0]        # multi-glob detail: pin the first lost corpus
    match = _INPUT_LOST_DETAIL_RE.match(first)
    if not match:
        return ""
    pinned = f"{match['directory'].rstrip('/')}/{match['pattern']}"
    return "" if pin_is_unsafe(pinned, backslash="unsafe") else pinned


def falsifier_repair(row: dict, *, timeout: int = EVIDENCE_TIMEOUT_SECONDS,
                     rewrites: list[tuple[str, str]] | None = None) -> tuple[str, str, str]:
    """Classify one dead verdict's falsifier: `(class, replacement_cmd, detail)`.

    `ROOT_MOVED` carries the rewritten command; the other classes carry the
    classifier detail that says why a rewrite cannot save the key, which is also the
    input a disposal pins. A row whose check runs answers `RUNNABLE` only when its input
    is on the disk: a command that executed but can no longer see what it claims to
    count is `INPUT_LOST_CORPUS`, decided by the predicate `audit` already applies to it.

    `evidence_cmd_status` decides both sides, deliberately. #1588 measured that judging
    a rewritten command by `rc in (0,1)` instead reports 62 repaired where the truth is
    31 — a probe shaped `grep <gone file> | head`, or a falsifier that dies by
    traceback, exits 0 or 1 over a missing input, so a zero reads as a clean answer.
    One classifier, shared with `audit` and `check`, or the pass and the audit report
    two numbers and one of them gets believed.

    A row whose check runs is not automatically healthy, though, and `RUNNABLE` used to
    swallow that: #2339's `INPUT_LOST` keys execute fine, print a zero they did not read,
    and exit 0 through their trailing `echo`, and `audit` names them from the row and a
    directory listing rather than from the execution. So the executing half of this
    function asks `audit`'s own predicate rather than assuming (#2433): the live ledger
    had `audit` exiting 1 over 6 such keys while `repair` printed six zeroes and exit 0
    over the same file, because the class the pass could not name was the class it
    dropped. Carried detail is that predicate's sentence, unchanged, so the repair and
    the audit quote one reader's finding about the same corpus.
    """
    status = evidence_cmd_status(row, timeout=timeout)
    if status is None or status[0] != UNRUNNABLE:
        lost = candidate_input_lost(row.get("evidence_cmd") or "")
        if lost is not None:
            return INPUT_LOST_CORPUS, "", lost
        return RUNNABLE, "", ""
    if _BASH_PARSE_ERROR_RE.search(status[1]):
        # Bash could not parse the field at all. Rows written before the parse-error
        # classifier existed hold a sentence where a command belongs, and no
        # substitution reaches prose.
        return UNPARSEABLE, "", status[1]
    rewritten, changed = rewrite_dead_roots(row.get("evidence_cmd") or "", rewrites=rewrites)
    if not changed:
        return NEEDS_RERECORD, "", status[1]
    after = evidence_cmd_status({**row, "evidence_cmd": rewritten}, timeout=timeout)
    if after is not None and after[0] == UNRUNNABLE:
        # The root was one dead input among others, so the rewrite does not save the key.
        # Which of three it is decides whether a person can still fix it, and the detail
        # says so without re-running anything: the OLD root surviving in the *output* is
        # a path inside a script or a nested string (the substitution reached the command,
        # not the string it hands to python); a `*` in the detail is a glob that sits
        # inside quotes, so no shell ever expanded it; neither, and the named input is
        # simply gone, which is retention's doing and no edit reaches it either.
        return NEEDS_RERECORD, "", after[1]
    return ROOT_MOVED, rewritten, status[1]


def rerecord_shape(detail: str, *, rewrites: list[tuple[str, str]] | None = None
                   ) -> str:
    """Which of the three shapes a `NEEDS_RERECORD` detail is, as one of the constants.

    Split out of `falsifier_repair` so the caller prints the cause and a test can name
    each shape without a subprocess: the two `REANCHORABLE` ones are worth separating
    because editing the command fixes them, while a pruned dated file is not repairable
    at all and only a disposal or a re-derivation answers it.
    """
    for old, _new in (rewrites or dead_root_spellings()):
        if old in detail:
            return NESTED_ROOT
    if "*" in detail:
        return QUOTED_GLOB
    return DATED_CORPUS


def reanchor_verdict(store: str | Path | None = None, *, pattern_key: str,
                     evidence_cmd: str, timeout: int = EVIDENCE_TIMEOUT_SECONDS,
                     dry_run: bool = False) -> dict:
    """Append a hand-authored falsifier for `pattern_key`, keeping its decision intact.

    The lane for the two `REANCHORABLE` classes, which a substitution cannot reach: the
    dead path is inside a script the stored command only names, or inside quotes bash
    never expands. Rewriting those is a person's judgement about what the signature is
    now owned by, so this takes the command as written — and then applies the discipline
    §0.6 asks of any falsifier, in code rather than in prose:

    * the command is executed through `evidence_cmd_status` before anything is written,
      and a command the classifier calls dead is **refused**. This is the #1586 hazard
      exactly: `record_verdict` only refuses a command that prints nothing, so a probe
      whose sole output is its own traceback would otherwise be born dead and re-appear
      in next night's `audit` as fresh damage from this pass;
    * the verdict, reason and `occurrences_at_decision` come from the superseded row, so
      re-anchoring cannot change what the key blocks or disarm the growth reopen;
    * the row is tagged `REANCHOR_DECIDED_BY`, which is what tells a reader that the
      command was authored by hand against a surviving artifact rather than substituted.

    Raises `ValueError` with the reason; the caller prints it.
    """
    table = load_verdicts(store_path(store))
    prior = table.get(pattern_key.strip())
    if not prior:
        raise ValueError(f"no verdict recorded for {pattern_key!r}; a re-anchor keeps an "
                         "existing decision, so there is nothing to keep — use `record`")
    cmd = (evidence_cmd or "").strip()
    if not cmd:
        raise ValueError("refused: evidence_cmd is empty")
    candidate = {**prior, "evidence_cmd": cmd}
    status = evidence_cmd_status(candidate, timeout=timeout)
    if status is not None and status[0] == UNRUNNABLE:
        raise ValueError(
            f"refused: the new falsifier is itself unrunnable ({status[1]}); a re-anchor "
            "must re-execute, or this pass mints a verdict that is born dead (#1586)")
    if dry_run:
        return {**prior, "evidence_cmd": cmd, "decided_by": REANCHOR_DECIDED_BY}
    return record_verdict(store, pattern_key=pattern_key, verdict=prior.get("verdict") or "",
                          reason=prior.get("reason") or "", evidence_cmd=cmd,
                          occurrences=None, decided_by=REANCHOR_DECIDED_BY,
                          source_candidate=prior.get("source_candidate") or "")


def disposal_verdict(prior: dict) -> tuple[str, str]:
    """The verdict a disposal records for `prior`, and how to say what it did to blocking.

    This is the whole safety of running a repair unattended: relabelling is only ever
    terminal → terminal. `proposed` and `below_threshold` stay as they were, because a
    pass over bookkeeping must not manufacture a block that stops a candidate being
    emitted when no such decision was ever made.
    """
    if is_terminal(prior):
        return (UNVERIFIABLE_VERDICT,
                f"recorded as {UNVERIFIABLE_VERDICT}, which is terminal like the verdict it "
                "supersedes, so nothing that was blocked stops being blocked")
    return (prior.get("verdict") or "",
            "kept unchanged, because a repair may not mint a block a non-terminal verdict "
            "never had")


def repair_verdicts(store: str | Path | None = None, *, dry_run: bool = False,
                    timeout: int = EVIDENCE_TIMEOUT_SECONDS,
                    rewrites: list[tuple[str, str]] | None = None,
                    dispose: bool = False,
                    reanchors: dict[str, str] | None = None) -> dict[str, list[str]]:
    """Append the corrections a moved root broke; report what only a re-derivation fixes.

    Every appended row goes through `record_verdict`, which is append-only by
    construction (#530) and refuses a command that observes nothing (#736 clause 2) — so
    a repair cannot mint a fresh born-dead verdict behind `audit`'s back, and the row it
    supersedes stays on disk byte for byte. `occurrences` is deliberately not passed:
    `carried_forward_occurrences` keeps the prior count, which is what keeps the growth
    reopen armed (#736 clause 1); passing `0` would disarm it.

    `dispose=True` additionally records the keys whose measurement corpus is provably
    gone, per `disposal_verdict`. It is opt-in because it relabels decisions however
    conservatively; the default fixes bookkeeping and touches nothing else.

    `INPUT_LOST_CORPUS` keys reach every branch below, which is the whole of #2433: the
    pass used to see none of them, because `falsifier_repair` called them `RUNNABLE` and
    the `continue` for that class sits above everything here. So on the live ledger of
    2026-10-08 `audit` exited 1 over 6 keys this function reported as six zeroes and exit
    0 — an operator handing it a `--reanchor-file` for one of those keys got `reanchored:
    0` and no reason, and the tombstone route that could retire them was unreachable. With
    the class named, a dry run prints `NEEDS_RERECORD <key> [input_lost]` and exits 1, an
    authored command for such a key is honoured, and `--dispose-unverifiable` appends the
    count-of-absence check that finally lets `audit` report `input_lost 0`.
    """
    table = load_verdicts(store)
    out: dict[str, list] = {"repaired": [], "disposed": [], "reanchored": [],
                            "needs_rerecord": [], "unparseable": [], "refused": [],
                            "mirror_skipped": [], "single_tree_rows": 0}
    for key in sorted(table):
        row = table[key]
        cls, new_cmd, detail = falsifier_repair(row, timeout=timeout, rewrites=rewrites)
        if cls == RUNNABLE:
            continue
        if cls == ROOT_MOVED:
            if dry_run:
                out["repaired"].append(key)
                continue
            try:
                # This module wrote `new_cmd`: it is the superseded command with one dead
                # root replaced, so the field the #2052 mandate asks for is not this pass's
                # to author — and refusing here would turn a repair into a `refused` entry
                # and leave the key's falsifier pointing at the root retention moved. The
                # row still lands `undeclared`, where `audit` keeps counting it.
                record_verdict(store, pattern_key=key, verdict=row["verdict"],
                               reason=row["reason"], evidence_cmd=new_cmd,
                               occurrences=None, decided_by=REPAIR_DECIDED_BY,
                               source_candidate=row.get("source_candidate") or "",
                               require_input_rows=False, require_body_scope=False)
                out["repaired"].append(key)
            except ValueError as exc:
                out["refused"].append(f"{key} :: {exc}")
            continue
        if cls == UNPARSEABLE:
            out["unparseable"].append(key)
            continue
        # Which shape it is decides who can fix it. A nested root or a quoted glob is a
        # dead path sitting where a substitution cannot reach but an *authored* command
        # can, so those are the shapes `reanchors` is for; a dated corpus is nobody's to
        # author around, since the input retention deleted is gone from every tree. An
        # input-lost corpus is the second of those for its count and the first of them for
        # its key: the dated snapshots are gone, but the signature they were mined from may
        # well have an owner-skill falsifier now, so an authored command is exactly what a
        # person would hand this key — which is why the classification happens above the
        # `authored` lookup below and not inside the disposal branch (#2433 clause 2).
        shape = (INPUT_LOST_SHAPE if cls == INPUT_LOST_CORPUS
                 else rerecord_shape(detail, rewrites=rewrites))
        authored = (reanchors or {}).get(key, "").strip()
        if authored:
            try:
                reanchor_verdict(store, pattern_key=key, evidence_cmd=authored,
                                 timeout=timeout, dry_run=dry_run)
                out["reanchored"].append(key)
            except ValueError as exc:
                out["refused"].append(f"{key} :: {exc}")
            continue
        # Two builders for one bracket: a `No such file or directory` detail names its whole
        # path, while an input-lost detail names a directory and a pattern separately and has
        # to be joined here. Both screens are `UNSAFE_PIN_CHARS`, so a key whose path would
        # splice shell into the tombstone is reported as work on either route.
        pinned = (input_lost_input(detail) if cls == INPUT_LOST_CORPUS
                  else disposable_input(detail))
        if not dispose or not pinned:
            out["needs_rerecord"].append(f"{key} [{shape}]")
            continue
        verdict, kept = disposal_verdict(row)
        out["disposed"].append(key)
        if dry_run:
            continue
        # The tombstone is this module's own template, and its whole claim is an absence:
        # `dated_corpus_files=0` over a corpus retention pruned. Asking it to declare a
        # denominator would put the #2048 EMPTY_INPUT rail and #1588's disposal in one
        # argument — the disposal exists precisely because the input is gone — so the
        # mandate is waived here and the row is published `undeclared` for `audit` to
        # count. A re-anchor, whose command a person authors against a surviving artifact,
        # is not waived: it reaches the mandate through `reanchor_verdict`.
        record_verdict(store, pattern_key=key, verdict=verdict,
                       reason=(f"{row['reason']} "
                               f"{DISPOSAL_NOTE.format(input=pinned, kept=kept)}"),
                       evidence_cmd=TOMBSTONE_TEMPLATE.format(input=pinned),
                       occurrences=None, decided_by=DISPOSE_DECIDED_BY,
                       source_candidate=row.get("source_candidate") or "",
                       require_input_rows=False, require_body_scope=False)
    # Carried in the returned tally, not only on `record_verdict`'s stderr, because this
    # pass is the writer that produced #1717: 79 appends, one tree, no line anywhere that a
    # nightly could have grepped. `single_tree_rows` is the count of rows this pass wrote
    # into the ledger alone — a caller that finished "repaired: 79" while that was nonzero
    # reported a repair and delivered half of one.
    written = (len(out["repaired"]) + len(out["reanchored"]) + len(out["disposed"]))
    if written and not dry_run and _mirror_target(store_path(store)) is None:
        out["mirror_skipped"] = [f"{mirror_skip_notice(store_path(store))} "
                                 f"rows_this_pass={written}"]
        out["single_tree_rows"] = written
    return out


def sync_durable_copy(store: str | Path | None = None, *, dry_run: bool = False,
                      hold_disagreements: bool = False) -> dict:
    """Append the live ledger's latest-per-key rows the durable copy does not hold.

    The re-runnable route the 79 one-tree writes need (#1717): `check` can now tell lag
    from disagreement and `record` can no longer write one tree silently, but neither
    puts the missing rows back, and the copy that is 79 lines behind is a restore that
    quietly loses 79 decisions — which is the exact failure #772 was filed for.

    What it writes is the live ledger's **latest row for the key, byte for byte**, appended
    to the copy. Never a rewrite, never a deletion: the copy is append-only for the same
    reason the ledger is, so its own history survives, and a key the copy holds that live
    has lost (`LEDGER_LOST`) is left standing — appending cannot answer a live-side
    disappearance, and deleting the surviving copy of a decision to make two files look
    alike would be the cure killing the patient.

    Idempotent by construction: a row is offered only when the copy's latest row for that
    key is absent or differs, so the second run has nothing to offer and adds 0 lines.

    Refuses, appending nothing, when this ledger has no durable copy of its own — a
    non-default `--store` with `$SKILL_VERDICTS_MIRROR` unset (`_mirror_target`'s guard).
    That guard exists so scratch decisions cannot reach the production vault copy, and a
    `sync` that wrote them there would be the loudest way to defeat it.

    `hold_disagreements=True` leaves a `DISAGREEING` key for a human instead of adopting
    the live verdict as the durable answer. The default copies it — the live ledger is the
    tree the pipeline reads, and #1717 asks for the two to be brought together — but the
    tally prints each such key with both verdicts, so a run that meant to preview the
    relabels has the flag rather than a re-derivation to do.
    """
    live = store_path(store)
    out: dict[str, object] = {"copied": [], LAGGED: 0, DISAGREEING: 0, "missing": 0,
                              "held": [], "refused": []}
    if not live.is_file():
        out["refused"] = [f"SYNC_REFUSED {live} :: the live ledger is not there to sync "
                          "from. The durable copy is left alone: an absent live file is "
                          "the incident #772 exists for, not an instruction to stop "
                          "keeping the copy current."]
        return out
    durable = mirror_for(live)
    if durable is None:
        out["refused"] = [f"SYNC_REFUSED {live} :: this ledger has no durable copy of its "
                          f"own to write: it is not the default {DEFAULT_STORE} and "
                          "$SKILL_VERDICTS_MIRROR is unset, so pointing `sync` at a "
                          f"scratch store must not append scratch rows through the "
                          f"production vault copy {DEFAULT_MIRROR}. Name the copy you "
                          "mean with $SKILL_VERDICTS_MIRROR, or run this against the "
                          "default ledger."]
        return out
    if Path(durable).resolve() == live.resolve():
        out["refused"] = [f"SYNC_REFUSED {live} :: the durable copy resolves to the ledger "
                          "itself, so there is one tree here and nothing to bring together."]
        return out

    live_table, durable_table = load_verdicts(live), load_verdicts(durable)
    lines: list[str] = []
    for key in sorted(live_table):
        live_row, durable_row = live_table[key], durable_table.get(key)
        if durable_row is None:
            kind, label = "missing", "MIRROR_MISSING"
        elif durable_row == live_row:
            continue
        else:
            kind = divergence_class(live_row, durable_row)
            label = kind
        if kind == DISAGREEING and hold_disagreements:
            out["held"].append(f"{key} :: live={live_row.get('verdict')} "
                               f"durable={durable_row.get('verdict')}")
            continue
        out["copied"].append(f"{key} :: {live_row.get('verdict')} {label}")
        out[kind] = out[kind] + 1
        lines.append(json.dumps(live_row, ensure_ascii=False) + "\n")
    if lines and not dry_run:
        with Path(durable).open("a", encoding="utf-8") as fh:
            fh.writelines(lines)
    out["durable"] = durable
    out["rows"] = 0 if dry_run else len(lines)
    out["would_copy"] = len(lines)
    return out


def scan_candidates(candidates_dir: Path, store: Path | str | None = None, *,
                    table: dict[str, dict] | None = None) -> list[dict]:
    """One row per candidate file: its key, whether a verdict blocks it, and why.

    `table` lets a caller that already resolved the verdict source (see
    `resolve_verdict_source`, and `cmd_check`) hand the loaded ledger over instead of
    reading the default store a second time.

    A candidate whose frontmatter says `signature_buckets: N` with N > 1 (#515) is one
    emission unit standing for N mined signature patterns, and its `occurrences:` is the
    sum over all of them. The ledger's `occurrences_at_decision` for such a key was
    recorded from ONE bucket's file (#530 seeded it before buckets were merged), so
    comparing the two is a units error: the >10x growth reopen is not evaluated for a
    merged unit, exactly as `mine_trajectories.verdict_for` decides from the other side
    of the same seam. Terminality and the 60-day expiry still bind, so the set of
    suppressed keys is the same one either program computes.
    """
    rows = []
    if table is None:
        table = load_verdicts(resolve_verdict_source(store)[0])
    for file in sorted(Path(candidates_dir).glob("candidate-*.md")):
        key, status, occurrences, buckets = read_candidate(file)
        if not key:
            continue
        row = table.get(key)
        terminal = is_terminal(row)
        growth_baseline_free = buckets > 1
        lift = (reopen_reason(row, occurrences=0 if growth_baseline_free
                              else occurrences) if terminal else None)
        rows.append({
            "file": file.name,
            "pattern_key": key,
            "status": status,
            "occurrences": occurrences,
            "signature_buckets": buckets,
            "growth_reopen_evaluated": not growth_baseline_free,
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
        # How many mined patterns one verdict stands in front of (#515): a coarse
        # `Bash/logic` rejection gating 19 signatures must not read as one judgement
        # about one pattern.
        merged = (f" [{row['signature_buckets']} mined patterns under this key]"
                  if row["signature_buckets"] > 1 else "")
        if row["blocked"]:
            print(
                f"SKIP {row['pattern_key']} :: {row['verdict']} :: {row['reason']} "
                f"(decided {row['decided_at']}, file {row['file']}){merged}"
            )
        elif row["reopened"]:
            print(f"REOPEN {row['pattern_key']} :: {row['reopened']}")
        else:
            print(f"PROCEED {row['pattern_key']}")
    # `LEDGER_ABSENT` first: it is the one finding that changes the exit status, and a
    # reader who sees only `skipped_by_verdict: 0` cannot tell a quiet night from a
    # deleted decision history. An empty candidates dir mints no alarm — with nothing
    # adjudicated, an absent ledger is a fresh install, and alarming there is how a
    # warning becomes noise (#736 clause 4).
    exit_code = 0
    if not live.exists():
        minted = [r["status"] for r in rows if r["status"] in MINTED_BY_LEDGER]
        if minted:
            exit_code = 1
            counts: dict[str, int] = {}
            for status in minted:
                counts[status] = counts.get(status, 0) + 1
            tally = ", ".join(f"{s}: {counts[s]}" for s in sorted(counts))
            print(
                f"LEDGER_ABSENT {live} :: the store is gone while {len(minted)} candidate "
                f"file(s) still carry a status only this ledger mints ({tally}); every key "
                "behind them now reads as undecided, so the consolidator and the miner "
                "resume proposing what was already rejected — the empty ledger is the "
                "incident, not a quiet night (#736 clause 4)"
            )
    # The findings below print above the totals, and the totals stay the last line: the
    # runbook and the nightly greps read the counts off `splitlines()[-1]`, so a
    # warning that displaced them would break a reader that is not looking for it.
    if not fell_back:  # when the copy *is* the source there is nothing to compare it with
        report = check_against_mirror_report(live, table)
        if report is not None:
            for line in report["lines"]:
                print(line)
            # The tally beside the classes (#1717 clause 4). Printed whether or not
            # anything diverged: `lagged: 0 disagreeing: 0` is the healthy answer, and a
            # denominator that only appears with findings cannot tell a clean night from a
            # comparison nobody ran. 2026-09-28's state is what the split is for — 79 lines
            # of one label, of which 54 were lag and 25 were the real thing.
            print(mirror_tally_line(report))
    for key in sorted({r["pattern_key"] for r in skipped}):
        rc, detail = evidence_cmd_status(table.get(key) or {})
        if rc == UNRUNNABLE:
            print(f"EVIDENCE_CMD_UNRUNNABLE {key} :: {detail}")
        elif rc == EMPTY_INPUT:
            # The same decision `audit` publishes, on the surface a nightly actually reads
            # between its SKIP choices: the verdict was honoured, and its own falsifier
            # says it read nothing. Named rather than silently honoured because
            # "honoured on a check that cannot see its input" is the state #2048 exists to
            # make visible, and this loop is where 2 of the ledger's keys get exercised.
            print(f"EVIDENCE_CMD_EMPTY_INPUT {key} :: {detail}")
        elif rc == STRANDED_CASE:
            # The third instrument failure, named here because this is the loop that reads a
            # stored `0` and decides what it means. A key whose case-sensitive literal grep
            # reads absent while `-i` finds the string is measuring its own grep, and the
            # inference a nightly draws from `evidence_observed: "0"` in this key family is
            # "the owner skill does not hold this section" — which is how a closed key gets
            # re-adjudicated on a casing artefact six weeks after it was minted (#2103). The
            # line carries the key and the classifier's detail, which names the `-i` command
            # that re-reads it, and nothing else changes: the block the verdict produced is
            # untouched, and the strand never enters `unverified`, because the classifier hands
            # this loop the instrument's state rather than the row's underlying exit status and
            # a strand arises at rc 0 and at rc 1 alike — counting it would need a third
            # execution to learn which, for a number `audit` already fails the run over. Same
            # classifier as `audit`, so a key named here is named there and cannot be named by
            # one surface alone.
            print(f"EVIDENCE_CMD_STRANDED_CASE {key} :: {detail}")
    if fell_back:
        print(f"verdict source: {source} (live ledger {live} is absent)")
    print(f"checked: {len(rows)}  skipped_by_verdict: {len(skipped)}")
    return exit_code


def cmd_audit(args: argparse.Namespace) -> int:
    """Re-execute every stored falsifier in the ledger and name the ones that cannot run.

    Why `check` does not already cover this: `check` runs `evidence_cmd` only for keys
    that just blocked a candidate in the scanned directory — two candidates tonight, so
    2 of the ledger's 103 latest-wins keys were exercised and the other 101 were
    honoured on the strength of a check nobody ran. #530/#525 made the falsifier
    mandatory precisely so a later run could *falsify* a verdict rather than inherit it;
    a falsifier nobody executes falsifies nothing, and Phase 0 keeps spending its SKIP
    decisions on it anyway. The `_pipeline` root that used to sit inside the code tree was
    deleted on 2026-09-22 (#1377), and 78 of those 103 stored commands still name that
    dead location — which is what this surface is for. (The root is described rather than
    spelled out: `tests/test_no_runtime_paths_in_code.py` refuses a home-rooted
    runtime path anywhere in tracked code, and a pattern match does not exempt prose.)

    The judgement is `evidence_cmd_status`'s — the same classifier `check` reports
    `EVIDENCE_CMD_UNRUNNABLE` from — asked of every key instead of the handful the scan
    happens to touch. A command that exits 0 or 1 has observed something and is not a
    finding even when it observed nothing; one that names a file that is gone, that bash
    cannot parse, that exits 127, or that outlives `--timeout` is. A key with no
    `evidence_cmd` at all is counted in `keys:` and not named, because there is no
    command to fail: that shape is the ledger's own invariant (#530 makes the field
    mandatory), and `record` refuses to write it.

    Output is read by the nightly jobs, so the shape is the contract: one
    `UNRUNNABLE <pattern_key> :: <detail>` line per dead key, one
    `EMPTY_INPUT <pattern_key> :: <detail>` line per key whose rc-0 check declared
    `input_rows=0`, and one `STRANDED_CASE <pattern_key> :: <detail>` line per key whose
    case-sensitive literal grep read absent while the same command with `-i` matches, and one
    `INPUT_LOST <pattern_key> :: <detail>` line per key whose stored command still names a
    candidate snapshot by glob or dated pattern while no file on disk matches it any more (#2339)
    — all four in key order, since they interleave and a reader greps for the class, not the
    position.

    `INPUT_LOST` is the fourth shape of the candidate-scoping trio and the only one decided from the
    row and the directory rather than from the row alone or the execution. It exists because the
    candidate directory is pruned: the pattern a key was recorded against outlives the dated snapshot
    it was recorded against, `$C/$N` then expands to the bare directory, `grep` prints `Is a
    directory` and emits `0`, the trailing `echo` exits 0, and the ledger reports `unrunnable: 0` over
    four falsifiers that can no longer see their input. Those rows are NOT counted in `whole_file`:
    a falsifier reading nothing is not counting a candidate whole, and naming it
    `WHOLE_CANDIDATE_COUNT` sends the repair to a missing `awk` when the repair is a re-mint. It is
    decided before `candidate_body_defect` for that reason, and it enters the exit code by name so
    excluding a row from one figure cannot quietly buy a green audit.

    Then the tallies, and their order is load-bearing: `exit_drivers:`, then
    `candidate_body_scoping: whole_file N dead_strip N input_lost N`, then `stranded:
    case_sensitive_grep N`, then `denominators: empty_input N undeclared M`, then
    `keys: N unrunnable: M` as the LAST line — `check`'s shape, so a reader that takes
    `splitlines()[-1]` gets the ledger tally and one that takes `[-2]` gets the denominator
    tally it has been reading since #2048. The new count enters ABOVE those two rather than
    between them, because displacing a published line to make room for a new one is how a
    reader that is not looking for the change silently reads the wrong figure; the tests pin
    those positions by index. Exit 1 when `unrunnable:` > 0 or ANY count above it does —
    `whole_file`, `dead_strip` and `input_lost` included, each counted in the same six-term
    rule rather than by inheriting another's — so an unverifiable ledger fails a run instead of
    printing into a log nobody re-reads.

    `exit_drivers:` exists because those two facts were printed separately and the reader was left
    to join them. The exit is driven by six figures, but the two the nightly runbooks carry into
    their report are `unrunnable:` and `empty_input`, so #2343 measured a run publishing
    `keys: 116 unrunnable: 0` and `denominators: empty_input 0 undeclared 102` beside exit 1 with
    nothing anywhere on stdout naming the term that fired (`input_lost 6`). This line names every
    term whose count is non-zero as `<term> <count>` pairs, spelled with the tokens the tally lines
    already publish — `input_lost`, never `lost_rows` — so it ties to a line the reader already has,
    and reads `exit_drivers: none` on a run that exits 0, because a tally that only appears with
    findings cannot tell a clean night from an audit nobody ran. It is the first line of the block
    and the exit reads the same six counts: the pair can never disagree.

    The stranded line exists because the ledger's own worst silent failure is not a command
    that cannot run. A falsifier that greps an installed `SKILL.md` for a literal string
    case-sensitively keeps exiting 1 and printing `0` after the owner's nightly rewrite
    re-cases that sentence, and `grep -c` leaves no trace of the difference in the row: no rc
    is stored, and this surface's `unrunnable: 0` was reported over exactly that reading the
    night #2103 was filed. It is a third question, asked of the same execution — can the
    check run, what did it read, and would it answer differently case-folded — which is why
    the count has its own line and is never folded into `unrunnable:`: a case-stranded key ran
    fine, and calling it unrunnable would both hide #1533's real figure and point the repair
    at the wrong thing (the fix is `grep -i`, not a re-anchored path). `check` names the same
    keys on the same classifier, one `EVIDENCE_CMD_STRANDED_CASE` line each, so the nightly
    that reaches for `evidence_observed` alone is the one most likely to conclude the owner
    deleted a section — that reader now sees the artefact instead.

    The tallies answer different questions and one key can appear in more than one.
    `unrunnable:` is about whether the check can execute; `denominators:` is about whether it
    says what it read — `undeclared` counts every key whose command named no `input_rows`, the
    unrunnable ones included, so `empty_input + undeclared + declared` adds up to `keys:`
    and a reader can check the arithmetic. `stranded:` is a third axis again, overlapping the
    other two freely: a key can be stranded and undeclared, and cannot be stranded and
    unrunnable, because the re-read that proves the artefact has to run, so #2103 leaves that
    pairing to `UNRUNNABLE`, which the classifier decides first. That is why each figure is a
    separate line and a separate exit: #2048's whole finding is that a ledger can be fully
    honoured on falsifiers that can no longer see their input while `unrunnable:` reads 0, and
    a number that already covers the case cannot expose it. An undeclared denominator is not a
    fault at this surface and does not move the exit code — 109 of the live ledger's keys are
    undeclared tonight, every one of them recorded before the field existed, and refusing
    them at read time would quarantine a ledger nobody authored wrongly. The mandate landed
    where it can still be obeyed instead: `record_verdict` refuses to mint a new one of them
    (#2052), so this figure falls only as keys are re-recorded, and it is published here with
    no target value attached — `nightly-skill-consolidation`'s Phase 0.6 states the
    convention, and this line is what measures whether it is biting.
    """
    table = load_verdicts(store_path(args.store))      # latest-wins, the table check reads
    dead, empty_input, stranded, undeclared = [], [], [], 0
    # #2166's two shapes, decided on the stored text alone: neither needs the command to run,
    # because the defect is what the command reads rather than what it answers. A key whose
    # falsifier counts a candidate file whole has been inflating all along and reported rc 0
    # doing it, so no execution-based tally in this function can ever reach it.
    whole, dead_strip = [], []
    # #2339's third shape on the same stored text: the row still names its candidate file, and the
    # file is gone. Kept out of `whole` by the order in the loop below — a falsifier that cannot see
    # its input is not inflating a count — and put into the exit code by the line at the end.
    lost_rows: list[str] = []
    for key in sorted(table):
        # The three-value form, so one audit still runs each stored command once: the
        # denominator comes off the same execution that decided runnability, never from a
        # second run that could disagree with the first. `STRANDED_CASE` costs a second
        # execution and only for a key whose first read already came back absent, so
        # publishing it here cannot slow the hundred-odd keys that answered normally.
        state, detail, declared = _run_stored_check(table[key], timeout=args.timeout)
        if declared is None:
            undeclared += 1
        if state == UNRUNNABLE:
            dead.append(key)
            print(f"UNRUNNABLE {key} :: {detail}")
        elif state == EMPTY_INPUT:
            empty_input.append(key)
            print(f"EMPTY_INPUT {key} :: {detail}")
        elif state == STRANDED_CASE:
            stranded.append(key)
            print(f"STRANDED_CASE {key} :: {detail}")
        # Both named from the row text, before and independent of the `if state` chain: a key can be
        # unrunnable AND leak prose, and the two facts need different fixes.
        stored_cmd = table[key].get("evidence_cmd") or ""
        # Input-lost is decided FIRST, and the two are then mutually exclusive by that order rather
        # than by luck (#2339): a row whose candidate pattern names no file is not counting a
        # candidate whole, it is counting nothing, and printing `WHOLE_CANDIDATE_COUNT` for it points
        # the repair at a missing `awk` when the repair is a re-mint. The one fact that must not
        # follow the other silently is the exit code — see the tail of this function.
        lost = candidate_input_lost(stored_cmd)
        if lost is not None:
            lost_rows.append(key)
            print(f"INPUT_LOST {key} :: {lost}")
        else:
            leak = candidate_body_defect(stored_cmd)
            if leak is not None:
                (whole if leak[0] == "whole_file" else dead_strip).append(key)
                print(f"{'WHOLE_CANDIDATE_COUNT' if leak[0] == 'whole_file' else 'DEAD_FRONT_MATTER_STRIP'}"
                      f" {key} :: {leak[1]}")
    # The exit has SIX terms and only two of them are figures the nightly runbooks carry: a run that
    # reports `ledger_unrunnable: 0` beside a non-zero exit has nothing on stdout saying what fired.
    # #2343 measured it live — `keys: 116 unrunnable: 0`, `denominators: empty_input 0 undeclared 102`
    # and exit 1, driven entirely by `input_lost 6`. So the drivers name themselves, one line per run,
    # spelled with the tokens the tally lines ABOVE and BELOW this one already publish (`input_lost`,
    # not `lost_rows`; `unrunnable`, not `dead`), because a reader who has the scoping line can tie
    # `input_lost 6` to the figure in it without knowing this function's local names.
    # ABOVE `candidate_body_scoping:` is the only legal slot. The lines a nightly parses are pinned by
    # INDEX, not by search: `keys:` at `splitlines()[-1]`, `denominators:` at `-2`, `stranded:` at
    # `-3`. A new tally therefore enters at the top of the block — displacing nothing below `-3` and
    # moving `candidate_body_scoping:` to `-5` — which is the same rule #2103 and #2166 arrived at.
    # This tuple and the return at the end are ONE fact, not two lists kept in step by hand: a ledger
    # cannot exit 1 on a term this line does not name, and cannot name a driver on a green run.
    # `undeclared` is absent on purpose — published, but never a reason to fail a run (#2052).
    exit_terms = (("unrunnable", len(dead)),
                  ("empty_input", len(empty_input)),
                  ("case_sensitive_grep", len(stranded)),
                  ("whole_file", len(whole)),
                  ("dead_strip", len(dead_strip)),
                  ("input_lost", len(lost_rows)))
    drivers = " ".join(f"{token} {count}" for token, count in exit_terms if count)
    print(f"exit_drivers: {drivers or 'none'}")
    # ABOVE `stranded:`, not below it: #2103's test pins `splitlines()[-3]` to that line, and
    # the rule that lands in #2166's own test is that a new tally arrives above the published
    # figure without moving any of the ones below it. `input_lost` is APPENDED to this line rather
    # than inserted into it for the same reason: a nightly that parses `whole_file (\d+)` out of it
    # keeps parsing it over a ledger where the third figure appeared.
    print(f"candidate_body_scoping: whole_file {len(whole)} dead_strip {len(dead_strip)} "
          f"input_lost {len(lost_rows)}")
    print(f"stranded: case_sensitive_grep {len(stranded)}")
    print(f"denominators: empty_input {len(empty_input)} undeclared {undeclared}")
    print(f"keys: {len(table)} unrunnable: {len(dead)}")
    # Still the six-term rule of #2048/#2166/#2339, read off `exit_terms`: `lost_rows` is one of the
    # six because the change that removed those four rows from `whole` is the same change that would
    # otherwise have flipped tonight's audit from exit 1 to exit 0 — excluded from one figure and
    # absent from the rest is a ledger that reports `unrunnable: 0` over falsifiers which can no
    # longer see their input, which is the exact sentence #2048 was raised to make impossible. A
    # ledger whose ONLY defect is input-lost falsifiers has to stay red, and since #2343 it also has
    # to SAY so: reading the exit and the driver line off the one tuple is what keeps a term from
    # entering the expression and being forgotten on stdout, which is the gap #2343 was filed for.
    return 1 if drivers else 0


def _read_reanchors(path: str | None) -> dict[str, str]:
    """Parse the `--reanchor-file` map: a JSON object of pattern_key -> command.

    A file rather than a flag because these commands are the long ones — a grep over an
    installed `SKILL.md` with several counts quoted, which no shell line carries
    readably. Blank values are dropped: an author who leaves a key's value empty meant to
    get to it, and appending an empty command is what `record_verdict` refuses.
    """
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--reanchor-file must hold a JSON object of key -> command")
    return {str(k): str(v) for k, v in raw.items() if str(v).strip()}


def cmd_reanchor(args: argparse.Namespace) -> int:
    """Append one hand-authored falsifier, keeping that key's decision.

    The single-key lane for the shapes `repair` cannot substitute: the dead path inside a
    mirrored script, a glob inside quotes, or a field that holds prose. Same guard as the
    pass — `reanchor_verdict` runs the command through `evidence_cmd_status` first and
    refuses it if the classifier calls it dead — so the answer to "did you re-anchor this
    against something that exists?" is the exit code, not a claim.
    """
    try:
        row = reanchor_verdict(args.store, pattern_key=args.pattern,
                               evidence_cmd=args.evidence_cmd, timeout=args.timeout,
                               dry_run=args.dry_run)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    tail = (f"observed: {row.get('evidence_observed') or 'nothing'}" if not args.dry_run
            else "nothing written")
    print(f"{'would re-anchor' if args.dry_run else 're-anchored'} {row['pattern_key']}"
          f" -> {row.get('verdict')} (decision kept, occurrences "
          f"{row.get('occurrences_at_decision')}) {tail}")
    return 0


def cmd_repair(args: argparse.Namespace) -> int:
    """Append the corrections a moved data root broke, and name what it cannot fix.

    `audit` gives one number; the repair needs three, because a dead falsifier has
    three causes and only one of them is a substitution. Measured on the live ledger
    2026-09-27, of the 79 keys `audit` reported: 46 name only the moved root and are
    correctable here; 30 name a *dated* input retention has since pruned, so no rewrite
    reaches a corpus that no longer exists anywhere on the box — 46 rather than the 31 in
    #1588's proving heredoc, because 15 of those were the same dead root inside the
    mirrored falsifier scripts, where a two-way substitution on the stored command
    cannot reach it and the fix was one `sed` over the script instead; 3 carry prose in
    the field, written before `record` refused one.

    So this subcommand does the first class mechanically, appends the second through
    `--reanchor-file` where a person has written the command, and refuses to guess at the
    rest. It never edits a line: each correction is a fresh append through
    `record_verdict`, which also refuses a command that observes nothing, so a repair
    cannot hand tomorrow's `audit` a fresh born-dead row (#1586's write-time half). What
    it reports as `NEEDS_RERECORD` is the honest remainder, each key tagged with the shape
    that decides who can fix it: `[nested_root]` and `[quoted_glob]` are a `sed` or a
    re-quote away, `[dated_corpus]` is retention's doing and only a re-derivation against
    an artifact that survives, or a disposal, answers it.

    Output shape is the contract, like `audit`'s: one finding line per key, then the
    tally as the LAST line, and exit 1 while any key is still outstanding — so a nightly
    that runs this cannot print a clean night over a ledger it did not finish. Under
    `--dry_run` the keys it *would* have written count as outstanding too: rehearsal zero
    has to mean "this pass would change nothing", or a preview and the real thing report
    the same success and only one of them is evidence.
    """
    store = store_path(args.store)
    rewrites = [tuple(r) for r in args.rewrite] if args.rewrite else None
    tally = repair_verdicts(store, dry_run=args.dry_run, timeout=args.timeout,
                            rewrites=rewrites, dispose=args.dispose_unverifiable,
                            reanchors=_read_reanchors(args.reanchor_file))
    for key in tally["repaired"]:
        print(f"REPAIRED {key}" + (" (dry-run)" if args.dry_run else ""))
    for key in tally["reanchored"]:
        print(f"REANCHORED {key}" + (" (dry-run)" if args.dry_run else ""))
    for key in tally["disposed"]:
        print(f"DISPOSED {key}" + (" (dry-run)" if args.dry_run else ""))
    for key in tally["needs_rerecord"]:
        print(f"NEEDS_RERECORD {key}")
    for key in tally["unparseable"]:
        print(f"UNPARSEABLE {key}")
    for line in tally["refused"]:
        print(f"REFUSED {line}")
    # Above the tally, not inside it: the last line's shape is this subcommand's contract
    # with the nightly (`splitlines()[-1]`), and #1717 asks only that the skipped half be
    # *named*. It is now named with the count of rows that went out single-tree, which is
    # the number the 2026-09-27 pass printed no line for while writing 79 of them.
    for line in tally["mirror_skipped"]:
        print(line)
    # Under `--dry-run` nothing was appended, so every key the pass would have written is
    # still outstanding — the same thing `audit` measures by. A caller that checks `$?`
    # has to be reading the ledger's state, not the rehearsal's politeness, or a dry run
    # over a ledger with a full ledger of keys to fix reports the same green as a
    # finished one.
    outstanding = (len(tally["needs_rerecord"]) + len(tally["unparseable"])
                   + len(tally["refused"])
                   + (len(tally["repaired"]) + len(tally["reanchored"])
                      + len(tally["disposed"]) if args.dry_run else 0))
    print(f"repaired: {len(tally['repaired'])}  reanchored: {len(tally['reanchored'])}  "
          f"disposed: {len(tally['disposed'])}  "
          f"needs_rerecord: {len(tally['needs_rerecord'])}  "
          f"unparseable: {len(tally['unparseable'])}  refused: {len(tally['refused'])}  "
          f"keys: {len(load_verdicts(store))}")
    return 1 if outstanding else 0


def cmd_sync(args: argparse.Namespace) -> int:
    """Bring the durable copy up to the live ledger, and name every key it copied.

    Why a route and not just a fix: the 79 rows #1717 is about are already on disk in one
    tree, and no amount of correct future writing puts them in the other. `check` now
    reports the two classes separately (`lagged:` and `disagreeing:` against
    `keys_compared:`) instead of 79 look-alike lines, and `record` now refuses to go quiet
    about a skipped copy — but the copy itself needs a re-runnable bring-up, because
    running it twice is the property that makes it safe to put in a nightly at all. It is
    idempotent: the second run finds each key's latest row already matching and adds 0
    lines.

    Output shape is the same contract as `repair`: one finding line per key, then the tally
    as the LAST line, and the tally repeats the same class words `check` prints so one
    number is not stated two ways by the two surfaces that compute it. Exit 1 on a refusal:
    a caller that asked for a sync and got none has to be able to tell from `$?`, which is
    exactly the mistake the 2026-09-27 pass made in the other direction (it wrote half the
    pairs and exited clean).
    """
    out = sync_durable_copy(args.store, dry_run=args.dry_run,
                            hold_disagreements=args.hold_disagreements)
    tag = " (dry-run)" if args.dry_run else ""
    for key_line in out["copied"]:
        print(f"SYNCED{tag} {key_line}")
    for key_line in out["held"]:
        print(f"SYNC_HELD {key_line} (left for a human: --hold-disagreements)")
    for line in out["refused"]:
        print(line)
    if out["refused"]:
        print("synced: 0  refused: 1  "
              f"durable: {out.get('durable', 'unresolved')}")
        return 1
    print(f"synced: {out['rows']}  lagged: {out[LAGGED]}  "
          f"disagreeing: {out[DISAGREEING]}  missing: {out['missing']}  "
          f"held: {len(out['held'])}  "
          f"durable: {out['durable']}")
    if not args.dry_run:
        # Re-read both trees after the append rather than reporting what the loop counted:
        # the claim this subcommand exists to make is that the two files agree *on disk*,
        # and only a fresh read of them says so.
        report = divergence_report(load_verdicts(store_path(args.store)),
                                   load_verdicts(Path(out["durable"])), Path(out["durable"]))
        print(mirror_tally_line(report))
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
    print(f"recorded {row['pattern_key']} -> {row['verdict']} at {row['decided_at']} "
          f"(occurrences {row['occurrences_at_decision']}, "
          f"observed: {row['evidence_observed']})")
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    """Harvest dispositions already on disk into the ledger, once.

    Reads the `status:` line of each candidate — the same hand-written verdicts the
    runbook used to re-make every night — and gives each one its re-executable check:
    the grep that re-derives that status from that file. Duplicate keys keep the
    newest file's disposition. Skips keys already in the ledger, so re-seeding after
    the key derivation changes is a re-run, not a cleanup.

    A merged candidate (#515: `signature_buckets` > 1) seeds `occurrences: 0`, not its
    `occurrences:` field: that number is the sum over every signature bucket behind the
    key, and a baseline recorded in those units would be compared forever after against
    counts recorded in one bucket's units. `reopen_reason` requires a baseline above 0,
    so a seeded merged key reopens on the 60-day expiry and not on a growth ratio it
    cannot measure — the same asymmetry `scan_candidates` and `verdict_for` apply.

    The filter below is `is_terminal`, so this harvest covers exactly the terminal set and
    nothing wider. That is what #830 unlocked: while `noise`, `consolidated` and
    `reviewed_authored` were absent from `TERMINAL_VERDICTS` a candidate carrying one of
    them could not be seeded at all, and the dispositions written by hand into
    frontmatter — roughly 125 keys on `status: noise`, measured at triage 2026-09-18 — had
    no path into the store that survives the next dated snapshot. Re-running this over
    `_pipeline/skills/candidates/` after that widening is what moves them in, and it is a
    human step because `_pipeline/` is gitignored and no round can commit its result.
    """
    table = load_verdicts(resolve_verdict_source(args.store)[0])
    newest: dict[str, tuple[Path, str, int]] = {}
    for file in sorted(Path(args.candidates).expanduser().glob("candidate-*.md")):
        key, status, occurrences, buckets = read_candidate(file)
        if key and is_terminal({"verdict": status}):
            prev = newest.get(key)
            if prev is None or file.name > prev[0].name:
                newest[key] = (file, status, 0 if buckets > 1 else occurrences)
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
        # Same waiver as #1588's two machine-written mints, and for the same reason: the
        # command below is this function's own template over one candidate file, so the
        # denominator it would print is a constant `1` that can never fall to 0 and would
        # buy the ledger a declaration that detects nothing. `audit` counts these rows
        # `undeclared`, which is the honest tally of a bootstrapped ledger (#2052).
        record_verdict(
            store=args.store,
            pattern_key=key,
            verdict=status,
            reason=reason or status,
            evidence_cmd=f"grep -m1 '^status:' {file}",
            occurrences=occurrences,
            decided_by="seed-from-candidates",
            source_candidate=file.name,
            require_input_rows=False,
            require_body_scope=False,
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
    p_record.add_argument(
        "--occurrences", type=int, default=None,
        help="pattern count at the decision, the baseline for the >10x growth reopen. "
             "Omit it on a correction: the count is carried forward from the row this "
             "line supersedes, so fixing a phrase cannot disarm that reopen (#736).")
    p_record.add_argument("--decided-by", dest="decided_by", default="agent")
    p_record.add_argument("--source-candidate", dest="source_candidate", default="")
    p_record.set_defaults(func=cmd_record)

    p_seed = with_store(sub.add_parser("seed", help="harvest existing candidate dispositions once"))
    p_seed.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    p_seed.set_defaults(func=cmd_seed)

    p_list = with_store(sub.add_parser("list", help="latest-wins view of the ledger"))
    p_list.set_defaults(func=cmd_list)

    # `audit` asks `list`'s subject a question `list` cannot: not what was decided,
    # but whether the decision can still be checked. `check` already knows how to ask
    # it (`evidence_cmd_status`) — of the two keys a scan happened to block.
    p_audit = with_store(sub.add_parser(
        "audit", help="re-run every stored evidence_cmd, name the keys whose check "
                      "cannot run, and exit 1 if any"))
    p_audit.add_argument("--timeout", type=int, default=EVIDENCE_TIMEOUT_SECONDS,
                         help=f"seconds per command before it counts as UNRUNNABLE "
                              f"(default {EVIDENCE_TIMEOUT_SECONDS})")
    p_audit.set_defaults(func=cmd_audit)

    # `audit` measures the damage a moved root did to stored falsifiers; `repair` fixes
    # the one cause a substitution can reach and prints the rest as work.
    p_repair = with_store(sub.add_parser(
        "repair", help="append the ledger corrections the moved data root broke, and "
                       "name the keys whose falsifier has to be re-derived"))
    p_repair.add_argument("--timeout", type=int, default=EVIDENCE_TIMEOUT_SECONDS,
                          help=f"seconds per command, same budget `audit` classifies "
                               f"with (default {EVIDENCE_TIMEOUT_SECONDS})")
    p_repair.add_argument("--dry-run", dest="dry_run", action="store_true",
                          help="classify and report, append nothing")
    p_repair.add_argument("--dispose-unverifiable", dest="dispose_unverifiable",
                          action="store_true",
                          help="also record a key whose measurement corpus is provably "
                               "gone: terminal verdicts become "
                               f"{UNVERIFIABLE_VERDICT} (still terminal, so nothing "
                               "unblocks), non-terminal verdicts keep theirs. Off by "
                               "default: it relabels decisions.")
    p_repair.add_argument("--rewrite", action="append", nargs=2, metavar=("OLD", "NEW"),
                          help="root spelling to substitute, repeatable; the default is "
                               "the pair `dead_root_spellings()` derives from app.paths")
    p_repair.add_argument("--reanchor-file", dest="reanchor_file", metavar="PATH",
                          help="JSON object of pattern_key -> hand-authored falsifier, "
                               "for the keys whose dead path sits inside a script or "
                               "inside quotes; each is executed before it is appended")
    p_repair.set_defaults(func=cmd_repair)

    # The bring-up half of #1717: `check` reports the two classes of divergence and
    # `record` no longer writes one tree silently, but the copy that is already behind
    # needs a route that closes the gap and can be run twice.
    p_sync = with_store(sub.add_parser(
        "sync", help="append to the durable copy the live ledger's latest-per-key rows it "
                     "does not hold; idempotent, and refuses a --store with no copy of its "
                     "own rather than writing through the production vault copy"))
    p_sync.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="name the keys it would copy and append nothing")
    p_sync.add_argument("--hold-disagreements", dest="hold_disagreements",
                        action="store_true",
                        help="copy lagged and missing keys but leave a key whose two trees "
                             "hold *different verdicts* for a human, instead of adopting "
                             "the live one as the durable answer")
    p_sync.set_defaults(func=cmd_sync)

    # The same guard as the pass, on one key, for the shapes the pass cannot substitute.
    p_reanchor = with_store(sub.add_parser(
        "reanchor", help="append a hand-authored falsifier for one key, keeping its "
                         "decision — for a dead path inside a script or inside quotes"))
    p_reanchor.add_argument("--pattern", required=True, help="pattern_key of the verdict")
    p_reanchor.add_argument("--evidence-cmd", dest="evidence_cmd", required=True,
                            help="the new check, executed verbatim before anything is written")
    p_reanchor.add_argument("--timeout", type=int, default=EVIDENCE_TIMEOUT_SECONDS)
    p_reanchor.add_argument("--dry-run", dest="dry_run", action="store_true",
                            help="execute the command and report, append nothing")
    p_reanchor.set_defaults(func=cmd_reanchor)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
