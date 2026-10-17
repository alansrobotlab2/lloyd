"""`architecture/qmd.md` §3's account of the embed backfill stays true after #1367.

The doc is where this item came from and where it will get re-filed from. §3
described `models: embed` as inert ("nothing re-embeds by itself") while pending
is counted per configured model, and once that was corrected (06ccbe28) the same
section described task #81's backfill as having "no ceiling" — accurate at the
time, and an open gap the moment the cap lands. An open gap in an architecture
doc is a to-do list, and a stale entry in one reads like a decision nobody made,
so the phrase is pinned out of the file here.

What replaces it is pinned the same way: the guard's name and its fraction are
asserted against `scripts/maintenance/qmd_index_maintenance.py`, so prose cannot
keep citing a constant that was renamed or re-tuned, and the doc's threshold and
the code's cannot drift apart.
"""

from __future__ import annotations

import hashlib
import json
import os
import pwd
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "qmd.md"
SCRIPT = ROOT / "scripts" / "maintenance" / "qmd_index_maintenance.py"
#: The other file that told a person to keep a backup the retention rule then deleted
#: (#2420). Read as text, exactly like `DOC`: what is pinned is the sentence.
EPISODIC_README = ROOT / "eval" / "episodic-arm" / "README.md"

#: The runtime record the #1992 paragraph cites as `$LLOYD_DATA/_pipeline/reflection/`.
#: Reached through the account's passwd entry, not `$HOME`: the gate runs the suite
#: with `$HOME` pointed at a round home where `~/lloyd-data` does not exist, which is
#: why `tests/conftest.py::_production_tree` reads the same anchor. The reports are
#: dated filenames, so the bytes a citation names do not move under a later reader.
REFLECTION = Path(pwd.getpwuid(os.getuid()).pw_dir) / "lloyd-data" / "_pipeline" / "reflection"

#: The one invocation that swapped, the one that verified without swapping, and the
#: two nightly before-snapshots either side of the swap.
SWAP_REPORT = "qmd-side-copy-rebuild-20261005-2026-10-05T072250.json"
REHEARSAL_REPORT = "qmd-side-copy-rebuild-20261004-2026-10-04T113501.json"
MAINTENANCE_BEFORE = "qmd-index-maintenance-2026-10-05.json"
MAINTENANCE_AFTER = "qmd-index-maintenance-2026-10-06.json"
#: The nightly that unlinked the two pre-wipe backups — the run this arm's README records
#: under "Found on the way", and the report it cites there. Committed for the first time by
#: #2539: until this round that citation resolved on no clone, only on the one box holding
#: `$LLOYD_DATA/_pipeline/reflection/`, which is the exact shape #2396 was cut to close.
MAINTENANCE_LOSS = "qmd-index-maintenance-2026-10-07.json"

#: The committed home of every report cited below. Before #2396 this file read all four
#: out of `REFLECTION` alone, which is not hermetic in either direction: the gate graded
#: the paragraph against bytes that are in no repository, and a plain clone, a second
#: account, or any host with no `_pipeline/reflection/` SKIPPED all three #2302 pins and
#: still reported a green suite — the shape #2282 was lost to, where the nodes never ran
#: and nothing between a pytest summary line and the rung's ceiling noticed. The runtime
#: record is now the control on these bytes, never their source.
FIXTURES = ROOT / "tests" / "fixtures"

#: Each cited report and the one scoped negation in `tests/fixtures/.gitignore` that
#: admits it past the root `*.json` rule. Both negations pre-date this change; an
#: `git add` of a name that no negation rescues exits 0 while staging NOTHING, so
#: `_ignore_rule` asks the real mechanism on every run (#2282 spent two review attempts
#: learning that).
CITED_REPORTS = {
    REHEARSAL_REPORT: "!qmd-side-copy-rebuild-*.json",
    SWAP_REPORT: "!qmd-side-copy-rebuild-*.json",
    MAINTENANCE_BEFORE: "!qmd-index-maintenance-*.json",
    MAINTENANCE_AFTER: "!qmd-index-maintenance-*.json",
    MAINTENANCE_LOSS: "!qmd-index-maintenance-*.json",
}

# Provenance of the five committed witnesses: the four #2396 copied byte-for-byte out of
# `$LLOYD_DATA/_pipeline/reflection/` on 2026-10-08, the day HEAD was `f3809a69`, and the
# fifth #2539 copied the same way on 2026-10-11, the day HEAD was `b2dedfe9`.
# Lines and bytes as `wc -l -c` prints them, then the md5:
#
#   rebuild-20261004-2026-10-04T113501   68 lines, 1,783 B  2957114779d1a5a7f7a9e6dc1b1a652c
#   rebuild-20261005-2026-10-05T072250   82 lines, 2,250 B  8af93f34848086e3b974cd5719bff0af
#   index-maintenance-2026-10-05        226 lines, 7,408 B  fc110dcbd5b248b5e4ccf0a7f3c9a175
#   index-maintenance-2026-10-06        243 lines, 7,989 B  9db5937166170b9cb342c1d0fff21b10
#   index-maintenance-2026-10-07        257 lines, 8,557 B  8f6e7b935d62ef681af2430b9a517b29
#
# Each name above is the tail of a constant in this file: the two rebuild reports are
# `REHEARSAL_REPORT` and `SWAP_REPORT`, the three nightlies `MAINTENANCE_BEFORE`,
# `MAINTENANCE_AFTER` and `MAINTENANCE_LOSS`.
#
# The 10-07 report has two further copies, not one: the board witness at
# `backlog/data/qmd-index-maintenance-2026-10-07.json` (vault commit `d0fc5ee4`), which
# #2539 clause 5 asked for and which `tests/test_qmd_index_maintenance.py` compares with
# these bytes through `board_presence.vault_root()` on the #1897 pattern, and nothing here.
# What `test_no_node_of_this_file_reaches_the_vault_copy` pins is this file's own route —
# the passwd anchor, which resolves even in a gate worktree and so would quietly grade a
# tree the gate does not run in — and it stays satisfied: no node of THIS file opens it.
#
# Those digests are bookkeeping, not an assertion. `_assert_matches_runtime` compares the
# committed bytes with the runtime copy whenever it can reach one, which is the
# family-7/family-9 convention in `tests/fixtures/.gitignore`: a compared byte is a better
# witness than a claimed one. The line and byte counts ARE asserted, by
# `test_the_three_new_cited_reports_are_exact_tracked_and_admissible`. The 10-04 report has
# one further copy, in the vault at `backlog/data/` under its own name (vault commit
# `53f68eaa`), as the human-facing witness clause 5 names; NO node opens it, and
# `test_no_node_of_this_file_reaches_the_vault_copy` keeps it that way.

#: The committed copy of the swap report, and the only place this file's backup name comes
#: from (#2323). A real `index.sqlite.bak-<stamp>` written into a scanned `.py` is a code
#: reference under `code_reference_hits`, and a code reference is a hold on the reclaim
#: list: on 2026-10-06 prose in this tree was keeping 2,577,506,304 B of retired backups
#: on disk, and a literal like the one this replaces was one of the namers — harmless only
#: while the file it named happened to be the newest copy in the directory, and a hold on
#: its own 990,171,136 B the moment a newer swap landed. `*.json` is outside
#: `CODE_REF_SUFFIXES`, so
#: the report that measured the name is where it lives, and the negation in
#: `tests/fixtures/.gitignore` is what admits these bytes past the root `*.json` rule.
SWAP_FIXTURE = FIXTURES / SWAP_REPORT

#: A sibling name with no negation of its own, the positive control for
#: `test_the_swap_fixture_is_admissible_and_tracked`: it must resolve to the root rule.
SWAP_FIXTURE_CONTROL = "tests/fixtures/side-copy-rebuild-no-negation-control.json"


def _fixture(name: str) -> Path:
    """A cited report's committed witness, resolved at call time.

    A function rather than four constants so a node can point `FIXTURES` at a directory
    one witness short and watch the resolution fail with a reason:
    `test_removing_a_witness_fails_the_three_pins_naming_the_missing_file` is the proof
    that these pins can fail at all, and a path baked in at import cannot be shown that.
    """
    return FIXTURES / name


def _sha(payload: bytes) -> str:
    """A short digest, for a failure message that has to tell two files apart."""
    return hashlib.sha256(payload).hexdigest()[:12]


def _ignore_rule(rel: str) -> tuple[int, str, str]:
    """Ask `git check-ignore` which rule wins for one repo-relative path.

    `--no-index -v` exits 0 for ANY matching rule, a negation included, and prints the
    winner as `source:line:pattern` — so the pattern half is the verdict and the exit
    status proves only that the walk ran. Hoisted out of
    `test_the_swap_fixture_is_admissible_and_tracked`, which #2323 wrote, so every
    witness in `CITED_REPORTS` is asked the same question the same way.
    """
    p = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--no-index", "-v", rel],
        capture_output=True, text=True)
    line = p.stdout.strip().split("\t")[0]
    source, _, pattern = line.rpartition(":")
    return p.returncode, source, pattern


def _swapped_backup() -> str:
    """The name of the database the 2026-10-05 swap moved aside.

    Read out of `SWAP_FIXTURE`'s bytes at call time rather than at import: an unreadable or
    untracked fixture has to fail a node with a reason, not turn this module into a
    collection error that the gate reports as a green suite whose nodes never ran (#2282).
    """
    return Path(json.loads(_fixture(SWAP_REPORT).read_text(encoding="utf-8"))
                ["swap"]["backup"]).name

#: The module constant the doc is allowed to name, and the only place the
#: threshold is defined. Read from the source text rather than imported so this
#: file does not depend on the module's import-time side effects (it stats the
#: live index at import in other contexts).
GUARD = "EMBED_PENDING_MAX_RATIO"


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def _section(heading: str) -> str:
    """From `heading` to the next `## ` line — one numbered section, sliced out."""
    text = _doc()
    i = text.index(heading)
    j = text.find("\n## ", i + len(heading))
    return text[i: j if j != -1 else len(text)]


def _reembed_bullet() -> str:
    """§3's bullet on changing the embed model — the one #1367 is about.

    Asserts its own absence, so a doc that reworded the heading fails here with a
    reason instead of at an index.
    """
    body = _section("## 3. Models")
    hits = [b for b in body.split("\n- ") if "full re-embed" in b]
    assert len(hits) == 1, f"expected exactly one 'full re-embed' bullet in §3, got {len(hits)}"
    return hits[0]


def _script_constant() -> float:
    hits = re.findall(rf"^{GUARD} = ([0-9.eE+]+)$", SCRIPT.read_text(encoding="utf-8"), re.M)
    assert len(hits) == 1, f"{SCRIPT.name}: expected exactly one {GUARD} assignment, got {hits}"
    return float(hits[0])


# --- clause 4: the doc stops describing the backfill as uncapped --------------

def test_the_doc_no_longer_describes_the_backfill_as_having_no_ceiling():
    """"no ceiling" was true on 2026-09-22 and must not survive the cap.

    Checked over the whole file, review log included: a dated entry that still
    says the job is uncapped is the sentence an architecture review re-files the
    item from, and the review log is read as a description of the job as often as
    it is read as history.
    """
    text = _doc()
    assert "no ceiling" not in text, "the doc still describes #81's backfill as uncapped"
    assert "uncapped" not in text.lower(), (
        "the backfill is described as uncapped somewhere in the doc")


def test_the_doc_names_the_guard_and_its_threshold_fraction():
    """§3 now says what refuses, by the name the code uses, with the fraction."""
    bullet = _reembed_bullet()
    assert GUARD in bullet, "§3 does not name the constant that caps the backfill"
    assert "model_change_suspected" in bullet, (
        "§3 names the cap but not the finding the run records, which is the half "
        "a reader of the report is looking for")
    assert re.search(r"0\.25|quarter", bullet), "§3 does not state the threshold fraction"


def test_the_docs_threshold_is_the_ones_the_code_carries():
    """The fraction in prose and the constant in the module are one number.

    Two places that can be edited separately is how a doc ends up describing a
    threshold nobody tuned.
    """
    code_value = _script_constant()
    assert code_value == 0.25, f"{SCRIPT.name} retuned the cap; update §3 in the same change"
    assert f"{code_value:.2f}" in _reembed_bullet() or "quarter" in _reembed_bullet()


def test_the_guard_is_attributed_to_the_job_that_runs_it_not_the_watcher():
    """The sentence that names the cap must not credit the watcher with it.

    #1367 capped task #81's backfill and deliberately not
    `agent-services/scripts/qmd-watcher.sh`, which embeds every cycle and is
    still the first responder to an embed-model edit. Prose that says otherwise
    would tell the next operator the machine is covered when it is not. A round
    that caps the watcher too rewrites this sentence and this test together.
    """
    for sentence in re.split(r"(?<=[.!?])\s+", _reembed_bullet()):
        if GUARD in sentence:
            assert "watcher" not in sentence.lower(), sentence


def test_the_doc_still_says_which_path_is_left_unprotected():
    """The residual gap is stated, because the fix is partial by decision.

    Only the mutating branch of the maintenance job embeds, and only one of the
    two unattended surfaces is capped; a doc that reads as "fixed" here is how
    the watcher's path survives the next six months.
    """
    bullet = _reembed_bullet()
    assert "watcher" in bullet.lower(), "§3 no longer says which embed path is uncapped"


def test_section_8_lists_the_test_that_pins_this_claim():
    """The doc's own test index carries this file, so the claim is findable."""
    assert "test_qmd_doc_claims.py" in _section("## 8.")


# --- the config the guard reads is the config the doc describes ---------------

def test_the_embed_model_the_live_config_template_names_still_parses_the_way_the_guard_reads_it():
    """`models: embed` is a mapping key, not a scalar under some other spelling.

    The guard's only useful field in the report is the model name, and it comes
    from `data["models"]["embed"]` in the file the daemon reads. The tracked
    template is byte-identical to that file today on this key, so it is the
    checkable stand-in — a rename upstream (a `model:` list, a `embedding:`
    nesting) would leave the guard reporting None for every run, which is the
    failure the report is supposed to make visible rather than cause.
    """
    cfg = yaml.safe_load(
        (ROOT / "agent-services" / "conf" / "qmd-index.yml").read_text(encoding="utf-8"))
    models = cfg["models"]
    assert isinstance(models, dict) and isinstance(models.get("embed"), str) and models["embed"]


# --- the #1992 route is described by what it did, not by what was owed --------
#
# "Not yet run for real" was true the day #1992 landed the route and stayed in the
# doc for five days after a real invocation swapped a live index, because the
# sentence was written beside the code and nothing re-read it against the reports
# the route itself writes. So the paragraph's facts are not restated as constants
# here: each one is read out of the report the citation names, and the citation is
# resolved on disk.

def _flat(text: str) -> str:
    """Prose with its line wraps folded to single spaces, so an assertion is about
    the sentence and not about where the file happened to break it."""
    return " ".join(text.split())


def _task81_bullet() -> str:
    """§5's `Task #81` bullet — the list item that carries the #1992 account.

    Splitting on `"\\n- "` works because the bullet's continuation lines are
    indented, so one item is one chunk; the assertion is the node's own absence
    check, so a reworded heading fails here with a reason instead of at an index.
    """
    hits = [b for b in _doc().split("\n- ") if "**Task #81**" in b]
    assert len(hits) == 1, f"expected exactly one Task #81 bullet, got {len(hits)}"
    return hits[0]


def _assert_matches_runtime(name: str, fixture: Path) -> None:
    """Establish the witness's provenance by comparison, when the runtime copy is reachable.

    The `SWAP_FIXTURE` idea of #2323 generalised to the whole cited set: those bytes are
    the only place in this tree allowed to carry a real `index.sqlite.bak-<stamp>` name,
    and the four figures the paragraph quotes are only the figures the invocations measured
    if the committed copy is the report itself. The runtime record is the machine's other
    copy of the same invocation and the only one that can say so, so agreement is asserted
    rather than a digest quoted — that is what makes the md5s in the comment above
    bookkeeping instead of a claim. A dated report is immutable, so the two can only
    disagree if one of them has been edited, moved, or truncated: exactly the moment the
    pins stop meaning anything.

    An unreachable runtime copy returns quietly. It is not a skip and not a failure: the
    committed bytes are the witness, and a host without this box's `_pipeline/reflection/`
    still has to be able to grade the paragraph, which is the whole reason #2396 exists.
    """
    runtime = REFLECTION / name
    if not runtime.is_file():
        return
    committed, live = fixture.read_bytes(), runtime.read_bytes()
    if committed == live:
        return
    raise AssertionError(
        f"the committed witness {fixture} (sha256 {_sha(committed)}, {len(committed)} B) "
        f"and the runtime report {runtime} (sha256 {_sha(live)}, {len(live)} B) disagree, "
        f"so neither one is the bytes the `{name}` invocation wrote and no figure quoted "
        "from either re-derives anything")


def _report(name: str) -> dict:
    """One cited report, read from its committed bytes.

    The witness is the fixture in `tests/fixtures/`; the runtime record the doc's citation
    points at is its control, checked by `_assert_matches_runtime`. This function used to
    open `REFLECTION / name` and SKIPPED when it was absent, which graded the paragraph
    against bytes in no repository on a box that had them and asserted nothing on a box
    that did not — three of the four cited reports had no committed copy at all until
    #2396. A missing witness now FAILS with its own path: the figure would then be in
    nothing, and a pin that skips instead of saying so is the false green this file exists
    to avoid.
    """
    fixture = _fixture(name)
    if not fixture.is_file():
        pytest.fail(
            f"no committed witness at {fixture}. The figure the paragraph cites would then "
            "live only in one machine's `_pipeline/reflection/`, so the clause cannot be "
            "re-derived: restore the dated report (immutable since the invocation that wrote "
            "it) — skipping here is how all three #2302 clauses read as satisfied on a clone.")
    _assert_matches_runtime(name, fixture)
    return json.loads(fixture.read_text(encoding="utf-8"))


def test_the_doc_no_longer_says_the_route_was_never_run_for_real():
    """#2302 clause 1: the stale claim is gone over the whole file.

    Checked file-wide, review log included, because a dated entry is re-read as a
    description of the job as often as it is read as history — and it is the
    sentence an architecture review re-files this item from.
    """
    bullet = _task81_bullet()
    assert "Not yet run for real" not in _doc(), (
        "the doc still says the rebuild-and-swap route has never been run, which "
        "the 2026-10-05 report contradicts")
    # Positive control: the passage is still there and still about the same
    # question, so the absence above is a rewrite and not a deletion.
    assert "**Task #81**" in bullet and "1801" in bullet


def test_the_paragraph_reports_the_swap_and_cites_the_report_behind_it():
    """#2302 clause 2: the run that happened is stated, with its citation.

    The seam is prose to witness record: the paragraph is checked against the committed
    copy of the report it points at, which #2323 chose as the one place in this tree
    allowed to name a real backup, and `_report` has just proved that copy agrees with the
    runtime one on any machine that still has it. A citation the repository cannot resolve
    is the failure mode being closed here: `wc -l` of a directory nobody commits is not a
    check a later reader can run.
    """
    swap = _report(SWAP_REPORT)
    flat = _flat(_task81_bullet())
    cites = re.findall(
        r"\$LLOYD_DATA/(_pipeline/reflection/qmd-side-copy-rebuild-2026[^` ]+)", flat)
    assert cites, "the paragraph cites no report under _pipeline/reflection/"
    assert any(c.endswith(SWAP_REPORT) for c in cites), cites
    for cite in cites:
        assert _fixture(Path(cite).name).is_file(), (
            f"the paragraph cites {cite}, which has no committed witness under "
            f"{FIXTURES}: the claim is then checkable only on a box with that runtime "
            "directory, and a reader with the repository alone cannot re-derive it")
    assert swap["swap"]["swapped"] is True and swap["swap"]["retrieval_ok"] is True, swap["swap"]
    assert swap["name"] == "rebuild-20261005" and "2026-10-05" in flat
    assert "swapped" in flat and "retrieval_ok" in flat, (
        "the paragraph no longer states what the swap report records")
    # The backup name comes out of the committed bytes, and the agreement between those
    # bytes and the runtime report the citation names is now asserted where the report is
    # resolved (`_assert_matches_runtime`, #2396 clause 4) rather than re-derived field by
    # field here: comparing the whole payload says everything a single field compared did,
    # and says it for all four cited reports instead of the one that happened to be read
    # first. What is this node's own is the prose: the name has to be in the paragraph.
    committed = _swapped_backup()
    assert committed in flat, "the backup the swap left is not named"


def test_the_reclaim_figures_are_the_ones_the_nightly_series_measured():
    """#2302 clause 3: 445.3 MiB dead at 0.2723 became 37.0 MiB at 0.822.

    Both pairs are read out of the two nightly before-snapshots the paragraph
    attributes them to, in that order, and the direction is checked too — a
    paragraph that reported the two mornings the wrong way round would still
    contain all four numbers. The route report's own `live_before` pair and the
    live-row gap between the two reads are pinned beside them.
    """
    before = _report(MAINTENANCE_BEFORE)["before"]["vec0"]
    after = _report(MAINTENANCE_AFTER)["before"]["vec0"]
    live_before = _report(SWAP_REPORT)["live_before"]["vec0"]
    flat = _flat(_task81_bullet())
    assert before["dead_mib"] > after["dead_mib"], "the swap did not reclaim anything"
    assert before["occupancy"] < after["occupancy"], "occupancy did not rise"
    for fig in (before["dead_mib"], before["occupancy"],
                after["dead_mib"], after["occupancy"],
                live_before["dead_mib"], live_before["occupancy"]):
        assert repr(fig) in flat, f"the doc does not state the measured {fig}"
    assert MAINTENANCE_BEFORE in flat and MAINTENANCE_AFTER in flat, (
        "the four figures are not attributed to the two nightly reports")
    assert flat.index(MAINTENANCE_BEFORE) < flat.index(MAINTENANCE_AFTER), (
        "the before/after pair is presented in the wrong order")
    gap = live_before["live_rows"] - before["live_rows"]
    assert f"{gap:,} more live rows" in flat, f"the two reads are {gap} rows apart"


def test_the_paragraph_keeps_both_caveats_and_says_no_clock_window_applies():
    """#2302 clause 4: the two mechanisms that survived the first real run, and
    the absence of any hour-of-day rule.

    The retry is pinned to the script's own constants and to the two embed passes
    each run logged, so the sentence cannot outlive the mechanism; the `.bak`
    claim is pinned to the retention series, which must still hold the swap's
    backup as its newest member; and the clock-window sentence is pinned against
    the one thing that would contradict it — a branch on the hour in the script.
    """
    flat = _flat(_task81_bullet())
    assert "embed lock is per directory" in flat and "exclude each other" in flat
    assert "the route retries" in flat, "the retry caveat was dropped"
    src = SCRIPT.read_text(encoding="utf-8")
    assert "SIDE_EMBED_ATTEMPTS" in src and "SIDE_EMBED_RETRY_SLEEP_S" in src, (
        "the retry the paragraph describes is no longer a constant in the script")
    for name in (REHEARSAL_REPORT, SWAP_REPORT):
        passes = [a for a in _report(name)["actions"] if a.startswith("embed #")]
        assert len(passes) == 2, f"{name} logged {len(passes)} embed passes"
    assert "`.bak` series" in flat and "bounds to its newest" in flat, (
        "the backup-retention caveat was dropped")
    series = _report(MAINTENANCE_AFTER)["stray_retention"]["bak_series"]
    assert series[0] == _swapped_backup(), (
        f"the swap's backup is not the series' newest: {series}")
    assert "no clock window applies" in flat, "the paragraph stopped saying the route has no window"
    assert "22:00" not in flat and "04:00" not in flat, "a clock window is implied again"
    assert ".hour" not in src, (
        "qmd_index_maintenance.py now branches on the hour, and the paragraph's "
        "'no clock window applies' has to be rewritten in the same change")


def test_the_swap_fixture_is_admissible_and_tracked():
    """The bytes this file's backup name comes from are really in the index.

    Root `.gitignore` ends `*.json`, so a fixture under `tests/fixtures/` is admissible only
    through one scoped negation in `tests/fixtures/.gitignore` — and `git add` of an ignored
    path is SILENT: it exits 0, stages nothing, and a round believes it committed the
    witness. #2282 lost two review attempts exactly that way, and the shape of the failure
    is what makes a node like this necessary rather than paranoid: an absent file read at
    import is a collection error, and the suite stays green with the file's nodes never
    executed. `_swapped_backup()` is a call rather than a constant so the failure lands on a
    node with a reason instead of on collection.

    Ask the real mechanism, and read what it actually reports: `git check-ignore
    --no-index -v` exits 0 for ANY matching rule, a negation included, and prints the winner
    as `source:line:pattern` — so the pattern half is the verdict, not the exit status. The
    control sibling has no negation of its own and must come back with the root rule, or a
    clean answer for the fixture would mean nothing.
    """
    code, source, pattern = _ignore_rule(SWAP_FIXTURE_CONTROL)
    assert code == 0 and pattern == "*.json" and not source.startswith("tests/fixtures/"), (
        f"positive control broken: a sibling name with no negation resolved to "
        f"{source!r} / {pattern!r} (exit {code}), so the root rule is not what is being "
        "beaten and a clean answer below would prove nothing")

    rel = SWAP_FIXTURE.relative_to(ROOT).as_posix()
    code, source, pattern = _ignore_rule(rel)
    assert code == 0 and source.startswith("tests/fixtures/.gitignore") \
        and pattern == "!qmd-side-copy-rebuild-*.json", (
        f"{rel} resolves to {source!r} / {pattern!r} (exit {code}): the scoped negation is "
        "not the last matching rule, so `git add` would stage nothing and the report this "
        "file reads its name from would be missing from every gate tree")

    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel],
        capture_output=True, text=True)
    assert tracked.returncode == 0, (
        f"{rel} is admissible but not in the index: {tracked.stderr.strip()} — the name the "
        "doc claim depends on has no committed source")

    raw = SWAP_FIXTURE.read_text(encoding="utf-8")
    assert raw.count("\n") == 82 and len(raw.encode()) == 2250, (
        "the committed swap report is not the bytes the 2026-10-05 invocation wrote "
        f"({raw.count(chr(10))} lines, {len(raw.encode())} B)")


def test_the_three_new_cited_reports_are_exact_tracked_and_admissible():
    """#2396 clauses 1 and 2: three more witnesses are the bytes, in the index, admitted
    by rules that already existed.

    Until #2396 only the swap report of the four the paragraph cites was committed, so
    `_report` was reading three reports out of a directory that is in no repository while
    looking like a passing pin. Byte counts are the cheap half and they are pinned anyway:
    a dated report is immutable, so `wc -l -c` of the runtime file is a figure a later
    reader can re-derive, and the agreement of the whole payload is asserted by
    `_assert_matches_runtime` on any machine holding the runtime copy. The expensive half
    is admission — root `.gitignore` ends `*.json`, and `git add` of a name no negation
    rescues exits 0 having staged nothing, which is the silent failure #2282 spent two
    review attempts on. So each new name goes through `_ignore_rule` with the control
    sibling beside it, and the whole tracked family is counted, because "6" is the number
    `git ls-files tests/fixtures | grep -c qmd` prints once the four #2302 witnesses,
    #2119's nightly and #2539's loss report are all in the index — a family count, so a
    round that commits a seventh witness widens this number and its own clause together.
    """
    exact = {  # name -> (lines, bytes), as `wc -l -c` of the runtime report prints them
        REHEARSAL_REPORT: (68, 1783),
        MAINTENANCE_BEFORE: (226, 7408),
        MAINTENANCE_AFTER: (243, 7989),
        MAINTENANCE_LOSS: (257, 8557),
    }
    for name, (lines, size) in exact.items():
        fixture = _fixture(name)
        raw = fixture.read_text(encoding="utf-8")
        assert raw.count("\n") == lines and len(raw.encode()) == size, (
            f"the committed {name} is {raw.count(chr(10))} lines / {len(raw.encode())} B, "
            f"not the {lines} lines / {size} B that "
            f"`wc -l -c $LLOYD_DATA/_pipeline/reflection/{name}` prints")

        rel = fixture.relative_to(ROOT).as_posix()
        code, source, pattern = _ignore_rule(rel)
        assert code == 0 and source.startswith("tests/fixtures/.gitignore") \
            and pattern == CITED_REPORTS[name], (
            f"{rel} resolves to {source!r} / {pattern!r} (exit {code}) instead of the "
            f"pre-existing negation {CITED_REPORTS[name]!r}: `git add` would have staged "
            "nothing here, and the witness the paragraph's figures come from would be "
            "absent from every gate tree while the suite stayed green about it")

        tracked = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel],
            capture_output=True, text=True)
        assert tracked.returncode == 0, (
            f"{rel} is admissible but not in the index: {tracked.stderr.strip()} — an "
            "untracked witness is a file only this worktree has")

    code, source, pattern = _ignore_rule(SWAP_FIXTURE_CONTROL)
    assert code == 0 and pattern == "*.json" and not source.startswith("tests/fixtures/"), (
        f"positive control broken: a sibling with no negation of its own resolved to "
        f"{source!r} / {pattern!r} (exit {code}), so the root rule is not what the four "
        "negations above are beating and their clean answers prove nothing")

    listed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "tests/fixtures"],
        capture_output=True, text=True)
    qmd = sorted(ln for ln in listed.stdout.split("\n") if "qmd" in ln)
    assert len(qmd) == 6, (
        f"`git ls-files tests/fixtures | grep -c qmd` prints {len(qmd)} ({qmd}), not 6: "
        "four cited reports plus #2119's nightly witness plus #2539's loss report is the "
        "whole committed family, and one fewer means a figure some paragraph quotes is in "
        "no repository")

    spec = (ROOT / "scripts" / "automod" / "spec.py").read_text(encoding="utf-8")
    denied = spec.split("DENIED_GLOBS", 1)[1].split(")", 1)[0]
    assert '".gitignore"' in denied, (
        "the round's scope check no longer refuses `.gitignore` outright. That refusal is "
        "what makes 'no ignore file was edited to admit these bytes' a standing property "
        "rather than a promise about this one diff: both negations above are pre-existing "
        "lines, and a round wanting a third rule would have to change a file it cannot.")


def test_the_cited_reports_resolve_to_committed_bytes_and_nothing_in_this_file_skips(
        monkeypatch, tmp_path):
    """#2396 clause 3: the four names come out of the repository, and no node may skip.

    Pointing `REFLECTION` at an empty directory is not a hypothetical here — it is exactly
    what a plain clone, a second account, or any host that has never run the rebuild route
    is. Every cited figure still has to resolve, or the pin was only ever as durable as one
    machine's `_pipeline/reflection/`. The file-level absence is pinned with it because
    `pytest`'s own ceiling is not a floor: `scripts/automod/gate.py::PYTEST_MAX_SKIPPED`
    allows 40 skips against a suite already sitting near 31, so a fourth silent-green shape
    fits inside the budget permanently. Assembled at call time so this node adds no fresh
    copy of the token it is looking for.
    """
    src = Path(__file__).read_text(encoding="utf-8")
    skip_call = "pytest." + "skip"
    assert skip_call not in src, (
        f"a {skip_call} call is back in this file. A node that skips grades nothing, and "
        "on a host without the runtime record that is how all four #2302 clauses read as "
        "satisfied while asserting nothing.")
    assert "skip" in src.lower(), (
        "the word is gone from this file entirely, which is how the absence above would "
        "also be satisfied by deleting the record of what it retires")

    unreachable = tmp_path / "no-runtime-record"
    unreachable.mkdir()
    monkeypatch.setattr(sys.modules[__name__], "REFLECTION", unreachable)
    for name in CITED_REPORTS:
        resolved = _report(name)
        assert resolved == json.loads(_fixture(name).read_text(encoding="utf-8")), (
            f"_report({name!r}) did not return the committed witness with the runtime "
            "record unreachable, so it is still reading the machine rather than the tree")
    assert _report(MAINTENANCE_BEFORE)["before"]["vec0"]["dead_mib"] == 445.3, (
        "the 10-05 nightly's measured dead figure is not what the paragraph quotes")


#: The three #2302 pins, in file order: the whole set the failure-half node below runs
#: against a witness directory that is one file short. Named explicitly rather than
#: collected from this module's globals, so the denominator is the three clauses of #2302
#: and not whatever nodes happen to exist the day someone adds a fourth.
_PINS = (
    test_the_paragraph_reports_the_swap_and_cites_the_report_behind_it,
    test_the_reclaim_figures_are_the_ones_the_nightly_series_measured,
    test_the_paragraph_keeps_both_caveats_and_says_no_clock_window_applies,
)


def test_removing_a_witness_fails_the_three_pins_naming_the_missing_file(monkeypatch, tmp_path):
    """#2396 clause 3, the failure half: a pin that cannot fail is not a pin.

    Each of the three #2302 pins is called directly against a fixture directory holding the
    other three witnesses and missing exactly the one that pin resolves — which is what a
    tree where `git add` silently refused to stage one file is. Each has to fail with the
    missing path in its reason, and none may SKIPPED instead: that substitution is the whole
    defect #2396 closes, since skipping on a box with no runtime record is how clauses 2, 3
    and 4 reported green for a month without reading anything.

    Which pin has to fall for which witness is the pairing clause 3 states; the rest is
    observed rather than assumed, because the three pins share witnesses (the reclaim pin
    also reads the swap report's own `live_before`, the caveats pin both rebuild reports) and
    a hand-written map of who reads what would be one more claim nobody re-runs. So every
    pin is run against every partial directory: none may SKIP, the named pin must fail with
    the removed path in its reason, and any other pin that fails must fail about that same
    file. At least one pin must still pass somewhere — the redirected `FIXTURES` is doing the
    failing here, and a redirect that broke every pin in every case would say so by leaving
    no pass at all.
    """
    must_fail = {  # the witness, and the pin clause 3 names for it
        SWAP_REPORT: test_the_paragraph_reports_the_swap_and_cites_the_report_behind_it,
        MAINTENANCE_BEFORE: test_the_reclaim_figures_are_the_ones_the_nightly_series_measured,
        REHEARSAL_REPORT:
            test_the_paragraph_keeps_both_caveats_and_says_no_clock_window_applies,
    }
    assert len(must_fail) == 3 and set(must_fail.values()) <= set(_PINS), (
        "the clause-3 pairings are not the three pins of #2302")
    # Assembled so this node adds no fresh copy of the token the node above is looking
    # for: `pytest` + `.skip` is a reference, writing it out is the string itself.
    skip_exception = getattr(pytest, "skip").Exception
    # Read before the first redirect: `FIXTURES` is the module attribute the loop below
    # rebinds, so resolving the copies through `_fixture()` from the second iteration on
    # would read them out of the previous case's one-file-short directory.
    committed = FIXTURES
    passes = 0
    for missing, required in must_fail.items():
        partial = tmp_path / f"fixtures-without-{missing}"
        partial.mkdir()
        for other in CITED_REPORTS:
            if other != missing:
                (partial / other).write_bytes((committed / other).read_bytes())
        assert len(list(partial.iterdir())) == len(CITED_REPORTS) - 1, (
            "the witness directory was not built one file short of the cited set, so the "
            "failure below could have any cause")
        monkeypatch.setattr(sys.modules[__name__], "FIXTURES", partial)
        seen = {}
        for node in _PINS:
            try:
                node()
            except BaseException as exc:        # noqa: BLE001 - the outcome IS the data
                seen[node.__name__] = exc
            else:
                passes += 1
        skipped = [n for n, exc in seen.items() if isinstance(exc, skip_exception)]
        assert not skipped, (
            f"{skipped} skipped with `{missing}` absent instead of failing it: a skip is "
            "exactly the substitution #2396 exists to close")
        assert missing in str(seen), (
            f"no pin failed about the removed witness ({missing}); they failed about "
            f"{[str(exc)[:120] for exc in seen.values()]}")
        assert isinstance(seen.get(required.__name__),
                          (pytest.fail.Exception, AssertionError)), (
            f"{required.__name__} did not fail on an absent `{missing}`: "
            f"{seen.get(required.__name__, 'it passed')}")
        assert str(partial / missing) in str(seen[required.__name__]), (
            f"{required.__name__} failed without naming the witness it could not read "
            f"({partial / missing}): {seen[required.__name__]}")
    assert passes, (
        "no pin passed in any of the four one-witness-short trees, so what is failing is "
        "the redirected FIXTURES and not the missing bytes")


def test_a_reachable_runtime_copy_has_to_agree_with_the_committed_witness(
        monkeypatch, tmp_path):
    """#2396 clause 4: provenance is a comparison, not a quoted digest.

    Three cases, because the code path has to be seen to run. The identical copy resolves
    (the positive control — without it, an `if runtime.is_file():` that never fired would
    look identical to one that always passed); a copy with one quoted figure changed fails,
    naming both files; and an absent copy is NOT a skip, because the committed witness is
    the durable one and a host without this box's runtime history still has to grade the
    paragraph. The digest pairs go into the failure message and nowhere else, so the md5s
    in the provenance comment above stay bookkeeping: a claimed digest proves only that a
    file is itself.
    """
    live = tmp_path / "reflection"
    live.mkdir()
    monkeypatch.setattr(sys.modules[__name__], "REFLECTION", live)
    target = _fixture(MAINTENANCE_BEFORE)

    (live / MAINTENANCE_BEFORE).write_bytes(target.read_bytes())
    assert _report(MAINTENANCE_BEFORE) == json.loads(target.read_text(encoding="utf-8")), (
        "an agreeing runtime copy did not resolve to the committed witness")

    mutated = json.loads((live / MAINTENANCE_BEFORE).read_text(encoding="utf-8"))
    mutated["before"]["vec0"]["dead_mib"] = 445.4
    (live / MAINTENANCE_BEFORE).write_text(
        json.dumps(mutated, indent=2), encoding="utf-8")
    with pytest.raises(AssertionError) as excinfo:
        _report(MAINTENANCE_BEFORE)
    reason = str(excinfo.value)
    assert str(live / MAINTENANCE_BEFORE) in reason and str(target) in reason, (
        f"a disagreeing runtime copy failed without naming both files: {reason}")
    assert "445.3" not in reason, (
        "the failure quoted the committed figure as though it were the measured one: "
        f"{reason}")

    (live / MAINTENANCE_BEFORE).unlink()
    assert _report(MAINTENANCE_BEFORE)["before"]["vec0"]["dead_mib"] == 445.3, (
        "an unreachable runtime record stopped meaning 'no control available' and became a "
        "skip or a failure again, which is the silent green this file exists to avoid")


def test_no_node_of_this_file_reaches_the_vault_copy():
    """#2396 clause 5: the vault witness is for a person, and a node that opened it would
    grade a tree the gate does not run in.

    `backlog/data/qmd-side-copy-rebuild-20261004-2026-10-04T113501.json` is the third copy
    of these bytes — human-facing, checked by eye with `wc -l -c` (68 lines, 1,783 B), which
    is why the family comments in `tests/fixtures/.gitignore` all say the re-derive happens
    against the committed bytes. The route this node closes is the one that would quietly
    work here and fail everywhere else: the passwd anchor that `REFLECTION` uses reaches
    `~/obsidian` too, from inside a node, in a gate worktree whose `$HOME` is a round home
    where the vault does not exist. One anchor, one purpose.
    """
    src = Path(__file__).read_text(encoding="utf-8")
    anchor = "getpw" + "uid"   # assembled, so this node is not the second occurrence
    assert src.count(anchor) == 1, (
        "a second passwd-derived home anchor appeared in this file. The one that exists is "
        "`REFLECTION`, and the other thing that anchor reaches is the vault — whose "
        "`backlog/data/` witness is the copy clause 5 says no test may read.")
    assert "backlog/data" in src, (
        "the vault witness is no longer named anywhere in this file, which is the other way "
        "the absence above could be satisfied: by forgetting the copy exists")


# ── #2420: the two backups the retention rule deleted, and the prose that outlived them ──
#
# Two files told a person to keep the pre-wipe session corpus, and neither could stop the
# rule that deleted it: `architecture/qmd.md` stated the Gemma-era copy "is kept as" its
# `.bak` name, and this arm's README said "Do not delete those backups". At 2026-10-07T05:00:51
# the nightly job unlinked both copies — 1,327,300,608 B and 1,250,205,696 B, 2,577,571,840 B
# in one run, `held: []` in `$LLOYD_DATA/_pipeline/reflection/qmd-index-maintenance-2026-10-07.json`
# — because a hold was a filename matching in a `*.py`/`*.ts`/`*.sh`/`*.yml`, markdown was
# never in that scan, and #2323 had banned writing one. The two nodes below pin the retired
# sentences out of the tree and pin the replacement facts in, so the next reader learns the
# corpus is gone instead of being told a deleted file is standing by.

#: Assembled at runtime, never spelled: a `.py` naming a `.bak` candidate is a code
#: reference, and a code reference is a hold — which is the mechanism that just held two
#: copies nobody wanted held. `tests/test_qmd_index_maintenance.py` walks the tree for
#: exactly these literals on every run.
GEMMA_BAK = "index.sqlite.bak-" + "gemma-20260921"
PREWIPE_BAK = "index.sqlite.bak-" + "20260919-203417"

#: The two claims as they stood at this round's base, and the two absences they must stay.
KEPT_AS_CLAIM = "is kept as `" + GEMMA_BAK + "`"
DO_NOT_DELETE = "Do not delete those backups"

#: What replaced them: the acting run, its total, and the surviving markdown corpus — 142
#: exported transcripts dated before 2026-09-22 under
#: `~/lloyd-data/_pipeline/vault-derived/sessions`, which is what a pre-wipe re-run can now
#: be built from. Counted at HEAD: 185 transcripts in that tree, 142 of them before the wipe.
ACTING_RUN = "2026-10-07T05:00:51"
DELETED_BYTES = "2,577,571,840"
SURVIVORS = "142"


def test_neither_doc_still_tells_a_person_a_deleted_backup_is_keeping_the_corpus():
    """Clause 5: the two dead claims are retired, and the control proves the patterns saw
    them.

    An absence alone is the 0-hit grep with no control. Both patterns here are run against a
    line assembled from the retired sentences first, and both files are then required to
    carry the acting run's date, its byte total, and the record of what actually survives —
    because a retired alarm that is replaced by silence is how the corpus reads as "still
    backed up" to the next reader who asks.
    """
    kept_as = re.compile(r"is kept as \`index\.sqlite\.bak-")
    dont_delete = re.compile(r"[Dd]o not delete those backups")
    assert kept_as.search("the old index " + KEPT_AS_CLAIM + ". For the live count")
    assert dont_delete.search("(`…bak-20260919-203417`). " + DO_NOT_DELETE + ". Restoring")

    qmd = DOC.read_text(encoding="utf-8")
    readme = EPISODIC_README.read_text(encoding="utf-8")
    assert not kept_as.search(qmd), (
        f"architecture/qmd.md claims the Gemma-era copy is still on disk again: "
        f"{[ln.strip() for ln in qmd.splitlines() if kept_as.search(ln)]}. It is not — the "
        f"retention rule unlinked {GEMMA_BAK} at {ACTING_RUN}, so the sentence is an "
        "instruction to rely on a file that is gone.")
    assert not dont_delete.search(readme), (
        "eval/episodic-arm/README.md is bargaining with the retention rule again. It has "
        "already won that argument once, on prose, and the corpus is what the loss cost.")

    for name, text in (("architecture/qmd.md", qmd), ("eval/episodic-arm/README.md", readme)):
        assert ACTING_RUN in text, f"{name} retires the claim without naming the run"
        assert DELETED_BYTES in text, (
            f"{name} says the copies went without saying how much went: that number is why "
            "this is a ruling and not a tidying")
    assert GEMMA_BAK in readme and PREWIPE_BAK in readme, (
        "the README names neither deleted file, so nothing in it can be checked against "
        "`find ~/.cache/qmd`")
    assert SURVIVORS in readme, (
        "the README does not record the 142 pre-2026-09-22 transcripts that do survive, "
        "which is the only corpus a pre-wipe re-run has now")
    assert "qmd-stray-keep-list.json" in qmd, (
        "architecture/qmd.md retired the dead claim without pointing at the channel #2420 "
        "replaced it with, so the next person in this position writes prose again")


def test_the_survivor_count_the_readme_quotes_is_the_one_the_tree_yields():
    """The 142 in `eval/episodic-arm/README.md` is re-derived here, not vouched for.

    Over `~/lloyd-data/_pipeline/vault-derived/sessions`, 185 exported transcripts exist and
    142 sit under a date directory earlier than 2026-09-22 — the pre-wipe residue #2323's
    owed entry calls the surviving corpus, and the only thing left of the arm's own 656
    documents. No skip when the archive is not on this host, which is what #2396's own guard
    a few nodes above forbids in this file: the figure is then checked for being traceable to
    the directory it was counted in, which is the weaker claim, but it is still a claim and
    not a silence. Where the archive is present the number is re-counted and compared.
    """
    readme = EPISODIC_README.read_text(encoding="utf-8")
    # The same passwd anchor every other runtime read in this file uses, one level up:
    # the gate's `$HOME` is a round home with no data home in it, and an absolute path
    # written here would be a second anchor that silently measures nothing there.
    sessions = REFLECTION.parents[1] / "_pipeline" / "vault-derived" / "sessions"
    assert f"{SURVIVORS} exported" in readme, (
        f"the README no longer states its survivor count as {SURVIVORS} transcripts over "
        f"`{sessions}`, so nothing in it can be checked against the archive it counted")
    if not sessions.is_dir():
        return
    days = [p for p in sessions.iterdir() if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)]
    assert days, f"{sessions} holds no date-named day directory to count"
    total = sum(len(list(d.glob("*.md"))) for d in days)
    before_wipe = sum(len(list(d.glob("*.md"))) for d in days if d.name < "2026-09-22")
    assert f"{SURVIVORS} exported" in readme, (
        f"the README no longer states its survivor count as {SURVIVORS} transcripts, so "
        f"there is nothing here to compare with the {before_wipe} the tree holds")
    assert before_wipe >= int(SURVIVORS), (
        f"the README says {SURVIVORS} pre-wipe transcripts survive; the tree holds "
        f"{before_wipe} of {total}. A count below the quoted one is a second loss on top "
        "of the one the README records, and prose that understates a loss is the failure "
        "#2420 exists to stop")
    assert total > before_wipe, (
        f"{sessions} holds {total} exported transcripts and every one of them predates the "
        "wipe, so this tree is a frozen copy rather than the live archive the README's "
        f"{SURVIVORS} is a count of")


# ── #2539: the report of the deleting run is now a witness with history ────────
#
# The README paragraph above cites one dated report as the record of the run that unlinked
# the last two copies of the pre-wipe corpus, and until this round that citation resolved on
# exactly one machine: the bytes sat in `$LLOYD_DATA/_pipeline/reflection/`, in no
# repository, while the hourly btrfs snapshots of the data home
# (`scripts/backup/snapshot-data.sh`) kept them alive for 48 hours each and no longer. That
# is the distinction this pair of nodes is written around — the file was not unprotected, it
# was unpinnable, and the fix is a copy with history plus a citation that reaches it.

def test_the_committed_loss_report_is_the_runtime_bytes_on_every_branch_of_the_comparison(
        monkeypatch, tmp_path):
    """Clause 2: the 2026-10-07 witness is the report, and the mechanism proving it is walked.

    `_report` is #2396's mechanism for exactly this claim, and it is already exercised on
    four other reports — but a mechanism applied to other files is not a pin on this one, so
    the loss report is put through all four branches here. They are the four ways this
    round's deliverable could fail to be one: never staged (#2282's silent `git add`), staged
    and then edited or truncated against the machine that measured it, a tree with no data
    home (the ordinary gate case, which still has to grade the paragraph), and no committed
    copy at all. The last two are the same branch seen from either side of #2396: the
    runtime half is the control, never the source.
    """
    committed = _fixture(MAINTENANCE_LOSS)
    live = tmp_path / "reflection"
    live.mkdir()
    monkeypatch.setattr(sys.modules[__name__], "REFLECTION", live)

    # Branch 1: an agreeing runtime copy resolves, and what it resolves to is the run
    # #2420 and this README both name.
    (live / MAINTENANCE_LOSS).write_bytes(committed.read_bytes())
    rep = _report(MAINTENANCE_LOSS)
    assert rep == json.loads(committed.read_text(encoding="utf-8")), (
        "an agreeing runtime copy did not resolve to the committed witness")
    assert rep["ran_at"] == "2026-10-07T05:00:51.896339", (
        f"the committed witness is not the 05:00:51 run the README cites: {rep['ran_at']}")

    # Branch 2: a disagreeing runtime copy fails, naming both files. One quoted figure is
    # changed rather than the whole payload, because the failure has to be about a byte a
    # paragraph quotes and not about formatting.
    mutated = json.loads((live / MAINTENANCE_LOSS).read_text(encoding="utf-8"))
    mutated["stray_retention"]["deleted_bytes"] = 2_577_571_839
    (live / MAINTENANCE_LOSS).write_text(json.dumps(mutated, indent=2), encoding="utf-8")
    with pytest.raises(AssertionError) as excinfo:
        _report(MAINTENANCE_LOSS)
    reason = str(excinfo.value)
    assert str(live / MAINTENANCE_LOSS) in reason and str(committed) in reason, (
        f"a disagreeing runtime copy failed without naming both files: {reason}")

    # Branch 3: no runtime copy is not a skip and not a failure — a plain clone has to be
    # able to grade the paragraph from the committed bytes alone.
    (live / MAINTENANCE_LOSS).unlink()
    assert _report(MAINTENANCE_LOSS)["stray_retention"]["deleted_bytes"] == 2_577_571_840, (
        "an unreachable runtime record stopped meaning 'no control available' and became a "
        "skip or a failure again, which is the silent green #2396 exists to avoid")

    # Branch 4: and the committed half failing is loud. An ignored witness — the #2282
    # shape, where `git add` exits 0 having staged nothing — is a missing witness here.
    empty = tmp_path / "fixtures-without-the-loss-report"
    empty.mkdir()
    for name in CITED_REPORTS:
        if name != MAINTENANCE_LOSS:
            (empty / name).write_bytes((FIXTURES / name).read_bytes())
    assert len(list(empty.iterdir())) == len(CITED_REPORTS) - 1, (
        "the witness directory was not built one file short of the cited set")
    monkeypatch.setattr(sys.modules[__name__], "FIXTURES", empty)
    with pytest.raises(pytest.fail.Exception) as failinfo:
        _report(MAINTENANCE_LOSS)
    assert str(empty / MAINTENANCE_LOSS) in str(failinfo.value), (
        "an unstaged loss report failed without naming the witness it could not read: "
        f"{failinfo.value}")


def test_the_readme_names_the_committed_copy_beside_the_runtime_citation():
    """Clause 5's prose half: the citation survives the loss of the data home it cites.

    The seam is prose to committed bytes. `eval/episodic-arm/README.md` has always named
    the runtime path, and a reader with the repository alone could not resolve it — the
    same defect #2396 closed for `architecture/qmd.md` §5, re-filed here because this
    README is the file #2420's owed entry pointed at. So the committed board path has to be
    named *beside* the runtime one, in the same bullet: a citation parked somewhere else in
    the file is a citation to a file nobody associates with the sentence. What is then
    checked is not that two strings coexist but that they describe the same event — the two
    backups the README spells out are exactly the two the committed report says the run
    deleted, and every byte figure the paragraph quotes is re-derived from those bytes,
    which is why no `.bak` name is typed into this file to do it.
    """
    raw = EPISODIC_README.read_text(encoding="utf-8")
    flat = _flat(raw)
    runtime_cite = "$LLOYD_DATA/_pipeline/reflection/" + MAINTENANCE_LOSS
    committed_cite = "backlog/data/" + MAINTENANCE_LOSS
    assert runtime_cite in flat, (
        "the README no longer names the runtime report at all, so there is no citation here "
        "for the committed copy to sit beside")
    assert committed_cite in flat, (
        f"the README cites only {runtime_cite}, which resolves on no clone: the report is "
        "committed, so the citation should reach the copy a reader can actually open")

    bullets = [b for b in raw.split("\n- ") if "qmd-index-maintenance" in b]
    assert len(bullets) == 1, (
        f"{len(bullets)} bullets of the README name the report, so the split that is 'beside' "
        "in this node is no longer the bullet the citation lives in")
    bullet = _flat(bullets[0])
    assert committed_cite in bullet and runtime_cite in bullet, (
        "the two citations are not in the same bullet: a committed path in some other "
        "section does not make this paragraph checkable")
    assert abs(bullet.index(runtime_cite) - bullet.index(committed_cite)) < 400, (
        "the committed path is named in the bullet but nowhere near the runtime citation "
        "it is standing in for")

    rep = _report(MAINTENANCE_LOSS)
    sr = rep["stray_retention"]
    named = set(re.findall(r"index\.sqlite\.bak-[\w-]+", raw))
    assert named == set(sr["deleted"]), (
        f"the README names {sorted(named)} as the unlinked backups and the committed report "
        f"says {sorted(sr['deleted'])}: one of the two is wrong about what the run did, and "
        "the report is the one with a timestamp on it")
    assert DELETED_BYTES in bullet, (
        f"the bullet no longer quotes the run's total, so {DELETED_BYTES} cannot be checked "
        "against the report that measured it")
    assert f"{sr['deleted_bytes']:,}" == DELETED_BYTES, (
        f"the constant this file quotes is no longer the report's own total "
        f"{sr['deleted_bytes']}")
    pile = {e["name"]: e for e in rep["stray"]}
    for name in sr["deleted"]:
        assert f"{pile[name]['bytes']:,} B" in bullet, (
            f"the bullet no longer states the {pile[name]['bytes']:,} B of {name}, so the "
            "figure is a claim about the report rather than a quote from it")


# --- #2728: §2's `subliminal` sentence describes a root-level collection --------
#
# The sentence read "`subliminal` is the whole vault as one collection". It is now
# false of the collection this repository ships: `agent-services/conf/qmd-index.yml`
# carries `pattern: "*.md"` for it, root-level only, per the 2026-10-07 ruling on
# #2180. It also happens to be the sentence a reader of #2180 arrived at and
# widened the glob from, so an unworded doc is not a cosmetic problem here.
#
# Why this node cannot be a one-line `assert` on the raw text: the file BREAKS the
# phrase across two lines — "`subliminal` is the whole" / "vault as one collection"
# — so `grep -n "whole vault" architecture/qmd.md` answers 0 hits on the UN-edited
# doc. A node written as `assert "the whole vault" not in _doc()` would have
# passed green against the exact wording it exists to ban. `_flat` is what lets the
# assertion be about a sentence rather than about where the file happened to break
# it, and the control below runs that same `_flat` over the bytes the file really
# broke at, so a fold that stopped folding is caught before the absence is trusted.

#: The banned wording as §2 carried it, including the newline and the two-space
#: continuation indent the file actually used. Deliberately not a one-line
#: paraphrase: `"the whole vault as one collection"` written on one line would
#: prove only that `_flat` is the identity function.
OLD_SUBLIMINAL_CLAIM = ("`subliminal` is the whole\n"
                        "  vault as one collection, and a hit that arrives through it")

#: The phrase the replacement uses to name the pattern the collection now carries.
#: Pinned inside the bullet rather than anywhere in the file: §2 states `*.md` in
#: §2's 'What gets indexed' bullet too, so file-wide presence would pass on a doc
#: that deleted the sentence and named nothing in its place.
ROOT_LEVEL_CLAIM = 'root at `pattern: "*.md"`'

# --- #2923: the same bullet still described a pair that had already reconciled ----
#
# The sentence after the one above read "the daemon's own `~/.config/qmd/index.yml`
# keeps the wide glob until the operator leg runs it, and the nightly `config_drift`
# reports the gap as a `pattern` row — the half-landed state, not a regression". It
# was true when #2728 wrote it and went false at 2026-10-15 21:15, when the leg ran:
# the live file's :28 is `pattern: "*.md"` (mtime measured, and `diff` against
# `index.yml.bak-20261015` is that one line), and the nightly reports go
# `drift_count: 1 / in_sync: false` on 2026-10-15 (run 05:01, before the edit) to
# `0 / true` on 2026-10-16 and 2026-10-17. So the doc is the stale half of a pair
# that has nothing left to reconcile, and #2728's own node could not see it: the
# wording it banned was the *other* sentence.
#
# The two banned phrases are not equally hard to find, which is why both need the
# fold. `keeps the wide glob until the operator leg runs it` sits whole on one line,
# so a raw grep hits it. `the half-landed state, not a regression` breaks across the
# next two lines with a two-space continuation indent, so `"the half-landed state,
# not a regression" in _doc()` is False on the UN-edited file and the ban would pass
# green doing nothing — the same trap #2728's comment above names for its own phrase.

#: The banned phrase, as one string, shared by the control and the absence check
#: below so neither can drift from the other: the control has to fold to a HIT of
#: exactly this string before its absence downstream is believed.
BANNED_HALF_LANDED = "the half-landed state, not a regression."

#: The file's real bytes around that break, measured at `a79aead3` over
#: `architecture/qmd.md:103-104`: `… a \`pattern\` row — the` then a newline and two
#: spaces. Deliberately not a one-line paraphrase — the assertion that guards that
#: is in the node, not in this comment.
HALF_LANDED_CLAIM = ("the nightly `config_drift` reports the gap as a `pattern` row — the\n"
                     "  half-landed state, not a regression.")

#: The other half of the now-false pair. Wholly on :102, so `_flat` costs nothing
#: here and the ban is a plain substring test through the same fold.
WIDE_GLOB_CLAIM = "keeps the wide glob until the operator leg runs it"

#: What replaces the pair, in the present tense. Pinned inside the bullet the way
#: `ROOT_LEVEL_CLAIM` is: a bullet that only deleted the old sentence clears the ban
#: and tells the next reader nothing about which side is true — and this bullet is
#: the file a person reads just before widening a glob.
RECONCILED_PAIR_CLAIM = 'both now carry `pattern: "*.md"`'

#: The reason #2180 measured, which is what makes the narrowing safe rather than
#: merely smaller. Nothing pinned it before this node: `git grep -n "second time" --
#: tests/test_qmd_doc_claims.py` returned 0 hits at filing, so a rewrite that trimmed
#: the clause for brevity passed green and left the glob with a rule and no reason.
DUPLICATE_INDEXING_REASON = "indexed a second time"


def _collections_bullet() -> str:
    """§2's `**Collections**` bullet — the one item #2728 and #2923 both edit.

    The split #2728's node used inline, factored out so the ban, the replacement and
    the retained reason are all aimed at the same bullet rather than at three copies
    of the slice. Asserts its own cardinality, so a reworded heading fails here with
    a reason instead of at an index.
    """
    hits = [b for b in _section("## 2. The daemon").split("\n- ")
            if "**Collections**" in b]
    assert len(hits) == 1, (
        f"§2 has {len(hits)} '**Collections**' bullets, so what this node calls 'the "
        "bullet' is no longer one bullet")
    return hits[0]


def test_the_collections_bullet_describes_a_root_level_collection_not_the_whole_vault():
    """#2728 clause 4: the whole-vault wording is gone, the root-level pattern is
    named in its place, and the fold doing the work is proven on the real string.

    The seam is prose to config — the template ships `pattern: "*.md"` for
    `subliminal`, and until this commit §2 asserted the opposite to whoever next
    reads a reported `pattern` drift and decides which side to move. Two ways to
    pass this node falsely exist, and each has its own half of the node:

    * **the ban matching nothing.** The control folds the same banned phrase
      through the same `_flat` and requires a hit, so a `_flat` that lost its
      whitespace join, or a constant whose line break drifted from the file's,
      reddens here before the absence downstream is believed.
    * **the replacement stating nothing.** A bullet that only deletes the sentence
      clears the ban, so the pattern string, the ruling behind it and the
      fold-back that makes a `subliminal/` hit land on its segment path are all
      asserted inside that one bullet.
    """
    control = _flat(OLD_SUBLIMINAL_CLAIM)
    assert "the whole vault as one collection" in control, (
        "the control wording no longer folds to the banned phrase, so the absence "
        "below is a match that silently never matches — which is the failure mode "
        "this node exists to avoid, not a result it may report")

    bullet = _flat(_collections_bullet())

    assert "the whole vault as one collection" not in _flat(_doc()), (
        "§2 still calls `subliminal` the whole vault. The 2026-10-07 ruling on #2180 "
        'narrowed it to the vault root, and the tracked template\'s `pattern: "*.md"` '
        "contradicts the sentence, so the doc is the stale half")
    assert ROOT_LEVEL_CLAIM in bullet, (
        f"the Collections bullet no longer names the pattern the collection carries "
        f"({ROOT_LEVEL_CLAIM!r}); deleting the old sentence without saying what "
        "replaced it leaves the reader who widens the glob with nothing to read")
    assert "#2180" in bullet and "2026-10-07" in bullet, (
        "the bullet lost the ruling the narrowing rests on, so the pattern it names "
        "has no reason attached to it")
    assert "`agent_mcp/vault.py::_qmd_normalize_global`" in bullet, (
        "the fold-back is gone from the bullet. It is the mechanism that makes "
        "narrowing cost retrieval nothing, and the reason the narrowing was allowed "
        "at all")


def test_the_collections_bullet_no_longer_describes_a_half_landed_pair():
    """#2923 clauses 1 and 3: the half-landed wording is gone, and the fold that
    proves it is proven on the bytes the file really broke at.

    The seam is prose to config — the bullet is what tells a reader whether the
    tracked template and the daemon's live `~/.config/qmd/index.yml` agree, and since
    2026-10-15 21:15 they do. Two ways to pass this node falsely exist, and each gets
    its own half:

    * **the ban matching nothing.** The banned phrase breaks :103→:104, so it is
      absent from the raw doc even BEFORE this fix — an unfolded `assert ... not in
      _doc()` was already green at the commit that wrote the sentence. The control
      folds the real two-line bytes through the same `_flat` and requires a hit, and
      the assertion after it refuses the constant if it ever stops being two lines,
      so a `_flat` that lost its whitespace join, or a constant that was quietly
      reflowed, reddens the control before the absence downstream is believed.
    * **the ban reading only the bullet.** A neighbouring paragraph could carry the
      same claim and the bullet alone would look clean, so both absences are asserted
      over the fold of the whole document.
    """
    control = _flat(HALF_LANDED_CLAIM)
    assert BANNED_HALF_LANDED in control, (
        "the control wording no longer folds to the banned phrase, so the absence "
        "below is a match that silently never matches — which is the failure #2728's "
        "node above was cut to avoid, not a result this one may report")
    assert BANNED_HALF_LANDED not in HALF_LANDED_CLAIM, (
        "the control constant now holds the banned phrase on one line, so it is a "
        "paraphrase rather than the file's real broken bytes and folding it proves "
        "only that `_flat` is the identity function")

    flat_doc = _flat(_doc())
    assert WIDE_GLOB_CLAIM not in flat_doc, (
        "the doc still says the daemon's own `~/.config/qmd/index.yml` keeps the wide "
        'glob until the operator leg runs it. The leg ran on 2026-10-15 and that '
        'file carries `pattern: "*.md"`, so the sentence is telling a reader to wait '
        "for an edit that has already happened")
    assert BANNED_HALF_LANDED not in flat_doc, (
        "the doc still calls the pair a half-landed state reported by `config_drift`. "
        "The nightly reports go `drift_count: 1 / in_sync: false` on 2026-10-15 (run "
        "05:01, before the 21:15 edit) to `0 / true` on 2026-10-16 and 2026-10-17, so "
        "there is no gap left to describe and a reader would go looking for one")
    assert "half-landed" not in flat_doc, (
        "'half-landed' survives somewhere in the doc — the phrase the replaced "
        "sentence coined, and the one this clause retires however it is reworded")


def test_the_collections_bullet_says_the_tracked_and_live_files_now_agree():
    """#2923 clause 2: the bullet states the reconciled pair, in the present tense.

    Deleting the stale sentence is half the edit and, on its own, the kind that reads
    as finished. The reader who lands here is the one a nightly `config_drift` row
    pointed at a live file, so a bullet that says only "`subliminal` is root-level"
    still leaves them unable to tell which side needs moving. The replacement is
    therefore pinned as a phrase constant asserted inside this one bullet — the shape
    `ROOT_LEVEL_CLAIM` uses above — and the constant carries the present-tense verb
    ("now carry") rather than a date nobody downstream will be able to re-verify.
    """
    bullet = _flat(_collections_bullet())
    assert RECONCILED_PAIR_CLAIM in bullet, (
        "the Collections bullet no longer states that both the tracked template and "
        f"the daemon's own `~/.config/qmd/index.yml` carry the root-level glob "
        f"({RECONCILED_PAIR_CLAIM!r} inside the bullet). Deleting the old half-landed "
        "sentence without naming the reconciled state is the deletion this clause "
        "exists to refuse")
    assert "`~/.config/qmd/index.yml`" in bullet, (
        "the reconciled statement has to name the live file by path: it is the file a "
        "reported drift points at and the one the operator leg edited")
    assert "#2728" in bullet, (
        "the bullet lost the attribution of which round narrowed the tracked half, so "
        "the two sides agreeing has no record of how it happened")


def test_the_collections_bullet_keeps_the_duplicate_indexing_reason():
    """#2923 clause 4: the reason behind the narrowing outlives the rewrite.

    Three of the four retentions in that bullet were already pinned by #2728's node
    (`ROOT_LEVEL_CLAIM`, `#2180` and `2026-10-07`, the `_qmd_normalize_global`
    fold-back), so clause 4 asks only that a rewrite keep them. This one had no node
    at all before #2923 — nothing in this file spoke of indexing twice — so a rewrite
    that trimmed the clause for length would have shipped a glob with a rule attached
    and no argument beside it, which is the shape that let the vault-wide glob stand
    from 2026-09-07 to 09-19.
    """
    bullet = _flat(_collections_bullet())
    assert DUPLICATE_INDEXING_REASON in bullet, (
        f"the bullet no longer says what was wrong with the wide glob "
        f"({DUPLICATE_INDEXING_REASON!r}). #2180's finding was that every note a "
        "segment collection owns was a `subliminal` row as well, so the same file "
        "reached the merge as two candidates; a narrowing with no reason beside it is "
        "undone by the next reported drift")
    assert "`**/*.md`" in bullet, (
        "the glob the ruling replaced is gone from the bullet, so the reason has "
        "nothing left to point at")


# ── The family-wide citation rule (#2755): a cited report resolves or the suite is red ──
#
# Every witness in this file was added by an item that had already shipped a dangling
# citation: #2302, #2396 and #2539 each spent a round on one missing report. Three rounds
# were needed because the citation-must-resolve check was single-family and single-bullet —
# `qmd-side-copy-rebuild-2026[^` ]+` inside
# `test_the_paragraph_reports_the_swap_and_cites_the_report_behind_it`, whose corpus is one
# `**Task #81**` bullet — while the nightly family was guarded only by the names enumerated
# in `CITED_REPORTS`. A list of what has been checked is not a rule over what may be cited:
# 21 nightly reports sat under `$LLOYD_DATA/_pipeline/reflection/` on 2026-10-15, 17 of them
# with no witness, so the next document to cite one needed an item rather than a diff.
# #2539's owed-check ruled for widening the scan and measured the cost at zero, which is what
# the node below re-measures on every run.

#: The corpus, as routes: tracked markdown and nothing else. Code stays out because prose in
#: a `.py` is a developer's note rather than a claim a reader of the architecture depends on,
#: and three nightly names live in exactly that kind of prose with no witness anywhere —
#: `tests/test_qmd_index_maintenance.py:1009` quotes `-2026-09-26.json`, and `:3251` and
#: `:3433` quote `-2026-09-30.json`. A corpus that read `.py` would be red on landing for
#: citations no document makes, which is how a rule ends up deleted instead of followed. The
#: shape is re-proved after expansion, so widening these routes with a code glob fails here.
MARKDOWN_ROUTES = ("architecture/*.md", "eval/**/README.md")

#: The two dated families, and only those. `qmd-stray-keep-list.json` and
#: `qmd-stray-release-list.json` are cited by the same doc through the same `$LLOYD_DATA`
#: prefix and are outside this rule on purpose: the maintenance job rewrites those in place,
#: so a committed copy would stop matching the live one by the next morning and the
#: byte-comparison control would be false inside a day. A dated report is the opposite case
#: — it names one finished invocation and never changes again — which is why it can be a
#: witness and those two cannot.
CITED_REPORT_RE = re.compile(
    r"\$LLOYD_DATA/(_pipeline/reflection/qmd-(?:side-copy-rebuild|index-maintenance)"
    r"-2026[0-9T:.Z_-]*\.json)")

#: A nightly report that is on this box and has no witness, named by #2755's own positive
#: control. Nothing here goes looking for it: no node in this section reads
#: `_pipeline/reflection/`, which is what keeps the section green in a gate worktree whose
#: `$HOME` is a `round_home` with no such directory at all (#2396 clause 5).
UNCITED_NIGHTLY = "qmd-index-maintenance-2026-10-14.json"


def _tracked_markdown() -> list[tuple[str, str]]:
    """`(repo-relative path, text)` for every tracked file under `MARKDOWN_ROUTES`.

    From the index rather than the working copy: a scratch `.md` sitting in a worktree is not
    a citation anyone depends on, and an unstaged document is not a claim the suite owes a
    check to. Both routes are markdown, and that is asserted rather than assumed.
    """
    listed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--", *MARKDOWN_ROUTES],
        capture_output=True, text=True)
    assert listed.returncode == 0, listed.stderr
    paths = sorted({ln for ln in listed.stdout.split("\n") if ln})
    assert paths, f"`git ls-files -- {' '.join(MARKDOWN_ROUTES)}` listed no file at all"
    code = [p for p in paths if not p.endswith(".md")]
    assert not code, f"a MARKDOWN_ROUTES entry expanded to non-markdown: {code}"
    return [(p, (ROOT / p).read_text(encoding="utf-8")) for p in paths]


def _cited_reports(corpus: list[tuple[str, str]]) -> dict[str, set[str]]:
    """`{report basename: the files citing it}` for one already-opened corpus."""
    found: dict[str, set[str]] = {}
    for rel, text in corpus:
        for citation in CITED_REPORT_RE.findall(text):
            found.setdefault(Path(citation).name, set()).add(rel)
    return found


def _assert_citations_resolve(corpus: list[tuple[str, str]]) -> dict[str, set[str]]:
    """Fail naming the witness path for every citation with nothing committed behind it.

    Resolution is the tree and its index alone — `_fixture()` and `git ls-files
    --error-unmatch` — never `REFLECTION`, never the passwd anchor, never a vault path. That
    is a deliberate narrowing: the byte-comparison against the live report is a different
    question and is already asked of every name in `CITED_REPORTS` by
    `test_the_three_new_cited_reports_are_exact_tracked_and_admissible`, and reaching the
    machine from here would make the open-set rule ungradeable on a host that has no
    `_pipeline/reflection/`. There is no skip in this function either, which
    `test_no_witness_check_may_skip_a_report_it_could_not_read` holds over this file's source.
    """
    cited = _cited_reports(corpus)
    gaps: list[str] = []
    for name in sorted(cited):
        witness = _fixture(name)
        rel = witness.relative_to(ROOT).as_posix()
        if not witness.is_file():
            gaps.append(f"{', '.join(sorted(cited[name]))} cites {name}, which has no "
                        f"committed witness at {rel}")
            continue
        tracked = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel], capture_output=True)
        if tracked.returncode != 0:
            gaps.append(f"{', '.join(sorted(cited[name]))} cites {name}; {rel} is on disk but "
                        "not in the index, so a fresh clone has nothing to read")
    assert not gaps, (
        "a tracked document cites a dated qmd report whose bytes are in no repository, so "
        "every figure that citation quotes is re-derivable only on the box that ran the job — "
        "the green-that-means-nothing #2396 was cut to close. Commit the report in the same "
        "diff as the document that cites it: " + "; ".join(gaps))
    return cited


def test_every_dated_qmd_report_a_doc_cites_has_a_committed_witness():
    """The rule #2539's owed-check ruled for, over the open set rather than a list of names.

    Green at HEAD because widening cost nothing today: the citations in the corpus are
    exactly the five names `CITED_REPORTS` enumerates — three nightly reports and the two
    rebuild reports — and all five witnesses are already in the index. The asserted direction
    is the superset: a sixth citation whose witness is committed in the same diff passes with
    no edit to this file, which is the difference between this node and the enumerated dict
    that let #2302, #2396 and #2539 repeat one fix three times.
    """
    cited = _assert_citations_resolve(_tracked_markdown())

    assert cited, (
        "the scan found no dated qmd citation anywhere in the corpus, so it is an empty loop "
        "rather than a rule — the citations live in architecture/qmd.md and "
        "eval/episodic-arm/README.md")
    dropped = sorted(set(CITED_REPORTS) - set(cited))
    assert not dropped, (
        f"a report `CITED_REPORTS` still enumerates is cited by no markdown under "
        f"{MARKDOWN_ROUTES} ({dropped}), so the widened scan has silently stopped covering "
        "it: either the citation moved to a file outside the corpus, which needs a new route, "
        "or it was deleted, which means the dict loses the name too")
    citing = {rel for rels in cited.values() for rel in rels}
    for source in (DOC, EPISODIC_README):
        rel = source.relative_to(ROOT).as_posix()
        assert rel in citing, (
            f"the scan read no citation in {rel} even though it is cited there; a corpus that "
            "quietly lost the file holding a citation reports a clean sweep")


def test_a_citation_with_no_witness_fails_the_family_scan_naming_the_missing_file():
    """This section can fail, and what it asks for is admissible: the two halves, in one node.

    The corpus is a copy — the tracked README's text plus one line citing `UNCITED_NIGHTLY` —
    and nothing is written to the tree. What is pointed at that corpus is the same
    `_assert_citations_resolve` the green node calls, so the failure demonstrated is the
    node's own assertion rather than a hand-written analogue of it.
    """
    readme = EPISODIC_README.relative_to(ROOT).as_posix()
    clean = EPISODIC_README.read_text(encoding="utf-8")
    assert not _fixture(UNCITED_NIGHTLY).exists(), (
        f"{UNCITED_NIGHTLY} now has a witness, so it is no longer a fixture-less name and "
        "cannot demonstrate a missing one: point UNCITED_NIGHTLY at any nightly report in "
        "the family that still has none. The rule above is unaffected — this node only proves "
        "the rule can fail")
    assert UNCITED_NIGHTLY not in _cited_reports([(readme, clean)]), (
        f"the tracked README already cites {UNCITED_NIGHTLY}, so appending that citation below "
        "would tamper with nothing; move the control to a report no document cites yet")

    tampered = (clean + f"\nSee `$LLOYD_DATA/_pipeline/reflection/{UNCITED_NIGHTLY}` "
                "for the run that lost nothing.\n")
    assert UNCITED_NIGHTLY in _cited_reports([(readme, tampered)]), (
        "the regex did not match the citation form this node writes, so the green node above "
        f"is not looking for a name like {UNCITED_NIGHTLY} either")

    with pytest.raises(AssertionError, match=re.escape(f"tests/fixtures/{UNCITED_NIGHTLY}")):
        _assert_citations_resolve([(readme, tampered)])

    rel = f"tests/fixtures/{UNCITED_NIGHTLY}"
    rc, source, pattern = _ignore_rule(rel)
    assert rc == 0 and pattern == "!qmd-index-maintenance-*.json", (
        f"{rel} is resolved by {source!r} with pattern {pattern!r}; the family negation in "
        "tests/fixtures/.gitignore is what admits a future witness past the root `*.json` "
        "rule, and `.gitignore` is a denied path for a round — a fix that needed an edit "
        "there would be a fix that cannot land")
    assert subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--no-index", rel],
                          capture_output=True).returncode != 0, (
        f"git check-ignore reports {rel} as ignored, so committing the witness the failure "
        "above asks for would stage nothing and the suite would stay red forever")
    assert subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--no-index",
         "tests/fixtures/some-random-witness.json"], capture_output=True).returncode == 0, (
        "an ordinary .json under tests/fixtures is no longer ignored, so the line above is "
        "proving nothing: it is the family negation, not a missing rule, that admits a "
        "witness")
