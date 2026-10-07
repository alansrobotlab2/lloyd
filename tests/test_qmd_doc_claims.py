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

import json
import os
import pwd
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "qmd.md"
SCRIPT = ROOT / "scripts" / "maintenance" / "qmd_index_maintenance.py"

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
SWAP_FIXTURE = ROOT / "tests" / "fixtures" / SWAP_REPORT

#: A sibling name with no negation of its own, the positive control for
#: `test_the_swap_fixture_is_admissible_and_tracked`: it must resolve to the root rule.
SWAP_FIXTURE_CONTROL = "tests/fixtures/side-copy-rebuild-no-negation-control.json"


def _swapped_backup() -> str:
    """The name of the database the 2026-10-05 swap moved aside.

    Read out of `SWAP_FIXTURE`'s bytes at call time rather than at import: an unreadable or
    untracked fixture has to fail a node with a reason, not turn this module into a
    collection error that the gate reports as a green suite whose nodes never ran (#2282).
    """
    return Path(json.loads(SWAP_FIXTURE.read_text(encoding="utf-8"))
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


def _report(name: str) -> dict:
    """One report from the runtime record the doc cites.

    Skips with the missing path when there is no history to read — a checkout with
    no `_pipeline/reflection/` cannot re-derive a figure, and passing there would
    be the false green this file exists to avoid.
    """
    path = REFLECTION / name
    if not path.is_file():
        pytest.skip(f"no runtime record at {path}, so the cited figure cannot be re-derived")
    return json.loads(path.read_text(encoding="utf-8"))


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

    The seam is prose to runtime record: the paragraph is only checked if the
    report it points at exists and still says `swapped` and `retrieval_ok`.
    """
    swap = _report(SWAP_REPORT)
    flat = _flat(_task81_bullet())
    cites = re.findall(
        r"\$LLOYD_DATA/(_pipeline/reflection/qmd-side-copy-rebuild-2026[^` ]+)", flat)
    assert cites, "the paragraph cites no report under _pipeline/reflection/"
    assert any(c.endswith(SWAP_REPORT) for c in cites), cites
    for cite in cites:
        assert (REFLECTION.parent.parent / cite).is_file(), f"cited report is not on disk: {cite}"
    assert swap["swap"]["swapped"] is True and swap["swap"]["retrieval_ok"] is True, swap["swap"]
    assert swap["name"] == "rebuild-20261005" and "2026-10-05" in flat
    assert "swapped" in flat and "retrieval_ok" in flat, (
        "the paragraph no longer states what the swap report records")
    # The name comes out of the committed copy of the report, and the runtime report the
    # paragraph cites has to agree with it: the fixture cannot be allowed to drift into
    # naming some other backup while the doc's claim is checked against that one instead.
    committed = _swapped_backup()
    assert committed == Path(swap["swap"]["backup"]).name, (
        "the committed swap report and the runtime report the doc cites disagree about "
        "which database was moved aside")
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
    def rule(rel: str) -> tuple[int, str, str]:
        p = subprocess.run(
            ["git", "-C", str(ROOT), "check-ignore", "--no-index", "-v", rel],
            capture_output=True, text=True)
        line = p.stdout.strip().split("\t")[0]
        source, _, pattern = line.rpartition(":")
        return p.returncode, source, pattern

    code, source, pattern = rule(SWAP_FIXTURE_CONTROL)
    assert code == 0 and pattern == "*.json" and not source.startswith("tests/fixtures/"), (
        f"positive control broken: a sibling name with no negation resolved to "
        f"{source!r} / {pattern!r} (exit {code}), so the root rule is not what is being "
        "beaten and a clean answer below would prove nothing")

    rel = SWAP_FIXTURE.relative_to(ROOT).as_posix()
    code, source, pattern = rule(rel)
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
