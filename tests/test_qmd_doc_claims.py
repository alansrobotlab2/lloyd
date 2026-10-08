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
}

# Provenance of the four committed witnesses, each copied byte-for-byte out of
# `$LLOYD_DATA/_pipeline/reflection/` on 2026-10-08, the day HEAD was `f3809a69`.
# Lines and bytes as `wc -l -c` prints them, then the md5:
#
#   rebuild-20261004-2026-10-04T113501   68 lines, 1,783 B  2957114779d1a5a7f7a9e6dc1b1a652c
#   rebuild-20261005-2026-10-05T072250   82 lines, 2,250 B  8af93f34848086e3b974cd5719bff0af
#   index-maintenance-2026-10-05        226 lines, 7,408 B  fc110dcbd5b248b5e4ccf0a7f3c9a175
#   index-maintenance-2026-10-06        243 lines, 7,989 B  9db5937166170b9cb342c1d0fff21b10
#
# Each name above is the tail of a constant in this file: the two rebuild reports are
# `REHEARSAL_REPORT` and `SWAP_REPORT`, the two nightlies `MAINTENANCE_BEFORE` and
# `MAINTENANCE_AFTER`.
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
    sibling beside it, and the whole tracked family is counted, because "5" is the number
    `git ls-files tests/fixtures | grep -c qmd` prints once all four witnesses and the
    #2119 one are in the index.
    """
    exact = {  # name -> (lines, bytes), as `wc -l -c` of the runtime report prints them
        REHEARSAL_REPORT: (68, 1783),
        MAINTENANCE_BEFORE: (226, 7408),
        MAINTENANCE_AFTER: (243, 7989),
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
    assert len(qmd) == 5, (
        f"`git ls-files tests/fixtures | grep -c qmd` prints {len(qmd)} ({qmd}), not 5: "
        "four cited reports plus #2119's nightly witness is the whole committed family, "
        "and one fewer means a figure the paragraph quotes is in no repository")

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
