"""#2717 clauses 4-5 — the source-grounding gate, and the one planted violation.

The paper's guard is mechanical: a factual edit to a past reflection must be supported by
that reflection's own source artifact, so a fact learned on day 10 cannot migrate into the
day-1 record. Everything else in ReHarness is a model's opinion; this is a literal
comparison, so it can be pinned the way nothing else in the item can. One node plants a
fabricated literal and asserts the refusal, which is the sentence the item's acceptance
section asks for verbatim ("pinned by a test with a planted violation").

The fixture below is one daily-note passage under a temporary source root; every proposal
cites it and only it. The source is deliberately richer than the memory line it produced
(a sha, a path, a date, a time, a wattage, a unit name, an entity), because a check that
only refuses when the source is empty would pass here and fail everywhere.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCRIPT = REPO / "scripts" / "memory_source_grounding.py"

#: The cited source artifact: `daily/2026-09-22.md` in the fixture root. Its literals are
#: what a proposal may assert — 2026-09-22, 07:10, 2b42094, 450, nvidia-power-limit.service,
#: service_health_check.py, /etc/systemd/system, install-services.sh, Alan, 07:41, GPU 2.
SOURCE_TEXT = (
    "2026-09-22 07:10 — the guardian reverted the live tree to 2b42094 after the 450 W "
    "clamp went missing from nvidia-power-limit.service. `service_health_check.py` "
    "reported Deploy: healthy because the pair it reads is a copy under "
    "/etc/systemd/system, not a link. Alan re-ran install-services.sh at 07:41 and the "
    "timer re-armed. GPU 2 stayed idle.\n")

SOURCE_FILE = "daily/2026-09-22.md"
ORIGINAL_LINE = "Clamp the power limit before blaming the driver."


def _load_module():
    """Load the CLI script by path: it has no package and nothing imports it."""
    spec = importlib.util.spec_from_file_location("msg2717", str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def msg():
    return _load_module()


@pytest.fixture
def source_root(tmp_path):
    """A vault-shaped root holding only the artifact every proposal cites."""
    note = tmp_path / SOURCE_FILE
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(SOURCE_TEXT, encoding="utf-8")
    return tmp_path


def _proposal(pid, proposed, current=ORIGINAL_LINE, cited=SOURCE_FILE):
    return {"id": pid, "entry": "memory:lloyd/USER.md:9a1c",
            "current_line": current, "proposed_line": proposed,
            "cited_source": cited, "triggering_runs": ["s1#4", "s2#1"]}


def _verdict(msg, proposal, source_text=SOURCE_TEXT):
    return msg.check_proposal(proposal, source_text)


# ------------------------------------------------------- clause 4: the planted violation

def test_a_planted_narrowing_that_invents_a_path_is_refused(msg):
    """Clause 4, the planted violation the item's acceptance names.

    The condition reads plausibly — a copy under `/etc/systemd/user` is exactly the sort of
    thing this box has — and that is the point: the artifact this line was written from
    never mentions that directory, so asserting it in a narrowing is inventing a fact about
    the machine. Refused, naming the literal and its kind.
    """
    verdict = _verdict(msg, _proposal(
        "invented-path",
        ORIGINAL_LINE + " Only when the unit was linked into /etc/systemd/user."))
    assert verdict["accepted"] is False, verdict
    assert verdict["reason"].startswith("ungrounded_literal:"), verdict
    assert "path" in verdict["reason"], verdict
    texts = [lit["text"] for lit in verdict["ungrounded"]]
    assert "/etc/systemd/user" in texts, verdict
    assert any(lit["kind"] == "path" for lit in verdict["ungrounded"]), verdict
    assert verdict["cited_source"] == SOURCE_FILE, verdict


def test_a_planted_narrowing_that_invents_a_date_a_number_or_a_sha_is_refused(msg):
    """The same violation in the three other classes the item names, one proposal each.

    A recurrence date, a wattage and a commit id are the three shapes a fabricated memory
    line actually takes here, and each is a *different* matcher: a date that shares digits
    with a real one must still be refused, which is why matching is on the normalised form
    and not on "some digits from the source appear".
    """
    cases = {
        "date": ("Only when it recurred on 2026-10-05.", "2026-10-05", "date"),
        "number": ("Only above 600 W.", "600", "number"),
        "hex-id": ("Reverted again at 9f3c0aa.", "9f3c0aa", "hex-id"),
    }
    for want, (tail, literal, kind) in cases.items():
        verdict = _verdict(msg, _proposal(f"invented-{want}", ORIGINAL_LINE + " " + tail))
        assert verdict["accepted"] is False, (want, verdict)
        assert kind in verdict["reason"], (want, verdict)
        assert literal in [lit["text"] for lit in verdict["ungrounded"]], (want, verdict)


def test_a_planted_symbol_or_entity_name_absent_from_the_source_is_refused(msg):
    """Clause 4's last two classes: a check named as if it exists, and a component named as
    if it reported something. Both read as citations and neither is in the artifact.
    """
    symbol = _verdict(msg, _proposal(
        "invented-symbol",
        ORIGINAL_LINE + " check_unit_enabledness would have caught it."))
    assert symbol["accepted"] is False, symbol
    assert "symbol" in symbol["reason"], symbol
    assert "check_unit_enabledness" in [lit["text"] for lit in symbol["ungrounded"]], symbol

    entity = _verdict(msg, _proposal(
        "invented-entity", ORIGINAL_LINE + " Mission Control logged the clamp instead."))
    assert entity["accepted"] is False, entity
    assert "entity" in entity["reason"], entity
    assert "Mission Control" in [lit["text"] for lit in entity["ungrounded"]], entity


def test_the_control_narrowing_that_adds_no_new_literal_is_accepted(msg):
    """Clause 4's control: a limiting condition expressed in ordinary words.

    This is the shape the paper says a good revision usually takes ("larger first **only
    when** descending"), and it must pass — a gate that refused it would be refused by
    every nightly job within a week. It also records that the check had nothing to read,
    because "accepted" on this row means "no literal was invented", not "verified true".
    """
    verdict = _verdict(msg, _proposal(
        "control-verbal",
        ORIGINAL_LINE + " Only when the deployed copy is the one the manager loads."))
    assert verdict["accepted"] is True, verdict
    assert verdict["reason"] is None, verdict
    assert verdict["ungrounded"] == [], verdict
    assert verdict["note"] == msg.NOTE_NO_LITERALS, verdict


def test_a_condition_supported_by_the_source_is_accepted_though_it_adds_literals(msg):
    """The control with substance: a narrowing that cites the artifact's own date,
    directory and unit name. Three literal classes are added and all three are in the
    source, which is what grounding is for — the gate is not a ban on detail.
    """
    verdict = _verdict(msg, _proposal(
        "control-grounded",
        ORIGINAL_LINE + " As the 2026-09-22 revert showed, when the copy under "
                        "/etc/systemd/system drifted from nvidia-power-limit.service."))
    assert verdict["accepted"] is True, verdict
    assert verdict["ungrounded"] == [], verdict
    assert "note" not in verdict, verdict   # literals WERE added, and all were grounded
    kinds = {lit["kind"] for lit in verdict["literals"]}
    assert {"path", "date"} <= kinds, verdict


def test_a_literal_carried_over_from_an_already_drifted_line_is_still_refused(msg):
    """The paper's actual rule, not merely the "new text" rule: day-10 facts may not live in
    the day-1 record.

    The original line already carries a wrong sha, and the proposed narrowing keeps it and
    adds only words. Grading only the added literals would accept that and let a re-narrowing
    launder the drift, so every literal in the new text is graded against the artifact — and
    the row says the offender was carried over rather than invented here.
    """
    drifted = "Reverts to 2b42095 are the guardian's default when the clamp goes missing."
    verdict = _verdict(msg, _proposal(
        "carried-drift", drifted + " Only when the unit was not linked.", current=drifted))
    assert verdict["accepted"] is False, verdict
    assert "hex-id" in verdict["reason"], verdict
    assert "2b42095" in [lit["text"] for lit in verdict["ungrounded"]], verdict
    assert all(lit["added_by_this_proposal"] is False
               for lit in verdict["ungrounded"]), verdict
    assert all(lit["class"] == msg.CLASS_CARRIED_OVER
               for lit in verdict["ungrounded"]), verdict


def test_a_literal_only_the_proposed_line_says_is_refused_and_labelled_introduced(
        msg, source_root):
    """The class of offender the gate exists for, pinned on its own.

    The review of the previous round asked for exactly this shape: a literal that
    `proposed_line` asserts, `current_line` does not contain, and the cited artifact never
    mentions — the invented `/etc/systemd/user`, word for word. It is asserted here as a node
    rather than left to the shared `_grounded` path because the comparison with
    `current_line` is the one place a fix could quietly turn the gate inside out: if
    "grading what the proposal adds" ever became "excusing what the proposal kept", this is the
    row that goes first.

    It goes through `run_batch` and `render` as well as `check_proposal`, because clause 5 asks
    for the accepted-versus-refused counts and the offending literal quoted *in the report*,
    not only in a returned dict.
    """
    proposal = _proposal("invented-only-in-proposed",
                         ORIGINAL_LINE + " Only when the unit was linked into "
                                         "/etc/systemd/user.")
    verdict = _verdict(msg, proposal)
    assert verdict["accepted"] is False, verdict
    assert verdict["reason"] == "ungrounded_literal:path", verdict
    assert [lit["text"] for lit in verdict["ungrounded"]] == ["/etc/systemd/user"], verdict
    assert verdict["ungrounded"][0]["class"] == msg.CLASS_INTRODUCED, verdict
    assert verdict["ungrounded"][0]["added_by_this_proposal"] is True, verdict
    # The two classes are reported separately, and nothing here is carried over.
    assert [lit["text"] for lit in verdict["introduced"]] == ["/etc/systemd/user"], verdict
    assert verdict["carried_over"] == [], verdict

    report = msg.run_batch([proposal], msg.file_source_reader(source_root))
    assert (report["checked"], report["accepted"], report["refused"]) == (1, 0, 1), report
    assert report["refusal_reasons"] == {"ungrounded_literal:path": 1}, report["refusal_reasons"]
    rendered = msg.render(report)
    assert "0 accepted, 1 refused of 1 proposed" in rendered, rendered
    # Quoted, so a reader can see the exact string that has no source.
    assert "'/etc/systemd/user'" in rendered, rendered
    assert "introduced" in rendered, rendered


# --------------------------------------------------------- what the gate deliberately misses

def test_a_backticked_citation_and_a_prose_hyphenation_are_not_refused(msg):
    """The over-refusal control, and the reason it exists: backlog **#653** was closed
    `rejected` on incidence because a wide literal gate cost a great deal and over-refused.

    A narrowing is *allowed* to cite the script it is about — the thing that makes a
    narrowing dangerous is an asserted fact, and a backticked identifier is a pointer, not a
    claim. `nvidia-thermal-guard.service` names no unit on this box and
    `scripts/check_nvidia_power.py` names no script, and neither is refused: both are inside
    backticks, which is the citation the check must not eat. A two-segment hyphenated word
    that the source never used ("hand-placed") is prose for the same reason, and is reported
    in `soft_ungrounded` so the soft class stays visible instead of quietly becoming the
    whole check.
    """
    verdict = _verdict(msg, _proposal(
        "citations",
        ORIGINAL_LINE + " See `nvidia-thermal-guard.service` in "
                        "`scripts/check_nvidia_power.py`; the guard was hand-placed."))
    assert verdict["accepted"] is True, verdict
    assert verdict["ungrounded"] == [], verdict
    soft = [lit["text"] for lit in verdict["soft_ungrounded"]]
    assert "hand-placed" in soft, verdict


def test_a_proposal_naming_no_source_is_refused_as_ungroundable(msg):
    """No cited artifact is not a pass: with nothing to compare against, every literal is
    unverified, and a batch that lost its citations would otherwise read as clean.
    """
    verdict = _verdict(msg, _proposal("no-source", ORIGINAL_LINE + " Usually.", cited=None))
    assert verdict["accepted"] is False, verdict
    assert verdict["reason"] == msg.REASON_NO_SOURCE, verdict


def test_a_cited_source_that_cannot_be_read_is_refused_and_says_so(msg):
    """An unreadable artifact is its own reason, so a moved file cannot be mistaken for a
    fabricated proposal when the refusal distribution is read back.
    """
    verdict = _verdict(msg, _proposal("unreadable", ORIGINAL_LINE + " Usually.",
                                      cited="daily/2026-09-23.md"), source_text=None)
    assert verdict["accepted"] is False, verdict
    assert verdict["reason"] == msg.REASON_SOURCE_UNREADABLE, verdict


# ------------------------------------------------- clause 5: counts, reasons, and the empty batch

def _mixed_batch(msg, source_root):
    """Five proposals over the fixture root: two grounded, two invented, one uncited."""
    proposals = [
        _proposal("ok-verbal", ORIGINAL_LINE + " Only when the copy is the loaded one."),
        _proposal("ok-grounded", ORIGINAL_LINE + " Seen on 2026-09-22 under "
                                                "/etc/systemd/system."),
        _proposal("bad-path", ORIGINAL_LINE + " Only under /etc/systemd/user."),
        _proposal("bad-symbol", ORIGINAL_LINE + " check_unit_enabledness agrees."),
        _proposal("no-source", ORIGINAL_LINE + " Usually.", cited=None),
    ]
    return msg.run_batch(proposals, msg.file_source_reader(source_root))


def test_the_batch_reports_accepted_and_refused_counts_with_a_reason_per_refusal(
        msg, source_root):
    """Clause 5: the report separates accepted from refused and names a reason for each
    refusal, so a reader can tell "the gate rejected everything" from "the gate saw
    nothing", and can see WHAT it rejected the proposals for.

    Five proposals in, two accepted, three refused: two distinct ungrounded-literal reasons
    (a path, a symbol) and one missing-citation reason. The reason strings are the same
    strings the per-verdict rows carry, which is what makes the histogram auditable rather
    than decorative.
    """
    report = _mixed_batch(msg, source_root)
    assert report["checked"] == 5, report["checked"]
    assert report["accepted"] == 2 and report["refused"] == 3, report
    assert report["accepted_ids"] == ["ok-verbal", "ok-grounded"], report["accepted_ids"]
    assert report["nothing_proposed"] is False, report
    by_id = {v["id"]: v for v in report["verdicts"]}
    assert by_id["bad-path"]["reason"] == "ungrounded_literal:path", by_id["bad-path"]
    assert by_id["bad-symbol"]["reason"] == "ungrounded_literal:symbol", by_id["bad-symbol"]
    assert by_id["no-source"]["reason"] == msg.REASON_NO_SOURCE, by_id["no-source"]
    assert report["refusal_reasons"] == {
        "ungrounded_literal:path": 1,
        "ungrounded_literal:symbol": 1,
        msg.REASON_NO_SOURCE: 1}, report["refusal_reasons"]
    assert report["denominators"] == {"proposals": 5,
                                      "distinct_refusal_reasons": 3}, report["denominators"]
    assert "does NOT mean" in report["meaning"], report["meaning"]


def test_an_empty_batch_is_reported_as_nothing_proposed_not_as_zero_accepted(msg):
    """Clause 5's whole point, and the distinction the item asks for by name.

    A nightly run that proposes nothing and a run whose every proposal was refused both
    print `accepted: 0`. Only `nothing_proposed` tells them apart, and without it the
    silence of an empty window reads as a gate working.
    """
    empty = msg.run_batch([], lambda cited: None)
    assert empty["nothing_proposed"] is True, empty
    assert (empty["checked"], empty["accepted"], empty["refused"]) == (0, 0, 0), empty
    assert empty["refusal_reasons"] == {}, empty
    assert empty["denominators"]["proposals"] == 0, empty
    assert "empty" in msg.render(empty), msg.render(empty)


# ---------------------------------------------------------------- the CLI boundary

def test_the_cli_grades_a_proposals_file_and_writes_the_report(tmp_path):
    """The boundary a nightly job crosses: a proposals file in, a report on disk, exit 0
    whatever the verdicts are — the gate reports, it does not decide the run's fate.

    Run as a subprocess and the report read back from the artifact directory, because the
    acceptance section is written from that file by a later job, not from this process.
    """
    root = tmp_path / "vault"
    (root / "daily").mkdir(parents=True)
    (root / SOURCE_FILE).write_text(SOURCE_TEXT, encoding="utf-8")
    proposals = tmp_path / "proposals.json"
    proposals.write_text(json.dumps([
        {"id": "a", "current_line": ORIGINAL_LINE,
         "proposed_line": ORIGINAL_LINE + " Seen on 2026-09-22.", "cited_source": SOURCE_FILE},
        {"id": "b", "current_line": ORIGINAL_LINE,
         "proposed_line": ORIGINAL_LINE + " Only under /etc/systemd/user.",
         "cited_source": SOURCE_FILE},
        {"id": "c", "current_line": ORIGINAL_LINE,
         "proposed_line": ORIGINAL_LINE + " Usually.", "cited_source": "daily/absent.md"},
    ]), encoding="utf-8")
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "--proposals", str(proposals),
         "--source-root", str(root), "--out-dir", str(out), "--as-of", "2026-10-14"],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    report_path = out / "source-grounding-2026-10-14.json"
    assert report_path.exists(), list(out.iterdir())
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert (report["checked"], report["accepted"], report["refused"]) == (3, 1, 2), report
    assert report["refusal_reasons"] == {
        "ungrounded_literal:path": 1,
        "cited_source_unreadable": 1}, report["refusal_reasons"]
    assert "1 accepted" in proc.stdout and "2 refused" in proc.stdout, proc.stdout


def test_a_run_writes_its_report_to_the_data_root_never_to_the_checkout(tmp_path, monkeypatch):
    """The filesystem boundary this script has: an artifact is run output, and `~/lloyd` is
    code only, so a report left beside the source would be a stray for the watchdog and
    would not survive a promote.

    The previous version of this node only read the resolved default path, and every CLI node
    passed `--out-dir`, so nothing had ever written through the default — which is the branch
    a nightly job takes. So this runs `main()` with NO `--out-dir` over a data root pointed at
    a temp directory, and asserts the bytes are actually there. The root is patched rather than
    measured, because a test that writes to the real data root is a test that deletes other
    people's artifacts.

    Also asserted on the two scripts agreeing on one directory — a nightly job that wrote
    proposals to one place and verdicts to another is the kind of split that makes an artifact
    impossible to find later.
    """
    msg = _load_module()
    import scripts.memory_revision_candidates as mrc  # noqa: E404  (repo on sys.path)
    from app import paths  # noqa: E404
    assert msg.default_out_dir() == paths.DATA_ROOT / "eval" / "memory-revision"
    assert msg.default_out_dir() == mrc.default_out_dir(), "two artifacts, two homes"
    out = str(msg.default_out_dir())
    assert not out.startswith(str(REPO) + "/"), f"artifact would land in the checkout: {out}"

    # The default, exercised: a run with no `--out-dir` at all.
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path / "data")
    root = tmp_path / "vault"
    (root / "daily").mkdir(parents=True)
    (root / SOURCE_FILE).write_text(SOURCE_TEXT, encoding="utf-8")
    proposals = tmp_path / "proposals.json"
    proposals.write_text(json.dumps([
        {"id": "a", "current_line": ORIGINAL_LINE,
         "proposed_line": ORIGINAL_LINE + " Only under /etc/systemd/user.",
         "cited_source": SOURCE_FILE},
    ]), encoding="utf-8")
    rc = msg.main(["--check", "--proposals", str(proposals),
                   "--source-root", str(root), "--as-of", "2026-10-14"])
    assert rc == 0, rc
    written = (tmp_path / "data" / "eval" / "memory-revision"
               / "source-grounding-2026-10-14.json")
    assert written.exists(), (msg.default_out_dir(), list(tmp_path.iterdir()))
    assert json.loads(written.read_text(encoding="utf-8"))["refused"] == 1, written
    assert not (REPO / "eval" / "memory-revision").exists(), \
        "the default write must resolve through DATA_ROOT, not the checkout"
