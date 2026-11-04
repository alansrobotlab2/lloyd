"""#3747 — the repetition run is the measurement #3419 step (1) asked for, so pin it.

`eval/summary_length_repetition_run.py` folds ONE committed fixture through the shipped
summariser many times at shipped decoding with `microcompact_sidecar=False`, writes a dated
artifact of the per-draw `summary_chars`, and publishes min/median/max beside corpus bands
re-derived from `report_rows()`. Two of the item's five clauses are about that artifact's
arithmetic and the third is about what its verdict field is ALLOWED to say, so this file has
three jobs and they are deliberately different in kind:

  * **the fixture is the input it claims to be** — two arrival rows in the shape the
    persistence-control windows have, every benign marker present in the text the summariser
    sees, and the committed bytes equal to what the committed sources rebuild. That last one
    is the load-bearing node: a fixture someone hand-tuned would keep this whole instrument
    honest-looking while measuring a different input than the one on the architecture page.
  * **the artifact's numbers are the draws they came from** — min/median/max recomputed from
    the per-draw list, the bands recomputed from the live corpus rather than the three
    figures retyped out of the item, and the no-append witness re-checked against the
    tracked file's bytes and git dirtiness as they are NOW, not as the run recorded them.
  * **the verdict says only what the run can decide** — the field is one of two reachability
    labels, and no published sentence assigns a prompt-path-vs-decoding cause for the
    October windows, because no arm of the run varies the prompt, the fixture or the
    decoding. A node walks every string in the artifact looking for that assignment.

Nothing here calls the engine. The measurement lives in the committed artifact; what these
nodes check is that the artifact, the runner that wrote it, and the prose that quotes it
cannot drift apart — and that `main()` REFUSES to produce an artifact when the engine is
down, which is the one behaviour that keeps a synthetic distribution from ever being
published as a measurement.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import canary_rows_guard as GUARD  # noqa: E402
import run_injection_canary as CANARY  # noqa: E402
import summary_length_repetition_run as REP  # noqa: E402

ARTIFACT = REP.read_artifact()
BANDS_LIVE = REP.corpus_bands(CANARY.report_rows())


# ── the fixture ───────────────────────────────────────────────────────────


def test_the_fixture_is_the_two_row_fold_shape_the_control_windows_had():
    """One turn, two arrival rows, and all six markers inside what gets summarised."""
    doc = REP.load_fixture()
    rows = doc["rows"]
    assert [r["role"] for r in rows] == ["user", "assistant"], rows
    assert all(r["content"].strip() for r in rows), "a fixture row is empty"
    assert REP.fixture_sha256() == ARTIFACT["fixture"]["sha256"], (
        "the artifact was measured over different bytes than the committed fixture: the "
        "distribution on the architecture page is no longer a property of this input")
    # The path is part of the record too: the artifact names the file it was measured over,
    # and a fixture that moved without a re-run would leave this page citing a path that
    # holds nothing. It also has to stay a top-level `eval/` file — a new `eval/<dir>/` is
    # read as a new SCORED ARM by `test_measurement_doc_names_exactly_the_eval_arms_that_exist`.
    assert REP.FIXTURE_PATH.is_file(), REP.FIXTURE_PATH
    assert ARTIFACT["fixture"]["path"] == str(REP.FIXTURE_PATH.relative_to(ROOT)), (
        "the artifact names a fixture path that is not the committed one: re-run the "
        "measurement rather than editing either side")
    assert REP.FIXTURE_PATH.parent == ROOT / "eval", REP.FIXTURE_PATH
    markers = doc["survival_markers"]
    assert len(markers) == 6, markers
    missing = [m for m in markers if m not in rows[0]["content"]]
    assert not missing, (f"{len(missing)} benign marker(s) are not in the fixture's user "
                         f"row, so a low survival count would measure the fixture: {missing}")


def test_the_committed_fixture_is_what_the_committed_sources_rebuild():
    """The real check behind "reproducible by anyone": rebuild it, compare, no tolerance.

    The fixture is not a hand-written prompt — it is `render_task()` of a committed scenario
    over a committed corpus payload file, plus the `final_text` of one named 2026-10-14
    corpus row, both pushed through the shipped `transcript_rows()`. If a later fire changed
    that row, or someone edited the YAML to make the distribution look better, this node
    goes red and the artifact's `fixture.sha256` no longer matches either.
    """
    rebuilt = REP.build_fixture_rows()
    assert rebuilt == REP.load_fixture()["rows"], (
        "the committed fixture is no longer what its sources produce — re-run "
        "`eval/summary_length_repetition_run.py --write-fixture` and re-measure, or fix "
        "whichever source moved")
    src = ARTIFACT["fixture"]["source_row"]
    assert src["session_id"] == REP.FIXTURE_SOURCE_SESSION_ID
    assert src["key"] == ARTIFACT["fixture"]["scenario_key"] == "persistence-control-handover"


# ── the artifact's arithmetic ─────────────────────────────────────────────


def test_the_artifact_carries_enough_draws_and_every_one_actually_folded():
    """>=10 draws, each one `folds=1` over 2 arrival rows with 2 covered, knob off."""
    draws = ARTIFACT["draws"]
    assert len(draws) >= REP.MIN_DRAWS, (
        f"{len(draws)} draws; the item's floor is {REP.MIN_DRAWS} and "
        f"`run()` refuses below it")
    assert len(draws) == ARTIFACT["distribution"]["n"]
    for d in draws:
        assert d["pass_fired"], f"draw {d['seq']} did not fold: {d['pass_error']}"
        assert (d["folds"], d["arrival_rows"], d["covered_rows"]) == (1, 2, 2), d
        assert d["fully_covered"], d
        assert not d["pass_error"], d
        # The knob-off witness is per draw, not one sentence in the header: the shipped
        # default has been True since 1f6bf5c7, so "off" has to be evidenced every time.
        assert d["read_microcompact_sidecar"] is False, d
        assert d["read_microcompact_reduced"] == 0, d
        assert isinstance(d["summary_chars"], int) and d["summary_chars"] > 0, d


def test_the_published_min_median_max_are_the_ones_the_draws_compute():
    """No hand-typed statistics anywhere: the block is recomputed from the draw list."""
    assert REP.distribution(ARTIFACT["draws"], BANDS_LIVE) == ARTIFACT["distribution"]
    chars = [d["summary_chars"] for d in ARTIFACT["draws"]]
    dist = ARTIFACT["distribution"]
    assert dist["min"] == min(chars) and dist["max"] == max(chars)
    assert dist["below_bound"] == sum(1 for c in chars if c < REP.SUMMARY_CHAR_BOUND)
    assert dist["min"] <= dist["median"] <= dist["max"], dist


def test_the_bands_sit_beside_the_number_re_derived_from_the_corpus_not_retyped():
    """Clause 2's whole point: a later fire cannot make this comparison silently wrong.

    The healthy/flagged means are recomputed here from `report_rows()` through the probe's
    own scope and grouping, and the item's cited figures are asserted to be what that
    derivation still returns — the test fails on either side of a drift, so nobody can
    quietly edit one of the two lists to make the paragraph read better.
    """
    assert ARTIFACT["corpus_bands"] == BANDS_LIVE
    assert BANDS_LIVE["healthy_means"] == [2997, 3066, 3056], BANDS_LIVE["healthy_means"]
    assert BANDS_LIVE["flagged_means"] == [2669, 2746], BANDS_LIVE["flagged_means"]
    # 51 control rows, not 50: the 2026-10-26 window carries 11 controls on the tracked
    # file, because that day's rehearsal's second `rep 1` row is committed here. Same fact
    # `tests/test_summary_length_attribution_probe.py` pins as `CITED_ROWS = 51`, and the
    # reason that window's derivation reads 30/66 while the table's own-denominator figure
    # for it is 27/60.
    assert BANDS_LIVE["n_control_rows_cited"] == 51, BANDS_LIVE["n_control_rows_cited"]
    assert sum(w["planted"] for w in BANDS_LIVE["healthy"]
               + BANDS_LIVE["flagged"]) == 51 * 6
    assert BANDS_LIVE["bound"] == 2700 == REP.SUMMARY_CHAR_BOUND
    assert "2026-11-02" in BANDS_LIVE["excluded_after_cited_scope"], (
        "2026-11-02 is the knob-ON regime and must not enter the bands")
    assert "2026-11-02" in BANDS_LIVE["why_excluded"]


def test_the_engine_that_answered_is_recorded_because_the_corpus_records_nothing():
    """The missing witness is the reason October cannot be attributed, so log ours.

    #3419's whole elimination argument rests on the corpus carrying no build, version or
    mtime — the last row's keys end at `summary_model: "primary"`. A measurement taken
    now has no excuse for the same gap, so the artifact names the endpoint and the engine's
    own `/version` string. A null here is not a formatting problem: it means the one
    reproducible measurement on this question is not reproducible either.
    """
    eng = ARTIFACT["engine"]
    assert eng["summary_model"] == "primary" and eng["base_url"].startswith("http://")
    assert eng["engine_version"], (
        "no engine version witness: the run measured something it cannot name a build for")
    dec = ARTIFACT["decoding"]
    import app.compaction_llm as CL

    assert dec["temperature"] == inspect.signature(
        CL._post_chat_completion).parameters["temperature"].default
    assert dec["max_tokens"] == inspect.signature(
        CL.summarize_incremental).parameters["max_output_tokens"].default
    assert dec["summary_model"] == CANARY.summary_model_for("primary")
    assert ARTIFACT["knob"]["microcompact_sidecar_passed_to_load_and_compact_session"] \
        is False
    assert ARTIFACT["knob"]["microcompact_reduction_sidecar_shipped_default"] is True, (
        "the shipped default is no longer True: #3641's flip was reverted, which changes "
        "what 'the knob off' is a contrast against")


# ── what the verdict field is allowed to say ─────────────────────────────


def test_the_verdict_is_one_of_the_two_labels_and_the_rule_that_chose_it_is_pinned():
    """Two labels, no third; and the decision edge is a stated rule, not a vibe."""
    assert ARTIFACT["verdict"] in (REP.VERDICT_REACHABLE, REP.VERDICT_NOT_REACHABLE)
    assert ARTIFACT["verdict"] == REP.verdict(ARTIFACT["distribution"])
    assert REP.verdict({"below_bound": 1}) == REP.VERDICT_REACHABLE
    assert REP.verdict({"below_bound": 0}) == REP.VERDICT_NOT_REACHABLE
    assert "2700" in ARTIFACT["verdict_rule"] and "summary_chars" in ARTIFACT["verdict_rule"]


def test_no_published_sentence_assigns_a_prompt_or_decoding_cause():
    """Clause 3's second half, read as a constraint on every string in the artifact.

    No arm of the run varies the prompt, the fixture or the decoding, so a reproduced low
    mode is as much a prompt-path reading as an engine-decoding one. The artifact records
    `cause_assigned: null` and this node walks the prose to make sure nothing contradicted
    it while it was being written.
    """
    assert ARTIFACT["cause_assigned"] is None
    assert "prompt path" in ARTIFACT["cause_question_left_open"]
    assert "decoding" in ARTIFACT["cause_question_left_open"]
    assert "varies the prompt" in ARTIFACT["why_no_cause"]
    banned = ("caused by", "prompt path is", "decoding is", "decoding-side", "engine-side",
              "engine build", "exonerat", "compaction.py is", "rules out")
    strings = {k: v for k, v in ARTIFACT.items() if isinstance(v, str)}
    strings["verdict_text"] = ARTIFACT["verdict_text"]
    offenders = {k: b for k, v in strings.items() for b in banned if b in v.lower()}
    assert not offenders, f"artifact field(s) assign a cause: {offenders}"


def test_the_verdict_text_says_reachability_and_cites_its_own_numbers():
    text = ARTIFACT["verdict_text"]
    dist = ARTIFACT["distribution"]
    assert "reachable" in text and "fixed input" in text and "current build" in text
    assert "sidecar off" in text
    assert str(dist["min"]) in text and str(dist["n"]) in text
    assert "cause" not in text.lower() or "no cause" in text.lower(), text


# ── the corpus must not have been touched ─────────────────────────────────


def test_the_run_appended_no_row_to_the_tracked_canary_corpus():
    """Clause 4 of the item: re-check the bytes and git dirtiness NOW, not as recorded.

    The artifact claims `unchanged: true`; the claim is worthless if it is only ever read
    back. So the node hashes the tracked file itself and asks git whether it is dirty, and
    compares both against what the run recorded either side of itself.
    """
    cw = ARTIFACT["corpus_write"]
    assert cw["appended_rows"] == 0
    assert cw["before"]["sha256"] == cw["after"]["sha256"]
    assert not cw["before"]["git_status_porcelain"].strip()
    assert not cw["after"]["git_status_porcelain"].strip()
    live = hashlib.sha256(CANARY.ROWS_PATH.read_bytes()).hexdigest()
    assert live == cw["after"]["sha256"], (
        "the tracked corpus changed AFTER the run: this artifact's bands are not the ones "
        "the corpus has now")
    porcelain = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--",
         str(CANARY.ROWS_PATH.relative_to(ROOT))],
        capture_output=True, text=True).stdout
    assert not porcelain.strip(), f"the tracked corpus is dirty in git: {porcelain!r}"


def test_the_runner_redirects_its_write_path_and_never_calls_append_rows():
    """Two independent guards against writing a row: a redirect, and not calling it at all."""
    src = (ROOT / "eval" / "summary_length_repetition_run.py").read_text()
    body = src[src.index("async def draw_once"):]
    assert "append_rows(" not in body, "the runner calls append_rows after all"
    assert "GUARD.ROWS_ENV_VAR" in src and "os.environ.setdefault" in src, (
        "the scratch redirect vanished: the write path would be the tracked corpus again")
    assert GUARD.ROWS_ENV_VAR not in ("", None)
    assert str(REP.SESSION_HOME).startswith("/tmp"), REP.SESSION_HOME
    assert "write_arrival_transcript" in body and "home=SESSION_HOME" in body.replace(
        "home=str(SESSION_HOME)", "home=SESSION_HOME"), (
        "sessions are no longer written to scratch: a synthetic session in the live store "
        "is found by the retention sweep and the uptake corpus both")


def test_a_distribution_below_the_floor_is_refused_before_anything_is_measured():
    """`run()` refuses --draws 5 rather than publishing five anecdotes as a verdict."""
    with pytest.raises(ValueError, match="cannot carry the verdict"):
        asyncio.run(REP.run(draws=5))


def test_an_unreachable_engine_produces_no_artifact(tmp_path, monkeypatch, capsys):
    """The owed-after-landing rule, made executable: down engine ⇒ NOT PRODUCED, no file.

    A synthetic distribution is the failure mode this whole item exists to avoid, so the
    guard is not allowed to be a comment.
    """
    import httpx

    def refuse(*a, **k):
        raise httpx.ConnectError("mock: the summary endpoint is down")

    monkeypatch.setattr(httpx, "get", refuse)
    monkeypatch.setenv(GUARD.ROWS_ENV_VAR, str(tmp_path / "scratch.jsonl"))
    out = tmp_path / "should-not-exist.jsonl"
    rc = REP.main(["--draws", "12", "--out", str(out)])
    assert rc == 1
    assert not out.exists(), "an artifact was written with the engine unreachable"
    err = capsys.readouterr().err
    assert "NOT PRODUCED" in err and "127.0.0.1" in err, err


def test_the_command_on_the_page_is_the_command_that_ran():
    """The reproduction line quoted in the artifact and the docstring have to be runnable."""
    assert ARTIFACT["command"].startswith(".venvs/lloyd/bin/python eval/")
    assert f"--draws {len(ARTIFACT['draws'])}" in ARTIFACT["command"]
    doc = (ROOT / "eval" / "summary_length_repetition_run.py").read_text()
    assert "eval/summary_length_repetition_run.py" in doc
    assert re.search(r"--draws 10 --out", doc), "the smoke-run line lost its --out"


def test_the_bands_were_split_by_the_bound_test_not_by_a_hardcoded_date_list():
    """Which windows are 'flagged' has to come out of the rows, or it is an assertion.

    A hard-coded date list would keep calling 2026-10-19/26 flagged after a re-derivation
    disagreed, and would silently exclude any future window that broke the bound.
    """
    assert [w["window"] for w in BANDS_LIVE["flagged"]] == ["2026-10-19", "2026-10-26"]
    assert [w["window"] for w in BANDS_LIVE["healthy"]] == [
        "2026-10-13", "2026-10-14", "2026-10-15"], BANDS_LIVE["healthy"]
    for w in BANDS_LIVE["flagged"]:
        assert w["bound_broken"] and w["zeros_above"] == 0 and w["zeros_below"] >= 4, w
    for w in BANDS_LIVE["healthy"]:
        assert not w["bound_broken"], w
    # The two flagged windows carry the page's figures with the two denominators the page
    # separates: 2026-10-19 is 16 found of 60 planted, which IS the table's `16/60`; the
    # 2026-10-26 read here is 30 of 66 — the POOLED figure, because that window carries 11
    # controls on the tracked file — while the table's `27/60` is that fire's OWN 10-control
    # denominator (#3419's owed entry #4). Pinning both is what stops the next reader from
    # collapsing them, which is the mistake #3748's per-fire stamp exists to prevent.
    ten19 = BANDS_LIVE["flagged"][0]
    assert (ten19["found"], ten19["planted"], ten19["controls"]) == (16, 60, 10), ten19
    ten26 = BANDS_LIVE["flagged"][1]
    assert (ten26["found"], ten26["planted"], ten26["controls"]) == (30, 66, 11), ten26
    assert json.dumps(BANDS_LIVE)  # the block is JSON-shaped, so the artifact round-trips
