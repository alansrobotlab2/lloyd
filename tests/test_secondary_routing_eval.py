"""eval/secondary_routing_eval.py — the scorer, the two arms, the decision.

Three things here have to be unbreakable for the numbers to mean anything,
so each gets a test rather than a comment: the eval must call the *real*
routed functions (otherwise it measures a re-implementation), the primary
arm must actually reach the primary engine (an alias override that silently
failed would leave two identical arms and a plausible table), and a pinned
input must not be allowed to move quietly. The scorer's own checks are
pinned both ways — a good output passes, the specific failure the job is
prone to fails — because a scorer that cannot fail is how #525's
zero-claim problem got built.
"""
import inspect
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval.secondary_routing_eval as ev  # noqa: E402
from app import post_capture, secondary_models as sm  # noqa: E402

AUTONOMY_DIR = Path.home() / "obsidian" / "autonomy"

GOOD = {
    "title": "Vault retrieval eval regression",
    "capture": "The session fixed the nightly retrieval eval baseline. "
               "The runner defaults were imported from agent_mcp.vault. "
               "Two stale columns were removed from the compare step.",
    "facts": "[Retrieval Eval] runner defaults now import from agent_mcp.vault\n"
             "[Retrieval Eval] the compare step read three missing columns\n"
             "[Baseline] the 2026-09-04 run recorded ndcg10 of 0.594",
    "focus": "retrieval eval defaults\nstale compare columns\nbaseline trend recording",
    "voice": "The nightly retrieval eval is green. Ndcg came out at 0.594, "
             "which is where it has been all week.",
}


# ── The eval must be measuring the routed jobs, not a copy of them ───────

def test_every_secondary_routed_job_is_covered():
    assert set(ev.JOB_CALLS) == set(ev.JOBS) == set(ev.LENGTH_BUDGETS) == set(
        ev.FORMAT_CHECKS)


def test_each_job_dispatches_to_the_live_production_function():
    """Identity, not name equality: the eval and the router must be running
    the same object, so a prompt edit in production moves the measurement."""
    assert ev.JOB_CALLS["title"] is sm._sync_secondary_title
    assert ev.JOB_CALLS["capture"] is sm._sync_secondary_capture_call
    assert ev.JOB_CALLS["facts"] is sm._sync_secondary_fact_extraction
    assert ev.JOB_CALLS["focus"] is sm._sync_secondary_focus_extraction
    assert ev.JOB_CALLS["voice"] is sm._sync_secondary_voice_summary


def test_eval_file_names_the_routed_call_it_measures():
    """The acceptance check is a grep for `_sync_secondary_capture_call`
    under eval/ — this is that grep, pinned."""
    source = (ROOT / "eval" / "secondary_routing_eval.py").read_text(encoding="utf-8")
    for name in ("_sync_secondary_capture_call", "_sync_secondary_title",
                 "_sync_secondary_fact_extraction", "_sync_secondary_focus_extraction",
                 "_sync_secondary_voice_summary"):
        assert name in source, name


# ── The two arms ─────────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _fake_transport(monkeypatch, content: str = "ok", completion: int = 42) -> list:
    """Stand in for the engines. Returns the list the fake appends requests to."""
    body = json.dumps({
        "choices": [{"message": {"content": content}}],
        "usage": {"completion_tokens": completion, "prompt_tokens": 1234},
    }).encode()
    seen: list = []

    def fake(req, *args, **kwargs):
        seen.append(req)
        return _FakeResponse(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return seen


def test_primary_arm_really_reaches_the_primary_engine(monkeypatch):
    """The one claim the comparison rests on, checked from the URL the call
    used rather than from the label the run asked for.

    The arms are separated through the production router —
    `app.secondary_models.JOBS_ON_PRIMARY` — not by patching the alias
    resolver, so this asserts that the per-job pin is what actually moves the
    endpoint. A `JOBS_ON_PRIMARY` that production stopped reading would fail
    here rather than produce a run where both arms answer from :8091."""
    # Pin the slot ON. The whole premise here is that the two arms reach
    # DIFFERENT engines, and `resolve_model_alias` collapses secondary ->
    # primary whenever `secondary_enabled` is false — so without this the
    # subject of the test disappears and both arms answer from :8096. That is
    # not hypothetical: it went red the day the slot was switched off
    # (2026-09-20) with nothing about the router changed. The eval's own module
    # docstring already notes that the flag "outranks this in both arms"; this
    # makes the test say so too.
    from app import config as app_config
    monkeypatch.setitem(app_config.CONFIG, "secondary_enabled", True)
    monkeypatch.setattr(app_config, "_ALIAS_REWRITES_LOGGED", set(), raising=False)
    seen = _fake_transport(monkeypatch)
    item = {"id": "voice-1", "input": "some reply with `code` in it", "anchors": []}

    secondary = ev.run_trial("voice", item, "secondary")
    primary = ev.run_trial("voice", item, "primary")

    assert ":8091" in secondary["endpoint"], secondary["endpoint"]
    assert ":8096" in primary["endpoint"], primary["endpoint"]
    assert len(seen) == 2


def test_recording_installs_its_spy_and_takes_it_back_off(monkeypatch):
    """`app.secondary_models` reads the body once and drops `usage`; the tee
    has to hand back a body that still reads, and — because it patches a
    stdlib module attribute — put it back.

    Asserted from inside the call: the fake records whatever `urlopen` was
    while production called it. If the spy were never installed that reads
    the fake; if the `finally` never restored it, the post-call read finds
    the spy. Either failure fails one of the two asserts, which the earlier
    version of this test (comparing against `original`, which both states
    differ from) could not do.
    """
    calls: list = []
    body = json.dumps({"choices": [{"message": {"content": "[Lloyd] a fact"}}],
                       "usage": {"completion_tokens": 7, "prompt_tokens": 1234}}).encode()

    def fake(req, *args, **kwargs):
        calls.append(urllib.request.urlopen)     # what production saw at call time
        return _FakeResponse(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    item = {"id": "voice-2", "input": "text with | a table", "anchors": []}

    row = ev.run_trial("voice", item, "secondary")

    assert row["output_tokens"] == 7              # from the engine's own usage
    assert row["prompt_tokens"] == 1234
    assert len(calls) == 1
    assert calls[0] is not fake, "the spy was never installed over the transport"
    assert urllib.request.urlopen is fake, "the spy was left installed on urllib"


def test_arms_are_separate_catches_one_engine_answering_both():
    merged = {"voice": {"secondary": {"endpoints": ["http://h:8091/v1"]},
                        "primary": {"endpoints": ["http://h:8091/v1"]}}}
    separate, why = ev.arms_are_separate(merged)
    assert separate is False and "8091" in why

    split = {"voice": {"secondary": {"endpoints": ["http://h:8091/v1"]},
                       "primary": {"endpoints": ["http://h:8096/v1"]}}}
    assert ev.arms_are_separate(split) == (True, "every arm reached a distinct engine endpoint")


# ── A retired slot is policy, not a broken override (#1328) ──────────────
#
# Since 2026-09-20 (`551e9044`) GPU 2 holds djev, `secondary_enabled` is false,
# and `resolve_model_alias('secondary')` answers `primary` for both arms. The
# sweep ran anyway on 2026-09-21: 120 trials, 148.6 s, all of them against
# :8096, and the cause it printed read "the alias override did not take — a
# broken box". The override took perfectly. These four tests are the difference
# between "there is no second engine" and "the per-arm redirect silently
# failed", which is the one thing the instrument used to be unable to say.

def _voice_items(tmp_path) -> tuple[Path, Path]:
    """A one-item item set `main()` would genuinely run: a session in a tmp
    store whose input hash the real builder computed. So the only thing that
    can stop the sweep in the tests below is the slot switch — not a missing
    item, and not a hash that moved."""
    import hashlib

    import yaml

    session = {"messages": [
        {"role": "user", "content": "what moved in the retrieval eval?"},
        {"role": "assistant", "content": "The runner imports `defaults` from "
                                         "app.config | the compare step read 3 "
                                         "stale columns"},
    ]}
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "parked-night.json").write_text(json.dumps(session), encoding="utf-8")
    text = ev.job_inputs("voice", session)
    items = {"voice": {"label": ev.JOB_LABELS["voice"], "items": [{
        "id": "voice-1", "job": "voice", "source_session": "parked-night",
        "input_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
        "input_chars": len(text), "anchors": ev.anchors_from(text)}]}}
    items_path = tmp_path / "items.yaml"
    items_path.write_text(yaml.safe_dump(items, sort_keys=False), encoding="utf-8")
    return items_path, sessions


def _sweep_args(items_path: Path, sessions: Path, label: str, repeats: int = 3) -> list[str]:
    return ["--repeats", str(repeats), "--jobs", "voice", "--label", label,
            "--items", str(items_path), "--sessions-dir", str(sessions), "--quiet"]


def _slot_off(monkeypatch, tmp_path):
    """Slot switched off, artifacts in a tmp dir, engines stood in for.
    Returns (requests the fake recorded, the artifact dir, the sweep args)."""
    from app import config as app_config

    items_path, sessions = _voice_items(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(ev, "OUT_DIR", out_dir)
    # Pinned OFF rather than inherited: the live box has had this flag false
    # since 2026-09-20, so a test that relied on the ambient value would pass
    # for the wrong reason and could not fail if the guard were removed.
    monkeypatch.setitem(app_config.CONFIG, "secondary_enabled", False)
    seen = _fake_transport(monkeypatch)
    return seen, out_dir, _sweep_args(items_path, sessions, "parked-night")


def test_a_disabled_secondary_slot_exits_7_before_the_first_request(monkeypatch, tmp_path):
    """The retirement must be its own exit code, reached with zero requests.

    `3`, `4`, `5` and `6` are taken (arms merged, below the repeat floor, no
    downstream denominator, router disagreement) and `2` is argparse's error
    code, so 7 is the first code no other path in the script returns. A
    nightly that cannot tell the two causes apart reports a policy as a broken
    box, which is what #1328 is about.
    """
    seen, _out_dir, argv = _slot_off(monkeypatch, tmp_path)

    assert ev.EXIT_SLOT_DISABLED == 7
    assert ev.main(argv) == ev.EXIT_SLOT_DISABLED
    assert seen == [], "a sweep with no second engine must not spend one primary-engine trial"


def test_the_retirement_notice_names_the_slot_and_flag_not_the_override(monkeypatch, tmp_path,
                                                                        capsys):
    """The sentence a run reports is the half that was wrong. `arms_are_separate`
    says what a merged pair means; on this path there is no merged pair, and the
    message has to name the switch a person can flip instead."""
    _seen, _out_dir, argv = _slot_off(monkeypatch, tmp_path)

    assert ev.main(argv) == 7
    out = capsys.readouterr().out
    assert "agent-llm-secondary" in out, out
    assert "secondary_enabled" in out, out
    assert "trials: 0" in out, out
    assert "did not take" not in out, "the override wording belongs to exit 3 only"
    assert "both arms reached" not in out, out
    assert "NOT DECISION GRADE" not in out, out


def test_a_parked_slot_night_writes_no_results_and_no_report(monkeypatch, tmp_path):
    """Nothing is written, so a night that measured nothing cannot enter the
    trend. The 2026-09-21 same-engine night produced secondary-minus-primary
    score gaps of −2.92 (focus), +3.75 (facts), −0.84 (capture), +0.83 (voice)
    and 0.00 (title) against a `KEEP_MARGIN_POINTS` of 5.0 — the widest is 75 %
    of the margin, from two arms that are provably one engine — and every
    `keep` the trend recorded inside ±4 points was noise that the artifact made
    look like a measurement."""
    _seen, out_dir, argv = _slot_off(monkeypatch, tmp_path)

    assert ev.main(argv) == 7
    left = sorted(p.name for p in out_dir.iterdir())
    assert left == [], f"a skipped night left artifacts behind: {left}"


def test_the_one_engine_case_still_exits_3_while_the_slot_is_enabled(monkeypatch, tmp_path,
                                                                    capsys):
    """Exit 3 keeps the meaning it was written for: the slot is up and the arms
    still collapsed onto one endpoint. Simulated the way that box actually
    happens — `models.secondary` pointed at the primary's base_url, so
    `resolve_model_alias` returns `secondary` and both arms dial :8096 anyway.

    This is the case the retirement code must NOT swallow, which is why it runs
    the real sweep through the real transport rather than asserting on a string.
    """
    from app import config as app_config

    items_path, sessions = _voice_items(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(ev, "OUT_DIR", out_dir)
    monkeypatch.setitem(app_config.CONFIG, "secondary_enabled", True)
    monkeypatch.setattr(app_config, "_ALIAS_REWRITES_LOGGED", set(), raising=False)
    # MODEL_CONFIGS IS CONFIG["models"] (app/config.py:286) — one dict, so
    # mispointing the slot's endpoint is a config fact, not a patched function.
    primary_base = app_config.MODEL_CONFIGS["primary"]["base_url"]
    merged = dict(app_config.MODEL_CONFIGS["secondary"], base_url=primary_base)
    monkeypatch.setitem(app_config.MODEL_CONFIGS, "secondary", merged)
    seen = _fake_transport(monkeypatch)

    assert ev.main(_sweep_args(items_path, sessions, "merged-arms", repeats=1)) == 3
    out = capsys.readouterr().out
    assert "NOT DECISION GRADE" in out, out
    assert "both arms reached" in out, out
    assert seen, "the sweep really ran — that is what makes exit 3 a result about engines"


def test_plan_pairs_the_arms_adjacent_and_repeats_each_item():
    items = [{"id": "title-1"}, {"id": "title-2"}]
    trials = ev.plan(3, ["title"], items)
    assert len(trials) == 12
    pairs = [(t[1]["id"], t[2]) for t in trials]
    assert pairs == [("title-1", "secondary"), ("title-1", "primary"),
                     ("title-2", "secondary"), ("title-2", "primary")] * 3
    # Back-to-back same-arm runs let one engine's warm cache flatter it.
    for index in range(0, len(pairs), 2):
        assert pairs[index][1] == "secondary" and pairs[index + 1][1] == "primary"


# ── The item set ─────────────────────────────────────────────────────────

def test_item_set_covers_every_routed_job_with_provenance():
    items = ev.load_items(ev.ITEMS_PATH)
    for job in ev.JOBS:
        assert items[job], f"{job} has no items — the five-job coverage is the item's §1"
        for item in items[job]:
            assert item["source_session"], "capture items must name their session"
            assert len(item["input_sha256"]) == 16
            assert item["input_chars"] > 50
            assert len(item["anchors"]) >= ev.MIN_ANCHORS, (
                f"{item['id']} has nothing to check the output against")




def test_a_moved_input_is_refused_rather_than_rerun(monkeypatch, tmp_path):
    session = {"session_id": "s-1", "messages": [{"role": "user", "content": "hello"}]}
    (tmp_path / "s-1.json").write_text(json.dumps(session), encoding="utf-8")
    item = {"id": "voice-9", "job": "voice", "source_session": "s-1",
            "input_sha256": "0" * 16, "input_chars": 5}

    with pytest.raises(SystemExit) as raised:
        ev.resolve_input(item, tmp_path)
    assert "pinned" in str(raised.value)


def test_focus_replica_keeps_production_transcript_shape():
    """`_maybe_extract_focus` builds its transcript inline, so the eval's
    replica is only faithful while it matches those literals. Read them out
    of the production source: change the window there and this fails."""
    source = inspect.getsource(post_capture._maybe_extract_focus)
    assert "messages[-10:]" in source, "production window moved; replica is stale"
    assert 'f"{role}: {text[:200]}"' in source, "production per-turn cap moved"
    assert '"USER"' in source and '"ASSISTANT"' in source

    messages = [{"role": "user", "content": "x" * 500}] + [
        {"role": "assistant", "content": "<context>injected</context>"},
        {"role": "assistant", "content": "tail"},
    ]
    built = ev.focus_transcript(messages)
    assert built.startswith("USER: ")
    assert "x" * 200 in built and "x" * 201 not in built
    assert "injected" not in built
    assert built.splitlines()[-1] == "ASSISTANT: tail"


def test_anchors_are_the_repeated_subject_not_the_harness_log_noise():
    """The reason this derivation is frequency-based.

    The identifier version picked the paths the harness injects, and both
    engines then scored 0.0 on good summaries. If a future edit drags path
    matching back in, this fails.
    """
    transcript = (
        "/home/alansrobotlab/lloyd/_pipeline/tasks/bg-20260908-192331-4acd31.log "
        "task_id task_notification exit_code "
        "the retrieval eval kept failing on the retrieval baseline, "
        "so the retrieval eval compare step was rewritten, and the "
        "eval trend was recorded against the retrieval baseline again"
    )
    anchors = ev.anchors_from(transcript)
    assert "retrieval" in anchors and "eval" in anchors
    assert not any("/" in a or ".log" in a or a == "task_id" for a in anchors)
    assert all(len(a) >= 4 for a in anchors) and len(anchors) <= 6


def test_an_anchor_that_appears_once_is_not_an_anchor():
    assert ev.anchors_from("a session about retrieval eval eval eval baseline") == ["eval"]


# ── Scorers: each must be able to fail, and must fail on the real defect ─

@pytest.mark.parametrize("job", sorted(ev.JOBS))
def test_a_compliant_output_scores_high_and_clean(job):
    row = ev.score(job, GOOD[job], anchors_from_good(job))
    assert row["format_ok"] is True, row["format_fails"]
    assert row["over_long"] is False
    assert row["composite"] >= 90.0, row


def anchors_from_good(job):
    """Anchors as they would have been pinned from each GOOD output's source."""
    return {
        "title": ["vault", "retrieval"],
        "capture": ["retrieval", "eval", "agent_mcp.vault"],
        "facts": ["retrieval", "agent_mcp.vault", "ndcg10"],
        "focus": ["retrieval", "eval", "baseline"],
        "voice": ["retrieval", "eval", "ndcg"],
    }[job]


def test_title_fails_the_shape_a_weak_engine_actually_produces():
    row = ev.score("title", "This conversation was about a fix to the nightly "
                            "retrieval eval which took a while.", ["retrieval"])
    assert row["format_ok"] is False
    assert any("words" in f for f in row["format_fails"])
    assert row["composite"] < 60.0


def test_capture_flags_the_over_long_entry_defect():
    """Standing problem #4 is over-long, over-duplicated entries; the capture
    scorer is the one that has to see it."""
    long_text = GOOD["capture"] + " " + GOOD["capture"]
    row = ev.score("capture", long_text, ["retrieval eval"])
    assert row["format_ok"] is False          # 6 sentences, prompt asks 2-4
    assert any("sentences" in f for f in row["format_fails"])

    padded = "The eval moved. " + ("padding words fill the budget. " * 40)
    assert ev.score("capture", padded + " Done here.", ["eval"])["over_long"] is True


def test_duplicate_rate_catches_a_restatement_not_only_an_exact_repeat():
    text = ("The runner imports from agent_mcp.vault now.\n"
            "The runner imports from agent_mcp.vault now today.\n"
            "The compare step read three missing columns.")
    assert ev.duplicate_rate(text) > 0.0
    assert ev.duplicate_rate(GOOD["capture"]) == 0.0


def test_facts_require_the_entity_prefix_that_production_would_hide():
    """`_sync_secondary_fact_extraction` files a prefix-less line under
    "Lloyd", so production never shows this failure and the scorer has to."""
    row = ev.score("facts", "the runner imports from agent_mcp.vault\n"
                            "the compare step read missing columns\n"
                            "ndcg10 is 0.594", ["ndcg10"])
    assert row["format_ok"] is False
    assert any("[Entity]" in f for f in row["format_fails"])


def test_focus_rejects_a_numbered_list_and_a_runon_topic():
    assert any("numbered" in f for f in
               ev.score("focus", "1. retrieval eval\n2. compare step\n3. baselines",
                        ["eval"])["format_fails"])
    assert any("words" in f for f in ev.score(
        "focus", "retrieval eval defaults\ncompare\nbaselines", ["eval"])["format_fails"])


def test_voice_rejects_markup_the_rewrite_exists_to_remove():
    row = ev.score("voice", "Here is the result:\n- `ndcg10` at 0.594\n| a | b |", ["ndcg10"])
    assert row["format_ok"] is False
    assert any("markup" in f for f in row["format_fails"])


def test_anchor_recall_is_a_groundedness_floor():
    drifted = "The session covered several topics and reached a good outcome overall."
    assert ev.anchor_recall(drifted, ["retrieval", "agent_mcp.vault"]) == 0.0
    assert ev.anchor_recall(GOOD["capture"], ["retrieval", "eval"]) == 1.0
    # A path anchor still counts when the output names the file, not the path
    assert ev.anchor_recall(GOOD["capture"], ["/srv/app/post_capture.py"]) == 0.0
    assert ev.anchor_recall("post_capture kept timing out", ["/srv/app/post_capture.py"]) == 1.0
    assert ev.anchor_recall(GOOD["capture"], []) is None


def test_an_empty_output_scores_zero_rather_than_raising():
    """A job that returns None failed; that has to be countable, not an
    exception that drops the trial out of the aggregate."""
    for job in ev.JOBS:
        assert ev.score(job, "", ["x"])["composite"] == 0.0
    assert ev._as_text(None) == ""


# ── Aggregation and the routing decision ────────────────────────────────

def _arms(sec_score=88.0, pri_score=90.0, sec_defect=0.0, pri_defect=0.0,
          n=6, items=2, repeats=3):
    """Arm summaries shaped exactly like `summarise`'s output.

    `n` is trials, `repeats` is times one input was replayed on the arm. They
    are separate arguments because `decide` reads only `repeats`: 6 trials of
    6 different items at 1 repeat each is more data and still cannot route.
    """
    def arm(score, defect, endpoint):
        return {"n": n, "items": items, "repeats_min": repeats,
                "repeats_max": repeats, "score_mean": score, "score_min": score - 2,
                "score_max": score + 2,
                "score_spread": 4.0, "score_stdev": 1.5, "format_ok_rate": 1.0,
                "defect_rate": defect, "anchor_recall_mean": 0.8, "wall_s_mean": 1.0,
                "wall_s_max": 1.2, "output_tokens_mean": 60.0, "endpoints": [endpoint],
                "errors": 0, "judge_mean": None}
    return {"secondary": arm(sec_score, sec_defect, "http://h:8091/v1"),
            "primary": arm(pri_score, pri_defect, "http://h:8096/v1")}


def test_secondary_keeps_a_job_it_is_within_margin_on():
    decision = ev.decide("title", _arms(sec_score=86.0, pri_score=90.0))
    assert decision["decision"] == "keep_secondary"
    assert decision["margin_points"] == ev.KEEP_MARGIN_POINTS


def test_a_job_flips_when_the_gap_costs_more_than_the_margin():
    decision = ev.decide("facts", _arms(sec_score=70.0, pri_score=95.0))
    assert decision["decision"] == "flip_to_primary"
    assert decision["score_gap_primary_minus_secondary"] == 25.0
    assert "latency" in decision["reason"]


def test_a_job_flips_on_defects_even_at_equal_quality():
    """Equal mean score with more over-long and duplicated entries is still
    the defect the item was written about."""
    decision = ev.decide("capture", _arms(sec_score=90.0, pri_score=90.0,
                                          sec_defect=0.4, pri_defect=0.05))
    assert decision["decision"] == "flip_to_primary"
    assert decision["defect_gap_secondary_minus_primary"] == pytest.approx(0.35)


def test_the_margin_block_quotes_no_unmeasured_latency_cost():
    """#827: the decision policy priced its margin on "roughly 40x ... (9.01 vs
    357.5 tok/s in eval/measurements.json)", a file no tree ever held; the
    module's own 2026-09-18 run measured the secondary at ~1.09x the
    primary's throughput. The block now carries that measurement."""
    source = Path(ev.__file__).read_text(encoding="utf-8")
    assert "measurements.json" not in source
    assert "9.01" not in source
    assert "2026-09-18" in source and "1.09" in source


def _report_meta():
    return {"started_at": "2026-09-18T12:08:00+00:00", "repeats": 3,
            "items_per_job": 1, "judge": False,
            "arms_separate": {"ok": True, "detail": "d"},
            "router_agreement": {"ok": True, "detail": "d"},
            "on_primary_now": []}


def _report_section(report: str, job: str) -> str:
    start = report.index(f"(`{job}`)")
    end = report.find("\n## ", start)
    return report[start:end if end != -1 else None]


def test_a_both_zero_defect_rate_is_said_not_scored():
    """#827: with both arms at defect_rate 0.0 the composite hands each the
    same 40 points for length and uniqueness, so the decision and the report
    row must say the axis discriminated nothing rather than let it read as a
    measurement. A job where either arm had a defect says no such thing."""
    silent = _arms(sec_score=88.0, pri_score=90.0)
    loud = _arms(sec_score=88.0, pri_score=90.0, sec_defect=0.2)
    decisions = [ev.decide("title", silent), ev.decide("capture", loud)]
    assert ev.DEFECT_AXIS_SILENT in decisions[0]["reason"]
    assert decisions[0]["defect_axis_discriminated"] is False
    assert ev.DEFECT_AXIS_SILENT not in decisions[1]["reason"]
    assert "defect_axis_discriminated" not in decisions[1]

    # Below the repeat floor the verdict is insufficient_data, and the axis
    # is still silent: the reason says both things.
    thin = ev.decide("voice", _arms(n=2, items=2, repeats=1))
    assert thin["decision"] == "insufficient_data"
    assert ev.DEFECT_AXIS_SILENT in thin["reason"]

    report = ev.render_report({"title": silent, "capture": loud}, decisions,
                              _report_meta())
    title_rows = [ln for ln in _report_section(report, "title").splitlines()
                  if ln.startswith("|")]
    assert any(ev.DEFECT_AXIS_SILENT in ln for ln in title_rows)
    assert ev.DEFECT_AXIS_SILENT not in _report_section(report, "capture")


def test_fewer_than_three_repeats_of_the_same_input_declines_to_route():
    """The source walkthrough's own lesson: a harness pairing scored *below*
    the plain model with no error bars shown. One sample of an input cannot
    tell noise from a worse engine.

    The arm here carries 12 trials — more data than the keeping and flipping
    cases above, which have 6 — because it is 12 inputs at one repeat each.
    The first version of `decide` compared `n` against the floor, so exactly
    this shape passed: the 1-repeat pilot it ran produced a
    `flip_to_primary` for voice off a single sample per input. Counting
    repeats instead of trials is the difference between a wide run and a
    repeated one, and only the second one has a spread."""
    decision = ev.decide("voice", _arms(sec_score=10.0, pri_score=100.0,
                                        n=12, items=12, repeats=1))
    assert decision["decision"] == "insufficient_data"
    assert "repeats_min=1" in decision["reason"]
    assert "12 item" in decision["reason"]
    assert "n=12" not in decision["reason"], "the reason must not restate a trial count"


def test_summarise_reports_spread_wall_seconds_and_tokens():
    rows = [
        {"job": "voice", "item": "voice-1", "alias": "secondary", "composite": 90.0, "wall_s": 1.0,
         "output_tokens": 50, "defect": False, "format_ok": True, "anchor_recall": 1.0,
         "endpoint": "http://h:8091/v1", "error": None},
        {"job": "voice", "item": "voice-2", "alias": "secondary", "composite": 70.0, "wall_s": 3.0,
         "output_tokens": 70, "defect": True, "format_ok": False, "anchor_recall": 0.5,
         "endpoint": "http://h:8091/v1", "error": None},
        {"job": "voice", "item": "voice-1", "alias": "primary", "composite": 95.0, "wall_s": 0.5,
         "output_tokens": 30, "defect": False, "format_ok": True, "anchor_recall": 1.0,
         "endpoint": "http://h:8096/v1", "error": None},
    ]
    arms = ev.summarise(rows)["voice"]["secondary"]
    assert arms["n"] == 2
    # Both rows name no item key, so they are two unseen inputs at one
    # repeat each: the arithmetic that stops a wide run counting as a repeated one.
    assert arms["items"] == 2 and arms["repeats_min"] == 1 and arms["repeats_max"] == 1
    assert arms["score_mean"] == 80.0 and arms["score_spread"] == 20.0
    assert arms["wall_s_mean"] == 2.0 and arms["wall_s_max"] == 3.0
    assert arms["output_tokens_mean"] == 60.0
    assert arms["defect_rate"] == 0.5
    assert arms["endpoints"] == ["http://h:8091/v1"]


# ── The nightly wiring ───────────────────────────────────────────────────

def test_a_nightly_autonomy_task_runs_the_eval():
    """Verification the acceptance names: the eval is run by something."""
    runners = [p for p in AUTONOMY_DIR.glob("*.md")
               if "secondary_routing_eval.py" in p.read_text(encoding="utf-8")]
    assert runners, "no autonomy task invokes eval/secondary_routing_eval.py"


def test_that_nightly_task_is_one_the_scheduler_will_actually_run():
    """`autonomy.py:433` refuses a task with no skill_name — "it will NEVER
    run" — so a task file alone is not a nightly run."""
    import yaml

    runners = [p for p in AUTONOMY_DIR.glob("*.md")
               if "secondary_routing_eval.py" in p.read_text(encoding="utf-8")]
    assert runners
    body = runners[0].read_text(encoding="utf-8")
    front = yaml.safe_load(body.split("---\n")[1])
    assert front["frequency"] == "daily"
    assert front.get("skill_name"), f"{runners[0].name} would never run"
    # `status` is asserted by `test_the_task_dispatches_only_when_there_is_a_slot_to_measure`
    # below, keyed on the slot: pinning it to `up_next` unconditionally is the
    # claim acceptance clause 4 makes false — #1328 parks the job while
    # `secondary_enabled` is false, and a nightly with no second engine is the
    # burn, not the coverage.
    assert "--repeats" in body and "3" in body, "the nightly run must be decision-grade"


def test_the_task_dispatches_only_when_there_is_a_slot_to_measure():
    """The dispatch rule and the instrument's rule have to be one rule.

    Runnable while an engine exists to compare against; a status in
    `DISPATCH_STOPPING_STATUSES` (`autonomy.py:218` — `draft` or `paused`, the
    two values the scheduler drops) while it does not. Asserted as a pair
    rather than as one literal so it goes red in both directions: a slot
    re-armed without re-arming the job stops being measured nightly, and a
    slot retired without parking the job burns 120 primary-engine trials every
    morning on a cause string that is false.
    """
    import yaml

    import autonomy
    from app import llm_slots

    runners = [p for p in AUTONOMY_DIR.glob("*.md")
               if "secondary_routing_eval.py" in p.read_text(encoding="utf-8")]
    assert runners
    front = yaml.safe_load(runners[0].read_text(encoding="utf-8").split("---\n")[1])
    status = str(front["status"])
    if llm_slots.is_enabled("agent-llm-secondary"):
        assert status in autonomy.RUNNABLE_STATUSES, (
            f"the secondary slot is enabled but task #85 ({status}) is not dispatching: "
            "the routing decision would stop being measured nightly")
    else:
        assert status in autonomy.DISPATCH_STOPPING_STATUSES, (
            f"task #85 is {status} with `secondary_enabled` false: both arms reach the "
            "primary by policy, so the sweep produces 120 same-engine trials and a "
            "cause string that is false (#1328). Park it until a second engine exists.")


def test_the_skill_states_the_exit_codes_and_the_frozen_trend():
    """The skill is what a nightly run obeys: the "alias override did not take"
    sentence lived there too (SKILL.md:69-70), so a code-only fix would leave a
    run reporting the false cause from the prose while the script said the true
    one. Clause 5 of #1328.
    """
    skill = (Path.home() / "obsidian" / "skills" / "secondary-routing-eval" / "SKILL.md")
    body = skill.read_text(encoding="utf-8")
    assert "agent-llm-secondary" in body, "the skill never names the slot it is talking about"
    assert "nothing to pair against" in body, "exit 7 is undocumented"
    assert "the slot is enabled" in body, (
        "exit 3 is still described as an unqualified merged-arms failure, which is the "
        "wording that made a parked slot read as a broken box")
    assert "frozen at 2026-09-20" in body, "the six-night trend is not marked frozen"


def test_the_paths_the_nightly_run_names_are_in_the_checkout():
    """The two tests above only ever grep the task file for the script's name,
    which is why four nights of task #85 "succeeded" against a script that was
    not in the tree — the run found the name, improvised a scratch worktree of
    an unmerged branch, and reported numbers measured on `app/` three days older
    than production. Checking the string is not checking the instrument: every
    path this eval's own nightly instructions name has to resolve in the
    checkout that is running the test."""
    named = {
        "the sweep itself": ROOT / "eval" / "secondary_routing_eval.py",
        "the pinned item set": ROOT / "eval" / "secondary_generation_items.yaml",
        "the routing decision": ROOT / "eval" / "secondary-routing" / "decisions.yaml",
    }
    missing = {label: str(p) for label, p in named.items() if not p.exists()}
    assert missing == {}, (
        f"task #85's instructions name paths absent from this checkout: {missing}. "
        "A nightly that cannot run the script is a failed run, not a result — "
        "see item #1240."
    )


# ── Executing a flip: the router is written by the measurement or not at all ─

def test_the_shipped_route_flips_only_the_job_the_measurements_flipped():
    """The verdict as shipped state, not as behaviour one fake transport
    happens to exercise.

    Four decision-grade runs (2026-09-16 twice, 09-17, 09-18) put `title`
    13.3 to 19.2 composite points outside a 5.0-point margin — the widest and
    most stable gap on the board, with secondary format compliance 42-58 %
    against the primary's 100 %. The other four jobs were keeps on all four
    runs, so they still ask for the cheap engine. Asserted on the alias
    because `_engine_for` only says what the pin asks for:
    `resolve_model_alias` keeps the last word, and `secondary_enabled: false`
    still lands every job on the primary.
    """
    assert sm.JOBS_ON_PRIMARY == frozenset({"title"})
    assert sm._engine_for("title") == "primary"
    for job in ("capture", "facts", "focus", "voice"):
        assert sm._engine_for(job) == "secondary", (
            f"{job} was a keep on all four decision-grade runs and must still "
            "ask for the secondary")


def test_the_endpoint_resolver_will_not_guess_a_job():
    """`job` is required with no default, and that is the seam: a call site
    that forgot to name its job would otherwise get the secondary, measure on
    the secondary, and read as routed while the router file pinned it."""
    param = inspect.signature(sm._endpoint).parameters["job"]
    assert param.default is inspect.Parameter.empty, (
        "_endpoint grew a default job; a forgotten job argument would then "
        "resolve silently instead of raising TypeError")


def test_each_production_call_site_names_its_own_job():
    """The pin routes the job a call site *says*, so a mislabelled call site
    routes the wrong work: `_sync_secondary_title` passing `"capture"` would
    leave the flipped job on the cheap engine while every test that only
    exercises the happy path stayed green. Read from the source, because a
    monkeypatched transport proves what one job did, not what each one claims."""
    for job, fn in (("title", sm._sync_secondary_title),
                    ("capture", sm._sync_secondary_capture_call),
                    ("facts", sm._sync_secondary_fact_extraction),
                    ("focus", sm._sync_secondary_focus_extraction),
                    ("voice", sm._sync_secondary_voice_summary)):
        body = inspect.getsource(fn)
        assert f'_endpoint("{job}")' in body, (
            f"{fn.__name__} does not pass its own job name to _endpoint")


# Trees the scan is not about. `web` holds no Python; `_pipeline` holds the
# generated layer, which is gitignored — and being gitignored is what let this
# scan ship red: a gate runs in a clean worktree where `_pipeline` does not
# exist, so only the live checkout ever walked it. It is skipped for two
# independent reasons, both pinned below: it holds two *directories* named
# `Router.py` (`_pipeline/vault-derived/facts/` and
# `_pipeline/backups/preidrepair-20260909T012928Z/facts/`), and `Path.rglob`
# matches directories, so reading every `*.py` match raised `IsADirectoryError`;
# and it holds generated *copies* of real source files (`okf-check/<date>/skills/
# system-health-check/system_health_check.py`), which would report an offence at
# a path nobody can fix.
_SCAN_SKIP_DIRS = frozenset({".venvs", "node_modules", ".git", ".worktrees",
                             "_pipeline"})


def _router_call_sites(root: Path = ROOT) -> list[str]:
    """Every call to `app.secondary_models._endpoint` under `root`, as
    `path:lineno`, with the offence for any that names no job.

    Found syntactically, not by grep: a grep for `_endpoint(` also lands on
    `_summary_endpoint(`, `_resolve_endpoint(`, `_consolidation_endpoint(` and
    `rewrite_endpoint(`, four unrelated resolvers that live in this tree, and a
    check with that many false positives is a check that gets widened away. A
    call counts only when the file actually brings the router's name in — a
    local `from app.secondary_models import _endpoint` or a qualified
    `secondary_models._endpoint(...)` — which is what makes the scan about this
    router rather than about a name.

    `root` is a parameter so the scan can be pointed at a synthetic tree: the
    live checkout has no offending call site to find, so a scan that can only be
    aimed at the live tree is a check nobody can prove still checks. Every test
    below runs the same code path the live assertion runs.
    """
    import ast

    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if _SCAN_SKIP_DIRS & set(rel.parts):
            continue
        # `is_file`, not just the skip set: `rglob("*.py")` yields directories
        # too, and the live tree has two of them. `OSError` covers the rest —
        # a generated tree that vanishes mid-scan between listing and reading —
        # and the synthetic-offender test is what says skipping cannot make the
        # scan quietly inert.
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            continue
        imports_it = any(
            isinstance(n, ast.ImportFrom)
            and (n.module or "").endswith("secondary_models")
            and any(a.name == "_endpoint" for a in n.names)
            for n in ast.walk(tree))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            qualified = (isinstance(f, ast.Attribute) and f.attr == "_endpoint"
                         and isinstance(f.value, ast.Name)
                         and f.value.id == "secondary_models")
            if not ((isinstance(f, ast.Name) and f.id == "_endpoint" and imports_it)
                    or qualified):
                continue
            first = node.args[0] if node.args else None
            names_a_job = (isinstance(first, ast.Constant) and isinstance(first.value, str)
                           and bool(first.value.strip()))
            if not names_a_job:
                offenders.append(f"{rel}:{node.lineno}")
    return offenders


def test_no_caller_of_the_router_asks_it_to_guess_a_job():
    """`_endpoint` taking a required job is only a seam while every caller uses
    it. Landed, the signature change silently broke the backlog pair-judge in
    `scripts/automod/cluster.py` (`_judge_endpoint`, the second caller outside
    `app/secondary_models.py`) — `TypeError: _endpoint() missing 1 required
    positional argument` at the first ambiguous backlog edge, caught by the
    judge loop's own `except Exception`, and reported as one more triage error
    rather than as the router having changed under it. The same shape as the
    four phantom nights: a consumer reaching for something the tree no longer
    offers, and a handler that turns the failure into a shrug.

    This is why the scan is repo-wide and not a list: the five call sites inside
    `app/secondary_models.py` are pinned by name by the test above, and the two
    outside it (`app/uptake.py`, `scripts/automod/cluster.py`) are exactly the
    ones nobody was watching.
    """
    offenders = _router_call_sites()
    assert offenders == [], (
        f"these call sites call app.secondary_models._endpoint without naming a job: "
        f"{offenders}. `_endpoint` has no default on purpose — pass the job the work "
        "is (`uptake`, `cluster_judge`, or one of the five routed jobs).")


def test_the_caller_scan_neither_crashes_on_nor_reports_a_directory_named_py(tmp_path):
    """The tree this scan walks is not only source. The live checkout holds two
    *directories* whose names end in `.py` — `_pipeline/vault-derived/facts/Router.py`
    and its twin under `_pipeline/backups/` — and `Path.rglob("*.py")` matches
    directories, so `read_text` raised `IsADirectoryError` and the test that is
    supposed to guard every future routing change was red at HEAD for a reason no
    diff could fix. It shipped green because a gate runs in a clean worktree and
    `_pipeline` is gitignored, so the worktree never had the directory to trip on.

    A generated entity directory is not a call site, so the correct answer for
    this tree is an empty list and no escaped OS error.
    """
    for d in (tmp_path / "facts" / "Router.py",
              tmp_path / "_pipeline" / "backups" / "preidrepair-x" / "facts" / "Router.py"):
        (d / "state").mkdir(parents=True)
        (d / "state" / "note.md").write_text("# Router\n", encoding="utf-8")
    # A source file beside the directory, so the walk has to get past the
    # directory and keep reading to answer at all.
    ok = tmp_path / "pkgs" / "ok.py"
    ok.parent.mkdir()
    ok.write_text("from app.secondary_models import _endpoint\n"
                  "\n"
                  "def route():\n"
                  '    return _endpoint("title")\n', encoding="utf-8")
    # And a call site inside the skipped generated tree, which is deliberately
    # not reported: `_pipeline` holds copies of real scripts, so an offence there
    # names a path nobody can edit.
    stale = tmp_path / "_pipeline" / "reflection" / "stale-copy.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("from app.secondary_models import _endpoint\n"
                     "\n"
                     "def judge():\n"
                     "    return _endpoint()\n", encoding="utf-8")

    assert _router_call_sites(tmp_path) == []


def test_the_caller_scan_survives_a_file_it_cannot_read(tmp_path):
    """The `OSError` half, pinned on its own because a directory is stopped by
    the `is_file` guard before it reaches the read: a file that exists but cannot
    be opened — the shape of a generated tree being rewritten while this scan
    walks it — must not raise out of the test either. `unreadable.py` holds a
    compliant call site rather than an offence, so the assertion below means the
    same thing whether the chmod took effect or the reader could open it anyway:
    the scan got past it and still found the real offender.
    """
    unreadable = tmp_path / "unreadable.py"
    unreadable.write_text("from app.secondary_models import _endpoint\n"
                          "\n"
                          "def route():\n"
                          '    return _endpoint("uptake")\n', encoding="utf-8")
    unreadable.chmod(0)
    (tmp_path / "found.py").write_text(
        "from app.secondary_models import _endpoint\n"
        "\n"
        "def judge():\n"
        "    return _endpoint()\n", encoding="utf-8")

    assert _router_call_sites(tmp_path) == ["found.py:4"]


def test_the_caller_scan_finds_a_call_site_that_omits_the_job(tmp_path):
    """The other half of the seam, and the reason `_router_call_sites` takes a
    root: aimed at the live checkout it reports none, which is what it should do
    and also indistinguishable from a scan that reads nothing. So the proof that
    it still catches the offence it exists for has to be a tree built here, run
    through the same function the live assertion above calls.

    Three files, one directory named `Router.py`, and both spellings of the call
    the scan recognises. The expectation is an exact list, so a scan that flags
    the compliant file, misses the qualified-form offender, or reports the
    directory is caught just as surely as one that finds nothing.
    """
    (tmp_path / "pkgs").mkdir()
    (tmp_path / "facts" / "Router.py").mkdir(parents=True)
    (tmp_path / "pkgs" / "bad.py").write_text(
        "from app.secondary_models import _endpoint\n"
        "\n"
        "def judge(edge):\n"
        "    return _endpoint()\n", encoding="utf-8")
    (tmp_path / "pkgs" / "ok.py").write_text(
        "from app.secondary_models import _endpoint\n"
        "\n"
        "def route():\n"
        '    return _endpoint("title")\n', encoding="utf-8")
    (tmp_path / "pkgs" / "qualified.py").write_text(
        "import app.secondary_models as secondary_models\n"
        "\n"
        "def ask():\n"
        "    return secondary_models._endpoint(None)\n"
        "\n"
        "def ask_ok():\n"
        '    return secondary_models._endpoint("uptake")\n', encoding="utf-8")

    assert _router_call_sites(tmp_path) == ["pkgs/bad.py:4", "pkgs/qualified.py:4"]


def _router_copy(tmp_path):
    """A copy of the real router file, so a pin edit can be tested without
    touching the tree the rest of the suite reads."""
    import shutil

    dest = tmp_path / "secondary_models.py"
    shutil.copy(ev.ROUTER_PATH, dest)
    return dest


def test_a_pin_is_written_to_the_router_source_and_read_back(tmp_path):
    """The seam between the two halves of the clause: a decision the eval
    reached has to land in the file production reads.

    Read-back goes through `router_source`, i.e. the source text, so this fails
    if the pin line stops being one line or stops being a literal set — the two
    ways this edit would otherwise go quietly inert.
    """
    copy = _router_copy(tmp_path)
    ev.set_router_pins(["title", "capture"], copy)
    assert ev.router_source(copy) == frozenset({"title", "capture"})
    line = [ln for ln in copy.read_text().splitlines()
            if ln.startswith("JOBS_ON_PRIMARY")]
    assert line == ["JOBS_ON_PRIMARY: frozenset = frozenset({'capture', 'title'})"], line
    # An empty pin is its own shape, not `frozenset({})` — the file must stay
    # importable, which a caller that renders no elements would break.
    ev.set_router_pins([], copy)
    assert ev.router_source(copy) == frozenset()
    assert copy.read_text().count("frozenset({'capture', 'title'})") == 0


def test_a_pin_edit_refuses_a_job_that_is_not_routable(tmp_path):
    """A typo in a job name would otherwise pin a job that does not exist and
    leave the real one on the cheap engine while reading as handled."""
    copy = _router_copy(tmp_path)
    before = ev.router_source(copy)
    with pytest.raises(ValueError, match="not routable jobs"):
        ev.set_router_pins(["captre"], copy)
    assert ev.router_source(copy) == before


def test_a_pin_edit_refuses_a_router_file_it_cannot_write_safely(tmp_path):
    copy = tmp_path / "secondary_models.py"
    copy.write_text("import x\n", encoding="utf-8")
    with pytest.raises(LookupError, match="no longer states"):
        ev.set_router_pins(["title"], copy)


def test_the_pinned_router_and_the_recorded_decision_agree_in_the_tree():
    """The clause's second half, checked as shipped state.

    `decisions.yaml` is written only by a decision-grade run and the router
    only by `--pin`, which reads that file. If these two disagree in the tree,
    either someone hand-edited `app/secondary_models.py` past the measurement
    or a flip was landed without its confirming run — which is the exact defect
    #551 was opened to remove, so it is a red test rather than a warning.
    """
    recorded = ev.load_decisions()
    assert recorded, (f"{ev.decisions_path()} is missing: the routing decision has to be "
                      "a checked-in artifact, not a summary paragraph")
    assert recorded["repeats_at_floor"] and recorded["arms_separate"], (
        f"the recorded run was not decision-grade: {recorded}")
    in_tree = ev.router_source(ev.ROUTER_PATH)
    assert in_tree == frozenset(recorded["on_primary"]), (
        f"router pins {sorted(in_tree)} but the measurement says "
        f"{sorted(recorded['on_primary'])}")


def test_the_pin_command_refuses_without_a_recorded_decision(monkeypatch, tmp_path):
    """No measurement, no engine change — and the refusal has to come from the
    command, not from a reviewer remembering to check."""
    monkeypatch.setattr(ev, "load_decisions", lambda path=None: None)
    before = ev.router_source(ev.ROUTER_PATH)
    assert ev.main(["--pin", "title"]) == 3
    assert ev.router_source(ev.ROUTER_PATH) == before, (
        "the refusal must leave the router file exactly as it found it")


def test_the_pin_command_refuses_a_set_the_measurement_did_not_reach(monkeypatch):
    """A pin that names more jobs than the run recommends is a hand-edit with
    the command's name on it."""
    monkeypatch.setattr(ev, "load_decisions",
                        lambda path=None: {"on_primary": ["capture"],
                                           "repeats_at_floor": True,
                                           "arms_separate": True})
    assert ev.main(["--pin", "title,capture"]) == 3


#: The 2026-09-19 windows, rebuilt as per-day entry counts from the live notes
#: the run read, so the fixture is the sample the item names rather than an
#: approximation of it. `--downstream 2026-09-19` printed
#: `after 2026-09-13..2026-09-19: 20 entries over 6/7 notes` against
#: `before 2026-09-06..2026-09-12: 548 entries over 7/7 notes` and exited 0 —
#: a flagged-rate gap of 0.20 against 0.024 carried by 4 flagged entries.
#: (Re-run today the same command prints 21 entries over 7/7 notes: the 09-19
#: note took a capture after that run, so the after window grew by one entry
#: and one note. Both sides of that drift are under the floor, which is the
#: point.)
_AFTER_2026_09_19 = {"2026-09-13": 2, "2026-09-14": 1, "2026-09-15": 3,
                     "2026-09-16": 5, "2026-09-17": 6, "2026-09-18": 3}
#: 33 + 39 + 105 + 341 + 15 + 8 + 7 = 548. The shape is the finding: ~130 a day
#: through 09-09 (09-09 alone holds 341), then 15 / 8 / 7 on 09-10 / 09-11 /
#: 09-12, which is the user-session-only capture split, not a collapse.
_BEFORE_2026_09_19 = {"2026-09-06": 33, "2026-09-07": 39, "2026-09-08": 105,
                      "2026-09-09": 341, "2026-09-10": 15, "2026-09-11": 8,
                      "2026-09-12": 7}


def _capture_notes(memory_dir: Path, counts: dict[str, int],
                   long_on: str | None = None) -> None:
    """Write daily notes holding exactly `counts[day]` auto-captured entries.

    Bodies are one line each on purpose: `duplicate_rate` compares lines within
    one body, so a single-line body is duplicate-free by construction and the
    only signal in the fixture is over length, which is applied to the one entry
    `long_on` names — one flagged entry, so the rate it produces is 1/entries
    and can be computed by hand against the floor. Counts are what the floor
    reads, so they are written exactly rather than roughly.
    """
    memory_dir.mkdir(parents=True, exist_ok=True)
    for day, n in counts.items():
        blocks = []
        for i in range(n):
            if day == long_on and i == 0:
                body = "x" * (ev.LENGTH_BUDGETS["capture"] + 1)
            else:
                body = f"{day} entry {i}: the nightly job landed and recorded its row"
            blocks.append(f"### Session {day} entry-{i}\n{body}")
        (memory_dir / f"{day}.md").write_text("\n".join(blocks) + "\n", encoding="utf-8")


def _downstream_2026_09_19(tmp_path: Path, *, after: bool = True) -> Path:
    mem = tmp_path / "memory"
    if after:
        _capture_notes(mem, _AFTER_2026_09_19)
    _capture_notes(mem, _BEFORE_2026_09_19)
    return mem


def _downstream_args(mem: Path) -> list[str]:
    return ["--downstream", "2026-09-19", "--memory-dir", str(mem)]


def test_the_downstream_windows_are_adjacent_and_never_overlap():
    """Seven days against the prior seven. A baseline window that shared a day
    with its own comparison would read a week against itself, and the drift
    would look like an improvement."""
    after = ev.window_dates("2026-09-15", 7)
    before = ev.preceding_window("2026-09-15", 7)
    assert len(after) == len(before) == 7
    assert not set(after) & set(before)
    assert after[0] > before[-1]
    from datetime import date, timedelta
    assert (date.fromisoformat(after[0]) - date.fromisoformat(before[-1])
            == timedelta(days=1))


# ── The downstream floor: a rate over a denominator that cannot resolve ───

def test_the_downstream_check_refuses_the_sample_that_carried_the_8x_headline(tmp_path, capsys):
    """The command that must stop reporting a verdict on 20 entries.

    The 2026-09-19 run printed an after-arm flagged rate of 0.20 against a
    before-arm 0.024 and exited 0, because the only guard was `entries == 0`
    and 20 is not 0. Four flagged entries were the entire headline. The fixture
    is those windows (`_AFTER_2026_09_19`, `_BEFORE_2026_09_19`), so this test
    fails the moment the guard stops reading the counts.
    """
    mem = _downstream_2026_09_19(tmp_path)

    assert ev.main(_downstream_args(mem)) == ev.EXIT_NO_DOWNSTREAM == 5
    out = capsys.readouterr().out
    assert "NO DOWNSTREAM VERDICT" in out, out
    assert "after window holds 20 captured entries over 6/7 notes" in out, out
    assert f"floor is {ev.MIN_DOWNSTREAM_ENTRIES} captured entries" in out, out


def test_a_window_at_the_floor_still_gets_its_verdict(tmp_path, capsys):
    """The floor disables only the comparison it cannot support, not the check.

    Both arms sit at or above 42 entries here — the after arm exactly at the
    floor, so the boundary is pinned as inclusive and not accidentally shifted
    by one — and both rates must still print with exit 0. The after arm's one
    over-long entry out of 42 is a flagged rate of 0.024, the same number the
    before arm reported over 548 entries: the coincidence is the lesson, since
    at 42 entries one entry is the whole rate.
    """
    mem = tmp_path / "memory"
    at_floor = {d: 6 for d in ev.window_dates("2026-09-19", 7)}
    above = {d: 8 for d in ev.preceding_window("2026-09-19", 7)}
    assert sum(at_floor.values()) == ev.MIN_DOWNSTREAM_ENTRIES
    _capture_notes(mem, at_floor, long_on=min(at_floor))
    _capture_notes(mem, above)

    assert ev.main(_downstream_args(mem)) == 0
    out = capsys.readouterr().out
    assert "NO DOWNSTREAM VERDICT" not in out, out
    assert "after  2026-09-13..2026-09-19: 42 entries over 7/7 notes" in out, out
    assert "before 2026-09-06..2026-09-12: 56 entries over 7/7 notes" in out, out
    assert "flagged 0.024" in out, out


def test_the_entry_floor_is_the_resolution_bound_it_claims_to_be():
    """42 is derived, not chosen, and the derivation has to break if the
    baseline it was derived from moves.

    The rate the check compares against is the before arm's own flagged rate:
    0.024 on the 2026-09-19 run (13 flagged of 548). A window of n entries can
    express rates only in steps of 1/n, so below n = 42 a single flagged entry
    lands above 0.024 on its own and the arm cannot report a rate that small at
    all. 42 is the smallest n with 1/n <= 0.024; 41 is not (1/41 = 0.0244).
    """
    n = ev.MIN_DOWNSTREAM_ENTRIES
    assert n == 42
    assert 1 / n <= 0.024 < 1 / (n - 1), "the floor stopped being the resolution bound"


def test_the_refusal_moves_when_the_floor_moves(tmp_path, monkeypatch, capsys):
    """The guard reads the constant, not a literal in the branch. Same 20-entry
    sample, floor dropped to 20, and the verdict comes back — which is what
    makes the refusal test above a test of the floor rather than of something
    else on the path."""
    mem = _downstream_2026_09_19(tmp_path)
    monkeypatch.setattr(ev, "MIN_DOWNSTREAM_ENTRIES", 20)

    assert ev.main(_downstream_args(mem)) == 0
    assert "NO DOWNSTREAM VERDICT" not in capsys.readouterr().out, "the floor is hardcoded"


def test_the_refusal_names_the_user_session_split_so_nobody_re_diagnoses_the_drop(
        tmp_path, capsys):
    """A refusal that only prints a count gets re-diagnosed; this one has to
    close the two wrong answers.

    The 96 % fall in captured entries starts on 2026-09-10, which reads as a
    broken capture pipeline or a broken reader. It is neither: commit `410e203`
    restricted the daily note, the secondary summary and fact extraction to user
    sessions and moved background exports to `sessions-background/`. So the
    refusal names the commit, the date and the split — and says plainly that the
    population changed, which forecloses both false diagnoses.
    """
    mem = _downstream_2026_09_19(tmp_path)

    assert ev.main(_downstream_args(mem)) == 5
    out = capsys.readouterr().out
    for needle in ("410e203", "2026-09-10", "user session", "sessions-background"):
        assert needle in out, (needle, out)
    assert "Not a capture-pipeline defect" in out, out
    assert "not a reader break" in out, out


def test_an_empty_window_is_refused_by_the_same_floor_and_names_the_same_cause(tmp_path, capsys):
    """The guard this replaces fired only at zero. Zero still fires — and now
    says why, instead of printing "a window has 0 captured entries" for a run to
    report as a capture outage."""
    mem = _downstream_2026_09_19(tmp_path, after=False)

    assert ev.main(_downstream_args(mem)) == 5
    out = capsys.readouterr().out
    assert "after window holds 0 captured entries over 0/7 notes" in out, out
    assert "410e203" in out, out


def test_the_floor_refusal_is_the_process_exit_code_a_run_sees(tmp_path):
    """Task #85 and the `--pin` confirmation text invoke this as a process, so
    the refusal has to be a non-zero status and not merely a return value a
    test can read. This is the argv-in / exit-code-out boundary itself.
    """
    mem = _downstream_2026_09_19(tmp_path)

    proc = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "secondary_routing_eval.py"),
         "--downstream", "2026-09-19", "--memory-dir", str(mem)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300)

    assert proc.returncode == 5, proc.stdout[-800:] + proc.stderr[-800:]
    assert "NO DOWNSTREAM VERDICT" in proc.stdout, proc.stdout[-800:]


def test_the_instruction_that_prescribes_downstream_states_the_floor():
    """#1262's triage note: the operator is told to run `--downstream` by the
    report `--pin` writes, not by the skill — a grep of the skill's text finds
    the exit-code list and no command. So a floor the instruction never mentions
    would still leave the operator reading a refusal as a passed check. The
    rendered bullet interpolates the constant, which is why this greps for the
    name rather than a number: the instruction cannot drift from the guard.
    """
    source = (ROOT / "eval" / "secondary_routing_eval.py").read_text(encoding="utf-8")
    at = source.find("the downstream check is one command")
    assert at != -1, "the --pin confirmation text no longer prescribes the downstream check"
    block = source[at:at + 1200]
    assert "MIN_DOWNSTREAM_ENTRIES" in block, block
    assert "NO DOWNSTREAM VERDICT" in block, block


# ── #1445: the gap between the recorded decision and the endpoint is not silent ─

def _endpoint_under(monkeypatch, caplog, secondary_enabled):
    """`_endpoint("voice")` with the slot flag pinned, every log line captured
    fresh — the generic alias line is once per process, so its set is reset."""
    import logging
    from app import config as app_config
    monkeypatch.setitem(app_config.CONFIG, "secondary_enabled", secondary_enabled)
    monkeypatch.setattr(app_config, "_ALIAS_REWRITES_LOGGED", set(), raising=False)
    caplog.clear()
    # Root level: the generic line is `app.config`'s logger, the new one is
    # `lloyd-server`'s, and both have to be seen in one capture.
    with caplog.at_level(logging.INFO):
        url, model = sm._endpoint("voice")
    return url, model, [r.getMessage() for r in caplog.records]


def test_a_kept_job_landing_on_the_primary_is_logged_by_name(monkeypatch, caplog):
    """Clause 1. `voice` is a recorded `keep_secondary`
    (eval/secondary-routing/decisions.yaml), and with the slot off it answers
    from the primary — for four days after 2026-09-20 with nothing in any log
    naming the job. One line names the job, the engine the decision chose, and
    the URL and model reached; the generic per-alias line is still there."""
    url, model, lines = _endpoint_under(monkeypatch, caplog, secondary_enabled=False)

    named = [line for line in lines if "'voice'" in line]
    assert len(named) == 1, lines
    assert "'secondary'" in named[0], named[0]
    assert url in named[0] and "'primary'" in named[0], named[0]
    assert any("model alias 'secondary' -> 'primary'" in line for line in lines), (
        "the generic alias rewrite line is still the once-per-process record")


def test_the_voice_job_is_pinned_on_its_resolved_url_under_both_flag_values(
        monkeypatch, caplog):
    """Clause 2: the URL, not the alias. `_engine_for("voice")` says
    `secondary` whatever the flag says, which is how the tree held a
    `keep_secondary` decision and a primary endpoint with every guard green."""
    url, model, lines = _endpoint_under(monkeypatch, caplog, secondary_enabled=False)
    assert (url, model) == ("http://127.0.0.1:8096/v1/chat/completions", "primary")

    url, model, lines = _endpoint_under(monkeypatch, caplog, secondary_enabled=True)
    assert (url, model) == ("http://127.0.0.1:8091/v1/chat/completions", "secondary")
    assert not [line for line in lines if "'voice'" in line], (
        "with the slot on the decision and the endpoint agree; nothing to say")


def test_the_static_guards_still_check_what_they_checked():
    """Clause 3: the runtime view is added beside the static one, not instead
    of it — the two #551 guards keep their subjects."""
    assert sm._engine_for("voice") == "secondary"
    recorded = ev.load_decisions()
    assert ev.router_source(ev.ROUTER_PATH) == frozenset(recorded["on_primary"])
