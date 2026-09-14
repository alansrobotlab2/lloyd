"""The architecture doc's load-bearing numbers must match the code.

`architecture/automod.md` states specific thresholds, path rules and
config placements as fact. A doc that quietly drifts from the implementation is
worse than no doc: it is the thing someone reads at 3am while deciding whether
the watchdog can be trusted.

Only claims where being wrong would mislead an operator are pinned here — not
prose. The one class of prose that IS load-bearing: a sentence that disarms a
metric the code has armed. #505 found `architecture/automod.md` stating the
document metrics were "reported and never fire" for a week after
`08e998a` armed all seven, 31 lines below the sentence that said they were
armed, and an arch-review pass graded the file `current` on top of it. A
reader who believes that sentence skips the gate on a document-ranking change,
or discounts a real 3σ rollback signal as noise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent-services" / "guardian"))

import policy  # noqa: E402

DOC = ROOT / "architecture" / "automod.md"
REGRESSION_SRC = ROOT / "workers" / "sources" / "automod_regression.py"

# The doc's canonical statement of the armed set, written so a test can read
# the number and the names out of it: "**All seven are armed:** `a`, `b`, ...".
ARMED_STATEMENT_RE = re.compile(
    r"\*\*All ([a-z]+) are armed:\*\*\s+((?:`[a-z0-9_]+`,?\s*)+)")

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

# Sentences that tell the reader the document metrics cannot fire. Each one
# was true while the corpus moved between arms and is false under the pin.
DISARMING_PHRASES = ("reported and never fire",
                     "doc-side four are now reported",
                     "Re-arming a doc-side metric")


def _flat(path: Path) -> str:
    """File text with every run of whitespace collapsed to one space.

    The doc wraps at ~80 columns, so "…reported and never\nfire." is one
    sentence split across two lines. Matching raw text would let exactly the
    violation this guards reproduce itself with a newline in the middle.
    """
    return " ".join(path.read_text(encoding="utf-8").split())


def test_the_doc_exists():
    assert DOC.exists()


# ── §7.2 probe budgets ──────────────────────────────────────────────────────

def test_probe_budgets_match_the_doc():
    """"refused: 3 ticks ... timeout: 24 ticks (2 minutes)" """
    assert policy.PROBE_FAIL_STREAK == 3
    assert policy.PROBE_TIMEOUT_STREAK == 24
    assert policy.PROBE_TIMEOUT_STREAK * policy.TICK_SECONDS == 120


def test_probe_timeout_matches_the_doc():
    """"the probe timeout went 2s -> 10s" """
    assert policy.PROBE_TIMEOUT_SECONDS == 10.0


def test_the_rpc_timeout_exceeds_stopwaitsecs():
    """§7.4: a blocking stop legitimately takes stopwaitsecs (15s)."""
    assert policy.SUPERVISOR_RPC_TIMEOUT > 15.0


# ── §7 the unit ─────────────────────────────────────────────────────────────

def test_start_limit_interval_is_in_the_unit_section():
    """§7: "StartLimitIntervalSec belongs in [Unit]".

    In [Service] systemd ignores it and applies the default 5-starts-in-10s
    limit, letting the watchdog rate-limit itself into a failed state.
    """
    unit = (ROOT / "agent-services" / "systemd" / "lloyd-guardian.service").read_text()
    section, placed = None, {}
    for line in unit.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
        elif stripped.startswith("StartLimitIntervalSec"):
            placed[section] = stripped
    assert list(placed) == ["[Unit]"], placed


@pytest.mark.parametrize("conf", ["lloyd-backend", "lloyd-mcp"])
def test_supervisor_confs_stop_process_groups(conf):
    """§7.4: without these a Bash tool's child outlives the stop and can write
    into the tree mid-reset."""
    text = (ROOT / "agent-services" / "supervisor" / "conf.d" / f"{conf}.conf").read_text()
    assert "stopasgroup=true" in text
    assert "killasgroup=true" in text


# ── §10 what Lloyd may change ───────────────────────────────────────────────

@pytest.mark.parametrize("path,expected", [
    ("config.yaml", "denied"),
    ("data/tool_overrides.yaml", "denied"),
    (".env", "denied"),
    ("pytest.ini", "denied"),
    (".gitignore", "denied"),
    ("web/src/App.tsx", "allowed"),
    ("web/package.json", "denied"),
    ("web/vite.config.ts", "denied"),
    ("agent-services/guardian/guardian.py", "protected"),
    ("scripts/automod/gate.py", "protected"),
    ("app/routers/health.py", "protected"),
    ("requirements.lock", "allowed"),
    ("app/harness/loop.py", "allowed"),
])
def test_path_policy_matches_the_doc(path, expected):
    from scripts.automod import spec
    assert spec.classify(path) == expected


# ── §4 the gate ─────────────────────────────────────────────────────────────

def test_the_gate_uses_reflink_always_not_auto():
    """§10: "--reflink=always, not auto — auto degrades to a real 6GB copy
    silently"."""
    gate = (ROOT / "scripts" / "automod" / "gate.py").read_text()
    assert "--reflink=always" in gate


def test_the_collected_floor_matches_the_doc():
    from scripts.automod import gate
    assert gate.PYTEST_MIN_COLLECTED == 1000


# ── §8.1 regression detector ────────────────────────────────────────────────

def test_latency_is_never_armed():
    """§8.1: "Only latency_ms_avg moved ... and it is never compared"."""
    from workers.sources import automod_regression as R
    assert "latency_ms_avg" not in R.ARMED_METRICS
    assert "latency_ms_avg" in R.REPORT_ONLY


def test_all_seven_are_armed_because_the_corpus_is_now_pinned():
    """§8.1: the armed set was wrong twice, in opposite directions.

    Disarming the document metrics was right while the corpus moved between
    arms. Once BOTH halves are pinned, a frozen qmd snapshot and one shared
    LLOYD_CODE_ROOT, the arms agree to 0.0000 on all seven and four repeat
    runs move 0.0000. So all seven are armed again.
    """
    from workers.sources import automod_regression as R
    assert set(R.ARMED_METRICS) == {
        "entity_hit_rate", "entity_recall_avg", "fact_entity_recall_avg",
        "ndcg10", "mrr_doc", "doc_hit_rate", "doc_recall_avg"}
    assert set(R.REPORT_ONLY) == {"latency_ms_avg", "n_queries"}


def test_the_pin_is_a_precondition_not_an_optimisation():
    """A comparison falling back to the live daemon would be the broken one
    wearing the fixed one's name."""
    src = (ROOT / "workers" / "sources" / "automod_regression.py").read_text()
    assert "PinError" in src and "pinned corpus unavailable" in src


def test_the_doc_states_the_armed_set_the_code_has():
    """§8.1: the doc must state the armed set the code actually holds.

    The armed set went seven → three → seven and the doc kept the middle
    value: `architecture/automod.md` said "the armed three" in two places
    while `ARMED_METRICS` had carried seven since `08e998a`. Pinning the
    number and every name is the only way a count in prose stays a fact.
    """
    from workers.sources import automod_regression as R
    match = ARMED_STATEMENT_RE.search(_flat(DOC))
    assert match, ('the doc must state the armed set as '
                   '"**All N are armed:**" followed by the metric names')
    stated_count = NUMBER_WORDS[match.group(1).lower()]
    assert stated_count == len(R.ARMED_METRICS), (
        f"the doc says {match.group(1)} armed, the code has "
        f"{len(R.ARMED_METRICS)}")
    named = re.findall(r"`([a-z0-9_]+)`", match.group(2))
    assert len(named) == len(R.ARMED_METRICS), (
        f"the doc names {len(named)} metrics, the code arms "
        f"{len(R.ARMED_METRICS)}")
    assert set(named) == set(R.ARMED_METRICS)


def test_the_doc_never_disarms_a_metric_the_code_has_armed():
    """§8.1: "reported and never fire" was false for a week and survived review.

    `evaluate` loops `ARMED_METRICS` and appends a rollback reason for any
    drop past `SIGMA_MULTIPLIER × σ`, and `execute` refuses to run without the
    pin (`PinError` → "pinned corpus unavailable"). So a document metric is
    the line that stops a promotion. A doc saying otherwise is not stale
    prose, it is an instruction to skip the gate.
    """
    from workers.sources import automod_regression as R
    assert "mrr_doc" in R.ARMED_METRICS, (
        "this test guards prose that is only false while the document "
        "metrics are armed; if they are genuinely disarmed, rewrite it")
    flat = _flat(DOC)
    for phrase in DISARMING_PHRASES:
        assert " ".join(phrase.split()) not in flat, (
            f"architecture/automod.md tells the reader {phrase!r} while the "
            "code has that metric armed")


def test_the_regression_module_comment_never_disarms_an_armed_metric():
    """§8.1: the same false sentence sat above `FACT_LAYER_METRICS` itself.

    `9a6861c` wrote "Re-arming a doc-side metric means first making the doc
    corpus part of the pairing"; `08e998a` shipped that pairing eleven hours
    later and left the sentence standing. The comment beside the tuple is the
    place a reader looks when they doubt the armed set.
    """
    from workers.sources import automod_regression as R
    assert "mrr_doc" in R.ARMED_METRICS
    flat = _flat(REGRESSION_SRC)
    for phrase in DISARMING_PHRASES:
        assert " ".join(phrase.split()) not in flat, (
            f"workers/sources/automod_regression.py says {phrase!r} while "
            "the code has that metric armed")


def test_the_doc_still_states_the_limits_that_survive_the_pin():
    """§8.1/§13: pinning the corpus fixed one limit and left two standing.

    `latency_ms_avg` still moves ~1.7s between a cold and warm embedding
    cache, and expiring 70% of the active edge set still moves nothing — so
    edge quality has no armed metric. Rewriting the disarmed-metric prose
    must not quietly drop either one.
    """
    from workers.sources import automod_regression as R
    assert "latency_ms_avg" not in R.ARMED_METRICS
    assert "latency_ms_avg" in R.REPORT_ONLY
    flat = _flat(DOC)
    assert "it is never compared" in flat, "the latency limit left the doc"
    assert "Edge quality has no armed metric" in flat, "the edge limit left the doc"
    assert "Graph EDGE quality is not checked by anything" in flat


def test_the_grep_corpus_is_pinnable():
    """§8.1: this retriever greps the repository it ships in, so the code
    under test is also part of the corpus it is scored against."""
    assert "LLOYD_CODE_ROOT" in (ROOT / "agent_mcp" / "vault.py").read_text()


def test_both_arms_score_the_same_questions():
    """The baseline arm runs the OLD run_eval.py out of a worktree, carrying
    the OLD query set. Editing the eval would otherwise ask the arms different
    questions and score the difference as a code regression."""
    src = (ROOT / "workers" / "sources" / "automod_regression.py").read_text()
    assert "LIVE_QUERIES" in src and '"--queries"' in src


def test_the_noise_file_is_not_in_the_eval_run_record_directory():
    """§13: eval/baselines holds run records, and test_eval_scorer globs it."""
    from workers.sources import automod_regression as R
    assert "eval/baselines" not in str(R.NOISE_PATH)


# ── §11 state ───────────────────────────────────────────────────────────────

def test_state_lives_outside_the_repo():
    """§11: so `git reset --hard` and `git clean -fdx` cannot reach it."""
    from scripts.automod import state as S
    assert ROOT not in S.STATE_DIR.parents and S.STATE_DIR != ROOT


def test_the_ledger_raises_where_autoresearchs_swallows(tmp_path):
    """§11: the documented divergence."""
    from scripts.automod import state as S

    blocker = tmp_path / "f"
    blocker.write_text("not a dir", encoding="utf-8")
    with pytest.raises(OSError):
        S.append_event({"event": "x"}, path=blocker / "nested" / "l.jsonl")


# ── §7.2 the third probe class ──────────────────────────────────────────────

def test_the_http_error_budget_matches_the_doc():
    """§7.2: "The backend's own 503 ... gets its own much wider budget: 36 ticks"."""
    assert policy.PROBE_HTTP_ERROR_STREAK == 36
    assert (policy.PROBE_FAIL_STREAK < policy.PROBE_TIMEOUT_STREAK
            < policy.PROBE_HTTP_ERROR_STREAK)


# ── §8 the chronic set expires ──────────────────────────────────────────────

def test_the_chronic_set_expires_daily():
    """§8: "The chronic set expires after 24 hours, in the cache and in the
    process"."""
    assert policy.CHRONIC_REFRESH_SECONDS == 24 * 3600


# ── §8.1 which metrics can actually see the graph ───────────────────────────

def test_the_armed_metrics_are_named_for_what_they_actually_read():
    """§8.1: measured, not assumed.

    Deleting 70% of fact_idx rows moves entity_hit_rate and entity_recall_avg
    well past tolerance. Expiring 70% of ACTIVE EDGES moves nothing at all —
    so these read the fact layer, not the edge set, and calling them "graph
    sensitive" would be the same overclaim this detector exists to avoid.
    """
    from workers.sources import automod_regression as R
    assert not hasattr(R, "GRAPH_SENSITIVE_METRICS"), \
        "the old name overclaims: edge expiry is invisible to every armed metric"
    assert set(R.FACT_LAYER_METRICS) == {
        "entity_hit_rate", "entity_recall_avg", "fact_entity_recall_avg"}
    for doc_side in ("mrr_doc", "ndcg10", "doc_hit_rate"):
        assert doc_side not in R.FACT_LAYER_METRICS


def test_the_eval_refuses_an_empty_corpus_by_default():
    """§8.1: "refuses an empty one unless --allow-empty-corpus is passed"."""
    src = (ROOT / "eval" / "run_eval.py").read_text()
    assert "--allow-empty-corpus" in src
    assert "corpus_ok" in src


def test_the_quality_check_compares_against_the_parent_not_the_lkg():
    """§8.1: after settling the LKG pointer IS the promoted commit."""
    src = (ROOT / "workers" / "sources" / "automod_regression.py").read_text()
    assert 'subject.get("parent")' in src


# ── §6 landing ──────────────────────────────────────────────────────────────

def test_the_landing_is_detached():
    """§2/§6: the promoter restarts the process it is usually called from."""
    assert "spawn_detached" in (ROOT / "agent_mcp" / "automod.py").read_text()
    assert "start_new_session=True" in (ROOT / "scripts" / "automod" / "state.py").read_text()


def test_only_one_promotion_may_be_under_observation():
    """§6 step 0: a second landing overwrote current.json, so the first never
    settled and the new rollback target had never survived a window."""
    src = (ROOT / "scripts" / "automod" / "promote.py").read_text()
    assert "still under observation" in src


def test_the_promoter_refuses_a_commit_the_gate_did_not_judge():
    """§6 step 0."""
    src = (ROOT / "scripts" / "automod" / "promote.py").read_text()
    assert "gate_head" in src and "re-gate before landing" in src


# ── §7.4 route selection ────────────────────────────────────────────────────

def test_a_rollback_can_revert_in_place():
    """§7.4: "Reset when HEAD is still the promotion; revert in place when it
    is not" — nightly jobs commit straight to live main."""
    import rollback as rb
    assert hasattr(rb, "revert_commit")
    assert "surgical" in (ROOT / "agent-services" / "guardian" / "guardian.py").read_text()


# ── §11 state ───────────────────────────────────────────────────────────────

def test_the_new_state_files_exist_where_the_doc_says():
    from scripts.automod import state as S
    assert S.LAST_SETTLED_PATH.name == "last_settled.json"
    assert S.ROLLBACK_REQUEST_PATH.name == "rollback_request.json"
    assert S.EVAL_LAST_PATH.name == "eval_last.json"


def test_the_aggregator_verdict_streak_matches_the_doc():
    """§7.2: the aggregator's verdict is confirmed across ticks."""
    assert policy.MCP_FATAL_STREAK >= 2


def test_a_round_requires_an_observer():
    """§2: "A round runs under Inner Voice, or not at all"."""
    src = (ROOT / "agent_mcp" / "automod.py").read_text()
    assert "_inner_voice_gate" in src
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert cfg["automod"]["require_inner_voice"] is True
