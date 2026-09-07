"""The architecture doc's load-bearing numbers must match the code.

`architecture/self-modification.md` states specific thresholds, path rules and
config placements as fact. A doc that quietly drifts from the implementation is
worse than no doc: it is the thing someone reads at 3am while deciding whether
the watchdog can be trusted.

Only claims where being wrong would mislead an operator are pinned here — not
prose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent-services" / "guardian"))

import policy  # noqa: E402

DOC = ROOT / "architecture" / "self-modification.md"


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
    ("scripts/selfmod/gate.py", "protected"),
    ("app/routers/health.py", "protected"),
    ("requirements.lock", "allowed"),
    ("app/harness/loop.py", "allowed"),
])
def test_path_policy_matches_the_doc(path, expected):
    from scripts.selfmod import spec
    assert spec.classify(path) == expected


# ── §4 the gate ─────────────────────────────────────────────────────────────

def test_the_gate_uses_reflink_always_not_auto():
    """§10: "--reflink=always, not auto — auto degrades to a real 6GB copy
    silently"."""
    gate = (ROOT / "scripts" / "selfmod" / "gate.py").read_text()
    assert "--reflink=always" in gate


def test_the_collected_floor_matches_the_doc():
    from scripts.selfmod import gate
    assert gate.PYTEST_MIN_COLLECTED == 1000


# ── §8.1 regression detector ────────────────────────────────────────────────

def test_latency_is_never_armed():
    """§8.1: "Only latency_ms_avg moved ... and it is never compared"."""
    from workers.sources import selfmod_regression as R
    assert "latency_ms_avg" not in R.ARMED_METRICS
    assert "latency_ms_avg" in R.REPORT_ONLY


def test_all_seven_are_armed_because_the_corpus_is_now_pinned():
    """§8.1: the armed set was wrong twice, in opposite directions.

    Disarming the document metrics was right while the corpus moved between
    arms. Once BOTH halves are pinned, a frozen qmd snapshot and one shared
    LLOYD_CODE_ROOT, the arms agree to 0.0000 on all seven and four repeat
    runs move 0.0000. So all seven are armed again.
    """
    from workers.sources import selfmod_regression as R
    assert set(R.ARMED_METRICS) == {
        "entity_hit_rate", "entity_recall_avg", "fact_entity_recall_avg",
        "ndcg10", "mrr_doc", "doc_hit_rate", "doc_recall_avg"}
    assert set(R.REPORT_ONLY) == {"latency_ms_avg", "n_queries"}


def test_the_pin_is_a_precondition_not_an_optimisation():
    """A comparison falling back to the live daemon would be the broken one
    wearing the fixed one's name."""
    src = (ROOT / "workers" / "sources" / "selfmod_regression.py").read_text()
    assert "PinError" in src and "pinned corpus unavailable" in src


def test_the_grep_corpus_is_pinnable():
    """§8.1: this retriever greps the repository it ships in, so the code
    under test is also part of the corpus it is scored against."""
    assert "LLOYD_CODE_ROOT" in (ROOT / "agent_mcp" / "vault.py").read_text()


def test_both_arms_score_the_same_questions():
    """The baseline arm runs the OLD run_eval.py out of a worktree, carrying
    the OLD query set. Editing the eval would otherwise ask the arms different
    questions and score the difference as a code regression."""
    src = (ROOT / "workers" / "sources" / "selfmod_regression.py").read_text()
    assert "LIVE_QUERIES" in src and '"--queries"' in src


def test_the_noise_file_is_not_in_the_eval_run_record_directory():
    """§13: eval/baselines holds run records, and test_eval_scorer globs it."""
    from workers.sources import selfmod_regression as R
    assert "eval/baselines" not in str(R.NOISE_PATH)


# ── §11 state ───────────────────────────────────────────────────────────────

def test_state_lives_outside_the_repo():
    """§11: so `git reset --hard` and `git clean -fdx` cannot reach it."""
    from scripts.selfmod import state as S
    assert ROOT not in S.STATE_DIR.parents and S.STATE_DIR != ROOT


def test_the_ledger_raises_where_autoresearchs_swallows(tmp_path):
    """§11: the documented divergence."""
    from scripts.selfmod import state as S

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
    from workers.sources import selfmod_regression as R
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
    src = (ROOT / "workers" / "sources" / "selfmod_regression.py").read_text()
    assert 'subject.get("parent")' in src


# ── §6 landing ──────────────────────────────────────────────────────────────

def test_the_landing_is_detached():
    """§2/§6: the promoter restarts the process it is usually called from."""
    assert "spawn_detached" in (ROOT / "agent_mcp" / "selfmod.py").read_text()
    assert "start_new_session=True" in (ROOT / "scripts" / "selfmod" / "state.py").read_text()


def test_only_one_promotion_may_be_under_observation():
    """§6 step 0: a second landing overwrote current.json, so the first never
    settled and the new rollback target had never survived a window."""
    src = (ROOT / "scripts" / "selfmod" / "promote.py").read_text()
    assert "still under observation" in src


def test_the_promoter_refuses_a_commit_the_gate_did_not_judge():
    """§6 step 0."""
    src = (ROOT / "scripts" / "selfmod" / "promote.py").read_text()
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
    from scripts.selfmod import state as S
    assert S.LAST_SETTLED_PATH.name == "last_settled.json"
    assert S.ROLLBACK_REQUEST_PATH.name == "rollback_request.json"
    assert S.EVAL_LAST_PATH.name == "eval_last.json"


def test_the_aggregator_verdict_streak_matches_the_doc():
    """§7.2: the aggregator's verdict is confirmed across ticks."""
    assert policy.MCP_FATAL_STREAK >= 2


def test_a_round_requires_an_observer():
    """§2: "A round runs under Inner Voice, or not at all"."""
    src = (ROOT / "agent_mcp" / "selfmod.py").read_text()
    assert "_inner_voice_gate" in src
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert cfg["selfmod"]["require_inner_voice"] is True
