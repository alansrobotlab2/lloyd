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
    ("web/src/App.tsx", "denied"),
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


def test_the_armed_metrics_are_the_seven_measured_deterministic_ones():
    from workers.sources import selfmod_regression as R
    assert set(R.ARMED_METRICS) == {
        "entity_hit_rate", "entity_recall_avg", "fact_entity_recall_avg",
        "ndcg10", "mrr_doc", "doc_hit_rate", "doc_recall_avg"}


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
