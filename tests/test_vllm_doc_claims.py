"""#1339: `architecture/vllm.md` §10 is a standing acceptance test, so its
numbers have to follow the fleet and the data they were read from.

It went stale twice over: its bar was written for `workers.slots` 2 while the
fleet ran 6 slots and four autocode rounds, and its only counted reading was a
2026-09-11 spot check off a five-minute in-memory ring. Nothing could fail when
either moved. Here:

* the fleet shape §10 names is read from `config.yaml`, the gate from the same
  file, and the ring's retention from `app/engine_pressure`;
* every figure the counted reading quotes is recomputed by
  `scripts/vllm_prefix_miss_window.derive` over the extract the doc names, so a
  number edited by hand, or a derivation changed under it, fails here;
* §6.1's baseline is still named beside a per-day figure, so a one-day reading
  is never set against a two-day total as an absolute.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from app import engine_pressure
from scripts import vllm_prefix_miss_window as W

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "vllm.md"
EXTRACT = ROOT / "tests" / "fixtures" / "vllm_prefix_miss_2026-09-23.json"


def _section(n: int) -> str:
    text = DOC.read_text(encoding="utf-8")
    m = re.search(rf"^## {n}\. .*?(?=^## )", text, re.M | re.S)
    assert m, f"§{n} not found"
    return " ".join(m.group(0).split())


@pytest.fixture(scope="module")
def s10() -> str:
    return _section(10)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def derived(cfg) -> dict:
    gate = float(cfg["workers"]["kv_gate"]["max_kv_usage"])
    return W.derive(json.loads(EXTRACT.read_text(encoding="utf-8")), gate=gate)


def _n(x: int) -> str:
    return f"{x:,}"


# ── clause 1: the fleet shape ────────────────────────────────────────────

def test_the_bar_names_the_fleet_shape_config_runs(s10, cfg):
    slots = int(cfg["workers"]["slots"])
    rounds = int(cfg["workers"]["sources"]["autocode"]["max_inflight"])
    gate = float(cfg["workers"]["kv_gate"]["max_kv_usage"])
    assert f"`workers.slots` {slots}" in s10
    assert f"`workers.sources.autocode.max_inflight` {rounds}" in s10
    assert f"`workers.kv_gate.max_kv_usage`, {gate:.2f})" in s10
    assert f"slots {slots}, {'four' if rounds == 4 else rounds} rounds" in s10


# ── clause 2: a named window after the slots change ──────────────────────

def test_the_counted_reading_is_a_named_window_after_the_raise(s10):
    m = re.search(r"The counted reading: (\d{4}-\d\d-\d\d) 00:00 → (\d{4}-\d\d-\d\d) 00:00 UTC", s10)
    assert m, "the window must be named in the text"
    assert m.group(1) >= "2026-09-18", "slots went to 6 on 2026-09-18 (e0098082)"
    ex = json.loads(EXTRACT.read_text(encoding="utf-8"))
    assert ex["window"] == [m.group(1), m.group(2)]
    assert EXTRACT.name in s10 and "scripts/vllm_prefix_miss_window.py" in s10


def test_the_counted_figures_are_the_derivation(s10, derived):
    d = derived
    assert (f"{d['turns']} turns, {d['measured']} measured, {d['turns_with_misses']} turns "
            f"carrying {d['misses']} misses and {_n(d['reprefill_tokens'])} re-prefilled tokens") in s10
    assert f"worst turn {_n(d['worst_turn_tokens'])}" in s10
    auto_misses, auto_tokens = d["by_kind"]["autocode"]
    assert f"autocode {auto_misses} misses ({auto_tokens / 1e6:.1f}M)" in s10
    assert "chat" not in d["by_kind"] and "Chat 0" in s10
    assert (f"{d['miss_events']} miss iterations logged, {d['cold_events']} of them fully cold "
            f"(nothing cached) and {d['miss_events'] - d['cold_events']} partial") in s10


def test_the_2026_09_11_spot_reading_is_history_not_the_reading(s10):
    assert "What the counter says so far" not in s10
    assert "2026-09-11 22:30 UTC" not in s10
    assert re.search(r"2026-09-11 spot reading this page used to quote", s10)


# ── clause 3: KV over the same window, and which branch ──────────────────

def test_kv_over_the_window_is_the_derivation(s10, derived):
    d = derived
    assert f"{_n(d['kv_samples'])} such lines" in s10
    assert (f"KV p50 {d['kv_p50']:.2f} / p90 {d['kv_p90']:.2f} / max {d['kv_max']:.2f}") in s10
    assert "GPU KV cache usage" in s10 and "agent-llm-primary.log" in s10


def test_the_miss_gaps_are_the_derivation_and_name_a_branch(s10, derived, cfg):
    d = derived
    assert d["misses_with_gap"] == d["miss_events"], "every miss joined to its gap"
    assert f"gap p50 {round(d['gap_s_p50'])} s" in s10
    assert f"p50 {d['miss_kv_gap_p50']:.3f} / p90 {d['miss_kv_gap_p90']:.3f}" in s10
    assert f"max {d['miss_kv_gap_max']:.3f}" in s10
    assert f"{d['misses_gap_over_gate']} of {d['miss_events']} at or over the" in s10
    assert d["misses_gap_over_90"] == 0 and "none over 0.90" in s10
    churned = d["misses_gap_churned_free_pool"]
    assert f"{churned} of {d['miss_events']}" in s10
    assert f"The other {d['miss_events'] - churned} misses" in s10
    assert "Eviction, but not by pressure at the gate" in s10
    assert _n(d["pool_tokens"]) in s10


def test_no_spot_reading_claims_survive(s10):
    assert "One spot reading" not in s10
    assert "three-minute-old" not in s10


# ── clause 4: where a KV history can and cannot come from ────────────────

def test_the_ring_retention_quoted_is_the_modules(s10, cfg):
    window = engine_pressure.DEFAULT_WINDOW_S
    assert f"its ring is {window:.0f} s (`DEFAULT_WINDOW_S`" in s10
    assert float(cfg["engine_pressure"]["window_seconds"]) == window, \
        "config and module disagree; §10 quotes one number for both"
    assert "emptied by every restart" in s10


# ── clause 5: per day against §6.1 ───────────────────────────────────────

def test_the_baseline_comparison_is_per_day(s10, derived):
    s61 = _section(6)
    assert "194 of 1,754 iterations" in s61 and "20.6M" in s61 and "09-08/09" in s61
    assert "194 misses and 20.6M tokens over *two* days" in s10
    assert "97 misses and 10.3M tokens a day" in s10
    d = derived
    assert f"{d['misses']} and {d['reprefill_tokens'] / 1e6:.1f}M in *one* day" in s10
    assert f"about {d['misses'] / 97:.1f}x the misses and {d['reprefill_tokens'] / 10.3e6:.1f}x the tokens" in s10


def test_the_derivation_reads_a_status_line_in_local_time():
    """The engine stamps `MM-DD HH:MM:SS` local with no year; the join to UTC
    rows is only right if the zone is applied, not assumed away."""
    line = ("(APIServer pid=2514) INFO 09-23 07:54:10 [loggers.py:315] Engine 000: Avg prompt "
            "throughput: 3289.4 tokens/s, Avg generation throughput: 256.9 tokens/s, Running: 4 "
            "reqs, Waiting: 0 reqs, GPU KV cache usage: 31.1%, Prefix cache hit rate: 64.9%")
    m = W._STATUS_RE.search(line)
    assert m and m.group(1) == "09-23 07:54:10" and m.group(4) == "4" and m.group(5) == "31.1"
