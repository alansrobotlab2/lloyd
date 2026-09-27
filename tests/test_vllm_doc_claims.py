"""#1339 / #1627: `architecture/vllm.md` §10 is a standing acceptance test, so its
numbers have to follow the fleet and the data they were read from.

It went stale twice over: its bar was written for `workers.slots` 2 while the
fleet ran 6 slots and four autocode rounds, and its only counted reading was a
2026-09-11 spot check off a five-minute in-memory ring. Nothing could fail when
either moved. Since #1627 it has a third duty, which is the one that bites:

* the counted reading is a full UTC day **with real chat in it** — 2026-09-25,
  the first day the counter has that tests criterion (a) at all — and §10 marks
  (a), (b), (c) and (d) pass or fail beside it, so a mark that no longer
  follows its own numbers fails here;
* every figure in that reading, including (d)'s two-request throughput, is
  recomputed by `scripts/vllm_prefix_miss_window.derive` over the extract the
  doc names. Only the two history paragraphs — the 2026-09-11 spot reading and
  §6.1's own baseline — quote numbers the extract cannot cover, and they are
  labelled history in the text;
* the fleet shape §10 names is read from `config.yaml`, the gate from the same
  file, and the ring's retention from `app/engine_pressure`.
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
# #1627: the day with chat in it. The 09-23 extract stays in the tree as the
# reading this one replaced (see test_the_replaced_reading_is_not_still_quoted).
EXTRACT = ROOT / "tests" / "fixtures" / "vllm_prefix_miss_2026-09-25.json"
OLD_EXTRACT = ROOT / "tests" / "fixtures" / "vllm_prefix_miss_2026-09-23.json"
CRITERIA = ("a", "b", "c", "d")


def _section(n: int) -> str:
    text = DOC.read_text(encoding="utf-8")
    m = re.search(rf"^## {n}\. .*?(?=^## )", text, re.M | re.S)
    assert m, f"§{n} not found"
    return " ".join(m.group(0).split())


def _counted_reading(s10: str) -> str:
    """Just the counted reading and the four criteria marked against it — not
    §10's bar, its history, or the open questions that carry no figures."""
    start = s10.index("The counted reading:")
    end = s10.index("YaRN's tool-choice effect")
    return s10[start:end]


@pytest.fixture(scope="module")
def s10() -> str:
    return _section(10)


@pytest.fixture(scope="module")
def reading(s10) -> str:
    return _counted_reading(s10)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def derived(cfg) -> dict:
    gate = float(cfg["workers"]["kv_gate"]["max_kv_usage"])
    return W.derive(json.loads(EXTRACT.read_text(encoding="utf-8")), gate=gate)


def _n(x: int) -> str:
    return f"{x:,}"


def _verdict(reading: str, letter: str) -> str:
    """Each criterion is marked exactly once in the counted reading, and which
    mark it is comes out of the test — so a figure edited by hand without the
    data moving, or a verdict edited without the mark agreeing, both fail."""
    hits = re.findall(rf"\({letter}\) (passes|fails)", reading)
    assert len(hits) == 1, f"({letter}) must carry exactly one pass/fail mark, found {hits}"
    return hits[0]


_NUMERALS = "zero one two three four five six seven eight nine ten eleven twelve".split()


def _word(n: int) -> str:
    return _NUMERALS[n] if 0 <= n < len(_NUMERALS) else str(n)


# ── clause 1: the fleet shape, and the day the reading is over ────────────

def test_the_bar_names_the_fleet_shape_config_runs(s10, cfg):
    slots = int(cfg["workers"]["slots"])
    rounds = int(cfg["workers"]["sources"]["autocode"]["max_inflight"])
    gate = float(cfg["workers"]["kv_gate"]["max_kv_usage"])
    assert f"`workers.slots` {slots}" in s10
    assert f"`workers.sources.autocode.max_inflight` {rounds}" in s10
    assert f"`workers.kv_gate.max_kv_usage`, {gate:.2f})" in s10
    assert f"slots {slots}, {'four' if rounds == 4 else rounds} rounds" in s10


def test_the_bar_states_d_as_a_measurable_threshold(s10):
    """(d) is a bar the page can be wrong about, so its own text has to name
    the figure and the exclusion rather than gesture at "slow"."""
    assert f"under {W.STALL_TOK_S:.0f} tok/s that is not a cold admission" in s10
    assert "prompt *and* generation" in s10


def test_the_counted_reading_is_a_named_full_utc_day_with_chat_in_it(s10, reading):
    m = re.search(r"The counted reading: (\d{4}-\d\d-\d\d) 00:00 → (\d{4}-\d\d-\d\d) 00:00 UTC", s10)
    assert m, "the window must be named in the text"
    assert m.group(1) >= "2026-09-25", \
        "#1627: the reading is owed over a full UTC day from 2026-09-25 onward"
    assert W._epoch(m.group(2)) - W._epoch(m.group(1)) == 86_400, "one full UTC day"
    ex = json.loads(EXTRACT.read_text(encoding="utf-8"))
    assert ex["window"] == [m.group(1), m.group(2)]
    assert EXTRACT.name in reading and "scripts/vllm_prefix_miss_window.py" in reading
    assert len([t for t in ex["turns"] if t[1] == "chat"]) >= 10, \
        "#1627 clause 1: the day must carry at least 10 chat turns to test (a)"


# ── clause 2: every counted figure is the derivation, and each mark follows ─

def test_the_counted_figures_are_the_derivation(reading, derived):
    d = derived
    assert (f"{d['turns']} turns, {d['measured']} measured, {d['turns_with_misses']} turns "
            f"carrying {d['misses']} misses and {_n(d['reprefill_tokens'])} re-prefilled tokens") in reading
    assert f"worst turn {_n(d['worst_turn_tokens'])}" in reading
    assert _n(d["pool_tokens"]) in reading, "the pool the free-pool arithmetic is against"
    kinds = ", ".join(f"{k} {misses} / {tokens / 1e6:.1f}M"
                      for k, (misses, tokens) in d["by_kind"].items())
    assert f"By session kind (misses / re-prefilled): {kinds}." in reading
    assert (f"{d['miss_events']} miss iterations logged, {d['cold_events']} of them fully cold "
            f"(nothing cached) and {d['miss_events'] - d['cold_events']} partial") in reading


def test_criterion_a_is_marked_and_is_the_chat_miss_count(reading, derived):
    d = derived
    assert f"{d['chat_turns_with_misses']} of the day's {d['chat_turns']} chat turns " \
           f"carries a prefix miss" in reading
    assert _verdict(reading, "a") == ("passes" if d["chat_turns_with_misses"] == 0 else "fails")


_BASELINE_RE = re.compile(
    r"Fleet baseline over \d\d-(\d\d)/(\d\d)[^:]*: (\d+) of [\d,]+ iterations .*?"
    r"— ([\d.]+)M tokens")


def _per_day_budget(s6: str) -> tuple[int, float]:
    """§10's (b) budget is §6.1's baseline divided by the two days it spans,
    counted here from §6.1's own sentence. The baseline itself is the one
    history paragraph on this page — the database it was read from was wiped on
    09-22 and cannot re-derive it — so the arithmetic is the half that can be
    pinned, and pinning it is what stops the two figures drifting apart."""
    m = _BASELINE_RE.search(s6)
    assert m, "§6.1 must state its baseline as N of M iterations over a date range"
    days = int(m.group(2)) - int(m.group(1)) + 1
    return round(int(m.group(3)) / days), round(float(m.group(4)) / days, 1)


def test_criterion_b_is_marked_and_is_the_per_day_budget(reading, derived, s10):
    """(b)'s mark follows the budget, and the budget follows §6.1."""
    s61 = _section(6)
    assert "20.6M tokens" in s61 and "09-08/09" in s61
    miss_budget, token_budget = _per_day_budget(s61)
    assert f"{miss_budget} misses and {token_budget}M tokens a day" in reading
    assert "history now" in reading, "the baseline is labelled, not presented as live"
    d = derived
    assert f"{d['misses']} and {d['reprefill_tokens'] / 1e6:.1f}M in *one* day" in reading
    assert (f"about {d['misses'] / miss_budget:.1f}x the misses and "
            f"{d['reprefill_tokens'] / (token_budget * 1e6):.1f}x the tokens") in reading
    passes = d["misses"] <= miss_budget and d["reprefill_tokens"] <= token_budget * 1e6
    assert _verdict(reading, "b") == ("passes" if passes else "fails")


def test_criterion_c_is_marked_and_is_the_kv_median(reading, derived, cfg):
    d = derived
    assert f"{_n(d['kv_samples'])} such lines" in reading
    assert (f"KV p50 {d['kv_p50']:.2f} / p90 {d['kv_p90']:.2f} / max {d['kv_max']:.2f}") in reading
    assert f"`Running:` {d['running_p50']} at the median" in reading
    assert "GPU KV cache usage" in reading and "agent-llm-primary.log" in reading
    gate = float(cfg["workers"]["kv_gate"]["max_kv_usage"])
    assert _verdict(reading, "c") == ("passes" if d["kv_p50"] < gate else "fails")


def test_criterion_d_is_marked_and_is_the_two_request_split(reading, derived):
    """The whole of #1627's second half: (d) used to be a sentence the page
    asserted. Every figure in it now comes out of the derivation."""
    d = derived
    assert (f"Lines with `Running:` {W.TWO_REQUEST_MIN} or more: "
            f"**{_n(d['two_request_windows'])}**, median **{d['two_request_tok_s_p50']} tok/s** "
            f"combined, slowest **{d['two_request_tok_s_min']}**") in reading
    assert f"under the bar's {W.STALL_TOK_S:.0f} tok/s: **{d['two_request_under_stall']}**" in reading
    assert (f"admission in flight: **{d['two_request_under_stall_cold_admission']}**") in reading
    assert (f"{d['two_request_under_stall_cold_in_flight']} of them with a cold re-admission"
            in reading)
    assert f"{d['two_request_under_stall_cold_prefill']} with a chunked prefill" in reading
    assert f"counted against (d): **{d['two_request_under_stall_not_cold']}**" in reading
    assert _verdict(reading, "d") == ("passes" if d["two_request_under_stall_not_cold"] == 0
                                      else "fails")


def test_every_criterion_carries_exactly_one_mark(reading):
    for letter in CRITERIA:
        _verdict(reading, letter)


# ── the gap join, and what it says on this day ────────────────────────────

def test_the_miss_gaps_are_the_derivation_and_name_a_branch(reading, derived):
    d = derived
    assert d["misses_with_gap"] == d["miss_events"], "every miss joined to its gap"
    assert f"gap p50 {d['gap_s_p50']:.1f} s" in reading
    assert f"p50 {d['miss_kv_gap_p50']:.3f} / p90 {d['miss_kv_gap_p90']:.3f}" in reading
    assert f"max {d['miss_kv_gap_max']:.3f}" in reading
    assert f"{d['misses_gap_over_gate']} of {d['miss_events']} at or over the" in reading
    assert f"**{d['misses_gap_churned_free_pool']} of {d['miss_events']}**" in reading
    assert f"The other {d['miss_events'] - d['misses_gap_churned_free_pool']} misses" in reading


def test_the_eviction_branch_named_is_the_one_the_gaps_support(reading, derived):
    """§10 used to head this "Eviction, but not by pressure at the gate". On
    2026-09-25 that heading was the wrong side of its own numbers — most gaps
    peak over the gate and three churn — so the heading is derived too."""
    d = derived
    assert f"free-pool churn in {_word(d['misses_gap_churned_free_pool'])}" in reading
    assert ("gate-level KV in most of the gaps" in reading) == \
        (d["misses_gap_over_gate"] * 2 > d["miss_events"])
    assert ("none over 0.90" in reading) == (d["misses_gap_over_90"] == 0)


def test_the_draft_group_branch_still_names_this_window(reading, derived):
    """The unannotated draft group is the branch for the misses eviction does
    not explain, so its "not off wholesale" evidence has to be this day's."""
    d = derived
    partial = d["miss_events"] - d["cold_events"]
    assert f"{partial} of the day's {d['miss_events']} misses still read part of " \
           f"their prompt from cache" in reading


# ── what must not come back ───────────────────────────────────────────────

def test_the_2026_09_11_spot_reading_is_history_not_the_reading(s10):
    assert "What the counter says so far" not in s10
    assert "2026-09-11 22:30 UTC" not in s10
    assert re.search(r"2026-09-11 spot reading this page used to quote", s10)


def test_the_replaced_reading_is_not_still_quoted(s10, reading):
    """The 09-23 day is history, and history that quotes counted figures is
    how this page went stale twice. Its extract stays on disk; its numbers do
    not appear in the counted reading."""
    old = W.derive(json.loads(OLD_EXTRACT.read_text(encoding="utf-8")), gate=0.60)
    assert f"{old['turns']} turns" not in reading
    assert _n(old["reprefill_tokens"]) not in reading
    assert f"{old['misses']} miss iterations logged" not in reading
    assert OLD_EXTRACT.name in s10, "the doc says where the replaced reading lives"


def test_no_spot_reading_claims_survive(s10):
    assert "One spot reading" not in s10
    assert "three-minute-old" not in s10


# ── clause 3: (d)'s derivation is a predicate, not a verdict ──────────────

def _ex(**over) -> dict:
    base = {"window": ["2026-09-25", "2026-09-26"], "pool_tokens": [844_969],
            "turns": [], "misses": [], "kv_samples": []}
    base.update(over)
    return base


def _cold_miss(arrived: float, dur_ms: int = 15_000) -> list:
    """A 150k-token iteration that read nothing from cache: §6.2's counted
    miss, and so a cold re-admission by the same floor."""
    return [arrived + dur_ms / 1000, dur_ms, 9, 150_000, 0, "autocode", None]


def test_the_cold_admission_exclusion_separates_a_stall_from_a_prefill():
    """Three windows under 15 tok/s with two requests resident, told apart
    only by their evidence: one has a cold re-admission across the interval,
    one has a chunked prefill climbing across it (KV up, tokens landing on the
    next line), and one is a two-request window that simply ran slowly. Only
    the last counts against (d) — if the predicate excused everything, or
    nothing, this fails."""
    t = W._epoch("2026-09-25") + 3600
    samples = [
        # (a) KV fell across it, no re-admission: a stall, counted.
        [t - 10, 50.0, 2, 5000], [t, 40.0, 2, 100],
        # (b) KV rose across it: a chunked prefill in flight, excused.
        [t + 90, 20.0, 2, 400], [t + 100, 60.0, 2, 100],
        # (c) KV flat and a cold re-admission across the interval: excused.
        [t + 190, 30.0, 2, 9000], [t + 200, 30.0, 2, 50],
        # A healthy two-request window, and a slow single-request one: neither
        # is in the population at all.
        [t + 290, 30.0, 3, 9000], [t + 300, 10.0, 1, 5],
    ]
    d = W.two_request_throughput(_ex(kv_samples=samples, misses=[_cold_miss(t + 190)]))
    assert d["two_request_windows"] == 7, "every line but the one-request one"
    assert d["two_request_under_stall"] == 3
    assert d["two_request_under_stall_cold_in_flight"] == 1
    assert d["two_request_under_stall_cold_prefill"] == 1
    assert d["two_request_under_stall_not_cold"] == 1
    assert d["two_request_under_stall_cold_admission"] == 2
    assert d["two_request_tok_s_min"] == 5.0


def test_the_cold_floor_is_the_miss_definition_not_any_admission():
    """A warm iteration — 150k prompt, three quarters of it cached — is an
    admission but not a *cold* one, so it cannot excuse a slow window."""
    t = W._epoch("2026-09-25") + 3600
    warm = [t + 195, 15_000, 9, 150_000, 120_000, "autocode", None]
    samples = [[t + 190, 30.0, 2, 9000], [t + 200, 30.0, 2, 50]]
    d = W.two_request_throughput(_ex(kv_samples=samples, misses=[warm]))
    assert d["two_request_under_stall_cold_in_flight"] == 0
    assert d["two_request_under_stall_not_cold"] == 1


def test_the_two_request_population_is_the_raw_status_lines():
    """The population and the slow count recomputed straight off the extract's
    rows, independent of the derivation: the numbers §10 quotes are those
    lines, the exclusion cannot swallow one, and there is at least one slow
    window on each day for the exclusion to be about at all."""
    for path in (EXTRACT, OLD_EXTRACT):
        ex = json.loads(path.read_text(encoding="utf-8"))
        t0, t1 = W._epoch(ex["window"][0]), W._epoch(ex["window"][1])
        raw = [s for s in ex["kv_samples"] if t0 <= s[0] < t1]
        two = [s for s in raw if s[2] >= W.TWO_REQUEST_MIN]
        slow = [s for s in two if s[3] / W.STATUS_INTERVAL_S < W.STALL_TOK_S]
        assert slow, f"{path.name}: no slow two-request window, so (d) proves nothing"
        d = W.two_request_throughput(ex)
        assert d["two_request_windows"] == len(two), path.name
        assert d["two_request_under_stall"] == len(slow), path.name
        assert 0 <= d["two_request_under_stall_cold_admission"] <= len(slow), path.name
        assert (d["two_request_under_stall_not_cold"]
                == len(slow) - d["two_request_under_stall_cold_admission"]), path.name


# ── (a)'s counting actually reads the chat rows ───────────────────────────

def test_chat_turns_are_counted_from_the_turn_rows():
    turns = [["2026-09-25T02:08:09", "chat", 0, 0],
             ["2026-09-25T03:08:09", "chat", 2, 250_000],
             ["2026-09-25T04:08:09", "autocode", 1, 120_000]]
    d = W.derive(_ex(turns=turns), gate=0.60)
    assert (d["chat_turns"], d["chat_turns_with_misses"]) == (2, 1)


# ── where a KV history can and cannot come from ───────────────────────────

def test_the_rotation_that_expires_the_engine_lines_is_the_unit_s(s10):
    """(c) and (d) can only be counted while the engine's own status lines are
    on disk, and what bounds them is supervisor's byte budget on this unit, not
    a number of days this page could keep true. The unit's setting is the
    checkable half of that claim; how many days it works out to is not, so the
    doc names the size and the expiry and quotes no day count."""
    conf = (ROOT / "agent-services" / "supervisor" / "conf.d"
            / "agent-llm-primary.conf").read_text(encoding="utf-8")
    m = re.search(r"^stdout_logfile=(\S*agent-llm-primary\.log)$", conf, re.M)
    assert m, "the unit must write its status lines where the window reads them"
    assert "agent-llm-primary.log*" in s10, "and §10 must name where it reads them"
    size = re.search(r"^stdout_logfile_maxbytes=(\d+)(MB|KB|B)$", conf, re.M)
    assert size, "the log has to rotate for this to be an expiry at all"
    mb = int(size.group(1)) * {"MB": 1, "KB": 1 / 1024, "B": 1 / 1024 ** 2}[size.group(2)]
    assert f"{mb:.0f} MB per file" in s10


def test_the_ring_retention_quoted_is_the_modules(s10, cfg):
    window = engine_pressure.DEFAULT_WINDOW_S
    assert f"its ring is {window:.0f} s (`DEFAULT_WINDOW_S`" in s10
    assert float(cfg["engine_pressure"]["window_seconds"]) == window, \
        "config and module disagree; §10 quotes one number for both"
    assert "emptied by every restart" in s10


def test_the_extractor_still_reads_this_day_out_of_the_database():
    """The extract is a snapshot, and a snapshot that no longer matches the
    database behind it is a fabricated one. The turn and miss rows are the
    half that survives rotation (`usage.db` retains them; the engine's own
    status lines roll off in days, which is why they are not compared here).
    Skipped on a machine whose database does not cover the counted day."""
    from app.paths import PRODUCTION_DATA_ROOT

    ex = json.loads(EXTRACT.read_text(encoding="utf-8"))
    db = PRODUCTION_DATA_ROOT / "usage.db"
    if not db.exists():
        pytest.skip(f"no usage.db at {db}")
    got = W.extract(ex["window"][0], ex["window"][1], data_root=PRODUCTION_DATA_ROOT)
    if not got["turns"]:
        pytest.skip(f"{db} holds no turns for {ex['window']}")
    assert got["turns"] == ex["turns"], "the counted day's turns are not the database's"
    assert got["misses"] == ex["misses"], "the counted day's misses are not the database's"


def test_the_derivation_reads_a_status_line_in_local_time():
    """The engine stamps `MM-DD HH:MM:SS` local with no year; the join to UTC
    rows is only right if the zone is applied, not assumed away."""
    line = ("(APIServer pid=2514) INFO 09-23 07:54:10 [loggers.py:315] Engine 000: Avg prompt "
            "throughput: 3289.4 tokens/s, Avg generation throughput: 256.9 tokens/s, Running: 4 "
            "reqs, Waiting: 0 reqs, GPU KV cache usage: 31.1%, Prefix cache hit rate: 64.9%")
    m = W._STATUS_RE.search(line)
    assert m and m.group(1) == "09-23 07:54:10" and m.group(4) == "4" and m.group(5) == "31.1"
