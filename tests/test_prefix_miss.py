"""Prefix-cache miss accounting (`app/prefix_miss.py`).

The 09-09 stall was 100-200k prompts re-prefilled from cold mid-loop, and the
per-iteration numbers that show it (`input_tokens`, `cache_read`) were on disk
all along. What was missing was a count that means something, and three
traps decide whether it does — each pinned here:

* iterations 1-2 are never counted (the engine reports 0 cached for the
  first two requests of any prefix);
* a turn that reads zero everywhere is *unmeasured* — the pre-5531f21
  signature of nothing parsing `cached_tokens` — not a turn full of misses;
* only a miss counts toward `reprefill_tokens`, or a healthy long round's
  appended tail (5-7k an iteration) would sum past the alert line.

The real numbers below are from sessions that are on disk:
`20260910_003817_autocode_308f` iteration 44 (147,117 in / 0 cached) and 45
(149,739 / 76,800).
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from app import engine_pressure as ep
from app import prefix_miss as pm


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    ep.reset()
    monkeypatch.setattr(pm, "_last_announce_at", 0.0)
    monkeypatch.setattr(pm, "_cfg", lambda: {})

    async def _no_reading(*a, **k):
        return None

    # Hermetic: nothing here may reach the live engine or the real fan-out.
    monkeypatch.setattr(ep, "scrape_once", _no_reading)
    monkeypatch.setattr(pm, "_announce", lambda *a, **k: {})
    yield
    ep.reset()


def _u(inp: int, cached: int) -> dict:
    return {"input_tokens": inp, "cache_read": cached}


# ── the count ─────────────────────────────────────────────────────────


def test_iterations_one_and_two_never_count():
    t = pm.TurnMissTracker()
    assert t.observe(1, _u(200_000, 0)) == []
    assert t.observe(2, _u(200_000, 0)) == []
    assert t.counted == 0
    assert t.summary() == {"reprefill_tokens": 0, "prefix_misses": 0}


def test_a_warm_long_loop_reads_zero_even_with_its_appended_tail():
    """Sixty iterations each re-prefilling a 6k tail. Summing every tail —
    the plan's first formula — would read 348k here and fire the alert on a
    perfectly healthy round."""
    t = pm.TurnMissTracker()
    for i in range(1, 61):
        inp = 200_000 + 3_000 * i
        t.observe(i, _u(inp, inp - 6_000 if i > 2 else 0))
    assert t.summary() == {"reprefill_tokens": 0, "prefix_misses": 0}


def test_a_cold_readmission_mid_loop_is_one_miss_costing_its_uncached_tokens():
    t = pm.TurnMissTracker()
    t.observe(1, _u(150_000, 0))
    t.observe(2, _u(152_000, 0))
    t.observe(3, _u(154_000, 150_000))
    released = t.observe(4, _u(156_000, 0))
    assert [m.iteration for m in released] == [4]
    t.observe(5, _u(158_000, 153_600))
    assert t.summary() == {"reprefill_tokens": 156_000, "prefix_misses": 1}


def test_the_line_is_half_the_prompt_at_100k_or_more():
    t = pm.TurnMissTracker(cache_seen=True)
    # The real iteration 45: 51% reused — a partial, not a miss.
    assert t.observe(45, _u(149_739, 76_800)) == []
    # Too small to be the stall, however cold.
    assert t.observe(46, _u(99_000, 0)) == []
    # The real iteration 44: fully cold.
    assert [m.uncached for m in t.observe(44, _u(147_117, 0))] == [147_117]


def test_a_turn_that_reads_zero_everywhere_is_unmeasured_not_cold():
    """The pre-5531f21 signature: nothing parsed `cached_tokens`, and every
    iteration of every session read 0."""
    t = pm.TurnMissTracker()
    for i in range(1, 8):
        assert t.observe(i, _u(120_000 + i, 0)) == []
    assert t.measured is False
    assert t.summary() == {"reprefill_tokens": None, "prefix_misses": None}
    assert len(t.pending) == 5


def test_a_zero_before_the_first_nonzero_read_is_confirmed_by_it():
    t = pm.TurnMissTracker()
    t.observe(1, _u(150_000, 0))
    t.observe(2, _u(151_000, 0))
    assert t.observe(3, _u(152_000, 0)) == []              # miss, or the bug
    released = t.observe(4, _u(153_000, 150_000))           # the field is live
    assert [m.iteration for m in released] == [3]
    assert t.summary() == {"reprefill_tokens": 152_000, "prefix_misses": 1}


def test_usage_numbers_accept_both_spellings():
    assert pm.usage_numbers({"prompt_tokens": 5, "prompt_tokens_cached": 3}) == (5, 3)
    assert pm.usage_numbers({"input_tokens": 7, "cache_read": 2}) == (7, 2)
    assert pm.usage_numbers(None) == (0, 0)


def test_the_label_names_a_background_source_or_a_chat():
    assert pm.label_for_session("20260910_120001_autocode_9f2a") == "autocode"
    assert pm.label_for_session("20260910_120001_ab12cd") == "a chat turn"


def test_record_iteration_logs_each_confirmed_miss():
    events: list = []
    t = pm.TurnMissTracker(cache_seen=True)
    released = pm.record_iteration(t, 4, _u(180_000, 0),
                                   log=lambda n, d: events.append((n, d)))
    assert len(released) == 1
    # The first call of a turn also writes P1's `brain1.turn_start_prefix`.
    assert [n for n, _ in events] == ["brain1.turn_start_prefix", "brain1.prefix_miss"]
    name, data = events[1]
    assert data["uncached_tokens"] == 180_000
    assert data["turn_prefix_misses"] == 1


def test_finish_says_so_when_a_turn_could_not_be_measured():
    events: list = []
    t = pm.TurnMissTracker()
    for i in range(1, 5):
        t.observe(i, _u(130_000, 0))
    summary = pm.finish(t, log=lambda n, d: events.append(n))
    assert summary == {"reprefill_tokens": None, "prefix_misses": None}
    assert events == ["brain1.prefix_miss_unmeasured"]


def test_the_kill_switch_stops_the_accounting(monkeypatch):
    monkeypatch.setattr(pm, "_cfg", lambda: {"enabled": False})
    t = pm.TurnMissTracker(cache_seen=True)
    assert pm.record_iteration(t, 4, _u(180_000, 0)) == []


# ── the bell ──────────────────────────────────────────────────────────


def _missed_turn(uncached: int = 150_000):
    t = pm.TurnMissTracker(label="autocode", cache_seen=True)
    t.observe(5, _u(uncached, 0), duration_ms=15_000, now=1000.0)
    return t, t.confirmed[-1]


def _sample(at: float, running: int) -> None:
    ep.record(ep.Sample(at, at, 0.4, running, 0), window=10_000)


@pytest.fixture
def announced(monkeypatch):
    calls: list = []
    monkeypatch.setattr(pm, "_announce",
                        lambda title, body, voice: calls.append((title, body, voice))
                        or {"journal": True, "desktop": True})
    return calls


def test_a_big_miss_beside_a_neighbour_is_announced_once(announced):
    t, miss = _missed_turn()
    _sample(990.0, 2)                     # during the iteration: it + one other
    assert asyncio.run(pm.maybe_announce(t, miss)) is True
    assert asyncio.run(pm.maybe_announce(t, miss)) is False    # once per turn
    assert len(announced) == 1
    title, body, voice = announced[0]
    assert title == "Prefix-cache miss: autocode"
    assert "150k" in body and "1 other request," in body
    assert voice is False                 # off unless config asks


def test_a_cold_prefill_with_the_engine_otherwise_idle_is_not_news(announced):
    t, miss = _missed_turn()
    _sample(990.0, 1)                     # nobody but itself
    assert asyncio.run(pm.maybe_announce(t, miss)) is False
    assert announced == []


def test_at_the_threshold_is_not_over_it(announced):
    t, miss = _missed_turn(uncached=100_000)
    _sample(990.0, 4)
    assert asyncio.run(pm.maybe_announce(t, miss)) is False


def test_the_cooldown_spans_turns(announced, monkeypatch):
    monkeypatch.setattr(pm, "_cfg", lambda: {"announce_cooldown_seconds": 1800})
    t1, m1 = _missed_turn()
    t2, m2 = _missed_turn()
    _sample(990.0, 3)
    assert asyncio.run(pm.maybe_announce(t1, m1)) is True
    assert asyncio.run(pm.maybe_announce(t2, m2)) is False
    assert len(announced) == 1


def test_no_reading_at_all_means_no_announcement(announced):
    t, miss = _missed_turn()              # no samples, and scrape_once -> None
    assert asyncio.run(pm.maybe_announce(t, miss)) is False


def test_a_miss_no_sample_landed_in_asks_the_engine_after_the_fact(announced, monkeypatch):
    """The miss's own request has finished by then, so whatever is running
    is somebody else."""
    async def _one_running(*a, **k):
        return {"requests_running": 1, "kv_cache_usage": 0.3, "requests_waiting": 0}

    monkeypatch.setattr(ep, "scrape_once", _one_running)
    t, miss = _missed_turn()
    assert asyncio.run(pm.maybe_announce(t, miss)) is True


def test_voice_is_a_config_decision(announced, monkeypatch):
    monkeypatch.setattr(pm, "_cfg", lambda: {"announce_voice": True})
    t, miss = _missed_turn()
    _sample(990.0, 2)
    asyncio.run(pm.maybe_announce(t, miss))
    assert announced[0][2] is True


# ── through a real writer ─────────────────────────────────────────────


def test_the_recorder_counts_a_miss_and_puts_it_on_the_usage_row(announced):
    """End to end through `app/run_recorder.py`: the event log carries the
    miss, the usage row carries the count, the transcript's final stats carry
    both numbers. conftest points all three stores at scratch files."""
    from app import usage_store
    from app import event_log
    from app.run_recorder import record_events
    from app.sessions_io import SESSIONS_DIR, create_session

    sid = "20260910_120000_autocode_beef"
    create_session(sid, platform="worker", model="primary", title="t",
                   source="autocode")
    events = [
        {"type": "assistant_message", "text": "", "iteration": it,
         "tool_calls": [], "duration_ms": 100,
         "usage": {"input_tokens": inp, "output_tokens": 10, "cache_read": cached}}
        for it, inp, cached in [(1, 150_000, 0), (2, 152_000, 0),
                                (3, 154_000, 150_000), (4, 156_000, 0),
                                (5, 158_000, 153_600)]
    ] + [{"type": "result", "stop_reason": "stop", "num_turns": 5,
          "duration_ms": 900, "response_text": "done",
          "usage": {"input_tokens": 158_000, "output_tokens": 50,
                    "cache_read": 457_600}}]

    async def _drive():
        async def _feed():
            for evt in events:
                yield evt
        async for _ in record_events(_feed(), session_id=sid, turn_id="t1",
                                     model="primary", source="autocode"):
            pass

    asyncio.run(_drive())

    row = usage_store._conn().execute(
        "SELECT reprefill_tokens, prefix_misses FROM usage WHERE session_id=?",
        (sid,)).fetchone()
    assert tuple(row) == (156_000, 1)

    logged = "".join(p.read_text() for p in event_log.EVENT_LOGS_DIR.rglob("*")
                     if p.is_file())
    assert "brain1.prefix_miss" in logged

    data = json.loads((SESSIONS_DIR / f"{sid}.json").read_text())
    final = [m for m in data["messages"] if m.get("role") == "assistant"][-1]
    assert final["stats"]["reprefill_tokens"] == 156_000
    assert final["stats"]["prefix_misses"] == 1


# ── #1785: what a cold re-prefill costs, at the chunk budget actually in force ──
#
# The module docstring priced a cold re-prefill at "one 8,192-token chunk per
# engine step" in the present tense. That stopped being the budget on
# 2026-09-10, when commit 62bad14a adopted a 4096-token chunk — half vLLM's
# default — and put it in the primary's supervisord `environment=`, because
# `architecture/vllm.md` §5.1 measured what the width costs a bystander: at the
# wider chunk a cold long prompt held its neighbour's step at 608 ms, at 4096 it
# came down to 333 ms. So the one number that decides *how bad* a miss is was 2x
# off in the only place a coding agent reads the mechanism before touching
# prefix accounting: a 200k prompt is ~50 engine steps, not ~25.
#
# The number is worth a real pin rather than a re-wording, because it lives in
# two places that no build step joins: prose in `app/prefix_miss.py` and a
# supervisor conf value. `architecture/context-window.md` carried the same 8,192
# and was corrected to 4,096 in the pass that filed this item, leaving the
# docstring as the last stale surface. The nodes below compare the prose to the
# conf on every run, so the next halving cannot leave the docstring behind.

#: The retired claim, verbatim, as the control for the bans below: this text
#: trips all three, so none is an empty pattern passing on an empty corpus.
RETIRED_CHUNK_CLAIM = (
    "the whole prompt is prefilled again from cold, one 8,192-token chunk per "
    "engine step, and every other request on the engine gets one token per "
    "step until it finishes.")

_ROOT = Path(__file__).resolve().parent.parent
CONF = _ROOT / "agent-services" / "supervisor" / "conf.d" / "agent-llm-primary.conf"
_SRC = _ROOT / "app" / "prefix_miss.py"


def _conf_env(text: str | None = None) -> dict[str, str]:
    """The supervisord `environment=` map, parsed as
    `tests/test_flash_next_launcher.py` parses it, so both files read the one
    declarative source of the engine's pins."""
    assert CONF.exists(), f"{CONF} is gone, so nothing here prices a chunk any more"
    text = CONF.read_text(encoding="utf-8") if text is None else text
    line = next((line for line in text.splitlines()
                 if line.startswith("environment=")), None)
    assert line, "the program's environment= line is gone from the conf"
    return dict(re.findall(r'(\w+)="([^"]*)"', line))


def _module_docstring() -> str:
    """The MODULE docstring only, whitespace-flattened.

    Through `ast`, not by grepping the file: this module's code and other
    docstrings mention token counts too, and the surface #1785 was filed against
    is the one a reader gets from `help(prefix_miss)`.
    """
    import ast
    doc = ast.get_docstring(ast.parse(_SRC.read_text(encoding="utf-8")))
    assert doc, "app/prefix_miss.py has no module docstring — nothing to price"
    return " ".join(doc.split())


def _stated_chunk(doc: str) -> int:
    """The magnitude the docstring prices a chunk at, or a failure saying so."""
    m = re.search(r"MAX_NUM_BATCHED_TOKENS\D*?(\d[\d,]{2,6})\s+tokens", doc)
    assert m, (
        "the docstring no longer states a token magnitude beside "
        "MAX_NUM_BATCHED_TOKENS, so the number this item was filed over has been "
        f"replaced by a bare key name: {doc[:200]!r}")
    return int(m.group(1).replace(",", ""))


def _chunk_drift(doc: str, conf_text: str) -> str | None:
    """None when the prose prices the chunk the conf sets; otherwise why not.

    A pure function of the two texts, so the node below can feed it a mutated
    conf and see the drift fire without touching the real file.
    """
    env = _conf_env(conf_text)
    assert "MAX_NUM_BATCHED_TOKENS" in env, (
        "the conf's environment= no longer sets MAX_NUM_BATCHED_TOKENS at all, so "
        "the engine would be on vLLM's default and the prose's key name is the "
        "stale part — re-read the conf before believing either number")
    stated, live = _stated_chunk(doc), int(env["MAX_NUM_BATCHED_TOKENS"])
    return None if stated == live else (
        f"the docstring prices a cold prefill chunk at {stated} tokens while the "
        f"conf sets MAX_NUM_BATCHED_TOKENS=\"{live}\" — the sentence is the "
        "defect #1785 was filed for, and 2x wrong is 2x wrong in either direction")


def test_a_cold_prefill_is_priced_at_the_chunk_budget_in_force():
    """Clause 1: the mechanism, at the width that is actually running.

    Graded in both directions, because a deletion would also silence the
    sentence: the retired figure is gone AND the conf's value is stated in the
    same passage as `MAX_NUM_BATCHED_TOKENS` AND the stall itself still reads as
    it did — every other request on the engine getting one token per step is the
    reason the module exists, and it must survive an edit that only changes a
    number. The figure is then compared to the conf rather than to a literal, so
    this node is the drift detector the acceptance asks for.
    """
    doc = _module_docstring()
    old = " ".join(RETIRED_CHUNK_CLAIM.split())

    # Controls first: the retired text has to trip the bans, and still carry the
    # stall clause that the rewrite must keep.
    assert "8,192-token chunk" in old, (
        "the retired sentence no longer contains the figure its own ban targets, "
        "so the check below would be a ban on nothing")
    assert "one token per step" in old, (
        "the retired text stopped carrying the stall clause, so requiring it of "
        "the new docstring would be free")

    for banned in ("8,192", "8192"):
        assert banned not in doc, (
            f"the docstring prices the live budget at {banned} tokens again — "
            "the budget since 2026-09-10 is the one in the conf")
    assert "MAX_NUM_BATCHED_TOKENS" in doc, (
        "the docstring no longer names the knob, so a reader has nothing to go "
        "update and the number has no owner")
    for stall in ("every other request on the engine", "one token per step"):
        assert stall in doc, (
            f"the docstring lost {stall!r}: the neighbour-starvation mechanism is "
            "the point of the module, and an edit that fixes a number by "
            "deleting the sentence has not fixed anything")

    drift = _chunk_drift(doc, CONF.read_text(encoding="utf-8"))
    assert drift is None, drift


def test_the_corrected_prose_names_an_owner_that_resolves():
    """Clause 2: where the knob lives, so the next halving has one place to edit.

    The clause offers a choice — the conf path or `architecture/vllm.md` §5.1 —
    so the test takes the disjunction as stated rather than demanding both, and
    then adds the half that makes a citation worth having: whatever is named has
    to resolve. A docstring that cited `conf.d/agent-llm.conf` (a plausible
    typo, and one no grep in this file would notice) would pass a plain `in`
    check and still send the next reader to a file that does not exist.
    """
    doc = _module_docstring()

    conf_rel = "agent-services/supervisor/conf.d/agent-llm-primary.conf"
    named_conf, named_doc = conf_rel in doc, "vllm.md" in doc
    assert named_conf or named_doc, (
        "the docstring states a chunk size with no owner to update from: "
        f"{doc[:220]!r} — the whole point of #1785's fix is that the next "
        "halving has one place to be written down")

    if named_conf:
        assert (_ROOT / conf_rel).exists(), (
            f"the docstring cites {conf_rel}, which is not on disk")
        assert "MAX_NUM_BATCHED_TOKENS" in _conf_env(), (
            "the cited conf no longer sets the key the prose is about")
    if named_doc:
        assert "§5.1" in doc, (
            "the docstring points at vllm.md without a section, and §5.1 is the "
            "one that carries the measurement")
        vllm = (_ROOT / "architecture" / "vllm.md").read_text(encoding="utf-8")
        head = [ln for ln in vllm.splitlines() if ln.startswith("#")]
        budget = str(int(_conf_env()["MAX_NUM_BATCHED_TOKENS"]))
        assert any("5.1" in ln and budget in ln for ln in head), (
            f"vllm.md has no §5.1 heading stating the budget the conf sets "
            f"({budget}), so the section the docstring cites is not the one "
            "that justifies the number — either the doc moved or the conf did")
    assert doc.index("MAX_NUM_BATCHED_TOKENS") < min(
        [i for i in (doc.find(conf_rel), doc.find("vllm.md")) if i >= 0]), (
        "the owner citation is no longer in the passage that states the chunk — "
        "a citation parked at the end of a page is not the pointer it claims to "
        "be (the `__doc__` is one flattened block, so order is all a reader has)")


def test_the_expected_chunk_size_is_read_from_the_conf_not_from_a_literal():
    """Clause 3: the pin has to move when the conf moves, or it pins nothing.

    Three halves. (a) This file contains no copy of the expected value, so the
    number can only come from the conf — that is checkable, not a claim about
    intent. (b) The pin passes against the conf as it stands. (c) The pin is
    re-run against a mutated conf, in-process, and goes red: the conf-only change
    that #1783-style prose drift actually looks like. The substitution itself is
    asserted to have landed, because a mutation that silently matched nothing
    would leave (c) green forever — a check with a zero denominator is not a
    check.
    """
    import ast
    live = int(_conf_env()["MAX_NUM_BATCHED_TOKENS"])
    doc = _module_docstring()

    # Scanned through the AST, not by grepping the file: the history block above
    # quotes 4096 as prose about commit 62bad14a and that is a citation, not a
    # copy. What must not exist is the value as a constant, because that is what
    # quietly becomes the expected number when the conf moves.
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    hard = [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and n.value in (live, f"{live:,}")]
    assert not hard, (
        f"this file hard-codes the expected chunk size at line(s) {hard}, so a "
        "conf change to anything else would leave the comparison reading the "
        "wrong side of itself — the value has to come from the conf")

    conf_text = CONF.read_text(encoding="utf-8")
    assert _chunk_drift(doc, conf_text) is None

    needle = f'MAX_NUM_BATCHED_TOKENS="{live}"'
    for mutant in (live * 2, live // 2):
        mutated = conf_text.replace(needle, f'MAX_NUM_BATCHED_TOKENS="{mutant}"')
        assert mutated != conf_text, (
            f"the mutation harness wrote nothing: {needle!r} is not in the conf, "
            "so the drift check below would be asserting on the unmutated file")
        reason = _chunk_drift(doc, mutated)
        assert reason, (
            f"a conf-only change to {mutant} left the pin green — the docstring "
            "and the conf are no longer being compared to each other")
