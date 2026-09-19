"""The preserved-thinking A/B must reach a turn boundary, or it measures nothing.

Backlog #617. `preserve_thinking_iterations` is enforced only at turn entry
(`_cap_history_reasoning`, `app/harness/loop.py:1561`, called from
`app/harness/loop.py:236`); inside a turn the knob is a bare boolean
(`app/harness/loop.py:532`). The eval that prices the window therefore has to
run a session that CROSSES a turn boundary, and until #617 it did not: its
`_one_run` drove a single `run_query` whose history was one user message, so
both arms carried either all of the turn's reasoning or none of it and the
published A/B priced a mechanism `2677dea` had already removed.

These tests are the discriminator the item could not settle on token counts:
the same scripted multi-turn session, run at two positive `keep` values, must
put an older assistant message's reasoning on the wire in one arm and not in
the other. Two positive values is the whole point — `keep=0` against `keep=6`
differs by every reasoning block in the turn, so it never showed the window.

Two layers, because the clause is about the run whose numbers get quoted:
`_drive` covers one `_one_run` (the mechanism), and `_drive_cli` covers
`main()` — one `--trials` invocation, the arms in the order the script prints
them, compared at one request index. The falsifier node
(`test_the_arms_collapse_when_the_turn_entry_cap_is_disabled`) disables the
turn-entry cap and requires the `window=N` and `all` arms to become
indistinguishable, so the two positive-value tests can only pass because of the
mechanism the eval claims to price, not because of anything the fixture itself
happens to drop.

Everything runs against a scripted engine: no `agent-llm-primary`, no GPU, and
no operator scheduling decision. What is NOT pinned here is any token figure
— those need a live re-run (see `eval/measurements/`).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

# tests/<this> -> parents[1] = repo root. Same home as tests/test_eval_scorer.py,
# which covers eval/run_eval.py the way this file covers the eval script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.harness import tool_search_cache  # noqa: E402
from app.harness.options import RunOptions  # noqa: E402

_EVAL_PATH = ROOT / "eval" / "run_preserve_thinking_eval.py"


def _load_eval():
    spec = importlib.util.spec_from_file_location("pt_eval", _EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


pt = _load_eval()

# Turn 1's scripted shape: four reasoning-bearing assistant turns that each
# fire a tool call, then a closing turn with no tool call. Distinct reasoning
# strings so an assertion can name WHICH turn survived the window.
TURN1 = [
    ("T1", "step one", [{"id": "c1", "name": "Bash", "arguments": {}}]),
    ("T2", "step two", [{"id": "c2", "name": "Bash", "arguments": {}}]),
    ("T3", "step three", [{"id": "c3", "name": "Bash", "arguments": {}}]),
    ("T4", "step four", [{"id": "c4", "name": "Bash", "arguments": {}}]),
    ("", "FINAL ANSWER", []),
]
# Turn 2 answers from memory, so it costs one request and its FIRST request
# is the one the turn-entry window acted on.
TURN2 = [("T5", "CONDENSED ANSWER", [])]
SCRIPT = TURN1 + TURN2


class _ScriptedEngine:
    """Stands in for vLLM: emits reasoning deltas, then text, then tool calls.

    Snapshots every request with a per-message `dict(m)` of its OWN, taken
    before the loop can touch the buffer again — so what is asserted is what
    went on the wire, not what the buffer looks like at the end. This capture
    is independent of the probe inside `_one_run`: if the eval's own counting
    and this one disagree, a test below fails rather than agreeing by
    construction.

    One engine can serve every arm of a `main()` run because the fixture calls
    `reset()` before each `_one_run`: each arm then replays the identical
    scripted session from position 0 and its request count is directly
    comparable to the others'. Without that reset, an arm that made one request
    more or fewer than the script would leave the next arm reading a script
    that had already advanced — the arms would be measured against different
    sessions and the summary table would still print three tidy rows.

    `reset()` files each finished arm away, so `arms()` can hand back one
    request list per arm and a test can say `arms()[i][k]` — request *k* of arm
    *i* — which is the only way to assert that two arms were compared at the
    same request index rather than at whichever index each happened to reach.
    """

    def __init__(self, turns: list[tuple[str, str, list[dict[str, Any]]]]):
        self.turns = turns
        self.requests: list[list[dict]] = []
        self.finished: list[list[list[dict]]] = []
        self.calls = 0
        self.trials = 0

    def arms(self) -> list[list[list[dict]]]:
        """Per-arm request snapshots, in run order, index-aligned across arms.

        Refuses unless every arm replayed the whole script. `arms()[i][k]`
        meaning the same request in both arms is a claim about the fixture, not
        about the harness, and it is exactly the premise a cross-arm "same
        request index" comparison rests on — an arm that issued one request
        more or fewer would otherwise be compared against a different session
        while every figure still looked plausible.
        """
        out = list(self.finished)
        if self.requests:
            out.append(self.requests)
        counts = {len(a) for a in out}
        if counts != {len(self.turns)}:
            raise AssertionError(
                f"arms issued {sorted(counts)} requests but the script has "
                f"{len(self.turns)} turns, so the arms did not replay the same "
                "session and request index k means a different request in each"
            )
        return out

    def reset(self) -> None:
        """Rewind to the top of the script for the next arm.

        An arm that issued no request at all is not filed away, which would
        silently drop it from `arms()`; `arms()`'s count check then fails on
        the missing arm rather than comparing the surviving ones.
        """
        if self.requests:
            self.finished.append(self.requests)
            self.requests = []
        self.calls = 0
        self.trials += 1

    def __call__(self, **kwargs):
        idx = self.calls
        if idx >= len(self.turns):
            raise AssertionError(
                f"engine got request {idx + 1} but only {len(self.turns)} are "
                "scripted — the eval asked the engine for more turns than the "
                "fixture gave it, so the session this arm measured is not the "
                "session the other arms measured"
            )
        self.calls += 1
        self.requests.append([dict(m) for m in (kwargs.get("messages") or [])])
        return self._gen(*self.turns[idx])

    async def _gen(self, reasoning: str, text: str, tool_calls: list[dict]):
        if reasoning:
            yield {"choices": [{"delta": {"reasoning_content": reasoning}}]}
        if text:
            yield {"choices": [{"delta": {"content": text}}]}
        for i, tc in enumerate(tool_calls):
            yield {"choices": [{"delta": {"tool_calls": [{
                "index": i, "id": tc["id"], "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc.get("arguments") or {}),
                },
            }]}}]}
        yield {"choices": [{"delta": {},
                            "finish_reason": "tool_calls" if tool_calls else "stop"}]}
        yield {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class _FakePool:
    @property
    def discovered(self):
        return [("lloyd-mcp", [{
            "name": "Bash",
            "description": "shell",
            "inputSchema": {"type": "object", "properties": {}},
        }])]

    async def call_tool(self, name: str, args: dict, *, session_id: str = "", **_kw):
        return {"content": f"FAKE_RESULT[{name}]", "is_error": False}


@pytest.fixture(autouse=True)
def _reset_cache():
    asyncio.run(tool_search_cache.clear())
    yield
    asyncio.run(tool_search_cache.clear())


def _script_engine(monkeypatch, turns=SCRIPT, *,
                   per_trial: bool = False) -> _ScriptedEngine:
    """Replace the production seams `_one_run` reaches.

    The MCP pool (no aggregator on :8500), the engine stream (no
    `agent-llm-primary`, no 95 GiB n-gram table), the system-prompt and config
    readers (which otherwise read the live vault and `config.yaml`), and the
    tool-disallow list.

    `per_trial=True` also wraps `pt._one_run` so the script rewinds between
    arms — needed by any test that drives `main()`, which runs several arms
    through one engine.
    """
    engine = _ScriptedEngine(turns)

    async def _build_pool(_options):
        return pool

    pool = _FakePool()
    monkeypatch.setattr("app.harness.loop._build_pool", _build_pool)
    monkeypatch.setattr("app.harness.loop.stream_chat", engine)
    monkeypatch.setattr("prompt_builder.build_system_prompt", lambda *a, **k: "SYS")
    monkeypatch.setattr("app.mcp_discovery._get_disallowed_tools", lambda *a, **k: set())
    monkeypatch.setattr("app.mcp_discovery._get_harness_kwargs",
                        lambda *a, **k: {"tool_search_enabled": False})
    if per_trial:
        real_one_run = pt._one_run

        async def _one_run_rewound(**kwargs):
            engine.reset()
            return await real_one_run(**kwargs)

        monkeypatch.setattr(pt, "_one_run", _one_run_rewound)
    return engine


def _drive(monkeypatch, keep: int, *, turns=SCRIPT, capture=True) -> dict:
    """Run one eval trial (`_one_run`) against the scripted engine."""
    engine = _script_engine(monkeypatch, turns)
    row = asyncio.run(pt._one_run(keep=keep, max_turns=8,
                                  followup_max_turns=3, capture=capture))
    row["_engine_requests"] = engine.requests
    return row



def _first_measured_request(row: dict) -> list[dict]:
    """The message list turn 2 actually sent — the window's only output."""
    idx = next(i for i, r in enumerate(row["requests"]) if r["turn"] == 2)
    return row["_engine_requests"][idx]


def _msg_with_content(msgs: list[dict], content: str) -> dict:
    """The assistant message that answers with `content`.

    Keyed on the visible text rather than on position because the window acts
    on a message's `reasoning` field, not on where the message sits: an
    assertion that names WHICH turn survived (`step one` -> `T1`) says the
    window kept the recent turns and dropped the old one, where an index would
    only say some message changed.
    """
    for m in msgs:
        if m.get("role") == "assistant" and m.get("content") == content:
            return m
    raise AssertionError(f"no assistant message with content {content!r}: {msgs}")


def _reasoning_by_content(msgs: list[dict], content: str) -> Any:
    return _msg_with_content(msgs, content).get("reasoning")


def _carried(msgs: list[dict]) -> list[str]:
    """The reasoning strings actually on the wire, oldest first."""
    return [m["reasoning"] for m in msgs
            if m.get("role") == "assistant" and m.get("reasoning")]


# ── clause 1: the session reaches the turn-entry cap ────────────────────────

def test_one_trial_hands_the_loop_prior_turn_reasoning(monkeypatch):
    """The request turn 2 sends must contain an older turn's reasoning.

    This is the boundary the old single-`run_query` eval never reached: its
    history at turn entry was one user message, so there was nothing for
    `_cap_history_reasoning` to window in either arm.
    """
    row = _drive(monkeypatch, keep=2)

    turn2 = [r for r in row["requests"] if r["turn"] == 2]
    assert turn2, f"no turn-2 request recorded: {row['requests']}"
    assert row["iterations"] == len(SCRIPT), (
        "the session should have ended after the scripted turns, not on "
        "max_turns — a turn boundary that never happens silently turns the "
        "A/B back into the single-turn measurement #617 is about"
    )

    msgs = _first_measured_request(row)
    carried = [m for m in msgs if m.get("role") == "assistant" and m.get("reasoning")]
    assert carried, (
        "turn 2's first request carried no assistant reasoning at all, so "
        "the turn-entry window had nothing to act on and both arms would "
        "report identical figures"
    )
    # keep=2 keeps the two most recent reasoning-bearing assistant turns of
    # turn 1 (T3, T4) out of four.
    assert [m["reasoning"] for m in carried] == ["T3", "T4"], carried


def test_windowed_arm_actually_narrows_the_carried_reasoning(monkeypatch):
    """Same request index, same session: `all` carries T1, `window=2` drops it.

    The clause the token-count version of this check could never settle:
    two POSITIVE keep values are what isolates the window, because the
    shipped mechanism makes `keep=0` differ by every reasoning block in the
    session rather than by the window's width.
    """
    narrow = _drive(monkeypatch, keep=2)
    wide = _drive(monkeypatch, keep=999)

    # Same script, so the measured request must be the same index in both —
    # otherwise "same request index" is not the comparison being made.
    n1 = sum(1 for r in narrow["requests"] if r["turn"] == 1)
    n2 = sum(1 for r in wide["requests"] if r["turn"] == 1)
    assert n1 == n2 == len(TURN1)

    assert _reasoning_by_content(_first_measured_request(wide), "step one") == "T1", (
        "the `all` arm is supposed to carry every prior turn's reasoning; if "
        "it does not, the window is binding somewhere the item does not "
        "believe it binds and the A/B is comparing two unknowns"
    )
    assert "reasoning" not in next(
        m for m in _first_measured_request(narrow)
        if m.get("role") == "assistant" and m.get("content") == "step one"
    ), "keep=2 must drop the oldest of turn 1's four reasoning blocks"

    # And the drop is what the row claims it is, on both sides of the ledger.
    assert narrow["carried_reasoning_blocks"] == 2
    assert wide["carried_reasoning_blocks"] == 4
    assert narrow["carried_reasoning_chars"] < wide["carried_reasoning_chars"]
    assert wide["window_removed_chars"] == 0
    assert narrow["window_removed_chars"] == len("T1") + len("T2")


def test_off_arm_is_the_no_reasoning_control(monkeypatch):
    """keep=0 attaches nothing anywhere, which is why it is the control.

    `off` is a legitimate arm — it is the pre-2026-09-05 behaviour — but it
    is NOT a window arm: under it nothing is carried at all, so the gap it
    produces against a positive arm prices preserved thinking, not the
    window's width.
    """
    row = _drive(monkeypatch, keep=0)

    assert row["prior_reasoning_chars"] == 0
    assert row["carried_reasoning_blocks"] == 0
    assert row["carried_reasoning_chars"] == 0
    for msgs in row["_engine_requests"]:
        for m in msgs:
            if m.get("role") == "assistant":
                assert "reasoning" not in m and "reasoning_content" not in m, m


def test_reported_reasoning_figures_match_the_wire(monkeypatch):
    """`prior_*` / `carried_*` are computed from what was really sent.

    The probe lives inside the code under test, so it is checked against the
    engine's own independent snapshot: turn 1 left four blocks behind, turn 2
    went out carrying the two the window let through.
    """
    row = _drive(monkeypatch, keep=2)

    assert row["prior_reasoning_chars"] == sum(len(t[0]) for t in TURN1)
    assert row["carried_reasoning_chars"] == len("T3") + len("T4")
    assert (row["window_removed_chars"]
            == row["prior_reasoning_chars"] - row["carried_reasoning_chars"])
    # The eval's own record and the engine's agree, request for request.
    for recorded, sent in zip(row["requests"], row["_engine_requests"]):
        assert recorded["n_messages"] == len(sent)
        assert recorded["reasoning_blocks"] == len(
            [m for m in sent
             if m.get("role") == "assistant" and m.get("reasoning")])


def test_run_turn_refuses_a_buffer_that_never_crossed_a_boundary(monkeypatch):
    """The guard that stops the script reporting a fake A/B.

    If `chat_messages_handle` ever stops being shared, each turn starts from
    an empty buffer and both arms produce identical numbers with nothing to
    say so. That is the failure that let #617's premise stand for a week, so
    it raises instead of measuring.
    """
    async def _fake_run_query(_msgs, options):
        if False:  # pragma: no cover — generator that yields nothing
            yield {}

    opts = RunOptions(model="primary", session_id="pt-eval-guard",
                      chat_messages_handle=[{"role": "user", "content": "other"}])
    with pytest.raises(RuntimeError, match="does not end on this turn"):
        asyncio.run(pt._run_turn(_fake_run_query, opts,
                                 expect_user_message="this turn"))

    # Buffer shared, but the loop appended nothing: no history for the next
    # turn to window, so that raises too rather than reporting an A/B.
    empty: list[dict] = [{"role": "user", "content": "this turn"}]
    opts2 = RunOptions(model="primary", session_id="pt-eval-guard",
                       chat_messages_handle=empty)
    with pytest.raises(RuntimeError, match="appended nothing"):
        asyncio.run(pt._run_turn(_fake_run_query, opts2,
                                 expect_user_message="this turn"))


# ── clauses 1 and 2 on the path an operator runs: one `--trials` invocation ──

def _drive_cli(monkeypatch, capsys, *, keep: int = 2,
               carry_all: int = 999) -> tuple[list[list[list[dict]]], list[str]]:
    """Run the CLI (`main`) over the scripted engine; return (per-arm requests, labels).

    `arms[i][k]` is request *k* of arm *i*, and `labels[i]` is that arm's
    printed label, so a test can compare two arms at one request index — which
    is the comparison clauses 1 and 2 are written as, and which `_drive` (a
    single `_one_run`) cannot make.

    The labels are read back out of the script's own `arms:` banner rather than
    restated here: the banner is what an operator reads when choosing which
    column is which, so if it ever names the arms in a different order the test
    follows it instead of quietly asserting against the wrong column.
    """
    engine = _script_engine(monkeypatch, per_trial=True)
    monkeypatch.setattr(sys, "argv", [
        "run_preserve_thinking_eval.py", "--trials", "1",
        "--keep", str(keep), "--carry-all", str(carry_all)])

    assert asyncio.run(pt.main()) == 0, "the CLI exited non-zero"

    printed = capsys.readouterr().out
    banner = next((line for line in printed.splitlines()
                   if line.startswith("arms: ")), None)
    assert banner, f"the run never printed its arm banner: {printed[:400]}"
    labels = [seg.split(" = ")[0].strip()
              for seg in banner[len("arms: "):].split(" | ")]
    arms = engine.arms()
    assert len(arms) == len(labels), (
        f"{len(arms)} arm(s) reached the engine but the banner named "
        f"{len(labels)} ({labels}): an arm that never ran still gets a column "
        "in the summary table"
    )
    return arms, labels


def test_a_trials_run_puts_prior_turn_reasoning_on_the_engine_s_wire(monkeypatch,
                                                                     capsys):
    """Clause 1, through `--trials` — the invocation whose numbers get quoted.

    `_drive` covers the mechanism; this covers the script an operator actually
    runs. It asserts on the engine's own snapshot of the request, not on the
    `prior_*`/`carried_*` fields, because the probe that computes those lives
    inside the code under test: agreement between two readings of the same
    buffer is the only thing that makes the published figure a measurement.

    The `off` arm is asserted EMPTY for the same reason. Without it, "at least
    one assistant message carries reasoning" would be satisfied by any session
    that ever sent reasoning anywhere, which is not the boundary clause 1 is
    about.
    """
    arms, labels = _drive_cli(monkeypatch, capsys)
    idx = len(TURN1)  # turn 1 issues one request per phase, so this is turn 2's first

    for label in ("window=2", "all"):
        msgs = arms[labels.index(label)][idx]
        assert msgs[-1] == {"role": "user", "content": pt.FOLLOWUP}, (
            f"arm {label}: request {idx} is not turn 2's opening request, so "
            "the index this file calls 'the measured one' is not the request "
            "the turn-entry window acted on"
        )
        assert _carried(msgs), (
            f"arm {label}: turn 2's first request carried no prior-turn "
            "reasoning, so the window had nothing to window and every arm of "
            "this run would report the same figure — the #617 defect, back"
        )

    off_msgs = arms[labels.index("off")][idx]
    assert _carried(off_msgs) == [], (
        "the keep=0 arm carried reasoning into turn 2, so `off` is no longer "
        f"the control arm and the table's baseline column is not zero: {_carried(off_msgs)}"
    )


def test_the_two_windowed_cli_arms_differ_at_the_same_request_index(monkeypatch,
                                                                    capsys):
    """Clause 2, through `--trials`: one session, one index, two verdicts on T1.

    `window=2` and `all` replay the identical scripted session and are compared
    at the identical request index (pinned by `_drive_cli` -> `engine.arms()`,
    which refuses mis-aligned arms). The oldest of turn 1's four reasoning
    blocks is on the wire for `all` and not on it for `window=2`: that
    difference is the turn-entry window, and it is the only gap in this eval
    that prices the window's *width* rather than preserved thinking itself.
    """
    arms, labels = _drive_cli(monkeypatch, capsys)
    idx = len(TURN1)
    narrow = arms[labels.index("window=2")][idx]
    wide = arms[labels.index("all")][idx]

    assert _reasoning_by_content(wide, "step one") == "T1", (
        "the `all` arm did not carry the oldest turn-1 reasoning, so the "
        "window is binding somewhere the item does not believe it binds and "
        "the two arms are measuring two unknowns"
    )
    assert "reasoning" not in _msg_with_content(narrow, "step one"), (
        "keep=2 left the oldest of four reasoning blocks on the wire, so the "
        "window is not being applied at this turn boundary and the eval's "
        "`window=N` column is the `all` column under another label"
    )
    # Named, not merely counted: which blocks survived is the claim.
    assert _carried(narrow) == ["T3", "T4"], _carried(narrow)
    assert _carried(wide) == ["T1", "T2", "T3", "T4"], _carried(wide)


def test_the_arms_collapse_when_the_turn_entry_cap_is_disabled(monkeypatch,
                                                               capsys):
    """The falsifier for clauses 1 and 2, kept in the tree instead of in a note.

    `_cap_history_reasoning` (`app/harness/loop.py:1561`, called at turn entry
    from `app/harness/loop.py:236`) is the only place the shipped mechanism
    enforces the window. Neutralise it and `window=2` must become
    indistinguishable from `all` — same blocks, same request index — because
    inside a turn the knob is a bare boolean (`app/harness/loop.py:532`) and
    there is no second enforcement point to blame for the difference.

    Without this, the two tests above only show the arms differ; they do not
    show the cap is what makes them differ. That is the exact shape of the
    #617 defect: a script whose two columns were assumed to measure a window
    that no longer existed anywhere in the code, published as if it did.
    """
    import app.harness.loop as loop_mod

    def _no_cap(_chat_messages, *, keep):
        return None

    monkeypatch.setattr(loop_mod, "_cap_history_reasoning", _no_cap)
    arms, labels = _drive_cli(monkeypatch, capsys)
    idx = len(TURN1)

    narrow = _carried(arms[labels.index("window=2")][idx])
    wide = _carried(arms[labels.index("all")][idx])
    assert narrow == wide == ["T1", "T2", "T3", "T4"], (
        f"with the turn-entry cap disabled the arms did not both carry every "
        f"prior-turn block (window=2: {narrow}, all: {wide}) — reasoning is "
        "being removed somewhere other than the cap, so this A/B is not "
        "pricing the turn-entry window"
    )


# ── clause 5: the shipped mechanism is the one being measured ───────────────

def test_intra_turn_behaviour_is_unchanged_through_the_eval(monkeypatch):
    """The eval must not have re-introduced the trim it is no longer measuring.

    `app/harness/tests/test_preserve_thinking.py::
    test_window_is_bounded_at_the_turn_boundary` asserts `["T1", "T2", "T3"]` —
    all three of one turn's reasoning phases stay on the wire with `keep=2` —
    and #617 forbids editing that file. This pins the same fact from the EVAL's
    own code path, so a change that quietly moved the window back inside the
    loop fails here as well as there: at turn 1's fourth request, with `keep=2`,
    all three phases are still being sent. The window the eval measures is then
    provably the turn-entry one and the only place reasoning tokens are removed
    is at the boundary.
    """
    row = _drive(monkeypatch, keep=2)

    turn1_requests = [i for i, r in enumerate(row["requests"]) if r["turn"] == 1]
    assert len(turn1_requests) == len(TURN1)
    fourth = row["_engine_requests"][turn1_requests[3]]
    phases = [m["reasoning"] for m in fourth
              if m.get("role") == "assistant" and m.get("reasoning")]
    assert phases == ["T1", "T2", "T3"], (
        f"intra-turn reasoning was windowed, which the shipped loop does not do: {phases}"
    )


def test_the_eval_drives_the_shipped_loop_and_the_mechanism_test_is_untouched():
    """Two halves of clause 5, from inside the diff.

    The eval has to go through `app.harness.run_query` — a private copy of the
    loop would measure a fiction — and the production test whose
    `["T1", "T2", "T3"]` assertion the item pins as intended behaviour must
    still carry it, so a later edit that moves the window cannot keep this
    eval's claims true by editing the evidence.
    """
    src = _EVAL_PATH.read_text()
    assert "from app.harness.loop import run_query" in src, (
        "the eval must call the real loop, not a local reimplementation"
    )
    prod = (ROOT / "app/harness/tests/test_preserve_thinking.py").read_text()
    assert 'assert kept == ["T1", "T2", "T3"], kept' in prod, (
        "the intra-turn assertion #617 depends on has moved; re-read the "
        "clause before trusting this file's numbers"
    )


# ── clauses 3 and 4: the script says what it measures ───────────────────────

def test_header_dates_every_published_number():
    """The header must date the mechanism, not just describe the knob.

    Every figure the eval published before `2677dea` came from the
    per-iteration window that commit removed. Without that sentence in the
    file the old numbers stay quotable, which is how #520's risk section
    ended up priced by an eval that no longer measured it.
    """
    src = _EVAL_PATH.read_text()
    assert "2677dea" in src, "the header must name the commit that moved the cap"
    assert "#520" in src, "and the backlog item that made it a turn-entry cap"
    assert "per-iteration mechanism" in src, (
        "the header has to state in words that pre-2677dea numbers describe "
        "the mechanism this script no longer measures"
    )
    # One exact sentence, not a disjunction: an `or` here is satisfiable by
    # any file that contains either fragment anywhere.
    assert ("must not be quoted as a measurement of shipped behaviour" in src), (
        "the header must say, in those words, what a reader may not do with a "
        "pre-2677dea figure"
    )


def _help_for(*flags: str) -> str:
    """The help string of ONE action, whitespace-normalised across its flags.

    Two reasons this is not `"".join(a.help for a in parser._actions)`:
    searching the blob lets unrelated actions satisfy a fragment, and argparse
    re-wraps help to the terminal width, which would split any multi-word
    phrase this file depends on. Whitespace is therefore collapsed back to
    single spaces, so a check here says what the source says regardless of
    `COLUMNS`.
    """
    parser = pt.build_argparser()
    by_flag = {
        flag: action.help or ""
        for action in parser._actions  # noqa: SLF001
        for flag in action.option_strings
    }
    return " ".join(" ".join(by_flag[f] for f in flags).split())


def test_arm_labels_and_keep_help_name_the_turn_entry_window():
    """Labels that say 'carry at most N iterations' describe a removed knob."""
    assert pt._arm_label(0, 999) == "off"
    assert pt._arm_label(6, 999) == "window=6"
    assert pt._arm_label(999, 999) == "all"

    keep_help = _help_for("--keep")
    assert "TURN-ENTRY" in keep_help, keep_help
    assert "NEXT TURN" in keep_help, (
        f"--keep must say the value bounds what crosses a turn boundary: {keep_help}"
    )
    assert "carry at most N iterations of reasoning" in keep_help, (
        "the --keep help must name the pre-#520 meaning, because that is the "
        "sentence every old run record uses — and the wording below is what "
        "makes it a disclaimer rather than a definition"
    )
    assert "NOT 'carry at most N iterations of reasoning'" in keep_help, keep_help

    assert "the arm the --keep window has to be compared against" in _help_for(
        "--carry-all"), "the window arm is meaningless without its comparator"


def test_arm_labels_are_distinct_so_the_table_cannot_collapse():
    """`--keep 999` would otherwise print two columns called `all`.

    Two arms under one label merge into one row of the summary table and the
    comparison silently disappears rather than failing.
    """
    assert pt._arm_label(6, 6) == "all"
    labels = [pt._arm_label(k, 999) for k in (0, 6, 999)]
    assert len(set(labels)) == 3, labels

    keeps = [0, 6, 999]
    labelled = {pt._arm_label(k, 999) for k in keeps}
    assert len(labelled) == len(keeps), (
        "arms keyed by label would overwrite each other's rows"
    )


def test_cli_runs_every_arm_and_records_the_mechanism(monkeypatch, tmp_path, capsys):
    """End to end across the CLI seam: three arms, labelled, with the date-stamp.

    The `--out` payload is what gets quoted later, so it has to carry the
    `mechanism` field naming the commit the numbers are from — otherwise a
    post-#617 run and a pre-`2677dea` run are indistinguishable in the file,
    which is how the stale figures stayed quotable in the first place.
    """
    engine = _script_engine(monkeypatch, per_trial=True)
    out_file = tmp_path / "pt.json"
    monkeypatch.setattr(sys, "argv", [
        "run_preserve_thinking_eval.py", "--trials", "1",
        "--keep", "2", "--carry-all", "999", "--out", str(out_file)])

    assert asyncio.run(pt.main()) == 0

    # Three arms, each replaying the whole scripted session from position 0.
    # `trials` is the reset count and `calls` is the last arm's request count:
    # together they say every arm saw the same script, which is the only
    # reason their figures are comparable.
    assert engine.trials == 3, engine.trials
    assert engine.calls == len(SCRIPT), engine.calls

    printed = capsys.readouterr().out
    for label in ("off", "window=2", "all"):
        assert label in printed, f"arm {label} never printed: {printed}"
    assert "turn 1" in printed and "turn 2" in printed, printed

    payload = json.loads(out_file.read_text())
    assert set(payload["summary"]) == {"off", "window=2", "all"}
    assert "2677dea" in payload["mechanism"]
    assert payload["summary"]["window=2"]["carried_reasoning_blocks"] == 2
    assert payload["summary"]["all"]["carried_reasoning_blocks"] == 4
    assert payload["summary"]["off"]["carried_reasoning_blocks"] == 0
    assert payload["arms"]["all"][0]["requests"] == "<omitted>", (
        "full request bodies would make the out file a copy of the session"
    )


def test_delta_against_a_zero_denominator_is_not_a_crash():
    """An arm that never answered has `answer_chars` 0 and `completed` 0.

    The percentage column divides by the baseline arm's figure, so the table
    itself must not be the thing that dies when an arm comes back empty —
    that is the run whose numbers you most need printed.
    """
    assert pt._delta_cell(5, 0) == "n/a"
    assert pt._delta_cell(12, 10) == "+20.0%"

