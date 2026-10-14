"""The `tool_choice: "none"`-with-tools probe (#2678), decided offline.

Nothing here reaches an engine: every client is a stub, and the verdict comes
from `decide()`, which is a pure function over the readings the live script
collects — the live-script-plus-offline-test split of
`eval/engine_output_probe.py`. What is being decided is one property: does the
engine still render the `tools` array into the prompt when `tool_choice` is
`"none"`? The finalizer (`app/harness/finalizer.py`) and P6(b)'s max-turns
wrap-up ship on it, and a build that started dropping `tools` there is silent —
the answer still arrives, the prefill just costs full price.

The stubs answer with the numbers measured on the live primary on 2026-10-14
(build `0.31.1rc1.dev260+ga98247ab4`): 7,982 prompt tokens at both `"auto"` and
`"none"` with a two-tool array, 7,639 for the tools-dropped control, so the
tool schema is worth 343 tokens of the rendered prompt.
"""
from __future__ import annotations

import json

import pytest

import eval.tool_choice_none_probe as tc

BUILD = "0.31.1rc1.dev260+ga98247ab4"
#: Measured sizes on the live build, and the gap between them.
AUTO_TOKENS, NONE_TOKENS, NO_TOOLS_TOKENS = 7982, 7982, 7639
SCHEMA_COST = AUTO_TOKENS - NO_TOOLS_TOKENS


class FakeEngine:
    """`/version` and `/v1/chat/completions` for one tool-rendering behaviour.

    `drop_tools_at_none` is the regression being watched for: this engine bills
    the `"none"` request as if the array had been removed. `fail_shape` refuses
    one graded request outright, and `no_usage` answers without a `usage` block.
    """

    base_url = "http://127.0.0.1:8096"

    def __init__(self, *, version: str | None = BUILD, drop_tools_at_none: bool = False,
                 fail_shape: str | None = None, tool_calls_at_none: int = 0,
                 no_usage: bool = False, cached: dict[str, int] | None = None):
        self.version = version
        self.drop_tools_at_none = drop_tools_at_none
        self.fail_shape = fail_shape
        self.tool_calls_at_none = tool_calls_at_none
        self.no_usage = no_usage
        self.cached = cached or {}
        self.bodies: list[dict] = []

    def get_text(self, path: str) -> str:
        assert path == "/version"
        if self.version is None:
            raise ConnectionError("404 not found")
        return json.dumps({"version": self.version})

    @staticmethod
    def shape_of(body: dict) -> str:
        if "tools" not in body:
            return tc.NO_TOOLS
        return body["tool_choice"]

    def post_json(self, path: str, body: dict) -> dict:
        assert path == "/v1/chat/completions"
        shape = self.shape_of(body)
        self.bodies.append(body)
        if shape == self.fail_shape:
            raise RuntimeError("400 Bad Request: tool_choice 'none' not accepted")
        tokens = {tc.AUTO: AUTO_TOKENS,
                  tc.NONE: NO_TOOLS_TOKENS if self.drop_tools_at_none else NONE_TOKENS,
                  tc.NO_TOOLS: NO_TOOLS_TOKENS}[shape]
        message: dict = {"role": "assistant", "content": "two sentences of prose"}
        if shape == tc.NONE and self.tool_calls_at_none:
            message["tool_calls"] = [{"id": f"c{i}", "type": "function"}
                                     for i in range(self.tool_calls_at_none)]
        resp: dict = {"choices": [{"message": message, "finish_reason": "length"}]}
        if not self.no_usage:
            resp["usage"] = {"prompt_tokens": tokens,
                             "prompt_tokens_details": {"cached_tokens": self.cached.get(shape, 3200)}}
        return resp


def readings(**engine_kwargs) -> dict:
    """Readings straight off a stub engine — the input `decide` sees live."""
    return tc.run_probe(FakeEngine(**engine_kwargs), warm_requests=1)


# ── clause 1: the request shapes, and a verdict from prompt_tokens ───────────

def test_the_three_requests_differ_only_in_the_tool_fields():
    """The whole probe hangs on this: one prefix, three tool shapes.

    If `none` were sent different messages, `none` vs `auto` would compare two
    different prompts and equality would mean nothing.
    """
    eng = FakeEngine()
    tc.run_probe(eng, warm_requests=1)
    by_shape = {eng.shape_of(b): b for b in eng.bodies}
    assert set(by_shape) == set(tc.SHAPES), "every graded shape must be sent"
    assert all(b["messages"] == by_shape[tc.AUTO]["messages"] for b in eng.bodies)
    assert all(b["temperature"] == 0.0 for b in eng.bodies)
    assert by_shape[tc.AUTO]["tools"] == by_shape[tc.NONE]["tools"]
    assert by_shape[tc.AUTO]["tool_choice"] == "auto"
    assert by_shape[tc.NONE]["tool_choice"] == "none"
    assert "tools" not in by_shape[tc.NO_TOOLS], (
        "the control is the 25.00 s failure shape: no array, and no tool_choice either")
    # The control diverges the prefix, so it must come last or it evicts what is
    # being measured (an alternating A/B caches both shapes — not production).
    shapes_in_order = [eng.shape_of(b) for b in eng.bodies]
    assert shapes_in_order[-1] == tc.NO_TOOLS
    assert shapes_in_order.index(tc.NONE) > shapes_in_order.index(tc.AUTO)


def test_live_build_sizes_decide_pass_as_a_pure_function():
    """`none` renders what `auto` renders, and the control is cheaper: PASS."""
    verdict = tc.decide(readings())
    assert verdict["passed"] is True, verdict["reasons"]
    assert verdict["reasons"] == []
    assert verdict["numbers"]["schema_cost_tokens"] == SCHEMA_COST
    json.dumps(verdict)  # the artifact is a plain dict, not a live object


def test_a_build_that_drops_tools_at_none_decides_fail_naming_the_sizes():
    """The regression, and the discriminator: `none` billed like the control."""
    verdict = tc.decide(readings(drop_tools_at_none=True))
    assert verdict["passed"] is False
    assert verdict["numbers"]["none_prompt_tokens"] == NO_TOOLS_TOKENS
    reason = " ".join(verdict["reasons"])
    assert f"none={NO_TOOLS_TOKENS}" in reason and f"auto={AUTO_TOKENS}" in reason
    assert "343 prompt_tokens" in reason, "the loss is stated as a number"


def test_an_equality_with_no_schema_in_the_prompt_cannot_pass():
    """The control's job. A template that never renders tools makes `none` ==
    `auto` trivially, and a PASS on that equality would be a green on nothing."""
    class NoToolsEver(FakeEngine):
        """Tools never reach the prompt, so all three shapes bill the same."""

        def post_json(self, path, body):
            body = {k: v for k, v in body.items() if k != "tools"}
            return super().post_json(path, body)

    verdict = tc.decide(tc.run_probe(NoToolsEver()))
    assert verdict["passed"] is False
    assert verdict["numbers"]["schema_cost_tokens"] == 0
    assert "witness nothing" in " ".join(verdict["reasons"])
    # Positive control: the same decide call passes when the schema is priced in,
    # so the failure above is the zero control and not something else.
    assert tc.decide(readings())["passed"] is True


# ── clause 1 (cont.): a refused request is a verdict, never an exception ─────

@pytest.mark.parametrize("shape", list(tc.SHAPES))
def test_a_refused_graded_request_fails_without_raising(shape):
    """Each graded shape is the one the wrap-up might need; none may crash."""
    data = readings(fail_shape=shape)
    assert data["rows"][shape]["ok"] is False
    verdict = tc.decide(data)
    assert verdict["passed"] is False
    assert f"refused the {shape!r} request" in " ".join(verdict["reasons"])
    assert verdict["numbers"][f"{shape}_prompt_tokens"] is None, (
        "a refused request must not be recorded as a measured size, even though "
        "the other two still are")


def test_an_answer_with_no_usage_is_a_fail_not_a_zero():
    verdict = tc.decide(readings(no_usage=True))
    assert verdict["passed"] is False
    assert "no usage.prompt_tokens" in " ".join(verdict["reasons"])
    assert verdict["numbers"]["auto_prompt_tokens"] is None, (
        "a missing field must not be recorded as a measured 0")


def test_a_none_answer_that_emits_a_tool_call_fails():
    """`none` with the tool parser attached would let the wrap-up answer with a
    call the loop then has to drop."""
    verdict = tc.decide(readings(tool_calls_at_none=2))
    assert verdict["passed"] is False
    assert "2 tool call" in " ".join(verdict["reasons"])


def test_an_engine_that_will_not_name_its_build_fails():
    """A measurement with no build beside it cannot be quoted, which is the
    reason this instrument exists at all."""
    verdict = tc.decide(readings(version=None))
    assert verdict["passed"] is False
    assert "did not name its build" in " ".join(verdict["reasons"])


def test_cached_tokens_never_moves_the_verdict():
    """The decision is prompt_tokens-only, on purpose: a cold prefix reads 0,
    which is indistinguishable from the dropped-tools failure. A `none` request
    that reuses nothing while rendering the same prompt is a PASS here."""
    cold = tc.decide(readings(cached={tc.AUTO: 3200, tc.NONE: 0, tc.NO_TOOLS: 0}))
    assert cold["passed"] is True, cold["reasons"]
    assert cold["numbers"]["none_cached_tokens"] == 0, "still reported, just not decided"


# ── clause 2: the artifact carries the build beside every reading ────────────

def test_the_artifact_records_the_build_beside_each_graded_reading():
    art = tc.build_artifact(readings(), tc.decide(readings()))
    for shape in (tc.AUTO, tc.NONE):  # the two readings the property is made of
        row = art["readings"][shape]
        assert row["engine_version"] == BUILD, shape
        assert row["prompt_tokens"] == AUTO_TOKENS, shape
        assert row["cached_tokens"] == 3200, shape
    assert art["readings"][tc.NO_TOOLS]["engine_version"] == BUILD
    assert art["verdict"] == "PASS"
    assert art["engine"]["version"] == BUILD
    assert art["numbers"]["schema_cost_tokens"] == SCHEMA_COST
    json.dumps(art)


def test_a_recorded_measurement_cannot_be_stripped_of_its_build():
    """Even the per-shape row — what someone pastes into a doc — names the
    build it came from."""
    art = tc.build_artifact(readings(), tc.decide(readings()))
    quoted = json.dumps(art["readings"][tc.NONE])
    assert BUILD in quoted and "prompt_tokens" in quoted


# ── the CLI: exit codes and the printed build ────────────────────────────────

def _factory(eng: FakeEngine):
    return lambda base_url: eng


def test_main_exits_zero_on_pass_and_prints_the_build(capsys):
    rc = tc.main([], client_factory=_factory(FakeEngine()))
    out = capsys.readouterr().out
    assert rc == 0
    assert BUILD in out, "a verdict must name the build it was decided on"
    assert f"tool-schema cost: {SCHEMA_COST} prompt_tokens" in out
    assert "PASS" in out


def test_main_exits_nonzero_on_fail_and_prints_the_build(capsys):
    rc = tc.main([], client_factory=_factory(FakeEngine(drop_tools_at_none=True)))
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out and BUILD in out
    assert f"tool-schema cost: {SCHEMA_COST} prompt_tokens" in out


def test_main_exits_nonzero_when_the_engine_refuses_the_shape(capsys):
    """No traceback, and no exit 2 that reads like a broken script: the wrap-up
    must not ship on an engine that will not take the request."""
    rc = tc.main([], client_factory=_factory(FakeEngine(fail_shape=tc.NONE)))
    out = capsys.readouterr().out
    assert rc == 1
    assert "REFUSED" in out and "FAIL" in out


def test_main_json_writes_the_artifact_where_asked(tmp_path, capsys):
    dest = tmp_path / "run" / "probe.json"
    rc = tc.main(["--json", "--json-out", str(dest)], client_factory=_factory(FakeEngine()))
    assert rc == 0
    art = json.loads(dest.read_text())
    assert art["readings"][tc.NONE]["engine_version"] == BUILD
    assert BUILD in capsys.readouterr().out
