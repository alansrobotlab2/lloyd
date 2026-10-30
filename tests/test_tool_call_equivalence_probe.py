"""#3188 clause 1 and clause 4: the canary fold's tool-call leg, offline.

`eval/action_equivalence.py` landed 2026-10-16 (`e7e402d5`) to grade which tool an engine
chose and what raw arguments it passed, and nothing outside its own test called it: the fold
both arm routes run drives `eval/engine_output_probe.py`, whose bodies post
`{model, prompt, temperature: 0, logprobs: 1}` to `/v1/completions` — a route with no `tools`
field at all. So every arm verdict in the canary log is prose only, including the bf16 arm's
`past-floor` row at 2026-10-12T23:16:34+00:00, four days before the comparator existed.

Everything here runs against a stub engine. The live half — an arm booted on each compared
dtype through the gated restart, and the first real cross-arm line — is #3188's owed check and
cannot be produced by a round; what a round can prove is that the instrument records the
decision, keeps the prose canary's inputs byte-identical, and separates "the engine decided
differently" from "the instrument could not look".
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval.engine_output_probe as probe                      # noqa: E402
import eval.tool_call_equivalence_probe as tcp                # noqa: E402

CORPUS = ROOT / "eval" / "tool_call_corpus.yaml"

#: The four files clause 1 keeps byte-unchanged, with the sha256 of the tree at this round's
#: base. They are the prose canary's whole measuring stick: a tool-call leg that quietly
#: edited the corpus or the floor would "improve" the canary by moving it.
UNCHANGED = {
    "eval/engine_output/corpus.yaml":
        "b880e4716d17316e7a6e21c138aea93ba32e5e9ec891faaa96814c987ad1063f",
    "eval/engine_output/floor.json":
        "f0c5a718e0f4e433a80eb8ab7758e941214d57bd3d5d9dc267bed8a0bd8fc02e",
    "eval/engine_output/floor-0924-dff1bde.json":
        "b0df625cfe01112045c1061f364296261bb55ff12fd24556c23102767bd76ba9",
    "eval/engine_output/idle/20261012T041722Z_idle1012-5.json":
        "a8e9aab16186567720d8e12fb81313b002ff10c5710e73ecc9192c851fe46e57",
}

#: What each stub arm answers, per corpus prompt id: (tool name, RAW argument STRING).
#: Raw on purpose — the capture must hand `classify` the same unexplained string the tool
#: boundary would pass (`action_equivalence.py:10-13`: "repairing a malformed call here would
#: hide the finding").
ARM_FP8 = {
    "tc_read_named_path": [("Read", '{"file_path": "/home/alansrobotlab/lloyd/scripts/automod/state.py"}')],
    "tc_bash_test_command": [("Bash", '{"command": "python -m pytest -q tests/test_automod_vault_round.py", "cwd": "/home/alansrobotlab/lloyd"}')],
    "tc_structured_key": [("fact_get", '{"entity": "2846"}')],
    "tc_no_call_expected": [],
}
#: The other arm, identical decisions except the path one read names, and the shell's cwd.
ARM_BF16 = {
    "tc_read_named_path": [("Read", '{"file_path": "/home/alansrobotlab/lloyd/scripts/automod/worktree.py"}')],
    "tc_bash_test_command": [("Bash", '{"command": "python -m pytest -q tests/test_automod_vault_round.py", "cwd": "/home/alansrobotlab/lloyd-work/SM_20261023_095700/home/lloyd"}')],
    "tc_structured_key": [("fact_get", '{"entity": "2846"}')],
    "tc_no_call_expected": [],
}

PROSE_TOKENS = ["The", " round", " state", " module", "."]
PROSE_LPS = [-0.5, -0.4, -0.3, -0.4, -0.2]


class StubEngine:
    """One stub answering BOTH probes, so clause 4's claim that the same two records read
    `args_differ` on the decision while reading within floor on prose is measured through one
    engine rather than asserted across two fixtures.

    Same surface as `EngineClient`: `post_json(path, body)` and `get_text(path)`. Every
    `/v1/completions` answer is byte-identical call to call — identical prose and identical
    logprobs — which is the strongest form of "within floor": agreement 1.0 against a floor
    whose `agreement_min` is 1.0, and agreement 1.0 is exactly what a prose-only canary sees
    while the two engines disagree on which file to read.
    """

    def __init__(self, replies: dict[str, list[tuple[str, str]]], *, prose: bool = True):
        self.replies = replies
        self.prose = prose
        self.chat_bodies: list[dict] = []
        self.completion_bodies: list[dict] = []

    def post_json(self, path: str, body: dict) -> dict:
        if path == "/v1/chat/completions":
            self.chat_bodies.append(body)
            assert body["temperature"] == 0, "clause 1 asks for temperature 0"
            assert isinstance(body.get("tools"), list) and body["tools"], "tools must be offered"
            assert "prompt" not in body, "a chat request carries `messages`, not `prompt`"
            prompt_text = body["messages"][0]["content"]
            pid = self._pid_for(prompt_text)
            calls = self.replies.get(pid, [])
            return {
                "id": "cmpl-stub", "model": "stub-engine-1",
                "choices": [{
                    "index": 0,
                    "finish_reason": "tool_calls" if calls else "stop",
                    "message": {
                        "role": "assistant",
                        "content": "" if calls else "idle",
                        "tool_calls": [
                            {"id": f"call_{i}", "type": "function",
                             "function": {"name": name, "arguments": raw}}
                            for i, (name, raw) in enumerate(calls)],
                    },
                }],
            }
        if path == "/v1/completions":
            if not self.prose:
                raise tcp.ProbeRefused("stub: prose leg switched off")
            self.completion_bodies.append(body)
            assert body["temperature"] == 0
            assert "tools" not in body, "the prose probe stays a /v1/completions client"
            return {
                "id": "cmpl-stub", "model": "stub-engine-1",
                "choices": [{
                    "index": 0, "finish_reason": "length",
                    "text": " ".join(PROSE_TOKENS),
                    "logprobs": {"tokens": list(PROSE_TOKENS),
                                 "token_logprobs": list(PROSE_LPS),
                                 "top_logprobs": [{"The": -0.5}] * len(PROSE_TOKENS),
                                 "text_offset": list(range(len(PROSE_TOKENS)))},
                }],
                "usage": {"prompt_tokens": 21, "completion_tokens": len(PROSE_TOKENS),
                          "prompt_tokens_details": {"cached_tokens": 0}},
            }
        raise AssertionError(f"stub engine got an unexpected path {path}")

    def get_text(self, path: str) -> str:
        if path == "/v1/models":
            return json.dumps({"object": "list", "data": [{"id": "stub-engine-1"}]})
        return ""

    def _pid_for(self, prompt_text: str) -> str:
        for spec in tcp.load_corpus(CORPUS)["prompts"]:
            if spec["text"] == prompt_text:
                return spec["id"]
        raise AssertionError(f"stub was asked something the corpus does not contain: {prompt_text!r}")


@pytest.fixture
def services(tmp_path, monkeypatch):
    """Point `$LLOYD_DATA` at a tmp dir: every node here writes its own captures, and the
    real `~/lloyd-data/logs/services` is read by nobody in this file."""
    root = tmp_path / "lloyd-data"
    (root / "logs" / "services").mkdir(parents=True)
    monkeypatch.setenv("LLOYD_DATA", str(root))
    monkeypatch.setenv("CANARY_WINDOW", "W1")
    return root / "logs" / "services"


def _captured(services: Path, arm: str) -> list[dict]:
    path = services / f"flash-next-canary-toolcalls-{arm}.jsonl"
    return tcp.read_rows(path)


def _run_capture(services, arm, replies):
    stub = StubEngine(replies)
    rc = tcp.main(["capture", "--label", arm], client_factory=lambda url: stub)
    return rc, stub


# ── clause 1: the capture records the decision, keyed by arm, holding no prompt text ──
def test_an_engine_that_called_when_none_was_asked_exits_1(services):
    """The other direction of the same instrument: the corpus's fourth prompt asks for the
    word `idle`, so a tool call there is the engine deciding wrong."""
    over = dict(ARM_FP8)
    over["tc_no_call_expected"] = [("Bash", '{"command": "echo idle"}')]
    rc, _ = _run_capture(services, "fp8-kv", over)
    assert rc == tcp.EXIT_FINDING == 1, rc
    rows = {r["prompt"]: r for r in _captured(services, "fp8-kv")}
    assert rows["tc_no_call_expected"]["flags"] == ["call_when_none_asked"], \
        rows["tc_no_call_expected"]
    assert rows["tc_no_call_expected"]["expect_tool_call"] is False, rows["tc_no_call_expected"]


def test_a_capture_records_tool_name_raw_arguments_and_arm_label(services):
    rc, stub = _run_capture(services, "fp8-kv", ARM_FP8)
    assert rc == tcp.EXIT_OK, "every prompt was answered with a usable call"
    path = services / "flash-next-canary-toolcalls-fp8-kv.jsonl"
    assert path.exists(), "clause 1 names the capture path explicitly"
    rows = _captured(services, "fp8-kv")
    assert len(rows) == 4 == stub.chat_bodies.__len__(), "one row per corpus prompt"
    assert {r["prompt"] for r in rows} == set(ARM_FP8)
    by_id = {r["prompt"]: r for r in rows}

    read = by_id["tc_read_named_path"]
    assert read["arm"] == "fp8-kv" and read["window"] == "W1", read
    assert read["kind"] == "tool_call" and read["schema"] == 1, read
    assert read["temperature"] == 0, read
    assert read["tool_calls"] == [{
        "name": "Read",
        "arguments": '{"file_path": "/home/alansrobotlab/lloyd/scripts/automod/state.py"}',
    }], "the raw argument STRING survives byte-for-byte, unrepairs"
    assert isinstance(read["tool_calls"][0]["arguments"], str), read
    assert read["flags"] == [] and read["n_tool_calls"] == 1, read
    assert read["expect_tool_call"] is True, read
    assert len(read["corpus_sha256"]) == 64 and len(read["prompt_sha256"]) == 64, read

    bash = by_id["tc_bash_test_command"]
    assert bash["tool_calls"][0]["name"] == "Bash", bash
    assert json.loads(bash["tool_calls"][0]["arguments"])["cwd"] == "/home/alansrobotlab/lloyd", bash

    # The corpus is asked with ITS OWN tools array each time, never one shared array.
    names_per_call = [sorted(t["function"]["name"] for t in body["tools"])
                      for body in stub.chat_bodies]
    assert names_per_call == [["Grep", "Read"], ["Bash"], ["fact_get", "vault_search"], ["Bash"]], \
        names_per_call
    assert [body["messages"][0]["role"] for body in stub.chat_bodies] == ["user"] * 4


def test_the_capture_carries_no_prompt_text_and_no_request_body(services):
    rc, _ = _run_capture(services, "fp8-kv", ARM_FP8)
    assert rc == tcp.EXIT_OK
    blob = (services / "flash-next-canary-toolcalls-fp8-kv.jsonl").read_text()
    for spec in tcp.load_corpus(CORPUS)["prompts"]:
        assert spec["text"] not in blob, f"the capture stored the prompt text of {spec['id']}"
    for row in _captured(services, "fp8-kv"):
        assert "text" not in row and "messages" not in row and "content" not in row, row
        assert "content_chars" in row, "prose is recorded as a count, never as prose"
    # `app/component_manifest.py:40` is the rule this respects: "No blob store and no
    # `rebuild_request`" — the row cannot rebuild the request it answers.
    assert "rebuild_request" not in blob, blob


def test_the_prose_canarys_corpus_floors_and_reference_are_unchanged():
    """Clause 1's other half: the tool leg must not move the measuring stick it joins."""
    for rel, sha in UNCHANGED.items():
        path = ROOT / rel
        assert path.exists(), rel
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, rel
    corpus = probe.load_corpus(ROOT / "eval" / "engine_output" / "corpus.yaml")
    assert len(corpus["prompts"]) == 21, "the prose corpus is still the 21-prompt one"
    prompts = json.loads((ROOT / "eval" / "engine_output" / "floor.json").read_text())["prompts"]
    assert len(prompts) == 21, "and still floored prompt for prompt"


def test_an_engine_that_answered_in_prose_when_asked_for_a_call_exits_1_and_still_records(services):
    """Exit 1 is a finding ABOUT THE ENGINE, and the row is the evidence for it."""
    declined = dict(ARM_FP8)
    declined["tc_read_named_path"] = []
    rc, _ = _run_capture(services, "fp8-kv", declined)
    assert rc == tcp.EXIT_FINDING == 1, rc
    rows = {r["prompt"]: r for r in _captured(services, "fp8-kv")}
    assert rows["tc_read_named_path"]["flags"] == ["no_tool_call", "no_tool_call_when_asked"], \
        rows["tc_read_named_path"]
    assert rows["tc_read_named_path"]["n_tool_calls"] == 0, rows["tc_read_named_path"]
    assert rows["tc_no_call_expected"]["flags"] == ["no_tool_call"], \
        "the fact is recorded; the FINDING flag is not set, because prose is the right answer"


def test_an_unreachable_engine_exits_2_and_records_nothing(services):
    """Exit 2 is the instrument, and a partial recording would read as a decision set:
    nothing is written (`engine_output_probe.py:163-165` refuses for the same reason)."""
    class Unreachable:
        def post_json(self, path, body):
            raise tcp.ProbeRefused("connect: connection refused")

        def get_text(self, path):
            return ""

    rc = tcp.main(["capture", "--label", "fp8-kv"], client_factory=lambda url: Unreachable())
    assert rc == tcp.EXIT_INSTRUMENT == 2, rc
    assert not (services / "flash-next-canary-toolcalls-fp8-kv.jsonl").exists()


def test_the_default_model_auto_resolves_to_the_served_name_and_is_never_sent(services):
    """`--model auto` (the default the canary step uses) is the name the engine reports on
    `/v1/models`. Sent verbatim it is a 404 on vLLM, which is how the 2026-10-30 bf16 arm
    (#3188) recorded no decisions."""
    rc, stub = _run_capture(services, "fp8-kv", ARM_FP8)
    assert rc == tcp.EXIT_OK, rc
    assert stub.chat_bodies, "the capture asked nothing"
    assert {b["model"] for b in stub.chat_bodies} == {"stub-engine-1"}


def test_auto_against_an_engine_that_names_no_model_exits_2_and_records_nothing(services):
    """No served name is the instrument failing: refuse before the first prompt."""
    class Nameless(StubEngine):
        def get_text(self, path):
            return json.dumps({"object": "list", "data": []})

    stub = Nameless(ARM_FP8)
    rc = tcp.main(["capture", "--label", "fp8-kv"], client_factory=lambda url: stub)
    assert rc == tcp.EXIT_INSTRUMENT == 2, rc
    assert stub.chat_bodies == []
    assert not (services / "flash-next-canary-toolcalls-fp8-kv.jsonl").exists()


def test_an_explicit_model_passes_through_without_asking_the_engine(services):
    stub = StubEngine(ARM_FP8)
    rc = tcp.main(["capture", "--label", "fp8-kv", "--model", "primary"],
                  client_factory=lambda url: stub)
    assert rc == tcp.EXIT_OK, rc
    assert {b["model"] for b in stub.chat_bodies} == {"primary"}


def test_a_report_with_one_arm_in_the_window_refuses_to_decide_and_appends_nothing(services):
    """One arm is not a comparison. This is the shape that made the prose canary report
    agreement across two different questions until #2683 refused it."""
    rc, _ = _run_capture(services, "fp8-kv", ARM_FP8)
    assert rc == tcp.EXIT_OK
    stub = StubEngine(ARM_FP8)
    rc2 = tcp.main(["report", "--label", "fp8-kv"], client_factory=lambda url: stub)
    assert rc2 == tcp.EXIT_INSTRUMENT, rc2
    kinds = [r["kind"] for r in _captured(services, "fp8-kv")]
    assert kinds == ["tool_call"] * 4, f"a verdict was invented where none could be: {kinds}"


# ── clause 4: a same-prose pair differing only in a path splits on the decision ──────
def test_a_pair_differing_only_in_a_path_is_args_differ_yet_within_floor_on_prose(services):
    """The whole reason the leg exists. Two arms, one window, the same prose answers: the
    prose verdict is `within floor` (agreement 1.0 against the committed floor whose
    `agreement_min` is 1.0) and the decision verdict names the prompt and the tool."""
    rc_a, _ = _run_capture(services, "fp8-kv", ARM_FP8)
    rc_b, _ = _run_capture(services, "kv-bfloat16", ARM_BF16)
    assert (rc_a, rc_b) == (tcp.EXIT_OK, tcp.EXIT_OK)

    stub = StubEngine(ARM_BF16)
    rc = tcp.main(["report", "--label", "kv-bfloat16"], client_factory=lambda url: stub)
    assert rc == tcp.EXIT_FINDING, "a split decision is a finding about the engine"
    rows = [r for r in _captured(services, "kv-bfloat16") if r["kind"] == "tool_equivalence"]
    assert len(rows) == 1, rows
    report = rows[0]
    assert report["pairs"] == 4, report
    assert report["tiers"]["args_differ"] == 2, report
    assert report["tiers"]["canonically_equal"] == 1, \
        "the one prompt both arms decided identically: fact_get with the same entity"
    assert report["tiers"]["different_tool"] == 0, report
    assert report["tiers"]["protocol_invalid"] == 1, \
        "both arms declined tc_no_call_expected, and the comparator has no decline tier"
    assert report["count_mismatch"] == [], "both declined, so the call counts agree at zero"
    assert [(item["prompt"], item["tool"]) for item in report["args_differ"]] == [
        ("tc_bash_test_command", "Bash"), ("tc_read_named_path", "Read")], report
    assert report["pair_arms"] == ["fp8-kv", "kv-bfloat16"], report

    line = tcp.render_report(report)
    assert "args_differ=2" in line, line
    assert "tc_read_named_path:Read" in line, f"clause 2 asks the line to NAME the pair: {line}"
    assert "tc_bash_test_command:Bash" in line, line

    # The same two arms' PROSE, through the prose probe against the same stub engine, is
    # within floor — which is what every arm verdict before this item said, and only that.
    out_dir = services / "prose-records"
    out_dir.mkdir()
    for arm in ("fp8-kv", "kv-bfloat16"):
        prose_stub = StubEngine(ARM_FP8)
        # The FULL 21-prompt prose corpus, not a two-prompt slice: the verdict this node
        # contrasts with the decision is the same verdict the prose canary reports for a
        # real arm, so `within floor` here means what it means on an operator's terminal.
        rc_run = probe.main(["run", "--label", arm, "--out-dir", str(out_dir)],
                            client_factory=lambda url: prose_stub)
        assert rc_run == 0, rc_run
    records = sorted(out_dir.glob("*.json"))
    assert len(records) == 2, records
    json_out = services / "prose-verdict.json"
    rc_cmp = probe.main(["compare", "--current", str(records[1]), "--reference", str(records[0]),
                         "--json", "--json-out", str(json_out)])
    assert rc_cmp == 0, "within floor: exit 0, the prose canary saw nothing"
    verdict = json.loads(json_out.read_text())
    assert verdict["diverged"] == 0, verdict
    assert len(verdict["rows"]) == 21, len(verdict["rows"])
    assert all(row["agreement"] == 1.0 for row in verdict["rows"]), verdict["rows"][:2]


def test_two_arms_in_different_windows_are_never_paired(services, monkeypatch):
    """Captures pair only inside one window: the second arm recorded yesterday's boot, and no
    later reader may turn two windows into a comparison (#2683's law, applied to windows)."""
    rc, _ = _run_capture(services, "fp8-kv", ARM_FP8)
    assert rc == tcp.EXIT_OK
    monkeypatch.setenv("CANARY_WINDOW", "W2")
    rc, _ = _run_capture(services, "kv-bfloat16", ARM_BF16)
    assert rc == tcp.EXIT_OK
    rc = tcp.main(["report", "--label", "kv-bfloat16"],
                  client_factory=lambda url: StubEngine(ARM_BF16))
    assert rc == tcp.EXIT_INSTRUMENT, "two windows share no pair to decide on"
    kinds = [r["kind"] for r in _captured(services, "kv-bfloat16")]
    assert kinds == ["tool_call"] * 4, kinds


# ── #3597: a budget a whole call fits inside, and the floor a split is read against ──

#: Every prompt's `text` and `tools` array as `{id: {text, tools}}`, sorted keys — the part of
#: the corpus that decides WHAT is asked, which clause 1 forbids this round to touch.
#: Re-derive it with:
#:   .venvs/lloyd/bin/python -c "import hashlib,json,yaml; \
#:   c=yaml.safe_load(open('eval/tool_call_corpus.yaml')); \
#:   b={p['id']:{'text':p['text'],'tools':p['tools']} for p in c['prompts']}; \
#:   print(hashlib.sha256(json.dumps(b,sort_keys=True).encode()).hexdigest())"
TEXT_AND_TOOLS_SHA = "c8db8afa09d16867d9bc38b2c214a79a3e01c45990c48cf098ae88948437a864"

#: The corpus that PRODUCED the noise line: `max_tokens: 64`. This is the `corpus_sha256` both
#: arms' captures of window kv-3188-20261030T041939Z carry, and that window's one report row
#: reads `pairs=4 tiers={'protocol_invalid': 4, 'different_tool': 0, 'args_differ': 0,
#: 'canonically_equal': 0}` — read out of
#: `~/lloyd-data/logs/services/flash-next-canary-toolcalls-kv-bfloat16.jsonl` on 2026-10-30.
NOISE_LINE_CORPUS_SHA = "3d6709b0ee91b74dbafdbf77f93be9bde623668b5fc020a3e609862ffd0fd421"

#: The shipped arm asked the same four questions a SECOND time in the same window. It parts
#: with itself on one prompt — `tc_read_named_path` comes back naming the `state.py` path this
#: time instead of `worktree.py` — and that single self-disagreement is the whole point of a
#: floor: it is what `temperature: 0` still does at `max_tokens: 256`, and the 2026-10-30
#: window had no way of knowing it because nothing ever asked the arm twice.
FP8_REPEAT = {
    "tc_read_named_path": [
        ("Read", '{"file_path": "/home/alansrobotlab/lloyd/app/harness/state.py"}')],
    "tc_bash_test_command": ARM_FP8["tc_bash_test_command"],
    "tc_structured_key": ARM_FP8["tc_structured_key"],
    "tc_no_call_expected": [],
}


def _report_of(services, label):
    """`report --label` over whatever captures sit in this node's tmp services dir."""
    return tcp.main(["report", "--label", label],
                    client_factory=lambda url: StubEngine(ARM_FP8))


def _last_report(services, arm):
    rows = [r for r in _captured(services, arm) if r["kind"] == "tool_equivalence"]
    assert rows, "the report row is what the fold appends and what the floor rides on"
    return rows[-1]


def test_the_corpus_budget_holds_a_whole_call_and_asks_the_same_questions():
    """Clause 1: the delta to the corpus is the BUDGET and nothing else.

    64 tokens does not hold a call from this model. Every prompt the 2026-10-30 window lost
    ended `finish_reason: length`, and WHICH of the three it lost was per-arm luck (bfloat16
    lost `tc_read_named_path` and `tc_structured_key`, fp8 lost `tc_bash_test_command`), so
    the window's line was a coin flip on truncation wearing a dtype's clothes. 256 held a
    whole call on all three calling prompts when #3597 measured it against live fp8.

    The `tools` arrays and prompt texts are the QUESTION, and a floor is only comparable
    across windows while the question is fixed, so the question's own digest is pinned here: a
    raise that quietly reworded a prompt would move this sha while `max_tokens` looked right.
    """
    corpus = tcp.load_corpus(CORPUS)
    budget = int(corpus["params"]["max_tokens"])
    assert budget >= 256, f"a call did not fit below 256; the file says {budget}"

    bodies = {p["id"]: {"text": p["text"], "tools": p["tools"]} for p in corpus["prompts"]}
    digest = hashlib.sha256(json.dumps(bodies, sort_keys=True).encode()).hexdigest()
    assert digest == TEXT_AND_TOOLS_SHA, (
        "a prompt's text or its tools array moved: the corpus delta is not the budget alone")
    assert [p["id"] for p in corpus["prompts"]] == [
        "tc_read_named_path", "tc_bash_test_command", "tc_structured_key",
        "tc_no_call_expected"], "the corpus is still the same four questions, in order"

    # Clause 1's other half: the shipped corpus is NOT the file that made the noise line. The
    # captures carry the whole file's digest, so a changed sha is precisely what refuses to
    # pair a future arm against those 2026-10-30 rows rather than comparing two questions.
    assert tcp.corpus_sha256(CORPUS) != NOISE_LINE_CORPUS_SHA, (
        "the file still hashes to the 64-token corpus that produced `protocol_invalid=4`")


def test_the_raised_budget_reaches_the_request_body_and_not_just_the_file(services):
    """The same budget read off the wire the engine is sent — the process boundary here is
    corpus bytes on disk versus the HTTP body `capture` posts.

    `capture` lifts `max_tokens` out of the corpus `params` and puts it in the body verbatim,
    so an edit that only moved the number to a key the loader does not read would leave every
    future window starved exactly as before while the YAML looked fixed.
    """
    rc, stub = _run_capture(services, "fp8-kv", ARM_FP8)
    assert rc == tcp.EXIT_OK, rc
    budget = int(tcp.load_corpus(CORPUS)["params"]["max_tokens"])
    assert [body["max_tokens"] for body in stub.chat_bodies] == [budget] * 4, (
        "the corpus budget did not reach all four request bodies")
    assert budget >= 256, budget


def test_a_second_capture_of_one_arm_in_one_window_survives_and_pairs(services):
    """Clause 2: the repeat is no longer thrown away on its way into the pairing.

    The load-bearing fact #3597's triage found is that `same_prompt_pairs` built
    `latest = {arm: {prompt: row}}`, so a second capture under the SAME `--label` inside one
    window overwrote the first and was never paired — capturing the shipped arm twice was a
    no-op before any floor code could exist. Appending always worked, which is why this counts
    the rows in the file AND asks the pairer what it makes of them.
    """
    rc, _ = _run_capture(services, "fp8-kv", ARM_FP8)
    assert rc == tcp.EXIT_OK
    rc, _ = _run_capture(services, "fp8-kv", FP8_REPEAT)
    assert rc == tcp.EXIT_OK, "a second capture of one arm is a recording like any other"
    rc, _ = _run_capture(services, "kv-bfloat16", ARM_BF16)
    assert rc == tcp.EXIT_OK

    fp8_rows = _captured(services, "fp8-kv")
    assert len(fp8_rows) == 8, f"two captures of four prompts is eight rows, not {len(fp8_rows)}"
    assert all(len([r for r in fp8_rows if r["prompt"] == pid]) == 2 for pid in ARM_FP8), \
        "an arm's rows were reduced to one per prompt again"

    pairs = tcp.same_prompt_pairs({"fp8-kv": fp8_rows,
                                   "kv-bfloat16": _captured(services, "kv-bfloat16")})
    same = [p for p in pairs if p["kind"] == tcp.SAME_ARM]
    cross = [p for p in pairs if p["kind"] == tcp.CROSS_ARM]

    assert len(same) == 4, (
        f"every prompt both captures of the arm answered owes a same-arm pair, got {len(same)}")
    assert {p["prompt"] for p in same} == set(ARM_FP8), "the floor must cover every prompt"
    assert all(p["arms"] == ["fp8-kv", "fp8-kv"] for p in same), same[0]
    assert len(cross) == 4, f"the dtype answer is unchanged: four pairs expected, got {len(cross)}"
    assert len(pairs) == 8, f"four dtype pairs plus four floor pairs, not {len(pairs)}"

    # Only the arm that was repeated has a floor: an arm with one capture contributes zero
    # same-arm pairs, and the other arm's repeat is never credited to it.
    lone = tcp.same_prompt_pairs({"fp8-kv": fp8_rows[:4], "kv-bfloat16": fp8_rows})
    floed = {tuple(p["arms"]) for p in lone if p["kind"] == tcp.SAME_ARM}
    assert floed == {("kv-bfloat16", "kv-bfloat16")}, (
        f"the repeat belongs to the arm that was repeated, not both: {sorted(floed)}")
    assert len([p for p in lone if p["kind"] == tcp.SAME_ARM]) == 4


def test_the_report_carries_the_floor_tiers_and_the_prompts_they_measure(services):
    """Clause 3: the shipped arm's self-disagreement is IN the report and ON the printed line.

    The line has to carry the denominator beside the tiers. One of the four prompts is the
    corpus's decline, which `explain` maps to `protocol_invalid` by design, so a reader who
    sees `protocol_invalid=1` with no `prompts=4` reads a structural fact as a measurement of
    engine noise — and sees `args_differ=0` as repeatability when three prompts was the whole
    sample it could have come from.
    """
    _run_capture(services, "fp8-kv", ARM_FP8)
    _run_capture(services, "fp8-kv", FP8_REPEAT)
    _run_capture(services, "kv-bfloat16", ARM_BF16)
    assert _report_of(services, "kv-bfloat16") == tcp.EXIT_OK
    report = _last_report(services, "kv-bfloat16")

    floor = report["floor"]
    assert floor["measured"] is True, floor
    assert floor["arms"] == ["fp8-kv"], "only the arm captured twice has a floor"
    assert floor["prompts"] == 4 and floor["pairs"] == 4, floor
    assert floor["tiers"] == {"protocol_invalid": 1, "different_tool": 0, "args_differ": 1,
                              "canonically_equal": 2}, floor
    assert report["pairs"] == 4 and report["tiers"]["args_differ"] == 2, (
        "the dtype tally must not absorb the same-arm pairs into the number the fold cuts on")

    line = tcp.render_report(report)
    assert ("floor=fp8-kv prompts=4 pairs=4 protocol_invalid=1 different_tool=0 "
            "args_differ=1 canonically_equal=2") in line, line


def test_a_split_the_floor_already_shows_stops_being_a_finding(services):
    """Clause 4, the absorbing half: a tier the shipped arm ALSO produced against itself is
    not a dtype finding, and the exit code the fold reads as "do not promote" has to know it.

    Both arms name different `Read` paths and different `cwd`s, so cross-arm `args_differ` is
    2; the shipped arm, asked twice, parts with itself on `tc_read_named_path` in the same
    tier. The floor's resolution is per-TIER and per-ARM rather than per-prompt because that
    is all a three-prompt sample can resolve — #3597's own finding — which is why the report
    prints the tiers and the denominator rather than a bare verdict.

    This is the case the prose canary has handled since #1743 and this leg never could: the
    disagreement is the engine, not the KV cache, and a veto here blocks a promotion.
    """
    _run_capture(services, "fp8-kv", ARM_FP8)
    _run_capture(services, "fp8-kv", FP8_REPEAT)
    _run_capture(services, "kv-bfloat16", ARM_BF16)
    rc = _report_of(services, "kv-bfloat16")
    assert rc == tcp.EXIT_OK, f"both splits are tiers the floor also shows: rc={rc}"

    report = _last_report(services, "kv-bfloat16")
    assert [item["prompt"] for item in report["args_differ"]] == [
        "tc_bash_test_command", "tc_read_named_path"], report["args_differ"]
    assert tcp.render_report(report).endswith(
        "split=[tc_bash_test_command:Bash, tc_read_named_path:Read]"), tcp.render_report(report)
    for item in report["args_differ"]:
        assert tcp.split_counts_as_finding(item, report["floor"]) is False, item


def test_an_unmeasured_floor_writes_off_no_cross_arm_split(services):
    """Clause 4, the loud half: no second capture means no floor, and a floor nobody took
    absorbs NOTHING.

    This is the shape every window has had until today — the fold captures each arm once
    (`flash-next-canary-step.sh:195`, `:204`) — so it is the half that must not regress: same
    exit code, same named split, plus the line SAYING the floor is unmeasured instead of
    printing four zeros that would read as a floor of zero. Treating "nobody measured" as "the
    engine is perfectly repeatable" would decide a dtype arm on whether someone happened to
    re-run a capture.
    """
    _run_capture(services, "fp8-kv", ARM_FP8)
    _run_capture(services, "kv-bfloat16", ARM_BF16)
    rc = _report_of(services, "kv-bfloat16")
    assert rc == tcp.EXIT_FINDING, f"an unmeasured floor counts every split: rc={rc}"

    report = _last_report(services, "kv-bfloat16")
    floor = report["floor"]
    assert floor["measured"] is False and floor["pairs"] == 0 and floor["prompts"] == 0, floor
    assert tcp.render_floor(floor) == \
        "floor=unmeasured(no second capture: every split counts)", tcp.render_floor(floor)
    assert tcp.split_counts_as_finding({"tier": "args_differ"}, floor) is True
    line = tcp.render_report(report)
    assert "floor=unmeasured" in line and "split=[tc_bash_test_command:Bash" in line, line
