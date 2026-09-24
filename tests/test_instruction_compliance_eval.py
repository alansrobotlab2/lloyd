"""Pins backlog #630: the instruction-density sweep, its scoring, and its artifact.

Clause-by-clause, because the acceptance is the contract:

  1. deterministic prompt — `test_word_list_is_*`, `test_prompt_*`,
     `test_sweep_points_are_the_six_densities_630_names`
  2. both slots through the config's own endpoint resolution, explicit
     `max_tokens` on every request and inside every record —
     `test_resolve_slot_asks_*`, `test_every_request_carries_max_tokens_*`,
     `test_a_disabled_slot_is_refused_*`
  3. mechanical scoring, and an incomplete run classified off its head and tail
     rather than scored as forgetting — `test_presence_is_a_word_boundary_match`,
     `test_an_incomplete_run_is_classified_*`
  4. one record per (engine, N) over all six densities plus an N* per engine —
     `test_baseline_document_*`, `test_n_star_is_the_largest_density_*`,
     `test_the_ceiling_filter_is_spelled_in_the_labels_the_classifier_emits`,
     `test_a_low_scoring_refusal_counts_against_the_ceiling_rather_than_being_excluded`,
     `test_a_density_with_no_comparable_run_cannot_qualify`,
     `test_the_committed_baseline_*`
  5. a one-line repro in the docstring and the JSON, with a stated band —
     `test_repro_command_*`, `test_reproduction_band_is_measured_*`
  6. the pointer into #624 and #552 — `test_pointer_lines_*`

The seam is HTTP. The sweep talks to an engine over a socket, and the code graph
cannot see across that: which model name goes in the body, whether `max_tokens`
is actually sent, whether the answer is read from `content` or from
`reasoning`. Those are claims about bytes on a wire, so a test that monkeypatched
the transport would prove nothing about them. `FakeEngine` is a real
`ThreadingHTTPServer` on a real loopback port, and the sweep's own `urllib` call
crosses it.
"""

from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL_DIR = ROOT / "eval"
SCRIPT = EVAL_DIR / "instruction_compliance_eval.py"
VOCAB = EVAL_DIR / "instruction_compliance_vocab.txt"
COMMITTED_DIR = EVAL_DIR / "instruction-compliance"


def _committed_jsonl() -> Path:
    """The one committed artifact, refusing to guess when there is not one.

    Every test that reads a number out of the tree goes through here, because
    `sorted(...)[-1]` silently picks a winner the moment two artifacts are
    present and the reader is left to trust whichever one won.
    """
    files = sorted(COMMITTED_DIR.glob("baseline-*.jsonl"))
    assert len(files) == 1, (
        f"expected exactly one current artifact under {COMMITTED_DIR}, found "
        f"{[f.name for f in files]}; two committed baselines means every figure "
        "cited from them depends on which one a glob happened to pick")
    return files[0]


def git_tracked(path: Path) -> bool:
    """True when git has this exact path in its index.

    Used instead of parsing `git check-ignore` because the question this file
    has to keep answering is "did the artifact actually land in the tree", and
    the index is the only answer to that.
    """
    return subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        cwd=str(ROOT), capture_output=True).returncode == 0


def _load_sweep():
    """Load the eval script as a module (it lives in `eval/`, not in a package)."""
    spec = importlib.util.spec_from_file_location(
        "instruction_compliance_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ic = _load_sweep()


class FakeEngine:
    """A loopback OpenAI-compatible engine that records what it was asked.

    `reply` is produced per request from the prompt, so a test can say "answer
    with exactly these required words" without the sweep being involved in
    deciding what a good answer is.
    """

    def __init__(self, model_id="FakeModel-1.0", reply_fn=None):
        self.model_id = model_id
        self.reply_fn = reply_fn or (lambda payload: {"content": "", "finish_reason": "stop"})
        self.requests: list[dict] = []
        # The path the body arrived on, per request. `requests` alone cannot
        # answer "did the sweep POST to /v1/chat/completions" — a payload
        # asserted in isolation is a dict, and a dict on the wrong path is a 404.
        self.paths: list[str] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "FakeEngine":
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence per-request stderr spam
                pass

            def _send(self, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802 — stdlib handler name
                self._send({"object": "list",
                            "data": [{"id": outer.model_id, "object": "model"}]})

            def do_POST(self):  # noqa: N802 — stdlib handler name
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length).decode())
                outer.requests.append(payload)
                outer.paths.append(self.path)
                reply = outer.reply_fn(payload)
                self._send({
                    "choices": [{"index": 0,
                                 "message": {"role": "assistant",
                                             "content": reply.get("content"),
                                             "reasoning": reply.get("reasoning", "")},
                                 "finish_reason": reply.get("finish_reason", "stop")}],
                    "usage": {"prompt_tokens": 1234, "completion_tokens": 56},
                })

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}"


def words_from_prompt(prompt: str) -> list[str]:
    """The required-word block a request asked for, read back out of it."""
    header = "each:\n"
    start = prompt.index(header) + len(header)
    end = prompt.index("\n\nRULES:", start)
    return [w.strip() for w in prompt[start:end].replace("\n", ",").split(",")
            if w.strip()]


def obeying_reply(payload: dict) -> dict:
    """An answer that contains every required word, wrapped so no line is a list."""
    words = words_from_prompt(payload["messages"][0]["content"])
    lines = [" ".join(words[i:i + 8]) for i in range(0, len(words), 8)]
    filler = " ".join(["zzq"] * 260)  # `zzq` is 3 letters: never in the vocabulary
    return {"content": "\n".join(lines) + "\n" + filler, "finish_reason": "stop"}


# ── Clause 1: one prompt, N words from a fixed list, same seed same bytes ──


def test_word_list_is_byte_identical_for_a_seed_and_changes_for_another():
    vocab = ic.load_vocab()
    a = ic.sample_words(vocab, 1600, ic.DEFAULT_SEED)
    b = ic.sample_words(vocab, 1600, ic.DEFAULT_SEED)
    assert a == b, "the same seed must yield the same word list, or no re-run is comparable"
    assert len(set(a)) == 1600, "sampling with replacement would silently drop constraints"
    assert ic.sample_words(vocab, 1600, ic.DEFAULT_SEED + 1) != a


def test_sampling_beyond_the_fixed_list_is_refused_not_duplicated():
    vocab = ic.load_vocab()
    with pytest.raises(ValueError, match="exceeds the fixed vocabulary"):
        ic.sample_words(vocab, len(vocab) + 1, 1)


def test_vocab_is_a_fixed_list_large_enough_for_the_top_of_the_sweep():
    words = ic.load_vocab()
    assert len(words) >= max(ic.SWEEP_N), (
        f"N={max(ic.SWEEP_N)} must be sampleable without replacement")
    assert len(set(words)) == len(words), "a duplicated vocabulary word is a hidden N inflation"
    assert all(w.isalpha() and w.isascii() and w.islower() and 5 <= len(w) <= 12
               for w in words), "the scorer's word-boundary match assumes bare lowercase words"


def test_prompt_asks_for_the_report_and_names_every_required_word():
    words = ic.sample_words(ic.load_vocab(), 137, ic.DEFAULT_SEED)
    prompt = ic.build_prompt(words)
    assert f"{ic.TARGET_REPORT_WORDS} words" in prompt
    for word in words:
        assert word in prompt
    assert words_from_prompt(prompt) == words, (
        "the block the prompt renders must be recoverable exactly — it is the "
        "denominator of every score computed from this request")


def test_the_prompt_frame_supplies_no_word_the_sweep_asks_for():
    """A required word already in the instruction would be scored for free."""
    frame = ic.frame_tokens()
    overlap = frame & set(ic.load_vocab())
    assert not overlap, f"vocabulary words the prompt itself says: {sorted(overlap)[:8]}"


def test_sweep_points_are_the_six_densities_630_names():
    assert ic.SWEEP_N == (50, 100, 200, 400, 800, 1600)


def test_the_profile_now_lists_this_arm_and_the_probe_it_replaced_still_disclaims():
    """`eval/lloyd_profile.md` is the inventory a proposer reads before filing work.

    It listed four arms and named none of the ones that landed since; #630's
    triage found the file still claiming there was no compliance eval. The
    sibling assertion holds the reason the profile can say that at all:
    `run_skill_dispatch_probe.py` reports a *predicted* compliance and says so.
    """
    profile = (EVAL_DIR / "lloyd_profile.md").read_text()
    assert "eval/instruction_compliance_eval.py" in profile
    assert "Instruction-density compliance" in profile
    probe = (EVAL_DIR / "run_skill_dispatch_probe.py").read_text()
    assert "predicted" in probe.lower(), (
        "the profile's account of why the compliance arm was missing rests on "
        "this probe disclaiming its own number; if it ever measures real "
        "obedience, both the profile sentence and this assertion are wrong")


# ── Clause 2: slot resolution, and max_tokens on the wire and in the record ──


def test_resolve_slot_asks_the_engine_what_it_serves(monkeypatch):
    import app.config

    with FakeEngine(model_id="Served-Model-9") as engine:
        monkeypatch.setattr(app.config, "_get_model_cfg",
                            lambda name: {"base_url": engine.base_url,
                                          "expect_model": "Served-Model"})
        slot = ic.resolve_slot("primary")
    assert slot["endpoint"] == f"{engine.base_url}/v1/chat/completions"
    assert slot["model"] == "Served-Model-9", (
        "the served id, not the alias: an OpenAI engine answers for the model "
        "it loaded and refuses one naming an alias it never registered")
    assert slot["identity_match"] is True


def test_every_request_carries_max_tokens_and_the_record_says_so():
    """The one line that keeps a truncation from being scored as forgetting."""
    with FakeEngine(reply_fn=obeying_reply) as engine:
        slot = {"slot": "primary", "model": "Served-Model-9",
                "endpoint": f"{engine.base_url}/v1/chat/completions"}
        words = ic.sample_words(ic.load_vocab(), 50, ic.DEFAULT_SEED)
        rec = ic.run_point(slot, words, n=50, seed=ic.DEFAULT_SEED,
                           vocab_sha="sha", max_tokens=6000, temperature=0.0,
                           target_words=800, timeout=30.0)
    assert rec["error"] is None
    sent = engine.requests[0]
    assert sent["max_tokens"] == 6000, "the budget must be sent, not assumed from config.yaml"
    assert sent["model"] == "Served-Model-9"
    assert rec["max_tokens"] == 6000, "and recorded beside the score it bounded"
    assert rec["compliance"] == 1.0
    assert rec["failure_class"] == "ok"
    assert rec["prompt_tokens"] == 1234, "the engine's own token count is kept, not estimated"


def test_the_score_reads_content_and_never_reasoning():
    """A thinking model's `reasoning` may mention every word; it did not write them."""
    def reply(payload):
        words = words_from_prompt(payload["messages"][0]["content"])
        return {"content": " ".join(["zzq"] * 400), "reasoning": " ".join(words),
                "finish_reason": "stop"}

    with FakeEngine(reply_fn=reply) as engine:
        slot = {"slot": "primary", "model": "m",
                "endpoint": f"{engine.base_url}/v1/chat/completions"}
        words = ic.sample_words(ic.load_vocab(), 50, ic.DEFAULT_SEED)
        rec = ic.run_point(slot, words, n=50, seed=1, vocab_sha="sha",
                           max_tokens=6000, temperature=0.0, target_words=800,
                           timeout=30.0)
    assert rec["compliance"] == 0.0
    assert rec["reasoning_kept_out_of_score"] is True


def test_the_thinking_switch_crosses_the_seam_as_vllm_reads_it():
    """The one claim about the wire that changes what every number means.

    `--thinking off` is not a local flag: it is the reason this engine answers
    in prose at all rather than spending the whole budget inside `reasoning` at
    N=50 (module docstring, THINKING). vLLM reads that switch at
    `chat_template_kwargs.enable_thinking`, nested inside `chat_template_kwargs`
    — a flat top-level `enable_thinking` is ignored, and the sweep would then
    publish a density curve that is really a curve of the reasoning budget. So
    the assertion is on the bytes the loopback engine received and the path they
    arrived on, not on the payload dict `run_point` built.
    """
    words = ic.sample_words(ic.load_vocab(), 50, ic.DEFAULT_SEED)
    with FakeEngine(reply_fn=obeying_reply) as engine:
        slot = {"slot": "primary", "model": "Served-Model-9",
                "endpoint": f"{engine.base_url}/v1/chat/completions"}
        off = ic.run_point(slot, words, n=50, seed=ic.DEFAULT_SEED,
                           vocab_sha="sha", max_tokens=6000, temperature=0.0,
                           target_words=800, timeout=30.0)
        sent_off = dict(engine.requests[0])
        on = ic.run_point(slot, words, n=50, seed=ic.DEFAULT_SEED,
                          vocab_sha="sha", max_tokens=6000, temperature=0.0,
                          target_words=800, timeout=30.0, thinking="on")

    assert engine.paths == ["/v1/chat/completions", "/v1/chat/completions"], (
        "both requests went to the completions path the slot's base_url "
        "resolved to; a body posted anywhere else is a 404 recorded as a score")
    assert sent_off["chat_template_kwargs"] == {"enable_thinking": False}, (
        "nested the way vLLM's chat renderer reads it")
    assert "enable_thinking" not in sent_off, (
        "a flat top-level key is the shape that silently does nothing")
    assert off["enable_thinking"] is False, (
        "the record states the switch the wire actually carried")
    assert "chat_template_kwargs" not in engine.requests[1], (
        "`--thinking on` leaves the engine default alone rather than sending a "
        "key an engine that never registered one might refuse outright")
    assert on["enable_thinking"] is None, (
        "None means this run did not close the channel, which changes what its "
        "compliance numbers are a measurement of")


def test_repro_command_names_both_engines_and_the_band_is_stated():
    """Clause 5's other half: the one line must be the WHOLE sweep.

    A repro that recorded only the engine that happened to be up would let the
    reader reproduce half the artifact and notice nothing, which is the same
    defect this file refuses at the slot resolver.
    """
    args = ic.parse_args(["--label", "ifscale-repro"])
    repro = ic.repro_command(args)
    for engine in ic.SLOT_ALIASES:
        assert engine in repro, f"{engine} missing from the recorded repro line"
    tail = repro.split("instruction_compliance_eval.py", 1)[1]
    assert "--reproduction-band" not in tail, (
        "the band is derived from the run, so it must not be a knob a repro "
        "can quietly re-tune")
    args.vocab_sha = "sha"
    baseline = ic.build_baseline(args, [], {}, repro)
    meaning = baseline["reproduction_band_meaning"]
    assert str(baseline["reproduction_band_abs"]) in meaning, (
        "the file states the band beside its definition; a reader who has only "
        "the artifact must be able to see what counts as the same measurement")


def test_the_committed_engine_rows_carry_the_endpoint_and_the_serving_model():
    """The anti-conflation evidence has to ship inside the baseline (#630 finding).

    A row that says only `engine: primary` cannot be audited later for whether
    the sweep measured two engines or the same one twice — the defect
    `app/config.py::resolve_model_alias` documents for Inner Voice, where every
    row said `model=eco` while the engine named eco did not exist.
    """
    engine_rows = [r for r in (json.loads(line)
                               for line in _committed_jsonl()
                               .read_text(encoding="utf-8").splitlines())
                   if r.get("record") == "engine"]
    assert sorted(r["engine"] for r in engine_rows) == list(ic.SLOT_ALIASES), (
        "the artifact must carry a row per slot in the contract, including one "
        "that was not measured, with the reason")
    live = [r for r in engine_rows if r.get("available")]
    assert live, "a baseline with nothing in it is not a measurement"
    for row in live:
        for key in ("configured_base_url", "endpoint", "served_models",
                    "model", "expect_model", "identity_match"):
            assert key in row, f"{row['engine']} row is missing {key}"
        assert row["model"] in row["served_models"], (
            "the name written into the baseline must be a name that engine "
            "actually served, not the name the config hoped for")
        assert row["identity_match"] is not False, (
            f"{row['engine']} was serving something its own config did not "
            "declare, so the row would attribute numbers to the wrong engine")
    for row in engine_rows:
        if not row.get("available"):
            assert "switched off" in row["reason"] and row["reason"], (
                "an unavailable slot must name the flag that closed it")


def test_the_committed_artifact_is_in_a_form_the_loop_can_commit():
    """The artifact's file form is a finding, so it gets a trip wire.

    `eval/baselines/` (the directory #630's acceptance names) is ignored
    outright, and `*.json` is ignored repo-wide — the tracked JSONs under `eval/`
    are the two directories with explicit `!` exceptions
    (`git check-ignore -v eval/uptake/uptake-2026-09-20.json` names `.gitignore`,
    which is why those exist). `.gitignore` is in the automod loop's
    DENIED_GLOBS (`scripts/automod/spec.py`), so no round can add the third
    exception this artifact would need to be a `.json`. The shipped form is
    therefore JSONL, and this asserts two things: my own directory contains no
    `.json` that could only exist via a hand-edited ignore, and no file in it is
    sitting untracked — an artifact git never sees is evidence nobody reads.
    """
    assert not list(COMMITTED_DIR.rglob("*.json")), (
        "a .json under eval/instruction-compliance/ would be untracked unless "
        "someone hand-edited the ignore file this loop may not touch")
    loose = [p for p in COMMITTED_DIR.iterdir()
             if p.is_file() and not git_tracked(p)]
    assert not loose, f"uncommitted artifact(s) in the tree: {loose}"


def test_a_disabled_slot_is_refused_rather_than_measured_on_the_other_engine(monkeypatch):
    """`resolve_model_alias` would hand this sweep the primary for "secondary".

    The sweep's whole output is one row per engine, so inheriting that rewrite
    would produce a baseline with two engines and one model on the machine —
    the failure `app/config.py:361` documents. This test asserts both halves:
    the rewrite really is live, and `resolve_slot` really does not use it.
    """
    import app.config
    import app.llm_slots

    monkeypatch.setattr(app.llm_slots, "is_enabled", lambda program, config=None: False)
    monkeypatch.setattr(app.config, "_get_model_cfg",
                        lambda name: {"base_url": "http://127.0.0.1:59999",
                                      "expect_model": "Whatever"})
    assert app.config.resolve_model_alias("secondary") == "primary", (
        "the trap this test exists for: if that ever stops being true, the "
        "guard below is dead code and this file should say so")
    with pytest.raises(ic.SlotUnavailable, match="secondary_enabled"):
        ic.resolve_slot("secondary")


def test_an_engine_that_is_on_but_not_answering_is_named_as_unavailable(monkeypatch):
    import app.config

    monkeypatch.setattr(app.config, "_get_model_cfg",
                        lambda name: {"base_url": "http://127.0.0.1:59999"})
    ic._served_models_cached.cache_clear()
    with pytest.raises(ic.SlotUnavailable, match="nothing is answering"):
        ic.resolve_slot("primary")


# ── Clause 3: mechanical presence, and why an incomplete run is incomplete ──


def test_presence_is_a_word_boundary_match_not_a_substring():
    words = ["praise", "praised", "quarterly"]
    text = "The quarterly memo praised the team, and praised it again."
    assert set(ic.present_words(text, words)) == {"praised", "quarterly"}, (
        "`praised` contains `praise` as a substring and must not credit it")
    score = ic.score_compliance("QUARTERLY review", ["quarterly"])
    assert score["compliance"] == 1.0, "case is not a forgotten word"
    assert ic.score_compliance("", words)["compliance"] == 0.0
    assert ic.score_compliance("quarterl", ["quarterly"])["compliance"] == 0.0


def test_an_incomplete_run_is_classified_from_its_head_and_tail():
    body = " ".join(["zzq"] * 500)
    # Thinking-budget exhaustion: no content at all, cut off mid-generation.
    assert ic.classify_run("", "length") == "truncated"
    assert ic.classify_run("", "stop") == "off-task"
    # Cut off by the output budget with prose on the page.
    assert ic.classify_run(body, "length") == "truncated"
    # Outright refusal, short, at the head.
    assert ic.classify_run(
        "I cannot include all of those words in a report. Sorry.", "stop") == "refused"
    # Substantial prose that merely opens with a hedge-shaped sentence is not
    # a refusal — the report is there, and so is the measurement.
    assert ic.classify_run(
        f"I cannot promise this reads well. {body}", "stop") == "ok"
    # Voss's polite stop: a finished-looking report that declines to carry on.
    assert ic.classify_run(
        f"{body} ... I'm sorry, but I cannot continue listing every remaining term.",
        "stop") == "late-refusal"
    # Voluntary near-empty answer: it answered a different question.
    assert ic.classify_run("There is no such thing as a report.", "stop") == "off-task"
    assert ic.classify_run(body, "stop") == "ok"


def test_forgetting_has_no_shape_in_the_output_so_it_is_the_absence_of_a_class():
    """`ok` is not a good score; it means none of the four failures was detected."""
    words = ["abelson", "prairie", "valuing"]
    text = f"abelson {' '.join(['zzq'] * 400)}"
    assert ic.classify_run(text, "stop") == "ok"
    assert ic.score_compliance(text, words)["compliance"] < 0.4


def test_a_pasted_constraint_list_is_recorded_as_an_echo_not_a_failure_class():
    words = ic.sample_words(ic.load_vocab(), 20, ic.DEFAULT_SEED)
    prose = " ".join(["zzq"] * 300)
    assert ic.list_echo(prose, words) is False
    pasted = prose + "\n" + ", ".join(words)
    assert ic.list_echo(pasted, words) is True
    assert ic.classify_run(pasted, "stop") == "ok", (
        "echo is a diagnostic beside the score, not a fifth failure class")


def test_a_dump_that_wrapped_across_lines_is_still_recognised_as_a_dump():
    """The shape that made the first committed baseline's N=200 row dishonest.

    Read off that artifact's own record: N=200 repeat-0 ends in a multi-line run
    of bare vocabulary words, `echo_line_count` says 0, and the run was recorded
    at `prose_compliance 0.98` for a report whose prose had written almost none
    of them. The metric #624 and #552 are told to cite was the one number the
    paste had not touched — which is the whole value of seeing the second shape.
    """
    words = ic.sample_words(ic.load_vocab(), 200, ic.DEFAULT_SEED)
    prose = "\n".join(
        f"The quarterly {words[i % 3]} outlook in the {words[(i + 7) % 5]} "
        f"sector stayed mixed while leadership reviewed staffing"
        for i in range(12))
    dumped = " ".join(words).split()
    wrapped = "\n".join(" ".join(dumped[i:i + 20])
                        for i in range(0, len(dumped), 20))
    report = prose + "\n" + wrapped

    echo = ic.echo_line_indexes(report, words)
    assert ic.list_echo(report, words) is True
    assert len(wrapped.splitlines()) == 10, (
        "no line of this dump holds 50 % of a 200-word list, so the one-line "
        "rule cannot see it — that is the property under test")
    assert echo == list(range(12, 22)), (
        "every wrapped dump line is flagged and none of the 12 prose lines is: "
        "the flagged set is exactly the dump. A one-word line would not be a "
        "dump line, which is why the fixture divides into full lines")

    text, echo_count = ic.prose_only(report, words)
    assert echo_count == 10
    assert ic.score_compliance(report, words)["compliance"] == 1.0, (
        "verbatim inclusion still credits the paste — that is IFScale's metric")
    assert ic.score_compliance(text, words)["compliance"] < 0.2, (
        "with the wrapped dump removed the report itself had almost nothing")


def test_a_wrapped_dump_is_not_confused_with_a_report_that_used_the_words():
    """Negative control, because the block rule reads density, not identity.

    Without this the detector could be satisfied by flagging everything: the
    same 200 words woven one-per-nine-tokens into long prose lines must score
    the same with and without the dump guard, or the guard is just a penalty.
    """
    words = ic.sample_words(ic.load_vocab(), 200, ic.DEFAULT_SEED)
    sprinkled = "\n".join(
        "The " + " ".join(words[(i * 3 + k) % 200] if k % 9 == 0 else f"zzq{k}"
                          for k in range(30))
        + " segment showed steady movement across the regions."
        for i in range(14))
    assert ic.echo_line_indexes(sprinkled, words) == []
    assert ic.list_echo(sprinkled, words) is False
    assert 0.05 < ic.score_compliance(sprinkled, words)["compliance"], (
        "the control text really does use the words, so a flag-free score here "
        "is a score the guard left alone, not a score it never saw")


def test_a_short_dump_block_is_flagged_at_the_top_of_the_sweep_too():
    """`ECHO_BLOCK_MIN_WORDS` is why the block rule can see a 1,600-word dump.

    At N=1,600 half the list is 800 words, and a model that pastes 60 of them
    and quits is not halfway through the constraint — it is dumping. The block
    rule therefore needs the smaller of "half the list" and a fixed number of
    distinct required words, or the density the sweep most cares about is the
    density the guard cannot see.
    """
    words = ic.sample_words(ic.load_vocab(), 1600, ic.DEFAULT_SEED)
    filler = " ".join(["zzq"] * 300)
    block = "\n".join(" ".join(words[i:i + 20]) for i in range(0, 60, 20))
    assert len(block.splitlines()) == 3
    assert ic.list_echo(f"{filler}\n{block}", words) is True, (
        "60 dumped words out of 1,600 is an echo, and `ECHO_BLOCK_MIN_WORDS` "
        "is the term that makes it one")
    assert ic.list_echo(filler, words) is False


def test_the_sweep_reports_two_metrics_because_copying_the_list_scores_high():
    """The finding that makes this file's headline number honest.

    Measured on the first sweep: from N=400 up, this engine ends the report by
    pasting the required-word list back. IFScale's metric — verbatim inclusion,
    which is also #630's acceptance metric — calls that full compliance, because
    the words really are in the output. It is a different measurement from
    holding N rules while writing, and a single number silently chosen between
    them would let a copy stand in for compliance.
    """
    words = ic.sample_words(ic.load_vocab(), 40, ic.DEFAULT_SEED)
    report = " ".join(words[:4]) + " " + " ".join(["zzq"] * 300)
    dumped = report + "\n" + ", ".join(words)

    assert ic.score_compliance(dumped, words)["compliance"] == 1.0, (
        "inclusion alone credits the dump — that is IFScale's metric, kept for "
        "comparability with the published cloud numbers")
    prose, echo_count = ic.prose_only(dumped, words)
    assert echo_count == 1
    assert ic.score_compliance(prose, words)["compliance"] < 0.2, (
        "with the dumped line removed the report itself carried almost none of "
        "them, which is the number a length or honouring decision should cite")

    clear = {n: 1.0 for n in ic.SWEEP_N}
    points = _fake_points(clear, engines=("primary",))
    for p in points:
        p["prose_compliance"] = (0.10 if p["n"] >= 400 else 1.0)
    assert ic.n_star(points) == 1600
    assert ic.n_star(points, metric="prose_compliance") == 200, (
        "the two ceilings must be able to disagree, and the record keeps both")


# ── Clause 4: the document, its records, and its N* ──────────────────────


def _fake_points(compliance_by_n: dict[int, float],
                 engines: tuple[str, ...] = ("primary", "secondary")):
    points = []
    for engine in engines:
        for n, value in compliance_by_n.items():
            points.append({"engine": engine, "n": n, "compliance": value,
                           "prose_compliance": value, "list_echo": False,
                           "failure_class": "ok" if value > 0.9 else "truncated",
                           "finish_reason": "stop", "seconds": 3.0})
    return points


def test_baseline_document_has_one_record_per_engine_and_density():
    args = ic.parse_args(["--label", "unit", "--sweep", "50,100,200,400,800,1600"])
    args.vocab_sha = "sha"
    points = _fake_points({n: 1.0 for n in ic.SWEEP_N})
    baseline = ic.build_baseline(args, points, {"primary": {"available": True}},
                                 ic.repro_command(args))
    assert len(baseline["records"]) == 2 * len(ic.SWEEP_N)
    for record in baseline["records"]:
        assert {"engine", "n", "compliance", "failure_class"} <= set(record)
    for engine in ("primary", "secondary"):
        summ = baseline["summary"][engine]
        assert sorted(int(k) for k in summ["points"]) == list(ic.SWEEP_N)
        assert summ["n_star_95"] == 1600
        assert summ["n_star_threshold"] == ic.N_STAR_THRESHOLD


def test_n_star_is_the_largest_density_where_every_repeat_clears_the_bar():
    clear = {n: 1.0 for n in (50, 100, 200, 400)}
    clear.update({800: 0.96, 1600: 0.61})
    points = _fake_points(clear, engines=("primary",))
    points.append({"engine": "primary", "n": 800, "compliance": 0.80,
                   "failure_class": "ok", "finish_reason": "stop", "seconds": 3.0})
    assert ic.n_star(points) == 400, (
        "a density that clears 95 % once and misses it once has no 95 % ceiling "
        "there; the mean would report the luckiest draw as a capacity")
    none = _fake_points({n: 0.5 for n in ic.SWEEP_N}, engines=("primary",))
    assert ic.n_star(none) is None, "below the first swept point is a real answer"


def test_the_ceiling_filter_is_spelled_in_the_labels_the_classifier_emits():
    """A filter that matches on names the classifier never returns filters nothing.

    The review rung of gate SM_20260924_040055 caught `COMPARABLE_CLASSES`
    carrying `off_task` and `late_refusal` while `classify_run` emits
    `off-task` and `late-refusal`. The two underscored names matched no record, so
    the partition inverted against its own stated intent: the two classes that ARE
    compliance failures were dropped from the ceiling and then counted as
    exclusions — which `pointer_lines` tells #624 means "output-budget truncation
    or engine error" — while the classes that measure the harness kept counting
    toward the ceiling.

    So the assertion is not "these constants are spelled right" (a typo can be
    fixed once and reintroduced); it is that the filter's alphabet and the
    classifier's alphabet are one alphabet, checked from the classifier's side:
    `classify_run` is driven until it has produced every label the file claims it
    can produce, and the two ceiling sets must then cover exactly those labels
    plus `engine-error`, with no overlap.
    """
    body = " ".join(["zzq"] * 500)
    emitted = {
        ic.classify_run("", "length"),          # thinking budget exhausted
        ic.classify_run("", "stop"),            # answered nothing
        ic.classify_run("I cannot include all of those words in a report. Sorry.",
                        "stop"),                # outright refusal
        ic.classify_run(f"{body} ... I'm sorry, but I cannot continue listing "
                        "every remaining term.", "stop"),   # polite late stop
        ic.classify_run(body, "stop"),          # finished report
    }
    assert emitted == set(ic.FAILURE_CLASSES), (
        "`FAILURE_CLASSES` claims the labels this classifier can return; a label "
        "in that tuple which no input produces is a name the records can never "
        "carry, and every set built on it is then a filter on an empty set")
    assert (set(ic.COMPARABLE_CLASSES) | set(ic.NON_COMPARABLE_CLASSES)
            == set(ic.FAILURE_CLASSES) | {ic.ENGINE_ERROR}), (
        "every label a record can hold is either comparable or excluded; a label "
        "outside both sets is silently non-comparable, and the exclusion count "
        "then describes it as a truncation to the reader of #624")
    assert not (set(ic.COMPARABLE_CLASSES) & set(ic.NON_COMPARABLE_CLASSES)), (
        "a class in both sets is counted against the ceiling and excused for it "
        "in the same breath")
    # And the semantic split itself, named rather than inferred from the sets:
    # a refusal or an off-topic answer is a compliance failure; a run the harness
    # cut off, or an endpoint that never answered, is not a measurement at all.
    assert set(ic.COMPARABLE_CLASSES) == {ic.OK, ic.REFUSED, ic.LATE_REFUSAL,
                                          ic.OFF_TASK}
    assert set(ic.NON_COMPARABLE_CLASSES) == {ic.TRUNCATED, ic.ENGINE_ERROR}


def test_a_low_scoring_refusal_counts_against_the_ceiling_rather_than_being_excluded():
    """The counterexample the review rung handed back, as a permanent trip wire.

    One run at 0.99 and one `late-refusal` at 0.10 at the same density. The
    refusal is the engine being asked to hold 1,600 rules and declining late in
    the answer: that is the capacity limit this eval exists to find, so it has to
    pull the ceiling down. Under the previous spelling it was excluded instead,
    the density qualified on the single clean run, `n_star_95` stayed at 1600, and
    `n_star_excluded_runs` rose — the number that is supposed to disclose a thin
    ceiling was silently reporting a refusal as an output-budget cutoff.

    The paired direction is pinned in the same test, because both halves are the
    rule: a `truncated` run at 0.10 (the harness's own `max_tokens`, not the
    engine) must NOT pull the ceiling down, or the eval reports the box as the
    model.
    """
    points = _fake_points({n: 0.99 for n in ic.SWEEP_N}, engines=("primary",))
    points.append({"engine": "primary", "n": 1600, "compliance": 0.10,
                   "prose_compliance": 0.10, "list_echo": False,
                   "failure_class": ic.LATE_REFUSAL,
                   "finish_reason": "stop", "seconds": 3.0})
    assert ic.n_star(points) == 800, (
        "a density whose second repeat is a 0.10 late-refusal has no 95 % "
        "ceiling at that density")
    assert ic.n_star_excluded(points) == 0, (
        "the refusal is a failing run, not an excluded one; an exclusion count "
        "that rises on a refusal is the ceiling laundering its own failures")

    cut = _fake_points({n: 0.99 for n in ic.SWEEP_N}, engines=("primary",))
    cut.append({"engine": "primary", "n": 1600, "compliance": 0.10,
                "prose_compliance": 0.10, "list_echo": False,
                "failure_class": ic.TRUNCATED,
                "finish_reason": "length", "seconds": 3.0})
    assert ic.n_star(cut) == 1600, (
        "an output-budget cutoff is the harness running out of tokens, which says "
        "nothing about the engine's rule-tracking and must not be scored as it")
    assert ic.n_star_excluded(cut) == 1, (
        "and the exclusion is disclosed beside the ceiling, not silent in it")


def test_a_density_with_no_comparable_run_cannot_qualify():
    """The rule fails closed: nothing to compare is not a cleared bar.

    `all()` over an empty list is True, so a density whose every repeat was cut
    off or never answered qualifies vacuously the moment the empty case is not
    guarded — and the vacuous density is the TOP of the sweep, so the reported N*
    becomes the largest number in the file. That is the failure mode this guards:
    a GPU someone else was holding turns a 1,600-rule ceiling into a headline.
    """
    points = _fake_points({n: 1.0 for n in (50, 100, 200, 400, 800)},
                          engines=("primary",))
    points.append({"engine": "primary", "n": 1600, "compliance": 0.0,
                   "prose_compliance": 0.0, "list_echo": False,
                   "failure_class": ic.TRUNCATED,
                   "finish_reason": "length", "seconds": 3.0})
    points.append({"engine": "primary", "n": 1600, "compliance": 0.0,
                   "prose_compliance": 0.0, "list_echo": False,
                   "failure_class": ic.ENGINE_ERROR,
                   "finish_reason": None, "seconds": 0.4})
    assert ic.n_star(points) == 800, (
        "the top density has zero comparable runs, so the ceiling is the highest "
        "density that has any")
    assert ic.n_star_excluded(points) == 2

    nothing = _fake_points({n: 1.0 for n in ic.SWEEP_N}, engines=("primary",))
    for p in nothing:
        p["failure_class"] = ic.ENGINE_ERROR
    assert ic.n_star(nothing) is None, (
        "if no repeat anywhere was comparable then no N* exists at all — None "
        "here means the sweep measured nothing, which is the honest answer and "
        "not a licence to report the largest swept N")


def test_reproduction_band_is_measured_from_the_run_not_asserted():
    """2x the observed within-run spread, floored at a DECLARED cross-run figure.

    Half of this is measured and half is carried, and the two halves are not the
    same kind of claim. What the tree re-measures is each point's within-run
    repeat spread, committed in the artifact; `band_from_spread` doubles it, which
    is what these assertions exercise. What it cannot re-measure is the floor:
    three sweeps of the identical command on 2026-09-24 moved N=50's mean by up to
    0.0625 and N=1600's by 0.0469, so 2x the largest is 0.125 and the floor rounds
    that up to 0.14 — but two of those three sweeps were overwritten by the next
    run of the same command, so 0.14 is a number this commit carries, not one it
    can show. Re-running the repro command re-measures the floor rather than
    checking it; asserting it here only pins that the constant did not drift.
    """
    assert ic.MIN_REPRODUCTION_BAND == 0.14
    assert ic.band_from_spread([0.0, 0.005]) == ic.MIN_REPRODUCTION_BAND
    assert ic.band_from_spread([]) == ic.MIN_REPRODUCTION_BAND
    assert ic.band_from_spread([0.04]) == ic.MIN_REPRODUCTION_BAND, (
        "2x a tight within-run spread still has to clear the drift the "
        "instrument actually shows between runs")
    assert ic.band_from_spread([0.20]) == 0.40, "a genuinely wide spread widens it"


def test_the_committed_baseline_covers_every_density_point_and_records_n_star():
    """The committed artifact, re-derived rather than admired.

    Every assertion here recomputes a recorded figure from the artifact's own
    per-repeat rows. That is deliberate: the previous form of this test asked
    `n_star_95 is None or n_star_95 in SWEEP_N` and `echo_runs >= 0`, which a
    baseline of nothing but nulls and zeroes passes — a trip wire that cannot
    fail is how an empty artifact ships as a measurement. Clause 4 asks for a
    recorded N* per engine, so an engine that was measured must have a NUMBER
    for N*, and the number must be the one its own repeats produce.
    """
    records = [json.loads(line) for line in
               _committed_jsonl().read_text(encoding="utf-8").splitlines()
               if line.strip()]
    by_kind: dict[str, list[dict]] = {}
    for record in records:
        by_kind.setdefault(record["record"], []).append(record)
    config = by_kind["config"][0]
    assert config["sweep_n"] == list(ic.SWEEP_N)
    assert config["enable_thinking"] is False, (
        "the committed curve has to be a curve of instruction density, not of "
        "the reasoning budget — see the module docstring's THINKING section")

    engine_rows = {row["engine"]: row for row in by_kind["engine"]}
    assert sorted(engine_rows) == list(ic.SLOT_ALIASES), (
        "both slots in the contract get a row, measured or honestly refused")
    measured = sorted(name for name, row in engine_rows.items()
                      if row.get("available"))
    assert measured, "a committed baseline with no engine measured measures nothing"
    summary_rows = {row["engine"]: row for row in by_kind["summary"]}
    assert sorted(summary_rows) == measured, (
        "a summary row appears for an engine the sweep actually measured, and "
        "only for one: the refused slot's row stays `available: false` with its "
        "reason and contributes no numbers")

    flat_records = [r for row in by_kind["records"] for r in row["records"]]
    for engine in measured:
        row = summary_rows[engine]
        runs = [r for r in flat_records if r["engine"] == engine]
        assert sorted(int(k) for k in row["points"]) == list(ic.SWEEP_N), (
            "every engine row must carry all six densities, not the ones that "
            "came back well")
        assert len(runs) == len(ic.SWEEP_N) * config["repeats"], (
            "one record per (engine, N, repeat) — the flat rows are what the "
            "summary is derived from, so a shortfall means a density is missing")
        assert row["n_star_95"] is not None, (
            "clause 4 asks for the largest N with compliance >= 95 %; an engine "
            "that was measured and recorded a null N* did not answer that "
            "question, and the artifact is not a baseline until it does")
        assert row["n_star_95"] == ic.n_star(runs) == ic.n_star(
            runs, threshold=row["n_star_threshold"]), (
            "the recorded ceiling is what this engine's own committed repeats "
            "produce under the scorer in this diff — not a hand-written number")
        assert row["n_star_95_prose"] == ic.n_star(runs, metric="prose_compliance"), (
            "the prose-scored ceiling is the one #624 and #552 cite, so it ships "
            "in the artifact and is recomputable, not derived by hand")
        assert row["echo_runs"] == sum(1 for r in runs if r["list_echo"]), (
            "`echo_runs >= 0` was the assertion this replaces: nothing about a "
            "run count can fail against a lower bound of zero")
        spreads = []
        for point in row["points"].values():
            assert "reproduction_band" in point and "mean_prose_compliance" in point
            vals = [float(rep["compliance"]) for rep in point["repeats"]]
            assert point["runs"] == len(point["repeats"]) == config["repeats"]
            spreads.append(max(vals) - min(vals))
        assert row["max_repeat_spread"] == round(max(spreads), 4), (
            "the number the headline reproduction band is derived from is "
            "recomputable from the repeats, or the band is an assertion")
        for run in runs:
            assert run["prose_compliance"] <= run["compliance"], (
                "prose is a subset of the output, so a prose score above the "
                "inclusion score is a scorer bug, not a result")
            assert (run["echo_line_count"] > 0) == run["list_echo"]
            assert run["failure_class"] in by_kind["meta"][0]["failure_classes"]
            assert run["max_tokens"] == config["max_tokens"], (
                "clause 2's other half: the budget is on the record, so a "
                "truncation cannot be read as forgetting by a later reader")
    flat = json.dumps(records)
    for needle in ('"repro_command"', '"reproduction_band_abs"', '"n_star_95"',
                   '"prose_compliance"'):
        assert needle in flat, f"{needle} is missing from the committed artifact"


# ── Clause 5: one command, in the prose and in the JSON ──────────────────


def test_repro_command_reparses_and_the_baseline_records_it():
    args = ic.parse_args(["--label", "ifscale-unit", "--repeats", "2"])
    repro = ic.repro_command(args)
    assert repro.startswith(".venvs/lloyd/bin/python eval/instruction_compliance_eval.py")
    tail = repro.split("instruction_compliance_eval.py", 1)[1]
    again = ic.parse_args(shlex.split(tail))
    assert again.label == "ifscale-unit"
    assert list(again.sweep) == list(ic.SWEEP_N), (
        "the default sweep must survive the round trip")
    assert again.repeats == 2 and again.max_tokens == ic.DEFAULT_MAX_TOKENS
    assert again.thinking == ic.DEFAULT_THINKING
    assert ic.repro_command(again) == repro
    args.vocab_sha = "sha"
    baseline = ic.build_baseline(args, [], {}, repro)
    assert baseline["repro_command"] == repro
    assert baseline["reproduction_band_abs"] >= ic.MIN_REPRODUCTION_BAND


def test_the_script_docstring_carries_the_repro_line_and_the_band_is_stated():
    doc = ic.__doc__ or ""
    assert "REPRO" in doc
    assert "eval/instruction_compliance_eval.py" in doc
    args = ic.parse_args([])
    args.vocab_sha = "sha"
    baseline = ic.build_baseline(args, [], {}, ic.repro_command(args))
    assert baseline["reproduction_band_meaning"], (
        "the band has to say what it bounds, or a re-run cannot be judged against it")


# ── Clause 6: the pointer into #624 and #552 ─────────────────────────────


def test_the_pointer_lines_cited_by_624_are_re_derivable_from_the_committed_file():
    """The sentence a person pastes into #624, rendered from the shipped artifact.

    Two reasons this reads the committed JSONL rather than a synthetic document:
    the committed one is in the shape the synthetic test cannot reach (its prose
    ceiling is None — no density cleared 95 % once the dumps are removed), and
    clause 6's whole purpose is that #624 cites a number anyone can re-derive.
    The first version of the renderer turned that null into "N* = N* below the
    swept floor", which the synthetic case — where every ceiling is a number —
    could not have caught.
    """
    baseline = ic.from_jsonl(_committed_jsonl().read_text(encoding="utf-8"))
    lines = ic.pointer_lines(baseline)
    text = "\n".join(lines)
    recorded = baseline["summary"]["primary"]["n_star_95"]
    assert recorded is not None
    assert f"N* = {recorded}" in text, (
        "the pointer carries the same ceiling the artifact records, generated "
        "from it rather than retyped")
    assert "N* = N*" not in text and "= N*" not in text, (
        "the null prose ceiling renders as a sentence, not a doubled prefix")
    assert "N* below the swept floor" in text, (
        "the prose ceiling in the shipped artifact is null, so the pointer must "
        "say what a null means instead of printing None")
    assert "#630" in text and "NOT MEASURED" in text
    assert baseline["repro_command"] in text, (
        "the cited number has to be re-derivable by the person citing it")


def test_the_committed_jsonl_reads_back_as_the_document_the_script_writes():
    """The JSONL is the durable form, so reading it is part of the contract.

    `.json` is ignored repo-wide, so the only committed copy of these numbers is
    the JSONL, and any later job that wants to compare two baselines goes through
    `from_jsonl`. A reader that loses a figure on the way in silently changes
    what the committed baseline says.
    """
    baseline = ic.from_jsonl(_committed_jsonl().read_text(encoding="utf-8"))
    assert baseline["schema"] and baseline["item"] == "#630"
    assert baseline["config"]["sweep_n"] == list(ic.SWEEP_N)
    assert len(baseline["records"]) == len(ic.SWEEP_N) * baseline["config"]["repeats"]
    assert baseline["summary"]["primary"]["n_star_95"] == ic.n_star(
        baseline["records"]), "a re-read document yields the same ceiling"
    assert baseline["summary"]["primary"]["points"]["1600"]["reproduction_band"] \
        == 0.425, "the per-point band survives the round trip through the wire form"
    assert baseline["engines"]["secondary"]["available"] is False


def test_from_jsonl_rejects_a_line_without_a_kind_and_keeps_an_unknown_one():
    """A truncated artifact and a future record type are different answers."""
    with pytest.raises(ValueError, match="without a record kind"):
        ic.from_jsonl('{"schema": "instruction-compliance/1"}\n')
    kept = ic.from_jsonl('{"record": "nightly_note", "said": "hello"}\n')
    assert kept["nightly_note"] == [{"said": "hello"}], (
        "an unknown record type is visible to an old reader rather than silently "
        "dropped from the figures it reads")


def test_pointer_lines_name_the_item_and_every_measured_n_star():
    assert ic.POINTER_TARGETS == ("#624", "#552")
    args = ic.parse_args(["--label", "ifscale-x"])
    args.vocab_sha = "sha"
    points = _fake_points({n: 1.0 for n in ic.SWEEP_N}, engines=("primary",))
    baseline = ic.build_baseline(args, points,
                                 {"primary": {"available": True,
                                              "configured_base_url": "http://127.0.0.1:9",
                                              "model": "Served-Model-9",
                                              "served_models": ["Served-Model-9"]},
                                  "secondary": {"available": False,
                                                "reason": "secondary_enabled is false"}},
                                 ic.repro_command(args))
    lines = ic.pointer_lines(baseline)
    text = "\n".join(lines)
    assert "#630" in text
    assert "N* = 1600" in text, "the length-cap decision must cite a number"
    assert "Served-Model-9" in text, (
        "a pointer that names the SLOT but not the engine serving it is the "
        "conflation this file refuses everywhere else")
    assert "N* = " in text and "dumps excluded" in text, (
        "both ceilings go in: the inclusion one for comparability with the "
        "paper, the prose one for the honouring decision")
    assert "NOT MEASURED" in text and "secondary_enabled" in text, (
        "an engine the sweep could not reach must appear in the pointer as "
        "unmeasured, or the pointer reads as a clean bill for two engines")
    assert ic.repro_command(args) in text, "the cited number has to be re-derivable"


# ── The whole sweep, across the HTTP seam ────────────────────────────────


def test_a_sweep_end_to_end_records_six_points_and_refuses_the_missing_engine(
        tmp_path, monkeypatch):
    import app.config
    import app.llm_slots
    import app.paths

    with FakeEngine(model_id="Served-Model-9", reply_fn=obeying_reply) as engine:
        monkeypatch.setattr(app.config, "_get_model_cfg",
                            lambda name: {"base_url": engine.base_url,
                                          "expect_model": "Served-Model"})
        monkeypatch.setattr(app.llm_slots, "is_enabled",
                            lambda program, config=None: program != "agent-llm-secondary")
        monkeypatch.setattr(app.paths, "EVAL_BASELINES_DIR", tmp_path)
        ic._served_models_cached.cache_clear()
        copy = tmp_path / "committed" / "baseline-unit.jsonl"
        code = ic.main(["--label", "unit", "--engines", "primary,secondary",
                        "--committed-copy", str(copy)])

    assert code == 0
    written = list(tmp_path.glob("unit-*.json"))
    assert len(written) == 1
    baseline = json.loads(written[0].read_text())
    assert len(baseline["records"]) == len(ic.SWEEP_N)
    assert {r["n"] for r in baseline["records"]} == set(ic.SWEEP_N)
    assert all(r["max_tokens"] == ic.DEFAULT_MAX_TOKENS for r in baseline["records"])
    assert all(r["engine"] == "primary" for r in baseline["records"]), (
        "the disabled slot contributed no compliance rows — it is not allowed "
        "to be measured by proxy on the engine that is up")
    assert baseline["summary"]["primary"]["n_star_95"] == 1600
    assert baseline["engines"]["secondary"]["available"] is False
    assert "secondary_enabled" in baseline["engines"]["secondary"]["reason"]
    assert len(engine.requests) == len(ic.SWEEP_N)
    assert set(engine.paths) == {"/v1/chat/completions"}, (
        "every request of the sweep crossed the chat-completions seam, so the "
        "payload assertions above describe what the engine was actually sent")

    assert copy.exists()
    rows = [json.loads(line) for line in
            copy.read_text(encoding="utf-8").splitlines() if line.strip()]
    kinds = {row["record"] for row in rows}
    assert {"meta", "config", "engine", "summary", "records"} <= kinds
    meta = next(row for row in rows if row["record"] == "meta")
    assert meta["repro_command"] == baseline["repro_command"]
    assert meta["reproduction_band_abs"] == baseline["reproduction_band_abs"]
    assert meta["scoring"] == "mechanical: regex word-boundary presence, no LLM judge"
