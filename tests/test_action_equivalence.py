"""#2843 pins `eval/action_equivalence.py`: a pairwise tool-call comparator whose
verdict is a *decision*, not a prose match.

The premise it exists against, verified on this tree: the engine-integrity
canary decides only on prose. `eval/engine_output_probe.py` sets each prompt's
tier from bitwise token agreement at :479 (`f["tier"] = "exact" if
f["pairs_bitwise"] == f["pairs"] else "jitter"`), its per-prompt row comes from
`compare_prompt` (:403), and every exceed reason inside `compare_records` (:534)
is an agreement or a logprob threshold — `grep -n tool_call` on that 822-line
file returns nothing. So a change that keeps the text and moves the tool call is
invisible to it, and no pairwise tool-call comparator existed: `git grep -ln
action_equivalence` and the four tier symbols all came back empty at triage.

Clause map — each acceptance clause has its own section, and the node names say
which clause they answer:

  1. one tier out of four, and an unreadable call is `protocol_invalid`
  2. canonicalisation is a named rule list; the notations it forgives
  3. no rule may rewrite a value that names a target
  4. the labelled fixture, its tier distribution, and per-tier precision/recall
     against `MIN_PER_TIER_SCORE`, with a non-zero exit when a tier is below it
  5. the miss: same prose, different path — the comparator says `args_differ`,
     the canary says "within floor"

One naming note, because clause 5 as written asks for `compare_runs`: no
function of that name is defined anywhere on this tree — `grep -rn
'def compare_runs' --include=*.py .` returns nothing (the only hits for the bare
string are this file's own prose). The canary's run-level comparison is
`compare_records`, and that is what is driven here — the same two records, the
same floor, the same verdict the `compare` subcommand prints.

Every engine here is a hand-built record; no request is made to any engine.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import eval.action_equivalence as ae
import eval.engine_output_probe as probe

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "eval" / "action_equivalence.py"

#: One real call, verbatim from session 20261016_090437_autocode_4bea msg38, used
#: as the base for every notational variant below. The variants are written as
#: literal strings on purpose: the question is about the BYTES the engine sent,
#: and a `json.dumps` in the test would rewrite them before the comparator saw
#: them and quietly prove nothing.
BASE_ARGS = ('{"command": "cd ~/lloyd && sed -n \'240,320p\' scripts/automod/round.py", '
             '"summary": "Reading lock release path"}')


def variant_key_order() -> str:
    return ('{"summary": "Reading lock release path", '
            '"command": "cd ~/lloyd && sed -n \'240,320p\' scripts/automod/round.py"}')


def variant_compact() -> str:
    return ('{"command":"cd ~/lloyd && sed -n \'240,320p\' scripts/automod/round.py",'
            '"summary":"Reading lock release path"}')


def variant_newline_outside() -> str:
    return BASE_ARGS + "\n"


def variant_prose_newline() -> str:
    # the `\n` goes INSIDE the summary value: drop the value's closing quote and
    # the object's brace, then put a newline escape and both closers back.
    return BASE_ARGS[:-2] + '\\n"}'


def variant_all_three() -> str:
    return ('{"summary":"Reading lock release path\\n",'
            '"command":"cd ~/lloyd && sed -n \'240,320p\' scripts/automod/round.py"}\n')


def pair(name: str, args: str) -> dict:
    return {"name": name, "arguments": args, "origin": "test"}


def fixture_pair(pair_id: str) -> dict:
    return next(p for p in ae.FIXTURE if p["id"] == pair_id)


# --- clause 1: exactly one tier, and an unreadable call is protocol_invalid ----

def test_classify_returns_exactly_one_of_the_four_tiers_for_every_fixture_pair():
    assert ae.TIERS == ("protocol_invalid", "different_tool", "args_differ",
                        "canonically_equal")
    for p in ae.FIXTURE:
        tier = ae.classify(p["a"], p["b"])
        assert isinstance(tier, str)
        assert tier in ae.TIERS, f"{p['id']} produced {tier!r}, outside TIERS"


@pytest.mark.parametrize("bad_name", [None, "", "   ", 42, ["Read"]])
def test_a_call_with_no_usable_tool_name_is_protocol_invalid(bad_name):
    good = pair("Read", '{"file_path": "/tmp/a"}')
    assert ae.classify(pair(bad_name, '{"file_path": "/tmp/a"}'), good) == ae.PROTOCOL_INVALID
    # and not the name difference: the refusal outranks the comparison
    assert ae.classify(good, pair(bad_name, '{"file_path": "/tmp/a"}')) == ae.PROTOCOL_INVALID


@pytest.mark.parametrize("bad_args", [
    None, "", '{"command": "cd ~/lloyd && sed -n \'300,400p\' app/djev_sh',  # cut mid-string
    '{"file_path": ',                  # cut mid-object
    '["/tmp/a"]',                      # parses, but not an object
    "3",                               # parses, but a scalar
    42,                                # not a string at all
])
def test_an_argument_string_that_will_not_parse_to_an_object_is_protocol_invalid(bad_args):
    good = pair("Glob", '{"pattern": "/tmp/*.md"}')
    assert ae.classify(pair("Glob", bad_args), good) == ae.PROTOCOL_INVALID
    assert ae.classify(good, pair("Glob", bad_args)) == ae.PROTOCOL_INVALID


def test_protocol_invalid_outranks_different_tool_because_there_is_nothing_to_compare():
    unreadable = {"name": None, "arguments": '{"file_path": "/tmp/a"}', "origin": "test"}
    other = pair("Glob", '{"pattern": "/tmp/*.md"}')
    assert ae.classify(unreadable, other) == ae.PROTOCOL_INVALID
    # the reason says so, rather than reporting the name difference as a decision
    assert "unreadable" in ae.explain(unreadable, other)["why"]
    # ... while two READABLE calls with different names do reach different_tool
    assert ae.classify(pair("Glob", '{"pattern": "/tmp/*.md"}'),
                       pair("Read", '{"file_path": "/tmp/a"}')) == ae.DIFFERENT_TOOL


def test_explain_and_classify_never_disagree_about_a_pair():
    # one decision path (`_decide`) shared by the verdict and the report, so a
    # corpus report cannot quietly describe a different decision than it made.
    for p in ae.FIXTURE:
        assert ae.explain(p["a"], p["b"])["tier"] == ae.classify(p["a"], p["b"])


# --- clause 2: canonicalisation is a named rule list --------------------------

def test_canonicalisation_is_an_explicit_named_ordered_rule_list():
    assert [r.name for r in ae.CANONICALISATION_RULES] == [
        "json_key_order", "trailing_newline_in_prose_value", "separator_whitespace"]
    assert [r.surface for r in ae.CANONICALISATION_RULES] == ["structure", "value", "structure"]
    for r in ae.CANONICALISATION_RULES:
        assert r.may_touch.strip(), f"{r.name} does not say what it may touch"
    # exactly one rule may reach into a value at all; the two structural ones are
    # functions of the parsed payload, so no string content can move under them.
    assert sum(1 for r in ae.CANONICALISATION_RULES if r.surface == "value") == 1


def test_applying_the_whole_rule_list_forgives_key_order_separators_and_a_trailing_newline():
    base = pair("Bash", BASE_ARGS)
    for name, args in [("key order", variant_key_order()),
                       ("whitespace around separators", variant_compact()),
                       ("a trailing newline outside the JSON", variant_newline_outside()),
                       ("a trailing newline on the summary value", variant_prose_newline()),
                       ("all three at once", variant_all_three())]:
        assert ae.classify(base, pair("Bash", args)) == ae.CANONICALLY_EQUAL, name
        assert ae.canonicalise(BASE_ARGS) == ae.canonicalise(args), name


def test_the_shipped_fixture_pair_that_carries_all_three_notations_is_labelled_and_classed_equal():
    p = fixture_pair("bash-all-three-notations")
    assert p["label"] == ae.CANONICALLY_EQUAL
    assert ae.classify(p["a"], p["b"]) == ae.CANONICALLY_EQUAL
    # the two strings really do differ byte-wise, or the pair would prove nothing
    assert p["a"]["arguments"] != p["b"]["arguments"]


def test_canonicalisation_is_idempotent_over_every_string_in_the_fixture():
    for p in ae.FIXTURE:
        for side in ("a", "b"):
            args = p[side]["arguments"]
            try:
                once = ae.canonicalise(args)
            except ValueError:
                continue          # a non-object payload is clause 1's business
            assert ae.canonicalise(once) == once, f"{p['id']}[{side}] not idempotent"


# --- clause 3: no rule may rewrite a value that names a target ----------------

def test_a_path_that_differs_only_by_a_trailing_newline_is_args_differ_not_equal():
    left = pair("Read", '{"file_path": "/home/alansrobotlab/obsidian/memory/2026-10-16.md"}')
    right = pair("Read", '{"file_path": "/home/alansrobotlab/obsidian/memory/2026-10-16.md\\n"}')
    assert ae.classify(left, right) == ae.ARGS_DIFFER
    # the same not a target-naming value: the identical newline on `summary` is equal
    assert ae.classify(pair("Bash", BASE_ARGS),
                       pair("Bash", variant_prose_newline())) == ae.CANONICALLY_EQUAL


def test_a_command_that_differs_only_by_a_trailing_newline_is_args_differ():
    left = pair("Bash", '{"command": "git status\\n", "summary": "Checking"}')
    right = pair("Bash", '{"command": "git status", "summary": "Checking"}')
    assert ae.classify(left, right) == ae.ARGS_DIFFER


def test_a_same_tool_call_with_the_wrong_path_is_never_reported_as_the_same_decision():
    # the #2320 shape: one character off the filename, everything else identical.
    p = fixture_pair("edit-wrong-file-one-character")
    assert p["label"] == ae.ARGS_DIFFER
    assert ae.classify(p["a"], p["b"]) == ae.ARGS_DIFFER
    assert p["a"]["arguments"] != p["b"]["arguments"]


def test_no_target_naming_value_survives_canonicalisation_changed():
    # the guard, on every target-naming value the fixture carries: key order and
    # compact serialisation move bytes around a path, never a byte of the path.
    for p in ae.FIXTURE:
        for side in ("a", "b"):
            args = p[side]["arguments"]
            try:
                original = json.loads(args)
                canon = ae.canonicalise(args)          # raises if a rule moved a target
            except ValueError:
                continue
            ae.assert_targets_preserved(original, json.loads(canon))


def test_a_rule_widened_to_rewrite_every_value_is_refused_instead_of_forgiving_a_path(monkeypatch):
    """The rule list cannot be silently widened.

    Install the rule someone will eventually be tempted to write — strip a
    trailing newline from *every* string value, "because whitespace" — and show
    two things: `assert_targets_preserved` refuses its output, and the pair that
    rule would have called equal is reported as NOT comparable rather than as
    the same decision. The failure direction is the whole point: an instrument
    that cannot decide must never answer "same".
    """
    def strip_every_value(obj):
        if isinstance(obj, dict):
            return {k: strip_every_value(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [strip_every_value(v) for v in obj]
        return obj.rstrip("\r\n") if isinstance(obj, str) else obj

    hostile = ae.CanonRule(
        "trailing_newline_in_every_value", "value", "the rule that must not exist",
        lambda c: replace(c, value=strip_every_value(c.value)))
    monkeypatch.setattr(ae, "CANONICALISATION_RULES",
                        (ae.CANONICALISATION_RULES[0], hostile, ae.CANONICALISATION_RULES[2]))

    args = '{"file_path": "/home/alansrobotlab/obsidian/memory/2026-10-16.md\\n"}'
    with pytest.raises(ae.CanonicalisationRefused) as exc:
        ae.canonicalise(args)
    assert "file_path" in str(exc.value)

    left = pair("Read", '{"file_path": "/home/alansrobotlab/obsidian/memory/2026-10-16.md"}')
    assert ae.classify(left, pair("Read", args)) != ae.CANONICALLY_EQUAL
    assert ae.classify(left, pair("Read", args)) == ae.PROTOCOL_INVALID


def test_prose_keys_are_a_named_smallest_possible_set_and_never_name_a_target():
    assert ae.PROSE_KEYS == frozenset({"summary", "note"})
    ae._prose_keys_are_not_targets()          # must not raise
    assert not any(ae.names_target(k) for k in ae.PROSE_KEYS)
    for key in ("file_path", "path", "command", "pattern", "to", "url", "task_id",
                "folderPath", "calendarId"):
        assert ae.names_target(key), key


# --- clause 4: the labelled fixture, its distribution and its score -----------

def test_the_fixture_ships_one_human_label_per_pair_and_supports_every_tier():
    assert len(ae.FIXTURE) >= 12
    ids = [p["id"] for p in ae.FIXTURE]
    assert len(set(ids)) == len(ids), "pair ids must be unique to cite them in a report"
    for p in ae.FIXTURE:
        assert set(p) == {"id", "label", "note", "a", "b"}, p["id"]
        assert p["label"] in ae.TIERS, p["id"]
        assert p["note"].strip(), f"{p['id']} has no stated reason for its label"
        for side in ("a", "b"):
            assert p[side]["origin"].strip(), f"{p['id']}[{side}] has no provenance"
            assert isinstance(p[side]["arguments"], str)
    score = ae.score_fixture()
    assert score["unscored_tiers"] == [], "a tier with no labelled pair is untested"
    assert set(score["label_distribution"]) == set(ae.TIERS)


def test_the_comparator_reproduces_its_own_tier_distribution_on_the_labels():
    score = ae.score_fixture()
    assert score["distribution"] == score["label_distribution"]
    assert sum(score["distribution"].values()) == len(ae.FIXTURE)
    assert score["n_pairs"] == len(ae.FIXTURE)


def test_every_tier_scores_at_or_above_the_named_threshold():
    assert ae.MIN_PER_TIER_SCORE == 1.0
    score = ae.score_fixture()
    for tier, s in score["per_tier"].items():
        assert s["support"] > 0, tier
        assert s["precision"] is not None and s["recall"] is not None, tier
        assert s["precision"] >= ae.MIN_PER_TIER_SCORE, (tier, s)
        assert s["recall"] >= ae.MIN_PER_TIER_SCORE, (tier, s)
    assert score["below_threshold"] == []
    assert score["mismatches"] == []


def test_the_score_is_capable_of_failing_a_wrong_label_is_reported():
    broken = copy.deepcopy(list(ae.FIXTURE))
    for p in broken:
        if p["label"] == ae.ARGS_DIFFER:
            p["label"] = ae.CANONICALLY_EQUAL
            break
    score = ae.score_fixture(tuple(broken))
    assert score["mismatches"], "a wrong label must be caught, or the score is decoration"
    assert ae.ARGS_DIFFER in score["below_threshold"] or ae.CANONICALLY_EQUAL in score["below_threshold"]


def test_main_exits_zero_on_the_shipped_fixture_and_names_the_distribution(capsys):
    assert ae.main([]) == 0
    out = capsys.readouterr().out
    for tier in ae.TIERS:
        assert tier in out
    assert "precision" in out and "recall" in out
    assert "PASS" in out


def test_main_exits_one_when_a_tier_falls_below_the_threshold(monkeypatch):
    broken = copy.deepcopy(list(ae.FIXTURE))
    for p in broken:
        if p["label"] == ae.ARGS_DIFFER:
            p["label"] = ae.CANONICALLY_EQUAL
            break
    monkeypatch.setattr(ae, "FIXTURE", tuple(broken))
    assert ae.main([]) == 1


def test_main_exits_two_when_a_tier_has_no_labelled_pair_at_all(monkeypatch, capsys):
    # an empty class must not read as clean: dropping every protocol_invalid pair
    # is a fixture that would silently stop testing that tier.
    monkeypatch.setattr(ae, "FIXTURE",
                        tuple(p for p in ae.FIXTURE if p["label"] != ae.PROTOCOL_INVALID))
    assert ae.main([]) == 2
    assert "no labelled pair" in capsys.readouterr().err


def test_running_the_script_as_a_program_reports_the_score_and_exits_zero():
    # the seam the owed corpus job crosses: this file is executed, not imported,
    # and its exit code is the contract.
    proc = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True,
                          cwd=str(ROOT), timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "canonically_equal" in proc.stdout
    json_proc = subprocess.run([sys.executable, str(SCRIPT), "--json"], capture_output=True,
                               text=True, cwd=str(ROOT), timeout=120)
    assert json_proc.returncode == 0, json_proc.stderr
    assert json.loads(json_proc.stdout)["distribution"] == ae.score_fixture()["distribution"]


# --- clause 5: the miss, demonstrated offline ---------------------------------

PROSE = ["I", " will", " read", " the", " daily", " note", " now", "."]
LOGPROBS = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]


def canary_record(label: str, *, path: str) -> dict:
    """One run record in the canary's own shape, for one prompt.

    `tokens`/`token_logprobs` are the whole of what `compare_prompt` reads, and
    the recorded window is what `max_tokens` cut: both runs' recorded windows are
    the same eight tokens, because the generated text agreed up to the tool call
    and the window closed before the divergent path. That is not a rigged
    fixture — it is the instrument's own boundary: `compare_prompt` computes
    agreement over the recorded prefix and reports `first_divergence: None` for
    two truncated-equal windows (:403). What differs is the tool call the harness
    parsed out of the two generations, which is not in a record at all — see
    `test_the_canary_record_has_no_field_that_could_hold_a_tool_call`.
    """
    row = {"id": "tool_call_path", "tags": ["tool_xml"], "tokens": list(PROSE),
           "token_logprobs": list(LOGPROBS), "text": "".join(PROSE),
           "finish_reason": "length", "prompt_tokens": 40, "cached_tokens": 0,
           "median_logprob": -0.45, "prompt_sha256": "0" * 64, "elapsed_s": 0.5,
           "running_before": 0.0}
    return {"schema": 1, "kind": "engine_output_probe", "label": label,
            "started": "2026-10-16T12:00:00+00:00", "finished": "2026-10-16T12:00:02+00:00",
            "engine": {"venv": "/venv", "vllm_version": "0.2.1", "kv_cache_dtype": "fp8",
                       "kv_cache_size_tokens": "844969"},
            "corpus": {"path": "eval/engine_output/corpus.yaml", "sha256": "0" * 64,
                       "n": 1, "max_tokens": 8},
            "params": {"temperature": 0, "logprobs": 1, "model": "primary", "cache_salt": "none"},
            "gauges_before": {}, "gauges_after": {},
            "prompts": [row]}


RIGHT = pair("Read", '{"file_path": "/home/alansrobotlab/obsidian/memory/2026-10-16.md"}')
WRONG = pair("Read", '{"file_path": "/home/alansrobotlab/obsidian/memory/2026-10-15.md"}')


def test_the_comparator_calls_the_same_prose_different_path_pair_args_differ():
    assert ae.classify(RIGHT, WRONG) == ae.ARGS_DIFFER


def test_the_canary_calls_that_same_pair_within_floor_on_its_own_measured_floor():
    ref, cur = canary_record("arm-bf16", path="2026-10-16"), canary_record("arm-fp8", path="2026-10-15")
    # The floor is not a loose threshold chosen here: it is `build_floor` over
    # these two records, so every axis is exactly what the pair produced.
    floor = probe.build_floor([ref, cur])
    assert floor["prompts"]["tool_call_path"]["tier"] == "exact", \
        "an identical pair is tier `exact` (:479) — the strictest tier, decided on everything"

    c = probe.compare_prompt(ref["prompts"][0], cur["prompts"][0])
    assert c["agreement"] == 1.0
    assert c["first_divergence"] is None
    assert c["median_lp_delta"] == 0.0
    assert c["token_lp_delta"] == 0.0

    result = probe.compare_records(ref, cur, floor)
    assert result["diverged"] == 0
    row = result["rows"][0]
    assert row["exceeds"] is False and row["reasons"] == []
    assert "within floor" in probe.render_comparison(result)

    # and the two verdicts really are about the same step: same prose tokens,
    # different decision. This one line is the gap #2843 is about.
    assert ae.classify(RIGHT, WRONG) == ae.ARGS_DIFFER


def test_the_canary_record_has_no_field_that_could_hold_a_tool_call():
    """Why the miss is structural, not a threshold that could be tightened.

    There is nothing in a run record for a path to disagree with: no
    `tool_calls`, no `function`, no `arguments` key at any depth. Tightening the
    floor cannot catch a decision the record cannot represent, which is why the
    comparator needs its own input rather than a stricter canary.
    """
    rec = canary_record("x", path="2026-10-16")

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)

    names = set(keys(rec)) | set(keys(probe.compare_records(
        rec, canary_record("y", path="2026-10-15"), probe.build_floor([rec, canary_record("y", path="2026-10-15")]))))
    assert not {k for k in names if "tool" in k.lower() or k.lower() in ("function", "arguments")}, \
        sorted(names)


# --- the seam a recorded corpus arrives across --------------------------------

def test_coerce_accepts_the_wire_shape_sessions_and_the_shadow_store_actually_hold():
    # `~/.local/state/...` and sessions/*.json store the OpenAI-shaped call, with
    # the arguments left as the raw string — the shape the owed corpus run feeds in.
    wire = {"id": "chatcmpl-tool-9501c66f1b5d518e", "call_id": "chatcmpl-tool-9501c66f1b5d518e",
            "type": "function",
            "function": {"name": "skills_read",
                         "arguments": '{"name": "automod-change-own-code"}'}}
    bare = {"name": "skills_read", "arguments": '{"name": "automod-change-own-code"}'}
    tuple_form = ("skills_read", '{"name": "automod-change-own-code"}')
    assert ae.classify(wire, bare) == ae.CANONICALLY_EQUAL
    assert ae.classify(tuple_form, wire) == ae.CANONICALLY_EQUAL
    assert ae.coerce(wire).name == "skills_read"
    # provenance never decides anything: same payload, different origin, still equal
    a = ae.ToolCall("skills_read", '{"name": "a"}', origin="run 1")
    b = ae.ToolCall("skills_read", '{"name": "a"}', origin="run 2")
    assert ae.classify(a, b) == ae.CANONICALLY_EQUAL


def test_the_comparator_touches_no_engine_and_no_store():
    # "pure, engine-free" has to be checkable, not asserted. The module may only
    # import the standard library, must not import any Lloyd module (above all
    # not the canary whose verdict it contradicts), and must not open anything:
    # no socket, no file, no subprocess. The owed corpus job runs it over
    # recorded pairs, so a hidden read here would silently change what a tier
    # means depending on where the job ran.
    src = SCRIPT.read_text(encoding="utf-8")
    imported = set()
    for ln in src.splitlines():
        ln = ln.strip()
        if ln.startswith("import "):
            imported.update(part.strip().split(".")[0]
                            for part in ln[len("import "):].split(","))
        elif ln.startswith("from ") and " import " in ln:
            imported.add(ln[len("from "):].split(" import ")[0].strip().split(".")[0])
    assert imported <= {"__future__", "argparse", "json", "sys", "dataclasses",
                        "typing"}, imported
    assert "eval" not in imported and "app" not in imported, imported
    for forbidden in ("open(", "socket", "subprocess", "httpx", "requests", "urllib",
                      "8096"):
        assert forbidden not in src, forbidden

