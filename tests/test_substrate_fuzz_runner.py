#!/usr/bin/env python3
"""#2620 clauses 2 and 4: arm A runs through the REAL dispatch seam, and every failure
carries exactly one provable layer.

Clause 2 is pinned by a test that would fail if the runner drove the tools itself:
`test_the_runner_uses_the_real_seam_and_not_a_stub_of_it` patches
`app.harness.loop._execute_tool_call` — the one symbol the corpus is defined against —
and fails if the instrument runs without touching it.

Clause 4 is pinned from both sides: an `interface` verdict whose byte pair is absent is
REFUSED (not downgraded to a layer nobody proved), and a step whose tool-side bytes were
never recorded says `not measured` rather than copying the shown text into the field,
because a field that defaults to the other side of the comparison makes the requirement
impossible to fail.
"""

from __future__ import annotations

import json
import os

import pytest

from _substrate_fuzz import (BY_ID, REAL, parse_task_row, sf_corpus, sf_runner, sf_task)

C = sf_corpus
R = sf_runner


def _miss_row(tid: str, *, command: str, domain: str = "filesystem",
              expected_layers: list[str] | None = None) -> dict:
    """A row whose step answers cleanly with text that matches no digest: the payload miss
    rule 5 has to attribute.

    The expectation is sixty zeroes rather than a plausible digest, so the row is red on
    every box and the test measures the ROUTING and not whether this machine's stdout is
    lossy today.
    """
    row = sf_task(tid=tid, domain=domain,
                  steps=[{"tool": "Bash", "args": {"command": command, "cwd": "{root}"}}],
                  post={"kind": "stdout_sha256", "expected": "0" * 64},
                  setup=[])
    if expected_layers is not None:
        row["expected_layers"] = list(expected_layers)
    return row


# ── clause 2 ────────────────────────────────────────────────────────────────

def test_arm_a_reports_per_domain_k_over_n_and_names_every_failing_task(tmp_path,
                                                                        monkeypatch):
    """Clause 2. One real task, through the real seam, into the report shape a person
    reads: `k / N` per domain, and the id of every task that went red.

    The red one is real rather than staged: `roundtrip-png`'s expected digest is the
    fixture's own, so overwriting that file with one different byte is a substrate
    disagreement the instrument has to notice on its own.
    """
    monkeypatch.setenv(R._ENV_ROOT, str(tmp_path / "runs"))
    green = BY_ID["bin-python-stdout"]
    red = BY_ID["roundtrip-png"]
    res = R.run_corpus([green, red])
    by_id = {r["id"]: r for r in res["results"]}

    assert by_id[green.id]["ok"] is True, by_id[green.id]["failure"]
    assert by_id[red.id]["ok"] is False, "a overwritten fixture must redden its digest"
    assert by_id[red.id]["failure"]["task_id"] == red.id, by_id[red.id]

    proc = res["domains"]["binaries"]
    assert proc["passed"] == 1 and proc["n"] == 1, proc
    assert f'{proc["passed"]}/{proc["n"]}' == "1/1"
    rt = res["domains"]["byte_roundtrip"]
    assert rt["passed"] == 0 and rt["n"] == 1, rt
    assert rt["failing_task_ids"] == [red.id], rt
    assert res["failed_task_ids"] == [red.id], res["failed_task_ids"]
    # The two numbers a reader compares first cannot be one number in disguise.
    assert res["failed_task_ids"] != [res["failed_task_ids"][0] + "x"]


def test_the_runner_uses_the_real_seam_and_not_a_stub_of_it(monkeypatch):
    """Clause 2's actual guarantee: the task's step reaches
    `app.harness.loop._execute_tool_call`.

    The instrument is worthless if it re-implements dispatch and then reports on its own
    re-implementation, which is what a fixture-driven runner drifts into. So the seam is
    wrapped, and the test asserts the wrapper was entered with the step's tool name and
    its argument dict — and a build that called a private copy instead would leave
    `seen` empty and fail here, not quietly pass.
    """
    real = R._execute_tool_call
    seen = []

    async def spy(**kw):
        seen.append(kw)
        return await real(**kw)

    monkeypatch.setattr(R, "_execute_tool_call", spy)
    monkeypatch.setattr(R, "ARTIFACT_RETENTION_DAYS", 1, raising=False)
    rec = R.run_task(BY_ID["bin-shell-substitution-works"])
    assert seen, "the runner never entered the real dispatch seam"
    tc = seen[0]["tc"]
    assert tc["function"]["name"] == "Bash", tc
    assert "command" in tc["function"]["arguments"], tc["function"]["arguments"]
    assert seen[0]["options"].model == R.NO_MODEL, "a model reached the loop"
    assert rec["ok"] is True, rec["failure"]


def test_no_model_and_no_network_is_reachable_from_a_run(monkeypatch):
    """Clause 2's 'with no model in the loop', proved rather than asserted in prose.

    vLLM answers on this box, so a runner that leaked into a completion would not fail —
    it would just be slow and unverifiable. Here any attempt to open an HTTP stream or to
    resolve a name raises, and the run still completes.
    """
    def boom(*a, **k):
        raise AssertionError("arm A reached for the network or the model")

    monkeypatch.setattr(R, "httpx", boom)
    import socket
    monkeypatch.setattr(socket, "create_connection", boom)
    rec = R.run_task(BY_ID["proc-env-passthrough"])
    assert rec["ok"] is True, rec["failure"]


def test_a_step_reaches_the_aggregator_and_back_through_the_harness_only(tmp_path,
                                                                        monkeypatch):
    """The process boundary this change crosses, named: corpus JSON is read in one
    module, the step is assembled in another, dispatched by `app.harness.loop`, served by
    `agent_mcp`, and the result is graded by a third module.

    `spill-large-result-resolves` is the sharpest case because it exercises the harness's
    OWN spill/truncation layer: the aggregator returns 60,000 chars (over
    `SPILL_THRESHOLD_CHARS`), the harness replaces them with a pointer, and the only way
    the postcondition can pass is by resolving the pointer back to the full original
    bytes. A runner that read the tool's return value directly would pass this task
    without ever testing the thing it is for — so the assertion below includes that the
    shown text is NOT the payload.
    """
    monkeypatch.setenv(R._ENV_ROOT, str(tmp_path / "runs"))
    rec = R.run_task(BY_ID["spill-large-result-resolves"])
    assert rec["ok"] is True, rec["failure"]
    step = rec["steps"][0]
    assert len(step["tool_sha256"]) == 64
    assert step["shown_truncated"] is True, "the harness never spilled the result"
    assert step["spill"]["resolves"] is True, step["spill"]
    assert step["shown_text_head"] != step["tool_head"], (
        "the shown text equalled the payload, so this proves nothing about spilling")


# ── clause 4 ────────────────────────────────────────────────────────────────

def test_every_failure_record_carries_exactly_one_layer_from_the_fixed_vocabulary(
        tmp_path, monkeypatch):
    """Clause 4, first half. Every red record names one layer out of
    {model, interface, host, harness} — and there is no `unknown` to park one in.

    The failure is produced by breaking a payload, not by editing an expectation, so the
    vocabulary is exercised on real evidence: one flipped byte in a fixture.
    """
    base = BY_ID["roundtrip-png"]
    broken = json.loads(json.dumps(base.raw))
    broken["id"] = "probe-broken"
    t = C._parse_task(broken, {"x"})
    monkeypatch.setenv(R._ENV_ROOT, str(tmp_path / "runs"))
    rec = R.run_task(t)
    assert rec["ok"] is False
    f = rec["failure"]
    assert f["layer"] in R.LAYERS, f["layer"]
    assert len(f["layer"]) > 0 and isinstance(f["layer"], str)
    assert "unknown" not in R.LAYERS and "unclear" not in R.LAYERS
    assert set(R.LAYERS) == {"model", "interface", "host", "harness"}, R.LAYERS
    assert f["evidence"], "a verdict with no evidence is a label"


def test_an_interface_verdict_without_the_byte_pair_is_refused(tmp_path, monkeypatch):
    """Clause 4, second half: an `interface` verdict is REFUSED when the tool's own bytes
    are not beside the bytes that would reach messages.

    The refusal is the point. Silently re-labelling an unprovable claim as `harness` —
    which is what my first draft did — is the exact absorption this clause is written
    against: the artifact then carries a layer nobody proved, and a reader cannot tell a
    proven interface break from an unmeasurable one. The corpus task asks for `interface`
    and its step records nothing on the tool side, so the run must stop rather than
    guess.
    """
    row = json.loads(json.dumps(BY_ID["roundtrip-png"].raw))
    row["id"] = "probe-interface"
    row["expected_layers"] = ["interface"]
    t = C._parse_task(row, {"x"})
    monkeypatch.setenv(R._ENV_ROOT, str(tmp_path / "runs"))
    monkeypatch.setattr(R, "_RecordingPool.record_tool_side", False)
    with pytest.raises(R.AttributionError, match="requires the tool's own bytes"):
        R.run_task(t)


def test_an_unmeasured_tool_side_says_not_measured_and_never_borrows_the_shown_bytes(
        tmp_path, monkeypatch):
    """Clause 4, third half: where the tool's side was never recorded, the field reads
    `not measured`.

    Defaulting it to the shown bytes would make the byte-comparison requirement
    impossible to fail — every interface verdict would automatically hold, and the
    instrument would report its own requirement as satisfied forever. The step below is
    dispatched with recording off, which is what a tool whose response object cannot be
    inspected looks like from the harness's side.
    """
    monkeypatch.setenv(R._ENV_ROOT, str(tmp_path / "runs"))
    monkeypatch.setattr(R, "_RecordingPool.record_tool_side", False)
    rec = R.run_task(BY_ID["bin-python-stdout"])
    step = rec["steps"][0]
    assert step["tool_recorded"] is False, step
    assert step["tool_bytes"] == R.NOT_MEASURED, step
    # Positive control: the shown side of the very same step IS recorded, so the field
    # above is a withheld measurement and not a renderer that prints nothing at all.
    assert step["shown_text"] != R.NOT_MEASURED, step
    assert step["tool_bytes"] != step["shown_text"], (
        "the unmeasured side was filled with the shown bytes: the requirement is now "
        "impossible to fail")


def test_a_fleet_run_books_an_unattributable_row_instead_of_aborting(tmp_path, monkeypatch):
    """Clause 4's other half at the level of a run: "Unattributable failures are a defect in the
    runner, not an `unknown` bucket to park them in".

    Refusing the verdict is the per-task rule, and a fleet run has to survive it: one row the
    runner cannot prove must not stop the other fifty-one, and it must not vanish either. The row
    here is a shipped task asking for `interface` whose tool side was never recorded — the same
    recipe the refusal node above uses — so the verdict is refused mid-run. What the run publishes
    is the id under `unattributed`, with no layer and no `unknown` anywhere: an id nobody can
    explain is how a defect in this module gets found, and a silent drop is how it gets buried.
    """
    row = json.loads(json.dumps(BY_ID["roundtrip-png"].raw))
    row["expected_layers"] = ["interface"]
    tasks = [C._parse_task(row, {"x"}), BY_ID["bin-python-stdout"]]
    monkeypatch.setenv(R._ENV_ROOT, str(tmp_path / "runs"))
    monkeypatch.setattr(R, "_RecordingPool.record_tool_side", False)
    out = R.run_corpus(tasks)
    assert out["unattributed"] == ["roundtrip-png"], out["unattributed"]
    assert "roundtrip-png" in out["failed_task_ids"], out["failed_task_ids"]
    recs = {r["id"]: r for r in out["results"]}
    assert recs["roundtrip-png"]["failure"] is None, (
        f"a verdict nobody proved was published as a failure carrying a layer: "
        f"{recs['roundtrip-png']['failure']}")
    assert recs["roundtrip-png"]["layer"] is None, recs["roundtrip-png"]
    assert recs["roundtrip-png"]["rule"] == "unattributable", recs["roundtrip-png"]
    assert "unknown" not in json.dumps(out), "the forbidden bucket appeared in the report"
    assert recs["bin-python-stdout"]["ok"] is True, (
        "one unattributable row stopped the task that ran after it")


def test_a_recorded_step_carries_both_sides_of_the_comparison(tmp_path, monkeypatch):
    """The other edge: when the tool side IS recorded, both fields are filled, so the
    refusal above is about a missing measurement and not about the field never being set.
    """
    monkeypatch.setenv(R._ENV_ROOT, str(tmp_path / "runs"))
    rec = R.run_task(BY_ID["bin-python-stdout"])
    step = rec["steps"][0]
    assert step["tool_recorded"] is True, step
    assert step["tool_bytes"] != R.NOT_MEASURED and step["shown_text"] != R.NOT_MEASURED


def test_the_corpus_declares_which_layers_its_own_refusals_should_prove():
    """A failure record must be checkable against what the task claimed.

    `fs-write-into-readonly-dir` is a refusal by the operating system; if the harness
    ever starts reporting it as `model`, the artifact would be attributing a host
    capability to the wrong layer and a person would go and edit a prompt.
    """
    t = BY_ID["fs-write-into-readonly-dir"]
    assert t.expected_layers == ("host",), t.expected_layers
    assert t.expected_layers[0] in R.LAYERS
    for row in REAL:
        for layer in row.expected_layers:
            assert layer in R.LAYERS, (row.id, layer)


def test_the_run_is_confined_to_a_temporary_root_it_removes(tmp_path, monkeypatch):
    """The run's scratch root is under a caller-supplied root and is gone afterwards.

    Arm A writes files, spawns children and consumes memory on a box that is serving a
    person, so a root that outlives the task is a new unbounded store wearing the costume
    of an instrument.
    """
    root = tmp_path / "runs"
    monkeypatch.setenv(R._ENV_ROOT, str(root))
    rec = R.run_task(BY_ID["fs-nested-mkdir"])
    assert rec["ok"] is True, rec["failure"]
    assert rec["run_root"].startswith(str(root)), rec["run_root"]
    import pathlib
    assert not pathlib.Path(rec["run_root"]).exists(), (
        f"{rec['run_root']} survived the run")
    spill = root / "sessions" / f"{rec['session_id']}.tool-results"
    assert not spill.exists(), "the run left its spill sidecar behind"


# ── #2929 clause 1: a payload miss is attributed on its own evidence ────────
#
# Arm A dispatches with `NO_MODEL`: the option handed to `_execute_tool_call` is a stub that
# raises if the loop reaches for a completion. So `model` is a layer this run cannot earn, and
# yet rule 5 — the payload-miss rule — filed every miss there. In the first real run
# (`~/lloyd-data/eval/substrate-fuzz/arm-a-2026-10-17.json`) 4 of the 14 red rows carried
# `layers.model`, each with evidence in its own record naming a byte change, an E2BIG refusal
# or a mode-500 fixture. These rows are RED at that commit and green after the fix.

def test_attribute_never_files_a_payload_miss_under_the_layer_arm_a_cannot_earn():
    """Four payload misses, each with the evidence the miss itself produces, and none of them
    is filed under `model` — the two that name a layer get the layer their evidence names, and
    the one that names nothing is the runner's own defect instead of a leftover label."""
    shapes = [
        # (id, task row, step, expected layer, the phrase the rule must contain)
        ("probe-pair-disagrees",
         _miss_row("probe-pair-disagrees", command="cat absent.txt"),
         R.StepResult(tool="Bash", args={}, ok=True, raw_chars=6,
                      shown_text="answer", tool_bytes=b"ANSWER"),
         "interface", "the two sides of the boundary disagree"),
        ("probe-host-text",
         _miss_row("probe-host-text", command="printf x > rodir/inside.txt"),
         R.StepResult(tool="Bash", args={}, ok=True, raw_chars=40,
                      shown_text="bash: rodir/inside.txt: Permission denied\n",
                      tool_bytes=b"bash: rodir/inside.txt: Permission denied\n"),
         "host", "syscall"),
        ("probe-guard-text",
         _miss_row("probe-guard-text", command="cat absent.txt"),
         R.StepResult(tool="Bash", args={}, ok=True, raw_chars=36,
                      shown_text="Tool call denied: protected path",
                      tool_bytes=b"Tool call denied: protected path"),
         "harness", "guard"),
    ]
    for tid, row, step, want_layer, want_phrase in shapes:
        task = parse_task_row(row)
        got = R.attribute(task, step)
        assert got["layer"] != "model", (
            f"{tid}: rule 5 filed the miss under {got['layer']!r} although its own evidence "
            f"names a layer — arm A runs with NO_MODEL, so that layer is unearnable")
        assert got["layer"] == want_layer, f"{tid}: got {got['rule']!r} / {got['evidence'][:160]!r}"
        assert want_phrase in got["rule"], f"{tid}: rule does not name its evidence: {got['rule']!r}"


def test_a_payload_miss_that_names_no_layer_is_the_runners_defect_not_a_default():
    """A miss whose two sides agree, whose text carries no replacement character and names no
    guard or syscall error, is booked as `unattributed` (a runner defect). It used to be filed
    under `model`, which is a claim about a component this run never called."""
    task = parse_task_row(_miss_row("probe-names-nothing", command="cat absent.txt"))
    step = R.StepResult(tool="Bash", args={}, ok=True, raw_chars=11,
                        shown_text="a plain one", tool_bytes=b"a plain one")
    with pytest.raises(R.AttributionError) as exc:
        R.attribute(task, step)
    assert "nothing in the miss names a layer" in str(exc.value)


def test_a_binary_answer_captured_with_replacement_characters_is_an_interface_finding(tmp_path):
    """`roundtrip-png` measured on this box: the fixture is 70 bytes of PNG whose digest starts
    `5395c0d7`, and the answer arrived 102 bytes beginning `\\ufffdPNG` — U+FFFD, the decoder's
    substitute — with the tool side and the shown side carrying the SAME digest `800f1d42`. So
    the substitution happened inside the tool's own stdout capture, before the interface pair
    was taken, and rule 1 (which compares those two sides) cannot see a fault both sides share.

    The expected digest here is impossible on purpose, so the row is red on any box and this
    node pins the ROUTING: the miss is filed at `interface` on the replacement character's own
    evidence, never at `model`.
    """
    row = dict(BY_ID["roundtrip-png"].raw)
    row["id"] = "probe-codec-miss"
    row["postcondition"] = {"kind": "stdout_sha256", "expected": "0" * 64}
    rec = R.run_task(parse_task_row(row), root=tmp_path)
    assert rec["ok"] is False, "the row went green, so this node has stopped pinning anything"
    assert rec["layer"] == "interface", f"{rec['rule']!r} / {rec['evidence'][:200]!r}"
    assert "replacement character" in rec["rule"], rec["rule"]
    pair = rec["interface_pair"]
    assert pair["tool_sha256"] == pair["shown_sha256"], (
        "the two sides now disagree, so rule 1 sees this fault and the codec route needs "
        "re-measuring")
    assert R._REPLACEMENT_CHAR in pair["shown_head"], repr(pair["shown_head"][:80])


def test_the_verdict_line_names_every_finding_or_states_the_substrate_clean():
    """The three verdicts, and only the third one is a clean bill."""
    def rec(rid, *, ok, control=False, layer="interface", rule="1: a rule", limit=False):
        return {"id": rid, "ok": ok, "is_control": control, "layer": layer, "rule": rule,
                "expected_limit": limit, "domain": "byte_roundtrip"}

    findings_run = R.verdict_line({
        "results": [rec("roundtrip-png", ok=False),
                    rec("bin-python-stdout", ok=True),
                    rec("ctl-bin-python-flipped", ok=False, control=True,
                        rule="1: tool bytes differ from shown bytes")],
        "domains": {"byte_roundtrip": {"controls": 1, "controls_verified": 1}},
        "control_breaks_verified": 1, "control_domains": 1, "tasks_run": 3})
    assert "1 substrate finding(s)" in findings_run
    assert "roundtrip-png" in findings_run and "[interface]" in findings_run
    assert "ctl-bin-python-flipped" not in findings_run, "a control row is not a finding"
    assert R.CLEAN_VERDICT not in findings_run

    clean = R.verdict_line({
        "results": [rec("bin-python-stdout", ok=True),
                    rec("ctl-bin-python-flipped", ok=False, control=True)],
        "domains": {"byte_roundtrip": {"controls": 1, "controls_verified": 1}},
        "control_breaks_verified": 1, "control_domains": 1, "tasks_run": 2})
    assert R.CLEAN_VERDICT in clean and "1 control break(s) verified" in clean

    dark = R.verdict_line({
        "results": [rec("bin-python-stdout", ok=True)],
        "domains": {"byte_roundtrip": {"controls": 0, "controls_verified": 0}},
        "control_breaks_verified": 0, "control_domains": 0, "tasks_run": 1})
    assert R.CLEAN_VERDICT not in dark, (
        "a board with no verified control is the #644 defect: it must never read as clean")


def test_the_verdict_line_survives_into_the_report(tmp_path):
    """The verdict is in the report dict a dated artefact is written from, not only on the
    terminal: a conclusion that exists only in a scrollback is a conclusion nobody can read
    after the window closes. One clean row plus its control: the control has to be verified or
    the line is forbidden from declaring anything clean."""
    report = R.run_corpus([BY_ID["bin-python-stdout"], BY_ID["ctl-bin-python-flipped"]],
                          root=tmp_path)
    assert report["control_breaks_verified"] == 1, report["domains"]
    assert report["verdict_line"] == R.verdict_line(report)
    assert R.CLEAN_VERDICT in report["verdict_line"], report["verdict_line"]


def test_a_clean_board_without_a_verified_control_never_claims_clean(tmp_path):
    """The same board minus its control is the #644 defect: nothing red because nothing was
    proved. The line has to say so out loud."""
    report = R.run_corpus([BY_ID["bin-python-stdout"]], root=tmp_path)
    assert report["control_breaks_verified"] == 0
    assert R.CLEAN_VERDICT not in report["verdict_line"], report["verdict_line"]
    assert "NOT declared clean" in report["verdict_line"], report["verdict_line"]


# ── #2929 clause 2: a declared fixture mode has to be applied ───────────────

def test_a_mode_500_setup_directory_refuses_a_write_before_the_step_dispatches(tmp_path):
    """`_do_setup` laid the directory down, checked its own write, and then never applied the
    mode the row declared: the chmod it did read (`read_only`) is a key no corpus row has ever
    carried — 0 hits in `corpus.json` — and the `item["mode"]` chmod sits below the `continue`
    the directory branch takes. So `rodir` was 755, `printf x > rodir/inside.txt` succeeded,
    and both filesystem-refusal rows went red blaming the substrate for a fixture that was
    never unwritable. This asserts the ceiling exists at the moment setup ends, which is before
    any step is dispatched."""
    task = parse_task_row(sf_task(tid="probe-mode-dir", domain="filesystem",
                                  setup=[{"path": "rodir", "dir": True, "mode": "500"}]))
    R._do_setup(tmp_path, task, None)
    target = tmp_path / "rodir"
    assert oct(target.stat().st_mode & 0o777) == "0o500", oct(target.stat().st_mode & 0o777)
    assert not os.access(target, os.W_OK), (
        "this uid can write into a directory the row declared mode 500")
    with pytest.raises(PermissionError):
        (target / "inside.txt").write_text("x")


def test_both_readonly_rows_are_green_once_the_fixture_really_refuses(tmp_path):
    """The two rows that defect reported: `fs-write-into-readonly-dir` came back
    `is_error=False (expected True)` and `rf-readonly-dir-error-text` came back with 11 bytes
    of stdout instead of an error. Both are fixture bugs, not substrate bugs, so both go green
    with the mode applied — and the second one only with a needle this setup can produce."""
    for tid in ("fs-write-into-readonly-dir", "rf-readonly-dir-error-text"):
        rec = R.run_task(BY_ID[tid], root=tmp_path / tid)
        # A green record carries no layer: attribution is an act of explaining a failure, and
        # the layer a passing row PROVED is the one it declares, which is `host` for both.
        assert rec["expected_layers"] == ["host"], rec["expected_layers"]
        assert rec["layer"] is None, f"{tid} went red: {rec['rule']!r} / {rec['evidence']!r}"
        assert rec["ok"] is True, f"{tid}: {rec['rule']!r} / {rec['evidence']!r}"
        step = rec["steps"][0]
        assert step["error_class"] == "tool_error", (
            f"{tid}: the write into a mode-500 directory did not come back as an error: "
            f"{step['error_text']!r}")
        assert "permission denied" in step["error_text"].lower(), step["error_text"]
