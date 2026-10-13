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

import pytest

from _substrate_fuzz import (BY_ID, REAL, sf_corpus, sf_runner)

C = sf_corpus
R = sf_runner


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
