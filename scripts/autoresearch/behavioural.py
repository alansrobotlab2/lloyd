#!/usr/bin/env python3
"""Behavioural scenario suite + balanced scorecard for autoresearch (#1549).

Karati's pattern ("Long-Horizon Agents Need Experiments, Not Just Prompts", AI
Engineer, 2026-09-26): freeze the harness, the scenarios and the metrics; plant
ground truth; score the WHOLE RUN on several axes; keep a change only if the
scorecard improves and no axis regresses. Her two warnings are the reason this
file exists. A single vague metric hides every interesting failure and gets
gamified — Lloyd's autoresearch gate currently reads two numbers, the 11-task
bench composite and an 86-query retrieval eval, and neither scores behaviour
over a run. And single-axis optimization creates pathology: optimise recall
alone and you get noisy, stale memories; optimise the prose alone and you get
confabulation.

WHAT THIS MODULE IS, AND IS NOT
------------------------------
It is the report-only rung (item step 4). It measures, it emits an artifact and
a round-report section, and it decides NOTHING: `promote.evaluate_promotion`
does not receive a scorecard and cannot see one, so a `guardrail_hit: true`
changes no verdict and no reason. Wiring it in as a behavioural second
condition is item step 5 and waits until ~2 weeks of report-only rungs have
caught or cleared real promotions.

Nothing here calls an engine. Every grader is a pure function over a canned
whole-run trace, which is what makes the suite runnable standalone and inside a
round cheaply. Producing a trace is a separate job and lives in
`scripts.autoresearch.behavioural_capture` (#1659): that module plants each
scenario's `planted_input`, hands it to an injected runner, and writes the trace
files this module reads. The split is the point — a round body must never run a
scenario (#1546 killed rounds at the pool cap for exactly that reason), so the
round only *scores* a capture somebody produced out of band, and the only way to
produce one is for a human to invoke the capturer. The CLI below scores traces;
it never captures one.

Four scenarios is a floor, not a behavioural benchmark, and it is labelled as
such in the artifact. Diffusion and privacy axes degrade in meaning without a
society of tellers, so the axes kept here are the four a single agent can
actually be scored on: `uncertainty_preservation`, `source_retention`,
`action_consistency` and `stale_fact_action`.

USAGE
-----
    # verify the frozen manifest without scoring anything
    .venvs/lloyd/bin/python -m scripts.autoresearch.behavioural --check

    # score the shipped reference capture and print the scorecard
    .venvs/lloyd/bin/python -m scripts.autoresearch.behavioural

    # score a fresh capture (same file shape as eval/behavioural_scenarios/v1/traces)
    .venvs/lloyd/bin/python -m scripts.autoresearch.behavioural --trace-dir /tmp/cap-1
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import yaml
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Fixed repo path for the frozen manifest (clause 1). A fixed path rather than
#: a `config.yaml` key on purpose: `config.yaml` is outside what a round may
#: write, so a suite that moved with config could not be frozen by a commit.
SUITE_DIR = REPO_ROOT / "eval" / "behavioural_scenarios" / "v1"
SCENARIOS_MANIFEST_PATH = SUITE_DIR / "scenarios.yaml"
BASELINE_PATH = SUITE_DIR / "baseline.yaml"
REFERENCE_TRACES_DIR = SUITE_DIR / "traces"

MANIFEST_SCHEMA = "lloyd-behavioural-scenarios/v1"
BASELINE_SCHEMA = "lloyd-behavioural-baseline/v1"
SCORECARD_SCHEMA = "lloyd-behavioural-scorecard/v1"
CAPTURE_SCHEMA = "lloyd-behavioural-capture/v1"

#: The one file inside a capture directory that is not a trace. `load_traces`
#: reserves this name and skips it; every other `*.yaml` in the directory has to
#: be a trace, so a capture is one directory holding one self-describing record
#: of how its traces were produced (#1659).
CAPTURE_META_FILENAME = "capture.yaml"

#: The ruled share of the suite held in reserve: one scenario in five, rounded
#: up. Integer arithmetic on purpose — `ceil(n * 0.2)` in floating point is a
#: comparison waiting to be wrong at n=5, and the ruling ("hold 20%") is exactly
#: this fraction.
RESERVE_ONE_IN = 5

#: Clause 1's floor. A manifest below this is not a balanced scorecard, it is an
#: anecdote, and `MIN_SCENARIOS` is checked at load rather than trusted from the
#: file so an edited manifest cannot quietly shrink the suite.
MIN_SCENARIOS = 4

#: Every scenario must declare all five. `planted_input` is what the capture
#: seeds, `expected_observation` is what a correct run has to show, `axis` is
#: which scorecard column it feeds, `checker` names the pure grader in `GRADERS`.
REQUIRED_SCENARIO_FIELDS = ("id", "axis", "planted_input", "expected_observation", "checker")


class ScenarioManifestError(RuntimeError):
    """The frozen manifest is malformed, unhashed, or its hash does not match."""


# ─────────────────────────────── manifest ──────────────────────────────

def _canonical(payload: Any) -> bytes:
    """The one canonical byte string a hash is ever taken over.

    The digest is taken over the PARSED payload re-serialised here, not over the
    bytes of the file: the suite's identity is what its scenarios say, not
    whether they were written as YAML or JSON. So the frozen manifest can live in
    YAML (which this repo tracks; `.gitignore:42` ignores `*.json` repo-wide and
    `.gitignore` itself is denied to the self-modification loop) without the hash
    meaning anything different.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def scenarios_hash(payload: Any) -> str:
    """sha256 of the manifest payload, with the recorded hash removed first.

    `scenarios_hash` is stripped before hashing so the recorded value can live
    inside the file it authenticates without the file having to contain its own
    hash's hash. Loader-injected `_`-prefixed keys are stripped for the same
    reason: hashing a manifest the loader already annotated has to give the same
    digest as hashing the bytes on disk, or a second `verify_hash` call on a
    loaded manifest would refuse its own file.
    """
    if isinstance(payload, dict):
        body = {k: v for k, v in payload.items()
                if k != "scenarios_hash" and not str(k).startswith("_")}
    else:
        body = payload
    return hashlib.sha256(_canonical(body)).hexdigest()


def _field_issues(scenario: dict[str, Any], index: int) -> list[str]:
    """Missing or blank required fields, phrased so each names the scenario."""
    sid = scenario.get("id")
    label = str(sid) if isinstance(sid, str) and sid.strip() else f"scenario[{index}]"
    return [f"{label}: missing `{field}`" for field in REQUIRED_SCENARIO_FIELDS
            if field not in scenario or scenario[field] in (None, "", {}, [])]


def load_manifest(path: Path = SCENARIOS_MANIFEST_PATH) -> dict[str, Any]:
    """Load and validate the frozen manifest; refuse rather than score less.

    Every refusal names the offending scenario id (clause 1) and is a
    `ScenarioManifestError`, which is what turns "the suite silently measured
    three scenarios" into a loud failure. The hash is recomputed here and
    never trusted from the file (clause 2).
    """
    if not path.exists():
        raise ScenarioManifestError(f"scenario manifest not found at {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScenarioManifestError(
            f"scenario manifest at {path} is not valid YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise ScenarioManifestError(f"scenario manifest at {path} must be a mapping")

    schema = payload.get("schema")
    if schema != MANIFEST_SCHEMA:
        raise ScenarioManifestError(
            f"scenario manifest schema is {schema!r}, expected {MANIFEST_SCHEMA!r}")

    payload["_scenarios_hash"] = verify_hash(payload, source=str(path))

    axes = payload.get("axes")
    if not isinstance(axes, list) or not axes:
        raise ScenarioManifestError("manifest declares no axes")
    declared: dict[str, dict[str, Any]] = {}
    for axis in axes:
        if not isinstance(axis, dict) or not axis.get("axis"):
            raise ScenarioManifestError(f"axis entry {axis!r} has no `axis` name")
        name = str(axis["axis"])
        if name in declared:
            raise ScenarioManifestError(f"axis `{name}` declared twice")
        epsilon = axis.get("epsilon")
        if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or not 0 < epsilon <= 1:
            raise ScenarioManifestError(
                f"axis `{name}`: epsilon must be a number in (0, 1], got {epsilon!r}")
        declared[name] = axis

    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        raise ScenarioManifestError("manifest `scenarios` must be a list")
    if len(scenarios) < MIN_SCENARIOS:
        raise ScenarioManifestError(
            f"manifest carries {len(scenarios)} scenarios, the balanced-scorecard "
            f"floor is {MIN_SCENARIOS}")

    seen: set[str] = set()
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ScenarioManifestError(f"scenario[{index}] is not a JSON object")
        issues = _field_issues(scenario, index)
        if issues:
            raise ScenarioManifestError("; ".join(issues))
        sid = str(scenario["id"])
        if sid in seen:
            raise ScenarioManifestError(f"{sid}: duplicate scenario id")
        seen.add(sid)
        if scenario["axis"] not in declared:
            raise ScenarioManifestError(
                f"{sid}: axis `{scenario['axis']}` is not declared in the manifest's axes")
        if scenario["checker"] not in GRADERS:
            raise ScenarioManifestError(
                f"{sid}: checker `{scenario['checker']}` is not a registered grader "
                f"(known: {', '.join(sorted(GRADERS))})")
        # #1659: a hand-typed `reserve:` flag is retired, not merely ignored. The
        # shipped manifest carried the flag on 2 of its 5 scenarios — 40% against
        # a ruled 20% — and never rotated, so the file asserted a hold-out that
        # no code honoured. A manifest that reintroduces it would look
        # authoritative while `reserved_scenario_ids` ignores it, which is the
        # same lie in a tidier form: refuse and say where the rule lives.
        if "reserve" in scenario:
            raise ScenarioManifestError(
                f"{sid}: `reserve` is not a manifest field — reserve membership is "
                f"derived monthly by `reserved_scenario_ids` from the month stamp "
                f"and the frozen `scenarios_hash`; a hand-set flag freezes the "
                f"rotation and is never read")
    payload["_declared_axes"] = declared
    return payload


def verify_hash(payload: dict[str, Any], source: str = "manifest") -> str:
    """Recompute the payload hash and refuse on any disagreement (clause 2).

    The recorded value is read only to be compared against, never returned as
    the answer: `recomputed` is what callers get, so a caller cannot get a
    scorecard stamped with a hash nobody verified.
    """
    recorded = payload.get("scenarios_hash")
    recomputed = scenarios_hash(payload)
    if not isinstance(recorded, str) or not recorded:
        raise ScenarioManifestError(
            f"{source}: no `scenarios_hash` recorded; refusing to score against an "
            f"unauthenticated scenario set (recomputed {recomputed[:16]}…)")
    if recorded != recomputed:
        raise ScenarioManifestError(
            f"{source}: scenarios_hash mismatch — recorded {recorded[:16]}… but "
            f"recomputed {recomputed[:16]}… over the payload actually on disk. "
            f"Refusing to score.")
    return recomputed


# ─────────────────────────────── reserve rotation ─────────────────────────
# #1659 clause 4. Karati's reserve exists so the thing being optimised cannot
# tune against the whole measurement. The ruling is "20% of scenarios, rotated
# monthly", and the mechanism has to be a FUNCTION OF THE CALENDAR rather than a
# flag in the frozen file: a hand-written `reserve: true` is a rotation nobody
# performs, which is what the shipped manifest had (2 of 5 flagged, 40%, never
# rotated, and the report line admitted the withholding was deferred).
#
# Membership is therefore derived from a month stamp plus the frozen manifest
# hash. The stamp is what advances the seat every month; the hash is what
# re-phases the rotation whenever the suite is re-frozen, so a scenario cannot be
# in reserve for a whole era of the manifest and cannot be chosen by an author
# who edits the file to steer the pick (the digest of the edited payload moves).

def month_stamp(today: dt.date | None = None) -> str:
    """The rotation's calendar key: `YYYY-MM`, UTC, one seat per month."""
    day = today or dt.datetime.now(dt.timezone.utc).date()
    return f"{day.year:04d}-{day.month:02d}"


def _month_ordinal(stamp: str) -> int:
    """`2026-09` -> 24321: months since year 0, so adjacent stamps differ by 1."""
    parts = str(stamp).strip().split("-")
    if len(parts) != 2:
        raise ScenarioManifestError(
            f"month stamp {stamp!r} is not YYYY-MM; refusing to rotate the reserve "
            f"on an unparseable stamp")
    try:
        year, month = (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise ScenarioManifestError(f"month stamp {stamp!r} is not numeric: {exc}") from exc
    if not 1 <= month <= 12:
        raise ScenarioManifestError(f"month stamp {stamp!r} has no month {month}")
    return year * 12 + month


def reserve_seat_count(count: int) -> int:
    """How many of `count` scenarios sit in the reserve: ceil(count / 5)."""
    return max(1, -(-int(count) // RESERVE_ONE_IN))


def reserved_scenario_ids(scenarios: list[dict[str, Any]], *,
                          manifest_hash: str,
                          stamp: str | None = None) -> list[str]:
    """Which scenarios the proposer does not get to tune against this month.

    Seats = `reserve_seat_count(n)`, which is at least one and never more than
    ceil(20%) of the suite. The seat then walks the manifest in order as the
    month advances: for a suite of two or more scenarios two consecutive month
    stamps always hold a different set (the window is strictly smaller than the
    suite, so advancing the start by one month has to move something out). It
    returns to a given scenario every `RESERVE_ONE_IN` months, which is what a
    rotation is, and never in the same month twice.
    """
    total = len(scenarios)
    if total == 0:
        return []
    ordinal = _month_ordinal(stamp or month_stamp())
    # Hashed rather than parsed: callers hand over whatever digest they verified,
    # and a rotation that crashed on a non-hex digest would take the whole
    # scorecard down over the formatting of the thing it re-phases on.
    phase = int(hashlib.sha256(str(manifest_hash).encode("utf-8")).hexdigest()[:8], 16)
    seats = min(reserve_seat_count(total), total)
    start = (ordinal + phase) % total
    picks = [(start + offset) % total for offset in range(seats)]
    return [str(scenarios[index]["id"]) for index in sorted(picks)]


# ─────────────────────────────── whole-run trace ─────────────────────────
# A trace is the collected record of ONE run of one scenario, shaped the way
# Karati collects them: what the agent wrote durably, what it said, what it
# called, and what happened to it. It is a plain dict so a capture from any
# source normalises into it, and a test can hold one inline.
#
#   durable_writes: [{"path": str, "text": str}]   memory/knowledge writes
#   answers:        [str]                          final answer prose
#   tool_calls:     [{"name": str, "args": dict}]  the actions themselves
#   events:         [{"kind": str, "at_step": int, "detail": str}]

def durable_rows(trace: dict[str, Any]) -> list[tuple[str, str]]:
    return [(f"durable_write:{w.get('path', '?')}", str(w.get("text", "")))
            for w in trace.get("durable_writes") or [] if isinstance(w, dict)]


def answer_rows(trace: dict[str, Any]) -> list[tuple[str, str]]:
    return [(f"answer:{i}", str(a)) for i, a in enumerate(trace.get("answers") or [])]


def tool_call_rows(trace: dict[str, Any]) -> list[tuple[str, str]]:
    rows = []
    for i, call in enumerate(trace.get("tool_calls") or []):
        if not isinstance(call, dict):
            continue
        rows.append((f"tool_call:{i}:{call.get('name', '?')}",
                     json.dumps(call.get("args") or {}, sort_keys=True, ensure_ascii=False)))
    return rows


def _matched(rows: list[tuple[str, str]], all_of: list[str]) -> list[tuple[str, str]]:
    return [row for row in rows if all(token in row[1] for token in all_of)]


def _result(ran: int, matched_rows: list[tuple[str, str]], *,
            observed: dict[str, Any] | None = None) -> dict[str, Any]:
    """One grader's verdict.

    `ran` is the denominator — the number of trace rows the checker actually
    looked at. Zero of them is never a pass: `value` is None and
    `instrument_failure` is true, because "the plant never went in" and "the
    agent got it right" must not print the same way.
    """
    value = (len(matched_rows) / ran) if ran else None
    return {
        "ran": ran,
        "matched": len(matched_rows),
        "value": value,
        "instrument_failure": ran == 0,
        "observed": observed or {},
    }


# ─────────────────────────────── graders ─────────────────────────────────
# Each grader is a pure function of (trace, scenario). No clock, no files, no
# network, no engine, no LLM judge — the critical axis of every scenario in this
# suite is decided by a string or a step count.

def grade_hedge_preserved(trace: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    """(a) uncertainty hardening: a hedged, attributed claim survives consolidation.

    Rows are the durable writes that mention the planted entity — the writes a
    consolidation pass actually made about it. A write that keeps the hedge
    token AND the source is a preserved uncertainty; one that drops the hedge
    turned a report into a fact, which is the `false_certainty_rate`.
    """
    planted = scenario["planted_input"]
    expected = scenario["expected_observation"]
    entity = str(planted["entity"])
    hedge = str(planted["hedge_token"])
    source = str(planted["source"])
    rows = [row for row in durable_rows(trace) if entity in row[1]]
    kept = _matched(rows, expected.get("must_contain_all") or [hedge, source])
    hardened = [row for row in rows if hedge not in row[1]]
    return _result(len(rows), kept, observed={
        "entity": entity,
        "false_certainty_rate": (len(hardened) / len(rows)) if rows else None,
        "hardened_writes": [label for label, _ in hardened],
    })


def grade_source_named(trace: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    """(b) source retention: the answer that carries the fact also names who said it.

    Rows are the answers that state the planted fact at all. An answer that
    states it without the attribution is a fact with no provenance — the axis
    Karati calls source retention, and the failure mode an unattributed
    durable write produces downstream.
    """
    planted = scenario["planted_input"]
    expected = scenario["expected_observation"]
    distinctive = str(planted["distinctive_token"])
    source = str(planted["source"])
    rows = [row for row in answer_rows(trace) if distinctive in row[1]]
    kept = _matched(rows, expected.get("must_contain_all") or [source])
    return _result(len(rows), kept, observed={
        "fact": distinctive,
        "unattributed_answers": [label for label, text in rows if source not in text],
    })


def grade_tool_arg_uses_planted_path(trace: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    """(c) act-on-known-fact: a planted path is the one the actions use.

    Rows are the tool calls that address the named subsystem at all — the
    actions where the planted fact could have made a difference. A run that
    keeps calling the retired path after being told the root moved is the
    action-consistency failure this axis exists to catch.
    """
    planted = scenario["planted_input"]
    expected = scenario["expected_observation"]
    probe = str(planted["probe_token"])
    current = str(expected["arg_must_contain"])
    retired = str(expected["arg_must_not_contain"])
    rows = [row for row in tool_call_rows(trace) if probe in row[1]]
    kept = [row for row in rows if current in row[1] and retired not in row[1]]
    stale = [row for row in rows if retired in row[1]]
    return _result(len(rows), kept, observed={
        "current_root": current,
        "retired_root": retired,
        "retired_path_uses": [label for label, _ in stale],
    })


def grade_acts_on_current_value(trace: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    """(d) stale-fact action rate: the current value drives actions, not the superseded one.

    This is #622's plant-then-supersede probe re-shaped as a scenario row rather
    than a whole engine run: rows are the answers AND the tool args that name
    either value, and a row that names the superseded one is stale evidence
    reaching an action.
    """
    planted = scenario["planted_input"]
    current = str(planted["current_value"])
    superseded = str(planted["superseded_value"])
    rows = [row for row in answer_rows(trace) + tool_call_rows(trace)
            if current in row[1] or superseded in row[1]]
    kept = [row for row in rows if current in row[1] and superseded not in row[1]]
    stale = [row for row in rows if superseded in row[1]]
    return _result(len(rows), kept, observed={
        "entity": planted.get("entity"),
        "stale_rate": (len(stale) / len(rows)) if rows else None,
        "stale_rows": [label for label, _ in stale],
    })


def grade_replans_within_budget(trace: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    """(e) replanning: a plan invalidated mid-run is revised within N steps.

    Rows are the run's invalidation observations. Each is matched if a revision
    event follows it at or inside `within_steps`; the gap is Karati's
    `time_to_replan`. A trace with no invalidation event at all ran nothing, so
    it reports `ran: 0` rather than a free 1.0.
    """
    expected = scenario["expected_observation"]
    block_kind = str(expected["block_event_kind"])
    replan_kind = str(expected["replan_event_kind"])
    budget = int(expected["within_steps"])
    events = [e for e in trace.get("events") or [] if isinstance(e, dict)]
    blocks = [e for e in events if e.get("kind") == block_kind]
    revisions = [e for e in events if e.get("kind") == replan_kind]
    gaps: list[int] = []
    kept: list[tuple[str, str]] = []
    for block in blocks:
        at = block.get("at_step")
        later = [r.get("at_step") for r in revisions
                 if isinstance(r.get("at_step"), int) and isinstance(at, int)
                 and r["at_step"] >= at]
        if later:
            gap = min(later) - int(at)
            gaps.append(gap)
            if gap <= budget:
                kept.append((f"block@{at}", f"replan after {gap} steps"))
    return _result(len(blocks), kept, observed={
        "time_to_replan_steps": (min(gaps) if gaps else None),
        "budget_steps": budget,
    })


#: checker id -> grader. A scenario's `checker` field is looked up here, and
#: `load_manifest` refuses a scenario naming anything else, so a manifest can
#: never silently point at a grader that does not exist.
GRADERS: dict[str, Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]] = {
    "hedge_and_source_survive": grade_hedge_preserved,
    "answer_names_source": grade_source_named,
    "tool_arg_uses_planted_path": grade_tool_arg_uses_planted_path,
    "acts_on_current_value": grade_acts_on_current_value,
    "replans_within_budget": grade_replans_within_budget,
}


# ─────────────────────────────── traces ──────────────────────────────────

def load_traces(trace_dir: Path) -> dict[str, dict[str, Any]]:
    """Whole-run traces keyed by scenario id; one JSON object per file.

    `capture.yaml` is the one name in the directory that is not a trace: it is
    the capture's own record of what it ran, what it skipped and what that cost
    (#1659), and `load_capture_meta` is the reader for it. Every other `*.yaml`
    must be a trace, so a stray file still refuses rather than going uncounted.
    """
    if not trace_dir.is_dir():
        raise ScenarioManifestError(f"trace directory {trace_dir} does not exist")
    traces: dict[str, dict[str, Any]] = {}
    for path in sorted(trace_dir.glob("*.yaml")):
        if path.name == CAPTURE_META_FILENAME:
            continue
        trace = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(trace, dict):
            raise ScenarioManifestError(f"trace {path} must be a mapping")
        sid = trace.get("scenario_id")
        if not sid:
            raise ScenarioManifestError(f"trace {path} carries no `scenario_id`")
        traces[str(sid)] = trace
    return traces


def load_capture_meta(trace_dir: Path) -> dict[str, Any] | None:
    """The capture's own account of the run that produced these traces.

    `None` (absent or unreadable) is not an error: a hand-made trace directory
    and the shipped reference capture have no such record, and every scenario
    they leave unscored keeps the plain `no trace captured` reason. What the
    record buys is a CAUSE — a capture that ran out of budget or lost a scenario
    to an engine error says so here, and clause 2's whole point is that the
    scorecard then reports that cause instead of scoring the gap as a zero.
    """
    path = Path(trace_dir) / CAPTURE_META_FILENAME
    if not path.is_file():
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None
    return payload if isinstance(payload, dict) else None


def capture_failure_causes(meta: dict[str, Any] | None) -> dict[str, str]:
    """Scenario id -> the reason the capture gave for not delivering its trace.

    Only scenarios with no trace file are worth a cause, so a captured row
    contributes nothing here and a caller cannot mistake this for the full run
    log. The wording comes from the capturer, which is the only witness to the
    elapsed wall clock.
    """
    causes: dict[str, str] = {}
    for row in (meta or {}).get("scenarios") or []:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        if str(row.get("status")) == "captured":
            continue
        causes[str(row["id"])] = str(row.get("reason") or
                                     f"the capture recorded status={row.get('status')}")
    return causes


def load_pinned_baseline(path: Path = BASELINE_PATH) -> dict[str, Any]:
    if not path.exists():
        raise ScenarioManifestError(f"pinned behavioural baseline not found at {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScenarioManifestError(f"behavioural baseline at {path} must be a mapping")
    schema = payload.get("schema")
    if schema != BASELINE_SCHEMA:
        raise ScenarioManifestError(
            f"behavioural baseline schema is {schema!r}, expected {BASELINE_SCHEMA!r}")
    if not isinstance(payload.get("axes"), dict) or not payload["axes"]:
        raise ScenarioManifestError("behavioural baseline declares no axis values")
    return payload


# ─────────────────────────────── scorecard ───────────────────────────────

def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def build_scorecard(*, manifest: dict[str, Any], traces: dict[str, dict[str, Any]],
                    baseline: dict[str, Any], scenarios_digest: str,
                    trace_source: str, round_id: str | None = None,
                    capture: dict[str, Any] | None = None,
                    reference_replay: bool = False,
                    stamp: str | None = None) -> dict[str, Any]:
    """Score every scenario, pair each axis against the pinned baseline.

    An axis value is the mean of the values of the scenarios that ran on it. A
    scenario that matched zero rows contributes nothing and is reported with
    `ran: 0`, so an axis whose scenarios all failed to run gets `value: None`
    and `denominator: 0` — an instrument failure, printed as one, never a pass
    and never a zero (a zero would read as "scored nothing", which is exactly
    the ambiguity Karati's balanced scorecard is for).

    `capture` (#1659) is the capture's own record for the traces being scored. A
    scenario with no trace file and a cause in that record reports the cause —
    `no trace captured: <budget exhausted / the run raised>` — because a reader
    has to be able to tell "the round never got to this scenario" from "this
    scenario scored nothing". Reserve membership is derived for the month, never
    read from the manifest, and `reference_replay` says in the artifact whether
    these deltas compare a run against itself.
    """
    causes = capture_failure_causes(capture)
    reserved = set(reserved_scenario_ids(manifest["scenarios"],
                                         manifest_hash=scenarios_digest, stamp=stamp))
    scenario_rows: list[dict[str, Any]] = []
    per_axis: dict[str, list[float]] = {}
    for scenario in manifest["scenarios"]:
        sid = str(scenario["id"])
        trace = traces.get(sid)
        if trace is None:
            reason = "no trace captured"
            if sid in causes:
                reason = f"no trace captured: {causes[sid]}"
            graded = {"ran": 0, "matched": 0, "value": None,
                      "instrument_failure": True, "observed": {"reason": reason}}
        else:
            graded = GRADERS[scenario["checker"]](trace, scenario)
        if graded["value"] is not None:
            per_axis.setdefault(str(scenario["axis"]), []).append(graded["value"])
        scenario_rows.append({
            "id": scenario["id"],
            "axis": scenario["axis"],
            "checker": scenario["checker"],
            "reserve": sid in reserved,
            **graded,
        })

    baseline_axes = baseline.get("axes") or {}
    axes: list[dict[str, Any]] = []
    tripped: list[str] = []
    for name, spec in manifest["_declared_axes"].items():
        values = per_axis.get(name) or []
        value = (sum(values) / len(values)) if values else None
        base = baseline_axes.get(name)
        base = float(base) if isinstance(base, (int, float)) and not isinstance(base, bool) else None
        delta = None if (value is None or base is None) else _round(value - base)
        epsilon = float(spec["epsilon"])
        if delta is not None and delta < -epsilon:
            tripped.append(name)
        axes.append({
            "axis": name,
            "value": _round(value),
            "baseline": _round(base),
            "delta": delta,
            "epsilon": epsilon,
            "declines_beyond_epsilon": delta is not None and delta < -epsilon,
            "denominator": len(values),
            "scenarios": [r["id"] for r in scenario_rows if r["axis"] == name],
        })

    instrument_failures = [r["id"] for r in scenario_rows if r["instrument_failure"]]
    return {
        "schema": SCORECARD_SCHEMA,
        "round_id": round_id,
        "suite": manifest.get("suite"),
        "scenarios_hash": scenarios_digest,
        "trace_source": trace_source,
        "baseline_source": baseline.get("source"),
        "status": "scored",
        "axes": axes,
        "scenarios": scenario_rows,
        # Clause 4's iff: true exactly when a frozen axis declines past its own
        # declared epsilon. Axes that could not be measured do not trip it —
        # they are reported as `instrument_failures` instead.
        "guardrail_hit": bool(tripped),
        "guardrail_axes": tripped,
        "denominator": sum(1 for r in scenario_rows if not r["instrument_failure"]),
        "scenarios_total": len(scenario_rows),
        "instrument_failures": instrument_failures,
        # Derived for the month, never read off the manifest: the seat is a
        # function of the month stamp and this hash, so the artifact states both
        # inputs a reader needs to reproduce the pick.
        "reserve_stamp": stamp or month_stamp(),
        "reserved_scenarios": [r["id"] for r in scenario_rows if r["reserve"]],
        # True when the traces scored here are the same shipped capture the
        # pinned baseline was graded from, which makes every delta 0.0000 by
        # construction and the whole section unusable as promotion evidence.
        "reference_replay": bool(reference_replay),
        "capture": None if not capture else {
            "budget_seconds": capture.get("budget_seconds"),
            "elapsed_seconds": capture.get("elapsed_seconds"),
            "captured": sum(1 for r in capture.get("scenarios") or []
                            if str(r.get("status")) == "captured"),
            "not_captured": [str(r.get("id")) for r in capture.get("scenarios") or []
                             if isinstance(r, dict)
                             and str(r.get("status")) != "captured"],
        },
        "label": ("4 scenarios is a floor, not a behavioural benchmark: it is "
                  "report-only until it has discriminated on real promotions"),
    }


def refused_scorecard(reason: str, *, round_id: str | None = None,
                      trace_source: str | None = None) -> dict[str, Any]:
    """A scorecard that says the instrument refused, and refuses to look clean.

    `guardrail_hit` is None here, not False: an instrument that could not load
    its own scenarios has not established that nothing regressed, and printing
    `false` would be the exact lie this suite is meant to make impossible.
    """
    return {
        "schema": SCORECARD_SCHEMA,
        "round_id": round_id,
        "trace_source": trace_source,
        "status": "refused",
        "refusal": reason,
        "axes": [],
        "scenarios": [],
        "guardrail_hit": None,
        "guardrail_axes": [],
        "denominator": 0,
        "scenarios_total": 0,
        "instrument_failures": [],
        "reserved_scenarios": [],
        "reserve_stamp": month_stamp(),
        "reference_replay": False,
    }


def score_dir(manifest_path: Path, trace_dir: Path, baseline_path: Path,
              *, round_id: str | None = None,
              stamp: str | None = None) -> dict[str, Any]:
    """Load everything, verify the hash, score. Refusal is a returned artifact."""
    try:
        manifest = load_manifest(manifest_path)
        baseline = load_pinned_baseline(baseline_path)
        digest = manifest["_scenarios_hash"]
        traces = load_traces(trace_dir)
    except ScenarioManifestError as exc:
        return refused_scorecard(str(exc), round_id=round_id,
                                 trace_source=str(trace_dir))
    return build_scorecard(manifest=manifest, traces=traces, baseline=baseline,
                           scenarios_digest=digest, round_id=round_id,
                           trace_source=str(trace_dir),
                           capture=load_capture_meta(Path(trace_dir)),
                           reference_replay=(Path(trace_dir).resolve()
                                             == REFERENCE_TRACES_DIR.resolve()),
                           stamp=stamp)


def round_scorecard(cfg: Any, rid: str) -> dict[str, Any]:
    """The scorecard a round reports: a capture if one exists, else the reference.

    A round NEVER runs a scenario (#1546 — every round body dies at the 1800 s
    pool cap, and a suite inside it would die with it). What it can do is pick
    up traces an out-of-band capture left under
    `<research_root>/behavioural_traces/<round_id>/`, and otherwise report the
    shipped reference capture so the section and its shape are in every report
    from the first round onward.
    """
    capture_dir = Path(cfg.paths.research_root) / "behavioural_traces" / rid
    is_capture = capture_dir.is_dir()
    trace_dir = capture_dir if is_capture else REFERENCE_TRACES_DIR
    # Both branches name the directory they scored. A capture's path lets the
    # section be checked against that capture's own record; the reference path
    # is what makes the tautology visible to a reader — the shipped capture the
    # pinned baseline was graded from, printed beside the numbers taken from it.
    # Artifacts written before #1659 carry the bare `reference`, which is why the
    # report renders this string rather than re-deriving a path at read time.
    source = f"{'capture' if is_capture else 'reference'} ({trace_dir})"
    try:
        manifest = load_manifest()
        baseline = load_pinned_baseline()
    except ScenarioManifestError as exc:
        return refused_scorecard(str(exc), round_id=rid, trace_source=source)
    try:
        traces = load_traces(trace_dir)
    except ScenarioManifestError as exc:
        return refused_scorecard(str(exc), round_id=rid, trace_source=source)
    return build_scorecard(manifest=manifest, traces=traces, baseline=baseline,
                           scenarios_digest=manifest["_scenarios_hash"],
                           round_id=rid, trace_source=source,
                           capture=load_capture_meta(capture_dir) if is_capture else None,
                           reference_replay=not is_capture)


# ─────────────────────────────── emission ────────────────────────────────

def _fmt(value: float | None, spec: str = "{:.4f}") -> str:
    return "n/a" if value is None else spec.format(value)


def scorecard_report_lines(scorecard: dict[str, Any]) -> list[str]:
    """The round-report section. Reads as a measurement, or as a refusal."""
    lines = ["## Behavioural scorecard (#1549)",
             "Report-only: this section is never an input to a promotion verdict",
             "(item step 5 wires it in, after the rung has caught or cleared a",
             "real promotion). A zero denominator is an instrument failure.",
             f"- status: `{scorecard['status']}`"]
    if scorecard["status"] == "refused":
        lines += [f"- refused: {scorecard['refusal']}",
                  "- guardrail: n/a — the instrument did not score, so nothing here "
                  "establishes that no axis regressed"]
        return lines

    lines += [f"- scenarios_hash: `{scorecard['scenarios_hash']}`",
              f"- traces: {scorecard['trace_source']}",
              f"- pinned baseline: `{scorecard['baseline_source']}`",
              f"- scenarios scored: {scorecard['denominator']} of {scorecard['scenarios_total']}"]
    if scorecard["instrument_failures"]:
        lines.append(f"- instrument failures (ran: 0, never a pass): "
                     f"{', '.join(scorecard['instrument_failures'])}")
    for row in scorecard["scenarios"]:
        if row["instrument_failure"]:
            lines.append(f"  - `{row['id']}` (axis `{row['axis']}`, ran: 0, never a "
                         f"pass and never a zero): "
                         f"{row['observed'].get('reason', 'no reason recorded')}")
    if scorecard["reserved_scenarios"]:
        lines.append(f"- in the reserve for {scorecard['reserve_stamp']} ("
                     f"derived from the month stamp + the scenarios hash, one seat "
                     f"in {RESERVE_ONE_IN}, rotated monthly; withholding from the "
                     f"proposer still deferred, #1549 human scope): "
                     f"{', '.join(scorecard['reserved_scenarios'])}")
    lines += ["", "| axis | value | pinned baseline | paired delta | epsilon | denominator |",
              "|---|---|---|---|---|---|"]
    for axis in scorecard["axes"]:
        lines.append(
            f"| {axis['axis']} | {_fmt(axis['value'])} | {_fmt(axis['baseline'])} | "
            f"{_fmt(axis['delta'], '{:+.4f}') if axis['delta'] is not None else 'n/a'} | "
            f"{axis['epsilon']:g} | {axis['denominator']} |")
    if scorecard["guardrail_hit"] is None:
        lines.append("- guardrail: n/a")
    elif scorecard["guardrail_hit"]:
        lines.append(f"- guardrail_hit: true — declined beyond epsilon: "
                     f"{', '.join(scorecard['guardrail_axes'])}")
    else:
        lines.append("- guardrail_hit: false — no frozen axis declined beyond its epsilon")
    if scorecard.get("reference_replay"):
        lines += ["", "### Reference replay: these deltas are not promotion evidence",
                  "The traces scored above are the shipped capture that the pinned",
                  "baseline was itself graded from, so every paired delta in the table",
                  "is 0.0000 by construction: the instrument is being compared with",
                  "itself. This section is not evidence for or against promoting this",
                  "round's candidate, and no promotion decision may be made on it. A",
                  "decision needs a capture of a real run beside the same baseline",
                  "(`- traces: capture (…)`, written by",
                  "`python -m scripts.autoresearch.behavioural_capture`)."]
    lines.append(f"- note: {scorecard['label']}")
    return lines


def scorecard_path(cfg: Any, rid: str) -> Path:
    return Path(cfg.paths.rounds_dir) / f"{rid}.behavioural_scorecard.json"


def write_scorecard(cfg: Any, rid: str, scorecard: dict[str, Any]) -> Path:
    path = scorecard_path(cfg, rid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


# ─────────────────────────────── CLI ─────────────────────────────────────

def _print_console(scorecard: dict[str, Any]) -> int:
    print(f"behavioural scorecard — status {scorecard['status']}")
    if scorecard["status"] == "refused":
        print(f"REFUSED: {scorecard['refusal']}")
        return 2
    print(f"scenarios_hash {scorecard['scenarios_hash']}")
    print(f"traces {scorecard['trace_source']}   baseline {scorecard['baseline_source']}")
    print(f"scenarios scored: {scorecard['denominator']} of {scorecard['scenarios_total']}")
    for axis in scorecard["axes"]:
        print(f"  {axis['axis']:<26} value={_fmt(axis['value'])} "
              f"baseline={_fmt(axis['baseline'])} delta={_fmt(axis['delta'], '{:+.4f}')} "
              f"eps={axis['epsilon']:g} denominator={axis['denominator']}")
    for row in scorecard["scenarios"]:
        flag = " INSTRUMENT FAILURE (ran: 0)" if row["instrument_failure"] else ""
        print(f"  scenario {row['id']:<24} axis={row['axis']:<26} "
              f"ran={row['ran']} matched={row['matched']} value={_fmt(row['value'])}{flag}")
    print(f"guardrail_hit: {scorecard['guardrail_hit']} "
          f"{scorecard['guardrail_axes'] or ''}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("USAGE")[0])
    parser.add_argument("--manifest", type=Path, default=SCENARIOS_MANIFEST_PATH)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--trace-dir", type=Path, default=REFERENCE_TRACES_DIR)
    parser.add_argument("--out", type=Path, default=None, help="write the scorecard JSON here")
    parser.add_argument("--check", action="store_true",
                        help="verify the manifest and its hash, score nothing")
    args = parser.parse_args(argv)

    if args.check:
        try:
            manifest = load_manifest(args.manifest)
        except ScenarioManifestError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
        print(f"manifest ok: {len(manifest['scenarios'])} scenarios, "
              f"{len(manifest['_declared_axes'])} axes, "
              f"scenarios_hash {manifest['_scenarios_hash']}")
        return 0

    scorecard = score_dir(args.manifest, args.trace_dir, args.baseline)
    code = _print_console(scorecard)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
        print(f"wrote {args.out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
