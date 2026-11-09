#!/usr/bin/env python3
"""#3747 (#3419 step (1)) — repeat the shipped summariser over ONE fixed fold fixture.

Why this exists. Two persistence windows (2026-10-19, 2026-10-26) came in with mean
`summary_chars` around 2700 and lost most of their benign markers, and #3200's owed
entry says no edit to `app/compaction.py` happens until a cause is named with a proving
command. A diff cannot name one: over the cited five windows the emitting code is
unchanged since `ec468e3f`, the input shape is identical (`folds=1`,
`arrival_rows`/`covered_rows` ~= 2, `summary_model='primary'` in every window), and the
corpus carries no engine witness — the last row's keys name `summary_model: "primary"`
and nothing naming a build, a version or an mtime. What is decidable on this box is
narrower, and it is what this run answers:

    Is the ~2700 low mode REACHABLE at all from one fixed 2-row fold, on the current
    build, at shipped decoding settings, with the microcompact reduction sidecar OFF?

An engine-build-change explanation for October predicts it is NOT reachable: the same
fixed input on the current build should land in the 2997/3066/3056 band every time. This
run measures the distribution and says which way it fell. It deliberately does NOT
adjudicate summariser-prompt-path against engine decoding: both outcomes of the run use
the SAME prompt, the SAME fixture and the SAME decoding, so no arm varies the prompt and
no result of this run can separate them. The artifact therefore carries
`cause_assigned: null`, and `tests/test_summary_length_repetition_run.py` pins that.

The two settled things this run rests on, cited rather than re-derived:

* The 2026-11-02 window is NOT poolable with the cited five and its 2694 mean is not a
  sixth data point on this question. Ruled by #3419's owed entry at 2026-11-03T23:47:40Z
  (session `20261103_154037_owedcheck_69a9`): #3641 flipped
  `microcompact_reduction_sidecar` to `True` in `1f6bf5c7` at 2026-10-31T14:44:26Z —
  `app/harness/options.py:300` is `microcompact_reduction_sidecar: bool = True` — about
  20 h before that fire, and the eval harness never overrides the knob
  (`git grep -n microcompact -- eval/` hits only a baseline JSON under
  `eval/baselines/`), so the 11-02
  fire ran knob-**on** while this run is knob-**off** by design. Every number quoted
  beside this one has to name which regime it is from.
* 2700 is a window-mean predictor and never a per-row tripwire. The cited five show 12 of
  12 zero-survival rows below it, but the 2026-11-02 window carries a zero-survival row
  at 2762 chars AT/above it, and the healthy 2026-10-14 window carries one at 3027. The
  bound is used here only as the decision edge of the reachability rule below, and that
  usage is a rule this module states, not a finding about any single row.

The verdict rule, stated because the artifact's label is only as honest as this line:

    REACHABLE      if at least one draw's `summary_chars` < SUMMARY_CHAR_BOUND (2700)
    NOT_REACHABLE  if every draw lands at/above it

Two legs per draw, because the shipped stack has two halves and only one of them is the
measurement:

1. **The fold** — `eval/run_injection_canary.py::force_summary_pass`, which is how every
   corpus row whose band this run is read against was produced: it writes the fixture as
   a session file and drives `app/compaction_state.py::manual_compact` with
   `forced_pass_cfg()` (`keep_recent_turns: 0`, `max_folds: 1`). `manual_compact` reaches
   the same `fold()` and the same `summary_message` row as the turn-start path at
   `app/compaction.py:497`, with `cfg["summary_model"]` from
   `app/compaction.py:865`'s sibling call site, and the shipped decoding is
   `app/compaction_llm.py`'s own: `temperature=0.3`, `max_tokens=8000`. The measurement
   is `len(render_summary(record))` — `summary_chars`, the corpus's own field. This fold
   never reads the sidecar knob (`app/compaction_state.py` has no reference to it), which
   is the half `tests/test_compaction.py` pins as the inert-when-off direction.
2. **The knob-off read** — `app/compaction.py::load_and_compact_session` with
   `microcompact_sidecar=False` passed EXPLICITLY, on the folded session. The knob is
   shipped `True` since `1f6bf5c7`, so an omitted argument would silently measure the
   other regime; passing `False` is what makes "knob off" a fact about the run rather
   than a hope. The result's `microcompact_sidecar` / `microcompact_reduced` are recorded
   per draw as the witness that the read happened under the reverted setting.

Nothing here writes a corpus row. The run never calls `append_rows`; on top of that it
redirects `LLOYD_CANARY_ROWS` to a scratch path for its whole process (the read side
merges the tracked corpus, so the bands still come from it), and it records the tracked
corpus file's sha256 and `git status --porcelain` before and after, both in the artifact.
`tests/test_summary_length_repetition_run.py` re-checks that pair.

    .venvs/lloyd/bin/python eval/summary_length_repetition_run.py                 # 24 draws
    .venvs/lloyd/bin/python eval/summary_length_repetition_run.py --write-fixture # rebuild the fixture
    .venvs/lloyd/bin/python eval/summary_length_repetition_run.py --draws 10 --out /tmp/s.jsonl
    #   `--draws` under MIN_DRAWS (10) is refused outright, so the smallest legal run is
    #   10 and it has to be told to write outside eval/measurements/ to be a smoke test:
    #   the committed artifact is one file per date and must not be spent on a smoke run
Tests: .venvs/lloyd/bin/python -m pytest tests/test_summary_length_repetition_run.py
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
for _p in (str(EVAL_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import canary_rows_guard as GUARD  # noqa: E402
import run_injection_canary as CANARY  # noqa: E402
from summary_length_attribution_probe import (  # noqa: E402
    SUMMARY_CHAR_BOUND,
    measure,
)

#: The one committed input every draw folds. Its shape is the corpus's own
#: persistence-control shape — 2 arrival rows, one turn, all of them covered by one fold
#: — so the distribution is comparable to the window means and not to some other input.
#:
#: YAML, not JSON, for one reason that is easy to miss and expensive to get wrong:
#: `.gitignore:42` is a bare `*.json`, so a JSON fixture is an UNTRACKED file no matter
#: where in `eval/` it sits, and `.gitignore` is a gate-denied path a round may not edit
#: to un-ignore its own artifact. Every eval that does ship a JSON artifact paid for it
#: with an explicit `!eval/...` rule (the six of them are lines 59-82). YAML is ignored by
#: nothing here and is already this directory's committed-input format
#: (`eval/counterfactual_perturbations.yaml`, `eval/vault_recall_queries.yaml`).
#:
#: A TOP-LEVEL file, not `eval/fixtures/…`, for a second reason: `eval/*` directories are
#: inventoried as scored arms by `tests/test_architecture_coverage_doc_claims.py::
#: test_measurement_doc_names_exactly_the_eval_arms_that_exist` (walk at :194, `NON_ARMS`
#: = baselines/measurements/__pycache__), so a new `eval/fixtures/` directory is a claim
#: that `architecture/measurement.md` documents a new scored arm — which it is not, it
#: holds one input. The name is also spoken for: the autoresearch bench builds each trial's
#: fixtures under `<data-root>/eval/fixtures/` (`scripts/autoresearch/bench_fixture.py:264`).
FIXTURE_PATH = EVAL_DIR / "summary_length_repetition_fixture.yaml"

#: Where the dated artifact goes. One file per run date, so a re-run never rewrites the
#: measurement the architecture page cites; the test reads the newest and demands the
#: page match it. `.jsonl` for the same reason as the fixture above, and it is the
#: convention of the corpus this run is read against:
#: `eval/measurements/injection-canary/rows.jsonl` is a tracked file of JSON records.
#: This file holds exactly one record, written as a single JSON line, so it is valid both
#: as JSON (`json.loads` reads it whole) and as one line of JSON Lines.
ARTIFACT_DIR = EVAL_DIR / "measurements" / "summary-length-repetition"

#: Sessions are written here, never under `SESSIONS_DIR` — the same reason the canary
#: keeps its arrival transcripts under /tmp (#2397): a synthetic session in the live
#: store is a session the retention sweep and the uptake corpus will both find.
SESSION_HOME = Path("/tmp/lloyd-summary-repetition")

#: The minimum a distribution needs to be a distribution rather than an anecdote — what
#: the item asks for, and the floor below which `run()` refuses.
MIN_DRAWS = 10

#: Twice the floor, because of what the first two runs of this module measured: twelve
#: folds of the SAME fixture bytes came back with means of 2463 and 2865 chars, so the
#: draw-to-draw spread is wider than the gap between the two window means this run is read
#: against. The verdict rule survives that (both runs put draws under 2700), but a
#: published median computed over 12 of these would move by a few hundred chars on a
#: re-run, and a number that moves that much has to be measured on more of them.
DEFAULT_DRAWS = 24

#: The fixture's turn id, matching `transcript_rows(turn_id=…)` in the runner that wrote
#: the corpus rows it is compared against.
FIXTURE_TURN_ID = "t001"

#: The corpus row whose arrival reply the fixture carries. Fixed by session id, not by
#: position, so a later append to the tracked corpus cannot move the fixture under it.
FIXTURE_SOURCE_SESSION_ID = "20261014_023446_bench_7a4c"

#: The five windows #3419 cites, through its last-cited fire. Everything on the file
#: after this date is a different regime (see the module docstring) and is excluded from
#: the bands this run is read against.
CITED_THROUGH = "2026-10-26"

#: The two labels the verdict field may hold. Nothing else, and neither of them names a
#: cause: see the module docstring's "what this run cannot decide".
VERDICT_REACHABLE = "low_mode_reachable"
VERDICT_NOT_REACHABLE = "low_mode_not_reachable"

#: The field the artifact sets to null on purpose: the one question this run cannot
#: answer, recorded as unanswered so a later reader cannot infer that it was answered.
UNANSWERED_CAUSE_QUESTION = (
    "summariser prompt path vs engine decoding for the 2026-10-19 / 2026-10-26 windows"
)


# ── the fixture ───────────────────────────────────────────────────────────


def _scenario(key: str = "persistence-control-handover") -> dict:
    """The committed scenario the fixture's user row is rendered from."""
    for sc in CANARY.load_scenarios():
        if sc.get("key") == key:
            return sc
    raise KeyError(f"scenario {key!r} is not in {CANARY.SCENARIOS_PATH}")


def _source_row() -> dict:
    """The tracked corpus row the fixture's assistant row is taken from, by session id."""
    rows = CANARY.report_rows()
    hits = [r for r in rows if r.get("session_id") == FIXTURE_SOURCE_SESSION_ID]
    if len(hits) != 1:
        raise ValueError(f"{FIXTURE_SOURCE_SESSION_ID!r} matched {len(hits)} corpus rows; "
                         "the fixture's source row has to be exactly one")
    return hits[0]


def build_fixture_rows() -> list[dict]:
    """The two arrival rows, built by the shipped row builder from committed sources.

    User row: `render_task()` of the control scenario, which for a `user-inline` medium
    interpolates the corpus payload document verbatim — the same text the 2026-10-13/14/15
    fires put in front of the summariser. Assistant row: the `final_text` the control's own
    arrival turn answered with on 2026-10-14, taken from the tracked corpus row named by
    `FIXTURE_SOURCE_SESSION_ID`. Both go through `CANARY.transcript_rows()`, the function
    that wrote those rows into the sessions the corpus rows were folded from, so the
    fixture is not a hand-rolled approximation of an arrival turn.
    """
    sc = _scenario()
    scratch = str(SESSION_HOME / FIXTURE_TURN_ID)
    task = CANARY.render_task(sc, base=str(SESSION_HOME), scratch=scratch)
    reply = (_source_row().get("final_text") or "").strip()
    if not reply:
        raise ValueError(f"{FIXTURE_SOURCE_SESSION_ID}: empty final_text — nothing to "
                         "carry into the fixture's assistant row")
    return CANARY.transcript_rows(turn_id=FIXTURE_TURN_ID, user_text=task,
                                  events=[{"type": "assistant_message", "text": reply}])


def write_fixture(dest: Path | None = None) -> Path:
    """Commit-ready YAML for the fixed fixture: the rows, and why those rows."""
    import yaml

    rows = build_fixture_rows()
    sc, src = _scenario(), _source_row()
    doc = {
        "schema": 1,
        "purpose": ("#3747: the one fixed input every draw of "
                    "eval/summary_length_repetition_run.py folds, so the summary_chars "
                    "distribution is a property of the summariser and not of the input."),
        "turn_id": FIXTURE_TURN_ID,
        "scenario_key": sc.get("key"),
        "payload_file": str(sc.get("file")),
        "survival_markers": list(sc.get("survival_markers") or []),
        "source_row": {
            "session_id": src.get("session_id"), "ts": src.get("ts"),
            "key": src.get("key"), "rep": src.get("rep"),
            "summary_chars": src.get("summary_chars"),
            "note": ("assistant row content is this row's final_text; user row is the "
                     "scenario task re-rendered from scenarios.yaml plus its corpus file"),
        },
        "rows": rows,
    }
    out = dest or FIXTURE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100),
                   encoding="utf-8")
    return out


def load_fixture(path: Path | None = None) -> dict:
    """The committed fixture document, refusing one whose shape is not the corpus shape."""
    import yaml

    p = path or FIXTURE_PATH
    doc = yaml.safe_load(Path(p).read_text(encoding="utf-8"))
    rows = doc.get("rows") or []
    if len(rows) != 2:
        raise ValueError(f"{p}: fixture holds {len(rows)} arrival rows, not the 2 the "
                         "corpus control windows fold")
    if [r.get("role") for r in rows] != ["user", "assistant"]:
        raise ValueError(f"{p}: fixture roles are {[r.get('role') for r in rows]}, "
                         "not the user+assistant arrival turn the control windows had")
    return doc


def fixture_sha256(path: Path | None = None) -> str:
    """The bytes the run was measured over, so the artifact can be tied to the fixture."""
    return hashlib.sha256(Path(path or FIXTURE_PATH).read_bytes()).hexdigest()


def committed_artifacts() -> list[Path]:
    """Every dated artifact on the tree, oldest first. The reader takes the newest."""
    return sorted(ARTIFACT_DIR.glob("repetition-*.jsonl"))


def read_artifact(path: Path | None = None) -> dict:
    """One committed artifact as a dict — the newest on the tree unless one is named.

    Lives here rather than in the test file because the architecture page, the test that
    ties the page to the measurement and anything else that quotes a min/median/max have
    to read the SAME file through the SAME reader, or the page and its guard can agree
    with each other while both disagree with the run.
    """
    if path is None:
        found = committed_artifacts()
        if not found:
            raise FileNotFoundError(
                f"no dated artifact under {ARTIFACT_DIR} — the measurement has not been "
                "produced on this tree, and a page or a test quoting one is quoting "
                "nothing")
        path = found[-1]
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text)


def load_fixture_rows(path: Path | None = None) -> list[dict]:
    """Just the fixture's arrival rows."""
    return load_fixture(path)["rows"]


def shipped_decoding(summary_model: str) -> dict:
    """The decoding this run is required to use, READ OFF THE SHIPPED CALLABLES.

    "Shipped decoding settings" is the clause, and a hand-typed `0.3` is not a reading of
    it: the values come out of `app/compaction_llm.py`'s own signatures, which is the
    mechanism `app/harness/options.py::from_legacy_config` uses to avoid the same mistake.
    The fold path calls `summarize_incremental`, which forwards to
    `_post_chat_completion` passing neither `temperature` nor `max_output_tokens`
    (`app/compaction_llm.py:478`), so the defaults of those two signatures ARE the shipped
    settings, and if either moves the artifact moves with it and the test that compares
    them stays green for the right reason.
    """
    import dataclasses
    import inspect

    import app.compaction_llm as CL
    from app.harness.options import RunOptions

    post = inspect.signature(CL._post_chat_completion).parameters
    fold = inspect.signature(CL.summarize_incremental).parameters
    sidecar = next(f for f in dataclasses.fields(RunOptions)
                   if f.name == "microcompact_reduction_sidecar")
    return {
        "source": ("read from app.compaction_llm's own signatures by inspect.signature, "
                   "not typed into this file"),
        "temperature": post["temperature"].default,
        "max_tokens": fold["max_output_tokens"].default,
        # `_post_chat_completion` takes no timeout of its own; the fold path's own
        # callable does, and that is the one this run goes through.
        "timeout_seconds": fold["timeout_seconds"].default,
        "summary_model": summary_model,
        "cfg_key": "compaction.summary_model (config.yaml)",
        "microcompact_reduction_sidecar_shipped_default": (
            sidecar.default if sidecar.default is not dataclasses.MISSING else None),
    }


# ── one draw ──────────────────────────────────────────────────────────────


def _corpus_sha256_and_porcelain() -> dict:
    """The tracked corpus's bytes and its git dirtiness — the no-append witness."""
    path = CANARY.ROWS_PATH
    try:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        sha = f"unreadable: {exc}"
    try:
        porcelain = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--",
             str(path.relative_to(REPO_ROOT))],
            capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        porcelain = f"unavailable: {exc}"
    return {"path": str(path), "sha256": sha, "git_status_porcelain": porcelain}


def corpus_grew_only_by_appending(recorded_sha256: str,
                                  live: Path | None = None) -> dict:
    """Is the corpus a run recorded still a PREFIX of the tracked corpus? (#4000).

    `_corpus_sha256_and_porcelain` writes a sha of the whole tracked file into the
    artifact, and the node that re-reads that witness used to ask the live bytes to
    EQUAL it. That question has a shelf life, because the corpus is append-only and
    not static: `lloyd-persistence-window.service` appends its arm's rows every
    Monday, and on 2026-11-09 it appended 30 of them — so the 2026-11-04 artifact's
    sha stopped naming the file's end, and a witness about a run that wrote nothing
    read as if the run had corrupted the store.

    What the witness was ever for is the half that survives growth: the rows that were
    there when the run measured are still there, unmodified, in the same order. Git can
    answer that from the recorded sha alone — the sha is the hash of a historical blob
    of this same path, and a file that only grew still begins with those bytes. A
    rewrite, a truncation, or a row edited in the middle makes `prefix_of_live` false,
    which the equality check could not distinguish from a healthy append anyway.

    `live` exists so the guard is falsifiable: a caller may point it at a mutated copy
    of the corpus and watch this say NO. The git half always runs over the tracked
    corpus's own path, because that is the store the recorded sha came from.
    """
    path = live or CANARY.ROWS_PATH
    rel = str(CANARY.ROWS_PATH.relative_to(REPO_ROOT))
    out = {"recorded_sha256": recorded_sha256, "path": str(path), "live_sha256": None,
           "live_bytes": None, "recorded_bytes": None, "found_at_commit": None,
           "commits_searched": 0, "prefix_of_live": False, "reason": ""}
    try:
        live_bytes = path.read_bytes()
    except OSError as exc:
        out["reason"] = f"the corpus could not be read: {exc}"
        return out
    out["live_sha256"] = hashlib.sha256(live_bytes).hexdigest()
    out["live_bytes"] = len(live_bytes)
    if out["live_sha256"] == recorded_sha256:
        out.update(found_at_commit="HEAD (the live bytes themselves)",
                   recorded_bytes=len(live_bytes), prefix_of_live=True,
                   reason="unchanged since the run was measured")
        return out
    log = subprocess.run(["git", "-C", str(REPO_ROOT), "log", "--format=%H", "--", rel],
                         capture_output=True, text=True, timeout=60)
    commits = log.stdout.split()
    out["commits_searched"] = len(commits)
    for sha in commits:
        blob = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{sha}:{rel}"],
                              capture_output=True, timeout=60).stdout
        if hashlib.sha256(blob).hexdigest() == recorded_sha256:
            out.update(found_at_commit=sha, recorded_bytes=len(blob),
                       prefix_of_live=live_bytes.startswith(blob))
            out["reason"] = ("the rows the run measured are a prefix of the corpus today"
                             if out["prefix_of_live"] else
                             "the bytes recorded are NOT a prefix of the corpus today: "
                             "history was rewritten, truncated, or a row edited in place")
            return out
    out["reason"] = (f"no commit touching {rel} holds a blob with sha256 "
                     f"{recorded_sha256} ({len(commits)} commit(s) searched): the bytes "
                     "the run recorded are not recoverable from this tree's history")
    return out


def cited_bands_differ(artifact_bands: dict, live_bands: dict) -> list[str]:
    """Which CITED figures moved between a run and now — the growth exception stated.

    `corpus_bands` publishes the cited healthy/flagged windows and, beside them,
    `excluded_after_cited_scope`: every window the corpus now carries that falls after
    `cited_through`. Comparing the two dicts whole therefore pinned the corpus against
    ever growing — a new fire after the cut lands in that list by definition, and the
    2026-11-09 persistence window did exactly that, going red on figures nobody cited.

    So the rule this states: every figure INSIDE the cited scope must be identical, and
    the recorded exclusion list must still be an ordered prefix of the live one. A cited
    mean that moved, a window that appeared or vanished inside the scope, or a recorded
    exclusion that disappeared from the live list all come back named here.
    """
    bad = [k for k in artifact_bands
           if k != "excluded_after_cited_scope"
           and artifact_bands.get(k) != live_bands.get(k)]
    rec, now = (artifact_bands.get("excluded_after_cited_scope") or [],
                live_bands.get("excluded_after_cited_scope") or [])
    if now[:len(rec)] != rec:
        bad.append(f"excluded_after_cited_scope (recorded {rec} is not an ordered "
                   f"prefix of the live {now})")
    for w in now[len(rec):]:
        if not w > artifact_bands["cited_through"]:
            bad.append(f"excluded_after_cited_scope gained {w!r}, which is not after "
                       f"cited_through {artifact_bands['cited_through']!r}")
    return bad


def _engine_witness(summary_model: str) -> dict:
    """What build answered, as far as this box can say. The corpus rows record only
    `summary_model: "primary"`; that absence is why October cannot be attributed, so the
    one measurement that is being taken now records what it can."""
    import app.compaction_llm as CL

    base_url, model_name = CL._summary_endpoint(summary_model)
    version = None
    try:
        import httpx

        with httpx.Client(timeout=5.0) as client:
            version = (client.get(f"{base_url}/version").json() or {}).get("version")
    except Exception:  # noqa: BLE401 — a witness that is missing says null, not a failure
        version = None
    return {"summary_model": summary_model, "base_url": base_url,
            "engine_model_name": model_name, "engine_version": version}


async def draw_once(rows: list[dict], *, summary_model: str,
                    seq: int) -> dict:
    """One fold of the fixed fixture plus the knob-off read of what it produced.

    Leg 1 is the shipped fold path (`force_summary_pass` -> `manual_compact`), leg 2 is
    `load_and_compact_session(..., microcompact_sidecar=False)`. Both are recorded; only
    leg 1's `summary_chars` is the measurement, and leg 2 exists so that "the knob was
    off" is a witnessed fact about this draw rather than an inference from a default.
    """
    import app.compaction as COMPACT

    session_id = f"repetition-{seq}"
    path = CANARY.write_arrival_transcript(session_id, rows, home=SESSION_HOME)
    passed = await CANARY.force_summary_pass(path, summary_model=summary_model)
    summary = passed["summary"]
    markers = _scenario().get("survival_markers") or []
    benign = CANARY.survival_split(markers, summary)
    read = await COMPACT.load_and_compact_session(
        path, model=summary_model, microcompact_sidecar=False)
    return {
        "seq": seq,
        "summary_chars": len(summary),
        "folds": passed["folds"],
        "arrival_rows": passed["arrival_rows"],
        "covered_rows": passed["covered_rows"],
        "fully_covered": passed["fully_covered"],
        "pass_fired": passed["fired"],
        "pass_error": passed["error"] or None,
        "summary_sha256": hashlib.sha256(summary.encode()).hexdigest(),
        "benign_survival": benign,
        # The knob-off witness, straight off the shipped read path's own result fields.
        "read_microcompact_sidecar": bool(read.get("microcompact_sidecar")),
        "read_microcompact_reduced": int(read.get("microcompact_reduced") or 0),
        "read_summarize_outcome": str(read.get("summarize_outcome") or ""),
    }


def distribution(draws: list[dict], bands: dict | None = None) -> dict:
    """min/median/max over the draws, plus both counts the verdict and the page need.

    `below_bound` is the verdict rule's count. `at_or_above_lowest_healthy_mean` is the
    other half of what a repeated fold of ONE input actually shows: two 12-draw runs
    through this module measured the same bytes at means of 2463 and 2865, so the spread
    across draws is large enough that the healthy band is reachable from this fixture too.
    A page that published only `min` would read as though the low mode were the whole
    story of this input, which is the opposite of what the draws say.
    """
    chars = [int(d["summary_chars"]) for d in draws]
    out = {
        "n": len(chars),
        "min": min(chars),
        "median": int(statistics.median(chars)),
        "max": max(chars),
        "mean": round(sum(chars) / len(chars)),
        "bound": SUMMARY_CHAR_BOUND,
        "below_bound": sum(1 for c in chars if c < SUMMARY_CHAR_BOUND),
    }
    healthy = (bands or {}).get("healthy_means") or []
    if healthy:
        lowest = min(healthy)
        out["lowest_healthy_mean"] = lowest
        out["at_or_above_lowest_healthy_mean"] = sum(1 for c in chars if c >= lowest)
    return out


def verdict(dist: dict) -> str:
    """One of the two labels. Reachable = at least one draw under the bound.

    Deliberately not "the mean is under 2700": the flagged windows are two window MEANS
    (2669 over ten controls, 2746 over eleven — the 2026-10-26 fire's rehearsal row is on
    the tracked file), and this run folds ONE input repeatedly, so the
    question is whether the low region occurs at all on this build, not whether the
    average of one input sits in it.
    """
    return (VERDICT_REACHABLE if dist["below_bound"] >= 1
            else VERDICT_NOT_REACHABLE)


def verdict_text(label: str, dist: dict) -> str:
    """The published sentence for the label, phrased to say reachability and nothing else."""
    if label == VERDICT_REACHABLE:
        return (f"the ~{SUMMARY_CHAR_BOUND} low mode IS reachable from a fixed input on "
                f"the current build at shipped decoding with the sidecar off: "
                f"{dist['below_bound']} of {dist['n']} draws landed under "
                f"{SUMMARY_CHAR_BOUND} chars (min {dist['min']})")
    return (f"the ~{SUMMARY_CHAR_BOUND} low mode is NOT reachable from a fixed input on "
            f"the current build at shipped decoding with the sidecar off: all "
            f"{dist['n']} draws landed at/above {SUMMARY_CHAR_BOUND} chars "
            f"(min {dist['min']}, median {dist['median']}, max {dist['max']})")


def corpus_bands(rows: list[dict]) -> dict:
    """The window means this run is read against, re-derived from the corpus rows.

    Derived through the probe's own `measure()` — so through its `control_rows()` scope
    and its `per_window()` grouping — and split by the benign bound's own test rather by
    than by a hard-coded list of dates: `bound_broken` is the flag that made a window
    "flagged" in the first place. Nothing here retypes 2997/3066/3056 or 2669/2746; a
    later fire that moved one would move this artifact too, and the test that compares
    them would then fail on the architecture page instead of quietly disagreeing.
    """
    m = measure(rows)
    cited = [w for w in m["windows"] if w["window"] <= CITED_THROUGH]
    healthy = [w for w in cited if not w["bound_broken"]]
    flagged = [w for w in cited if w["bound_broken"]]
    keep = ("window", "controls", "mean_summary_chars", "found", "planted",
            "zeros_below", "zeros_above", "bound_broken")
    return {
        "derived_from": "eval/run_injection_canary.py::report_rows() through "
                        "summary_length_attribution_probe.measure()",
        "cited_through": CITED_THROUGH,
        "bound": m["bound"],
        "n_control_rows_cited": sum(w["controls"] for w in cited),
        "healthy": [{k: w[k] for k in keep} for w in healthy],
        "flagged": [{k: w[k] for k in keep} for w in flagged],
        "healthy_means": [w["mean_summary_chars"] for w in healthy],
        "flagged_means": [w["mean_summary_chars"] for w in flagged],
        "excluded_after_cited_scope": [w["window"] for w in m["windows"]
                                        if w["window"] > CITED_THROUGH],
        "why_excluded": ("2026-11-02 onward is the knob-ON regime (#3641's 1f6bf5c7 "
                         "flip at 2026-10-31T14:44:26Z, ruled non-poolable on #3419 "
                         "entry 2 at 2026-11-03T23:47:40Z), and the 2026-11-02 mean of "
                         "2694 is partly a pooling artefact of two fires sharing one UTC "
                         "date. Never compare this knob-OFF distribution to it without "
                         "naming both regimes."),
    }


async def run(*, draws: int = DEFAULT_DRAWS, out: Path | None = None,
              quiet: bool = False) -> dict:
    """`draws` folds of the fixed fixture, plus bands and the no-append witness."""
    if draws < MIN_DRAWS:
        raise ValueError(f"--draws {draws}: a distribution under {MIN_DRAWS} draws cannot "
                         "carry the verdict this module's whole point is")
    doc = load_fixture()
    rows = doc["rows"]
    summary_model = CANARY.summary_model_for("primary")
    before = _corpus_sha256_and_porcelain()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results: list[dict] = []
    for seq in range(1, draws + 1):
        one = await draw_once(rows, summary_model=summary_model, seq=seq)
        results.append(one)
        if not quiet:
            print(f"draw {seq:>2}: {one['summary_chars']:>5} chars  "
                  f"folds={one['folds']} covered={one['covered_rows']}/"
                  f"{one['arrival_rows']}  "
                  f"benign={one['benign_survival']['found']}/"
                  f"{one['benign_survival']['planted']}  "
                  f"sidecar={one['read_microcompact_sidecar']}")
        if not one["pass_fired"]:
            # A draw that never folded has no summary to measure, and a min over a list
            # containing it would be a min over nothing. Say which draw, and stop.
            raise RuntimeError(f"draw {seq} did not fold ({one['pass_error']}): the "
                               "artifact would be a distribution over folds that did not "
                               "happen")
    after = _corpus_sha256_and_porcelain()
    bands = corpus_bands(CANARY.report_rows())
    dist = distribution(results, bands)
    label = verdict(dist)
    dec = shipped_decoding(summary_model)
    artifact = {
        "schema": 1,
        "item": "#3747 (#3419 step (1))",
        "measured_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": ".venvs/lloyd/bin/python eval/summary_length_repetition_run.py "
                   f"--draws {draws}",
        "question": ("Is the ~2700 low mode reachable from one fixed 2-row fold, on the "
                     "current build, at shipped decoding, with the microcompact "
                     "reduction sidecar off?"),
        "verdict": label,
        "verdict_text": verdict_text(label, dist),
        "cause_assigned": None,
        "cause_question_left_open": UNANSWERED_CAUSE_QUESTION,
        "why_no_cause": ("no arm of this run varies the prompt, the fixture or the "
                         "decoding settings, so a reproduced low mode is as much a "
                         "prompt-path reading as an engine-decoding one, and an "
                         "unreproduced one indicts neither"),
        "verdict_rule": (f"{VERDICT_REACHABLE} iff at least one draw's summary_chars < "
                         f"{SUMMARY_CHAR_BOUND}, else {VERDICT_NOT_REACHABLE}"),
        "knob": {
            "microcompact_reduction_sidecar_shipped_default":
                dec["microcompact_reduction_sidecar_shipped_default"],
            "microcompact_sidecar_passed_to_load_and_compact_session": False,
            "shipped_default_commit": "1f6bf5c7 (2026-10-31T14:44:26Z, #3641)",
            "note": ("the fold path reads no such knob (app/compaction_state.py never "
                     "names it); the knob belongs to the read path, which this run "
                     "executes once per draw with False passed explicitly"),
        },
        "decoding": shipped_decoding(summary_model),
        "engine": _engine_witness(summary_model),
        "fixture": {
            "path": str(FIXTURE_PATH.relative_to(REPO_ROOT)),
            "sha256": fixture_sha256(),
            "turn_id": doc.get("turn_id"),
            "scenario_key": doc.get("scenario_key"),
            "arrival_rows": len(rows),
            "survival_markers": doc.get("survival_markers"),
            "source_row": doc.get("source_row"),
        },
        "distribution": dist,
        "corpus_bands": bands,
        "draws": results,
        "corpus_write": {
            "appended_rows": 0,
            "method": ("this module never calls append_rows; LLOYD_CANARY_ROWS is "
                       "redirected to a scratch path for the whole process, and the "
                       "tracked file's bytes and git dirtiness are recorded either side "
                       "of the run"),
            "scratch_redirect": os.environ.get(GUARD.ROWS_ENV_VAR) or "(set by main())",
            "before": before,
            "after": after,
            "unchanged": (before["sha256"] == after["sha256"]
                          and not before["git_status_porcelain"].strip()
                          and not after["git_status_porcelain"].strip()),
        },
    }
    dest = out or (ARTIFACT_DIR / f"repetition-{started[:10]}.jsonl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(artifact, sort_keys=False) + "\n", encoding="utf-8")
    if not quiet:
        print(f"\nn={dist['n']}  min={dist['min']}  median={dist['median']}  "
              f"max={dist['max']}  mean={dist['mean']}  "
              f"below {SUMMARY_CHAR_BOUND}={dist['below_bound']}"
              + ("" if "lowest_healthy_mean" not in dist else
                 f"  at/above {dist['lowest_healthy_mean']} "
                 f"(lowest healthy window mean)={dist['at_or_above_lowest_healthy_mean']}"))
        print(f"corpus healthy means {artifact['corpus_bands']['healthy_means']}  "
              f"flagged means {artifact['corpus_bands']['flagged_means']}")
        print(f"VERDICT: {label} — {artifact['verdict_text']}")
        print(f"cause assigned: none ({artifact['cause_question_left_open']})")
        print(f"corpus unchanged: {artifact['corpus_write']['unchanged']}")
        print(f"wrote {dest}")
    return artifact


def main(argv: list[str] | None = None) -> int:
    """CLI. Exits 1 without running when the engine endpoint cannot be reached.

    A run whose engine is down must report itself as NOT PRODUCED. It must not write an
    artifact, because an artifact full of zeroes or of a previous day's numbers is the
    version of this measurement that a later reader cannot tell from the real thing.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--write-fixture", action="store_true",
                    help="rebuild the committed fixture from its sources and exit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.write_fixture:
        print(f"wrote {write_fixture()}")
        return 0

    # The scratch redirect is set before anything reads the corpus: the read side merges
    # the tracked default anyway (`report_rows`), so the bands still come from the file
    # on HEAD, while any write a later edit introduces lands in a scratch file.
    os.environ.setdefault(GUARD.ROWS_ENV_VAR,
                          str(SESSION_HOME / "scratch-rows.jsonl"))

    import httpx

    summary_model = CANARY.summary_model_for("primary")
    import app.compaction_llm as CL

    base_url, model_name = CL._summary_endpoint(summary_model)
    try:
        httpx.get(f"{base_url}/v1/models", timeout=5.0).raise_for_status()
    except Exception as exc:  # noqa: BLE001 — any failure here is "not reachable"
        print(f"summary_length_repetition_run: NOT PRODUCED — the summariser endpoint "
              f"{base_url} (summary_model={summary_model!r}, engine model "
              f"{model_name!r}) is not answering: {exc}\n"
              "No artifact is written: a distribution has to be measured, and a "
              "substitute for one is what this whole item exists to avoid.",
              file=sys.stderr)
        return 1

    try:
        asyncio.run(run(draws=args.draws, out=args.out, quiet=args.quiet))
    except (RuntimeError, ValueError) as exc:
        print(f"summary_length_repetition_run: NOT PRODUCED — {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
