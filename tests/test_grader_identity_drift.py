"""#3753 clause 4: the report-only check for calibrations that outlived their judge.

What each test refuses to let this script become:

1. A changed judge must be PRINTED with the artifact's path, its age and both
   identities. A count with no path is a number a person cannot open, and the item's
   acceptance says the ceiling artifact must appear "by path and age".
2. The zero must be reachable and readable: with every recorded identity equal to what
   the engine now reports, the run prints `0 mismatched` (the positive control — an
   instrument that always reports drift is as useless as one that never does).
3. An artifact that recorded NO identity must never read as a match. `"" in live` is
   True in Python, so an artifact written before #3753 — every label-agreement ceiling
   on the box today, and the whole judge-compare baseline, which records no judge at
   all — would otherwise pass silently. That is the exact tautology the item names for
   the djev role (`djev == djev`, because :8010's served `id` is a static name), and it
   is why `unrecorded` is a verdict of its own rather than a zero.
4. A silent engine is reported as `unreachable`, not as a match either.
5. It reports and gates nothing: exit code 0 with a mismatch on screen, no file written
   beside the artifacts, and no import of the gate anywhere in the module. A review rung
   that refused whenever the engine moved could refuse a good build because the grader
   got stricter, which is why #3753 step 5 keeps the decision out of this file and
   #1549/#3749 set the convention.

Everything runs against injected identities and tmp artifact dirs, so no engine, GPU or
running endpoint is involved: the live probes are pinned separately, by identity of the
function they call, not by hitting a socket.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.maintenance.grader_identity_drift as gid  # noqa: E402

#: The checkpoint the djev engine at :8010 reports in `root` today — the field that is
#: NOT the static served `id` of `djev`, which is the whole reason the comparison is
#: worth running for that role.
DJEV_ROOT = ("/home/alansrobotlab/lloyd/agent-services/llm/models/"
             "nvidia-diffusiongemma-26B-A4B-it-NVFP4")
OTHER_DJEV_ROOT = ("/home/alansrobotlab/lloyd/agent-services/llm/models/"
                   "nvidia-diffusiongemma-27B-A5B-it-FP8")
PRIMARY_SERVED = ("/home/alansrobotlab/lloyd/agent-services/llm/models/"
                  "Inferact-Qwen3.8-Flash-Next-NVFP4 Qwen3.8-Flash-Next-nvfp4 primary")


def _label_artifact(path: Path, *, served_root: str | None,
                    ran_at: str = "2026-10-14T10:38:47Z") -> Path:
    """A ceiling artifact shaped like the live one the nightly reads its ceiling from."""
    labeler: dict = {"kind": "engine", "model": "djev",
                     "endpoint": "http://127.0.0.1:8010/v1/chat/completions"}
    if served_root is not None:
        labeler["served_root"] = served_root
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": 1, "kind": "gold_label_surrogate",
                                "ran_at": ran_at, "labeler": labeler}),
                    encoding="utf-8")
    return path


def _live(**by_role: str):
    """An injected live-identity probe: a role's answer, or 'unreachable'."""
    return lambda role: by_role.get(role, gid.UNREACHABLE)


def _rows(tmp_path, *, artifact: Path, live: str) -> list[dict]:
    return gid.judged(gid.artifact_rows("label-agreement", artifact.parent,
                                        "label-agreement-*.json",
                                        gid._identity_from_label_agreement),
                      _live(djev=live))


# ── clause 4: the report itself ─────────────────────────────────────────────

def test_a_changed_judge_is_printed_with_the_artifacts_path_and_age(tmp_path):
    """Clause 4: path, recorded identity, live identity and a verdict, one line each.

    The artifact is the ceiling the retrieval-eval nightly reads `entity_hit_rate`
    against, so a stale labeler is not a curiosity — it is a bound scored against a
    labeler that may no longer exist, which is the failure the talk measured at a 0.8
    to 0.2 swing from swapping a judge and nothing else.
    """
    art = _label_artifact(tmp_path / "label-agreement-20261014-103853.json",
                          served_root=DJEV_ROOT)
    rows = _rows(tmp_path, artifact=art, live=OTHER_DJEV_ROOT)
    assert len(rows) == 1 and rows[0]["verdict"] == gid.DIFFERENT
    line = gid.render(rows)[0]
    assert art.name in line, f"the path is not printed: {line}"
    assert DJEV_ROOT in line and OTHER_DJEV_ROOT in line, (
        f"both identities must be on the page to be compared by eye: {line}")
    assert "role=djev" in line and "age=" in line, line
    assert gid.summary_line(rows).startswith(
        "grader identity drift: 1 artifact(s) — 1 mismatched"), gid.summary_line(rows)


def test_an_unchanged_judge_prints_zero_mismatches(tmp_path):
    """The positive control: an instrument that always cries drift is useless.

    Containment, not equality, is the matching rule — an artifact recorded by
    checkpoint and an engine reporting `"<root> <served-name>"` name one model, and
    demanding equality would print a mismatch on every vLLM that appends its served
    name. That is the substring convention `eval/instruction_compliance_eval.py`
    already uses to decide whether a served string is the expected model.
    """
    art = _label_artifact(tmp_path / "label-agreement-now.json", served_root=DJEV_ROOT)
    rows = _rows(tmp_path, artifact=art, live=f"{DJEV_ROOT} djev")
    assert rows[0]["verdict"] == gid.SAME
    summary = gid.summary_line(rows)
    assert "0 mismatched" in summary and "1 same" in summary, summary


def test_an_artifact_that_recorded_no_identity_is_never_a_match(tmp_path):
    """The tautology this item exists to break, tested from the artifact's side.

    Every ceiling artifact on the box predates `labeler.served_root`, and the
    judge-compare baseline records no judge identity at all. An empty recorded string
    is `in` any live string, so the order of the checks in `compare` is the safety of
    the whole report: `unrecorded` BEFORE the containment test. The count stays honest
    by printing both numbers — `0 mismatched` beside `1 unrecorded` is a coverage gap,
    not an all-clear.
    """
    art = _label_artifact(tmp_path / "label-agreement-old.json", served_root=None)
    rows = _rows(tmp_path, artifact=art, live=DJEV_ROOT)
    assert rows[0]["verdict"] == gid.UNRECORDED, (
        "an artifact that names no checkpoint read as a match — the djev == djev "
        "tautology the item was written about")
    assert gid.compare("", DJEV_ROOT) == gid.UNRECORDED
    assert gid.compare("unresolved", DJEV_ROOT) == gid.UNRECORDED, (
        "`coverage_leg` writes `unresolved` when its own probe did not answer; "
        "matching that against itself would certify a measurement nobody took")
    summary = gid.summary_line(rows)
    assert "0 mismatched" in summary and "1 unrecorded" in summary, summary


def test_an_engine_that_will_not_name_itself_is_reported_not_matched(tmp_path):
    art = _label_artifact(tmp_path / "label-agreement-x.json", served_root=DJEV_ROOT)
    rows = _rows(tmp_path, artifact=art, live="")
    assert rows[0]["verdict"] == gid.UNREACHABLE
    assert gid.UNREACHABLE in gid.summary_line(rows)
    assert "0 mismatched" in gid.summary_line(rows), (
        "a probe that could not see must not add to the drift count")


def test_only_the_writers_own_artifact_names_are_read(tmp_path):
    """A stray JSON in the directory is not an instrument: the globs are the writers'.

    `label-agreement-*.json` is `label_agreement_ceiling.ARTIFACT_GLOB` and `*-n*.json`
    is the shape `coverage_leg.artifact_path` builds, so this reads artifacts where
    they are written and nothing else that happens to sit beside them.
    """
    _label_artifact(tmp_path / "label-agreement-real.json", served_root=DJEV_ROOT)
    (tmp_path / "nightly-20261103.json").write_text("{}", encoding="utf-8")
    rows = gid.artifact_rows("label-agreement", tmp_path, "label-agreement-*.json",
                             gid._identity_from_label_agreement)
    assert [Path(r["path"]).name for r in rows] == ["label-agreement-real.json"]
    assert gid.artifact_rows("label-agreement", tmp_path / "absent",
                             "label-agreement-*.json",
                             gid._identity_from_label_agreement) == []


def test_a_coverage_arm_is_compared_against_the_alias_it_recorded(tmp_path):
    """The class that can already print a real verdict, because its writer records a
    served `model_id` — the autoresearch figure a promotion is judged on.

    Both readings are here: an arm drawn under the engine now serving `primary` is
    `same`, and one drawn before the slot moved is `different`. The second is what
    standing problem 5 has no instrument for — a promotion mean that moved with the
    grader and was read as the variants moving.
    """
    same = tmp_path / "primary-direct-n8.json"
    same.write_text(json.dumps({"model_alias": "primary", "model_id": PRIMARY_SERVED,
                                "route": "direct", "n_draws": 8,
                                "generated_at": "2026-11-01T00:00:00Z"}),
                    encoding="utf-8")
    moved = tmp_path / "primary-direct-n4.json"
    moved.write_text(json.dumps({"model_alias": "primary",
                                 "model_id": "nova2-9b-cpt-sft-rl primary",
                                 "route": "direct", "n_draws": 4,
                                 "generated_at": "2026-11-02T00:00:00Z"}),
                     encoding="utf-8")
    rows = gid.judged(gid.artifact_rows("coverage-arm", tmp_path, "*-n*.json",
                                        gid._identity_from_coverage),
                      _live(primary=PRIMARY_SERVED))
    verdicts = {Path(r["path"]).name: r["verdict"] for r in rows}
    assert verdicts == {"primary-direct-n8.json": gid.SAME,
                        "primary-direct-n4.json": gid.DIFFERENT}


def test_the_judge_compare_baseline_is_reported_as_recording_no_judge(tmp_path):
    """The item's second named artifact, printed as the silence it is.

    `eval/autoresearch_judge_compare.py` grades with `--model`, whose default is
    `primary`, and its artifact stores `judges`, `coverage` and `judge_agreement` and
    no model identity. Naming `primary` here would substitute the alias for the
    measurement #3753 exists to make, so the row is `unrecorded` — the finding, on the
    page rather than asserted in prose.
    """
    art = tmp_path / "autoresearch-judge-compare-2026-09-24.json"
    art.write_text(json.dumps({"n_traces": 138, "judges": {"scalar": {"calls": 690}},
                               "judge_agreement": {"n": 23, "pearson": 0.42}}),
                   encoding="utf-8")
    rows = gid.judged(gid.artifact_rows("judge-compare", tmp_path,
                                        "autoresearch-judge-compare-*.json",
                                        gid._identity_from_judge_compare),
                      _live(primary=PRIMARY_SERVED))
    assert len(rows) == 1
    assert rows[0]["role"] == gid.JUDGE_COMPARE_ROLE
    assert rows[0]["verdict"] == gid.UNRECORDED and rows[0]["recorded"] == ""


# ── the CLI: report, and change nothing ─────────────────────────────────────

def test_the_command_reports_a_mismatch_and_still_exits_zero(tmp_path, capsys):
    """Exit code 0 with a drift on screen: report-only, in the one way a job can read.

    Contrast `scripts/maintenance/engine_build_drift.py`, which owns a property a build
    can break and therefore encodes DRIFT as exit 1. Nothing here consumes this exit
    code as a verdict, and a nightly that read a judge swap as a job failure would be
    gating on an identity change — the decision #3753 step 5 explicitly leaves to a
    person.
    """
    art_dir = tmp_path / "la"
    _label_artifact(art_dir / "label-agreement-20261014-103853.json",
                    served_root=DJEV_ROOT)
    before = {p.name: os.stat(p).st_mtime_ns for p in art_dir.iterdir()}
    rc = gid.main(["--label-agreement-dir", str(art_dir),
                   "--judge-compare-dir", str(tmp_path / "no-measurements"),
                   "--coverage-dir", str(tmp_path / "no-coverage")],
                  live_for=_live(djev=OTHER_DJEV_ROOT, primary=PRIMARY_SERVED))
    out = capsys.readouterr().out
    assert rc == 0, "a report-only check that fails its own exit code is a gate"
    assert "[label-agreement]" in out and "1 artifact(s)" in out, out
    assert "label-agreement-20261014-103853.json" in out, out
    assert "DIFFERENT" in out, out
    assert "grader identity drift: 1 artifact(s) — 1 mismatched" in out, out
    assert "report-only: gates nothing" in out, out
    assert {p.name: os.stat(p).st_mtime_ns for p in art_dir.iterdir()} == before, (
        "the report wrote beside the artifacts it reads")


def test_the_command_prints_the_positive_control_over_injected_dirs(tmp_path, capsys):
    """Clause 4's zero, from the same argv path a nightly job would use."""
    _label_artifact(tmp_path / "label-agreement-new.json", served_root=DJEV_ROOT)
    rc = gid.main(["--label-agreement-dir", str(tmp_path),
                   "--judge-compare-dir", str(tmp_path / "none-a"),
                   "--coverage-dir", str(tmp_path / "none-b")],
                  live_for=_live(djev=DJEV_ROOT))
    out = capsys.readouterr().out
    assert rc == 0
    assert "SAME" in out and "0 mismatched" in out and "1 same" in out, out


def test_the_script_imports_no_gate_and_holds_no_write_call():
    """The 'changes no gate decision' clause, checked against the module's own text.

    `import scripts.automod` and the artifact-write names are what would let this
    report become a decision, so the absence of both is asserted where a reviewer can
    see it. The stronger form — a gate run whose verdict is unchanged by this script —
    is not observable from a unit test, and the report's exit code is pinned to 0 by
    the CLI tests above for exactly that reason.
    """
    src = (ROOT / "scripts" / "maintenance"
           / "grader_identity_drift.py").read_text(encoding="utf-8")
    for forbidden in ("scripts.automod", "import gate", "automod_gate", "promote",
                      "write_text", "mkdir", "unlink", "open("):
        assert forbidden not in src, f"the report-only script mentions {forbidden!r}"


# ── the live-identity door: reused, never re-cut ────────────────────────────

def test_the_djev_role_is_resolved_through_the_labelers_own_root_reader(monkeypatch):
    """No fourth probe path: the djev identity comes from `served_checkpoint_root`.

    That function is the one #3753 clause 5 makes the ceiling artifact record against,
    so the artifact and the live side of the comparison are read the same way by
    construction — two different readers of one identity is precisely how the
    `djev == djev` tautology came about.
    """
    import eval.label_agreement_ceiling as lac
    from app.config import CONFIG
    asked: list[str] = []
    monkeypatch.setattr(lac, "served_checkpoint_root", lambda ep: asked.append(ep) or DJEV_ROOT)
    monkeypatch.setitem(CONFIG, "djev", {"base_url": "http://127.0.0.1:8010"})
    assert gid.live_served_identity("djev") == DJEV_ROOT
    assert asked == ["http://127.0.0.1:8010"]


def test_an_alias_role_is_resolved_through_the_existing_alias_to_served_door(monkeypatch):
    """`primary` goes through `coverage_leg.resolve_served_model`, and its `None` —
    the same non-guess that makes `measurement_context` write `unresolved` — becomes
    `unreachable` here rather than the alias."""
    import scripts.autoresearch.coverage_leg as cov
    monkeypatch.setattr(cov, "resolve_served_model", lambda alias: PRIMARY_SERVED)
    assert gid.live_served_identity("primary") == PRIMARY_SERVED
    monkeypatch.setattr(cov, "resolve_served_model", lambda alias: None)
    assert gid.live_served_identity("primary") == gid.UNREACHABLE


def test_the_gate_records_the_same_served_string_this_script_reads(monkeypatch):
    """One door for both halves of #3753.

    The gate's `grader_model_served` and this script's live column must be the same
    measurement taken twice, or a `same` verdict here would not mean the review event
    was written by the engine it claims. Both call
    `scripts.autoresearch.coverage_leg.resolve_served_model`, so the equivalence is
    asserted rather than hoped for.
    """
    from scripts.automod import gate as G
    import scripts.autoresearch.coverage_leg as cov
    monkeypatch.setattr(cov, "resolve_served_model", lambda alias: PRIMARY_SERVED)
    assert G.grader_served_model("primary") == gid.live_served_identity("primary")
    monkeypatch.setattr(cov, "resolve_served_model", lambda alias: None)
    assert G.grader_served_model("primary") == G.GRADER_SERVED_UNKNOWN == "unknown"
    assert gid.live_served_identity("primary") == gid.UNREACHABLE
