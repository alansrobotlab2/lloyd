"""`/api/workers/enable` — the switch that used to rewrite config.yaml.

The endpoint did `config.yaml.write_text(yaml.dump(CONFIG))`, and each of the
three consequences was serious on its own:

  * `CONFIG` is the *loaded* config, with `${VAR}` already expanded — so the
    dump would have written `livekit.api_secret` in clear into a tracked file.
  * It flattened every comment out of a 600-line file that is mostly comments.
  * It left the live tree dirty, which `scripts/automod/gate.py` and
    `promote.py` both refuse — one click silently stopping the
    self-modification loop until a human committed the damage.

That is the identical defect the Tools page was moved off `config.yaml` to
avoid (see `test_tool_overrides.py`); this endpoint kept it because nothing in
the frontend calls it yet. `workers.enabled` now travels the same route.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import yaml

import app.config as appconfig
import app.routers.workers as router

ROOT = Path(__file__).resolve().parent.parent


def test_the_endpoint_does_not_write_config_yaml():
    src = inspect.getsource(router.workers_enable)
    code = "\n".join(line for line in src.splitlines()
                     if not line.strip().startswith("#")).split('"""')[2]
    assert "config.yaml" not in code, "the workers switch writes the tracked config again"
    assert "yaml.dump" not in code
    assert "save_tool_overrides()" in code


def test_the_switch_is_persisted_to_the_untracked_override_file(monkeypatch, tmp_path):
    overrides = tmp_path / "tool_overrides.yaml"
    monkeypatch.setattr(appconfig, "TOOL_OVERRIDES_PATH", overrides)
    monkeypatch.setitem(appconfig.CONFIG, "workers", {"enabled": False, "slots": 2})

    appconfig.save_tool_overrides()
    written = yaml.safe_load(overrides.read_text())
    assert written["workers"] == {"enabled": False}
    assert "slots" not in written["workers"], "only the UI-mutable key belongs here"


def test_the_override_decides_what_the_pool_does_on_the_next_boot(monkeypatch, tmp_path):
    overrides = tmp_path / "tool_overrides.yaml"
    overrides.write_text(yaml.dump({"workers": {"enabled": False}}))
    monkeypatch.setattr(appconfig, "TOOL_OVERRIDES_PATH", overrides)

    merged = appconfig._merge_tool_overrides({"workers": {"enabled": True, "slots": 2}})
    assert merged["workers"]["enabled"] is False
    assert merged["workers"]["slots"] == 2, "the override must not replace the block"


def test_a_disagreement_with_the_tracked_config_is_logged(monkeypatch, tmp_path, caplog):
    """Agreement stays silent; a silent win is how the tracked file starts
    describing a state nobody is serving."""
    overrides = tmp_path / "tool_overrides.yaml"
    overrides.write_text(yaml.dump({"workers": {"enabled": False}}))
    monkeypatch.setattr(appconfig, "TOOL_OVERRIDES_PATH", overrides)

    with caplog.at_level("WARNING"):
        appconfig._merge_tool_overrides({"workers": {"enabled": True}})
    assert "workers.enabled" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        appconfig._merge_tool_overrides({"workers": {"enabled": False}})
    assert "workers.enabled" not in caplog.text, "agreement should not warn every boot"


def test_no_secret_can_reach_the_override_file(monkeypatch, tmp_path):
    """The dump wrote the *expanded* config. Only three keys are emitted here,
    so an expanded secret elsewhere in CONFIG cannot ride along."""
    overrides = tmp_path / "tool_overrides.yaml"
    monkeypatch.setattr(appconfig, "TOOL_OVERRIDES_PATH", overrides)
    monkeypatch.setitem(appconfig.CONFIG, "livekit",
                        {"api_secret": "super-secret-value"})
    monkeypatch.setitem(appconfig.CONFIG, "workers", {"enabled": True})

    appconfig.save_tool_overrides()
    text = overrides.read_text()
    assert "super-secret-value" not in text
    assert set(yaml.safe_load(text)) <= {"mcp_servers", "harness", "workers"}


def test_the_override_file_stays_untracked():
    """It now carries `workers.enabled` too, so re-tracking it would arm the
    same trap for a second endpoint."""
    import subprocess
    r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
                        "data/tool_overrides.yaml"], capture_output=True, text=True)
    assert r.returncode != 0, "data/tool_overrides.yaml is tracked again"


def test_config_yaml_holds_no_expanded_secret():
    """A regression canary for the whole class: if some future endpoint dumps
    CONFIG over the tracked file again, the placeholders disappear and this
    fails."""
    raw = (ROOT / "config.yaml").read_text()
    assert "${LIVEKIT_API_SECRET}" in raw, \
        "config.yaml no longer carries its placeholder — a secret may be in the tree"


# ---------------------------------------------------------------------------
# #706 — promoting a `bench-mine` candidate must land the bench TASK, not the
# mining run's staging metadata.
#
# A staged candidate is two documents: `write_staging_note`
# (`workers/sources/_common.py`) owns the file's first frontmatter block
# (`calibration`, `confidence`, `source_refs`, `generated_at`), and the mining
# turn's answer — the candidate's own bench-task block, sometimes wrapped in a
# ``` fence — sits in the body. The endpoint used to write that first block to
# `lloyd/bench/<name>.md` and count it a promotion. `load_bench_tasks`
# (`scripts/autoresearch/common.py`) reads only the first block, so the landed
# task had no `id`, no `category`, no `prompt` and no `objective_checks`;
# `bench_runner` then posts `task.get("prompt") or task.get("_body")` — the
# calibration YAML — as the prompt, and `judge._score_objective` hands a task
# with no checks `1.0` ("no objective layer -> full marks"). Every one of the
# 11 tasks promoted that way would have been a guaranteed pass, and the bench
# file count — the number #522's acceptance checked — would still have gone up.
# ---------------------------------------------------------------------------

import yaml as _yaml  # noqa: E402  (module already imports yaml; kept explicit below)
from fastapi import FastAPI as _FastAPI  # noqa: E402
from fastapi.testclient import TestClient as _TestClient  # noqa: E402

from scripts.autoresearch.common import load_bench_tasks  # noqa: E402

TASK_ID = "bench_012_replay_report_deliverable"

STAGING_FM = {
    "source": "bench-mine",
    "confidence": 0.5,
    "review_status": "pending",
    "rationale": "mined from failed autonomy run run_38_20260911_050055",
    "source_refs": ["/home/alansrobotlab/lloyd/autonomy-runs/38/"
                    "run_38_20260911_050055.md"],
    "generated_at": "2026-09-11T06:04:54+00:00",
    "calibration": {"status": "ok", "runs": 10, "mean": 0.5, "min": 0.1,
                    "max": 0.9, "in_band": True, "error": "",
                    "band": [0.05, 0.95], "composites": [0.5] * 10},
    "edge_direction": "scenario-novelty",
}

TASK_FM = {
    "segment": "lloyd",
    "id": TASK_ID,
    "category": "replay",
    "tags": ["replay", "mined-from-run-38"],
    "objective": "Deliver the report instead of narrating the churn.",
    "max_tool_calls": 4,
    "edge_direction": "scenario-novelty",
    "requires_runtime": True,
    "prompt": ("Your last four calls all failed. Answer from those four lines "
               "alone. Report in exactly this shape: VERDICT: <next action>"),
    "objective_checks": [
        {"type": "contains", "value": "VERDICT:"},
        {"type": "max_tool_calls", "value": "4"},
    ],
    "rubric_criteria": ["report_shape", "conciseness"],
    "safety_critical": False,
}

TASK_PROSE = ("Success is a sentinelled report that names the next action.\n\n"
              "Mined from `autonomy-runs/38/run_38_20260911_050055.md`: "
              "status=failed, empty response at max_turns.")


def _staged(tmp_path: Path, name: str, *, task_fm: dict | None = None,
            prose: str = TASK_PROSE, fence: bool = False,
            staging_fm: dict | None = None, source: str = "bench-mine",
            body: str | None = None) -> Path:
    """Write one staged artifact the way `write_staging_note` writes it."""
    d = tmp_path / "pending-research" / source / "2026-09-18"
    d.mkdir(parents=True, exist_ok=True)
    task_fm = TASK_FM if task_fm is None else task_fm
    if body is None:
        block = (f"---\n{_yaml.dump(task_fm, default_flow_style=False, allow_unicode=True)}"
                 f"---\n\n{prose}\n")
        body = f"```\n{block}```\n" if fence else block
    fm = STAGING_FM if staging_fm is None else staging_fm
    p = d / name
    p.write_text(f"---\n{_yaml.dump(fm, default_flow_style=False, allow_unicode=True)}"
                 f"---\n\n{body}", encoding="utf-8")
    return p


def _client(monkeypatch, tmp_path: Path) -> _TestClient:
    """A client whose two roots are `tmp_path`, so nothing touches the vault."""
    monkeypatch.setattr(router, "PENDING_ROOT", tmp_path / "pending-research")
    monkeypatch.setattr(router, "VAULT_ROOT", tmp_path / "vault")
    app = _FastAPI()
    app.include_router(router.router)
    return _TestClient(app)


def _promote(client: _TestClient, path: Path, **extra) -> dict:
    r = client.post("/api/workers/pending/promote",
                    json={"path": str(path), **extra})
    return {"status": r.status_code, **(r.json() if r.content else {})}


def test_a_staged_candidate_lands_its_bench_task_block(monkeypatch, tmp_path):
    """Clause 1: the destination's FIRST frontmatter block is the task block,
    and the staging keys appear nowhere in the landed file."""
    src = _staged(tmp_path, "060454-mined-from-run-38-20260911-050055.md")
    out = _promote(_client(monkeypatch, tmp_path), src)
    assert out["status"] == 200, out

    dest = tmp_path / "vault" / "lloyd" / "bench"
    landed = (dest / f"{TASK_ID}.md").read_text(encoding="utf-8")
    tasks = load_bench_tasks(dest)
    assert len(tasks) == 1
    task = tasks[0]
    assert str(task.get("id")) == TASK_ID
    assert str(task.get("category")) == "replay"
    assert str(task.get("prompt")).strip() == TASK_FM["prompt"]
    assert len(task.get("objective_checks") or []) == 2, task.get("objective_checks")
    assert task.get("_body").startswith("Success is a sentinelled report")
    for key in router.BENCH_STAGING_KEYS:
        assert key not in landed, f"{key} rode along into the graded bench"
    assert not src.exists(), "the staged artifact must be consumed by the move"


def test_a_fenced_candidate_lands_the_same_task(monkeypatch, tmp_path):
    """4 of the 14 staged files under `_pipeline/vault-derived/pending-research/"
    "bench-mine/` have their task block inside a ``` fence — the same fence
    `bench_mine._candidate_frontmatter` strips off a raw mining answer."""
    src = _staged(tmp_path, "104039-mined-from-run-58.md", fence=True)
    out = _promote(_client(monkeypatch, tmp_path), src)
    assert out["status"] == 200, out

    dest = tmp_path / "vault" / "lloyd" / "bench"
    tasks = load_bench_tasks(dest)
    assert len(tasks) == 1
    assert tasks[0].get("id") == TASK_ID
    assert "```" not in (dest / f"{TASK_ID}.md").read_text(encoding="utf-8")


def test_the_landed_task_is_graded_on_its_checks_rather_than_by_default(monkeypatch,
                                                                        tmp_path):
    """The consequence the 11 -> >=15 count hid: a check-less task gets full
    marks, so promoting the staging block ADDED a guaranteed pass."""
    from scripts.autoresearch.judge import _score_objective

    src = _staged(tmp_path, "060454-mined-from-run-38-20260911-050055.md")
    out = _promote(_client(monkeypatch, tmp_path), src)
    assert out["status"] == 200, out

    dest = tmp_path / "vault" / "lloyd" / "bench"
    landed = load_bench_tasks(dest)[0]
    trace = {"status": "success", "final_text": "sorry, retrying", "tool_calls": []}
    score, _results = _score_objective(landed, trace)
    # `contains VERDICT:` misses; `max_tool_calls 4` passes vacuously on an
    # empty trace — so 1 of the task's 2 checks, not 2 of 2.
    assert score == 0.5, "a landed task must be able to fail its own checks"

    # What the same file looked like promoted the old way: staging block on top
    # (what `load_bench_tasks` reads) and the task block down in the body.
    stale = _staged(tmp_path / "stale", "hand-cp.md").read_text(encoding="utf-8")
    (tmp_path / "oldstyle").mkdir(exist_ok=True)
    (tmp_path / "oldstyle" / "hand-cp.md").write_text(stale, encoding="utf-8")
    old = load_bench_tasks(tmp_path / "oldstyle")[0]
    assert not old.get("objective_checks")
    assert _score_objective(old, trace)[0] == 1.0, \
        "the pre-#706 artifact no longer reads as a guaranteed pass"


def test_a_candidate_with_no_prompt_is_refused_and_writes_nothing(monkeypatch,
                                                                  tmp_path):
    src = _staged(tmp_path, "no-prompt.md",
                  task_fm={**TASK_FM, "prompt": "   "})
    out = _promote(_client(monkeypatch, tmp_path), src)
    assert out["status"] == 400, out
    assert "prompt" in out["detail"]
    assert not (tmp_path / "vault" / "lloyd" / "bench").exists()
    assert src.exists(), "a refused candidate stays reviewable"


def test_a_candidate_with_no_objective_checks_is_refused_and_writes_nothing(
        monkeypatch, tmp_path):
    """This is what the human gate exists to catch: `judge._score_objective`
    awards 1.0 to a task with no checks, so a self-authored check-less task is
    a guaranteed pass the moment it lands."""
    from scripts.autoresearch.judge import _score_objective

    checkless = {k: v for k, v in TASK_FM.items() if k != "objective_checks"}
    assert _score_objective(checkless, {"status": "success",
                                        "final_text": "", "tool_calls": []})[0] == 1.0

    src = _staged(tmp_path, "checkless.md", task_fm=checkless)
    out = _promote(_client(monkeypatch, tmp_path), src)
    assert out["status"] == 400, out
    assert "objective_checks" in out["detail"]
    assert not (tmp_path / "vault" / "lloyd" / "bench").exists()
    assert src.exists()


def test_a_candidate_with_no_category_is_refused_and_writes_nothing(monkeypatch,
                                                                    tmp_path):
    """`bench_runner` records `task_category` from this key; without it every
    row the task produces is filed under 'unknown'."""
    src = _staged(tmp_path, "no-category.md", task_fm={**TASK_FM, "category": ""})
    out = _promote(_client(monkeypatch, tmp_path), src)
    assert out["status"] == 400, out
    assert "category" in out["detail"]
    assert not (tmp_path / "vault" / "lloyd" / "bench").exists()


def test_an_artifact_whose_body_carries_no_task_block_is_refused(monkeypatch,
                                                                tmp_path):
    src = _staged(tmp_path, "prose-only.md", body="## Notes\n\nnothing here\n")
    out = _promote(_client(monkeypatch, tmp_path), src)
    assert out["status"] == 400, out
    assert not (tmp_path / "vault" / "lloyd" / "bench").exists()
    assert src.exists()


def test_the_landed_id_is_rewritten_to_the_destination_stem(monkeypatch, tmp_path):
    """Four candidates were staged as `id: bench_012_*`; the human renaming one
    to `bench_014_*` must not leave a file whose declared id is another task's."""
    src = _staged(tmp_path, "060454-mined-from-run-38-20260911-050055.md")
    out = _promote(_client(monkeypatch, tmp_path), src,
                   filename="bench_013_renamed_report_deliverable.md")
    assert out["status"] == 200, out

    dest = tmp_path / "vault" / "lloyd" / "bench"
    tasks = load_bench_tasks(dest)
    assert len(tasks) == 1
    assert tasks[0].get("id") == "bench_013_renamed_report_deliverable"
    assert tasks[0].get("id") == (dest / "bench_013_renamed_report_deliverable.md").stem


def test_the_default_filename_is_the_candidates_own_id(monkeypatch, tmp_path):
    src = _staged(tmp_path, "060454-mined-from-run-38-20260911-050055.md")
    out = _promote(_client(monkeypatch, tmp_path), src)
    assert out["status"] == 200, out
    assert out["to"].endswith(f"/lloyd/bench/{TASK_ID}.md"), out


def test_a_bench_id_that_is_already_declared_is_refused(monkeypatch, tmp_path):
    dest = tmp_path / "vault" / "lloyd" / "bench"
    dest.mkdir(parents=True)
    (dest / "bench_012_replay_report_deliverable.md").write_text(
        f"---\n{_yaml.dump(TASK_FM, default_flow_style=False)}---\n\nprose\n",
        encoding="utf-8")

    src = _staged(tmp_path, "060454-mined-from-run-38-20260911-050055.md")
    out = _promote(_client(monkeypatch, tmp_path), src)
    assert out["status"] == 409, out
    assert len(list(dest.glob("*.md"))) == 1


def test_an_id_declared_under_an_unrelated_filename_still_blocks_the_landing(
        monkeypatch, tmp_path):
    """`load_bench_tasks` keys a task on its frontmatter `id`, not its filename.
    A candidate hand-`cp`'d into `lloyd/bench/` under some other name — which is
    the other documented way a candidate gets promoted — declares its id without
    owning the `<id>.md` path, so the free filename must not read as a free
    landing."""
    dest = tmp_path / "vault" / "lloyd" / "bench"
    dest.mkdir(parents=True)
    (dest / "hand-copied-under-another-name.md").write_text(
        f"---\n{_yaml.dump(TASK_FM, default_flow_style=False)}---\n\nprose\n",
        encoding="utf-8")

    src = _staged(tmp_path, "060454-mined-from-run-38-20260911-050055.md")
    out = _promote(_client(monkeypatch, tmp_path), src)
    assert out["status"] == 409, out
    assert not (dest / f"{TASK_ID}.md").exists()
    assert len(list(dest.glob("*.md"))) == 1
    assert src.exists(), "a refused candidate stays reviewable"


def test_a_renamed_landing_cannot_create_a_second_declaration_of_the_id(
        monkeypatch, tmp_path):
    """The rewrite is what makes a 200 safe: promoting the same candidate under a
    third name retires the staged `id` rather than duplicating it, so every id in
    the bench dir afterwards is owned by exactly one file — the file named after
    it."""
    dest = tmp_path / "vault" / "lloyd" / "bench"
    dest.mkdir(parents=True)
    (dest / "hand-copied-under-another-name.md").write_text(
        f"---\n{_yaml.dump(TASK_FM, default_flow_style=False)}---\n\nprose\n",
        encoding="utf-8")

    src = _staged(tmp_path, "060454-mined-from-run-38-20260911-050055.md")
    out = _promote(_client(monkeypatch, tmp_path), src,
                   filename="bench_014_renamed_report_deliverable.md")
    assert out["status"] == 200, out

    tasks = load_bench_tasks(dest)
    ids = [str(t.get("id")) for t in tasks]
    assert len(ids) == 2
    assert len(set(ids)) == 2, f"duplicate bench ids: {ids}"
    landed = [t for t in tasks
              if Path(str(t.get("_path"))).name == "bench_014_renamed_report_deliverable.md"]
    assert len(landed) == 1
    assert landed[0].get("id") == "bench_014_renamed_report_deliverable"


def test_four_candidates_staged_under_one_id_cannot_all_land(monkeypatch, tmp_path):
    """The literal #706 case: every one of the 5 pending candidates declares
    `id: bench_012_*`, and promoting them in sequence used to grow the bench by
    four files that all declared the same id."""
    client = _client(monkeypatch, tmp_path)
    names = [f"{ts}-mined-from-run-{n}.md"
             for ts, n in (("060454", 38), ("120719", 83), ("161113", 68),
                           ("043253", 57))]
    statuses = []
    for name in names:
        src = _staged(tmp_path, name)
        statuses.append(_promote(client, src)["status"])
    assert statuses == [200, 409, 409, 409], statuses

    dest = tmp_path / "vault" / "lloyd" / "bench"
    tasks = load_bench_tasks(dest)
    assert len(tasks) == 1
    assert len({str(t.get("id")) for t in tasks}) == 1


def test_a_non_bench_artifact_still_lands_its_own_frontmatter(monkeypatch, tmp_path):
    """The other sources have one frontmatter block and no task to extract —
    the staging rewrite must not reach them."""
    src = _staged(tmp_path, "note.md", source="domain-research",
                  staging_fm={"source": "domain-research", "confidence": 0.7,
                              "review_status": "pending"},
                  body="## Finding\n\nthe queue never drains\n")
    out = _promote(_client(monkeypatch, tmp_path), src)
    assert out["status"] == 200, out

    dest = tmp_path / "vault" / "knowledge" / "note.md"
    fm, body = router._parse_frontmatter(dest.read_text(encoding="utf-8"))
    assert fm["review_status"] == "promoted"
    assert fm["confidence"] == 0.7
    assert body.strip().startswith("## Finding")


def test_a_bench_it_cannot_read_is_not_reported_as_a_free_landing(monkeypatch,
                                                                  tmp_path):
    """An empty id set is the answer that lets a duplicate in, so it has to cost
    a promotion rather than be reported as a clean scan."""
    import scripts.autoresearch.common as AR

    src = _staged(tmp_path, "060454-mined-from-run-38-20260911-050055.md")
    client = _client(monkeypatch, tmp_path)

    def boom(_dir):
        raise OSError("bench dir unreadable")

    monkeypatch.setattr(AR, "load_bench_tasks", boom)
    out = _promote(client, src)
    assert out["status"] == 500, out
    assert "bench" in out["detail"]
    assert not (tmp_path / "vault" / "lloyd" / "bench").exists()
    assert src.exists()
