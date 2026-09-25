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
import re
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
    # `contains VERDICT:` misses, and `max_tool_calls 4` — which used to pass
    # vacuously on an empty trace and hand this task half its objective layer — is
    # now NOT_MEASURABLE (#416): the trace has no dispatch record, so there is no
    # count to compare against the cap. The fraction is over the one measured
    # check, which is the miss: 0.0, not 1-of-2. This is the exclusion making the
    # layer *stricter*, which is the direction #416 is meant to move in — a mined
    # task must not collect marks from a check that cannot fail here.
    assert score == 0.0, "a landed task must be able to fail its own checks"

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
    the staging rewrite must not reach them. `session-distill` has no default
    destination (the retired `domain-research`'s went with its notes, #1278),
    so the request names one, as the Review tab does for such a source."""
    src = _staged(tmp_path, "note.md", source="session-distill",
                  staging_fm={"source": "session-distill", "confidence": 0.7,
                              "review_status": "pending"},
                  body="## Finding\n\nthe queue never drains\n")
    out = _promote(_client(monkeypatch, tmp_path), src, destination="knowledge")
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


# ---------------------------------------------------------------------------
# #1016 — the run-to-transcript join in `runs.meta_json` was write-only.
#
# `workers/pool.py` binds `current_run_sessions` around each claimed job and
# writes the collected list into the run's meta blob on the normal, timeout and
# exception branches; a session-backed source additionally stamps a singular
# `session_id` into the same blob. Both keys reached a human only as one opaque
# string inside `meta_json` — `list_runs` is `SELECT *` and the two endpoints
# pass the row straight through — so the Background tab could not name, let
# alone open, the transcript a run produced. `architecture/background-runs.md`
# §7's "one record of a run names the transcripts it produced" was resting on
# the write half alone.
#
# These go through the real seam: an HTTP request in, the serialised JSON body
# out, over a queue whose rows carry the meta blobs the pool actually writes.
# ---------------------------------------------------------------------------

META_COLLECTED = '{"session_ids": ["20260918_161828_autonomy_fcb5"]}'
META_SINGULAR = '{"session_id": "20260918_161828_autonomy_fcb5"}'


class _JoinQueue:
    """The three queue methods the two endpoints call, over fixed run rows.

    `depth_by_source` names the one source on purpose: `/api/workers/health`
    builds its source list from config ∪ rollup ∪ depth, so this is what puts a
    source in the response without editing the tracked `config.yaml`.
    """

    def __init__(self, rows):
        self.rows = rows
        # Every call, so a test can pin what the endpoint *asked* for — see
        # `list_runs`.
        self.calls = []

    def run_rollup_by_source(self, since_iso):
        return {}

    def depth_by_source(self):
        return {"scheduled-task": {"queued": 0}}

    def list_runs(self, source=None, task_id=None, limit=100):
        """Filters and clips the way the real one does (`SELECT * FROM runs` with
        a WHERE on `source`/`task_id` and a LIMIT), and records the call.

        A fake that returned every row whatever the arguments would let the
        endpoint attribute another source's runs to this one, or fetch at a limit
        nobody asked for, and still read green — the join would be exercised for
        real while the query behind it was not.
        """
        self.calls.append({"source": source, "task_id": task_id, "limit": limit})
        out = [dict(r) for r in self.rows
               if (source is None or r.get("source") == source)
               and (task_id is None or r.get("task_id") == task_id)]
        return out[:max(1, int(limit))]


def _run_rows(*metas, source="scheduled-task"):
    """Rows shaped as `list_runs` returns them, all from one `source` — pass a
    different one to put another source's runs in the same table."""
    return [{
        "run_id": f"run_{source}_20260924_00000{i}_aa{i}",
        "source": source,
        "task_id": "24",
        "status": "success",
        "started_at": "2026-09-24T00:00:00+00:00",
        "completed_at": "2026-09-24T00:01:00+00:00",
        "duration_seconds": 60.0,
        "summary": "done",
        "meta_json": m,
    } for i, m in enumerate(metas)]


def _join_client(monkeypatch, tmp_path, rows):
    client = _client(monkeypatch, tmp_path)
    queue = _JoinQueue(rows)
    monkeypatch.setattr(router, "get_queue", lambda: queue)
    # What the endpoint asked the queue for, once the response is in hand.
    client.join_queue = queue
    return client


def test_a_runs_row_names_the_transcripts_it_produced(monkeypatch, tmp_path):
    """The union of the two meta keys, in the order the run recorded them.

    Two writers own the two keys — the pool binds the collected list, a source
    stamps its own `session_id` (`workers/sources/arch_review.py`,
    `autocode.py`) — and neither knows about the other, so the field is the union.
    Both keys are seeded here because a union of two literals is the only way to
    see that the order is the recorded one and not alphabetical by key; how many
    live rows carry both is never quoted, because the table is pruned (see
    `run_session_ids`).
    """
    client = _join_client(monkeypatch, tmp_path,
                          _run_rows('{"session_ids": ["a"], "session_id": "b"}'))

    row = client.get("/api/workers/runs").json()["runs"][0]

    assert row["session_ids"] == ["a", "b"]
    assert row["meta_json"] == '{"session_ids": ["a"], "session_id": "b"}', (
        "the raw blob still ships — api.ts declares it and dropping it from a "
        "row is a contract break on its own")


def test_the_health_recent_rows_carry_the_same_join(monkeypatch, tmp_path):
    """`/api/workers/health` builds `recent` from the same rows, so the Sources
    panel gets the field from the endpoint it actually polls — and from that
    source's own rows at the limit the caller asked for, not whatever the table
    happens to hold: another source's run in the same table must not reach this
    card, whose whole purpose is naming one source's recent runs.
    """
    client = _join_client(monkeypatch, tmp_path,
                          _run_rows(META_COLLECTED, META_SINGULAR)
                          + _run_rows(META_COLLECTED, source="bench-mine"))

    body = client.get("/api/workers/health?days=7&runs=5").json()
    src = next(s for s in body["sources"] if s["name"] == "scheduled-task")

    assert [r["session_ids"] for r in src["recent"]] == [
        ["20260918_161828_autonomy_fcb5"],
        ["20260918_161828_autonomy_fcb5"],
    ], "a source whose rows only carry the singular key must still link, and "       "only its own two rows may appear"
    assert {"source": "scheduled-task", "task_id": None, "limit": 5} \
        in client.join_queue.calls, (
        "the card must ask for its own source at the requested limit; a fetch "
        "with no source would have returned three rows and read as one more "
        "transcript this job never produced")


def test_a_row_naming_no_session_yields_an_empty_list_not_an_error(
        monkeypatch, tmp_path):
    """Six ways a row can name no transcript, and none of them is an exception.

    A Sources panel that 500s, or a row whose key is missing so the renderer
    has to null-check, is worse than an honest empty list. `architecture/
    background-runs.md` §12 lists the sources that record no transcript by design
    — `automod-regression`, `autoresearch` among them — so the empty case is
    those rows' normal shape, not the defect.
    """
    client = _join_client(monkeypatch, tmp_path, _run_rows(
        None,                      # column NULL
        "not json",                # truncated write
        "{}",                      # blob with neither key
        '{"session_ids": []}',     # collected list stayed empty
        '{"session_id": ""}',      # a source that stamped nothing
        "[1, 2]",                  # valid JSON, not an object
    ))

    runs = client.get("/api/workers/runs").json()["runs"]

    assert [r["session_ids"] for r in runs] == [[] for _ in range(6)], (
        "every one of the six must serialise as [], the value the renderer "
        "renders as no control")


def test_one_entry_per_transcript_even_when_both_keys_name_it(
        monkeypatch, tmp_path):
    """De-duplicated, and still one entry per distinct id."""
    client = _join_client(monkeypatch, tmp_path, _run_rows(
        '{"session_ids": ["x", "x"], "session_id": "x"}',
        '{"session_id": "second", "session_ids": ["first"]}',
    ))

    runs = client.get("/api/workers/runs").json()["runs"]

    assert [r["session_ids"] for r in runs] == [["x"], ["first", "second"]], (
        "the collected list leads, because it is what the pool bound; the "
        "singular key only appends ids the list never held")


PAGE_TSX = ROOT / "web/src/components/pages/BackgroundPage.tsx"
RUN_SESSIONS_TS = ROOT / "web/src/lib/runSessions.ts"
RUN_SESSIONS_SPEC = ROOT / "web/src/lib/runSessions.test.ts"
ARCH_PAGE = ROOT / "architecture/background-runs.md"


def test_the_background_tab_reads_the_join_through_a_pure_module():
    """The derivation stays reachable by a test, and its vitest sibling still
    asserts rather than merely describes.

    `web/src/lib/runSessions.ts` holds a run row's transcript list rather than
    deriving it inside `BackgroundPage.tsx` for one reason: this project's
    `web/vitest.config.ts` runs `environment: "node"` with no renderer, so logic
    living in a component has no test node, and the dependency that would change
    that (`web/package.json`) is human-only. The rule that keeps the derivation
    testable is therefore structural, and this node — beside the endpoint's own
    tests, which is where a regression in the join would be noticed — is what
    enforces it: the module imports nothing, and its vitest sibling is a set of
    cases each of which actually asserts something.

    Checked case by case rather than by the spec containing the right *words*: a
    spec whose assertions had been hollowed out kept every identifier in its
    comments and would satisfy a substring pin, which is precisely the hole the
    first version of this guard had. A case with no `expect(` in its own body is
    caught here, and the page's own markup is caught by
    `test_the_run_row_in_the_sources_panel_renders_one_control_per_transcript`.
    """
    module = RUN_SESSIONS_TS.read_text(encoding="utf-8")
    spec = RUN_SESSIONS_SPEC.read_text(encoding="utf-8")

    imports = [ln.strip() for ln in module.splitlines()
               if ln.strip().startswith("import ")]
    assert imports == [], (
        f"`runSessions.ts` would no longer be renderable by a node-environment "
        f"vitest suite: {imports}")
    assert "export function runTranscriptIds" in module, (
        "the module no longer exports the derivation the page calls")
    # The spec reads the page as text through Vite's raw import — the only
    # form that type-checks here, since `web/tsconfig.json` has no node types
    # and `web/package.json` is human-only.
    assert "?raw'" in spec, (
        "`runSessions.test.ts` no longer reads the page as raw text, so the "
        "page's wiring to this module is unpinned from the vitest side")

    bodies = spec.split("  it(")[1:]
    assert len(bodies) >= 8, (
        f"the vitest sibling has decayed to {len(bodies)} cases; the join's "
        "render half has no other CI-reachable pin")
    silent = [b.splitlines()[0].strip().rstrip(",") for b in bodies
              if "expect(" not in b]
    assert not silent, f"these cases describe without asserting: {silent}"
    map_case = [b for b in bodies if "transcripts.map(" in b]
    assert map_case, (
        "no case reads the per-entry control's markup, so a page rendering only "
        "the first transcript would pass")
    # Backslashes stripped: in the spec that call site lives inside a regex
    # literal, where the parens are escaped.
    assert "expect(" in map_case[0], "the per-entry case asserts nothing"
    assert "onOpen(sid)" in map_case[0].replace("\\", ""), (
        "the per-entry case no longer pins that the control opens the mapped id")


def test_the_run_row_in_the_sources_panel_renders_one_control_per_transcript():
    """Clause 2's render half, asserted against the page itself.

    Duplicating the vitest assertions here is the point, not an oversight: the
    gate resolves a clause's test node inside a pytest file, and the React render
    cannot be asserted in this suite (`environment: "node"`, no jsdom,
    `web/package.json` human-only). So the page's markup is read as text and the
    same structural facts are pinned from CI-reachable Python. A page that
    regressed — dropping the panel's callback, or rendering one transcript out of
    the list — fails here even if the vitest spec were deleted outright.
    """
    page = PAGE_TSX.read_text(encoding="utf-8")

    assert re.search(
        r"import\s*\{[^}]*runTranscriptIds[^}]*\}\s*from\s*'@/lib/runSessions'",
        page), "the page must derive the list through the pure module, not inline it"
    assert re.search(r"<SourcesPanel\b[^>]*onOpen=\{openInReader\}", page), (
        "the Sources panel is no longer handed the page's transcript-opening "
        "callback, so a run row has nothing to activate")
    assert re.search(r"<SourceCard\b[^>]*onOpen=\{onOpen\}", page), (
        "the callback no longer reaches the card that renders the run rows")
    assert "const transcripts = runTranscriptIds(run)" in page, (
        "the row list must come from the run, not from a field the row already shows")

    assert re.search(r"\{transcripts\.map\(sid =>", page), (
        "the controls are no longer mapped straight over the derived list, so a "
        "run that produced two transcripts could offer one")
    # Every shortcut that renders a prefix of the list while still mapping: the
    # first round of this guard caught `transcripts[0]` and would have passed
    # `transcripts.slice(0, 1).map(...)`, which drops the same transcripts.
    assert not re.search(
        r"transcripts\s*\[\d|transcripts\.(slice|at|filter|shift|pop|find)\b", page), (
        "the derived list is being narrowed before it is rendered; one control "
        "per *entry* is the clause, one per entry of a prefix is not")
    map_at = page.find("transcripts.map(sid => (")
    assert map_at > -1
    end = page.find("<span", map_at)
    region = page[map_at:end if end > -1 else len(page)]
    assert region.count("<button") == 1, (
        f"expected exactly one control per entry, found {region.count('<button')}")
    assert "key={sid}" in region, "the control must be keyed by the mapped id"
    assert "onClick={() => onOpen(sid)}" in region, (
        "the control must open the id this iteration mapped, not a fixed one")
    assert not re.search(r"transcripts\[\d", page), (
        "indexing the list is the shortcut that would pass a loose grep for "
        "`onOpen` while dropping every transcript after the first")


def test_the_architecture_page_names_the_surface_that_follows_the_join():
    """§7's "Worker run row → transcripts" bullet must name where a human
    actually follows the join — asserted as text, because the sentence is the
    artifact (the precedent for a text pin on this page is
    `tests/test_trajectory_extraction.py::test_the_architecture_page_attributes_browser_sessions_to_the_extension`).

    Clause 3 exists because the bullet claimed "one record of a run names the
    transcripts it produced" while the mechanism behind it was write-only: the
    only way to follow it was to hand-type an id into the reader. So what must
    not come back is the *absence of a surface* — the bullet naming the panel,
    the row, the reader, and the two functions that carry the value between them.
    """
    assert ARCH_PAGE.is_file(), f"missing {ARCH_PAGE}"
    text = " ".join(ARCH_PAGE.read_text(encoding="utf-8").split())
    # Positive control: a page whose §7 went missing would otherwise make the
    # block lookup below a pass on nothing.
    assert "## 7. Joins" in text, "`background-runs.md` no longer has a §7 to pin"

    blocks = [b for b in text.split("- **")
              if b.startswith("Worker run row → transcripts")]
    assert len(blocks) == 1, f"expected exactly one join bullet, found {len(blocks)}"
    block = blocks[0]

    assert re.search(r"Background tab →.*?→ (?:the )?Inner Voice reader", block), (
        f"the bullet no longer names the path a human walks: {block[:300]}")
    for surface in ("Background tab", "Sources", "run row", "Inner Voice reader"):
        assert surface in block, f"the surface `{surface}` dropped out of §7"
    assert "run_session_ids" in block and "runTranscriptIds" in block, (
        "the bullet must name the endpoint function that parses the field and "
        "the module that renders it, not just the panels")
    assert "session_ids" in block and "/api/workers/runs" in block, (
        "the join has to be named as the parsed field the endpoint ships")
    assert "meta_json" in block, (
        "the blob the join still lives in must stay named, or the reader cannot "
        "find it in SQL either")
    # Fix shape (b) — "say it is forensics-only and stop claiming a surface" —
    # must not come back as the bullet's present tense. The history sentence
    # naming what following the join used to cost stays: it is true and it is
    # past tense, which is what the two greps above cannot tell apart.
    assert "post-hoc sql" not in block.lower(), (
        "the bullet is back to describing the join as forensics-only, which is "
        "the write-only state this item was filed against")


def test_the_join_helper_is_pure_over_the_row_it_is_given():
    """No mutation: `list_runs` hands the same dict shape to the dashboard and
    to `agent_mcp/autoresearch.py`, and a helper that stamped its own key into
    those dicts would leak the field into callers that never asked for it."""
    row = {"run_id": "r", "meta_json": META_COLLECTED}

    assert router.run_session_ids(row) == ["20260918_161828_autonomy_fcb5"]
    assert set(row) == {"run_id", "meta_json"}


def test_the_sync_message_route_is_gone():
    """P13.6: `POST /api/message` — the synchronous one-shot chat route — had no
    caller anywhere (web/src, agent-services/, scripts/, workers, the tests
    other than its own pins), bypassed the session queue, and was the one turn
    path whose options, hooks and usage row every change had to be threaded
    through a fourth time. Deleted; the streaming route is the one chat path, a
    worker's loopback turn included. Asserted on the mounted router, so a route
    re-added under another function name is caught too."""
    from app.routers import messages

    methods = {(r.path, m) for r in messages.router.routes
               for m in getattr(r, "methods", set())}
    assert ("/api/message/stream", "POST") in methods  # positive control
    assert not any(path == "/api/message" for path, _m in methods), methods
    assert not hasattr(messages, "post_message")
    web = ROOT / "web" / "src"
    # The client builds paths as `${API_BASE}/message`; `api.sendMessage` was
    # defined there with no caller and went with the route.
    pattern = re.compile(r"""["'`]/api/message["'`?]|\$\{API_BASE\}/message[`?]""")
    callers = [p for p in web.rglob("*.ts*") if pattern.search(p.read_text(errors="replace"))]
    assert callers == [], callers
    assert pattern.search("fetch(`${API_BASE}/message`, {")  # the pattern bites
