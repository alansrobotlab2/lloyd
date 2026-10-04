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
from datetime import datetime, timedelta, timezone
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
from workers.queue import WorkQueue, new_run_id  # noqa: E402

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


# ---------------------------------------------------------------------------
# #1550 clause 4 — the panel and the alert state the same pause duration
#
# `(_pool, operator_paused)` carried the fact that a pool was held and nothing
# else: `status()` named the holder (`paused_by`) but not the instant, so no
# surface could answer "for how long", and the 16.5 h hold of 2026-09-24/25 was
# reconstructable only by comparing `watermarks.updated_at` with the `claimed_at`
# of the rows that finally drained. `paused_since` is that column, exposed.
# ---------------------------------------------------------------------------


def _pause_client(monkeypatch, tmp_path: Path, *, held_for_hours: float | None = None):
    """A client whose pool is a real `WorkerPool` over a real `WorkQueue`.

    Only `get_pool` is faked, and with a real pool: the field under test is read
    off the watermark table by the pool itself, so a stub pool would assert
    nothing about provenance. `held_for_hours` winds the persisted row's
    `updated_at` back, because `wm_set` stamps it with the wall clock.
    """
    import sqlite3

    from workers.pool import PAUSE_WM_KEY, PAUSE_WM_SOURCE, WorkerPool
    from workers.queue import WorkQueue

    queue = WorkQueue(tmp_path / "workers.db")
    pool = WorkerPool(queue, slots=1)
    since = None
    if held_for_hours is not None:
        pool.pause(True)
        since = (datetime.now(timezone.utc)
                 - timedelta(hours=held_for_hours)).isoformat()
        with sqlite3.connect(str(queue.db_path)) as conn:
            conn.execute("UPDATE watermarks SET updated_at=? "
                         "WHERE source=? AND key=?",
                         (since, PAUSE_WM_SOURCE, PAUSE_WM_KEY))
            conn.commit()
    monkeypatch.setattr(router, "get_pool", lambda: pool)
    app = _FastAPI()
    app.include_router(router.router)
    return _TestClient(app), pool, since


def test_the_pause_route_and_status_report_the_instant_the_pause_was_taken(
        monkeypatch, tmp_path):
    """A 16.5 h hold is one field, on every surface, all from one column.

    Three readers of the same instant: `GET /api/workers/pause` (the read the
    panel and any alerting script can poll without mutating anything), the POST
    that takes the pause (its response is what a click renders immediately), and
    `WorkerPool.status()`, which `/api/workers/status` embeds. They must agree
    byte for byte, because an alert and a panel that quote different durations
    for one hold is the disagreement #1550 is about.
    """
    client, pool, since = _pause_client(monkeypatch, tmp_path, held_for_hours=16.5)

    body = client.get("/api/workers/pause").json()
    assert body["paused"] is True and body["paused_by"] == ["operator"], body
    assert body["paused_since"] == since, body
    assert datetime.fromisoformat(body["paused_since"]).year > 2020, (
        "paused_since must be an ISO instant, not a duration")
    assert pool.status()["paused_since"] == since, (
        "`GET /api/workers/status` embeds status(); it must carry the same instant")

    resuming = client.post("/api/workers/pause", json={"paused": False}).json()
    assert resuming["paused"] is False and resuming["paused_since"] is None, resuming


def test_paused_since_is_only_the_operator_pause_instant(monkeypatch, tmp_path):
    """Nothing unpaused, and nothing persisted, can report an instant.

    The automod half of the pair is the interesting case: it holds the pool
    (`paused: true`, `paused_by: ["automod"]`) and is deliberately NOT persisted,
    because the promoter counts on a landing's own restart clearing it. So the
    honest answer there is None — a synthesized timestamp would claim a durable
    hold that a restart silently erases, which is the misreporting this item is
    about in the other direction.
    """
    client, pool, _ = _pause_client(monkeypatch, tmp_path)
    assert client.get("/api/workers/pause").json()["paused_since"] is None

    pool.pause(True, owner="automod")
    body = client.get("/api/workers/pause").json()
    assert body["paused"] is True and body["paused_by"] == ["automod"], body
    assert body["paused_since"] is None, (
        "an automod hold has no persisted instant to report; inventing one would "
        "assert a pause that the landing's restart does not survive")


# ── /api/workers/health carries the dispatch verdict (#1681) ───────────────


class _WmQueue:
    """A queue whose only interesting behaviour is the two enqueue watermarks.

    Stands in for `WorkQueue` so the route's four reads are all served from a
    dict: the point of these tests is what the endpoint puts NEXT TO `depth` and
    `health`, not whether sqlite returns rows.
    """

    def __init__(self, watermarks: dict[tuple[str, str], str]):
        self.wm = watermarks

    def wm_get(self, source, key):
        return self.wm.get((source, key))

    def run_rollup_by_source(self, since):
        # Both sources have identical rollups on purpose: whatever distinguishes
        # them in this response must come from `dispatch`, not from here.
        return {"live": {"runs": 3, "success": 3, "failed": 0},
                "dead": {"runs": 3, "success": 3, "failed": 0}}

    def depth_by_source(self):
        return {"live": {"queued": 0}, "dead": {"queued": 0}}

    def list_runs(self, source="", limit=10):
        return []


def _health_client(monkeypatch, tmp_path, watermarks, sources):
    client = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(router, "get_queue", lambda: _WmQueue(watermarks))
    monkeypatch.setattr(router, "CONFIG",
                        {"workers": {"sources": sources}}, raising=False)
    return client


def _row(body, name):
    return next(s for s in body["sources"] if s["name"] == name)


def test_health_distinguishes_a_stopped_dispatcher_from_a_quiet_one(
        monkeypatch, tmp_path):
    """The two rows that used to be identical, told apart.

    `depth` and `health` come off the `runs` table, so a source whose
    `enqueue_if_due` raises on every tick and one with genuinely nothing to do
    both keep the shape of their last good run — the same look the fleet watchdog
    would have, and that watchdog is dispatched by the very source it would
    catch. `dispatch` is the field that separates them, with the age and the
    verdict both on the page.
    """
    from datetime import datetime, timedelta, timezone
    from workers import dispatch_watch

    now = datetime.now(timezone.utc)
    sources = {
        "dead": {"enabled": True, "interval_seconds": 60},
        "live": {"enabled": True, "interval_seconds": 60},
    }
    wm = {
        ("live", dispatch_watch.ATTEMPT_WM_KEY): now.isoformat(),
        ("live", dispatch_watch.OK_WM_KEY): now.isoformat(),
        ("dead", dispatch_watch.ATTEMPT_WM_KEY): now.isoformat(),
        ("dead", dispatch_watch.OK_WM_KEY): (
            now - timedelta(seconds=400)).isoformat(),
    }
    q = _WmQueue(wm)
    # The pair that makes the two rows identical on the fields that already
    # existed: same depth, same outcome rollup.
    assert q.run_rollup_by_source("x")["live"] == q.run_rollup_by_source("x")["dead"]
    assert q.depth_by_source()["live"] == q.depth_by_source()["dead"]

    client = _health_client(monkeypatch, tmp_path, wm, sources)
    body = client.get("/api/workers/health?days=7&runs=0").json()
    dead, live = _row(body, "dead"), _row(body, "live")

    assert dead["depth"] == live["depth"] and dead["health"] == live["health"], (
        "the premise of this test is that the older fields cannot tell these apart")
    assert dead["dispatch"]["stalled"] is True, dead["dispatch"]
    assert dead["dispatch"]["state"] == dispatch_watch.STATE_STALLED
    assert dead["dispatch"]["age_seconds"] > 180, dead["dispatch"]
    assert dead["dispatch"]["threshold_seconds"] == 180.0
    assert live["dispatch"]["stalled"] is False, live["dispatch"]
    assert live["dispatch"]["state"] == dispatch_watch.STATE_OK


def test_health_dispatch_agrees_with_the_watch_and_survives_its_failure(
        monkeypatch, tmp_path):
    """The page must not invent its own verdict, and must not die on its newest
    field. Agreement matters because the same threshold drives an announcement:
    a dashboard that says `ok` beside an alert that says `stalled` teaches a
    person to disbelieve one of them."""
    from datetime import datetime, timezone
    from workers import dispatch_watch

    now = datetime.now(timezone.utc)
    wm = {("dead", dispatch_watch.ATTEMPT_WM_KEY): now.isoformat()}
    sources = {"dead": {"enabled": True, "interval_seconds": 60}}
    client = _health_client(monkeypatch, tmp_path, wm, sources)
    body = client.get("/api/workers/health?days=7&runs=0").json()
    row = _row(body, "dead")["dispatch"]
    direct = dispatch_watch.dispatch_health(_WmQueue(wm), sources)["dead"]
    # `age_seconds` is measured at read time, so the two calls differ by the
    # milliseconds between them; everything that carries a verdict is compared
    # exactly.
    assert {k: v for k, v in row.items() if k != "age_seconds"} == \
        {k: v for k, v in {**direct, "name": "dead"}.items() if k != "age_seconds"}, (
            "the route and the watch must return the same verdict, not two readings")
    assert abs(row["age_seconds"] - direct["age_seconds"]) < 1.0
    assert row["state"] == dispatch_watch.STATE_PENDING, row

    def _boom(*a, **k):
        raise RuntimeError("watermarks unreadable")

    monkeypatch.setattr(dispatch_watch, "dispatch_health", _boom)
    body2 = client.get("/api/workers/health?days=7&runs=0").json()
    assert _row(body2, "dead")["dispatch"] is None, (
        "an unreadable dispatch section must not 500 the health page")
    assert body2["sources"], "and the rest of the page still has its rows"

# ── /api/workers/health counts runs whose matrix never finished (#1687) ─────

_STOPPED = '{"round_id": "%s", "deadline_stopped": true, ' \
           '"tasks_not_reached": ["bench_015", "bench_010"]}'
_FINISHED = '{"round_id": "%s", "deadline_stopped": false, ' \
            '"matrix_dropped_tasks": [], "promoted": false}'
# #1857: the third shape the rollup now has to tell apart — the pre-trial cost
# projection (#1605, with #1715's shrink fallback) dropped four tasks and the
# round then ran what was left to the end, so `deadline_stopped` is false and
# this is the fix working, not the failure #1687's alarm was built for.
_SHRUNK = '{"round_id": "%s", "deadline_stopped": false, ' \
          '"matrix_projection": {"fits": true}, "tasks_not_reached": [], ' \
          '"matrix_dropped_tasks": ["bench_017_audit_unresolved_task_skills", ' \
          '"bench_016_audit_skill_dead_paths"]}'
# Shrunk up front AND cut at the deadline: `run_round.py:1428` derives
# `deadline_stopped` from `tasks_not_reached` while `:1753` writes
# `matrix_dropped_tasks` independently, so both keys can be on one round.
_SHRUNK_AND_CUT = '{"round_id": "%s", "deadline_stopped": true, ' \
                  '"tasks_not_reached": ["bench_010"], ' \
                  '"matrix_dropped_tasks": ["bench_017_audit_unresolved_task_skills"]}'


def _matrix_client(monkeypatch, tmp_path,
                   in_window: dict[str, list[tuple[bool | str, str]]],
                   *, outside_window: list[str] = ()):
    """A health endpoint over a REAL `WorkQueue`, not a stand-in.

    `in_window` maps a source name to `(cause, round_id)` pairs, one per run
    completed now, where `cause` is `True` for a deadline cut, `False` for a
    round that ran its whole matrix, or a JSON template naming some other shape
    (`_SHRUNK`, `_SHRUNK_AND_CUT`). `outside_window` names sources that get
    exactly one `deadline_stopped` run dated 2020, so they are configured but
    unseen by the window. The count these tests pin is computed by SQL over the
    stored `response_json`, which the hand-written rollup dicts the fake queues
    in this file return cannot exercise: it needs the real table, the real query
    and the real serialised body.
    """
    queue = WorkQueue(tmp_path / "workers.db")

    def record(source, response, completed_at):
        queue.record_run(run_id=new_run_id(source), queue_id=None, source=source,
                         status="success", started_at=completed_at,
                         completed_at=completed_at, duration_seconds=1500.0,
                         response_json=response)

    for name, runs in in_window.items():
        for cause, round_id in runs:
            template = (cause if isinstance(cause, str)
                        else _STOPPED if cause else _FINISHED)
            record(name, template % round_id,
                   datetime.now(timezone.utc).isoformat())
    for name in outside_window:
        record(name, _STOPPED % "R_old", "2020-01-01T00:01:00+00:00")

    client = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(router, "get_queue", lambda: queue)
    sources = {n: {"enabled": True, "interval_seconds": 14400}
               for n in sorted(set(in_window) | set(outside_window))}
    monkeypatch.setattr(router, "CONFIG", {"workers": {"sources": sources}},
                        raising=False)
    return client


def _health_of(client, name: str):
    body = client.get("/api/workers/health?days=7&runs=0").json()
    return next(s for s in body["sources"] if s["name"] == name)["health"]


def test_the_health_block_reports_matrix_incomplete_beside_ok(monkeypatch, tmp_path):
    """Clause 2: five "successful" runs that measured nothing report 5, not ok: 5.

    Every autoresearch round in the window completed and stopped before the end
    of its trial matrix, so this block read `{total: 5, ok: 5, failed: 0,
    fail_rate: 0.0}` — the exact shape of a source doing good work, while in
    fact it had promoted nothing at all since the cap fix. The count now travels
    in the same block, beside `ok`.
    """
    client = _matrix_client(monkeypatch, tmp_path,
                            {"autoresearch": [(True, f"R_{i}") for i in range(5)]})

    health = _health_of(client, "autoresearch")
    assert (health["total"], health["ok"], health["failed"]) == (5, 5, 0)
    assert health["unfinished_matrix"] == 5, \
        "all five stopped short of the end of their matrix, and `ok: 5` alone " \
        "is what made that invisible"
    assert health["unfinished_matrix"] <= health["total"], \
        "it is a subset of the window's runs, never a second tally"


def test_the_health_incomplete_count_is_never_invented_for_a_source_with_no_run(
        monkeypatch, tmp_path):
    """Clause 3: `unfinished_matrix: 0` must mean "looked, none", never "no data".

    Three readings in one response, because the distinction is the whole use of
    the number: 1 of 3 runs stopped for `autoresearch`, a genuine 0 of 2 for
    `youtube-digest`, and `health: null` for `arch-review`, whose one stopped
    round predates the window. A zero standing in for no observation is the same
    error this endpoint already refuses for `fail_rate`, which it reports as
    null rather than 0% over zero runs.
    """
    client = _matrix_client(
        monkeypatch, tmp_path,
        {"autoresearch": [(True, "R_a"), (False, "R_b"), (False, "R_c")],
         "youtube-digest": [(False, "Y_0"), (False, "Y_1")]},
        outside_window=["arch-review"])

    stopped = _health_of(client, "autoresearch")
    assert (stopped["unfinished_matrix"], stopped["total"]) == (1, 3), \
        "the count is emitted beside the window's run total, so 1 of 3 reads " \
        "as a fraction of real traffic rather than as a bare number"
    clean = _health_of(client, "youtube-digest")
    assert (clean["unfinished_matrix"], clean["total"]) == (0, 2), \
        "a zero here is an observation: two runs, both finished their matrix"
    assert _health_of(client, "arch-review") is None, \
        "a source with no run in the window gets no block at all, so there is " \
        "no zero anywhere for a reader to mistake for 'checked, all complete'"


def test_the_health_block_names_which_kind_of_matrix_did_not_finish(
        monkeypatch, tmp_path):
    """#1857 clause 5: `deadline_cut` and `matrix_shrunk` reach the payload, per
    source, beside the window's `total` — and a source the window never saw
    still gets no block at all.

    `app/routers/workers.py:263` passes `rollup.get(name)` into `health`
    verbatim, which is why the router needed no change and why this test is the
    only thing standing between the new SQL columns and a reader who cannot see
    them: the seam is the serialised body, so it is the body that gets asserted.

    The numbers are the live shape measured 2026-09-29 over the 7-day window:
    autoresearch had 7 deadline cuts and 9 rounds whose matrix the projection
    shrank before any trial ran, and the deadline half had been dead for a day
    while the union still read 16. Here `autoresearch` is 2 cuts + 1 shrink of 4,
    `youtube-digest` is the post-fix case — 2 shrunk rounds, 0 cuts — and
    `arch-review` has nothing in the window. A `0` in `deadline_cut` is only
    evidence if `total` says a run was actually looked at, which is what the
    `arch-review` half pins: no dict, so no invented zero for either new key.
    """
    client = _matrix_client(
        monkeypatch, tmp_path,
        {"autoresearch": [(True, "R_cut_1"), (_SHRUNK_AND_CUT, "R_both"),
                          (_SHRUNK, "R_shrunk"), (False, "R_clean")],
         "youtube-digest": [(_SHRUNK, "Y_0"), (_SHRUNK, "Y_1")]},
        outside_window=["arch-review"])

    health = _health_of(client, "autoresearch")
    assert {"total", "ok", "deadline_cut", "matrix_shrunk",
            "unfinished_matrix"} <= set(health), \
        f"both new keys must travel in the same block as ok and total: {health}"
    assert (health["total"], health["ok"]) == (4, 4), \
        "every one of these runs returned cleanly, which is the reading that " \
        "made the #1687 alarm necessary"
    assert (health["deadline_cut"], health["matrix_shrunk"]) == (2, 1), \
        "R_both carries both keys and is a deadline cut alone, so the two " \
        "counts are disjoint in the payload as well as in the SQL"
    assert health["unfinished_matrix"] == health["deadline_cut"] + \
        health["matrix_shrunk"] == 3, \
        "the union keeps its old meaning beside its two causes"

    shrunk = _health_of(client, "youtube-digest")
    assert (shrunk["total"], shrunk["deadline_cut"], shrunk["matrix_shrunk"]) == (
        2, 0, 2), \
        "a source that only ever shrinks its matrix must read deadline_cut 0 " \
        "against a real total — that is the whole point of the split"
    assert _health_of(client, "arch-review") is None, \
        "and with no run in the window there is still no block, so neither new " \
        "key can read as a zero that means 'looked, nothing incomplete'"


# ---------------------------------------------------------------------------
# #1769 — relabelling a staged note as uncalibrated must never block the one
# decision that is a human's to make.
#
# `scripts/maintenance/relabel_stale_bench_calibration.py` puts 119 pre-#1710
# notes into `review_status: uncalibrated`, and the promotion UI reads that
# front matter through `GET /api/workers/pending`. The label exists to stop a
# human promoting on a false band verdict; it must not become a gate that stops
# a human promoting anyway, on the strength of having read the candidate task
# themselves. The route has never read the staged note's `review_status` (it
# stamps `promoted` on the LANDED copy and unlinks the staged one), so this is a
# pin on behaviour as it stands — the regression it prevents is someone adding
# the check later and silently stranding the relabelled queue.
# ---------------------------------------------------------------------------

def test_an_uncalibrated_note_still_promotes_its_task(monkeypatch, tmp_path):
    """Clause 4's pin of #1769: a note the relabeller touched, whose task block IS
    gradeable, lands exactly as the `pending` note in
    `test_a_staged_candidate_lands_its_bench_task_block` does.

    The staging envelope carries the whole stale shape — `status:
    stale_envelope`, `in_band: null`, `measured_against`, no `task_id` — because
    the thing being pinned is that none of those fields is consulted on the way
    through.
    """
    fm = {**STAGING_FM, "review_status": "uncalibrated",
          "calibration": {**STAGING_FM["calibration"],
                          "status": "stale_envelope", "in_band": None,
                          "measured_against": "staging envelope (pre-#1710)"}}
    src = _staged(tmp_path, "060454-mined-from-run-38-20260911-050055.md",
                  staging_fm=fm)
    assert _yaml.safe_load(src.read_text().split("---", 2)[1])["review_status"] \
        == "uncalibrated", "the fixture is not the note the relabeller produces"

    out = _promote(_client(monkeypatch, tmp_path), src)
    assert out["status"] == 200, (
        f"a relabelled note was refused promotion: {out}")

    dest = tmp_path / "vault" / "lloyd" / "bench"
    tasks = load_bench_tasks(dest)
    assert [str(t.get("id")) for t in tasks] == [TASK_ID], tasks
    landed = (dest / f"{TASK_ID}.md").read_text(encoding="utf-8")
    assert "review_status: promoted" in landed, (
        "the landed copy must carry the human's decision")
    for phrase in ("uncalibrated", "stale_envelope", "staging envelope", "null"):
        assert phrase not in landed, (
            f"the staging label ({phrase!r}) rode along into the graded bench, "
            "where the bench loader would hand it to a runner as task metadata")
    assert not src.exists(), "the staged note is consumed by the move, as ever"


# ── #1970: the Background per-source row renders #1857's split ──────────────
#
# Pinned as source text for the reason spelled out above
# `test_the_background_tab_reads_the_join_through_a_pure_module`: vitest runs
# `environment: "node"` with no renderer, so a component's markup has no render
# test, and `web/package.json` is human-only.

API_TS = ROOT / "web/src/api.ts"


def _health_type() -> str:
    """The body of `WorkerSourceHealth.health`'s object type in api.ts."""
    api = API_TS.read_text(encoding="utf-8")
    iface = api.split("export interface WorkerSourceHealth {", 1)[1]
    body = iface.split("  health: {", 1)[1].split("  } | null", 1)[0]
    assert "total: number" in body and "last_completed" in body, (
        "the span extracted from api.ts is not the health object")
    return body


def _source_row() -> tuple[str, str]:
    """(the `h === null` branch, the has-runs branch) of the per-source flex row."""
    page = PAGE_TSX.read_text(encoding="utf-8")
    row = page.split("{h === null ? (", 1)[1].split("{source.recent.length > 0", 1)[0]
    null_branch, runs_branch = row.split(") : (", 1)
    assert "{h.total} run" in runs_branch, "the has-runs branch lost its run count"
    return null_branch, runs_branch


def _code(src: str) -> str:
    """`src` with `//` line comments and JSX `{/* */}` comments removed."""
    import re
    src = re.sub(r"\{/\*.*?\*/\}", "", src, flags=re.S)
    return "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("//"))


def test_the_health_type_declares_the_split_as_optional():
    body = _code(_health_type())
    for key in ("unfinished_matrix", "deadline_cut", "matrix_shrunk"):
        assert f"{key}?: number" in body, (
            f"{key} is not an optional number on WorkerSourceHealth.health — a "
            "required field would read a pre-#1857 backend as a type error, and a "
            "defaulted one as a healthy 0")


def test_the_row_renders_a_rose_deadline_cut_only_above_zero():
    import re
    _, runs = _source_row()
    m = re.search(r"\{\(h\.deadline_cut \?\? 0\) > 0 && \(\s*"
                  r"<span className=\"([^\"]*)\">\{h\.deadline_cut\} [^<]*</span>", runs)
    assert m, "no span guarded by `deadline_cut > 0` in the per-source row"
    assert "text-rose-400" in m.group(1)
    assert runs.index("{h.total} run") < m.start(), (
        "the deadline-cut span must sit beside (after) the run count, never alone")


def test_the_row_renders_a_muted_matrix_shrunk_only_above_zero():
    import re
    _, runs = _source_row()
    m = re.search(r"\{\(h\.matrix_shrunk \?\? 0\) > 0 && \(\s*"
                  r"<span([^>]*)>\{h\.matrix_shrunk\} [^<]*</span>", runs)
    assert m, "no span guarded by `matrix_shrunk > 0` in the per-source row"
    assert "rose" not in m.group(1) and "className" not in m.group(1), (
        "matrix_shrunk is context, not the alarm: it inherits the row's "
        "text-muted-foreground and takes no colour of its own")
    page = PAGE_TSX.read_text(encoding="utf-8")
    container = page.split("{h === null ? (", 1)[0].rsplit("<div className=\"", 1)[1]
    assert "text-muted-foreground" in container.split("\"", 1)[0]


def test_unfinished_matrix_is_rendered_nowhere():
    """The union is declared on the wire type and read by nothing.

    The type declaration in api.ts is the one allowed occurrence; any other use
    in web/src — a component, a helper, a second type — is a render path.
    """
    hits = {}
    for path in sorted((ROOT / "web/src").rglob("*")):
        if path.suffix not in (".ts", ".tsx") or not path.is_file():
            continue
        n = _code(path.read_text(encoding="utf-8")).count("unfinished_matrix")
        if n:
            hits[str(path.relative_to(ROOT))] = n
    assert hits == {"web/src/api.ts": 1}, hits
    assert "unfinished_matrix?: number" in _code(_health_type())


def test_a_source_with_no_runs_still_reads_no_runs_and_neither_span():
    null_branch, _ = _source_row()
    spans = [ln.strip() for ln in _code(null_branch).splitlines() if "<span" in ln]
    assert spans == ["<span>no runs in the window</span>"], spans
    assert "deadline_cut" not in null_branch and "matrix_shrunk" not in null_branch


# ── /api/workers/health names its own window clamp (#2127) ───────────────────
#
# `days` is a request, not a measurement, and until now this endpoint reported
# nothing to check it against. Measured on the live box 2026-10-03:
# `?days=90` returned top-level keys `['days', 'initialized', 'sources']` over a
# `workers.db` whose oldest run row was 10.8 days old, so every per-source
# `fail_rate` and `gpu_hours` in that payload was a ~10.8-day figure wearing a
# 90-day label — while `/api/autonomy/health`, over the same table, named
# `fleet.window_clamped_to_hours: 257.81` (#1401). These tests drive the real
# route over a real `WorkQueue` sqlite file: the clamp is a
# `SELECT MIN(completed_at) FROM runs` with no date predicate, and a stubbed
# queue can return whatever its caller asks for, which proves nothing about the
# read being unfiltered.


def _clamp_client(monkeypatch, tmp_path: Path, name: str = "workers-health.db"):
    """The real route over a real queue file with one configured source."""
    from workers.queue import WorkQueue

    q = WorkQueue(tmp_path / name)
    client = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(router, "get_queue", lambda: q)
    monkeypatch.setattr(router, "CONFIG", {"workers": {"sources": {
        "probe": {"enabled": True, "interval_seconds": 60}}}}, raising=False)
    return client, q


def _seed_run(q, when, *, run_id: str = "run-1", source: str = "probe",
              status: str = "success") -> str:
    """One `runs` row stamped exactly `when`, returning the stamp it stored.

    `record_run` takes an explicit `completed_at`, which is the only way a
    ten-day-old store exists in a test that takes milliseconds.
    """
    stamp = when.isoformat()
    q.record_run(run_id=run_id, queue_id=None, source=source, status=status,
                 started_at=stamp, completed_at=stamp, duration_seconds=60.0,
                 summary="ok")
    return stamp


def test_days_beyond_the_store_names_the_hours_the_verdict_covers(
        monkeypatch, tmp_path):
    """Clause 1 + 2: the clamp is the span the numbers really rest on.

    A store holding one run at `now - 10 days`, asked for 90 days: the answer
    must say 240 hours, not silently agree it covered the 2160 it was asked for.
    """
    client, q = _clamp_client(monkeypatch, tmp_path)
    stamp = _seed_run(q, datetime.now(timezone.utc) - timedelta(days=10))

    body = client.get("/api/workers/health?days=90").json()

    assert body["initialized"] is True and body["days"] == 90, body
    assert body["oldest_input"] == stamp, body
    assert isinstance(body["window_clamped_to_hours"], float), body
    assert abs(body["window_clamped_to_hours"] - 240.0) < 0.05, (
        f"the rollup spans ~10 days, so the field must say ~240 h and not the "
        f"2160 h `days=90` asked for: {body['window_clamped_to_hours']}")


def test_a_store_that_covers_the_window_keeps_both_fields_null(
        monkeypatch, tmp_path):
    """Clause 1, the null half — and the discriminator for a window-min build.

    `days=7` over a store whose rows are 10 days and 1 day old: the window has
    one row in it and the store reaches back past the window edge, so the
    verdict genuinely spans seven days and both fields go. An implementation
    that derived the clamp from the WINDOW-filtered rollup would see its oldest
    in-window row (1 day old) here and report a 24-hour clamp — the wrong answer
    on a healthy store, which is the other way this pair could mislead.
    """
    client, q = _clamp_client(monkeypatch, tmp_path)
    now = datetime.now(timezone.utc)
    _seed_run(q, now - timedelta(days=10), run_id="old")
    _seed_run(q, now - timedelta(days=1), run_id="new")

    body = client.get("/api/workers/health?days=7").json()

    assert body["days"] == 7 and body["initialized"] is True, body
    assert body["oldest_input"] is None, body
    assert body["window_clamped_to_hours"] is None, body
    assert body["sources"][0]["health"]["total"] == 1, (
        "the window is not empty here; the nulls mean it is fully covered")


def test_a_window_holding_no_run_still_names_the_store_age_at_zero(
        monkeypatch, tmp_path):
    """Clause 3: an empty window is never shaped like a clean bill of health.

    Every row predates the window, so `run_rollup_by_source` returns nothing —
    the exact store the clamp exists for. A min over that rollup is `None`, and
    `None` is what a covered window also prints; the unfiltered store read is
    the only thing that tells these two apart.
    """
    client, q = _clamp_client(monkeypatch, tmp_path)
    stamp = _seed_run(q, datetime.now(timezone.utc) - timedelta(days=30))

    body = client.get("/api/workers/health?days=7").json()

    assert body["oldest_input"] == stamp, body
    assert body["window_clamped_to_hours"] == 0.0, body
    assert body["sources"][0]["health"] is None, (
        "the row itself still reads as no runs in the window: the clamp pair is "
        "the only signal that the store is old rather than empty")


def test_the_window_edge_and_the_clamp_share_one_clock_reading(
        monkeypatch, tmp_path):
    """Clause 3's second half: one `now`, feeding both.

    Two readings would let the clamp report a span the query did not run over,
    and nothing else in the payload could reveal the disagreement. Pinning the
    clock turns `~240 h` into `240.0` exactly and makes the count of clock reads
    an observable rather than a property of the code.
    """
    client, q = _clamp_client(monkeypatch, tmp_path)
    pin = datetime(2026, 10, 3, 12, 0, 0, tzinfo=timezone.utc)
    stamp = _seed_run(q, pin - timedelta(days=10))
    reads: list[int] = []

    class _Pinned(datetime):
        @classmethod
        def now(cls, tz=None):
            reads.append(1)
            instant = cls(2026, 10, 3, 12, 0, 0, tzinfo=timezone.utc)
            return instant if tz is None else instant.astimezone(tz)

    monkeypatch.setattr(router, "datetime", _Pinned)

    body = client.get("/api/workers/health?days=90").json()

    assert len(reads) == 1, (
        f"the route read the clock {len(reads)} times; the clamp is only "
        "trustworthy if the `since` bound and the clamp come from one reading")
    assert body["oldest_input"] == stamp, body
    assert body["window_clamped_to_hours"] == 240.0, body


def test_the_clamp_arrives_without_renaming_anything_else(
        monkeypatch, tmp_path):
    """Clause 4 at the route: the envelope grew by exactly two keys.

    Top-level set-equality is the pin — an additive change may add, and the
    per-source rows must come back with the eleven keys `web/src/api.ts` types
    and the Background tab reads, unchanged.
    """
    client, q = _clamp_client(monkeypatch, tmp_path)
    _seed_run(q, datetime.now(timezone.utc) - timedelta(hours=6))

    body = client.get("/api/workers/health?days=7&runs=1").json()

    assert set(body) == {"initialized", "days", "oldest_input",
                         "window_clamped_to_hours", "sources"}, sorted(body)
    row = body["sources"][0]
    # #2185 grew this row by exactly three keys, additively: the queue's wait, the
    # count over the bound, and the bound the count was taken against so the number
    # cannot be read against a different threshold later. The eleven keys
    # `web/src/api.ts` types and the Background tab reads are otherwise untouched —
    # that is what set-equality here is for, and a fourth key appearing is a rename
    # or a drop wearing an addition's clothes.
    assert set(row) == {"name", "configured", "enabled", "inner_voice",
                        "interval_seconds", "max_inflight", "priority", "depth",
                        "health", "dispatch", "recent",
                        "pending_wait_max_seconds",
                        "pending_wait_over_bound_count",
                        "pending_wait_bound_seconds"}, sorted(row)
    assert row["name"] == "probe" and row["configured"] is True
    health = row["health"]
    assert health["total"] == 1 and health["ok"] == 1, health
    assert health["fail_rate"] == 0.0, health
    assert health["gpu_hours"] == round(60.0 / 3600.0, 2), health
    assert len(row["recent"]) == 1 and row["recent"][0]["status"] == "success"


def test_the_store_age_read_is_unfiltered_across_sources(monkeypatch, tmp_path):
    """The queue read behind the clamp, pinned at its own level.

    `oldest_run_completed_at()` with no source is the whole `runs` table; with a
    source it stays what `/api/autonomy/health` has always needed — that source
    alone. Both matter: the autonomy route must not silently start reading the
    fleet's age as its own.
    """
    from workers.queue import WorkQueue

    q = WorkQueue(tmp_path / "oldest.db")
    now = datetime.now(timezone.utc)
    old_stamp = _seed_run(q, now - timedelta(days=30), run_id="old", source="probe")
    _seed_run(q, now - timedelta(days=2), run_id="new", source="other")

    fleet_oldest = q.oldest_run_completed_at()
    per_source = q.oldest_run_completed_at("other")

    assert fleet_oldest == old_stamp, (fleet_oldest, old_stamp)
    assert per_source > fleet_oldest, (
        "the all-source read must be the oldest row in the table, older than any "
        "single source's: %r vs %r" % (fleet_oldest, per_source))
    assert q.oldest_run_completed_at("nobody") is None, (
        "a source with no rows has no age, which is not the same as age zero")

    empty = WorkQueue(tmp_path / "empty.db")
    assert empty.oldest_run_completed_at() is None, (
        "an empty store is nameless, not clamped to zero")



# ── #2185: the queue's WAIT, not just its depth ──────────────────────────────
#
# `depth` answers how many rows wait and `health` how the last runs went. Neither
# answers how long the oldest request has been standing, which is the figure
# #1526's four-day outage was finally diagnosed with: row 515 of the live queue went
# from `enqueued_at` 2026-09-25T08:00:33Z to `claimed_at` 19:10:40Z — 11 h 10 min on
# source `scheduled-task`, with no failing run anywhere to point at, and autonomy #24
# (`frequency: 6x-daily`) lost two slots the same way. These tests pin the two numbers
# at the route, the retry-back-off exclusion, and the fact that the route reaches them
# through `WorkQueue` rather than through SQL of its own.


def _backdate(q, queue_id: int, *, seconds: float) -> None:
    """Move one queued row's `enqueued_at` back by `seconds`, test-local SQL.

    `enqueue()` has no timestamp parameter and the leg's own clock is not the
    subject, so a queued row that has waited two hours can only be made by stamping
    it. This is the test's database, not the router's: clause 4 keeps queries out of
    `app/routers/workers.py`, and a fixture reaching sqlite directly is how every
    other back-dated row in this file is made (`_seed_run` takes an explicit
    `completed_at` for the same reason).
    """
    import sqlite3

    stamp = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    with sqlite3.connect(str(q.db_path)) as conn:
        conn.execute("UPDATE queue SET enqueued_at=? WHERE id=?", (stamp, queue_id))


def _park(q, queue_id: int, *, seconds_ahead: float) -> None:
    """Give one queued row a `not_before` in the future, as `mark_failed` does."""
    import sqlite3

    stamp = (datetime.now(timezone.utc) + timedelta(seconds=seconds_ahead)).isoformat()
    with sqlite3.connect(str(q.db_path)) as conn:
        conn.execute("UPDATE queue SET not_before=? WHERE id=?", (stamp, queue_id))


def _wait_row(body, name: str = "probe") -> dict:
    rows = [s for s in body["sources"] if s["name"] == name]
    assert len(rows) == 1, f"{name} appears {len(rows)} times in {body['sources']}"
    return rows[0]


def test_a_queued_row_two_hours_old_reports_its_wait_and_one_over_the_bound(
        monkeypatch, tmp_path):
    """Clause 1: the wait is a number on the page, at the bound the page states.

    One queued row back-dated 2 h. The figure must be ~7200 s and exactly one row
    must be over the bound, which is 3600 s and travels in the payload rather than
    living only in `workers/queue.py` — otherwise a reader looking at the count
    cannot say what it was counted against.
    """
    client, q = _clamp_client(monkeypatch, tmp_path)
    q.enqueue("probe", "task", payload={})
    _backdate(q, 1, seconds=7200)

    body = client.get("/api/workers/health?days=7&runs=0").json()
    row = _wait_row(body)

    assert 7190 <= row["pending_wait_max_seconds"] <= 7230, row
    assert row["pending_wait_over_bound_count"] == 1, row
    assert row["pending_wait_bound_seconds"] == 3600, row
    assert row["depth"]["queued"] == 1, \
        "the two figures describe the same row the depth already counted"


def test_a_queued_row_under_the_bound_moves_the_wait_but_not_the_count(
        monkeypatch, tmp_path):
    """The pair is not one number twice: 30 minutes of wait is not a starved pool.

    Per-day maximum claim latency on this box has read 0.00-0.30 h since 2026-09-26,
    so the bound sits an order of magnitude above the pool's 1800 s duration fallback
    and a healthy queue reads zero. Without this case a rising `max` and a real
    starvation alarm would be indistinguishable in the payload.
    """
    client, q = _clamp_client(monkeypatch, tmp_path)
    q.enqueue("probe", "task", payload={})
    _backdate(q, 1, seconds=1800)

    row = _wait_row(client.get("/api/workers/health?days=7&runs=0").json())

    assert 1790 <= row["pending_wait_max_seconds"] <= 1830, row
    assert row["pending_wait_over_bound_count"] == 0, row


def test_a_source_with_nothing_queued_reads_zeros_and_the_route_still_answers(
        monkeypatch, tmp_path):
    """Clause 2: absence of a wait is zero, never null.

    One run in the window, nothing queued: the source still appears on the page, the
    route still answers 200, and both figures are numeric zeros — a dashboard
    rendering `wait * something` must not meet `None` on the healthy path, which is
    the common path.
    """
    client, q = _clamp_client(monkeypatch, tmp_path)
    _seed_run(q, datetime.now(timezone.utc) - timedelta(hours=6))

    response = client.get("/api/workers/health?days=7&runs=0")
    assert response.status_code == 200, response.status_code
    row = _wait_row(response.json())

    assert row["pending_wait_max_seconds"] == 0.0, row
    assert row["pending_wait_over_bound_count"] == 0, row
    assert row["pending_wait_bound_seconds"] == 3600, row


def test_a_back_off_parked_row_is_not_reported_as_waiting(monkeypatch, tmp_path):
    """Clause 3: `not_before` in the future is a row waiting BY DESIGN.

    `mark_failed` parks a retry hours ahead, and that row is in state 'queued' like
    any starved one. Counting it would let one failing item with a long back-off
    report its whole source as starving — the exact false alarm that would get this
    field ignored. `workers/fleet_watchdog.py`'s own `_queue_starving` skips these
    rows for the same reason; the route and the watchdog must not disagree about what
    counts as waiting. A second, genuinely-old row is enqueued so the test proves the
    parked row was excluded rather than the whole source skipped.
    """
    client, q = _clamp_client(monkeypatch, tmp_path)
    q.enqueue("probe", "task", payload={})
    _backdate(q, 1, seconds=9000)
    _park(q, 1, seconds_ahead=7200)
    q.enqueue("probe", "task", payload={})
    _backdate(q, 2, seconds=4000)

    row = _wait_row(client.get("/api/workers/health?days=7&runs=0").json())

    assert row["depth"]["queued"] == 2, row
    assert row["pending_wait_max_seconds"] < 5000, row
    assert row["pending_wait_over_bound_count"] == 1, row


def test_the_route_reaches_the_wait_through_the_queue_not_through_sql():
    """Clause 4: one reader of `queue`, so `queued` cannot mean two things.

    The four assertions are the ones
    `tests/test_dashboard_sections.py::test_the_router_reaches_run_outcomes_only_through_workqueue`
    applies to `app/routers/dashboard.py`. No equivalent existed for
    `app/routers/workers.py`: triage found `sqlite3` in this test file only as
    test-local connects (`:853`, `:865`), which say nothing about the router's own
    text. Positive control first, because an absence read out of an empty or wrong
    file looks identical to an absence read out of the right one for the right
    reason.
    """
    src = (Path(__file__).resolve().parents[1] / "app" / "routers" / "workers.py").read_text()
    lines = src.count("\n")
    assert lines > 500, f"positive control: read the wrong file, only {lines} lines"
    assert "pending_wait_by_source" in src, "must call through WorkQueue"
    # Needle-level positive control: the three needles are spelled right, because the
    # file that legitimately holds this SQL is `workers/queue.py`, and the same
    # needles must be PRESENT there. A typo'd needle reads as absent from the router
    # and green from the queue, which is the one way this node could pass for nothing
    # — and a review pass on this round read the absence as a false zero on the theory
    # that `from fastapi.responses import JSONResponse` satisfies `"SELECT "`. It does
    # not (`'SELECT' in 'JSONRESPONSE'` is False), but the control below is what makes
    # that argument unnecessary rather than merely correct.
    queue_src = (Path(__file__).resolve().parents[1] / "workers" / "queue.py").read_text()
    for needle in ("SELECT ", "sqlite3", "execute("):
        assert needle in queue_src, (
            f"positive control: {needle!r} matches nothing anywhere, so its absence "
            f"in the router proves nothing")
    assert "SELECT " not in src.upper(), "no SQL of its own in the router"
    assert "sqlite3" not in src, "no direct sqlite connection in the router"
    assert "execute(" not in src, "no query executed from the router"


def test_the_wait_is_one_query_and_carries_the_bound_default(monkeypatch, tmp_path):
    """The read behind the field, at its own level.

    A method on `WorkQueue` rather than a router-side computation, so the figure the
    dashboard shows is the figure `/api/workers/status` would show if it ever grows
    one. And the bound has a default — the route passes none — because the number
    that makes a wait reportable is one decision in one place, not a per-caller
    argument two callers could set differently. `bound_seconds=` stays on the
    signature for a caller asking a DIFFERENT question (an experiment, a stricter
    alert under review); what the clause forbids is the route choosing its own.
    """
    from workers.queue import PENDING_WAIT_BOUND_SECONDS, WorkQueue

    q = WorkQueue(tmp_path / "wait.db")
    assert q.pending_wait_by_source() == {}, "nothing queued: no source key at all"

    q.enqueue("scheduled-task", "task", payload={})    # row 1: ~11 h, row 515's shape
    q.enqueue("autocode", "task", payload={})          # row 2: just over the bound
    q.enqueue("autoresearch", "task", payload={})      # row 3: two minutes old
    _backdate(q, 1, seconds=40_000)
    _backdate(q, 2, seconds=PENDING_WAIT_BOUND_SECONDS + 5)
    _backdate(q, 3, seconds=120)

    out = q.pending_wait_by_source()
    assert sorted(out) == ["autocode", "autoresearch", "scheduled-task"], out
    assert 39_990 <= out["scheduled-task"]["max_wait_seconds"] <= 40_010, out
    assert out["scheduled-task"]["over_bound_count"] == 1, out
    assert out["autocode"]["over_bound_count"] == 1, \
        "3 605 s is over a 3 600 s bound by five seconds — the boundary is `>`"
    assert out["autoresearch"]["over_bound_count"] == 0, \
        "two minutes of wait is not starvation: the max moves, the count does not"

    assert PENDING_WAIT_BOUND_SECONDS == 3600
    # A keyword assert has to be one a broken method cannot pass. Asserting on the
    # 40 000 s row reads 1 whether `bound_seconds` was honoured or ignored, because
    # that row is over both 10 and the 3 600 default; `autoresearch`'s 120 s row is
    # over 10 and UNDER the default, so its count only flips if the argument is used.
    # The route passes no bound, which is the point: one decision, in one place.
    tight = q.pending_wait_by_source(bound_seconds=10)
    assert tight["autoresearch"]["over_bound_count"] == 1, \
        "the keyword works; the route simply does not pass it"
    assert tight["autoresearch"]["max_wait_seconds"] == \
        out["autoresearch"]["max_wait_seconds"], \
        "tightening the bound moves the COUNT, never the measured wait"


#: The queue extract #2185's triage landed in the VAULT (not this repo): the whole
#: `queue` table at 2026-10-04T15:07Z, 4 587 rows, of which 3 are still `queued`.
#: Every figure quoted below is read out of those bytes by the node that follows, so
#: none of it is a live-box reading that rots: re-run the node and it re-derives them.
QUEUE_WITNESS = Path("backlog/data/workers-queue-2185.db")


def test_the_landed_witness_bytes_carry_a_real_starving_source(tmp_path):
    """The outage this field exists for, re-measured from committed bytes — N=1 of the
    item's own acceptance, and not a synthetic row.

    The witness is `~/obsidian/backlog/data/workers-queue-2185.db`, whose landing commit
    is read from those bytes rather than transcribed here (`git -C ~/obsidian log
    --oneline -1 -- backlog/data/workers-queue-2185.db`): the whole `queue` table as it
    stood at 2026-10-04T15:07Z. Reading it through the real method at its own newest stamp —
    `2026-10-04T15:07:50.126953+00:00`, selected out of the bytes rather than typed —
    gives exactly:

        {'owed-check': {'max_wait_seconds': 9151.464, 'over_bound_count': 1}}

    i.e. ONE `owed-check` row (`id` 5005, `enqueued_at` 12:35:18.662728+00:00) sitting
    unclaimed for 2 h 32 m, one row past the 3 600 s bound, and a max that is NOT the
    bound — the other two queued rows are 559.97 s and 0 s old (`id` 5048 and 5055), so
    the count and the max come from different draws on the same source. All three carry
    `not_before` NULL, so the clause-3 exclusion is not what is shrinking the figure.

    Why these bytes and not the path clause 6 first named: `backlog/data/workers.db` is
    occupied by #1949's schema-only authority witness — 126,976 bytes, whose
    `count(*) from sqlite_master` is THAT item's contract figure, relied on by
    tests/test_grant_mint_quota_default.py and tests/unit/test_grant_policy.py — so a
    queue extract written there would retire another item's evidence. `backlog/data/` is
    vault state and is not in this repo's diff; the node reads it through
    `app.paths.VAULT_ROOT`, the same seam the skill pin uses.
    """
    import shutil
    import sqlite3

    from app.paths import VAULT_ROOT
    from workers.queue import WorkQueue

    src = VAULT_ROOT / QUEUE_WITNESS
    assert src.is_file(), (
        f"the witness is gone: {src} is not in the vault, so clause 6's figures have "
        "no bytes behind them — re-extract from a live workers.db, do not retype them")
    db = tmp_path / "witness.db"
    shutil.copy(src, db)

    conn = sqlite3.connect(str(db))
    try:
        queued = conn.execute(
            "SELECT id, source, enqueued_at, not_before FROM queue"
            " WHERE state='queued' ORDER BY id").fetchall()
        newest = conn.execute("SELECT max(enqueued_at) FROM queue").fetchone()[0]
    finally:
        conn.close()

    assert len(queued) == 3, queued
    assert [r[0] for r in queued] == [5005, 5048, 5055], queued
    assert all(r[3] is None for r in queued), \
        "a not_before appeared in the witness: the exclusion, not the wait, is now " \
        "deciding this figure — re-read before quoting it"

    out = WorkQueue(db).pending_wait_by_source(
        now=datetime.fromisoformat(newest))
    wait = out["owed-check"]
    assert abs(wait["max_wait_seconds"] - 9151.464) < 0.001, wait
    assert wait["over_bound_count"] == 1, wait
    assert list(out) == ["owed-check"], out
    assert wait["max_wait_seconds"] > 3600, \
        "the quoted 2 h 32 m has to actually be past the bound the payload states"


def _skill_step(text: str, heading: str) -> str:
    """One step's own body, from just under `heading` to the NEXT `## Step` heading.

    Cut at whichever heading comes next, not at one named later in the file: Step 3b
    ends at `## Step 3c`, so a slice taken to `## Step 4` also carries 3c and 3d, and a
    guard whose docstring says "scoped to Step 3b" while reading three steps would pass
    on a field that only ever appears in the wrong one.
    """
    assert heading in text, f"{heading} is not in the skill at all"
    body = text[text.index(heading) + len(heading):]
    body = body[body.index("\n") + 1:]
    nxt = re.search(r"^## Step", body, re.M)
    if nxt:
        body = body[:nxt.start()]
    return body


def _documented_jq_program(step: str) -> str:
    """The jq program the step tells an operator to paste, read off the page.

    Extracted rather than retyped because the thing under test is the PUBLISHED filter:
    a copy of it written in this file keeps passing when the skill's one-liner drifts,
    which is the exact gap that left Step 3b pre-labelling an empty result as health.
    """
    m = re.search(r"jq\s+-r\s+'(.*?)'", step, re.S)
    assert m, "Step 3b publishes no `jq -r '<program>'` filter to test"
    return m.group(1)


def test_the_now_keyword_is_what_makes_a_wait_reproducible(tmp_path):
    """`now=` is what lets one wait be measured twice and read the same number.

    The figure is a difference against a clock, and the previous round died on exactly
    that: its own node stamped a row from one `datetime.now()` and measured it against
    a later one, then asserted the difference was `4000.0` — true only when the two
    calls land in the same millisecond, and read 4000.001 under the gate's 8 workers.
    Handing the method one instant makes the figure exact, which is what a caller that
    has to assert or replay needs; the route passes no instant, so the page reads real
    elapsed time. Both branches are pinned against the SAME instant: the subtraction,
    and the `not_before` exclusion — a `now=` that reached only the arithmetic would
    report a row parked ahead of that instant as starving.
    """
    from workers.queue import WorkQueue

    moment = datetime.now(timezone.utc)
    q = WorkQueue(tmp_path / "clock.db")
    q.enqueue("counted", "task", payload={})   # row 1: released 60 s BEFORE `moment`
    q.enqueue("parked", "task", payload={})    # row 2: parked until 60 s AFTER it
    for row_id in (1, 2):
        _restamp(q, row_id, (moment - timedelta(seconds=4000)).isoformat())
    _park(q, 1, seconds_ahead=-60)
    _park(q, 2, seconds_ahead=60)

    assert q.pending_wait_by_source(now=moment) == {
        "counted": {"max_wait_seconds": 4000.0, "over_bound_count": 1},
        "parked": {"max_wait_seconds": 0.0, "over_bound_count": 0},
    }, "one instant in, one exact figure out — and the exclusion read that instant too"

    wall = q.pending_wait_by_source()
    assert wall["counted"]["max_wait_seconds"] > 4000.0, \
        "with no instant the read is real elapsed time, which is why asserting an " \
        "exact number against a wall-clock read is the flake this knob exists to kill"


def test_the_health_skill_reads_the_wait_off_the_api_and_names_the_bound():
    """Clause 5: the field and the procedure have to say the same thing.

    The subject is a vault file — `app.paths.VAULT_ROOT` is `Path.home() / "obsidian"`
    (`app/paths.py:11`), so every worktree reads the same live vault — and there is no
    skip when it is missing: a guard that could go quietly unverified would repeat the
    defect it guards, the same reasoning
    `tests/test_automod_doc_claims.py::…` gives for reading
    `skills/automod-change-own-code/SKILL.md` from a test.

    Scoped to Step 3b, which is the step that today read queue state from
    `/api/workers/status` and raw `workers.db`. Step 1 curls `/api/autonomy/health` and
    is not this field's home — asserting on the whole file would let Step 3b stay
    hand-run while some other section grew the word.
    """
    import re

    from app.paths import VAULT_ROOT

    skill = VAULT_ROOT / "skills" / "queue-health-check" / "SKILL.md"
    assert skill.is_file(), f"clause 5's subject is absent: {skill} does not exist"
    step = _skill_step(skill.read_text(encoding="utf-8"), "## Step 3b")
    assert len(step) > 400, f"positive control: Step 3b is only {len(step)} chars"
    # The slice has to END at the next step, not merely start at this one. Step 3c
    # follows 3b and is about `abandoned_unseen` attribution, so a slice that ran to
    # `## Step 4` would pass every needle below on prose belonging to a different
    # diagnosis — the bug an earlier version of this helper had.
    assert "## Step 3c" not in step, "the slice ran past the end of Step 3b"

    for field in ("pending_wait_max_seconds", "pending_wait_over_bound_count",
                  "pending_wait_bound_seconds"):
        assert field in step, f"Step 3b never names {field}"
    assert re.search(r"3600", step), "Step 3b does not state the 3600 s bound"
    assert "not_before" in step, \
        "Step 3b does not say a back-off-parked row is excluded from both figures"
    assert "api/workers/health" in step, "Step 3b must read the wait off the health route"


def test_the_published_filter_names_a_starving_source_and_says_nothing_otherwise(
        monkeypatch, tmp_path):
    """Clause 5's other half: the skill's own one-liner over the route's real bytes.

    Naming three keys in prose is half the contract; the other half is that the program
    an operator pastes from Step 3b prints a line for a source that IS waiting. Against
    the pre-#2185 payload that exact filter printed 0 bytes at exit 0, because `null > 0`
    is false in jq — a key that does not exist and a pool that is healthy produced the
    same answer, and the step defined that silence as the healthy result. So both halves
    are pinned here over bytes the real route built: one row back-dated 2 h must make the
    filter name `probe` with its seconds, its count and the bound; a queue with nothing
    in it must print nothing, which is the only way that silence now means anything.

    The filter is EXTRACTED from the skill (see `_documented_jq_program`), never retyped,
    and run through the same `jq -r` the step tells a human to run.
    """
    import json
    import shutil
    import subprocess

    from app.paths import VAULT_ROOT

    skill = VAULT_ROOT / "skills" / "queue-health-check" / "SKILL.md"
    step = _skill_step(skill.read_text(encoding="utf-8"), "## Step 3b")
    program = _documented_jq_program(step)
    jq = shutil.which("jq")
    # The step's own procedure is `curl … | jq -r …`, so a box without jq cannot run
    # Step 3b either; that is a broken skill, not a reason to skip the pin quietly.
    assert jq, "Step 3b's published procedure needs jq, and this box has none"

    starving, q = _clamp_client(monkeypatch, tmp_path)
    q.enqueue("probe", "task", payload={})
    _backdate(q, 1, seconds=7200)
    body = starving.get("/api/workers/health?days=7&runs=0").content.decode()

    ran = subprocess.run([jq, "-r", program], input=body, capture_output=True,
                         text=True)
    assert ran.returncode == 0, f"jq refused the published filter: {ran.stderr}"
    lines = [ln for ln in ran.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, f"exactly one starving source expected, got {ran.stdout!r}"
    assert re.match(r"^probe: oldest queued 7[0-9]{3}s, 1 row\(s\) over 3600s$",
                    lines[0]), lines[0]
    # The line's three numbers are the payload's own, not three more literals: the
    # filter is a VIEW of the fields, and this is what says so.
    row = json.loads(body)["sources"][0]
    assert int(re.search(r"queued (\d+)s", lines[0]).group(1)) == int(
        row["pending_wait_max_seconds"])
    assert row["pending_wait_over_bound_count"] == 1

    healthy, q2 = _clamp_client(monkeypatch, tmp_path, name="healthy.db")
    _seed_run(q2, datetime.now(timezone.utc) - timedelta(hours=6))
    quiet = subprocess.run([jq, "-r", program],
                           input=healthy.get(
                               "/api/workers/health?days=7&runs=0").content.decode(),
                           capture_output=True, text=True)
    assert quiet.returncode == 0, quiet.stderr
    assert quiet.stdout.strip() == "", \
        "a healthy pool must print nothing, and only now can that silence be read"


def _restamp(q, queue_id: int, value: str) -> None:
    """Write one raw string into a queued row's `enqueued_at`, test-local SQL."""
    import sqlite3

    with sqlite3.connect(str(q.db_path)) as conn:
        conn.execute("UPDATE queue SET enqueued_at=? WHERE id=?", (value, queue_id))


def test_an_unparseable_enqueued_at_gives_the_row_no_age_at_all(tmp_path):
    """The one exclusion that is not about the row's state: no instant, no wait.

    A row in state 'queued' whose `enqueued_at` `datetime.fromisoformat` refuses
    (a bare epoch float here) has nothing to subtract from now. It must contribute
    neither to the max nor to the count — and the source must still appear with
    zeros rather than vanish, because "this source has a row waiting" is a fact the
    depth already established and the wait has no standing to contradict.
    """
    from workers.queue import WorkQueue

    # A bare epoch-seconds value with its fraction, as text: no `T`, no offset, and
    # `datetime.fromisoformat` raises ValueError on it. The `.5` is load-bearing for
    # this file, not for the branch: an integer-shaped 10-digit string is the one
    # literal shape a reader (or a review pass resolving citations) can mistake for an
    # identifier, and this value names nothing — it is only a stamp no writer here
    # would produce.
    epoch_seconds_as_text = "1791127000.5"
    q = WorkQueue(tmp_path / "garbage.db")
    q.enqueue("probe", "task", payload={})
    _restamp(q, 1, epoch_seconds_as_text)
    assert q.pending_wait_by_source() == {
        "probe": {"max_wait_seconds": 0.0, "over_bound_count": 0}}

    # The row that CAN be parsed decides the figure, and the source keeps its entry.
    # A window, not equality: the stamp is written by `_backdate`'s clock and the age
    # is read by `pending_wait_by_source`'s, so the true value is 4000 s plus however
    # long the two calls took — an exact `4000.0` here passes only when the process
    # gets the same millisecond twice, which is what made the gate's 8-worker run
    # read 4000.001 and call this round's own new test a new failure.
    q.enqueue("probe", "task", payload={})
    _backdate(q, 2, seconds=4000)
    out = q.pending_wait_by_source()
    assert list(out) == ["probe"], "the unparseable row is skipped, not the source"
    assert 4000 <= out["probe"]["max_wait_seconds"] <= 4030, out
    assert out["probe"]["over_bound_count"] == 1, out


def test_a_naive_enqueued_at_is_read_as_utc_like_every_other_duration(tmp_path):
    """The convention, on the record: no offset means UTC, and the wait pays for it.

    `_iso_or_none` does not leave a naive stamp naive — it stamps it UTC, the
    assumption `_iso_seconds_between` bills GPU-hours on and the one
    `fleet_watchdog._queue_starving` makes through `autonomy._parse_iso`. This read
    keeps that convention so the wait cannot disagree with the GPU-hours on the same
    row, and the cost is stated rather than hidden: this box runs at UTC-7, so a
    naive stamp naming local noon is placed at 12:00Z and ages 7 hours more than a
    local reader would expect. A foreign writer doing that inflates this figure, and
    a future change to `_iso_or_none` that stopped assuming UTC would break THIS node
    first — which is the point of pinning it from the outside.
    """
    from datetime import timedelta
    from workers.queue import WorkQueue

    q = WorkQueue(tmp_path / "naive.db")
    q.enqueue("probe", "task", payload={})
    naive = (datetime.now(timezone.utc) - timedelta(seconds=4000)
             ).replace(tzinfo=None).isoformat()
    assert "+" not in naive and naive.count("T") == 1
    _restamp(q, 1, naive)

    out = q.pending_wait_by_source()
    assert 3990 <= out["probe"]["max_wait_seconds"] <= 4010, out
    assert out["probe"]["over_bound_count"] == 1, \
        "read as UTC, so it ages by real elapsed time and not by the box's offset"
