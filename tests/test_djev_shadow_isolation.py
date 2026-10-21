"""No test run ever appends to production's djev shadow log (#1324).

`~/.local/state/lloyd-djev/shadow.jsonl` is the corpus `eval/djev/replay.py
--floors` reads to set the `label_mass` floors — step 5 of
`architecture/djev.md` §9.3 is "let the shadow rows accumulate, then set the
floor from them". Its path is a `Path.home()` literal bound at import
(`app/djev_shadow.py:68-71`), so before this file existed a plain `pytest` run
appended its fixtures' rows to the same file production writes. Re-measured
2026-09-21 08:19Z over the live log, 96 rows spanning 01:55:15Z→08:19:15Z: 36
of its 44 `dedupe` rows carried one of the two fixture titles in
`tests/test_backlog_dedupe.py:61` and `tests/test_backlog_spawn_loop.py:792`,
and 27 of its 32 `rerank` rows carried an empty `meta`. A floor read off that
file is a floor set from fixtures.

Per-test isolation cannot hold here, which is why `tests/test_djev_shadow.py`
leaked with its own patch in place: the recorder's worker is a daemon thread
and `_write()` opens whatever `SHADOW_LOG` is at WRITE time
(`app/djev_shadow.py:295`) — after a function-scoped `monkeypatch` has torn
down, that is the home path again, and the row still arrives. Session scope is
the fix, because the session outlives every drain — and with no teardown,
because restoring at session end is itself a drain window. Paths are repointed
rather than `LLOYD_DJEV_SHADOW=0` set, because that variable is asserted on by
`test_djev_rerank_arm.py::test_the_eval_mutes_the_shadow_recorder`, which reads
it out of the process to prove `eval/run_eval.py`'s own `os.environ.setdefault("LLOYD_DJEV_SHADOW", "0")` set it; an ambient mute
would make that assertion hold with the line under test deleted.

The pins, in the order they would fail:

* `test_the_session_fixture_repoints_all_three_shadow_paths` — the three path
  globals are off the home path and share one directory;
* `test_a_dedupe_create_lands_in_the_repointed_log_not_the_home_one` — a real
  `backlog_write_task` create, flushed, lands in the repointed log and adds no
  fixture row to the home one;
* `test_a_late_drain_after_a_function_scoped_patch_still_cannot_reach_home` —
  the row of a job whose engine read is still in flight when a per-test patch
  goes away lands in the session dir, not on the calibration corpus;
* `test_a_child_pytest_run_leaves_no_shadow_log_in_its_home` — #1324's own
  reproduction as an assertion, with a positive control that the recorder was
  live in that child.

The prose half of the item — what `architecture/djev.md` may say now — is pinned
in `tests/test_djev_doc_claims.py`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from app import djev, djev_shadow

ROOT = Path(__file__).resolve().parent.parent
#: The production state dir, resolved the way `app/djev_shadow.py:68` does.
PRODUCTION_DIR = Path.home() / ".local" / "state" / "lloyd-djev"
PRODUCTION_LOG = PRODUCTION_DIR / "shadow.jsonl"
DOC = ROOT / "architecture" / "djev.md"

#: The fixture title from `tests/test_backlog_dedupe.py:61` — 32 of the live
#: log's rows carried this one string before the conftest fixture landed.
NAME = "http_fetch error body says only the status code on a quarter of calls"
BODY = ("The error body from http_fetch carries only the status code; 71 errors "
        "over 268 calls in 21 days, and the model has to spend a turn deciding "
        "whether to retry.")
EXISTING = ("http_fetch fails on a quarter of its calls and its error body says only the status code",
            "`agent_mcp/http_tools.py:254` returns only `HTTP <n>`; 71 flagged errors over 268 calls.")
#: Marks THIS file's own rows, so "no new row at home" is never confused with a
#: concurrent production writer's row in the same window.
PROBE = "late-drain probe row for #1324"

#: The modules #1324's reproduction runs — the ones that call the dedupe seam
#: with no shadow isolation of their own — plus this file's own create test,
#: which FLUSHES, so the child cannot finish without having written a row
#: somewhere. That is what makes "no row in its HOME" a result and not a
#: silence: without the probe node, a recorder that recorded nothing anywhere
#: would satisfy the leak assertion just as well.
CHILD_TARGETS = (*("tests/test_backlog_dedupe.py", "tests/test_backlog_spawn_loop.py"),
                 f"{Path(__file__).relative_to(ROOT)}::test_a_dedupe_create_lands_in_the_repointed_log_not_the_home_one")


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _mine(rows: list[dict]) -> list[dict]:
    """Only the rows this file's own writes could have made."""
    return [r for r in rows
            if (r.get("meta") or {}).get("name") in {NAME, PROBE}]


def _answers(kw: dict) -> djev.Answers:
    return djev.Answers(answers={}, latency_ms=1.0, server_ms=1.0, prompt_tokens=1,
                        chunks=[], uninformative=False, floor=kw.get("floor"),
                        seam=kw.get("seam", ""))


@pytest.fixture
def board(tmp_path, monkeypatch):
    """A backlog the test owns, patched the way `test_backlog_dedupe.py` patches one."""
    from agent_mcp import backlog as BL
    from agent_mcp import backlog_similar as SIM
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(BL, "BACKLOG_DIR", d)
    monkeypatch.setattr(SIM, "DEDUPE_LOG", tmp_path / "dedupe.jsonl")
    monkeypatch.setattr(SIM, "dedupe_config", lambda: dict(SIM.DEFAULTS))
    monkeypatch.setattr(SIM, "semantic_candidates", lambda text, **kw: [])
    return d


@pytest.fixture
def djev_answered(monkeypatch):
    """The recorder enabled, and its engine answered, without reaching GPU 2.

    `djev.enabled()` is a config read and `ask_sync` is a socket; neither is
    this file's subject. What has to be true is that the recorder ENQUEUED, so
    the row exists to be located.
    """
    monkeypatch.setenv("LLOYD_DJEV_SHADOW", "1")
    monkeypatch.setattr(djev, "enabled", lambda: True)
    monkeypatch.setattr(djev, "ask_sync", lambda state, questions, **kw: _answers(kw))
    djev_shadow.reset_for_tests()
    yield
    djev_shadow.reset_for_tests()


def _create(board, monkeypatch, *, name: str = NAME, body: str = BODY) -> dict:
    """One real `backlog_write_task` create against a board holding something
    lexically similar — the shape that makes `_djev_shadow_dedupe` fire. No
    `spawned-by-*` tag, so dedupe advises and the write CREATES."""
    from agent_mcp import backlog as BL
    from agent_mcp import backlog_similar as SIM
    created = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    fm = {"status": "draft", "priority": "medium", "created": created,
          "board": "lloyd", "tags": ["backlog"]}
    (board / "10-existing.md").write_text(
        "---\n" + yaml.dump(fm, default_flow_style=False) +
        f"---\n\n# {EXISTING[0]}\n\n{EXISTING[1]}\n", encoding="utf-8")
    monkeypatch.setattr(SIM, "semantic_candidates", lambda text, **kw: [{"id": 10, "score": 0.9}])
    return json.loads(BL._handle_write({"name": name, "description": body,
                                        "board": "lloyd", "tags": ["backlog"]}))


def _section(text: str, heading: str) -> str:
    """From `heading` to the next `## ` — §11 sliced out of the whole doc."""
    i = text.index(heading)
    j = text.find("\n## ", i + len(heading))
    return text[i: j if j != -1 else len(text)]


# ── the session fixture ───────────────────────────────────────────────────

def test_the_session_fixture_repoints_all_three_shadow_paths():
    """All three globals move together, or a write still finds the home log.

    `PENDING_DROPS` is bound from `STATE_DIR` at import, and `_write()` opens
    `SHADOW_LOG` rather than `STATE_DIR / "shadow.jsonl"`, so the two names are
    two independent leaks: repointing either alone leaves the other on the
    calibration corpus, and `_record_pending_drops()` mkdirs `STATE_DIR` before
    it writes.
    """
    state = Path(djev_shadow.STATE_DIR)
    log = Path(djev_shadow.SHADOW_LOG)
    drops = Path(djev_shadow.PENDING_DROPS)

    assert log.parent == drops.parent, "one directory, or a drain reaches one and not the other"
    assert state == log.parent, "STATE_DIR is where _write mkdirs before it appends"
    assert log.parent != PRODUCTION_DIR, f"the shadow log is on the calibration corpus: {log}"
    assert drops.parent != PRODUCTION_DIR, f"shutdown drops are on the production dir: {drops}"


def test_a_dedupe_create_lands_in_the_repointed_log_not_the_home_one(board, djev_answered, monkeypatch):
    """The clause that makes this a pin: it failed on the base commit, because
    that is where the 36 fixture rows came from. `_djev_shadow_dedupe` gates on
    the LEXICAL `similar[:3]`, which a tmp-backed board supplies, so the shadow
    call fires and the recorder's own path is the only thing deciding where the
    row goes.
    """
    log = Path(djev_shadow.SHADOW_LOG)
    before = _rows(PRODUCTION_LOG)

    out = _create(board, monkeypatch)
    left = djev_shadow.flush(timeout=30.0)

    assert out.get("created") is True, out
    assert left == 0, "the queue did not drain, so 'no row in the home log' proves nothing"
    mine = [r for r in _rows(log) if (r.get("meta") or {}).get("name") == NAME]
    assert mine, f"the recorder wrote nothing to its own repointed log {log}"
    assert mine[-1]["seam"] == "dedupe", mine[-1]

    # Attribution, not the file's size: the live aggregator is a concurrent
    # writer to this same path, and one of HER rows appearing while this test
    # runs is not this test's failure. What must not appear is a row naming
    # this file's fixture.
    leaked = _mine(_rows(PRODUCTION_LOG)[len(before):])
    assert not leaked, (f"the create appended {len(leaked)} row(s) to {PRODUCTION_LOG}: "
                        f"{[(r.get('seam'), (r.get('meta') or {}).get('name')) for r in leaked]}")


def test_a_late_drain_after_a_function_scoped_patch_still_cannot_reach_home(tmp_path, djev_answered, monkeypatch):
    """The race `tests/test_djev_shadow.py` loses, pinned while it is running.

    Its `_isolated` fixture patches the same three names per test and calls
    `reset_for_tests()`, which nulls the queue handle without joining the
    worker. What decides where a row lands is not the patch but where the
    globals point at WRITE time, so this enqueues a job, waits until the worker
    is inside `ask_sync`, and then removes the per-test patch around it — the
    same moment teardown arrives for a real test — before letting the write
    happen.
    """
    entered, released = threading.Event(), threading.Event()

    def _blocked(state, questions, **kw):
        entered.set()
        released.wait(10.0)
        return _answers(kw)

    monkeypatch.setattr(djev, "ask_sync", _blocked)

    per_test = pytest.MonkeyPatch()          # not the test's own: it must end mid-test
    per_test.setattr(djev_shadow, "STATE_DIR", tmp_path)
    per_test.setattr(djev_shadow, "SHADOW_LOG", tmp_path / "shadow.jsonl")
    per_test.setattr(djev_shadow, "PENDING_DROPS", tmp_path / "drops.json")

    home_before = len(_rows(PRODUCTION_LOG))
    djev_shadow.shadow(seam="rerank", state="s", questions={"q": {}}, meta={"name": PROBE})
    assert entered.wait(10.0), "the worker never reached ask_sync, so nothing was in flight"

    per_test.undo()                          # == the per-test fixture's teardown, mid-job
    released.set()
    left = djev_shadow.flush(timeout=30.0)

    assert left == 0, "the queue never drained, so 'no row at home' would prove nothing"
    landed = _mine(_rows(Path(djev_shadow.SHADOW_LOG)))
    assert landed, (f"the late drain left the session dir — the module was pointing at "
                    f"{djev_shadow.SHADOW_LOG} when it wrote")
    assert not (tmp_path / "shadow.jsonl").exists(), (
        "the job wrote to the per-test patch that had already been removed")
    assert not _mine(_rows(PRODUCTION_LOG)[home_before:]), (
        f"the late drain reached the calibration corpus {PRODUCTION_LOG}")


def test_a_child_pytest_run_leaves_no_shadow_log_in_its_home(tmp_path):
    """#1324's reproduction, run as an assertion.

    `HOME` is redirected rather than the path patched, because the three globals
    bind at import: a patch cannot imitate a fresh process, and a fresh process
    is what a test run is. This is also the shape the automod gate runs its
    tests rung in — `scripts/automod/gate.py::_child_env` hands the child
    `HOME = str(Path.home())` and redirects the automod and guardian state dirs
    but no djev path (`grep -n DJEV scripts/automod/gate.py` → 0 hits) — so what
    this node proves is the property a promotion's own suite runs under.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    env = os.environ.copy()
    env["HOME"] = str(fake_home)
    env["LLOYD_DJEV_SHADOW"] = "1"          # the recorder is ON for the child
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *CHILD_TARGETS, "-q", "--no-header",
         "-p", "no:cacheprovider", f"--basetemp={tmp_path / 'basetemp'}"],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)

    assert proc.returncode == 0, (proc.stdout[-2500:] + proc.stderr[-1500:])
    leak = fake_home / ".local" / "state" / "lloyd-djev" / "shadow.jsonl"
    assert not leak.exists(), (
        f"a plain test run still wrote {len(_rows(leak))} row(s) to {leak}: "
        f"{[(r.get('seam'), (r.get('meta') or {}).get('name')) for r in _rows(leak)]}")

    # The positive control, and it is the ROW rather than a directory: `mktemp`
    # makes the session shadow dir whether or not anything was recorded, and the
    # leak assertion above is already true for a recorder that recorded nothing
    # anywhere. So: the child's dedupe row must exist somewhere, and it must not
    # be under the fake HOME — that is the whole property, phrased as evidence.
    # The probe node flushes, so the row is on disk before the child exits;
    # scanning every shadow log under the tree rather than one guessed dir is
    # what makes this survive a basetemp layout that is not mine to own.
    mine = [r for base in tmp_path.rglob("shadow.jsonl") for r in _mine(_rows(base))]
    assert any(r.get("seam") == "dedupe" for r in mine), (
        f"the child produced no dedupe row anywhere under {tmp_path}, so 'no row in "
        f"its HOME' was a silence and not a result: {proc.stdout[-600:]}")


# ── #3114: a null `merged_into` has to say which of its causes it was ────────
#
# `actual.merged_into` is null for three different writes: one that was never
# eligible to merge (no `spawned-by-*` tag, or a `blocker`/`umbrella` tag, or
# observation mode), one the filer forced past the merge, and one that WAS
# eligible and simply had nothing mergeable. Only the last is evidence about
# djev's judgement, and until the writer passed `judged`, `spawn_write` and
# `forced` into `meta` a reader could not tell them apart: `eval/cross_engine_
# disagreement_arm1/rubric.py::label_dedupe` therefore returns `None` for every
# null, which is why the seam has no negative class at all and 857 of the live
# log's 897 `dedupe` rows are inadmissible. These nodes pin the four keys that
# fix it — `judged`, `spawn_write`, `forced` and the normalized `tags` — one
# shape each, plus the shape (`eligible`) that the other three must not be
# confusable with.
#
# Each probe name is distinct so three rows written into the ONE session log
# stay attributable, the same way `_mine` above attributes by `meta.name`.

#: Marks each eligibility shape's row. `_mine` keys on `NAME`/`PROBE`, so these
#: never enter the leak assertions of the four nodes above.
SPAWN_TAG = "spawned-by-autocode"
PROBE_NAMES = {
    "eligible": "#3114 probe an eligible spawn write whose candidates merge nothing",
    "no-spawn-tag": "#3114 probe a human write carrying no spawned-by tag",
    "forced": "#3114 probe a force true create its filer pushed past the merge",
    "blocker-tag": "#3114 probe a blocker tagged spawn write never offered a merge",
    #: The clause-3 probe asks the same shape with a `tags` argument in the
    #: string shape `normalize_tags` exists for, so it needs its own name: the
    #: session shadow log is shared, and two rows would make "exactly one" a
    #: collision rather than a result.
    "blocker-tag-raw": "#3114 probe a blocker tagged spawn write whose tags arrived as prose",
}


def _probe_row(board, monkeypatch, shape: str, *, tags, force: bool = False,
               score: float = 0.5) -> dict:
    """One real create of one eligibility shape, flushed, and its `dedupe` row.

    Goes through `backlog_write_task`'s own handler rather than calling the hook,
    because the boundary under test is the whole write path: `_handle_write` →
    `_dedupe`, which computes the three flags → `djev_shadow.shadow()`, which
    enqueues → the recorder's daemon thread, which writes the row. A direct call
    would pass with the call site still dropping the arguments, which is exactly
    the defect.

    `score` sits under `DEFAULTS["threshold"]` (0.78) and the board item was
    created an hour ago, so neither merge rule can reach it: rule A needs the
    score AND `lexical_min`, rule B needs a candidate created inside
    `recent_window_seconds` (600). Every shape here therefore leaves
    `merged_into` null — the state whose cause is in question.
    """
    from agent_mcp import backlog as BL
    from agent_mcp import backlog_similar as SIM
    name = PROBE_NAMES[shape]
    created = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    fm = {"status": "draft", "priority": "medium", "created": created,
          "board": "lloyd", "tags": ["backlog"]}
    (board / "10-existing.md").write_text(
        "---\n" + yaml.dump(fm, default_flow_style=False) +
        f"---\n\n# {EXISTING[0]}\n\n{EXISTING[1]}\n", encoding="utf-8")
    monkeypatch.setattr(SIM, "semantic_candidates",
                        lambda text, **kw: [{"id": 10, "score": score}])
    args = {"name": name, "description": BODY, "board": "lloyd", "tags": tags}
    if force:
        args["force"] = True

    log = Path(djev_shadow.SHADOW_LOG)
    before = len(_rows(log))          # the session log is shared: only the tail is ours
    out = json.loads(BL._handle_write(args))
    left = djev_shadow.flush(timeout=30.0)

    assert out.get("created") is True, f"{shape}: the write did not create: {out}"
    assert left == 0, f"{shape}: the queue did not drain, so the row below is not all of it"
    rows = [r for r in _rows(log)[before:] if (r.get("meta") or {}).get("name") == name]
    assert len(rows) == 1, (f"{shape}: this create appended {len(rows)} shadow rows for "
                            f"{name!r}, not the one the seam owes every shape")
    assert rows[0]["seam"] == "dedupe", rows[0]
    return rows[0]


def test_an_eligible_spawn_create_records_why_its_merged_into_is_null(board, djev_answered,
                                                                     monkeypatch):
    """Clause 1: for a write that WAS eligible, the null reads as
    "eligible and not merged" — `judged: true`, `spawn_write: true`,
    `forced: false`, with nothing mergeable on offer.

    This is the positive case the three shapes below are read against: with the
    flags present but all false the row would be indistinguishable from a human
    write, so asserting the true side here is what stops the change from being
    satisfied by four constant keys.
    """
    from agent_mcp import backlog_similar as SIM

    row = _probe_row(board, monkeypatch, "eligible", tags=[SPAWN_TAG])

    assert row["actual"]["merged_into"] is None, row["actual"]
    meta = row["meta"]
    assert meta["judged"] is True, f"a spawn write was recorded as never judged: {meta}"
    assert meta["spawn_write"] is True, meta
    assert meta["forced"] is False, meta
    # And the null is genuinely "nothing mergeable", not a merge that was
    # skipped: the one candidate listed sits under the threshold rule A needs.
    assert [c["score"] for c in meta["candidates"]] == [0.5], meta["candidates"]
    assert 0.5 < float(SIM.DEFAULTS["threshold"]), "the probe's candidate is mergeable"
    # The key set `architecture/djev.md` §6.2 now states. `test_djev_doc_claims.py`
    # pins the prose to these names, so a key added or renamed here fails there.
    assert set(meta) == {"name", "judged", "spawn_write", "forced", "tags", "candidates"}, (
        f"the recorded key set moved to {sorted(meta)}, and §6.2 still describes the old one")


#: Each never-eligible shape, the argument that makes it one, the KEY that
#: carries its cause, and the full eligibility triple the row must read. The
#: forced write is the interesting row: `judged` stays TRUE there, because the
#: pair was eligible to be judged and it was the filer's `force`, not the flags,
#: that skipped the append — which is why the negative class is `judged` true
#: AND `forced` false, and why `forced` is one of the four recorded keys rather
#: than something a reader may infer from `judged`.
NEVER_ELIGIBLE = (
    ("no-spawn-tag", ["backlog"], False, "spawn_write",
     {"spawn_write": False, "judged": False, "forced": False}),
    ("forced", [SPAWN_TAG], True, "forced",
     {"spawn_write": True, "judged": True, "forced": True}),
    ("blocker-tag", [SPAWN_TAG, "blocker"], False, "judged",
     {"spawn_write": True, "judged": False, "forced": False}),
)


@pytest.mark.parametrize("shape,tags,force,cause,expect", NEVER_ELIGIBLE,
                         ids=[s[0] for s in NEVER_ELIGIBLE])
def test_a_never_eligible_create_records_the_condition_that_excluded_it(
        board, djev_answered, monkeypatch, shape, tags, force, cause, expect):
    """Clause 2: the seam still fires for every shape that was never eligible to
    merge — `_probe_row` asserts this create appended exactly one row — and the
    row says WHICH condition excluded it. The missing spawn tag reads on
    `spawn_write`, `force: true` reads on `forced`, a `blocker` tag reads on
    `judged`; the other two keys stay at their true values so the three rows are
    three different readings, not one row written three times.

    Before #3114 all three were byte-identical in the only fields that could
    tell them apart, which is why `merged_into: null` could not be read as
    anything.
    """
    row = _probe_row(board, monkeypatch, shape, tags=tags, force=force)
    meta = row["meta"]

    assert row["actual"]["merged_into"] is None, row["actual"]
    for key, want in expect.items():
        assert meta[key] is want, (
            f"{shape}: `meta.{key}` read {meta[key]!r}, expected {want!r}. The key that "
            f"names this shape's cause is `{cause}`; on any other value the null is "
            f"readable as some other shape's null: {meta}")


def test_the_row_carries_the_normalized_tags_that_decided_eligibility(board, djev_answered,
                                                                     monkeypatch):
    """Clause 3: `meta.tags` is the list `_dedupe` computes at
    `agent_mcp/backlog.py:607` — `normalize_tags` of the caller's `tags`, the
    same value `judged` and `spawn_write` were tested against — not the raw
    argument.

    The argument is deliberately the string shape `normalize_tags` exists for
    (`app/backlog_tags.py:90`): models answer the schema's array with
    `'[a, b]'` prose, and four such rows are on the board. Recording the raw
    value would put a string in the row and leave a reader unable to match the
    tag that caused the eligibility failure, while recording
    `args["tags"]` verbatim is the bug this node refuses. Duplicates go too, so
    the list is the one the flags were computed from.
    """
    from app.backlog_tags import normalize_tags

    raw = "['spawned-by-autocode', 'blocker', 'blocker']"
    row = _probe_row(board, monkeypatch, "blocker-tag-raw", tags=raw)

    assert row["meta"]["tags"] == ["spawned-by-autocode", "blocker"], row["meta"]
    assert row["meta"]["tags"] == normalize_tags(raw), "the recorded list is not the normalized one"
    # The tag, not the caller, is what excluded this write — the pairing the
    # key exists for: `judged` false beside `spawn_write` true and a tag in
    # `_NEVER_MERGE_TAGS` says so without re-reading the board.
    assert row["meta"]["judged"] is False, row["meta"]
    assert row["meta"]["spawn_write"] is True, row["meta"]
