"""P7's context-rot runner: grid, needle placement, salts, guards, decision.

Everything here is offline. The engine is never reached: `wait_idle` and
`stream_chat` are patched, the backend is an `httpx.MockTransport`, and
`usage.db` is a scratch file.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "run_context_rot_eval", ROOT / "eval" / "run_context_rot_eval.py")
E = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = E
_spec.loader.exec_module(E)


@pytest.fixture(scope="module")
def hays():
    return {sh: asyncio.run(E.sized_haystack(sh, 50_000, 1, None)) for sh in E.SHAPES}


# ── grid ──────────────────────────────────────────────────────────────


def test_the_grid_is_the_plans_225_per_shape_and_deterministic():
    one = E.build_grid(shapes=("session",))
    assert len(one) == 5 * 5 * 3 * 3 == 225
    assert E.build_grid() == E.build_grid()
    assert len(E.build_grid()) == 450
    assert len({c.key for c in E.build_grid()}) == 450
    # a haystack serves consecutive cells: (shape, length, seed) never recurs
    runs = [(c.shape, c.length, c.seed) for c in one]
    changes = [k for i, k in enumerate(runs) if i == 0 or k != runs[i - 1]]
    assert len(changes) == len(set(changes)) == 15


def test_limit_and_lengths_select_a_prefix_of_the_grid():
    a = E.parse_args(["--lengths", "50000,100k", "--limit", "7", "--shape", "repo"])
    cells = E.select_cells(a)
    assert len(cells) == 7 and {c.length for c in cells} == {50_000}
    assert cells == E.build_grid(shapes=("repo",), lengths=(50_000, 100_000))[:7]


# ── needles ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", E.SHAPES)
def test_every_needle_lands_within_two_percent_of_its_depth(hays, shape):
    hay = hays[shape]
    for depth in E.DEPTHS:
        for cond in E.CONDITIONS:
            cell = E.Cell(shape, 50_000, 1, depth, cond)
            n = E.make_needles(1, 50_000)
            _msgs, where = E.plant(hay, E.placements(cell, n))
            for w in where:
                assert w["got"] is not None, w
                if w["kind"] != "distractor":
                    assert abs(w["got"] - depth) <= E.DEPTH_TOLERANCE, (cell.key, w)
                else:
                    assert abs(w["got"] - depth) >= 0.15, (cell.key, w)
            assert len(where) == {"single": 1, "multi3": 3, "distract4": 5}[cond]


def test_planting_does_not_touch_the_cached_haystack(hays):
    hay = hays["session"]
    before = json.dumps(hay.messages)
    E.plant(hay, E.placements(E.Cell("session", 50_000, 1, 0.5, "distract4"),
                              E.make_needles(1, 50_000)))
    assert json.dumps(hay.messages) == before


@pytest.mark.parametrize("seed", range(1, 40))
def test_salts_are_never_the_answer(seed):
    n = E.make_needles(seed, 150_000)
    wrong = n.wrong_ports()
    assert n.port not in wrong and len(wrong) == 4
    for line in E.distractor_lines(n):
        assert n.port not in line
    # digit permutations of the answer: the hard kind of distractor
    assert all(sorted(p) == sorted(n.port) for p in wrong)


def test_the_session_haystack_carries_none_of_build_sessions_own_plants(hays):
    blob = json.dumps(hays["session"].messages)
    assert "# ops:" not in blob                 # its sibling-port salts
    assert "Halcyon deploy notes" not in blob   # its planted notes turn
    assert "call_planted" not in blob


def test_the_question_is_last_and_the_nonce_is_first(hays):
    for shape, hay in hays.items():
        cell = E.Cell(shape, 50_000, 1, 0.5, "multi3")
        planted, _ = E.plant(hay, E.placements(cell, E.make_needles(1, 50_000)))
        msgs = E.request_messages(hay, planted, cell, "abc123")
        assert msgs[0]["role"] == "system" and msgs[0]["content"].startswith("[probe abc123]")
        assert msgs[-1]["role"] == "user" and msgs[-1]["content"].endswith("WINDOW: <HH:MM>")


# ── grading ───────────────────────────────────────────────────────────


def test_grading_counts_a_salt_as_wrong_and_a_hedge_as_its_own_class():
    """The 2026-09-24 grid's `distract4` misses were mostly this shape — both
    ports named, budget spent explaining the conflict — and its recorded
    follow-up is that a hedge has to be a class of its own rather than a miss.
    `hedged` is that class: `accuracy` still refuses to call it a hit (#3801)."""
    n = E.make_needles(2, 50_000)
    old = n.old_port
    assert E.grade(f"PORT: {n.port}", n, "single")["accuracy"] == 1.0
    assert E.grade(f"PORT: {old}", n, "distract4")["verdicts"]["port"] == "wrong"
    hedge = E.grade(f"PORT: {n.port} or {old}", n, "distract4")
    assert hedge["outcomes"]["port"] == "hedged"
    assert hedge["outcomes"]["port"] != "wrong"
    assert hedge["counts"] == {"correct": 0, "hedged": 1, "wrong": 0, "miss": 0}
    assert hedge["accuracy"] == 0.0
    assert hedge["verdicts"]["port"] == "undecided"   # the shipped codes still land
    g = E.grade(f"CODENAME: {n.codename}\nPORT: {old}\nWINDOW: {n.window}", n, "multi3")
    assert g["accuracy"] == pytest.approx(2 / 3)
    assert g["outcomes"] == {"codename": "correct", "port": "wrong", "window": "correct"}


def test_the_four_outcomes_stay_apart():
    """`hedged` needs a second PLANTED value, not any second number: naming the
    true port next to an unrelated digit is a decision, and a wrong pick is not a
    hedge however politely it is phrased."""
    n = E.needles_for(E.Cell("repo", 50_000, 1, 0.5, "distractK", 4))
    assert E.grade(f"PORT: {n.port}", n, "distractK")["outcomes"]["port"] == "correct"
    assert E.grade(f"PORT: {n.lookalikes[1]}", n, "distractK")["outcomes"]["port"] == "wrong"
    assert E.grade("I cannot tell from the notes", n, "distractK")["outcomes"]["port"] == "miss"
    assert E.grade(f"PORT: {n.port} (the window opens at 9999)",
                   n, "distractK")["outcomes"]["port"] == "correct"
    assert E.grade(f"PORT: {n.old_port} rather than {n.lookalikes[2]}",
                   n, "distractK")["outcomes"]["port"] == "wrong"


def test_a_hedge_needs_no_label_and_multi3_reports_every_fact():
    n = E.make_needles(1, 50_000)
    free = E.grade(f"the codename is {n.codename} and the port is {n.old_port} "
                   f"or maybe {n.port}", n, "multi3")
    assert free["outcomes"] == {"codename": "correct", "port": "hedged", "window": "miss"}
    assert free["counts"]["hedged"] == 1 and free["accuracy"] == pytest.approx(1 / 3)


# ── guards ────────────────────────────────────────────────────────────


def test_a_busy_engine_exits_before_anything_is_sent(monkeypatch, hays):
    from app import vllm_metrics

    async def busy(*a, **k):
        raise TimeoutError("engine never went idle")

    sent = []

    async def one(*a, **k):
        sent.append(a)
        return {}

    async def tok(*a, **k):
        sent.append("tokenize")
        return 1

    monkeypatch.setattr(vllm_metrics, "wait_idle", busy)
    monkeypatch.setattr(E, "one_request", one)
    monkeypatch.setattr(E, "count_tokens", tok)
    with pytest.raises(E.EngineBusy):
        asyncio.run(E.run_cells(E.build_grid(lengths=(50_000,))[:2], base="http://x",
                                client=None, write=lambda rows: None,
                                first_idle_limit_s=1, idle_limit_s=1))
    assert sent == []


def test_a_request_is_priority_one_thinking_off_greedy_and_short(monkeypatch):
    import app.harness.client as client_mod
    seen = {}

    async def fake_stream(**kw):
        seen.update(kw)
        yield {"choices": [{"delta": {"content": "PORT: 7123"}}]}
        yield {"usage": {"prompt_tokens": 100, "completion_tokens": 3,
                         "prompt_tokens_details": {"cached_tokens": 0}}}

    monkeypatch.setattr(client_mod, "stream_chat", fake_stream)
    out = asyncio.run(E.one_request("http://x", [{"role": "user", "content": "q"}]))
    assert seen["priority"] == 1 and seen["tools"] is None
    body = seen["extra_body"]
    assert body["max_tokens"] == 64 and body["temperature"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert out["text"] == "PORT: 7123" and out["prompt_tokens"] == 100


def test_djev_shadow_is_muted_before_any_app_import():
    src = (ROOT / "eval" / "run_context_rot_eval.py").read_text()
    mute = src.index('os.environ.setdefault("LLOYD_DJEV_SHADOW", "0")')
    assert mute < src.index("from app")
    assert "LLOYD_DJEV_SHADOW" in E.os.environ


class _Backend:
    """A fake /api/workers/{status,pause}."""

    def __init__(self, paused=False, paused_by=None):
        self.paused = paused
        self.paused_by = list(paused_by or [])
        self.posts: list[bool] = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/workers/status":
            return httpx.Response(200, json={"pool": {"paused": self.paused,
                                                      "paused_by": self.paused_by}})
        if req.url.path == "/api/workers/pause":
            p = json.loads(req.content)["paused"]
            self.posts.append(p)
            self.paused = p
            self.paused_by = ["operator"] if p else []
            return httpx.Response(200, json={"paused": p, "paused_by": self.paused_by})
        return httpx.Response(404)


def _client(backend: _Backend) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(backend.handler))


def test_the_pool_is_paused_for_the_run_and_resumed_after_a_failure():
    b = _Backend()

    async def go():
        async with _client(b) as c:
            with pytest.raises(RuntimeError):
                async with E.PoolPause("http://be", c):
                    assert b.paused
                    raise RuntimeError("boom")
    asyncio.run(go())
    assert b.posts == [True, False]


def test_a_pool_already_paused_is_never_resumed():
    b = _Backend(paused=True, paused_by=["operator"])

    async def go():
        async with _client(b) as c:
            async with E.PoolPause("http://be", c) as p:
                assert not p.ours
    asyncio.run(go())
    assert b.posts == [] and b.paused


def test_an_automod_pause_at_the_end_is_not_lifted():
    b = _Backend()

    async def go():
        async with _client(b) as c:
            async with E.PoolPause("http://be", c, automod_wait_s=0, poll_s=0):
                b.paused_by = ["operator", "automod"]
    asyncio.run(go())
    assert b.posts == [True]


def test_an_unreadable_pool_refuses_the_run():
    async def go():
        t = httpx.MockTransport(lambda r: httpx.Response(503, text="down"))
        async with httpx.AsyncClient(transport=t) as c:
            async with E.PoolPause("http://be", c):
                pass
    with pytest.raises(SystemExit):
        asyncio.run(go())


def test_live_run_resumes_the_pool_when_the_engine_is_busy(monkeypatch, tmp_path):
    from app import vllm_metrics
    b = _Backend()
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda *a, **k: real(transport=httpx.MockTransport(b.handler)))

    async def busy(*a, **k):
        raise TimeoutError("busy")

    async def meta(*a, **k):
        return {}

    monkeypatch.setattr(vllm_metrics, "wait_idle", busy)
    monkeypatch.setattr(E, "engine_meta", meta)
    rc = E.main(["--lengths", "50000", "--limit", "1", "--out", str(tmp_path / "r.json"),
                 "--base", "http://engine"])
    assert rc == 3
    assert b.posts == [True, False]
    doc = json.loads((tmp_path / "r.json").read_text())
    assert doc["rows"] == [] and "pool resumed" in doc["meta"]["pool"]


def test_dry_run_touches_neither_engine_nor_pool(monkeypatch, capsys):
    def refuse(*a, **k):
        raise AssertionError("dry run opened a client")

    monkeypatch.setattr(httpx, "AsyncClient", refuse)
    monkeypatch.setattr(httpx, "Client", refuse)
    assert E.main(["--dry-run", "--lengths", "50000", "--limit", "4"]) == 0
    out = capsys.readouterr().out
    assert "4 cells" in out and "nothing is sent" in out


# ── decision ──────────────────────────────────────────────────────────

THRESH = 210_144


def _summary(curve: dict[int, float], low_depth: dict[int, float] | None = None):
    lengths = sorted(curve)
    depth = {}
    for L in lengths:
        depth[str(L)] = {str(d): curve[L] for d in E.DEPTHS}
        if low_depth and L in low_depth:
            depth[str(L)]["0.5"] = low_depth[L]
    return {"session": {"lengths": lengths,
                        "accuracy": {str(L): {"distract4": curve[L]} for L in lengths},
                        "distract4_by_depth": depth, "ttft": {}}}


def _rec(summary):
    return E.recommend(summary, threshold=THRESH, current_trigger=0.72, current_target=0.52)


def test_a_flat_curve_keeps_todays_values():
    r = _rec(_summary({50_000: 1.0, 100_000: 1.0, 150_000: 0.95, 200_000: 0.93, 240_000: 0.5}))
    assert r["L_star"] == 200_000 and r["action"] == "keep" and r["trigger"] == 0.72


def test_an_early_fall_lowers_the_trigger_by_the_rule():
    r = _rec(_summary({50_000: 1.0, 100_000: 0.95, 150_000: 0.6, 200_000: 0.95, 240_000: 0.95}))
    # the prefix rule: 200k passing after 150k failed does not count
    assert r["L_star"] == 100_000 and r["action"] == "lower"
    assert r["trigger"] == pytest.approx(0.45) and r["target"] == pytest.approx(0.25)
    assert r["trigger_tokens"] == int(0.45 * THRESH)


def test_one_weak_position_fails_a_length():
    r = _rec(_summary({50_000: 1.0, 100_000: 1.0, 150_000: 1.0, 200_000: 1.0, 240_000: 1.0},
                      low_depth={150_000: 0.7}))
    assert r["L_star"] == 100_000
    assert r["per_shape"]["session"]["first_fail"]["low_positions"] == {"0.5": 0.7}


def test_the_worse_shape_decides_and_nothing_passing_is_inconclusive():
    s = _summary({50_000: 1.0, 100_000: 1.0, 150_000: 1.0, 200_000: 1.0, 240_000: 1.0})
    s["repo"] = _summary({50_000: 1.0, 100_000: 1.0, 150_000: 0.5, 200_000: 1.0,
                          240_000: 1.0})["session"]
    assert _rec(s)["L_star"] == 100_000
    zero = _summary({50_000: 0.0, 100_000: 0.0})
    assert _rec(zero)["action"] == "inconclusive"


def test_summary_reads_rows_the_run_writes():
    rows = []
    for L, acc in ((50_000, 1.0), (100_000, 0.5)):
        for d in E.DEPTHS:
            for c in E.CONDITIONS:
                rows.append({"shape": "session", "length": L, "depth": d, "condition": c,
                             "accuracy": acc, "cold": {"ttft_s": L / 10_000, "prompt_tokens": L},
                             "warm": {"ttft_s": 0.2, "same_answer": True}})
    s = E.summarize(rows)["session"]
    assert s["accuracy"]["100000"]["distract4"] == 0.5
    assert s["ttft"]["50000"]["cold_p50"] == 5.0 and s["ttft"]["50000"]["warm_p50"] == 0.2
    assert E.interp_ttft(s["ttft"], 75_000) == pytest.approx(7.5)


# ── cost side ─────────────────────────────────────────────────────────


def _usage_db(path: Path, now: datetime) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE usage (id INTEGER PRIMARY KEY, ts TEXT, session_id TEXT, "
                 "input_tokens INTEGER, duration_ms INTEGER, compaction TEXT, "
                 "prefix_misses INTEGER, reprefill_tokens INTEGER)")
    rows = [
        # in the new band [94k, 151k): one by input_tokens, one by a relief pass
        (1, now - timedelta(hours=1), "20260924_120000_abcd", 120_000, 100_000, None, 1, 50_000),
        (2, now - timedelta(hours=2), "20260924_110000_autocode_9f2a", 40_000, 100_000,
         json.dumps({"relief": [{"used_before": 130_000}]}), 0, 0),
        # below the new trigger, above the old one, and outside the window
        (3, now - timedelta(hours=3), "20260924_100000_abcd", 60_000, 100_000, None, 0, 0),
        (4, now - timedelta(hours=4), "20260924_090000_abcd", 160_000, 100_000, None, 0, 0),
        (5, now - timedelta(days=20), "20260904_090000_abcd", 120_000, 100_000, None, 0, 0),
    ]
    conn.executemany("INSERT INTO usage VALUES (?,?,?,?,?,?,?,?)",
                     [(i, t.strftime("%Y-%m-%dT%H:%M:%S"), s, a, d, c, m, r)
                      for i, t, s, a, d, c, m, r in rows])
    conn.commit()
    conn.close()


def test_cost_side_counts_the_turns_a_lower_trigger_would_add(tmp_path):
    now = datetime(2026, 9, 24, 12, 0, 0)
    db = tmp_path / "usage.db"
    _usage_db(db, now)
    before = db.stat().st_mtime_ns
    c = E.cost_side(db, new_trigger_tokens=94_000, current_trigger_tokens=151_303,
                    cold_ttft_s=10.0, now=now)
    assert db.stat().st_mtime_ns == before
    assert c["turns"] == 4
    assert c["extra_compactions"] == 2
    assert c["extra_background"] == 1 and c["extra_user"] == 1
    assert c["busy_seconds"] == 400                  # four disjoint 100 s turns
    assert c["cost_fraction"] == pytest.approx((2 / 14) * 10 / (400 / 14), rel=1e-3)
    assert c["soak_baseline"]["prefix_misses"] == 1
    keep = E.cost_side(db, new_trigger_tokens=None, current_trigger_tokens=151_303,
                       cold_ttft_s=None, now=now)
    assert keep["extra_compactions"] is None


def test_union_seconds_merges_overlapping_turns():
    assert E._union_seconds([(0, 10), (5, 20), (30, 40)]) == 30


def test_decide_rewrites_the_report_from_a_json(tmp_path):
    rows = []
    for L in E.LENGTHS:
        for d in E.DEPTHS:
            for c in E.CONDITIONS:
                rows.append({"shape": "session", "length": L, "depth": d, "condition": c,
                             "accuracy": 1.0, "cold": {"ttft_s": 1.0, "prompt_tokens": L}})
    p = tmp_path / "context-rot-2026-09-25.json"
    p.write_text(json.dumps({"meta": {"date": "2026-09-25", "grid": {"lengths": list(E.LENGTHS)}},
                             "rows": rows}))
    db = tmp_path / "usage.db"
    _usage_db(db, datetime.now(timezone.utc).replace(tzinfo=None))
    assert E.main(["--decide", str(p), "--usage-db", str(db)]) == 0
    doc = json.loads(p.read_text())
    assert doc["decision"]["action"] == "keep"
    assert "Verdict: keep" in p.with_suffix(".md").read_text()


# ── the k arms (#3801): look-alike count at fixed length ─────────────────────


def _planted_ports(lines):
    """The four-digit values a distractor family actually states."""
    return [v for line in lines for v in re.findall(r"\b\d{4}\b", line)]


def test_distractk_plants_exactly_k_lookalikes_at_the_distract4_offsets():
    """Clause 2: k is a parameter, the family is the sibling-port family, and the
    depths are the ones `distract4` already uses — so a k cell differs from a
    shipped cell in the count of look-alikes and in nothing else."""
    for k in E.K_VALUES:
        cell = E.Cell("repo", 50_000, 1, 0.3, "distractK", k)
        items = E.placements(cell, E.needles_for(cell))
        dist = [p for p in items if p[1] == "distractor"]
        assert len(dist) == k, k
        assert sum(p[1] == "port" for p in items) == 1, k       # the needle is still one line
        want = sorted(min(0.98, max(0.02, (0.3 + j * E.DISTRACTOR_STEP) % 1.0))
                      for j in range(1, k + 1))
        assert sorted(p[0] for p in dist) == pytest.approx(want), k
    four = E.Cell("repo", 50_000, 1, 0.3, "distractK", 4)
    got = sorted(round(p[0], 2) for p in E.placements(four, E.needles_for(four))
                 if p[1] == "distractor")
    shipped = sorted(round(p[0], 2) for p in E.placements(
        E.Cell("repo", 50_000, 1, 0.3, "distract4"), E.make_needles(1, 50_000))
        if p[1] == "distractor")
    assert got == shipped == [0.1, 0.5, 0.7, 0.9]


def test_every_lookalike_is_a_wrong_answer_and_none_of_them_is_the_answer():
    """k = 32 means 32 planted values, and every one of them has to be graded as
    wrong or a hedge: a cell that plants 32 look-alikes but still salt-lists 4 is
    a cell where 28 of the interference sources are free."""
    for k in E.K_VALUES:
        n = E.needles_for(E.Cell("repo", 50_000, 1, 0.5, "distractK", k))
        lines = E.distract_lines(n, k)
        planted = _planted_ports(lines)
        assert len(lines) == k and len(planted) == k, k
        assert len(set(planted)) == k, k                        # no value twice
        assert n.port not in planted, k
        assert set(planted) <= set(n.wrong_ports()), k
        # the salt set grew with k: naming any planted value grades `wrong`, so a
        # cell with 32 look-alikes cannot have 28 of them graded as a free pass.
        for v in planted:
            assert E.grade(f"PORT: {v}", n, "distractK")["outcomes"]["port"] == "wrong", (k, v)
        svcs = re.findall(r"the ([a-z-]+) relay", "\n".join(lines))
        assert len(set(svcs)) == len(svcs), k                   # distinct relays
        if k > 1:
            assert E.GRADED_RELAY not in svcs[1:], k            # stale port stated once


def test_a_family_that_reused_a_relay_name_would_cross_the_duplicate_threshold():
    """Why the labels are drawn from 35 distinct names: the same relay with two
    ports is 0.714 under the calibrated predicate — above 0.6, i.e. #2271's
    near-duplicate class, which is not the stimulus this arm is about."""
    thr = E.dup_detect().DUP_JACCARD_THRESHOLD
    same = E.dup_detect().similarity("# ops note: the billing-east relay listens on port 7254.",
                                     "# ops note: the billing-east relay listens on port 7245.")
    assert same == pytest.approx(0.7143, abs=1e-3) and same >= thr
    n = E.needles_for(E.Cell("repo", 50_000, 1, 0.5, "distractK", 32))
    assert E.family_jaccard(E.distract_lines(n, 32)) == 0.0909


def test_a_k_cell_records_its_worst_pair_and_the_cap_above_it():
    """The risk this pins: at k = 16 or 32 the look-alike family could quietly
    become a duplicate family. Each cell carries its own worst pair under
    `dup_detect`'s predicate plus the cap, and the measured families stay far
    under the threshold, so the cap guards rather than binds here."""
    for length in E.K_LENGTHS:
        for seed in E.SEEDS:
            n = E.needles_for(E.Cell("repo", length, seed, 0.5, "distractK", 32))
            for arm in E.K_ARMS:
                worst = max(E.family_jaccard(E.arm_lines(n, arm, k)) for k in E.K_VALUES)
                assert worst < E.dup_detect().DUP_JACCARD_THRESHOLD, (length, seed, arm)
                assert E.cap_k(n, arm) == max(E.K_VALUES), (length, seed, arm)
    n = E.needles_for(E.Cell("repo", 50_000, 1, 0.5, "distractK", 16))
    got = E.cell_distractors(E.Cell("repo", 50_000, 1, 0.5, "distractK", 16), n)
    assert got == {"arm": "distractK", "k": 16, "planted": 16,
                   "max_pairwise_jaccard": 0.0909, "dup_threshold": 0.6,
                   "over_threshold": False, "cap_k": 32}
    assert E.cell_distractors(E.Cell("repo", 50_000, 1, 0.5, "distract4"), E.make_needles(1, 50_000)) == {}


def test_dissimilark_is_a_different_kind_of_fact_carrying_no_answer_value():
    """Clause 4: the control arm holds the count and the note form and changes
    the KIND — a relay's on-call owner, a name — so no value it states can be the
    accepted port or one of the planted wrong ones. Similarity and count are only
    separable because of this arm."""
    for k in E.K_VALUES:
        n = E.needles_for(E.Cell("repo", 50_000, 1, 0.5, "dissimilarK", k))
        lines = E.dissimilar_lines(n, k)
        text = "\n".join(lines)
        assert len(lines) == k, k
        assert k == 0 or all("on-call owner is" in line for line in lines), k
        assert n.port not in text, k
        assert not any(w in text for w in n.wrong_ports()), k
        assert not re.search(r"\b\d{4}\b", text), k   # no port-shaped token at all
        cell = E.Cell("repo", 50_000, 1, 0.3, "dissimilarK", k)
        dist = [p for p in E.placements(cell, n) if p[1] == "distractor"]
        assert len(dist) == k, k                                    # same count, same offsets
        look = E.arm_lines(n, "distractK", k)
        assert len(look) == len(lines), k                           # arms are the same size
        assert set(_planted_ports(look)) & set(_planted_ports(lines)) == set(), k


def test_the_k_arms_are_selectable_without_the_length_grid(monkeypatch, capsys):
    """Clause 3: a run can name conditions, k values, lengths and depths, and
    `--dry-run` reports the cell count having sent nothing and paused nothing."""
    async def never(*a, **k):
        raise AssertionError("dry run reached the engine or the pool")

    monkeypatch.setattr(E, "ensure_idle", never)
    monkeypatch.setattr(E, "one_request", never)
    args = E.parse_args(["--dry-run", "--conditions", "distractK,dissimilarK",
                         "--ks", "0,1,2,4,8,16,32", "--lengths", "50k,150k"])
    cells = E.select_cells(args)
    assert {c.length for c in cells} == set(E.K_LENGTHS)
    assert {c.condition for c in cells} == set(E.K_ARMS)
    assert {c.k for c in cells} == set(E.K_VALUES)
    assert len(cells) == 2 * 2 * 3 * 5 * 2 * 7 == 840                # 2 shapes x lengths x seeds x depths x arms x k
    assert E.dry_run(args) == 0
    out = capsys.readouterr().out
    assert "grid: 840 cells" in out
    assert "distractK k=32" in out and "dissimilarK k=0" in out
    assert "distract4" not in out.split("by condition:")[1].split("\n")[0]

    sweep = E.parse_args(["--dry-run", "--k-sweep"])
    assert len(E.select_cells(sweep)) == 168                          # depth held at K_DEPTH
    plan = E.run_plan(sweep)
    assert plan["lengths"] == E.K_LENGTHS and plan["depths"] == (E.K_DEPTH,)
    assert plan["conditions"] == E.K_ARMS and plan["ks"] == E.K_VALUES
    assert E.dry_run(sweep) == 0
    out = capsys.readouterr().out
    assert "grid: 168 cells" in out and "cap_k=32" in out

    shipped = E.select_cells(E.parse_args([]))
    assert len(shipped) == 450 and {c.k for c in shipped} == {0}       # default run is unchanged
    with pytest.raises(SystemExit):
        E.parse_args(["--ks", "3"])
    with pytest.raises(SystemExit):
        E.parse_args(["--conditions", "distract9"])


def test_the_curve_reports_accuracy_and_hedging_against_k():
    """What the pinned artifact has to be able to say per arm and length: the
    largest k still at 0.9 correct, and the k where hedging first clears 0.1."""
    def row(arm, k, correct, hedged, wrong=0, miss=0, length=50_000):
        tot = correct + hedged + wrong + miss
        return {"shape": "session", "length": length, "condition": arm, "k": k,
                "counts": {"correct": correct, "hedged": hedged, "wrong": wrong, "miss": miss},
                "accuracy": correct / tot,
                "distractors": {"max_pairwise_jaccard": 0.091}}

    rows = ([row("distractK", k, c, h) for k, c, h in
             [(0, 3, 0), (1, 3, 0), (2, 3, 0), (4, 2, 1), (8, 1, 2), (16, 0, 3), (32, 0, 3)]]
            + [row("dissimilarK", k, 3, 0) for k in E.K_VALUES]
            + [row("distractK", 4, 2, 1, length=150_000)])
    curve = E.k_curve(rows)
    assert set(curve) == set(E.K_ARMS)
    lo = curve["distractK"]["50000"]
    assert lo["curve"]["8"]["correct"] == 0.3333
    assert lo["curve"]["8"]["hedged"] == 0.6667
    assert lo["largest_k_correct_at_least_0.9"] == 2
    assert lo["first_k_hedged_over_0.1"] == 4
    assert curve["distractK"]["150000"]["curve"]["4"]["n"] == 1
    assert curve["dissimilarK"]["50000"]["largest_k_correct_at_least_0.9"] == 32
    assert curve["dissimilarK"]["50000"]["first_k_hedged_over_0.1"] is None
    assert E.k_curve([{"shape": "session", "length": 50_000, "condition": "distract4",
                       "accuracy": 0.0}]) == {}


def test_a_k_row_carries_its_k_its_budget_and_its_family(hays, monkeypatch):
    """The row is the artifact's atom: without `k`, the raised `max_tokens` the
    k arms ask for, and the family's worst pair, a pinned curve could be neither
    re-read nor re-decided by a later reader. The engine is never reached — the
    request is a stub that only records the budget it was handed."""
    async def hay(shape, length, seed, counter):
        return hays[shape]

    async def idle(*a, **k):
        return None

    async def cold(base, msgs, max_tokens=E.MAX_TOKENS):
        cold.seen.append(max_tokens)
        return {"text": "PORT: 9999", "finish_reason": "stop", "ttft_s": 1.0,
                "prompt_tokens": 50_000, "completion_tokens": 5}

    cold.seen = []
    monkeypatch.setattr(E, "sized_haystack", hay)
    monkeypatch.setattr(E, "ensure_idle", idle)
    monkeypatch.setattr(E, "one_request", cold)
    cell = E.Cell("session", 50_000, 1, E.K_DEPTH, "distractK", 8)
    rows = asyncio.run(E.run_cells([cell], base="http://x", client=None,
                                   write=lambda r: None, first_idle_limit_s=1,
                                   idle_limit_s=1, warm=False))
    row = rows[0]
    assert cold.seen == [E.K_ARM_MAX_TOKENS] == [128]      # the shipped 64 is not used here
    assert row["k"] == 8 and row["max_tokens"] == 128
    assert row["cell"] == "session-50k-s1-d50-distractK-k8"
    assert row["distractors"] == {"arm": "distractK", "k": 8, "planted": 8,
                                  "max_pairwise_jaccard": 0.0909, "dup_threshold": 0.6,
                                  "over_threshold": False, "cap_k": 32}
    assert row["needles"]["port"] and row["needles"]["port"] not in row["needles"]["wrong_ports"]
    # the shipped 4 salts plus this cell's 8 look-alikes, deduped. Seed 1 / 50k
    # answers 7658 and its look-alike family starts with the stale port 7586,
    # which the shipped salts already name, so 4 + 8 dedupes to 11, not 12.
    planted = _planted_ports(E.distract_lines(E.needles_for(cell), 8))
    assert set(planted) <= set(row["needles"]["wrong_ports"])
    assert len(row["needles"]["wrong_ports"]) == 11
    assert row["verdicts"]["port"] == "wrong" and row["outcomes"]["port"] == "wrong"
    assert sum(1 for w in row["placed"] if w["kind"] == "distractor") == 8
    assert E.summarize(rows)["session"]["k_curve"]["distractK"]["50000"]["curve"]["8"]["wrong"] == 1.0


# ── #3989: the landed ksweep doc, its checker, and the five ways a doc lies ──────
#
# The 2026-11-08 k sweep's raw artifact cannot be committed (`eval/baselines/*` and
# `*.json` are both ignored and a round may not write `.gitignore`), so what lands is
# the measurement doc plus the checker that proves the doc against the bytes wherever
# they live. No node here reads the runtime directory: each one runs the landed checker
# over an artifact+doc pair generated into `tmp_path` by the runner's own
# `summarize`/`recommend`/`render_md`, so the clean pair holds exactly the relationship
# the real run's two files had, and a doc mutation is then the only difference between
# a pass and a refusal. A committed `.json` fixture is not available to this round,
# which is why the pair is built rather than stored.

CHECKER = ROOT / "eval" / "check_doc_matches_json.py"
KS_DOC = ROOT / "eval" / "measurements" / "context-rot-ksweep-2026-11-08.md"

#: sha256 of the runtime artifact the landed doc reports, measured 2026-11-09 over
#: `/home/alansrobotlab/lloyd-data/ksweep-2026-11-08/context-rot-ksweep-2026-11-08.json`.
#: Re-measuring it against those bytes is the owed post-landing check; this constant is
#: what pins the doc's quote to the figure the triage recorded, in the meantime.
KS_ARTIFACT_SHA = "e463040b404ed7b743dc55e3595feb9a0a053656d776e5f89f1cb2767980459f"

KS_SHAPE, KS_LENGTHS, KS_KS, KS_SEEDS, KS_IV_MAX = "repo", [50_000, 150_000], [0, 1, 2, 4], [1, 2, 3], 3
#: one shape x 2 lengths x 4 ks x 3 seeds x 2 k arms x depth 0.5
KS_CELLS = len(KS_LENGTHS) * len(KS_KS) * len(KS_SEEDS) * len(E.K_ARMS)
KS_JACCARD = {"distractK": {0: 0.0, 1: 0.143, 2: 0.286, 4: 0.429},
              "dissimilarK": {0: 0.0, 1: 0.077, 2: 0.154, 4: 0.231}}
#: The two appendix rows and the headline row the mutations below rewrite, spelled as
#: `render_md` renders them for this fixture: 3 seeds graded `correct`, so n is 3 and
#: every rate is 1.00/0.00.
KS_ROW_K2 = "| 2 | 1.00 | 0.00 | 0.00 | 0.00 | 0.286 | 3 |"
KS_ROW_K4 = "| 4 | 1.00 | 0.00 | 0.00 | 0.00 | 0.429 | 3 |"
KS_HEADLINE_50K = "| repo | distractK | 50,000 | 4 | None |"


def _ks_rows() -> list[dict]:
    """The fixture's 48 k-arm rows: every (arm, length, k, seed) of a one-shape tranche,
    each grading one fact `correct`, so the curve is flat and no hedged cliff exists."""
    return [{"shape": KS_SHAPE, "length": L, "depth": 0.5, "condition": arm, "k": k,
             "accuracy": 1.0, "counts": {"correct": 1, "hedged": 0, "wrong": 0, "miss": 0},
             "cold": {"ttft_s": L / 10_000, "prompt_tokens": L},
             "distractors": {"max_pairwise_jaccard": KS_JACCARD[arm][k]}}
            for arm in E.K_ARMS for L in KS_LENGTHS for k in KS_KS for seed in KS_SEEDS]


def _ks_artifact() -> dict:
    """An artifact shaped like the run's, computed by the runner rather than typed out:
    `summarize` derives `k_curve`, `recommend` with `full_grid=False` derives the
    inconclusive/partial-grid decision block the doc reports."""
    rows = _ks_rows()
    a = {"meta": {"date": "2026-11-08", "commit": "21d1ae0020abcd",
                  "expect_model": "Qwen3.8-Flash-Next",
                  "engine": {"served": "Qwen3.8-Flash-Next-nvfp4", "max_model_len": 262144},
                  "pool": ["pool paused by this run; lease refreshing every 60 s",
                           "pool resumed"],
                  "grid": {"shapes": [KS_SHAPE], "lengths": list(KS_LENGTHS),
                           "depths": [0.5], "conditions": list(E.K_ARMS), "ks": list(KS_KS),
                           "seeds": list(KS_SEEDS), "cells": KS_CELLS}},
         "complete": True, "rows": rows, "summary": E.summarize(rows),
         "cost": {"turns": 8, "verdict": "not applicable: the trigger does not move"}}
    a["decision"] = E.recommend(a["summary"], threshold=2048, current_trigger=0.74,
                                current_target=0.62, full_grid=False)
    return a


def _ks_doc(a: dict) -> str:
    """The doc half: the prose the checker is told to require, the headline table whose
    rows it compares whole, and the runner's own `render_md` appendix below."""
    g = a["meta"]["grid"]
    headline = [f"| {shape} | {arm} | {L:,} | "
                f"{a['summary'][shape]['k_curve'][arm][str(L)]['largest_k_correct_at_least_0.9']} | "
                f"{a['summary'][shape]['k_curve'][arm][str(L)]['first_k_hedged_over_0.1']} |"
                for shape in g["shapes"] for arm in g["conditions"] for L in g["lengths"]]
    top = "\n".join([
        "# Context rot fixture: the k axis, for the doc-vs-JSON checker",
        "",
        "Artifact: the generated file beside this one. `meta.expect_model` "
        "`Qwen3.8-Flash-Next`, `complete: True`.",
        "Grid actually run: shapes " + str(g["shapes"]) + ", arms " + str(g["conditions"])
        + ", lengths " + str(g["lengths"]) + ", depths " + str(g["depths"]) + ", ks "
        + json.dumps(g["ks"]).replace(" ", "") + ", seeds " + str(g["seeds"])
        + " -> " + str(g["cells"]) + " cells.",
        "**The curve is TRUNCATED** -- depths " + str(g["depths"]) + ", ks " + str(g["ks"])
        + " only, so the k axis stops at 4.",
        "",
        "| shape | arm | length | largest k, correct >= 0.9 | first k, hedged > 0.1 |",
        "|---|---|---|---|---|",
        *headline,
        "",
        "Pool status read before: `{'running': True, 'paused': False, 'paused_by': []}`",
        "",
        "Pool status read after: `{'running': True, 'paused': False, 'paused_by': []}`",
        "",
        "**The hold was not clean**, and the log says so: a first launch died before its release.",
        "",
        "**Cliff verdict, one line:** no hedged cliff was measured: no arm's hedged rate "
        "exceeded 0.1 anywhere in the k range that ran (" + str(g["ks"]) + "), while real "
        "interactive turns reach k = " + str(KS_IV_MAX) + " -- so the reading is censored.",
        "",
    ])
    return top + E.render_md(a)


@pytest.fixture(scope="module")
def ks_pair():
    return _ks_artifact(), _ks_doc(_ks_artifact())


def _pair_files(tmp_path, ks_pair, mutate=None):
    artifact, text = ks_pair
    if mutate is not None:
        text = mutate(text)
    ap, dp = tmp_path / "context-rot-ksweep-mini.json", tmp_path / "context-rot-ksweep-mini.md"
    ap.write_text(json.dumps(artifact), encoding="utf-8")
    dp.write_text(text, encoding="utf-8")
    iv = tmp_path / "conflicts-iv.json"
    iv.write_text(json.dumps({"max_items_seen": KS_IV_MAX}), encoding="utf-8")
    return ap, dp, iv


def _run_checker(artifact: Path, doc: Path, iv: Path | None = None,
                 lloyd_data: str | None = None):
    cmd = [sys.executable, str(CHECKER), str(artifact), str(doc)]
    if iv is not None:
        cmd.append(str(iv))
    env = {k: v for k, v in os.environ.items() if k != "LLOYD_DATA"}
    if lloyd_data is not None:
        env["LLOYD_DATA"] = lloyd_data
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env)


def _report(p) -> dict:
    """The checker's verdict. A traceback prints no JSON, so a crash surfaces here as a
    decode error naming the stdout it got rather than as a missing-file pass."""
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:                                  # noqa: TRY301
        raise AssertionError(f"the checker printed no verdict:\n{p.stdout[:400]}\n"
                             f"stderr:\n{p.stderr[:600]}")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def test_the_checker_accepts_a_generated_pair_and_says_what_it_counted(tmp_path, ks_pair):
    """The clean case has to be clean for stated reasons: 48 cells, the 4 curve rows the
    grid implies, no cliff, a truncated tranche. A fixture the checker waved through for
    the wrong reason would satisfy every mutation node below and prove nothing."""
    ap, dp, iv = _pair_files(tmp_path, ks_pair)
    r = _report(_run_checker(ap, dp, iv))
    assert r["ok"] is True and r["failures"] == []
    assert (r["cells"], r["curve_rows"], r["cliff"], r["truncated"]) == (KS_CELLS, 4, None, True)
    assert r["sample_read"] is True and r["conflicts_sample"] == str(iv)


def _perturb_cell(t: str) -> str:
    return t.replace(KS_ROW_K2, "| 2 | 0.75 | 0.25 | 0.00 | 0.00 | 0.286 | 3 |", 1)


def _delete_row(t: str) -> str:
    return t.replace(KS_ROW_K2 + "\n", "", 1)


def _fabricate_row(t: str) -> str:
    """A k the grid never ran, printed as though it had."""
    return t.replace(KS_ROW_K4 + "\n",
                     KS_ROW_K4 + "\n| 8 | 1.00 | 0.00 | 0.00 | 0.00 | 0.500 | 3 |\n", 1)


def _move_headline(t: str) -> str:
    """The SECOND headline cell rewritten: the lie a prefix comparison cannot see, which
    is why the checker compares the whole row."""
    return t.replace(KS_HEADLINE_50K, "| repo | distractK | 50,000 | 2 | None |", 1)


def _delete_cliff(t: str) -> str:
    return "\n".join(ln for ln in t.splitlines()
                     if not ln.startswith("**Cliff verdict, one line:**")) + "\n"


def _spell_length_not_k(t: str) -> str:
    return t.replace("### repo distractK at 50k", "### repo distractK at 50000k", 1)


@pytest.mark.parametrize("mutate,why", [
    (_perturb_cell, "doc row"),
    (_delete_row, "k rows"),
    (_fabricate_row, "k rows"),
    (_move_headline, "headline table row"),
    (_delete_cliff, "exactly one cliff verdict line"),
], ids=["perturbed-cell", "deleted-row", "fabricated-row", "moved-headline-figure",
        "deleted-cliff-line"])
def test_each_of_the_five_doc_mutations_is_refused_and_says_which_one(
        tmp_path, ks_pair, mutate, why):
    """A checker that passes a doc whose figures were rewritten is not a check. Each
    mutation is applied alone to the generated doc, and the refusal has to name the kind
    of lie it caught — exiting non-zero for an unrelated reason would not close this."""
    ap, dp, iv = _pair_files(tmp_path, ks_pair, mutate)
    p = _run_checker(ap, dp, iv)
    r = _report(p)
    assert p.returncode != 0 and r["ok"] is False, f"{mutate.__name__} passed the checker"
    assert any(why in f for f in r["failures"]), \
        f"{mutate.__name__} refused for the wrong reason: {r['failures']}"


def test_the_curve_sections_are_found_by_the_heading_the_runner_derives(tmp_path, ks_pair):
    """`### {shape} {arm} at {L // 1000}k` is built from the artifact's own `k_curve`
    keys, reproducing `render_md`. Spelling the heading from `meta.grid.lengths` instead
    yields `at 50000k`, matches nothing, and reports a complete doc as broken — so the
    checker must fail here on the MISSING SECTION message, not on a row comparison."""
    ap, dp, iv = _pair_files(tmp_path, ks_pair, _spell_length_not_k)
    r = _report(_run_checker(ap, dp, iv))
    assert r["ok"] is False
    assert any("no k_curve section headed '### repo distractK at 50k'" in f
               for f in r["failures"]), r["failures"]
    assert not any("doc row" in f for f in r["failures"]), \
        f"a missing heading must not be reported as a row disagreement: {r['failures']}"


def test_the_cliff_sample_resolves_from_lloyd_data_and_its_absence_is_a_refusal(
        tmp_path, ks_pair):
    """The checker's third input used to default to an absolute path inside the one
    runtime directory this run's bytes sit in, while the doc's own command passes two
    arguments — so the documented invocation died on `FileNotFoundError` as soon as that
    directory was cleaned. It resolves under `LLOYD_DATA` now, and a missing input is a
    refusal naming the path, never a traceback and never a verdict."""
    ap, dp, iv = _pair_files(tmp_path, ks_pair)
    staged = tmp_path / "data" / "ksweep-2026-11-08"
    staged.mkdir(parents=True)
    (staged / "conflicts-iv.json").write_text(iv.read_text(encoding="utf-8"), encoding="utf-8")

    found = _report(_run_checker(ap, dp, lloyd_data=str(tmp_path / "data")))
    assert found["ok"] is True and found["sample_read"] is True, found["failures"]
    assert found["conflicts_sample"] == str(staged / "conflicts-iv.json")

    gone = tmp_path / "gone"
    p = _run_checker(ap, dp, lloyd_data=str(gone))
    assert p.returncode != 0, "a missing cliff sample must not read as a clean doc"
    assert "Traceback" not in p.stderr and "FileNotFoundError" not in p.stderr, p.stderr[:400]
    r = _report(p)
    assert r["ok"] is False and r["sample_read"] is False
    assert str(gone / "ksweep-2026-11-08" / "conflicts-iv.json") in p.stdout, \
        "the refusal has to name the path it looked for"
    assert not any("doc row" in f or "cliff verdict" in f for f in r["failures"]), \
        f"an unreadable input is not a doc disagreement: {r['failures']}"


def test_the_landed_checker_carries_no_absolute_path_to_one_machine():
    src = CHECKER.read_text(encoding="utf-8")
    assert "/home/alansrobotlab" not in src, \
        "one machine's home directory in the checker is how it dies on the next box"
    assert "LLOYD_DATA" in src


def test_the_doc_names_the_checker_by_a_path_that_resolves_from_the_repo_root():
    text = KS_DOC.read_text(encoding="utf-8")
    hits = [m.start() for m in re.finditer(r"check_doc_matches_json\.py", text)]
    assert len(hits) >= 2, "the doc must cite the checker in prose and in its command block"
    assert all(text[i - 5:i] == "eval/" for i in hits), \
        "every checker citation has to carry the repo-root path; a bare filename is a file " \
        f"that is not there: {[text[max(0, i - 12):i + 26] for i in hits if text[i - 5:i] != 'eval/']}"

    cmds = [ln for ln in text.splitlines() if "check_doc_matches_json.py " in ln]
    assert len(cmds) == 1, cmds
    toks = cmds[0].split()
    i = toks.index("eval/check_doc_matches_json.py")
    assert toks[i + 1] == ("/home/alansrobotlab/lloyd-data/ksweep-2026-11-08/"
                           "context-rot-ksweep-2026-11-08.json"), \
        "the documented command must name the bytes the artifact line names"
    assert toks[i + 2] == "eval/measurements/context-rot-ksweep-2026-11-08.md"
    assert cmds[0].startswith("cd ~/lloyd"), "the command resolves from the repo root"
    for rel in (toks[i], toks[i + 2]):
        assert (ROOT / rel).is_file(), f"{rel} does not resolve from the repo root"


def test_the_doc_names_where_the_bytes_live_and_promises_no_repo_copy():
    text = KS_DOC.read_text(encoding="utf-8")
    lines = text.splitlines()
    art = [ln for ln in lines if ln.startswith("Artifact")]
    assert len(art) == 1, "one artifact line, so no stale uncommittable path survives beside it"
    a = art[0]
    assert ("/home/alansrobotlab/lloyd-data/ksweep-2026-11-08/"
            "context-rot-ksweep-2026-11-08.json") in a, a
    assert KS_ARTIFACT_SHA in a and KS_ARTIFACT_SHA.startswith("e463040b")
    assert "`2d2aaba3`" in a, "the provenance commit the bytes came from"
    assert "backlog/data/2026-11-09.3923-context-rot-ksweep-2026-11-08.json" in a, \
        "the vault copy is where the bytes outlive the runtime directory"
    assert "No copy of the JSON is ever added to this repo" in a
    assert "eval/baselines" not in text, \
        "the doc must not point at a repo path the ignore rules make impossible to land"

    by = [ln for ln in lines if "partial_grid" in ln]
    assert len(by) == 1 and "by construction" in by[0].lower(), \
        "the inconclusive verdict must be labelled as structural, once"
    assert "eval/run_context_rot_eval.py:1239" in by[0] and "inconclusive" in by[0]
    assert "summary[shape].k_curve" in by[0]
    src = (ROOT / "eval" / "run_context_rot_eval.py").read_text(encoding="utf-8").splitlines()
    assert '"partial_grid": not full_grid' in src[1238], "the :1239 citation must resolve"
    assert 'action="inconclusive"' in src[1240], "the :1241 citation must resolve"


def test_the_doc_and_the_checker_are_tracked_and_neither_needed_an_ignore_rule_moved():
    """The witness is `git ls-files eval/ | grep -cE 'ksweep|check_doc_matches_json'`
    going 0 -> 2. Pinned with the reason the raw JSON is not a third landed file: the
    artifact path still matches its blanket ignore rule, and the two paths this round
    added are not ignored at all, so no `.gitignore` edit was needed — if a negation ever
    makes the artifact committable, this node is what says the doc's promise is stale."""
    tracked = _git("ls-files", "--", "eval/").stdout.split()
    assert "eval/measurements/context-rot-ksweep-2026-11-08.md" in tracked
    assert "eval/check_doc_matches_json.py" in tracked
    assert len([p for p in tracked if re.search(r"ksweep|check_doc_matches_json", p)]) == 2
    for rel in ("eval/measurements/context-rot-ksweep-2026-11-08.md",
                "eval/check_doc_matches_json.py"):
        assert _git("check-ignore", "-q", rel).returncode == 1, f"{rel} is git-ignored"
    held = _git("check-ignore", "-v", "eval/baselines/context-rot-ksweep-2026-11-08.json")
    assert held.returncode == 0 and "eval/baselines/*" in held.stdout, held.stdout
