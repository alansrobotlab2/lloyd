"""P7's context-rot runner: grid, needle placement, salts, guards, decision.

Everything here is offline. The engine is never reached: `wait_idle` and
`stream_chat` are patched, the backend is an `httpx.MockTransport`, and
`usage.db` is a scratch file.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sqlite3
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


def test_grading_counts_a_salt_as_wrong_and_a_hedge_as_not_a_hit():
    n = E.make_needles(2, 50_000)
    old = n.old_port
    assert E.grade(f"PORT: {n.port}", n, "single")["accuracy"] == 1.0
    assert E.grade(f"PORT: {old}", n, "distract4")["verdicts"]["port"] == "wrong"
    assert E.grade(f"PORT: {n.port} or {old}", n, "distract4")["accuracy"] == 0.0
    g = E.grade(f"CODENAME: {n.codename}\nPORT: {old}\nWINDOW: {n.window}", n, "multi3")
    assert g["accuracy"] == pytest.approx(2 / 3)


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
