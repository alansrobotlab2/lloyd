"""#1495: the log join that prices GPU-0 contention (eval/qmd_contention.py)."""
import datetime as dt

from eval.qmd_contention import analyse, parse_daemon, parse_pin_windows

DAEMON = """QMD MCP server listening on http://localhost:8181/mcp
23:59:58.000 POST /query 2 queries (50ms) [fts=2 embed=7 vec=30 chunk=2]
00:00:30.000 POST /query 2 queries (300ms) [fts=2 embed=160 vec=30 chunk=2]
00:00:31.000 POST /query 1 queries (5ms) [fts=2 vec=0 chunk=2]
00:05:00.000 POST /query 2 queries (900ms) [fts=2 embed=9 vec=40 chunk=2 rerank=800]
"""
REG = """2026-09-25 17:00:10,000 [INFO] lloyd-workers.automod-regression: measuring abcdef12 against 12345678 (1 pending)
2026-09-25 17:01:00,000 [INFO] lloyd-workers.automod-regression: abcdef12: success — no regression
"""


def test_the_clock_wrap_is_the_next_day_and_phases_parse():
    rows = parse_daemon(DAEMON, dt.date(2026, 9, 25))
    assert [r["ts"].day for r in rows] == [25, 26, 26, 26]
    assert rows[1]["embed"] == 160 and "rerank" in rows[3]


def test_pin_windows_are_local_time_and_classify_requests():
    rows = parse_daemon(DAEMON, dt.date(2026, 9, 25))
    wins = parse_pin_windows(REG)
    assert len(wins) == 1 and wins[0][0].utcoffset() == dt.timedelta(hours=-7)
    out = analyse(rows, wins)
    assert out["recall_doc_leg__pin"]["total_ms"]["n"] == 1      # 00:00:30Z = 17:00:30 PDT
    assert out["recall_doc_leg__no_pin"]["total_ms"]["n"] == 1
    assert out["rerank__no_pin"]["rerank_ms"]["p50"] == 800
