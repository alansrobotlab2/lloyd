"""The /tmp headroom watch, and the pytest setting that keeps the suite off the budget.

`/tmp` is a tmpfs with a fixed 1,048,576-inode budget. Both production-tree
deletions (2026-09-22, 2026-09-29) happened with it at 100%, where `df -h`
reads healthy and every mkdir fails. Fake `statvfs` readings and a fake /tmp
tree throughout; the placement test drives the real `Guardian.check_tmp`.
"""
from __future__ import annotations

import configparser
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent-services" / "guardian"))

import guardian as G  # noqa: E402
import tmpwatch as TW  # noqa: E402

BUDGET = 1_048_576


def _statvfs(used_inodes: int, used_blocks: int = 10, blocks: int = 1000):
    def fake(_path):
        return types.SimpleNamespace(f_files=BUDGET, f_ffree=BUDGET - used_inodes,
                                     f_blocks=blocks, f_bfree=blocks - used_blocks,
                                     f_frsize=4096)
    return fake


def test_levels_fire_at_eighty_and_ninety_five_percent():
    assert TW.level_for(0.79) is None
    assert TW.level_for(0.80) == "error"
    assert TW.level_for(0.949) == "error"
    assert TW.level_for(0.95) == "critical"
    assert TW.CLEAR_FRACTION < TW.WARN_FRACTION < TW.CRITICAL_FRACTION


def test_the_09_29_reading_is_critical_though_bytes_are_nearly_empty():
    """1% of bytes, 100% of inodes: the reading `df -h` hid."""
    r = TW.measure("/tmp", _statvfs(BUDGET, used_blocks=10))
    assert r.byte_fraction == pytest.approx(0.01)
    assert r.inode_fraction == 1.0
    assert TW.level_for(r.fraction) == "critical"


def test_bytes_alone_can_fire_it():
    r = TW.measure("/tmp", _statvfs(100, used_blocks=900))
    assert TW.level_for(r.fraction) == "error"


def test_latched_one_alert_per_crossing_one_escalation_one_clear(tmp_path):
    readings = iter([int(BUDGET * f) for f in
                     (0.50, 0.85, 0.86, 0.90, 0.97, 0.99, 0.75, 0.69, 0.60, 0.85)])
    state = {"used": 0}

    def fake(_p):
        return _statvfs(state["used"])(_p)

    w = TW.TmpWatch(path=str(tmp_path), statvfs=fake)
    actions = []
    for used in readings:
        state["used"] = used
        action, level, _body = w.tick()
        actions.append((action, level) if action != "none" else "none")
    assert actions == ["none", ("alert", "error"), "none", "none",
                       ("alert", "critical"), "none", "none",
                       ("resolve", None), "none", ("alert", "error")]


def test_the_alert_names_the_largest_consumers(tmp_path):
    (tmp_path / "pytest-of-u" / "pytest-1").mkdir(parents=True)
    for i in range(30):
        (tmp_path / "pytest-of-u" / "pytest-1" / f"f{i}").write_text("x")
    (tmp_path / "small").mkdir()
    (tmp_path / "small" / "one").write_text("x")
    top, truncated = TW.top_consumers(str(tmp_path))
    assert not truncated
    assert top[0][0] == "pytest-of-u" and top[0][1] >= 32
    w = TW.TmpWatch(path=str(tmp_path), statvfs=_statvfs(int(BUDGET * 0.99)))
    action, level, body = w.tick()
    assert (action, level) == ("alert", "critical")
    assert f"{tmp_path}/pytest-of-u" in body
    assert "1,038,090 of 1,048,576 inodes" in body


def test_the_breakdown_is_bounded(tmp_path):
    d = tmp_path / "big"
    d.mkdir()
    for i in range(50):
        (d / f"f{i}").write_text("x")
    top, truncated = TW.top_consumers(str(tmp_path), limit=10)
    assert truncated
    assert sum(n for _, n in top) <= 11


def test_check_tmp_alerts_through_the_notifier_and_never_raises():
    calls = []
    g = G.Guardian.__new__(G.Guardian)
    g.notifier = types.SimpleNamespace(
        alert=lambda level, title, body, **kw: calls.append(("alert", level, title, kw)),
        resolve=lambda title, body: calls.append(("resolve", title)))
    g.tmp = types.SimpleNamespace(tick=lambda: ("alert", "critical", "line one\nmore"))
    g.check_tmp()
    assert calls == [("alert", "critical", TW.ALERT_TITLE, {"coalesce": True})]

    calls.clear()
    g.tmp = types.SimpleNamespace(tick=lambda: ("resolve", None, "back to 2%"))
    g.check_tmp()
    assert calls == [("resolve", TW.ALERT_TITLE)]

    def boom():
        raise OSError("statvfs failed")
    calls.clear()
    g.tmp = types.SimpleNamespace(tick=boom)
    g.check_tmp()
    assert calls == []


def test_check_tmp_runs_on_every_tick_above_the_early_returns():
    src = (ROOT / "agent-services" / "guardian" / "guardian.py").read_text()
    tick = src[src.index("    def tick(self) -> str:"):]
    assert tick.index("self.check_tmp()") < tick.index('if snap["supervisord"] == "unreachable":')


def test_pytest_keeps_a_tmp_path_only_when_its_test_failed():
    """pytest's default keeps the last three runs' basetemps whole; the
    suite's three runs plus killed ones were 666,565 of the 1,048,576 inodes
    on 2026-09-29."""
    ini = configparser.ConfigParser()
    ini.read(ROOT / "pytest.ini")
    assert ini["pytest"].get("tmp_path_retention_policy") == "failed"


def test_no_route_in_the_tree_sends_a_worktree_to_tmp():
    for rel in ("tests/conftest.py", "architecture/testing.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "worktree add --detach /tmp" not in text, rel
