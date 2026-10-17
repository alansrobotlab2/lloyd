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


# ── #2891: the alert has to bound its own counter ─────────────────────────
# `/tmp` is a tmpfs, so `df -i /tmp` counts only what exists since the last
# boot and a reboot clears it to zero. The body used to print the bare
# percentage, and every pass that read it projected a linear climb off an
# uptime-bounded counter: the 2026-10-16 note asserted "exhaustion in days,
# >= ~3,290 inodes/hour" while the four firings of that day read 858,619 ->
# 858,709 -> 858,778 -> 859,400 — +781 over >= 8 h, ~98/hour — and the reboot
# at 17:26 PDT reset the count. The framing has to be inside `describe()`:
# `guardian.py:1374` hands that string verbatim to `Notify.alert`, which fans
# it out unchanged to the ledger, ALERT.md, the daily note, the toast and
# speech, so no wrapper is left to add one.

#: A chosen elapsed-since-boot for every pinned figure: 30,300 s = 8 h 25 m.
SINCE_BOOT = 8 * 3600 + 25 * 60
#: The reading every node below drives: 891,289 of 1,048,576 inodes = 85%,
#: which is over `WARN_FRACTION` (0.80) and under `CRITICAL_FRACTION` (0.95).
OVER_LINE = int(BUDGET * 0.85)


def _framed_watch(tmp_path, boot_elapsed):
    """A watch over a two-name fake `/tmp` at 85% of the inode budget.

    `boot_elapsed` is always passed explicitly: a node that left it out would
    inherit the host's own `/proc/uptime`, which is the weather-driven reading
    #2290 is about."""
    (tmp_path / "pytest-of-u" / "pytest-1").mkdir(parents=True)
    for i in range(30):
        (tmp_path / "pytest-of-u" / "pytest-1" / f"f{i}").write_text("x")
    (tmp_path / "small").mkdir()
    (tmp_path / "small" / "one").write_text("x")
    return TW.TmpWatch(path=str(tmp_path), statvfs=_statvfs(OVER_LINE),
                       boot_elapsed=boot_elapsed)


def _consumer_rows(body: str) -> list[tuple[str, int]]:
    """The `(name, entries)` pairs of the body's top-consumer block."""
    rows = []
    for line in body.splitlines():
        if line.startswith("  ") and "/" in line:
            count, where = line.strip().split(None, 1)
            rows.append((where.rsplit("/", 1)[1], int(count.replace(",", ""))))
    return rows


def test_the_alert_body_states_the_counter_is_a_tmpfs_counter_cleared_by_reboot(tmp_path):
    """#2891 clause 1. The bound the body never carried, in one sentence, on
    its own line, beside the percentage and the ranking it bounds."""
    _action, _level, body = _framed_watch(tmp_path, lambda: SINCE_BOOT).tick()
    tmpfs_lines = [ln for ln in body.splitlines() if "tmpfs" in ln.lower()]
    assert len(tmpfs_lines) == 1, f"expected one framing sentence, got {tmpfs_lines}"
    framing = tmpfs_lines[0].lower()
    assert "/tmp is a tmpfs" in framing, framing
    assert "reboot" in framing and "clear" in framing, framing
    # ...alongside the two halves that were already there and stay untouched.
    assert f"{OVER_LINE:,} of {BUDGET:,} inodes used (85%)" in body, body
    assert f"{tmp_path}/pytest-of-u" in body, body
    assert _consumer_rows(body)[0][0] == "pytest-of-u", body


def test_the_inode_line_reports_the_elapsed_time_since_boot(tmp_path):
    """#2891 clause 2, on the same line as the inode percentage, from the
    boot-elapsed input the caller supplies — 30,300 s has to read `8h25m`."""
    _action, _level, body = _framed_watch(tmp_path, lambda: SINCE_BOOT).tick()
    inode_line = next(ln for ln in body.splitlines() if "inodes used" in ln)
    assert "(85%)" in inode_line, inode_line
    assert "8h25m" in inode_line and "since this boot" in inode_line, inode_line


@pytest.mark.parametrize("seconds,written", [
    (SINCE_BOOT, "8h25m"),
    (30, "30s"),
    (3599, "59m"),
    (3600, "1h00m"),
    (90000, "1d01h"),
])
def test_the_elapsed_rendering(seconds, written):
    assert TW.format_elapsed(seconds) == written


def test_the_body_bounds_the_counter_even_with_no_uptime_to_read(tmp_path):
    """#2891 clause 3, the unavailable half: no figure is invented, and the
    sentence that bounds the reading still arrives."""
    _action, _level, body = _framed_watch(tmp_path, lambda: None).tick()
    assert "/tmp is a tmpfs" in body, body
    assert "since this boot" not in body, body


def test_a_raising_boot_elapsed_source_never_reaches_the_guardian_tick(tmp_path):
    """#2891 clause 3, the raising half. `check_tmp` catches anything `tick()`
    raises but only logs it and returns — so a raise here is a silently missing
    alert, which is the failure the framing exists to prevent."""
    def boom():
        raise OSError(2, "No such file or directory", "/proc/uptime")

    action, level, body = _framed_watch(tmp_path, boom).tick()
    assert (action, level) == ("alert", "error"), (action, level)
    assert "/tmp is a tmpfs" in body, body
    assert "since this boot" not in body, body


def test_the_framing_survives_the_hand_to_the_notifier(tmp_path):
    """The process seam: `Guardian.check_tmp` posts `describe()`'s string to
    `Notify.alert` unchanged, so pin the framing on the delivered payload —
    the body is the only surface a reader of ALERT.md or the daily note sees."""
    delivered = []
    g = G.Guardian.__new__(G.Guardian)
    g.notifier = types.SimpleNamespace(
        alert=lambda level, title, body, **kw: delivered.append((title, level, body)),
        resolve=lambda title, body: None)
    g.tmp = _framed_watch(tmp_path, lambda: SINCE_BOOT)
    g.check_tmp()
    assert len(delivered) == 1, delivered
    title, level, body = delivered[0]
    assert (title, level) == (TW.ALERT_TITLE, "error"), (title, level)
    assert "/tmp is a tmpfs" in body, body
    assert "8h25m since this boot" in body, body


def test_the_consumer_list_stays_a_measured_ranking_not_a_filtered_one(tmp_path):
    """#2891 clause 4. Nothing is dropped for being a known scratch name: the
    list is whatever the walk measured, ranked. Includes the 2026-10-16 top
    consumers by their real names plus one name nobody has ever seen, and
    checks the module grew no literal table of permitted names."""
    names = ["pytest-of-alansrobotlab", "torchinductor_alansrobotlab",
             "node-compile-cache", "lloyd-ceiling-planted-tree",
             "a-scratch-dir-with-no-precedent"]
    expected: dict[str, int] = {}
    for i, name in enumerate(names):
        d = tmp_path / name
        d.mkdir()
        for k in range(30 - i):
            (d / f"f{k}").write_text("x")
        expected[name] = 31 - i          # the directory entry itself + its files
    # Built directly, not through `_framed_watch`: that helper plants its own
    # two names in the tree, and this node is about the whole listing.
    _action, _level, body = TW.TmpWatch(
        path=str(tmp_path), statvfs=_statvfs(OVER_LINE),
        boot_elapsed=lambda: SINCE_BOOT).tick()
    rows = _consumer_rows(body)
    assert {n for n, _ in rows} == set(names), rows
    assert [n for n, _ in rows] == sorted(names, key=lambda n: -expected[n]), rows
    assert [n for _, n in rows] == [expected[n] for n, _ in rows], rows

    src = (ROOT / "agent-services" / "guardian" / "tmpwatch.py").read_text()
    code = src[src.index('"""', src.index('"""') + 3) + 3:]   # docstring names the incidents
    assert "KNOWN_GOOD" not in code, "the 2026-10-01 `.t` decline: no allowlist"
    tables = [n for n, v in vars(TW).items()
              if isinstance(v, (set, frozenset, list, tuple)) and v
              and all(isinstance(x, str) for x in v)]
    assert tables == [], f"tmpwatch grew a literal table of /tmp names: {tables}"


def test_read_boot_elapsed_parses_uptime_field_one(tmp_path):
    p = tmp_path / "uptime"
    p.write_text("30300.42 921265.31\n", encoding="utf-8")
    assert TW.read_boot_elapsed(str(p)) == pytest.approx(30300.42)


def test_read_boot_elapsed_yields_none_for_every_unusable_source(tmp_path):
    """The same four ways the real source can fail, each through the real
    reader: absent, empty, unparsable, and not a file at all."""
    assert TW.read_boot_elapsed(str(tmp_path / "absent")) is None
    empty = tmp_path / "empty"
    empty.write_text("", encoding="utf-8")
    assert TW.read_boot_elapsed(str(empty)) is None
    garbage = tmp_path / "garbage"
    garbage.write_text("no number here\n", encoding="utf-8")
    assert TW.read_boot_elapsed(str(garbage)) is None
    assert TW.read_boot_elapsed(str(tmp_path)) is None


def test_the_production_default_is_the_hosts_own_uptime_file():
    """The wiring: what the tests inject is a stand-in, and what the guardian
    runs is a read of `/proc/uptime` — field 1 of it, in this box's boot."""
    assert TW.TmpWatch().boot_elapsed is TW.read_boot_elapsed
    assert TW.read_boot_elapsed() > 0
