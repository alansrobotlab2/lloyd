"""#2665: the KV pool a window is counted against comes from that window's own log lines.

`scripts/vllm_prefix_miss_window.py` had two defects that made the daily
prefix-miss count (`autonomy` task #92) red on every day after 2026-10-12:

1. `extract` collected `GPU KV cache size` values from EVERY matching line in
   every `agent-llm-primary.log*` file with no time test, while the status lines
   read from the same files ARE window-filtered. One guard, two sides reading
   differently: a boot on any day in the rotation poisoned every window that did
   not contain it, which is how the 2026-10-05 window — counted green on 2026-10-06
   with `pool_tokens: [844969]` — now dies on `[484612, 844969]`.
2. `derive` hard-unpacked exactly one pool (`(pool,) = ex["pool_tokens"]`) and so
   had no defined behaviour for a window that genuinely spans two sizes.

Everything here is a synthetic `--data-root` under `tmp_path`, never the live
logs: the live rotation is ~1 file/day, so 2026-10-05 has already lost the status
lines before `16:01:48` local, and a regression that greps live logs passes or
fails on rotation rather than on the fix.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import vllm_prefix_miss_window as W

REPO_ROOT = Path(__file__).resolve().parents[1]

# The item's own control window, 2026-10-05 -> 2026-10-06 UTC: the day whose
# committed-time extract still says what that window measured —
# ~/lloyd-data/vllm_prefix_miss_2026-10-05.json, `pool_tokens: [844969]`.
START, END = "2026-10-05", "2026-10-06"
T0 = datetime(2026, 10, 5, tzinfo=timezone.utc).timestamp()
T1 = datetime(2026, 10, 6, tzinfo=timezone.utc).timestamp()

BIG_POOL = 844_969     # what the engine boots with at full free memory
SMALL_POOL = 484_612   # the 2026-10-12 overlapping restart that got less

# Gap timeline: the previous iteration's last proposed tool call at +2 h, the
# miss's request sent an hour later, its iteration ending 20 s after that.
GAP_START = T0 + 2 * 3600
MISS_ITERATION_END = T0 + 3 * 3600     # when the miss's iteration finished
MISS_ITERATION_MS = 20_000             # ... and how long it took

# Three status lines inside the gap, each reporting 20,000 computed tokens at
# 90 % KV. Their sum is 60,000 tokens, which sits BETWEEN the two free-pool
# thresholds at that peak: (1 - 0.9) * 844,969 = 84,496.9 (not churned) and
# (1 - 0.9) * 484,612 = 48,461.2 (churned). So the churn count names which
# denominator `derive` actually used, instead of just asserting it picked one.
GAP_TOKENS_PER_LINE = 2_000.0
GAP_KV_PCT = 90.0


def _local(t: float) -> str:
    """The engine's own stamp: month-day and clock, local zone, no year."""
    return datetime.fromtimestamp(t, ZoneInfo(W.ENGINE_TZ)).strftime("%m-%d %H:%M:%S")


def _boot_line(t: float, tokens: int) -> str:
    return (f"(EngineCore pid=1314507) INFO {_local(t)} [kv_cache_utils.py:2320] "
            f"GPU KV cache size: {tokens:,} tokens, Maximum concurrency for "
            f"262,144 tokens per request: 3.22x")


def _status_line(t: float, kv_pct: float, *, running: int = 1,
                 prompt_tok_s: float = 0.0, gen_tok_s: float = 10.0) -> str:
    return (f"INFO {_local(t)} [loggers.py:1234] Engine 000: Avg prompt throughput: "
            f"{prompt_tok_s:.1f} tokens/s, Avg generation throughput: {gen_tok_s:.1f} "
            f"tokens/s, Running: {running} reqs, Waiting: 0 reqs, "
            f"GPU KV cache usage: {kv_pct:.1f}%")


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_root(tmp_path: Path, *, boots: list[tuple[float, int]]) -> Path:
    """A `--data-root` with one counted turn, one miss with a gap, and `boots`.

    `boots` are (UTC instant, KV pool size) pairs written across two log files
    the way a byte rotation leaves them.
    """
    root = tmp_path / "data-root"
    (root / "event_logs").mkdir(parents=True)
    (root / "logs" / "services").mkdir(parents=True)

    db = sqlite3.connect(root / "usage.db")
    db.execute("CREATE TABLE usage (ts TEXT, session_id TEXT, "
               "prefix_misses INTEGER, reprefill_tokens INTEGER)")
    db.executemany("INSERT INTO usage VALUES (?, ?, ?, ?)", [
        (_iso(MISS_ITERATION_END), "20261005_030000_chat_a1b2c3", 1, 200_000),
        (_iso(T0 + 3600), "20261005_010000_chat_d4e5f6", 0, None),
    ])
    db.commit()
    db.close()

    events = [
        {"ts": _iso(GAP_START), "event": "brain1.tool_call_proposed",
         "session_id": "20261005_030000_chat_a1b2c3", "turn_id": "T1",
         "data": {"tool": "Bash"}},
        {"ts": _iso(MISS_ITERATION_END), "event": "brain1.prefix_miss",
         "session_id": "20261005_030000_chat_a1b2c3", "turn_id": "T1",
         "data": {"duration_ms": MISS_ITERATION_MS, "iteration": 7, "input_tokens": 200_000,
                  "cache_read": 0}},
    ]
    (root / "event_logs" / "lab.events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")

    lines = [_status_line(T0 + 3600 + n * 10, 30.0) for n in range(6)]
    lines += [_status_line(GAP_START + off, GAP_KV_PCT, prompt_tok_s=GAP_TOKENS_PER_LINE)
              for off in (30, 60, 90)]
    lines += [_status_line(MISS_ITERATION_END + 5, 95.0)]
    logs = sorted(boots, key=lambda b: b[0])
    (root / "logs" / "services" / "agent-llm-primary.log").write_text(
        "\n".join(_boot_line(t, n) for t, n in logs if t >= T0) + "\n"
        + "\n".join(lines) + "\n", encoding="utf-8")
    (root / "logs" / "services" / "agent-llm-primary.log.1").write_text(
        "\n".join(_boot_line(t, n) for t, n in logs if t < T0) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def one_pool(tmp_path: Path) -> Path:
    """One in-window boot at 844,969; the 484,612 boot sits OUTSIDE the window.

    Both out-of-window sides are filled: 48 h before the window's start (past the
    3,600 s lookback the status lines use) and 17 h after its end.
    """
    return make_root(tmp_path, boots=[
        (T0 - 48 * 3600, SMALL_POOL),
        (T0 + 9 * 3600, BIG_POOL),
        (T1 + 17 * 3600, SMALL_POOL),
    ])


@pytest.fixture
def two_pools(tmp_path: Path) -> Path:
    """Two in-window boots, so the window genuinely spans two pool sizes."""
    return make_root(tmp_path, boots=[
        (T0 - 48 * 3600, SMALL_POOL),
        (T0 + 9 * 3600, BIG_POOL),
        (T0 + 12 * 3600, SMALL_POOL),
        (T1 + 17 * 3600, SMALL_POOL),
    ])


@pytest.fixture
def no_boot_in_window(tmp_path: Path) -> Path:
    """A window with no boot line in it at all — the ordinary case for a day
    whose engine was not restarted."""
    return make_root(tmp_path, boots=[
        (T0 - 48 * 3600, SMALL_POOL),
        (T1 + 17 * 3600, SMALL_POOL),
    ])


def test_pool_sizes_come_only_from_lines_inside_the_window(one_pool: Path):
    """Clause 1: the pool line is window-filtered exactly like the status line.

    Before #2665 `extract` added every `GPU KV cache size` match in every
    `agent-llm-primary.log*` to `pools`, so the single 484,612 boot of
    2026-10-12 sat in `pool_tokens` for EVERY window in the rotation — including
    2026-10-05, whose committed extract carries `[844969]` alone.
    """
    ex = W.extract(START, END, data_root=one_pool)
    assert ex["pool_tokens"] == [BIG_POOL], (
        "an out-of-window boot widened pool_tokens, so derive's free-pool "
        "arithmetic is against a pool this window never ran on")
    # The status lines from the same files are still read, so the filter is the
    # window's, not an accident of parsing.
    assert ex["kv_samples"], "the status lines vanished with the pool line"
    assert W.derive(ex, gate=0.60)["pool_tokens"] == BIG_POOL


def test_a_window_spanning_two_pool_sizes_derives_instead_of_raising(two_pools: Path):
    """Clause 2: two in-window boots give two sizes, and `derive` still answers.

    `(pool,) = ex["pool_tokens"]` was a `ValueError: too many values to unpack
    (expected 1)` on this fixture, which is what made the daily job red. `derive`
    now returns the full dict, and the churn figure is the one for the LARGEST
    pool — the fixture's gap tokens (60,000) sit between the two thresholds, so
    this also proves which denominator was used, not merely that one was.
    """
    ex = W.extract(START, END, data_root=two_pools)
    assert ex["pool_tokens"] == [SMALL_POOL, BIG_POOL], (
        "both in-window boots must be in the extract; hiding one is not handling it")
    out = W.derive(ex, gate=0.60)
    assert isinstance(out, dict)
    for key in ("turns", "misses", "reprefill_tokens", "misses_gap_churned_free_pool"):
        assert key in out, key
    assert (out["turns"], out["misses"], out["reprefill_tokens"]) == (2, 1, 200_000)
    assert out["pool_tokens"] == BIG_POOL
    assert out["misses_gap_churned_free_pool"] == 0, (
        "60,000 gap tokens do not cross (1 - 0.9) * 844,969 = 84,496.9; a count of "
        "1 here means the denominator was the small pool")
    # The same gaps DO churn against the smaller pool, which is what the clause
    # below says the docstring has to own.
    small = dict(ex, pool_tokens=[SMALL_POOL])
    assert W.derive(small, gate=0.60)["misses_gap_churned_free_pool"] == 1


def test_derive_states_which_pool_the_churn_denominator_uses():
    """Clause 3: the rule is written beside the arithmetic it feeds, and pinned.

    `misses_gap_churned_free_pool` is the figure that reads the denominator, and
    on the live 2026-10-12 extract the choice is worth 27 events (65 at 844,969
    vs 92 at 484,612, per the task-#92 run record), so a silent choice is a
    silent 40 % swing in a figure `architecture/vllm.md` §10 quotes.
    """
    doc = " ".join((W.derive.__doc__ or "").split())
    assert doc, "derive has no docstring to state the rule in"
    assert ("the churn denominator is the largest of them" in doc
            and "misses_gap_churned_free_pool" in doc
            and "LOWER BOUND" in doc), doc


def test_the_cli_prints_derived_json_for_a_window_with_a_foreign_pool_size(
        one_pool: Path, tmp_path: Path):
    """Clause 4 (+ item clause c): the real invocation exits 0 on stdout JSON.

    Run as a subprocess because the contract is the CLI's own: the skill counts
    stdout's derived fields plus the extract file as the product, and the run
    that dies in `derive` prints nothing at all (measured at the base commit:
    exit 1, stdout 0 bytes) while still writing the extract file.
    """
    extract_to = tmp_path / "extract.json"
    run = subprocess.run(
        [sys.executable, "-m", "scripts.vllm_prefix_miss_window",
         "--start", START, "--end", END, "--data-root", str(one_pool),
         "--gate", "0.6", "--write-extract", str(extract_to)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
    assert run.returncode == 0, run.stderr[-1200:]
    out = json.loads(run.stdout)
    for key in ("turns", "misses", "reprefill_tokens", "kv_p90", "misses_gap_over_gate"):
        assert key in out, f"{key} missing from the derived JSON: {sorted(out)}"
    assert (out["turns"], out["misses"]) == (2, 1)
    assert out["kv_p90"] == pytest.approx(0.9)
    assert out["misses_gap_over_gate"] == 1
    # The extract file is the other half of the product, and it is the artefact
    # the rotation makes unrecoverable later — so its pool list is pinned too.
    assert json.loads(extract_to.read_text(encoding="utf-8"))["pool_tokens"] == [BIG_POOL]


def test_the_flag_surface_is_the_five_documented_flags(one_pool: Path):
    """Item clause (c): no new flag rides in with the fix.

    The handling had to be inside `extract`/`derive`, not a caller-side opt-in:
    the daily job invokes five flags and a sixth would silently leave it on the
    broken path.
    """
    run = subprocess.run([sys.executable, "-m", "scripts.vllm_prefix_miss_window",
                          "--help"], cwd=REPO_ROOT, capture_output=True, text=True,
                         timeout=120)
    assert run.returncode == 0
    options = {opt for line in run.stdout.splitlines() if line.startswith("  -")
               for opt in re.findall(r"--[a-z][a-z0-9-]*", line)}
    assert options == {"--start", "--end", "--data-root", "--gate", "--write-extract",
                       "--help"}, sorted(options)


def test_a_window_with_no_boot_line_reports_an_unmeasured_churn_count(
        no_boot_in_window: Path):
    """The window filter's own new case: no boot in the window is now normal.

    Before the filter, `pool_tokens` could not be empty — it carried every size
    in the rotation. After it, a day with no restart has NO pool line in its
    window or its lookback, which is the ordinary case rather than an edge one.
    Reporting the churn count as `None` keeps the codebase's standing rule that
    unmeasured is NULL and never zero, and a day is still counted rather than
    crashing on `(pool,) = []`.
    """
    ex = W.extract(START, END, data_root=no_boot_in_window)
    assert ex["pool_tokens"] == []
    out = W.derive(ex, gate=0.60)
    assert out["pool_tokens"] is None
    assert out["misses_gap_churned_free_pool"] is None
    # Everything the pool is not needed for is still derived.
    assert (out["turns"], out["misses"], out["reprefill_tokens"]) == (2, 1, 200_000)
    assert out["misses_gap_over_gate"] == 1
