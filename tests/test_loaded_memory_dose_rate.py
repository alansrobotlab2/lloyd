"""#4122: dose rate and clearance for the three loaded memory files.

The byte ceilings in `app/prompt_surface.py` can refuse a write that is already
over the line; they cannot see how fast the file is climbing, which is the only
thing that made `lloyd/USER.md` at 16,247 B of 16,384 a five-nights-notice event
rather than a surprise. `scripts/maintenance/loaded_memory_dose_rate.py` publishes
that rate, and this file is what makes it fail loudly if the arithmetic drifts.

Everything here runs against a hand-built vault: a throwaway git repository with
`lloyd/{SOUL,MEMORY,USER}.md` and commits planted at absolute author dates. A
fixture whose data were the live vault would be a rate instrument graded by the
thing it measures, and the second thing it would do is fail on whatever the
nightly writer did last night — the exact failure `tests/test_prompt_surface_guard.py`
documents for the #377 guards. The ceilings, though, are the live ones: a row's
`ceiling` comes from `app.prompt_surface.memory_ceiling`, so the node below grades
it against `MEMORY_CEILINGS` and the seam between the report and the module that
owns the number is pinned rather than assumed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "maintenance" / "loaded_memory_dose_rate.py"
SIBLING = ROOT / "scripts" / "maintenance" / "gate_to_land_lag.py"

from app.data_root import CLI_REFUSED  # noqa: E402  (after ROOT is importable)
from app.prompt_surface import MEMORY_CEILINGS  # noqa: E402  (after ROOT is importable)


def _load(name: str, path: Path):
    """`scripts/maintenance` has no `__init__.py`, so load the file the way
    `tests/test_gate_to_land_lag.py:42-50` does rather than by package import."""
    assert path.is_file(), f"{path} is not on disk"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def lmrd():
    return _load("loaded_memory_dose_rate", SCRIPT)


@pytest.fixture(scope="module")
def gtl():
    return _load("gate_to_land_lag", SIBLING)


#: Every fixture reads against this instant, and the commits below are planted at
#: exact multiples of a day before it, so a rate that comes out right to three
#: decimal places is a rate whose window arithmetic was actually tested.
NOW = datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)

#: The row's contract as the item states it, plus the identity columns a dated
#: series needs (`ts`, `date`, `file`) and, since #4311, `vault` — the root the row
#: was measured against, which is the only thing in a row that says WHICH vault.
#: Graded as a superset check: an EXTRA key is an announcement, a MISSING one is a
#: broken row.
REQUIRED_KEYS = {"ts", "date", "file", "vault", "window_days", "bytes_in",
                 "bytes_out", "net", "pool", "ceiling", "headroom_days"}

LOADED = ("SOUL.md", "MEMORY.md", "USER.md")


def body(n: int, tag: str = "x") -> bytes:
    """Exactly `n` bytes of body, so a byte-count test can state its own size."""
    assert n >= 2 and len(tag) == 1
    out = (tag * (n - 1)).encode() + b"\n"
    assert len(out) == n, out[-8:]
    return out


def _git(repo: Path, *args: str, env: "dict | None" = None) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                       env=dict(os.environ, **(env or {})))
    assert r.returncode == 0, f"git {' '.join(args)} in {repo}: {r.stdout}{r.stderr}"
    return r.stdout.strip()


def make_vault(dirs: Path, steps: "list[tuple[int, dict[str, bytes]]]",
               name: str = "vault", base: datetime | None = None) -> Path:
    """A git vault holding `lloyd/<file>`, with each step committed at `days_ago`.

    Author AND committer dates are planted and the last one is read back out of git
    before the caller is handed the repo: a fixture that silently failed to move the
    clock back would put every commit inside every window, and the rates would then
    be right for the wrong reason — the helper would be reporting the whole history
    as if it were the trailing month.

    `base` is the instant the fixture's `days_ago` counts back from, and defaults to
    `NOW`, which every in-process call below passes to the code under test. A node
    that runs the SCRIPT as a child has to pass the real clock instead: argv carries
    no instant (`--record` is the only flag the daily job uses and there is no
    `--now`), so the child measures its window from actual `datetime.now` and commits
    planted against a frozen 2026-11-15 drift out of that window a day at a time.
    That is why the subprocess nodes here assert shapes and counts, and the one node
    that pins a printed line byte for byte plants its fixture against `base=now`.
    """
    base = base or NOW
    repo = dirs / name
    (repo / "lloyd").mkdir(parents=True)
    _git(repo, "init", "--quiet", "--initial-branch=main")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "dose fixture")
    _git(repo, "config", "commit.gpgsign", "false")
    for days_ago, writes in steps:
        for filename, content in writes.items():
            (repo / "lloyd" / filename).write_bytes(content)
        stamp = int(base.timestamp()) - days_ago * 86_400
        _git(repo, "add", "--all")
        _git(repo, "commit", "--quiet", "--no-verify", "-m", f"day-{days_ago}",
             env={"GIT_AUTHOR_DATE": f"@{stamp} +0000",
                  "GIT_COMMITTER_DATE": f"@{stamp} +0000"})
    last_stamp = int(base.timestamp()) - steps[-1][0] * 86_400
    stored = int(_git(repo, "log", "-1", "--format=%at"))
    assert stored == last_stamp, f"git stored author time {stored}, not {last_stamp}"
    return repo


#: What a checkout needs in order to RUN the script: the script itself plus the
#: modules it imports as a file — `app.paths` (which imports `app.data_root` since
#: #1415), `app.prompt_surface` for `memory_ceiling`, and `scripts.automod.state` for
#: `STATE_DIR`, whose own package `__init__` is a docstring.
#:
#: This list exists because `#4311`'s guard keys on `app.paths.IS_WORKTREE`, which is
#: `.git` existing as a FILE — a property of the checkout on disk that no monkeypatch
#: can hand a subprocess, and `ROOT` cannot serve as either answer because under the
#: gate `ROOT` IS a linked worktree. So the test names the tree the script is loaded
#: from by building one. Same construction as
#: `tests/unit/test_paths_worktree_anchor.py:134-151`; an import added to the script
#: outside this list fails these nodes loudly with a `ModuleNotFoundError` rather
#: than quietly measuring the real checkout instead.
STANDIN_FILES = (
    "app/__init__.py", "app/paths.py", "app/data_root.py", "app/prompt_surface.py",
    "scripts/automod/__init__.py", "scripts/automod/state.py",
    "scripts/maintenance/loaded_memory_dose_rate.py",
)


def make_tree(dirs: Path, *, gitfile: bool, name: str) -> Path:
    """A checkout holding the script under test, its `.git` a FILE or a DIRECTORY.

    The two spellings are the entire discriminator `app.paths.IS_WORKTREE` reads
    (#733): `git worktree add` leaves a one-line `gitdir:` file where the main
    checkout has a directory. The metadata's content is never read, only its form —
    which is why this needs no live-root computation and no `$HOME` preference.
    """
    tree = dirs / name
    for rel in STANDIN_FILES:
        dst = tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dst)
    if gitfile:
        (tree / ".git").write_text("gitdir: /nonexistent/.git/worktrees/dose-fixture\n")
    else:
        (tree / ".git").mkdir()
        (tree / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    return tree


def run_record(tree: Path, vault: Path, state: Path, window_days: int = 30):
    """One `--record` invocation from `tree`, exactly as the daily job invokes it.

    `PYTHONPATH` is pinned to the named tree so `import app` cannot fall through to
    whichever checkout collected this suite — under the gate that is a linked
    worktree, and the stand-in-main runs below would then be graded against the guard
    firing for a reason the test never set up.
    """
    return subprocess.run(
        [sys.executable, str(tree / "scripts" / "maintenance"
                             / "loaded_memory_dose_rate.py"),
         "--vault", str(vault), "--state-dir", str(state),
         "--window-days", str(window_days), "--record"],
        cwd=tree, capture_output=True, text=True,
        env=dict(os.environ, PYTHONPATH=str(tree)))


#: The three grows behind every fixture below: creation 20 days back, +100/+200/+50
#: 12 days back, USER.md +80 2 days back — 3,180 B of gross inflow over 30 days.
QUANT_STEPS: "list[tuple[int, dict[str, bytes]]]" = [
    (20, {"USER.md": body(3000), "MEMORY.md": body(2000), "SOUL.md": body(1000)}),
    (12, {"USER.md": body(3100), "MEMORY.md": body(2200), "SOUL.md": body(1050)}),
    (2, {"USER.md": body(3180)}),
]


# ── clause 1: one row per loaded file, seven fields, a window, and the journal ──


def test_one_row_per_loaded_file_with_the_ceiling_from_the_writers_module(tmp_path, lmrd):
    """Every loaded file gets a row with the full field set; SOUL.md's bound is null.

    Fixture: three files created 20 days back at 3,000 / 2,000 / 1,000 B, grown 12
    days back (+100 / +200 / +50), USER.md grown again 2 days back (+80). At a
    30-day window the creation itself is in the window, so USER.md's gross inflow is
    3,180 B / 30 d = 106.0 B/day — the number below is asserted, not derived from the
    code under test.
    """
    vault = make_vault(tmp_path, [
        (20, {"USER.md": body(3000), "MEMORY.md": body(2000), "SOUL.md": body(1000)}),
        (12, {"USER.md": body(3100), "MEMORY.md": body(2200), "SOUL.md": body(1050)}),
        (2, {"USER.md": body(3180)}),
    ])
    rows = lmrd.dose_rows(vault, 30, NOW)

    assert [r["file"] for r in rows] == list(LOADED) == list(lmrd.LOAD_ORDER)
    for row in rows:
        assert REQUIRED_KEYS <= set(row), f"{row['file']} row is missing {REQUIRED_KEYS - set(row)}"
        assert row["window_days"] == 30, "every row must state the window it divided by"

    user = next(r for r in rows if r["file"] == "USER.md")
    assert user["bytes_in"] == pytest.approx(3180 / 30), user
    assert user["bytes_out"] == 0.0 and user["net"] == pytest.approx(106.0), user
    assert user["pool"] == 3180, "pool is the live file's byte size, not HEAD's"
    assert user["ceiling"] == MEMORY_CEILINGS["USER.md"] == 16_384
    assert user["headroom_days"] == pytest.approx((16_384 - 3180) / 106.0, abs=0.01)

    memory = next(r for r in rows if r["file"] == "MEMORY.md")
    # `abs` is the row's own rounding step: rates are published to 3 decimals, so a
    # tolerance tighter than that grades the formatter and not the arithmetic.
    assert memory["bytes_in"] == pytest.approx(2200 / 30, abs=0.001), memory
    assert memory["ceiling"] == MEMORY_CEILINGS["MEMORY.md"]

    soul = next(r for r in rows if r["file"] == "SOUL.md")
    assert soul["bytes_in"] == pytest.approx(1050 / 30), "SOUL.md's dose is still reported"
    # #377 bounds SOUL.md with the shape ratios and load-bearing markers, not bytes
    # (app/prompt_surface.py:84-86), so clearance-days are undefined for it by ruling
    # — and the row must say null rather than inventing a bound.
    assert soul["ceiling"] is None and soul["headroom_days"] is None, soul

    # The same fixture is the alarm's negative control: ~125 days of clearance on
    # USER.md is exactly the quiet reading the triage predicted for today's dose.
    assert lmrd.alarm_lines(rows) == []


def test_a_file_created_inside_the_window_lands_as_inflow_not_as_a_zero(tmp_path, lmrd):
    """The first commit that ever touches a file is a dose, not a free pass.

    `prev` starts at 0, so a creation of 1,200 B in a 30-day window reads 40.0 B/day.
    Reporting 0 there would let a brand-new writer's first month read as a clean
    file, which is the state this instrument exists to distinguish from silence.
    """
    vault = make_vault(tmp_path, [(5, {"USER.md": body(1200), "MEMORY.md": body(10),
                                       "SOUL.md": body(10)})])
    row = next(r for r in lmrd.dose_rows(vault, 30, NOW) if r["file"] == "USER.md")
    assert row["bytes_in"] == pytest.approx(1200 / 30), row
    assert row["pool"] == 1200


def test_record_appends_one_row_per_file_and_a_same_date_rerun_adds_nothing(tmp_path, lmrd):
    """The journal seam, run as the daily job runs it: a child interpreter, argv, an
    exit code, and rows on disk. Two invocations on one UTC date, one append.

    Exit code 0 here is part of the assertion (nothing under the cut), and so is the
    `appended=0 / skipped_already_recorded=3` on the second run: a recorder that
    re-appended daily would double the series' tail and the next reader could not
    tell a second reading from a duplicate.

    #4311 clause 2 is the sentence in that first paragraph: from the MAIN CHECKOUT the
    behaviour is unchanged by the worktree guard, so this node asserts the same
    `appended=3` then `appended=0 / skipped_already_recorded=3` it asserted before.
    It now runs the script from a stand-in main checkout instead of from `ROOT`,
    because `ROOT` is whichever tree collected the suite and, inside a gate, that is a
    linked worktree — where the guard the item asked for correctly refuses. The tree
    is the only thing that changed about the invocation; see
    `test_record_from_a_linked_git_worktree_…` for the other half of that seam.

    Clause 4 is the date assertion: the length-based arm is gone, and a row whose
    date is not the day the child ran can no longer pass.
    """
    vault = make_vault(tmp_path, QUANT_STEPS)
    main = make_tree(tmp_path, gitfile=False, name="main-checkout")
    state = tmp_path / "state"
    journal = lmrd.record_path(state)

    started = datetime.now(timezone.utc)
    r1 = run_record(main, vault, state)
    ended = datetime.now(timezone.utc)
    out1 = r1.stdout + r1.stderr
    assert r1.returncode == 0, out1
    assert "appended=3" in out1, out1
    lines = [ln for ln in journal.read_text().splitlines() if ln.strip()]
    assert len(lines) == 3, lines
    parsed = [json.loads(ln) for ln in lines]
    assert {p["file"] for p in parsed} == set(LOADED)
    for p in parsed:
        assert REQUIRED_KEYS <= set(p), p
        # No length-based escape hatch (#4311 clause 4). The arm this replaces —
        # `or len(p["date"]) == 10` — was satisfied by "1999-01-01", by "yesterday!",
        # and by any other ten characters, so the node passed on whatever date the row
        # carried. The child stamps `ts` and `date` from one instant, so the pin is
        # the pair: the stamp must fall inside the seconds this node spent running the
        # child (which no stale or fabricated date can do), and `date` must be that
        # stamp's UTC date (which a locally-formatted date fails across a UTC-day
        # boundary). There is no clock injection on argv — `NOW` reaches only the
        # in-process calls above — so the run instant is what "today" means here.
        stamp = datetime.strptime(p["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        assert started - timedelta(seconds=120) <= stamp <= ended + timedelta(seconds=120), \
            f"row stamped {stamp}, outside the run window {started}…{ended}: {p}"
        assert p["date"] == stamp.strftime("%Y-%m-%d"), p
    assert len({p["date"] for p in parsed}) == 1, "three rows, one UTC date"

    assert out1.count(" bytes_in=") == len(LOADED), f"not one line per loaded file\n{out1}"
    for filename in LOADED:
        assert f"file={filename} " in out1, f"{filename} printed no row\n{out1}"
    assert "window_days=30" in out1

    r2 = run_record(main, vault, state)
    out2 = r2.stdout + r2.stderr
    assert r2.returncode == 0, out2
    assert "appended=0" in out2 and "skipped_already_recorded=3" in out2, out2
    assert len([ln for ln in journal.read_text().splitlines() if ln.strip()]) == 3


# ── #4311: a row is only trustworthy if it says which tree and vault made it ────


def test_record_from_a_linked_git_worktree_refuses_and_writes_no_journal(tmp_path, lmrd):
    """`--record` from a linked worktree appends nothing, creates nothing, exits
    non-zero, and names the tree it refused (#4311 clause 1).

    `record()` dedups on `(date, file)`, so the first row of a UTC date is permanent
    for that date, and `VAULT_ROOT` and `STATE_DIR` both resolve off `$HOME` while
    `app.paths` resolves off the imported tree (SM_20261115_205336 finding 1). A
    `--record` launched from a round's checkout therefore measures `pool` and
    `bytes_in` against one tree and `ceiling` against another, files the mixture in
    the LIVE journal, and cannot be corrected until the next UTC midnight — with 18
    linked worktrees registered on this box, that is a Tuesday, not a hypothesis.

    The control is inside the node, because a refusal is exactly as trustworthy as
    its control: the same script bytes, the same argv, the same vault and a state dir
    of the same shape, run from a stand-in whose `.git` is a directory, appends three
    rows. `.git`'s form is the only difference between the two runs, so a guard that
    fired for any other reason — a bad vault, a bad window, an unrelated crash — fails
    the control instead of passing on a refusal that means nothing.
    """
    vault = make_vault(tmp_path, QUANT_STEPS)
    worktree = make_tree(tmp_path, gitfile=True, name="linked-worktree")
    main = make_tree(tmp_path, gitfile=False, name="main-checkout")

    state_wt = tmp_path / "state-worktree"
    r_wt = run_record(worktree, vault, state_wt)
    out_wt = r_wt.stdout + r_wt.stderr
    # 2, and specifically not the alarm's 1: a scheduler that read the refusal as a
    # red reading would re-run it, which the dedup makes pointless, and would report
    # a run that measured nothing as a dose that happened to be high.
    assert r_wt.returncode == CLI_REFUSED == 2, f"refused run exited {r_wt.returncode}\n{out_wt}"
    assert "REFUSED" in out_wt and "--record" in out_wt, out_wt
    assert "tree=worktree" in out_wt, f"the refusal must say it refused a worktree\n{out_wt}"
    named = re.search(r"LLOYD_HOME=(\S+)", out_wt)
    assert named, f"the refusal must name the tree it refused to record from\n{out_wt}"
    assert Path(named.group(1)).resolve() == worktree.resolve(), (
        f"the refusal named {named.group(1)}, not the worktree the script was loaded "
        f"from ({worktree})")
    assert "appended=" not in out_wt, f"a refused run must not report appending\n{out_wt}"
    assert not lmrd.record_path(state_wt).exists(), \
        "a refused --record must not create the journal"
    assert not state_wt.exists(), \
        "a refused --record must not create the state directory either"

    state_main = tmp_path / "state-main"
    r_main = run_record(main, vault, state_main)
    out_main = r_main.stdout + r_main.stderr
    assert r_main.returncode == 0, out_main
    assert "appended=3" in out_main, \
        f"the same argv from a main checkout must still record\n{out_main}"
    assert len([ln for ln in lmrd.record_path(state_main).read_text().splitlines()
                if ln.strip()]) == 3


def test_every_recorded_row_names_the_vault_it_measured_and_the_print_is_unchanged(
        tmp_path, lmrd):
    """The row grows a `vault` column; `format_rows` does not (#4311 clause 3).

    Both halves are on purpose. `vault` is what makes a journal row answer "which
    vault's bytes is this" — the field the round-`HOME` variant of the hazard needs,
    where `pool` came from the round's `~/obsidian` (left UNLINKED by
    `worktree.ensure_round_home`, unlike `lloyd` and `lloyd-data`) and read
    `pool: null` for all three files while the run printed `appended=3`. And the
    per-file lines stay byte-identical because they are named keys, not columns: a
    reader greps `bytes_in=` today and gets the same line shape as yesterday's rows,
    so adding a field to a row must not re-flow the print. The literal below is that
    line, byte for byte, numbers computed from the fixture and not from the code.
    """
    # Anchored to the real clock rather than to `NOW`: the child computes its window
    # from `datetime.now` and argv carries no instant, so a fixture planted against a
    # frozen date drifts out of that window a day at a time and the literal below
    # would rot (`make_vault`'s `base` exists for this node).
    base = datetime.now(timezone.utc).replace(microsecond=0)
    vault = make_vault(tmp_path, QUANT_STEPS, base=base)
    rows = lmrd.dose_rows(vault, 30, base)
    for row in rows:
        assert row["vault"] == str(vault), f"{row['file']} row does not name its vault: {row}"
    # The field is the path as the caller handed it. `resolve()` would erase the one
    # difference it exists to record: a round's `~/obsidian` is a symlink whose target
    # is the live vault.
    assert Path(rows[0]["vault"]) == vault, rows[0]
    assert next(r for r in rows if r["file"] == "USER.md")["bytes_in"] == 106.0, \
        "the fixture must really produce the rates the literal below quotes"

    main = make_tree(tmp_path, gitfile=False, name="main-checkout")
    state = tmp_path / "state"
    r = run_record(main, vault, state)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    disk = [json.loads(ln)
            for ln in lmrd.record_path(state).read_text().splitlines() if ln.strip()]
    assert len(disk) == 3, disk
    assert {p["vault"] for p in disk} == {str(vault)}, \
        f"every appended row must name the one vault it measured: {disk}"

    per_file = [ln for ln in r.stdout.splitlines() if ln.startswith("file=")]
    assert len(per_file) == len(LOADED), f"not one per-file line each\n{out}"
    # Key ORDER and spelling, not a substring hunt. For a line whose numbers move with
    # the clock, "byte-unchanged" can only mean this: the same eight named keys, in the
    # same sequence, spelled the same way, so a grep that worked on yesterday's output
    # works on today's. A field inserted anywhere in the line, a reordering, or a
    # switch to fixed-width columns all fail it; the `vault` this round added to the
    # ROW must not appear here.
    shape = re.compile(
        r"file=(?:SOUL|MEMORY|USER)\.md "
        r"bytes_in=-?\d+(?:\.\d+)? bytes_out=-?\d+(?:\.\d+)? net=-?\d+(?:\.\d+)? "
        r"pool=(?:\d+|null) ceiling=(?:\d+|null) "
        r"headroom_days=(?:-?\d+(?:\.\d+)?|null) window_days=\d+")
    for ln in per_file:
        assert shape.fullmatch(ln), f"the per-file print changed shape: {ln!r}"
    assert not any("vault=" in ln for ln in per_file), \
        f"the vault belongs in the row, not the print\n{per_file}"
    user_line = next(ln for ln in per_file if ln.startswith("file=USER.md "))
    assert user_line == ("file=USER.md bytes_in=106.0 bytes_out=0.0 net=106.0 "
                         "pool=3180 ceiling=16384 headroom_days=124.566 "
                         "window_days=30"), \
        f"the per-file print changed: {user_line}"


def test_the_journal_beside_gate_to_land_lag_and_not_a_new_store(lmrd, gtl):
    """`--record` writes into the store the loop already has, not a new one.

    Both recorders resolve through `scripts.automod.state.STATE_DIR`, so the parent
    directories must be the same object's same path — and the file must not be the
    sibling's own. A new store is what the item rules out ("No new store"), and a
    journal inside the checkout is what #1667 has the guardian alerting on hourly.
    """
    assert lmrd.RECORD_NAME.endswith(".jsonl") and lmrd.RECORD_NAME != gtl.RECORD_NAME
    assert lmrd.record_path().parent == gtl.record_path().parent
    assert lmrd.record_path().name == lmrd.RECORD_NAME
    assert ROOT not in lmrd.record_path().parents, "the journal must not live in the tree"
    from scripts.automod import state as S
    assert lmrd.record_path().parent == Path(S.STATE_DIR)


# ── clause 2: the divisor is gross inflow, never net ────────────────────────────


def test_a_net_negative_file_still_reports_finite_clearance_days(tmp_path, lmrd):
    """Gross in 300 B/day, gross out 366.7 B/day, net −66.7 — and a real number out.

    Fixture: USER.md 3,000 B 40 days back (outside the window, so its creation is
    not inflow), 12,000 B 12 days back, 1,000 B 2 days back — gross in 9,000 B, gross
    out 11,000 B, over 30 days. Divided by NET the clearance would come out negative
    (a file with 15,384 B left under the line reported as already past it); divided
    by the gross inflow it is 51.3 days. This is the whole reason the instrument
    exists: USER.md's live net is about −2 B/day and reads healthy forever while
    appends keep hitting the ceiling.
    """
    vault = make_vault(tmp_path, [
        (40, {"USER.md": body(3000), "MEMORY.md": body(10), "SOUL.md": body(10)}),
        (12, {"USER.md": body(12000)}),
        (2, {"USER.md": body(1000)}),
    ])
    rows = lmrd.dose_rows(vault, 30, NOW)
    user = next(r for r in rows if r["file"] == "USER.md")

    assert user["bytes_in"] == pytest.approx(9000 / 30), user
    assert user["bytes_out"] == pytest.approx(11000 / 30, abs=0.001), user
    assert user["net"] < 0, "the fixture must actually be net-negative to be worth anything"
    assert user["headroom_days"] == pytest.approx((16_384 - 1000) / 300.0, abs=0.01)
    assert user["headroom_days"] > 0, "a net divisor would have gone negative here"


def test_zero_gross_inflow_gives_no_headroom_number_and_no_alarm(tmp_path, lmrd):
    """A window with only retirements in it divides by nothing and alarms nobody.

    `bytes_in` is 0, so `headroom_days` is None by the same branch that makes SOUL.md
    null — the reading is not 0 days (which would scream) and not infinity (which
    would be a division error at the line above it). `alarm_lines` then has no number
    to compare, and `shortfall_bytes` reports 0.0 rather than raising.
    """
    vault = make_vault(tmp_path, [
        (20, {"USER.md": body(5000), "MEMORY.md": body(10), "SOUL.md": body(10)}),
        (5, {"USER.md": body(3000)}),
    ])
    rows = lmrd.dose_rows(vault, 10, NOW)
    user = next(r for r in rows if r["file"] == "USER.md")

    assert user["bytes_in"] == 0.0, user
    assert user["bytes_out"] == pytest.approx(2000 / 10), "the outflow half is still reported"
    assert user["net"] < 0
    assert user["headroom_days"] is None, user
    assert lmrd.alarm_lines(rows) == []
    assert lmrd.shortfall_bytes(user) == 0.0


# ── clause 3: the window is part of the row, because it changes the answer ─────


def test_the_same_fixture_reads_two_different_values_at_two_windows(tmp_path, lmrd):
    """Same bytes, same formula, two windows, two shipped rows.

    The triage measured USER.md at 12.4 / 8.1 / 5.2 / 0.14 days over 30 / 45 / 60 /
    84 days: a number quoted without its window is not comparable to anything, so
    the series has to carry it. Here the same vault read at 10 and 30 days must
    disagree, and each persisted row must name the window behind its own value —
    which is what a later reader compares, not prose.
    """
    steps = [
        (20, {"USER.md": body(3000), "MEMORY.md": body(10), "SOUL.md": body(10)}),
        (12, {"USER.md": body(3100)}),
        (2, {"USER.md": body(3180)}),
    ]
    vault_a = make_vault(tmp_path, steps, name="vault-a")
    vault_b = make_vault(tmp_path, steps, name="vault-b")
    rows_10 = lmrd.dose_rows(vault_a, 10, NOW)
    rows_30 = lmrd.dose_rows(vault_b, 30, NOW)
    u10 = next(r for r in rows_10 if r["file"] == "USER.md")
    u30 = next(r for r in rows_30 if r["file"] == "USER.md")

    # Window 10 sees only the day-2 commit (80 B / 10 d); window 30 sees the
    # creation and both grows (3,180 B / 30 d), so the short window reads the slower
    # rate here and the longer clearance.
    assert u10["bytes_in"] == pytest.approx(8.0), u10
    assert u30["bytes_in"] == pytest.approx(106.0), u30
    assert u10["headroom_days"] != u30["headroom_days"], (u10, u30)
    assert u10["headroom_days"] > u30["headroom_days"]

    state_a, state_b = tmp_path / "state-10", tmp_path / "state-30"
    lmrd.record(rows_10, state_a)
    lmrd.record(rows_30, state_b)
    disk_a = [json.loads(ln) for ln in lmrd.record_path(state_a).read_text().splitlines()]
    disk_b = [json.loads(ln) for ln in lmrd.record_path(state_b).read_text().splitlines()]
    got_a = {r["file"]: r for r in disk_a}
    got_b = {r["file"]: r for r in disk_b}
    assert {r["window_days"] for r in disk_a} == {10}, disk_a
    assert {r["window_days"] for r in disk_b} == {30}, disk_b
    assert got_a["USER.md"]["headroom_days"] != got_b["USER.md"]["headroom_days"], (
        "the persisted rows lost the window difference: the rate got published "
        "without the thing that produced it")
    # A SOUL.md row is the null case at both windows, so the difference cannot be
    # an artefact of the two files disagreeing about whether they have a ceiling.
    assert got_a["SOUL.md"]["headroom_days"] is None
    assert got_b["SOUL.md"]["headroom_days"] is None


# ── clause 4: the alarm fires, names the shortfall, and touches nothing ────────


def _alarm_fixture(tmp_path: Path) -> Path:
    """USER.md pushed to 16,300 B of its 16,384 B ceiling by one +12,300 B commit.

    The file is created 40 days back — deliberately outside the 30-day window, so the
    window holds exactly one in-window delta of +12,300 B = 410.0 B/day of gross
    inflow against 84 B of clearance: 0.205 days, under the 1.0-day cut. The
    magnitude is deliberate — it is the shape of the September regime the threshold
    is actually for (974.9 B/day gross with a +12,063 B growth commit read 0.14 days
    live), not a scaled toy. MEMORY.md and SOUL.md are created in the same early
    commit and never touched again, so their dose is 0, their `headroom_days` is
    null, and the alarm line below names exactly one file.
    """
    return make_vault(tmp_path, [
        (40, {"USER.md": body(4000), "MEMORY.md": body(2000), "SOUL.md": body(1000)}),
        (5, {"USER.md": body(16300)}),
    ])


def test_the_alarm_line_names_the_file_the_shortfall_and_the_days(tmp_path, lmrd):
    """One alarming file, one line, and the three numbers a human can act on.

    Graded on the unit path; the subprocess run below grades the same fixture
    across the process boundary. `shortfall_b` is 326.0 = 1.0 day x 410 B/day − 84 B
    of clearance: the size of the retire that would settle it, which "0 B left" alone
    does not say.
    """
    vault = _alarm_fixture(tmp_path)
    rows = lmrd.dose_rows(vault, 30, NOW)
    lines = lmrd.alarm_lines(rows)

    assert len(lines) == 1, lines
    line = lines[0]
    assert line.startswith("ALARM USER.md:")
    assert "headroom_days=0.205" in line, line
    assert "clearance_b=84" in line, line
    assert "shortfall_b=326.0" in line, line
    assert "bytes_in=410.0" in line and "window_days=30" in line, line
    assert "#1881" in line and "trims nothing" in line, line
    assert lmrd.shortfall_bytes(next(r for r in rows if r["file"] == "USER.md")) == 326.0


def test_an_alarming_run_exits_1_and_leaves_every_loaded_file_byte_identical(tmp_path, lmrd):
    """The alarm is a verdict, not an action (#1881: never make room).

    Runs the shipped script as a child process — the boundary the daily job actually
    crosses — against a fixture where a trim is the only remedy that would help, and
    asserts three things: the child exits 1 on the verdict, its stdout carries the
    ALARM line for USER.md and only one, and every byte of all three loaded files is
    unchanged afterwards, with no new path in `lloyd/`. An instrument that "helpfully"
    retired lines to fix the number it was reporting would fail this node on the
    hash, which is the only assertion here that cannot be satisfied by a print.
    """
    vault = _alarm_fixture(tmp_path)
    before = {f: hashlib.sha256((vault / "lloyd" / f).read_bytes()).hexdigest()
              for f in LOADED}
    listing = sorted(p.name for p in (vault / "lloyd").iterdir())
    state = tmp_path / "state"

    # A stand-in MAIN checkout, because this node records (`--record`) and #4311's
    # guard refuses a record from a worktree — which is what `ROOT` is under a gate.
    # The verdict this node pins is 1 for a red READING, so it has to be a tree the
    # script is allowed to write from; the refusal path is pinned beside it.
    main = make_tree(tmp_path, gitfile=False, name="main-checkout-alarm")
    r = run_record(main, vault, state)
    out = r.stdout + r.stderr

    assert r.returncode == 1, f"the alarm did not reach the exit code\n{out}"
    assert out.count("ALARM ") == 1, out
    assert "ALARM USER.md:" in out, out
    assert "ALARM MEMORY.md" not in out and "ALARM SOUL.md" not in out, out
    for filename, digest in before.items():
        assert hashlib.sha256((vault / "lloyd" / filename).read_bytes()).hexdigest() \
            == digest, f"{filename} was modified by an alarming run"
    assert sorted(p.name for p in (vault / "lloyd").iterdir()) == listing
    assert lmrd.record_path(state).exists(), "the reading is recorded even on a red verdict"


# ── the reading itself must be able to be wrong ────────────────────────────────


def test_a_growing_window_shrinks_the_reported_clearance_days(tmp_path, lmrd):
    """The direction of the instrument, pinned separately from its arithmetic.

    One commit of +500 B, measured over 5 and over 50 days: 100 B/day against 10, so
    the 5-day reading must be the smaller clearance. If a refactor ever divided by the
    commit span instead of the declared window, both readings would come out equal
    and every clause above would still pass on the number — this node is the one that
    notices the window stopped meaning anything.
    """
    steps = [(1, {"USER.md": body(500), "MEMORY.md": body(10), "SOUL.md": body(10)})]
    vault = make_vault(tmp_path, steps, name="vault-dir")
    row_5 = next(r for r in lmrd.dose_rows(vault, 5, NOW) if r["file"] == "USER.md")
    row_50 = next(r for r in lmrd.dose_rows(vault, 50, NOW) if r["file"] == "USER.md")

    assert row_5["bytes_in"] == pytest.approx(100.0), row_5
    assert row_50["bytes_in"] == pytest.approx(10.0), row_50
    assert row_5["headroom_days"] < row_50["headroom_days"], (row_5, row_50)


def test_an_unreadable_vault_raises_rather_than_reporting_zero(tmp_path, lmrd):
    """A git failure is a crash, never a quiet `bytes_in=0`.

    Zero inflow is a legitimate reading and the alarm is shut by it, so a probe that
    returned zeros on a broken repository would be indistinguishable from the file
    nobody is writing to — the exact ambiguity #4122 was filed to remove. A
    nonexistent vault must raise.
    """
    with pytest.raises((RuntimeError, OSError, subprocess.SubprocessError)):
        lmrd.dose_row(tmp_path / "no-such-vault", "USER.md", 30, NOW)


def test_a_nonpositive_window_is_refused_not_divided_by(tmp_path, lmrd):
    """`window_days=0` is the one argument that could make the row a lie.

    The divisor is the declared window, so 0 is a division by zero and a negative one
    would publish a negative rate with an alarming clearance. Both must be refused by
    name rather than reaching the arithmetic.
    """
    vault = make_vault(tmp_path, [(5, {"USER.md": body(500), "MEMORY.md": body(10),
                                      "SOUL.md": body(10)})])
    for bad in (0, -30):
        with pytest.raises(ValueError, match="window_days"):
            lmrd.gross_flux(vault, "lloyd/USER.md", bad, NOW)
