"""backup-graph.sh must refuse to snapshot a graph whose state it cannot verify.

The script is driven through a fake $HOME so it never touches the real
_pipeline. Each case builds the minimum tree the script reads:
<HOME>/lloyd/_pipeline/{vault-derived/{facts,kg.sqlite}, memory-graph}.

Two refusals are pinned here, one per unreadable input: the store will not open,
and there is no usable active-edge baseline. The second is what #917 is about —
the guard used to skip itself silently when graph-baseline.json was missing or
unparseable, which is the exact state the 2026-08-22 destruction left behind.
"""
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.kg_store import KGStore  # noqa: E402

SCRIPT = ROOT / "scripts" / "backup" / "backup-graph.sh"


def _baseline_path(home: Path) -> Path:
    return home / "lloyd-data" / "_pipeline" / "memory-graph" / "graph-baseline.json"


def _tarballs(home: Path) -> list[Path]:
    return sorted((home / "lloyd-data" / "_pipeline" / "backups" / "daily").glob("graph-*.tar.gz"))


def _fake_home(tmp_path: Path, *, active: int = 0, expired: int = 0,
               baseline=None, baseline_raw: str | None = None,
               corrupt: bool = False) -> Path:
    """Build the tree the script reads.

    `baseline` is an active_edges count to record, a full dict to write as the
    baseline JSON verbatim, or None to leave the file absent. `baseline_raw`
    writes the file exactly as given, for a body that is not valid JSON at all,
    and wins over `baseline`.
    """
    home = tmp_path / "home"
    pipeline = home / "lloyd-data" / "_pipeline"
    facts = pipeline / "vault-derived" / "facts"
    facts.mkdir(parents=True)
    (pipeline / "memory-graph").mkdir()
    db = pipeline / "vault-derived" / "kg.sqlite"
    if corrupt:
        db.write_bytes(b"not a database, just 48 bytes of noise ........")
    else:
        s = KGStore(db)
        for i in range(active):
            s.edges.add({"source": f"A{i}", "target": f"B{i}", "type": "uses"}, origin="test")
        for i in range(expired):
            eid = s.edges.add({"source": f"X{i}", "target": f"Y{i}", "type": "uses"}, origin="test")
            s.edges.expire(eid, "test")
        s.close()
    if baseline_raw is not None:
        _baseline_path(home).write_text(baseline_raw, encoding="utf-8")
    elif isinstance(baseline, dict):
        _baseline_path(home).write_text(json.dumps(baseline), encoding="utf-8")
    elif baseline is not None:
        _baseline_path(home).write_text(
            json.dumps({"active_edges": baseline}), encoding="utf-8"
        )
    return home


def _run(home: Path, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, HOME=str(home), LLOYD_PYTHON=sys.executable,
               LLOYD_DATA=str(home / "lloyd-data"))
    env.update(extra_env or {})
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True,
                          cwd=str(ROOT))


def test_the_script_is_executable_because_systemd_execs_it_directly(tmp_path):
    """lloyd-graph-backup.service runs `ExecStart=%h/lloyd/scripts/backup/backup-graph.sh`.

    A oneshot unit execs the file itself rather than handing it to a shell, so
    the mode bit is part of the interface and an editor that rewrote the file
    without it would take the nightly backup down with EACCES. The tests here all
    invoke `bash <script>`, which passes whatever the mode is, so nothing else
    in this file would notice.
    """
    assert SCRIPT.stat().st_mode & 0o111, "backup-graph.sh must stay executable for systemd"

    # And the exec path itself: same invocation shape as the unit, no shell
    # named in front of it. A missing bit surfaces here as EACCES.
    home = _fake_home(tmp_path / "direct", active=5, baseline=None)
    env = dict(os.environ, HOME=str(home), LLOYD_PYTHON=sys.executable,
               LLOYD_DATA=str(home / "lloyd-data"))
    result = subprocess.run([str(SCRIPT)], env=env, capture_output=True, text=True,
                            cwd=str(ROOT))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "REFUSING" in result.stderr


def _guard_line(result: subprocess.CompletedProcess) -> str:
    """The stdout line reporting the counts the guard compared."""
    return next(line for line in result.stdout.splitlines() if "active edges" in line)


def test_refuses_below_half_baseline_and_keeps_previous_tarball(tmp_path):
    home = _fake_home(tmp_path, active=100, baseline=3894)
    dest = home / "lloyd-data" / "_pipeline" / "backups" / "daily"
    dest.mkdir(parents=True)
    previous = dest / "graph-20260101.tar.gz"
    previous.write_bytes(b"previous snapshot")

    result = _run(home)

    assert result.returncode == 1, result.stderr
    assert "REFUSING" in result.stderr
    assert "below 50%" in result.stderr
    assert previous.read_bytes() == b"previous snapshot"
    assert list(dest.glob("graph-*.tar.gz")) == [previous], "no new tarball must be written"


def test_refuses_on_an_unreadable_store(tmp_path):
    home = _fake_home(tmp_path, baseline=3894, corrupt=True)
    result = _run(home)
    assert result.returncode == 1
    assert "will not open" in result.stderr


def test_refuses_when_no_baseline_exists_and_keeps_previous_tarball(tmp_path):
    """#917: an absent reference is a refusal, not a skipped comparison.

    This replaces test_no_baseline_means_no_comparison, which asserted
    returncode == 0 for exactly this state and so pinned the fail-open. The
    acceptance for #917 is that a missing baseline is refused the way an
    unreadable store already was: exit 1, the path named, no tarball, and the
    snapshot window not rotated.
    """
    home = _fake_home(tmp_path, active=5, baseline=None)
    dest = home / "lloyd-data" / "_pipeline" / "backups" / "daily"
    dest.mkdir(parents=True)
    previous = dest / "graph-20260101.tar.gz"
    previous.write_bytes(b"previous snapshot")

    result = _run(home)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "REFUSING" in result.stderr
    assert str(_baseline_path(home)) in result.stderr, result.stderr
    assert previous.read_bytes() == b"previous snapshot"
    assert _tarballs(home) == [previous], "no new tarball may be written without a reference"


def test_refuses_when_the_baseline_is_not_json(tmp_path):
    home = _fake_home(tmp_path, active=3900, baseline_raw="not json at all {{{")
    result = _run(home)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "REFUSING" in result.stderr
    assert str(_baseline_path(home)) in result.stderr
    assert _tarballs(home) == []


def test_refuses_when_the_baseline_carries_no_active_edges(tmp_path):
    home = _fake_home(tmp_path, active=3900,
                      baseline_raw=json.dumps({"recorded_at": "2026-09-16T00:03:08+00:00"}))
    result = _run(home)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "REFUSING" in result.stderr
    assert str(_baseline_path(home)) in result.stderr
    assert _tarballs(home) == []


def test_refuses_when_the_baseline_is_not_positive(tmp_path):
    """A reference of 0 edges compares against nothing, so it is no reference."""
    home = _fake_home(tmp_path, active=3900,
                      baseline_raw=json.dumps({"active_edges": 0,
                                               "recorded_at": "2026-09-16T00:03:08+00:00"}))
    result = _run(home)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "REFUSING" in result.stderr
    assert str(_baseline_path(home)) in result.stderr
    assert _tarballs(home) == []


def test_the_script_reads_no_override_switch_at_all():
    """The refusal has no escape hatch, and no knob that could grow one.

    Enumerating switch names would only prove those names do nothing, so this
    reads the property instead: every variable the script expands is either its
    own path/loop local or one of the two documented inputs, and none of them is
    a permission flag. Adding `ALLOW_NO_BASELINE=1` later would trip this on the
    name alone, without anyone having to think of the name first.
    """
    import re

    source = SCRIPT.read_text(encoding="utf-8")
    # HOME and LLOYD_PYTHON are the only two the script takes from the
    # environment; everything else in the set is its own path/loop local.
    expanded = set(re.findall(r"\$\{?([A-Z][A-Z0-9_]*)", source))
    switches = {n for n in expanded if re.search(r"ALLOW|SKIP|FORCE|IGNORE|OVERRIDE|BYPASS|NO_BASELINE", n)}
    assert switches == set(), f"guard override switches appeared: {sorted(switches)}"
    assert "HOME" in expanded and "LLOYD_PYTHON" in expanded
    assert not re.search(r"os\.environ|getenv", source), "the guard must not read the environment either"


def test_no_environment_override_disables_the_baseline_check(tmp_path):
    """And the refusal still refuses with the most plausible names set."""
    home = _fake_home(tmp_path, active=5, baseline=None)
    result = _run(home, extra_env={
        "ALLOW_NO_BASELINE": "1",
        "BACKUP_GRAPH_SKIP_GUARD": "1",
    })
    assert result.returncode == 1, result.stdout + result.stderr
    assert "REFUSING" in result.stderr
    assert _tarballs(home) == []


def test_writes_snapshot_when_at_baseline(tmp_path):
    home = _fake_home(tmp_path, active=3900, baseline=3894)
    result = _run(home)
    assert result.returncode == 0, result.stderr
    dest = home / "lloyd-data" / "_pipeline" / "backups" / "daily"
    tarballs = list(dest.glob("graph-*.tar.gz"))
    assert len(tarballs) == 1
    assert "3900 active edges" in result.stdout


def test_success_line_names_when_the_baseline_was_recorded(tmp_path):
    home = _fake_home(tmp_path, active=3900,
                      baseline={"active_edges": 3894, "recorded_at": "2026-09-16T00:03:08+00:00"})
    result = _run(home)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _guard_line(result) == (
        "backup-graph: 3900 active edges (baseline 3894, recorded 2026-09-16T00:03:08+00:00)"
    )


def test_success_line_says_unknown_when_recorded_at_is_absent(tmp_path):
    home = _fake_home(tmp_path, active=3900, baseline=3894)
    result = _run(home)
    assert result.returncode == 0, result.stdout + result.stderr
    line = _guard_line(result)
    assert "baseline 3894" in line
    assert "recorded unknown" in line


def test_the_baseline_the_sweep_actually_writes_passes_the_guard(tmp_path):
    """The reference is written by another program and read by this one.

    entity-resolution-sweep.py owns graph-baseline.json: `update_baseline` is a
    max-ratchet that stamps `recorded_at` with a timezone-aware isoformat. The
    shell script never imports it, so nothing but this test puts the writer's
    real bytes in front of the guard.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "entity_resolution_sweep", ROOT / "scripts" / "memory" / "entity-resolution-sweep.py")
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)

    home = _fake_home(tmp_path, active=3900)  # no baseline file yet
    assert sweep.update_baseline(3900, _baseline_path(home)) == 3900
    recorded = json.loads(_baseline_path(home).read_text())["recorded_at"]
    assert recorded.startswith("20"), recorded

    result = _run(home)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _guard_line(result) == (
        f"backup-graph: 3900 active edges (baseline 3900, recorded {recorded})")


def test_success_line_prints_a_spaced_timestamp_without_reshaping_it(tmp_path):
    """A stamp written with a space in it survives the field split intact.

    The guard reads the line with `read -r ACTIVE BASE RECORDED`, and a
    hand-edited baseline can carry any string, so the value must print as
    written rather than have its separator deleted.
    """
    stamp = "2026-09-16 00:03:08 +0000"
    home = _fake_home(tmp_path, active=3900,
                      baseline={"active_edges": 3894, "recorded_at": stamp})
    result = _run(home)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _guard_line(result) == f"backup-graph: 3900 active edges (baseline 3894, recorded {stamp})"


def test_snapshot_contains_a_restorable_store_and_a_json_export(tmp_path):
    home = _fake_home(tmp_path, active=120, baseline=100)
    assert _run(home).returncode == 0
    tarball = next(iter(_tarballs(home)))
    with tarfile.open(tarball) as tf:
        names = tf.getnames()
        db_member = next(n for n in names if n.endswith("kg.sqlite"))
        assert any(n.endswith("json-export/_relationships.json") for n in names)
        assert any(n.endswith("json-export/entity-aliases.json") for n in names)
        tf.extract(db_member, tmp_path / "restored", filter="data")
    restored = KGStore(tmp_path / "restored" / db_member)
    assert restored.edges.count() == 120
    assert restored.integrity_check() == "ok"
    restored.close()


def test_staging_dir_is_cleaned_up(tmp_path):
    home = _fake_home(tmp_path, active=120, baseline=100)
    assert _run(home).returncode == 0
    backups = home / "lloyd-data" / "_pipeline" / "backups"
    assert not list(backups.glob(".staging-*"))


def test_expired_edges_do_not_count_as_active(tmp_path):
    home = _fake_home(tmp_path, active=100, expired=4000, baseline=3894)
    result = _run(home)
    assert result.returncode == 1
    assert "100 active edges" in result.stderr


# ───────────────────────── the pair's systemd units (#2864) ──────────────────────────────
#
# `backup-graph.sh` had a tracked payload and an untracked pair of units: both files lived
# only as plain copies in ~/.config/systemd/user/ (mtime 2026-09-03), so a rebuild either
# wrote them from SETUP.md's inline text — which had already dropped the live service's
# `Documentation=` line — or lost the nightly snapshot of the one store that cannot be
# regenerated. #2864 tracked them.
#
# What these nodes read, precisely: the TREE (git's index, the unit files, the prose sites);
# the live unit FILES under ~/.config/systemd/user/ byte-for-byte, which is the one live
# read that belongs here because reproducing it is exactly what a rebuild must do; and the
# real checker run over this checkout and this box's deploy directory. What none of them
# reads is live unit STATE: `is-enabled`, `is-active`, and whether the deployed file is a
# symlink or a plain copy are asserted nowhere, because re-linking those copies is #2714's
# by-design finding and a post-landing host step — `tests/test_cert_renew_units.py` declines
# enabledness for the same reason. The two live reads do mean these nodes go red on a box
# whose deploy directory has drifted or been wiped, which is the honest price of pinning an
# equality across that boundary instead of describing it in prose; each node says so in its
# own failure message.

UNIT_DIR = ROOT / "agent-services" / "systemd"
SERVICE_UNIT = "lloyd-graph-backup.service"
TIMER_UNIT = "lloyd-graph-backup.timer"
# The live deploy directory, read by the equality node only, and read as FILES: `systemctl`
# is never called, so the box is not asked what it currently runs.
LIVE_UNIT_DIR = Path.home() / ".config" / "systemd" / "user"
#: Stem of the tracked pair, for prose checks that must see the bare name too — the
#: installer's static list writes units without a suffix, so a filename-scoped search
#: misses the one mention that is supposed to remain.
UNIT_STEM = "lloyd-graph-backup"


def _parse(text: str) -> dict[str, dict[str, str]]:
    """Minimal systemd-unit INI view: section -> {key: last value}.

    Local rather than imported from `tests/test_egress_window_unit.py`, which carries the
    same helper for its own pair, because importing one test module to borrow a helper makes
    the other file's collection a dependency of this one's. Comments are skipped because
    systemd skips them — the tracked timer's 05:30 comment is load-bearing prose and must
    not read as a directive. No unit here uses a continuation line, so none is parsed for
    one.
    """
    out: dict[str, dict[str, str]] = {}
    section = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            out.setdefault(section, {})
            continue
        if section is not None and "=" in line:
            key, _, value = line.partition("=")
            out[section][key.strip()] = value.strip()
    return out


def _git_ls_files(pattern: str) -> list[str]:
    """Repo-relative paths git tracks under `pattern`, run from THIS checkout.

    `git ls-files` answers from the index, so the answer is the tree the suite is grading,
    not the machine's copy — which is what makes this node work in a worktree.
    """
    r = subprocess.run(["git", "ls-files", pattern], capture_output=True, text=True,
                       cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    return sorted(r.stdout.split())


def test_the_graph_backup_units_are_tracked_verbatim_copies_of_the_live_pair():
    """Clause 1: the units are in git, byte-identical to what this box runs today.

    `diff` against ~/.config/systemd/user/<name> being empty is the whole point of this
    node, and it is the reason the tracked files are a COPY of the live text rather than a
    tidy-up of it: the live .service carries a `Documentation=` line SETUP.md's inline block
    omitted, and the live .timer's comments carry the reason for 05:30 that the doc block
    flattened. Editing either side would make the sentence false, and a rebuild that
    silently changed what runs is the failure this whole item is about.

    The `.service` must stay `static`: no `[Install]` section, because its own timer starts
    it — `install-services.sh`'s arming filter exists precisely so the installer does not
    `enable` such a unit and die on the error (#2541's sibling convention is
    `lloyd-persistence-window.service`). A rebuild that "helpfully" added one would put a
    oneshot backup into `default.target.wants/` and run it at every login.
    """
    tracked = _git_ls_files("agent-services/systemd/lloyd-graph-backup.*")
    assert tracked == [f"agent-services/systemd/{SERVICE_UNIT}",
                       f"agent-services/systemd/{TIMER_UNIT}"], tracked

    for name in (SERVICE_UNIT, TIMER_UNIT):
        repo = (UNIT_DIR / name).read_text(encoding="utf-8")
        live = LIVE_UNIT_DIR / name
        assert live.is_file(), (
            f"{live} is gone, so this node's byte-equality claim has nothing to "
            "compare against — re-pin it to the tracked copy alone and say why")
        assert repo == live.read_text(encoding="utf-8"), (
            f"agent-services/systemd/{name} is not byte-identical to {live}")

    parsed = _parse((UNIT_DIR / SERVICE_UNIT).read_text(encoding="utf-8"))
    assert "Install" not in parsed, (
        f"{SERVICE_UNIT} gained an [Install] section, so it is no longer static and the "
        f"installer would arm it: {parsed}")
    assert parsed["Service"]["ExecStart"].endswith(
        "scripts/backup/backup-graph.sh"), parsed["Service"]
    assert parsed["Unit"]["Documentation"] == "https://github.com/alansrobotlab/lloyd", (
        "the live unit carries this line and the tracked copy was required to keep it")

    timer = _parse((UNIT_DIR / TIMER_UNIT).read_text(encoding="utf-8"))
    assert timer["Timer"]["OnCalendar"] == "*-*-* 05:30:00", timer["Timer"]
    assert timer["Install"]["WantedBy"] == "timers.target", timer["Install"]


def test_the_health_checker_no_longer_calls_the_tracked_pair_orphans():
    """Clause 2: the deploy check stops reporting these two units as untracked.

    `scripts/service_health_check.py`'s orphan pass reports a live user unit with no file
    behind it in the tree it was launched from, so this is the real checker over the real
    tree and this box's deploy directory — the process boundary the fix actually moves. Its
    remediation text used to name `agent-services/systemd/` as the place the pair should
    live, and the assertion below is that sentence becoming true.

    Two things are deliberately NOT asserted, both by the item's own clause: that the pair's
    `unitlink:` rows are healthy (the deployed files are still the plain copies that predate
    tracking, so `copy:` standing is #2714's by-design finding until someone re-links them),
    and that `Deploy:` turns green (freeclip's row and the other plain copies keep it
    degraded, and a node that asserted green would be asserting a host state this diff does
    not cause). The freeclip row IS asserted: it is the witness that the checker really
    walked this box's live directory, so the absent graph-backup rows above are caused by
    the repo and not by a check that silently stopped running.
    """
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "service_health_check.py"),
                          "--category", "deploy"],
                         capture_output=True, text=True, cwd=str(ROOT), timeout=180)
    assert out.returncode in (0, 1), f"{out.returncode}:\n{out.stdout[-2000:]}{out.stderr[-800:]}"

    orphans = [ln for ln in out.stdout.splitlines()
               if "orphanunit:lloyd-graph-backup" in ln]
    assert orphans == [], (
        "the deploy check still calls the tracked pair orphans:\n" + "\n".join(orphans))

    pair_rows = sorted(ln.split("] ", 1)[1].split("—")[0].strip()
                       for ln in out.stdout.splitlines() if "lloyd-graph-backup." in ln)
    assert pair_rows == [f"unitlink:{SERVICE_UNIT}", f"unitlink:{TIMER_UNIT}"], (
        "the pair must now be graded as tracked-but-not-symlinked, one row each: "
        f"{pair_rows}")

    # Positive control: the orphan pass must be seen grading SOMETHING, or the empty
    # assertion above is a green the checker never granted. Named by property and not by
    # unit, because the one live-only unit on this box (`freeclip-a2dp-fix.service`) is
    # exactly what #2864's third owed entry has to rule on — tracking it or dispositioning
    # it by name here would make this node fail the day that ruling lands. If a future box
    # has no live-only user unit at all, plant one in a synthetic fleet the way
    # tests/test_service_health_check_unit_links.py's `fleet` fixture does, which is the
    # only durable way to hold a positive control over host state.
    others = [ln.split("] ", 1)[1].split("—")[0].strip()
              for ln in out.stdout.splitlines()
              if "orphanunit:" in ln and "lloyd-graph-backup" not in ln]
    assert others, (
        "the orphan pass graded no live-only unit at all, so `no orphan row for the "
        "tracked pair` is unproven here — plant a live-only unit in a synthetic fleet "
        "and pin the witness there instead")


def _paragraphs(text: str) -> list[str]:
    """Markdown blocks, split on blank lines — the unit a prose claim is true or false in.

    Paragraph-scoped rather than line-scoped because these docs wrap: the sentence
    `architecture/infrastructure.md` carried into #2864 about this pair spanned four lines,
    so a one-line grep for it returns nothing and a reviewer could read that 0 as absence.
    """
    return [b.strip() for b in re.split(r"\n[ \t]*\n", text) if b.strip()]


def _flat(text: str) -> str:
    """Whitespace-collapsed text, so a wrapped sentence can be matched as one string."""
    return " ".join(text.split())


def test_no_repo_sentence_still_tells_a_rebuilder_these_units_are_untracked():
    """Clause 5: the re-pointed prose is pinned, not just true today.

    Four sites told a rebuilder to write these units by hand or that nothing owned them:
    SETUP.md's "units are not in the repo — write them on a fresh install" with its two
    inline `ini` blocks, `architecture/infrastructure.md`'s gap sentence, the installer's
    arming-comment example, and the enabledness guard's header bullet. All four were
    re-pointed in #2864, and a claim that lives only in a diff is a claim the next round can
    quietly undo: the sentence that made this item necessary was itself a doc restoration
    (#1109) that later prose overwrote. So each site is pinned twice over — the false
    sentence must be absent AND the true pointer present in the same paragraph — because an
    absence-only node passes on a doc that deleted the section.

    Scope matters for the absence half: it is asserted over the paragraphs that name
    `lloyd-graph-backup`, not over the whole file. `SETUP.md` documents a whole rebuild and
    some other unit genuinely may be untracked; a doc-wide ban would be broken by an
    unrelated true sentence, and the third such edit buys a permanent skip.

    Every read here is of the tree the suite is grading — no subprocess, no box.
    """
    setup = (ROOT / "SETUP.md").read_text(encoding="utf-8")
    graph_paras = [p for p in _paragraphs(setup) if "lloyd-graph-backup" in p]
    assert graph_paras, (
        "SETUP.md no longer mentions the graph backup outside the enable line, so the "
        "rebuilder this paragraph exists to instruct has lost the units entirely")
    for para in graph_paras:
        for stale in ("not in the repo", "tracked nowhere", "write them on a fresh install"):
            assert stale not in para, (
                f"a paragraph naming lloyd-graph-backup still says it is {stale!r}:\n{para}")
    prose = " ".join(graph_paras)
    assert "tracked in `agent-services/systemd" in prose, (
        "SETUP.md no longer points the graph backup at the tracked unit dir; the paragraph "
        f"reads:\n{prose}")
    assert "install-services.sh" in prose, (
        "the pointer has to name the route that places the units, the way the qmd-cleanup "
        "paragraph does — naming a directory is not an install step")
    for name in (SERVICE_UNIT, TIMER_UNIT):
        assert f"# ~/.config/systemd/user/{name}" not in setup, (
            f"SETUP.md still carries an inline copy of {name}; a second source of the unit "
            "text is what drifted from what ran in the first place")
    assert f"enable --now {TIMER_UNIT}" in setup, (
        "the enable line is the whole install now; without it the paragraph points at a "
        "unit nobody arms")
    qmd = [p for p in _paragraphs(setup) if "lloyd-qmd-cleanup.timer" in p
           and "tracked in `agent-services/systemd" in _flat(p)]
    assert qmd, (
        "the qmd-cleanup paragraph this clause names as the shape to copy no longer has "
        "that shape, so the comparison above is matching a pattern nothing uses")

    infra = _flat((ROOT / "architecture" / "infrastructure.md").read_text(encoding="utf-8"))
    assert "a gap rather than a stale copy" not in infra, (
        "the sentence that reported this pair as tracked-only-on-the-box is back, and it is "
        "false: agent-services/systemd/ holds both files")
    for stale in ("exists only as a plain untracked file", "rebuild from `SETUP.md` loses it"):
        assert stale not in infra, f"infrastructure.md still says the pair {stale!r}"
    at = infra.find("lloyd-graph-backup.{service,timer}")
    assert at != -1, "infrastructure.md no longer lists the pair in the unit table at all"
    assert "agent-services/systemd/" in infra[at:at + 400], (
        "the paragraph naming the pair must say where its tracked copy lives, since that is "
        f"the fact #2864 changed: …{infra[at:at + 400]}")

    installer = (ROOT / "agent-services" / "setup" / "install-services.sh").read_text(
        encoding="utf-8")
    # Stem-scoped, not filename-scoped: rule 3's static list writes these units BARE
    # (`lloyd-graph-backup`), and a `lloyd-graph-backup.service`-shaped search misses the
    # one mention that is supposed to remain — a green from a pattern that matches nothing.
    linked_note = installer.split("LINKED=()")[0]
    assert UNIT_STEM not in linked_note, (
        "the arming comment is again justifying its existence by this pair, which the repo "
        "now owns — the example there has to be a unit the repo really does not own")
    named = [ln for ln in installer.splitlines() if f"`{UNIT_STEM}`" in ln]
    assert len(named) == 1 and "lloyd-cert-renew" in named[0], (
        f"the installer should name the pair exactly once, in the list of co-located "
        f"services that read static and must not be armed: {named}")

    guard = (ROOT / "scripts" / "maintenance" / "check-unit-enabledness.sh").read_text(
        encoding="utf-8")
    assert "tracked nowhere" not in guard, (
        "the guard's header is citing a live-only unit as its reason for one-directionality "
        "again; the rule stays and the witness must be a name, not box state")
    assert "test_unit_enabledness_check.py" in guard, (
        "the header has to name the node that owns the rule it states, or the next reader "
        "deletes the bullet along with the example that motivated it")

    data_home = (ROOT / "tests" / "test_data_home.py").read_text(encoding="utf-8")
    assert "tracked nowhere" not in data_home, (
        "the cutover fleet's membership rule is justified by a live-but-untracked witness "
        "again; #2864 moved that witness and the rule has to stand on its own wording")
