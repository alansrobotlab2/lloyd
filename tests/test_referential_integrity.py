"""#882: dead file cites in the always-loaded memory files are reported, never fixed."""
from __future__ import annotations

import hashlib
import importlib.util
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.maintenance import referential_integrity as ri  # noqa: E402
from scripts.vault import validate_okf as okf  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

MEMORY = """\
# Memory

- see `knowledge/software/mc-frontend-https-probe.md` for why http reads DOWN.
- the probe lives at `knowledge/software/present.md`.
- guardian alerts go through `notify.py:3`; the old `notify.py:999` cite drifted.
- `/goal` and `/state` are slash commands; `/v1/messages` is an endpoint.
- daily notes are `memory/YYYY-MM-DD.md`, sessions under `~/lloyd-data/sessions/`.
- `knowledge/old-thing.md` was deleted on purpose.
- `knowledge/other-thing.md` is kept dead on purpose <!-- ri:known-absent -->
```
cat `knowledge/in-a-fence.md`
```
"""

USER = """\
# User

- the trim is archived at `reviews/2026-09-14-user-md-trim-archive.md`.
- a sibling note: `notes/sibling.md`.
"""

MENTAL = "\n" * 142 + "- ambiguity sweep: `memory-graph/entity-ambiguous-2026-09-04.md`.\n"

DEAD = {"knowledge/software/mc-frontend-https-probe.md",
        "memory-graph/entity-ambiguous-2026-09-04.md", "notify.py:999"}


@pytest.fixture
def world(tmp_path):
    vault, repo, home = tmp_path / "vault", tmp_path / "repo", tmp_path / "home"
    for rel, text in {
        "lloyd/MEMORY.md": MEMORY,
        "lloyd/USER.md": USER,
        "memory/mental-models.md": MENTAL,
        "knowledge/software/present.md": "x\n",
        "reviews/2026-09-14-user-md-trim-archive.md": "x\n",
        "lloyd/notes/sibling.md": "x\n",
    }.items():
        (vault / rel).parent.mkdir(parents=True, exist_ok=True)
        (vault / rel).write_text(text)
    (repo / "agent-services" / "guardian").mkdir(parents=True)
    (repo / "agent-services" / "guardian" / "notify.py").write_text("a\nb\nc\n")
    (home / "lloyd-data" / "sessions").mkdir(parents=True)
    return {"vault": vault, "repo": repo, "home": home}


def _args(world):
    return ["--vault", str(world["vault"]), "--repo", str(world["repo"]),
            "--home", str(world["home"])]


def _by_target(world):
    return {r["target"]: r for r in ri.scan(**world)}


def test_every_record_carries_its_fields_and_its_check(world):
    records = ri.scan(**world)
    assert records
    for r in records:
        assert set(r) == {"file", "line", "kind", "target", "verdict", "how_checked"}
        assert r["how_checked"], r
        if r["verdict"] == "resolves":
            assert r["how_checked"].startswith("stat:"), r
    kinds = {r["target"]: r["kind"] for r in records}
    assert kinds["notify.py:3"] == "line"
    assert kinds["knowledge/software/present.md"] == "path"


def test_the_known_dead_cites_are_dangling(world):
    got = _by_target(world)
    for target in DEAD:
        assert got[target]["verdict"] == "dangling", got[target]
    assert got["knowledge/software/mc-frontend-https-probe.md"]["file"] == "lloyd/MEMORY.md"
    assert got["memory-graph/entity-ambiguous-2026-09-04.md"]["line"] == 143
    assert "beyond eof (3)" in got["notify.py:999"]["how_checked"]


def test_exit_code_and_the_fixture_self_check(world, capsys):
    assert ri.main(_args(world)) == 1
    expect = [a for t in sorted(DEAD) for a in ("--expect-dangling", t)]
    assert ri.main(_args(world) + expect) == 1
    # A fixture that stopped being reported (here: one that resolves) fails loudly.
    assert ri.main(_args(world) + ["--expect-dangling",
                                   "knowledge/software/present.md"]) == 2
    assert "expected dangling but not reported" in capsys.readouterr().err


def test_resolution_roots_and_what_is_never_a_path(world):
    got = _by_target(world)
    assert got["reviews/2026-09-14-user-md-trim-archive.md"]["how_checked"] == "stat:vault_root"
    assert got["notes/sibling.md"]["how_checked"] == "stat:citing_dir"
    assert got["~/lloyd-data/sessions/"]["how_checked"] == "stat:home"
    assert got["notify.py:3"]["how_checked"].startswith("stat:repo_basename")
    for noise in ("/goal", "/state", "/v1/messages", "memory/YYYY-MM-DD.md",
                  "knowledge/in-a-fence.md"):
        assert noise not in got
    for span in ("/goal", "/v1/audio/voice-clone", "memory/YYYY-MM-DD.md",
                 "~/.claude/projects/x/86b5c9d4-", "subliminal/", "uv pip install x",
                 "https://example.com/a.md", "notes.md"):
        assert ri.extract_target(span) is None, span


def test_documented_absence_is_exempt_and_does_not_fail_the_run(world):
    got = _by_target(world)
    assert got["knowledge/old-thing.md"]["verdict"] == "exempt"
    assert "absence prose" in got["knowledge/old-thing.md"]["how_checked"]
    assert got["knowledge/other-thing.md"]["verdict"] == "exempt"
    assert ri.EXEMPT_MARKER in got["knowledge/other-thing.md"]["how_checked"]

    # Only exempt cites left -> a clean exit.
    mem = world["vault"] / "lloyd" / "MEMORY.md"
    mem.write_text("\n".join(l for l in MEMORY.splitlines()
                             if "probe.md" not in l and "999" not in l))
    (world["vault"] / "memory" / "mental-models.md").write_text("nothing cited\n")
    assert ri.main(_args(world)) == 0


def _snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha1(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_a_run_writes_only_its_report_and_counts_new_dangling(world):
    vault = world["vault"]
    report = vault / "autonomy" / "referential-integrity-latest.md"
    before = _snapshot(vault)
    ri.main(_args(world) + ["--report", str(report)])
    after = _snapshot(vault)
    assert set(after) - set(before) == {"autonomy/referential-integrity-latest.md"}
    assert {k: after[k] for k in before} == before

    text = report.read_text()
    assert "- dangling: 3" in text
    assert "- exempt: 2" in text
    assert "no previous report" in text

    (vault / "knowledge" / "software" / "present.md").unlink()
    ri.main(_args(world) + ["--report", str(report)])
    text = report.read_text()
    assert "- dangling: 4" in text
    assert "- newly dangling since the previous report: 1" in text
    assert "`knowledge/software/present.md` — " in text and "**(new)**" in text


def _run_gate(root: Path, dir_name: str) -> subprocess.CompletedProcess:
    """The weekly OKF gate over one directory of a fixture vault, as a subprocess.

    Crossed as a process boundary on purpose: `scripts/vault/validate_okf.py` is the thing
    that decides whether a note is conformant, and it is run as a command line by the
    nightly and by the `okf-conformance-check` skill. Importing a helper here would let
    this node agree with a private function while the gate that actually reports the
    violation to a human kept saying `no parseable frontmatter block`, which is exactly the
    disagreement #2326 was filed about: four of this script's own reports sat under a green
    scan because nothing between the writer and the reader checked the bytes.
    """
    return subprocess.run(
        [sys.executable, "-m", "scripts.vault.validate_okf",
         "--root", str(root), "--dir", dir_name],
        cwd=REPO_ROOT, capture_output=True, text=True)


def test_the_report_the_script_writes_opens_with_a_frontmatter_block_the_gate_accepts(world):
    """The generator owns its frontmatter: `--report` bytes pass the real OKF gate.

    Four files under `~/obsidian/autonomy/` — `referential-integrity-latest.md` and its
    three dated copies — were non-conformant by construction on 2026-10-07, because
    `render_report` started its output at an H1. `validate_okf.py --dir autonomy` reported
    all four as `no parseable frontmatter block` and exited 1 while the seven-segment
    `segment_scan` said `total missing: 0`. The two scanners disagreed, and the one that
    was right scans a directory the other one never walks — which is the denominator half
    of the same item.

    The keys are asserted on the parsed block, not on the text: a `segment:` that is a
    list, or a `tags:` written as a bare string, would satisfy a `read_text().count("---")`
    check and still be rejected by the gate. `generated_at` is tied to the run stamp in the
    body rather than to a fresh `utcnow()` — two timestamps in one file drift apart, and a
    dated copy made by autonomy task #94's `cp` step then disagrees with its own header.
    """
    vault = world["vault"]
    report = vault / "autonomy" / "referential-integrity-latest.md"
    ri.main(_args(world) + ["--report", str(report)])
    text = report.read_text()

    block = okf.STRICT_FM_RE.match(text)
    assert block, f"no frontmatter fence at offset 0: {text[:80]!r}"
    fm = yaml.safe_load(block.group(1))
    assert isinstance(fm, dict), fm
    assert fm["segment"] == "autonomy", fm
    assert isinstance(fm["tags"], list) and len(fm["tags"]) >= 1, fm
    assert isinstance(fm["summary"], str) and fm["summary"].strip(), fm
    assert fm["generated_at"] == re.search(r"^Run (\S+) by", text, re.M).group(1), (
        "the frontmatter stamp and the body's run stamp are two different times")
    assert set(fm) == {"type", "segment", "tags", "generated_at", "summary"}, (
        "the block declares no counts of its own: a `dangling:` key here would be a second "
        "source of truth for a number the body already carries")
    assert text.count("---\n") == 2, "exactly one fence pair: an inner one breaks the regex"
    assert "# Referential integrity — loaded memory" in text, "the H1 was displaced, not prefixed"

    checked = _run_gate(vault, "autonomy")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "referential-integrity-latest.md" not in checked.stdout, checked.stdout

    # The control that says the gate is grading these bytes and not agreeing vacuously: the
    # same report with its fence cut off must come back as the violation the nightly
    # reported on each of 2026-10-04, 10-05 and 10-06.
    stripped = text[text.index("\n---\n") + len("\n---\n"):]
    assert stripped.startswith("\n# Referential integrity"), stripped[:40]
    report.write_text(stripped)
    assert not okf.STRICT_FM_RE.match(stripped), "the control must fail the fence check itself"
    refused = _run_gate(vault, "autonomy")
    assert refused.returncode == 1, refused.stdout + refused.stderr
    assert "no parseable frontmatter block" in refused.stdout, refused.stdout


def test_the_fence_sits_in_front_of_the_marker_the_new_dangling_memory_reads(world):
    """Two runs over one path still report only the newly-dangling cite, with the fence in place.

    The previous run's cite list is read back out of the report file itself, from the
    `<!-- ri:dangling [...] -->` state marker `render_report` closes with — there is no side
    file. A fence is exactly the kind of edit that breaks that quietly: had the marker been
    moved or a second one introduced, `_STATE_RE` would find the wrong list (or nothing) and
    every dangling cite in the second run would read as newly dangling, which looks like a
    healthy report. So this pins the fence and one marker of the right shape, and that the
    second run names one new cite rather than four.
    """
    vault = world["vault"]
    report = vault / "autonomy" / "referential-integrity-latest.md"
    ri.main(_args(world) + ["--report", str(report)])
    first = report.read_text()
    assert first.startswith("---\n"), "run one must already carry the fence"
    assert first.count("<!-- ri:dangling ") == 1, "exactly one state marker per report"

    (vault / "knowledge" / "software" / "present.md").unlink()
    ri.main(_args(world) + ["--report", str(report)])
    second = report.read_text()
    assert second.startswith("---\n"), "run two must carry it too"
    assert second.count("<!-- ri:dangling ") == 1, (
        "a duplicated marker makes the state read ambiguous")
    assert "no previous report" not in second, (
        "the fence hid the previous run's marker, so nothing carried over")
    assert "- newly dangling since the previous report: 1" in second, (
        "every dangling cite reading as new is the symptom of an unread marker")


def test_the_nightly_dated_copy_is_fenced_by_the_same_bytes_and_still_graded(world):
    """#2402: the `cp` step's copy is graded by the gate, and it never becomes the delta base.

    The ledger skill copies `-latest.md` to `referential-integrity/<UTC date>.md` AFTER
    the generator exits, and derives its run count from `ls -1` over that directory. Three
    things have to hold for that route to keep working and none of them was pinned on the
    writer's side: the copy is conformant (it is the same bytes as a report the gate
    already accepts, so a fence regression surfaces there too); the copy is WALKED — the
    OKF scan descends into that subdirectory, which is the opposite of the Mission Control
    tab, which deliberately never sees it (`tests/test_mc_summarize_autonomy_gate.py`
    pins the tab half against these very bytes); and the copy never becomes the delta
    base, because `previous_dangling()` reads its state marker from the `--report` path
    and nothing else.

    That last clause is what the three unfenced copies were the visible symptom of: the
    generator was fixed at 5ae75298, but the back-history files written before it stood
    red from 2026-10-04 to 2026-10-08 with no prune path to age them out, so the only
    drift canary for `autonomy/` carried a standing 3. The control at the bottom is the
    same shape #2402 found — one fenced-away copy in that subdirectory — and it must be
    reported while `-latest.md` stays clean.
    """
    vault = world["vault"]
    report = vault / "autonomy" / "referential-integrity-latest.md"
    ri.main(_args(world) + ["--report", str(report)])
    latest = report.read_text()
    assert latest.startswith("---\n"), "the bytes the `cp` copies must carry the fence"

    dated = vault / "autonomy" / "referential-integrity"
    dated.mkdir()
    copy = dated / "2026-10-08.md"
    copy.write_text(latest, encoding="utf-8")        # task #94's `cp`, byte for byte
    assert copy.read_bytes() == report.read_bytes(), "the copy diverged from its base"

    checked = _run_gate(vault, "autonomy")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "referential-integrity" not in checked.stdout, checked.stdout

    # The delta base has to survive the copy sitting beside it: a second run, one cite
    # newly dead, must still report exactly one new dangling cite out of the SAME
    # `--report` path. `n/a (no previous report)` here would mean the marker was read
    # from somewhere else — and the ledger skill's own rule is that a fresh base is the
    # bug that marks every dangling cite new, every night, forever.
    (vault / "knowledge" / "software" / "present.md").unlink()
    ri.main(_args(world) + ["--report", str(report)])
    second = report.read_text()
    assert "- newly dangling since the previous report: 1" in second, (
        "with a dated copy in the subdirectory the run fell back to `n/a (no previous "
        "report)`: the state marker was read from something other than --report")
    assert copy.read_text() == latest, (
        "a generator run wrote into the dated-copy subdirectory, which is the ledger "
        "skill's run count and no longer a pure copy")

    # Control: the same copy with its fence cut off is the state the three pre-fix files
    # sat in, and the walk has to name it — while `-latest.md`, the delta base, is not
    # the file reported.
    copy.write_text(latest[latest.index("\n---\n") + len("\n---\n"):], encoding="utf-8")
    assert not okf.STRICT_FM_RE.match(copy.read_text()), (
        "the control must fail the fence check itself")
    refused = _run_gate(vault, "autonomy")
    assert refused.returncode == 1, refused.stdout + refused.stderr
    assert ("autonomy/referential-integrity/2026-10-08.md: no parseable frontmatter "
            "block") in refused.stdout, refused.stdout
    assert "referential-integrity-latest.md" not in refused.stdout, refused.stdout


# ────────────────────────── the retention-swept arm (#3637) ───────────────────
# A memory line that cites `autonomy-runs/<id>/run_*.md` cites a file the
# groundskeeper deletes at `RUN_RECORD_MAX_AGE_DAYS`. Reporting that as plain
# `dangling` tells the reader to re-point a file that cannot come back: knowledge
# health report #60 printed `NEWLY DANGLING: 1` for
# `memory/mental-models.md` -> `autonomy-runs/74/run_74_20260929_230732.md` on
# 2026-10-31, in the same list as cites a human could actually repair.


def _load_sweeper():
    """`retention-sweep.py` has a hyphen and cannot be imported by name, so it loads
    by path — the way `scripts/maintenance/guard_vacuity.py` loads its own mutation
    targets. This is how the tests read the horizon instead of trusting a copy of it.
    """
    spec = importlib.util.spec_from_file_location(
        "_ri_sweeper_under_test",
        REPO_ROOT / "scripts" / "groundskeeper" / "retention-sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SWEEPER = _load_sweeper()


def _run_record(days_ago: float, task: int = 74) -> str:
    """A run-record file name `days_ago` old, in the `run_<task>_<UTC stamp>` form
    the engine writes — the name is the clock the sweeper ages on.
    """
    stamp = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y%m%d_%H%M%S")
    return f"run_{task}_{stamp}.md"


SWEPT_OLD = _run_record(45)        # past RUN_RECORD_MAX_AGE_DAYS = 30
SWEPT_OLDER = _run_record(120)     # further past it
WITHIN_WINDOW = _run_record(0.25)  # six hours ago: inside the horizon
PARENT_GONE = "run_99_20260104_230732.md"  # task 99's whole directory is past

SWEPT_MEMORY = "\n".join([
    "# Memory",
    "",
    f"- task 74 timed out inside the scheduler: `autonomy-runs/74/{SWEPT_OLD}`.",
    f"- a second task 74 timeout behind the same window: `autonomy-runs/74/{SWEPT_OLDER}`.",
    f"- task 74 failed six hours ago: `autonomy-runs/74/{WITHIN_WINDOW}`.",
    f"- task 99 stopped dispatching in January: `autonomy-runs/99/{PARENT_GONE}`.",
    "- the resolver used to be `scripts/maintenance/gone_module.py`.",
    "",
])


@pytest.fixture
def swept_world(tmp_path):
    """Task 74's directory survives with its newer records, so the two records older
    than the horizon were swept and the six-hour one is missing for some other
    reason; task 99's directory is past entirely, and `gone_module.py` is the moved
    module the swept arm must never be able to hide.
    """
    vault, repo, home = tmp_path / "vault", tmp_path / "repo", tmp_path / "home"
    files = {
        "lloyd/MEMORY.md": SWEPT_MEMORY,
        "lloyd/USER.md": "# User\n\n- nothing cited here.\n",
        "memory/mental-models.md": "\n- nothing cited here either.\n",
    }
    for rel, text in files.items():
        (vault / rel).parent.mkdir(parents=True, exist_ok=True)
        (vault / rel).write_text(text, encoding="utf-8")
    survivor = home / "lloyd-data" / "autonomy-runs" / "74"
    survivor.mkdir(parents=True)
    (survivor / _run_record(2)).write_text("---\ntask_id: 74\n---\n\nkept\n", encoding="utf-8")
    return {"vault": vault, "repo": repo, "home": home}


def _swept_by_target(world):
    return {r["target"]: r for r in ri.scan(**world)}


def test_a_swept_run_record_is_reported_as_swept_not_as_a_moved_file(swept_world):
    """The record is gone because the retention window closed over it, which is a
    different fact from "a file moved and someone should re-point the cite".

    The row has to carry the reason, not just the label: `how_checked` names the
    directory that survived, because that surviving directory is the whole evidence
    that a sweep rather than a loss happened, and a reader who cannot re-check it has
    been handed a new claim with no way to test it.
    """
    got = _swept_by_target(swept_world)
    survivor = swept_world["home"] / "lloyd-data" / "autonomy-runs" / "74"
    for name in (f"autonomy-runs/74/{SWEPT_OLD}", f"autonomy-runs/74/{SWEPT_OLDER}"):
        assert got[name]["verdict"] == "swept", got[name]
        assert "retention" in got[name]["how_checked"].lower(), got[name]
        assert str(survivor) in got[name]["how_checked"], got[name]


def test_the_swept_verdict_names_the_horizon_it_took_from_the_sweeper(swept_world):
    """`how_checked` has to carry the number and where it came from: a verdict that
    says only "swept" is a claim about a rule nobody can re-check, and a number the
    resolver restated would drift the day the sweeper's constant changes.
    """
    how = _swept_by_target(swept_world)[f"autonomy-runs/74/{SWEPT_OLD}"]["how_checked"]
    assert f"RUN_RECORD_MAX_AGE_DAYS = {SWEEPER.RUN_RECORD_MAX_AGE_DAYS}" in how, how
    assert "retention-sweep.py" in how, how


def test_a_run_record_inside_the_window_is_still_dangling(swept_world):
    """The precision guard. The sweep deletes on age, so a record the window still
    covers went missing for a reason a reader can act on — that stays dangling.
    """
    got = _swept_by_target(swept_world)
    assert got[f"autonomy-runs/74/{WITHIN_WINDOW}"]["verdict"] == "dangling", \
        got[f"autonomy-runs/74/{WITHIN_WINDOW}"]


def test_a_run_record_whose_parent_directory_is_past_is_still_dangling(swept_world):
    """The other precision guard. `autonomy-runs/99/` is itself gone, so there is no
    sweep here to blame — the whole subtree vanished, which is a real loss to chase.
    """
    got = _swept_by_target(swept_world)
    assert got[f"autonomy-runs/99/{PARENT_GONE}"]["verdict"] == "dangling", \
        got[f"autonomy-runs/99/{PARENT_GONE}"]


def test_the_swept_horizon_is_read_live_from_the_sweeper_not_copied(tmp_path, swept_world,
                                                                    monkeypatch):
    """Raise the sweeper's horizon to 90 days and the same two cites split.

    Both numbers must move with the source file: a resolver holding its own copy of
    `30` would keep calling the 45-day record swept (horizon 90 covers it), and a
    classifier that ignored the horizon entirely would call both swept. The fake
    source is a file with the one constant the loader is allowed to read.
    """
    fake = tmp_path / "retention-sweep.py"
    fake.write_text("RUN_RECORD_MAX_AGE_DAYS = 90\n", encoding="utf-8")
    monkeypatch.setattr(ri, "SWEEP_SOURCE", fake)
    got = _swept_by_target(swept_world)
    assert got[f"autonomy-runs/74/{SWEPT_OLD}"]["verdict"] == "dangling", \
        got[f"autonomy-runs/74/{SWEPT_OLD}"]
    assert got[f"autonomy-runs/74/{SWEPT_OLDER}"]["verdict"] == "swept", \
        got[f"autonomy-runs/74/{SWEPT_OLDER}"]
    assert "RUN_RECORD_MAX_AGE_DAYS = 90" in \
        got[f"autonomy-runs/74/{SWEPT_OLDER}"]["how_checked"]
    # And with the real sweeper restored, the resolver's number is its number.
    monkeypatch.undo()
    assert ri._run_record_max_age_days(ri.SWEEP_SOURCE) == SWEEPER.RUN_RECORD_MAX_AGE_DAYS


def test_classing_swept_records_out_still_fails_the_run_on_a_moved_module(swept_world,
                                                                          capsys):
    """The purpose of the check this change narrows (#882's whole point).

    `swept` rows leave the exit code, so the exit code has to keep catching what it
    was built to catch: a cite whose target moved and can still be repaired. Of this
    corpus's 5 cites exactly the two the sweep's own rule explains leave `dangling`,
    and the other three — the in-window record, the one whose directory is gone too,
    and the moved module — are still there, so the run still exits 1.
    """
    report = swept_world["vault"] / "report.md"
    rc = ri.main([*_args(swept_world), "--report", str(report)])
    assert rc == 1, rc
    out = capsys.readouterr().out
    assert "5 cites, 3 dangling, 0 exempt, 2 swept run records" in out, out
    dangling = [r["target"] for r in ri.scan(**swept_world)
                if r["verdict"] == "dangling"]
    assert dangling == [f"autonomy-runs/74/{WITHIN_WINDOW}",
                        f"autonomy-runs/99/{PARENT_GONE}",
                        "scripts/maintenance/gone_module.py"], dangling


# ─────────────── clause 1: the swept cite left the memory rule that made it ──────
# The other half of #3637 is a vault edit — `memory/mental-models.md` re-pointed at the
# ledger that survives — and this file is the only place in the diff a node for it can
# live. It reads the live vault, unmarked like the other vault-claim nodes in this tree
# (`test_the_automod_skill_states_the_fork_rule_where_a_round_reads_it` in
# tests/test_automod_doc_claims.py): a `live_vault`-marked node is deselected by the
# gate's own `-m "not live_vault"` and certifies nothing, and `app.paths.VAULT_ROOT` is
# `Path.home() / "obsidian"`, so every worktree reads the same live vault and a missing
# file here is a regression in the subject, not an artefact of where the test ran.

#: The five fields the rule quotes out of task 74's 09-29 23:07 run, verbatim, as they
#: appear on both surfaces — the rule's own line and the ledger it now cites.
SWEPT_RUN_FIELDS = ("started_at 23:07:32.045Z", "duration_seconds: 0.5",
                    "exception: FileNotFoundError", "failure_kind: task",
                    "trigger: scheduler")


def test_the_memory_rule_no_longer_cites_the_swept_run_record():
    """Clause 1 of #3637: the cite went dead, so the rule cites the ledger instead.

    `memory/mental-models.md` built its second instance on task 74's 09-29 23:07 run
    and named the record file by name. The groundskeeper's retention window then
    deleted that file, which left a rule whose evidence could no longer be opened and
    a cite no one could re-point — knowledge health report #60 printed it on 2026-10-31
    as `NEWLY DANGLING: 1`. The re-point keeps the quoted fields and names
    `projects/lloyd/fleet-timeout-ledger.md`, which carries the same list, so the rule
    still shows its work; the dead file name has to be gone from every line.
    """
    from app.paths import VAULT_ROOT

    memory = VAULT_ROOT / "memory" / "mental-models.md"
    assert memory.is_file(), f"clause 1's subject is absent: {memory}"
    lines = memory.read_text(encoding="utf-8").splitlines()
    assert [ln for ln in lines if "run_74_20260929_230732" in ln] == [], \
        "the swept run record is cited again"

    # Both instances of the rule are bullets with the same bolded title (the first is
    # task 53's, still citing its own swept record — the classifier's job, not this
    # clause's), so the one under test is the one that names the ledger it was
    # re-pointed to. Anchoring on the re-point rather than on a phrase inside the
    # sentence survives a reflow of the sentence and does not survive losing the
    # re-point.
    rule = [ln for ln in lines
            if ln.startswith("- **Classify a failed run by the deepest layer")
            and "projects/lloyd/fleet-timeout-ledger.md" in ln]
    assert len(rule) == 1, f"expected the one rule line that names the ledger, got {len(rule)}"
    for field in SWEPT_RUN_FIELDS:
        assert field in rule[0], f"the rule lost the quoted field {field!r}"

    ledger = VAULT_ROOT / "projects" / "lloyd" / "fleet-timeout-ledger.md"
    assert ledger.is_file(), f"the re-point target is absent: {ledger}"
    ledger_text = ledger.read_text(encoding="utf-8")
    assert "run_74_20260929_230732.md" in ledger_text, \
        "the ledger no longer holds the record the rule's evidence came from"
    for field in SWEPT_RUN_FIELDS:
        assert field in ledger_text, f"the ledger does not carry {field!r}"
