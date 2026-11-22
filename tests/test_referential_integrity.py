"""#882: dead file cites in the always-loaded memory files are reported, never fixed."""
from __future__ import annotations

import hashlib
import importlib.util
import re
import shutil
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


# ── the two absence spellings that sat outside the vocabulary (#3944) ───────
# The exempt rows in `MEMORY` above are the postpositive form ("`x` was deleted").
# The two forms a person reaches for when recording that a path is gone were not in
# `_ABSENCE_PROSE`, so `memory/mental-models.md:2407` ("… run a watchdog under a
# deleted `lloyd-work/SM_20261104_123801/` round root …") and :2199 ("`ls
# ~/.local/state/lloyd-fleet-watchdog/` → `No such file or directory`") each printed
# in `## Dangling`, in a report the module docstring defines as a red run. Each case
# below is the shape of one of those two lines, plus the control that keeps the
# guard worth having.
_GONE_ROOT = "lloyd-work/SM_20261104_123801/"
_GONE_STATE = "~/.local/state/lloyd-fleet-watchdog/alert.json"

_ABSENCE_CASES = {
    # Clause 1: `deleted` used attributively, sitting before the path it describes.
    "attributive-deleted": (
        f"- a watchdog ran under a deleted `{_GONE_ROOT}` round root.\n",
        "exempt", _GONE_ROOT),
    # Clause 2: the literal error string the shell printed for the path.
    "shell-error-string": (
        f"- the alarm named a state file at `{_GONE_STATE}`: "
        "`ls ~/.local/state/lloyd-fleet-watchdog/` → `No such file or directory`.\n",
        "exempt", _GONE_STATE),
    # Clause 3's control: the same absent path, in a sentence that reports no
    # absence at all. A cite whose target is not where the line says it is — the
    # event the report exists for — must stay red whatever the vocabulary grows.
    "no-absence-prose": (
        f"- the watchdog runs from `{_GONE_ROOT}` and writes its log under it.\n",
        "dangling", _GONE_ROOT),
}


@pytest.mark.parametrize("spelling", sorted(_ABSENCE_CASES))
def test_recording_that_a_path_is_gone_exempts_its_cite(tmp_path, spelling):
    """Clauses 1-3: exempt on either absence spelling, red on neither.

    `how_checked` is asserted too: the row has to name the check that moved it out
    of `## Dangling`, because a verdict nobody can re-run is #882's whole complaint.
    """
    line, want, target = _ABSENCE_CASES[spelling]
    vault = tmp_path / "vault"
    (vault / "memory").mkdir(parents=True)
    (vault / "memory" / "mental-models.md").write_text(line, encoding="utf-8")
    got = [r for r in ri.scan(vault=vault, repo=tmp_path / "repo",
                              home=tmp_path / "home",
                              corpus=("memory/mental-models.md",))
           if r["target"] == target]
    assert len(got) == 1, (spelling, got)
    assert got[0]["verdict"] == want, (spelling, got[0])
    if want == "exempt":
        assert "absence prose" in got[0]["how_checked"], got[0]
    else:
        assert "absence prose" not in got[0]["how_checked"], got[0]


def test_a_cite_that_moved_stays_dangling_under_the_widened_vocabulary(tmp_path):
    """Clause 3, the code half: a `path:N` whose file moved out from under it.

    The shape #3637 clause 4 refuses to let an exemption swallow — past EOF, in a
    line with no absence word in it — is the case the widened vocabulary must not
    reach. `test_documented_absence_is_exempt_and_does_not_fail_the_run` holds the
    postpositive spelling and the clean-exit half; this node holds the red half.
    """
    vault, repo = tmp_path / "vault", tmp_path / "repo"
    (vault / "memory").mkdir(parents=True)
    (repo / "app").mkdir(parents=True)
    (repo / "app" / "queue.py").write_text("a\nb\nc\n", encoding="utf-8")
    (vault / "memory" / "mental-models.md").write_text(
        "- the claim is published by `app/queue.py:999`.\n", encoding="utf-8")
    got = ri.scan(vault=vault, repo=repo, home=tmp_path / "home",
                  corpus=("memory/mental-models.md",))
    assert [r["verdict"] for r in got] == ["dangling"], got
    assert "beyond eof (3)" in got[0]["how_checked"], got[0]


def test_the_widened_vocabulary_is_the_one_the_gate_matches_with(tmp_path,
                                                                 monkeypatch):
    """Clause 4's seam: the two consumers of this rule are one object, not two.

    `scripts/automod/prose_consumers.py:439` binds `exempt_lines =
    RI.ABSENCE_PROSE` rather than compiling its own (#3309), so widening it also
    widens what the gate's observe-only rung declines to flag as a consumer. The
    assertion is on the pattern object the matcher RECEIVES: identity proves one
    copy, and searching it with the two new spellings proves the widening is the
    half the gate got, not a vocabulary only the nightly sweep ever sees.
    """
    from scripts.automod import prose_consumers as PC

    base, head = tmp_path / "base", tmp_path / "head"
    for tree, body in ((base, "KEEP = 1\ndef retired_worker_entrypoint():\n"
                               "    return KEEP\n"), (head, "KEEP = 1\n")):
        (tree / "app").mkdir(parents=True)
        (tree / "app" / "m.py").write_text(body, encoding="utf-8")
    vault = tmp_path / "vault"
    vault.mkdir()
    diff = ("\n".join(["--- a/app/m.py", "+++ b/app/m.py", "@@ -1,2 +0,0 @@",
                       "-def retired_worker_entrypoint():", "-    return KEEP"]) + "\n")
    seen: dict = {}
    real = PC._match

    def spy(vault_: Path, candidates: list, **kw):
        seen.update(kw)
        return real(vault_, candidates, **kw)

    monkeypatch.setattr(PC, "_match", spy)
    PC.scan(head, base, ["app/m.py"], diff, checkout=head, vault=vault,
            home=tmp_path / "home")
    exempt_lines = seen.get("exempt_lines")
    assert exempt_lines is ri.ABSENCE_PROSE, (
        "the gate received a pattern of its own: the two copies of this rule are "
        "what #3309 exists to refuse")
    assert seen.get("exempt_marker") == ri.EXEMPT_MARKER
    assert exempt_lines.search("a watchdog ran under a deleted "
                               "`lloyd-work/SM_20261104_123801/` round root")
    assert exempt_lines.search("`ls ~/nowhere/` → `No such file or directory`")


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


# ─────────────── the existence-gated state-path arm (#4462) ───────────────
# `memory/mental-models.md:2540` cites `~/.local/state/lloyd-guardian/data-tripped.json`
# as the MECHANISM that arms the guardian's data-trip hold. That file exists only while
# the hold is armed — `pause_journal` ids 71 and 73, both 2026-11-19, lasted 33 m 36 s
# and 64 s — so a sweep that runs while the tripwire is down books the row `dangling`,
# and no line-level arm of the exempt cascade can ever fire for it, because the citing
# sentence is not saying the file is gone. #94's ledger therefore re-reported a state
# transition as a broken cite, and knowledge health report #60 printed
# `NEWLY DANGLING: 1` for exactly this row on 2026-11-21.

#: The tripwire cite as `memory/mental-models.md:2540` spells it.
TRIPWIRE = "~/.local/state/lloyd-guardian/data-tripped.json"
#: A daemon the resolver has never heard of. The class is keyed on the state root the
#: scan was given, so an unlisted daemon name has to behave exactly like the guardian's.
UNLISTED = "~/.local/state/lloyd-widget-watchdog/last-scan.json"
#: One level too deep: a helper inside the daemon's own subdirectory, not an immediate
#: child of the daemon directory.
DEEP = "~/.local/state/lloyd-guardian/bin/helper.py"
#: An ordinary source path outside the state root — the moved module the sweep exists to
#: catch, which is why this class has to stay exactly one level wide.
MOVED = "scripts/maintenance/gone_module.py"


def _state_memory(abs_cite: str) -> str:
    """The corpus, five cites, no line containing a token `_ABSENCE_PROSE` matches.

    That last part is the fixture's whole point: the row under test is exempt because of
    the class rule, and a sentence that let the prose arm fire would turn the node green
    for the wrong reason. `test_the_citing_line_still_defies_the_absence_prose_arm`
    re-checks it, so widening that vocabulary cannot silently vacuate these nodes.
    """
    return "\n".join([
        "# Memory",
        "",
        f"- the data-trip hold is armed by a marker the handler writes and clears: "
        f"`{TRIPWIRE}`.",
        f"- another daemon records its last pass the same way: `{UNLISTED}`.",
        f"- the same tree spelled as an absolute path: `{abs_cite}`.",
        f"- a helper the daemon's own installer drops under bin: `{DEEP}`.",
        f"- the resolver used to be `{MOVED}`.",
        "",
    ])


@pytest.fixture
def state_world(tmp_path):
    """Every daemon IDLE: each state directory present with its standing files, each
    cited transient marker absent — the state `ls -la ~/.local/state/lloyd-guardian/`
    shows when the data-trip hold is not armed, which is the condition the acceptance
    check is measured under.
    """
    vault, repo, home = tmp_path / "vault", tmp_path / "repo", tmp_path / "home"
    state = home / ".local" / "state"
    for daemon in ("lloyd-guardian", "lloyd-widget-watchdog", "lloyd-sec-eval"):
        (state / daemon).mkdir(parents=True)
        (state / daemon / "heartbeat.json").write_text("{}\n", encoding="utf-8")
    (state / "lloyd-guardian" / "bin").mkdir(parents=True)
    abs_cite = str(state / "lloyd-sec-eval" / "scan-state.json")
    files = {
        "lloyd/MEMORY.md": _state_memory(abs_cite),
        "lloyd/USER.md": "# User\n\n- nothing cited here.\n",
        "memory/mental-models.md": "\n- nothing cited here either.\n",
    }
    for rel, text in files.items():
        (vault / rel).parent.mkdir(parents=True, exist_ok=True)
        (vault / rel).write_text(text, encoding="utf-8")
    # Keys starting with `_` are the node's handles, not `scan` parameters.
    return {"vault": vault, "repo": repo, "home": home,
            "_state": state, "_abs_cite": abs_cite}


def _state_scan(world, **kw):
    args = {k: v for k, v in world.items() if not k.startswith("_")}
    return ri.scan(**args, **kw)


def _state_by_target(world, **kw):
    return {r["target"]: r for r in _state_scan(world, **kw)}


def _state_args(world, *, state: Path | None = None) -> list[str]:
    """The CLI arguments for one run, so a node can cross the argparse seam with the
    same roots its `scan` call used. `--state` is how a caller hands the sweep a state
    root, exactly as `--data` hands it a data root.
    """
    out = ["--vault", str(world["vault"]), "--repo", str(world["repo"]),
           "--home", str(world["home"])]
    if state is not None:
        out += ["--state", str(state)]
    return out


def _report_sections(report: str) -> dict[str, str]:
    """Split a rendered report into its `## ` sections. A row's bytes are identical in
    the Dangling and the Exempt list, so "which list" is the only thing worth asserting
    and grepping the whole file cannot answer it.
    """
    out: dict[str, str] = {}
    name = None
    for line in report.splitlines():
        if line.startswith("## "):
            name = line.strip()
            out.setdefault(name, "")
        elif name is not None:
            out[name] += line + "\n"
    return out


def test_the_citing_line_still_defies_the_absence_prose_arm(state_world):
    """The premise of the whole item, pinned: no line-level rule can exempt this row.

    #4462 is filed because the cite is structurally unexemptable, and the fixture is
    meant to reproduce that. If the fixture's sentence ever starts matching
    `_ABSENCE_PROSE`, every node below passes for the wrong reason, so this is the
    vacuity guard for the section, not a fifth opinion about the verdict.
    """
    memory = (state_world["vault"] / "lloyd" / "MEMORY.md").read_text(encoding="utf-8")
    lines = [ln for ln in memory.splitlines() if f"`{TRIPWIRE}`" in ln]
    assert len(lines) == 1, lines
    assert ri._ABSENCE_PROSE.search(lines[0]) is None, \
        f"the fixture line now matches the prose arm, so the class rule is ungraded: {lines[0]}"
    assert ri.EXEMPT_MARKER not in lines[0]


def test_an_absent_daemon_state_file_is_exempt_as_an_existence_gated_state_path(state_world):
    """Clause 1 of #4462: the tripwire is down, the cite is a mechanism, and the row is
    exempt for a reason that names the class.

    The reason has to carry the surviving daemon directory, because that directory is
    the evidence: a daemon that is running and not holding one marker is a different
    fact from a file that moved, and a reader who cannot re-check which of the two the
    sweep saw has been handed a new claim with no way to test it.
    """
    got = _state_by_target(state_world)
    for target in (TRIPWIRE, UNLISTED, state_world["_abs_cite"]):
        assert got[target]["verdict"] == "exempt", got[target]
        assert "existence-gated state path" in got[target]["how_checked"], got[target]
        assert "absence prose" not in got[target]["how_checked"], got[target]
        daemon = (state_world["home"] / target[2:] if target.startswith("~/")
                  else Path(target)).parent
        assert str(daemon) in got[target]["how_checked"], got[target]


def test_the_exempt_state_path_row_renders_under_exempt_and_leaves_the_dangling_blob(state_world):
    """Clause 1's other half: the row moves lists, and leaves the machine block.

    The `ri:dangling` blob is what `previous_dangling` diffs night against night, so
    staying out of it is what stops a state transition spending the newly-dangling
    figure that knowledge health report #60 treats as reportable. The moved module is
    asserted INSIDE the same blob beside the assertion, because an empty blob would
    make "not in it" green while nothing worked.
    """
    report = ri.render_report(_state_scan(state_world), previous=None)
    sections = _report_sections(report)
    assert f"`{TRIPWIRE}`" in sections["## Exempt"], sections["## Exempt"]
    assert "existence-gated state path" in sections["## Exempt"], sections["## Exempt"]
    assert f"`{TRIPWIRE}`" not in sections["## Dangling"], sections["## Dangling"]

    written = state_world["vault"] / "report.md"
    written.write_text(report, encoding="utf-8")
    keys = ri.previous_dangling(written)
    assert keys is not None, "the rendered report lost the block the delta is diffed on"
    assert f"lloyd/MEMORY.md -> {TRIPWIRE}" not in keys, keys
    assert f"lloyd/MEMORY.md -> {MOVED}" in keys, keys


def test_the_same_cite_with_the_state_file_present_resolves_and_appears_in_neither_section(state_world):
    """Clause 2: arming the tripwire changes nothing a reader has to interpret.

    Not Dangling, not Exempt — the cite resolves, which is the only outcome that is the
    same story whether or not the daemon happened to be tripped at 13:00Z. The other
    daemon's row is asserted still in the Exempt list so that "absent from Exempt"
    cannot be satisfied by an empty section.
    """
    armed = state_world["home"] / TRIPWIRE[2:]
    armed.write_text('{"armed_by": "guardian-data-trip", "at": "2026-11-19T01:01:22Z"}\n',
                     encoding="utf-8")
    got = _state_by_target(state_world)
    assert got[TRIPWIRE]["verdict"] == "resolves", got[TRIPWIRE]
    assert got[TRIPWIRE]["how_checked"].startswith("stat:"), got[TRIPWIRE]

    sections = _report_sections(ri.render_report(_state_scan(state_world), previous=None))
    for name in ("## Dangling", "## Exempt"):
        assert f"`{TRIPWIRE}`" not in sections[name], (name, sections[name])
    assert f"`{UNLISTED}`" in sections["## Exempt"], sections["## Exempt"]


def test_a_state_path_one_level_deeper_and_an_ordinary_source_path_stay_dangling(state_world):
    """Clause 3, the teeth. Two rows the class must not reach, in a run where the class
    rule demonstrably IS firing: the same scan exempts three cites, so a node that
    found nothing exempted at all would not be measuring scope.
    """
    got = _state_by_target(state_world)
    exempt = sorted(t for t, r in got.items() if r["verdict"] == "exempt")
    assert exempt == sorted([TRIPWIRE, UNLISTED, state_world["_abs_cite"]]), exempt
    for target in (DEEP, MOVED):
        assert got[target]["verdict"] == "dangling", got[target]
        assert "existence-gated" not in got[target]["how_checked"], got[target]


def test_a_run_whose_only_dangling_cites_are_out_of_class_still_exits_1(state_world, capsys):
    """Clause 3's exit code. Three cites leave `dangling` for a reason a reader cannot
    act on; the two that stay are the moved module and a path inside a daemon's own
    subtree, and the run is red on exactly those. No exit-code behaviour changed here:
    `main` still returns 1 on any dangling cite (`:526` at filing).
    """
    rc = ri.main(_state_args(state_world))
    out = capsys.readouterr().out
    assert rc == 1, (rc, out)
    assert "5 cites, 2 dangling, 3 exempt, 0 swept run records" in out, out
    dangling = [r["target"] for r in _state_scan(state_world) if r["verdict"] == "dangling"]
    assert dangling == [DEEP, MOVED], dangling


def test_the_class_is_keyed_on_the_state_root_the_scan_was_given(tmp_path, state_world,
                                                                capsys):
    """The exemption follows the root, not the spelling and not a list of names.

    Two scans of the SAME corpus bytes. Given a home whose own state tree holds the
    daemon directory, the cite is exempt — so the rule cannot be a list of paths, and
    cannot be a pattern on `~/.local/state`, either. Given the first home with its state
    root pointed somewhere else, the identical cite is dangling again — so the verdict is
    computed against the root the caller named. The CLI route is asserted with the same
    two roots because `--state` is how the nightly job would move it.
    """
    default = _state_by_target(state_world)
    assert default[TRIPWIRE]["verdict"] == "exempt", default[TRIPWIRE]

    elsewhere = tmp_path / "other-state"
    moved = _state_by_target(state_world, state=elsewhere)
    assert moved[TRIPWIRE]["verdict"] == "dangling", moved[TRIPWIRE]
    assert moved[state_world["_abs_cite"]]["verdict"] == "dangling", moved

    # The positive direction: a second home, the same cite bytes, a root under it.
    home_b = tmp_path / "home-b"
    (home_b / ".local" / "state" / "lloyd-guardian").mkdir(parents=True)
    world_b = {"vault": state_world["vault"], "repo": state_world["repo"], "home": home_b}
    assert _state_by_target(world_b)[TRIPWIRE]["verdict"] == "exempt", \
        "the exemption did not follow a state root that moved with --home"

    # The same two roots across the argparse seam, where the nightly job reaches them.
    assert ri.main(_state_args(state_world)) == 1
    assert "5 cites, 2 dangling, 3 exempt, 0 swept run records" in \
        capsys.readouterr().out
    assert ri.main(_state_args(state_world, state=elsewhere)) == 1
    assert "5 cites, 5 dangling, 0 exempt, 0 swept run records" in \
        capsys.readouterr().out


def test_a_daemon_whole_state_directory_is_gone_stays_dangling(state_world):
    """The precision guard that keeps the class honest, and the reason the arm reads
    like the swept one: `<state>/lloyd-widget-watchdog/` is itself gone, so there is no
    daemon here merely not holding a marker — the state tree for that daemon vanished,
    which is a loss a reader can and should chase.
    """
    shutil.rmtree(state_world["_state"] / "lloyd-widget-watchdog")
    got = _state_by_target(state_world)
    assert got[UNLISTED]["verdict"] == "dangling", got[UNLISTED]
    assert "existence-gated" not in got[UNLISTED]["how_checked"], got[UNLISTED]
    # The guardian's own marker is unaffected: one daemon's wiped root does not
    # excuse, or condemn, another daemon's flapping file.
    assert got[TRIPWIRE]["verdict"] == "exempt", got[TRIPWIRE]
