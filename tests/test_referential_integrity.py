"""#882: dead file cites in the always-loaded memory files are reported, never fixed."""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
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
