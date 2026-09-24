"""#882: dead file cites in the always-loaded memory files are reported, never fixed."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.maintenance import referential_integrity as ri  # noqa: E402

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
