"""Backlog #1167 (members #977, #1153) — `segment:` complete and self-maintaining.

`segment:` is the vault's convention key (`knowledge`, `backlog`, ...). Only the
backlog create paths declared it, so every other route that wrote vault markdown
left it absent for good: 128 pre-#518 backlog files, and knowledge notes from the
deep-research and documentation-digester templates, which recurred on 2026-09-23
after a hand backfill. The one reader is `relations_index.py`, which files a
keyless note under `segment: "unknown"`.

What this file pins, by acceptance clause (clause 2, the backlog update path, is
`tests/test_backlog_okf_frontmatter.py`):

* 3 — a deep-research note carries `segment: knowledge`: the worker restores it
  on disk (`deep_research.ensure_segment`, offline), and the skill's template
  declares it (live vault);
* 4 — the doc-digester stack-update template declares `segment: knowledge` and a
  non-empty `tags:` (live vault);
* 5 — `scripts/vault/segment_scan.py` prints per-directory counts over the
  extractor allow-list plus `backlog/`, exits 0 on a conformant tree and 1 when
  one file in an otherwise conformant directory omits the key.

The template checks read the live vault's skills, so they carry `live_vault`: the
gate deselects them, and they go red if a nightly skills pass drops the key again.
"""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.vault import segment_scan  # noqa: E402
from scripts.vault.validate_okf import STRICT_FM_RE  # noqa: E402
from workers.sources import deep_research as DR  # noqa: E402

SKILLS = Path.home() / "obsidian" / "skills"
DEEP_DIVE = SKILLS / "deep-dive-research" / "SKILL.md"
DIGESTER = SKILLS / "documentation-digester" / "SKILL.md"


def _fm(text: str) -> dict:
    m = STRICT_FM_RE.match(text)
    assert m, "no front matter block at offset 0"
    fm = yaml.safe_load(m.group(1))
    assert isinstance(fm, dict)
    return fm


# ── clause 3, code side: the worker restores the key on the note it verified ──

NOTE_WITHOUT_SEGMENT = (
    "---\ntype: research\ntags: [research, agents]\nsource: deep-research\n"
    "topic_id: 4\nresearched_at: 2026-09-23T20:00:00Z\n---\n\n# A topic\n\n"
    "## Summary\n" + "Body text. " * 60 + "\n"
)


def test_ensure_segment_inserts_one_line_after_type(tmp_path):
    note = tmp_path / "2026-09-23-a-topic.md"
    note.write_text(NOTE_WITHOUT_SEGMENT, encoding="utf-8")

    assert DR.ensure_segment(note) is True

    after = note.read_text(encoding="utf-8")
    assert _fm(after)["segment"] == "knowledge"
    # A pure insertion: the old text is the new text minus exactly one line.
    old_lines = NOTE_WITHOUT_SEGMENT.splitlines()
    new_lines = after.splitlines()
    assert len(new_lines) == len(old_lines) + 1
    assert new_lines[:2] + new_lines[3:] == old_lines
    assert new_lines[2] == "segment: knowledge"
    # Idempotent, and a chosen value is never overwritten.
    assert DR.ensure_segment(note) is False
    assert note.read_text(encoding="utf-8") == after


def test_ensure_segment_finds_the_closing_fence_past_a_long_block(tmp_path):
    """A bounded read window calls a 200-line block keyless; the whole file is read."""
    long_block = "".join(f"k{i}: v{i}\n" for i in range(250))
    note = tmp_path / "long.md"
    note.write_text(f"---\n{long_block}segment: projects\n---\n\n# Long\n", encoding="utf-8")
    assert DR.ensure_segment(note) is False
    assert _fm(note.read_text())["segment"] == "projects"


def test_ensure_segment_leaves_a_note_without_front_matter(tmp_path):
    note = tmp_path / "bare.md"
    note.write_text("# No block\n\nbody\n", encoding="utf-8")
    assert DR.ensure_segment(note) is False
    assert note.read_text() == "# No block\n\nbody\n"


def test_execute_adds_segment_to_the_note_the_turn_wrote(tmp_path, monkeypatch):
    """The seam: a `written` turn whose note came out keyless is recorded with it."""
    path = tmp_path / "knowledge" / "research" / "2026-09-23-a-topic.md"
    path.parent.mkdir(parents=True)

    class _Store:
        finished: dict = {}

        def claim(self, topic_id, by, queue_id):
            return {"topic": "a topic", "attempts": 1, "domain": "agents"}

        def finish(self, topic_id, result, **kw):
            _Store.finished = {"result": result, **kw}
            return {}

    async def fake_run(prompt, **kw):
        path.write_text(NOTE_WITHOUT_SEGMENT, encoding="utf-8")
        return {"session_id": "s1", "text": f"RESULT: written\nNOTE: {path}\n"
                "DUPLICATE_OF:\nFACTS: 3\nSOURCES: 4\n", "structured": None}

    import autonomy
    monkeypatch.setattr(DR, "_store", lambda: _Store())
    monkeypatch.setattr(DR, "run_prompt_in_session", fake_run)
    monkeypatch.setattr(DR, "_vault_dirty_paths", lambda: set())
    monkeypatch.setattr(autonomy, "_load_skill_content", lambda name: "skill text")

    class _Item:
        id = 1
        payload = {"topic_id": 4, "artifact_path": str(path), "structured_verdict": False}

    out = asyncio.run(DR.execute(_Item()))

    assert out["status"] == "success", out
    assert out["meta"]["result"] == "written"
    assert out["meta"].get("segment_added") is True
    assert _fm(path.read_text())["segment"] == "knowledge"


# ── clauses 3 and 4, vault side: the templates declare it at write time ───────

def _template_frontmatter(skill: Path, type_value: str) -> dict:
    """The front matter a note written from the skill's template would carry.

    Finds the fenced markdown block whose front matter declares `type_value`,
    fills every `{placeholder}` with a sample value the way the model does, and
    parses the result as the OKF gate would.
    """
    if not skill.is_file():
        pytest.fail(f"the live vault is not at {skill}")
    text = skill.read_text(encoding="utf-8")
    for block in re.findall(r"```markdown\n(.*?)```", text, re.DOTALL):
        if re.search(rf"^type:\s*{re.escape(type_value)}\s*$", block, re.M):
            filled = re.sub(r"\{[^{}\n]*\}", "sample", block)
            return _fm(filled)
    pytest.fail(f"{skill}: no markdown template with `type: {type_value}`")


@pytest.mark.live_vault
def test_deep_research_template_declares_segment_knowledge():
    fm = _template_frontmatter(DEEP_DIVE, "research")
    assert fm.get("segment") == "knowledge", fm


@pytest.mark.live_vault
def test_doc_digester_template_declares_segment_and_tags():
    fm = _template_frontmatter(DIGESTER, "stack-update")
    assert fm.get("segment") == "knowledge", fm
    assert fm.get("tags"), fm


# ── clause 5: the committed scan ─────────────────────────────────────────────

def _conformant_vault(root: Path) -> None:
    for d in ("knowledge/research", "projects", "people", "personal", "work",
              "memory", "backlog"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "knowledge/research/a.md").write_text(
        "---\ntype: research\nsegment: knowledge\n---\n\n# A\n")
    (root / "projects/p.md").write_text("---\ntype: project\nsegment: projects\n---\n\n# P\n")
    (root / "memory/2026-09-23.md").write_text(
        "---\ntype: daily\nsegment: memory\n---\n\n# Day\n")
    long_block = "".join(f"k{i}: v{i}\n" for i in range(250))
    (root / "backlog/1-long.md").write_text(
        f"---\ntype: backlog\n{long_block}segment: backlog\n---\n\n# Long\n")
    # Reserved and excluded files are not concept notes and never count.
    (root / "knowledge/index.md").write_text("# index, no front matter\n")
    (root / "memory/vault-maintenance").mkdir()
    (root / "memory/vault-maintenance/run.md").write_text("# a run log\n")


def test_scan_covers_the_extractor_allow_list_plus_backlog():
    dirs, _ = segment_scan.scan_dirs()
    assert dirs[-1] == "backlog"
    for d in ("knowledge", "projects", "people", "personal", "work", "memory"):
        assert d in dirs


def test_scan_exits_0_on_a_conformant_tree(tmp_path, capsys):
    _conformant_vault(tmp_path)
    assert segment_scan.main(["--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    for d in ("knowledge/", "projects/", "memory/", "backlog/"):
        assert re.search(rf"^\s+{re.escape(d)}\s+missing segment: 0$", out, re.M), out
    assert "total missing: 0" in out


def test_scan_exits_1_on_one_keyless_file_and_names_its_directory(tmp_path, capsys):
    _conformant_vault(tmp_path)
    (tmp_path / "knowledge/research/b.md").write_text(
        "---\ntype: research\ntags: [research]\n---\n\n# B\n")

    assert segment_scan.main(["--root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert re.search(r"^\s+knowledge/\s+missing segment: 1$", out, re.M), out
    assert re.search(r"^\s+backlog/\s+missing segment: 0$", out, re.M), out
    assert "knowledge/research/b.md" in out


@pytest.mark.live_vault
def test_scan_exits_0_on_the_live_vault():
    if not (Path.home() / "obsidian").is_dir():
        pytest.fail("the live vault is not at ~/obsidian")
    assert segment_scan.main(["--root", str(Path.home() / "obsidian")]) == 0
