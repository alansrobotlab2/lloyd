"""Review 2026-09-24, P4: bounded, typed long-term memory — the build half.

MEMORY.md is meant to become a typed index with topic files behind it. What
ships now is additive and leaves today's prompt byte-identical: typed entries on
`memory_add`, `topics/<slug>` files through the memory tools, a render-time
overflow branch that defaults to `render_all`, the consolidation hint on a size
refusal, the validator, the dry-run consolidator and the eval runner. Lowering
`prompt_surface.MEMORY_MD_CEILING_BYTES` to `MEMORY_MD_INDEX_CEILING_BYTES` is the
deploy step, gated on `eval/run_memory_index_ab.py`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

import pytest

import prompt_builder
import prompt_surface as ps
from app import memory_ceiling as ceiling
from app import uptake

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "scripts" / "maintenance" / "vault-memory-index-skills.patch"
VALIDATOR = ROOT / "scripts" / "memory" / "validate_memory_index.py"

MEMORY_FIXTURE = """---
type: note
---
# Lloyd Long-Term Memory

## Infrastructure
- **Probe rule.** Calendar probe is calendar_list then calendar_events; a non-empty set is the positive control.
- **Restart split.** Backend restarts go through the round CLI, never bare supervisorctl, because the guardian reads it as a crash.

## Key Knowledge

### Retired decisions (Alan's rulings — do not re-propose)
- **Task #68 stays draft.** Alan's ruling 2026-09-17: it is deliberately parked.

### Corrections
- **Scope Preservation**: If user specifies N items, create N distinct tasks.
- **A long lesson with a proof.** """ + "It carries a proving command and many details. " * 12 + """

A free paragraph that is not a bullet but still an entry of its own, with detail.
"""


@pytest.fixture
def memories_root(tmp_path, monkeypatch):
    from agent_mcp import session

    root = tmp_path / "lloyd"
    root.mkdir()
    monkeypatch.setattr(session, "MEMORIES_ROOT", root)
    monkeypatch.setattr(ceiling, "MEMORIES_DIR", root)
    return session, root


def _set_cfg(monkeypatch, key, value):
    from app import config

    monkeypatch.setitem(config.CONFIG, key, value)


# ── size refusal names what to consolidate ─────────────────────────────────

def test_a_refusal_over_the_ceiling_names_the_largest_sections_and_untyped_count():
    big = MEMORY_FIXTURE + "\n## Bulky\n" + "- filler line that is long enough\n" * 3000
    msg = ps.size_error("MEMORY.md", big)
    assert msg and msg.startswith("MEMORY.md is ")
    assert "largest sections: 'Bulky'" in msg, msg
    assert "top-level entries carry no [type] tag" in msg
    assert "topics/<slug>" in msg


def test_the_hint_counts_only_untyped_top_level_entries():
    text = "- [feedback] a typed ruling here\n- an untyped entry here\n  - nested\n"
    assert ps.untyped_entry_count(text) == 1


def test_the_index_ceiling_is_one_constant_and_live():
    """Deployed 2026-09-25: `MEMORY_MD_CEILING_BYTES = MEMORY_MD_INDEX_CEILING_BYTES`."""
    assert ps.MEMORY_MD_INDEX_CEILING_BYTES == 25_600
    assert ps.MEMORY_MD_CEILING_BYTES == ps.MEMORY_MD_INDEX_CEILING_BYTES
    assert ps.MEMORY_CEILINGS["MEMORY.md"] == ps.MEMORY_MD_CEILING_BYTES


# ── topic files ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["topics/../SOUL", "topics/a/b", "topics/Upper",
                                 "topics/", "topics/" + "x" * 49, "../MEMORY.md",
                                 "SOUL.md", "memory/voice", "topics/voice.txt"])
def test_a_topic_file_lives_under_memory_only_and_traversal_is_refused(memories_root, bad):
    session, root = memories_root
    for fn in (session._memory_read, session._memory_add):
        res = fn({"file": bad, "entry": "an entry long enough"})
        assert res.get("code") == "INVALID_PARAM", (bad, res)
    assert not (root.parent / "SOUL.md").exists()


def test_memory_add_then_memory_read_a_topic_file(memories_root):
    session, root = memories_root
    assert session._memory_add({"file": "topics/voice-mode", "entry": "- detail one"})["success"]
    assert session._memory_add({"file": "topics/voice-mode.md", "entry": "- detail two"})["success"]
    path = root / "memory" / "voice-mode.md"
    assert path.read_text() == "# topics/voice-mode\n\n- detail one\n- detail two\n"
    got = session._memory_read({"file": "topics/voice-mode"})
    assert got == {"content": path.read_text(), "file": "topics/voice-mode"}


def test_a_missing_topic_read_lists_the_topics_that_exist(memories_root):
    session, root = memories_root
    session._memory_add({"file": "topics/alpha", "entry": "- a"})
    got = session._memory_read({"file": "topics/beta"})
    assert got["exists"] is False and got["topics"] == ["topics/alpha"]


def test_a_topic_file_is_refused_past_its_ceiling_by_every_writer(memories_root):
    session, root = memories_root
    session._memory_add({"file": "topics/big", "entry": "x" * 100})
    res = session._memory_add({"file": "topics/big", "entry": "y" * ceiling.TOPIC_FILE_CEILING_BYTES})
    assert res.get("code") == "INVALID_PARAM"
    assert "topic file ceiling" in res["error"]
    path = root / "memory" / "big.md"
    assert ceiling.memory_write_error(path, "z" * (ceiling.TOPIC_FILE_CEILING_BYTES + 1))
    assert ceiling.memory_write_error(root / "elsewhere.md", "z" * 10**6) is None


def test_topic_files_are_never_rendered_into_the_prompt(tmp_path, monkeypatch):
    overlay = tmp_path / "ov"
    (overlay / "memory").mkdir(parents=True)
    (overlay / "MEMORY.md").write_text("- [project] hook → topics/secret\n")
    (overlay / "memory" / "secret.md").write_text("TOPIC-BODY-SENTINEL\n")
    out = prompt_builder._load_memories(overlay, files=("MEMORY.md",))
    assert "TOPIC-BODY-SENTINEL" not in out and "hook → topics/secret" in out


# ── typed entries ──────────────────────────────────────────────────────────

def test_a_typed_entry_matches_the_uptake_grammar(memories_root, monkeypatch):
    session, root = memories_root
    _set_cfg(monkeypatch, "memory_tools", {"typed_entries": True, "date_stamp_entries": True})
    monkeypatch.setattr(session, "_entry_date", lambda: __import__("datetime").date(2026, 9, 24))
    assert session._memory_add({"entry": "Alan ruled task 68 stays draft",
                                "type": "feedback"})["success"]
    assert session._memory_add({"file": "USER.md", "entry": "- prefers short answers"})["success"]
    line = (root / "MEMORY.md").read_text().splitlines()[-1]
    assert line == "- [feedback] (2026-09-24) Alan ruled task 68 stays draft"
    assert (root / "USER.md").read_text().splitlines()[-1] == \
        "- [user] (2026-09-24) prefers short answers"
    assert uptake._MEMORY_DOC_RE.match(line)
    assert ps.TYPED_ENTRY_RE.match(line)


def test_a_tag_or_date_the_writer_already_wrote_is_not_doubled(memories_root, monkeypatch):
    session, root = memories_root
    _set_cfg(monkeypatch, "memory_tools", {"typed_entries": True, "date_stamp_entries": True})
    session._memory_add({"entry": "- [reference] (2026-01-02) lives in knowledge/x.md"})
    assert (root / "MEMORY.md").read_text() == "- [reference] (2026-01-02) lives in knowledge/x.md\n"


def test_an_unknown_type_is_refused(memories_root, monkeypatch):
    session, _ = memories_root
    _set_cfg(monkeypatch, "memory_tools", {"typed_entries": True})
    assert session._memory_add({"entry": "some entry", "type": "gossip"})["code"] == "INVALID_PARAM"


def test_typed_entries_off_writes_what_it_wrote_before(memories_root):
    session, root = memories_root  # conftest holds both switches off
    session._memory_add({"entry": "plain entry text"})
    assert (root / "MEMORY.md").read_text() == "plain entry text\n"


def test_memory_add_advertises_the_type_and_the_topic_grammar():
    from agent_mcp import session

    tools = {t.name: t for t in asyncio.run(session.list_tools())}
    props = tools["memory_add"].input_schema["properties"]
    assert props["type"]["enum"] == list(ps.ENTRY_TYPES)
    pat = re.compile(props["file"]["pattern"])
    assert pat.match("topics/voice") and pat.match("MEMORY.md")
    assert not pat.match("topics/../x")


# ── render-time overflow ───────────────────────────────────────────────────

def _over_ceiling_overlay(tmp_path, monkeypatch, limit=600):
    monkeypatch.setitem(ps.MEMORY_CEILINGS, "MEMORY.md", limit)
    overlay = tmp_path / "ov"
    overlay.mkdir()
    (overlay / "MEMORY.md").write_text(MEMORY_FIXTURE)
    return overlay


def test_render_all_is_bytewise_today(tmp_path, monkeypatch):
    overlay = _over_ceiling_overlay(tmp_path, monkeypatch)
    _set_cfg(monkeypatch, "memory", {"render_overflow": "render_all"})
    over = prompt_builder._load_memories(overlay, files=("MEMORY.md",))
    monkeypatch.setitem(ps.MEMORY_CEILINGS, "MEMORY.md", 10**9)
    under = prompt_builder._load_memories(overlay, files=("MEMORY.md",))
    assert over == under == f"## MEMORY.md\n{MEMORY_FIXTURE.strip()}"


def test_the_default_mode_is_render_all(monkeypatch):
    from app import config

    monkeypatch.setitem(config.CONFIG, "memory", {})
    assert prompt_builder._render_overflow_mode() == "render_all"
    monkeypatch.setitem(config.CONFIG, "memory", {"render_overflow": "bogus"})
    assert prompt_builder._render_overflow_mode() == "render_all"


def test_render_overflow_annotates_at_an_entry_boundary(tmp_path, monkeypatch, caplog):
    overlay = _over_ceiling_overlay(tmp_path, monkeypatch, limit=600)
    _set_cfg(monkeypatch, "memory", {"render_overflow": "annotate"})
    rang: list = []
    monkeypatch.setattr(prompt_builder, "_overflow_announce", lambda t, b: rang.append(t))
    monkeypatch.setattr(prompt_builder, "_overflow_noted", {})
    with caplog.at_level(logging.ERROR, logger="lloyd.prompt"):
        out = prompt_builder._load_memories(overlay, files=("MEMORY.md",))
        prompt_builder._load_memories(overlay, files=("MEMORY.md",))
    body, marker = out.split("\n\n<memory_overflow ")
    kept = body[len("## MEMORY.md\n"):]
    assert len(kept.encode()) <= 600
    assert MEMORY_FIXTURE.strip().startswith(kept)
    rest = MEMORY_FIXTURE.strip()[len(kept):]
    assert rest.startswith("\n"), "cut fell mid-line"
    nxt = rest.lstrip("\n").split("\n", 1)[0]
    assert nxt.startswith(("- ", "#")) or rest.startswith("\n\n"), f"cut fell mid-entry: {nxt!r}"
    dropped = int(re.search(r'dropped_bytes="(\d+)"', marker).group(1))
    assert dropped == len(MEMORY_FIXTURE.strip().encode()) - len(kept.encode())
    assert 'file="MEMORY.md"' in marker and 'memory_read(file="MEMORY.md")' in marker
    assert any("over its 600-byte ceiling" in r.message for r in caplog.records)
    assert len(rang) == 1, "announce is once a day per file"


def test_the_cut_never_splits_an_entry():
    text = "- one entry line\n- two entry line\n- three entry line"
    kept, dropped = prompt_builder._cut_at_entry_boundary(text, 25)
    assert kept == "- one entry line"
    assert dropped == len(text) - len(kept)


# ── validator, consolidator ────────────────────────────────────────────────

def _vmi():
    import importlib.util

    spec = importlib.util.spec_from_file_location("validate_memory_index", VALIDATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_validator_passes_a_built_index_and_fails_a_dangling_link(tmp_path):
    from scripts.memory import consolidate_memory_index as cmi

    src = tmp_path / "src"
    src.mkdir()
    (src / "MEMORY.md").write_text(MEMORY_FIXTURE)
    (src / "SOUL.md").write_text("# soul\n")
    (src / "USER.md").write_text("# user\n")
    out = tmp_path / "overlay"
    report = cmi.write_overlay(src, out)
    assert report["validation"]["ok"], report["validation"]
    vmi = _vmi()
    assert vmi.main(["--root", str(out), "--ceiling", str(ps.MEMORY_MD_INDEX_CEILING_BYTES)]) == 0
    index = (out / "MEMORY.md").read_text()
    assert ps.untyped_entry_count(index) == 0
    assert "- [feedback] **Task #68 stays draft.** Alan's ruling 2026-09-17: it is deliberately parked." in index
    # Lossless: every source entry is verbatim in exactly one topic file.
    topics = [p.read_text() for p in (out / "memory").glob("*.md")]
    _, units = cmi.parse_units(ps.body(MEMORY_FIXTURE))
    for u in units:
        for e in u.entries:
            assert sum(e.text in t for t in topics) == 1, e.text[:40]
    (out / "MEMORY.md").write_text(index + "- [project] gone → topics/nowhere\n")
    assert vmi.main(["--root", str(out), "--ceiling", "25600"]) == 1
    assert vmi.main(["--root", str(out), "--mode", "structure", "--ceiling", "25600"]) == 1


def test_the_validator_full_mode_refuses_an_untyped_unconsolidated_file(tmp_path):
    (tmp_path / "MEMORY.md").write_text(MEMORY_FIXTURE)
    vmi = _vmi()
    r = vmi.check(tmp_path, ceiling=100_000, mode="full")
    assert not r["ok"] and r["untyped_entries"] > 0
    assert vmi.check(tmp_path, ceiling=100_000, mode="structure")["ok"]


def test_the_consolidator_refuses_to_write_into_the_vault(tmp_path, monkeypatch):
    from scripts.memory import consolidate_memory_index as cmi

    vault = tmp_path / "obsidian"
    (vault / "lloyd").mkdir(parents=True)
    monkeypatch.setattr(cmi, "VAULT_ROOT", vault)
    with pytest.raises(cmi.RefusedOutput):
        cmi.write_overlay(tmp_path, vault / "lloyd")
    link = tmp_path / "sneaky"
    link.symlink_to(vault)
    with pytest.raises(cmi.RefusedOutput):
        cmi.write_overlay(tmp_path, link / "x")
    assert cmi.main(["--out", str(tmp_path / "o")]) == 2, "--dry-run is required"


# ── the dream skill names the validator, and it exists (#661 class) ────────

def test_the_vault_skill_patch_names_the_validator_and_it_exists():
    patch = PATCH.read_text()
    assert "skills/dream-consolidation/SKILL.md" in patch
    assert "skills/nightly-reflection-knowledge-write/SKILL.md" in patch
    added = "\n".join(ln for ln in patch.splitlines() if ln.startswith("+"))
    assert "scripts/memory/validate_memory_index.py" in added
    assert VALIDATOR.is_file()


@pytest.mark.live_vault
def test_the_live_dream_skill_names_the_validator_once_the_patch_is_applied():
    skill = Path.home() / "obsidian" / "skills" / "dream-consolidation" / "SKILL.md"
    if not skill.exists():
        pytest.skip("no vault")
    if "validate_memory_index.py" not in skill.read_text():
        pytest.skip("vault patch not applied (applied only if the index eval promotes): "
                    f"git -C ~/obsidian apply {PATCH}")
    assert VALIDATOR.is_file()


# ── eval runner ────────────────────────────────────────────────────────────

def test_the_probe_set_is_thirty_ten_of_each_kind():
    from eval.run_memory_index_ab import load_probes

    probes = load_probes()
    assert len(probes) == 30
    assert {k: sum(p.kind == k for p in probes) for k in ("index", "topic", "feedback")} == \
        {"index": 10, "topic": 10, "feedback": 10}
    assert all(p.objective_checks.get("tool_called") == "memory_read"
               for p in probes if p.kind == "topic")
    assert len(load_probes(with_trim=True)) == 50


def test_the_arm_deliverer_answers_memory_read_from_the_overlay(tmp_path):
    from eval.run_memory_index_ab import memory_read_deliverer

    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "voice.md").write_text("ARM-TOPIC\n")
    cb = memory_read_deliverer(tmp_path)
    out = asyncio.run(cb({"tool_name": "memory_read", "tool_input": {"file": "topics/voice"}},
                         None, None))
    content = json.loads(out["hookSpecificOutput"]["skillDeliver"]["content"])
    assert content == {"content": "ARM-TOPIC\n", "file": "topics/voice"}
    assert asyncio.run(cb({"tool_name": "Read", "tool_input": {}}, None, None)) == {}


def test_the_decision_needs_feedback_intact_and_topics_read():
    from eval.run_memory_index_ab import decide

    def rec(pid, kind, c, r, i, read=True):
        g = lambda v: {"verdict": v}  # noqa: E731
        return {"probe_id": pid, "kind": kind,
                "grades": {"canonical": g(c), "canonical_rep": g(r), "indexed": g(i)},
                "objective": {"indexed": {"tool_called": read}}}

    good = [rec(f"t{i}", "topic", "PASS", "PASS", "PASS") for i in range(10)]
    good += [rec(f"f{i}", "feedback", "PASS", "PASS", "PASS") for i in range(10)]
    assert decide(good)["promote_if_live_d_holds"] is True
    bad = good[:-1] + [rec("f9", "feedback", "PASS", "PASS", "FAIL")]
    d = decide(bad)
    assert d["criteria"]["b_no_feedback_lost"]["ok"] is False
    assert d["promote_if_live_d_holds"] is False
    unread = [rec(f"t{i}", "topic", "PASS", "PASS", "PASS", read=i < 7) for i in range(10)]
    assert decide(unread)["criteria"]["c_topic_answered_by_read"]["ok"] is False
    assert decide(good)["criteria"]["d_live_reads_per_user_turn"]["ok"] is None


def test_a_topic_probe_whose_answer_terms_reach_the_prompt_is_refused(tmp_path):
    """2026-09-25: a verbatim anchor is trivially absent from a clipped hook that
    still paraphrases the answer, so a topic probe's answer terms are checked
    against the whole prompt — the index and the shared SOUL.md/USER.md."""
    from eval.run_memory_index_ab import Probe, ProbeRejected, check_probe_anchors

    canon, idx = tmp_path / "canonical", tmp_path / "indexed"
    canon.mkdir()
    (idx / "memory").mkdir(parents=True)
    entry = "- **Gate misread (09-08).** The 19:36 full check reported 2,495 tests passing."
    (canon / "MEMORY.md").write_text(f"# M\n\n## Lessons\n{entry}\n")
    (idx / "memory" / "lessons.md").write_text(f"# Lessons\n\n{entry}\n")
    (idx / "SOUL.md").write_text("# soul\n")
    (idx / "USER.md").write_text("# user\n")

    def probe(terms):
        return Probe(id="t", kind="topic", prompt="?", criterion="?",
                     anchor="The 19:36 full check reported 2,495", answer_terms=terms)

    (idx / "MEMORY.md").write_text("- [project] **Gate misread (09-08).** → topics/lessons\n")
    p = probe(["2,495"])
    check_probe_anchors([p], canon, idx)
    assert p.topic == "lessons"
    with pytest.raises(ProbeRejected, match="not in topics/lessons"):
        check_probe_anchors([probe(["2,496"])], canon, idx)
    # The hook paraphrases the answer: the anchor is still absent, the term is not.
    (idx / "MEMORY.md").write_text(
        "- [project] **Gate misread (09-08).** Suite green, 2,495 passing… → topics/lessons\n")
    with pytest.raises(ProbeRejected, match="answer term '2,495'"):
        check_probe_anchors([probe(["2,495"])], canon, idx)
    (idx / "MEMORY.md").write_text("- [project] **Gate misread (09-08).** → topics/lessons\n")
    (idx / "USER.md").write_text("# user\nthe suite was 2,495 green\n")
    with pytest.raises(ProbeRejected, match="shared prompt files"):
        check_probe_anchors([probe(["2,495"])], canon, idx)


def test_every_topic_probe_names_answer_terms(tmp_path):
    import yaml

    from eval.run_memory_index_ab import ProbeRejected, load_probes

    assert all(p.answer_terms for p in load_probes() if p.kind == "topic")
    bad = tmp_path / "p.yaml"
    probes = [{"id": f"{k}{i}", "kind": k, "prompt": "?", "criterion": "?", "anchor": "a"}
              for k in ("index", "topic", "feedback") for i in range(10)]
    bad.write_text(yaml.safe_dump({"probes": probes}))
    with pytest.raises(ProbeRejected, match="answer_terms"):
        load_probes(bad)
