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

from app import prompt_builder
from app import prompt_surface as ps
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
    # Whole-dict equality kept on purpose; the one added key is #1796 clause 2 —
    # `memory_read` now echoes the absolute `path` its content came from, the same
    # field all three writers already return, because a caller cannot reconstruct
    # it from a destination argument it may have mis-named.
    assert got == {"content": path.read_text(), "file": "topics/voice-mode",
                   "path": str(path)}


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


def _hooked_root(tmp_path) -> Path:
    """The smallest root that is legal in BOTH modes: one index line, one topic file.

    Deliberately clean in `full` as well as `structure`, so a node that drops one
    unlinked file into it can attribute the failure to that file alone — and so the
    passing case is measured on the same fixture as the failing one, rather than on a
    second root whose cleanliness is assumed.
    """
    root = tmp_path / "lloyd"
    (root / "memory").mkdir(parents=True)
    (root / "MEMORY.md").write_text(
        "---\ntype: note\n---\n\n# Lloyd Long-Term Memory\n\n"
        "## Infra\n- [project] **Hooked rule.** A linked detail is a readable one."
        " → topics/hooked-rule\n",
        encoding="utf-8")
    (root / "memory" / "hooked-rule.md").write_text("# Hooked rule\ndetail\n",
                                                    encoding="utf-8")
    return root


def _orphan(root: Path, slug: str = "orphan-rule") -> Path:
    f = root / "memory" / f"{slug}.md"
    f.write_text(f"# {slug}\ndetail\n", encoding="utf-8")
    return f


def test_an_unlinked_topic_file_is_named_and_fails_both_modes(tmp_path, capsys):
    """The direction the validator did not have: topic file → index, not index → file.

    #2399. Dream consolidation #47 found six topic files on disk that no index line
    named, and every prompt was blind to all six, because `memory_read` is only ever
    reached from a `→ topics/<slug>` hook line and this script checked only that a
    link resolves. An unhooked file is a rule that exists and is never applied, so it
    is an error, in both modes, with no exemption for files that were already orphaned
    when the check was written — a warning nobody acts on is how six became the state
    of the live index, and a hooked root still passes in both modes today.
    """
    root = _hooked_root(tmp_path)
    _orphan(root)
    vmi = _vmi()
    r = vmi.check(root, ceiling=100_000, mode="structure")
    assert r["topic_files_unlinked"] == 1, r
    assert r["unlinked_topic_files"] == ["orphan-rule"], r
    assert not r["ok"]
    assert any("orphan-rule" in e for e in r["errors"]), r["errors"]
    assert not any("hooked-rule" in e for e in r["errors"]), (
        f"the linked file was reported too, so the check names every file: {r['errors']}")
    # Human mode says the slug out loud, and the count rides the summary line.
    for mode in ("full", "structure"):
        assert vmi.main(["--root", str(root), "--mode", mode, "--ceiling", "100000"]) == 1
    out = capsys.readouterr().out
    assert "orphan-rule" in out, out
    assert "1 unlinked" in out, out
    # The same fixture without the orphan is clean in both modes: the failure above
    # is the orphan, and no baseline exemption is needed to keep a nightly green.
    clean = _hooked_root(tmp_path / "clean")
    for mode in ("full", "structure"):
        assert vmi.main(["--root", str(clean), "--mode", mode, "--ceiling", "100000"]) == 0


def test_the_summary_line_carries_the_unlinked_count_beside_the_linked_count(tmp_path,
                                                                            capsys):
    """`0 unlinked` must read as "every file is hooked", not as "the check ran" (#2399).

    The denominator sits beside the verdict: the linked count alone was already
    ambiguous, because `85 links` over a root with an unhooked file and one without it
    printed the same thing. Asserted against the printed fragment rather than a regex
    over the whole line, so rewording the byte counts cannot hide the missing field.
    """
    vmi = _vmi()
    assert vmi.main(["--root", str(_hooked_root(tmp_path)), "--mode", "structure",
                     "--ceiling", "100000"]) == 0
    ok_line = capsys.readouterr().out.splitlines()[0]
    assert ok_line.startswith("OK:"), ok_line
    assert "1 topic files, 1 links, 0 unlinked (structure)" in ok_line, ok_line
    broken = _hooked_root(tmp_path / "broken")
    _orphan(broken)
    assert vmi.main(["--root", str(broken), "--mode", "structure",
                     "--ceiling", "100000"]) == 1
    fail_line = capsys.readouterr().out.splitlines()[0]
    assert fail_line.startswith("FAIL:"), fail_line
    # Two files on disk, one link, one orphan: the count that differs from the OK
    # line's is the unlinked one, which is the point of printing it.
    assert "2 topic files, 1 links, 1 unlinked (structure)" in fail_line, fail_line


def test_a_dangling_link_and_an_unlinked_file_are_independent_findings(tmp_path):
    """Both directions still fire, alone and together (#2399 clause 4).

    The pre-existing index → topic error had to survive the new one unchanged, and
    neither may mask the other: a root whose link dangles AND whose one file is
    unhooked reports both, while a root with only the dangling link reports zero
    unlinked files. That last half is the control that keeps the new set difference
    from being a check that fires on any malformed index.
    """
    vmi = _vmi()
    # The file that IS hooked stays on disk, so this root's set difference is
    # non-trivially empty: a link dangles, one file is linked, and the new direction
    # must still report nothing.
    only_dangling = _hooked_root(tmp_path / "dangling")
    (only_dangling / "MEMORY.md").write_text(
        (only_dangling / "MEMORY.md").read_text() + "- [project] gone → topics/nowhere\n",
        encoding="utf-8")
    r = vmi.check(only_dangling, ceiling=100_000, mode="structure")
    assert not r["ok"] and r["topic_files_unlinked"] == 0, r
    assert any("topics/nowhere" in e and "does not exist" in e for e in r["errors"]), r["errors"]
    assert vmi.main(["--root", str(only_dangling), "--mode", "structure",
                     "--ceiling", "100000"]) == 1
    both = _hooked_root(tmp_path / "both")
    _orphan(both)
    (both / "MEMORY.md").write_text(
        (both / "MEMORY.md").read_text() + "- [project] gone → topics/nowhere\n",
        encoding="utf-8")
    r2 = vmi.check(both, ceiling=100_000, mode="structure")
    assert any("topics/nowhere" in e for e in r2["errors"]), r2["errors"]
    assert r2["unlinked_topic_files"] == ["orphan-rule"], r2["errors"]
    assert vmi.main(["--root", str(both), "--mode", "structure", "--ceiling", "100000"]) == 1


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


# ── #1729: what a writer reports, and the route the guard must not break ─────

def test_each_memory_writer_reports_the_absolute_path_it_wrote(memories_root):
    """Clause 4: the result has to name the file, not echo the `file` argument.

    Each of the five mis-routed calls answered `{"success": true, "file":
    "MEMORY.md"}` — true, and about a file the caller never asked for, so the
    wrongness was something the run had to notice for itself. The absolute path is
    the half it cannot misread, and a write aimed at a topic can never print
    MEMORY.md in it.
    """
    session, root = memories_root
    (root / "MEMORY.md").write_text("# Lloyd Long-Term Memory\n- [project] one\n",
                                    encoding="utf-8")
    add = session._memory_add({"file": "MEMORY.md", "entry": "- [project] two"})
    replace = session._memory_replace({"file": "MEMORY.md", "old_text": "two",
                                       "new_text": "three"})
    remove = session._memory_remove({"file": "MEMORY.md", "entry": "three"})
    for res in (add, replace, remove):
        assert res.get("success") is True, res
        assert res["path"] == str(root / "MEMORY.md"), res
        assert Path(res["path"]).is_absolute(), res
    topic = session._memory_add({"file": "topics/user-md-ledger", "entry": "- a row"})
    assert topic["path"] == str(root / "memory" / "user-md-ledger.md"), topic
    assert "MEMORY.md" not in topic["path"], topic


def test_a_topic_write_still_lands_under_memory_with_the_index_untouched(memories_root):
    """Clause 5: the route the argument-surface change must not break.

    `topics/<slug>` resolution was never the defect — `file="topics/<slug>"` was
    answering correctly the whole time — so after the alias guard a topic write
    still lands at `<root>/memory/<slug>.md`, still reports that path, and leaves
    the loaded index byte for byte as it found it.
    """
    session, root = memories_root
    index = ("# Lloyd Long-Term Memory\n"
             "- [project] ledger → topics/user-md-ledger\n")
    (root / "MEMORY.md").write_text(index, encoding="utf-8")
    res = session._memory_add({"file": "topics/user-md-ledger",
                               "entry": "- why: the ledger backfill rows"})
    assert res.get("success") is True, res
    topic = root / "memory" / "user-md-ledger.md"
    assert topic.read_text(encoding="utf-8").endswith("- why: the ledger backfill rows\n")
    assert res["path"] == str(topic), res
    assert (root / "MEMORY.md").read_text(encoding="utf-8") == index


# ── #1895: the index's own bounds on the WRITE path ──────────────────────────
#
# Both bounds existed on the read side only until #1895: `git grep -n
# INDEX_LINE_MAX_CHARS` named the validator and the consolidator and no writer,
# while `memory_write_error` stopped at the 25,600 B ceiling. Vault git measured
# `lloyd/MEMORY.md` at 20,466 B (09-25) → 23,941 B (09-29 07:58Z) — 3,461 B over
# the 20,480 B tight limit for ~15 h with nothing refusing the appends, and the
# red `test_the_live_memory_index_validates` node nobody noticed until Pre-Flight.
# These nodes pin the write-side tripwire, the 300-character line cap, and that
# neither one reaches USER.md, topic files, lookalikes, or a shrinking repair.

INDEX_HEAD = "---\ntype: note\n---\n# Lloyd Long-Term Memory\n\n## Section\n"


def _index_of_size(size: int) -> str:
    """An index body whose UTF-8 size is exactly `size` bytes of short bullets."""
    text = INDEX_HEAD
    line = "- [project] filler entry that is comfortably short\n"
    while len(text.encode()) + len(line.encode()) < size:
        text += line
    rem = size - len(text.encode())
    if rem:
        text += "#" * (rem - 1) + "\n"   # a comment can never trip the line cap
    assert len(text.encode()) == size, "fixture is not the size asked for"
    return text


def test_the_index_bounds_are_defined_once_under_app_and_shared():
    """Clause 2: `git grep -n INDEX_LINE_MAX_CHARS` now names a file under `app/`,
    and both scripts resolve the cap and the 0.8 ratio through it instead of
    carrying their own copy — the reader and the writer cannot disagree by
    construction any more, which is the whole defect #1895 is about."""
    import subprocess

    hits = subprocess.run(["git", "grep", "-n", "INDEX_LINE_MAX_CHARS", "--", "app"],
                          cwd=ROOT, capture_output=True, text=True)
    assert hits.returncode == 0 and "app/memory_ceiling.py" in hits.stdout, hits.stdout
    vmi = _vmi()
    assert vmi.INDEX_LINE_MAX_CHARS == ceiling.INDEX_LINE_MAX_CHARS == 300
    assert vmi.DEFAULT_TIGHTNESS == ceiling.MEMORY_TIGHTNESS == 0.80
    for rel in ("scripts/memory/validate_memory_index.py",
                "scripts/memory/consolidate_memory_index.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert not re.search(
            r"^(?:INDEX_LINE_MAX_CHARS|DEFAULT_TIGHTNESS)\s*=\s*[\d.]", src, re.M), \
            f"{rel} restates one of the numbers instead of importing it"
        assert re.search(r"\bmc\.(INDEX_LINE_MAX_CHARS|MEMORY_TIGHTNESS)\b", src), rel


def test_a_growing_index_past_the_tight_limit_is_refused_and_routed(memories_root):
    """Clause 1: 20,480 B (80% of the live 25,600 B ceiling) is on the write path,
    and the refusal is the route — detail to `lloyd/memory/<slug>.md`, the index
    gets one ≤300-char `→ topics/<slug>` line — not just a number, because a bare
    byte count is what sends a writer to trim an unrelated entry (§2a-bis)."""
    session, root = memories_root
    limit = ceiling.tight_limit("MEMORY.md")
    assert limit == int(0.8 * ceiling.memory_ceiling("MEMORY.md")) == 20_480
    (root / "MEMORY.md").write_text(_index_of_size(limit - 512), encoding="utf-8")

    msg = ceiling.memory_write_error(root / "MEMORY.md", _index_of_size(limit + 665))
    assert msg is not None, "growth past the tight limit was allowed"
    assert "20,480" in msg and "21,145" in msg, msg          # the line, and the size
    assert "lloyd/memory/<slug>.md" in msg and "topics/<slug>" in msg, msg
    assert "300" in msg, msg                                 # the hook it routes to
    # The same append landing exactly ON the line is the largest legal write.
    assert ceiling.memory_write_error(root / "MEMORY.md", _index_of_size(limit)) is None


def test_an_index_line_past_the_cap_is_refused_at_301_characters(memories_root):
    """Clause 2's boundary: the cap is `>` and not `>=`. The live index's longest
    top-level line is 299 characters of 88, so the first refusal a nightly writer
    ever meets is this one, and it has to name the number as precisely as the byte
    rule does."""
    session, root = memories_root
    cap = ceiling.INDEX_LINE_MAX_CHARS
    ok = "- [project] " + "a" * (cap - len("- [project] "))
    assert len(ok) == cap, "fixture is not exactly at the cap"
    (root / "MEMORY.md").write_text(INDEX_HEAD + ok + "\n", encoding="utf-8")

    assert ceiling.memory_write_error(
        root / "MEMORY.md", INDEX_HEAD + ok + "\n") is None
    over = ok + "b"
    msg = ceiling.memory_write_error(root / "MEMORY.md", INDEX_HEAD + over + "\n")
    assert msg and f"{cap + 1:,} characters" in msg and f"{cap}-character" in msg, msg
    assert "lloyd/memory/<slug>.md" in msg and "topics/<slug>" in msg, msg


def test_the_write_guard_and_the_validator_measure_the_same_long_lines(memories_root):
    """Clause 2's predicate: dash bullets and star bullets at column zero, an
    indented line never, in `prompt_surface.body` — one function behind both the
    refusal and the validator's `long_lines` count, so they cannot drift."""
    session, root = memories_root
    cap = ceiling.INDEX_LINE_MAX_CHARS
    text = (INDEX_HEAD
            + "- [project] " + "a" * (cap + 40) + "\n"
            + "  - nested detail far longer than a top-level line " + "b" * (cap + 200) + "\n"
            + "* an untyped star bullet " + "c" * (cap + 5) + "\n")
    lines = ceiling.overlong_index_lines(text)
    assert len(lines) == 2 and lines[0].startswith("- ") and lines[1].startswith("* "), lines

    (root / "MEMORY.md").write_text(text, encoding="utf-8")
    vmi = _vmi()
    report = vmi.check(root, ceiling=4 * ceiling.memory_ceiling("MEMORY.md"), mode="full")
    assert report["long_lines"] == len(lines), report
    assert any("index lines over 300 chars" in e for e in report["errors"]), report["errors"]
    assert ceiling.memory_write_error(root / "MEMORY.md", text + "- x\n") is not None


def test_an_index_past_its_bounds_can_still_be_repaired(memories_root):
    """Clause 3's shrink escape, now over two more bounds: the file that is
    already past the line — the 09-29 state, 23,941 B with an over-long line — has
    to stay writable by the very trim that fixes it, or the guard refuses its own
    repair and the only writer left is the one it is refusing. Escape is strictly
    smaller, so rewriting the same bytes over is not a repair."""
    session, root = memories_root
    limit = ceiling.tight_limit("MEMORY.md")
    long_line = "- [project] " + "a" * (ceiling.INDEX_LINE_MAX_CHARS + 100)
    over = _index_of_size(limit + 3461) + long_line + "\n"
    (root / "MEMORY.md").write_text(over, encoding="utf-8")

    assert len(over.encode()) > limit, "fixture is not over the tight limit"
    smaller = _index_of_size(limit + 1000) + long_line + "\n"
    assert ceiling.memory_write_error(root / "MEMORY.md", smaller) is None, \
        "a shrinking write that keeps the over-long line was refused"
    assert ceiling.memory_write_error(
        root / "MEMORY.md", _index_of_size(limit + 1000)) is None
    assert ceiling.memory_write_error(root / "MEMORY.md", over) is not None, \
        "an identical rewrite of an over-limit index is not a shrink"


def test_the_new_bounds_leave_user_md_topic_files_and_lookalikes_alone(
        memories_root, monkeypatch):
    """Clause 3: nothing allowed today may start refusing. Live `USER.md` is
    16,373 of its own 16,384 B and carries 17 top-level bullets over 300 chars, so
    a rule that reached it would refuse every non-shrinking write the moment it
    shipped — freezing the one file §2a-ter tells the curator to act on."""
    session, root = memories_root
    # A ceiling small enough that the fixture sits BETWEEN USER.md's tight limit
    # and its ceiling: only there does a wrongly-scoped rule bite, and the live
    # file (16,373 of 16,384 B) sits in exactly that band today.
    monkeypatch.setitem(ps.MEMORY_CEILINGS, "USER.md", 600)
    user = "---\ntype: note\n---\n# User\n\n" + "- [user] " + "u" * 500 + "\n"
    (root / "USER.md").write_text(user, encoding="utf-8")
    size = len(user.encode())
    assert ceiling.tight_limit("USER.md") < size < ceiling.memory_ceiling("USER.md"), size
    assert ceiling.overlong_index_lines(user), "fixture must carry an over-cap line"
    assert ceiling.memory_write_error(root / "USER.md", user + "- [user] more\n") is None

    topic = root / "memory" / "voice.md"
    topic.parent.mkdir(exist_ok=True)
    topic.write_text("- detail\n", encoding="utf-8")
    assert ceiling.memory_write_error(topic, "- detail\n- more detail\n") is None

    lookalike = root.parent / "sandbox" / "MEMORY.md"
    lookalike.parent.mkdir(exist_ok=True)
    lookalike.write_text("short\n", encoding="utf-8")
    assert ceiling.memory_write_error(lookalike, _index_of_size(40_000)) is None


@pytest.mark.live_vault
def test_the_live_index_refuses_growth_and_the_live_user_md_does_not():
    """The reporting copy of both rules, measured on the live vault.

    Marked `live_vault` and so not run by the gate (pytest.ini): a nightly job owns
    these two files, and an assertion pinned to their current bytes would fail the
    next author for the previous writer's change. It re-measures the numbers rather
    than quoting them, so it stays true as the files move — and it skips, rather
    than passes, on the leg that has stopped meaning anything.
    """
    mem = Path.home() / "obsidian" / "lloyd"
    if not (mem / "MEMORY.md").is_file():
        pytest.skip("no vault")
    text = (mem / "MEMORY.md").read_text(encoding="utf-8")
    limit = ceiling.tight_limit("MEMORY.md")
    # Short lines only, so the ONLY rule that can fire is the byte one: an
    # over-long probe line would be refused for the wrong reason and still pass.
    probe = text
    unit = "- [project] filler entry that is comfortably short\n"
    while len(probe.encode()) <= limit:
        probe += unit
    size = len(probe.encode())
    assert size > limit and not ceiling.overlong_index_lines(probe), "probe setup"
    msg = ceiling.memory_write_error(mem / "MEMORY.md", probe)
    assert msg is not None, f"a {size:,} B MEMORY.md was allowed"
    assert f"{limit:,}" in msg and str(limit) in msg.replace(",", ""), msg
    assert f"{size:,}" in msg, msg
    assert "lloyd/memory/<slug>.md" in msg and "topics/<slug>" in msg, msg
    # The file as it stands, written back unchanged, is inside both bounds or the
    # byte rule is the only thing that can refuse it — never the line rule.
    same = ceiling.memory_write_error(mem / "MEMORY.md", text)
    assert same is None or f"{limit:,}" in same, same

    user = (mem / "USER.md").read_text(encoding="utf-8")
    user_bytes = len(user.encode("utf-8"))
    if user_bytes <= ceiling.tight_limit("USER.md"):
        pytest.skip("USER.md is inside its own tight limit: the freeze leg is vacuous")
    if ceiling.memory_ceiling("USER.md") - user_bytes >= 2:
        assert ceiling.memory_write_error(mem / "USER.md", user + "#\n") is None, \
            "the index rules reached USER.md and froze the curator's own file"


# ── the topic-slug cap on the WRITE guard (#2173) ───────────────────────────
# `TOPIC_SLUG_RE`'s 48-character cap sat in two places and neither was the write
# guard: the `file=` JSON-schema pattern on the memory tools, and
# `validate_memory_index.py`, whose own header calls this module "the write guard
# that has to refuse the same two conditions this script reports". `_topic_file_name`
# answered None for a stem outside the grammar, so the guard had no opinion — and
# the two lanes the knowledge-write skill routes a topic file through (`Write`,
# `vault_write`, neither of which passes the `file=` pattern) reached neither place
# that does have one. That is how vault commit `886f2d5b` (2026-10-04) left a
# 52-character topic file (`a-done-closure-over-a-refuted-premise-is-refuted-not.md`)
# standing: the validator's single reported error that night, found only by the
# next day's test run. Every node here runs against the tmp `memories_root`
# fixture, so it grades the guard and not a vault another job owns.

#: The incident's own stem: 52 characters, four over the cap.
INCIDENT_SLUG = "a-done-closure-over-a-refuted-premise-is-refuted-not"


def test_the_slug_cap_the_refusal_names_is_the_cap_the_regex_enforces():
    """One number, not two: the refusal quotes what `TOPIC_SLUG_RE` enforces.

    #1010's rule for ceilings — the number a refusal quotes and the number an
    auditor reports must be one constant — applied to the slug, which until now
    had 48 only as a literal inside the regex.
    """
    assert ceiling.TOPIC_SLUG_MAX_CHARS == 48
    assert ceiling.TOPIC_SLUG_RE.pattern == f"[a-z0-9-]{{1,{ceiling.TOPIC_SLUG_MAX_CHARS}}}"
    assert ceiling.TOPIC_SLUG_RE.fullmatch("a" * 48)
    assert not ceiling.TOPIC_SLUG_RE.fullmatch("a" * 49)
    msg = ceiling.topic_slug_error("a" * 52)
    assert "48-character topic-slug cap" in msg and "52" in msg, msg


@pytest.mark.parametrize("stem", [INCIDENT_SLUG, "a" * 49, "a" * 60])
def test_a_topic_stem_over_the_cap_is_refused_by_the_one_entry_point(memories_root, stem):
    """Clause 1: `memory_write_error` itself refuses it, naming the cap.

    This is the probe from the triage record — the call that returned `None` at
    base. The content is one short line on purpose, far under
    `TOPIC_FILE_CEILING_BYTES`, so the only rule that can fire is the slug one and
    the node cannot pass by catching the byte ceiling by accident.
    """
    _, root = memories_root
    assert len(stem) > ceiling.TOPIC_SLUG_MAX_CHARS, "probe setup"
    target = root / ceiling.TOPICS_SUBDIR / f"{stem}.md"
    msg = ceiling.memory_write_error(target, "- detail\n")
    assert msg is not None, f"a {len(stem)}-character topic slug was accepted"
    assert f"{ceiling.TOPIC_SLUG_MAX_CHARS}-character topic-slug cap" in msg, msg
    assert str(len(stem)) in msg, msg
    assert "topics/" in msg, msg


def test_a_stem_at_the_cap_still_writes_and_an_illegal_one_only_refuses_growth(
        memories_root):
    """The other side of the bound. A guard that refused every write to a file it
    cannot rename would leave the bad name standing forever, because the one lane
    left would be the one it is refusing — so the slug bound sits under the same
    shrink escape `MEMORY.md` gets, and a stem of exactly 48 stays writable.
    """
    _, root = memories_root
    tdir = root / ceiling.TOPICS_SUBDIR
    tdir.mkdir(parents=True, exist_ok=True)
    assert ceiling.memory_write_error(tdir / f"{'a' * 48}.md", "- detail\n") is None, \
        "a slug of exactly the cap is legal and must stay writable"
    assert ceiling.memory_write_error(tdir / "voice-mode.md", "- detail\n") is None

    over = tdir / f"{INCIDENT_SLUG}.md"
    over.write_text("- the detail this file was created to hold, in full\n",
                    encoding="utf-8")
    assert ceiling.memory_write_error(over, "- trimmed\n") is None, \
        "the shrinking repair write must stay possible after a refusal"
    assert ceiling.memory_write_error(over, "- " + "x" * 200 + "\n") is not None, \
        "growing the same file is what has to be refused"


@pytest.mark.parametrize("stem", ["Voice_Mode", "with.dot", "UPPER",
                                  "9" * 48 + "-x"])
def test_a_stem_outside_the_grammar_is_refused_as_the_validator_reports(
        memories_root, stem):
    """The guard now refuses what `validate_memory_index.check` reports for a topic
    file (`slug is not [a-z0-9-]{1,48}`), which is the condition that script's
    header says this guard owns. A short stem outside the grammar is the case
    clause 1's "at or under 48 characters" wording does not reach: it is not a
    length problem, and the refusal says which of the two it is while still naming
    the cap.
    """
    _, root = memories_root
    assert not ceiling.TOPIC_SLUG_RE.fullmatch(stem), "probe setup"
    msg = ceiling.memory_write_error(root / ceiling.TOPICS_SUBDIR / f"{stem}.md",
                                     "- detail\n")
    assert msg is not None, f"{stem}.md was accepted as a topic file"
    assert "48-character topic-slug cap" in msg, msg


def test_a_long_name_outside_the_topics_directory_is_still_nobody_s_business(
        memories_root):
    """The widening must not turn into a naming policy for the whole vault. A
    60-character knowledge note and a lookalike `memory/<long>.md` in a sandbox are
    two shapes a round and a research job write every day; both are writable today
    and both must stay writable, or this is an outage with a test attached.
    """
    _, root = memories_root
    long_note = root.parent / "knowledge" / "ai" / (f"{'a' * 60}.md")
    long_note.parent.mkdir(parents=True, exist_ok=True)
    assert ceiling.memory_write_error(long_note, "# note\n") is None
    lookalike = root.parent / "sandbox" / "memory" / f"{'a' * 60}.md"
    lookalike.parent.mkdir(parents=True, exist_ok=True)
    assert ceiling.memory_write_error(lookalike, "- detail\n") is None
    assert ceiling.memory_write_error(root / "MEMORY.md", "- [project] hook\n") is None


# ── #2212 — a ledger is measured against the file it audits ──────────────────
#
# `test_a_topic_file_is_refused_past_its_ceiling_by_every_writer` above is the
# shared ceiling, and it stays: an ordinary topic is pulled whole by `memory_read`,
# so 32,768 B is a prompt-budget bound and it still applies. A ledger is a different
# kind of file — never read into a prompt, one row per loaded index line — and until
# #2212 it was measured by that same number, which the live arithmetic had already
# outrun: 90 `MEMORY.md` lines x the ledger's own 675 B mean row is ~60,750 B, and
# 32,768 B holds ~48 rows. The audit would have stopped mid-file on a night that did
# nothing wrong. The four nodes below are clauses 1-3 from the caller's side — the
# acceptance probe, the ordinary topic that must not move, and the two halves of the
# shrink pair (a write inside the bound, and the trim the escape exists for); the
# derived number itself, its non-spread, and the prose that must agree with it are in
# `tests/test_memory_ledger_bound.py`.


def test_a_ledger_write_of_40000_bytes_is_allowed_and_78000_names_76800(memories_root):
    """Clause 1. The exact probe this item was triaged on: before the change,
    `memory_write_error` answered a 40,000 B prospective `memory-md-ledger` write
    with "over the 32,768-byte topic file ceiling (7,232 B over)", refusing bytes
    the file is entitled to hold.

    After it, the same call answers None, and the 78,000 B call refuses naming
    76,800 — the bound it was actually measured against — while the string 32,768
    appears nowhere in that message. Naming the wrong bound is the failure that
    would send a curator to split a live ledger, which `LEDGERS` cannot read back.
    """
    _, root = memories_root
    ledger = root / "memory" / "memory-md-ledger.md"

    assert ceiling.memory_write_error(ledger, "x" * 40_000) is None
    assert ceiling.ledger_ceiling("memory-md-ledger") == 76_800, "the derived bound moved"

    msg = ceiling.memory_write_error(ledger, "x" * 78_000)
    assert msg is not None, "a ledger is still bounded, just not at 32,768 B"
    assert "76,800" in msg, msg
    assert "32,768" not in msg, f"a ledger refusal must not quote the shared ceiling: {msg}"
    assert "1,200 B over" in msg, msg


def test_an_ordinary_topic_stem_is_still_refused_one_byte_past_32768(memories_root):
    """Clause 2, from the side the change must not touch.

    One size, two verdicts, both asserted in the same call sequence: 40,000 bytes is
    legal for `memory-md-ledger` (the node above) and refused for `topics/big`, so
    the derived bound cannot be a ceiling raised for the directory. And the refusal
    keeps its pre-#2212 wording byte for byte, because
    `test_a_topic_file_is_refused_past_its_ceiling_by_every_writer` and
    `scripts/memory/validate_memory_index.py` both quote it.
    """
    _, root = memories_root
    big = root / "memory" / "big.md"

    assert ceiling.memory_write_error(big, "z" * 32_768) is None
    msg = ceiling.memory_write_error(big, "z" * 32_769)
    assert msg is not None and "topic file ceiling" in msg, msg
    assert "32,768-byte topic file ceiling (1 B over)" in msg, msg
    assert ceiling.TOPIC_FILE_CEILING_BYTES == 32_768


def test_a_ledger_write_inside_its_bound_is_accepted_however_full_the_disk_is(memories_root):
    """Clause 3's measuring half — NOT the shrink escape, and the previous version of
    this docstring claimed it was, which the review of round SM_20261006_015104 named:
    a 70,000 B write sits inside the derived 76,800 B bound, so it passes with no escape
    in the tree at all. The case the escape exists for is the node below.

    What this node owns is that the guard still measures an over-bound ledger: 80,000
    bytes on disk, a 70,000 B write accepted, a 90,000 B write refused naming 76,800 —
    so the derived ceiling cannot read as "this file is unbounded now", and the refusal
    cannot name the wrong number. The fixture is deliberately an already-over-bound
    ledger: it is the state a real `memory-md-ledger` reaches the first night its rows
    outgrow 76,800 B, and the only state where the two guards can be told apart.
    """
    _, root = memories_root
    (root / "memory").mkdir(parents=True, exist_ok=True)
    ledger = root / "memory" / "memory-md-ledger.md"
    ledger.write_text("x" * 80_000, encoding="utf-8")

    assert ceiling.ledger_ceiling("memory-md-ledger") == 76_800, "the derived bound moved"
    assert ceiling.memory_write_error(ledger, "x" * 70_000) is None
    msg = ceiling.memory_write_error(ledger, "x" * 90_000)
    assert msg is not None and "76,800" in msg, msg
    assert "32,768" not in msg, f"a ledger refusal must not quote the shared ceiling: {msg}"


def test_an_over_bound_ledger_uses_the_shrink_escape_over_its_own_bound(memories_root):
    """Clause 3, the case the shrink escape (#2173) exists for — untested anywhere for a
    ledger until the review of round SM_20261006_015104 named it.

    The escape is #2173's first check — `size < _on_disk_bytes(path)`, STRICTLY smaller
    than the bytes already on disk, measured before either bound so one rule covers the
    byte ceiling, the tight limit and the name — so a file over its ceiling can always be
    rewritten shorter. For a ledger the only interesting instance of that is a write that
    is BOTH over the derived bound AND shorter than the file: 90,000 B on disk, trimmed to
    78,000 B — still 1,200 B over its bound, and still the repair the file needs. Without
    the escape that write is refused, and an over-bound ledger could only ever be
    repaired by a rewrite that lands under the bound in a single cut: the freeze #2173
    exists to prevent, moved from the shared ceiling onto the derived one.

    Four calls say where the rule stops. 78,000 and 80,000 both accepted — every strictly
    smaller write is, so a curator may trim in stages and stop above the bound without the
    guard blocking the next cut. 90,000, equal to the bytes on disk and therefore not
    smaller than them, refused naming 76,800 and not 32,768: the escape ends at strictly
    smaller, and a same-size rewrite keeps the overrun exactly as over as it was. 90,001
    refused the same way — a growth is measured against the derived bound, which is also
    the half proving the two acceptances above were the escape firing and not a guard that
    stopped measuring.
    """
    _, root = memories_root
    (root / "memory").mkdir(parents=True, exist_ok=True)
    ledger = root / "memory" / "memory-md-ledger.md"
    ledger.write_text("x" * 90_000, encoding="utf-8")

    bound = ceiling.ledger_ceiling("memory-md-ledger")
    assert bound == 76_800, "the derived bound moved, and the sizing below with it"
    trim = 78_000
    assert trim > bound, (
        f"the write under test must be OVER the bound ({bound}) or this node is the one "
        "above again, which is the finding it was written to answer")
    for smaller in (trim, 80_000):
        assert ceiling.memory_write_error(ledger, "x" * smaller) is None, (
            f"the shrink escape did not fire for a {smaller} B write into a 90,000 B "
            "ledger: an over-bound ledger can now only be repaired by a write that lands "
            "under the bound in one go")

    for not_smaller in (90_000, 90_001):
        msg = ceiling.memory_write_error(ledger, "x" * not_smaller)
        assert msg is not None and "76,800" in msg, (not_smaller, msg)
        assert "32,768" not in msg, (
            f"a ledger refusal must not quote the shared ceiling: {msg}")
