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

    The route half of the message re-measures too, since #3298. It used to demand the
    generic `<slug>` route unconditionally, which is true only while the index has room
    for a line; the same file asserts at :1188 that those spellings must be ABSENT from
    a refusal at room 20, so one of the two had to be wrong and it was this one. The
    contradiction was latent until 2026-10-25, when the nightly wrote the live index to
    20,465 B — 15 B of room, under the 24-byte floor — and it has been red ever since,
    which is why this leg is not simply pinned to today's state: a dream #47 trim under
    20,456 B turns the old leg green again and the two assertions would contradict each
    other unseen until the next fat nightly. The leg now asks the live file how much room
    it has and grades whichever refusal that room selects, through the function a fixture
    test drives in both states — so the branch is covered by the gate-visible suite and
    this node reports which side the vault is on.
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
    # The room the guard is about to read, read the way the guard reads it: bytes on
    # disk, not the length of the text `read_text` handed back, which loses any CRLF
    # the file has and would quietly move this across the 24-byte floor.
    room = limit - len((mem / "MEMORY.md").read_bytes())
    msg = ceiling.memory_write_error(mem / "MEMORY.md", probe)
    assert msg is not None, f"a {size:,} B MEMORY.md was allowed"
    assert f"{limit:,}" in msg and str(limit) in msg.replace(",", ""), msg
    assert f"{size:,}" in msg, msg
    _assert_index_route_matches_room(msg, room, "live lloyd/MEMORY.md")
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


def test_a_ledger_write_of_40000_bytes_is_allowed_and_one_byte_past_the_bound_names_it(memories_root):
    """Clause 1 of #2212, re-pointed by #3015 at the bound that replaced the first one.
    The probe's shape is unchanged and so is its purpose: before #2212,
    `memory_write_error` answered a 40,000 B prospective `memory-md-ledger` write with
    "over the 32,768-byte topic file ceiling (7,232 B over)", refusing bytes the file
    is entitled to hold.

    After it, the same call answers None, and a write one byte past the derived bound
    refuses naming that bound — 102,400 since #3015 raised the multiple to 4 — while the
    string 32,768 appears nowhere in that message. Naming the wrong bound is the failure
    that would send a curator to split a live ledger, which `LEDGERS` cannot read back;
    over-bound by exactly one byte is where a wrong number does most damage, because it
    is also the size a nightly's next row actually is (#3015's own 937 B of slack was
    four such writes away from being zero).
    """
    _, root = memories_root
    ledger = root / "memory" / "memory-md-ledger.md"
    bound = ceiling.ledger_ceiling("memory-md-ledger")

    assert ceiling.memory_write_error(ledger, "x" * 40_000) is None
    assert bound == 102_400, "the derived bound moved"

    msg = ceiling.memory_write_error(ledger, "x" * (bound + 1))
    assert msg is not None, "a ledger is still bounded, just not at 32,768 B"
    assert "102,400" in msg, msg
    assert "32,768" not in msg, f"a ledger refusal must not quote the shared ceiling: {msg}"
    assert "1 B over" in msg, msg


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

    What this node owns is that the guard still measures an over-bound ledger: 110,000
    bytes on disk, a 100,000 B write accepted, a 110,001 B write refused naming 102,400 —
    so the derived ceiling cannot read as "this file is unbounded now", and the refusal
    cannot name the wrong number. The sizes are the bound's own arithmetic, so the node
    survives a multiplier the way the guard does. The fixture is deliberately an
    already-over-bound ledger: it is the state a real `memory-md-ledger` reaches the
    first night its rows outgrow its bound, and the only state where the two guards can
    be told apart.
    """
    _, root = memories_root
    (root / "memory").mkdir(parents=True, exist_ok=True)
    ledger = root / "memory" / "memory-md-ledger.md"
    bound = ceiling.ledger_ceiling("memory-md-ledger")
    ledger.write_text("x" * (bound + 7_600), encoding="utf-8")

    assert bound == 102_400, "the derived bound moved"
    assert ceiling.memory_write_error(ledger, "x" * (bound - 2_400)) is None
    msg = ceiling.memory_write_error(ledger, "x" * (bound + 7_601))
    assert msg is not None and "102,400" in msg, msg
    assert "32,768" not in msg, f"a ledger refusal must not quote the shared ceiling: {msg}"


def test_an_over_bound_ledger_uses_the_shrink_escape_over_its_own_bound(memories_root):
    """Clause 3, the case the shrink escape (#2173) exists for — untested anywhere for a
    ledger until the review of round SM_20261006_015104 named it.

    The escape is #2173's first check — `size < _on_disk_bytes(path)`, STRICTLY smaller
    than the bytes already on disk, measured before either bound so one rule covers the
    byte ceiling, the tight limit and the name — so a file over its ceiling can always be
    rewritten shorter. For a ledger the only interesting instance of that is a write that
    is BOTH over the derived bound AND shorter than the file: 120,000 B on disk, trimmed
    to 104,000 B — still 1,600 B over its bound, and still the repair the file needs.
    Without the escape that write is refused, and an over-bound ledger could only ever be
    repaired by a rewrite that lands under the bound in a single cut: the freeze #2173
    exists to prevent, moved from the shared ceiling onto the derived one.

    Four calls say where the rule stops. 104,000 and 110,000 both accepted — every strictly
    smaller write is, so a curator may trim in stages and stop above the bound without the
    guard blocking the next cut. 120,000, equal to the bytes on disk and therefore not
    smaller than them, refused naming 102,400 and not 32,768: the escape ends at strictly
    smaller, and a same-size rewrite keeps the overrun exactly as over as it was. 120,001
    refused the same way — a growth is measured against the derived bound, which is also
    the half proving the two acceptances above were the escape firing and not a guard that
    stopped measuring.
    """
    _, root = memories_root
    (root / "memory").mkdir(parents=True, exist_ok=True)
    ledger = root / "memory" / "memory-md-ledger.md"
    bound = ceiling.ledger_ceiling("memory-md-ledger")
    disk = bound + 17_600
    ledger.write_text("x" * disk, encoding="utf-8")

    assert bound == 102_400, "the derived bound moved, and the sizing below with it"
    trim = bound + 1_600
    assert trim > bound, (
        f"the write under test must be OVER the bound ({bound}) or this node is the one "
        "above again, which is the finding it was written to answer")
    for smaller in (trim, bound + 7_600):
        assert ceiling.memory_write_error(ledger, "x" * smaller) is None, (
            f"the shrink escape did not fire for a {smaller} B write into a {disk} B "
            "ledger: an over-bound ledger can now only be repaired by a write that lands "
            "under the bound in one go")

    for not_smaller in (disk, disk + 1):
        msg = ceiling.memory_write_error(ledger, "x" * not_smaller)
        assert msg is not None and "102,400" in msg, (not_smaller, msg)
        assert "32,768" not in msg, (
            f"a ledger refusal must not quote the shared ceiling: {msg}")


# ── #2838: a no-room index refusal must not order an index line ───────────────
#
# `index_file_refusal` tells a writer to write a topic file AND to "give the index ONE typed
# line … ending `→ topics/<slug>`". That second half is advice only while a line fits, and the
# tight limit is the bound that stops one fitting: at the size the index actually fails at —
# 20,463 B on disk against the 20,480 B limit, the figure #2649's autotriage record quotes —
# the shortest legal typed line (24 B) is refused, so the refusal ordered the edit it refuses
# next. A writer that complied anyway left a detail file the index never linked: #2613, #2649.
#
# `index_no_room_error` answers that case. Its predicate is the index's room ON DISK, which is
# why every fixture below writes the file before asking — `_on_disk_bytes` measures there and
# refuses to be handed a number (`app/memory_ceiling.py:248-259`). `TOPIC_LINE_MIN_BYTES` is
# 24, so `room=20` is four bytes short of the cheapest line the old sentence could order, and
# `room=512` is the room the tight-limit fixture above leaves.

def _index_linking(target: str, room: int, limit: int) -> str:
    """Index text of exactly `limit - room` bytes, every bullet of it linking `target`.

    The slack is absorbed by a `#` heading line — neither a bullet nor an index line, so it
    cannot trip the 300-character cap the same guard enforces — which lets the fixture be
    sized to the byte. The assert is part of the helper: on this path the room IS the branch.
    """
    line = f"- [project] filler entry that is comfortably short → {target}\n"
    size = limit - room
    text = INDEX_HEAD
    while len(text.encode("utf-8")) + len(line.encode("utf-8")) <= size - 2:
        text += line
    rem = size - len(text.encode("utf-8"))
    assert rem >= 2, f"no room left for a padding line at size {size}"
    text += "#" * (rem - 1) + "\n"
    assert len(text.encode("utf-8")) == size, (
        f"fixture is {len(text.encode())} B, not {size}: the room is the branch")
    return text


def _grow_by(text: str, extra: int) -> str:
    """`text` plus exactly `extra` more bytes, as a heading line no rule reads."""
    assert extra >= 2
    return text + "#" * (extra - 1) + "\n"


def _assert_index_route_matches_room(msg: str, room: int, where: str) -> None:
    """What route text a refusal must carry, decided by the index's room. One place.

    This file held two assertions that contradict each other until #3298: the live node
    demanded `lloyd/memory/<slug>.md` and `topics/<slug>` of a live `lloyd/MEMORY.md`
    at 20,465 B against its 20,480 B tight limit — 15 B of room, nine under the 24-byte
    `TOPIC_LINE_MIN_BYTES` — while `test_a_no_room_index_refusal_orders_no_index_line`
    (:1188) demanded those same two strings be ABSENT from a refusal at room 20 (#2838,
    `bb99f4e0`). The message is right in both cases and the older assertion was wrong
    about which refusal the live state selects: below the floor `index_no_room_error`
    answers, and stopping the writer being told to "give the index ONE typed line …
    ending `→ topics/<slug>`" is the entire reason that branch exists — an index line
    costs 24 B and the guard has just measured fewer than that. The placeholders are
    advice again only in `index_tight_limit_error`, the roomy branch that can honour
    them. Room picks the text; `where` says which room this call read, because the same
    assert off the live vault and off a fixture is a different claim about the world.
    """
    if room >= ceiling.TOPIC_LINE_MIN_BYTES:
        assert "lloyd/memory/<slug>.md" in msg and "topics/<slug>" in msg, (
            f"{where}: with {room} B free — at least the "
            f"{ceiling.TOPIC_LINE_MIN_BYTES} B the shortest legal index line costs — the "
            "refusal must carry the generic add-a-line route, since that is the route "
            f"an agent can actually follow there: {msg}")
    else:
        assert "consolidation owed" in msg, (
            f"{where}: with {room} B free the refusal must send the writer to "
            f"dream-consolidation (#47), the only action that reopens room: {msg}")
        for placeholder in ("lloyd/memory/<slug>.md", "topics/<slug>"):
            assert placeholder not in msg, (
                f"{where}: {placeholder!r} appears in a refusal whose index has {room} B "
                "free — that is the instruction #2838 removed, ordering an edit the same "
                f"guard then refuses: {msg}")


def _chain_before_2838(path, prospective: str):
    """`memory_write_error` exactly as it read before #2838, for verdict parity.

    Re-typed from the shipped helpers rather than loaded out of git so the comparison uses the
    same live constants the new chain does — a change to a bound moves both sides together. The
    shrink escape is read through `_on_disk_bytes` like the real one, not through
    `read_bytes`, so a missing file answers 0 here too.
    """
    filename = Path(path).name
    topic = ceiling.topic_slug(filename)
    if topic is not None:
        return ceiling.topic_size_error(filename, prospective)
    bound = ps.memory_ceiling(filename) or 0
    size = len(prospective.encode("utf-8"))
    if size < ceiling._on_disk_bytes(Path(path)):
        return None
    if size > bound:
        return ps.size_error(filename, prospective)
    if filename == "MEMORY.md":
        return (ceiling.index_line_length_error(filename, prospective)
                or ceiling.index_tight_limit_error(filename, prospective))
    return None


def test_a_no_room_index_refusal_orders_no_index_line(memories_root):
    """Clause 1: at 20 B of room the refusal stops telling the writer to add a line.

    The index sits 20 B under `tight_limit('MEMORY.md')` on disk and the write adds an entry, so
    the prospective file is over the limit and the verdict is the refusal
    `index_tight_limit_error` returns today — only the text changes. Three phrases are pinned
    out: the order itself ("ONE typed line"), the placeholder it targets ("topics/<slug>"), and
    "give the index", the clause that sentence introduces itself with. The third is what keeps a
    re-word honest: a message that dropped the byte count but kept that clause would still order
    an edit the same guard refuses on the next call.
    """
    session, root = memories_root
    limit = ceiling.tight_limit("MEMORY.md")
    index = _index_linking("topics/a-topic", room=20, limit=limit)
    (root / "MEMORY.md").write_text(index, encoding="utf-8")

    msg = ceiling.memory_write_error(root / "MEMORY.md",
                                     index + "- [project] a new finding → topics/new-rule\n")

    assert msg is not None, "the write must stay refused — this clause is about text only"
    assert "ONE typed line" not in msg, msg
    assert "topics/<slug>" not in msg, msg
    assert "give the index" not in msg, msg
    assert "consolidation owed" in msg, msg


def test_the_no_room_refusal_names_a_linked_topic_and_the_tight_limit(memories_root):
    """Clause 2: the route it gives names a linked topic, its room, and the byte bound.

    The fixture's topic sits at about half `topic_ceiling` — the live counterpart,
    `lloyd/memory/operational-corrections-anti-patterns.md`, measured 19,069 B of 32,768 B on
    2026-10-16 — and the index links it, so the merge the refusal proposes is a write this same
    guard admits, which the last assertion checks instead of assuming. Four facts have to be
    printed for a writer to act without guessing: the linked name as the index spells it, the
    bytes free of it, the ceiling those bytes are free under, and the tight limit as the reason
    no line can be added — with its thousands separator, the form every other byte count in
    this module's refusals uses and the one a nightly greps.
    """
    session, root = memories_root
    limit = ceiling.tight_limit("MEMORY.md")
    topic = root / "memory" / "a-topic.md"
    topic.parent.mkdir(exist_ok=True)
    topic.write_text("# A topic\n\n" + "detail. " * 2000, encoding="utf-8")   # ~16 KB of 32,768
    index = _index_linking("topics/a-topic", room=20, limit=limit)
    (root / "MEMORY.md").write_text(index, encoding="utf-8")

    msg = ceiling.memory_write_error(root / "MEMORY.md",
                                     index + "- [project] a new finding → topics/new-rule\n")

    assert "topics/a-topic" in msg, msg
    assert f"{limit:,}" in msg, f"the tight limit has to be the stated reason: {msg}"
    assert "20 of those bytes are free" in msg, msg
    bound = ceiling.topic_ceiling("topics/a-topic")
    assert f"{bound - len(topic.read_bytes()):,}" in msg and f"{bound:,}" in msg, msg
    assert "No new index line can be added" in msg, msg
    assert "dream-consolidation" in msg, (
        "the run that reopens the room has to be named: " + msg)
    assert "consolidation owed" in msg, msg
    assert ceiling.topic_size_error("topics/a-topic", topic.read_text() + "x" * 2000) is None, (
        "the refusal offered a target its own topic ceiling would refuse")


def test_the_no_room_refusal_invents_no_target_when_no_linked_topic_has_room(memories_root):
    """Clause 2's negative: no linked topic with room means no name, and still the record.

    The one topic this index links is 100 B under its ceiling, below the
    `MIN_TOPIC_MERGE_BYTES` the module documents — the guard cannot know the length of a detail
    it has not been handed, so it must not offer a file that may not hold it. Naming
    `topics/nearly-full` anyway would have the writer's next call refused by the topic ceiling,
    which is how an orphan is made: an instruction the same guard rejects.
    """
    session, root = memories_root
    limit = ceiling.tight_limit("MEMORY.md")
    bound = ceiling.topic_ceiling("topics/nearly-full")
    (root / "memory").mkdir(exist_ok=True)
    (root / "memory" / "nearly-full.md").write_text(
        "# Nearly full\n\n" + "x" * (bound - 100), encoding="utf-8")
    index = _index_linking("topics/nearly-full", room=20, limit=limit)
    (root / "MEMORY.md").write_text(index, encoding="utf-8")

    msg = ceiling.memory_write_error(root / "MEMORY.md",
                                     index + "- [project] a new finding → topics/new-rule\n")

    assert "topics/nearly-full" not in msg, msg
    assert "ONE typed line" not in msg, msg
    assert "cannot be created" in msg, msg
    assert "consolidation owed" in msg, msg


def test_an_index_with_a_lines_worth_of_room_keeps_the_add_a_line_route(memories_root):
    """Clause 3: with room for a line the refusal keeps its present text, byte for byte.

    Two rooms are sampled because the branch is a comparison against a floor and not a flag. At
    `limit - 512`, the room the pre-existing test above leaves, and at `limit - 24` =
    `TOPIC_LINE_MIN_BYTES` exactly, the write is over the tight limit and both must still read
    "give the index ONE typed line": at 24 B of room a line of the floor size lands the file ON
    the limit, which `index_tight_limit_error` does not refuse (`>` not `>=`). Neither case is
    one to re-text, and the second is exactly where a branch written as "the tight refusal
    fired" rather than "no line fits" would have gone wrong. The final assertion keeps the
    branch itself out of the entry point: `memory_write_error` grew no message of its own, so
    its verdicts stay the ones its docstring describes.
    """
    session, root = memories_root
    limit = ceiling.tight_limit("MEMORY.md")
    two_lines = ("- [project] an entry with something worth recording here → topics/a-topic\n"
                 "- [project] and a second entry the same write must also carry → topics/x\n")
    assert len(two_lines.encode("utf-8")) > ceiling.TOPIC_LINE_MIN_BYTES

    for room in (512, ceiling.TOPIC_LINE_MIN_BYTES):
        index = _index_linking("topics/a-topic", room=room, limit=limit)
        (root / "MEMORY.md").write_text(index, encoding="utf-8")
        extra = 665 if room == 512 else len(two_lines.encode("utf-8"))
        prospective = index + two_lines if room != 512 else _grow_by(index, extra)
        assert len(prospective.encode("utf-8")) > limit, "fixture must be a growing write"

        msg = ceiling.memory_write_error(root / "MEMORY.md", prospective)

        assert msg is not None, f"room {room}: the write must still be refused"
        assert f"over the {limit:,}-byte tight limit" in msg, msg
        assert "ONE typed line" in msg and "topics/<slug>" in msg, msg
        assert "This is stop-and-write-a-topic-file" in msg, msg

    # The entry point grew no message of its own. Its docstring is the one long string that
    # belongs in `co_consts`, so it is excluded by identity rather than by length — a length
    # filter alone would trip on the docstring and read as a verdict change.
    entry_consts = [c for c in ceiling.memory_write_error.__code__.co_consts
                    if isinstance(c, str) and c is not ceiling.memory_write_error.__doc__]
    assert not [c for c in entry_consts if len(c) > 40], (
        "a refusal message appeared in the entry point instead of in the callee it delegates "
        f"to: {[c for c in entry_consts if len(c) > 40]}")


def test_the_route_a_refusal_carries_is_decided_by_the_room_the_index_has(memories_root):
    """#3298 clause 2: both sides of the route leg, graded with no vault read.

    `_assert_index_route_matches_room` is what the live node now reads, and the gate
    deselects `live_vault` — so a helper reached only from that node would have each of
    its branches checked once a night, by whichever branch the nightly job happens to
    leave the index in. This drives both from fixtures instead: room 20, the no-room
    state the live index has been in since the 2026-10-25 nightly wrote it to 20,465 B of
    its 20,480 B limit, and room `TOPIC_LINE_MIN_BYTES` exactly, the floor where a
    24-byte line lands the file ON the limit and adding a line is still honest advice.

    Two things keep the driver honest rather than decorative. The room each message is
    graded against is re-read from the file the fixture wrote, not taken from the loop
    argument, so a fixture that failed to land the intended size cannot pass on the
    branch its argument names. And the final block compares the two refusals against
    each other: the placeholders must be absent from the no-room one and present in the
    floor one, which is the only assertion here that fails if the guard ever learns to
    answer both rooms with a single text — a case in which the helper would otherwise
    grade one message twice and call it two branches.
    """
    session, root = memories_root
    limit = ceiling.tight_limit("MEMORY.md")
    topic = root / "memory" / "a-topic.md"
    topic.parent.mkdir(exist_ok=True)
    topic.write_text("# A topic\n\n" + "detail. " * 2000, encoding="utf-8")   # ~16 KB of 32,768
    two_lines = ("- [project] an entry with something worth recording here → topics/a-topic\n"
                 "- [project] and a second entry the same write must also carry → topics/x\n")

    msgs = {}
    for room, label in ((20, "below the floor"),
                        (ceiling.TOPIC_LINE_MIN_BYTES, "at the floor")):
        index = _index_linking("topics/a-topic", room=room, limit=limit)
        (root / "MEMORY.md").write_text(index, encoding="utf-8")
        prospective = index + two_lines
        assert len(prospective.encode("utf-8")) > limit, "the write must be a growing one"

        msg = ceiling.memory_write_error(root / "MEMORY.md", prospective)
        assert msg is not None, f"{label}: the write must stay refused"
        assert f"over the {limit:,}-byte tight limit" in msg, msg

        room_read = limit - len((root / "MEMORY.md").read_bytes())
        assert room_read == room, (
            f"{label}: the guard saw {room_read} B of room, not the {room} the fixture "
            "asked for, so it did not produce the refusal this case is about")
        _assert_index_route_matches_room(msg, room_read, f"fixture {label}")
        msgs[room] = msg

    for placeholder in ("lloyd/memory/<slug>.md", "topics/<slug>"):
        assert placeholder not in msgs[20] and placeholder in msgs[ceiling.TOPIC_LINE_MIN_BYTES], (
            f"{placeholder!r} is not the pair's discriminator any more — both branches "
            f"below and at the floor read alike, so the helper above is grading one "
            f"message twice: {msgs[20]!r}")


def test_the_no_room_branch_moves_no_verdict(memories_root):
    """Clause 4: the new branch answers where the old chain already refused, never wider.

    The chain this module ran before #2838 is re-typed as `_chain_before_2838` and sampled
    against it over prospective sizes covering every path the entry point can take: a shrinking
    file, a line that fits the room, one landing exactly on the limit, one a byte over, one past
    the ceiling, a long line in a roomy index, a long line in a NO-ROOM index, and the no-room
    new-entry case. Verdict parity is asserted in both directions because the half that matters
    here is "nothing that passed now fails" — a new refusal is a new way to lose a finding — and
    the text is asserted identical everywhere except the one case the clause is about. The long
    line over a no-room index is the ordering proof: the line-cap refusal is still first, so
    `index_no_room_error` sits between the two and not above them. The shrink escape is sampled
    at the no-room size on purpose, since that is the state where losing it would strand the
    file over its limit with no way back.
    """
    session, root = memories_root
    limit = ceiling.tight_limit("MEMORY.md")
    roomy = _index_linking("topics/a-topic", room=512, limit=limit)
    tight = _index_linking("topics/a-topic", room=20, limit=limit)
    long_line = f"- [project] {'k' * 400} → topics/a-topic\n"
    assert len(long_line) - 1 > ceiling.INDEX_LINE_MAX_CHARS

    cases = {
        "shrinking an over-tight file": (tight, tight[:2000]),
        "a line that fits the room": (roomy, roomy + "- [project] fits → topics/a-topic\n"),
        "exactly on the limit": (roomy, _grow_by(roomy, limit - len(roomy.encode("utf-8")))),
        "one byte over": (roomy, _grow_by(roomy, limit - len(roomy.encode("utf-8")) + 1)),
        "past the ceiling": (roomy, _grow_by(roomy, 6000)),
        "a long line, room on disk": (roomy, roomy + long_line),
        "a long line, no room on disk": (tight, tight + long_line),
        "no room, a new entry": (tight, tight + "- [project] new → topics/new-rule\n"),
    }
    # Nothing below samples a refusal of a fixture that is itself illegal: if the filler
    # bullets carried an over-long line, the text compared against the old chain would be
    # `index_line_length_error`'s for a reason unrelated to the line each case appends, and
    # the comparison would pass while testing the wrong branch.
    assert ceiling.overlong_index_lines(tight) == [], (
        f"the fixture holds {len(ceiling.overlong_index_lines(tight))} over-long line(s)")
    assert ceiling.overlong_index_lines(roomy) == [], (
        f"the fixture holds {len(ceiling.overlong_index_lines(roomy))} over-long line(s)")
    assert [len(ln) for ln in ceiling.overlong_index_lines(tight + long_line)] == [
        len(long_line) - 1], "exactly one line is over the cap: the line the case appended"

    for label, (on_disk, prospective) in cases.items():
        (root / "MEMORY.md").write_text(on_disk, encoding="utf-8")
        got = ceiling.memory_write_error(root / "MEMORY.md", prospective)
        want = _chain_before_2838(root / "MEMORY.md", prospective)
        assert (got is None) == (want is None), (
            f"{label}: the new branch moved a verdict — now "
            f"{'refused' if got else 'allowed'}, before #2838 "
            f"{'refused' if want else 'allowed'}")
        if label == "no room, a new entry":
            # The one case the clause is about: refused before, refused after, and the TEXT
            # is the only thing that differs — asserted as different, not skipped.
            assert got != want and got is not None and want is not None, (
                "the no-room case is the whole change: its verdict must be unchanged and "
                "its text must not be")
            assert "ONE typed line" in want and "ONE typed line" not in got, (
                "the old text ordered a line, the new one must not")
        else:
            assert got == want, f"{label}: text moved on a case the clause does not cover"

    # The case the first gate read reasoned about rather than ran: a long line over an index
    # with no room. `index_line_length_error` was the FIRST term of the old chain, so it was
    # already the answer there — the new term is second in the chain and never reached, which
    # is what makes the equality below possible at all. A write cannot be "newly refused" by a
    # term that is never consulted on it.
    long_prospective = tight + long_line
    (root / "MEMORY.md").write_text(tight, encoding="utf-8")
    got_long = ceiling.memory_write_error(root / "MEMORY.md", long_prospective)
    assert got_long == _chain_before_2838(root / "MEMORY.md", long_prospective) == (
        ceiling.index_line_length_error("MEMORY.md", long_prospective)), (
        "the long-line refusal over a no-room index is not the term that answered before "
        "#2838 either")
    # Called on its own the new term does answer this text — it knows only the tight limit,
    # not the chain's priority. What keeps it off a long-line write is the chain: the first
    # term refuses and `or` short-circuits, which is the equality above. So the term is
    # reachable only for texts that hold no over-long line.
    assert ceiling.index_no_room_error(root / "MEMORY.md", "MEMORY.md",
                                       long_prospective) is not None

    (root / "MEMORY.md").write_text(tight, encoding="utf-8")
    assert ceiling.memory_write_error(root / "MEMORY.md", tight[:2000]) is None, (
        "the shrink escape at `memory_ceiling.py:333-337` stopped admitting a shrinking "
        "over-limit index")
    no_room = ceiling.memory_write_error(root / "MEMORY.md",
                                         tight + "- [project] new → topics/new-rule\n")
    assert "consolidation owed" in no_room, no_room
    assert "consolidation owed" in ceiling.index_tight_limit_error(
        "MEMORY.md", _grow_by(roomy, limit - len(roomy.encode("utf-8")) + 1)), (
        "the record the new branch copies has gone out of the refusal it re-texts")
    # `index_line_length_error` is the OTHER index refusal, and clause 4 asks that its text
    # be untouched. It has never carried the owed record — the record belongs to the
    # tight-limit pair — so it is pinned byte-for-byte against the shipped builder instead
    # of being asserted to contain a sentence that was never in it.
    (root / "MEMORY.md").write_text(roomy, encoding="utf-8")
    assert ceiling.memory_write_error(root / "MEMORY.md", roomy + long_line) == (
        ceiling.index_line_length_error("MEMORY.md", roomy + long_line)), (
        "the long-line refusal no longer answers as its own builder does")
    # The long-line refusal's own route, quoted as it is shipped: shortening the line IS an
    # addable edit, which is why clause 4 leaves this text alone. It says "keep one typed
    # line", not the tight refusal's "give the index ONE typed line" — the two sentences are
    # different advice and this assertion is where that difference is pinned.
    assert "keep one typed line" in ceiling.index_line_length_error("MEMORY.md", long_line)


def test_the_no_room_term_only_ever_narrows_the_tight_refusal(memories_root):
    """Clause 4 as a property, not a sample: the new term refines the tight refusal.

    Every text below is asked of both builders over the same file on disk, and the claim is
    one-directional: where `index_no_room_error` answers, `index_tight_limit_error` must
    already have been answering. That implication is why no write can become newly refused —
    the entry point's chain only ever reaches the new term on a text the old chain rejected,
    and the text it returns is a re-wording, so a pass can only lose the old advice, never
    add a refusal. The direction matters and is asserted as such: the converse fails by
    design, since a roomy index is over the limit on the PROSPECTIVE size and keeps the
    add-a-line route.

    The last three rows are the term's other exits: a non-index memory file has no tight
    limit to be out of room under, a topic file is measured by a different bound, and a file
    that does not exist yet has the whole limit free.
    """
    session, root = memories_root
    limit = ceiling.tight_limit("MEMORY.md")
    roomy = _index_linking("topics/a-topic", room=512, limit=limit)
    tight = _index_linking("topics/a-topic", room=20, limit=limit)
    (root / "MEMORY.md").write_text(tight, encoding="utf-8")

    texts = {
        "empty file": "",
        "well under the limit": tight[:2000],
        "roomy index, one line more": roomy + "- [project] fits → topics/a-topic\n",
        "roomy index, past the limit": _grow_by(roomy, 665),
        "no-room index, one line more": tight + "- [project] new → topics/new-rule\n",
        "no-room index, a long line": tight + f"- [project] {'k' * 400} → topics/a-topic\n",
        "past the ceiling": _grow_by(roomy, 6000),
    }
    for label, text in texts.items():
        narrowed = ceiling.index_no_room_error(root / "MEMORY.md", "MEMORY.md", text)
        tight_refusal = ceiling.index_tight_limit_error("MEMORY.md", text)
        assert narrowed is None or tight_refusal is not None, (
            f"{label}: the no-room term refused a text the tight limit allowed, which is a "
            "new refusal and the one thing this clause forbids")

    assert ceiling.index_no_room_error(root / "USER.md", "USER.md", "x" * 40_000) is None, (
        "USER.md has no tight limit, so it has no no-room case")
    assert ceiling.index_no_room_error(root / "memory" / "a-topic.md",
                                      "topics/a-topic.md", "x" * 40_000) is None
    assert ceiling.index_no_room_error(root / "no-such-file.md", "MEMORY.md",
                                       tight + "- [project] new → topics/new-rule\n") is None, (
        "a file that does not exist yet has the whole tight limit free to grow into")


def test_the_three_audit_citations_still_land_on_their_symbols():
    """Clause 5: the lines `lloyd/reviews/2026-09-14-user-md-audit.md` cites by number.

    `tests/test_prompt_surface_budget.py` reads that note from the vault and resolves 114, 162
    and 262 against this file; this node is the copy that fails inside the suite this round
    changes, and it pins symbols rather than prose — 114 binds `MEMORY_TIGHTNESS`, 162 the
    `tight_limit` def, 262 the `memory_write_error` def. The last assertion is why #2838's
    section was appended at the bottom of the module instead of sitting beside the refusal it
    re-texts: anything inserted above line 262 moves a cited line, which is the hazard the
    citation block inside `memory_write_error` warns about.
    """
    src = (Path(__file__).resolve().parents[1] / "app" / "memory_ceiling.py").read_text()
    lines = src.split("\n")

    assert lines[114 - 1] == "MEMORY_TIGHTNESS = 0.80", lines[114 - 1]
    assert lines[162 - 1].startswith("def tight_limit("), lines[162 - 1]
    assert lines[262 - 1].startswith("def memory_write_error("), lines[262 - 1]
    assert src.index("def index_no_room_error(") > src.index("def memory_write_error("), (
        "the #2838 section moved above a cited line")


def test_the_line_floor_is_the_line_the_add_a_line_route_orders():
    """The 24-byte floor is the cheapest line the sentence it replaces could order.

    `TOPIC_LINE_MIN_BYTES` derives from the shortest entry type and the shortest legal slug, so
    it is pinned against both grammars that define them: `TYPED_ENTRY_RE` must match the line,
    `TOPIC_LINK_RE` must find its pointer, and `topic_slug` must accept its one-character slug.
    A widened `ENTRY_TYPES` or a changed pointer would otherwise move the constant while the
    branch went on comparing against a line no writer can produce. And `20 < 24` is the number
    clause 1's fixture is built on, so the comparison that makes it no-room is asserted too.
    """
    shortest = "- [user] x → topics/a\n"

    assert ceiling.SHORTEST_TOPIC_LINE == shortest
    assert ceiling.TOPIC_LINE_MIN_BYTES == len(shortest.encode("utf-8")) == 24
    assert ps.TYPED_ENTRY_RE.match(shortest), shortest
    assert ceiling.TOPIC_LINK_RE.search(shortest), shortest
    assert ceiling.topic_slug("topics/a") == "a"
    assert ceiling.linked_topic_files(shortest) == ["topics/a"]
    assert 20 < ceiling.TOPIC_LINE_MIN_BYTES
