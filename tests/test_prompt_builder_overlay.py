"""Overlay resolution and prompt loading — the premise of every variant score.

Why this file exists
--------------------
The whole autoresearch loop rests on one sentence in `prompt_builder`: the
overlay dir supplies a variant's files, and *anything it doesn't supply falls
through to the canonical vault*. The baseline variant is an empty directory, so
"baseline" means "canonical prompts by fallthrough" — and baseline is what every
variant is compared against before its prompts get written into the live vault.

Before this file, nothing asserted that fallthrough works. `build_system_prompt`
was actively monkeypatched *away* in `test_autonomy_scheduler.py:32`. If
`_resolve_overlay` regressed to returning None, the bench would grade a stripped
prompt as the baseline, variants would win against a phantom, and a promotion
would fire — with every test in the repo still green.

Isolation: `_CANON_SOUL_PATH` / `_CANON_MEMORIES_DIR` are module-level
constants pointing at the real vault. They are patched to a fake vault here, so
no test in this module reads or writes `~/obsidian/`.
"""
from __future__ import annotations

import pytest

import prompt_builder as pb


@pytest.fixture
def fake_vault(tmp_path, monkeypatch):
    """A stand-in canonical vault, patched in place."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "SOUL.md").write_text(
        "---\nfront: matter\n---\nCANONICAL SOUL BODY\n", encoding="utf-8"
    )
    (vault / "MEMORY.md").write_text("CANONICAL MEMORY BODY", encoding="utf-8")
    (vault / "USER.md").write_text("CANONICAL USER BODY", encoding="utf-8")
    skills_vault = tmp_path / "skills-vault"
    skills_repo = tmp_path / "skills-repo"
    skills_vault.mkdir(); skills_repo.mkdir()
    monkeypatch.setattr(pb, "_CANON_SOUL_PATH", vault / "SOUL.md")
    monkeypatch.setattr(pb, "_CANON_MEMORIES_DIR", vault)
    monkeypatch.setattr(pb, "_CANON_SKILLS_DIRS", [skills_vault, skills_repo])
    monkeypatch.delenv("LLOYD_OVERLAY_DIR", raising=False)
    return vault


# ── _resolve_overlay ─────────────────────────────────────────────────────────

def test_no_overlay_means_none(fake_vault):
    assert pb._resolve_overlay(None) is None


def test_nonexistent_overlay_path_is_none(fake_vault, tmp_path):
    assert pb._resolve_overlay(tmp_path / "does-not-exist") is None


def test_existing_overlay_path_is_returned(fake_vault, tmp_path):
    d = tmp_path / "overlay"; d.mkdir()
    assert pb._resolve_overlay(d) == d


def test_overlay_dir_env_var_is_the_fallback(fake_vault, tmp_path, monkeypatch):
    d = tmp_path / "env-overlay"; d.mkdir()
    monkeypatch.setenv("LLOYD_OVERLAY_DIR", str(d))
    assert pb._resolve_overlay(None) == d


def test_explicit_argument_beats_the_env_var(fake_vault, tmp_path, monkeypatch):
    env_dir = tmp_path / "env"; env_dir.mkdir()
    arg_dir = tmp_path / "arg"; arg_dir.mkdir()
    monkeypatch.setenv("LLOYD_OVERLAY_DIR", str(env_dir))
    assert pb._resolve_overlay(arg_dir) == arg_dir


def test_env_var_pointing_at_nothing_falls_back_to_canonical(fake_vault, monkeypatch):
    """A stale env var must degrade to canonical, not to an empty prompt."""
    monkeypatch.setenv("LLOYD_OVERLAY_DIR", "/nonexistent/overlay-dir")
    assert pb._resolve_overlay(None) is None


def test_string_overlay_path_is_accepted_and_expanded(fake_vault, tmp_path, monkeypatch):
    d = tmp_path / "as-string"; d.mkdir()
    assert pb._resolve_overlay(str(d)) == d
    monkeypatch.setenv("HOME", str(tmp_path))
    assert pb._resolve_overlay("~") == tmp_path          # expanduser applied


# ── _load_soul ───────────────────────────────────────────────────────────────

def test_soul_falls_through_to_canonical_when_overlay_lacks_it(fake_vault, tmp_path):
    overlay = tmp_path / "overlay"; overlay.mkdir()
    assert pb._load_soul(overlay) == "CANONICAL SOUL BODY"


def test_soul_canonical_strips_yaml_frontmatter(fake_vault):
    assert pb._load_soul(None) == "CANONICAL SOUL BODY"


def test_soul_overlay_wins_over_canonical(fake_vault, tmp_path):
    overlay = tmp_path / "overlay"; overlay.mkdir()
    (overlay / "SOUL.md").write_text("VARIANT SOUL", encoding="utf-8")
    assert pb._load_soul(overlay) == "VARIANT SOUL"


def test_soul_overlay_frontmatter_is_stripped_too(fake_vault, tmp_path):
    overlay = tmp_path / "overlay"; overlay.mkdir()
    (overlay / "SOUL.md").write_text("---\na: b\n---\nVARIANT BODY\n", encoding="utf-8")
    assert pb._load_soul(overlay) == "VARIANT BODY"


def test_soul_returns_none_when_neither_source_exists(fake_vault, tmp_path, monkeypatch):
    monkeypatch.setattr(pb, "_CANON_SOUL_PATH", tmp_path / "absent.md")
    assert pb._load_soul(None) is None


def test_soul_whitespace_only_file_reads_as_none(fake_vault, tmp_path):
    overlay = tmp_path / "overlay"; overlay.mkdir()
    (overlay / "SOUL.md").write_text("   \n  ", encoding="utf-8")
    assert pb._load_soul(overlay) is None


def test_empty_frontmatter_stripping_does_not_erase_a_body(fake_vault, tmp_path):
    """A file that only *looks* like it has frontmatter must not come back empty."""
    overlay = tmp_path / "overlay"; overlay.mkdir()
    (overlay / "SOUL.md").write_text("no closing delimiter here\n", encoding="utf-8")
    assert pb._load_soul(overlay) == "no closing delimiter here"


# ── _load_memories ───────────────────────────────────────────────────────────

def test_memories_fall_through_to_canonical(fake_vault, tmp_path):
    overlay = tmp_path / "overlay"; overlay.mkdir()
    out = pb._load_memories(overlay)
    assert "CANONICAL MEMORY BODY" in out and "CANONICAL USER BODY" in out


def test_memories_label_each_file(fake_vault):
    out = pb._load_memories(None)
    assert "## MEMORY.md" in out and "## USER.md" in out


def test_memories_overlay_is_per_file_not_per_directory(fake_vault, tmp_path):
    """Supplying MEMORY.md must not suppress the canonical USER.md.

    This per-file granularity is the property that makes a single-surface
    variant (the common case — `_parse_single_variant` keeps one file) score
    the change it actually made.
    """
    overlay = tmp_path / "overlay"; overlay.mkdir()
    (overlay / "MEMORY.md").write_text("VARIANT MEMORY", encoding="utf-8")
    out = pb._load_memories(overlay)
    assert "VARIANT MEMORY" in out
    assert "CANONICAL USER BODY" in out


def test_memories_skip_an_empty_overlay_file_without_suppressing_canonical(fake_vault, tmp_path):
    """Characterized asymmetry: an empty overlay MEMORY.md yields neither the
    variant nor the canonical content — the section just disappears."""
    overlay = tmp_path / "overlay"; overlay.mkdir()
    (overlay / "MEMORY.md").write_text("   ", encoding="utf-8")
    out = pb._load_memories(overlay)
    assert "CANONICAL MEMORY BODY" not in out
    assert "## MEMORY.md" not in out
    assert "CANONICAL USER BODY" in out


def test_memories_none_when_nothing_exists(fake_vault, tmp_path, monkeypatch):
    monkeypatch.setattr(pb, "_CANON_MEMORIES_DIR", tmp_path / "nowhere")
    assert pb._load_memories(None) is None


# ── quarantine vocabulary ────────────────────────────────────────────────────

def _skill(tmp_path, name, status):
    d = tmp_path / "skills" / name
    d.mkdir(parents=True)
    f = d / "SKILL.md"
    f.write_text(f"---\nname: {name}\nstatus: {status}\n---\nbody\n", encoding="utf-8")
    return f


@pytest.mark.parametrize("status", ["inactive", "archived", "disabled", "retired", "quarantined"])
def test_quarantined_skills_are_excluded_from_the_index(fake_vault, tmp_path, status):
    f = _skill(tmp_path, "s1", status)
    assert pb._is_quarantined_skill(f) is True


def test_active_skill_is_included(fake_vault, tmp_path):
    assert pb._is_quarantined_skill(_skill(tmp_path, "s2", "active")) is False


def test_quarantine_vocabulary_matches_the_mcp_skills_module(fake_vault):
    """The advertised index and the readable set must not drift — a skill the
    index promises but `skills_read` declines is a broken promise to the model."""
    from agent_mcp.skills import _QUARANTINE_STATUSES

    assert pb._QUARANTINE_STATUSES == _QUARANTINE_STATUSES


def test_quarantine_check_survives_an_unreadable_file(fake_vault, tmp_path):
    assert pb._is_quarantined_skill(tmp_path / "missing" / "SKILL.md") is False


# ── build_system_prompt: the composition the loop actually uses ──────────────

def test_build_with_no_overlay_uses_canonical_surfaces(fake_vault):
    out = pb.build_system_prompt(include_skills_index=False)
    assert "CANONICAL SOUL BODY" in out
    assert "CANONICAL MEMORY BODY" in out


def test_build_with_a_variant_soul_keeps_canonical_memories(fake_vault, tmp_path):
    """The exact shape of a real promotion: one file overridden, the rest
    falling through. If this ever stops holding, every delta measured by the
    bench is meaningless."""
    overlay = tmp_path / "overlay"; overlay.mkdir()
    (overlay / "SOUL.md").write_text("VARIANT SOUL", encoding="utf-8")
    out = pb.build_system_prompt(include_skills_index=False, overlay_dir=overlay)
    assert "VARIANT SOUL" in out and "CANONICAL SOUL BODY" not in out
    assert "CANONICAL MEMORY BODY" in out and "CANONICAL USER BODY" in out


def test_build_with_an_empty_overlay_equals_the_canonical_build(fake_vault, tmp_path):
    """`materialize_baseline()` produces an empty dir; the baseline score is only
    comparable to the live prompt if this holds."""
    overlay = tmp_path / "overlay"; overlay.mkdir()
    assert (pb.build_system_prompt(include_skills_index=False, overlay_dir=overlay)
            == pb.build_system_prompt(include_skills_index=False))


def test_build_never_renders_the_overlay_path_itself(fake_vault, tmp_path):
    overlay = tmp_path / "overlay"; overlay.mkdir()
    (overlay / "SOUL.md").write_text("VARIANT SOUL", encoding="utf-8")
    out = pb.build_system_prompt(include_skills_index=False, overlay_dir=overlay)
    assert str(overlay) not in out


# ── the pure blocks the loop's variants can perturb ──────────────────────────

def test_todos_block_absent_when_empty(fake_vault):
    assert pb._format_active_todos(None) is None
    assert pb._format_active_todos([]) is None


def test_todos_block_renders_status_and_content(fake_vault):
    out = pb._format_active_todos([
        {"status": "completed", "content": "Do A"},
        {"status": "in_progress", "content": "Do B"},
    ])
    assert out.startswith("<active_todos>")
    assert "[completed] Do A" in out and "[in_progress] Do B" in out


def test_todos_block_skips_entries_with_no_content(fake_vault):
    out = pb._format_active_todos([{"status": "pending", "content": "  "}])
    assert out is None                      # nothing left to render


def test_todos_block_survives_a_missing_status(fake_vault):
    out = pb._format_active_todos([{"content": "Do C"}])
    assert "[?] Do C" in out


def test_goal_block_reports_the_attempt_count(fake_vault):
    out = pb._format_goal_block({"text": "make the suite green", "attempts": 3})
    assert "make the suite green" in out
    assert "Attempts so far: 3." in out
    assert out.startswith("<goal>") and out.rstrip().endswith("</goal>")


def test_goal_block_omits_the_attempt_line_at_zero_attempts(fake_vault):
    """Cache-friendly: a fresh goal must not carry a noise line, or every turn
    rewrites the block."""
    out = pb._format_goal_block({"text": "goal text"})
    assert "Attempts so far" not in out


def test_achieved_goal_renders_the_terminal_block(fake_vault):
    out = pb._format_goal_block({
        "text": "ship it", "achieved_at": "2026-09-05T20:00:00Z", "attempts": 4,
    })
    assert out.startswith("<goal achieved>")
    assert "ship it" in out and "2026-09-05T20:00:00Z" in out
    # The achieved branch never leaks the attempt counter.
    assert "Attempts so far" not in out


def test_goal_block_ignores_fields_it_does_not_render(fake_vault):
    """Characterized: `last_reason` is surfaced by the inner voice on the
    follow-up turn, not baked into the prompt block — so a changing reason does
    not invalidate the prompt cache every turn."""
    with_reason = pb._format_goal_block({"text": "same", "last_reason": "why it failed"})
    without = pb._format_goal_block({"text": "same"})
    assert with_reason == without
    assert "why it failed" not in with_reason


def test_goal_block_absent_without_text(fake_vault):
    assert pb._format_goal_block(None) is None
    assert pb._format_goal_block({"text": ""}) is None
    assert pb._format_goal_block({"text": "   "}) is None
    assert pb._format_goal_block({}) is None


def test_goal_block_strips_surrounding_whitespace_from_the_text(fake_vault):
    out = pb._format_goal_block({"text": "\n  padded  \n"})
    assert "\n  padded  \n" not in out
    assert "padded" in out
