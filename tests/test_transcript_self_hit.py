"""#1511: a session transcript that echoes the prompt never reaches `<vault-context>`.

Hermetic: qmd, the skill/fact/session/backlog workers and the IDE mirror are all
monkeypatched, and the note reader points at a tmp_path collection root.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prefetch  # noqa: E402
from agent_mcp import transcript_self_hit as tsh  # noqa: E402

PROBE = ("E2E harness check (Claude Code verifying today's landings; no action needed "
         "beyond this). Read /home/alansrobotlab/lloyd/app/harness/client.py in full and "
         "tell me in one sentence what stream_chat does.")
PRIOR_NOTE = (
    "# 20260924_221452_ivb794\n# 2026-09-24T22:14:52\n\n"
    f"user: {PROBE}\n"
    "  → [OK] [full result on disk]      1\t\"\"\"vLLM SSE chat client.\n"
    "lloyd: `stream_chat` is the agent loop's single streaming send site.\n"
)


@pytest.fixture
def notes(tmp_path, monkeypatch):
    root = tmp_path / "sessions"
    (root / "2026-09-24").mkdir(parents=True)
    (root / "2026-09-24" / "20260924_221452_ivb794.md").write_text(PRIOR_NOTE)
    monkeypatch.setattr(tsh, "_sessions_root", lambda: root)
    return root


def _hit(file, snippet, score=1.0, title="t"):
    return {"file": file, "snippet": snippet, "score": score, "title": title}


# ── the rule ─────────────────────────────────────────────────────────────────

def test_prior_transcript_of_the_same_prompt_is_a_verbatim_self_hit(notes):
    # The snippet the daemon returned opens mid-answer, so the verdict has to
    # come from the note's own user turn, read off disk.
    h = _hit("qmd://sessions/2026-09-24/20260924_221452_ivb794.md",
             "lloyd: `stream_chat` is the agent loop's single streaming send site.")
    assert tsh.self_hit_reason(PROBE, h) == "verbatim"


def test_the_snippet_alone_convicts_when_the_note_cannot_be_read(monkeypatch):
    monkeypatch.setattr(tsh, "_sessions_root", lambda: None)
    h = _hit("qmd://sessions/2026-09-24/x.md", f"user: {PROBE}")
    assert tsh.self_hit_reason(PROBE, h) == "verbatim"
    assert tsh.self_hit_reason(PROBE, h, read_note=False) == "verbatim"


def test_a_rewrapped_prompt_is_still_an_echo(notes):
    rewrapped = PROBE.replace(" ", "\n  ").upper()
    h = _hit("qmd://sessions/2026-09-24/20260924_221452_ivb794.md", "")
    assert tsh.self_hit_reason(rewrapped, h) == "verbatim"


def test_own_session_export_is_a_self_hit_whatever_its_text():
    h = _hit("qmd://sessions/2026-09-25/20260925_082940_iv30d7.md", "unrelated words")
    assert tsh.self_hit_reason("hi", h, session_id="20260925_082940_iv30d7") == "own_session"


def test_a_related_session_that_does_not_repeat_the_prompt_stays(notes):
    h = _hit("qmd://sessions/2026-09-24/other.md",
             "user: can you explain how the harness streams chat completions from vllm "
             "and where the stream_chat client lives")
    assert tsh.self_hit_reason(PROBE, h, read_note=False) is None


def test_a_template_reused_with_a_new_argument_is_not_an_echo():
    ask = "pull the youtube transcript and give me the highlights https://www.youtube.com/watch?v=CgsDEFaGLJ0"
    prior = "user: pull the youtube transcript and give me the highlights https://www.youtube.com/watch?v=wgZQGxdjbGM"
    h = _hit("qmd://sessions/2026-09-23/a.md", prior)
    assert tsh.self_hit_reason(ask, h, read_note=False) is None
    same = _hit("qmd://sessions/2026-09-23/b.md", "user: " + ask)
    assert tsh.self_hit_reason(ask, same, read_note=False) == "verbatim"


def test_a_short_prompt_is_never_judged_verbatim():
    # "what did we decide about qmd" legitimately recalls where it was asked.
    q = "what did we decide"
    h = _hit("qmd://sessions/2026-09-23/a.md", "user: " + q)
    assert len(q.split()) < tsh.SELF_HIT_MIN_WORDS
    assert tsh.self_hit_reason(q, h, read_note=False) is None


def test_only_the_sessions_collection_is_judged():
    # A backlog item quoting the prompt is a document about it, not its echo.
    h = _hit("qmd://backlog/1511-prior-session.md", f"user: {PROBE}")
    assert tsh.self_hit_reason(PROBE, h, session_id="x") is None


def test_a_path_escaping_the_collection_root_is_not_read(notes):
    h = _hit("qmd://sessions/../../etc/passwd", "")
    assert tsh._note_user_turns("../../etc/passwd") is None
    assert tsh.self_hit_reason(PROBE, h) is None


def test_the_check_never_raises():
    assert tsh.self_hit_reason(PROBE, {"file": object()}) is None
    assert tsh.drop_self_hits(PROBE, None) == ([], [])


# ── prefetch end to end ──────────────────────────────────────────────────────

@pytest.fixture
def quiet_workers(monkeypatch):
    monkeypatch.setattr(prefetch, "_search_skills", lambda q: [])
    monkeypatch.setattr(prefetch, "_search_facts", lambda t: [])
    monkeypatch.setattr(prefetch, "_search_recent_sessions", lambda t: [])
    monkeypatch.setattr(prefetch, "_search_backlog_refs", lambda t: [])
    monkeypatch.setattr(prefetch, "_format_ide_state", lambda: "")
    monkeypatch.setattr(prefetch, "PREFETCH_BUDGET_MS", 300)


def _vault_with_echo(query, focus=None, legs=("lex", "vec"), **kw):
    rows = [_hit("qmd://knowledge/client-notes.md", "stream_chat notes", 0.8, "Client notes")]
    if legs != ("lex",):
        rows.insert(0, _hit("qmd://sessions/2026-09-24/20260924_221452_ivb794.md",
                            f"user: {PROBE}\nlloyd: last night's answer", 1.0,
                            "20260924_221452_ivb794"))
    return rows


def test_prefetch_drops_the_echo_and_keeps_the_rest(quiet_workers, notes, monkeypatch):
    monkeypatch.setattr(prefetch, "_search_vault", _vault_with_echo)
    monkeypatch.setattr(prefetch, "exclude_self_transcripts_enabled", lambda: True)
    out = prefetch.prefetch_context(PROBE, session_id=None, plan_mode=False)
    assert "last night's answer" not in out
    assert "20260924_221452_ivb794" not in out
    assert "Client notes" in out


def test_the_switch_off_restores_the_old_block(quiet_workers, notes, monkeypatch):
    monkeypatch.setattr(prefetch, "_search_vault", _vault_with_echo)
    monkeypatch.setattr(prefetch, "exclude_self_transcripts_enabled", lambda: False)
    out = prefetch.prefetch_context(PROBE, session_id=None, plan_mode=False)
    assert "last night's answer" in out


def test_a_session_never_sees_its_own_export(quiet_workers, monkeypatch):
    sid = "20260925_101010_ivabcd"

    def _vault(query, focus=None, legs=("lex", "vec"), **kw):
        return [_hit(f"qmd://sessions/2026-09-25/{sid}.md", "our own chat so far", 1.0, sid),
                _hit("qmd://knowledge/k.md", "a real note", 0.7, "Real")]

    monkeypatch.setattr(prefetch, "_search_vault", _vault)
    monkeypatch.setattr(prefetch, "exclude_self_transcripts_enabled", lambda: True)
    out = prefetch.prefetch_context("tell me more about that servo tuning", session_id=sid,
                                    plan_mode=False)
    assert "our own chat so far" not in out
    assert "a real note" in out


def test_the_switch_reads_config(monkeypatch):
    from app import config as config_mod
    monkeypatch.setitem(config_mod.CONFIG, "prefetch", {"exclude_self_transcripts": False})
    assert prefetch.exclude_self_transcripts_enabled() is False
    monkeypatch.setitem(config_mod.CONFIG, "prefetch", {})
    assert prefetch.exclude_self_transcripts_enabled() is True
