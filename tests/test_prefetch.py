"""Tests for prefetch.py + the #306 subliminal capture helpers.

Hermetic: every worker that would touch disk, qmd, or the facts store is
monkeypatched. Run:
  .venvs/lloyd/bin/python -m pytest tests/test_prefetch.py -q
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prefetch  # noqa: E402
from agent_mcp import session as session_mod  # noqa: E402
from agent_mcp import vault as vault_mod  # noqa: E402
from agent_mcp.skills import _score_skill, _skill_token_sets, _tokenize  # noqa: E402
from app.routers._messages_subliminal import (  # noqa: E402
    _classify_subliminal,
    _detect_subliminal_sources,
    _extract_subliminal_prefix,
)


# ── #306 capture helpers ──────────────────────────────────────────────────────

def test_extract_prefix_double_newline():
    text = "hello there world"
    pre = "<context>\n<facts>\n- x\n</facts>\n</context>"
    assert _extract_subliminal_prefix(pre + "\n\n" + text, text) == pre


def test_extract_prefix_nudge_without_context_single_newline():
    # A 20-turn nudge on a turn with no <context> block used to glue the
    # nudge to the user text with one "\n"; the extractor then swept the
    # user's own text into the subliminal entry.
    text = "hello there world"
    nudge = "<system-reminder>This session has 20 turns.</system-reminder>"
    assert _extract_subliminal_prefix(nudge + "\n" + text, text) == nudge
    assert _classify_subliminal(nudge) == "memory_nudge"


def test_extract_prefix_ambient_envelope_is_whole_text():
    text = "nightly job finished"
    env = f'<ambient priority="notable" source="cron" session_id="s">\n{text}\n</ambient>\n\nfooter'
    assert _extract_subliminal_prefix(env, text) == env
    assert _classify_subliminal(env) == "ambient_envelope"


def test_extract_prefix_no_injection():
    assert _extract_subliminal_prefix("same", "same") == ""


def test_detect_sources_includes_ide():
    prefix = "<context>\n<vault-context>\n- a\n</vault-context>\n<ide_state>\n  visible_file: x\n</ide_state>\n</context>"
    assert _detect_subliminal_sources(prefix) == ["vault", "ide"]


# ── Focus tracking ────────────────────────────────────────────────────────────

def test_focus_keywords_strip_trailing_punctuation():
    kws = prefetch._extract_focus_keywords("Look at the alfie servo configuration. Then config.yaml.")
    assert "servo" in kws
    assert "configuration" in kws
    assert "configuration." not in kws
    assert "config.yaml" in kws


def test_focus_update_is_thread_safe():
    focus = prefetch.SessionFocus()
    errors: list[BaseException] = []

    def hammer(word: str):
        try:
            for i in range(300):
                focus.update(f"{word}{i % 7} servo shoulder pid gains oscillation")
                focus.enrich_query("what about it")
        except BaseException as e:  # pragma: no cover - only on failure
            errors.append(e)

    threads = [threading.Thread(target=hammer, args=(w,)) for w in ("alpha", "beta", "gamma")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert focus.turn_count == 900


def test_focus_topic_attempt_is_recorded():
    focus = prefetch.SessionFocus()
    for _ in range(5):
        focus.update("servo shoulder pid tuning session")
    assert focus.needs_topic_extraction()
    focus.mark_topic_attempt()  # extraction ran but returned nothing
    assert not focus.needs_topic_extraction()


# ── Continuation detection (skill-hint suppression) ──────────────────────────

@pytest.mark.parametrize("text,is_cont", [
    ("ok", True), ("yes please", True), ("please continue", True), ("let's go", True),
    ("let's do it", True), ("sounds good, proceed", True), ("carry on", True),
    ("continue with the plan", True),
    ("please review the subliminal module", False),
    ("let's build a new feature for the vault", False),
    ("please add a test for the backlog index", False),
])
def test_continuation_regex(text, is_cont):
    assert bool(prefetch._CONTINUATION_RE.match(text)) == is_cont


# ── Backlog task-ref precision ────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("what's left on #294", {294}),
    ("302 is resolved", {302}),
    ("#300 and 300ms", {300}),
    ("PREFETCH_BUDGET_MS = 300", set()),
    ("the endpoint returned a 302 redirect", set()),
    ("HTTP 404 from the daemon", set()),
    ("port 8080 is taken", set()),
    ("took 250 ms and 3 sec", set()),
    ("status 500 on /query", set()),
    ("dated 20260421", set()),
    ("task 311 needs a look", {311}),
])
def test_task_ref_candidates(text, expected):
    assert prefetch._task_ref_candidates(text) == expected


# ── Skill scoring memo ────────────────────────────────────────────────────────

def test_score_skill_memoized_equals_fresh():
    skill = {
        "name": "system-health-check",
        "description": "Full systems check of services and disks",
        "tags": ["health", "supervisord"],
        "body": "Run supervisorctl status. Check disks. " * 50,
    }
    fresh = dict(skill)
    q = {"system", "health", "check", "disk"}
    s1 = _score_skill(skill, q)
    assert "_tok" in skill  # memoized on the dict
    s2 = _score_skill(skill, q)  # second call uses the memo
    s3 = _score_skill(fresh, q)  # separate dict, computed from scratch
    assert s1 == s2 == s3 > 0
    name_tokens, _, tag_tokens, _ = _skill_token_sets(skill)
    assert name_tokens == _tokenize("system health check")
    assert "health" in tag_tokens


def test_score_skill_tolerates_odd_metadata():
    skill = {"name": "x-y", "description": None, "tags": "single", "body": None}
    assert _score_skill(skill, {"single"}) == 1.5  # one tag hit


# ── Vault merge ───────────────────────────────────────────────────────────────

def test_merge_vault_results_dedup_and_cap():
    fresh = [{"file": "a.md", "title": "A", "score": 0.9, "snippet": "a"}]
    carried = [
        {"file": "a.md", "title": "A", "score": 0.95, "snippet": "a"},   # dup of fresh
        {"file": "b.md", "title": "B", "score": 0.7, "snippet": "b"},
        {"file": "c.md", "title": "C", "score": 0.6, "snippet": "c"},
        {"file": "d.md", "title": "D", "score": 0.55, "snippet": "d"},
        {"file": "e.md", "title": "E", "score": 0.52, "snippet": "e"},
        {"file": "f.md", "title": "F", "score": 0.51, "snippet": "f"},
    ]
    merged = prefetch._merge_vault_results(fresh, carried)
    files = [m["file"] for m in merged]
    assert files == ["a.md", "b.md", "c.md", "d.md", "e.md"]
    assert not merged[0].get("carried")
    assert all(m.get("carried") for m in merged[1:])


def test_merge_reserves_slots_for_carried_hits():
    fresh = [{"file": f"f{i}.md", "title": f"F{i}", "score": 0.9 - i * 0.05, "snippet": "x"} for i in range(5)]
    carried = [{"file": "sem1.md", "title": "S1", "score": 0.45, "snippet": "x"},
               {"file": "sem2.md", "title": "S2", "score": 0.40, "snippet": "x"},
               {"file": "sem3.md", "title": "S3", "score": 0.35, "snippet": "x"}]
    merged = prefetch._merge_vault_results(fresh, carried)
    files = [m["file"] for m in merged]
    assert len(files) == prefetch.VAULT_MAX_RESULTS
    assert "sem1.md" in files and "sem2.md" in files and "sem3.md" not in files
    assert files[:3] == ["f0.md", "f1.md", "f2.md"]  # still score-ordered


# ── #471: an injected vault hit names its file and says when it was cut ───────
#
# Every vault snippet reaching the agent is a fragment (`VAULT_SNIPPET_MAX`),
# and until now the rendered line named neither the file it came from nor that
# it had been cut — so a fragment was a dead end instead of a pointer to
# `Read`. These pin the two attributes, and pin that the marker is conditional.

def _render_vault(*hits):
    return prefetch._format_context([], [], vault_results=list(hits))


def test_vault_context_names_the_file_of_the_hit():
    out = _render_vault(
        {"file": "knowledge/meetings/march-5-transcript.md",
         "title": "March 5 Transcript", "score": 0.92,
         "snippet": "action items from the March 5 review"}
    )
    assert "<vault-context>" in out
    assert "knowledge/meetings/march-5-transcript.md" in out


def test_vault_context_renders_a_qmd_uri_vault_relative():
    # The qmd daemon spells hits `qmd://obsidian/<path>`. The whole use of the
    # path is that it goes straight into `Read`, so the URI scheme and the vault
    # root have to come off before it is rendered.
    out = _render_vault(
        {"file": "qmd://obsidian/projects/lloyd/march-5-transcript.md",
         "title": "March 5 Transcript", "score": 0.91,
         "snippet": "action items from the March 5 review"}
    )
    assert "projects/lloyd/march-5-transcript.md" in out
    assert "qmd://" not in out and "obsidian/" not in out


def test_vault_context_names_the_file_on_a_carried_hit():
    out = _render_vault(
        {"file": "knowledge/meetings/march-5-transcript.md",
         "title": "March 5 Transcript", "score": 0.7, "snippet": "hit",
         "carried": True}
    )
    assert "semantic hit from the previous turn's query" in out
    assert "knowledge/meetings/march-5-transcript.md" in out


def test_vault_context_hit_without_a_file_still_renders():
    # An unavailable path must cost the path attribute, never the hit.
    out = _render_vault(
        {"file": "", "title": "No Path Hit", "score": 0.8,
         "snippet": "text from a hit whose path is unavailable"}
    )
    assert "No Path Hit" in out
    assert "text from a hit whose path is unavailable" in out
    # Assert on the attribute window of the rendered entry, not on the whole
    # block: "no empty path attribute" is a property of the entry, and a
    # whole-block substring would also be satisfied by a snippet that happens
    # not to contain the word.
    entry = next(ln for ln in out.splitlines() if ln.startswith("- **"))
    assert entry.startswith("- **No Path Hit** (score: 0.80): ")
    assert "file:" not in entry.split("): ", 1)[0]


_MARCH = "knowledge/meetings/march-5-transcript.md"


def _fake_daemon(snippet_len: int, file: str = f"qmd://{_MARCH}"):
    """A stand-in for the qmd daemon's reply list, in the daemon's own shape:
    `file` spelled as a `qmd://obsidian/...` URI, snippet already hunk-marked.
    """
    def _search(query, limit, collections, **kw):
        return [{"file": file, "title": "March 5 Transcript", "score": 0.9,
                 "snippet": "x" * snippet_len}]
    return _search


# >= VAULT_MIN_QUERY_LEN, and the phrasing an item-#471 acceptance probe uses.
_PROBE_QUERY = "give me the full march 5 transcript from the vault"


def test_snippet_cut_by_the_cap_is_flagged_and_marked(monkeypatch):
    monkeypatch.setattr(prefetch, "_qmd_daemon_search",
                        _fake_daemon(prefetch.VAULT_SNIPPET_MAX + 400))
    hits = prefetch._search_vault(_PROBE_QUERY)
    assert len(hits) == 1
    assert len(hits[0]["snippet"]) == prefetch.VAULT_SNIPPET_MAX
    assert hits[0]["truncated"] is True
    out = _render_vault(*hits)
    assert "[... truncated]" in out
    assert "knowledge/meetings/march-5-transcript.md" in out


def test_snippet_under_the_cap_renders_no_marker(monkeypatch):
    monkeypatch.setattr(prefetch, "_qmd_daemon_search", _fake_daemon(40))
    hits = prefetch._search_vault(_PROBE_QUERY)
    assert len(hits) == 1
    assert hits[0]["truncated"] is False
    out = _render_vault(*hits)
    assert "[... truncated]" not in out


def test_injected_block_carries_path_and_marker_end_to_end(quiet_workers, monkeypatch):
    # Whole path: qmd row -> `_search_vault` (where the cap lives) -> merge ->
    # renderer -> the string handed to the model. Patching the *daemon*, not
    # `_search_vault`, is what makes this cross the real seam: the hit dict, the
    # capped snippet and the `truncated` flag are all produced by the code.
    monkeypatch.setattr(prefetch, "_qmd_daemon_search",
                        _fake_daemon(prefetch.VAULT_SNIPPET_MAX + 300))
    # `quiet_workers` drops the budget to 120 ms; this change lengthens the
    # injected payload, so pin it under the budget that actually ships.
    monkeypatch.setattr(prefetch, "PREFETCH_BUDGET_MS", 300)
    monkeypatch.setattr(prefetch, "_qmd_daemon_search",
                        _fake_daemon(prefetch.VAULT_SNIPPET_MAX + 300))
    t0 = time.monotonic()
    out = prefetch.prefetch_context(_PROBE_QUERY, session_id="test-471-path", plan_mode=False)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.30, f"injection with path+marker took {elapsed:.3f}s past the 300ms budget"
    assert "<vault-context>" in out
    assert f"file: {_MARCH}" in out
    assert "[... truncated]" in out
    # the injected text still ends with the user's own message
    assert out.endswith("\n\n" + _PROBE_QUERY)


def test_daemon_row_reaches_the_block_unchanged_in_its_fields(monkeypatch):
    # The process boundary itself: `_qmd_post` is the loopback HTTP call to the
    # qmd daemon, and `file`/`snippet` are whatever it puts on the wire. Every
    # other test here stubs one level above, which would pass with the wire
    # spellings (`qmd://…` URIs, hunk-marked snippet, daemon-side line numbers)
    # unhandled. This one runs the real `_qmd_daemon_search` over a canned reply.
    def fake_post(payload):
        return [{"file": f"qmd://{_MARCH}", "title": "March 5 Transcript",
                 "score": 0.92,
                 "snippet": "@@ -117,4 @@ (116 before, 365 after)\n" + "z" * 900}]

    monkeypatch.setattr(vault_mod, "_qmd_post", fake_post)
    hits = prefetch._search_vault(_PROBE_QUERY)
    assert len(hits) == 1
    assert hits[0]["file"] == f"qmd://{_MARCH}"  # URI form survives to render
    assert hits[0]["truncated"] is True
    out = _render_vault(*hits)
    assert f"file: {_MARCH}" in out                       # stripped at render
    assert "qmd://" not in out
    assert "[... truncated]" in out


def test_wire_spelling_with_the_vault_root_is_stripped_too(monkeypatch):
    # The other spelling the daemon uses: a collection rooted at the vault, so
    # the row arrives `qmd://obsidian/<path>`. Without both prefixes off, the
    # path handed to `Read` would not resolve. Named collections, not a global
    # scan, is the shape prefetch actually sends (global_scan is False for
    # VAULT_COLLECTIONS), and that is the path that keeps the raw spelling.
    def fake_post(payload):
        return [{"file": f"qmd://obsidian/{_MARCH}", "title": "March 5 Transcript",
                 "score": 0.9, "snippet": "w" * 700}]

    monkeypatch.setattr(vault_mod, "_qmd_post", fake_post)
    hits = prefetch._search_vault(_PROBE_QUERY)
    assert len(hits) == 1
    out = _render_vault(*hits)
    assert f"file: {_MARCH}" in out
    assert "obsidian/" not in out and "qmd://" not in out
    assert "[... truncated]" in out


# ── #994: qmd's diff formatting comes off before the cap ────────────────────
#
# The daemon hands back a snippet as a diff hunk. vault_search stripped the
# header and the `NN:` prefixes; the live prefetch injection did not, so ~16%
# of every injected snippet was formatting. One helper strips for both now,
# and the header's start line survives as a `line:` attribute.

_HUNK = "@@ -39,4 @@ (38 before, 11 after)"
_HUNKED = (f"40: {_HUNK}\n41: \n42: The assistant extracted the auto-generated "
           "transcript\n43: and filed the action items.")


def _hunk_daemon(snippet: str):
    def _search(query, limit, collections, **kw):
        return [{"file": f"qmd://{_MARCH}", "title": "March 5 Transcript",
                 "score": 0.9, "snippet": snippet}]
    return _search


def _entry(out: str) -> str:
    return next(ln for ln in out.split("<vault-context>\n", 1)[1].split("\n- **")
                if "March 5" in ln)


def test_injected_entry_carries_no_hunk_header(monkeypatch):
    monkeypatch.setattr(prefetch, "_qmd_daemon_search", _hunk_daemon(_HUNKED))
    out = _render_vault(*prefetch._search_vault(_PROBE_QUERY))
    assert "@@" not in out.split("<vault-context>", 1)[1]


def test_injected_entry_carries_no_line_prefixes_and_loses_no_content(monkeypatch):
    monkeypatch.setattr(prefetch, "_qmd_daemon_search", _hunk_daemon(_HUNKED))
    hits = prefetch._search_vault(_PROBE_QUERY)
    entry = _entry(_render_vault(*hits))
    assert not re.search(r"\b\d{2,4}: ", entry.split("): ", 1)[1])
    # Stripped, not truncated: every non-metadata character is still there.
    content = re.sub(r"@@[^@]*@@ *(?:\([^)]*\))?", "", _HUNKED)
    content = re.sub(r"^\d+: ?", "", content, flags=re.MULTILINE)
    assert "".join(content.split()) in "".join(entry.split())
    assert hits[0]["truncated"] is False


def test_start_line_is_an_attribute_exactly_when_a_header_came_in(monkeypatch):
    monkeypatch.setattr(prefetch, "_qmd_daemon_search", _hunk_daemon(_HUNKED))
    hits = prefetch._search_vault(_PROBE_QUERY)
    assert hits[0]["line"] == 39
    head = _entry(_render_vault(*hits)).split("): ", 1)[0]
    assert f"file: {_MARCH}, line: 39" in head

    monkeypatch.setattr(prefetch, "_qmd_daemon_search", _hunk_daemon("plain text hit"))
    hits = prefetch._search_vault(_PROBE_QUERY)
    assert hits[0]["line"] is None
    assert "line:" not in _entry(_render_vault(*hits)).split("): ", 1)[0]


def test_strip_runs_before_the_cap(monkeypatch):
    body = "\n".join(f"{n}: " + "q" * 60 for n in range(40, 60))
    monkeypatch.setattr(prefetch, "_qmd_daemon_search",
                        _hunk_daemon(f"{_HUNK}\n{body}"))
    hits = prefetch._search_vault(_PROBE_QUERY)
    assert len(hits[0]["snippet"]) == prefetch.VAULT_SNIPPET_MAX
    assert hits[0]["truncated"] is True
    assert "@@" not in hits[0]["snippet"]
    assert not re.search(r"(?m)^\d+:", hits[0]["snippet"])
    assert "[... truncated]" in _render_vault(*hits)


# qmd's own shape: header first, then the lines (what `extractSnippet` emits).
_HEADER_FIRST = (f"{_HUNK}\n40: The assistant extracted the auto-generated "
                 "transcript\n41: and filed the action items.")


def _both_surfaces(monkeypatch, snippet: str) -> tuple[str, str]:
    row = {"file": f"qmd://{_MARCH}", "title": "March 5 Transcript",
           "score": 0.9, "snippet": snippet}
    # vault_search, through its real handler body.
    monkeypatch.setattr(vault_mod, "_qmd_daemon_search", lambda *a, **k: [dict(row)])
    monkeypatch.setattr(vault_mod, "_grep_lloyd_code", lambda *a, **k: [])
    got = vault_mod._run_vault_search("march 5 transcript", 5, 0.0, "", False)
    monkeypatch.setattr(prefetch, "_qmd_daemon_search", _hunk_daemon(snippet))
    return got["results"][0]["snippet"], prefetch._search_vault(_PROBE_QUERY)[0]["snippet"]


def test_both_surfaces_strip_the_same_row_and_vault_search_is_unchanged(monkeypatch):
    vs, pf = _both_surfaces(monkeypatch, _HEADER_FIRST)
    # Byte-identical to what the handler produced before #994 (its old inline
    # strip, restated here as the reference).
    old = re.sub(r"@@[^@]*@@\s*(?:\([^)]*\)\s*)?", "", _HEADER_FIRST).strip()
    old = re.sub(r"^\d+:\s*", "", old, flags=re.MULTILINE).strip()[:300]
    assert vs == old
    # prefetch, same row: the same text, free of both artefacts.
    assert pf == vs
    for s in (vs, pf):
        assert "@@" not in s and not re.search(r"(?m)^\d+:", s)


def test_a_numbered_header_line_is_stripped_on_both_surfaces(monkeypatch):
    # The shape the injected corpus shows (`40: @@ -39,4 @@ …`). The old inline
    # strip left `41: ` standing here; the shared helper does not, on either.
    vs, pf = _both_surfaces(monkeypatch, _HUNKED)
    assert pf == vs
    for s in (vs, pf):
        assert "@@" not in s and not re.search(r"(?m)^\d+:", s)
        assert s.startswith("The assistant extracted")


def test_carried_hit_crosses_the_thread_hand_off_still_marked(quiet_workers, monkeypatch):
    # The hybrid (vec) leg runs on a worker thread and stashes its result on the
    # SessionFocus; the NEXT turn drains that stash, merges it, and renders it.
    # So `truncated` has to survive a dict written by another thread, an age
    # check and `_merge_vault_results`. Firing the legs by payload shape (the
    # lex leg sends one search, the hybrid two) keeps `_search_vault`, the stash
    # and the merge all real.
    def fake_post(payload):
        if len(payload["searches"]) > 1:      # the lex+vec hybrid leg
            time.sleep(0.4)
            return [{"file": f"qmd://{_MARCH}", "title": "March 5 Transcript",
                     "score": 0.7, "snippet": "y" * (prefetch.VAULT_SNIPPET_MAX + 200)}]
        return []                              # lex leg finds nothing in budget

    monkeypatch.setattr(vault_mod, "_qmd_post", fake_post)
    sid = "test-471-carried"
    out1 = prefetch.prefetch_context(_PROBE_QUERY, session_id=sid, plan_mode=False)
    assert "March 5 Transcript" not in out1    # straggler not back yet
    time.sleep(0.6)
    out2 = prefetch.prefetch_context(_PROBE_QUERY, session_id=sid, plan_mode=False)
    assert "semantic hit from the previous turn's query" in out2
    assert f"file: {_MARCH}" in out2
    assert "[... truncated]" in out2


# ── Budget + carry-over end to end (workers patched) ──────────────────────────

@pytest.fixture
def quiet_workers(monkeypatch):
    monkeypatch.setattr(prefetch, "_search_skills", lambda q: [])
    monkeypatch.setattr(prefetch, "_search_facts", lambda t: [])
    monkeypatch.setattr(prefetch, "_search_recent_sessions", lambda t: [])
    monkeypatch.setattr(prefetch, "_search_backlog_refs", lambda t: [])
    monkeypatch.setattr(prefetch, "_format_ide_state", lambda: "")
    monkeypatch.setattr(prefetch, "PREFETCH_BUDGET_MS", 120)


def _fake_vault(hybrid_delay: float):
    def _search(query, focus=None, legs=("lex", "vec"), **kw):
        if legs == ("lex",):
            return [{"file": "lex.md", "title": "Lex Hit", "score": 0.9, "snippet": "from lex"}]
        time.sleep(hybrid_delay)
        return [
            {"file": "lex.md", "title": "Lex Hit", "score": 0.9, "snippet": "from lex"},
            {"file": "vec.md", "title": "Vec Hit", "score": 0.7, "snippet": "from vec"},
        ]
    return _search


def test_slow_hybrid_is_dropped_then_carried_over(quiet_workers, monkeypatch):
    monkeypatch.setattr(prefetch, "_search_vault", _fake_vault(hybrid_delay=0.5))
    sid = "test-carry"
    msg = "tell me about the alfie servo shoulder pid gains please"

    t0 = time.monotonic()
    out1 = prefetch.prefetch_context(msg, session_id=sid, plan_mode=False)
    elapsed = time.monotonic() - t0
    # The hybrid leg is waited on since 2026-09-24, but only up to the
    # budget: a leg stuck behind a busy daemon costs the turn the 120ms
    # budget and no more, then carries over.
    assert 0.10 <= elapsed < 0.25, f"prefetch returned after {elapsed:.3f}s against a 120ms budget"
    assert "Lex Hit" in out1
    assert "Vec Hit" not in out1  # straggler not ready yet
    assert out1.endswith("\n\n" + msg)

    time.sleep(0.6)  # let the straggler finish and stash
    out2 = prefetch.prefetch_context(msg, session_id=sid, plan_mode=False)
    assert "Vec Hit" in out2
    assert "semantic hit from the previous turn's query" in out2
    assert "Lex Hit" in out2
    # the carry-over was consumed (the new straggler hasn't finished yet)
    focus = prefetch._get_session_focus(sid)
    assert "hybrid" not in focus.pending_vault


def test_hybrid_leg_starts_only_after_lex_returns(quiet_workers, monkeypatch):
    # qmd serializes requests, so the hybrid (vec) leg must not reach the
    # daemon before the lex leg has come back.
    events: list[tuple[str, float]] = []

    def _search(query, focus=None, legs=("lex", "vec"), **kw):
        if legs == ("lex",):
            time.sleep(0.05)
            events.append(("lex_done", time.monotonic()))
            return [{"file": "lex.md", "title": "Lex Hit", "score": 0.9, "snippet": "x"}]
        events.append(("hybrid_start", time.monotonic()))
        return []

    monkeypatch.setattr(prefetch, "_search_vault", _search)
    prefetch.prefetch_context("tell me about the alfie servo shoulder pid", session_id="test-order", plan_mode=False)
    time.sleep(0.1)
    order = [e[0] for e in events]
    assert order == ["lex_done", "hybrid_start"], order


def test_fast_hybrid_is_used_and_stash_cleared(quiet_workers, monkeypatch):
    # Every other worker is instant here: the wait loop must still hold the
    # turn for the hybrid leg (~55 ms in production) instead of returning on
    # the lex leg alone and leaving the semantic hits to the next turn.
    monkeypatch.setattr(prefetch, "_search_vault", _fake_vault(hybrid_delay=0.03))
    sid = "test-fast-hybrid"
    out = prefetch.prefetch_context("tell me about the alfie servo shoulder pid", session_id=sid, plan_mode=False)
    assert "Vec Hit" in out and "previous turn" not in out
    assert prefetch._get_session_focus(sid).pending_vault == {}


def test_in_turn_hybrid_keeps_lex_hits_it_does_not_return(quiet_workers, monkeypatch):
    # The hybrid leg is not a superset of the lex leg (different lex query,
    # vec rows displace lex rows under fusion): on the 86-query prefetch eval
    # lex alone scored doc_hit 0.140, hybrid alone 0.151, the two fused 0.221.
    # A lex-only hit must survive the hybrid landing, and neither leg's hits
    # may be labelled as carried from a previous turn.
    def _search(query, focus=None, legs=("lex", "vec"), **kw):
        if legs == ("lex",):
            return [{"file": "lexonly.md", "title": "Lex Only", "score": 0.9, "snippet": "x"}]
        return [{"file": "veconly.md", "title": "Vec Only", "score": 0.7, "snippet": "y"}]

    monkeypatch.setattr(prefetch, "_search_vault", _search)
    out = prefetch.prefetch_context("tell me about the alfie servo shoulder pid",
                                    session_id="test-fuse", plan_mode=False)
    assert "Lex Only" in out and "Vec Only" in out
    assert "previous turn" not in out


def test_fuse_fresh_vault_without_hybrid_is_the_lex_leg():
    lex = [{"file": "a.md", "title": "A", "score": 0.9}]
    assert prefetch._fuse_fresh_vault(lex, None) == lex
    fused = prefetch._fuse_fresh_vault(lex, [{"file": "b.md", "title": "B", "score": 0.6}])
    assert [r["file"] for r in fused] == ["a.md", "b.md"]
    assert not any(r.get("carried") for r in fused)


def test_stale_carry_over_is_discarded(quiet_workers, monkeypatch):
    monkeypatch.setattr(prefetch, "_search_vault", _fake_vault(hybrid_delay=0.0))
    focus = prefetch._get_session_focus("test-stale")
    focus.stash_vault([{"file": "old.md", "title": "Old", "score": 0.9, "snippet": "x"}])
    ts, res = focus.pending_vault["hybrid"]
    focus.pending_vault["hybrid"] = (ts - prefetch.VAULT_CARRY_MAX_AGE_S - 1, res)
    assert focus.take_vault() == []


def test_take_vault_merges_both_legs_hybrid_first():
    focus = prefetch.SessionFocus()
    focus.stash_vault([{"file": "a.md", "title": "A", "score": 0.6, "snippet": "x"},
                       {"file": "b.md", "title": "B", "score": 0.9, "snippet": "x"}], leg="lex")
    focus.stash_vault([{"file": "a.md", "title": "A", "score": 0.8, "snippet": "x"},
                       {"file": "c.md", "title": "C", "score": 0.7, "snippet": "x"}], leg="hybrid")
    got = focus.take_vault()
    assert [(r["file"], r["score"]) for r in got] == [("b.md", 0.9), ("a.md", 0.8), ("c.md", 0.7)]
    assert focus.pending_vault == {}


def test_cold_lex_leg_straggles_and_carries_over(quiet_workers, monkeypatch):
    # Lex takes longer than the soft wait: the turn must return at ~soft
    # wait, not at the full budget, and the lex result must arrive next turn.
    def _search(query, focus=None, legs=("lex", "vec"), **kw):
        if legs == ("lex",):
            time.sleep(0.25)
            return [{"file": "cold.md", "title": "Cold Lex Hit", "score": 0.8, "snippet": "x"}]
        return []

    monkeypatch.setattr(prefetch, "_search_vault", _search)
    monkeypatch.setattr(prefetch, "PREFETCH_BUDGET_MS", 400)
    monkeypatch.setattr(prefetch, "VAULT_LEX_SOFT_WAIT_MS", 100)
    sid = "test-cold-lex"
    t0 = time.monotonic()
    out1 = prefetch.prefetch_context("tell me about the cold topic please", session_id=sid, plan_mode=False)
    elapsed = time.monotonic() - t0
    assert 0.09 <= elapsed < 0.22, elapsed
    assert "Cold Lex Hit" not in out1
    time.sleep(0.4)
    out2 = prefetch.prefetch_context("and more about the cold topic please", session_id=sid, plan_mode=False)
    assert "Cold Lex Hit" in out2 and "previous turn" in out2


def test_short_message_skips_search_but_keeps_ambient(quiet_workers, monkeypatch):
    from app.sessions_io import AmbientPrefetchEntry, enqueue_ambient_prefetch
    monkeypatch.setattr(prefetch, "_search_vault", _fake_vault(hybrid_delay=0.0))
    sid = "test-ambient-short"
    enqueue_ambient_prefetch(sid, AmbientPrefetchEntry(source="cron:x", summary="job done", enqueued_at=time.time()))
    out = prefetch.prefetch_context("ok", session_id=sid, plan_mode=False)
    assert "<ambient-signals>" in out and "job done" in out
    assert "Lex Hit" not in out  # search phase skipped for short text
    assert prefetch.prefetch_context("ok", session_id=sid, plan_mode=False) == "ok"  # queue drained


def test_prefetch_context_async_offloads(quiet_workers, monkeypatch):
    monkeypatch.setattr(prefetch, "_search_vault", _fake_vault(hybrid_delay=0.0))

    async def main():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        t = asyncio.create_task(ticker())
        out = await prefetch.prefetch_context_async(
            "tell me about the alfie servo shoulder pid gains", session_id="test-async", plan_mode=False,
        )
        t.cancel()
        return out, ticks

    out, ticks = asyncio.run(main())
    assert "Lex Hit" in out
    # The loop kept running while the worker thread did the search.
    assert ticks >= 0


# ── Lex leg: short AND sub-queries with a drop-a-term ladder ─────────────────

def test_enrich_query_does_not_repeat_message_words():
    focus = prefetch.SessionFocus()
    focus.update("alfie servo shoulder tuning")
    focus.update("more alfie servo work")
    q = focus.enrich_query("alfie servo pid")
    assert q.split().count("alfie") == 1 and q.split().count("servo") == 1
    assert "shoulder" in q  # prior-turn context still appended


def test_lex_subqueries_are_short_and_weighted():
    focus = prefetch.SessionFocus()
    focus.update("lets look at the alfie servo shoulder pid gains")
    focus.update("ok, and what did we decide about the stepper closed loop yesterday?")
    qs = focus.lex_subqueries("ok, and what did we decide about the stepper closed loop yesterday?")
    assert qs and all(len(q) <= prefetch.VAULT_LEX_MAX_TERMS for q in qs)
    assert "ok" not in qs[0]
    assert {"stepper", "closed", "loop"} <= set(qs[0])
    assert any("shoulder" in q or "servo" in q for q in qs[1:])  # prior-turn focus query


def test_lex_ladder_drops_terms_until_a_hit(monkeypatch):
    calls: list[str] = []

    def _search(query, focus=None, legs=("lex", "vec"), **kw):
        calls.append(query)
        terms = query.split()
        if legs == ("lex",) and len(terms) <= 3 and "stepper" in terms:
            return [{"file": "s.md", "title": "Closed-Loop Stepper", "score": 0.9, "snippet": "x"}]
        return []

    monkeypatch.setattr(prefetch, "_search_vault", _search)
    focus = prefetch.SessionFocus()
    focus.update("what did we decide about the stepper closed loop yesterday?")
    hits = prefetch._search_vault_lex("what did we decide about the stepper closed loop yesterday?", focus)
    assert [h["file"] for h in hits] == ["s.md"]
    assert len(calls[0].split()) == 4 and len(calls[1].split()) == 3  # ladder stepped down once
    assert len(calls) <= prefetch.VAULT_LEX_MAX_CALLS


def test_single_term_query_still_runs(monkeypatch):
    calls: list[str] = []

    def _search(query, focus=None, legs=("lex", "vec"), **kw):
        calls.append(query)
        return [{"file": "qmd.md", "title": "QMD", "score": 0.9, "snippet": "x"}]

    monkeypatch.setattr(prefetch, "_search_vault", _search)
    focus = prefetch.SessionFocus()
    focus.update("qmd")
    hits = prefetch._search_vault_lex("qmd", focus)
    assert calls == ["qmd"] and hits


def test_lex_ladder_stops_at_deadline(monkeypatch):
    """The ladder must stop starting new calls once it is past the deadline.

    Driven by a fake clock rather than real sleeps. The behaviour under test is
    a comparison against `time.monotonic()`, not real timing, and the earlier
    version — `time.sleep(0.05)` per call plus a wall-clock bound — failed
    whenever the machine was busy. That matters more than usual here: the
    self-modification gate runs this suite while a canary is booting, so a
    load-sensitive test would randomly block promotions for reasons that have
    nothing to do with the change being gated.
    """
    calls: list[str] = []
    clock = {"t": 1000.0}

    def _fake_monotonic():
        return clock["t"]

    def _search(query, focus=None, legs=("lex", "vec"), **kw):
        calls.append(query)
        clock["t"] += 0.05          # each daemon round-trip "costs" 50ms
        return []                   # never hits → the ladder keeps stepping down

    monkeypatch.setattr(prefetch.time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(prefetch, "_search_vault", _search)
    focus = prefetch.SessionFocus()
    focus.update("alfie servo shoulder pid gains oscillation tuning")

    t0 = clock["t"]
    # Calls start at t0, t0+0.05, t0+0.10. A deadline of t0+0.08 (+margin) lets
    # the second start and must stop the third.
    hits = prefetch._search_vault_lex("alfie servo shoulder pid gains oscillation tuning", focus,
                                      deadline=t0 + 0.08 + prefetch.VAULT_LEX_DEADLINE_MARGIN_S)
    assert hits == []
    assert len(calls) == 2, calls   # not the 6-call maximum, and not 1


# ── Session index cache window ────────────────────────────────────────────────

def test_session_index_cache_respects_requested_window(tmp_path, monkeypatch):
    import datetime as dt
    today = dt.datetime.now().strftime("%Y%m%d")
    old = (dt.datetime.now() - dt.timedelta(days=10)).strftime("%Y%m%d")
    for stamp in (today, old):
        (tmp_path / f"{stamp}_120000_abc123.json").write_text(json.dumps({
            "session_id": stamp, "created_at": stamp, "model": "primary",
            "messages": [{"role": "user", "content": "servo tuning talk"}],
        }))
    monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(session_mod, "_session_index_cache", None)

    narrow = session_mod._load_session_index(max_days=3)
    assert len(narrow) == 1
    wide = session_mod._load_session_index(max_days=14)   # must NOT be served the 3-day cache
    assert len(wide) == 2
    narrow_again = session_mod._load_session_index(max_days=3)  # wide cache is fine to reuse
    assert len(narrow_again) == 2


# ── Backlog index incremental rebuild ─────────────────────────────────────────

def test_backlog_index_reparses_only_changed_files(tmp_path, monkeypatch):
    backlog = tmp_path / "obsidian" / "backlog"
    backlog.mkdir(parents=True)
    (backlog / "300-first.md").write_text("---\nstatus: open\npriority: high\nboard: lloyd\n---\n# First task\nbody one")
    (backlog / "301-second.md").write_text("---\nstatus: done\npriority: low\nboard: lloyd\n---\n# Second task\nbody two")
    monkeypatch.setattr(prefetch.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(prefetch, "_backlog_id_cache", {})
    monkeypatch.setattr(prefetch, "_backlog_file_cache", {})
    monkeypatch.setattr(prefetch, "_backlog_id_cache_ts", 0.0)

    parsed: list[str] = []
    real_parse = prefetch._parse_backlog_file

    def counting_parse(path):
        parsed.append(Path(path).name)
        return real_parse(path)

    monkeypatch.setattr(prefetch, "_parse_backlog_file", counting_parse)

    idx = prefetch._get_backlog_index()
    assert idx[300]["title"] == "First task" and idx[301]["status"] == "done"
    assert sorted(parsed) == ["300-first.md", "301-second.md"]

    # Touch one file with a strictly newer mtime, force a rescan.
    parsed.clear()
    f = backlog / "300-first.md"
    f.write_text("---\nstatus: closed\npriority: high\nboard: lloyd\n---\n# First task renamed\nbody")
    import os
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    monkeypatch.setattr(prefetch, "_backlog_id_cache_ts", 0.0)
    idx = prefetch._get_backlog_index()
    assert parsed == ["300-first.md"]
    assert idx[300]["title"] == "First task renamed" and idx[300]["status"] == "closed"
    assert idx[301]["title"] == "Second task"

    refs = prefetch._search_backlog_refs("what's left on #300 and 301?")
    assert len(refs) == 2 and "[Task #300]" in refs[0] and "[Task #301]" in refs[1]


# ── #1197: ambient delivery arrives with a clock the server measured ──────────
#
# The autotriage brief injected at 10:32Z on 2026-09-16 printed
# `Brief + Triage — 2026-09-17 02:53 PDT` and `📅 Today: (no events)` while
# Ben's Birthday sat on the 16th. The run had no date anywhere in its context
# — the scheduler's own stamp is in the run-record filename, not in the prompt —
# so it invented one, queried the invented day, got [] back legitimately, and
# finished `status: success`. A transcript pass over autonomy-task:68 sessions
# found 58 of 147 dated brief headers naming a calendar day other than their own
# run's. Nothing strips a model-invented date, so both ambient delivery paths
# have to arrive carrying a measured one. Every test below uses a payload that
# supplies no date at all, so any date in the delivered text can only be the
# server's.

_FIXED_ENQUEUED = datetime(2026, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
_LATER_ENQUEUED = datetime(2026, 3, 4, 9, 15, 0, tzinfo=timezone.utc)
# Two fixed instants chosen so their LOCAL calendar days differ from each other
# (2026-03-03 21:06 PST and 2026-03-04 01:15 PST), from the day this test runs,
# and from their own UTC day (both the 4th). A renderer that read the wall clock
# instead of `entry.enqueued_at`, or rendered UTC, cannot produce either string.


def _stamp(dt: datetime) -> str:
    """What the server owes the model: that instant, in the box's own zone."""
    return dt.astimezone().strftime("%Y-%m-%d %H:%M %Z")


def test_ambient_signal_lines_carry_server_measured_clocks(quiet_workers, monkeypatch):
    """Clause 1: every entry drained into `<ambient-signals>` shows a timestamp
    the server computed from that entry's `enqueued_at`. Two entries, two
    different local days — so the stamp is per entry, not one block-level
    "now" — and the summaries are date-free, so the block's only dates are the
    ones the queue supplied."""
    from app.sessions_io import AmbientPrefetchEntry, enqueue_ambient_prefetch
    monkeypatch.setattr(prefetch, "_search_vault", _fake_vault(hybrid_delay=0.0))
    sid = "test-ambient-stamp"
    summaries = ["Brief + Triage", "Digest is ready"]
    assert all(not re.search(r"\d{4}-\d{2}-\d{2}", s) for s in summaries)
    enqueue_ambient_prefetch(sid, AmbientPrefetchEntry(
        source="autotriage", summary=summaries[0],
        enqueued_at=_FIXED_ENQUEUED.timestamp()))
    enqueue_ambient_prefetch(sid, AmbientPrefetchEntry(
        source="research:digest", summary=summaries[1],
        enqueued_at=_LATER_ENQUEUED.timestamp()))

    out = prefetch.prefetch_context("ok", session_id=sid, plan_mode=False)

    assert "<ambient-signals>" in out and "Brief + Triage" in out and "Digest is ready" in out
    assert _stamp(_FIXED_ENQUEUED) in out, out
    assert _stamp(_LATER_ENQUEUED) in out, out
    assert _stamp(_FIXED_ENQUEUED)[:10] != _stamp(_LATER_ENQUEUED)[:10]


def test_ambient_signal_line_omits_a_stamp_the_server_never_measured(quiet_workers, monkeypatch):
    """An entry whose `enqueued_at` is the dataclass default (0.0 — no producer
    ever stamped it) must not be handed a 1969 calendar date. No measurement,
    no stamp; the line still carries the signal."""
    from app.sessions_io import AmbientPrefetchEntry, enqueue_ambient_prefetch
    monkeypatch.setattr(prefetch, "_search_vault", _fake_vault(hybrid_delay=0.0))
    sid = "test-ambient-unstamped"
    enqueue_ambient_prefetch(sid, AmbientPrefetchEntry(
        source="cron:x", summary="job done", enqueued_at=0.0))

    out = prefetch.prefetch_context("ok", session_id=sid, plan_mode=False)

    assert "<ambient-signals>" in out and "job done" in out
    assert "1969" not in out and "1970" not in out, out


def test_ambient_envelope_carries_the_server_measured_clock(tmp_path, monkeypatch):
    """Clause 2: the `<ambient …>` envelope built for a notable/urgent
    injection carries a server-composed timestamp while the producer's summary
    supplies no date. Pinned against `turn.enqueued_at` rather than a literal,
    because the envelope and the turn must be stamped from one measured
    instant — the same `datetime.now()` that ordered the queue."""
    from app.routers import messages as M
    monkeypatch.setattr(M, "SESSIONS_DIR", tmp_path)
    sid = "amb-stamp"
    (tmp_path / f"{sid}.json").write_text(json.dumps(
        {"session_id": sid, "messages": [], "platform": "mission-control"}))

    turn = asyncio.run(M.build_ambient_turn(
        sid, "Digest is ready", dedup_key="digest", priority="notable",
        source="research:digest", summary="Digest is ready"))

    text = turn.payload["prefetched_text"]
    assert text.startswith("<ambient ") and "Digest is ready" in text
    assert not re.search(r"\d{4}-\d{2}-\d{2}", turn.payload["text"])
    assert _stamp(turn.enqueued_at) in text, text


# ── #1013: the ambient summary is capped where the block is rendered ─────────

def _ambient_block(out: str) -> str:
    start = out.index("<ambient-signals>")
    end = out.index("</ambient-signals>") + len("</ambient-signals>")
    return out[start:end]


def _drain_one_summary(monkeypatch, sid: str, summary: str) -> str:
    from app.sessions_io import AmbientPrefetchEntry, enqueue_ambient_prefetch
    monkeypatch.setattr(prefetch, "_search_vault", _fake_vault(hybrid_delay=0.0))
    enqueue_ambient_prefetch(sid, AmbientPrefetchEntry(
        source="cron:x", summary=summary, content="c" * 900, enqueued_at=0.0))
    return prefetch.prefetch_context("ok", session_id=sid, plan_mode=False)


def test_oversized_ambient_summary_is_bounded_at_the_render_site(quiet_workers, monkeypatch):
    """Clause 1+2: a 5,000-char summary reaches the model as at most
    AMBIENT_SUMMARY_MAX chars plus the marker plus the block's fixed prose —
    the envelope is measured from a well-behaved entry, so the bound is on the
    summary alone and does not depend on restating the prose here."""
    big = "s" * 5000
    out = _drain_one_summary(monkeypatch, "test-ambient-big", big)
    block = _ambient_block(out)
    assert big not in block
    assert "s" * prefetch.AMBIENT_SUMMARY_MAX + " [... truncated]" in block
    envelope = len(_ambient_block(_drain_one_summary(monkeypatch, "test-ambient-env", "x")))
    assert len(block) <= envelope - 1 + prefetch.AMBIENT_SUMMARY_MAX + len(" [... truncated]"), len(block)
    # Clause 4: the content cap beside it is untouched — 800 chars and "…".
    assert "c" * 800 + "…" in block and "c" * 801 not in block


def test_well_behaved_ambient_summary_renders_unchanged(quiet_workers, monkeypatch):
    """Clause 3: a summary under the cap is rendered byte-for-byte — real
    traffic (every persisted summary is under 50 chars) must not shift."""
    summary = ("summary " * 15)[:120]
    assert len(summary) == 120 < prefetch.AMBIENT_SUMMARY_MAX
    out = _drain_one_summary(monkeypatch, "test-ambient-120", summary)
    block = _ambient_block(out)
    assert f"- **[cron:x]** {summary}\n" in block
    assert "truncated" not in block


# ── #1024: the ambient fact leg must not answer about other items ────────────

def _class_row_facts_tree(tmp_path, monkeypatch, *, with_specific=True):
    """A temp facts tree with the live shape: `Backlog Item` (51 facts about
    OTHER items, one over FACT_GODNODE_THRESHOLD so the god-node filter is
    engaged) plus, when asked for, the queried item's own row.

    Returns (named_other_ids, bullet_query).
    """
    import yaml
    import agent_mcp._shared as shared
    from agent_mcp import retrieval, facts as facts_mod

    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    for mod in (shared, retrieval, facts_mod):
        monkeypatch.setattr(mod, "FACTS_ROOT", facts_root)
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()
    retrieval._entity_index_cache = None

    def write(entity, facts):
        d = facts_root / entity
        d.mkdir(parents=True, exist_ok=True)
        fm = {"type": "facts", "entity": entity, "category": "activity",
              "facts": facts}
        (d / f"{entity}-activity.md").write_text(
            f"---\n{yaml.dump(fm, sort_keys=False)}---\n")

    named = [str(900 + i) for i in range(60)]
    write("Backlog Item", [{"fact": f"Backlog item #{n} was created on the "
                                  f"lloyd board.", "id": f"g{i:03d}",
                            "confidence": 1.0}
                           for i, n in enumerate(named)])
    if with_specific:
        write("Task #363", [{"fact": "Task #363 was created on the lloyd board "
                                     "with Medium priority.", "id": "s001",
                             "confidence": 0.6}])
    return named


def test_ambient_facts_for_one_item_never_come_from_the_class_row(tmp_path, monkeypatch):
    """`prefetch._search_facts` runs on EVERY message ≥ MIN_MESSAGE_LEN and
    applies no query-token filter at all: `FACT_MAX_ENTITIES = 2` then each
    entity's facts by confidence alone. The class row's confidence-1.0 facts
    about other items therefore reached the ambient block unconditionally.

    The seam here is import-identity: `prefetch` binds
    `_extract_entities_from_query` from `agent_mcp.facts`, which re-exports
    `agent_mcp.retrieval.extract_entities_from_query`. The suppression lives in
    retrieval, so this test is what proves the fix crosses that boundary.
    """
    named = _class_row_facts_tree(tmp_path, monkeypatch)
    bullets = prefetch._search_facts("tell me about backlog item 363")
    assert bullets, "the item has a fact; silence is not the intended outcome here"
    wrong = [b for b in bullets if any(f"#{n}" in b for n in named)]
    assert not wrong, (
        f"{len(wrong)} of {len(bullets)} ambient bullets name another item: {wrong}")


def test_ambient_facts_are_empty_rather_than_wrong_for_an_unwritten_item(
        tmp_path, monkeypatch):
    """Same leg, the case with no row for the named id: an empty ambient fact
    block is required, not other items' facts."""
    named = _class_row_facts_tree(tmp_path, monkeypatch, with_specific=False)
    bullets = prefetch._search_facts("tell me about backlog item 363")
    wrong = [b for b in bullets if any(f"#{n}" in b for n in named)]
    assert not wrong, f"ambient block carried other items' facts: {wrong}"


# ── #839: a filename-shaped row may never speak in the ambient block ─────────

FILENAME_FACT_ENTITIES = ("Stompy Robotics", "stompy-robotics-build.md",
                          "2026-09-08-stompy-robotics")


def _filename_shape_facts_tree(tmp_path, monkeypatch):
    """Facts tree over the query's subject plus one filename-shaped row of each
    shape, every one carrying a confidence-1.0 fact, so all three are eligible
    at the same confidence and only the seed ranking can choose between them.

    The subject's fact is written so the test can tell the three apart by
    content as well as by the `[entity]` prefix.
    """
    import yaml
    import agent_mcp._shared as shared
    from agent_mcp import retrieval, facts as facts_mod

    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    for mod in (shared, retrieval, facts_mod):
        monkeypatch.setattr(mod, "FACTS_ROOT", facts_root)
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()
    retrieval._entity_index_cache = None

    for entity in FILENAME_FACT_ENTITIES:
        d = facts_root / entity
        d.mkdir(parents=True, exist_ok=True)
        fm = {"type": "facts", "entity": entity, "category": "activity",
              "facts": [{"fact": f"{entity} says this", "id": "f-000",
                         "confidence": 1.0}]}
        (d / f"{entity}-activity.md").write_text(
            f"---\n{yaml.dump(fm, sort_keys=False)}---\n")
    return facts_root


def test_ambient_fact_lines_are_never_prefixed_by_a_filename_shaped_entity(
        tmp_path, monkeypatch):
    """Clause 3. `prefetch._search_facts` takes the extractor's top
    `FACT_MAX_ENTITIES` names and quotes each one's best facts with no query
    token filter of its own, so on this consumer a filename-shaped seed is not
    a wasted slot — it is a line in the prompt that speaks as the note's own
    entity. The budget is 2, which is what makes one bad seed half the block.

    The seam is the same import-identity one as #1024: `prefetch` binds
    `_extract_entities_from_query` from `agent_mcp.facts`, which re-exports
    `agent_mcp.retrieval.extract_entities_from_query`. The filter lives in the
    candidate list retrieval reads, so this test is what shows the boundary
    holds across the import.
    """
    _filename_shape_facts_tree(tmp_path, monkeypatch)
    assert prefetch.FACT_MAX_ENTITIES == 2, (
        f"the clause is stated at a budget of 2; it is now "
        f"{prefetch.FACT_MAX_ENTITIES}, so this test no longer measures it")

    bullets = prefetch._search_facts("stompy robotics")
    assert bullets, "the subject has a fact; silence here is not the fix"
    prefixed = [b.split("]")[0].lstrip("- [") for b in bullets]
    shaped = [p for p in prefixed
              if p.endswith(".md") or re.match(r"^\d{4}-\d{2}-\d{2}", p)]
    assert not shaped, (
        f"{len(shaped)} of {len(bullets)} ambient fact lines speak as a "
        f"filename-shaped entity {shaped}: {bullets}")
    assert "Stompy Robotics says this" in " ".join(bullets), (
        f"the subject's own fact is missing from the ambient block: {bullets}")

    # Positive control: over the UNFILTERED candidate list — what this path read
    # before #839 — one of the two slots must go to a filename row. If the
    # fixture ever stops producing that, this fires rather than letting the
    # assert above pass for the wrong reason.
    from agent_mcp import retrieval
    import agent_mcp._shared as shared
    monkeypatch.setattr(retrieval, "_get_rankable_entity_dirs_cached",
                        shared._get_entity_dirs_cached)
    unfixed = prefetch._search_facts("stompy robotics")
    unfixed_prefixed = [b.split("]")[0].lstrip("- [") for b in unfixed]
    assert any(p.endswith(".md") or re.match(r"^\d{4}-\d{2}-\d{2}", p)
               for p in unfixed_prefixed), (
        f"the fixture can no longer fail: no filename-shaped entity spoke even "
        f"unfiltered — {unfixed}")


# ── #1025: a digit directory may not take an ambient slot for a task id ──────

NUMERIC_TASK_ROWS = ("Task #294", "294", "Backlog Item #294")


def _numeric_task_facts_tree(tmp_path, monkeypatch):
    """One task id, three rows, and every ranking input pinned.

    Three rows because the width is what makes the harm. `Task #294` is the
    entity the query asks about; `294` is the digit directory the fact
    extractor minted from notes that named the item by number alone; `Backlog
    Item #294` is a row that names the same item and lands on the same 0.5.
    Before the guard `294` scored 10.0 beside the canonical, so with
    `FACT_MAX_ENTITIES = 2` it took the second slot and `Backlog Item #294`
    never appeared; after it the tie at 0.5 breaks on name length — the sort's
    own second key — and the digit row loses the slot it never earned.

    All three carry one confidence-1.0 fact whose text says which row is
    speaking, so no confidence filter can be doing the choosing.
    `_alias_surface_map` and `_edge_counts_or_empty` are emptied because the
    tie-break at equal score consults graph degree: a run from a tree with the
    live store reachable would let live edges decide what this test asserts.
    """
    import yaml
    import agent_mcp._shared as shared
    from agent_mcp import retrieval, facts as facts_mod

    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    for mod in (shared, retrieval, facts_mod):
        monkeypatch.setattr(mod, "FACTS_ROOT", facts_root)
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()
    retrieval._entity_index_cache = None
    monkeypatch.setattr(retrieval, "_alias_surface_map", lambda: {})
    monkeypatch.setattr(retrieval, "_edge_counts_or_empty", lambda *a, **k: {})

    says = {
        "Task #294": "Task #294 is the PPR rescoping item.",
        "294": "294 was a YAML scanner false positive on 2026-08-30.",
        "Backlog Item #294": "Backlog Item #294 is Memory Graph.",
    }
    for i, entity in enumerate(NUMERIC_TASK_ROWS):
        d = facts_root / entity
        d.mkdir(parents=True, exist_ok=True)
        fm = {"type": "facts", "entity": entity, "category": "activity",
              "facts": [{"fact": says[entity], "id": f"n{i:03d}",
                         "confidence": 1.0}]}
        (d / f"{entity}-activity.md").write_text(
            f"---\n{yaml.dump(fm, sort_keys=False)}---\n")


def test_ambient_fact_lines_are_never_prefixed_by_a_bare_task_id(tmp_path,
                                                                 monkeypatch):
    """The consumer the harm was measured on. `_search_facts` runs on every
    message ≥ MIN_MESSAGE_LEN, takes the extractor's top
    `FACT_MAX_ENTITIES` names and quotes each one's best facts with no query
    token filter of its own — so on the live tree `what is task 294 about` put
    3 of its 6 ambient bullets' facts about a YAML scanner bug under the prefix
    `[294]`, beside the canonical `[Task #294]` lines.

    The seam is the same import-identity one as #1024 and #839: `prefetch`
    binds `_extract_entities_from_query` from `agent_mcp.facts`, which
    re-exports `agent_mcp.retrieval.extract_entities_from_query`. The cap lives
    in retrieval's `_bump`, so this is the test that shows it reaches the block
    that is actually injected.
    """
    _numeric_task_facts_tree(tmp_path, monkeypatch)
    assert prefetch.FACT_MAX_ENTITIES == 2, (
        f"the clause is stated at a budget of 2; it is now "
        f"{prefetch.FACT_MAX_ENTITIES}, so this test no longer measures it")

    bullets = prefetch._search_facts("what is task 294 about")
    assert bullets, "the task has a fact; silence here is not the fix"
    prefixed = [b.split("]")[0].lstrip("- [") for b in bullets]
    numeric = [p for p in prefixed if re.fullmatch(r"#?\d+", p)]
    assert not numeric, (
        f"{len(numeric)} of {len(bullets)} ambient fact lines speak as a bare "
        f"task id {numeric}: {bullets}")
    assert "Task #294 is the PPR rescoping item." in " ".join(bullets), (
        f"the task's own fact is missing from the ambient block: {bullets}")

    # Positive control: with the name-shape guard answering False — what this
    # path did before #1025 — the digit row must take a slot again. If the
    # fixture ever stops producing that, this fires instead of letting the
    # assert above pass for the wrong reason.
    from agent_mcp import retrieval
    monkeypatch.setattr(retrieval, "_is_numeric_seed_name", lambda _n: False)
    retrieval._entity_index_cache = None
    unfixed = prefetch._search_facts("what is task 294 about")
    unfixed_prefixed = [b.split("]")[0].lstrip("- [") for b in unfixed]
    assert any(re.fullmatch(r"#?\d+", p) for p in unfixed_prefixed), (
        f"the fixture can no longer fail: `[294]` did not speak even with the "
        f"guard disabled — {unfixed}")


# ── #657: hard constraints past the skill cut ────────────────────────────────

def _long_skill(tail: str) -> dict:
    filler = "".join(f"step {i}: do the ordinary thing carefully.\n" for i in range(400))
    assert len(filler) > prefetch.SKILL_BODY_MAX
    return {"name": "pipeline", "raw": "# Pipeline\n" + filler + tail}


def test_truncated_skill_keeps_its_hard_constraints():
    rule = "- **HARD RULE — NEVER modify `~/lloyd/.venvs/`**. SIGNAL:BLOCKED instead."
    tail = ("## Notes\nsome prose that must not be carried.\n"
            "```json\n{\"note\": \"NEVER in a sample\"}\n```\n" + rule + "\n")
    skill = _long_skill(tail)
    out = prefetch._format_context([(9.0, skill)], [])
    assert rule.strip() in out
    assert "NEVER in a sample" not in out           # fenced code is not a rule
    assert "some prose that must" not in out         # lowercase prose is not either
    assert "[... truncated]\n</skill>" in out
    assert len(out) < prefetch.SKILL_BODY_MAX + prefetch.SKILL_CONSTRAINTS_MAX + 200


def test_carried_constraints_are_bounded():
    tail = "".join(f"- NEVER do bad thing number {i} under any circumstance.\n"
                   for i in range(200))
    out = prefetch._format_context([(9.0, _long_skill(tail))], [])
    carried = out.split("[... hard constraints from beyond the cut]\n", 1)[1]
    carried = carried.split("\n[... truncated]", 1)[0]
    assert 0 < len(carried) <= prefetch.SKILL_CONSTRAINTS_MAX


def test_second_skill_excerpt_keeps_its_hard_constraints():
    rule = "- ALWAYS run the dry-run first."
    first = {"name": "a", "raw": "short"}
    second = _long_skill(rule + "\n")
    out = prefetch._format_context([(9.0, first), (9.0, second)], [])
    excerpt = out.split('excerpt="true">', 1)[1]
    assert rule in excerpt
    assert len(excerpt) < (prefetch.SKILL_EXCERPT_MAX
                           + prefetch.SKILL_EXCERPT_CONSTRAINTS_MAX + 200)


def test_short_skill_is_injected_unchanged():
    raw = "# Tiny\nNEVER do the thing.\n"
    out = prefetch._format_context([(9.0, {"name": "tiny", "raw": raw})], [])
    assert f'<skill name="tiny" score="9.0">\n{raw}\n</skill>' in out
    assert "truncated" not in out


# ── #435 skill-injection telemetry: prefetch.skill_match events ───────────────
#
# The half of skill usage that had no source: `_search_skills` kept the matches
# and threw the rest away, and the only record of any of it was an aggregate
# `logger.debug` count line. These tests pin the per-turn event, its agreement
# with what the renderer actually put in `<context>`, and that a broken event
# writer cannot touch the turn. The reader over those events is
# `app.skill_telemetry`, tested in tests/test_skill_injection_telemetry.py.

_SKM_QUERY = "please frobnicate the widget"


def _skm_skill(name: str, *, desc: str = "", body: str = "", raw: str | None = None) -> dict:
    """A skill dict shaped like `_iter_skills`' output for the two consumers
    that matter here: the scorer reads `description`/`tags`/`body`, the
    renderer reads `raw`."""
    return {"name": name, "description": desc, "tags": [],
            "raw": body if raw is None else raw, "body": body}


def _skm_offers(*pairs) -> list[tuple[float, dict]]:
    """`[(score, name)]` in the shape `_search_skills` returns."""
    return [(score, _skm_skill(name, raw=f"{name} body text"))
            for score, name in pairs]


def _skm_turn(tmp_path, monkeypatch, offers, session_id: str) -> str:
    """One real `prefetch_context` turn, with every other leg stubbed to
    nothing and the event-log root aimed at this test's `tmp_path`.

    Only `_search_skills` is stubbed, because the offer list is what these
    tests fix; the emitter, the renderer and the event writer are the real
    ones.
    """
    log_root = tmp_path / "event_logs"
    monkeypatch.setattr("app.event_log.EVENT_LOGS_DIR", log_root)
    monkeypatch.setattr("app.event_log.BLOBS_DIR", log_root / "blobs")
    monkeypatch.setattr(prefetch, "_search_skills", lambda q: list(offers))
    monkeypatch.setattr(prefetch, "_search_facts", lambda t: [])
    monkeypatch.setattr(prefetch, "_search_recent_sessions", lambda t: [])
    monkeypatch.setattr(prefetch, "_search_backlog_refs", lambda t: [])
    monkeypatch.setattr(prefetch, "_search_vault", lambda *a, **k: [])
    monkeypatch.setattr(prefetch, "_format_ide_state", lambda: "")
    monkeypatch.setattr(prefetch, "PREFETCH_BUDGET_MS", 300)
    return prefetch.prefetch_context(_SKM_QUERY, session_id=session_id,
                                     plan_mode=False)


def _skm_events(tmp_path, session_id: str) -> list[dict]:
    """Parsed rows of the session's event log, as the on-disk file holds them."""
    path = tmp_path / "event_logs" / f"{session_id}.events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _skm_matches(tmp_path, session_id: str) -> dict[str, dict]:
    rows = [e for e in _skm_events(tmp_path, session_id)
            if e.get("event") == "prefetch.skill_match"]
    return {r["data"]["skill"]: r for r in rows}


def test_matched_skill_appends_one_skill_match_event_per_skill(tmp_path, monkeypatch):
    """Clause 1: three matched skills, three rows, in this session's log file,
    each carrying a name, a numeric score and a boolean `landed`."""
    _skm_turn(tmp_path, monkeypatch,
              _skm_offers((6.0, "alpha"), (4.5, "beta"), (3.2, "gamma")),
              "skm-clause1")

    rows = _skm_matches(tmp_path, "skm-clause1")

    assert (tmp_path / "event_logs" / "skm-clause1.events.jsonl").exists(), (
        "the events went somewhere other than the session's own log file")
    assert set(rows) == {"alpha", "beta", "gamma"}, rows
    for name, row in rows.items():
        data = row["data"]
        assert isinstance(data["skill"], str) and data["skill"] == name
        assert isinstance(data["score"], (int, float)), f"{name}: {data!r}"
        assert isinstance(data["landed"], bool), f"{name}: {data!r}"
        assert row["session_id"] == "skm-clause1"
    assert rows["alpha"]["data"]["score"] == 6.0
    assert rows["beta"]["data"]["score"] == 4.5
    assert rows["gamma"]["data"]["score"] == 3.2


def test_turn_with_no_skill_match_appends_no_event(tmp_path, monkeypatch):
    """Clause 1's negative half: an empty offer list writes nothing at all, so
    the reader's empty store stays a store with no telemetry rather than a
    store full of zero-score rows."""
    _skm_turn(tmp_path, monkeypatch, [], "skm-none")

    assert _skm_events(tmp_path, "skm-none") == []


def test_landed_marks_exactly_the_skills_rendered_into_context(tmp_path, monkeypatch):
    """Clause 2: the first matched skill is rendered as a full body, the second
    only at or above `SKILL_THRESHOLD_SECOND`, and everything else — including
    a skill above the injection threshold — is an offer with `landed: false`.

    One rule drives both sides: `_skill_injection_plan` is what the renderer
    renders from and what the emitter marks, so a row cannot claim a skill
    landed that the renderer skipped.
    """
    out = _skm_turn(tmp_path, monkeypatch,
                    _skm_offers((6.0, "alpha"), (4.5, "beta"), (3.2, "gamma")),
                    "skm-landed")

    assert '<skill name="alpha"' in out, out
    assert '<skill name="beta" score="4.5" excerpt="true">' in out, out
    assert 'name="gamma"' not in out, (
        f"gamma sits at 3.2 — injected as #2 only at 4.0 — but rendered: {out}")

    rows = _skm_matches(tmp_path, "skm-landed")
    assert rows["alpha"]["data"]["landed"] is True
    assert rows["alpha"]["data"]["injected_body"] is True
    assert rows["beta"]["data"]["landed"] is True
    assert rows["beta"]["data"]["injected_body"] is False
    assert rows["gamma"]["data"]["landed"] is False, (
        "a matched skill the renderer dropped produced no row, or claimed to land")
    assert rows["gamma"]["data"]["injected_body"] is False


def test_second_skill_below_the_excerpt_threshold_is_offered_not_landed(tmp_path, monkeypatch):
    """The discrimination `landed` has to carry: 3.5 clears the injection gate
    but not the excerpt gate, so it is an offer that was not rendered — the
    exact row a retirement rule needs and #435 never had a source for."""
    out = _skm_turn(tmp_path, monkeypatch,
                    _skm_offers((6.0, "alpha"), (3.5, "delta")),
                    "skm-below")

    assert 'name="delta"' not in out, out
    rows = _skm_matches(tmp_path, "skm-below")
    assert rows["alpha"]["data"]["landed"] is True
    assert rows["delta"]["data"]["landed"] is False
    assert rows["delta"]["data"]["score"] == 3.5


def test_search_skills_reports_every_skill_above_the_reporting_floor(tmp_path, monkeypatch):
    """The floor itself, through the real scorer.

    `SKILL_THRESHOLD_FIRST` is 3.0 and discards everything below it, so an
    offer set defined by that gate would record almost no `landed: false` rows.
    The reporting floor is `score > 0` instead, which `_score_skill` bounds by
    construction (a skill with no name/desc/tag token hit scores 0.0 however
    much its body matches). Scores below are the real scorer's on this query:
    a name hit lands `widget-frobnicator` on exactly 3.0, a description-only
    hit puts `gizmo` at 2.0 — the near-miss band the old code threw away
    before anything was recorded — and `unrelated` matches on body text only,
    which `require_metadata_hit` scores 0.0.
    """
    hot = _skm_skill("widget-frobnicator")
    warm = _skm_skill("gizmo", desc="handles widget calibration")
    cold = _skm_skill("unrelated", body="widget frobnicator widget frobnicator")
    monkeypatch.setattr(prefetch, "_get_skills_cached", lambda: [hot, warm, cold])

    scored = prefetch._search_skills(prefetch._query_tokens(_SKM_QUERY))

    assert [(round(s, 1), sk["name"]) for s, sk in scored] == [
        (3.0, "widget-frobnicator"), (2.0, "gizmo")], scored
    assert prefetch._injectable_skills(scored) == [scored[0]], (
        "the injection gate moved: `gizmo` at 2.0 must stay below "
        "SKILL_THRESHOLD_FIRST while still being an offer")


def test_raising_event_writer_leaves_the_injected_context_unchanged(tmp_path, monkeypatch):
    """Clause 3: telemetry is a side effect. With the writer raising, the turn
    gets byte-identical context — same skills, same bodies, same order."""
    offers = _skm_offers((6.0, "alpha"), (4.5, "beta"), (3.2, "gamma"))
    out_ok = _skm_turn(tmp_path, monkeypatch, offers, "skm-writer-ok")

    def _boom(*args, **kwargs):
        raise RuntimeError("event store on fire")

    monkeypatch.setattr(prefetch, "log_event", _boom)
    out_raising = _skm_turn(tmp_path, monkeypatch, offers, "skm-writer-raises")

    assert '<skill name="alpha"' in out_ok, out_ok
    assert out_raising == out_ok, (
        "the event writer changed what the turn was handed")
    assert _skm_events(tmp_path, "skm-writer-raises") == [], (
        "a raising writer still left rows behind")


def test_reported_offers_are_capped_at_skill_report_top_k(monkeypatch, tmp_path):
    """`SKILL_REPORT_FLOOR` = 0.0 records everything the matcher had a signal
    about, which measured over the 382 most recent sessions' first real message
    (2026-09-24) is a mean of 110.5 and a median of 88 of the 187 cached skills —
    so the floor alone would write ~110 rows per prefetched turn.
    `SKILL_REPORT_TOP_K` = 8 (4× the two slots that can render) bounds that;
    the cap is a scope call a person owns (#435's `needs-human` clause), so the
    two numbers are pinned here rather than left implicit in a slice.
    """
    skills = [_skm_skill(f"sk{i:02d}", desc="widget calibration")
              for i in range(12)]
    monkeypatch.setattr(prefetch, "_get_skills_cached", lambda: skills)

    scored = prefetch._search_skills(prefetch._query_tokens(_SKM_QUERY))

    assert len(scored) == 12, "all twelve match `widget`; the floor is 0.0"
    assert len(prefetch._reported_offers(scored)) == prefetch.SKILL_REPORT_TOP_K == 8
    assert [sk["name"] for _, sk in prefetch._reported_offers(scored)] == [
        f"sk{i:02d}" for i in range(8)], "the cap must keep the top K, not any K"

    monkeypatch.setattr("app.event_log.EVENT_LOGS_DIR", tmp_path / "event_logs")
    prefetch._emit_skill_match_events(
        "cap-session", scored,
        prefetch._skill_injection_plan(prefetch._injectable_skills(scored)))

    assert [row["data"]["skill"] for row in _skm_events(tmp_path, "cap-session")] == [
        f"sk{i:02d}" for i in range(8)], "the cap is what reaches the log, not just the slice"


# ── #1482 rider 1: the <facts> block ordered by query relevance ─────────────

def _relevance_facts_tree(tmp_path, monkeypatch, facts):
    """One entity, `Harbor Relay`, carrying `facts` (dicts with fact/id/confidence)."""
    import yaml
    import agent_mcp._shared as shared
    from agent_mcp import retrieval, facts as facts_mod

    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    for mod in (shared, retrieval, facts_mod):
        monkeypatch.setattr(mod, "FACTS_ROOT", facts_root)
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()
    retrieval._entity_index_cache = None
    d = facts_root / "Harbor Relay"
    d.mkdir()
    fm = {"type": "facts", "entity": "Harbor Relay", "category": "state", "facts": facts}
    (d / "Harbor Relay-state.md").write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n")
    return facts_root


_HARBOR = [
    {"fact": "Harbor Relay was rewritten in Rust last spring.", "id": "s001", "confidence": 0.99},
    {"fact": "Harbor Relay is owned by the platform group.", "id": "s002", "confidence": 0.98},
    {"fact": "Harbor Relay ships nightly builds.", "id": "s003", "confidence": 0.97},
    {"fact": "Harbor Relay listens on port 7443 for TLS clients.", "id": "s004", "confidence": 0.60},
    {"fact": "Harbor Relay stores its port map in etcd.", "id": "s005", "confidence": 0.70},
    {"fact": "Harbor Relay's admin port is 9090.", "id": "s006", "confidence": 0.50},
]


def test_facts_relevance_order_keeps_the_matching_fact_and_drops_the_offtopic_one(
        tmp_path, monkeypatch):
    """Clause 1. The on-topic fact (confidence 0.60) sits below three
    off-topic ones by confidence, so today's order never shows it; relevance
    ranks it in and drops the highest-confidence off-topic fact. Two facts with
    EQUAL overlap are ordered by confidence."""
    _relevance_facts_tree(tmp_path, monkeypatch, [dict(f) for f in _HARBOR])
    q = "which port does Harbor Relay listen on for tls clients"
    today = prefetch._search_facts(q, rank="confidence")
    assert not any("7443" in b for b in today), f"control: confidence order must miss it: {today}"
    ranked = prefetch._search_facts(q, rank="relevance")
    assert any("7443" in b for b in ranked), ranked
    assert not any("rewritten in Rust" in b for b in ranked), ranked
    assert len(ranked) == prefetch.FACT_MAX_PER_ENTITY

    # tie-break: "port" is the only distinguishing token and both port facts
    # carry it, so the 0.70 fact precedes the 0.60 one
    ranked2 = prefetch._search_facts("Harbor Relay port", rank="relevance")
    idx = {k: next(i for i, b in enumerate(ranked2) if k in b) for k in ("etcd", "7443")}
    assert idx["etcd"] < idx["7443"], ranked2


def test_facts_relevance_adds_no_read(tmp_path, monkeypatch):
    """Clause 2. At most FACT_MAX_ENTITIES `_get_facts_sync` calls and no
    daemon or model call: the ranking is computed from the facts in hand."""
    _relevance_facts_tree(tmp_path, monkeypatch, [dict(f) for f in _HARBOR])
    calls = []
    real = prefetch._get_facts_sync

    def counting(entity, *a, **k):
        calls.append(entity)
        return real(entity, *a, **k)

    def forbidden(*a, **k):
        raise AssertionError("ranking the facts block must not call a daemon or model")

    monkeypatch.setattr(prefetch, "_get_facts_sync", counting)
    monkeypatch.setattr(prefetch, "_qmd_daemon_search", forbidden)
    import app.djev as djev
    monkeypatch.setattr(djev, "ask_sync", forbidden)
    out = prefetch._search_facts("which port does Harbor Relay listen on", rank="relevance")
    assert out and 1 <= len(calls) <= prefetch.FACT_MAX_ENTITIES, calls


def test_facts_relevance_fails_open_without_usable_tokens(tmp_path, monkeypatch):
    """Clause 3. A query with no usable token (empty, or every term a
    stopword/too short) keeps today's confidence-descending order, and the
    real path still returns FACT_MAX_PER_ENTITY facts for the entity."""
    _relevance_facts_tree(tmp_path, monkeypatch, [dict(f) for f in _HARBOR])
    from agent_mcp.retrieval import fact_query_tokens
    for q in ("what is the", ""):
        assert fact_query_tokens(q) == []
        rows = [dict(f) for f in _HARBOR]
        assert prefetch._rank_entity_facts(rows, fact_query_tokens(q), "relevance") == \
            sorted(rows, key=lambda f: f["confidence"], reverse=True)
    today = prefetch._search_facts("tell me about Harbor Relay", rank="confidence")
    ranked = prefetch._search_facts("tell me about Harbor Relay", rank="relevance")
    assert ranked == today and len(ranked) == prefetch.FACT_MAX_PER_ENTITY


def test_facts_rank_mode_reads_config_and_defaults_to_confidence(monkeypatch):
    import app.config as cfg
    monkeypatch.setitem(cfg.CONFIG, "prefetch", {"facts": {"rank": "relevance"}})
    assert prefetch._facts_rank_mode() == "relevance"
    monkeypatch.setitem(cfg.CONFIG, "prefetch", {"facts": {"rank": "bogus"}})
    assert prefetch._facts_rank_mode() == "confidence"
    monkeypatch.setitem(cfg.CONFIG, "prefetch", {})
    assert prefetch._facts_rank_mode() == "confidence"
