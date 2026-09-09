"""A thinking trace is captured in the logs and absent from the transcripts.

Alan's requirement, stated directly: the reasoning should be in the chat
logs and should *not* be reproduced in the transcripts generated from
those logs. Nothing enforces that with a filter — it holds because every
producer dispatches on role and reads only `content` blocks of type
"text", while a `role="thinking"` row keeps its text in `reasoning` and
leaves `content` empty.

That is a property of six programs agreeing, not of one rule in one place,
so it is worth a test that actually runs all of them. A future producer
that reads whole messages breaks it, and nothing else would say so.

Which half does the work was measured rather than assumed, because the
intuitive answer is wrong: it is the **role**. Changing it to "assistant"
leaks the reasoning from six of the seven producers; leaving it alone and
filling `content` leaks from none. `test_the_role_is_what_protects_the
_transcripts` pins that, so nobody reads the empty `content` as the
reason and concludes the role is free to change.
"""

from __future__ import annotations

import importlib.util
import json
import sys

from app.paths import LLOYD_HOME
from app.routers._messages_thinking import _build_thinking_entry


# A phrase that appears only inside the reasoning, so any leak is
# unambiguous no matter which producer let it through.
SECRET = "zarquon-deliberation-marker"


class _Turn:
    turn_id = "T1"


def _session() -> dict:
    """A session shaped like a real tool-using turn, thinking included."""
    return {
        "session_id": "20260908_120000_test",
        "created_at": "2026-09-08T12:00:00",
        "model": "primary",
        "messages": [
            {
                "id": "u1", "role": "user",
                "content": [{"type": "text", "text": "How much disk is left?"}],
                "timestamp": "2026-09-08T12:00:00",
            },
            _build_thinking_entry(
                _Turn(), f"I should check the root filesystem. {SECRET}",
                4200, 0, 1, "2026-09-08T12:00:01",
            ),
            {
                "id": "a1", "role": "assistant",
                "content": [{"type": "text", "text": "The disk is at 78 percent."}],
                "timestamp": "2026-09-08T12:00:05",
            },
        ],
    }


def _load_script(relpath: str, name: str):
    """Import a `scripts/` file that is not an importable module."""
    path = LLOYD_HOME / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------- vault markdown


def test_vault_session_export_omits_reasoning(tmp_path, monkeypatch):
    """The export that runs automatically after every turn."""
    from app import post_capture

    monkeypatch.setattr(post_capture, "VAULT_SESSIONS_DIR", tmp_path)
    out = post_capture._export_session_markdown("s1", _session())

    assert out is not None, "the fixture must produce a real export"
    body = out.read_text()
    assert SECRET not in body
    # And the transcript is otherwise intact — an empty file would pass
    # the assertion above for the wrong reason.
    assert "How much disk is left?" in body
    assert "The disk is at 78 percent." in body


# ---------------------------------------------------- LLM-input transcripts


def test_capture_transcript_omits_reasoning():
    from app.post_capture import _build_capture_transcript

    out = _build_capture_transcript(_session()["messages"])
    assert SECRET not in out
    assert "How much disk is left?" in out


def test_session_title_transcript_omits_reasoning():
    from app.session_titles import build_transcript

    out = build_transcript(_session())
    assert SECRET not in out
    assert "How much disk is left?" in out


# ------------------------------------------------------- memory pipeline


def test_extract_transcript_script_omits_reasoning(tmp_path):
    mod = _load_script("scripts/memory/extract-transcript.py", "_et")

    path = tmp_path / "20260908_120000_test.json"
    path.write_text(json.dumps(_session()))

    _sid, _ts, entries = mod.process_lloyd_session(str(path), 0)
    rendered = "\n".join(f"{role}: {text}" for _ts2, role, text in entries)

    assert SECRET not in rendered
    assert "How much disk is left?" in rendered


def test_backfill_session_markdown_omits_reasoning(tmp_path, monkeypatch):
    mod = _load_script("scripts/memory/backfill-session-markdown.py", "_bsm")
    monkeypatch.setattr(mod, "VAULT_SESSIONS_DIR", tmp_path / "out")
    # The exporter reports its path relative to home; keep it in tmp.
    monkeypatch.setattr(mod.Path, "home", classmethod(lambda cls: tmp_path))

    src = tmp_path / "20260908_120000_test.json"
    src.write_text(json.dumps(_session()))

    _name, written, _reason = mod.export_session(src)

    assert written, "the fixture must produce a real export"
    body = next((tmp_path / "out").rglob("*.md")).read_text()
    assert SECRET not in body
    assert "How much disk is left?" in body


# ------------------------------------------------------------ recall corpus


def test_session_recall_corpus_omits_reasoning(tmp_path, monkeypatch):
    """`session_recall` searches sessions; a hit on a thought is a leak too."""
    from agent_mcp import session as session_mod

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "20260908_120000_test.json").write_text(json.dumps(_session()))
    monkeypatch.setattr(session_mod, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(session_mod, "_session_index_cache", None)

    index = session_mod._load_session_index()
    blob = json.dumps(index)

    assert index, "the fixture session must be indexed"
    assert SECRET not in blob
    assert "disk" in blob.lower()


# ----------------------------------------------------------- trajectories


def test_trajectory_extraction_omits_reasoning(tmp_path):
    mod = _load_script("scripts/extract-trajectories.py", "_xt")

    src = tmp_path / "20260908_120000_test.json"
    src.write_text(json.dumps(_session()))

    traj = mod.parse_session(src)
    assert SECRET not in json.dumps(traj, default=str)


# ------------------------------------------------- which half does the work


def test_the_role_is_what_protects_the_transcripts():
    """The protection is the role, not the empty content block.

    Every producer branches on role and none has a `"thinking"` case, so
    the row renders as nothing whatever its content holds. This is the
    counterfactual in miniature: the same entry under an `assistant` role
    is reproduced verbatim, which is what the real one would do if the
    role were ever "simplified" away.
    """
    from app.post_capture import _build_capture_transcript

    row = _build_thinking_entry(
        _Turn(), f"deliberating {SECRET}", 100, 0, 1, "2026-09-08T12:00:01")
    # Content populated *and* role intact: still nothing.
    leaky_content = dict(row, content=[{"type": "text", "text": row["reasoning"]}])
    assert SECRET not in _build_capture_transcript([leaky_content])

    # Role changed, content populated: reproduced. If this stops failing to
    # exclude, the producers have grown a `thinking` branch and the rows
    # above are no longer safe by role alone.
    leaky_role = dict(leaky_content, role="assistant")
    assert SECRET in _build_capture_transcript([leaky_role])
