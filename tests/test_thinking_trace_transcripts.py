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


# ------------------------------------ which half does the work


def test_the_role_is_what_protects_the_transcripts():
    """The protection is the role, not the empty content block.

    Every producer branches on role and none has a `"thinking"` case, so the row
    renders as nothing whatever its content holds. The assertions above all say
    `SECRET not in`, which seven of them can satisfy by rendering *nothing at
    all* — an empty transcript passes seven "not in" checks. So this test carries
    the `in` direction as well, for both builders in `app/post_capture.py`: the
    same entry under an `assistant` role with its content populated is
    reproduced verbatim. That is the counterfactual in miniature — what the real
    row would do if the role were ever "simplified" away — and it is also the
    positive control that says the two builders render ordinary text at all,
    rather than returning "" and passing every exclusion test beside it.
    """
    from app.post_capture import _build_capture_transcript, _build_fact_transcript

    row = _build_thinking_entry(
        _Turn(), f"deliberating {SECRET}", 100, 0, 1, "2026-09-08T12:00:01")
    # Content populated *and* role intact: still nothing, from either builder.
    leaky_content = dict(row, content=[{"type": "text", "text": row["reasoning"]}])
    assert SECRET not in _build_capture_transcript([leaky_content])
    assert SECRET not in _build_fact_transcript([leaky_content])[0]

    # Role changed, content populated: reproduced by both. If either stops
    # rendering it, the producers have grown a `thinking` branch (or lost the
    # role dispatch) and the rows above are no longer safe by role alone.
    leaky_role = dict(leaky_content, role="assistant")
    assert SECRET in _build_capture_transcript([leaky_role]), (
        "the summary builder stopped rendering an assistant row, so every "
        "`SECRET not in` above is passing on an empty transcript"
    )
    assert SECRET in _build_fact_transcript([leaky_role])[0], (
        "the extraction builder stopped rendering an assistant row, so its "
        "exclusion tests are passing on an empty transcript"
    )


# ------------------------------------ the window the fact extractor is handed
#
# Same builder as above, one argument more. `_build_capture_transcript` truncated
# to 4000 characters by keeping the first 2000 and the last 2000 of the WHOLE
# session, so on a long conversation everything in the middle — which is where
# nearly all of a spoken conversation is — was invisible to the model however
# recently it was said. #1159 added `start`, the per-session fact watermark, and
# the budget now applies to the slice from there. These two tests are one claim
# read from each side: a late fact is unreachable at `start=0`, and is inside the
# text once the extractor passes its watermark.


#: Index, from the end, of the user turn that carries the late fact: six
#: exchanges back. The last exchange cannot be the one, because the pre-fix
#: builder always keeps the LAST 2000 characters of the transcript, so a fact in
#: the final turn was never the case the budget hid. Six back puts it past the
#: tail — which is the actual defect.
LATE_FROM_END = 12

MARKER = "LATE-FACT-MARKER-ZQX"


def _long_session(turns: int = 30, *, late: bool = True) -> dict:
    """A chat long enough that the 4000-char budget is the thing doing the cutting.

    Thirty filler exchanges with the durable sentence sitting six exchanges back
    from the end, and sized so the whole transcript runs to several times the
    budget: if the fixture fit inside it, the `start=0` test below would pass for
    the wrong reason. `late=False` is the same filler with no durable sentence,
    for the test that only compares bytes.
    """
    messages: list[dict] = []
    for n in range(1, turns + 1):
        filler = " ".join(f"filler{n}_{w}" for w in range(30))
        messages.append({
            "id": f"u{n}", "role": "user",
            "content": [{"type": "text", "text": filler}],
            "timestamp": "2026-09-08T12:00:00",
        })
        messages.append({
            "id": f"a{n}", "role": "assistant",
            "content": [{"type": "text", "text": f"{filler} — done."}],
            "timestamp": "2026-09-08T12:00:01",
        })
    if not late:
        return {
            "session_id": "20260908_120000_plainrun",
            "created_at": "2026-09-08T12:00:00",
            "model": "primary",
            "messages": messages,
        }
    late_msg = messages[-LATE_FROM_END]
    assert late_msg["role"] == "user", "the fixture's late turn must be a user turn"
    late_msg["content"] = [{"type": "text", "text": (
        "Note for later: the live voice room is the same session id as the chat "
        f"session already open, so extraction has to re-arm on that one. {MARKER}"
    )}]
    return {
        "session_id": "20260908_120000_longrun",
        "created_at": "2026-09-08T12:00:00",
        "model": "primary",
        "messages": messages,
    }


def _fact_window(messages: list, start: int):
    from app.post_capture import _build_fact_transcript
    return _build_fact_transcript(messages, start=start)


def test_a_late_fact_is_outside_the_window_a_session_start_gets():
    """The failure mode, pinned as a property of the summary-shaped builder.

    `_build_capture_transcript` keeps the first 2000 and last 2000 characters of
    the whole transcript. The marker sentence is six exchanges from the end —
    past the tail that survives — and the assertion that the transcript came back
    at full budget is what makes that a truncation rather than a short fixture.
    This is why extraction does not use that builder: a window that hides its own
    middle cannot say what it covers.
    """
    from app.post_capture import _build_capture_transcript

    messages = _long_session()["messages"]
    whole = _build_capture_transcript(messages)

    # 2000 + the 19-character seam + 2000: the budget was fully spent, so a
    # message missing from the middle is missing because of the budget.
    assert len(whole) == 4019, (
        f"the fixture must have overflowed the 4000-char budget and come back "
        f"head-and-tail truncated, got {len(whole)} chars"
    )
    assert "[...truncated...]" in whole
    assert MARKER not in whole, (
        "the durable sentence survived the whole-session window, so the fixture "
        "is too short to show what #1159 was losing"
    )


def test_the_watermark_window_reaches_the_late_fact():
    """Clause 3: the window the extractor is handed contains the late fact.

    Three things at once, each failing on its own defect: a slice starting at the
    watermark has to include the message the whole-session window drops, it must
    not re-include one the watermark says is already read, and the covered count
    has to be the builder's own and not `len(messages)` — a trimmed slice stamped
    as full coverage is the loss made permanent.
    """
    from app.post_capture import FACT_TRANSCRIPT_BUDGET

    messages = _long_session()["messages"]
    watermark = 44                       # six user turns of unseen tail

    window, covered = _fact_window(messages, watermark)

    assert len(window) <= FACT_TRANSCRIPT_BUDGET, (
        f"the extractor was handed {len(window)} chars over a "
        f"{FACT_TRANSCRIPT_BUDGET}-char budget"
    )
    assert MARKER in window, (
        "a durable fact stated six exchanges from the end of a 30-turn session "
        "did not reach the model"
    )
    assert "filler21_1 " not in window, (
        "content before the watermark was re-sent, so the slice is still the "
        "whole transcript wearing a start argument"
    )
    assert watermark < covered < len(messages), (
        f"covered={covered} claims the whole {len(messages)}-message session from "
        f"a {len(window)}-char slice"
    )
    assert covered == 55, (
        f"the builder reported covering {covered} messages where the slice it "
        "returned ends at 55; the count is what the session file is stamped with"
    )


def test_the_late_fact_reaches_the_model_by_draining_from_the_session_start():
    """The guarantee that actually matters for a fact at the end of a long chat.

    One budget cannot hold a 30-turn session, so the promise is not "the tail is
    in the first window" but "successive passes advance until the tail has been
    shown". Each slice is capped at `FACT_TRANSCRIPT_BUDGET`, each pass advances
    strictly, and the marker is inside one of them. What this cannot show is a
    live engine's answer over the real traffic — that is the post-landing check
    the item leaves to days of sessions, not something a hermetic test can
    substitute for.
    """
    from app.post_capture import FACT_TRANSCRIPT_BUDGET

    messages = _long_session()["messages"]
    watermark, hits, passes = 0, [], 0

    while passes < 20:
        window, covered = _fact_window(messages, watermark)
        if covered <= watermark:
            break
        assert len(window) <= FACT_TRANSCRIPT_BUDGET
        assert covered > watermark, "a pass covered nothing and advanced anyway"
        hits.append(MARKER in window)
        watermark = covered
        passes += 1

    assert any(hits), (
        f"{passes} extraction windows over the session and none of them contained "
        "the durable fact"
    )
    assert watermark == len(messages), (
        f"the drain stopped at {watermark} of {len(messages)} messages, leaving "
        "the tail permanently unseen"
    )


def test_the_summary_builder_still_returns_the_bytes_the_eval_hash_is_built_from():
    """`eval/secondary_routing_eval.py` hashes this builder's output; equivalence
    is not enough, it needs the same bytes.

    Pinned here as the head/seam/tail construction itself, so a future "tidy" of
    the truncation — a different marker, a different split — fails against the
    eval's own recorded input hash and against this test, not only in review.
    """
    from app.post_capture import _build_capture_transcript

    short = _long_session(turns=6, late=False)["messages"]
    rendered = _build_capture_transcript(short)

    assert rendered.startswith("USER: filler1_0"), "the head of the session is no longer kept"
    assert rendered.endswith("filler6_29 — done."), "the tail of the session is no longer kept"
    assert "[...truncated...]" not in rendered, (
        "a session inside the budget must be returned whole, seam included"
    )

    long = _build_capture_transcript(_long_session(turns=30, late=False)["messages"])
    assert len(long) == 4019 and "\n[...truncated...]\n" in long, (
        "the truncation seam changed, which re-bases every input the eval has "
        f"hashed; got {len(long)} chars"
    )

    assert _build_capture_transcript([]) == ""
    assert _build_capture_transcript([{"role": "user", "content": "x" * 5000}]) == (
        "USER: " + "x" * 600
    ), "the 600-char per-line cap moved, which re-bases every rendered transcript"
