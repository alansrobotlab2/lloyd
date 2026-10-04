"""Is a shipped prompt-fix LIVE in the process answering right now?  (#2176)

Two facts about a prompt-side fix, and only one of them was checkable. SHIPPED:
its commit is an ancestor of HEAD. LIVE: the process that answered the turn
assembled a prompt containing it. The reflection chain needed LIVE for two cycles
and could not get it — "was `_data_home_hint` in the live prompt when
`~/lloyd/workers.db` was re-created at 14:00 PDT 2026-10-03?" The shipped half is
provable (`5b8d3e8d`, authored 2026-10-02T09:10:54-07:00, an ancestor of HEAD) and
so is the restart (2026-10-03 20:59:09 PDT, after the creation); what nobody could
say is whether the running prompt had the paragraph.

Neither existing instrument says. `log_prompt_size` emits `name=<chars>c/<est>t`
and nothing else; `app/component_manifest.py` keeps `sha256:` + `bytes` per
component under a retention policy that explicitly retains no content, pinned by
`tests/test_component_manifest.py::test_no_written_line_carries_component_text`.
A digest cannot be grepped for a paragraph, and the session transcript is not a
third witness either: in
`~/lloyd-data/sessions/20261004_023513_autotriage_33f5.json` `messages[0]` is the
USER turn (16,926 chars) and does not contain the hint's paragraph — a worker's
prompt is greppable only because dispatch hands it over as a user message.

So the fix is presence: a declared marker name plus one bit per assembly, in a log
line beside `PROMPT_BUDGET`. These seven nodes pin the seven things that make that
safe — one record per build, computed on the returned string, from data rather
than from a branch, seeded with the marker that answers the 2026-10-03 question,
carrying no prompt text, never reaching the turn when it fails, and with the
transcript figures the item quotes pinned in committed bytes rather than left in a
prose claim.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest

from app import component_manifest as cm
from app import prompt_builder as pb

ROOT = Path(__file__).resolve().parents[1]

TAG = "PROMPT_MARKERS"


@pytest.fixture
def turn_vault(tmp_path, monkeypatch):
    """A vault this file owns, so every byte of the built prompt is predictable.

    The same shape as `fake_turn_vault` in `tests/test_prompt_surface_budget.py`:
    point the canon paths at a temporary directory and no live vault file is read.
    It has to be real content rather than a stub of the return value, because two
    nodes below probe the prompt as a string — one for a needle straddling a join,
    one for a sentinel planted in a component — and a mocked prompt would let both
    pass while the builder was doing something else entirely.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "SOUL.md").write_text("# Lloyd Operating Contract\n\nBE BOUNDED\n",
                                   encoding="utf-8")
    (vault / "MEMORY.md").write_text("# Lloyd Long-Term Memory\n\nA fact.\n",
                                     encoding="utf-8")
    (vault / "USER.md").write_text("# User\n\nprefers short answers\n", encoding="utf-8")
    monkeypatch.setattr(pb, "_CANON_SOUL_PATH", vault / "SOUL.md")
    monkeypatch.setattr(pb, "_CANON_MEMORIES_DIR", vault)
    monkeypatch.setattr(pb, "_CANON_SKILLS_DIRS", [tmp_path / "no-skills"])
    monkeypatch.delenv("LLOYD_OVERLAY_DIR", raising=False)
    return vault


def _marker_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if TAG in r.getMessage()]


def test_one_record_per_assembly_names_every_declared_marker(turn_vault, caplog):
    """Clause 1 (#2176), the record half: exactly one line per build, one field per
    declared marker, and none when no build was measured.

    "None without a `session_id`" is half the clause, not a detail: the witness shares
    `log_prompt_size`'s call site and so its condition, and a second record per turn on
    every un-instrumented build would be log noise that buys nothing — an unattributed
    line cannot answer "which process" either. The `session=` and `platform=` fields are
    what make it an answer: the prompt differs per platform (a worker turn drops
    USER.md), so presence is a claim about one session, not about the code.
    """
    with caplog.at_level(logging.INFO, logger=pb.logger.name):
        pb.build_system_prompt(session_id="turn1")
        pb.build_system_prompt()

    lines = _marker_lines(caplog)
    assert len(lines) == 1, lines
    assert "session=turn1" in lines[0] and "platform=user" in lines[0], lines[0]
    for name in pb.PROMPT_MARKERS:
        assert re.search(rf"(?:^|\s){name}=(present|absent)(?:\s|$)", lines[0]), lines[0]


def test_presence_is_computed_on_the_string_that_is_returned(turn_vault, caplog,
                                                            monkeypatch):
    """Clause 1 (#2176), the semantics half: `present` iff the RETURNED string has it.

    Two probes registered, and no builder touched. `cross-join-probe` is a needle sliced
    across the `\\n\\n` between two ADJACENT COMPONENTS, which the build hands over as a
    dict at `component_manifest.note_components` — the seam the item names as the trap,
    because that dict is what `log_prompt_size` is given and it is the obvious thing to
    ask. A needle inside one component proves nothing: `_data_home_hint`'s paragraph and
    the platform note before it are both inside the single `harness_hints` component, so
    a per-component scan finds either one. Spanning two components is what a per-component
    scan cannot find, and the node asserts that directly (`in_no_component`) before it
    trusts the probe — the assertion is the fixture checking itself, so the day the
    builder groups components differently the node says "this probe no longer
    discriminates" instead of passing on a weaker property.

    `nowhere-probe` is the other half: without a case that must read `absent`, a
    hard-coded `present` passes every other node in this file.
    """
    captured: dict = {}
    real_note = cm.note_components

    def note(session_id, components):
        captured.update(components)
        return real_note(session_id, components)

    monkeypatch.setattr(cm, "note_components", note)
    prompt = pb.build_system_prompt(session_id="probe")

    parts = [c for c in captured.values() if c]
    assert "\n\n".join(parts) == prompt, (
        "the components no longer join into the returned string, so a boundary between "
        "two of them is no longer a join in the prompt")
    cross = next(
        (a[-8:] + "\n\n" + b[:12] for a, b in zip(parts, parts[1:])
         if a + "\n\n" + b in prompt),
        None,
    )
    assert cross, "no two adjacent components meet in the prompt: no seam to probe"
    in_no_component = all(cross not in c for c in parts)
    assert in_no_component and cross in prompt, (
        "the probe no longer straddles a component join, so it can no longer tell a "
        "witness that reads the parts from one that reads the prompt")

    monkeypatch.setitem(pb.PROMPT_MARKERS, "cross-join-probe", cross)
    monkeypatch.setitem(pb.PROMPT_MARKERS, "nowhere-probe",
                        "no such text in any prompt 2176")

    with caplog.at_level(logging.INFO, logger=pb.logger.name):
        again = pb.build_system_prompt(session_id="turn2")

    assert again == prompt, "the second build differed, so the probes compare nothing"
    line = _marker_lines(caplog)[-1]
    assert "cross-join-probe=present" in line, line
    assert "nowhere-probe=absent" in line, line


def test_a_marker_registered_by_a_test_appears_in_the_next_record(turn_vault, caplog,
                                                                  monkeypatch):
    """Clause 2 (#2176): the registry is all the feature knows, so registering is data.

    The whole of what this test does is add one key to `PROMPT_MARKERS`. It does not
    touch `build_system_prompt`, `marker_liveness` or `log_marker_liveness`, and the
    record it then reads names the new marker — which is the property that has to
    survive the next twenty prompt-fixes: adding a witness is one line of data, never a
    new branch in the builder, and a marker that needed code would be a marker nobody
    registers under time pressure at 2am.

    The name list is asserted exactly, not as a substring search: a record whose fields
    drift out of step with the registry (a marker dropped, a stale one kept) is the
    witness lying about coverage, which is worse than no witness.
    """
    assert isinstance(pb.PROMPT_MARKERS, dict)
    assert all(isinstance(k, str) and isinstance(v, str)
               for k, v in pb.PROMPT_MARKERS.items()), pb.PROMPT_MARKERS
    assert pb.PROMPT_MARKERS, "an empty registry makes every node below vacuous"

    monkeypatch.setitem(pb.PROMPT_MARKERS, "clause-two-probe", "BE BOUNDED")
    with caplog.at_level(logging.INFO, logger=pb.logger.name):
        prompt = pb.build_system_prompt(session_id="turn3")

    assert "BE BOUNDED" in prompt
    line = _marker_lines(caplog)[-1]
    assert "clause-two-probe=present" in line, line
    named = [n for n, _ in re.findall(r"([\w.-]+)=(present|absent)", line)]
    assert named == list(pb.PROMPT_MARKERS), (
        f"the record names {named}, the registry declares {list(pb.PROMPT_MARKERS)}")


def test_the_shipped_marker_names_the_data_home_paragraph(turn_vault, caplog):
    """Clause 3 (#2176): the registry as shipped already answers 2026-10-03's question.

    Two directions pinned. The literal must occur in `_data_home_hint()`'s return value,
    or the marker is decorative and the incident question is still unanswerable by grep.
    And it must occur NOWHERE ELSE in the prompt this build assembles, once the hint's own
    paragraph is removed — a marker that several parts satisfy reports `present` for
    reasons that have nothing to do with the fix it witnesses, which is how a liveness bit
    becomes worse than the digest it replaced: a digest at least never claims to identify
    a paragraph.

    The scope of the second direction is what a node can reach: the prompt is built from
    the fixture vault, so the parts it rules out are everything `prompt_builder` itself
    contributes plus the synthetic stand-ins. It cannot rule out a sentence a person later
    writes into the live `SOUL.md` or a memory file, and no node should read the live vault
    to try. What closes that residual is the literal's shape — 61 characters copied out of
    the one function it witnesses, not a phrase — and the fact that a collision would show
    up as a false `present` in the log, which is the same failure the marker is there to
    make visible rather than hide.
    """
    hint = pb._data_home_hint()
    hits = [name for name, needle in pb.PROMPT_MARKERS.items() if needle and needle in hint]
    assert hits, (
        "no declared marker occurs in `_data_home_hint()`, so the item's own question "
        "is still not a grep"
    )
    assert "data-home-hint" in hits, hits

    with caplog.at_level(logging.INFO, logger=pb.logger.name):
        prompt = pb.build_system_prompt(session_id="turn4")

    assert "data-home-hint=present" in _marker_lines(caplog)[-1], _marker_lines(caplog)
    rest = prompt.replace(hint, "", 1)
    assert pb.PROMPT_MARKERS["data-home-hint"] not in rest, (
        "the literal is satisfied by some other part of the prompt, so `present` would "
        "not be evidence that the data-home paragraph shipped into this build"
    )


def test_the_record_carries_no_component_text(turn_vault, caplog):
    """Clause 4 (#2176): the witness is a bit about the prompt, not a copy of it.

    `component_manifest` decided its confidentiality boundary in writing — digests and
    byte counts, no content, "a sentinel string placed inside a component cannot be
    found in the store" — and a witness that logged prompt text to be more useful would
    quietly move that boundary into `server.err`, a file the item itself expects to
    survive in as `.8/.9/.10`. So the sentinel is planted here in the same shape the
    manifest test plants its own sentinel, and asserted absent from the
    record twice over: the sentinel itself, and every declared literal. The second is
    the one that constrains the design, because a marker's literal IS prompt text — an
    implementation that echoed `data-home-hint="Runtime data: …"` because it looked
    helpful would carry a paragraph of the identity surface into every log line.

    The `PROMPT_BUDGET` line is checked on the way past: it shares this call site, and
    the two instruments are read together.
    """
    sentinel = "SOUL-SENTINEL-prompt2176"
    (turn_vault / "SOUL.md").write_text(
        f"# Lloyd Operating Contract\n\n{sentinel}\n", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger=pb.logger.name):
        prompt = pb.build_system_prompt(session_id="turn5")

    assert sentinel in prompt, "the sentinel never reached a component: nothing tested"
    lines = _marker_lines(caplog)
    assert len(lines) == 1, lines
    assert sentinel not in lines[0], lines[0]
    for name, needle in pb.PROMPT_MARKERS.items():
        assert needle not in lines[0], (
            f"{name}'s literal is prompt text and it is in the record: the witness "
            "logged what it was asked about instead of a bit about it"
        )
    budget = [r.getMessage() for r in caplog.records
              if "PROMPT_BUDGET" in r.getMessage()]
    assert len(budget) == 1 and sentinel not in budget[0], budget


def test_a_raising_witness_never_reaches_the_turn(turn_vault, caplog, monkeypatch):
    """Clause 5 (#2176): instrumentation that can break the turn is not instrumentation.

    `PROMPT_MARKERS` exists to answer a question after an incident, so it runs on every
    turn of a prompt that the operator cannot afford to lose. The emitter is monkeypatched
    to raise — the shape a future version produces, e.g. a registry entry that is `None`
    or a logger that throws on a broken handler — and the prompt that comes back must be
    the byte-identical string the healthy path returned, with the failure visible at
    WARNING rather than swallowed, and with `PROMPT_BUDGET` still on the wire: the new
    instrument sits last at the call site precisely so its failure cannot take either
    older instrument down with it.
    """
    with caplog.at_level(logging.INFO, logger=pb.logger.name):
        healthy = pb.build_system_prompt(session_id="turn6")

    def boom(prompt, **kwargs):
        raise RuntimeError("witness down")

    monkeypatch.setattr(pb, "log_marker_liveness", boom)
    with caplog.at_level(logging.INFO, logger=pb.logger.name):
        broken = pb.build_system_prompt(session_id="turn6")

    assert broken == healthy, "a failing witness changed the prompt it was watching"
    assert any(TAG in r.getMessage() and "witness down" in r.getMessage()
               for r in caplog.records), "a witness failure nobody can see is a gap"
    assert len([r for r in caplog.records if "PROMPT_BUDGET" in r.getMessage()]) == 2, (
        "the size line stopped firing because the marker line raised")


#: The one transcript whose inspection produced this item's "a session log is not a third
#: witness either", committed under `tests/fixtures/` so the gate that has no vault can
#: still read it. `tests/fixtures/.gitignore` explains why the bytes live in two places.
WITNESS = ROOT / "tests" / "fixtures" / "session_witness_20261004_023513_autotriage_33f5.json"


def test_the_committed_witness_bytes_still_carry_the_measured_gap():
    """Clause 6 (#2176): the figures behind this item's third-witness claim, re-derived.

    The durable witness is the vault file
    `backlog/data/20261004_023513_autotriage_33f5.json` (vault commit `3f160dc7`, landed by
    this round) and the bytes are byte-identical here at commit time — 34 lines, 4,159
    bytes, which is what the node below re-derives rather than a digest that only restates itself. A node cannot read the vault one, because the gate runs the suite
    with `HOME` pointed at a round home where `~/obsidian` does not exist, so the re-derive
    happens against these committed bytes and the vault copy is checked by eye with::

        wc -l < backlog/data/20261004_023513_autotriage_33f5.json   # -> 34
        sha256sum backlog/data/20261004_023513_autotriage_33f5.json \\
          tests/fixtures/session_witness_20261004_023513_autotriage_33f5.json  # -> one digest

    What is re-derived is what the item quotes about the transcript: `messages[0]` is the
    user turn, 16,926 chars, and does not contain the clause this registry's literal is
    cut from; the file's 39 messages hold no `system` role at all, and the paragraph
    appears only at indexes 23/24/28/32/36. Together those are why a transcript cannot
    answer "was the fix live": it records what was said, not what was assembled — and the
    two `PROMPT_BUDGET`-shaped instruments that DO record the assembly carry sizes and
    digests, which is the gap the emitter in this module closes.

    A digest pinned in code would only restate that the file is itself, so each figure is
    read out of the parsed extract instead. The extract's own line count, not the source's
    917, is the figure the round report quotes; `source_lines` and `source_bytes` are the
    source's, kept here so the source stays checkable while it survives.
    """
    raw = WITNESS.read_text(encoding="utf-8")
    assert raw.count("\n") == 34, f"`wc -l` of the witness is not 34: {len(raw)} B"
    assert len(raw.encode()) == 4159, "the committed witness is not the file #2176 quoted"
    w = json.loads(raw)

    assert w["source_lines"] == 917 and w["source_bytes"] == 113563, w
    assert w["message_count"] == len(w["roles_in_order"]) == 39, (
        "the role list and the message count disagree, so neither is the file's shape")
    assert "system" not in w["roles_in_order"], (
        "a transcript with a system message WOULD be a witness, and this claim is that "
        "this one has nothing to record the assembly with")

    assert w["messages_0_role"] == "user", w
    assert w["messages_0_text_chars"] == 16926, w
    assert w["data_home_hint_in_messages_0"] is False, w
    assert w["data_home_hint_in_message_indexes"] == [23, 24, 28, 32, 36], w

    # The claim's own hinge, checked against the shipped code rather than restated. The
    # transcript was probed for the paragraph's OPENING clause; the registry witnesses a
    # longer slice of the same paragraph. The relation that has to hold is that the probe
    # is a prefix of the witnessed literal — if the two strings diverged, the transcript
    # had measured the absence of one thing and the log would report the presence of
    # another, which is the mistake this whole item is about.
    probe = w["data_home_hint_literal"]
    shipped = pb.PROMPT_MARKERS["data-home-hint"]
    assert shipped.startswith(probe), (
        f"the witness probed for {probe!r} and the registry witnesses {shipped!r}: "
        "different strings, so the transcript's absence says nothing about this marker")
    assert shipped in pb._data_home_hint(), (
        "the registry's literal is not in the function it claims to witness")
