"""What the djev prose is allowed to say about the seams it describes.

Two items live here, both because a description was read as a fact. The first
is `architecture/djev.md`'s record of the #1324 shadow-log leak: the behaviour
that closes the leak is pinned in `tests/test_djev_shadow_isolation.py`, and
this file pins the *other* half — what the doc may now say, and the numbers
inside that prose.

The second is #1372: `app/djev_shadow.py`, `agent_mcp/vault.py` and
`config.yaml` all described the `rerank` seam as a live one production passes
through, while since #1336 the dispatch has made it unreachable except under
the kill switch. Those assertions sit at the bottom of this file because they
are the same shape as the ones above: they read words out of files rather than
running code, so they need no daemon, no engine and no fixture, and every
needle is a string a later reader can `grep` to re-measure.

It matters because the doc is where the item came from. §11 carried "Test runs
write to production's shadow log" as an open gap and named #1324, so any
architecture review that read §11 after the fix would have re-filed the item
from the doc's own text — an open-gap list is a to-do list, and a stale entry in
one is worse than no entry, because it looks like a decision. The same section
now carries counts (36 of 44 `dedupe` rows, 27 of 32 `rerank`, 63 fixture rows
left in the log) that name a fixture by its title; if that title is renamed in
`tests/test_backlog_dedupe.py` the counts silently stop meaning anything, so
those titles are pinned against the module that defines them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_mcp import vault

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "djev.md"

#: The fixture title in `tests/test_backlog_dedupe.py:58` whose 32 rows the doc
#: counts, and the one in `tests/test_backlog_spawn_loop.py:792` that accounts
#: for 4 more. Both are asserted against those modules below, so a rename that
#: would make the doc's counts unfalsifiable fails a test.
FIXTURE_DEDUPE = "http_fetch error body says only the status code on a quarter of calls"
FIXTURE_SPAWN = "graph_refresh is advertised but never called by any tool"

CLOSED_HEADING = "### Closed 2026-09-21 — test runs wrote to production's shadow log (#1324)"


def _section(text: str, heading: str) -> str:
    """From `heading` to the next `## ` line — one section, sliced out."""
    i = text.index(heading)
    j = text.find("\n## ", i + len(heading))
    return text[i: j if j != -1 else len(text)]


def _closed_record(text: str) -> str:
    """The Closed subsection of §11. Asserts its own absence, so a doc that
    dropped the record fails here instead of at some `[1]` index."""
    body = _section(text, "## 11. Known gaps")
    assert CLOSED_HEADING in body, (
        "§11 has no closed record for the shadow-log leak, which means its open-gap "
        "list is still describing #1324 as live — the exact state that re-files it")
    return body.split(CLOSED_HEADING, 1)[1]


def test_the_open_gaps_no_longer_describe_the_shadow_log_leak():
    """Clause 3, first half: `architecture/djev.md` §11 stops listing "Test runs
    write to production's shadow log" as an open gap, and stops pointing at the
    item that closed it.
    """
    open_part = _section(DOC.read_text(), "## 11. Known gaps").split(CLOSED_HEADING, 1)[0]
    assert "Test runs write to production's shadow log" not in open_part, (
        "§11 still lists the leak as an open gap")
    assert "#1324" not in open_part, "the open gaps still point at the item that closed it"
    assert "fixture" not in open_part.lower(), (
        "the open gaps still carry the fixture-row complaint this item closed")


def test_the_closed_record_carries_the_mechanism_the_check_and_the_residue():
    """Clause 3, second half: the replacement has to be usable, not merely
    present. Each needle is something a later reader needs and cannot re-derive:
    which fixture the counts refer to, why the paths are redirected instead of
    muted (reverting to `LLOYD_DJEV_SHADOW=0` would make
    `test_djev_rerank_arm.py::test_the_eval_mutes_the_shadow_recorder` pass with
    the `eval/run_eval.py` line under it deleted), why the session fixture never
    restores (a restore at session end is the drain window), the command that
    shows the leak is gone, and the rows still in the file poisoning
    `eval/djev/replay.py --floors`.
    """
    closed = _closed_record(DOC.read_text())
    for needle in ("`tests/conftest.py::_isolate_djev_shadow`",   # the fix, by name
                   "does not restore them",                       # the mechanism that matters
                   FIXTURE_DEDUPE,                         # the fixture the counts name
                   "Paths are redirected rather than",            # the decision…
                   "`LLOYD_DJEV_SHADOW=0` set",                   # …and its reason
                   "test_the_eval_mutes_the_shadow_recorder",     # the test a mute would vacuate
                   "$FH/.local/state/lloyd-djev/shadow.jsonl",    # the check
                   "**does not exist**",                          # its expected result
                   "**What is left, and it is a person's call.**"):  # the residue, still owed
        assert needle in closed, f"the closed record no longer carries {needle!r}"


def test_the_closed_record_still_names_the_two_things_reading_the_log():
    """Closing the leak does not clean the corpus, and the doc must not read as
    if it had. Two writers survive this change: the fixture rows already in the
    file, and the automod gate's tests rung, which runs pytest with `HOME` set to
    the real home — `scripts/automod/gate.py::_child_env` redirects the automod
    and guardian state dirs and sets no djev path (`grep -n DJEV
    scripts/automod/gate.py` → 0 hits). Under the gate a repointed `SHADOW_LOG`
    proves nothing about `Path.home()`, because `Path.home()` is the real home
    there; the property that holds is the line count of the real file across the
    run. Either fact goes missing and the next reader of §9.3 step 5 gets a
    fixture-dominated floor or a false clean bill.
    """
    closed = _closed_record(DOC.read_text())
    assert "63" in closed, "the closed record no longer says how many fixture rows are still in the log"
    assert "replay.py --floors" in closed, "the closed record no longer names what the residue poisons"
    assert "_child_env" in closed, "the closed record no longer names the gate as the un-repointed runner"


def test_the_calibration_procedure_points_at_the_quarantine_first():
    """§9.3 step 5 is where a practitioner goes looking, not §11. It now says the
    floor comes *after* quarantining the pre-fix rows, so the calibration
    procedure cannot be followed in the order that produces a fixture-dominated
    floor.
    """
    doc = DOC.read_text()
    step = doc[doc.index("### 9.3 Adding a seam"):doc.index("### 9.4 Turning a schema")]
    assert step.count("#1324") == 1, "§9.3 step 5 no longer points at the quarantine the floor needs first"
    assert "fixture" in step, "§9.3 step 5 no longer says why quarantine precedes --floors"


def test_the_fixture_titles_the_doc_counts_still_exist():
    """The doc's counts are only checkable while the titles they name are still
    the titles the fixtures write. `tests/test_backlog_dedupe.py` and
    `tests/test_backlog_spawn_loop.py` write the first string into `meta.name`
    and the second into the backlog `name` of every row their dedupe seam
    shadows, which is how 36 of the log's 44 `dedupe` rows are attributable.
    Rename either and the doc's numbers become prose nobody can re-measure.
    """
    doc = DOC.read_text()
    closed = _closed_record(doc)
    assert FIXTURE_DEDUPE in closed, "the closed record no longer names the fixture it counts"
    assert FIXTURE_DEDUPE in (ROOT / "tests" / "test_backlog_dedupe.py").read_text(), (
        f"{FIXTURE_DEDUPE!r} is no longer the fixture title in test_backlog_dedupe.py, "
        "so the row counts in the closed record are no longer checkable")
    assert FIXTURE_SPAWN in (ROOT / "tests" / "test_backlog_spawn_loop.py").read_text(), (
        f"{FIXTURE_SPAWN!r} is no longer the fixture title in test_backlog_spawn_loop.py")


# ── #1372: the rerank seam's prose says when it can fire ────────────────────
#
# The item was filed from prose, not from code. `app/djev_shadow.py` opened with
# "Three seams call this — reranking in `agent_mcp/vault.py`", `config.yaml`
# still calls it "The lead seam: both `vault_recall` and `memory_ops.recall`
# pass through it", and `architecture/djev.md` says the seam runs "on every
# recall". Since #1336 none of those is true: djev IS the recall's ranker, so
# the hook sits under an `elif` the first arm never reaches, and `rerank`'s last
# production row is 2026-09-21T08:57:40Z while `dedupe` and `entity` keep
# flowing. A description that presents a structurally dark seam as a live one is
# what makes its silence unreadable — so the descriptions are asserted, not just
# corrected once.

SHADOW_SRC = ROOT / "app" / "djev_shadow.py"
VAULT_SRC = ROOT / "agent_mcp" / "vault.py"


def _module_docstring(path: Path) -> str:
    """The module docstring of `path`, by splitting on the delimiters rather
    than importing: this test is about the words, and it must not need a live
    qmd daemon or djev engine to read them."""
    text = path.read_text()
    body = text.split('"""', 2)
    assert len(body) >= 3, f"{path.name} no longer opens with a module docstring"
    return body[1]


def test_the_shadow_docstring_names_when_the_rerank_seam_can_fire():
    """Clause 3. The replacement has to carry the condition, not just drop the
    claim: the kill switch pulled to `qmd` with the engine still answering, and
    structurally dark while djev is the ranker."""
    head = _module_docstring(SHADOW_SRC)
    assert "Three seams call this" not in head, (
        "the module still opens by naming three live callers, one of which "
        "cannot reach the recorder while djev is the recall's ranker")
    for needle in ("kill switch", "qmd", "dark"):
        assert needle in head, (
            f"the module docstring no longer says the rerank seam is dark "
            f"except under the kill switch to qmd — it lost {needle!r}")


#: Every file that describes the seams: the recorder, the caller, and the doc
#: the #1324 record above is guarded in. #1372 is a defect about descriptions,
#: so the scan covers all three rather than only the two the clause names.
SEAM_DESCRIPTIONS = [SHADOW_SRC, VAULT_SRC, DOC]


@pytest.mark.parametrize("path", SEAM_DESCRIPTIONS,
                         ids=["djev_shadow", "vault", "djev.md"])
def test_no_djev_description_calls_rerank_a_seam_production_reaches(path):
    """Clause 2, second half. "Lead seam" was the phrase in two of these files —
    `app/djev_shadow.py`'s eval-muting paragraph and
    `vault._djev_shadow_rerank`'s own docstring ("The lead seam of the three,
    because every `vault_recall` passes through it") — and "on every recall" was
    in the doc, two sections from the one that knew better (§8.1: "The shadow
    seam does not run when djev ranks"). Those are the claims that make a dark
    seam read as a quiet one, so the needles are asserted, not just corrected
    once: a rank or a frequency is where a stale seam description shows up."""
    text = path.read_text().lower()
    needles = ["lead seam", "the three shadow seams"]
    if path != VAULT_SRC:
        # `vault.py` is 2,000 lines and says "on every recall" about the #1335
        # latency of the graph arm, which is true of it; the frequency claim
        # that matters there is the hook's own, checked just below.
        needles.append("on every recall")
    for needle in needles:
        assert needle not in text, (
            f"{path.name} still says {needle!r} about the rerank seam. While "
            "djev is the recall's ranker the dispatch never reaches it, so state "
            "the condition it fires under instead of a volume or a rank.")
    if path is VAULT_SRC:
        import inspect
        hook = inspect.getsource(vault._djev_shadow_rerank).lower()
        for needle in ("on every recall", "passes through it", "unconditional"):
            assert needle not in hook, (
                f"`_djev_shadow_rerank` still claims it happens {needle!r}, "
                "which is the sentence that made the seam look live")


def test_the_seam_section_states_the_row_stream_it_left_behind():
    """The doc has to carry the measured end of the stream, because "dark" is
    only checkable against something: the newest `rerank` row is a fact a reader
    can re-measure in `shadow.jsonl`, and the day it stopped is the day #1336
    landed."""
    text = DOC.read_text()
    section = text[text.index("### 6.1"):text.index("### 6.2")]
    assert "structurally dark" in section, (
        "§6.1 no longer says the seam is unreachable rather than idle")
    assert "2026-09-21T08:57:40" in section, (
        "§6.1 lost the newest-rerank-row timestamp, which is the sentence that "
        "makes 'dark' a claim someone else can re-measure")
    assert "kill switch" in section, "§6.1 no longer names what would lift it"


def test_the_shadow_hook_says_it_runs_only_under_the_other_ranker():
    """`vault._djev_shadow_rerank` is the one function a reader of the dispatch
    lands on, so its docstring carries the firing condition."""
    import inspect
    doc = inspect.getsource(vault._djev_shadow_rerank)
    assert "qmd" in doc, "the hook's docstring no longer names the ranker it needs"
    assert "djev is the recall's ranker" in doc or "djev ranks" in doc, (
        "the hook's docstring no longer says it is dark while djev ranks")


def test_the_stats_docstring_names_only_a_reader_that_exists():
    """`stats()` says the tool route is its only production reader, and used to
    name `/state` as a second one that never read it — the small version of this
    item's whole shape, a description carrying a consumer nobody wired up. So
    the sentence is pinned: add a reader and the test tells you to say so."""
    import inspect

    from app import djev_shadow
    doc = inspect.getsource(djev_shadow.stats)
    assert "djev_status" in doc, "stats() no longer names the route that reads it"
    assert "/state" not in doc, (
        "stats() advertises /state as a reader again. Grep for its call sites "
        "before adding a consumer back — the claim, not the consumer, is what "
        "has to be true here.")
    assert "djev_shadow.stats()" in (ROOT / "agent_mcp" / "djev.py").read_text(), (
        "the named reader stopped calling stats(), so the docstring is now the "
        "false half")
    readers = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "app").rglob("*.py")
                     if p.name != "djev_shadow.py" and "djev_shadow.stats()" in p.read_text())
    assert readers == [], (
        f"stats() gained a reader under app/ that its docstring does not name: {readers}")


def test_the_rerank_arm_says_it_applies_only_under_qmd_ranking():
    """Clause 2, first half, on the code side: the arm's own docstring and the
    note a caller gets must both name the ranker that lets it run."""
    import inspect
    assert "qmd" in inspect.getsource(vault._djev_rerank_pool), (
        "`_djev_rerank_pool` no longer says it is reached only when qmd ranks")
    assert "qmd" in vault.RECALL_ARM_UNUSED_NOTE, (
        "the unused-knob note no longer tells the caller which ranker would "
        "have applied its knob")


def test_the_recall_schema_does_not_advertise_the_arm_knobs():
    """Clause 2, schema half, as it stands after `d1b2f8e` took the seven eval
    knobs out of the `vault_recall` schema: an undeclared knob is now reported
    as stripped rather than silently obeyed, and the schema itself offers
    neither arm knob. If one is re-added it has to carry the condition in its
    own description — a parameter whose description promises an effect the
    dispatch cannot produce is how this item got filed."""
    import asyncio

    tool = next(t for t in asyncio.run(vault.list_tools()) if t.name == "vault_recall")
    props = tool.input_schema["properties"]
    for knob in ("djev_rerank", "djev_rerank_top"):
        assert knob not in props, (
            f"{knob} is advertised as a `vault_recall` parameter again, and its "
            "description cannot promise an effect the dispatch refuses whenever "
            "djev is the ranker — which is the sentence #1372 was filed from. "
            "It belongs in RECALL_EVAL_KNOBS, where a client that sends it gets "
            "told the knob was stripped.")
        assert knob in vault.RECALL_EVAL_KNOBS, (
            f"{knob} left RECALL_EVAL_KNOBS, so call_tool would let a client set it")
