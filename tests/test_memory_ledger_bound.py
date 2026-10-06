"""#2212 — a memory ledger is bounded by the file it audits, not by the shared
topic-file ceiling.

A live ledger holds one row per loaded index line, so the bytes it needs is a
function of the file it audits — not of the topic directory it happens to live
in. Measured 2026-10-05: `MEMORY.md` carries 90 index lines and the
`memory-md-ledger` mean row is 675 B, so full coverage needs ~60,750 B while the
shared 32,768 B topic ceiling holds ~48 rows — coverage would stop at little over
half the file, mid-audit, on a night that did nothing wrong.

So the guard derives a ledger's bound: 3 x `prompt_surface.memory_ceiling()` of
the audited file, which is 76,800 B for `memory-md-ledger` (3 x 25,600) and
49,152 B for `user-md-ledger` (3 x 16,384). Every other topic file keeps
`TOPIC_FILE_CEILING_BYTES` = 32,768, unchanged — a topic is pulled whole into a
tool result, a ledger never is (`step-2a-ter-curation.md` §1: it "is never loaded
into a prompt"), so the reason a topic is bounded does not apply to a ledger.

This file also pins the three things that make the derivation safe rather than a
bigger number: the ledger set is exactly the one the ledger script reads (`LEDGERS`),
so a renamed ledger cannot quietly keep the small bound; the prose a curator follows
does not state a ledger's cap as the shared topic ceiling; and the gate report this
item's own history is written from has committed bytes behind it (clause 6,
`fixtures/…gate-witness.jsonl`).

Clause 5's corpus is a vault file, so it is pinned twice: the ban runs over a
committed whole-file copy of the §5 skill (`fixtures/step_2a_ter_curation_witness_2212.md`)
inside the gate, and one node carries `live_vault` to prove the live file has not
drifted from it. The gate deselects that marker — a round home has no `~/obsidian` —
so run the drift check explicitly with `-m live_vault`.

Run: .venvs/lloyd/bin/python -m pytest tests/test_memory_ledger_bound.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from app import memory_ceiling as ceiling
from app import prompt_surface as ps

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "memory_ledger", ROOT / "scripts/memory/memory_ledger.py")
ml = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ml)

VALIDATOR = ROOT / "scripts" / "memory" / "validate_memory_index.py"
SKILL = (Path.home() / "obsidian" / "skills" / "nightly-reflection-knowledge-write"
         / "step-2a-ter-curation.md")
LEDGER_SCRIPT = ROOT / "scripts" / "memory" / "memory_ledger.py"

#: Every spelling of the shared topic ceiling, comma or no comma. `topic_size_error`
#: prints the comma form, so a check that covered only one of the two would miss a
#: prose line that dropped the separator.
THIRTY_TWO_K = re.compile(r"32[,_ ]?768")


def _states_a_ledger_cap_at_32768(line: str) -> bool:
    """True if THIS line attaches the shared topic ceiling to a ledger.

    Both halves must be on the same line: a spelling of 32,768 and the word
    "ledger". Markdown prose and a code comment each carry one claim per line, and a
    ledger's cap and an ordinary topic's cap are two different claims — so the unit
    the ban reads at is the line, not the file.

    The review of round SM_20261006_015104 named the wider version a defect: banning
    every spelling anywhere in the file "bans every 32[, _]?768 spelling in the whole
    skill file, while clause 5 permits non-ledger-ceiling claims there — the same
    'guard that fires on a correct sentence' the author explicitly avoids for
    `memory_ledger.py` by scoping to the docstring". `topic_ceiling`'s docstring and
    §1's ordinary-topic sentence are both correct prose about a number that did not
    move, and a guard that reddens them is a guard somebody switches off.

    Residual, stated rather than hidden: one line that CONTRASTS the two figures ("a
    ledger gets 3 x its audited file's ceiling, not the 32,768 B topic ceiling")
    carries both halves and fires. The conservative direction is deliberate — the
    failure mode is a red node and a rephrased line, never a curator silently told to
    stop an audit early.
    """
    return "ledger" in line.lower() and bool(THIRTY_TWO_K.search(line))

#: A fixture that is NOT a ledger, for the half of the rule that says the derived
#: bound must not spread.
ORDINARY = "voice-loop"


# ── clause 4: the bound is derived, and it does not spread ────────────────────

def test_the_ledger_bound_is_three_times_the_ceiling_of_the_file_it_audits():
    """Clause 4: the number is arithmetic over the audited file's own ceiling, so
    the two stay together if `prompt_surface` ever moves a ceiling — and neither
    number is a literal someone has to remember to update.

    `memory-md-ledger` audits MEMORY.md (25,600 B) -> 76,800 B; `user-md-ledger`
    audits USER.md (16,384 B) -> 49,152 B. 49,152 B over `user-md-ledger`'s live
    496 B mean row is ~99 rows against 53 loaded lines, which is why USER.md is
    not yet hitting a wall and MEMORY.md is.
    """
    assert ps.memory_ceiling("MEMORY.md") == 25_600
    assert ps.memory_ceiling("USER.md") == 16_384

    assert ceiling.ledger_ceiling("memory-md-ledger") == 76_800
    assert ceiling.ledger_ceiling("user-md-ledger") == 49_152
    for stem, audited in ceiling.MEMORY_LEDGERS.items():
        assert ceiling.ledger_ceiling(stem) == ceiling.LEDGER_MULTIPLIER * ps.memory_ceiling(audited)
        assert ceiling.topic_ceiling(f"{ceiling.TOPIC_PREFIX}{stem}") == ceiling.ledger_ceiling(stem)


def test_the_input_that_restores_32768_is_the_audited_file_ceiling(tmp_path, monkeypatch):
    """The fallback the commit message promises, measured at the guard instead of read
    off the source: `ledger_ceiling` derives from `memory_ceiling(loaded_file)`, so an
    audited file that answers None — renamed, or no longer a loaded-memory file — has to
    put that ledger back under `TOPIC_FILE_CEILING_BYTES` rather than leave it unbounded.

    Four halves, because the weaker versions of this check pass on a broken change. The
    first two drive the real trigger — the audited file removed from
    `prompt_surface.MEMORY_CEILINGS`, the mapping that lookup actually reads — with two
    guards on the patch itself, so a mapping edit that removed everything (or nothing)
    fails instead of quietly producing the expected None. The third stubs
    `memory_ceiling`, the named input, rather than the ledger map, and that is the half
    which pins the bound answers. Fourth, the guard and its wording:
    exactly 32,768 (a `ledger_ceiling` that returned None instead would still leave
    `topic_ceiling` answering 32,768 through its own `or`, so only the direct call pins
    which fallback fired); the guard refuses 40,000 bytes against it; and the refusal
    carries the ordinary topic-file wording, not a ledger message quoting "3 x MEMORY.md's
    None-byte ceiling", which is what clamping the number while keeping the ledger prose
    would have printed.
    """
    root = tmp_path
    (root / "memory").mkdir(parents=True, exist_ok=True)
    ledger = root / "memory" / "memory-md-ledger.md"
    monkeypatch.setattr(ceiling, "MEMORIES_DIR", root)

    # The real trigger first. `prompt_surface.memory_ceiling` is
    # `MEMORY_CEILINGS.get(filename)`, so dropping the audited file from that mapping —
    # which is what a rename out of the loaded-memory set actually does — takes the
    # unstubbed lookup to None on its own. The stub below only proves the consumer
    # honours a None; this half is what proves the None is reachable the way the commit
    # message claims it is.
    monkeypatch.setattr(ps, "MEMORY_CEILINGS", {
        k: v for k, v in ps.MEMORY_CEILINGS.items() if k != "MEMORY.md"})
    assert ps.memory_ceiling("MEMORY.md") is None, (
        "the mapping edit did not take, so this half would be asserting nothing")
    assert ps.memory_ceiling("USER.md") is not None, (
        "the mapping edit removed more than the audited file, so it is not the trigger "
        "the message names")
    assert ceiling.ledger_ceiling("memory-md-ledger") == 32_768, (
        "with the audited file genuinely absent from the ceilings, the ledger must sit "
        "back under the shared topic bound")

    monkeypatch.setattr(ceiling, "memory_ceiling", lambda name: None)

    assert ceiling.ledger_ceiling("memory-md-ledger") == 32_768
    msg = ceiling.memory_write_error(ledger, "x" * 40_000)
    assert msg is not None and "32,768-byte topic file ceiling" in msg, msg
    assert "None-byte" not in msg, msg


def test_a_third_topic_stem_never_receives_the_ledger_bound():
    """Clause 4's other half, and the one that keeps this change small: the derived
    bound is a property of two named files, not of the topics directory.

    `topics/voice-loop` and `topics/big` are measured at 32,768 B exactly as before,
    and `ledger_ceiling` answers None for them, so no future topic inherits 76,800 B
    by living in the same directory as a ledger. `topics/big` is the stem
    `tests/test_memory_index_cap.py` has pinned since review 2026-09-24, so the two
    files assert the same number from opposite directions.
    """
    for stem in (ORDINARY, "big", "infrastructure-email-calendar-health"):
        assert ceiling.ledger_ceiling(stem) is None, stem
        assert ceiling.topic_ceiling(f"{ceiling.TOPIC_PREFIX}{stem}") == 32_768, stem
    assert ceiling.TOPIC_FILE_CEILING_BYTES == 32_768


def test_the_stems_the_guard_treats_as_ledgers_are_the_ones_the_script_reads():
    """Clause 4's pin against drift, and the reason the map lives in `app/`.

    `scripts/memory/memory_ledger.py` imports `app.memory_ceiling` for `tight_limit`
    (:74), so `app/` cannot import it back to read `LEDGERS` — the two sides
    therefore declare the pair independently and this node is what stops them
    disagreeing. `LEDGERS` maps a loaded file to the ledger FILENAME (`"MEMORY.md":
    "memory-md-ledger.md"`); the guard keys on the slug stem, so the comparison adds
    `.md` rather than assuming the two spellings are identical.

    A ledger renamed on one side and not the other would keep the 32,768 B bound
    while its audited file kept growing — the failure this item exists to fix,
    re-created quietly.
    """
    declared = {f"{stem}.md" for stem in ceiling.MEMORY_LEDGERS}
    assert declared == set(ml.LEDGERS.values()), (declared, set(ml.LEDGERS.values()))
    assert len(declared) == 2
    for loaded, filename in ml.LEDGERS.items():
        stem = filename[: -len(".md")]
        assert ceiling.MEMORY_LEDGERS[stem] == loaded, filename
        assert ps.memory_ceiling(loaded) is not None, loaded


# ── the seam: a second surface measures the same bytes ───────────────────────

def _fixture_vault(tmp_path: Path, slug: str, size: int) -> Path:
    """A `--root` the validator reads: one typed MEMORY.md line and one topic file.

    The root is the `lloyd/` directory itself — `validate_memory_index.py` globs
    `root/memory/*.md` — so handing it the vault parent finds no topics at all and
    every assertion below would pass on an empty denominator.
    """
    root = tmp_path / "lloyd"
    (root / "memory").mkdir(parents=True)
    (root / "MEMORY.md").write_text(
        "---\ntype: note\n---\n\n# Lloyd Long-Term Memory\n\n"
        "## Infra\n- [project] **Probe rule.** A non-empty set is the positive control.\n",
        encoding="utf-8")
    (root / "memory" / f"{slug}.md").write_text("x" * size, encoding="utf-8")
    return root


def test_the_validator_measures_a_ledger_against_the_same_bound(tmp_path):
    """The same property, read at the surface that reports it rather than refusing.

    `scripts/memory/validate_memory_index.py` compared every topic against
    `TOPIC_FILE_CEILING_BYTES` with its own copy of the comparison, so a guard that
    allows a 40,000 B ledger while the validator still calls 32,768 B an error is a
    green write lane feeding a red nightly report — which is the whole point of the
    item (the wall is ~4 curation nights away at §2's 10-row cap). Both sides now
    ask `ceiling.topic_ceiling()`; this node holds one size, 40,000 B, and demands
    opposite verdicts from the two stems that differ only in whether they are a
    ledger.
    """
    def topic_errors(slug: str) -> list[str]:
        root = _fixture_vault(tmp_path / slug, slug, 40_000)
        proc = subprocess.run(
            [sys.executable, str(VALIDATOR), "--root", str(root), "--json"],
            capture_output=True, text=True, timeout=120)
        report = json.loads(proc.stdout)
        assert report["topic_files"] == 1, (
            f"the validator saw no topic file under {root}, so a 0-error answer "
            f"proves nothing: {report}")
        return [e for e in report["errors"] if "topic file" in e]

    assert topic_errors("memory-md-ledger") == [], (
        "a 40,000 B ledger is inside its derived 76,800 B bound, so the report must "
        "stay clean — at the base blob this same fixture reported the shared 32,768 B "
        "topic ceiling as an error")
    ordinary = topic_errors(ORDINARY)
    assert len(ordinary) == 1 and "32,768 B topic ceiling" in ordinary[0], ordinary


# ── clause 5: the prose a curator follows states the bound the code enforces ──

def _assert_no_ledger_ceiling_claim(rel: str, text: str) -> None:
    """Fail if any line of `text` states a ledger's cap as the shared topic ceiling,
    and prove the check can fire AND can hold its peace.

    Two controls, one per direction, because a narrowed ban has two ways to be wrong
    and the corpus proves neither. The first is the positive control — the pattern must
    match a synthetic sentence of exactly the shape clause 5 bans, or the 0-hit result
    below is the grep that reads as a clean bill of health while matching nothing. The
    second is the negative one: the same predicate must stay silent on the correct
    sentence about an ordinary topic file, which is the over-broad ban the review of
    round SM_20261006_015104 refused this file for. A check with only the first control
    would have passed with the over-broad version still in it.
    """
    assert _states_a_ledger_cap_at_32768(
        "the ledger is bounded by the 32,768 B topic-file ceiling"), (
        "the predicate does not fire on the sentence clause 5 bans, so every 0-hit "
        "below proves nothing")
    assert not _states_a_ledger_cap_at_32768(
        "an ordinary topic file is bounded by the 32,768 B topic-file ceiling"), (
        "the predicate fires on a correct sentence about a non-ledger topic, which is "
        "the over-broad ban the review of SM_20261006_015104 named")
    assert text, f"{rel} is empty, so the 0-hit result below proves nothing"
    hits = [line for line in text.splitlines() if _states_a_ledger_cap_at_32768(line)]
    assert not hits, (
        f"{rel} still states a live ledger's cap as the shared topic ceiling: "
        f"{hits[0].strip()[:200]!r}")


def _module_docstring(rel: str) -> str:
    """The module docstring of repo file `rel` — the prose a person reads first.

    The ban is scoped to it deliberately, not to the whole file. `memory_ledger.py`
    is 345 lines, and a line of it may legitimately discuss the shared ceiling when
    it is talking about an ordinary topic file; a guard that fires on a sentence that
    is *correct* is a guard somebody switches off. The clause's claim is about the
    sentence that calls a ledger a 32,768-byte topic file, and the docstring is that
    sentence. The two asserts below are what make the narrower slice honest: an empty
    or wrong slice would also produce a clean 0-hit.
    """
    import ast
    src = (ROOT / rel).read_text(encoding="utf-8")
    doc = ast.get_docstring(ast.parse(src))
    assert doc, f"{rel} has no module docstring, so the slice below is empty"
    assert "ledger" in doc.lower(), (
        f"{rel}'s docstring never mentions ledgers — this is the wrong slice, and a "
        "0-hit in it would prove nothing about the sentence the clause bans")
    return doc


#: The §5 curation skill is vault content: outside this repo, outside this diff, on
#: vault main since `87e4c6e1` (landed by the nightly pre-flight, not by a `vault_land`
#: event naming #2212 — human clause 4). Clause 5's corpus is that one file, so this
#: clause needs two nodes and neither one is a skip.
#:
#: The previous round had ONE node that opened the vault path and called `pytest.skip`
#: when it was not there. The review of SM_20261006_015104 refused that as a test-honesty
#: defect, and it was right twice over: under the gate HOME is the round home and
#: `~/obsidian` does not exist, so the node skipped on every rung while reading like an
#: assertion — and `scripts/automod/reasoning_bank.py`'s own rule says "a clause pinned
#: only by a live_vault / skipped test is pinned by nothing under the gate". Converting
#: the skip into a `live_vault` marker alone would have satisfied the first half and
#: stayed inside the second.
#:
#: So the ban runs over a committed copy of the whole file, the way #1938, #2119, #2240
#: and #2255 each pinned a measurement the gate could not reach — see
#: `tests/fixtures/.gitignore`, whose reason is literally this one: the gate runs with
#: HOME at the round home, "a node that opened it would skip, and a skipping node pins
#: nothing". The whole file, not an extract: an extract of a prose ban is the exact
#: shape that can hide the offending line, and the file is 73 lines.
#:
#: What the frozen corpus cannot do is notice the LIVE file drifting, so that is the
#: `live_vault` node's whole job — the marker `pytest.ini` registers for "assertions
#: about a file an hourly autoresearch promotion or a nightly job can rewrite between
#: rounds", deselected by the gate (`scripts/automod/gate.py:TESTS_MARK_EXPR`) because a
#: round home has no vault to read. Read them as a pair: the gate node proves the ban is
#: real and the corpus it was corrected to is clean; the marked node proves the frozen
#: corpus is still what the nightly actually reads.
SKILL_WITNESS = Path(__file__).parent / "fixtures" / "step_2a_ter_curation_witness_2212.md"


def test_the_curation_skill_witness_carries_no_ledger_cap_at_the_topic_ceiling():
    """Clause 5, the half that runs inside the gate.

    §1 once told the curator a ledger "is bounded by the 32,768 B topic-file ceiling" and
    to report `ledger: at ceiling (<bytes> / 32768 B) — split owed`; §5 repeated both for
    `memory-md-ledger`. A nightly that followed that text stopped the audit at 32,768 B
    and asked for a split that `LEDGERS` cannot read back — before the code ever refused
    anything.

    The asserts, in the order that makes each one falsifiable:
      * the corpus is the whole file at its exact bytes (14,036 B / 73 lines), and it
        names `topics/memory-md-ledger`, so a 0-hit below cannot be a wrong path or an
        empty slice;
      * no LINE states a ledger's cap as 32,768 B — `_assert_no_ledger_ceiling_claim`
        carries its own positive and negative control, so the pattern is proven able to
        fire and unable to fire on the correct ordinary-topic sentence;
      * the phrase the false instruction ended with ("split owed") is gone, which is the
        behaviour §1 was sending a curator off to do;
      * the numbers that REPLACE it are present (76,800, 49,152, and `ledger_ceiling` as
        the thing to ask rather than remember), because clause 5 asks that the prose
        "name the new cap", not merely omit the old one.

    None of that is asserted against the live vault, and that is the honest limit of a
    repo node: it pins this corpus. `live_vault` below is what keeps the pin from
    becoming a fossil.
    """
    assert SKILL_WITNESS.is_file(), (
        f"{SKILL_WITNESS.name} is the committed copy of the §5 curation skill; without "
        "it clause 5's corpus is a vault file the gate cannot open")
    raw = SKILL_WITNESS.read_bytes()
    text = raw.decode("utf-8")
    # Bytes, not characters: the file is full of em dashes, so `len(text)` reads 95 short
    # of `wc -c` and a char count would fail on a file nobody truncated.
    assert len(raw) == 14_036 and text.count("\n") == 73, (
        f"{SKILL_WITNESS.name} is {len(raw)} B / {text.count(chr(10))} newlines, not the "
        "14,036 B / 73 lines of the whole skill file — a partial copy would make the "
        "0-hit below a coverage gap rather than a clean result")
    assert "topics/memory-md-ledger" in text and "ledger" in text.lower(), (
        f"{SKILL_WITNESS.name} does not name a ledger, so it is the wrong corpus")

    _assert_no_ledger_ceiling_claim("step-2a-ter-curation.md (committed witness)", text)

    assert "split owed" not in text, (
        "the completion-note instruction is back to telling a curator to split a live "
        "ledger, which `LEDGERS` reads only one file per index")
    for figure in ("76,800", "49,152", "ledger_ceiling"):
        assert figure in text, (
            f"the witness never states {figure!r}, so the ban above has nothing it "
            "replaced the false cap with")


#: Same reason as the gate node, opposite question: is the file the nightly reads still
#: the file that was frozen? A frozen corpus with no drift check pins yesterday's prose
#: and reports it as today's.
@pytest.mark.live_vault
def test_the_live_curation_skill_has_not_drifted_from_the_witness_the_ban_reads():
    """Clause 5, the live half — and the node that makes the freeze above honest.

    Run it explicitly: `pytest tests/test_memory_ledger_bound.py -m live_vault`. The
    automod gate deselects `live_vault` (`scripts/automod/gate.py:TESTS_MARK_EXPR`)
    precisely because there is no vault at a round home, so this is a node about the
    machine it runs on, marked as one rather than skipping like one.

    Two asserts. The live skill's bytes equal the frozen corpus: if a later nightly,
    promotion or human edit puts a ledger-cap sentence back, or corrects the prose
    further, this goes red and the witness gets re-frozen by whoever changed it — a
    drift is then a decision somebody made, not a gap nobody noticed. And the ban re-runs
    over the live text, so the reading of "no line states a ledger cap at 32,768 B" is
    checked against the file that is actually loaded into the nightly's prompt, not only
    against this tree's copy.
    """
    assert SKILL.is_file(), (
        f"{SKILL} is not on this box. The node carries `live_vault`: it reads the live "
        "vault, and the gate deselects it because a round home has no ~/obsidian. If you "
        "are running it deliberately, the vault must be mounted")
    live = SKILL.read_text(encoding="utf-8")
    assert live == SKILL_WITNESS.read_text(encoding="utf-8"), (
        f"{SKILL} has drifted from {SKILL_WITNESS.name}: the ban in the node above is "
        "reading a corpus that is no longer what the nightly reads. Re-freeze the witness "
        "in the same change that moved the prose.")
    _assert_no_ledger_ceiling_claim("step-2a-ter-curation.md (live vault)", live)


def test_the_vault_write_lane_measures_a_ledger_against_the_derived_bound(tmp_path, monkeypatch):
    """The third lane onto a ledger path refuses at the derived bound, in process.

    `agent_mcp/vault.py`'s `_vault_write` is how a session writes a vault file whole,
    and a ceiling enforced on `memory_add` but not on this lane is a ceiling the
    writer can step around by choosing a different tool — which is the same property
    #1010/#507 was about when the nightly grew USER.md to 95,302 B through an
    ungated lane. The two calls differ only in byte count, so the pair attributes the
    difference in outcome to the bound and not to the content: at 40,000 B the ceiling
    does not refuse at all, at 78,000 B it refuses naming 76,800.

    Both `VAULT` and `MEMORIES_DIR` are redirected together because the guard decides
    "is this a topic file?" by realpath against the memories root; patching only the
    lane's root would leave the path outside the guard's corpus, where every size is
    permitted and this node would pass while measuring nothing.
    """
    import agent_mcp.vault as V
    root = tmp_path / "lloyd"
    (root / "memory").mkdir(parents=True)
    monkeypatch.setattr(V, "VAULT", tmp_path)
    monkeypatch.setattr(ceiling, "MEMORIES_DIR", root)
    rel, pad = "lloyd/memory/memory-md-ledger.md", "# lloyd/MEMORY.md ledger\n\n- "
    written = root / "memory" / "memory-md-ledger.md"

    small = V._vault_write({"path": rel, "content": pad + "x" * 40_000})
    assert "ceiling" not in str(small.get("error", "")).lower(), (
        f"the vault_write lane still refuses a 40,000 B ledger write: {small}")
    accepted = written.stat().st_size
    assert accepted == len(pad) + 40_000, (
        f"the accepted half must actually write, or the assertion above is an "
        f"unreached branch: {accepted} bytes on disk")

    big = V._vault_write({"path": rel, "content": pad + "x" * 78_000})
    err = str(big.get("error", ""))
    assert "76,800" in err, f"a ledger over its bound must name the bound: {big}"
    assert "32,768" not in err, err
    assert written.stat().st_size == accepted, (
        "a refused write must enter the vault nowhere; the guard returns before the "
        "lock and the ledger, so a partial write here is a different bug")


def test_the_ledger_script_stops_calling_itself_a_32768_byte_topic_file():
    """Clause 5, second half — prose the item did not name, found by reading what
    the change makes false. `scripts/memory/memory_ledger.py`'s module docstring used
    to say a ledger "sits under the 32,768 B topic-file ceiling"; it is the text a
    person reads first, and it is now wrong for both files the script manages.

    `memory_ledger.py` is a repo file, so the corpus is proven here rather than
    assumed: it is tracked in git and has bytes, and the same pattern that must not
    match it is demonstrated to match a sentence of the same shape. The slice is the
    docstring, not the file: see `_module_docstring` for why the wider ban would be a
    trap, and that helper's own asserts are what stop the narrower slice from being an
    empty region that trivially has no hits.
    """
    assert LEDGER_SCRIPT.is_file() and LEDGER_SCRIPT.stat().st_size > 4_000
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
                              "scripts/memory/memory_ledger.py"],
                             capture_output=True, text=True)
    assert tracked.returncode == 0, f"not tracked: {tracked.stderr.strip()}"
    doc = _module_docstring("scripts/memory/memory_ledger.py")
    _assert_no_ledger_ceiling_claim("scripts/memory/memory_ledger.py (module docstring)",
                                    doc)


# ── clause 6: the gate report this item's history is written from has bytes ────
#
# Round SM_20261005_044345 is the round every claim on this item descends from — "died
# at the `tests` rung on 44 skips, zero failures, no `review` rung, both attempts
# unspent" — and those bytes lived only in one machine's state directory, which nothing
# commits. Clause 6's durable home is the vault copy at
# `~/obsidian/backlog/data/2026-10-06.2212-SM_20261005_044345.gate-witness.jsonl` (vault
# commit `09c4df26`, one JSONL row per rung, copied verbatim, the `tests` row carrying
# that rung's own `data` dict). The file below holds the same bytes for the reason
# #2119, #2240 and #2255 each wrote down in `tests/fixtures/.gitignore`: the gate runs
# with HOME at the round home, where `~/obsidian` does not exist, so a node that opened
# only the vault copy could not run on the rung that grades this item — and a node that
# skipped when it was missing is what the review of SM_20261006_015104 refused for test
# honesty above.

WITNESS = Path(__file__).parent / "fixtures" / \
    "2026-10-06.2212-SM_20261005_044345.gate-witness.jsonl"

#: Clause 6's archive is per-item on purpose. `backlog/data/gate.json` is #1883's
#: witness for round SM_20260930_063800 (vault commit `cfe9a113`), and the amended
#: clause moved this item off that path because clobbering it was the one thing the
#: clause's own last sentence forbade.
NO_CLOBBER = "backlog/data/gate.json"
NO_CLOBBER_LINES, NO_CLOBBER_BYTES = 34, 1_143

#: The six files that round gated, from its own report — the same change this branch
#: continues, so a witness for a different diff is a different round's witness.
ROUND_HEAD = "31d2b1141f097ffa60d07ae1fbc145590b420dcb"
ROUND_BASE = "506e250af9302a55534c43da2c559b50f8325be8"
ROUND_PATHS = ["app/memory_ceiling.py", "scripts/memory/memory_ledger.py",
               "scripts/memory/validate_memory_index.py", "tests/test_memory_index_cap.py",
               "tests/test_memory_ledger.py", "tests/test_memory_ledger_bound.py"]


def test_the_archived_gate_report_still_says_the_round_refused_on_skips_not_review():
    """Clause 6: re-derived by JSON from committed bytes, never by line count and never
    from the live state directory that keeps being rewritten.

    Six rows. Row 0 is the provenance header (`_witness`, `round_id`, `head`, `base`,
    `changed_paths`, `ok: false`); rows 1-5 are that round's rungs verbatim — and the
    whole report the item quotes falls out of them:

    1. `tests` is `ok: false` with `detail` "44 tests skipped (limit 40) …", while
       `data.tests_skipped` is 44 and `data.failed` is 0. Both numbers are read from one
       artifact and compared to each other, so the count in the sentence cannot move
       away from the count in the data — and a refused rung with zero failures is a
       suite-wide rail, not this diff. That pairing is the item's premise for being
       re-offered, and it is the claim that had no bytes behind it.
    2. There is NO `review` row. The five rung names are exactly
       preflight/vet/static/frontend/tests, so no clause was ever graded on that round
       and its two review attempts were unspent. An absent rung is the finding here, so
       the list is asserted as a list: a rung appended later fails the node, which is
       correct — this is a frozen report of one round, not a log.
    3. The `tests` row still carries `pin_findings` naming the 10 dashboard pins that
       did not execute — #2214's mechanism, which is why a later round of the same
       change gated at 33 skips and reached review.

    The vault copy is checked byte-for-byte against this one whenever `~/obsidian` is
    reachable, so no digest is quoted as a fact nobody verified; and #1883's witness at
    `backlog/data/gate.json` is still 34 lines / 1,143 B there, which is the no-clobber
    half of the amendment. Under the gate `~/obsidian` is not there — and that costs this
    node nothing, because every figure it refuses on is asserted from the committed bytes
    above, not from that comparison. The vault branch is a corroboration, never the
    substance: which is the difference between this node and the copy-compare the review
    of round SM_20261006_015104 refused as a hidden skip.
    """
    assert WITNESS.is_file(), (
        f"{WITNESS.name} is the committed witness; without it every figure this item "
        "quotes about round SM_20261005_044345 is a claim about a file nobody can read")
    raw = WITNESS.read_bytes()
    text = raw.decode("utf-8")
    assert len(raw) == 2_723 and text.count("\n") == 6, (
        f"{WITNESS.name} is {len(raw)} B / {text.count(chr(10))} newlines; the frozen "
        f"copy is 2,723 B / 6 lines, so `wc -c` and `wc -l` no longer answer the "
        "figures this node's asserts were written against")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    assert len(rows) == 6, f"the extract holds {len(rows)} rows, not the 6 archived rungs"

    head = rows[0]
    assert head.get("round_id") == "SM_20261005_044345", head.get("round_id")
    assert head.get("ok") is False, "the header no longer records the round as refused"
    assert head.get("head") == ROUND_HEAD and head.get("base") == ROUND_BASE, (
        f"this is not the report of {ROUND_HEAD[:8]} on {ROUND_BASE[:8]}: "
        f"{head.get('head')} / {head.get('base')}")
    assert head.get("changed_paths") == ROUND_PATHS, head.get("changed_paths")
    assert "_witness" in head and "SM_20261005_044345/gate.json" in head["_witness"], (
        "the provenance line that says where these rows were copied from is gone")

    rungs = rows[1:]
    assert [r.get("rung") for r in rungs] == ["preflight", "vet", "static", "frontend",
                                             "tests"], (
        "the rung list moved, so this is no longer the frozen report of one round — and "
        "in particular a `review` row here would contradict the finding below")
    assert all(r["ok"] for r in rungs[:4]), [r["rung"] for r in rungs if not r["ok"]]
    assert "review" not in [r.get("rung") for r in rungs], (
        "the item's whole re-offer rests on this round never reaching review; if a "
        "review row exists, the report is not the one the triage read")

    tests = rungs[-1]
    assert tests["rung"] == "tests" and tests["ok"] is False, tests["detail"]
    data = tests.get("data") or {}
    assert data.get("tests_skipped") == 44 and data.get("failed") == 0, (
        f"the refusal was a count, not a failure: {data.get('tests_skipped')} skipped "
        f"against {data.get('failed')} failed")
    assert data.get("passed") == 16_127 and data.get("collected") == 16_172, data
    # One artifact, both readings: the count the sentence prints must be the count the
    # data holds, or the rung's prose and its payload are two numbers drifting apart.
    stated = re.search(r"^(\d+) tests skipped \(limit (\d+)\)", tests["detail"])
    assert stated and int(stated.group(1)) == data["tests_skipped"], (
        f"`detail` no longer states the count its own data carries: {tests['detail']!r}")
    assert int(stated.group(2)) == 40, (
        f"the refusal text names limit {stated.group(2)}, not the 40 this item quotes; "
        "the frozen report and the quoted number have parted company")
    assert "a round that skips its way to green is not a round that passed" in \
            tests["detail"], tests["detail"]
    assert any("DASHBOARD_PINS_NOT_EXECUTED" in p for p in data.get("pin_findings", [])), (
        "the rung's own diagnosis of whose rail this was is gone from the witness, and "
        "with it the evidence that the skips were not this diff's")

    vault = Path.home() / "obsidian" / "backlog" / "data" / WITNESS.name
    if vault.is_file():
        assert vault.read_bytes() == WITNESS.read_bytes(), (
            f"the committed copy and {vault} are no longer the same bytes, so one of "
            "them is no longer the report this item quotes")
        other = Path.home() / "obsidian" / NO_CLOBBER
        assert other.is_file(), (
            f"{NO_CLOBBER} is #1883's witness and is gone — clause 6's archive was "
            "amended off that path precisely so it would survive")
        other_bytes = other.read_bytes()
        assert (other_bytes.count(b"\n"), len(other_bytes)) == (NO_CLOBBER_LINES,
                                                               NO_CLOBBER_BYTES), (
            f"{NO_CLOBBER} is no longer the 34 lines / 1,143 B that round's witness "
            "was, so the no-clobber half of the amendment no longer holds")
