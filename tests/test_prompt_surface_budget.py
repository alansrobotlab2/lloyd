"""The prompt surface is bounded, measured, and cannot be pasted into twice.

Why this file exists
--------------------
Backlog #377 measured the identity surface on 2026-09-08: `SOUL.md` carried 62%
of its bytes in the gate stack, 29% of its nonblank lines were prohibitions, and
`MEMORY.md` had been overwritten with a byte-for-byte copy of the operating
contract — so every turn paid ~4.4K duplicate tokens and *nothing in the code
could see it*: `grep -n "PROMPT_BUDGET|prompt_chars|len(system_prompt)"
prompt_builder.py prefetch.py` returned zero hits. The trim that closed the item
(vault `010d233`) only holds if something pins it, and the mechanism that made
the mess — autoresearch's promote path writing `SOUL.md`/`MEMORY.md` in the vault
directly, no gate, no test — is still live. These tests are the only check
between one audit and the next mutation.

Two groups, deliberately different:

* **Live-vault invariants** — the acceptance check of #377, against the real
  `~/obsidian/lloyd/`. They skip when that vault is absent (a canary's fake
  HOME, a fresh clone) so they can never be satisfied by a fixture.
* **prompt_builder / prefetch behaviour** — against a fake vault: the size
  measurement that was missing, the per-turn log line, and the guard that drops
  a pasted contract out of a memory file instead of injecting it twice.

The `LOAD_BEARING` markers exist because a byte-ratio test can be passed by
deleting the safety gate. Ratio tests are necessary and not sufficient: the
behaviours the benches score must still be *named* in the prompt.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

import prompt_builder as pb
import prompt_surface as ps
from prompt_surface import (
    DUPLICATE_CONTRACT_CEILING,
    GATE_HEADS,
    GATE_STACK_CEILING,
    LOAD_BEARING,
    PROHIBITION_RATIO_CEILING,
)

VAULT = Path.home() / "obsidian" / "lloyd"
SOUL = VAULT / "SOUL.md"
LLOYD_REPO = Path(pb.__file__).parent

# Two marks, and the second is the one that matters. `skipif` keeps these from
# failing where the vault is absent; `live_vault` lets the automod gate exclude
# them with `-m "not live_vault"`.
#
# They read a file no round under test controls — an hourly autoresearch
# promotion or a nightly reflection job can rewrite `SOUL.md` between one round
# and the next. Left on the gate's hard `tests` rung, a re-inflation would fail
# every future round whatever its diff, which is how `test_tool_overrides.py`
# aborted three rounds in fifteen hours on 2026-09-07. The invariants are
# enforced at the writers instead — `scripts/automod/vault_round.py` and
# `scripts/autoresearch/promote.py` both call `prompt_surface.check_contract`
# before they commit — so this group is the reporting copy, not the enforcement.
vault_only = pytest.mark.skipif(
    not SOUL.exists(), reason="live vault prompt surface not on disk here"
)
live_vault = pytest.mark.live_vault

# `_sections`, `_gate_share` and `_prohibition_ratio` live in `prompt_surface`
# now, because the writers have to run the same checks and a second private
# definition of "is the contract bloated" is how the two halves drift apart.
# Thin aliases keep the assertions below readable. There is deliberately no
# alias or local copy for the #464 check: see
# `test_memory_md_is_not_a_copy_of_the_contract`.
_sections = ps.sections
_gate_share = ps.gate_share
_prohibition_ratio = ps.prohibition_ratio


@pytest.fixture(scope="module")
def soul_text() -> str:
    return SOUL.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def memory_text() -> str:
    return (VAULT / "MEMORY.md").read_text(encoding="utf-8")


def _normalize(text: str) -> str:
    """Lowercase, every run of non-alphanumerics to a single space.

    Headings carry hyphens, ampersands and parenthetical flags; the audit table
    is prose. Comparing raw strings makes `Anti-Compliance` and `anti compliance`
    different sections, which would report a covered section as missing.
    """
    return re.sub(r"[^a-z0-9&]+", " ", text.lower()).strip()


# ── live vault: the #377 acceptance check ────────────────────────────────────

@vault_only
@live_vault
def test_gate_stack_is_a_minority_of_the_contract(soul_text):
    share, gate, total = _gate_share(soul_text)
    assert share <= GATE_STACK_CEILING, (
        f"gate stack is {gate} of {total} bytes ({share:.1%}); #377 landed it at "
        f"48.2% and {GATE_STACK_CEILING:.0%} is the ceiling — a gate stack that "
        "outweighs the contract is the constraint bloat the item was filed on"
    )


@vault_only
@live_vault
def test_gate_sections_still_exist(soul_text):
    """A ratio test passes at 0% if the gate sections were renamed away."""
    found = {h for h, _ in _sections(soul_text) if h.startswith(GATE_HEADS)}
    assert len(found) >= 3, f"only {sorted(found)} of the gate roles survived"


@vault_only
@live_vault
def test_trim_kept_the_load_bearing_behaviours(soul_text):
    missing = [label for label, marker in LOAD_BEARING.items() if marker not in soul_text]
    assert not missing, f"trim removed behaviour the benches score: {missing}"


@vault_only
@live_vault
def test_prohibition_line_ratio_is_under_the_ceiling(soul_text):
    ratio, hits, nonblank = _prohibition_ratio(soul_text)
    assert ratio <= PROHIBITION_RATIO_CEILING, (
        f"{hits} of {nonblank} nonblank lines ({ratio:.0%}) are prohibitions; "
        f"triage measured 29%, the trim 19%, ceiling "
        f"{PROHIBITION_RATIO_CEILING:.0%}"
    )


@vault_only
@live_vault
def test_memory_md_is_not_a_copy_of_the_contract(memory_text, soul_text):
    """#464 — MEMORY.md was overwritten with SOUL.md; the contract arrived twice.

    Asserted through `prompt_surface.duplicate_contract_errors` — the exact
    function `vault_round` and `promote` call — and not through a local `_h1()`
    plus a local line-share loop. That private copy is the reason this file read
    green while both writers stayed red: `9ca4fc6` corrected the read *here* only,
    so the report said healthy and `check_contract` went on comparing raw line 1,
    refusing every real file pair until #1069. A local reimplementation of an
    invariant cannot fail when the module breaks, which is the only job a
    reporting copy has.

    The two halves are then named individually, through module functions, so a
    failure says which one moved rather than only "not empty".
    """
    assert ps.duplicate_contract_errors(soul_text, memory_text) == []

    assert ps.h1(memory_text) and ps.h1(memory_text) != ps.h1(soul_text), (
        f"MEMORY.md opens with SOUL.md's H1 {ps.h1(soul_text)!r} — the "
        "operating-contract paste is back"
    )
    dup = ps.shared_line_share(memory_text, soul_text)
    assert dup <= DUPLICATE_CONTRACT_CEILING, (
        f"{dup:.0%} of MEMORY.md's lines are verbatim SOUL.md lines — the "
        "operating contract is being injected twice per turn again"
    )


@vault_only
@live_vault
def test_anti_compliance_frame_lives_in_exactly_one_place(soul_text):
    """#465 — the same six rules shipped in Python and in the vault, diverged."""
    code = (LLOYD_REPO / "prompt_builder.py").read_text(encoding="utf-8")
    total = code.count("Certainly") + soul_text.count("Certainly")
    assert total == 1, (
        f"'Certainly' appears {total}x across prompt_builder.py and SOUL.md; the "
        "no-blanket-agreement rule must be stated once, in the vault copy the "
        "promotion path can actually edit"
    )


@vault_only
@live_vault
def test_section_audit_covers_every_section(soul_text):
    """Acceptance (1): keep/cut/condense recorded for each `## ` section."""
    audits = sorted((VAULT / "reviews").glob("*soul*audit*.md"))
    assert audits, "no SOUL.md section audit under lloyd/reviews/ — #377's artifact is gone"
    text = "\n".join(p.read_text(encoding="utf-8") for p in audits).lower()
    haystack = _normalize(text)
    uncovered = []
    for heading, _ in _sections(soul_text):
        # Two words: the audit table is keyed on the pre-trim heading for the
        # sections that were renamed, so match on a punctuation-free prefix.
        key = " ".join(_normalize(heading).split()[:2])
        if key and key not in haystack:
            uncovered.append(heading)
    assert not uncovered, f"sections missing from the audit artifact: {uncovered}"


@vault_only
@live_vault
def test_the_live_memory_index_validates():
    """Review 2026-09-24 P4: `scripts/memory/validate_memory_index.py` over the vault.

    `structure` (links resolve, topic files legal and bounded, MEMORY.md under its
    ceiling) until the index ceiling is deployed; `full` (≤ 80% of the ceiling,
    typed lines only, short index lines) from the commit that lowers
    `MEMORY_MD_CEILING_BYTES` to `MEMORY_MD_INDEX_CEILING_BYTES` — that flip arms
    this check with no edit here.
    """
    import subprocess
    import sys

    mode = ("full" if ps.MEMORY_MD_CEILING_BYTES == ps.MEMORY_MD_INDEX_CEILING_BYTES
            else "structure")
    script = Path(__file__).resolve().parents[1] / "scripts/memory/validate_memory_index.py"
    proc = subprocess.run([sys.executable, str(script), "--root", str(VAULT),
                           "--mode", mode], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr

# ── prompt_builder: the measurement that did not exist (#466) ───────────────

def test_prompt_budget_constant_exists():
    assert isinstance(pb.PROMPT_BUDGET_CHARS, int) and pb.PROMPT_BUDGET_CHARS > 0


def test_measure_prompt_breaks_down_by_component():
    report = pb.measure_prompt({"SOUL.md": "x" * 1000, "skills_index": "y" * 4000})
    assert report["total_chars"] == 5000
    assert report["over_budget"] is False
    assert report["components"]["SOUL.md"]["chars"] == 1000
    assert report["components"]["skills_index"]["est_tokens"] == 1000
    assert report["total_est_tokens"] == 1250


def test_measure_prompt_flags_over_budget():
    report = pb.measure_prompt({"SOUL.md": "x" * (pb.PROMPT_BUDGET_CHARS + 1)})
    assert report["over_budget"] is True


def test_log_prompt_size_emits_component_breakdown(caplog):
    with caplog.at_level(logging.INFO, logger=pb.logger.name):
        pb.log_prompt_size({"SOUL.md": "a" * 500, "MEMORY.md": "b" * 600},
                           session_id="s42")
    lines = [r.getMessage() for r in caplog.records if "PROMPT_BUDGET" in r.getMessage()]
    assert len(lines) == 1, lines
    line = lines[0]
    for part in ("session=s42", "SOUL.md=500", "MEMORY.md=600", "total=1100", "budget="):
        assert part in line, f"{part!r} missing from: {line}"


@pytest.fixture
def fake_turn_vault(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "SOUL.md").write_text("# Lloyd Operating Contract\n\nBE BOUNDED\n",
                                   encoding="utf-8")
    (vault / "MEMORY.md").write_text("# Lloyd Long-Term Memory\n\nA fact.\n",
                                     encoding="utf-8")
    monkeypatch.setattr(pb, "_CANON_SOUL_PATH", vault / "SOUL.md")
    monkeypatch.setattr(pb, "_CANON_MEMORIES_DIR", vault)
    monkeypatch.setattr(pb, "_CANON_SKILLS_DIRS", [tmp_path / "no-skills"])
    monkeypatch.delenv("LLOYD_OVERLAY_DIR", raising=False)
    return vault


def test_build_system_prompt_logs_its_own_breakdown(fake_turn_vault, caplog):
    with caplog.at_level(logging.INFO, logger=pb.logger.name):
        prompt = pb.build_system_prompt(session_id="turn1")
        silent = pb.build_system_prompt()   # no session_id -> no log line
    assert silent == prompt, "logging changed the prompt it logged"
    lines = [r.getMessage() for r in caplog.records if "PROMPT_BUDGET" in r.getMessage()]
    assert len(lines) == 1, lines
    assert "session=turn1" in lines[0]
    assert "SOUL.md=" in lines[0] and "memories=" in lines[0]
    assert "harness_hints=" in lines[0]
    # The logged total is the sum of the components; the joined string is that
    # plus the "\n\n" separators between them.
    logged = int(re.search(r"total=(\d+)", lines[0]).group(1))
    assert 0 < len(prompt) - logged < 100, (logged, len(prompt))


# ── prompt_builder: the paste guard (#464) ──────────────────────────────────

@pytest.fixture
def paste_vault(tmp_path, monkeypatch):
    """A vault whose MEMORY.md is SOUL.md with one real note appended."""
    soul = "# Lloyd Operating Contract\n\n" + "\n\n".join(
        f"rule {i}: do the thing." for i in range(60)
    )
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "SOUL.md").write_text(soul, encoding="utf-8")
    return vault, soul


def _point_prompt_builder_at(vault, monkeypatch, tmp_path):
    monkeypatch.setattr(pb, "_CANON_SOUL_PATH", vault / "SOUL.md")
    monkeypatch.setattr(pb, "_CANON_MEMORIES_DIR", vault)
    monkeypatch.setattr(pb, "_CANON_SKILLS_DIRS", [tmp_path / "no-skills"])
    monkeypatch.delenv("LLOYD_OVERLAY_DIR", raising=False)


def test_pasted_contract_is_dropped_from_memory(paste_vault, monkeypatch, caplog, tmp_path):
    vault, soul = paste_vault
    (vault / "MEMORY.md").write_text(
        soul + "\n\n## Key Knowledge\n- a real memory line\n", encoding="utf-8")
    (vault / "USER.md").write_text("# User\n\npref\n", encoding="utf-8")
    _point_prompt_builder_at(vault, monkeypatch, tmp_path)

    with caplog.at_level(logging.WARNING, logger=pb.logger.name):
        prompt = pb.build_system_prompt()

    assert prompt.count("# Lloyd Operating Contract") == 1, "contract injected twice"
    assert "a real memory line" in prompt, "the guard ate the real memory content"
    assert "pref" in prompt
    assert any("dropping" in r.getMessage().lower() for r in caplog.records), (
        "dropping 60 lines of the contract without a word is how #464 stayed "
        "unnoticed for a day"
    )


def test_memory_md_without_a_paste_is_untouched(paste_vault, monkeypatch, caplog, tmp_path):
    """A memory file quoting two contract lines is a note, not a clobber."""
    vault, soul = paste_vault
    (vault / "MEMORY.md").write_text(
        "# Lloyd Long-Term Memory\n\n- rule 3: do the thing.\n"
        "- rule 4: do the thing.\n- an unrelated note\n", encoding="utf-8")
    _point_prompt_builder_at(vault, monkeypatch, tmp_path)

    with caplog.at_level(logging.WARNING, logger=pb.logger.name):
        prompt = pb.build_system_prompt()

    assert "an unrelated note" in prompt
    assert "rule 3: do the thing." in prompt and "rule 4: do the thing." in prompt
    assert not [r for r in caplog.records if "dropping" in r.getMessage().lower()]


def test_duplicate_run_detection_spans_blank_lines():
    """The contract is markdown: a blank line between paragraphs must not
    break the run, or a real paste measures as runs of 1-3 and slips through."""
    soul = "\n".join(f"line {i}" for i in range(60))
    pasted = "\n\n".join(f"line {i}" for i in range(60)) + "\n\nreal note"
    share, run, total = pb._memory_duplicate_share(pasted, soul)
    assert run == 60, (run, total)
    assert share > 0.9
    dropped = pb._drop_longest_duplicate_run(pasted, soul)
    assert dropped == "real note"


# ── prefetch: the other half of the turn (#466) ─────────────────────────────

def test_prefetch_shares_the_budget_constant():
    import prefetch

    src = Path(prefetch.__file__).read_text(encoding="utf-8")
    assert "PROMPT_BUDGET_CHARS" in src, "prefetch does not reference the budget"


def test_log_turn_prompt_budget_reports_totals_vs_budget(caplog):
    import prefetch

    with caplog.at_level(logging.INFO, logger=prefetch.logger.name):
        prefetch.log_turn_prompt_budget(
            "y" * 2000, session_id="s7",
            system_prompt_chars=pb.PROMPT_BUDGET_CHARS - 1000,
        )
    lines = [r.getMessage() for r in caplog.records if "PROMPT_BUDGET" in r.getMessage()]
    assert len(lines) == 1, lines
    line = lines[0]
    assert "session=s7" in line and "context=2000" in line
    assert f"system={pb.PROMPT_BUDGET_CHARS - 1000}" in line
    assert f"turn_total={pb.PROMPT_BUDGET_CHARS + 1000}" in line
    assert "budget=" in line and "over_budget=True" in line


# ── the byte ceiling on the loaded memory files (#1010, from #507) ───────────
#
# Both loaded memory files load into every user-platform system prompt, and until
# #1010 no byte constant anywhere bounded either of them: `prompt_surface`'s three
# ceilings were all ratios against `SOUL.md`, so there was no number for a growth
# to cross. `lloyd/USER.md` went 48,068 B → 67,081 → 68,164 → 80,227 → 95,302 B
# across five nights of vault commits, and `lloyd/MEMORY.md` went 4,551 B →
# 21,869 B in a week, each one individually unremarkable and each one landing
# unopposed. A test on the *result* is the wrong shape for that, and was already
# tried: a live-vault size test on the gate's hard `tests` rung punishes the next
# author rather than the writer (the `data/tool_overrides.yaml` failure of
# 2026-09-07, three rounds aborted in fifteen hours), which is why `prompt_surface`
# exists. So the ceiling is a constant in the module the writers call, and what
# this file pins is the constant's behaviour and its reach — never the live
# vault's current size.
#
# Two boundaries, and this section deliberately sits on neither:
#
# * The numbers are `prompt_surface`'s, read as attributes, never re-typed here.
#   A test that hard-codes 16,384 passes when somebody raises the constant to
#   16,384,000; a test that reads it would pass whatever it is. What is asserted
#   is the *relationship* — refuses above, accepts at, refuses one byte over —
#   and that it reaches the tool that actually writes the file.
# * Everything here runs against a fixture or a temp directory. A ceiling test
#   whose data is the live file cannot fail, which is the standing rule of
#   `tests/test_prompt_surface_guard.py` and the reason the live group above is
#   deselected by the gate rather than load-bearing.

USER_MD = VAULT / "USER.md"

#: Held-constant SOUL.md text for the size tests. Deliberately not a valid
#: contract: `check_contract` reports the contract's shape too, and none of that is
#: what a size test is about, so each size assertion here is made as a *difference*
#: between two calls that pass this same stub. Giving it a real contract would
#: quietly turn a size test into #377's test.
_SOUL_STUB = "# Not the operating contract\n\nNothing below is what this test reads.\n"

#: Marker line the fixture USER.md carries, so a test can prove *this file's*
#: bytes reached the string being measured rather than merely that the string was
#: large. A size floor cannot tell those apart, which is the whole of clause 5.
USER_MARKER = "USER-MARKER-line that exists in no other file"


@pytest.fixture
def memories_root(tmp_path, monkeypatch):
    """`agent_mcp.session` with its memories root pointed at a temp dir.

    The same seam `tests/test_memory_writer_lane.py` uses for the lock tests:
    `MEMORIES_ROOT` is a module-level `Path.home()` literal, so patching the
    attribute is the only way a test reaches the tool without writing to the live
    vault — and writing to the live vault from a gate test is the thing this
    file's own header forbids.

    Both roots are patched together, and that is the point rather than an
    oversight: `app.memory_ceiling` decides which files *are* memory files by
    resolving a path against its own `MEMORIES_DIR`, while the tool builds the path
    from `session.MEMORIES_ROOT`. Two constants, one property — so a test that
    moved only the tool's root would be testing a guard that had just been told the
    file was nowhere near the vault, and would see every write allowed.
    `test_the_two_roots_that_decide_what_a_memory_file_is_agree` pins them together
    for the live tree.
    """
    import agent_mcp.session as session
    import app.memory_ceiling as ceiling

    root = tmp_path / "lloyd"
    root.mkdir()
    monkeypatch.setattr(session, "MEMORIES_ROOT", root)
    monkeypatch.setattr(ceiling, "MEMORIES_DIR", root)
    return session, root


@pytest.mark.parametrize("name", sorted(ps.MEMORY_CEILINGS))
def test_a_memory_surface_over_its_ceiling_is_refused_by_name_and_limit(name):
    """Clause 1: the refusal names the file and the byte limit, per surface.

    Compared against a second `check_contract` call on the same SOUL.md text, so
    whatever the contract-shape checks contribute is held constant by construction
    and the only difference is the memory slot. That is why the soul text here is
    a stub rather than a valid contract: a fixture that had to satisfy #377's
    ratios would make this test about SOUL.md, and the sizes are not.

    The wrong surface must not be named, either. Each slot is bound to its own
    filename in `check_contract`, so a MEMORY.md refusal that quoted USER.md's
    limit — or the reverse — is a refusal about a file nobody wrote.
    """
    ceiling = ps.MEMORY_CEILINGS[name]
    baseline = ps.check_contract(_SOUL_STUB)
    over = "x" * (ceiling + 1)

    errs = ps.check_contract(
        _SOUL_STUB,
        memory_text=over if name == "MEMORY.md" else None,
        user_text=over if name == "USER.md" else None,
    )

    assert len(errs) == len(baseline) + 1, errs
    msg = errs[-1]
    assert msg.startswith(f"{name} is {ceiling + 1:,} bytes"), msg
    assert f"{ceiling:,}-byte ceiling" in msg, msg
    assert "1 B over" in msg, msg
    other = "USER.md" if name == "MEMORY.md" else "MEMORY.md"
    assert other not in msg, f"a {name} refusal named {other}: {msg}"


@pytest.mark.parametrize("name", sorted(ps.MEMORY_CEILINGS))
def test_the_ceiling_is_the_largest_legal_size_and_not_the_first_refused_byte(name):
    """Clause 1's second half, at the boundary rather than in the middle.

    One byte *under* is the clause's own wording; the boundary itself is the
    tighter read, so both are asserted, and `ceiling + 1` is what separates a
    ceiling from a freeze. A guard that refused at exactly the constant made the
    #507 trim unfinishable — the trim target *is* the constant, so the last byte
    of the trim would have been refused by the guard the trim exists to satisfy.
    """
    ceiling = ps.MEMORY_CEILINGS[name]
    assert ps.size_error(name, "x" * (ceiling - 1)) is None
    assert ps.size_error(name, "x" * ceiling) is None
    assert ps.size_error(name, "x" * (ceiling + 1)) is not None


def test_the_ceiling_is_measured_in_bytes_not_in_characters():
    """An em-dash is three bytes and one character, and only one of them is on disk.

    The first draft of this guard compared `len(text)` to a byte limit and passed
    a 20,815-byte file as 20,687 characters. On these two files that gap is not
    academic: both are written by an agent that uses em-dashes and arrow glyphs as
    routinely as punctuation, so a character count is a ceiling that quietly moves
    by the text's share of multibyte characters.
    """
    ceiling = ps.USER_MD_CEILING_BYTES
    glyph = "—"  # one character, three bytes
    assert len(glyph) == 1
    assert len(glyph.encode("utf-8")) == 3
    multibyte = glyph * (ceiling // 3 + 1)
    assert len(multibyte) < ceiling < len(multibyte.encode("utf-8")), (
        "fixture is not a character-count false negative"
    )
    assert ps.size_error("USER.md", multibyte) is not None, (
        "a USER.md under the ceiling in characters is over it in bytes, and the "
        "guard read the character count"
    )


def test_the_two_roots_that_decide_what_a_memory_file_is_agree():
    """The guard's root and the tool's root are the same directory, or the guard is dead.

    `agent_mcp.session` builds a memory file's path from `MEMORIES_ROOT`;
    `app.memory_ceiling` decides whether a path is a memory surface by resolving it
    against `MEMORIES_DIR`. Both are `Path.home() / "obsidian" / "lloyd"` literals,
    in two modules, with nothing between them — and if they ever differ, the guard
    silently allows every write in the lane it exists to police, because "not under
    the memories dir" and "not a memory file" are the same `None`.

    That is the same defect shape as #1010 itself, one level down: a check that reads
    a location the writer never writes to, and reports clean. Asserted as equality
    rather than patched in the fixture, so the fixture's own patching is what makes
    the pair movable in a test and this test is what keeps them movable *together*.
    """
    import agent_mcp.session as session
    import app.memory_ceiling as ceiling

    assert session.MEMORIES_ROOT == ceiling.MEMORIES_DIR, (
        f"the memory tools write under {session.MEMORIES_ROOT} while the byte "
        f"ceiling guards {ceiling.MEMORIES_DIR}: the guard would see neither"
    )


def test_an_unrelated_file_named_user_md_is_not_a_memory_surface(tmp_path):
    """The safety half of the root check, asserted as an absence.

    `Write` and `Edit` reach any path on the box, and every one of them asks this
    module whether to refuse. A copy of a memory file in a sandbox, a worktree or
    another test's `tmp_path` is a different document with the same name, and a
    guard that bounded it would break ordinary work in exactly the way that gets a
    guard disabled wholesale. The name alone is not the property; the path *under
    the memories root* is.
    """
    import app.memory_ceiling as ceiling

    lookalike = tmp_path / "sandbox" / "USER.md"
    lookalike.parent.mkdir()
    lookalike.write_text("short", encoding="utf-8")
    huge = "x" * (ceiling.memory_ceiling("USER.md") * 4)

    assert lookalike.resolve().parent != ceiling.MEMORIES_DIR
    assert ceiling.memory_write_error(lookalike, huge) is None
    # Same bytes, one directory over: the difference is the location, not the text.
    assert ceiling.memory_write_error(ceiling.MEMORIES_DIR / "USER.md", huge) is not None


@pytest.mark.parametrize("name", ["SOUL.md", "NOTES.md", "SKILL.md"])
def test_a_surface_that_is_not_a_loaded_memory_file_has_no_ceiling(name):
    """The absence half of the scope, asserted as an absence.

    `Write`, `Edit` and `vault_write` all route through the same guard, so its
    first job is to be silent about every other file in the tree — a `USER.md` in
    a worktree or a sandbox is a different document, and a guard that refused it
    would be a guard a writer works around. `SOUL.md` is the pointed case: it is
    loaded, and its bound is #377's two ratios plus the load-bearing markers, not
    a byte count. A size constant here would be a second, unblessed answer to a
    question that item already answered.
    """
    assert ps.memory_ceiling(name) is None
    assert ps.size_error(name, "x" * (ps.USER_MD_CEILING_BYTES * 4)) is None


@pytest.mark.parametrize("name", sorted(ps.MEMORY_CEILINGS))
def test_memory_add_refuses_an_entry_that_would_cross_the_ceiling(name, memories_root):
    """Clause 4: the refusal is at the write site, priced on the bytes it would write.

    Asserted on the file's bytes after the call as well as on the message, because
    "returned an error" and "refused the write" are different claims: the guard sits
    inside `commit_lock`, after the read and before `write_text_durable`, and a
    refusal that left the entry appended would be a refusal in the report only.
    """
    session, root = memories_root
    ceiling = ps.MEMORY_CEILINGS[name]
    before = "# Heading\n\n" + "x" * (ceiling - 120) + "\n"
    (root / name).write_text(before, encoding="utf-8")

    out = session._memory_add({"file": name, "entry": "- " + "y" * 300})

    assert "error" in out, out
    assert out["error"].startswith(f"{name} is "), out["error"]
    assert f"{ceiling:,}-byte ceiling" in out["error"], out["error"]
    assert (root / name).read_text(encoding="utf-8") == before


@pytest.mark.parametrize("name", sorted(ps.MEMORY_CEILINGS))
def test_memory_add_still_writes_an_entry_that_stays_under_the_ceiling(name, memories_root):
    """The same guard on the same file, one entry earlier: it must write.

    Without this the refusal test is satisfiable by a guard that always refuses —
    which is not hypothetical: the unlanded draft this replaced documented a
    shrink exemption it did not implement, so every write to an already-over file
    failed whatever it did to the size.
    """
    session, root = memories_root
    ceiling = ps.MEMORY_CEILINGS[name]
    (root / name).write_text("# Heading\n\n" + "x" * (ceiling - 400) + "\n",
                             encoding="utf-8")

    out = session._memory_add({"file": name, "entry": "- a remembered thing"})

    assert out.get("success") is True, out
    assert (root / name).read_text(encoding="utf-8").endswith("- a remembered thing\n")


@pytest.mark.parametrize("name", sorted(ps.MEMORY_CEILINGS))
def test_memory_replace_refuses_a_growth_past_the_ceiling(name, memories_root):
    """Clause 4, second tool: a replace is an append with a disguise.

    `new_text` longer than `old_text` grows the same file `memory_add` is refused
    for. Guarding only the append tool would leave the door the wider route walks
    through open — and `memory_replace` is how an entry gets *re*-written, which is
    how a memory file actually inflates: not one 95 KB write, but the same
    paragraph rewritten slightly longer a hundred times.
    """
    session, root = memories_root
    ceiling = ps.MEMORY_CEILINGS[name]
    before = "# Heading\n\nSOLE-MATCH\n" + "x" * (ceiling - 90) + "\n"
    (root / name).write_text(before, encoding="utf-8")

    out = session._memory_replace(
        {"file": name, "old_text": "SOLE-MATCH", "new_text": "z" * 300}
    )

    assert "error" in out, out
    assert out["error"].startswith(f"{name} is "), out["error"]
    assert f"{ceiling:,}-byte ceiling" in out["error"], out["error"]
    assert (root / name).read_text(encoding="utf-8") == before


@pytest.mark.parametrize("name", sorted(ps.MEMORY_CEILINGS))
def test_a_shrink_of_an_already_over_ceiling_memory_file_is_still_allowed(name,
                                                                         memories_root):
    """The difference between a ceiling and a freeze, on a file that starts over.

    `lloyd/MEMORY.md` was 69,849 B when this constant was set, so a guard that
    refused every over-ceiling write would refuse the trim that fixes it — and the
    only writer left is the one it is refusing. So the rule is directional: refuse
    what grows the file, allow what shrinks it, even while the result is still
    above the line. Above the ceiling the ceiling is absolute; below it the file's
    own size is the ratchet, because a smaller file cannot have regrown.
    """
    session, root = memories_root
    ceiling = ps.MEMORY_CEILINGS[name]
    before = "# Heading\n\nSOLE-MATCH\n" + "x" * (ceiling + 2000) + "\n"
    (root / name).write_text(before, encoding="utf-8")

    out = session._memory_replace(
        {"file": name, "old_text": "SOLE-MATCH", "new_text": "gone"}
    )

    assert out.get("success") is True, out
    after = (root / name).read_text(encoding="utf-8")
    assert "gone" in after and "SOLE-MATCH" not in after
    assert len(after.encode("utf-8")) < len(before.encode("utf-8"))
    assert len(after.encode("utf-8")) > ceiling, "fixture must stay over the line"


# ── the measurement has to contain the file it claims to bound (#1010 clause 5) ─


@pytest.fixture
def prompt_vault(tmp_path, monkeypatch):
    """A fake vault with both loaded memory files present, for platform measurement.

    `prompt_builder` resolves the canonical files through module-level
    `_CANON_*` constants, so patching those three is what makes an assertion about
    *platform selection* run without reading the live vault — the platform
    decision is the code path under test; which files are on disk is not.
    """
    vault = tmp_path / "vault"
    mems = vault / "lloyd"
    mems.mkdir(parents=True)
    soul = mems / "SOUL.md"
    soul.write_text(
        "---\ntype: note\n---\n# Lloyd Operating Contract\n\nBE BOUNDED\n",
        encoding="utf-8",
    )
    (mems / "MEMORY.md").write_text(
        "---\ntype: note\n---\n# Lloyd Long-Term Memory\n\nA remembered fact.\n",
        encoding="utf-8",
    )
    (mems / "USER.md").write_text(
        "---\ntype: note\n---\n# User (Alan)\n\n" + USER_MARKER + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(pb, "_CANON_SOUL_PATH", soul)
    monkeypatch.setattr(pb, "_CANON_MEMORIES_DIR", mems)
    monkeypatch.setattr(pb, "_CANON_SKILLS_DIRS", [tmp_path / "no-skills"])
    monkeypatch.delenv("LLOYD_OVERLAY_DIR", raising=False)
    return mems


def test_the_user_platform_ceiling_is_measured_on_a_string_that_contains_user_md(
    prompt_vault,
):
    """Clause 5: the measured string must contain USER.md's own bytes, on a user platform.

    Before #1010 the only size assertion in this file floored the measured
    `memories` component (`> 10_000` on the branch that died unlanded), and
    MEMORY.md alone clears any floor of that shape — so `over_budget=False` was a
    verdict about a string that had never held the file being bounded. A floor on
    a total cannot discriminate one file from another; *inclusion* can, which is
    what is asserted here: the fixture USER.md's own text, inside the string that
    is about to be measured.

    The platform is named explicitly. `_memory_files_for("")` falls open to the
    full set, so a test that passed no platform would still pass on a box where
    the default flipped — the direction `_memory_files_for` documents in its own
    docstring, and the reason the user platform here is spelled rather than
    defaulted.
    """
    soul = pb._load_soul(None)
    files = pb._memory_files_for("mission-control")
    assert "USER.md" in files, files
    memories = pb._load_memories(soul=soul, files=files)

    user_text = (prompt_vault / "USER.md").read_text(encoding="utf-8").strip()
    assert user_text in memories, "USER.md's bytes are not inside the measured string"

    report = pb.measure_prompt({"SOUL.md": soul, "memories": memories})
    assert report["over_budget"] is False, report
    # The same strings, checked per file: the total passing is not what bounds a
    # component, and the total here could be under budget with either file at its
    # ceiling, because the two ceilings together exceed `PROMPT_BUDGET_CHARS`.
    assert ps.size_error("USER.md", user_text) is None
    assert ps.size_error(
        "MEMORY.md", (prompt_vault / "MEMORY.md").read_text(encoding="utf-8")
    ) is None


def test_a_worker_platform_measurement_cannot_certify_the_user_ceiling(prompt_vault):
    """Clause 5's negative control: the worker-path measurement is green and means nothing.

    `prompt_builder._memory_files_for` drops USER.md for a non-user platform, and
    that is a deliberate change (`harness.worker_prompt.drop_memory_files`) which
    costs a worker turn ~20k tokens of the person Lloyd works for. It is also the
    reason live `PROMPT_BUDGET` telemetry structurally cannot see USER.md regrowth:
    a worker turn logs `memories=21777c` for MEMORY.md alone while a chat turn on
    the same file logs 38,056c. So a ceiling assertion satisfied by that line is
    not a ceiling assertion, and the way to say so in code is to demonstrate the
    failure with the file *doubled*: the naive measurement still reports
    `over_budget=False`, because the file it measured never contained the growth.

    Paired with the test above, which asserts inclusion on the user platform, so
    the pair pins both halves: what a real measurement must contain, and what a
    measurement without it cannot detect.
    """
    soul = pb._load_soul(None)
    files = pb._memory_files_for("worker")
    assert "USER.md" not in files, files

    user_file = prompt_vault / "USER.md"
    doubled = user_file.read_text(encoding="utf-8") + "z" * ps.USER_MD_CEILING_BYTES
    user_file.write_text(doubled, encoding="utf-8")
    # The file is now unambiguously over its ceiling...
    assert ps.size_error("USER.md", doubled) is not None

    memories = pb._load_memories(soul=soul, files=files)
    assert USER_MARKER not in memories, "the worker platform loaded USER.md"
    report = pb.measure_prompt({"SOUL.md": soul, "memories": memories})
    assert report["over_budget"] is False, (
        "the worker-path measurement caught a doubled USER.md, so this negative "
        "control no longer proves what clause 5 needs it to prove"
    )


# ── which assertions are allowed to certify the ceiling (#1010 clause 5) ──────
#
# Clause 5 is a claim about an assertion, not about a value: the size bound must
# not be satisfiable by a measurement that omits USER.md. A value can be asserted;
# "which function produced this number" can only be read. That is a legitimate
# source assertion and `tests/test_prompt_surface_guard.py` sets the precedent (it
# greps `vault_round` for the calls it replaced, for the same reason). Here the
# node set is the sharp half: a future edit that moves the bound to a
# platform-blind measurement would otherwise delete this guard's own subject and
# still pass.

#: Every node in this repo that makes a size claim about the loaded-memory prompt,
#: by node id — the positive bound and the negative control that shows what a
#: platform-blind bound cannot detect. An assertion not named here is not qualified
#: to certify the ceiling, and both of these read `prompt_builder` because it is the
#: only module that knows which memory files a given platform loads.
CEILING_ASSERTION_NODES = {
    "tests/test_prompt_surface_budget.py"
    "::test_the_user_platform_ceiling_is_measured_on_a_string_that_contains_user_md",
    "tests/test_prompt_surface_budget.py"
    "::test_a_worker_platform_measurement_cannot_certify_the_user_ceiling",
}

#: The files allowed to hold one. Both are loaded-prompt guards; an assertion in a
#: file that does not run on the gate could certify the bound invisibly.
CEILING_ASSERTION_FILES = [
    Path(__file__),
    Path(__file__).with_name("test_prompt_surface_guard.py"),
]


def _ceiling_assertion_provenance() -> tuple[set[str], set[str]]:
    """Which certified nodes were located, and which `prompt_builder` calls they make.

    A name match against the file's whole text would be satisfied by a docstring or
    a comment that merely mentions one of the nodes, so an AST walk extracts the
    actual function definitions and reports the ones it located — the count is
    derived from what the walk matched, never from the size of the set it was given.

    The second element is collected from `ast.Call` nodes only, which is what makes
    it worth computing: prose is free, a call is not. A body that names
    `prompt_builder` in its docstring and measures files it opened itself would
    satisfy a substring test — the first draft of this guard did exactly that, and
    clause 5 is a claim about where a number came from, not about which words appear
    near it.
    """
    import ast

    found: set[str] = set()
    pb_calls: set[str] = set()
    for path in CEILING_ASSERTION_FILES:
        node_id_prefix = f"{path.parent.name}/{path.name}::"
        src = path.read_text(encoding="utf-8")
        for node in ast.parse(src).body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            node_id = f"{node_id_prefix}{node.name}"
            if node_id not in CEILING_ASSERTION_NODES:
                continue
            found.add(node_id)
            for call in ast.walk(node):
                func = getattr(call, "func", None)
                if (isinstance(func, ast.Attribute)
                        and isinstance(func.value, ast.Name)
                        and func.value.id in ("pb", "prompt_builder")):
                    pb_calls.add(func.attr)
    return found, pb_calls


def test_only_the_prompt_builder_measurement_can_certify_the_ceiling():
    """Clause 5's guard: the certified assertions CALL `prompt_builder`, not re-derive.

    The failure this pins is the specific one #507 shipped with: an assertion that
    measured `SOUL.md + memories` by summing the files it chose to read, on the
    platform whose file list is *not* the user's. `prompt_builder` is what answers
    "which memory files does this platform load", so an assertion written anywhere
    else can pass while USER.md is not in the prompt at all.

    It can fail in the two ways that matter. A node renamed out of the set is
    reported by the first assertion; an assertion rewritten to open the files
    itself is reported by the second, which reads `ast.Call` nodes rather than the
    source text and so cannot be satisfied by a docstring that merely names the
    module. It is a guard over test prose and it is not a proof: the proof is that
    the positive test asserts USER.md's own text is inside the measured string, and
    the negative control proves the platform-blind reading cannot see a doubled
    USER.md at all.
    """
    found, pb_calls = _ceiling_assertion_provenance()
    assert found == CEILING_ASSERTION_NODES, (
        f"located {len(found)} of {len(CEILING_ASSERTION_NODES)} ceiling assertions; "
        "a node was renamed or deleted out of the set, and an unlisted assertion is "
        "not qualified to certify the bound. Missing: "
        f"{sorted(CEILING_ASSERTION_NODES - found)}"
    )
    assert {"_memory_files_for", "_load_memories"} <= pb_calls, (
        f"{sorted(CEILING_ASSERTION_NODES)} must call into prompt_builder, and these "
        f"are the calls it actually makes: {sorted(pb_calls) or 'none'}. "
        "`_memory_files_for` decides which memory files a platform loads and "
        "`_load_memories` builds the string that is then measured; an assertion that "
        "names the module only in prose, or re-derives the prompt from files it "
        "picked itself, cannot tell a user-platform prompt from a worker-platform one "
        "— which is the whole of clause 5"
    )


# ── the lanes the nightly job actually walks: Write, Edit, vault_write ────────
#
# The first gate's review rung refused this round for exactly one reason, and it is
# the reason this section exists: "Write (agent_mcp/builtin_fs.py:437), Edit (:539)
# and vault_write (agent_mcp/vault.py:1433) lanes now carry the ceiling but no test in
# either changed file exercises them — the lanes the nightly knowledge-write job
# walks; no clause requires them, so nothing refuses." It is right, and it is the
# sharper half of the item: #507's 48 KB → 95 KB climb happened through `Write` on an
# absolute path, because the knowledge-write skill's own text says "Absolute paths,
# `Read`/`Write`/`Bash` only. `vault_read`/`vault_write` reject…". Clause 4 names
# `memory_add`/`memory_replace`; those are the lane that was already guarded and the
# lane the regrowth did NOT come through. A ceiling asserted only on the append tool
# is a ceiling on the route nobody walks.
#
# Each lane gets both halves. The refusal half says the guard is wired into that
# handler; the acceptance half says it is wired in without becoming a freeze — the
# `Write` of a file under its ceiling and the `Edit` that shrinks a file over it are
# the two operations a trim needs, and a guard that refuses them is the guard that
# makes its own repair impossible (the bug the unlanded `f51ecd3` draft carried).

#: Session id bound so the Write/Edit lane runs its real gate rather than the
#: no-session short-circuit unit tests normally fall into.
LANE_SID = "20260923_1010_lanes"


@pytest.fixture
def lane_roots(tmp_path, monkeypatch):
    """A scratch `$HOME` whose `obsidian/lloyd/` is also the ceiling's root.

    Both roots move together, for the reason `test_the_two_roots_that_decide_what_a_
    memory_file_is_agree` pins: the Write/Edit deny-set is `$HOME`-relative and
    resolved per call (#1049), so a scratch home is the only way onto the lane at all,
    while `app.memory_ceiling.MEMORIES_DIR` is a `Path.home()` literal frozen at import
    and therefore does NOT follow `HOME`. Move one and not the other and the guard
    resolves the target outside its root, answers "not a loaded memory file", and
    every write to a relocated memory file is silently allowed — #1010's own defect
    shape one level down.

    The change ledger is switched off because these tests assert on the bytes on disk,
    exactly as `tests/test_builtin_fs_protected_write.py` does for the same lane.
    """
    import agent_mcp.builtin_fs as FS
    import agent_mcp.vault as VT
    import app.memory_ceiling as ceiling
    from agent_mcp import _change_ledger

    home = tmp_path / "home"
    mems = home / "obsidian" / "lloyd"
    mems.mkdir(parents=True)
    (mems / "MEMORY.md").write_text("# Lloyd Long-Term Memory\n\nA fact.\n",
                                    encoding="utf-8")
    (mems / "USER.md").write_text("# User (Alan)\n\nA preference.\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(ceiling, "MEMORIES_DIR", mems)
    monkeypatch.setattr(VT, "VAULT", home / "obsidian")
    monkeypatch.setattr(FS, "get_bound_session", lambda: LANE_SID)
    monkeypatch.setattr(_change_ledger, "enabled", lambda: False)
    FS.reset_read_records()
    return FS, VT, mems


def _lane_text(res) -> str:
    return res.content[0].text


def _names_file_and_limit(msg: str, name: str, ceiling_bytes: int) -> None:
    """The one assertion every refusal here shares: file, limit, and nothing else.

    Reused rather than repeated per lane because a lane that refuses with a generic
    "too big", or with the other file's number, is the failure the whole item is
    about — a refusal a reader cannot act on.
    """
    assert f"{name} is " in msg, msg
    assert f"{ceiling_bytes:,}-byte ceiling" in msg, msg


@pytest.mark.parametrize("name", sorted(ps.MEMORY_CEILINGS))
async def test_the_write_lane_refuses_a_memory_file_it_would_push_over(
    name, lane_roots,
):
    """`Write` across the real tool boundary, not the guard function.

    Called through `call_tool` after a real `Read`, so the request crosses the same
    dispatch, `_gate_check` and `commit_lock` path a nightly turn crosses. `Read`
    first is load-bearing, not ceremony: without it the refusal under test would be
    the #1049 clobber gate's, and this file would be pinning the wrong guard.
    """
    FS, _VT, mems = lane_roots
    target = mems / name
    before = target.read_text(encoding="utf-8")
    assert not (await FS.call_tool("Read", {"file_path": str(target)})).is_error

    res = await FS.call_tool("Write", {"file_path": str(target),
                                       "content": "x" * (ps.MEMORY_CEILINGS[name] + 1)})

    assert res.is_error is True, _lane_text(res)
    _names_file_and_limit(_lane_text(res), name, ps.MEMORY_CEILINGS[name])
    assert target.read_text(encoding="utf-8") == before, (
        "a refused Write must leave the file exactly as it was")


async def test_the_write_lane_still_writes_a_memory_file_under_its_ceiling(lane_roots):
    """The acceptance half on the lane the nightly job uses.

    `USER.md` at one byte under the ceiling is precisely where #507's trim left the
    live file (16,368 B against 16,384 B), so a guard that refused this write would
    refuse the maintenance write that keeps it there — a freeze with an error message.
    """
    FS, _VT, mems = lane_roots
    target = mems / "USER.md"
    content = "x" * (ps.USER_MD_CEILING_BYTES - 1)
    assert not (await FS.call_tool("Read", {"file_path": str(target)})).is_error

    res = await FS.call_tool("Write", {"file_path": str(target), "content": content})

    assert res.is_error is False, _lane_text(res)
    assert target.read_text(encoding="utf-8") == content


@pytest.mark.parametrize("name", sorted(ps.MEMORY_CEILINGS))
async def test_the_edit_lane_refuses_a_growth_past_the_ceiling(name, lane_roots):
    """`Edit`: the lane by which an entry gets *re*-written, which is how inflation lands.

    A replace with a longer `new_string` grows the same file `memory_add` is refused
    for, and unlike a whole-file `Write` it never asks the caller to hold the current
    bytes — so the prospective text has to be computed inside the lock, from the bytes
    the lane just read, which is where the guard sits.
    """
    FS, _VT, mems = lane_roots
    target = mems / name
    target.write_text("MARKER-TEXT\n" + "x" * (ps.MEMORY_CEILINGS[name] - 200),
                      encoding="utf-8")
    before = target.read_text(encoding="utf-8")
    assert not (await FS.call_tool("Read", {"file_path": str(target)})).is_error

    res = await FS.call_tool("Edit", {"file_path": str(target),
                                      "old_string": "MARKER-TEXT",
                                      "new_string": "z" * 900})

    assert res.is_error is True, _lane_text(res)
    _names_file_and_limit(_lane_text(res), name, ps.MEMORY_CEILINGS[name])
    assert target.read_text(encoding="utf-8") == before


async def test_the_edit_lane_still_shrinks_a_file_that_starts_over_its_ceiling(
    lane_roots,
):
    """A trim through the same lane that just refused a growth: the anti-freeze half.

    Starts two thousand bytes over the line, edits a long marker down to four
    characters, and the result is STILL over the ceiling — allowed anyway, because the
    rule is directional: refuse what grows, allow what shrinks. This is the behaviour
    the `f51ecd3` draft documented in its docstring and did not implement, which is how
    a note landed claiming a guard that could not be repaired by its own tool.
    """
    FS, _VT, mems = lane_roots
    target = mems / "USER.md"
    target.write_text("MARKER-TEXT\n" + "x" * (ps.USER_MD_CEILING_BYTES + 2000),
                      encoding="utf-8")
    before = target.read_bytes()
    assert not (await FS.call_tool("Read", {"file_path": str(target)})).is_error

    res = await FS.call_tool("Edit", {"file_path": str(target),
                                      "old_string": "MARKER-TEXT",
                                      "new_string": "gone"})

    assert res.is_error is False, _lane_text(res)
    after = target.read_bytes()
    assert b"gone" in after and b"MARKER-TEXT" not in after
    assert len(after) < len(before), "the trim itself was refused"
    assert len(after) > ps.USER_MD_CEILING_BYTES, (
        "the fixture has to stay over the line after the edit, or this asserts an "
        "ordinary write under the ceiling instead of the shrink exemption that keeps "
        "a trim possible once a file is over it")


@pytest.mark.parametrize("name", sorted(ps.MEMORY_CEILINGS))
def test_vault_write_refuses_a_memory_file_over_its_ceiling(name, lane_roots):
    """`vault_write`, the third lane — and the one whose root the guard must follow.

    `_vault_write` builds its target from the module-level `VAULT`, so patching that
    alone moves the write; the ceiling decides on `MEMORIES_DIR`. If those two stop
    naming one directory this refusal disappears without an error anywhere, which is
    why the fixture patches both and this test sits beside the constant-equality guard
    rather than replacing it.
    """
    _FS, VT, mems = lane_roots
    target = mems / name
    before = target.read_text(encoding="utf-8")

    out = VT._vault_write({"path": f"lloyd/{name}",
                           "content": "x" * (ps.MEMORY_CEILINGS[name] + 1)})

    assert "error" in out, out
    _names_file_and_limit(out["error"], name, ps.MEMORY_CEILINGS[name])
    assert target.read_text(encoding="utf-8") == before


def test_vault_write_still_writes_a_memory_file_under_its_ceiling(lane_roots):
    """The same lane's acceptance half, so the ceiling is not a read-only vault."""
    _FS, VT, mems = lane_roots
    content = "# User (Alan)\n\n" + "x" * 500

    out = VT._vault_write({"path": "lloyd/USER.md", "content": content})

    assert out.get("success") is True, out
    assert (mems / "USER.md").read_text(encoding="utf-8") == content
