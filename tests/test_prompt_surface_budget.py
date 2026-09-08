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
# failing where the vault is absent; `live_vault` lets the selfmod gate exclude
# them with `-m "not live_vault"`.
#
# They read a file no round under test controls — an hourly autoresearch
# promotion or a nightly reflection job can rewrite `SOUL.md` between one round
# and the next. Left on the gate's hard `tests` rung, a re-inflation would fail
# every future round whatever its diff, which is how `test_tool_overrides.py`
# aborted three rounds in fifteen hours on 2026-09-07. The invariants are
# enforced at the writers instead — `scripts/selfmod/vault_round.py` and
# `scripts/autoresearch/promote.py` both call `prompt_surface.check_contract`
# before they commit — so this group is the reporting copy, not the enforcement.
vault_only = pytest.mark.skipif(
    not SOUL.exists(), reason="live vault prompt surface not on disk here"
)
live_vault = pytest.mark.live_vault

# `_sections`, `_gate_share`, `_prohibition_ratio` and `_shared_line_share`
# live in `prompt_surface` now, because the writers have to run the same checks
# and a second private definition of "is the contract bloated" is how the two
# halves drift apart. Thin aliases keep the assertions below readable.
_sections = ps.sections
_gate_share = ps.gate_share
_prohibition_ratio = ps.prohibition_ratio
_shared_line_share = ps.shared_line_share


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


def _shared_line_share(memory: str, soul: str) -> float:
    """Share of MEMORY.md's nonblank lines that are verbatim SOUL.md lines."""
    mem_lines = [ln for ln in memory.split("\n") if ln.strip()]
    soul_lines = {ln for ln in soul.split("\n") if ln.strip()}
    if not mem_lines:
        return 0.0
    return sum(1 for ln in mem_lines if ln in soul_lines) / len(mem_lines)


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
    """#464 — MEMORY.md was overwritten with SOUL.md; the contract arrived twice."""
    assert (memory_text.split("\n", 1)[0].strip()
            != soul_text.split("\n", 1)[0].strip()), (
        "MEMORY.md opens with SOUL.md's H1 — the operating-contract paste is back"
    )
    dup = _shared_line_share(memory_text, soul_text)
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
