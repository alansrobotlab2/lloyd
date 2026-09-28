"""`architecture/harness.md` must not tell a reader a shipped switch ships off.

The P1 section of that doc described the three `harness.prompt_layout` switches
as "all shipping off" for three days after `ca7eb481` (2026-09-25, a
config-only commit) rolled `session_state` out to `system_tail`, and two later
commits touched the same file without correcting the sentence. That is the
#505 class recorded in `tests/test_automod_doc_claims.py`: a doc sentence that
disarms a mechanism the code has already armed. Here the cost of believing it
is that a reader skips the prefix-reuse path when reasoning about cache
behaviour, or re-runs a rollout step that already shipped.

So the doc's shipped-state claims are pinned to the shipped config, read the
way the runtime reads it (`CONFIG['harness']['prompt_layout']`), not to a
literal in this file. A future flip of a switch fails a test here and says to
update the prose, instead of quietly making the prose stale again.

Only claims a reader would act on are pinned — which placement is shipped, and
which switches are still off. Prose about what each layout renders is not
pinned, and the two switches that genuinely do ship off
(`freeze_memory`, `replay_injected_context`) stay asserted as off so a fix for
the one wrong sentence cannot sweep them into the rewrite.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.config import CONFIG

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "harness.md"
DOC_TEXT = DOC.read_text(encoding="utf-8")

#: The three switches `architecture/harness.md`'s P1 section describes, named
#: as `config.yaml` names them under `harness.prompt_layout`.
SWITCHES = ("session_state", "freeze_memory", "replay_injected_context")

#: The commit that rolled `session_state` out to `system_tail` (P1 rollout
#: step 1, 2026-09-25). The doc records a rollout step as shipped by naming it.
ROLLOUT_SHA = "ca7eb481"

P1_HEADING = "### P1 — cross-turn prefix reuse"
CLOSING_HEADING = "### Closing summary"
PENDING_HEADER = "**Pending measurements and decisions**"
SHIPS_ON_HEADER = "**Ships on**"
SHIPS_IN_SHADOW_HEADER = "**Ships in shadow**"
SHIPS_OFF_HEADER = "**Ships off / as today**"

#: The form the doc states a shipped value in: the backticked
#: "`<key>: <value>`" pair, e.g. `session_state: system_tail`. A value followed
#: by anything else (the option enum `system_head | system_tail | user_tail`)
#: is not a shipped-state claim, and this pattern does not match it.
SHIPPED_PAIR_RE = re.compile(
    r"`(" + "|".join(SWITCHES) + r"):\s*([A-Za-z0-9_]+)`")

#: The doc's own option list for the placement switch: the line
#: `- **`session_state: system_head | system_tail | user_tail`**`. Requires the
#: pipe, so a shipped-state pair (`session_state: system_tail`) cannot be
#: mistaken for the option list. Used for the negative control, so the
#: substituted values are the legal ones and not literals this file invented.
LAYOUT_ENUM_RE = re.compile(r"`session_state:\s*([A-Za-z_ |]*\|[A-Za-z_ |]*)`")

#: Sentences claiming the whole switch set is off. Per-switch off-claims are
#: NOT banned — two of the three switches really do ship off — so each pattern
#: carries a collective quantifier ("all", "three").
COLLECTIVE_OFF_RES = [
    re.compile(r"\ball shipping off\b", re.I),
    re.compile(r"\ball (?:ship|ships) (?:off|as today)\b", re.I),
    re.compile(r"\ball three\b[^\n]{0,60}\boff\b", re.I),
    re.compile(r"\b(?:all|both) (?:switches|prompt_layout switches)\b[^\n]{0,40}\boff\b", re.I),
]

# The two sentences this item found stale, kept verbatim as the controls'
# inputs: the detector below must fire on the pre-fix prose and stay quiet on
# the true prose, or a green here says nothing.
PREFIX_LEAD_SENTENCE = ("Three switches under `harness.prompt_layout`, "
                        "all shipping off:")
PREFIX_PENDING_P1_BULLET = ("- **P1** — rollout `system_tail`, then the "
                            "`user_tail` A/B (`brain1.turn_start_prefix`), "
                            "then `freeze_memory`.")
PREFIX_SHIPS_OFF_CLAUSE = ("P1's `prompt_layout` (`system_head`, "
                           "`freeze_memory: false`)")
# True prose about two of the three switches, which the collective ban must
# NOT match.
TRUE_PER_SWITCH_OFF_SENTENCE = ("`freeze_memory` and `replay_injected_context` "
                                "still ship off.")


# ── extraction ──────────────────────────────────────────────────────────────

def _section(text: str, heading: str) -> str:
    """The body under `heading`, up to the next `###`/`##` heading.

    Asserts the heading exists: a green must never come from a scan that found
    no section to read.
    """
    out, seen = [], False
    for line in text.splitlines(keepends=True):
        if line.startswith("### ") or line.startswith("## "):
            if seen:
                break
            seen = line.startswith(heading)
            continue
        if seen:
            out.append(line)
    assert seen, f"no heading starting {heading!r} in {DOC.relative_to(ROOT)}"
    return "".join(out)


def _paragraph(text: str, start_header: str, end_header: str) -> str:
    start = text.find(start_header)
    assert start >= 0, f"no {start_header!r} in the closing summary"
    end = text.find(end_header, start)
    assert end > start, f"no {end_header!r} after {start_header!r}"
    return text[start:end]


def _stated_placements(text: str) -> dict[str, list[str]]:
    """Every shipped-state pair the text states, key -> sorted distinct values.

    Distinct values, so a doc that says `system_head` here and `system_tail`
    there shows up as two values instead of whichever the scan happened to
    keep.
    """
    got: dict[str, set[str]] = {}
    for key, value in SHIPPED_PAIR_RE.findall(text):
        got.setdefault(key, set()).add(value)
    return {key: sorted(values) for key, values in got.items()}


def _coerce(value: str):
    return {"false": False, "true": True}.get(value, value)


def _shipped_config() -> dict:
    """`harness.prompt_layout` as the runtime reads it, all three keys."""
    layout = dict(((CONFIG.get("harness") or {}).get("prompt_layout") or {}))
    missing = [key for key in SWITCHES if key not in layout]
    assert not missing, f"CONFIG has no harness.prompt_layout key(s): {missing}"
    return layout


def _legal_layouts(text: str) -> list[str]:
    """The option list the doc itself carries for `session_state`."""
    match = LAYOUT_ENUM_RE.search(text)
    assert match, "the P1 section states no `session_state: …` option list"
    layouts = [item.strip() for item in match.group(1).split("|") if item.strip()]
    assert layouts, "the `session_state` option list is empty"
    return layouts


def _placement_problems(text: str, prompt_layout: dict) -> list[str]:
    """Where the doc's stated shipped values disagree with the shipped config.

    Three ways to be wrong: a key stated with two different values, a stated
    value that is not what config ships, or a stated key config does not have.
    An empty extraction is not silence but a hard failure — a regex that
    stopped matching the prose must not read as "the doc agrees".
    """
    stated = _stated_placements(text)
    assert stated, (
        "no `key: value` shipped-state pair extracted from "
        f"{DOC.relative_to(ROOT)} — the doc stopped stating its shipped state "
        "in the form this test reads, so agreement cannot be checked")
    problems = []
    for key, values in sorted(stated.items()):
        if len(values) > 1:
            problems.append(f"{key}: stated with {len(values)} values {values}")
        elif key not in prompt_layout:
            problems.append(f"{key}: the doc names a switch config has not")
        elif _coerce(values[0]) != prompt_layout[key]:
            problems.append(
                f"{key}: the doc states {values[0]!r}, config ships "
                f"{prompt_layout[key]!r}")
    for key in sorted(prompt_layout):
        if key not in stated:
            problems.append(f"{key}: no shipped value stated in the doc")
    return problems


def _assert_doc_matches(text: str, prompt_layout: dict) -> None:
    problems = _placement_problems(text, prompt_layout)
    assert not problems, "doc/config drift: " + "; ".join(problems)


def _collective_off_claims(text: str) -> list[str]:
    """Lines claiming the whole prompt_layout switch set is off."""
    return [line.strip() for line in text.splitlines()
            if any(pattern.search(line) for pattern in COLLECTIVE_OFF_RES)]


def _pending_bullets(closing: str) -> list[str]:
    """The bullets of the closing summary's pending list, joined per bullet."""
    start = closing.find(PENDING_HEADER)
    assert start >= 0, f"no {PENDING_HEADER!r} in the closing summary"
    bullets: list[str] = []
    current: str | None = None
    for line in closing[start:].splitlines():
        if line.startswith("- "):
            if current:
                bullets.append(current)
            current = line
        elif current and line.strip():
            current += " " + line.strip()
        elif current:
            bullets.append(current)
            current = None
    if current:
        bullets.append(current)
    return bullets


def _bullet_ids(bullets: list[str]) -> set[str]:
    return {match.group(1) for bullet in bullets
            if (match := re.match(r"- \*\*([\w.]+)\*\*", bullet))}


def _pending_steps_already_shipped(bullets: list[str], shipped_layout: str,
                                   sha: str) -> list[str]:
    """Pending bullets naming the shipped layout without naming its commit.

    A pending step may name any layout it likes as long as it says which
    already-shipped layout it is not: `system_tail` in a pending bullet with no
    commit beside it is the doc listing a shipped rollout step as to-do.
    """
    needle = f"`{shipped_layout}`"
    return [bullet for bullet in bullets
            if needle in bullet and sha not in bullet]


def _layout_names_in(paragraph: str, layouts: list[str]) -> list[str]:
    """Layout values named anywhere in `paragraph`, as `system_head` etc."""
    return [layout for layout in layouts
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(layout)}(?![A-Za-z0-9_])",
                         paragraph)]


# ── clause 1: no "all shipping off", and the P1 section says what ships ──────

def test_the_doc_never_claims_all_three_prompt_layout_switches_ship_off():
    """One of the three switches is live, so "all shipping off" is false.

    Only collective claims are banned: `freeze_memory` and
    `replay_injected_context` really do ship off, and a per-switch sentence
    saying so must stay green — a ban that fired on those would push the fix
    towards deleting true sentences.
    """
    assert _collective_off_claims(DOC_TEXT) == [], (
        "the doc still claims the whole prompt_layout switch set ships off; "
        "session_state is live")
    # Positive control: the detector fires on the sentence this item found.
    assert _collective_off_claims(PREFIX_LEAD_SENTENCE) == [PREFIX_LEAD_SENTENCE]
    # Precision control: it does not fire on a true two-of-three sentence.
    assert _collective_off_claims(TRUE_PER_SWITCH_OFF_SENTENCE) == []


def test_the_p1_section_states_the_shipped_session_state_placement():
    """The P1 section states `session_state` in the shipped-state form, and its
    value is the one config ships."""
    stated = _stated_placements(_section(DOC_TEXT, P1_HEADING))
    assert "session_state" in stated, (
        "the P1 section states no shipped `session_state` value — the shipped "
        "placement belongs in that section, not only in the closing summary")
    assert _placement_problems(_section(DOC_TEXT, P1_HEADING), _shipped_config()) == []
    assert stated["session_state"] == [_shipped_config()["session_state"]]


# ── clause 2: the two switches that really are off stay named as off ─────────

def test_the_two_switches_that_ship_off_are_still_named_as_shipping_off():
    """`freeze_memory` and `replay_injected_context` stay stated as `false`.

    The fix for the one wrong sentence must not sweep these two into the
    rewrite: they are `false` in `config.yaml`, and the doc says so.
    """
    config = _shipped_config()
    assert config["freeze_memory"] is False and config[
        "replay_injected_context"] is False, (
        "this node asserts the doc against two switches config no longer "
        "ships off — update it to the state it finds")
    stated = _stated_placements(_section(DOC_TEXT, P1_HEADING))
    for key in ("freeze_memory", "replay_injected_context"):
        assert key in stated, f"the P1 section stopped stating {key}'s shipped state"
        assert stated[key] == ["false"], f"{key}: doc states {stated[key]}, not off"


# ── clause 3: the pin, over the whole doc, with a non-empty denominator ──────

def test_the_stated_placements_equal_the_shipped_config_over_the_whole_doc():
    """Every shipped-state claim in the doc equals the shipped config value.

    The denominator is asserted twice over: the extraction must be non-empty
    (`_placement_problems` raises on an empty scan), and all three switches
    must be stated somewhere in the file, so a doc that quietly dropped the
    claims passes nothing.
    """
    config = _shipped_config()
    assert _placement_problems(DOC_TEXT, config) == []
    stated = _stated_placements(DOC_TEXT)
    assert stated, "empty extraction"
    assert set(stated) == set(SWITCHES), (
        f"the doc states {sorted(stated)} of the three switches; the missing "
        "ones have no claim left to pin")


# ── clause 4: negative controls — the pin can fail ───────────────────────────

def test_the_pin_fails_when_the_doc_names_a_different_legal_layout():
    """Substituting another legal layout for the stated one makes it fail.

    The substituted values come from the doc's own option list, so the control
    covers the layouts a real flip could pick rather than literals chosen
    here. The pre-fix sentence is fed through the same detector as a fourth
    control: a scan of the file this item is fixing must not come back clean.
    """
    config = _shipped_config()
    layouts = _legal_layouts(_section(DOC_TEXT, P1_HEADING))
    assert config["session_state"] in layouts, (
        f"config ships {config['session_state']!r}, which the doc's option "
        "list does not offer — the doc is out of date in a way this test "
        "cannot express; add the value to the enum")
    alternatives = [item for item in layouts if item != config["session_state"]]
    assert alternatives, "the doc offers no alternative layout to test against"
    stated = _stated_placements(DOC_TEXT)
    assert stated["session_state"] == [config["session_state"]]
    for other in alternatives:
        bad = DOC_TEXT.replace(
            f"`session_state: {config['session_state']}`",
            f"`session_state: {other}`")
        with pytest.raises(AssertionError, match="session_state"):
            _assert_doc_matches(bad, config)
    # The same doc read against a flipped switch: the future rollout step the
    # doc already forecasts (`user_tail`) must fail here, not stale the prose.
    for other in alternatives:
        with pytest.raises(AssertionError, match="session_state"):
            _assert_doc_matches(DOC_TEXT, {**config, "session_state": other})
    # An extractor that matched nothing must not read as agreement: a doc that
    # dropped the pair for one switch, and prose with no pairs at all, both
    # fail rather than passing on a scan that found nothing.
    stripped = DOC_TEXT.replace(
        f"`session_state: {config['session_state']}`", "`session_state`")
    with pytest.raises(AssertionError, match="no shipped value stated"):
        _assert_doc_matches(stripped, config)
    with pytest.raises(AssertionError, match="pair extracted"):
        _assert_doc_matches("", config)


# ── clause 5: the closing summary records step 1 as shipped ──────────────────

def test_the_closing_summary_records_p1_step_one_as_shipped():
    """P1's `system_tail` step is recorded as shipped, commit and all, and the
    pending list no longer offers it as work.

    The pending list keeps its other entries: the control is that the pre-fix
    bullet — `system_tail` named with no commit — is caught by the same
    predicate, and P6/P9/P10 are still pending, so the list was edited rather
    than deleted to get here.
    """
    config = _shipped_config()
    closing = _section(DOC_TEXT, CLOSING_HEADING)
    bullets = _pending_bullets(closing)
    assert bullets, "the closing summary has no pending list to read"
    assert {"P6", "P9", "P10"} <= _bullet_ids(bullets), (
        "the pending list lost entries unrelated to P1; only P1's step-1 "
        "claim changed")
    assert _pending_steps_already_shipped(
        bullets, config["session_state"], ROLLOUT_SHA) == []
    assert _pending_steps_already_shipped(
        [PREFIX_PENDING_P1_BULLET], config["session_state"], ROLLOUT_SHA), (
        "the control bullet no longer trips the predicate")
    ships_on = _paragraph(closing, SHIPS_ON_HEADER, SHIPS_IN_SHADOW_HEADER)
    assert "P1" in ships_on and ROLLOUT_SHA in ships_on, (
        "the ships-on paragraph does not record P1 step 1 with its commit")
    assert _stated_placements(ships_on).get("session_state") == [
        config["session_state"]]


def test_the_ships_off_list_no_longer_names_a_session_state_layout():
    """The closing summary's off-list used to carry P1 as `system_head`.

    A layout value in the off-list is a placement claim about a switch that
    ships, so the off-list names no layout at all: the shipped placement is
    stated once, in the ships-on paragraph, and pinned there.
    """
    layouts = _legal_layouts(_section(DOC_TEXT, P1_HEADING))
    ships_off = _paragraph(_section(DOC_TEXT, CLOSING_HEADING),
                           SHIPS_OFF_HEADER, PENDING_HEADER)
    assert _layout_names_in(ships_off, layouts) == []
    assert _layout_names_in(PREFIX_SHIPS_OFF_CLAUSE, layouts) == ["system_head"], (
        "the control no longer trips the predicate")
