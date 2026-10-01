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


# ── #1786: the code's own annotations must not claim which layout ships ──────
#
# `architecture/harness.md` was fixed for this rollout by the pass above; the
# five annotations of the same switches inside the code were not, and four days
# after `ca7eb481` all five still named `system_head` as what runs — including
# the one in `config.yaml`, four lines above `session_state: system_tail`. The
# doc was pinned and the code was not because this file reads only the doc, so
# the same extraction is extended over the modules here.
#
# The failure mode is not a stale sentence a reader shrugs at. `system_head`
# annotated as "today's layout, byte-identical" tells a reader the session state
# rides at the head of the system prompt and the P1 tail mechanism is inert, and
# invites a replay-diff or prefix-stability test to assert `system_head` output
# while production renders `system_tail`. So the rule pinned here is structural
# rather than a value to re-sync: an annotation may state what a layout COSTS the
# prefix cache, and must not state which one is in force — `config.yaml` owns a
# moving value, and an annotation that repeats it is stale the day someone flips
# the switch. Each pre-fix phrase is kept verbatim below as a firing control, so
# a green cannot come from a scan that matched nothing.

LAYOUTS = ("system_head", "system_tail", "user_tail")

#: Which one is in force, stated in an annotation. Each phrase binds a layout
#: name (or the whole enum) to the present tense; a flip makes any of them a
#: lie, which is exactly what #1786 found on all five sites at once.
CLAIMS_A_LAYOUT_SHIPS_RES = [
    re.compile(r"today'?s (?:layout|output|prompt|default)", re.I),
    re.compile(r"\(today,?\s", re.I),
    re.compile(r"shipped defaults?\s*\(", re.I),
    re.compile(r"\b(?:ships|in force|is shipping)\b[^.]{0,40}\bsystem_head\b", re.I),
    re.compile(r"\bsystem_head\b[^.]{0,40}\b(?:today|ships|in force)\b", re.I),
]

#: The five pre-fix annotations, verbatim, as the controls' inputs.
PREFILLED_TAIL_CLAIM = (
    "With the shipped defaults (`system_head`, freeze off) the tail is always "
    '"\\" and every caller\'s `prefetched_text` is exactly what it was.')
PREFILLED_LAYOUT_NOTE = (
    "system_head — inside the system prompt, ahead of the harness hints "
    "(today's layout, byte-identical).")
PREFILLED_PLACEMENT_ARG = (
    "`system_head` is today's output byte for byte; `user_tail` leaves the "
    "state out entirely.")
PREFILLED_FALLBACK_CLAIM = "anything unreadable or unknown is today's layout"
PREFILLED_CONFIG_OPTION = "system_head (today, byte-identical)"

#: What an annotation must say instead: the cost to the prefix cache. Every one
#: of the three layouts has an honest answer, so requiring a marker of this set
#: in each annotation is a demand for content, not for a phrase.
CACHE_COST_MARKERS = ("prefix", "re-prefill", "re-prefills", "invalidate",
                      "invalidates", "byte-stable", "byte-stability",
                      "prefix-cache", "prefix-cached", "cache")

APP_PROMPT_LAYOUT = "app/prompt_layout.py"
APP_PROMPT_BUILDER = "app/prompt_builder.py"


def _prose(rel: str, func: str | None = None) -> str:
    """A module or function docstring, whitespace-flattened, read through `ast`.

    Not a file-text grep: `prompt_builder.py` runs to some 1,900 lines and
    mentions every layout name in code as well as prose, so a whole-file `in`
    check would let the annotation that matters be satisfied by an unrelated
    branch, and the line-wrapping #1786's triage recorded (`today's output`
    straddling two lines) makes a one-line grep a false zero.
    """
    import ast
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    if func is None:
        got = ast.get_docstring(tree)
    else:
        node = next((n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == func), None)
        assert node is not None, f"no function {func!r} in {rel}"
        got = ast.get_docstring(node)
    assert got, f"{rel}{'.' + func if func else ''} has no docstring to read"
    return " ".join(got.split())


def _annotation_block(rel: str, anchor: str) -> str:
    """The `#:` comment lines sitting directly above `anchor`, markers stripped."""
    lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
    hits = [i for i, ln in enumerate(lines) if ln.startswith(anchor)]
    assert len(hits) == 1, (
        f"{anchor!r} matched {len(hits)} lines in {rel}; the annotation graded "
        "below is only trustworthy if the anchor reads exactly one")
    block: list[str] = []
    i = hits[0]
    while i > 0 and lines[i - 1].lstrip().startswith("#:"):
        block.append(lines[i - 1].lstrip().lstrip("#:").strip())
        i -= 1
    assert block, f"no `#:` annotation above {anchor} in {rel} — deleted, not fixed"
    return " ".join(" ".join(reversed(block)).split())


def _comment_block_above(rel: str, key: str) -> str:
    """The `#` comment lines directly above `key:` in a YAML file."""
    lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
    hits = [i for i, ln in enumerate(lines) if ln.strip() == key]
    assert len(hits) == 1, f"{key!r} matched {len(hits)} lines in {rel}"
    block: list[str] = []
    i = hits[0]
    while i > 0 and lines[i - 1].lstrip().startswith("#"):
        block.append(lines[i - 1].lstrip().lstrip("#").strip())
        i -= 1
    assert block, f"no comment block above {key} in {rel}"
    return " ".join(" ".join(reversed(block)).split())


def _half_reason(text: str, half: str, other: str) -> str:
    """The words binding `half` of the turn tail to its reason.

    Anchored on the half's LAST mention on purpose. This docstring's bullet list
    maps each half to its switch earlier on, so a window taken from the FIRST
    mention reads those pre-existing bullets — the split paragraph #1786 added
    could then be deleted whole and the node would stay green, which is the gap
    the gate's review rung named. From the anchor the window runs to whichever
    comes first: the other half's mention, or the end of the sentence — where
    "end of sentence" is a period followed by whitespace or the end of the text,
    NOT a bare `". "`, which would close the window on the dot in `config.yaml`
    and have the assert read the tail of a filename.

    Two reasons stay separable because a docstring that explains both halves with
    one shared clause cannot put a switch, or a reason, in one window only.
    """
    i = text.rindex(half) + len(half)
    rest = text[i:]
    ends = [m.end() for m in re.finditer(r"\.(?:\s|$)", rest)]
    bounds = [j for j in (rest.find(other), *ends) if j >= 0]
    return rest[:min(bounds)] if bounds else rest


def _claims_a_layout_ships(text: str) -> list[str]:
    return [p.pattern for p in CLAIMS_A_LAYOUT_SHIPS_RES if p.search(text)]


def _priced(text: str, layout: str) -> bool:
    """Does the annotation for `layout` state a cache cost?

    Windowed on the annotation itself: the phrase `system_head` followed by the
    text up to the next layout name, so a cache word earned by a neighbouring
    layout's entry cannot carry this one.
    """
    start = text.find(layout)
    assert start >= 0, f"{layout!r} is not annotated at all in this text"
    rest = text[start + len(layout):]
    nxt = [x for x in (rest.find(other) for other in LAYOUTS if other != layout)
           if x >= 0]
    window = rest[:min(nxt)] if nxt else rest
    return any(marker in window for marker in CACHE_COST_MARKERS)


def test_the_turn_tail_docstring_gives_two_reasons_and_names_no_shipped_layout():
    """Clause 2: `app/prompt_layout.py` said the tail is always "" "with the
    shipped defaults (`system_head`, freeze off)".

    Two claims were fused there into one, and the fused form was wrong twice:
    with `system_tail` in force the `<session_state>` half of the tail is empty
    because the block moved inside the system prompt — a different fact from
    `system_head` — and the `<memory_delta>` half is empty because
    `freeze_memory` is off. The node therefore wants the reasons SPLIT (each
    half's reason must name its own switch and not the other's, which is what
    makes the sentence repairable rather than merely re-worded), the owner named
    as `config.yaml`, and no claim about which layout ships.
    """
    doc = _prose(APP_PROMPT_LAYOUT)

    for text, label in ((PREFILLED_TAIL_CLAIM, "the pre-fix tail claim"),
                        (PREFILLED_LAYOUT_NOTE, "the pre-fix layout annotation"),
                        (PREFILLED_PLACEMENT_ARG, "the pre-fix argument note"),
                        (PREFILLED_CONFIG_OPTION, "the pre-fix config option"),
                        (PREFILLED_FALLBACK_CLAIM, "the pre-fix fallback claim")):
        assert _claims_a_layout_ships(" ".join(text.split())), (
            f"{label} no longer trips the detector, so the check below on the "
            "real docstring would be a ban on nothing")
    assert _claims_a_layout_ships(doc) == [], (
        f"the module docstring still asserts which layout ships: "
        f"{_claims_a_layout_ships(doc)} — {doc[:180]!r}")

    halves = {"<session_state>": "user_tail", "<memory_delta>": "freeze_memory"}
    for tag, switch in halves.items():
        assert f"`{tag}`" in doc, (
            f"the docstring no longer names the `{tag}` half of the turn tail, "
            "so there is no reason left to split")
    state_half = _half_reason(doc, "`<session_state>`", "`<memory_delta>`")
    delta_half = _half_reason(doc, "`<memory_delta>`", "`<session_state>`")
    # Each half's window has to carry the REASON, not merely the switch: the
    # pre-existing bullet list already maps each half to its switch and to
    # `harness.prompt_layout.*`, so a window checked only for the switch name
    # would stay green with the explanatory paragraph deleted entirely — the
    # reading bug #1786 is about would come straight back.
    assert "system prompt" in state_half, (
        "the `<session_state>` half no longer says WHERE the block goes instead "
        f"of the tail, which is the reason it is empty: {state_half[:160]!r}")
    assert "snapshot" in delta_half, (
        "the `<memory_delta>` half no longer says WHY an unfrozen memory leaves "
        f"the note out: {delta_half[:160]!r}")
    assert "user_tail" in state_half and "freeze_memory" not in state_half, (
        "the `<session_state>` half of the tail is no longer explained by the "
        f"placement switch alone: {state_half[:180]!r} — the two reasons have "
        "been fused back into one, which is the conflation #1786 was filed on")
    assert "freeze_memory" in delta_half and "user_tail" not in delta_half, (
        "the `<memory_delta>` half of the tail is no longer explained by "
        "`freeze_memory` alone: {delta_half[:180]!r} — that switch, not the "
        "placement, is what leaves the delta note out")
    assert "config.yaml" in doc, (
        "the docstring states switch behaviour without naming the owner of the "
        "moving value, so the next reader has nowhere to look it up")
    # The same detector, aimed at the resolver's docstring in the other module:
    # #1786 found five sites, and the one that called the system_head fallback
    # "today's layout" is the one that tells a reader a silent fallback is what
    # production renders anyway.
    assert _claims_a_layout_ships(
        _prose(APP_PROMPT_BUILDER, "session_state_layout")) == [], (
        "the resolver's own docstring again calls its fallback what is in force")
    assert '"\\"' in doc or '""' in doc, (
        "the docstring no longer says what an empty tail is, which is the thing "
        "callers of `turn_tail` actually want to know")


def test_the_layout_annotations_price_a_cache_and_never_name_what_is_in_force():
    """Clause 3: the `SESSION_STATE_LAYOUTS` annotation and the
    `build_system_prompt` argument note both claimed a layout was current, and
    the note named two of the three layouts — the one in force among them
    nowhere.

    Each annotation has to earn its place by saying what that placement costs
    the prefix cache; that is stable across every flip, which is why it can be
    written down at all. The argument note is additionally required to name all
    three layouts: with `system_tail` missing, a reader choosing a value for
    `config.yaml` had no idea the shipped option existed.
    """
    enum_note = _annotation_block(APP_PROMPT_BUILDER, "SESSION_STATE_LAYOUTS =")
    arg_note = _prose(APP_PROMPT_BUILDER, "build_system_prompt")
    arg_note = arg_note[arg_note.rindex("`session_state` —"):]

    for text, label in ((PREFILLED_LAYOUT_NOTE, "the pre-fix layout annotation"),
                        (PREFILLED_PLACEMENT_ARG, "the pre-fix argument note")):
        assert _claims_a_layout_ships(" ".join(text.split())), (
            f"{label} no longer trips the detector and the ban below is vacuous")
    for text, label in ((enum_note, "the `SESSION_STATE_LAYOUTS` annotation"),
                        (arg_note, "the `build_system_prompt` note")):
        assert _claims_a_layout_ships(text) == [], (
            f"{label} states which layout is in force "
            f"({_claims_a_layout_ships(text)}) — it moved once already, on "
            "2026-09-25, and a comment cannot hold a value")
        for layout in LAYOUTS:
            assert _priced(text, layout), (
                f"{label} annotates {layout!r} without saying what it costs the "
                f"prefix cache: {text[:200]!r}")
    assert "`system_tail`" in arg_note and "system_tail" in enum_note, (
        "the layout in force since 2026-09-25 is not named by one of the two "
        "annotations, so a reader picking a value cannot see it")
    assert "config.yaml" in enum_note, (
        "the enum annotation names no owner for the moving value")


def test_the_config_annotation_prices_the_options_and_leaves_the_value_to_config():
    """Clause 5: `config.yaml` annotated `system_head (today, byte-identical)`
    four lines above `session_state: system_tail`.

    The same annotation contradicted the file it lives in, which is how the next
    flip gets missed: a reader editing the value has the wrong answer in their
    peripheral vision. The comment is therefore required to price the options and
    state no shipped value at all — so it cannot go stale when the switch moves,
    which is the property the gate's YAML token check protects by keeping this
    edit to comment lines. `CONFIG` still resolving `session_state` is asserted
    here from the runtime read, so a comment that quietly came with a value
    change fails too.
    """
    block = _comment_block_above("config.yaml", "prompt_layout:")

    assert _claims_a_layout_ships(PREFILLED_CONFIG_OPTION), (
        "the pre-fix option label no longer trips the detector")
    assert _claims_a_layout_ships(block) == [], (
        f"the comment above `prompt_layout:` labels a layout as current "
        f"({_claims_a_layout_ships(block)}); the value four lines below is the "
        "only statement of what is in force")
    for layout in LAYOUTS:
        assert _priced(block, layout), (
            f"the comment annotates {layout!r} without its cache cost: "
            f"{block[:220]!r}")
    assert not SHIPPED_PAIR_RE.search(block), (
        "the comment block states a shipped `key: value` pair, so it is a second "
        "place that has to be edited on every flip — the drift #1786 is about")

    shipped = _shipped_config()["session_state"]
    assert shipped in LAYOUTS, (
        f"CONFIG resolves session_state to {shipped!r}, which is not one of "
        f"{LAYOUTS} — the runtime and the enum have parted company")
    value_line = [ln for ln in (ROOT / "config.yaml").read_text().splitlines()
                  if ln.strip().startswith("session_state:")]
    assert len(value_line) == 1, (
        f"{len(value_line)} `session_state:` value lines in config.yaml; the "
        "single value is what this clause leaves as the only statement of force")
    assert value_line[0].strip() == f"session_state: {shipped}", (
        f"config.yaml's own value line ({value_line[0]!r}) is not what CONFIG "
        f"resolves ({shipped!r}) — an override or a second writer is in play, "
        "and no annotation in this repo can be trusted to describe it")


# ── #1962: config.yaml must not promise an action_review gate that was ruled out ─

#: The sentence `config.yaml:751` carried until #1962, verbatim. #1944 took
#: `warn` out of `action_review.MODES` and #1948 mapped a surviving `warn` onto
#: `shadow`; neither touched this comment, so the one sentence a person editing
#: the value reads kept promising a gate that was waiting for a number.
PREFIX_ACTION_REVIEW_SENTENCE = (
    "action_review.py). `warn` records like `shadow` until a threshold exists.")
STALE_ACTION_REVIEW_PHRASES = ("records like", "until a threshold exists")


def _stale_action_review_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines()
            if any(p in ln for p in STALE_ACTION_REVIEW_PHRASES)]


def _guard_half(block: str, name: str, until: str | None) -> str:
    """One guard's half of the P10 comment block: from `<name>:` to `<until>:`."""
    start = block.index(f"{name}:")
    end = block.index(f"{until}:", start) if until else len(block)
    return block[start:end]


def test_config_yaml_no_longer_promises_an_action_review_gate_awaiting_a_threshold():
    """#1962 clause 1. The pre-fix sentence is kept as the firing control."""
    assert len(_stale_action_review_lines(PREFIX_ACTION_REVIEW_SENTENCE)) == 1, (
        "the pre-fix sentence no longer trips the scan, so a clean config.yaml "
        "proves nothing")
    stale = _stale_action_review_lines(
        (ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert stale == [], f"config.yaml still carries the pre-#1944 promise: {stale}"


def test_the_action_review_comment_says_warn_is_deprecated_and_shadow_is_permanent():
    """#1962 clauses 2 and 3, read off the block above `action_review:` and
    checked against the module, so the comment cannot outlive a second ruling."""
    import app.harness.action_review as AR

    block = _comment_block_above("config.yaml", "action_review:")
    half = _guard_half(block, "action_review", "injection_probe")

    assert "warn" not in AR.MODES and AR.DEPRECATED_MODES.get("warn") == "shadow", (
        "action_review honours `warn` again or maps it elsewhere — the comment "
        "this test pins is now the stale side")
    assert "`warn` is deprecated for action_review" in half, half
    assert "maps to `shadow`" in half, half
    assert "off | shadow." in half and "off | shadow | warn" not in half, (
        f"the action_review half lists a mode the module does not honour: {half!r}")

    assert "records only" in half, half
    assert "permanent" in half and "reopening needs new evidence" in half, half
    assert ("module docstring" in half
            and "knowledge/ai/action-review-threshold-measurement.md" in half), (
        f"the block states a ruling without naming where it is written: {half!r}")
    assert "reopening needs new evidence" in " ".join((AR.__doc__ or "").split()), (
        "the module docstring the comment cites no longer carries the ruling")


def test_warn_stays_documented_as_a_live_mode_of_the_injection_probe():
    """#1962 clause 4: the mode-list change is scoped to `action_review`."""
    from agent_mcp import _injection_probe as P

    block = _comment_block_above("config.yaml", "action_review:")
    half = _guard_half(block, "injection_probe", None)
    assert "warn" in P.MODES
    assert "off | shadow | warn" in half, half
    assert "warn also appends one <warning>" in half, half
    assert "deprecated" not in half, (
        f"the probe half calls a mode it still honours deprecated: {half!r}")
