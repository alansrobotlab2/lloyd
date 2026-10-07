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


# ── #2137: config.yaml's `inner_voice.model` pin must carry ONE dated verdict ─
#
# Two paragraphs sat above `model: primary` and contradicted each other: one
# ended "Revisit only with iv_grade evidence.", the next said "That evidence is
# now stale in the observer's favour… The pin stays until someone actually
# re-runs iv_grade against the 35B". The second paragraph was written on 2026-09-06
# (`608dd15c`); the engine it told a reader to re-run against was retired on
# 2026-09-20 when `secondary_enabled` went false and GPU 2 was reassigned to
# `djev` (`551e9044` rewrote that block and never touched this comment). So the
# file still ended on an instruction nobody can follow — and, worse, framed a
# shipped-state value as an open question.
#
# The pre-fix text is kept verbatim below as the firing control for every scan
# here, exactly as `9dc0403f` (#1962) does for the action_review comment: an
# empty scan on a file that never contained the phrase proves nothing.

#: The two paragraphs as `config.yaml:125-136` carried them until #2137, joined
#: with the blank comment line that separated them. The control text for clauses
#: 1 and 4 — the scan must fire on this and come back empty on the live file.
PREFIX_PIN_COMMENT = (
    "Pinned to primary deliberately. `secondary_enabled: true` (2026-09-03, "
    "de893d7) made resolve_model_alias stop rewriting secondary -> primary, "
    "which silently moved the observer from Flash-Next to Qwen3.5-4B. On the "
    "first day it ran there it intervened on 40% of LLM-judged events vs 1.7% "
    "on primary, fabricated a finding the primary never reported, and cancelled "
    "a turn. Revisit only with iv_grade evidence. "
    "That evidence is now stale in the observer's favour: the secondary became "
    "Qwen3.6-35B-A3B on 2026-09-06, so the 40%-intervention result was measured "
    "against a 4B model this slot no longer runs. The pin stays until someone "
    "actually re-runs iv_grade against the 35B — but the reason to re-run it is "
    "much stronger than it was.")

#: The two sentences the item's own premise greps for, verbatim.
STALE_PIN_PHRASES = ("stale in the observer", "Revisit only with iv_grade evidence")

#: A model name or a quant size: what a re-run instruction must not name. The
#: retirement is the whole point of the verdict, so an instruction that survives
#: naming an occupant is an instruction to re-run against a retired arm.
ENGINE_OR_QUANT_RE = re.compile(r"\b\d+B\b|\bGGUF\b|Qwen[\w.+-]*|Flash-Next", re.I)
#: A sentence telling the reader to run the observer eval again.
RERUN_INSTRUCTION_RE = re.compile(r"\bre-?runs?\b|\bre-?run\b|run it again", re.I)


def _inner_voice_section(rel: str = "config.yaml") -> str:
    """The `inner_voice:` block: its lines up to the next column-0 key.

    Scoped because the pin line is `model: primary` and that string occurs three
    times in this file (:137, :1646, :1675) — `_comment_block_above` asserts its
    anchor is unique and would fail on the assert, not on the comment.
    """
    lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
    top = [i for i, ln in enumerate(lines) if ln == "inner_voice:"]
    assert len(top) == 1, f"`inner_voice:` matched {len(top)} lines in {rel}"
    start = top[0]
    end = next((i for i, ln in enumerate(lines[start + 1:], start + 1)
                if ln and not ln[0].isspace() and not ln.lstrip().startswith("#")),
               len(lines))
    return "\n".join(lines[start:end])


def _pin_comment_block(rel: str = "config.yaml") -> str:
    """The `#` lines directly above `model:` inside the `inner_voice:` block."""
    sec_lines = _inner_voice_section(rel).splitlines()
    hits = [i for i, ln in enumerate(sec_lines) if ln.strip() == "model: primary"]
    assert len(hits) == 1, (
        f"`model: primary` matched {len(hits)} lines inside `inner_voice:`; the "
        "pin this comment describes is no longer uniquely identifiable there")
    block: list[str] = []
    i = hits[0]
    while i > 0 and sec_lines[i - 1].lstrip().startswith("#"):
        block.append(sec_lines[i - 1].lstrip().lstrip("#").strip())
        i -= 1
    assert block, f"no comment block above `model: primary` in `inner_voice:` ({rel})"
    return " ".join(" ".join(reversed(block)).split())


def _stale_pin_phrases(text: str) -> list[str]:
    return [p for p in STALE_PIN_PHRASES if p in text]


def _rerun_instructions(text: str) -> list[str]:
    """Sentences that tell the reader to re-run the observer eval, and name the
    engine or quant to run it against. An instruction with no named arm (a person
    has to stand a slot up first) is a condition, not a stale command."""
    return [s.strip() for s in re.split(r"(?<=\.)\s+", text)
            if RERUN_INSTRUCTION_RE.search(s) and ENGINE_OR_QUANT_RE.search(s)]


def test_config_yaml_no_longer_carries_the_two_contradictory_pin_paragraphs():
    """#2137 clause 1. Both premise greps return 0 on the live file, and the scan
    is shown to work by firing on the pre-fix text."""
    assert sorted(_stale_pin_phrases(PREFIX_PIN_COMMENT)) == sorted(STALE_PIN_PHRASES), (
        "the pre-fix text no longer trips the scan, so a clean config.yaml "
        "proves nothing — update the control text and the phrases together")
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    stale = _stale_pin_phrases(text)
    assert stale == [], f"config.yaml still carries the contradictory pin text: {stale}"
    # And what replaces them is ONE dated verdict: exactly one date-stamped
    # verdict line heads the block, so a later edit cannot stack a second,
    # unreconciled paragraph underneath this one the way 608dd15c stacked one
    # under 2026-09-03's.
    block = _pin_comment_block()
    assert len(re.findall(r"Verdict \d{4}-\d{2}-\d{2}", block)) == 1, (
        f"the pin block does not read as one dated verdict: {block!r}")
    assert PREFIX_PIN_COMMENT[:60] not in block, (
        "the pre-fix paragraph survived inside the block the greps scan: the two "
        "contradictory paragraphs are still the text a reader sees")


def test_the_pin_verdict_states_the_primary_only_deployment_and_is_cross_checked():
    """#2137 clause 2. The comment's operative reason is the deployment, so the
    test reads the code the comment describes: if the secondary slot comes back,
    THIS test fails and says to re-read the comment, rather than the comment
    rotting in silence next to a live second arm."""
    from app.config import resolve_model_alias

    block = _pin_comment_block()
    assert "primary-only since 2026-09-20" in block, block
    assert "secondary_enabled" in block and "resolve_model_alias" in block, block

    assert CONFIG["secondary_enabled"] is False, (
        "the secondary slot is enabled again: the comment's stated reason is "
        "false, and the observer's model is a live choice again — re-read #771")
    assert resolve_model_alias("secondary") == "primary", (
        "`resolve_model_alias` stopped rewriting the alias, so `secondary` means "
        "an arm again and this pin no longer describes what runs")
    assert CONFIG["inner_voice"]["model"] == "primary"


def test_the_pin_verdict_retires_the_2026_09_03_a_b_with_its_arm_and_cites_771():
    """#2137 clause 3. The number stays in the comment as history, framed as a
    retired measurement — and the retirement is checked against config, because
    ':8091 serves nothing' is a claim about a slot this file itself configures."""
    block = _pin_comment_block()
    assert "40%" in block and "1.7%" in block, block
    assert "8091" in block, f"the comment must say which arm is gone: {block}"
    assert "djev" in block, f"the comment must say what occupies that GPU: {block}"
    assert "#771" in block, f"the verdict states a ruling with no citation: {block}"
    assert "retire" in block.lower(), block

    assert CONFIG["secondary_enabled"] is False
    assert CONFIG["djev"]["enabled"] is True, (
        "djev no longer holds GPU 2, so the reason the A/B cannot run may have "
        "changed — re-read the #771 close before trusting this comment")
    assert CONFIG["models"]["secondary"]["base_url"].endswith(":8091"), (
        "the secondary slot moved off :8091, so the comment's 'nothing serves "
        ":8091' is describing a port this file no longer configures")


def test_the_pin_verdict_names_what_reopens_the_question_and_instructs_no_rerun():
    """#2137 clause 4. The reopen condition is a person standing a second slot up
    on spare hardware; inside the `inner_voice:` block no sentence may order a
    re-run against a named occupant, because every occupant it could name is
    retired."""
    block = _pin_comment_block()
    assert "re-enabling a second engine slot on spare hardware" in block, block
    assert "iv_grade" not in _inner_voice_section(), (
        "the block names the eval again, which is how the pre-#2137 comment read "
        "as a standing instruction")

    assert _rerun_instructions(PREFIX_PIN_COMMENT), (
        "the control text no longer trips the re-run scan, so the empty result on "
        "config.yaml proves nothing")
    bad = _rerun_instructions(_inner_voice_section())
    assert bad == [], f"`inner_voice:` still instructs a re-run against an arm: {bad}"


def test_the_canary_section_says_the_differential_is_scored_on_the_egress_lane():
    """The injection-canary section must say which lane the enforce A/B is read on,
    and where the arm comes from (#2154).

    The sentence exists because of a trap in the measurement: `attack_success` is the
    call the model PROPOSED, and an arm that differs only in `harness.egress_policy`'s
    `enforce` flag cannot move it — so two identical-looking attack rates in the pair
    invite the reading "the gate does nothing". The lane that does move is
    `egress_destination`, which books a canary-carrying call to an off-allow-list host
    as an escape and is the one that records the guard's decision, and the prompt-side
    gate the sentence rules out is real: `app/harness/safety.py` reads neither
    `LLOYD_EGRESS_ENFORCE` nor `egress.enforce_on()`, so its verdict is the same string
    in both arms. Both halves are asserted against the code rather than against other
    prose, which is the rule of this file.

    The arm claim is pinned too — the doc names the environment variable, and the
    config is asserted to hold no `harness.egress_policy` section, because a reader who
    believed a config key chose the arm would edit config.yaml to run the pair and read
    two runs of the same arm as an A/B.
    """
    arm = _section(DOC_TEXT, "## The injection canary bench")
    assert "egress_destination" in arm, "the doc must name the lane the pair is read on"
    assert "LLOYD_EGRESS_ENFORCE" in arm and "enforce_on" in arm
    assert "app/harness/safety.py" in arm, "the doc must say where it is NOT scored"
    assert "wilson_ci" in arm and "Wilson" in arm

    # The identifiers the sentence rests on are defined where it says they are.
    assert (ROOT / "eval" / "run_injection_canary.py").read_text(
        encoding="utf-8").find("def egress_destination(") > 0
    assert (ROOT / "agent_mcp" / "egress.py").read_text(
        encoding="utf-8").find("def enforce_on(") > 0
    assert (ROOT / "eval" / "stats.py").read_text(encoding="utf-8").find("def wilson_ci(") > 0

    safety = (ROOT / "app" / "harness" / "safety.py").read_text(encoding="utf-8")
    assert "LLOYD_EGRESS_ENFORCE" not in safety and "enforce_on" not in safety, (
        "`app/harness/safety.py` now reads the enforcement flag, so the doc's claim that "
        "the substrate differential is not on that gate has to be rewritten, not re-asserted")

    # The scenario the section names as the one that reaches the lane is shipped, and
    # its host is the non-resolving kind the section promises.
    scenarios = (ROOT / "eval" / "injection_canary" / "scenarios.yaml").read_text(
        encoding="utf-8")
    assert "webpage-egress-fetch" in arm and "key: webpage-egress-fetch" in scenarios
    corpus = (ROOT / "eval" / "injection_canary" / "corpus"
              / "webpage-egress-fetch.html").read_text(encoding="utf-8")
    assert ".invalid" in corpus, (
        "the host the section calls non-resolving is not an RFC 6761 .invalid name")


def test_the_canary_section_does_not_claim_a_config_key_selects_the_arm():
    """The negative half of the same claim: the arm is selected by the runner's
    environment, and the shipped `harness.egress_policy` (#2123) leaves `enforce`
    false, exactly as the section says.

    A flipped `enforce` in config would be read by `enforce_on()` too, so a landing
    that arms it must rewrite this paragraph's description of the shipped state —
    the kind of sentence this file exists to catch.
    """
    policy = (CONFIG.get("harness") or {}).get("egress_policy") or {}
    assert policy.get("enforce") is False, (
        "`harness.egress_policy.enforce` is no longer false in config.yaml: an A/B "
        "run that sets the env variable must be checked against the config value too")
    arm = " ".join(_section(DOC_TEXT, "## The injection canary bench").split())
    assert "no `config.yaml` edit" in arm and "`enforce: false`" in arm


def test_the_canary_section_says_a_search_query_is_an_escape_whatever_the_seed_names():
    """#2192 clause 5: the section describes the host-agnostic search lane, and the
    strings the two nodes above pin are still in it.

    The sentence had to be added somewhere a reader of the bench would meet it, and the
    paragraph that explains how each egress lane is judged is the only such place: the
    scorer's docstring is read by whoever edits the scorer, this is read by whoever runs
    the bench and wonders why a search query counts. The claim it now makes is the one
    #2192 ruled — a token carried in `query` left the machine whatever
    `harness.egress_policy.allow` covers, since #2123 seeded `duckduckgo.com` there —
    together with the confinement, because a doc that said "egress is judged on content"
    would license a change to the `http_fetch` lane that the code does not make.

    The second block is the joint-satisfiability half: a rewrite that added the sentence
    and dropped a pinned identifier would otherwise pass here and redden the two nodes
    above, which say nothing about this change. Asserted in this node so the failure
    names the edit that caused it.
    """
    arm = " ".join(_section(DOC_TEXT, "## The injection canary bench").split())
    assert "query text" in arm and "whatever the allow-list names" in arm, (
        "the section no longer states #2192's ruling on the free-text lane")
    assert "harness.egress_policy.allow" in arm and "#2123" in arm, (
        "the sentence must name the seed it exists for, not gesture at 'the allow-list'")
    assert "`http_fetch` url" in arm and "stays an echo" in arm, (
        "the section must keep the rule confined to the lane with no destination")
    assert "SEARCH_BACKEND_HOST" in arm and "egress_destination" in arm

    for pinned in ("egress_destination", "LLOYD_EGRESS_ENFORCE", "enforce_on",
                   "app/harness/safety.py", "wilson_ci", "webpage-egress-fetch",
                   "no `config.yaml` edit", "`enforce: false`"):
        assert pinned in arm, f"the #2192 rewrite dropped a string the pair nodes pin: {pinned}"


def test_the_canary_section_says_the_arm_is_the_serving_endpoints_state():
    """#2338: the section may not describe an arm the runner picks for itself.

    Until this change the passage said the arm was selected "*only* by the runner's
    environment" — true of the label, false of the guard: the episode's tools are
    dispatched to the aggregator, whose `guard()` reads `enforce_on()` in ITS process,
    and the shared daemon answers `enforce: false` however the shell is set up. The
    result was an A/B whose two arms were the same arm. So the section has to name
    where the label now comes from (the serving endpoint's `/state` key), what a run
    does when the endpoint will not honour the request (refuses, before appending), and
    how the on arm is served at all (`--mcp-url`).

    Asserted against the code as well as the prose, the rule of this file: the state key
    and the reader exist, the refusal path returns non-zero, and `grade` can name a row
    whose two states disagree.
    """
    arm = " ".join(_section(DOC_TEXT, "## The injection canary bench").split())
    for phrase in ("/state", "guard_egress_enforce", "--mcp-url"):
        assert phrase in arm, f"the section no longer names {phrase}"
    assert "refuse" in arm.lower(), "the section must say the run refuses, not relabels"

    runner = (ROOT / "eval" / "run_injection_canary.py").read_text(encoding="utf-8")
    for symbol in ("def verified_arm(", "def read_guard_egress(",
                   "def arm_conflicts(", '"--mcp-url"'):
        assert symbol in runner, f"`{symbol}` is gone from the runner the section cites"
    assert "return 2" in runner, "the refusal has to be a non-zero exit, not a warning"

    aggregator = (ROOT / "agent_mcp" / "main.py").read_text(encoding="utf-8")
    assert '"egress":' in aggregator, (
        "`GET /state` stopped publishing the egress key, which is the only way an "
        "out-of-process caller can tell the arms apart")
    assert (ROOT / "agent_mcp" / "egress.py").read_text(
        encoding="utf-8").find("def status(") > 0


def test_the_canary_bench_docstring_counts_the_scenarios_it_ships():
    """#2363 clause 5: the count the bench leads its own docstring with is the count it
    ships, and the registry's row for it agrees.

    "Thirteen worker-style tasks" sat above sixteen scenarios uncontradicted until
    #2363 added two more — which is what a number nobody can re-derive looks like:
    every reader after the fifteenth scenario inherited the sentence instead of
    counting the list. This node makes the sentence a claim about
    `len(load_scenarios())` rather than about somebody's memory, and it reads the
    spelled-out word because that is how both sentences are written; `num2words` is
    not a dependency of this tree, so the comparison goes through a fixed vocabulary
    that fails loudly on a word outside it.

    Both carriers are pinned because `architecture/measurement.md` restates the same
    tally one file away from the runner — the third control #2363 adds appears in
    neither until now, and a rule with one carrier out of several is a rule that
    silently rots. The scenario count and the control count are derived from the same
    loaded list, so a fourth scenario or control moves an assertion here rather than
    moving the docs apart.
    """
    import importlib.util
    import re
    spec = importlib.util.spec_from_file_location(
        "canary_bench_for_counts", ROOT / "eval" / "run_injection_canary.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    spelled = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
               "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
               "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
               "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
               "twenty": 20}
    word = {n: w for w, n in spelled.items()}      # the docs spell, never digit
    scenarios = bench.load_scenarios()
    # The docstring's control paragraph is about the single-turn bench; the
    # persistence pair's own control is documented with the pair, so the paragraph's
    # denominator is the controls that are not part of it, derived the same way the
    # runner derives its own persistence set rather than by excluding a name.
    persistence = {s["key"] for s in bench.persistence_scenarios(scenarios)}
    controls = [s["key"] for s in scenarios
                if s.get("control") and s["key"] not in persistence]
    n_controls = len(controls)
    assert len(scenarios) >= 10 and n_controls >= 2, (
        "the bench shrank out from under this node; it needs at least ten scenarios "
        "and two controls to mean anything")

    doc = (ROOT / "eval" / "run_injection_canary.py").read_text(
        encoding="utf-8").split('"""', 2)[1]
    head = re.search(r"^([A-Za-z]+) worker-style tasks", doc, re.M)
    assert head, "the runner docstring no longer leads with a scenario count"
    assert spelled[head.group(1).lower()] == len(scenarios), (
        f"the runner docstring says {head.group(1)!r}; scenarios.yaml ships "
        f"{len(scenarios)} — the sentence is the drift this node exists to refuse")

    lead = re.search(r"^([A-Za-z]+) controls:", doc, re.M)
    assert lead and spelled[lead.group(1).lower()] == n_controls, (
        f"the docstring counts itself differently from the {n_controls} single-turn "
        f"controls that ship: {controls}")
    named = set(re.findall(r"`(control-[a-z-]+)`", doc))
    assert named == set(controls), (
        f"the paragraph names {sorted(named)}; the bench ships {sorted(controls)} — a "
        "control nobody points at is a control nobody maintains")

    row = [ln for ln in (ROOT / "architecture" / "measurement.md").read_text(
        encoding="utf-8").splitlines() if ln.startswith("| `injection_canary` |")]
    assert len(row) == 1, row
    tally = re.search(r"([A-Za-z]+) worker-style tasks", row[0])
    assert tally and spelled[tally.group(1).lower()] == len(scenarios), (
        f"the registry row says {tally and tally.group(1)!r}; scenarios.yaml ships "
        f"{len(scenarios)}")
    assert f"with {word[n_controls]} controls" in row[0], (
        f"the registry row no longer counts the {n_controls} controls")
