"""#3906 clause 4 read from Python: the `by_kind` split renders as context, never as a
second alarm.

The contract lives in `web/src/components/pages/SourceHealthRow.tsx` and its executing pin is
`web/src/components/pages/BackgroundPage.test.ts`, which renders the leaf and reads its HTML.
That file cannot be what the gate's `review` rung cites: the rung resolves a `test_node_id`
only inside a test file under `pytest.ini`'s testpaths, and `scripts/automod/testpaths.py
::is_test_file` requires a `.py`. A clause pinned only in a `.test.ts` is therefore refused as
`test_node_id not in a test file this diff changed` however well it was run — which is exactly
the downgrade round `SM_20261107_205309` took on this clause, with that round's own grader text
saying its substance was met. This file is the in-repo remedy, the one
`tests/test_dashboard_label_contract.py` records from #2202 and
`tests/test_background_runs_panel_contract.py` from #2648: the same contract, as pytest nodes,
over the same bytes.

Being a translation of a test that already exists is worth nothing on its own, so these nodes
do the three things the vitest file cannot:

  * They read the RENDER SITE and its guard out of the JSX — `{kindSplit !== '' && <span …>}` —
    identifier-bound and comment-blind. Rendered HTML cannot show a dropped guard at all (an
    empty string renders nothing either way) and cannot show a second render site unless the
    fixture happens to be non-empty at that moment.
  * They cross the seam the vitest file stubs. The field names the formatter reads are compared
    with the keys `workers/queue.py` builds into each `by_kind` entry, and the component is
    checked NOT to name a kind, so the row renders whatever partition arrives rather than the
    two kinds one fixture happens to carry. A literal seeded into a test cannot see the emitter
    rename a key.
  * They hold the two pins together. The vitest file's expected strings are PARSED out of that
    file and re-divided here against that file's own fixture, and every static fragment of the
    component's template literal has to appear in them — so a joiner moving from ` · ` to ` | `
    reddens this file even though both files stay green on their own terms.

And, like the two precedents, the checks are themselves checked:
`test_the_mirror_bites_on_each_of_five_mutations` breaks the component five ways in memory and
requires a named check to go red for each.

Nothing here renders a pixel and nothing here needs `web/node_modules`: run it with
`pytest tests/test_source_health_row_kind_split.py`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TSX = ROOT / "web/src/components/pages/SourceHealthRow.tsx"
VITEST = ROOT / "web/src/components/pages/BackgroundPage.test.ts"
QUEUE = ROOT / "workers/queue.py"

#: The muted ink the split is printed in, and the alarm colour it must never wear.
MUTED = "text-zinc-500"
ROSE = "text-rose-400"


def _strip_comments(text: str) -> str:
    """`text` with every block and line comment blanked out, line count preserved.

    Every structural check below reads code, and this component's comments are the substance
    of the clause (`never in the alarm colour`, `Absent when the payload has no by_kind`).
    Leaving them in would let a comment satisfy a `not in` and a `count == 1`.
    """
    text = re.sub(r"/\*[\s\S]*?\*/", lambda m: "\n" * m.group(0).count("\n"), text)
    return re.sub(r"(?m)//[^\n]*$", "", text)


def _component(text: str) -> str:
    """The `SourceHealthRow` function, from its signature to the end of the file."""
    return text[text.index("export function SourceHealthRow"):]


def _formatter(text: str) -> str:
    """The `kindSplitText` function body."""
    start = text.index("export function kindSplitText")
    return text[start:text.index("export function rateIsAlarming", start)]


def _alarm(text: str) -> str:
    """The `rateIsAlarming` function."""
    m = re.search(r"export function rateIsAlarming[\s\S]*?\n}", text)
    assert m, "no rateIsAlarming in the file the clause is about"
    return m.group(0)


def _split_locals(text: str) -> list[str]:
    """Identifiers in the component bound to `kindSplitText(h)`."""
    return re.findall(r"\bconst\s+(\w+)\s*=\s*kindSplitText\(\s*h\s*\)", _component(text))


def _split_render_sites(text: str) -> list[tuple[str, str]]:
    """Each `(identifier, span-attributes)` guarded JSX render of a split local.

    The pattern is identifier-bound on purpose: `{kindSplit !== '' && <span>{something}</span>}`
    printing a different expression than the one it tested would render a stale copy, so the
    backreference has to see the SAME name inside the span.
    """
    comp = _component(text)
    return [(m.group(1), m.group(2)) for m in re.finditer(
        r"\{\s*(\w+)\s*!==\s*''\s*&&\s*<span([^>]*?)>\{\1\}</span>\s*\}", comp)]


def _check_render_discipline(text: str) -> None:
    """Clause 4's markup half: rendered when present, absent when not, muted ink only."""
    code = _strip_comments(text)
    locals_ = _split_locals(code)
    assert len(locals_) == 1, (
        f"{locals_}: the row is meant to render the formatter's output through exactly one "
        "local, so the guard and the printed value can be the same value. More than one means "
        "the page formats the block twice; none means it does not call the formatter at all.")
    name = locals_[0]

    sites = _split_render_sites(code)
    assert len(sites) == 1, (
        f"found {len(sites)} guarded renders of the split. Rendering it unguarded would print "
        f"`<span></span>` for every payload with nothing to split — a no-key backend older "
        f"than `64ff6c01`, an empty block, or a source whose one kind IS the whole row — and "
        "that is most of the panel. Rendering it twice prints the same halves twice.")
    guarded, attrs = sites[0]
    assert guarded == name, (
        f"the guard tests `{guarded}` but the span prints `{name}`: the emptiness that hides "
        "the markup is not the emptiness of what the markup prints.")
    assert MUTED in attrs, (
        f"the split's span carries {attrs!r}, not the muted ink. A kind's figures are context "
        "beside the blended rate, and the row has exactly one colour for a decision.")
    assert ROSE not in attrs and "alarming" not in attrs, (
        f"the split's span is coloured by a rate ({attrs!r}). Clause 4's alarm is the "
        "SOURCE-level one: a kind can be 1 failed of 1 run and the source is not on fire.")

    # Nothing else in the row may print a kind: the guarded span is the only exit for the
    # block, so "renders no per-kind markup when absent" is decided in one place, not five.
    rest = re.sub(r"\bconst\s+\w+\s*=\s*kindSplitText\(\s*h\s*\)", "", _component(code))
    only_span = re.sub(
        r"\{\s*\w+\s*!==\s*''\s*&&\s*<span[^>]*>\{\w+\}</span>\s*\}", "", rest)
    assert not re.search(r"\bkindSplit\w*\b|\bby_kind\b", only_span), (
        "the split, or the raw `by_kind` block, is printed somewhere outside the guarded span "
        "— a second render site is markup the absence clause does not cover.")


def _check_alarm_reads_only_the_source_rate(text: str) -> None:
    """Clause 4's decision half: the alarm sees the source's own rate and no kind's."""
    code = _strip_comments(text)
    body = _alarm(code)
    assert "h.fail_rate" in body, (
        "rateIsAlarming no longer reads the source's blended `fail_rate`, which is the only "
        "figure #3526's doc says may gate the row")
    assert "ALARM_FAIL_RATE" in body, (
        "the threshold is inlined, so the constant the vitest node names is no longer the "
        "number that decides the colour")
    lowered = body.lower()
    for forbidden in ("by_kind", "kind", "kindsplit"):
        assert forbidden not in lowered, (
            f"the alarm decision now reaches for `{forbidden}`. One kind at 100 % of one run "
            "is not a source on fire, and blending is the whole reason the source rate exists.")
    receivers = set(re.findall(r"(\w+)\.fail_rate", body))
    assert receivers <= {"h"}, (
        f"`fail_rate` is read off {sorted(receivers)} in the alarm; only the health block's own "
        "field may decide it.")


def _emitter_keys() -> dict[str, str]:
    """The keys `workers/queue.py` builds into one `by_kind` entry, and `NO_KIND`'s spelling."""
    from workers.queue import NO_KIND

    src = QUEUE.read_text(encoding="utf-8")
    m = re.search(r"by_kind\.setdefault\(src, \{\}\)\[key\]\s*=\s*\{([\s\S]*?)\n\s*\}", src)
    assert m, ("the per-kind entry is no longer built as a dict literal in workers/queue.py, "
               "so the field names the row formats have no producer to check against")
    return dict.fromkeys(re.findall(r'"(\w+)":', m.group(1))), NO_KIND


def _check_fields_match_the_emitter(text: str) -> None:
    """The formatter reads exactly what the payload carries, and invents none of it."""
    code = _strip_comments(text)
    body = _formatter(code)
    built, no_kind = _emitter_keys()
    read = set(re.findall(r"\bv\.(\w+)", body))
    assert read == set(built), (
        f"the row formats {sorted(read)} from each kind while the queue publishes "
        f"{sorted(built)} — the two sides of `/api/workers/health` disagree, and the panel "
        "would print `undefined` for a renamed key while every test that seeds a literal "
        "stays green.")
    assert "h.by_kind" in body, (
        "the formatter no longer reads the block off the payload, so the split is coming from "
        "somewhere else")
    for kind_name in ("paper", "video", no_kind):
        assert kind_name not in body, (
            f"the formatter names `{kind_name}` in code. The block is a map keyed by kind: "
            "hardcoding a name fixes the row to the sources that exist today and prints "
            "nothing for the ones that arrive tomorrow.")
    assert re.search(r"\.sort\(", body), (
        "the kinds are no longer ordered by name, so two reads of one window can print the "
        "same halves in two orders — the vitest node's byte-equality expects one.")


def _template_fragments(text: str) -> tuple[list[str], str, list[str]]:
    """`(static segments, joiner, rate-template statics)` of the split's own formatting."""
    body = _formatter(_strip_comments(text))
    tmpl = re.search(r"return\s+`([^`]*)`", body)
    assert tmpl, "the split is no longer built from one template literal"
    statics = [s for s in re.split(r"\$\{[^}]*\}", tmpl.group(1)) if s]
    join = re.search(r"\.join\('([^']*)'\)", body)
    assert join, "the per-kind clauses are no longer joined by a literal"
    rate_tmpl = re.search(r"const rate =[\s\S]*?`([^`]*)`", body)
    assert rate_tmpl, "the per-kind rate is no longer built from a template literal"
    return statics, join.group(1), [s for s in re.split(r"\$\{[^}]*\}", rate_tmpl.group(1)) if s]


def _clause_pattern(text: str) -> tuple[re.Pattern, list[str]]:
    """A clause matcher BUILT from the component's own template, plus each group's role.

    Hardcoding `^(\\w+): (\\d+) runs?…` here would only ever prove that the component still
    matches a sentence I typed into Python: the day its wording changes this file goes red for
    the wrong reason and green for the right one. So the pattern is the template's statics,
    escaped, with a capture group standing in for each substitution, and the rate branch comes
    from the rate template's own suffix and its `n/a` word.

    The one deliberate looseness: the singular/plural ternary (`run${v.total === 1 ? '' : 's'}`)
    becomes the permissive group `(s?)`, because deciding which side it takes means deciding
    what `Math.round` and `=== 1` do, and transcribing those into Python is a second
    implementation to be wrong about. The exact bytes are the vitest file's job; this node's
    job is that the two files agree about the SHAPE and about every number inside it.
    """
    body = _formatter(_strip_comments(text))
    tmpl = re.search(r"return\s+`([^`]*)`", body)
    assert tmpl, "the split is no longer built from one template literal"
    rate_line = re.search(r"const rate =([^\n]*)", body)
    rate_tmpl = re.search(r"`([^`]*)`", rate_line.group(1))
    null_word = re.search(r"\?\s*'([^']*)'", rate_line.group(1))
    assert rate_tmpl and null_word, "the per-kind rate is no longer a template with an n/a arm"
    # The suffix is what follows the substitution — `0% failing` is `${…}` + '% failing'.
    rate_suffix = re.split(r"\$\{[^}]*\}", rate_tmpl.group(1))[-1]

    roles: list[str] = []
    parts: list[str] = []
    pieces = re.split(r"(\$\{[^}]*\})", tmpl.group(1))
    for i, piece in enumerate(pieces):
        if i % 2 == 0:
            parts.append(re.escape(piece))
            continue
        expr = piece[2:-1].strip()
        ternary = re.match(r"[\s\S]*\?\s*'([^']*)'\s*:\s*'([^']*)'$", expr)
        if ternary:
            parts.append(f"({ternary.group(1)}|{ternary.group(2)})")
            roles.append("plural")
        elif "rate" in expr:
            parts.append(f"(\\d+{re.escape(rate_suffix)}|{re.escape(null_word.group(1))})")
            roles.append("rate")
        elif "total" in expr:
            parts.append(r"(\d+)")
            roles.append("total")
        elif "failed" in expr:
            parts.append(r"(\d+)")
            roles.append("failed")
        elif expr == "kind":
            parts.append(r"([^\s:]+)")
            roles.append("kind")
        else:
            raise AssertionError(f"unmapped substitution {expr!r} in the split's template")
    return re.compile("^" + "".join(parts) + "$"), roles


def _vitest_fixture_and_expectation() -> tuple[list[tuple[str, int, int, float]], str]:
    """`TWO_KINDS`' kinds and the expected one-line split, both parsed out of the vitest file.

    Parsed and never retyped: the whole value of this node is that the two pins cannot drift
    apart, and a table copied into Python would drift with them.
    """
    vit = VITEST.read_text(encoding="utf-8")
    block = vit[vit.index("const TWO_KINDS: SourceHealth = {"):].split("\n  }", 1)[0]
    kinds = [(m.group(1), int(m.group(2)), int(m.group(3)),
              0.0 if m.group(4) == "null" else float(m.group(4)))
             for m in re.finditer(
                 r"(\w+):\s*\{\s*total:\s*(\d+),\s*failed:\s*(\d+),"
                 r"\s*fail_rate:\s*([0-9.]+|null)\s*\}", block)]
    assert len(kinds) >= 2, (
        f"the vitest fixture carries {len(kinds)} kinds, so the file's own two-kind node is "
        "no longer a two-kind node and there is nothing here to compare against")
    want = re.search(r"kindSplitText\(TWO_KINDS\)\)\.toBe\(\s*'([^']+)'", vit)
    assert want, (
        "the vitest file no longer states the expected split as one literal, so the format "
        "contract is asserted in neither file")
    return kinds, want.group(1)


def _check_format_agrees_with_the_vitest_pin(text: str) -> None:
    """One contract, two files: the component's literals and the test's expectation agree."""
    statics, joiner, rate_statics = _template_fragments(text)
    kinds, expected = _vitest_fixture_and_expectation()

    for frag in statics + rate_statics:
        assert frag in expected, (
            f"the component's template carries {frag!r} and the vitest expectation "
            f"{expected!r} does not — one of the two files changed the wording of the split "
            "and the other did not, so each is green on its own and the panel prints a "
            "sentence neither of them pins.")
    clauses = expected.split(joiner)
    assert len(clauses) == len(kinds), (
        f"{expected!r} splits into {len(clauses)} clauses on {joiner!r} but the fixture has "
        f"{len(kinds)} kinds, so a kind is printed twice or not at all")

    pattern, roles = _clause_pattern(text)
    ordered = sorted(kinds, key=lambda k: k[0])
    for clause, (kind, total, failed, rate) in zip(clauses, ordered):
        m = pattern.match(clause)
        assert m, (
            f"{clause!r} is not the shape the component's own template builds "
            f"({pattern.pattern!r}), so the vitest expectation is describing a format the "
            "component no longer has")
        got = dict(zip(roles, m.groups()))
        assert got["kind"] == kind, (
            f"the clauses come out as {[c.split(':')[0] for c in clauses]} while the fixture "
            f"declares {[k[0] for k in kinds]}: the split is not printed in name order, so "
            "two reads of one window print two different lines")
        assert (int(got["total"]), int(got["failed"])) == (total, failed), (
            f"{clause!r} quotes figures the fixture never set ({kind} is {total} runs / "
            f"{failed} failed), so the expectation and the seed have split")
        shown = got["rate"]
        if not shown.endswith("n/a"):
            assert shown.split("%")[0] == str(round(rate * 100)), (
                f"{clause!r} prints {shown!r} for a kind whose fail_rate is {rate}; the "
                "component rounds that rate to a whole percent and the expectation stopped "
                "dividing the same way")
    assert [k[0] for k in kinds] != [k[0] for k in ordered], (
        "the fixture now declares its kinds in name order, so the name-order assert above is "
        "ungraded — keep the declaration out of order or that line can never fail")


def clauses_and_kinds(clauses, kinds):  # pragma: no cover - helper kept out of collection
    raise AssertionError("unused")


def test_the_row_renders_the_split_behind_its_own_guard_and_in_muted_ink():
    _check_render_discipline(TSX.read_text(encoding="utf-8"))


def test_the_alarm_decision_never_reads_a_kinds_rate():
    _check_alarm_reads_only_the_source_rate(TSX.read_text(encoding="utf-8"))


def test_the_kinds_the_row_formats_are_the_ones_the_queue_publishes():
    _check_fields_match_the_emitter(TSX.read_text(encoding="utf-8"))


def test_the_split_the_component_builds_is_the_split_the_vitest_file_expects():
    _check_format_agrees_with_the_vitest_pin(TSX.read_text(encoding="utf-8"))


#: Five ways the component could be wrong that look right, each with the check that must
#: notice. Written as text edits, not as a described risk: the precedent in
#: `test_background_runs_panel_contract.py` is that a mirror nobody can break is decoration.
MUTATIONS: list[tuple[str, str, str, object]] = [
    ("the guard is dropped, so an empty split renders an empty span",
     "{kindSplit !== '' && <span className=\"text-zinc-500\">{kindSplit}</span>}",
     "<span className=\"text-zinc-500\">{kindSplit}</span>",
     _check_render_discipline),
    ("the split is printed in the alarm colour",
     "{kindSplit !== '' && <span className=\"text-zinc-500\">{kindSplit}</span>}",
     "{kindSplit !== '' && <span className=\"text-rose-400\">{kindSplit}</span>}",
     _check_render_discipline),
    ("a kind's rate is OR-ed into the alarm",
     "return h.fail_rate !== null && h.fail_rate >= ALARM_FAIL_RATE",
     "return (h.by_kind?.paper?.fail_rate ?? 0) > 0.5 || h.fail_rate >= ALARM_FAIL_RATE",
     _check_alarm_reads_only_the_source_rate),
    ("the joiner between kinds changes",
     ".join(' · ')", ".join(' | ')",
     _check_format_agrees_with_the_vitest_pin),
    ("the formatter names a kind instead of the key it was given",
     "return `${kind}: ${v.total} run",
     "return `paper: ${v.total} run",
     _check_fields_match_the_emitter),
]


def test_the_mirror_bites_on_each_of_five_mutations():
    text = TSX.read_text(encoding="utf-8")
    for label, old, new, check in MUTATIONS:
        assert old in text, f"{label}: the component no longer contains {old!r}"
        mutated = text.replace(old, new, 1)
        assert mutated != text, f"{label}: the mutation was a no-op"
        with pytest.raises(AssertionError) as caught:
            check(mutated)
        assert str(caught.value), f"{label}: the check failed with no message"
