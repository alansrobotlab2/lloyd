"""`architecture/knowledge-graph.md`'s apply-rule statements stay true.

That doc is the incident-derived rulebook for the edge store, and §Rules that came
out of the incidents is the half an operator reads *before* deciding whether a
nightly write into `kg.sqlite` needs watching. Until #2641 its rule 2 carried the
blanket form — pinned here as `DEAD_RULE_SENTENCE`, and asserted absent rather than
quoted into prose — and the write-path diagram labelled #48's daily run
dry-run-only (`DEAD_DIAGRAM[0]`) and ended with a human applying the plan
(`DEAD_DIAGRAM[1]`).

Neither has been true since Alan's ruling on #990 (2026-09-16). Step 3 of the job's
own skill prompt ends every scheduled run with `--apply --tiers CASE,PUNCT`; apply
reports land in `_pipeline/memory-graph/` daily, with the last one on 2026-10-12;
and the run records leave the gate-approved SUFFIX clusters for a person —
`run_48_20261012_145936.md` ends "22 gate-approved suffix clusters left for you
(not applied — a suffix is a meaning claim)". A reader who obeyed this doc was
therefore *stopping* the nightly write the job is authorised to make, which is the
failure a rulebook causes by being stale rather than by being wrong.

`architecture/autonomy.md:1090` and the paragraph under it already carried the
narrowed rule, so the wording existed and this file was simply never brought along.
The narrowing, not a repeal, is the part worth pinning: the 2026-08-22 wipe and the
2026-09-03 151-merge incident were both unattended applies of exactly the shape #48
now performs, so what makes the two mechanical tiers survivable is the shape of the
apply — degraded-graph refusal, high-value exclusion, one store transaction, a
resumable apply report — and not the absence of risk. Nothing pinned any of this
before: `grep -rn "Rules that came out of the incidents" tests/*.py` returned the
doc's own heading and nothing else.

Checks are section-scoped through `workers.sources.arch_review.find_section`, the
same matcher the arch-review unit uses, because #2641's two stale claims sat in two
different sections of one file: a whole-file "somewhere it says the right thing"
search would let the corrected §Review log excuse an uncorrected rule 2, or rule 2
excuse a diagram that still says the run is a dry run.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "knowledge-graph.md"

#: The section an operator reads as the rulebook, and the section that records what
#: each pass corrected — both graded separately, on purpose.
RULES_SECTION = "Rules that came out of the incidents"
REVIEW_SECTION = "Review log"

#: Rule 2's blanket form as it stood until #2641. `DEAD_RULE_TITLE` is the sentence
#: heading; the CHECK grep reads it too, because the heading is the sentence an
#: operator skims.
DEAD_RULE_SENTENCE = "#48 and #67 propose; a human runs `--apply`"
DEAD_RULE_TITLE = "Applies are attended"

#: The diagram's two stale claims, in the order the block carried them: the label on
#: #48's line, and the arrow's destination.
DEAD_DIAGRAM = ("DRY RUN ONLY", "a human reads the plan and runs --apply")

#: What #48 actually runs on schedule, spelled the way the command is spelled.
APPLY_CMD = "`--apply --tiers CASE,PUNCT`"


def _text() -> str:
    return DOC.read_text()


def _section(text: str, name: str) -> str:
    """One `## ` section's own lines, found through the arch-review matcher.

    Section-scoped rather than whole-file: the defect was one file disagreeing with
    itself, and a whole-file search lets one section's correct wording excuse
    another's false claim.
    """
    from workers.sources import arch_review as A

    span = A.find_section(text, name)
    assert span, f"architecture/knowledge-graph.md lost its `## {name}` heading"
    return "\n".join(text.splitlines()[span[0] - 1:span[1]])


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _apply_sentence(section: str) -> str | None:
    """The one sentence saying #48 *applies* the mechanical tiers *unattended*.

    Both halves are required: "merges CASE/PUNCT" without "unattended" describes an
    operator act, and "runs nightly" without the tiers describes a job that writes
    whatever the judge likes.
    """
    for s in _sentences(section):
        if (re.search(r"#48\b", s)
                and re.search(r"\bappl(?:y|ies|ied)\b", s, re.I)
                and "CASE" in s
                and re.search(r"\bunattended\b|\bon schedule\b|\bnightly\b", s, re.I)):
            return s
    return None


def _sweep_node(text: str) -> str:
    """The write-path diagram's #48 node: its line and the lines under it."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines)
                  if "entity-resolution-sweep.py" in l and "#48" in l), None)
    assert start is not None, "the write-path diagram lost its #48 line entirely"
    for k, l in enumerate(lines[start + 1:]):
        if l.startswith("```"):
            return "\n".join(lines[start:start + 1 + k])
    raise AssertionError("the write-path diagram's #48 node never reaches its fence")


def _review_entries(review: str) -> list[str]:
    """§Review log as bullets, so one entry cannot borrow another's date or pin."""
    entries: list[str] = []
    cur: str | None = None
    for line in review.splitlines():
        if line.startswith("- "):
            if cur is not None:
                entries.append(cur)
            cur = line
        elif cur is not None:
            cur += "\n" + line
    if cur is not None:
        entries.append(cur)
    return entries


def _rule_block(text: str, marker: str) -> str:
    """One numbered rule's whole block, from its `N. **` line to the next."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith(f"{marker} **")), None)
    assert start is not None, f"rule {marker} is gone from the document"
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"\d+\. \*\*", lines[i])),
               len(lines))
    return "\n".join(lines[start:end])


def test_rule_2_says_48_applies_the_mechanical_tiers_unattended():
    """Clause 1. §Rules that came out of the incidents carried the blanket form on
    both sides of the sentence: the heading read `DEAD_RULE_TITLE`, and the body
    `DEAD_RULE_SENTENCE`. #48 has ended its scheduled run with `--apply
    --tiers CASE,PUNCT` since 2026-09-16, so the rule read as a prohibition on the
    job's own authorised behaviour.
    """
    text = _text()
    for dead in (DEAD_RULE_SENTENCE, DEAD_RULE_TITLE):
        assert dead not in text, (
            f"the document still carries {dead!r}. It is not a nuance but the "
            f"sentence a person reads before deciding whether to stop a nightly "
            f"apply against the edge store")
    sentence = _apply_sentence(_section(text, RULES_SECTION))
    assert sentence is not None, (
        "§Rules that came out of the incidents never says, in one sentence, that "
        f"#48 *applies* {APPLY_CMD} *unattended*. What runs on schedule is that "
        "command with nobody watching, and a rule that omits it forbids the "
        "behaviour the rulebook is supposed to govern")
    assert re.search(r"--apply\s+--tiers\s+CASE,PUNCT", sentence), (
        f"the sentence names an apply but not {APPLY_CMD}, so a reader cannot tell "
        "mechanical merges from meaning claims: " + sentence[:120])
    assert "#990" in sentence and "2026-09-16" in sentence, (
        "an unattended nightly write into the store has to be attributable on the "
        "page: whoever wants to retire the ruling needs to see which ruling it is "
        "and when — " + sentence[:120])


def test_the_write_path_diagram_shows_48_applying_the_mechanical_tiers():
    """Clause 2. The diagram is the first thing a reader consults to learn who
    writes to the store, and it said the opposite of the live job: #48's line was
    labelled `DEAD_DIAGRAM[0]` and the arrow below it ended in `DEAD_DIAGRAM[1]`.
    Fixing rule 2 alone leaves that reader still stopping the nightly apply.
    """
    text = _text()
    for dead in DEAD_DIAGRAM:
        assert dead not in text, (
            f"{dead!r} is still in the document — the diagram is the write-path "
            "map, so a stale label there is what makes an unattended apply look "
            "like an unauthorised one")
    node = _sweep_node(text)
    header = node.splitlines()[0]
    assert re.search(r"\bappl(?:y|ies)\b", header, re.I) and "CASE" in header \
        and "PUNCT" in header, (
        "#48's own line in the write-path diagram does not show the daily run "
        f"applying the CASE/PUNCT tiers itself: {header!r}")
    assert re.search(r"\bunattended\b|\bon schedule\b|\bnightly\b", node, re.I), (
        "the node describes an apply but never says it happens with nobody "
        "watching, which is the half that changed")
    assert re.search(r"SUFFIX[^.]*?\b(human|person)\b|\b(human|person)\b[^.]*?SUFFIX",
                     node, re.I), (
        "the node states the apply without its bound, so a reader following the "
        "arrows learns that #48 merges and never learns where it stops")


def test_the_rule_names_the_narrowing_not_a_repeal():
    """Clause 3. Two incidents were unattended applies, so the honest correction is
    a narrowing with its reason attached — not "applies are unattended now". A
    SUFFIX merge is a claim about meaning and still needs the semantic gate's
    unanimous verdict *and* a person; what bounds the two mechanical tiers is the
    shape of the apply, four guards named in the rule itself.
    """
    rules = _section(_text(), RULES_SECTION)
    assert re.search(r"SUFFIX[^.]*?unanimous[^.]*?\b(human|person)\b", rules, re.I), (
        "the section never marks a SUFFIX merge as gated and human, so the "
        "unattended tiers read as the general case and the narrowing is lost")
    # Every bound below is matched on `\s+`, not a space: this is a wrapped
    # document, and a pattern that only matches one line break away from the words
    # fails the day somebody reflows a paragraph.
    assert re.search(r"not\s+(?:the\s+)?absence\s+of\s+risk", rules, re.I), (
        "the section names the guards but not the reason they are load-bearing; "
        "the wipe and the 151-merge incident were both unattended applies, and a "
        "rule that implies the risk went away is how the next one is waved in")
    for bound in (r"degraded-graph\s+refusal", r"high-value\s+exclusion",
                  r"one\s+store\s+transaction", r"--resume"):
        assert re.search(bound, rules, re.I), (
            f"the rule bounds the unattended tiers but never names {bound!r}, and "
            "an unnamed guard is a guard a future refactor can drop")
    assert re.search(r"#67[^.]*?propos", rules, re.I), (
        "#67's propose-only status is the true half of the old sentence — its own "
        "`--apply` was retired 2026-09-04 — and correcting #48 must not sweep it "
        "into the same apply")


def test_review_log_names_2641_and_both_corrected_sites():
    """Clause 4. The section's existing form is a dated bullet naming what was
    stale and ending `Pinned by …` with the nodes that hold it; a correction with no
    entry is a correction the next review pass re-litigates.
    """
    entry = next((e for e in _review_entries(_section(_text(), REVIEW_SECTION))
                  if "#2641" in e), None)
    assert entry is not None, (
        "§Review log has no entry for #2641, so nothing records that the blanket "
        "apply rule was corrected here rather than in the archived vault copy")
    assert entry.startswith("- **2026-10-13"), (
        "the #2641 entry must carry the date the correction was made, the way every "
        "other entry in this section does: " + entry[:40])
    for site in ("rule 2", "diagram"):
        assert site in entry.lower(), (
            f"the entry names #2641 but never the {site} — both sites said it, and "
            "an entry that records only one leaves the other to be re-found")
    assert "Pinned by" in entry, (
        "the entry does not follow the section's `Pinned by …` form, so it reads as "
        "a claim rather than a checked one")
    assert "tests/test_kg_doc_claims.py" in entry, (
        "the pin must name the file that holds the nodes, not just 'a test'")
    cited = set(re.findall(r"::(test_\w+)", entry))
    assert cited, "the entry cites no test node by name, so its pin cannot resolve"
    defined = {n.name for n in ast.walk(ast.parse(Path(__file__).read_text()))
               if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
    missing = cited - defined
    assert not missing, (
        f"the entry pins {sorted(missing)}, which this module does not define — a "
        "pin to a node that does not exist is worse than no pin, because it closes "
        "the report that would have found the gap")


def test_the_two_stale_sites_fail_their_own_checks_when_restored():
    """The nodes above must be able to fail: each is run against the same bytes with
    the pre-#2641 wording spliced back in, and both must reject it.

    This is the shape of the bug rather than a hypothetical — the two claims lived in
    different sections, so this restores each one alone and shows the other section's
    correct wording does not excuse it.
    """
    text = _text()
    restored_rule = text.replace(
        _rule_block(text, "2."),
        f"2. **{DEAD_RULE_TITLE}.** {DEAD_RULE_SENTENCE}. Both\n"
        "   incidents were unattended applies.\n",
        1)
    assert DEAD_RULE_SENTENCE in restored_rule, "the splice did not take — nothing was tested"
    assert _apply_sentence(_section(restored_rule, RULES_SECTION)) is None, (
        "the apply-sentence predicate passed the blanket rule, so it is not "
        "checking for an unattended apply at all")

    node = _sweep_node(text)
    restored_diagram = text.replace(
        node,
        "entity-resolution-sweep.py  (#48, daily, " + DEAD_DIAGRAM[0] + ")\n"
        "        |  clusters near-duplicates, runs the semantic gate, prints a plan\n"
        "        v\n"
        "   " + DEAD_DIAGRAM[1],
        1)
    assert DEAD_DIAGRAM[0] in restored_diagram, "the splice did not take"
    header = _sweep_node(restored_diagram).splitlines()[0]
    assert not (re.search(r"\bappl(?:y|ies)\b", header, re.I) and "CASE" in header), (
        "the diagram predicate passed a #48 line that says the run is a dry run")
