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

import pytest

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

#: Rule 3's pre-#3221 wording for the two sentences #3221 re-derived. Both were TRUE
#: on disk and both named the wrong mechanism, which is why they are asserted absent
#: rather than merely superseded: a rule that is true for a reason it does not state is
#: how the next refactor drops the mechanism and keeps the sentence.
#:
#: `DEAD_BOOTSTRAP_SENTENCE` covered the floor with "bootstrapped only by a run that
#: went on to apply" and never said where the max-ratchet lived. After #2968 moved
#: `update_baseline` below the guard inside `if not baseline_measured:`, the ratchet was
#: unreachable on every run that READ a floor — every nightly #48 apply — and the file
#: froze at 50,152 (`recorded_at` 2026-10-17T15:19:37Z) while six applies grew the store
#: to 52,878, each reporting `baseline_active_edges: 50152`.
#:
#: `DEAD_SELF_HEAL` credited `update_baseline` being max-only with re-arming a backup
#: that had refused for want of a reference. That refusal is about an ABSENT file, which
#: only a bootstrap can answer, and the bootstrap that answers it is #48's dry-run — the
#: half the job's skill runs before the apply every night.
#:
#: `DEAD_DRY_RUN` is the pre-#3221 shape of the read-only bound: it names the seed and
#: says nothing about a floor the dry-run *read*, which is the half that has to be on the
#: page now the apply path is the one that raises.
DEAD_BOOTSTRAP_SENTENCE = "A floor is bootstrapped only by a run that went on to apply"
DEAD_SELF_HEAL = ("(b) `update_baseline` is max-only, so a slow leak never trips it, "
                  "which is also what self-heals a refused backup within one nightly sweep")
DEAD_DRY_RUN = "A dry-run bootstraps a floor it could not read (the fresh-install route)"


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


# ── rule 3's floor: where the ratchet lives, and what re-heals a refused backup ──
#
# #3221 is a documentation bug as much as a code bug. The code defect was three lines —
# `update_baseline` wrapped in `if not baseline_measured:` on both call sites — and every
# one of those lines was written deliberately, by #2968, to close a worse hole. What let
# the ratchet die unnoticed for a week was that rule 3 described the floor in a sentence
# that stayed true while the mechanism behind it went missing ("bootstrapped only by a
# run that went on to apply"), and credited max-only with a self-heal that was actually
# being performed by a different run than the one the sentence was about.


def _rule3(text: str) -> str:
    """Rule 3's whole block, scoped to §Rules that came out of the incidents.

    Section-scoped before `_rule_block` rather than handed straight to it: that matcher
    answers on the first `N. **` line in whatever text it is given, and this file numbers
    more than one list, so a "rule 3" somewhere else in the document must not be able to
    satisfy a claim about THE rule that governs `--apply`.
    """
    return _rule_block(_section(text, RULES_SECTION), "3.")


def _ratchet_sentence(block: str) -> str | None:
    """The one sentence stating where the floor may move, and to which count.

    Four requirements because the sentence has to be the rule and not the story of the
    bug: the floor moves on an APPLY that got past the guard's DECISION (not "a run that
    went on to apply", which after #2968 described a bootstrap and nothing else); it moves
    by RATCHET; it moves to the store's count BEFORE the apply, which is #1557's
    measure-before-write order; and the block's own explanatory sentence about what #2968
    broke — which mentions a ratchet, an apply and a decision — carries no `pre-apply`, so
    only the normative sentence clears this.
    """
    for s in _sentences(block):
        if (re.search(r"\bratchet", s, re.I)
                and re.search(r"\bappl(?:y|ies)\b", s, re.I)
                and re.search(r"\bdecision\b|\brefus", s, re.I)
                and re.search(r"pre-apply|before[^\n]{0,40}appl", s, re.I)):
            return s
    return None


def _dry_run_sentence(block: str) -> str | None:
    """The sentence bounding the dry-run to bootstrapping: it seeds a floor it could not
    read and NEVER raises one it read.

    This is the half that has to be on the page once the apply path raises the floor: with
    both halves able to write, "a dry-run seeds the floor" reads as "a dry-run maintains
    the baseline", and a reader then expects the nightly dry-run to keep the reference
    young — which is precisely the belief that let 50,152 ride for a week.
    """
    for s in _sentences(block):
        if (re.search(r"dry[- ]?run", s, re.I)
                and re.search(r"never\s+(?:raises|ratchets|writes|touches)", s, re.I)):
            return s
    return None


def _self_heal_sentence(block: str) -> str | None:
    """The sentence naming WHAT re-arms a backup that refused for want of a reference.

    It must be a BOOTSTRAP, and specifically #48's DRY-RUN — not max-only. Max-only cannot
    answer an absent file by construction: there is nothing to take a maximum with, and
    #2968 made an absent floor a refusal, so an unattended apply is not the run that
    re-seeds it.
    """
    for s in _sentences(block):
        if (re.search(r"self-heal", s, re.I)
                and re.search(r"bootstrap", s, re.I)
                and re.search(r"dry[- ]?run", s, re.I)
                and "#48" in s):
            return s
    return None


def _defined_nodes() -> set[str]:
    """Every `test_*` node defined in this module or in `tests/test_entity_sweep.py`.

    The #3221 review-log entry pins nodes in BOTH files — the behaviour in the sweep's own
    suite, the prose here — so its citations can only resolve against both, where #2641's
    entry was deliberately held to this module alone.
    """
    names: set[str] = set()
    for f in (Path(__file__), ROOT / "tests" / "test_entity_sweep.py"):
        for n in ast.walk(ast.parse(f.read_text())):
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_"):
                names.add(n.name)
    return names


def test_rule_3_names_where_the_ratchet_lives_and_what_re_heals_a_refused_backup():
    """Clause 5 of #3221: rule 3 has to say which run may move the floor, and which run
    re-arms a backup — the two things its old sentences left a reader to guess.

    The three dead sentences above are asserted absent, then each live replacement is
    located, then each is spliced back out and re-graded against its own predicate, so a
    node that only ever sees the current bytes cannot pass by accident of phrasing.
    Rule 3's (c) — the floor is written by a job in the family that reads it, which is
    Deliverable 3's independent-reference ask — is re-checked rather than rewritten: the
    ratchet is more reasons for that complaint, not fewer, and (c) is the sentence a
    reader needs to still be there after the paragraph around it moved.
    """
    text = _text()
    block = _rule3(text)
    for dead in (DEAD_BOOTSTRAP_SENTENCE, DEAD_SELF_HEAL, DEAD_DRY_RUN):
        assert dead not in block, (
            f"rule 3 still carries {dead!r}. It is not wrong so much as unfalsifiable: "
            f"it stayed true while the mechanism it described went missing, and it is "
            f"why nobody noticed the floor had stopped moving")

    ratchet = _ratchet_sentence(block)
    assert ratchet is not None, (
        "rule 3 never states, in one sentence, that the floor RATCHETS on an apply that "
        "got past the guard decision to the store's PRE-apply count. Without it, "
        "'where does the reference come from' has no answer on the page — which is how "
        "the floor sat at 50,152 while the store grew to 52,878 and three guards kept "
        "trip-pointing off it")
    dry_run = _dry_run_sentence(block)
    assert dry_run is not None, (
        "rule 3 never says the dry-run bootstraps a floor it could not read and NEVER "
        "raises one it read. With the apply now able to raise the floor, that bound is "
        "the difference between a read-only run and the writer of the reference")
    self_heal = _self_heal_sentence(block)
    assert self_heal is not None, (
        "rule 3 never says the one-night self-heal of a refused backup comes from #48's "
        "dry-run BOOTSTRAP. Crediting max-only instead points the next change at the "
        "wrong run: a backup refuses on an ABSENT reference, and max-only has no "
        "opinion about absence")
    assert re.search(r"max-only", self_heal, re.I) and re.search(r"not", self_heal, re.I), (
        "the self-heal sentence should also say it is NOT max-only, since that is the "
        "claim it is replacing: " + self_heal[:120])
    assert re.search(r"\(c\)[\s\S]{0,240}?written by a job in the family that\s+reads it",
                     block, re.I), (
        "rule 3's (c) is gone or reworded past recognition. The ratchet gives that "
        "complaint MORE weight, not less — the job that gates on the reference now also "
        "raises it — so (c) has to survive the paragraph around it")

    # Each predicate must reject the wording it replaced, one at a time.
    no_ratchet = block.replace(ratchet, DEAD_BOOTSTRAP_SENTENCE + ".", 1)
    assert DEAD_BOOTSTRAP_SENTENCE in no_ratchet, "the splice did not take — nothing tested"
    assert _ratchet_sentence(no_ratchet) is None, (
        "the ratchet predicate passed the pre-#3221 sentence, so it is not checking for "
        "a floor that moves on a passed decision")

    no_dry_run = block.replace(dry_run, DEAD_DRY_RUN + ".", 1)
    assert DEAD_DRY_RUN in no_dry_run, "the splice did not take — nothing tested"
    assert _dry_run_sentence(no_dry_run) is None, (
        "the dry-run predicate passed a sentence that only names the seed, which is the "
        "phrasing that let a ratcheting dry-run read as normal")

    no_self_heal = block.replace(self_heal, DEAD_SELF_HEAL + ";", 1)
    assert DEAD_SELF_HEAL in no_self_heal, "the splice did not take — nothing tested"
    assert _self_heal_sentence(no_self_heal) is None, (
        "the self-heal predicate passed the max-only credit, so it is not checking which "
        "run re-seeds an absent floor")


def test_review_log_names_3221_and_where_the_ratchet_moved():
    """The section's own form: a dated bullet naming what was stale, ending `Pinned by …`
    with the nodes that hold it. #3221 corrected rule 3 in a paragraph an operator reads
    before deciding whether a nightly write into `kg.sqlite` needs watching, and a
    correction with no entry is one the next review pass re-litigates.

    The entry also has to carry the measured pair — 50,152 frozen against 52,878 of
    store — because that is what makes it a record rather than a paraphrase of the rule,
    and its citations have to resolve across both test files it pins.
    """
    entry = next((e for e in _review_entries(_section(_text(), REVIEW_SECTION))
                  if "#3221" in e), None)
    assert entry is not None, (
        "§Review log has no entry for #3221, so nothing records that rule 3's floor "
        "sentence was corrected here rather than left as a true-but-empty claim")
    assert entry.startswith("- **2026-10-24"), (
        "the #3221 entry must carry the date the correction was made, the way every "
        "other entry in this section does: " + entry[:40])
    for site in ("rule 3", "dry-run", "max-only"):
        assert site in entry.lower(), (
            f"the entry names #3221 but never the {site} — the entry exists to say "
            "which sentence moved and to which mechanism")
    for figure in ("50,152", "52,878"):
        assert figure in entry, (
            f"the entry does not carry the measured {figure}; without the frozen floor "
            "beside the store's own count it restates the rule instead of recording what "
            "the rule failed to catch")
    assert "Pinned by" in entry, (
        "the entry does not follow the section's `Pinned by …` form, so it reads as a "
        "claim rather than a checked one")
    cited = set(re.findall(r"::(test_\w+)", entry))
    assert cited, "the entry cites no test node by name, so its pin cannot resolve"
    assert "tests/test_kg_doc_claims.py" in entry and "tests/test_entity_sweep.py" in entry, (
        "the entry pins only one file; the behaviour and the prose are pinned in two, and "
        "a reader needs to find both")
    missing = cited - _defined_nodes()
    assert not missing, (
        f"the entry pins {sorted(missing)}, which neither test file defines — a pin to a "
        "node that does not exist is worse than no pin, because it closes the report "
        "that would have found the gap")


# ── #3354: who types the edge #51 lands, and in which document ────────────────────
#
# #420 (`e4f6370f`, 2026-09-12) changed `land_approved_edges()` from
# `"type": p.get("relation_type") or "co_accessed"` to
# `"type": p.get("type") or p.get("relation_type") or "co_accessed"`
# (`scripts/memory/conversation_relations.py`, inside `land_approved_edges`), so an
# approved pair lands as the type the Stage-2 classifier returned and `co_accessed`
# survives only as the fallback for a pair nothing classified. Three documents kept
# asserting the pre-#420 behaviour, one of them this one; measured 2026-10-26 through
# `app.kg_store`, **0** of the 562 `conversation`-origin edges in the store are
# `co_accessed`, and none anywhere in it, with the newest conversation-origin edge
# stamped 2026-10-25T06:08:03Z — so the sentence described a type the live job had not
# written in six weeks, in a doc a person reads before trusting the graph's provenance.
#
# The corpus for clause 3 is `architecture/*.md` at the top level, which is bounded and
# enumerable today (`git grep -l co_accessed -- 'architecture/*.md'` answers three
# files), so the check globs rather than carrying a list a new doc can fall off.

ARCH_DIR = ROOT / "architecture"
AUTONOMY_DOC = ARCH_DIR / "autonomy.md"
CHAIN_TABLE_SECTION = "Knowledge-Graph Chain"
CHAIN_SECTION = "The write chain"

#: The writer whose behaviour the #51 sentences describe. Read with `ast` by
#: `test_the_quoted_fallback_chain_is_the_chain_the_code_runs`, which is what makes
#: this section's prose falsifiable against something other than itself.
RELATIONS_SRC = ROOT / "scripts" / "memory" / "conversation_relations.py"

#: The vault's fork of this note. It is a 2026-09-04 copy of the repo doc, retrievable
#: prose in segment `architecture`, so a stale sentence in it is served to recall just
#: like one here.
VAULT_KG = Path.home() / "obsidian" / "architecture" / "knowledge-graph.md"

#: The two shapes the #51 claim took: prose in the writer bullet, a cell in the chain
#: table. Both are asserted absent, so correcting one form and leaving the other is
#: caught rather than traded.
DEAD_RELATION_CLAIMS = (r"becom(?:e|es)\s+`?co_accessed`?\s+edges",
                        r"->\s+`?co_accessed`?\s+edges")

#: The three halves of the corrected sentence. See `_classifier_typed` for why each is
#: required and not merely nice to have.
_CLASSIFIER_RE = re.compile(r"stage[- ]2|classifier", re.I)
_FALLBACK_RE = re.compile(r"\bfallback\b", re.I)
_NOTHING_CLASSIFIED_RE = re.compile(r"nothing classif|unclassified", re.I)

#: An inline citation to a test node is a pointer to a check, not a claim about the
#: edge store, so it comes out before a unit is graded. This is not a hedge: §Review
#: log's `Pinned by …` lists name the nodes that hold the correction, and one of those
#: node names carries the very string this section polices, so an unstripped guard
#: fires on its own pin.
_CITATION_RE = re.compile(r"`(?:tests/\S+|::test_\w+)`")


def _claim(unit: str) -> str:
    """A unit with its test-node citations removed — see `_CITATION_RE`."""
    return _CITATION_RE.sub("", unit)


def _units(text: str) -> list[str]:
    """Each markdown claim as one grading unit: a table row, or one prose sentence.

    Two kinds because the corpus holds both. A `|` row is a unit on its own — the
    Knowledge-Graph Chain table packs one claim per row, so folding the table into a
    paragraph would let one row's `fallback` excuse another row's pre-#420 claim, the
    same cross-section leak `_section` exists to stop. Prose lines are joined before
    splitting because these docs are hard-wrapped near 78 columns and `co_accessed` and
    the word `fallback` routinely sit on adjacent lines of one sentence; splitting at
    every newline would put them in different units and fail a correct document.
    """
    units: list[str] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            units.extend(_sentences(" ".join(para)))
            para.clear()

    for line in text.splitlines():
        if not line.strip():
            flush()
        elif line.lstrip().startswith("|"):
            flush()
            units.append(line.strip())
        elif re.match(r"[-*]\s|^\d+\.\s", line.lstrip()):
            flush()          # a new list item is a new claim, not a wrapped sentence
            para.append(line.strip())
        else:
            para.append(line.strip())
    flush()
    return units


def _classifier_typed(units: list[str]) -> str | None:
    """The one unit saying #51's landed edge carries the classifier's type.

    All three halves are required. "Typed by the classifier" with no fallback erases
    the type an abstaining Stage 2 still lands; "`co_accessed` is a fallback" with no
    classifier leaves the pre-#420 default as the headline the reader skims; and a
    fallback bounded by nothing ("fallback if needed") never says when it fires, which
    is the vagueness that let the old sentence stay readable for six weeks after the
    code stopped matching it.
    """
    for u in units:
        c = _claim(u)
        if ("co_accessed" in c
                and _CLASSIFIER_RE.search(c)
                and _FALLBACK_RE.search(c)
                and _NOTHING_CLASSIFIED_RE.search(c)):
            return u
    return None


def _fallback_named(units: list[str]) -> list[str]:
    """Units that mention `co_accessed` without bounding it as the unclassified fallback.

    Weaker than `_classifier_typed` on purpose: clause 3 governs every surviving
    mention in the corpus, including ones that are only measuring the type's absence,
    and demanding each restate the classifier would push a real fact ("0 of 562 are
    `co_accessed`") out of the document. What a mention may never do is assert the type
    as the plain outcome.
    """
    return [u for u in units
            if "co_accessed" in _claim(u)
            and not (_FALLBACK_RE.search(_claim(u))
                     and _NOTHING_CLASSIFIED_RE.search(_claim(u)))]


def _writer_bullet(text: str) -> str:
    """The `conversation_relations.py` (#51) bullet, from its `- **` line to the next.

    Bullet-scoped, not section-scoped: §The write chain also describes #24, #74 and #48,
    and a true sentence about one of those must not be able to answer a claim about the
    edges #51 lands.
    """
    section = _section(text, CHAIN_SECTION)
    lines = section.splitlines()
    start = next((i for i, l in enumerate(lines)
                  if l.lstrip().startswith("- **") and "conversation_relations.py" in l),
                 None)
    assert start is not None, (
        f"§{CHAIN_SECTION} lost its `conversation_relations.py` bullet entirely — every "
        "check on #51's edge type has been silently deleted with it")
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].lstrip().startswith("- **")), len(lines))
    return "\n".join(lines[start:end])


def _bullet_units(text: str) -> list[str]:
    return _sentences(" ".join(l.strip() for l in _writer_bullet(text).splitlines()))


def _vault_note() -> str | None:
    """The vault's copy of this note, or None when the note is not on disk.

    Absent is a legitimate future state, not a broken checkout: the note is a
    2026-09-04 fork of this doc, and the ruling #3354 leaves open is whether to keep it
    maintained at all or archive it the way #2330 archived
    `~/obsidian/architecture/autonomy-system.md`. This file is the authority; the two
    vault nodes below exist so the fork cannot keep serving a refuted sentence to
    retrieval while it is still live, so they grade it when it is on disk and say
    nothing when it is not.
    """
    return VAULT_KG.read_text() if VAULT_KG.is_file() else None


def _chain_table_row(text: str) -> str:
    """#51's row in the Knowledge-Graph Chain table."""
    section = _section(text, CHAIN_TABLE_SECTION)
    row = next((u for u in _units(section) if re.match(r"\|\s*#51\s*\|", u)), None)
    assert row is not None, (
        f"architecture/autonomy.md has no #51 row in §{CHAIN_TABLE_SECTION} any more; "
        "the job is daily and still writing edges, so the table is wrong by omission")
    return row


def test_the_51_writer_bullet_says_the_edge_is_classifier_typed():
    """Clause 1. §The write chain's #51 bullet said co-access pairs "become
    `co_accessed` edges", which has not been the rule since #420 (`e4f6370f`,
    2026-09-12): `land_approved_edges()` writes the classifier's verdict first.

    The corrected sentence has to be attributable, so the node also asks for the
    change id and the measurement it rests on — `app.kg_store` on 2026-10-26, 0
    `co_accessed` among the 562 `conversation`-origin edges. A rule with no
    provenance is the sentence that goes stale next.
    """
    text = _text()
    section = _claim(_section(text, CHAIN_SECTION))
    for pat in DEAD_RELATION_CLAIMS:
        assert not re.search(pat, section), (
            f"§{CHAIN_SECTION} still matches {pat!r}: it names the type #51 lands "
            "before Stage 2 has a verdict, and the store has not held that type since "
            "the change that made it false")
    claim = _classifier_typed(_bullet_units(text))
    assert claim is not None, (
        "the #51 writer bullet never says, in one sentence, that the landed edge "
        "carries the Stage-2 classifier's type and `co_accessed` is only the fallback "
        "for a pair nothing classified. `land_approved_edges()` writes "
        "`p.get('type') or p.get('relation_type') or 'co_accessed'`; a reader "
        "reasoning about the graph's provenance from this bullet gets it backwards")
    assert "#420" in claim, (
        "the sentence states the rule but not the change that made it, so the next "
        "reader cannot tell which side of 2026-09-12 a bug report is on: " + claim[:120])
    assert "app.kg_store" in claim and "2026-10-26" in claim, (
        "the sentence recites the code without the measurement that would have caught "
        "the old wording — a code quote is a claim about intent, the store is the "
        "claim about behaviour: " + claim[:120])


def test_the_kg_chain_table_row_says_the_edge_is_classifier_typed():
    """Clause 2. `architecture/autonomy.md`'s Knowledge-Graph Chain table is the
    surface an operator reads to learn what each job writes, and #51's row read
    "-> `co_accessed` edges (provenance INFERRED)" — the pre-#420 default as the
    job's whole role, in the one column nobody cross-checks against the code.
    """
    row = _chain_table_row(AUTONOMY_DOC.read_text())
    for pat in DEAD_RELATION_CLAIMS:
        assert not re.search(pat, _claim(row)), (
            f"the #51 row still matches {pat!r}: {row[:160]!r}")
    assert _classifier_typed([row]) is not None, (
        "the #51 row never says the edge is typed by the Stage-2 classifier with "
        f"`co_accessed` as the unclassified fallback: {row[:160]!r}")
    assert "INFERRED" in row.upper(), (
        "correcting the type must not drop the provenance half — INFERRED is what "
        f"tells a reader nobody curated the edge: {row[:160]!r}")


def test_no_architecture_doc_says_pairs_simply_become_co_accessed_edges():
    """Clause 3, over the bounded corpus `architecture/*.md`.

    Three files mention `co_accessed` today and the glob is the point: a hand-carried
    allow-list closes a property only over the files someone remembered, and the defect
    being fixed here is precisely one doc being corrected while its twin kept the
    sentence. Every surviving mention has to sit in a sentence that bounds it as the
    unclassified fallback, and neither dead form may appear at all.
    """
    documents = {p.name: p.read_text() for p in sorted(ARCH_DIR.glob("*.md"))}
    assert documents, f"{ARCH_DIR} matched no markdown — the corpus glob resolved to nothing"
    # Positive controls: the phrase under guard must still be somewhere in the corpus,
    # or a 0-hit loop here would be indistinguishable from a passing check.
    mentioning = sorted(n for n, t in documents.items() if "co_accessed" in t)
    assert mentioning, (
        "no architecture doc mentions `co_accessed` at all, so this node proves "
        "nothing — the sentence was deleted rather than corrected")
    for pat in DEAD_RELATION_CLAIMS:
        hits = sorted(n for n, t in documents.items() if re.search(pat, _claim(t)))
        assert not hits, f"{pat!r} still appears in {hits}"
    bad = [(n, u) for n, t in documents.items() for u in _fallback_named(_units(t))]
    assert not bad, ("these sentences name `co_accessed` without bounding it as the "
                     "unclassified fallback:\n"
                     + "\n".join(f"  {n}: {u[:110]}" for n, u in bad))


def test_the_quoted_fallback_chain_is_the_chain_the_code_runs():
    """The seam between the sentence and the writer.

    Every node above grades prose against prose, which is how the pre-#420 sentence
    survived six weeks: it read fine and nothing re-checked it against
    `land_approved_edges()`. This one parses the function with `ast` and takes the
    `or`-chain that decides an edge's `type`, so the literal the doc calls the fallback
    has to be the literal the code falls back to. `ast` rather than a grep because a
    grep is satisfied by the string in a comment just as happily as by the expression
    that writes the row.
    """
    tree = ast.parse(RELATIONS_SRC.read_text())
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name == "land_approved_edges"), None)
    assert fn is not None, (
        f"{RELATIONS_SRC.relative_to(ROOT)} no longer defines `land_approved_edges`; "
        "the three docs citing it are wrong by that much too")
    for node in ast.walk(fn):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if (isinstance(k, ast.Constant) and k.value == "type"
                        and isinstance(v, ast.BoolOp) and isinstance(v.op, ast.Or)):
                    literals = [x.value for x in v.values if isinstance(x, ast.Constant)]
                    assert literals == ["co_accessed"], (
                        f"the writer's `type` chain now falls back through {literals}; "
                        "every architecture doc naming `co_accessed` as the unclassified "
                        "fallback is stale the moment that changes")
                    assert isinstance(v.values[-1], ast.Constant), (
                        "`co_accessed` is no longer the last resort in the `type` chain, "
                        "so the docs' ''only the fallback'' wording describes an "
                        "expression that does not exist")
                    bullet = _writer_bullet(_text())
                    assert re.search(r"or\s+`?\"?co_accessed\"?`?", bullet), (
                        "the code still falls back to `co_accessed` but the #51 bullet "
                        "no longer quotes it, so the reader has lost the half that says "
                        "what an unclassified pair lands as")
                    return
    raise AssertionError(
        "`land_approved_edges` no longer sets a `type` key from an `or` chain — the "
        "sentence three docs carry describes an edge type the writer cannot produce")


def test_the_51_claims_reject_the_pre_420_wording_when_restored():
    """The nodes above must be able to fail, so each predicate is re-run against the
    bytes with the pre-#420 wording spliced back — the writer bullet in this doc, the
    row in `architecture/autonomy.md`, one site at a time.

    One at a time is the shape of the original bug: the claim lived in three files, so
    a check that passes because a *corrected* file mentions the classifier somewhere is
    exactly the check that let the stale ones stand.
    """
    text = _text()
    restored_doc = text.replace(
        _writer_bullet(text),
        "- **`conversation_relations.py`** (#51) — co-access pairs from session\n"
        "  trajectories become `co_accessed` edges, `provenance: INFERRED`, with the\n"
        "  trajectory as `source_doc`.\n", 1)
    assert "become `co_accessed` edges" in restored_doc, (
        "the splice did not take — nothing was tested")
    assert _classifier_typed(_bullet_units(restored_doc)) is None, (
        "the classifier-typed predicate passed the pre-#420 bullet, so it is not "
        "reading which type an approved pair lands as")
    assert _fallback_named(_units(restored_doc)), (
        "the corpus predicate passed the pre-#420 bullet, so a mention with no "
        "fallback bound walks past it")
    assert re.search(DEAD_RELATION_CLAIMS[0], _writer_bullet(restored_doc)), (
        "the dead-prose pattern does not match the sentence it replaced")

    auto = AUTONOMY_DOC.read_text()
    restored_auto = auto.replace(
        _chain_table_row(auto),
        "| #51 | Conversation Relation Linking | daily | #56 | yes | Co-access pairs "
        "from trajectories -> `co_accessed` edges (provenance INFERRED) |", 1)
    assert "-> `co_accessed` edges" in restored_auto, "the splice did not take"
    assert _classifier_typed(_units(restored_auto)) is None, (
        "the classifier-typed predicate passed the pre-#420 table row")
    assert re.search(DEAD_RELATION_CLAIMS[1], _chain_table_row(restored_auto)), (
        "the dead-table pattern does not match the row it replaced")


def test_the_vault_note_says_the_edge_is_classifier_typed():
    """Clause 4. The vault's `architecture/knowledge-graph.md` carried this doc's old
    #51 bullet verbatim — the same sentence, the same falsity, in the copy retrieval
    reads — and its frontmatter still claimed `updated: 2026-09-04`, which is what made
    the fork look authoritative to a reader who could not see it predated #420 by a week.

    Graded on the note only while it is on disk: see `_vault_note`.
    """
    note = _vault_note()
    if note is None:
        pytest.skip(f"{VAULT_KG} is not on disk — see _vault_note for why that is a "
                    "state and not a broken checkout")
    for pat in DEAD_RELATION_CLAIMS:
        assert not re.search(pat, _claim(note)), f"the vault note still matches {pat!r}"
    claim = _classifier_typed(_bullet_units(note))
    assert claim is not None, (
        "the vault note's #51 bullet never says the landed edge carries the Stage-2 "
        "classifier's type with `co_accessed` as the fallback for a pair nothing "
        "classified — the sentence #3354 was filed to remove")
    assert "#420" in claim, (
        "the corrected vault sentence names no change, so it cannot be checked against "
        "the code it describes: " + claim[:120])
    updated = re.search(r"^updated:\s*(\S+)\s*$", note, re.M)
    assert updated is not None, "the vault note lost its `updated:` frontmatter"
    assert updated.group(1) != "2026-09-04", (
        f"`updated: {updated.group(1)}` — the fork still presents itself as predating "
        "#420, which is the reason its stale prose was trusted")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", updated.group(1)), (
        f"`updated: {updated.group(1)}` is not a date")


def test_the_vault_note_diagram_shows_48_applying_the_mechanical_tiers():
    """Clause 5. The fork's write-path diagram still labelled #48's daily run
    `DRY RUN ONLY` and ended its arrow at a human running `--apply` — the identical
    pre-#2641 defect, uncorrected in the copy while the original was fixed.

    The note's own §Rules sentence for the same claim is graded here too, with this
    module's existing predicates: correcting the diagram while rule 2 still read
    "Applies are attended" would leave one file telling a reader both that the nightly
    apply happens unattended and that a person must run it, which is the disagreement
    that caused the original failure rather than a symptom of it.
    """
    note = _vault_note()
    if note is None:
        pytest.skip(f"{VAULT_KG} is not on disk — see _vault_note")
    for dead in DEAD_DIAGRAM:
        assert dead not in note, (
            f"{dead!r} is still in the vault note; the diagram is the write-path map, "
            "so a stale label there is what makes an unattended apply look "
            "unauthorised")
    node = _sweep_node(note)
    header = node.splitlines()[0]
    assert re.search(r"\bappl(?:y|ies)\b", header, re.I) and "CASE" in header \
        and "PUNCT" in header, (
        "#48's line in the vault diagram does not show the daily run applying the "
        f"CASE/PUNCT tiers itself: {header!r}")
    assert re.search(r"\bunattended\b|\bon schedule\b|\bnightly\b", node, re.I), (
        "the vault node describes an apply but never says it happens with nobody "
        "watching, which is the half that changed")
    assert re.search(r"SUFFIX[^.]*?\b(human|person)\b|\b(human|person)\b[^.]*?SUFFIX",
                     node, re.I), (
        "the vault node states the apply without its bound, so a reader following the "
        "arrows learns that #48 merges and never learns where it stops")
    for dead in (DEAD_RULE_TITLE, DEAD_RULE_SENTENCE):
        assert dead not in note, (
            f"the diagram was corrected but the note's rulebook still carries {dead!r}, "
            "so the same file now contradicts itself about whether a person must run "
            "`--apply`")
    assert _apply_sentence(_section(note, RULES_SECTION)) is not None, (
        f"§{RULES_SECTION} of the vault note never says in one sentence that #48 "
        f"applies {APPLY_CMD} unattended — the repo doc's rule, narrowed not repealed")


def test_review_log_names_3354_and_the_three_sites():
    """The section's own form, as for #2641 and #3221: a dated bullet naming what was
    stale and ending `Pinned by …`. #3354's defect was specifically that one copy of a
    sentence got corrected and its twins did not, so the entry has to name all three
    sites — an entry that records only this file is the seed of the next round of the
    same bug.
    """
    entry = next((e for e in _review_entries(_section(_text(), REVIEW_SECTION))
                  if "#3354" in e), None)
    assert entry is not None, (
        "§Review log has no entry for #3354, so nothing records that #51's edge type "
        "was corrected in three places rather than one")
    assert entry.startswith("- **2026-10-26"), (
        "the #3354 entry must carry the date the correction was made, the way every "
        "other entry in this section does: " + entry[:40])
    for site in ("write chain", "autonomy.md", "vault"):
        assert site in entry.lower(), (
            f"the entry names #3354 but never the {site} — three copies carried the "
            "sentence and an entry that records one leaves two to be re-found")
    assert "Pinned by" in entry and "tests/test_kg_doc_claims.py" in entry, (
        "the entry does not follow the section's `Pinned by …` form naming this file, "
        "so it reads as a claim rather than a checked one")
    cited = set(re.findall(r"::(test_\w+)", entry))
    assert cited, "the entry cites no test node by name, so its pin cannot resolve"
    defined = {n.name for n in ast.walk(ast.parse(Path(__file__).read_text()))
               if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
    missing = cited - defined
    assert not missing, (
        f"the entry pins {sorted(missing)}, which this module does not define — a pin "
        "to a node that does not exist is worse than no pin, because it closes the "
        "report that would have found the gap")
