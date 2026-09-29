"""#605: the two files that describe `vllm:prefix_cache_hits_total` /
`prefix_cache_queries_total` must not disagree about what that ratio means on
the engine that is actually running.

`agent-services/bin/bench-prefix-reuse.py` trap #1 (written 2026-09-06 at
`9a0a1d8`) said the pair "is NOT a reuse rate": it summed queries across every
KV cache group, so one 60k-token prompt logged 673,179 queries -- an 11.2x
amplification -- and the ratio read ~68% while real cross-request reuse was
zero. `app/vllm_metrics.py` fed exactly that pair to the Mission Control
`vllm` card as `prefix_cache_hit_rate`. One engine, two mutually exclusive
statements.

The live counters settle it in the card's favour, and the newest reading is the
one a reader can reproduce with one curl: 2026-09-22T05:47Z on a primary booted
that same day, `prefix_cache_queries_total` 148,645,180 against
`prompt_tokens_total` 148,644,807 -- 373 tokens apart, 1.0000025x, not 11.2x --
with the djev engine equal outright at 6,636,812 == 6,636,812. Earlier reads of
the same pair on Qwen3.8-Flash-Next-nvfp4: equal outright on 2026-09-13 (both
2,050,518,768) and 423 apart at a 3.03-billion-token lifetime
(2026-09-21T21:47Z). One 2026-09-21 probe window carrying production traffic
moved Δqueries == Δprompt_tokens at 6,308,395 == 6,308,395, the bench's own
60,005-token prompt logged 60,005 queries in its quiet window with d_hits 54,400
equal to that request's `usage.prompt_tokens_details.cached_tokens` 54,400, and
the 2026-09-13 probe read delta-queries over delta-prompt_tokens at 1.000 on all
8 of its windows. The 11.2x belonged to the 2026-09-06 qwen4_exp MTP boot
(`9a0a1d8`, whose own subject is "proof the eagle-group warning is benign"),
whose speculative-decode draft head adds a second KV cache group. So the
docstring was the false artifact, and the cost of trusting it was real: a reader
who believed it would wave away a token-level reuse rate the card had been
reporting honestly the whole time.

The gap between the two series is small but not constant -- 239 tokens at a
2.64e9 lifetime, 423 at 3.03e9, 373 at 1.49e8, all read 2026-09-21 or
2026-09-22 -- so what is pinned here is parity in the sense the defect is
measured in: a few hundred absolute tokens on every read, never the factor of
11.2 the stale docstring asserted. No arithmetic bound is pinned, because none is
needed -- no rounding of a token count reaches 11.2x, and a counter pair that
multiplies is exactly what the wording has to stop claiming.

What this file pins is the *shape* of the reconciliation, not the readings. The
historical figures stay -- they are the evidence, and dropping them would leave
the next reader without the reason the caveat exists -- but an inflation claim
may only appear in a sentence that names the boot it was measured on, and each
file must state the parity condition beside the pair it conditions. Pinning the
counters themselves would be wrong: lifetime counters reset on every engine
boot, so only a dated measurement survives re-reading.

Positive controls, because a pattern that is empty against the corpus returns 0
exactly like an absent claim does: every "the stale wording is gone" assertion
is paired with a figure that must still be present, and the `_RATIOS` site
finder asserts it recovered that comment's own pre-existing "Ratio counters"
header before it judges what the comment now says.
"""

from __future__ import annotations

import ast
import os
import re
import stat as statmod
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "agent-services" / "bin" / "bench-prefix-reuse.py"
METRICS = ROOT / "app" / "vllm_metrics.py"
LAUNCHER = ROOT / "agent-services" / "bin" / "start-qwen38-flash-next.sh"

#: The two constants trap #2 and the launcher's MTP block both stated as rules
#: until #1847. Checked WITHOUT the tilde on purpose: a rewrite that keeps the
#: number and drops the tilde has moved the claim, not retired it.
RETIRED_CONSTANTS = ("96%", "18x")

# The standing, unqualified denial, verbatim from 9a0a1d8 through 333ca748.
# Denominator at triage: 1 hit in the bench script, 0 anywhere else.
STALE_DENIAL = "is NOT a reuse rate"

# A sentence may assert an inflated query count only if it says *where* it was
# inflated. These are the anchors that make the claim historical.
BOOT_ANCHORS = ("2026-09-06", "qwen4_exp", "MTP", "eagle")

#: The subset that names a *configuration* rather than a date. Clause 1 asks for
#: "the boot/config the 11.2x was measured on", and a date alone does not say
#: what to look for on a future boot -- a same-family speculative-decode build
#: could carry the defect forward while every sentence stayed merely dated.
CONFIG_ANCHORS = ("qwen4_exp", "MTP", "eagle")

# The inflation claims themselves, in whatever file carries them.
INFLATION_CLAIMS = ("673,179", "11.2x", "~68%", "across every group",
                    "across every KV cache group")


def _bench_doc() -> str:
    doc = ast.get_docstring(ast.parse(BENCH.read_text()))
    assert doc, f"{BENCH} has no module docstring to qualify anything"
    return doc


def _trap_one() -> str:
    """Trap #1's bullet, up to but excluding trap #2's."""
    doc = _bench_doc()
    assert "\n  1. " in doc, "the trap list lost its numbering"
    rest = doc[doc.index("\n  1. "):]
    end = rest.find("\n  2. ")
    assert end > 0, "trap #2 vanished: PASSES = 5 has no stated reason left"
    return rest[:end]


def _trap_two() -> str:
    """Trap #2's bullet: its heading to the end of the module docstring.

    The mirror of `_trap_one()`, and the same positive control applies -- the
    assert below is that the split point exists, so a finder that located
    nothing fails loudly instead of reading as "the qualifier is missing".
    """
    doc = _bench_doc()
    start = doc.find("\n  2. ")
    assert start > 0, "trap #2 vanished: PASSES = 5 has no stated reason left"
    return doc[start:]


def _launcher_mtp_comment() -> str:
    """The comment block physically attached above the `MTP_ENABLED=` default.

    The second carrier of trap #2's constants: `bench-prefix-reuse.py` sends a
    reader to this block "for the full result table", so a corrected bullet whose
    reader lands on the same retired rule here has been corrected on paper only.
    """
    lines = LAUNCHER.read_text().splitlines()
    sites = [i for i, line in enumerate(lines) if line.startswith("MTP_ENABLED=")]
    assert len(sites) == 1, f"expected one MTP_ENABLED default, found {len(sites)}"
    block: list[str] = []
    j = sites[0] - 1
    while j >= 0 and lines[j].lstrip().startswith("#"):
        block.append(lines[j])
        j -= 1
    found = "\n".join(reversed(block))
    # Positive control: the walk really reached this launcher's own MTP block, so
    # a finder that recovered nothing cannot read as "the stale rule is gone".
    assert "SCARY BOOT WARNING IS BENIGN" in found, (
        "site finder recovered no MTP comment above MTP_ENABLED; this assertion is "
        "empty against the file, not a verdict on it")
    return found


def _flat(text: str) -> str:
    """One space between every token, so a rewrapped bullet is the same text."""
    return " ".join(text.split())


def _sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))


def _inflation_sentences(text: str) -> list[str]:
    return [s for s in _sentences(text)
            if any(claim in s for claim in INFLATION_CLAIMS)]


def _ratios_comment() -> str:
    """The comment block physically attached above `_RATIOS = (`."""
    lines = METRICS.read_text().splitlines()
    sites = [i for i, line in enumerate(lines) if line.startswith("_RATIOS = (")]
    assert len(sites) == 1, f"expected one _RATIOS definition, found {len(sites)}"
    block: list[str] = []
    j = sites[0] - 1
    while j >= 0 and lines[j].lstrip().startswith("#"):
        block.append(lines[j])
        j -= 1
    found = "\n".join(reversed(block))
    # Positive control: the walk really reached this site's own comment, so a
    # finder that grabbed nothing cannot read as "the caveat is missing".
    assert "Ratio counters" in found, (
        "site finder recovered no comment above _RATIOS; this assertion is "
        "empty against the file, not a verdict on it")
    return found


# ── clause 1: the bench docstring ──────────────────────────────────────

def test_the_amplification_survives_only_inside_a_sentence_naming_its_boot():
    """Delete the date or the boot name and the 11.2x silently becomes a
    standing property of the counter again -- which is the exact defect."""
    claims = _inflation_sentences(_trap_one())
    assert claims, (
        "trap #1 no longer states the 673,179 / 11.2x / ~68% evidence at all; "
        "qualifying the claim is not the same as deleting it")
    for sentence in claims:
        assert any(anchor in sentence for anchor in BOOT_ANCHORS), (
            f"unanchored inflation claim: {sentence[:90]!r}")
    assert any(cfg in sentence for sentence in claims for cfg in CONFIG_ANCHORS), (
        "every inflation claim is dated but none says which configuration "
        "produced it, so a future speculative-decode boot inherits the warning "
        "with nothing to match it against")


def test_the_parity_statement_names_prompt_tokens_and_a_verified_build():
    """Clause 1's parity half. The equality is the clause's own wording
    (`prefix_cache_queries_total == prompt_tokens_total`), so a rewording that
    drops the denominator or the date goes red rather than quietly weakening the
    claim to "the ratio looks fine"."""
    bullet = _trap_one()
    parity = [s for s in _sentences(bullet)
              if "1:1" in s or "parity" in s or "==" in s]
    assert parity, "trap #1 no longer states the 1:1 parity condition"
    for sentence in parity:
        assert "prompt_tokens_total" in sentence, (
            f"parity stated without its denominator: {sentence[:90]!r}")
    assert "prefix_cache_queries_total == prompt_tokens_total" in bullet, (
        "parity is no longer stated as the equality the clause names")
    assert "Qwen3.8-Flash-Next-nvfp4" in bullet, "which build was it verified on?"
    assert "2026-09-13" in bullet, "the parity statement carries no date"


def test_the_bullet_equates_the_counter_ratio_with_cached_tokens():
    """The decisive half: on a parity boot d_hits/d_queries *is* the
    per-request cached-token figure, so the counter is not merely
    un-inflated, it is the same measurement the bench prints."""
    bullet = _trap_one()
    # The fully-qualified response field, not bare "cached_tokens": the busy-
    # window caveat sentence below the equivalence also mentions both terms to
    # say the opposite (the window's d_hits/d_queries is the fleet's rate, not
    # yours), so a pattern that loose stays green with the equivalence deleted.
    equated = [s for s in _sentences(bullet)
               if "d_hits" in s and "usage.prompt_tokens_details.cached_tokens" in s]
    assert equated, (
        "nothing in trap #1 says the counter ratio equals per-request "
        "usage.prompt_tokens_details.cached_tokens on current builds")
    assert any("60,005" in s for s in equated), (
        "the equivalence is asserted without the request it was measured on, "
        "so it is prose about a number rather than the number itself")


def test_the_stale_denial_is_gone_from_both_files():
    for text in (_bench_doc(), METRICS.read_text()):
        assert STALE_DENIAL not in text
    # Positive control: the files still carry the figure the denial was about,
    # so the assertion above is a retraction and not an accidental deletion.
    assert "673,179" in _bench_doc() and "673,179" in METRICS.read_text()


def test_bench_still_runs_five_passes():
    """PASSES >= 3 is trap #2's requirement and the item says leave it at 5."""
    assert re.search(r"(?m)^PASSES = 5(\s+#.*)?$", BENCH.read_text()), (
        "PASSES moved off 5; trap #2's two warm-up passes need three more")


# ── #1635: trap #2's constant is dated, and the script still executes ────
#
# #605 rewrote trap #1 -- re-read, re-timed, dated to the hour -- and left trap
# #2 as the bare 2026-09-06 constant beside it, so one docstring offered a
# measured claim and an unmeasured one in the same numbered list. These nodes
# hold the qualifier, its evidence, and the mode bit that same landing dropped.

def test_trap_two_dates_the_two_pass_figure_as_unverified_on_this_build():
    """Clause 1: the figure may stay, but only beside a date and a statement that
    this build has never confirmed it.

    Deleting the constant is not the fix -- `test_bench_still_runs_five_passes`
    needs PASSES = 5 to keep a stated reason, and the next reader still needs to
    know what the run is guarding against -- so the qualifier is checked *beside*
    the claim it qualifies, and the three requirements (the word unverified, a
    date, and "current build") have to land in one sentence: scattered words pin
    nothing.
    """
    bullet = _flat(_trap_two())
    assert "TWO warm-up passes" in bullet, (
        "trap #2 no longer states the figure that needs qualifying; a qualifier "
        "parked next to a deleted claim pins nothing")
    qualified = [s for s in _sentences(bullet) if "unverified" in s.lower()]
    assert qualified, (
        "trap #2 states the two-warm-up-pass figure with no statement that it is "
        "unverified")
    assert any("current build" in s.lower()
               and re.search(r"\b20\d\d-\d\d-\d\d\b", s) for s in qualified), (
        "no sentence says the figure is unverified on the CURRENT build and dates "
        "that statement, so the number still reads as measured on this boot")
    assert "2026-09-06" in bullet, (
        "the date the constant was written on is gone, so nothing tells a reader "
        "which boot it was measured under")


def test_trap_two_carries_the_2026_09_21_measurement_that_contradicts_the_count():
    """Clause 2: the qualifier has to hold the measurement, not only the doubt.

    "Unverified" tells a reader to distrust the number; the dated table tells them
    why, and in which direction it is wrong. Each figure is required inside the
    sentence naming the run it came from, so a rewording that keeps the words and
    drops the evidence -- or parks the numbers in an unrelated line -- goes red.
    """
    bullet = _flat(_trap_two())
    cited = [s for s in _sentences(bullet) if "2026-09-21" in s]
    assert cited, "trap #2 cites no 2026-09-21 observation as its reason"
    assert any("54,400" in s and "60,005" in s for s in cited), (
        "the 2026-09-21 citation lost the pass-2 figures -- cached_tokens 54,400 "
        "of that pass's own 60,005-token prompt -- and is an accusation rather "
        "than a measurement")
    assert any("pass 1" in s and "pass 2" in s for s in cited), (
        "nothing says which passes were cold in the run being cited")
    assert any(s.lower().count("one cold pass") for s in cited), (
        "the run is cited without saying what it shows: ONE cold pass, not two")
    evicted = [s for s in _sentences(bullet) if "1,349,507" in s]
    assert evicted and any("pass 5" in s for s in evicted), (
        "pass 5 returning to cached_tokens 0 after 1,349,507 tokens of other "
        "traffic is gone, so the cited table has nothing explaining a cached "
        "count that falls back to zero")


# ── #1847: trap #2's headline is the measurement, and the retired run is not owed
#
# 53b761f9 date-stamped the 2026-09-06 constants as unverified and added the
# 2026-09-21 table *under* them, which left a reader meeting the bullet top-down
# with the retired numbers first and their correction somewhere below. #1635's
# owed_settled ruling then retired the idle-boot run that was the only route to
# re-timing them, so the headline had no surviving evidence and no route to new
# evidence at once. These nodes pin the swap the ruling ordered: the measured
# figures as the headline, the retired constants gone from the whole bullet, no
# forward pointer, the asymmetry that makes the loaded run enough to bound the
# claim, and the launcher that carried the same constants re-attributed.

def test_trap_two_headlines_the_measured_figures_and_drops_the_retired_constants():
    """Clause 1: every headline figure sits in a sentence naming production load
    on 2026-09-21, and both 2026-09-06 constants are gone from the whole bullet.

    The figures are checked sentence-scoped rather than bullet-scoped so a
    rewrite that keeps the words "production load" in one line and parks the
    numbers in another goes red: the labelling IS the claim, since an unlabelled
    90.7% reads as a property of the build rather than of one loaded run.
    """
    bullet = _flat(_trap_two())
    measured = [s for s in _sentences(bullet)
                if "2026-09-21" in s and "production load" in s.lower()]
    assert measured, (
        "trap #2's figures are stated in no sentence naming production load on "
        "2026-09-21, so the run they were measured on is unstated")
    assert any("90.7%" in s and "54,400" in s and "60,005" in s for s in measured), (
        "the headline lost the reuse it was measured to: 90.7% = 54,400 of that "
        "pass's own 60,005-token prompt")
    assert any("5.95" in s and "0.54" in s and "11.0x" in s for s in measured), (
        "the headline lost the timing it was measured to: 5.95 s cold against "
        "0.54 s warm, an 11.0x ratio")
    assert any("one cold pass" in s.lower() for s in measured), (
        "the headline states figures but not what they show: ONE cold pass")
    for constant in RETIRED_CONSTANTS:
        assert constant not in bullet, (
            f"the retired 2026-09-06 constant {constant!r} is still in trap #2, so "
            "the bullet still promises a speed-up no run since has reproduced")
    # Positive control: the replacements are in the bullet, so the loop above is
    # a retirement of the old constants and not a bullet that lost all its numbers.
    assert "90.7%" in bullet and "11.0x" in bullet, (
        "positive control failed: neither the reuse figure nor the speed-up ratio "
        "is in the bullet, so the absence checks above pinned nothing")


def test_trap_two_sends_the_reader_to_no_idle_boot_run():
    """Clause 3: the pointer is gone, and its absence carries a date and a reason.

    The reason is structural rather than a preference: the window that measurement
    needs opens only at a boot a human triggers, so no round can book it and no
    reader can be told to wait for it. The positive control is the retirement's
    provenance -- #1635 is still named, which is what makes a missing obligation
    a retirement instead of an omission a later run will re-file.
    """
    bullet = _flat(_trap_two())
    assert "owe" not in bullet.lower(), (
        "trap #2 still tells a reader that an item, or some future run, owes a "
        "measurement -- the pointer #1635's ruling retired is back")
    retired = [s for s in _sentences(bullet) if "retired" in s.lower()]
    assert retired, (
        "the idle-boot pointer left nothing behind: no dated removal, no reason")
    assert any("2026-09-29" in s and "_restart_primary" in s
               and "scripts/automod/promote.py:1880" in s and "warranted" in s
               for s in retired), (
        "the drop is not dated with its reason: the window closes minutes after a "
        "boot only Alan gates (_restart_primary, "
        "scripts/automod/promote.py:1880), so no dedicated restart is warranted")
    # The gate the reason cites has to be where the sentence says it is, or the
    # sentence is a citation to nothing and re-opens the argument it settles.
    promote = (ROOT / "scripts" / "automod" / "promote.py").read_text().splitlines()
    assert "_restart_primary" in promote[1879], (
        "scripts/automod/promote.py:1880 does not name _restart_primary, so trap "
        "#2's reason cites a line that has moved")
    assert "#1635" in bullet, (
        "positive control: the retirement names no item, so a later reader has "
        "nothing to check the ruling against and will re-file the run")


def test_trap_two_keeps_the_eviction_asymmetry_that_justifies_the_margin():
    """Clause 4: the direction of the bias is what lets PASSES = 5 stand with no
    idle measurement, so it is load-bearing prose, not a caveat.

    Delete the direction and the loaded run stops bounding the idle case, which
    puts the retired boot back on the critical path -- the exact state #1635's
    ruling closed. Checked in one sentence because two sentences holding "load"
    and "PASSES = 5" separately pin no reasoning.
    """
    bullet = _flat(_trap_two())
    asym = [s for s in _sentences(bullet) if "can only add" in s.lower()]
    assert asym, (
        "nothing says load can only ADD cold passes, so the loaded run no longer "
        "bounds the idle case and the retired measurement is needed again")
    assert any("bounds the idle case at one cold pass" in s for s in asym), (
        "the asymmetry is stated without its consequence: one cold pass is a bound "
        "on the idle case, not a second measurement of it")
    assert any("eviction" in s and "load-dependent" in s
               and "not a property of this build" in s for s in asym), (
        "the cold-pass count is no longer called eviction- and load-dependent "
        "rather than a property of this build")
    assert any("PASSES = 5" in s and "margin" in s for s in _sentences(bullet)), (
        "PASSES = 5 no longer states its reason: a safety margin around that bound")


def test_the_launcher_attributes_its_two_pass_figure_to_the_boot_measured_on():
    """Clause 5: start-qwen38-flash-next.sh carried the same two constants as an
    undated rule, and the bench docstring sends a reader there for the table.

    The table is genuinely that boot's and stays; what goes is the rule shape. So
    the check is scoped to the block's warm-up sentences -- the same block carries
    "96.0% reuse vs 99.3%", which is a real 2026-09-06 measurement and must not
    be deleted to satisfy a grep.
    """
    block = _flat(_launcher_mtp_comment())
    warmup = [s for s in _sentences(block) if "warm-up" in s.lower()]
    assert warmup, (
        "the MTP block no longer mentions the warm-up at all, so what follows is "
        "judging a claim that is not in the file")
    for sentence in warmup:
        for constant in RETIRED_CONSTANTS:
            assert constant not in sentence, (
                f"the launcher's warm-up sentence still states the retired "
                f"2026-09-06 constant {constant!r}, which the bench then hands to "
                f"a reader as this block's result table")
    assert any("TWO warm-up passes" in s and "2026-09-06" in s and "qwen4_exp" in s
               for s in warmup), (
        "the two-warm-up-pass claim is not attributed to the boot it was measured "
        "on, so it still reads as a property of every boot")
    # Positive control: the figures that boot really produced are still there, so
    # the absence checks above are a re-attribution, not a deleted table.
    assert "96.0% reuse vs 99.3%" in block, (
        "positive control failed: the MTP block lost the A/B reuse figures that "
        "WERE measured, so nothing here is pinning a real correction")


def test_the_bench_script_keeps_its_exec_bit():
    """Clause 3: `fb962366` landed this script at 100644 and the whole suite was
    green -- `git show --raw fb962366` records `:100755 100644` for the path.

    The round's prepared tag held the bit restored via `git update-index
    --chmod=+x` and the squash-landing dropped that as well; no `prepared-*` tag
    survives in this checkout (`git tag -l | grep -c prepared` is 0), so the one
    artifact a later reader can actually run is `git show --raw fb962366`.
    Nothing else in the suite can see the bit, for the reason `test_llm_slot_vram_preflight.py::test_the_launchers_keep_their_exec_bits`
    gives for pinning the launchers: this file's Usage header runs it through
    `.venvs/lloyd/bin/python`, so every doc-claim test in this file stays green at
    100644 while the direct-exec form its own shebang advertises -- and the form
    any bench run is typed as -- fails with EACCES before pass 1.
    Working file and committed tree are both checked, because the regression was
    committed rather than local.
    """
    mode = BENCH.stat().st_mode
    assert mode & statmod.S_IXUSR, f"{BENCH.name} lost its exec bit (mode {oct(mode)})"
    assert os.access(BENCH, os.X_OK), f"{BENCH.name} is not executable for this user"
    rel = BENCH.relative_to(ROOT).as_posix()
    tree = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "HEAD", rel],
                          capture_output=True, text=True).stdout.strip()
    assert tree, f"git ls-tree HEAD named no entry for {rel}: it is not tracked"
    assert tree.startswith("100755 blob "), (
        f"HEAD holds {rel} at {tree.split()[0]}, not 100755 -- the mode bit the "
        "fb962366 landing dropped and no rung could see")


# ── clause 2: the caveat sits at the _RATIOS mapping ───────────────────

def test_vllm_metrics_qualifies_the_ratio_at_the_mapping_it_conditions():
    comment = _ratios_comment()
    assert "KV cache group" in comment, "the amplification condition is missing"
    assert "eagle" in comment.lower(), "the boot layout is not named"
    for figure in ("673,179", "11.2x"):
        assert figure in comment, f"the mapping lost its magnitude: {figure}"
    assert "1:1" in comment, "the parity condition is missing"
    assert "prompt_tokens_total" in comment, "parity stated without its term"
    assert "prefix_cache_queries_total" in comment, (
        "the caveat does not name the series it is about")


def test_the_caveat_is_reachable_only_from_the_ratios_it_explains():
    """Guards the grep-pinning itself: the comment must be the block touching
    `_RATIOS`, not the same paragraph parked somewhere else in the module."""
    lines = METRICS.read_text().splitlines()
    site = next(i for i, line in enumerate(lines) if line.startswith("_RATIOS = ("))
    assert lines[site - 1].lstrip().startswith("#"), (
        "no comment attaches to _RATIOS; the caveat moved off its site")


# ── clause 3: a bench run's pass table lives in the item ─────────────────
#
# The header below is the string the bench prints (`bench-prefix-reuse.py:106`,
# asserted against the source so a reformatted table invalidates the quoted
# evidence rather than leaving it silently unverifiable). Clause 3 is the only
# clause here whose subject is the backlog rather than this checkout, so it reads
# the item file through `board_presence.board_files_or_stop`, which -- per that
# module's own policy, and the two review findings on #1204 recorded there --
# fails rather than skips when the board is absent. A vault-less box therefore
# gets a red line naming the unpinned clause, never a green one that read
# nothing.

TABLE_HEADER = "  pass  wall     cached_tokens   d_queries   d_hits   hit%"
PASS_ROW = re.compile(
    r"\s+(\d)\s+\d+\.\d+s\s+\d[\d,]*\s+\d[\d,]*\s+\d[\d,]*\s+\d+\.\d+%")


def _quoted_table_rows(body: str) -> list[str]:
    """The pass rows of the table quoted under the bench's own header. Anchored
    on the header rather than counting every pass-shaped line in the file, so a
    second table quoted by a later run -- a re-run on some future boot, which
    #1635 retired as an obligation but not as a possibility -- cannot turn this
    red by adding rows."""
    lines = body.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if TABLE_HEADER in ln)
    except StopIteration:
        return []
    rows = []
    for ln in lines[start + 1:]:
        if PASS_ROW.fullmatch(ln):
            rows.append(ln)
        elif rows:
            break
    return rows


def test_the_table_this_test_checks_is_the_table_the_bench_prints():
    """Positive control for the clause-3 check below: it greps for a header, so
    first prove that header is the one the script emits. A drifted print makes a
    quoted table unverifiable, and a matcher that matched nothing would have
    looked like 'the item lost its table' instead of 'the script changed'."""
    lines = BENCH.read_text().splitlines()
    assert any(ln.strip() == f'print("{TABLE_HEADER}")' for ln in lines), (
        "the bench no longer prints the header clause 3 quotes by")


def test_item_605_quotes_a_five_pass_bench_table():
    """Clause 3: the pass table is in the item file, so the parity claim is
    checkable by a later reader who has no engine in front of them. Five rows
    numbered 1-5, because `PASSES = 5` is what the clause requires the run to
    keep -- and the row numbers are what make it a table and not five stray
    digits."""
    import board_presence
    files = board_presence.board_files_or_stop(what="#605's own backlog board")
    items = [p for p in files if p.name.startswith("605-")]
    assert items, "the board holds no item whose name starts with 605-"
    body = items[0].read_text(encoding="utf-8", errors="replace")
    rows = _quoted_table_rows(body)
    assert rows, (
        f"{items[0].name} no longer quotes a bench pass table under the header "
        f"the bench itself prints")
    numbers = [PASS_ROW.fullmatch(ln).group(1) for ln in rows]
    assert numbers == ["1", "2", "3", "4", "5"], (
        f"expected the 5 sequentially numbered pass rows of a PASSES=5 run in "
        f"{items[0].name}, got {numbers}")
