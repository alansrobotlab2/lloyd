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
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "agent-services" / "bin" / "bench-prefix-reuse.py"
METRICS = ROOT / "app" / "vllm_metrics.py"

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
    second table quoted by a later run -- a re-run on an idle boot, which #605
    still owes -- cannot turn this red by adding rows."""
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
