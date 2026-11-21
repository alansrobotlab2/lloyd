"""Guards on eval/lloyd_profile.md — the standing design brief for Lloyd's harness.

The youtube-digest session reads this profile before deciding whether a video
holds an idea worth filing, and nothing at read time can override a number in
it. It said "~275" skills for weeks while the loader saw 189 (#807), and three
backlog items inherited the figure. Three live counts exist and differ — loaded
by `_iter_skills`, top-level dirs, `SKILL.md` files on disk — so the sentence
must name which one it states, and the guard recomputes the loaded count rather
than comparing against a literal: nightly jobs add and archive skills, and an
equality test would be a permanent flake while a band still catches a 1.45x
drift.

Standing problem 7 is guarded here for the same reason in the other direction. It
claimed "speculative decoding not in use" while the primary's own boot passed
`--speculative-config` and `/metrics` published both spec-decode counters
non-zero (#2565's triage read the pair at 0.584 accepted/drafted). A brief that
lists an absent optimization invites a round to build the thing that is already
running, so this file pins what the bullet may and may not say about it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent_mcp.skills import _iter_skills

ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / "eval" / "lloyd_profile.md"

#: How far the stated count may sit from the live one before the sentence is
#: stale. Ordinary churn is a few skills; the drift this caught was 1.45x.
TOLERANCE = 0.15

#: The two engine counters speculative-decoding acceptance is derived from, and the
#: endpoint they are read from. `tests/test_bench_concurrency_sweep.py` reads the same
#: pair for the sweep's artifact header; a drift between the two readers is how the
#: bullet went stale in the first place, so the names live here once.
SPEC_ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
SPEC_DRAFTED = "vllm:spec_decode_num_draft_tokens_total"


def _skills_sentence() -> str:
    """The vault bullet's parenthetical about skills, flattened to one line."""
    text = " ".join(PROFILE.read_text().split())
    m = re.search(r"skills \(`SKILL\.md` per skill;(.*?)\), knowledge notes", text)
    assert m, "the vault bullet no longer carries its skills parenthetical"
    return m.group(1)


def _stated_loaded_count(sentence: str) -> int:
    m = re.search(r"(\d+) loaded", sentence)
    assert m, f"no '<n> loaded' count in the skills sentence: {sentence!r}"
    return int(m.group(1))


def _standing_problem(n: int) -> str:
    """The nth numbered standing problem, continuation lines included."""
    section = (PROFILE.read_text(encoding="utf-8", errors="replace")
               .split("## Standing problems worth solving", 1)[1]
               .split("\n## ", 1)[0])
    for block in re.split(r"(?m)^(?=\d+\.\s)", section):
        if block.startswith(f"{n}. "):
            return block.rstrip()
    raise AssertionError(f"no numbered standing problem {n} in the section")


def _counter_sum(body: str, name: str) -> float:
    """Sum one Prometheus counter across its label sets, ignoring `_created`/`_sum`."""
    total = 0.0
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        base = head.split("{", 1)[0]
        if base != name:
            continue
        try:
            total += float(value)
        except ValueError:
            continue
    return total


def _engine_metrics() -> str | None:
    """The primary's `/metrics` body, or None when it cannot be reached.

    Read through the engine registry rather than a literal host, so this node asks the
    same question `app/vllm_metrics.py` asks and cannot point at a port that moved.
    """
    import urllib.request

    from app.vllm_metrics import configured_engines
    base = (configured_engines().get("primary") or "").rstrip("/")
    if not base:
        return None
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=5) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


def test_no_275_skill_count_survives():
    assert "275" not in PROFILE.read_text()


def test_sentence_names_its_derivation_and_separates_the_disk_counts():
    sentence = _skills_sentence()
    assert "_iter_skills" in sentence, "the loaded count must name how it was derived"
    assert ".archived/" in sentence and "on disk" in sentence, (
        "the on-disk / .archived/ figure must be stated apart from the loaded one"
    )
    _stated_loaded_count(sentence)
    assert re.search(r"\d+ top-level skill dirs", sentence) and re.search(
        r"\d+ `SKILL\.md` files", sentence
    ), "the on-disk counts must be stated as numbers of their own"


@pytest.mark.live_vault
def test_stated_count_is_within_band_of_the_live_loader():
    live = len(list(_iter_skills()))
    if live == 0:
        pytest.skip("no skills visible to the loader (vault not present)")
    stated = _stated_loaded_count(_skills_sentence())
    assert abs(stated - live) <= TOLERANCE * live, (
        f"eval/lloyd_profile.md says {stated} skills loaded; _iter_skills() "
        f"returns {live} — refresh the sentence"
    )


@pytest.mark.parametrize("stated,live,ok", [(187, 189, True), (275, 189, False),
                                             (160, 189, False)])
def test_the_band_is_a_band_not_an_equality(stated, live, ok):
    assert (abs(stated - live) <= TOLERANCE * live) is ok


# ── #2565 clause 5: standing problem 7 and the spec-decode counters ────────────

def test_standing_problem_7_no_longer_claims_speculative_decoding_is_unused():
    """Clause 5, static half: the false sentence is gone and the derivation is in it.

    The bullet read `speculative decoding not in use`. The primary's boot passes
    `--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'`
    (`agent-services/bin/start-qwen38-flash-next.sh`), so the line was not a stale
    measurement but a claim about a flag that was set. What replaces it names the two
    counters its own value is derived from, because the list has a standing rule against
    restating a rate in prose — `test_no_standing_problem_carries_a_bare_decimal_figure`
    in `tests/test_profile_doc_claims.py` — and a named derivation is re-readable while a
    number is only re-memberable.
    """
    text = PROFILE.read_text(encoding="utf-8", errors="replace")
    bullet = _standing_problem(7)

    assert "not in use" not in text, (
        "the retired claim is back somewhere in the brief")
    assert "speculative decoding" in bullet.lower(), bullet
    assert SPEC_ACCEPTED in bullet, f"problem 7 must name its derivation: {bullet}"
    assert SPEC_DRAFTED in bullet, f"problem 7 must name its derivation: {bullet}"
    assert re.search(r"\bis in use\b|\bin use\b", bullet), (
        f"problem 7 must say the decoder is running, not merely cite counters: {bullet}")
    # The rule the rest of the list already enforces, checked on this bullet too: no
    # rate or count written into the brief, since nothing re-reads it. `24 GB` and the
    # MTP token count are configuration this document owns; a measured ratio is not.
    assert not re.search(r"\b0?\.\d+\b", bullet), (
        f"problem 7 restates a measured rate: {bullet}")


def test_standing_problem_7s_derivation_recomputes_on_the_live_engine():
    """Clause 5, live half: the pair the bullet points at resolves to a real rate.

    The claim "speculative decoding is in use" is checkable two ways — the flag on argv,
    or the counters moving — and only the second says it is IN USE rather than merely
    configured. So this node reads the two counters the sentence names and asserts
    drafting and acceptance are both non-zero and acceptance is a rate. It asserts
    nothing about the VALUE: it is a boot-summed ratio that drifts with the mix of work,
    which is exactly why the bullet states the derivation instead of a figure. Skips when
    the primary is unreachable, the way `tests/test_prefix_cache_doc_claims.py` does.
    """
    body = _engine_metrics()
    if body is None:
        pytest.skip("primary's /metrics is not reachable; the derivation is pinned "
                    "statically by the node above")
    accepted = _counter_sum(body, SPEC_ACCEPTED)
    drafted = _counter_sum(body, SPEC_DRAFTED)
    assert drafted > 0, f"{drafted=}: nothing drafted, so 'in use' would be false"
    assert accepted > 0, f"{accepted=}: nothing accepted, so 'in use' would be false"
    assert 0.0 < accepted / drafted <= 1.0, f"{accepted=} {drafted=}"


# ── #4309: the Hardware section describes the box the work is priced against ──
#
# `2db8a25f` (2026-09-08) wrote `- Two 24 GB RTX 3090-class GPUs. Anything that
# needs more than ~22 GB per GPU, or a cloud API, is not adoptable as-is.` and
# `2e08e6d3` ("moar gpus", 2026-10-07) took the box to five cards without ever
# reaching this file — `git log -S'Two 24 GB' -- eval/lloyd_profile.md` named that
# one commit. Nothing at read time can override it: the youtube-digest session
# prices every idea it reads against the machine this file describes, so a two-card
# box means it declines anything that wants more than one card and never asks
# whether the box has one. That is the whole cost of the line, and it is why these
# nodes read the profile against the in-tree roster (`architecture/infrastructure.md`
# § GPUs, the declared owner, and `architecture/index.md` §GPU allocation for the
# index set) and never against the live box — the rule
# `tests/test_infrastructure_gpu_inventory.py` keeps for the same reason.

ARCH = ROOT / "architecture"
INFRA = ARCH / "infrastructure.md"
INDEX = ARCH / "index.md"
DJEV = ARCH / "djev.md"

#: The box's cards, cross-checked against the two in-tree copies of the roster. The
#: literal is what makes a dropped card fail rather than merely disagree, and it is
#: the same constant `tests/test_infrastructure_gpu_inventory.py` carries.
CARD_COUNT = 5

_GPU_ROW = re.compile(r"^\|\s*(\d+)\s*\|(.*)$")

#: An occupancy stated the way this file requires: a MiB figure and the command and
#: date behind it. `95106 of its 97887 MiB (measured 2026-11-21, nvidia-smi …)`.
_OCCUPANCY = re.compile(r"(\d{3,6})\s+of\s+(?:its\s+)?(\d{3,6})\s+MiB", re.IGNORECASE)
_MEASURED_DATE = re.compile(r"measured\s+(\d{4}-\d{2}-\d{2})")


def _section_body(text: str, heading: str) -> str:
    """The body of `heading`, up to the next `## ` heading. Non-empty by assertion."""
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == heading)
    except StopIteration:
        raise AssertionError(f"{heading} is not a heading any more") from None
    rest = lines[start + 1:]
    end = next((i for i, ln in enumerate(rest) if ln.startswith("## ")), len(rest))
    body = "\n".join(rest[:end])
    assert body.strip(), f"{heading} has no body"
    return body


def _hardware_section(text: str | None = None) -> str:
    return _section_body(text if text is not None else PROFILE.read_text(encoding="utf-8", errors="replace"),
                         "## Hardware and models")


def _gpu_roster() -> dict[int, tuple[str, str]]:
    """`architecture/infrastructure.md` § GPUs as `{index: (card model, VRAM)}`.

    The declared owner of the card-by-card map, read with the row regex
    `tests/test_infrastructure_gpu_inventory.py` uses for the same table, so a
    change to the table's spelling breaks both files instead of quietly dropping
    one. Asserts it extracted a roster: a green from a regex that matched nothing
    is how a doc-claim test goes decorative.
    """
    rows: dict[int, tuple[str, str]] = {}
    for ln in _section_body(INFRA.read_text(encoding="utf-8"), "## GPUs").splitlines():
        m = _GPU_ROW.match(ln)
        if not m:
            continue
        idx = int(m.group(1))
        cells = [c.strip() for c in m.group(2).split("|")]
        assert idx not in rows, f"§ GPUs has two rows for GPU {idx}"
        rows[idx] = (cells[0], cells[1])
    assert len(rows) >= CARD_COUNT, (
        f"§ GPUs yielded {sorted(rows)} — the extraction is broken, not the profile")
    return rows


def _index_allocation_indices() -> list[int]:
    """The indices `architecture/index.md` §GPU allocation lists: the glance map
    whose row set clause 2 is written against."""
    rows = {int(m.group(1)) for ln
            in _section_body(INDEX.read_text(encoding="utf-8"), "## GPU allocation").splitlines()
            if (m := _GPU_ROW.match(ln))}
    assert len(rows) >= CARD_COUNT, f"index.md's GPU table only lists {sorted(rows)}"
    return sorted(rows)


def _dated_occupancy(section: str) -> tuple[re.Match, str | None, str]:
    """The occupancy claim: its match, its `measured <date>` or None, its sentence.

    The sentence is a ±200-character window around the figure, which is how the
    node checks that the date and the refreshing command travel WITH the number
    rather than merely existing elsewhere in the section.
    """
    m = _OCCUPANCY.search(section)
    if not m:
        return None, None, ""
    window = " ".join(section[max(0, m.start() - 200):m.end() + 200].split())
    d = _MEASURED_DATE.search(window)
    return m, (d.group(1) if d else None), window


def test_the_two_card_line_and_its_per_gpu_ceiling_are_gone():
    """Clause 1: the retired sentence is out, and the roster replaced it.

    Both halves of the old bullet are gone — `Two 24 GB` and the `~22 GB` per-GPU
    adoptability bound. The bound was a fact about one 3090 presented as a fact
    about the box: on a five-card box with a 96 GB card the ceiling it set was
    wrong in both directions, telling a reader there was no room and no capacity.
    The retirement had to be surgical, so the node also asserts `24 GB` is still
    said there four times over: the string `Two 24 GB` going missing because every
    mention of a 3090 went with it would pass the absence check and fail the item.
    """
    text = PROFILE.read_text(encoding="utf-8", errors="replace")
    assert "Two 24 GB" not in text, "the two-card claim is back"
    assert "~22 GB" not in text, "the per-GPU adoptability ceiling is back"
    section = _hardware_section(text)
    assert section.count("24 GB") >= CARD_COUNT - 1, (
        f"the Hardware section says `24 GB` {section.count('24 GB')} time(s); "
        f"the four RTX 3090s must each still name their size — this is a roster "
        f"replacement, not a deletion")


def test_the_hardware_section_carries_one_entry_per_card_with_model_and_vram():
    """Clause 2: every index the roster owns is in the brief with model and VRAM.

    The index set comes from `architecture/index.md` §GPU allocation — the copy
    clause 2 names — and each card's model and VRAM from
    `architecture/infrastructure.md` § GPUs, its declared owner. The profile is
    checked against both, not against a transcription in this test file, so the
    next card added to § GPUs and not to the brief is what goes red.
    """
    roster = _gpu_roster()
    indices = _index_allocation_indices()
    assert indices == list(range(CARD_COUNT)), (
        f"index.md §GPU allocation lists {indices}, not 0-{CARD_COUNT - 1}")
    section = " ".join(_hardware_section().split())
    unpriced = []
    for idx in indices:
        model, vram = roster[idx]
        # The 120 characters that follow each mention of the index: an entry counts
        # when the card's model and its size are said there, so a bare "GPU 3" in
        # prose is not accepted as the card being listed.
        ok = any(model in window and vram in window
                 for window in (section[m.end():m.end() + 120]
                                for m in re.finditer(rf"GPU {idx}\b", section)))
        if not ok:
            unpriced.append(f"GPU {idx}: § GPUs says {model!r} {vram!r}")
    assert not unpriced, (
        "the Hardware section does not price every card the roster owns: "
        + "; ".join(unpriced))


def test_the_blackwell_is_named_as_the_primarys_card_with_a_dated_occupancy():
    """Clause 3: the primary's card is named, and its figure carries its date.

    The file's own header rule is that a figure names the call its live value
    comes from, because a number that names no source "is a hint, not a
    measurement" and is re-membered rather than re-read. So the node pins the
    SHAPE — a MiB figure, a `measured YYYY-MM-DD`, and the `nvidia-smi` query that
    refreshes it, all in the window around the figure — and deliberately not the
    value: occupancy of the card running the primary moves with the workload, and
    pinning `95106` here would be a permanent flake whose only other product is a
    number nobody re-reads. What the date buys is exactly what clause 3 asks: a
    reader can tell a fresh measurement from a carried one.
    """
    section = _hardware_section()
    flat = " ".join(section.split())
    assert "RTX PRO 6000 Blackwell" in flat, (
        "the Hardware section names no Blackwell card")
    m, date, window = _dated_occupancy(section)
    assert m, "the Hardware section states no occupancy figure like `N of M MiB`"
    assert re.search(r"[Pp]rimary", window), (
        f"the occupancy figure is not attached to the primary: {window!r}")
    assert "Blackwell" in window, f"the occupancy is not the Blackwell's: {window!r}"
    assert date, f"the occupancy figure carries no `measured YYYY-MM-DD`: {window!r}"
    assert "nvidia-smi" in window, (
        f"the occupancy figure names no command to refresh it: {window!r}")


def test_an_undated_occupancy_figure_would_not_pass():
    """Clause 3's positive control: the date requirement bites.

    Without this the node above could be satisfied by a window that happens to
    contain a date somewhere. The planted sentence is the same shape with the
    `measured …` clause and the command stripped, which is the exact form of rot
    the header rule exists to catch.
    """
    section = _hardware_section()
    m, date, window = _dated_occupancy(section)
    assert m and date and "nvidia-smi" in window, (
        "the real section already fails the shape, so this node controls nothing")
    stripped = re.sub(r"\s*\([^()]*measured[^()]*\)", "", section, count=1)
    assert stripped != section, (
        "the date and command do not travel in one bracketed clause, so the "
        "control plants nothing and the shape assertion is ungraded")
    m2, date2, window2 = _dated_occupancy(stripped)
    assert m2, "the control removed the figure as well, so it proves nothing"
    assert not (date2 and "nvidia-smi" in window2), (
        "an occupancy with neither date nor refreshing command still passes, so the "
        "date is being read from somewhere else in the section")


def test_the_fp4_claim_names_the_blackwell_native_path_and_the_marlin_precedent():
    """Clause 4: only NATIVE FP4 is the Blackwell's, and the 3090 precedent stands.

    The item's headline wording — "the box's only FP4-capable card" — would have
    pinned a falsehood. `architecture/djev.md` §"Why an SM86 card can serve NVFP4
    at all" records DiffusionGemma's NVFP4 running on the GPU 2 3090 through vLLM's
    NVFP4 **Marlin W4A16** weight-only path, which gates at capability 75 with this
    checkpoint's group size of 16, and its decider table says the checkpoint "runs
    on the 3090 as | weight-only NVFP4 Marlin". Say "only FP4-capable" and the brief
    talks readers out of the one FP4 path measurable without displacing the primary,
    which is the opposite of what noting the precedent is for. The Blackwell-only
    thing is the CUTLASS/FlashInfer native path, and this node cross-checks the
    precedent against djev.md rather than against memory.
    """
    section = _hardware_section()
    low = " ".join(section.split()).lower()
    assert "blackwell" in low
    assert re.search(r"only[^\n]{0,120}native fp4", low), (
        "the section must call the Blackwell the only card with NATIVE FP4")
    assert re.search(r"committed to the primary|not a spare card", low), (
        "the section must say the Blackwell has no free card-role for a new idea")
    assert "marlin" in low and "w4a16" in low, (
        "the 3090 weight-only NVFP4 precedent must be named by its kernel")
    assert "architecture/djev.md" in section, (
        "the precedent must cite the file that owns it")
    assert "only fp4" not in low, (
        "'only FP4-capable' is false: a 3090 serves weight-only NVFP4 via Marlin")
    djev = DJEV.read_text(encoding="utf-8", errors="replace")
    assert "Marlin W4A16" in djev, (
        "architecture/djev.md no longer documents the Marlin path; re-read the "
        "precedent before repeating it in the brief")


def test_only_the_hardware_bullet_moved_and_standing_problem_7_stands():
    """Clause 5: the rest of the brief is untouched — nine problems, one NVFP4/GGUF.

    `grep -c 'NVFP4/GGUF'` read 1 at triage and stays 1. Problem 7's "Inference
    efficiency on 24 GB" is a per-3090 budget, a different claim from the box
    roster, and the parent owed entry will not let the NVFP4/GGUF trade-off be
    retired until an FP8-vs-NVFP4 quality number exists — so this diff must leave
    the bullet exactly as it was. The nine is the same count
    `tests/test_profile_doc_claims.py::test_standing_problems_list_still_has_its_nine_items`
    pins from that file; it is re-pinned here because this round's diff is the one
    that edits the document, and the clause being graded is "only the Hardware
    bullet moved".
    """
    text = PROFILE.read_text(encoding="utf-8", errors="replace")
    assert text.count("NVFP4/GGUF") == 1, (
        f"{text.count('NVFP4/GGUF')} NVFP4/GGUF mentions; the Hardware rewrite "
        "must not add or remove one")
    section = text.split("## Standing problems worth solving", 1)[1].split("\n## ", 1)[0]
    numbered = re.findall(r"(?m)^\d+\.\s", section)
    assert len(numbered) == 9, f"expected nine standing problems, found {len(numbered)}"
    assert _standing_problem(7).startswith("7. Inference efficiency on 24 GB"), (
        "standing problem 7's own wording moved; this item leaves it alone")
