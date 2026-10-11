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
