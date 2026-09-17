"""#1204 clause 5: the dashboard's board-size and latency comments are claims,
not decor.

Two comments in `app/routers/dashboard.py` described a board that no longer
exists. One said the backlog is "300+ markdown files" against a real 1,142; the
other said `board_health` costs "~2 s on the live board" against a measured
4–9 s. Both were true when written. A future session inherits them as facts —
that is the sentence in the item, and it is the reason the fix's diff has to
carry the comment change rather than a follow-up.

The literal strings are only the half that could be grepped for. The durable
half is the rule, applied to **comment blocks only**: a comment that asserts a
count of the board/ledger with a unit, or a duration, must also carry the
evidence for it — an ISO date it was taken at, the word `measured`, or a
backticked command at least three words long that re-measures it. Two filters
keep the rule on the board rather than over the whole file: docstrings are out
(the file's narrative prose, including `board_health`'s own re-measured claim
in `scripts/automod/backlog.py`, is graded where it lives), and the block must
name the backlog, the board or the ledger — which is why the dated size claims
at `:184`, `:200` and `:269`, all about the sessions directory, are not
flagged. A comment block that says the board is some size with no date and no
command is exactly the artifact this file exists to catch at write time.

The positive controls matter here more than anywhere: a `grep -c` that returns
0 because the pattern is malformed against the corpus is the false-zero the box
already has a catalogued habit of, so each assertion is paired with a count of
a string known to be present, and the file's line count is printed so an empty
file cannot pass as a clean one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DASHBOARD = Path(__file__).resolve().parent.parent / "app" / "routers" / "dashboard.py"
FRONTEND = Path(__file__).resolve().parent.parent / "web" / "src" / "components" / "pages" / "DashboardPage.tsx"

# The two stale claims by their exact wording, so the assertion cannot drift
# into "some comment that looks stale". Denominator at triage (2026-09-16):
# 1 hit each, at :56 and ~:672.
STALE = ("300+", "~2 s on the live board")

# A claim about *quantity of the thing the dashboard walks*, stated with a
# unit; or any stated duration. `\d+\+? +markdown|files|items|rows|bytes` and
# `~?\d+(\.\d+)? (s|ms|seconds)`.
QUANTITY = re.compile(r"\d[\d,]*\+?\s+(markdown|files|items|rows|front matters|bytes)", re.I)
DURATION = re.compile(r"~?\s*\d[\d,]*(?:\.\d+)?\s*(s|ms|seconds)\b", re.I)
SUBJECT = re.compile(r"\b(backlog|board|ledger)\b", re.I)

# What makes a stated number auditable later. A backticked span counts only if
# it has two spaces in it, i.e. it is a command and not an identifier: bare
# `` `board_health` `` and `` `up_next` `` are names, not evidence, and an
# earlier draft of this rule let the ~2 s comment through on them.
EVIDENCE = re.compile(
    r"(\bmeasured\b|\bre-?measure|20\d\d-\d\d-\d\d|`[^`\n]*\s[^`\n]*\s[^`\n]*`)", re.I)


def _comment_blocks(text: str) -> list[tuple[int, str]]:
    """Contiguous `#` comment blocks, as (first line number, joined text)."""
    out, block = [], []
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            block.append((lineno, line))
        else:
            if block:
                out.append((block[0][0], " ".join(l for _, l in block)))
                block = []
    if block:
        out.append((block[0][0], " ".join(l for _, l in block)))
    return out


@pytest.fixture
def source() -> str:
    assert DASHBOARD.is_file(), f"{DASHBOARD} is not a file"
    return DASHBOARD.read_text(encoding="utf-8")


def test_the_two_stale_board_claims_are_gone(source: str):
    """Clause 5, literal: `grep -n "300+\\|~2 s on the live board"` → 0 hits."""
    lines = source.splitlines()
    print(f"{DASHBOARD} lines: {len(lines)}")
    assert len(lines) > 100, (
        f"{DASHBOARD} is {len(lines)} lines; a truncated file has no stale "
        f"comments either, and must not be read as a fixed one"
    )
    # Positive control: a string that must survive, so a 0 for the stale
    # patterns cannot be a read-the-wrong-file artifact.
    assert "_VAULT_SCAN_TTL_S" in source and "_SCORECARD_TTL_S" in source, (
        "the TTL constants are absent, so this is not the dashboard router "
        "the stale comments lived in"
    )

    offenders = []
    for needle in STALE:
        hits = [n for n, line in enumerate(lines, 1) if needle in line]
        print(f"{needle!r}: {len(hits)} hit(s)")
        if hits:
            offenders.append(f"{needle!r} still at line(s) {hits}")
    assert not offenders, (
        "the stale board claims are still in the file: " + "; ".join(offenders) +
        ". Rule already in force on this box: a 'currently ~N seconds' comment "
        "is a measurement with a decay date — state the board size it was "
        "taken at, or name the command that re-measures it."
    )


def test_every_board_or_ledger_quantity_claim_in_a_comment_carries_evidence(source: str):
    """The structural half: no comment may assert a board size or a latency bare.

    A comment block that mentions the backlog/board/ledger *and* states a
    counted quantity or a duration must also state when it was measured, or how
    to measure it. Triage denominator: 2 blocks flagged, both the two stale
    ones above; the replacement comments carry `pytest tests/
    test_dashboard_cold_render.py` and a dated count, so this file has a
    positive control of its own — it is not merely the negation of one grep.
    """
    blocks = _comment_blocks(source)
    print(f"comment blocks: {len(blocks)}")
    assert blocks, f"{DASHBOARD} has no comment blocks — nothing to grade"

    bare = []
    flagged = 0
    for lineno, text in blocks:
        if not SUBJECT.search(text):
            continue
        if not (QUANTITY.search(text) or DURATION.search(text)):
            continue
        flagged += 1
        if not EVIDENCE.search(text):
            bare.append(f":{lineno}: {text[:110]}")
    print(f"quantity/latency comment blocks about the board: {flagged}")
    assert not bare, (
        f"{len(bare)} comment block(s) assert a board/ledger size or a latency "
        f"with no date and no re-measuring command:\n" + "\n".join(bare) +
        "\nEither name the size the figure was taken at with its date, or name "
        "the command that re-measures it."
    )
    # And the rule must actually be exercised: at least one board-shaped
    # quantity claim exists in the file and passes.
    assert flagged >= 1, (
        "no comment block in the file mentions a board size or a latency, so "
        "the rule above asserted over an empty set. A denominator of zero is "
        "not a passing check."
    )


def test_the_ttl_comments_name_both_numbers_they_reconcile(source: str):
    """Recommendation E: the TTL against the poll rate has to be a decision.

    #1204 section 5 is that `_VAULT_SCAN_TTL_S = 10.0` against the frontend's
    2-second poll is the reason a hitch fires every fifth request. The fix is
    not to pick a magic number, it is to write the pair down; this test makes
    the pair unreadable-by-accident by requiring both constants' values in the
    comment above each TTL assignment.
    """
    assert FRONTEND.is_file(), f"{FRONTEND} is not a file"
    fe = FRONTEND.read_text(encoding="utf-8")
    poll = re.search(r"POLL_MS\s*=\s*(\d+)", fe)
    assert poll, "POLL_MS is gone from DashboardPage.tsx, so the pairing has no anchor"
    poll_s = int(poll.group(1)) / 1000
    print(f"frontend POLL_MS: {poll.group(1)} ({poll_s:g} s)")

    lines = source.splitlines()
    for const in ("_VAULT_SCAN_TTL_S", "_SCORECARD_TTL_S"):
        m = re.search(rf"^{const}\s*=\s*([\d.]+)", source, re.M)
        assert m, f"{const} is no longer assigned in {DASHBOARD}"
        ttl = float(m.group(1))
        # The comment block immediately above the assignment.
        assign_line = source[:m.start()].count("\n") + 1
        block = []
        n = assign_line - 1
        while n >= 1 and lines[n - 1].lstrip().startswith("#"):
            block.insert(0, lines[n - 1])
            n -= 1
        joined = " ".join(block)
        print(f"{const} = {ttl}: {len(block)} comment lines above it")
        assert joined, f"{const} has no comment above it naming the pairing"
        assert str(int(ttl)) in joined or str(ttl) in joined, (
            f"the comment above {const} = {ttl} does not state its own value")
        assert re.search(rf"\b{poll_s:g}[\s-]?(s|second)", joined, re.I) or f"{poll.group(1)}" in joined, (
            f"the comment above {const} = {ttl} does not name the frontend poll "
            f"interval ({poll.group(1)} ms = {poll_s:g} s in DashboardPage.tsx). "
            f"The 10 s TTL against a 2 s poll is what made a hitch fire every "
            f"fifth request; a TTL that outlives the poll by an unstated "
            f"multiple is the accident #1204 section 5 describes.")
