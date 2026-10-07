"""#1339 / #1627: `architecture/vllm.md` §10 is a standing acceptance test, so its
numbers have to follow the fleet and the data they were read from.

It went stale twice over: its bar was written for `workers.slots` 2 while the
fleet ran 6 slots and four autocode rounds, and its only counted reading was a
2026-09-11 spot check off a five-minute in-memory ring. Nothing could fail when
either moved. Since #1627 it has a third duty, which is the one that bites:

* the counted reading is a full UTC day **with real chat in it** — 2026-09-25,
  the first day the counter has that tests criterion (a) at all — and §10 marks
  (a), (b), (c) and (d) pass or fail beside it, so a mark that no longer
  follows its own numbers fails here;
* every figure in that reading, including (d)'s two-request throughput, is
  recomputed by `scripts/vllm_prefix_miss_window.derive` over the extract the
  doc names. Only the two history paragraphs — the 2026-09-11 spot reading and
  §6.1's own baseline — quote numbers the extract cannot cover, and they are
  labelled history in the text;
* the fleet shape §10 names is read from `config.yaml`, the gate from the same
  file, and the ring's retention from `app/engine_pressure`;
* and since #1720 it holds §10's accepted-loss ruling on the non-churn misses to
  account: the deferral sentence that got the question re-filed three times stays
  gone, and the two re-open triggers are pinned to the same derivations as the
  criteria — trigger one to `chat_turns_with_misses`, trigger two to §6.1's
  per-day budget — so neither figure can be re-cut by hand when it suits;
* and since #1921 it holds the accepted-loss ruling's own FILING condition to account:
  2026-09-29 is the first counted day on which a chat-turn miss had neither free-pool
  churn nor a gate-level KV gap, and its extract is the third in the tree — every figure
  in that paragraph, down to the gap's KV peak against the gate and its tokens against
  the free pool the peak implies, is recomputed over it here, with `EXTRACT` and
  `REOPEN_EXTRACT` left pointing at the days they already grade;
* and since #1810 it holds (a)'s two mints and the population ruling to account: the
  `iv` id prefix may never be bound to the Inner-Voice opt-in inside one clause (the
  mint that appends it reads no body first), owed-check's ruling that a day at the
  machine's own chat volume clears the population bar is on the record as CLOSED with
  its owner and its expiry, and the one read-only `usage.db` query that would re-open
  it is run here and checked against `session_kind` rather than admired;
* and since #1918 it holds the SHAPE of that ruling's grounding to account: the superlative
  it rests on names the window it is dated to instead of "the re-derivable record" (an
  undated superlative widens itself every time a turn is logged), its re-open trigger is the
  conjunction that can actually falsify it — a day at or above the counted volume on which a
  chat turn carries a miss — stated with no figure of its own, and pinned as wording only,
  so no day of traffic can redden it. Which day is busiest is still deliberately not pinned.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from app import engine_pressure
from scripts import vllm_prefix_miss_window as W

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "vllm.md"
# #1627: the day with chat in it. The 09-23 extract stays in the tree as the
# reading this one replaced (see test_the_replaced_reading_is_not_still_quoted).
EXTRACT = ROOT / "tests" / "fixtures" / "vllm_prefix_miss_2026-09-25.json"
OLD_EXTRACT = ROOT / "tests" / "fixtures" / "vllm_prefix_miss_2026-09-23.json"
CRITERIA = ("a", "b", "c", "d")


def _section(n: int) -> str:
    text = DOC.read_text(encoding="utf-8")
    m = re.search(rf"^## {n}\. .*?(?=^## )", text, re.M | re.S)
    assert m, f"§{n} not found"
    return " ".join(m.group(0).split())


def _counted_reading(s10: str) -> str:
    """Just the counted reading and the four criteria marked against it — not
    §10's bar, its history, or the open questions that carry no figures."""
    start = s10.index("The counted reading:")
    end = s10.index("YaRN's tool-choice effect")
    return s10[start:end]


@pytest.fixture(scope="module")
def s10() -> str:
    return _section(10)


@pytest.fixture(scope="module")
def reading(s10) -> str:
    return _counted_reading(s10)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def derived(cfg) -> dict:
    gate = float(cfg["workers"]["kv_gate"]["max_kv_usage"])
    return W.derive(json.loads(EXTRACT.read_text(encoding="utf-8")), gate=gate)


def _n(x: int) -> str:
    return f"{x:,}"


def _verdict(reading: str, letter: str) -> str:
    """Each criterion is marked exactly once in the counted reading, and which
    mark it is comes out of the test — so a figure edited by hand without the
    data moving, or a verdict edited without the mark agreeing, both fail."""
    hits = re.findall(rf"\({letter}\) (passes|fails)", reading)
    assert len(hits) == 1, f"({letter}) must carry exactly one pass/fail mark, found {hits}"
    return hits[0]


_NUMERALS = "zero one two three four five six seven eight nine ten eleven twelve".split()


def _word(n: int) -> str:
    return _NUMERALS[n] if 0 <= n < len(_NUMERALS) else str(n)


# ── clause 1: the fleet shape, and the day the reading is over ────────────

def test_the_bar_names_the_fleet_shape_config_runs(s10, cfg):
    slots = int(cfg["workers"]["slots"])
    rounds = int(cfg["workers"]["sources"]["autocode"]["max_inflight"])
    gate = float(cfg["workers"]["kv_gate"]["max_kv_usage"])
    assert f"`workers.slots` {slots}" in s10
    assert f"`workers.sources.autocode.max_inflight` {rounds}" in s10
    assert f"`workers.kv_gate.max_kv_usage`, {gate:.2f})" in s10
    assert f"slots {slots}, {'four' if rounds == 4 else rounds} rounds" in s10


def test_the_bar_states_d_as_a_measurable_threshold(s10):
    """(d) is a bar the page can be wrong about, so its own text has to name
    the figure and the exclusion rather than gesture at "slow"."""
    assert f"under {W.STALL_TOK_S:.0f} tok/s that is not a cold admission" in s10
    assert "prompt *and* generation" in s10


def test_the_counted_reading_is_a_named_full_utc_day_with_chat_in_it(s10, reading):
    m = re.search(r"The counted reading: (\d{4}-\d\d-\d\d) 00:00 → (\d{4}-\d\d-\d\d) 00:00 UTC", s10)
    assert m, "the window must be named in the text"
    assert m.group(1) >= "2026-09-25", \
        "#1627: the reading is owed over a full UTC day from 2026-09-25 onward"
    assert W._epoch(m.group(2)) - W._epoch(m.group(1)) == 86_400, "one full UTC day"
    ex = json.loads(EXTRACT.read_text(encoding="utf-8"))
    assert ex["window"] == [m.group(1), m.group(2)]
    assert EXTRACT.name in reading and "scripts/vllm_prefix_miss_window.py" in reading
    assert len([t for t in ex["turns"] if t[1] == "chat"]) >= 10, \
        "#1627 clause 1: the day must carry at least 10 chat turns to test (a)"


# ── clause 2: every counted figure is the derivation, and each mark follows ─

def test_the_counted_figures_are_the_derivation(reading, derived):
    d = derived
    assert (f"{d['turns']} turns, {d['measured']} measured, {d['turns_with_misses']} turns "
            f"carrying {d['misses']} misses and {_n(d['reprefill_tokens'])} re-prefilled tokens") in reading
    assert f"worst turn {_n(d['worst_turn_tokens'])}" in reading
    assert _n(d["pool_tokens"]) in reading, "the pool the free-pool arithmetic is against"
    kinds = ", ".join(f"{k} {misses} / {tokens / 1e6:.1f}M"
                      for k, (misses, tokens) in d["by_kind"].items())
    assert f"By session kind (misses / re-prefilled): {kinds}." in reading
    assert (f"{d['miss_events']} miss iterations logged, {d['cold_events']} of them fully cold "
            f"(nothing cached) and {d['miss_events'] - d['cold_events']} partial") in reading


def test_criterion_a_is_marked_and_is_the_chat_miss_count(reading, derived):
    d = derived
    assert f"{d['chat_turns_with_misses']} of the day's {d['chat_turns']} chat turns " \
           f"carries a prefix miss" in reading
    assert _verdict(reading, "a") == ("passes" if d["chat_turns_with_misses"] == 0 else "fails")


_BASELINE_RE = re.compile(
    r"Fleet baseline over \d\d-(\d\d)/(\d\d)[^:]*: (\d+) of [\d,]+ iterations .*?"
    r"— ([\d.]+)M tokens")


def _per_day_budget(s6: str) -> tuple[int, float]:
    """§10's (b) budget is §6.1's baseline divided by the two days it spans,
    counted here from §6.1's own sentence. The baseline itself is the one
    history paragraph on this page — the database it was read from was wiped on
    09-22 and cannot re-derive it — so the arithmetic is the half that can be
    pinned, and pinning it is what stops the two figures drifting apart."""
    m = _BASELINE_RE.search(s6)
    assert m, "§6.1 must state its baseline as N of M iterations over a date range"
    days = int(m.group(2)) - int(m.group(1)) + 1
    return round(int(m.group(3)) / days), round(float(m.group(4)) / days, 1)


# ── #1719: (a)'s population is named by its rule, and the bar has measured ──
# content. Before this, "(a) passes: 0 of the day's 23 chat turns" said nothing
# about what a chat turn is, and the bar was cleared by "one normal day at that
# shape with Alan chatting" — a phrase the owed-check had to re-litigate because
# nothing in the doc could be checked against it.

def _criterion_bullet(reading: str, letter: str) -> str:
    """The bullet that marks one criterion, whole.

    `s10` arrives whitespace-FLATTENED by `_section`, so bullets are delimited by
    the ` - **` that opens the next one, not by newlines. Scoped to the bullet
    rather than to §10 because the point of these nodes is WHERE the doc says it:
    a mint citation parked anywhere else on the page would still leave a reader of
    (a) unable to tell which 23 turns were counted.
    """
    start = reading.find(f"- **({letter})")
    assert start >= 0, f"no '({letter})' bullet in the counted reading — it was retitled"
    nxt = reading.find(" - **", start + 1)
    return reading[start:nxt if nxt > start else len(reading)]


def test_criterion_a_names_the_population_it_counted(reading):
    """Clause 1: the reader can see that a turn whose id parses to another kind
    is not in the 23.

    `chat` is what `session_kind` falls through to, not a positive test, so the
    doc has to say the rule — an id that splits on `_` into fewer than four parts
    — or the count is a claim about a set nobody can re-form from the extract.
    """
    b = _criterion_bullet(reading, "a")
    assert "session_kind" in b, "(a) names no classifier, so 'chat turn' is undefined"
    assert "scripts/vllm_prefix_miss_window.py" in b and re.search(
        r"scripts/vllm_prefix_miss_window\.py:\d+", b), \
        "(a) must cite the classifier where it lives, by line"
    for phrase in ("four", "chat"):
        assert phrase in b.lower(), f"(a) does not state the rule (missing {phrase!r})"
    assert re.search(r"fewer than four|four or more", b), \
        "(a) must give the part-count rule that decides chat vs a producer slug"
    assert re.search(r"invisible to \(a\)|not counted by \(a\)|rather than counted in", b), \
        "(a) must say explicitly that a turn parsing to another kind is outside the 23"


def test_criterion_a_cites_both_chat_mints_and_no_four_part_one(reading):
    """Clause 2: both chat-id mints, path and line, and the four-part background
    mint named as the one that is NOT in the count.

    The line numbers are checked against the code they cite, not just grepped: a
    citation that only has to appear in the doc is a citation that silently rots
    the first time `sessions.py` grows a function above it. And #1719 was filed
    asking for `app/sessions_io.py:279` — `new_background_session_id`, which mints
    exactly the four-part ids `session_kind` parses OUT of chat — so that path may
    appear in this bullet only as the exclusion, never as the chat id's origin.
    """
    b = _criterion_bullet(reading, "a")
    cited = dict(re.findall(r"`(app/[\w/]+\.py):(\d+)`", b))
    assert set(cited) == {"app/routers/sessions.py", "app/routers/messages.py",
                          "app/sessions_io.py"}, \
        f"(a) cites {sorted(cited)}; it must name both chat mints and the background one"

    def _line(path: str, n: int) -> str:
        return (ROOT / path).read_text(encoding="utf-8").splitlines()[n - 1]

    iv_line = _line("app/routers/sessions.py", int(cited["app/routers/sessions.py"]))
    assert "iv" in iv_line and "token_hex" in iv_line, (
        f"app/routers/sessions.py:{cited['app/routers/sessions.py']} is {iv_line.strip()!r} "
        "— the doc's iv-mint citation has drifted off `suffix = \"iv\" + "
        "secrets.token_hex(2)`, which is the line that makes an iv id a chat id")
    plain_line = _line("app/routers/messages.py", int(cited["app/routers/messages.py"]))
    assert "uuid4" in plain_line and "%Y%m%d_%H%M%S" in plain_line, (
        f"app/routers/messages.py:{cited['app/routers/messages.py']} is {plain_line.strip()!r} "
        "— the doc's plain-chat-mint citation has drifted off the 3-part id literal")
    bg_line = _line("app/sessions_io.py", int(cited["app/sessions_io.py"]))
    assert "def new_background_session_id" in bg_line, (
        f"app/sessions_io.py:{cited['app/sessions_io.py']} is {bg_line.strip()!r} — the "
        "line the doc names as the four-part mint is not that mint's def")

    # The exclusion has to be stated, not merely left unstated: #1719's own text
    # asked for this path as the chat mint, and "absent" would not survive the
    # next reader who walks in with the same idea.
    excl = re.search(r"app/sessions_io\.py:\d+`?[^\n]{0,400}", b, re.S)
    assert excl and re.search(r"not in the 23|excluded|counted as whichever slug", excl.group(0)), \
        ("the background mint appears in (a) without being named as excluded from the 23 — "
         "the exact confusion #1719 was filed over")
    assert not re.search(r"(iv|Inner.Voice)[^\n]{0,120}app/sessions_io\.py", b) \
        and not re.search(r"app/sessions_io\.py:\d+[^\n]{0,120}(iv\b|Inner Voice)", b), \
        "(a) attributes the iv-prefixed chat id to the background mint"


def test_the_bar_names_its_population_instead_of_a_normal_day(s10):
    """Clause 3: "one normal day at that shape with Alan chatting" is gone from
    §10, and what replaced it is checkable — conditions (a)-(d) plus chat turns
    present in the window.

    The phrase is not banned for style. It has no measured content, which is why
    the owed check that had to decide whether 23 turns cleared it could only
    re-litigate it. §7's "KV p90 < 60% on a normal day" is a different page's
    problem and deliberately out of scope here.
    """
    assert "normal day" not in s10.lower(), (
        "§10 still asks for a 'normal day': the bar's population is a phrase, not a "
        "measurement, and that is what owed-check had to re-decide")
    bar = s10[:s10.index("- **The counted reading")]
    assert re.search(r"one named full UTC day", bar), \
        "the bar must name what it is met over — a day an extract on disk covers"
    for letter in "abcd":
        assert re.search(rf"\({letter}\)", bar), f"the bar no longer lists condition ({letter})"
    assert re.search(r"chat turns[^\n]{0,200}(population requirement|not a fifth measurement)",
                     bar, re.S) or \
        re.search(r"(population requirement|not a fifth measurement)[^\n]{0,200}chat turn",
                  bar, re.S), \
        "the chat-turns requirement must be stated AS a population requirement, so a reader " \
        "does not go looking for a fifth measurement"
    assert "zero of zero" in bar, (
        "the reason chat turns must be present is that (a) is vacuously true with none in "
        "the window; the bar has to say so or the requirement reads as a preference")


def test_the_counting_day_may_be_full_of_worker_turns(reading, derived):
    """Clause 4: the day need not be free of bench/worker turns, the by-kind list
    is called what it is, and no figure is hand-written to fill the gap.

    `by_kind` is built over `missing` (`scripts/vllm_prefix_miss_window.py:225-229`), so it
    is a breakdown of the turns that missed and says nothing about the 443-turn mix; calling
    it a turn mix would be a false statement in derived prose, which is the worst kind this
    page has. The mix IS computable from the extract — its turn rows carry the kind — which
    is what makes the ban below a check rather than a wish: for every kind whose turn count
    differs from its miss count, writing `kind <turn count>` could only have come from
    outside `derive`. Where the two coincide (`deepresearch` missed on all 4 of its turns)
    the pairing is genuinely ambiguous and is left alone: a test cannot ban a number that is
    also a legal one, and pretending otherwise would be an invention from the other side.
    """
    ex = json.loads(EXTRACT.read_text(encoding="utf-8"))
    assert re.search(r"need not be free of bench[^\n]{0,40}worker|bench and worker turns",
                     reading), "§10 must say the counting day is allowed to be busy"
    assert "miss breakdown" in reading, "the by-kind list must be called a miss breakdown"
    assert re.search(r"not[^\n]{0,30}turn mix", reading), \
        "the text must say the by-kind list is NOT the day's turn mix"

    import collections
    true_turns = collections.Counter(t[1] for t in ex["turns"])
    derived_misses = {k: v[0] for k, v in derived["by_kind"].items()}
    assert true_turns["chat"] == derived["chat_turns"], \
        "the extract's own chat count no longer agrees with derive; the 23 is unverified"
    ambiguous = {k for k, n in true_turns.items() if n == derived_misses.get(k)}
    banned = {k: n for k, n in true_turns.items() if n != derived_misses.get(k)}
    assert len(banned) >= 10 and "chat" in banned and "bench" in banned, (
        f"only {len(banned)} kinds separate a turn count from a miss count ({sorted(ambiguous)}"
        ") — too few for the ban below to be worth running")
    for kind, n in banned.items():
        assert not re.search(rf"\b{re.escape(kind)} {n}\b", reading), (
            f"§10 pairs {kind!r} with {n}: that is its TURN count in the extract and is not "
            "a figure `derive` returns, so it can only have been hand-written")
    assert not re.search(r"\d+ chat sessions?", reading, re.I), \
        "a chat-session count is not derivable (turn rows carry no session id)"


# ── #1810: (a)'s mints say what the code does, and the population ruling is on
# the record.
#
# The (a) bullet used to read "an Inner-Voice-on chat takes the three-part
# `<ts>_iv<hex>` shape": `app/routers/sessions.py:724` is `suffix = "iv" +
# secrets.token_hex(2)`, and it runs unconditionally inside `create_session`, ahead of
# the line that first reads the body — so the prefix is what a pre-created chat id
# HAS, and a reader following the old sentence would go looking for the chat population
# in the wrong set of sessions and find the iv-shaped ones with the option off. The
# other half of the item is that the population ruling owed-check made on 2026-09-29
# ("23 chat turns … DO constitute a day that clears (a) … the last time that question is
# asked") was never written here, so the page still deferred it.

#: The wording §10 carried before #1810, verbatim. The ban below reads it, so the
#: ban is demonstrably a check and not a description of the current text.
_PRE_FIX_IV_CLAUSE = (
    "The two mints whose ids land in the 23 cover the conversations this engine serves: "
    "an Inner-Voice-on chat takes the three-part `<ts>_iv<hex>` shape from "
    "`app/routers/sessions.py:724` (`suffix = \"iv\" + secrets.token_hex(2)`), and a plain "
    "one `<ts>_<6 hex>` from `app/routers/messages.py:2117` — three parts each, so neither "
    "is mistaken for a producer's.")

#: Ways the bullet can point at the id prefix itself.
_IV_PREFIX = re.compile(r'"iv"|_iv<|(?<!\w)iv(?!\w)')
#: Ways it can point at the opt-in that does NOT decide that prefix.
_IV_OPT_IN = re.compile(r"inner[ _-]?voice", re.I)
#: Clause boundaries for the association ban. `_section` flattens the doc to single
#: spaces, so `;` and `:`-free sentences are all there is; semicolon is a boundary too,
#: because "the prefix is `iv`; it is not the inner_voice flag" would be a TRUE clause
#: that the ban must not fire on, and joining two claims with `;` must not smuggle the
#: association past a node that is about one clause being false.
_CLAUSE_BREAK = re.compile(r"[.!?;]+\s+")


def _iv_associations(text: str) -> list[str]:
    """Clauses naming both the `iv` id prefix and the Inner-Voice opt-in.

    The co-occurrence IS the misattribution: `create_session` appends the prefix
    unconditionally, so any clause that presents the two as one statement — "an
    Inner-Voice-on chat takes the `iv` shape" — states a condition the code does not
    implement. The two true facts (this is the prefix; this is a separate flag read
    later) are two clauses, and the bullet is written that way.
    """
    return [c.strip() for c in _CLAUSE_BREAK.split(text)
            if _IV_PREFIX.search(c) and _IV_OPT_IN.search(c)]


def test_criterion_a_never_binds_the_iv_prefix_to_the_inner_voice_opt_in(reading):
    """Clause 2: the association is banned, and provably not by a test that cannot fire."""
    offenders = _iv_associations(reading)
    assert not offenders, \
        f"(a) attributes the iv prefix to Inner-Voice opt-in in {offenders}"
    # Non-vacuity, in-suite: the exact sentence this item was filed to delete.
    restored = _iv_associations(_PRE_FIX_IV_CLAUSE)
    assert len(restored) == 1, (
        f"the ban cannot see the pre-fix wording ({restored}) — it is a description of "
        "the current text, not a check on the next one")


def test_criterion_a_names_create_session_as_the_mint_that_ignores_the_flag(reading):
    """Clause 1: what the bullet says INSTEAD, checked against the code it cites.

    Not just "the false phrase is absent": the bullet has to state the mint, its
    unconditionality, and the fallback's role. `messages.py:2117` stays qualitative —
    the fallback's share of the record is a `usage.db` figure, and this page's rule is
    to cite the query rather than hand-copy the number.
    """
    b = _criterion_bullet(reading, "a")
    assert "`app/routers/sessions.py:724`" in b and "`app/routers/messages.py:2117`" in b, \
        "(a) must carry both chat mints by path and line: " \
        "test_criterion_a_cites_both_chat_mints_and_no_four_part_one asserts the SET of " \
        "cited paths and checks each line against the code; this pins neither was dropped"
    mint = [c for c in _CLAUSE_BREAK.split(b) if "create_session" in c and _IV_PREFIX.search(c)]
    assert mint, "(a) never states what `create_session` does to the id"
    assert any("every chat session it pre-creates" in c for c in mint), \
        f"the mint clause says {mint} — it must say the prefix is appended to EVERY " \
        "pre-created chat session, which is what makes it not an opt-in"
    assert "not a mark of anything the caller opted into" in b, \
        "(a) must say the shape is not conditional on the request"
    flag = [c for c in _CLAUSE_BREAK.split(b) if _IV_OPT_IN.search(c)]
    assert flag and any("never what it is called" in c for c in flag), \
        f"(a) mentions the opt-in in {flag} without saying it does not name the session"
    fb = [c for c in _CLAUSE_BREAK.split(b) if "messages.py" in c]
    assert fb and re.search(r"fallback[^\n]{0,160}no session id", fb[0]), \
        f"(a) does not scope the fallback mint to a message with no session id: {fb}"
    assert not re.search(r"\b(zero|0)\b", " ".join(fb)), \
        "(a) quotes a live count for the fallback's contribution; state the role and " \
        "cite the query, as the preamble to this item requires"


#: The single read-only query §10 is allowed to carry for the chat population. It is
#: `session_kind`'s rule in SQL: not `task:`-prefixed, and too few `_` parts to carry a
#: producer slug. Verified row-for-row equal to `session_kind` over `usage.db`.
_CHAT_QUERY = re.compile(r"`sqlite3 -readonly[^`]*`")
_POPULATION_CLOSED = re.compile(r"population requirement is CLOSED")

#: The window §10's population superlative is dated to: the record owed-check's ruling was
#: measured over, which #1918 froze into a claim with an end date. A superlative with no
#: date in it re-scopes itself every time the record grows — that is what left a standing
#: re-count as its only guard. The COUNT in that same sentence is deliberately not here:
#: it is `derived['chat_turns']`, so re-cutting the fixture moves the figure while the
#: window stays where the measurement was actually taken.
_POP_WINDOW_END = "2026-09-30"
_SUPERLATIVE = re.compile(
    r"(\d+) chat turns is the machine's own chat volume and the busiest chat day "
    r"in the record up to (\d{4}-\d\d-\d\d)")

#: The falsifier that replaced the beat-N re-open clause, by half. Half one: a day whose
#: chat volume is AT OR ABOVE the counted one. Half two: a chat turn on that day carrying a
#: miss. Either half alone cannot un-meet "a day at that volume clears the bar's population
#: requirement" — volume alone is the retired trigger (a busier miss-free day only says the
#: machine chats more), and a miss alone is §10's accepted-loss trigger one, which fires on
#: any chat-turn miss and moves no population ruling.
_FALSIFIER_VOLUME = re.compile(r"a day at or above (?:that|the counted) volume")
_FALSIFIER_MISS = re.compile(r"chat turn carries (?:a|at least one) (?:chat )?miss")

#: The wording #1918 replaced, verbatim. The nodes below read it, so each pin is
#: demonstrably a check on the next edit and not a description of the current text.
_PRE_1918_SUPERLATIVE = (
    "busiest chat day in the re-derivable record — the record being what that query can "
    "see, which starts where (b) says `usage.db` starts")
_PRE_1918_REOPEN = "moving it means that query returning a day that beats this one"
#: The two one-halved forms, spelled out so the conjunction pin is falsifiable: the first
#: is the retired trigger with its volume half kept, the second is trigger one's wording.
_VOLUME_ONLY_FALSIFIER = "moving it means that query returning a day at or above that volume"
_MISS_ONLY_FALSIFIER = ("moving it means that query returning a day on which a chat turn "
                        "carries a miss")


def _population_falsifiers(text: str) -> list[str]:
    """Clauses of `text` stating (a)'s re-count falsifier — BOTH halves in the one clause.

    Clause-scoped, not paragraph-scoped: halves parked in two separate sentences can be
    edited apart, and a reader who lands on the volume sentence alone would take it for the
    retired trigger — which is exactly the reading #1918 was filed to make impossible.
    """
    return [c.strip() for c in _CLAUSE_BREAK.split(text)
            if _FALSIFIER_VOLUME.search(c) and _FALSIFIER_MISS.search(c)]


def test_the_population_ruling_is_closed_and_named_owed_checks(s10, reading, derived):
    """Clause 3: owed-check's ruling replaces the deferral, with the derived count in it.

    The number is `derived['chat_turns']`, so the ruling moves if the counted day is
    ever re-cut — a ruling pinned to a hand-typed 23 would survive a fixture change
    that had already invalidated it.

    #1918 added the other half of the sentence's honesty: the superlative has to name the
    window it is the busiest day OF, with a date in it. "The busiest chat day in the
    re-derivable record" was true only of the record as short as it happened to be on the
    day it was written, and nothing in the text said which record that was, so the claim
    silently widened every time a turn was logged and a standing re-count was the only
    thing standing between the page and a false superlative. The count still comes from the
    derivation; only the window is pinned as prose, because the window is when the
    measurement was taken and no fixture knows that.
    """
    assert "not a call this page can make" not in s10, \
        "§10 still defers the population question that owed-check ruled on 2026-09-29"
    b = _criterion_bullet(reading, "a")
    assert _POPULATION_CLOSED.search(b), "(a) does not record the population bar as CLOSED"
    m = _SUPERLATIVE.search(b)
    assert m, (
        "(a)'s superlative is not window-explicit: it must read \"N chat turns is the "
        "machine's own chat volume and the busiest chat day in the record up to <date>\" — "
        f"undated, it means whatever the record means next week. Bullet opens: {b[:150]!r}")
    assert int(m.group(1)) == derived["chat_turns"], \
        f"(a) states its own count as {m.group(1)}, but the derivation over the named " \
        f"extract says {derived['chat_turns']} — the ruling has to move with the data"
    assert m.group(2) == _POP_WINDOW_END, \
        f"(a) dates its window to {m.group(2)}; the record this ruling was measured over " \
        f"ends {_POP_WINDOW_END}, and the clause has to name the window it was read from"
    assert _PRE_1918_SUPERLATIVE not in b, \
        "the undated superlative is back — 'the re-derivable record' names no window, so " \
        "the day after this one's turns land the sentence is a different claim"
    assert re.search(r"owed-check ruled on \d{4}-\d\d-\d\d", b), \
        "the ruling must be named as owed-check's and dated"
    assert "day at that volume clears the bar's population requirement" in b, \
        "the ruling must say the requirement is cleared, not restate it as a question"
    assert "owed-check's and stands until a re-count moves it" in b, \
        "the ruling must carry its own expiry: a re-count moves it, a round does not"


def test_the_population_recount_trigger_falsifies_the_ruling_not_the_calendar(reading, derived):
    """#1918 clause 2: the re-count's trigger is the observation that can un-meet the
    ruling, and it is stated as a measurement rather than as a second hand-typed figure.

    The retired clause fired on volume alone ("a day that beats this one"). Over the record
    the ruling was measured through, every day that carries a chat miss sits well below the
    counted volume and no day above it does — so the clause could only ever be fired by a
    day that changed nothing about (a), and could never be fired by the thing that actually
    would: the population thinning out, or the same volume arriving with a miss on it. The
    replacement is the conjunction of the two columns the cited query already prints. It
    carries no digit of its own, because its volume half points back at the derived count
    named earlier in the bullet — so a re-cut fixture moves the bar without anyone editing
    a number into this sentence, and the re-count stays one query.

    Deliberately NOT a parameter here: `derived`. The count this clause forbids the sentence
    from repeating lives up-bullet, and the node that binds it to the derivation is
    `test_the_population_ruling_is_closed_and_named_owed_checks`; a fixture used only to
    decorate a failure message would make this node look data-dependent when it is not.
    """
    b = _criterion_bullet(reading, "a")
    assert _PRE_1918_REOPEN not in b and "beats this one" not in b, \
        "the beat-N clause is still (a)'s re-open trigger"
    fired = _population_falsifiers(b)
    assert len(fired) == 1, (
        "(a) must state its falsifier exactly once, in one clause: "
        f"{[c for c in _CLAUSE_BREAK.split(b) if 're-count' in c or 'volume' in c]}")
    for c in fired:
        assert not re.search(r"\d", c), \
            f"the falsifier hand-wrote a figure ({c!r}); its volume half is the count named " \
            f"up-bullet, which is the derivation's, not a number copied down here"
    # §10-wide, not the counted-reading slice: the clause grades the section, and a second
    # read-only query added anywhere in §10 makes the falsifier ambiguous about which one
    # the re-count runs. The slice would let that through.
    q = _CHAT_QUERY.findall(_section(10))
    assert len(q) == 1, \
        f"the falsifier has to stay answerable by the ONE query §10 already cites, got {q}"


def test_the_population_falsifier_is_a_conjunction_with_no_eye_on_the_record(reading):
    """#1918 clause 3: drop either half of the falsifier and this node goes red — and no
    day of traffic can do it instead.

    The conjunction is the whole of what makes the trigger a falsifier, and #1918's
    predecessor is the proof: a trigger with one half sat there for a fortnight unable to
    fire. So the half-alone forms are checked rather than assumed. The volume-only form is
    verbatim the retired trigger, so a pin that accepted it would let six deleted words
    restore the defect; the miss-only form is §10's accepted-loss trigger one, which moves
    no population ruling.

    The "no eye on the record" half of the name is a constraint on this node's INPUTS, not a
    property it can demonstrate at runtime. Its only input is `reading`, which is doc text,
    and the three other values it reads are the constants defined above, so there is no
    record route to take — a trap around `sqlite3.connect` here could never fire, which is
    exactly what the review rung of SM_20260930_201725 caught when it called that line
    decoration. Why the constraint is worth stating at all: the module's one record-reading
    node, `test_the_cited_chat_query_selects_what_the_classifier_selects`, already succeeds,
    fails or SKIPS on the state of `usage.db`, and a WORDING pin sharing that dependency would
    go red the week a chat turn misses, over a defect no round can fix from a commit. The
    witness for the separation is this signature — adding a fixture that touches the record
    here is the change that breaks it, and the next editor is the audience for that sentence.
    """
    b = _criterion_bullet(reading, "a")
    assert len(_population_falsifiers(b)) == 1, \
        "(a)'s falsifier is not one clause carrying both halves; the two one-halved " \
        "forms asserted below are the ways it can decay"
    assert not _population_falsifiers(_VOLUME_ONLY_FALSIFIER), \
        "a volume-only clause passes the pin, so the retired beat-N trigger could come " \
        "back by deleting the miss half of the sentence"
    assert not _population_falsifiers(_MISS_ONLY_FALSIFIER), \
        "a miss-only clause passes the pin, which would make (a)'s re-count the same " \
        "condition as the accepted-loss ruling's trigger one"
    assert _population_falsifiers(_VOLUME_ONLY_FALSIFIER
                                  + " on which a chat turn carries a miss"), \
        "the pin cannot see a real conjunction at all, so nothing above it is a check"


def test_the_chat_population_is_day_scoped_and_re_derivable_by_one_query(reading):
    """Clause 4: the discriminating claim, the day it is true of, and the ONE query.

    Two rails here. The query: exactly one, read-only, and shaped so it actually
    re-derives BOTH counts per day — chat turns and chat turns carrying a miss — with
    the classifier spelled the way `session_kind` spells it. The prose: the clause that
    asserts chat turns miss on other days carries no digit at all, because its number
    is the query's output and a figure copied in here is a figure nobody re-runs.
    """
    b = _criterion_bullet(reading, "a")
    q = _CHAT_QUERY.findall(reading)
    assert len(q) == 1, f"§10 must cite exactly one read-only chat-population query, got {q}"
    for clause in ("substr(ts,1,10)", "count(*)", "sum(prefix_misses>0)",
                   "not like 'task:%'",
                   "length(session_id)-length(replace(session_id,'_',''))<3",
                   "group by day"):
        assert clause in q[0], f"the query cannot re-derive (a)'s population without {clause}"
    assert "`sqlite3 -readonly" in q[0], "the cited read must not be able to write usage.db"
    assert re.search(r"\*\*\(a\) is day-scoped — the mark names the day", b), \
        "(a) must say its mark is a claim about the day it is counted over"
    discriminating = [c for c in _CLAUSE_BREAK.split(b) if "discriminate" in c]
    assert discriminating and any("other days" in c for c in discriminating), \
        f"(a) must be discriminating — chat turns carry misses on other days: {discriminating}"
    rates = [c for c in _CLAUSE_BREAK.split(b) if "pooled" in c]
    assert rates, "(a) must say the other-day misses are a measured pooled rate"
    for c in rates:
        assert not re.search(r"\d", c), \
            f"the pooled-rate clause hand-wrote a figure ({c!r}); the query re-derives it"


#: Ids that decide the WHERE clause, chosen so every branch of it is taken: the
#: pre-created chat shape, the fallback's shape, a producer's four-part id, a deeper
#: one, and two `task:` ids whose underscore count alone would otherwise pass them —
#: which is why the exclusion needs its own witness (see the node below).
_PROBE_IDS = ("20260925_041134_ivb794", "20260925_041134_a1b2c3",
              "20260925_041134_autocode_bd3f", "20260925_041134_owedcheck_46d6_x",
              "task:autonomy-42", "task:1", "")


def test_the_cited_querys_chat_predicate_is_the_classifier():
    """The WHERE clause, run over ids that force every branch — no database required.

    Why this node exists beside the one that reads the production record: `usage.db`
    holds ZERO `task:`-prefixed session ids today, so on that corpus the exclusion is
    unexercised and deleting it from the doc's query changes no row. A check whose
    denominator can be zero is not a check, so the branch gets a synthetic witness
    instead, and the branch stays covered whatever the record does next.
    """
    import sqlite3

    q = _CHAT_QUERY.search(_section(10))
    assert q, "§10 cites no read-only chat-population query"
    w = re.search(r"\bwhere\b(.+?)\bgroup by\b", q.group(0), re.S)
    assert w, "the cited query has no WHERE clause to compare with the classifier"
    con = sqlite3.connect(":memory:")
    try:
        con.execute("create table usage (ts text, session_id text, prefix_misses integer)")
        con.executemany("insert into usage values ('2026-09-25T04:11:34',?,0)",
                        [(i,) for i in _PROBE_IDS])
        selected = {r[0] for r in con.execute(f"select session_id from usage where {w.group(1)}")}
    finally:
        con.close()
    want = {i for i in _PROBE_IDS if W.session_kind(i) == "chat"}
    assert want and len(want) < len(_PROBE_IDS), \
        "the probe set stopped separating chat from non-chat, so this node proves nothing"
    assert selected == want, (
        f"§10's WHERE selects {sorted(selected)} where `session_kind` calls chat "
        f"{sorted(want)} — the ruling and criterion (a) would count different turns")


def test_the_cited_chat_query_selects_what_the_classifier_selects(reading):
    """The one process boundary this ruling crosses, put under test rather than asserted.

    (a)'s population is decided in Python (`session_kind`), the ruling's "busiest chat
    day in the record up to <date>" is decided in SQL over `usage.db`, and the doc claims
    they are the same set. Two languages, one population, no shared code — a grep cannot
    see the disagreement, only a run can. So the query is read OUT of §10 (the test does
    not carry its own copy, which is what would let the doc and the check drift), and the
    per-day table it returns is compared to the per-day table the classifier implies.
    Skipped where there is no database to ask, exactly as the extractor's node does.

    Deliberately NOT pinned here: which day is the busiest, or how many turns it carries.
    #1918 dated the superlative, so widening the record can no longer refute it from this
    page, and the re-count that WOULD move the ruling now needs a day at or above the
    counted volume carrying a chat miss. A suite assertion on either figure would be a test
    that goes red for the one job that cannot fix it from a round.
    """
    import collections
    import sqlite3

    from app.paths import PRODUCTION_DATA_ROOT

    cmd = _CHAT_QUERY.search(reading)
    assert cmd, "§10 cites no `sqlite3 -readonly` command to re-derive the chat population"
    m = re.search(r"sqlite3 -readonly\s+(\S+)[^\"]*\"(.+?)\"", cmd.group(0))
    assert m, "the cited command must name a database and carry its SQL"
    assert m.group(1).endswith("/usage.db"), \
        f"the doc's command reads {m.group(1)}; it must name the usage database"
    db = PRODUCTION_DATA_ROOT / "usage.db"
    if not db.exists():
        pytest.skip(f"no usage.db at {db}")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        from_doc = {r[0]: (r[1], r[2]) for r in con.execute(m.group(2))}
        implied: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for ts, sid, misses in con.execute("select ts, session_id, prefix_misses from usage"):
            if W.session_kind(sid) == "chat":
                implied[ts[:10]][0] += 1
                implied[ts[:10]][1] += 1 if (misses or 0) > 0 else 0
    finally:
        con.close()
    assert from_doc, "the doc's query returns no chat days, so it cannot re-derive anything"
    assert sum(v[0] for v in from_doc.values()) >= 10, \
        "the record must carry enough chat turns for the population ruling to be about anything"
    assert from_doc == {d: tuple(v) for d, v in implied.items()}, (
        "§10's query and `session_kind` do not select the same chat population — the "
        "ruling and criterion (a) would then be counting different turns")


def test_the_raw_ids_paragraph_records_a_decision_and_its_reopen_trigger(reading):
    """Clause 5: the open question became a decision, with a trigger that is two things.

    The old sentence parked the choice ("is a scope decision … not a figure to write in
    from elsewhere"), so the next reader asked it again. The record now says the extract
    is NOT extended, why, and the one combination that re-opens it — a figure both
    load-bearing for acceptance and NOT answerable live, which is the conjunction that
    keeps "we could add raw ids" from re-opening it on its own.
    """
    assert "Whether carrying raw ids would make a turn mix derivable is a scope decision" \
        not in reading, "the raw-ids question is still parked as a question"
    assert "Carrying raw session ids in the extract is therefore decided against, " \
           "not left open" in reading, "(a)'s page must record the decision, not the option"
    assert "would add no figure (a)-(d) needs" in reading, "and the reason: no criterion needs one"
    assert "answerable by a live" in reading and "usage.db" in reading, \
        "and the standing alternative: a live query answers what a raw id would"
    assert "Re-open trigger, both halves required:" in reading, \
        "the decision must say what would re-open it"
    assert "load-bearing for acceptance *and* not answerable by a live `usage.db` query" \
        in reading, "the trigger is a conjunction; either half alone is not enough"
    assert "`extract` is extended and this fixture re-cut in the same round" in reading, \
        "and what re-opening means: extract and fixture move together, not the prose alone"


def test_criterion_b_is_marked_and_is_the_per_day_budget(reading, derived, s10):
    """(b)'s mark follows the budget, and the budget follows §6.1."""
    s61 = _section(6)
    assert "20.6M tokens" in s61 and "09-08/09" in s61
    miss_budget, token_budget = _per_day_budget(s61)
    assert f"{miss_budget} misses and {token_budget}M tokens a day" in reading
    assert "history now" in reading, "the baseline is labelled, not presented as live"
    d = derived
    assert f"{d['misses']} and {d['reprefill_tokens'] / 1e6:.1f}M in *one* day" in reading
    assert (f"about {d['misses'] / miss_budget:.1f}x the misses and "
            f"{d['reprefill_tokens'] / (token_budget * 1e6):.1f}x the tokens") in reading
    passes = d["misses"] <= miss_budget and d["reprefill_tokens"] <= token_budget * 1e6
    assert _verdict(reading, "b") == ("passes" if passes else "fails")


def test_criterion_c_is_marked_and_is_the_kv_median(reading, derived, cfg):
    d = derived
    assert f"{_n(d['kv_samples'])} such lines" in reading
    assert (f"KV p50 {d['kv_p50']:.2f} / p90 {d['kv_p90']:.2f} / max {d['kv_max']:.2f}") in reading
    assert f"`Running:` {d['running_p50']} at the median" in reading
    assert "GPU KV cache usage" in reading and "agent-llm-primary.log" in reading
    gate = float(cfg["workers"]["kv_gate"]["max_kv_usage"])
    assert _verdict(reading, "c") == ("passes" if d["kv_p50"] < gate else "fails")


def test_criterion_d_is_marked_and_is_the_two_request_split(reading, derived):
    """The whole of #1627's second half: (d) used to be a sentence the page
    asserted. Every figure in it now comes out of the derivation."""
    d = derived
    assert (f"Lines with `Running:` {W.TWO_REQUEST_MIN} or more: "
            f"**{_n(d['two_request_windows'])}**, median **{d['two_request_tok_s_p50']} tok/s** "
            f"combined, slowest **{d['two_request_tok_s_min']}**") in reading
    assert f"under the bar's {W.STALL_TOK_S:.0f} tok/s: **{d['two_request_under_stall']}**" in reading
    assert (f"admission in flight: **{d['two_request_under_stall_cold_admission']}**") in reading
    assert (f"{d['two_request_under_stall_cold_in_flight']} of them with a cold re-admission"
            in reading)
    assert f"{d['two_request_under_stall_cold_prefill']} with a chunked prefill" in reading
    assert f"counted against (d): **{d['two_request_under_stall_not_cold']}**" in reading
    assert _verdict(reading, "d") == ("passes" if d["two_request_under_stall_not_cold"] == 0
                                      else "fails")


def test_every_criterion_carries_exactly_one_mark(reading):
    for letter in CRITERIA:
        _verdict(reading, letter)


# ── the gap join, and what it says on this day ────────────────────────────

def test_the_miss_gaps_are_the_derivation_and_name_a_branch(reading, derived):
    d = derived
    assert d["misses_with_gap"] == d["miss_events"], "every miss joined to its gap"
    assert f"gap p50 {d['gap_s_p50']:.1f} s" in reading
    assert f"p50 {d['miss_kv_gap_p50']:.3f} / p90 {d['miss_kv_gap_p90']:.3f}" in reading
    assert f"max {d['miss_kv_gap_max']:.3f}" in reading
    assert f"{d['misses_gap_over_gate']} of {d['miss_events']} at or over the" in reading
    assert f"**{d['misses_gap_churned_free_pool']} of {d['miss_events']}**" in reading
    assert f"The other {d['miss_events'] - d['misses_gap_churned_free_pool']} misses" in reading


def test_the_eviction_branch_named_is_the_one_the_gaps_support(reading, derived):
    """§10 used to head this "Eviction, but not by pressure at the gate". On
    2026-09-25 that heading was the wrong side of its own numbers — most gaps
    peak over the gate and three churn — so the heading is derived too."""
    d = derived
    assert f"free-pool churn in {_word(d['misses_gap_churned_free_pool'])}" in reading
    assert ("gate-level KV in most of the gaps" in reading) == \
        (d["misses_gap_over_gate"] * 2 > d["miss_events"])
    assert ("none over 0.90" in reading) == (d["misses_gap_over_90"] == 0)


def test_the_draft_group_branch_still_names_this_window(reading, derived):
    """The unannotated draft group is the branch for the misses eviction does
    not explain, so its "not off wholesale" evidence has to be this day's."""
    d = derived
    partial = d["miss_events"] - d["cold_events"]
    assert f"{partial} of the day's {d['miss_events']} misses still read part of " \
           f"their prompt from cache" in reading


# ── #1720: the accepted-loss ruling on the non-churn misses, and what re-opens it ─
#
# Three sweeps asked the same question (#1339's owed clause 2 → #1627's owed entry
# 2 → this item) because §10 closed the eviction bullet by parking the call on a
# person, and `aa68ea74` ("nothing parks on Alan — owed work is settled by Lloyd")
# edited this page without touching that line. owed-check run
# `20260927_232609_owedcheck_46d6` ruled it on 2026-09-28; these nodes are what
# keeps the ruling and, more importantly, the two figures it re-opens on from
# drifting into hand-edited prose. Every number below comes out of `derived` or
# out of `_per_day_budget`, so a figure edited in the doc without the data moving
# fails here the same way a counted figure already does.

#: The predicate criterion (a) is marked on, spelled once so the trigger and the
#: criterion cannot quietly come to mean different things.
_A_PREDICATE = "carries a prefix miss"

_TRIGGER_CHAT_RE = re.compile(rf"any counted window in which a chat turn ({_A_PREDICATE})")
_TRIGGER_BUDGET_RE = re.compile(
    r"any counted day exceeding §6\.1's per-day budget of (\d+) misses / ([\d.]+)M "
    r"re-prefilled tokens")


def test_the_accepted_loss_ruling_replaced_the_deferral(s10, reading, derived):
    """The deferral sentence is gone and the decision it stood in for is in the
    text, in the ruling's own words, with the reason the ruling gives for it."""
    assert "Alan's call, not a round's" not in s10, "the sentence owed-check ruled on is still standing"
    assert "Whether to chase that upstream" not in s10
    assert "accepted as a bounded, documented loss" in reading
    assert "the upstream unannotated draft-group annotation is not chased" in reading
    assert "unmeasured rather than merely unfinished" in reading
    assert "does not claim reuse is disabled" in reading
    assert "capability chase with no measured gain" in reading
    partial = derived["miss_events"] - derived["cold_events"]
    assert f"the day's {partial} partial misses show reuse works" in reading, \
        "the reason cites this day's partial misses, so it cites their derived count"


def test_the_ruling_names_both_reopen_triggers(reading):
    """A loss is only *bounded* if something bounded re-opens it, and the two
    conditions the ruling named are the whole of that bound."""
    assert _TRIGGER_CHAT_RE.search(reading), \
        "trigger one — a counted window in which a chat turn carries a prefix miss — is missing"
    assert _TRIGGER_BUDGET_RE.search(reading), \
        "trigger two — a counted day over §6.1's per-day budget — is missing"


def test_reopen_trigger_two_s_figures_are_the_budget_derived_from_six_one(reading):
    """The half that makes the bound a measurement: edit `97` or `10.3` in the
    trigger without moving §6.1's baseline and this fails, and move §6.1's
    baseline and the trigger has to move with it."""
    miss_budget, token_budget = _per_day_budget(_section(6))
    m = _TRIGGER_BUDGET_RE.search(reading)
    assert m, "the second trigger must state both figures, not gesture at the budget"
    assert int(m.group(1)) == miss_budget, \
        f"§10's trigger says {m.group(1)} misses; §6.1 derives {miss_budget} a day"
    assert float(m.group(2)) == token_budget, \
        f"§10's trigger says {m.group(2)}M; §6.1 derives {token_budget}M a day"


def test_reopen_trigger_one_is_the_measurement_criterion_a_is_marked_on(reading, derived):
    """The trigger and the criterion read one number. (a) is marked from
    `chat_turns_with_misses`; the trigger has to fire on that same count and use
    that same predicate, or a day could pass (a) and re-open the branch at the
    same time."""
    d = derived
    m = _TRIGGER_CHAT_RE.search(reading)
    assert m, "trigger one is missing from the counted reading"
    assert m.group(1) == _A_PREDICATE
    assert f"{d['chat_turns_with_misses']} of the day's {d['chat_turns']} chat turns " \
           f"{_A_PREDICATE}" in reading, "(a) is no longer marked on that count"
    fired = d["chat_turns_with_misses"] > 0
    assert (_verdict(reading, "a") == "fails") == fired, (
        "the trigger and (a)'s mark read the same count and disagree")


def test_the_ruling_breaks_nothing_the_suite_already_pins(reading, derived):
    """The ruling was inserted into a bullet two other nodes already read, inside
    the span that may carry exactly one verdict mark per criterion. The rewritten
    bullet keeps both derived phrases, and the new prose added no second mark for
    a, b, c or d."""
    d = derived
    assert f"The other {d['miss_events'] - d['misses_gap_churned_free_pool']} misses" in reading
    partial = d["miss_events"] - d["cold_events"]
    assert f"{partial} of the day's {d['miss_events']} misses still read part of " \
           f"their prompt from cache" in reading
    for letter in CRITERIA:
        _verdict(reading, letter)


def test_both_triggers_need_nothing_the_derivation_does_not_already_return(derived):
    """Why no script change was needed, and the check that stays true: each
    trigger counts something `derive` already emits over the extract — (i) the
    chat-turn miss count, (ii) the day's misses and re-prefilled tokens. A
    trigger the derivation cannot count is a trigger nobody will ever fire."""
    for key in ("chat_turns_with_misses", "misses", "reprefill_tokens"):
        assert key in derived, f"the triggers need `{key}`, which derive no longer emits"


# ── what must not come back ───────────────────────────────────────────────

def test_the_2026_09_11_spot_reading_is_history_not_the_reading(s10):
    assert "What the counter says so far" not in s10
    assert "2026-09-11 22:30 UTC" not in s10
    assert re.search(r"2026-09-11 spot reading this page used to quote", s10)


def test_the_replaced_reading_is_not_still_quoted(s10, reading):
    """The 09-23 day is history, and history that quotes counted figures is
    how this page went stale twice. Its extract stays on disk; its numbers do
    not appear in the counted reading."""
    old = W.derive(json.loads(OLD_EXTRACT.read_text(encoding="utf-8")), gate=0.60)
    assert f"{old['turns']} turns" not in reading
    assert _n(old["reprefill_tokens"]) not in reading
    assert f"{old['misses']} miss iterations logged" not in reading
    assert OLD_EXTRACT.name in s10, "the doc says where the replaced reading lives"


def test_no_spot_reading_claims_survive(s10):
    assert "One spot reading" not in s10
    assert "three-minute-old" not in s10


# ── clause 3: (d)'s derivation is a predicate, not a verdict ──────────────

def _ex(**over) -> dict:
    base = {"window": ["2026-09-25", "2026-09-26"], "pool_tokens": [844_969],
            "turns": [], "misses": [], "kv_samples": []}
    base.update(over)
    return base


def _cold_miss(arrived: float, dur_ms: int = 15_000) -> list:
    """A 150k-token iteration that read nothing from cache: §6.2's counted
    miss, and so a cold re-admission by the same floor."""
    return [arrived + dur_ms / 1000, dur_ms, 9, 150_000, 0, "autocode", None]


def test_the_cold_admission_exclusion_separates_a_stall_from_a_prefill():
    """Three windows under 15 tok/s with two requests resident, told apart
    only by their evidence: one has a cold re-admission across the interval,
    one has a chunked prefill climbing across it (KV up, tokens landing on the
    next line), and one is a two-request window that simply ran slowly. Only
    the last counts against (d) — if the predicate excused everything, or
    nothing, this fails."""
    t = W._epoch("2026-09-25") + 3600
    samples = [
        # (a) KV fell across it, no re-admission: a stall, counted.
        [t - 10, 50.0, 2, 5000], [t, 40.0, 2, 100],
        # (b) KV rose across it: a chunked prefill in flight, excused.
        [t + 90, 20.0, 2, 400], [t + 100, 60.0, 2, 100],
        # (c) KV flat and a cold re-admission across the interval: excused.
        [t + 190, 30.0, 2, 9000], [t + 200, 30.0, 2, 50],
        # A healthy two-request window, and a slow single-request one: neither
        # is in the population at all.
        [t + 290, 30.0, 3, 9000], [t + 300, 10.0, 1, 5],
    ]
    d = W.two_request_throughput(_ex(kv_samples=samples, misses=[_cold_miss(t + 190)]))
    assert d["two_request_windows"] == 7, "every line but the one-request one"
    assert d["two_request_under_stall"] == 3
    assert d["two_request_under_stall_cold_in_flight"] == 1
    assert d["two_request_under_stall_cold_prefill"] == 1
    assert d["two_request_under_stall_not_cold"] == 1
    assert d["two_request_under_stall_cold_admission"] == 2
    assert d["two_request_tok_s_min"] == 5.0


def test_the_cold_floor_is_the_miss_definition_not_any_admission():
    """A warm iteration — 150k prompt, three quarters of it cached — is an
    admission but not a *cold* one, so it cannot excuse a slow window."""
    t = W._epoch("2026-09-25") + 3600
    warm = [t + 195, 15_000, 9, 150_000, 120_000, "autocode", None]
    samples = [[t + 190, 30.0, 2, 9000], [t + 200, 30.0, 2, 50]]
    d = W.two_request_throughput(_ex(kv_samples=samples, misses=[warm]))
    assert d["two_request_under_stall_cold_in_flight"] == 0
    assert d["two_request_under_stall_not_cold"] == 1


def test_the_two_request_population_is_the_raw_status_lines():
    """The population and the slow count recomputed straight off the extract's
    rows, independent of the derivation: the numbers §10 quotes are those
    lines, the exclusion cannot swallow one, and there is at least one slow
    window on each day for the exclusion to be about at all. #1921 adds the
    2026-09-29 extract, whose paragraph prints the same four figures: a day
    that re-uses (d)'s rule is a day (d)'s rule has to be independently true
    of, and re-checking it here is why that day's (d) claim is not merely
    `derive` agreeing with itself."""
    for path in (EXTRACT, OLD_EXTRACT, FILING_EXTRACT):
        ex = json.loads(path.read_text(encoding="utf-8"))
        t0, t1 = W._epoch(ex["window"][0]), W._epoch(ex["window"][1])
        raw = [s for s in ex["kv_samples"] if t0 <= s[0] < t1]
        two = [s for s in raw if s[2] >= W.TWO_REQUEST_MIN]
        slow = [s for s in two if s[3] / W.STATUS_INTERVAL_S < W.STALL_TOK_S]
        assert slow, f"{path.name}: no slow two-request window, so (d) proves nothing"
        d = W.two_request_throughput(ex)
        assert d["two_request_windows"] == len(two), path.name
        assert d["two_request_under_stall"] == len(slow), path.name
        assert 0 <= d["two_request_under_stall_cold_admission"] <= len(slow), path.name
        assert (d["two_request_under_stall_not_cold"]
                == len(slow) - d["two_request_under_stall_cold_admission"]), path.name


# ── (a)'s counting actually reads the chat rows ───────────────────────────

def test_chat_turns_are_counted_from_the_turn_rows():
    turns = [["2026-09-25T02:08:09", "chat", 0, 0],
             ["2026-09-25T03:08:09", "chat", 2, 250_000],
             ["2026-09-25T04:08:09", "autocode", 1, 120_000]]
    d = W.derive(_ex(turns=turns), gate=0.60)
    assert (d["chat_turns"], d["chat_turns_with_misses"]) == (2, 1)


# ── where a KV history can and cannot come from ───────────────────────────

def test_the_rotation_that_expires_the_engine_lines_is_the_unit_s(s10):
    """(c) and (d) can only be counted while the engine's own status lines are
    on disk, and what bounds them is supervisor's byte budget on this unit, not
    a number of days this page could keep true. The unit's setting is the
    checkable half of that claim; how many days it works out to is not, so the
    doc names the size and the expiry and quotes no day count."""
    conf = (ROOT / "agent-services" / "supervisor" / "conf.d"
            / "agent-llm-primary.conf").read_text(encoding="utf-8")
    m = re.search(r"^stdout_logfile=(\S*agent-llm-primary\.log)$", conf, re.M)
    assert m, "the unit must write its status lines where the window reads them"
    assert "agent-llm-primary.log*" in s10, "and §10 must name where it reads them"
    size = re.search(r"^stdout_logfile_maxbytes=(\d+)(MB|KB|B)$", conf, re.M)
    assert size, "the log has to rotate for this to be an expiry at all"
    mb = int(size.group(1)) * {"MB": 1, "KB": 1 / 1024, "B": 1 / 1024 ** 2}[size.group(2)]
    assert f"{mb:.0f} MB per file" in s10


def test_the_ring_retention_quoted_is_the_modules(s10, cfg):
    window = engine_pressure.DEFAULT_WINDOW_S
    assert f"its ring is {window:.0f} s (`DEFAULT_WINDOW_S`" in s10
    assert float(cfg["engine_pressure"]["window_seconds"]) == window, \
        "config and module disagree; §10 quotes one number for both"
    assert "emptied by every restart" in s10


def test_the_extractor_still_reads_this_day_out_of_the_database():
    """The extract is a snapshot, and a snapshot that no longer matches the
    database behind it is a fabricated one. The turn and miss rows are the
    half that survives rotation (`usage.db` retains them; the engine's own
    status lines roll off in days, which is why they are not compared here).
    Skipped on a machine whose database does not cover the counted day."""
    from app.paths import PRODUCTION_DATA_ROOT

    ex = json.loads(EXTRACT.read_text(encoding="utf-8"))
    db = PRODUCTION_DATA_ROOT / "usage.db"
    if not db.exists():
        pytest.skip(f"no usage.db at {db}")
    got = W.extract(ex["window"][0], ex["window"][1], data_root=PRODUCTION_DATA_ROOT)
    if not got["turns"]:
        pytest.skip(f"{db} holds no turns for {ex['window']}")
    assert got["turns"] == ex["turns"], "the counted day's turns are not the database's"
    assert got["misses"] == ex["misses"], "the counted day's misses are not the database's"


def test_the_derivation_reads_a_status_line_in_local_time():
    """The engine stamps `MM-DD HH:MM:SS` local with no year; the join to UTC
    rows is only right if the zone is applied, not assumed away."""
    line = ("(APIServer pid=2514) INFO 09-23 07:54:10 [loggers.py:315] Engine 000: Avg prompt "
            "throughput: 3289.4 tokens/s, Avg generation throughput: 256.9 tokens/s, Running: 4 "
            "reqs, Waiting: 0 reqs, GPU KV cache usage: 31.1%, Prefix cache hit rate: 64.9%")
    m = W._STATUS_RE.search(line)
    assert m and m.group(1) == "09-23 07:54:10" and m.group(4) == "4" and m.group(5) == "31.1"


# ── #1812: the 2026-09-28 re-open, recorded as a derived reading ─────────
# The ruling above closes with two triggers that can only fire on a counted day.
# 2026-09-28 fired both, so §10 now carries what that day measured — and because
# the paragraph's whole argument is the churn split and one chat turn's shape,
# every figure in it is recomputed here over the extract it names.
#
# A SECOND extract constant, deliberately. `test_the_counted_reading_is_a_named_
# full_utc_day_with_chat_in_it` pins `EXTRACT`'s window to the day §10 calls the
# counted reading, and 2026-09-28 cannot be that day: its chat turns carry a
# miss, which is criterion (a)'s failure and trigger (i)'s firing. Re-pointing
# `EXTRACT` at it would move the counted reading with it and then demand pass/fail
# marks this item does not ask for.

REOPEN_EXTRACT = ROOT / "tests" / "fixtures" / "vllm_prefix_miss_2026-09-28.json"


@pytest.fixture(scope="module")
def reopen_raw() -> dict:
    return json.loads(REOPEN_EXTRACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reopen(reopen_raw, cfg) -> dict:
    return W.derive(reopen_raw, gate=float(cfg["workers"]["kv_gate"]["max_kv_usage"]))


def _gap_events(ex: dict, gate: float) -> list[dict]:
    """Each miss event that has a gap, with `derive`'s churn rule applied to it.

    `derive` publishes only the aggregate (`misses_gap_churned_free_pool`), so the
    per-event half of §10's split — which of that chat turn's three misses churned,
    and which of them also sat at or over the gate — cannot be read from its output.
    This walks the extract's own rows with `derive`'s own arithmetic: the gap's KV
    peak, the tokens computed inside the gap, and the free blocks that peak implies
    on this pool, churn when the first reaches the last. Nothing in the script is
    edited to expose it, which #1812's step 6 forbids; instead the tally is checked
    against the aggregate, so this walk cannot drift from the rule it copies. #1921 adds
    `n_lines` — how many status lines the gap spans at all — because its paragraph says
    the one chat miss's gap is "a single engine status line", and a claim about a gap that
    thin is only worth checking if the line count comes from the same join.
    """
    samples = ex["kv_samples"]
    (pool,) = ex["pool_tokens"]
    out = []
    for m in ex["misses"]:
        start = m[0] - m[1] / 1000
        if m[6] is None:
            continue
        # Status lines report the 10 s BEFORE their stamp, so a gap's lines run
        # one interval past its end — `derive`'s bound, copied exactly.
        g = [s for s in samples if m[6] <= s[0] <= start + W.STATUS_INTERVAL_S]
        if not g:
            continue
        peak = max(s[1] for s in g) / 100
        computed = sum(s[3] for s in g)
        out.append({"kind": m[5], "gap_s": start - m[6], "peak": peak,
                    "computed": computed, "free_at_peak": round((1 - peak) * pool),
                    "n_lines": len(g),
                    # The miss row's own columns, so an event can be named (#2005).
                    "end": m[0], "iteration": m[2], "tokens": m[3], "cache_read": m[4],
                    "churn": computed >= (1 - peak) * pool})
    d = W.derive(ex, gate=gate)
    assert len(out) == d["miss_events"] and \
        sum(1 for e in out if e["churn"]) == d["misses_gap_churned_free_pool"], (
        "this walk's churn rule no longer agrees with derive's aggregate, so "
        "no per-event figure read off it is the split the doc cites")
    return out


def _reopen_bullet() -> str:
    """§10's eviction bullet — the ruling and the re-open, one bullet, collapsed.

    Scoped to the bullet rather than to §10 so every phrase these nodes demand is
    one the eviction argument itself carries, and so the ruling's pinned wording and
    the re-open that cites it are read from the same stretch of prose the reader
    walks. Raw text is re-split here because `_section` collapses whitespace, which
    would erase the bullet boundaries this needs.
    """
    raw = DOC.read_text(encoding="utf-8")
    start = raw.index("- **Eviction: gate-level KV in most of the gaps")
    end = raw.index("- **Two-request windows, and what (d) counts.**", start)
    return " ".join(raw[start:end].split()).replace("**", "")


def test_the_reopen_extract_is_committed_and_derives_the_cited_day(reopen_raw,
                                                                   reopen):
    """#1812 clause 1: the re-open's day is in the tree, not in /tmp.

    §10's counted reading is built from engine status lines that are byte-rotated
    (about 17 hours per 10 MB file at this load), so without the extract committed
    every figure in the re-open becomes unverifiable inside a week — and the copy
    this was counted into lived on tmpfs, where a reboot, not the ten-day sweep, is
    what deletes it. The four values are the ones #1812 was filed with, so this also
    catches a fixture swapped for one from some other window.
    """
    assert reopen_raw["window"] == ["2026-09-28", "2026-09-29"], (
        "the re-open is pinned to a 00:00→00:00 UTC day, like the counted reading")
    assert reopen["misses"] == 243
    assert reopen["reprefill_tokens"] == 30028609
    assert reopen["chat_turns"] == 10
    assert reopen["chat_turns_with_misses"] == 1
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
         str(REOPEN_EXTRACT.relative_to(ROOT))],
        capture_output=True, text=True)
    assert tracked.returncode == 0, (
        f"{REOPEN_EXTRACT.name} is on disk but not in git: an uncommitted fixture "
        "is gone the moment the working copy is, which is the whole hazard this "
        "item was filed over")
    # And it is the extract the WRITER emitted, not a hand-assembled approximation
    # of one: `--write-extract` serialises with `separators=(",", ":")` plus a
    # trailing newline (scripts/vllm_prefix_miss_window.py's `main`), so re-dumping
    # what the file parses to must reproduce its bytes exactly. Anything edited in a
    # text editor — a row dropped, a number nudged — fails here even though it still
    # parses, and this one-line file is why a citation into it is always line 1.
    raw = REOPEN_EXTRACT.read_text(encoding="utf-8")
    assert raw == json.dumps(reopen_raw, separators=(",", ":")) + "\n", (
        f"{REOPEN_EXTRACT.name} is not byte-for-byte what "
        "`vllm_prefix_miss_window --write-extract` writes for the rows it holds, "
        "so the committed reading has been edited by hand")
    assert len(raw.splitlines()) == 1, (
        "the extract is one compact line; a multi-line fixture would make every "
        "figure in it diffable by hand")


def test_the_reopen_trigger_i_figures_are_the_extract_s(s10, reopen_raw, reopen):
    """#1812 clause 2: trigger (i)'s numbers are read, not typed.

    Trigger (i) — a chat turn carrying a prefix miss at all — is the condition the
    ruling wrote the loss open for, so the paragraph that names it cannot be free to
    say whatever it likes about that turn. Edit either figure in the doc and this
    fails; edit the extract and the doc fails beside it. The turn's own stamp comes
    from the extract's rows too, so the day cannot be quietly re-pointed either.
    """
    p = _reopen_bullet()
    chat_row = [t for t in reopen_raw["turns"] if t[1] == "chat" and (t[2] or 0) > 0]
    assert len(chat_row) == 1 == reopen["chat_turns_with_misses"]
    assert f"{chat_row[0][0]}" in p, "the re-open must name the turn it counted"
    assert f"{reopen['chat_turns_with_misses']} of the day's " \
           f"{reopen['chat_turns']} chat turns" in p, (
        f"the doc's trigger-(i) share is not the extract's "
        f"({reopen['chat_turns_with_misses']} of {reopen['chat_turns']})")
    misses, tokens = reopen["by_kind"]["chat"]
    assert f"{misses} misses cost {tokens:,} re-prefilled tokens" in p, (
        f"that turn's misses and tokens are not what the extract has "
        f"({misses} / {tokens:,})")
    assert f"{reopen_raw['window'][0]} 00:00 → {reopen_raw['window'][1]} 00:00 UTC" \
        in p, ("the re-open has to name the window it counted, in the same shape "
               "the counted reading uses")


def test_the_reopen_budget_side_still_derives_from_6_1(s10, reopen):
    """#1812 clause 3: trigger (ii) is a ratio, and each side keeps its own source.

    The day's side is the new extract; the budget's side is §6.1's own baseline
    sentence through `_per_day_budget`, the same helper that scales the bar's
    aggregate threshold. So neither half can be re-cut by hand — editing the doc's
    97 breaks against §6.1, editing its 243 breaks against the extract. Step 6 keeps
    that baseline sentence untouched precisely so this comparison keeps meaning what
    it says.
    """
    p = _reopen_bullet()
    misses_per_day, tokens_per_day = _per_day_budget(_section(6))
    assert (misses_per_day, tokens_per_day) == (97, 10.3), (
        "§6.1's baseline moved, so every budget figure in §10 needs re-reading")
    day_side = f"{reopen['misses']} misses / {reopen['reprefill_tokens']:,} tokens"
    assert day_side in p, (
        f"the day's side is not the extract's "
        f"({reopen['misses']} / {reopen['reprefill_tokens']:,})")
    assert f"{misses_per_day} misses / {tokens_per_day}M tokens" in p, (
        f"the budget side is not what §6.1's sentence yields "
        f"({misses_per_day} / {tokens_per_day}M)")
    assert reopen["misses"] > misses_per_day \
        and reopen["reprefill_tokens"] > tokens_per_day * 1e6, (
        "the paragraph states trigger (ii) fired; the two derived sides must "
        "agree that it did")


def test_the_reopen_load_mix_caveat_is_derived_from_by_kind(s10, reopen):
    """#1812 clause 4: a trigger fired over a different mix is not an upstream case.

    The budget is a 2026-09-08/09 fleet baseline; this day is overwhelmingly one
    autonomous kind, whose turns are long and whose prompts change. The doc names
    which mix each side was measured over, and the share itself comes out of the
    extract's `by_kind` — including the baseline's own date, which is read from §6.1
    rather than retyped — so a reader cannot lift the ratio into an upstream report.
    """
    p = _reopen_bullet()
    auto_misses, auto_tokens = reopen["by_kind"]["autocode"]
    assert f"{auto_misses} of those {reopen['misses']} misses and " \
           f"{auto_tokens:,} of the {reopen['reprefill_tokens']:,} tokens are " \
           "autocode" in p, (
        f"the caveat's share is not the extract's by_kind: "
        f"{reopen['by_kind']['autocode']}")
    assert f"{auto_misses / reopen['misses']:.0%}" in p, (
        "the rounding has to be the extract's too, not a percentage someone "
        "estimated beside the table")
    m = _BASELINE_RE.search(_section(6))
    assert m, "#1812 needs §6.1's baseline sentence, and step 6 keeps it"
    assert f"2026-09-{m.group(1)}/{m.group(2)}" in p, (
        "the caveat must name the baseline's own date range, derived from §6.1")
    assert "not by itself evidence for the upstream chase" in p, (
        "the caveat has to say what the trigger does NOT establish, or the next "
        "reader treats a 95%-autocode day as a filing")


def test_the_reopen_churn_split_and_chat_attribution_close_the_filing(s10, reopen,
                                                                      reopen_raw,
                                                                      cfg):
    """#1812 clause 5: attribution ran before the chase, and it closed the question.

    Three things, all derived. The split: 89 of the day's 242 miss events churned,
    153 did not, on `derive`'s LRU rule. That chat turn: all three of its misses
    churned, two of them with the gap's KV peak at or over the usage gate, and the
    third at a shallow peak whose implied free blocks the gap's own tokens walked
    anyway. The conclusion: the ruling's condition for an upstream filing is a
    chat-turn miss with NEITHER churn NOR a gate-level KV explanation, this day
    produced none, and the draft-group filing therefore stays closed on its own
    evidence. The phrases the accepted-loss ruling is pinned by, and both trigger
    sentences, survive verbatim — the re-open was written around them, not over them.
    """
    gate = float(cfg["workers"]["kv_gate"]["max_kv_usage"])
    p = _reopen_bullet()
    ev = _gap_events(reopen_raw, gate=gate)
    chat = [e for e in ev if e["kind"] == "chat"]
    assert len(chat) == reopen["by_kind"]["chat"][0] == 3
    assert all(e["churn"] for e in chat), (
        f"the doc says every chat miss churned; the extract says {chat}")
    over = [e for e in chat if e["peak"] >= gate]
    assert len(over) == 2, (
        f"exactly two of the three should sit at or over the gate: "
        f"{[(round(e['peak'], 3), e['gap_s']) for e in chat]}")

    assert f"{reopen['miss_events']} miss events, " \
           f"{reopen['misses_gap_churned_free_pool']} had free-pool churn in their " \
           f"gap and {reopen['miss_events'] - reopen['misses_gap_churned_free_pool']} " \
           "did not" in p, (
        f"the split is not the extract's: "
        f"{reopen['misses_gap_churned_free_pool']} / "
        f"{reopen['miss_events'] - reopen['misses_gap_churned_free_pool']} of "
        f"{reopen['miss_events']}")
    for e in over:
        assert f"{e['peak']:.3f}" in p and f"{e['gap_s']:.1f}" in p, (
            f"an at-or-over-gate chat miss is cited by its own numbers: "
            f"{round(e['peak'], 3)} over {round(e['gap_s'], 1)} s")
    shallow = [e for e in chat if e["peak"] < gate]
    assert len(shallow) == 1
    s0 = shallow[0]
    assert f"{s0['peak']:.3f}" in p and f"{s0['computed']:,}" in p \
        and f"{s0['free_at_peak']:,}" in p, (
        "the one miss below the gate is explained by its tokens against its own "
        f"implied free blocks: {s0['computed']:,} vs {s0['free_at_peak']:,}")
    assert "closes on its own evidence" in p, (
        "the paragraph must state what the attribution concluded, not just its data")
    assert reopen["misses_gap_over_90"] == 0 and \
        "the day's non-churn misses are the same accepted loss" in p, (
        "no miss crossed 0.90, so the accepted loss is the right conclusion — if "
        "that ever changes the sentence and the figure must change together")

    # The ruling's own words, and the triggers that caught the day: all intact.
    for pinned in ("accepted as a bounded, documented loss",
                   "the upstream unannotated draft-group annotation is not chased",
                   "unmeasured rather than merely unfinished",
                   "does not claim reuse is disabled",
                   "capability chase with no measured gain"):
        assert pinned in p, f"the re-open moved a pinned ruling phrase: {pinned}"
    for trigger in ("Re-open conditions, both countable from a future counted "
                    "window", "any counted window in which a chat turn carries a "
                    "prefix miss", "any counted day exceeding §6.1's per-day "
                    "budget of 97 misses / 10.3M re-prefilled tokens"):
        assert trigger in s10, f"a re-open trigger sentence was edited: {trigger}"
    # The re-open is inside the counted reading's span, which may carry exactly one
    # pass/fail mark per criterion: the fired triggers are (i) and (ii), so none of
    # a, b, c or d gained a second mark from this paragraph.
    for letter in CRITERIA:
        _verdict(_counted_reading(s10), letter)


# ── #1919: trigger (ii) is a tripwire, and an (ii)-only day is closed by attribution ──
#
# Both paragraphs above record a day the page attributed, so before #1919 nothing said what an
# unattributed over-budget day earns. `usage.db`'s per-day fleet totals — the standing §10 query
# returns per-day CHAT turns and cannot answer this — put five of the seven complete days from
# 09-23 through 09-29 over §6.1's per-day budget (09-25 and 09-26 are the two under it;
# `tests/fixtures/usage_day_totals_2026-09-23_29.json` holds those bytes, and the node that
# reads them recomputes the five). Two claims the item and my first draft both made are NOT
# supported by those bytes and are not in the rule either: the mix is not autocode-heavy (the
# largest autocode share on any of the seven days is 31.5% of turns, on 09-23, while bench
# leads three of the seven days and review two), and firing trigger (i) is not what the two
# paragraphs represent
# — chat turns carrying a miss appear on four of the seven days, and trigger (i) waits on the
# churn join, whose engine status lines byte-rotate (~17 h per 10 MB file) and so cannot be
# re-joined for a past day unless that day's extract was committed. So the premise the rule
# can rest on
# is the exceedance frequency and the fact that §6.1's fleet mix is unrecorded here. What these
# nodes pin is the rule's WORDING, deliberately not the frequency: a per-day series copied into
# §10 is the item's forbidden shape and the half that rots when the record moves, while the
# witness with history is the fixture, not the page.

_RULE_ANCHOR = "Trigger (ii) is a tripwire"
#: Three shapes are forbidden in the rule's prose, because the item's clause 1 is "no per-day
#: figures copied into the doc": a measured day's own totals in this page's spelling
#: ("243 misses / 30,028,609 tokens"), a day COUNT whether spelled or numeric, and a date
#: sharing a sentence with a fleet total (a date plus "139 misses" is one day of a series even
#: with no comma in it). The positive controls at the end of the node are what make all three
#: checks rather than descriptions.
_DAY_TOTALS = re.compile(r"\d+ misses / \d{1,3},\d{3}")
_DAY_COUNT = re.compile(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
                        r"twelve) of (?:the )?(?:\d+|one|two|three|four|five|six|seven|eight|"
                        r"nine|ten|eleven|twelve) (?:complete |working |full )?days\b", re.I)
_A_DATE = re.compile(r"\b20\d\d-\d\d-\d\d\b|\b\d\d-\d\d/\d\d\b|\b0?9-\d\d\b")
#: Session-kind names, as `scripts.vllm_prefix_miss_window.session_kind` spells them, plus the
#: hyphenated compound this item's wording used. None may appear in the RULE's prose, whose claim
#: is about the standing mix: the committed witness is the only mix data that exists for these
#: days, and on TURNS no kind reaches 50%. Bare "chat" is deliberately NOT in here — (a)'s bullet
#: legitimately talks about chat turns, and a rail that fires on ordinary prose trains the next
#: editor to delete the rail rather than the claim. A day's own MISS split (the 09-28 paragraph's
#: "231 of those 243 misses ... are autocode") is outside this scope and stays writable there.
_SESSION_KIND_CLAIMS = ("autocode", "autotriage", "owedcheck", "bench", "review", "chat-heavy")
_A_TOTAL = re.compile(r"\d+[,.]?\d*[MKk]?\s*(?:misses|tokens)", re.I)

_ROUTINE_EXCEEDANCE = re.compile(
    r"current [\w -]*load mix a counted day exceeds (?:that|the) budget "
    r"(routinely|on most days|more often than not)")
#: The consequence and BOTH halves of it, in one sentence. Split across two sentences does not
#: count: #1918's neighbouring node was filed against a trigger whose halves could be edited
#: apart, and here the halves are the rule — "fires (ii) and not (i)" without "no new §10
#: paragraph" is a rule that still admits a paragraph every second day, and attribution
#: without it is an instruction nothing on the page can be checked against.
_RULE_CONSEQUENCE = re.compile(
    r"a day that fires \(ii\) and not \(i\)[^.]*load-mix[^.]*churn[^.]*attribution"
    r"[^.]*no new §10 paragraph", re.I)
_BASELINE_WINDOW = re.compile(r"Fleet baseline over (\d\d-\d\d/\d\d)")
#: §6.1's baseline sentence, byte-for-byte as it stands. It is the budget's source and #1919
#: requires it untouched, so the whole sentence is pinned rather than re-derived: the numbers
#: alone would still pass with the window, the denominator's meaning, or the "valid data only"
#: qualifier quietly edited, and all three change what 97 / 10.3M would mean.
_S6_BASELINE = ("Fleet baseline over 09-08/09, valid data only: 194 of 1,754 iterations at "
                "≥100k re-prefilled ≥50k uncached tokens — 20.6M tokens, ~34 minutes of "
                "prefill, none at iteration 1–2.")


def _rule_prose() -> str:
    """The tail of the eviction bullet: #1919's rule, whole, collapsed.

    Scoped from the rule's own opening words to the bullet's end, over `_reopen_bullet`'s
    bounds, so what these nodes demand is carried by the RULE. A sentence parked anywhere else
    in §10 would not do — and the 09-28 paragraph already carries a load-mix caveat, so a
    pin reading the whole bullet could be satisfied by prose that predates this item.
    """
    p = _reopen_bullet()
    start = p.find(_RULE_ANCHOR)
    assert start >= 0, (
        "the eviction bullet no longer carries the (ii)-only rule, so the page is back to "
        "deciding per over-budget day whether it earned a paragraph — which is what #1919 "
        "was filed to settle")
    return p[start:]


def test_the_rule_names_its_budget_as_6_1_s_reading_and_says_it_is_routinely_exceeded():
    """#1919 clause 1: the rule grounds the trigger in §6.1's dated baseline, names the
    divisor that produces the two figures, and says a counted day exceeds them routinely at
    the current mix — WITHOUT copying a per-day series in.

    The date range is read out of §6.1's own sentence, so "which fleet" cannot be edited in
    one place and left in the other. The frequency claim is a WORDING pin and not a
    measurement on purpose: the item's check 2 forbids per-day figures in §10, the standing
    query in (a)'s bullet returns chat turns rather than fleet totals and so cannot back the
    claim at all, and a second read-only query would break the one-query rail the same span
    enforces. What the numbers would add is rot: 5 of 7 days is a fact about a record that
    grows, while the rule's claim is about the mix.
    """
    r = _rule_prose()
    s6 = _section(6)
    m = _BASELINE_WINDOW.search(s6)
    assert m, "§6.1 no longer states its baseline over a dated range, so the rule cannot cite it"
    assert f"§6.1's {m.group(1)} fleet baseline" in r, (
        f"the rule must name the fleet its budget is calibrated to, as §6.1 dates it "
        f"({m.group(1)}) — an undated '§6.1's baseline' is the same open-ended-superlative "
        "defect #1918 fixed one bullet earlier")
    days = int(_BASELINE_RE.search(s6).group(2)) - int(_BASELINE_RE.search(s6).group(1)) + 1
    assert f"divided by the {_word(days)} days" in r, (
        f"the two figures are a quotient; the rule has to name its divisor "
        f"({_word(days)}, from §6.1's own range), because that divisor is the input a "
        "re-point has to state to keep meaning the same thing")
    assert _ROUTINE_EXCEEDANCE.search(r), (
        "the rule does not say a counted day exceeds the budget routinely at the current "
        "load mix — without that sentence the trigger reads like a rare exception, which is "
        "what made every firing look like it needed a paragraph")
    assert "load mix" in r, (
        "the sentence must say the exceedance is a function of the LOAD MIX, not of volume "
        "alone — that qualifier is what makes the attribution the consequence owes sufficient, "
        "and the dated fleet it is contrasted against is pinned above out of §6.1's own window")
    # The rule grounds the routine exceedance in §6.1's fleet mix being UNRECORDED on this page,
    # and names no session kind. Measured from usage.db on 2026-09-30 over the seven complete
    # days, and recorded in the committed witness: autocode never reaches half a day's TURNS
    # (max 31.5%, on 09-23) — bench leads 09-25 at 64.6% and 09-26 at 56.5%. So "the current
    # autocode-heavy mix", the wording this item filed, is false as a statement about the mix,
    # and an earlier draft of this paragraph carried it through a GREEN suite run. The fix is a
    # rail, not a memory. Scope is the reason it is safe: the 09-28 paragraph, outside this
    # scope, legitimately records 231 of that day's 243 misses as autocode — a kind can dominate
    # one attributed day's MISS split while never leading that day's turns, which is exactly the
    # distinction a standing-mix claim must not blur.
    named = [k for k in _SESSION_KIND_CLAIMS if k in r]
    assert not named, \
        f"the rule names a session kind ({named}) in a sentence about the STANDING mix; measured " \
        f"on 2026-09-30 no kind leads these days' turns at 50% (autocode's max is 31.5%), so the " \
        "sentence has to rest on §6.1's fleet mix being unrecorded here instead"
    probe = _rule_prose() + " at the current autocode-heavy mix"
    assert [k for k in _SESSION_KIND_CLAIMS if k in probe], \
        "the mix rail matched not even 'autocode-heavy', so the assert above is vacuous"
    # Three shapes count as "a per-day series", because the phrase is not one format: a
    # totals pair in this page's spelling, a day COUNT in either spelling, and a date sharing
    # a sentence with a fleet total (which is one day of a series with its comma removed).
    assert not _DAY_TOTALS.search(r), (
        "the rule copied a per-day totals pair into §10; the frequency is a wording claim and "
        "the numbers belong to usage.db and to the committed witness")
    assert not _DAY_COUNT.search(r), (
        "the rule copied a per-day COUNT into §10 — \"5 of 7 complete days\" and \"five of the "
        "seven complete days\" are both a series, and the count is the half that moves as the "
        "record widens")
    dated = [s for s in _CLAUSE_BREAK.split(r) if _A_DATE.search(s) and _A_TOTAL.search(s)]
    assert not dated, \
        f"a date shares a sentence with a fleet total, which IS a per-day series: {dated[:2]}"
    # Positive controls: a `not` grep says nothing alone, since a pattern can be empty against
    # the corpus in three ways and all three read as clean. Each probe must trip what is named.
    probe = ("5 of 7 complete days ran 243 misses / 30,028,609 tokens. Five of the seven "
             "complete days ran 139 misses. On 2026-09-29 the fleet ran 139 misses.")
    assert _DAY_TOTALS.search(probe), "the totals rail matched not even a totals pair"
    assert _DAY_COUNT.search(probe), \
        "the day-count rail matched neither the numeric nor the spelled count"
    assert [s for s in _CLAUSE_BREAK.split(probe)
            if _A_DATE.search(s) and _A_TOTAL.search(s)], \
        "the date-plus-total rail matched not even an explicit \"On 2026-09-29 the fleet ran " \
        "139 misses\" sentence"


def test_an_only_trigger_two_day_earns_attribution_and_no_new_paragraph():
    """#1919 clause 2: the ruled consequence for the ordinary day, both halves in one
    sentence, and no verdict mark smuggled in with it.

    The one-halved forms are asserted as negatives because they are the two ways this rule
    decays into the status quo: keep the attribution and drop the no-new-paragraph and every
    over-budget day is a paragraph again; keep the paragraph ban and drop the attribution and
    the page forbids the paragraph without saying what the day is instead — which is the
    reading a future reader would settle by filing the day anyway.
    """
    r = _rule_prose()
    assert _RULE_CONSEQUENCE.search(r), (
        "the rule does not state, in one sentence, that a day firing only (ii) earns the "
        f"load-mix and churn attribution AND produces no new §10 paragraph. Rule opens: "
        f"{r[:120]!r}")
    assert not _RULE_CONSEQUENCE.search(
        "a day that fires (ii) and not (i) is closed with no new §10 paragraph"), \
        "a paragraph-ban-only clause passes the pin, so the attribution half could vanish"
    assert not _RULE_CONSEQUENCE.search(
        "a day that fires (ii) and not (i) earns the load-mix and churn attribution"), \
        "an attribution-only clause passes the pin, which is exactly the per-day-paragraph " \
        "behaviour the rule exists to stop"
    assert not _RULE_CONSEQUENCE.search(
        "a day that fires (ii) and not (i) earns the load-mix and churn attribution. "
        "It also produces no new §10 paragraph"), \
        "halves split across two sentences pass the pin, so they could be edited apart"
    assert not re.search(r"\((a|b|c|d)\) (passes|fails)", r), \
        "the rule marks a criterion pass/fail inside the span where _verdict allows exactly " \
        "one such mark — the fired triggers are roman, as the 09-28 paragraph's are"


def test_the_budget_keeps_its_home_in_6_1_and_a_repoint_must_name_its_input():
    """#1919 clause 3: where the budget lives, and what moving it would take.

    The two figures are composed from `_per_day_budget`, so if §6.1's baseline is ever
    re-cut the rule's figures have to move with it and this node is what says so. §6.1 is
    pinned byte-for-byte because the item's whole posture is that the baseline STAYS until
    owed-check rules: the rule is a reading of the budget, not a new budget, and a re-point
    that does not state the divisor it is replacing would leave the two per-day figures
    unattributable to any measurement.
    """
    r = _rule_prose()
    s6 = _section(6)
    assert _S6_BASELINE in s6, (
        "§6.1's baseline sentence changed — it is the budget's source, #1919 keeps it "
        "standing, and its bytes are what (b)'s budget divides")
    assert _per_day_budget(s6) == (97, 10.3), "the derivation the rule quotes is not §6.1's"
    assert _RULE_ANCHOR not in s6, \
        "the rule moved into §6.1, where it would read as part of the baseline it cites"
    assert re.search(r"[Tt]he budget's home (?:is|stays|remains|has been) §6\.1", r), \
        "the rule does not say where the budget lives; without that, 'the budget is stale' " \
        "becomes a licence to edit whichever figure fired"
    assert re.search(r"re-?point", r, re.I) and "current-mix baseline" in r, \
        "the rule does not name what a re-point needs: a measured current-mix baseline"
    assert re.search(r"naming the input (?:that|which) restores", r), \
        "the re-point's own disclosure clause is missing — the item requires the new " \
        "baseline to state what it replaced"
    misses_per_day, tokens_per_day = _per_day_budget(s6)
    assert f"{misses_per_day} misses / {tokens_per_day}M tokens a day" in r, (
        f"the restored figures the re-point must name are §6.1's ({misses_per_day} / "
        f"{tokens_per_day}M), composed here from the derivation rather than hand-typed")


def test_the_repoint_question_is_recorded_as_ruled_not_as_owed():
    """#2008: the rule used to end on an ownership clause — re-pointing the budget "is
    owed-check's to rule" — and owed-check then ruled, on #1919, a closed item nobody reads
    while editing this page. An ownership citation rots when its item closes: left open-tense,
    the next round reads a live question and re-opens a settled one. So the span must carry the
    verdict and its substance, and must not pose the question again.

    Scoped to `_rule_prose()` because the span it grades is the rule's own prose. It used to
    add that the same wording survived once more further down the page about a question that
    was still open there; #2268 rewrote that instance to its ruled answer, and
    `test_the_third_day_question_is_recorded_as_ruled_not_as_owed` now bans the phrase over
    the whole file — so the scoping here is about what this node measures, not about a live
    question living elsewhere.
    """
    r = _rule_prose()
    assert "owed-check's to rule" not in r, \
        "the rule still hands the re-point question to owed-check, which has answered it"
    ruled = [s for s in re.split(r"[.!?;]+\s+", r)
             if "owed-check ruled" in s and "#1919" in s]
    assert ruled, "no sentence records that owed-check ruled the budget question, naming #1919"
    assert re.search(r"no re-?point", r, re.I), "the verdict itself (no re-point) is not stated"
    assert re.search(r"[Rr]outine firing of trigger \(ii\) at the current load mix is not "
                     r"grounds", r), \
        "the span does not say that (ii) firing routinely is not grounds for a re-point"
    assert re.search(r"tripwire and not as a quota", r), \
        "the span does not say what the budget is: a tripwire, not a quota"


def test_the_rule_keeps_2026_09_28_as_the_worked_exemplar_and_the_ruling_verbatim(reopen):
    """#1919 clause 4: the new prose sits AFTER the exemplar and displaces nothing.

    09-28's day-side figures are composed from its committed extract, and the budget side
    from §6.1, so this node fails if the exemplar's arithmetic is edited away as well as if
    the four ruling phrases are reworded. The exemplar-reference half is what makes the rule
    itself falsifiable here rather than merely additive: a rule that quietly promotes itself
    over the exemplar — "this is now the general case, see the rule" — drops the one day on
    the page whose attribution is fully worked, and the sentence that says which day is the
    worked example is a claim about the page, so pinning it is a wording check, not a
    measurement.
    """
    p = _reopen_bullet()
    r = _rule_prose()
    assert "fired both of its triggers" in p, \
        "2026-09-28 is no longer recorded as the day that fired both triggers"
    assert f"{reopen['misses']} misses / {reopen['reprefill_tokens']:,} tokens" in p, \
        "the exemplar's own day-side totals are gone from the bullet, so the rule's " \
        "exemplar claim points at a paragraph that no longer shows its arithmetic"
    for pinned in ("accepted as a bounded, documented loss",
                   "the upstream unannotated draft-group annotation is not chased",
                   "unmeasured rather than merely unfinished",
                   "capability chase with no measured gain"):
        assert pinned in p, f"the new prose moved a pinned ruling phrase: {pinned}"
    assert re.search(r"2026-09-28 [^.]*\b(?:stays|stands|remains|is recorded)\b[^.]*exemplar",
                     r), "the rule does not name 2026-09-28 as the worked exemplar"
    assert not re.search(r"(?:supersed|replaces|displaces|obsolete)s? 2026-09-28", r), \
        "the rule displaces the exemplar it was supposed to keep"


def test_the_rule_breaks_neither_the_one_mark_nor_the_one_query_rule(reading):
    """#1919 clause 5: the rule is inside the span that may carry exactly one pass/fail mark
    per criterion and exactly one read-only population query, and it adds neither.

    Same shape as #1921's rails node, because the failure it guards is the same one and it is
    a failure of THIS prose, not of the pre-existing counts: a rule sentence that cited a
    per-day totals query would leave `test_the_chat_population_is_day_scoped_and_re_derivable_by_one_query`
    red in a file its author never opened, and the way to see that coming is to check the two
    counts the span is capped at from inside the node that changes them.
    """
    r = _rule_prose()
    assert not _CHAT_QUERY.search(r), (
        "the rule cites a second read-only query inside the counted reading's span, where "
        "(a)'s bullet already owns the only one — cite it, or put a fleet-totals query "
        "outside the span (§11) if one is ever needed")
    assert not re.search(r"\((a|b|c|d)\) (passes|fails)", r), \
        "the rule added a pass/fail mark for a criterion"
    marks = sum(len(re.findall(rf"\({letter}\) (passes|fails)", reading))
                for letter in CRITERIA)
    assert marks == 4, (
        f"positive control: the span carries {marks} marks, not one per criterion — the ban "
        "above was tested against a span that had already drifted")
    queries = _CHAT_QUERY.findall(reading)
    assert len(queries) == 1, \
        f"positive control: the span carries {len(queries)} queries, not one"
    for letter in CRITERIA:
        _verdict(reading, letter)


# ── #1921: the 2026-09-29 re-open, and the first day the filing condition is met ─
#
# The ruling #1720 closed with names one exception that would re-open it: a chat-turn
# miss with NEITHER free-pool churn NOR a gate-level KV explanation. 2026-09-28 fired
# both triggers and produced none (#1812). 2026-09-29 fired both triggers and produced
# exactly one, so §10 now carries that day — and it had to be counted the day it
# happened, because the engine status lines (c) and (d) are read from live in an
# 11-file byte rotation, and because the working copies of these extracts have
# historically landed on tmpfs, where a reboot is what deletes them (#1812).
#
# A THIRD extract constant, for the reason #1812 added a second one:
# `test_the_counted_reading_is_a_named_full_utc_day_with_chat_in_it` pins `EXTRACT` to
# the day §10 calls the counted reading, and 09-29 cannot be that day — its chat turns
# carry a miss. Re-pointing `EXTRACT` or `REOPEN_EXTRACT` would move a reading this file
# already grades, so both keep pointing where they do.
#
# Two rails the new paragraph has to respect, and is checked against rather than
# trusted: `_verdict` allows exactly one pass/fail mark per criterion inside the counted
# reading's span, and `test_the_chat_population_is_day_scoped_and_re_derivable_by_one_query`
# allows exactly one read-only population query in it. The paragraph is placed AFTER the
# two-request bullet, so `_reopen_bullet()` — which runs from the eviction bullet up to
# that one — keeps reading only 2026-09-28's text, and #1812's five nodes are untouched.

FILING_EXTRACT = ROOT / "tests" / "fixtures" / "vllm_prefix_miss_2026-09-29.json"
#: The bullet that follows the 09-29 one in §10 (#2005), and so where that one ends.
_REPLICATION_HEADING = "- **The 2026-09-27 count:"
#: The bullet that follows the 09-27 one in §10 (#2248), and so where that one ends.
#: Declared here rather than in the #2248 section below because `_replication_bullet`
#: reads it, and pyflakes rightly refuses to let a graded helper name a constant that is
#: only introduced 400 lines further down.
_SIXTH_HEADING = "- **The 2026-10-01 count:"


@pytest.fixture(scope="module")
def filing_raw() -> dict:
    return json.loads(FILING_EXTRACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def filing(filing_raw, cfg) -> dict:
    return W.derive(filing_raw, gate=float(cfg["workers"]["kv_gate"]["max_kv_usage"]))


def _filing_bullet() -> str:
    """§10's 2026-09-29 paragraph, whole and collapsed.

    Raw text rather than `_section`, and bounded at the next bullet: the span has to
    stop there so a phrase demanded of THIS day cannot be satisfied by another day's
    prose. Until #2005 the next bullet was the draft-group one; it is now 2026-09-27's,
    which walks its own chat misses iteration by iteration and would otherwise answer
    for this day's single event.
    """
    raw = DOC.read_text(encoding="utf-8")
    start = raw.index("- **The 2026-09-29 re-open")
    end = raw.index(_REPLICATION_HEADING, start)
    return " ".join(raw[start:end].split()).replace("**", "")


def test_the_filing_extract_is_committed_and_is_a_third_extract(filing_raw, filing):
    """#1921 clauses 1 and 3: the day is in the tree, and it moved nothing.

    The five hard values are the ones #1921 was filed with, exactly as #1812 pinned
    its own four: they catch a fixture swapped for one from some other window, which
    a byte-identity check alone cannot, since a hand-built extract for a different
    day is still byte-identical to itself. The tracked-file check is the hazard the
    item was filed over — an uncommitted fixture is gone with the working copy. And
    clause 3 is behavioural, not a literal: the older two constants still resolve to
    the days §10 grades them on, and the counted reading still names 09-25.
    """
    assert filing_raw["window"] == ["2026-09-29", "2026-09-30"], (
        "the filing day is pinned to a 00:00→00:00 UTC window, like the other readings")
    assert filing["misses"] == 139
    assert filing["reprefill_tokens"] == 16403836
    assert filing["chat_turns"] == 3
    assert filing["chat_turns_with_misses"] == 1
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
         str(FILING_EXTRACT.relative_to(ROOT))],
        capture_output=True, text=True)
    assert tracked.returncode == 0, (
        f"{FILING_EXTRACT.name} is on disk but not in git: the status lines it was "
        "counted from are in a byte rotation, so an uncommitted fixture means the day "
        "silently stops being countable — which is what #1921 was filed to prevent")
    raw = FILING_EXTRACT.read_text(encoding="utf-8")
    assert raw == json.dumps(filing_raw, separators=(",", ":")) + "\n", (
        f"{FILING_EXTRACT.name} is not byte-for-byte what "
        "`vllm_prefix_miss_window --write-extract` writes for the rows it holds, "
        "so the committed reading has been edited by hand")
    assert len(raw.splitlines()) == 1, (
        "the extract is one compact line; a multi-line fixture would make every "
        "figure in it diffable by hand")
    assert len({EXTRACT, OLD_EXTRACT, REOPEN_EXTRACT, FILING_EXTRACT}) == 4, (
        "the four readings are four fixtures: a re-pointed constant moves a day this "
        "file already grades")
    assert json.loads(EXTRACT.read_text(encoding="utf-8"))["window"] == ["2026-09-25",
                                                                        "2026-09-26"]
    assert json.loads(REOPEN_EXTRACT.read_text(
        encoding="utf-8"))["window"] == ["2026-09-28", "2026-09-29"]
    assert EXTRACT.name in _counted_reading(_section(10)), (
        "the counted reading no longer names 2026-09-25's extract, so 09-29 displaced it")
    assert FILING_EXTRACT.name in _filing_bullet()
    assert f"{FILING_EXTRACT.name}:1" not in _section(10), (
        "§10 cites a line inside the extract, which is one line long — #1812's "
        "attempt-1 finding. Cite the test node that derives the figure instead")


def test_the_filing_day_round_trips_and_the_fixture_outlives_the_lines(filing_raw):
    """The committed copy is the database's, and it is already the only copy of half of it.

    Two surfaces, one claim. The turn and miss rows come back out of `usage.db` and
    the event logs, so re-running `extract` over the same window must return them
    unchanged — a snapshot that no longer matches the record behind it is a fabricated
    one. The KV samples come from the engine's rotating status lines, and there the
    only honest assertion one way: re-reading the disk can lose rows now and can never
    invent one, so whatever the logs still hold must be lines the fixture already
    carries. That subset relation IS the item's point — after the rotation moves past
    this day, the fixture is the only copy of those lines left in the world.
    """
    from app.paths import PRODUCTION_DATA_ROOT

    db = PRODUCTION_DATA_ROOT / "usage.db"
    if not db.exists():
        pytest.skip(f"no usage.db at {db}")
    got = W.extract(filing_raw["window"][0], filing_raw["window"][1],
                    data_root=PRODUCTION_DATA_ROOT)
    if not got["turns"]:
        pytest.skip(f"{db} holds no turns for {filing_raw['window']}")
    assert got["turns"] == filing_raw["turns"], "the counted day's turns are not the database's"
    assert got["misses"] == filing_raw["misses"], "the counted day's misses are not the database's"
    assert set(map(tuple, got["kv_samples"])) <= set(map(tuple, filing_raw["kv_samples"])), (
        "the engine log reports a status line for this window that the committed "
        "extract does not hold, so the fixture is not what `extract` emitted")


def test_the_filing_day_figures_are_the_derivation(filing, filing_raw, cfg):
    """#1921 clause 2: every figure the 09-29 paragraph cites is `derive`'s, at the gate
    `config.yaml` runs.

    Each assertion is a PHRASE, so editing 139, 16,403,836, 3 or 1 in the doc without the
    extract moving fails here — the check #1921 was filed to have. The renderings are the
    doc's own (`:,` for thousands, `.2f` for the KV quartile, `.3f` for a gap peak, `.1f`
    for seconds, `_n` where the page writes a comma-grouped count), so a re-rounded or
    re-comma'd figure cannot pass as "the same number", and the two sides of trigger (ii)
    keep their separate sources: the day's from this extract, the budget's from §6.1's own
    sentence through `_per_day_budget`.
    """
    p = _filing_bullet()
    gate = float(cfg["workers"]["kv_gate"]["max_kv_usage"])
    assert f"{filing_raw['window'][0]} 00:00 → {filing_raw['window'][1]} 00:00 UTC" in p, (
        "the paragraph must name its window in the shape the other readings use")
    assert f"{filing['chat_turns_with_misses']} of the day's {filing['chat_turns']} chat turns" in p
    assert f"{filing['misses']} misses / {filing['reprefill_tokens']:,} tokens" in p
    misses_per_day, tokens_per_day = _per_day_budget(_section(6))
    assert f"{misses_per_day} misses / {tokens_per_day}M tokens" in p
    assert filing["misses"] > misses_per_day \
        and filing["reprefill_tokens"] > tokens_per_day * 1e6, (
        "the paragraph says the day is over the budget on both sides; the two derived "
        "sides have to agree that it is")
    auto_misses, auto_tokens = filing["by_kind"]["autocode"]
    assert (f"{auto_misses} of those {filing['misses']} misses and {auto_tokens:,} of the "
            f"{filing['reprefill_tokens']:,} tokens are autocode") in p
    assert f"{auto_misses / filing['misses']:.0%}" in p, "the share's rounding is the extract's"
    m = _BASELINE_RE.search(_section(6))
    assert m and f"2026-09-{m.group(1)}/{m.group(2)}" in p, (
        "the caveat must name the baseline's own date range, derived from §6.1")
    assert f"{_n(filing['kv_samples'])} such lines" in p
    assert (f"KV p50 {filing['kv_p50']:.2f} / p90 {filing['kv_p90']:.2f} / max "
            f"{filing['kv_max']:.2f}") in p
    assert f"`Running:` {_n(filing['running_p50'])} at the median" in p
    assert f"of {_n(filing['two_request_windows'])} lines with two requests resident" in p
    assert f"median {filing['two_request_tok_s_p50']} tok/s" in p
    assert f"slowest {filing['two_request_tok_s_min']}" in p
    assert f"the {filing['two_request_under_stall']} under the bar's " \
           f"{W.STALL_TOK_S:.0f} tok/s" in p
    assert f"{filing['two_request_under_stall_cold_in_flight']} of them with a cold " \
           "re-admission" in p
    assert f"{filing['two_request_under_stall_cold_prefill']} with a chunked prefill" in p
    assert f"gap p50 {filing['gap_s_p50']:.1f} s" in p
    assert f"p50 {filing['miss_kv_gap_p50']:.3f} / p90 {filing['miss_kv_gap_p90']:.3f}" in p
    assert f"max {filing['miss_kv_gap_max']:.3f}" in p
    assert (f"{filing['misses_gap_over_gate']} of {filing['miss_events']} at or over the "
            f"{gate:.2f} gate") in p
    assert (f"of the day's {filing['miss_events']} miss events, "
            f"{filing['misses_gap_churned_free_pool']} had free-pool churn in their gap and "
            f"{filing['miss_events'] - filing['misses_gap_churned_free_pool']} did not") in p
    assert ("none over 0.90" in p) == (filing["misses_gap_over_90"] == 0), (
        "the 0.90 sentence and the derived count disagree")
    assert f"{_n(filing['pool_tokens'])}-token pool" in p, (
        "the free-pool arithmetic is against this pool, so the paragraph names it")


def test_the_filing_paragraph_breaks_neither_the_one_mark_nor_the_one_query_rule(reading):
    """#1921 clause 4: the paragraph lives inside the span that may carry exactly one
    mark per criterion and exactly one population query — and it carries neither.

    The positive control beside each ban is what makes this a check rather than a
    description: the same two patterns DO find the four marks and the one query the page
    already has. Without it, a paragraph that had been deleted, or a pattern that had
    drifted off the page's spelling, would satisfy the bans just as well.
    """
    p = _filing_bullet()
    for letter in CRITERIA:
        assert not re.search(rf"\({letter}\) (passes|fails)", p), (
            f"the 09-29 paragraph marks ({letter}) pass/fail inside the counted reading's "
            "span, where _verdict allows exactly one such mark — write prose verdicts, as "
            "the 09-28 paragraph does")
    marks = sum(len(re.findall(rf"\({letter}\) (passes|fails)", reading)) for letter in CRITERIA)
    assert marks == 4, (
        f"positive control: the span carries {marks} marks, not one per criterion, so the "
        "ban above was tested against nothing")
    assert not _CHAT_QUERY.search(p), (
        "the 09-29 paragraph cites a second read-only population query; cite the one in "
        "(a)'s bullet instead, as this paragraph's own rule is to quote the extract")
    queries = _CHAT_QUERY.findall(reading)
    assert len(queries) == 1, (
        f"positive control: the counted reading carries {len(queries)} queries, not one")


def test_the_filing_day_verdicts_are_derived_and_the_two_reopens_agree(filing, filing_raw,
                                                                      reopen, reopen_raw,
                                                                      cfg, s10):
    """#1921's Change 3: the verdicts are read off the derivation, 09-25 is not displaced,
    and "produced one" does not contradict 09-28's "produced none".

    The verdict sentences are prose, not marks, so they are pinned the way
    `test_the_eviction_branch_named_is_the_one_the_gaps_support` pins its heading: as a
    biconditional against the numbers. The disagreement between the two re-open
    paragraphs is then not a wording matter at all — it comes out of the two extracts,
    which say all three of 09-28's chat misses churned and that 09-29's one did not and
    never reached the gate.
    """
    gate = float(cfg["workers"]["kv_gate"]["max_kv_usage"])
    p = _filing_bullet()
    misses_per_day, tokens_per_day = _per_day_budget(_section(6))
    assert ("criterion (a) does not hold" in p) == (filing["chat_turns_with_misses"] > 0), (
        f"(a)'s prose verdict disagrees with chat_turns_with_misses="
        f"{filing['chat_turns_with_misses']}")
    assert ("criterion (b) does not hold" in p) == (
        filing["misses"] > misses_per_day
        and filing["reprefill_tokens"] > tokens_per_day * 1e6), (
        "(b)'s prose verdict disagrees with the day against §6.1's budget")
    assert ("(c) and (d) both hold" in p) == (
        filing["kv_p50"] < gate and filing["two_request_under_stall_not_cold"] == 0), (
        "(c)/(d)'s prose verdict disagrees with the KV median and (d)'s counted lines")
    assert "does not displace 2026-09-25 as the counted reading" in p
    assert "produced none" in _reopen_bullet() and "produced one" in p, (
        "each re-open states its own conclusion; neither may be softened into agreeing "
        "with the other")
    r = [e for e in _gap_events(reopen_raw, gate=gate) if e["kind"] == "chat"]
    f = [e for e in _gap_events(filing_raw, gate=gate) if e["kind"] == "chat"]
    assert len(r) == 3 and all(e["churn"] for e in r), (
        f"09-28's paragraph rests on all three of its chat misses churning: {r}")
    assert len(f) == 1 and not f[0]["churn"] and f[0]["peak"] < gate, (
        f"09-29's paragraph rests on its one chat miss having neither: {f}")
    for letter in CRITERIA:
        _verdict(_counted_reading(s10), letter)


def test_the_filing_chat_miss_is_one_event_with_its_measured_attribution(filing, filing_raw,
                                                                        cfg):
    """#1921 clause 5: the day's one chat miss, attributed by measurement, row by row.

    `_gap_events` walks `derive`'s own churn rule per event and asserts its tally
    against `derive`'s aggregate, so these are the same numbers the criteria are marked
    on and not a second opinion. The filing condition is `not churn AND peak under the
    gate`, and the paragraph has to carry all six measured quantities #1921 names —
    iteration, prompt tokens, cache_read, gap length, gap KV peak against the gate, and
    the tokens computed inside the gap against the free pool that peak implies — as one
    event on one day: a second chat miss, or a different turn, would make the sentence a
    different claim.
    """
    gate = float(cfg["workers"]["kv_gate"]["max_kv_usage"])
    p = _filing_bullet()
    chat = [e for e in _gap_events(filing_raw, gate=gate) if e["kind"] == "chat"]
    row = [m for m in filing_raw["misses"] if m[5] == "chat"]
    assert len(row) == len(chat) == filing["by_kind"]["chat"][0] \
        == filing["chat_turns_with_misses"] == 1, (
        f"the paragraph is about ONE event; the extract has rows={row} events={chat} "
        f"by_kind={filing['by_kind'].get('chat')}")
    r0, = row
    e0, = chat
    assert not e0["churn"] and e0["peak"] < gate, (
        "the condition the ruling named is neither churn nor a gate-level peak, and this "
        f"event has one of them: peak={e0['peak']} churn={e0['churn']}")
    assert f"iteration {r0[2]}" in p
    assert f"{r0[3]:,} tokens" in p
    assert f"cache_read {r0[4]}" in p
    assert f"{e0['gap_s']:.1f} s" in p
    assert ("a single engine status line" in p) == (e0["n_lines"] == 1), (
        f"the gap spans {e0['n_lines']} status lines; the paragraph's claim about how thin "
        "the evidence is has to follow that")
    assert f"{e0['peak']:.3f}" in p
    assert f"under the {gate:.2f} gate" in p
    assert f"{e0['computed']:,}" in p
    assert f"{e0['free_at_peak']:,}" in p
    turn = [t for t in filing_raw["turns"] if t[1] == "chat" and (t[2] or 0) > 0]
    assert len(turn) == 1, f"the extract has {len(turn)} chat turns carrying a miss"
    assert f"`{turn[0][0]}`" in p, "the paragraph must name the one turn it counted"
    # The stamp is named twice — once at trigger (i), once in the attribution — so
    # naming it somewhere on the page is not enough: the sentence that carries the
    # measured attribution has to carry the stamp of THAT turn, or the iteration,
    # token count and gap below could be quietly re-pointed at a different one.
    attrib = [s for s in p.split(". ") if "iteration" in s]
    assert len(attrib) == 1 and turn[0][0] in attrib[0], (
        "the sentence making the iteration claim does not name the turn the extract "
        f"counted ({turn[0][0]}), so the attribution is not pinned to an event")
    assert "neither free-pool churn nor a gate-level KV explanation" in p
    assert "one event on one day" in p
    assert "first counted day on which the condition is met" in p, (
        "the claim is that this is the first COUNTED day to meet the condition — not the "
        "only day, which the uncounted 09-27 has not been checked against")


#: The frozen per-day witness behind the routine-exceedance claim (#1919 clause 6): the day
#: totals `usage.db` gave on 2026-09-30, kept because the row churn the same database keeps
#: makes the same totals a moving target with no history.
_DAY_WITNESS = ROOT / "tests" / "fixtures" / "usage_day_totals_2026-09-23_29.json"


def test_the_routine_exceedance_witness_is_committed_and_states_the_premise(derived):
    """#1919 clause 6: the frequency claim's bytes are in the tree, and the premise comes out
    of those bytes plus §6.1's parsed budget — not out of a chat session's memory.

    The seam is doc prose to committed evidence, which is the seam this whole module exists to
    close: §10 copies no per-day figure at all (clause 1's three rails above refuse it), so
    without this fixture the sentence under the rule — five of the seven complete days over
    budget — would be a claim with nothing to re-measure against once usage.db's rows churn.
    Two independent sources meet here: the fixture's own day totals, and the threshold parsed
    from §6.1's baseline sentence by `_per_day_budget`, which the rule's budget half already
    asserts. The node cannot check the fixture against the live database — usage.db is runtime
    data, outside any round's write surface and outside this repo — and does not claim to: the
    cross-check it CAN run is the committed 09-25 extract, whose chat-turn count is an
    independent measurement of one of these seven days.
    """
    import subprocess

    tracked = subprocess.run(["git", "ls-files", "--error-unmatch",
                              str(_DAY_WITNESS.relative_to(ROOT))],
                             cwd=ROOT, capture_output=True, text=True)
    assert tracked.returncode == 0, (
        f"{_DAY_WITNESS.name} is on disk but not in git: the root .gitignore ends every new "
        "JSON, and a witness that only exists in one worktree is not evidence — that is "
        "exactly how #1812's 09-28 reading went missing")
    wit = json.loads(_DAY_WITNESS.read_text(encoding="utf-8"))
    days = wit["days"]
    assert [d["day"] for d in days] == [f"2026-09-{n:02d}" for n in range(23, 30)], (
        "the witness must hold the complete days its claim counts; a window edited at either "
        "end is how a 5-of-7 becomes a 5-of-6 without any figure changing")
    miss_budget, tok_budget = _per_day_budget(_section(6))
    over = [d["day"] for d in days
            if d["misses"] > miss_budget or d["reprefill_tokens"] > tok_budget * 1_000_000]
    assert len(over) == 5, (
        f"the premise under the (ii)-only rule is that the budget is exceeded routinely; the "
        f"committed day totals give {len(over)} of {len(days)} days over "
        f"{miss_budget} misses / {tok_budget}M tokens")
    assert [d["day"] for d in days if d["day"] not in over] == ["2026-09-25", "2026-09-26"], \
        "the two days under budget are part of the premise; if they moved, 'routinely' is " \
        "the wrong word and the rule's sentence has to change with the data"
    busiest = max(days, key=lambda d: d["misses"])
    assert busiest["day"] == "2026-09-24", (
        f"the worst day in the witness moved to {busiest['day']}, and #1919's item text quotes "
        "09-24 as the 318 / 32.23M day — the rule does not cite it, but the ruling that "
        "re-points the budget will")
    cross = [d for d in days if d["day"] == "2026-09-25"][0]
    assert cross["chat_turns"] == derived["chat_turns"] and \
           cross["chat_turns_with_misses"] == derived["chat_turns_with_misses"], (
        f"the witness says 09-25 carried {cross['chat_turns']} chat turns with "
        f"{cross['chat_turns_with_misses']} carrying a miss, but the committed 09-25 extract "
        f"derives {derived['chat_turns']} / {derived['chat_turns_with_misses']}: two copies of "
        "the same day disagree, so neither can be trusted for the premise")
    assert [d["day"] for d in days if d["chat_turns_with_misses"]] == [
            "2026-09-24", "2026-09-27", "2026-09-28", "2026-09-29"], (
        "four days in the witness carry a chat turn with a miss, which is why the rule cannot "
        "claim the two paragraphs are 'the days that fired trigger (i)' — chat misses are "
        "trigger (i)'s candidate set, not its verdict, which needs the churn join")
    # The mix half of the premise, measured rather than assumed. What is FALSE is
    # "autocode-heavy": the largest autocode share on any of these seven days is 0.315 (09-23),
    # while bench leads 09-25 at 0.646 and 09-26 at 0.565. So days ARE sometimes led by one
    # session kind — just never by autocode — which is why the rule's sentence names no kind and
    # rests the exceedance on §6.1's fleet being unrecorded here instead.
    assert max(d["autocode_share"] for d in days) < 0.5, (
        "autocode leading half a day's turns would put a 'current autocode-heavy mix' wording "
        "back in play; measured max on 2026-09-30 was 0.315, on 09-23")
    assert sorted(d["top_session_kind"] for d in days).count("bench") == 3 and \
           sorted(d["top_session_kind"] for d in days).count("review") == 2, \
        "bench leading three of the seven days and review two is the load-mix fact the rule " \
        "replaces 'autocode-heavy' with; if the leaders moved, the sentence needs re-measuring"


# ── #2005: 2026-09-27 is counted, and the filing condition replicates on it ───────
#
# #1921's paragraph ended on a deferral: the condition had been met on one counted day,
# and an earlier day whose chat turns also carry misses had never been counted. That day
# is 2026-09-27, and its 7,039 engine status lines were five rotations from deletion when
# this extract was taken. It is a FIFTH constant for the reason the third and fourth were
# added: every other one is pinned to a day §10 already grades.
#
# What the day turns out to hold is not one event but seventeen, so the walk is asserted
# event by event and the paragraph's own tally (churned / at the gate / neither) is
# recomputed, not quoted. One of the "neither" events sits 0.005 under the gate; the
# paragraph has to say so and the verdict has to survive without it, which is a
# biconditional below and not a wording pin alone.

REPLICATION_EXTRACT = ROOT / "tests" / "fixtures" / "vllm_prefix_miss_2026-09-27.json"


@pytest.fixture(scope="module")
def replication_raw() -> dict:
    return json.loads(REPLICATION_EXTRACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def replication(replication_raw, cfg) -> dict:
    return W.derive(replication_raw, gate=float(cfg["workers"]["kv_gate"]["max_kv_usage"]))


def _replication_bullet() -> str:
    """§10's 2026-09-27 paragraph, whole and collapsed.

    Bounded at #2248's 2026-10-01 bullet for the reason #2005 bounded #1921's at this
    day's heading: the span has to stop at the next bullet so a phrase demanded of
    09-27 cannot be answered by a later day's prose. Until #2248 the next bullet was the
    draft-group one; it is now 2026-10-01's, which walks its own chat miss and would
    otherwise answer for this day's four.
    """
    raw = DOC.read_text(encoding="utf-8")
    start = raw.index(_REPLICATION_HEADING)
    end = raw.index(_SIXTH_HEADING, start)
    return " ".join(raw[start:end].split()).replace("**", "")


def _chat_events(raw: dict, gate: float) -> list[dict]:
    return [e for e in _gap_events(raw, gate=gate) if e["kind"] == "chat"]


def test_the_replication_extract_is_committed_and_is_a_fifth_extract(replication_raw,
                                                                    replication):
    """#2005 clause 1: the day is in the tree, byte-for-byte what the script writes, and
    it moved none of the four readings already graded."""
    assert replication_raw["window"] == ["2026-09-27", "2026-09-28"]
    assert replication["misses"] == 201
    assert replication["reprefill_tokens"] == 25082453
    assert replication["chat_turns"] == 4
    assert replication["chat_turns_with_misses"] == 2
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
         str(REPLICATION_EXTRACT.relative_to(ROOT))],
        capture_output=True, text=True)
    assert tracked.returncode == 0, (
        f"{REPLICATION_EXTRACT.name} is on disk but not in git; its status lines leave the "
        "rotation about 2026-10-06, after which an uncommitted fixture is the day lost")
    raw = REPLICATION_EXTRACT.read_text(encoding="utf-8")
    assert raw == json.dumps(replication_raw, separators=(",", ":")) + "\n", (
        f"{REPLICATION_EXTRACT.name} is not byte-for-byte what `--write-extract` writes")
    assert len(raw.splitlines()) == 1
    # The 4-way assert in the filing node stays true over four constants whatever this
    # one is, so the 5-way set is asserted here.
    five = {EXTRACT, OLD_EXTRACT, REOPEN_EXTRACT, FILING_EXTRACT, REPLICATION_EXTRACT}
    assert len(five) == 5, "a re-pointed constant moves a day this file already grades"
    windows = {c.name: json.loads(c.read_text(encoding="utf-8"))["window"][0] for c in five}
    assert windows == {
        EXTRACT.name: "2026-09-25", OLD_EXTRACT.name: "2026-09-23",
        REOPEN_EXTRACT.name: "2026-09-28", FILING_EXTRACT.name: "2026-09-29",
        REPLICATION_EXTRACT.name: "2026-09-27"}, windows
    assert EXTRACT.name in _counted_reading(_section(10)), (
        "the counted reading no longer names 2026-09-25's extract")
    p = _replication_bullet()
    assert REPLICATION_EXTRACT.name in p
    assert "does not displace 2026-09-25 as the counted reading" in p
    assert f"{REPLICATION_EXTRACT.name}:1" not in _section(10)


def test_the_replication_day_round_trips_and_the_fixture_outlives_the_lines(replication_raw):
    """The committed turns and misses are the database's; the status lines the logs still
    hold are a subset of the fixture's. After the rotation passes this day the second
    half is the only assertion left, and the fixture the only copy."""
    from app.paths import PRODUCTION_DATA_ROOT

    db = PRODUCTION_DATA_ROOT / "usage.db"
    if not db.exists():
        pytest.skip(f"no usage.db at {db}")
    got = W.extract(replication_raw["window"][0], replication_raw["window"][1],
                    data_root=PRODUCTION_DATA_ROOT)
    if not got["turns"]:
        pytest.skip(f"{db} holds no turns for {replication_raw['window']}")
    assert got["turns"] == replication_raw["turns"]
    assert got["misses"] == replication_raw["misses"]
    assert set(map(tuple, got["kv_samples"])) <= set(map(tuple, replication_raw["kv_samples"]))


def test_the_replication_day_figures_are_the_derivation(replication, replication_raw, cfg):
    """#2005 clause 2: every figure the 09-27 paragraph prints is `derive`'s at the gate
    `config.yaml` runs, each asserted as a phrase in the doc's own rendering."""
    p = _replication_bullet()
    d = replication
    gate = float(cfg["workers"]["kv_gate"]["max_kv_usage"])
    assert (f"{replication_raw['window'][0]} 00:00 → {replication_raw['window'][1]} "
            "00:00 UTC") in p
    assert f"{d['chat_turns_with_misses']} of the day's {d['chat_turns']} chat turns" in p
    chat_misses, chat_tokens = d["by_kind"]["chat"]
    assert f"{chat_misses} misses / {chat_tokens:,} tokens" in p
    assert f"{d['misses']} misses / {d['reprefill_tokens']:,} tokens" in p
    misses_per_day, tokens_per_day = _per_day_budget(_section(6))
    assert f"{misses_per_day} misses / {tokens_per_day}M tokens" in p
    assert d["misses"] > misses_per_day and d["reprefill_tokens"] > tokens_per_day * 1e6
    auto_misses, auto_tokens = d["by_kind"]["autocode"]
    assert (f"{auto_misses} of those {d['misses']} misses and {auto_tokens:,} of the "
            f"{d['reprefill_tokens']:,} tokens are autocode") in p
    assert f"({auto_misses / d['misses']:.0%})" in p
    m = _BASELINE_RE.search(_section(6))
    assert m and f"2026-09-{m.group(1)}/{m.group(2)}" in p
    assert f"{_n(d['kv_samples'])} such lines" in p
    assert f"KV p50 {d['kv_p50']:.2f} / p90 {d['kv_p90']:.2f} / max {d['kv_max']:.2f}" in p
    assert f"`Running:` {_n(d['running_p50'])} at the median" in p
    assert f"of {_n(d['two_request_windows'])} lines with two requests resident" in p
    assert f"median {d['two_request_tok_s_p50']} tok/s" in p
    assert f"slowest {d['two_request_tok_s_min']}" in p
    assert f"the {d['two_request_under_stall']} under the bar's {W.STALL_TOK_S:.0f} tok/s" in p
    assert f"{d['two_request_under_stall_cold_in_flight']} of them with a cold re-admission" in p
    assert f"{d['two_request_under_stall_cold_prefill']} with a chunked prefill" in p
    assert f"gap p50 {d['gap_s_p50']:.1f} s" in p
    assert f"p50 {d['miss_kv_gap_p50']:.3f} / p90 {d['miss_kv_gap_p90']:.3f}" in p
    assert f"max {d['miss_kv_gap_max']:.3f}" in p
    assert (f"{d['misses_gap_over_gate']} of {d['miss_events']} at or over the "
            f"{gate:.2f} gate") in p
    assert f"{d['misses_gap_over_90']} of them over 0.90" in p
    assert (f"of the day's {d['miss_events']} miss events, "
            f"{d['misses_gap_churned_free_pool']} had free-pool churn in their gap and "
            f"{d['miss_events'] - d['misses_gap_churned_free_pool']} did not") in p
    assert f"{_n(d['pool_tokens'])}-token pool" in p
    # The prose verdicts, as biconditionals against the numbers.
    assert ("criterion (a) does not hold" in p) == (d["chat_turns_with_misses"] > 0)
    assert ("criterion (b) does not hold" in p) == (
        d["misses"] > misses_per_day and d["reprefill_tokens"] > tokens_per_day * 1e6)
    assert ("(c) and (d) both hold" in p) == (
        d["kv_p50"] < gate and d["two_request_under_stall_not_cold"] == 0)
    for turn in (t for t in replication_raw["turns"] if t[1] == "chat" and (t[2] or 0) > 0):
        assert f"`{turn[0]}`" in p, f"the paragraph must name the chat turn {turn[0]}"


def test_the_replication_paragraph_breaks_neither_the_one_mark_nor_the_one_query_rule(reading):
    """#2005 clause 3: no pass/fail mark and no second population query in the new
    paragraph, with the positive control that the span still carries the four and the one."""
    p = _replication_bullet()
    assert p in " ".join(reading.replace("**", "").split()), (
        "the 09-27 paragraph is not inside the counted reading's span, so the two rails "
        "below would be checked against text they do not govern")
    for letter in CRITERIA:
        assert not re.search(rf"\({letter}\) (passes|fails)", p), letter
    marks = sum(len(re.findall(rf"\({letter}\) (passes|fails)", reading)) for letter in CRITERIA)
    assert marks == 4, f"positive control: {marks} marks in the span"
    assert not _CHAT_QUERY.search(p)
    assert len(_CHAT_QUERY.findall(reading)) == 1


def test_each_replication_chat_miss_is_attributed_and_the_marginal_one_carries_no_weight(
        replication, replication_raw, cfg):
    """#2005 clause 4: the day's chat miss events, walked one by one with `derive`'s own
    churn rule, and the tally the paragraph prints recomputed from that walk.

    An event meets the filing condition when it has neither free-pool churn in its gap
    nor a gap KV peak at or over the gate. Each such event is asserted with its
    iteration, prompt size, gap length, gap peak, tokens computed in the gap and the
    free blocks that peak implies — all six in ONE clause of the paragraph, so a figure
    cannot be re-pointed at a different event. The event 0.005 under the gate is named,
    and the verdict is asserted to hold with it removed.
    """
    gate = float(cfg["workers"]["kv_gate"]["max_kv_usage"])
    p = _replication_bullet()
    chat = _chat_events(replication_raw, gate)
    assert len(chat) == replication["by_kind"]["chat"][0]
    churned = [e for e in chat if e["churn"]]
    at_gate = [e for e in chat if not e["churn"] and e["peak"] >= gate]
    neither = [e for e in chat if not e["churn"] and e["peak"] < gate]
    assert f"of the {len(chat)} chat miss events" in p
    assert (f"{len(churned)} had free-pool churn in their gap, {len(at_gate)} more had a gap "
            f"KV peak at or over the gate, and {len(neither)} had neither") in p
    assert f"Those {len(neither)} are all in the turn" in p

    # Which turn: the chat turn whose stamp is the first at or after each event's end.
    from datetime import datetime, timezone
    stamps = sorted(t[0] for t in replication_raw["turns"] if t[1] == "chat" and (t[2] or 0) > 0)

    def turn_of(e):
        end = datetime.fromtimestamp(e["end"], timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        return next(s for s in stamps if s >= end)

    owners = {turn_of(e) for e in neither}
    assert len(owners) == 1 and f"all in the turn `{owners.pop()}`" in p, owners

    clauses = re.split(r"; | they are: ", p)
    for e in neither:
        mine = [c for c in clauses if re.search(rf"\biteration {e['iteration']},", c)]
        assert len(mine) == 1, f"iteration {e['iteration']} is named in {len(mine)} clauses"
        c = mine[0]
        for said in (f"a prompt of {e['tokens']:,} tokens", f"a gap of {e['gap_s']:.1f} s",
                     f"gap KV peak {e['peak']:.3f}",
                     f"{e['computed']:,} tokens computed in the gap",
                     f"against {e['free_at_peak']:,} free blocks"):
            assert said in c, f"iteration {e['iteration']}: {said!r} not in {c!r}"
        assert e["cache_read"] == 0
    assert "every one a prompt with cache_read 0" in p
    # No event that churned or sat at the gate is listed among them.
    for e in churned + at_gate:
        assert not any(f"gap KV peak {e['peak']:.3f}, with {e['computed']:,} tokens" in c
                       for c in clauses), e

    # The marginal event: named with its margin, and carrying no weight.
    marginal = [e for e in neither if gate - e["peak"] < 0.01]
    assert len(marginal) == 1, marginal
    m0, = marginal
    assert f"The iteration {m0['iteration']} event is marginal" in p
    assert (f"its peak of {m0['peak']:.3f} is {gate - m0['peak']:.3f} under the "
            f"{gate:.2f} gate") in p
    assert "the verdict does not rest on it" in p
    clear = [e for e in neither if e is not m0]
    assert clear and all(gate - e["peak"] > 0.05 for e in clear), (
        "with the marginal event set aside no event is left clear of the gate, so the "
        "verdict WOULD rest on it and the paragraph's sentence is false")
    assert f"{_word(len(clear))} events remain" in p
    emptiest = min(clear, key=lambda e: e["peak"])
    assert f"{1 - emptiest['peak']:.0%} free with {emptiest['computed']:,} tokens computed" in p


def test_the_replication_outcome_is_stated_where_the_deferral_and_the_ruling_stood(
        replication_raw, filing_raw, cfg):
    """#2005 clause 5: §10 states the outcome the derivation yields, and the two places
    that deferred to it now carry it and name both committed extracts.

    The outcome is computed here from the two extracts — each has at least one chat miss
    with neither churn nor a gate-level peak — and the three statements are required to
    agree with it. The deferral's old wording and the 'decided rather than parked'
    wording are both required gone, since each asserts a state the record has left.
    """
    gate = float(cfg["workers"]["kv_gate"]["max_kv_usage"])

    def unexplained(raw):
        return [e for e in _chat_events(raw, gate) if not e["churn"] and e["peak"] < gate]

    met_on_both = bool(unexplained(replication_raw)) and bool(unexplained(filing_raw))
    assert met_on_both, "one of the two extracts no longer carries an unexplained chat miss"
    p = _replication_bullet()
    assert ("the condition is met on two independent counted days" in p) == met_on_both
    assert "neither free-pool churn nor a gate-level KV explanation" in p
    assert (f"2026-09-29 on {_word(len(unexplained(filing_raw)))} event and this day on "
            f"{_word(len(unexplained(replication_raw)) - 1)}, or "
            f"{_word(len(unexplained(replication_raw)))} with the marginal one") in p
    # An exclusion, not a cause; and not filed.
    assert "an exclusion and not a cause" in p
    assert "it is not filed" in p
    assert not re.search(r"github\.com/vllm-project/vllm/issues/\d+", _section(10)), (
        "§10 names an upstream issue; the 'not filed' sentences must go when it does")

    both = (FILING_EXTRACT.name, REPLICATION_EXTRACT.name)
    ruling = _reopen_bullet()
    ruling = ruling[:ruling.index(_RULE_ANCHOR)]
    assert "decided rather than parked" not in ruling
    assert "has since been met on two independent counted days" in ruling
    assert "the re-open has fired and the upstream report is owed" in ruling
    assert "It has not been filed" in ruling
    for name in both:
        assert name in ruling, f"the accepted-loss paragraph does not name {name}"

    f = _filing_bullet()
    assert "is owed-check's to rule" not in f and "has never been counted" not in f
    assert "it now stands on two independent days and two committed extracts" in f
    assert REPLICATION_EXTRACT.name in f and FILING_EXTRACT.name in f


# ── #2248: 2026-10-01 is counted, and the filing condition replicates a third time ──
#
# #2005 left the condition standing on two counted days and two committed extracts. The
# standing daily job (`vllm-prefix-miss-daily`) then caught four more days while their
# status lines were still in the rotation — the surface #2006's ruling gave it — and exactly
# one of the four has a chat turn carrying a miss: 2026-10-01, whose single event has
# neither free-pool churn nor a gate-level gap. So this is a SIXTH extract constant for the
# reason the third, fourth and fifth were added: every other one is pinned to a day §10
# already grades. It is the first counted-day fixture that came off the standing job rather
# than a round racing the rotation, and re-deriving it from the logs here reproduces the
# job's bytes exactly, which is the round trip this file's third node checks.
#
# The day also settles the question the item was filed with and could not answer: whether a
# qualifying gap can be minutes long, and so whether a tenth of a second is the same
# condition at all. On the record it cannot. Every one of the six qualifying events across
# the three days sits in a sub-second gap spanned by ONE status line; the minutes-long
# figure in the record is a DAY's gap p50 (100.1 s on 2026-09-29), a median over that day's
# miss events, and no qualifying event sits in a gap like that. The last node below derives
# both halves from the three fixtures with no figure written into it, and keeps the day-side
# median as the positive control that the contrast is a real difference.

SIXTH_EXTRACT = ROOT / "tests" / "fixtures" / "vllm_prefix_miss_2026-10-01.json"


@pytest.fixture(scope="module")
def sixth_raw() -> dict:
    return json.loads(SIXTH_EXTRACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sixth(sixth_raw, cfg) -> dict:
    return W.derive(sixth_raw, gate=float(cfg["workers"]["kv_gate"]["max_kv_usage"]))


def _sixth_bullet() -> str:
    """§10's 2026-10-01 paragraph, whole and collapsed, bounded at the draft-group bullet.

    One bullet per span, for the reason every predecessor said so: a phrase demanded of
    this day must not be answerable by the draft-group prose that follows it.
    """
    raw = DOC.read_text(encoding="utf-8")
    start = raw.index(_SIXTH_HEADING)
    end = raw.index("- **The unannotated draft group.**", start)
    return " ".join(raw[start:end].split()).replace("**", "")


def _qualifying(raw: dict, gate: float) -> list[dict]:
    """The chat miss events meeting the accepted-loss ruling's filing condition: neither
    free-pool churn in the gap nor a gap KV peak at or over the gate.

    Walked by `_gap_events`, whose own assert against `derive`'s aggregate is what keeps
    this a measurement of the same rule the criteria are marked on rather than a second
    opinion with its own arithmetic.
    """
    return [e for e in _chat_events(raw, gate) if not e["churn"] and e["peak"] < gate]


def _gaps_said(gaps) -> str:
    """One day's qualifying gaps in the paragraph's own rendering: `0.1, 0.7, 0.1 and 0.1 s`."""
    said = [f"{g:.1f}" for g in gaps]
    body = f"{', '.join(said[:-1])} and {said[-1]}" if len(said) > 1 else said[0]
    return f"{body} s"


def _gap_shape(days) -> str:
    """Every qualifying event's gap, from the fixtures, in one sentence:
    `2026-09-27's four at 0.1, 0.7, 0.1 and 0.1 s, 2026-09-29's one at 0.2 s and ...`"""
    parts = [f"{day}'s {_word(len(evs))} at {_gaps_said([e['gap_s'] for e in evs])}"
             for day, evs in days]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def test_the_sixth_extract_is_committed_and_moves_no_graded_reading(sixth_raw, sixth):
    """#2248 clause 1: the day is in the tree, tracked, one line, and it is 10-01.

    The hard values sit in THIS node for the reason each predecessor put them in its own
    "is committed" node: they catch a fixture swapped for one from another window, which a
    byte-identity check cannot, since a hand-built extract for some other day is still
    byte-identical to itself. The tracked-file check is the hazard the item exists for —
    the day sat five positions from the edge of a roughly one-file-a-day rotation when it
    was counted, so an uncommitted fixture is that day lost — and the byte-shape check is
    the half that catches a fixture cut down to the fields somebody quoted.
    """
    assert sixth_raw["window"] == ["2026-10-01", "2026-10-02"], (
        "the sixth reading is pinned to a 00:00→00:00 UTC window, like every other reading")
    assert sixth["turns"] == 1189
    assert sixth["misses"] == 154
    assert sixth["reprefill_tokens"] == 19597119
    assert sixth["chat_turns"] == 10
    assert sixth["chat_turns_with_misses"] == 1
    assert sixth["misses_gap_over_gate"] == 24
    assert sixth["two_request_under_stall"] == 16
    assert sixth["two_request_under_stall_not_cold"] == 0
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
         str(SIXTH_EXTRACT.relative_to(ROOT))],
        capture_output=True, text=True)
    assert tracked.returncode == 0, (
        f"{SIXTH_EXTRACT.name} is on disk but not in git: the engine status lines it was "
        "counted from live in a byte rotation, and `tests/fixtures/.gitignore` already "
        "carries `!vllm_prefix_miss_*.json` precisely so that counting the next day is one "
        "`git add` rather than a force-add nobody can see")
    raw = SIXTH_EXTRACT.read_text(encoding="utf-8")
    assert raw == json.dumps(sixth_raw, separators=(",", ":")) + "\n", (
        f"{SIXTH_EXTRACT.name} is not byte-for-byte what `vllm_prefix_miss_window "
        "--write-extract` writes for the rows it holds, so the committed reading has been "
        "edited down by hand and no longer answers a re-derive")
    assert len(raw.splitlines()) == 1, (
        "the extract is one compact line; a multi-line fixture would make every figure in "
        "it diffable by hand instead of by derivation")
    p = _sixth_bullet()
    assert SIXTH_EXTRACT.name in p
    assert "SIXTH extract" in p
    assert "does not displace 2026-09-25 as the counted reading" in p
    assert f"{SIXTH_EXTRACT.name}:1" not in _section(10), (
        "§10 cites a line inside the extract, which is one line long — cite the node that "
        "derives the figure instead")
    assert EXTRACT.name in _counted_reading(_section(10)), (
        "the counted reading no longer names 2026-09-25's extract, so 10-01 displaced it")


def test_the_committed_sixth_extract_yields_its_quoted_counts_from_its_own_bytes(
        sixth_raw):
    """#2248 clause 6 as amended: the witness has history, and its bytes carry the figures.

    The clause the review rung graded `unsatisfiable` ordered a second copy of these bytes
    into the vault and proved the report with `wc -l` over it. Both halves failed: an
    auto-confirmed round cannot write a vault path, and `wc -l` over a canonical
    `--write-extract` file returns 1 — the extract is ONE line by construction, so a line
    count can never be a quoted figure, whatever directory the copy sits in. The amended
    clause keeps what the original was for (the bytes must be committed, and a reader must
    be able to pull the quoted figure off them) and names a command that does answer:
    `jq '.misses|length' tests/fixtures/vllm_prefix_miss_2026-10-01.json` → 154 and
    `jq '.turns|length' …` → 1,189. This node is that command as an assertion, over the
    committed bytes rather than a derivation of them, so a fixture trimmed to the fields
    somebody quoted reddens the length as well as the byte-shape check above.

    The one-line shape is asserted here too, with its own message, because it is the fact
    that made the retired proving command wrong: 1 is the line count of every extract this
    family has ever committed, including 2026-09-25's.
    """
    assert len(sixth_raw["turns"]) == 1189, (
        "the committed extract no longer holds the day's 1,189 turn records, so "
        "`jq '.turns|length'` over it stops answering the figure §10 quotes")
    assert len(sixth_raw["misses"]) == 154, (
        "the committed extract no longer holds the day's 154 miss records, so "
        "`jq '.misses|length'` over it stops answering the figure §10 quotes")
    text = SIXTH_EXTRACT.read_text(encoding="utf-8")
    assert text.count("\n") == 1 and not text.endswith("\n\n"), (
        f"{SIXTH_EXTRACT.name} is not one terminated line, so `wc -l` stops returning 1 — "
        "the retired clause's proving command was wrong for every extract in this family, "
        "and this node's notice with it")


def test_the_six_counted_readings_are_six_distinct_extracts():
    """#2248 clause 3: six constants, six days, and none of the five moved.

    Behavioural rather than five literals: the windows come off the six files, so a
    re-pointed constant fails by naming a day its own nodes do not grade. And the sixth
    paragraph may not cite another day's fixture — the gap shape it claims is derived from
    those files by `test_the_third_counted_day_and_the_gap_shape_of_every_qualifying_event`,
    and prose borrowing another day's file would let it look derived while being quoted.
    """
    six = {EXTRACT, OLD_EXTRACT, REOPEN_EXTRACT, FILING_EXTRACT, REPLICATION_EXTRACT,
           SIXTH_EXTRACT}
    assert len(six) == 6, "a re-pointed constant moves a day this file already grades"
    windows = {c.name: json.loads(c.read_text(encoding="utf-8"))["window"][0] for c in six}
    assert windows == {
        EXTRACT.name: "2026-09-25", OLD_EXTRACT.name: "2026-09-23",
        REOPEN_EXTRACT.name: "2026-09-28", FILING_EXTRACT.name: "2026-09-29",
        REPLICATION_EXTRACT.name: "2026-09-27",
        SIXTH_EXTRACT.name: "2026-10-01"}, windows
    p = _sixth_bullet()
    for c in (EXTRACT, OLD_EXTRACT, REOPEN_EXTRACT, FILING_EXTRACT, REPLICATION_EXTRACT):
        assert c.name not in p, (
            f"the 2026-10-01 paragraph cites {c.name}, which is another day's reading; its "
            "cross-day figures have to come from those files, not from prose that names "
            "them and was copied from a paragraph about something else")


def test_the_sixth_day_round_trips_and_the_fixture_outlives_the_lines(sixth_raw):
    """The fixture is the standing daily job's bytes, the committed turns and misses are
    the database's, and the status lines the logs still hold are a subset of the fixture's.

    The seam the item's step 1 is really about: this fixture was produced by the daily job
    reading `~/lloyd-data`, and the tree's copy has to still be that measurement. Three
    sources say so, in decreasing order of how long they will keep saying it. The daily
    job's own extract is checkable for as long as the job keeps it and is what makes §12's
    "byte-identical" a measurement rather than an intent. `usage.db` is the source of the
    turn and miss rows and outlives the logs. The status lines are the depleting half — the
    engine can report a line the fixture does not hold and thereby prove the fixture wrong,
    but can never invent one to prove it right, so once the rotation passes 10-01 the subset
    relation is all that is left and the fixture is the only copy of those lines. Each half
    is guarded on its own store, and the node skips only when no live store has anything
    left to say about the day, so the byte check never rides along on the rotation's skip.
    """
    from app.paths import PRODUCTION_DATA_ROOT

    # The provenance seam first, because it is the half that is checkable forever: this
    # fixture is the standing daily job's measurement, and §12 says so. The two files are
    # the same bytes by construction and only by construction, so the comparison is what
    # makes the page's claim a measurement rather than an intent. It is guarded by
    # existence and NOT by a skip, because the data-root extract outlives the rotation: if
    # the job's copy is ever pruned the fixture stands alone and this node says nothing
    # about it, which is the truth, rather than failing on a housekeeping change.
    witness = PRODUCTION_DATA_ROOT / "vllm-prefix-miss" / SIXTH_EXTRACT.name
    if witness.exists():
        assert witness.read_bytes() == SIXTH_EXTRACT.read_bytes(), (
            f"{witness} — the standing daily job's own extract for this day — is not the "
            f"bytes committed as {SIXTH_EXTRACT.name}, so §12's claim that the fixture is "
            "the job's measurement is false and one of the two has been edited since")

    db = PRODUCTION_DATA_ROOT / "usage.db"
    # The two live stores are read together and each half is guarded on its own, so the
    # durable byte check above never rides along on the depleting half's skip: #2005 dated
    # the comparable engine lines' expiry at about 2026-10-06, and a node that skipped the
    # whole batch once the rotation passed the day would quietly take the byte check and the
    # database check with it. The node skips only when NO live store has anything to say.
    got = W.extract(sixth_raw["window"][0], sixth_raw["window"][1],
                    data_root=PRODUCTION_DATA_ROOT)
    if not (got["turns"] or got["kv_samples"]):
        pytest.skip(f"neither {db} nor the engine log rotation holds "
                    f"{sixth_raw['window']} any more, so the day rests on the committed "
                    f"fixture and, if the job still keeps it, {witness}")
    if got["turns"]:
        assert got["turns"] == sixth_raw["turns"], (
            "the counted day's turns are not the database's")
        assert got["misses"] == sixth_raw["misses"], (
            "the counted day's misses are not the database's")
    if got["kv_samples"]:
        assert set(map(tuple, got["kv_samples"])) <= set(
            map(tuple, sixth_raw["kv_samples"])), (
            "the engine log reports a status line for this window that the committed extract "
            "does not hold, so the fixture is not what `extract` emitted")


def test_the_sixth_day_figures_are_the_derivation(sixth, sixth_raw, cfg):
    """#2248 clause 2: every figure the 2026-10-01 paragraph prints is `derive`'s at the
    gate `config.yaml` runs, each asserted as a phrase in the doc's own rendering.

    The six figures the item names — 1,189 turns, 154 misses, 19,597,119 re-prefill tokens,
    10 chat turns of which 1 carries a miss, 24 gaps at or over the gate, 16 under-stall
    windows with 0 not-cold — are pinned twice each: once as the derived value, so the data
    cannot move silently, and once as the phrase, so the sentence cannot. The rest of the
    renderings are here because the paragraph prints them, and #1921's finding was that a
    partial pin leaves the remainder of a paragraph free to drift.
    """
    p = _sixth_bullet()
    d = sixth
    gate = float(cfg["workers"]["kv_gate"]["max_kv_usage"])
    assert f"{sixth_raw['window'][0]} 00:00 → {sixth_raw['window'][1]} 00:00 UTC" in p, (
        "the paragraph must name its window in the shape the other readings use")
    assert f"{d['turns']:,} turns, {d['turns_with_misses']} of them carrying" in p
    assert d["turns"] == 1189 and d["turns_with_misses"] == 47
    assert f"{d['chat_turns_with_misses']} of the day's {d['chat_turns']} chat turns" in p
    assert d["chat_turns"] == 10 and d["chat_turns_with_misses"] == 1
    assert f"{d['misses']} misses / {d['reprefill_tokens']:,} tokens" in p
    assert d["misses"] == 154 and d["reprefill_tokens"] == 19597119
    misses_per_day, tokens_per_day = _per_day_budget(_section(6))
    assert f"{misses_per_day} misses / {tokens_per_day}M tokens" in p
    assert d["misses"] > misses_per_day and d["reprefill_tokens"] > tokens_per_day * 1e6, (
        "the paragraph says the day is over §6.1's budget on BOTH sides; the two derived "
        "sides have to agree that it is")
    auto_misses, auto_tokens = d["by_kind"]["autocode"]
    assert (f"{auto_misses} of those {d['misses']} misses and {auto_tokens:,} of the "
            f"{d['reprefill_tokens']:,} tokens are autocode") in p
    assert f"{auto_misses / d['misses']:.0%} of the day is one autonomous kind" in p, (
        "the share's rounding is the extract's, not a number written into the prose")
    m = _BASELINE_RE.search(_section(6))
    assert m and f"2026-09-{m.group(1)}/{m.group(2)}" in p, (
        "the caveat must name the baseline's own date range, derived from §6.1")
    assert f"{_n(d['kv_samples'])} such lines" in p
    assert f"KV p50 {d['kv_p50']:.2f} / p90 {d['kv_p90']:.2f} / max {d['kv_max']:.2f}" in p
    assert f"`Running:` {_n(d['running_p50'])} at the median" in p
    assert f"of {_n(d['two_request_windows'])} lines with two requests resident" in p
    assert f"median {d['two_request_tok_s_p50']} tok/s" in p
    assert f"slowest {d['two_request_tok_s_min']}" in p
    assert f"the {d['two_request_under_stall']} under the bar's {W.STALL_TOK_S:.0f} tok/s" in p
    assert f"{d['two_request_under_stall_cold_in_flight']} of them with a cold re-admission" in p
    assert f"{d['two_request_under_stall_cold_prefill']} with a chunked prefill" in p
    assert d["two_request_under_stall"] == 16 and d["two_request_under_stall_not_cold"] == 0
    assert (d["two_request_under_stall_cold_in_flight"]
            + d["two_request_under_stall_cold_prefill"]) == d["two_request_under_stall"], (
        "the paragraph says every under-bar window is excluded by one of the two evidences, "
        f"and the two counts are {d['two_request_under_stall_cold_in_flight']} and "
        f"{d['two_request_under_stall_cold_prefill']} against "
        f"{d['two_request_under_stall']}")
    assert f"gap p50 {d['gap_s_p50']:.1f} s" in p
    assert f"p50 {d['miss_kv_gap_p50']:.3f} / p90 {d['miss_kv_gap_p90']:.3f}" in p
    assert f"max {d['miss_kv_gap_max']:.3f}" in p
    assert (f"{d['misses_gap_over_gate']} of {d['miss_events']} at or over the "
            f"{gate:.2f} gate") in p
    assert d["misses_gap_over_gate"] == 24
    assert f"{d['misses_gap_over_90']} of them over 0.90" in p
    assert (f"of the day's {d['miss_events']} miss events, "
            f"{d['misses_gap_churned_free_pool']} had free-pool churn in their gap and "
            f"{d['miss_events'] - d['misses_gap_churned_free_pool']} did not") in p
    assert f"{_n(d['pool_tokens'])}-token pool" in p, (
        "the free-pool arithmetic is against this pool, so the paragraph names it")
    # The prose verdicts, as biconditionals against the numbers.
    assert ("criterion (a) does not hold" in p) == (d["chat_turns_with_misses"] > 0), (
        f"(a)'s prose verdict disagrees with chat_turns_with_misses="
        f"{d['chat_turns_with_misses']}")
    assert ("criterion (b) does not hold" in p) == (
        d["misses"] > misses_per_day and d["reprefill_tokens"] > tokens_per_day * 1e6), (
        "(b)'s prose verdict disagrees with the day against §6.1's budget")
    assert ("(c) and (d) both hold" in p) == (
        d["kv_p50"] < gate and d["two_request_under_stall_not_cold"] == 0), (
        "(c)/(d)'s prose verdict disagrees with the KV median and (d)'s counted lines")


def test_the_sixth_chat_miss_is_attributed_with_its_own_gap_line_count(sixth, sixth_raw, cfg):
    """#2248 clause 4: the day's one chat miss, attributed by measurement, and every figure
    of it pinned inside the ONE clause that attributes it.

    The filing node pins its event's six quantities across its paragraph; this one tightens
    that to a single sentence, because the figure this item exists to settle is the gap and a
    gap figure sitting in a different sentence from the iteration it belongs to is exactly how
    an attribution gets quietly re-pointed. The line count is the new quantity: the item
    asserted the earlier qualifying events sat in gaps of minutes and that a 0.1 s gap might
    therefore be a different condition, so `n_lines` — how many status lines the gap spans at
    all — is what a claim about a gap this thin has to be checked against.
    """
    gate = float(cfg["workers"]["kv_gate"]["max_kv_usage"])
    p = _sixth_bullet()
    chat = _chat_events(sixth_raw, gate)
    row = [m for m in sixth_raw["misses"] if m[5] == "chat"]
    assert len(row) == len(chat) == sixth["by_kind"]["chat"][0] \
        == sixth["chat_turns_with_misses"] == 1, (
        f"the paragraph is about ONE event; the extract has rows={row} events={chat} "
        f"by_kind={sixth['by_kind'].get('chat')}")
    r0, = row
    e0, = chat
    assert not e0["churn"] and e0["peak"] < gate, (
        "the condition the ruling named is neither churn nor a gate-level peak, and this "
        f"event has one of them: peak={e0['peak']} churn={e0['churn']}")
    turn = [t for t in sixth_raw["turns"] if t[1] == "chat" and (t[2] or 0) > 0]
    assert len(turn) == 1, f"the extract has {len(turn)} chat turns carrying a miss"
    assert f"`{turn[0][0]}`" in p, "the paragraph must name the one turn it counted"
    attrib = [s for s in p.split(". ") if "iteration" in s]
    assert len(attrib) == 1 and turn[0][0] in attrib[0], (
        "the sentence making the iteration claim does not name the turn the extract counted "
        f"({turn[0][0]}), so the attribution is not pinned to an event")
    for said in (f"iteration {r0[2]}", f"{r0[3]:,} tokens", f"cache_read {r0[4]}",
                 f"a gap of {e0['gap_s']:.1f} s", f"{e0['peak']:.3f}",
                 f"under the {gate:.2f} gate", f"{e0['computed']:,} tokens computed in the gap",
                 f"{e0['free_at_peak']:,} free blocks"):
        assert said in attrib[0], (
            f"{said!r} is not in the clause that attributes the event: {attrib[0]!r}")
    assert e0["n_lines"] == 1, (
        f"the qualifying gap spans {e0['n_lines']} status lines, so 'a single engine status "
        "line' is no longer what the join says")
    assert "a single engine status line" in attrib[0]
    assert "neither free-pool churn nor a gate-level KV explanation" in p
    assert "one event on one day" in p


def test_the_third_counted_day_and_the_gap_shape_of_every_qualifying_event(
        sixth_raw, replication_raw, filing_raw, cfg):
    """#2248 clauses 4 and 5: three counted days now meet the condition, the gap of every
    qualifying event is on the page, and it is separated from the days' own gap medians.

    This is the node the item's own caveat could not have produced. The item asserted that
    09-27's and 09-29's qualifying events "sat in gaps of minutes" and asked whether a 0.1 s
    gap is therefore a different condition. Measured over the committed fixtures with the same
    join, that premise is false — the qualifying gaps are 0.1/0.7/0.1/0.1 s on 09-27 and 0.2 s
    on 09-29 — and the minutes-long figure is a DAY's gap p50. So nothing here is quoted: the
    counts, the gap list, the widest event gap and the three day medians are all derived, and
    the paragraph has to carry each rendering. The positive control is the day-side median
    being minutes long; without it, a node that only asserted every event gap is small would
    pass even on a page that never distinguished an event's gap from a day's.
    """
    gate = float(cfg["workers"]["kv_gate"]["max_kv_usage"])
    raws = [replication_raw, filing_raw, sixth_raw]
    qualifying = [_qualifying(raw, gate) for raw in raws]
    for raw, evs in zip(raws, qualifying):
        assert evs, f"{raw['window'][0]} no longer carries a qualifying chat miss"
    labelled = list(zip([raw["window"][0] for raw in raws], qualifying))
    assert [len(evs) for evs in qualifying] == [4, 1, 1], [len(evs) for evs in qualifying]
    all_evs = [e for evs in qualifying for e in evs]
    p = _sixth_bullet()

    # Every qualifying event's gap, in one sentence, derived from the three fixtures.
    assert (f"The {_word(len(all_evs))} chat miss events now meeting the condition are "
            f"{_gap_shape(labelled)}") in p, _gap_shape(labelled)
    assert (f"none of the {_word(len(all_evs))} exceeds "
            f"{max(e['gap_s'] for e in all_evs):.1f} s") in p
    assert all(e["n_lines"] == 1 for e in all_evs), (
        "a qualifying gap now spans more than one status line, so the paragraph's claim that "
        "each is spanned by one is no longer what the join says")
    assert "each one is spanned by a single engine status line" in p

    # The separation, with the control that makes it a contrast rather than a tautology.
    assert "Those are the qualifying events' gaps, not the days'" in p
    derived = [W.derive(raw, gate=gate) for raw in raws]
    p50s = [d["gap_s_p50"] for d in derived]
    said = ", ".join(f"{v:.1f}" for v in p50s[:-1]) + f" and {p50s[-1]:.1f} s"
    assert f"the same {_word(len(p50s))} windows' overall gap p50s are {said}" in p, said
    widest = max(range(len(p50s)), key=lambda i: p50s[i])
    assert p50s[widest] >= 60.0, (
        f"no day's gap p50 is minutes long (max {p50s}), so the page's contrast between a "
        "day's median and the qualifying events' gaps would be asserted against nothing")
    assert (f"the largest of them, {p50s[widest]:.1f} s, is {labelled[widest][0]}'s, a day "
            f"whose longest qualifying gap is "
            f"{max(e['gap_s'] for e in labelled[widest][1]):.1f} s") in p
    assert max(e["gap_s"] for e in all_evs) < 1.0, (
        "a qualifying gap now runs to a second or more, so the paragraph's shape claim has to "
        "be re-derived rather than left reading as three sub-second days")

    # The third-day claim itself, derived over every fixture this file grades.
    every = [json.loads(c.read_text(encoding="utf-8")) for c in
             (EXTRACT, OLD_EXTRACT, REOPEN_EXTRACT, FILING_EXTRACT, REPLICATION_EXTRACT,
              SIXTH_EXTRACT)]
    qualifying_days = sorted(raw["window"][0] for raw in every if _qualifying(raw, gate))
    assert qualifying_days == ["2026-09-27", "2026-09-29", "2026-10-01"], qualifying_days
    assert _word(len(qualifying_days)) == "three", (
        f"{len(qualifying_days)} counted days now qualify, so §10's 'third counted day' and "
        "the three-window gap-p50 sentence both need re-cutting against the new count")
    assert "third counted day on which the condition is met" in p
    assert "the filing condition is met on a third counted day" in p
    assert "it is not filed" in p, (
        "the paragraph must keep the upstream report unfiled; the ruling's consequence is "
        "unchanged by this day")
    assert not re.search(r"github\.com/vllm-project/vllm/issues/\d+", _section(10)), (
        "§10 names an upstream issue; the 'not filed' sentences must go when it does")


def test_the_sixth_paragraph_breaks_neither_the_one_mark_nor_the_one_query_rule(reading):
    """The two rails every counted-day paragraph is checked against, not merely left
    unbroken: `_verdict` allows exactly one pass/fail mark per criterion inside the counted
    reading's span, and the population rule allows exactly one read-only query in it.

    The positive controls beside the bans are what make this a check rather than a
    description — the same patterns DO find the four marks and the one query the page
    already has, so a paragraph deleted from §10 or a pattern that drifted off the page's
    spelling cannot satisfy the bans by accident.
    """
    p = _sixth_bullet()
    assert p in " ".join(reading.replace("**", "").split()), (
        "the 2026-10-01 paragraph is not inside the counted reading's span, so the two rails "
        "below are being checked against text they do not govern")
    for letter in CRITERIA:
        assert not re.search(rf"\({letter}\) (passes|fails)", p), (
            f"the 10-01 paragraph marks ({letter}) pass/fail inside the counted reading's "
            "span, where _verdict allows exactly one such mark — write prose verdicts, as "
            "the two paragraphs above it do")
    marks = sum(len(re.findall(rf"\({letter}\) (passes|fails)", reading)) for letter in CRITERIA)
    assert marks == 4, (
        f"positive control: the span carries {marks} marks, not one per criterion, so the "
        "ban above was tested against nothing")
    assert not _CHAT_QUERY.search(p), (
        "the 10-01 paragraph cites a second read-only population query; the paragraph's own "
        "rule is to quote the extract, as (a)'s bullet holds the query")
    queries = _CHAT_QUERY.findall(reading)
    assert len(queries) == 1, (
        f"positive control: the counted reading carries {len(queries)} queries, not one")


# ── #2268: the third-day question is recorded as ruled, not as owed-check's ──────────
#
# owed-check's run `20261005_214129_owedcheck_4cc3` (stamped 2026-10-06T04:47:43) settled
# #2248's owed entry 3 — whether a THIRD qualifying day changes the accepted-loss ruling's
# stance, i.e. whether the upstream report should be filed at all — and ruled that it does
# NOT: the condition was already met on the second counted day with its consequence
# published (report owed, unfiled), a third replication of the same shape adds evidence and
# not stance, and only the report's EVIDENCE SET changes — three extracts. Filing remains
# Alan's alone, owned by #2250 Half A. That ruling closed with an order: "The §10 sentence
# publishing this as owed-check's must be rewritten to the ruled answer so no later pass
# re-litigates it." The round that landed the day (`e5c97549`) added the day, its figures
# and its derivation nodes, but never the ruling — so the page kept publishing an answered
# question, in TWO places: §10's 2026-10-01 paragraph and the Review-log bullet for #2248.
#
# `test_the_repoint_question_is_recorded_as_ruled_not_as_owed` (#2008) is the precedent, and
# the reason this is a node rather than a taste: an ownership citation rots the moment its
# question is answered, and "left open-tense, the next round reads a live question and
# re-opens a settled one."
#
# Nothing below types the count the ruled stance rests on. It comes out of `_qualifying`
# over EVERY extract in `tests/fixtures`, so the day a fourth qualifying day is committed
# the evidence set, the ordinal and the ruled wording stop agreeing, and these nodes are
# what says so.

#: The Review-log bullet that published the third day, and so the bullet whose owed-clause
#: the ruling ordered rewritten.
_REVIEW_2026_10_05 = "- 2026-10-05 — **§10 counts 2026-10-01 (#2248).**"


_ORDINALS = "zeroth first second third fourth fifth sixth seventh eighth ninth tenth".split()


def _ordinal(n: int) -> str:
    """`third` for a count of three. §10 counts its qualifying days ORDINALLY ('a third
    counted day', 'the third counted day on which the condition is met') while `_word` gives
    the cardinal a list length needs, so the derived ordinal is its own rendering — and it is
    the one that goes stale first when a fourth qualifying day lands."""
    return _ORDINALS[n] if 0 <= n < len(_ORDINALS) else f"{n}th"


def _said_list(items) -> str:
    """`2026-09-27, 2026-09-29 and 2026-10-01` — the page's own join for an enumerated
    set, so a derived list renders the way §10 writes one rather than the way a test would."""
    items = list(items)
    if len(items) > 1:
        return ", ".join(items[:-1]) + f" and {items[-1]}"
    return items[0]


def _tracked_extracts() -> list[Path]:
    """Every counted-day extract in the tree, enumerated by name family rather than by
    constant, and cross-checked against what git has.

    The pair is the point and not decoration. A counted day enters this family as a FILE —
    that is how #1921, #2005 and #2248 each arrived — so a node naming its fixtures by
    constant would go on passing while leaving the newest day out of the very count the
    ruling says is the only thing that changed. And `tests/fixtures/.gitignore` negates this
    family on purpose, which means an extract dropped in without `git add` is invisible: the
    day is counted, the sentence quotes it, and nothing re-derives it once the rotation
    moves past it.
    """
    on_disk = sorted((ROOT / "tests" / "fixtures").glob("vllm_prefix_miss_*.json"))
    listed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "tests/fixtures"],
        capture_output=True, text=True)
    in_git = {ROOT / line for line in listed.stdout.split()
              if Path(line).name.startswith("vllm_prefix_miss_")}
    assert on_disk and set(on_disk) == in_git, (
        f"the extracts on disk {sorted(p.name for p in on_disk)} are not the ones in git "
        f"{sorted(p.name for p in in_git)}: the ruled stance names an evidence set, and an "
        "untracked extract is a counted day that evidence does not outlive")
    return on_disk


def _qualifying_days(gate: float) -> list[str]:
    """Every counted day the tracked extracts still carry a filing-condition chat miss for.

    One join with the criteria: `_qualifying` is the same walk the (a)/(b)/(c)/(d) marks and
    both re-open triggers are read from, so the evidence set can no more be re-cut by hand
    than a criterion's verdict can.
    """
    return sorted(
        raw["window"][0]
        for raw in (json.loads(p.read_text(encoding="utf-8")) for p in _tracked_extracts())
        if _qualifying(raw, gate))


def _review_bullet_2026_10_05() -> str:
    """The Review-log bullet for #2248, whole and collapsed, bounded at the next bullet or
    heading — and at the end of the page, which is where it currently sits.

    One bullet per span, for the reason every predecessor said so: an owed-clause deleted
    from THIS bullet must not be answerable by the older bullet above it, which carries its
    own owed list about a different day. The end is searched for rather than assumed because
    the Review log grows at the bottom, and a fourth counted-day bullet appended below this
    one would otherwise widen the span silently and let the new entry's prose answer for the
    old one's.
    """
    raw = DOC.read_text(encoding="utf-8")
    start = raw.index(_REVIEW_2026_10_05)
    body = start + len(_REVIEW_2026_10_05)
    nxt = re.search(r"^- |^## ", raw[body:], re.M)
    end = body + nxt.start() if nxt else len(raw)
    return " ".join(raw[start:end].split()).replace("**", "")


def test_the_third_day_question_is_recorded_as_ruled_not_as_owed():
    """#2268 clause 1: §10 states the third-day stance as ruled, and no longer hands the
    question to a party that answered it.

    The ban is over the WHOLE page rather than one span, which is a deliberate difference
    from #2008's node: that one scoped itself because a second, genuinely open instance
    lived further down, and this ruling is that the instance down there is not open. So the
    page-wide ban is the assertion this item exists for, and it is the file's whole copy of
    `architecture/vllm.md`, not a slice, that has to be clean.
    """
    raw = DOC.read_text(encoding="utf-8")
    assert "owed-check's to decide" not in raw, (
        "the page still poses the third-day question as owed-check's to decide, which "
        "owed-check ruled on 2026-10-06 (#2248's owed entry 3) — left open-tense, the next "
        "round editing this page reads a live question and re-opens a settled one")
    p = _sixth_bullet()
    assert "adds evidence, not stance" in p, (
        "the paragraph does not carry the ruling's substance: a third replication of the "
        "same shape is evidence, and evidence is not stance")
    assert re.search(r"owed-check ruled on \d{4}-\d\d-\d\d", p), (
        "the ruled stance is not named as owed-check's and dated, so a later pass cannot "
        "tell a ruling from a preference — the page's other closed rulings are dated too")
    assert "the accepted-loss ruling and its consequence stand as written" in p, (
        "the paragraph does not say what the third day did NOT change: the ruling and its "
        "published consequence")
    assert "the report is owed and it is not filed" in p, (
        "the ruled consequence has to be the published one — report owed, and unfiled")
    assert "#2250 Half A" in p, (
        "filing stays Alan's alone; the paragraph names the item owning the external write "
        "rather than leaving it to whoever reads this next")


def test_the_review_log_bullet_records_the_ruling_instead_of_an_owed_question():
    """#2268 clause 2: the Review-log bullet for #2248 no longer prints the ruled question
    as owed, and is read through a span bounded to that one bullet.

    The bullet is where the day was actually published, and until this node no span read it
    at all — `grep -n 'Review log' tests/test_vllm_doc_claims.py` was empty — so an owed
    clause could be edited into or out of it with nothing in the suite able to notice. The
    two span controls are what make reading it a measurement: the start witness says which
    bullet was opened, and the ban on the older bullet's own heading says the span did not
    run upward into the 2026-10-01 (#2005) entry, whose owed list is about another day and is
    not this ruling's to settle.
    """
    b = _review_bullet_2026_10_05()
    assert b.startswith("- 2026-10-05 — §10 counts 2026-10-01 (#2248)."), (
        f"the span did not open on the #2248 bullet: {b[:80]!r}")
    assert "§10 counts 2026-09-27 (#2005)" not in b, (
        "the span ran upward into the previous Review-log bullet, whose owed list is about "
        "another day and is not this ruling's to settle")
    assert not re.search(r"owed: whether a third qualifying day", b), (
        "the bullet still lists the third-day question as owed")
    assert "(owed-check)" not in b, (
        "the bullet still routes the question to owed-check, which has ruled it")
    assert "adds evidence, not stance" in b, "the bullet does not record the ruling's substance"
    assert "owed-check ruled on 2026-10-06" in b, (
        "the bullet does not say who ruled and when, so the entry reads as the reviewer's "
        "own view rather than a settled ruling")
    assert "the report stays owed and unfiled" in b, (
        "the bullet must carry the consequence unchanged: owed, and not filed")
    assert "#2250 Half A" in b, "the bullet keeps the external write owned by #2250"


def test_the_ruled_stance_names_the_evidence_set_the_extract_set_derives(cfg):
    """#2268 clause 3: the day count behind the ruled sentence is derived from every tracked
    extract through the criteria's own join, never typed.

    Three things make this a check rather than a caption. The extract set is enumerated by
    name family, so a fourth counted day entering the tree as a file enters the count with
    nobody editing a node. The wording is COMPOSED from the derived list, so when that list
    grows the composed phrase (`the four extracts …`, `fourth counted day …`) simply is not
    the sentence on the page and both asserts below fail — which is the failure this item
    wants: a ruled stance about three days has to become false the day the third stops being
    the last. And the control beside the derivation is that not every tracked extract
    qualifies — 2026-09-23, 09-25 and 09-28 are counted readings carrying no filing-condition
    chat miss — so `_qualifying` is discriminating rather than returning the whole directory,
    which is the failure mode a count-on-nothing has.
    """
    gate = float(cfg["workers"]["kv_gate"]["max_kv_usage"])
    days = _qualifying_days(gate)
    assert days == ["2026-09-27", "2026-09-29", "2026-10-01"], days
    assert len(days) < len(_tracked_extracts()), (
        "every tracked extract now qualifies, so the join selected the whole directory and "
        "the count below is a restatement of `ls`")
    evidence = f"the {_word(len(days))} extracts {_said_list(days)}"
    p = _sixth_bullet()
    assert evidence in p, (
        f"§10's 2026-10-01 paragraph does not name its evidence set as the derived one "
        f"({evidence!r}); a count written into prose by hand is a count nobody re-runs, and "
        "this one is the whole content of what the third day changed")
    assert f"{_ordinal(len(days))} counted day on which the condition is met" in p, (
        f"the paragraph's ordinal is not the derived one: {_word(len(days))} counted days "
        f"qualify, so 'the {_ordinal(len(days))} counted day' has to move with the fixtures "
        "rather than with somebody's recollection")
    assert evidence in _review_bullet_2026_10_05(), (
        f"the Review-log entry names a different evidence set than the derived {evidence!r} "
        "— two sentences about one ruling may not rest on two different counts")


def test_the_reword_keeps_every_graded_span_bounded_and_both_rails_intact(reading):
    """#2268 clause 4: re-wording §10's tail broke neither the spans that grade it nor the
    two counts those spans are capped at.

    The anchor check is not decoration: `_SIXTH_HEADING` and `_REPLICATION_HEADING` are the
    END of two other graded spans, so a re-word that shifted a heading left `_filing_bullet`
    and `_replication_bullet` reading a wider slice of the page than their nodes assume —
    #2248's own finding, 'Adding a §10 bullet moves the end-anchor'. The span controls check
    that bound from both sides instead of trusting the anchors to resolve, and the rail
    counts are the same positive controls #1921 and #2248 each re-ran in their own node,
    because this change is the one editing the paragraph they cap.
    """
    raw = DOC.read_text(encoding="utf-8")
    assert _SIXTH_HEADING in raw and _REPLICATION_HEADING in raw, (
        "a counted-day bullet's heading moved, so the graded span that ENDS on it now "
        "silently reads further than its node thinks it does")
    p = _sixth_bullet()
    assert "The unannotated draft group" not in p, (
        "the 2026-10-01 span runs past the next bullet, so a phrase demanded of this day "
        "could be satisfied by prose about the draft group")
    assert "2026-09-27 count" not in p, (
        "the 2026-10-01 span starts before its own heading and would let 09-27's four chat "
        "misses answer for this day's one")
    assert "2026-10-01 count" not in _replication_bullet(), (
        "the 2026-09-27 span now runs into the 10-01 paragraph it is supposed to stop at")
    assert "it is not filed" in p, (
        "the ruled consequence survives the re-wording — the report is owed and it is NOT "
        "filed, and those two sentences are what leave §10 free of an issue URL")
    assert not re.search(r"github\.com/vllm-project/vllm/issues/\d+", _section(10)), (
        "§10 names an upstream issue; the 'not filed' sentences must go when it does")
    assert not re.search(r"\((a|b|c|d)\) (passes|fails)", p), (
        "the re-worded paragraph marks a criterion pass/fail inside the counted reading's "
        "span, where _verdict allows exactly one such mark")
    assert not _CHAT_QUERY.search(p), (
        "the re-worded paragraph cites a second read-only population query; (a)'s bullet "
        "holds the only one the span may carry")
    marks = sum(len(re.findall(rf"\({letter}\) (passes|fails)", reading)) for letter in CRITERIA)
    assert marks == 4, (
        f"positive control: the counted reading carries {marks} marks, not one per "
        "criterion, so the ban above was tested against a span that had already drifted")
    queries = _CHAT_QUERY.findall(reading)
    assert len(queries) == 1, (
        f"positive control: the counted reading carries {len(queries)} queries, not one")
