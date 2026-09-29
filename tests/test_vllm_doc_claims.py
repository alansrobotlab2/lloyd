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
* and since #1810 it holds (a)'s two mints and the population ruling to account: the
  `iv` id prefix may never be bound to the Inner-Voice opt-in inside one clause (the
  mint that appends it reads no body first), owed-check's ruling that a day at the
  machine's own chat volume clears the population bar is on the record as CLOSED with
  its owner and its expiry, and the one read-only `usage.db` query that would re-open
  it is run here and checked against `session_kind` rather than admired.
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
    "one `<ts>_<6 hex>` from `app/routers/messages.py:2113` — three parts each, so neither "
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
    unconditionality, and the fallback's role. `messages.py:2113` stays qualitative —
    the fallback's share of the record is a `usage.db` figure, and this page's rule is
    to cite the query rather than hand-copy the number.
    """
    b = _criterion_bullet(reading, "a")
    assert "`app/routers/sessions.py:724`" in b and "`app/routers/messages.py:2113`" in b, \
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


def test_the_population_ruling_is_closed_and_named_owed_checks(s10, reading, derived):
    """Clause 3: owed-check's ruling replaces the deferral, with the derived count in it.

    The number is `derived['chat_turns']`, so the ruling moves if the counted day is
    ever re-cut — a ruling pinned to a hand-typed 23 would survive a fixture change
    that had already invalidated it.
    """
    assert "not a call this page can make" not in s10, \
        "§10 still defers the population question that owed-check ruled on 2026-09-29"
    b = _criterion_bullet(reading, "a")
    assert _POPULATION_CLOSED.search(b), "(a) does not record the population bar as CLOSED"
    assert f"{derived['chat_turns']} chat turns is the machine's own chat volume and the " \
           f"busiest chat day in the re-derivable record" in b, \
        "the ruling must state its own count, and that count is the derived chat_turns"
    assert re.search(r"owed-check ruled on \d{4}-\d\d-\d\d", b), \
        "the ruling must be named as owed-check's and dated"
    assert "day at that volume clears the bar's population requirement" in b, \
        "the ruling must say the requirement is cleared, not restate it as a question"
    assert "owed-check's and stands until a re-count moves it" in b, \
        "the ruling must carry its own expiry: a re-count moves it, a round does not"


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
    day in the re-derivable record" is decided in SQL over `usage.db`, and the doc claims
    they are the same set. Two languages, one population, no shared code — a grep cannot
    see the disagreement, only a run can. So the query is read OUT of §10 (the test does
    not carry its own copy, which is what would let the doc and the check drift), and the
    per-day table it returns is compared to the per-day table the classifier implies.
    Skipped where there is no database to ask, exactly as the extractor's node does.

    Deliberately NOT pinned here: which day is the busiest. That is owed-check's re-count
    to run as the record widens, and a suite assertion on it would be a test that goes red
    for the one job that cannot fix it from a round.
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
    window on each day for the exclusion to be about at all."""
    for path in (EXTRACT, OLD_EXTRACT):
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
    against the aggregate, so this walk cannot drift from the rule it copies.
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
