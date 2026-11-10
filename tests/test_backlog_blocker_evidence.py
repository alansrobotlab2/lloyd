"""A blocker that quotes a gate line must have copied it out of the file it names.

An implement round may file one item of its own — a *blocker*, tagged
`spawned-by-autocode` + `blocker`, first line "Blocks #N" — and the loop's own
rule is that a **confirmed** blocker skips the depth gate and is taken before
the rest of the implement pool (`workers/sources/autotriage.py` says so to the
triage — the `a BLOCKER of #…` line at `:542-544`, re-pointed at `227858b1` by
`grep -n "skips the depth gate" workers/sources/autotriage.py` — and `:1123-1124`
holds the blocked item's pool while one waits).
So the authority a blocker carries is the authority of the item it blocks.

`#3720` (filed 2026-11-03, `priority: high`, blocker of `#3609`) spent that
authority on an invention. It named its source explicitly —
`/home/alansrobotlab/.local/state/lloyd-automod/rounds/SM_20261103_035337/gate.json`
— and quoted seven strings "out of" it. All seven return 0 hits in that file AND
in its sibling `gate.log` (`grep -Fc` over both, re-run 2026-11-04 at HEAD
`86cb6be7`), while the artifact really says `"base": "e4e8b0a8…"`,
`"passed": 22907` and `base_probe: … 0 already failing`. The phrase it put in
quotation marks — `full-suite failures that pass in isolation` — appears in no
report line `scripts/automod/gate.py` writes (`git grep -c "pass in isolation" --
scripts/automod/gate.py` → no match at `227858b1`), so no gate report could have
been its source. Only one triage's refutation stood between that item and the front
of the pool.

The prose forbidding it exists and is intact (`workers/sources/autocode.py:175-181`:
"copy any gate or ledger line you quote out of that artifact **re-read from
disk** … and name the path you copied from"), but prose is the whole guard.
`tests/test_backlog_blockers.py` pins a blocker's *shape* — the tag pair, the
"Blocks #N" first line, the spawn cap, liveness, expiry — and nothing read a
quoted line back against the artifact it was said to come from. This file is
that check on the write path: `backlog_write_task`'s create branch, where a
fabricated quote used to reach `draft` unannotated.

**Two rulings the check had to make, because the cheap version eats honest
blockers.** (1) A truthful quote of a gate report is *not* a byte substring of
`gate.json`: the JSON stores `"passed": 22907` and the line a human would quote,
`[FAIL] tests (1220.1s) all 5 failure(s) are new in this round (…)`, exists only
in the SIBLING `gate.log`. So a miss is judged against every readable file in
the named round directory, not against the one path the body happened to name.
(2) A quote is only evidence when it could have been a line of the file, so a
span with an ellipsis in it (an elision, a paraphrase), a single bare token (a
path, a sha, a test id), a shell command the author wrote themselves, and a
round directory that has been swept are all read as **unverifiable, not absent**
— `#3720` is unambiguous precisely because every one of its multi-word quotes
misses both files. The unit nodes at the bottom of this file pin each of those
four against a quote that would otherwise be refused.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent_mcp.backlog as MCP  # noqa: E402
from app import backlog_evidence as EV  # noqa: E402
from scripts.automod import state as S  # noqa: E402

#: The round `#3720` cited. Its real report is reproduced below at fixture scale.
ROUND_ID = "SM_20261103_035337"

GATE_JSON = json.dumps({
    "round_id": ROUND_ID,
    "base": "e4e8b0a81bd8af5abe08b25e5b22158da1f61f8e",
    "head": "0ec76a74d3bd349f175572def818e9cd9b572e37",
    "ok": False,
    "rungs": [
        {"name": "tests", "ok": False, "seconds": 1220.14,
         "data": {"passed": 22907, "failed": 5, "collected": 22953}},
    ],
    "base_probe": "probed 1 file(s) at base e4e8b0a8 in a fresh data root: 0 already failing",
}, indent=2)

GATE_LOG = (
    "[PASS] preflight (1.4s) 2 file(s) in scope\n"
    "[FAIL] tests (1220.1s) all 5 failure(s) are new in this round "
    "(tests/test_architecture_coverage_doc_claims.py::"
    "test_the_corpus_fallback_decided_anchors_are_at_or_under_their_ceiling)\n"
)

#: Verbatim off the real `#3720`. Absent from both files, which is the fact the
#: guard has to act on.
FABRICATED = ("FAIL tests (1382.79s): 3 failed, 22886 passed, 1 xfailed, "
              "40 skipped in 1240.63s (0:20:40)")

#: A line that IS in the sibling `gate.log` while the body names `gate.json`:
#: the honest case the single-file version of this check would false-refuse.
HONEST_LOG_LINE = "all 5 failure(s) are new in this round"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A scratch board and a scratch rounds root holding one round's report.

    Same isolation `tests/test_backlog_blockers.py:56-60` uses for the board (the
    `isolated` fixture): the
    rounds root moves the same way the gate's rungs move it — by rebinding the
    `state` module attribute, which is read at call time.
    """
    board = tmp_path / "backlog"
    board.mkdir()
    monkeypatch.setattr(MCP, "BACKLOG_DIR", board)
    rounds = tmp_path / "rounds"
    one = rounds / ROUND_ID
    one.mkdir(parents=True)
    (one / "gate.json").write_text(GATE_JSON, encoding="utf-8")
    (one / "gate.log").write_text(GATE_LOG, encoding="utf-8")
    monkeypatch.setattr(S, "ROUNDS_DIR", rounds)
    store = {"board": board, "rounds": rounds, "round_dir": one,
             "gate_json": one / "gate.json"}
    return store


def _write(store, *, description, name="Blocks #3609: the tests rung cannot go green",
           tags=("spawned-by-autocode", "blocker"), **kw):
    """One `backlog_write_task` create, answered as the tool answers it."""
    out = MCP._handle_write({"name": name, "description": description,
                             "board": "lloyd", "tags": list(tags), **kw})
    return json.loads(out)


def _body(source: str, quote: str) -> str:
    """A blocker body in `#3720`'s shape: names the artifact, then quotes it.

    The round it cites is the one in the named path, so a body pointing at a swept
    round cites only that round — the case clause 3 says must stay unverifiable.
    """
    cited = re.search(r"SM_\d{8}_\d{6}", source)
    where = f" for round {cited.group(0)}" if cited else ""
    return (f"Blocks #3609\n\n**Evidence, re-read from disk{where}.** "
            f"From that round's gate report (`{source}`):\n`{quote}`\n\n"
            "**What to fix.** Decide which is wrong and fix that one thing.")


def _files(board: Path) -> list[str]:
    return sorted(p.name for p in board.iterdir())


# ===========================================================================
# Clause 1 — an absent quote is refused, and nothing is written
# ===========================================================================

def test_a_blocker_whose_quoted_gate_line_is_absent_from_the_named_round_is_refused(store):
    """The `#3720` instance: the quote is in no part of the round's report."""
    out = _write(store, description=_body(str(store["gate_json"]), FABRICATED),
                 priority="high")
    assert out.get("success") is False, out
    # The refusal has to be actionable by a reader holding only the round's own
    # prose: name the string, name the file it was said to come from.
    assert FABRICATED in out["error"]
    assert str(store["gate_json"]) in out["error"]
    assert str(store["round_dir"] / "gate.log") in out["error"]
    assert "gate.json" in out["error"] and "re-read" in out["error"].lower()


def test_a_refused_blocker_writes_no_board_file(store):
    """Refused means it never reached `draft`, so the picker cannot take it."""
    _write(store, description=_body(str(store["gate_json"]), FABRICATED),
           priority="high")
    assert _files(store["board"]) == []


def test_the_refusal_reaches_the_round_over_the_mcp_tool_seam(store):
    """A round calls the tool, not the handler: the refusal has to survive `call_tool`.

    `backlog_write_task` is dispatched by `agent_mcp/backlog.py::call_tool` in the
    MCP server process, and the round reads one text block back. A guard that
    refused inside `_handle_write` but whose answer was swallowed at that seam
    would look enforced to every unit node above and still let `#3720` through.
    """
    args = {"name": "Blocks #3609: the tests rung cannot go green",
            "description": _body(str(store["gate_json"]), FABRICATED),
            "board": "lloyd", "tags": ["spawned-by-autocode", "blocker"],
            "priority": "high"}
    result = asyncio.run(MCP.call_tool("backlog_write_task", args))
    text = "".join(getattr(block, "text", "") for block in result.content)
    assert json.loads(text)["success"] is False
    assert FABRICATED in text and str(store["gate_json"]) in text
    assert _files(store["board"]) == []


def test_a_quote_found_only_in_the_sibling_log_still_proves_the_quote(store):
    """Naming `gate.json` and quoting the `gate.log` rendering is honest, not absent.

    `gate.json` holds `"passed": 22907`; the sentence a round would copy,
    `[FAIL] tests (1220.1s) all 5 failure(s) are new in this round (…)`, is only
    in `gate.log:8`. A check that grepped the single named path would refuse this
    item — which is the failure mode `#3721`'s triage named as the design hazard.
    """
    out = _write(store, description=_body(str(store["gate_json"]), HONEST_LOG_LINE))
    assert out.get("success") is True, out
    assert len(_files(store["board"])) == 1


# ===========================================================================
# Clause 2 — a truthful blocker is not delayed, and not stamped
# ===========================================================================

def test_a_blocker_whose_quote_reproduces_files_exactly_as_untouched(store):
    out = _write(store, description=_body(str(store["gate_json"]), HONEST_LOG_LINE))
    assert out.get("success") is True and out.get("created") is True, out
    (item,) = list(store["board"].iterdir())
    fm = MCP.parse_frontmatter(item.read_text(encoding="utf-8"))[0]
    assert fm["status"] == "draft"
    assert "blocker" in fm["tags"] and "spawned-by-autocode" in fm["tags"]
    # A quote that reproduces is not a finding about the item: nothing is added
    # to its front matter, so its keys are the keys any other create writes.
    plain = _write(store, name="An ordinary item",
                   description="Do the thing.", tags=("backlog",),
                   )
    assert plain.get("success") is True
    other = [p for p in store["board"].iterdir() if p != item][0]
    other_fm = MCP.parse_frontmatter(other.read_text(encoding="utf-8"))[0]
    assert set(fm) == set(other_fm), (set(fm) ^ set(other_fm))
    assert HONEST_LOG_LINE in item.read_text(encoding="utf-8")


def test_a_quote_taken_out_of_the_json_itself_reproduces_too(store):
    """`"passed": 22907` is a byte-for-byte line of the named `gate.json`."""
    out = _write(store, description=_body(str(store["gate_json"]),
                                          '"passed": 22907'))
    assert out.get("success") is True, out


# ===========================================================================
# Clause 3 — the three cases where a miss is not evidence
# ===========================================================================

def test_a_round_directory_that_has_been_swept_files_the_item_anyway(store):
    """The retention sweep reclaims round directories; an old round id is not a lie."""
    swept = store["rounds"] / "SM_20261001_010101" / "gate.json"
    out = _write(store, description=_body(str(swept), FABRICATED))
    assert out.get("success") is True, out
    assert len(_files(store["board"])) == 1


def test_an_elided_or_paraphrased_quote_is_unverifiable_not_absent(store):
    """A quote with an ellipsis in it claims "this text is cut", so it cannot be one line."""
    for elided in ("all 5 failure(s) are new … (tests/test_architecture_coverage_doc_claims.py)",
                   "tests (1382.79s): 3 failed, 22886 passed, ... in 1240.63s",
                   "the gate says the failures are new in this round…"):
        out = _write(store, description=_body(str(store["gate_json"]), elided))
        assert out.get("success") is True, (elided, out)
    assert len(_files(store["board"])) == 3


def test_an_item_without_the_blocker_tag_is_never_checked(store):
    """The guard is on the one tag pair that buys the item its early take."""
    out = _write(store, name="An ordinary finding", description=_body(
        str(store["gate_json"]), FABRICATED), tags=("spawned-by-autocode",))
    assert out.get("success") is True, out
    assert len(_files(store["board"])) == 1


# ===========================================================================
# The check itself: what it reads as a quote, and what it refuses to read as one
# ===========================================================================

def test_the_command_a_blocker_asks_a_reader_to_run_is_not_treated_as_a_quote(store):
    """Every honest blocker body carries a `Check:` command; it is never in the artifact.

    The house shape is "Check: `cd ~/lloyd && git grep -n …`" written in the same
    paragraph as the round it cites. Judging that span against `gate.json` would
    refuse an honest item on the spelling of a command.
    """
    checked = EV.absent_quotes(
        f"Blocks #3609\n\nFrom `{store['gate_json']}`:\n"
        "`cd ~/lloyd && git grep -n passed -- scripts/automod/gate.py`\n"
        f"`{HONEST_LOG_LINE}`\n", rounds_dir=store["rounds"])
    assert checked == []
    # …and the span rule is shown able to bite, on the fabricated line, not only
    # on the command it let through.
    missing = EV.absent_quotes(
        f"Blocks #3609\n\nFrom `{store['gate_json']}`:\n`{FABRICATED}`\n",
        rounds_dir=store["rounds"])
    assert [m["quote"] for m in missing] == [FABRICATED]


def test_a_quote_in_a_paragraph_that_names_no_round_is_not_checked(store):
    """Attribution is per paragraph: a repro result quoted beside its own command
    is not being offered as the gate's output."""
    assert EV.absent_quotes(
        f"Blocks #3609\n\nReproduce it yourself:\n"
        f"`pytest tests/test_skill_verdicts.py -q` → `1 failed, 187 passed in 39.38s`\n"
        f"\nFrom `{store['gate_json']}`: `{HONEST_LOG_LINE}`\n",
        rounds_dir=store["rounds"]) == []


def test_a_single_token_span_a_path_or_a_sha_is_out_of_the_net(store):
    """Node ids, paths and shas carry no phrasing to compare, so a miss proves nothing."""
    assert EV.absent_quotes(
        f"Blocks #3609\n\nFrom `{store['gate_json']}`: the node is "
        f"`tests/test_skill_verdicts.py::test_a_run_row_whose_falsifier_cannot_see_its_input` "
        f"on base `a4473fda9d6c00d0e0c0deadbeef000011112222`\n",
        rounds_dir=store["rounds"]) == []


def test_an_unreadable_round_directory_yields_no_verdict(store):
    """Swept, or never existed, or unreadable: no absence is reported either way."""
    gone = store["rounds"] / "SM_20260101_000000"
    assert EV.absent_quotes(f"From `{gone}/gate.json`:\n`{FABRICATED}`\n",
                            rounds_dir=store["rounds"]) == []


def test_the_absence_is_judged_against_every_file_in_the_round_directory(store):
    """A quote carried by any file the round wrote counts; only an all-file miss is absent."""
    (store["round_dir"] / "live_baseline.json").write_text(
        '{"note": "the string that only lives here"}\n', encoding="utf-8")
    assert EV.absent_quotes(
        f"From `{store['gate_json']}`: `the string that only lives here`\n",
        rounds_dir=store["rounds"]) == []
    assert len(EV.absent_quotes(
        f"From `{store['gate_json']}`: `the string that lives in no file at all`\n",
        rounds_dir=store["rounds"])) == 1


def test_whitespace_is_collapsed_before_a_line_is_called_absent(store):
    """A log line wrapped or padded is still the line the round copied."""
    assert EV.absent_quotes(
        f"From `{store['gate_json']}`: `all   5 failure(s)    are new in  this round`\n",
        rounds_dir=store["rounds"]) == []


# ===========================================================================
# #3960 — a `key: value` rendering of a field the artifact holds is the quote
#
# The fixture's `gate.json` writes `"head":
# "0ec76a74d3bd349f175572def818e9cd9b572e37"`; prose that copies that field writes
# `head: 0ec76a74d3bd349f175572def818e9cd9b572e37`, and an abbreviating round
# writes `head: 0ec76a74`. Neither is a substring of the JSON, which is the whole
# reason the guard's one real firing (2026-11-07, `mcp.err:13135`, round
# SM_20261107_150028, `head: a8773819`) refused an honest blocker.
# ===========================================================================

HEAD_SHA = "0ec76a74d3bd349f175572def818e9cd9b572e37"


def _plain_keys(store, blocker_file):
    """The front matter keys an ordinary create writes, for a no-new-key compare."""
    plain = _write(store, name="An ordinary item", description="Do the thing.",
                   tags=("backlog",))
    assert plain.get("success") is True, plain
    other = [p for p in store["board"].iterdir() if p != blocker_file][0]
    return set(MCP.parse_frontmatter(other.read_text(encoding="utf-8"))[0])


# --- Clause 1: the honest rendering files, and files exactly as untouched ----

def test_an_abbreviated_head_quote_of_the_field_the_json_holds_is_accepted(store):
    """`head: 0ec76a74` against `"head": "0ec76a74d3bd349f…"` is a copy, not a lie."""
    out = _write(store, description=_body(str(store["gate_json"]), "head: 0ec76a74"))
    assert out.get("success") is True, out
    assert out.get("created") is True, out
    assert len(_files(store["board"])) == 1
    (item,) = list(store["board"].iterdir())
    fm = MCP.parse_frontmatter(item.read_text(encoding="utf-8"))[0]
    assert fm["status"] == "draft"
    # Clause 1's other half: a quote that reproduces is not a finding about the
    # item, so the filed blocker carries no front matter key an ordinary create
    # does not also write.
    assert set(fm) == _plain_keys(store, item), (set(fm) ^ _plain_keys(store, item))


def test_a_byte_exact_head_quote_of_the_field_the_json_holds_is_accepted(store):
    """Even the whole sha refused: prose writes `head: X`, the JSON `"head": "X"`."""
    out = _write(store, description=_body(str(store["gate_json"]), f"head: {HEAD_SHA}"))
    assert out.get("success") is True, out
    assert len(_files(store["board"])) == 1


def test_the_accepted_abbreviation_reaches_the_board_over_the_mcp_tool_seam(store):
    """The round calls the tool, so acceptance has to survive `call_tool` too.

    The refusal seam is pinned above; the false refusal this item is about was a
    refusal *answer*, so the same seam is pinned going the other way: an honest
    abbreviation must come back `success: true` with a board file behind it.
    """
    args = {"name": "Blocks #3609: the gate report cannot be read for a head",
            "description": _body(str(store["gate_json"]), "head: 0ec76a74"),
            "board": "lloyd", "tags": ["spawned-by-autocode", "blocker"]}
    result = asyncio.run(MCP.call_tool("backlog_write_task", args))
    text = "".join(getattr(block, "text", "") for block in result.content)
    assert json.loads(text)["success"] is True, text
    assert len(_files(store["board"])) == 1


# --- Clause 2: the rule itself, on the unit surface -------------------------

def test_a_key_value_span_reproduces_when_a_line_holds_the_key_and_the_value(store):
    """Equality, a prefix at the 6-character floor, and a key spelled bare or quoted."""
    json_key = str(store["gate_json"])  # artifact line: "head": "0ec76a74d3bd…"
    for quoted in (f"head: {HEAD_SHA}",          # equal to the value token
                   "head: 0ec76a74",              # an 8-char prefix of it
                   "head: 0ec76a",                # exactly MIN_VALUE_PREFIX_CHARS
                   '"head": 0ec76a74'):           # the span itself quoted: key stripped
        assert EV.absent_quotes(f"From `{json_key}`: `{quoted}`\n",
                                rounds_dir=store["rounds"]) == [], quoted
    # …and the same rule off a bare (YAML-shaped) key in another file the round wrote.
    (store["round_dir"] / "run_spec.yaml").write_text(f"head: {HEAD_SHA}\n",
                                                      encoding="utf-8")
    assert EV.absent_quotes(f"From `{json_key}`: `head: 0ec76a74`\n",
                            rounds_dir=store["rounds"]) == []


# --- Clause 3: the fabricated quote is still caught --------------------------

def test_a_key_value_quote_of_a_value_the_round_never_held_is_still_refused(store):
    """`head: 9999dead`: the key is real, the value is in no file, so the catch bites.

    This is also the clause that fails any fix that took the easy way out and made
    `key: value` spans *unverifiable* in `_is_verifiable` — exempting the shape
    there would file this item. The span stays verifiable, and the refusal is the
    answer, which is what keeps the fix inside the match step.
    """
    assert EV._is_verifiable("head: 9999dead") is True
    out = _write(store, description=_body(str(store["gate_json"]), "head: 9999dead"))
    assert out.get("success") is False, out
    assert "head: 9999dead" in out["error"]
    assert str(store["gate_json"]) in out["error"]
    assert _files(store["board"]) == []


# --- Clause 4: the prefix floor bites ---------------------------------------

def test_a_short_prefix_of_a_value_is_refused_though_the_key_is_real(store):
    """`passed: 2290` against `"passed": 22907` is a guess about a count, not a copy."""
    assert EV._is_verifiable("passed: 2290") is True  # 12 chars: MIN_QUOTE_CHARS clears
    out = _write(store, description=_body(str(store["gate_json"]), "passed: 2290"))
    assert out.get("success") is False, out
    assert "passed: 2290" in out["error"]
    assert _files(store["board"]) == []
    # One character longer on the value side is not a prefix, it is the value:
    # equality needs no floor, so this renders and files.
    files = _write(store, description=_body(str(store["gate_json"]), "passed: 22907"))
    assert files.get("success") is True, files


# --- Clause 5: nothing else about what is checked moved ---------------------

def test_the_field_arm_exempts_the_spreads_it_always_exempted_and_no_more(store):
    """#3720's prose quote, an elision, a bare token, a command, an unattributed span.

    Every one of those paragraphs names the round, so each span is offered as the
    gate's output; only the multi-word fabrication is absent, exactly as before the
    field arm existed.
    """
    cited = f"From `{store['gate_json']}`"
    missing = EV.absent_quotes(
        f"Blocks #3609\n\n{cited}: `{FABRICATED}`\n\n"
        f"{cited}, elided: `all 5 failure(s) are new … in this round`\n\n"
        f"{cited}, the node is "
        "`tests/test_skill_verdicts.py::test_a_run_row_whose_falsifier_cannot_see_its_input`\n\n"
        f"{cited}, Check: `cd ~/lloyd && git grep -n passed -- scripts/automod/gate.py`\n\n"
        "No round is named in this paragraph, so `head: 9999dead` is not a claim "
        "about its report.\n", rounds_dir=store["rounds"])
    assert [m["quote"] for m in missing] == [FABRICATED]


def test_the_normaliser_names_the_false_refusal_it_exists_for():
    """A later reader can re-run the incident from the comment beside the code.

    The comment has to be readable where the normaliser is, not merely somewhere in
    the file: the slice under test runs from the field-arm patterns to `absent_quotes`.
    """
    src = (ROOT / "app" / "backlog_evidence.py").read_text(encoding="utf-8")
    start = src.index("_KEY_VALUE = re.compile")
    block = src[start:src.index("def absent_quotes", start)]
    assert "mcp.err:13135" in block, block
    assert "head: a8773819" in block
    assert "SM_20261107_150028" in block


def test_the_guard_is_still_reached_from_only_the_two_files_it_was():
    """The fix is inside `absent_quotes`, so no new caller may appear (item clause 4).

    The symbol is spelled in two fragments on purpose: a file whose own text
    contains the string being grepped shows up as a hit, and this file must not be
    one — the grep's result set is the thing under test.
    """
    proc = subprocess.run(["git", "grep", "-n", "blocker_quote_" + "refusal",
                           "--", "*.py"], cwd=ROOT, capture_output=True, text=True)
    files = sorted({line.split(":", 1)[0] for line in proc.stdout.splitlines()})
    assert files == ["agent_mcp/backlog.py", "app/backlog_evidence.py"], proc.stdout
