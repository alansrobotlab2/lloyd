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

The prose forbidding it exists and is intact (`workers/sources/autocode.py:174-181`:
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
