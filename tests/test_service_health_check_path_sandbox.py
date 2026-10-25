"""#2320: the health check grades the protected-path substrate, red when it is not enforcing.

`agent_mcp/_path_sandbox.py` puts its verdict on the aggregator's `/state` under
`protected_path_sandbox` — `enforcing`, `fallback`, `bwrap`, `entries`, `error` — and
`agent_mcp/main.py:914` serves it. Before this item nothing read it: `git grep -n
"protected_path_sandbox" -- scripts/` returned only a fixture, `scripts/service_health_check.py`
had no HTTP client at all (its one network primitive, `http_check`, is a TCP connect that never
reads a body), and a live run printed `Overall: 23/23 services healthy` with no sandbox row. So a
box whose `bwrap` cannot build a namespace — where every Bash call runs UNSANDBOXED on the
fail-open branch at `_path_sandbox.py:164`, whose only signal is a `logger.warning` nobody greps
— looked exactly like a healthy machine. The absence of a guard was indistinguishable from a
guard that worked.

These tests drive the row from stubbed payloads, with no live aggregator anywhere in the file:
the thing under test is what the check does with an answer, not whether this box answers.

Two properties matter more than the happy path, and both are the item's clauses:

  * a fault that is not counted is not a fault. `_scored_rows` drops any row carrying an
    `advisory` key, and `format_text`'s `Overall:` is arithmetic over `_scored_rows` alone, so an
    `advisory` fallback row would print under a separate `Advisory:` heading and leave
    `Overall: 23/23` green — the exact failure this item exists to close. So the red row is
    asserted to have NO `advisory` key, and `Overall:` is asserted through the module's own
    `main()`, never over a hand-typed summary.
  * an unreadable `/state` is not a pass and not a skip. Connection refused, the 401 an
    uncredentialed request gets, and a body with no such key each have to produce a graded FAIL
    naming its reason. A row that silently went missing would put the box back exactly where this
    item found it.

The entry count in the PASS row comes from the payload's own `entries` list, which is
`protected_shell_ro_paths()` as the serving process holds it — so the check asserts a number the
kernel is honouring rather than a number this repo also keeps in a constant. The second enforcing
test below feeds a DIFFERENT length to prove it: a row that read a local constant would print the
same count for both payloads and go red.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "service_health_check.py"


def _load():
    spec = importlib.util.spec_from_file_location("shc_path_sandbox", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shc = _load()

STATE_URL = "http://127.0.0.1:8500/state"
TOKEN = "stub-aggregator-token"
HEADER = "X-Lloyd-Aggregator-Token"

# The two answers `/state` can give about the substrate, verbatim in shape to
# `_path_sandbox.status()` (`agent_mcp/_path_sandbox.py:230-241`): `enforcing` and `fallback` are
# the #2109 pair, `error` is the probe's own reason, `entries` is the list the process would bind.
FALLBACK = {
    "protected_path_sandbox": {
        "enforcing": False,
        "fallback": True,
        "bwrap": True,
        "entries": [],
        "error": "bwrap: can't unshare namespace",
    }
}
ENFORCING_SIX = {
    "protected_path_sandbox": {
        "enforcing": True,
        "fallback": False,
        "bwrap": True,
        "entries": ["/home/alansrobotlab/obsidian", "/home/alansrobotlab/.openclaw",
                    "/home/alansrobotlab/lloyd/agent-services", "/etc/passwd",
                    "/etc/shadow", "/home/alansrobotlab/.ssh"],
        "error": None,
    }
}
# The same substrate with three paths bound instead of six. Its only job is to be a different
# LENGTH: `entries` is the payload's own list, so the row's count must move with it.
ENFORCING_THREE = {
    "protected_path_sandbox": dict(ENFORCING_SIX["protected_path_sandbox"],
                                   entries=["/a", "/b", "/c"])
}

ROW_NAME = "path-sandbox"


def _rows(payload=None, exc=None):
    """The row set for one stubbed `/state` answer — payload dict, or the exception it raised."""
    def fetcher():
        if exc is not None:
            raise exc
        return payload

    return shc.check_path_sandbox(fetcher=fetcher)


def _only(rows):
    """Exactly one row, which is the promise: the check never goes quiet on this subject."""
    assert len(rows) == 1, f"expected exactly one row, got {[r['name'] for r in rows]}"
    row = rows[0]
    assert row["name"] == ROW_NAME, row
    assert row["category"] == shc.PATH_SANDBOX_CATEGORY, row
    return row


def test_a_fallback_payload_is_a_graded_fail_that_names_the_error():
    """#2320 clause 1: `enforcing: false` is one row, unhealthy, NOT advisory, naming `.error`.

    The `advisory` key is the trap: `_scored_rows` drops every row that carries one, so an
    advisory fallback row still lets the run print `Overall: N/N`. Its absence is asserted, not
    assumed.
    """
    row = _only(_rows(payload=FALLBACK))
    assert row["healthy"] is False, row
    assert "advisory" not in row, (
        "an advisory fallback row is dropped by _scored_rows and leaves Overall: green, "
        "which is the failure this item exists to close")
    assert "bwrap: can't unshare namespace" in row["status"], row["status"]
    assert row["enforcing"] is False and row["fallback"] is True, row


def test_overall_counts_the_fallback_row_so_the_run_cannot_read_all_healthy(monkeypatch,
                                                                            capsys):
    """#2320 clause 2: `Overall:` over the fallback answer counts the new row.

    Driven through `main()` rather than `format_text(results, summary)` because the summary is
    `main`'s arithmetic — a hand-typed one would prove nothing about a real run. Every other row
    producer is stubbed to nothing, so the two numbers in the line are exactly this row's: a
    substrate that is not enforcing prints `Overall: 0/1`, and the same drive with an enforcing
    payload prints `Overall: 1/1`. Neither number existed before this item.
    """
    def overall(payload=FALLBACK, argv_extra=()):
        text = _drive(shc, monkeypatch, capsys, payload,
                      fmt="text", argv_extra=argv_extra)
        found = re.search(r"Overall: (\d+)/(\d+)", text)
        assert found, f"no Overall: line in:\n{text}"
        assert "Advisory:" not in text, (
            "the fallback row was reported as advisory, so `Overall:` never saw it")
        return int(found.group(1)), int(found.group(2))

    healthy, total = overall()
    assert (healthy, total) == (0, 1), f"Overall: {healthy}/{total}"

    monkeypatch.undo()
    healthy, total = overall(ENFORCING_SIX)  # nothing to count as a fault this time
    assert (healthy, total) == (1, 1), f"Overall: {healthy}/{total}"


def test_an_enforcing_payload_passes_and_names_the_count_from_its_own_entries():
    """#2320 clause 3: `enforcing: true` is a healthy row whose count is the payload's own.

    Six entries in, six in the status. The second half is the half that matters: a payload with
    THREE entries must report three, which a row reading a constant kept in this script could
    never do.
    """
    six = _only(_rows(payload=ENFORCING_SIX))
    assert six["healthy"] is True, six
    assert "advisory" not in six, six
    assert re.search(r"\b6\b", six["status"]), six["status"]
    assert six["entry_count"] == 6, six

    three = _only(_rows(payload=ENFORCING_THREE))
    assert three["healthy"] is True, three
    assert re.search(r"\b3\b", three["status"]), three["status"]
    assert not re.search(r"\b6\b", three["status"]), (
        "the count did not move with the payload's entries — it is being read from a "
        f"constant in the script: {three['status']}")
    assert three["entry_count"] == 3, three


# Every way `/state` can fail to answer, with the reason each must name. The 401 row is the
# measured uncredentialed reply (#1053): `/state` is credential-bearing, so a check that forgot
# the header would read as "no substrate information" and go quiet.
UNREADABLE = [
    ("connection refused",
     urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")),
     "Connection refused"),
    ("the 401 an uncredentialed request gets",
     urllib.error.HTTPError(STATE_URL, 401, "Unauthorized", None, None),
     "401"),
    ("a body that is not JSON",
     json.JSONDecodeError("Expecting value", "<html>", 0),
     "Expecting value"),
    # The third way clause 4 names — a 200 body that lacks the key — is not a transport
    # exception at all, so it is pinned as a payload below, where the aggregator is answering.
]


@pytest.mark.parametrize("label,exc,expected",
                         [(u[0], u[1], u[2]) for u in UNREADABLE],
                         ids=[u[0] for u in UNREADABLE])
def test_an_unreadable_state_is_a_graded_fail_naming_its_reason(label, exc, expected):
    """#2320 clause 4: an unreadable `/state` is a FAIL naming the reason, never a silent skip.

    Each of these is also asserted to be non-advisory for the clause-1 reason: dropped from the
    arithmetic, "I could not read it" would print under `Advisory:` and leave `Overall:` green.
    """
    row = _only(_rows(exc=exc))
    assert row["healthy"] is False, f"{label}: {row}"
    assert "advisory" not in row, f"{label}: a dropped row cannot fault the verdict: {row}"
    assert expected in row["status"], f"{label}: {row['status']!r} does not name the reason"


@pytest.mark.parametrize("payload,expected", [
    ({}, "no protected_path_sandbox"),
    ({"protected_path_sandbox": None}, "no protected_path_sandbox"),
    ({"protected_path_sandbox": {"enforcing": True, "fallback": False}}, "no entries"),
    ({"protected_path_sandbox": {"enforcing": True, "entries": "not-a-list"}}, "no entries"),
])
def test_a_payload_that_does_not_answer_the_question_is_a_fail_not_a_guess(payload, expected):
    """A body that answers 200 and says nothing usable is graded, not smoothed over.

    The first two rows are clause 4's "JSON lacking the key": the aggregator answered, and the
    answer says nothing about the substrate. The second pair is the same discipline applied to the
    denominator — `entries` is what the PASS row counts, so an enforcing answer with no entry list
    cannot print a number and reports the missing denominator instead of a bare green line
    (#1726's rule that a check whose denominator is unknown is not a check).
    """
    row = _only(shc.check_path_sandbox(fetcher=lambda: payload))
    assert row["healthy"] is False, row
    assert "advisory" not in row, row
    assert expected in row["status"], row["status"]


def test_the_fetcher_sends_the_state_route_with_the_aggregator_credential(monkeypatch):
    """The process boundary: one authenticated GET of `aggregator_config.route("state")`.

    `/state` is credential-bearing (#1053) and no httpx exists in the interpreter this script runs
    under (`/usr/bin/python3 -c "import httpx"` -> ModuleNotFoundError), so the row is a stdlib
    `urllib.request` call carrying `auth_headers_for()`. This node pins all three halves of that
    without an aggregator: the URL the request was made to, the header it carried, and that a
    200 body is parsed as JSON. A second node below sends the same call a real `HTTPError` to show
    the transport's failure path reaches the row rather than the traceback.
    """
    seen = {}

    class _Resp:
        def read(self):
            return json.dumps(ENFORCING_SIX).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        seen["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(shc, "_state_url_and_headers", lambda: (STATE_URL, {HEADER: TOKEN}))
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    body = shc._state_payload()
    assert seen["url"] == STATE_URL, seen
    assert seen["headers"].get(HEADER.lower()) == TOKEN, seen["headers"]
    assert seen["timeout"] == shc.PATH_SANDBOX_TIMEOUT_S, seen
    assert body["protected_path_sandbox"]["enforcing"] is True, body

    row = _only(shc.check_path_sandbox())
    assert row["healthy"] is True and row["entry_count"] == 6, row


def test_the_state_url_and_headers_come_from_the_aggregator_module(monkeypatch):
    """No second definition of where `/state` is or what authenticates it (#2320 clause 1).

    `app.aggregator_config.route("state")` is the route table and `auth_headers_for` is the
    credential; the check is a consumer of both, so the entry list, the port and the token each
    stay defined in exactly one place.
    """
    calls = []

    class _Cfg:
        def route(self, name):
            calls.append(("route", name))
            return STATE_URL

        def auth_headers_for(self, url):
            calls.append(("auth", url))
            return {HEADER: TOKEN}

    monkeypatch.setitem(sys.modules, "app.aggregator_config", types.SimpleNamespace(**{
        "route": _Cfg().route, "auth_headers_for": _Cfg().auth_headers_for}))
    monkeypatch.setattr(shc, "_aggregator_config", lambda: sys.modules["app.aggregator_config"])

    assert shc._state_url_and_headers() == (STATE_URL, {HEADER: TOKEN})
    assert ("route", "state") in calls, calls
    assert ("auth", STATE_URL) in calls, calls


# --------------------------------------------------------------------- the CLI gate
# The row is a fleet report, not an answer to "is lloyd-mcp up". These two nodes are `main`'s
# contract, which is why they drive `main()` and read its own output rather than calling the
# check: `--services` must not produce it, and `--format json` must carry the same healthy flag
# the text row printed.

# `check_unit_file_links` joins the list for #2714: its rows ride the default run under
# the `deploy` category, so `Overall:` would otherwise count eighteen rows this file never
# asked for and the one-row arithmetic below would read as a substrate fault.
SIBLING_CHECKS = ("check_deployed_copies", "check_ca_trust", "check_unit_enabledness",
                  "check_gpu_power_limit", "check_unit_file_links")


def _drive(module, monkeypatch, capsys, payload, fmt="json", argv_extra=()):
    """Run `main()` with every other row producer silenced and one stubbed `/state` answer."""
    for name in SIBLING_CHECKS:
        monkeypatch.setattr(module, name, lambda *a, **k: [])
    monkeypatch.setattr(module, "derived_supervisor_rows", lambda *a, **k: [])
    monkeypatch.setattr(module, "_switched_off", lambda *a, **k: set())
    # `all` empty makes main's per-service loop body never run; the other keys stay so
    # `--category`'s argparse choices still name path-sandbox.
    monkeypatch.setattr(module, "CATEGORIES", dict(module.CATEGORIES, **{"all": []}))
    monkeypatch.setattr(module, "_state_payload", lambda: payload)
    monkeypatch.setattr(sys, "argv",
                        [str(module.__file__), "--format", fmt, *argv_extra])
    module.main()
    out = capsys.readouterr().out
    return out if fmt == "text" else json.loads(out)


def _fresh():
    """`shc` under a function-scoped name: pytest refuses a module fixture beside `monkeypatch`."""
    return shc


def test_the_row_rides_the_default_run_and_the_json_flag_matches_the_text_row(monkeypatch,
                                                                             capsys):
    """#2320 clause 5: whole-fleet run only, and `--json` reports the flag the text row printed.

    Three assertions over two `main()` drives: the default run emits the row and its `healthy`
    flag is the flag the `[✓]`/`[✗]` icon in the text run was chosen from (same stubbed payload,
    both formats); `--services lloyd-mcp` gets no unsolicited sandbox row; and the row's own
    category still answers it, the way every other private category does.
    """
    shc_mod = _fresh()

    doc = _drive(shc_mod, monkeypatch, capsys, FALLBACK)
    rows = [s for s in doc["services"] if s["name"] == ROW_NAME]
    assert len(rows) == 1, doc["services"]
    assert rows[0]["healthy"] is False, rows[0]
    assert "advisory" not in rows[0], rows[0]
    assert doc["unhealthy"] >= 1, doc
    assert doc["summary"][shc_mod.PATH_SANDBOX_CATEGORY] == "unhealthy", doc["summary"]

    monkeypatch.undo()
    text = _drive(shc_mod, monkeypatch, capsys, ENFORCING_SIX, fmt="text")
    assert f"[✓] {ROW_NAME}\u2014" in text, text

    monkeypatch.undo()
    doc = _drive(shc_mod, monkeypatch, capsys, ENFORCING_SIX)
    row = [s for s in doc["services"] if s["name"] == ROW_NAME][0]
    assert row["healthy"] is True, row
    assert row["entry_count"] == 6, row

    monkeypatch.undo()
    doc = _drive(shc_mod, monkeypatch, capsys, ENFORCING_SIX,
                 argv_extra=("--services", "lloyd-mcp"))
    assert [s["name"] for s in doc["services"]] == ["lloyd-mcp"], (
        "a one-service ask gained an unsolicited substrate row: "
        f"{[s['name'] for s in doc['services']]}")

    monkeypatch.undo()
    doc = _drive(shc_mod, monkeypatch, capsys, ENFORCING_SIX,
                 argv_extra=("--category", shc_mod.PATH_SANDBOX_CATEGORY))
    assert [s["name"] for s in doc["services"]] == [ROW_NAME], doc["services"]


# ---------------------------------------------------------------------------------------------
# #3327: what the block above `PATH_SANDBOX_CATEGORY` is allowed to SAY about the policy.
#
# Everything above pins what the row DOES with a `/state` answer. These nodes pin the comment a
# future editor reads before touching that behaviour, and that comment told the truth about the
# code while lying about the policy: it presented #2109's fail-open question as one nobody had
# answered. It was wrong from the minute it landed — `431ef5d2`, the commit that shipped #2320,
# wrote it at 2026-10-07T01:46:19Z, thirty minutes AFTER the owed-check pass recorded the ruling
# at 01:16:07Z, copying the question from #2109's still-owed list instead of from the settled
# entry that pass had just appended. The same defect class #2957 pinned at
# `tests/test_funnel_census.py:873` and #2958 at `tests/test_corpus_shape.py:1026`; the node
# shape is #3092's at `tests/test_component_manifest_retention.py:812`.
# ---------------------------------------------------------------------------------------------

#: The owed-check pass that ruled #2109's fail-open question, read out of vault
#: `backlog/2109-enforce-protected-paths-in-the-bash-substrate-unsh.md`'s `owed_settled`. The
#: `outcome: ruling` entry (owed item 3, `at: '2026-10-07T01:16:07.640517'`, `follow_up: 2320`)
#: carries NO `artifact:` key at all, so the id comes from the sibling `outcome: settled` entry
#: whose `at` is byte-identical — the same pass writing both. The session is witnessed on disk at
#: `~/lloyd-data/sessions/20261006_181232_owedcheck_fbd7.json`. This constant is the only place
#: the literal is spelled in the repo, which is what makes an invented id fail below rather than
#: merely read badly.
SHC_RULING_OWED_CHECK = "20261006_181232_owedcheck_fbd7"

#: The three spellings the block used for a question that had already been settled. FRAGMENTS on
#: purpose, not phrases: #3327's own acceptance check is that a `git grep` of the contiguous
#: phrase this list refuses prints nothing repo-wide, so a refusal list that spelled the phrase
#: out would itself be the surviving hit and the clause could never pass. #3092 escaped the same
#: collision by storing the fragment `open ruling`, and the comment there says it is the only
#: place its phrases appear. Applied to THIS block only: the first fragment is a verdict token in
#: `agent-services/guardian/guardian.py` and ordinary prose in `architecture/automod.md`, neither
#: of which this item owns.
SHC_PENDING_MARKERS = ("still-open", "takes no position",
                       "only its wording may move")

#: The phrase no tracked file may hold, assembled at runtime from the fragments above so that
#: naming it here does not put it back into the corpus this suite is checking.
SHC_REFUSED_PHRASE = SHC_PENDING_MARKERS[0] + " ruling"

#: The shape a real owed-check pass id has, so a plausible-looking invention cannot satisfy a
#: node that only asks for "some token that looks like a pass".
_OWED_CHECK_ID = re.compile(r"^\d{8}_\d{6}_owedcheck_[0-9a-f]{4}$")


def _shc_prose_of(block: str) -> str:
    """A comment block as the continuous prose a reader actually reads.

    `#:` is markup, not text, and a comment wraps mid-sentence: a phrase pinned against the lines
    joined whole would grade the column the comment happened to wrap at instead of the sentence.
    `_prose_of` at `tests/test_component_manifest_retention.py:799` does the identical fold.
    """
    return " ".join(re.sub(r"^\s*#+:?\s*", "", line) for line in block.splitlines())


def _shc_sentences(block: str) -> "list[str]":
    """That prose cut into sentences, which is the scope every claim below is graded in.

    A block-wide search is satisfiable by any sentence in the neighbourhood: a block that says
    the question is UNRULED still contains the letters `ruled` inside `unruled`, and the #3092
    review caught exactly that shape.
    """
    flat = _shc_prose_of(block)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if s.strip()]


def _comment_above_shc(symbol: str) -> str:
    """The comment block sitting directly above `symbol` in the shipped script."""
    lines = SCRIPT.read_text(encoding="utf-8").splitlines()
    at = next((i for i, ln in enumerate(lines) if ln.startswith(f"{symbol} =")), None)
    assert at is not None, (
        f"`{symbol}` is not a module-level constant of the script any more, so this block has "
        "no address and the checks below have no subject")
    start = at
    while start > 0 and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    return "\n".join(lines[start:at])


def _assert_decided(sentences: "list[str]", *, where: str) -> str:
    """Clause 2 and 3: the sentence that records the ruling has to record all of it.

    One sentence carries the pass id, and that same sentence must name the item, be dated, say
    the question was ruled, and say WHICH WAY it went. Sentence-scoped, so deleting it fails
    rather than leaving the claim to some other line that never made it.
    """
    hits = [s for s in sentences if SHC_RULING_OWED_CHECK in s]
    assert hits, (
        f"no sentence of {where} names the owed-check pass {SHC_RULING_OWED_CHECK} that ruled "
        "#2109's fail-open question, so the ruling in this block has no witness a reader can go "
        "and read")
    decided = hits[0]
    assert "#2109" in decided, (
        f"the sentence naming the pass does not name the item that decided, so the ruling has "
        f"no owner: {decided}")
    assert re.search(r"\bruled\b", decided, re.IGNORECASE), (
        f"the sentence that cites the pass never says the question was ruled — `unruled` and "
        f"`overruled` both contain the letters, which is why this is a word-boundary match: "
        f"{decided}")
    assert "2026-10-07" in decided, (
        f"the ruling has to arrive dated in the sentence that records it, or a reader cannot "
        f"tell it from the open question it replaced: {decided}")
    for way in ("keep fail-open as shipped", "decline refusing Bash outright"):
        assert way in decided, (
            f"the sentence records that something was ruled but not that it went {way!r}, so "
            f"the direction of the decision is left to guesswork: {decided}")
    return decided


def _assert_grade_tied_to_ruling(sentences: "list[str]", *, where: str) -> str:
    """Clause 4's prose half: the red grade is stated as the ruling's other half."""
    hits = [s for s in sentences if "advisory" in s]
    assert hits, (
        f"no sentence of {where} says what an `advisory` row would do, so the block no longer "
        "explains why this row is graded FAIL — the grade starts looking like this script being "
        "strict on its own account")
    grade = hits[0]
    assert "FAIL" in grade, (
        f"the sentence about `advisory` does not name the grade the row gives instead: {grade}")
    assert re.search(r"\brul(ing|ed)\b", grade, re.IGNORECASE), (
        f"the grade is no longer attributed to the ruling, which is the whole of clause 4: "
        f"{grade}")
    assert "visible where a human reads" in grade, (
        f"the ruling's own condition — the degraded state must be visible where a human reads — "
        f"is what makes the red row the ruling rather than a preference, and it is gone: {grade}")
    assert "`_scored_rows`" in grade and "`Overall:`" in grade, (
        f"the mechanism has to be named for the claim to be checkable: `advisory` rows are "
        f"dropped by `_scored_rows`, which is what would leave `Overall:` green: {grade}")
    return grade


def test_the_block_above_the_category_no_longer_shows_the_ruling_as_pending():
    """#3327 clause 1: three phrases that read "nobody has decided this yet" are gone.

    Folded first, because the block wraps: `still-open` was hyphenated across no line but the
    other two markers were, and a line-wise search would grade the wrap column.
    """
    block = _comment_above_shc("PATH_SANDBOX_CATEGORY")
    assert block.strip(), (
        "`PATH_SANDBOX_CATEGORY` has no comment block above it any more, so the row's policy "
        "rationale is gone with it")
    prose = _shc_prose_of(block)
    for marker in SHC_PENDING_MARKERS:
        assert marker not in prose, (
            f"the block calls #2109's fail-open policy pending again with {marker!r}, but the "
            f"owed-check pass {SHC_RULING_OWED_CHECK} ruled it on 2026-10-07 — a reader who "
            "believes this sentence books work that is already done")


def test_the_block_records_the_ruling_as_made_and_attributed_to_its_pass():
    """#3327 clause 2: the block says the question is RULED, and which way it went.

    Deleting the stale sentence would not be enough on its own: a block that merely stops asking
    leaves a reader who meets this constant in a year unable to tell whether the fail-open branch
    was examined and kept or simply inherited — which is how #2958's and #2957's stale sentences
    got re-filed as new work.
    """
    _assert_decided(_shc_sentences(_comment_above_shc("PATH_SANDBOX_CATEGORY")),
                    where="the block above PATH_SANDBOX_CATEGORY")


def test_the_ruling_names_the_owed_check_pass_that_recorded_it_and_no_invented_id_passes():
    """#3327 clause 3: the literal, and the reason it is THAT literal.

    The pass id is not this round's to compose. #2109's `outcome: ruling` entry carries no
    `artifact:` key, so the id is the sibling `owed_settled` entry's — the one whose `at` is
    byte-identical to the ruling's `2026-10-07T01:16:07.640517`, which is the same pass:
    `20261006_181232_owedcheck_fbd7`. Two rails keep an invented id out: the constant's own shape
    (a date, a time, the literal `owedcheck`, four hex digits), and the requirement that the
    id-bearing sentence is the ruling-bearing sentence, so a plausible token parked in some
    unrelated line satisfies nothing.
    """
    assert _OWED_CHECK_ID.match(SHC_RULING_OWED_CHECK), (
        f"{SHC_RULING_OWED_CHECK!r} is not shaped like an owed-check pass id, so this test's "
        "own constant is the invented thing it exists to refuse")
    block = _comment_above_shc("PATH_SANDBOX_CATEGORY")
    sentences = _shc_sentences(block)
    with_id = [s for s in sentences if SHC_RULING_OWED_CHECK in s]
    assert len(with_id) == 1, (
        f"the pass id appears in {len(with_id)} sentences; one ruling wants exactly one "
        "sentence carrying its witness, or a second, unruled claim can borrow the id")
    assert with_id[0] == _assert_decided(sentences, where="the block above PATH_SANDBOX_CATEGORY")
    assert SHC_RULING_OWED_CHECK in _shc_prose_of(block), (
        "the id folded out of the block is not the id in the constant — a wrap inside the id "
        "would break every grep a reader follows")


def test_the_block_ties_the_red_grade_to_the_ruling_and_the_row_honours_it():
    """#3327 clause 4, both halves: the sentence says the red row IS the ruling, and the code
    still behaves the way that sentence says.

    The seam this crosses is the one #2320 opened: a `/state` payload read over HTTP becomes a
    graded row, and `_scored_rows` is the gate that decides whether `Overall:` sees it. The
    comment now claims the red grade is the ruling's bound — visible where a human reads — so the
    claim is checked against the shipped function rather than left as prose: a fallback payload
    must still yield one graded-FAIL row with NO `advisory` key. That is the same property
    `test_a_fallback_payload_is_a_graded_fail_that_names_the_error` pins for #2320; re-asserting
    the minimum here is what makes THIS sentence falsifiable, and it edits nothing above.

    The tuple half of the clause is checked with the phrase assembled from its own fragments: no
    entry may contain the contiguous phrase, and neither the script nor this file may contain it,
    because the refusal list being the surviving hit is how the grep fails while the rule passes.
    """
    _assert_grade_tied_to_ruling(_shc_sentences(_comment_above_shc("PATH_SANDBOX_CATEGORY")),
                                 where="the block above PATH_SANDBOX_CATEGORY")

    for marker in SHC_PENDING_MARKERS:
        assert SHC_REFUSED_PHRASE not in marker, (
            f"{marker!r} spells the whole phrase it exists to refuse, so the refusal list is "
            "itself the repo-wide hit and clause 4 can never pass")
    for path in (SCRIPT, Path(__file__)):
        text = path.read_text(encoding="utf-8")
        assert SHC_REFUSED_PHRASE not in text, (
            f"{path.name} holds the contiguous phrase again, which is the string #3327's "
            "acceptance check greps for repo-wide")

    row = _only(_rows(payload=FALLBACK))
    assert row["healthy"] is False, row
    assert "advisory" not in row, (
        "the block says a fallback is graded rather than advisory; an `advisory` key would be "
        f"dropped by `_scored_rows` and the sentence would be a lie: {row}")


def test_the_pending_wording_check_refuses_the_wording_it_exists_to_refuse():
    """The red side, without which clauses 1-4 are unfalsifiable.

    The seed is the block as #3327 found it — reconstructed from the fragments rather than quoted,
    because quoting it would reintroduce the very phrase the clause forbids anywhere in the repo.
    It must trip the pending-marker guard AND both claim guards; a guard that had gone deaf would
    sail through it. The paired green half is the shipped block, so a guard that had gone deaf
    fails here as well.
    """
    shipped = ("It reports the state and " + SHC_PENDING_MARKERS[1] + " on #2109's "
               + SHC_PENDING_MARKERS[0] + " ruling (whether a fallback should refuse Bash "
               "outright the way `_tool_sandbox.refusal` does, or keep failing open with a "
               "warning): whichever way that goes, \"is it enforcing\" is the question this row "
               "answers, and " + SHC_PENDING_MARKERS[2] + " afterwards.")
    assert SHC_REFUSED_PHRASE in shipped, (
        "the seed is no longer the sentence #3327 was filed on, so refusing it proves nothing")

    for marker in SHC_PENDING_MARKERS:
        assert marker in shipped, (
            f"the seed no longer carries {marker!r}, so the green half below is all this node "
            "would be checking")
    seed = _shc_sentences(shipped)
    with pytest.raises(AssertionError):
        _assert_decided(seed, where="the seed")
    with pytest.raises(AssertionError):
        _assert_grade_tied_to_ruling(seed, where="the seed")

    live = _shc_sentences(_comment_above_shc("PATH_SANDBOX_CATEGORY"))
    for marker in SHC_PENDING_MARKERS:
        assert marker not in _shc_prose_of(_comment_above_shc("PATH_SANDBOX_CATEGORY"))
    _assert_decided(live, where="the block above PATH_SANDBOX_CATEGORY")
    _assert_grade_tied_to_ruling(live, where="the block above PATH_SANDBOX_CATEGORY")


def test_the_ruling_prose_is_a_comment_and_no_path_sandbox_value_moved():
    """#3327 clause 5: this item's edit is prose, and the bound it sits on is untouched.

    Every line of the block is a comment line, which is what makes the edit reviewable as prose;
    and the three constants the row is built from keep the exact values #2320 shipped, because
    "only its wording may move" is a promise the next reader can no longer make and this is where
    it is enforced instead. The regex, the verdict vocabulary and the exit codes are the same
    objects the nodes above already drive, and none of them is named here a second time with a
    looser value.
    """
    block = _comment_above_shc("PATH_SANDBOX_CATEGORY")
    for ln in block.splitlines():
        assert ln.lstrip().startswith("#"), (
            f"a line of the block is not a comment, so prose and behaviour are now the same "
            f"edit: {ln!r}")
    assert shc.PATH_SANDBOX_CATEGORY == "path-sandbox", shc.PATH_SANDBOX_CATEGORY
    assert shc.PATH_SANDBOX_KEY == "protected_path_sandbox", shc.PATH_SANDBOX_KEY
    assert shc.PATH_SANDBOX_TIMEOUT_S == 10.0, shc.PATH_SANDBOX_TIMEOUT_S
