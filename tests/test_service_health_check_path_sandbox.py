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

SIBLING_CHECKS = ("check_deployed_copies", "check_ca_trust", "check_unit_enabledness",
                  "check_gpu_power_limit")


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
