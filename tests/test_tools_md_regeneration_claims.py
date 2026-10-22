"""`architecture/tools.md` §11 is the section a regeneration pass obeys, and until
#3134 it told that pass two things the code contradicts, and left the one load-bearing
route undocumented.

What the page used to assert: that `tools/list` "is the only route that can see the
Thunderbird bridge's 40 rows at all, so a regeneration that skips it drops or freezes
them", and — one paragraph later — "Offline, or in a worktree (Thunderbird will read
zero there)". The two halves were wrong in opposite directions and the second hid the
first. The offline dump prints all 40 rows in the live checkout (measured 2026-10-21 at
`fee96288` and again for this change: 147 rows, 40 of them `thunderbird`, each carrying
its marks and its `required` list), because `agent_mcp/thunderbird.py` opens its own
`node mcp-bridge.cjs` stdio session from inside the module, so any process importing
`agent_mcp.main` reaches the bridge; #1053 gated the HTTP routes only
(`OPEN_PATHS = frozenset({"/health"})`). The two routes' name sets are identical — 147
vs 147, zero names in either direction. The zero is real, but only inside a worktree
(the bridge directory is gitignored, so `BRIDGE_PATH` resolves to a file that is not
there) or with Thunderbird closed. A pass that believes the first sentence regenerates
§6 from the wire and loses three of its seven marks, because `FX`/`RX`/`PM` and the
module attribution are not on the wire — the reverse of what that sentence claimed the
wire was for.

The nodes below pin the corrected section, clause by clause, plus the code it cites.

- **The route claim.** The sentence is gone, and §11 states affirmatively that the
  offline dump carries the bridge's rows whenever the bridge is reachable, naming the
  in-module stdio session as the reason and `OPEN_PATHS` as the reason #1053 does not
  apply.
- **The zero conditions.** The worktree condition carries its mechanism
  (`agent-services/.gitignore`, `BRIDGE_PATH`, `FileNotFoundError`, the cached-empty
  `_discover`); the not-running condition is named; the false "Offline, or in a
  worktree … will read zero" attribution may not come back.
- **The served-route command.** §11 gained a fenced block that reads the aggregator
  token file and POSTs a `tools/list` request with `X-Lloyd-Aggregator-Token`. Two
  nodes parse it and three run it against a stub aggregator on a loopback port, which
  is what makes the header, the `Accept` the streamable-HTTP mount demands, and the
  four-column row shape real rather than described.

Two traps this file is written around.

**The fence parser accepts a language tag.** The shared helper in
`tests/test_architecture_coverage_doc_claims.py` is
`re.compile(r"^```\\n.*?^```$", re.S | re.M)` — fences with **no** tag only — and every
block in §11 opens with ``` ```bash ```, so reusing it would have matched zero of the
three blocks and then reported whichever defect it was hunting for as absent. `_FENCE`
below takes an optional tag, and `test_the_fence_parser_accepts_a_language_tag` feeds
both patterns a synthetic page and fails if this file's pattern degenerates into that
one.

**The live aggregator is not a precondition.** :8500 being up during a gate rung is not
a property of a doc diff, and #3134 names the live POST as its post-landing check. So
the HTTP leg runs against a stub that answers the envelope the real mount sends
(`{"jsonrpc": "2.0", "id": …, "result": {"tools": […]}}`) and gates in the order the
live server gates in, which the second half of
`test_a_copy_that_sends_no_credential_is_refused` pins: credential first, because
`require_credential()` wraps the whole app (`agent_mcp/main.py:1148`) — so a request
missing both halves measures as the 401, not the 406 — and behind it the mount's own
rule, that an `Accept` header exist at all (`*/*` is served; the bare `urllib.request`
default, no header, is the 406). The substitutions that redirect it are the same
mechanical kind
`tests/test_tools_md_inventory_parity.py` already makes for `.venvs/lloyd/bin/python`.
Comparing the two dumps is a set comparison and §11 now says so: this box's default
`en_US.UTF-8` collation is punctuation-insensitive and sorts `_BackgroundTaskDrain`
first where Python's `sorted()` puts `Bash` first, which between two identical
147-name dumps reads as 134 phantom differences.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_mcp import annotations as A  # noqa: E402
from agent_mcp import thunderbird as TB  # noqa: E402

PAGE = (ROOT / "architecture" / "tools.md").read_text(encoding="utf-8")

HEADER = "X-Lloyd-Aggregator-Token"
REFUSAL_CODE = "AGGREGATOR_UNAUTHENTICATED"
#: §11's heading and the heading that closes the section. A moved heading must raise,
#: not return "": an empty section makes every set below empty and every defect list
#: clean, which is a guard that reports "no defects" forever.
_S11_HEAD = "## 11. Regenerating the inventory"
_S11_END = "## Review log"
#: Language tag optional. `^```\n` alone (the coverage file's helper) matches none of
#: §11's blocks, all of which are tagged ```bash.
_FENCE = re.compile(r"^```[ \t]*([A-Za-z0-9_+-]*)\n(.*?)^```[ \t]*$", re.S | re.M)
#: That helper, kept here so the tolerance node can prove the difference instead of
#: asserting it in a comment.
_FENCE_NO_TAG = re.compile(r"^```\n.*?^```$", re.S | re.M)
_METHOD = '"method": "tools/list"'
#: The credential clause of §11's served command, and the Accept clause. Matched by
#: regex so a re-indent of the page does not silently turn a planted defect into a
#: no-op replace.
#: …and it stops before the dict's closing brace, so deleting the clause leaves a
#: command that still runs — one that sends no credential, which is the state under
#: test — rather than a SyntaxError that would exit non-zero for the wrong reason.
_HEADER_CLAUSE = re.compile(
    r',[ \t]*\n[ \t]*"X-Lloyd-Aggregator-Token": tok\.read_text\(\)\.strip\(\)')
_ACCEPT_CLAUSE = '"Accept": "application/json",'


def _section11(page: str) -> str:
    if _S11_HEAD not in page:
        raise LookupError(f"{_S11_HEAD!r} is not a heading of the page any more, so "
                          "every §11 claim below would be graded against nothing")
    start = page.index(_S11_HEAD)
    end = page.find(_S11_END, start)
    if end == -1:
        raise LookupError(f"no {_S11_END!r} after {_S11_HEAD!r}: the boundary this file "
                          "parses to is gone")
    return page[start:end]


def _perturb(page: str, needle: str | re.Pattern, repl: str = "") -> str:
    """Delete `needle` **inside §11** and return a copy of the page.

    Whole-page replaces are not good enough: several of these strings also occur
    earlier in the page, so `replace(..., 1)` would edit a paragraph that is not under
    test and leave §11 untouched, and the node would report a planted defect as found
    when it was never planted.
    """
    sec = _section11(page)
    if isinstance(needle, re.Pattern):
        new, n = needle.subn(repl, sec)
        assert n, f"the planted defect {needle.pattern!r} is not in §11 to remove"
    else:
        assert needle in sec, f"the planted defect {needle!r} is not in §11 to remove"
        new = sec.replace(needle, repl)
    return page.replace(sec, new, 1)


def _fences(page: str) -> list[tuple[str, str]]:
    blocks = _FENCE.findall(_section11(page))
    if not blocks:
        raise LookupError("no fenced block parsed out of §11 — the fence pattern "
                          "matched nothing at all, which is not the same as §11 having "
                          "no commands in it")
    return blocks


def _served_block(page: str) -> str:
    """§11's token-bearing `tools/list` command: the one block carrying both the
    credential header and the method."""
    hits = [b for _tag, b in _fences(page) if HEADER in b and _METHOD in b]
    if len(hits) != 1:
        raise LookupError(
            f"§11 has {len(hits)} fenced blocks carrying both {HEADER} and {_METHOD}, "
            "expected exactly 1: the route #3134 added has to be one copy-pasteable "
            "command, and two blocks means the page documents the route twice with "
            "nothing keeping the copies in step")
    return hits[0]


# ── clause 1: the route claim ───────────────────────────────────────────────

#: The false sentence, in the words it was written in. Held as one fragment so a
#: re-phased revival of the claim is caught by the affirmative checks, not by this one.
_ONLY_ROUTE = "only route that can see"


def _route_claim_defects(page: str) -> list[str]:
    sec = _section11(page)
    defects: list[str] = []
    if _ONLY_ROUTE in sec:
        defects.append(f"§11 still says tools/list is the {_ONLY_ROUTE!r} route for the "
                       "bridge's rows; the offline dump prints all 40 whenever the "
                       "bridge is reachable")
    # The affirmative half. A page that merely deletes the sentence has not told a
    # regeneration pass what is true, and the pass fills the gap from the wire.
    for needle, why in (
        ("whenever the bridge is reachable",
         "the corrected claim that the dump carries the bridge's rows is missing"),
        ("_get_pool",
         "the reason the offline dump reaches the bridge — the stdio session the module "
         "opens inside itself — is not named"),
        ("OPEN_PATHS",
         "the reason #1053's credential does not apply offline is not named"),
        ("inputSchema",
         "the page no longer states which columns the wire does carry, so the "
         "offline-only status of FX/RX/PM and of the module column is unstated"),
    ):
        if needle not in sec:
            defects.append(needle + ": " + why)
    return defects


def test_section11_no_longer_claims_tools_list_is_the_only_route():
    """#3134 clause 1."""
    assert _route_claim_defects(PAGE) == []


def test_a_copy_replanting_the_only_route_sentence_is_reported():
    """The clause-1 node can fail: the deleted sentence, planted back in."""
    planted = _perturb(PAGE, "Neither route is the only one that sees the Thunderbird",
                       f"That is the {_ONLY_ROUTE} the Thunderbird")
    defects = _route_claim_defects(planted)
    assert any(_ONLY_ROUTE in d for d in defects), defects


def test_a_copy_that_strips_the_reachable_bridge_claim_is_reported():
    """Deleting the affirmative claim piece by piece is how a true section quietly
    becomes a silent one."""
    for needle in ("whenever the bridge is reachable", "_get_pool", "OPEN_PATHS",
                   "inputSchema"):
        defects = _route_claim_defects(_perturb(PAGE, needle, "that route"))
        assert any(needle in d for d in defects), (needle, defects)


# ── clause 2: the conditions that really zero the bridge ───────────────────

_FALSE_ZERO = re.compile(r"Offline, or in a worktree|Thunderbird will read zero")


def _zero_condition_defects(page: str) -> list[str]:
    sec = _section11(page)
    defects: list[str] = []
    if _FALSE_ZERO.search(sec):
        defects.append("§11 again attributes a zero to simply being offline "
                       f"({_FALSE_ZERO.pattern}) — the live checkout prints the 40 rows")
    for group, needles, why in (
        ("worktree", ("automod worktree", "services/thunderbird-mcp/", "BRIDGE_PATH",
                      "FileNotFoundError", "exporting no tools"),
         "the worktree zero is not stated with its mechanism"),
        ("not-running", ("Thunderbird not running",),
         "the Thunderbird-closed zero is not named"),
    ):
        missing = [n for n in needles if n not in sec]
        if missing:
            defects.append(f"{group}: {why}; absent {missing}")
    return defects


def test_section11_names_the_two_conditions_that_really_zero_the_bridge_rows():
    """#3134 clause 2: a worktree (bridge directory untracked, so `BRIDGE_PATH` resolves
    to a file that is not there) or Thunderbird not running — never "offline"."""
    assert _zero_condition_defects(PAGE) == []


def test_a_copy_restoring_the_false_offline_zero_is_reported():
    planted = _perturb(PAGE, "Offline, this prints every tool",
                       "Offline, or in a worktree (Thunderbird will read zero there), "
                       "this prints every tool")
    defects = _zero_condition_defects(planted)
    assert any("attributes a zero" in d for d in defects), defects


@pytest.mark.parametrize("stripped", (
    "automod worktree",            # the first condition's name
    "services/thunderbird-mcp/",   # the gitignore entry that keeps the bridge out
    "BRIDGE_PATH",                 # the path that resolves inside the worktree
    "FileNotFoundError",           # what _get_pool raises there
    "exporting no tools",          # what _discover logs and caches
    "Thunderbird not running",     # the second condition
))
def test_a_copy_that_drops_part_of_a_zero_condition_is_reported(stripped):
    """The worktree half has to carry its reason or the next reader re-derives the wrong
    one — which is how "offline" got written there in the first place."""
    defects = _zero_condition_defects(_perturb(PAGE, stripped, "the bridge"))
    assert defects, defects


# ── clauses 3 and 4: the served-route command, run for real ────────────────

_TOKEN_TEXT = "stub-aggregator-token-0123456789abcdef"

#: Canned `tools/list` result in the envelope the real mount returns. `email_send` is a
#: name the gitignored bridge serves, so it exercises the module column's `thunderbird`
#: side; `Bash` is served by Python, so it exercises the `-` side; `stub_noargs` has no
#: `required` list at all, so a row must still print four fields.
_CANNED = [
    {"name": "email_send", "description": "Send one email.",
     "inputSchema": {"type": "object", "properties": {"to": {"type": "string"}},
                     "required": ["to", "subject", "body"]},
     "annotations": {"readOnlyHint": False, "destructiveHint": False,
                     "idempotentHint": False, "openWorldHint": True}},
    {"name": "Bash", "description": "Run a shell command.",
     "inputSchema": {"type": "object", "properties": {"command": {"type": "string"}},
                     "required": ["command"]},
     "annotations": {"readOnlyHint": False, "destructiveHint": True,
                     "idempotentHint": False, "openWorldHint": False}},
    {"name": "stub_noargs", "description": "A tool with no required arguments.",
     "inputSchema": {"type": "object", "properties": {}},
     "annotations": {"readOnlyHint": True, "destructiveHint": False,
                     "idempotentHint": True, "openWorldHint": False}},
]
_BRIDGE_NAMES = set(TB.TOOL_NAME_MAP.values())


def _wire_marks(name: str, ann: dict) -> str:
    """§11's marks vocabulary — the offline dump's own rule, transcribed once.

    `RO`/`DX`/`ID`/`OW` are the four hints the wire carries; `ID` is suppressed on a
    read-only tool; `RX`/`FX`/`PM` are computed from `agent_mcp/annotations.py` because
    they are not served. Pinning the served block against this transcription is what
    says the two routes print the same column: it fails if the block drops a mark,
    drops the `ID` suppression, or invents one the dump does not print.
    """
    return " ".join(k for k, on in (
        ("RO", ann.get("readOnlyHint")),
        ("DX", ann.get("destructiveHint")),
        ("ID", ann.get("idempotentHint") and not ann.get("readOnlyHint")),
        ("OW", ann.get("openWorldHint")),
        ("RX", name in A.REPEAT_EXPECTED),
        ("FX", A.side_effecting(name)),
        ("PM", name in A.PLAN_MODE_ALWAYS_ALLOWED)) if on)


class _Stub(BaseHTTPRequestHandler):
    """Answers like the streamable-HTTP mount: `Accept` first, then the credential."""

    expected_token = _TOKEN_TEXT
    seen: dict = {}

    def do_POST(self):  # noqa: N802 — BaseHTTPRequestHandler's name
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
        type(self).seen = {"path": self.path,
                           "header": self.headers.get(HEADER),
                           "accept": self.headers.get("Accept"),
                           "body": json.loads(raw) if raw.strip() else {}}
        # This order is the live server's, not the other way round:
        # `require_credential()` wraps the whole aggregator app
        # (`agent_mcp/main.py:1148`), so a request missing the credential is a 401
        # whatever else it says, and a request missing both halves measures as the
        # 401 — which is what
        # `test_a_copy_that_sends_no_credential_is_refused` asserts by deleting both.
        if self.headers.get(HEADER) != self.expected_token:
            self._send(401, {"error": f"the aggregator requires a credential in the "
                                      f"{HEADER} header (#1053)",
                             "code": REFUSAL_CODE})
            return
        # Behind the credential gate, the mount's own rule is on the *presence* of an
        # Accept header: curl's default `*/*` is served, and so is
        # `application/json`; a bare `urllib.request`, which sends no Accept line at
        # all, gets the 406. Measured against :8500 on 2026-10-21.
        if self.headers.get("Accept") is None:
            self._send(406, {"error": "406: Not Acceptable"})
            return
        self._send(200, {"jsonrpc": "2.0", "id": 1, "result": {"tools": _CANNED}})

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence the per-request line
        pass


@pytest.fixture()
def stub():
    _Stub.seen = {}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


def _materialize(block: str, port: int, token_file: Path) -> str:
    """Turn the page's command into one this test may run.

    Three literal substitutions, each asserted, so the node dies if the documented
    command stopped containing the thing being substituted — a block that had lost the
    token path or the port would otherwise run against the live aggregator, or not at
    all.
    """
    for old, new in ((".venvs/lloyd/bin/python", sys.executable),
                     ("127.0.0.1:8500", f"127.0.0.1:{port}"),
                     ("~/.local/state/lloyd/aggregator-token", str(token_file))):
        assert old in block, f"§11's served command no longer contains {old!r}"
        block = block.replace(old, new)
    return block


def _run(block: str, port: int, token_file: Path) -> subprocess.CompletedProcess:
    return subprocess.run(_materialize(block, port, token_file), shell=True, cwd=ROOT,
                          capture_output=True, text=True, timeout=300,
                          env=dict(os.environ, PYTHONPATH=str(ROOT)))


def _token_file(tmp_path: Path) -> Path:
    tok = tmp_path / "aggregator-token"
    tok.write_text(_TOKEN_TEXT, encoding="utf-8")
    tok.chmod(0o600)
    return tok


def test_the_served_block_carries_the_token_header_and_the_method():
    """#3134 clause 3 (parse): §11 ships a command block with both halves of the request
    the aggregator requires."""
    block = _served_block(PAGE)
    assert HEADER in block and _METHOD in block
    assert "aggregator-token" in block, (
        "the block must read the token file rather than carry a secret of its own")


def test_a_block_copy_missing_either_half_is_not_a_served_route_command():
    """With either half gone, §11 has no runnable token-bearing `tools/list` command —
    which is the state #3134 was filed against."""
    block = _served_block(PAGE)
    no_header = _HEADER_CLAUSE.sub(",", block, count=1)
    assert no_header != block, "the credential clause this node removes has moved"
    with pytest.raises(LookupError):
        _served_block(_perturb(PAGE, block, no_header))
    with pytest.raises(LookupError):
        _served_block(_perturb(PAGE, _METHOD, '"method": "tools/call"'))


def test_the_fence_parser_accepts_a_language_tag():
    """The trap #3134 warned about, pinned: §11's blocks are all tagged ```bash, and the
    coverage file's `^```\\n` helper matches none of them — a guard built on it would
    have reported "no defects" because it had found no commands at all."""
    mine = _fences(PAGE)
    assert len(mine) == 3, (
        f"§11 should hold the /health curl, the offline dump and the served dump; "
        f"parsed {len(mine)}: {[t for t, _ in mine]}")
    assert all(tag == "bash" for tag, _ in mine), [tag for tag, _ in mine]
    # The tag-intolerant helper does not simply find nothing here: run over §11 it finds
    # exactly one span, opened at a closing fence and shut at the next block's, whose
    # body is a paragraph of prose plus somebody else's command. That is the failure
    # mode worth pinning — a guard built on it is not visibly blind, it grades the
    # wrong text and calls the result a pass.
    spurious = _FENCE_NO_TAG.findall(_section11(PAGE))
    assert len(spurious) == 1, spurious
    assert not set(spurious) & {b for _t, b in mine}, (
        "the tag-intolerant pattern recovered a real §11 block; the contrast this node "
        "exists to draw no longer holds")
    planted = ("```bash\necho tagged\n```\n\n```sh\necho other-tag\n```\n\n"
               "```\necho no-tag\n```\n")
    assert [b for _t, b in _FENCE.findall(planted)] == ["echo tagged\n",
                                                        "echo other-tag\n",
                                                        "echo no-tag\n"]
    blind = set(_FENCE_NO_TAG.findall(planted))
    assert not {"echo tagged\n", "echo other-tag\n"} & blind, (
        "the two patterns have converged, so this file's fence parser is once again "
        "blind to every tagged block")


def test_the_served_block_posts_the_credential_and_the_method_it_documents(stub,
                                                                           tmp_path):
    """Clause 3 (run): the header is on the wire, the method is `tools/list`, the path is
    `/mcp`, and the request asks for JSON — the four things the prose promises."""
    proc = _run(_served_block(PAGE), stub.server_address[1], _token_file(tmp_path))
    assert proc.returncode == 0, proc.stdout[-800:] + proc.stderr[-1500:]
    seen = _Stub.seen
    assert seen["header"] == _TOKEN_TEXT, (
        f"the aggregator saw {HEADER}={seen['header']!r}; the block must send the value "
        "it read from the token file")
    assert seen["body"].get("method") == "tools/list", seen["body"]
    assert seen["path"] == "/mcp", seen
    assert "application/json" in (seen["accept"] or ""), seen


def test_a_copy_that_sends_no_credential_is_refused(stub, tmp_path):
    """The green above is earned: with no header the mount refuses and the documented
    command exits non-zero rather than printing an empty table.

    The second half of this node is the ordering §11 states — that the credential gate is
    `require_credential()` wrapping the app (`agent_mcp/main.py:1148`), so it is the 401
    that answers a request missing *both* halves. It is also the ordering this file's
    stub has to keep: were the gate modelled Accept-first, the request below would come
    back 406 and the claim in the page would be standing on a fixture that contradicts
    the server it stands in for.
    """
    tok = _token_file(tmp_path)
    block = _HEADER_CLAUSE.sub(",", _served_block(PAGE), count=1)
    assert HEADER not in block, "the credential clause this node removes has moved"
    proc = _run(block, stub.server_address[1], tok)
    assert proc.returncode != 0, (
        "a request with no credential printed a tool table anyway, so the stub is not "
        "gating the header and the passing node above proves nothing")
    assert _Stub.seen["header"] is None, _Stub.seen

    both_gone = _perturb(PAGE, _ACCEPT_CLAUSE, "")
    both_gone = _HEADER_CLAUSE.sub(",", _served_block(both_gone), count=1)
    proc = _run(both_gone, stub.server_address[1], tok)
    assert proc.returncode != 0, (
        "a request with neither the credential nor an Accept header printed a tool "
        "table; neither gate is doing anything")
    assert "401" in proc.stderr and "406" not in proc.stderr, (
        f"a request missing both halves answered from the wrong gate: "
        f"{proc.stderr[-400:]}")
    assert _Stub.seen["header"] is None and _Stub.seen["accept"] is None, _Stub.seen


def test_a_copy_that_sends_no_accept_header_is_refused(stub, tmp_path):
    """§11 says the block's explicit `Accept` line is what makes it portable across HTTP
    clients: behind the credential gate the mount answers 406 to a POST carrying no
    `Accept` header at all, which is exactly what a bare `urllib.request` sends."""
    served = _served_block(_perturb(PAGE, _ACCEPT_CLAUSE, ""))
    assert _ACCEPT_CLAUSE not in served, "the Accept clause this node removes has moved"
    proc = _run(served, stub.server_address[1], _token_file(tmp_path))
    assert proc.returncode != 0, (
        "the page claims a request with no Accept header is refused; the stub answered "
        f"it: {_Stub.seen}")
    assert _Stub.seen["accept"] is None, _Stub.seen
    assert "406" in proc.stderr, proc.stderr[-400:]


def test_a_wildcard_accept_is_served_as_our_own(stub, tmp_path):
    """The other half of the same sentence, which is why §11 says the rule is on the
    header's presence and not on its content: curl's default `Accept: */*` is served.
    Delete this node and the page could claim the 406 is a content negotiation and no
    one would notice, because the node above only ever sees the refusal."""
    served = _served_block(PAGE).replace('"Accept": "application/json"',
                                         '"Accept": "*/*"', 1)
    proc = _run(served, stub.server_address[1], _token_file(tmp_path))
    assert proc.returncode == 0, proc.stdout[-400:] + proc.stderr[-600:]
    assert _Stub.seen["accept"] == "*/*", _Stub.seen


def test_the_served_block_prints_the_offline_row_shape(stub, tmp_path):
    """#3134 clause 4: one line per tool, `module name marks required` tab-separated, so
    the two dumps diff name for name."""
    proc = _run(_served_block(PAGE), stub.server_address[1], _token_file(tmp_path))
    assert proc.returncode == 0, proc.stdout[-800:] + proc.stderr[-1500:]
    rows = [l for l in proc.stdout.splitlines() if not l.startswith("TOTALS")]
    assert len(rows) == len(_CANNED), f"printed {len(rows)} rows for {len(_CANNED)} tools"
    by_name = {l.split("\t")[1]: l for l in rows}
    assert set(by_name) == {t["name"] for t in _CANNED}, rows
    for tool in _CANNED:
        name = tool["name"]
        fields = by_name[name].split("\t")
        assert len(fields) == 4, (
            f"{name}: {len(fields)} tab-separated fields, not module/name/marks/required "
            "— the diff against the offline dump is column by column, so a missing "
            "column is a whole-file diff")
        module, _n, marks, reqd = fields
        want_module = "thunderbird" if name in _BRIDGE_NAMES else "-"
        assert module == want_module, (
            f"{name}: module column {module!r}, expected {want_module!r} — the wire "
            "attributes no module, so only the bridge's own names may be attributed")
        assert marks == _wire_marks(name, tool["annotations"]), by_name[name]
        assert reqd == ",".join(tool["inputSchema"].get("required") or []), by_name[name]
    totals = [l for l in proc.stdout.splitlines() if l.startswith("TOTALS")]
    assert totals == [f"TOTALS\ttools={len(_CANNED)}"
                      f"\tbridge={sum(t['name'] in _BRIDGE_NAMES for t in _CANNED)}"], (
        "the block stopped printing its totals; §11 says the served dump reports how "
        "many of its rows are the bridge's")


def test_a_copy_that_drops_a_column_is_caught(stub, tmp_path):
    """The four-column assertion can fail: the required-arguments column deleted."""
    block = _served_block(PAGE).replace('",".join(reqd),', "", 1)
    assert block != _served_block(PAGE), "the print this node perturbs has moved"
    proc = _run(block, stub.server_address[1], _token_file(tmp_path))
    assert proc.returncode == 0, proc.stderr[-800:]
    rows = [l for l in proc.stdout.splitlines() if not l.startswith("TOTALS")]
    assert any(len(l.split("\t")) != 4 for l in rows), (
        "deleting the required-arguments column still printed four fields, so the shape "
        "node above is counting something other than the columns")


# ── clause 5: how the token comes to exist ─────────────────────────────────

def _token_note_defects(page: str) -> list[str]:
    sec = _section11(page)
    defects: list[str] = []
    for needle, why in (
        ("os.O_CREAT | os.O_EXCL",
         "the create-exclusive write that makes the token a boot-time artefact is not "
         "cited: nobody provisions this file"),
        ("0o600", "the mode the token is created with is not stated"),
        ("no token to read",
         "the consequence is not stated: a machine where the aggregator has never run "
         "has no token, so this route has no credential to use there"),
    ):
        if needle not in sec:
            defects.append(needle + ": " + why)
    return defects


def test_section11_records_that_the_token_file_is_created_exclusive_at_0600():
    """#3134 clause 5."""
    assert _token_note_defects(PAGE) == []


def test_a_copy_that_deletes_the_token_provenance_is_reported():
    for needle in ("os.O_CREAT | os.O_EXCL", "0o600", "no token to read"):
        defects = _token_note_defects(_perturb(PAGE, needle, "the file"))
        assert any(needle in d for d in defects), (needle, defects)


# ── the citations the new prose makes against tracked code ─────────────────

def test_every_code_citation_in_the_new_prose_still_resolves():
    """§11 now points at tracked code as the reason for each claim, and a doc-claim test
    that only reads the page cannot catch a citation being edited out from under it, so
    each one is re-read from the file it names.

    Line *numbers* are deliberately not asserted — #3134 measured them and they will
    move — but the text at each one may not change without this node saying that the
    page's reason needs re-checking.
    """
    gitignore = (ROOT / "agent-services" / ".gitignore").read_text(encoding="utf-8")
    tb = (ROOT / "agent_mcp" / "thunderbird.py").read_text(encoding="utf-8")
    auth = (ROOT / "agent_mcp" / "aggregator_auth.py").read_text(encoding="utf-8")
    paths = (ROOT / "app" / "paths.py").read_text(encoding="utf-8")
    assert "services/thunderbird-mcp/" in gitignore.splitlines(), (
        "the bridge is tracked now, which retires the worktree-zero reason §11 gives")
    assert 'BRIDGE_PATH = _LH / "agent-services" / "services" / "thunderbird-mcp"' in tb
    assert "raise FileNotFoundError" in tb, (
        "_get_pool no longer raises when the bridge is absent, so the worktree zero §11 "
        "describes has changed shape")
    assert "exporting no tools" in tb
    assert "_cached_tools, _cached_at = [], " in tb, (
        "_discover no longer caches an empty list on failure, so the zero a worktree "
        "sees is no longer the one §11 describes")
    assert "LLOYD_HOME = Path(__file__).resolve().parent.parent" in paths, (
        "LLOYD_HOME is no longer anchored on the checkout, so BRIDGE_PATH may not "
        "resolve inside a worktree after all")
    assert 'OPEN_PATHS = frozenset({"/health"})' in auth, (
        "#1053's open-path set changed; §11's claim that /health is the only "
        "credential-free route needs re-reading")
    assert "os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600" in auth, (
        "the token is no longer written create-exclusive at 0600, so clause 5's "
        "sentence needs re-writing")
    assert 'AUTH_HEADER = "X-Lloyd-Aggregator-Token"' in auth
    assert f'"{REFUSAL_CODE}"' in auth, (
        "the refusal code §11 quotes is not the one the aggregator sends")
    assert "email_send" in _BRIDGE_NAMES, (
        "email_send is no longer a bridge-served name, so the module-column rule and "
        "this file's stub fixture have both drifted from the page")
