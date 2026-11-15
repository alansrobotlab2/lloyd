"""The engine-build drift check (#3648), decided offline over build strings.

Everything here is offline on purpose, and the reason is the check's own subject: a test
that asserted the build serving this box right now would go red the first time the box is
rebuilt — which is the moment the check is meant to be RIGHT about something else. So
every case below is a synthetic `(recorded, live)` pair, and the live engine is contacted
by exactly one node, over a port nothing listens on, because a refused socket is
deterministic and carries no build string at all.

What each clause needs proved, and why a weaker test would pass without it:

  1  `verdict` decides from its two arguments and nothing else. A function that reached
     for the clock, the environment, or a socket while answering could still satisfy a
     table of pairs — and would be undecidable offline, which is clause 3's premise. So
     purity is checked as absence in the source AND as a decision taken with the network
     genuinely closed (`socket.socket` replaced by one that raises). The "no `/v1`
     request" half is checked by recording the URLs the module actually opens: a check
     that answered the property by SENDING A COMPLETION REQUEST would be the very cost
     (#GPU, and a perturbed prefix cache) this instrument exists to avoid.
  2  The expectation is parsed out of `finalizer.py`'s docstring. A test that compared
     `recorded()` to the build string written out in this file would add a fourth copy of
     it — which is the defect class this item exists to close — so the record is graded
     against a REWRITTEN COPY of the docstring: edit the record, the expectation must
     move. And the script's own source is asserted free of any build-shaped literal.
  3  Equal → 0 and the build printed; different → non-zero, both builds printed, and the
     probe named. Table-driven over pairs that are not this box's build.
  4  A socket that refuses, a body that is not JSON, and a JSON object with no `version`
     all exit non-zero as UNKNOWN, and UNKNOWN is asserted never to read as PASS.

Run: /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python -m pytest tests/test_engine_build_drift.py -q
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import socket
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/maintenance/engine_build_drift.py"

#: A build string that is not this box's, used wherever a test needs A build rather
#: than this one. Two reasons it is the wrong-looking choice on purpose: it is from a
#: real earlier release, and it is the string `finalizer.py`'s docstring itself records
#: as the one that went stale silently — so a test that hardcoded it would be pinning the
#: very accident the check exists to catch, not this box's live build.
OTHER_BUILD = "0.28.1"
RECORDED_DATE = "2026-10-14"


def _load():
    spec = importlib.util.spec_from_file_location("engine_build_drift", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


EBD = _load()


# ── clause 1: a verdict over two strings, and nothing else ─────────────────────────


def test_verdict_takes_only_the_recorded_pair_and_the_live_build():
    """Two parameters, no defaults reaching past them: the whole interface is (recorded,
    live). A third parameter with a default would let the function keep a value of its
    own — a built-in expected build — and the table below would grade a constant.
    """
    sig = inspect.signature(EBD.verdict)
    params = list(sig.parameters.values())
    assert len(params) == 2, (
        f"verdict takes {len(params)} parameters {list(sig.parameters)}; clause 1 is "
        "that its ONLY inputs are the recorded build and the live build")
    assert all(p.default is inspect.Parameter.empty for p in params), (
        "a default here is a hidden input, and a hidden expected build is the bug this "
        "whole script is a answer to")
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params)


def test_verdict_is_a_pure_decision_over_the_pair():
    """The function body touches no I/O, and decides with the network actually closed.

    The source check is the cheap half and it is not enough on its own: `eval`,
    `getattr(socket, ...)` and a helper call all hide a read from a name scan. So the
    three verdicts are also taken with `socket.socket` replaced by one that raises —
    a verdict that reached the network to answer would raise through this call rather
    than return.
    """
    src = inspect.getsource(EBD.verdict)
    for forbidden in ("urlopen", "urllib", "open(", "Path(", "time.", "os.environ",
                      "requests", "FINALIZER.read"):
        assert forbidden not in src, (
            f"verdict references {forbidden!r}, so it is not a function of its arguments "
            "and clause 3's offline table would be grading an live value")

    class _NoSockets:
        def __init__(self, *a, **k):
            raise AssertionError("verdict opened a socket")

    real = socket.socket
    socket.socket = _NoSockets
    try:
        ok = EBD.verdict((RECORDED_DATE, OTHER_BUILD), OTHER_BUILD)
        drift = EBD.verdict((RECORDED_DATE, OTHER_BUILD), "9.9.9+gdeadbeef")
        unknown = EBD.verdict((RECORDED_DATE, OTHER_BUILD), None)
    finally:
        socket.socket = real
    assert ok[0] == EBD.PASS and drift[0] == EBD.DRIFT and unknown[0] == EBD.UNKNOWN


def test_the_verdict_path_opens_no_v1_endpoint():
    """What the module actually requests, recorded rather than inferred from its text.

    `eval/tool_choice_none_probe.py` decides on `usage.prompt_tokens` from real
    completions. A drift check that reached for the same lever would cost GPU time on
    every scheduled run and — worse for the property itself — would populate the prefix
    cache it is guarding, so the numbers the probe later reads would be numbers this
    check influenced. One request, to `/version`, is the whole budget.
    """
    opened: list[str] = []

    class _Resp:
        def __enter__(self):
            return self

        def read(self):
            return json.dumps({"version": OTHER_BUILD}).encode()

        def __exit__(self, *exc):
            return False

    def fake(req, *a, **k):
        opened.append(req.full_url if hasattr(req, "full_url") else str(req))
        return _Resp()

    real = EBD.urllib.request.urlopen
    EBD.urllib.request.urlopen = fake
    try:
        assert EBD.live_build("http://127.0.0.1:9") == OTHER_BUILD
        code, _ = EBD.verdict((RECORDED_DATE, OTHER_BUILD), OTHER_BUILD)
    finally:
        EBD.urllib.request.urlopen = real
    assert code == EBD.PASS
    assert opened == ["http://127.0.0.1:9/version"], (
        f"the live-build read opened {opened}; exactly one GET /version is the budget")
    assert not [u for u in opened if "/v1" in u], (
        "a completion request would cost GPU and perturb the prefix cache the probe "
        "measures")
    assert "/v1" not in SCRIPT.read_text(encoding="utf-8"), (
        "the script names a /v1 path anywhere, and clause 1 says its verdict path "
        "issues no completion request")


# ── clause 2: the expectation is PARSED, so the two can never disagree ─────────────


def test_recorded_build_is_read_from_the_finalizer_docstring():
    """Parse, not copy: the script must agree with the record whatever the record says.

    Graded against a rewritten copy of the docstring, because comparing `recorded()` to a
    build string typed into this file would be a fourth copy of it — this item counts
    three existing holders and forbids a fourth. The edit below is the whole clause: move
    the record, the expectation has to move with it.
    """
    tmp = tmp_path_of("finalizer-rewrite.py")
    tmp.write_text(
        '"""P11 finalizer.\n\n'
        "Re-verified 2027-01-02 on 9.9.9rc1.dev7+g0123456789\n"
        '"""\n', encoding="utf-8")
    assert EBD.recorded(tmp) == ("2027-01-02", "9.9.9rc1.dev7+g0123456789"), (
        "editing the docstring did not move the expectation")

    tmp2 = tmp_path_of("finalizer-line-moved.py")
    tmp2.write_text('"""Header.\n\nSome prose.\n\n'
                    "Re-verified 2026-10-14 on a.b.c+gdeadbeef\n"
                    "More prose.\n"
                    '"""\n', encoding="utf-8")
    assert EBD.recorded(tmp2) == ("2026-10-14", "a.b.c+gdeadbeef")

    # The line has to be the DOCSTRING's. A comment or a string literal in the module
    # that happens to read like the record must not become the expectation, or `# `
    # Re-verified 2020-01-01 on fake would silently win over the real record.
    commented = tmp_path_of("finalizer-comment.py")
    commented.write_text('"""No record here.\n"""\n\n'
                         "# Re-verified 2020-01-01 on fake+gdeadbeef\n", encoding="utf-8")
    assert EBD.recorded(commented) is None, (
        "a comment masquerading as the record was parsed as the record")

    missing = tmp_path_of("finalizer-no-record.py")
    missing.write_text('"""A module with no build line.\n"""\n', encoding="utf-8")
    assert EBD.recorded(missing) is None
    gone = tmp_path_of("does-not-exist.py")
    assert EBD.recorded(gone) is None, "an unreadable record must be None, not a crash"


def tmp_path_of(name: str) -> Path:
    """A path under this session's tmp dir. pytest's `tmp_path` is a fixture, and these
    helpers are called from inside test bodies that already take it; a module-level
    directory keeps the tests above readable. Removed with the session's tmp root."""
    d = Path("/tmp/engine-build-drift-tests")
    d.mkdir(exist_ok=True)
    return d / name


def test_the_record_for_the_real_finalizer_exists_without_naming_its_value():
    """The real `finalizer.py` must carry a parseable record — asserted WITHOUT naming
    which build that is, so this node survives a rebuild and still catches the failure it
    is about: a re-measurement that writes the date and forgets the build.

    Also asserts the parsed pair is exactly what the file's own text says, read here
    independently. That is the cross-check that keeps a lenient regex — one that matched
    a date but captured a trailing word, say — from passing quietly.
    """
    import re
    text = (ROOT / "app/harness/finalizer.py").read_text(encoding="utf-8")
    got = EBD.recorded()
    assert got is not None, (
        "app/harness/finalizer.py's module docstring no longer carries a "
        "`Re-verified <date> on <build>` line, so there is no record to drift against")
    date, build = got
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date), date
    assert build and not build.isspace(), repr(build)
    assert f"Re-verified {date} on {build}" in " ".join(text.split()), (
        f"parsed ({date!r}, {build!r}) out of a docstring that does not say that")


def test_the_script_carries_no_build_string_of_its_own():
    """Two copies of a build string is the defect, told from the inside.

    `finalizer.py`'s docstring said "the installed vLLM 0.28.1" and became false the day
    the box moved, with no test to notice (`app/harness/tests/test_finalizer.py:879`). A
    script that held its own expected build could not notice twice over, and the drift it
    reported would be between two files rather than between a record and an engine.
    """
    src = SCRIPT.read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.strip().startswith("#"))
    assert not re_search_build(code), (
        "the script embeds what looks like a vLLM build string; the expected build must "
        "come from finalizer.py's docstring and nowhere else")
    # The install-side pin is the one string a rebuild DOES change, and reading it would
    # make the check tautological: pin and docstring drift together, so the comparison
    # would pass on exactly the day it matters. (The docstring may NAME the pin to
    # explain why it is not read — that prose is the point of this assertion.)
    code_no_doc = code.replace(EBD.__doc__ or "", "")
    assert "agent-services" not in code_no_doc and "setup-vllm" not in code_no_doc \
        and "versions.txt" not in code_no_doc, (
        "the script reaches for the install-side pin somewhere in its code, not just its "
        "prose")


def re_search_build(code: str):
    import re
    return re.search(r"\b\d+\.\d+\.\d+(rc\d+)?(\.dev\d+)?\+g[0-9a-f]{7,}\b", code)


# ── clause 3: PASS and DRIFT, over pairs this box is not part of ───────────────────


PASS_TABLE = [
    # (recorded build, live build, recorded date) — none of these is this box's build,
    # so none of these assertions can rot when the engine is rebuilt.
    ("0.28.1", "0.28.1", "2026-09-08"),
    ("1.2.3+g000000000", "1.2.3+g000000000", "2027-03-04"),
    ("0.31.1rc1.dev260+ga98247ab4", "0.31.1rc1.dev260+ga98247ab4", "2026-10-14"),
]

DRIFT_TABLE = [
    # A rebuild that changed ONLY the commit hash must be DRIFT: the local segment IS
    # the build, so a comparison that normalised versions and compared releases would
    # call this a match and the probe would never be re-run.
    ("0.31.1rc1.dev260+ga98247ab4", "0.31.1rc1.dev260+g111111111", "2026-10-14"),
    # A dev-count bump, same release: still a different build, still re-measure.
    ("0.31.1rc1.dev260+ga98247ab4", "0.31.1rc1.dev261+ga98247ab4", "2026-10-14"),
    ("0.28.1", "0.31.1rc1.dev260+ga98247ab4", "2026-09-08"),
    # Case IS part of a git sha's identity, and of a version's local segment.
    ("1.2.3+gAaBbCcDd", "1.2.3+gaabbccdd", "2027-03-04"),
]


@pytest.mark.parametrize("want,live,date", PASS_TABLE)
def test_equal_builds_pass_and_print_the_build(want, live, date, capsys):
    code, line = EBD.verdict((date, want), live)
    assert code == 0, f"{want} vs {live} must exit 0, got {code}: {line}"
    assert want in line, f"the PASS line does not print the build: {line!r}"
    assert "PASS" in line and "DRIFT" not in line and "UNKNOWN" not in line


@pytest.mark.parametrize("want,live,date", DRIFT_TABLE)
def test_differing_builds_exit_nonzero_naming_both_and_the_probe(want, live, date,
                                                                capsys):
    code, line = EBD.verdict((date, want), live)
    assert code != 0, f"a mismatch must not exit 0: {want} vs {live} → {line}"
    assert code == EBD.DRIFT
    assert want in line and live.strip() in line, (
        f"both build strings have to be in the line or a human reading the job log "
        f"cannot tell what changed: {line!r}")
    assert EBD.PROBE in line, (
        f"a mismatch is not a repair; the line has to send the reader to re-run the "
        f"probe: {line!r}")
    assert "eval/tool_choice_none_probe.py" in line


@pytest.mark.parametrize("want,live", [
    ("0.28.1", "0.28.1"),
    ("0.31.1rc1.dev260+ga98247ab4", "0.31.1rc1.dev260+g111111111"),
])
def test_main_exits_with_the_verdict_code_and_prints_its_line(want, live, capsys,
                                                              monkeypatch):
    """The exit code a scheduled job sees, wired to the decision rather than to a
    second copy of it. `main` is driven with its two inputs stubbed, so this node grades
    the plumbing (`raise SystemExit(main())` forwarding the code, the print reaching
    stdout) without depending on which engine answers :8096 today.
    """
    monkeypatch.setattr(EBD, "recorded", lambda *a, **k: (RECORDED_DATE, want))
    monkeypatch.setattr(EBD, "live_build", lambda *a, **k: live)
    code = EBD.main([])
    out = capsys.readouterr().out
    assert code == (0 if want == live.strip() else EBD.DRIFT)
    assert out.strip(), "main printed nothing, so a job log would show a code and no why"
    assert (want in out) and (live.strip() in out)
    if want != live.strip():
        assert "eval/tool_choice_none_probe.py" in out


def test_the_script_exits_nonzero_through_the_process_boundary(tmp_path):
    """The interpreter-level exit code, not just the function's return value.

    A `main()` that returns 1 while `__main__` swallows it is invisible to a job runner:
    the queue records success, Mission Control shows green, and the drift is reported by
    nobody. Run as a subprocess against a rewritten record and a port nothing listens on,
    so the expected outcome is derived here rather than copied from the box's state.
    """
    fake = tmp_path / "finalizer.py"
    fake.write_text('"""x.\n\nRe-verified 2026-01-01 on 1.2.3+g000000000\n"""\n',
                    encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--finalizer", str(fake),
         "--base-url", "http://127.0.0.1:9"],
        capture_output=True, text=True, timeout=30)
    assert r.returncode == EBD.UNKNOWN, (
        f"unreachable engine must exit {EBD.UNKNOWN}, got {r.returncode}: {r.stdout!r}")
    assert "UNKNOWN" in r.stdout and "PASS" not in r.stdout
    assert "1.2.3+g000000000" in r.stdout, (
        "the recorded build should still be named, so the log says what could not be "
        f"confirmed: {r.stdout!r}")


# ── clause 4: an engine that will not say is UNKNOWN, and never PASS ───────────────


def test_live_build_returns_none_for_every_way_the_answer_can_fail(monkeypatch):
    cases = [
        ("refused socket", _raise(urllib.error.URLError("connection refused"))),
        ("timed out", _raise(socket.timeout("timed out"))),
        ("not JSON", _reply(b"<html>502 Bad Gateway</html>")),
        ("no version key", _reply(json.dumps({"ok": True}).encode())),
        ("version null", _reply(json.dumps({"version": None}).encode())),
        ("version blank", _reply(json.dumps({"version": "   "}).encode())),
        ("version not a string", _reply(json.dumps({"version": 31}).encode())),
        ("body is a list", _reply(b"[0, 1]")),
    ]
    for name, behaviour in cases:
        monkeypatch.setattr(EBD.urllib.request, "urlopen", behaviour)
        assert EBD.live_build("http://127.0.0.1:8096") is None, (
            f"{name}: an unreadable build must come back None (→ UNKNOWN), not a "
            "value that could be diffed against the record")


def test_padding_round_a_stamp_is_the_same_build(monkeypatch):
    """Whitespace is not a rebuild, and the READER is where that is decided.

    `verdict` cannot express it: by the time a string reaches it, it is whatever the
    reader chose to do with padding, so a table over pairs grades a decision that has
    already been made. Asserted here instead — a padded reply comes back stripped, and
    matches the record — because a server that padded a body would otherwise send someone
    to re-run a 25-second probe and rewrite the record for an engine that did not change.
    """
    monkeypatch.setattr(EBD.urllib.request, "urlopen",
                        _reply(json.dumps({"version": f"  {OTHER_BUILD}  "}).encode()))
    live = EBD.live_build("http://127.0.0.1:8096")
    assert live == OTHER_BUILD, repr(live)
    assert EBD.verdict((RECORDED_DATE, OTHER_BUILD), live)[0] == EBD.PASS


class _Raises:
    def __init__(self, exc):
        self.exc = exc

    def __call__(self, *a, **k):
        raise self.exc


def _raise(exc):
    return _Raises(exc)


class _Reply:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def read(self):
        return self.payload

    def __exit__(self, *exc):
        return False


def _reply(payload):
    def f(*a, **k):
        return _Reply(payload)
    return f


@pytest.mark.parametrize("live", [None, ""])
def test_an_unreadable_live_build_is_unknown_and_never_a_pass(live):
    code, line = EBD.verdict((RECORDED_DATE, OTHER_BUILD), live)
    assert code != 0, f"an engine that named no build exited 0 ({live!r})"
    assert code == EBD.UNKNOWN
    assert "UNKNOWN" in line
    assert "PASS" not in line, (
        f"the UNKNOWN line reads as a pass, which is the one thing that makes this "
        f"check safe to schedule: {line!r}")
    assert OTHER_BUILD in line, (
        "the recorded build belongs in the line: the reader needs to know what could "
        f"not be confirmed: {line!r}")


def test_a_record_that_names_no_build_is_unknown_not_a_pass():
    """The record side of UNKNOWN. A re-measurement that writes `Re-verified 2026-11-15`
    and drops the build leaves nothing to compare, and `None != anything` must not be
    allowed to read as either PASS or DRIFT — a drift report naming `None` as the
    recorded build would send a human to re-measure a property that is fine.
    """
    code, line = EBD.verdict(None, OTHER_BUILD)
    assert code == EBD.UNKNOWN and code != 0
    assert "UNKNOWN" in line and "PASS" not in line and "DRIFT" not in line
    assert "finalizer.py" in line, (
        "the fix lives in the record, so the line has to name where: " + line)


def test_no_engine_listening_is_unknown_end_to_end():
    """One contact with a real socket, on a port nothing can answer.

    This is the only node that touches the network, and it is written as a FAILURE of
    the network so it cannot depend on the box: port 9 (discard) refuses, and whatever
    build is installed cannot change that. The PASS path against the live primary is
    owed after landing precisely because no node here can or should pin it.
    """
    code, line = EBD.verdict(EBD.recorded(), EBD.live_build("http://127.0.0.1:9",
                                                            timeout=1.0))
    assert code == EBD.UNKNOWN, f"expected UNKNOWN against a dead port, got {line}"
    assert "Not a pass" in line or "neither confirmed nor contradicted" in line, (
        "an unreachable engine must say what it did NOT establish: " + line)


# ── #4137 clause 4: the three exit codes across the real process boundary ───────
#
# The nodes above decide `verdict()` and `main()` in-process with a fixture pair. These
# run the SCRIPT as a subprocess against an HTTP server that really answers `/version`,
# with the expected build read out of `finalizer.py`'s docstring rather than spelled —
# so what is proved is the whole contract the caller relies on: 0 when the engine answers
# the build the record names, 1 when it answers another, 2 when nothing answers.

def _serve_version(payload: str):
    """A one-request-per-connection `/version` server; yields its base URL."""
    import http.server
    import threading
    from contextlib import contextmanager

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):                       # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload.encode())

        def log_message(self, *args):
            pass

    @contextmanager
    def serve():
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{srv.server_address[1]}"
        finally:
            srv.shutdown()
            srv.server_close()
            thread.join(timeout=2)

    return serve()


def _run_script(*argv) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *argv],
                          capture_output=True, text=True, timeout=60)


def test_the_process_exits_zero_against_an_engine_answering_the_recorded_build():
    """Clause 4: PASS is 0, end to end, on the build THIS tree records.

    The expected build comes from `EBD.recorded()` — this file's rule again — so the node
    cannot go green by naming a build the record does not, and cannot go red because the
    record was re-verified: it asks the same question the caller asks.
    """
    want = EBD.recorded()
    assert want is not None, "finalizer.py's record is unreadable, so nothing here proves"
    with _serve_version(json.dumps({"version": want[1]})) as base_url:
        proc = _run_script("--base-url", base_url)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.startswith("engine-build-drift PASS"), proc.stdout


def test_the_process_exits_one_against_an_engine_answering_another_build():
    """Clause 4: DRIFT is 1 through the process, naming both builds."""
    want = EBD.recorded()
    other = "0.99.0rc1.dev1+gdeadbeef"
    assert other != want[1]
    with _serve_version(json.dumps({"version": other})) as base_url:
        proc = _run_script("--base-url", base_url)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert want[1] in proc.stdout and other in proc.stdout, proc.stdout


def test_the_process_exits_two_when_no_engine_answers():
    """Clause 4: an unreadable `/version` is 2 (UNKNOWN), never a silent 0."""
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    proc = _run_script("--base-url", f"http://127.0.0.1:{port}")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert proc.stdout.startswith("engine-build-drift UNKNOWN"), proc.stdout


# ── #4137 clause 5: the detector has a caller, and the doc names it ────────────
#
# #3648 built this script and closed; `git grep -l engine_build_drift` then named the
# script, its own test and two architecture docs, and no runner — so a true DRIFT printed
# `RC=1` for 32 days with nobody hearing it, while `architecture/harness.md` P6(b) called
# the script "the detector that turns that re-run into something that happens". These
# nodes are the wiring's pins: the caller's argv names THIS script, its runner seam is
# the subprocess boundary (not an in-process re-decision), the watchdog arm alerts on
# DRIFT and not on UNKNOWN, and the doc names the arm instead of the promise.

def _maintenance():
    from workers import maintenance
    return maintenance


def _watchdog():
    import workers.fleet_watchdog as fw
    return fw


def test_the_caller_builds_a_command_that_runs_this_script():
    """The caller's argv is this script, so the wiring cannot silently re-point."""
    maintenance = _maintenance()
    argv = maintenance.engine_build_argv()
    assert argv[1] == str(SCRIPT), argv
    assert "--base-url" not in argv, argv          # the default engine, by default
    assert maintenance.engine_build_argv(base_url="http://h:1")[-2:] == [
        "--base-url", "http://h:1"]


def test_the_caller_reads_the_verdict_word_and_decides_nothing_of_its_own():
    """Clause 5 across the subprocess seam: the caller parses, the script decides.

    The runner seam stands in for `subprocess.run`, so a fake line here grades only this
    function's reading of a line the script really printed — and the three verdicts are
    taken verbatim from the script's own prefixes. A run that cannot happen is NOT folded
    into UNKNOWN: "the detector did not run" and "the engine will not name its build" are
    different things to wake for, and only the second is the script's verdict.
    """
    maintenance = _maintenance()

    class Proc:
        def __init__(self, out, rc):
            self.stdout, self.stderr, self.returncode = out, "", rc

    for word, rc in (("PASS", 0), ("DRIFT", 1), ("UNKNOWN", 2)):
        line = f"engine-build-drift {word}: the body a person has to read"
        out = maintenance.engine_build_verdict(["x"],
                                               runner=lambda *a, _l=line, _r=rc, **k: Proc(_l, _r))
        assert out["verdict"] == word, out
        assert out["line"] == line and out["rc"] == rc, out

    def boom(*a, **k):
        raise OSError("no interpreter")

    out = maintenance.engine_build_verdict(["x"], runner=boom)
    assert out["verdict"] is None and "no interpreter" in out["line"], out


async def test_the_watchdog_alerts_a_person_on_a_drift_and_never_on_an_unknown(tmp_path, monkeypatch):
    """Clause 5: DRIFT reaches `_alert`; UNKNOWN is recorded and quiet.

    `_alert` is the watchdog's one route to a human (`app.discord_notify.discord_alert`,
    which on this box falls through to today's daily note), so the alarm is asserted there
    and carries BOTH builds — the recorded one and the live one — because a drift alarm
    naming only one of them cannot be acted on. UNKNOWN stays silent on purpose: this
    watchdog runs while the engine is restarting for entirely ordinary reasons, and an
    alarm that fires at every restart is an alarm that gets ignored at the one it is for.
    The cooldown is graded from BOTH sides, and each step is taken PAST
    `_ENGINE_BUILD_SCAN_SECONDS` — which is what the 2026-11-15 review found missing. An
    earlier draft advanced one hour between calls, and at one hour the 6 h scan cadence
    returns None before the alert branch is ever reached, so "no second alert" held with
    `_ENGINE_BUILD_ALERT_INTERVAL_SECONDS` deleted outright: a leg that cannot fail grades
    no constant. So the cadence stamp is stepped over at +7 h (the script really runs —
    asserted by a non-None reading, not by silence) and the interval is read at +7 h,
    inside the day, where deleting the cooldown posts a second alarm; then again at +31 h,
    outside it, where a cooldown stretched past a day stays silent forever. The condition
    itself persists until someone re-measures, which is why a day is the right interval
    rather than never or six hours.
    """
    import datetime as dt
    import json

    fw = _watchdog()
    from workers.queue import WorkQueue

    # `_state` is the watchdog's module-global pacing dict and `monkeypatch`
    # restores it: `clear()` would take the cooldowns every other watchdog test in
    # the session shares, which reads as their failure and is this file's.
    monkeypatch.setattr(fw, "_state", {})
    queue = WorkQueue(str(tmp_path / "drift.db"))
    sent: list[str] = []

    async def fake_alert(message):
        sent.append(message)

    # Patched through monkeypatch, not assignment: `_alert` is the watchdog's one
    # route to a person, and a bare assignment leaves a fake installed for every
    # later test in the session that asserts an alarm reaches the daily note.
    monkeypatch.setattr(fw, "_alert", fake_alert)
    want = EBD.recorded()
    assert want is not None
    drift = (f"engine-build-drift DRIFT: recorded build {want[1]} (measured {want[0]}), "
             f"live build 0.33.9rc1.dev9+gcafebabe.")

    def runner(line, rc):
        def run(argv, **kw):
            class Proc:
                stdout, stderr, returncode = line, "", rc
            return Proc()
        return run

    now = dt.datetime.now(dt.timezone.utc)
    reading = await fw._check_engine_build(queue, now=now, runner=runner(drift, 1))
    assert reading["verdict"] == "DRIFT", reading
    assert len(sent) == 1, sent
    assert want[1] in sent[0] and "0.33.9rc1.dev9+gcafebabe" in sent[0], sent[0]
    assert "eval/tool_choice_none_probe.py" in sent[0], sent[0]
    assert json.loads(queue.wm_get(fw._ENGINE_BUILD_WM_SOURCE,
                                   fw._ENGINE_BUILD_STATE_KEY))["verdict"] == "DRIFT"

    # +7 h: past the 6 h scan cadence, so the detector RUNS again — the non-None reading
    # is what makes the next line mean "the cooldown held" rather than "the cadence
    # skipped". A still-true DRIFT inside the day posts no second alarm.
    again = await fw._check_engine_build(queue, now=now + dt.timedelta(hours=7),
                                         runner=runner(drift, 1))
    assert again is not None and again["verdict"] == "DRIFT", (
        f"the arm skipped the check at +7 h on a 6 h cadence, so the line below could "
        f"not have failed however the cooldown were set: {again}")
    assert len(sent) == 1, (
        f"a second alarm posted at +7 h, inside the "
        f"{fw._ENGINE_BUILD_ALERT_INTERVAL_SECONDS}s interval: {sent}")

    # +31 h: past the interval, so the same unbroken drift posts again. This is the leg
    # that fails if the interval is stretched to a year or the cooldown becomes a mute.
    sent.clear()
    later = await fw._check_engine_build(queue, now=now + dt.timedelta(hours=31),
                                         runner=runner(drift, 1))
    assert later is not None and later["verdict"] == "DRIFT", later
    assert len(sent) == 1, (
        f"the drift was still true 31 h on and no second alarm went out, so the "
        f"cooldown reads as a mute: {sent}")

    sent.clear()
    unknown = "engine-build-drift UNKNOWN: no engine answering on the base url."
    out = await fw._check_engine_build(
        queue, now=now + dt.timedelta(days=2, hours=1), runner=runner(unknown, 2))
    assert out["verdict"] == "UNKNOWN", out
    assert sent == [], sent


async def test_the_arm_awaits_its_cadence_and_the_loop_calls_it(tmp_path, monkeypatch):
    """Clause 5, the two seams the drift actually has to cross.

    1. Inside `tick`: the arm is awaited from the loop that runs whether or not a task is
       due, so "nobody typed the command" can never again be why a DRIFT went unheard.
    2. Its own cadence: two calls inside the interval run the script ONCE, and a cadence
       stamp is not an alert cooldown — the two keys are separate because an interval and
       a cooldown answer different questions.
    """
    import inspect

    fw = _watchdog()
    src = inspect.getsource(fw.tick)
    assert "_check_engine_build(" in src, \
        "fleet_watchdog.tick no longer calls the engine-build arm"

    from workers.queue import WorkQueue
    import datetime as dt

    # `_state` is the watchdog's module-global pacing dict and `monkeypatch`
    # restores it: `clear()` would take the cooldowns every other watchdog test in
    # the session shares, which reads as their failure and is this file's.
    monkeypatch.setattr(fw, "_state", {})
    queue = WorkQueue(str(tmp_path / "cadence.db"))
    calls: list[list] = []

    def runner(argv, **kw):
        calls.append(argv)
        class Proc:
            stdout, stderr, returncode = "engine-build-drift PASS: x", "", 0
        return Proc()

    now = dt.datetime.now(dt.timezone.utc)
    assert await fw._check_engine_build(queue, now=now, runner=runner) is not None
    assert await fw._check_engine_build(queue,
                                        now=now + dt.timedelta(minutes=5),
                                        runner=runner) is None
    assert len(calls) == 1, calls
    assert calls[0][1].endswith("scripts/maintenance/engine_build_drift.py"), calls


def test_the_arm_runs_behind_both_stall_alarms_in_the_tick():
    """The placement the arm has to keep, pinned where nothing else can see it.

    The arm is the only detector in `tick` that spawns a process and waits on a socket,
    so its position decides what a slow engine delays. It shipped at the top of the tick
    and the first thing that showed it was `tests/test_fleet_watchdog.py`'s scheduler-loop
    test going intermittently red: on a fresh queue the first tick really runs the script,
    and the next_run stall alarm behind it had not run by the time the test stopped
    polling. Every test in this file calls `_check_engine_build` directly, so none of them
    can see an order — only a reading of the tick can, which is what this node is.

    Read against the executed statements, not the prose: `#4137`, the arm's name and the
    stall detectors' names all appear in `tick`'s comments, and a comment-reading version
    of this node would pass on a tick that never called the arm. So comments are dropped
    first, and each name must then occur EXACTLY once — the detectors are reached through
    `run_in_executor`, so it is the single reference that is being ordered, and one is the
    only count that has an order to grade.
    """
    import inspect
    import re

    fw = _watchdog()
    src = "\n".join(ln.split("#")[0] for ln in inspect.getsource(fw.tick).splitlines())

    def call_at(name: str) -> int:
        hits = [m.start() for m in re.finditer(rf"\b{name}\b", src)]
        assert len(hits) == 1, (
            f"`tick` refers to `{name}` {len(hits)} times outside comments, so this "
            f"ordering node has no single position to grade: {hits}")
        return hits[0]

    arm = call_at("_check_engine_build")
    for earlier in ("_grossly_overdue", "_next_run_stalled"):
        assert arm > call_at(earlier), (
            f"the engine-build arm runs before `{earlier}` in the tick, so an engine "
            "that stalls its 5 s `/version` read is once again on the stall alarm's "
            "critical path")
    assert arm < call_at("_census_failure_throughput"), (
        "the arm moved past the census, and the census's comment claims it is last in "
        "the tick — one of the two is false")


async def test_the_real_scheduler_loop_runs_the_arm_without_anyone_typing_it(tmp_path,
                                                                              monkeypatch):
    """Clause 5's outermost seam, EXECUTED rather than read.

    The 2026-11-15 review marked this seam unverified, and it is the one the item is
    about: "nobody runs it" was the whole defect, so the proof has to be the pool's own
    scheduler loop starting, ticking, and reaching the arm with no human and no test
    calling `_check_engine_build`. Only the subprocess is stood in for — a unit test that
    POSTed to the live engine's `/version` would be measuring what the engine happens to
    serve today, and the script's own process boundary is crossed by
    `test_the_process_exits_zero_against_an_engine_answering_the_recorded_build` instead.

    Dispatch is switched off the way `tests/test_fleet_watchdog.py` switches it off, the
    autonomy board is empty so no stall alarm has anything to name, and every other seat
    of the loop that would touch the machine is stubbed: what is left running is the
    scheduler loop, `tick`, the arm, and the queue.
    """
    import asyncio

    from app import autonomy
    from workers.pool import WorkerPool
    from workers.queue import WorkQueue
    import workers.sources as sources
    import workers.sources.scheduled_task as st

    fw = _watchdog()
    monkeypatch.setattr(fw, "_state", {})
    board = tmp_path / "autonomy"
    board.mkdir()
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", board)
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(sources, "SOURCE_REGISTRY", {"scheduled-task": st}, raising=False)
    monkeypatch.setattr(sources, "get_sources_config", lambda: {
        "scheduled-task": {"enabled": False, "interval_seconds": 0,
                           "max_duration_seconds": 1800}}, raising=False)

    sent: list[str] = []

    async def _capture(msg):
        sent.append(msg)

    monkeypatch.setattr(fw, "_alert", _capture)

    run_argv: list[list] = []

    class Proc:
        stdout = ("engine-build-drift DRIFT: recorded build 0.33.0rc1.dev1148+g8f2ba935a "
                  "(measured 2026-11-15), live build 0.99.9rc1.dev9+gcafebabe.")
        stderr = ""
        returncode = 1

    def fake_runner(argv, **_kw):
        run_argv.append(list(argv))
        return Proc()

    real = fw._check_engine_build

    async def spy(queue, **kw):
        # Injects the stub runner into the REAL arm: everything between the loop and the
        # subprocess — the cadence, the verdict read, the watermark writes, the alert —
        # is the shipped code, not a stand-in for it.
        return await real(queue, runner=fake_runner, **kw)

    monkeypatch.setattr(fw, "_check_engine_build", spy)

    async def _noop(*_a, **_k):
        return None

    queue = WorkQueue(str(tmp_path / "loop.db"))
    pool = WorkerPool(queue, slots=0, poll_idle_seconds=0.01)
    monkeypatch.setattr(pool, "_maybe_sweep_poisoned", _noop)
    monkeypatch.setattr(pool, "_probe_services", _noop)
    monkeypatch.setattr(pool, "_watch_dispatch", _noop)
    await pool.start()
    try:
        for _ in range(300):
            if sent:
                break
            await asyncio.sleep(0.02)
    finally:
        await pool.stop()

    assert run_argv, (
        "the scheduler loop never reached the engine-build arm, so the detector is "
        "unowned again — the exact state #3648 left behind")
    assert run_argv[0][1] == str(SCRIPT), run_argv[0]
    assert len(sent) == 1, f"expected the one drift alarm, got {sent}"
    assert "0.33.0rc1.dev1148+g8f2ba935a" in sent[0] and "0.99.9rc1.dev9+gcafebabe" in (
        sent[0]), "the alarm the loop produced names neither build, so nobody can act on it"


def test_the_doc_names_the_caller_and_no_longer_promises_it():
    """Clause 5's last half: the doc's promise became a name.

    P6(b) used to say this script "is the detector that turns that re-run into something
    that happens". That sentence was false for 32 days — an unrun exit code is not an
    alarm — and it is what let the class stay closed after #3648 shipped the detector, so
    it is gone rather than softened: the bullet now names the arm that runs it, and the
    ban below is what stops the promise returning in place of the wiring.
    """
    doc = (ROOT / "architecture" / "harness.md").read_text(encoding="utf-8")
    fw = _watchdog()
    assert "_check_engine_build" in doc, \
        "harness.md names no caller, so the detector is unowned again"
    banned = "the detector that turns that re-run into something that happens"
    assert banned not in doc, \
        "the promise is back in the doc while nothing runs the script"
    # The named caller is the real symbol, not prose about one.
    assert callable(getattr(fw, "_check_engine_build", None)), \
        "the doc names a watchdog function that does not exist"
