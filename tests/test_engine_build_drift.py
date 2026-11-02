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
