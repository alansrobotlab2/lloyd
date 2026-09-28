"""#1726: the health check reads the Lloyd CA trust store, read-only.

`scripts/install-ca.sh --check` has existed since #1668 with a test and no production
caller (`git grep -n "install-ca.sh --check"` returned only its own usage comments and
this repo's test file), so the drift it detects stayed invisible: the CA was re-minted
on 2026-09-22 while the NSS nickname `Lloyd CA` kept the retired August key, and
`cert9.db`'s mtime (2026-09-08) shows nothing was installed since. Nothing breaks only
because `web/vite.config.ts` prefers the Let's Encrypt leaf while it exists; the day
that leaf stops being served, the browser arm verifies the private `lloyd.crt` against
a CA the store no longer holds.

`scripts/service_health_check.py` is now the caller, as one advisory row. These tests
force the three answers the guard can give, because the state of the store on the
machine running the suite is not one of the things under test: on this box right now
the check exits 1 with drift, and on a freshly-installed box it exits 0.

Two properties are worth more than the happy path, and both are the item's clauses:

  * `warn` is a claim about a measured mismatch. An answer that never reached the
    store — no `certutil`, an unreadable CA file, an unparseable reply — is `unknown`.
    Scoring that healthy would be the #1431 / #1141 failure: an exit code, not a
    measurement, deciding the verdict. So every scenario the second test walks asserts
    "not ok", not just "not warn".
  * the row cannot turn a healthy run into a failure. Clause 3 is graded over the
    arithmetic in `format_json`/`format_text`, not over a hand-typed expectation, so
    the third test grades every pre-existing row through the module's own `grade()`
    before and after the drift row is added.

The fake runner returns the bytes `install-ca.sh` prints, tag and padding included;
`test_the_faked_answers_are_the_answers_the_script_prints` pins that against the
script's own format strings, so a future edit to either side goes red instead of
leaving the parser silently reading nothing.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "service_health_check.py"
INSTALL_CA = ROOT / "scripts" / "install-ca.sh"

STORE = "/home/alansrobotlab/.pki/nssdb"
CA_CRT = "/home/alansrobotlab/lloyd/agent-services/cert/ca.crt"
STORED_FP = ("98:1E:A0:F3:2D:BD:CB:D6:86:DB:9F:C7:7E:90:D1:E0"
             ":1B:1D:30:93:64:3A:22:5A:DD:5C:5F:81:4A:CB:EA:7E")
EXPECTED_FP = ("C5:A0:C2:DC:85:9A:0A:EB:35:A2:F0:FA:DB:65:DA:AC"
               ":DA:24:C4:4A:0B:B0:C6:BA:AE:C6:92:C5:5F:2E:0D:53")

# The three verdicts plus the two failure lines, as the script prints them: the
# `[install-ca] ` tag on every line, the `%-10s` padding on the fingerprints, the
# `Run:` hint that follows a FAILED. Verbatim, because the parser reads these bytes.
CHECK_OK = (
    f"[install-ca] CHECK OK: 'Lloyd CA' in {STORE} is {CA_CRT}\n"
    f"[install-ca]   stored   sha256 {EXPECTED_FP}\n"
    f"[install-ca]   expected sha256 {EXPECTED_FP}\n")
CHECK_FAILED = (
    f"[install-ca] CHECK FAILED: 'Lloyd CA' in {STORE} is a different certificate "
    f"than {CA_CRT}\n"
    f"[install-ca]   stored   sha256 {STORED_FP}\n"
    f"[install-ca]   expected sha256 {EXPECTED_FP}\n"
    f"[install-ca]   Run: bash scripts/install-ca.sh {CA_CRT}\n")
# install-ca.sh:153-155 — the store WAS read and holds no such nickname. That is
# drift, not a no-data read, and its `stored` line says `none (no such nickname)`
# rather than a hash, so the row has to say which fingerprint is missing.
CHECK_FAILED_NO_NICKNAME = (
    f"[install-ca] CHECK FAILED: no 'Lloyd CA' entry in {STORE} "
    f"(store absent, empty, or unreadable)\n"
    f"[install-ca]   stored   sha256 none (no such nickname)\n"
    f"[install-ca]   expected sha256 {EXPECTED_FP}\n"
    f"[install-ca]   Run: bash scripts/install-ca.sh {CA_CRT}\n")
CHECK_INCONCLUSIVE = (
    "[install-ca] CHECK INCONCLUSIVE: no certutil, so no NSS store could be read —\n"
    "[install-ca]          reporting no verdict rather than 'up to date'.\n")
# install-ca.sh's own argument guard: nonzero, but the store was never opened.
NO_CA_FILE = (
    f"[install-ca] ERROR: no CA certificate at {CA_CRT} "
    f"(run scripts/gen-cert.sh first)\n")
GIBBERISH = "bash: /somewhere/install-ca.sh: No such file or directory\n"


def _load():
    spec = importlib.util.spec_from_file_location("shc_ca", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


shc = _load()


class _Answer:
    """A stand-in `subprocess.run` that records the argv it was handed.

    `check_ca_trust` calls it the way production does — argv list, `capture_output`,
    `text`, `timeout`, `env` — so the call signature is part of what is graded here:
    a refactor that stopped passing `env` would silently reintroduce the
    `LLOYD_NSS_DB` redirect this entry exists to avoid.
    """

    def __init__(self, returncode=1, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr
        self.calls = []

    def __call__(self, argv, **kw):
        self.calls.append((list(argv), kw))
        return self


def _row(answer):
    rows = shc.check_ca_trust(runner=answer)
    assert len(rows) == 1, f"the entry emits exactly one row, got {len(rows)}"
    return rows[0]


def test_the_faked_answers_are_the_answers_the_script_prints():
    """The fixtures match `install-ca.sh`'s format strings, or these tests prove nothing.

    A parser test written against invented output passes forever while the real check
    prints something the parser cannot read — which is exactly how a drift row with no
    fingerprints in it would first be noticed, in production. So the script's own
    printf lines are re-read here, and the words the parser keys on must still be
    there.
    """
    src = INSTALL_CA.read_text(encoding="utf-8")

    # Each fixture's own leading text, minus the values, must be in the script.
    for literal in (
        "[install-ca] CHECK OK: ",
        "[install-ca] CHECK FAILED: ",
        "[install-ca] CHECK INCONCLUSIVE: ",
        "[install-ca]   stored   sha256 ",
        "[install-ca]   expected sha256 ",
        "[install-ca] ERROR: no CA certificate at ",
    ):
        assert literal in src, (
            f"the script no longer prints {literal!r}, so the fixture built on it is "
            "parsing bytes production never emits")
    # The three words the verdict is graded on, and the exit codes behind two of them.
    assert "exit 2" in src, (
        "CHECK INCONCLUSIVE no longer exits 2, which is how the entry distinguishes "
        "an unreadable store from drift")
    assert "stored   sha256 none" in src, (
        "the no-such-nickname branch no longer prints a non-hash stored line, which "
        "is the case the missing-fingerprint detail is written for")


def test_a_drifted_store_is_a_warn_naming_both_fingerprints_and_no_fault():
    """Clause 1: exit 1 with `CHECK FAILED` is `warn`, and the detail carries both hashes.

    Never `ok` and never a hard failure: `advisory` True is the second half — #1108's
    `drift:` rows are NOT the model here, because a stale trust store is latent while
    the frontend serves a public leaf, where a missing root-owned script is live.
    """
    answer = _Answer(returncode=1, stdout=CHECK_FAILED)
    row = _row(answer)

    assert row["verdict"] == "warn", row["status"]
    assert row["status"].startswith("warn:"), row["status"]
    assert row["healthy"] is False, "a drifted store is not healthy"
    assert row["advisory"] is True, (
        "a drifted store must not count as a fault: clause 3 is that the row cannot "
        "flip an otherwise-healthy run into a failure")
    assert STORED_FP in row["status"], row["status"]
    assert EXPECTED_FP in row["status"], row["status"]
    assert "stored" in row["status"] and "expected" in row["status"], row["status"]
    assert row["stored_sha256"] == STORED_FP
    assert row["expected_sha256"] == EXPECTED_FP

    # The control: the same code path scores a matching store ok and not advisory.
    ok = _row(_Answer(returncode=0, stdout=CHECK_OK))
    assert ok["verdict"] == "ok" and ok["healthy"] is True
    assert ok["advisory"] is False, "an ok row is graded, not advisory"

    # The store was read and the nickname is simply not there: still drift, and the
    # script prints `stored   sha256 none (no such nickname)`, which is not a hash.
    # The detail has to say which fingerprint it could not give rather than print the
    # word `none` as if it were one.
    missing = _row(_Answer(returncode=1, stdout=CHECK_FAILED_NO_NICKNAME))
    assert missing["verdict"] == "warn", missing["status"]
    assert EXPECTED_FP in missing["status"], missing["status"]
    assert "stored" in missing["status"] and "no stored" in missing["status"], (
        missing["status"])
    assert "none (no such nickname)" not in missing["status"]
    assert missing["stored_sha256"] is None


def _raiser(argv, **kw):
    """The command cannot be run at all: no bash, a spawn failure, a timeout."""
    raise OSError("No such file or directory: 'bash'")


@pytest.mark.parametrize("runner", [
    # install-ca.sh:88-89 — no certutil, or no HOME and no override: exit 2.
    pytest.param(_Answer(returncode=2, stdout=CHECK_INCONCLUSIVE),
                 id="exit2-inconclusive"),
    # The script's argument guard: exit 1, but the store was never opened. An exit
    # code alone would call this drift, which is clause 2 in the shape of a green row.
    pytest.param(_Answer(returncode=1, stdout=NO_CA_FILE), id="exit1-no-ca-file"),
    # A zero that printed no verdict word at all.
    pytest.param(_Answer(returncode=0, stdout=GIBBERISH), id="exit0-unparseable"),
    # A FAILED word arriving under a code that is not the drift code.
    pytest.param(_Answer(returncode=7, stdout=CHECK_FAILED),
                 id="failed-wrong-exit-code"),
    pytest.param(_raiser, id="command-cannot-run"),
])
def test_a_read_that_never_reached_the_store_is_unknown_never_ok(runner):
    """Clause 2: no-data reads report `unknown`, so an inconclusive check cannot be scored healthy.

    Four shapes of "nothing was measured" and one of "the command could not run", all
    asserting the same three things. `verdict != "ok"` is the property; `warn` is
    excluded too, because `warn` is itself a claim about a mismatch that was measured.
    """
    row = _row(runner)

    assert row["verdict"] == "unknown", row["status"]
    assert row["healthy"] is False, "a read with no data is never a pass"
    assert row["advisory"] is True
    assert row["status"].startswith("unknown:"), row["status"]


def test_the_drift_row_cannot_flip_an_otherwise_healthy_run_into_a_failure():
    """Clause 3: with `--check` forced to exit 1, every pre-existing row keeps its verdict.

    The pre-change expectation is not hand-typed: each row's verdict comes from the
    module's own `_state_verdict()` applied to the same states before and after the
    drift row is added, so the assertion is "the arithmetic over graded rows is
    unchanged", and the only thing the new row may move is the advisory count.
    """
    states = ["RUNNING", "STOPPED", "STARTING", "BACKOFF", "STOPPING", "EXITED",
              "FATAL", "UNKNOWN", "NOT-A-STATE", None]
    graded_before = []
    for state in states:
        healthy, status = shc._state_verdict(f"agent-{state}", state)
        graded_before.append({"name": f"agent-{state}", "healthy": healthy,
                              "status": status, "category": "supervisor"})
    assert sum(r["healthy"] for r in graded_before) == 1, (
        "the graded set stopped containing exactly one healthy state (RUNNING), so "
        "the before/after comparison below is comparing nothing")

    drift = _row(_Answer(returncode=1, stdout=CHECK_FAILED))
    after = graded_before + [drift]

    summary = {"supervisor": "degraded", "ca": "warn"}
    doc = json.loads(shc.format_json(after, summary))
    text = shc.format_text(after, summary)

    for before, after_row in zip(graded_before,
                                 [r for r in doc["services"]
                                  if not r.get("advisory")]):
        assert after_row["healthy"] == before["healthy"], (
            f"{before['name']} changed verdict when the CA row was added")

    # Adding the drift row moves NO fault arithmetic...
    graded_doc = json.loads(shc.format_json(graded_before, summary))
    assert doc["unhealthy"] == graded_doc["unhealthy"], (
        "the drift row was counted as a fault")
    assert doc["healthy"] == graded_doc["healthy"]
    assert f"Overall: {doc['healthy']}/{doc['total_services'] - doc['advisory']} " in text
    assert f"Overall: {graded_doc['healthy']}/{graded_doc['total_services']} " in text
    # ...and the row is still reported, in both formats, visibly.
    assert doc["advisory"] == 1, "the row vanished from the JSON"
    assert "ca-trust" in doc["services"][len(graded_before)]["name"]
    assert "Advisory" in text and "ca-trust" in text
    assert "[!]" in text, "an advisory row is printed with its own marker"


def test_the_entry_always_asks_for_the_read_only_form(monkeypatch, tmp_path):
    """Clause 4: the argv always contains `--check`, and install mode is never invoked.

    Without `--check` the same script WRITES the store (`certutil -D` then `-A`), so a
    health check that costs the thing it measures is #1141's failure mode, not a
    hypothetical. Every scenario the entry can end in is driven here — drift, ok,
    inconclusive, the no-CA line, gibberish and a runner that raises — and each call
    recorded by the fake is asserted to carry the flag.
    """
    argv = shc._ca_argv()
    assert argv[0] == "bash" and argv[-1] == "--check", argv
    assert str(INSTALL_CA) in argv[1], (
        "the entry must run the tree's own script, not a name resolved off PATH")

    scenarios = [
        _Answer(returncode=1, stdout=CHECK_FAILED),
        _Answer(returncode=0, stdout=CHECK_OK),
        _Answer(returncode=2, stdout=CHECK_INCONCLUSIVE),
        _Answer(returncode=1, stdout=NO_CA_FILE),
        _Answer(returncode=0, stdout=GIBBERISH),
    ]
    for answer in scenarios:
        shc.check_ca_trust(runner=answer)
        assert len(answer.calls) == 1, answer.calls
        call_argv, kwargs = answer.calls[0]
        assert "--check" in call_argv, (
            f"the entry built an argv without --check: {call_argv}")
    # The store the row reports on is the operator's own, and that is an assertion
    # about what the child is handed, not about the argv. Set up the parent the way a
    # test of install-ca.sh would — LLOYD_NSS_DB pointing at a scratch DB — and require
    # the child's environment to have dropped exactly that while keeping the rest
    # (`HOME`, which decides the default store path). Written as an absence check on
    # `kwargs.get("env", {})` it would pass even if the entry stopped passing an
    # environment at all, which is the whole guard failing open.
    monkeypatch.setenv("LLOYD_NSS_DB", str(tmp_path / "redirected-nssdb"))
    probe = _Answer(returncode=1, stdout=CHECK_FAILED)
    shc.check_ca_trust(runner=probe)
    _, handed = probe.calls[0]
    child_env = handed.get("env")
    assert child_env is not None, (
        "the entry passed no environment of its own, so the child inherits "
        "LLOYD_NSS_DB and reports on a scratch database")
    assert "LLOYD_NSS_DB" not in child_env, (
        "the child env still carries the override: the row would report on a "
        "redirected store")
    assert child_env.get("HOME") == os.environ.get("HOME"), (
        "HOME did not survive into the child, so the default store path the check "
        "resolves is not the one this shell would read")

    def raiser(argv, **kw):
        raise OSError("bash not found")

    shc.check_ca_trust(runner=raiser)

    # No path added by the diff invokes the script in install mode. The form to look
    # for is a call naming the script with no --check: read the module's source for
    # any invocation of install-ca.sh, and the only argv built there is _ca_argv's.
    src = SCRIPT.read_text(encoding="utf-8")
    assert src.count("_ca_argv()") >= 2, (
        "the argv is supposed to be built in exactly one place, and the built argv "
        "is what is checked above")
    build_body = re.search(r"def _ca_argv\(\).*?\n\n\n", src, re.S).group(0)
    assert '"--check"' in build_body, build_body
    assert "update-ca-trust" not in src, "#1241's ruling: nothing here touches /etc"
    assert "/etc/ca-certificates" not in src, "#1241's ruling"
