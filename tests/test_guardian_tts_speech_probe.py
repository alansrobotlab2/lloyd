"""#2282: an `agent-tts` recovery is confirmed by speech, not by `/health`.

Measured 2026-10-06: `GET :8090/health` answered 200 `healthy` in 0.000757 s while a
synthesis against the stock voice `Vivian` returned 500, and the shipped
`recover_service` reported that box recovered — `return True, f"{msg}; {policy.TTS_HEALTH_URL}
answering"`, with zero references to `speak.`/`synthesize` in its body. `guardian.py::_url_for`'s
own docstring says "'the supervisor says it is up' and 'the box can speak' are different
claims ... a recovery has to be confirmed by the second, not the first"; the first is what the
verdict rested on. The stakes are the restart budget, not the wording: the `service_recovery`
ledger row is written BEFORE the attempt, `policy.FLAP_HALT_AFTER` is 2 inside a 6 h window,
and `_recover_infra` re-speaks the newest swallowed alert on the strength of the True returned
here, advancing the replay cursor on a successful SPAWN. Two confirmations of a silent box
therefore lock the guardian out of the only alert route it has and permanently lose the alert
it claims to have re-said.

WHAT THIS FILE IS FOR, SPECIFICALLY. The round before this one wrote the same production code
and was refused twice with every clause graded "partial — the pinning node cannot run": its
calibration fixture landed under `tests/fixtures/`, where the root `.gitignore`'s `*.json`
rule meant `git add` silently did nothing, and this module read that file at import time, so
the absence was a COLLECTION ERROR, not a skip — 16 698 tests green with none of these nodes
executed. So the fixture is committed through a scoped negation in `tests/fixtures/.gitignore`
and `test_the_calibration_fixture_is_tracked_despite_the_root_json_rule` passes the name
through the real mechanism (`git check-ignore`, with a sibling name as the positive control)
instead of trusting that somebody remembered the negation.

WHY EVERY NODE HERE RUNS `guardian.py` IN A CHILD INTERPRETER. `guardian.py` imports its
siblings by bare name (`policy`, `probes`, `speak`), so it is importable only with
`agent-services/guardian` on `sys.path`, and that directory holds modules whose names collide
with `scripts/automod` and `agent_services_lib` — `conftest.py` keeps it off the session path
for the whole run. `test_guardian_recovery_respeak.py` solves this by running a whole guardian
in a subprocess; this file does the same at function scale through `_child()` below, and its
`_guardian_sibling()` is the loader for the nodes that read `probes.py` and `speak.py`
directly. The stub lives in the child, at `urllib.request.urlopen` AND
`urllib.request.build_opener` — `probes.probe` builds its own opener, so stubbing `urlopen`
alone leaves the health stage hitting the live service on this box, which is exactly how a
"refused" arm once passed while talking to a real server.

The stub's numbers are calibrated to the live service and GENERATED from
`tests/fixtures/tts_speech_probe_calibration.json`, and one node asserts that generation
against the module's own measurement functions (another re-derives the committed rms and
duration from the bytes the accepted arm is built from; a third, marked, asks the real
synthesiser): configured `clone:dave_cullen` measured 2.56 s at rms 2250, the stock-voice 500
is `does not support generate_custom_voice`, and the non-utterance that clears the loudness
floor measured 0.22 s at rms 311. A sine's rms is its amplitude over sqrt(2), which is why
`tone()` takes an rms and not an amplitude — the first cut of this file asked for 2250, got
1591, and the detail string dutifully reported the wrong figure.
"""

import json
import math
import subprocess
import sys
import importlib.util
import os
from array import array
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
GUARDIAN_DIR = REPO_ROOT / "agent-services" / "guardian"
FIXTURE_REL = "tests/fixtures/tts_speech_probe_calibration.json"
#: A name in the same directory, one character away from the fixture's, with no negation of
#: its own. It exists to be ignored: without it, a `check-ignore` answer of "not ignored" on
#: the fixture is indistinguishable from a `check-ignore` that answers "not ignored" to
#: everything — which is what a broken positive control always looks like.
CONTROL_REL = "tests/fixtures/tts_speech_probe_uncalibrated.json"


def _calib() -> dict:
    """The committed measurement bytes. Read per call, never at import: an import-time read
    is what turned last round's missing fixture into a collection error for the whole file."""
    return json.loads((REPO_ROOT / FIXTURE_REL).read_text(encoding="utf-8"))


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                          capture_output=True, text=True)


def _guardian_sibling(name: str):
    """One of the guardian's sibling modules, loaded from THIS tree and unloaded after.

    `probes.py` starts with `import policy`, so reading it means putting the guardian
    directory on `sys.path` FOR THE LOAD — and loading it by NAME is not enough either: a
    `sys.modules` binding can predate this file and point at the STAGED copy under
    `~/.local/state/lloyd-guardian/bin`, the one systemd runs and which does not contain the
    diff under test — the trap the previous round's clause-4 node fell into, printing the
    pre-change `answering` detail while the new speech check ran for real. So the module is
    loaded by PATH, under a key this call owns, with the guardian directory on the path only
    while it executes (that is what lets its own `import policy` resolve), and every key the
    load created — `policy` included, not just the module asked for — is gone on the way out.
    Nothing is left bound under a bare name, because a lingering `policy` rebinds a constant
    another test reads, and that test then asserts against the wrong module and still passes.
    """
    path = GUARDIAN_DIR / f"{name}.py"
    key = f"_probe_pin_{name}"
    saved_module = sys.modules.get(name)
    saved_path, had_modules = list(sys.path), set(sys.modules)
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    sys.path.insert(0, str(GUARDIAN_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved_path
        for created in set(sys.modules) - had_modules:
            sys.modules.pop(created, None)
        if saved_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved_module
    return module


def tone(seconds: float, rms: int, freq: float = 200.0,
         sample_rate: int = 24000) -> bytes:
    """Mono s16le PCM with the asked-for ROOT-MEAN-SQUARE, not amplitude.

    A sine's rms is its amplitude over sqrt(2), so the amplitude is derived from the rms
    named. Getting that backwards is what made a first cut of this file report `rms=1591`
    where it meant 2250; these fixtures are calibrated to a real measurement, so the
    parameter's NAME is what keeps that honest.
    """
    amplitude = int(round(rms * math.sqrt(2)))
    n = int(sample_rate * seconds)
    a = array("h", (int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
                    for i in range(n)))
    return a.tobytes()


class _Stream:
    """A streaming response: `read(n)` in chunks, like the one `_read_bounded` consumes."""

    def __init__(self, body: bytes):
        self._b = body

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            b, self._b = self._b, b""
            return b
        b, self._b = self._b[:n], self._b[n:]
        return b

    def close(self) -> None:
        self._b = b""


# ── The two configs the probe reads ────────────────────────────────────────
#
# Both are written into a tmp guardian state dir as `voice.json` and loaded by the real
# `speak.load_config`, which overlays that file on `speak.DEFAULTS`.

def _live_cfg() -> dict:
    """This box's `~/.local/state/lloyd-guardian/voice.json`, as the calibration recorded it.

    Its values deliberately coincide with `speak.DEFAULTS` on `model` and `voice`, because
    that is the truth on this machine — which is exactly why this config is NOT the one that
    pins provenance. See `_provenance_cfg` for the arm that can tell a loaded file from the
    built-in defaults.
    """
    cal = _calib()["configured_voice"]
    return {"api_url": "http://127.0.0.1:8090", "model": cal["model"], "voice": cal["voice"],
            "sample_rate": cal["sample_rate"], "max_audio_seconds": 60.0,
            "synth_timeout": 60.0, "synth_wall_clock": 90.0, "max_chars": 240,
            "quiet_hours": {"enabled": False, "start": 23, "end": 7, "allow_critical": False}}


def _provenance_cfg(speak_defaults: dict) -> dict:
    """A config that disagrees with `speak.DEFAULTS` on every key clause 3 names.

    `speak.DEFAULTS` already carries `model: qwen3-tts` and `voice: clone:dave_cullen` — the
    review of the previous round caught the fixture that equalled it on all five keys, and
    said so: it "pins the values, not their provenance", because a cfg built from DEFAULTS
    with the state dir never opened would satisfy it. Here every asserted value is distinct,
    so the arm fails if the probe reads the defaults.
    """
    cfg = _live_cfg()
    assert (cfg["model"], cfg["voice"]) == (speak_defaults["model"], speak_defaults["voice"]), (
        "the live-shaped arm is supposed to mirror DEFAULTS on these two keys — that equality "
        "is the whole reason a separate provenance arm has to exist")
    distinct = {"api_url": "http://127.0.0.1:8099",
                "model": "probe-model-not-the-default-one",
                "voice": "clone:probe-voice-not-the-default-one",
                "sample_rate": 16000, "max_audio_seconds": 12.0, "synth_timeout": 7.0,
                "synth_wall_clock": 20.0, "max_chars": 240,
                "quiet_hours": {"enabled": False, "start": 23, "end": 7,
                                "allow_critical": False}}
    for key in ("model", "voice", "sample_rate", "synth_timeout", "max_audio_seconds"):
        assert distinct[key] != speak_defaults[key], (
            f"the provenance arm stops pinning anything if DEFAULTS moves to {key}="
            f"{distinct[key]!r}")
    return distinct


# ── The child harness ──────────────────────────────────────────────────────
#
# A scenario is a (speech, health, mode) triple naming what each side does. Nothing here
# imports `guardian` in-process, and the scenario travels in the environment rather than in
# `str.format`, so a brace in the script can never be silently eaten.

CHILD = r"""
import json, os, socket, sys, types, urllib.error, urllib.request
from array import array
from math import pi, sin, sqrt
from pathlib import Path

SC = json.loads(os.environ["PROBE_SCENARIO"])
sys.path.insert(0, SC["guardian_dir"])
import guardian as G                                      # noqa: E402
import speak                                              # noqa: E402

STATE = Path(SC["state"])
SR = int(SC["voice_cfg"]["sample_rate"])
CAPTURED = {"health_calls": 0, "synth_calls": 0}
SPAWNED: list = []


def tone(seconds, rms, freq=200.0, sr=SR):
    amp = int(round(rms * sqrt(2)))          # a sine's rms is amplitude / sqrt(2)
    n = int(sr * seconds)
    a = array("h", (int(amp * sin(2 * pi * freq * i / sr)) for i in range(n)))
    return a.tobytes()


BODIES = {"empty": b""}
for _row in SC["bodies"]:
    BODIES[_row["name"]] = tone(_row["seconds"], _row["rms"])

RAISES = {
    "http": urllib.error.HTTPError("http://x", 500, "Server Error", {}, None),
    "url": urllib.error.URLError("[Errno 111] Connection refused"),
    "timeout": socket.timeout("synth timed out"),
}


class Resp:
    # `synthesize` consumes the response inside a `with`, so this is a context manager and
    # not a bag of bytes; `read(n)` is per-call because `_read_bounded` reads in chunks and
    # grades the byte ceiling as it goes.
    def __init__(self, body):
        self._b = body
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=-1):
        if n is None or n < 0:
            b, self._b = self._b, b""
            return b
        b, self._b = self._b[:n], self._b[n:]
        return b

    def close(self):
        self._b = b""


def fake_urlopen(req, timeout=None):
    if req.full_url.endswith("/health"):
        CAPTURED["health_calls"] += 1
        if SC["health"] == "refused":
            raise urllib.error.URLError("[Errno 111] Connection refused")
        if SC["health"] == "initializing":
            return Resp(b'{"status": "initializing"}')
        return Resp(b'{"status": "healthy"}')
    CAPTURED["synth_calls"] += 1
    CAPTURED["url"] = req.full_url
    CAPTURED["payload"] = json.loads(req.data.decode())
    CAPTURED["timeout"] = timeout
    if SC["speech"] in RAISES:
        raise RAISES[SC["speech"]]
    return Resp(BODIES[SC["speech"]])


class _Opener:
    # `probes.probe` builds its own no-proxy opener with `build_opener` rather than calling
    # `urlopen`. Stubbing only `urlopen` would leave the health stage reaching the REAL
    # :8090, which on this box answers healthy — so a "refused" arm would have passed against
    # the live service. Both routes go through this one fake.
    def open(self, req, timeout=None):
        return fake_urlopen(req, timeout)


urllib.request.urlopen = fake_urlopen
urllib.request.build_opener = lambda *a, **k: _Opener()

# `wait_healthy` and `probe` import `time` inside their bodies, and the no-answer arm must
# exhaust the REAL `policy.HEALTH_WAIT_TTS` = 420 s budget, not a stub of it. This clock
# advances only when the code sleeps, so each 1 s poll moves monotonic 1 s and the refused
# arm runs its real 420 polls in milliseconds. `synthesize`'s own wall-clock budget reads the
# same clock and never sleeps, so it measures normally — which is why the "good" arm still
# gets its 2.56 s of audio.
import time as _time                                          # noqa: E402
_vclock = [0.0]


def _fake_monotonic():
    return _vclock[0]


def _fake_sleep(seconds):
    _vclock[0] += float(seconds)


_time.monotonic = _fake_monotonic
_time.sleep = _fake_sleep

g = object.__new__(G.Guardian)
# A real Path: `gstate.append_event` opens it, and the row this attempt writes is the row a
# reader would otherwise be told had said "Recovered".
g.state = types.SimpleNamespace(ledger=STATE / "promotions.jsonl")
g.ledger = g.state.ledger
g.sup = type("Sup", (), {"stop": lambda self, p, wait=True: (True, "stopped"),
                         "start": lambda self, p, wait=False: (True, "started")})()
g.gdir = STATE
g.heartbeat = lambda *a, **k: None
g._beat = lambda: None
g._recent_recoveries = lambda program: 0
# Only the alert-routing arms need these; setting them always costs nothing and keeps the
# two modes driving one object rather than two fixtures that can drift apart.
g.programs = ("agent-tts",)
g.interval = 5.0
g.liveness_fail_streak = 1
g.alerts = []
g.alert = lambda level, title, body, **kw: g.alerts.append(
    {"level": level, "title": title, "body": body, **kw})
g.notifier = types.SimpleNamespace(voice_window=3600.0)

if SC["mode"] == "infra":
    # The re-speak is a detached worker; capture the spawn rather than perform it, and let
    # the master mute be open (pytest's own session sets it to "0", which this child would
    # otherwise inherit and then report a muted channel as a routing decision).
    os.environ["LLOYD_VOICE_ALERTS"] = "1"
    speak.subprocess.Popen = lambda *a, **k: SPAWNED.append(a[0])
    for _text in SC.get("loss_texts", ()):
        speak._record_loss(STATE, _text, "ConnectionRefusedError: [Errno 111]")
    _recoverable = list(g._recoverable_down(SC["live_reason"]))
    try:
        ok, detail = g._recover_infra(SC["live_reason"]), None
    except Exception as exc:                                   # noqa: BLE001
        ok, detail = "CRASHED", f"{type(exc).__name__}: {exc}"
else:
    _recoverable = None
    try:
        ok, detail = g.recover_service("agent-tts", SC["reason"])
    except Exception as exc:                                   # noqa: BLE001
        ok, detail = "CRASHED", f"{type(exc).__name__}: {exc}"

_log_path = STATE / speak.LOG_NAME
print(json.dumps({
    "ok": ok, "detail": detail, "captured": CAPTURED,
    "payload": CAPTURED.get("payload"), "synth_url": CAPTURED.get("url"),
    "synth_timeout": CAPTURED.get("timeout"), "probe_text": speak.PROBE_TEXT,
    "floors": [speak.PROBE_MIN_SECONDS, speak.PROBE_MIN_RMS],
    "max_audio": speak.load_config(STATE)["max_audio_seconds"],
    "voice_log": _log_path.read_text(encoding="utf-8") if _log_path.exists() else "",
    "alerts": g.alerts, "spawned": SPAWNED, "recoverable": _recoverable,
    "loss": speak.read_loss_record(STATE),
}))
""".strip()


def _child(tmp_path: Path, speech: str = "good", *, health: str = "ok", mode: str = "recover",
           cfg: dict | None = None, seed_loss: bool = False) -> dict:
    """Run the real `recover_service` (or `_recover_infra`) against one scenario.

    A fresh interpreter for every arm, on purpose: the previous arm's stubbed
    `urllib.request` must not be what the next one reads, and `urlopen` is monkeypatched on a
    module the whole session shares.
    """
    cal = _calib()
    cfg = cfg or _live_cfg()
    state = tmp_path / "guardian"
    state.mkdir(exist_ok=True)
    (state / "voice.json").write_text(json.dumps(cfg), encoding="utf-8")
    bodies = {"good": [cal["configured_voice"]["seconds"], cal["configured_voice"]["rms"]],
              "short": [cal["rejected"][0]["seconds"], cal["rejected"][0]["rms"]],
              "quiet": [cal["rejected"][1]["seconds"], cal["rejected"][1]["rms"]]}
    scenario = {
        "guardian_dir": str(GUARDIAN_DIR), "state": str(state),
        "voice_cfg": cfg, "speech": speech, "health": health, "mode": mode,
        "bodies": [{"name": k, "seconds": v[0], "rms": v[1]} for k, v in bodies.items()],
        "reason": "synth failures (URLError: [Errno 111] Connection refused)",
        "live_reason": "agent-tts: STOPPED without an intentional stop.",
        "loss_texts": (["Landed: 3b7db4e2", "Rolled back: memory/kg.sqlite"]
                       if seed_loss else []),
    }
    env = {**os.environ, "PROBE_SCENARIO": json.dumps(scenario),
           "LLOYD_GUARDIAN_STATE": str(state)}
    r = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True,
                       timeout=300, env=env, cwd=str(REPO_ROOT))
    assert r.returncode == 0, f"child failed: {r.stderr[-4000:]}"
    return json.loads(r.stdout.strip().splitlines()[-1])


def _measured_rms(seconds: float, rms: int, cal: dict) -> int:
    """What the module under test measures a generated stub at — not the number asked for.

    Integer quantisation moves a generated sine's rms by a unit (the 0.22 s row asked for
    312 and measures 311), and a detail string that quoted 312 would be quoting a figure the
    code never produced. This derives the honest one through the same functions the probe
    uses.
    """
    speak = _guardian_sibling("speak")
    pcm = tone(seconds, rms, sample_rate=cal["configured_voice"]["sample_rate"])
    return speak._pcm_rms(pcm)


# ── Clause 1: a healthy endpoint with failing speech is not a recovery ─────

@pytest.mark.parametrize("speech,why", [
    ("url", "the synthesiser refused the connection outright"),
    ("http", "the server answered the speech route 5xx"),
    ("timeout", "the speech route hung past the config's own synth_timeout"),
    ("empty", "the server answered 200 and sent no audio at all"),
])
def test_a_healthy_endpoint_with_failing_speech_refuses_the_recovery(tmp_path, speech, why):
    """`recover_service` returns False, naming the speech failure, while /health says healthy.

    Four shapes because the fix must not be `if pcm is None`. Measured against the live
    synthesiser today: it answers `invalid_model` with 400, `profile_not_found` with 404 and
    `invalid_input` with 400 by RAISING `HTTPError` out of `urlopen`, which propagates through
    `synthesize` — its own non-2xx branch is unreachable through `urllib` — while a 200 with an
    empty body is the only shape that yields no bytes. A probe written as a None-check would
    therefore let every status-coded failure escape the recovery as an exception instead of
    turning it into a refusal, which is the commonest real failure there is.

    "answers healthy" in the reason is the clause rather than decoration: the endpoint stage
    still reports what it saw, so a reader can see the endpoint is NOT what failed.
    """
    r = _child(tmp_path, speech)
    assert r["ok"] is False, f"{why}: reported success without speech: {r['detail']}"
    assert "cannot speak" in r["detail"], r["detail"]
    assert "answers healthy" in r["detail"], r["detail"]
    assert r["captured"]["health_calls"] >= 1, "the endpoint stage must still run first"
    assert r["captured"]["synth_calls"] == 1, f"{why}: {r['captured']}"


def test_the_refusal_reason_names_which_speech_claim_failed(tmp_path):
    """The reader of the ledger has to learn WHICH claim failed, in one line.

    A raise names the exception type, a short clip names its measured length against the
    floor it missed, a silent one its rms: three different repairs, and a reason that said
    only "speech failed" would send a human to check all three. The numbers are the ones
    `tests/fixtures/tts_speech_probe_calibration.json` commits, re-derived from the child's
    own measurement of the generated stub.
    """
    cal = _calib()
    r = _child(tmp_path, "http")
    assert "HTTPError" in r["detail"], r["detail"]

    r = _child(tmp_path, "empty")
    assert "no audio" in r["detail"], r["detail"]

    short = cal["rejected"][0]
    r = _child(tmp_path, "short")
    assert f"{short['seconds']:.3f}s" in r["detail"], r["detail"]
    assert f"{cal['floors']['min_seconds']}s floor" in r["detail"], r["detail"]

    quiet = cal["rejected"][1]
    r = _child(tmp_path, "quiet")
    assert f"rms={_measured_rms(quiet['seconds'], quiet['rms'], cal)}" in r["detail"], r["detail"]
    assert f"{cal['floors']['min_rms']} floor" in r["detail"], r["detail"]


# ── Clause 2: only audio over BOTH floors is a recovery, and the detail quotes them ──

def test_the_recovery_is_confirmed_only_over_both_floors_and_quotes_both_numbers(tmp_path):
    """The configured voice's live numbers recover the box; a stub and a silence do not.

    The two floors are not redundant and this is where that is pinned: the 0.22 s stub row
    measured rms 311 and would clear a loudness floor alone, and a 2.56 s stream of near-zeros
    clears a duration floor alone. One check would let the other class through, which is why
    `recover_service` refuses both — and reports the numbers it measured for the one it
    accepted, so `:8090 answering`, the string this item exists to delete, is joined by a
    measurement a reader can check.
    """
    cal = _calib()
    cfg = cal["configured_voice"]
    r = _child(tmp_path, "good")
    assert r["ok"] is True, r["detail"]
    assert cfg["voice"] in r["detail"], r["detail"]
    assert f"{cfg['seconds']:.2f}s" in r["detail"], r["detail"]
    assert f"rms={_measured_rms(cfg['seconds'], cfg['rms'], cal)}" in r["detail"], r["detail"]
    assert r["detail"].index("rms=") > r["detail"].index("answering"), (
        f"the measurement has to be the end of the claim, not the endpoint: {r['detail']}")


def test_audio_under_either_floor_is_refused_naming_the_floor_it_missed(tmp_path):
    """Each floor refuses a class the other one lets through.

    The short row is the one a single-check probe misses: it is loud enough (rms 311 against a
    200 floor) and too short to be an utterance. The quiet row is the mirror. A change that
    dropped either floor would keep this node red while the other arm stayed green, which is
    the point of asserting the floor NAME rather than just `ok is False`.
    """
    cal = _calib()
    short, quiet = cal["rejected"][0], cal["rejected"][1]
    assert (short["fails"], quiet["fails"]) == ("min_seconds", "min_rms"), cal["rejected"]

    r = _child(tmp_path, "short")
    assert r["ok"] is False, r["detail"]
    assert "cannot speak" in r["detail"], r["detail"]
    assert f"{cal['floors']['min_seconds']}s floor" in r["detail"], r["detail"]

    r = _child(tmp_path, "quiet")
    assert r["ok"] is False, r["detail"]
    assert "cannot speak" in r["detail"], r["detail"]
    assert f"{cal['floors']['min_rms']} floor" in r["detail"], r["detail"]


def test_the_floors_are_the_ones_the_committed_calibration_states():
    """Clause 6 as amended: the floors and the calibration bytes cannot drift apart.

    The previous round's review called a sibling assertion "constants and a fixture echo
    rather than behaviour", so this node is narrowed to the one thing that is genuinely a
    constant — the two floor values the probe enforces — and the behavioural half of the same
    worry lives in `test_the_accepted_stub_reproduces_the_committed_calibration_numbers`.
    """
    cal = _calib()
    speak = _guardian_sibling("speak")
    assert speak.PROBE_MIN_SECONDS == cal["floors"]["min_seconds"], speak.PROBE_MIN_SECONDS
    assert speak.PROBE_MIN_RMS == cal["floors"]["min_rms"], speak.PROBE_MIN_RMS
    for row in cal["rejected"]:
        assert (row["seconds"] < speak.PROBE_MIN_SECONDS
                or row["rms"] < speak.PROBE_MIN_RMS), (
            f"a 'rejected' calibration row would now pass the probe: {row}")
    assert cal["endpoint_that_500s"]["voice"] != cal["configured_voice"]["voice"], (
        "the catalog voice that answers 500 must not be the probe's voice")


def test_the_accepted_stub_reproduces_the_committed_calibration_numbers():
    """The stub is generated FROM the committed bytes and then measured back through the
    probe's own functions — so the accepted arm is the calibration, not a copy of it.

    If `_pcm_rms` or `_pcm_seconds` were wrong (the sqrt(2) error this file's own history
    records), this fails even though every stub-driven arm above would still be self-consistent
    and green.
    """
    cal = _calib()
    cfg = cal["configured_voice"]
    speak = _guardian_sibling("speak")
    pcm = tone(cfg["seconds"], cfg["rms"], sample_rate=cfg["sample_rate"])
    assert speak._pcm_seconds(len(pcm), cfg["sample_rate"]) == pytest.approx(
        cfg["seconds"], abs=1e-4)
    assert speak._pcm_rms(pcm) == cfg["rms"], (
        f"the accepted arm's stub measures rms={speak._pcm_rms(pcm)} against a committed "
        f"{cfg['rms']}: the floors calibrated on it are calibrated on something else")


def test_the_calibration_fixture_is_tracked_despite_the_root_json_rule():
    """The mechanism that ate the last round, exercised on every run.

    Root `.gitignore` ends `*.json`; the fixture is admissible only through one scoped
    negation in `tests/fixtures/.gitignore`, and `git add` of an ignored path is SILENT — it
    exits 0 and commits nothing, which is how a round got refused twice for a fixture its
    author believed he had committed.

    So ask the real mechanism, and read what it actually says: `git check-ignore -v` exits 0
    for ANY matching rule, including a negation, and prints the winning rule as
    `file:line:pattern` — a leading `!` on that pattern is the answer "the last rule says this
    is NOT ignored", which is why reading the exit status alone would have called this fixture
    ignored a minute before this node was written. The sibling name with no negation of its own
    is the positive control: it must come back with a rule that has no `!`, or an answer of
    "negation" for the fixture would mean nothing — which is what a broken control always looks
    like. `ls-files` then carries the part no ignore rule can prove: that the bytes are in the
    index, which is the thing that was silently missing.
    """
    def rule(rel: str) -> tuple[int, str]:
        p = _git("check-ignore", "--no-index", "-v", rel)
        return p.returncode, (p.stdout.strip().split("\t")[0] if p.stdout.strip() else "")

    ctl_code, ctl_rule = rule(CONTROL_REL)
    assert ctl_code == 0 and ctl_rule and not ctl_rule.endswith(":!"), (
        f"positive control broken: a sibling fixture name resolved to {ctl_rule!r} "
        f"(exit {ctl_code}), so `check-ignore` is not grading these paths at all")
    assert not ctl_rule.startswith("tests/fixtures/"), (
        f"the control is being ignored by the fixtures file itself, so the negation block this "
        f"node depends on is not scoped the way it looks: {ctl_rule}")

    code, matched = rule(FIXTURE_REL)
    assert code == 0 and matched.endswith(":!tts_speech_probe_calibration.json"), (
        f"{FIXTURE_REL} resolves to {matched!r} (exit {code}): the scoped negation in "
        "tests/fixtures/.gitignore is not the last matching rule, so `git add` would commit "
        "nothing and every node in this file that reads the fixture would error at collection")

    tracked = _git("ls-files", "--error-unmatch", FIXTURE_REL)
    assert tracked.returncode == 0, (
        f"{FIXTURE_REL} is admissible but not in the index: {tracked.stderr.strip()}")


# ── Clause 3: the request asks the alerts' question, of the loaded config ───

def test_the_confirming_request_carries_the_loaded_configs_values_not_the_defaults(tmp_path):
    """The payload `synthesize` built, captured at the wire, from a config that DISAGREES
    with `speak.DEFAULTS` on every key the clause names.

    Asserted as the bytes that would have gone over the socket, not a dict this file
    constructed. This is the node the previous round's fixture could not be: its `voice.json`
    equalled DEFAULTS on `model`, `voice`, `sample_rate`, `synth_timeout` and
    `max_audio_seconds`, so it would have passed against a cfg built from DEFAULTS with the
    state dir never opened. Every value here is distinct from the built-in one, so the arm
    fails if the probe stops reading the file.
    """
    speak = _guardian_sibling("speak")
    cfg = _provenance_cfg(speak.DEFAULTS)
    r = _child(tmp_path, "good", cfg=cfg)
    payload = r["payload"]
    assert payload is not None, "the probe made no synthesis call at all"
    assert payload["model"] == cfg["model"], payload
    assert payload["model"] != speak.DEFAULTS["model"], "read from DEFAULTS, not the file"
    assert payload["voice"] == cfg["voice"], payload
    assert payload["voice"] != speak.DEFAULTS["voice"], "read from DEFAULTS, not the file"
    assert payload["voice"].startswith("clone:"), payload["voice"]
    assert r["synth_timeout"] == cfg["synth_timeout"], "the config's, not a new one"
    assert r["max_audio"] == cfg["max_audio_seconds"], r["max_audio"]
    assert r["synth_url"].endswith("/v1/audio/speech"), r["synth_url"]


def test_the_probe_asks_for_the_configs_voice_rather_than_a_name_it_hardcodes(tmp_path):
    """Point the config at the stock voice and the probe asks for the stock voice.

    The clause's "never a catalog voice name" is a statement about where the name comes from,
    and the only way to test that is to make the config and the catalog disagree. With
    `voice: Vivian` in `voice.json` the probe must send `Vivian` — proving the name travels
    from the config and nothing in the probe supplies a catalog default behind it. (On this
    box that request answers 500, which is #2281's standing mismatch and precisely why the
    shipped config must name the clone: the same server answers a `clone:` voice and a catalog
    voice differently.)
    """
    cfg = _live_cfg()
    cfg["voice"] = _calib()["endpoint_that_500s"]["voice"]
    r = _child(tmp_path, "http", cfg=cfg)
    assert r["payload"]["voice"] == "Vivian", r["payload"]
    assert r["captured"]["synth_calls"] == 1, r["captured"]


def test_the_canned_utterance_is_short_enough_to_sit_under_the_configs_ceiling(tmp_path):
    """The probe's text is 12 characters, and the ceiling that would truncate its reply is
    the config's — with 46x of margin between the two.

    `synthesize` asks for `max_chars: 240` worth of alert and `_read_bounded` CUTS a reply at
    `max_audio_seconds`; a canned text long enough to approach that ceiling would report a
    truncation as a duration, which measures the bound rather than the voice. So the canned
    text has to be far under the cap, and the arm below proves the ceiling never fires for it.
    """
    cal = _calib()
    r = _child(tmp_path, "good")
    speak = _guardian_sibling("speak")
    assert r["payload"]["input"] == speak.PROBE_TEXT == "Voice check."
    assert len(speak.PROBE_TEXT) == 12, repr(speak.PROBE_TEXT)
    assert len(speak.PROBE_TEXT) * 5 <= speak.DEFAULTS["max_chars"], (
        "the canned text must sit far under the cap the alerts use, not near it")
    assert "ceiling fired" not in r["voice_log"], (
        f"the probe's own reply was truncated, so its duration measured the bound: "
        f"{r['voice_log']}")
    assert (r["captured"]["synth_calls"] == 1
            and cal["configured_voice"]["seconds"] < r["max_audio"]), r


def test_the_bounded_read_guard_still_cuts_an_over_run_stream(tmp_path):
    """The guard the clause leans on is live: an over-run stream is cut, kept as a head, and
    logged — not trusted.

    Driven straight into `speak._read_bounded` with a 1.0 s ceiling and 4 s arriving in
    64 KiB chunks, which is how the 2026-09-28 runaway (655.57 s of scrambled audio for a
    160-character alert) arrived too. The answer must be a truncation rather than a mute,
    because the decoder emits the real sentence first; what matters for #2282 is that the
    probe's duration is a measurement of audio that was actually bounded, and that a stream
    which over-runs is recorded rather than silently shortened.
    """
    speak = _guardian_sibling("speak")
    cfg = dict(speak.DEFAULTS)
    cfg["max_audio_seconds"] = 1.0
    cfg["synth_wall_clock"] = 90.0
    incoming = tone(4.0, 2250, sample_rate=cfg["sample_rate"])
    pcm = speak._read_bounded(_Stream(incoming), cfg, tmp_path)
    kept = speak._pcm_seconds(len(pcm), cfg["sample_rate"])
    assert kept == pytest.approx(1.0, abs=1e-6), f"kept {kept}s of a 1.0s ceiling"
    assert len(pcm) < len(incoming), "the guard let an over-run stream through whole"
    logged = (tmp_path / speak.LOG_NAME).read_text(encoding="utf-8")
    assert "synth duration ceiling fired" in logged, logged
    assert "max_audio_seconds=1.00s" in logged, logged
    assert "kept the first 1.00s" in logged, logged


# ── Clause 4: the endpoint stage is unchanged, and still the stage before ───

def test_an_endpoint_that_never_answers_healthy_is_refused_before_any_speech(tmp_path):
    """A server that never answers `healthy` in `policy.HEALTH_WAIT_TTS` still fails the old
    way, and spends no synthesis attempt doing it.

    Two arms: a refused connection, and the `initializing` status the synthesiser serves while
    it warms — which `probes.ok_statuses_for` rejects for this URL, so warming is not
    recovered. In both the speech stage must NOT run: the endpoint is the cheaper question and
    the more specific answer, and a POST issued during the four-minute cold compile would be
    refused at connect time and report the wrong reason. The budget is the real
    `policy.HEALTH_WAIT_TTS`, spent for real by the real poll loop (the child's clock advances
    only on `sleep`, so this costs milliseconds, not seven minutes).
    """
    url = _guardian_sibling("probes").policy.TTS_HEALTH_URL
    for health in ("refused", "initializing"):
        r = _child(tmp_path, "good", health=health)
        assert r["ok"] is False, f"{health}: recovered without a healthy endpoint"
        assert "never answered" in r["detail"], f"{health}: {r['detail']}"
        assert url in r["detail"], f"{health}: {r['detail']}"
        assert "cannot speak" not in r["detail"], f"{health}: {r['detail']}"
        assert r["captured"]["synth_calls"] == 0, (
            f"{health} reached the speech stage: {r['captured']}")


def test_ok_statuses_for_still_requires_healthy_only_for_the_synthesisers_url():
    """`probes.ok_statuses_for` is untouched: `healthy` is accepted for :8090 and nowhere else.

    Read through `_guardian_sibling`, which loads the guardian's own `probes` from this tree
    and leaves nothing behind — the leak this file must not create is a silent one: a lingering
    `policy` rebinds a constant another test reads, and that test then asserts against the
    wrong module and still passes.
    """
    probes = _guardian_sibling("probes")
    tts = probes.policy.TTS_HEALTH_URL
    assert probes.ok_statuses_for(tts) == probes.HEALTHY_OK_STATUSES
    assert "healthy" in probes.ok_statuses_for(tts)
    for url in ("http://127.0.0.1:8080/health", "http://127.0.0.1:8500/health",
                "http://127.0.0.1:8091/health"):
        assert probes.ok_statuses_for(url) == probes.OK_STATUSES, url
        assert "healthy" not in probes.ok_statuses_for(url), url


def test_loading_a_guardian_sibling_leaves_no_module_behind():
    """The loader's own lease, asserted: no `sys.modules` key and no `sys.path` entry survives.

    A file with one loader and no witness for what it promises is how a leak gets into the
    suite, and this file is the only place in it that needs a guardian sibling loaded by path.
    """
    before_modules, before_path = set(sys.modules), list(sys.path)
    _guardian_sibling("probes")
    assert set(sys.modules) == before_modules, sorted(set(sys.modules) - before_modules)
    assert sys.path == before_path, "left a sys.path entry behind"


# ── The alert the verdict produces ─────────────────────────────────────────
#
# `_recover_infra` is the caller, and the two nodes below are the difference between a wrong
# verdict costing a sentence and costing the alert channel: it writes the `service_recovery`
# ledger row BEFORE the attempt, and on True it re-speaks the newest swallowed alert, whose
# replay cursor moves on a successful SPAWN (`dispatch` returns right after `Popen`). So a
# speech refusal that still said "Recovered" would report a re-speak the box never made.
#
# Driven in the child for the reason the last two reviews recorded: in-process, an
# `import guardian` here can resolve to the STAGED copy systemd runs
# (`~/.local/state/lloyd-guardian/bin/guardian.py`), which does not have this diff — the
# previous round's clause-4 node printed the pre-change `answering` detail while the new
# speech check ran for real. The child's `sys.path` names this tree, so what it drives is the
# file under test.

def test_a_recovery_refused_by_speech_alerts_a_human_and_writes_no_recovered_row(tmp_path):
    """No `Recovered:` info alert, no "has just been said again", and `needs_human=True`.

    The loss record is seeded through the real writer with two texts, so the sentence the
    refusal must NOT write is a sentence this run's other branch does write (see the node
    below) — a refusal node on an empty record would pass against code that never had the
    branch at all.
    """
    r = _child(tmp_path, "http", mode="infra", seed_loss=True)
    assert r["recoverable"] == ["agent-tts"], r["recoverable"]
    assert r["loss"] and r["loss"]["occurrences"] == 2, r["loss"]
    assert r["ok"] == "needs_human", r
    assert len(r["alerts"]) == 1, r["alerts"]
    alert = r["alerts"][0]
    assert alert["level"] == "error", alert
    assert alert["needs_human"] is True, alert
    assert alert["title"] == "Voice alert channel cannot be recovered: agent-tts", alert
    assert "cannot speak" in alert["body"], alert["body"]
    assert not [a for a in r["alerts"] if a["title"].startswith("Recovered:")], r["alerts"]
    assert "said again" not in alert["body"], alert["body"]
    assert r["spawned"] == [], f"a speech-refused recovery still dispatched speech: {r['spawned']}"


def test_a_recovery_confirmed_by_speech_is_still_said_as_recovered(tmp_path):
    """The positive control for the node above, and the pin that the measurement reaches the
    alert body rather than only the log.

    Same scenario with audio over both floors: `Recovered: agent-tts`, no needs_human alert,
    the re-speak dispatched, and the body carrying the measured seconds and rms. Without this
    arm the refusal node above would pass on a `_recover_infra` whose True branch had gone
    missing too.
    """
    cal = _calib()["configured_voice"]
    r = _child(tmp_path, "good", mode="infra", seed_loss=True)
    assert r["ok"] == "recovered", r["detail"]
    assert len(r["alerts"]) == 1, r["alerts"]
    alert = r["alerts"][0]
    assert alert["level"] == "info", alert
    assert alert["title"] == "Recovered: agent-tts", alert
    assert "needs_human" not in alert, alert
    assert f"{cal['seconds']:.2f}s" in alert["body"], alert["body"]
    assert "said again" in alert["body"], alert["body"]
    assert len(r["spawned"]) == 1, r["spawned"]


# ── Calibration against the real service ───────────────────────────────────

@pytest.mark.live_vault
def test_the_configured_voice_clears_both_floors_on_the_live_synthesiser():
    """The calibration node: every stub above is shaped like what this returns.

    Marked `live_vault` because it needs `agent-tts` up, which CI has no right to assume. It
    exists so the floors cannot drift into fiction: this is the call that measured 2.56 s at
    rms 2250 on 2026-10-06, and if the configured voice ever stops clearing 0.5 s / 200 rms
    this node fails before a recovery verdict does. It synthesises about 2.5 s of audio and
    plays nothing.
    """
    speak = _guardian_sibling("speak")
    cfg = speak.load_config(Path.home() / ".local/state/lloyd-guardian")
    seconds, rms = speak.confirm_speech(cfg)
    assert cfg["voice"].startswith("clone:"), cfg["voice"]
    assert seconds >= speak.PROBE_MIN_SECONDS and rms >= speak.PROBE_MIN_RMS, (
        f"the configured voice cleared only {seconds:.2f}s at rms={rms}")
