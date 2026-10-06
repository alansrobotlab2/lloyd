"""#2300 — a Base model refuses a built-in voice with a 400 and stops advertising one.

The defect, re-measured on the live service on 2026-10-06: `voice: "Vivian"` and
`voice: "Nope"` both answered `500` in ~1 ms with body
`{"error":"processing_error","type":"server_error", ... "does not support
generate_custom_voice"}` while `/health` said
`...Qwen3-TTS-12Hz-1.7B-Base","ready":true`, and `GET /v1/voices` advertised all nine
stock speakers. A 500 typed `server_error` is an *outage* to any alert path that reads
it, which is how a capability mismatch came to be read as one. Worse than the 500: the
same request with `"stream": true, "response_format": "pcm"` answered `200` with **zero
bytes** (`http_code=200 time=0.009498s size=0`, against `200 0.881649s 65280` for
`clone:dave_cullen`), because the streaming branch calls
`backend.generate_speech_streaming` directly and never enters `generate_speech`, so
`ValueError: model with tts_model_type=base does not support
stream_generate_custom_voice` is raised *inside the generator* — after the 200 headers
are already on the wire. A silent empty stream reads as "worked".

So the guard sits where the route has not yet chosen a generation path
(`_capability_rejection`, above the `if request.stream:` branch), and
`test_the_guard_runs_before_both_generation_calls_in_the_route` pins that ordering out
of the patched source instead of trusting it.

Why the patch text is the artefact: `agent-services/services/tts/qwen3-tts/` is an
untracked vendored clone, so an edit inside it yields no diff and no gate artefact, and
`qwen3-tts-local.patch` is the only tracked thing this change can touch. The patch is
the source of truth and the clone's working tree is one human step behind it until
SETUP.md's in-place upgrade runs, so no node here talks to `:8090`. Two mechanisms, in
that order:

  A. `_lift_block` + `exec` — the helpers are lifted out of the patch's own added lines
     and executed, so the code under review is the code the patch ships, and these
     nodes run on a machine with no clone at all.
  B. `_patched_tree` — the pristine clone (`git archive HEAD` at the pinned commit) is
     reconstructed under tmp and the tracked patch is `git apply`-ed to it; the patched
     router is imported and driven through FastAPI's `TestClient` against a fake
     backend. That exercises the real route — status codes, the branch taken, and which
     synthesis calls are never made — with no server. It skips when the clone is not on
     this machine.

The clauses and the nodes that pin them:

  1. non-`clone:` voice on a base model → 400, never 500, never a started stream, on
     both routes, before either generation call
     → `test_a_stock_voice_is_refused_on_both_routes_without_reaching_synthesis`
     → `test_the_guard_runs_before_both_generation_calls_in_the_route`
     → `test_a_built_in_voice_is_refused_by_the_patched_code_on_a_base_model`
  2. the 400 names the reason in caller-actionable terms, typed invalid-request, and
     one log line names the voice and the loaded type
     → `test_the_400_body_is_typed_invalid_request_and_names_the_fix`
     → `test_the_refusal_logs_one_line_naming_the_voice_and_the_loaded_type`
  3. `GET /v1/voices` on a base backend lists no non-`clone:` id in any of its three
     branches, and voice-library profiles keep the `clone:` prefix
     → `test_a_base_backend_lists_no_built_in_voice_in_any_of_the_three_branches`
     → `test_the_voices_filter_keeps_clone_profiles_and_drops_the_rest_by_type`
  4. the rule keys on the loaded model type, not the voice name: a CustomVoice backend
     still lists and still synthesizes the built-ins, and the `clone:` path is unchanged
     → `test_a_customvoice_backend_still_lists_the_builtins`
     → `test_a_customvoice_backend_still_synthesizes_a_built_in_on_both_routes`
     → `test_the_clone_path_is_untouched_on_a_base_backend`

Two asymmetries are deliberate and both are pinned, because a reader will otherwise
"fix" one of them into a bug. A *refusal* fires only for a type known to fail
(`base`), so a backend that reports `unknown` still synthesizes: falsely refusing a
request would break a deployment this file has no business judging. A *listing*
requires a type known to serve built-ins (`customvoice`), so the backend-not-loaded
branch advertises only the clone profiles: falsely advertising a voice is the very
defect being fixed, and those profiles are files on disk rather than a property of a
model that failed to load.

The `default_model` and `_customvoice_model_key` questions — whether to restore a
CustomVoice swap so built-ins work — are ruled out of scope here; nothing below asks
which model should be default, only that the server must not claim or crash on a voice
the loaded one cannot speak.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TTS = Path("agent-services") / "services" / "tts"
PATCH = REPO / TTS / "qwen3-tts-local.patch"
PIN_FILE = REPO / TTS / "qwen3-tts-upstream-commit.txt"
ROUTER = "api/routers/openai_compatible.py"

#: The nine speakers `config.yaml`'s `voices:` section hands to
#: `get_supported_voices()`, and the six OpenAI aliases `list_voices` appends for a
#: non-base model. Every one of them was advertisable and unusable on this box.
STOCK_SPEAKERS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden",
                  "Ono_Anna", "Sohee"]
OPENAI_ALIAS_IDS = ["alloy", "echo", "fable", "nova", "onyx", "shimmer"]

#: The two model names the live probes use (`agent-services/guardian/speak.py` sends
#: `tts-1`, the voice-mode client sends `qwen3-tts`), and the two shapes of the same
#: unusable voice: one the server advertised, one nobody ever listed.
MODELS = ["qwen3-tts", "tts-1"]
REFUSED_VOICES = ["Vivian", "Ryan", "Nope"]

#: The URLs the live service answers, rebuilt under the same `/v1` prefix `api/main.py`
#: uses. Spelled out rather than discovered by suffix because the router also declares
#: `/audio/voices`, which is a different route.
SPEECH_URL = "/v1/audio/speech"
VOICES_URL = "/v1/voices"


# ── A. reading the patch, and executing what it adds ─────────────────────────

def _file_section(patch_text: str, wanted: str) -> list[str]:
    """The raw diff lines for one file's section, keyed off its `+++` line.

    Keyed off `+++` rather than `diff --git` because this patch carries the clone's
    `i/`/`w/` mnemonic prefixes (`diff.mnemonicPrefix`), not `a/`/`b/`, and stripping
    one path component is exactly what `git apply -p1` does.
    """
    out: list[str] = []
    current: str | None = None
    keep = False
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            keep = False
            current = None
        elif line.startswith("+++ "):
            path = line[4:].strip()
            current = path.split("/", 1)[1] if "/" in path else path
            keep = current == wanted
        elif keep:
            out.append(line)
    assert current is not None, f"the tracked patch has no +++ line for {wanted!r}"
    return out


def _patch_added(patch_text: str, wanted: str) -> list[str]:
    """Every line the patch adds to `wanted`, `+` stripped, in order."""
    return [ln[1:] for ln in _file_section(patch_text, wanted)
            if ln.startswith("+") and not ln.startswith("+++")]


def _hunk_postimages(section: list[str]) -> list[list[str]]:
    """Each hunk's post-image: its context and added lines, as contiguous file content.

    Needed because the added lines alone are every hunk of the file concatenated, so a
    block read out of them would run straight on into a later hunk's lines.
    """
    hunks: list[list[str]] = []
    current: list[str] | None = None
    for ln in section:
        if ln.startswith("@@"):
            current = []
            hunks.append(current)
        elif current is not None and (ln.startswith("+") or ln.startswith(" ")):
            current.append(ln[1:])
    return hunks


def _lift_block(postimages: list[list[str]], marker: str) -> str:
    """One module-level block of the patched file: from `marker` until the next
    top-level statement, taken whole out of the hunk that contains the marker.

    "Until the next top-level statement" is decided by compiling: an indent-zero line
    ends the block only once what has been collected so far is valid Python, which is
    the same rule the parser uses and which a brace counter would get wrong. Asserted
    unique across hunks, so a helper that moved into a second place is a failure here
    rather than whichever copy a search happened to find first.
    """
    hits = [h for h in postimages if any(ln.startswith(marker) for ln in h)]
    assert len(hits) == 1, (
        f"{marker!r} starts in {len(hits)} hunks of {ROUTER}, expected exactly 1 — the "
        "capability guard is not in the patch where this file says it is")
    hunk = hits[0]
    start = next(i for i, ln in enumerate(hunk) if ln.startswith(marker))
    body: list[str] = []
    for ln in hunk[start:]:
        if body and ln.strip() and not ln.startswith((" ", "\t")):
            try:
                compile("\n".join(body) + "\n", "<lift>", "exec")
            except SyntaxError:
                body.append(ln)
                continue
            break
        body.append(ln)
    while body and body[-1].strip() == "":
        body.pop()
    assert body, f"nothing lifted at {marker!r}"
    src = "\n".join(body) + "\n"
    compile(src, "<lift>", "exec")
    return src


class _RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def warning(self, msg, *args) -> None:
        self.warnings.append(str(msg) % args if args else str(msg))

    def info(self, msg, *args) -> None:
        self.infos.append(str(msg) % args if args else str(msg))

    def __getattr__(self, name):          # debug/exception/etc: record nothing
        return lambda *a, **k: None


class _HTTPException(Exception):
    """Stand-in carrying the two attributes FastAPI's own class has and the route uses."""

    def __init__(self, status_code: int = 500, detail=None) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@pytest.fixture(scope="module")
def capability():
    """The patch's added capability code, executed in a namespace it can run in.

    `logger` is a recorder (clause 2 asserts on the line the code emits, not on a
    literal this file retypes), `HTTPException` is a stand-in, and `List` is bound
    because the filter's annotations are evaluated at def time.
    """
    from typing import List

    text = PATCH.read_text(encoding="utf-8")
    added = _patch_added(text, ROUTER)
    for name in ("BASE_MODEL_TYPE", "BUILTIN_SPEAKER_MODEL_TYPES",
                 "def _serves_builtin_speakers", "def _voice_capability_rejection",
                 "def _visible_builtin_voices", "def _model_type_of"):
        assert any(a.lstrip().startswith(name) for a in added), (
            f"the tracked patch adds no {name!r}: the capability guard is not in the "
            "patch, so the live service still answers a stock voice with 500")

    postimages = _hunk_postimages(_file_section(text, ROUTER))
    # One block per name, because a single lift from the first constant would stop at
    # the second (an indent-zero line ends a block the moment what precedes compiles).
    consts = "".join([
        _lift_block(postimages, "BASE_MODEL_TYPE = "),
        _lift_block(postimages, "UNKNOWN_MODEL_TYPE = "),
        _lift_block(postimages, "BUILTIN_SPEAKER_MODEL_TYPES = "),
    ])
    funcs = "".join([
        _lift_block(postimages, "def _serves_builtin_speakers"),
        _lift_block(postimages, "def _model_type_of"),
        _lift_block(postimages, "def _voice_capability_rejection"),
        _lift_block(postimages, "def _visible_builtin_voices"),
    ])
    log = _RecordingLogger()
    ns: dict = {"logger": log, "HTTPException": _HTTPException, "List": List}
    exec(consts + funcs, ns)                      # noqa: S102 — the patch's own source
    ns["logger"] = log
    return ns


def test_a_built_in_voice_is_refused_by_the_patched_code_on_a_base_model(capability):
    """Clause 1 at the code level: a 400, keyed on the model type, on a base model.

    `Nope` matters as much as `Vivian`: the refusal cannot be a list of names this file
    was told about, because the whole failure was a name that looked valid.
    """
    reject = capability["_voice_capability_rejection"]
    for voice in REFUSED_VOICES:
        exc = reject(voice, "base")
        assert exc is not None, f"{voice} was not refused on a base model"
        assert exc.status_code == 400, f"{voice}: {exc.status_code}, expected 400"
    assert reject("clone:dave_cullen", "base") is None, (
        "the clone path is the one thing a base model does serve")
    assert reject("Vivian", "customvoice") is None
    assert reject("Vivian", "unknown") is None, (
        "a backend that does not report its type must not be falsely refused")
    assert capability["BASE_MODEL_TYPE"] == "base"
    assert capability["BUILTIN_SPEAKER_MODEL_TYPES"] == ("customvoice",)


def test_the_refusal_logs_one_line_naming_the_voice_and_the_loaded_type(capability):
    """Clause 2's log half: exactly one WARNING per rejection, naming both facts."""
    log = capability["logger"]
    reject = capability["_voice_capability_rejection"]
    log.warnings.clear()
    reject("Nope", "base")
    assert len(log.warnings) == 1, log.warnings
    line = log.warnings[0]
    assert "Nope" in line and "base" in line, line

    log.warnings.clear()
    reject("clone:dave_cullen", "base")
    reject("Vivian", "customvoice")
    assert log.warnings == [], "an accepted request must not look like a rejection"


def test_the_voices_filter_keeps_clone_profiles_and_drops_the_rest_by_type(capability):
    """Clause 3/4 at the code level: the key is the loaded model type, not the id."""
    listing = ([{"id": s, "name": s} for s in STOCK_SPEAKERS]
               + [{"id": a, "name": a} for a in OPENAI_ALIAS_IDS]
               + [{"id": "clone:dave_cullen", "name": "clone:dave_cullen"}])
    visible = capability["_visible_builtin_voices"]

    on_base = visible(listing, "base")
    assert [v["id"] for v in on_base] == ["clone:dave_cullen"], on_base
    assert visible(listing, "customvoice") == listing, "a CustomVoice deploy is unchanged"
    assert [v["id"] for v in visible(listing, "unknown")] == ["clone:dave_cullen"], (
        "an unconfirmed capability is not advertised — see the module docstring")
    assert visible([], "base") == []


# ── B. the real route, driven from the reconstructed patched tree ────────────

def _clone_dir() -> Path | None:
    """The vendored clone the patch applies to, or None when this machine has none.

    Deliberately not `REPO / TTS / "qwen3-tts"`: that tree is untracked, so a linked
    worktree — which is where this file runs under a gate — does not contain it, and a
    skip that quietly proves nothing is the failure mode #2030 left behind. The checkout
    that holds it is found through git's common dir, and the pin assertion in
    `patched_tree` still refuses to grade the patch against a different upstream.
    """
    roots = [REPO]
    probe = subprocess.run(["git", "-C", str(REPO), "rev-parse",
                            "--path-format=absolute", "--git-common-dir"],
                           capture_output=True, text=True)
    if probe.returncode == 0 and probe.stdout.strip():
        roots.append(Path(probe.stdout.strip()).parent)
    for root in roots:
        candidate = root / TTS / "qwen3-tts"
        if (candidate / ".git").exists():
            return candidate
    return None


@pytest.fixture(scope="module")
def patched_tree(tmp_path_factory):
    """Pristine pinned clone + the tracked patch, under tmp — the tree being reviewed.

    Reconstructed from `git archive HEAD` rather than read from the clone's working
    tree on purpose: the working tree carries whatever was last applied by hand, while
    the tracked `.patch` is what this round changed and what the human step applies. A
    clone at any commit other than the pin is a hard failure, not a skip — grading a
    diff against the wrong upstream is worse than not grading it.
    """
    clone = _clone_dir()
    if clone is None:
        pytest.skip("no qwen3-tts clone on this machine, so the tracked patch has no "
                    "upstream here to apply against")
    head = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    pin = PIN_FILE.read_text().split()[0]
    assert head.startswith(pin), (
        f"the clone at {clone} is at {head[:8]}, not the pinned {pin} named in "
        f"{PIN_FILE.name}; the patch is a diff against the pinned commit")

    tree = tmp_path_factory.mktemp("voice-capability-clone")
    archive = subprocess.run(["git", "-C", str(clone), "archive", "HEAD"],
                             capture_output=True)
    assert archive.returncode == 0, archive.stderr.decode()[:300]
    blob = tmp_path_factory.mktemp("archive") / "upstream.tar"
    blob.write_bytes(archive.stdout)
    with tarfile.open(blob) as tar:
        tar.extractall(tree, filter="data")

    for args in (["-p1", "--check"], ["-p1"]):
        applied = subprocess.run(["git", "-C", str(tree), "apply", *args, str(PATCH)],
                                 capture_output=True, text=True)
        assert applied.returncode == 0, (
            f"`git apply {' '.join(args)}` of the tracked patch failed: "
            f"{applied.stderr[:600]}")
    return tree


@pytest.fixture(scope="module")
def router_module(patched_tree):
    """The patched `api.routers.openai_compatible`, imported for real.

    Imported under its own package name from `patched_tree`; the path entry and the
    modules are removed afterwards so no other test can pick up the vendored package.
    """
    sys.path.insert(0, str(patched_tree))
    for name in [m for m in sys.modules if m == "api" or m.startswith("api.")]:
        sys.modules.pop(name, None)
    import importlib
    try:
        yield importlib.import_module("api.routers.openai_compatible")
    finally:
        sys.path.remove(str(patched_tree))
        for name in [m for m in sys.modules if m == "api" or m.startswith("api.")]:
            sys.modules.pop(name, None)


class FakeBackend:
    """Records every synthesis the route attempts, so "never reached generation" is a
    measurement and not an inference."""

    def __init__(self, model_type="base", speakers=None, clone_capable=True):
        self.model_type = model_type
        self.speakers = STOCK_SPEAKERS if speakers is None else list(speakers)
        self.clone_capable = clone_capable
        self.attempts: list[str] = []

    def get_model_type(self) -> str:
        return self.model_type

    def get_supported_voices(self):
        return list(self.speakers)

    def get_supported_languages(self):
        return ["English"]

    def is_custom_voice(self, speaker) -> bool:
        return False

    def supports_voice_cloning(self) -> bool:
        return self.clone_capable

    async def generate_speech_streaming(self, **kwargs):
        """Yields what the real backend yields: an audio array and its sample rate.

        Yielding encoded bytes instead fails inside `encode_audio` with `shape ()`,
        which is a fault in this fake and not in the route — the distinction is worth
        the comment, because the first run of this file blamed the router for it.
        """
        self.attempts.append("generate_speech_streaming")
        import numpy as np
        yield np.full(240, 0.1, dtype="float32"), 24000

    async def generate_voice_clone(self, **kwargs):
        self.attempts.append("generate_voice_clone")
        import numpy as np
        return np.full(480, 0.1, dtype="float32"), 24000


@pytest.fixture
def drive(router_module, tmp_path):
    """A TestClient over the patched router, with the service swapped for fakes.

    `get_tts_backend`, `generate_speech`, `_load_voice_profile` and the profiles
    directory are the four seams the route reaches outside itself; each is restored
    afterwards. PCM output needs no encoder, so the fake backend's audio is encodable
    whatever optional codecs this venv has.
    """
    import numpy as np
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    originals = {k: getattr(router_module, k) for k in
                 ("get_tts_backend", "generate_speech", "_load_voice_profile",
                  "VOICE_LIBRARY_DIR", "logger")}

    def build(backend=None, *, load_error=None, voices=("dave_cullen",)):
        log = _RecordingLogger()
        router_module.logger = log
        profiles = tmp_path / "voice_library" / "profiles"
        for name in voices:
            profile = profiles / name
            profile.mkdir(parents=True, exist_ok=True)
            (profile / "meta.json").write_text(json.dumps(
                {"name": name, "ref_audio_filename": "ref.wav"}))

        async def get_tts_backend():
            if load_error is not None:
                raise load_error
            return backend

        async def generate_speech(**kwargs):
            backend.attempts.append("generate_speech")
            return np.full(480, 0.1, dtype="float32"), 24000

        def load_profile(name):
            backend.attempts.append("load_profile")
            raise ValueError(f"no profile named {name}")

        router_module.get_tts_backend = get_tts_backend
        router_module.generate_speech = generate_speech
        router_module._load_voice_profile = load_profile
        router_module.VOICE_LIBRARY_DIR = tmp_path / "voice_library"

        app = FastAPI()
        app.include_router(router_module.router, prefix="/v1")
        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert {SPEECH_URL, VOICES_URL} <= paths, (
            f"the patched router does not declare the live URLs: {sorted(paths)}")
        return TestClient(app), log

    yield build
    for key, value in originals.items():
        setattr(router_module, key, value)


def _speech(client, voice, *, model="qwen3-tts", stream=False, fmt="wav"):
    return client.post(SPEECH_URL, json={"model": model, "voice": voice,
                                         "input": "Hello testing", "stream": stream,
                                         "response_format": fmt})


def test_a_stock_voice_is_refused_on_both_routes_without_reaching_synthesis(drive):
    """Clause 1 at the boundary: 400 on both routes, and no generation attempted.

    `stream=True` is the case that used to answer `200` with zero bytes, so the
    assertion is on the status *and* the body: a started-but-empty stream must fail this
    node, not merely a wrong code.
    """
    backend = FakeBackend("base")
    client, log = drive(backend)
    for voice in REFUSED_VOICES:
        for model in MODELS:
            for stream, fmt in ((False, "wav"), (True, "pcm")):
                r = _speech(client, voice, model=model, stream=stream, fmt=fmt)
                assert r.status_code == 400, (
                    f"voice={voice} model={model} stream={stream} -> {r.status_code} "
                    f"{r.text[:200]!r}; expected 400, never 500 and never a 200 stream")
                assert r.content, "the refusal arrived as an empty body"
    assert backend.attempts == [], (
        f"a refused voice still reached generation: {backend.attempts}")


def test_the_400_body_is_typed_invalid_request_and_names_the_fix(drive):
    """Clause 2 at the boundary: the caller can act on it, and it is not an outage.

    The live body said `"error":"processing_error","type":"server_error"` — the typing
    that made an alert path read a capability mismatch as a downed service.
    """
    client, log = drive(FakeBackend("base"))
    r = _speech(client, "Vivian")
    detail = r.json()["detail"]
    assert detail["type"] == "invalid_request_error", detail
    message = detail["message"]
    for token in ("base", "CustomVoice", "Vivian", "/v1/voices"):
        assert token in message, f"{token!r} missing from the reason: {message}"
    assert len(log.warnings) == 1 and "Vivian" in log.warnings[0], log.warnings


def test_a_base_backend_lists_no_built_in_voice_in_any_of_the_three_branches(drive):
    """Clause 3, branch by branch, through the real endpoint.

    The three branches are the speakers list, the `default_voices` fallback reached when
    a backend reports no speakers, and the `except` that answers when the backend will
    not load; a filter bolted onto the first alone would leave nine stock names in the
    other two. The clone profile is written into a tmp profiles dir, so "only `clone:`
    ids" is measured against a listing that really has one to keep.
    """
    client, _ = drive(FakeBackend("base", speakers=STOCK_SPEAKERS))
    listed = [v["id"] for v in client.get(VOICES_URL).json()["voices"]]
    assert listed == ["clone:dave_cullen"], f"speakers branch: {listed}"

    client2, _ = drive(FakeBackend("base", speakers=[]))
    listed2 = [v["id"] for v in client2.get(VOICES_URL).json()["voices"]]
    assert listed2 == ["clone:dave_cullen"], f"default_voices fallback: {listed2}"

    client3, _ = drive(load_error=RuntimeError("backend will not load"))
    listed3 = [v["id"] for v in client3.get(VOICES_URL).json()["voices"]]
    assert listed3 == ["clone:dave_cullen"], f"backend-not-loaded fallback: {listed3}"

    for listed in (listed, listed2, listed3):
        for leak in STOCK_SPEAKERS + OPENAI_ALIAS_IDS:
            assert leak not in listed, f"{leak} still advertised: {listed}"


def test_a_customvoice_backend_still_lists_the_builtins(drive):
    """Clause 4: the filter's key is the model type, so a CustomVoice deploy is unchanged."""
    client, _ = drive(FakeBackend("customvoice"))
    listed = [v["id"] for v in client.get(VOICES_URL).json()["voices"]]
    for expected in STOCK_SPEAKERS + OPENAI_ALIAS_IDS + ["clone:dave_cullen"]:
        assert expected in listed, f"{expected} dropped from a CustomVoice listing"


def test_a_customvoice_backend_still_synthesizes_a_built_in_on_both_routes(drive):
    """Clause 4's second half: the same guard accepts what a CustomVoice model can do."""
    backend = FakeBackend("customvoice")
    client, log = drive(backend)
    r = _speech(client, "Vivian")
    assert r.status_code == 200, r.text[:300]
    assert len(r.content) > 0
    assert "generate_speech" in backend.attempts

    backend2 = FakeBackend("customvoice")
    client2, _ = drive(backend2)
    r2 = _speech(client2, "Ryan", stream=True, fmt="pcm")
    assert r2.status_code == 200, r2.text[:300]
    assert len(r2.content) > 0, "an accepted stream must actually carry bytes"
    assert "generate_speech_streaming" in backend2.attempts
    assert log.warnings == [], "nothing here was a refusal"


def test_the_clone_path_is_untouched_on_a_base_backend(drive):
    """Clause 4's `clone:` half, on the deployment that actually runs this way.

    The profile loader is made to fail, so the expected answer is the route's own 404
    `profile_not_found` — proof the request travelled the clone branch rather than
    being turned away by the capability guard, which would have answered 400.
    """
    backend = FakeBackend("base")
    client, log = drive(backend)
    r = _speech(client, "clone:dave_cullen")
    assert r.status_code == 404, f"{r.status_code} {r.text[:200]!r}"
    assert r.json()["detail"]["error"] == "profile_not_found", r.text[:200]
    assert "load_profile" in backend.attempts
    assert log.warnings == [], "the guard must not fire on a clone: voice"


# ── the ordering the whole fix turns on ─────────────────────────────────────

def test_the_guard_runs_before_both_generation_calls_in_the_route(patched_tree):
    """Clause 1's "before", read out of the patched source rather than asserted of it.

    A guard anywhere below `if request.stream:` cannot cover the streaming route, which
    is the path the item's stated mechanism (`before generate_speech`) misses. So this
    walks the patched file and requires: exactly one `_capability_rejection` binding, its
    line before every `generate_speech`/`generate_speech_streaming` call inside
    `create_speech`, and its indentation equal to the `if request.stream:` statement's —
    a sibling above that branch, not a statement nested inside something it cannot cover.
    """
    src = (patched_tree / ROUTER).read_text(encoding="utf-8")
    tree = ast.parse(src)
    route = next((n for n in ast.walk(tree)
                  if isinstance(n, ast.AsyncFunctionDef) and n.name == "create_speech"),
                 None)
    assert route is not None, "no create_speech in the patched router"

    guards = [n for n in ast.walk(route) if isinstance(n, ast.Assign)
              and any(getattr(t, "id", "") == "_capability_rejection" for t in n.targets)]
    assert len(guards) == 1, f"{len(guards)} capability guards, expected exactly 1"
    guard = guards[0]

    calls = [n for n in ast.walk(route) if isinstance(n, ast.Call)
             and (getattr(n.func, "attr", "") or getattr(n.func, "id", ""))
             in ("generate_speech", "generate_speech_streaming")]
    streaming = [c for c in calls
                 if getattr(c.func, "attr", "") == "generate_speech_streaming"]
    assert streaming, "the patched route no longer has a streaming call for the guard to precede"
    assert len(calls) >= 2, f"only {len(calls)} generation call(s) found"
    late = [f"{getattr(c.func, 'attr', '') or c.func.id}:{c.lineno}"
            for c in calls if c.lineno <= guard.lineno]
    assert not late, (
        f"the guard is on line {guard.lineno} but generation happens at {late} — a "
        "request could synthesize before anything was refused")

    stream_branches = [n for n in ast.walk(route) if isinstance(n, ast.If)
                       and getattr(getattr(n.test, "value", None), "id", "") == "request"
                       and getattr(n.test, "attr", "") == "stream"]
    assert stream_branches, "no `if request.stream:` in the patched route"
    first_stream = min(stream_branches, key=lambda n: n.lineno)
    assert guard.lineno < first_stream.lineno, (
        f"guard at {guard.lineno} is not above the stream branch at "
        f"{first_stream.lineno}")
    assert guard.col_offset == first_stream.col_offset, (
        f"guard indent {guard.col_offset} != stream-branch indent "
        f"{first_stream.col_offset}: it is nested where it cannot cover both routes")
