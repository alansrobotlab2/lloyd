"""Every voice model the shipped config reads must arrive with the checkout.

#3252 is about one file: `config.yaml` has said `turn_detection.backend:
turn-1-mini` since `fed5ad3f` (2026-10-09) and pointed at
`agent-services/models/turn-1-mini/onnx/turn-1-mini.step.onnx`, which lived on
exactly one disk. `fetch-voice-models.sh --check` exited 0 on a tree without it,
the tests skipped it as a machine fact, and a rebuilt checkout died in
`WorkerManager.__init__` before supervisor could start `lloyd-agent-worker`. The
first node here is that file's witness: tracked, right size, and still matched by
the `models/` ignore rule it was force-added past — both rc=0, because bare
`check-ignore` returns 1 for a tracked path and would read green on the tree this
clause refuses: one where the rule was deleted instead of bypassed.

The rest is the class, because turn-1-mini was not special. Six model paths are
named outside comments in `config.yaml` — two ASR directories, the end-of-turn
detector, the two wake-word directories' contents and the acoustic stop word — and
exactly one of them had no restore route. The guard enumerates them from the
config rather than from a list a person maintains, so a seventh model wired into
config without a route fails the tests rung instead of the worker at boot. Both
sides of that guard are pinned in the file: an unrestorable config path must be
reported, and a `FILES` row that is malformed or claims zero bytes must be
reported. A guard whose reader or classifier quietly returns nothing is the same
`--check`-exits-0 bug in a new costume.

Two ways to satisfy a node like this are refused by design, and both are
pinned here rather than trusted to prose: catching the boot `RuntimeError` so the
worker starts deaf — `test_the_worker_still_refuses_to_boot_without_the_detector`
walks the real `WorkerManager.__init__` for a handler and then proves the walk can
see one — and adding a `FILES` row with a byte count or a URL nobody verified,
which is why turn-1-mini is tracked rather than fetched, its upstream URL being
unverified at filing. `test_the_tracked_detector_loads_so_the_boot_path_has_a_witness`
then crosses the boundary the whole item is about: the bytes a clone arrives with,
through the function the worker calls, into a session that scores audio.

The boot-refusal node lives here rather than in
`tests/test_voice_wake_and_turn.py`, where the item's clause 4 names it, because
that file is one of the five closed machine-fact populations in
`tests/test_skip_inventory.py` and its
`test_no_machine_fact_skip_was_deleted_to_buy_headroom` raises on any diff that
touches one of them — measured at this round's first commit: 2 failed, naming
`['tests/test_voice_wake_and_turn.py']`, and 21 passed once the file was returned to
its base text. So no node can be added there while a diff exists, and it is the
assertions, not the file path, that keep the boot loud.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.yaml"
FETCH = ROOT / "agent-services" / "setup" / "fetch-voice-models.sh"

#: The detector `livekit.turn_detection.model_path` names, and the two licence files
#: that travel with it: the model's NOTICE is Apache-2.0 §4(d) — "Redistributed copies
#: … must retain this notice" — so a silent blob commit is a licence violation on every
#: clone, on top of being the thing that makes it a licence at all.
DETECTOR = "agent-services/models/turn-1-mini/onnx/turn-1-mini.step.onnx"
DETECTOR_BYTES = 21825125
LICENCE_FILES = (
    ("agent-services/models/turn-1-mini/LICENSE", 11357),
    ("agent-services/models/turn-1-mini/NOTICE", 1860),
)

MODEL_PATH = "agent-services/models/"
#: Rows in the fetch table at filing. Pinning the count is the cheap way to notice a row
#: disappearing: the script's own loop skips an empty element without a word, so a row
#: removed in prose is eight rows that quietly stopped covering a model.
FETCH_ROW_COUNT = 8


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True, check=False)


def _tracked_paths() -> set[str]:
    """Everything in the index. The index, not the working tree: a file that exists on
    this disk because somebody hand-copied it, and a file a clone receives, are two
    different claims, and only the second one is what a rebuilt tree gets."""
    r = _git("ls-files", "-z")
    assert r.returncode == 0, r.stderr
    return {p for p in r.stdout.split("\0") if p}


def _model_refs(text: str) -> list[tuple[int, str]]:
    """(line, path) for every `agent-services/models/…` reference outside a comment.

    Read off the text rather than parsed as YAML: the paths sit under keys this repo does
    not own the schema of (`model_dir`, `engine_dir`, `model_path`, `stop.model`), and a
    comment naming a path is a hint rather than a boot requirement — `smart-turn`, the
    fallback end-of-turn backend, is only ever named in one.
    """
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("#"):
            continue
        for token in line.split():
            token = token.strip().strip("'\"").rstrip(",")
            if token.startswith(MODEL_PATH) and token != MODEL_PATH:
                out.append((n, token.rstrip("/")))
    return out


def _files_table(fetch_text: str) -> str:
    start = fetch_text.index("FILES=(") + len("FILES=(")
    return fetch_text[start:fetch_text.index("\n)", start)]


def _file_rows(fetch_text: str) -> tuple[dict[str, int], list[str]]:
    """The `FILES` table as (installed path -> expected bytes), plus rows that no-parse.

    A row is `"dir|file|bytes|url"`, installed at `agent-services/models/dir/file`, so
    that join is the key — the same one the script's own `--check` loop makes. The script
    drops an unparseable row without printing anything (an empty element is `continue`d,
    then `IFS='|' read` fills short and the size test compares against nothing), so a
    typo'd row removes a model from the size check in silence. Returning the malformed
    rows instead of dropping them is the classifier's failure control.
    """
    rows: dict[str, int] = {}
    malformed: list[str] = []
    for raw in _files_table(fetch_text).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.strip('"').split("|")
        if len(parts) != 4 or not parts[2].isdigit():
            malformed.append(line)
            continue
        rows[f"{MODEL_PATH}{parts[0]}/{parts[1]}"] = int(parts[2])
    return rows, malformed


def _has_restore_route(ref: str, tracked: set[str], rows: dict[str, int]) -> bool:
    """Whether a config reference is covered by the clone or by the fetch table.

    Prefix-matched on purpose: half of what `config.yaml` names is a directory
    (`models_dir`, `engine_dir`) and half a file (`model_path`, `stop.model`), and what
    the worker needs in the first case is a directory it can read models out of. The
    narrowness is not lost — a file-level reference is compared exactly, and a new model
    whose own file is in neither place still has no route.
    """
    return (ref in tracked or rows.get(ref, 0) > 0
            or any(p == ref or p.startswith(ref + "/") for p in tracked)
            or any(p.startswith(ref + "/") for p, size in rows.items() if size > 0))


def _missing_restore_routes(config_text: str, fetch_text: str,
                           tracked: set[str] | None = None) -> list[str]:
    """Why each config-named model file does or does not arrive with the clone.

    Over text, not over the two live files, for the reason every check in this
    neighbourhood is one: the control has to be able to hand it a tree that is broken and
    watch it say so.
    """
    tracked = _tracked_paths() if tracked is None else tracked
    rows, malformed = _file_rows(fetch_text)
    problems = [f"fetch-voice-models.sh FILES has an unparseable row: {row!r}"
                for row in malformed]
    problems += [f"fetch-voice-models.sh FILES row {path!r} claims {size} bytes"
                 for path, size in rows.items() if size <= 0]
    for n, path in _model_refs(config_text):
        if _has_restore_route(path, tracked, rows):
            continue
        problems.append(
            f"config.yaml:{n} names {path!r}, which is neither tracked in git nor a "
            "size-checked row in fetch-voice-models.sh — a rebuilt checkout boots "
            "without it and nothing in CI notices (#3252). Give it a route: `git add -f` "
            "it (with its LICENSE and NOTICE if it has any), or add a FILES row with a "
            "verified byte count and a URL that actually serves it")
    return problems


def _swap_files_table(fetch_text: str, new_table_text: str) -> str:
    """`fetch_text` with its `FILES=(…)` block replaced, for the controls below."""
    start = fetch_text.index("FILES=(")
    end = fetch_text.index("\n)", start) + len("\n)")
    return fetch_text[:start] + new_table_text + fetch_text[end:]


#: A config naming a model that is in neither place — the shape of #3252 in four lines,
#: for the control that says the reader can fail.
UNRESTORABLE_CONFIG = """\
livekit:
  turn_detection:
    enabled: true
    backend: brand-new-model
    model_path: agent-services/models/brand-new-model/model.onnx
"""

#: A row whose byte field is not a number. The script skips it and `--check` still exits
#: 0, which is the bug class the item called the important half of the fix.
UNPARSEABLE_FILES = """FILES=(
  "smart-turn|smart-turn-v3.2-cpu.onnx|8679182|HF/a"|
  "brand-new-model|model.onnx|to-be-measured|HF/b"
)"""

#: The same row syntactically fine, with a byte count nobody measured: accepted by the
#: script, and the size check it enables is a check against zero.
ZERO_BYTE_FILES = """FILES=(
  "brand-new-model|model.onnx|0|HF/b"
)"""


def test_the_detector_the_shipped_config_names_arrives_with_the_clone():
    """Clause 1: the fp32 the config boots is tracked at its size, licence and all.

    Both witnesses per path, both rc=0. `git check-ignore --no-index -q` proves the
    unanchored `models/` rule still matches the path, i.e. that "tracked" was not reached
    by negating it in `agent-services/.gitignore` — a file in `spec.DENIED_GLOBS` that
    SETUP.md also tells a reader not to negate — while `git ls-files --error-unmatch`
    proves this file was force-added past it. `--no-index` is what makes the first witness
    mean anything: without it, git reports "not ignored" for a tracked path and the pair
    collapses to one check.
    """
    for path, size in ((DETECTOR, DETECTOR_BYTES), *LICENCE_FILES):
        ignored = _git("check-ignore", "--no-index", "-q", path)
        assert ignored.returncode == 0, (
            f"{path} is no longer matched by an ignore rule, so it became tracked by "
            f"editing agent-services/.gitignore — which the item forbids and the gate "
            f"refuses. The route is `git add -f` past the rule, as the wake models are")
        listed = _git("ls-files", "--error-unmatch", path)
        assert listed.returncode == 0, (
            f"{path} is not tracked, so a fresh clone has no end-of-turn detector and "
            f"lloyd-agent-worker dies in WorkerManager.__init__ before supervisor sees it "
            f"(#3252)")
        on_disk = ROOT / path
        assert on_disk.is_file(), f"{path} is tracked but was not checked out"
        assert on_disk.stat().st_size == size, (
            f"{path} is {on_disk.stat().st_size} bytes, expected {size} — a weight "
            "committed at the wrong size boots and answers wrong, which is worse than "
            "the missing file this item fixed")


def test_every_voice_model_named_in_config_has_a_restore_route():
    """Clause 3: the class guard over `config.yaml`, the index and the fetch table."""
    problems = _missing_restore_routes(
        CONFIG.read_text(encoding="utf-8"), FETCH.read_text(encoding="utf-8"))
    assert not problems, "\n".join(problems)
    refs = _model_refs(CONFIG.read_text(encoding="utf-8"))
    assert len(refs) == 6, (
        f"the config reader found {len(refs)} model paths ({refs}); 6 were priced at "
        f"filing, and a reader that quietly finds fewer makes the guard above vacuous — "
        f"that is the `--check`-exits-0 failure this file exists to refuse")
    paths = {p for _, p in refs}
    assert DETECTOR in paths, (
        f"config.yaml no longer names {DETECTOR} outside a comment, so the node above is "
        f"pinning a file nothing boots — re-check both before editing either")


def test_the_restore_route_guard_fails_on_an_unrestorable_config_path():
    """The reader's control: the #3252 shape, in text, has to be reported."""
    fetch = FETCH.read_text(encoding="utf-8")
    problems = _missing_restore_routes(UNRESTORABLE_CONFIG, fetch)
    assert problems, "the guard reported nothing for a model with no restore route"
    joined = "\n".join(problems)
    assert "brand-new-model/model.onnx" in joined, joined
    assert "git add -f" in joined and "FILES" in joined, (
        f"the failure has to name both routes or the next reader re-derives them: "
        f"{joined}")
    # The shipped config over the same function: no problems. Together the two results
    # are the function's edge rather than the tree's luck.
    assert _missing_restore_routes(CONFIG.read_text(encoding="utf-8"), fetch) == []


def test_the_restore_route_guard_fails_on_an_unparseable_or_unmeasured_files_row():
    """The classifier's control: a row the script itself skips is still a failure."""
    fetch = FETCH.read_text(encoding="utf-8")
    rows, malformed = _file_rows(fetch)
    assert rows and not malformed, (
        f"the live FILES table did not parse cleanly on this tree: {malformed}")
    assert len(rows) == FETCH_ROW_COUNT, (
        f"fetch-voice-models.sh installs {len(rows)} files ({sorted(rows)}), "
        f"{FETCH_ROW_COUNT} at filing. A dropped row means the model it covered has to be "
        f"tracked now — and this node and the class guard would then disagree about "
        f"whether it has a route")
    for table, expect in ((UNPARSEABLE_FILES, "unparseable"),
                          (ZERO_BYTE_FILES, "claims 0 bytes")):
        swapped = _swap_files_table(fetch, table)
        assert swapped != fetch
        problems = _missing_restore_routes(UNRESTORABLE_CONFIG, swapped)
        joined = "\n".join(problems)
        assert expect in joined, f"{table!r} produced {problems!r}"
        assert "brand-new-model" in joined, (
            f"a row the table cannot carry has to show up as a missing route too, "
            f"because the script skips it in silence: {problems!r}")
    # Both controls reach the live guard's own entry point, not a copy of it.
    assert _missing_restore_routes(CONFIG.read_text(encoding="utf-8"), fetch) == []


# ── clause 4: the loud boot refusal, and the boundary the weights arrive across ──

#: The worker process supervisor runs. `lloyd-agent-worker.conf` starts this file, and
#: its `WorkerManager.__init__` is where the shipped detector gets loaded, so this is the
#: line the restore route exists to protect.
WORKER = ROOT / "agent-services" / "livekit_worker.py"

#: The detector's absence wrapped in a handler — what a "fix" that catches the
#: `RuntimeError` looks like. It is also the control proving the walk below can see a wrap
#: at all: without it, `== []` on the live source is a claim no checker has ever read.
COUNTERFACTUAL_WRAPPED_CALL = '''
class WorkerManager:
    def __init__(self, lk_cfg):
        try:
            self.smart_turn = voice_turn.build_turn_detector(
                lk_cfg.get("turn_detection", {}) or {}
            )
        except RuntimeError as exc:
            LOG.warning("turn detection off: %s", exc)
            self.smart_turn = None
'''


def _try_bodies_wrapping(source: str, qualified: str, call: str) -> list[str]:
    """Labels for the `try` blocks whose body contains a `call(...)` in `Class.method`.

    Qualified rather than a bare method name: `livekit_worker.py` is ~3.6k lines and
    dozens of classes define `__init__`, so a module-wide search for the name returns the
    first of them and reports "no try" about a method that was never looked at — a
    `== []` that would stay green however many handlers wrapped the real call. That is the
    vacuity this file exists to refuse, so the class is part of the question.

    Over source text rather than the live module so the counterfactual can hand it the
    wrapped shape and watch it report a handler. Only a try's own body counts: a call
    sitting in a handler or a `finally` is not guarded by that try, and the shortcut this
    refuses is precisely a body-level wrap.
    """
    cls_name, meth = qualified.split(".")
    tree = ast.parse(source)
    cls = next((n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == cls_name), None)
    assert cls is not None, f"no class named {cls_name!r} in the source handed in"
    fn = next((n for n in cls.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == meth),
              None)
    assert fn is not None, f"no {meth}() on class {cls_name} in the source handed in"
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call)
             and getattr(n.func, "attr", getattr(n.func, "id", None)) == call]
    assert calls, (
        f"no call to {call}() inside {qualified}(): the walk is answering a question the "
        f"source no longer asks, which is how an absence assertion goes vacuous")
    labels = []
    for try_node in [n for n in ast.walk(fn) if isinstance(n, ast.Try)]:
        if any(node is call_node for stmt in try_node.body
               for node in ast.walk(stmt) for call_node in calls):
            handled = ", ".join(ast.unparse(h.type) if h.type else "bare except"
                                for h in try_node.handlers)
            labels.append(f"try at line {try_node.lineno} handling {handled}")
    return labels


def test_the_worker_still_refuses_to_boot_without_the_detector():
    """Clause 4: the refusal is the design, and the loud version stays the design.

    Two halves that have to be read together, because the fix #3252 refuses is a
    one-liner in the file this node reads: catch the `RuntimeError` in
    `WorkerManager.__init__`, log a warning, set `self.smart_turn = None`, and the worker
    boots — with the room's end-of-turn detection silently gone, which is the same class
    of loss as the missing model, only quieter. So the absence of a handler is asserted
    against the real source AND the walk is proved able to report one, and the raise's
    type and message are asserted at the seam, because a handler that caught
    `RuntimeError` is only refused by a test that knows the raise is a `RuntimeError`.
    """
    source = WORKER.read_text(encoding="utf-8")
    assert _try_bodies_wrapping(source, "WorkerManager.__init__",
                                "build_turn_detector") == [], (
        "the build_turn_detector call in WorkerManager.__init__ is inside a try again, so "
        "a missing end-of-turn model is now a warning and a room with no detector — the "
        "silent-loss fix #3252 refuses. The model is tracked: restore it instead of "
        "catching the error")
    wrapped = _try_bodies_wrapping(COUNTERFACTUAL_WRAPPED_CALL, "WorkerManager.__init__",
                                   "build_turn_detector")
    assert len(wrapped) == 1, (
        f"the walk reports {wrapped} on a source that plainly wraps the call in "
        f"`except RuntimeError`, so `== []` above was never a check of anything")
    assert "RuntimeError" in wrapped[0] and "line" in wrapped[0], wrapped[0]

    pytest.importorskip("numpy", reason="voice.turn imports numpy at module scope")
    sys.path.insert(0, str(ROOT / "agent-services"))
    from voice.turn import Turn1Mini, build_turn_detector

    absent = ROOT / "tests" / "fixtures" / "no-such-detector-3252.onnx"
    assert not absent.exists(), f"{absent} exists, so this is not the missing-model case"
    for label, call in (("build_turn_detector",
                         lambda: build_turn_detector(
                             {"enabled": True, "backend": "turn-1-mini",
                              "model_path": str(absent)})),
                        ("Turn1Mini.load", lambda: Turn1Mini(absent).load())):
        with pytest.raises(RuntimeError) as raised:
            call()
        assert str(absent) in str(raised.value), (
            f"{label} raised {type(raised.value).__name__} without naming the path: a "
            f"worker's death log has to say which file the clone is missing, and the "
            f"message is the only thing the operator reads (#3252)")


def test_the_tracked_detector_loads_so_the_boot_path_has_a_witness():
    """The other side of clause 4, and the seam the weights actually arrive across.

    Everything above is about a model that is absent. The claim #3252 makes is that on a
    fresh clone this file is there and the worker boots, so one node has to cross that
    boundary for real: read the detector path out of `config.yaml`, hand it to the
    function the worker calls, load it. Skipping when the blob is missing would reproduce
    the very blindness this item was filed against — `turn-1-mini model not present` was a
    permitted machine-fact skip while the model was the one the config shipped — so the
    absence is a failure here, and the file being tracked in git is what makes that
    honest.
    """
    pytest.importorskip("numpy", reason="voice.turn imports numpy at module scope")
    sys.path.insert(0, str(ROOT / "agent-services"))
    import numpy as np

    from voice.turn import Turn1Mini, build_turn_detector

    detector = ROOT / DETECTOR
    assert detector.is_file(), (
        f"{DETECTOR} is absent from this checkout but is tracked in git (see "
        f"test_the_detector_the_shipped_config_names_arrives_with_the_clone), so the "
        f"worker that boots from it cannot start either")
    det = build_turn_detector({"enabled": True, "backend": "turn-1-mini",
                               "model_path": str(detector), "threshold": 0.1})
    assert isinstance(det, Turn1Mini), type(det)
    assert det.threshold == 0.1
    det.load()
    verdict = det.predict(np.zeros(16000, dtype=np.float32))
    assert verdict.ran, (
        "the tracked bytes load but `predict` still fails open — `ran=False` is the "
        "per-call failure path, so a clone can carry a detector that scores nothing and "
        "the room simply never ends its turns")
    assert 0.0 <= verdict.probability <= 1.0, verdict.probability
