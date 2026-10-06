"""The transcript extractor's watermark survives a pass, and the pass that writes it cannot crash.

`scripts/memory/extract-transcript.py` is the memory-capture job's reader: it walks
`SESSIONS_DIR` and prints only sessions touched since its last run, keeping that
"last run" in `app.paths.MEMORY_CAPTURE_STATE_PATH`. Until #2294 it kept it in
`state.json` beside the script — a path `app.paths` knows nothing about and no
checkout tracks — so the file never existed, `load_state()` answered
`{"lastRunTs": 0}` forever, and the filter was dead: a pass printed
`DEBUG: Processing 7552 recent files` against the live corpus (measured 2026-10-06)
and then dropped all but the last 50 KB of what it built (`:166-168`). Two things
were being paid for that no code ever read: a full re-parse of the sessions store on
every run, and a transcript that is the tail of the corpus rather than the delta the
caller asked for.

What this pins, across the boundary that matters — the script is a standalone
`python3 extract-transcript.py` invoked by `skills/memory-capture/SKILL.md:30` and
`skills/periodic-memory-capture-lloyd/SKILL.md:84`, so every node here runs it as a
subprocess with `LLOYD_DATA` set to a scratch root and nothing else overridden. In
-process would let the suite's own already-imported `app.paths` decide the root for
me; the child re-resolves it from the environment, which is the thing production
does and the thing the old in-tree binding could not honour: two checkouts sharing
one machine-wide file.

The second node exists because moving the file created a new failure mode rather
than merely relocating the old one: `app.paths` creates NOTHING under `DATA_ROOT` on
import (#712) and this script never calls `ensure_dirs()`, so the first pass against
a root that does not exist yet had to make the directory itself or die in `open(…,
"w")`. A watermark writer that raises leaves `lastRunTs` at 0 exactly as a watermark
file nobody ever wrote did — the silent-zero bug, reached a different way.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "memory" / "extract-transcript.py"
#: The watermark's filename, spelled here rather than imported so that a node can
#: assert on the name and not on the constant the code under test chose. `app.paths`
#: is the one place the path is composed; this is the witness.
STATE_NAME = "memory-capture-state.json"
#: One fixed mtime, older than any run: the selector compares `getmtime(f) > lastRunTs`,
#: and a file stamped "now" would race the pass that reads it.
SEED_MTIME = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()


def _session(sid: str, text: str) -> dict:
    """A session shaped like a real one. An assistant line under 10 chars is skipped by
    the extractor (`:98`), so the fixture text has to clear that to be a witness at all.
    """
    return {
        "session_id": sid,
        "created_at": "2026-01-01T00:00:00",
        "messages": [
            {"role": "user", "content": f"question about {sid}",
             "timestamp": "2026-01-01T00:00:01"},
            {"role": "assistant", "content": f"answer about {text}",
             "timestamp": "2026-01-01T00:00:02"},
        ],
    }


def _root(tmp_path, *, sessions: int = 2, create: bool = True) -> Path:
    """A scratch `LLOYD_DATA` root with `sessions/` seeded, or without it at all."""
    root = tmp_path / "data-root"
    if not create:
        return root
    sessions_dir = root / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    for n in range(sessions):
        path = sessions_dir / f"2026010{n}_seed.json"
        path.write_text(json.dumps(_session(f"sid-{n}", f"topic-{n}")), encoding="utf-8")
        os.utime(path, (SEED_MTIME, SEED_MTIME))
    return root


def _run(root: Path, *args: str) -> subprocess.CompletedProcess:
    """One pass, as a skill runs it: this tree's script, this scratch root, real clock."""
    env = {**os.environ, "LLOYD_DATA": str(root)}
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True,
                          text=True, timeout=120, env=env)


def _read_state(root: Path) -> dict:
    return json.loads((root / STATE_NAME).read_text(encoding="utf-8"))


def test_a_second_pass_over_the_same_corpus_selects_zero_files(tmp_path):
    """Clause 3: after one pass records its timestamp, the next pass finds nothing to say.

    The measurement is the extractor's own, not an inference from its stdout. With
    `lastRunTs == 0` the selector at `:134` takes the `else` branch and returns EVERY
    file, printing `DEBUG: Processing N recent files`; with a watermark it filters to
    files touched since, and when that set is empty the pass exits at `:136-139` before
    that print is reached. So "the pass that follows reports no `Processing N` line at
    all" is the extractor reporting a zero-length selection, and "no `Processing 2`"
    would not do: the first pass must have selected 2, or the second pass's silence
    would only prove the corpus was empty.
    """
    root = _root(tmp_path, sessions=2)

    first = _run(root)
    assert first.returncode == 0, first.stderr[-500:]
    assert "DEBUG: Processing 2 recent files" in first.stderr, (
        "the first pass must select the whole seeded corpus, or the second pass proves "
        f"nothing. stderr: {first.stderr[-400:]}")
    assert "sid-0" in first.stdout and "sid-1" in first.stdout, first.stdout[:300]

    state = _read_state(root)
    assert state["lastRunTs"] > 0, f"the pass wrote no watermark: {state}"

    second = _run(root)
    assert second.returncode == 0, second.stderr[-500:]
    assert "DEBUG: Processing" not in second.stderr, (
        "the second pass still selected files — the watermark is not being read back: "
        f"{second.stderr[-400:]}")
    assert second.stdout.strip() == "", (
        f"a pass with nothing new must print nothing, got {second.stdout[:200]!r}")

    # Positive control on the selector itself: a session newer than the watermark has
    # to come back, or the silence above is just a filter that never passes anything.
    fresh = root / "sessions" / "20260109_fresh.json"
    fresh.write_text(json.dumps(_session("sid-fresh", "genuinely new")), encoding="utf-8")
    third = _run(root)
    assert "DEBUG: Processing 1 recent files" in third.stderr, (
        "the watermark filters, but not on mtime — a file newer than it must be "
        f"selected. stderr: {third.stderr[-400:]}")
    assert "sid-fresh" in third.stdout and "sid-0" not in third.stdout, third.stdout[:300]


def test_a_pass_against_a_data_root_that_does_not_exist_yet_writes_the_watermark(tmp_path):
    """Clause 4: the writer makes its own directory rather than raising.

    `app.paths` resolves a `DATA_ROOT` it was never asked to create, and importing it
    creates nothing (#712); this script is not a service and calls no `ensure_dirs()`.
    So the directory holding the watermark is `save_state`'s to make, and the assertion
    that does the work here is `returncode == 0` with no traceback: a bare
    `open(STATE_FILE, "w")` over this path raised `FileNotFoundError` straight out of
    `main`, which `skills/memory-capture/SKILL.md` hands to a session as a successful
    empty run — a watermark that stays 0 because the writer died is indistinguishable
    from a watermark that was never written, and this file exists because of exactly
    that indistinguishability.

    No sessions at all is deliberate: with an empty corpus the pass exits at `:129-132`,
    which is the shortest route to the writer and the state a fresh box is really in.
    """
    root = _root(tmp_path, create=False)
    assert not root.exists(), "the fixture must start with a root that is not there"

    run = _run(root)
    assert "Traceback" not in run.stderr, run.stderr[-600:]
    assert run.returncode == 0, f"exit {run.returncode}: {run.stderr[-600:]}"
    assert root.is_dir(), "the pass resolved a root it never created"

    state = _read_state(root)
    assert state["lastRunTs"] > 0, f"watermark written empty or not at all: {state}"

    # And the directory it made is the one the constant names, not a sibling it
    # happened to land in: a second pass reads the same bytes back.
    second = _run(root)
    assert second.returncode == 0, second.stderr[-500:]
    assert "DEBUG: Processing" not in second.stderr, (
        "the watermark this pass wrote is not the one the next pass reads")
