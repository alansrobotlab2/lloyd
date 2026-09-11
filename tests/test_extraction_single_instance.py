"""Two nightly extractors must never run at once.

Why this file exists
--------------------
`nightly_extraction.py` rewrites `_pipeline/content-hashes.json` and the fact
tree. Nothing in the script prevented two copies doing that to each other; the
only guard was autonomy task #24's `in_progress` status, which is a guard on the
*turn*, not on the process.

That guard has a hole, and task #24's skill widens it deliberately: the script
routinely takes 545-1666s while the Bash tool caps at 600s, so Step 1 launches it
with `run_in_background`. The child therefore outlives its turn by design — and
`_recover_stuck_tasks` frees a task stuck past its timeout, so a turn killed by a
backend restart leaves an extractor running against a task that is once again
`up_next`. On 2026-09-08 that was the live state: an orphaned extractor at
document 19 of 1395, with the task due to be freed 55 minutes later.

flock rather than a pidfile, because the kernel drops it when the holder dies:
a `kill -9` or an OOM cannot strand a lock that wedges the pipeline forever.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_NGM = "/home/alansrobotlab/lloyd/scripts/memory/next-gen-memory"


def _in_child(body: str) -> str:
    """Run `body` in a separate interpreter and return its stdout."""
    src = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {_NGM!r})
        sys.path.insert(0, '/home/alansrobotlab/lloyd')
        import nightly_extraction as ne
        {body}
    """)
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.strip()


@pytest.fixture
def ne(monkeypatch, tmp_path):
    """The real module, with the lock pointed at a tmp path."""
    sys.path.insert(0, _NGM)
    sys.path.insert(0, "/home/alansrobotlab/lloyd")
    import nightly_extraction as mod
    monkeypatch.setattr(mod, "_LOCK_PATH", tmp_path / "x.lock")
    return mod


def test_the_lock_is_taken_when_free(ne):
    held = ne.acquire_single_instance_lock()
    assert held, "a free lock must be acquirable"
    held.close()


@pytest.fixture
def own_lock(tmp_path, monkeypatch):
    """A lock path of the test's own, in the parent AND in every child.

    The live path is `~/lloyd/_pipeline/nightly_extraction.lock`, and on
    2026-09-11 a real extractor (task #24) held it while the automod gate ran
    this file: two tests red for every round that gated in that window, none
    of them about the round. `LLOYD_EXTRACTION_LOCK` reaches the children
    through the environment; the already-imported module is patched directly.
    """
    lock = tmp_path / "extraction.lock"
    monkeypatch.setenv("LLOYD_EXTRACTION_LOCK", str(lock))
    sys.path.insert(0, _NGM)
    sys.path.insert(0, "/home/alansrobotlab/lloyd")
    import nightly_extraction as mod
    monkeypatch.setattr(mod, "_LOCK_PATH", lock)
    return mod


def test_a_second_holder_is_refused_and_a_release_frees_it(own_lock):
    """One process at a time — and `None` specifically, since the caller
    branches on it to print `status=locked` and exit 0 rather than fail."""
    mod = own_lock

    held = mod.acquire_single_instance_lock()
    assert held
    try:
        assert _in_child(
            "print('REFUSED' if ne.acquire_single_instance_lock() is None "
            "else 'ACQUIRED')") == "REFUSED"
    finally:
        held.close()
    assert _in_child(
        "print('REFUSED' if ne.acquire_single_instance_lock() is None "
        "else 'ACQUIRED')") == "ACQUIRED"


def test_a_dead_holder_does_not_strand_the_lock(own_lock):
    """The whole reason for flock over a pidfile: a killed extractor must not
    wedge every future run. The child exits without releasing anything."""
    assert _in_child("ne.acquire_single_instance_lock(); print('ok')") == "ok"
    assert _in_child(
        "print('REFUSED' if ne.acquire_single_instance_lock() is None "
        "else 'ACQUIRED')") == "ACQUIRED"


def test_an_unusable_lock_path_does_not_block_extraction(ne, monkeypatch):
    """Fail open, not closed. This guard protects against a rare overlap; a
    read-only `_pipeline` must not turn that into a total pipeline outage.
    `False` is distinct from `None` — only `None` means "someone else has it"."""
    monkeypatch.setattr(ne, "_LOCK_PATH", ne.Path("/proc/nope/x.lock"))
    assert ne.acquire_single_instance_lock() is False
