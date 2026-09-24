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

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Tree-relative, like every other path in this checkout (#755). The absolute
# form named the LIVE checkout, so a child interpreter spawned from a
# self-modification worktree imported the live `nightly_extraction` while the
# parent inspected its own copy — the silent hybrid that would let a
# `PIPELINE_RESULT` assertion grade whatever happened to be deployed instead of
# the code under test.
ROOT = Path(__file__).resolve().parent.parent
_NGM = str(ROOT / "scripts" / "memory" / "next-gen-memory")


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


# ── the machine-readable summary carries coverage (#1151 clause 4) ───────────
#
# `PIPELINE_RESULT` is what `autonomy-data-pipeline` reads to decide whether the
# corpus is healthy, and until now its whole vocabulary was
# `files_processed` / `facts` / `failed` / `status`. A 60,000-char document from
# which one pass can read at most 47,000 chars therefore reported exactly like a
# document that was read: `failed=0 status=ran`. The shortfall existed only as a
# `⚠️ capped at 6 chunks` line in prose (`run_24_20260917_153605.md:66`), and the
# run then hashed the file and skipped it forever.
#

_MAIN_DRIVER = '''
"""Run `nightly_extraction.main()` as the separate process it is.

`argv[1]` is the test's tmp directory, which the environment points every state
path at: `LLOYD_DATA` (the `_pipeline` root, and so the content-hash index),
`LLOYD_FACTS_ROOT`, `LLOYD_EXTRACTION_LOCK`. `TEST_REPO_ROOT` names the checkout
under test.

Three things are replaced, and none of them is the code clause 4 is about: the
model call answers with an empty fact list; Steps 2 and 3 (the relations-index
rebuild and the overview pass) are stubbed, because they read the live fact tree
and cost minutes; and the corpus is the one document the test wrote.
`run_full_extraction`, the content-hash gate, the resume offset and the
`PIPELINE_RESULT` line `main()` prints are all the real ones.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ["TEST_REPO_ROOT"])
sys.path.insert(0, str(ROOT / "scripts" / "memory" / "next-gen-memory"))
sys.path.insert(0, str(ROOT))

from app import kg_store                      # noqa: E402
import fact_extractor as fx                   # noqa: E402
import nightly_extraction as ne               # noqa: E402

TMP = Path(sys.argv[1]).resolve()
# The pipeline log and the content-hash index both live under `_pipeline`;
# `ContentHasher.save` makes that directory, but the noop path writes its log
# without ever saving, and a missing parent there is an OSError, not a noop.
(TMP / "_pipeline").mkdir(parents=True, exist_ok=True)
kg_store.configure(TMP / "kg.sqlite")


class _StubIndex:
    def rebuild(self):
        return {"total_relationships": 0}


class _StubOverviews:
    def regenerate_all(self, workers=8):
        return 0


class _Headless(ne.NightlyExtraction):
    def __init__(self):
        super().__init__()
        self.rel_generator = _StubIndex()
        self.profile_generator = _StubOverviews()


ne.NightlyExtraction = _Headless
ne.VAULT = TMP
ne._load_pipeline_config = lambda: {"sources": {"paths": ["docs"]}}
fx.FactExtractor._call_llm = lambda self, prompt: json.dumps(
    {"entity": "", "category": "state", "facts": []})

sys.argv = ["nightly_extraction.py"]
ne.main()
'''

DOC_CHARS = 60_000


def _run_main(tmp_path) -> str:
    """One real `main()` in a child interpreter over `tmp_path`; returns stdout."""
    env = dict(os.environ,
               TEST_REPO_ROOT=str(ROOT),
               LLOYD_DATA=str(tmp_path),
               LLOYD_FACTS_ROOT=str(tmp_path / "facts"),
               LLOYD_EXTRACTION_LOCK=str(tmp_path / "extraction.lock"))
    # The index under test is the one under LLOYD_DATA, not an inherited
    # override pointing at a live run's resume point.
    env.pop("LLOYD_CONTENT_HASHES", None)
    r = subprocess.run([sys.executable, str(tmp_path / "driver.py"), str(tmp_path)],
                       capture_output=True, text=True, timeout=300, env=env,
                       cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


def _result_fields(stdout: str) -> dict:
    lines = [ln for ln in stdout.splitlines() if ln.startswith("PIPELINE_RESULT")]
    assert len(lines) == 1, f"expected exactly one summary line, got:\n{stdout[-2000:]}"
    return dict(kv.split("=", 1) for kv in lines[0].split()[1:])


def _result_json(stdout: str) -> dict:
    """The summary dict `main()` prints as `Result: {…}`.

    Anchored at a line start, not split on the bare substring: the extraction log
    interleaves model chatter, and one `Result:` inside a log line or a fact body
    would silently hand back the wrong object — a JSON decoder starting at the
    wrong brace usually still decodes *something*.
    """
    hit = re.search(r"^Result: ", stdout, re.MULTILINE)
    assert hit, f"no `Result:` summary line in\n{stdout[-2000:]}"
    return json.JSONDecoder().raw_decode(stdout[hit.end():].strip())[0]


def test_the_pipeline_result_names_documents_whose_tail_went_unread(tmp_path):
    """Three runs over one 60,000-char document — past the 47,000 one pass can
    cover. Run 1 reports `truncated=1` and names the document in the summary
    dict while still reporting `failed=0`; run 2 resumes at the stored offset,
    reads the tail and reports `truncated=0`; run 3 has nothing left to read and
    still carries the field."""
    (tmp_path / "driver.py").write_text(_MAIN_DRIVER, encoding="utf-8")
    (tmp_path / "facts").mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    doc = docs / "feed.md"
    doc.write_text("".join("CELL%06d" % c + "y" * 90 for c in range(600)),
                   encoding="utf-8")
    assert len(doc.read_text(encoding="utf-8")) == DOC_CHARS

    first = _run_main(tmp_path)
    fields = _result_fields(first)
    assert fields["failed"] == "0" and fields["files_processed"] == "1"
    assert fields["truncated"] == "1", (
        "a document whose tail went unread is invisible in the summary: "
        "failed=0 again means full coverage")
    assert _result_json(first)["truncated_files"] == ["docs/feed.md"]

    second = _run_main(tmp_path)
    assert _result_fields(second)["truncated"] == "0", (
        "the second pass never finished the document, or the field is sticky")
    entry = json.loads((tmp_path / "_pipeline" / "content-hashes.json").read_text()) \
        ["hashes"][str(doc)]
    assert entry["covered_through"] == DOC_CHARS
    assert entry["complete"] is True

    third = _run_main(tmp_path)
    late = _result_fields(third)
    assert late["files_processed"] == "0", "a fully covered document was extracted again"
    assert late["truncated"] == "0"
