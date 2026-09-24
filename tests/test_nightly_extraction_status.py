"""#1405 — a run whose every changed file failed is `status=failed`, not `noop`.

On 2026-09-23 `nightly_extraction.py` printed
`PIPELINE_RESULT files_processed=0 facts=0 failed=6 status=noop`: six changed
documents had all timed out behind a saturated engine, and the short-circuit
that skips the downstream steps keyed on `files_processed == 0` alone. `noop`
is the data-pipeline skill's NO_NEW_DATA gate, so a starvation failure was
recorded as an idle vault. The real `main()` runs here in a child interpreter,
as in `test_extraction_single_instance.py`, with the model call stubbed to fail
or to succeed; the gate, the summary dict and the printed line are the real ones.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_DRIVER = '''
import json, os, sys
from pathlib import Path
ROOT = Path(os.environ["TEST_REPO_ROOT"])
sys.path.insert(0, str(ROOT / "scripts" / "memory" / "next-gen-memory"))
sys.path.insert(0, str(ROOT))
from app import kg_store
import fact_extractor as fx
import nightly_extraction as ne

TMP = Path(sys.argv[1]).resolve()
MODE = sys.argv[2]
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


def _fail(self, prompt):
    raise fx.ExtractionFailed("LLM call failed: timed out")


ne.NightlyExtraction = _Headless
ne.VAULT = TMP
ne._load_pipeline_config = lambda: {"sources": {"paths": ["docs"]}}
fx.FactExtractor._call_llm = _fail if MODE == "fail" else (
    lambda self, prompt: json.dumps({"entity": "", "category": "state", "facts": []}))
sys.argv = ["nightly_extraction.py"]
ne.main()
'''


def _run(tmp_path, mode):
    env = dict(os.environ, TEST_REPO_ROOT=str(ROOT), LLOYD_DATA=str(tmp_path),
               LLOYD_FACTS_ROOT=str(tmp_path / "facts"),
               LLOYD_EXTRACTION_LOCK=str(tmp_path / "extraction.lock"))
    env.pop("LLOYD_CONTENT_HASHES", None)
    r = subprocess.run([sys.executable, str(tmp_path / "driver.py"), str(tmp_path), mode],
                       capture_output=True, text=True, timeout=300, env=env,
                       cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-3000:]
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("PIPELINE_RESULT")]
    assert len(lines) == 1, r.stdout[-2000:]
    fields = dict(kv.split("=", 1) for kv in lines[0].split()[1:])
    hit = re.search(r"^Result: ", r.stdout, re.MULTILINE)
    result = json.JSONDecoder().raw_decode(r.stdout[hit.end():].strip())[0]
    return fields, result


def _corpus(tmp_path, n=3):
    (tmp_path / "driver.py").write_text(_DRIVER, encoding="utf-8")
    (tmp_path / "facts").mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(n):
        (docs / f"note{i}.md").write_text(f"# Note {i}\n\nAlan configured thing {i}.\n",
                                          encoding="utf-8")


def test_every_changed_file_failing_reports_failed_not_noop(tmp_path):
    _corpus(tmp_path)
    fields, result = _run(tmp_path, "fail")
    assert fields["files_processed"] == "0" and fields["failed"] == "3"
    assert fields["status"] == "failed", (
        "a run whose changed files all failed printed status="
        f"{fields['status']}: the skill's gate reads that as an idle vault")
    assert result["noop"] is False and result["status"] == "failed"


def test_the_failed_files_are_offered_again_and_a_clean_rerun_is_ran_then_noop(tmp_path):
    _corpus(tmp_path)
    _run(tmp_path, "fail")
    fields, _ = _run(tmp_path, "ok")
    assert fields["status"] == "ran" and fields["files_processed"] == "3"
    fields, result = _run(tmp_path, "ok")
    # The positive control: with nothing changed, noop still means noop.
    assert fields["status"] == "noop" and fields["failed"] == "0"
    assert result["noop"] is True


def test_the_model_call_ceiling_covers_queueing_not_only_generation():
    """A flat 120 s is what a priority-2 call queued behind another stage cannot
    meet; the call must use the module's ceiling, not a literal."""
    src = (ROOT / "scripts" / "memory" / "next-gen-memory" / "fact_extractor.py").read_text()
    assert "timeout=120" not in src
    assert "urlopen(req, timeout=LLM_CALL_TIMEOUT_S)" in src
    sys.path.insert(0, str(ROOT / "scripts" / "memory" / "next-gen-memory"))
    sys.path.insert(0, str(ROOT))
    import fact_extractor as fx
    assert fx.LLM_CALL_TIMEOUT_S >= 600 or os.environ.get("LLOYD_EXTRACTION_CALL_TIMEOUT_S")
