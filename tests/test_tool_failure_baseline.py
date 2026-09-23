"""eval/tool_failure_baseline.py — the per-tool failure-signature aggregate (#851).

Every row these tests build goes through `app.transcript_entries`, the module
that actually writes `sessions/*.json` (`app/routers/messages.py:1065`,
`app/run_recorder.py:267`). That is the one process boundary this change
crosses — the script reads files a *different* process wrote — so a fixture
hand-shaped here would keep passing after the producer renamed a field and the
aggregate silently read zero flags. Hand-shaped dicts would; these do not.

Clause map (the acceptance clauses on #851):
  1  test_the_days_window_selects_by_mtime..., test_two_runs_over_one_corpus...
  2  test_a_tool_line_carries_calls_flagged_and_signature_counts,
     test_two_instances_of_one_cause_share_a_signature
  3  test_two_bare_shell_exits_count_as_shell_exit_and_no_signature
  4  test_an_unclassifiable_flagged_row_is_totalled_outside_every_signature
  5  test_the_default_output_is_inferior_to_the_scorer_test_glob,
     test_the_scorer_test_still_passes_with_a_newer_file_in_the_output_dir
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval.tool_failure_baseline as tfb  # noqa: E402
from app import transcript_entries as te  # noqa: E402

TS = "2026-09-22T00:00:00"

# The two bodies #851 exists to rank: Edit's bare `old_string not found` and
# http_fetch's bare `HTTP <n>`. Copied from real transcripts, not invented —
# a fixture whose text drifted from the tool's would make the baseline's whole
# purpose untestable.
EDIT_NO_MATCH = '{"error": "old_string not found in file (must match exactly)"}'
HTTP_404 = '{"error": "HTTP 404"}'


def _call(name: str, call_id: str) -> dict:
    """The assistant row that names the tool, via the real builder."""
    return te.build_tool_call_entry(te.build_tool_call(call_id, name, "{}"),
                                    timestamp=TS)


def _result(body: str, call_id: str, is_error: bool) -> dict:
    """The `role="tool"` row whose `stats.is_error` this whole file reads."""
    return te.build_tool_result_entry(call_id, body, timestamp=TS,
                                      is_error=is_error)


def _session(tmp_path: Path, name: str, rows: list[dict], *,
             age_days: float = 0.0) -> Path:
    """One transcript file, aged so the `--days` window has something to do."""
    path = tmp_path / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"session_id": name, "messages": rows}))
    stamp = time.time() - age_days * 86400
    os.utime(path, (stamp, stamp))
    return path


def _pairs(*specs: tuple[str, str, bool]) -> list[dict]:
    """(tool, body, is_error) triples → call+result rows with fresh call ids."""
    rows: list[dict] = []
    for i, (tool, body, flag) in enumerate(specs):
        cid = f"c{i}"
        rows.append(_call(tool, cid))
        rows.append(_result(body, cid, flag))
    return rows


# ── clause 1: the window, and re-runnability ─────────────────────────────────

def test_the_days_window_selects_sessions_by_mtime_and_never_reads_the_rest(tmp_path):
    """`--days` is the whole reason the number means anything: an unbounded
    scan would mix a Bash from before a fix with one after it."""
    corpus = tmp_path / "sessions"
    _session(corpus, "fresh", _pairs(("Bash", "x\n[exit code: 1]", True),
                                     ("Read", '{"ok": true}', False)),
             age_days=0.0)
    _session(corpus, "stale", _pairs(("Edit", EDIT_NO_MATCH, True)), age_days=40.0)

    near = tfb.build_record(corpus, 21)
    assert near["sessions_scanned"] == 1
    assert near["tool_calls_total"] == 2
    assert near["flagged_total"] == 1
    assert "Edit" not in near["tools"], "a session 40 days old leaked into a 21-day window"

    wide = tfb.build_record(corpus, 60)
    assert wide["sessions_scanned"] == 2
    assert wide["tools"]["Edit"]["flagged"] == 1

    assert tfb.build_parser().get_default("days") == tfb.DEFAULT_DAYS == 21
    assert tfb.build_record(tmp_path / "does-not-exist", 21)["tool_calls_total"] == 0


def test_two_runs_over_one_corpus_produce_identical_numbers(tmp_path):
    """Re-runnable is the item's whole point — a baseline nobody can reproduce
    is a screenshot. Only `generated_at` may move between two runs, so the
    window anchors on the newest session mtime rather than on the clock."""
    corpus = tmp_path / "sessions"
    _session(corpus, "a", _pairs(("Edit", EDIT_NO_MATCH, True),
                                 ("Edit", EDIT_NO_MATCH, True),
                                 ("Bash", "(no output)\n[exit code: 2]", True)),
             age_days=1.0)
    _session(corpus, "b", _pairs(("http_fetch", HTTP_404, True)), age_days=2.0)

    first, second = (tfb.build_record(corpus, 21) for _ in range(2))
    assert set(first) == set(second)
    volatile = {k for k in first if first[k] != second[k]}
    assert volatile <= {"generated_at"}, volatile
    assert first["tools"]["Edit"]["signatures"] == second["tools"]["Edit"]["signatures"]

    # And through the CLI: argv parsing and the file write are the caller's
    # actual path, not the library call this test otherwise stops at.
    out_a, out_b = tmp_path / "a.json", tmp_path / "b.json"
    for out in (out_a, out_b):
        assert tfb.main(["--sessions-dir", str(corpus), "--days", "21",
                         "--out", str(out)]) == 0
    rec_a = json.loads(out_a.read_text())
    rec_b = json.loads(out_b.read_text())
    rec_a.pop("generated_at"), rec_b.pop("generated_at")
    assert rec_a == rec_b
    # Same bytes apart from the wall-clock stamp, whose width is fixed: two runs
    # diff to nothing, which is what makes the file a baseline.
    assert out_a.stat().st_size == out_b.stat().st_size


# ── clause 2: the per-tool line ─────────────────────────────────────────────

def test_a_tool_line_carries_calls_flagged_and_signature_counts(tmp_path):
    """Per tool name: total calls, flagged count, and signatures each with a
    count — where a signature is the normalised FIRST LINE of the result body,
    which is why the error's own text has to survive normalisation."""
    corpus = tmp_path / "sessions"
    _session(corpus, "a", _pairs(
        ("Edit", EDIT_NO_MATCH, True),
        ("Edit", EDIT_NO_MATCH, True),
        ("Edit", '{"error": "file_path is required"}', True),
        ("Edit", '{"count": 3}', False),
    ))

    line = tfb.build_record(corpus, 21)["tools"]["Edit"]
    assert line["calls"] == 4
    assert line["flagged"] == 3
    counts = {s["signature"]: s["count"] for s in line["signatures"]}
    assert counts == {EDIT_NO_MATCH: 2,
                      '{"error": "file_path is required"}': 3 - 2}
    # The ranking is only worth reading if the cause is legible in it: a
    # length-based mask would have collapsed both keys to `{"error": "<long>"}`.
    assert any("old_string not found in file" in s for s in counts)


def test_two_instances_of_one_cause_share_a_signature_while_two_causes_do_not(tmp_path):
    """Normalisation is what turns 1,405 flagged Edit rows into a ranking.
    Paths, positional numbers and echoed payloads vary between two instances of
    one cause; an HTTP status code does not — it IS the cause (#850)."""
    same_cause = [
        ('{"error": "Edit refused: /home/alansrobotlab/lloyd/a.py changed on '
         'disk since you last Read it (size or mtime differ)."}',
         '{"error": "Edit refused: /home/alansrobotlab/lloyd/b.py changed on '
         'disk since you last Read it (size or mtime differ)."}'),
        ('{"error": "old_string not found in file (must match exactly); '
         "nearest lines 251, 827: 'value = compute(1)'\"}",
         '{"error": "old_string not found in file (must match exactly); '
         "nearest lines 4, 9: 'other = f(2)'\"}"),
        ('{"error": "HTTP 404", "retry_class": "no-retry-gone", '
         '"final_url": "https://example.com/one", "body": "<html>a</html>"}',
         '{"error": "HTTP 404", "retry_class": "no-retry-gone", '
         '"final_url": "https://other.example/two", "body": "<html>b</html>"}'),
    ]
    for a, b in same_cause:
        assert tfb.normalise_signature(a) == tfb.normalise_signature(b), a

    assert (tfb.normalise_signature('{"error": "HTTP 404"}')
            != tfb.normalise_signature('{"error": "HTTP 403"}'))
    assert "404" in tfb.normalise_signature('{"error": "HTTP 404"}')

    # A pretty-printed error body opens with a bare `{`: structural lines carry
    # no cause, so the signature comes from the first line that has a letter.
    assert (tfb.normalise_signature('{\n  "error": "gate did not pass",\n}')
            == tfb.normalise_signature('{\n  "error": "gate did not pass",\n}'))
    assert tfb.normalise_signature('{\n  "error": "gate did not pass",\n}') != "{"

    corpus = tmp_path / "sessions"
    _session(corpus, "a", _pairs(("Edit", same_cause[0][0], True),
                                 ("Edit", same_cause[0][1], True)))
    line = tfb.build_record(corpus, 21)["tools"]["Edit"]
    assert [s["count"] for s in line["signatures"]] == [2]


# ── clause 3: the shell_exit split ──────────────────────────────────────────

def test_two_bare_shell_exits_count_as_shell_exit_and_no_signature(tmp_path):
    """#500's reason this baseline cannot just count `is_error`: a `grep` with
    no match exits 1 and the whole corpus is full of them. Bash's flagged share
    is 2,582 of 3,376 flagged rows over the 21-day window as this shipped, all
    of them raw output — so they are counted, and ranked as nothing."""
    corpus = tmp_path / "sessions"
    _session(corpus, "a", _pairs(("Bash", "(no output)\n[exit code: 1]", True),
                                 ("Bash", "total 216\ndrwx r-x tests\n"
                                          "[exit code: 2]", True)))

    rec = tfb.build_record(corpus, 21)
    assert rec["tools"]["Bash"]["shell_exit"] == 2
    assert rec["tools"]["Bash"]["signatures"] == []
    assert rec["tools"]["Bash"]["flagged"] == 2
    assert rec["shell_exit_total"] == 2
    assert rec["flagged_total"] == 2
    assert tfb.classify("(no output)\n[exit code: 1]")[0] == tfb.CLASS_SHELL_EXIT


def test_a_program_that_prints_error_shaped_output_and_then_exits_is_a_shell_exit(tmp_path):
    """The ordering the split depends on: a test runner that prints a failing
    case as JSON and exits 1 has an exit code, not a tool defect. If the JSON
    shape were tested first, this row would rank as an error signature."""
    corpus = tmp_path / "sessions"
    _session(corpus, "a", _pairs(("Bash", '{"error": "1 failed"}\n[exit code: 1]', True),
                                 ("Bash", '{"error": "command timed out after 120000ms",'
                                          ' "command": "sleep 900"}', True)))

    bash = tfb.build_record(corpus, 21)["tools"]["Bash"]
    assert bash["shell_exit"] == 1
    assert [s["count"] for s in bash["signatures"]] == [1]
    # The timeout body has no exit-code suffix at all: Bash returns it before
    # the exit code exists (`agent_mcp/builtin_bash.py:197`), so it stays a
    # real, rankable cost.
    assert "command timed out" in bash["signatures"][0]["signature"]
    assert "<n>ms" in bash["signatures"][0]["signature"]


# ── clause 4: guess_class ───────────────────────────────────────────────────

def test_an_unclassifiable_flagged_row_is_totalled_outside_every_signature(tmp_path):
    """An unknown body must not inflate a ranked cause. The identity that makes
    the file trustworthy, per tool and in total: flagged = signatures +
    shell_exit + guess_class, with guess_class holding exactly the residue."""
    corpus = tmp_path / "sessions"
    _session(corpus, "a", _pairs(
        ("Edit", EDIT_NO_MATCH, True),                       # structured → sig
        ("Edit", "Tool 'Edit' is disabled by configuration.",  # harness → sig
         True),                                             # loop.py:2057
        ("Edit", "---", True),                               # unknown → guess
        ("Bash", "(no output)\n[exit code: 1]", True),       # shell exit
    ))

    rec = tfb.build_record(corpus, 21)
    edit = rec["tools"]["Edit"]
    assert edit["guess_class"] == 1
    assert edit["flagged"] == 3
    # Clause 4's exact form, on a tool with no shell exits: the signature
    # counts fall short of the flagged count by exactly guess_class.
    assert (sum(s["count"] for s in edit["signatures"])
            == edit["flagged"] - edit["guess_class"])
    assert not any("---" in s["signature"] for s in edit["signatures"])

    assert rec["guess_class_total"] == 1
    all_sigs = sum(s["count"] for t in rec["tools"].values()
                   for s in t["signatures"])
    assert rec["flagged_total"] == (all_sigs + rec["shell_exit_total"]
                                   + rec["guess_class_total"])
    assert rec["classes"] == list(tfb.CLASSES)


def test_a_known_failure_text_that_is_neither_json_nor_a_shell_exit_is_a_class():
    """The harness writes plain-text failures that the `text_result` sniffer
    cannot see and that are not shell exits either. Each marker here names its
    writer in HARNESS_MARKERS, so a prose drift shows up as a widening
    guess_class rather than a silently ranked cause."""
    for text in ("Tool call denied: harness safety: blocked 'sudo' on 'sudo'",
                 "Tool call arguments could not be parsed as JSON: Unterminated",
                 "Tool dispatch failed: boom",
                 "Tool 'Bash' is disabled by configuration.",
                 "Tool 'Bash' cancelled by user.",
                 "Bash: no server claims tool 'Bash'",
                 "Bash: transport error: unhandled errors in a TaskGroup"):
        cls, sig = tfb.classify(text)
        assert cls == tfb.CLASS_HARNESS_ERROR, text
        assert sig, text
    assert tfb.classify("just some prose nobody wrote a matcher for")[0] == tfb.CLASS_GUESS
    assert tfb.classify("")[1] == ""


# ── clause 5: the output path, and the scorer test that reads that directory ──

def test_the_default_output_is_inferior_to_the_scorer_test_glob(tmp_path):
    """`tests/test_eval_scorer.py::test_a_run_record_declares_whether_it_matched_production`
    takes the newest `eval/baselines/*.json` by mtime and demands the retrieval
    knobs in it. A sibling file here would turn that test red, so the default
    output has to be one level deeper — which `glob('*.json')` cannot see."""
    out = tfb.default_out_path({"date": "2026-09-22"})
    assert out.parent == tfb.DEFAULT_OUT_DIR
    assert out == tfb.EVAL_BASELINES_DIR / "tool-failures" / "2026-09-22.json"
    assert out.parent.name not in ("", ".")
    assert out.parent.relative_to(tfb.EVAL_BASELINES_DIR).parts == ("tool-failures",)

    # The glob's own semantics, demonstrated rather than asserted about: a
    # non-recursive `*.json` cannot reach a subdirectory, `**/*.json` can.
    root = tmp_path / "baselines"
    (root / "tool-failures").mkdir(parents=True)
    (root / "run.json").write_text("{}")
    (root / "tool-failures" / "2026-09-22.json").write_text("{}")
    assert [p.name for p in root.glob("*.json")] == ["run.json"]
    assert (root / "tool-failures" / "2026-09-22.json") in list(root.glob("**/*.json"))


def test_the_scorer_test_still_passes_with_a_newer_file_in_the_output_dir(tmp_path):
    """The seam, crossed for real rather than described: a NEWER file sitting in
    the output directory, carrying none of the keys that test demands. If the
    glob were recursive the nested test would pick this file up and die on
    `matches_production_defaults`; it is not, so the newest top-level baseline
    is still what it reads."""
    baselines = tfb.EVAL_BASELINES_DIR
    shadow_dir = tfb.DEFAULT_OUT_DIR
    shadow_dir.mkdir(parents=True, exist_ok=True)
    made_top_level = None
    if not [p for p in baselines.glob("*.json") if p.is_file()]:
        # A fresh worktree has no baselines at all (the directory is gitignored),
        # and the scorer test skips on an empty glob — which would let this test
        # pass without exercising anything. Give it a real newest record.
        made_top_level = baselines / "zz-tfb-probe-run.json"
        made_top_level.write_text(json.dumps({
            "matches_production_defaults": True, "graph_rerank": True,
            "rerank_alpha": 0.3, "graph_top_k": 5, "graph_hops": 1,
            "note": "probe written by tests/test_tool_failure_baseline.py",
        }))
    shadow = shadow_dir / "zz-tfb-probe-shadow.json"
    try:
        shadow.write_text(json.dumps({"schema": 1, "tools": {}}))
        future = shadow.stat().st_mtime + 3600
        os.utime(shadow, (future, future))   # newer than every real baseline
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
             "tests/test_eval_scorer.py::test_a_run_record_declares_whether_it_matched_production"],
            cwd=ROOT, capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1500:]
        assert "1 passed" in proc.stdout, proc.stdout[-800:]
    finally:
        shadow.unlink(missing_ok=True)
        if made_top_level is not None:
            made_top_level.unlink(missing_ok=True)


# ── the acceptance invariant: this stays an offline read of logs ─────────────

def test_a_tool_results_sidecar_directory_is_not_read_as_transcripts(tmp_path):
    """`sessions/` holds a `<session-id>.tool-results/` directory per session
    whose truncated tool results are written back as JSON — ~2,200 files as this
    shipped, at least two flagged ones per directory. Recursing would count that
    backlog as calls; the glob has to stay `*.json`."""
    corpus = tmp_path / "sessions"
    _session(corpus, "a", _pairs(("Edit", EDIT_NO_MATCH, True)))
    sidecar = corpus / "a.tool-results"
    sidecar.mkdir(parents=True)
    (sidecar / "call-1.truncated.json").write_text(json.dumps(
        {"role": "tool", "content": "a whole tool result dumped to disk",
         "stats": {"is_error": True}}))
    (sidecar / "call-2.truncated.json").write_text(json.dumps(
        {"role": "tool", "content": "another one", "stats": {"is_error": True}}))

    rec = tfb.build_record(corpus, 21)
    assert rec["sessions_scanned"] == 1
    assert rec["tool_calls_total"] == 1
    assert rec["flagged_total"] == 1


def test_an_explicit_out_that_would_become_the_newest_top_level_baseline_is_refused(tmp_path):
    """The default sidesteps the scorer's mtime glob by being one level deeper;
    an explicit `--out` pointing AT the glob would turn that test red from a
    file it never mentions, so it is refused rather than warned about."""
    corpus = tmp_path / "sessions"
    _session(corpus, "a", _pairs(("Edit", EDIT_NO_MATCH, True)))
    collide = tfb.EVAL_BASELINES_DIR / "zz-tfb-collision-probe.json"
    try:
        assert tfb.main(["--sessions-dir", str(corpus), "--out", str(collide)]) == 2
        assert not collide.exists(), "the refusal still wrote the colliding file"
    finally:
        collide.unlink(missing_ok=True)
    assert not (tfb.EVAL_BASELINES_DIR / "zz-tfb-collision-probe.json").exists()


def test_nothing_outside_eval_reads_the_aggregate(tmp_path):
    """`eval/baselines/` is gitignored output. No per-run prompt may grow a
    field from it, so the module name appears nowhere in the two packages the
    running services import. Denominator first, positive control second — a
    0-hit grep over a corpus that is empty, or a pattern that cannot match
    anything, would otherwise read as the same answer."""
    sources = [*sorted((ROOT / "app").rglob("*.py")),
               *sorted((ROOT / "agent_mcp").rglob("*.py"))]
    assert len(sources) > 100, "corpus of service code vanished; this proves nothing"
    hits = [str(p.relative_to(ROOT)) for p in sources
            if "tool_failure_baseline" in p.read_text(errors="replace")]
    assert hits == []
    control = (ROOT / "eval" / "tool_failure_baseline.py").read_text()
    assert "tool_failure_baseline" in control  # the pattern is findable
