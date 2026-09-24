"""eval/artifact_named_in_final_message.py — the "where did this land?" metric (#1035).

Rows are built with `app.transcript_entries`, the module that writes
`sessions/*.json`, so a producer that renames a field turns these red rather
than leaving the metric reading zero.

Clause map (#1035 "How to verify"):
  1  every test here pins the classifier's pass/fail on synthetic sessions
  2  test_two_runs_over_one_tree_print_identical_numbers
  3  test_a_session_that_writes_only_bookkeeping_is_not_counted
  4  test_an_empty_or_placeholder_ending_is_excluded_and_owned_by_832
  5  test_a_filename_with_a_space_counts_as_named
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval.artifact_named_in_final_message as anf  # noqa: E402
from app import transcript_entries as te  # noqa: E402

TS = "2026-09-24T00:00:00"
PLACEHOLDER = ("*(Lloyd completed its tool calls but did not produce a summary. "
               "Ask again if you'd like an answer.)*")
# One of the two real legacy filenames with a literal U+0020 in it.
SPACED = ("knowledge/youtube/AI_Engineer/20260717-the-great-loops-debate- "
          "dex-horthy-geoff-huntley-ian-livingstone-greg-pstrucha-i.md")


def _vault(tmp_path: Path, *rels: str) -> Path:
    vault = tmp_path / "obsidian"
    for rel in rels:
        (vault / rel).parent.mkdir(parents=True, exist_ok=True)
        (vault / rel).write_text("# note\n")
    return vault


def _write(tool: str, target: str, cid: str = "c0") -> dict:
    key = "path" if tool == "vault_write" else "file_path"
    args = json.dumps({key: target, "content": "x"})
    return te.build_tool_call_entry(te.build_tool_call(cid, tool, args), timestamp=TS)


def _session(rows, final=None, **flags) -> dict:
    rows = list(rows)
    if final is not None:
        rows.append(te.build_assistant_text_entry(final, timestamp=TS, **flags))
    return {"session_id": "s", "source": "youtube-digest", "messages": rows}


def _file(corpus: Path, name: str, session: dict, age_days: float = 0.0) -> None:
    corpus.mkdir(parents=True, exist_ok=True)
    path = corpus / f"{name}.json"
    path.write_text(json.dumps(session))
    stamp = time.time() - age_days * 86400
    os.utime(path, (stamp, stamp))


def test_naming_an_existing_note_passes_and_a_missing_one_fails(tmp_path):
    vault = _vault(tmp_path, "knowledge/ai/foo.md", "projects/lloyd/bar.md")
    wrote = [_write("Write", f"{vault}/knowledge/ai/foo.md")]
    for final in ("Written to `~/obsidian/knowledge/ai/foo.md`.",
                  "Saved at /home/alansrobotlab/obsidian/knowledge/ai/foo.md",
                  "Note: knowledge/ai/foo.md — done."):
        assert anf.classify(_session(wrote, final), vault) == "pass", final
    assert anf.classify(_session(wrote, "Note written and verified."), vault) == "fail"
    assert anf.classify(_session(wrote, "See knowledge/ai/nope.md"), vault) == "fail"
    # vault_write's argument is vault-relative; projects/ is in scope too.
    rel = [_write("vault_write", "projects/lloyd/bar.md")]
    assert anf.classify(_session(rel, "At projects/lloyd/bar.md"), vault) == "pass"


def test_a_session_that_writes_only_bookkeeping_is_not_counted(tmp_path):
    vault = _vault(tmp_path, "skills/x/SKILL.md")
    for target in (f"{vault}/skills/x/SKILL.md", "~/obsidian/backlog/12.md",
                   "~/obsidian/autonomy/task-39.md", "~/obsidian/lloyd/SOUL.md",
                   "/home/alansrobotlab/lloyd/architecture/x.md"):
        s = _session([_write("Write", target)], "Updated the skill.")
        assert anf.classify(s, vault) == "out_of_scope", target
    corpus = tmp_path / "sessions"
    _file(corpus, "skill_only", _session([_write("Edit", f"{vault}/skills/x/SKILL.md")], "ok"))
    rec = anf.build_record(corpus, 7, vault)
    assert (rec["sessions_scanned"], rec["wrote_a_note"], rec["fail"]) == (1, 0, 0)


def test_an_empty_or_placeholder_ending_is_excluded_and_owned_by_832(tmp_path):
    vault = _vault(tmp_path, "knowledge/a.md")
    wrote = [_write("Write", f"{vault}/knowledge/a.md")]
    corpus = tmp_path / "sessions"
    _file(corpus, "tool_last", _session(wrote))  # ends on the tool-call row
    _file(corpus, "blank", _session(wrote, "  \n"))
    _file(corpus, "placeholder", _session(wrote, PLACEHOLDER, synthetic_empty_terminal=True))
    _file(corpus, "legacy_placeholder", _session(wrote, PLACEHOLDER))
    rec = anf.build_record(corpus, 7, vault)
    assert rec["empty_terminal"]["count"] == 4
    assert rec["empty_terminal"]["owner"] == "#832"
    assert (rec["pass"], rec["fail"]) == (0, 0), "an empty ending was charged to this metric"


def test_a_filename_with_a_space_counts_as_named(tmp_path):
    vault = _vault(tmp_path, SPACED)
    s = _session([_write("Write", f"{vault}/{SPACED}")],
                 f"Note written: `~/obsidian/{SPACED}` — evaluation below.")
    assert anf.classify(s, vault) == "pass"
    s = _session([_write("Write", f"{vault}/{SPACED}")], f"Wrote {SPACED}, then stopped.")
    assert anf.classify(s, vault) == "pass"


def test_two_runs_over_one_tree_print_identical_numbers(tmp_path, capsys):
    vault = _vault(tmp_path, "knowledge/a.md")
    wrote = [_write("Write", f"{vault}/knowledge/a.md")]
    corpus = tmp_path / "sessions"
    _file(corpus, "named", _session(wrote, "knowledge/a.md"), age_days=1)
    _file(corpus, "silent", _session(wrote, "done"), age_days=2)
    _file(corpus, "stale", _session(wrote, "done"), age_days=30)
    (corpus / "named.tool-results").mkdir()  # spill dir, not a transcript
    argv = ["--sessions-dir", str(corpus), "--vault", str(vault), "--json"]
    assert anf.main(argv) == 0
    first = capsys.readouterr().out
    assert anf.main(argv) == 0
    assert capsys.readouterr().out == first
    rec = json.loads(first)
    assert (rec["sessions_scanned"], rec["pass"], rec["fail"]) == (2, 1, 1)
    assert rec["failures"] == [{"session": "silent", "source": "youtube-digest"}]
