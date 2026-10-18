"""The non-round owner of the find-all arms run (#2962): lock-waiting launcher, loud refusals.

Why an owner that is not a round. ``eval/run_find_all_arms.py`` has been fixed-but-unused
since ``244ae542`` shipped on 2026-10-15 (#2741): the only artifact on the box is still
``eval/measurements/find-all-2026-09-24/`` (mtime Sep 29), whose 20 rows all lack
``turn_budget``, so #647's phrasing clause — ``recall(all) > recall(one)`` over answered
trials on #014 and #016 at n>=3 — has never been measurable. An automod round cannot hold
the engine lock and run a harness about itself, so the owner is this launcher plus a skill
plus one autonomy task that fires it detached.

What this file pins, clause by clause (item #2962's acceptance):

1. A run that wrote no ``trials.jsonl`` exits NON-ZERO printing WHICH precondition stopped
   it — the lock still held when the bounded wait ended, the driver's tool-sandbox refusal,
   or a gold mismatch. Never exit 0 having run nothing.
2. ``--dry-run`` prints the resolved command (venv interpreter, the driver, ``--trials 3
   --max-turns 25 --timeout 900``, ``--out eval/measurements/find-all-<UTC date>``), exits
   0, and never opens ``~/.local/state/lloyd-automod/primary.lock``.
3. Before any trial is spawned the launcher re-derives each of the four gold counts outside
   pytest, prints derived beside declared per task, and stops non-zero where they differ.
4. A summarise step reaches the table through the driver's existing ``--summarise`` with no
   engine call and writes ``eval/measurements/find-all-<date>.md`` holding BOTH the
   per-{arm}/{task} table and the ``judge_independence`` disagree-rate block.
5. That ``.md`` prints the at-cap accounting per cell and NO ``recall(all)>recall(one)``
   comparison: the #647 ruling is a reading, not a print.

Two things this file deliberately does NOT assert, because they are owed after landing and
no round can grade them: that the detached run survives its launching session's teardown,
and the ``recall(all) > recall(one)`` ruling itself.

Everything here runs offline. The real driver is never asked to spawn a session: ``--driver``
points at a stand-in that writes rows or writes nothing, and the summarise subprocess is the
real driver reading a fixture ``trials.jsonl`` — zero engine time by construction. The gold
re-derivation is exercised against a fixture vault, never the live one, because
``tests/conftest.py``'s ``_production_tree`` guard exists precisely to keep a test from
measuring the tree it runs in.
"""
from __future__ import annotations

import fcntl
import json
import re
import shlex         # the launcher prints its command with shlex.join; a test reads it back
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml          # the fixture bench front matter is written by the lib that parses it

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "find_all_arms_run.py"

#: The task id a REAL trial row carries. ``eval/run_find_all_arms.py:127`` writes
#: ``"task_id": task["id"]``, and ``task["id"]`` is the bench note's own id —
#: ``bench_014_audit_dead_wikilinks``, not ``#014`` — so the report table a real run publishes
#: is keyed by these strings. A fixture keyed ``#014`` would only ever exercise the pretty
#: spelling of a cell key, and the launcher's cell/row join would go untested at the shape
#: production actually produces.
REAL_TID = "bench_014_audit_dead_wikilinks"

#: The #016 note's own id, for a fixture that needs the pair: #647's phrasing clause is read
#: over #014 AND #016, so the report shape worth pinning is the one only those two rows make.
REAL_TID_016 = "bench_016_audit_skill_dead_paths"

sys.path.insert(0, str(REPO_ROOT))


# ── fixtures: a bench, a vault, a skills tree and an autonomy dir ─────────────

def _bench_task(note_id: str, gold: list) -> str:
    """One bench note, keyed exactly the way the four real notes are keyed.

    Real, not convenient: ``~/obsidian/lloyd/bench/bench_014_audit_dead_wikilinks.md`` and its
    three siblings identify themselves with ``id: bench_014_...`` and carry **no** ``task_id``
    key (``grep -c '^task_id:'`` over the four → ``0`` each, 2026-10-18). A fixture written with
    ``task_id: '#014'`` would let a launcher that only reads ``task_id`` pass every node here
    and then refuse its first real fire, because the real notes hand it a stem it matches
    against nothing. The checks sit under ``objective_checks``, the key
    ``scripts/autoresearch/common.load_bench_tasks`` hands back.
    """
    check = {"type": "find_all", "value": "gold_member", "dedupe_key": "regex",
             "dedupe_regex": "(x)", "gold_items": gold}
    return ("---\nsegment: lloyd\nid: %s\ncategory: synthetic\nobjective_checks:\n%s---\nBody.\n"
            % (note_id, yaml.safe_dump([check], sort_keys=False, width=10**6)))


def _fixture_vault(root: Path) -> Path:
    """A vault whose four answer keys are 3, 2 and 1 — and where each is NOT zero.

    Every fixture answer is nonzero on purpose. A derivation that always returns 0 passes
    a zero-item gold set for the wrong reason, which is the #017 control's whole lesson.

    * #014 = 3: ``projects/`` notes link ``[[knowledge/gone-one]]``, ``[[knowledge/gone-two]]``
      and the bare ``[[bare-missing]]``; ``[[knowledge/real]]`` resolves, ``[[#heading]]`` and
      a link inside a code span do not count, and ``bare-missing`` is linked from two notes
      yet is one answer.
    * #016 = 2: the active skill cites ``~/lloyd/scripts/gone.py`` and
      ``/home/alansrobotlab/lloyd/scripts/gone.py`` (one key, so one finding) plus
      ``~/lloyd/data/gone2.json``, while ``~/lloyd/scripts/exists.py`` exists,
      ``~/lloyd/docs/<date>.md`` is templated, ``~/lloyd/configs/`` is a directory, and a
      skill under ``.archived/`` is skipped.
    * #017 = 1: ``10-alpha.md`` names the ``alpha`` skill which exists; ``11-beta.md`` names
      ``beta``, whose SKILL.md does not; an unnumbered note is out of scope.
    """
    (root / "projects" / "sub").mkdir(parents=True)
    (root / "projects" / "sub" / "a.md").write_text(
        "[[knowledge/gone-one]] and [[knowledge/gone-two]] and [[bare-missing]].\n"
        "Real: [[knowledge/real]]. Heading: [[#Section]]. "
        "In code: `[[knowledge/in-code]]`.\n", encoding="utf-8")
    (root / "projects" / "b.md").write_text("again [[bare-missing]] and [[knowledge/real]]\n",
                                            encoding="utf-8")
    (root / "knowledge").mkdir(parents=True)
    (root / "knowledge" / "real.md").write_text("a real note\n", encoding="utf-8")

    skill = root / "skills" / "alpha" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "Run `~/lloyd/scripts/gone.py`, or `/home/alansrobotlab/lloyd/scripts/gone.py` "
        "same file. Also `~/lloyd/data/gone2.json`. Fine: `~/lloyd/scripts/exists.py`. "
        "Templated: `~/lloyd/docs/<date>.md`. Dir: `~/lloyd/configs/`.\n", encoding="utf-8")
    archived = root / "skills" / ".archived" / "old" / "SKILL.md"
    archived.parent.mkdir(parents=True)
    archived.write_text("`~/lloyd/scripts/archived-gone.py`\n", encoding="utf-8")

    (root / "autonomy").mkdir(parents=True)
    (root / "autonomy" / "10-alpha.md").write_text(
        "---\nname: alpha task\nskill_name: alpha\nstatus: draft\n---\nBody.\n",
        encoding="utf-8")
    (root / "autonomy" / "11-beta.md").write_text(
        "---\nname: beta task\nskill_name: beta\nstatus: draft\n---\nBody.\n",
        encoding="utf-8")
    (root / "autonomy" / "unnumbered.md").write_text(
        "---\nname: not a task\nskill_name: nowhere\n---\nBody.\n", encoding="utf-8")
    return root


def _fixture_home(root: Path) -> Path:
    """A home whose `lloyd/` tree holds exactly the one path the fixture skill cites as live."""
    live = root / "lloyd" / "scripts"
    live.mkdir(parents=True)
    (live / "exists.py").write_text("print(1)\n", encoding="utf-8")
    return root


def _fixture_bench(root: Path, *, n014: int = 3, n015: int = 3, n016: int = 2,
                   n017: int = 1) -> Path:
    """A bench dir holding the four find_all tasks, each declaring the gold size asked.

    A non-default ``n014`` is how a test makes DECLARED disagree with DERIVED: the fixture
    vault above says 3 and nothing else, so asking for 4 is a drifted answer key.
    """
    root.mkdir(parents=True, exist_ok=True)
    #: Plain strings, as the real notes spell them: #014's gold items are vault paths, #016's
    #: are paths under `lloyd/`, #017's are note names, #015's are 16-hex text_hash values.
    #: None of them is a mapping, and `declared` is just their length.
    gold = lambda n, fmt: [fmt.format(i=i) for i in range(n)]
    (root / "bench_014_audit_dead_wikilinks.md").write_text(
        _bench_task("bench_014_audit_dead_wikilinks", gold(n014, "knowledge/dead-{i}")),
        encoding="utf-8")
    (root / "bench_015_audit_cross_entity_fact_copies.md").write_text(
        _bench_task("bench_015_audit_cross_entity_fact_copies", _copy_hashes()[:n015]),
        encoding="utf-8")
    (root / "bench_016_audit_skill_dead_paths.md").write_text(
        _bench_task("bench_016_audit_skill_dead_paths", gold(n016, "scripts/dead-{i}.py")),
        encoding="utf-8")
    (root / "bench_017_audit_unresolved_task_skills.md").write_text(
        _bench_task("bench_017_audit_unresolved_task_skills",
                    gold(n017, "{i}-naming-a-missing-skill.md")),
        encoding="utf-8")
    return root


_COPY_CLAIMS = ("shared claim zero", "shared claim one", "shared claim two")


def _facts_fixture(root: Path) -> Path:
    """A derived-fact tree holding each ``#015`` claim under TWO entities, so it is a copy.

    #015's gold items are ``text_hash`` values and its answer is "this hash is held by two or
    more distinct entities", so the offline nodes need a walkable tree: the default root is the
    real ``~/obsidian/facts`` (41,759 files), and pointing a test at it would make the node a
    minute-long measurement of live ground truth rather than of the launcher.
    """
    for i, claim in enumerate(_COPY_CLAIMS):
        for entity in ("Alpha", "Beta"):
            p = root / entity / f"note-{i}.md"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("---\ntype: fact\nentity: %s\ncategory: overview\nfacts:\n"
                         "- fact: %s\n---\n\n# %s\n" % (entity, claim, entity),
                         encoding="utf-8")
    return root


def _copy_hashes() -> list[str]:
    """The gold items #015 declares for `_facts_fixture`: the store's own hash of each claim."""
    from app.kg_store import text_hash
    return [text_hash(c) for c in _COPY_CLAIMS]


def _fixture_args(tmp_path: Path) -> list[str]:
    """The four overrides every offline run needs: bench, vault, ``~`` for path checks, and
    a fact tree for #015's copy count."""
    return ["--bench-dir", str(_fixture_bench(tmp_path / "bench")),
            "--vault-root", str(_fixture_vault(tmp_path / "vault")),
            "--home-root", str(_fixture_home(tmp_path / "home")),
            "--facts-root", str(_facts_fixture(tmp_path / "facts"))]


def _trials_fixture(path: Path) -> None:
    """Rows shaped as ``run_find_all_arms.run_trial`` writes them, post-``244ae542``.

    Two arms over one task: one answered trial, one capped trial and one timeout, so the
    table has a nonzero ``n_truncated`` and ``n_timeout`` to land; and two ``all`` trials
    whose ``rubric_status`` is ``"ok"`` and whose rubric and P×R fall on opposite sides of
    0.5, so the judge-disagreement block has a rate to print.
    """
    rows = [
        {"task_id": REAL_TID, "trial": 0, "arm": "all", "status": "ok", "session_id": "s1",
         "duration_s": 10.0, "turns": 9, "turn_budget": 25, "turn_truncated": False,
         "timeout_budget": 900, "precision": 1.0, "recall": 1.0, "f1": 1.0, "items": {},
         "objective": 1.0, "rubric": 0.2, "rubric_status": "ok", "composite": 0.64,
         "final_text": "a" * 40, "tool_calls": 3, "denied_calls": 0,
         "bench_probe_count": 0},
        {"task_id": REAL_TID, "trial": 1, "arm": "all", "status": "ok", "session_id": "s2",
         "duration_s": 12.0, "turns": 25, "turn_budget": 25, "turn_truncated": True,
         "timeout_budget": 900, "precision": 0.0, "recall": 0.0, "f1": 0.0, "items": {},
         "objective": 0.0, "rubric": 0.9, "rubric_status": "ok", "composite": 0.45,
         "final_text": "b" * 40, "tool_calls": 1, "denied_calls": 0,
         "bench_probe_count": 0},
        {"task_id": REAL_TID, "trial": 2, "arm": "one", "status": "timeout",
         "session_id": "s3", "duration_s": 900.0, "turns": 25, "turn_budget": 25,
         "turn_truncated": True, "timeout_budget": 900, "final_text": "", "tool_calls": 2,
         "denied_calls": 0, "bench_probe_count": 0},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _gap_trials(path: Path) -> list[dict]:
    """Rows in which `all` strictly beats `one` on recall over the ANSWERED trials.

    The shape #647's clause is ruled on, built deliberately rather than stumbled into: the two
    tasks the clause names (#014 and #016), two answered trials in every cell, every `all`
    recall above every `one` recall, and NO truncated or timed-out row — so nothing but the
    phrasing separates the arms, and a renderer that meant to state a verdict is handed one.
    Returned as well as written, so a node can ask the driver the same question about the same
    rows the launcher rendered.

    Two honesty notes on the numbers. Each row's `precision`, `recall` and `f1` are set equal
    to each other, so the P/R/F1 columns print one level per cell and no assertion here depends
    on F1's formula; the means that follow are exact at two decimals (0.75, 0.80, 0.20, 0.20).
    And `items` is EMPTY on every row while the score fields are nonzero and unequal between
    cells — the input a re-implementation would have to be written against: derive a score from
    `items` and the P column prints `0.00` or `-` instead of the means asserted.

    `rubric_status` is `"ok"` because that is the value the judge emits for a graded trial
    (`rubric_status`, `scripts/autoresearch/judge.py:181-191`; the other possible values are
    `"skipped"`, `"not_scored"` at `:841`, or a `RUBRIC_FAILURES` entry named at `:110`), and
    `summarise` admits a row to the judge block only when the status is exactly `"ok"`
    (`eval/run_find_all_arms.py:284`). Spell it anything else and `n` collapses to 0 with both
    headings still printed — the false green an earlier draft of this file carried.
    """
    rows = [
        {"task_id": REAL_TID, "trial": 0, "arm": "all", "objective": 1.0, "rubric": 1.0,
         "p": 1.0},
        {"task_id": REAL_TID, "trial": 1, "arm": "all", "objective": 1.0, "rubric": 0.2,
         "p": 0.5},
        {"task_id": REAL_TID, "trial": 0, "arm": "one", "objective": 0.0, "rubric": 0.1,
         "p": 0.2},
        {"task_id": REAL_TID, "trial": 1, "arm": "one", "objective": 0.0, "rubric": 0.9,
         "p": 0.2},
        {"task_id": REAL_TID_016, "trial": 0, "arm": "all", "objective": 0.0, "rubric": 0.9,
         "p": 0.6},
        {"task_id": REAL_TID_016, "trial": 1, "arm": "all", "objective": 1.0, "rubric": 0.1,
         "p": 1.0},
        {"task_id": REAL_TID_016, "trial": 0, "arm": "one", "objective": 0.0, "rubric": 0.6,
         "p": 0.1},
        {"task_id": REAL_TID_016, "trial": 1, "arm": "one", "objective": 0.0, "rubric": 0.6,
         "p": 0.3},
    ]
    out = []
    for i, r in enumerate(rows):
        out.append({"task_id": r["task_id"], "trial": r["trial"], "arm": r["arm"],
                    "status": "ok", "session_id": f"s{i}", "duration_s": 30.0 + i,
                    "turns": 8 + (i % 3), "turn_budget": 25, "turn_truncated": False,
                    "timeout_budget": 900, "precision": r["p"], "recall": r["p"],
                    "f1": r["p"], "items": {}, "objective": r["objective"],
                    "rubric": r["rubric"], "rubric_status": "ok",
                    "composite": round((r["objective"] + r["rubric"]) / 2, 3),
                    "final_text": "found them\n" * 4, "tool_calls": 3, "denied_calls": 0,
                    "bench_probe_count": 0})
    path.write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    return out


def _run(args: list[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(LAUNCHER), *args], capture_output=True,
                          text=True, timeout=180, cwd=str(cwd))


def _fake_driver(root: Path, *, writes_rows: bool, exit_code: int = 0,
                 stderr: str = "") -> Path:
    """A stand-in for ``eval/run_find_all_arms.py``: writes rows, or writes nothing.

    Its whole job is to make "no trials.jsonl was written" reachable without eighteen
    harness sessions, and to record the argv it was given so a test can prove the launcher
    spawns what it printed.
    """
    p = root / "fake_driver.py"
    marker = root / "spawned.json"
    body = f"""import json, sys, pathlib
argv = sys.argv[1:]
if "--summarise" not in argv:
    #: Only the RUN invocation is recorded. The launcher re-enters this driver with
    #: --summarise afterwards, and an unguarded marker would be overwritten by that
    #: second call, so every test reading the marker would see the summarise argv.
    pathlib.Path({str(marker)!r}).write_text(
        json.dumps([sys.executable, sys.argv[0]] + argv))
{f"sys.stderr.write({stderr!r})" if stderr else "pass"}
out = None
for i, a in enumerate(argv):
    if a == "--out":
        out = pathlib.Path(argv[i + 1])
if "--summarise" in argv:
    #: Answer the summarise call with the REAL driver's summarise over the rows this driver
    #: wrote, so the run's tail (table and judge block) is exercised without an engine and no
    #: summary number is faked: the rows on disk are the only source.
    import importlib.util as _u
    _spec = _u.spec_from_file_location("_d0", "{REPO_ROOT}/eval/run_find_all_arms.py")
    _mod = _u.module_from_spec(_spec); _spec.loader.exec_module(_mod)
    print(json.dumps(_mod.summarise([json.loads(l) for l in (out / "trials.jsonl").open()])))
    raise SystemExit(0)
if {writes_rows!r} and out is not None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "trials.jsonl").write_text("".join(
        json.dumps({{"task_id": {REAL_TID!r}, "trial": i, "arm": "all", "status": "ok",
                    "turns": 9, "turn_budget": 25, "turn_truncated": False,
                    "precision": 1.0, "recall": 1.0, "f1": 1.0, "objective": 1.0,
                    "rubric": 1.0, "rubric_status": "ok", "composite": 1.0,
                    "final_text": "x" * 50}}) + "\\n" for i in range(3)))
"""
    if exit_code:
        body += f"sys.exit({exit_code!r})\n"
    p.write_text(body, encoding="utf-8")
    return p


# ── clause 2: --dry-run prints the resolved command and never opens the lock ──

def test_dry_run_prints_the_resolved_command_and_exits_zero():
    """The printed command is the whole skill body, so it must be complete and exact."""
    proc = _run(["--dry-run"])
    assert proc.returncode == 0, proc.stderr[-800:]
    out = proc.stdout
    assert str(REPO_ROOT / "eval" / "run_find_all_arms.py") in out, out
    assert "--trials 3" in out, out          # n>=3 is what #647's clause needs
    assert "--max-turns 25" in out, out      # not the bench default's 12
    assert "--timeout 900" in out, out       # the driver's own default is 600 (#2160)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert f"eval/measurements/find-all-{today}" in out, out


def test_dry_run_never_opens_the_engine_lock(tmp_path):
    """``flock`` CREATES its file, so an absent lock path afterwards is the witness.

    The one clause that must hold absolutely: a dry run that touched the lock would block
    the engine for whoever is measuring.
    """
    lock = tmp_path / "primary.lock"
    proc = _run(["--dry-run", "--lock", str(lock)])
    assert proc.returncode == 0, proc.stderr[-800:]
    assert not lock.exists(), "a dry run opened the engine lock it was only supposed to name"


# ── clause 3: the gold counts, re-derived and printed before any trial ────────

def test_every_gold_count_is_derived_beside_declared(tmp_path):
    """Each derived count comes from the tree, not from the length of gold_items."""
    import scripts.find_all_arms_run as L

    rows = L.gold_checks(_fixture_bench(tmp_path / "bench"),
                         vault_root=_fixture_vault(tmp_path / "vault"),
                         home_root=_fixture_home(tmp_path / "home"),
                         store_lookup=lambda h: True)
    assert [r.task_id for r in rows] == ["#014", "#015", "#016", "#017"], rows
    assert [r.derived for r in rows] == [3, 3, 2, 1], rows       # what the fixture tree says
    assert [r.declared for r in rows] == [3, 3, 2, 1], rows
    assert all(r.ok for r in rows), [r.render() for r in rows]
    printed = "\n".join(r.render() for r in rows)
    assert "derived=3 declared=3" in printed, printed
    assert "derived=1 declared=1" in printed, printed


def test_the_dead_wikilink_count_dedupes_a_target_linked_from_two_notes(tmp_path):
    """``bare-missing`` is linked from both project notes and is ONE answer.

    A derivation that counted mentions would print 4 where the bench says 3, refuse the
    run, and send whoever reads it to fix a gold set that was right.
    """
    import scripts.find_all_arms_run as L

    assert L._derive_dead_wikilinks(_fixture_vault(tmp_path / "vault")) == 3


def test_the_skill_path_count_folds_two_spellings_of_one_path(tmp_path):
    """``~`` and ``/home/alansrobotlab`` are one key, a templated path is no key, a
    directory is not a file, and a skill under ``.archived`` is not active."""
    import scripts.find_all_arms_run as L

    vault = _fixture_vault(tmp_path / "vault")
    assert L._derive_dead_skill_paths(vault, home_root=_fixture_home(tmp_path / "home")) == 2


def _example_vault(root: Path) -> Path:
    """A vault with one dead path a skill USES and one dead path a skill cites as a wrong example.

    Mirrors the live shape that decides #016: `file-path-resolution` line 25 shows
    `/home/alansrobotlab/lloyd/lloyd/inner-voice/system-prompt.md` under
    `### Pattern 1: Guessing Wrong File Paths`, and the bench note records it under "Not gold,
    deliberately" with a hand-checked `judge.grade_answer_set` NEGATIVE case beside it.
    """
    for slug, body in (
            ("uses-it", "## Known Path Patterns\n\n"
                        "- Run `~/lloyd/scripts/used.py` first.\n"),
            ("wrong-example", "## Common Failure Patterns\n\n"
                              "### Pattern 1: Guessing Wrong File Paths\n"
                              "- **Common mistakes:**\n"
                              "  - Double directory names: "
                              "`/home/alansrobotlab/lloyd/lloyd/inner-voice/system-prompt.md`\n"),
            ("cited-both", "## Common Failure Patterns\n\n"
                           "### Pattern 1: Guessing Wrong File Paths\n"
                           "  - Double directory names: `~/lloyd/scripts/used.py`\n\n"
                           "## Known Path Patterns\n\n"
                           "- Run `~/lloyd/scripts/used.py` first.\n")):
        d = root / "skills" / slug
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(body, encoding="utf-8")
    return root


def test_a_dead_path_cited_as_an_example_of_a_wrong_path_is_not_counted_as_gold(tmp_path):
    """#016's prompt says to skip them, so the derivation must too or it disagrees with its own label.

    A bare "does the file exist" walk over `~/lloyd/...` citations derives 5 on the live box
    against a bench declaring 4 (measured 2026-10-18), and the extra is exactly the note's
    not-gold case — so the launcher would REFUSE a fire whose answer key is correct, which is
    the guard misfiring on the thing it exists to protect. The exclusion is the enclosing
    HEADING, because line 25 carries no marker word on its own line.
    """
    import scripts.find_all_arms_run as L

    vault = _example_vault(tmp_path / "vault")
    missing, skipped = L._dead_skill_path_citations(vault, home_root=tmp_path / "home")
    assert missing == {"scripts/used.py"}, sorted(missing)
    assert [(s[0], s[2]) for s in skipped] == [
        ("wrong-example", "lloyd/inner-voice/system-prompt.md")], skipped
    assert "Pattern 1: Guessing Wrong File Paths" in skipped[0][3], skipped
    #: A path that is ALSO cited as an instruction somewhere is a finding, exclusion or not —
    #: `cited-both` names `scripts/used.py` under a wrong-path heading and under a real one.
    assert L._derive_dead_skill_paths(vault, home_root=tmp_path / "home") == 1


# ── clause 5: the two VAULT surfaces, pinned from this file ───────────────────
#
# Deliberately UNMARKED, though they read the live vault. `pytest.ini:14` defines
# `live_vault` and `scripts/automod/gate.py` runs the hard rung with `-m "not
# live_vault"`, so a marked assertion about the skill and the task file is never run by
# the one rung that can block a landing — which is exactly how the round that first
# built this owner lost clause 5 at review: the vault files exist, and nothing in the
# diff pinned them. The precedent for paying the coupling instead of the blindness is
# `tests/test_archived_skill_artifacts.py:30-41`; the assertions are structural (does
# the command exist, does the slug resolve) rather than quotes of prose a nightly job
# may legitimately rewrite. `Path.home()` is the round home under the gate, and
# `<round home>/obsidian` is a symlink to the real vault: the `tests` rung runs with
# `isolate_home=True` (`scripts/automod/gate.py:1495-1498` → `W.ensure_round_home`), and
# that farm symlinks EVERY home entry except `HOME_LINK_SKIP = {"lloyd", "lloyd-data"}`
# (`scripts/automod/worktree.py:65`, `:99-108`) — `obsidian` is not in the skip set, so
# the vault is present by construction wherever this file is enforced. Which is why
# `_live_vault` below asserts instead of skipping: there is no run of this file, gate or
# bench, in which the vault is legitimately absent, and a skip on clause 5's only pin is
# how a missing owner reads as a pass.

LIVE_VAULT = Path.home() / "obsidian"
SKILL_MD = LIVE_VAULT / "skills" / "find-all-arms-run" / "SKILL.md"
#: The four answer keys, spelled the way the launcher spells its own default
#: (`BENCH_DIR`, `scripts/find_all_arms_run.py:91`) so a node that asks #017's control is
#: asking the same directory the real fire will ask.
L_BENCH_DIR = LIVE_VAULT / "lloyd" / "bench"


def _live_vault() -> Path:
    """The real vault, asserted to be there — never skipped.

    The distinction that matters is between "this machine has no vault" and "the vault is
    here and the owner is missing". The second is the failure clause 5 exists to catch, and
    the first cannot happen on any runner that enforces this file (see the block above), so
    the guard is an assert. A vault without a `skills/` dir is a broken vault, and a broken
    vault is exactly what these nodes should refuse to walk past.
    """
    assert (LIVE_VAULT / "skills").is_dir(), (
        f"no vault at {LIVE_VAULT} — clause 5's owner lives in {SKILL_MD} and "
        f"~/obsidian/autonomy/, and nothing here can check them")
    assert L_BENCH_DIR.is_dir(), (
        f"no bench dir at {L_BENCH_DIR} — the #017 control below needs its answer key")
    return LIVE_VAULT


def test_the_vault_skill_holds_one_launch_command_naming_the_lock_and_the_cap():
    """Clause 5's skill half: one command, the lock it waits on, the 3570 s reason.

    Breaking inputs: the skill deleted; the fenced command rewritten so it no longer
    names `setsid`+`nohup` (a run that dies with its session); the lock path dropped (a
    reader then blames the wrong lock, which is what happened for a week); the pool cap
    dropped, which is what invites a `timeout_seconds >= 5400` task — the bug
    `~/obsidian/autonomy/81-qmd-index-maintenance.md` records fixing on 2026-09-03.
    """
    _live_vault()
    assert SKILL_MD.is_file(), f"{SKILL_MD} is absent: the owner lost its skill"
    text = SKILL_MD.read_text(encoding="utf-8", errors="replace")
    front = text.split("---")[1] if text.startswith("---") else ""
    assert re.search(r"^name:\s*find-all-arms-run\s*$", front, re.M), front
    blocks = re.findall(r"```(?:bash|sh)?\n(.*?)```", text, re.S)
    launches = [b for b in blocks
                if "scripts/find_all_arms_run.py" in b and "--summarise-only" not in b]
    assert len(launches) == 1, (
        f"clause 5 asks for ONE launch command; found {len(launches)}")
    cmd = launches[0]
    assert "setsid" in cmd and "nohup" in cmd, cmd
    assert "find_all_arms_run.py" in cmd, cmd
    assert "~/.local/state/lloyd-automod/primary.lock" in text, \
        "the skill must name the lock the launcher waits on"
    assert "3570" in text, "the skill must state why the run cannot live in a session"
    #: The pool cap is quoted with its source, or the next reader re-invents 5400.
    assert re.search(r"max_duration\s*[-−]\s*30\s*=\s*3570", text), text[:1500]


def test_the_autonomy_task_fires_that_skill_inside_the_pool_cap():
    """Clause 5's task half: it names the skill, stays under 3000 s, and says why.

    `timeout_seconds` is asserted `<= 3000` because the scheduled-task pool clamps a run
    to `max_duration - 30 = 3570 s` (`workers/sources/scheduled_task.py:415`,
    `app/autonomy.py:399` + `:4576`), and a run cancelled by the pool produces NO run
    record at all — the 2026-09-03 entry in `81-qmd-index-maintenance.md` cut 5700 to
    3000 for that reason. A 1.5-2.5 h arms run therefore has to be fired detached, which
    is what the description says to do.
    """
    vault = _live_vault()
    hits = []
    for f in sorted((vault / "autonomy").glob("*.md")):
        parts = f.read_text(encoding="utf-8", errors="replace").split("---")
        try:
            fm = yaml.safe_load(parts[1]) or {}
        except yaml.YAMLError:
            continue
        if isinstance(fm, dict) and fm.get("skill_name") == "find-all-arms-run":
            hits.append((f, fm))
    assert len(hits) == 1, [str(f) for f, _ in hits]
    path, fm = hits[0]
    timeout = fm.get("timeout_seconds")
    assert isinstance(timeout, int), f"{path.name} timeout_seconds={timeout!r}"
    assert timeout <= 3000, f"{path.name} timeout_seconds={timeout} exceeds the pool cap"
    #: Parsed with yaml, not one regex line: this description is a folded scalar, and the
    #: lock sits on its second line. A line-matching check would go red on a file that
    #: says exactly what clause 5 asks, and prose a guard cannot read is prose nobody
    #: keeps. `description` plus the SKILL.md is ALL `_build_task_prompt` delivers.
    assert "primary.lock" in str(fm.get("description", "")), (
        f"{path.name}'s description must name the lock — it is the only text besides "
        f"the SKILL.md the worker is ever given: {fm.get('description')!r}")
    #: Resolution, by the rule #017 uses: a slug whose SKILL.md is not there loads as
    #: nothing, so the task would run with no procedure at all.
    assert (vault / "skills" / "find-all-arms-run" / "SKILL.md").is_file(), path.name


def test_the_live_zero_item_control_still_resolves_the_new_task_skill(tmp_path):
    """The clause's own pin, run here instead of in the deselected `live_vault` file.

    `tests/test_bench_audit_tasks.py::test_the_zero_item_control_is_still_zero` is what
    clause 5 names, and the gate deselects it. This node asks the same question of the
    same rule — #017's derivation counts `autonomy/NN-*.md` whose `skill_name` has no
    `skills/<slug>/SKILL.md` — through the launcher's code, over the real vault, and only
    for `#017` so it does not also walk 41,759 fact files. `derived=0 declared=0` is the
    witness that this round's new task did not break the control; a task naming a slug
    that does not resolve moves `derived` to 1 and reddens the hard rung.
    """
    vault = _live_vault()
    import scripts.find_all_arms_run as L

    rows = L.gold_checks(L.BENCH_DIR, vault_root=vault, only={"#017"})
    assert len(rows) == 1, [r.render() for r in rows]
    row = rows[0]
    assert (row.task_id, row.derived, row.declared) == ("#017", 0, 0), row.render()


def test_the_zero_control_count_is_derived_not_assumed(tmp_path):
    """#017 declares 0 on the live box; the derivation is a real scan, so a planted
    unresolved ``skill_name`` reads as 1 rather than being trusted at zero.
    """
    import scripts.find_all_arms_run as L

    vault = _fixture_vault(tmp_path / "vault")
    assert L._derive_unresolved_task_skills(vault) == 1
    (vault / "skills" / "beta").mkdir(parents=True)
    (vault / "skills" / "beta" / "SKILL.md").write_text("now it exists\n", encoding="utf-8")
    assert L._derive_unresolved_task_skills(vault) == 0, "the control is not a scan"


def test_a_gold_mismatch_stops_the_run_before_anything_is_spawned(tmp_path):
    """DECLARED 4 vs DERIVED 3 on #014 refuses, and the driver is never reached.

    The 2026-09-24 report's own closing line is why this gates the run — "re-label on
    rot: #015 changes with every extraction run; the live test flags #014/#016/#017" — and
    a drifted gold scores every arm wrong identically, which reads as a null result
    instead of a broken answer key.
    """
    driver = _fake_driver(tmp_path, writes_rows=True)
    bench = _fixture_bench(tmp_path / "bench", n014=3)
    vault, home = _fixture_vault(tmp_path / "vault"), _fixture_home(tmp_path / "home")
    facts = _facts_fixture(tmp_path / "facts")
    args = ["--bench-dir", str(bench), "--vault-root", str(vault), "--home-root", str(home),
           "--driver", str(driver), "--out", str(tmp_path / "out"),
           "--lock", str(tmp_path / "primary.lock")]

    #: CONTROL FIRST, inside this node. `spawned.json` is written by the fake driver, so its
    #: absence on the refusal path proves the launcher never reached spawn ONLY if the same
    #: marker is known to appear on a path that does — otherwise an unreadable fixture would
    #: satisfy the node by never running at all, which is the same false green as a grep
    #: against a path that does not exist. Gold matching: exit 0, marker written.
    ok = _run(args + ["--facts-root", str(facts)])
    assert ok.returncode == 0, ok.stdout[-500:] + ok.stderr[-500:]
    assert (tmp_path / "spawned.json").exists(), \
        "the fake driver did not run at all, so the negative below would prove nothing"
    #: And the four keys were DERIVED, not skipped: `#015 = 3` is only reachable by walking
    #: `facts_fixture` (an absent or empty tree is a refusal, and a tree with no shared hash
    #: derives 0), and `#016 = 2` is only reachable by resolving the skill's citations against
    #: `home` — the fixture that brings `exists.py` to life. Without these two lines the
    #: refusal below could be produced by a launcher that never read either tree.
    for line in ("gold #014: derived=3 declared=3 ok", "gold #015: derived=3 declared=3 ok",
                 "gold #016: derived=2 declared=2 ok", "gold #017: derived=1 declared=1 ok"):
        assert line in ok.stdout, (line, ok.stdout[-900:])

    #: Now drift #014's label by one and re-run against the same everything.
    (bench / "bench_014_audit_dead_wikilinks.md").unlink()
    (bench / "bench_014_audit_dead_wikilinks.md").write_text(
        _bench_task("bench_014_audit_dead_wikilinks",
                    [f"knowledge/dead-{i}" for i in range(4)]), encoding="utf-8")
    (tmp_path / "spawned.json").unlink()
    proc = _run(args + ["--facts-root", str(facts)])
    assert proc.returncode != 0, proc.stdout[-500:]
    msg = proc.stdout + proc.stderr
    assert "gold" in msg.lower(), msg[-600:]
    assert "#014" in msg, msg[-600:]
    assert "derived=3 declared=4" in msg, msg[-600:]
    #: The other three agree, so the refusal names one drifted key rather than a wall of
    #: mismatched numbers a reader cannot act on.
    assert "gold #015: derived=3 declared=3 ok" in msg, msg[-600:]
    assert not (tmp_path / "spawned.json").exists(), "a drifted gold set still spawned trials"


def test_the_gold_check_is_answered_before_the_lock_is_waited_on(tmp_path):
    """Both preconditions refuse, so the engine is never waited on for a run whose answer
    key is already wrong.

    This node replaces an assertion an earlier draft of this file carried — `assert not
    (tmp_path / "primary.lock").exists()` inside a node that only called `gold_checks`, which
    neither takes a lock nor is ever handed that path, so it could not fire whichever way the
    code moved. The ordering it wanted is real and belongs to `main`: the gold re-derivation
    at `scripts/find_all_arms_run.py:726`, `acquire_lock` at `:739`, reached only when no row
    disagreed. Here the two refusals are made to collide — the lock is HELD and
    `--wait-seconds 0` asks for no patience, while #014's label is drifted by one — and the
    two messages are disjoint, so only one launcher order can print what is asserted below.
    """
    lock = tmp_path / "primary.lock"
    lock.touch()
    holder = open(lock, "a+")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        #: DECLARED 4 against a fixture tree that can only ever derive 3.
        bench = _fixture_bench(tmp_path / "bench", n014=4)
        proc = _run(["--bench-dir", str(bench),
                     "--vault-root", str(_fixture_vault(tmp_path / "vault")),
                     "--home-root", str(_fixture_home(tmp_path / "home")),
                     "--facts-root", str(_facts_fixture(tmp_path / "facts")),
                     "--driver", str(_fake_driver(tmp_path, writes_rows=True)),
                     "--out", str(tmp_path / "out"),
                     "--lock", str(lock), "--wait-seconds", "0"])
        msg = proc.stdout + proc.stderr
        assert proc.returncode != 0, msg[-500:]
        assert "gold mismatch" in msg, msg[-600:]
        assert "derived=3 declared=4" in msg, msg[-600:]
        #: The lock refusal is the OTHER branch's message, and it is the one a reader has
        #: been wrongly sent to for a week. Its absence here is what proves the order.
        assert "lock still held when the wait ended" not in msg, msg[-600:]
        assert not (tmp_path / "spawned.json").exists(), "spawned while gold disagreed"
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()


def test_a_bench_note_keyed_by_its_own_id_is_matched_to_its_derivation():
    """The four live notes are keyed `id: bench_014_...` and matched to `#014`'s derivation.

    They carry no `task_id` key (`grep -c '^task_id:'` over the four notes → 0 each), and the
    driver keys everything on `task["id"]` (`eval/run_find_all_arms.py:127`). A launcher that
    read only `task_id` and fell back to the file stem would find no entry in DERIVATIONS and
    refuse its own bench — the "refused and nothing ran" shape this item exists to end, from a
    cause nobody predicted. The fixtures here are written the same way, so this is not a
    reading of a key format the launcher invented.
    """
    import scripts.find_all_arms_run as L

    for note_id, want in (("bench_014_audit_dead_wikilinks", "#014"),
                          ("bench_015_audit_cross_entity_fact_copies", "#015"),
                          ("bench_016_audit_skill_dead_paths", "#016"),
                          ("bench_017_audit_unresolved_task_skills", "#017")):
        assert L._bench_key({"id": note_id}) == want, note_id
    #: An explicit `task_id` still wins, and a note with no `bench_NNN` id keeps its own name
    #: so the no-derivation refusal can quote it back.
    assert L._bench_key({"task_id": "#014", "id": "anything"}) == "#014"
    assert L._bench_key({"id": "some_other_note"}) == "some_other_note"


def test_an_empty_fact_tree_is_refused_by_name_rather_than_read_as_drift(tmp_path):
    """A fact tree holding no markdown refuses the run and says so; it is never a 0-vs-3 gold mismatch.

    The live shape of this bug: `~/obsidian/facts` EXISTS and holds 0 `*.md` files (measured
    2026-10-18), while the index the answer key describes is built from
    `app.paths.VAULT_FACTS_ROOT`, which holds 41,759. Pointing a walk at the empty one derives
    0 copies against a bench declaring 3 — which prints as "the answer key drifted" and sends
    the next reader to re-label a gold set that is fine. The refusal has to name the tree.
    """
    import scripts.find_all_arms_run as L

    (tmp_path / "facts").mkdir()
    try:
        L.gold_checks(_fixture_bench(tmp_path / "bench"),
                      vault_root=_fixture_vault(tmp_path / "vault"),
                      home_root=_fixture_home(tmp_path / "home"),
                      facts_root=tmp_path / "facts")
    except L.Refusal as exc:
        assert "holds no markdown fact files" in str(exc), str(exc)
        assert str(tmp_path / "facts") in str(exc), "the refusal must name the tree it walked"
    else:
        assert False, "an empty fact tree silently derived a gold count"


def test_the_skill_path_check_resolves_a_tilde_without_a_home_root_given(tmp_path):
    """`main`'s real-fire defaults — no `--home-root` — still reach the skill-path count.

    `main` passes `home_root=None` when the flag is absent, which is every real fire there
    will ever be: `Path(None)` raised TypeError, so the launcher died with a traceback on its
    own defaults, printed no reason at all, and wrote no trials.jsonl. The vault root comes
    from `--vault-root`; `~` in a skill's `~/lloyd/...` resolves to the running home, which is
    the box that has to answer for the path.
    """
    import scripts.find_all_arms_run as L

    bench = _fixture_bench(tmp_path / "bench")
    #: `_fixture_vault(root)` puts the SKILL.md under `root/vault/skills/` and its two
    #: SURVIVING citations under `root/lloyd/`, so `root` is the home in which those two
    #: citations exist — the launcher resolves `~/lloyd/<key>` against `home_root`.
    vault = _fixture_vault(tmp_path / "env")
    facts = _facts_fixture(tmp_path / "facts")
    #: FIRST a comparison that can actually move, because "default == Path.home()" alone
    #: compares one value with itself: `gold_checks` has no home default of its own, so both
    #: sides resolve identically whatever the code does. What proves the home is READ is two
    #: homes that answer differently. `_fixture_vault(env)` puts the active skill at
    #: `env/skills/alpha/SKILL.md`, citing `gone.py` (twice, two spellings, one key) and
    #: `gone2.json` as dead plus `exists.py` as live, and `_fixture_home` brings exactly
    #: `exists.py` to life under that same home — so the count is 2 there and 3 in a home
    #: where nothing exists at all.
    home_live = _fixture_home(tmp_path / "env")
    live = L.gold_checks(bench, vault_root=vault, facts_root=facts, home_root=home_live)
    empty = L.gold_checks(bench, vault_root=vault, facts_root=facts,
                          home_root=tmp_path / "home-empty")
    n_live = next(r.derived for r in live if r.task_id == "#016")
    n_empty = next(r.derived for r in empty if r.task_id == "#016")
    assert (n_live, n_empty) == (2, 3), (n_live, n_empty)
    #: THEN the real-fire default, which is what this node exists for. Not a number: with no
    #: `--home-root` the citations resolve against the REAL home, and baking that tree's
    #: ground truth into an expected count is a fixture that reddens whenever someone fixes a
    #: skill. Pinned is that the default reaches the four rows instead of dying in `Path(None)`.
    defaulted = L.gold_checks(bench, vault_root=vault, facts_root=facts)
    assert [r.task_id for r in defaulted] == ["#014", "#015", "#016", "#017"], defaulted
    assert all(r.derived is not None for r in defaulted), [r.render() for r in defaulted]


def test_the_store_side_gold_count_counts_rows_that_are_actually_live(tmp_path):
    """#015's count is re-derived as "the gold row is a live row of its named entity".

    Same key the bench's own dedupe uses (``_FactsIdx.find_duplicate`` on
    ``(entity, text_hash)``), so the derivation cannot disagree with the judge about what
    "the same claim" means. An expired or invalidated copy is NOT a hit: re-stating a
    superseded claim is a new claim, and a gold row nobody holds leaves nothing to find.
    """
    import scripts.find_all_arms_run as L

    vault = _fixture_vault(tmp_path / "vault")
    hashes = _copy_hashes()

    def lookup(h):
        return h != hashes[1]              # the second copy has since been expired

    rows = L.gold_checks(_fixture_bench(tmp_path / "bench"), vault_root=vault,
                         home_root=_fixture_home(tmp_path / "home"), store_lookup=lookup)
    row15 = [r for r in rows if r.task_id == "#015"][0]
    assert (row15.derived, row15.declared) == (2, 3), row15
    assert not row15.ok, row15


def test_a_find_all_task_with_no_derivation_refuses_rather_than_being_skipped(tmp_path):
    """Re-deriving three of four answer keys is the half-measure that lets a rot through."""
    import scripts.find_all_arms_run as L

    bench = _fixture_bench(tmp_path / "bench")
    (bench / "bench_099_audit_a_new_kind.md").write_text(
        _bench_task("bench_099_audit_a_new_kind", ["knowledge/new-one"]), encoding="utf-8")
    try:
        L.gold_checks(bench, vault_root=_fixture_vault(tmp_path / "vault"),
                      home_root=_fixture_home(tmp_path / "home"),
                      store_lookup=lambda h: True)
    except L.Refusal as exc:
        assert "#099" in str(exc), str(exc)
    else:
        raise AssertionError("an un-derived find_all task was silently accepted")


# ── clause 1: named refusals, and never a zero that ran nothing ──────────────

def test_a_fact_tree_that_is_absent_is_refused_by_name(tmp_path):
    """No #015 count without a tree to walk, and no zero standing in for one.

    A missing tree would otherwise derive 0, and a bench declaring 0 gold items would PASS
    on it - which is how an environment accident reads as a verified answer key.
    """
    fa = _fixture_args(tmp_path)
    args = [a for i, a in enumerate(fa)
            if not (a == "--facts-root" or (i and fa[i - 1] == "--facts-root"))]
    args += ["--facts-root", str(tmp_path / "facts" / "absent"),
             "--driver", str(_fake_driver(tmp_path, writes_rows=True)),
             "--out", str(tmp_path / "out"), "--lock", str(tmp_path / "pl"),
             "--wait-seconds", "1"]
    proc = _run(args, cwd=REPO_ROOT)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "no fact tree at" in proc.stdout
    assert "No trial was spawned" in proc.stdout
    assert not (tmp_path / "spawned.json").exists()


def test_a_held_lock_is_refused_by_name_after_the_wait_ends(tmp_path):
    """The lock refusal must name the lock and its wait, because the lock is the cause
    that was wrongly blamed for a week.

    ``--wait-seconds 0`` asks for no patience, so a held lock refuses on the spot instead
    of the launcher polling for an hour inside a test.
    """
    lock = tmp_path / "primary.lock"
    lock.touch()
    holder = open(lock, "a+")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        driver = _fake_driver(tmp_path, writes_rows=True)
        proc = _run(_fixture_args(tmp_path) + ["--driver", str(driver),
                                               "--out", str(tmp_path / "out"),
                                               "--lock", str(lock), "--wait-seconds", "0"])
        assert proc.returncode != 0, proc.stdout[-500:]
        msg = proc.stdout + proc.stderr
        #: The launcher's OWN refusal line, not the helper's exception. `_flock_exclusive`
        #: raises `TimeoutError("could not lock <path> within Ns")` on expiry
        #: (`app/atomic_io.py:234`), which also mentions "lock" and also exits non-zero — so a
        #: bare `"lock" in msg` passes on an uncaught traceback. Clause 1 asks for a printed
        #: reason a reader can act on, and an uncaught traceback is not one.
        assert "FIND-ALL ARMS RUN REFUSED: lock still held when the wait ended" in msg, msg[-800:]
        assert str(lock) in msg, msg[-800:]
        assert "within 0.0 s" in msg or "wait ended" in msg, msg[-800:]
        assert "Traceback" not in msg, msg[-1200:]
        assert not (tmp_path / "spawned.json").exists(), "ran on without the engine lock"
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()


def test_a_driver_that_writes_no_trials_exits_nonzero_naming_its_refusal(tmp_path):
    """The tool-sandbox precondition is the driver's own, so the launcher reports ITS word.

    ``require_tool_sandbox`` (``scripts/autoresearch/bench_runner_sdk.py:349-367``) asks the
    aggregator's ``/state`` for its ``tool_sandbox`` key — runtime state, nowhere in
    ``config.yaml``, and a precondition this item never mentions — and raises
    ``ToolSandboxUnavailable("the aggregator does not report an enforced read-only tool sandbox
    ...")``. THAT is the text the launcher has to recognise, and it says "tool sandbox" with a
    space: a classifier matching only the ``tool_sandbox`` field name would fall through to
    "the driver exited rc=1", and a stop nobody can name is how the lock got blamed for a week.
    """
    driver = _fake_driver(tmp_path, writes_rows=False, exit_code=1,
                          stderr="ToolSandboxUnavailable: the aggregator does not report an "
                                 "enforced read-only tool sandbox ({'enforced': False}); "
                                 "refusing to run a bench trial with live tools\n")
    proc = _run(_fixture_args(tmp_path) + ["--driver", str(driver),
                                           "--out", str(tmp_path / "out"),
                                           "--lock", str(tmp_path / "primary.lock")])
    assert proc.returncode != 0, proc.stdout[-500:]
    msg = proc.stdout + proc.stderr
    assert "tool-sandbox precondition refused" in msg, msg[-700:]
    #: The reason must also clear the lock, because the lock is the cause that gets named by
    #: default when a reason is vague.
    assert "NOT the primary.lock" in msg, msg[-700:]
    assert "rc=1" not in msg, "the generic rc-only reason must not win the branch"
    assert not (tmp_path / "out" / "trials.jsonl").exists()


def test_a_zero_exit_requires_trials_jsonl_in_the_out_dir(tmp_path):
    """A driver that exits 0 and writes nothing must NOT be a green run.

    This is the exact shape of the 10-11→10-18 window: something kept being attempted,
    nothing ever landed, and no artifact directory was written to say so.
    """
    driver = _fake_driver(tmp_path, writes_rows=False, exit_code=0)
    out = tmp_path / "out"
    proc = _run(_fixture_args(tmp_path) + ["--driver", str(driver), "--out", str(out),
                                           "--lock", str(tmp_path / "primary.lock")])
    assert proc.returncode != 0, proc.stdout[-500:]
    assert "trials.jsonl" in (proc.stdout + proc.stderr)


def test_a_run_that_produces_trials_exits_zero(tmp_path):
    """The happy path: gold agrees, lock acquired, rows landed, table written, exit 0."""
    driver = _fake_driver(tmp_path, writes_rows=True)
    out = tmp_path / "find-all-2026-10-18"
    proc = _run(_fixture_args(tmp_path) + ["--driver", str(driver), "--out", str(out),
                                           "--lock", str(tmp_path / "primary.lock")])
    assert proc.returncode == 0, proc.stdout[-800:] + proc.stderr[-800:]
    assert (out / "trials.jsonl").exists()
    assert (out.parent / (out.name + ".md")).exists()


def test_the_resolved_argv_is_what_the_driver_is_actually_given(tmp_path):
    """What --dry-run prints and what gets spawned must be the same list.

    A skill that prints one command and runs another is how an owner silently measures
    something else: the 2026-09-24 run used --max-turns 12 and --timeout 600, and its
    table could never separate the cap from the phrasing.
    """
    driver = _fake_driver(tmp_path, writes_rows=True)
    out = tmp_path / "find-all-2026-10-18"
    proc = _run(_fixture_args(tmp_path) + ["--driver", str(driver), "--out", str(out),
                                           "--lock", str(tmp_path / "primary.lock")])
    assert proc.returncode == 0, proc.stderr[-800:]
    #: The marker holds ``[sys.executable] + argv`` as the child saw them. Element 0 is NOT
    #: comparable across the boundary — each process reports its own resolved interpreter —
    #: so everything from the driver path onward is compared, which is the part that must
    #: never differ between what a skill prints and what actually runs.
    spawned = json.loads((tmp_path / "spawned.json").read_text())
    assert spawned[0] and Path(spawned[0]).is_file(), spawned[0]
    assert spawned[1] == str(driver), spawned
    for flag in ("--trials", "3", "--max-turns", "25", "--timeout", "900"):
        assert flag in spawned, spawned
    assert spawned[spawned.index("--out") + 1] == str(out)

    #: A second fixture tree: `_fixture_args` CREATES its vault, and pointing two
    #: invocations at one tmp_path collides on mkdir.
    dry = _run(_fixture_args(tmp_path / "dry") + ["--dry-run", "--out", str(out),
                                                "--lock", str(tmp_path / "lock-probe")])
    assert dry.returncode == 0, dry.stderr[-800:]
    #: From --trials onward the two lists must be identical. The two paths before it may
    #: differ legitimately: the run was pointed at a stand-in driver via --driver while
    #: --dry-run prints the real eval/run_find_all_arms.py, and element 0 of each list is
    #: whichever process BUILT it (pytest's venv python vs the launcher child's resolved
    #: /usr/bin/python3.12), which is why element 0 is never compared across the boundary.
    printed = shlex.split(dry.stdout)
    assert printed[printed.index("--trials"):] == spawned[spawned.index("--trials"):], (
        printed, spawned)
    assert printed[1].endswith("eval/run_find_all_arms.py"), printed
    assert not printed[0].endswith("pytest"), printed[0]
    assert not (tmp_path / "lock-probe").exists(), (
        "the dry run opened the engine lock it was only naming")


# ── clause 4 + 5: the published table, from the driver's own summarise ────────

def test_the_summarise_step_writes_both_report_blocks(tmp_path):
    """The per-{arm}/{task} table AND the judge-independence block, from one JSONL.

    ``--summarise`` already exists in the driver and prints JSON writing no file; the
    missing half was the ``.md``. Re-invoking it is zero-engine by construction, and the
    launcher adds no scoring code: P/R/F1 are means of the fields the driver wrote.
    """
    out = tmp_path / "find-all-2026-10-18"
    out.mkdir()
    _trials_fixture(out / "trials.jsonl")
    proc = _run(["--summarise-only", str(out)])
    assert proc.returncode == 0, proc.stderr[-800:]
    report = (out.parent / "find-all-2026-10-18.md").read_text(encoding="utf-8")
    assert "| arm / task |" in report, report[:900]
    assert "| n | n_scored | n_truncated | n_timeout |" in report, report[:900]
    assert "| P | R | F1 |" in report, report[:900]
    assert f"all / {REAL_TID}" in report, report[:900]
    assert (f"| all / {REAL_TID} | 2 | 1 | 1 | 0 | 1/2 | 25 |") in report, report
    #: Computed numbers, not the section's scaffolding. `summarise` filters on
    #: `rubric_status == "ok"` (`eval/run_find_all_arms.py:282-284`; the values the judge can
    #: emit are `ok`, `skipped`, `not_scored` or a `RUBRIC_FAILURES` entry — `rubric_status` at
    #: `scripts/autoresearch/judge.py:181-191`, `:110`, `:841`),
    #: so with the status spelled anything else the block collapses to "over the 0 rankable
    #: trials" and "0 / 0 (0%)" while both headings still print: asserting the headings would
    #: pass on a block that had read nothing. `1 rankable` is the fixture's one un-cut graded
    #: row (the turn-truncated `all` trial is excluded by `_cut`, `eval/run_find_all_arms.py
    #: -208), and its rubric 0.2 against P×R 1.0 is the disagreement. Pearson is `-` because
    #: `_pearson` returns None under 3 points (`eval/run_find_all_arms.py:216-219`).
    ji = report[report.index("## Judge independence"):]
    assert "over the 1 rankable" in ji, ji
    assert "disagreement at the 0.5 line: 1 / 1 (100%)" in ji, ji


def test_the_report_prints_the_at_cap_accounting_per_cell(tmp_path):
    """The cap, not the phrasing, decided three of eight ``all`` trials on 2026-09-24.

    Every cell prints its cap-hit share and the turn budget it was measured against,
    because that is what the clause has to be read on top of.
    """
    out = tmp_path / "find-all-2026-10-18"
    out.mkdir()
    _trials_fixture(out / "trials.jsonl")
    proc = _run(["--summarise-only", str(out)])
    assert proc.returncode == 0, proc.stderr[-800:]
    report = (out.parent / "find-all-2026-10-18.md").read_text(encoding="utf-8")
    assert "at-cap" in report, report[:1200]
    assert "turn budget" in report, report[:1200]
    assert "| 1/2 | 25 |" in report, report     # the `all` cell: 1 cap hit of 2, budget 25


#: The published table's column order, verbatim from `write_report`'s header
#: (`scripts/find_all_arms_run.py:614-616`), so a node reads the R column by name instead of
#: counting pipes.
_TABLE_COLS = ["arm / task", "n", "n_scored", "n_truncated", "n_timeout", "at-cap",
               "turn budget", "P", "R", "F1"]


def _table_cells(report: str) -> dict:
    """The rendered per-{arm}/{task} table as {(arm, task): {column: printed value}}."""
    head = "| " + " | ".join(_TABLE_COLS) + " |"
    assert head in report, f"the published table's header moved; header asserted was:\n{head}\n" \
                          f"first 1200 chars of report:\n{report[:1200]}"
    cells = {}
    for ln in report.splitlines():
        if not ln.startswith("| ") or ln.startswith("|---") or ln.startswith("| arm /"):
            continue
        parts = [c.strip() for c in ln.strip().strip("|").split("|")]
        assert len(parts) == len(_TABLE_COLS), ln
        arm, task = parts[0].split(" / ", 1)
        cells[(arm, task)] = dict(zip(_TABLE_COLS[1:], parts[1:]))
    assert cells, report[:1200]
    return cells


#: A line that RULES on the arms instead of measuring them. The report is entitled to say
#: everything the clause needs ABOUT recall ("Read every recall above on top of it"), so what
#: is caught is a comparison: the two arm names joined by an ordering sign, a numeric
#: inequality anywhere in the file, a verb of beating with an arm in the same sentence, or the
#: word "clause" within a sentence of a verb of ruling. The report's own disclaimers — "No arm
#: is compared to another here", "a rubric gain on its own is not a win", "too thin for the
#: clause either way" — match none of these branches, and the three controls in the node below
#: prove the pattern fires on verdicts worded three different ways.
_VERDICT = re.compile(
    r"recall\s*\(\s*all"
    r"|\ball\b\s*<>?=?\s*`?\s*one\b"
    r"|[0-9](?:\.[0-9]+)?\s*[><]\s*[0-9](?:\.[0-9]+)?"
    r"|\b(?:all|one)\b[^.\n]{0,40}\b(?:beats?|beaten|outperforms?|superior|inferior"
    r"|better than|worse than)\b"
    r"|\bclause\b[^.\n]{0,40}\b(?:holds?|met|unmet|confirmed|refuted|answered|settled)\b",
    re.I)


def test_the_published_report_states_no_verdict_even_with_the_gap_in_front_of_it(tmp_path):
    """The #647 ruling is a READING, and this reads the REPORT, not the launcher's source.

    An earlier draft asserted `"recall(all)" not in LAUNCHER.read_text()` plus a "phrasing
    clause" substring ban. The review was right that that is a spelling test: a verdict worded
    any other way in the `.md` passed it, and renaming a string in the launcher would have
    reddened it while the published artifact went on saying whatever it said. So the artifact
    is what is read here, over a fixture that hands the verdict to the renderer: `all` beats
    `one` on recall in BOTH tasks, on both rows of every cell, with nothing truncated. The two
    gap assertions are the node's own positive control — "no verdict appears" proves nothing
    unless the verdict was available to appear, the same reason a 0-hit grep needs a path that
    is known to match.
    """
    out = tmp_path / "find-all-2026-10-18"
    out.mkdir()
    _gap_trials(out / "trials.jsonl")
    proc = _run(["--summarise-only", str(out)])
    assert proc.returncode == 0, proc.stderr[-800:]
    report = (out.parent / "find-all-2026-10-18.md").read_text(encoding="utf-8")

    cells = _table_cells(report)
    #: POSITIVE CONTROL, first: the gap is in the published table, on both tasks the clause
    #: names. (Two answered trials a cell here; the clause wants n>=3, which is why the real
    #: fire pins --trials 3 and why the report says a thinner cell is too thin either way.)
    for tid in (REAL_TID, REAL_TID_016):
        assert float(cells[("all", tid)]["R"]) > float(cells[("one", tid)]["R"]), \
            (tid, cells[("all", tid)], cells[("one", tid)])
    #: The four levels `_gap_trials` is built to produce: each row sets recall equal to its
    #: own precision, so a cell's R is the mean of its two trials' levels.
    assert cells[("all", REAL_TID)]["R"] == "0.75", cells[("all", REAL_TID)]
    assert cells[("one", REAL_TID)]["R"] == "0.20", cells[("one", REAL_TID)]
    assert cells[("all", REAL_TID_016)]["R"] == "0.80", cells[("all", REAL_TID_016)]
    assert cells[("one", REAL_TID_016)]["R"] == "0.20", cells[("one", REAL_TID_016)]

    hits = [(n, ln) for n, ln in enumerate(report.splitlines(), 1) if _VERDICT.search(ln)]
    assert hits == [], hits
    #: And the detector is not a null net: three ways a verdict could have been worded, each
    #: one it must catch.
    for verdict in (
            "**Verdict: recall(all) > recall(one) on #014 and #016, so the phrasing clause "
            "holds.**",
            "The `all` arm beats the `one` arm here (0.75 against 0.20).",
            "Reading the table, the clause is settled: one phrasing is worse than the other."):
        assert _VERDICT.search(verdict), verdict


def test_the_published_numbers_are_the_drivers_own_not_a_second_scoring_pass(tmp_path):
    """Every score in the table is a value the driver's own `summarise` computes for the same
    rows — which is the property the ban on new scoring code exists to keep.

    An earlier draft banned the strings `tp + fp`, `gold_set`, `compute_f1` and `def precision`
    in the launcher: a spelling test again, and one a re-implementation under any other name
    walked straight past. This node computes the truth in-process from
    `eval/run_find_all_arms.py::summarise`, renders the report through the launcher's own
    route, and compares cell by cell, so what is pinned is the artifact a reader trusts. The
    fixture's `items` maps are empty while the rows carry their own nonzero scores, so a
    launcher that recomputed a score from `items` would print `0.00` or `-` where 0.75 is
    asserted. The limit of comparing outputs is stated plainly: a second implementation that
    agrees with the driver to two decimals passes this node, and what makes that unlikely is
    structural — `write_report` copies its numbers out of the dict it is handed
    (`scripts/find_all_arms_run.py:621-637`) and this diff adds no arithmetic to it.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_arms_driver_truth", REPO_ROOT / "eval" / "run_find_all_arms.py")
    driver_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver_mod)

    out = tmp_path / "find-all-2026-10-18"
    out.mkdir()
    rows = _gap_trials(out / "trials.jsonl")
    truth = driver_mod.summarise(rows)

    proc = _run(["--summarise-only", str(out)])
    assert proc.returncode == 0, proc.stderr[-800:]
    report = (out.parent / "find-all-2026-10-18.md").read_text(encoding="utf-8")
    cells = _table_cells(report)

    assert sorted(cells) == sorted(
        {(k.rsplit("/", 1)[0], k.rsplit("/", 1)[1]) for k in truth["cells"]}), cells.keys()
    for key, want in sorted(truth["cells"].items()):
        arm, task = key.rsplit("/", 1)
        got = cells[(arm, task)]
        for col in ("n", "n_scored", "n_truncated", "n_timeout"):
            assert got[col] == str(want[col]), (key, col, got, want)
        for col, field in (("P", "precision"), ("R", "recall"), ("F1", "f1")):
            expect = "-" if want[field] is None else f"{want[field]:.2f}"
            assert got[col] == expect, (key, col, got[col], expect)
    #: The headline the ban is about: 0.75 is the mean of the two `all`/#014 trials' OWN
    #: `precision` fields, a number nothing readable from an empty item map could produce.
    assert cells[("all", REAL_TID)]["P"] == "0.75", cells[("all", REAL_TID)]
    #: The judge block read all eight fixture rows — 8 answered `all`/`one` trials carrying an
    #: objective, a rubric and `rubric_status: "ok"`, which is the whole of `summarise`'s
    #: filter at `eval/run_find_all_arms.py:282-284` — and 8 points is past the 3-point floor
    #: under which `_pearson` returns None (`eval/run_find_all_arms.py:216-219`), so the line
    #: below is a number and not a dash.
    ji = truth["judge_independence"]
    assert ji["n"] == 8, ji
    assert f"over the {ji['n']} rankable" in report, report[-900:]
    assert f"- disagreement at the 0.5 line: {ji['disagree_at_0.5']} / {ji['n']}" in report, \
        report[-900:]
    assert re.search(r"Pearson r \(rubric vs P.R\): -?\d", report), report[-900:]


def test_the_launcher_waits_on_the_shared_flock_helper_not_its_own_loop():
    """``app.atomic_io._flock_exclusive`` is already in the tree for exactly this; a
    hand-rolled sleep/poll loop is a second implementation with a second set of semantics.
    """
    src = LAUNCHER.read_text(encoding="utf-8")
    assert "_flock_exclusive" in src, src[:600]
    assert "time.sleep" not in src, "an open-coded wait loop; use the shared helper"


def test_summarise_only_refuses_loudly_when_there_are_no_rows_to_summarise(tmp_path):
    """No rows means no table, so a refused publish is the only honest exit — and no
    half-written report is left beside an empty run directory.
    """
    out = tmp_path / "find-all-empty"
    out.mkdir()
    (out / "trials.jsonl").write_text("", encoding="utf-8")
    proc = _run(["--summarise-only", str(out)])
    assert proc.returncode != 0, proc.stdout[-500:]
    assert "no trials" in (proc.stdout + proc.stderr).lower(), proc.stdout[-500:]
    assert not (out.parent / "find-all-empty.md").exists()
