"""The Knowledge Health Report's Loaded-memory referential integrity section (#2821).

#94's resolver has always written `~/obsidian/autonomy/referential-integrity-latest.md`
and a dated copy, and nothing read either of them: the 2026-10-15 dated copy read
`361 cites, 23 dangling, 8 exempt, 0 newly dangling` against the 2026-10-04 copy's
`206 cites, 12 dangling, 5 exempt`, with no open backlog item on the population. The
noticing stopped at the file. This section is the consumer: the four numbers are read
at run time out of the ledger, so the report shows what the sweep measured rather
than a copy of it that can rot.

Two properties the section is written to, both from the #1227 precedent pinned in
`tests/test_knowledge_health_retention.py`:

* A check that could not run must say so in words. An absent or unreadable ledger
  prints `referential integrity: ledger absent` / `referential integrity: ledger
  unreadable`, and is never an omitted section — a silent section reads the same as
  a check that never ran.
* The standing `dangling` TOTAL is a baseline, not a finding. The reportable change
  is the `newly dangling` DELTA, which is the only number the resolver prints that
  moves when something breaks today. So `NEWLY DANGLING: n` appears only when n is a
  real nonzero count, and never for `0` or `n/a (no previous report)`.

The corpus the section names is the resolver's own `CORPUS` tuple, and the section
says the resolver edits nothing — a reader must not take it as a fix that happened.
"""
import importlib.util
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location(
    "khr_refint", ROOT / "scripts/memory/knowledge-health-report.py")
khr = importlib.util.module_from_spec(_spec)
sys.modules["khr_refint"] = khr
_spec.loader.exec_module(khr)

SCRIPT = ROOT / "scripts/memory/knowledge-health-report.py"
RESOLVER = ROOT / "scripts/maintenance/referential_integrity.py"

#: A ledger with all four counts present and a nonzero delta. The numbers are the
#: live 2026-10-16 shape (372 cites / 23 dangling / 9 exempt) with a delta of 2, so a
#: fixture that looks like the real file cannot quietly be a fixture of nothing.
LEDGER_MEASURED = """---
type: note
segment: autonomy
tags: [referential-integrity, loaded-memory, dangling-cites]
generated_at: "2026-10-16T12:01:03Z"
summary: "Referential-integrity sweep over loaded memory: 372 cites, 23 dangling. Generated, do not hand-edit."
---

# Referential integrity — loaded memory

Run 2026-10-16T12:01:03Z by `scripts/maintenance/referential_integrity.py` (#882). \
Report only: nothing cited here was edited.

- total cites: 372
- dangling: 23
- exempt: 9
- newly dangling since the previous report: 2

## Dangling

- `lloyd/USER.md:28` → `voice_library/profiles/` — stat:repo_root|vault_root|data_root|citing_dir (none exist) **(new)**
- `memory/mental-models.md:1069` → `~/lloyd/MEMORY.md` — stat:home (none exist) **(new)**

## Exempt

- `memory/mental-models.md:99` → `scripts/gone.py` — says "no longer exists"
"""


def _ledger(tmp_path: Path, body: str = LEDGER_MEASURED) -> Path:
    path = tmp_path / "referential-integrity-latest.md"
    path.write_text(body, encoding="utf-8")
    return path


def _report(ledger_path: Path | None = None) -> str:
    """The generated report, with the ledger read from `ledger_path`.

    Passing a path is the whole point of the seam: the numbers in the section are
    those bytes, not a constant the script carries.
    """
    now = datetime.now(timezone.utc)
    return khr.generate_report({}, khr.compute_relationship_stats([], {}), [], [], now,
                               referential_ledger=ledger_path)


def _section(report: str) -> str:
    assert "## Loaded-memory referential integrity" in report, (
        "the section is absent: a check that could not read its input must still "
        "print, not vanish (#1227)")
    return report.split("## Loaded-memory referential integrity", 1)[1].split("\n## ", 1)[0]


def _stated(section: str, label: str) -> str:
    m = re.search(rf"^- {re.escape(label)}: (.+)$", section, re.M)
    assert m, f"{label!r} not printed in the section:\n{section}"
    return m.group(1)


# ── clause 1: the four numbers come from the ledger file, at run time ─────────


def test_the_four_counts_are_read_from_the_ledger_file(tmp_path):
    """Total cites, dangling, exempt and newly dangling all arrive from the bytes.

    Every one of the four is asserted by its own label, because a section that
    printed three of them and hardcoded the fourth would still read as the ledger.
    """
    section = _section(_report(_ledger(tmp_path)))
    assert _stated(section, "total cites") == "372"
    assert _stated(section, "dangling") == "23"
    assert _stated(section, "exempt") == "9"
    assert _stated(section, "newly dangling since the previous report") == "2"


def test_rewriting_the_ledger_changes_the_section(tmp_path):
    """The section follows the file, so it cannot be a transcription of last month.

    #2821 filed because a ledger that nearly doubled its dangling population was
    read by nobody; a section that reproduced one morning's numbers forever would be
    the same failure with one more hop.
    """
    path = _ledger(tmp_path)
    before = _section(_report(path))
    path.write_text(
        LEDGER_MEASURED
        .replace("- total cites: 372", "- total cites: 401")
        .replace("- dangling: 23", "- dangling: 31")
        .replace("- exempt: 9", "- exempt: 12")
        .replace("- newly dangling since the previous report: 2",
                 "- newly dangling since the previous report: 0"),
        encoding="utf-8")
    after = _section(_report(path))
    assert _stated(after, "total cites") == "401"
    assert _stated(after, "dangling") == "31"
    assert _stated(after, "exempt") == "12"
    assert _stated(after, "newly dangling since the previous report") == "0"
    for label, old in (("total cites", "372"), ("dangling", "23"), ("exempt", "9")):
        assert _stated(before, label) == old
        assert _stated(before, label) != _stated(after, label)


def test_the_reader_returns_the_ledger_counts_as_data(tmp_path):
    """`read_referential_ledger` is the function the report and a person can call.

    Asserted as data rather than prose because this is the half the acceptance check
    names: `git grep referential-integrity-latest -- app/ scripts/` has to name a
    reader, and a reader is a function that returns the numbers.
    """
    got = khr.read_referential_ledger(_ledger(tmp_path))
    assert got["state"] == "measured", got
    assert (got["cites"], got["dangling"], got["exempt"], got["newly"]) == (372, 23, 9, 2)
    assert got["stamp"] == "2026-10-16T12:01:03Z", got


# ── clause 2: absent / unreadable / countless prints a named not-measured state ───


def test_a_missing_ledger_prints_ledger_absent(tmp_path):
    """No file is a named state, printed in words, with no numbers to misread as 0.

    #94's 2026-10-13 run made no dated copy and still recorded `status: success`, so
    "the ledger was not written tonight" is a state this box has actually been in —
    and a section that printed `dangling: 0` in that state would be worse than the
    silent one it replaces.
    """
    missing = tmp_path / "referential-integrity-latest.md"
    assert not missing.exists()
    section = _section(_report(missing))
    assert "referential integrity: ledger absent" in section
    assert "unreadable" not in section, section
    for label in ("total cites", "dangling", "exempt",
                  "newly dangling since the previous report"):
        assert not re.search(rf"^- {re.escape(label)}: ", section, re.M), (
            f"{label!r} was printed as a count the ledger never supplied")
    assert "NEWLY DANGLING" not in section


def test_an_unreadable_ledger_prints_ledger_unreadable(tmp_path):
    """A path that exists but cannot be read is its own named state.

    Built as a directory named like the ledger, which is the state a half-finished
    `cp` or a wrong `--report` argument leaves behind: `open()` raises, and an
    exception here would lose the whole report, not just this section.
    """
    path = tmp_path / "referential-integrity-latest.md"
    path.mkdir()
    section = _section(_report(path))
    assert "referential integrity: ledger unreadable" in section
    assert "ledger absent" not in section, section
    assert "NEWLY DANGLING" not in section
    assert re.search(r"\bedits nothing\b", section), section
    notice = next(ln for ln in section.splitlines()
                  if ln.startswith("referential integrity: ledger unreadable"))
    for name in ("lloyd/USER.md", "lloyd/MEMORY.md", "memory/mental-models.md"):
        assert name in notice, f"{name} missing from the unreadable notice: {notice}"


def test_a_ledger_with_no_parseable_counts_is_reported_as_unreadable(tmp_path):
    """A ledger whose four lines are gone reads as unreadable, not as a clean sweep.

    The shape is a real one: the counts are emitted by `render_report`, so a run that
    died mid-write or a hand-truncated copy leaves front matter and prose with no
    counts. A section that printed `dangling: 0` there would report health from
    whatever the parser failed to find.
    """
    path = _ledger(tmp_path, "---\ntype: note\nsegment: autonomy\n---\n\n"
                            "# Referential integrity — loaded memory\n\n"
                            "Run truncated.\n")
    got = khr.read_referential_ledger(path)
    assert got["state"] == "unreadable", got
    section = _section(_report(path))
    assert "referential integrity: ledger unreadable" in section
    for label in ("total cites", "dangling", "exempt"):
        assert not re.search(rf"^- {re.escape(label)}: ", section, re.M), (
            f"{label!r} was printed as a count the ledger never supplied")


def test_a_partial_ledger_is_unreadable_rather_than_partially_measured(tmp_path):
    """Three of four counts is still no measurement — the missing one is the delta.

    `newly dangling` is the number the finding hinges on, so a section that printed
    cites/dangling/exempt and stayed quiet about the delta would print a healthy-
    looking table built from a truncated file.
    """
    path = _ledger(tmp_path, LEDGER_MEASURED.replace(
        "- newly dangling since the previous report: 2\n", ""))
    got = khr.read_referential_ledger(path)
    assert got["state"] == "unreadable", got
    assert "NEWLY DANGLING" not in _section(_report(path))


def _run_cli(tmp_path, ledger: Path, *extra: str):
    """One CLI run against an empty facts dir; returns (process, output_dir).

    `--facts-dir` points at an empty temp directory so the entity walk finds nothing
    and the run costs about a second: what is under test is the script's own argument
    plumbing and the exit sentinel, not the corpus.
    """
    tag = abs(hash(str(ledger))) % 10**8
    out_dir = tmp_path / f"out-{tag}"
    facts = tmp_path / f"facts-{tag}"
    facts.mkdir()
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(out_dir),
         "--facts-dir", str(facts), "--referential-ledger", str(ledger), *extra],
        capture_output=True, text=True, timeout=300, cwd=str(ROOT))
    return proc, out_dir


def test_an_absent_ledger_adds_no_alarm_and_does_not_change_the_exit_code(tmp_path):
    """No ledger is a printed state, not an alarm condition and not a red run.

    Run as a subprocess twice, against the same tree, because the clause is about the
    exit code the scheduler reads. The two runs differ only in whether the ledger file
    is there, and three things are asserted across the pair: their exit codes are EQUAL,
    their ALARM blocks (read from stderr, where the script puts them) are EQUAL, and
    neither block names referential integrity or dangling cites. Equal-and-silent is the
    difference between "an unreadable input is a printed state" and "an unreadable input
    is a red run", and it is asserted as a difference rather than as a literal zero
    because this tree already carries alarm conditions of its own that are not this
    change's: in a round worktree the knowledge-graph store is absent altogether, and
    against the real data root 3,206 of 158,469 fact records carry no entity tag, above
    the 0.5% floor. Either one exits 2 on its own, before this section is reached — which
    is why no literal 0 is asserted anywhere in this node, and why the sibling CLI node
    passes `--no-alarm-exit` and asserts only that the run completed and wrote the file.
    """
    missing = tmp_path / "no-such-ledger.md"
    with_ledger, out_present = _run_cli(tmp_path, _ledger(tmp_path))
    without, out_missing = _run_cli(tmp_path, missing)
    assert with_ledger.returncode == without.returncode, (
        f"an absent ledger changed the exit code: {with_ledger.returncode} present vs "
        f"{without.returncode} absent\n{without.stdout[-1500:]}")
    # `ALARM:` and its lines go to STDERR (the emit site passes file=sys.stderr), so a
    # block sliced out of stdout is always empty and a guard written against it passes
    # whatever the run did. The first gate round of #2821 was refused on exactly this,
    # and it is load-bearing here: the equal-returncode check above is satisfied
    # trivially by two runs that both hit the tree's own alarm conditions.
    blocks = []
    for proc in (with_ledger, without):
        assert "ALARM:" not in proc.stdout, (
            f"the script wrote its alarm block to stdout, where the guard below would "
            f"have been reading the wrong stream: {proc.stdout[-800:]}")
        blocks.append(proc.stderr.split("ALARM:", 1)[1]
                      if "ALARM:" in proc.stderr else "")
        assert not re.search(r"referential|dangling", blocks[-1], re.I), (
            f"the ledger's state raised an alarm: {blocks[-1][:600]}")
    assert blocks[0] == blocks[1], (
        "an absent ledger changed which alarms fire:\n  ledger present: "
        f"{blocks[0][:400]!r}\n  ledger absent: {blocks[1][:400]!r}")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    absent_body = (out_missing / f"knowledge-health-{date_str}.md").read_text(encoding="utf-8")
    present_body = (out_present / f"knowledge-health-{date_str}.md").read_text(encoding="utf-8")
    assert "referential integrity: ledger absent" in absent_body
    assert "referential integrity: ledger absent" not in present_body, (
        "the run that had a readable ledger printed the absent state anyway")


def test_the_report_still_writes_and_exits_zero_with_no_ledger(tmp_path):
    """With a ledger missing, the dated file is written and the sentinel reports 0.

    `--no-alarm-exit` is passed, and it suppresses only the tree's unrelated alarm
    conditions described above: an absent ledger raises no alarm of its own, which the
    node directly above establishes by comparing two runs started without the flag. So
    this node's 0 is the claim that the run completes — the report is generated, the
    dated file lands, the section prints its named absent state — where a version that
    treated an unreadable ledger as fatal would produce no file at all.
    """
    proc, out_dir = _run_cli(tmp_path, tmp_path / "no-such-ledger.md", "--no-alarm-exit")
    assert proc.returncode == 0, (proc.returncode, proc.stdout[-1500:], proc.stderr[-800:])
    assert "LLOYD_SCRIPT_EXIT: knowledge-health-report.py exit=0" in proc.stdout, \
        proc.stdout[-800:]
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    written = out_dir / f"knowledge-health-{date_str}.md"
    assert written.exists(), f"no dated report in {out_dir}: {list(out_dir.iterdir())}"
    body = written.read_text(encoding="utf-8")
    assert "## Loaded-memory referential integrity" in body
    assert "referential integrity: ledger absent" in body


# ── clause 3: the delta is the finding, the total is the baseline ─────────────


def test_a_nonzero_newly_dangling_delta_is_marked_as_a_finding(tmp_path):
    """`NEWLY DANGLING: n` on its own line, with n the ledger's own count."""
    section = _section(_report(_ledger(tmp_path)))
    assert re.search(r"^NEWLY DANGLING: 2$", section, re.M), section


def test_a_zero_delta_prints_the_counts_without_any_finding_marker(tmp_path):
    """The live state: 23 dangling and 0 new is a baseline, and must not read as news.

    This is the exact ledger the item was filed against (23 dangling, 0 newly
    dangling on 2026-10-15). Flagging the total would make every nightly report an
    alarm about a population nobody is being asked to fix, which is how #2795's
    exemption cleanup got read as though it had made the corpus worse.
    """
    path = _ledger(tmp_path, LEDGER_MEASURED.replace(
        "- newly dangling since the previous report: 2",
        "- newly dangling since the previous report: 0"))
    section = _section(_report(path))
    assert "NEWLY DANGLING" not in section, section
    assert _stated(section, "dangling") == "23"
    assert _stated(section, "newly dangling since the previous report") == "0"


def test_an_first_run_delta_of_n_a_prints_the_counts_without_a_finding_marker(tmp_path):
    """`n/a (no previous report)` is not a zero and not a finding.

    The resolver writes that string on its first run against a dated copy. A parser
    that read it as 0 would silently report "nothing new" on the night the baseline
    is established, and a parser that read it as nonzero would cry wolf every time
    the dated series is restarted.
    """
    path = _ledger(tmp_path, LEDGER_MEASURED.replace(
        "- newly dangling since the previous report: 2",
        "- newly dangling since the previous report: n/a (no previous report)"))
    assert khr.read_referential_ledger(path)["newly"] is None
    section = _section(_report(path))
    assert "NEWLY DANGLING" not in section, section
    assert "n/a" in section


# ── clause 4: what was weighed, and that nothing was edited ──────────────────


def test_the_section_names_the_corpus_and_says_the_resolver_edits_nothing(tmp_path):
    """Both in the measured state, so neither depends on a finding existing."""
    section = _section(_report(_ledger(tmp_path)))
    for name in ("lloyd/USER.md", "lloyd/MEMORY.md", "memory/mental-models.md"):
        assert name in section, f"{name} not named as weighed"
    assert re.search(r"\bedits nothing\b", section), section


def test_the_absent_state_also_names_the_corpus_and_edits_nothing(tmp_path):
    """A reader who only ever sees the not-measured state must learn the same scope.

    The corpus sentence describes the instrument, not tonight's measurement, so
    dropping it when the ledger is missing would leave the named-state section
    reading as a check over an unnamed something. Each path is asserted twice: over
    the section, and inside the notice line beginning `referential integrity: ledger
    absent`. The first gate round of #2821 was refused on clause 4 by a grader
    reading the generated section in isolation, so the property is pinned at line
    level — a clause that holds only while its neighbouring lines are also being
    read is not a clause.
    """
    section = _section(_report(tmp_path / "nope.md"))
    for name in ("lloyd/USER.md", "lloyd/MEMORY.md", "memory/mental-models.md"):
        assert name in section, f"{name} not named as weighed"
    assert re.search(r"\bedits nothing\b", section), section
    notice = next(ln for ln in section.splitlines()
                  if ln.startswith("referential integrity: ledger absent"))
    for name in ("lloyd/USER.md", "lloyd/MEMORY.md", "memory/mental-models.md"):
        assert name in notice, f"{name} missing from the absent notice: {notice}"


def test_the_corpus_the_section_names_is_the_resolver_s_own():
    """The section cannot quietly weigh a different set of files than the sweep did.

    Read out of the resolver's source rather than imported: the report must not gain
    a dependency on the resolver to describe it, and the resolver is the file this
    round is forbidden to change, so a drift between the two is a bug in the consumer
    and this node is where it surfaces.
    """
    src = RESOLVER.read_text(encoding="utf-8")
    m = re.search(r"^CORPUS = \(([^)]*)\)", src, re.M)
    assert m, "the resolver's CORPUS tuple is not where this node expects it"
    weighed = tuple(re.findall(r'"([^"]+)"', m.group(1)))
    assert weighed == khr.REFERENTIAL_CORPUS, (weighed, khr.REFERENTIAL_CORPUS)


# ── the surface the change is for: #65 and #48 read the dated file ───────────


def test_the_cli_writes_the_section_with_the_ledger_s_own_numbers(tmp_path):
    """Fixture ledger in, the same four numbers in the dated file the next job reads.

    This is the seam #2821 exists to close: `research-queue-generator` (#65) and
    `entity-resolution-sweep` (#48) read the dated file under
    `--output-dir`, and until now no writer put the ledger's counts in it. Run as a
    subprocess against a temp facts dir so the real CLI arg plumbing is what carries
    the path — a direct `generate_report` call would pass even if `main()` never
    forwarded the flag. `--no-alarm-exit` is passed because the alarm conditions this
    tree reports on its own (the entity-tag floor, the store) are not this claim; what
    is asserted is that the process ran, the sentinel said 0, and the file holds the
    numbers.
    """
    ledger = _ledger(tmp_path)
    proc, out_dir = _run_cli(tmp_path, ledger, "--no-alarm-exit")
    assert proc.returncode == 0, (proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:])
    assert "LLOYD_SCRIPT_EXIT: knowledge-health-report.py exit=0" in proc.stdout, \
        proc.stdout[-800:]
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    section = _section((out_dir / f"knowledge-health-{date_str}.md").read_text(encoding="utf-8"))
    # The path the section names is the one the flag carried. Without this the node
    # could pass on a run that quietly fell back to the default vault ledger, whose
    # cites / dangling / exempt counts are the same three numbers on 2026-10-16.
    assert f"Ledger: `{ledger}`" in section, section
    assert _stated(section, "total cites") == "372"
    assert _stated(section, "dangling") == "23"
    assert _stated(section, "exempt") == "9"
    assert re.search(r"^NEWLY DANGLING: 2$", section, re.M), section


def test_the_default_ledger_path_is_the_one_94_writes():
    """The script's default must be the resolver's report path, not a copy of its name.

    #94 is a Bash job whose `--report` argument is the literal
    `~/obsidian/autonomy/referential-integrity-latest.md`; a default that drifted
    (a renamed file, a different directory) would leave the section reporting
    `ledger absent` forever while the nightly wrote its ledger one path away.
    """
    assert khr.REFERENTIAL_LEDGER == Path.home() / "obsidian/autonomy/referential-integrity-latest.md"
    assert khr.REFERENTIAL_LEDGER.name == "referential-integrity-latest.md"
    src = RESOLVER.read_text(encoding="utf-8")
    assert "autonomy/referential-integrity-latest.md" in src


# ── the seam between the two processes: the resolver's own bytes, parsed ───────


def _resolver_module():
    """Load the resolver by path, from a consumer that must never import it.

    `scripts/maintenance/` is not an importable package and the report is designed not
    to depend on the job it reports about, so the module is loaded here the way
    `knowledge-health-report.py` loads itself — by file. The point is to read the
    resolver's REAL rendering code, not a hand-written imitation of its output.
    """
    import importlib.util

    path = ROOT / "scripts" / "maintenance" / "referential_integrity.py"
    assert path.exists(), f"{path} is not in the tree, so this node cannot reach the writer"
    spec = importlib.util.spec_from_file_location("_ri_writer_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rec(file, line, target, verdict):
    return {"file": file, "line": line, "kind": "path", "target": target,
            "verdict": verdict, "how_checked": "path under vault/repo/data"}


def test_the_resolver_s_own_rendered_report_is_what_the_section_parses(tmp_path):
    """One process renders the ledger, another parses it hours later (#2821 clause 1).

    `render_report` writes the four count lines; `read_referential_ledger` is the only
    thing that reads them into this report. Until now the two had never met in a test:
    every other node used a hand-written fixture, so the reader could have drifted from
    the writer's exact strings and stayed green while the nightly section printed
    `ledger unreadable` forever. This node renders with the resolver, writes the bytes,
    and reads them back — the writer is never modified, only called.
    """
    ri = _resolver_module()
    records = [
        _rec("lloyd/USER.md", 28, "voice_library/profiles/", "dangling"),
        _rec("memory/mental-models.md", 41, "~/lloyd/workers.db", "dangling"),
        _rec("lloyd/MEMORY.md", 7, "knowledge/foo.md", "exempt"),
        _rec("lloyd/MEMORY.md", 9, "app/prompt_builder.py", "resolves"),
    ]
    path = tmp_path / "referential-integrity-latest.md"
    path.write_text(ri.render_report(records, previous=None), encoding="utf-8")

    got = khr.read_referential_ledger(path)
    assert got["state"] == "measured", got
    # 4 records weighed, 2 dangling, 1 exempt; `previous=None` is the resolver's own
    # first-night case, which it prints as `n/a (no previous report)`, not as a count.
    assert (got["cites"], got["dangling"], got["exempt"]) == (4, 2, 1), got
    assert got["newly"] is None, got
    section = _section(_report(path))
    assert _stated(section, "total cites") == "4"
    assert _stated(section, "dangling") == "2"
    assert _stated(section, "exempt") == "1"
    assert _stated(section, "newly dangling since the previous report") == \
        "n/a (no previous report)"
    assert "NEWLY DANGLING" not in section, section


def test_a_second_night_of_the_resolver_s_render_marks_the_new_dangling_cite(tmp_path):
    """The delta the section reports comes from the writer's own key comparison.

    Same corpus, one extra dangling cite, with `previous` set to the first night's
    state block: the resolver emits `newly dangling since the previous report: 1` and
    the section must carry that as the finding, while the `dangling` total (now 3) stays
    a baseline. Rendered twice by the writer rather than typed, so a change to the
    resolver's wording that this reader cannot follow fails here instead of nightly.
    """
    ri = _resolver_module()
    night_one = [
        _rec("lloyd/USER.md", 28, "voice_library/profiles/", "dangling"),
        _rec("memory/mental-models.md", 41, "~/lloyd/workers.db", "dangling"),
        _rec("lloyd/MEMORY.md", 7, "knowledge/foo.md", "exempt"),
    ]
    first = tmp_path / "night-one.md"
    first.write_text(ri.render_report(night_one, previous=None), encoding="utf-8")
    previous = {ri._key(r) for r in night_one if r["verdict"] == "dangling"}

    night_two = night_one + [
        _rec("memory/mental-models.md", 900, "scripts/autonomy.py:4400", "dangling")]
    path = tmp_path / "referential-integrity-latest.md"
    path.write_text(ri.render_report(night_two, previous=previous), encoding="utf-8")

    got = khr.read_referential_ledger(path)
    assert (got["state"], got["cites"], got["dangling"], got["newly"]) == \
        ("measured", 4, 3, 1), got
    section = _section(_report(path))
    assert re.search(r"^NEWLY DANGLING: 1$", section, re.M), section
    assert _stated(section, "dangling") == "3", section


# ── clause 5: the three vault files, read from disk ───────────────────────────
#
# The vault cannot carry a test, so the repo reads the published definitions and
# refuses them when they stop meaning what the item asked for — the shape
# tests/test_autonomy_jobs_doc_claims.py:3940-3950 uses for #98's and #60's own
# task files (`_RETAG_JOB`, `_ALARM_JOB`).

_JOB60 = "60-knowledge-health-report.md"
_JOB94 = "94-referential-integrity-ledger.md"
_SKILL_KHR = ("skills", "knowledge-health-report", "SKILL.md")


def _vault(*parts):
    from tests.board_presence import vault_root

    path = vault_root().joinpath(*parts)
    assert path.is_file(), f"{path} is not on disk, so the published text is unreadable"
    return path.read_text(encoding="utf-8")


def _front_matter(text):
    import yaml

    parts = text.split("---", 2)
    assert len(parts) == 3, "no front matter to read a description from"
    return yaml.safe_load(parts[1])


def test_job_60_names_itself_the_ledger_s_consumer_in_the_delivered_field():
    """`description` is the only front-matter field _build_task_prompt delivers to #60.

    So a consumer named anywhere else in that file is a fact about the job nobody runs
    would learn. The pre-existing sentence naming #65 and #48 has to survive too: it is
    the reason #60 was chosen as the surface at all.
    """
    fm = _front_matter(_vault("autonomy", _JOB60))
    desc = fm["description"]
    assert "referential-integrity ledger" in desc, desc
    assert "(#94)" in desc, f"the writing job is not named: {desc}"
    assert "Loaded-memory referential integrity" in desc, desc
    assert "dangling TOTAL is the standing" in desc, desc
    assert "newly-dangling delta is a finding" in desc, desc
    assert ("Research-queue-generator (#65) and entity-resolution-sweep (#48) "
            "read the report") in desc, (
        "the downstream readers that make #60 the right surface were dropped")


def test_the_skill_documents_the_section_and_its_baseline_rule():
    """The skill is what a person running #60 by hand reads; it must not contradict the code."""
    body = _vault(*_SKILL_KHR)
    assert "Loaded-memory referential integrity" in body, body
    assert "consumer of #94" in body, body
    assert "dangling *total* is the standing baseline" in body, body
    assert "newly dangling" in body.lower(), body
    for state in ("ledger absent", "ledger unreadable"):
        assert state in body, f"{state!r} not documented in the skill"


def test_the_ledger_task_names_60_as_its_reader_and_no_longer_denies_a_surface():
    """§Where the one line goes used to argue that no #60 surface exists.

    It cited `grep -in ledger autonomy/60-knowledge-health-report.md` returning 0 hits,
    which was true when written and is false now, so the section had become a published
    claim against the fleet's own state. It must now name #60 as the reader, keep the
    baseline-vs-delta rule this file already reasoned to, and still confine this task's
    writes to the ledger and its dated copy — #94's prohibitions are not this round's to
    lift, and the run's own acceptance-graded summary line stays what the scheduler reads.
    """
    body = _vault("autonomy", _JOB94)
    heading = [ln for ln in body.splitlines() if ln.startswith("## Where the one line goes")]
    assert len(heading) == 1, heading
    assert "why not into #60" not in heading[0], heading[0]
    section = body.split(heading[0], 1)[1].split("\n## ", 1)[0]
    assert "#60" in section, section
    assert "knowledge-health-report.py" in section, section
    assert "Loaded-memory referential integrity" in section, section
    assert "newly" in section, section
    assert "returns 0 hits" not in section, (
        f"the stale 0-hits claim is still published: {section}")
    assert "only two paths this task" in section or "only two paths" in section, (
        f"the task's own write scope is no longer stated: {section}")
    assert "acceptance" in section, (
        "the summary-line surface the scheduler still grades was dropped")


def test_the_three_files_do_not_disagree_about_what_is_reportable():
    """The clause is that they agree, so the agreement is what gets pinned.

    All three must carry both halves of the rule on disk at once — a baseline total and
    a delta — because the failure this item recorded was not a wrong number but three
    documents that each described a different state of the world about whether anyone
    read the ledger at all.
    """
    docs = {
        "autonomy/" + _JOB60: _front_matter(_vault("autonomy", _JOB60))["description"],
        "skills/knowledge-health-report/SKILL.md": _vault(*_SKILL_KHR),
        "autonomy/" + _JOB94: _vault("autonomy", _JOB94),
    }
    for name, text in docs.items():
        low = text.lower()
        assert "baseline" in low, f"{name} states no baseline"
        assert "newly" in low, f"{name} never names the delta"
        # Only the two files that are not #60 have to name it: #60's own description
        # speaks as itself ("never written here"), so requiring the literal "#60" in it
        # would assert nothing about whether anyone else knows who reads the ledger.
        if not name.endswith(_JOB60):
            assert "#60" in text, f"{name} does not name #60 as the consumer"
