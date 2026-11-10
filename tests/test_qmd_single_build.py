"""One qmd on the machine: the fork in ~/lloyd/qmd.

Until 2026-09-19 the daemon served the index from the fork while the watcher,
the index-maintenance task and both setup scripts ran the published
`@tobilu/qmd` from bun's global install -- same version string (2.8.3),
different commit. Nothing had broken, because the fork's changes are
serve-side; what existed was a second definition of how the index gets built,
and a `qmd` on PATH that was not the one answering queries. These tests are the
reason a fifth caller written next month cannot bring that back quietly.

Nothing here needs the fork to be present: `qmd/` is a gitignored clone and is
absent from every automod worktree, so the pins read tracked text only.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from app import paths

ROOT = Path(__file__).resolve().parent.parent
FORK_JS = "lloyd/qmd/dist/cli/qmd.js"

DAEMON_CONF = ROOT / "agent-services/supervisor/conf.d/agent-qmd-daemon.conf"
WATCHER = ROOT / "agent-services/scripts/qmd-watcher.sh"
CLEANUP_UNIT = ROOT / "agent-services/systemd/lloyd-qmd-cleanup.service"
MAINTENANCE = ROOT / "scripts/maintenance/qmd_index_maintenance.py"
#: The architecture page's §5 states the same no-op walk cost the watcher's cost block
#: justifies the cooldown with. `tests/test_qmd_doc_claims.py` reads this file for other
#: sections and never once for that bullet, which is how it rotted into a present-tense
#: figure the script had already dated — so the node that reads it lives here, beside
#: the block it has to agree with.
ARCH_QMD = ROOT / "architecture/qmd.md"


def _code_lines(path: Path) -> list[str]:
    """Lines that execute: comments carry this file's history on purpose."""
    return [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if not ln.lstrip().startswith(("#", ";"))]


def _line(path: Path, pattern: str) -> str:
    hits = [ln for ln in _code_lines(path) if re.search(pattern, ln)]
    assert len(hits) == 1, f"{path.name}: expected exactly one line matching {pattern!r}, got {hits}"
    return hits[0]


def test_every_caller_runs_the_forks_cli():
    callers = {
        DAEMON_CONF: r"^command=",
        WATCHER: r"^QMD_CLI=",
        CLEANUP_UNIT: r"^ExecStart=",
        MAINTENANCE: r"^QMD_CLI\s*=",
    }
    for path, pattern in callers.items():
        line = _line(path, pattern)
        assert FORK_JS in line.replace("%h/", "lloyd/../").replace("$HOME/", "") or \
            "qmd/dist/cli/qmd.js" in line and ".bun" not in line, (
            f"{path.relative_to(ROOT)} does not run the fork: {line.strip()}")
        assert ".bun" not in line and "node_modules/@tobilu" not in line, (
            f"{path.relative_to(ROOT)} runs the published build: {line.strip()}")


def test_no_tracked_code_reaches_the_published_build():
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.split("\0")
    suffixes = {".py", ".sh", ".conf", ".service", ".timer", ".yaml", ".yml", ".ts", ".tsx"}
    bad = re.compile(r"\.bun/bin/qmd|node_modules/@tobilu/qmd|bun (install|add) -g @tobilu/qmd")
    offenders = []
    for rel in tracked:
        path = ROOT / rel
        if not rel or path.suffix not in suffixes or not path.is_file():
            continue
        if path == Path(__file__).resolve():
            continue
        for n, ln in enumerate(_code_lines(path), 1):
            if bad.search(ln):
                offenders.append(f"{rel}: {ln.strip()[:120]}")
    assert not offenders, "the published qmd is back:\n  " + "\n  ".join(offenders)


def test_the_maintenance_task_never_takes_retrieval_down():
    """It stopped the daemon on eight of its last eight runs to embed one to four
    documents: pending embeddings are never zero while the loop is writing, and
    the stop was a requirement of the published build's cleanup, not the fork's."""
    src = "\n".join(_code_lines(MAINTENANCE))
    assert 'supervisor("stop")' not in src
    assert 'supervisor("restart")' in src, "an unhealthy daemon must still be brought back"


def test_the_watcher_defines_its_helpers_before_the_pipeline():
    """`bash -n` passes with a function definition sitting between `inotifywait |`
    and `while read` -- and the loop then reads the script's stdin instead of
    the event stream. The first cut of the throttle shipped exactly that."""
    text = WATCHER.read_text()
    pipe = text.index("inotifywait -m")
    for helper in ("debounce()", "drain_for()", "watch_loop()"):
        assert text.index(helper) < pipe, f"{helper} must be defined above the pipeline"
    tail = text[pipe:]
    # Since 2026-11-10 the loop is the function `watch_loop` (so tests can source and
    # drive it); the pipe's reader is that call, with nothing defined in between.
    assert re.search(r"\|\s*\nwatch_loop\s*$", tail, re.M), "the loop is no longer the pipe's reader"
    assert "read -r" in text[text.index("watch_loop()"):pipe], (
        "watch_loop no longer reads the event stream")


def test_the_watcher_throttles_a_continuous_event_stream(tmp_path):
    """A no-op `qmd update` re-hashes every file of every collection `index.yml`
    declares and forces the daemon's in-memory vector index to rebuild on the next
    query. No corpus count is stated here, in this file or in the script it pins:
    the one this docstring carried was unreproducible within three weeks (#3161),
    and the shape to copy is the template's (#2089, `aa47d6ec`) — measure the corpus
    with `sqlite3 ~/.cache/qmd/index.sqlite "select count(*) from documents where active=1"`
    and read 7.5 s as the dated 2026-09-19 observation it was. The debounce alone let
    one run per write, which is what the cooldown below is for and what this test
    exercises."""
    calls = tmp_path / "calls"
    stub = tmp_path / "qmd-stub"
    stub.write_text(f"#!/bin/bash\necho \"$2\" >> {calls}\n")
    stub.chmod(0o755)
    text = WATCHER.read_text()
    text = re.sub(r"^QMD_CLI=.*$", f'QMD_CLI=("{stub}" "x")', text, flags=re.M)
    text = re.sub(r"^COOLDOWN_SEC=.*$", "COOLDOWN_SEC=3", text, flags=re.M)
    text = re.sub(r"^MAX_DEBOUNCE_SEC=.*$", "MAX_DEBOUNCE_SEC=1", text, flags=re.M)
    text = re.sub(r"^DEBOUNCE_SEC=.*$", "DEBOUNCE_SEC=1", text, flags=re.M)
    text, n = re.subn(r"^inotifywait -m .*\\\n.*\|\s*$",
                      "(while true; do echo ev.md; sleep 0.1; done) |", text, flags=re.M)
    assert n == 1, "could not swap the event source; the pipeline's shape changed"
    text = text.replace('mkdir -p "$SESSIONS"', ":")
    script = tmp_path / "watcher.sh"
    script.write_text(text)
    try:
        subprocess.run(["bash", str(script)], timeout=7, capture_output=True)
    except subprocess.TimeoutExpired:
        pass
    updates = calls.read_text().split().count("update") if calls.exists() else 0
    # 7 s of events every 0.1 s: unthrottled that is a cycle per debounce (5+);
    # with a 3 s floor it is two, three if the clock is unkind.
    assert 1 <= updates <= 3, f"{updates} index walks in 7 s under a continuous stream"


# --- #3161: the watcher's cost model states no figure nothing can measure -----
#
# `agent-services/scripts/qmd-watcher.sh` justifies `COOLDOWN_SEC` by telling its
# reader what a no-op `qmd update` costs. The two counts in that sentence had one
# author (`5dfe914b`, 2026-09-19) and no command behind either of them. The collection
# count described the tracked template's set, and the template's own header — "Between
# 2026-09-07 and 2026-09-19 nobody ran the re-sync" — records that the live `index.yml`
# the daemon actually reads had already dropped its 15th collection in that window, on
# or before the day the comment was stamped: no reader could have checked the figure
# against the running config. The file count predated #1064's retraction of the 514
# non-conversation session exports, and when #3161's triage measured the walk on
# 2026-10-22 it came in at roughly 1.7x below the quoted figure. `architecture/qmd.md`
# §5 quotes the surviving half — "a no-op `update` re-hashes every file (~7.5 s)" — as
# the cost model of a no-op update, which is why the block is pinned below rather than
# re-stamped: it is the sole stated justification for the cooldown, and a figure nobody
# can re-measure is one the next reader has to take on trust.
#
# The shape is #2089's (`aa47d6ec`, already applied to the tracked template's
# `sessions` block): state no count, name the query that measures the corpus, and date
# the one measurement you keep. The nodes below read the block out of the script and
# assert all three, so the next edit that reintroduces a remembered figure fails a test
# instead of outliving one. Landing that script edit needed a second change first: the
# path sat outside the gate's writable set — in neither `ALLOWED_GLOBS` nor
# `PROTECTED_GLOBS` in `scripts/automod/spec.py` — until #3162 admitted
# `agent-services/scripts/qmd-watcher.sh` to `ALLOWED_GLOBS` by verbatim name
# (fa00ec8f), which is what lets a diff carry this edit at all.
#
# The two figures #3161 is about are assembled from pieces rather than spelled out
# below, because the acceptance check is a grep over this tree: a denylist that quotes
# its own quarry can only ever report itself. The same need with the same remedy, and
# its own explanation of it, is `ROTTED_FIGURE` in `tests/test_architecture_index_parity.py`.
STALE_FILE_COUNTS = ("15" + ",8" + "00", "15" + ",7" + "78")
STALE_COLLECTION_COUNT = "15" + " collections"

#: The query that measures the corpus, verbatim, as both the watcher's cost block and
#: the throttle test's docstring state it. Asserted as a literal rather than a
#: paraphrase so the script and its test cannot drift into telling a reader two
#: different commands.
CORPUS_QUERY = "select count(*) from documents where active=1"

# --- #3165: the walk now has TWO dated figures, and the newer one is the current one --
#
# #3161 left `7.5 s` standing alone as the sole timed claim about a no-op walk, with a
# pointer saying another item owed a fresh measurement. The measurement exists: read off
# the watcher's own supervisor log on 2026-10-22, a walk is a median of 2 s, a mean of
# 1.8 s and a maximum of 39 s. The cost block now carries that beside the 2026-09-19
# figure and the pointer is gone. No cycle count is pinned here or in the block: the
# log is live, so two passes an hour apart do not agree on `n`, and a figure that
# only reproduces alongside a number nobody can re-obtain is the same #3161 defect
# with an extra step. The window the figures cover, why the date is the day of
# computation rather than the span, and which pairs the derivation throws away are
# the script's job — and the nodes below demand it said all three.
WALK_DATE = "2026-10-22"

#: The three figures, spelled exactly as the cost block must spell them, on the block's
#: date line. Kept as one tuple because clause 1's whole demand is co-location: median,
#: mean and max beside the date, on one physical line.
WALK_FIGURES = ("median 2 s", "mean 1.8 s", "max 39 s")


def _stale_in(text: str) -> str | None:
    """The first forbidden figure `text` carries, or None. One helper so a further
    stale number added to the fleet is checked by the same code everywhere."""
    for stale in (*STALE_FILE_COUNTS, STALE_COLLECTION_COUNT):
        if stale in text:
            return stale
    return None


def _suite_sweep() -> list[str]:
    """Every line in `tests/` carrying one of the two corpus figures, found the way
    #3161's verification finds it — a `git grep` of that tree, its `-n` and its
    `tests/` pathspec reproduced here so the node goes red on exactly the commit the
    item's check would.

    A sibling node used to scan only `Path(__file__)`, and SM_20261022_163807's review
    named that as an advisory: a suite green over a tree whose watcher still carried the
    count. So this runs the item's command rather than a Python scan of one file —
    fixed-strings, because the item wrote its pattern as prose and a regex reading of a
    comma-separated figure is a different pattern. The figures arrive assembled for the
    reason given above. Exit status 1 is grep's "no match", which is the outcome wanted,
    hence the `(0, 1)` assert instead of `check=True`, and anything louder than those two
    codes is asserted rather than read as clean.
    """
    args = ["git", "grep", "-n", "--fixed-strings"]
    for figure in STALE_FILE_COUNTS:
        args += ["-e", figure]
    run = subprocess.run(args + ["--", "tests/"], cwd=ROOT, capture_output=True, text=True)
    assert run.returncode in (0, 1), (
        f"git grep could not sweep tests/ (rc={run.returncode}): {run.stderr.strip()}")
    return [ln for ln in run.stdout.splitlines() if ln.strip()]


def test_the_throttle_test_states_no_corpus_figure_of_its_own():
    """#3161 clause 5: neither this file nor the suite around it gives the walk a size.

    Three scopes, because the previous round was told its scan was too narrow. The
    function's own `__doc__` is the object a reader sees when the throttle test fails;
    this whole file is where a comment could reintroduce the count somewhere the
    docstring check would not look; and `_suite_sweep` is the item's tree-wide grep over
    all of `tests/`, which is the scope its verification command is written in.

    The collection phrase is checked file-scoped while the file counts get the tree-wide
    sweep, and that split is deliberate: `tests/test_qmd_index_template.py`'s own rail
    forbids that phrase in SETUP.md's Collections section by writing it into an
    assertion, so a suite-wide ban on the same string would collide with a guard that is
    doing its job. The two file counts appear nowhere in the suite, so nothing is
    conceded by scoping the phrase this way. (This docstring names those figures by
    description only, for the reason given above: a node that quotes its quarry fails
    itself before it fails anyone.)
    """
    doc = test_the_watcher_throttles_a_continuous_event_stream.__doc__ or ""

    assert _stale_in(doc) is None, "the stale corpus count is back in the throttle test"
    assert _stale_in(Path(__file__).read_text()) is None, (
        "a stale corpus figure is back somewhere in the file that forbids them")
    hits = _suite_sweep()
    assert hits == [], f"a stale corpus figure is back in the suite: {hits}"
    assert CORPUS_QUERY in doc, "the docstring does not say how to measure the corpus"
    at = doc.find("7.5")
    assert at != -1, (
        "the docstring carries no timed figure, so dating one below attributes it to "
        "nothing — the gap SM_20261022_163807's review advisory named exactly")
    doc_line = doc.splitlines()[doc[:at].count("\n")]
    assert "2026-09-19" in doc_line, (
        "the 7.5 s figure is quoted in the docstring but not attributed to the day it "
        "was measured on the line that quotes it, which is the only reading that keeps "
        "it honest — and the same line the node below demands of the script")


def test_the_watcher_still_sets_its_cooldown_on_one_rewriteable_line():
    """#3161 clause 4: the shape the throttle test depends on, in the script #3161 edits.

    The throttle test above rewrites the cooldown with `^COOLDOWN_SEC=.*$`, so the
    constant has to stay one assignment at column 0: an edit that folded it into the
    comment block above it, or wrapped it, would leave that test rewriting a line the
    script no longer executes and still passing. It matters most for the edit #3161
    makes to the comment directly above that line, which is exactly the kind of edit
    that tends to swallow an assignment — and `bash -n` would not notice.
    """
    text = WATCHER.read_text()

    assert re.search(r"^COOLDOWN_SEC=60$", text, flags=re.M), (
        "COOLDOWN_SEC is no longer a single-line column-0 assignment")
    _, n = re.subn(r"^COOLDOWN_SEC=.*$", "COOLDOWN_SEC=3", text, flags=re.M)
    assert n == 1, f"the throttle test's rewrite matches {n} lines, not 1"


def _cooldown_cost_block(text: str) -> str:
    """The comment block that justifies `COOLDOWN_SEC`: the contiguous comment lines
    sitting directly above the assignment.

    Cut positionally rather than by keyword, so rewording the justification cannot step
    around the nodes below it. An empty block is itself the failure: three asserts
    against `""` all pass, which is the shape of a guard that has quietly stopped
    guarding — the same reason `_line` above asserts its hit count.
    """
    lines = text.splitlines()
    hits = [i for i, ln in enumerate(lines) if re.match(r"^COOLDOWN_SEC=", ln)]
    assert len(hits) == 1, f"expected exactly one COOLDOWN_SEC assignment, got {hits}"
    start = end = hits[0]
    while start > 0 and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    block = "\n".join(lines[start:end])
    assert len(block) > 200, (
        f"the cooldown's cost block is down to {len(block)} chars: the text these nodes "
        "exist to read is gone, and an assertion against it would pass by absence")
    return block


def test_the_watcher_cost_block_states_no_remembered_count():
    """#3161 clause 1, including the grep the item names as its own verification.

    Two checks because they catch different regressions. `_stale_in` looks for the exact
    figures this item was filed about, and it runs over the whole script because the
    acceptance check is a grep over the whole file: a count that merely moved out of the
    block would still be quoted by the next reader. The second check is the general
    rail — a count stated beside the unit it counts is precisely what makes a comment
    unreproducible, and a hand-written denylist cannot close a property that has to hold
    over whatever number someone types next.
    """
    text = WATCHER.read_text()
    block = _cooldown_cost_block(text)

    stale = _stale_in(block)
    assert stale is None, f"the cost block states a figure no command reproduces: {stale!r}"
    stale = _stale_in(text)
    assert stale is None, f"the script states a figure no command reproduces: {stale!r}"
    stated = re.findall(r"\d[\d,]*\s+(?:files?|documents?|collections?)\b", block)
    assert stated == [], f"the cost block states a bare count again: {stated}"


def test_the_watcher_cost_block_names_the_query_that_measures_the_corpus():
    """#3161 clause 2: the corpus is stated as the command that measures it.

    The literal is asserted rather than a paraphrase, because the whole point of the
    rewrite is that a reader can run what the comment says. `sqlite3
    ~/.cache/qmd/index.sqlite` is pinned beside it for the same reason: that SQL alone
    names a table in no database.
    """
    block = _cooldown_cost_block(WATCHER.read_text())

    assert CORPUS_QUERY in block, (
        "the cost block no longer says how to measure the corpus, which leaves the next "
        "writer nothing but a remembered number to reach for")
    assert "sqlite3 ~/.cache/qmd/index.sqlite" in block, (
        f"`{CORPUS_QUERY}` is there but not the database to run it against")


def test_the_watcher_cost_block_dates_its_seven_and_a_half_seconds():
    """#3161 clause 3's other half: the 7.5 s figure is attributed to its measurement
    rather than presented as today's cost.

    The block has a job — making a 60-second floor look like the right answer to a real
    cost — so "re-hashes every file ... before it discovers that one of them changed"
    has to outlive the edit that removed the counts. The 7.5 s may stay: it is the
    figure #3161's acceptance permits. What it may not do is read as today's price,
    which is exactly how SM_20261022_163807's review graded this clause `partial`: its
    tree still said "that costs 7.5s" in the present tense, attached to the counts
    clause 1 forbids.
    """
    block = _cooldown_cost_block(WATCHER.read_text())

    assert "re-hashes" in block and "every file" in block, (
        "the block no longer says a no-op walk re-reads the whole corpus, so it no "
        "longer explains what the cooldown is a floor under")
    assert "changed" in block, (
        "the block no longer says the walk finds that one file changed, which is the "
        "asymmetry the cooldown exists to bound")
    at = block.find("7.5")
    assert at != -1, "the 7.5 s measurement was deleted rather than dated"
    # Clause 3 asks for the figure and its date on ONE line: a window of a few hundred
    # characters would let an edit split them across a paragraph and still read as
    # attributed, which is how a dated figure drifts back into an undated one.
    line = block[:at].count("\n")
    stated = block.splitlines()[line]
    assert "2026-09-19" in stated, (
        f"the 7.5 s figure and its date are no longer on one line: {stated!r}")
    assert "measured" in stated, (
        "the 7.5 s figure is quoted on a line that never says it was measured")
    assert not re.search(r"(?:that|it)\s+costs\s+7\.5", block), (
        "the block is back to asserting 7.5 s as the current cost")

    # --- #3165: the block carries a SECOND timed line, and that one is the current cost.
    #
    # Held to the same one-line discipline as the 7.5 s, for the same reason: a figure
    # whose date sits in the previous sentence is a figure the next reader quotes
    # without it, which is how `architecture/qmd.md` came to state 7.5 s as though it
    # were today's price.
    at2 = block.find(WALK_DATE)
    assert at2 != -1, (
        f"the cost block no longer states the {WALK_DATE} walk measurement, so its only "
        "timed figure is the 2026-09-19 one again — the state #3161's owed entry existed "
        "to retire, and the reason this item was filed")
    assert block.find("7.5") < at2, (
        "the dated line moved above the 7.5 s line. The asserts above read the FIRST "
        "'7.5' in the block and demand 2026-09-19 on that same line, so inserting here "
        "silently attributes that date to whatever moved in front of it")
    stated = block.splitlines()[block[:at2].count("\n")]
    for figure in WALK_FIGURES:
        assert figure in stated, (
            f"{figure} is off the {WALK_DATE} line, which asks for the date and all "
            f"three figures together; that line is {stated!r}")
    assert "supervisor log" in stated, (
        f"the {WALK_DATE} line gives figures without saying what they were measured "
        "from, which is how a sample becomes a constant")


def _flat(block: str) -> str:
    """A comment block or bullet as one line.

    Both surfaces wrap at about 80 columns, so a phrase one of them uses can sit across
    two lines and a raw substring test reports it absent — the exact shape
    `tests/test_qmd_doc_claims.py` documents for `architecture/qmd.md` at its own
    `OLD_SUBLIMINAL_CLAIM`, and the reason it has its own copy of this helper.
    """
    return " ".join(ln.lstrip().lstrip("#").strip() for ln in block.splitlines())


def _line_with(text: str, needle: str, what: str) -> str:
    """The single physical line of `text` holding `needle`, asserted to exist."""
    at = text.find(needle)
    assert at != -1, f"{what} is gone from the text it was supposed to be in"
    return text.splitlines()[text[:at].count("\n")]


def _watcher_bullet() -> str:
    """§5's `**Watcher**` bullet of `architecture/qmd.md`, found by its label.

    Selected rather than sliced by line number, because the number moves every time the
    page grows, and asserted to be exactly one: two bullets claiming the label would
    make "the page says" mean nothing.
    """
    text = ARCH_QMD.read_text(encoding="utf-8")
    bullets = [b for b in text.split("\n- ") if b.lstrip().startswith("**Watcher**")]
    assert len(bullets) == 1, (
        f"{len(bullets)} bullets of architecture/qmd.md open with **Watcher**, so the "
        "node below is not reading a defined object")
    return bullets[0]


def test_the_watcher_cost_block_no_longer_defers_its_figure_to_another_item():
    """#3165 clause 2: the block stops owing its own number to another item.

    The item's verification is `git grep -n 'owes a re-measurement' -- agent-services/`,
    and that command is a false green: the sentence it bans WRAPS, ending
    `... not as today's cost -- #3161 owes a` at one line break and resuming
    `# re-measurement.` at the next, so a line-oriented grep could not have matched the
    phrase on the UN-edited tree either — it exits 1 at the base commit 461f6b4b. Two
    substring asserts over the script are therefore the real check: no `owes a`, no
    `re-measurement`, wrapped or not. The grep is run too, so the node still fails on a
    tree where the pointer comes back unwrapped. It is scoped to this script and not to
    `agent-services/`, because the one other `re-measurement` in that tree is
    `agent-services/voice/addressee.py`'s note about djev thresholds, which is a
    different sentence about a different thing.
    """
    text = WATCHER.read_text()

    assert "owes a" not in text, (
        "the cost block is deferring its figure to another item again")
    assert "re-measurement" not in text, (
        "the pointer that owed the measurement is back — check the wrapped spelling "
        "before trusting the grep that cannot see it")
    run = subprocess.run(
        ["git", "grep", "-n", "-e", "owes a re-measurement", "--", "agent-services/"],
        cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 1, (
        f"`git grep -n 'owes a re-measurement' -- agent-services/` matched "
        f"(rc={run.returncode}): {run.stdout.strip()}")


def test_the_watcher_cost_block_states_how_the_dated_figures_were_derived():
    """#3165 clause 4: the new figures come with a derivation a reader can run.

    A date on its own is the #3161 defect wearing a new number. The supervisor log's
    stamps are time-only and the file rotates at 10 MB, so these figures describe a
    rolling window rather than a day, and the one thing that survives the rotation is
    the method. Six things, each pinned as words the block has to use: the source (the
    watcher's own supervisor log), where to find it (`stdout_logfile` in the watcher's
    supervisor drop-in), what one walk is (a `Change detected` stamp to the next `Index
    updated` stamp), how wide the window is (since the last 10 MB rotation), which
    pairs the method discards (a negative gap, and one straddling a restart banner),
    and what the date means (the day of computation, not the span). The last two are
    not pedantry: a reader who follows the comment but not the dropped-pair rule turns
    a walk that crossed midnight into a negative cycle and a restart into a gap
    between two processes, and gets a different figure from the same file; and a
    reader who takes `2026-10-22` for the span takes a multi-day sample for a single
    day's cost, which is the reading that made 7.5 s look like a constant.
    """
    block = _flat(_cooldown_cost_block(WATCHER.read_text()))

    assert "watcher's own supervisor log" in block, (
        "the figures no longer name the log they came from")
    assert "stdout_logfile" in block and "agent-qmd-watcher.conf" in block, (
        "the block says 'supervisor log' but no longer says which file that is, so "
        "there is nothing for the next reader to open")
    assert "Change detected" in block and "Index updated" in block, (
        "the derivation no longer says which two stamps a walk is measured between")
    assert "10 MB rotation" in block, (
        "the window is unstated, which leaves the figures looking like an all-time "
        "average rather than what the log happens to still hold")
    assert "restart banner" in block and "negative" in block, (
        "the derivation no longer names the pairs it drops. The stamps are time-only, "
        "so a walk that crossed midnight comes out negative and a pair straddling a "
        "restart is a gap between two processes: a reader following the comment alone "
        "would pair them anyway and publish a different figure from the same file")
    assert "day these were computed" in block, (
        "the block no longer says 2026-10-22 is the day of computation, so the span the "
        "figures cover is a guess")


def test_the_architecture_page_dates_the_walk_the_script_dates():
    """#3165 clause 5: two surfaces, one account of what a no-op walk costs.

    The seam is prose in one file against prose in another, joined by no code and by no
    test until now: the script's cost block justifies the cooldown, `architecture/qmd.md`
    §5 tells whoever is diagnosing a stale index the same cost, and the page said
    `(~7.5 s)` in the present tense while the script had already dated that figure to
    2026-09-19. SM_20261022_163807's review graded that present-tense framing `partial`
    on #3161, and the reason it could survive a suite of green is that nothing read the
    bullet.

    So the page is graded against the SCRIPT'S line, not against a figure typed twice in
    this file: the three numbers are taken out of the cost block and demanded back in the
    bullet, which is what makes a future re-measurement in one file fail the other. Each
    of those comparisons runs against the block's own line first, because demanding a
    figure the block no longer states is demanding nothing. Presence in the bullet is not
    attribution either: a date parked anywhere in a ten-line bullet leaves the sentence
    quoting the figures free to read as today's cost, and reading it that way is what this
    page did before this diff — so each figure set is finally demanded inside the SENTENCE
    that carries the date belonging to it.
    """
    block = _cooldown_cost_block(WATCHER.read_text())
    dated = _line_with(block, WALK_DATE, "the dated walk figure")
    bullet = _flat(_watcher_bullet())

    assert "~7.5" not in ARCH_QMD.read_text(encoding="utf-8"), (
        "architecture/qmd.md is back to an undated `(~7.5 s)`, which is the "
        "present-tense framing this node exists to retire")
    assert "7.5 s" in bullet and "2026-09-19" in bullet, (
        "the page no longer carries the 2026-09-19 figure, so the two accounts have "
        "diverged in the other direction")
    for figure in WALK_FIGURES:
        assert figure in dated, (
            f"the cost block no longer states {figure}, so comparing the page to it "
            "would be comparing it to nothing")
        assert figure in bullet, (
            f"the launcher states {figure} and architecture/qmd.md does not: two "
            "costs for one walk again")
    # Presence in the bullet is not attribution. A date parked anywhere in a ten-line
    # bullet leaves the sentence quoting the figures free to read as today's cost, and
    # reading it that way is exactly what this page did before this diff, so each date
    # is demanded in the SAME sentence as the figures it dates — the one-line rule the
    # node above holds the script's cost block to, applied across a sentence wrap.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", bullet) if s.strip()]
    dated_sent = [s for s in sentences if WALK_DATE in s]
    assert dated_sent, (
        f"architecture/qmd.md quotes the walk without {WALK_DATE} anywhere in the "
        "bullet, which is how the figures read as today's cost")
    assert "median" in dated_sent[0] and "max" in dated_sent[0], (
        f"{WALK_DATE} is in the page but not in the sentence stating the figures it "
        f"dates — that sentence is {dated_sent[0]!r}")
    old_sent = [s for s in sentences if "7.5 s" in s]
    assert old_sent, "the page stopped stating the 7.5 s figure altogether"
    assert "2026-09-19" in old_sent[0], (
        "the page quotes 7.5 s in a sentence that never says when it was measured, "
        "which is the present-tense framing SM_20261022_163807 graded `partial`")
    # Negative control on the splitter, so the rule above is about a sentence and not
    # about a bullet: a date standing in its own sentence must not count as sitting
    # beside the figures. Without this the sentence asserts would pass on a page that
    # says the date nowhere near them, which is the green that hid the original rot.
    control = [s for s in re.split(
        r"(?<=[.!?])\s+",
        f"The walk costs median 2 s, mean 1.8 s and max 39 s. Sampled {WALK_DATE}.")
        if WALK_DATE in s]
    assert control and "median" not in control[0], (
        f"the sentence split merged a standalone date into the figures' sentence "
        f"({control!r}), so the co-location asserts above are a bullet-wide substring "
        "test and nothing more")


# ── #3508: the dated walk figures have a committed witness ────────────────────────
#
# `median 2 s, mean 1.8 s, max 39 s` in the cost block above and in architecture/qmd.md §5
# were computed on 2026-10-22 off the watcher's own supervisor log, which supervisord
# overwrites at 10 MB. #3165's owed entry found no committed copy of the sample behind
# them, so those three numbers were re-derivable only while a rotation happened to still
# hold the window they were cut from. The window is committed now: the verbatim stamp-line
# projection of the 2026-10-15 09:27 -> 2026-10-26 15:51 span, with a header that states the
# projection grep, the pairing rule and the commands that re-derive the figures.
#
# Why the bytes are mirrored in-tree: the gate runs this suite with `$HOME` pointed at the
# round's own home, where the vault is not, so a node that opened ONLY the vault copy would
# not fail on a hard gate rung -- it would skip, and a skipping node pins nothing. The same
# trap and the same answer are documented for a promotions extract at
# tests/test_automod_code_rules.py:2122. The copy is what makes the figures re-derivable on
# a gate. Where a vault IS reachable the same bytes are compared against it, which is what
# stops the two copies drifting -- but that comparison is an extra assert inside the node
# below rather than its own node, because a node of its own would have to skip where no vault
# is reachable, and this suite's skip count is read against a derived ceiling (`_skip_ceiling`
# in scripts/automod/gate.py) that a new skip withdraws from. The precedent's byte-comparison
# node does skip; this file buys the same protection without spending one.
#
# No path under the runtime data root is spelled anywhere in this file. That is the point:
# the published figures are reachable from committed bytes alone.

#: The witness's name in the vault's dated-witness store, and the copy this suite can read.
WALK_EXTRACT_BASENAME = "2026-10-22.3165-services-agent-qmd-watcher.log"
WALK_EXTRACT_VAULT_PATH = "backlog/data/" + WALK_EXTRACT_BASENAME
WALK_EXTRACT_COPY = (ROOT / "tests" / "fixtures"
                     / "qmd_watcher_walk_stamps_2026-10-22-item3508.log")

#: The ceiling the witness was cut under -- `WITNESS_MAX_BYTES` in
#: scripts/automod/backlog.py. Restated rather than imported, because that constant lives in
#: a maintenance script this suite does not otherwise load, and because a raised ceiling
#: should reach this file as a red node rather than as a silent import.
WITNESS_MAX_BYTES = 1_048_576

#: The projection's own size, as the extract's header quotes it: 8,357 stamp lines, 398,795
#: bytes -- 38.0 % of the ceiling above.
WALK_PROJECTION = (8357, 398795)

#: What the pairing rule prints over these bytes. The median, mean and max are the three
#: figures the cost block and §5 publish; n and p99 fall out of the same pass, and the
#: extract's header states them because a reader re-running it will see them.
WALK_DERIVED = {"n": 4170, "median": 2.0, "mean": 1.7583, "max": 39, "p99": 3}


def _walk_header(text: str) -> list[str]:
    """The extract's `#` header -- the lines its own re-derive commands skip.

    Kept separate from the body on purpose: the header quotes figures, the body produces
    them, and every statistic below runs on the body alone so the quoted figures cannot
    confirm themselves.
    """
    header = [ln for ln in text.splitlines() if ln.startswith("#")]
    assert len(header) > 40, (
        f"the extract's header is down to {len(header)} lines; the source, the projection "
        "grep, the window span, the pairing rule and the re-derive commands are what make "
        "these bytes checkable, and this node reads nothing else")
    return header


def _walk_body(text: str) -> list[str]:
    """The projection itself: every line that is not part of the `#` header.

    `grep -v '^#'` in the header's own first re-derive command, so re-deriving here and
    re-deriving by following the header are the same operation.
    """
    body = [ln for ln in text.splitlines() if not ln.startswith("#")]
    assert body, "the extract has no body lines once its '#' header is removed"
    return body


def _walk_gaps(body: list[str]) -> list[int]:
    """The pairing rule as the extract's header states it, over the extract's own lines.

    One walk is the gap from an `HH:MM:SS Change detected` stamp to the NEXT `HH:MM:SS Index
    updated`. A negative gap is a pair that crossed midnight and is dropped; a pending
    `Change detected` is voided by a `Watching` restart banner, because a gap straddling a
    banner belongs to two processes rather than one walk. Both drops are implemented even
    though neither fires on these bytes (0 of each): the rule has to be the one the header
    describes, not the one that happens to reproduce one particular window.
    """
    gaps: list[int] = []
    pending: int | None = None
    for line in body:
        m = re.match(r"^(\d\d):(\d\d):(\d\d) (Change detected|Index updated|Watching )", line)
        if not m:
            continue
        secs = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
        if m[4] == "Change detected":
            pending = secs
        elif m[4] == "Watching ":
            pending = None
        elif pending is not None:
            gap, pending = secs - pending, None
            if gap >= 0:
                gaps.append(gap)
    assert len(gaps) > 1000, (
        f"the pairing produced {len(gaps)} gaps from {len(body)} stamp lines, which is not a "
        "sample of a window any more -- check the stamp regex against the body's shape")
    return gaps


def _walk_stats(gaps: list[int]) -> dict:
    """median / mean / max / p99, the way the header's `sort -n` pipeline computes them.

    p99 is the value AT the 99th percentile position (round up), which is what the `sort -n`
    + index the header prints selects -- not an interpolation, which would agree here and
    disagree on a sparser window.
    """
    ordered = sorted(gaps)
    n = len(ordered)
    median = (ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0)
    return {"n": n, "median": float(median), "mean": sum(ordered) / n,
            "max": ordered[-1], "p99": ordered[int(-(-0.99 * n // 1)) - 1]}


def test_the_committed_walk_extract_reproduces_the_published_walk_figures():
    """#3508 clauses 1, 2 and 4: the witness is these bytes, and these bytes print the
    figures the script and the architecture page state.

    Read from the in-tree copy alone. Nothing here opens a log: that is the property #3165's
    owed entry was after -- a number that survives only while a rotating file happens to hold
    the window it was measured in is not a published figure, it is a rumour with a date.
    """
    # Built by concatenation, exactly as the path-rail nodes elsewhere in this suite build
    # theirs: the marker has to survive appearing in the file it is scanning for.
    runtime_root_marker = "lloyd" + "-data"
    assert runtime_root_marker not in Path(__file__).read_text(encoding="utf-8"), (
        "this file now names the runtime data root, so the node below can no longer be "
        "trusted to be reading committed bytes rather than the live log")
    assert WALK_EXTRACT_COPY.exists(), (
        f"{WALK_EXTRACT_COPY.name} is gone, and it is the only copy of the 2026-10-22 walk "
        "sample this suite can read")
    raw = WALK_EXTRACT_COPY.read_bytes()
    assert len(raw) <= WITNESS_MAX_BYTES, (
        f"the extract is {len(raw):,} bytes, over the {WITNESS_MAX_BYTES:,} ceiling a witness "
        "is cut to fit; the projection is 38.0 % of it, so this is growth, not a mis-set bound")
    text = raw.decode("utf-8")
    body = _walk_body(text)

    # clause 2's bounds, which are the two rotation edges: the first stamp is the line the
    # watcher printed as the 2026-10-15 rotation handed it the file, the last is the moment
    # the 2026-10-26 rotation ended the window. They pin WHICH window these bytes are -- the
    # figures themselves would survive a wrong window, and a wrong window is a different
    # measurement wearing these numbers.
    assert body[0] == "09:27:24 Index updated, embedding new documents...", (
        f"the extract's first body line is {body[0]!r}, not the 2026-10-15 09:27 rotation edge")
    assert body[-1] == "15:51:03 Change detected, updating index...", (
        f"the extract's last body line is {body[-1]!r}, not the 2026-10-26 15:51 rotation edge")
    assert (len(body), sum(len(ln) + 1 for ln in body)) == WALK_PROJECTION, (
        f"the projection is {len(body):,} lines / "
        f"{sum(len(ln) + 1 for ln in body):,} bytes, not the "
        f"{WALK_PROJECTION[0]:,} / {WALK_PROJECTION[1]:,} its header quotes -- either the "
        "window or the grep behind these bytes changed")

    stats = _walk_stats(_walk_gaps(body))
    assert stats["n"] == WALK_DERIVED["n"], (
        f"the pairing yields n={stats['n']}, not {WALK_DERIVED['n']:,}: the published median "
        "was computed over the old sample, so re-derive the cost block before quoting it")
    assert stats["median"] == WALK_DERIVED["median"], (
        f"median {stats['median']} from the committed witness, "
        f"{WALK_DERIVED['median']} published")
    assert abs(stats["mean"] - WALK_DERIVED["mean"]) <= 0.05, (
        f"mean {stats['mean']:.4f} from the committed witness, "
        f"{WALK_DERIVED['mean']} published (tolerance 0.05: the cost block rounds it to "
        "1.8 s, and a mean that moved past a rounding step is a different sample)")
    assert stats["max"] == WALK_DERIVED["max"], (
        f"max {stats['max']} s from the committed witness, {WALK_DERIVED['max']} s published "
        "-- and the cooldown's stated margin is argued against exactly this tail")
    assert stats["p99"] == WALK_DERIVED["p99"], (
        f"p99 {stats['p99']} s from the committed witness, {WALK_DERIVED['p99']} s in its "
        "header; neither prose surface publishes p99, so a drift here says the window and "
        "the header have parted rather than that a published figure is stale")

    # Where the vault is reachable, the two copies must be the same bytes. Unconditional
    # would be a skip in a round home (see the block above), so this is an extra assert on a
    # node that already stood on its own: the copy in this tree is a mirror, and a mirror that
    # drifts is how a green suite ends up vouching for sample nobody published.
    vault_copy = Path(paths.VAULT_ROOT) / WALK_EXTRACT_VAULT_PATH
    if vault_copy.exists():
        assert vault_copy.read_bytes() == raw, (
            f"the vault's {WALK_EXTRACT_VAULT_PATH} and this suite's "
            f"{WALK_EXTRACT_COPY.name} have parted: one of them is no longer the 2026-10-22 "
            "window, and the figures are pinned to whichever the header's `cmp` still passes")

    # The tie back to the prose. A witness that reproduces a number nobody publishes, next to
    # a script that publishes a number no witness reproduces, is two green tests and the same
    # rot -- so the derived figures must equal the strings the cost block states.
    block = _cooldown_cost_block(WATCHER.read_text(encoding="utf-8"))
    line = _line_with(block, WALK_DATE, "the date-stamped line in the cost block")
    assert f"median {int(round(stats['median']))} s" in line, (
        f"the cost block states a median the committed witness does not print "
        f"({stats['median']} s derived): {_flat(line)!r}")
    assert f"mean {stats['mean']:.1f} s" in line, (
        f"the cost block states a mean the committed witness does not print "
        f"({stats['mean']:.4f} derived): {_flat(line)!r}")
    assert f"max {stats['max']} s" in line, (
        f"the cost block states a max the committed witness does not print "
        f"({stats['max']} s derived): {_flat(line)!r}")


def test_the_walk_extract_header_states_its_own_source_rule_and_rederive():
    """#3508 clause 3: the header is a route back to the store, not a caption.

    Bytes that cannot be traced to a source and re-derived from it are a second prose copy of
    the same figures -- the thing #3165 owed. So the header must name the file it came out of
    and WHEN that file was cut, the grep that produced the body verbatim, the span the window
    covers, the pairing rule including what it drops, and both re-derive commands.
    """
    assert WALK_EXTRACT_COPY.exists(), f"{WALK_EXTRACT_COPY.name} is gone"
    header = _walk_header(WALK_EXTRACT_COPY.read_bytes().decode("utf-8"))
    joined = "\n".join(header)

    assert re.search(r"^# Source store\s+: .*agent-qmd-watcher\.log\.1", joined, re.M), (
        "the header stopped naming the store these lines came out of")
    assert "mtime 2026-10-26 15:51" in joined and "2026-10-26T22:51:04Z" in joined, (
        "the header no longer gives the source's mtime in BOTH the form `ls` prints and UTC; "
        "this box is UTC-7/UTC-8, so a reader holding only the listing form is 7 hours wrong "
        "about when the window was cut")
    # The grep verbatim, so copying it out of the header is the operation rather than a
    # paraphrase of it. It is the same string clause 2's cmp runs.
    assert ("grep -E '^[0-9]{2}:[0-9]{2}:[0-9]{2} (Change detected|Index updated"
            "|Watching )'") in joined, (
        "the projection grep is no longer quoted verbatim, so a reader cannot reproduce the "
        "body by copy-paste and will paraphrase it instead")
    assert "2026-10-15 09:27:21" in joined and "2026-10-26 15:51:04" in joined, (
        "the header lost one of the two rotation stamps that bound the window; the span is "
        "inferred from the mtimes of the two .N files, and without both it is a guess")
    assert "PAIRING RULE" in joined and "negative gap" in joined and "Watching" in joined, (
        "the pairing rule is no longer stated with what it drops. It reports zero drops on "
        "these bytes, which is exactly why the rule has to be written down: a reader on a "
        "longer window WILL hit both drops, and a pairing that silently omits them is a "
        "different statistic")
    assert 'wc -l -c < "$EX"' in joined, "the header's size re-derive went missing"
    assert "median=" in joined and "p99=" in joined, (
        "the header no longer carries the pairing re-derive pipeline, so the figures in it "
        "are assertions rather than commands")
    # A header that quotes a figure is a claim; a header that prints the command is checkable.
    # Positive control on the two commands above: they must be inside the header, which is
    # what makes them re-derivable BY the header and not merely mentioned near it.
    assert sum(1 for ln in header if ln.startswith("# ")) >= 40, "the header is not a header"


def test_the_watcher_cost_block_names_the_walk_extract_below_its_date():
    """#3508 clause 5: the cost block points at the witness, placed where the other nodes
    can still read it, without losing the file's mode.

    Placement is the whole node. The extract's name begins with the same date the figures
    line carries, so a pointer inserted ABOVE it becomes the block's first `2026-10-22` -- and
    the node that demands that line hold the figures would then read the pointer, fail, and
    look like the witness had broken the script.
    """
    text = WATCHER.read_text(encoding="utf-8")
    block = _cooldown_cost_block(text)
    assert WALK_EXTRACT_BASENAME in block, (
        "the cooldown's cost block no longer names the committed witness, so a reader at the "
        f"figures has nowhere to go -- {WALK_EXTRACT_VAULT_PATH} is that witness")
    at_date = block.find(WALK_DATE)
    at_extract = block.find(WALK_EXTRACT_BASENAME)
    assert at_date >= 0, f"the cost block lost {WALK_DATE} entirely"
    assert at_date < at_extract, (
        f"the extract pointer (offset {at_extract}) sits ABOVE the {WALK_DATE} figures line "
        f"(offset {at_date}); `test_the_watcher_walk_figure_sits_on_one_line` reads the "
        "block's FIRST 2026-10-22 and would then be handed a file name instead of the "
        "figures")
    assert (_line_with(block, WALK_DATE, "the date-stamped line")
            == _line_with(block, "median 2 s", "the median line")), (
        "the date-stamped line and the figures line drifted apart, so the node that pins "
        "them together is pinning one line")
    assert not re.findall(r"\d[\d,]*\s+(?:files?|documents?|collections?)\b",
                          _line_with(block, WALK_EXTRACT_BASENAME,
                                           "the extract pointer line")), (
        "the pointer line quotes a corpus count, which is the class of number #3161 retired "
        "from this file because the tree outgrows it")

    # The exec bit. `git ls-files -s` is the mode that gets promoted, and an edit that
    # silently drops it leaves a supervisor that can no longer start the program.
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-s",
                              str(WATCHER.relative_to(ROOT))],
                             capture_output=True, text=True).stdout.strip()
    assert tracked.split()[0] == "100755", (
        f"the watcher is tracked as mode {tracked.split()[0] if tracked else 'NOT TRACKED'}; "
        "supervisord execs it, so 100644 is a service that will not start")
    assert os.access(WATCHER, os.X_OK), (
        "the working copy is not executable even though the index says 100755 -- an Edit "
        "that rewrote the file without restoring the bit")
