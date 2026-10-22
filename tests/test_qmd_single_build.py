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

import re
import subprocess
from pathlib import Path

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
    for helper in ("debounce()", "drain_for()"):
        assert text.index(helper) < pipe, f"{helper} must be defined above the pipeline"
    tail = text[pipe:]
    assert re.search(r"\|\s*\nwhile read -r; do", tail), "the loop is no longer the pipe's reader"


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
