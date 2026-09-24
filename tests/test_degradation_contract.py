"""What the gate's own tests rung must catch about the degradation matrix (#644).

Everything here runs on the plain `pytest -q` the gate's tests rung types, which is why
it is the file that holds the clauses whose whole point is that they hold *there*: the
matrix's shape, the `-m` widening, the row ids that cannot regress, the declared string
each blinded consumer must emit, and the four process-boundary seams at the bottom of
the file. The sibling `tests/degradation/test_rows.py` is the suite that walks all twenty
rows through every adapter; it carries the `fault_injection` marker and the rung skips
it, exactly as clause 4 requires of that suite.

The seam nodes are the deliberate exception to the "no sockets, no subprocess" reading
of that split, and they are scoped to be nothing else: each crosses one boundary — a
subprocess running the CLI, a subprocess running the skill's own gate block, a loopback
TLS listener with a throwaway certificate, a fixture git repo being repacked — and
asserts the crossing rather than the row's verdict. They stay unmarked because a seam
asserted only in the deselected file is not verified by the gate that is supposed to be
guarded: the refusal, the CLI's exit code and the fact that the TLS fault really was a
TLS fault would each be green on a runner whose injection had silently stopped
happening. A row's verdict does not belong here; it would make the rung that must skip
the injector into a second copy of it.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEG = ROOT / "tests" / "degradation"
MATRIX_PATH = DEG / "matrix.yaml"
RUNNER_PATH = DEG / "runner.py"
GATE_PATH = ROOT / "scripts" / "automod" / "gate.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.automod import gate as G
from tests.degradation import runner as R  # noqa: E402


def test_the_matrix_is_data_at_a_non_data_path_with_rows_over_dependencies():
    """The clause's own numbers: >=15 rows, >=8 distinct dependencies, and the file
    lives under ``tests/`` rather than in the gitignored ``data/`` tree a round cannot
    write. The row count comes from declared behavior, not from an endpoint scanner —
    every row names one consumer and one fault — which is the item's stated precondition
    for the construct being worth building at all.
    """
    assert MATRIX_PATH.is_relative_to(ROOT / "tests")
    rows = yaml.safe_load(MATRIX_PATH.read_text())["rows"]
    dependencies = {row["dependency"] for row in rows}
    assert len(rows) >= 15, f"only {len(rows)} rows"
    assert len(dependencies) >= 8, f"only {len(dependencies)} dependencies"
    assert not R.validate_rows(rows), "the matrix is malformed"

def test_every_row_declares_its_behavior_and_either_an_incident_or_unmotivated():
    """Rows are the contract, so a row missing its expected outcome is not a test — it
    is a green nothing. Incident links are the same discipline in the other direction:
    a row that cites a bug must cite one that exists, and a row invented without a
    bug has to say so out loud rather than read like it was earned.
    """
    rows = R.load_matrix()
    for row in rows:
        assert row.get("consumer", "").count("::") == 1, row["id"]
        assert row.get("dependency") and row.get("fault"), row["id"]
        assert row.get("declared_behavior", "").strip(), f"{row['id']} has no behavior"
        assert row.get("incident") == "unmotivated" or R._incident_exists(row["incident"]), (
            f"{row['id']} cites {row.get('incident')!r}, which resolves nowhere — a "
            f"prose citation is not evidence, so it is validated like any other claim")
    unmotivated = [row["id"] for row in rows if row.get("incident") == "unmotivated"]
    assert unmotivated, "a matrix with no `unmotivated` row is a matrix nobody argued with"

def test_the_marker_is_unregistered_and_the_gate_carries_the_widened_expression():
    """Registration does not drive selection — `-m` matches the marks applied to an item,
    so these nodes are deselected by the expression whether or not the name appears in `ini`.
    What makes pinning the three facts together the whole point is that there is no second
    place the exclusion could hide: `pytest.ini` is a path the loop may not write and #644's
    acceptance check forbids editing it, so if this expression is ever dropped from the gate
    nothing else stands between a whole-suite run and 25 nodes that bind ports and write state
    files on a box whose worker pool may be live.
    """
    ini = (ROOT / "pytest.ini").read_text()
    assert "fault_injection" not in ini
    assert "markers =" in ini, "the file still registers live_vault; only it should"
    assert "degradation" not in ini, "testpaths must stay as the repo declares it"
    gate_src = GATE_PATH.read_text()
    assert gate_src.count('"-m", TESTS_MARK_EXPR') == 3, (
        "the whole-suite rung, the review-rung re-run and the base probe all select; "
        "leaving one behind would run fault-injection tests under that rung")
    assert f'TESTS_MARK_EXPR = "{G.TESTS_MARK_EXPR}"' in gate_src, (
        "the expression must be defined once in the gate and referenced at every site")
    # The cross-module claim, which no string comparison can fake: the expression the gate
    # carries must name the marker this suite's rows actually apply. Copying a literal into a
    # test would keep this green after the marker was renamed and the gate had stopped
    # excluding anything.
    assert f"not {R.MARKER_NAME}" in G.TESTS_MARK_EXPR.split(" and "), (
        f"the gate excludes {G.TESTS_MARK_EXPR!r}, which no longer names "
        f"{R.MARKER_NAME!r} — `pytest -q` would run the injector")
    assert "not live_vault" in G.TESTS_MARK_EXPR, "the pre-existing exclusion is still owed"
    rows_py = (DEG / "test_rows.py").read_text()
    assert "pytestmark = pytest.mark.fault_injection" in rows_py, (
        "the rows are collected by file, so the marker has to be the module-level one; "
        "without it the gate's -m expression excludes nothing")

def test_the_matrix_consumers_are_real_functions_not_copies():
    """A matrix whose consumer strings were documentation would silently test a copy of
    the code, which is the mocked-driver failure this whole construct rejects. Each
    string is resolved and called on every run, so a renamed function fails its row.
    """
    for row in R.load_matrix():
        path, _, attr = row["consumer"].partition("::")
        target = (ROOT / path) if (ROOT / path).exists() else (Path.home() / "obsidian" / path)
        assert target.exists(), f"{row['id']}: {target} does not exist"
        if path.endswith(".py"):
            # A python consumer is imported and the attribute looked up, so a rename
            # fails the row rather than testing a stale copy of the function.
            module, fn = R.resolve_consumer(row["consumer"])
            assert callable(fn), row["consumer"]
        else:
            # A markdown consumer is a procedure a person or a task runs, so the
            # verifiable claim is that the named step is really in the file.
            text = target.read_text()
            assert attr in text, f"{row['id']}: {attr!r} is not in {target}"

def test_the_suite_lands_without_data_or_a_pytest_ini_edit():
    """The acceptance check's own words about where this may live. `data/` is the
    gitignored runtime tree a round cannot write and the gate's scope check denies, so
    the matrix sits next to its runner under `tests/`; and `pytest.ini` is untouched,
    because the exclusion belongs to the gate's expression, not to the repo's config.
    """
    assert MATRIX_PATH.is_relative_to(ROOT / "tests" / "degradation")
    assert not list((ROOT / "data").glob("*degradation*")) if (ROOT / "data").exists() else True
    assert "fault_injection" not in (ROOT / "pytest.ini").read_text()
    assert DEG.is_dir()


# ------------------------------------------------- clause 5: calibration, fail-before-fix half
# Each of the three still-live incidents was fixed before this code half landed, so at test
# time the pre-fix consumer is nowhere on disk: `git -C ~/obsidian show 41f545e5^:skills/
# system-health-check/system_health_check.py` and `git -C ~/lloyd show add6692^:scripts/automod/
# promote.py` are the only places those verdicts survive. What survives in-tree is the matcher,
# so the fail-before-fix half is pinned by handing `verdict_problems` the recorded old verdict
# and requiring it to reject it. Every string below is quoted from those pre-fix bytes, not
# paraphrased, and the voice rows' is the honest one: the pre-fix checker contained no
# voice-media probe at all (`grep -cE "check_voice_media|net/udp|50000" on 41f545e5^` -> 0), so
# the old verdict for the media plane was no verdict, which is precisely how the report read
# green over zero bound media sockets.
PREFIX_VERDICTS = {
    "D-HTTP-404-IGNORED": (
        # The HTTPError branch, verbatim:
        #   # 404 means the server is answering, it just has no route there
        #   if e.code == 404 and endpoint.get('ignore_404'):
        #       row.update({'healthy': True, 'response_code': e.code, 'note': '404'})
        # Loaded and pointed at a fixture 404 listener, that branch returns
        #   {'healthy': True, 'state': None, 'note': '404', 'error': None}
        "",
        "healthy=True state=None note='404' response_code=404",
    ),
    "D-VAULT-COUNT-GIT-REPACK": (
        # Both sides of the comparison were
        #   sum(1 for p in root.rglob("*") if p.is_file())
        # so a `git gc` moved the safety number with zero notes lost. The real trip, quoted
        # from knowledge/software/guardian-data-damage-false-trip.md:184, reverted #1206 on
        # 2026-09-17 08:59:56Z reading "vault files dropped 6.7% (6075 -> 5667)" — 408 loose
        # objects, note count 5620 -> 5622 over the same window.
        "dropped",
        "promoted=-408 guardian=-408 vault files dropped 6.7% (6075 -> 5667) porcelain_empty=True",
    ),
    "D-VOICE-CLIENT-GRANTED-NO-MEDIA": ("", ""),
    "D-VOICE-PROC-UNREADABLE": ("", ""),
}


@pytest.mark.parametrize("row_id", sorted(PREFIX_VERDICTS))
def test_each_calibrated_row_rejects_the_verdict_the_old_code_gave(row_id):
    """Clause 5's fail-before-fix half, as a node the gate runs.

    This is the assertion that makes "the suite is calibrated on real incidents" more than a
    claim about the matrix's provenance comments: the same matcher that judges a live row must
    refuse the verdict the code produced before its fix. If somebody weakens a row until the
    old behaviour satisfies it again, this node is what goes red.
    """
    row = {r["id"]: r for r in R.load_matrix()}[row_id]
    reported, evidence = PREFIX_VERDICTS[row_id]
    problems = R.verdict_problems(row, reported, evidence)
    assert problems, (
        f"{row['id']} ACCEPTED the pre-fix verdict {reported!r} / {evidence!r}: the row no "
        f"longer distinguishes the fixed consumer from the broken one it replaced")


def test_the_matcher_still_accepts_a_matching_verdict():
    """The control the node above needs, because "rejects the old verdict" is also satisfied by
    a matcher that rejects everything. Each row's own declared verdict, with its declared
    substrings as the evidence, must produce no problems.
    """
    for row in R.load_matrix():
        expect = row["expect"]
        evidence = " ".join([str(expect["reported"])] + list(expect.get("contains", [])))
        assert R.verdict_problems(row, expect["reported"], evidence) == [], row["id"]


# ------------------------------------------------- clause 1's last half: the two CLOSED incidents
# Clause 1 does not stop at a row count. It requires the two incidents that are already fixed to
# be present as rows *whose declared behavior holds today* — regression rows, so the construct
# keeps watching the two cases it was not written to catch. `tests/degradation/test_rows.py`
# executes both (one node per row, ids `...[D-LOCK-STAMPED-PATH-MISSING]` and
# `...[D-DEP-UPSTREAM-PAUSED]`), but that file carries the `fault_injection` marker and the gate's
# tests rung deselects it — so the presence claim has to be asserted in a file the gate actually
# runs. Naming the ids here is also the anchor the parametrization has no other way to have: a row
# deleted from the matrix silently deletes its node there, and only this node can tell that a
# required row is gone rather than simply untested.
CLOSED_INCIDENTS = {
    # row id -> (the incident that motivated it, the verdict the broken code produced)
    "D-LOCK-STAMPED-PATH-MISSING": (
        "skills/dream-consolidation/SKILL.md",
        # Phase 0 gated on the mtime of `lloyd/.consolidate-lock` while Phase 5 stamped a path
        # whose parent had been deleted, so every stamp raised FileNotFoundError, `hours_since`
        # came out ~56 years and the 24 h gate passed on every run from 2026-09-03 to 09-09.
        # The honest pre-fix verdict is the empty string: the gate printed nothing at all.
        "",
    ),
    "D-DEP-UPSTREAM-PAUSED": (
        "knowledge/software/autonomy-dispatch-hold-reason.md",
        # Before `dependency_resolution_set()` (commit a726d33, #870) the resolution set came
        # from `_all_runnable_tasks`, which keeps only up_next/in_progress/failed — so a
        # `paused` upstream was invisible and the dependent started. That is the 09-08 nightly
        # chain inversion, and again the pre-fix consumer emitted no verdict: it just ran.
        "",
    ),
}


def test_the_two_closed_incidents_are_rows_whose_declared_behavior_holds_today():
    """Each closed incident is a named row, cites the artifact that records its incident, and
    declares a verdict where the broken code said nothing.

    The third assertion is the one that keeps this from being a presence check: the row must
    *refuse* the pre-fix outcome through the same matcher that judges a live run. Two rows whose
    expected behaviour had quietly loosened back to "says nothing, proceeds" would still satisfy
    a test that only asked whether the row existed.
    """
    rows = {row["id"]: row for row in R.load_matrix()}
    for row_id, (incident, pre_fix_verdict) in CLOSED_INCIDENTS.items():
        row = rows.get(row_id)
        assert row is not None, (
            f"{row_id} is clause 1's named regression row and is absent from the matrix; the "
            f"node that executes it in test_rows.py disappeared with it, silently")
        assert row["incident"] == incident, (
            f"{row_id} cites {row['incident']!r}, not the artifact that records its incident")
        reported = str(row["expect"]["reported"]).strip()
        assert reported, f"{row_id} declares no verdict"
        assert reported not in R.HEALTHY_TOKENS, (
            f"{row_id} declares {reported!r}, which reads as healthy")
        problems = R.verdict_problems(row, pre_fix_verdict, "")
        assert problems, (
            f"{row_id} ACCEPTS the pre-fix verdict {pre_fix_verdict!r}: the row no longer "
            f"distinguishes today's behaviour from the behaviour its incident describes")


# ---------------------------------------------------------------- clause 6: no silent healthy
# The rows whose fault removes the input the consumer reads, rather than degrading it. Flagged
# in the matrix (`expect.non_healthy`) so the set is data: a row that gains or loses that
# property moves into or out of these assertions on its own, and the denominator is asserted
# rather than assumed — a parametrization that quietly empties is the same class of defect this
# whole item is named for.

def test_the_unreadable_probe_rows_are_enumerated_and_all_name_a_verdict():
    """>=6 rows, >=4 distinct consumers, and no non-healthy row may declare a healthy expected
    token — a spec that expects `answered` from a probe it just blinded is a contradiction, and
    it would pass every other check in this file.
    """
    rows = R.load_matrix()
    blinded = [r for r in rows if r["expect"].get("non_healthy")]
    assert len(blinded) >= 6, f"only {len(blinded)} rows flag a removed probe"
    consumers = {r["consumer"] for r in blinded}
    assert len(consumers) >= 4, f"only {len(consumers)} distinct consumers: {sorted(consumers)}"
    for row in blinded:
        assert row["expect"]["reported"].strip(), f"{row['id']} names no verdict"
        assert row["expect"]["reported"] not in R.HEALTHY_TOKENS, (
            f"{row['id']} expects {row['expect']['reported']!r}, which reads as healthy")
    # The positive arm: a row satisfied by an empty verdict is not naming anything.
    for row in blinded:
        assert not R.verdict_holds(row, "", row["expect"].get("contains", [""])[0]), (
            f"{row['id']} is satisfied by a consumer that said nothing")

def test_the_matcher_refuses_a_healthy_or_a_silent_verdict_on_a_blinded_row():
    """Clause 6 driven directly, one assertion per failure shape rather than one that passes on
    either. `ignore_404` is the reason there are three arms and not one: that consumer emitted
    no state token at all *and* wrote `healthy: True` into the row, so a matcher that checked
    only the token would have passed it, and one that checked only the evidence would pass a
    consumer that names a real state while smuggling the flag in beside it.
    """
    blinded = {"id": "CLAUSE-6-SYNTH", "expect": {"reported": "no_probe", "non_healthy": True}}
    token = R.verdict_problems(blinded, "answered", "state=answered")
    assert any("reads as healthy" in p for p in token), token
    silent = R.verdict_problems(blinded, "", "nothing reported")
    assert any("empty verdict" in p for p in silent), silent
    smuggled = R.verdict_problems(blinded, "no_probe", "state=no_probe healthy=True")
    assert any("healthy=True" in p for p in smuggled), smuggled


# ---------------------------------------------------- clause 6: the ban, as a plain read
# The executed half lives in `tests/degradation/test_rows.py` and `test_degradation_consumers.py`,
# both under the `fault_injection` marker. What is pinned HERE is the part that needs no fault at
# all: that every blinded row's own declared verdict is a non-healthy string, and that the ban's
# vocabulary is the runner's. A gate run therefore checks the ban even on a box where the
# injector is skipped, which is the case that matters — the reviewer's objection was that the
# clause lived only in a deselected file.
_BLINDED = [row for row in R.load_matrix() if row["expect"].get("non_healthy")]
_BY_CONSUMER: dict[str, list] = {}
for _row in _BLINDED:
    _BY_CONSUMER.setdefault(_row["consumer"], []).append(_row)


def test_every_row_blinding_a_probe_declares_a_non_healthy_verdict():
    """Clause 6, structurally: each row whose fault removes the input flags `non_healthy`, and
    the verdict it declares is not a word that reads as healthy. The flag is what makes the set
    enumerable, so the assertion is written over the set rather than over a hand-list that
    drifts the day a consumer is renamed.
    """
    assert len(_BLINDED) >= 6, (
        f"only {len(_BLINDED)} rows carry expect.non_healthy; the ban's scope has narrowed to "
        f"{sorted(r['id'] for r in _BLINDED)}")
    for row in _BLINDED:
        reported = str(row["expect"]["reported"]).strip()
        assert reported, f"{row['id']} declares an empty verdict"
        assert reported not in R.HEALTHY_TOKENS, (
            f"{row['id']} declares {row['expect']['reported']!r}, which reads as healthy, and "
            f"the ban on a healthy answer is only real if this list contains the token")


def test_the_blinded_set_covers_every_consumer_that_reads_a_missing_input():
    """The same clause's coverage claim: the ban must reach every consumer that reads something
    that can go missing, so the set is grouped by consumer and the membership asserted rather
    than counted. `check_voice_media` is called out by name because its incident is the one where
    a healthy verdict with no backing probe cost Alan a voice threshold-tuning session.
    """
    assert len(_BY_CONSUMER) >= 4, (
        f"only {len(_BY_CONSUMER)} consumers enumerated: {sorted(_BY_CONSUMER)}")
    for ref in ("skills/system-health-check/system_health_check.py::check_voice_media",
                "skills/system-health-check/system_health_check.py::check_endpoints",
                "app/kg_store.py::KGStore",
                "autonomy.py::_is_task_due"):
        assert ref in _BY_CONSUMER, f"{ref} reads an input that can go missing but has no row"
    assert "skills/dream-consolidation/SKILL.md::LOCK_MISSING" in _BY_CONSUMER, (
        "the gate that passed on a missing lock must itself be in the banned set")


# ------------------------------------------------------------------ seam crossings
# Four nodes, four boundaries. Each calls the runner's own machinery — never a re-do of
# it in the test — because the review of the previous commit refused on exactly that:
# lifting the skill's block with a regex is not executing it, and asserting on
# `run_row`'s dict is not running the command. Each node injects into a fixture only:
# a loopback socket on an ephemeral port, a temp directory, a throwaway certificate. No
# live service is bound, probed or written, which is what clause 3 requires and why
# these can sit on the hard rung at all. They are not rows: none of them injects at one
# of Lloyd's real dependencies or judges a consumer's verdict, so none belongs to the
# marked set that clause 4 keeps out of this rung.
def test_the_one_command_is_the_surface_because_it_runs_as_a_subprocess():
    """The boundary pytest → `python -m tests.degradation.runner` is the clause, not the
    library call. Clause 2's "one command" is what a person or a gate rung types, so this
    invokes it the way they would — a child interpreter with the repo as its cwd, argv as
    argv — and asserts the child's own exit code and its own stdout.

    It selects the unopenable-store row: its fault is a temp file with a header of wrong
    bytes, so the node costs one child interpreter and cannot touch a live service. What
    this is NOT: a re-assertion of the twenty-row pass — that is clause 2's node in the
    deselected module, which is the right home for a run that binds sockets and loads the
    live health checker. What only the child can show is that the exit code and the line
    shape survive the process boundary: `run_matrix`'s return value can be green while the
    command a gate would type prints something unparseable, and the runner exists to produce
    the artifact a person reads out of a shell.
    """
    proc = subprocess.run([sys.executable, "-m", "tests.degradation.runner",
                           "--row", "D-KG-STORE-UNOPENABLE"],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, (
        f"the one command exited {proc.returncode} on a row the runner reports as holding:\n"
        f"{proc.stdout}\n{proc.stderr}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.count(" | ") >= 4]
    assert len(lines) == 1, (
        f"expected exactly one evidence line from a one-row run, got {lines}")
    fields = lines[0].split(" | ")
    assert fields[0] == "D-KG-STORE-UNOPENABLE", lines[0]
    assert len(fields) == 5, (
        f"the printed line has {len(fields)} fields, not the contract's five: {lines[0]}")
    assert fields[2] == "store-unavailable", (
        f"the child reported {fields[2]!r}, not the verdict the row declares: {lines[0]}")
    assert fields[3] == "ok", f"the child's verdict field is not a verdict: {lines[0]}"
    assert "StoreUnavailable" in fields[4], (
        f"the evidence field carried a verdict and no evidence: {lines[0]}")


def test_the_skill_gate_block_is_executed_rather_than_mirrored():
    """The boundary runner → subprocess running the skill's own Phase-0 block.

    `_probe_consolidation` lifts the block out of the vault's `SKILL.md`, rewrites only its
    two roots into a temp memory root, and runs the result in a child interpreter: the
    verdict comes from the same text a run executes, which is the thing the six-day
    permanently-open gate destroyed. Running it is only observable as a *difference*, so
    this node drives the block twice — no lock, and a lock stamped now — and requires the
    two answers to differ, each in its own declared way. With no lock the block must name
    itself unevaluable and print the *fixture's* lock path, which a probe whose roots were
    never rewritten cannot know; with a lock stamped this instant it must fall through to
    the time gate and report an elapsed-hours figure under 0.5 h, which is arithmetic on
    the mtime the harness wrote microseconds earlier rather than a verdict recited from a
    file.

    What this is NOT: the contract node above the seam block, which only proves the block is
    liftable and is silent about whether it answers. What it does not claim either: that the
    gate's *decision* is right, which needs a clock the row controls and belongs to the
    staleness rows in the marked suite.
    """
    missing, missing_evidence = R._probe_consolidation({"args": {"lock": "absent"}})
    assert missing == "LOCK_MISSING", (
        f"the executed Phase-0 gate said {missing!r} with no lock present: {missing_evidence}")
    assert "unevaluable" in missing_evidence, (
        f"a lock-free gate printed a verdict instead of naming itself unevaluable: "
        f"{missing_evidence}")
    assert ".consolidate-lock" in missing_evidence and "/tmp" in missing_evidence, (
        f"the block did not report the fixture's lock path, so it was not run against the "
        f"fixture memory root: {missing_evidence}")

    fresh, fresh_evidence = R._probe_consolidation({"args": {"lock": "fresh"}})
    assert fresh != missing, (
        f"a freshly stamped lock changed nothing in what the block printed ({fresh!r} / "
        f"{fresh_evidence!r}), so the block was evaluated once and canned, not executed "
        f"against the fixture state")
    assert fresh == "GATE_FAIL", (
        f"a lock stamped this instant passed the 24 h gate ({fresh!r} / {fresh_evidence!r}), "
        f"so the executed block is not reasoning about the age of the fixture lock")
    ages = re.findall(r"(\d+(?:\.\d+)?)h", fresh_evidence)
    assert ages and float(ages[0]) < 0.5, (
        f"the block reported {fresh_evidence!r}, which names no elapsed-hours figure under "
        f"0.5 h, so it did not compute an age from the mtime written moments ago — reading "
        f"an age it could not see is exactly how the 24 h gate stayed open for six days")
    # What this node cannot falsify, stated rather than glossed: a stub that printed a
    # hand-written `GATE_FAIL: only 0.0h …` would satisfy the two legs. Every other failure
    # mode the review named is caught — a block that is lifted but never executed answers
    # `no-verdict`, and a block whose roots were not rewritten names the real
    # `~/obsidian/lloyd/.consolidate-lock` instead of the fixture's temp path — and closing
    # the last one needs a clock the row controls, which belongs to the staleness rows.


#: A Phase-0 gate whose two roots are spelled *neither* as the extractor's first
#: version expected nor as the skill spells them today. Vault commit `2fdd5b32`
#: (2026-09-23) relocated `SESSIONS_DIR` from `~/lloyd/sessions` to
#: `~/lloyd-data/sessions`, and a rewrite keyed on the old right-hand side turned into
#: a silent no-op that left the lifted gate reading the live filesystem (#1426).
MOVED_ROOT_GATE = '''\
import sys
from pathlib import Path
MEMORY_ROOT = Path.home() / "moved-away" / "lloyd"
SESSIONS_DIR = Path.home() / "moved-away" / "sessions"
LOCK_FILE = MEMORY_ROOT / ".consolidate-lock"
print("LOCK:", LOCK_FILE)
print("SESSIONS_DIR_IS_DIR:", SESSIONS_DIR.is_dir())
'''


def test_the_gate_root_rewrite_follows_the_skill_whenever_a_root_moves():
    """The pairing between this probe and the skill it lifts is pinned, not assumed.

    The row above proves the *current* skill text is rewritten; it cannot prove the
    rewrite still works after the next relocation, because it matches the skill only
    in its present spelling. So this node hands the extractor a gate that sends both
    roots somewhere neither the old literal nor the current one appears, runs the
    result in a child interpreter — the same boundary the probe crosses, and the only
    place an un-rewritten root becomes observable rather than merely present in a
    string — and requires the child's own output to name the fixture. A rewrite that
    missed an assignment would print `/home/.../moved-away/...` from a process that
    was never isolated, which is precisely the report that read as a verdict while
    measuring `~/obsidian`.

    The second leg is the one that keeps the first honest: a gate the extractor
    cannot fully redirect has to raise, not run. A partially-redirected gate prints a
    real answer about the live vault, and the whole point of this row is that a
    printed answer is only worth having when its inputs are the fixture's.
    """
    with R.fixture_root("consolidation") as root:
        memory_root = root / "lloyd"
        memory_root.mkdir()
        sessions = root / "sessions"
        sessions.mkdir()

        rewritten = R._rewrite_gate_roots(MOVED_ROOT_GATE, memory_root, sessions)
        assert 'MEMORY_ROOT = Path(' in rewritten and 'SESSIONS_DIR = Path(' in rewritten
        assert "moved-away" not in rewritten, "the relocated literal survived the rewrite"
        script = root / "phase0.py"
        script.write_text(rewritten)
        result = subprocess.run([sys.executable, str(script)], capture_output=True,
                                text=True, timeout=30, cwd=str(root))
        assert result.returncode == 0, result.stderr
        output = result.stdout

    lock_line = next(line for line in output.splitlines() if line.startswith("LOCK:"))
    printed = lock_line.split("LOCK:", 1)[1].strip()
    assert printed == str(memory_root / ".consolidate-lock"), (
        f"the child gate reported {printed!r}, which is not the fixture's lock: the "
        "extractor rewrote neither or only one root, so the row would be reading "
        "~/obsidian while reporting a fault")
    R.assert_fixture_path(memory_root / ".consolidate-lock", root)
    assert "SESSIONS_DIR_IS_DIR: True" in output, (
        f"the child did not see the fixture sessions directory: {output!r}")

    # The negative leg: an assignment the top-level rewrite cannot reach — here one a
    # refactor moved inside a function, which is how a skill edit would stop being
    # liftable without changing a single path string.
    half_redirected = ("MEMORY_ROOT = Path.home() / 'obsidian' / 'lloyd'\n"
                       "def paths():\n"
                       "    SESSIONS_DIR = Path.home() / 'lloyd' / 'sessions'\n"
                       "    return SESSIONS_DIR\n")
    with pytest.raises(R.RowError, match="SESSIONS_DIR"):
        R._rewrite_gate_roots(half_redirected, memory_root, sessions)


def test_the_tls_faults_come_from_a_socket_that_really_negotiates_tls():
    """The boundary runner → loopback TLS listener with a throwaway certificate.

    Asserted at the handshake rather than through a consumer, because the fault this node
    owns is the injector's: a listener that quietly served plaintext under an `https://`
    URL, or handed out a file that was not a certificate, would still yield a
    plausible-looking row, and no row-level expectation would notice. The three facts
    proved are the three that make the scheme and trust faults real — the key pair is a
    key pair `openssl` wrote; a client that trusts it completes a handshake and gets the
    body back; and a client holding the box's default trust is refused *at the handshake*,
    which is the incident where the TLS-only frontend probed over plain HTTP reported the
    UI unusable while Alan was typing into it.

    The vault's health checker is deliberately not imported here, and that is the
    reviewer's own finding from the previous commit: a hard-rung node that imports it
    reddens every later round on a vault-side rename, so the consumer's *wording* for these
    faults is pinned in the deselected suite, where importing it is the point. What this is
    NOT: anything about this box's real certificate — the pair is minted per process into a
    temp directory and no service on the box is contacted.
    """
    import http.client
    import ssl

    cert, key = R._tls_material()
    assert cert.is_file() and key.is_file(), f"no throwaway key pair at {cert}"
    assert cert.read_text(encoding="utf-8").startswith("-----BEGIN CERTIFICATE-----"), (
        f"{cert} is not a PEM certificate, so no handshake could have used it")
    assert "PRIVATE KEY" in key.read_text(encoding="utf-8"), (
        f"{key} is not a private key, so the listener could not have completed a handshake")

    listener = R.FixtureListener(code=200, body=b"tls-ok", tls=True, schemes=("https",))
    try:
        trusting = ssl.create_default_context(cafile=str(cert))
        client = http.client.HTTPSConnection("localhost", listener.port,
                                             context=trusting, timeout=10)
        try:
            client.request("GET", "/")
            answer = client.getresponse().read()
        finally:
            client.close()
        assert answer == b"tls-ok", (
            f"a client that trusts the certificate received {answer!r}, so the listener "
            f"never served what the scheme and trust rows claim it serves")

        refused = None
        untrusting = http.client.HTTPSConnection("localhost", listener.port,
                                                 context=ssl.create_default_context(),
                                                 timeout=10)
        try:
            untrusting.request("GET", "/")
            untrusting.getresponse().read()
        except ssl.SSLError as exc:
            refused = exc
        finally:
            untrusting.close()
        assert refused is not None, (
            "a client holding the box's default trust accepted the throwaway certificate, "
            "so the trust fault was never injected and the row's `TLS TRUST FAILED` is word "
            "the runner invented")
        assert "CERTIFICATE_VERIFY_FAILED" in str(refused), (
            f"the handshake failed for a reason other than trust, so the row would name a "
            f"trust failure it did not inject: {refused}")
    finally:
        listener.close()


def test_the_count_row_mutates_a_real_git_repository_and_moves_no_note():
    """The boundary runner → the git subprocess set inside a fixture vault.

    `remove_loose_objects` is the whole incident: `git repack -a -d` collapses hundreds of
    loose objects into one pack, `git status --porcelain` stays empty, and the old private
    `rglob("*")` read that as six per cent of the vault's notes vanishing. So the node
    asserts the mutation happened at git's end *and* that neither counter moved, in one
    fixture — the two halves are one claim, since a fixture whose `.git` never moved would
    pass the counter assertion for the wrong reason, which is the exact defect the
    runner's own docstring refuses to allow ("a hand-made `.git` does not shrink on repack
    and the control that distinguishes 'the filter held' from 'the fixture never moved'
    would report the former either way").

    What this is NOT: the matrix's own `D-VAULT-COUNT-GIT-REPACK` row, which asserts the
    declared behaviour through the adapter; this asserts that the adapter's git really
    ran, by counting git's own files before and after.
    """
    import os
    promote_count = R.resolve_consumer(
        "scripts/automod/promote.py::count_vault_files")[1]
    guardian_count = R.resolve_consumer(
        "agent-services/guardian/guardian.py::count_vault_files")[1]
    with R.fixture_root("contract-vault") as root:
        vault = R._fixture_vault(root, notes=12, loose_objects=12)
        objects = vault / ".git" / "objects"

        def git_files():
            return sum(1 for dirpath, _dirs, files in os.walk(objects) for _f in files)

        before_files, before_promoted = git_files(), promote_count(vault)
        before_guardian = guardian_count(str(vault))
        assert before_files > 0, (
            "the fixture vault has no loose objects, so a repack here can prove nothing")
        R.remove_loose_objects(vault)
        after_files, after_promoted = git_files(), promote_count(vault)
        after_guardian = guardian_count(str(vault))
        assert after_files < before_files, (
            f"repacking left git's object count at {after_files} (was {before_files}), so "
            f"the fixture never had loose objects to fold and the counters below are green "
            f"for nothing")
        assert R.git_porcelain(vault).strip() == "", (
            "the fixture's working tree went dirty during repack, which is not the incident")
        assert before_promoted > 0, "the promoted counter saw no files at all"
        assert after_promoted == before_promoted, (
            f"a .git-only repack moved the promoter's count {before_promoted} → "
            f"{after_promoted}")
        assert after_guardian == before_guardian, (
            f"a .git-only repack moved the guardian's count {before_guardian} → "
            f"{after_guardian}")


def test_a_row_has_no_way_to_name_a_live_port_and_the_bind_guard_holds_anyway():
    """The clause-3 boundary on the TCP leg, in the file the gate actually runs.

    Two claims, one boundary. First the guard itself: `bind_fixture_socket` refuses a port
    that is already answering, naming the port and the word live — this round's review
    refused the previous commit for pinning the refusal only in the deselected suite, and a
    guard the guarded rung cannot see is not a guard. Second, and the reason the guard is
    belt-and-braces rather than the last line of defence: a row-built listener is
    constructed with no port at all, so no matrix row can aim a fault at a service serving
    Alan in the first place. The kernel chooses, the bind proves ownership, and liveness is
    checked before the bind because afterwards a fixture always looks like a live service.

    The held port here is one this node bound itself, ephemeral and local, which is what
    makes the test admissible on the hard rung. What this is NOT: the state-file half of
    clause 3 — refusing to age the real `.consolidate-lock` is pinned in the marked suite,
    against the real file, because that half cannot be demonstrated on a fixture.
    """
    import inspect

    held = R.FixtureListener()
    try:
        assert R.port_in_use(held.port), "the held fixture is not answering; nothing is held"
        with pytest.raises(R.InjectRefused) as caught:
            R.bind_fixture_socket(preferred=held.port)
        message = str(caught.value)
        assert "live" in message.lower(), message
        assert str(held.port) in message, (
            f"the refusal did not name the port it refused, so a report could not tell a "
            f"guard from a bug: {message}")
        named = set(inspect.signature(R.FixtureListener.__init__).parameters) - {"self"}
        assert "port" not in named and "preferred" not in named, (
            f"FixtureListener grew a way to name a port ({sorted(named)}), which is how a "
            f"matrix row gets to inject into a live dependency")
        built = R.FixtureListener()
        try:
            assert built.port != held.port, (
                f"a row-built listener reused the held port {built.port}")
            assert R.port_in_use(built.port), "the row-built listener is not answering"
        finally:
            built.close()
    finally:
        held.close()
