"""#1769: `scripts/maintenance/relabel_stale_bench_calibration.py`.

A staged bench-mine note is two documents: `write_staging_note`'s envelope front
matter, and the mined candidate task block in the body. Before #1710 the
calibration ran against the ENVELOPE — no prompt, no checks, the objective layer
awarded by default — yet the verdict it produced was stamped onto the note as
`calibration.status: ok` with `in_band: true`. #1710 fixed the writer, so a note
whose `calibration` block carries a `task_id` is a measurement of a task; one
that does not is a measurement of nothing, and every one of the notes staged
before that commit (119 of them as at 2026-09-29) is in that second set while
sitting in the promotion queue reading "inside the capability edge".

This script is the relabel: it rewrites exactly the notes with no
`calibration.task_id`, states plainly what the numbers were measured against,
voids the band verdict (`in_band: null`), and leaves the candidate task block —
the one part a human promotes — byte-for-byte alone.

Every test builds its own staging tree under `tmp_path` and passes `--root`.
Nothing here reads or writes the live `~/lloyd-data` staging root, which is
#1769's owed step and a human's measurement, not a test's.
"""
from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "maintenance" / "relabel_stale_bench_calibration.py"

#: The three values the contract names, spelled once so a drift between the
#: script's constants and the queue's expectation is a failing test rather than
#: a note that reads three different ways.
MEASURED_AGAINST = "staging envelope (pre-#1710)"
STALE_STATUS = "stale_envelope"
UNCALIBRATED = "uncalibrated"

#: review_status values that still assert a band verdict to the promotion UI.
STILL_ASSERTING = ("pending", "out_of_band")


def _load():
    spec = importlib.util.spec_from_file_location(
        "relabel_stale_bench_calibration", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rs = _load()


# ---------------------------------------------------------------------------
# fixture: staged notes written the way the writer writes them
# ---------------------------------------------------------------------------

def _task_block(task_id: str = "bench_099_mined_probe") -> str:
    """The mined candidate's bench-TASK block, as the mining turn's answer lands
    it in the body (`_stage_and_calibrate` writes the turn's text verbatim)."""
    fm = {
        "segment": "lloyd",
        "id": task_id,
        "category": "synthetic",
        "objective": "Name the skill that covers the symptom.",
        "max_tool_calls": 5,
        "requires_runtime": True,
        "prompt": "A probe died with exit 127 naming the interpreter. What now?",
        "objective_checks": [
            {"type": "tool_called", "value": "mcp__lloyd-mcp__skills_search"},
            {"type": "contains", "value": "probe-interpreter-mismatch"},
        ],
        "rubric_criteria": ["skill_awareness"],
    }
    return ("---\n"
            + yaml.dump(fm, default_flow_style=False, allow_unicode=True)
            + "---\n\nSuccess is the skill named, not the neighbour it "
              "resembles.\n")


def _calibration(*, task_id: str | None = None, status: str = "ok",
                 in_band: bool | None = True, mean: float = 0.6625,
                 measured_against: str | None = None) -> dict:
    """A `calibration` block in each of the two shapes the live tree holds."""
    cal: dict = {"band": [0.05, 0.95], "composites": [mean] * 10,
                 "error": "", "in_band": in_band, "max": mean, "mean": mean,
                 "min": mean, "runs": 10, "status": status}
    if task_id is not None:
        cal["task_id"] = task_id
    if measured_against is not None:
        cal["measured_against"] = measured_against
    return cal


def _note(path: Path, *, calibration, review_status: str = "pending",
          body: str | None = None, task_id: str = "bench_099_mined_probe") -> Path:
    """One staged note, byte-shaped like `workers.sources._common.write_staging_note`:
    `---\\n{envelope}---\\n\\n{body}` — envelope first, candidate task in the body."""
    fm = {"source": "bench-mine", "confidence": 0.5,
          "review_status": review_status,
          "rationale": "derived from baseline loss on bench_007_skill_invocation",
          "source_refs": ["~/obsidian/lloyd/bench/bench_007_skill_invocation.md"],
          "generated_at": "2026-09-27T00:05:45.978938+00:00"}
    if calibration is not None:
        fm["calibration"] = calibration
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\n"
                 + yaml.dump(fm, default_flow_style=False, allow_unicode=True)
                 + "---\n\n" + (body if body is not None else _task_block(task_id)),
                 encoding="utf-8")
    return path


def _build(root: Path) -> dict:
    """A staging tree holding every shape the sweep found, plus the shapes it
    must not touch. Returns the paths by role, and the ones whose bytes are the
    assertion (`untouched`)."""
    d27, d28 = root / "2026-09-27", root / "2026-09-28"
    stale = {
        # the 118: envelope measurement, stamped in-band
        "pending_in_band": _note(d27 / "000545-pending-in-band.md",
                                 calibration=_calibration()),
        # the 1: envelope measurement, stamped out-of-band
        "out_of_band": _note(d27 / "010101-out-of-band.md",
                             calibration=_calibration(in_band=False, mean=0.97),
                             review_status="out_of_band"),
        # a fenced candidate body — 4 of the live notes wrap the task in ```
        "fenced": _note(d28 / "020202-fenced.md",
                        calibration=_calibration(),
                        body=f"```\n{_task_block('bench_098_fenced_probe')}```\n"),
        # the calibration block is missing entirely: still no task_id, so still
        # a note that asserts nothing about a task
        "no_calibration_key": _note(d28 / "030303-no-calibration-key.md",
                                    calibration=None),
        # an empty block, same reading
        "empty_calibration": _note(d28 / "040404-empty-calibration.md",
                                   calibration={}),
        # already relabelled by an earlier run: rewriting it must change nothing
        "already_stale": _note(d28 / "050505-already-stale.md",
                               calibration=_calibration(status=STALE_STATUS,
                                                        in_band=None,
                                                        measured_against=MEASURED_AGAINST),
                               review_status=UNCALIBRATED),
    }
    calibrated = {
        # the 4 post-#1710 notes: a mean that IS a measurement of a task
        "calibrated_out_of_band_1": _note(
            d27 / "052247-calibrated-one.md",
            calibration=_calibration(task_id="bench_052_probe", in_band=False,
                                     mean=0.97),
            review_status="out_of_band", task_id="bench_052_probe"),
        "calibrated_out_of_band_2": _note(
            d27 / "063300-calibrated-two.md",
            calibration=_calibration(task_id="bench_063_probe", in_band=False,
                                     mean=0.99),
            review_status="out_of_band", task_id="bench_063_probe"),
        "calibrated_out_of_band_3": _note(
            d28 / "083119-calibrated-three.md",
            calibration=_calibration(task_id="bench_083_probe", in_band=False,
                                     mean=0.98),
            review_status="out_of_band", task_id="bench_083_probe"),
        "calibrated_pending": _note(
            d28 / "090041-calibrated-four.md",
            calibration=_calibration(task_id="bench_090_probe"),
            task_id="bench_090_probe"),
    }
    untouched = dict(calibrated)
    # A note the parser cannot read: rewriting it would replace a document with
    # one nobody wrote, so the script reports it instead.
    broken = d27 / "060606-broken-frontmatter.md"
    broken.write_text("---\nsource: bench-mine\n  calibration: [unclosed\n---\n\nbody\n",
                      encoding="utf-8")
    untouched["broken_frontmatter"] = broken
    # Not a front-matter note at all.
    bare = d27 / "README.md"
    bare.write_text("staged candidates for 2026-09-27\n", encoding="utf-8")
    untouched["readme"] = bare
    # A rejected note lives outside the date dirs in production (`/pending/reject`
    # moves it to `pending-research/_rejected/…`); a `_`-prefixed subtree inside
    # the root is not this queue's surface either.
    rejected = _note(root / "_rejected" / "2026-09-26" / "070707-rejected.md",
                     calibration=_calibration())
    untouched["rejected_subtree"] = rejected
    return {"stale": stale, "calibrated": calibrated, "untouched": untouched}


def _fm(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1]) or {}


def _body(text: str) -> str:
    """Everything after the CLOSING `---` of the first front matter block — the
    same splice `_record_calibration` and `_split_fm` use, applied independently
    here so a byte shifted by the rewrite cannot hide inside the rewritten half."""
    end = text.find("\n---\n", 3)
    assert end >= 0, "fixture note has no closing front matter fence"
    return text[end + 5:]


def _sweep(root: Path) -> list[Path]:
    """The check #1769 states: notes with no `calibration.task_id` whose
    `review_status` still asserts a band verdict.

    The WALK comes from the script — `rs.iter_notes`, the same function the run
    uses, which is also the walk `GET /api/workers/pending` approximates — and
    only the verdict is recomputed here. #1769 recorded what two walkers cost:
    `iter_notes` skips `README.md` and this loop did not, and the README fixture
    had no front matter, so PyYAML rejected it and the disagreement stayed hidden.
    A README with valid front matter claiming a verdict is a note to the old loop
    and not a note to the script, so `--apply` exits 0 while the check that is
    supposed to certify it reports one file. One walker, one denominator;
    `test_a_readme_with_front_matter_is_excluded_by_both_sides_of_the_check` pins
    it. The independent live sweep the owed step re-runs is the same reading over
    the same tree, not a second convention.

    A note whose envelope will not parse is skipped here, exactly as that live
    PyYAML sweep skips it: this check is the owed measurement, and what it cannot
    read is asserted by name in the unreadable-note tests, not silently omitted
    from a total.
    """
    out = []
    for p in rs.iter_notes(root):
        try:
            fm = _fm(p)
        except Exception:
            continue
        if "task_id" not in (fm.get("calibration") or {}):
            if fm.get("review_status") in STILL_ASSERTING:
                out.append(p)
    return out


def _run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *argv],
                          capture_output=True, text=True)


# ---------------------------------------------------------------------------
# clause 1 — the two modes, and what each writes
# ---------------------------------------------------------------------------

def test_the_script_labels_with_the_values_the_contract_names():
    """The four values are the deliverable, so they are pinned at the source the
    report is written from as well as in the rewritten bytes: a run that wrote
    `stale` instead of `stale_envelope`, or quoted the envelope differently, would
    still satisfy every behavioural test above while leaving the queue with a
    label nothing else can filter on.

    `is_stale` is the one key the whole item turns on, and the three shapes that
    key can be absent in — no block, a null block, a block without the key — are
    each asserted, because "a note with no calibration block is not in the stale
    set" is exactly the misreading that would leave notes unlabelled while the
    post-check printed OK.
    """
    assert rs.REVIEW_STATUS == UNCALIBRATED
    assert rs.CALIBRATION_STATUS == STALE_STATUS
    assert rs.MEASURED_AGAINST == MEASURED_AGAINST

    assert rs.is_stale({}) is True
    assert rs.is_stale({"calibration": None}) is True
    assert rs.is_stale({"calibration": {}}) is True
    assert rs.is_stale({"calibration": {"status": "ok", "in_band": True}}) is True
    assert rs.is_stale({"calibration": {"task_id": "bench_001_x"}}) is False


def test_dry_run_changes_no_bytes_in_any_file(tmp_path):
    """`--dry-run` is a report, not a rehearsal of damage: every file keeps its
    bytes AND its mtime, so a later reader can tell it was never opened for
    writing. The dry-run count has to name the same set `--apply` would write,
    or the safe mode is telling a lie about the dangerous one."""
    root = tmp_path / "bench-mine"
    roles = _build(root)
    before = {p: p.read_bytes() for p in sorted(root.rglob("*.md"))}
    stamps = {p: p.stat().st_mtime_ns for p in before}

    r = _run("--root", str(root), "--dry-run")
    assert r.returncode == 0, r.stderr

    for p, raw in before.items():
        assert p.read_bytes() == raw, f"dry-run rewrote {p.name}"
        assert p.stat().st_mtime_ns == stamps[p], f"dry-run opened {p.name} for writing"

    out = r.stdout
    for role, path in roles["stale"].items():
        if role == "already_stale":
            continue
        assert path.name in out, f"dry-run did not name {role} ({path.name})"
    assert "dry-run" in out.lower(), "the safe mode must say so in its own output"


def test_apply_sets_the_four_fields_on_every_note_with_no_task_id(tmp_path):
    """`--apply` rewrites every note whose `calibration` block has no `task_id`,
    recursing into each `{yyyy-mm-dd}` subdirectory, and writes exactly the four
    values the contract names. The relabelled block keeps NO `task_id`: the
    trials still were not a measurement of any task, and inventing an id here
    would be the same lie in a different shape."""
    root = tmp_path / "bench-mine"
    roles = _build(root)
    assert _sweep(root), "the fixture asserts nothing before the run — nothing to fix"

    r = _run("--root", str(root), "--apply")
    assert r.returncode == 0, r.stderr

    for role, path in roles["stale"].items():
        assert path.parent.name in ("2026-09-27", "2026-09-28"), role
        fm = _fm(path)
        cal = fm["calibration"]
        assert fm["review_status"] == UNCALIBRATED, role
        assert cal["status"] == STALE_STATUS, role
        assert cal["in_band"] is None, f"{role}: a voided verdict must not stay true"
        assert cal["measured_against"] == MEASURED_AGAINST, role
        assert "task_id" not in cal, f"{role}: invented an id the trials never had"
        # The evidence stays: the composites and the mean are what a human
        # re-measures against, and deleting them is not what relabelling means.
        if role in ("pending_in_band", "out_of_band", "fenced", "already_stale"):
            assert cal["runs"] == 10, role
            assert len(cal["composites"]) == 10, role
        else:
            # A note that recorded no trials gets the label and nothing else: a
            # relabel that fabricated `runs: 10` would be inventing evidence in
            # the same breath as it disowned the verdict.
            assert "runs" not in cal and "composites" not in cal, role
    for name, p in roles["calibrated"].items():
        # `measured_against` on a note that HAS a task_id is owed ruling 2 of
        # #1769 — a decision about what the writer should stamp going forward,
        # which this script has no business making silently for four notes.
        assert "measured_against" not in _fm(p)["calibration"], (
            f"{name}: stamped a provenance the relabel has no ruling for")


def test_the_two_modes_are_not_both_accepted_at_once(tmp_path):
    """`--dry-run --apply` would be an instruction to both write and not write.
    argparse has to refuse it (exit 2) rather than let the last flag win, because
    a run whose mode is decided by shell history is a run nobody can audit."""
    root = tmp_path / "bench-mine"
    _build(root)
    r = _run("--root", str(root), "--dry-run", "--apply")
    assert r.returncode == 2, f"both modes accepted: {r.stdout}{r.stderr}"


def test_a_missing_root_is_reported_not_counted_as_clean(tmp_path):
    """0 notes scanned and 0 stale notes look identical in the summary, and the
    one thing that distinguishes them is the exit code: a check whose denominator
    can be zero is not a check."""
    r = _run("--root", str(tmp_path / "no-such-staging"), "--dry-run")
    assert r.returncode == 2, r.stdout
    assert "no-such-staging" in r.stdout or "no-such-staging" in r.stderr


# ---------------------------------------------------------------------------
# clause 2 — the sweep, and the notes that are already measurements
# ---------------------------------------------------------------------------

def test_after_apply_the_sweep_finds_no_note_still_asserting_a_verdict(tmp_path):
    """#1769's own check: after `--apply`, a PyYAML sweep of the same root finds
    0 notes with no `calibration.task_id` whose `review_status` is `pending` or
    `out_of_band`."""
    root = tmp_path / "bench-mine"
    _build(root)
    assert len(_sweep(root)) == 5, (
        "the fixture's own starting count moved, so the sweep below is not the "
        "one #1769 states: 5 stale notes expected")

    r = _run("--root", str(root), "--apply")
    assert r.returncode == 0, r.stderr

    left = _sweep(root)
    assert not left, f"still asserting a band verdict: {[p.name for p in left]}"


def test_every_note_that_already_had_a_task_id_is_untouched(tmp_path):
    """The 4 post-#1710 notes carry a mean that IS a measurement of the task in
    `calibration.task_id`. This script's whole claim is that a note with a
    measurement is not stale, so those four keep their bytes AND their
    `review_status` — including the one sitting at `pending`, which is a human's
    queue decision and not this script's to make."""
    root = tmp_path / "bench-mine"
    roles = _build(root)
    before = {name: (p, p.read_bytes(), _fm(p).get("review_status"))
              for name, p in roles["calibrated"].items()}

    r = _run("--root", str(root), "--apply")
    assert r.returncode == 0, r.stderr

    for name, (p, raw, review_status) in before.items():
        assert p.exists(), f"{name} vanished"
        assert p.read_bytes() == raw, f"{name} was rewritten — it has a task_id"
        assert _fm(p).get("review_status") == review_status, name
    assert _fm(roles["calibrated"]["calibrated_pending"])["review_status"] == "pending"
    for name in ("calibrated_out_of_band_1", "calibrated_out_of_band_2",
                 "calibrated_out_of_band_3"):
        assert _fm(roles["calibrated"][name])["review_status"] == "out_of_band", name


def test_unreadable_notes_are_reported_and_left_alone(tmp_path):
    """A note whose front matter does not parse cannot be relabelled safely —
    writing one would put bytes the script invented where a document was. It is
    reported by name instead, and the summary's denominator includes it, so a
    human sees the file the sweep cannot see."""
    root = tmp_path / "bench-mine"
    roles = _build(root)
    before = {name: p.read_bytes() for name, p in roles["untouched"].items()}

    r = _run("--root", str(root), "--apply")
    assert r.returncode == 0, r.stderr

    for name, raw in before.items():
        p = roles["untouched"][name]
        assert p.exists(), f"{name} was deleted"
        assert p.read_bytes() == raw, f"{name} was rewritten but should not have been"
    assert "broken-frontmatter" in r.stdout, (
        "the note the sweep cannot read must be named, not silently skipped")


def test_a_second_run_writes_nothing_new(tmp_path):
    """The relabel is idempotent by bytes, not by a marker: a second `--apply`
    finds the notes already labelled and opens none of them for writing. This is
    what makes the owed re-run after the owed `--apply` safe to repeat."""
    root = tmp_path / "bench-mine"
    _build(root)
    assert _run("--root", str(root), "--apply").returncode == 0
    before = {p: p.read_bytes() for p in sorted(root.rglob("*.md"))}
    stamps = {p: p.stat().st_mtime_ns for p in before}

    assert _run("--root", str(root), "--apply").returncode == 0
    for p, raw in before.items():
        assert p.read_bytes() == raw, f"second run rewrote {p.name}"
        assert p.stat().st_mtime_ns == stamps[p], f"second run opened {p.name}"


# ---------------------------------------------------------------------------
# clause 3 — the candidate task block is the thing a human promotes
# ---------------------------------------------------------------------------

def test_rewritten_notes_keep_their_body_and_their_path(tmp_path):
    """Everything after the closing `---` of the staging front matter — the mined
    candidate's own bench-task block — is byte-identical, and no file moves or
    disappears. The promote route builds the landed bench from that body alone
    (`_bench_task_block(body)`), so a rewrite that reflowed it would ship a
    different task than the one a human read."""
    root = tmp_path / "bench-mine"
    roles = _build(root)
    paths_before = sorted(p for p in root.rglob("*.md"))
    # README.md has no front matter at all, so it has no body to compare; every
    # other file in the tree does, including the one whose YAML is broken.
    bodies = {p: _body(p.read_text(encoding="utf-8"))
              for p in paths_before if p.name != "README.md"}
    stale = [p for name, p in roles["stale"].items() if name != "already_stale"]

    assert _run("--root", str(root), "--apply").returncode == 0

    assert sorted(p for p in root.rglob("*.md")) == paths_before, (
        "a note appeared or disappeared: this script relabels, it does not move")
    for p, body in bodies.items():
        assert _body(p.read_text(encoding="utf-8")) == body, (
            f"{p.name}: the candidate task block was rewritten")
    for p in stale:
        assert p.exists() and p.read_bytes() != b"", p


def test_the_staging_envelope_still_parses_after_the_rewrite(tmp_path):
    """The rewritten half has to still be readable by the two readers that
    matter: the promotion listing's own front-matter parse, and
    `bench_mine._candidate_frontmatter` on the body — the function the on-demand
    re-measurement route is built on. A relabel that broke either would strand
    the note in the queue with no way to re-measure it."""
    root = tmp_path / "bench-mine"
    roles = _build(root)
    assert _run("--root", str(root), "--apply").returncode == 0

    import workers.sources.bench_mine as BM

    for role in ("pending_in_band", "fenced", "no_calibration_key"):
        p = roles["stale"][role]
        fm = _fm(p)
        assert fm.get("source") == "bench-mine", role
        assert fm.get("generated_at"), f"{role}: the envelope lost a key it had"
        task = BM._candidate_frontmatter(_body(p.read_text(encoding="utf-8")))
        assert task and task.get("objective_checks"), (
            f"{role}: the body no longer parses as a bench task")


# ---------------------------------------------------------------------------
# clause 4 — no GPU, and it cannot become a calibration run
# ---------------------------------------------------------------------------

def _imports(source: str) -> list[str]:
    tree = ast.parse(source)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            found.append(node.module or "")
    return found


def test_the_script_imports_nothing_from_scripts_autoresearch():
    """The relabel is a front-matter edit. `scripts.autoresearch` is where the
    engine that spends GPU hours on ten trials lives, and a maintenance script
    that reached for it could turn a label pass into a calibration run — which is
    the thing #1769 excludes. Checked on the AST, so a commented-out or
    string-spelled import cannot pass either."""
    source = SCRIPT.read_text(encoding="utf-8")
    imports = _imports(source)
    # Positive control: an extractor that matched nothing would make the ban
    # below vacuously true (#1689's rule for a `not in` over a scan with no eyes).
    assert any("yaml" in m for m in imports), (
        f"the import extractor found {imports} — not even yaml, so it is blind")
    for mod in imports:
        assert not mod.startswith("scripts"), f"{mod} reaches into the engine"
        assert "autoresearch" not in mod, f"{mod} is the calibration engine"
    assert not any("torch" in m or "vllm" in m for m in imports)


def test_no_call_to_the_calibration_entry_point_under_scripts_maintenance():
    """#1769 states the check as `git grep -n calibrate_candidate --
    scripts/maintenance/` returning no match, and the gate runs the same command
    here. The needle is assembled from fragments so this file cannot be the hit
    if the scope is ever widened — the same trick
    `test_the_band_premise_is_stated_without_a_dated_task_count` uses."""
    needle = "calibrate_" + "candidate"
    out = subprocess.run(["git", "grep", "-n", needle, "--", "scripts/maintenance/"],
                         cwd=str(ROOT), capture_output=True, text=True)
    assert out.returncode in (0, 1), out.stderr
    assert not out.stdout.strip(), (
        f"the calibration entry point is being called under scripts/maintenance/:\n"
        f"{out.stdout}")

    # And the same fact about this script's own source, independent of git:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    names = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert needle not in called | names, "this script calls the calibrator directly"


# ---------------------------------------------------------------------------
# #1873 clause 1 — a SKIP is a report, never a clean bill of health
#
# #1769 landed the relabel but recorded that its own post-check could not catch
# the one note class it exists for: `still_asserting` parsed every note with
# PyYAML inside `try: … except Exception: continue`, so an unreadable note never
# entered `before` or `after`, `after` came back empty, and `--apply` printed
# `post-check OK — 0 notes …` and returned 0. Since that exit 0 is the only
# certificate the owed live run gets, "0 notes assert a verdict" had quietly come
# to mean "0 PARSEABLE notes assert one".
# ---------------------------------------------------------------------------

def _unreadable_note(path: Path, *, review_status: str | None,
                     truncated: bool = False) -> str:
    """Write an unreadable note and return its bytes.

    Two unreadable shapes, one per test below, and the difference between them is
    the whole clause:

      * `review_status="pending"` (default) — the front matter will not parse
        (`  calibration: [unclosed` is bad indentation over an unclosed flow
        sequence), yet the BYTES carry a band verdict with no `task_id` behind it.
        A reader of the file sees `review_status: pending`; a reader that starts
        with `yaml.safe_load` sees nothing at all.
      * `truncated=True` — the file stops mid-envelope with no closing `---`, so
        the envelope splice itself returns None. This is the shape the old
        post-check could not COUNT: `yaml.safe_load(split_front_matter(…)[0])`
        raises `TypeError` on `None[0]`, and `except Exception: continue` ate it.
        Its bytes carry `review_status: out_of_band`.
      * `review_status=None` — an unreadable note claiming no verdict, the exact
        fixture #1769 left behind (`test_unreadable_notes_are_reported_and_left_alone`
        pins that behaviour, and it is the control inside the tests below: the fix
        has to fail on the first note and ONLY the first note).

    Written verbatim rather than through `_note`, because a note whose front
    matter round-trips through PyYAML is the readable case by definition.
    """
    lines = ["---", "source: bench-mine"]
    if review_status is not None:
        lines.append(f"review_status: {review_status}")
    lines += ["generated_at: '2026-09-27T00:05:45.978938+00:00'",
              "rationale: derived from baseline loss on bench_007_skill_invocation",
              "  calibration: [unclosed"]
    if not truncated:
        lines += ["---", "", "The candidate task, which no apply may rewrite.", ""]
    raw = "\n".join(lines) + ("\n" if not truncated else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw, encoding="utf-8")
    return raw


def _failure_lines(out: str) -> list[str]:
    """The run's own failure lines — the ones that name notes it could not clear."""
    return [ln for ln in out.splitlines() if "STILL assert" in ln]


def test_apply_exits_2_when_an_unreadable_note_still_asserts_a_verdict(tmp_path):
    """Clause 1: a note the apply could not relabel, whose bytes still read as a
    band verdict with no task behind them, is named on stdout and the run exits 2.

    Before #1873 this same tree produced `SKIP 040404-unparseable-verdict.md: front
    matter does not parse`, then `relabel: post-check OK — 0 notes …`, then exit 0
    — the certificate covering every note except the one that needed it. Now the
    note is named twice (its SKIP line, and the failure line) and the exit code is
    2. The bytes are still not touched: clause 1 changes what a skip PROVES, not
    what a skip DOES, and inventing front matter over somebody's document is still
    the worse option.

    The unreadable note that claims nothing (`040405-unreadable-quiet.md`) shares
    the tree on purpose. It stays byte-identical and stays out of the failure
    line, which is what separates "the guard reads the note" from "the guard gives
    up on unreadable notes" — a fix that failed any run containing an unreadable
    file would pass the exit-code half of this test and fail this half.
    """
    root = tmp_path / "bench-mine"
    loud = root / "2026-09-27" / "040404-unparseable-verdict.md"
    quiet = root / "2026-09-27" / "040405-unreadable-quiet.md"
    loud_raw = _unreadable_note(loud, review_status="pending")
    quiet_raw = _unreadable_note(quiet, review_status=None)

    dry = _run("--root", str(root))
    assert dry.returncode == 0, (
        "a dry run writes nothing, so it makes no claim to fail on; the guard "
        f"belongs to the mode that wrote\n{dry.stdout}")
    assert loud.name in dry.stdout, "the unreadable note must be named in a dry run"
    assert _failure_lines(dry.stdout), (
        "a dry run must still report the note it cannot clear, or the operator "
        f"reading the dry run sees a clean tree\n{dry.stdout}")

    r = _run("--root", str(root), "--apply")
    assert r.returncode == 2, (
        "--apply left a note asserting a band verdict it cannot name and exited "
        f"{r.returncode}: that exit 0 is the certificate #1769 never had\n{r.stdout}")
    assert loud.name in r.stdout, r.stdout
    failed = _failure_lines(r.stdout)
    assert len(failed) == 1, f"expected one failure line, got {failed}\n{r.stdout}"
    assert loud.name in failed[0] and quiet.name not in failed[0], (
        f"the failure is the note that asserts a verdict, not every unreadable "
        f"note: {failed[0]}")
    assert "POST-CHECK FAILED" in r.stdout, r.stdout
    for p, raw in ((loud, loud_raw), (quiet, quiet_raw)):
        assert p.read_text(encoding="utf-8") == raw, f"{p.name} was rewritten"


def test_apply_exits_2_on_a_truncated_note_the_post_check_cannot_count(tmp_path):
    """The unreadable shape the old post-check could not even enumerate, so its
    exit 0 was unfalsifiable by construction (#1769 blind spot 1, second file).

    A note truncated mid-envelope has no closing `---`. `split_front_matter`
    returns None for it, the post-check's `yaml.safe_load(split(…)[0])` then
    raises `TypeError` on `None[0]`, and `except Exception: continue` swallowed
    that — so at no point in the run was this file in the denominator, `after`
    was empty, and `--apply` returned 0 over a note reading
    `review_status: out_of_band` with no `calibration.task_id` anywhere in it.
    The envelope is now taken whole when there is no closing fence, so the note is
    in the denominator, named, and the run exits 2.
    """
    root = tmp_path / "bench-mine"
    p = root / "2026-09-27" / "041741-truncated-envelope.md"
    raw = _unreadable_note(p, review_status="out_of_band", truncated=True)
    assert "task_id" not in raw and raw.count("---") == 1, (
        "fixture drift: this note has to be truncated and task-less")

    r = _run("--root", str(root), "--apply")
    assert r.returncode == 2, (
        f"a note with no closing fence still asserts a verdict; got "
        f"{r.returncode}\n{r.stdout}")
    assert p.name in r.stdout, r.stdout
    assert _failure_lines(r.stdout), r.stdout
    assert p.read_text(encoding="utf-8") == raw, "the truncated note was rewritten"


# ---------------------------------------------------------------------------
# #1873 clause 3 — the check and the run walk the tree once, not twice
# ---------------------------------------------------------------------------

def test_a_readme_with_front_matter_is_excluded_by_both_sides_of_the_check(tmp_path):
    """One denominator: the sweep is the script's walker, not a second `rglob`.

    #1769 recorded the divergence — `iter_notes` skips `README.md` because
    `GET /api/workers/pending` skips it (`app/routers/workers.py`: the same
    `artifact.name == "README.md"` test over the same tree), while the test-side
    `_sweep` did not, and the fixture README happened to carry no front matter at
    all, so PyYAML rejected it and the two walkers agreed by accident. Give that
    README valid front matter claiming a band verdict and the accident is gone:
    the sweep counts a note the script never opens, so `--apply` would exit 0
    while the very check meant to certify it reports one.

    `_sweep` now walks through `rs.iter_notes`, which is why the two lists below
    are the same list rather than two lists that happen to match. The lower-case
    `readme.md` is the control: the exclusion is the queue's exact name test, not
    a name prefix, so that file IS a note to both walkers and does get relabelled.
    """
    root = tmp_path / "bench-mine"
    stale = _note(root / "2026-09-27" / "000545-pending-in-band.md",
                  calibration=_calibration())
    readme = _note(root / "2026-09-27" / "README.md", calibration=_calibration())
    lower = _note(root / "2026-09-27" / "readme.md", calibration=_calibration())
    readme_raw = readme.read_bytes()

    readme_fm = _fm(readme)
    assert readme_fm["review_status"] in STILL_ASSERTING, (
        "fixture drift: this README has to claim a band verdict")
    assert "task_id" not in (readme_fm.get("calibration") or {}), (
        "fixture drift: and it has to be stale, so a naive walker would sweep it "
        "as a note — #1769's README fixture had no front matter at all, which is "
        "why the two walkers only agreed by accident")

    assert _sweep(root) == [stale, lower], (
        "the sweep and iter_notes disagree on which files are notes: "
        f"{_sweep(root)} vs {rs.iter_notes(root)}")
    assert rs.iter_notes(root) == [stale, lower], (
        f"iter_notes changed its own denominator: {rs.iter_notes(root)}")

    r = _run("--root", str(root), "--apply")
    assert r.returncode == 0, r.stdout
    assert _sweep(root) == [], (
        f"the queue's surface must be empty of band claims: {_sweep(root)}")
    assert readme.read_bytes() == readme_raw, (
        "README.md is not a note: it must not be relabelled")
    assert _fm(lower).get("review_status") == UNCALIBRATED, (
        "readme.md is a note: the exclusion is the exact name README.md, the "
        "queue's own test, not a case-insensitive prefix")
