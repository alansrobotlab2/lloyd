"""The canary's own tests (#1872): clauses 1-5 of the seeded-detection contract.

What is pinned here, and what is deliberately not:

* the seed table is pinned as DATA — its counts, its one control, and the four break
  shapes #1872 names as table entries that really lack the file or global they
  reference, checked off the built directory rather than off a docstring;
* the scoring rule is pinned as a PURE function, so `detected` requires a named
  failing check and the blind/control exclusions can be exercised against verdicts no
  browser had to produce;
* one node runs the shipped CLI end to end in a subprocess with `LLOYD_AUTOMOD_STATE`
  aimed at its own tmp dir, which is what pins the printed rate, the artifact it
  prints, and — the pollution hazard #1872's triage named — that the run created
  nothing inside `STATE_DIR / "frontend_probe"`, the directory the gate's real
  per-round verdicts live in and the one #1601 owed entry 1 counts.

The browser nodes call `_require_browser()` rather than carrying a skip marker, for
the reason `tests/test_automod_frontend_probe.py` records: the honesty prechecks count
a skip marker's spelling in the file's raw text and that marker blocks unconditionally,
while a conditional skip in one helper is the shape the same checker demotes to advice
(#1204) — and it is one occurrence in the file instead of one per node.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import tokenize
from pathlib import Path

import pytest

from scripts.automod import frontend_layout as FL
from scripts.automod import frontend_probe as FP
from scripts.automod import frontend_probe_canary as FC
from scripts.automod import state as S


def _require_browser() -> None:
    """Skip, conditionally, when chromium is not on this box.

    The reason is the claim: a detection rate measured by no browser is not a rate,
    and this file must not report that it produced one.
    """
    if not Path(FP.CHROMIUM).is_file():
        pytest.skip(f"no chromium at {FP.CHROMIUM} — the canary cannot measure "
                    f"anything here, and a green row would be a lie about that")


def _seed(name: str) -> FC.Seed:
    """The table's seed called `name`, or a failure that lists what IS there."""
    for s in FC.SEEDS:
        if s.name == name:
            return s
    raise AssertionError(f"no seed {name!r}; the table holds "
                         f"{', '.join(s.name for s in FC.SEEDS)}")


# ── clause 2: the table, pinned as data ─────────────────────────────────────

def test_the_seed_table_is_sixteen_must_detect_seeds_one_control_and_one_blind():
    """Clause 2's arithmetic, and the reason the counts are exact: `m` in the printed
    `detected n/m` is the must-detect count, the bar is `m >= 10` at `>= 90%`, and a
    table that quietly lost a seed would still print a passing-looking fraction.

    Sixteen, not the twelve #1872 shipped: four layout seeds joined the table on
    #2130, and the count of declared-blind dropped from two to one because
    `stylesheet_loads_but_matches_no_rule` moved channels. That move is the whole
    point of pinning the numbers here — the blind declaration is not a free win, it
    is one fewer exemption and four more seeds that must be caught.
    """
    must = FC.must_detect_seeds()
    assert len(must) == 16, [s.name for s in must]
    assert len(must) >= 10, "the contract's floor: a rate over fewer is not this number"
    controls = FC.control_seeds()
    assert len(controls) == 1, [s.name for s in controls]
    assert controls[0].name == "healthy_build_that_warns", controls[0].name
    assert len(FC.blind_seeds()) == 1, [s.name for s in FC.blind_seeds()]
    assert FC.blind_seeds()[0].name == "throw_long_after_the_settle_window"
    # Twelve over the load channel, four over the layout channel, and no seed is in
    # both: the rate stays one number over one instrument per row.
    assert len([s for s in must if s.channel == "load"]) == 12
    assert len([s for s in must if s.channel == "layout"]) == 4
    assert {s.channel for s in FC.SEEDS} == {"load", "layout"}
    # The four shapes the item names are must-detect entries, not blind escapes.
    for name in FC.REQUIRED_SHAPES:
        seed = _seed(name)
        assert seed.expect == "broken", name
        assert not seed.blind_reason, f"{name} is required and cannot be exempted"
    assert len(set(FC.REQUIRED_SHAPES)) == 4, FC.REQUIRED_SHAPES
    assert set(FC.REQUIRED_SHAPES) <= {s.name for s in FC.SEEDS}
    names = [s.name for s in FC.SEEDS]
    assert len(names) == len(set(names)), f"a seed name appears twice: {names}"


def test_each_required_break_shape_is_a_seed_whose_build_lacks_what_it_references(
        tmp_path):
    """Clause 2's substance: the four shapes are named in the table, and each build
    really is the broken thing it claims to be. Checked off the files on disk, because
    "the seed is a missing module" is a claim about a directory, not about a name.
    """
    for name in FC.REQUIRED_SHAPES:
        seed = _seed(name)
        built = FC.write_build(tmp_path / "builds" / name, seed)
        html = (built / "index.html").read_text(encoding="utf-8")
        js = (built / "assets" / FC.ASSET_JS).read_text(encoding="utf-8")

        if name == "import_of_a_module_the_build_never_emitted":
            missing = "/assets/chunk-late-abc123.js"
            assert missing in js, "the seed must import it at runtime"
            assert not (built / "assets" / "chunk-late-abc123.js").exists(), (
                "the imported module has to be a file the build never emitted")
        elif name == "stylesheet_the_build_never_emitted":
            assert f"/assets/{FC.ASSET_CSS}" in html, "index.html must still ask for it"
            assert not (built / "assets" / FC.ASSET_CSS).exists(), (
                "a stylesheet that exists is not one the build never emitted")
        elif name == "throw_after_first_paint_inside_the_settle_window":
            assert seed.throws_after_ms is not None, "the table must say when it throws"
            assert seed.throws_after_ms < FP.SETTLE_MS, (
                f"a throw at {seed.throws_after_ms} ms is not inside the probe's "
                f"{FP.SETTLE_MS} ms settle window")
            assert "setTimeout" in js and "throw" in js, js
            assert "first paint landed" in js, (
                "it must mount something before it throws, or it is the top-level "
                "throw shape again")
        elif name == "undefined_global_read_at_mount":
            assert "window.LLOYD_RUNTIME." in js, "the read is the shape"
            assert "window.LLOYD_RUNTIME =" not in js, "a read, not a definition"
            for f in sorted((built / "assets").iterdir()):
                assert "LLOYD_RUNTIME =" not in f.read_text(encoding="utf-8"), (
                    f"{f.name} defines the global the seed is supposed to miss")


# ── clause 3: the scoring rule, pinned with no browser in the room ──────────

def test_only_a_verdict_carrying_a_named_failing_check_scores_as_detected():
    """Clause 3, and the two ways a verdict can say "bad" without being a detection:
    no checks at all, or a check that names nothing. Both are `missed` — counting
    either would inflate the number this module exists to report honestly.
    """
    broken = _seed("uncaught_error_at_module_top_level")
    seen = FC.score_seed(broken, {"ok": False,
                                  "checks": [{"check": "pageerror",
                                              "problem": "uncaught: boom"}]})
    assert seen["score"] == "detected", seen
    assert seen["checks"] == ["pageerror"], seen

    silent = FC.score_seed(broken, {"ok": False, "checks": []})
    assert silent["score"] == "missed", silent
    nameless = FC.score_seed(broken, {"ok": False, "checks": [{"check": ""}]})
    assert nameless["score"] == "missed", nameless
    passed = FC.score_seed(broken, {"ok": True, "checks": []})
    assert passed["score"] == "missed", passed

    for record in (seen, silent, nameless, passed):
        assert record["score"] in FC.SCORES, record


def test_the_control_is_scored_control_and_a_declared_blind_seed_loses_its_exemption_when_detected():
    """Clause 3's two exclusions, and the guard that keeps the blind declaration from
    becoming a loophole. A control the probe fails is a FALSE BLOCK, reported as one
    and scored `control` — it never joins a numerator, because a detection rate that
    counts a healthy build as a catch is measuring the wrong thing in both
    directions. A seed declared blind that the probe nevertheless flags is scored
    `detected` and counted: declaring a visible seed blind buys the run nothing.
    """
    control = _seed("healthy_build_that_warns")
    clean = FC.score_seed(control, {"ok": True, "checks": []})
    assert clean["score"] == "control", clean
    assert clean["checks"] == [], clean

    blocked = FC.score_seed(control, {"ok": False,
                                      "checks": [{"check": "console-error",
                                                  "problem": "a warning judged"}]})
    assert blocked["score"] == "control", (
        "the control is never `detected`, even when the probe fails it")
    assert blocked["note"].startswith("the probe FAILED a healthy build"), blocked
    assert blocked["checks"] == ["console-error"], blocked

    blind = _seed("throw_long_after_the_settle_window")
    unseen = FC.score_seed(blind, {"ok": True, "checks": []})
    assert unseen["score"] == "blind", unseen
    assert unseen["declared_blind"] is True and unseen["blind_reason"], unseen

    caught = FC.score_seed(blind, {"ok": False,
                                   "checks": [{"check": "console-error",
                                               "problem": "404"}]})
    assert caught["score"] == "detected", (
        "the probe saw it, so the blind declaration was wrong and the seed counts")

    # The table shipped with TWO exemptions; #2130's layout leg took one away. A
    # declaration is not a permanent classification, it is a claim about the
    # instrument standing in front of it — and `stylesheet_loads_but_matches_no_rule`
    # is the row that moved channels rather than the row that got a better score, so
    # the seed is pinned here as no-longer-declared and the node that measures it
    # lives beside the four new layout seeds below.
    moved = _seed("stylesheet_loads_but_matches_no_rule")
    assert moved.channel == "layout" and not moved.blind_reason, moved
    assert FC.blind_seeds() == [blind], [x.name for x in FC.blind_seeds()]


def test_a_skipped_probe_is_scored_blind_and_never_as_a_miss():
    """A probe that could not run measured nothing, which is not the same fact as a
    probe that ran and missed. `frontend_probe` returns `{"skipped": …}` with no `ok`
    precisely so a caller cannot read it as a verdict; reading it as a miss would
    blame the round's diff for the box's missing browser.
    """
    broken = _seed("boots_and_renders_nothing")
    rec = FC.score_seed(broken, {"skipped": "no chromium at /usr/bin/chromium"})
    assert rec["score"] == "blind", rec
    assert rec["probe_skipped"] is True, rec
    assert "no chromium" in rec["note"], rec


def test_blind_and_control_seeds_leave_both_n_and_m():
    """Clause 3's denominator, computed over synthetic scores so the arithmetic is
    pinned at numbers a browser never had to produce: 3 detected + 1 missed gives
    n=3, m=4 — five seeds here, four in the rate, the fifth exempt.

    Six seeds as shipped, before #2130 moved a second exemption out of the table:
    the pessimistic fraction below divides by `m + 1` because exactly ONE seed is
    left that the instrument cannot see, and that count going down is the point of
    the leg rather than a detail of this arithmetic.
    """
    records = [
        FC.score_seed(_seed("uncaught_error_at_module_top_level"),
                      {"ok": False, "checks": [{"check": "pageerror"}]}),
        FC.score_seed(_seed("boots_and_renders_nothing"),
                      {"ok": False, "checks": [{"check": "app-mounted"}]}),
        FC.score_seed(_seed("stylesheet_the_build_never_emitted"),
                      {"ok": False, "checks": [{"check": "console-error"}]}),
        FC.score_seed(_seed("mounted_into_a_detached_node"), {"ok": True, "checks": []}),
        FC.score_seed(_seed("throw_long_after_the_settle_window"),
                      {"ok": True, "checks": []}),
        FC.score_seed(_seed("healthy_build_that_warns"), {"ok": True, "checks": []}),
    ]
    summary = FC.summarise(records, min_seeds=10, min_rate=0.90)
    assert (summary["detected"], summary["seeds_measured"]) == (3, 4), summary
    assert summary["rate"] == 0.75, summary
    assert summary["blind"] == ["throw_long_after_the_settle_window"], summary["blind"]
    assert summary["control"] == ["healthy_build_that_warns"], summary["control"]
    assert summary["blind_as_misses"] == 1, summary
    assert summary["rate_counting_blind_as_misses"] == 0.6, (
        "the pessimistic number charges the one remaining blind seed as a miss: 3/5")
    assert summary["passes"] is False, "0.75 is under the 0.90 bar"


def test_a_false_block_on_the_control_fails_the_run_even_at_a_perfect_rate():
    """The other half of `passes`: a run that catches every broken seed and still
    refuses a healthy build is a probe that would block every landing, and the canary
    exists to detect exactly that.
    """
    records = [FC.score_seed(_seed("uncaught_error_at_module_top_level"),
                             {"ok": False, "checks": [{"check": "pageerror"}]}),
               FC.score_seed(_seed("healthy_build_that_warns"),
                             {"ok": False, "checks": [{"check": "console-error"}]})]
    summary = FC.summarise(records, min_seeds=1, min_rate=0.90)
    assert (summary["detected"], summary["seeds_measured"]) == (1, 1), summary
    assert summary["control_clean"] is False, summary
    assert summary["control_failures"] == ["healthy_build_that_warns"], summary
    assert summary["passes"] is False, summary


# ── clauses 1, 4 and 5: the shipped CLI, end to end ─────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_shipped_cli_measures_the_table_and_prints_the_artifact_it_wrote(tmp_path):
    """Clauses 1, 3, 4 and 5 together, against the real thing: the module run as the
    acceptance check runs it, in a subprocess, with `LLOYD_AUTOMOD_STATE` aimed at
    this node's own directory so the artifact is pinned where clause 4 says it must
    go without writing a single byte into `~/.local/state/lloyd-automod/`.

    Measured as written on 2026-10-03: all 16 must-detect seeds detected — 12 over
    the load channel and 4 over the layout leg — the one remaining declared-blind
    seed invisible to the load probe, the control clean. The assertion below is the
    contract's bar (m >= 10 at >= 90%), not that number — a canary that pinned 16/16
    would fail the day one shape legitimately moves, and a canary that only pinned
    the bar would not notice the table emptying.
    """
    _require_browser()
    state = tmp_path / "state"
    env = dict(os.environ, LLOYD_AUTOMOD_STATE=str(state))
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.automod.frontend_probe_canary",
         "--seeds", "10"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=900)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"exit {proc.returncode}\n{out}"

    line = re.search(r"^detected (\d+)/(\d+)", out, re.M)
    assert line, f"no `detected <n>/<m>` line in:\n{out}"
    n, m = int(line.group(1)), int(line.group(2))
    assert m >= 10, f"clause 1: only {m} seeds were measured"
    assert m == len(FC.must_detect_seeds()), (
        f"the rate covered {m} of the table's {len(FC.must_detect_seeds())} "
        "must-detect seeds")
    assert n / m >= 0.90, f"clause 1: {n}/{m} is under the 90% bar\n{out}"

    art_line = re.search(r"^artifact (\S+)", out, re.M)
    assert art_line, f"no `artifact <path>` line in:\n{out}"
    artifact = Path(art_line.group(1))
    assert artifact.is_file(), f"clause 4: the printed artifact {artifact} is not there"
    assert artifact.is_relative_to(state), (
        f"clause 4: {artifact} is not under the LLOYD_AUTOMOD_STATE dir {state}")

    report = json.loads(artifact.read_text(encoding="utf-8"))
    assert report["detected"] == n and report["seeds_measured"] == m, report["detected"]
    assert report["rate"] == n / m, report["rate"]
    assert report["thresholds"] == {"min_seeds": 10, "min_rate": 0.9}, report["thresholds"]

    per_seed = {r["name"]: r for r in report["seeds"]}
    assert len(per_seed) == len(FC.SEEDS), sorted(per_seed)
    for r in report["seeds"]:
        assert r["score"] in FC.SCORES, r
        assert isinstance(r["checks"], list), r
    assert sorted(per_seed) == sorted(s.name for s in FC.SEEDS)

    # Clause 3: blind named and excluded, control excluded and never a detection.
    assert sorted(report["blind"]) == sorted(s.name for s in FC.blind_seeds()), report
    blind_line = next((ln for ln in out.splitlines()
                       if ln.startswith("blind, excluded from n and m:")), "")
    assert blind_line, f"the excluded seeds were never named on stdout:\n{out}"
    for name in report["blind"]:
        assert name in blind_line, f"{name} is not on the blind line: {blind_line!r}"
        assert per_seed[name]["score"] == "blind", per_seed[name]
        assert per_seed[name]["checks"] == [], (
            f"{name} is blind only while the probe names nothing: {per_seed[name]}")
    control = per_seed["healthy_build_that_warns"]
    assert control["score"] == "control", control
    assert control["checks"] == [], f"the control must stay clean: {control}"

    # Clause 5: nothing this run wrote may sit in the gate's real-verdict directory,
    # and every build it served came out of the scratch tree, not the checkout.
    assert not (state / "frontend_probe").exists(), (
        "the canary let probe_build's default shots_dir fire, which creates the "
        "directory the gate's real verdicts live in and #1601 owed entry 1 counts")
    tmp_root = Path(tempfile.gettempdir()).resolve()
    for r in report["seeds"]:
        for key in ("build_dir", "shots_dir"):
            p = Path(r[key]).resolve()
            assert p.is_relative_to(tmp_root), f"{key}={p} is not in the scratch tree"
            assert not p.is_relative_to(REPO_ROOT.resolve()), f"{key}={p} is in a checkout"
            assert not p.is_relative_to(state), f"{key}={p} is in the state dir"

    for name in FC.REQUIRED_SHAPES:
        assert per_seed[name]["score"] == "detected", per_seed[name]
        assert per_seed[name]["checks"], f"{name} was caught with no named check"


def test_one_shot_lands_in_the_scratch_shots_dir_and_never_in_the_verdict_dir(tmp_path,
                                                                             monkeypatch):
    """Clause 5's mechanism, witnessed rather than inferred: the screenshot a broken
    build produces has to turn up under the run's scratch `shots/`, and the state
    directory this node owns must still have no `frontend_probe` in it afterwards.
    That directory is #1601 owed entry 1's denominator, so a canary that fabricated
    rows in it would corrupt a ruling that is already waiting on the count.

    One seed, because the point is where the file goes, not how many there are.
    """
    _require_browser()
    state = tmp_path / "state"
    monkeypatch.setattr(S, "STATE_DIR", state)
    scratch = tmp_path / "scratch"
    seed = _seed("uncaught_error_at_module_top_level")

    report = FC.run_canary(root=scratch, seeds=[seed], min_seeds=1)

    assert report["detected"] == 1 and report["seeds_measured"] == 1, report
    record = report["seeds"][0]
    shot = Path(record["screenshot"])
    assert shot.is_file(), f"no screenshot for a caught seed: {record}"
    assert shot.is_relative_to(scratch / "shots" / seed.name), (
        f"the evidence landed outside the scratch dir: {shot}")
    assert Path(record["build_dir"]).is_relative_to(scratch), record["build_dir"]
    assert not (state / "frontend_probe").exists(), (
        "probe_build's default shots_dir fired — see the canary's module docstring")
    written = FC.write_run_artifact(report)
    assert written.is_file() and written.is_relative_to(state / FC.ARTIFACT_DIRNAME), written
    assert (state / FC.ARTIFACT_DIRNAME / "latest.json").is_file(), (
        "the stable pointer the owed-check job is sent to was not written")


# ── a rate the run did not measure must not be printed as one ───────────────

def test_a_run_in_which_the_probe_could_not_run_reports_no_rate(tmp_path, monkeypatch,
                                                                capsys):
    """The failure mode a measurement tool must not have: chromium gone, every seed
    skipped, and a `detected 0/0` line that a grep reads as a result. Exit 2, no
    fraction, no artifact — an unmeasured canary leaves nothing for the owed-check
    job to mistake for the number it is owed.
    """
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(FP, "run", lambda *a, **kw: {"skipped": "no chromium at /x"})
    monkeypatch.setattr(FP, "tmp_build_dir", lambda: tmp_path / "scratch")
    # Both instruments, because the table now holds two of them: stub only the load
    # probe and the four layout seeds run for real against a chromium that this node
    # just declared missing, so the "rate for a run that probed nothing" it is
    # pinning would be a rate for a run that measured four seeds.
    monkeypatch.setattr(FL, "capture_build",
                        lambda *a, **kw: {"skipped": "no chromium at /x"})

    rc = FC.main(["--seeds", "10"])
    out = capsys.readouterr().out

    assert rc == 2, out
    assert not re.search(r"^detected \d+/\d+", out, re.M), (
        f"a rate was printed for a run that probed nothing:\n{out}")
    assert "not measured" in out, out
    assert not (tmp_path / "state" / FC.ARTIFACT_DIRNAME).exists(), (
        "an unmeasured run wrote an artifact anyway")


def test_asking_for_more_seeds_than_the_table_holds_refuses_to_measure(tmp_path,
                                                                       monkeypatch,
                                                                       capsys):
    """`--seeds` is a floor, and a floor above the table is a request the canary has
    to refuse rather than answer with the shorter table it happens to have. Without
    this, `--seeds 99` would print `detected 12/12` and look like it complied.
    """
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(FP, "tmp_build_dir", lambda: tmp_path / "scratch")

    rc = FC.main(["--seeds", str(len(FC.must_detect_seeds()) + 1)])
    out = capsys.readouterr().out

    assert rc == 2, out
    assert "nothing measured" in out, out
    assert not re.search(r"^detected \d+/\d+", out, re.M), out
    assert not (tmp_path / "state" / FC.ARTIFACT_DIRNAME).exists(), out


# ── the rule the probe's own suite already enforces ─────────────────────────

def test_the_canary_never_names_the_dev_server_port():
    """:5173 serves the LIVE tree, so grading it mid-gate grades a tree the diff is
    not in — the whole reason the probe serves a build directory instead. `#1872`'s
    contract repeats the rule for the canary, so the canary is checked the way the
    probe is: no occurrence in executable code, prose still free to explain why.
    """
    src = (REPO_ROOT / "scripts" / "automod" / "frontend_probe_canary.py")
    assert "5173" not in _code_only(src.read_text(encoding="utf-8")), (
        "the dev-server port appears in executable code, not prose")


def _code_only(src: str) -> str:
    """`src` with every string literal and comment token dropped, via `tokenize`.

    The rule is about code, and the file has good reason to discuss the port in
    prose; a line-wise filter gets that wrong on a docstring or an apostrophe, and a
    check whose view of the file is unreliable reports a violation that is not there.
    Same approach as `tests/test_automod_frontend_probe.py::_code_only`.
    """
    out: list[str] = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        out.append(tok.string)
    return "".join(out)


# ── clause 4 of #2130: the layout channel's four seeds ──────────────────────

def _layout_seeds() -> list[FC.Seed]:
    return [seed for seed in FC.SEEDS if seed.channel == "layout"]


def test_the_layout_seeds_are_four_and_each_breaks_css_without_touching_the_dom(
        tmp_path):
    """The table's layout rows, pinned as data before any browser grades them.

    Three claims, none of which needs chromium and all of which a browser could not
    tell you:

    * four seeds, one per mechanism #2130's clause 4 names — a stylesheet whose rule
      matches nothing, a shared-variable shift, a hidden panel, an overflowing panel
      — and none of them is the `control` row, which is not a break;
    * each names the sections its injection is allowed to move, from the seven
      headings the leg's own denominator counts, so a seed cannot be scored against
      a section the fixture does not have;
    * the injection is a STYLESHEET edit and nothing else. `index.html` comes out
      byte-identical to the healthy build while `styles.css` differs: a seed that
      reached for the DOM would be grading the collector's ability to see HTML
      changes, not the thing a CSS ripple is.
    """
    seeds = _layout_seeds()
    assert len(seeds) == 4, [seed.name for seed in seeds]
    assert len({seed.layout_kind for seed in seeds}) == 4, "one mechanism each"
    whole = FC.layout_app(tmp_path / "control", "control")
    for seed in seeds:
        assert seed.layout_kind in FC.LAYOUT_BREAKS, seed.layout_kind
        assert seed.layout_kind != "control", seed.name
        assert seed.expect == "broken", seed.name
        assert not seed.blind_reason, (
            f"{seed.name} cannot be declared blind: the channel exists to carry it")
        spec = FC.LAYOUT_BREAKS[seed.layout_kind]
        assert spec["affected"], f"{seed.name} names no section, so nothing is caught"
        assert set(spec["affected"]) <= set(FC.LAYOUT_HEADS), spec["affected"]
        broken = FC.layout_app(tmp_path / seed.name, seed.layout_kind)
        assert (broken / "index.html").read_bytes() == (whole / "index.html").read_bytes(), (
            f"{seed.layout_kind} changed the markup; a layout seed breaks the sheet")
        assert (broken / "styles.css").read_bytes() != (whole / "styles.css").read_bytes(), (
            f"{seed.layout_kind} changed nothing, so a miss would be about the seed")
    # Both attribution tags have to be reachable in one artifact, or the leg can
    # only ever demonstrate the case it does not exist for.
    untouched = [seed.name for seed in seeds
                 if FC.LAYOUT_BREAKS[seed.layout_kind]["changed_paths"]
                 and not set(FC.LAYOUT_BREAKS[seed.layout_kind]["changed_paths"])
                 & {o for owners in FC.LAYOUT_BREAKS[seed.layout_kind]["owners"].values()
                    for o in owners}]
    assert len(untouched) == 3, untouched
    assert len(seeds) - len(untouched) == 1, "one seed is the in-diff case"


def test_every_layout_seed_yields_a_failing_check_naming_its_section_and_the_healthy_app_yields_none(
        tmp_path):
    """Clause 4 as the browser measures it: four breaks, four named sections, and the
    unchanged build still clean in the same run.

    This is the node the item's acceptance check is written against. The control half
    is not a courtesy: a leg that fired on every section would pass the first
    assertion by accident, and the false-positive floor the owed 10-landing
    measurement is supposed to establish starts here, with zero on a build that
    changed nothing.

    `attribution` is asserted alongside the score because the two together are the
    leg's whole output — `detected` with the wrong tag would still leave a reviewer
    unable to tell a ripple from a change the author made on purpose.
    """
    _require_browser()
    caught = {}
    for seed in _layout_seeds():
        verdict = FC.measure_layout_seed(seed, tmp_path / seed.name)
        scored = FC.score_seed(seed, verdict)
        checks = verdict.get("checks") or []
        assert scored["score"] == "detected", (
            f"{seed.name}: {scored['score']} — {scored.get('note') or verdict}")
        assert checks, f"{seed.name}: detected with no failing check to show"
        affected = FC.LAYOUT_BREAKS[seed.layout_kind]["affected"]
        named = [c for c in checks
                 if any(head in str(c.get("section")) for head in affected)]
        assert named, f"{seed.name} fired on {[c['section'] for c in checks]}, " \
                      f"not on {list(affected)}"
        assert all(c["check"] == f"layout-changed:{c['section']}" for c in checks), checks
        caught[seed.name] = (named[0]["attribution"],
                             {f: sorted(d) for f, d in
                              list(named[0]["delta"].values())[0].items()})
    assert sorted(caught) == sorted(seed.name for seed in _layout_seeds())
    assert sorted(a for a, _ in caught.values()) == ["UNTOUCHED", "UNTOUCHED",
                                                     "UNTOUCHED", "in-diff"], caught

    control = FC.layout_app(tmp_path / "healthy", "control")
    first = FL.capture_build(control, widths=FC.LAYOUT_WIDTHS, chromium=FP.CHROMIUM)
    second = FC.layout_app(tmp_path / "healthy2", "control")
    again = FL.capture_build(second, widths=FC.LAYOUT_WIDTHS, chromium=FP.CHROMIUM)
    assert not first.get("skipped"), first
    assert not again.get("skipped"), again
    fresh = FL.diff(FL.fingerprint(first["views"], round_id="a"),
                    FL.fingerprint(again["views"], round_id="b"))
    assert fresh == [], f"the healthy fixture diffed against itself: {fresh}"


def test_the_load_channel_is_blind_to_the_stylesheet_seed_and_the_layout_channel_is_not(
        tmp_path):
    """The seam this increment is actually about, measured across both instruments.

    `frontend_probe_canary/latest.json` shipped with
    `stylesheet_loads_but_matches_no_rule` scored `blind` and `"checks": []`: no
    channel the probe watches can carry a stylesheet that loads with a 200 while the
    rule the page needed no longer applies. So the same served directory is run
    through both instruments here. The load probe must still see nothing — that is
    the recorded blind spot, and if it started catching this on its own the layout
    leg would be redundant — and the layout leg must produce the named, section-
    carrying check the blind row could not.
    """
    _require_browser()
    seed = _seed("stylesheet_loads_but_matches_no_rule")
    build = FC.layout_app(tmp_path / seed.name / "broken", seed.layout_kind)
    load = FP.run(build, shots_dir=tmp_path / "shots")
    assert not load.get("skipped"), load
    assert load["checks"] == [], (
        f"the load channel caught it, so this node's premise rotted: {load['checks']}")
    assert load["ok"] is True, "the shipped probe calls this healthy build ok"

    verdict = FC.measure_layout_seed(seed, tmp_path / "onward")
    checks = verdict.get("checks") or []
    assert [c["check"] for c in checks] == ["layout-changed:Tokens"], checks
    assert FC.score_seed(seed, verdict)["score"] == "detected", verdict
