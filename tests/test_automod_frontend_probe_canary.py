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

def test_the_seed_table_is_twelve_must_detect_seeds_one_control_and_two_blind():
    """Clause 2's arithmetic, and the reason the counts are exact: `m` in the printed
    `detected n/m` is the must-detect count, the bar is `m >= 10` at `>= 90%`, and a
    table that quietly lost a seed would still print a passing-looking fraction.
    """
    must = FC.must_detect_seeds()
    assert len(must) == 12, [s.name for s in must]
    assert len(must) >= 10, "the contract's floor: a rate over fewer is not this number"
    controls = FC.control_seeds()
    assert len(controls) == 1, [s.name for s in controls]
    assert controls[0].name == "healthy_build_that_warns", controls[0].name
    assert len(FC.blind_seeds()) == 2, [s.name for s in FC.blind_seeds()]
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

    blind = _seed("stylesheet_loads_but_matches_no_rule")
    unseen = FC.score_seed(blind, {"ok": True, "checks": []})
    assert unseen["score"] == "blind", unseen
    assert unseen["declared_blind"] is True and unseen["blind_reason"], unseen

    caught = FC.score_seed(blind, {"ok": False,
                                   "checks": [{"check": "console-error",
                                               "problem": "404"}]})
    assert caught["score"] == "detected", (
        "the probe saw it, so the blind declaration was wrong and the seed counts")


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
    n=3, m=4 — six seeds in the table, four in the rate.
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
        FC.score_seed(_seed("stylesheet_loads_but_matches_no_rule"),
                      {"ok": True, "checks": []}),
        FC.score_seed(_seed("healthy_build_that_warns"), {"ok": True, "checks": []}),
    ]
    summary = FC.summarise(records, min_seeds=10, min_rate=0.90)
    assert (summary["detected"], summary["seeds_measured"]) == (3, 4), summary
    assert summary["rate"] == 0.75, summary
    assert summary["blind"] == ["throw_long_after_the_settle_window",
                                "stylesheet_loads_but_matches_no_rule"], summary["blind"]
    assert summary["control"] == ["healthy_build_that_warns"], summary["control"]
    assert summary["blind_as_misses"] == 2, summary
    assert summary["rate_counting_blind_as_misses"] == 0.5, (
        "the pessimistic number charges both blind seeds as misses: 3/6")
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

    Measured as written on 2026-09-30: all 12 must-detect seeds detected, both blind
    seeds invisible to the shipped probe, the control clean. The assertion below is
    the contract's bar (m >= 10 at >= 90%), not that number — a canary that pinned
    12/12 would fail the day one shape legitimately moves, and a canary that only
    pinned the bar would not notice the table emptying.
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
